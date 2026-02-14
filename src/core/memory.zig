const std = @import("std");
const builtin = @import("builtin");

const mem = std.mem;
const Allocator = mem.Allocator;

const Mutex = std.Thread.Mutex;
const CondVar = std.Thread.Condition;
const Semaphore = std.Thread.Semaphore;

const AtomicBool = std.atomic.Value(bool);
const AtomicU64 = std.atomic.Value(u64);
const AtomicUsize = std.atomic.Value(usize);

pub const MemoryConfig = struct {
    pub const PAGE_SIZE: usize = std.mem.page_size;
    pub const CACHE_LINE_SIZE: usize = 128;
};

pub const PageSize: usize = MemoryConfig.PAGE_SIZE;

var g_empty_u8: [0]u8 = .{};

fn emptyU8Slice() []u8 {
    return g_empty_u8[0..];
}

fn isPow2(x: usize) bool {
    return x != 0 and (x & (x - 1)) == 0;
}

fn alignForwardChecked(addr: usize, alignment: usize) !usize {
    if (alignment == 0) return error.InvalidAlignment;
    if (!isPow2(alignment)) return error.InvalidAlignment;
    return mem.alignForward(usize, addr, alignment);
}

fn addChecked(a: usize, b: usize) !usize {
    const r = @addWithOverflow(a, b);
    if (r[1] != 0) return error.Overflow;
    return r[0];
}

fn mulChecked(a: usize, b: usize) !usize {
    const r = @mulWithOverflow(a, b);
    if (r[1] != 0) return error.Overflow;
    return r[0];
}

pub const Arena = struct {
    buffer: []u8,
    offset: usize,
    allocator: Allocator,
    mutex: Mutex,

    pub fn init(allocator: Allocator, size: usize) !Arena {
        if (size == 0) return error.InvalidSize;
        const aligned_size = mem.alignForward(usize, size, PageSize);
        const buffer = try allocator.alignedAlloc(u8, PageSize, aligned_size);
        @memset(buffer, 0);
        return .{
            .buffer = buffer,
            .offset = 0,
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Arena) void {
        self.secureReset();
        self.allocator.free(self.buffer);
        self.buffer = emptyU8Slice();
        self.offset = 0;
    }

    pub fn alloc(self: *Arena, size: usize, alignment: usize) ?[]u8 {
        if (size == 0) return emptyU8Slice();
        if (alignment == 0 or !isPow2(alignment)) return null;

        self.mutex.lock();
        defer self.mutex.unlock();

        const aligned_offset = mem.alignForward(usize, self.offset, alignment);
        const end = aligned_offset + size;
        if (end < aligned_offset) return null;
        if (end > self.buffer.len) return null;

        const out = self.buffer[aligned_offset..end];
        self.offset = end;
        return out;
    }

    pub fn allocBytes(self: *Arena, size: usize) ?[]u8 {
        return self.alloc(size, @alignOf(usize));
    }

    pub fn reset(self: *Arena) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.offset = 0;
    }

    pub fn secureReset(self: *Arena) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        secureZeroMemory(self.buffer.ptr, self.buffer.len);
        self.offset = 0;
    }

    pub fn allocated(self: *Arena) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.offset;
    }

    pub fn remaining(self: *Arena) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.offset > self.buffer.len) return 0;
        return self.buffer.len - self.offset;
    }
};

pub const ArenaAllocator = struct {
    parent_allocator: Allocator,
    buffers: std.ArrayList([]u8),
    current_buffer: []u8,
    pos: usize,
    buffer_size: usize,
    mutex: Mutex,

    pub fn init(parent_allocator: Allocator, buffer_size: usize) ArenaAllocator {
        return .{
            .parent_allocator = parent_allocator,
            .buffers = std.ArrayList([]u8).init(parent_allocator),
            .current_buffer = emptyU8Slice(),
            .pos = 0,
            .buffer_size = if (buffer_size == 0) 4096 else buffer_size,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.buffers.items) |buf| {
            secureZeroMemory(buf.ptr, buf.len);
            self.parent_allocator.free(buf);
        }
        self.buffers.deinit();
        self.current_buffer = emptyU8Slice();
        self.pos = 0;
    }

    pub fn allocator(self: *ArenaAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = arenaAlloc,
                .resize = arenaResize,
                .free = arenaFree,
            },
        };
    }

    fn arenaAlloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));

        if (len == 0) return emptyU8Slice().ptr;

        const shift: usize = ptr_align;
        if (shift >= @bitSizeOf(usize)) return null;
        const alignment: usize = (@as(usize, 1) << @intCast(shift));
        if (alignment == 0 or !isPow2(alignment)) return null;

        self.mutex.lock();
        defer self.mutex.unlock();

        const current_pos = self.pos;
        const aligned_pos = mem.alignForward(usize, current_pos, alignment);

        const end = aligned_pos + len;
        if (end < aligned_pos) return null;

        if (self.current_buffer.len == 0 or end > self.current_buffer.len) {
            const need = len + alignment;
            const new_size = if (self.buffer_size > need) self.buffer_size else need;
            const new_buf = self.parent_allocator.alignedAlloc(u8, alignment, new_size) catch return null;
            self.buffers.append(new_buf) catch {
                secureZeroMemory(new_buf.ptr, new_buf.len);
                self.parent_allocator.free(new_buf);
                return null;
            };
            self.current_buffer = new_buf;
            self.pos = 0;

            const new_aligned_pos = mem.alignForward(usize, 0, alignment);
            const new_end = new_aligned_pos + len;
            if (new_end < new_aligned_pos or new_end > self.current_buffer.len) return null;

            const p = self.current_buffer.ptr + new_aligned_pos;
            self.pos = new_end;
            return p;
        }

        const p = self.current_buffer.ptr + aligned_pos;
        self.pos = end;
        return p;
    }

    fn arenaResize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = buf_align;
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_buffer.len == 0) return false;

        const buf_end = @intFromPtr(buf.ptr) + buf.len;
        const cur_end = @intFromPtr(self.current_buffer.ptr) + self.pos;
        if (buf_end != cur_end) return false;

        if (new_len == buf.len) return true;

        if (new_len > buf.len) {
            const additional = new_len - buf.len;
            const new_pos = self.pos + additional;
            if (new_pos < self.pos) return false;
            if (new_pos > self.current_buffer.len) return false;
            self.pos = new_pos;
            return true;
        } else {
            const shrink = buf.len - new_len;
            if (shrink > self.pos) return false;
            self.pos -= shrink;
            return true;
        }
    }

    fn arenaFree(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf_align;
        _ = ret_addr;
        if (buf.len == 0) return;
        secureZeroMemory(buf.ptr, buf.len);
    }
};

pub const SlabAllocator = struct {
    slabs: []Slab,
    next_id: usize,
    allocator: Allocator,
    slab_size: usize,
    block_size: usize,
    mutex: Mutex,

    const Slab = struct {
        data: []u8,
        bitmap: []u64,
        num_blocks: usize,

        fn isBlockFree(self: *const Slab, block_idx: usize) bool {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            if (word_idx >= self.bitmap.len) return false;
            return (self.bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0;
        }

        fn setBlockUsed(self: *Slab, block_idx: usize) void {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            if (word_idx >= self.bitmap.len) return;
            self.bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
        }

        fn setBlockFree(self: *Slab, block_idx: usize) void {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            if (word_idx >= self.bitmap.len) return;
            self.bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
        }
    };

    pub fn init(allocator: Allocator, slab_size: usize, num_slabs: usize, block_size: usize) !SlabAllocator {
        if (slab_size == 0) return error.InvalidSize;
        if (num_slabs == 0) return error.InvalidSlabCount;
        if (block_size == 0) return error.InvalidBlockSize;
        if (!isPow2(block_size)) return error.InvalidBlockSize;
        if (slab_size < block_size) return error.InvalidSize;

        const slabs = try allocator.alloc(Slab, num_slabs);
        errdefer allocator.free(slabs);

        const num_blocks = slab_size / block_size;
        if (num_blocks == 0) return error.InvalidSize;

        const bitmap_words = (num_blocks + 63) / 64;

        var i: usize = 0;
        while (i < slabs.len) : (i += 1) {
            slabs[i].data = try allocator.alignedAlloc(u8, block_size, slab_size);
            errdefer allocator.free(slabs[i].data);
            slabs[i].bitmap = try allocator.alloc(u64, bitmap_words);
            errdefer allocator.free(slabs[i].bitmap);
            @memset(slabs[i].data, 0);
            @memset(slabs[i].bitmap, 0);
            slabs[i].num_blocks = num_blocks;
        }

        return .{
            .slabs = slabs,
            .next_id = 0,
            .allocator = allocator,
            .slab_size = slab_size,
            .block_size = block_size,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *SlabAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.slabs) |slab| {
            secureZeroMemory(slab.data.ptr, slab.data.len);
            self.allocator.free(slab.bitmap);
            self.allocator.free(slab.data);
        }
        self.allocator.free(self.slabs);
        self.slabs = self.slabs[0..0];
        self.next_id = 0;
    }

    pub fn alloc(self: *SlabAllocator, size: usize) ?[]u8 {
        if (size == 0) return emptyU8Slice();
        if (size > self.slab_size) return null;

        self.mutex.lock();
        defer self.mutex.unlock();

        const blocks_needed = (size + self.block_size - 1) / self.block_size;

        const start_id = self.next_id;
        var search_count: usize = 0;

        while (search_count < self.slabs.len) : (search_count += 1) {
            const slab_idx = (start_id + search_count) % self.slabs.len;
            var slab = &self.slabs[slab_idx];

            var consecutive: usize = 0;
            var start_idx: usize = 0;

            var i: usize = 0;
            while (i < slab.num_blocks) : (i += 1) {
                if (slab.isBlockFree(i)) {
                    if (consecutive == 0) start_idx = i;
                    consecutive += 1;
                    if (consecutive >= blocks_needed) {
                        var j: usize = start_idx;
                        const endj = start_idx + blocks_needed;
                        while (j < endj) : (j += 1) {
                            slab.setBlockUsed(j);
                        }
                        const offset = start_idx * self.block_size;
                        self.next_id = (slab_idx + 1) % self.slabs.len;
                        return slab.data[offset .. offset + size];
                    }
                } else {
                    consecutive = 0;
                }
            }
        }

        self.next_id = 0;
        return null;
    }

    pub fn free(self: *SlabAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const ptr_addr = @intFromPtr(ptr.ptr);

        for (self.slabs) |*slab| {
            const slab_start = @intFromPtr(slab.data.ptr);
            const slab_end = slab_start + slab.data.len;

            if (ptr_addr >= slab_start and ptr_addr < slab_end) {
                const offset = ptr_addr - slab_start;
                if (offset % self.block_size != 0) return error.InvalidPointer;
                if (offset + ptr.len > slab.data.len) return error.InvalidPointer;

                const start_block = offset / self.block_size;
                const blocks_used = (ptr.len + self.block_size - 1) / self.block_size;
                if (start_block + blocks_used > slab.num_blocks) return error.InvalidPointer;

                var i: usize = start_block;
                const end_block = start_block + blocks_used;
                while (i < end_block) : (i += 1) {
                    if (slab.isBlockFree(i)) return error.DoubleFree;
                    slab.setBlockFree(i);
                }

                secureZeroMemory(@ptrCast(ptr.ptr), blocks_used * self.block_size);
                return;
            }
        }

        return error.InvalidPointer;
    }
};

pub const PoolAllocator = struct {
    pools: []Pool,
    allocator: Allocator,
    mutex: Mutex,

    const Pool = struct {
        buffer: []u8,
        block_size: usize,
        num_blocks: usize,
        free_list_head: ?usize,
        used: usize,

        fn initFreeList(self: *Pool) void {
            if (self.num_blocks == 0) {
                self.free_list_head = null;
                return;
            }
            var i: usize = 0;
            while (i < self.num_blocks) : (i += 1) {
                const base = self.buffer.ptr + i * self.block_size;
                const p: *?usize = @ptrCast(@alignCast(base));
                if (i + 1 < self.num_blocks) {
                    p.* = i + 1;
                } else {
                    p.* = null;
                }
            }
            self.free_list_head = 0;
        }

        fn owns(self: *const Pool, ptr: [*]u8) bool {
            const start = @intFromPtr(self.buffer.ptr);
            const end = start + self.buffer.len;
            const a = @intFromPtr(ptr);
            return a >= start and a < end;
        }

        fn blockIndex(self: *const Pool, ptr: [*]u8) !usize {
            const start = @intFromPtr(self.buffer.ptr);
            const a = @intFromPtr(ptr);
            const offset = a - start;
            if (offset % self.block_size != 0) return error.InvalidPointer;
            const idx = offset / self.block_size;
            if (idx >= self.num_blocks) return error.InvalidPointer;
            return idx;
        }
    };

    pub fn init(allocator: Allocator, block_size: usize, num_blocks: usize, num_pools: usize) !PoolAllocator {
        if (block_size == 0) return error.InvalidBlockSize;
        if (num_blocks == 0) return error.InvalidBlockCount;
        if (num_pools == 0) return error.InvalidPoolCount;

        const actual_block_size = mem.alignForward(usize, @max(block_size, @sizeOf(?usize)), @alignOf(?usize));
        const pools = try allocator.alloc(Pool, num_pools);
        errdefer allocator.free(pools);

        for (pools) |*pool| {
            const total = try mulChecked(actual_block_size, num_blocks);
            pool.buffer = try allocator.alignedAlloc(u8, @alignOf(?usize), total);
            errdefer allocator.free(pool.buffer);
            @memset(pool.buffer, 0);
            pool.block_size = actual_block_size;
            pool.num_blocks = num_blocks;
            pool.free_list_head = null;
            pool.used = 0;
            pool.initFreeList();
        }

        return .{
            .pools = pools,
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PoolAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.pools) |pool| {
            secureZeroMemory(pool.buffer.ptr, pool.buffer.len);
            self.allocator.free(pool.buffer);
        }
        self.allocator.free(self.pools);
        self.pools = self.pools[0..0];
    }

    pub fn alloc(self: *PoolAllocator, size: usize) ?[]u8 {
        if (size == 0) return emptyU8Slice();

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.pools) |*pool| {
            if (size > pool.block_size) continue;
            const head_idx = pool.free_list_head orelse continue;

            const base = pool.buffer.ptr + head_idx * pool.block_size;
            const next_ptr: *?usize = @ptrCast(@alignCast(base));
            pool.free_list_head = next_ptr.*;
            pool.used += 1;

            const full = pool.buffer[head_idx * pool.block_size .. (head_idx + 1) * pool.block_size];
            return full[0..size];
        }

        return null;
    }

    pub fn free(self: *PoolAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.pools) |*pool| {
            if (!pool.owns(@ptrCast(ptr.ptr))) continue;

            const idx = try pool.blockIndex(@ptrCast(ptr.ptr));
            const base = pool.buffer.ptr + idx * pool.block_size;
            const p: *?usize = @ptrCast(@alignCast(base));
            p.* = pool.free_list_head;
            pool.free_list_head = idx;

            if (pool.used == 0) return error.DoubleFree;
            pool.used -= 1;

            secureZeroMemory(base, pool.block_size);
            return;
        }

        return error.InvalidPointer;
    }
};

pub const BuddyAllocator = struct {
    allocator: Allocator,
    memory: []u8,
    tree: []u8,
    max_order: u32,
    min_order: u32,
    size_map: std.AutoHashMap(usize, u32),
    mutex: Mutex,

    const State = enum(u8) { free = 0, split = 1, full = 2 };

    pub fn init(allocator: Allocator, size: usize, min_order: u32) !BuddyAllocator {
        if (size == 0) return error.InvalidSize;
        const min_block = @as(usize, 1) << @intCast(min_order);
        if (min_block == 0) return error.InvalidSize;
        if (size < min_block) return error.SizeTooSmall;

        const capacity = std.math.ceilPowerOfTwo(usize, size) catch return error.InvalidSize;
        const max_order: u32 = @intCast(std.math.log2_int(usize, capacity));
        if (max_order < min_order) return error.SizeTooSmall;

        const tree_nodes = (@as(usize, 1) << @intCast(max_order + 1)) - 1;
        const tree = try allocator.alloc(u8, tree_nodes);
        errdefer allocator.free(tree);
        @memset(tree, @intFromEnum(State.free));

        const memory = try allocator.alignedAlloc(u8, PageSize, capacity);
        errdefer allocator.free(memory);
        @memset(memory, 0);

        var size_map = std.AutoHashMap(usize, u32).init(allocator);
        errdefer size_map.deinit();

        return .{
            .allocator = allocator,
            .memory = memory,
            .tree = tree,
            .max_order = max_order,
            .min_order = min_order,
            .size_map = size_map,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *BuddyAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.size_map.deinit();
        secureZeroMemory(self.memory.ptr, self.memory.len);
        self.allocator.free(self.memory);
        self.allocator.free(self.tree);
        self.memory = emptyU8Slice();
        self.tree = self.tree[0..0];
    }

    fn nodeState(self: *const BuddyAllocator, idx: usize) State {
        return @enumFromInt(self.tree[idx]);
    }

    fn setNodeState(self: *BuddyAllocator, idx: usize, st: State) void {
        self.tree[idx] = @intFromEnum(st);
    }

    fn leftChild(idx: usize) usize {
        return 2 * idx + 1;
    }

    fn rightChild(idx: usize) usize {
        return 2 * idx + 2;
    }

    fn parent(idx: usize) usize {
        return (idx - 1) / 2;
    }

    fn levelStart(level: u32) usize {
        return (@as(usize, 1) << @intCast(level)) - 1;
    }

    fn allocRec(self: *BuddyAllocator, idx: usize, cur_order: u32, want_order: u32) ?usize {
        const st = self.nodeState(idx);
        if (st == .full) return null;

        if (cur_order == want_order) {
            if (st != .free) return null;
            self.setNodeState(idx, .full);
            return idx;
        }

        if (st == .free) {
            self.setNodeState(idx, .split);
            const l = leftChild(idx);
            const r = rightChild(idx);
            self.setNodeState(l, .free);
            self.setNodeState(r, .free);
        }

        const l = leftChild(idx);
        const r = rightChild(idx);

        if (self.allocRec(l, cur_order - 1, want_order)) |found| {
            self.updateUp(idx);
            return found;
        }
        if (self.allocRec(r, cur_order - 1, want_order)) |found| {
            self.updateUp(idx);
            return found;
        }

        self.updateUp(idx);
        return null;
    }

    fn updateUp(self: *BuddyAllocator, idx: usize) void {
        const l = leftChild(idx);
        const r = rightChild(idx);
        const ls = self.nodeState(l);
        const rs = self.nodeState(r);

        if (ls == .full and rs == .full) {
            self.setNodeState(idx, .full);
        } else if (ls == .free and rs == .free) {
            self.setNodeState(idx, .free);
        } else {
            self.setNodeState(idx, .split);
        }
    }

    fn ptrFromIndex(self: *BuddyAllocator, idx: usize, order: u32) []u8 {
        const level = self.max_order - order;
        const start = levelStart(level);
        const offset_in_level = idx - start;
        const block_size = @as(usize, 1) << @intCast(order);
        const byte_offset = offset_in_level * block_size;
        return self.memory[byte_offset .. byte_offset + block_size];
    }

    pub fn alloc(self: *BuddyAllocator, size: usize) ![]u8 {
        if (size == 0) return error.InvalidSize;

        self.mutex.lock();
        defer self.mutex.unlock();

        var want_order: u32 = self.min_order;
        while ((@as(usize, 1) << @intCast(want_order)) < size) {
            want_order += 1;
            if (want_order > self.max_order) return error.OutOfMemory;
        }

        const found = self.allocRec(0, self.max_order, want_order) orelse return error.OutOfMemory;
        const block = self.ptrFromIndex(found, want_order);
        const slice = block[0..size];
        try self.size_map.put(@intFromPtr(slice.ptr), want_order);
        return slice;
    }

    fn buddyIndex(idx: usize) ?usize {
        if (idx == 0) return null;
        if ((idx & 1) == 1) return idx + 1;
        return idx - 1;
    }

    pub fn free(self: *BuddyAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const ptr_addr = @intFromPtr(ptr.ptr);
        const base = @intFromPtr(self.memory.ptr);
        if (ptr_addr < base or ptr_addr >= base + self.memory.len) return error.InvalidPointer;

        const order = self.size_map.get(ptr_addr) orelse return error.InvalidPointer;
        const block_size = @as(usize, 1) << @intCast(order);

        const offset = ptr_addr - base;
        if (offset % block_size != 0) return error.InvalidPointer;

        const level = self.max_order - order;
        const start = levelStart(level);
        const offset_in_level = offset / block_size;
        const idx = start + offset_in_level;

        if (idx >= self.tree.len) return error.InvalidPointer;
        if (self.nodeState(idx) != .full) return error.DoubleFree;

        const full_block = self.ptrFromIndex(idx, order);
        secureZeroMemory(full_block.ptr, full_block.len);
        self.setNodeState(idx, .free);

        var cur = idx;
        while (cur != 0) {
            const p = parent(cur);
            const l = leftChild(p);
            const r = rightChild(p);
            const ls = self.nodeState(l);
            const rs = self.nodeState(r);

            if (ls == .free and rs == .free) {
                self.setNodeState(p, .free);
            } else if (ls == .full and rs == .full) {
                self.setNodeState(p, .full);
            } else {
                self.setNodeState(p, .split);
            }

            cur = p;
        }

        _ = self.size_map.remove(ptr_addr);
    }
};

pub const MutexQueue = struct {
    head: usize,
    tail: usize,
    buffer: []?*anyopaque,
    allocator: Allocator,
    capacity: usize,
    mutex: Mutex,

    pub fn init(allocator: Allocator, capacity: usize) !MutexQueue {
        if (capacity < 2) return error.InvalidSize;

        const buf = try allocator.alloc(?*anyopaque, capacity);
        errdefer allocator.free(buf);

        var i: usize = 0;
        while (i < capacity) : (i += 1) {
            buf[i] = null;
        }

        return .{
            .head = 0,
            .tail = 0,
            .buffer = buf,
            .allocator = allocator,
            .capacity = capacity,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *MutexQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.allocator.free(self.buffer);
        self.buffer = self.buffer[0..0];
        self.capacity = 0;
    }

    pub fn enqueue(self: *MutexQueue, item: *anyopaque) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const next_tail = (self.tail + 1) % self.capacity;
        if (next_tail == self.head) return false;

        self.buffer[self.tail] = item;
        self.tail = next_tail;
        return true;
    }

    pub fn dequeue(self: *MutexQueue) ?*anyopaque {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.head == self.tail) return null;

        const item = self.buffer[self.head];
        self.buffer[self.head] = null;
        self.head = (self.head + 1) % self.capacity;
        return item;
    }
};

pub const LockFreeQueue = MutexQueue;

pub const MutexStack = struct {
    mutex: Mutex,
    top: ?*Node,
    allocator: Allocator,

    const Node = struct {
        value: *anyopaque,
        next: ?*Node,
    };

    pub fn init(allocator: Allocator) MutexStack {
        return .{
            .mutex = .{},
            .top = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MutexStack) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var cur = self.top;
        while (cur) |n| {
            const next = n.next;
            self.allocator.destroy(n);
            cur = next;
        }
        self.top = null;
    }

    pub fn push(self: *MutexStack, value: *anyopaque) !void {
        const node = try self.allocator.create(Node);
        node.* = .{ .value = value, .next = null };

        self.mutex.lock();
        defer self.mutex.unlock();

        node.next = self.top;
        self.top = node;
    }

    pub fn pop(self: *MutexStack) ?*anyopaque {
        self.mutex.lock();
        defer self.mutex.unlock();

        const node = self.top orelse return null;
        self.top = node.next;
        const v = node.value;
        self.allocator.destroy(node);
        return v;
    }
};

pub const LockFreeStack = MutexStack;

pub const PageAllocator = struct {
    pages: []u8,
    allocator: Allocator,
    page_size: usize,
    bitmap: []u64,
    mutex: Mutex,

    pub fn init(allocator: Allocator, num_pages: usize) !PageAllocator {
        if (num_pages == 0) return error.InvalidSize;

        const total = try mulChecked(num_pages, PageSize);
        const pages = try allocator.alignedAlloc(u8, PageSize, total);
        errdefer allocator.free(pages);
        @memset(pages, 0);

        const bitmap_words = (num_pages + 63) / 64;
        const bitmap = try allocator.alloc(u64, bitmap_words);
        errdefer allocator.free(bitmap);
        @memset(bitmap, 0);

        return .{
            .pages = pages,
            .allocator = allocator,
            .page_size = PageSize,
            .bitmap = bitmap,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PageAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        secureZeroMemory(self.pages.ptr, self.pages.len);
        self.allocator.free(self.bitmap);
        self.allocator.free(self.pages);
        self.pages = emptyU8Slice();
        self.bitmap = self.bitmap[0..0];
    }

    fn isPageFree(self: *const PageAllocator, page_idx: usize) bool {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        if (word_idx >= self.bitmap.len) return false;
        return (self.bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0;
    }

    fn setPageUsed(self: *PageAllocator, page_idx: usize) void {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        self.bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
    }

    fn setPageFree(self: *PageAllocator, page_idx: usize) void {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        self.bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
    }

    pub fn allocPages(self: *PageAllocator, num_pages: usize) ?[]u8 {
        if (num_pages == 0) return emptyU8Slice();

        self.mutex.lock();
        defer self.mutex.unlock();

        const total_pages = self.pages.len / self.page_size;
        if (num_pages > total_pages) return null;

        var consecutive: usize = 0;
        var start_page: usize = 0;

        var i: usize = 0;
        while (i < total_pages) : (i += 1) {
            if (self.isPageFree(i)) {
                if (consecutive == 0) start_page = i;
                consecutive += 1;
                if (consecutive >= num_pages) {
                    var j: usize = start_page;
                    const endj = start_page + num_pages;
                    while (j < endj) : (j += 1) {
                        self.setPageUsed(j);
                    }
                    const offset = start_page * self.page_size;
                    const size = num_pages * self.page_size;
                    return self.pages[offset .. offset + size];
                }
            } else {
                consecutive = 0;
            }
        }

        return null;
    }

    pub fn freePages(self: *PageAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;
        if (ptr.len % self.page_size != 0) return error.InvalidPointer;

        self.mutex.lock();
        defer self.mutex.unlock();

        const pages_start = @intFromPtr(self.pages.ptr);
        const pages_end = pages_start + self.pages.len;
        const ptr_addr = @intFromPtr(ptr.ptr);

        if (ptr_addr < pages_start or ptr_addr >= pages_end) return error.InvalidPointer;
        const offset = ptr_addr - pages_start;
        if (offset % self.page_size != 0) return error.InvalidPointer;

        const start_page = offset / self.page_size;
        const num_pages = ptr.len / self.page_size;

        const total_pages = self.pages.len / self.page_size;
        if (start_page + num_pages > total_pages) return error.InvalidPointer;

        var i: usize = start_page;
        const endi = start_page + num_pages;
        while (i < endi) : (i += 1) {
            if (self.isPageFree(i)) return error.DoubleFree;
            self.setPageFree(i);
        }

        secureZeroMemory(@ptrCast(ptr.ptr), ptr.len);
    }

    pub fn mapPage(self: *PageAllocator, page_idx: usize) ?[*]align(PageSize) u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const total_pages = self.pages.len / self.page_size;
        if (page_idx >= total_pages) return null;

        const offset = page_idx * self.page_size;
        return @ptrCast(@alignCast(self.pages.ptr + offset));
    }
};

pub const ZeroCopySlice = struct {
    ptr: [*]const u8,
    len: usize,

    pub fn init(ptr: [*]const u8, len: usize) ZeroCopySlice {
        return .{ .ptr = ptr, .len = len };
    }

    pub fn slice(self: *const ZeroCopySlice, start: usize, end: usize) ZeroCopySlice {
        if (start > end) return .{ .ptr = self.ptr, .len = 0 };
        if (end > self.len) return .{ .ptr = self.ptr, .len = 0 };
        return .{ .ptr = self.ptr + start, .len = end - start };
    }

    pub fn copyTo(self: *const ZeroCopySlice, allocator: Allocator) ![]u8 {
        const buf = try allocator.alloc(u8, self.len);
        @memcpy(buf, self.asBytes());
        return buf;
    }

    pub fn asBytes(self: *const ZeroCopySlice) []const u8 {
        return self.ptr[0..self.len];
    }
};

pub const ResizeBuffer = struct {
    buffer: []u8,
    len: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ResizeBuffer {
        return .{ .buffer = emptyU8Slice(), .len = 0, .allocator = allocator };
    }

    pub fn deinit(self: *ResizeBuffer) void {
        if (self.buffer.len != 0) {
            secureZeroMemory(self.buffer.ptr, self.buffer.len);
            self.allocator.free(self.buffer);
        }
        self.buffer = emptyU8Slice();
        self.len = 0;
    }

    pub fn append(self: *ResizeBuffer, data: []const u8) !void {
        const new_len = try addChecked(self.len, data.len);
        if (new_len > self.buffer.len) {
            const new_cap = std.math.ceilPowerOfTwo(usize, @max(new_len, 16)) catch new_len;
            if (self.buffer.len == 0) {
                self.buffer = try self.allocator.alloc(u8, new_cap);
                @memset(self.buffer, 0);
            } else {
                self.buffer = try self.allocator.realloc(self.buffer, new_cap);
            }
        }
        @memcpy(self.buffer[self.len..new_len], data);
        self.len = new_len;
    }

    pub fn clear(self: *ResizeBuffer) void {
        self.len = 0;
    }

    pub fn toOwnedSlice(self: *ResizeBuffer) ![]u8 {
        const out = if (self.buffer.len == 0) emptyU8Slice() else try self.allocator.realloc(self.buffer, self.len);
        self.buffer = emptyU8Slice();
        self.len = 0;
        return out;
    }
};

pub fn zeroCopyTransfer(src: []const u8, dest: []u8) void {
    const n = @min(src.len, dest.len);
    if (n == 0) return;
    @memcpy(dest[0..n], src[0..n]);
}

pub fn alignedAlloc(allocator: Allocator, comptime T: type, n: usize, alignment: usize) ![]T {
    if (alignment == 0 or !isPow2(alignment)) return error.InvalidAlignment;
    if (n == 0) return @as([]T, @ptrCast(emptyU8Slice()));
    const bytes = try allocator.alignedAlloc(u8, alignment, try mulChecked(@sizeOf(T), n));
    return @as([*]T, @ptrCast(@alignCast(bytes.ptr)))[0..n];
}

pub fn cacheAlignedAlloc(allocator: Allocator, size: usize, cache_line_size: usize) ![]u8 {
    const alignment = if (cache_line_size != 0 and isPow2(cache_line_size)) cache_line_size else MemoryConfig.CACHE_LINE_SIZE;
    if (size == 0) return emptyU8Slice();
    return try allocator.alignedAlloc(u8, alignment, size);
}

pub fn sliceMemory(base: [*]u8, offset: usize, size: usize, buffer_size: usize) ![]u8 {
    if (offset > buffer_size) return error.OffsetOutOfBounds;
    const end = @addWithOverflow(offset, size);
    if (end[1] != 0) return error.SliceOverflow;
    if (end[0] > buffer_size) return error.SliceOutOfBounds;
    return base[offset .. offset + size];
}

pub fn zeroInitMemory(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    @memset(ptr[0..size], 0);
}

pub fn secureZeroMemory(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    const p: [*]volatile u8 = ptr;
    var i: usize = 0;
    while (i < size) : (i += 1) {
        p[i] = 0;
    }
    std.atomic.fence(.SeqCst);
}

pub fn constantTimeCompare(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

pub fn compareMemory(a: []const u8, b: []const u8) bool {
    return constantTimeCompare(a, b);
}

pub fn hashMemory(data: []const u8) u64 {
    var hasher = std.hash.Wyhash.init(0x9E3779B97F4A7C15);
    hasher.update(data);
    return hasher.final();
}

pub fn alignForward(addr: usize, alignment: usize) !usize {
    return alignForwardChecked(addr, alignment);
}

pub fn alignBackward(addr: usize, alignment: usize) !usize {
    if (alignment == 0 or !isPow2(alignment)) return error.InvalidAlignment;
    return mem.alignBackward(usize, addr, alignment);
}

pub fn isAligned(addr: usize, alignment: usize) bool {
    return mem.isAligned(addr, alignment);
}

pub fn pageAlignedSize(size: usize) usize {
    return mem.alignForward(usize, size, PageSize);
}

pub fn memoryBarrier() void {
    std.atomic.fence(.SeqCst);
}

pub fn readMemoryFence() void {
    std.atomic.fence(.Acquire);
}

pub fn writeMemoryFence() void {
    std.atomic.fence(.Release);
}

pub fn compareExchangeMemory(ptr: *u64, expected: u64, desired: u64) bool {
    return std.atomic.compareExchangeStrong(u64, ptr, expected, desired, .SeqCst, .SeqCst) == null;
}

pub fn atomicLoad(ptr: *u64) u64 {
    return std.atomic.load(u64, ptr, .SeqCst);
}

pub fn atomicStore(ptr: *u64, value: u64) void {
    std.atomic.store(u64, ptr, value, .SeqCst);
}

pub fn atomicAdd(ptr: *u64, delta: u64) u64 {
    return std.atomic.fetchAdd(u64, ptr, delta, .SeqCst);
}

pub fn atomicSub(ptr: *u64, delta: u64) u64 {
    return std.atomic.fetchSub(u64, ptr, delta, .SeqCst);
}

pub fn atomicAnd(ptr: *u64, mask: u64) u64 {
    return std.atomic.fetchAnd(u64, ptr, mask, .SeqCst);
}

pub fn atomicOr(ptr: *u64, mask: u64) u64 {
    return std.atomic.fetchOr(u64, ptr, mask, .SeqCst);
}

pub fn atomicXor(ptr: *u64, mask: u64) u64 {
    return std.atomic.fetchXor(u64, ptr, mask, .SeqCst);
}

pub fn atomicInc(ptr: *u64) u64 {
    return atomicAdd(ptr, 1);
}

pub fn atomicDec(ptr: *u64) u64 {
    return atomicSub(ptr, 1);
}

pub fn memoryEfficientCopy(src: []const u8, dest: []u8) !void {
    if (dest.len < src.len) return error.DestinationTooSmall;
    var i: usize = 0;
    while (i < src.len) : (i += MemoryConfig.CACHE_LINE_SIZE) {
        const end = @min(i + MemoryConfig.CACHE_LINE_SIZE, src.len);
        @memcpy(dest[i..end], src[i..end]);
    }
}

pub fn secureErase(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    const p: [*]volatile u8 = ptr;
    var i: usize = 0;
    while (i < size) : (i += 1) p[i] = 0x55;
    i = 0;
    while (i < size) : (i += 1) p[i] = 0xAA;
    i = 0;
    while (i < size) : (i += 1) p[i] = 0x00;
    std.atomic.fence(.SeqCst);
}

pub fn duplicateMemory(allocator: Allocator, data: []const u8) ![]u8 {
    const dup = try allocator.alloc(u8, data.len);
    @memcpy(dup, data);
    return dup;
}

pub fn concatenateMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    const total = try addChecked(a.len, b.len);
    const cat = try allocator.alloc(u8, total);
    @memcpy(cat[0..a.len], a);
    @memcpy(cat[a.len..total], b);
    return cat;
}

pub fn searchMemory(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return null;
    if (needle.len > haystack.len) return null;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (mem.eql(u8, haystack[i .. i + needle.len], needle)) return i;
    }
    return null;
}

pub fn replaceMemory(data: []u8, old: u8, new: u8) void {
    for (data) |*c| {
        if (c.* == old) c.* = new;
    }
}

pub fn reverseMemory(data: []u8) void {
    mem.reverse(u8, data);
}

pub fn rotateMemory(data: []u8, shift: usize) void {
    mem.rotate(u8, data, shift);
}

pub fn countMemory(data: []const u8, value: u8) usize {
    var count: usize = 0;
    for (data) |c| {
        if (c == value) count += 1;
    }
    return count;
}

pub fn sumMemory(data: []const u8) u64 {
    var sum: u64 = 0;
    for (data) |c| sum += c;
    return sum;
}

pub fn productMemory(data: []const u8) u64 {
    var prod: u64 = 1;
    for (data) |c| prod *= c;
    return prod;
}

pub fn minMemory(data: []const u8) !u8 {
    if (data.len == 0) return error.Empty;
    return mem.min(u8, data);
}

pub fn maxMemory(data: []const u8) !u8 {
    if (data.len == 0) return error.Empty;
    return mem.max(u8, data);
}

pub fn sortMemory(data: []u8) void {
    std.sort.heap(u8, data, {}, std.sort.asc(u8));
}

pub fn shuffleMemory(data: []u8, seed: u64) void {
    var prng = std.rand.DefaultPrng.init(seed);
    prng.random().shuffle(u8, data);
}

pub fn uniqueMemory(allocator: Allocator, data: []const u8) ![]u8 {
    var set = std.AutoHashMap(u8, void).init(allocator);
    defer set.deinit();
    for (data) |c| try set.put(c, {});
    var out = try allocator.alloc(u8, set.count());
    var it = set.keyIterator();
    var i: usize = 0;
    while (it.next()) |k| {
        out[i] = k.*;
        i += 1;
    }
    return out;
}

pub fn intersectMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set_a = std.AutoHashMap(u8, void).init(allocator);
    defer set_a.deinit();
    for (a) |c| try set_a.put(c, {});
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    for (b) |c| {
        if (set_a.contains(c)) try list.append(c);
    }
    return try list.toOwnedSlice();
}

pub fn unionMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set = std.AutoHashMap(u8, void).init(allocator);
    defer set.deinit();
    for (a) |c| try set.put(c, {});
    for (b) |c| try set.put(c, {});
    var out = try allocator.alloc(u8, set.count());
    var it = set.keyIterator();
    var i: usize = 0;
    while (it.next()) |k| {
        out[i] = k.*;
        i += 1;
    }
    return out;
}

pub fn differenceMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set_b = std.AutoHashMap(u8, void).init(allocator);
    defer set_b.deinit();
    for (b) |c| try set_b.put(c, {});
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    for (a) |c| {
        if (!set_b.contains(c)) try list.append(c);
    }
    return try list.toOwnedSlice();
}

pub fn isSubsetMemory(allocator: Allocator, a: []const u8, b: []const u8) !bool {
    var set_b = std.AutoHashMap(u8, void).init(allocator);
    defer set_b.deinit();
    for (b) |c| try set_b.put(c, {});
    for (a) |c| {
        if (!set_b.contains(c)) return false;
    }
    return true;
}

pub fn isSupersetMemory(allocator: Allocator, a: []const u8, b: []const u8) !bool {
    return isSubsetMemory(allocator, b, a);
}

pub fn isDisjointMemory(allocator: Allocator, a: []const u8, b: []const u8) !bool {
    var set_a = std.AutoHashMap(u8, void).init(allocator);
    defer set_a.deinit();
    for (a) |c| try set_a.put(c, {});
    for (b) |c| {
        if (set_a.contains(c)) return false;
    }
    return true;
}

pub const MemoryStats = struct {
    allocated: usize,
    freed: usize,
    peak: usize,
};

var global_memory_stats: std.atomic.Value(MemoryStats) = std.atomic.Value(MemoryStats).init(.{ .allocated = 0, .freed = 0, .peak = 0 });

pub fn trackAllocation(size: usize) void {
    _ = global_memory_stats.fetchUpdate(.SeqCst, .SeqCst, struct {
        fn f(s: MemoryStats) ?MemoryStats {
            const a = s.allocated + size;
            if (a < s.allocated) return null;
            const current = a - s.freed;
            const peak = if (current > s.peak) current else s.peak;
            return .{ .allocated = a, .freed = s.freed, .peak = peak };
        }
    }.f) catch {};
}

pub fn trackFree(size: usize) void {
    _ = global_memory_stats.fetchUpdate(.SeqCst, .SeqCst, struct {
        fn f(s: MemoryStats) ?MemoryStats {
            const f0 = s.freed + size;
            if (f0 < s.freed) return null;
            return .{ .allocated = s.allocated, .freed = f0, .peak = s.peak };
        }
    }.f) catch {};
}

pub fn getMemoryStats() MemoryStats {
    return global_memory_stats.load(.SeqCst);
}

pub fn resetMemoryStats() void {
    global_memory_stats.store(.{ .allocated = 0, .freed = 0, .peak = 0 }, .SeqCst);
}

pub fn memoryFootprint() usize {
    const s = getMemoryStats();
    if (s.allocated < s.freed) return 0;
    return s.allocated - s.freed;
}

pub fn memoryPressure() f32 {
    const s = getMemoryStats();
    const cur = if (s.allocated < s.freed) 0 else s.allocated - s.freed;
    if (s.peak == 0) return 0.0;
    return @as(f32, @floatFromInt(cur)) / @as(f32, @floatFromInt(s.peak));
}

pub const TrackingAllocator = struct {
    parent: Allocator,

    pub fn init(parent: Allocator) TrackingAllocator {
        return .{ .parent = parent };
    }

    pub fn allocator(self: *TrackingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = trackingAlloc,
                .resize = trackingResize,
                .free = trackingFree,
            },
        };
    }

    fn trackingAlloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const p = self.parent.rawAlloc(len, ptr_align, ret_addr);
        if (p != null) trackAllocation(len);
        return p;
    }

    fn trackingResize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const old_len = buf.len;
        const ok = self.parent.rawResize(buf, buf_align, new_len, ret_addr);
        if (ok) {
            if (new_len > old_len) {
                trackAllocation(new_len - old_len);
            } else {
                trackFree(old_len - new_len);
            }
        }
        return ok;
    }

    fn trackingFree(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        trackFree(buf.len);
        self.parent.rawFree(buf, buf_align, ret_addr);
    }
};

pub const ReadWriteLock = struct {
    readers: AtomicU64,
    writer: AtomicBool,
    mutex: Mutex,

    pub fn init() ReadWriteLock {
        return .{
            .readers = AtomicU64.init(0),
            .writer = AtomicBool.init(false),
            .mutex = .{},
        };
    }

    pub fn readLock(self: *ReadWriteLock) void {
        while (true) {
            self.mutex.lock();
            if (!self.writer.load(.Acquire)) {
                _ = self.readers.fetchAdd(1, .AcqRel);
                self.mutex.unlock();
                return;
            }
            self.mutex.unlock();
            std.atomic.spinLoopHint();
        }
    }

    pub fn readUnlock(self: *ReadWriteLock) void {
        _ = self.readers.fetchSub(1, .AcqRel);
    }

    pub fn writeLock(self: *ReadWriteLock) void {
        while (true) {
            self.mutex.lock();
            if (!self.writer.load(.Acquire) and self.readers.load(.Acquire) == 0) {
                self.writer.store(true, .Release);
                self.mutex.unlock();
                return;
            }
            self.mutex.unlock();
            std.atomic.spinLoopHint();
        }
    }

    pub fn writeUnlock(self: *ReadWriteLock) void {
        self.mutex.lock();
        self.writer.store(false, .Release);
        self.mutex.unlock();
    }
};

pub fn atomicFlagTestAndSet(flag: *AtomicBool) bool {
    return flag.swap(true, .SeqCst);
}

pub fn atomicFlagClear(flag: *AtomicBool) void {
    flag.store(false, .SeqCst);
}

pub fn spinLockAcquire(lock: *AtomicU64) void {
    while (lock.compareExchangeStrong(0, 1, .Acquire, .Monotonic) != null) {
        std.atomic.spinLoopHint();
    }
}

pub fn spinLockRelease(lock: *AtomicU64) void {
    lock.store(0, .Release);
}

pub fn memoryPatternFill(ptr: [*]u8, size: usize, pattern: []const u8) !void {
    if (size == 0) return;
    if (pattern.len == 0) return error.InvalidPattern;
    var i: usize = 0;
    while (i < size) : (i += pattern.len) {
        const copy_len = @min(pattern.len, size - i);
        @memcpy(ptr[i .. i + copy_len], pattern[0..copy_len]);
    }
}

pub fn memoryPatternVerify(ptr: [*]const u8, size: usize, pattern: []const u8) !bool {
    if (size == 0) return true;
    if (pattern.len == 0) return error.InvalidPattern;
    var i: usize = 0;
    while (i < size) : (i += pattern.len) {
        const check_len = @min(pattern.len, size - i);
        if (!mem.eql(u8, ptr[i .. i + check_len], pattern[0..check_len])) return false;
    }
    return true;
}

pub fn virtualMemoryMap(addr: ?*anyopaque, size: usize, prot: u32, flags: u32) !*anyopaque {
    if (size == 0) return error.InvalidSize;
    const hint: ?[*]u8 = if (addr) |a| @ptrCast(a) else null;
    const mapped = try std.os.mmap(hint, size, prot, flags, -1, 0);
    return @ptrCast(mapped.ptr);
}

pub fn virtualMemoryUnmap(addr: *anyopaque, size: usize) !void {
    if (size == 0) return;
    try std.os.munmap(@as([*]u8, @ptrCast(addr))[0..size]);
}

pub fn protectMemory(addr: *anyopaque, size: usize, prot: u32) !void {
    if (size == 0) return error.InvalidSize;

    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);

    try std.os.mprotect(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size], prot);
}

pub fn lockMemory(addr: *anyopaque, size: usize) !void {
    if (size == 0) return error.InvalidSize;

    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);

    try std.os.mlock(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size]);
}

pub fn unlockMemory(addr: *anyopaque, size: usize) !void {
    if (size == 0) return;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    try std.os.munlock(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size]);
}

pub fn adviseMemory(addr: *anyopaque, size: usize, advice: i32) !void {
    if (size == 0) return error.InvalidSize;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    try std.os.madvise(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size], advice);
}

pub fn prefetchMemory(addr: *const anyopaque, size: usize) void {
    const cache_line = MemoryConfig.CACHE_LINE_SIZE;
    const p = @as([*]const u8, @ptrCast(addr));
    var i: usize = 0;
    while (i < size) : (i += cache_line) {
        @prefetch(p + i, .{ .rw = .read, .locality = 3, .cache = .data });
    }
}

pub fn trimExcessCapacity(allocator: Allocator, buf: []u8, used: usize) ![]u8 {
    if (used > buf.len) return error.InvalidSize;
    if (used == buf.len) return buf;
    return try allocator.realloc(buf, used);
}

pub fn splitMemory(allocator: Allocator, data: []const u8, delim: u8) ![][]const u8 {
    var count: usize = 1;
    for (data) |c| {
        if (c == delim) count += 1;
    }

    const parts = try allocator.alloc([]const u8, count);
    var start: usize = 0;
    var i: usize = 0;

    var j: usize = 0;
    while (j < data.len) : (j += 1) {
        if (data[j] == delim) {
            parts[i] = data[start..j];
            start = j + 1;
            i += 1;
        }
    }

    parts[i] = data[start..];
    return parts;
}

pub fn branchlessSelect(cond: bool, true_val: usize, false_val: usize) usize {
    const mask: usize = @as(usize, 0) - @as(usize, @intFromBool(cond));
    return (true_val & mask) | (false_val & ~mask);
}

pub fn criticalSectionEnter(mutex: *Mutex) void {
    mutex.lock();
}

pub fn criticalSectionExit(mutex: *Mutex) void {
    mutex.unlock();
}

pub fn waitOnCondition(cond: *CondVar, mutex: *Mutex, predicate: *const fn () bool) void {
    while (!predicate()) {
        cond.wait(mutex);
    }
}

pub fn signalCondition(cond: *CondVar) void {
    cond.signal();
}

pub fn broadcastCondition(cond: *CondVar) void {
    cond.broadcast();
}

pub fn semaphoreWait(sem: *Semaphore) void {
    sem.wait();
}

pub fn semaphorePost(sem: *Semaphore) void {
    sem.post();
}

pub fn compressMemory(data: []const u8, allocator: Allocator) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    errdefer list.deinit();

    var compressor = try std.compress.deflate.compressor(list.writer(), .{ .level = .best_compression });
    try compressor.writeAll(data);
    try compressor.finish();

    return try list.toOwnedSlice();
}

pub fn decompressMemory(data: []const u8, allocator: Allocator) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    errdefer list.deinit();

    var fbs = std.io.fixedBufferStream(data);
    var decompressor = try std.compress.deflate.decompressor(fbs.reader());
    var reader = decompressor.reader();

    var buf: [4096]u8 = undefined;
    while (true) {
        const n = try reader.read(&buf);
        if (n == 0) break;
        try list.appendSlice(buf[0..n]);
    }

    return try list.toOwnedSlice();
}

pub const EncryptedBlob = struct {
    nonce: [12]u8,
    tag: [16]u8,
    ciphertext: []u8,
    allocator: Allocator,

    pub fn deinit(self: *EncryptedBlob) void {
        secureZeroMemory(@as([*]u8, @ptrCast(&self.nonce)), self.nonce.len);
        secureZeroMemory(@as([*]u8, @ptrCast(&self.tag)), self.tag.len);
        secureZeroMemory(self.ciphertext.ptr, self.ciphertext.len);
        self.allocator.free(self.ciphertext);
        self.ciphertext = emptyU8Slice();
    }
};

pub fn encryptMemory(allocator: Allocator, plaintext: []const u8, key: [32]u8) !EncryptedBlob {
    var nonce: [12]u8 = undefined;
    std.crypto.random.bytes(&nonce);

    const ciphertext = try allocator.alloc(u8, plaintext.len);
    @memcpy(ciphertext, plaintext);

    var tag: [16]u8 = undefined;
    std.crypto.aead.chacha20poly1305.encrypt(ciphertext, &tag, ciphertext, "", nonce, key);

    return .{
        .nonce = nonce,
        .tag = tag,
        .ciphertext = ciphertext,
        .allocator = allocator,
    };
}

pub fn decryptMemory(allocator: Allocator, blob: EncryptedBlob, key: [32]u8) ![]u8 {
    const out = try allocator.alloc(u8, blob.ciphertext.len);
    errdefer {
        secureZeroMemory(out.ptr, out.len);
        allocator.free(out);
    }

    std.crypto.aead.chacha20poly1305.decrypt(out, blob.ciphertext, blob.tag, "", blob.nonce, key) catch {
        return error.AuthenticationFailed;
    };

    return out;
}

pub const CompressedStorage = struct {
    compressed: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: []const u8) !CompressedStorage {
        const compressed = try compressMemory(data, allocator);
        return .{ .compressed = compressed, .allocator = allocator };
    }

    pub fn deinit(self: *CompressedStorage) void {
        secureZeroMemory(self.compressed.ptr, self.compressed.len);
        self.allocator.free(self.compressed);
        self.compressed = emptyU8Slice();
    }

    pub fn decompress(self: *const CompressedStorage) ![]u8 {
        return try decompressMemory(self.compressed, self.allocator);
    }
};

pub const EncryptedStorage = struct {
    encrypted: EncryptedBlob,
    key: [32]u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: []const u8, key: [32]u8) !EncryptedStorage {
        const encrypted = try encryptMemory(allocator, data, key);
        return .{ .encrypted = encrypted, .key = key, .allocator = allocator };
    }

    pub fn deinit(self: *EncryptedStorage) void {
        self.encrypted.deinit();
        secureZeroMemory(@as([*]u8, @ptrCast(&self.key)), self.key.len);
    }

    pub fn decrypt(self: *const EncryptedStorage) ![]u8 {
        return try decryptMemory(self.allocator, self.encrypted, self.key);
    }
};

pub fn memoryAlign(ptr: *anyopaque, alignment: usize) !*anyopaque {
    if (alignment == 0 or !isPow2(alignment)) return error.InvalidAlignment;
    const addr = @intFromPtr(ptr);
    const aligned_addr = mem.alignForward(usize, addr, alignment);
    return @ptrFromInt(aligned_addr);
}

pub fn isMemoryOverlap(a_start: *const anyopaque, a_size: usize, b_start: *const anyopaque, b_size: usize) !bool {
    const a_addr = @intFromPtr(a_start);
    const b_addr = @intFromPtr(b_start);
    const a_end = try addChecked(a_addr, a_size);
    const b_end = try addChecked(b_addr, b_size);
    return (a_addr < b_end) and (b_addr < a_end);
}

pub fn copyNonOverlapping(dest: []u8, src: []const u8) !void {
    if (dest.len != src.len) return error.SizeMismatch;
    if (try isMemoryOverlap(dest.ptr, dest.len, src.ptr, src.len)) return error.Overlap;
    @memcpy(dest, src);
}

pub fn moveMemory(dest: []u8, src: []const u8) !void {
    if (dest.len != src.len) return error.SizeMismatch;
    mem.copyBackwards(u8, dest, src);
}

pub const MemoryPool = PoolAllocator;
pub const MemoryArena = Arena;
pub const MemorySlab = SlabAllocator;
pub const MemoryBuddy = BuddyAllocator;
pub const MemoryLockFreeQueue = LockFreeQueue;
pub const MemoryLockFreeStack = MutexStack;

const testing = std.testing;

test "Arena allocation" {
    var arena = try Arena.init(testing.allocator, 1024);
    defer arena.deinit();
    const ptr1 = arena.alloc(128, 8).?;
    const ptr2 = arena.alloc(64, 4).?;
    try testing.expect(ptr1.len == 128);
    try testing.expect(ptr2.len == 64);
}

test "SlabAllocator" {
    var slab = try SlabAllocator.init(testing.allocator, 256, 4, 64);
    defer slab.deinit();
    const ptr1 = slab.alloc(100).?;
    const ptr2 = slab.alloc(150).?;
    try testing.expect(ptr1.len == 100);
    try testing.expect(ptr2.len == 150);
    try slab.free(ptr1);
    try slab.free(ptr2);
}

test "PoolAllocator" {
    var pool = try PoolAllocator.init(testing.allocator, 64, 16, 2);
    defer pool.deinit();
    const ptr1 = pool.alloc(64).?;
    const ptr2 = pool.alloc(64).?;
    try testing.expect(ptr1.len == 64);
    try testing.expect(ptr2.len == 64);
    try pool.free(ptr1);
    try pool.free(ptr2);
}

test "PageAllocator" {
    var page_alloc = try PageAllocator.init(testing.allocator, 4);
    defer page_alloc.deinit();
    const pages = page_alloc.allocPages(2).?;
    try testing.expect(pages.len == 8192);
    try page_alloc.freePages(pages);
}

test "ZeroCopySlice" {
    const data = "hello world";
    const zcs = ZeroCopySlice.init(@as([*]const u8, @ptrCast(data.ptr)), data.len);
    const slice = zcs.slice(0, 5);
    try testing.expectEqualStrings("hello", slice.asBytes());
}

test "ResizeBuffer" {
    var buf = ResizeBuffer.init(testing.allocator);
    defer buf.deinit();
    try buf.append("hello");
    try buf.append(" world");
    const owned = try buf.toOwnedSlice();
    defer testing.allocator.free(owned);
    try testing.expectEqualStrings("hello world", owned);
}

test "ArenaAllocator basic allocation" {
    var arena = ArenaAllocator.init(testing.allocator, 1024);
    defer arena.deinit();

    const alloc = arena.allocator();
    const slice1 = try alloc.alloc(u8, 100);
    const slice2 = try alloc.alloc(u8, 100);

    @memset(slice1, 42);
    @memset(slice2, 84);

    try testing.expectEqual(@as(u8, 42), slice1[0]);
    try testing.expectEqual(@as(u8, 84), slice2[0]);
}

test "zero copy transfer" {
    var src = [_]u8{ 1, 2, 3, 4, 5 };
    var dest: [5]u8 = undefined;

    zeroCopyTransfer(&src, &dest);

    try testing.expectEqualSlices(u8, &src, &dest);
}

test "memory hashing" {
    const data1 = "hello world";
    const data2 = "hello world";
    const data3 = "hello world!";

    const hash1 = hashMemory(data1);
    const hash2 = hashMemory(data2);
    const hash3 = hashMemory(data3);

    try testing.expectEqual(hash1, hash2);
    try testing.expect(hash1 != hash3);
}

test "memory comparison constant time" {
    const data1 = "test";
    const data2 = "test";
    const data3 = "best";

    try testing.expect(compareMemory(data1, data2));
    try testing.expect(!compareMemory(data1, data3));
}

test "search memory" {
    const haystack = "hello world, hello universe";
    const needle = "world";

    const pos = searchMemory(haystack, needle);
    try testing.expect(pos != null);
    try testing.expectEqual(@as(usize, 6), pos.?);
}

test "count memory" {
    const data = "hello world";
    const count = countMemory(data, 'l');
    try testing.expectEqual(@as(usize, 3), count);
}

test "unique memory" {
    const data = "aabbccddaa";
    const uniq = try uniqueMemory(testing.allocator, data);
    defer testing.allocator.free(uniq);
    try testing.expect(uniq.len == 4);
}

test "atomic operations" {
    var value: u64 = 0;

    const prev = atomicAdd(&value, 5);
    try testing.expectEqual(@as(u64, 0), prev);
    try testing.expectEqual(@as(u64, 5), atomicLoad(&value));

    atomicStore(&value, 10);
    try testing.expectEqual(@as(u64, 10), atomicLoad(&value));

    _ = atomicInc(&value);
    try testing.expectEqual(@as(u64, 11), atomicLoad(&value));
}

test "ReadWriteLock" {
    var rwlock = ReadWriteLock.init();

    rwlock.readLock();
    rwlock.readUnlock();

    rwlock.writeLock();
    rwlock.writeUnlock();
}

test "BuddyAllocator" {
    var buddy = try BuddyAllocator.init(testing.allocator, 4096, 6);
    defer buddy.deinit();

    const ptr1 = try buddy.alloc(128);
    try testing.expect(ptr1.len == 128);
    try buddy.free(ptr1);
}

test "LockFreeQueue" {
    var queue = try LockFreeQueue.init(testing.allocator, 16);
    defer queue.deinit();

    var item: usize = 42;
    try testing.expect(queue.enqueue(@as(*anyopaque, @ptrCast(&item))));

    const retrieved = queue.dequeue();
    try testing.expect(retrieved != null);
    try testing.expectEqual(@intFromPtr(@as(*anyopaque, @ptrCast(&item))), @intFromPtr(retrieved.?));
}

test "LockFreeStack" {
    var stack = LockFreeStack.init(testing.allocator);
    defer stack.deinit();

    var item: usize = 42;
    try stack.push(@as(*anyopaque, @ptrCast(&item)));

    const retrieved = stack.pop();
    try testing.expect(retrieved != null);
    try testing.expectEqual(@intFromPtr(@as(*anyopaque, @ptrCast(&item))), @intFromPtr(retrieved.?));
}

test "memory stats tracking" {
    resetMemoryStats();

    trackAllocation(100);
    trackAllocation(200);
    trackFree(50);

    const stats = getMemoryStats();
    try testing.expectEqual(@as(usize, 300), stats.allocated);
    try testing.expectEqual(@as(usize, 50), stats.freed);
    try testing.expectEqual(@as(usize, 250), memoryFootprint());
}