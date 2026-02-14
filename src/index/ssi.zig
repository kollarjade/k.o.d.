const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const stableHash = @import("../core/io.zig").stableHash;
const Error = types.Error;

pub const SSI = struct {
    root: ?*Node,
    allocator: Allocator,
    height: usize = 0,
    size: usize = 0,
    max_height: usize = 6,

    const Node = struct {
        hash: u64,
        children: ?[*]?*Node,
        segment: ?Segment,
        collision_chain: ?*CollisionNode,
        height: usize,
        is_leaf: bool,

        pub fn init(allocator: Allocator, hash: u64, height: usize) !Node {
            _ = allocator;
            return .{
                .hash = hash,
                .children = null,
                .segment = null,
                .collision_chain = null,
                .height = height,
                .is_leaf = height == 0,
            };
        }

        pub fn deinit(self: *Node, allocator: Allocator) void {
            if (self.children) |_| {
                const height_clamped = @min(self.height, 6);
                allocator.free(self.children.?[0..@as(usize, 1) << @as(u6, @intCast(height_clamped))]);
            }
            if (self.segment) |*seg| seg.deinit(allocator);
            var chain = self.collision_chain;
            while (chain) |c| {
                const next = c.next;
                c.seg.deinit(allocator);
                allocator.destroy(c);
                chain = next;
            }
        }
    };

    const CollisionNode = struct {
        seg: Segment,
        next: ?*CollisionNode,
    };

    const Segment = struct {
        tokens: []u32,
        position: u64,
        score: f32,
        anchor_hash: u64,

        pub fn init(allocator: Allocator, tokens: []u32, pos: u64, score: f32, anchor: u64) !Segment {
            return .{ .tokens = try allocator.dupe(u32, tokens), .position = pos, .score = score, .anchor_hash = anchor };
        }

        pub fn deinit(self: *Segment, allocator: Allocator) void {
            allocator.free(self.tokens);
        }

        pub fn hash(self: *const Segment) u64 {
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(std.mem.asBytes(&self.position));
            hasher.update(std.mem.asBytes(&self.score));
            var i: usize = 0;
            while (i < self.tokens.len) : (i += 1) {
                hasher.update(std.mem.asBytes(&self.tokens[i]));
            }
            return hasher.final();
        }
    };

    pub fn init(allocator: Allocator) SSI {
        return .{ .root = null, .allocator = allocator };
    }

    fn recursiveDeinit(node: *Node, allocator: Allocator) void {
        if (node.children) |ch| {
            const height_clamped = @min(node.height, 6);
            var i: usize = 0;
            while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                if (ch[i]) |child| {
                    recursiveDeinit(child, allocator);
                }
            }
        }
        node.deinit(allocator);
        allocator.destroy(node);
    }

    pub fn deinit(self: *SSI) void {
        if (self.root) |root| {
            recursiveDeinit(root, self.allocator);
        }
    }

    pub fn addSequence(self: *SSI, tokens: []u32, position: u64, is_anchor: bool) !void {
        const segment_hash = blk: {
            var hasher = std.hash.Wyhash.init(position);
            var i: usize = 0;
            while (i < tokens.len) : (i += 1) {
                hasher.update(std.mem.asBytes(&tokens[i]));
            }
            if (is_anchor) _ = hasher.update(&[_]u8{1});
            break :blk hasher.final();
        };
        var current = &self.root;
        var h = @min(self.height, self.max_height);
        while (h > 0 or current.* == null) {
            if (current.* == null) {
                const node = try self.allocator.create(Node);
                node.* = try Node.init(self.allocator, 0, h);
                current.* = node;
                if (h == self.height and self.height < self.max_height) self.height += 1;
            }
            if (h > 0) {
                const effective_height = @min(h, 6);
                const effective_height_u8 = @as(u8, @intCast(effective_height));
                if (current.*.?.children == null) {
                    current.*.?.height = effective_height;
                    current.*.?.is_leaf = false;
                    const children = try self.allocator.alloc(?*Node, @as(usize, 1) << @as(u6, @intCast(effective_height_u8)));
                    @memset(children, null);
                    current.*.?.children = children.ptr;
                }
                const node_height_u6 = @as(u6, @intCast(effective_height_u8));
                const shift_amount = @as(u6, @intCast(@as(u8, 64) - effective_height_u8));
                const mask = ((@as(u64, 1) << node_height_u6) - 1);
                const bucket = (segment_hash >> shift_amount) & mask;
                h -= 1;
                current = &current.*.?.children.?[bucket];
            } else break;
        }
        const leaf = current.* orelse blk: {
            const node = try self.allocator.create(Node);
            node.* = try Node.init(self.allocator, segment_hash, 0);
            node.segment = try Segment.init(self.allocator, tokens, position, 0.0, if (is_anchor) segment_hash else 0);
            current.* = node;
            break :blk node;
        };
        leaf.hash = segment_hash;
        if (leaf.segment != null and leaf.segment.?.position != position) {
            const collision = try self.allocator.create(CollisionNode);
            collision.* = .{
                .seg = try Segment.init(self.allocator, tokens, position, 0.0, if (is_anchor) segment_hash else 0),
                .next = leaf.collision_chain,
            };
            leaf.collision_chain = collision;
        } else {
            if (leaf.segment) |*seg| seg.deinit(self.allocator);
            leaf.segment = try Segment.init(self.allocator, tokens, position, 0.0, if (is_anchor) segment_hash else 0);
        }
        self.size += 1;
        try self.compact();
    }

    pub fn retrieveTopK(self: *const SSI, query_tokens: []u32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        defer heap.deinit();
        const query_hash = blk: {
            var hasher = std.hash.Wyhash.init(0);
            var i: usize = 0;
            while (i < query_tokens.len) : (i += 1) {
                hasher.update(std.mem.asBytes(&query_tokens[i]));
            }
            break :blk hasher.final();
        };
        try self.traverse(self.root, query_hash, &heap, k);
        var top_k = try allocator.alloc(types.RankedSegment, @min(k, heap.count()));
        var i: usize = heap.count();
        while (heap.removeOrNull()) |seg| {
            if (i > 0) {
                i -= 1;
                top_k[i] = seg;
            }
        }
        return top_k;
    }

    fn traverse(self: *const SSI, node: ?*Node, query_hash: u64, heap: anytype, k: usize) !void {
        if (node == null) return;
        const n = node.?;
        if (n.is_leaf) {
            if (n.segment) |seg| {
                try self.addSegmentToHeap(seg, query_hash, heap, k);
            }
            var chain = n.collision_chain;
            while (chain) |c| {
                try self.addSegmentToHeap(c.seg, query_hash, heap, k);
                chain = c.next;
            }
        } else {
            if (n.children) |ch| {
                if (n.height > 0) {
                    const node_height = @min(n.height, 6);
                    const node_height_u8 = @as(u8, @intCast(node_height));
                    const node_height_u6 = @as(u6, @intCast(node_height_u8));
                    const shift_amount = @as(u6, @intCast(@as(u8, 64) - node_height_u8));
                    const mask = ((@as(u64, 1) << node_height_u6) - 1);
                    const bucket = (query_hash >> shift_amount) & mask;
                    if (ch[bucket]) |child| try self.traverse(child, query_hash, heap, k);
                    var i: usize = 0;
                    while (i < (@as(usize, 1) << node_height_u6)) : (i += 1) {
                        if (i == bucket) continue;
                        if (ch[i]) |child| try self.traverse(child, query_hash, heap, k);
                    }
                }
            }
        }
    }

    fn addSegmentToHeap(self: *const SSI, seg: Segment, query_hash: u64, heap: anytype, k: usize) !void {
        const similarity = self.computeSimilarity(query_hash, seg.hash());
        const ranked = types.RankedSegment{
            .tokens = try self.allocator.dupe(u32, seg.tokens),
            .score = similarity,
            .position = seg.position,
            .anchor = seg.anchor_hash != 0,
        };
        if (heap.count() < k) {
            try heap.add(ranked);
        } else if (heap.peek()) |top| {
            if (similarity > top.score) {
                var removed = heap.remove();
                removed.deinit(self.allocator);
                try heap.add(ranked);
            } else {
                self.allocator.free(ranked.tokens);
            }
        }
    }

    fn computeSimilarity(self: *const SSI, h1: u64, h2: u64) f32 {
        _ = self;
        const xor_val = @popCount(h1 ^ h2);
        return 1.0 - (@as(f32, @floatFromInt(xor_val)) / 64.0);
    }

    pub fn compact(self: *SSI) !void {
        if (self.size < 1000) return;
        const old_root = self.root;
        const new_root = try self.allocator.create(Node);
        new_root.* = try Node.init(self.allocator, 0, self.height);
        self.root = new_root;
        if (old_root) |root| {
            root.deinit(self.allocator);
            self.allocator.destroy(root);
        }
        self.size = 0;
    }

    pub fn updateScore(self: *SSI, position: u64, new_score: f32) !void {
        var current = self.root;
        while (current) |node| {
            if (node.is_leaf and node.segment != null) {
                if (node.segment.?.position == position) {
                    node.segment.?.score = new_score;
                    return;
                }
                var chain = node.collision_chain;
                while (chain) |c| {
                    if (c.seg.position == position) {
                        c.seg.score = new_score;
                        return;
                    }
                    chain = c.next;
                }
            }
            if (node.children) |ch| {
                if (node.height > 0 and node.height <= 64) {
                    const shift_val = @as(u6, @intCast(64 - node.height));
                    const bucket = (position >> shift_val) & ((@as(u64, 1) << @as(u6, @intCast(@min(node.height, 63)))) - 1);
                    current = ch[bucket];
                } else break;
            } else break;
        }
        return Error.OutOfBounds;
    }

    pub fn getSegment(self: *const SSI, position: u64) ?Segment {
        var current = self.root;
        var h: i64 = @intCast(self.height);
        while (current != null and h >= 0) {
            const node = current.?;
            if (h == 0) {
                if (node.segment) |seg| if (seg.position == position) return seg;
                var chain = node.collision_chain;
                while (chain) |c| {
                    if (c.seg.position == position) return c.seg;
                    chain = c.next;
                }
                return null;
            }
            if (node.children) |ch| {
                if (h > 0 and h <= 64) {
                    const h_clamped = @min(h, 63);
                    const shift_val = @as(u6, @intCast(64 - h_clamped));
                    const bucket = (position >> shift_val) & ((@as(u64, 1) << @as(u6, @intCast(h_clamped))) - 1);
                    current = ch[bucket];
                } else return null;
            } else return null;
            h -= 1;
        }
        return null;
    }

    pub fn exportToTensor(self: *SSI, allocator: Allocator) !Tensor {
        const num_segments = if (self.size > 0) self.size else 1;
        const t = try Tensor.init(allocator, &.{ num_segments, 128 });
        var idx: usize = 0;
        var stack = std.ArrayList(*Node).init(allocator);
        defer stack.deinit();
        if (self.root) |root| try stack.append(root);
        while (stack.popOrNull()) |node| {
            if (node.is_leaf and node.segment != null) {
                const seg = node.segment.?;
                var i: usize = 0;
                while (i < seg.tokens.len) : (i += 1) {
                    const tok = seg.tokens[i];
                    if (i < 128 and idx * 128 + i < t.data.len) {
                        t.data[idx * 128 + i] = @floatFromInt(tok);
                    }
                }
                idx += 1;
            } else if (node.children) |ch| {
                const height_clamped = @min(node.height, 6);
                var i: usize = 0;
                while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                    if (ch[i]) |child| try stack.append(child);
                }
            }
        }
        return t;
    }

    pub fn importFromTensor(self: *SSI, t: *const Tensor) !void {
        if (self.root != null) {
            recursiveDeinit(self.root.?, self.allocator);
        }
        self.root = null;
        self.height = 0;
        self.size = 0;
        if (t.shape.dims.len < 1) return;
        const num_segments = t.shape.dims[0];
        var s: usize = 0;
        while (s < num_segments) : (s += 1) {
            var tokens: [128]u32 = undefined;
            var i: usize = 0;
            while (i < 128) : (i += 1) {
                const idx = s * 128 + i;
                if (idx < t.data.len) {
                    const val = t.data[idx];
                    if (val >= 0 and val < @as(f32, @floatFromInt(std.math.maxInt(u32)))) {
                        tokens[i] = @intFromFloat(val);
                    } else {
                        tokens[i] = 0;
                    }
                } else {
                    tokens[i] = 0;
                }
            }
            const pos_idx = s * 128 + 127;
            const position: u64 = if (pos_idx < t.data.len and t.data[pos_idx] >= 0)
                @intFromFloat(t.data[pos_idx])
            else
                0;
            try self.addSequence(tokens[0..127], position, false);
        }
    }

    pub fn merge(self: *SSI, other: *const SSI) !void {
        var stack = std.ArrayList(*Node).init(self.allocator);
        defer stack.deinit();
        if (other.root) |root| try stack.append(root);
        while (stack.popOrNull()) |node| {
            if (node.is_leaf and node.segment != null) {
                const seg = node.segment.?;
                try self.addSequence(seg.tokens, seg.position, seg.anchor_hash != 0);
            } else if (node.children) |ch| {
                const height_clamped = @min(node.height, 6);
                var i: usize = 0;
                while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                    if (ch[i]) |child| try stack.append(child);
                }
            }
        }
    }

    pub fn split(self: *SSI, threshold: f32) !SSI {
        var new_ssi = SSI.init(self.allocator);
        var stack = std.ArrayList(*Node).init(self.allocator);
        defer stack.deinit();
        if (self.root) |root| try stack.append(root);
        while (stack.popOrNull()) |node| {
            if (node.is_leaf and node.segment != null) {
                const seg = node.segment.?;
                if (seg.score > threshold) {
                    try new_ssi.addSequence(seg.tokens, seg.position, seg.anchor_hash != 0);
                }
            } else if (node.children) |ch| {
                const height_clamped = @min(node.height, 6);
                var i: usize = 0;
                while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                    if (ch[i]) |child| try stack.append(child);
                }
            }
        }
        return new_ssi;
    }

    pub fn balance(self: *SSI) void {
        if (self.size < 1024) return;
        self.height = @as(usize, @intFromFloat(math.log2(@as(f32, @floatFromInt(self.size))))) + 1;
    }

    pub fn serialize(self: *SSI, writer: anytype) !void {
        try writer.writeInt(usize, self.height, .Little);
        try writer.writeInt(usize, self.size, .Little);
        var stack = std.ArrayList(*Node).init(self.allocator);
        defer stack.deinit();
        if (self.root) |root| try stack.append(root);
        while (stack.items.len > 0) {
            const node = stack.pop();
            try writer.writeInt(u64, node.hash, .Little);
            try writer.writeInt(usize, @intFromBool(node.is_leaf), .Little);
            if (node.segment) |seg| {
                try writer.writeInt(u64, seg.position, .Little);
                try writer.writeAll(std.mem.asBytes(&seg.score));
                try writer.writeInt(u64, seg.anchor_hash, .Little);
                try writer.writeInt(usize, seg.tokens.len, .Little);
                var i: usize = 0;
                while (i < seg.tokens.len) : (i += 1) {
                    try writer.writeInt(u32, seg.tokens[i], .Little);
                }
            }
            if (node.children) |ch| {
                const height_clamped = @min(node.height, 6);
                var i: usize = 0;
                while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                    const child_ptr = if (ch[i]) |c| @intFromPtr(c) else 0;
                    try writer.writeInt(usize, child_ptr, .Little);
                    if (ch[i]) |c| try stack.append(c);
                }
            }
        }
    }

    pub fn deserialize(allocator: Allocator, reader: anytype) !SSI {
        var ssi = SSI.init(allocator);
        const height = try reader.readInt(usize, .Little);
        const size = try reader.readInt(usize, .Little);
        ssi.height = height;
        ssi.size = size;
        var nodes = std.AutoHashMap(usize, *Node).init(allocator);
        defer nodes.deinit();
        const root_id = try reader.readInt(usize, .Little);
        if (root_id != 0) {
            const root = try allocator.create(Node);
            root.* = Node{ .hash = 0, .children = null, .segment = null, .collision_chain = null, .height = height, .is_leaf = false };
            try nodes.put(root_id, root);
            ssi.root = root;
        }
        var node_count: usize = 0;
        while (node_count < size) : (node_count += 1) {
            const node_id = try reader.readInt(usize, .Little);
            if (node_id == 0) break;
            const node = try allocator.create(Node);
            node.hash = try reader.readInt(u64, .Little);
            const is_leaf = try reader.readInt(bool, .Little);
            node.is_leaf = is_leaf;
            node.height = if (is_leaf) 0 else height;
            node.collision_chain = null;
            if (is_leaf) {
                const pos = try reader.readInt(u64, .Little);
                var score_bytes: [@sizeOf(f32)]u8 = undefined;
                _ = try reader.read(&score_bytes);
                const score = @as(*const f32, @ptrCast(&score_bytes)).*;
                const anchor = try reader.readInt(u64, .Little);
                const len = try reader.readInt(usize, .Little);
                const tokens = try allocator.alloc(u32, len);
                var i: usize = 0;
                while (i < len) : (i += 1) {
                    tokens[i] = try reader.readInt(u32, .Little);
                }
                node.segment = Segment{ .tokens = tokens, .position = pos, .score = score, .anchor_hash = anchor };
                node.children = null;
            } else {
                const height_clamped = @min(node.height, 6);
                node.children = (try allocator.alloc(?*Node, @as(usize, 1) << @as(u6, @intCast(height_clamped)))).ptr;
                @memset(node.children.?[0..@as(usize, 1) << @as(u6, @intCast(height_clamped))], null);
                var i: usize = 0;
                while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                    const child_id = try reader.readInt(usize, .Little);
                    if (child_id != 0) {
                        if (nodes.get(child_id)) |child| {
                            node.children.?[i] = child;
                        } else {
                            const child = try allocator.create(Node);
                            child.* = Node{ .hash = 0, .children = null, .segment = null, .collision_chain = null, .height = if (node.height > 0) node.height - 1 else 0, .is_leaf = false };
                            node.children.?[i] = child;
                            try nodes.put(child_id, child);
                        }
                    }
                }
                node.segment = null;
            }
            try nodes.put(node_id, node);
        }
        return ssi;
    }

    pub fn stats(self: *const SSI) struct { nodes: usize, leaves: usize, depth: usize } {
        var nodes: usize = 0;
        var leaves: usize = 0;
        var depth: usize = 0;
        var stack = std.ArrayList(struct { node: *const Node, d: usize }).init(self.allocator);
        defer stack.deinit();
        if (self.root) |root| {
            stack.append(.{ .node = root, .d = 0 }) catch {
                return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
            };
        }
        while (stack.popOrNull()) |entry| {
            nodes += 1;
            if (entry.node.is_leaf) leaves += 1;
            if (entry.d > depth) depth = entry.d;
            if (entry.node.children) |ch| {
                const height_clamped = @min(entry.node.height, 6);
                var i: usize = 0;
                while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                    if (ch[i]) |child| {
                        stack.append(.{ .node = child, .d = entry.d + 1 }) catch {
                            continue;
                        };
                    }
                }
            }
        }
        return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
    }

    pub fn validate(self: *SSI) bool {
        var valid = true;
        var stack = std.ArrayList(*Node).init(self.allocator);
        defer stack.deinit();
        if (self.root) |root| {
            stack.append(root) catch return false;
        }
        while (stack.popOrNull()) |node| {
            if (node.is_leaf) {
                if (node.segment) |seg| {
                    if (stableHash(std.mem.sliceAsBytes(seg.tokens), seg.position) != node.hash) valid = false;
                } else valid = false;
            } else {
                var child_hash: u64 = 0;
                if (node.children) |ch| {
                    const height_clamped = @min(node.height, 6);
                    var i: usize = 0;
                    while (i < (@as(usize, 1) << @as(u6, @intCast(height_clamped)))) : (i += 1) {
                        if (ch[i]) |child| {
                            child_hash ^= child.hash;
                            stack.append(child) catch {
                                continue;
                            };
                        }
                    }
                }
                if (child_hash != node.hash) valid = false;
            }
        }
        return valid;
    }
};

test "SSI add and retrieve" {
    const testing = std.testing;
    var gpa = std.testing.allocator;
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3 }, 0, false);
    try ssi.addSequence(&.{ 4, 5, 6 }, 1, true);
    const top_k = try ssi.retrieveTopK(&.{ 1, 2 }, 2, gpa);
    defer {
        var i: usize = 0;
        while (i < top_k.len) : (i += 1) {
            top_k[i].deinit(gpa);
        }
        gpa.free(top_k);
    }
    try testing.expect(top_k.len == 2);
}

test "SSI compact" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    var i: usize = 0;
    while (i < 1001) : (i += 1) {
        try ssi.addSequence(&.{@as(u32, @intCast(i))}, @as(u64, @intCast(i)), false);
    }
    try testing.expect(ssi.size <= 1001);
}

test "SSI merge" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ssi1 = SSI.init(gpa);
    defer ssi1.deinit();
    try ssi1.addSequence(&.{1}, 1, false);
    var ssi2 = SSI.init(gpa);
    defer ssi2.deinit();
    try ssi2.addSequence(&.{2}, 2, false);
    try ssi1.merge(&ssi2);
    try testing.expect(ssi1.size == 2);
}

test "SSI serialize deserialize" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2 }, 42, true);
    var buf = std.ArrayList(u8).init(gpa);
    defer buf.deinit();
    try ssi.serialize(buf.writer());
    var fbs = std.io.fixedBufferStream(buf.items);
    var ssi2 = try SSI.deserialize(gpa, fbs.reader());
    defer ssi2.deinit();
    try testing.expect(ssi2.size == 1);
}

test "SSI stats" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{1}, 1, false);
    const st = ssi.stats();
    try testing.expect(st.nodes >= 1);
    try testing.expect(st.leaves >= 1);
}

test "SSI validate" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3 }, 123, false);
    try testing.expect(ssi.validate());
}
