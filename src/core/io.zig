const std = @import("std");
const fs = std.fs;
const mem = std.mem;
const math = std.math;
const builtin = @import("builtin");
const Allocator = mem.Allocator;

pub const IoConfig = struct {
    pub const BUFFER_SIZE: usize = 8192;
    pub const LARGE_CHUNK_SIZE: usize = 65536;
    pub const MAX_READ_BYTES: usize = 100 * 1024 * 1024;
    pub const MAX_FILE_SIZE: usize = 1024 * 1024 * 1024;
    pub const PAGE_SIZE: usize = mem.page_size;
    pub const MIX_SHIFT: u6 = 33;
    pub const TRUNCATE_SHIFT: u6 = 32;
    pub const MAX_FLUSH_DEPTH: usize = 10;
    pub const SECURE_FILE_MODE: u9 = 0o600;
    pub const MAX_PATH_LEN: usize = 4096;
    pub const CACHE_LINE_SIZE: usize = 128;
};

pub const IoError = error{
    InvalidFileSize,
    FileTooLarge,
    FileIsEmpty,
    BufferNotMapped,
    OutOfBounds,
    RecursionDepthExceeded,
    MaxBytesExceeded,
    InvalidPathCharacter,
    EndOfStream,
    UnexpectedEndOfFile,
    FileNotFound,
    AccessDenied,
    PathAlreadyExists,
    InvalidPath,
    NotADirectory,
    NotAFile,
    OperationFailed,
    ReadOnlyFile,
    Overflow,
    PathTooLong,
};

fn generateRuntimeSeed() u64 {
    var entropy_buf: [32]u8 = undefined;
    std.crypto.random.bytes(&entropy_buf);
    var hasher = std.crypto.hash.Blake2b256.init(.{});
    hasher.update(&entropy_buf);
    const digest = hasher.finalResult();
    secureZeroBytes(&entropy_buf);
    return mem.readInt(u64, digest[0..8], .Little);
}

fn secureZeroBytes(buf: []u8) void {
    const p: [*]volatile u8 = @ptrCast(buf.ptr);
    var i: usize = 0;
    while (i < buf.len) : (i += 1) {
        p[i] = 0;
    }
}

fn mixHash(h: u64) u64 {
    const prime1: u64 = 0xff51afd7ed558ccd;
    const prime2: u64 = 0xc4ceb9fe1a85ec53;
    var mixed = h ^ (h >> IoConfig.MIX_SHIFT);
    mixed *%= prime1;
    mixed ^= mixed >> IoConfig.MIX_SHIFT;
    mixed *%= prime2;
    return mixed ^ (mixed >> IoConfig.MIX_SHIFT);
}

fn addChecked(a: usize, b: usize) !usize {
    const r = @addWithOverflow(a, b);
    if (r[1] != 0) return IoError.Overflow;
    return r[0];
}

pub const MMAP = struct {
    file: fs.File,
    buffer: ?[]align(mem.page_size) u8,
    allocator: Allocator,
    is_writable: bool,
    actual_size: usize,
    mutex: std.Thread.Mutex,

    pub fn open(allocator: Allocator, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try fs.cwd().openFile(path, mode);
        errdefer file.close();
        return openFromFile(allocator, file, mode);
    }

    pub fn openWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try dir.openFile(path, mode);
        errdefer file.close();
        return openFromFile(allocator, file, mode);
    }

    fn openFromFile(allocator: Allocator, file: fs.File, mode: fs.File.OpenFlags) !MMAP {
        const stat = try file.stat();
        const size_u64: u64 = stat.size;
        if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
        if (size_u64 > IoConfig.MAX_FILE_SIZE) return error.FileTooLarge;
        var file_size: usize = @intCast(size_u64);

        const is_writable = mode.mode == .read_write or mode.mode == .write_only;

        if (file_size == 0) {
            if (is_writable) {
                try file.setEndPos(IoConfig.PAGE_SIZE);
                file_size = IoConfig.PAGE_SIZE;
            } else {
                return IoError.FileIsEmpty;
            }
        }

        var prot_flags: u32 = std.os.PROT.READ;
        if (is_writable) {
            prot_flags |= std.os.PROT.WRITE;
        }

        const aligned_size = mem.alignForward(usize, file_size, IoConfig.PAGE_SIZE);
        const map_type: std.os.MAP.TYPE = if (is_writable) .SHARED else .PRIVATE;

        const buffer = try std.os.mmap(
            null,
            aligned_size,
            prot_flags,
            .{ .TYPE = map_type },
            file.handle,
            0
        );

        return .{
            .file = file,
            .buffer = buffer,
            .allocator = allocator,
            .is_writable = is_writable,
            .actual_size = file_size,
            .mutex = .{},
        };
    }

    pub fn close(self: *MMAP) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.buffer) |buf| {
            std.os.munmap(buf);
            self.buffer = null;
        }
        self.file.close();
    }

    pub fn read(self: *MMAP, offset: usize, len: usize) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        if (offset >= self.actual_size) return IoError.OutOfBounds;
        const end = try addChecked(offset, len);
        if (end > self.actual_size) {
            return buf[offset..self.actual_size];
        }
        return buf[offset..end];
    }

    pub const SyncMode = enum {
        sync,
        nosync,
    };

    pub fn write(self: *MMAP, offset: usize, data: []const u8, sync_mode: SyncMode) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.is_writable) return IoError.ReadOnlyFile;
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        if (offset > self.actual_size) return IoError.OutOfBounds;
        const end = try addChecked(offset, data.len);
        if (end > buf.len) return IoError.OutOfBounds;
        @memcpy(buf[offset..end], data);
        if (sync_mode == .sync) {
            try std.os.msync(buf, .{ .SYNC = true });
        }
    }

    pub fn append(self: *MMAP, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.is_writable) return IoError.ReadOnlyFile;
        const buf = self.buffer orelse return IoError.BufferNotMapped;

        const current_size = self.actual_size;
        const new_size = try addChecked(current_size, data.len);
        if (new_size > IoConfig.MAX_FILE_SIZE) return error.FileTooLarge;

        std.os.munmap(buf);
        self.buffer = null;

        try self.file.setEndPos(new_size);
        try self.file.pwriteAll(data, current_size);

        const aligned_size = mem.alignForward(usize, new_size, IoConfig.PAGE_SIZE);

        const new_buf = try std.os.mmap(
            null,
            aligned_size,
            std.os.PROT.READ | std.os.PROT.WRITE,
            .{ .TYPE = .SHARED },
            self.file.handle,
            0
        );

        self.buffer = new_buf;
        self.actual_size = new_size;
    }

    pub fn sync(self: *MMAP) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        try std.os.msync(buf, .{ .SYNC = true });
    }

    pub fn size(self: *const MMAP) usize {
        return self.actual_size;
    }
};

pub const DurableWriter = struct {
    file: fs.File,
    buffer: [IoConfig.BUFFER_SIZE]u8,
    pos: usize,
    flush_depth: usize,
    enable_sync: bool,

    pub fn init(path: []const u8, enable_sync: bool) !DurableWriter {
        const file = try fs.cwd().createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .flush_depth = 0,
            .enable_sync = enable_sync,
        };
    }

    pub fn initWithDir(dir: fs.Dir, path: []const u8, enable_sync: bool) !DurableWriter {
        const file = try dir.createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .flush_depth = 0,
            .enable_sync = enable_sync,
        };
    }

    pub fn deinit(self: *DurableWriter) !void {
        try self.flush();
        if (self.enable_sync) {
            try self.file.sync();
        }
        self.file.close();
    }

    pub fn deinitNoError(self: *DurableWriter) void {
        self.flush() catch {};
        if (self.enable_sync) {
            self.file.sync() catch {};
        }
        self.file.close();
    }

    pub fn write(self: *DurableWriter, data: []const u8) !void {
        var remaining = data;
        while (remaining.len > 0) {
            const space = self.buffer.len - self.pos;
            if (space == 0) {
                try self.flush();
                continue;
            }
            const to_copy = @min(remaining.len, space);
            @memcpy(self.buffer[self.pos..self.pos + to_copy], remaining[0..to_copy]);
            self.pos += to_copy;
            remaining = remaining[to_copy..];
        }
    }

    pub fn flush(self: *DurableWriter) !void {
        if (self.flush_depth > IoConfig.MAX_FLUSH_DEPTH) {
            return IoError.RecursionDepthExceeded;
        }
        self.flush_depth += 1;
        defer self.flush_depth -= 1;

        if (self.pos > 0) {
            try self.file.writeAll(self.buffer[0..self.pos]);
            self.pos = 0;
        }
    }

    pub fn writeAll(self: *DurableWriter, data: []const u8) !void {
        try self.write(data);
        try self.flush();
    }
};

pub const BufferedReader = struct {
    file: fs.File,
    buffer: [IoConfig.BUFFER_SIZE]u8,
    pos: usize,
    limit: usize,
    max_read_bytes: usize,
    total_read: usize,

    pub fn init(path: []const u8) !BufferedReader {
        const file = try fs.cwd().openFile(path, .{});
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = IoConfig.MAX_READ_BYTES,
            .total_read = 0,
        };
    }

    pub fn initWithDir(dir: fs.Dir, path: []const u8) !BufferedReader {
        const file = try dir.openFile(path, .{});
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = IoConfig.MAX_READ_BYTES,
            .total_read = 0,
        };
    }

    pub fn initWithMaxBytes(path: []const u8, max_bytes: usize) !BufferedReader {
        const file = try fs.cwd().openFile(path, .{});
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = max_bytes,
            .total_read = 0,
        };
    }

    pub fn deinit(self: *BufferedReader) void {
        self.file.close();
    }

    fn fillBuffer(self: *BufferedReader) !bool {
        if (self.total_read >= self.max_read_bytes) return false;
        const remaining_allowed = self.max_read_bytes - self.total_read;
        const to_read = @min(self.buffer.len, remaining_allowed);
        const n = try self.file.read(self.buffer[0..to_read]);
        self.limit = n;
        self.pos = 0;
        self.total_read += n;
        return n > 0;
    }

    pub fn read(self: *BufferedReader, buf: []u8) !usize {
        var total: usize = 0;
        while (total < buf.len) {
            if (self.pos < self.limit) {
                const avail = @min(self.limit - self.pos, buf.len - total);
                @memcpy(buf[total..total + avail], self.buffer[self.pos..self.pos + avail]);
                self.pos += avail;
                total += avail;
            } else {
                if (!try self.fillBuffer()) break;
            }
        }
        return total;
    }

    pub fn readUntil(self: *BufferedReader, delim: u8, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();

        while (list.items.len < self.max_read_bytes) {
            if (self.pos < self.limit) {
                const chunk = self.buffer[self.pos..self.limit];
                if (mem.indexOfScalar(u8, chunk, delim)) |idx| {
                    try list.appendSlice(chunk[0..idx + 1]);
                    self.pos += idx + 1;
                    return list.toOwnedSlice();
                } else {
                    try list.appendSlice(chunk);
                    self.pos = self.limit;
                }
            } else {
                if (!try self.fillBuffer()) return list.toOwnedSlice();
            }
        }
        return IoError.MaxBytesExceeded;
    }

    pub fn readLine(self: *BufferedReader, allocator: Allocator) !?[]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();

        while (list.items.len < self.max_read_bytes) {
            if (self.pos < self.limit) {
                const chunk = self.buffer[self.pos..self.limit];
                if (mem.indexOfScalar(u8, chunk, '\n')) |idx| {
                    try list.appendSlice(chunk[0..idx]);
                    self.pos += idx + 1;
                    return list.toOwnedSlice();
                } else {
                    try list.appendSlice(chunk);
                    self.pos = self.limit;
                }
            } else {
                if (!try self.fillBuffer()) {
                    if (list.items.len == 0) return null;
                    return list.toOwnedSlice();
                }
            }
        }
        return IoError.MaxBytesExceeded;
    }

    pub fn peek(self: *BufferedReader) !?u8 {
        if (self.pos < self.limit) return self.buffer[self.pos];
        if (!try self.fillBuffer()) return null;
        if (self.pos < self.limit) return self.buffer[self.pos];
        return null;
    }
};

pub const BufferedWriter = struct {
    file: fs.File,
    buffer: []u8,
    pos: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, file: fs.File, buffer_size: usize) !BufferedWriter {
        const buffer = try allocator.alloc(u8, buffer_size);
        errdefer allocator.free(buffer);
        return .{
            .file = file,
            .buffer = buffer,
            .pos = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferedWriter) !void {
        try self.flush();
        self.allocator.free(self.buffer);
    }

    pub fn deinitNoError(self: *BufferedWriter) void {
        self.flush() catch {};
        self.allocator.free(self.buffer);
    }

    pub fn writeByte(self: *BufferedWriter, byte: u8) !void {
        if (self.pos >= self.buffer.len) {
            try self.flush();
        }
        self.buffer[self.pos] = byte;
        self.pos += 1;
    }

    pub fn writeBytes(self: *BufferedWriter, data: []const u8) !void {
        var remaining = data;
        while (remaining.len > 0) {
            if (self.pos >= self.buffer.len) {
                try self.flush();
            }
            const available = self.buffer.len - self.pos;
            const to_write = @min(available, remaining.len);
            @memcpy(self.buffer[self.pos..self.pos + to_write], remaining[0..to_write]);
            self.pos += to_write;
            remaining = remaining[to_write..];
        }
    }

    pub fn flush(self: *BufferedWriter) !void {
        if (self.pos > 0) {
            try self.file.writeAll(self.buffer[0..self.pos]);
            self.pos = 0;
        }
    }
};

pub fn stableHash(data: []const u8, seed: u64) u64 {
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return mixHash(hasher.final());
}

var g_hash_seed: u64 = 0;
var g_hash_seed_initialized: bool = false;
var g_hash_seed_mutex: std.Thread.Mutex = .{};

fn getHashSeed() u64 {
    g_hash_seed_mutex.lock();
    defer g_hash_seed_mutex.unlock();
    if (!g_hash_seed_initialized) {
        g_hash_seed = generateRuntimeSeed();
        g_hash_seed_initialized = true;
    }
    return g_hash_seed;
}

pub fn hash64(data: []const u8) u64 {
    const seed = getHashSeed();
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return mixHash(hasher.final());
}

pub fn hash32(data: []const u8) u32 {
    const h64 = hash64(data);
    const mixed = h64 ^ (h64 >> IoConfig.TRUNCATE_SHIFT);
    return @truncate(mixed);
}

pub fn pathJoin(allocator: Allocator, parts: []const []const u8) ![]u8 {
    return std.fs.path.join(allocator, parts);
}

pub fn pathExists(path: []const u8) bool {
    _ = fs.cwd().statFile(path) catch return false;
    return true;
}

pub fn pathExistsWithAccess(path: []const u8) !bool {
    _ = fs.cwd().statFile(path) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    return true;
}

pub fn createDirRecursive(path: []const u8) !void {
    if (path.len == 0) return;
    try fs.cwd().makePath(path);
}

pub fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
    return readFileLimited(allocator, path, IoConfig.MAX_FILE_SIZE);
}

pub fn readFileWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8) ![]u8 {
    const file = try dir.openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    if (size_u64 > IoConfig.MAX_FILE_SIZE) return error.FileTooLarge;
    const size: usize = @intCast(size_u64);
    if (size == 0) {
        return allocator.alloc(u8, 0);
    }
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != size) return IoError.UnexpectedEndOfFile;
    return buf;
}

pub fn readFileLimited(allocator: Allocator, path: []const u8, max_size: usize) ![]u8 {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    const size: usize = @intCast(size_u64);
    if (size > max_size) return error.FileTooLarge;
    if (size == 0) {
        return allocator.alloc(u8, 0);
    }
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != size) return IoError.UnexpectedEndOfFile;
    return buf;
}

pub const WriteFileOptions = struct {
    create_backup: bool = false,
    sync_after_write: bool = false,
};

pub fn writeFile(path: []const u8, data: []const u8) !void {
    return writeFileWithOptions(path, data, .{});
}

pub fn writeFileWithOptions(path: []const u8, data: []const u8, options: WriteFileOptions) !void {
    if (options.create_backup) {
        const exists = fs.cwd().statFile(path) catch |err| {
            if (err != error.FileNotFound) return err;
            null;
        };
        if (exists != null) {
            var backup_buf: [IoConfig.MAX_PATH_LEN]u8 = undefined;
            const backup_path = std.fmt.bufPrint(&backup_buf, "{s}.bak", .{path}) catch return IoError.PathTooLong;
            fs.cwd().copyFile(path, fs.cwd(), backup_path, .{}) catch |copy_err| {
                return copy_err;
            };
        }
    }
    const file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    defer file.close();
    try file.writeAll(data);
    if (options.sync_after_write) try file.sync();
}

pub fn appendFile(path: []const u8, data: []const u8) !void {
    const file = fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| {
        if (err == error.FileNotFound) {
            const new_file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
            defer new_file.close();
            try new_file.writeAll(data);
            return;
        }
        return err;
    };
    defer file.close();
    try file.seekFromEnd(0);
    try file.writeAll(data);
}

pub fn deleteFile(path: []const u8) !void {
    const stat_result = try fs.cwd().statFile(path);
    if (stat_result.kind == .directory) {
        return fs.cwd().deleteTree(path);
    }
    try fs.cwd().deleteFile(path);
}

pub const CopyProgress = struct {
    bytes_copied: u64,
    total_bytes: u64,
};

pub fn copyFile(allocator: Allocator, src: []const u8, dst: []const u8) !void {
    return copyFileWithProgress(allocator, src, dst, null);
}

pub fn copyFileWithProgress(
    allocator: Allocator,
    src: []const u8,
    dst: []const u8,
    progress_callback: ?*const fn(CopyProgress) void
) !void {
    const src_file = try fs.cwd().openFile(src, .{});
    defer src_file.close();

    const dst_file = try fs.cwd().createFile(dst, .{ .mode = IoConfig.SECURE_FILE_MODE });
    errdefer dst_file.close();

    const stat = try src_file.stat();
    const total_size: u64 = stat.size;

    const buffer = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buffer);

    var bytes_copied: u64 = 0;
    while (true) {
        const n = try src_file.read(buffer);
        if (n == 0) break;
        try dst_file.writeAll(buffer[0..n]);
        bytes_copied += n;
        if (progress_callback) |cb| {
            cb(.{ .bytes_copied = bytes_copied, .total_bytes = total_size });
        }
    }

    try dst_file.sync();
    dst_file.close();
}

pub fn moveFile(allocator: Allocator, old: []const u8, new: []const u8) !void {
    fs.cwd().rename(old, new) catch |err| {
        if (err == error.RenameAcrossMountPoints or err == error.NotSameFileSystem) {
            try copyFile(allocator, old, new);
            try fs.cwd().deleteFile(old);
            return;
        }
        return err;
    };
}

pub fn listDir(allocator: Allocator, path: []const u8) ![][]u8 {
    var dir = try fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var list = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (list.items) |item| allocator.free(item);
        list.deinit();
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        try list.append(name);
    }
    return list.toOwnedSlice();
}

pub fn createDir(path: []const u8) !void {
    try createDirRecursive(path);
}

pub fn removeDir(path: []const u8) !void {
    const stat_result = try fs.cwd().statFile(path);
    if (stat_result.kind == .sym_link) {
        try fs.cwd().deleteFile(path);
        return;
    }
    try fs.cwd().deleteTree(path);
}

pub fn removeFile(path: []const u8) !void {
    try fs.cwd().deleteFile(path);
}

pub fn renameFile(old: []const u8, new: []const u8) !void {
    try fs.cwd().rename(old, new);
}

pub fn getFileSize(path: []const u8) !usize {
    const stat = try fs.cwd().statFile(path);
    const size_u64: u64 = stat.size;
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    return @intCast(size_u64);
}

pub fn isDir(path: []const u8) bool {
    const stat_result = fs.cwd().statFile(path) catch return false;
    return stat_result.kind == .directory;
}

pub fn isFile(path: []const u8) bool {
    const stat_result = fs.cwd().statFile(path) catch return false;
    return stat_result.kind == .file;
}

pub inline fn toLittleEndian(comptime T: type, value: T) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .Int) @compileError("toLittleEndian requires integer type");
    }
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .Little => value,
        .Big => @byteSwap(value),
    };
}

pub inline fn fromLittleEndian(comptime T: type, bytes: *const [@sizeOf(T)]u8) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .Int) @compileError("fromLittleEndian requires integer type");
    }
    return mem.readInt(T, bytes, .Little);
}

pub inline fn toBigEndian(comptime T: type, value: T) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .Int) @compileError("toBigEndian requires integer type");
    }
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .Little => @byteSwap(value),
        .Big => value,
    };
}

pub inline fn fromBigEndian(comptime T: type, bytes: *const [@sizeOf(T)]u8) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .Int) @compileError("fromBigEndian requires integer type");
    }
    return mem.readInt(T, bytes, .Big);
}

pub fn sequentialWrite(allocator: Allocator, path: []const u8, data: []const []const u8) !void {
    const file = try fs.cwd().createFile(path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    defer file.close();

    var writer = try BufferedWriter.init(allocator, file, IoConfig.LARGE_CHUNK_SIZE);
    defer writer.deinitNoError();

    for (data) |chunk| {
        try writer.writeBytes(chunk);
    }
    try writer.flush();
    try file.sync();
}

pub fn sequentialRead(allocator: Allocator, path: []const u8, chunk_callback: *const fn([]const u8) anyerror!void) !void {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    const buffer = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buffer);

    while (true) {
        const n = try file.read(buffer);
        if (n == 0) break;
        try chunk_callback(buffer[0..n]);
    }
}

pub fn atomicWrite(allocator: Allocator, path: []const u8, data: []const u8) !void {
    _ = allocator;
    var temp_buf: [IoConfig.MAX_PATH_LEN]u8 = undefined;
    const temp_path = std.fmt.bufPrint(&temp_buf, "{s}.tmp.{d}", .{path, generateRuntimeSeed()}) catch return IoError.PathTooLong;

    const file = try fs.cwd().createFile(temp_path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    errdefer {
        file.close();
        fs.cwd().deleteFile(temp_path) catch {};
    }

    try file.writeAll(data);
    try file.sync();
    file.close();

    try fs.cwd().rename(temp_path, path);
}

pub const FileCompareResult = enum {
    equal,
    different,
    first_not_found,
    second_not_found,
    both_not_found,
    read_error,
};

pub fn compareFiles(allocator: Allocator, path1: []const u8, path2: []const u8) FileCompareResult {
    const file1 = fs.cwd().openFile(path1, .{}) catch |err| {
        if (err == error.FileNotFound) {
            const file2 = fs.cwd().openFile(path2, .{}) catch |err2| {
                if (err2 == error.FileNotFound) return .both_not_found;
                return .read_error;
            };
            file2.close();
            return .first_not_found;
        }
        return .read_error;
    };
    defer file1.close();

    const file2 = fs.cwd().openFile(path2, .{}) catch |err| {
        if (err == error.FileNotFound) return .second_not_found;
        return .read_error;
    };
    defer file2.close();

    const stat1 = file1.stat() catch return .read_error;
    const stat2 = file2.stat() catch return .read_error;
    if (stat1.size != stat2.size) return .different;

    const buf1 = allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE) catch return .read_error;
    defer allocator.free(buf1);
    const buf2 = allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE) catch return .read_error;
    defer allocator.free(buf2);

    while (true) {
        const n1 = file1.read(buf1) catch return .read_error;
        const n2 = file2.read(buf2) catch return .read_error;
        if (n1 != n2) return .different;
        if (n1 == 0) break;
        if (!mem.eql(u8, buf1[0..n1], buf2[0..n2])) return .different;
    }

    return .equal;
}

pub fn compareFilesEqual(allocator: Allocator, path1: []const u8, path2: []const u8) bool {
    return compareFiles(allocator, path1, path2) == .equal;
}

test "MMAP open and close" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_mmap.bin", .{});
    try file.writeAll("test data for mmap");
    file.close();

    var mmap = try MMAP.openWithDir(gpa, tmp_dir.dir, "test_mmap.bin", .{ .mode = .read_only });
    defer mmap.close();

    const content = try mmap.read(0, 9);
    try std.testing.expectEqualStrings("test data", content);
}

test "DurableWriter with sync" {
    var gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    var writer = try DurableWriter.initWithDir(tmp_dir.dir, "test_durable.txt", false);
    try writer.writeAll("hello world");
    try writer.deinit();

    const content = try readFileWithDir(gpa, tmp_dir.dir, "test_durable.txt");
    defer gpa.free(content);
    try std.testing.expectEqualStrings("hello world", content);
}

test "BufferedReader zero init" {
    var gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_buffered.txt", .{});
    try file.writeAll("line1\nline2\nline3");
    file.close();

    var reader = try BufferedReader.initWithDir(tmp_dir.dir, "test_buffered.txt");
    defer reader.deinit();

    const line1 = try reader.readUntil('\n', gpa);
    defer gpa.free(line1);
    try std.testing.expectEqualStrings("line1\n", line1);

    const line2 = try reader.readUntil('\n', gpa);
    defer gpa.free(line2);
    try std.testing.expectEqualStrings("line2\n", line2);

    const line3 = try reader.readUntil('\n', gpa);
    defer gpa.free(line3);
    try std.testing.expectEqualStrings("line3", line3);
}

test "Stable hash mixing" {
    const data = "test";
    const seed: u64 = 12345;
    const hash1 = stableHash(data, seed);
    const hash2 = stableHash(data, seed);
    const hash3 = stableHash(data, 67890);

    try std.testing.expectEqual(hash1, hash2);
    try std.testing.expect(hash1 != hash3);
}

test "Path join" {
    var gpa = std.testing.allocator;
    const path1 = try pathJoin(gpa, &.{ "a", "b", "c" });
    defer gpa.free(path1);
    try std.testing.expectEqualStrings("a/b/c", path1);
}

test "Atomic write" {
    var gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    var path_buf: [256]u8 = undefined;
    const full_path = try tmp_dir.dir.realpath(".", &path_buf);
    const test_path = try std.fmt.allocPrint(gpa, "{s}/test_atomic.txt", .{full_path});
    defer gpa.free(test_path);

    try atomicWrite(gpa, test_path, "data");
    defer fs.cwd().deleteFile(test_path) catch {};
    const content = try readFile(gpa, test_path);
    defer gpa.free(content);
    try std.testing.expectEqualStrings("data", content);
}
