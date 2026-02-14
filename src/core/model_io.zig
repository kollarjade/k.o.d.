const std = @import("std");
const mem = std.mem;
const fs = std.fs;
const Allocator = mem.Allocator;
const RSF = @import("../processor/rsf.zig").RSF;
const Ranker = @import("../ranker/ranker.zig").Ranker;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const Tensor = @import("tensor.zig").Tensor;

pub const ModelError = error{
    InvalidMagicHeader,
    UnsupportedVersion,
    CorruptedData,
    ChecksumMismatch,
    InvalidMetadata,
    MissingComponent,
    InvalidUtf8,
};

pub const MAGIC_HEADER = "JAIDE40\x00";
pub const CURRENT_VERSION: u32 = 1;

const MAX_METADATA_SIZE = 10 * 1024 * 1024;
const MAX_COMPONENT_SIZE = std.math.maxInt(u32);

pub const ModelMetadata = struct {
    model_name: []const u8,
    version: u32,
    created_timestamp: i64,
    rsf_layers: usize,
    rsf_dim: usize,
    ranker_ngrams: usize,
    ranker_lsh_tables: usize,
    mgt_vocab_size: usize,
    description: []const u8,

    fn writeJsonEscapedString(writer: anytype, str: []const u8) !void {
        try writer.writeByte('"');
        for (str) |c| {
            switch (c) {
                '"' => try writer.writeAll("\\\""),
                '\\' => try writer.writeAll("\\\\"),
                '\n' => try writer.writeAll("\\n"),
                '\r' => try writer.writeAll("\\r"),
                '\t' => try writer.writeAll("\\t"),
                0x00...0x1F => {
                    try writer.print("\\u{x:0>4}", .{c});
                },
                else => try writer.writeByte(c),
            }
        }
        try writer.writeByte('"');
    }

    pub fn toJson(self: *const ModelMetadata, allocator: Allocator) ![]u8 {
        if (!std.unicode.utf8ValidateSlice(self.model_name) or !std.unicode.utf8ValidateSlice(self.description)) {
            return error.InvalidUtf8;
        }

        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{\"model_name\":");
        try writeJsonEscapedString(writer, self.model_name);
        try writer.print(",\"version\":{},", .{self.version});
        try writer.print("\"created_timestamp\":{},", .{self.created_timestamp});
        try writer.print("\"rsf_layers\":{},", .{self.rsf_layers});
        try writer.print("\"rsf_dim\":{},", .{self.rsf_dim});
        try writer.print("\"ranker_ngrams\":{},", .{self.ranker_ngrams});
        try writer.print("\"ranker_lsh_tables\":{},", .{self.ranker_lsh_tables});
        try writer.print("\"mgt_vocab_size\":{},", .{self.mgt_vocab_size});
        try writer.writeAll("\"description\":");
        try writeJsonEscapedString(writer, self.description);
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }

    pub fn fromJson(allocator: Allocator, json: []const u8) !ModelMetadata {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return ModelError.InvalidMetadata;
        const obj = root.object;

        const model_name_val = obj.get("model_name") orelse return ModelError.InvalidMetadata;
        const version_val = obj.get("version") orelse return ModelError.InvalidMetadata;
        const created_val = obj.get("created_timestamp") orelse return ModelError.InvalidMetadata;
        const rsf_layers_val = obj.get("rsf_layers") orelse return ModelError.InvalidMetadata;
        const rsf_dim_val = obj.get("rsf_dim") orelse return ModelError.InvalidMetadata;
        const ranker_ngrams_val = obj.get("ranker_ngrams") orelse return ModelError.InvalidMetadata;
        const ranker_lsh_val = obj.get("ranker_lsh_tables") orelse return ModelError.InvalidMetadata;
        const mgt_vocab_val = obj.get("mgt_vocab_size") orelse return ModelError.InvalidMetadata;
        const description_val = obj.get("description") orelse return ModelError.InvalidMetadata;

        const model_name = switch (model_name_val) { .string => |s| s, else => return ModelError.InvalidMetadata };
        const version = switch (version_val) { .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @as(u32, @intCast(i)) else return ModelError.InvalidMetadata, else => return ModelError.InvalidMetadata };
        const created = switch (created_val) { .integer => |i| i, else => return ModelError.InvalidMetadata };
        const rsf_layers_raw = switch (rsf_layers_val) { .integer => |i| if (i >= 0 and i <= std.math.maxInt(usize)) i else return ModelError.InvalidMetadata, else => return ModelError.InvalidMetadata };
        const rsf_dim_raw = switch (rsf_dim_val) { .integer => |i| if (i >= 0 and i <= std.math.maxInt(usize)) i else return ModelError.InvalidMetadata, else => return ModelError.InvalidMetadata };
        const ranker_ngrams_raw = switch (ranker_ngrams_val) { .integer => |i| if (i >= 0 and i <= std.math.maxInt(usize)) i else return ModelError.InvalidMetadata, else => return ModelError.InvalidMetadata };
        const ranker_lsh_raw = switch (ranker_lsh_val) { .integer => |i| if (i >= 0 and i <= std.math.maxInt(usize)) i else return ModelError.InvalidMetadata, else => return ModelError.InvalidMetadata };
        const mgt_vocab_raw = switch (mgt_vocab_val) { .integer => |i| if (i >= 0 and i <= std.math.maxInt(usize)) i else return ModelError.InvalidMetadata, else => return ModelError.InvalidMetadata };
        const description = switch (description_val) { .string => |s| s, else => return ModelError.InvalidMetadata };

        const model_name_duped = try allocator.dupe(u8, model_name);
        errdefer allocator.free(model_name_duped);
        const description_duped = try allocator.dupe(u8, description);
        errdefer allocator.free(description_duped);

        return ModelMetadata{
            .model_name = model_name_duped,
            .version = version,
            .created_timestamp = created,
            .rsf_layers = @intCast(rsf_layers_raw),
            .rsf_dim = @intCast(rsf_dim_raw),
            .ranker_ngrams = @intCast(ranker_ngrams_raw),
            .ranker_lsh_tables = @intCast(ranker_lsh_raw),
            .mgt_vocab_size = @intCast(mgt_vocab_raw),
            .description = description_duped,
        };
    }

    pub fn deinit(self: *ModelMetadata, allocator: Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.description);
    }
};

pub const ModelFormat = struct {
    metadata: ModelMetadata,
    rsf: ?*RSF = null,
    ranker: ?*Ranker = null,
    mgt: ?*MGT = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, description: []const u8) !ModelFormat {
        const timestamp = std.time.timestamp();
        const name_duped = try allocator.dupe(u8, name);
        errdefer allocator.free(name_duped);
        const desc_duped = try allocator.dupe(u8, description);

        return ModelFormat{
            .metadata = ModelMetadata{
                .model_name = name_duped,
                .version = CURRENT_VERSION,
                .created_timestamp = timestamp,
                .rsf_layers = 0,
                .rsf_dim = 0,
                .ranker_ngrams = 0,
                .ranker_lsh_tables = 0,
                .mgt_vocab_size = 0,
                .description = desc_duped,
            },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ModelFormat) void {
        if (self.rsf) |rsf| {
            rsf.deinit();
            self.allocator.destroy(rsf);
        }
        if (self.ranker) |ranker| {
            ranker.deinit();
            self.allocator.destroy(ranker);
        }
        if (self.mgt) |mgt| {
            mgt.deinit();
            self.allocator.destroy(mgt);
        }
        self.metadata.deinit(self.allocator);
    }

    pub fn setRSF(self: *ModelFormat, rsf: *RSF) void {
        self.rsf = rsf;
        self.metadata.rsf_layers = rsf.num_layers;
        self.metadata.rsf_dim = rsf.dim;
    }

    pub fn setRanker(self: *ModelFormat, ranker: *Ranker) void {
        self.ranker = ranker;
        self.metadata.ranker_ngrams = ranker.num_ngrams;
        self.metadata.ranker_lsh_tables = ranker.num_hash_functions;
    }

    pub fn setMGT(self: *ModelFormat, mgt: *MGT) void {
        self.mgt = mgt;
        self.metadata.mgt_vocab_size = mgt.vocabSize();
    }
};

fn hashIntLittleEndian(comptime T: type, hasher: *std.crypto.hash.sha2.Sha256, value: T) void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    mem.writeInt(T, &bytes, value, .Little);
    hasher.update(bytes[0..]);
}

fn writeHashIntLittleEndian(comptime T: type, writer: anytype, hasher: *std.crypto.hash.sha2.Sha256, value: T) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    mem.writeInt(T, &bytes, value, .Little);
    hasher.update(bytes[0..]);
    try writer.writeAll(&bytes);
}

pub fn exportModel(model: *ModelFormat, path: []const u8) !void {
    var file = try fs.cwd().createFile(path, .{});
    defer file.close();

    var buffered_writer = std.io.bufferedWriter(file.writer());
    const writer = buffered_writer.writer();

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});

    try writer.writeAll(MAGIC_HEADER);
    hasher.update(MAGIC_HEADER);

    try writeHashIntLittleEndian(u32, writer, &hasher, CURRENT_VERSION);

    if (model.rsf) |rsf| {
        model.metadata.rsf_layers = rsf.num_layers;
        model.metadata.rsf_dim = rsf.dim;
    }
    if (model.ranker) |ranker| {
        model.metadata.ranker_ngrams = ranker.num_ngrams;
        model.metadata.ranker_lsh_tables = ranker.num_hash_functions;
    }
    if (model.mgt) |mgt| {
        model.metadata.mgt_vocab_size = mgt.vocabSize();
    }

    const metadata_json = try model.metadata.toJson(model.allocator);
    defer model.allocator.free(metadata_json);

    if (metadata_json.len > std.math.maxInt(u32)) return ModelError.CorruptedData;
    const metadata_len: u32 = @intCast(metadata_json.len);
    try writeHashIntLittleEndian(u32, writer, &hasher, metadata_len);
    try writer.writeAll(metadata_json);
    hasher.update(metadata_json);

    if (model.rsf) |rsf| {
        try writer.writeByte(1);
        hasher.update(&.{1});

        var rsf_buf = std.ArrayList(u8).init(model.allocator);
        defer rsf_buf.deinit();
        var rsf_writer = rsf_buf.writer();

        const num_layers_u64: u64 = @intCast(rsf.num_layers);
        const dim_u64: u64 = @intCast(rsf.dim);

        try rsf_writer.writeInt(u64, num_layers_u64, .Little);
        try rsf_writer.writeInt(u64, dim_u64, .Little);

        for (rsf.layers) |layer| {
            try layer.s_weight.save(rsf_writer);
            try layer.t_weight.save(rsf_writer);
            try layer.s_bias.save(rsf_writer);
            try layer.t_bias.save(rsf_writer);
        }

        const rsf_data = try rsf_buf.toOwnedSlice();
        defer model.allocator.free(rsf_data);

        if (rsf_data.len > MAX_COMPONENT_SIZE) return ModelError.CorruptedData;
        const rsf_len: u32 = @intCast(rsf_data.len);
        try writeHashIntLittleEndian(u32, writer, &hasher, rsf_len);
        try writer.writeAll(rsf_data);
        hasher.update(rsf_data);
    } else {
        try writer.writeByte(0);
        hasher.update(&.{0});
    }

    if (model.ranker) |ranker| {
        try writer.writeByte(1);
        hasher.update(&.{1});

        var ranker_buf = std.ArrayList(u8).init(model.allocator);
        defer ranker_buf.deinit();
        var ranker_writer = ranker_buf.writer();

        try ranker_writer.writeByte(1);

        const num_weights_u64: u64 = @intCast(ranker.ngram_weights.len);
        try ranker_writer.writeInt(u64, num_weights_u64, .Little);

        for (ranker.ngram_weights) |w| {
            const w_bits: u32 = @bitCast(w);
            try ranker_writer.writeInt(u32, w_bits, .Little);
        }

        const num_hash_funcs_u64: u64 = @intCast(ranker.num_hash_functions);
        try ranker_writer.writeInt(u64, num_hash_funcs_u64, .Little);

        if (ranker.lsh_hash_params.len != ranker.num_hash_functions * 2) return ModelError.CorruptedData;
        for (ranker.lsh_hash_params) |t| {
            try ranker_writer.writeInt(u64, t, .Little);
        }

        try ranker_writer.writeInt(u64, ranker.seed, .Little);

        const ranker_data = try ranker_buf.toOwnedSlice();
        defer model.allocator.free(ranker_data);

        if (ranker_data.len > MAX_COMPONENT_SIZE) return ModelError.CorruptedData;
        const ranker_len: u32 = @intCast(ranker_data.len);
        try writeHashIntLittleEndian(u32, writer, &hasher, ranker_len);
        try writer.writeAll(ranker_data);
        hasher.update(ranker_data);
    } else {
        try writer.writeByte(0);
        hasher.update(&.{0});
    }

    if (model.mgt) |mgt| {
        try writer.writeByte(1);
        hasher.update(&.{1});

        var mgt_buf = std.ArrayList(u8).init(model.allocator);
        defer mgt_buf.deinit();
        var mgt_writer = mgt_buf.writer();

        const vocab_size_u32: u32 = @intCast(mgt.vocabSize());
        try mgt_writer.writeInt(u32, vocab_size_u32, .Little);

        var keys = try model.allocator.alloc([]const u8, mgt.token_to_id.count());
        defer model.allocator.free(keys);
        {
            var i: usize = 0;
            var it = mgt.token_to_id.iterator();
            while (it.next()) |entry| {
                keys[i] = entry.key_ptr.*;
                i += 1;
            }
        }
        std.sort.heap([]const u8, keys, {}, struct {
            fn compare(ctx: void, a: []const u8, b: []const u8) bool {
                _ = ctx;
                return std.mem.order(u8, a, b) == .lt;
            }
        }.compare);

        for (keys) |word| {
            const word_len: u32 = @intCast(word.len);
            try mgt_writer.writeInt(u32, word_len, .Little);
            try mgt_writer.writeAll(word);
        }

        const mgt_data = try mgt_buf.toOwnedSlice();
        defer model.allocator.free(mgt_data);

        if (mgt_data.len > MAX_COMPONENT_SIZE) return ModelError.CorruptedData;
        const mgt_len: u32 = @intCast(mgt_data.len);
        try writeHashIntLittleEndian(u32, writer, &hasher, mgt_len);
        try writer.writeAll(mgt_data);
        hasher.update(mgt_data);
    } else {
        try writer.writeByte(0);
        hasher.update(&.{0});
    }

    var checksum: [32]u8 = undefined;
    hasher.final(&checksum);
    try writer.writeAll(&checksum);

    try buffered_writer.flush();
}

pub fn importModel(path: []const u8, allocator: Allocator) !ModelFormat {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    const reader = buffered_reader.reader();

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});

    var magic: [MAGIC_HEADER.len]u8 = undefined;
    try reader.readNoEof(&magic);
    if (!mem.eql(u8, magic[0..], MAGIC_HEADER)) {
        return ModelError.InvalidMagicHeader;
    }
    hasher.update(magic[0..]);

    const version = try reader.readInt(u32, .Little);
    hashIntLittleEndian(u32, &hasher, version);

    if (version != CURRENT_VERSION) {
        return ModelError.UnsupportedVersion;
    }

    const metadata_len = try reader.readInt(u32, .Little);
    hashIntLittleEndian(u32, &hasher, metadata_len);

    if (metadata_len > MAX_METADATA_SIZE) return ModelError.CorruptedData;

    const metadata_json = try allocator.alloc(u8, metadata_len);
    defer allocator.free(metadata_json);
    try reader.readNoEof(metadata_json);
    hasher.update(metadata_json);

    var metadata = try ModelMetadata.fromJson(allocator, metadata_json);

    var model = ModelFormat{
        .metadata = metadata,
        .allocator = allocator,
    };
    errdefer model.deinit();

    const has_rsf = try reader.readByte();
    hasher.update(&.{has_rsf});

    if (has_rsf == 1) {
        const rsf_len = try reader.readInt(u32, .Little);
        hashIntLittleEndian(u32, &hasher, rsf_len);

        if (rsf_len > MAX_COMPONENT_SIZE) return ModelError.CorruptedData;

        const rsf_data = try allocator.alloc(u8, rsf_len);
        defer allocator.free(rsf_data);
        try reader.readNoEof(rsf_data);
        hasher.update(rsf_data);

        var rsf_stream = std.io.fixedBufferStream(rsf_data);
        var rsf_reader = rsf_stream.reader();

        const num_layers_u64 = try rsf_reader.readInt(u64, .Little);
        const dim_u64 = try rsf_reader.readInt(u64, .Little);

        if (num_layers_u64 > 1000000 or dim_u64 > 10000000 or num_layers_u64 > std.math.maxInt(usize) or dim_u64 > std.math.maxInt(usize)) return ModelError.CorruptedData;

        const num_layers = @as(usize, @intCast(num_layers_u64));
        const dim = @as(usize, @intCast(dim_u64));

        var rsf = try allocator.create(RSF);
        errdefer allocator.destroy(rsf);
        rsf.* = try RSF.init(allocator, dim, num_layers);
        errdefer rsf.deinit();

        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            rsf.layers[l].s_weight.deinit();
            rsf.layers[l].t_weight.deinit();
            rsf.layers[l].s_bias.deinit();
            rsf.layers[l].t_bias.deinit();

            rsf.layers[l].s_weight = try Tensor.load(allocator, rsf_reader);
            rsf.layers[l].t_weight = try Tensor.load(allocator, rsf_reader);
            rsf.layers[l].s_bias = try Tensor.load(allocator, rsf_reader);
            rsf.layers[l].t_bias = try Tensor.load(allocator, rsf_reader);
        }

        if (rsf_stream.getPos() != rsf_data.len) return ModelError.CorruptedData;

        model.rsf = rsf;
        model.metadata.rsf_layers = num_layers;
        model.metadata.rsf_dim = dim;
    } else if (has_rsf != 0) {
        return ModelError.CorruptedData;
    }

    const has_ranker = try reader.readByte();
    hasher.update(&.{has_ranker});

    if (has_ranker == 1) {
        const ranker_len = try reader.readInt(u32, .Little);
        hashIntLittleEndian(u32, &hasher, ranker_len);

        if (ranker_len > MAX_COMPONENT_SIZE) return ModelError.CorruptedData;

        const ranker_data = try allocator.alloc(u8, ranker_len);
        defer allocator.free(ranker_data);
        try reader.readNoEof(ranker_data);
        hasher.update(ranker_data);

        var ranker_stream = std.io.fixedBufferStream(ranker_data);
        var ranker_reader = ranker_stream.reader();

        const ranker_version = try ranker_reader.readByte();
        if (ranker_version != 1) return ModelError.UnsupportedVersion;

        const num_weights_u64 = try ranker_reader.readInt(u64, .Little);
        if (num_weights_u64 > 1000000000 or num_weights_u64 > std.math.maxInt(usize)) return ModelError.CorruptedData;

        var ranker = try allocator.create(Ranker);
        errdefer allocator.destroy(ranker);

        const num_weights = @as(usize, @intCast(num_weights_u64));
        const ngram_weights_alloc = try allocator.alloc(f32, num_weights);
        errdefer allocator.free(ngram_weights_alloc);

        @memset(ngram_weights_alloc, 0);

        for (ngram_weights_alloc) |*w| {
            const w_int = try ranker_reader.readInt(u32, .Little);
            w.* = @bitCast(w_int);
        }

        const num_hash_funcs_u64 = try ranker_reader.readInt(u64, .Little);
        if (num_hash_funcs_u64 > 1000 or num_hash_funcs_u64 > std.math.maxInt(usize)) return ModelError.CorruptedData;
        const num_hash_funcs = @as(usize, @intCast(num_hash_funcs_u64));

        const lsh_hash_params = try allocator.alloc(u64, num_hash_funcs * 2);
        errdefer allocator.free(lsh_hash_params);

        for (lsh_hash_params) |*t| {
            t.* = try ranker_reader.readInt(u64, .Little);
        }

        const seed = try ranker_reader.readInt(u64, .Little);

        ranker.* = Ranker{
            .ngram_weights = ngram_weights_alloc,
            .lsh_hash_params = lsh_hash_params,
            .num_hash_functions = num_hash_funcs,
            .num_ngrams = num_weights,
            .seed = seed,
            .allocator = allocator,
        };

        if (ranker_stream.getPos() != ranker_data.len) {
            allocator.free(ngram_weights_alloc);
            allocator.free(lsh_hash_params);
            return ModelError.CorruptedData;
        }

        model.ranker = ranker;
        model.metadata.ranker_ngrams = num_weights;
        model.metadata.ranker_lsh_tables = num_hash_funcs;
    } else if (has_ranker != 0) {
        return ModelError.CorruptedData;
    }

    const has_mgt = try reader.readByte();
    hasher.update(&.{has_mgt});

    if (has_mgt == 1) {
        const mgt_len = try reader.readInt(u32, .Little);
        hashIntLittleEndian(u32, &hasher, mgt_len);

        if (mgt_len > MAX_COMPONENT_SIZE) return ModelError.CorruptedData;

        const mgt_data = try allocator.alloc(u8, mgt_len);
        defer allocator.free(mgt_data);
        try reader.readNoEof(mgt_data);
        hasher.update(mgt_data);

        var mgt_stream = std.io.fixedBufferStream(mgt_data);
        var mgt_reader = mgt_stream.reader();

        const vocab_size = try mgt_reader.readInt(u32, .Little);
        if (vocab_size > 10000000) return ModelError.CorruptedData;

        var words_list = std.ArrayList([]u8).init(allocator);
        errdefer {
            for (words_list.items) |w| allocator.free(w);
            words_list.deinit();
        }

        var i: u32 = 0;
        while (i < vocab_size) : (i += 1) {
            const word_len = try mgt_reader.readInt(u32, .Little);
            if (word_len > 1024 * 1024) return ModelError.CorruptedData;
            var word = try allocator.alloc(u8, word_len);
            try mgt_reader.readNoEof(word);
            try words_list.append(word);
        }

        const words_const = try allocator.alloc([]const u8, words_list.items.len);
        errdefer allocator.free(words_const);

        var idx: usize = 0;
        while (idx < words_list.items.len) : (idx += 1) {
            words_const[idx] = words_list.items[idx];
        }

        var mgt = try allocator.create(MGT);
        errdefer allocator.destroy(mgt);
        mgt.* = try MGT.init(allocator, words_const, &.{});

        for (words_list.items) |w| {
            allocator.free(w);
        }
        words_list.deinit();
        allocator.free(words_const);

        if (mgt_stream.getPos() != mgt_data.len) {
            mgt.deinit();
            return ModelError.CorruptedData;
        }

        model.mgt = mgt;
        model.metadata.mgt_vocab_size = @intCast(vocab_size);
    } else if (has_mgt != 0) {
        return ModelError.CorruptedData;
    }

    var expected_checksum: [32]u8 = undefined;
    hasher.final(&expected_checksum);

    var stored_checksum: [32]u8 = undefined;
    try reader.readNoEof(&stored_checksum);

    if (!constantTimeCompare(expected_checksum[0..], stored_checksum[0..])) {
        return ModelError.ChecksumMismatch;
    }

    const end_byte = reader.readByte() catch |err| {
        if (err == error.EndOfStream) return model;
        return err;
    };
    _ = end_byte;
    return ModelError.CorruptedData;
}

fn constantTimeCompare(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

test "ModelFormat creation and metadata" {
    const testing = std.testing;
    var gpa = testing.allocator;

    var model = try ModelFormat.init(gpa, "TestModel", "A test model");
    defer model.deinit();

    try testing.expectEqualStrings("TestModel", model.metadata.model_name);
    try testing.expectEqualStrings("A test model", model.metadata.description);
    try testing.expectEqual(CURRENT_VERSION, model.metadata.version);
}

test "Metadata JSON serialization" {
    const testing = std.testing;
    var gpa = testing.allocator;

    var metadata = ModelMetadata{
        .model_name = try gpa.dupe(u8, "Test"),
        .version = 1,
        .created_timestamp = 1234567890,
        .rsf_layers = 4,
        .rsf_dim = 128,
        .ranker_ngrams = 5,
        .ranker_lsh_tables = 8,
        .mgt_vocab_size = 1000,
        .description = try gpa.dupe(u8, "Test model"),
    };
    defer metadata.deinit(gpa);

    const json = try metadata.toJson(gpa);
    defer gpa.free(json);

    try testing.expect(json.len > 0);

    const parsed_metadata = try ModelMetadata.fromJson(gpa, json);
    defer parsed_metadata.deinit(gpa);

    try testing.expectEqualStrings(metadata.model_name, parsed_metadata.model_name);
    try testing.expectEqual(metadata.version, parsed_metadata.version);
    try testing.expectEqual(metadata.created_timestamp, parsed_metadata.created_timestamp);
    try testing.expectEqual(metadata.rsf_layers, parsed_metadata.rsf_layers);
}