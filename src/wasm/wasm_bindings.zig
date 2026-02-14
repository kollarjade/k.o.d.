const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const deps = @import("../wasm_deps.zig");
const MGT = deps.MGT;
const RSF = deps.RSF;
const Tensor = deps.Tensor;
const ModelFormat = deps.ModelFormat;
const importModel = deps.importModel;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

var global_model: ?ModelFormat = null;
var global_mgt: ?MGT = null;
var global_rsf: ?RSF = null;
var global_state_mutex: std.Thread.Mutex = .{};

pub export fn wasmAlloc(size: usize) ?[*]u8 {
    const slice = allocator.alloc(u8, size) catch return null;
    return slice.ptr;
}

pub export fn wasmFree(ptr: [*]u8, size: usize) void {
    const slice = ptr[0..size];
    allocator.free(slice);
}

pub export fn wasmGetMemory() usize {
    return @intFromPtr(&gpa);
}

pub export fn wasmInitModel(vocab_ptr: [*]const u8, vocab_len: usize, vocab_size: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    global_mgt = null;

    const vocab_data = vocab_ptr[0..vocab_len];

    var vocab_list = std.ArrayList([]const u8).init(allocator);
    defer vocab_list.deinit();

    var i: usize = 0;
    var start: usize = 0;
    while (i < vocab_data.len) : (i += 1) {
        if (vocab_data[i] == '\n' or i == vocab_data.len - 1) {
            const end = if (i == vocab_data.len - 1) i + 1 else i;
            if (end > start) {
                const word = vocab_data[start..end];
                vocab_list.append(word) catch return -1;
            }
            start = i + 1;
        }
    }

    while (vocab_list.items.len < vocab_size) {
        vocab_list.append("") catch return -1;
    }

    global_mgt = MGT.init(allocator, vocab_list.items, &.{}) catch return -1;

    return 0;
}

pub export fn wasmInitRSF(layers: usize, dim: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    global_rsf = null;

    global_rsf = RSF.init(allocator, dim, layers) catch return -1;
    return 0;
}

pub export fn wasmLoadModel(path_ptr: [*]const u8, path_len: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_model) |*existing| {
        global_mgt = null;
        global_rsf = null;
        existing.deinit();
        global_model = null;
    }

    const path = path_ptr[0..path_len];

    const new_model = importModel(path, allocator) catch return -1;

    global_model = new_model;

    if (new_model.mgt) |mgt| {
        global_mgt = mgt;
    }

    if (new_model.rsf) |rsf| {
        global_rsf = rsf;
    }

    return 0;
}

pub export fn wasmEncode(text_ptr: [*]const u8, text_len: usize, out_ptr: [*]u32, max_tokens: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt) |*mgt| {
        const text = text_ptr[0..text_len];

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        mgt.encode(text, &tokens) catch return -1;

        const count = @min(tokens.items.len, max_tokens);
        @memcpy(out_ptr[0..count], tokens.items[0..count]);

        return @intCast(count);
    }

    return -1;
}

pub export fn wasmDecode(tokens_ptr: [*]const u32, tokens_len: usize, out_ptr: [*]u8, max_len: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt) |*mgt| {
        const tokens = tokens_ptr[0..tokens_len];

        var text = std.ArrayList(u8).init(allocator);
        defer text.deinit();

        mgt.decode(tokens, &text) catch return -1;

        const count = @min(text.items.len, max_len);
        @memcpy(out_ptr[0..count], text.items[0..count]);

        return @intCast(count);
    }

    return -1;
}

pub export fn wasmInference(text_ptr: [*]const u8, text_len: usize, tokens_out: [*]u32, embeddings_out: [*]f32, max_tokens: usize, max_embeddings: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt) |*mgt| {
        const text = text_ptr[0..text_len];

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        mgt.encode(text, &tokens) catch return -2;

        const token_count = @min(tokens.items.len, max_tokens);
        @memcpy(tokens_out[0..token_count], tokens.items[0..token_count]);

        if (global_rsf) |*rsf| {
            const dim = rsf.dim;
            const batch_size: usize = 1;

            var input_tensor = Tensor.init(allocator, &.{ batch_size, dim * 2 }) catch return -3;
            defer input_tensor.deinit();

            var i: usize = 0;
            while (i < input_tensor.data.len) : (i += 1) {
                input_tensor.data[i] = if (i < token_count)
                    @as(f32, @floatFromInt(tokens.items[i])) / 1000.0
                else
                    0.0;
            }

            rsf.forward(&input_tensor) catch return -4;

            const emb_count = @min(input_tensor.data.len, max_embeddings);
            @memcpy(embeddings_out[0..emb_count], input_tensor.data[0..emb_count]);

            return @intCast((token_count << 16) | emb_count);
        }

        return @intCast(token_count);
    }

    return -1;
}

pub export fn wasmGetEmbeddings(tokens_ptr: [*]const u32, tokens_len: usize, out_ptr: [*]f32, max_len: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_rsf) |*rsf| {
        const tokens = tokens_ptr[0..tokens_len];

        const dim = rsf.dim;
        const batch_size: usize = 1;

        var input_tensor = Tensor.init(allocator, &.{ batch_size, dim * 2 }) catch return -2;
        defer input_tensor.deinit();

        var i: usize = 0;
        while (i < input_tensor.data.len) : (i += 1) {
            input_tensor.data[i] = if (i < tokens.len)
                @as(f32, @floatFromInt(tokens[i])) / 1000.0
            else
                0.0;
        }

        rsf.forward(&input_tensor) catch return -3;

        const count = @min(input_tensor.data.len, max_len);
        @memcpy(out_ptr[0..count], input_tensor.data[0..count]);

        return @intCast(count);
    }

    return -1;
}

pub export fn wasmBatchEncode(texts_ptr: [*]const u8, texts_len: usize, text_offsets: [*]const usize, num_texts: usize, out_ptr: [*]u32, out_offsets: [*]usize, max_tokens: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt) |*mgt| {
        const texts_data = texts_ptr[0..texts_len];
        const offsets = text_offsets[0..num_texts];

        var total_tokens: usize = 0;
        var batch_idx: usize = 0;

        while (batch_idx < num_texts) : (batch_idx += 1) {
            const start = offsets[batch_idx];
            const end = if (batch_idx + 1 < num_texts) offsets[batch_idx + 1] else texts_len;
            const text = texts_data[start..end];

            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            mgt.encode(text, &tokens) catch return -2;

            const available = max_tokens - total_tokens;
            const count = @min(tokens.items.len, available);

            if (count > 0) {
                @memcpy(out_ptr[total_tokens .. total_tokens + count], tokens.items[0..count]);
            }

            out_offsets[batch_idx] = total_tokens;
            total_tokens += count;

            if (total_tokens >= max_tokens) break;
        }

        return @intCast(total_tokens);
    }

    return -1;
}

pub export fn wasmGetVocabSize() i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    const mgt = global_mgt orelse return 0;
    return @intCast(mgt.vocabSize());
}

pub export fn wasmGetRSFDim() i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    const rsf = global_rsf orelse return 0;
    return @intCast(rsf.dim);
}

pub export fn wasmGetRSFLayers() i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    const rsf = global_rsf orelse return 0;
    return @intCast(rsf.num_layers);
}

var cleanup_performed: bool = false;

pub export fn wasmCleanup() void {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (cleanup_performed) return;

    global_mgt = null;
    global_rsf = null;

    if (global_model) |*model| {
        model.deinit();
        global_model = null;
    }

    cleanup_performed = true;
}

pub export fn wasmVersion() [*:0]const u8 {
    return "JAIDE-WASM-1.0.0";
}

pub export fn wasmReady() i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt != null) return 1;
    if (global_model) |model| {
        if (model.mgt != null) return 1;
    }
    return 0;
}

pub export fn wasmTokenize(text_ptr: [*]const u8, text_len: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt) |*mgt| {
        const text = text_ptr[0..text_len];

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        mgt.encode(text, &tokens) catch return -1;

        return @intCast(tokens.items.len);
    }

    return -1;
}

pub export fn wasmDetokenize(tokens_ptr: [*]const u32, tokens_len: usize) i32 {
    global_state_mutex.lock();
    defer global_state_mutex.unlock();

    if (global_mgt) |*mgt| {
        const tokens = tokens_ptr[0..tokens_len];

        var text = std.ArrayList(u8).init(allocator);
        defer text.deinit();

        mgt.decode(tokens, &text) catch return -1;

        return @intCast(text.items.len);
    }

    return -1;
}

comptime {
    if (@import("builtin").target.cpu.arch == .wasm32 or @import("builtin").target.cpu.arch == .wasm64) {
        @export(wasmAlloc, .{ .name = "alloc", .linkage = .Strong });
        @export(wasmFree, .{ .name = "free", .linkage = .Strong });
        @export(wasmGetMemory, .{ .name = "getMemory", .linkage = .Strong });
        @export(wasmInitModel, .{ .name = "initModel", .linkage = .Strong });
        @export(wasmInitRSF, .{ .name = "initRSF", .linkage = .Strong });
        @export(wasmLoadModel, .{ .name = "loadModel", .linkage = .Strong });
        @export(wasmEncode, .{ .name = "encode", .linkage = .Strong });
        @export(wasmDecode, .{ .name = "decode", .linkage = .Strong });
        @export(wasmInference, .{ .name = "inference", .linkage = .Strong });
        @export(wasmGetEmbeddings, .{ .name = "getEmbeddings", .linkage = .Strong });
        @export(wasmBatchEncode, .{ .name = "batchEncode", .linkage = .Strong });
        @export(wasmGetVocabSize, .{ .name = "getVocabSize", .linkage = .Strong });
        @export(wasmGetRSFDim, .{ .name = "getRSFDim", .linkage = .Strong });
        @export(wasmGetRSFLayers, .{ .name = "getRSFLayers", .linkage = .Strong });
        @export(wasmCleanup, .{ .name = "cleanup", .linkage = .Strong });
        @export(wasmVersion, .{ .name = "version", .linkage = .Strong });
        @export(wasmReady, .{ .name = "ready", .linkage = .Strong });
        @export(wasmTokenize, .{ .name = "tokenize", .linkage = .Strong });
        @export(wasmDetokenize, .{ .name = "detokenize", .linkage = .Strong });
    }
}
