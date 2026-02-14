const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
    PartialRowCleanup,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        futhark.futhark_context_config_set_device(cfg, 0);
        futhark.futhark_context_config_set_default_group_size(cfg, 256);
        futhark.futhark_context_config_set_default_num_groups(cfg, 128);
        futhark.futhark_context_config_set_default_tile_size(cfg, 32);

        const ctx = futhark.futhark_context_new(cfg);
        futhark.futhark_context_config_free(cfg);

        if (ctx == null) return AccelError.FutharkContextFailed;

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }

    pub fn getDataPointer(self: *Self, array: *FutharkArray2DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_2d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }
};

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0 };
        }

        var ptr: ?*anyopaque = null;
        const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
        if (err != cuda.cudaSuccess) {
            return AccelError.CudaHostAllocFailed;
        }

        return Self{
            .ptr = ptr,
            .size = size,
        };
    }

    pub fn free(self: *Self) void {
        if (self.ptr) |p| {
            _ = cuda.cudaFreeHost(p);
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        if (self.size == 0) return &[_]T{};
        const count = self.size / @sizeOf(T);
        if (count == 0) return &[_]T{};
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(ctx: *FutharkContext, data: []const []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;

        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return AccelError.InvalidDimensions;

        for (data) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
        }

        const total = rows * cols;
        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        flat_data.ensureTotalCapacity(total) catch return AccelError.AllocationFailed;

        for (data) |row| {
            flat_data.appendSlice(row) catch return AccelError.AllocationFailed;
        }

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.items.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != rows * cols) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = rows * cols;
        const zeros = std.heap.page_allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn values(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![][]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;

        var dims: [2]i64 = undefined;
        if (futhark.futhark_shape_f16_2d(ctx.ctx, self.arr, &dims) != 0) {
            return AccelError.FutharkShapeFailed;
        }
        const rows = @as(usize, @intCast(dims[0]));
        const cols = @as(usize, @intCast(dims[1]));

        if (rows == 0 or cols == 0) {
            return allocator.alloc([]f16, 0) catch return AccelError.AllocationFailed;
        }

        const flat = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(flat.ptr)) != 0) {
            return AccelError.FutharkValuesFailed;
        }

        const result = allocator.alloc([]f16, rows) catch return AccelError.AllocationFailed;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = allocator.alloc(f16, cols) catch {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    allocator.free(result[j]);
                }
                allocator.free(result);
                return AccelError.PartialRowCleanup;
            };
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,
    velocity_s: FutharkArray2DF16,
    velocity_t: FutharkArray2DF16,
    model_dim: usize,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        var weights_s = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer weights_s.free(&ctx);

        var weights_t = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer weights_t.free(&ctx);

        var velocity_s = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer velocity_s.free(&ctx);

        var velocity_t = try FutharkArray2DF16.newZeros(&ctx, model_dim, model_dim);
        errdefer velocity_t.free(&ctx);

        return Self{
            .ctx = ctx,
            .weights_s = weights_s,
            .weights_t = weights_t,
            .velocity_s = velocity_s,
            .velocity_t = velocity_t,
            .model_dim = model_dim,
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        self.velocity_t.free(&self.ctx);
        self.velocity_s.free(&self.ctx);
        self.weights_t.free(&self.ctx);
        self.weights_s.free(&self.ctx);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null) return AccelError.NullPointer;
        if (self.weights_t.arr == null) return AccelError.NullPointer;

        var output: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_rsf_forward(
            self.ctx.ctx,
            &output,
            input.arr,
            self.weights_s.arr,
            self.weights_t.arr,
        );

        if (result != 0) {
            return AccelError.FutharkForwardFailed;
        }

        if (output == null) {
            return AccelError.NullPointer;
        }

        return FutharkArray2DF16{
            .arr = output,
            .rows = input.rows,
            .cols = input.cols,
        };
    }

    pub fn trainingStep(
        self: *Self,
        inputs: *FutharkArray2DF16,
        targets: *FutharkArray2DF16,
        learning_rate: f16,
        momentum: f16,
    ) AccelError!f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;
        if (self.velocity_s.arr == null or self.velocity_t.arr == null) return AccelError.NullPointer;

        var new_ws: ?*futhark.struct_futhark_f16_2d = null;
        var new_wt: ?*futhark.struct_futhark_f16_2d = null;
        var new_vs: ?*futhark.struct_futhark_f16_2d = null;
        var new_vt: ?*futhark.struct_futhark_f16_2d = null;
        var loss: u16 = 0;

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);

        const result = futhark.futhark_entry_training_step(
            self.ctx.ctx,
            &new_ws,
            &new_wt,
            &new_vs,
            &new_vt,
            &loss,
            inputs.arr,
            targets.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.velocity_s.arr,
            self.velocity_t.arr,
            lr_bits,
            momentum_bits,
        );

        if (result != 0) {
            return AccelError.FutharkTrainingStepFailed;
        }

        if (new_ws == null or new_wt == null or new_vs == null or new_vt == null) {
            return AccelError.NullPointer;
        }

        const old_ws = self.weights_s.arr;
        const old_wt = self.weights_t.arr;
        const old_vs = self.velocity_s.arr;
        const old_vt = self.velocity_t.arr;

        self.weights_s.arr = new_ws;
        self.weights_t.arr = new_wt;
        self.velocity_s.arr = new_vs;
        self.velocity_t.arr = new_vt;

        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_ws);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_wt);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_vs);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_vt);

        const loss_f16: f16 = @bitCast(loss);
        return loss_f16;
    }

    pub fn scaleWeights(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;

        const scale_f32: f32 = @floatCast(scale_factor);
        if (scale_f32 == 0.0) return AccelError.InvalidDimensions;

        const result_s = futhark.futhark_entry_scale_weights_inplace(
            self.ctx.ctx,
            self.weights_s.arr,
            scale_f32,
        );

        if (result_s != 0) {
            return AccelError.FutharkScaleWeightsFailed;
        }

        const result_t = futhark.futhark_entry_scale_weights_inplace(
            self.ctx.ctx,
            self.weights_t.arr,
            scale_f32,
        );

        if (result_t != 0) {
            return AccelError.FutharkScaleWeightsFailed;
        }
    }

    pub fn getWeightsSDataPointer(self: *Self) AccelError!*anyopaque {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.getDataPointer(&self.weights_s);
    }

    pub fn getWeightsTDataPointer(self: *Self) AccelError!*anyopaque {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.getDataPointer(&self.weights_t);
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn setWeightsS(self: *Self, data: []const f16, rows: usize, cols: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;

        self.weights_s.free(&self.ctx);
        self.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setWeightsT(self: *Self, data: []const f16, rows: usize, cols: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;

        self.weights_t.free(&self.ctx);
        self.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }
};
