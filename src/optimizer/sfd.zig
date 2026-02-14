const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const Precision = enum {
    fp4,
    fp8,
    fp16,
    fp32,
    fp64,
};

pub const Shape = struct {
    dims: []const usize,

    pub fn totalSize(self: Shape) usize {
        var size: usize = 1;
        for (self.dims) |dim| {
            size *= dim;
        }
        return size;
    }
};

pub const TensorFlags = struct {
    in_tensor_memory: bool = false,
    requires_grad: bool = true,
    is_compressed: bool = false,
};

pub const Tensor = struct {
    data: []f32,
    shape: Shape,
    dtype: Precision = .fp32,
    flags: TensorFlags = .{},
    allocator: Allocator,

    pub fn init(allocator: Allocator, dims: []const usize) !Tensor {
        const owned_dims = try allocator.dupe(usize, dims);
        errdefer allocator.free(owned_dims);

        const shape = Shape{ .dims = owned_dims };
        const size = shape.totalSize();

        const data = try allocator.alloc(f32, size);
        errdefer allocator.free(data);

        return Tensor{
            .data = data,
            .shape = shape,
            .allocator = allocator,
        };
    }

    pub fn zeros(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        try tensor.fill(0.0);
        return tensor;
    }

    pub fn ones(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        try tensor.fill(1.0);
        return tensor;
    }

    pub fn eye(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        try tensor.fill(0.0);
        if (dims.len == 2 and dims[0] == dims[1]) {
            const n = dims[0];
            var i: usize = 0;
            while (i < n) : (i += 1) {
                tensor.data[i * n + i] = 1.0;
            }
        }
        return tensor;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape.dims);
    }

    pub fn fill(self: *Tensor, value: f32) !void {
        for (self.data) |*v| {
            v.* = value;
        }
    }

    pub fn fillRandomNormal(self: *Tensor, mean: f32, std_dev: f32) !void {
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();

        for (self.data) |*v| {
            const rand_u1 = random.float(f32);
            const rand_u2 = random.float(f32);
            const z0 = @sqrt(-2.0 * @log(rand_u1)) * @cos(2.0 * std.math.pi * rand_u2);
            v.* = mean + std_dev * z0;
        }
    }

    pub fn fillRademacher(self: *Tensor) !void {
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();

        for (self.data) |*v| {
            v.* = if (random.float(f32) < 0.5) -1.0 else 1.0;
        }
    }

    pub fn clone(self: *const Tensor, allocator: Allocator) !Tensor {
        var new_tensor = try Tensor.init(allocator, self.shape.dims);
        @memcpy(new_tensor.data, self.data);
        new_tensor.dtype = self.dtype;
        new_tensor.flags = self.flags;
        return new_tensor;
    }

    pub fn copyFrom(self: *Tensor, other: *const Tensor) !void {
        const min_len = @min(self.data.len, other.data.len);
        @memcpy(self.data[0..min_len], other.data[0..min_len]);
    }

    pub fn copyFromWithCast(self: *Tensor, other: *const Tensor) !void {
        try self.copyFrom(other);
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) !void {
        for (self.data) |*v| {
            v.* *= scalar;
        }
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        const min_len = @min(self.data.len, other.data.len);
        var i: usize = 0;
        while (i < min_len) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        const min_len = @min(self.data.len, other.data.len);
        var i: usize = 0;
        while (i < min_len) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    pub fn normL2(self: *const Tensor) f32 {
        var sum: f64 = 0.0;
        for (self.data) |v| {
            if (std.math.isFinite(v)) {
                sum += @as(f64, v) * @as(f64, v);
            }
        }
        return @floatCast(@sqrt(sum));
    }

    pub fn spectralNorm(self: *const Tensor, allocator: Allocator, max_iter: usize, eps: f32) !f32 {
        if (self.shape.dims.len != 2) return 1.0;

        const m = self.shape.dims[0];
        const n = self.shape.dims[1];

        var v = try Tensor.init(allocator, &[_]usize{n});
        defer v.deinit();
        try v.fillRandomNormal(0.0, 1.0);

        var u = try Tensor.init(allocator, &[_]usize{m});
        defer u.deinit();

        var iter: usize = 0;
        while (iter < max_iter) : (iter += 1) {
            try u.fill(0.0);
            var i: usize = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    u.data[i] += self.data[i * n + j] * v.data[j];
                }
            }

            const u_norm = u.normL2();
            if (u_norm > eps) {
                try u.mulScalar(1.0 / u_norm);
            }

            try v.fill(0.0);
            i = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    v.data[j] += self.data[i * n + j] * u.data[i];
                }
            }

            const v_norm = v.normL2();
            if (v_norm > eps) {
                try v.mulScalar(1.0 / v_norm);
            }
        }

        var sigma: f32 = 0.0;
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                sigma += u.data[i] * self.data[i * n + j] * v.data[j];
            }
        }

        return @fabs(sigma);
    }

    pub fn matmul(self: *Tensor, A: *const Tensor, B: *const Tensor) !void {
        if (A.shape.dims.len != 2 or B.shape.dims.len != 2) return error.InvalidShape;

        const m = A.shape.dims[0];
        const k = A.shape.dims[1];
        const n = B.shape.dims[1];

        if (B.shape.dims[0] != k) return error.ShapeMismatch;
        if (self.data.len != m * n) return error.ShapeMismatch;

        try self.fill(0.0);

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var sum: f32 = 0.0;
                var p: usize = 0;
                while (p < k) : (p += 1) {
                    sum += A.data[i * k + p] * B.data[p * n + j];
                }
                self.data[i * n + j] = sum;
            }
        }
    }

    pub fn outerProduct(self: *const Tensor, allocator: Allocator, other: *const Tensor) !Tensor {
        const m = self.data.len;
        const n = other.data.len;

        var result = try Tensor.init(allocator, &[_]usize{ m, n });

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                result.data[i * n + j] = self.data[i] * other.data[j];
            }
        }

        return result;
    }

    pub fn sizeBytes(self: *const Tensor) usize {
        return self.data.len * @sizeOf(f32);
    }

    pub fn convertToFP4(self: *Tensor) !void {
        self.flags.is_compressed = true;
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        try writer.writeInt(u64, self.shape.dims.len, .little);
        for (self.shape.dims) |dim| {
            try writer.writeInt(u64, dim, .little);
        }
        for (self.data) |val| {
            try writer.writeInt(u32, @bitCast(val), .little);
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const ndims = try reader.readInt(u64, .little);
        var dims = try allocator.alloc(usize, ndims);
        errdefer allocator.free(dims);

        var i: usize = 0;
        while (i < ndims) : (i += 1) {
            dims[i] = try reader.readInt(u64, .little);
        }

        var tensor = try Tensor.init(allocator, dims);
        errdefer tensor.deinit();

        allocator.free(dims);

        i = 0;
        while (i < tensor.data.len) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            tensor.data[i] = @bitCast(bits);
        }

        return tensor;
    }
};

pub const LossFn = *const fn (params: *const Tensor, context: ?*anyopaque) anyerror!f32;

pub const SFDConfig = struct {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    clip_threshold: f32 = 1.0,
    fisher_max: f32 = 1e6,
    warmup_steps: usize = 10,
    finite_diff_eps: f32 = 1e-5,
    second_order_eps: f32 = 1e-4,
};

pub const KFACBlock = struct {
    A_inv: Tensor,
    G_inv: Tensor,
    damping: f32,
    update_freq: usize,
    last_update: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, input_dim: usize, output_dim: usize, damping: f32) !KFACBlock {
        const A_shape = [_]usize{ input_dim, input_dim };
        const G_shape = [_]usize{ output_dim, output_dim };

        var A = try Tensor.eye(allocator, &A_shape);
        errdefer A.deinit();

        var G = try Tensor.eye(allocator, &G_shape);
        errdefer G.deinit();

        return KFACBlock{
            .A_inv = A,
            .G_inv = G,
            .damping = damping,
            .update_freq = 10,
            .last_update = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KFACBlock) void {
        self.A_inv.deinit();
        self.G_inv.deinit();
    }

    pub fn updateStatistics(self: *KFACBlock, activations: *const Tensor, gradients: *const Tensor) !void {
        var aa_T = try activations.outerProduct(self.allocator, activations);
        defer aa_T.deinit();

        var gg_T = try gradients.outerProduct(self.allocator, gradients);
        defer gg_T.deinit();

        const alpha: f32 = 0.95;
        const min_len_A = @min(self.A_inv.data.len, aa_T.data.len);
        var i: usize = 0;
        while (i < min_len_A) : (i += 1) {
            self.A_inv.data[i] = alpha * self.A_inv.data[i] + (1.0 - alpha) * aa_T.data[i];
        }

        const min_len_G = @min(self.G_inv.data.len, gg_T.data.len);
        i = 0;
        while (i < min_len_G) : (i += 1) {
            self.G_inv.data[i] = alpha * self.G_inv.data[i] + (1.0 - alpha) * gg_T.data[i];
        }
    }

    pub fn preconditionGradient(self: *const KFACBlock, grad: *Tensor) !void {
        var A_inv_sqrt = try self.computeInverseSqrt(&self.A_inv);
        defer A_inv_sqrt.deinit();

        var G_inv_sqrt = try self.computeInverseSqrt(&self.G_inv);
        defer G_inv_sqrt.deinit();

        try grad.matmul(&G_inv_sqrt, &A_inv_sqrt);
    }

    fn computeInverseSqrt(self: *const KFACBlock, M: *const Tensor) !Tensor {
        var Y = try M.clone(self.allocator);
        errdefer Y.deinit();

        const max_iter: usize = 5;
        var iter: usize = 0;
        while (iter < max_iter) : (iter += 1) {
            var Y_squared = try Tensor.init(self.allocator, M.shape.dims);
            defer Y_squared.deinit();
            try Y_squared.matmul(&Y, &Y);

            var M_Y_sq = try Tensor.init(self.allocator, M.shape.dims);
            defer M_Y_sq.deinit();
            try M_Y_sq.matmul(M, &Y_squared);

            var three_I = try Tensor.eye(self.allocator, M.shape.dims);
            defer three_I.deinit();
            try three_I.mulScalar(3.0);

            try three_I.sub(&M_Y_sq);
            try three_I.mulScalar(0.5);

            var Y_new = try Tensor.init(self.allocator, M.shape.dims);
            defer Y.deinit();
            try Y_new.matmul(&Y, &three_I);

            Y = Y_new;
        }

        return Y;
    }
};

pub const SpectralNormalizer = struct {
    power_iterations: usize,
    eps: f32,
    max_singular_value: f32,

    pub fn init(power_iterations: usize) SpectralNormalizer {
        return SpectralNormalizer{
            .power_iterations = power_iterations,
            .eps = 1e-12,
            .max_singular_value = 1.0,
        };
    }

    pub fn normalizeWeights(self: *SpectralNormalizer, weights: *Tensor, allocator: Allocator) !void {
        const sigma = try weights.spectralNorm(allocator, self.power_iterations, self.eps);

        if (sigma > self.max_singular_value) {
            try weights.mulScalar(self.max_singular_value / sigma);
        }
    }

    pub fn lipschitzRegularization(_: *const SpectralNormalizer, loss: f32, spectral_norms: []const f32, lambda: f32) f32 {
        var reg_term: f32 = 0.0;
        for (spectral_norms) |sigma| {
            const deviation = sigma - 1.0;
            reg_term += deviation * deviation;
        }

        return loss + lambda * reg_term;
    }
};

pub const GradientFlowController = struct {
    spectral_normalizer: SpectralNormalizer,
    gradient_clip_norm: f32,
    use_normalized_gradient_flow: bool,

    pub fn init() GradientFlowController {
        return GradientFlowController{
            .spectral_normalizer = SpectralNormalizer.init(20),
            .gradient_clip_norm = 1.0,
            .use_normalized_gradient_flow = true,
        };
    }

    pub fn stabilizeGradients(self: *GradientFlowController, gradients: []*Tensor, weights: []*Tensor, allocator: Allocator) !void {
        for (weights) |w| {
            try self.spectral_normalizer.normalizeWeights(w, allocator);
        }

        if (self.use_normalized_gradient_flow) {
            for (gradients) |grad| {
                const norm = grad.normL2();
                if (norm > self.gradient_clip_norm) {
                    const scale = self.gradient_clip_norm / (norm + 1e-8);
                    try grad.mulScalar(scale);
                }
            }
        }

        for (gradients) |grad| {
            var mean: f32 = 0.0;
            var variance: f32 = 0.0;

            for (grad.data) |g| {
                mean += g;
            }
            mean /= @as(f32, @floatFromInt(grad.data.len));

            for (grad.data) |g| {
                const diff = g - mean;
                variance += diff * diff;
            }
            variance /= @as(f32, @floatFromInt(grad.data.len));

            const std_dev = @sqrt(variance + 1e-8);
            for (grad.data) |*g| {
                g.* = (g.* - mean) / std_dev;
            }
        }
    }
};

pub const MARSVarianceReducer = struct {
    reference_gradients: []Tensor,
    snapshot_freq: usize,
    scale_factor: f32,
    momentum: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, param_shapes: []const []const usize) !MARSVarianceReducer {
        var ref_grads = try allocator.alloc(Tensor, param_shapes.len);
        errdefer allocator.free(ref_grads);

        for (param_shapes, 0..) |shape, i| {
            ref_grads[i] = try Tensor.init(allocator, shape);
            errdefer ref_grads[i].deinit();
            try ref_grads[i].fill(0.0);
        }

        return MARSVarianceReducer{
            .reference_gradients = ref_grads,
            .snapshot_freq = 100,
            .scale_factor = 1.0,
            .momentum = 0.9,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MARSVarianceReducer) void {
        for (self.reference_gradients) |*rg| {
            rg.deinit();
        }
        self.allocator.free(self.reference_gradients);
    }

    pub fn varianceReducedGradient(self: *MARSVarianceReducer, current_grad: *const Tensor, reference_grad: *const Tensor, param_idx: usize) !Tensor {
        var vr_grad = try Tensor.init(self.allocator, current_grad.shape.dims);
        errdefer vr_grad.deinit();

        for (vr_grad.data, 0..) |*g, i| {
            const g_current = current_grad.data[i];
            const g_ref = self.reference_gradients[param_idx].data[i];

            g.* = g_current - reference_grad.data[i] + g_ref;
            g.* = self.momentum * g.* + (1.0 - self.momentum) * g_current;
        }

        return vr_grad;
    }

    pub fn updateReferenceGradients(self: *MARSVarianceReducer, full_batch_gradients: []const Tensor) !void {
        for (full_batch_gradients, 0..) |grad, i| {
            if (i < self.reference_gradients.len) {
                @memcpy(self.reference_gradients[i].data, grad.data[0..grad.data.len]);
            }
        }
    }
};

pub const ReversibleOptimizerState = struct {
    forward_cache_policy: CachePolicy,
    recompute_threshold: f32,
    jacobian_cache: std.AutoHashMap(usize, Tensor),
    allocator: Allocator,

    pub const CachePolicy = enum {
        cache_all,
        recompute_all,
        adaptive,
    };

    pub fn init(allocator: Allocator) ReversibleOptimizerState {
        return ReversibleOptimizerState{
            .forward_cache_policy = .adaptive,
            .recompute_threshold = 0.5,
            .jacobian_cache = std.AutoHashMap(usize, Tensor).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ReversibleOptimizerState) void {
        var iter = self.jacobian_cache.iterator();
        while (iter.next()) |entry| {
            var tensor = entry.value_ptr.*;
            tensor.deinit();
        }
        self.jacobian_cache.deinit();
    }

    pub fn shouldRecompute(self: *ReversibleOptimizerState, layer_idx: usize, computation_cost: f32, memory_cost: f32, available_memory: f32) bool {
        _ = layer_idx;

        switch (self.forward_cache_policy) {
            .cache_all => return false,
            .recompute_all => return true,
            .adaptive => {
                if (available_memory < memory_cost) {
                    return true;
                }

                const recompute_cost = computation_cost;
                const cache_cost = memory_cost * self.recompute_threshold;

                return recompute_cost < cache_cost;
            },
        }
    }

    pub fn backwardPassReversible(self: *ReversibleOptimizerState, forward_outputs: []const Tensor, grad_output: *const Tensor) ![]Tensor {
        var grad_inputs = try self.allocator.alloc(Tensor, forward_outputs.len);
        errdefer self.allocator.free(grad_inputs);

        var current_grad = try grad_output.clone(self.allocator);
        defer current_grad.deinit();

        var i: usize = forward_outputs.len;
        while (i > 0) : (i -= 1) {
            const layer_idx = i - 1;

            const should_recomp = self.shouldRecompute(layer_idx, 1.0, forward_outputs[layer_idx].sizeBytes(), 1024 * 1024 * 1024);

            if (should_recomp) {
                var reconstructed_input = try self.reverseLayer(&forward_outputs[layer_idx], layer_idx);
                defer reconstructed_input.deinit();

                grad_inputs[layer_idx] = try self.computeGradient(&reconstructed_input, &current_grad);
            } else {
                grad_inputs[layer_idx] = try current_grad.clone(self.allocator);
            }

            try current_grad.copyFrom(&grad_inputs[layer_idx]);
        }

        return grad_inputs;
    }

    fn reverseLayer(self: *ReversibleOptimizerState, output: *const Tensor, layer_idx: usize) !Tensor {
        var x = try output.clone(self.allocator);
        errdefer x.deinit();

        const max_iter: usize = 10;
        var iter: usize = 0;
        while (iter < max_iter) : (iter += 1) {
            var g_x = try self.computeResidual(&x, layer_idx);
            defer g_x.deinit();

            try x.copyFrom(output);
            try x.sub(&g_x);

            const delta = g_x.normL2();
            if (delta < 1e-6) break;
        }

        return x;
    }

    fn computeResidual(self: *ReversibleOptimizerState, input: *const Tensor, layer_idx: usize) !Tensor {
        _ = layer_idx;
        var residual = try input.clone(self.allocator);
        try residual.mulScalar(0.1);
        return residual;
    }

    fn computeGradient(self: *ReversibleOptimizerState, input: *const Tensor, grad_output: *const Tensor) !Tensor {
        _ = self;
        var grad_input = try input.clone(input.allocator);

        for (grad_input.data, 0..) |*g, i| {
            g.* = grad_output.data[i];
        }

        return grad_input;
    }
};

pub const LRScheduleType = enum {
    cosine_annealing,
    cosine_annealing_with_warmup,
    polynomial_decay,
    exponential_decay,
    one_cycle,
    sophia_style,
};

pub const LRScheduler = struct {
    schedule_type: LRScheduleType,
    base_lr: f32,
    min_lr: f32,
    max_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, schedule_type: LRScheduleType, base_lr: f32, warmup_steps: usize, total_steps: usize) LRScheduler {
        return LRScheduler{
            .schedule_type = schedule_type,
            .base_lr = base_lr,
            .min_lr = base_lr * 0.01,
            .max_lr = base_lr * 10.0,
            .warmup_steps = warmup_steps,
            .total_steps = total_steps,
            .current_step = 0,
            .allocator = allocator,
        };
    }

    pub fn getLearningRate(self: *LRScheduler, hessian_info: ?*const Tensor) !f32 {
        var lr = self.base_lr;

        if (self.current_step < self.warmup_steps) {
            const warmup_progress = @as(f32, @floatFromInt(self.current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
            lr = self.base_lr * warmup_progress;
        } else {
            switch (self.schedule_type) {
                .cosine_annealing => {
                    const progress = @as(f32, @floatFromInt(self.current_step - self.warmup_steps)) / @as(f32, @floatFromInt(self.total_steps - self.warmup_steps));
                    lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + @cos(std.math.pi * progress));
                },
                .cosine_annealing_with_warmup => {
                    const progress = @as(f32, @floatFromInt(self.current_step - self.warmup_steps)) / @as(f32, @floatFromInt(self.total_steps - self.warmup_steps));
                    lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + @cos(std.math.pi * progress));
                },
                .polynomial_decay => {
                    const progress = @as(f32, @floatFromInt(self.current_step - self.warmup_steps)) / @as(f32, @floatFromInt(self.total_steps - self.warmup_steps));
                    const power: f32 = 2.0;
                    lr = self.base_lr * std.math.pow(f32, 1.0 - progress, power);
                },
                .exponential_decay => {
                    const decay_rate: f32 = 0.96;
                    const decay_steps: f32 = 1000.0;
                    const steps_since_warmup = @as(f32, @floatFromInt(self.current_step - self.warmup_steps));
                    lr = self.base_lr * std.math.pow(f32, decay_rate, steps_since_warmup / decay_steps);
                },
                .one_cycle => {
                    const mid_point = self.total_steps / 2;
                    if (self.current_step < mid_point) {
                        const progress = @as(f32, @floatFromInt(self.current_step - self.warmup_steps)) / @as(f32, @floatFromInt(mid_point - self.warmup_steps));
                        lr = self.base_lr + (self.max_lr - self.base_lr) * progress;
                    } else {
                        const progress = @as(f32, @floatFromInt(self.current_step - mid_point)) / @as(f32, @floatFromInt(self.total_steps - mid_point));
                        lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1.0 + @cos(std.math.pi * progress));
                    }
                },
                .sophia_style => {
                    if (hessian_info) |hess| {
                        var hess_values = try self.allocator.alloc(f32, hess.data.len);
                        defer self.allocator.free(hess_values);
                        @memcpy(hess_values, hess.data);

                        std.mem.sort(f32, hess_values, {}, comptime std.sort.asc(f32));
                        const median_hess = hess_values[hess_values.len / 2];

                        lr = self.base_lr / @sqrt(median_hess + 1e-8);
                        lr = std.math.clamp(lr, self.min_lr, self.max_lr);
                    } else {
                        lr = self.base_lr;
                    }
                },
            }
        }

        self.current_step += 1;
        return std.math.clamp(lr, self.min_lr, self.max_lr);
    }
};

pub const MixedPrecisionConfig = struct {
    use_fp4: bool = true,
    use_fp8: bool = true,
    use_fp16: bool = true,
    master_weights_precision: Precision = .fp32,
    gradient_accumulation_steps: usize = 4,
    loss_scale: f32 = 1024.0,
    dynamic_loss_scaling: bool = true,
};

pub const DynamicLossScaler = struct {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_last_overflow: usize,

    pub fn init(initial_scale: f32) DynamicLossScaler {
        return DynamicLossScaler{
            .scale = initial_scale,
            .growth_factor = 2.0,
            .backoff_factor = 0.5,
            .growth_interval = 2000,
            .steps_since_last_overflow = 0,
        };
    }

    pub fn update(self: *DynamicLossScaler, gradients: []const Tensor) void {
        var has_overflow = false;
        for (gradients) |grad| {
            for (grad.data) |g| {
                if (!std.math.isFinite(g)) {
                    has_overflow = true;
                    break;
                }
            }
            if (has_overflow) break;
        }

        if (has_overflow) {
            self.scale *= self.backoff_factor;
            self.steps_since_last_overflow = 0;
        } else {
            self.steps_since_last_overflow += 1;
            if (self.steps_since_last_overflow >= self.growth_interval) {
                self.scale *= self.growth_factor;
                self.steps_since_last_overflow = 0;
            }
        }

        self.scale = std.math.clamp(self.scale, 1.0, 65536.0);
    }
};

pub const MixedPrecisionTrainer = struct {
    config: MixedPrecisionConfig,
    master_weights: []Tensor,
    working_weights: []Tensor,
    accumulated_gradients: []Tensor,
    accumulation_counter: usize,
    loss_scaler: DynamicLossScaler,
    allocator: Allocator,

    pub fn init(allocator: Allocator, weight_shapes: []const []const usize, config: MixedPrecisionConfig) !MixedPrecisionTrainer {
        var master_w = try allocator.alloc(Tensor, weight_shapes.len);
        errdefer allocator.free(master_w);

        var working_w = try allocator.alloc(Tensor, weight_shapes.len);
        errdefer allocator.free(working_w);

        var accum_g = try allocator.alloc(Tensor, weight_shapes.len);
        errdefer allocator.free(accum_g);

        for (weight_shapes, 0..) |shape, i| {
            master_w[i] = try Tensor.init(allocator, shape);
            master_w[i].dtype = .fp32;
            try master_w[i].fillRandomNormal(0.0, 0.02);

            working_w[i] = try Tensor.init(allocator, shape);
            working_w[i].dtype = if (config.use_fp4) .fp4 else if (config.use_fp8) .fp8 else .fp16;
            try working_w[i].copyFrom(&master_w[i]);

            accum_g[i] = try Tensor.init(allocator, shape);
            accum_g[i].dtype = .fp32;
            try accum_g[i].fill(0.0);
        }

        return MixedPrecisionTrainer{
            .config = config,
            .master_weights = master_w,
            .working_weights = working_w,
            .accumulated_gradients = accum_g,
            .accumulation_counter = 0,
            .loss_scaler = DynamicLossScaler.init(config.loss_scale),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MixedPrecisionTrainer) void {
        for (self.master_weights) |*w| {
            w.deinit();
        }
        self.allocator.free(self.master_weights);

        for (self.working_weights) |*w| {
            w.deinit();
        }
        self.allocator.free(self.working_weights);

        for (self.accumulated_gradients) |*g| {
            g.deinit();
        }
        self.allocator.free(self.accumulated_gradients);
    }

    pub fn updateWeights(self: *MixedPrecisionTrainer, lr: f32) !void {
        const scale = 1.0 / @as(f32, @floatFromInt(self.config.gradient_accumulation_steps));

        for (self.accumulated_gradients) |*grad| {
            try grad.mulScalar(scale);
        }

        for (self.master_weights, 0..) |*master, i| {
            for (master.data, 0..) |*w, j| {
                w.* -= lr * self.accumulated_gradients[i].data[j];
            }
        }

        for (self.working_weights, 0..) |*working, i| {
            try working.copyFromWithCast(&self.master_weights[i]);
        }
    }
};

pub const B200OptimizationConfig = struct {
    use_fp4_tensor_cores: bool = true,
    use_tensor_memory: bool = true,
    nvlink_bandwidth_tbps: f32 = 1.8,
    hbm_bandwidth_tbps: f32 = 8.0,
    decompression_engine: bool = true,
    multi_instance_gpu: bool = false,
    l2_cache_size_mb: usize = 50,
    tmem_size_mb: usize = 32,
};

pub const B200MemoryManager = struct {
    config: B200OptimizationConfig,
    tensor_memory_pool: []u8,
    prefetch_queue: ArrayList(usize),
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: B200OptimizationConfig) !B200MemoryManager {
        const tmem_size = config.tmem_size_mb * 1024 * 1024;
        var tmem_pool = try allocator.alloc(u8, tmem_size);
        errdefer allocator.free(tmem_pool);

        var prefetch_q = ArrayList(usize).init(allocator);
        errdefer prefetch_q.deinit();

        return B200MemoryManager{
            .config = config,
            .tensor_memory_pool = tmem_pool,
            .prefetch_queue = prefetch_q,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *B200MemoryManager) void {
        self.allocator.free(self.tensor_memory_pool);
        self.prefetch_queue.deinit();
    }

    pub fn optimizeMemoryAccess(self: *B200MemoryManager, tensors: []*Tensor, access_pattern: []const usize) !void {
        for (tensors, 0..) |tensor, i| {
            const access_freq = access_pattern[i];

            if (access_freq > 10 and self.config.use_tensor_memory) {
                try self.moveToTensorMemory(tensor);
            }
        }

        for (access_pattern, 0..) |freq, idx| {
            if (freq > 0) {
                try self.prefetch_queue.append(idx);
            }
        }

        if (self.config.decompression_engine) {
            for (tensors) |tensor| {
                try self.compressIfBeneficial(tensor);
            }
        }
    }

    fn moveToTensorMemory(self: *B200MemoryManager, tensor: *Tensor) !void {
        const tensor_size = tensor.data.len * @sizeOf(f32);

        if (tensor_size <= self.tensor_memory_pool.len) {
            tensor.flags.in_tensor_memory = true;
        }
    }

    fn compressIfBeneficial(self: *B200MemoryManager, tensor: *Tensor) !void {
        if (tensor.dtype == .fp32 and self.config.use_fp4_tensor_cores) {
            try tensor.convertToFP4();
        }
    }
};

pub const OpType = enum {
    matmul,
    add,
    activation,
    fused_gemm_bias_act,
};

pub const FusedKernel = struct {
    operations: []OpType,
    use_fp4: bool,
    allocator: Allocator,

    pub fn deinit(self: *FusedKernel) void {
        self.allocator.free(self.operations);
    }
};

pub const B200KernelOptimizer = struct {
    config: B200OptimizationConfig,

    pub fn init(config: B200OptimizationConfig) B200KernelOptimizer {
        return B200KernelOptimizer{ .config = config };
    }

    pub fn fuseOperations(self: *B200KernelOptimizer, operations: []const OpType, allocator: Allocator) !FusedKernel {
        var fused_ops = ArrayList(OpType).init(allocator);
        defer fused_ops.deinit();

        var i: usize = 0;
        while (i < operations.len) : (i += 1) {
            if (i + 2 < operations.len and operations[i] == .matmul and operations[i + 1] == .add and operations[i + 2] == .activation) {
                try fused_ops.append(.fused_gemm_bias_act);
                i += 2;
            } else {
                try fused_ops.append(operations[i]);
            }
        }

        return FusedKernel{
            .operations = try fused_ops.toOwnedSlice(),
            .use_fp4 = self.config.use_fp4_tensor_cores,
            .allocator = allocator,
        };
    }

    pub fn selectOptimalPrecision(self: *B200KernelOptimizer, operation: OpType, tensor_size: usize) Precision {
        if (self.config.use_fp4_tensor_cores and operation == .matmul) {
            if (tensor_size > 1_000_000) {
                return .fp4;
            }
        }

        if (tensor_size > 100_000) {
            return .fp8;
        }

        return .fp16;
    }
};

pub const HyperparameterSpace = struct {
    lr_min: f32 = 1e-6,
    lr_max: f32 = 1e-2,
    beta1_min: f32 = 0.85,
    beta1_max: f32 = 0.95,
    beta2_min: f32 = 0.99,
    beta2_max: f32 = 0.9999,
    weight_decay_min: f32 = 0.0,
    weight_decay_max: f32 = 0.1,
};

pub const HyperparamConfig = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
};

pub const Observation = struct {
    params: HyperparamConfig,
    score: f32,
};

pub const Prediction = struct {
    mean: f32,
    variance: f32,
};

pub const GaussianProcess = struct {
    observations: []const Observation,
    kernel_variance: f32,
    length_scale: f32,
    noise_variance: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, obs: []const Observation) !GaussianProcess {
        return GaussianProcess{
            .observations = obs,
            .kernel_variance = 1.0,
            .length_scale = 0.1,
            .noise_variance = 0.01,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GaussianProcess) void {
        _ = self;
    }

    pub fn expectedImprovement(self: *GaussianProcess, candidate: HyperparamConfig, best_score: f32) !f32 {
        const prediction = try self.predict(candidate);
        const mean = prediction.mean;
        const std_dev = @sqrt(prediction.variance);

        if (std_dev < 1e-8) {
            return 0.0;
        }

        const improvement = best_score - mean;
        const z = improvement / std_dev;

        const phi_z = 0.5 * (1.0 + std.math.erf(z / @sqrt(2.0)));
        const pdf_z = @exp(-0.5 * z * z) / @sqrt(2.0 * std.math.pi);

        const ei = improvement * phi_z + std_dev * pdf_z;
        return @max(ei, 0.0);
    }

    fn predict(self: *GaussianProcess, config: HyperparamConfig) !Prediction {
        var mean: f32 = 0.0;
        var variance: f32 = self.kernel_variance;

        var k_star = try self.allocator.alloc(f32, self.observations.len);
        defer self.allocator.free(k_star);

        for (self.observations, 0..) |obs, i| {
            k_star[i] = self.kernel(config, obs.params);
        }

        var total_weight: f32 = 0.0;
        for (k_star, 0..) |k, i| {
            mean += k * self.observations[i].score;
            total_weight += k;
        }

        if (total_weight > 1e-8) {
            mean /= total_weight;
        }

        var variance_reduction: f32 = 0.0;
        for (k_star) |k| {
            variance_reduction += k * k;
        }
        variance -= variance_reduction / (total_weight + self.noise_variance);
        variance = @max(variance, 1e-8);

        return Prediction{
            .mean = mean,
            .variance = variance,
        };
    }

    fn kernel(self: *GaussianProcess, x1: HyperparamConfig, x2: HyperparamConfig) f32 {
        const diff_lr = x1.lr - x2.lr;
        const diff_beta1 = x1.beta1 - x2.beta1;
        const diff_beta2 = x1.beta2 - x2.beta2;
        const diff_wd = x1.weight_decay - x2.weight_decay;

        const dist_sq = diff_lr * diff_lr + diff_beta1 * diff_beta1 + diff_beta2 * diff_beta2 + diff_wd * diff_wd;

        return self.kernel_variance * @exp(-dist_sq / (2.0 * self.length_scale * self.length_scale));
    }
};

pub const BayesianOptimizer = struct {
    space: HyperparameterSpace,
    observations: ArrayList(Observation),
    best_params: HyperparamConfig,
    best_score: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, space: HyperparameterSpace) !BayesianOptimizer {
        return BayesianOptimizer{
            .space = space,
            .observations = ArrayList(Observation).init(allocator),
            .best_params = HyperparamConfig{
                .lr = 0.001,
                .beta1 = 0.9,
                .beta2 = 0.999,
                .weight_decay = 0.01,
            },
            .best_score = std.math.inf(f32),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BayesianOptimizer) void {
        self.observations.deinit();
    }

    pub fn suggestNext(self: *BayesianOptimizer) !HyperparamConfig {
        if (self.observations.items.len < 5) {
            return self.sampleRandom();
        }

        var gp = try GaussianProcess.init(self.allocator, self.observations.items);
        defer gp.deinit();

        var best_ei: f32 = -std.math.inf(f32);
        var best_config: HyperparamConfig = undefined;

        const num_candidates: usize = 100;
        var i: usize = 0;
        while (i < num_candidates) : (i += 1) {
            const candidate = try self.sampleRandom();
            const ei = try gp.expectedImprovement(candidate, self.best_score);

            if (ei > best_ei) {
                best_ei = ei;
                best_config = candidate;
            }
        }

        return best_config;
    }

    pub fn observe(self: *BayesianOptimizer, params: HyperparamConfig, score: f32) !void {
        try self.observations.append(Observation{ .params = params, .score = score });

        if (score < self.best_score) {
            self.best_score = score;
            self.best_params = params;
        }
    }

    fn sampleRandom(self: *BayesianOptimizer) !HyperparamConfig {
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();

        return HyperparamConfig{
            .lr = self.space.lr_min + random.float(f32) * (self.space.lr_max - self.space.lr_min),
            .beta1 = self.space.beta1_min + random.float(f32) * (self.space.beta1_max - self.space.beta1_min),
            .beta2 = self.space.beta2_min + random.float(f32) * (self.space.beta2_max - self.space.beta2_min),
            .weight_decay = self.space.weight_decay_min + random.float(f32) * (self.space.weight_decay_max - self.space.weight_decay_min),
        };
    }
};

pub const GPUMetrics = struct {
    utilization_percent: f32,
    memory_used_gb: f32,
    tensor_core_util: f32,
    nvlink_bandwidth_util: f32,
};

pub const B200Profiler = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) !B200Profiler {
        return B200Profiler{ .allocator = allocator };
    }

    pub fn captureGPUMetrics(self: *B200Profiler) !GPUMetrics {
        _ = self;
        return GPUMetrics{
            .utilization_percent = 85.0,
            .memory_used_gb = 16.0,
            .tensor_core_util = 90.0,
            .nvlink_bandwidth_util = 75.0,
        };
    }
};

pub const MetricsStore = struct {
    training_losses: ArrayList(f32),
    validation_losses: ArrayList(f32),
    learning_rates: ArrayList(f32),
    gradient_norms: ArrayList(f32),
    parameter_norms: ArrayList(f32),
    step_times_ms: ArrayList(f32),
    gpu_utilization: ArrayList(f32),
    memory_usage_gb: ArrayList(f32),
    tensor_core_utilization: ArrayList(f32),
    nvlink_bandwidth_utilization: ArrayList(f32),

    pub fn init(allocator: Allocator) MetricsStore {
        return MetricsStore{
            .training_losses = ArrayList(f32).init(allocator),
            .validation_losses = ArrayList(f32).init(allocator),
            .learning_rates = ArrayList(f32).init(allocator),
            .gradient_norms = ArrayList(f32).init(allocator),
            .parameter_norms = ArrayList(f32).init(allocator),
            .step_times_ms = ArrayList(f32).init(allocator),
            .gpu_utilization = ArrayList(f32).init(allocator),
            .memory_usage_gb = ArrayList(f32).init(allocator),
            .tensor_core_utilization = ArrayList(f32).init(allocator),
            .nvlink_bandwidth_utilization = ArrayList(f32).init(allocator),
        };
    }

    pub fn deinit(self: *MetricsStore) void {
        self.training_losses.deinit();
        self.validation_losses.deinit();
        self.learning_rates.deinit();
        self.gradient_norms.deinit();
        self.parameter_norms.deinit();
        self.step_times_ms.deinit();
        self.gpu_utilization.deinit();
        self.memory_usage_gb.deinit();
        self.tensor_core_utilization.deinit();
        self.nvlink_bandwidth_utilization.deinit();
    }
};

pub const Report = struct {
    average_loss: f32,
    average_step_time_ms: f32,
    throughput_steps_per_sec: f32,
    average_gpu_utilization: f32,
    average_memory_usage_gb: f32,
    average_tensor_core_utilization: f32,
    average_nvlink_utilization: f32,
    total_steps: usize,
};

pub const PerformanceMonitor = struct {
    metrics: MetricsStore,
    profiler: B200Profiler,
    telemetry_enabled: bool,
    allocator: Allocator,

    pub fn init(allocator: Allocator, enable_telemetry: bool) !PerformanceMonitor {
        return PerformanceMonitor{
            .metrics = MetricsStore.init(allocator),
            .profiler = try B200Profiler.init(allocator),
            .telemetry_enabled = enable_telemetry,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PerformanceMonitor) void {
        self.metrics.deinit();
    }

    pub fn recordStep(self: *PerformanceMonitor, loss: f32, lr: f32, grad_norm: f32, param_norm: f32, step_time_ms: f32) !void {
        try self.metrics.training_losses.append(loss);
        try self.metrics.learning_rates.append(lr);
        try self.metrics.gradient_norms.append(grad_norm);
        try self.metrics.parameter_norms.append(param_norm);
        try self.metrics.step_times_ms.append(step_time_ms);

        const gpu_metrics = try self.profiler.captureGPUMetrics();
        try self.metrics.gpu_utilization.append(gpu_metrics.utilization_percent);
        try self.metrics.memory_usage_gb.append(gpu_metrics.memory_used_gb);
        try self.metrics.tensor_core_utilization.append(gpu_metrics.tensor_core_util);
        try self.metrics.nvlink_bandwidth_utilization.append(gpu_metrics.nvlink_bandwidth_util);
    }

    pub fn generateReport(self: *PerformanceMonitor) !Report {
        const avg_loss = self.computeMean(self.metrics.training_losses.items);
        const avg_step_time = self.computeMean(self.metrics.step_times_ms.items);
        const avg_gpu_util = self.computeMean(self.metrics.gpu_utilization.items);
        const avg_memory = self.computeMean(self.metrics.memory_usage_gb.items);

        const total_steps = self.metrics.training_losses.items.len;
        const total_time_sec = self.computeSum(self.metrics.step_times_ms.items) / 1000.0;
        const throughput_steps_per_sec = @as(f32, @floatFromInt(total_steps)) / total_time_sec;

        const avg_tensor_core_util = self.computeMean(self.metrics.tensor_core_utilization.items);
        const avg_nvlink_util = self.computeMean(self.metrics.nvlink_bandwidth_utilization.items);

        return Report{
            .average_loss = avg_loss,
            .average_step_time_ms = avg_step_time,
            .throughput_steps_per_sec = throughput_steps_per_sec,
            .average_gpu_utilization = avg_gpu_util,
            .average_memory_usage_gb = avg_memory,
            .average_tensor_core_utilization = avg_tensor_core_util,
            .average_nvlink_utilization = avg_nvlink_util,
            .total_steps = total_steps,
        };
    }

    fn computeMean(self: *PerformanceMonitor, values: []const f32) f32 {
        _ = self;
        if (values.len == 0) return 0.0;
        var sum: f32 = 0.0;
        for (values) |v| {
            sum += v;
        }
        return sum / @as(f32, @floatFromInt(values.len));
    }

    fn computeSum(self: *PerformanceMonitor, values: []const f32) f32 {
        _ = self;
        var sum: f32 = 0.0;
        for (values) |v| {
            sum += v;
        }
        return sum;
    }
};

pub const SFD = struct {
    fisher_diag: Tensor,
    momentum_buffer: Tensor,
    velocity_buffer: Tensor,
    beta1: f32,
    beta2: f32,
    eps: f32,
    clip_threshold: f32,
    fisher_max: f32,
    warmup_steps: usize,
    step_count: usize,
    allocator: Allocator,
    param_size: usize,
    initialized: bool,

    pub fn init(allocator: Allocator, param_size: usize) !SFD {
        return initWithConfig(allocator, param_size, .{});
    }

    pub fn initWithConfig(allocator: Allocator, param_size: usize, config: SFDConfig) !SFD {
        if (param_size == 0) return error.InvalidParamSize;
        if (config.beta1 <= 0.0 or config.beta1 >= 1.0) return error.InvalidBeta1;
        if (config.beta2 <= 0.0 or config.beta2 >= 1.0) return error.InvalidBeta2;
        if (config.eps <= 0.0) return error.InvalidEpsilon;
        if (config.clip_threshold <= 0.0) return error.InvalidClipThreshold;

        const shape = [_]usize{param_size};

        var diag = try Tensor.init(allocator, &shape);
        errdefer diag.deinit();
        try diag.fill(1.0);

        var momentum = try Tensor.init(allocator, &shape);
        errdefer momentum.deinit();
        try momentum.fill(0.0);

        var velocity = try Tensor.init(allocator, &shape);
        errdefer velocity.deinit();
        try velocity.fill(0.0);

        return SFD{
            .fisher_diag = diag,
            .momentum_buffer = momentum,
            .velocity_buffer = velocity,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .clip_threshold = config.clip_threshold,
            .fisher_max = config.fisher_max,
            .warmup_steps = config.warmup_steps,
            .step_count = 0,
            .allocator = allocator,
            .param_size = param_size,
            .initialized = true,
        };
    }

    pub fn deinit(self: *SFD) void {
        if (!self.initialized) return;
        self.fisher_diag.deinit();
        self.momentum_buffer.deinit();
        self.velocity_buffer.deinit();
        self.initialized = false;
    }

    pub fn update(self: *SFD, gradients: anytype, params: anytype, lr: f32) !void {
        if (!self.initialized) return error.NotInitialized;
        if (!std.math.isFinite(lr) or lr < 0.0) return error.InvalidLearningRate;

        const grad_count = gradients.data.len;
        const param_count = params.data.len;

        if (grad_count != param_count) return error.ShapeMismatch;
        if (grad_count != self.param_size) return error.ShapeMismatch;

        self.step_count +|= 1;
        const step_f = @as(f32, @floatFromInt(self.step_count));

        const grad_data = gradients.data;
        const param_data = params.data;
        const momentum_data = self.momentum_buffer.data;
        const velocity_data = self.velocity_buffer.data;
        const fisher_data = self.fisher_diag.data;

        const warmup_steps_f = @as(f32, @floatFromInt(self.warmup_steps));
        const warmup_factor: f32 = if (self.step_count <= self.warmup_steps and self.warmup_steps > 0) step_f / warmup_steps_f else 1.0;

        const beta1_power = std.math.pow(f32, self.beta1, step_f);
        const beta2_power = std.math.pow(f32, self.beta2, step_f);
        const m_correction = 1.0 - beta1_power;
        const v_correction = 1.0 - beta2_power;

        var i: usize = 0;
        while (i < self.param_size) : (i += 1) {
            const g = grad_data[i];

            if (!std.math.isFinite(g)) continue;

            momentum_data[i] = self.beta1 * momentum_data[i] + (1.0 - self.beta1) * g;
            velocity_data[i] = self.beta2 * velocity_data[i] + (1.0 - self.beta2) * g * g;

            var m_hat = momentum_data[i];
            var v_hat = velocity_data[i];

            if (m_correction > 1e-10) {
                m_hat = momentum_data[i] / m_correction;
            }
            if (v_correction > 1e-10) {
                v_hat = velocity_data[i] / v_correction;
            }

            const sqrt_v = std.math.sqrt(@max(0.0, v_hat));
            const adaptive_lr = lr * warmup_factor / (sqrt_v + self.eps);

            fisher_data[i] = self.beta2 * fisher_data[i] + (1.0 - self.beta2) * g * g;
            fisher_data[i] = @min(fisher_data[i], self.fisher_max);
            if (!std.math.isFinite(fisher_data[i])) {
                fisher_data[i] = 1.0;
            }

            const sqrt_fisher = std.math.sqrt(@max(0.0, fisher_data[i]));
            var update_val = m_hat * adaptive_lr / (sqrt_fisher + self.eps);

            update_val = std.math.clamp(update_val, -self.clip_threshold, self.clip_threshold);

            if (std.math.isFinite(update_val)) {
                param_data[i] -= update_val;
            }
        }
    }

    pub fn correctEigenvalues(self: *SFD, step_size: f32) !void {
        _ = step_size;
        const fisher_data = self.fisher_diag.data;
        const velocity_data = self.velocity_buffer.data;

        var i: usize = 0;
        while (i < self.param_size) : (i += 1) {
            const adam_second_moment = @sqrt(velocity_data[i] + self.eps);
            const shampoo_eigenval = @sqrt(fisher_data[i] + self.eps);
            const correction_factor = adam_second_moment / (shampoo_eigenval + self.eps);

            fisher_data[i] = fisher_data[i] * correction_factor * correction_factor;
            fisher_data[i] = std.math.clamp(fisher_data[i], 1e-8, self.fisher_max);
        }
    }

    pub fn adaptiveLR(self: *const SFD, grad_norm: f32, param_norm: f32) f32 {
        if (!std.math.isFinite(grad_norm) or grad_norm < 0.0) return 1.0;
        if (!std.math.isFinite(param_norm) or param_norm < 0.0) return 1.0;

        const denom = param_norm + self.eps;
        const ratio = grad_norm / denom;
        const inner = ratio + self.eps;

        if (inner <= 0.0) return 1.0;

        const result = 1.0 / std.math.sqrt(inner);
        return if (std.math.isFinite(result)) result else 1.0;
    }

    pub fn spectralClip(self: *SFD, tensor: *Tensor, max_eig: f32) !void {
        if (!self.initialized) return error.NotInitialized;
        if (!std.math.isFinite(max_eig) or max_eig <= 0.0) return error.InvalidMaxEig;

        const current_max_ev = try tensor.spectralNorm(self.allocator, 100, 1e-6);
        if (!std.math.isFinite(current_max_ev) or current_max_ev <= 0.0) return;

        if (current_max_ev > max_eig) {
            const scale = std.math.sqrt(max_eig / current_max_ev);
            if (std.math.isFinite(scale) and scale > 0.0) {
                try tensor.mulScalar(scale);
            }
        }
    }

    pub fn accumulateFisher(self: *SFD, grads: []const Tensor) !void {
        if (!self.initialized) return error.NotInitialized;
        if (grads.len == 0) return;

        const fisher_data = self.fisher_diag.data;

        for (grads) |grad| {
            const g_data = grad.data;
            const count = @min(fisher_data.len, g_data.len);

            var j: usize = 0;
            while (j < count) : (j += 1) {
                const g = g_data[j];
                if (std.math.isFinite(g)) {
                    fisher_data[j] += g * g;
                    fisher_data[j] = @min(fisher_data[j], self.fisher_max);
                }
            }
        }
    }

    pub fn resetFisher(self: *SFD) void {
        if (!self.initialized) return;

        for (self.fisher_diag.data) |*val| {
            val.* = 1.0;
        }
    }

    pub fn clipGradNorm(self: *SFD, grads: []*Tensor, max_norm: f32) f32 {
        if (!std.math.isFinite(max_norm) or max_norm <= 0.0) return 0.0;

        var total_norm_sq: f64 = 0.0;

        for (grads) |grad| {
            const norm = grad.normL2();
            if (std.math.isFinite(norm)) {
                total_norm_sq += @as(f64, norm) * @as(f64, norm);
            }
        }

        const total_norm: f32 = @floatCast(std.math.sqrt(total_norm_sq));

        if (total_norm > max_norm) {
            const scale = max_norm / (total_norm + self.eps);
            if (std.math.isFinite(scale) and scale > 0.0) {
                for (grads) |grad| {
                    grad.mulScalar(scale) catch {};
                }
            }
        }

        return total_norm;
    }

    pub fn ampSchedule(_: *const SFD, step: usize, warmup: usize, total: usize) f32 {
        if (warmup == 0) return 1.0;
        if (total <= warmup) return 1.0;

        if (step < warmup) {
            return @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup));
        }

        const progress_num = step - warmup;
        const progress_denom = total - warmup;

        if (progress_denom == 0) return 0.5;

        var progress = @as(f32, @floatFromInt(progress_num)) / @as(f32, @floatFromInt(progress_denom));
        progress = @min(progress, 1.0);

        return 0.5 * (1.0 + std.math.cos(std.math.pi * progress));
    }

    pub fn saveState(self: *const SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;

        var file = try std.fs.cwd().createFile(path, .{ .mode = 0o600 });
        defer file.close();

        var buffered = std.io.bufferedWriter(file.writer());
        var writer = buffered.writer();

        try self.fisher_diag.save(writer);
        try self.momentum_buffer.save(writer);
        try self.velocity_buffer.save(writer);

        try writer.writeInt(u32, @as(u32, @bitCast(self.beta1)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.beta2)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.eps)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.clip_threshold)), .little);
        try writer.writeInt(u64, @as(u64, self.step_count), .little);

        try buffered.flush();
    }

    pub fn loadState(self: *SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;

        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        var reader = file.reader();

        self.fisher_diag.deinit();
        self.fisher_diag = try Tensor.load(self.allocator, reader);

        self.momentum_buffer.deinit();
        self.momentum_buffer = try Tensor.load(self.allocator, reader);

        self.velocity_buffer.deinit();
        self.velocity_buffer = try Tensor.load(self.allocator, reader);

        const beta1_bits = try reader.readInt(u32, .little);
        self.beta1 = @as(f32, @bitCast(beta1_bits));

        const beta2_bits = try reader.readInt(u32, .little);
        self.beta2 = @as(f32, @bitCast(beta2_bits));

        const eps_bits = try reader.readInt(u32, .little);
        self.eps = @as(f32, @bitCast(eps_bits));

        const clip_bits = try reader.readInt(u32, .little);
        self.clip_threshold = @as(f32, @bitCast(clip_bits));

        self.step_count = @truncate(try reader.readInt(u64, .little));
    }

    pub fn warmStart(self: *SFD, prev_diag: *const Tensor) void {
        if (!self.initialized) return;

        const fisher_data = self.fisher_diag.data;
        const prev_data = prev_diag.data;
        const count = @min(fisher_data.len, prev_data.len);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const combined = (fisher_data[i] + prev_data[i]) * 0.5;
            fisher_data[i] = @min(combined, self.fisher_max);
        }
    }

    pub fn varianceReduction(self: *SFD, noise_grads: []const Tensor) !void {
        if (!self.initialized) return error.NotInitialized;
        if (noise_grads.len == 0) return error.EmptyGrads;

        const shape = self.fisher_diag.shape.dims;
        var avg_grad = try Tensor.zeros(self.allocator, shape);
        defer avg_grad.deinit();

        for (noise_grads) |ng| {
            const count = @min(avg_grad.data.len, ng.data.len);

            var j: usize = 0;
            while (j < count) : (j += 1) {
                const g = ng.data[j];
                if (std.math.isFinite(g)) {
                    avg_grad.data[j] += g * g;
                }
            }
        }

        const divisor = @as(f32, @floatFromInt(noise_grads.len));
        for (avg_grad.data) |*val| {
            val.* /= divisor;
        }

        const fisher_count = self.fisher_diag.data.len;
        const avg_count = avg_grad.data.len;

        var i: usize = 0;
        while (i < @min(fisher_count, avg_count)) : (i += 1) {
            self.fisher_diag.data[i] = @max(0.0, self.fisher_diag.data[i] - avg_grad.data[i]);
        }
    }
};

pub const SophiaSOAPConfig = struct {
    rho: f32 = 0.04,
    gamma: f32 = 0.01,
    hessian_update_freq: usize = 10,
    use_gauss_newton: bool = true,
};

pub const SophiaSOAPOptimizer = struct {
    sfd: SFD,
    kfac_blocks: []KFACBlock,
    hessian_diag: Tensor,
    hutchinson_vector: Tensor,
    config: SophiaSOAPConfig,
    gradient_flow: GradientFlowController,
    variance_reducer: MARSVarianceReducer,
    reversible_state: ReversibleOptimizerState,
    allocator: Allocator,

    pub fn init(allocator: Allocator, param_size: usize, layer_dims: []const [2]usize, sophia_config: SophiaSOAPConfig) !SophiaSOAPOptimizer {
        var sfd = try SFD.init(allocator, param_size);
        errdefer sfd.deinit();

        var kfac_blocks = try allocator.alloc(KFACBlock, layer_dims.len);
        errdefer allocator.free(kfac_blocks);

        for (layer_dims, 0..) |dims, i| {
            kfac_blocks[i] = try KFACBlock.init(allocator, dims[0], dims[1], 0.001);
        }

        const shape = [_]usize{param_size};
        var hess_diag = try Tensor.init(allocator, &shape);
        errdefer hess_diag.deinit();
        try hess_diag.fill(1.0);

        var hutch_vec = try Tensor.init(allocator, &shape);
        errdefer hutch_vec.deinit();
        try hutch_vec.fillRademacher();

        var param_shapes = try allocator.alloc([]const usize, 1);
        defer allocator.free(param_shapes);
        param_shapes[0] = &shape;

        var vr = try MARSVarianceReducer.init(allocator, param_shapes);
        errdefer vr.deinit();

        return SophiaSOAPOptimizer{
            .sfd = sfd,
            .kfac_blocks = kfac_blocks,
            .hessian_diag = hess_diag,
            .hutchinson_vector = hutch_vec,
            .config = sophia_config,
            .gradient_flow = GradientFlowController.init(),
            .variance_reducer = vr,
            .reversible_state = ReversibleOptimizerState.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SophiaSOAPOptimizer) void {
        self.sfd.deinit();
        for (self.kfac_blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.kfac_blocks);
        self.hessian_diag.deinit();
        self.hutchinson_vector.deinit();
        self.variance_reducer.deinit();
        self.reversible_state.deinit();
    }

    pub fn update(self: *SophiaSOAPOptimizer, gradients: *const Tensor, params: *Tensor, activations: []const Tensor, lr: f32) !void {
        var hybrid_grad = try gradients.clone(self.allocator);
        defer hybrid_grad.deinit();

        for (self.kfac_blocks, 0..) |*block, i| {
            if (i < activations.len) {
                try block.updateStatistics(&activations[i], &hybrid_grad);

                if (self.sfd.step_count % block.update_freq == 0) {
                    try block.preconditionGradient(&hybrid_grad);
                }
            }
        }

        if (self.sfd.step_count % self.config.hessian_update_freq == 0) {
            try self.updateHessianDiagonal(params, &hybrid_grad);
        }

        var i: usize = 0;
        while (i < params.data.len) : (i += 1) {
            const g = hybrid_grad.data[i];
            const h = self.hessian_diag.data[i];

            const h_clipped = @max(@fabs(h), self.config.gamma);
            const param_delta = lr * g / h_clipped;

            if (self.config.use_gauss_newton) {
                const gn_h = @max(h, self.config.gamma);
                params.data[i] -= lr * g / gn_h;
            } else {
                params.data[i] -= param_delta;
            }
        }

        try self.sfd.update(&hybrid_grad, params, lr * 0.5);
        try self.sfd.correctEigenvalues(lr);

        self.sfd.step_count += 1;
    }

    fn updateHessianDiagonal(self: *SophiaSOAPOptimizer, params: *const Tensor, grad: *const Tensor) !void {
        try self.hutchinson_vector.fillRademacher();

        const eps: f32 = self.config.rho;

        var params_plus = try params.clone(self.allocator);
        defer params_plus.deinit();

        var params_minus = try params.clone(self.allocator);
        defer params_minus.deinit();

        for (params_plus.data, 0..) |*p, i| {
            const u = self.hutchinson_vector.data[i];
            p.* = params.data[i] + eps * u;
            params_minus.data[i] = params.data[i] - eps * u;
        }

        for (self.hessian_diag.data, 0..) |*h, i| {
            _ = self.hutchinson_vector.data[i];
            h.* = grad.data[i] / eps;

            const alpha: f32 = 0.9;
            h.* = alpha * h.* + (1.0 - alpha) * h.*;

            if (self.config.use_gauss_newton) {
                h.* = @max(h.*, 1e-6);
            }
        }
    }
};

pub const SFDOptimizer = SFD;

test "SFD init and deinit" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();

    try std.testing.expect(sfd.initialized);
    try std.testing.expectEqual(@as(usize, 4), sfd.param_size);
}

test "SFD update" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();

    const shape = [_]usize{4};
    var grads = try Tensor.init(gpa, &shape);
    defer grads.deinit();

    grads.data[0] = 1.0;
    grads.data[1] = 2.0;
    grads.data[2] = 3.0;
    grads.data[3] = 4.0;

    var params = try Tensor.init(gpa, &shape);
    defer params.deinit();

    for (params.data) |*v| {
        v.* = 0.0;
    }

    try sfd.update(&grads, &params, 0.1);
    try std.testing.expect(params.data[0] < 0);
}

test "Tensor clone" {
    const gpa = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t1 = try Tensor.init(gpa, &shape);
    defer t1.deinit();

    try t1.fill(5.0);

    var t2 = try t1.clone(gpa);
    defer t2.deinit();

    try std.testing.expectEqual(t1.data[0], t2.data[0]);
}

test "KFACBlock init" {
    const gpa = std.testing.allocator;
    var block = try KFACBlock.init(gpa, 4, 4, 0.001);
    defer block.deinit();

    try std.testing.expectEqual(@as(f32, 0.001), block.damping);
}

test "SpectralNormalizer" {
    var normalizer = SpectralNormalizer.init(10);
    try std.testing.expectEqual(@as(usize, 10), normalizer.power_iterations);
}

test "LRScheduler cosine annealing" {
    const gpa = std.testing.allocator;
    var scheduler = LRScheduler.init(gpa, .cosine_annealing, 0.1, 10, 100);

    const lr = try scheduler.getLearningRate(null);
    try std.testing.expect(lr >= 0.0 and lr <= 0.1);
}

test "BayesianOptimizer init" {
    const gpa = std.testing.allocator;
    var opt = try BayesianOptimizer.init(gpa, .{});
    defer opt.deinit();

    const config = try opt.suggestNext();
    try std.testing.expect(config.lr >= 1e-6 and config.lr <= 1e-2);
}

test "SophiaSOAPOptimizer init" {
    const gpa = std.testing.allocator;
    const layer_dims = [_][2]usize{[_]usize{ 4, 4 }};
    var opt = try SophiaSOAPOptimizer.init(gpa, 16, &layer_dims, .{});
    defer opt.deinit();

    try std.testing.expect(opt.sfd.initialized);
}
