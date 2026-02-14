const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../core/tensor.zig").Tensor;

pub const RSFLayerConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    seed_offset: u64 = 0,
    grad_mean: bool = true,
};

pub const RSFLayer = struct {
    s_weight: Tensor,
    t_weight: Tensor,
    s_bias: Tensor,
    t_bias: Tensor,
    s_weight_grad: Tensor,
    t_weight_grad: Tensor,
    s_bias_grad: Tensor,
    t_bias_grad: Tensor,
    dim: usize,
    allocator: Allocator,
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,

    pub fn init(allocator: Allocator, dim: usize) !RSFLayer {
        return initWithConfig(allocator, dim, .{});
    }

    pub fn initWithConfig(allocator: Allocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        if (dim == 0) return error.InvalidDimension;
        if (!std.math.isFinite(config.clip_min) or !std.math.isFinite(config.clip_max)) return error.NonFinite;
        if (!(config.clip_min < config.clip_max)) return error.InvalidConfig;

        const fan_in = @as(f32, @floatFromInt(dim));
        const fan_out = @as(f32, @floatFromInt(dim));
        const fan_sum = fan_in + fan_out;
        if (!(fan_sum > 0.0)) return error.InvalidDimension;

        const xavier_bound: f32 = std.math.sqrt(@as(f32, 6.0) / fan_sum);

        const weight_shape = [_]usize{ dim, dim };
        const bias_shape = [_]usize{ 1, dim };

        const seed1 = try std.math.add(u64, 42, config.seed_offset);
        const seed2 = try std.math.add(u64, 43, config.seed_offset);

        var s_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed1);
        errdefer s_w.deinit();

        var t_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed2);
        errdefer t_w.deinit();

        var s_b = try Tensor.zeros(allocator, &bias_shape);
        errdefer s_b.deinit();

        var t_b = try Tensor.zeros(allocator, &bias_shape);
        errdefer t_b.deinit();

        var s_w_grad = try Tensor.zeros(allocator, &weight_shape);
        errdefer s_w_grad.deinit();

        var t_w_grad = try Tensor.zeros(allocator, &weight_shape);
        errdefer t_w_grad.deinit();

        var s_b_grad = try Tensor.zeros(allocator, &bias_shape);
        errdefer s_b_grad.deinit();

        var t_b_grad = try Tensor.zeros(allocator, &bias_shape);
        errdefer t_b_grad.deinit();

        return RSFLayer{
            .s_weight = s_w,
            .t_weight = t_w,
            .s_bias = s_b,
            .t_bias = t_b,
            .s_weight_grad = s_w_grad,
            .t_weight_grad = t_w_grad,
            .s_bias_grad = s_b_grad,
            .t_bias_grad = t_b_grad,
            .dim = dim,
            .allocator = allocator,
            .clip_min = config.clip_min,
            .clip_max = config.clip_max,
            .grad_mean = config.grad_mean,
        };
    }

    pub fn deinit(self: *RSFLayer) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        self.s_bias.deinit();
        self.t_bias.deinit();
        self.s_weight_grad.deinit();
        self.t_weight_grad.deinit();
        self.s_bias_grad.deinit();
        self.t_bias_grad.deinit();
    }

    pub fn zeroGradients(self: *RSFLayer) void {
        for (self.s_weight_grad.data) |*v| v.* = 0.0;
        for (self.t_weight_grad.data) |*v| v.* = 0.0;
        for (self.s_bias_grad.data) |*v| v.* = 0.0;
        for (self.t_bias_grad.data) |*v| v.* = 0.0;
    }

    fn assert2DAndLen(t: *const Tensor) !void {
        if (t.shape.dims.len != 2) return error.ShapeMismatch;
        const rows = t.shape.dims[0];
        const cols = t.shape.dims[1];
        const expected = rows * cols;
        if (t.data.len != expected) return error.DataLengthMismatch;
    }

    fn assertPair(self: *const RSFLayer, a: *const Tensor, b: *const Tensor) !usize {
        try assert2DAndLen(a);
        try assert2DAndLen(b);
        if (a.shape.dims[1] != self.dim or b.shape.dims[1] != self.dim) return error.ShapeMismatch;
        if (a.shape.dims[0] != b.shape.dims[0]) return error.ShapeMismatch;
        if (self.s_bias.data.len != self.dim or self.t_bias.data.len != self.dim) return error.ShapeMismatch;
        if (!std.math.isFinite(self.clip_min) or !std.math.isFinite(self.clip_max)) return error.NonFinite;
        if (!(self.clip_min < self.clip_max)) return error.InvalidConfig;
        return a.shape.dims[0];
    }

    fn clampFiniteInPlace(t: *Tensor) void {
        for (t.data) |*v| {
            if (!std.math.isFinite(v.*)) v.* = 0.0;
        }
    }

    pub fn forward(self: *const RSFLayer, x1: *Tensor, x2: *Tensor) !void {
        const batch_size = try self.assertPair(x1, x2);

        var x2_t = try x2.transpose(&.{ 1, 0 });
        defer x2_t.deinit();

        var s_x2_t = try self.s_weight.matmul(&x2_t, self.allocator);
        defer s_x2_t.deinit();

        var s_x2 = try s_x2_t.transpose(&.{ 1, 0 });
        defer s_x2.deinit();

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                s_x2.data[b * self.dim + d] += self.s_bias.data[d];
            }
        }

        clampFiniteInPlace(&s_x2);
        try s_x2.clip(self.clip_min, self.clip_max);
        clampFiniteInPlace(&s_x2);
        try s_x2.exp();
        clampFiniteInPlace(&s_x2);

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                x1.data[b * self.dim + d] *= s_x2.data[b * self.dim + d];
            }
        }

        var x1_t = try x1.transpose(&.{ 1, 0 });
        defer x1_t.deinit();

        var t_x1_t = try self.t_weight.matmul(&x1_t, self.allocator);
        defer t_x1_t.deinit();

        var t_x1 = try t_x1_t.transpose(&.{ 1, 0 });
        defer t_x1.deinit();

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                t_x1.data[b * self.dim + d] += self.t_bias.data[d];
            }
        }

        clampFiniteInPlace(&t_x1);
        try x2.add(&t_x1);
        clampFiniteInPlace(x2);
    }

    pub fn inverse(self: *const RSFLayer, y1: *Tensor, y2: *Tensor) !void {
        const batch_size = try self.assertPair(y1, y2);

        var y1_t = try y1.transpose(&.{ 1, 0 });
        defer y1_t.deinit();

        var t_y1_t = try self.t_weight.matmul(&y1_t, self.allocator);
        defer t_y1_t.deinit();

        var t_y1 = try t_y1_t.transpose(&.{ 1, 0 });
        defer t_y1.deinit();

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                t_y1.data[b * self.dim + d] += self.t_bias.data[d];
            }
        }

        clampFiniteInPlace(&t_y1);
        try y2.sub(&t_y1);
        clampFiniteInPlace(y2);

        var y2_t = try y2.transpose(&.{ 1, 0 });
        defer y2_t.deinit();

        var s_y2_t = try self.s_weight.matmul(&y2_t, self.allocator);
        defer s_y2_t.deinit();

        var s_y2 = try s_y2_t.transpose(&.{ 1, 0 });
        defer s_y2.deinit();

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                s_y2.data[b * self.dim + d] += self.s_bias.data[d];
            }
        }

        clampFiniteInPlace(&s_y2);
        try s_y2.clip(self.clip_min, self.clip_max);
        clampFiniteInPlace(&s_y2);
        try s_y2.exp();
        clampFiniteInPlace(&s_y2);

        for (s_y2.data) |v| {
            if (v == 0.0 or !std.math.isFinite(v)) return error.NumericFailure;
        }

        try y1.div(&s_y2);
        clampFiniteInPlace(y1);
    }

    pub fn backward(
        self: *RSFLayer,
        y1: *const Tensor,
        y2: *const Tensor,
        dy1_in: *const Tensor,
        dy2_in: *const Tensor,
        x1_out: *Tensor,
        x2_out: *Tensor,
        dx1_out: *Tensor,
        dx2_out: *Tensor,
    ) !void {
        const batch_size = try self.assertPair(y1, y2);
        try self.assertPair(dy1_in, dy2_in);

        if (x1_out.shape.dims.len != 2 or x2_out.shape.dims.len != 2 or dx1_out.shape.dims.len != 2 or dx2_out.shape.dims.len != 2) return error.ShapeMismatch;
        if (x1_out.shape.dims[0] != batch_size or x2_out.shape.dims[0] != batch_size or dx1_out.shape.dims[0] != batch_size or dx2_out.shape.dims[0] != batch_size) return error.ShapeMismatch;
        if (x1_out.shape.dims[1] != self.dim or x2_out.shape.dims[1] != self.dim or dx1_out.shape.dims[1] != self.dim or dx2_out.shape.dims[1] != self.dim) return error.ShapeMismatch;
        if (x1_out.data.len != batch_size * self.dim or x2_out.data.len != batch_size * self.dim or dx1_out.data.len != batch_size * self.dim or dx2_out.data.len != batch_size * self.dim) return error.DataLengthMismatch;

        var dy1 = try dy1_in.copy(self.allocator);
        defer dy1.deinit();

        var dy2 = try dy2_in.copy(self.allocator);
        defer dy2.deinit();

        var y1_t = try y1.transpose(&.{ 1, 0 });
        defer y1_t.deinit();

        var t_y1_t = try self.t_weight.matmul(&y1_t, self.allocator);
        defer t_y1_t.deinit();

        var t_y1 = try t_y1_t.transpose(&.{ 1, 0 });
        defer t_y1.deinit();

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                t_y1.data[b * self.dim + d] += self.t_bias.data[d];
            }
        }

        clampFiniteInPlace(&t_y1);

        var x2 = try y2.copy(self.allocator);
        defer x2.deinit();
        try x2.sub(&t_y1);
        clampFiniteInPlace(&x2);

        var dy2_t = try dy2.transpose(&.{ 1, 0 });
        defer dy2_t.deinit();

        var dt = try dy2_t.matmul(y1, self.allocator);
        defer dt.deinit();

        const grad_scale: f32 = if (self.grad_mean) (1.0 / @as(f32, @floatFromInt(batch_size))) else 1.0;

        var i: usize = 0;
        while (i < dt.data.len) : (i += 1) {
            self.t_weight_grad.data[i] += dt.data[i] * grad_scale;
        }

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                self.t_bias_grad.data[d] += dy2.data[b * self.dim + d] * grad_scale;
            }
        }

        var t_weight_t = try self.t_weight.transpose(&.{ 1, 0 });
        defer t_weight_t.deinit();

        var grad_from_t_t = try t_weight_t.matmul(&dy2_t, self.allocator);
        defer grad_from_t_t.deinit();

        var grad_from_t = try grad_from_t_t.transpose(&.{ 1, 0 });
        defer grad_from_t.deinit();

        try dy1.add(&grad_from_t);
        clampFiniteInPlace(&dy1);

        var x2_t = try x2.transpose(&.{ 1, 0 });
        defer x2_t.deinit();

        var s_pre_t = try self.s_weight.matmul(&x2_t, self.allocator);
        defer s_pre_t.deinit();

        var s_pre = try s_pre_t.transpose(&.{ 1, 0 });
        defer s_pre.deinit();

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                s_pre.data[b * self.dim + d] += self.s_bias.data[d];
            }
        }

        clampFiniteInPlace(&s_pre);

        var s_clipped = try s_pre.copy(self.allocator);
        defer s_clipped.deinit();
        try s_clipped.clip(self.clip_min, self.clip_max);
        clampFiniteInPlace(&s_clipped);

        var exp_s = try s_clipped.copy(self.allocator);
        defer exp_s.deinit();
        try exp_s.exp();
        clampFiniteInPlace(&exp_s);

        for (exp_s.data) |v| {
            if (v == 0.0 or !std.math.isFinite(v)) return error.NumericFailure;
        }

        var x1 = try y1.copy(self.allocator);
        defer x1.deinit();
        try x1.div(&exp_s);
        clampFiniteInPlace(&x1);

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                x1_out.data[b * self.dim + d] = x1.data[b * self.dim + d];
                x2_out.data[b * self.dim + d] = x2.data[b * self.dim + d];
            }
        }

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                dx1_out.data[b * self.dim + d] = dy1.data[b * self.dim + d] * exp_s.data[b * self.dim + d];
            }
        }

        var dscale = try dy1.copy(self.allocator);
        defer dscale.deinit();
        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                dscale.data[b * self.dim + d] *= x1.data[b * self.dim + d];
            }
        }

        var ds = try dscale.copy(self.allocator);
        defer ds.deinit();
        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                ds.data[b * self.dim + d] *= exp_s.data[b * self.dim + d];
            }
        }

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                const v = s_pre.data[b * self.dim + d];
                if (!std.math.isFinite(v) or v < self.clip_min or v > self.clip_max) {
                    ds.data[b * self.dim + d] = 0.0;
                }
            }
        }

        var ds_t = try ds.transpose(&.{ 1, 0 });
        defer ds_t.deinit();

        var ds_w = try ds_t.matmul(&x2, self.allocator);
        defer ds_w.deinit();

        i = 0;
        while (i < ds_w.data.len) : (i += 1) {
            self.s_weight_grad.data[i] += ds_w.data[i] * grad_scale;
        }

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                self.s_bias_grad.data[d] += ds.data[b * self.dim + d] * grad_scale;
            }
        }

        var s_weight_t = try self.s_weight.transpose(&.{ 1, 0 });
        defer s_weight_t.deinit();

        var grad_from_s_t = try s_weight_t.matmul(&ds_t, self.allocator);
        defer grad_from_s_t.deinit();

        var grad_from_s = try grad_from_s_t.transpose(&.{ 1, 0 });
        defer grad_from_s.deinit();

        b = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.dim) : (d += 1) {
                dx2_out.data[b * self.dim + d] = dy2.data[b * self.dim + d] + grad_from_s.data[b * self.dim + d];
            }
        }

        clampFiniteInPlace(dx1_out);
        clampFiniteInPlace(dx2_out);
    }
};

pub const RSFConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    max_dim: usize = 1 << 20,
    max_layers: usize = 1 << 20,
};

const ControlBlock = struct {
    freed: std.atomic.Atomic(u8) = std.atomic.Atomic(u8).init(0),
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    layers: []RSFLayer,
    cfg: RSFConfig,
};

pub const RSF = struct {
    ctrl: *ControlBlock,

    pub fn init(allocator: Allocator, dim: usize, num_layers: usize) !RSF {
        return initWithConfig(allocator, dim, num_layers, .{});
    }

    pub fn initWithConfig(allocator: Allocator, dim: usize, num_layers: usize, cfg: RSFConfig) !RSF {
        if (dim == 0) return error.InvalidDimension;
        if (num_layers == 0) return error.InvalidLayerCount;
        if (dim > cfg.max_dim or num_layers > cfg.max_layers) return error.TooLarge;
        if (!std.math.isFinite(cfg.clip_min) or !std.math.isFinite(cfg.clip_max)) return error.NonFinite;
        if (!(cfg.clip_min < cfg.clip_max)) return error.InvalidConfig;

        var ctrl = try allocator.create(ControlBlock);
        errdefer allocator.destroy(ctrl);

        ctrl.* = .{
            .allocator = allocator,
            .dim = dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(RSFLayer, num_layers),
            .cfg = cfg,
        };
        errdefer allocator.free(ctrl.layers);

        var initialized_count: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized_count) : (j += 1) {
                ctrl.layers[j].deinit();
            }
        }

        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            const layer_cfg = RSFLayerConfig{
                .clip_min = cfg.clip_min,
                .clip_max = cfg.clip_max,
                .seed_offset = @as(u64, @intCast(l)) * 1000,
                .grad_mean = cfg.grad_mean,
            };
            ctrl.layers[l] = try RSFLayer.initWithConfig(allocator, dim, layer_cfg);
            initialized_count += 1;
        }

        return RSF{ .ctrl = ctrl };
    }

    pub fn deinit(self: *RSF) void {
        if (self.ctrl.freed.swap(1, .SeqCst) != 0) return;

        var i: usize = 0;
        while (i < self.ctrl.num_layers) : (i += 1) {
            self.ctrl.layers[i].deinit();
        }
        self.ctrl.allocator.free(self.ctrl.layers);
        self.ctrl.allocator.destroy(self.ctrl);
    }

    fn assert2DAndLen(t: *const Tensor) !void {
        if (t.shape.dims.len != 2) return error.ShapeMismatch;
        const rows = t.shape.dims[0];
        const cols = t.shape.dims[1];
        const expected = rows * cols;
        if (t.data.len != expected) return error.DataLengthMismatch;
    }

    fn split(self: *const RSF, x: *const Tensor, x1: *Tensor, x2: *Tensor) !usize {
        try assert2DAndLen(x);
        if (x.shape.dims[1] != self.ctrl.dim * 2) return error.ShapeMismatch;
        const batch_size = x.shape.dims[0];

        if (x1.shape.dims.len != 2 or x2.shape.dims.len != 2) return error.ShapeMismatch;
        if (x1.shape.dims[0] != batch_size or x2.shape.dims[0] != batch_size) return error.ShapeMismatch;
        if (x1.shape.dims[1] != self.ctrl.dim or x2.shape.dims[1] != self.ctrl.dim) return error.ShapeMismatch;
        if (x1.data.len != batch_size * self.ctrl.dim or x2.data.len != batch_size * self.ctrl.dim) return error.DataLengthMismatch;

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.ctrl.dim) : (d += 1) {
                x1.data[b * self.ctrl.dim + d] = x.data[b * self.ctrl.dim * 2 + d];
                x2.data[b * self.ctrl.dim + d] = x.data[b * self.ctrl.dim * 2 + self.ctrl.dim + d];
            }
        }

        return batch_size;
    }

    fn merge(self: *const RSF, x1: *const Tensor, x2: *const Tensor, out: *Tensor) !void {
        try assert2DAndLen(out);
        try assert2DAndLen(x1);
        try assert2DAndLen(x2);
        if (x1.shape.dims[0] != x2.shape.dims[0]) return error.ShapeMismatch;
        if (x1.shape.dims[1] != self.ctrl.dim or x2.shape.dims[1] != self.ctrl.dim) return error.ShapeMismatch;
        if (out.shape.dims[0] != x1.shape.dims[0] or out.shape.dims[1] != self.ctrl.dim * 2) return error.ShapeMismatch;

        const batch_size = x1.shape.dims[0];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            var d: usize = 0;
            while (d < self.ctrl.dim) : (d += 1) {
                out.data[b * self.ctrl.dim * 2 + d] = x1.data[b * self.ctrl.dim + d];
                out.data[b * self.ctrl.dim * 2 + self.ctrl.dim + d] = x2.data[b * self.ctrl.dim + d];
            }
        }
    }

    pub fn zeroGradients(self: *RSF) void {
        var i: usize = 0;
        while (i < self.ctrl.num_layers) : (i += 1) {
            self.ctrl.layers[i].zeroGradients();
        }
    }

    pub fn forward(self: *RSF, x: *Tensor) !void {
        if (self.ctrl.freed.load(.SeqCst) != 0) return error.NotInitialized;

        try assert2DAndLen(x);
        if (x.shape.dims[1] != self.ctrl.dim * 2) return error.ShapeMismatch;

        const batch_size = x.shape.dims[0];

        var shape_half = [_]usize{ batch_size, self.ctrl.dim };
        var x1 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer x1.deinit();
        var x2 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer x2.deinit();

        _ = try self.split(x, &x1, &x2);

        var i: usize = 0;
        while (i < self.ctrl.num_layers) : (i += 1) {
            try self.ctrl.layers[i].forward(&x1, &x2);
        }

        try self.merge(&x1, &x2, x);
    }

    pub fn inverse(self: *RSF, y: *Tensor) !void {
        if (self.ctrl.freed.load(.SeqCst) != 0) return error.NotInitialized;

        try assert2DAndLen(y);
        if (y.shape.dims[1] != self.ctrl.dim * 2) return error.ShapeMismatch;

        const batch_size = y.shape.dims[0];

        var shape_half = [_]usize{ batch_size, self.ctrl.dim };
        var y1 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer y1.deinit();
        var y2 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer y2.deinit();

        _ = try self.split(y, &y1, &y2);

        var idx = self.ctrl.num_layers;
        while (idx > 0) : (idx -= 1) {
            try self.ctrl.layers[idx - 1].inverse(&y1, &y2);
        }

        try self.merge(&y1, &y2, y);
    }

    pub fn backward(self: *RSF, grad_output: *const Tensor, input: *const Tensor, grad_input_out: *Tensor) !void {
        if (self.ctrl.freed.load(.SeqCst) != 0) return error.NotInitialized;

        try assert2DAndLen(grad_output);
        try assert2DAndLen(input);
        try assert2DAndLen(grad_input_out);

        if (input.shape.dims[1] != self.ctrl.dim * 2) return error.ShapeMismatch;
        if (grad_output.shape.dims[0] != input.shape.dims[0] or grad_output.shape.dims[1] != input.shape.dims[1]) return error.ShapeMismatch;
        if (grad_input_out.shape.dims[0] != input.shape.dims[0] or grad_input_out.shape.dims[1] != input.shape.dims[1]) return error.ShapeMismatch;

        const batch_size = input.shape.dims[0];
        var shape_half = [_]usize{ batch_size, self.ctrl.dim };

        var x = try input.copy(self.ctrl.allocator);
        defer x.deinit();
        try self.forward(&x);

        var y1 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer y1.deinit();
        var y2 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer y2.deinit();
        _ = try self.split(&x, &y1, &y2);

        var dy = try grad_output.copy(self.ctrl.allocator);
        defer dy.deinit();

        var dy1 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer dy1.deinit();
        var dy2 = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer dy2.deinit();
        _ = try self.split(&dy, &dy1, &dy2);

        var cur_y1 = try y1.copy(self.ctrl.allocator);
        defer cur_y1.deinit();
        var cur_y2 = try y2.copy(self.ctrl.allocator);
        defer cur_y2.deinit();
        var cur_dy1 = try dy1.copy(self.ctrl.allocator);
        defer cur_dy1.deinit();
        var cur_dy2 = try dy2.copy(self.ctrl.allocator);
        defer cur_dy2.deinit();

        var x1_prev = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer x1_prev.deinit();
        var x2_prev = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer x2_prev.deinit();
        var dx1_prev = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer dx1_prev.deinit();
        var dx2_prev = try Tensor.zeros(self.ctrl.allocator, &shape_half);
        defer dx2_prev.deinit();

        var idx = self.ctrl.num_layers;
        while (idx > 0) : (idx -= 1) {
            try self.ctrl.layers[idx - 1].backward(&cur_y1, &cur_y2, &cur_dy1, &cur_dy2, &x1_prev, &x2_prev, &dx1_prev, &dx2_prev);

            var tmp_y1 = cur_y1;
            cur_y1 = x1_prev;
            x1_prev = tmp_y1;

            var tmp_y2 = cur_y2;
            cur_y2 = x2_prev;
            x2_prev = tmp_y2;

            var tmp_dy1 = cur_dy1;
            cur_dy1 = dx1_prev;
            dx1_prev = tmp_dy1;

            var tmp_dy2 = cur_dy2;
            cur_dy2 = dx2_prev;
            dx2_prev = tmp_dy2;
        }

        try self.merge(&cur_dy1, &cur_dy2, grad_input_out);
    }

    pub fn save(self: *const RSF, path: []const u8) !void {
        if (self.ctrl.freed.load(.SeqCst) != 0) return error.NotInitialized;

        var tmp_name_buf: [512]u8 = undefined;
        const tmp_name = try std.fmt.bufPrint(&tmp_name_buf, "{s}.tmp", .{path});

        var file = try std.fs.cwd().createFile(tmp_name, .{ .mode = 0o600 });
        defer file.close();

        var buffered = std.io.bufferedWriter(file.writer());
        const w = buffered.writer();

        try w.writeAll("RSF0");
        try w.writeInt(u32, 1, .little);
        try w.writeInt(u64, @intCast(self.ctrl.num_layers), .little);
        try w.writeInt(u64, @intCast(self.ctrl.dim), .little);
        try w.writeInt(u32, @as(u32, @bitCast(self.ctrl.cfg.clip_min)), .little);
        try w.writeInt(u32, @as(u32, @bitCast(self.ctrl.cfg.clip_max)), .little);
        try w.writeByte(if (self.ctrl.cfg.grad_mean) 1 else 0);

        var i: usize = 0;
        while (i < self.ctrl.num_layers) : (i += 1) {
            const layer = &self.ctrl.layers[i];
            try w.writeInt(u32, @as(u32, @bitCast(layer.clip_min)), .little);
            try w.writeInt(u32, @as(u32, @bitCast(layer.clip_max)), .little);
            try w.writeByte(if (layer.grad_mean) 1 else 0);
            try layer.s_weight.save(w);
            try layer.t_weight.save(w);
            try layer.s_bias.save(w);
            try layer.t_bias.save(w);
        }

        try buffered.flush();
        try file.sync();

        try std.fs.cwd().rename(tmp_name, path);
    }

    pub fn load(allocator: Allocator, path: []const u8) !RSF {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        var r = file.reader();

        var magic: [4]u8 = undefined;
        try r.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "RSF0")) return error.BadFileFormat;

        const version = try r.readInt(u32, .little);
        if (version != 1) return error.UnsupportedVersion;

        const num_layers_u64 = try r.readInt(u64, .little);
        const dim_u64 = try r.readInt(u64, .little);
        if (num_layers_u64 == 0 or dim_u64 == 0) return error.InvalidDimension;
        if (num_layers_u64 > @as(u64, 1 << 20) or dim_u64 > @as(u64, 1 << 20)) return error.TooLarge;

        const num_layers: usize = @intCast(num_layers_u64);
        const dim: usize = @intCast(dim_u64);

        const clip_min = @as(f32, @bitCast(try r.readInt(u32, .little)));
        const clip_max = @as(f32, @bitCast(try r.readInt(u32, .little)));
        const grad_mean = (try r.readByte()) != 0;

        if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max) or !(clip_min < clip_max)) return error.InvalidConfig;

        var rsf = try RSF.initWithConfig(allocator, dim, num_layers, .{
            .clip_min = clip_min,
            .clip_max = clip_max,
            .grad_mean = grad_mean,
            .max_dim = 1 << 20,
            .max_layers = 1 << 20,
        });
        errdefer rsf.deinit();

        var i: usize = 0;
        while (i < rsf.ctrl.num_layers) : (i += 1) {
            const layer_clip_min = @as(f32, @bitCast(try r.readInt(u32, .little)));
            const layer_clip_max = @as(f32, @bitCast(try r.readInt(u32, .little)));
            const layer_grad_mean = (try r.readByte()) != 0;
            if (!std.math.isFinite(layer_clip_min) or !std.math.isFinite(layer_clip_max) or !(layer_clip_min < layer_clip_max)) return error.InvalidConfig;

            var s_w_new = try Tensor.load(allocator, r);
            errdefer s_w_new.deinit();
            var t_w_new = try Tensor.load(allocator, r);
            errdefer t_w_new.deinit();
            var s_b_new = try Tensor.load(allocator, r);
            errdefer s_b_new.deinit();
            var t_b_new = try Tensor.load(allocator, r);
            errdefer t_b_new.deinit();

            if (s_w_new.shape.dims.len != 2 or s_w_new.shape.dims[0] != dim or s_w_new.shape.dims[1] != dim) return error.ShapeMismatch;
            if (t_w_new.shape.dims.len != 2 or t_w_new.shape.dims[0] != dim or t_w_new.shape.dims[1] != dim) return error.ShapeMismatch;
            if (s_b_new.shape.dims.len != 2 or s_b_new.shape.dims[0] != 1 or s_b_new.shape.dims[1] != dim) return error.ShapeMismatch;
            if (t_b_new.shape.dims.len != 2 or t_b_new.shape.dims[0] != 1 or t_b_new.shape.dims[1] != dim) return error.ShapeMismatch;

            const layer = &rsf.ctrl.layers[i];

            layer.s_weight.deinit();
            layer.t_weight.deinit();
            layer.s_bias.deinit();
            layer.t_bias.deinit();

            layer.s_weight = s_w_new;
            layer.t_weight = t_w_new;
            layer.s_bias = s_b_new;
            layer.t_bias = t_b_new;

            layer.clip_min = layer_clip_min;
            layer.clip_max = layer_clip_max;
            layer.grad_mean = layer_grad_mean;
        }

        return rsf;
    }
};