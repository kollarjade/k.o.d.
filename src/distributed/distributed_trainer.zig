const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;

const types = struct {
    pub const Error = error{
        Overflow,
        InvalidShape,
        InvalidAxis,
        OutOfBounds,
        ShapeMismatch,
        DivideByZero,
        InvalidConv2D,
        InvalidPads,
        InvalidReps,
        EmptyInput,
        InvalidForOneHot,
        MustBeSquare,
        SingularMatrix,
    };

    pub const Fixed32_32 = struct {
        value: i64,

        pub fn init(val: f32) types.Fixed32_32 {
            return .{ .value = @intFromFloat(@round(val * 65536.0 * 65536.0)) };
        }
        pub fn toFloat(self: types.Fixed32_32) f32 {
            return @as(f32, @floatFromInt(self.value)) / (65536.0 * 65536.0);
        }
    };

    pub const PRNG = struct {
        state: u64,

        pub fn init(seed: u64) PRNG {
            var p = PRNG{ .state = seed };
            if (p.state == 0) p.state = 1;
            return p;
        }
        pub fn random(self: *PRNG) u64 {
            self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
            return self.state;
        }
        pub fn float(self: *PRNG) f32 {
            return @as(f32, @floatFromInt(self.random() >> 11)) / @as(f32, @floatFromInt(1 << 53));
        }
    };
};

const Error = types.Error;
const TrainerFixed32_32 = types.Fixed32_32;

const Shape = struct {
    dims: []usize,
    strides: []usize,

    pub fn init(allocator: Allocator, shape: []const usize) !Shape {
        const n = shape.len;
        const dims = try allocator.alloc(usize, n);
        errdefer allocator.free(dims);
        const strides = try allocator.alloc(usize, n);
        errdefer allocator.free(strides);
        @memcpy(dims, shape);
        if (n > 0) {
            strides[n - 1] = 1;
            var i: usize = n - 1;
            while (i > 0) : (i -= 1) {
                const r = @mulWithOverflow(strides[i], dims[i]);
                if (r[1] != 0) return Error.Overflow;
                strides[i - 1] = r[0];
            }
        }
        return .{ .dims = dims, .strides = strides };
    }

    pub fn deinit(self: *Shape, allocator: Allocator) void {
        allocator.free(self.dims);
        allocator.free(self.strides);
    }

    pub fn copy(self: *const Shape, allocator: Allocator) !Shape {
        return .{ .dims = try allocator.dupe(usize, self.dims), .strides = try allocator.dupe(usize, self.strides) };
    }

    pub fn totalSize(self: *const Shape) Error!usize {
        var total: usize = 1;
        for (self.dims) |d| {
            const r = @mulWithOverflow(total, d);
            if (r[1] != 0) return Error.Overflow;
            total = r[0];
        }
        return total;
    }

    pub fn equals(self: *const Shape, other: *const Shape) bool {
        return mem.eql(usize, self.dims, other.dims) and mem.eql(usize, self.strides, other.strides);
    }

    pub fn broadcastCompatible(self: *const Shape, target: *const Shape) bool {
        if (target.dims.len < self.dims.len) return false;
        const offset = target.dims.len - self.dims.len;
        var i: usize = 0;
        while (i < self.dims.len) : (i += 1) {
            const self_dim = self.dims[i];
            const target_dim = target.dims[offset + i];
            if (self_dim != target_dim and self_dim != 1) {
                return false;
            }
        }
        return true;
    }

    pub fn isContiguous(self: *const Shape) bool {
        if (self.dims.len == 0) return true;
        var expected: usize = 1;
        var i: usize = self.dims.len;
        while (i > 0) : (i -= 1) {
            const idx = i - 1;
            if (self.strides[idx] != expected) return false;
            const r = @mulWithOverflow(expected, self.dims[idx]);
            if (r[1] != 0) return false;
            expected = r[0];
        }
        return true;
    }
};

const TensorData = struct {
    ptr: [*]f32,
    len: usize,
    refcount: usize,
    allocator: Allocator,
};

pub const Tensor = struct {
    data: *TensorData,
    offset: usize,
    shape: Shape,
    cow: bool,

    pub fn init(allocator: Allocator, shape: []const usize) !Tensor {
        var total_size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const r = @mulWithOverflow(total_size, dim);
            if (r[1] != 0) return Error.Overflow;
            total_size = r[0];
        }
        const ptr = try allocator.alignedAlloc(f32, 16, total_size);
        errdefer allocator.free(ptr);
        @memset(ptr, 0);
        const data_block = try allocator.create(TensorData);
        data_block.ptr = ptr.ptr;
        data_block.len = ptr.len;
        data_block.refcount = 1;
        data_block.allocator = allocator;
        const sh = try Shape.init(allocator, shape);
        return .{ .data = data_block, .offset = 0, .shape = sh, .cow = false };
    }

    pub fn retain(self: *Tensor) void {
        _ = @atomicRmw(usize, &self.data.refcount, .Add, 1, .monotonic);
    }

    pub fn release(self: *Tensor) void {
        if (@atomicRmw(usize, &self.data.refcount, .Sub, 1, .release) == 1) {
            @fence(.acquire);
            const data_slice = self.data.ptr[0..self.data.len];
            self.data.allocator.free(data_slice);
            self.data.allocator.destroy(self.data);
        }
    }

    pub fn deinit(self: *Tensor) void {
        self.release();
        self.shape.deinit(self.data.allocator);
    }

    fn ensureWritable(self: *Tensor) !void {
        if (self.data.refcount == 1 and !self.cow and self.offset == 0 and self.shape.isContiguous()) {
            return;
        }
        const total_size = try self.shape.totalSize();
        const new_ptr = try self.data.allocator.alignedAlloc(f32, 16, total_size);
        errdefer self.data.allocator.free(new_ptr);

        var indices = try self.data.allocator.alloc(usize, self.shape.dims.len);
        defer self.data.allocator.free(indices);
        @memset(indices, 0);

        var flat_idx: usize = 0;
        while (flat_idx < total_size) : (flat_idx += 1) {
            var src_idx: usize = 0;
            for (indices, 0..) |idx, i| {
                src_idx += idx * self.shape.strides[i];
            }
            new_ptr[flat_idx] = self.data.ptr[self.offset + src_idx];

            var carry = true;
            var dim_idx: usize = self.shape.dims.len;
            while (carry and dim_idx > 0) : (dim_idx -= 1) {
                const d = dim_idx - 1;
                indices[d] += 1;
                if (indices[d] < self.shape.dims[d]) {
                    carry = false;
                } else {
                    indices[d] = 0;
                }
            }
        }

        const new_data = try self.data.allocator.create(TensorData);
        new_data.ptr = new_ptr.ptr;
        new_data.len = new_ptr.len;
        new_data.refcount = 1;
        new_data.allocator = self.data.allocator;

        self.release();
        self.data = new_data;
        self.offset = 0;
        self.cow = false;

        var new_sh = try Shape.init(self.data.allocator, self.shape.dims);
        self.shape.deinit(self.data.allocator);
        self.shape = new_sh;
    }

    pub fn newView(self: *Tensor, shape: Shape) !Tensor {
        const shape_size = try shape.totalSize();
        const self_size = try self.shape.totalSize();
        if (shape_size != self_size) return Error.InvalidShape;
        self.retain();
        return .{ .data = self.data, .offset = self.offset, .shape = shape, .cow = true };
    }

    pub fn reshape(self: *Tensor, new_shape: []const usize) !void {
        if (new_shape.len == 0) return Error.InvalidShape;
        var total: usize = 1;
        for (new_shape) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const r = @mulWithOverflow(total, dim);
            if (r[1] != 0) return Error.Overflow;
            total = r[0];
        }
        const self_size = try self.shape.totalSize();
        if (total != self_size) return Error.InvalidShape;

        if (!self.shape.isContiguous()) {
            try self.ensureWritable();
        }

        const new_sh = try Shape.init(self.data.allocator, new_shape);
        self.shape.deinit(self.data.allocator);
        self.shape = new_sh;
    }

    pub fn view(self: *Tensor, new_shape: []const usize) !Tensor {
        if (new_shape.len == 0) return Error.InvalidShape;
        var total: usize = 1;
        for (new_shape) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const r = @mulWithOverflow(total, dim);
            if (r[1] != 0) return Error.Overflow;
            total = r[0];
        }
        const self_size = try self.shape.totalSize();
        if (total != self_size) return Error.InvalidShape;

        var new_strides = try self.data.allocator.alloc(usize, new_shape.len);
        errdefer self.data.allocator.free(new_strides);

        if (new_shape.len > 0) {
            new_strides[new_shape.len - 1] = 1;
            var i: usize = new_shape.len - 1;
            while (i > 0) : (i -= 1) {
                const r = @mulWithOverflow(new_strides[i], new_shape[i]);
                if (r[1] != 0) return Error.Overflow;
                new_strides[i - 1] = r[0];
            }
        }

        const new_sh = .{ .dims = try self.data.allocator.dupe(usize, new_shape), .strides = new_strides };
        self.retain();
        return .{ .data = self.data, .offset = self.offset, .shape = new_sh, .cow = true };
    }

    pub fn slice(self: *Tensor, starts: []const usize, ends: []const usize) !Tensor {
        if (starts.len != self.shape.dims.len or ends.len != self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try self.data.allocator.alloc(usize, self.shape.dims.len);
        errdefer self.data.allocator.free(new_dims);
        var new_offset: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (starts[i] >= ends[i] or ends[i] > self.shape.dims[i]) return Error.OutOfBounds;
            new_dims[i] = ends[i] - starts[i];
            new_offset += starts[i] * self.shape.strides[i];
        }
        const new_sh = try Shape.init(self.data.allocator, new_dims);
        self.data.allocator.free(new_dims);
        self.retain();
        return .{ .data = self.data, .offset = self.offset + new_offset, .shape = new_sh, .cow = true };
    }

    pub fn transpose(self: *Tensor, axes: []const usize) !Tensor {
        if (axes.len != self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try self.data.allocator.alloc(usize, self.shape.dims.len);
        errdefer self.data.allocator.free(new_dims);
        var new_strides = try self.data.allocator.alloc(usize, self.shape.dims.len);
        errdefer self.data.allocator.free(new_strides);

        var seen_axes = try self.data.allocator.alloc(bool, self.shape.dims.len);
        defer self.data.allocator.free(seen_axes);
        @memset(seen_axes, false);

        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (axes[i] >= self.shape.dims.len or seen_axes[axes[i]]) return Error.InvalidAxis;
            seen_axes[axes[i]] = true;
            new_dims[i] = self.shape.dims[axes[i]];
            new_strides[i] = self.shape.strides[axes[i]];
        }

        const new_sh = .{ .dims = new_dims, .strides = new_strides };
        self.retain();
        return .{ .data = self.data, .offset = self.offset, .shape = new_sh, .cow = true };
    }

    fn computeIndex(self: *const Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.dims.len) return Error.InvalidAxis;
        var idx: usize = 0;
        for (indices, 0..) |val, i| {
            if (val >= self.shape.dims[i]) return Error.OutOfBounds;
            idx += val * self.shape.strides[i];
        }
        return idx;
    }

    pub fn get(self: *const Tensor, indices: []const usize) !f32 {
        const idx = try computeIndex(self, indices);
        return self.data.ptr[self.offset + idx];
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) !void {
        try ensureWritable(self);
        const idx = try computeIndex(self, indices);
        self.data.ptr[self.offset + idx] = value;
    }

    pub fn fill(self: *Tensor, value: f32) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[i] = value;
        }
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] += other.data.ptr[other.offset + i];
        }
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] -= other.data.ptr[other.offset + i];
        }
    }

    pub fn mul(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] *= other.data.ptr[other.offset + i];
        }
    }

    pub fn div(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            if (other.data.ptr[other.offset + i] == 0.0) return Error.DivideByZero;
            self.data.ptr[self.offset + i] /= other.data.ptr[other.offset + i];
        }
    }

    pub fn addScalar(self: *Tensor, scalar: f32) !void {
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] += scalar;
        }
    }

    pub fn subScalar(self: *Tensor, scalar: f32) !void {
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] -= scalar;
        }
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) !void {
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] *= scalar;
        }
    }

    pub fn divScalar(self: *Tensor, scalar: f32) !void {
        if (scalar == 0.0) return Error.DivideByZero;
        try ensureWritable(self);
        const N = self.data.len;
        var i: usize = 0;
        while (i < N) : (i += 1) {
            self.data.ptr[self.offset + i] /= scalar;
        }
    }

    pub fn exp(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @exp(self.data.ptr[self.offset + i]);
        }
    }

    pub fn log(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            const val = self.data.ptr[self.offset + i];
            self.data.ptr[self.offset + i] = if (val > 0.0) @log(val) else -math.inf(f32);
        }
    }

    pub fn sin(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @sin(self.data.ptr[self.offset + i]);
        }
    }

    pub fn cos(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @cos(self.data.ptr[self.offset + i]);
        }
    }

    pub fn tan(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @tan(self.data.ptr[self.offset + i]);
        }
    }

    pub fn sqrt(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @sqrt(@max(0.0, self.data.ptr[self.offset + i]));
        }
    }

    pub fn pow(self: *Tensor, exponent: f32) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = math.pow(f32, self.data.ptr[self.offset + i], exponent);
        }
    }

    pub fn abs(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @fabs(self.data.ptr[self.offset + i]);
        }
    }

    pub fn max(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(new_dims);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (i != axis) {
                new_dims[j] = self.shape.dims[i];
                j += 1;
            }
        }
        const result = try init(allocator, new_dims);
        const total_elements = try result.shape.totalSize();
        const max_axis_len = self.shape.dims[axis];

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);

        var idx: usize = 0;
        while (idx < total_elements) : (idx += 1) {
            var remaining = idx;
            var k: usize = 0;
            while (k < self.shape.dims.len) : (k += 1) {
                if (k == axis) continue;
                indices[k] = remaining % new_dims[k];
                remaining /= new_dims[k];
            }

            var max_val: f32 = -math.inf(f32);
            var l: usize = 0;
            while (l < max_axis_len) : (l += 1) {
                indices[axis] = l;
                const val = try self.get(indices);
                if (val > max_val) max_val = val;
            }
            result.data.ptr[idx] = max_val;
        }
        return result;
    }

    pub fn min(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(new_dims);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (i != axis) {
                new_dims[j] = self.shape.dims[i];
                j += 1;
            }
        }
        const result = try init(allocator, new_dims);
        const total_elements = try result.shape.totalSize();
        const max_axis_len = self.shape.dims[axis];

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);

        var idx: usize = 0;
        while (idx < total_elements) : (idx += 1) {
            var remaining = idx;
            var k: usize = 0;
            while (k < self.shape.dims.len) : (k += 1) {
                if (k == axis) continue;
                indices[k] = remaining % new_dims[k];
                remaining /= new_dims[k];
            }

            var min_val: f32 = math.inf(f32);
            var l: usize = 0;
            while (l < max_axis_len) : (l += 1) {
                indices[axis] = l;
                const val = try self.get(indices);
                if (val < min_val) min_val = val;
            }
            result.data.ptr[idx] = min_val;
        }
        return result;
    }

    pub fn sum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(new_dims);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (i != axis) {
                new_dims[j] = self.shape.dims[i];
                j += 1;
            }
        }
        const result = try init(allocator, new_dims);
        const total_elements = try result.shape.totalSize();
        const max_axis_len = self.shape.dims[axis];

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);

        var idx: usize = 0;
        while (idx < total_elements) : (idx += 1) {
            var remaining = idx;
            var k: usize = 0;
            while (k < self.shape.dims.len) : (k += 1) {
                if (k == axis) continue;
                indices[k] = remaining % new_dims[k];
                remaining /= new_dims[k];
            }

            var total: f32 = 0.0;
            var l: usize = 0;
            while (l < max_axis_len) : (l += 1) {
                indices[axis] = l;
                total += try self.get(indices);
            }
            result.data.ptr[idx] = total;
        }
        return result;
    }

    pub fn mean(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var summed = try self.sum(allocator, axis);
        try summed.divScalar(@as(f32, @floatFromInt(self.shape.dims[axis])));
        return summed;
    }

    pub fn matmul(a: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        if (a.shape.dims.len != 2 or b.shape.dims.len != 2 or a.shape.dims[1] != b.shape.dims[0]) return Error.ShapeMismatch;
        const m = a.shape.dims[0];
        const k = a.shape.dims[1];
        const n = b.shape.dims[1];
        const c = try init(allocator, &.{ m, n });

        const TILE = 32;
        var i: usize = 0;
        while (i < m) : (i += TILE) {
            var j: usize = 0;
            while (j < n) : (j += TILE) {
                var l: usize = 0;
                while (l < k) : (l += TILE) {
                    var ii = i;
                    while (ii < @min(i + TILE, m)) : (ii += 1) {
                        var jj = j;
                        while (jj < @min(j + TILE, n)) : (jj += 1) {
                            var sum_val: f32 = 0.0;
                            var ll = l;
                            while (ll < @min(l + TILE, k)) : (ll += 1) {
                                const val_a = a.data.ptr[a.offset + ii * a.shape.strides[0] + ll * a.shape.strides[1]];
                                const val_b = b.data.ptr[b.offset + ll * b.shape.strides[0] + jj * b.shape.strides[1]];
                                sum_val += val_a * val_b;
                            }
                            c.data.ptr[c.offset + ii * c.shape.strides[0] + jj * c.shape.strides[1]] += sum_val;
                        }
                    }
                }
            }
        }
        return c;
    }

    pub fn broadcast(self: *const Tensor, target_shape: []const usize) !Tensor {
        const new_sh = try Shape.init(self.data.allocator, target_shape);
        if (!self.shape.broadcastCompatible(&new_sh)) return Error.ShapeMismatch;
        const result = try init(self.data.allocator, target_shape);

        var indices = try self.data.allocator.alloc(usize, target_shape.len);
        defer self.data.allocator.free(indices);
        @memset(indices, 0);

        const offset = target_shape.len - self.shape.dims.len;
        const total = try result.shape.totalSize();

        var flat_idx: usize = 0;
        while (flat_idx < total) : (flat_idx += 1) {
            var src_idx: usize = 0;
            var i: usize = 0;
            while (i < self.shape.dims.len) : (i += 1) {
                const target_i = indices[offset + i];
                const src_i = if (self.shape.dims[i] == 1) 0 else target_i;
                src_idx += src_i * self.shape.strides[i];
            }
            result.data.ptr[result.offset + flat_idx] = self.data.ptr[self.offset + src_idx];

            var carry = true;
            var dim = target_shape.len;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < target_shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return result;
    }

    pub fn unsqueeze(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis > self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try allocator.alloc(usize, self.shape.dims.len + 1);
        defer allocator.free(new_dims);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len + 1) : (i += 1) {
            if (i == axis) {
                new_dims[i] = 1;
            } else {
                new_dims[i] = self.shape.dims[j];
                j += 1;
            }
        }
        return self.broadcast(new_dims);
    }

    pub fn relu(self: *Tensor) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = @max(0.0, self.data.ptr[self.offset + i]);
        }
    }

    pub fn softmax(self: *Tensor, axis: usize) !void {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        try self.ensureWritable();
        if (self.shape.dims.len == 1) {
            var max_val: f32 = -math.inf(f32);
            var i: usize = 0;
            while (i < self.data.len) : (i += 1) {
                const val = self.data.ptr[self.offset + i];
                if (val > max_val) max_val = val;
            }
            var sum_val: f32 = 0.0;
            i = 0;
            while (i < self.data.len) : (i += 1) {
                self.data.ptr[self.offset + i] = @exp(self.data.ptr[self.offset + i] - max_val);
                sum_val += self.data.ptr[self.offset + i];
            }
            const epsilon: f32 = 1e-10;
            if (sum_val < epsilon) sum_val = epsilon;
            i = 0;
            while (i < self.data.len) : (i += 1) {
                self.data.ptr[self.offset + i] /= sum_val;
            }
            return;
        }
        var max_t = try self.max(self.data.allocator, axis);
        defer max_t.deinit();
        var b_shape = try self.data.allocator.alloc(usize, self.shape.dims.len);
        defer self.data.allocator.free(b_shape);
        @memcpy(b_shape, self.shape.dims);
        b_shape[axis] = 1;
        var b_max = try max_t.broadcast(b_shape);
        defer b_max.deinit();
        try self.sub(&b_max);
        try self.exp();
        var sum_t = try self.sum(self.data.allocator, axis);
        defer sum_t.deinit();
        var b_sum = try sum_t.broadcast(b_shape);
        defer b_sum.deinit();
        try self.div(&b_sum);
    }

    pub fn zeros(allocator: Allocator, shape: []const usize) !Tensor {
        return init(allocator, shape);
    }

    pub fn ones(allocator: Allocator, shape: []const usize) !Tensor {
        var t = try init(allocator, shape);
        try t.fill(1.0);
        return t;
    }

    pub fn full(allocator: Allocator, shape: []const usize, value: f32) !Tensor {
        var t = try init(allocator, shape);
        try t.fill(value);
        return t;
    }

    pub fn randomUniform(allocator: Allocator, shape: []const usize, min_val: f32, max_val: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        const t = try init(allocator, shape);
        var i: usize = 0;
        while (i < t.data.len) : (i += 1) {
            t.data.ptr[i] = prng.float() * (max_val - min_val) + min_val;
        }
        return t;
    }

    pub fn randomNormal(allocator: Allocator, shape: []const usize, mean_val: f32, stddev_val: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        const t = try init(allocator, shape);
        var i: usize = 0;
        while (i < t.data.len) : (i += 2) {
            if (i + 1 >= t.data.len) {
                t.data.ptr[i] = mean_val; 
                break;
            }
            const val_u1 = prng.float();
            const val_u2 = prng.float();
            const radius = @sqrt(-2.0 * @log(@max(1e-10, val_u1)));
            const theta = 2.0 * math.pi * val_u2;
            t.data.ptr[i] = mean_val + stddev_val * radius * @cos(theta);
            t.data.ptr[i + 1] = mean_val + stddev_val * radius * @sin(theta);
        }
        return t;
    }

    pub fn identity(allocator: Allocator, n: usize) !Tensor {
        const t = try init(allocator, &.{ n, n });
        var i: usize = 0;
        while (i < n) : (i += 1) {
            t.data.ptr[i * n + i] = 1.0;
        }
        return t;
    }

    pub fn conv2d(self: *const Tensor, kernel: *const Tensor, allocator: Allocator, stride: [2]usize, padding: [2]usize) !Tensor {
        if (self.shape.dims.len != 4 or kernel.shape.dims.len != 4 or self.shape.dims[3] != kernel.shape.dims[2]) return Error.InvalidConv2D;
        const batch = self.shape.dims[0];
        const in_h = self.shape.dims[1];
        const in_w = self.shape.dims[2];
        const in_c = self.shape.dims[3];
        const k_h = kernel.shape.dims[0];
        const k_w = kernel.shape.dims[1];
        const out_c = kernel.shape.dims[3];
        const out_h = ((in_h + 2 * padding[0] - k_h) / stride[0]) + 1;
        const out_w = ((in_w + 2 * padding[1] - k_w) / stride[1]) + 1;
        const output = try init(allocator, &.{ batch, out_h, out_w, out_c });

        var padded_input = if (padding[0] > 0 or padding[1] > 0) try self.pad(allocator, &[_][2]usize{ .{ 0, 0 }, .{ padding[0], padding[0] }, .{ padding[1], padding[1] }, .{ 0, 0 } }) else self.*;
        defer if (padding[0] > 0 or padding[1] > 0) padded_input.deinit();

        var b: usize = 0;
        while (b < batch) : (b += 1) {
            var oh: usize = 0;
            while (oh < out_h) : (oh += 1) {
                var ow: usize = 0;
                while (ow < out_w) : (ow += 1) {
                    var oc: usize = 0;
                    while (oc < out_c) : (oc += 1) {
                        var sum_result: f32 = 0.0;
                        var kh: usize = 0;
                        while (kh < k_h) : (kh += 1) {
                            var kw: usize = 0;
                            while (kw < k_w) : (kw += 1) {
                                var ic: usize = 0;
                                while (ic < in_c) : (ic += 1) {
                                    const ih = oh * stride[0] + kh;
                                    const iw = ow * stride[1] + kw;
                                    sum_result += try padded_input.get(&.{ b, ih, iw, ic }) * try kernel.get(&.{ kh, kw, ic, oc });
                                }
                            }
                        }
                        try output.set(&.{ b, oh, ow, oc }, sum_result);
                    }
                }
            }
        }
        return output;
    }

    pub fn pad(self: *const Tensor, allocator: Allocator, pads: []const [2]usize) !Tensor {
        if (pads.len != self.shape.dims.len) return Error.InvalidPads;
        var new_shape = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(new_shape);
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            new_shape[i] = self.shape.dims[i] + pads[i][0] + pads[i][1];
        }
        const new_t = try init(allocator, new_shape);

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);
        @memset(indices, 0);

        var src_indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(src_indices);

        const total_new = try new_t.shape.totalSize();
        var flat_idx: usize = 0;
        while (flat_idx < total_new) : (flat_idx += 1) {
            var is_pad = false;
            for (indices, 0..) |idx, d| {
                if (idx < pads[d][0] or idx >= pads[d][0] + self.shape.dims[d]) {
                    is_pad = true;
                } else {
                    src_indices[d] = idx - pads[d][0];
                }
            }
            if (!is_pad) {
                const val = try self.get(src_indices);
                try new_t.set(indices, val);
            }

            var carry = true;
            var dim = self.shape.dims.len;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn tile(self: *const Tensor, allocator: Allocator, reps: []const usize) !Tensor {
        if (reps.len != self.shape.dims.len) return Error.InvalidReps;
        var new_shape = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(new_shape);
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            new_shape[i] = self.shape.dims[i] * reps[i];
        }
        const new_t = try init(allocator, new_shape);

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);
        @memset(indices, 0);

        var src_indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(src_indices);

        const total_new = try new_t.shape.totalSize();
        var flat_idx: usize = 0;
        while (flat_idx < total_new) : (flat_idx += 1) {
            for (indices, 0..) |idx, d| {
                src_indices[d] = idx % self.shape.dims[d];
            }
            const val = try self.get(src_indices);
            try new_t.set(indices, val);

            var carry = true;
            var dim = self.shape.dims.len;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn concat(allocator: Allocator, tensors: []const Tensor, axis: usize) !Tensor {
        if (tensors.len == 0) return Error.EmptyInput;
        const ndim = tensors[0].shape.dims.len;
        if (axis >= ndim) return Error.InvalidAxis;
        for (tensors) |ten| {
            if (ten.shape.dims.len != ndim) return Error.ShapeMismatch;
            var i: usize = 0;
            while (i < ndim) : (i += 1) {
                if (i != axis and ten.shape.dims[i] != tensors[0].shape.dims[i]) return Error.ShapeMismatch;
            }
        }
        var new_shape = try allocator.alloc(usize, ndim);
        defer allocator.free(new_shape);
        @memcpy(new_shape, tensors[0].shape.dims);
        var total_axis: usize = 0;
        for (tensors) |ten| {
            total_axis += ten.shape.dims[axis];
        }
        new_shape[axis] = total_axis;
        const new_t = try init(allocator, new_shape);

        var write_idx = try allocator.alloc(usize, ndim);
        defer allocator.free(write_idx);
        @memset(write_idx, 0);

        var read_idx = try allocator.alloc(usize, ndim);
        defer allocator.free(read_idx);

        const total_new = try new_t.shape.totalSize();
        var flat: usize = 0;
        while (flat < total_new) : (flat += 1) {
            var src_tensor_idx: usize = 0;
            var offset_in_tensor: usize = 0;

            var acc: usize = 0;
            var t_idx: usize = 0;
            while (t_idx < tensors.len) : (t_idx += 1) {
                if (write_idx[axis] < acc + tensors[t_idx].shape.dims[axis]) {
                    src_tensor_idx = t_idx;
                    offset_in_tensor = write_idx[axis] - acc;
                    break;
                }
                acc += tensors[t_idx].shape.dims[axis];
            }

            @memcpy(read_idx, write_idx);
            read_idx[axis] = offset_in_tensor;

            const val = try tensors[src_tensor_idx].get(read_idx);
            try new_t.set(write_idx, val);

            var carry = true;
            var dim = ndim;
            while (carry and dim > 0) : (dim -= 1) {
                write_idx[dim - 1] += 1;
                if (write_idx[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    write_idx[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn stack(allocator: Allocator, tensors: []const Tensor, axis: usize) !Tensor {
        if (tensors.len == 0) return Error.EmptyInput;
        const ndim = tensors[0].shape.dims.len;
        if (axis > ndim) return Error.InvalidAxis;
        for (tensors) |ten| {
            if (ten.shape.dims.len != ndim or !ten.shape.equals(&tensors[0].shape)) return Error.ShapeMismatch;
        }
        var new_shape = try allocator.alloc(usize, ndim + 1);
        defer allocator.free(new_shape);
        new_shape[axis] = tensors.len;
        var k: usize = 0;
        var i: usize = 0;
        while (i < ndim + 1) : (i += 1) {
            if (i == axis) continue;
            new_shape[i] = tensors[0].shape.dims[k];
            k += 1;
        }
        const new_t = try init(allocator, new_shape);

        var write_idx = try allocator.alloc(usize, ndim + 1);
        defer allocator.free(write_idx);
        @memset(write_idx, 0);

        var read_idx = try allocator.alloc(usize, ndim);
        defer allocator.free(read_idx);

        const total_new = try new_t.shape.totalSize();
        var flat: usize = 0;
        while (flat < total_new) : (flat += 1) {
            const t_idx = write_idx[axis];

            var kk: usize = 0;
            var d: usize = 0;
            while (d < ndim + 1) : (d += 1) {
                if (d == axis) continue;
                read_idx[kk] = write_idx[d];
                kk += 1;
            }

            const val = try tensors[t_idx].get(read_idx);
            try new_t.set(write_idx, val);

            var carry = true;
            var dim = ndim + 1;
            while (carry and dim > 0) : (dim -= 1) {
                write_idx[dim - 1] += 1;
                if (write_idx[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    write_idx[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn argmax(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(new_dims);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (i != axis) {
                new_dims[j] = self.shape.dims[i];
                j += 1;
            }
        }
        const result = try init(allocator, new_dims);
        const total_elements = try result.shape.totalSize();
        const max_axis_len = self.shape.dims[axis];

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);

        var idx: usize = 0;
        while (idx < total_elements) : (idx += 1) {
            var remaining = idx;
            var k: usize = 0;
            while (k < self.shape.dims.len) : (k += 1) {
                if (k == axis) continue;
                indices[k] = remaining % new_dims[k];
                remaining /= new_dims[k];
            }

            var max_idx: usize = 0;
            var max_val: f32 = -math.inf(f32);
            var l: usize = 0;
            while (l < max_axis_len) : (l += 1) {
                indices[axis] = l;
                const val = try self.get(indices);
                if (val > max_val) {
                    max_val = val;
                    max_idx = l;
                }
            }
            result.data.ptr[idx] = @as(f32, @floatFromInt(max_idx));
        }
        return result;
    }

    pub fn cumsum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        const new_t = try self.copy(allocator);
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);
        @memset(indices, 0);

        const total = try new_t.shape.totalSize();
        var flat: usize = 0;
        while (flat < total) : (flat += 1) {
            if (indices[axis] > 0) {
                var prev_indices = try allocator.alloc(usize, self.shape.dims.len);
                defer allocator.free(prev_indices);
                @memcpy(prev_indices, indices);
                prev_indices[axis] -= 1;

                const prev = try new_t.get(prev_indices);
                const curr = try new_t.get(indices);
                try new_t.set(indices, prev + curr);
            }

            var carry = true;
            var dim = self.shape.dims.len;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < self.shape.dims[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn variance(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var mean_t = try self.mean(allocator, axis);
        defer mean_t.deinit();
        var mean_unsqueezed = try mean_t.unsqueeze(allocator, axis);
        defer mean_unsqueezed.deinit();
        var mean_broadcasted = try mean_unsqueezed.broadcast(self.shape.dims);
        defer mean_broadcasted.deinit();
        var diff = try self.copy(allocator);
        defer diff.deinit();
        try diff.sub(&mean_broadcasted);
        var sq = try diff.copy(allocator);
        defer sq.deinit();
        try sq.mul(&diff);
        return try sq.mean(allocator, axis);
    }

    pub fn stddev(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var var_t = try self.variance(allocator, axis);
        try var_t.sqrt();
        return var_t;
    }

    pub fn argmin(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(new_dims);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (i != axis) {
                new_dims[j] = self.shape.dims[i];
                j += 1;
            }
        }
        const result = try init(allocator, new_dims);
        const total_elements = try result.shape.totalSize();
        const max_axis_len = self.shape.dims[axis];

        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);

        var idx: usize = 0;
        while (idx < total_elements) : (idx += 1) {
            var remaining = idx;
            var k: usize = 0;
            while (k < self.shape.dims.len) : (k += 1) {
                if (k == axis) continue;
                indices[k] = remaining % new_dims[k];
                remaining /= new_dims[k];
            }

            var min_idx: usize = 0;
            var min_val: f32 = math.inf(f32);
            var l: usize = 0;
            while (l < max_axis_len) : (l += 1) {
                indices[axis] = l;
                const val = try self.get(indices);
                if (val < min_val) {
                    min_val = val;
                    min_idx = l;
                }
            }
            result.data.ptr[idx] = @as(f32, @floatFromInt(min_idx));
        }
        return result;
    }

    pub fn sort(self: *const Tensor, allocator: Allocator, axis: usize, descending: bool) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        const new_t = try self.copy(allocator);

        var reduced_shape = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape.dims[i];
                j += 1;
            }
        }

        var common_indices = try allocator.alloc(usize, self.shape.dims.len - 1);
        defer allocator.free(common_indices);
        @memset(common_indices, 0);

        var temp = try allocator.alloc(f32, self.shape.dims[axis]);
        defer allocator.free(temp);

        const flat_size = try self.shape.totalSize();
        var flat_idx: usize = 0;

        while (flat_idx < flat_size) : (flat_idx += 1) {
            var base_idx: usize = 0;
            var k: usize = 0;
            i = 0;
            while (i < self.shape.dims.len) : (i += 1) {
                if (i != axis) {
                    base_idx += common_indices[k] * new_t.shape.strides[i];
                    k += 1;
                }
            }

            i = 0;
            while (i < self.shape.dims[axis]) : (i += 1) {
                const idx = base_idx + i * new_t.shape.strides[axis];
                temp[i] = new_t.data.ptr[new_t.offset + idx];
            }

            if (descending) {
                mem.sort(f32, temp, {}, comptime struct { pub fn lessThan(_: void, a: f32, b: f32) bool { return a > b; } }.lessThan);
            } else {
                mem.sort(f32, temp, {}, comptime struct { pub fn lessThan(_: void, a: f32, b: f32) bool { return a < b; } }.lessThan);
            }

            i = 0;
            while (i < self.shape.dims[axis]) : (i += 1) {
                const idx = base_idx + i * new_t.shape.strides[axis];
                new_t.data.ptr[new_t.offset + idx] = temp[i];
            }

            var carry = true;
            var dim: isize = @as(isize, @intCast(self.shape.dims.len - 1)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                const d: usize = @intCast(dim + 1);
                if (d == axis) continue;
                const rd = if (d > axis) d - 1 else d;
                common_indices[rd] += 1;
                if (common_indices[rd] < reduced_shape[rd]) {
                    carry = false;
                } else {
                    common_indices[rd] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn unique(self: *const Tensor, allocator: Allocator) !Tensor {
        var unique_set = std.AutoHashMap(f32, void).init(allocator);
        defer unique_set.deinit();
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            try unique_set.put(self.data.ptr[self.offset + i], {});
        }
        const unique_len = unique_set.count();
        const unique_t = try init(allocator, &.{unique_len});
        i = 0;
        var iter = unique_set.iterator();
        while (iter.next()) |entry| {
            unique_t.data.ptr[i] = entry.key_ptr.*;
            i += 1;
        }
        return unique_t;
    }

    pub fn oneHot(self: *const Tensor, allocator: Allocator, num_classes: usize) !Tensor {
        if (self.shape.dims.len != 1) return Error.InvalidForOneHot;
        const new_shape = &.{ self.shape.dims[0], num_classes };
        const new_t = try init(allocator, new_shape);
        try new_t.fill(0.0);
        var i: usize = 0;
        while (i < self.shape.dims[0]) : (i += 1) {
            const val = try self.get(&.{i});
            const idx_float = @trunc(val);
            if (idx_float >= 0 and idx_float < @as(f32, @floatFromInt(num_classes))) {
                const idx = @as(usize, @intFromFloat(idx_float));
                try new_t.set(&.{ i, idx }, 1.0);
            }
        }
        return new_t;
    }

    pub fn isClose(self: *const Tensor, other: *const Tensor, rtol: f32, atol: f32) !bool {
        if (!self.shape.equals(&other.shape)) return false;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            const diff = @fabs(self.data.ptr[self.offset + i] - other.data.ptr[other.offset + i]);
            if (diff > atol + rtol * @fabs(other.data.ptr[other.offset + i])) return false;
        }
        return true;
    }

    pub fn toInt(self: *const Tensor, allocator: Allocator) !Tensor {
        const new_t = try init(allocator, self.shape.dims);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            new_t.data.ptr[i] = @trunc(self.data.ptr[self.offset + i]);
        }
        return new_t;
    }

    pub fn spectralNorm(self: *const Tensor, allocator: Allocator, max_iter: u32, tol: f32) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var v = try randomUniform(allocator, &.{n}, -1.0, 1.0, 42);
        defer v.deinit();
        const norm_v = v.normL2();
        try v.divScalar(norm_v);
        var last_radius: f32 = 0.0;
        var iter: usize = 0;
        while (iter < max_iter) : (iter += 1) {
            var av = try matmul(self, &v, allocator);
            defer av.deinit();
            const norm_av = av.normL2();
            if (norm_av == 0.0) return 0.0;
            try av.divScalar(norm_av);
            v.deinit();
            v = av;
            var radius: f32 = 0.0;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                radius += v.data.ptr[i] * self.data.ptr[self.offset + i * n + i];
            }
            if (@fabs(radius - last_radius) < tol) return @fabs(radius);
            last_radius = radius;
        }
        return @fabs(last_radius);
    }

    pub fn normL2(self: *const Tensor) f32 {
        var sum_sq: f32 = 0.0;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            sum_sq += self.data.ptr[self.offset + i] * self.data.ptr[self.offset + i];
        }
        return @sqrt(sum_sq);
    }

    pub fn dot(self: *const Tensor, other: *const Tensor) !f32 {
        if (self.data.len != other.data.len) return Error.ShapeMismatch;
        var sum_result: f32 = 0.0;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            sum_result += self.data.ptr[self.offset + i] * other.data.ptr[other.offset + i];
        }
        return sum_result;
    }

    pub fn outer(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        if (a.shape.dims.len == 1 and b.shape.dims.len == 1) {
            const m = a.shape.dims[0];
            const n = b.shape.dims[0];
            const result = try init(allocator, &.{ m, n });
            var i: usize = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    result.data.ptr[i * n + j] = a.data.ptr[a.offset + i] * b.data.ptr[b.offset + j];
                }
            }
            return result;
        }
        return Error.ShapeMismatch;
    }

    pub fn inverse(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var mat = try self.copy(allocator);
        defer mat.deinit();
        var inv = try identity(allocator, n);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var j: usize = i + 1;
            while (j < n) : (j += 1) {
                if (@fabs(mat.data.ptr[j * n + i]) > @fabs(mat.data.ptr[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (@fabs(mat.data.ptr[pivot * n + i]) < 1e-10) return Error.SingularMatrix;
            if (pivot != i) {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const temp_mat = mat.data.ptr[i * n + k];
                    mat.data.ptr[i * n + k] = mat.data.ptr[pivot * n + k];
                    mat.data.ptr[pivot * n + k] = temp_mat;
                    const temp_inv = inv.data.ptr[i * n + k];
                    inv.data.ptr[i * n + k] = inv.data.ptr[pivot * n + k];
                    inv.data.ptr[pivot * n + k] = temp_inv;
                }
            }
            const diag = mat.data.ptr[i * n + i];
            var k: usize = 0;
            while (k < n) : (k += 1) {
                mat.data.ptr[i * n + k] /= diag;
                inv.data.ptr[i * n + k] /= diag;
            }
            j = 0;
            while (j < n) : (j += 1) {
                if (j != i) {
                    const factor = mat.data.ptr[j * n + i];
                    k = 0;
                    while (k < n) : (k += 1) {
                        mat.data.ptr[j * n + k] -= factor * mat.data.ptr[i * n + k];
                        inv.data.ptr[j * n + k] -= factor * inv.data.ptr[i * n + k];
                    }
                }
            }
        }
        return inv;
    }

    pub fn eig(self: *const Tensor, allocator: Allocator) !struct { vals: Tensor, vecs: Tensor } {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var mat = try self.copy(allocator);
        defer mat.deinit();
        var vecs = try identity(allocator, n);
        var iter: usize = 0;
        while (iter < 100) : (iter += 1) {
            const qr_result = try mat.qr(allocator);
            const new_mat = try matmul(&qr_result.r, &qr_result.q, allocator);
            mat.deinit();
            mat = new_mat;
            const new_vecs = try matmul(&vecs, &qr_result.q, allocator);
            vecs.deinit();
            vecs = new_vecs;
            qr_result.q.deinit();
            qr_result.r.deinit();
            var converged = true;
            var ii: usize = 1;
            while (ii < n) : (ii += 1) {
                if (@fabs(mat.data.ptr[ii * n + (ii - 1)]) > 1e-10) {
                    converged = false;
                    break;
                }
            }
            if (converged) break;
        }
        var vals = try init(allocator, &.{n});
        var i: usize = 0;
        while (i < n) : (i += 1) {
            vals.data.ptr[i] = mat.data.ptr[i * n + i];
        }
        return .{ .vals = vals, .vecs = vecs };
    }

    pub fn qr(self: *const Tensor, allocator: Allocator) !struct { q: Tensor, r: Tensor } {
        const m = self.shape.dims[0];
        const n = self.shape.dims[1];
        var q = try identity(allocator, m);
        var r = try self.copy(allocator);
        var j: usize = 0;
        while (j < @min(m, n)) : (j += 1) {
            var x = try allocator.alloc(f32, m - j);
            defer allocator.free(x);
            var i: usize = j;
            while (i < m) : (i += 1) {
                x[i - j] = r.data.ptr[i * n + j];
            }
            var norm_x: f32 = 0.0;
            for (x) |val| norm_x += val * val;
            norm_x = @sqrt(norm_x);
            if (norm_x == 0.0) continue;
            const sign: f32 = if (x[0] >= 0.0) 1.0 else -1.0;
            var u = try allocator.alloc(f32, m - j);
            defer allocator.free(u);
            u[0] = x[0] + sign * norm_x;
            i = 1;
            while (i < m - j) : (i += 1) u[i] = x[i];
            var norm_u: f32 = 0.0;
            for (u) |val| norm_u += val * val;
            norm_u = @sqrt(norm_u);
            if (norm_u > 0.0) {
                for (u) |*val| val.* /= norm_u;
            }
            var k: usize = j;
            while (k < n) : (k += 1) {
                var dot_prod: f32 = 0.0;
                i = j;
                while (i < m) : (i += 1) {
                    dot_prod += r.data.ptr[i * n + k] * u[i - j];
                }
                dot_prod *= 2.0;
                i = j;
                while (i < m) : (i += 1) {
                    r.data.ptr[i * n + k] -= dot_prod * u[i - j];
                }
            }
            k = 0;
            while (k < m) : (k += 1) {
                var dot_prod: f32 = 0.0;
                i = j;
                while (i < m) : (i += 1) {
                    dot_prod += q.data.ptr[i * m + k] * u[i - j];
                }
                dot_prod *= 2.0;
                i = j;
                while (i < m) : (i += 1) {
                    q.data.ptr[i * m + k] -= dot_prod * u[i - j];
                }
            }
        }
        return .{ .q = q, .r = r };
    }

    pub fn svd(self: *const Tensor, allocator: Allocator) !struct { u: Tensor, s: Tensor, v: Tensor } {
        const m = self.shape.dims[0];
        const n = self.shape.dims[1];

        var ata = try self.transpose(&.{1, 0});
        defer ata.deinit();
        var ata_self = try matmul(&ata, self, allocator);
        defer ata_self.deinit();
        var eigen = try eig(&ata_self, allocator);

        var s = try init(allocator, &.{eigen.vals.shape.dims[0]});
        var i: usize = 0;
        while (i < s.data.len) : (i += 1) {
            s.data.ptr[i] = @sqrt(@max(0.0, eigen.vals.data.ptr[i]));
        }
        defer eigen.vals.deinit();

        var v = eigen.vecs;
        var u = try matmul(self, &v, allocator);

        i = 0;
        while (i < s.data.len) : (i += 1) {
            if (s.data.ptr[i] > 1e-10) {
                var jj: usize = 0;
                while (jj < m) : (jj += 1) {
                    u.data.ptr[jj * n + i] /= s.data.ptr[i];
                }
            } else {
                var jj: usize = 0;
                while (jj < m) : (jj += 1) {
                    u.data.ptr[jj * n + i] = 0.0;
                }
            }
        }
        return .{ .u = u, .s = s, .v = v };
    }

    pub fn cholesky(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var l = try init(allocator, self.shape.dims);
        try l.fill(0.0);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < i + 1) : (j += 1) {
                var sum_result: f32 = 0.0;
                var k: usize = 0;
                while (k < j) : (k += 1) {
                    sum_result += l.data.ptr[i * n + k] * l.data.ptr[j * n + k];
                }
                if (i == j) {
                    const diag_val = self.data.ptr[self.offset + i * n + j] - sum_result;
                    if (diag_val <= 0.0) return Error.SingularMatrix;
                    l.data.ptr[i * n + j] = @sqrt(diag_val);
                } else {
                    l.data.ptr[i * n + j] = (self.data.ptr[self.offset + i * n + j] - sum_result) / l.data.ptr[j * n + j];
                }
            }
        }
        return l;
    }

    pub fn solve(self: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        const lu_result = try self.lu(allocator);
        defer lu_result.l.deinit();
        defer lu_result.u.deinit();

        const n = self.shape.dims[0];
        const y = try init(allocator, b.shape.dims);

        const cols = if (b.shape.dims.len == 1) 1 else b.shape.dims[1];

        var col: usize = 0;
        while (col < cols) : (col += 1) {
            var i: usize = 0;
            while (i < n) : (i += 1) {
                var sum_l: f32 = 0.0;
                var k: usize = 0;
                while (k < i) : (k += 1) {
                    sum_l += lu_result.l.data.ptr[i * n + k] * y.data.ptr[k * cols + col];
                }
                y.data.ptr[i * cols + col] = (if (b.shape.dims.len == 1) b.data.ptr[b.offset + i] else b.data.ptr[b.offset + i * cols + col]) - sum_l;
            }
        }

        const x = try init(allocator, b.shape.dims);

        col = 0;
        while (col < cols) : (col += 1) {
            var i: isize = @as(isize, @intCast(n)) - 1;
            while (i >= 0) : (i -= 1) {
                var sum_u: f32 = 0.0;
                var k: usize = @intCast(i + 1);
                while (k < n) : (k += 1) {
                    sum_u += lu_result.u.data.ptr[@as(usize, @intCast(i)) * n + k] * x.data.ptr[k * cols + col];
                }
                x.data.ptr[@as(usize, @intCast(i)) * cols + col] = (y.data.ptr[@as(usize, @intCast(i)) * cols + col] - sum_u) / lu_result.u.data.ptr[@as(usize, @intCast(i)) * n + @as(usize, @intCast(i))];
            }
        }

        y.deinit();
        return x;
    }

    pub fn lu(self: *const Tensor, allocator: Allocator) !struct { l: Tensor, u: Tensor } {
        const n = self.shape.dims[0];
        var l = try identity(allocator, n);
        var u = try self.copy(allocator);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = i;
            while (j < n) : (j += 1) {
                var sum_result: f32 = 0.0;
                var k: usize = 0;
                while (k < i) : (k += 1) {
                    sum_result += l.data.ptr[j * n + k] * u.data.ptr[k * n + i];
                }
                u.data.ptr[j * n + i] = self.data.ptr[self.offset + j * n + i] - sum_result;
            }
            j = i + 1;
            while (j < n) : (j += 1) {
                var sum_result2: f32 = 0.0;
                var k: usize = 0;
                while (k < i) : (k += 1) {
                    sum_result2 += l.data.ptr[j * n + k] * u.data.ptr[k * n + i];
                }
                if (u.data.ptr[i * n + i] == 0.0) return Error.SingularMatrix;
                l.data.ptr[j * n + i] = (self.data.ptr[self.offset + j * n + i] - sum_result2) / u.data.ptr[i * n + i];
            }
        }
        return .{ .l = l, .u = u };
    }

    pub fn trace(self: *const Tensor) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        var sum_result: f32 = 0.0;
        const n = self.shape.dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) {
            sum_result += self.data.ptr[self.offset + i * n + i];
        }
        return sum_result;
    }

    pub fn det(self: *const Tensor, allocator: Allocator) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var mat = try self.copy(allocator);
        defer mat.deinit();
        var det_val: f32 = 1.0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var j: usize = i + 1;
            while (j < n) : (j += 1) {
                if (@fabs(mat.data.ptr[j * n + i]) > @fabs(mat.data.ptr[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (@fabs(mat.data.ptr[pivot * n + i]) < 1e-10) return 0.0;
            if (pivot != i) {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const temp = mat.data.ptr[i * n + k];
                    mat.data.ptr[i * n + k] = mat.data.ptr[pivot * n + k];
                    mat.data.ptr[pivot * n + k] = temp;
                }
                det_val = -det_val;
            }
            det_val *= mat.data.ptr[i * n + i];
            j = i + 1;
            while (j < n) : (j += 1) {
                const factor = mat.data.ptr[j * n + i] / mat.data.ptr[i * n + i];
                var k: usize = i;
                while (k < n) : (k += 1) {
                    mat.data.ptr[j * n + k] -= factor * mat.data.ptr[i * n + k];
                }
            }
        }
        return det_val;
    }

    pub fn clip(self: *Tensor, min_val: f32, max_val: f32) !void {
        try ensureWritable(self);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data.ptr[self.offset + i] = math.clamp(self.data.ptr[self.offset + i], min_val, max_val);
        }
    }

    pub fn norm(self: *const Tensor, order: f32) f32 {
        var total: f32 = 0.0;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            total += math.pow(f32, @fabs(self.data.ptr[self.offset + i]), order);
        }
        return math.pow(f32, total, 1.0 / order);
    }

    pub fn toFixed(self: *const Tensor, allocator: Allocator) !Tensor {
        const fixed_t = try init(allocator, self.shape.dims);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            fixed_t.data.ptr[i] = TrainerFixed32_32.init(self.data.ptr[self.offset + i]).toFloat();
        }
        return fixed_t;
    }

    pub fn arange(allocator: Allocator, start: f32, end: f32, step: f32) !Tensor {
        if (step == 0) return Error.InvalidShape;
        const size = @as(usize, @intFromFloat(@ceil((end - start) / step)));
        const t = try init(allocator, &.{size});
        var i: usize = 0;
        while (i < size) : (i += 1) {
            t.data.ptr[i] = start + @as(f32, @floatFromInt(i)) * step;
        }
        return t;
    }

    pub fn linspace(allocator: Allocator, start: f32, end: f32, num: usize) !Tensor {
        const t = try init(allocator, &.{num});
        if (num == 0) return t;
        if (num == 1) {
            t.data.ptr[0] = start;
            return t;
        }
        const step = (end - start) / @as(f32, @floatFromInt(num - 1));
        var val = start;
        var i: usize = 0;
        while (i < num - 1) : (i += 1) {
            t.data.ptr[i] = val;
            val += step;
        }
        t.data.ptr[num - 1] = end;
        return t;
    }

    pub fn toString(self: *const Tensor, allocator: Allocator) ![]u8 {
        var buf = std.ArrayList(u8).init(allocator);
        const writer = buf.writer();
        try writer.print("Tensor(shape=[", .{});
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            const dim = self.shape.dims[i];
            try writer.print("{d}", .{dim});
            if (i < self.shape.dims.len - 1) try writer.print(", ", .{});
        }
        try writer.print("], data=[", .{});
        const print_limit = @min(self.data.len, 10);
        i = 0;
        while (i < print_limit) : (i += 1) {
            const val = self.data.ptr[self.offset + i];
            try writer.print("{d:.4}", .{val});
            if (i < print_limit - 1) try writer.print(", ", .{});
        }
        if (self.data.len > 10) {
            try writer.print(", ...", .{});
        }
        try writer.print("])", .{});
        return buf.toOwnedSlice();
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        try writer.writeInt(u64, @as(u64, self.shape.dims.len), .little);
        for (self.shape.dims) |dim| {
            try writer.writeInt(u64, @as(u64, dim), .little);
        }
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            const val = self.data.ptr[self.offset + i];
            const val_bits: u32 = @bitCast(val);
            try writer.writeInt(u32, val_bits, .little);
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const ndim64 = try reader.readInt(u64, .little);
        if (ndim64 > math.maxInt(usize)) return Error.Overflow;
        const ndim: usize = @intCast(ndim64);
        if (ndim == 0) return Error.EmptyInput;
        if (ndim > 16) return Error.InvalidShape;
        var shape = try allocator.alloc(usize, ndim);
        errdefer allocator.free(shape);
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            const dim64 = try reader.readInt(u64, .little);
            if (dim64 > math.maxInt(usize)) return Error.Overflow;
            shape[i] = @intCast(dim64);
        }
        const tensor = try init(allocator, shape);
        allocator.free(shape);
        i = 0;
        while (i < tensor.data.len) : (i += 1) {
            const val_bits = try reader.readInt(u32, .little);
            tensor.data.ptr[i] = @bitCast(val_bits);
        }
        return tensor;
    }
};

const nccl = @import("nccl_bindings.zig");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;

pub const QuantumTrainingConfig = struct {
    ibm_crn: []const u8,
    ibm_api_key: []const u8,
    num_qubits: usize,
    vqe_layers: usize,
    quantum_shots: usize,
    enable_hybrid: bool,
    enable_verification: bool,
    quantum_learning_rate: f64,
    max_quantum_iterations: usize,
    verification_frequency: usize,
};

pub const HybridStepResult = struct {
    classical_loss: f64,
    quantum_loss: f64,
    combined_loss: f64,
    quantum_contribution: f64,
    gradient_norm: f64,
    verification_passed: bool,
};

pub const QuantumStatistics = struct {
    total_shots: usize,
    successful_verifications: usize,
    z_runtime_memory_used: usize,
    z_runtime_variables: usize,
    ve_total_verifications: usize,
    ve_successful_verifications: usize,
    ve_invariant_count: usize,
    quantum_enabled: bool,
    hybrid_enabled: bool,
    verification_enabled: bool,
};

pub const DistributedTrainer = struct {
    allocator: Allocator,
    coordinator: *GPUCoordinator,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    local_batch_size: usize,
    quantum_config: ?QuantumTrainingConfig,
    weights: ?Tensor,
    epoch: usize,
    quantum_stats: ?QuantumStatistics,

    pub fn init(
        allocator: Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        local_batch_size: usize,
    ) !DistributedTrainer {
        return DistributedTrainer{
            .allocator = allocator,
            .coordinator = coordinator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .vocab_size = vocab_size,
            .local_batch_size = local_batch_size,
            .quantum_config = null,
            .weights = null,
            .epoch = 0,
            .quantum_stats = null,
        };
    }

    pub fn initWithQuantum(
        allocator: Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        local_batch_size: usize,
        quantum_config: QuantumTrainingConfig,
    ) !DistributedTrainer {
        return DistributedTrainer{
            .allocator = allocator,
            .coordinator = coordinator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .vocab_size = vocab_size,
            .local_batch_size = local_batch_size,
            .quantum_config = quantum_config,
            .weights = null,
            .epoch = 0,
            .quantum_stats = QuantumStatistics{
                .total_shots = 0,
                .successful_verifications = 0,
                .z_runtime_memory_used = 0,
                .z_runtime_variables = 0,
                .ve_total_verifications = 0,
                .ve_successful_verifications = 0,
                .ve_invariant_count = 0,
                .quantum_enabled = true,
                .hybrid_enabled = quantum_config.enable_hybrid,
                .verification_enabled = quantum_config.enable_verification,
            },
        };
    }

    pub fn deinit(self: *DistributedTrainer) void {
        if (self.weights) |*w| w.deinit();
    }

    pub fn writeNcclId(_: Allocator, _: *nccl.ncclUniqueId) !void {}

    pub fn readNcclId(_: Allocator, _: *nccl.ncclUniqueId, _: u64) !void {}

    pub fn ensureCheckpointDirExists(_: Allocator) !void {
        std.fs.cwd().makeDir("checkpoints") catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
    }

    pub fn getCheckpointPath(allocator: Allocator, filename: []const u8) ![]u8 {
        return std.fmt.allocPrint(allocator, "checkpoints/{s}", .{filename});
    }

    pub fn loadCheckpoint(self: *DistributedTrainer, _: []const u8) !void {
        _ = self;
        return error.FileNotFound;
    }

    pub fn saveCheckpoint(self: *const DistributedTrainer, _: []const u8) !void {
        _ = self;
    }

    pub fn loadDataset(self: *DistributedTrainer, _: []const u8) ![][]const u8 {
        _ = self;
        return &[_][]const u8{};
    }

    pub fn trainEpoch(self: *DistributedTrainer, _: [][]const u8) !f64 {
        self.epoch += 1;
        return 0.0;
    }

    pub fn trainEpochHybrid(self: *DistributedTrainer, _: [][]const u8) !HybridStepResult {
        self.epoch += 1;
        return HybridStepResult{
            .classical_loss = 0.0,
            .quantum_loss = 0.0,
            .combined_loss = 0.0,
            .quantum_contribution = 0.0,
            .gradient_norm = 0.0,
            .verification_passed = true,
        };
    }

    pub fn getQuantumStatistics(self: *const DistributedTrainer) ?QuantumStatistics {
        return self.quantum_stats;
    }
};