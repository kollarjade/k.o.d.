const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("types.zig");
const Error = types.Error;
const Fixed32_32 = types.Fixed32_32;

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
        return mem.eql(usize, self.dims, other.dims);
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
            expected *= self.dims[idx];
        }
        return true;
    }
};

pub const Tensor = struct {
    data: []f32,
    shape: Shape,
    allocator: Allocator,
    refcount: *usize,
    cow: *bool,
    mutex: *std.Thread.Mutex,

    pub fn init(allocator: Allocator, shape: []const usize) !Tensor {
        var total_size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const r = @mulWithOverflow(total_size, dim);
            if (r[1] != 0) return Error.Overflow;
            total_size = r[0];
        }
        const data = try allocator.alloc(f32, total_size);
        @memset(data, 0);
        const sh = try Shape.init(allocator, shape);
        const refcount = try allocator.create(usize);
        refcount.* = 1;
        const cow = try allocator.create(bool);
        cow.* = false;
        const mutex = try allocator.create(std.Thread.Mutex);
        mutex.* = .{};
        return .{ .data = data, .shape = sh, .allocator = allocator, .refcount = refcount, .cow = cow, .mutex = mutex };
    }

    pub fn retain(self: *Tensor) void {
        _ = @atomicRmw(usize, self.refcount, .Add, 1, .AcqRel);
    }

    pub fn release(self: *Tensor) void {
        const prev = @atomicRmw(usize, self.refcount, .Sub, 1, .AcqRel);
        if (prev == 1) {
            self.deallocate();
        }
    }

    fn deallocate(self: *Tensor) void {
        self.allocator.free(self.data);
        self.shape.deinit(self.allocator);
        self.allocator.destroy(self.refcount);
        self.allocator.destroy(self.cow);
        self.allocator.destroy(self.mutex);
    }

    pub fn deinit(self: *Tensor) void {
        const prev = @atomicRmw(usize, self.refcount, .Sub, 1, .AcqRel);
        if (prev == 1) {
            self.deallocate();
        }
    }

    pub fn copy(self: *const Tensor, allocator: Allocator) !Tensor {
        const new_t = try init(allocator, self.shape.dims);
        @memcpy(new_t.data, self.data);
        return new_t;
    }

    fn ensureWritable(self: *Tensor) !void {
        const original_mutex = self.mutex;
        original_mutex.lock();
        
        if (!self.cow.*) {
            original_mutex.unlock();
            return;
        }
        
        const new_data = self.allocator.alloc(f32, self.data.len) catch |err| {
            original_mutex.unlock();
            return err;
        };
        @memcpy(new_data, self.data);
        
        const new_refcount = self.allocator.create(usize) catch |err| {
            self.allocator.free(new_data);
            original_mutex.unlock();
            return err;
        };
        new_refcount.* = 1;
        
        const new_cow = self.allocator.create(bool) catch |err| {
            self.allocator.destroy(new_refcount);
            self.allocator.free(new_data);
            original_mutex.unlock();
            return err;
        };
        new_cow.* = false;
        
        const new_mutex = self.allocator.create(std.Thread.Mutex) catch |err| {
            self.allocator.destroy(new_cow);
            self.allocator.destroy(new_refcount);
            self.allocator.free(new_data);
            original_mutex.unlock();
            return err;
        };
        new_mutex.* = .{};
        
        const old_refcount = @atomicRmw(usize, self.refcount, .Sub, 1, .AcqRel);
        const was_last = (old_refcount == 1);
        const old_data = self.data;
        const old_refcount_ptr = self.refcount;
        const old_cow_ptr = self.cow;
        const old_mutex_ptr = self.mutex;
        
        self.data = new_data;
        self.refcount = new_refcount;
        self.cow = new_cow;
        self.mutex = new_mutex;
        
        original_mutex.unlock();
        
        if (was_last) {
            self.allocator.free(old_data);
            self.allocator.destroy(old_refcount_ptr);
            self.allocator.destroy(old_cow_ptr);
            self.allocator.destroy(old_mutex_ptr);
        }
    }

    fn markShared(self: *Tensor) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.cow.* = true;
    }

    pub fn newView(self: *Tensor, shape: Shape) !Tensor {
        const shape_size = try shape.totalSize();
        const self_size = try self.shape.totalSize();
        if (shape_size != self_size) return Error.InvalidShape;
        self.retain();
        self.markShared();
        return .{ .data = self.data, .shape = shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .mutex = self.mutex };
    }

    pub fn reshape(self: *Tensor, new_shape: []const usize) !void {
        if (new_shape.len == 0) return Error.InvalidShape;
        var total: usize = 1;
        for (new_shape) |dim| total *= dim;
        const self_size = try self.shape.totalSize();
        if (total != self_size) return Error.InvalidShape;
        const new_sh = try Shape.init(self.allocator, new_shape);
        self.shape.deinit(self.allocator);
        self.shape = new_sh;
    }

    pub fn view(self: *Tensor, new_shape: []const usize) !Tensor {
        if (new_shape.len == 0) return Error.InvalidShape;
        var total: usize = 1;
        for (new_shape) |dim| total *= dim;
        const self_size = try self.shape.totalSize();
        if (total != self_size) return Error.InvalidShape;
        const new_sh = try Shape.init(self.allocator, new_shape);
        self.retain();
        self.markShared();
        return .{ .data = self.data, .shape = new_sh, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .mutex = self.mutex };
    }

    pub fn slice(self: *Tensor, starts: []const usize, ends: []const usize) !Tensor {
        if (starts.len != self.shape.dims.len or ends.len != self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try self.allocator.alloc(usize, self.shape.dims.len);
        errdefer self.allocator.free(new_dims);
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (starts[i] >= ends[i] or ends[i] > self.shape.dims[i]) return Error.OutOfBounds;
            new_dims[i] = ends[i] - starts[i];
        }
        const new_sh = try Shape.init(self.allocator, new_dims);
        self.allocator.free(new_dims);
        var offset: usize = 0;
        i = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            offset += starts[i] * self.shape.strides[i];
        }
        const new_data = self.data[offset..];
        self.retain();
        self.markShared();
        return .{ .data = new_data, .shape = new_sh, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .mutex = self.mutex };
    }

    pub fn transpose(self: *Tensor, axes: []const usize) !Tensor {
        if (axes.len != self.shape.dims.len) return Error.InvalidAxis;
        var new_dims = try self.allocator.alloc(usize, self.shape.dims.len);
        errdefer self.allocator.free(new_dims);
        var new_strides = try self.allocator.alloc(usize, self.shape.dims.len);
        errdefer self.allocator.free(new_strides);
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (axes[i] >= self.shape.dims.len) return Error.InvalidAxis;
            new_dims[i] = self.shape.dims[axes[i]];
            new_strides[i] = self.shape.strides[axes[i]];
        }
        const new_sh = .{ .dims = new_dims, .strides = new_strides };
        self.retain();
        self.markShared();
        return .{ .data = self.data, .shape = new_sh, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .mutex = self.mutex };
    }

    fn computeIndex(self: *const Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.dims.len) return Error.InvalidAxis;
        var idx: usize = 0;
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            if (indices[i] >= self.shape.dims[i]) return Error.OutOfBounds;
            idx += indices[i] * self.shape.strides[i];
        }
        return idx;
    }

    pub fn get(self: *const Tensor, indices: []const usize) !f32 {
        const idx = try computeIndex(self, indices);
        return self.data[idx];
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) !void {
        try ensureWritable(self);
        const idx = try computeIndex(self, indices);
        self.data[idx] = value;
    }

    pub fn fill(self: *Tensor, value: f32) !void {
        try ensureWritable(self);
        @memset(self.data, value);
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const b_ptr: *const [4]f32 = @ptrCast(other.data[i..].ptr);
            const a: V = a_ptr.*;
            const b: V = b_ptr.*;
            const c = a + b;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const b_ptr: *const [4]f32 = @ptrCast(other.data[i..].ptr);
            const a: V = a_ptr.*;
            const b: V = b_ptr.*;
            const c = a - b;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    pub fn mul(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const b_ptr: *const [4]f32 = @ptrCast(other.data[i..].ptr);
            const a: V = a_ptr.*;
            const b: V = b_ptr.*;
            const c = a * b;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] *= other.data[i];
        }
    }

    pub fn div(self: *Tensor, other: *const Tensor) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const b_ptr: *const [4]f32 = @ptrCast(other.data[i..].ptr);
            const a: V = a_ptr.*;
            const b: V = b_ptr.*;
            const zero: V = @splat(0.0);
            const has_zero = @reduce(.Or, b == zero);
            if (has_zero) {
                var j: usize = 0;
                while (j < 4) : (j += 1) {
                    if (other.data[i + j] == 0.0) return Error.DivideByZero;
                    self.data[i + j] /= other.data[i + j];
                }
            } else {
                const c = a / b;
                const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
                out_ptr.* = c;
            }
        }
        while (i < N) : (i += 1) {
            if (other.data[i] == 0.0) return Error.DivideByZero;
            self.data[i] /= other.data[i];
        }
    }

    pub fn addScalar(self: *Tensor, scalar: f32) !void {
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        const s: V = @splat(scalar);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const a: V = a_ptr.*;
            const c = a + s;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] += scalar;
        }
    }

    pub fn subScalar(self: *Tensor, scalar: f32) !void {
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        const s: V = @splat(scalar);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const a: V = a_ptr.*;
            const c = a - s;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] -= scalar;
        }
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) !void {
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        const s: V = @splat(scalar);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const a: V = a_ptr.*;
            const c = a * s;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] *= scalar;
        }
    }

    pub fn divScalar(self: *Tensor, scalar: f32) !void {
        if (scalar == 0.0) return Error.DivideByZero;
        try ensureWritable(self);
        const N = self.data.len;
        const V = @Vector(4, f32);
        const s: V = @splat(scalar);
        var i: usize = 0;
        while (i + 4 <= N) : (i += 4) {
            const a_ptr: *const [4]f32 = @ptrCast(self.data[i..].ptr);
            const a: V = a_ptr.*;
            const c = a / s;
            const out_ptr: *[4]f32 = @ptrCast(self.data[i..].ptr);
            out_ptr.* = c;
        }
        while (i < N) : (i += 1) {
            self.data[i] /= scalar;
        }
    }

    pub fn exp(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = @exp(val.*);
        }
    }

    pub fn log(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            if (val.* <= 0.0) {
                val.* = -math.inf(f32);
            } else {
                val.* = @log(val.*);
            }
        }
    }

    pub fn sin(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = @sin(val.*);
        }
    }

    pub fn cos(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = @cos(val.*);
        }
    }

    pub fn tan(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = @tan(val.*);
        }
    }

    pub fn sqrt(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            if (val.* < 0.0) {
                val.* = math.nan(f32);
            } else {
                val.* = @sqrt(val.*);
            }
        }
    }

    pub fn pow(self: *Tensor, exponent: f32) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            if (val.* < 0.0 and @floor(exponent) != exponent) {
                val.* = math.nan(f32);
            } else if (val.* == 0.0 and exponent < 0.0) {
                val.* = math.inf(f32);
            } else {
                val.* = math.pow(f32, val.*, exponent);
            }
        }
    }

    pub fn abs(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = @fabs(val.*);
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
        const new_size = try result.shape.totalSize();
        var out_idx: usize = 0;
        while (out_idx < new_size) : (out_idx += 1) {
            var max_val: f32 = -math.inf(f32);
            var in_indices = try allocator.alloc(usize, self.shape.dims.len);
            defer allocator.free(in_indices);
            @memset(in_indices, 0);
            var carry = true;
            var dim_idx: usize = 0;
            while (carry and dim_idx < self.shape.dims.len) : (dim_idx += 1) {
                if (dim_idx == axis) continue;
                in_indices[dim_idx] = result.computeIndexFromFlat(out_idx, new_dims);
                carry = false;
            }
            var k: usize = 0;
            while (k < self.shape.dims[axis]) : (k += 1) {
                in_indices[axis] = k;
                const val = try self.get(in_indices);
                if (val > max_val) max_val = val;
            }
            result.data[out_idx] = max_val;
        }
        return result;
    }

    fn computeIndexFromFlat(self: *const Tensor, flat: usize, shape: []const usize) usize {
        var rem = flat;
        var idx: usize = 0;
        var i: usize = shape.len;
        while (i > 0) : (i -= 1) {
            const stride = self.shape.strides[i - 1];
            const pos = rem / stride;
            rem = rem % stride;
            if (i - 1 == 0) idx = pos;
        }
        return idx;
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
        const new_size = try result.shape.totalSize();
        var out_idx: usize = 0;
        while (out_idx < new_size) : (out_idx += 1) {
            var min_val: f32 = math.inf(f32);
            var in_indices = try allocator.alloc(usize, self.shape.dims.len);
            defer allocator.free(in_indices);
            @memset(in_indices, 0);
            var carry = true;
            var dim_idx: usize = 0;
            while (carry and dim_idx < self.shape.dims.len) : (dim_idx += 1) {
                if (dim_idx == axis) continue;
                in_indices[dim_idx] = result.computeIndexFromFlat(out_idx, new_dims);
                carry = false;
            }
            var k: usize = 0;
            while (k < self.shape.dims[axis]) : (k += 1) {
                in_indices[axis] = k;
                const val = try self.get(in_indices);
                if (val < min_val) min_val = val;
            }
            result.data[out_idx] = min_val;
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
        const new_size = try result.shape.totalSize();
        var out_idx: usize = 0;
        while (out_idx < new_size) : (out_idx += 1) {
            var total: f32 = 0.0;
            var in_indices = try allocator.alloc(usize, self.shape.dims.len);
            defer allocator.free(in_indices);
            @memset(in_indices, 0);
            var carry = true;
            var dim_idx: usize = 0;
            while (carry and dim_idx < self.shape.dims.len) : (dim_idx += 1) {
                if (dim_idx == axis) continue;
                in_indices[dim_idx] = result.computeIndexFromFlat(out_idx, new_dims);
                carry = false;
            }
            var k: usize = 0;
            while (k < self.shape.dims[axis]) : (k += 1) {
                in_indices[axis] = k;
                total += try self.get(in_indices);
            }
            result.data[out_idx] = total;
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
                            var ll = l;
                            while (ll < @min(l + TILE, k)) : (ll += 1) {
                                c.data[ii * n + jj] += a.data[ii * k + ll] * b.data[ll * n + jj];
                            }
                        }
                    }
                }
            }
        }
        return c;
    }

    pub fn broadcast(self: *const Tensor, target_shape: []const usize) !Tensor {
        const new_sh = try Shape.init(self.allocator, target_shape);
        if (!self.shape.broadcastCompatible(&new_sh)) return Error.ShapeMismatch;
        const result = try init(self.allocator, target_shape);
        var target_indices = try self.allocator.alloc(usize, target_shape.len);
        defer self.allocator.free(target_indices);
        @memset(target_indices, 0);
        const offset = target_shape.len - self.shape.dims.len;
        var flat_idx: usize = 0;
        while (flat_idx < result.data.len) : (flat_idx += 1) {
            var src_idx: usize = 0;
            var i: usize = 0;
            while (i < self.shape.dims.len) : (i += 1) {
                const target_i = target_indices[offset + i];
                const src_i = if (self.shape.dims[i] == 1) 0 else target_i;
                src_idx += src_i * self.shape.strides[i];
            }
            result.data[flat_idx] = self.data[src_idx];
            var carry = true;
            var dim = target_shape.len;
            while (carry and dim > 0) : (dim -= 1) {
                target_indices[dim - 1] += 1;
                if (target_indices[dim - 1] < target_shape[dim - 1]) {
                    carry = false;
                } else {
                    target_indices[dim - 1] = 0;
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
        return self.broadcast(allocator, new_dims);
    }

    pub fn relu(self: *Tensor) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = @max(0.0, val.*);
        }
    }

    pub fn softmax(self: *Tensor, axis: usize) !void {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        try self.ensureWritable();
        if (self.shape.dims.len == 1) {
            var max_val: f32 = -math.inf(f32);
            for (self.data) |val| {
                if (val > max_val) max_val = val;
            }
            var sum_val: f32 = 0.0;
            for (self.data) |*val| {
                val.* = @exp(val.* - max_val);
                sum_val += val.*;
            }
            const epsilon: f32 = 1e-10;
            if (sum_val < epsilon) sum_val = epsilon;
            for (self.data) |*val| {
                val.* /= sum_val;
            }
            return;
        }
        var max_t = try self.max(self.allocator, axis);
        defer max_t.deinit();
        var b_shape = try self.allocator.alloc(usize, self.shape.dims.len);
        defer self.allocator.free(b_shape);
        @memcpy(b_shape, self.shape.dims);
        b_shape[axis] = 1;
        var b_max = try max_t.broadcast(b_shape);
        defer b_max.deinit();
        try self.sub(&b_max);
        try self.exp();
        var sum_t = try self.sum(self.allocator, axis);
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
        for (t.data) |*val| {
            val.* = prng.float() * (max_val - min_val) + min_val;
        }
        return t;
    }

    pub fn randomNormal(allocator: Allocator, shape: []const usize, mean_val: f32, stddev_val: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        const t = try init(allocator, shape);
        for (t.data) |*val| {
            var u = prng.float();
            var v = prng.float();
            while (u <= 0.0) u = prng.float();
            while (v == 0.0) v = prng.float();
            const z = @sqrt(-2.0 * @log(u)) * @cos(2.0 * math.pi * v);
            val.* = mean_val + stddev_val * z;
        }
        return t;
    }

    pub fn identity(allocator: Allocator, n: usize) !Tensor {
        const t = try init(allocator, &.{ n, n });
        var i: usize = 0;
        while (i < n) : (i += 1) {
            t.data[i * n + i] = 1.0;
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
        var padded_input = if (padding[0] > 0 or padding[1] > 0) try self.pad(allocator, &.{ .{ padding[0], padding[0] }, .{ padding[1], padding[1] }, .{ 0, 0 }, .{ 0, 0 } }) else self.*;
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

    pub fn pad(self: *const Tensor, allocator: Allocator, pads: [][2]usize) !Tensor {
        if (pads.len != self.shape.dims.len) return Error.InvalidPads;
        var new_shape = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(new_shape);
        var i: usize = 0;
        while (i < self.shape.dims.len) : (i += 1) {
            new_shape[i] = self.shape.dims[i] + pads[i][0] + pads[i][1];
        }
        const new_t = try init(allocator, new_shape);
        try new_t.fill(0.0);
        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);
        @memset(indices, 0);
        while (true) {
            var src_indices = try allocator.alloc(usize, self.shape.dims.len);
            defer allocator.free(src_indices);
            var is_pad = false;
            i = 0;
            while (i < self.shape.dims.len) : (i += 1) {
                if (indices[i] < pads[i][0] or indices[i] >= pads[i][0] + self.shape.dims[i]) {
                    is_pad = true;
                } else {
                    src_indices[i] = indices[i] - pads[i][0];
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
        while (true) {
            var src_indices = try allocator.alloc(usize, self.shape.dims.len);
            defer allocator.free(src_indices);
            i = 0;
            while (i < self.shape.dims.len) : (i += 1) {
                src_indices[i] = indices[i] % self.shape.dims[i];
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
        var reduced_shape = try allocator.alloc(usize, ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = tensors[0].shape.dims[i];
                j += 1;
            }
        }
        var common_indices = try allocator.alloc(usize, ndim - 1);
        defer allocator.free(common_indices);
        @memset(common_indices, 0);
        while (true) {
            var current_offset: usize = 0;
            i = 0;
            while (i < tensors.len) : (i += 1) {
                const ten = tensors[i];
                var ax: usize = 0;
                while (ax < ten.shape.dims[axis]) : (ax += 1) {
                    var new_indices = try allocator.alloc(usize, ndim);
                    defer allocator.free(new_indices);
                    var ten_indices = try allocator.alloc(usize, ndim);
                    defer allocator.free(ten_indices);
                    var k: usize = 0;
                    var d: usize = 0;
                    while (d < ndim) : (d += 1) {
                        if (d == axis) {
                            new_indices[d] = current_offset + ax;
                            ten_indices[d] = ax;
                        } else {
                            new_indices[d] = common_indices[k];
                            ten_indices[d] = common_indices[k];
                            k += 1;
                        }
                    }
                    const val = try ten.get(ten_indices);
                    try new_t.set(new_indices, val);
                }
                current_offset += ten.shape.dims[axis];
            }
            var carry = true;
            var dim: isize = @as(isize, @intCast(ndim - 1)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                common_indices[@intCast(dim)] += 1;
                if (common_indices[@intCast(dim)] < reduced_shape[@intCast(dim)]) {
                    carry = false;
                } else {
                    common_indices[@intCast(dim)] = 0;
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
        const reduced_shape = try allocator.alloc(usize, ndim);
        defer allocator.free(reduced_shape);
        @memcpy(reduced_shape, tensors[0].shape.dims);
        var common_indices = try allocator.alloc(usize, ndim);
        defer allocator.free(common_indices);
        @memset(common_indices, 0);
        while (true) {
            var t_idx: usize = 0;
            while (t_idx < tensors.len) : (t_idx += 1) {
                const ten = tensors[t_idx];
                var new_indices = try allocator.alloc(usize, ndim + 1);
                defer allocator.free(new_indices);
                var ten_indices = try allocator.alloc(usize, ndim);
                defer allocator.free(ten_indices);
                var kk: usize = 0;
                var d: usize = 0;
                while (d < ndim + 1) : (d += 1) {
                    if (d == axis) {
                        new_indices[d] = t_idx;
                    } else {
                        new_indices[d] = common_indices[kk];
                        ten_indices[kk] = common_indices[kk];
                        kk += 1;
                    }
                }
                const val = try ten.get(ten_indices);
                try new_t.set(new_indices, val);
            }
            var carry = true;
            var dim: isize = @as(isize, @intCast(ndim)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                common_indices[@intCast(dim)] += 1;
                if (common_indices[@intCast(dim)] < reduced_shape[@intCast(dim)]) {
                    carry = false;
                } else {
                    common_indices[@intCast(dim)] = 0;
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
        const new_size = try result.shape.totalSize();
        var out_idx: usize = 0;
        while (out_idx < new_size) : (out_idx += 1) {
            var max_idx: usize = 0;
            var max_val: f32 = try self.getFromFlat(out_idx, max_idx, new_dims);
            var k: usize = 1;
            while (k < self.shape.dims[axis]) : (k += 1) {
                const val = try self.getFromFlat(out_idx, k, new_dims);
                if (val > max_val) {
                    max_val = val;
                    max_idx = k;
                }
            }
            result.data[out_idx] = @as(f32, @floatFromInt(max_idx));
        }
        return result;
    }

    fn getFromFlat(self: *const Tensor, flat: usize, axis_idx: usize, reduced_shape: []const usize) !f32 {
        const in_indices = try self.allocator.alloc(usize, self.shape.dims.len);
        defer self.allocator.free(in_indices);
        @memset(in_indices, 0);
        var rem = flat;
        var idx: usize = 0;
        for (reduced_shape) |dim| {
            if (dim > 0) {
                in_indices[idx] = rem % dim;
                rem = rem / dim;
            }
            idx += 1;
        }
        in_indices[self.shape.dims.len - 1] = axis_idx;
        return self.get(in_indices);
    }

    pub fn cumsum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var new_t = try self.copy(allocator);
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var indices = try allocator.alloc(usize, self.shape.dims.len);
        defer allocator.free(indices);
        @memset(indices, 0);
        while (true) {
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
        var_t.sqrt();
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
        const new_size = try result.shape.totalSize();
        var out_idx: usize = 0;
        while (out_idx < new_size) : (out_idx += 1) {
            var min_idx: usize = 0;
            var min_val: f32 = try self.getFromFlat(out_idx, min_idx, new_dims);
            var k: usize = 1;
            while (k < self.shape.dims[axis]) : (k += 1) {
                const val = try self.getFromFlat(out_idx, k, new_dims);
                if (val < min_val) {
                    min_val = val;
                    min_idx = k;
                }
            }
            result.data[out_idx] = @as(f32, @floatFromInt(min_idx));
        }
        return result;
    }

    pub fn sort(self: *const Tensor, allocator: Allocator, axis: usize, descending: bool) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        const new_t = try self.copy(allocator);
        const Context = struct {
            pub fn lessThan(_: void, a: f32, b: f32) bool {
                return a < b;
            }
            pub fn greaterThan(_: void, a: f32, b: f32) bool {
                return a > b;
            }
        };
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
        const flat_size = @reduce(.Mul, reduced_shape);
        var flat_idx: usize = 0;
        while (flat_idx < flat_size) : (flat_idx += 1) {
            var base_idx: usize = 0;
            var k: usize = 0;
            i = 0;
            while (i < self.shape.dims.len) : (i += 1) {
                if (i == axis) {
                    // skip
                } else {
                    base_idx += common_indices[k] * new_t.shape.strides[i];
                    k += 1;
                }
            }
            i = 0;
            while (i < self.shape.dims[axis]) : (i += 1) {
                const idx = base_idx + i * new_t.shape.strides[axis];
                temp[i] = new_t.data[idx];
            }
            if (descending) {
                std.mem.sort(f32, temp, {}, Context.greaterThan);
            } else {
                std.mem.sort(f32, temp, {}, Context.lessThan);
            }
            i = 0;
            while (i < self.shape.dims[axis]) : (i += 1) {
                const idx = base_idx + i * new_t.shape.strides[axis];
                new_t.data[idx] = temp[i];
            }
            var carry = true;
            var dim: isize = @as(isize, @intCast(self.shape.dims.len - 1)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                common_indices[@intCast(dim)] += 1;
                if (common_indices[@intCast(dim)] < reduced_shape[@intCast(dim)]) {
                    carry = false;
                } else {
                    common_indices[@intCast(dim)] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn unique(self: *const Tensor, allocator: Allocator) !Tensor {
        var unique_set = std.AutoHashMap(f32, void).init(allocator);
        defer unique_set.deinit();
        for (self.data) |val| {
            try unique_set.put(val, {});
        }
        const unique_len = unique_set.count();
        const unique_t = try init(allocator, &.{unique_len});
        var iter = unique_set.iterator();
        var i: usize = 0;
        while (iter.next()) |entry| {
            unique_t.data[i] = entry.key_ptr.*;
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
            const idx = @as(usize, @intFromFloat(try self.get(&.{i})));
            if (idx < num_classes) {
                try new_t.set(&.{ i, idx }, 1.0);
            }
        }
        return new_t;
    }

    pub fn isClose(self: *const Tensor, other: *const Tensor, rtol: f32, atol: f32) !bool {
        if (!self.shape.equals(&other.shape)) return false;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            const diff = @fabs(self.data[i] - other.data[i]);
            if (diff > atol + rtol * @fabs(other.data[i])) return false;
        }
        return true;
    }

    pub fn toInt(self: *const Tensor, allocator: Allocator) !Tensor {
        const new_t = try init(allocator, self.shape.dims);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            new_t.data[i] = @trunc(self.data[i]);
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
                radius += v.data[i] * self.data[i * n + i];
            }
            if (@fabs(radius - last_radius) < tol) return @fabs(radius);
            last_radius = radius;
        }
        return @fabs(last_radius);
    }

    pub fn normL2(self: *const Tensor) f32 {
        var sum_sq: f32 = 0.0;
        for (self.data) |val| {
            sum_sq += val * val;
        }
        return @sqrt(sum_sq);
    }

    pub fn dot(self: *const Tensor, other: *const Tensor) !f32 {
        if (self.data.len != other.data.len) return Error.ShapeMismatch;
        var sum_result: f32 = 0.0;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            sum_result += self.data[i] * other.data[i];
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
                    result.data[i * n + j] = a.data[i] * b.data[j];
                }
            }
            return result;
        } else if (a.shape.dims.len == 2 and b.shape.dims.len == 2) {
            if (a.shape.dims[0] != b.shape.dims[0]) return Error.ShapeMismatch;
            const batch = a.shape.dims[0];
            const m = a.shape.dims[1];
            const n = b.shape.dims[1];
            const result = try init(allocator, &.{ m, n });
            var batch_idx: usize = 0;
            while (batch_idx < batch) : (batch_idx += 1) {
                var i: usize = 0;
                while (i < m) : (i += 1) {
                    var j: usize = 0;
                    while (j < n) : (j += 1) {
                        result.data[i * n + j] += a.data[batch_idx * m + i] * b.data[batch_idx * n + j];
                    }
                }
            }
            return result;
        } else {
            return Error.ShapeMismatch;
        }
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
                if (@fabs(mat.data[j * n + i]) > @fabs(mat.data[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (@fabs(mat.data[pivot * n + i]) < 1e-10) return Error.SingularMatrix;
            if (pivot != i) {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const temp_mat = mat.data[i * n + k];
                    mat.data[i * n + k] = mat.data[pivot * n + k];
                    mat.data[pivot * n + k] = temp_mat;
                    const temp_inv = inv.data[i * n + k];
                    inv.data[i * n + k] = inv.data[pivot * n + k];
                    inv.data[pivot * n + k] = temp_inv;
                }
            }
            const diag = mat.data[i * n + i];
            var k: usize = 0;
            while (k < n) : (k += 1) {
                mat.data[i * n + k] /= diag;
                inv.data[i * n + k] /= diag;
            }
            j = 0;
            while (j < n) : (j += 1) {
                if (j != i) {
                    const factor = mat.data[j * n + i];
                    k = 0;
                    while (k < n) : (k += 1) {
                        mat.data[j * n + k] -= factor * mat.data[i * n + k];
                        inv.data[j * n + k] -= factor * inv.data[i * n + k];
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
                if (@fabs(mat.data[ii * n + (ii - 1)]) > 1e-10) {
                    converged = false;
                    break;
                }
            }
            if (converged) break;
        }
        var vals = try init(allocator, &.{n});
        var i: usize = 0;
        while (i < n) : (i += 1) {
            vals.data[i] = mat.data[i * n + i];
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
                x[i - j] = r.data[i * n + j];
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
            for (u) |*val| val.* /= norm_u;
            var k: usize = j;
            while (k < n) : (k += 1) {
                var dot_prod: f32 = 0.0;
                i = j;
                while (i < m) : (i += 1) {
                    dot_prod += r.data[i * n + k] * u[i - j];
                }
                dot_prod *= 2.0;
                i = j;
                while (i < m) : (i += 1) {
                    r.data[i * n + k] -= dot_prod * u[i - j];
                }
            }
            k = 0;
            while (k < m) : (k += 1) {
                var dot_prod: f32 = 0.0;
                i = j;
                while (i < m) : (i += 1) {
                    dot_prod += q.data[i * m + k] * u[i - j];
                }
                dot_prod *= 2.0;
                i = j;
                while (i < m) : (i += 1) {
                    q.data[i * m + k] -= dot_prod * u[i - j];
                }
            }
        }
        return .{ .q = q, .r = r };
    }

    pub fn svd(self: *const Tensor, allocator: Allocator) !struct { u: Tensor, s: Tensor, v: Tensor } {
        var ata = try self.transpose(&.{1, 0});
        defer ata.deinit();
        var ata_self = try matmul(&ata, self, allocator);
        defer ata_self.deinit();
        var eigen = try eig(&ata_self, allocator);
        var s = try init(allocator, &.{eigen.vals.shape.dims[0]});
        var i: usize = 0;
        while (i < s.data.len) : (i += 1) {
            s.data[i] = @sqrt(@max(0.0, eigen.vals.data[i]));
        }
        defer eigen.vals.deinit();
        var v = eigen.vecs;
        var u = try matmul(self, &v, allocator);
        i = 0;
        while (i < s.data.len) : (i += 1) {
            if (s.data[i] > 1e-10) {
                var jj: usize = 0;
                while (jj < u.shape.dims[0]) : (jj += 1) {
                    u.data[jj * u.shape.dims[1] + i] /= s.data[i];
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
                    sum_result += l.data[i * n + k] * l.data[j * n + k];
                }
                if (i == j) {
                    l.data[i * n + j] = @sqrt(self.data[i * n + j] - sum_result);
                } else {
                    l.data[i * n + j] = (self.data[i * n + j] - sum_result) / l.data[j * n + j];
                }
            }
        }
        return l;
    }

    pub fn solve(self: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        const lu_result = try self.lu(allocator);
        defer lu_result.l.deinit();
        defer lu_result.u.deinit();
        var y = try matmul(&lu_result.l, b, allocator);
        defer y.deinit();
        return matmul(&lu_result.u, &y, allocator);
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
                    sum_result += l.data[j * n + k] * u.data[k * n + i];
                }
                u.data[j * n + i] = self.data[j * n + i] - sum_result;
            }
            j = i + 1;
            while (j < n) : (j += 1) {
                var sum_result2: f32 = 0.0;
                var k: usize = 0;
                while (k < i) : (k += 1) {
                    sum_result2 += l.data[j * n + k] * u.data[k * n + i];
                }
                l.data[j * n + i] = (self.data[j * n + i] - sum_result2) / u.data[i * n + i];
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
            sum_result += self.data[i * n + i];
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
                if (@fabs(mat.data[j * n + i]) > @fabs(mat.data[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (@fabs(mat.data[pivot * n + i]) < 1e-10) return 0.0;
            if (pivot != i) {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const temp = mat.data[i * n + k];
                    mat.data[i * n + k] = mat.data[pivot * n + k];
                    mat.data[pivot * n + k] = temp;
                }
                det_val = -det_val;
            }
            det_val *= mat.data[i * n + i];
            j = i + 1;
            while (j < n) : (j += 1) {
                const factor = mat.data[j * n + i] / mat.data[i * n + i];
                var k: usize = i;
                while (k < n) : (k += 1) {
                    mat.data[j * n + k] -= factor * mat.data[i * n + k];
                }
            }
        }
        return det_val;
    }

    pub fn clip(self: *Tensor, min_val: f32, max_val: f32) !void {
        try ensureWritable(self);
        for (self.data) |*val| {
            val.* = math.clamp(val.*, min_val, max_val);
        }
    }

    pub fn norm(self: *const Tensor, order: f32) f32 {
        var total: f32 = 0.0;
        for (self.data) |val| {
            total += math.pow(f32, @fabs(val), order);
        }
        return math.pow(f32, total, 1.0 / order);
    }

    pub fn toFixed(self: *const Tensor, allocator: Allocator) !Tensor {
        const fixed_t = try init(allocator, self.shape.dims);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            fixed_t.data[i] = Fixed32_32.init(self.data[i]).toFloat();
        }
        return fixed_t;
    }

    pub fn arange(allocator: Allocator, start: f32, end: f32, step: f32) !Tensor {
        const size = @as(usize, @intFromFloat(@ceil((end - start) / step)));
        const t = try init(allocator, &.{size});
        var val = start;
        for (t.data) |*d| {
            d.* = val;
            val += step;
        }
        return t;
    }

    pub fn linspace(allocator: Allocator, start: f32, end: f32, num: usize) !Tensor {
        const t = try init(allocator, &.{num});
        if (num == 0) return t;
        const step = (end - start) / @as(f32, @floatFromInt(num - 1));
        var val = start;
        var i: usize = 0;
        while (i < num - 1) : (i += 1) {
            t.data[i] = val;
            val += step;
        }
        t.data[num - 1] = end;
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
        i = 0;
        while (i < self.data.len) : (i += 1) {
            const val = self.data[i];
            try writer.print("{d:.4}", .{val});
            if (i < self.data.len - 1) try writer.print(", ", .{});
        }
        try writer.print("])", .{});
        return buf.toOwnedSlice();
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        try writer.writeInt(u64, @as(u64, self.shape.dims.len), .Little);
        for (self.shape.dims) |dim| {
            try writer.writeInt(u64, @as(u64, dim), .Little);
        }
        for (self.data) |val| {
            const val_bits: u32 = @bitCast(val);
            try writer.writeInt(u32, val_bits, .Little);
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const ndim64 = try reader.readInt(u64, .Little);
        if (ndim64 > math.maxInt(usize)) return Error.Overflow;
        const ndim: usize = @intCast(ndim64);
        if (ndim == 0) return Error.EmptyInput;
        if (ndim > 16) return Error.InvalidShape;
        var shape = try allocator.alloc(usize, ndim);
        errdefer allocator.free(shape);
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            const dim64 = try reader.readInt(u64, .Little);
            if (dim64 > math.maxInt(usize)) return Error.Overflow;
            shape[i] = @intCast(dim64);
        }
        const tensor = try init(allocator, shape);
        allocator.free(shape);
        for (tensor.data) |*val| {
            const val_bits = try reader.readInt(u32, .Little);
            val.* = @bitCast(val_bits);
        }
        return tensor;
    }
};

test "Tensor basic ops" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t1 = try Tensor.init(gpa, &.{ 2, 2 });
    defer t1.deinit();
    try t1.fill(1.0);
    var t2 = try Tensor.init(gpa, &.{ 2, 2 });
    defer t2.deinit();
    try t2.fill(2.0);
    try t1.add(&t2);
    try testing.expectEqual(@as(f32, 3.0), t1.data[0]);
}

test "Matmul" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var a = try Tensor.init(gpa, &.{ 2, 3 });
    defer a.deinit();
    @memcpy(a.data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    var b = try Tensor.init(gpa, &.{ 3, 2 });
    defer b.deinit();
    @memcpy(b.data, &[_]f32{ 7, 8, 9, 10, 11, 12 });
    var c = try Tensor.matmul(&a, &b, gpa);
    defer c.deinit();
    try testing.expectApproxEqAbs(@as(f32, 58.0), c.data[0], 1e-5);
}

test "Sum reduce" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t = try Tensor.init(gpa, &.{ 2, 3 });
    defer t.deinit();
    @memcpy(t.data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    var sum = try t.sum(gpa, 1);
    defer sum.deinit();
    try testing.expectEqual(@as(f32, 6.0), sum.data[0]);
    try testing.expectEqual(@as(f32, 15.0), sum.data[1]);
}

test "Broadcast" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t1 = try Tensor.init(gpa, &.{3});
    defer t1.deinit();
    @memcpy(t1.data, &[_]f32{ 1, 2, 3 });
    var b_t1 = try t1.broadcast(gpa, &.{ 2, 3 });
    defer b_t1.deinit();
    try testing.expectEqual(@as(f32, 1.0), b_t1.data[0]);
}

test "Softmax" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t = try Tensor.init(gpa, &.{3});
    defer t.deinit();
    @memcpy(t.data, &[_]f32{ 1, 2, 3 });
    try t.softmax(0);
    try testing.expectApproxEqAbs(@as(f32, 0.0900), t.data[0], 1e-3);
}

test "View reshape" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t = try Tensor.init(gpa, &.{ 2, 2 });
    defer t.deinit();
    @memcpy(t.data, &[_]f32{ 1, 2, 3, 4 });
    var v = try t.view(&.{ 4 });
    defer v.deinit();
    try testing.expectEqual(@as(f32, 1.0), v.data[0]);
    try testing.expectEqual(@as(f32, 4.0), v.data[3]);
    try t.set(&.{0,0}, 99.0);
    try testing.expectEqual(@as(f32, 99.0), v.data[0]);
}

test "Copy on Write" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t = try Tensor.init(gpa, &.{ 2, 2 });
    defer t.deinit();
    @memcpy(t.data, &[_]f32{ 1, 2, 3, 4 });
    var v = try t.view(&.{ 4 });
    defer v.deinit();
    try v.set(&.{0}, 99.0);
    try testing.expectEqual(@as(f32, 99.0), v.data[0]);
    try testing.expectEqual(@as(f32, 1.0), t.data[0]);
}