const std = @import("std");
const Tensor = @import("../src/core/tensor.zig").Tensor;
const types = @import("../src/core/types.zig");

const FuzzConfig = struct {
    iterations: usize = 5000,
    max_dim_size: usize = 256,
    max_rank: usize = 4,
    seed: u64 = 42,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = FuzzConfig{};

    std.debug.print("Fuzz testing tensor operations...\n", .{});
    std.debug.print("Iterations: {d}\n", .{config.iterations});
    std.debug.print("Max dimension size: {d}\n", .{config.max_dim_size});
    std.debug.print("Max rank: {d}\n\n", .{config.max_rank});

    var prng = std.rand.DefaultPrng.init(config.seed);
    const rand = prng.random();

    var successful_ops: usize = 0;
    var failed_ops: usize = 0;

    var i: usize = 0;
    while (i < config.iterations) : (i += 1) {
        const rank = rand.intRangeAtMost(usize, 1, config.max_rank);
        var shape = std.ArrayList(usize).init(allocator);
        defer shape.deinit();

        var total_elements: usize = 1;
        var j: usize = 0;
        while (j < rank) : (j += 1) {
            const dim = rand.intRangeAtMost(usize, 1, config.max_dim_size);
            try shape.append(dim);
            total_elements *= dim;
        }

        if (total_elements > 1024 * 1024) {
            continue;
        }

        var tensor = Tensor.init(allocator, shape.items) catch {
            failed_ops += 1;
            continue;
        };
        defer tensor.deinit();

        for (tensor.data) |*val| {
            val.* = rand.float(f32) * 2.0 - 1.0;
        }

        const operation = rand.intRangeAtMost(u8, 0, 3);

        const result = switch (operation) {
            0 => blk: {
                var sum: f32 = 0.0;
                for (tensor.data) |val| {
                    sum += val;
                }
                break :blk sum;
            },
            1 => blk: {
                var max: f32 = -std.math.inf(f32);
                for (tensor.data) |val| {
                    if (val > max) max = val;
                }
                break :blk max;
            },
            2 => blk: {
                var sum_sq: f32 = 0.0;
                for (tensor.data) |val| {
                    sum_sq += val * val;
                }
                break :blk @sqrt(sum_sq);
            },
            3 => blk: {
                const scale = rand.float(f32) * 2.0;
                for (tensor.data) |*val| {
                    val.* *= scale;
                }
                break :blk scale;
            },
            else => unreachable,
        };

        if (std.math.isNan(result) or std.math.isInf(result)) {
            failed_ops += 1;
            std.debug.print("WARNING: Invalid result detected at iteration {d}\n", .{i});
        } else {
            successful_ops += 1;
        }

        if (i % 500 == 0 and i > 0) {
            std.debug.print("Progress: {d}/{d} iterations\n", .{i, config.iterations});
        }
    }

    std.debug.print("\nFuzz test completed!\n", .{});
    std.debug.print("Successful operations: {d}\n", .{successful_ops});
    std.debug.print("Failed operations: {d}\n", .{failed_ops});

    if (failed_ops > config.iterations / 10) {
        std.debug.print("\nWARNING: High failure rate detected!\n", .{});
        return error.HighFailureRate;
    }
}