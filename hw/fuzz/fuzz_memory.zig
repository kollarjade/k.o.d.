const std = @import("std");
const Memory = @import("../src/core/memory.zig");

const FuzzConfig = struct {
    iterations: usize = 10000,
    max_alloc_size: usize = 1024 * 1024,
    seed: u64 = 0,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = FuzzConfig{};

    std.debug.print("Fuzz testing memory allocation system...\n", .{});
    std.debug.print("Iterations: {d}\n", .{config.iterations});
    std.debug.print("Max allocation size: {d} bytes\n\n", .{config.max_alloc_size});

    var prng = std.rand.DefaultPrng.init(config.seed);
    const rand = prng.random();

    var allocations = std.ArrayList([]u8).init(allocator);
    defer {
        for (allocations.items) |allocation| {
            allocator.free(allocation);
        }
        allocations.deinit();
    }

    var successful_allocs: usize = 0;
    var successful_frees: usize = 0;
    var failed_allocs: usize = 0;

    var i: usize = 0;
    while (i < config.iterations) : (i += 1) {
        const operation = rand.intRangeAtMost(u8, 0, 2);

        switch (operation) {
            0 => {
                const size = rand.intRangeAtMost(usize, 1, config.max_alloc_size);
                const alignment = @as(usize, 1) << rand.intRangeAtMost(u6, 0, 6);

                if (allocator.alignedAlloc(u8, alignment, size)) |allocation| {
                    @memset(allocation, @as(u8, @truncate(rand.int(u8))));
                    try allocations.append(allocation);
                    successful_allocs += 1;
                } else |_| {
                    failed_allocs += 1;
                }
            },
            1 => {
                if (allocations.items.len > 0) {
                    const index = rand.intRangeLessThan(usize, 0, allocations.items.len);
                    const allocation = allocations.swapRemove(index);
                    allocator.free(allocation);
                    successful_frees += 1;
                }
            },
            2 => {
                if (allocations.items.len > 0) {
                    const index = rand.intRangeLessThan(usize, 0, allocations.items.len);
                    const old_allocation = allocations.items[index];
                    const new_size = rand.intRangeAtMost(usize, 1, config.max_alloc_size);

                    if (allocator.realloc(old_allocation, new_size)) |new_allocation| {
                        allocations.items[index] = new_allocation;
                        successful_allocs += 1;
                    } else |_| {
                        failed_allocs += 1;
                    }
                }
            },
            else => unreachable,
        }

        if (i % 1000 == 0 and i > 0) {
            std.debug.print("Progress: {d}/{d} iterations, {d} active allocations\n", 
                .{i, config.iterations, allocations.items.len});
        }
    }

    std.debug.print("\nFuzz test completed successfully!\n", .{});
    std.debug.print("Successful allocations: {d}\n", .{successful_allocs});
    std.debug.print("Successful frees: {d}\n", .{successful_frees});
    std.debug.print("Failed allocations: {d}\n", .{failed_allocs});
    std.debug.print("Final active allocations: {d}\n", .{allocations.items.len});

    if (allocations.items.len > 0) {
        std.debug.print("\nWARNING: Memory leak detected! {d} allocations not freed\n", 
            .{allocations.items.len});
        return error.MemoryLeak;
    }
}