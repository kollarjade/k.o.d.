const std = @import("std");
const SSI = @import("../src/index/ssi.zig").SSI;

const FuzzConfig = struct {
    iterations: usize = 5000,
    max_tokens: usize = 1024,
    max_query_tokens: usize = 128,
    seed: u64 = 12345,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = FuzzConfig{};

    std.debug.print("Fuzz testing SSI (Succinct Semantic Index)...\n", .{});
    std.debug.print("Iterations: {d}\n", .{config.iterations});
    std.debug.print("Max tokens per document: {d}\n", .{config.max_tokens});
    std.debug.print("Max query tokens: {d}\n\n", .{config.max_query_tokens});

    var prng = std.rand.DefaultPrng.init(config.seed);
    const rand = prng.random();

    var successful_inserts: usize = 0;
    var successful_queries: usize = 0;
    var failed_ops: usize = 0;

    var documents = std.ArrayList([]u32).init(allocator);
    defer {
        for (documents.items) |doc| {
            allocator.free(doc);
        }
        documents.deinit();
    }

    var ssi = SSI.init(allocator);
    defer ssi.deinit();

    var i: usize = 0;
    while (i < config.iterations) : (i += 1) {
        const operation = rand.intRangeAtMost(u8, 0, 1);

        switch (operation) {
            0 => {
                const num_tokens = rand.intRangeAtMost(usize, 1, config.max_tokens);
                const tokens = try allocator.alloc(u32, num_tokens);

                for (tokens) |*token| {
                    token.* = rand.int(u32) % 50000;
                }

                const position = @as(u64, i);
                const is_anchor = rand.boolean();

                ssi.addSequence(tokens, position, is_anchor) catch {
                    allocator.free(tokens);
                    failed_ops += 1;
                    continue;
                };

                try documents.append(tokens);
                successful_inserts += 1;
            },
            1 => {
                if (documents.items.len > 0) {
                    const query_len = rand.intRangeAtMost(usize, 1, config.max_query_tokens);
                    const query = try allocator.alloc(u32, query_len);
                    defer allocator.free(query);

                    for (query) |*token| {
                        token.* = rand.int(u32) % 50000;
                    }

                    const results = ssi.retrieveTopK(query, 10, allocator) catch {
                        failed_ops += 1;
                        continue;
                    };

                    defer {
                        for (results) |*result| {
                            result.deinit(allocator);
                        }
                        allocator.free(results);
                    }

                    successful_queries += 1;
                } else {
                    failed_ops += 1;
                }
            },
            else => unreachable,
        }

        if (i % 500 == 0 and i > 0) {
            std.debug.print("Progress: {d}/{d} iterations, {d} documents indexed\n", 
                .{i, config.iterations, documents.items.len});
        }
    }

    std.debug.print("\nFuzz test completed!\n", .{});
    std.debug.print("Successful inserts: {d}\n", .{successful_inserts});
    std.debug.print("Successful queries: {d}\n", .{successful_queries});
    std.debug.print("Failed operations: {d}\n", .{failed_ops});
    std.debug.print("Total documents: {d}\n", .{documents.items.len});

    var total_tokens: usize = 0;
    for (documents.items) |doc| {
        total_tokens += doc.len;
    }
    std.debug.print("Total tokens indexed: {d}\n", .{total_tokens});
}