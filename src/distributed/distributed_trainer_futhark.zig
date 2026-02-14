const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray2DF16 = accel.FutharkArray2DF16;
const PinnedMemory = accel.PinnedMemory;

pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.9,
    max_line_size: usize = 10 * 1024 * 1024,
    checkpoint_version: u32 = 1,
};

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: RSFAccelerator,
    model_dim: usize,
    local_batch_size: usize,
    global_step: usize,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, local_batch_size, .{});
    }

    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        if (model_dim == 0) return error.InvalidModelDim;
        if (local_batch_size == 0) return error.InvalidBatchSize;
        if (coordinator.world_size == 0) return error.InvalidWorldSize;

        const vocab = &[_][]const u8{
            "a",     "about",   "all",   "also",  "and",   "as",    "at",
            "be",    "because", "but",   "by",    "can",   "come",  "could",
            "day",   "do",      "even",  "find",  "first", "for",   "from",
            "get",   "give",    "go",    "have",  "he",    "her",   "here",
            "him",   "his",     "how",   "i",     "if",    "in",    "into",
            "it",    "its",     "just",  "know",  "like",  "look",  "make",
            "man",   "many",    "me",    "more",  "my",    "new",   "no",
            "not",   "now",     "of",    "on",    "one",   "only",  "or",
            "other", "our",     "out",   "people", "say",  "see",   "she",
            "so",    "some",    "take",  "tell",  "than",  "that",  "the",
            "their", "them",    "then",  "there", "these", "they",  "thing",
            "think", "this",    "those", "time",  "to",    "two",   "up",
            "use",   "very",    "want",  "way",   "we",    "well",  "what",
            "when",  "which",   "who",   "will",  "with",  "would", "year",
            "you",   "your",
        };
        const empty_anchors: []const []const u8 = &.{};

        var tokenizer = try MGT.init(allocator, vocab, empty_anchors);
        errdefer tokenizer.deinit();

        var accelerator = try RSFAccelerator.init(model_dim);
        errdefer accelerator.deinit();

        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = tokenizer,
            .accelerator = accelerator,
            .model_dim = model_dim,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
        };
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch {};
        self.accelerator.deinit();
        self.tokenizer.deinit();
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        if (self.coordinator.world_size == 0) return error.InvalidWorldSize;

        var line_count: usize = 0;

        {
            const count_file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| {
                std.debug.print("[Rank {d}] ERROR: Cannot open dataset: {}\n", .{ self.coordinator.rank, err });
                return err;
            };
            defer count_file.close();

            var count_buf_reader = std.io.bufferedReader(count_file.reader());
            var count_stream = count_buf_reader.reader();

            while (true) {
                count_stream.skipUntilDelimiterOrEof('\n') catch break;
                line_count += 1;
            }
        }

        if (line_count == 0) {
            std.debug.print("[Rank {d}] ERROR: Dataset is empty\n", .{self.coordinator.rank});
            return error.EmptyDataset;
        }

        const base_samples_per_rank = line_count / self.coordinator.world_size;
        const remainder = line_count % self.coordinator.world_size;

        if (self.coordinator.rank >= self.coordinator.world_size) {
            return error.InvalidRank;
        }

        const samples_for_this_rank = if (self.coordinator.rank < remainder)
            base_samples_per_rank + 1
        else
            base_samples_per_rank;

        var start_line: usize = 0;
        var r: usize = 0;
        while (r < self.coordinator.rank) : (r += 1) {
            start_line += if (r < remainder) base_samples_per_rank + 1 else base_samples_per_rank;
        }

        const end_line = start_line +| samples_for_this_rank;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| {
                self.allocator.free(sample);
            }
            samples.deinit();
        }

        const load_file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| {
            std.debug.print("[Rank {d}] ERROR: Cannot reopen dataset: {}\n", .{ self.coordinator.rank, err });
            return err;
        };
        defer load_file.close();

        var load_buf_reader = std.io.bufferedReader(load_file.reader());
        var load_stream = load_buf_reader.reader();

        var current_line: usize = 0;
        while (current_line < start_line) : (current_line += 1) {
            load_stream.skipUntilDelimiterOrEof('\n') catch {
                return error.UnexpectedEndOfFile;
            };
        }

        while (current_line < end_line) : (current_line += 1) {
            const line = load_stream.readUntilDelimiterOrEofAlloc(
                self.allocator,
                '\n',
                self.config.max_line_size,
            ) catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            } orelse break;

            if (line.len == 0) {
                self.allocator.free(line);
                continue;
            }

            const parsed = std.json.parseFromSlice(
                std.json.Value,
                self.allocator,
                line,
                .{},
            ) catch {
                self.allocator.free(line);
                continue;
            };
            defer parsed.deinit();
            self.allocator.free(line);

            switch (parsed.value) {
                .object => |obj| {
                    if (obj.get("text")) |text_value| {
                        switch (text_value) {
                            .string => |text| {
                                if (text.len > 0) {
                                    const text_copy = try self.allocator.dupe(u8, text);
                                    errdefer self.allocator.free(text_copy);
                                    try samples.append(text_copy);
                                }
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }
        }

        if (self.coordinator.isRoot()) {
            const display_end = if (end_line > 0) end_line - 1 else 0;
            std.debug.print("[Rank {d}] Loaded {d} samples (lines {d}-{d} of {d} total)\n", .{
                self.coordinator.rank,
                samples.items.len,
                start_line,
                display_end,
                line_count,
            });
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (samples.len == 0) return 0.0;
        if (self.local_batch_size == 0) return error.InvalidBatchSize;

        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;

        var batch_start: usize = 0;
        while (batch_start < samples.len) {
            const batch_end = @min(batch_start + self.local_batch_size, samples.len);
            const batch = samples[batch_start..batch_end];

            const loss = try self.trainStepFuthark(batch);
            total_loss += loss;
            num_batches += 1;

            if (self.coordinator.isRoot() and self.global_step % 10 == 0) {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{ self.global_step, loss });
            }

            self.global_step +|= 1;
            batch_start = batch_end;
        }

        if (num_batches == 0) {
            std.debug.print("[WARNING] No batches processed\n", .{});
            return 0.0;
        }

        var loss_and_count = [2]f32{ total_loss, @floatFromInt(num_batches) };
        const loss_and_count_dev = try self.coordinator.allocDeviceMemory(2 * @sizeOf(f32));
        defer self.coordinator.freeDeviceMemory(loss_and_count_dev);

        try self.coordinator.copyHostToDevice(loss_and_count_dev, std.mem.asBytes(&loss_and_count), 2 * @sizeOf(f32));
        try self.coordinator.allReduceFloat32(loss_and_count_dev, loss_and_count_dev, 2);
        try self.coordinator.copyDeviceToHost(std.mem.asBytes(&loss_and_count), loss_and_count_dev, 2 * @sizeOf(f32));
        try self.coordinator.synchronize();

        const global_loss_sum = loss_and_count[0];
        const global_batch_count = loss_and_count[1];

        if (global_batch_count < 1.0) {
            std.debug.print("[WARNING] No batches processed across all ranks\n", .{});
            return 0.0;
        }

        return global_loss_sum / global_batch_count;
    }

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !f32 {
        if (batch.len == 0) return 0.0;

        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| {
                list.deinit();
            }
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            errdefer token_list.deinit();
            try self.tokenizer.encode(text, &token_list);
            try token_lists.append(token_list);
        }

        var max_seq_len: usize = 0;
        for (token_lists.items) |list| {
            max_seq_len = @max(max_seq_len, list.items.len);
        }

        if (max_seq_len == 0) return 0.0;

        const batch_size = batch.len;
        const data_elements = batch_size *| max_seq_len *| self.model_dim;
        if (data_elements == 0) return 0.0;

        const data_size = data_elements *| @sizeOf(f16);
        if (data_size / @sizeOf(f16) != data_elements) return error.Overflow;

        var pinned_mem = try PinnedMemory.alloc(data_size);
        defer pinned_mem.free();

        const input_f16_data = pinned_mem.asSlice(f16) orelse return error.AllocationFailed;
        @memset(input_f16_data, @as(f16, 0.0));

        var batch_idx: usize = 0;
        while (batch_idx < token_lists.items.len) : (batch_idx += 1) {
            const list = token_lists.items[batch_idx].items;
            var seq_idx: usize = 0;
            while (seq_idx < list.len) : (seq_idx += 1) {
                const token = list[seq_idx];
                if (token >= self.model_dim) return error.InvalidToken;

                const base_idx = (batch_idx *| max_seq_len +| seq_idx) *| self.model_dim;
                if (base_idx >= input_f16_data.len) return error.IndexOutOfBounds;

                const final_idx = base_idx +| token;
                if (final_idx >= input_f16_data.len) return error.IndexOutOfBounds;

                input_f16_data[final_idx] = @as(f16, 1.0);
            }
        }

        const total_rows = batch_size *| max_seq_len;
        var inputs = try FutharkArray2DF16.newFromFlat(&self.accelerator.ctx, input_f16_data, total_rows, self.model_dim);
        defer inputs.free(&self.accelerator.ctx);

        var targets = try FutharkArray2DF16.newFromFlat(&self.accelerator.ctx, input_f16_data, total_rows, self.model_dim);
        defer targets.free(&self.accelerator.ctx);

        const lr_f16: f16 = @floatCast(self.learning_rate);
        const mom_f16: f16 = @floatCast(self.momentum);

        const loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, lr_f16, mom_f16);

        const weights_s_ptr = try self.accelerator.getWeightsSDataPointer();
        const weights_t_ptr = try self.accelerator.getWeightsTDataPointer();

        const weight_count = self.model_dim *| self.model_dim;
        if (weight_count / self.model_dim != self.model_dim) return error.Overflow;

        try self.coordinator.allReduceFloat16(weights_s_ptr, weights_s_ptr, weight_count);
        try self.coordinator.allReduceFloat16(weights_t_ptr, weights_t_ptr, weight_count);
        try self.coordinator.synchronize();

        if (self.coordinator.world_size > 0) {
            const world_size_f16: f16 = @floatFromInt(self.coordinator.world_size);
            if (world_size_f16 > 0.0) {
                const scale_factor: f16 = @as(f16, 1.0) / world_size_f16;
                try self.accelerator.scaleWeights(scale_factor);
            }
        }
        try self.accelerator.sync();

        return @as(f32, @floatCast(loss_f16));
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        if (!self.coordinator.isRoot()) {
            return;
        }

        try self.coordinator.synchronize();

        const file = std.fs.createFileAbsolute(path, .{ .mode = 0o600 }) catch |err| {
            std.debug.print("Failed to create checkpoint file: {}\n", .{err});
            return err;
        };
        defer file.close();

        var buffered_writer = std.io.bufferedWriter(file.writer());
        var writer = buffered_writer.writer();

        try writer.writeInt(u32, self.config.checkpoint_version, .Little);
        try writer.writeInt(u64, @as(u64, @intCast(self.global_step)), .Little);
        try writer.writeInt(u64, @as(u64, @intCast(self.model_dim)), .Little);

        const weights_s_vals = try self.accelerator.weights_s.values(&self.accelerator.ctx, self.allocator);
        defer {
            for (weights_s_vals) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(weights_s_vals);
        }

        for (weights_s_vals) |row| {
            for (row) |weight| {
                const weight_f32: f32 = @floatCast(weight);
                try writer.writeInt(u32, @bitCast(weight_f32), .Little);
            }
        }

        const weights_t_vals = try self.accelerator.weights_t.values(&self.accelerator.ctx, self.allocator);
        defer {
            for (weights_t_vals) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(weights_t_vals);
        }

        for (weights_t_vals) |row| {
            for (row) |weight| {
                const weight_f32: f32 = @floatCast(weight);
                try writer.writeInt(u32, @bitCast(weight_f32), .Little);
            }
        }

        try buffered_writer.flush();
        try file.sync();

        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, self.global_step });
    }

    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const file = std.fs.openFileAbsolute(path, .{ .mode = .read_only }) catch |err| {
            std.debug.print("Failed to open checkpoint file: {}\n", .{err});
            return err;
        };
        defer file.close();

        var reader = file.reader();

        const version = try reader.readInt(u32, .Little);
        if (version != self.config.checkpoint_version) {
            return error.CheckpointVersionMismatch;
        }

        self.global_step = @intCast(try reader.readInt(u64, .Little));
        const saved_model_dim = try reader.readInt(u64, .Little);

        if (saved_model_dim != self.model_dim) {
            return error.ModelDimMismatch;
        }

        const weight_count = self.model_dim * self.model_dim;
        const s_weights = try self.allocator.alloc(f16, weight_count);
        defer self.allocator.free(s_weights);

        for (s_weights) |*w| {
            const bits = try reader.readInt(u32, .Little);
            const f32_val: f32 = @bitCast(bits);
            w.* = @floatCast(f32_val);
        }

        const t_weights = try self.allocator.alloc(f16, weight_count);
        defer self.allocator.free(t_weights);

        for (t_weights) |*w| {
            const bits = try reader.readInt(u32, .Little);
            const f32_val: f32 = @bitCast(bits);
            w.* = @floatCast(f32_val);
        }

        try self.accelerator.setWeightsS(s_weights, self.model_dim, self.model_dim);
        try self.accelerator.setWeightsT(t_weights, self.model_dim, self.model_dim);

        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }
};
