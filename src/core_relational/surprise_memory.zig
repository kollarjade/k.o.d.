const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Mutex = std.Thread.Mutex;

const chaos = @import("chaos_core.zig");
const MemoryBlock = chaos.MemoryBlock;
const MemoryBlockState = chaos.MemoryBlockState;
const ContentAddressableStorage = chaos.ContentAddressableStorage;
const DataFlowAnalyzer = chaos.DataFlowAnalyzer;

const RETENTION_AGE_WEIGHT: f64 = 0.3;
const RETENTION_FREQUENCY_WEIGHT: f64 = 0.2;
const RETENTION_BASE_WEIGHT: f64 = 0.5;
const NANOSECONDS_TO_MILLISECONDS: f64 = 1_000_000.0;
const HASH_SIZE: usize = 16;
const HASH_BITS: usize = HASH_SIZE * 8;
const MAX_INPUT_SIZE: usize = 100 * 1024 * 1024;
const JACCARD_SAMPLE_SIZE: usize = 1000;
const MAX_ENTANGLEMENT_PAIRS: usize = 100;
const DEFAULT_SURPRISE_THRESHOLD: f64 = 0.3;

pub const SurpriseMetrics = struct {
    jaccard_dissimilarity: f64,
    content_hash_distance: f64,
    temporal_novelty: f64,
    combined_surprise: f64,

    pub fn init(jaccard: f64, hash_dist: f64, temporal: f64) SurpriseMetrics {
        const clamped_jaccard = @max(0.0, @min(1.0, jaccard));
        const clamped_hash = @max(0.0, @min(1.0, hash_dist));
        const clamped_temporal = @max(0.0, @min(1.0, temporal));
        const combined = (clamped_jaccard + clamped_hash + clamped_temporal) / 3.0;
        return SurpriseMetrics{
            .jaccard_dissimilarity = clamped_jaccard,
            .content_hash_distance = clamped_hash,
            .temporal_novelty = clamped_temporal,
            .combined_surprise = combined,
        };
    }

    pub fn exceedsThreshold(self: *const SurpriseMetrics, threshold: f64) bool {
        return self.combined_surprise > threshold;
    }
};

pub const SurpriseRecord = struct {
    block_id: [HASH_SIZE]u8,
    surprise_score: f64,
    creation_time: i128,
    last_access_time: i128,
    retention_priority: f64,
    access_frequency: usize,

    pub fn init(block_id: [HASH_SIZE]u8, score: f64) SurpriseRecord {
        const now = std.time.nanoTimestamp();
        return SurpriseRecord{
            .block_id = block_id,
            .surprise_score = score,
            .creation_time = now,
            .last_access_time = now,
            .retention_priority = score,
            .access_frequency = 1,
        };
    }

    pub fn updateRetention(self: *SurpriseRecord) void {
        const now = std.time.nanoTimestamp();
        const age_ns = now - self.creation_time;
        const abs_age_ns: i128 = if (age_ns < 0) -age_ns else age_ns;
        const clamped_age: i64 = @intCast(@min(abs_age_ns, @as(i128, std.math.maxInt(i64))));
        const age_ms: f64 = @as(f64, @floatFromInt(clamped_age)) / NANOSECONDS_TO_MILLISECONDS;
        const age_factor = 1.0 / (1.0 + age_ms);
        const freq_val: f64 = @floatFromInt(self.access_frequency + 1);
        const frequency_factor = @log(freq_val);
        self.retention_priority = self.surprise_score * (RETENTION_BASE_WEIGHT + RETENTION_AGE_WEIGHT * age_factor + RETENTION_FREQUENCY_WEIGHT * frequency_factor);
        self.last_access_time = now;
    }

    pub fn recordAccess(self: *SurpriseRecord) void {
        self.access_frequency += 1;
        self.last_access_time = std.time.nanoTimestamp();
        self.updateRetention();
    }

    pub fn getRetentionPriority(self: *const SurpriseRecord) f64 {
        return self.retention_priority;
    }

    pub fn getAccessFrequency(self: *const SurpriseRecord) usize {
        return self.access_frequency;
    }
};

pub const SurpriseMemoryStatistics = struct {
    total_blocks: usize,
    high_surprise_blocks: usize,
    low_surprise_blocks: usize,
    average_surprise: f64,
    surprise_threshold: f64,
    evictions_due_to_low_surprise: usize,
    novel_block_allocations: usize,
    total_surprise_sum: f64,

    pub fn init(threshold: f64) SurpriseMemoryStatistics {
        return SurpriseMemoryStatistics{
            .total_blocks = 0,
            .high_surprise_blocks = 0,
            .low_surprise_blocks = 0,
            .average_surprise = 0.0,
            .surprise_threshold = threshold,
            .evictions_due_to_low_surprise = 0,
            .novel_block_allocations = 0,
            .total_surprise_sum = 0.0,
        };
    }

    pub fn addBlock(self: *SurpriseMemoryStatistics, surprise_score: f64, threshold: f64) void {
        self.total_blocks += 1;
        self.total_surprise_sum += surprise_score;
        if (surprise_score > threshold) {
            self.high_surprise_blocks += 1;
        } else {
            self.low_surprise_blocks += 1;
        }
        self.recalculateAverage();
    }

    pub fn removeBlock(self: *SurpriseMemoryStatistics, surprise_score: f64, threshold: f64) void {
        if (self.total_blocks > 0) {
            self.total_blocks -= 1;
            self.total_surprise_sum -= surprise_score;
            if (self.total_surprise_sum < 0.0) {
                self.total_surprise_sum = 0.0;
            }
            if (surprise_score > threshold) {
                if (self.high_surprise_blocks > 0) {
                    self.high_surprise_blocks -= 1;
                }
            } else {
                if (self.low_surprise_blocks > 0) {
                    self.low_surprise_blocks -= 1;
                }
            }
            self.recalculateAverage();
        }
    }

    fn recalculateAverage(self: *SurpriseMemoryStatistics) void {
        if (self.total_blocks > 0) {
            self.average_surprise = self.total_surprise_sum / @as(f64, @floatFromInt(self.total_blocks));
        } else {
            self.average_surprise = 0.0;
            self.total_surprise_sum = 0.0;
        }
    }
};

const CandidateItem = struct {
    block_id: [HASH_SIZE]u8,
    priority: f64,
};

pub const SurpriseMemoryManager = struct {
    storage: *ContentAddressableStorage,
    flow_analyzer: *DataFlowAnalyzer,
    surprise_records: std.HashMap([HASH_SIZE]u8, SurpriseRecord, chaos.BlockIdContext, std.hash_map.default_max_load_percentage),
    surprise_threshold: f64,
    statistics: SurpriseMemoryStatistics,
    allocator: Allocator,
    mutex: Mutex,
    owns_storage: bool,
    owns_analyzer: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, storage: *ContentAddressableStorage, analyzer: *DataFlowAnalyzer) Self {
        return Self{
            .storage = storage,
            .flow_analyzer = analyzer,
            .surprise_records = std.HashMap([HASH_SIZE]u8, SurpriseRecord, chaos.BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .surprise_threshold = DEFAULT_SURPRISE_THRESHOLD,
            .statistics = SurpriseMemoryStatistics.init(DEFAULT_SURPRISE_THRESHOLD),
            .allocator = allocator,
            .mutex = Mutex{},
            .owns_storage = false,
            .owns_analyzer = false,
        };
    }

    pub fn initWithOwnership(allocator: Allocator, storage: *ContentAddressableStorage, analyzer: *DataFlowAnalyzer, owns_storage: bool, owns_analyzer: bool) Self {
        var self = init(allocator, storage, analyzer);
        self.owns_storage = owns_storage;
        self.owns_analyzer = owns_analyzer;
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.surprise_records.deinit();
        if (self.owns_storage) {
            self.storage.deinit();
        }
        if (self.owns_analyzer) {
            self.flow_analyzer.deinit();
        }
    }

    pub fn setSurpriseThreshold(self: *Self, threshold: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const clamped_threshold = @max(0.0, @min(1.0, threshold));
        self.surprise_threshold = clamped_threshold;
        self.statistics.surprise_threshold = clamped_threshold;
    }

    pub fn getSurpriseThreshold(self: *Self) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_threshold;
    }

    pub fn computeSurprise(self: *Self, new_data: []const u8) !SurpriseMetrics {
        if (new_data.len > MAX_INPUT_SIZE) {
            return error.InputTooLarge;
        }

        const new_hash = computeContentHash(new_data);
        var min_jaccard_dist: f64 = 1.0;
        var min_hash_dist: f64 = 1.0;
        const block_count = self.storage.storage.count();

        if (block_count == 0) {
            return SurpriseMetrics.init(0.0, 0.0, 0.0);
        }

        var sample_count: usize = 0;
        const max_samples = @min(block_count, JACCARD_SAMPLE_SIZE);
        var storage_iter = self.storage.storage.iterator();

        while (storage_iter.next()) |entry| {
            if (sample_count >= max_samples) break;
            const existing_block = entry.value_ptr;
            const jaccard = try self.computeJaccardDistance(new_data, existing_block.data);
            if (jaccard < min_jaccard_dist) {
                min_jaccard_dist = jaccard;
            }
            const hash_dist = computeHashDistance(new_hash, existing_block.content_hash);
            if (hash_dist < min_hash_dist) {
                min_hash_dist = hash_dist;
            }
            sample_count += 1;
        }

        const count_f: f64 = @floatFromInt(block_count);
        const temporal_novelty = 1.0 / (1.0 + @sqrt(count_f));

        return SurpriseMetrics.init(min_jaccard_dist, min_hash_dist, temporal_novelty);
    }

    fn computeContentHash(data: []const u8) [HASH_SIZE]u8 {
        var hash_out: [32]u8 = undefined;
        Sha256.hash(data, &hash_out, .{});
        var result: [HASH_SIZE]u8 = undefined;
        @memcpy(&result, hash_out[0..HASH_SIZE]);
        return result;
    }

    fn computeJaccardDistance(self: *Self, data_a: []const u8, data_b: []const u8) !f64 {
        var set_a = std.AutoHashMap(u8, void).init(self.allocator);
        defer set_a.deinit();
        var set_b = std.AutoHashMap(u8, void).init(self.allocator);
        defer set_b.deinit();

        const sample_limit_a = @min(data_a.len, MAX_INPUT_SIZE);
        const sample_limit_b = @min(data_b.len, MAX_INPUT_SIZE);

        for (data_a[0..sample_limit_a]) |byte| {
            try set_a.put(byte, {});
        }
        for (data_b[0..sample_limit_b]) |byte| {
            try set_b.put(byte, {});
        }

        var intersection_count: usize = 0;
        var iter = set_a.iterator();
        while (iter.next()) |entry| {
            if (set_b.contains(entry.key_ptr.*)) {
                intersection_count += 1;
            }
        }

        const union_count = set_a.count() + set_b.count() - intersection_count;
        if (union_count == 0) return 0.0;

        const jaccard_similarity = @as(f64, @floatFromInt(intersection_count)) / @as(f64, @floatFromInt(union_count));
        return 1.0 - jaccard_similarity;
    }

    fn computeHashDistance(hash_a: [HASH_SIZE]u8, hash_b: [HASH_SIZE]u8) f64 {
        var hamming_dist: usize = 0;
        for (hash_a, hash_b) |a, b| {
            hamming_dist += @popCount(a ^ b);
        }
        return @as(f64, @floatFromInt(hamming_dist)) / @as(f64, @floatFromInt(HASH_BITS));
    }

    fn storeBlockInternal(self: *Self, data: []const u8, preferred_core: ?usize, surprise: SurpriseMetrics) ![HASH_SIZE]u8 {
        const block_id = try self.storage.store(data, preferred_core);
        const record = SurpriseRecord.init(block_id, surprise.combined_surprise);
        try self.surprise_records.put(block_id, record);
        return block_id;
    }

    pub fn storeWithSurprise(self: *Self, data: []const u8, preferred_core: ?usize) ![HASH_SIZE]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (data.len > MAX_INPUT_SIZE) {
            return error.InputTooLarge;
        }

        const existing_block_id = self.storage.retrieveByContent(data);
        if (existing_block_id) |block_id| {
            if (self.surprise_records.getPtr(block_id)) |record| {
                record.recordAccess();
                return block_id;
            } else {
                const surprise = try self.computeSurprise(data);
                const new_record = SurpriseRecord.init(block_id, surprise.combined_surprise);
                try self.surprise_records.put(block_id, new_record);
                self.statistics.addBlock(surprise.combined_surprise, self.surprise_threshold);
                return block_id;
            }
        }

        const surprise = try self.computeSurprise(data);
        const block_id = try self.storeBlockInternal(data, preferred_core, surprise);

        if (surprise.exceedsThreshold(self.surprise_threshold)) {
            self.statistics.novel_block_allocations += 1;
        }
        self.statistics.addBlock(surprise.combined_surprise, self.surprise_threshold);

        return block_id;
    }

    pub fn evictLowSurpriseBlocks(self: *Self, target_capacity: usize) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current_size = self.storage.storage.count();
        if (current_size <= target_capacity) return 0;

        const to_evict = current_size - target_capacity;
        var candidates = ArrayList(CandidateItem).init(self.allocator);
        defer candidates.deinit();

        try candidates.ensureTotalCapacity(self.surprise_records.count());

        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            candidates.appendAssumeCapacity(.{
                .block_id = entry.key_ptr.*,
                .priority = entry.value_ptr.retention_priority,
            });
        }

        if (candidates.items.len == 0) return 0;

        const k = @min(to_evict, candidates.items.len);
        partialSort(candidates.items, k);

        var evicted_count: usize = 0;
        for (candidates.items[0..k]) |candidate| {
            if (self.storage.containsBlock(candidate.block_id)) {
                if (self.surprise_records.get(candidate.block_id)) |record| {
                    self.statistics.removeBlock(record.surprise_score, self.surprise_threshold);
                }
                self.storage.removeBlock(candidate.block_id);
                _ = self.surprise_records.remove(candidate.block_id);
                evicted_count += 1;
                self.statistics.evictions_due_to_low_surprise += 1;
            }
        }

        return evicted_count;
    }

    fn partialSort(items: []CandidateItem, k: usize) void {
        if (items.len <= 1 or k == 0) return;
        const actual_k = @min(k, items.len);
        var i: usize = 0;
        while (i < actual_k) : (i += 1) {
            var min_idx: usize = i;
            var j: usize = i + 1;
            while (j < items.len) : (j += 1) {
                if (items[j].priority < items[min_idx].priority) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                const temp = items[i];
                items[i] = items[min_idx];
                items[min_idx] = temp;
            }
        }
    }

    pub fn organizeByEntanglement(self: *Self) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        var entangled_pairs: usize = 0;
        var high_surprise_ids: [MAX_ENTANGLEMENT_PAIRS][HASH_SIZE]u8 = undefined;
        var high_count: usize = 0;

        var iter = self.surprise_records.iterator();
        while (iter.next()) |entry| {
            if (high_count >= MAX_ENTANGLEMENT_PAIRS) break;
            if (entry.value_ptr.surprise_score > self.surprise_threshold) {
                high_surprise_ids[high_count] = entry.key_ptr.*;
                high_count += 1;
            }
        }

        var i: usize = 0;
        while (i < high_count) : (i += 1) {
            var j: usize = i + 1;
            while (j < high_count) : (j += 1) {
                const result = self.storage.entangleBlocks(high_surprise_ids[i], high_surprise_ids[j]);
                if (result) |_| {
                    entangled_pairs += 1;
                } else |_| {
                    continue;
                }
            }
        }

        return entangled_pairs;
    }

    pub fn getStatistics(self: *Self) SurpriseMemoryStatistics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.statistics;
    }

    pub fn getStatisticsConst(self: *const Self) SurpriseMemoryStatistics {
        return self.statistics;
    }

    pub fn getSurpriseRecord(self: *Self, block_id: [HASH_SIZE]u8) ?SurpriseRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.get(block_id);
    }

    pub fn getSurpriseRecordConst(self: *const Self, block_id: [HASH_SIZE]u8) ?SurpriseRecord {
        return self.surprise_records.get(block_id);
    }

    pub fn containsRecord(self: *Self, block_id: [HASH_SIZE]u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.contains(block_id);
    }

    pub fn getRecordCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.surprise_records.count();
    }
};

test "surprise_memory_basic" {
    const allocator = std.testing.allocator;

    var storage = ContentAddressableStorage.init(allocator);
    defer storage.deinit();

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var manager = SurpriseMemoryManager.init(allocator, &storage, &analyzer);
    defer manager.deinit();

    const data1 = "unique_data_content_1";
    const data2 = "unique_data_content_2";

    const block1 = try manager.storeWithSurprise(data1, null);
    const block2 = try manager.storeWithSurprise(data2, null);

    try std.testing.expect(!std.mem.eql(u8, &block1, &block2));

    const stats = manager.getStatistics();
    try std.testing.expect(stats.total_blocks >= 2);
}

test "surprise_metrics_validation" {
    const metrics = SurpriseMetrics.init(1.5, -0.5, 0.5);
    try std.testing.expectApproxEqAbs(metrics.jaccard_dissimilarity, 1.0, 0.001);
    try std.testing.expectApproxEqAbs(metrics.content_hash_distance, 0.0, 0.001);
    try std.testing.expectApproxEqAbs(metrics.temporal_novelty, 0.5, 0.001);
}

test "surprise_record_retention" {
    var record = SurpriseRecord.init([_]u8{0} ** HASH_SIZE, 0.8);
    const initial_priority = record.getRetentionPriority();
    record.recordAccess();
    try std.testing.expect(record.getAccessFrequency() == 2);
    try std.testing.expect(record.getRetentionPriority() >= initial_priority * 0.5);
}

test "statistics_incremental_update" {
    var stats = SurpriseMemoryStatistics.init(0.5);
    stats.addBlock(0.8, 0.5);
    try std.testing.expect(stats.total_blocks == 1);
    try std.testing.expect(stats.high_surprise_blocks == 1);
    stats.addBlock(0.3, 0.5);
    try std.testing.expect(stats.total_blocks == 2);
    try std.testing.expect(stats.low_surprise_blocks == 1);
    stats.removeBlock(0.8, 0.5);
    try std.testing.expect(stats.total_blocks == 1);
    try std.testing.expect(stats.high_surprise_blocks == 0);
}

test "hash_distance_calculation" {
    const hash1 = [_]u8{0xFF} ** HASH_SIZE;
    const hash2 = [_]u8{0x00} ** HASH_SIZE;
    const distance = SurpriseMemoryManager.computeHashDistance(hash1, hash2);
    try std.testing.expectApproxEqAbs(distance, 1.0, 0.001);
    const hash3 = [_]u8{0xFF} ** HASH_SIZE;
    const same_distance = SurpriseMemoryManager.computeHashDistance(hash1, hash3);
    try std.testing.expectApproxEqAbs(same_distance, 0.0, 0.001);
}

test "surprise_metrics_threshold" {
    const low_metrics = SurpriseMetrics.init(0.1, 0.1, 0.1);
    try std.testing.expect(!low_metrics.exceedsThreshold(0.3));
    const high_metrics = SurpriseMetrics.init(0.9, 0.9, 0.9);
    try std.testing.expect(high_metrics.exceedsThreshold(0.3));
}

test "partial_sort_correctness" {
    var items = [_]CandidateItem{
        .{ .block_id = [_]u8{5} ** HASH_SIZE, .priority = 5.0 },
        .{ .block_id = [_]u8{1} ** HASH_SIZE, .priority = 1.0 },
        .{ .block_id = [_]u8{3} ** HASH_SIZE, .priority = 3.0 },
        .{ .block_id = [_]u8{2} ** HASH_SIZE, .priority = 2.0 },
        .{ .block_id = [_]u8{4} ** HASH_SIZE, .priority = 4.0 },
    };
    SurpriseMemoryManager.partialSort(&items, 3);
    try std.testing.expectApproxEqAbs(items[0].priority, 1.0, 0.001);
    try std.testing.expectApproxEqAbs(items[1].priority, 2.0, 0.001);
    try std.testing.expectApproxEqAbs(items[2].priority, 3.0, 0.001);
}

test "statistics_edge_cases" {
    var stats = SurpriseMemoryStatistics.init(0.5);
    stats.removeBlock(0.5, 0.5);
    try std.testing.expect(stats.total_blocks == 0);
    try std.testing.expect(stats.average_surprise == 0.0);
    stats.addBlock(0.0, 0.5);
    try std.testing.expect(stats.total_blocks == 1);
    try std.testing.expect(stats.low_surprise_blocks == 1);
}

test "content_hash_consistency" {
    const data = "test_data_for_hashing";
    const hash1 = SurpriseMemoryManager.computeContentHash(data);
    const hash2 = SurpriseMemoryManager.computeContentHash(data);
    try std.testing.expect(std.mem.eql(u8, &hash1, &hash2));
    const different_data = "different_test_data";
    const hash3 = SurpriseMemoryManager.computeContentHash(different_data);
    try std.testing.expect(!std.mem.eql(u8, &hash1, &hash3));
}
