const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;
const Random = std.crypto.random;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;

pub const FNDSStatistics = struct {
    total_trees: usize,
    total_indices: usize,
    cache_hits: usize,
    cache_misses: usize,
    average_tree_depth: f64,
    memory_used: usize,
    total_nodes_across_trees: usize,
    total_patterns_indexed: usize,
    cache_hit_ratio: f64,
    last_operation_time_ns: i64,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .total_trees = 0,
            .total_indices = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .average_tree_depth = 0.0,
            .memory_used = 0,
            .total_nodes_across_trees = 0,
            .total_patterns_indexed = 0,
            .cache_hit_ratio = 0.0,
            .last_operation_time_ns = 0,
        };
    }

    pub fn updateCacheHitRatio(self: *Self) void {
        const total = self.cache_hits + self.cache_misses;
        if (total > 0) {
            self.cache_hit_ratio = @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total));
        } else {
            self.cache_hit_ratio = 0.0;
        }
    }

    pub fn recordCacheHit(self: *Self) void {
        self.cache_hits += 1;
    }

    pub fn recordCacheMiss(self: *Self) void {
        self.cache_misses += 1;
    }

    pub fn updateAverageTreeDepth(self: *Self, depths: []const usize) void {
        if (depths.len == 0) {
            self.average_tree_depth = 0.0;
            return;
        }
        var sum: usize = 0;
        for (depths) |d| {
            sum += d;
        }
        self.average_tree_depth = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(depths.len));
    }
};

pub const FractalNodeData = struct {
    id: []const u8,
    data: []const u8,
    weight: f64,
    scale: f64,
    fractal_signature: [32]u8,
    children_count: usize,
    metadata: StringHashMap([]const u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: []const u8, data: []const u8, weight: f64, scale: f64) !Self {
        const id_copy = try allocator.dupe(u8, id);
        errdefer allocator.free(id_copy);
        const data_copy = try allocator.dupe(u8, data);
        errdefer allocator.free(data_copy);

        var signature: [32]u8 = undefined;
        var hasher = Sha256.init(.{});
        hasher.update(id);
        hasher.update(data);
        hasher.update(std.mem.asBytes(&weight));
        hasher.update(std.mem.asBytes(&scale));
        const hash_result = hasher.finalResult();
        @memcpy(&signature, &hash_result);

        return Self{
            .id = id_copy,
            .data = data_copy,
            .weight = weight,
            .scale = scale,
            .fractal_signature = signature,
            .children_count = 0,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.id);
        self.allocator.free(self.data);
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_node = Self{
            .id = try allocator.dupe(u8, self.id),
            .data = try allocator.dupe(u8, self.data),
            .weight = self.weight,
            .scale = self.scale,
            .fractal_signature = self.fractal_signature,
            .children_count = self.children_count,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        errdefer new_node.deinit();

        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(key_copy);
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            errdefer allocator.free(val_copy);
            try new_node.metadata.put(key_copy, val_copy);
        }
        return new_node;
    }

    pub fn setMetadata(self: *Self, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.metadata.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }
        try self.metadata.put(key_copy, val_copy);
    }

    pub fn getMetadata(self: *const Self, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn computeHash(self: *const Self) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.id);
        hasher.update(self.data);
        hasher.update(std.mem.asBytes(&self.weight));
        hasher.update(std.mem.asBytes(&self.scale));
        hasher.update(&self.fractal_signature);

        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            hasher.update(entry.key_ptr.*);
            hasher.update(entry.value_ptr.*);
        }
        return hasher.final();
    }
};

pub const FractalEdgeData = struct {
    source_id: []const u8,
    target_id: []const u8,
    weight: f64,
    scale_ratio: f64,
    edge_type: EdgeType,
    fractal_correlation: f64,
    allocator: Allocator,

    pub const EdgeType = enum(u8) {
        hierarchical = 0,
        sibling = 1,
        cross_level = 2,
        self_similar = 3,

        pub fn toString(self: EdgeType) []const u8 {
            return switch (self) {
                .hierarchical => "hierarchical",
                .sibling => "sibling",
                .cross_level => "cross_level",
                .self_similar => "self_similar",
            };
        }
    };

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        source_id: []const u8,
        target_id: []const u8,
        weight: f64,
        scale_ratio: f64,
        edge_type: EdgeType,
    ) !Self {
        return Self{
            .source_id = try allocator.dupe(u8, source_id),
            .target_id = try allocator.dupe(u8, target_id),
            .weight = weight,
            .scale_ratio = scale_ratio,
            .edge_type = edge_type,
            .fractal_correlation = 1.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.source_id);
        self.allocator.free(self.target_id);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        return Self{
            .source_id = try allocator.dupe(u8, self.source_id),
            .target_id = try allocator.dupe(u8, self.target_id),
            .weight = self.weight,
            .scale_ratio = self.scale_ratio,
            .edge_type = self.edge_type,
            .fractal_correlation = self.fractal_correlation,
            .allocator = allocator,
        };
    }
};

pub const FractalLevel = struct {
    level: usize,
    scale_factor: f64,
    nodes: StringHashMap(FractalNodeData),
    edges: StringHashMap(ArrayList(FractalEdgeData)),
    parent_level: ?*FractalLevel,
    child_levels: ArrayList(*FractalLevel),
    node_count: usize,
    edge_count: usize,
    fractal_dimension: f64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, level: usize, scale_factor: f64) Self {
        return Self{
            .level = level,
            .scale_factor = scale_factor,
            .nodes = StringHashMap(FractalNodeData).init(allocator),
            .edges = StringHashMap(ArrayList(FractalEdgeData)).init(allocator),
            .parent_level = null,
            .child_levels = ArrayList(*FractalLevel).init(allocator),
            .node_count = 0,
            .edge_count = 0,
            .fractal_dimension = 1.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.nodes.deinit();

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            var edge_list = entry.value_ptr;
            for (edge_list.items) |edge| {
                edge.deinit();
            }
            edge_list.deinit();
        }
        self.edges.deinit();

        for (self.child_levels.items) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.child_levels.deinit();
    }

    pub fn getNode(self: *Self, node_id: []const u8) ?*FractalNodeData {
        return self.nodes.getPtr(node_id);
    }

    pub fn getNodeConst(self: *const Self, node_id: []const u8) ?FractalNodeData {
        return self.nodes.get(node_id);
    }

    pub fn addNode(self: *Self, node: FractalNodeData) !void {
        const gop = try self.nodes.getOrPut(node.id);
        if (gop.found_existing) {
            gop.value_ptr.deinit();
        } else {
            self.node_count += 1;
        }
        gop.value_ptr.* = node;
    }

    pub fn removeNode(self: *Self, node_id: []const u8) bool {
        if (self.nodes.fetchRemove(node_id)) |removed| {
            removed.value.deinit();
            self.allocator.free(removed.key);
            self.node_count -= 1;

            if (self.edges.fetchRemove(node_id)) |edge_removed| {
                var edge_list = edge_removed.value;
                for (edge_list.items) |edge| {
                    edge.deinit();
                    self.edge_count -= 1;
                }
                edge_list.deinit();
                self.allocator.free(edge_removed.key);
            }
            return true;
        }
        return false;
    }

    pub fn getEdges(self: *Self, source_id: []const u8) ?*ArrayList(FractalEdgeData) {
        return self.edges.getPtr(source_id);
    }

    pub fn getEdgesConst(self: *const Self, source_id: []const u8) []const FractalEdgeData {
        if (self.edges.get(source_id)) |list| {
            return list.items;
        }
        return &[_]FractalEdgeData{};
    }

    pub fn addEdge(self: *Self, edge: FractalEdgeData) !void {
        const gop = try self.edges.getOrPut(edge.source_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = ArrayList(FractalEdgeData).init(self.allocator);
        }
        try gop.value_ptr.append(edge);
        self.edge_count += 1;
    }

    pub fn removeEdge(self: *Self, source_id: []const u8, target_id: []const u8) bool {
        if (self.edges.getPtr(source_id)) |edge_list| {
            for (edge_list.items, 0..) |edge, i| {
                if (std.mem.eql(u8, edge.target_id, target_id)) {
                    var removed = edge_list.orderedRemove(i);
                    removed.deinit();
                    self.edge_count -= 1;
                    return true;
                }
            }
        }
        return false;
    }

    pub fn addChildLevel(self: *Self, child: *FractalLevel) !void {
        child.parent_level = self;
        try self.child_levels.append(child);
    }

    pub fn getChildLevel(self: *Self, index: usize) ?*FractalLevel {
        if (index < self.child_levels.items.len) {
            return self.child_levels.items[index];
        }
        return null;
    }

    pub fn computeLocalFractalDimension(self: *Self) f64 {
        if (self.node_count < 2) {
            return 0.0;
        }

        const box_sizes = [_]usize{ 1, 2, 4, 8 };
        var log_n_sum: f64 = 0.0;
        var log_r_sum: f64 = 0.0;
        var log_nr_sum: f64 = 0.0;
        var log_r2_sum: f64 = 0.0;
        var count: usize = 0;

        inline for (box_sizes) |size| {
            const box_count = self.estimateBoxCount(size);
            if (box_count > 0) {
                const log_n = @log(@as(f64, @floatFromInt(box_count)));
                const log_r = @log(1.0 / @as(f64, @floatFromInt(size)));
                log_n_sum += log_n;
                log_r_sum += log_r;
                log_nr_sum += log_n * log_r;
                log_r2_sum += log_r * log_r;
                count += 1;
            }
        }

        if (count < 2) {
            self.fractal_dimension = 1.0;
            return 1.0;
        }

        const n = @as(f64, @floatFromInt(count));
        const denominator = n * log_r2_sum - log_r_sum * log_r_sum;
        if (@fabs(denominator) < 1e-10) {
            self.fractal_dimension = 1.0;
            return 1.0;
        }

        const slope = (n * log_nr_sum - log_n_sum * log_r_sum) / denominator;
        self.fractal_dimension = @fabs(slope);
        return self.fractal_dimension;
    }

    fn estimateBoxCount(self: *Self, box_size: usize) usize {
        if (box_size == 0) return 0;
        if (self.node_count == 0) return 0;

        const box_size_f = @as(f64, @floatFromInt(box_size));
        const estimated = @as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(self.node_count)) / box_size_f)));
        return @max(1, estimated);
    }

    pub fn getDepth(self: *const Self) usize {
        if (self.child_levels.items.len == 0) return 1;

        var max_depth: usize = 0;
        var stack = std.ArrayList(*const FractalLevel).init(self.allocator);
        defer stack.deinit();

        try stack.append(self);

        while (stack.popOrNull()) |curr| {
            if (curr.level > max_depth) max_depth = curr.level;
            for (curr.child_levels.items) |child| {
                try stack.append(child);
            }
        }
        return max_depth;
    }

    pub fn getTotalNodeCount(self: *const Self) usize {
        var total = self.node_count;
        var stack = std.ArrayList(*const FractalLevel).init(self.allocator);
        defer stack.deinit();

        for (self.child_levels.items) |child| {
            try stack.append(child);
        }

        while (stack.popOrNull()) |curr| {
            total += curr.node_count;
            for (curr.child_levels.items) |child| {
                try stack.append(child);
            }
        }
        return total;
    }
};

pub const TraversalOrder = enum {
    pre_order,
    post_order,
    level_order,
    fractal_order,
};

pub const TraversalCallback = *const fn (*FractalLevel, usize) void;

pub const FractalTree = struct {
    root: *FractalLevel,
    max_depth: usize,
    branching_factor: usize,
    total_nodes: usize,
    tree_id: [32]u8,
    creation_time: i64,
    last_modified: i64,
    allocator: Allocator,
    is_balanced: bool,
    depth_cache: ?usize,

    const Self = @This();

    pub fn init(allocator: Allocator, max_depth: usize, branching_factor: usize) !Self {
        const root = try allocator.create(FractalLevel);
        root.* = FractalLevel.init(allocator, 0, 1.0);

        var tree_id: [32]u8 = undefined;
        Random.bytes(&tree_id);

        return Self{
            .root = root,
            .max_depth = max_depth,
            .branching_factor = branching_factor,
            .total_nodes = 0,
            .tree_id = tree_id,
            .creation_time = std.time.nanoTimestamp(),
            .last_modified = std.time.nanoTimestamp(),
            .allocator = allocator,
            .is_balanced = true,
            .depth_cache = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.root.deinit();
        self.allocator.destroy(self.root);
    }

    pub fn insert(self: *Self, node_id: []const u8, data: []const u8, target_level: usize) !bool {
        if (target_level > self.max_depth) {
            return false;
        }

        var current_level = self.root;
        var depth: usize = 0;

        while (depth < target_level) : (depth += 1) {
            if (current_level.child_levels.items.len == 0) {
                const child_scale = current_level.scale_factor / @as(f64, @floatFromInt(self.branching_factor));
                const new_child = try self.allocator.create(FractalLevel);
                new_child.* = FractalLevel.init(self.allocator, depth + 1, child_scale);
                try current_level.addChildLevel(new_child);
            }

            const child_index = self.computeChildIndex(node_id, current_level.child_levels.items.len);
            if (child_index >= current_level.child_levels.items.len) {
                const child_scale = current_level.scale_factor / @as(f64, @floatFromInt(self.branching_factor));
                const new_child = try self.allocator.create(FractalLevel);
                new_child.* = FractalLevel.init(self.allocator, depth + 1, child_scale);
                try current_level.addChildLevel(new_child);
            }
            current_level = current_level.child_levels.items[child_index];
        }

        var node = try FractalNodeData.init(self.allocator, node_id, data, 1.0, current_level.scale_factor);
        errdefer node.deinit();
        try current_level.addNode(node);

        self.total_nodes += 1;
        self.last_modified = std.time.nanoTimestamp();
        self.depth_cache = null;

        if (self.total_nodes % 100 == 0) {
            self.checkBalance();
        }

        return true;
    }

    fn computeChildIndex(self: *const Self, node_id: []const u8, max_children: usize) usize {
        _ = self;
        if (max_children == 0) return 0;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(node_id);
        const hash = hasher.final();
        return @as(usize, @intCast(hash % @as(u64, @intCast(max_children))));
    }

    pub fn search(self: *Self, node_id: []const u8) ?*FractalNodeData {
        return self.searchLevel(self.root, node_id);
    }

    fn searchLevel(self: *Self, level: *FractalLevel, node_id: []const u8) ?*FractalNodeData {
        if (level.getNode(node_id)) |node| {
            return node;
        }

        var stack = std.ArrayList(*FractalLevel).init(self.allocator);
        defer stack.deinit();

        for (level.child_levels.items) |child| {
            try stack.append(child);
        }

        while (stack.popOrNull()) |curr| {
            if (curr.getNode(node_id)) |node| return node;
            for (curr.child_levels.items) |child| {
                try stack.append(child);
            }
        }

        return null;
    }

    pub fn searchConst(self: *const Self, node_id: []const u8) ?FractalNodeData {
        return self.searchLevelConst(self.root, node_id);
    }

    fn searchLevelConst(self: *const Self, level: *FractalLevel, node_id: []const u8) ?FractalNodeData {
        if (level.getNodeConst(node_id)) |node| {
            return node;
        }

        var stack = std.ArrayList(*const FractalLevel).init(self.allocator);
        defer stack.deinit();

        for (level.child_levels.items) |child| {
            try stack.append(child);
        }

        while (stack.popOrNull()) |curr| {
            if (curr.getNodeConst(node_id)) |node| return node;
            for (curr.child_levels.items) |child| {
                try stack.append(child);
            }
        }

        return null;
    }

    pub fn delete(self: *Self, node_id: []const u8) bool {
        if (self.deleteFromLevel(self.root, node_id)) {
            self.total_nodes -= 1;
            self.last_modified = std.time.nanoTimestamp();
            self.depth_cache = null;
            self.checkBalance();
            return true;
        }
        return false;
    }

    fn deleteFromLevel(self: *Self, level: *FractalLevel, node_id: []const u8) bool {
        if (level.removeNode(node_id)) {
            return true;
        }

        for (level.child_levels.items) |child| {
            if (self.deleteFromLevel(child, node_id)) {
                return true;
            }
        }

        return false;
    }

    pub fn traverse(self: *Self, order: TraversalOrder, callback: TraversalCallback) void {
        switch (order) {
            .pre_order => self.traversePreOrder(self.root, 0, callback),
            .post_order => self.traversePostOrder(self.root, 0, callback),
            .level_order => self.traverseLevelOrder(callback),
            .fractal_order => self.traverseFractalOrder(self.root, 0, callback),
        }
    }

    fn traversePreOrder(self: *Self, level: *FractalLevel, depth: usize, callback: TraversalCallback) void {
        callback(level, depth);
        for (level.child_levels.items) |child| {
            self.traversePreOrder(child, depth + 1, callback);
        }
    }

    fn traversePostOrder(self: *Self, level: *FractalLevel, depth: usize, callback: TraversalCallback) void {
        for (level.child_levels.items) |child| {
            self.traversePostOrder(child, depth + 1, callback);
        }
        callback(level, depth);
    }

    fn traverseLevelOrder(self: *Self, callback: TraversalCallback) void {
        var fifo = std.fifo.LinearFifo(struct { level: *FractalLevel, depth: usize }, .Dynamic).init(self.allocator);
        defer fifo.deinit();

        fifo.writeItem(.{ .level = self.root, .depth = 0 }) catch return;

        while (fifo.readItem()) |item| {
            callback(item.level, item.depth);

            for (item.level.child_levels.items) |child| {
                fifo.writeItem(.{ .level = child, .depth = item.depth + 1 }) catch continue;
            }
        }
    }

    fn traverseFractalOrder(self: *Self, level: *FractalLevel, depth: usize, callback: TraversalCallback) void {
        callback(level, depth);

        if (level.child_levels.items.len == 0) return;

        const mid = level.child_levels.items.len / 2;
        var i: usize = 0;
        while (i < mid) : (i += 1) {
            self.traverseFractalOrder(level.child_levels.items[i], depth + 1, callback);
        }

        i = level.child_levels.items.len;
        while (i > mid) {
            i -= 1;
            self.traverseFractalOrder(level.child_levels.items[i], depth + 1, callback);
        }
    }

    pub fn getDepth(self: *Self) usize {
        if (self.depth_cache) |cached| {
            return cached;
        }
        const depth = self.root.getDepth();
        self.depth_cache = depth;
        return depth;
    }

    pub fn balance(self: *Self) !void {
        if (self.is_balanced) return;

        var all_nodes = ArrayList(FractalNodeData).init(self.allocator);
        defer {
            for (all_nodes.items) |n| n.deinit();
            all_nodes.deinit();
        }

        try self.collectAllNodes(self.root, &all_nodes);

        self.root.deinit();
        self.root.* = FractalLevel.init(self.allocator, 0, 1.0);
        self.total_nodes = 0;

        for (all_nodes.items) |node| {
            const target_level = self.computeOptimalLevel(node.id);
            _ = try self.insert(node.id, node.data, target_level);
        }

        self.is_balanced = true;
        self.depth_cache = null;
    }

    fn collectAllNodes(self: *Self, level: *FractalLevel, nodes: *ArrayList(FractalNodeData)) !void {
        var node_iter = level.nodes.iterator();
        while (node_iter.next()) |entry| {
            const cloned = try entry.value_ptr.clone(self.allocator);
            try nodes.append(cloned);
        }

        var stack = std.ArrayList(*FractalLevel).init(self.allocator);
        defer stack.deinit();

        for (level.child_levels.items) |child| {
            try stack.append(child);
        }

        while (stack.popOrNull()) |curr| {
            var iter = curr.nodes.iterator();
            while (iter.next()) |entry| {
                const cloned = try entry.value_ptr.clone(self.allocator);
                try nodes.append(cloned);
            }
            for (curr.child_levels.items) |child| {
                try stack.append(child);
            }
        }
    }

    fn computeOptimalLevel(self: *Self, node_id: []const u8) usize {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(node_id);
        const hash = hasher.final();
        const level = @as(usize, @intCast(hash % @as(u64, @intCast(self.max_depth + 1))));
        return @min(level, self.max_depth);
    }

    fn checkBalance(self: *Self) void {
        const depth = self.getDepth();
        const optimal_depth = self.computeOptimalDepth();
        self.is_balanced = (depth <= optimal_depth + 2);
    }

    fn computeOptimalDepth(self: *Self) usize {
        if (self.total_nodes == 0) return 0;
        const log_base = @log(@as(f64, @floatFromInt(@max(2, self.branching_factor))));
        const log_nodes = @log(@as(f64, @floatFromInt(self.total_nodes)));
        return @as(usize, @intFromFloat(@ceil(log_nodes / log_base)));
    }

    pub fn getTreeIdHex(self: *const Self) [64]u8 {
        var hex_buf: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&hex_buf, "{s}", .{std.fmt.fmtSliceHexLower(&self.tree_id)}) catch unreachable;
        return hex_buf;
    }

    pub fn computeFractalDimension(self: *Self) f64 {
        return self.computeLevelDimension(self.root);
    }

    fn computeLevelDimension(self: *Self, level: *FractalLevel) f64 {
        var local_dim = level.computeLocalFractalDimension();

        if (level.child_levels.items.len == 0) {
            return local_dim;
        }

        var child_dim_sum: f64 = 0.0;
        for (level.child_levels.items) |child| {
            child_dim_sum += self.computeLevelDimension(child);
        }
        const child_avg = child_dim_sum / @as(f64, @floatFromInt(level.child_levels.items.len));

        return (local_dim + child_avg) / 2.0;
    }
};

pub const PatternLocation = struct {
    tree_id: [32]u8,
    level: usize,
    node_id: []const u8,
    offset: usize,
    length: usize,
    confidence: f64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        tree_id: [32]u8,
        level: usize,
        node_id: []const u8,
        offset: usize,
        length: usize,
        confidence: f64,
    ) !Self {
        return Self{
            .tree_id = tree_id,
            .level = level,
            .node_id = try allocator.dupe(u8, node_id),
            .offset = offset,
            .length = length,
            .confidence = confidence,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.node_id);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        return Self{
            .tree_id = self.tree_id,
            .level = self.level,
            .node_id = try allocator.dupe(u8, self.node_id),
            .offset = self.offset,
            .length = self.length,
            .confidence = self.confidence,
            .allocator = allocator,
        };
    }
};

pub const SelfSimilarIndex = struct {
    patterns: StringHashMap(ArrayList(PatternLocation)),
    dimension_estimate: f64,
    pattern_count: usize,
    total_locations: usize,
    min_pattern_length: usize,
    max_pattern_length: usize,
    similarity_threshold: f64,
    allocator: Allocator,
    pattern_keys: ArrayList([]const u8),
    creation_time: i64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .patterns = StringHashMap(ArrayList(PatternLocation)).init(allocator),
            .dimension_estimate = 0.0,
            .pattern_count = 0,
            .total_locations = 0,
            .min_pattern_length = 1,
            .max_pattern_length = 256,
            .similarity_threshold = 0.8,
            .allocator = allocator,
            .pattern_keys = ArrayList([]const u8).init(allocator),
            .creation_time = std.time.nanoTimestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            var locations = entry.value_ptr;
            for (locations.items) |loc| loc.deinit();
            locations.deinit();
        }
        self.patterns.deinit();

        for (self.pattern_keys.items) |k| self.allocator.free(k);
        self.pattern_keys.deinit();
    }

    pub fn addPattern(self: *Self, pattern: []const u8, location: PatternLocation) !void {
        if (pattern.len < self.min_pattern_length or pattern.len > self.max_pattern_length) {
            return error.PatternLengthOutOfRange;
        }

        const pattern_copy = try self.allocator.dupe(u8, pattern);
        errdefer self.allocator.free(pattern_copy);

        const gop = try self.patterns.getOrPut(pattern_copy);
        if (!gop.found_existing) {
            gop.value_ptr.* = ArrayList(PatternLocation).init(self.allocator);
            try self.pattern_keys.append(pattern_copy);
            self.pattern_count += 1;
        } else {
            self.allocator.free(pattern_copy);
        }

        try gop.value_ptr.append(location);
        self.total_locations += 1;
    }

    pub fn findPattern(self: *Self, pattern: []const u8) []PatternLocation {
        if (self.patterns.getPtr(pattern)) |locations| {
            return locations.items;
        }
        return &[_]PatternLocation{};
    }

    pub fn findSimilarPatterns(self: *Self, pattern: []const u8, results: *ArrayList(struct { pattern: []const u8, similarity: f64 })) !void {
        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            const similarity = self.computeSimilarity(pattern, entry.key_ptr.*);
            if (similarity >= self.similarity_threshold) {
                try results.append(.{ .pattern = entry.key_ptr.*, .similarity = similarity });
            }
        }
    }

    fn computeSimilarity(self: *const Self, a: []const u8, b: []const u8) f64 {
        _ = self;
        if (a.len == 0 or b.len == 0) return 0.0;
        if (std.mem.eql(u8, a, b)) return 1.0;

        const max_len = @max(a.len, b.len);
        const min_len = @min(a.len, b.len);

        var matches: usize = 0;
        var i: usize = 0;
        while (i < min_len) : (i += 1) {
            if (a[i] == b[i]) {
                matches += 1;
            }
        }

        const length_ratio = @as(f64, @floatFromInt(min_len)) / @as(f64, @floatFromInt(max_len));
        const match_ratio = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(max_len));

        return (length_ratio + match_ratio) / 2.0;
    }

    pub fn computeFractalDimension(self: *Self) f64 {
        if (self.pattern_count < 2) {
            self.dimension_estimate = 0.0;
            return 0.0;
        }

        var length_counts = AutoHashMap(usize, usize).init(self.allocator);
        defer length_counts.deinit();

        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            const len = entry.key_ptr.len;
            const current = length_counts.get(len) orelse 0;
            length_counts.put(len, current + 1) catch continue;
        }

        var log_l_sum: f64 = 0.0;
        var log_n_sum: f64 = 0.0;
        var log_ln_sum: f64 = 0.0;
        var log_l2_sum: f64 = 0.0;
        var count: usize = 0;

        var lc_iter = length_counts.iterator();
        while (lc_iter.next()) |entry| {
            const len = entry.key_ptr.*;
            const cnt = entry.value_ptr.*;
            if (len > 0 and cnt > 0) {
                const log_l = @log(@as(f64, @floatFromInt(len)));
                const log_n = @log(@as(f64, @floatFromInt(cnt)));
                log_l_sum += log_l;
                log_n_sum += log_n;
                log_ln_sum += log_l * log_n;
                log_l2_sum += log_l * log_l;
                count += 1;
            }
        }

        if (count < 2) {
            self.dimension_estimate = 1.0;
            return 1.0;
        }

        const n = @as(f64, @floatFromInt(count));
        const denominator = n * log_l2_sum - log_l_sum * log_l_sum;
        if (@fabs(denominator) < 1e-10) {
            self.dimension_estimate = 1.0;
            return 1.0;
        }

        const slope = (n * log_ln_sum - log_l_sum * log_n_sum) / denominator;
        self.dimension_estimate = @fabs(slope);
        return self.dimension_estimate;
    }

    pub fn removePattern(self: *Self, pattern: []const u8) bool {
        if (self.patterns.fetchRemove(pattern)) |removed| {
            var locations = removed.value;
            for (locations.items) |loc| {
                loc.deinit();
                self.total_locations -= 1;
            }
            locations.deinit();
            self.allocator.free(removed.key);
            self.pattern_count -= 1;
            return true;
        }
        return false;
    }

    pub fn getPatternCount(self: *const Self) usize {
        return self.pattern_count;
    }

    pub fn getTotalLocations(self: *const Self) usize {
        return self.total_locations;
    }
};

pub fn CoalescedEntry(comptime K: type, comptime V: type) type {
    return struct {
        key: K,
        value: V,
        next_index: ?usize,
        is_primary: bool,
    };
}

pub fn CoalescedHashMap(comptime K: type, comptime V: type) type {
    return struct {
        buckets: []?Entry,
        capacity: usize,
        size: usize,
        collision_chain: ArrayList(usize),
        load_factor: f64,
        max_load_factor: f64,
        cellar_start: usize,
        cellar_next: usize,
        allocator: Allocator,

        const Entry = CoalescedEntry(K, V);
        const Self = @This();
        const DEFAULT_CAPACITY: usize = 16;
        const DEFAULT_MAX_LOAD_FACTOR: f64 = 0.86;
        const CELLAR_RATIO: f64 = 0.14;

        pub fn init(allocator: Allocator) !Self {
            return initWithCapacity(allocator, DEFAULT_CAPACITY);
        }

        pub fn initWithCapacity(allocator: Allocator, initial_capacity: usize) !Self {
            const capacity = @max(initial_capacity, 4);
            const buckets = try allocator.alloc(?Entry, capacity);
            @memset(buckets, null);

            const cellar_size = @as(usize, @intFromFloat(@as(f64, @floatFromInt(capacity)) * CELLAR_RATIO));
            const cellar_start = capacity - @max(cellar_size, 1);

            return Self{
                .buckets = buckets,
                .capacity = capacity,
                .size = 0,
                .collision_chain = ArrayList(usize).init(allocator),
                .load_factor = 0.0,
                .max_load_factor = DEFAULT_MAX_LOAD_FACTOR,
                .cellar_start = cellar_start,
                .cellar_next = cellar_start,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buckets);
            self.collision_chain.deinit();
        }

        fn hash(self: *const Self, key: K) usize {
            _ = self;
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            return @as(usize, @intCast(hasher.final()));
        }

        fn eql(self: *const Self, a: K, b: K) bool {
            _ = self;
            return std.meta.eql(a, b);
        }

        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.load_factor >= self.max_load_factor) {
                try self.resize(self.capacity * 2);
            }

            const index = self.hash(key) % self.cellar_start;

            if (self.buckets[index]) |*existing| {
                if (self.eql(existing.key, key)) {
                    existing.value = value;
                    return;
                }

                var current_idx = index;
                while (self.buckets[current_idx]) |entry| {
                    if (self.eql(entry.key, key)) {
                        self.buckets[current_idx].?.value = value;
                        return;
                    }
                    if (entry.next_index) |next| {
                        current_idx = next;
                    } else {
                        break;
                    }
                }

                const new_idx = self.findEmptySlot();
                if (new_idx) |slot| {
                    self.buckets[slot] = Entry{
                        .key = key,
                        .value = value,
                        .next_index = null,
                        .is_primary = false,
                    };
                    self.buckets[current_idx].?.next_index = slot;
                    try self.collision_chain.append(slot);
                    self.size += 1;
                    self.updateLoadFactor();
                } else {
                    try self.resize(self.capacity * 2);
                    try self.put(key, value);
                }
            } else {
                self.buckets[index] = Entry{
                    .key = key,
                    .value = value,
                    .next_index = null,
                    .is_primary = true,
                };
                self.size += 1;
                self.updateLoadFactor();
            }
        }

        fn findEmptySlot(self: *Self) ?usize {
            if (self.cellar_next < self.capacity) {
                while (self.cellar_next < self.capacity) {
                    if (self.buckets[self.cellar_next] == null) {
                        const slot = self.cellar_next;
                        self.cellar_next += 1;
                        return slot;
                    }
                    self.cellar_next += 1;
                }
            }

            var i: usize = 0;
            while (i < self.cellar_start) : (i += 1) {
                if (self.buckets[i] == null) {
                    return i;
                }
            }

            return null;
        }

        pub fn get(self: *Self, key: K) ?V {
            const index = self.hash(key) % self.cellar_start;

            var current_idx: ?usize = index;
            while (current_idx) |idx| {
                if (self.buckets[idx]) |entry| {
                    if (self.eql(entry.key, key)) {
                        return entry.value;
                    }
                    current_idx = entry.next_index;
                } else {
                    return null;
                }
            }

            return null;
        }

        pub fn getPtr(self: *Self, key: K) ?*V {
            const index = self.hash(key) % self.cellar_start;

            var current_idx: ?usize = index;
            while (current_idx) |idx| {
                if (self.buckets[idx]) |*entry| {
                    if (self.eql(entry.key, key)) {
                        return &entry.value;
                    }
                    current_idx = entry.next_index;
                } else {
                    return null;
                }
            }

            return null;
        }

        pub fn contains(self: *Self, key: K) bool {
            return self.get(key) != null;
        }

        pub fn remove(self: *Self, key: K) bool {
            const index = self.hash(key) % self.cellar_start;

            if (self.buckets[index] == null) {
                return false;
            }

            if (self.eql(self.buckets[index].?.key, key)) {
                if (self.buckets[index].?.next_index) |next| {
                    self.buckets[index] = self.buckets[next];
                    self.buckets[next] = null;
                } else {
                    self.buckets[index] = null;
                }
                self.size -= 1;
                self.updateLoadFactor();
                return true;
            }

            var prev_idx = index;
            var current_idx = self.buckets[index].?.next_index;

            while (current_idx) |idx| {
                if (self.buckets[idx]) |entry| {
                    if (self.eql(entry.key, key)) {
                        self.buckets[prev_idx].?.next_index = entry.next_index;
                        self.buckets[idx] = null;
                        self.size -= 1;
                        self.updateLoadFactor();
                        return true;
                    }
                    prev_idx = idx;
                    current_idx = entry.next_index;
                } else {
                    break;
                }
            }

            return false;
        }

        pub fn resize(self: *Self, new_capacity: usize) !void {
            const old_buckets = self.buckets;
            const old_capacity = self.capacity;

            const capacity = @max(new_capacity, 4);
            const new_buckets = try self.allocator.alloc(?Entry, capacity);
            @memset(new_buckets, null);

            const cellar_size = @as(usize, @intFromFloat(@as(f64, @floatFromInt(capacity)) * CELLAR_RATIO));
            self.cellar_start = capacity - @max(cellar_size, 1);
            self.cellar_next = self.cellar_start;

            self.buckets = new_buckets;
            self.capacity = capacity;
            self.size = 0;
            self.collision_chain.clearRetainingCapacity();
            self.load_factor = 0.0;

            var i: usize = 0;
            while (i < old_capacity) : (i += 1) {
                if (old_buckets[i]) |entry| {
                    try self.putInternal(entry.key, entry.value);
                }
            }

            self.allocator.free(old_buckets);
        }

        fn putInternal(self: *Self, key: K, value: V) !void {
            const index = self.hash(key) % self.cellar_start;

            if (self.buckets[index]) |*existing| {
                if (self.eql(existing.key, key)) {
                    existing.value = value;
                    return;
                }

                var current_idx = index;
                while (self.buckets[current_idx]) |ent| {
                    if (self.eql(ent.key, key)) {
                        self.buckets[current_idx].?.value = value;
                        return;
                    }
                    if (ent.next_index) |next| {
                        current_idx = next;
                    } else {
                        break;
                    }
                }

                const new_idx = self.findEmptySlot();
                if (new_idx) |slot| {
                    self.buckets[slot] = Entry{
                        .key = key,
                        .value = value,
                        .next_index = null,
                        .is_primary = false,
                    };
                    self.buckets[current_idx].?.next_index = slot;
                    try self.collision_chain.append(slot);
                    self.size += 1;
                    self.updateLoadFactor();
                }
            } else {
                self.buckets[index] = Entry{
                    .key = key,
                    .value = value,
                    .next_index = null,
                    .is_primary = true,
                };
                self.size += 1;
                self.updateLoadFactor();
            }
        }

        fn updateLoadFactor(self: *Self) void {
            self.load_factor = @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(self.capacity));
        }

        pub fn getLoadFactor(self: *const Self) f64 {
            return self.load_factor;
        }

        pub fn count(self: *const Self) usize {
            return self.size;
        }

        pub fn getCapacity(self: *const Self) usize {
            return self.capacity;
        }

        pub fn clear(self: *Self) void {
            @memset(self.buckets, null);
            self.size = 0;
            self.load_factor = 0.0;
            self.cellar_next = self.cellar_start;
            self.collision_chain.clearRetainingCapacity();
        }
    };
}

pub const LRUCache = struct {
    const Entry = struct {
        key: []const u8,
        value: []const u8,
        prev: ?*Entry = null,
        next: ?*Entry = null,
    };

    map: StringHashMap(*Entry),
    head: ?*Entry,
    tail: ?*Entry,
    capacity: usize,
    current_size: usize,
    max_memory: usize,
    current_memory: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, capacity: usize, max_memory: usize) Self {
        return Self{
            .map = StringHashMap(*Entry).init(allocator),
            .head = null,
            .tail = null,
            .capacity = capacity,
            .current_size = 0,
            .max_memory = max_memory,
            .current_memory = 0,
            .hits = 0,
            .misses = 0,
            .evictions = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            const e = entry.value_ptr.*;
            self.allocator.free(e.key);
            self.allocator.free(e.value);
            self.allocator.destroy(e);
        }
        self.map.deinit();
    }

    fn unlink(self: *Self, node: *Entry) void {
        if (node.prev) |prev| {
            prev.next = node.next;
        } else {
            self.head = node.next;
        }

        if (node.next) |next| {
            next.prev = node.prev;
        } else {
            self.tail = node.prev;
        }
    }

    fn insertHead(self: *Self, node: *Entry) void {
        node.prev = null;
        node.next = self.head;

        if (self.head) |head| {
            head.prev = node;
        }
        self.head = node;

        if (self.tail == null) {
            self.tail = node;
        }
    }

    pub fn get(self: *Self, key: []const u8) ?[]const u8 {
        if (self.map.get(key)) |node_ptr| {
            const node = node_ptr.*;
            self.unlink(node);
            self.insertHead(node);
            self.hits += 1;
            return node.value;
        }
        self.misses += 1;
        return null;
    }

    pub fn put(self: *Self, key: []const u8, value: []const u8) !void {
        const entry_size = key.len + value.len;

        if (self.map.get(key)) |node_ptr| {
            const node = node_ptr.*;
            self.unlink(node);
            self.current_memory -= (node.key.len + node.value.len);
            self.allocator.free(node.key);
            self.allocator.free(node.value);

            node.key = try self.allocator.dupe(u8, key);
            node.value = try self.allocator.dupe(u8, value);
            self.current_memory += entry_size;

            self.insertHead(node);
            return;
        }

        while ((self.current_size >= self.capacity) or (self.current_memory + entry_size > self.max_memory and self.current_size > 0)) {
            if (!self.evict()) break;
        }

        const node = try self.allocator.create(Entry);
        node.* = Entry{
            .key = try self.allocator.dupe(u8, key),
            .value = try self.allocator.dupe(u8, value),
            .prev = null,
            .next = null,
        };
        self.current_memory += entry_size;

        try self.map.put(node.key, node);
        self.insertHead(node);
        self.current_size += 1;
    }

    fn evict(self: *Self) bool {
        if (self.tail == null) return false;

        const lru = self.tail.?;
        self.unlink(lru);

        _ = self.map.remove(lru.key);
        self.allocator.free(lru.key);
        self.allocator.free(lru.value);
        self.allocator.destroy(lru);

        self.current_size -= 1;
        self.current_memory -= (lru.key.len + lru.value.len);
        self.evictions += 1;
        return true;
    }

    pub fn remove(self: *Self, key: []const u8) bool {
        if (self.map.fetchRemove(key)) |kv| {
            const node = kv.value;
            self.unlink(node);
            self.allocator.free(node.key);
            self.allocator.free(node.value);
            self.allocator.destroy(node);
            self.current_size -= 1;
            self.current_memory -= (node.key.len + node.value.len);
            return true;
        }
        return false;
    }

    pub fn clear(self: *Self) void {
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            const node = entry.value_ptr.*;
            self.allocator.free(node.key);
            self.allocator.free(node.value);
            self.allocator.destroy(node);
        }
        self.map.clearRetainingCapacity();
        self.head = null;
        self.tail = null;
        self.current_size = 0;
        self.current_memory = 0;
    }

    pub fn getHitRatio(self: *const Self) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn getSize(self: *const Self) usize {
        return self.current_size;
    }

    pub fn getMemoryUsage(self: *const Self) usize {
        return self.current_memory;
    }
};

pub const TreeIdContext = struct {
    pub fn hash(_: @This(), key: [32]u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(&key);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: [32]u8, b: [32]u8) bool {
        return std.mem.eql(u8, &a, &b);
    }
};

pub const FNDSManager = struct {
    fractal_trees: std.HashMap([32]u8, FractalTree, TreeIdContext, std.hash_map.default_max_load_percentage),
    indices: StringHashMap(SelfSimilarIndex),
    cache: LRUCache,
    statistics: FNDSStatistics,
    allocator: Allocator,
    tree_id_counter: usize,
    index_keys: ArrayList([]const u8),
    creation_time: i64,

    const Self = @This();
    const DEFAULT_CACHE_CAPACITY: usize = 1000;
    const DEFAULT_CACHE_MEMORY: usize = 10 * 1024 * 1024;

    pub fn init(allocator: Allocator) Self {
        return Self{
            .fractal_trees = std.HashMap([32]u8, FractalTree, TreeIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .indices = StringHashMap(SelfSimilarIndex).init(allocator),
            .cache = LRUCache.init(allocator, DEFAULT_CACHE_CAPACITY, DEFAULT_CACHE_MEMORY),
            .statistics = FNDSStatistics.init(),
            .allocator = allocator,
            .tree_id_counter = 0,
            .index_keys = ArrayList([]const u8).init(allocator),
            .creation_time = std.time.nanoTimestamp(),
        };
    }

    pub fn initWithCache(allocator: Allocator, cache_capacity: usize, cache_memory: usize) Self {
        return Self{
            .fractal_trees = std.HashMap([32]u8, FractalTree, TreeIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .indices = StringHashMap(SelfSimilarIndex).init(allocator),
            .cache = LRUCache.init(allocator, cache_capacity, cache_memory),
            .statistics = FNDSStatistics.init(),
            .allocator = allocator,
            .tree_id_counter = 0,
            .index_keys = ArrayList([]const u8).init(allocator),
            .creation_time = std.time.nanoTimestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        var tree_iter = self.fractal_trees.iterator();
        while (tree_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.fractal_trees.deinit();

        var index_iter = self.indices.iterator();
        while (index_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.indices.deinit();

        self.cache.deinit();

        for (self.index_keys.items) |k| self.allocator.free(k);
        self.index_keys.deinit();
    }

    pub fn createTree(self: *Self, max_depth: usize, branching_factor: usize) ![32]u8 {
        const start_time = std.time.nanoTimestamp();

        var tree = try FractalTree.init(self.allocator, max_depth, branching_factor);
        errdefer tree.deinit();

        const tree_id = tree.tree_id;
        try self.fractal_trees.put(tree_id, tree);

        self.statistics.total_trees += 1;
        self.statistics.last_operation_time_ns = std.time.nanoTimestamp() - start_time;

        return tree_id;
    }

    pub fn getTree(self: *Self, tree_id: [32]u8) ?*FractalTree {
        return self.fractal_trees.getPtr(tree_id);
    }

    pub fn getTreeConst(self: *const Self, tree_id: [32]u8) ?FractalTree {
        return self.fractal_trees.get(tree_id);
    }

    pub fn removeTree(self: *Self, tree_id: [32]u8) bool {
        if (self.fractal_trees.fetchRemove(tree_id)) |removed| {
            var tree = removed.value;
            self.statistics.total_nodes_across_trees -= tree.total_nodes;
            tree.deinit();
            self.statistics.total_trees -= 1;
            return true;
        }
        return false;
    }

    pub fn createIndex(self: *Self, index_id: []const u8) !void {
        const start_time = std.time.nanoTimestamp();

        const id_copy = try self.allocator.dupe(u8, index_id);
        errdefer self.allocator.free(id_copy);

        const index = SelfSimilarIndex.init(self.allocator);
        try self.indices.put(id_copy, index);
        try self.index_keys.append(id_copy);

        self.statistics.total_indices += 1;
        self.statistics.last_operation_time_ns = std.time.nanoTimestamp() - start_time;
    }

    pub fn getIndex(self: *Self, index_id: []const u8) ?*SelfSimilarIndex {
        return self.indices.getPtr(index_id);
    }

    pub fn getIndexConst(self: *const Self, index_id: []const u8) ?SelfSimilarIndex {
        return self.indices.get(index_id);
    }

    pub fn removeIndex(self: *Self, index_id: []const u8) bool {
        if (self.indices.fetchRemove(index_id)) |removed| {
            var index = removed.value;
            self.statistics.total_patterns_indexed -= index.pattern_count;
            index.deinit();
            self.allocator.free(removed.key);
            self.statistics.total_indices -= 1;
            return true;
        }
        return false;
    }

    pub fn cacheGet(self: *Self, key: []const u8) ?[]const u8 {
        if (self.cache.get(key)) |value| {
            self.statistics.recordCacheHit();
            return value;
        }
        self.statistics.recordCacheMiss();
        return null;
    }

    pub fn cachePut(self: *Self, key: []const u8, value: []const u8) !void {
        try self.cache.put(key, value);
    }

    pub fn cacheRemove(self: *Self, key: []const u8) bool {
        return self.cache.remove(key);
    }

    pub fn cacheClear(self: *Self) void {
        self.cache.clear();
    }

    pub fn getStatistics(self: *Self) FNDSStatistics {
        self.updateStatistics();
        return self.statistics;
    }

    fn updateStatistics(self: *Self) void {
        self.statistics.total_trees = self.fractal_trees.count();
        self.statistics.total_indices = self.indices.count();
        self.statistics.cache_hits = self.cache.hits;
        self.statistics.cache_misses = self.cache.misses;
        self.statistics.updateCacheHitRatio();

        var total_nodes: usize = 0;
        var total_patterns: usize = 0;
        var depths = ArrayList(usize).init(self.allocator);
        defer depths.deinit();

        var tree_iter = self.fractal_trees.iterator();
        while (tree_iter.next()) |entry| {
            var tree = entry.value_ptr;
            total_nodes += tree.total_nodes;
            depths.append(tree.getDepth()) catch continue;
        }

        var index_iter = self.indices.iterator();
        while (index_iter.next()) |entry| {
            total_patterns += entry.value_ptr.pattern_count;
        }

        self.statistics.total_nodes_across_trees = total_nodes;
        self.statistics.total_patterns_indexed = total_patterns;
        self.statistics.updateAverageTreeDepth(depths.items);

        var memory_estimate: usize = 0;
        memory_estimate += self.cache.current_memory;
        memory_estimate += self.fractal_trees.count() * @sizeOf(FractalTree);
        memory_estimate += self.indices.count() * @sizeOf(SelfSimilarIndex);
        self.statistics.memory_used = memory_estimate;
    }

    pub fn getTreeCount(self: *const Self) usize {
        return self.fractal_trees.count();
    }

    pub fn getIndexCount(self: *const Self) usize {
        return self.indices.count();
    }

    pub fn getCacheHitRatio(self: *const Self) f64 {
        return self.cache.getHitRatio();
    }

    pub fn insertIntoTree(self: *Self, tree_id: [32]u8, node_id: []const u8, data: []const u8, level: usize) !bool {
        if (self.getTree(tree_id)) |tree| {
            const result = try tree.insert(node_id, data, level);
            if (result) {
                self.statistics.total_nodes_across_trees += 1;
            }
            return result;
        }
        return false;
    }

    pub fn searchInTree(self: *Self, tree_id: [32]u8, node_id: []const u8) ?*FractalNodeData {
        const cache_key = self.buildCacheKey(tree_id, node_id);
        _ = self.cacheGet(&cache_key);

        if (self.getTree(tree_id)) |tree| {
            const node = tree.search(node_id);
            if (node) |n| {
                _ = self.cachePut(&cache_key, n.data) catch {};
            }
            return node;
        }
        return null;
    }

    fn buildCacheKey(self: *const Self, tree_id: [32]u8, node_id: []const u8) [64]u8 {
        _ = self;
        var key: [64]u8 = undefined;
        @memcpy(key[0..32], &tree_id);
        const copy_len = @min(node_id.len, 32);
        @memcpy(key[32 .. 32 + copy_len], node_id[0..copy_len]);
        if (copy_len < 32) {
            @memset(key[32 + copy_len .. 64], 0);
        }
        return key;
    }

    pub fn addPatternToIndex(self: *Self, index_id: []const u8, pattern: []const u8, location: PatternLocation) !void {
        if (self.getIndex(index_id)) |index| {
            try index.addPattern(pattern, location);
            self.statistics.total_patterns_indexed += 1;
        }
    }

    pub fn findPatternInIndex(self: *Self, index_id: []const u8, pattern: []const u8) []PatternLocation {
        if (self.getIndex(index_id)) |index| {
            return index.findPattern(pattern);
        }
        return &[_]PatternLocation{};
    }

    pub fn computeGlobalFractalDimension(self: *Self) f64 {
        var sum: f64 = 0.0;
        var count: usize = 0;

        var tree_iter = self.fractal_trees.iterator();
        while (tree_iter.next()) |entry| {
            var tree = entry.value_ptr;
            sum += tree.computeFractalDimension();
            count += 1;
        }

        var index_iter = self.indices.iterator();
        while (index_iter.next()) |entry| {
            var index = entry.value_ptr;
            sum += index.computeFractalDimension();
            count += 1;
        }

        if (count == 0) return 0.0;
        return sum / @as(f64, @floatFromInt(count));
    }
};

pub const PatternLengthOutOfRange = error.PatternLengthOutOfRange;

test "FractalLevel basic operations" {
    const allocator = std.testing.allocator;

    var level = FractalLevel.init(allocator, 0, 1.0);
    defer level.deinit();

    var node1 = try FractalNodeData.init(allocator, "node1", "data1", 1.0, 1.0);
    try level.addNode(node1);

    try std.testing.expect(level.node_count == 1);
    try std.testing.expect(level.getNode("node1") != null);

    var edge = try FractalEdgeData.init(allocator, "node1", "node2", 0.5, 1.0, .hierarchical);
    try level.addEdge(edge);

    try std.testing.expect(level.edge_count == 1);

    try std.testing.expect(level.removeNode("node1") == true);
    try std.testing.expect(level.node_count == 0);
}

test "FractalTree insert and search" {
    const allocator = std.testing.allocator;

    var tree = try FractalTree.init(allocator, 5, 4);
    defer tree.deinit();

    const inserted = try tree.insert("test_node", "test_data", 2);
    try std.testing.expect(inserted == true);
    try std.testing.expect(tree.total_nodes == 1);

    const found = tree.search("test_node");
    try std.testing.expect(found != null);
    try std.testing.expect(std.mem.eql(u8, found.?.data, "test_data"));

    const deleted = tree.delete("test_node");
    try std.testing.expect(deleted == true);
    try std.testing.expect(tree.total_nodes == 0);
}

test "SelfSimilarIndex pattern operations" {
    const allocator = std.testing.allocator;

    var index = SelfSimilarIndex.init(allocator);
    defer index.deinit();

    var tree_id: [32]u8 = undefined;
    @memset(&tree_id, 0);

    var location = try PatternLocation.init(allocator, tree_id, 0, "node1", 0, 5, 1.0);
    try index.addPattern("test_pattern", location);

    try std.testing.expect(index.pattern_count == 1);

    const found = index.findPattern("test_pattern");
    try std.testing.expect(found.len == 1);
}

test "CoalescedHashMap operations" {
    const allocator = std.testing.allocator;

    var map = try CoalescedHashMap(u64, u64).init(allocator);
    defer map.deinit();

    try map.put(1, 100);
    try map.put(2, 200);
    try map.put(3, 300);

    try std.testing.expect(map.count() == 3);
    try std.testing.expect(map.get(1) == 100);
    try std.testing.expect(map.get(2) == 200);
    try std.testing.expect(map.get(3) == 300);

    try std.testing.expect(map.remove(2) == true);
    try std.testing.expect(map.count() == 2);
    try std.testing.expect(map.get(2) == null);
}

test "LRUCache operations" {
    const allocator = std.testing.allocator;

    var cache = LRUCache.init(allocator, 3, 1024);
    defer cache.deinit();

    try cache.put("key1", "value1");
    try cache.put("key2", "value2");
    try cache.put("key3", "value3");

    try std.testing.expect(cache.getSize() == 3);

    const v1 = cache.get("key1");
    try std.testing.expect(v1 != null);
    try std.testing.expect(std.mem.eql(u8, v1.?, "value1"));

    try cache.put("key4", "value4");

    try std.testing.expect(cache.getSize() == 3);
}

test "FNDSManager full workflow" {
    const allocator = std.testing.allocator;

    var manager = FNDSManager.init(allocator);
    defer manager.deinit();

    const tree_id = try manager.createTree(5, 4);
    try std.testing.expect(manager.getTreeCount() == 1);

    _ = try manager.insertIntoTree(tree_id, "node1", "data1", 1);

    try manager.createIndex("test_index");
    try std.testing.expect(manager.getIndexCount() == 1);

    try manager.cachePut("cache_key", "cache_value");
    const cached = manager.cacheGet("cache_key");
    try std.testing.expect(cached != null);

    const stats = manager.getStatistics();
    try std.testing.expect(stats.total_trees == 1);
    try std.testing.expect(stats.total_indices == 1);
    try std.testing.expect(stats.cache_hits == 1);
}