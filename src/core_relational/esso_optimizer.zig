const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const quantum_logic = @import("quantum_logic.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
const Node = nsir_core.Node;
const Edge = nsir_core.Edge;
const EdgeQuality = nsir_core.EdgeQuality;
const EdgeKey = nsir_core.EdgeKey;

const QuantumState = quantum_logic.QuantumState;
const RelationalQuantumLogic = quantum_logic.RelationalQuantumLogic;
const LogicGate = quantum_logic.LogicGate;

pub const ObjectiveFunction = *const fn (*const SelfSimilarRelationalGraph) f64;

pub fn cloneGraph(allocator: Allocator, source: *const SelfSimilarRelationalGraph) !SelfSimilarRelationalGraph {
    var new_graph = SelfSimilarRelationalGraph.init(allocator);
    errdefer new_graph.deinit();

    var node_iter = source.nodes.iterator();
    while (node_iter.next()) |entry| {
        var cloned_node = try entry.value_ptr.clone(allocator);
        try new_graph.addNode(cloned_node);
    }

    var edge_iter = source.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |*edge| {
            var cloned_edge = try edge.clone(allocator);
            try new_graph.addEdge(cloned_edge);
        }
    }

    new_graph.fractal_depth = source.fractal_depth;
    @memcpy(&new_graph.topology_hash, &source.topology_hash);

    return new_graph;
}

pub const SymmetryGroup = enum(u8) {
    identity = 0,
    reflection = 1,
    rotation_90 = 2,
    rotation_180 = 3,
    rotation_270 = 4,
    translation = 5,

    pub fn toString(self: SymmetryGroup) []const u8 {
        return switch (self) {
            .identity => "identity",
            .reflection => "reflection",
            .rotation_90 => "rotation_90",
            .rotation_180 => "rotation_180",
            .rotation_270 => "rotation_270",
            .translation => "translation",
        };
    }

    pub fn fromString(s: []const u8) ?SymmetryGroup {
        if (std.mem.eql(u8, s, "identity")) return .identity;
        if (std.mem.eql(u8, s, "reflection")) return .reflection;
        if (std.mem.eql(u8, s, "rotation_90")) return .rotation_90;
        if (std.mem.eql(u8, s, "rotation_180")) return .rotation_180;
        if (std.mem.eql(u8, s, "rotation_270")) return .rotation_270;
        if (std.mem.eql(u8, s, "translation")) return .translation;
        return null;
    }

    pub fn getAngle(self: SymmetryGroup) f64 {
        return switch (self) {
            .identity => 0.0,
            .reflection => std.math.pi,
            .rotation_90 => std.math.pi / 2.0,
            .rotation_180 => std.math.pi,
            .rotation_270 => 3.0 * std.math.pi / 2.0,
            .translation => 0.0,
        };
    }

    pub fn getOrder(self: SymmetryGroup) usize {
        return switch (self) {
            .identity => 1,
            .reflection => 2,
            .rotation_90 => 4,
            .rotation_180 => 2,
            .rotation_270 => 4,
            .translation => 1,
        };
    }
};

const SymmetryTransform = struct {
    group: SymmetryGroup,
    origin_x: f64,
    origin_y: f64,
    parameters: [4]f64,
    scale_factor: f64,

    const Self = @This();

    pub fn init(group: SymmetryGroup) Self {
        return Self{
            .group = group,
            .origin_x = 0.0,
            .origin_y = 0.0,
            .parameters = [4]f64{ 0.0, 0.0, 1.0, 0.0 },
            .scale_factor = 1.0,
        };
    }

    pub fn initWithParams(group: SymmetryGroup, params: [4]f64) Self {
        return Self{
            .group = group,
            .origin_x = params[0],
            .origin_y = params[1],
            .parameters = params,
            .scale_factor = if (params[2] != 0.0) params[2] else 1.0,
        };
    }

    pub fn apply(self: *const Self, x: f64, y: f64) struct { x: f64, y: f64 } {
        const dx = x - self.origin_x;
        const dy = y - self.origin_y;

        return switch (self.group) {
            .identity => .{
                .x = x,
                .y = y,
            },
            .reflection => .{
                .x = self.origin_x + dx * std.math.cos(f64, 2.0 * self.parameters[3]) + dy * std.math.sin(f64, 2.0 * self.parameters[3]),
                .y = self.origin_y + dx * std.math.sin(f64, 2.0 * self.parameters[3]) - dy * std.math.cos(f64, 2.0 * self.parameters[3]),
            },
            .rotation_90 => .{
                .x = self.origin_x - dy * self.scale_factor,
                .y = self.origin_y + dx * self.scale_factor,
            },
            .rotation_180 => .{
                .x = self.origin_x - dx * self.scale_factor,
                .y = self.origin_y - dy * self.scale_factor,
            },
            .rotation_270 => .{
                .x = self.origin_x + dy * self.scale_factor,
                .y = self.origin_y - dx * self.scale_factor,
            },
            .translation => .{
                .x = x + self.parameters[0] * self.scale_factor,
                .y = y + self.parameters[1] * self.scale_factor,
            },
        };
    }

    pub fn applyToComplex(self: *const Self, z: Complex(f64)) Complex(f64) {
        const result = self.apply(z.re, z.im);
        return Complex(f64){ .re = result.x, .im = result.y };
    }

    pub fn applyToQuantumState(self: *const Self, state: *const QuantumState) QuantumState {
        const transformed = self.apply(state.amplitude_real, state.amplitude_imag);
        const phase_shift = self.group.getAngle() + self.parameters[3];
        return QuantumState{
            .amplitude_real = transformed.x,
            .amplitude_imag = transformed.y,
            .phase = state.phase + phase_shift,
            .entanglement_degree = state.entanglement_degree,
        };
    }

    pub fn inverse(self: *const Self) Self {
        return switch (self.group) {
            .identity => self.*,
            .reflection => self.*,
            .rotation_90 => Self{
                .group = .rotation_270,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = self.parameters,
                .scale_factor = self.scale_factor,
            },
            .rotation_180 => self.*,
            .rotation_270 => Self{
                .group = .rotation_90,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = self.parameters,
                .scale_factor = self.scale_factor,
            },
            .translation => Self{
                .group = .translation,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ -self.parameters[0], -self.parameters[1], self.parameters[2], self.parameters[3] },
                .scale_factor = self.scale_factor,
            },
        };
    }

    pub fn compose(self: *const Self, other: *const Self) Self {
        if (self.group == .identity) {
            return other.*;
        }
        if (other.group == .identity) {
            return self.*;
        }
        if (self.group == .rotation_90 and other.group == .rotation_90) {
            return Self.init(.rotation_180);
        }
        if (self.group == .rotation_90 and other.group == .rotation_180) {
            return Self.init(.rotation_270);
        }
        if (self.group == .rotation_180 and other.group == .rotation_90) {
            return Self.init(.rotation_270);
        }
        if (self.group == .rotation_180 and other.group == .rotation_180) {
            return Self.init(.identity);
        }
        if (self.group == .rotation_270 and other.group == .rotation_90) {
            return Self.init(.identity);
        }
        if (self.group == .reflection and other.group == .reflection) {
            const angle_diff = self.parameters[3] - other.parameters[3];
            if (@fabs(angle_diff) < 1e-10) {
                return Self.init(.identity);
            }
            return Self.initWithParams(.rotation_180, [4]f64{ 0.0, 0.0, 1.0, angle_diff });
        }
        if (self.group == .translation and other.group == .translation) {
            return Self.initWithParams(.translation, [4]f64{
                self.parameters[0] + other.parameters[0],
                self.parameters[1] + other.parameters[1],
                self.parameters[2],
                self.parameters[3],
            });
        }
        var composed_params: [4]f64 = undefined;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            composed_params[i] = (self.parameters[i] + other.parameters[i]) / 2.0;
        }
        return Self.initWithParams(self.group, composed_params);
    }

    pub fn getRotationMatrix(self: *const Self) [2][2]f64 {
        const angle = self.group.getAngle();
        const cos_a = std.math.cos(f64, angle);
        const sin_a = std.math.sin(f64, angle);

        return switch (self.group) {
            .identity => [2][2]f64{ [2]f64{ 1.0, 0.0 }, [2]f64{ 0.0, 1.0 } },
            .reflection => [2][2]f64{
                [2]f64{ std.math.cos(f64, 2.0 * self.parameters[3]), std.math.sin(f64, 2.0 * self.parameters[3]) },
                [2]f64{ std.math.sin(f64, 2.0 * self.parameters[3]), -std.math.cos(f64, 2.0 * self.parameters[3]) },
            },
            .rotation_90, .rotation_180, .rotation_270 => [2][2]f64{
                [2]f64{ cos_a, -sin_a },
                [2]f64{ sin_a, cos_a },
            },
            .translation => [2][2]f64{ [2]f64{ 1.0, 0.0 }, [2]f64{ 0.0, 1.0 } },
        };
    }

    pub fn determinant(self: *const Self) f64 {
        const mat = self.getRotationMatrix();
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }

    pub fn isIsometry(self: *const Self) bool {
        const det = self.determinant();
        return @fabs(@fabs(det) - 1.0) < 1e-10;
    }
};

pub const NodePairKey = struct {
    node1: []const u8,
    node2: []const u8,
};

pub const NodePairKeyContext = struct {
    pub fn hash(self: @This(), key: NodePairKey) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key.node1);
        hasher.update(key.node2);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: NodePairKey, b: NodePairKey) bool {
        _ = self;
        return std.mem.eql(u8, a.node1, b.node1) and std.mem.eql(u8, a.node2, b.node2);
    }
};

pub const EntanglementInfo = struct {
    correlation_strength: f64,
    phase_difference: f64,
    creation_time: i64,
    interaction_count: usize,

    pub fn init(correlation: f64, phase_diff: f64) EntanglementInfo {
        const now = std.time.nanoTimestamp() catch 0;
        return EntanglementInfo{
            .correlation_strength = correlation,
            .phase_difference = phase_diff,
            .creation_time = now,
            .interaction_count = 0,
        };
    }

    pub fn update(self: *EntanglementInfo, new_correlation: f64, new_phase: f64) void {
        self.correlation_strength = (self.correlation_strength + new_correlation) / 2.0;
        self.phase_difference = (self.phase_difference + new_phase) / 2.0;
        self.interaction_count += 1;
    }

    pub fn getAge(self: *const EntanglementInfo) i64 {
        return (std.time.nanoTimestamp() catch 0) - self.creation_time;
    }

    pub fn getDecayFactor(self: *const EntanglementInfo, half_life_ms: i64) f64 {
        const age = self.getAge();
        const half_life_ns = half_life_ms * 1_000_000;
        const decay_constant = std.math.log(f64, 2.0) / @as(f64, @floatFromInt(half_life_ns));
        return std.math.exp(f64, -decay_constant * @as(f64, @floatFromInt(age)));
    }
};

pub const OptimizationState = struct {
    graph: *SelfSimilarRelationalGraph,
    energy: f64,
    entanglement_percentage: f64,
    iteration: usize,
    allocator: Allocator,
    owns_graph: bool,
    entanglement_map: std.HashMap(NodePairKey, EntanglementInfo, NodePairKeyContext, std.hash_map.default_max_load_percentage),

    const Self = @This();

    pub fn init(allocator: Allocator, graph: *SelfSimilarRelationalGraph, energy: f64, owns_graph: bool) Self {
        return Self{
            .graph = graph,
            .energy = energy,
            .entanglement_percentage = 0.0,
            .iteration = 0,
            .allocator = allocator,
            .owns_graph = owns_graph,
            .entanglement_map = std.HashMap(NodePairKey, EntanglementInfo, NodePairKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.entanglement_map.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.node1);
            self.allocator.free(entry.key_ptr.node2);
        }
        self.entanglement_map.deinit();
        if (self.owns_graph) {
            self.graph.deinit();
            self.allocator.destroy(self.graph);
        }
    }

    pub fn addEntanglement(self: *Self, node1: []const u8, node2: []const u8, info: EntanglementInfo) !void {
        const key1 = try self.allocator.dupe(u8, node1);
        errdefer self.allocator.free(key1);
        const key2 = try self.allocator.dupe(u8, node2);
        errdefer self.allocator.free(key2);

        const pair_key = NodePairKey{ .node1 = key1, .node2 = key2 };
        try self.entanglement_map.put(pair_key, info);
    }

    pub fn getEntanglement(self: *const Self, node1: []const u8, node2: []const u8) ?EntanglementInfo {
        const pair_key = NodePairKey{ .node1 = node1, .node2 = node2 };
        if (self.entanglement_map.get(pair_key)) |info| {
            return info;
        }
        const reverse_key = NodePairKey{ .node1 = node2, .node2 = node1 };
        return self.entanglement_map.get(reverse_key);
    }

    pub fn hasEntanglement(self: *const Self, node1: []const u8, node2: []const u8) bool {
        return self.getEntanglement(node1, node2) != null;
    }

    pub fn entangledPairsCount(self: *const Self) usize {
        return self.entanglement_map.count();
    }

    pub fn updateEntanglement(self: *Self, node1: []const u8, node2: []const u8, new_correlation: f64, new_phase: f64) void {
        const pair_key = NodePairKey{ .node1 = node1, .node2 = node2 };
        if (self.entanglement_map.getPtr(pair_key)) |info| {
            info.update(new_correlation, new_phase);
            return;
        }
        const reverse_key = NodePairKey{ .node1 = node2, .node2 = node1 };
        if (self.entanglement_map.getPtr(reverse_key)) |info| {
            info.update(new_correlation, new_phase);
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        const new_graph = try allocator.create(SelfSimilarRelationalGraph);
        errdefer allocator.destroy(new_graph);
        new_graph.* = try cloneGraph(allocator, self.graph);

        var new_state = Self{
            .graph = new_graph,
            .energy = self.energy,
            .entanglement_percentage = self.entanglement_percentage,
            .iteration = self.iteration,
            .allocator = allocator,
            .owns_graph = true,
            .entanglement_map = std.HashMap(NodePairKey, EntanglementInfo, NodePairKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
        };

        var iter = self.entanglement_map.iterator();
        while (iter.next()) |entry| {
            const key1 = try allocator.dupe(u8, entry.key_ptr.node1);
            errdefer allocator.free(key1);
            const key2 = try allocator.dupe(u8, entry.key_ptr.node2);
            errdefer allocator.free(key2);
            const new_key = NodePairKey{ .node1 = key1, .node2 = key2 };
            try new_state.entanglement_map.put(new_key, entry.value_ptr.*);
        }

        return new_state;
    }

    pub fn averageEntanglement(self: *const Self) f64 {
        if (self.entanglement_map.count() == 0) return 0.0;
        var total: f64 = 0.0;
        var iter = self.entanglement_map.iterator();
        while (iter.next()) |entry| {
            total += entry.value_ptr.correlation_strength;
        }
        return total / @as(f64, @floatFromInt(self.entanglement_map.count()));
    }
};

pub const OptimizationStatistics = struct {
    iterations_completed: usize,
    moves_accepted: usize,
    moves_rejected: usize,
    best_energy: f64,
    current_energy: f64,
    symmetries_detected: usize,
    entangled_pairs: usize,
    elapsed_time_ms: i64,
    start_time_ms: i64,
    acceptance_rate: f64,
    cooling_factor_applied: usize,
    local_minima_escapes: usize,
    convergence_delta: f64,
    temperature: f64,
    total_energy_evaluations: usize,
    average_move_delta: f64,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .iterations_completed = 0,
            .moves_accepted = 0,
            .moves_rejected = 0,
            .best_energy = std.math.inf(f64),
            .current_energy = std.math.inf(f64),
            .symmetries_detected = 0,
            .entangled_pairs = 0,
            .elapsed_time_ms = 0,
            .start_time_ms = std.time.nanoTimestamp() catch 0,
            .acceptance_rate = 0.0,
            .cooling_factor_applied = 0,
            .local_minima_escapes = 0,
            .convergence_delta = 0.0,
            .temperature = 0.0,
            .total_energy_evaluations = 0,
            .average_move_delta = 0.0,
        };
    }

    pub fn updateAcceptanceRate(self: *Self) void {
        const total_moves = self.moves_accepted + self.moves_rejected;
        if (total_moves > 0) {
            self.acceptance_rate = @as(f64, @floatFromInt(self.moves_accepted)) / @as(f64, @floatFromInt(total_moves));
        }
    }

    pub fn updateElapsedTime(self: *Self) void {
        const now = std.time.nanoTimestamp() catch 0;
        self.elapsed_time_ms = (now - self.start_time_ms) / 1_000_000;
    }

    pub fn iterationsPerSecond(self: *const Self) f64 {
        if (self.elapsed_time_ms == 0) return 0.0;
        return @as(f64, @floatFromInt(self.iterations_completed)) * 1000.0 / @as(f64, @floatFromInt(self.elapsed_time_ms));
    }

    pub fn isConverged(self: *const Self, threshold: f64) bool {
        return @fabs(self.convergence_delta) < threshold and self.iterations_completed > 10;
    }
};

pub const SymmetryPattern = struct {
    pattern_id: [16]u8,
    transform: SymmetryTransform,
    nodes: ArrayList([]const u8),
    symmetry_score: f64,
    resonance_frequency: f64,
    creation_timestamp: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, transform: SymmetryTransform) Self {
        var hasher = Sha256.init(.{});
        const timestamp = std.time.nanoTimestamp() catch 0;
        hasher.update(std.mem.asBytes(&timestamp));
        hasher.update(std.mem.asBytes(&transform.group));
        var id: [16]u8 = undefined;
        const hash_result = hasher.finalResult();
        @memcpy(&id, hash_result[0..16]);

        return Self{
            .pattern_id = id,
            .transform = transform,
            .nodes = ArrayList([]const u8).init(allocator),
            .symmetry_score = 0.0,
            .resonance_frequency = 0.0,
            .creation_timestamp = timestamp,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.nodes.items) |node_id| {
            self.allocator.free(node_id);
        }
        self.nodes.deinit();
    }

    pub fn addNode(self: *Self, node_id: []const u8) !void {
        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);
        try self.nodes.append(id_copy);
    }

    pub fn getPatternIdHex(self: *const Self) [32]u8 {
        var hex_buf: [64]u8 = undefined;
        const len = std.fmt.bufPrint(&hex_buf, "{s}", .{std.fmt.fmtSliceHexLower(&self.pattern_id)}) catch unreachable;
        var result: [32]u8 = undefined;
        @memcpy(&result, len[0..32]);
        return result;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_pattern = Self{
            .pattern_id = self.pattern_id,
            .transform = self.transform,
            .nodes = ArrayList([]const u8).init(allocator),
            .symmetry_score = self.symmetry_score,
            .resonance_frequency = self.resonance_frequency,
            .creation_timestamp = self.creation_timestamp,
            .allocator = allocator,
        };
        for (self.nodes.items) |node_id| {
            const id_copy = try allocator.dupe(u8, node_id);
            errdefer allocator.free(id_copy);
            try new_pattern.nodes.append(id_copy);
        }
        return new_pattern;
    }
};

pub const EntangledStochasticSymmetryOptimizer = struct {
    temperature: f64,
    cooling_rate: f64,
    max_iterations: usize,
    current_iteration: usize,
    min_temperature: f64,
    current_state: ?OptimizationState,
    best_state: ?OptimizationState,
    detected_patterns: ArrayList(SymmetryPattern),
    symmetry_transforms: ArrayList(SymmetryTransform),
    allocator: Allocator,
    statistics: OptimizationStatistics,
    objective_fn: ?ObjectiveFunction,
    prng: std.rand.DefaultPrng,
    energy_history: ArrayList(f64),
    temperature_history: ArrayList(f64),
    reheat_factor: f64,
    entanglement_decay_half_life: i64,
    symmetry_detection_interval: usize,
    convergence_threshold: f64,
    adaptive_cooling: bool,

    const Self = @This();
    const DEFAULT_INITIAL_TEMP: f64 = 100.0;
    const DEFAULT_COOLING_RATE: f64 = 0.95;
    const DEFAULT_MAX_ITERATIONS: usize = 10000;
    const DEFAULT_MIN_TEMP: f64 = 0.001;
    const DEFAULT_REHEAT_THRESHOLD: f64 = 0.1;
    const DEFAULT_REHEAT_FACTOR: f64 = 2.0;
    const DEFAULT_ENTANGLEMENT_HALF_LIFE: i64 = 60000;
    const DEFAULT_SYMMETRY_INTERVAL: usize = 50;
    const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 1e-8;

    pub fn init(allocator: Allocator, initial_temp: f64, cooling_rate: f64, max_iterations: usize) Self {
        const seed = @as(u64, @truncate(@fabs(std.time.nanoTimestamp() catch 0)));
        return Self{
            .temperature = initial_temp,
            .cooling_rate = cooling_rate,
            .max_iterations = max_iterations,
            .current_iteration = 0,
            .min_temperature = DEFAULT_MIN_TEMP,
            .current_state = null,
            .best_state = null,
            .symmetry_transforms = ArrayList(SymmetryTransform).init(allocator),
            .detected_patterns = ArrayList(SymmetryPattern).init(allocator),
            .allocator = allocator,
            .statistics = OptimizationStatistics.init(),
            .objective_fn = null,
            .prng = std.rand.DefaultPrng.init(seed),
            .energy_history = ArrayList(f64).init(allocator),
            .temperature_history = ArrayList(f64).init(allocator),
            .reheat_factor = DEFAULT_REHEAT_FACTOR,
            .entanglement_decay_half_life = DEFAULT_ENTANGLEMENT_HALF_LIFE,
            .symmetry_detection_interval = DEFAULT_SYMMETRY_INTERVAL,
            .convergence_threshold = DEFAULT_CONVERGENCE_THRESHOLD,
            .adaptive_cooling = true,
        };
    }

    pub fn initWithSeed(allocator: Allocator, initial_temp: f64, cooling_rate: f64, max_iterations: usize, seed: u64) Self {
        var optimizer = Self.init(allocator, initial_temp, cooling_rate, max_iterations);
        optimizer.prng = std.rand.DefaultPrng.init(seed);
        return optimizer;
    }

    pub fn initDefault(allocator: Allocator) Self {
        return Self.init(allocator, DEFAULT_INITIAL_TEMP, DEFAULT_COOLING_RATE, DEFAULT_MAX_ITERATIONS);
    }

    pub fn deinit(self: *Self) void {
        if (self.current_state) |*state| {
            state.deinit();
        }
        if (self.best_state) |*state| {
            state.deinit();
        }
        for (self.detected_patterns.items) |*pattern| {
            pattern.deinit();
        }
        self.detected_patterns.deinit();
        self.symmetry_transforms.deinit();
        self.energy_history.deinit();
        self.temperature_history.deinit();
    }

    pub fn setObjectiveFunction(self: *Self, obj_fn: ObjectiveFunction) void {
        self.objective_fn = obj_fn;
    }

    pub fn setAdaptiveCooling(self: *Self, enabled: bool) void {
        self.adaptive_cooling = enabled;
    }

    pub fn setMinTemperature(self: *Self, min_temp: f64) void {
        self.min_temperature = min_temp;
    }

    pub fn setReheatFactor(self: *Self, factor: f64) void {
        self.reheat_factor = factor;
    }

    pub fn setSymmetryDetectionInterval(self: *Self, interval: usize) void {
        self.symmetry_detection_interval = interval;
    }

    pub fn optimize(self: *Self, graph: *const SelfSimilarRelationalGraph, objective_fn: ObjectiveFunction) !*SelfSimilarRelationalGraph {
        self.objective_fn = objective_fn;
        self.statistics = OptimizationStatistics.init();

        const initial_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        errdefer self.allocator.destroy(initial_graph);
        initial_graph.* = try cloneGraph(self.allocator, graph);

        const initial_energy = self.computeEnergy(initial_graph, objective_fn);

        self.current_state = OptimizationState.init(self.allocator, initial_graph, initial_energy, true);
        self.best_state = try self.current_state.?.clone(self.allocator);

        self.statistics.best_energy = initial_energy;
        self.statistics.current_energy = initial_energy;

        const initial_transforms = try self.detectSymmetries(graph);
        defer self.allocator.free(initial_transforms);
        for (initial_transforms) |transform| {
            try self.symmetry_transforms.append(transform);
        }
        self.statistics.symmetries_detected = initial_transforms.len;

        var stagnation_counter: usize = 0;
        var previous_energy = initial_energy;

        while (self.current_iteration < self.max_iterations) : (self.current_iteration += 1) {
            try self.energy_history.append(self.current_state.?.energy);
            try self.temperature_history.append(self.temperature);

            if (self.current_iteration % self.symmetry_detection_interval == 0 and self.current_iteration > 0) {
                const new_transforms = try self.detectSymmetries(self.current_state.?.graph);
                defer self.allocator.free(new_transforms);
                for (new_transforms) |transform| {
                    var is_duplicate = false;
                    for (self.symmetry_transforms.items) |existing| {
                        if (existing.group == transform.group) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    if (!is_duplicate) {
                        try self.symmetry_transforms.append(transform);
                    }
                }
                self.statistics.symmetries_detected = self.symmetry_transforms.items.len;
            }

            var candidate_state = try self.proposeMove();
            defer if (candidate_state.owns_graph) {
                candidate_state.deinit();
            };

            const delta_energy = candidate_state.energy - self.current_state.?.energy;

            if (self.acceptMove(delta_energy)) {
                var accepted_state = try candidate_state.clone(self.allocator);
                if (self.current_state) |*old_state| {
                    old_state.deinit();
                }
                self.current_state = accepted_state;
                self.statistics.moves_accepted += 1;

                if (candidate_state.energy < self.best_state.?.energy) {
                    if (self.best_state) |*old_best| {
                        old_best.deinit();
                    }
                    self.best_state = try candidate_state.clone(self.allocator);
                    self.statistics.best_energy = candidate_state.energy;
                }
                stagnation_counter = 0;
            } else {
                self.statistics.moves_rejected += 1;
                stagnation_counter += 1;
            }

            self.statistics.current_energy = self.current_state.?.energy;
            self.statistics.entangled_pairs = self.current_state.?.entangledPairsCount();

            const energy_change = @fabs(self.current_state.?.energy - previous_energy);
            self.statistics.convergence_delta = energy_change;
            previous_energy = self.current_state.?.energy;

            if (self.adaptive_cooling) {
                self.adaptiveCoolTemperature();
            } else {
                self.coolTemperature();
            }

            if (stagnation_counter > self.max_iterations / 10) {
                self.temperature *= self.reheat_factor;
                self.statistics.local_minima_escapes += 1;
                stagnation_counter = 0;
            }

            self.statistics.iterations_completed = self.current_iteration + 1;
            self.statistics.temperature = self.temperature;
            self.statistics.updateAcceptanceRate();
            self.statistics.updateElapsedTime();

            if (self.statistics.isConverged(self.convergence_threshold)) {
                break;
            }
        }

        self.statistics.total_energy_evaluations = self.statistics.moves_accepted + self.statistics.moves_rejected;

        if (self.energy_history.items.len > 1) {
            var total_delta: f64 = 0.0;
            var delta_idx: usize = 1;
            while (delta_idx < self.energy_history.items.len) : (delta_idx += 1) {
                total_delta += @fabs(self.energy_history.items[delta_idx] - self.energy_history.items[delta_idx - 1]);
            }
            self.statistics.average_move_delta = total_delta / @as(f64, @floatFromInt(self.energy_history.items.len - 1));
        }

        return self.best_state.?.graph;
    }

    fn computeEnergy(self: *Self, graph: *const SelfSimilarRelationalGraph, objective_fn: ObjectiveFunction) f64 {
        _ = self;
        return objective_fn(graph);
    }

    fn computeObjective(self: *Self, graph: *const SelfSimilarRelationalGraph) f64 {
        if (self.objective_fn) |obj_fn| {
            return obj_fn(graph);
        }
        return self.defaultObjective(graph);
    }

    fn defaultObjective(self: *Self, graph: *const SelfSimilarRelationalGraph) f64 {
        _ = self;
        var total_energy: f64 = 0.0;
        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_energy += edge.weight * edge.fractal_dimension;
                total_energy += edge.quantum_correlation.magnitude();
            }
        }
        return total_energy;
    }

    fn proposeMove(self: *Self) !OptimizationState {
        const current = &self.current_state.?;

        const new_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        errdefer self.allocator.destroy(new_graph);
        new_graph.* = try cloneGraph(self.allocator, current.graph);

        var new_state = OptimizationState.init(self.allocator, new_graph, 0.0, true);

        var ent_iter = current.entanglement_map.iterator();
        while (ent_iter.next()) |entry| {
            try new_state.addEntanglement(entry.key_ptr.node1, entry.key_ptr.node2, entry.value_ptr.*);
        }

        new_state.iteration = current.iteration + 1;

        const move_type = self.prng.random().int(usize) % 5;

        switch (move_type) {
            0 => self.perturbEdgeWeights(new_graph),
            1 => self.perturbQuantumPhases(new_graph),
            2 => try self.createNewEntanglement(new_graph, &new_state),
            3 => self.applySymmetryTransform(new_graph),
            4 => self.perturbNodeStates(new_graph),
            else => {},
        }

        new_state.energy = self.computeEnergy(new_graph, self.objective_fn.?);

        return new_state;
    }

    fn perturbEdgeWeights(self: *Self, graph: *SelfSimilarRelationalGraph) void {
        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                const perturbation = (self.prng.random().float(f64) - 0.5) * self.temperature * 0.1;
                edge.weight = @max(0.0, @min(1.0, edge.weight + perturbation));
            }
        }
    }

    fn perturbQuantumPhases(self: *Self, graph: *SelfSimilarRelationalGraph) void {
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            var node = entry.value_ptr;
            const phase_delta = (self.prng.random().float(f64) - 0.5) * self.temperature * 0.2;
            node.phase = @mod(node.phase + phase_delta, 2.0 * std.math.pi);
        }
    }

    fn perturbNodeStates(self: *Self, graph: *SelfSimilarRelationalGraph) void {
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const state = entry.value_ptr.quantum_state;
            const angle = self.prng.random().float(f64) * 2.0 * std.math.pi;
            const radius = state.magnitude();
            const perturbation = self.temperature * 0.05;

            const new_real = state.re + perturbation * std.math.cos(f64, angle);
            const new_imag = state.im + perturbation * std.math.sin(f64, angle);

            var new_state = Complex(f64){ .re = new_real, .im = new_imag };
            const new_mag = new_state.magnitude();
            if (new_mag > 0.0) {
                new_state = Complex(f64){ .re = new_state.re / new_mag * radius, .im = new_state.im / new_mag * radius };
            }
            entry.value_ptr.quantum_state = new_state;
        }
    }

    fn applySymmetryTransform(self: *Self, graph: *SelfSimilarRelationalGraph) void {
        if (self.symmetry_transforms.items.len == 0) return;

        const transform_idx = self.prng.random().int(usize) % self.symmetry_transforms.items.len;
        const transform = self.symmetry_transforms.items[transform_idx];

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            const transformed_state = transform.applyToQuantumState(&QuantumState{
                .amplitude_real = node.quantum_state.re,
                .amplitude_imag = node.quantum_state.im,
                .phase = node.phase,
                .entanglement_degree = 0.0,
            });
            node.quantum_state = Complex(f64){ .re = transformed_state.amplitude_real, .im = transformed_state.amplitude_imag };
            node.phase = transformed_state.phase;
        }
    }

    fn createNewEntanglement(self: *Self, graph: *SelfSimilarRelationalGraph, state: *OptimizationState) !void {
        const node_count = graph.nodeCount();
        if (node_count < 2) return;

        var node_ids = ArrayList([]const u8).init(self.allocator);
        defer node_ids.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_ids.append(entry.key_ptr.*);
        }

        if (node_ids.items.len < 2) return;

        const idx1 = self.prng.random().int(usize) % node_ids.items.len;
        var idx2 = self.prng.random().int(usize) % node_ids.items.len;
        while (idx2 == idx1 and node_ids.items.len > 1) {
            idx2 = self.prng.random().int(usize) % node_ids.items.len;
        }

        const node1_id = node_ids.items[idx1];
        const node2_id = node_ids.items[idx2];

        if (state.hasEntanglement(node1_id, node2_id)) return;

        const node1 = graph.getNode(node1_id);
        const node2 = graph.getNode(node2_id);

        if (node1 == null or node2 == null) return;

        const phase_diff = @fabs(node1.?.phase - node2.?.phase);
        const correlation = self.prng.random().float(f64) * 0.5 + 0.5;
        const info = EntanglementInfo.init(correlation, phase_diff);

        try state.addEntanglement(node1_id, node2_id, info);
    }

    fn acceptMove(self: *Self, delta_energy: f64) bool {
        if (delta_energy < 0) {
            return true;
        }

        if (self.temperature < self.min_temperature) {
            return false;
        }

        const acceptance_probability = std.math.exp(f64, -delta_energy / self.temperature);
        const random_value = self.prng.random().float(f64);

        return random_value < acceptance_probability;
    }

    fn coolTemperature(self: *Self) void {
        self.temperature *= self.cooling_rate;
        if (self.temperature < self.min_temperature) {
            self.temperature = self.min_temperature;
        }
        self.statistics.cooling_factor_applied += 1;
    }

    fn adaptiveCoolTemperature(self: *Self) void {
        const acceptance_rate = self.statistics.acceptance_rate;

        var rate = self.cooling_rate;
        if (acceptance_rate > 0.6) {
            rate = self.cooling_rate * 0.98;
        } else if (acceptance_rate < 0.2) {
            rate = self.cooling_rate * 1.02;
        }

        self.temperature *= rate;
        if (self.temperature < self.min_temperature) {
            self.temperature = self.min_temperature;
        }
        self.statistics.cooling_factor_applied += 1;
    }

    fn detectSymmetries(self: *Self, graph: *const SelfSimilarRelationalGraph) ![]SymmetryTransform {
        var transforms = ArrayList(SymmetryTransform).init(self.allocator);
        errdefer transforms.deinit();

        try transforms.append(SymmetryTransform.init(.identity));

        const degree_map = try self.computeDegreeMap(graph);
        defer self.allocator.free(degree_map);

        if (degree_map.len >= 2) {
            try transforms.append(SymmetryTransform.init(.reflection));
        }

        const node_count = graph.nodeCount();
        if (node_count >= 4) {
            const edge_count = graph.edgeCount();
            const ratio = @as(f64, @floatFromInt(edge_count)) / @as(f64, @floatFromInt(node_count));
            if (ratio > 1.5) {
                try transforms.append(SymmetryTransform.init(.rotation_90));
            }
        }

        var avg_phase: f64 = 0.0;
        var phase_count: usize = 0;
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            avg_phase += entry.value_ptr.phase;
            phase_count += 1;
        }
        if (phase_count > 0) {
            avg_phase /= @as(f64, @floatFromInt(phase_count));
        }

        node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const phase_diff = @fabs(entry.value_ptr.phase - avg_phase);
            if (phase_diff < 0.1) {
                try transforms.append(SymmetryTransform.init(.rotation_180));
                break;
            }
        }

        return transforms.toOwnedSlice();
    }

    const DegreeEntry = struct {
        node_id: []const u8,
        degree_count: usize,
    };

    fn computeDegreeMap(self: *Self, graph: *const SelfSimilarRelationalGraph) ![]DegreeEntry {
        var entries = ArrayList(DegreeEntry).init(self.allocator);
        errdefer entries.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            var degree: usize = 0;
            var edge_iter = graph.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                if (std.mem.eql(u8, edge_entry.key_ptr.source, entry.key_ptr.*) or
                    std.mem.eql(u8, edge_entry.key_ptr.target, entry.key_ptr.*))
                {
                    degree += edge_entry.value_ptr.items.len;
                }
            }
            try entries.append(DegreeEntry{ .node_id = entry.key_ptr.*, .degree_count = degree });
        }

        return entries.toOwnedSlice();
    }

    fn applyTransformToGraph(self: *Self, graph: *SelfSimilarRelationalGraph, transform: SymmetryTransform) void {
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            const transformed_state = transform.applyToQuantumState(&QuantumState{
                .amplitude_real = node.quantum_state.re,
                .amplitude_imag = node.quantum_state.im,
                .phase = node.phase,
                .entanglement_degree = 0.0,
            });
            node.quantum_state = Complex(f64){ .re = transformed_state.amplitude_real, .im = transformed_state.amplitude_imag };
            node.phase = transformed_state.phase;
        }
        _ = self;
    }

    fn updateEntanglementMap(self: *Self, node_id1: []const u8, node_id2: []const u8) !void {
        _ = node_id1;
        _ = node_id2;

        if (self.current_state == null) return;
        var state = &self.current_state.?;

        var updates = ArrayList(struct { n1: []const u8, n2: []const u8, corr: f64, phase: f64 }).init(self.allocator);
        defer updates.deinit();

        var iter = state.entanglement_map.iterator();
        while (iter.next()) |entry| {
            const info = entry.value_ptr;
            const decay_factor = info.getDecayFactor(self.entanglement_decay_half_life);
            const new_correlation = info.correlation_strength * decay_factor;
            const phase_evolution = info.phase_difference + self.temperature * 0.01;

            if (new_correlation > 0.01) {
                try updates.append(.{
                    .n1 = entry.key_ptr.node1,
                    .n2 = entry.key_ptr.node2,
                    .corr = new_correlation,
                    .phase = phase_evolution,
                });
            }
        }

        for (updates.items) |update| {
            state.updateEntanglement(update.n1, update.n2, update.corr, update.phase);
        }

        var node_iter = state.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            var entanglement_sum: f64 = 0.0;
            var entanglement_count: usize = 0;

            var ent_iter = state.entanglement_map.iterator();
            while (ent_iter.next()) |ent_entry| {
                if (std.mem.eql(u8, ent_entry.key_ptr.node1, node_id) or
                    std.mem.eql(u8, ent_entry.key_ptr.node2, node_id))
                {
                    entanglement_sum += ent_entry.value_ptr.correlation_strength;
                    entanglement_count += 1;
                }
            }

            if (entanglement_count > 0) {
                const avg_entanglement = entanglement_sum / @as(f64, @floatFromInt(entanglement_count));
                const phase_adjustment = avg_entanglement * 0.1;
                entry.value_ptr.phase += phase_adjustment;
            }
        }
    }

    pub fn getStatistics(self: *const Self) OptimizationStatistics {
        return self.statistics;
    }

    pub fn getBestGraph(self: *const Self) ?*SelfSimilarRelationalGraph {
        if (self.best_state) |state| {
            return state.graph;
        }
        return null;
    }

    pub fn getCurrentTemperature(self: *const Self) f64 {
        return self.temperature;
    }

    pub fn getCurrentIteration(self: *const Self) usize {
        return self.current_iteration;
    }

    pub fn getEnergyHistory(self: *const Self) []const f64 {
        return self.energy_history.items;
    }

    pub fn getTemperatureHistory(self: *const Self) []const f64 {
        return self.temperature_history.items;
    }

    pub fn getDetectedPatterns(self: *const Self) []const SymmetryPattern {
        return self.detected_patterns.items;
    }

    pub fn reset(self: *Self) void {
        self.current_iteration = 0;
        self.temperature = DEFAULT_INITIAL_TEMP;

        if (self.current_state) |*state| {
            state.deinit();
        }
        self.current_state = null;

        if (self.best_state) |*state| {
            state.deinit();
        }
        self.best_state = null;

        self.energy_history.clearRetainingCapacity();
        for (self.detected_patterns.items) |*pattern| {
            pattern.deinit();
        }
        self.detected_patterns.clearRetainingCapacity();
        self.symmetry_transforms.clearRetainingCapacity();
        self.temperature_history.clearRetainingCapacity();
        self.statistics = OptimizationStatistics.init();
    }
};

pub fn defaultGraphObjective(graph: *const SelfSimilarRelationalGraph) f64 {
    var total_energy: f64 = 0.0;

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_energy += edge.weight * edge.fractal_dimension;
            total_energy += edge.quantum_correlation.magnitude();
        }
    }

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        total_energy += std.math.cos(f64, node.phase);
    }

    return total_energy;
}

pub fn connectivityObjective(graph: *const SelfSimilarRelationalGraph) f64 {
    const node_count = graph.nodeCount();
    const edge_count = graph.edgeCount();

    if (node_count == 0) return 0.0;

    const max_edges = node_count * (node_count - 1) / 2;
    const connectivity_ratio = if (max_edges > 0)
        @as(f64, @floatFromInt(edge_count)) / @as(f64, @floatFromInt(max_edges))
    else
        0.0;

    var total_weight: f64 = 0.0;
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_weight += edge.weight;
        }
    }

    const avg_weight = if (edge_count > 0) total_weight / @as(f64, @floatFromInt(edge_count)) else 0.0;

    return (1.0 - connectivity_ratio) + (1.0 - avg_weight);
}

pub fn quantumCoherenceObjective(graph: *const SelfSimilarRelationalGraph) f64 {
    var total_coherence: f64 = 0.0;
    var node_count: usize = 0;

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        total_coherence += node.quantum_state.magnitude();
        node_count += 1;
    }

    var total_correlation: f64 = 0.0;
    var edge_count: usize = 0;

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_correlation += edge.quantum_correlation.magnitude();
            edge_count += 1;
        }
    }

    const avg_coherence = if (node_count > 0) total_coherence / @as(f64, @floatFromInt(node_count)) else 0.0;
    const avg_correlation = if (edge_count > 0) total_correlation / @as(f64, @floatFromInt(edge_count)) else 0.0;

    return 1.0 - (avg_coherence + avg_correlation);
}

pub fn fractalDimensionObjective(graph: *const SelfSimilarRelationalGraph) f64 {
    var total_dimension: f64 = 0.0;
    var edge_count: usize = 0;

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_dimension += edge.fractal_dimension;
            edge_count += 1;
        }
    }

    if (edge_count == 0) return 0.0;

    const avg_dimension = total_dimension / @as(f64, @floatFromInt(edge_count));
    const target_dimension: f64 = 1.5;

    return @fabs(avg_dimension - target_dimension);
}

test "SymmetryGroup basic operations" {
    const identity = SymmetryGroup.identity;
    try std.testing.expectEqualStrings("identity", identity.toString());
    try std.testing.expectEqual(@as(usize, 1), identity.getOrder());

    const rotation = SymmetryGroup.rotation_90;
    const expected_angle: f64 = std.math.pi / 2.0;
    const actual_angle = rotation.getAngle();
    try std.testing.expectApproxEqAbs(expected_angle, actual_angle, 0.001);
}

test "SymmetryTransform apply" {
    const identity = SymmetryTransform.init(.identity);
    const result = identity.apply(1.0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.y, 0.001);

    const rotation_180 = SymmetryTransform.init(.rotation_180);
    const rotated = rotation_180.apply(1.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), rotated.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rotated.y, 0.001);
}

test "SymmetryTransform inverse" {
    const rotation_90 = SymmetryTransform.init(.rotation_90);
    const inverse = rotation_90.inverse();
    try std.testing.expectEqual(SymmetryGroup.rotation_270, inverse.group);

    const identity = SymmetryTransform.init(.identity);
    const id_inv = identity.inverse();
    try std.testing.expectEqual(SymmetryGroup.identity, id_inv.group);
}

test "SymmetryTransform compose" {
    const rot90 = SymmetryTransform.init(.rotation_90);
    const composed = rot90.compose(&rot90);
    try std.testing.expectEqual(SymmetryGroup.rotation_180, composed.group);

    const identity = SymmetryTransform.init(.identity);
    const with_id = rot90.compose(&identity);
    try std.testing.expectEqual(SymmetryGroup.rotation_90, with_id.group);
}

test "OptimizationState basic" {
    const allocator = std.testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var node = try Node.init(allocator, "test", "data", 1.0, 0.0, 0.0);
    try graph.addNode(node);

    var state = OptimizationState.init(allocator, &graph, 1.5, false);
    defer state.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 1.5), state.energy, 0.001);
    try std.testing.expectEqual(@as(usize, 0), state.iteration);
}

test "OptimizationState entanglement" {
    const allocator = std.testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "d1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "d2", 0.0, 1.0, 0.5);
    try graph.addNode(n2);

    var state = OptimizationState.init(allocator, &graph, 1.0, false);
    defer state.deinit();

    const info = EntanglementInfo.init(0.8, 0.5);
    try state.addEntanglement("n1", "n2", info);

    try std.testing.expect(state.hasEntanglement("n1", "n2"));
    try std.testing.expectEqual(@as(usize, 1), state.entangledPairsCount());

    const retrieved = state.getEntanglement("n1", "n2");
    try std.testing.expect(retrieved != null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), retrieved.?.correlation_strength, 0.001);
}

test "OptimizationStatistics" {
    var stats = OptimizationStatistics.init();

    stats.moves_accepted = 75;
    stats.moves_rejected = 25;
    stats.updateAcceptanceRate();

    try std.testing.expectApproxEqAbs(@as(f64, 0.75), stats.acceptance_rate, 0.001);
}

test "EntangledStochasticSymmetryOptimizer init" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.init(allocator, 100.0, 0.95, 1000);
    defer optimizer.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 100.0), optimizer.temperature, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.95), optimizer.cooling_rate, 0.001);
    try std.testing.expectEqual(@as(usize, 1000), optimizer.max_iterations);
}

test "EntangledStochasticSymmetryOptimizer coolTemperature" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.init(allocator, 100.0, 0.9, 1000);
    defer optimizer.deinit();

    optimizer.coolTemperature();
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), optimizer.temperature, 0.001);

    optimizer.coolTemperature();
    try std.testing.expectApproxEqAbs(@as(f64, 81.0), optimizer.temperature, 0.001);
}

test "EntangledStochasticSymmetryOptimizer acceptMove" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 100.0, 0.95, 1000, 12345);
    defer optimizer.deinit();

    try std.testing.expect(optimizer.acceptMove(-10.0));

    optimizer.temperature = 0.0001;
    try std.testing.expect(!optimizer.acceptMove(10.0));
}

test "EntangledStochasticSymmetryOptimizer detectSymmetries" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.init(allocator, 100.0, 0.95, 1000);
    defer optimizer.deinit();

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "d1", 1.0, 0.0, 0.0);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "d2", 0.0, 1.0, 0.5);
    try graph.addNode(n2);

    const transforms = try optimizer.detectSymmetries(&graph);
    defer allocator.free(transforms);

    try std.testing.expect(transforms.len >= 1);
    try std.testing.expectEqual(SymmetryGroup.identity, transforms[0].group);
}

test "EntangledStochasticSymmetryOptimizer simple optimization" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 42);
    defer optimizer.deinit();

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "data1", 0.8, 0.2, 0.1);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 0.3, 0.7, 0.3);
    try graph.addNode(n2);
    var n3 = try Node.init(allocator, "n3", "data3", 0.5, 0.5, 0.5);
    try graph.addNode(n3);

    var e1 = try Edge.init(allocator, "n1", "n2", .coherent, 0.8, 0.5, 0.5, 1.2);
    try graph.addEdge(e1);
    var e2 = try Edge.init(allocator, "n2", "n3", .entangled, 0.6, 0.3, 0.3, 1.1);
    try graph.addEdge(e2);

    _ = try optimizer.optimize(&graph, defaultGraphObjective);

    const stats = optimizer.getStatistics();
    try std.testing.expect(stats.iterations_completed > 0);
    try std.testing.expect(stats.moves_accepted + stats.moves_rejected > 0);
}

test "Objective functions" {
    const allocator = std.testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "data1", 0.8, 0.2, 0.1);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 0.3, 0.7, 0.3);
    try graph.addNode(n2);

    var e1 = try Edge.init(allocator, "n1", "n2", .coherent, 0.8, 0.5, 0.5, 1.2);
    try graph.addEdge(e1);

    const default_energy = defaultGraphObjective(&graph);
    try std.testing.expect(std.math.isFinite(default_energy));

    const connectivity_energy = connectivityObjective(&graph);
    try std.testing.expect(std.math.isFinite(connectivity_energy));

    const coherence_energy = quantumCoherenceObjective(&graph);
    try std.testing.expect(std.math.isFinite(coherence_energy));

    const fractal_energy = fractalDimensionObjective(&graph);
    try std.testing.expect(std.math.isFinite(fractal_energy));
}

test "SymmetryPattern basic" {
    const allocator = std.testing.allocator;

    var pattern = SymmetryPattern.init(allocator, SymmetryTransform.init(.rotation_90));
    defer pattern.deinit();

    try pattern.addNode("node1");
    try pattern.addNode("node2");

    try std.testing.expectEqual(@as(usize, 2), pattern.nodes.items.len);
}

test "EntanglementInfo decay" {
    const info = EntanglementInfo.init(0.8, 0.5);

    const decay_factor = info.getDecayFactor(1000);
    try std.testing.expect(decay_factor >= 0.0);
    try std.testing.expect(decay_factor <= 1.0);
}