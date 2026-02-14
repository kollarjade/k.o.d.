const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;

const nsir = @import("nsir_core.zig");
const esso = @import("esso_optimizer.zig");
const chaos = @import("chaos_core.zig");
const fnds = @import("fnds.zig");

const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
const Node = nsir.Node;
const Edge = nsir.Edge;
const EntangledStochasticSymmetryOptimizer = esso.EntangledStochasticSymmetryOptimizer;
const ChaosCoreKernel = chaos.ChaosCoreKernel;
const FractalTree = fnds.FractalTree;
const SymmetryPattern = esso.SymmetryPattern;

pub const ThoughtLevel = enum(u8) {
    local = 0,
    global = 1,
    meta = 2,

    pub fn toString(self: ThoughtLevel) []const u8 {
        return switch (self) {
            .local => "local",
            .global => "global",
            .meta => "meta",
        };
    }
};

pub const ReasoningPhase = struct {
    phase_id: u64,
    level: ThoughtLevel,
    inner_iterations: usize,
    outer_iterations: usize,
    target_energy: f64,
    current_energy: f64,
    convergence_threshold: f64,
    phase_start_time: i64,
    phase_end_time: i64,
    pattern_captures: ArrayList([32]u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, level: ThoughtLevel, inner: usize, outer: usize) Self {
        return Self{
            .phase_id = @as(u64, @bitCast(std.time.nanoTimestamp())),
            .level = level,
            .inner_iterations = inner,
            .outer_iterations = outer,
            .target_energy = 0.1,
            .current_energy = 1e6,
            .convergence_threshold = 1e-6,
            .phase_start_time = std.time.nanoTimestamp(),
            .phase_end_time = 0,
            .pattern_captures = ArrayList([32]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.pattern_captures.deinit();
    }

    pub fn recordPattern(self: *Self, pattern_id: [32]u8) !void {
        try self.pattern_captures.append(pattern_id);
    }

    pub fn hasConverged(self: *const Self) bool {
        return @fabs(self.current_energy - self.target_energy) < self.convergence_threshold;
    }

    pub fn finalize(self: *Self) void {
        self.phase_end_time = std.time.nanoTimestamp();
    }

    pub fn getDuration(self: *const Self) i64 {
        if (self.phase_end_time > 0) {
            return self.phase_end_time - self.phase_start_time;
        }
        return std.time.nanoTimestamp() - self.phase_start_time;
    }
};

pub const OrchestratorStatistics = struct {
    total_phases: usize,
    local_phases: usize,
    global_phases: usize,
    meta_phases: usize,
    total_inner_loops: usize,
    total_outer_loops: usize,
    average_convergence_time: f64,
    best_energy_achieved: f64,
    patterns_discovered: usize,
    orchestration_start_time: i64,

    pub fn init() OrchestratorStatistics {
        return OrchestratorStatistics{
            .total_phases = 0,
            .local_phases = 0,
            .global_phases = 0,
            .meta_phases = 0,
            .total_inner_loops = 0,
            .total_outer_loops = 0,
            .average_convergence_time = 0.0,
            .best_energy_achieved = std.math.inf(f64),
            .patterns_discovered = 0,
            .orchestration_start_time = std.time.nanoTimestamp(),
        };
    }

    pub fn recordPhase(self: *OrchestratorStatistics, phase: *const ReasoningPhase) void {
        self.total_phases += 1;
        switch (phase.level) {
            .local => self.local_phases += 1,
            .global => self.global_phases += 1,
            .meta => self.meta_phases += 1,
        }
        self.total_inner_loops += phase.inner_iterations;
        self.total_outer_loops += phase.outer_iterations;
        if (phase.current_energy < self.best_energy_achieved) {
            self.best_energy_achieved = phase.current_energy;
        }
        self.patterns_discovered += phase.pattern_captures.items.len;

        const duration = @as(f64, @floatFromInt(phase.getDuration()));
        const prev_total = self.average_convergence_time * @as(f64, @floatFromInt(self.total_phases - 1));
        self.average_convergence_time = (prev_total + duration) / @as(f64, @floatFromInt(self.total_phases));
    }
};

pub const ReasoningOrchestrator = struct {
    graph: *SelfSimilarRelationalGraph,
    esso: *EntangledStochasticSymmetryOptimizer,
    chaos_kernel: *ChaosCoreKernel,
    active_phase: ?ReasoningPhase,
    phase_history: ArrayList(ReasoningPhase),
    statistics: OrchestratorStatistics,
    fast_inner_steps: usize,
    slow_outer_steps: usize,
    hierarchical_depth: usize,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        graph: *SelfSimilarRelationalGraph,
        esso_opt: *EntangledStochasticSymmetryOptimizer,
        kernel: *ChaosCoreKernel,
    ) Self {
        return Self{
            .graph = graph,
            .esso = esso_opt,
            .chaos_kernel = kernel,
            .active_phase = null,
            .phase_history = ArrayList(ReasoningPhase).init(allocator),
            .statistics = OrchestratorStatistics.init(),
            .fast_inner_steps = 50,
            .slow_outer_steps = 10,
            .hierarchical_depth = 3,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.active_phase) |*phase| {
            phase.deinit();
        }
        for (self.phase_history.items) |*phase| {
            phase.deinit();
        }
        self.phase_history.deinit();
    }

    pub fn setParameters(self: *Self, inner: usize, outer: usize, depth: usize) void {
        self.fast_inner_steps = inner;
        self.slow_outer_steps = outer;
        self.hierarchical_depth = depth;
    }

    pub fn executeLocalPhase(self: *Self) !f64 {
        var phase = ReasoningPhase.init(self.allocator, .local, self.fast_inner_steps, 1);
        errdefer phase.deinit();

        var iteration: usize = 0;
        while (iteration < self.fast_inner_steps) : (iteration += 1) {
            try self.perturbLocalNodes();
            try self.updateLocalEdges();

            const energy = self.computeGraphEnergy();
            phase.current_energy = energy;

            if (phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;
        self.statistics.recordPhase(&phase);
        try self.phase_history.append(phase);

        return final_energy;
    }

    fn perturbLocalNodes(self: *Self) !void {
        var node_iter = self.graph.nodes.iterator();
        var count: usize = 0;
        while (node_iter.next()) |entry| {
            if (count >= 10) break;
            var node = entry.value_ptr;
            const perturbation = (self.esso.prng.random().float(f64) - 0.5) * 0.1;
            node.phase += perturbation;
            const magnitude = node.quantum_state.magnitude();
            if (magnitude > 0.0) {
                const new_real = node.quantum_state.re + perturbation * 0.01;
                const new_imag = node.quantum_state.im + perturbation * 0.01;
                const new_mag = @sqrt(new_real * new_real + new_imag * new_imag);
                if (new_mag > 0.0) {
                    node.quantum_state.re = new_real / new_mag * magnitude;
                    node.quantum_state.im = new_imag / new_mag * magnitude;
                }
            }
            count += 1;
        }
    }

    fn updateLocalEdges(self: *Self) !void {
        var edge_iter = self.graph.edges.iterator();
        var count: usize = 0;
        while (edge_iter.next()) |entry| {
            if (count >= 10) break;
            for (entry.value_ptr.items) |*edge| {
                const delta = (self.esso.prng.random().float(f64) - 0.5) * 0.05;
                edge.weight = @max(0.0, @min(1.0, edge.weight + delta));
                const corr_delta = delta * 0.1;
                edge.quantum_correlation.re += corr_delta;
                edge.quantum_correlation.im += corr_delta * 0.5;
            }
            count += 1;
        }
    }

    pub fn executeGlobalPhase(self: *Self) !f64 {
        var phase = ReasoningPhase.init(self.allocator, .global, self.fast_inner_steps, self.slow_outer_steps);
        errdefer phase.deinit();

        var outer_iteration: usize = 0;
        while (outer_iteration < self.slow_outer_steps) : (outer_iteration += 1) {
            try self.transformSymmetryPatterns();
            try self.rebalanceFractalStructures();

            var inner: usize = 0;
            while (inner < self.fast_inner_steps) : (inner += 1) {
                try self.chaos_kernel.executeCycle();
            }

            const energy = self.computeGraphEnergy();
            phase.current_energy = energy;

            if (phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;
        self.statistics.recordPhase(&phase);
        try self.phase_history.append(phase);

        return final_energy;
    }

    fn transformSymmetryPatterns(self: *Self) !void {
        const transforms = try self.esso.detectSymmetries(self.graph);
        defer self.allocator.free(transforms);

        for (transforms) |transform| {
            var node_iter = self.graph.nodes.iterator();
            var count: usize = 0;
            while (node_iter.next()) |entry| {
                if (count >= 5) break;
                const node = entry.value_ptr;
                const quantum_state = esso.QuantumState.init(
                    node.quantum_state.re,
                    node.quantum_state.im,
                    node.phase,
                    0.0,
                );
                const transformed = transform.applyToQuantumState(&quantum_state);
                node.quantum_state.re = transformed.amplitude_real;
                node.quantum_state.im = transformed.amplitude_imag;
                node.phase = transformed.phase;
                count += 1;
            }
        }
    }

    fn rebalanceFractalStructures(self: *Self) !void {
        var edge_iter = self.graph.edges.iterator();
        var total_dimension: f64 = 0.0;
        var edge_count: usize = 0;

        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_dimension += edge.fractal_dimension;
                edge_count += 1;
            }
        }

        if (edge_count > 0) {
            const avg_dimension = total_dimension / @as(f64, @floatFromInt(edge_count));
            edge_iter = self.graph.edges.iterator();
            while (edge_iter.next()) |entry| {
                for (entry.value_ptr.items) |*edge| {
                    const adjustment = (avg_dimension - edge.fractal_dimension) * 0.1;
                    edge.fractal_dimension += adjustment;
                    edge.fractal_dimension = @max(1.0, @min(3.0, edge.fractal_dimension));
                }
            }
        }
    }

    pub fn executeMetaPhase(self: *Self) !f64 {
        var phase = ReasoningPhase.init(self.allocator, .meta, self.fast_inner_steps, self.slow_outer_steps);
        errdefer phase.deinit();

        var depth: usize = 0;
        while (depth < self.hierarchical_depth) : (depth += 1) {
            const local_energy = try self.executeLocalPhase();
            const global_energy = try self.executeGlobalPhase();

            phase.current_energy = (local_energy + global_energy) / 2.0;

            if (phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;
        self.statistics.recordPhase(&phase);
        try self.phase_history.append(phase);

        return final_energy;
    }

    fn computeGraphEnergy(self: *Self) f64 {
        var total_energy: f64 = 0.0;
        var count: usize = 0;

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_energy += edge.weight * edge.fractal_dimension;
                total_energy += edge.quantum_correlation.magnitude();
                count += 1;
            }
        }

        var node_iter = self.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            total_energy += @cos(node.phase);
            count += 1;
        }

        if (count > 0) {
            return total_energy / @as(f64, @floatFromInt(count));
        }
        return 1e6;
    }

    pub fn runHierarchicalReasoning(self: *Self, max_cycles: usize) !f64 {
        var cycle: usize = 0;
        var best_energy: f64 = std.math.inf(f64);

        while (cycle < max_cycles) : (cycle += 1) {
            const local_e = try self.executeLocalPhase();
            const global_e = try self.executeGlobalPhase();
            const meta_e = try self.executeMetaPhase();

            const combined = (local_e + global_e + meta_e) / 3.0;
            if (combined < best_energy) {
                best_energy = combined;
            }

            if (combined < 0.01) {
                break;
            }
        }

        return best_energy;
    }

    pub fn getStatistics(self: *const Self) OrchestratorStatistics {
        return self.statistics;
    }

    pub fn getPhaseHistory(self: *const Self) []const ReasoningPhase {
        return self.phase_history.items;
    }
};

test "reasoning_orchestrator_local_phase" {
    const allocator = std.testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "data1", 0.5, 0.5, 0.1);
    try graph.addNode(n1);
    var n2 = try Node.init(allocator, "n2", "data2", 0.3, 0.7, 0.2);
    try graph.addNode(n2);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 100, 12345);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    const energy = try orchestrator.executeLocalPhase();
    try std.testing.expect(std.math.isFinite(energy));
}

test "reasoning_orchestrator_global_phase" {
    const allocator = std.testing.allocator;

    var graph = SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var n1 = try Node.init(allocator, "n1", "data1", 0.5, 0.5, 0.1);
    try graph.addNode(n1);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 42);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    const energy = try orchestrator.executeGlobalPhase();
    try std.testing.expect(std.math.isFinite(energy));
}
