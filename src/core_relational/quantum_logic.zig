const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Complex = std.math.Complex;

pub const LogicGate = enum(u8) {
    HADAMARD = 0,
    PAULI_X = 1,
    PAULI_Y = 2,
    PAULI_Z = 3,
    PHASE = 4,
    CNOT = 5,
    TOFFOLI = 6,
    RELATIONAL_AND = 7,
    RELATIONAL_OR = 8,
    RELATIONAL_NOT = 9,
    RELATIONAL_XOR = 10,
    FRACTAL_TRANSFORM = 11,

    pub fn toString(self: LogicGate) []const u8 {
        return switch (self) {
            .HADAMARD => "hadamard",
            .PAULI_X => "pauli_x",
            .PAULI_Y => "pauli_y",
            .PAULI_Z => "pauli_z",
            .PHASE => "phase",
            .CNOT => "cnot",
            .TOFFOLI => "toffoli",
            .RELATIONAL_AND => "relational_and",
            .RELATIONAL_OR => "relational_or",
            .RELATIONAL_NOT => "relational_not",
            .RELATIONAL_XOR => "relational_xor",
            .FRACTAL_TRANSFORM => "fractal_transform",
        };
    }

    pub fn fromString(s: []const u8) ?LogicGate {
        if (std.mem.eql(u8, s, "hadamard")) return .HADAMARD;
        if (std.mem.eql(u8, s, "pauli_x")) return .PAULI_X;
        if (std.mem.eql(u8, s, "pauli_y")) return .PAULI_Y;
        if (std.mem.eql(u8, s, "pauli_z")) return .PAULI_Z;
        if (std.mem.eql(u8, s, "phase")) return .PHASE;
        if (std.mem.eql(u8, s, "cnot")) return .CNOT;
        if (std.mem.eql(u8, s, "toffoli")) return .TOFFOLI;
        if (std.mem.eql(u8, s, "relational_and")) return .RELATIONAL_AND;
        if (std.mem.eql(u8, s, "relational_or")) return .RELATIONAL_OR;
        if (std.mem.eql(u8, s, "relational_not")) return .RELATIONAL_NOT;
        if (std.mem.eql(u8, s, "relational_xor")) return .RELATIONAL_XOR;
        if (std.mem.eql(u8, s, "fractal_transform")) return .FRACTAL_TRANSFORM;
        return null;
    }

    pub fn isSingleQubit(self: LogicGate) bool {
        return switch (self) {
            .HADAMARD, .PAULI_X, .PAULI_Y, .PAULI_Z, .PHASE, .RELATIONAL_NOT, .FRACTAL_TRANSFORM => true,
            .CNOT, .RELATIONAL_AND, .RELATIONAL_OR, .RELATIONAL_XOR => false,
            .TOFFOLI => false,
        };
    }

    pub fn requiredQubits(self: LogicGate) usize {
        return switch (self) {
            .HADAMARD, .PAULI_X, .PAULI_Y, .PAULI_Z, .PHASE, .RELATIONAL_NOT, .FRACTAL_TRANSFORM => 1,
            .CNOT, .RELATIONAL_AND, .RELATIONAL_OR, .RELATIONAL_XOR => 2,
            .TOFFOLI => 3,
        };
    }
};

pub const QuantumState = struct {
    amplitude_real: f64,
    amplitude_imag: f64,
    phase: f64,
    entanglement_degree: f64,

    const Self = @This();

    pub fn init(amp_real: f64, amp_imag: f64, phase_val: f64, entangle: f64) Self {
        return Self{
            .amplitude_real = amp_real,
            .amplitude_imag = amp_imag,
            .phase = phase_val,
            .entanglement_degree = entangle,
        };
    }

    pub fn initFromComplex(amplitude: Complex(f64), phase_val: f64, entangle: f64) Self {
        return Self{
            .amplitude_real = amplitude.re,
            .amplitude_imag = amplitude.im,
            .phase = phase_val,
            .entanglement_degree = entangle,
        };
    }

    pub fn normalize(self: *Self) void {
        const mag = self.magnitude();
        if (mag > 0.0) {
            self.amplitude_real /= mag;
            self.amplitude_imag /= mag;
        }
    }

    pub fn magnitude(self: *const Self) f64 {
        return @sqrt(self.amplitude_real * self.amplitude_real + self.amplitude_imag * self.amplitude_imag);
    }

    pub fn probability(self: *const Self) f64 {
        return self.amplitude_real * self.amplitude_real + self.amplitude_imag * self.amplitude_imag;
    }

    pub fn toComplex(self: *const Self) Complex(f64) {
        return Complex(f64).init(self.amplitude_real, self.amplitude_imag);
    }

    pub fn conjugate(self: *const Self) Self {
        return Self{
            .amplitude_real = self.amplitude_real,
            .amplitude_imag = -self.amplitude_imag,
            .phase = -self.phase,
            .entanglement_degree = self.entanglement_degree,
        };
    }

    pub fn add(self: *const Self, other: *const Self) Self {
        return Self{
            .amplitude_real = self.amplitude_real + other.amplitude_real,
            .amplitude_imag = self.amplitude_imag + other.amplitude_imag,
            .phase = (self.phase + other.phase) / 2.0,
            .entanglement_degree = @max(self.entanglement_degree, other.entanglement_degree),
        };
    }

    pub fn multiply(self: *const Self, other: *const Self) Self {
        const real = self.amplitude_real * other.amplitude_real - self.amplitude_imag * other.amplitude_imag;
        const imag = self.amplitude_real * other.amplitude_imag + self.amplitude_imag * other.amplitude_real;
        return Self{
            .amplitude_real = real,
            .amplitude_imag = imag,
            .phase = self.phase + other.phase,
            .entanglement_degree = @min(1.0, self.entanglement_degree + other.entanglement_degree),
        };
    }

    pub fn scale(self: *const Self, factor: f64) Self {
        return Self{
            .amplitude_real = self.amplitude_real * factor,
            .amplitude_imag = self.amplitude_imag * factor,
            .phase = self.phase,
            .entanglement_degree = self.entanglement_degree,
        };
    }

    pub fn clone(self: *const Self) Self {
        return Self{
            .amplitude_real = self.amplitude_real,
            .amplitude_imag = self.amplitude_imag,
            .phase = self.phase,
            .entanglement_degree = self.entanglement_degree,
        };
    }

    pub fn isNormalized(self: *const Self, epsilon: f64) bool {
        const prob = self.probability();
        return @fabs(prob - 1.0) < epsilon;
    }

    pub fn fidelity(self: *const Self, other: *const Self) f64 {
        const inner_real = self.amplitude_real * other.amplitude_real + self.amplitude_imag * other.amplitude_imag;
        const inner_imag = self.amplitude_real * other.amplitude_imag - self.amplitude_imag * other.amplitude_real;
        return inner_real * inner_real + inner_imag * inner_imag;
    }
};

pub const GateHistoryEntry = struct {
    gate: LogicGate,
    indices: ArrayList(usize),
    params: ?ArrayList(f64),
    timestamp: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, gate: LogicGate, indices: []const usize, params: ?[]const f64) !Self {
        var indices_list = ArrayList(usize).init(allocator);
        for (indices) |idx| {
            try indices_list.append(idx);
        }

        var params_list: ?ArrayList(f64) = null;
        if (params) |p| {
            params_list = ArrayList(f64).init(allocator);
            for (p) |param| {
                try params_list.?.append(param);
            }
        }

        return Self{
            .gate = gate,
            .indices = indices_list,
            .params = params_list,
            .timestamp = std.time.nanoTimestamp(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.indices.deinit();
        if (self.params) |*p| {
            p.deinit();
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var indices_copy = ArrayList(usize).init(allocator);
        for (self.indices.items) |idx| {
            try indices_copy.append(idx);
        }

        var params_copy: ?ArrayList(f64) = null;
        if (self.params) |p| {
            params_copy = ArrayList(f64).init(allocator);
            for (p.items) |param| {
                try params_copy.?.append(param);
            }
        }

        return Self{
            .gate = self.gate,
            .indices = indices_copy,
            .params = params_copy,
            .timestamp = self.timestamp,
        };
    }
};

pub const GateSequenceEntry = struct {
    gate: LogicGate,
    indices: []const usize,
    params: ?[]const f64,
};

pub const MeasurementResult = struct {
    result: i32,
    probability: f64,
    collapsed_state: QuantumState,
};

pub const RelationalQuantumLogic = struct {
    states: ArrayList(QuantumState),
    gate_history: ArrayList(GateHistoryEntry),
    allocator: Allocator,
    coherence_threshold: f64,
    max_entanglement_depth: usize,

    const Self = @This();
    const DEFAULT_COHERENCE_THRESHOLD: f64 = 1e-10;
    const DEFAULT_MAX_ENTANGLEMENT_DEPTH: usize = 64;
    const SQRT2_INV: f64 = 0.7071067811865476;
    const DEFAULT_PHASE_ANGLE: f64 = 0.7853981633974483;

    pub fn init(allocator: Allocator) Self {
        return Self{
            .states = ArrayList(QuantumState).init(allocator),
            .gate_history = ArrayList(GateHistoryEntry).init(allocator),
            .allocator = allocator,
            .coherence_threshold = DEFAULT_COHERENCE_THRESHOLD,
            .max_entanglement_depth = DEFAULT_MAX_ENTANGLEMENT_DEPTH,
        };
    }

    pub fn initWithOptions(allocator: Allocator, coherence_threshold: f64, max_entanglement_depth: usize) Self {
        return Self{
            .states = ArrayList(QuantumState).init(allocator),
            .gate_history = ArrayList(GateHistoryEntry).init(allocator),
            .allocator = allocator,
            .coherence_threshold = coherence_threshold,
            .max_entanglement_depth = max_entanglement_depth,
        };
    }

    pub fn deinit(self: *Self) void {
        self.states.deinit();
        for (self.gate_history.items) |*entry| {
            entry.deinit();
        }
        self.gate_history.deinit();
    }

    pub fn reset(self: *Self) void {
        self.states.clearRetainingCapacity();
        for (self.gate_history.items) |*entry| {
            entry.deinit();
        }
        self.gate_history.clearRetainingCapacity();
    }

    pub fn initializeState(self: *Self, amp_real: f64, amp_imag: f64, phase: f64) !usize {
        var state = QuantumState.init(amp_real, amp_imag, phase, 0.0);
        state.normalize();
        try self.states.append(state);
        return self.states.items.len - 1;
    }

    pub fn initializeStateFromComplex(self: *Self, amplitude: Complex(f64), phase: f64) !usize {
        var state = QuantumState.initFromComplex(amplitude, phase, 0.0);
        state.normalize();
        try self.states.append(state);
        return self.states.items.len - 1;
    }

    pub fn initializeBasisState(self: *Self, is_one: bool) !usize {
        const amp_real: f64 = if (is_one) 0.0 else 1.0;
        const amp_imag: f64 = 0.0;
        return self.initializeState(amp_real, amp_imag, 0.0);
    }

    pub fn getState(self: *const Self, qubit_index: usize) ?QuantumState {
        if (qubit_index >= self.states.items.len) {
            return null;
        }
        return self.states.items[qubit_index];
    }

    pub fn getStatePtr(self: *Self, qubit_index: usize) ?*QuantumState {
        if (qubit_index >= self.states.items.len) {
            return null;
        }
        return &self.states.items[qubit_index];
    }

    pub fn stateCount(self: *const Self) usize {
        return self.states.items.len;
    }

    pub fn historyCount(self: *const Self) usize {
        return self.gate_history.items.len;
    }

    pub fn applyGate(self: *Self, gate: LogicGate, qubit_indices: []const usize, params: ?[]const f64) !void {
        const required = gate.requiredQubits();
        if (qubit_indices.len < required) {
            return error.InsufficientQubits;
        }

        for (qubit_indices) |idx| {
            if (idx >= self.states.items.len) {
                return error.InvalidQubitIndex;
            }
        }

        switch (gate) {
            .HADAMARD => self.applyHadamard(qubit_indices[0]),
            .PAULI_X => self.applyPauliX(qubit_indices[0]),
            .PAULI_Y => self.applyPauliY(qubit_indices[0]),
            .PAULI_Z => self.applyPauliZ(qubit_indices[0]),
            .PHASE => {
                const theta = if (params != null and params.?.len > 0) params.?[0] else DEFAULT_PHASE_ANGLE;
                self.applyPhase(qubit_indices[0], theta);
            },
            .CNOT => self.applyCNOT(qubit_indices[0], qubit_indices[1]),
            .TOFFOLI => self.applyToffoli(qubit_indices[0], qubit_indices[1], qubit_indices[2]),
            .RELATIONAL_AND => try self.applyRelationalAnd(qubit_indices[0], qubit_indices[1]),
            .RELATIONAL_OR => try self.applyRelationalOr(qubit_indices[0], qubit_indices[1]),
            .RELATIONAL_NOT => self.applyRelationalNot(qubit_indices[0]),
            .RELATIONAL_XOR => try self.applyRelationalXor(qubit_indices[0], qubit_indices[1]),
            .FRACTAL_TRANSFORM => {
                const depth: i32 = if (params != null and params.?.len > 0) @intFromFloat(params.?[0]) else 3;
                self.applyFractalTransform(qubit_indices[0], depth);
            },
        }

        const history_entry = try GateHistoryEntry.init(self.allocator, gate, qubit_indices, params);
        try self.gate_history.append(history_entry);
    }

    fn applyHadamard(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        const new_real = (state.amplitude_real + 1.0) * SQRT2_INV;
        const new_imag = state.amplitude_imag * SQRT2_INV;
        state.amplitude_real = new_real;
        state.amplitude_imag = new_imag;
        state.normalize();
    }

    fn applyPauliX(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        const temp = state.amplitude_real;
        state.amplitude_real = state.amplitude_imag;
        state.amplitude_imag = temp;
    }

    fn applyPauliY(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        const new_real = -state.amplitude_imag;
        const new_imag = state.amplitude_real;
        state.amplitude_real = new_real;
        state.amplitude_imag = new_imag;
    }

    fn applyPauliZ(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        state.phase += std.math.pi;
        const cos_pi = @cos(std.math.pi);
        const sin_pi = @sin(std.math.pi);
        const new_real = state.amplitude_real * cos_pi - state.amplitude_imag * sin_pi;
        const new_imag = state.amplitude_real * sin_pi + state.amplitude_imag * cos_pi;
        state.amplitude_real = new_real;
        state.amplitude_imag = new_imag;
    }

    fn applyPhase(self: *Self, qubit_idx: usize, theta: f64) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        state.phase += theta;
        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);
        const new_real = state.amplitude_real * cos_theta - state.amplitude_imag * sin_theta;
        const new_imag = state.amplitude_real * sin_theta + state.amplitude_imag * cos_theta;
        state.amplitude_real = new_real;
        state.amplitude_imag = new_imag;
    }

    fn applyCNOT(self: *Self, control_idx: usize, target_idx: usize) void {
        if (control_idx >= self.states.items.len or target_idx >= self.states.items.len) return;
        const control = self.states.items[control_idx];
        var target = &self.states.items[target_idx];

        if (control.magnitude() > 0.5) {
            const temp = target.amplitude_real;
            target.amplitude_real = target.amplitude_imag;
            target.amplitude_imag = temp;
        }

        var control_ptr = &self.states.items[control_idx];
        control_ptr.entanglement_degree = @min(1.0, control_ptr.entanglement_degree + 0.5);
        target.entanglement_degree = @min(1.0, target.entanglement_degree + 0.5);
    }

    fn applyToffoli(self: *Self, control1_idx: usize, control2_idx: usize, target_idx: usize) void {
        if (control1_idx >= self.states.items.len or
            control2_idx >= self.states.items.len or
            target_idx >= self.states.items.len) return;

        const control1 = self.states.items[control1_idx];
        const control2 = self.states.items[control2_idx];
        var target = &self.states.items[target_idx];

        if (control1.magnitude() > 0.5 and control2.magnitude() > 0.5) {
            const temp = target.amplitude_real;
            target.amplitude_real = target.amplitude_imag;
            target.amplitude_imag = temp;
        }

        var c1 = &self.states.items[control1_idx];
        var c2 = &self.states.items[control2_idx];
        c1.entanglement_degree = @min(1.0, c1.entanglement_degree + 0.33);
        c2.entanglement_degree = @min(1.0, c2.entanglement_degree + 0.33);
        target.entanglement_degree = @min(1.0, target.entanglement_degree + 0.33);
    }

    fn applyRelationalAnd(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return;

        const state1 = self.states.items[idx1];
        const state2 = self.states.items[idx2];

        const result_real = state1.amplitude_real * state2.amplitude_real -
            state1.amplitude_imag * state2.amplitude_imag;
        const result_imag = state1.amplitude_real * state2.amplitude_imag +
            state1.amplitude_imag * state2.amplitude_real;

        var result_state = QuantumState.init(
            result_real,
            result_imag,
            (state1.phase + state2.phase) / 2.0,
            @min(1.0, state1.entanglement_degree + state2.entanglement_degree),
        );
        result_state.normalize();
        try self.states.append(result_state);
    }

    fn applyRelationalOr(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return;

        const state1 = self.states.items[idx1];
        const state2 = self.states.items[idx2];

        const result_real = (state1.amplitude_real + state2.amplitude_real) * SQRT2_INV;
        const result_imag = (state1.amplitude_imag + state2.amplitude_imag) * SQRT2_INV;

        var result_state = QuantumState.init(
            result_real,
            result_imag,
            (state1.phase + state2.phase) / 2.0,
            @max(state1.entanglement_degree, state2.entanglement_degree),
        );
        result_state.normalize();
        try self.states.append(result_state);
    }

    fn applyRelationalNot(self: *Self, idx: usize) void {
        if (idx >= self.states.items.len) return;
        var state = &self.states.items[idx];
        state.amplitude_real = 1.0 - state.amplitude_real;
        state.amplitude_imag = -state.amplitude_imag;
        state.phase += std.math.pi;
        state.normalize();
    }

    fn applyRelationalXor(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return;

        const state1 = self.states.items[idx1];
        const state2 = self.states.items[idx2];

        const result_real = (state1.amplitude_real - state2.amplitude_real) * SQRT2_INV;
        const result_imag = (state1.amplitude_imag - state2.amplitude_imag) * SQRT2_INV;

        var result_state = QuantumState.init(
            result_real,
            result_imag,
            @fabs(state1.phase - state2.phase),
            (state1.entanglement_degree + state2.entanglement_degree) / 2.0,
        );
        result_state.normalize();
        try self.states.append(result_state);
    }

    fn applyFractalTransform(self: *Self, idx: usize, depth: i32) void {
        if (idx >= self.states.items.len) return;
        var state = &self.states.items[idx];

        var i: i32 = 0;
        while (i < depth) : (i += 1) {
            const scale_factor = 1.0 / std.math.pow(f64, 2.0, @as(f64, @floatFromInt(i)));
            const angle = state.phase * scale_factor;
            const cos_angle = @cos(angle);
            const sin_angle = @sin(angle);

            const normalization = @sqrt(1.0 + scale_factor * scale_factor);
            const new_real = (state.amplitude_real + cos_angle * scale_factor) / normalization;
            const new_imag = (state.amplitude_imag + sin_angle * scale_factor) / normalization;

            state.amplitude_real = new_real;
            state.amplitude_imag = new_imag;
            state.phase = std.math.atan2(f64, state.amplitude_imag, state.amplitude_real);
        }
        state.normalize();
    }

    pub fn measure(self: *Self, qubit_idx: usize) MeasurementResult {
        if (qubit_idx >= self.states.items.len) {
            return MeasurementResult{
                .result = 0,
                .probability = 0.0,
                .collapsed_state = QuantumState.init(0.0, 0.0, 0.0, 0.0),
            };
        }

        var state = &self.states.items[qubit_idx];
        const prob = state.probability();
        const result: i32 = if (prob > 0.5) 1 else 0;

        state.amplitude_real = @as(f64, @floatFromInt(result));
        state.amplitude_imag = 0.0;
        state.entanglement_degree = 0.0;

        return MeasurementResult{
            .result = result,
            .probability = prob,
            .collapsed_state = state.clone(),
        };
    }

    pub fn measureWithRandomness(self: *Self, qubit_idx: usize, random_value: f64) MeasurementResult {
        if (qubit_idx >= self.states.items.len) {
            return MeasurementResult{
                .result = 0,
                .probability = 0.0,
                .collapsed_state = QuantumState.init(0.0, 0.0, 0.0, 0.0),
            };
        }

        var state = &self.states.items[qubit_idx];
        const prob = state.probability();
        const result: i32 = if (random_value < prob) 1 else 0;

        state.amplitude_real = @as(f64, @floatFromInt(result));
        state.amplitude_imag = 0.0;
        state.entanglement_degree = 0.0;

        return MeasurementResult{
            .result = result,
            .probability = prob,
            .collapsed_state = state.clone(),
        };
    }

    pub fn entangle(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return;

        var state1 = &self.states.items[idx1];
        var state2 = &self.states.items[idx2];

        const entangled_real = (state1.amplitude_real + state2.amplitude_real) * SQRT2_INV;
        const entangled_imag = (state1.amplitude_imag + state2.amplitude_imag) * SQRT2_INV;

        state1.amplitude_real = entangled_real;
        state1.amplitude_imag = entangled_imag;
        state2.amplitude_real = entangled_real;
        state2.amplitude_imag = entangled_imag;

        state1.entanglement_degree = 1.0;
        state2.entanglement_degree = 1.0;

        state1.normalize();
        state2.normalize();
    }

    pub fn computeRelationalOutput(
        self: *Self,
        input_indices: []const usize,
        gate_sequence: []const GateSequenceEntry,
    ) !ArrayList(Complex(f64)) {
        for (gate_sequence) |entry| {
            try self.applyGate(entry.gate, entry.indices, entry.params);
        }

        var result = ArrayList(Complex(f64)).init(self.allocator);
        for (input_indices) |idx| {
            if (idx < self.states.items.len) {
                const state = self.states.items[idx];
                try result.append(Complex(f64).init(state.amplitude_real, state.amplitude_imag));
            }
        }
        return result;
    }

    pub fn computeRelationalOutputRaw(
        self: *Self,
        input_indices: []const usize,
    ) !ArrayList(Complex(f64)) {
        var result = ArrayList(Complex(f64)).init(self.allocator);
        for (input_indices) |idx| {
            if (idx < self.states.items.len) {
                const state = self.states.items[idx];
                try result.append(Complex(f64).init(state.amplitude_real, state.amplitude_imag));
            }
        }
        return result;
    }

    pub fn getTotalProbability(self: *const Self) f64 {
        var total: f64 = 0.0;
        for (self.states.items) |state| {
            total += state.probability();
        }
        return total;
    }

    pub fn getAverageEntanglement(self: *const Self) f64 {
        if (self.states.items.len == 0) return 0.0;
        var total: f64 = 0.0;
        for (self.states.items) |state| {
            total += state.entanglement_degree;
        }
        return total / @as(f64, @floatFromInt(self.states.items.len));
    }

    pub fn isCoherent(self: *const Self) bool {
        for (self.states.items) |state| {
            if (state.probability() < self.coherence_threshold) {
                return false;
            }
        }
        return true;
    }

    pub fn cloneStates(self: *const Self, allocator: Allocator) !ArrayList(QuantumState) {
        var cloned = ArrayList(QuantumState).init(allocator);
        for (self.states.items) |state| {
            try cloned.append(state.clone());
        }
        return cloned;
    }

    pub fn applyControlledGate(
        self: *Self,
        control_idx: usize,
        target_idx: usize,
        gate: LogicGate,
        params: ?[]const f64,
    ) !void {
        if (control_idx >= self.states.items.len or target_idx >= self.states.items.len) {
            return error.InvalidQubitIndex;
        }

        const control = self.states.items[control_idx];
        if (control.magnitude() > 0.5) {
            switch (gate) {
                .HADAMARD => self.applyHadamard(target_idx),
                .PAULI_X => self.applyPauliX(target_idx),
                .PAULI_Y => self.applyPauliY(target_idx),
                .PAULI_Z => self.applyPauliZ(target_idx),
                .PHASE => {
                    const theta = if (params != null and params.?.len > 0) params.?[0] else DEFAULT_PHASE_ANGLE;
                    self.applyPhase(target_idx, theta);
                },
                .RELATIONAL_NOT => self.applyRelationalNot(target_idx),
                .FRACTAL_TRANSFORM => {
                    const depth: i32 = if (params != null and params.?.len > 0) @intFromFloat(params.?[0]) else 3;
                    self.applyFractalTransform(target_idx, depth);
                },
                else => {},
            }
        }

        var control_ptr = &self.states.items[control_idx];
        control_ptr.entanglement_degree = @min(1.0, control_ptr.entanglement_degree + 0.25);
        var target = &self.states.items[target_idx];
        target.entanglement_degree = @min(1.0, target.entanglement_degree + 0.25);
    }

    pub fn serialize(self: *const Self, allocator: Allocator) !ArrayList(u8) {
        var buffer = ArrayList(u8).init(allocator);

        var state_count_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &state_count_buf, @as(u64, self.states.items.len), .Little);
        try buffer.appendSlice(&state_count_buf);

        for (self.states.items) |state| {
            var real_buf: [8]u8 = undefined;
            var imag_buf: [8]u8 = undefined;
            var phase_buf: [8]u8 = undefined;
            var entangle_buf: [8]u8 = undefined;

            @memcpy(&real_buf, std.mem.asBytes(&state.amplitude_real));
            @memcpy(&imag_buf, std.mem.asBytes(&state.amplitude_imag));
            @memcpy(&phase_buf, std.mem.asBytes(&state.phase));
            @memcpy(&entangle_buf, std.mem.asBytes(&state.entanglement_degree));

            try buffer.appendSlice(&real_buf);
            try buffer.appendSlice(&imag_buf);
            try buffer.appendSlice(&phase_buf);
            try buffer.appendSlice(&entangle_buf);
        }

        return buffer;
    }

    pub fn deserialize(allocator: Allocator, data: []const u8) !Self {
        if (data.len < 8) return error.InvalidData;

        var self = Self.init(allocator);

        const state_count = std.mem.readInt(u64, data[0..8], .Little);
        var offset: usize = 8;

        var i: u64 = 0;
        while (i < state_count) : (i += 1) {
            if (offset + 32 > data.len) return error.InvalidData;

            const real_val = @as(*const f64, @ptrCast(@alignCast(data[offset..].ptr))).*;
            const imag_val = @as(*const f64, @ptrCast(@alignCast(data[offset + 8 ..].ptr))).*;
            const phase_val = @as(*const f64, @ptrCast(@alignCast(data[offset + 16 ..].ptr))).*;
            const entangle_val = @as(*const f64, @ptrCast(@alignCast(data[offset + 24 ..].ptr))).*;

            try self.states.append(QuantumState.init(real_val, imag_val, phase_val, entangle_val));
            offset += 32;
        }

        return self;
    }
};

pub const QuantumCircuit = struct {
    gates: ArrayList(GateSequenceEntry),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .gates = ArrayList(GateSequenceEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.gates.items) |entry| {
            self.allocator.free(entry.indices);
            if (entry.params) |p| {
                self.allocator.free(p);
            }
        }
        self.gates.deinit();
    }

    pub fn addGate(self: *Self, gate: LogicGate, indices: []const usize, params: ?[]const f64) !void {
        const indices_copy = try self.allocator.dupe(usize, indices);
        errdefer self.allocator.free(indices_copy);

        var params_copy: ?[]const f64 = null;
        if (params) |p| {
            params_copy = try self.allocator.dupe(f64, p);
        }

        try self.gates.append(.{
            .gate = gate,
            .indices = indices_copy,
            .params = params_copy,
        });
    }

    pub fn execute(self: *const Self, logic: *RelationalQuantumLogic) !void {
        for (self.gates.items) |entry| {
            try logic.applyGate(entry.gate, entry.indices, entry.params);
        }
    }

    pub fn gateCount(self: *const Self) usize {
        return self.gates.items.len;
    }
};

test "quantum_state_normalize" {
    var state = QuantumState.init(3.0, 4.0, 0.0, 0.0);
    state.normalize();
    const expected_prob: f64 = 1.0;
    const actual_prob = state.probability();
    try std.testing.expectApproxEqAbs(expected_prob, actual_prob, 0.0001);
}

test "quantum_state_probability" {
    var state = QuantumState.init(0.6, 0.8, 0.0, 0.0);
    const prob = state.probability();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), prob, 0.0001);
}

test "relational_quantum_logic_init_deinit" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    const idx = try logic.initializeState(1.0, 0.0, 0.0);
    try std.testing.expectEqual(@as(usize, 0), idx);
    try std.testing.expectEqual(@as(usize, 1), logic.stateCount());
}

test "apply_hadamard_gate" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(1.0, 0.0, 0.0);
    try logic.applyGate(.HADAMARD, &[_]usize{0}, null);

    const state = logic.getState(0).?;
    try std.testing.expect(state.probability() > 0.0);
}

test "apply_cnot_gate" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(1.0, 0.0, 0.0);
    _ = try logic.initializeState(0.0, 1.0, 0.0);

    try logic.applyGate(.CNOT, &[_]usize{ 0, 1 }, null);

    const control = logic.getState(0).?;
    const target = logic.getState(1).?;
    try std.testing.expect(control.entanglement_degree > 0.0);
    try std.testing.expect(target.entanglement_degree > 0.0);
}

test "measure_qubit" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(0.8, 0.0, 0.0);

    const result = logic.measure(0);
    try std.testing.expect(result.result == 0 or result.result == 1);
    try std.testing.expect(result.probability >= 0.0 and result.probability <= 1.0);
}

test "entangle_qubits" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(1.0, 0.0, 0.0);
    _ = try logic.initializeState(0.0, 1.0, 0.0);

    try logic.entangle(0, 1);

    const state1 = logic.getState(0).?;
    const state2 = logic.getState(1).?;
    try std.testing.expectApproxEqAbs(state1.amplitude_real, state2.amplitude_real, 0.0001);
    try std.testing.expectApproxEqAbs(state1.amplitude_imag, state2.amplitude_imag, 0.0001);
    try std.testing.expectApproxEqAbs(state1.entanglement_degree, 1.0, 0.0001);
}

test "relational_and_gate" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(0.8, 0.0, 0.0);
    _ = try logic.initializeState(0.6, 0.0, 0.0);

    try logic.applyGate(.RELATIONAL_AND, &[_]usize{ 0, 1 }, null);

    try std.testing.expectEqual(@as(usize, 3), logic.stateCount());
    const result_state = logic.getState(2).?;
    try std.testing.expect(result_state.probability() > 0.0);
}

test "fractal_transform" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(0.707, 0.707, std.math.pi / 4.0);

    const params = [_]f64{3.0};
    try logic.applyGate(.FRACTAL_TRANSFORM, &[_]usize{0}, &params);

    const state = logic.getState(0).?;
    try std.testing.expect(state.isNormalized(0.01));
}

test "compute_relational_output" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(1.0, 0.0, 0.0);
    _ = try logic.initializeState(0.0, 1.0, 0.0);

    const gate_sequence = [_]GateSequenceEntry{
        .{ .gate = .HADAMARD, .indices = &[_]usize{0}, .params = null },
        .{ .gate = .CNOT, .indices = &[_]usize{ 0, 1 }, .params = null },
    };

    var result = try logic.computeRelationalOutput(&[_]usize{ 0, 1 }, &gate_sequence);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.items.len);
}

test "logic_gate_enum" {
    try std.testing.expectEqual(@as(usize, 1), LogicGate.HADAMARD.requiredQubits());
    try std.testing.expectEqual(@as(usize, 2), LogicGate.CNOT.requiredQubits());
    try std.testing.expectEqual(@as(usize, 3), LogicGate.TOFFOLI.requiredQubits());
    try std.testing.expect(LogicGate.HADAMARD.isSingleQubit());
    try std.testing.expect(!LogicGate.CNOT.isSingleQubit());
}

test "gate_history_tracking" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(1.0, 0.0, 0.0);
    try logic.applyGate(.HADAMARD, &[_]usize{0}, null);
    try logic.applyGate(.PAULI_X, &[_]usize{0}, null);

    try std.testing.expectEqual(@as(usize, 2), logic.historyCount());
}
