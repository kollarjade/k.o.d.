const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

fn magSq(c: Complex(f64)) f64 {
    return c.re * c.re + c.im * c.im;
}

fn freeOwnedSlice(alloc: Allocator, s: []const u8) void {
    alloc.free(@constCast(s));
}

fn writeU64Le(buf: *[8]u8, v: u64) void {
    buf[0] = @as(u8, @truncate(v));
    buf[1] = @as(u8, @truncate(v >> 8));
    buf[2] = @as(u8, @truncate(v >> 16));
    buf[3] = @as(u8, @truncate(v >> 24));
    buf[4] = @as(u8, @truncate(v >> 32));
    buf[5] = @as(u8, @truncate(v >> 40));
    buf[6] = @as(u8, @truncate(v >> 48));
    buf[7] = @as(u8, @truncate(v >> 56));
}

fn asciiEqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        if (std.ascii.toLower(a[i]) != std.ascii.toLower(b[i])) return false;
    }
    return true;
}

pub const EdgeQuality = enum(u8) {
    superposition = 0,
    entangled = 1,
    coherent = 2,
    collapsed = 3,
    fractal = 4,

    pub fn toString(self: EdgeQuality) []const u8 {
        return switch (self) {
            .superposition => "superposition",
            .entangled => "entangled",
            .coherent => "coherent",
            .collapsed => "collapsed",
            .fractal => "fractal",
        };
    }

    pub fn toInt(self: EdgeQuality) u8 {
        return @intFromEnum(self);
    }

    pub fn fromString(s: []const u8) ?EdgeQuality {
        const trimmed = std.mem.trim(u8, s, " 	
");
        if (asciiEqlIgnoreCase(trimmed, "superposition")) return .superposition;
        if (asciiEqlIgnoreCase(trimmed, "entangled")) return .entangled;
        if (asciiEqlIgnoreCase(trimmed, "coherent")) return .coherent;
        if (asciiEqlIgnoreCase(trimmed, "collapsed")) return .collapsed;
        if (asciiEqlIgnoreCase(trimmed, "fractal")) return .fractal;
        return null;
    }
};

fn dupeBytes(allocator: Allocator, b: []const u8) ![]u8 {
    return allocator.dupe(u8, b);
}

fn isValidFloat(v: f64) bool {
    return !std.math.isNan(v) and !std.math.isInf(v);
}

fn canonicalizeFloat(v: f64) u64 {
    if (std.math.isNan(v)) return 0x7FF8000000000000;
    if (v == 0.0) return 0;
    return @as(u64, @bitCast(v));
}

fn freeMapStringBytes(map: *StringHashMap([]u8), allocator: Allocator) void {
    var it = map.iterator();
    while (it.next()) |e| {
        allocator.free(e.value_ptr.*);
        freeOwnedSlice(allocator, e.key_ptr.*);
    }
    map.deinit();
}

fn putOwnedStringBytes(map: *StringHashMap([]u8), allocator: Allocator, key: []const u8, value: []const u8) !void {
    if (map.fetchRemove(key)) |old| {
        const k = allocator.dupe(u8, key) catch |err| {
            map.putAssumeCapacity(old.key, old.value);
            return err;
        };
        const v = allocator.dupe(u8, value) catch |err| {
            allocator.free(k);
            map.putAssumeCapacity(old.key, old.value);
            return err;
        };
        map.putAssumeCapacity(k, v);
        allocator.free(old.value);
        freeOwnedSlice(allocator, old.key);
    } else {
        const k = try allocator.dupe(u8, key);
        errdefer allocator.free(k);
        const v = try allocator.dupe(u8, value);
        errdefer allocator.free(v);
        try map.put(k, v);
    }
}

fn copyMetadataMap(dest: *StringHashMap([]u8), src: *const StringHashMap([]u8), allocator: Allocator) !void {
    try dest.ensureUnusedCapacity(@intCast(src.count()));
    var it = src.iterator();
    while (it.next()) |e| {
        const k = try dupeBytes(allocator, e.key_ptr.*);
        errdefer allocator.free(k);
        const v = try dupeBytes(allocator, e.value_ptr.*);
        errdefer allocator.free(v);
        try dest.put(k, v);
    }
}

pub const Qubit = struct {
    a: Complex(f64),
    b: Complex(f64),

    pub fn init(a_in: Complex(f64), b_in: Complex(f64)) Qubit {
        var q = Qubit{ .a = a_in, .b = b_in };
        if (!isValidFloat(q.a.re) or !isValidFloat(q.a.im) or
            !isValidFloat(q.b.re) or !isValidFloat(q.b.im))
        {
            return Qubit.initBasis0();
        }
        q.normalizeInPlace();
        return q;
    }

    pub fn initBasis0() Qubit {
        return Qubit{ .a = Complex(f64).init(1.0, 0.0), .b = Complex(f64).init(0.0, 0.0) };
    }

    pub fn initBasis1() Qubit {
        return Qubit{ .a = Complex(f64).init(0.0, 0.0), .b = Complex(f64).init(1.0, 0.0) };
    }

    pub fn normSquared(self: Qubit) f64 {
        return magSq(self.a) + magSq(self.b);
    }

    pub fn normalizeInPlace(self: *Qubit) void {
        const ns = self.normSquared();
        if (!(ns > 1e-30) or std.math.isNan(ns) or std.math.isInf(ns)) {
            self.* = Qubit.initBasis0();
            return;
        }
        const inv = 1.0 / std.math.sqrt(ns);
        if (!isValidFloat(inv)) {
            self.* = Qubit.initBasis0();
            return;
        }
        const s = Complex(f64).init(inv, 0.0);
        self.a = self.a.mul(s);
        self.b = self.b.mul(s);
        if (!isValidFloat(self.a.re) or !isValidFloat(self.a.im) or
            !isValidFloat(self.b.re) or !isValidFloat(self.b.im))
        {
            self.* = Qubit.initBasis0();
        }
    }

    pub fn prob0(self: Qubit) f64 {
        const p = magSq(self.a);
        if (std.math.isNan(p)) return 0.5;
        return std.math.clamp(p, 0.0, 1.0);
    }

    pub fn prob1(self: Qubit) f64 {
        const p = magSq(self.b);
        if (std.math.isNan(p)) return 0.5;
        return std.math.clamp(p, 0.0, 1.0);
    }
};

pub const Node = struct {
    id: []u8,
    data: []u8,
    qubit: Qubit,
    phase: f64,
    metadata: StringHashMap([]u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, data: []const u8, qubit: Qubit, phase: f64) !Node {
        const duped_id = try dupeBytes(allocator, id);
        errdefer allocator.free(duped_id);
        const duped_data = try dupeBytes(allocator, data);
        var validated_phase = phase;
        if (!isValidFloat(validated_phase)) validated_phase = 0.0;
        return Node{
            .id = duped_id,
            .data = duped_data,
            .qubit = qubit,
            .phase = validated_phase,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Node) void {
        freeMapStringBytes(&self.metadata, self.allocator);
        self.allocator.free(self.data);
        self.allocator.free(self.id);
    }

    pub fn clone(self: *const Node, allocator: Allocator) !Node {
        const new_id = try dupeBytes(allocator, self.id);
        errdefer allocator.free(new_id);
        const new_data = try dupeBytes(allocator, self.data);
        errdefer allocator.free(new_data);

        var new_meta = StringHashMap([]u8).init(allocator);
        errdefer freeMapStringBytes(&new_meta, allocator);

        try copyMetadataMap(&new_meta, &self.metadata, allocator);

        return Node{
            .id = new_id,
            .data = new_data,
            .qubit = self.qubit,
            .phase = self.phase,
            .metadata = new_meta,
            .allocator = allocator,
        };
    }

    pub fn setMetadata(self: *Node, key: []const u8, value: []const u8) !void {
        try putOwnedStringBytes(&self.metadata, self.allocator, key, value);
    }

    pub fn getMetadata(self: *const Node, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }
};

pub const Edge = struct {
    source: []u8,
    target: []u8,
    quality: EdgeQuality,
    weight: f64,
    quantum_correlation: Complex(f64),
    fractal_dimension: f64,
    metadata: StringHashMap([]u8),
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        source: []const u8,
        target: []const u8,
        quality: EdgeQuality,
        weight: f64,
        quantum_correlation: Complex(f64),
        fractal_dimension: f64,
    ) !Edge {
        var validated_weight = weight;
        if (!isValidFloat(validated_weight)) validated_weight = 0.0;
        var validated_fd = fractal_dimension;
        if (!isValidFloat(validated_fd)) validated_fd = 0.0;

        const duped_s = try dupeBytes(allocator, source);
        errdefer allocator.free(duped_s);
        const duped_t = try dupeBytes(allocator, target);

        return Edge{
            .source = duped_s,
            .target = duped_t,
            .quality = quality,
            .weight = validated_weight,
            .quantum_correlation = quantum_correlation,
            .fractal_dimension = validated_fd,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Edge) void {
        freeMapStringBytes(&self.metadata, self.allocator);
        self.allocator.free(self.source);
        self.allocator.free(self.target);
    }

    pub fn clone(self: *const Edge, allocator: Allocator) !Edge {
        const new_source = try dupeBytes(allocator, self.source);
        errdefer allocator.free(new_source);
        const new_target = try dupeBytes(allocator, self.target);
        errdefer allocator.free(new_target);

        var new_meta = StringHashMap([]u8).init(allocator);
        errdefer freeMapStringBytes(&new_meta, allocator);
        try copyMetadataMap(&new_meta, &self.metadata, allocator);

        return Edge{
            .source = new_source,
            .target = new_target,
            .quality = self.quality,
            .weight = self.weight,
            .quantum_correlation = self.quantum_correlation,
            .fractal_dimension = self.fractal_dimension,
            .metadata = new_meta,
            .allocator = allocator,
        };
    }

    pub fn setMetadata(self: *Edge, key: []const u8, value: []const u8) !void {
        try putOwnedStringBytes(&self.metadata, self.allocator, key, value);
    }

    pub fn getMetadata(self: *const Edge, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn correlationMagnitude(self: *const Edge) f64 {
        return self.quantum_correlation.magnitude();
    }
};

pub const EdgeKey = struct {
    source: []const u8,
    target: []const u8,
};

fn hashSlice(h: *std.hash.Wyhash, s: []const u8) void {
    var len_buf: [8]u8 = undefined;
    writeU64Le(&len_buf, @as(u64, @intCast(s.len)));
    h.update(&len_buf);
    h.update(s);
}

pub const EdgeKeyContext = struct {
    pub fn hash(_: @This(), k: EdgeKey) u64 {
        var h = std.hash.Wyhash.init(0);
        hashSlice(&h, k.source);
        hashSlice(&h, k.target);
        return h.final();
    }

    pub fn eql(_: @This(), a: EdgeKey, b: EdgeKey) bool {
        return std.mem.eql(u8, a.source, b.source) and std.mem.eql(u8, a.target, b.target);
    }
};

pub const PairKey = struct {
    a: []const u8,
    b: []const u8,
};

pub const PairKeyContext = struct {
    pub fn hash(_: @This(), k: PairKey) u64 {
        var h = std.hash.Wyhash.init(1);
        hashSlice(&h, k.a);
        hashSlice(&h, k.b);
        return h.final();
    }

    pub fn eql(_: @This(), x: PairKey, y: PairKey) bool {
        return std.mem.eql(u8, x.a, y.a) and std.mem.eql(u8, x.b, y.b);
    }
};

pub const TwoQubit = struct {
    amps: [4]Complex(f64),

    pub fn initBellPhiPlus() TwoQubit {
        const inv_sqrt2 = 1.0 / std.math.sqrt(2.0);
        return TwoQubit{
            .amps = .{
                Complex(f64).init(inv_sqrt2, 0.0),
                Complex(f64).init(0.0, 0.0),
                Complex(f64).init(0.0, 0.0),
                Complex(f64).init(inv_sqrt2, 0.0),
            },
        };
    }

    pub fn normalizeInPlace(self: *TwoQubit) void {
        var ns: f64 = 0.0;
        var idx: usize = 0;
        while (idx < 4) : (idx += 1) ns += magSq(self.amps[idx]);
        if (!(ns > 1e-30) or std.math.isNan(ns) or std.math.isInf(ns)) {
            self.* = TwoQubit.initBellPhiPlus();
            return;
        }
        const inv = 1.0 / std.math.sqrt(ns);
        if (!isValidFloat(inv)) {
            self.* = TwoQubit.initBellPhiPlus();
            return;
        }
        const s = Complex(f64).init(inv, 0.0);
        idx = 0;
        while (idx < 4) : (idx += 1) {
            self.amps[idx] = self.amps[idx].mul(s);
        }
    }

    pub fn totalProbability(self: TwoQubit) f64 {
        var sum: f64 = 0.0;
        var idx: usize = 0;
        while (idx < 4) : (idx += 1) sum += magSq(self.amps[idx]);
        return sum;
    }
};

pub const Gate = union(enum) {
    hadamard,
    pauli_x,
    pauli_y,
    pauli_z,
    identity,
    phase: f64,

    pub fn apply(self: Gate, q: Qubit) Qubit {
        return switch (self) {
            .hadamard => blk: {
                const inv_sqrt2 = 1.0 / std.math.sqrt(2.0);
                const s = Complex(f64).init(inv_sqrt2, 0.0);
                const new_a = q.a.add(q.b).mul(s);
                const new_b = q.a.sub(q.b).mul(s);
                break :blk Qubit.init(new_a, new_b);
            },
            .pauli_x => Qubit.init(q.b, q.a),
            .pauli_y => blk: {
                const i_pos = Complex(f64).init(0.0, 1.0);
                const i_neg = Complex(f64).init(0.0, -1.0);
                break :blk Qubit.init(i_neg.mul(q.b), i_pos.mul(q.a));
            },
            .pauli_z => blk: {
                const neg = Complex(f64).init(-1.0, 0.0);
                break :blk Qubit.init(q.a, q.b.mul(neg));
            },
            .identity => Qubit.init(q.a, q.b),
            .phase => |p| blk: {
                if (!isValidFloat(p)) break :blk Qubit.init(q.a, q.b);
                const c = std.math.cos(p);
                const si = std.math.sin(p);
                if (!isValidFloat(c) or !isValidFloat(si)) break :blk Qubit.init(q.a, q.b);
                const factor = Complex(f64).init(c, si);
                break :blk Qubit.init(q.a, q.b.mul(factor));
            },
        };
    }
};

pub fn hadamardGate(q: Qubit) Qubit {
    return (Gate{ .hadamard = {} }).apply(q);
}

pub fn pauliXGate(q: Qubit) Qubit {
    return (Gate{ .pauli_x = {} }).apply(q);
}

pub fn pauliYGate(q: Qubit) Qubit {
    return (Gate{ .pauli_y = {} }).apply(q);
}

pub fn pauliZGate(q: Qubit) Qubit {
    return (Gate{ .pauli_z = {} }).apply(q);
}

pub fn identityGate(q: Qubit) Qubit {
    return (Gate{ .identity = {} }).apply(q);
}

pub fn phaseGate(p: f64) Gate {
    return Gate{ .phase = p };
}

fn shaUpdateU64(h: *Sha256, v: u64) void {
    var b: [8]u8 = undefined;
    writeU64Le(&b, v);
    h.update(&b);
}

fn shaUpdateBytes(h: *Sha256, b: []const u8) void {
    shaUpdateU64(h, @as(u64, @intCast(b.len)));
    h.update(b);
}

fn shaUpdateF64(h: *Sha256, v: f64) void {
    shaUpdateU64(h, canonicalizeFloat(v));
}

fn addDigest(acc: *[Sha256.digest_length]u8, d: *const [Sha256.digest_length]u8) void {
    var carry: u16 = 0;
    var i: usize = 0;
    while (i < Sha256.digest_length) : (i += 1) {
        carry += @as(u16, acc[i]) + @as(u16, d[i]);
        acc[i] = @as(u8, @truncate(carry));
        carry >>= 8;
    }
}

const EdgeMap = std.HashMap(EdgeKey, ArrayList(Edge), EdgeKeyContext, std.hash_map.default_max_load_percentage);
const EntMap = std.HashMap(PairKey, TwoQubit, PairKeyContext, std.hash_map.default_max_load_percentage);

const MeasureHit = struct {
    key: PairKey,
    state: TwoQubit,
};

fn measureHitLessThan(_: void, a: MeasureHit, b: MeasureHit) bool {
    if (std.mem.lessThan(u8, a.key.a, b.key.a)) return true;
    if (std.mem.lessThan(u8, b.key.a, a.key.a)) return false;
    return std.mem.lessThan(u8, a.key.b, b.key.b);
}

fn sliceLessThanConst(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn sliceLessThanMut(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

pub const SelfSimilarRelationalGraph = struct {
    allocator: Allocator,
    nodes: StringHashMap(Node),
    edges: EdgeMap,
    entanglements: EntMap,
    quantum_register: StringHashMap(Qubit),
    topology_hash: [65]u8,
    rng: std.rand.DefaultPrng,
    total_edge_count: usize,
    topology_dirty: bool,

    pub fn init(allocator: Allocator) !SelfSimilarRelationalGraph {
        const ts = std.time.nanoTimestamp();
        const seed: u64 = std.hash.Wyhash.hash(0, std.mem.asBytes(&ts));
        return initWithSeed(allocator, seed);
    }

    pub fn initWithSeed(allocator: Allocator, seed: u64) !SelfSimilarRelationalGraph {
        var g = SelfSimilarRelationalGraph{
            .allocator = allocator,
            .nodes = StringHashMap(Node).init(allocator),
            .edges = EdgeMap.init(allocator),
            .entanglements = EntMap.init(allocator),
            .quantum_register = StringHashMap(Qubit).init(allocator),
            .topology_hash = [_]u8{0} ** 65,
            .rng = std.rand.DefaultPrng.init(seed),
            .total_edge_count = 0,
            .topology_dirty = true,
        };
        g.recomputeTopologyHash();
        return g;
    }

    pub fn deinit(self: *SelfSimilarRelationalGraph) void {
        var ed_it = self.edges.iterator();
        while (ed_it.next()) |e| {
            var idx: usize = 0;
            while (idx < e.value_ptr.items.len) : (idx += 1) {
                e.value_ptr.items[idx].deinit();
            }
            e.value_ptr.deinit();
        }
        self.edges.deinit();

        self.entanglements.deinit();

        var qr_it = self.quantum_register.iterator();
        while (qr_it.next()) |e| {
            freeOwnedSlice(self.allocator, e.key_ptr.*);
        }
        self.quantum_register.deinit();

        var n_it = self.nodes.iterator();
        while (n_it.next()) |e| e.value_ptr.deinit();
        self.nodes.deinit();
    }

    fn ensureQrEntry(self: *SelfSimilarRelationalGraph, node_id: []const u8, q: Qubit) !void {
        if (self.quantum_register.getPtr(node_id)) |qptr| {
            qptr.* = q;
        } else {
            const k = try dupeBytes(self.allocator, node_id);
            errdefer self.allocator.free(k);
            try self.quantum_register.put(k, q);
        }
    }

    fn canonicalIdPtr(self: *SelfSimilarRelationalGraph, id: []const u8) ?[]const u8 {
        if (self.nodes.getPtr(id)) |n| return n.id;
        return null;
    }

    pub fn addNode(self: *SelfSimilarRelationalGraph, node: *const Node) !void {
        if (self.nodes.getPtr(node.id)) |existing| {
            const new_data = try dupeBytes(self.allocator, node.data);
            errdefer self.allocator.free(new_data);

            var new_meta = StringHashMap([]u8).init(self.allocator);
            errdefer freeMapStringBytes(&new_meta, self.allocator);
            try copyMetadataMap(&new_meta, &node.metadata, self.allocator);

            try self.ensureQrEntry(existing.id, node.qubit);

            existing.allocator.free(existing.data);
            freeMapStringBytes(&existing.metadata, existing.allocator);

            existing.data = new_data;
            existing.metadata = new_meta;
            existing.qubit = node.qubit;
            existing.phase = if (isValidFloat(node.phase)) node.phase else 0.0;
            existing.allocator = self.allocator;
        } else {
            var cloned = try node.clone(self.allocator);

            const qr_key = try dupeBytes(self.allocator, cloned.id);
            self.quantum_register.put(qr_key, cloned.qubit) catch |err| {
                self.allocator.free(qr_key);
                cloned.deinit();
                return err;
            };

            self.nodes.put(cloned.id, cloned) catch |err| {
                if (self.quantum_register.fetchRemove(cloned.id)) |removed| {
                    freeOwnedSlice(self.allocator, removed.key);
                }
                cloned.deinit();
                return err;
            };
        }
        self.topology_dirty = true;
    }

    fn buildStoredEdge(self: *SelfSimilarRelationalGraph, s: []const u8, t: []const u8, edge_in: *const Edge) !Edge {
        const duped_s = try dupeBytes(self.allocator, s);
        errdefer self.allocator.free(duped_s);
        const duped_t = try dupeBytes(self.allocator, t);
        errdefer self.allocator.free(duped_t);
        var new_meta = StringHashMap([]u8).init(self.allocator);
        errdefer freeMapStringBytes(&new_meta, self.allocator);
        try copyMetadataMap(&new_meta, &edge_in.metadata, self.allocator);
        return Edge{
            .source = duped_s,
            .target = duped_t,
            .quality = edge_in.quality,
            .weight = if (isValidFloat(edge_in.weight)) edge_in.weight else 0.0,
            .quantum_correlation = edge_in.quantum_correlation,
            .fractal_dimension = if (isValidFloat(edge_in.fractal_dimension)) edge_in.fractal_dimension else 0.0,
            .metadata = new_meta,
            .allocator = self.allocator,
        };
    }

    fn addEdgeInternal(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8, edge_in: *const Edge) !void {
        const s = self.canonicalIdPtr(source) orelse return error.SourceNodeNotFound;
        const t = self.canonicalIdPtr(target) orelse return error.TargetNodeNotFound;

        var stored = try self.buildStoredEdge(s, t, edge_in);

        const key = EdgeKey{ .source = s, .target = t };
        var gop = self.edges.getOrPut(key) catch |err| {
            stored.deinit();
            return err;
        };
        if (!gop.found_existing) gop.value_ptr.* = ArrayList(Edge).init(self.allocator);

        gop.value_ptr.append(stored) catch |err| {
            stored.deinit();
            if (!gop.found_existing and gop.value_ptr.items.len == 0) {
                gop.value_ptr.deinit();
                _ = self.edges.remove(key);
            }
            return err;
        };

        self.total_edge_count += 1;
    }

    pub fn addEdge(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8, edge: *const Edge) !void {
        try self.addEdgeInternal(source, target, edge);
        self.topology_dirty = true;
    }

    fn removeLastEdge(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8) void {
        const key = EdgeKey{ .source = source, .target = target };
        if (self.edges.getPtr(key)) |lst| {
            if (lst.items.len > 0) {
                var last = lst.pop();
                last.deinit();
                self.total_edge_count -= 1;
            }
            if (lst.items.len == 0) {
                lst.deinit();
                _ = self.edges.remove(key);
            }
        }
    }

    fn collapseEdgesBetween(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8) void {
        const key = EdgeKey{ .source = source, .target = target };
        if (self.edges.getPtr(key)) |lst| {
            var idx: usize = 0;
            while (idx < lst.items.len) : (idx += 1) {
                lst.items[idx].quality = .collapsed;
                lst.items[idx].quantum_correlation = Complex(f64).init(0.0, 0.0);
            }
        }
    }

    pub fn getNodeConst(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?*const Node {
        if (@constCast(&self.nodes).getPtr(node_id)) |p| return p;
        return null;
    }

    pub fn getEdgesConst(self: *const SelfSimilarRelationalGraph, source: []const u8, target: []const u8) ?[]const Edge {
        const key = EdgeKey{ .source = source, .target = target };
        if (@constCast(&self.edges).getPtr(key)) |list| return list.items;
        return null;
    }

    pub fn clear(self: *SelfSimilarRelationalGraph) void {
        var ed_it = self.edges.iterator();
        while (ed_it.next()) |e| {
            var idx: usize = 0;
            while (idx < e.value_ptr.items.len) : (idx += 1) {
                e.value_ptr.items[idx].deinit();
            }
            e.value_ptr.deinit();
        }
        self.edges.clearRetainingCapacity();

        self.entanglements.clearRetainingCapacity();

        var qr_it = self.quantum_register.iterator();
        while (qr_it.next()) |e| {
            freeOwnedSlice(self.allocator, e.key_ptr.*);
        }
        self.quantum_register.clearRetainingCapacity();

        var n_it = self.nodes.iterator();
        while (n_it.next()) |e| e.value_ptr.deinit();
        self.nodes.clearRetainingCapacity();

        self.total_edge_count = 0;
        self.topology_dirty = true;
    }

    pub fn setQuantumState(self: *SelfSimilarRelationalGraph, node_id: []const u8, q: Qubit) !void {
        const n = self.nodes.getPtr(node_id) orelse return error.NodeNotFound;

        var validated = q;
        validated.normalizeInPlace();

        try self.ensureQrEntry(node_id, validated);
        n.qubit = validated;
        self.topology_dirty = true;
    }

    pub fn getQuantumState(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?Qubit {
        return self.quantum_register.get(node_id);
    }

    pub fn applyQuantumGate(self: *SelfSimilarRelationalGraph, node_id: []const u8, gate: Gate) !void {
        const n = self.nodes.getPtr(node_id) orelse return error.NodeNotFound;
        var new_qubit = gate.apply(n.qubit);
        new_qubit.normalizeInPlace();

        try self.ensureQrEntry(node_id, new_qubit);
        n.qubit = new_qubit;
        self.topology_dirty = true;
    }

    pub fn entangleNodes(self: *SelfSimilarRelationalGraph, a_id: []const u8, b_id: []const u8) !void {
        if (std.mem.eql(u8, a_id, b_id)) return error.SelfEntanglement;

        const a = self.canonicalIdPtr(a_id) orelse return error.NodeNotFound;
        const b = self.canonicalIdPtr(b_id) orelse return error.NodeNotFound;

        const pk = if (std.mem.lessThan(u8, a, b)) PairKey{ .a = a, .b = b } else PairKey{ .a = b, .b = a };

        if (self.entanglements.contains(pk)) return error.AlreadyEntangled;

        var edge_ab = try Edge.init(self.allocator, a, b, .entangled, 1.0, Complex(f64).init(1.0, 0.0), 0.0);
        defer edge_ab.deinit();
        try self.addEdgeInternal(a, b, &edge_ab);
        errdefer self.removeLastEdge(a, b);

        var edge_ba = try Edge.init(self.allocator, b, a, .entangled, 1.0, Complex(f64).init(1.0, 0.0), 0.0);
        defer edge_ba.deinit();
        try self.addEdgeInternal(b, a, &edge_ba);
        errdefer self.removeLastEdge(b, a);

        try self.entanglements.put(pk, TwoQubit.initBellPhiPlus());

        self.topology_dirty = true;
    }

    pub fn measure(self: *SelfSimilarRelationalGraph, node_id: []const u8) !u1 {
        _ = self.canonicalIdPtr(node_id) orelse return error.NodeNotFound;

        var hits = ArrayList(MeasureHit).init(self.allocator);
        defer hits.deinit();

        var it = self.entanglements.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.a, node_id) or std.mem.eql(u8, e.key_ptr.b, node_id)) {
                try hits.append(MeasureHit{ .key = e.key_ptr.*, .state = e.value_ptr.* });
            }
        }

        if (hits.items.len > 0) {
            std.mem.sort(MeasureHit, hits.items, {}, measureHitLessThan);

            var primary = hits.items[0].state;
            primary.normalizeInPlace();

            const r = self.rng.random().float(f64);
            var cum: f64 = 0.0;
            var outcome: usize = 3;
            var si: usize = 0;
            while (si < 4) : (si += 1) {
                const prob = magSq(primary.amps[si]);
                if (std.math.isNan(prob)) continue;
                cum += prob;
                if (r <= cum) {
                    outcome = si;
                    break;
                }
            }

            const pk = hits.items[0].key;
            const a_id = pk.a;
            const b_id = pk.b;

            const a_ptr = self.nodes.getPtr(a_id) orelse return error.InconsistentState;
            const b_ptr = self.nodes.getPtr(b_id) orelse return error.InconsistentState;

            switch (outcome) {
                0 => {
                    a_ptr.qubit = Qubit.initBasis0();
                    b_ptr.qubit = Qubit.initBasis0();
                },
                1 => {
                    a_ptr.qubit = Qubit.initBasis0();
                    b_ptr.qubit = Qubit.initBasis1();
                },
                2 => {
                    a_ptr.qubit = Qubit.initBasis1();
                    b_ptr.qubit = Qubit.initBasis0();
                },
                else => {
                    a_ptr.qubit = Qubit.initBasis1();
                    b_ptr.qubit = Qubit.initBasis1();
                },
            }

            try self.ensureQrEntry(a_id, a_ptr.qubit);
            try self.ensureQrEntry(b_id, b_ptr.qubit);

            var h: usize = 0;
            while (h < hits.items.len) : (h += 1) {
                const key = hits.items[h].key;
                self.collapseEdgesBetween(key.a, key.b);
                self.collapseEdgesBetween(key.b, key.a);

                if (h > 0) {
                    const partner_id = if (std.mem.eql(u8, node_id, key.a)) key.b else key.a;
                    if (self.nodes.getPtr(partner_id)) |partner| {
                        partner.qubit = Qubit.initBasis0();
                        self.ensureQrEntry(partner_id, partner.qubit) catch {};
                    }
                }

                _ = self.entanglements.remove(key);
            }

            const bit: u1 = if (std.mem.eql(u8, node_id, a_id))
                @as(u1, @intCast((outcome >> 1) & 1))
            else
                @as(u1, @intCast(outcome & 1));

            self.topology_dirty = true;
            return bit;
        }

        const n = self.nodes.getPtr(node_id) orelse return error.NodeNotFound;
        var p0 = n.qubit.prob0();
        if (std.math.isNan(p0)) p0 = 0.5;
        const r0 = self.rng.random().float(f64);
        const bit: u1 = if (r0 <= p0) 0 else 1;

        n.qubit = if (bit == 0) Qubit.initBasis0() else Qubit.initBasis1();
        try self.ensureQrEntry(node_id, n.qubit);

        self.topology_dirty = true;
        return bit;
    }

    pub fn nodeCount(self: *const SelfSimilarRelationalGraph) usize {
        return self.nodes.count();
    }

    pub fn edgeCount(self: *const SelfSimilarRelationalGraph) usize {
        return self.total_edge_count;
    }

    pub fn getAllNodeIds(self: *const SelfSimilarRelationalGraph, allocator: Allocator) !ArrayList([]u8) {
        var out = ArrayList([]u8).init(allocator);
        errdefer {
            var idx: usize = 0;
            while (idx < out.items.len) : (idx += 1) allocator.free(out.items[idx]);
            out.deinit();
        }
        try out.ensureTotalCapacity(self.nodes.count());
        var nit = @constCast(&self.nodes).iterator();
        while (nit.next()) |e| {
            const copy = try allocator.dupe(u8, e.value_ptr.id);
            out.appendAssumeCapacity(copy);
        }
        std.mem.sort([]u8, out.items, {}, sliceLessThanMut);
        return out;
    }

    fn recomputeTopologyHash(self: *SelfSimilarRelationalGraph) void {
        var acc_nodes: [Sha256.digest_length]u8 = [_]u8{0} ** Sha256.digest_length;
        var acc_edges: [Sha256.digest_length]u8 = [_]u8{0} ** Sha256.digest_length;
        var acc_ent: [Sha256.digest_length]u8 = [_]u8{0} ** Sha256.digest_length;

        var node_count: u64 = 0;
        var edgekey_count: u64 = 0;
        var edge_total: u64 = 0;
        var ent_count: u64 = 0;

        var n_it = self.nodes.iterator();
        while (n_it.next()) |e| {
            node_count +%= 1;

            var meta_acc: [Sha256.digest_length]u8 = [_]u8{0} ** Sha256.digest_length;
            var meta_count: u64 = 0;

            var mit = e.value_ptr.metadata.iterator();
            while (mit.next()) |me| {
                meta_count +%= 1;
                var mh = Sha256.init(.{});
                shaUpdateBytes(&mh, me.key_ptr.*);
                shaUpdateBytes(&mh, me.value_ptr.*);
                var md: [Sha256.digest_length]u8 = undefined;
                mh.final(&md);
                addDigest(&meta_acc, &md);
            }

            var nh = Sha256.init(.{});
            shaUpdateBytes(&nh, e.value_ptr.id);
            shaUpdateBytes(&nh, e.value_ptr.data);
            shaUpdateF64(&nh, e.value_ptr.phase);
            shaUpdateF64(&nh, e.value_ptr.qubit.a.re);
            shaUpdateF64(&nh, e.value_ptr.qubit.a.im);
            shaUpdateF64(&nh, e.value_ptr.qubit.b.re);
            shaUpdateF64(&nh, e.value_ptr.qubit.b.im);
            nh.update(&meta_acc);
            shaUpdateU64(&nh, meta_count);

            var d: [Sha256.digest_length]u8 = undefined;
            nh.final(&d);
            addDigest(&acc_nodes, &d);
        }

        var e_it = self.edges.iterator();
        while (e_it.next()) |kv| {
            edgekey_count +%= 1;

            var edges_acc: [Sha256.digest_length]u8 = [_]u8{0} ** Sha256.digest_length;
            var edges_count: u64 = 0;

            var eidx: usize = 0;
            while (eidx < kv.value_ptr.items.len) : (eidx += 1) {
                const edge = &kv.value_ptr.items[eidx];
                edges_count +%= 1;
                edge_total +%= 1;

                var emeta_acc: [Sha256.digest_length]u8 = [_]u8{0} ** Sha256.digest_length;
                var emeta_count: u64 = 0;

                var emi = edge.metadata.iterator();
                while (emi.next()) |me| {
                    emeta_count +%= 1;
                    var emh = Sha256.init(.{});
                    shaUpdateBytes(&emh, me.key_ptr.*);
                    shaUpdateBytes(&emh, me.value_ptr.*);
                    var emd: [Sha256.digest_length]u8 = undefined;
                    emh.final(&emd);
                    addDigest(&emeta_acc, &emd);
                }

                var eh = Sha256.init(.{});
                shaUpdateU64(&eh, @as(u64, edge.quality.toInt()));
                shaUpdateF64(&eh, edge.weight);
                shaUpdateF64(&eh, edge.fractal_dimension);
                shaUpdateF64(&eh, edge.quantum_correlation.re);
                shaUpdateF64(&eh, edge.quantum_correlation.im);
                eh.update(&emeta_acc);
                shaUpdateU64(&eh, emeta_count);

                var ed: [Sha256.digest_length]u8 = undefined;
                eh.final(&ed);
                addDigest(&edges_acc, &ed);
            }

            var kh = Sha256.init(.{});
            shaUpdateBytes(&kh, kv.key_ptr.source);
            shaUpdateBytes(&kh, kv.key_ptr.target);
            kh.update(&edges_acc);
            shaUpdateU64(&kh, edges_count);

            var kd: [Sha256.digest_length]u8 = undefined;
            kh.final(&kd);
            addDigest(&acc_edges, &kd);
        }

        var en_it = self.entanglements.iterator();
        while (en_it.next()) |kv| {
            ent_count +%= 1;

            var eh = Sha256.init(.{});
            shaUpdateBytes(&eh, kv.key_ptr.a);
            shaUpdateBytes(&eh, kv.key_ptr.b);
            var ai: usize = 0;
            while (ai < 4) : (ai += 1) {
                shaUpdateF64(&eh, kv.value_ptr.amps[ai].re);
                shaUpdateF64(&eh, kv.value_ptr.amps[ai].im);
            }

            var ed: [Sha256.digest_length]u8 = undefined;
            eh.final(&ed);
            addDigest(&acc_ent, &ed);
        }

        var final_h = Sha256.init(.{});
        final_h.update(&acc_nodes);
        shaUpdateU64(&final_h, node_count);
        final_h.update(&acc_edges);
        shaUpdateU64(&final_h, edgekey_count);
        shaUpdateU64(&final_h, edge_total);
        final_h.update(&acc_ent);
        shaUpdateU64(&final_h, ent_count);

        var digest: [Sha256.digest_length]u8 = undefined;
        final_h.final(&digest);

        const hex_chars = "0123456789abcdef";
        var out: [65]u8 = undefined;
        var di: usize = 0;
        while (di < Sha256.digest_length) : (di += 1) {
            out[di * 2] = hex_chars[@as(usize, digest[di] >> 4)];
            out[di * 2 + 1] = hex_chars[@as(usize, digest[di] & 0x0f)];
        }
        out[64] = 0;
        self.topology_hash = out;
        self.topology_dirty = false;
    }

    pub fn getTopologyHashHex(self: *SelfSimilarRelationalGraph) []const u8 {
        if (self.topology_dirty) {
            self.recomputeTopologyHash();
        }
        return self.topology_hash[0..64];
    }

    pub fn encodeInformation(self: *SelfSimilarRelationalGraph, data: []const u8) ![]u8 {
        var hash_buf: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(data, &hash_buf, .{});

        var id_buf: [32]u8 = undefined;
        const hex_chars = "0123456789abcdef";
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            id_buf[i * 2] = hex_chars[@as(usize, hash_buf[i] >> 4)];
            id_buf[i * 2 + 1] = hex_chars[@as(usize, hash_buf[i] & 0x0f)];
        }

        const id_slice: []const u8 = id_buf[0..];
        const is_update = self.nodes.contains(id_slice);

        var node = try Node.init(self.allocator, id_slice, data, Qubit.initBasis0(), 0.0);

        const ts_str = std.fmt.allocPrint(self.allocator, "{d}", .{std.time.timestamp()}) catch |err| {
            node.deinit();
            return err;
        };
        defer self.allocator.free(ts_str);

        node.setMetadata("encoding_time", ts_str) catch |err| {
            node.deinit();
            return err;
        };

        var old_data_copy: ?[]u8 = null;
        var old_meta_copy: ?StringHashMap([]u8) = null;
        var old_qubit_copy = Qubit.initBasis0();
        var old_phase_copy: f64 = 0.0;
        var has_backup = false;

        if (is_update) {
            if (self.nodes.getPtr(id_slice)) |existing| {
                old_data_copy = dupeBytes(self.allocator, existing.data) catch |err| {
                    node.deinit();
                    return err;
                };
                var mb = StringHashMap([]u8).init(self.allocator);
                copyMetadataMap(&mb, &existing.metadata, self.allocator) catch |err| {
                    self.allocator.free(old_data_copy.?);
                    old_data_copy = null;
                    freeMapStringBytes(&mb, self.allocator);
                    node.deinit();
                    return err;
                };
                old_meta_copy = mb;
                old_qubit_copy = existing.qubit;
                old_phase_copy = existing.phase;
                has_backup = true;
            }
        }

        self.addNode(&node) catch |err| {
            if (has_backup) {
                self.allocator.free(old_data_copy.?);
                freeMapStringBytes(&old_meta_copy.?, self.allocator);
            }
            node.deinit();
            return err;
        };
        node.deinit();

        const result_id = try dupeBytes(self.allocator, id_slice);
        errdefer self.allocator.free(result_id);

        var added_edge_keys: [3]EdgeKey = undefined;
        var added_edge_count: usize = 0;

        errdefer {
            var k: usize = added_edge_count;
            while (k > 0) {
                k -= 1;
                self.removeLastEdge(added_edge_keys[k].source, added_edge_keys[k].target);
            }
            if (is_update and has_backup) {
                if (self.nodes.getPtr(id_slice)) |existing| {
                    self.allocator.free(existing.data);
                    freeMapStringBytes(&existing.metadata, self.allocator);
                    existing.data = old_data_copy.?;
                    existing.metadata = old_meta_copy.?;
                    existing.qubit = old_qubit_copy;
                    existing.phase = old_phase_copy;
                    has_backup = false;
                    if (self.quantum_register.getPtr(id_slice)) |qp| qp.* = old_qubit_copy;
                }
            } else if (!is_update) {
                if (self.nodes.fetchRemove(id_slice)) |removed| {
                    var v = removed.value;
                    v.deinit();
                }
                if (self.quantum_register.fetchRemove(id_slice)) |removed| {
                    freeOwnedSlice(self.allocator, removed.key);
                }
            }
            self.topology_dirty = true;
        }

        const src = self.canonicalIdPtr(id_slice) orelse return error.InconsistentState;

        var sorted_targets = ArrayList([]const u8).init(self.allocator);
        defer sorted_targets.deinit();

        var nit = self.nodes.iterator();
        while (nit.next()) |entry| {
            if (!std.mem.eql(u8, entry.value_ptr.id, id_slice)) {
                try sorted_targets.append(entry.value_ptr.id);
            }
        }
        std.mem.sort([]const u8, sorted_targets.items, {}, sliceLessThanConst);

        var linked: usize = 0;
        while (linked < sorted_targets.items.len and linked < 3) : (linked += 1) {
            const dst = sorted_targets.items[linked];
            var edge = try Edge.init(self.allocator, src, dst, .coherent, 0.5, Complex(f64).init(0.0, 0.0), 0.0);
            defer edge.deinit();
            try self.addEdgeInternal(src, dst, &edge);
            added_edge_keys[linked] = EdgeKey{ .source = src, .target = dst };
            added_edge_count = linked + 1;
        }

        if (has_backup) {
            self.allocator.free(old_data_copy.?);
            freeMapStringBytes(&old_meta_copy.?, self.allocator);
        }

        self.topology_dirty = true;
        return result_id;
    }

    pub fn decodeInformation(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?[]const u8 {
        if (@constCast(&self.nodes).getPtr(node_id)) |n| return n.data;
        return null;
    }
};

export fn verify_canonicalize_float(v: u64) callconv(.C) u64 {
    const f: f64 = @bitCast(v);
    return canonicalizeFloat(f);
}

export fn verify_write_u64_le(buf: *[8]u8, v: u64) callconv(.C) void {
    writeU64Le(buf, v);
}

export fn verify_add_digest(acc: *[Sha256.digest_length]u8, d: *const [Sha256.digest_length]u8) callconv(.C) void {
    addDigest(acc, d);
}
