const std = @import("std");

const types = @import("core/types.zig");
const tensor_mod = @import("core/tensor.zig");
const io_mod = @import("core/io.zig");
const memory_mod = @import("core/memory.zig");
const model_io_mod = @import("core/model_io.zig");

const rsf_mod = @import("processor/rsf.zig");
const mgt_mod = @import("tokenizer/mgt.zig");
const sfd_mod = @import("optimizer/sfd.zig");
const ssi_mod = @import("index/ssi.zig");
const ranker_mod = @import("ranker/ranker.zig");

const accel_interface = @import("hw/accel/accel_interface.zig");
const cuda_bindings = @import("hw/accel/cuda_bindings.zig");
const futhark_bindings = @import("hw/accel/futhark_bindings.zig");
const fractal_lpu = @import("hw/accel/fractal_lpu.zig");

const distributed_trainer = @import("distributed/distributed_trainer.zig");
const distributed_trainer_futhark = @import("distributed/distributed_trainer_futhark.zig");
const gpu_coordinator = @import("distributed/gpu_coordinator.zig");
const modal_gpu = @import("distributed/modal_gpu.zig");
const nccl_bindings = @import("distributed/nccl_bindings.zig");

const core_relational = @import("core_relational/mod.zig");
const chaos_core = @import("core_relational/chaos_core.zig");
const crev_pipeline = @import("core_relational/crev_pipeline.zig");
const dataset_obfuscation = @import("core_relational/dataset_obfuscation.zig");
const esso_optimizer = @import("core_relational/esso_optimizer.zig");
const fnds = @import("core_relational/fnds.zig");
const formal_verification = @import("core_relational/formal_verification.zig");
const ibm_quantum = @import("core_relational/ibm_quantum.zig");
const nsir_core = @import("core_relational/nsir_core.zig");
const quantum_hardware = @import("core_relational/quantum_hardware.zig");
const quantum_logic = @import("core_relational/quantum_logic.zig");
const quantum_task_adapter = @import("core_relational/quantum_task_adapter.zig");
const r_gpu = @import("core_relational/r_gpu.zig");
const reasoning_orchestrator = @import("core_relational/reasoning_orchestrator.zig");
const safety = @import("core_relational/safety.zig");
const security_proofs = @import("core_relational/security_proofs.zig");
const signal_propagation = @import("core_relational/signal_propagation.zig");
const surprise_memory = @import("core_relational/surprise_memory.zig");
const temporal_graph = @import("core_relational/temporal_graph.zig");
const type_theory = @import("core_relational/type_theory.zig");
const verified_inference_engine = @import("core_relational/verified_inference_engine.zig");
const vpu = @import("core_relational/vpu.zig");
const z_runtime = @import("core_relational/z_runtime.zig");
const zk_verification = @import("core_relational/zk_verification.zig");
const c_api = @import("core_relational/c_api.zig");

const inference_server = @import("api/inference_server.zig");

const Tensor = tensor_mod.Tensor;
const RSF = rsf_mod.RSF;
const MGT = mgt_mod.MGT;
const SFD = sfd_mod.SFD;
const SSI = ssi_mod.SSI;
const Ranker = ranker_mod.Ranker;
const PRNG = types.PRNG;

pub const MainConfig = struct {
    pub const DEFAULT_EMBEDDING_DIM: usize = 128;
    pub const MIN_EMBEDDING_DIM: usize = 8;
    pub const MAX_EMBEDDING_DIM: usize = 16384;
    pub const DEFAULT_RSF_LAYERS: usize = 4;
    pub const MIN_RSF_LAYERS: usize = 1;
    pub const MAX_RSF_LAYERS: usize = 256;
    pub const DEFAULT_BATCH_SIZE: usize = 16;
    pub const MIN_BATCH_SIZE: usize = 1;
    pub const MAX_BATCH_SIZE: usize = 4096;
    pub const DEFAULT_NUM_EPOCHS: usize = 10;
    pub const MAX_NUM_EPOCHS: usize = 100000;
    pub const DEFAULT_LEARNING_RATE: f32 = 0.001;
    pub const MIN_LEARNING_RATE: f32 = 1e-10;
    pub const MAX_LEARNING_RATE: f32 = 10.0;
    pub const DEFAULT_TRAINING_SAMPLES: usize = 100;
    pub const DEFAULT_VALIDATION_SAMPLES: usize = 100;
    pub const MIN_SAMPLES: usize = 1;
    pub const MAX_SAMPLES: usize = 1000000;
    pub const DEFAULT_SAMPLE_LIMIT: usize = 1000000;
    pub const DEFAULT_GRADIENT_CLIP_NORM: f32 = 5.0;
    pub const DEFAULT_SEQUENCE_LENGTH: usize = 64;
    pub const DEFAULT_TOP_K: usize = 5;
    pub const RANKER_NGRAM_SIZE: usize = 10;
    pub const RANKER_LSH_TABLES: usize = 16;
    pub const RANKER_SEED: u64 = 42;
    pub const TEST_DIM: usize = 128;
    pub const TEST_LAYERS: usize = 4;
    pub const TEST_PARAM_SIZE: usize = 128;
    pub const TEST_TOKEN_COUNT: usize = 8;
    pub const REPL_LINE_BUFFER_SIZE: usize = 8192;
    pub const ANCHOR_MODULO: usize = 3;
    pub const TENSOR_INIT_SCALE: f32 = 0.1;
    pub const PARAM_UPDATE_DELTA: f32 = 0.001;
    pub const GRADIENT_SCALE: f32 = 0.01;
    pub const GRADIENT_RANGE_SCALE: f32 = 10.0;
    pub const NORM_TOLERANCE: f32 = 0.1;
    pub const CHANGE_THRESHOLD: f32 = 1e-6;
    pub const GRADIENT_THRESHOLD: f32 = 1e-9;
    pub const R_SQUARED_EPSILON: f64 = 1e-10;
    pub const CONFIDENCE_Z_SCORE: f64 = 1.96;
    pub const PRNG_SEED_FORWARD: u64 = 54321;
    pub const PRNG_SEED_VALIDATION: u64 = 12345;
    pub const PRNG_SEED_GRADIENT: u64 = 99999;
    pub const PRNG_SEED_SYNTHETIC: u64 = 42;
    pub const MAX_VALID_POSITION: u64 = 10000;
    pub const MAX_TOKEN_COUNT: usize = 1000;
    pub const GRADIENT_TENSOR_SIZE: usize = 100;
    pub const PARSE_BASE: u8 = 10;
    pub const DEFAULT_MODELS_DIR: []const u8 = "models";
    pub const FILE_MAGIC_RSF: u32 = 0x4A524653;
    pub const FILE_MAGIC_MGT: u32 = 0x4A4D4754;
    pub const FILE_MAGIC_RANKER: u32 = 0x4A524E4B;
    pub const FILE_MAGIC_PROJ: u32 = 0x4A50524A;
    pub const FILE_VERSION: u32 = 1;
    pub const MAX_LINE_LENGTH: usize = 65536;
    pub const MAX_VOCAB_SIZE: u32 = std.math.maxInt(u32);
    pub const MAX_TOKEN_LENGTH: u32 = 65536;
};

const Config = struct {
    embedding_dim: usize,
    rsf_layers: usize,
    batch_size: usize,
    num_epochs: usize,
    learning_rate: f32,
    num_training_samples: usize,
    num_validation_samples: usize,
    models_dir: []const u8,
    vocab_file: ?[]const u8,
    dataset_path: ?[]const u8,
    sample_limit: usize,
    gradient_clip_norm: f32,
    sequence_length: usize,
    top_k: usize,
    allocator: std.mem.Allocator,
    models_dir_allocated: ?[]u8,
    vocab_file_allocated: ?[]u8,
    dataset_path_allocated: ?[]u8,
    mode: []const u8,

    pub fn init(allocator: std.mem.Allocator) !Config {
        const models_dir_copy = try allocator.dupe(u8, MainConfig.DEFAULT_MODELS_DIR);
        const mode_str = try allocator.dupe(u8, "interactive");
        return Config{
            .embedding_dim = MainConfig.DEFAULT_EMBEDDING_DIM,
            .rsf_layers = MainConfig.DEFAULT_RSF_LAYERS,
            .batch_size = MainConfig.DEFAULT_BATCH_SIZE,
            .num_epochs = MainConfig.DEFAULT_NUM_EPOCHS,
            .learning_rate = MainConfig.DEFAULT_LEARNING_RATE,
            .num_training_samples = MainConfig.DEFAULT_TRAINING_SAMPLES,
            .num_validation_samples = MainConfig.DEFAULT_VALIDATION_SAMPLES,
            .models_dir = models_dir_copy,
            .vocab_file = null,
            .dataset_path = null,
            .sample_limit = MainConfig.DEFAULT_SAMPLE_LIMIT,
            .gradient_clip_norm = MainConfig.DEFAULT_GRADIENT_CLIP_NORM,
            .sequence_length = MainConfig.DEFAULT_SEQUENCE_LENGTH,
            .top_k = MainConfig.DEFAULT_TOP_K,
            .allocator = allocator,
            .models_dir_allocated = models_dir_copy,
            .vocab_file_allocated = null,
            .dataset_path_allocated = null,
            .mode = mode_str,
        };
    }

    pub fn deinit(self: *Config) void {
        if (self.models_dir_allocated) |dir| {
            self.allocator.free(dir);
        }
        if (self.vocab_file_allocated) |file| {
            self.allocator.free(file);
        }
        if (self.dataset_path_allocated) |path| {
            self.allocator.free(path);
        }
        self.allocator.free(self.mode);
    }

    pub fn validate(self: *const Config) error{InvalidConfig}!void {
        if (self.embedding_dim < MainConfig.MIN_EMBEDDING_DIM or self.embedding_dim > MainConfig.MAX_EMBEDDING_DIM) {
            return error.InvalidConfig;
        }
        if (self.rsf_layers < MainConfig.MIN_RSF_LAYERS or self.rsf_layers > MainConfig.MAX_RSF_LAYERS) {
            return error.InvalidConfig;
        }
        if (self.batch_size < MainConfig.MIN_BATCH_SIZE or self.batch_size > MainConfig.MAX_BATCH_SIZE) {
            return error.InvalidConfig;
        }
        if (self.num_epochs > MainConfig.MAX_NUM_EPOCHS) {
            return error.InvalidConfig;
        }
        if (self.learning_rate < MainConfig.MIN_LEARNING_RATE or self.learning_rate > MainConfig.MAX_LEARNING_RATE) {
            return error.InvalidConfig;
        }
        if (std.math.isNan(self.learning_rate) or std.math.isInf(self.learning_rate)) {
            return error.InvalidConfig;
        }
        if (self.num_training_samples < MainConfig.MIN_SAMPLES or self.num_training_samples > MainConfig.MAX_SAMPLES) {
            return error.InvalidConfig;
        }
        if (self.num_validation_samples > MainConfig.MAX_SAMPLES) {
            return error.InvalidConfig;
        }
        if (self.top_k == 0) {
            return error.InvalidConfig;
        }
        if (std.math.isNan(self.gradient_clip_norm) or std.math.isInf(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0) {
            return error.InvalidConfig;
        }
    }

    pub fn parseArgs(allocator: std.mem.Allocator) !Config {
        var config = try Config.init(allocator);
        errdefer config.deinit();

        var args = try std.process.argsWithAllocator(allocator);
        defer args.deinit();

        _ = args.skip();

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--embedding-dim")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.embedding_dim = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--layers")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.rsf_layers = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--batch-size")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.batch_size = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--epochs")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.num_epochs = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--lr")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                const lr = std.fmt.parseFloat(f32, val) catch return error.InvalidArgumentValue;
                if (std.math.isNan(lr) or std.math.isInf(lr)) return error.InvalidArgumentValue;
                config.learning_rate = lr;
            } else if (std.mem.eql(u8, arg, "--samples")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.num_training_samples = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--models-dir")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.models_dir_allocated) |old| {
                    config.allocator.free(old);
                }
                const duped = try allocator.dupe(u8, val);
                config.models_dir_allocated = duped;
                config.models_dir = duped;
            } else if (std.mem.eql(u8, arg, "--vocab-file")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.vocab_file_allocated) |old| {
                    config.allocator.free(old);
                }
                const duped = try allocator.dupe(u8, val);
                config.vocab_file_allocated = duped;
                config.vocab_file = duped;
            } else if (std.mem.eql(u8, arg, "--dataset-path")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                if (config.dataset_path_allocated) |old| {
                    config.allocator.free(old);
                }
                const duped = try allocator.dupe(u8, val);
                config.dataset_path_allocated = duped;
                config.dataset_path = duped;
            } else if (std.mem.eql(u8, arg, "--sample-limit")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.sample_limit = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--gradient-clip")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                const clip = std.fmt.parseFloat(f32, val) catch return error.InvalidArgumentValue;
                if (std.math.isNan(clip) or std.math.isInf(clip) or clip <= 0.0) return error.InvalidArgumentValue;
                config.gradient_clip_norm = clip;
            } else if (std.mem.eql(u8, arg, "--sequence-length")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.sequence_length = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                const val = args.next() orelse return error.MissingArgumentValue;
                config.top_k = std.fmt.parseInt(usize, val, MainConfig.PARSE_BASE) catch return error.InvalidArgumentValue;
            } else if (std.mem.eql(u8, arg, "--help")) {
                try printHelp();
                return error.HelpRequested;
            } else if (std.mem.eql(u8, arg, "--mode")) {
                const mode_val = args.next() orelse return error.MissingArgumentValue;
                allocator.free(config.mode);
                config.mode = try allocator.dupe(u8, mode_val);
            }
        }

        try config.validate();
        return config;
    }
};

const TrainingSample = struct {
    text: []u8,
    tokens: []u32,

    pub fn deinit(self: *TrainingSample, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.tokens);
    }
};

const ValidationMetrics = struct {
    mse: f32,
    rmse: f32,
    mae: f32,
    r_squared: f32,
    mean_prediction: f32,
    mean_target: f32,
    confidence_interval_lower: f32,
    confidence_interval_upper: f32,
    num_samples: usize,
};

const TerminalColors = struct {
    enabled: bool,
    reset: []const u8,
    bold: []const u8,
    cyan: []const u8,
    green: []const u8,
    yellow: []const u8,
    red: []const u8,

    fn detect() TerminalColors {
        const stdout = std.io.getStdOut();
        const tty_config = std.io.tty.detectConfig(stdout);
        const enabled = tty_config != .no_color;

        if (enabled) {
            return TerminalColors{
                .enabled = true,
                .reset = "\x1b[0m",
                .bold = "\x1b[1m",
                .cyan = "\x1b[36m",
                .green = "\x1b[32m",
                .yellow = "\x1b[33m",
                .red = "\x1b[31m",
            };
        } else {
            return TerminalColors{
                .enabled = false,
                .reset = "",
                .bold = "",
                .cyan = "",
                .green = "",
                .yellow = "",
                .red = "",
            };
        }
    }
};

fn runKgruTest(allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    const colors = TerminalColors.detect();

    try stdout.print("{s}{s}========================================{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    try stdout.print("{s}{s}  Component Test Suite{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    try stdout.print("{s}{s}========================================{s}\n\n", .{ colors.bold, colors.cyan, colors.reset });

    var tests_passed: usize = 0;
    var tests_failed: usize = 0;

    const test1_passed = blk: {
        try stdout.print("{s}[TEST 1]{s} RSF Processor Initialization & Forward Pass...\n", .{ colors.yellow, colors.reset });
        const dim: usize = MainConfig.TEST_DIM;
        const layers: usize = MainConfig.TEST_LAYERS;

        var rsf = RSF.init(allocator, dim, layers) catch |err| {
            try stdout.print("  {s}FAILED{s}: RSF init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer rsf.deinit();

        const tensor_dim = std.math.mul(usize, dim, 2) catch {
            try stdout.print("  {s}FAILED{s}: Dimension overflow\n", .{ colors.red, colors.reset });
            break :blk false;
        };

        var input_tensor = Tensor.init(allocator, &.{ 1, tensor_dim }) catch |err| {
            try stdout.print("  {s}FAILED{s}: Tensor init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer input_tensor.deinit();

        var prng = PRNG.init(MainConfig.PRNG_SEED_SYNTHETIC);
        var ti: usize = 0;
        while (ti < input_tensor.data.len) : (ti += 1) {
            input_tensor.data[ti] = ((@as(f32, @floatFromInt(prng.next())) / @as(f32, @floatFromInt(std.math.maxInt(u64)))) * 2.0 - 1.0) * MainConfig.TENSOR_INIT_SCALE;
        }

        rsf.forward(&input_tensor) catch |err| {
            try stdout.print("  {s}FAILED{s}: RSF forward error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };

        var valid_count: usize = 0;
        for (input_tensor.data) |v| {
            if (v != 0.0 and !std.math.isNan(v) and !std.math.isInf(v)) {
                valid_count += 1;
            }
        }

        if (valid_count == input_tensor.data.len) {
            try stdout.print("  {s}PASSED{s}: RSF forward pass produces valid output\n", .{ colors.green, colors.reset });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: RSF forward pass returned invalid data ({d}/{d} valid)\n", .{ colors.red, colors.reset, valid_count, input_tensor.data.len });
            break :blk false;
        }
    };

    if (test1_passed) tests_passed += 1 else tests_failed += 1;

    const test2_passed = blk: {
        try stdout.print("{s}[TEST 2]{s} SFD Optimizer Initialization...\n", .{ colors.yellow, colors.reset });
        const param_size: usize = MainConfig.TEST_PARAM_SIZE;

        var optimizer = SFD.init(allocator, param_size) catch |err| {
            try stdout.print("  {s}FAILED{s}: SFD init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer optimizer.deinit();

        var gradients = Tensor.init(allocator, &.{param_size}) catch |err| {
            try stdout.print("  {s}FAILED{s}: Gradient tensor init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer gradients.deinit();

        var params = Tensor.init(allocator, &.{param_size}) catch |err| {
            try stdout.print("  {s}FAILED{s}: Params tensor init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer params.deinit();

        var param_prng = PRNG.init(MainConfig.PRNG_SEED_GRADIENT);
        var pi: usize = 0;
        while (pi < param_size) : (pi += 1) {
            gradients.data[pi] = ((@as(f32, @floatFromInt(param_prng.next())) / @as(f32, @floatFromInt(std.math.maxInt(u64)))) * 2.0 - 1.0) * MainConfig.GRADIENT_SCALE;
            params.data[pi] = @as(f32, @floatFromInt(pi)) * MainConfig.PARAM_UPDATE_DELTA;
        }

        optimizer.update(&gradients, &params, MainConfig.DEFAULT_LEARNING_RATE) catch |err| {
            try stdout.print("  {s}FAILED{s}: SFD update error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };

        var params_changed = false;
        pi = 0;
        while (pi < param_size) : (pi += 1) {
            if (std.math.fabs(params.data[pi] - (@as(f32, @floatFromInt(pi)) * MainConfig.PARAM_UPDATE_DELTA)) > MainConfig.GRADIENT_THRESHOLD) {
                params_changed = true;
                break;
            }
        }

        if (params_changed) {
            try stdout.print("  {s}PASSED{s}: SFD optimizer update completed and params changed\n", .{ colors.green, colors.reset });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: SFD optimizer update did not change params\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test2_passed) tests_passed += 1 else tests_failed += 1;

    const test3_passed = blk: {
        try stdout.print("{s}[TEST 3]{s} MGT Tokenizer Initialization...\n", .{ colors.yellow, colors.reset });

        var mgt = initTokenizer(allocator) catch |err| {
            try stdout.print("  {s}FAILED{s}: MGT init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer mgt.deinit();

        const vocab_size = mgt.vocabSize();
        if (vocab_size > 0) {
            try stdout.print("  {s}PASSED{s}: MGT tokenizer initialized with vocab size {d}\n", .{ colors.green, colors.reset, vocab_size });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: MGT tokenizer has empty vocabulary\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test3_passed) tests_passed += 1 else tests_failed += 1;

    const test4_passed = blk: {
        try stdout.print("{s}[TEST 4]{s} SSI Index Operations...\n", .{ colors.yellow, colors.reset });

        var ssi = SSI.init(allocator);
        defer ssi.deinit();

        var test_tokens = try allocator.alloc(u32, MainConfig.TEST_TOKEN_COUNT);
        defer allocator.free(test_tokens);

        var idx: usize = 0;
        while (idx < MainConfig.TEST_TOKEN_COUNT) : (idx += 1) {
            test_tokens[idx] = @as(u32, @intCast(idx + 1));
        }

        ssi.addSequence(test_tokens, 0, true) catch |err| {
            try stdout.print("  {s}FAILED{s}: SSI addSequence error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };

        const stats = ssi.stats();
        if (stats.nodes > 0) {
            try stdout.print("  {s}PASSED{s}: SSI index created with {d} nodes\n", .{ colors.green, colors.reset, stats.nodes });
            break :blk true;
        } else {
            try stdout.print("  {s}FAILED{s}: SSI index is empty after adding sequence\n", .{ colors.red, colors.reset });
            break :blk false;
        }
    };

    if (test4_passed) tests_passed += 1 else tests_failed += 1;

    const test5_passed = blk: {
        try stdout.print("{s}[TEST 5]{s} Ranker Initialization...\n", .{ colors.yellow, colors.reset });

        var ranker = Ranker.init(allocator, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES, MainConfig.RANKER_SEED) catch |err| {
            try stdout.print("  {s}FAILED{s}: Ranker init error: {any}\n", .{ colors.red, colors.reset, err });
            break :blk false;
        };
        defer ranker.deinit();

        try stdout.print("  {s}PASSED{s}: Ranker initialized with ngrams={d}, lsh_tables={d}\n", .{ colors.green, colors.reset, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES });
        break :blk true;
    };

    if (test5_passed) tests_passed += 1 else tests_failed += 1;

    try stdout.writeAll("\n");
    try stdout.print("{s}{s}========================================{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    try stdout.print("{s}  Test Results: {d} passed, {d} failed{s}\n", .{
        if (tests_failed == 0) colors.green else colors.red,
        tests_passed,
        tests_failed,
        colors.reset,
    });
    try stdout.print("{s}{s}========================================{s}\n", .{ colors.bold, colors.cyan, colors.reset });

    if (tests_failed > 0) {
        return error.TestsFailed;
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("Memory leak detected\n", .{});
            std.process.exit(1);
        }
    }
    const allocator = gpa.allocator();

    var config = Config.parseArgs(allocator) catch |err| {
        if (err == error.HelpRequested) {
            return;
        }
        std.debug.print("Configuration error: {any}\n", .{err});
        return err;
    };
    defer config.deinit();

    if (std.mem.eql(u8, config.mode, "test")) {
        runKgruTest(allocator) catch |err| {
            if (err == error.TestsFailed) {
                return err;
            }
            return err;
        };
        return;
    } else if (std.mem.eql(u8, config.mode, "train")) {
        try runTraining(allocator, &config);
        return;
    } else if (std.mem.eql(u8, config.mode, "validate")) {
        try runValidation(allocator, &config);
        return;
    }

    var mgt = try initTokenizer(allocator);
    defer mgt.deinit();

    var ssi = SSI.init(allocator);
    defer ssi.deinit();

    var ranker = try Ranker.init(allocator, MainConfig.RANKER_NGRAM_SIZE, MainConfig.RANKER_LSH_TABLES, MainConfig.RANKER_SEED);
    defer ranker.deinit();

    try runInteractiveREPL(allocator, &mgt, &ssi, &ranker, &config);
}

fn printHelp() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(
        \\JAIDE v40 - Root-Level LLM
        \\
        \\Usage: jaide [OPTIONS]
        \\
        \\Options:
        \\  --mode <MODE>         Operating mode: test, train, validate, or interactive (default)
        \\  --embedding-dim <N>   Embedding dimension (default: 128, range: 8-16384)
        \\  --layers <N>          Number of RSF layers (default: 4, range: 1-256)
        \\  --batch-size <N>      Batch size for training (default: 16, range: 1-4096)
        \\  --epochs <N>          Number of training epochs (default: 10, max: 100000)
        \\  --lr <RATE>           Learning rate (default: 0.001, range: 1e-10 to 10.0)
        \\  --samples <N>         Number of training samples (default: 100)
        \\  --models-dir <PATH>   Directory for model files (default: models)
        \\  --vocab-file <PATH>   Path to vocabulary file
        \\  --dataset-path <PATH> Path to training dataset (JSONL format)
        \\  --sample-limit <N>    Maximum samples to load from dataset
        \\  --gradient-clip <N>   Gradient clipping norm (default: 5.0)
        \\  --sequence-length <N> Sequence length for training (default: 64)
        \\  --top-k <N>           Top-K results for retrieval (default: 5)
        \\  --help                Show this help message
        \\
    );
}

fn initTokenizer(allocator: std.mem.Allocator) !MGT {
    const sample_vocab = [_][]const u8{
        "a", "az", "es", "is", "nem", "de", "hogy", "egy", "mert", "vagy",
        "minden", "csak", "meg", "mar", "most", "itt", "ott", "ki", "mi", "ez",
        "neural", "network", "learning", "deep", "machine", "intelligence", "artificial",
        "data", "model", "training", "optimization", "algorithm", "computer", "science",
        "mesterseges", "intelligencia", "neuralis", "halozat", "tanulas", "gepi",
        "adattudomany", "optimalizalas", "algoritmus", "kvantum", "robotika",
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "weights", "bias", "matrix", "vector", "scalar", "tensor", "gradient", "descent",
        "forward", "backward", "epoch", "batch", "loss", "accuracy", "precision", "recall",
        "f1", "score", "roc", "auc", "confusion", "matrix", "classification", "regression",
        "clustering", "anomaly", "detection", "reinforcement", "policy", "value", "actor",
        "critic", "agent", "environment", "reward", "state", "action", "transition",
        "probability", "distribution", "normal", "gaussian", "uniform", "poisson", "binomial",
        "sampling", "monte", "carlo", "markov", "chain", "decision", "process", "bellman",
        "equation", "dynamic", "programming", "optimal", "control", "theory", "graph",
        "node", "edge", "vertex", "path", "cycle", "tree", "forest", "directed", "undirected",
        "weighted", "bipartite", "connected", "component", "spanning", "shortest", "longest",
    };

    const sample_anchors = [_][]const u8{ "a", "az", "es", "the", "neural", "mesterseges" };

    return try MGT.init(allocator, &sample_vocab, &sample_anchors);
}

fn calculateTotalParams(dim: usize, layers: usize) error{Overflow}!usize {
    const dim_squared = std.math.mul(usize, dim, dim) catch return error.Overflow;
    const weights_per_layer = std.math.mul(usize, dim_squared, 2) catch return error.Overflow;
    const biases_per_layer = std.math.mul(usize, dim, 2) catch return error.Overflow;
    const params_per_layer = std.math.add(usize, weights_per_layer, biases_per_layer) catch return error.Overflow;
    return std.math.mul(usize, params_per_layer, layers) catch return error.Overflow;
}

fn runTraining(allocator: std.mem.Allocator, config: *const Config) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("Initializing training...\n");

    const dim = config.embedding_dim;
    const layers = config.rsf_layers;

    const total_params = try calculateTotalParams(dim, layers);
    try stdout.print("Model parameters: {d}\n", .{total_params});

    var rsf = try RSF.init(allocator, dim, layers);
    defer rsf.deinit();

    var mgt = try initTokenizer(allocator);
    defer mgt.deinit();

    const vocab_size = mgt.vocabSize();
    
    var proj = try Projection.init(allocator, dim * 2, vocab_size);
    defer proj.deinit();

    try stdout.print("Projection layer: hidden_dim={d}, vocab_size={d}\n", .{ dim * 2, vocab_size });

    var samples: []TrainingSample = undefined;
    
    if (config.dataset_path) |dataset_path| {
        try stdout.print("Loading dataset from: {s}\n", .{dataset_path});
        samples = loadDatasetSamples(allocator, &mgt, dataset_path, config.sample_limit) catch |err| blk: {
            try stdout.print("Dataset load failed ({any}), using synthetic samples\n", .{err});
            break :blk try generateSyntheticSamples(allocator, &mgt, config.num_training_samples);
        };
        try stdout.print("Loaded {d} samples from dataset\n", .{samples.len});
    } else {
        samples = try generateSyntheticSamples(allocator, &mgt, config.num_training_samples);
        try stdout.print("Generated {d} synthetic samples\n", .{samples.len});
    }
    defer {
        for (samples) |*sample| {
            sample.deinit(allocator);
        }
        allocator.free(samples);
    }

    const tensor_dim = dim * 2;
    var input = try Tensor.init(allocator, &.{ 1, tensor_dim });
    defer input.deinit();

    var logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);

    var probs = try allocator.alloc(f32, vocab_size);
    defer allocator.free(probs);

    var hidden_grad = try allocator.alloc(f32, dim * 2);
    defer allocator.free(hidden_grad);

    var epoch: usize = 0;
    while (epoch < config.num_epochs) : (epoch += 1) {
        var epoch_loss: f64 = 0.0;
        var batch_count: usize = 0;

        var batch_start: usize = 0;
        while (batch_start < samples.len) : (batch_start += config.batch_size) {
            const batch_end = @min(batch_start + config.batch_size, samples.len);
            const batch = samples[batch_start..batch_end];

            @memset(hidden_grad, 0.0);

            var batch_loss: f64 = 0.0;
            for (batch) |sample| {
                if (sample.tokens.len < 2) continue;

                const context_len = @min(sample.tokens.len - 1, dim);
                try createEmbeddingInPlace(&input, sample.tokens[0..context_len], mgt.vocabSize(), dim);
                
                rsf.forward(&input) catch return error.RSFForwardFailed;

                proj.forward(&input, logits);

                const target_token = sample.tokens[context_len];
                var max_logit: f32 = logits[0];
                for (logits) |l| {
                    if (l > max_logit) max_logit = l;
                }

                var sum_exp: f64 = 0.0;
                var li: usize = 0;
                while (li < logits.len) : (li += 1) {
                    const exp_val = std.math.exp(@as(f64, logits[li] - max_logit));
                    probs[li] = @floatCast(exp_val);
                    sum_exp += exp_val;
                }

                li = 0;
                while (li < probs.len) : (li += 1) {
                    probs[li] /= @floatCast(sum_exp);
                }

                const target_idx = @min(@as(usize, target_token), logits.len - 1);
                const log_prob = @as(f64, logits[target_idx] - max_logit) - std.math.log(f64, std.math.e, sum_exp);
                batch_loss -= log_prob;

                probs[target_idx] -= 1.0;

                const lr: f32 = config.learning_rate;
                const h_dim = @min(input.data.len, proj.hidden_dim);
                
                var vi: usize = 0;
                while (vi < vocab_size) : (vi += 1) {
                    const grad_v = probs[vi];
                    proj.bias.data[vi] -= lr * grad_v;

                    var hi: usize = 0;
                    while (hi < h_dim) : (hi += 1) {
                        const w_idx = hi * vocab_size + vi;
                        if (w_idx < proj.weights.data.len) {
                            proj.weights.data[w_idx] -= lr * grad_v * input.data[hi];
                        }
                    }
                }

                vi = 0;
                while (vi < vocab_size) : (vi += 1) {
                    var hi: usize = 0;
                    while (hi < h_dim) : (hi += 1) {
                        const w_idx = hi * vocab_size + vi;
                        if (w_idx < proj.weights.data.len) {
                            hidden_grad[hi] += probs[vi] * proj.weights.data[w_idx];
                        }
                    }
                }
                
                const clip_norm = config.gradient_clip_norm;
                var grad_sq_sum: f32 = 0.0;
                for (hidden_grad) |g| grad_sq_sum += g * g;
                if (grad_sq_sum > clip_norm * clip_norm) {
                    const scale = clip_norm / std.math.sqrt(grad_sq_sum);
                    var gi: usize = 0;
                    while (gi < hidden_grad.len) : (gi += 1) {
                        hidden_grad[gi] *= scale;
                    }
                }

                var layer_idx: i64 = @intCast(rsf.ctrl.layers.len - 1);
                while (layer_idx >= 0) : (layer_idx -= 1) {
                    const l_idx: usize = @intCast(layer_idx);
                    const layer = &rsf.ctrl.layers[l_idx];
                    const layer_dim = rsf.ctrl.dim;
                    
                    var di: usize = 0;
                    while (di < layer_dim) : (di += 1) {
                        const grad_d = if (di < hidden_grad.len) hidden_grad[di] else 0.0;
                        
                        if (di < layer.s_bias.data.len) {
                            layer.s_bias.data[di] -= lr * grad_d;
                        }
                        if (di < layer.t_bias.data.len) {
                            layer.t_bias.data[di] -= lr * grad_d;
                        }

                        var dj: usize = 0;
                        while (dj < layer_dim) : (dj += 1) {
                            const input_j = if (dj < input.data.len) input.data[dj] else 0.0;
                            const w_idx = di * layer_dim + dj;
                            if (w_idx < layer.s_weight.data.len) {
                                layer.s_weight.data[w_idx] -= lr * grad_d * input_j;
                            }
                            if (w_idx < layer.t_weight.data.len) {
                                layer.t_weight.data[w_idx] -= lr * grad_d * input_data_dj: {
                                    const d_val = if (di < input.data.len) input.data[di] else 0.0;
                                    break :input_data_dj d_val;
                                };
                            }
                        }
                    }
                    
                    if (layer_idx > 0) {
                        @memset(hidden_grad, 0.0);
                    }
                }
            }

            epoch_loss += batch_loss;
            batch_count += 1;
        }

        const avg_loss = if (batch_count > 0) epoch_loss / @as(f64, @floatFromInt(batch_count)) else 0.0;
        try stdout.print("Epoch {d}/{d} - Loss: {d:.6}\n", .{ epoch + 1, config.num_epochs, avg_loss });
    }

    try std.fs.cwd().makePath(config.models_dir);

    const rsf_path = try std.fmt.allocPrint(allocator, "{s}/rsf_trained.bin", .{config.models_dir});
    defer allocator.free(rsf_path);
    try saveRSF(&rsf, rsf_path);

    const mgt_path = try std.fmt.allocPrint(allocator, "{s}/mgt_vocab.bin", .{config.models_dir});
    defer allocator.free(mgt_path);
    try saveMGT(&mgt, mgt_path);

    const proj_path = try std.fmt.allocPrint(allocator, "{s}/projection.bin", .{config.models_dir});
    defer allocator.free(proj_path);
    try proj.save(proj_path);

    try stdout.writeAll("Training complete. Models saved.\n");
}

fn runValidation(allocator: std.mem.Allocator, config: *const Config) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("Running validation...\n");

    const dim = config.embedding_dim;
    const layers = config.rsf_layers;

    var rsf = try RSF.init(allocator, dim, layers);
    defer rsf.deinit();

    var mgt = try initTokenizer(allocator);
    defer mgt.deinit();

    const metrics = try validateModel(allocator, &rsf, &mgt, config);
    try printValidationMetrics(stdout, &metrics);
}

fn generateSyntheticSamples(allocator: std.mem.Allocator, mgt: *MGT, count: usize) ![]TrainingSample {
    var samples = std.ArrayList(TrainingSample).init(allocator);
    errdefer {
        for (samples.items) |*sample| {
            sample.deinit(allocator);
        }
        samples.deinit();
    }

    const base_texts = [_][]const u8{
        "A mesterseges intelligencia a jovo kulcsa.",
        "Az adattudomany es gepi tanulas osszekapcsolodik.",
        "A neuralis halozatok komplex mintakat ismernek fel.",
        "Az automatizalas noveli a termelekenyseget.",
        "A kvantumszamitogepek uj lehetosegeket nyitnak.",
        "Az algoritmusok optimalizaljak a donteshozatalt.",
        "A termeszetes nyelvfeldolgozas emberi kommunikaciot ertelmez.",
        "A szamitogepek latas kepeket es videokat elemez.",
        "A robotika es automatizalas atalakitja az ipart.",
        "Az etikus AI fejlesztes fontos tarsadalmi kerdes.",
        "Deep learning requires large datasets.",
        "Gradient descent optimizes the loss function.",
        "The attention mechanism allows the model to focus.",
        "Transformers have revolutionized natural language processing.",
        "Backpropagation computes gradients efficiently.",
    };

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const base_text = base_texts[i % base_texts.len];
        const text_copy = try allocator.dupe(u8, base_text);
        errdefer allocator.free(text_copy);

        var tokens_list = std.ArrayList(u32).init(allocator);
        errdefer tokens_list.deinit();

        try mgt.encode(base_text, &tokens_list);

        const tokens = try tokens_list.toOwnedSlice();
        errdefer allocator.free(tokens);

        try samples.append(.{
            .text = text_copy,
            .tokens = tokens,
        });
    }

    return samples.toOwnedSlice();
}

fn loadDatasetSamples(allocator: std.mem.Allocator, mgt: *MGT, dataset_path: []const u8, limit: usize) ![]TrainingSample {
    var samples = std.ArrayList(TrainingSample).init(allocator);
    errdefer {
        for (samples.items) |*sample| {
            sample.deinit(allocator);
        }
        samples.deinit();
    }

    const file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| {
        std.debug.print("Cannot open dataset file: {s}, error: {any}\n", .{ dataset_path, err });
        return err;
    };
    defer file.close();

    var buf_reader = std.io.bufferedReader(file.reader());
    var reader = buf_reader.reader();

    var line_buf = try allocator.alloc(u8, MainConfig.MAX_LINE_LENGTH);
    defer allocator.free(line_buf);

    var lines_read: usize = 0;
    const actual_limit = @min(limit, MainConfig.MAX_SAMPLES);

    while (lines_read < actual_limit) {
        const line = reader.readUntilDelimiterOrEof(line_buf, '\n') catch |err| {
            if (err == error.EndOfStream) break;
            if (err == error.StreamTooLong) {
                try reader.skipUntilDelimiterOrEof('\n');
                continue;
            }
            continue;
        };

        if (line == null or line.?.len < 2) continue;
        
        const clean_line = std.mem.trim(u8, line.?, " \t\r\n");
        if (clean_line.len == 0) continue;

        const text_copy = allocator.dupe(u8, clean_line) catch continue;
        errdefer allocator.free(text_copy);

        var tokens_list = std.ArrayList(u32).init(allocator);
        
        mgt.encode(clean_line, &tokens_list) catch {
            allocator.free(text_copy);
            tokens_list.deinit();
            continue;
        };

        if (tokens_list.items.len == 0) {
            allocator.free(text_copy);
            tokens_list.deinit();
            continue;
        }

        const tokens = tokens_list.toOwnedSlice() catch {
            allocator.free(text_copy);
            tokens_list.deinit();
            continue;
        };

        samples.append(.{
            .text = text_copy,
            .tokens = tokens,
        }) catch {
            allocator.free(text_copy);
            allocator.free(tokens);
            continue;
        };

        lines_read += 1;
    }

    if (samples.items.len == 0) {
        return error.EmptyDataset;
    }

    return samples.toOwnedSlice();
}

fn createEmbeddingInPlace(embedding: *Tensor, tokens: []const u32, vocab_size: usize, dim: usize) !void {
    if (embedding.data.len == 0) return;
    if (vocab_size == 0) {
        @memset(embedding.data, 0.0);
        return;
    }

    var i: usize = 0;
    while (i < embedding.data.len) : (i += 1) {
        embedding.data[i] = 0.0;
    }

    const max_tokens = @min(tokens.len, dim);
    i = 0;
    while (i < max_tokens) : (i += 1) {
        const token_id = tokens[i];
        const vocab_f: f32 = @floatFromInt(vocab_size);
        const token_f: f32 = @floatFromInt(token_id);
        const normalized = std.math.clamp(token_f / vocab_f, 0.0, 1.0);

        if (i < embedding.data.len) {
            embedding.data[i] = normalized;
        }
        if (i + dim < embedding.data.len) {
            embedding.data[i + dim] = normalized * 0.5;
        }
    }
}

fn validateModel(allocator: std.mem.Allocator, rsf: *RSF, mgt: *MGT, config: *const Config) !ValidationMetrics {
    const n_samples = @max(config.num_validation_samples, 1);

    var predictions = try allocator.alloc(f32, n_samples);
    defer allocator.free(predictions);

    var targets = try allocator.alloc(f32, n_samples);
    defer allocator.free(targets);

    const tensor_dim = config.embedding_dim * 2;
    var input = try Tensor.init(allocator, &.{ 1, tensor_dim });
    defer input.deinit();

    var prng = PRNG.init(MainConfig.PRNG_SEED_VALIDATION);
    const dim = config.embedding_dim;
    
    var synthetic_tokens = try allocator.alloc(u32, dim);
    defer allocator.free(synthetic_tokens);

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        var ti: usize = 0;
        while (ti < dim) : (ti += 1) {
            synthetic_tokens[ti] = @as(u32, @intCast(prng.next() % mgt.vocabSize()));
        }

        try createEmbeddingInPlace(&input, synthetic_tokens, mgt.vocabSize(), dim);
        try rsf.forward(&input);

        var sum: f64 = 0.0;
        for (input.data) |val| {
            if (!std.math.isNan(val) and !std.math.isInf(val)) {
                sum += @as(f64, val);
            }
        }
        predictions[i] = @floatCast(sum / @as(f64, @floatFromInt(@max(input.data.len, 1))));

        const next_token = synthetic_tokens[0];
        const token_f: f32 = @floatFromInt(next_token);
        const vocab_f: f32 = @floatFromInt(mgt.vocabSize());
        targets[i] = token_f / vocab_f;
    }

    var sum_pred: f64 = 0.0;
    var sum_target: f64 = 0.0;
    for (predictions) |p| sum_pred += @as(f64, p);
    for (targets) |t| sum_target += @as(f64, t);

    const n_f64 = @as(f64, @floatFromInt(n_samples));
    const mean_pred: f32 = @floatCast(sum_pred / n_f64);
    const mean_target: f32 = @floatCast(sum_target / n_f64);

    var mse: f64 = 0.0;
    var mae: f64 = 0.0;
    var ss_res: f64 = 0.0;
    var ss_tot: f64 = 0.0;

    i = 0;
    while (i < n_samples) : (i += 1) {
        const pred_f64 = @as(f64, predictions[i]);
        const target_f64 = @as(f64, targets[i]);
        const diff = pred_f64 - target_f64;
        mse += diff * diff;
        mae += std.math.fabs(diff);
        ss_res += diff * diff;
        const target_diff = target_f64 - @as(f64, mean_target);
        ss_tot += target_diff * target_diff;
    }

    mse /= n_f64;
    mae /= n_f64;

    const rmse = std.math.sqrt(mse);
    const r_squared = if (ss_tot > MainConfig.R_SQUARED_EPSILON) 1.0 - (ss_res / ss_tot) else 0.0;

    const std_err = rmse / std.math.sqrt(n_f64);
    const margin = MainConfig.CONFIDENCE_Z_SCORE * std_err;

    return ValidationMetrics{
        .mse = @floatCast(mse),
        .rmse = @floatCast(rmse),
        .mae = @floatCast(mae),
        .r_squared = @floatCast(r_squared),
        .mean_prediction = mean_pred,
        .mean_target = mean_target,
        .confidence_interval_lower = @floatCast(mse - margin),
        .confidence_interval_upper = @floatCast(mse + margin),
        .num_samples = n_samples,
    };
}

fn printValidationMetrics(writer: anytype, metrics: *const ValidationMetrics) !void {
    try writer.print("Validation Metrics (n={d}):\n", .{metrics.num_samples});
    try writer.print("  MSE: {d:.8}\n", .{metrics.mse});
    try writer.print("  RMSE: {d:.8}\n", .{metrics.rmse});
    try writer.print("  MAE: {d:.8}\n", .{metrics.mae});
    try writer.print("  R2 Score: {d:.6}\n", .{metrics.r_squared});
    try writer.print("  Mean Prediction: {d:.6}\n", .{metrics.mean_prediction});
    try writer.print("  Mean Target: {d:.6}\n", .{metrics.mean_target});
    try writer.print("  95% CI: [{d:.8}, {d:.8}]\n", .{ metrics.confidence_interval_lower, metrics.confidence_interval_upper });
}

fn saveRSF(rsf: *const RSF, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var buf_writer = std.io.bufferedWriter(file.writer());
    const writer = buf_writer.writer();

    try writer.writeInt(u32, MainConfig.FILE_MAGIC_RSF, .Little);
    try writer.writeInt(u32, MainConfig.FILE_VERSION, .Little);
    try writer.writeInt(u64, @as(u64, @intCast(rsf.ctrl.num_layers)), .Little);
    try writer.writeInt(u64, @as(u64, @intCast(rsf.ctrl.dim)), .Little);

    var l: usize = 0;
    while (l < rsf.ctrl.num_layers) : (l += 1) {
        const layer = &rsf.ctrl.layers[l];
        try layer.s_weight.save(writer);
        try layer.t_weight.save(writer);
        try layer.s_bias.save(writer);
        try layer.t_bias.save(writer);
    }
    
    try buf_writer.flush();
}

fn saveMGT(mgt: *const MGT, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var buf_writer = std.io.bufferedWriter(file.writer());
    const writer = buf_writer.writer();

    try writer.writeInt(u32, MainConfig.FILE_MAGIC_MGT, .Little);
    try writer.writeInt(u32, MainConfig.FILE_VERSION, .Little);

    const vocab_size = mgt.vocabSize();
    if (vocab_size > MainConfig.MAX_VOCAB_SIZE) return error.VocabTooLarge;
    try writer.writeInt(u32, @as(u32, @intCast(vocab_size)), .Little);

    const VocabEntry = struct { key: []const u8, val: u32 };

    var sorted_entries = std.ArrayList(VocabEntry).init(mgt.allocator);
    defer sorted_entries.deinit();

    var it = mgt.token_to_id.iterator();
    while (it.next()) |entry| {
        try sorted_entries.append(.{ .key = entry.key_ptr.*, .val = entry.value_ptr.* });
    }

    const SortContext = struct {
        fn lessThan(_: void, a: VocabEntry, b: VocabEntry) bool {
            return a.val < b.val;
        }
    };
    std.mem.sort(VocabEntry, sorted_entries.items, {}, SortContext.lessThan);

    for (sorted_entries.items) |entry| {
        const token = entry.key;
        const id = entry.val;
        if (token.len > MainConfig.MAX_TOKEN_LENGTH) return error.TokenTooLong;
        try writer.writeInt(u32, @as(u32, @intCast(token.len)), .Little);
        try writer.writeAll(token);
        try writer.writeInt(u32, id, .Little);
    }

    try buf_writer.flush();
}

const Projection = struct {
    weights: Tensor,
    bias: Tensor,
    hidden_dim: usize,
    vocab_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, hidden_dim: usize, vocab_sz: usize) !Projection {
        const fan_in = @as(f32, @floatFromInt(hidden_dim));
        const fan_out = @as(f32, @floatFromInt(vocab_sz));
        const xavier_std = std.math.sqrt(2.0 / (fan_in + fan_out));

        var weights = try Tensor.randomUniform(alloc, &.{ hidden_dim, vocab_sz }, -xavier_std, xavier_std, 12345);
        errdefer weights.deinit();

        var bias = try Tensor.zeros(alloc, &.{vocab_sz});
        errdefer bias.deinit();

        return Projection{
            .weights = weights,
            .bias = bias,
            .hidden_dim = hidden_dim,
            .vocab_size = vocab_sz,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *Projection) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: *const Projection, hidden: *const Tensor, output: []f32) void {
        const h_dim = @min(hidden.data.len, self.hidden_dim);
        var v: usize = 0;
        while (v < output.len) : (v += 1) {
            var sum: f32 = self.bias.data[v];
            var h: usize = 0;
            while (h < h_dim) : (h += 1) {
                const w_idx = h * self.vocab_size + v;
                if (w_idx < self.weights.data.len) {
                    sum += hidden.data[h] * self.weights.data[w_idx];
                }
            }
            output[v] = sum;
        }
    }

    pub fn save(self: *const Projection, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        
        var buf_writer = std.io.bufferedWriter(file.writer());
        const writer = buf_writer.writer();

        try writer.writeInt(u32, MainConfig.FILE_MAGIC_PROJ, .Little);
        try writer.writeInt(u32, MainConfig.FILE_VERSION, .Little);
        try writer.writeInt(u64, @as(u64, @intCast(self.hidden_dim)), .Little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .Little);

        for (self.weights.data) |w| {
            try writer.writeInt(u32, @as(u32, @bitCast(w)), .Little);
        }
        for (self.bias.data) |b| {
            try writer.writeInt(u32, @as(u32, @bitCast(b)), .Little);
        }
        
        try buf_writer.flush();
    }

    pub fn load(alloc: std.mem.Allocator, path: []const u8) !Projection {
        const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
        defer file.close();
        
        var buf_reader = std.io.bufferedReader(file.reader());
        const reader = buf_reader.reader();

        const magic = try reader.readInt(u32, .Little);
        if (magic != MainConfig.FILE_MAGIC_PROJ) return error.InvalidFormat;
        _ = try reader.readInt(u32, .Little);

        const hidden_dim = @as(usize, @intCast(try reader.readInt(u64, .Little)));
        const vocab_sz = @as(usize, @intCast(try reader.readInt(u64, .Little)));

        const w_size = hidden_dim * vocab_sz;
        var weights = try Tensor.init(alloc, &.{ hidden_dim, vocab_sz });
        errdefer weights.deinit();

        var wi: usize = 0;
        while (wi < w_size) : (wi += 1) {
            weights.data[wi] = @bitCast(try reader.readInt(u32, .Little));
        }

        var bias = try Tensor.init(alloc, &.{vocab_sz});
        errdefer bias.deinit();

        var bi: usize = 0;
        while (bi < vocab_sz) : (bi += 1) {
            bias.data[bi] = @bitCast(try reader.readInt(u32, .Little));
        }

        return Projection{
            .weights = weights,
            .bias = bias,
            .hidden_dim = hidden_dim,
            .vocab_size = vocab_sz,
            .allocator = alloc,
        };
    }
};

fn loadRSFWeights(rsf: *RSF, path: []const u8) !void {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();
    
    var buf_reader = std.io.bufferedReader(file.reader());
    const reader = buf_reader.reader();

    const magic = try reader.readInt(u32, .Little);
    if (magic != MainConfig.FILE_MAGIC_RSF) return error.InvalidFormat;
    _ = try reader.readInt(u32, .Little);

    const num_layers = @as(usize, @intCast(try reader.readInt(u64, .Little)));
    const dim = @as(usize, @intCast(try reader.readInt(u64, .Little)));

    if (num_layers != rsf.ctrl.num_layers or dim != rsf.ctrl.dim) return error.DimensionMismatch;

    var l: usize = 0;
    while (l < num_layers) : (l += 1) {
        const layer = &rsf.ctrl.layers[l];
        var wi: usize = 0;
        while (wi < layer.s_weight.data.len) : (wi += 1) {
            layer.s_weight.data[wi] = @bitCast(try reader.readInt(u32, .Little));
        }
        wi = 0;
        while (wi < layer.t_weight.data.len) : (wi += 1) {
            layer.t_weight.data[wi] = @bitCast(try reader.readInt(u32, .Little));
        }
        wi = 0;
        while (wi < layer.s_bias.data.len) : (wi += 1) {
            layer.s_bias.data[wi] = @bitCast(try reader.readInt(u32, .Little));
        }
        wi = 0;
        while (wi < layer.t_bias.data.len) : (wi += 1) {
            layer.t_bias.data[wi] = @bitCast(try reader.readInt(u32, .Little));
        }
    }
}

fn sampleTopK(logits: []f32, k_param: usize, prng: *PRNG) !u32 {
    var k = k_param;
    if (logits.len == 0) return 0;
    if (k >= logits.len) k = logits.len;
    
    var indices = std.ArrayList(usize).init(std.heap.page_allocator);
    defer indices.deinit();
    
    indices.resize(logits.len) catch unreachable;
    for (0..logits.len) |i| {
        indices.items[i] = i;
    }

    const SortContext = struct {
        logits: []const f32,
        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.logits[a] > ctx.logits[b];
        }
    };
    
    std.mem.sort(usize, indices.items, SortContext{ .logits = logits }, SortContext.lessThan);
    
    const top_k_indices = indices.items[0..k];
    
    var top_k_logits = try std.heap.page_allocator.alloc(f32, k);
    defer std.heap.page_allocator.free(top_k_logits);
    
    var max_logit: f32 = logits[top_k_indices[0]];
    var i: usize = 0;
    while (i < k) : (i += 1) {
        if (logits[top_k_indices[i]] > max_logit) {
            max_logit = logits[top_k_indices[i]];
        }
    }
    
    var sum_exp: f64 = 0.0;
    i = 0;
    while (i < k) : (i += 1) {
        const val = std.math.exp(@as(f64, logits[top_k_indices[i]] - max_logit));
        top_k_logits[i] = @floatCast(val);
        sum_exp += val;
    }
    
    i = 0;
    while (i < k) : (i += 1) {
        top_k_logits[i] /= @floatCast(sum_exp);
    }
    
    var r = @as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(std.math.maxInt(u64)));
    var cumulative: f64 = 0.0;
    
    i = 0;
    while (i < k - 1) : (i += 1) {
        cumulative += @as(f64, top_k_logits[i]);
        if (r < cumulative) {
            return @intCast(top_k_indices[i]);
        }
    }
    
    return @intCast(top_k_indices[k - 1]);
}

fn generateText(allocator: std.mem.Allocator, rsf: *RSF, proj: *const Projection, mgt: *MGT, prompt_tokens: []const u32, max_len: usize, config: *const Config) ![]u8 {
    const dim = config.embedding_dim;
    const tensor_dim = dim * 2;

    var input = try Tensor.init(allocator, &.{ 1, tensor_dim });
    defer input.deinit();

    var generated = std.ArrayList(u32).init(allocator);
    defer generated.deinit();

    try generated.appendSlice(prompt_tokens);

    var logits = try allocator.alloc(f32, proj.vocab_size);
    defer allocator.free(logits);
    
    var prng = PRNG.init(MainConfig.PRNG_SEED_SYNTHETIC);

    var step: usize = 0;
    while (step < max_len) : (step += 1) {
        const context_start = if (generated.items.len > dim) generated.items.len - dim else 0;
        const context = generated.items[context_start..];
        try createEmbeddingInPlace(&input, context, mgt.vocabSize(), dim);
        try rsf.forward(&input);

        proj.forward(&input, logits);
        
        const next_token = try sampleTopK(logits, config.top_k, &prng);

        try generated.append(next_token);
    }

    var output = std.ArrayList(u8).init(allocator);
    errdefer output.deinit();

    try mgt.decode(generated.items, &output);
    return output.toOwnedSlice();
}

fn runInteractiveREPL(allocator: std.mem.Allocator, mgt: *MGT, _: *SSI, _: *Ranker, config: *const Config) !void {
    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();
    const stdin = std.io.getStdIn().reader();

    const dim = config.embedding_dim;
    const layers = config.rsf_layers;
    const vocab_size = mgt.vocabSize();

    var rsf = RSF.init(allocator, dim, layers) catch |err| {
        try stdout.print("Failed to initialize RSF: {any}\n", .{err});
        return err;
    };
    defer rsf.deinit();

    var proj = Projection.init(allocator, dim * 2, vocab_size) catch |err| {
        try stdout.print("Failed to initialize projection: {any}\n", .{err});
        return err;
    };
    defer proj.deinit();

    const rsf_path = try std.fmt.allocPrint(allocator, "{s}/rsf_trained.bin", .{config.models_dir});
    defer allocator.free(rsf_path);
    const proj_path = try std.fmt.allocPrint(allocator, "{s}/projection.bin", .{config.models_dir});
    defer allocator.free(proj_path);

    var model_loaded = false;
    if (loadRSFWeights(&rsf, rsf_path)) {
        if (Projection.load(allocator, proj_path)) |loaded_proj| {
            proj.deinit();
            proj = loaded_proj;
            model_loaded = true;
            try stdout.writeAll("Trained model loaded. Using generative mode.\n");
        } else |_| {
            try stdout.writeAll("WARNING: No projection weights found. Using random init.\n");
        }
    } else |_| {
        try stdout.writeAll("WARNING: No trained RSF model found. Using random init.\n");
    }

    try stdout.print("Model: dim={d}, layers={d}, vocab={d}\n", .{ dim, layers, vocab_size });
    try stdout.writeAll("READY\n");

    var line_buf = try allocator.alloc(u8, MainConfig.REPL_LINE_BUFFER_SIZE);
    defer allocator.free(line_buf);

    while (true) {
        try stdout.print("> ", .{});
        // try stdout.flush();

        const line = stdin.readUntilDelimiterOrEof(line_buf, '\n') catch |err| {
            try stdout.print("Input error: {any}.\n", .{err});
            continue;
        };

        if (line == null) break;

        const input = std.mem.trim(u8, line.?, " \t\r\n");
        if (input.len == 0) continue;

        if (std.mem.eql(u8, input, "exit") or std.mem.eql(u8, input, "quit")) {
            try stdout.writeAll("Goodbye.\n");
            break;
        }

        if (std.mem.eql(u8, input, "help")) {
            try stdout.writeAll(
                \\Commands:
                \\  help   - Show this help
                \\  status - Show system status
                \\  exit   - Exit the program
                \\  <text> - Generate text based on input
                \\
            );
            continue;
        }

        if (std.mem.eql(u8, input, "status")) {
            try stdout.print("Model: dim={d}, layers={d}, vocab={d} | Loaded: {s}\n", .{
                dim,
                layers,
                vocab_size,
                if (model_loaded) "yes" else "no",
            });
            continue;
        }

        var query_tokens = std.ArrayList(u32).init(allocator);
        defer query_tokens.deinit();

        mgt.encode(input, &query_tokens) catch |err| {
            try stdout.print("Tokenization error: {any}\n", .{err});
            continue;
        };

        if (query_tokens.items.len == 0) {
            try stdout.writeAll("Empty query after tokenization.\n");
            continue;
        }

        const response = generateText(allocator, &rsf, &proj, mgt, query_tokens.items, config.sequence_length, config) catch |err| {
            try stdout.print("Generation error: {any}\n", .{err});
            continue;
        };
        defer allocator.free(response);

        try stdout.print("{s}\n", .{response});
    }
}
