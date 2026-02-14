// WASM dependency wrapper - provides access to all WASM-needed modules
// This avoids circular dependency issues by being the single entry point

pub const MGT = @import("tokenizer/mgt.zig").MGT;
pub const RSF = @import("processor/rsf.zig").RSF;
pub const Tensor = @import("core/tensor.zig").Tensor;
pub const ModelFormat = @import("core/model_io.zig").ModelFormat;
pub const importModel = @import("core/model_io.zig").importModel;
pub const types = @import("core/types.zig");
pub const memory = @import("core/memory.zig");
