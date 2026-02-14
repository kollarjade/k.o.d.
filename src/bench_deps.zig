// Wrapper module for benchmark dependencies
// Re-exports all needed types to avoid circular dependency issues with anonymous modules

pub const ssi = @import("index/ssi.zig");
pub const rsf = @import("processor/rsf.zig");
pub const tensor = @import("core/tensor.zig");
pub const types = @import("core/types.zig");

// Re-export commonly used types
pub const SSI = ssi.SSI;
pub const RSF = rsf.RSF;
pub const Tensor = tensor.Tensor;
pub const RankedSegment = types.RankedSegment;
