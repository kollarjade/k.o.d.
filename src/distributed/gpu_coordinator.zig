const std = @import("std");
const nccl = @import("nccl_bindings.zig");
const Allocator = std.mem.Allocator;

pub const GPUCoordinator = struct {
    allocator: Allocator,
    world_size: usize,
    rank: usize,
    device_id: i32,
    nccl_comm: *nccl.ncclComm,
    cuda_stream: *anyopaque,

    pub fn init(allocator: Allocator, world_size: usize, rank: usize, nccl_id: nccl.ncclUniqueId) !GPUCoordinator {
        var device_count: c_int = 0;
        var cuda_err = nccl.cudaGetDeviceCount(&device_count);
        if (cuda_err != .cudaSuccess) {
            return error.CudaGetDeviceCountFailed;
        }

        if (device_count < world_size) {
            std.debug.print("Error: Only {d} GPUs available, need {d}\n", .{device_count, world_size});
            return error.InsufficientGPUs;
        }

        const device_id: i32 = @intCast(rank);
        cuda_err = nccl.cudaSetDevice(device_id);
        if (cuda_err != .cudaSuccess) {
            return error.CudaSetDeviceFailed;
        }

        var nccl_comm: *nccl.ncclComm = undefined;
        const nccl_err = nccl.ncclCommInitRank(&nccl_comm, @intCast(world_size), nccl_id, @intCast(rank));
        if (nccl_err != .ncclSuccess) {
            const err_str = nccl.ncclGetErrorString(nccl_err);
            std.debug.print("NCCL Error: {s}\n", .{err_str});
            return error.NCCLCommInitFailed;
        }

        var cuda_stream: *anyopaque = undefined;
        cuda_err = nccl.cudaStreamCreate(&cuda_stream);
        if (cuda_err != .cudaSuccess) {
            return error.CudaStreamCreateFailed;
        }

        return GPUCoordinator{
            .allocator = allocator,
            .world_size = world_size,
            .rank = rank,
            .device_id = device_id,
            .nccl_comm = nccl_comm,
            .cuda_stream = cuda_stream,
        };
    }

    pub fn deinit(self: *GPUCoordinator) void {
        _ = nccl.cudaStreamDestroy(self.cuda_stream);
        _ = nccl.ncclCommDestroy(self.nccl_comm);
    }

    pub fn allocDeviceMemory(self: *GPUCoordinator, size: usize) !*anyopaque {
        _ = self;
        var dev_ptr: ?*anyopaque = null;
        const err = nccl.cudaMalloc(&dev_ptr, size);
        if (err != .cudaSuccess) {
            return error.CudaMallocFailed;
        }
        return dev_ptr.?;
    }

    pub fn freeDeviceMemory(self: *GPUCoordinator, ptr: *anyopaque) void {
        _ = self;
        _ = nccl.cudaFree(ptr);
    }

    pub fn copyHostToDevice(self: *GPUCoordinator, dst: *anyopaque, src: *const anyopaque, size: usize) !void {
        _ = self;
        const err = nccl.cudaMemcpy(dst, src, size, nccl.cudaMemcpyKind.cudaMemcpyHostToDevice);
        if (err != .cudaSuccess) {
            return error.CudaMemcpyFailed;
        }
    }

    pub fn copyDeviceToHost(self: *GPUCoordinator, dst: *anyopaque, src: *const anyopaque, size: usize) !void {
        _ = self;
        const err = nccl.cudaMemcpy(dst, src, size, nccl.cudaMemcpyKind.cudaMemcpyDeviceToHost);
        if (err != .cudaSuccess) {
            return error.CudaMemcpyFailed;
        }
    }

    pub fn allReduceFloat32(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        const err = nccl.ncclAllReduce(
            send_buf,
            recv_buf,
            count,
            .ncclFloat32,
            .ncclSum,
            self.nccl_comm,
            self.cuda_stream
        );
        if (err != .ncclSuccess) {
            const err_str = nccl.ncclGetErrorString(err);
            std.debug.print("NCCL AllReduce Error: {s}\n", .{err_str});
            return error.NCCLAllReduceFailed;
        }
    }

    pub fn allReduceFloat16(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        const err = nccl.ncclAllReduce(
            send_buf,
            recv_buf,
            count,
            .ncclFloat16,
            .ncclSum,
            self.nccl_comm,
            self.cuda_stream
        );
        if (err != .ncclSuccess) {
            const err_str = nccl.ncclGetErrorString(err);
            std.debug.print("NCCL AllReduce (f16) Error: {s}\n", .{err_str});
            return error.NCCLAllReduceFailed;
        }
    }

    pub fn broadcastFloat32(self: *GPUCoordinator, buf: *anyopaque, count: usize, root: usize) !void {
        const err = nccl.ncclBroadcast(
            buf,
            buf,
            count,
            .ncclFloat32,
            @intCast(root),
            self.nccl_comm,
            self.cuda_stream
        );
        if (err != .ncclSuccess) {
            return error.NCCLBroadcastFailed;
        }
    }

    pub fn synchronize(self: *GPUCoordinator) !void {
        const err = nccl.cudaStreamSynchronize(self.cuda_stream);
        if (err != .cudaSuccess) {
            return error.CudaSynchronizeFailed;
        }
    }

    pub fn barrier(self: *GPUCoordinator) !void {
        var sync_value: f32 = 0.0;
        const sync_buffer = try self.allocDeviceMemory(@sizeOf(f32));
        defer self.freeDeviceMemory(sync_buffer);

        try self.copyHostToDevice(sync_buffer, &sync_value, @sizeOf(f32));
        try self.allReduceFloat32(sync_buffer, sync_buffer, 1);
        try self.synchronize();
    }

    pub fn isRoot(self: *const GPUCoordinator) bool {
        return self.rank == 0;
    }
};
