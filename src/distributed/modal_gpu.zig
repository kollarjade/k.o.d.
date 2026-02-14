
const std = @import("std");
const http = std.http;

pub const ModalGPUClient = struct {
    allocator: std.mem.Allocator,
    api_token: []const u8,
    http_client: http.Client,
    gpu_count: usize,

    pub fn init(allocator: std.mem.Allocator, api_token: []const u8) !ModalGPUClient {
        return .{
            .allocator = allocator,
            .api_token = try allocator.dupe(u8, api_token),
            .http_client = http.Client{ .allocator = allocator },
            .gpu_count = 8,
        };
    }

    pub fn deinit(self: *ModalGPUClient) void {
        self.allocator.free(self.api_token);
        self.http_client.deinit();
    }

    pub fn deployTrainingJob(self: *ModalGPUClient, model_path: []const u8, dataset_path: []const u8) ![]const u8 {
        const uri = try std.Uri.parse("https://api.modal.com/v1/functions/deploy");

        var headers = http.Headers{ .allocator = self.allocator };
        defer headers.deinit();

        try headers.append("Authorization", try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token}));
        try headers.append("Content-Type", "application/json");

        const payload = try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "gpu": "B200",
            \\  "gpu_count": {d},
            \\  "image": "jaide-v40-training",
            \\  "model_path": "{s}",
            \\  "dataset_path": "{s}",
            \\  "batch_size": 32,
            \\  "epochs": 10
            \\}}
        , .{ self.gpu_count, model_path, dataset_path });
        defer self.allocator.free(payload);

        var req = try self.http_client.open(.POST, uri, headers, .{});
        defer req.deinit();

        req.transfer_encoding = .chunked;
        try req.send(.{});
        try req.writeAll(payload);
        try req.finish();
        try req.wait();

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return body;
    }

    pub fn getJobStatus(self: *ModalGPUClient, job_id: []const u8) ![]const u8 {
        const uri_str = try std.fmt.allocPrint(self.allocator, "https://api.modal.com/v1/functions/{s}/status", .{job_id});
        defer self.allocator.free(uri_str);

        const uri = try std.Uri.parse(uri_str);

        var headers = http.Headers{ .allocator = self.allocator };
        defer headers.deinit();

        try headers.append("Authorization", try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token}));

        var req = try self.http_client.open(.GET, uri, headers, .{});
        defer req.deinit();

        try req.send(.{});
        try req.finish();
        try req.wait();

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return body;
    }
};
