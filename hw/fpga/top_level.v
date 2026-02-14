module top_level #(
    parameter AXI_ADDR_WIDTH = 16,
    parameter AXI_DATA_WIDTH = 32,
    parameter AXI_STRB_WIDTH = 4,
    parameter AXI_PROT_WIDTH = 3,
    parameter AXI_RESP_WIDTH = 2,
    parameter MEM_ADDR_WIDTH = 32,
    parameter MEM_DATA_WIDTH = 16,
    parameter LED_STATUS_WIDTH = 8,
    parameter REG_WIDTH = 32,
    parameter KEY_WIDTH = 64,
    parameter SSI_DEPTH_WIDTH = 8,
    parameter RANKER_RANK_WIDTH = 16,
    parameter CLIENT_COUNT = 4
) (
    input wire clk,
    input wire rst_n,
    input  wire [AXI_ADDR_WIDTH-1:0] axi_awaddr,
    input  wire                      axi_awvalid,
    output wire                      axi_awready,
    input  wire [AXI_PROT_WIDTH-1:0] axi_awprot,
    input  wire [AXI_DATA_WIDTH-1:0] axi_wdata,
    input  wire [AXI_STRB_WIDTH-1:0] axi_wstrb,
    input  wire                      axi_wvalid,
    output wire                      axi_wready,
    output wire [AXI_RESP_WIDTH-1:0] axi_bresp,
    output wire                      axi_bvalid,
    input  wire                      axi_bready,
    input  wire [AXI_ADDR_WIDTH-1:0] axi_araddr,
    input  wire                      axi_arvalid,
    output wire                      axi_arready,
    input  wire [AXI_PROT_WIDTH-1:0] axi_arprot,
    output wire [AXI_DATA_WIDTH-1:0] axi_rdata,
    output wire [AXI_RESP_WIDTH-1:0] axi_rresp,
    output wire                      axi_rvalid,
    input  wire                      axi_rready,
    output wire [MEM_ADDR_WIDTH-1:0] mem_addr,
    output wire [MEM_DATA_WIDTH-1:0] mem_wdata,
    input  wire [MEM_DATA_WIDTH-1:0] mem_rdata,
    output wire                      mem_we,
    output wire                      mem_oe,
    output wire                      mem_ce,
    input  wire                      mem_ready,
    output wire [LED_STATUS_WIDTH-1:0] led_status,
    output wire                        led_error,
    output wire                        irq_out
);

    localparam BYTE_IDX_0 = 7;
    localparam BYTE_IDX_1_LOW = 8;
    localparam BYTE_IDX_1_HIGH = 15;
    localparam BYTE_IDX_2_LOW = 16;
    localparam BYTE_IDX_2_HIGH = 23;
    localparam BYTE_IDX_3_LOW = 24;
    localparam BYTE_IDX_3_HIGH = 31;
    localparam BYTE_IDX_4_LOW = 32;
    localparam BYTE_IDX_4_HIGH = 39;
    localparam BYTE_IDX_5_LOW = 40;
    localparam BYTE_IDX_5_HIGH = 47;
    localparam BYTE_IDX_6_LOW = 48;
    localparam BYTE_IDX_6_HIGH = 55;
    localparam BYTE_IDX_7_LOW = 56;
    localparam BYTE_IDX_7_HIGH = 63;
    localparam STRB_IDX_0 = 0;
    localparam STRB_IDX_1 = 1;
    localparam STRB_IDX_2 = 2;
    localparam STRB_IDX_3 = 3;
    localparam CTRL_SSI_START_BIT = 0;
    localparam CTRL_RANKER_VALID_BIT = 1;
    localparam STATUS_SSI_FOUND_BIT = 0;
    localparam STATUS_RANKER_DONE_BIT = 1;
    localparam STATUS_DEPTH_LOW = 8;
    localparam STATUS_DEPTH_HIGH = 15;
    localparam STATUS_RANK_LOW = 16;
    localparam STATUS_RANK_HIGH = 31;
    localparam CLIENT_0 = 0;
    localparam CLIENT_1 = 1;
    localparam CLIENT_2 = 2;
    localparam CLIENT_3 = 3;
    localparam LOGIC_HIGH = 1'b1;
    localparam LOGIC_LOW = 1'b0;
    localparam RESP_OKAY = 2'b00;
    localparam RESP_SLVERR = 2'b10;
    localparam ZERO_REG = 32'h0;
    localparam ZERO_KEY = 64'h0;
    localparam ZERO_ADDR = 16'h0;
    localparam ZERO_WDATA = 16'h0;
    localparam ZERO_MEM_ADDR = 32'h0;

    localparam ADDR_CONTROL   = 16'h0000;
    localparam ADDR_STATUS    = 16'h0004;
    localparam ADDR_CONFIG    = 16'h0008;
    localparam ADDR_RESULT    = 16'h000C;
    localparam ADDR_SSI_KEY_L = 16'h0010;
    localparam ADDR_SSI_KEY_H = 16'h0014;
    localparam ADDR_SSI_ROOT  = 16'h0018;
    localparam ADDR_SSI_RES   = 16'h001C;
    localparam ADDR_RNK_HASH_L = 16'h0020;
    localparam ADDR_RNK_HASH_H = 16'h0024;
    localparam ADDR_RNK_SEG_L = 16'h0028;
    localparam ADDR_RNK_SEG_H = 16'h002C;
    localparam ADDR_RNK_POS_L = 16'h0030;
    localparam ADDR_RNK_POS_H = 16'h0034;
    localparam ADDR_RNK_SCORE = 16'h0038;
    localparam ADDR_RNK_RES   = 16'h003C;

    wire reset;
    reg rst_sync1, rst_sync2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync1 <= LOGIC_LOW;
        else rst_sync1 <= LOGIC_HIGH;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync2 <= LOGIC_LOW;
        else rst_sync2 <= rst_sync1;
    end

    assign reset = !rst_sync2;

    reg [REG_WIDTH-1:0] control_reg;
    reg [REG_WIDTH-1:0] status_reg;
    reg [REG_WIDTH-1:0] config_reg;
    reg [REG_WIDTH-1:0] result_reg;

    wire [KEY_WIDTH-1:0] ssi_search_key;
    wire [REG_WIDTH-1:0] ssi_root_addr;
    wire                 ssi_start;
    wire [REG_WIDTH-1:0] ssi_result_addr;
    wire                 ssi_found;
    wire [SSI_DEPTH_WIDTH-1:0] ssi_depth;
    wire                 ssi_done;
    wire [REG_WIDTH-1:0] ssi_mem_addr;
    wire                 ssi_mem_req;
    wire                 ssi_node_valid;

    wire [KEY_WIDTH-1:0] ranker_query_hash;
    wire [KEY_WIDTH-1:0] ranker_segment_id;
    wire [KEY_WIDTH-1:0] ranker_segment_pos;
    wire [REG_WIDTH-1:0] ranker_base_score;
    wire                 ranker_valid;
    wire [REG_WIDTH-1:0] ranker_final_score;
    wire [RANKER_RANK_WIDTH-1:0] ranker_rank;
    wire                 ranker_done;

    wire [REG_WIDTH-1:0]      arbiter_mem_addr;
    wire [MEM_DATA_WIDTH-1:0] arbiter_mem_wdata;
    wire                      arbiter_mem_we;
    wire                      arbiter_mem_req;
    wire                      arbiter_grant;

    wire [CLIENT_COUNT-1:0] client_req;
    wire [CLIENT_COUNT-1:0] client_grant;

    reg aw_ready_reg;
    reg w_ready_reg;
    reg b_valid_reg;
    reg [AXI_RESP_WIDTH-1:0] b_resp_reg;
    reg ar_ready_reg;
    reg r_valid_reg;
    reg [AXI_RESP_WIDTH-1:0] r_resp_reg;
    reg [AXI_ADDR_WIDTH-1:0] aw_addr_reg;
    reg aw_valid_reg;
    reg [AXI_ADDR_WIDTH-1:0] ar_addr_reg;
    reg [AXI_DATA_WIDTH-1:0] r_data_reg;

    assign axi_awready = aw_ready_reg;
    assign axi_wready = w_ready_reg;
    assign axi_bvalid = b_valid_reg;
    assign axi_bresp = b_resp_reg;
    assign axi_arready = ar_ready_reg;
    assign axi_rvalid = r_valid_reg;
    assign axi_rresp = r_resp_reg;
    assign axi_rdata = r_data_reg;

    reg [KEY_WIDTH-1:0] ssi_key_reg;
    reg [REG_WIDTH-1:0] ssi_root_reg;
    reg [REG_WIDTH-1:0] ssi_result_reg;
    reg [KEY_WIDTH-1:0] ranker_hash_reg;
    reg [KEY_WIDTH-1:0] ranker_seg_reg;
    reg [KEY_WIDTH-1:0] ranker_pos_reg;
    reg [REG_WIDTH-1:0] ranker_score_reg;
    reg [REG_WIDTH-1:0] ranker_result_reg;

    wire aw_handshake = axi_awvalid && axi_awready;
    wire w_handshake = axi_wvalid && axi_wready;
    wire b_handshake = axi_bvalid && axi_bready;
    wire ar_handshake = axi_arvalid && axi_arready;
    wire r_handshake = axi_rvalid && axi_rready;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            aw_ready_reg <= LOGIC_HIGH;
            w_ready_reg <= LOGIC_HIGH;
            b_valid_reg <= LOGIC_LOW;
            b_resp_reg <= RESP_OKAY;
            aw_addr_reg <= ZERO_ADDR;
            aw_valid_reg <= LOGIC_LOW;
            control_reg <= ZERO_REG;
            config_reg <= ZERO_REG;
            ssi_key_reg <= ZERO_KEY;
            ssi_root_reg <= ZERO_REG;
            ranker_hash_reg <= ZERO_KEY;
            ranker_seg_reg <= ZERO_KEY;
            ranker_pos_reg <= ZERO_KEY;
            ranker_score_reg <= ZERO_REG;
        end else begin
            if (aw_handshake && !aw_valid_reg) begin
                aw_addr_reg <= axi_awaddr;
                aw_valid_reg <= LOGIC_HIGH;
                aw_ready_reg <= LOGIC_LOW;
            end
            if (w_handshake && aw_valid_reg && !b_valid_reg) begin
                b_valid_reg <= LOGIC_HIGH;
                aw_valid_reg <= LOGIC_LOW;
                w_ready_reg <= LOGIC_LOW;
                b_resp_reg <= RESP_OKAY;
                case (aw_addr_reg)
                    ADDR_CONTROL: begin
                        if (axi_wstrb[STRB_IDX_0]) control_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) control_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) control_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) control_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_CONFIG: begin
                        if (axi_wstrb[STRB_IDX_0]) config_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) config_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) config_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) config_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_SSI_KEY_L: begin
                        if (axi_wstrb[STRB_IDX_0]) ssi_key_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ssi_key_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ssi_key_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ssi_key_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_SSI_KEY_H: begin
                        if (axi_wstrb[STRB_IDX_0]) ssi_key_reg[BYTE_IDX_4_HIGH:BYTE_IDX_4_LOW] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ssi_key_reg[BYTE_IDX_5_HIGH:BYTE_IDX_5_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ssi_key_reg[BYTE_IDX_6_HIGH:BYTE_IDX_6_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ssi_key_reg[BYTE_IDX_7_HIGH:BYTE_IDX_7_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_SSI_ROOT: begin
                        if (axi_wstrb[STRB_IDX_0]) ssi_root_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ssi_root_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ssi_root_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ssi_root_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_HASH_L: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_hash_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_hash_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_hash_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_hash_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_HASH_H: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_hash_reg[BYTE_IDX_4_HIGH:BYTE_IDX_4_LOW] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_hash_reg[BYTE_IDX_5_HIGH:BYTE_IDX_5_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_hash_reg[BYTE_IDX_6_HIGH:BYTE_IDX_6_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_hash_reg[BYTE_IDX_7_HIGH:BYTE_IDX_7_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_SEG_L: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_seg_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_seg_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_seg_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_seg_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_SEG_H: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_seg_reg[BYTE_IDX_4_HIGH:BYTE_IDX_4_LOW] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_seg_reg[BYTE_IDX_5_HIGH:BYTE_IDX_5_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_seg_reg[BYTE_IDX_6_HIGH:BYTE_IDX_6_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_seg_reg[BYTE_IDX_7_HIGH:BYTE_IDX_7_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_POS_L: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_pos_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_pos_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_pos_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_pos_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_POS_H: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_pos_reg[BYTE_IDX_4_HIGH:BYTE_IDX_4_LOW] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_pos_reg[BYTE_IDX_5_HIGH:BYTE_IDX_5_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_pos_reg[BYTE_IDX_6_HIGH:BYTE_IDX_6_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_pos_reg[BYTE_IDX_7_HIGH:BYTE_IDX_7_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    ADDR_RNK_SCORE: begin
                        if (axi_wstrb[STRB_IDX_0]) ranker_score_reg[BYTE_IDX_0:0] <= axi_wdata[BYTE_IDX_0:0];
                        if (axi_wstrb[STRB_IDX_1]) ranker_score_reg[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW] <= axi_wdata[BYTE_IDX_1_HIGH:BYTE_IDX_1_LOW];
                        if (axi_wstrb[STRB_IDX_2]) ranker_score_reg[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW] <= axi_wdata[BYTE_IDX_2_HIGH:BYTE_IDX_2_LOW];
                        if (axi_wstrb[STRB_IDX_3]) ranker_score_reg[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW] <= axi_wdata[BYTE_IDX_3_HIGH:BYTE_IDX_3_LOW];
                    end
                    default: b_resp_reg <= RESP_SLVERR;
                endcase
            end
            if (b_handshake) begin
                b_valid_reg <= LOGIC_LOW;
            end
            if (!aw_valid_reg && !b_valid_reg) begin
                aw_ready_reg <= LOGIC_HIGH;
                w_ready_reg <= LOGIC_HIGH;
            end
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ar_ready_reg <= LOGIC_HIGH;
            r_valid_reg <= LOGIC_LOW;
            r_resp_reg <= RESP_OKAY;
            ar_addr_reg <= ZERO_ADDR;
            r_data_reg <= ZERO_REG;
        end else begin
            if (ar_handshake && !r_valid_reg) begin
                ar_addr_reg <= axi_araddr;
                r_valid_reg <= LOGIC_HIGH;
                ar_ready_reg <= LOGIC_LOW;
                r_resp_reg <= RESP_OKAY;
                case (axi_araddr)
                    ADDR_CONTROL: r_data_reg <= control_reg;
                    ADDR_STATUS: r_data_reg <= status_reg;
                    ADDR_CONFIG: r_data_reg <= config_reg;
                    ADDR_RESULT: r_data_reg <= result_reg;
                    ADDR_SSI_KEY_L: r_data_reg <= ssi_key_reg[REG_WIDTH-1:0];
                    ADDR_SSI_KEY_H: r_data_reg <= ssi_key_reg[KEY_WIDTH-1:REG_WIDTH];
                    ADDR_SSI_ROOT: r_data_reg <= ssi_root_reg;
                    ADDR_SSI_RES: r_data_reg <= ssi_result_reg;
                    ADDR_RNK_HASH_L: r_data_reg <= ranker_hash_reg[REG_WIDTH-1:0];
                    ADDR_RNK_HASH_H: r_data_reg <= ranker_hash_reg[KEY_WIDTH-1:REG_WIDTH];
                    ADDR_RNK_SEG_L: r_data_reg <= ranker_seg_reg[REG_WIDTH-1:0];
                    ADDR_RNK_SEG_H: r_data_reg <= ranker_seg_reg[KEY_WIDTH-1:REG_WIDTH];
                    ADDR_RNK_POS_L: r_data_reg <= ranker_pos_reg[REG_WIDTH-1:0];
                    ADDR_RNK_POS_H: r_data_reg <= ranker_pos_reg[KEY_WIDTH-1:REG_WIDTH];
                    ADDR_RNK_SCORE: r_data_reg <= ranker_score_reg;
                    ADDR_RNK_RES: r_data_reg <= ranker_result_reg;
                    default: begin
                        r_data_reg <= ZERO_REG;
                        r_resp_reg <= RESP_SLVERR;
                    end
                endcase
            end
            if (r_handshake) begin
                r_valid_reg <= LOGIC_LOW;
            end
            if (!r_valid_reg) begin
                ar_ready_reg <= LOGIC_HIGH;
            end
        end
    end

    assign ssi_search_key = ssi_key_reg;
    assign ssi_root_addr = ssi_root_reg;
    assign ssi_start = control_reg[CTRL_SSI_START_BIT];

    assign ranker_query_hash = ranker_hash_reg;
    assign ranker_segment_id = ranker_seg_reg;
    assign ranker_segment_pos = ranker_pos_reg;
    assign ranker_base_score = ranker_score_reg;
    assign ranker_valid = control_reg[CTRL_RANKER_VALID_BIT];

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ssi_result_reg <= ZERO_REG;
            ranker_result_reg <= ZERO_REG;
            status_reg <= ZERO_REG;
        end else begin
            if (ssi_done) begin
                ssi_result_reg <= ssi_result_addr;
                status_reg[STATUS_SSI_FOUND_BIT] <= ssi_found;
                status_reg[STATUS_DEPTH_HIGH:STATUS_DEPTH_LOW] <= ssi_depth;
            end
            if (ranker_done) begin
                ranker_result_reg <= ranker_final_score;
                status_reg[STATUS_RANKER_DONE_BIT] <= LOGIC_HIGH;
                status_reg[STATUS_RANK_HIGH:STATUS_RANK_LOW] <= ranker_rank;
            end
        end
    end

    assign ssi_node_valid = client_grant[CLIENT_0] && mem_ready;

    SSISearch_topEntity ssi_search (
        .clk(clk),
        .rst(reset),
        .enable(LOGIC_HIGH),
        .searchRequest_key(ssi_search_key),
        .searchRequest_root(ssi_root_addr),
        .searchRequest_valid(ssi_start),
        .nodeData(mem_rdata),
        .nodeValid(ssi_node_valid),
        .memAddr(ssi_mem_addr),
        .memReq(ssi_mem_req),
        .memGrant(client_grant[CLIENT_0]),
        .resultAddr(ssi_result_addr),
        .resultFound(ssi_found),
        .resultDepth(ssi_depth),
        .resultValid(ssi_done)
    );

    RankerCore_topEntity ranker_core (
        .clk(clk),
        .rst(reset),
        .enable(LOGIC_HIGH),
        .queryHash(ranker_query_hash),
        .segmentID(ranker_segment_id),
        .segmentPos(ranker_segment_pos),
        .baseScore(ranker_base_score),
        .inputValid(ranker_valid),
        .finalScore(ranker_final_score),
        .rank(ranker_rank),
        .outputValid(ranker_done)
    );

    assign client_req[CLIENT_0] = ssi_mem_req;
    assign client_req[CLIENT_1] = LOGIC_LOW;
    assign client_req[CLIENT_2] = LOGIC_LOW;
    assign client_req[CLIENT_3] = LOGIC_LOW;

    MemoryArbiter_topEntity mem_arbiter (
        .clk(clk),
        .rst(reset),
        .enable(LOGIC_HIGH),
        .client0_req(client_req[CLIENT_0]),
        .client0_addr(ssi_mem_addr),
        .client0_wdata(ZERO_WDATA),
        .client0_we(LOGIC_LOW),
        .client1_req(client_req[CLIENT_1]),
        .client1_addr(ZERO_MEM_ADDR),
        .client1_wdata(ZERO_WDATA),
        .client1_we(LOGIC_LOW),
        .client2_req(client_req[CLIENT_2]),
        .client2_addr(ZERO_MEM_ADDR),
        .client2_wdata(ZERO_WDATA),
        .client2_we(LOGIC_LOW),
        .client3_req(client_req[CLIENT_3]),
        .client3_addr(ZERO_MEM_ADDR),
        .client3_wdata(ZERO_WDATA),
        .client3_we(LOGIC_LOW),
        .client0_grant(client_grant[CLIENT_0]),
        .client1_grant(client_grant[CLIENT_1]),
        .client2_grant(client_grant[CLIENT_2]),
        .client3_grant(client_grant[CLIENT_3]),
        .memAddr(arbiter_mem_addr),
        .memWData(arbiter_mem_wdata),
        .memWE(arbiter_mem_we),
        .memReq(arbiter_mem_req),
        .memReady(mem_ready)
    );

    assign mem_addr = arbiter_mem_addr;
    assign mem_wdata = arbiter_mem_wdata;
    assign mem_we = arbiter_mem_we;
    assign mem_oe = !arbiter_mem_we && arbiter_mem_req;
    assign mem_ce = arbiter_mem_req;

    assign led_status = {
        ssi_done,
        ranker_done,
        arbiter_mem_req,
        mem_ready,
        client_grant[CLIENT_COUNT-1:0]
    };
    assign led_error = (status_reg[STATUS_SSI_FOUND_BIT] == LOGIC_LOW) && ssi_done;
    assign irq_out = ssi_done || ranker_done;

endmodule
