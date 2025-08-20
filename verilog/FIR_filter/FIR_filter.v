`timescale 1ns / 1ps
// FIR, 63 taps
// in  : Q1.31  (signed [31:0])
// coef: Q5.27  (signed [31:0])
// out : Q5.27  (signed [31:0])

module FIR_filter #(
    parameter  COEFF_FILE = "/home/harsh/Projects/FIR_gui/output/coeff.txt"  // optional $readmemh file
) (
    input  wire               clk,
    input  wire               rst_n,  // active-low synchronous reset
    input  wire signed [31:0] din,    // Q1.31
    output reg signed  [31:0] dout    // Q5.27
);

  // ----- Fixed-point layout -----
  localparam integer TAPS = 63;

  // Input / Coef / Output widths & fractional bits
  localparam integer IN_W = 32;
  localparam integer COEF_W = 32;
  localparam integer OUT_W = 32;

  localparam integer IN_FRAC = 31;  // Q1.31
  localparam integer COEF_FRAC = 27;  // Q5.27
  localparam integer OUT_FRAC = 27;  // Q5.27

  // Product and accumulator widths
  localparam integer PROD_W = IN_W + COEF_W;  // 64
  localparam integer PROD_FRAC = IN_FRAC + COEF_FRAC;  // 58
  localparam integer GUARD_W = $clog2(TAPS);  // 6 (for 63 taps)
  localparam integer ACC_W = PROD_W + GUARD_W;  // 70 total
  localparam integer SHIFT = PROD_FRAC - OUT_FRAC;  // 31 (58 -> 27)

  initial begin
    if (SHIFT <= 0) $error("Bad configuration: SHIFT must be > 0");
    if (TAPS != 63) $display("Note: TAPS is %0d, this module targets 63.", TAPS);
  end

  // ----- Storage for past input samples (shift register) -----
  reg signed [IN_W-1:0] x[0:TAPS-1];

  integer i;
  always @(posedge clk) begin
    if (!rst_n) begin
      for (i = 0; i < TAPS; i = i + 1) x[i] <= 0;
    end else begin
      // shift down
      for (i = TAPS - 1; i > 0; i = i - 1) x[i] <= x[i-1];
      x[0] <= din;
    end
  end

  // ----- Coefficients (Q5.27) -----
  reg signed [COEF_W-1:0] h[0:TAPS-1];
  integer k;
  initial begin
    // default to 0 to avoid X if file missing
    for (k = 0; k < TAPS; k = k + 1) h[k] = 0;
    // optional: load from hex file (32-bit two's complement words)
    // Each line: 8 hex digits, e.g. 0F123456 for Q5.27
    $readmemh(COEFF_FILE, h);
  end

  // ----- Parallel products -----
  wire signed [PROD_W-1:0] prod[0:TAPS-1];
  genvar g;
  generate
    for (g = 0; g < TAPS; g = g + 1) begin : GEN_MUL
      assign prod[g] = $signed(x[g]) * $signed(h[g]);  // 32x32 -> 64 (Q6.58)
    end
  endgenerate

  // ----- Wide accumulation with guard bits -----
  reg signed [ACC_W-1:0] acc;
  always @* begin
    acc = 0;
    for (i = 0; i < TAPS; i = i + 1) begin
      // sign-extend each product from 64 to ACC_W (70)
      acc = acc + {{(ACC_W - PROD_W) {prod[i][PROD_W-1]}}, prod[i]};
    end
    // 'acc' is approximately Q(6 + log2(63)).58 -> Q12.58 (fits in 70 bits)
  end

  // ----- Rounding to nearest, symmetric (±0.5 LSB before shift) -----
  // Build a single-bit-at-position (SHIFT-1) constant, width ACC_W
  wire [ACC_W-1:0] lsb_bias_mag = {{(ACC_W - SHIFT) {1'b0}}, 1'b1, {(SHIFT - 1) {1'b0}}};
  wire signed [ACC_W-1:0] lsb_bias = acc[ACC_W-1] ? -$signed(lsb_bias_mag) : $signed(lsb_bias_mag);

  wire signed [ACC_W-1:0] acc_rounded = acc + lsb_bias;

  // Arithmetic right shift by 31 to go from 58 frac -> 27 frac
  wire signed [ACC_W-1:0] scaled = acc_rounded >>> SHIFT;  // still width ACC_W, now Q~12.27

  // ----- Saturation to 32-bit two's complement (Q5.27 range) -----
  // Max/min representable in signed 32-bit with 27 fractional bits:
  // max =  0x7FFFFFFF  (≈ +15.99999993)
  // min =  0x80000000  (≈ -16.0)
  wire signed [ACC_W-1:0] MAX32_W = {{(ACC_W - 32) {1'b0}}, 32'sh7FFFFFFF};
  wire signed [ACC_W-1:0] MIN32_W = {{(ACC_W - 32) {1'b1}}, 32'sh80000000};

  wire signed [ACC_W-1:0] clamped_wide =
        (scaled > MAX32_W) ? MAX32_W :
        (scaled < MIN32_W) ? MIN32_W :
                             scaled;

  // Register output (single-cycle latency from the shift)
  always @(posedge clk) begin
    if (!rst_n) begin
      dout <= 0;
    end else begin
      dout <= clamped_wide[31:0];  // Q5.27
    end
  end

endmodule
