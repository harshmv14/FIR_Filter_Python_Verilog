`timescale 1ns / 1ps

module tb_fir_filter;

  // Clock & reset
  reg clk;
  reg rst_n;

  // DUT I/O
  reg signed [31:0] din;
  wire signed [31:0] dout;

  // Instantiate FIR
  FIR_filter #(
      .COEFF_FILE("/home/harsh/Projects/FIR_gui/output/coeff.txt")
  ) dut (
      .clk  (clk),
      .rst_n(rst_n),
      .din  (din),
      .dout (dout)
  );

  // Clock gen: 10ns period = 100MHz
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Stimulus storage
  integer in_file, out_file, scan_status;
  integer sample_count;
  reg signed [31:0] sample_mem[0:999999];  // adjust if needed
  integer total_samples;
  integer idx;

  initial begin
    // Load input samples from text file
    in_file = $fopen("/home/harsh/Projects/FIR_gui/output/sine_wave_quant.txt", "r");
    if (in_file == 0) begin
      $display("ERROR: Cannot open input.txt");
      $finish;
    end

    sample_count = 0;
    while (!$feof(
        in_file
    )) begin
      scan_status = $fscanf(in_file, "%h\n", sample_mem[sample_count]);
      if (scan_status != 1) begin
        $display("WARNING: Bad data at line %0d", sample_count + 1);
      end else begin
        sample_count = sample_count + 1;
      end
    end
    $fclose(in_file);
    total_samples = sample_count;
    $display("Loaded %0d input samples", total_samples);

    // Open output file
    out_file = $fopen("/home/harsh/Projects/FIR_gui/input/filter_response_verilog.txt", "w");
    if (out_file == 0) begin
      $display("ERROR: Cannot open output.txt");
      $finish;
    end

    // Reset sequence
    rst_n = 0;
    din   = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;

    // Apply samples to DUT
    for (idx = 0; idx < total_samples; idx = idx + 1) begin
      din = sample_mem[idx];
      @(posedge clk);
      // write the previous output (pipeline delay = 1 sample here)
      $fwrite(out_file, "%08h\n", dout);
    end

    // Flush remaining pipeline outputs
    repeat (63) begin
      din = 0;
      @(posedge clk);
      $fwrite(out_file, "%08h\n", dout);
    end

    $fclose(out_file);
    $display("Simulation complete, results in output.txt");
    $finish;
  end

endmodule
