#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HW_RTL_DIR="$PROJECT_ROOT/src/hw/rtl"
HW_FPGA_DIR="$PROJECT_ROOT/hw/fpga"
BUILD_DIR="$PROJECT_ROOT/build/fpga"

echo "========================================"
echo "JAIDE v40 FPGA Synthesis Pipeline"
echo "Target: iCE40-HX8K Breakout Board"
echo "========================================"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "[1/6] Clash HDL Compilation (.hs → Verilog)"
echo "--------------------------------------------"

if ! command -v clash &> /dev/null; then
    echo "ERROR: Clash compiler not found. Install with: cabal install clash-ghc"
    exit 1
fi

echo "Compiling MemoryArbiter.hs..."
clash --verilog "$HW_RTL_DIR/MemoryArbiter.hs" -outputdir "$BUILD_DIR/arbiter_out"

echo "Compiling SSISearch.hs..."
clash --verilog "$HW_RTL_DIR/SSISearch.hs" -outputdir "$BUILD_DIR/ssi_out"

echo "Compiling RankerCore.hs..."
clash --verilog "$HW_RTL_DIR/RankerCore.hs" -outputdir "$BUILD_DIR/ranker_out"

echo "✓ Clash compilation complete"

echo ""
echo "[2/6] Copying Verilog modules to build directory"
echo "-------------------------------------------------"

find "$BUILD_DIR" -name "*.v" -exec cp {} "$BUILD_DIR/" \;

cp "$HW_FPGA_DIR/top_level.v" "$BUILD_DIR/"
cp "$HW_FPGA_DIR/constraints.pcf" "$BUILD_DIR/"

echo "✓ Verilog modules ready"

echo ""
echo "[3/6] Yosys Synthesis (Verilog → netlist)"
echo "------------------------------------------"

if ! command -v yosys &> /dev/null; then
    echo "ERROR: Yosys not found. Install with: sudo apt install yosys"
    exit 1
fi

cat > "$BUILD_DIR/synth.ys" << 'EOF'
read_verilog top_level.v
read_verilog -lib MemoryArbiter.topEntity.v
read_verilog -lib SSISearch.topEntity.v
read_verilog -lib RankerCore.topEntity.v

hierarchy -check -top top_level

proc
flatten
tribuf -logic
deminout

synth_ice40 -top top_level -json top_level.json

stat
check

write_verilog -attr2comment top_level_synth.v
EOF

yosys -s "$BUILD_DIR/synth.ys" 2>&1 | tee "$BUILD_DIR/yosys.log"

if [ ! -f "$BUILD_DIR/top_level.json" ]; then
    echo "ERROR: Synthesis failed - JSON netlist not generated"
    exit 1
fi

echo "✓ Synthesis complete"

echo ""
echo "[4/6] nextpnr Place-and-Route"
echo "------------------------------"

if ! command -v nextpnr-ice40 &> /dev/null; then
    echo "ERROR: nextpnr-ice40 not found. Install with: sudo apt install nextpnr-ice40"
    exit 1
fi

nextpnr-ice40 \
    --hx8k \
    --package ct256 \
    --json "$BUILD_DIR/top_level.json" \
    --pcf "$BUILD_DIR/constraints.pcf" \
    --asc "$BUILD_DIR/top_level.asc" \
    --freq 100 \
    --timing-allow-fail \
    2>&1 | tee "$BUILD_DIR/nextpnr.log"

if [ ! -f "$BUILD_DIR/top_level.asc" ]; then
    echo "ERROR: Place-and-route failed - ASC file not generated"
    exit 1
fi

echo "✓ Place-and-route complete"

echo ""
echo "[5/6] icestorm Bitstream Generation"
echo "------------------------------------"

if ! command -v icepack &> /dev/null; then
    echo "ERROR: icepack not found. Install with: sudo apt install fpga-icestorm"
    exit 1
fi

icepack "$BUILD_DIR/top_level.asc" "$BUILD_DIR/jaide_v40.bin"

if [ ! -f "$BUILD_DIR/jaide_v40.bin" ]; then
    echo "ERROR: Bitstream generation failed"
    exit 1
fi

BITSTREAM_SIZE=$(stat -f%z "$BUILD_DIR/jaide_v40.bin" 2>/dev/null || stat -c%s "$BUILD_DIR/jaide_v40.bin")
echo "✓ Bitstream generated: jaide_v40.bin ($BITSTREAM_SIZE bytes)"

echo ""
echo "[6/6] Timing Analysis & Resource Utilization"
echo "---------------------------------------------"

icetime -d hx8k -mtr "$BUILD_DIR/timing_report.txt" "$BUILD_DIR/top_level.asc" 2>&1 | tee "$BUILD_DIR/icetime.log"

cat > "$BUILD_DIR/resource_report.txt" << 'EOF'
JAIDE v40 FPGA Resource Utilization Report
==========================================
Target Device: iCE40-HX8K (Lattice Semiconductor)
Package: CT256
Clock Frequency: 100 MHz

Generated from: Yosys and nextpnr logs
EOF

echo "" >> "$BUILD_DIR/resource_report.txt"
echo "Logic Cells (LCs):" >> "$BUILD_DIR/resource_report.txt"
grep -A 5 "Device utilisation" "$BUILD_DIR/nextpnr.log" >> "$BUILD_DIR/resource_report.txt" || echo "  (See nextpnr.log for details)" >> "$BUILD_DIR/resource_report.txt"

echo "" >> "$BUILD_DIR/resource_report.txt"
echo "Memory Blocks:" >> "$BUILD_DIR/resource_report.txt"
grep "ICESTORM_RAM" "$BUILD_DIR/yosys.log" >> "$BUILD_DIR/resource_report.txt" || echo "  (No RAM blocks used)" >> "$BUILD_DIR/resource_report.txt"

echo "" >> "$BUILD_DIR/resource_report.txt"
echo "IO Pins:" >> "$BUILD_DIR/resource_report.txt"
grep "SB_IO" "$BUILD_DIR/yosys.log" >> "$BUILD_DIR/resource_report.txt" || echo "  (See constraints.pcf for IO count)" >> "$BUILD_DIR/resource_report.txt"

echo "" >> "$BUILD_DIR/resource_report.txt"
echo "Timing Summary:" >> "$BUILD_DIR/resource_report.txt"
head -20 "$BUILD_DIR/timing_report.txt" >> "$BUILD_DIR/resource_report.txt" 2>/dev/null || echo "  (See timing_report.txt)" >> "$BUILD_DIR/resource_report.txt"

echo "✓ Analysis complete"

echo ""
echo "========================================"
echo "FPGA Synthesis Complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Bitstream:        build/fpga/jaide_v40.bin"
echo "  Timing Report:    build/fpga/timing_report.txt"
echo "  Resource Report:  build/fpga/resource_report.txt"
echo "  Synthesis Log:    build/fpga/yosys.log"
echo "  P&R Log:          build/fpga/nextpnr.log"
echo ""
echo "To program the FPGA:"
echo "  iceprog build/fpga/jaide_v40.bin"
echo ""
echo "To verify with simulation:"
echo "  iverilog -o build/fpga/sim build/fpga/top_level_synth.v"
echo "  vvp build/fpga/sim"
echo ""
