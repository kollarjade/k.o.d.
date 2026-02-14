
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE_DIR="$PROJECT_ROOT/profiling_results/edge_fractal"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PROFILE_DIR/$TIMESTAMP"

mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo "Edge-Focused Profiling: Fractal LPU + Agent Kernel"
echo "======================================================================"
echo "Results: $RESULTS_DIR"
echo ""

cd "$PROJECT_ROOT/src"

echo "[1/5] Building profiling binary with fractal LPU instrumentation..."
zig build-exe main.zig \
  -O ReleaseFast \
  -fstrip \
  -femit-bin="$RESULTS_DIR/jaide_fractal_profile" \
  2>&1 | tee "$RESULTS_DIR/build.log"
echo "✓ Binary built: $RESULTS_DIR/jaide_fractal_profile"

echo ""
echo "[2/5] Running edge workload benchmark..."
"$RESULTS_DIR/jaide_fractal_profile" --mode fractal-bench 2>&1 | tee "$RESULTS_DIR/fractal_bench.log"
echo "✓ Benchmark completed"

echo ""
echo "[3/5] Measuring power consumption (simulated via CPU usage)..."
if command -v perf &> /dev/null; then
  perf stat -e power/energy-pkg/,power/energy-ram/ \
    "$RESULTS_DIR/jaide_fractal_profile" --mode fractal-bench \
    2>&1 | tee "$RESULTS_DIR/power.log" || echo "⚠ perf unavailable, skipping power measurement"
else
  echo "⚠ perf not installed, skipping power measurement"
fi

echo ""
echo "[4/5] Extracting ESSO temperature/reheat metrics from logs..."
grep -E "ESSO.*temperature|reheat" "$RESULTS_DIR/fractal_bench.log" > "$RESULTS_DIR/esso_params.txt" || echo "No ESSO metrics found"

echo ""
echo "[5/5] Generating summary report..."
cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# Edge Profiling Summary
**Timestamp:** $TIMESTAMP

## Workload
- Fractal LPU with SSRG integration
- Agent kernel multi-node simulation

## Metrics
- Binary: \`jaide_fractal_profile\`
- Benchmark log: \`fractal_bench.log\`
- Power log: \`power.log\`
- ESSO params: \`esso_params.txt\`

## Next Steps
1. Analyze power log for energy-per-operation
2. Tune ESSO cooling/reheat parameters based on measured latency
3. Compare against baseline non-fractal implementation
4. Profile on actual edge hardware (Raspberry Pi, Jetson Nano)
EOF

echo "✓ Summary: $RESULTS_DIR/SUMMARY.md"
echo ""
echo "======================================================================"
echo "Edge profiling complete!"
echo "======================================================================"
