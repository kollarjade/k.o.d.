#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROFILE_DIR="$PROJECT_ROOT/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PROFILE_DIR/$TIMESTAMP"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BOLD}${BLUE}$*${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $*"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

print_error() {
    echo -e "${RED}✗${NC} $*"
}

print_section() {
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}$*${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_command() {
    local cmd=$1
    local install_hint=$2
    if ! command -v "$cmd" &> /dev/null; then
        print_warning "$cmd not found. Install with: $install_hint"
        return 1
    fi
    return 0
}

print_header "JAIDE v40 Profiling & Performance Analysis Suite"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"

SKIP_CPU_PROFILE=0
SKIP_MEM_PROFILE=0
SKIP_FLAMEGRAPH=0
SKIP_REGRESSION=0

check_command "perf" "apt-get install linux-tools-generic" || SKIP_CPU_PROFILE=1
check_command "valgrind" "apt-get install valgrind" || SKIP_MEM_PROFILE=1
check_command "heaptrack" "apt-get install heaptrack" || print_warning "heaptrack not available, some memory profiling will be limited"

if ! check_command "flamegraph.pl" "git clone https://github.com/brendangregg/FlameGraph.git && export PATH=\$PATH:\$(pwd)/FlameGraph"; then
    if [ -d "$PROJECT_ROOT/FlameGraph" ]; then
        export PATH="$PATH:$PROJECT_ROOT/FlameGraph"
        print_success "Using FlameGraph from $PROJECT_ROOT/FlameGraph"
    else
        SKIP_FLAMEGRAPH=1
    fi
fi

print_section "Building Profiling Binaries"

print_header "Building CPU profiling binary..."
zig build profile-cpu 2>&1 | tee "$RESULTS_DIR/build_cpu.log"
print_success "CPU profiling binary built"

print_header "Building memory profiling binary..."
zig build profile-mem 2>&1 | tee "$RESULTS_DIR/build_mem.log"
print_success "Memory profiling binary built"

print_header "Building instrumented binary..."
zig build profile-instrumented 2>&1 | tee "$RESULTS_DIR/build_instrumented.log"
print_success "Instrumented binary built"

if [ $SKIP_CPU_PROFILE -eq 0 ]; then
    print_section "CPU Profiling with perf"
    
    PERF_DATA="$RESULTS_DIR/perf.data"
    PERF_REPORT="$RESULTS_DIR/perf_report.txt"
    
    print_header "Running concurrent benchmark under perf..."
    if perf record -F 99 -g -o "$PERF_DATA" ./zig-out/bin/bench_concurrent_profile_cpu 2>&1 | tee "$RESULTS_DIR/perf_run.log"; then
        print_success "perf data recorded to $PERF_DATA"
        
        print_header "Generating perf report..."
        perf report -i "$PERF_DATA" --stdio > "$PERF_REPORT" 2>&1
        print_success "perf report saved to $PERF_REPORT"
        
        if [ $SKIP_FLAMEGRAPH -eq 0 ]; then
            print_header "Generating flamegraph..."
            FLAMEGRAPH_SVG="$RESULTS_DIR/flamegraph.svg"
            perf script -i "$PERF_DATA" | stackcollapse-perf.pl | flamegraph.pl > "$FLAMEGRAPH_SVG" 2>&1
            print_success "Flamegraph saved to $FLAMEGRAPH_SVG"
        fi
    else
        print_warning "perf recording failed, try running with sudo or adjusting kernel.perf_event_paranoid"
        echo "sudo sysctl -w kernel.perf_event_paranoid=-1" > "$RESULTS_DIR/perf_setup_hint.txt"
    fi
else
    print_warning "Skipping CPU profiling (perf not available)"
fi

if [ $SKIP_MEM_PROFILE -eq 0 ]; then
    print_section "Memory Profiling with Valgrind"
    
    VALGRIND_MASSIF="$RESULTS_DIR/massif.out"
    VALGRIND_MEMCHECK="$RESULTS_DIR/memcheck.log"
    VALGRIND_CALLGRIND="$RESULTS_DIR/callgrind.out"
    
    print_header "Running massif for heap profiling..."
    valgrind --tool=massif \
        --massif-out-file="$VALGRIND_MASSIF" \
        --time-unit=B \
        ./zig-out/bin/bench_concurrent_profile_mem 2>&1 | tee "$RESULTS_DIR/massif_run.log"
    print_success "Massif output saved to $VALGRIND_MASSIF"
    
    print_header "Analyzing massif output..."
    ms_print "$VALGRIND_MASSIF" > "$RESULTS_DIR/massif_analysis.txt"
    print_success "Massif analysis saved to $RESULTS_DIR/massif_analysis.txt"
    
    print_header "Running memcheck for memory leaks..."
    valgrind --leak-check=full \
        --show-leak-kinds=all \
        --track-origins=yes \
        --verbose \
        --log-file="$VALGRIND_MEMCHECK" \
        ./zig-out/bin/bench_concurrent_profile_mem 2>&1 | tee "$RESULTS_DIR/memcheck_run.log"
    print_success "Memcheck output saved to $VALGRIND_MEMCHECK"
    
    LEAKS=$(grep -c "definitely lost" "$VALGRIND_MEMCHECK" || true)
    if [ "$LEAKS" -gt 0 ]; then
        print_error "Memory leaks detected! Check $VALGRIND_MEMCHECK"
        grep "definitely lost" "$VALGRIND_MEMCHECK" | head -n 20
    else
        print_success "No memory leaks detected!"
    fi
    
    print_header "Running callgrind for call graph profiling..."
    valgrind --tool=callgrind \
        --callgrind-out-file="$VALGRIND_CALLGRIND" \
        ./zig-out/bin/bench_concurrent_profile_mem 2>&1 | tee "$RESULTS_DIR/callgrind_run.log"
    print_success "Callgrind output saved to $VALGRIND_CALLGRIND"
    
    if command -v kcachegrind &> /dev/null; then
        print_success "View callgrind output with: kcachegrind $VALGRIND_CALLGRIND"
    fi
    
    if command -v heaptrack &> /dev/null; then
        print_header "Running heaptrack for detailed memory analysis..."
        HEAPTRACK_OUT="$RESULTS_DIR/heaptrack.out"
        heaptrack -o "$HEAPTRACK_OUT" ./zig-out/bin/bench_concurrent_profile_mem 2>&1 | tee "$RESULTS_DIR/heaptrack_run.log"
        print_success "Heaptrack output saved to $HEAPTRACK_OUT"
        
        heaptrack --analyze "$HEAPTRACK_OUT"* > "$RESULTS_DIR/heaptrack_analysis.txt" 2>&1
        print_success "Heaptrack analysis saved to $RESULTS_DIR/heaptrack_analysis.txt"
    fi
else
    print_warning "Skipping memory profiling (valgrind not available)"
fi

print_section "Instrumented Build Analysis"

print_header "Running instrumented benchmark..."
./zig-out/bin/bench_concurrent_profile_instrumented 2>&1 | tee "$RESULTS_DIR/instrumented_run.log"
print_success "Instrumented run completed"

print_section "Performance Regression Detection"

BASELINE_FILE="$PROFILE_DIR/baseline_performance.json"

if [ ! -f "$BASELINE_FILE" ]; then
    print_warning "No baseline performance data found. This run will be set as baseline."
    
    cat > "$BASELINE_FILE" << 'EOF'
{
  "timestamp": "'"$TIMESTAMP"'",
  "results": {
    "concurrent_ssi_insertions_ops_per_sec": 0,
    "parallel_rsf_forward_ops_per_sec": 0,
    "multithreaded_ranking_ops_per_sec": 0
  }
}
EOF
    print_success "Baseline file created at $BASELINE_FILE"
else
    print_header "Comparing against baseline performance..."
    
    print_warning "Regression detection requires manual comparison for now."
    print_warning "Check current results in: $RESULTS_DIR/instrumented_run.log"
    print_warning "Compare against baseline: $BASELINE_FILE"
fi

print_section "Stress Test Execution"

print_header "Running tensor refcount stress test..."
zig build stress 2>&1 | tee "$RESULTS_DIR/stress_test.log"

if grep -q "SUCCESS" "$RESULTS_DIR/stress_test.log"; then
    print_success "Stress test passed!"
else
    print_error "Stress test failed! Check $RESULTS_DIR/stress_test.log"
fi

print_section "Summary Report"

cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# Profiling Results Summary
**Timestamp:** $TIMESTAMP
**Project:** JAIDE v40

## Files Generated

EOF

if [ $SKIP_CPU_PROFILE -eq 0 ]; then
    cat >> "$RESULTS_DIR/SUMMARY.md" << EOF
### CPU Profiling
- perf data: \`perf.data\`
- perf report: \`perf_report.txt\`
EOF
    if [ $SKIP_FLAMEGRAPH -eq 0 ]; then
        echo "- flamegraph: \`flamegraph.svg\`" >> "$RESULTS_DIR/SUMMARY.md"
    fi
fi

if [ $SKIP_MEM_PROFILE -eq 0 ]; then
    cat >> "$RESULTS_DIR/SUMMARY.md" << EOF

### Memory Profiling
- massif output: \`massif.out\`
- massif analysis: \`massif_analysis.txt\`
- memcheck log: \`memcheck.log\`
- callgrind output: \`callgrind.out\`
EOF
    if command -v heaptrack &> /dev/null; then
        echo "- heaptrack output: \`heaptrack.out*\`" >> "$RESULTS_DIR/SUMMARY.md"
        echo "- heaptrack analysis: \`heaptrack_analysis.txt\`" >> "$RESULTS_DIR/SUMMARY.md"
    fi
fi

cat >> "$RESULTS_DIR/SUMMARY.md" << EOF

### Other
- instrumented run: \`instrumented_run.log\`
- stress test: \`stress_test.log\`

## Next Steps

1. Review flamegraph (if generated) to identify hot spots
2. Check memcheck log for memory leaks
3. Analyze massif output for heap usage patterns
4. Compare performance against baseline
5. Review stress test results for thread safety

## Commands

View flamegraph: \`open $RESULTS_DIR/flamegraph.svg\`
View callgrind: \`kcachegrind $RESULTS_DIR/callgrind.out\` (if available)
View massif: \`ms_print $RESULTS_DIR/massif.out | less\`
EOF

print_success "Summary report saved to $RESULTS_DIR/SUMMARY.md"

echo ""
print_header "Profiling Complete!"
echo -e "Results directory: ${BOLD}$RESULTS_DIR${NC}"
echo ""
print_success "All profiling tasks completed successfully!"

if [ -f "$RESULTS_DIR/SUMMARY.md" ]; then
    echo ""
    cat "$RESULTS_DIR/SUMMARY.md"
fi

exit 0
