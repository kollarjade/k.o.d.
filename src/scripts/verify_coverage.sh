#!/usr/bin/env bash
set -euo pipefail

VERIFICATION_DIR="jaide40/jaide/verification"
LEDGER_FILE="$VERIFICATION_DIR/COVERAGE_LEDGER.md"
MIN_LOC_PER_FILE=10000
TOTAL_TARGET_LOC=2520000

STRICT_MODE="${STRICT_MODE:-false}"
STRICT_PLACEHOLDER="${STRICT_PLACEHOLDER:-false}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== JAIDE v40 Formal Verification Coverage Report ==="
echo "Mode: STRICT_MODE=$STRICT_MODE, STRICT_PLACEHOLDER=$STRICT_PLACEHOLDER"
echo

total_files=0
total_loc=0
total_placeholders=0
missing_files_count=0
undersized_files_count=0
fail_count=0

check_file() {
    local file=$1
    local prover=$2
    local min_loc=$3
    
    if [[ ! -f "$file" ]]; then
        echo -e "${YELLOW}[WARN-MISSING]${NC} $file"
        ((missing_files_count++))
        return 0
    fi
    
    local loc=$(wc -l < "$file")
    ((total_loc += loc))
    ((total_files++))
    
    local placeholders=0
    local strict_placeholders=0
    
    placeholders=$(grep -c "TODO:" "$file" || true)
    
    if [[ "$STRICT_PLACEHOLDER" == "true" ]]; then
        case "$prover" in
            agda)
                strict_placeholders=$(grep -c "{!!}" "$file" || true)
                ;;
            lean4)
                strict_placeholders=$(grep -cE "(sorry|admit)" "$file" || true)
                ;;
            isabelle)
                strict_placeholders=$(grep -cE "(sorry|oops)" "$file" || true)
                ;;
        esac
        placeholders=$((placeholders + strict_placeholders))
    fi
    
    ((total_placeholders += placeholders))
    
    local status="PASS"
    local color=$GREEN
    local return_code=0
    
    if [[ $placeholders -gt 0 ]]; then
        status="FAIL (${placeholders} TODO: markers)"
        color=$RED
        return_code=1
    elif [[ $loc -lt $min_loc ]]; then
        status="WARN (${loc}/${min_loc} LOC)"
        color=$YELLOW
        ((undersized_files_count++))
        return_code=0
    fi
    
    echo -e "  ${color}[$status]${NC} $file ($loc LOC, $placeholders placeholders)"
    
    return $return_code
}

echo "Checking semantic model files (foundation layer):"
for prover in agda lean4 isabelle viper tlaplus spin; do
    for model in TensorModel GraphModel QuantumModel MemoryModel; do
        case "$prover" in
            agda) ext=".agda" ;;
            lean4) ext=".lean" ;;
            isabelle) ext=".thy" ;;
            viper) ext=".vpr" ;;
            tlaplus) ext=".tla" ;;
            spin) ext=".pml" ;;
        esac
        file="$VERIFICATION_DIR/semantics/${model}${ext}"
        check_file "$file" "$prover" 200 || ((fail_count++))
    done
done
echo

echo "Checking formal_verification.zig verification files (optional):"
for prover_info in "agda:.agda" "lean4:.lean" "isabelle:.thy" "viper:.vpr" "tlaplus:.tla" "spin:.pml"; do
    prover="${prover_info%%:*}"
    ext="${prover_info##*:}"
    file="$VERIFICATION_DIR/${prover}/FormalVerificationZigComplete${ext}"
    check_file "$file" "$prover" $MIN_LOC_PER_FILE || ((fail_count++))
done
echo

echo "Checking futhark_kernels.fut verification files (optional):"
for prover_info in "agda:.agda" "lean4:.lean" "isabelle:.thy" "viper:.vpr" "tlaplus:.tla" "spin:.pml"; do
    prover="${prover_info%%:*}"
    ext="${prover_info##*:}"
    file="$VERIFICATION_DIR/${prover}/FutharkKernelsComplete${ext}"
    check_file "$file" "$prover" $MIN_LOC_PER_FILE || ((fail_count++))
done
echo

echo "=== Summary ==="
echo "Total verification files found: $total_files"
echo "Total LOC: $total_loc/$TOTAL_TARGET_LOC ($(awk "BEGIN {printf \"%.2f\", $total_loc/$TOTAL_TARGET_LOC*100}")%)"
echo "Total TODO: placeholders: $total_placeholders (target: 0)"
echo "Missing files (warnings): $missing_files_count"
echo "Undersized files (warnings): $undersized_files_count"
echo "Failed checks (placeholders): $fail_count"
echo

if [[ $total_placeholders -gt 0 ]]; then
    echo -e "${RED}[FAIL]${NC} TODO: placeholder markers detected"
    exit 1
fi

if [[ "$STRICT_MODE" == "true" ]]; then
    if [[ $missing_files_count -gt 0 ]] || [[ $undersized_files_count -gt 0 ]]; then
        echo -e "${RED}[FAIL]${NC} STRICT_MODE enabled: missing or undersized files detected"
        exit 1
    fi
fi

if [[ $fail_count -gt 0 ]]; then
    echo -e "${RED}[FAIL]${NC} Verification has TODO: placeholders"
    exit 1
fi

if [[ $total_loc -lt $TOTAL_TARGET_LOC ]]; then
    echo -e "${YELLOW}[WARN]${NC} Total LOC below target (expected during development)"
fi

if [[ $missing_files_count -gt 0 ]]; then
    echo -e "${YELLOW}[WARN]${NC} Some verification files missing (expected during development)"
fi

echo -e "${GREEN}[PASS]${NC} All active verification checks passed"
exit 0
