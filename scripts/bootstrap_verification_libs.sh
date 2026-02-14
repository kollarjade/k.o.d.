#!/usr/bin/env bash
set -e

echo "======================================================================="
echo "JAIDE v40 Formal Verification Library Bootstrap"
echo "======================================================================="
echo "This script downloads and builds verification library dependencies."
echo "This is a ONE-TIME setup that creates vendored artifacts for fast"
echo "verification runs. Expected time: ~10 minutes. Download size: ~10GB."
echo "======================================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="$PROJECT_ROOT/.verification_cache"

START_TIME=$(date +%s)

# Create cache directory structure
mkdir -p "$CACHE_DIR"
mkdir -p "$CACHE_DIR/mathlib"
mkdir -p "$CACHE_DIR/isabelle"
mkdir -p "$CACHE_DIR/agda-stdlib"

echo "======================================================================="
echo "1/4 Downloading Mathlib for Lean4"
echo "======================================================================="
echo "Mathlib provides real number arithmetic, tactics, and analysis tools."
echo "Download size: ~3GB | Build artifacts: ~2GB"
echo ""

# FIX ERROR 3: Add Lean4 version checking and graceful error handling
MATHLIB_COUNT=0
MATHLIB_SKIPPED=false

# Check if Lean4 is available
if ! command -v lean &> /dev/null; then
    echo "⚠ WARNING: Lean4 not found in PATH. Skipping Mathlib setup."
    echo "Install Lean4 with: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
    MATHLIB_SKIPPED=true
else
    # Get Lean4 version
    LEAN_VERSION=$(lean --version 2>/dev/null | head -n1 | grep -oP 'v\K[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    echo "Detected Lean4 version: v${LEAN_VERSION}"
    
    # Check if lake is available
    if ! command -v lake &> /dev/null; then
        echo "⚠ WARNING: Lake (Lean build tool) not found. Skipping Mathlib setup."
        MATHLIB_SKIPPED=true
    else
        cd "$CACHE_DIR/mathlib"
        
        if [ ! -f "lakefile.lean" ]; then
            echo "Cloning Mathlib repository (this may take a few minutes)..."
            if git clone --depth=1 https://github.com/leanprover-community/mathlib4.git . 2>/dev/null; then
                echo "✓ Mathlib repository cloned"
            else
                echo "⚠ WARNING: Failed to clone Mathlib repository. Skipping Mathlib setup."
                MATHLIB_SKIPPED=true
            fi
        else
            echo "✓ Mathlib already cloned, updating..."
            git pull 2>/dev/null || echo "Note: Could not update Mathlib (using cached version)"
        fi
        
        if [ "$MATHLIB_SKIPPED" = false ]; then
            echo ""
            echo "Building Mathlib (this generates .olean compiled artifacts)..."
            echo "This step may take 5-8 minutes depending on your system..."
            
            # Try to build Mathlib with error handling
            if lake build 2>&1; then
                # Count .olean files from correct location (.lake/build/lib/)
                MATHLIB_COUNT=$(find .lake/build/lib -name "*.olean" -type f 2>/dev/null | wc -l || echo "0")
                echo "✓ Mathlib build complete: $MATHLIB_COUNT .olean files generated in .lake/build/lib/"
            else
                echo "⚠ WARNING: Mathlib build failed (likely version incompatibility)."
                echo "This is non-critical - verification will continue with Isabelle and Agda."
                echo "To fix: Update Lean4 version or use compatible Mathlib release."
                MATHLIB_SKIPPED=true
                MATHLIB_COUNT=0
            fi
        fi
    fi
fi

if [ "$MATHLIB_SKIPPED" = true ]; then
    echo "→ Mathlib setup skipped. Other verification libraries will still be built."
fi
echo ""

echo "======================================================================="
echo "2/4 Downloading Isabelle/HOL-Analysis"
echo "======================================================================="
echo "HOL-Analysis provides real analysis and multiset theories."
echo "Download size: ~1.5GB | Heap size: ~500MB"
echo ""

cd "$CACHE_DIR/isabelle"

if [ ! -d "AFP" ]; then
    echo "Downloading Isabelle Archive of Formal Proofs (AFP)..."
    wget -q https://www.isa-afp.org/release/afp-current.tar.gz -O afp.tar.gz
    echo "Extracting AFP archive..."
    tar xzf afp.tar.gz
    mv afp-* AFP
    rm afp.tar.gz
    echo "✓ AFP downloaded and extracted"
else
    echo "✓ AFP already present"
fi

echo ""
echo "Building HOL-Analysis heap files..."
# Create Isabelle user directory in cache
mkdir -p "$CACHE_DIR/isabelle_user"
export ISABELLE_HOME_USER="$CACHE_DIR/isabelle_user"
isabelle build -d AFP -b HOL-Analysis

# Count heap files from cache location
HEAP_COUNT=$(find "$CACHE_DIR/isabelle_user" -name "*.heap" -type f 2>/dev/null | wc -l || echo "0")
echo "✓ Isabelle build complete: $HEAP_COUNT heap files generated in $CACHE_DIR/isabelle_user/heaps/"
echo ""

echo "======================================================================="
echo "3/4 Downloading Agda Standard Library"
echo "======================================================================="
echo "Agda stdlib provides dependent types, vectors, and equality proofs."
echo "Download size: ~50MB | Interface files: ~500MB"
echo ""

cd "$CACHE_DIR/agda-stdlib"

if [ ! -f "standard-library.agda-lib" ]; then
    echo "Downloading Agda standard library..."
    AGDA_VERSION=$(agda --version | head -n1 | cut -d' ' -f3 || echo "2.6.4")
    STDLIB_VERSION="v2.0"
    
    wget -q "https://github.com/agda/agda-stdlib/archive/refs/tags/${STDLIB_VERSION}.tar.gz" -O agda-stdlib.tar.gz
    echo "Extracting Agda stdlib..."
    tar xzf agda-stdlib.tar.gz --strip-components=1
    rm agda-stdlib.tar.gz
    echo "✓ Agda stdlib downloaded"
else
    echo "✓ Agda stdlib already present"
fi

echo ""
echo "Pre-compiling Agda stdlib modules (generates .agdai interface files)..."
cd "$CACHE_DIR/agda-stdlib"

# Create .agda directory structure
mkdir -p "$CACHE_DIR/.agda"

# Create library configuration file
cat > "$CACHE_DIR/.agda/libraries" << AGDA_LIBS
$CACHE_DIR/agda-stdlib/standard-library.agda-lib
AGDA_LIBS

echo "✓ Agda library configuration created at $CACHE_DIR/.agda/libraries"

# Compile commonly used stdlib modules
echo "Compiling core stdlib modules..."
agda --library-file="$CACHE_DIR/.agda/libraries" src/Everything.agda 2>/dev/null || echo "Note: Some stdlib modules may require additional dependencies"

AGDAI_COUNT=$(find . -name "*.agdai" -type f | wc -l)
echo "✓ Agda stdlib compilation complete: $AGDAI_COUNT .agdai files generated"
echo ""

echo "======================================================================="
echo "4/4 Creating verification cache metadata"
echo "======================================================================="

cd "$PROJECT_ROOT"

# Create READY marker file with metadata
MATHLIB_STATUS="$MATHLIB_COUNT .olean files"
if [ "$MATHLIB_SKIPPED" = true ]; then
    MATHLIB_STATUS="SKIPPED (Lean4 not available or incompatible)"
fi

cat > "$CACHE_DIR/READY" << METADATA
JAIDE v40 Verification Cache
Created: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Mathlib artifacts: $MATHLIB_STATUS
Isabelle heaps: $HEAP_COUNT .heap files
Agda interfaces: $AGDAI_COUNT .agdai files
Total cache size: $(du -sh "$CACHE_DIR" | cut -f1)

This cache enables fast verification runs (<2 min) without re-downloading
or re-compiling external proof libraries.

To run verification with these cached libraries:
  ./scripts/verify_all.sh

To rebuild cache (if libraries are updated):
  rm -rf .verification_cache
  ./scripts/bootstrap_verification_libs.sh
METADATA

echo "✓ Cache metadata created"
echo ""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "======================================================================="
echo "✓ BOOTSTRAP COMPLETE"
echo "======================================================================="
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Cache location: $CACHE_DIR"
echo "Cache size: $(du -sh "$CACHE_DIR" | cut -f1)"
echo ""
echo "Verification libraries are ready! You can now run:"
echo "  zig build verify"
echo ""
echo "Or directly:"
echo "  ./scripts/verify_all.sh"
echo "======================================================================="
