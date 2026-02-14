#!/usr/bin/env bash
set -euo pipefail

CRYPTOL_VER="3.1.0"
SAW_VER="1.1"
DIST="ubuntu-22.04-x86_64-with-solvers"

TOOLS_DIR="$PWD/tools"
mkdir -p "$TOOLS_DIR"

CRYPTOL_DIR="$TOOLS_DIR/cryptol-${CRYPTOL_VER}-${DIST}"
SAW_DIR="$TOOLS_DIR/saw-${SAW_VER}-${DIST}"

if [ ! -d "$CRYPTOL_DIR" ]; then
    tmp="$(mktemp)"
    wget -q -O "$tmp" "https://github.com/GaloisInc/cryptol/releases/download/${CRYPTOL_VER}/cryptol-${CRYPTOL_VER}-${DIST}.tar.gz"
    tar -xzf "$tmp" -C "$TOOLS_DIR"
    rm -f "$tmp"
fi

if [ ! -d "$SAW_DIR" ]; then
    tmp="$(mktemp)"
    wget -q -O "$tmp" "https://github.com/GaloisInc/saw-script/releases/download/v${SAW_VER}/saw-${SAW_VER}-${DIST}.tar.gz"
    tar -xzf "$tmp" -C "$TOOLS_DIR"
    rm -f "$tmp"
fi

export PATH="$CRYPTOL_DIR/bin:$SAW_DIR/bin:$PATH"

zig build-obj main.zig -O ReleaseSmall -femit-llvm-bc=main.bc

saw verify.saw
