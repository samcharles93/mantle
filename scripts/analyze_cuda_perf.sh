#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANTLE_BIN="${SCRIPT_DIR}/../bin/mantle"
MODEL="${1:-/work/models/mcf/LFM2.5-1.2B-Instruct.mcf}"
STEPS="${2:-128}"

if [ ! -f "$MANTLE_BIN" ]; then
    echo "Error: mantle binary not found at $MANTLE_BIN"
    echo "Run: CUDA=1 task build -f"
    exit 1
fi

echo "=================================="
echo "CUDA Performance Analysis"
echo "=================================="
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo ""

# Run multiple times and capture stats
TMP_OUT=$(mktemp)

echo "Running baseline (no optimizations)..."
CUDA=1 MANTLE_CUDA_TRACE=1 "$MANTLE_BIN" run --backend cuda \
    -m "$MODEL" --steps "$STEPS" --temp 0 --stream-mode quiet \
    --prompt "test" > "$TMP_OUT" 2>&1

# Extract key metrics
TOKENS=$(grep "Tokens processed:" "$TMP_OUT" | awk '{print $3}')
MATVEC=$(grep "MatVec calls:" "$TMP_OUT" | awk '{print $3}')
RMSNORM=$(grep "RMSNorm calls:" "$TMP_OUT" | awk '{print $3}')
SYNCS=$(grep "Stream syncs:" "$TMP_OUT" | awk '{print $3}')
H2D=$(grep "H2D bytes:" "$TMP_OUT" | awk '{print $3}')
D2H=$(grep "D2H bytes:" "$TMP_OUT" | awk '{print $3}')
GEN_TPS=$(grep "gen_tps=" "$TMP_OUT" | sed 's/.*gen_tps=\([0-9.]*\).*/\1/')

echo ""
echo "=================================="
echo "Baseline Performance"
echo "=================================="
echo "Generation TPS:    $GEN_TPS"
echo "Tokens processed:  $TOKENS"
echo ""
echo "Operation counts:"
echo "  MatVec calls:    $MATVEC ($(echo "$MATVEC / $TOKENS" | bc -l | xargs printf "%.1f") per token)"
echo "  RMSNorm calls:   $RMSNORM ($(echo "$RMSNORM / $TOKENS" | bc -l | xargs printf "%.1f") per token)"
echo "  Stream syncs:    $SYNCS ($(echo "$SYNCS / $TOKENS" | bc -l | xargs printf "%.1f") per token)"
echo ""
echo "Memory transfers:"
echo "  Host→Device:     $H2D MB ($(echo "$H2D / $TOKENS" | bc -l | xargs printf "%.1f") MB/token)"
echo "  Device→Host:     $D2H MB ($(echo "$D2H / $TOKENS" | bc -l | xargs printf "%.1f") MB/token)"
echo ""

# Calculate per-token metrics
echo "=================================="
echo "Per-Token Breakdown"
echo "=================================="
echo ""
MATVEC_PER_TOK=$(awk "BEGIN {printf \"%.1f\", $MATVEC / $TOKENS}")
RMSNORM_PER_TOK=$(awk "BEGIN {printf \"%.1f\", $RMSNORM / $TOKENS}")
SYNCS_PER_TOK=$(awk "BEGIN {printf \"%.1f\", $SYNCS / $TOKENS}")
H2D_PER_TOK=$(awk "BEGIN {printf \"%.1f\", $H2D / $TOKENS}")
D2H_PER_TOK=$(awk "BEGIN {printf \"%.1f\", $D2H / $TOKENS}")

echo "Operations per token:"
echo "  MatVec:   $MATVEC_PER_TOK"
echo "  RMSNorm:  $RMSNORM_PER_TOK"
echo "  Syncs:    $SYNCS_PER_TOK"
echo ""
echo "Data per token:"
echo "  H2D: $H2D_PER_TOK MB"
echo "  D2H: $D2H_PER_TOK MB"
echo ""

# Simple bottleneck heuristics
echo "=================================="
echo "Bottleneck Analysis"
echo "=================================="
echo ""

TOTAL_TRANSFER_PER_TOK=$(awk "BEGIN {printf \"%.1f\", $H2D_PER_TOK + $D2H_PER_TOK}")

if (( $(awk "BEGIN {print ($SYNCS_PER_TOK >= 5)}") )); then
    echo "⚠️  HIGH stream sync overhead ($SYNCS_PER_TOK per token)"
    echo "   → Each sync stalls CPU-GPU pipeline"
    echo ""
fi

if (( $(awk "BEGIN {print ($TOTAL_TRANSFER_PER_TOK >= 8)}") )); then
    echo "⚠️  HIGH memory transfer ($TOTAL_TRANSFER_PER_TOK MB per token)"
    echo "   → Memory bandwidth likely limiting factor"
    echo ""
fi

if (( $(awk "BEGIN {print ($RMSNORM_PER_TOK >= 250)}") )); then
    echo "⚠️  VERY HIGH RMSNorm count ($RMSNORM_PER_TOK per token)"
    echo "   → Many small kernels, launch overhead"
    echo ""
fi

echo "=================================="
echo "Optimization Priority"
echo "=================================="
echo ""

if (( $(awk "BEGIN {print ($SYNCS_PER_TOK >= 5)}") )); then
    echo "1. REDUCE STREAM SYNCS"
    echo "   Current: $SYNCS_PER_TOK syncs/token"
    echo "   Target:  1-2 syncs/token"
    echo "   How:     Keep data on GPU, batch D2H copies"
    echo ""
fi

if (( $(awk "BEGIN {print ($RMSNORM_PER_TOK >= 200)}") )); then
    echo "2. FUSE RMSNORM OPERATIONS"
    echo "   Current: $RMSNORM_PER_TOK RMSNorm/token"
    echo "   Target:  Fuse with MatVec in Attention/FFN"
    echo "   How:     Combine RMSNorm+QKV, RMSNorm+FFN gates"
    echo ""
fi

if (( $(awk "BEGIN {print ($TOTAL_TRANSFER_PER_TOK >= 8)}") )); then
    echo "3. REDUCE MEMORY TRANSFERS"
    echo "   Current: $TOTAL_TRANSFER_PER_TOK MB/token"
    echo "   Target:  <5 MB/token"
    echo "   How:     Device-resident intermediates, async transfers"
    echo ""
fi

echo "Current TPS: $GEN_TPS"
echo ""

rm -f "$TMP_OUT"
