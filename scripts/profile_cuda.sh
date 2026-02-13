#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANTLE_BIN="${SCRIPT_DIR}/../bin/mantle"
MODEL="${1:-/work/models/mcf/LFM2.5-1.2B-Instruct.k4.mcf}"
STEPS="${2:-128}"
KV_TYPE="${3:-q8_0}"

if [ ! -f "$MANTLE_BIN" ]; then
    echo "Error: mantle binary not found at $MANTLE_BIN"
    echo "Run: CUDA=1 task build -f"
    exit 1
fi

PROFILE_DIR="${SCRIPT_DIR}/../profiles"
mkdir -p "$PROFILE_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_BASE="${PROFILE_DIR}/cuda_profile_${TIMESTAMP}"

echo "=================================="
echo "CUDA Profiling"
echo "=================================="
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "Cache Type K/V: $KV_TYPE"
echo "Output: ${REPORT_BASE}.*"
echo ""

# Check if nsys is available
if command -v nsys &> /dev/null; then
    echo "Using NVIDIA Nsight Systems (nsys)..."
    echo ""

    # Profile with nsys
    nsys profile \
        --output="${REPORT_BASE}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        --export=sqlite \
        "$MANTLE_BIN" run --backend cuda \
        -m "$MODEL" \
        --steps "$STEPS"  --ctv "$KV_TYPE" --ctk "$KV_TYPE" \
        --prompt "Write a short story about Python (programming language)"

    echo ""
    echo "=================================="
    echo "Generating statistics..."
    echo ""

    # Generate detailed stats
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "${REPORT_BASE}.nsys-rep" \
        > "${REPORT_BASE}_stats.txt"

    echo "Profile captured: ${REPORT_BASE}.nsys-rep"
    echo "Statistics saved: ${REPORT_BASE}_stats.txt"
    echo ""
    echo "View with: nsys-ui ${REPORT_BASE}.nsys-rep"
    echo "Or read stats: cat ${REPORT_BASE}_stats.txt"
    echo ""

elif command -v nvprof &> /dev/null; then
    echo "Using nvprof (legacy)..."
    echo ""

    # Profile with nvprof
    nvprof \
        --print-gpu-trace \
        --print-api-trace \
        --log-file "${REPORT_BASE}_trace.txt" \
        --export-profile "${REPORT_BASE}.nvvp" \
        "$MANTLE_BIN" run --backend cuda \
        -m "$MODEL" \
        --steps "$STEPS" --ctv "$KV_TYPE" --ctk "$KV_TYPE" \
        --prompt "Write a short story about Python (programming language)"

    echo ""
    echo "=================================="
    echo "Profile captured:"
    echo "  Trace: ${REPORT_BASE}_trace.txt"
    echo "  NVVP: ${REPORT_BASE}.nvvp"
    echo ""
    echo "View trace: less ${REPORT_BASE}_trace.txt"
    echo "View NVVP: nvvp ${REPORT_BASE}.nvvp"
    echo ""

else
    echo "Error: No CUDA profiler found!"
    echo "Install NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems"
    echo "Or use nvprof (deprecated)"
    exit 1
fi

# Also run with internal trace for comparison
echo "=================================="
echo "Running with internal CUDA trace..."
echo ""

CUDA=1 MANTLE_CUDA_TRACE=1 "$MANTLE_BIN" run --backend cuda \
    -m "$MODEL" \
    --steps "$STEPS" --ctv "$KV_TYPE" --ctk "$KV_TYPE"  \
    --prompt "Write a short story about Python (programming language)" \
    > "${REPORT_BASE}_internal.txt" 2>&1

echo ""
echo "Internal trace saved: ${REPORT_BASE}_internal.txt"
echo ""

# Summary
echo "=================================="
echo "Profiling Complete!"
echo "=================================="
echo ""
echo "Files generated:"
ls -lh "${REPORT_BASE}"* 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. View nsys report: nsys-ui ${REPORT_BASE}.nsys-rep"
echo "  2. View internal trace: cat ${REPORT_BASE}_internal.txt"
echo "  3. Export stats: nsys stats ${REPORT_BASE}.nsys-rep"
