#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANTLE_BIN="${SCRIPT_DIR}/../bin/mantle"
MODEL="${1:-/work/models/mcf/LFM2.5-1.2B-Instruct.k4.mcf}"
STEPS="${2:-128}"
KV_TYPE="${3:-q8_0}"
PROMPT="${PROMPT:-Write a short story about rainbows}"
SEED="${SEED:-123}"
TEMP="${TEMP:-0}"
STREAM_MODE="${STREAM_MODE:-quiet}"
GRAPHS="${GRAPHS:-1}"
NSYS_GRAPHS="${NSYS_GRAPHS:-0}"
TRACE_SYNC="${TRACE_SYNC:-0}"
CUDA_GRAPH_TRACE="${CUDA_GRAPH_TRACE:-node}"

if [ -t 1 ]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m'
    C_CYAN=$'\033[36m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_MAGENTA=$'\033[35m'
else
    C_RESET=""
    C_BOLD=""
    C_BLUE=""
    C_CYAN=""
    C_GREEN=""
    C_YELLOW=""
    C_MAGENTA=""
fi

section() {
    echo "${C_BOLD}${C_BLUE}==================================${C_RESET}"
    echo "${C_BOLD}${C_BLUE}$1${C_RESET}"
    echo "${C_BOLD}${C_BLUE}==================================${C_RESET}"
}

kv() {
    printf "%s%-18s%s %s\n" "${C_CYAN}" "$1:" "${C_RESET}" "$2"
}

if [ ! -f "$MANTLE_BIN" ]; then
    echo "Error: mantle binary not found at $MANTLE_BIN"
    echo "Run: CUDA=1 task build -f"
    exit 1
fi

PROFILE_DIR="${SCRIPT_DIR}/../profiles"
mkdir -p "$PROFILE_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_BASE="${PROFILE_DIR}/cuda_profile_${TIMESTAMP}"

section "CUDA Profiling"
kv "Model" "$MODEL"
kv "Steps" "$STEPS"
kv "Cache Type K/V" "$KV_TYPE"
kv "Graphs" "$GRAPHS (MANTLE_CUDA_GRAPHS)"
kv "Nsight Graphs" "$NSYS_GRAPHS (nsys capture)"
kv "Graph trace" "$CUDA_GRAPH_TRACE (--cuda-graph-trace)"
kv "Trace sync checks" "$TRACE_SYNC (MANTLE_CUDA_TRACE_SYNC)"
kv "Stream mode" "$STREAM_MODE"
kv "Seed/Temp" "$SEED / $TEMP"
kv "Output" "${REPORT_BASE}.*"
echo ""

# Check if nsys is available
if command -v nsys &> /dev/null; then
    echo "${C_BOLD}${C_MAGENTA}Using NVIDIA Nsight Systems (nsys)...${C_RESET}"
    echo ""

    # Profile with nsys
    CUDA=1 MANTLE_CUDA_GRAPHS="$NSYS_GRAPHS" MANTLE_CUDA_TRACE_SYNC="$TRACE_SYNC" nsys profile \
        --output="${REPORT_BASE}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace="$CUDA_GRAPH_TRACE" \
        --cuda-memory-usage=true \
        --stats=true \
        --export=sqlite \
        "$MANTLE_BIN" run --backend cuda \
        -m "$MODEL" \
        --steps "$STEPS" --ctv "$KV_TYPE" --ctk "$KV_TYPE" \
        --seed "$SEED" --temp "$TEMP" --stream-mode "$STREAM_MODE" \
        --prompt "$PROMPT"

    echo ""
    section "Generating Statistics"
    echo ""

    # Generate detailed stats
    nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum "${REPORT_BASE}.nsys-rep" \
        > "${REPORT_BASE}_stats.txt"

    kv "Profile captured" "${REPORT_BASE}.nsys-rep"
    kv "Statistics saved" "${REPORT_BASE}_stats.txt"
    echo ""
    kv "View with" "nsys-ui ${REPORT_BASE}.nsys-rep"
    kv "Read stats" "cat ${REPORT_BASE}_stats.txt"
    echo ""

elif command -v nvprof &> /dev/null; then
    echo "${C_BOLD}${C_MAGENTA}Using nvprof (legacy)...${C_RESET}"
    echo ""

    # Profile with nvprof
    CUDA=1 MANTLE_CUDA_GRAPHS="$GRAPHS" MANTLE_CUDA_TRACE_SYNC="$TRACE_SYNC" nvprof \
        --print-gpu-trace \
        --print-api-trace \
        --log-file "${REPORT_BASE}_trace.txt" \
        --export-profile "${REPORT_BASE}.nvvp" \
        "$MANTLE_BIN" run --backend cuda \
        -m "$MODEL" \
        --steps "$STEPS" --ctv "$KV_TYPE" --ctk "$KV_TYPE" \
        --seed "$SEED" --temp "$TEMP" --stream-mode "$STREAM_MODE" \
        --prompt "$PROMPT"

    echo ""
    section "Profile Captured"
    kv "Trace" "${REPORT_BASE}_trace.txt"
    kv "NVVP" "${REPORT_BASE}.nvvp"
    echo ""
    kv "View trace" "less ${REPORT_BASE}_trace.txt"
    kv "View NVVP" "nvvp ${REPORT_BASE}.nvvp"
    echo ""

else
    echo "Error: No CUDA profiler found!"
    echo "Install NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems"
    echo "Or use nvprof (deprecated)"
    exit 1
fi

# Also run with internal trace for comparison
section "Running Internal CUDA Trace"
echo ""

CUDA=1 MANTLE_CUDA_TRACE=1 MANTLE_CUDA_GRAPHS="$GRAPHS" MANTLE_CUDA_TRACE_SYNC="$TRACE_SYNC" "$MANTLE_BIN" run --backend cuda \
    -m "$MODEL" \
    --steps "$STEPS" --ctv "$KV_TYPE" --ctk "$KV_TYPE" \
    --seed "$SEED" --temp "$TEMP" --stream-mode "$STREAM_MODE" \
    --prompt "$PROMPT" \
    > "${REPORT_BASE}_internal.txt" 2>&1

echo ""
kv "Internal trace saved" "${REPORT_BASE}_internal.txt"
echo "${C_BOLD}${C_YELLOW}Internal trace summary:${C_RESET}"
rg -n "generation complete|CUDA graph replay|-- Decode Counters --|-- Preload Counters --|Graph captures:|Graph launches:|Graph failures:|Stream syncs:|H2D bytes:|D2H bytes:" "${REPORT_BASE}_internal.txt" \
    | awk -v c_hdr="${C_YELLOW}" -v c_ok="${C_GREEN}" -v c_metric="${C_CYAN}" -v c_reset="${C_RESET}" '
{
    gsub(/\x1b\[[0-9;]*[A-Za-z]/, "", $0)
    if ($0 ~ /generation complete/) {
        print c_ok $0 c_reset
    } else if ($0 ~ /-- Decode Counters --|-- Preload Counters --/) {
        print c_hdr $0 c_reset
    } else if ($0 ~ /CUDA graph replay:|Graph captures:|Graph launches:|Graph failures:|Stream syncs:|H2D bytes:|D2H bytes:/) {
        print c_metric $0 c_reset
    } else {
        print $0
    }
}
' || true
echo ""

# Summary
section "Profiling Complete"
echo ""
echo "${C_BOLD}${C_GREEN}Files generated:${C_RESET}"
ls -lh "${REPORT_BASE}"* 2>/dev/null || true
echo ""
echo "${C_BOLD}${C_GREEN}Next steps:${C_RESET}"
echo "  ${C_CYAN}1.${C_RESET} View nsys report: nsys-ui ${REPORT_BASE}.nsys-rep"
echo "  ${C_CYAN}2.${C_RESET} View internal trace: cat ${REPORT_BASE}_internal.txt"
echo "  ${C_CYAN}3.${C_RESET} Export stats: nsys stats ${REPORT_BASE}.nsys-rep"
