#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage:"
	echo "  $0 [benchmark_regex] [count]"
	echo ""
	echo "Defaults:"
	echo "  benchmark_regex: BenchmarkMatVec(Q4|K4)$"
	echo "  count: 1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE_DIR="${ROOT_DIR}/profiles"
mkdir -p "$PROFILE_DIR"

BENCH_REGEX="${1:-BenchmarkMatVec(Q4|K4)$}"
COUNT="${2:-1}"

if [[ "${BENCH_REGEX}" == "-h" || "${BENCH_REGEX}" == "--help" ]]; then
	usage
	exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ID="simd_profile_${TIMESTAMP}"
RAW_OUT="${PROFILE_DIR}/${PROFILE_ID}_bench.txt"
STATS_OUT="${PROFILE_DIR}/${PROFILE_ID}_stats.csv"

echo "=================================="
echo "SIMD Profiling"
echo "=================================="
echo "Benchmark regex: ${BENCH_REGEX}"
echo "Count: ${COUNT}"
echo "Profile ID: ${PROFILE_ID}"
echo ""

GOEXPERIMENT=simd go1.26rc3 test ./internal/backend/simd \
	-run '^$' \
	-bench "${BENCH_REGEX}" \
	-benchmem \
	-count "${COUNT}" | tee "${RAW_OUT}"

{
	echo "benchmark,ns_per_op,b_per_op,allocs_per_op"
	awk '
		$1 ~ /^Benchmark/ && $0 ~ /ns\/op/ {
			bench=$1
			sub(/-[0-9]+$/, "", bench)
			print bench "," $3 "," $5 "," $7
		}
	' "${RAW_OUT}"
} >"${STATS_OUT}"

echo ""
echo "Saved:"
echo "  ${RAW_OUT}"
echo "  ${STATS_OUT}"
echo ""
echo "Profile ID: ${PROFILE_ID}"
