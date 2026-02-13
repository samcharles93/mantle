#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage:"
	echo "  $0 --final <simd_profile_id> --optimisation <text> --functions <text> [--baseline <simd_profile_id>] [--notes <text>] [--csv <path>] [--dry-run]"
	echo ""
	echo "Example:"
	echo "  $0 --baseline simd_profile_20260213_180000 \\"
	echo "     --final simd_profile_20260213_181500 \\"
	echo "     --optimisation \"Reuse quant decode scratch in single-threaded quant matvec\" \\"
	echo "     --functions \"simd.matVecRangeQWithWorker; simd.matVecRangeKWithWorker\""
}

csv_escape() {
	local s="${1//\"/\"\"}"
	printf '"%s"' "$s"
}

extract_metric() {
	local file="$1"
	local bench="$2"
	local col="$3"
	awk -F, -v b="$bench" -v c="$col" '$1 == b {print $c; exit}' "$file"
}

extract_metric_or_empty() {
	local file="$1"
	local bench="$2"
	local col="$3"
	if [[ -z "$file" ]]; then
		echo ""
		return
	fi
	extract_metric "$file" "$bench" "$col"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_ID=""
FINAL_ID=""
OPTIMISATION=""
FUNCTIONS=""
NOTES=""
CSV_PATH="${ROOT_DIR}/profiles/simd_optimisation_log.csv"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--baseline)
		BASE_ID="$2"
		shift 2
		;;
	--final)
		FINAL_ID="$2"
		shift 2
		;;
	--optimisation)
		OPTIMISATION="$2"
		shift 2
		;;
	--functions)
		FUNCTIONS="$2"
		shift 2
		;;
	--notes)
		NOTES="$2"
		shift 2
		;;
	--csv)
		CSV_PATH="$2"
		shift 2
		;;
	--dry-run)
		DRY_RUN=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "error: unknown argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

if [[ -z "$FINAL_ID" || -z "$OPTIMISATION" || -z "$FUNCTIONS" ]]; then
	echo "error: --final, --optimisation, and --functions are required" >&2
	usage
	exit 1
fi

BASE_STATS=""
if [[ -n "$BASE_ID" ]]; then
	BASE_STATS="${ROOT_DIR}/profiles/${BASE_ID}_stats.csv"
	if [[ ! -f "$BASE_STATS" ]]; then
		echo "error: baseline stats not found: $BASE_STATS" >&2
		exit 1
	fi
fi

FINAL_STATS="${ROOT_DIR}/profiles/${FINAL_ID}_stats.csv"
if [[ ! -f "$FINAL_STATS" ]]; then
	echo "error: final stats not found: $FINAL_STATS" >&2
	exit 1
fi

DATE_VAL="$(date +%Y-%m-%d)"
TIME_VAL="$(date +%H:%M:%S)"

BASE_Q4_NS="$(extract_metric_or_empty "$BASE_STATS" "BenchmarkMatVecQ4" 2)"
FINAL_Q4_NS="$(extract_metric "$FINAL_STATS" "BenchmarkMatVecQ4" 2)"
BASE_Q4_B="$(extract_metric_or_empty "$BASE_STATS" "BenchmarkMatVecQ4" 3)"
FINAL_Q4_B="$(extract_metric "$FINAL_STATS" "BenchmarkMatVecQ4" 3)"
BASE_Q4_ALLOCS="$(extract_metric_or_empty "$BASE_STATS" "BenchmarkMatVecQ4" 4)"
FINAL_Q4_ALLOCS="$(extract_metric "$FINAL_STATS" "BenchmarkMatVecQ4" 4)"

BASE_K4_NS="$(extract_metric_or_empty "$BASE_STATS" "BenchmarkMatVecK4" 2)"
FINAL_K4_NS="$(extract_metric "$FINAL_STATS" "BenchmarkMatVecK4" 2)"
BASE_K4_B="$(extract_metric_or_empty "$BASE_STATS" "BenchmarkMatVecK4" 3)"
FINAL_K4_B="$(extract_metric "$FINAL_STATS" "BenchmarkMatVecK4" 3)"
BASE_K4_ALLOCS="$(extract_metric_or_empty "$BASE_STATS" "BenchmarkMatVecK4" 4)"
FINAL_K4_ALLOCS="$(extract_metric "$FINAL_STATS" "BenchmarkMatVecK4" 4)"

if [[ ! -f "$CSV_PATH" ]]; then
	mkdir -p "$(dirname "$CSV_PATH")"
	cat >"$CSV_PATH" <<'EOF'
date,time,optimisation,functions_touched,baseline_matvec_q4_ns_per_op,final_matvec_q4_ns_per_op,baseline_matvec_q4_b_per_op,final_matvec_q4_b_per_op,baseline_matvec_q4_allocs_per_op,final_matvec_q4_allocs_per_op,baseline_matvec_k4_ns_per_op,final_matvec_k4_ns_per_op,baseline_matvec_k4_b_per_op,final_matvec_k4_b_per_op,baseline_matvec_k4_allocs_per_op,final_matvec_k4_allocs_per_op,baseline_profile_ref,final_profile_ref,notes
EOF
fi

ROW="${DATE_VAL},${TIME_VAL},$(csv_escape "$OPTIMISATION"),$(csv_escape "$FUNCTIONS"),${BASE_Q4_NS},${FINAL_Q4_NS},${BASE_Q4_B},${FINAL_Q4_B},${BASE_Q4_ALLOCS},${FINAL_Q4_ALLOCS},${BASE_K4_NS},${FINAL_K4_NS},${BASE_K4_B},${FINAL_K4_B},${BASE_K4_ALLOCS},${FINAL_K4_ALLOCS},${BASE_ID},${FINAL_ID},$(csv_escape "$NOTES")"

if [[ "$DRY_RUN" -eq 1 ]]; then
	echo "$ROW"
	exit 0
fi

echo "$ROW" >>"$CSV_PATH"
echo "Appended optimisation row to ${CSV_PATH}"
