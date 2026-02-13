#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage:"
	echo "  $0 --baseline <profile_id> --final <profile_id> --optimisation <text> --functions <text> [--notes <text>] [--csv <path>] [--dry-run]"
	echo ""
	echo "Example:"
	echo "  $0 --baseline cuda_profile_20260213_170540 \\"
	echo "     --final cuda_profile_20260213_171203 \\"
	echo "     --optimisation \"Keep residual x device-resident\" \\"
	echo "     --functions \"simd.ForwardToken; cuda.Ops.DeviceAdd\""
}

csv_escape() {
	local s="${1//\"/\"\"}"
	printf '"%s"' "$s"
}

extract_first() {
	local cmd="$1"
	local value
	value="$(eval "$cmd" | head -n 1 || true)"
	if [[ -z "$value" ]]; then
		echo "error: failed to extract metric using: $cmd" >&2
		exit 1
	fi
	echo "$value"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_ID=""
FINAL_ID=""
OPTIMISATION=""
FUNCTIONS=""
NOTES=""
CSV_PATH="${ROOT_DIR}/profiles/cuda_optimisation_log.csv"
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

if [[ -z "$BASE_ID" || -z "$FINAL_ID" || -z "$OPTIMISATION" || -z "$FUNCTIONS" ]]; then
	echo "error: missing required arguments" >&2
	usage
	exit 1
fi

BASE_STATS="${ROOT_DIR}/profiles/${BASE_ID}_stats.txt"
FINAL_STATS="${ROOT_DIR}/profiles/${FINAL_ID}_stats.txt"
BASE_INTERNAL="${ROOT_DIR}/profiles/${BASE_ID}_internal.txt"
FINAL_INTERNAL="${ROOT_DIR}/profiles/${FINAL_ID}_internal.txt"

for f in "$BASE_STATS" "$FINAL_STATS" "$BASE_INTERNAL" "$FINAL_INTERNAL"; do
	if [[ ! -f "$f" ]]; then
		echo "error: required profile artifact not found: $f" >&2
		exit 1
	fi
done

DATE_VAL="$(extract_first "rg -o '\\[[0-9-]+ [0-9:]+\\].*generation complete' '$FINAL_INTERNAL' | sed -E 's/^\\[([0-9-]+) ([0-9:]+)\\].*/\\1/'")"
TIME_VAL="$(extract_first "rg -o '\\[[0-9-]+ [0-9:]+\\].*generation complete' '$FINAL_INTERNAL' | sed -E 's/^\\[([0-9-]+) ([0-9:]+)\\].*/\\2/'")"

BASE_TPS="$(extract_first "rg -o 'tps=[0-9.]+' '$BASE_INTERNAL' | cut -d= -f2")"
FINAL_TPS="$(extract_first "rg -o 'tps=[0-9.]+' '$FINAL_INTERNAL' | cut -d= -f2")"
BASE_PROMPT_TPS="$(extract_first "rg -o 'prompt_tps=[0-9.]+' '$BASE_INTERNAL' | cut -d= -f2")"
FINAL_PROMPT_TPS="$(extract_first "rg -o 'prompt_tps=[0-9.]+' '$FINAL_INTERNAL' | cut -d= -f2")"
BASE_GEN_TPS="$(extract_first "rg -o 'gen_tps=[0-9.]+' '$BASE_INTERNAL' | cut -d= -f2")"
FINAL_GEN_TPS="$(extract_first "rg -o 'gen_tps=[0-9.]+' '$FINAL_INTERNAL' | cut -d= -f2")"

BASE_CUDAMEMCPY_API_NS="$(extract_first "awk '/cudaMemcpy[[:space:]]*$/ {print \$2; exit}' '$BASE_STATS'")"
FINAL_CUDAMEMCPY_API_NS="$(extract_first "awk '/cudaMemcpy[[:space:]]*$/ {print \$2; exit}' '$FINAL_STATS'")"

BASE_H2D_COUNT="$(extract_first "rg '\\[CUDA memcpy Host-to-Device\\]' '$BASE_STATS' | head -1 | awk '{print \$3}'")"
FINAL_H2D_COUNT="$(extract_first "rg '\\[CUDA memcpy Host-to-Device\\]' '$FINAL_STATS' | head -1 | awk '{print \$3}'")"
BASE_D2H_COUNT="$(extract_first "rg '\\[CUDA memcpy Device-to-Host\\]' '$BASE_STATS' | head -1 | awk '{print \$3}'")"
FINAL_D2H_COUNT="$(extract_first "rg '\\[CUDA memcpy Device-to-Host\\]' '$FINAL_STATS' | head -1 | awk '{print \$3}'")"

BASE_H2D_TOTAL_NS="$(extract_first "rg '\\[CUDA memcpy Host-to-Device\\]' '$BASE_STATS' | head -1 | awk '{print \$2}'")"
FINAL_H2D_TOTAL_NS="$(extract_first "rg '\\[CUDA memcpy Host-to-Device\\]' '$FINAL_STATS' | head -1 | awk '{print \$2}'")"
BASE_D2H_TOTAL_NS="$(extract_first "rg '\\[CUDA memcpy Device-to-Host\\]' '$BASE_STATS' | head -1 | awk '{print \$2}'")"
FINAL_D2H_TOTAL_NS="$(extract_first "rg '\\[CUDA memcpy Device-to-Host\\]' '$FINAL_STATS' | head -1 | awk '{print \$2}'")"

BASE_H2D_MB="$(extract_first "rg -o 'H2D bytes: [0-9]+' '$BASE_INTERNAL' | awk '{print \$3}'")"
FINAL_H2D_MB="$(extract_first "rg -o 'H2D bytes: [0-9]+' '$FINAL_INTERNAL' | awk '{print \$3}'")"
BASE_D2H_MB="$(extract_first "rg -o 'D2H bytes: [0-9]+' '$BASE_INTERNAL' | awk '{print \$3}'")"
FINAL_D2H_MB="$(extract_first "rg -o 'D2H bytes: [0-9]+' '$FINAL_INTERNAL' | awk '{print \$3}'")"

if [[ ! -f "$CSV_PATH" ]]; then
	mkdir -p "$(dirname "$CSV_PATH")"
	cat >"$CSV_PATH" <<'EOF'
date,time,optimisation,functions_touched,baseline_tps,final_tps,baseline_prompt_tps,final_prompt_tps,baseline_gen_tps,final_gen_tps,baseline_cudaMemcpy_api_total_ns,final_cudaMemcpy_api_total_ns,baseline_memcpy_h2d_count,final_memcpy_h2d_count,baseline_memcpy_d2h_count,final_memcpy_d2h_count,baseline_memcpy_h2d_total_ns,final_memcpy_h2d_total_ns,baseline_memcpy_d2h_total_ns,final_memcpy_d2h_total_ns,baseline_internal_h2d_mb,final_internal_h2d_mb,baseline_internal_d2h_mb,final_internal_d2h_mb,baseline_profile_ref,final_profile_ref,notes
EOF
fi

ROW="${DATE_VAL},${TIME_VAL},$(csv_escape "$OPTIMISATION"),$(csv_escape "$FUNCTIONS"),${BASE_TPS},${FINAL_TPS},${BASE_PROMPT_TPS},${FINAL_PROMPT_TPS},${BASE_GEN_TPS},${FINAL_GEN_TPS},${BASE_CUDAMEMCPY_API_NS},${FINAL_CUDAMEMCPY_API_NS},${BASE_H2D_COUNT},${FINAL_H2D_COUNT},${BASE_D2H_COUNT},${FINAL_D2H_COUNT},${BASE_H2D_TOTAL_NS},${FINAL_H2D_TOTAL_NS},${BASE_D2H_TOTAL_NS},${FINAL_D2H_TOTAL_NS},${BASE_H2D_MB},${FINAL_H2D_MB},${BASE_D2H_MB},${FINAL_D2H_MB},${BASE_ID},${FINAL_ID},$(csv_escape "$NOTES")"

if [[ "$DRY_RUN" -eq 1 ]]; then
	echo "$ROW"
	exit 0
fi

echo "$ROW" >>"$CSV_PATH"
echo "Appended optimisation row to ${CSV_PATH}"
