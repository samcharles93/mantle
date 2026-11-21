#!/usr/bin/env bash
# Measures CUDA throughput, peak VRAM and peak RSS for MCF containers.
# Usage: bench_cuda.sh <out.tsv> <repeat> <mcf>...
# Modes come from CUDA_MODES (default "quant dequant"); each mode is a separate row.
set -u

BIN="${BIN:-/work/wt/mantle-quant14/bin/mantle-cuda}"
OUT="$1"
REPEAT="$2"
shift 2

STEPS="${STEPS:-64}"
RUNS="${RUNS:-2}"
WARMUP="${WARMUP:-1}"
CUDA_MODES="${CUDA_MODES:-quant dequant}"
PROMPT="${PROMPT:-Explain in one paragraph why the sky appears blue, then list three colours.}"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'label\tmode\tsize_bytes\tpp_tps\ttg_tps\tpeak_vram_mib\tpeak_rss_kb\n' >"$OUT"
fi

for mcf in "$@"; do
	label="$(basename "$mcf" .mcf)"
	for mode in $CUDA_MODES; do
		for rep in $(seq 1 "$REPEAT"); do
			raw="$(mktemp)"
			vram="$(mktemp)"
			(
				while true; do
					nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
					sleep 0.1
				done >"$vram"
			) &
			sampler=$!
			MANTLE_CUDA_WEIGHT_MODE="$mode" /usr/bin/time -v "$BIN" benchmark \
				--model "$mcf" --backend cuda --prompt "$PROMPT" \
				--steps "$STEPS" --warmup "$WARMUP" --runs "$RUNS" >"$raw" 2>&1
			rc=$?
			kill "$sampler" 2>/dev/null
			wait "$sampler" 2>/dev/null
			if [ "$rc" -ne 0 ]; then
				echo "FAIL $label/$mode rep=$rep rc=$rc" >&2
				tail -5 "$raw" >&2
				rm -f "$raw" "$vram"
				continue
			fi
			peakvram=$(sort -n "$vram" | tail -1)
			pp=$(awk '/^Avg/ {print $2; exit}' "$raw")
			tg=$(awk '/^Avg/ {print $3; exit}' "$raw")
			rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
			printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
				"$label" "$mode" "$(stat -c %s "$mcf")" "${pp:-0}" "${tg:-0}" "${peakvram:-0}" "${rss:-0}" >>"$OUT"
			echo "$label mode=$mode rep=$rep pp=$pp tg=$tg vram=${peakvram}MiB rss=${rss}kB" >&2
			rm -f "$raw" "$vram"
		done
	done
done
