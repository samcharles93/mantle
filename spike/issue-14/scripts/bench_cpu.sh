#!/usr/bin/env bash
# Measures CPU decode throughput and peak RSS for one or more MCF containers.
# Usage: bench_cpu.sh <binary> <out.tsv> <repeat> <mcf>...
# One TSV row per (container, repeat): label, size_bytes, load_s, prompt_tps,
# gen_tps, peak_rss_kb.
set -u

BIN="$1"
OUT="$2"
REPEAT="$3"
shift 3

PROMPT="${PROMPT:-Explain in one paragraph why the sky appears blue, then list three colours.}"
STEPS="${STEPS:-128}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'label\tsize_bytes\tload_s\tprompt_tps\tgen_tps\tpeak_rss_kb\n' >"$OUT"
fi

for mcf in "$@"; do
	label="$(basename "$mcf" .mcf)"
	for rep in $(seq 1 "$REPEAT"); do
		tmp="$(mktemp)"
		/usr/bin/time -v "$BIN" benchmark \
			--model "$mcf" --backend cpu --prompt "$PROMPT" \
			--steps "$STEPS" --warmup "$WARMUP" --runs "$RUNS" \
			>"$tmp" 2>&1
		rc=$?
		if [ "$rc" -ne 0 ]; then
			echo "FAIL $label rep=$rep rc=$rc" >&2
			cat "$tmp" >&2
			rm -f "$tmp"
			continue
		fi
		size=$(stat -c %s "$mcf")
		load=$(sed -n 's/^Load:  *\([0-9.]*\)s$/\1/p' "$tmp" | head -1)
		avg=$(grep -E '^Avg' "$tmp" | head -1)
		ptps=$(echo "$avg" | awk '{print $2}')
		gtps=$(echo "$avg" | awk '{print $3}')
		rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$tmp" | head -1)
		printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
			"$label" "$size" "${load:-0}" "${ptps:-0}" "${gtps:-0}" "${rss:-0}" >>"$OUT"
		echo "$label rep=$rep gen=$gtps pp=$ptps rss=${rss}kB" >&2
		rm -f "$tmp"
	done
done
