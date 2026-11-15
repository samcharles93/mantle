#!/usr/bin/env bash
# Measures llama.cpp CPU prompt/decode throughput and peak RSS for one or more GGUF files.
# Usage: bench_llamacpp.sh <out.tsv> <repeat> <gguf>...
# One TSV row per (file, repeat): label, size_bytes, pp_tps, tg_tps, peak_rss_kb.
set -u

BENCH="${LLAMA_BENCH:-/work/ai/llama.cpp/build/bin/llama-bench}"
OUT="$1"
REPEAT="$2"
shift 2

THREADS="${THREADS:-16}"
NPROMPT="${NPROMPT:-128}"
NGEN="${NGEN:-128}"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'label\tsize_bytes\tpp_tps\ttg_tps\tpeak_rss_kb\n' >"$OUT"
fi

for f in "$@"; do
	label="$(basename "$f" .gguf)"
	for rep in $(seq 1 "$REPEAT"); do
		raw="$(mktemp)"
		/usr/bin/time -v "$BENCH" -m "$f" -p "$NPROMPT" -n "$NGEN" -r 3 \
			-t "$THREADS" -dev CPU >"$raw" 2>&1
		rc=$?
		if [ "$rc" -ne 0 ]; then
			echo "FAIL $label rep=$rep rc=$rc" >&2
			tail -20 "$raw" >&2
			rm -f "$raw"
			continue
		fi
		size=$(stat -c %s "$f")
		pp=$(awk -F'|' '/\| *pp[0-9]+ *\|/ {gsub(/ /,"",$8); print $8; exit}' "$raw")
		tg=$(awk -F'|' '/\| *tg[0-9]+ *\|/ {gsub(/ /,"",$8); print $8; exit}' "$raw")
		rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
		printf '%s\t%s\t%s\t%s\t%s\n' \
			"$label" "$size" "${pp:-0}" "${tg:-0}" "${rss:-0}" >>"$OUT"
		echo "$label rep=$rep pp=$pp tg=$tg rss=${rss}kB" >&2
		rm -f "$raw"
	done
done
