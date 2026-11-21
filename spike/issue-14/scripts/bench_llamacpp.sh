#!/usr/bin/env bash
# Measures llama.cpp CPU prompt/decode throughput and peak RSS for one or more GGUF files.
# Uses the CPU-only build by default: a CUDA-enabled build still offloads prefill
# operators to the GPU under -ngl 0 unless -nopo 1 is passed, which inflates pp 4-5x.
# Usage: bench_llamacpp.sh <out.tsv> <repeat> <gguf>...
set -u

BENCH="${LLAMA_BENCH:-/work/tmp/llama-cpu/bin/llama-bench}"
OUT="$1"
REPEAT="$2"
shift 2

THREADS="${THREADS:-16}"
NPROMPT="${NPROMPT:-96}"
NGEN="${NGEN:-96}"
INNER="${INNER:-2}"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'label\tsize_bytes\tpp_tps\ttg_tps\tpeak_rss_kb\n' >"$OUT"
fi

for f in "$@"; do
	label="$(basename "$f" .gguf)"
	for rep in $(seq 1 "$REPEAT"); do
		raw="$(mktemp)"
		/usr/bin/time -v "$BENCH" -m "$f" -p "$NPROMPT" -n "$NGEN" -r "$INNER" \
			-t "$THREADS" -ngl 0 >"$raw" 2>&1
		rc=$?
		if [ "$rc" -ne 0 ]; then
			echo "FAIL $label rep=$rep rc=$rc" >&2
			tail -20 "$raw" >&2
			rm -f "$raw"
			continue
		fi
		size=$(stat -c %s "$f")
		# t/s is always the last data column, whatever the build's extra columns are.
		pp=$(awk -F'|' '/\| *pp[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"±"); print a[1]; exit}' "$raw")
		tg=$(awk -F'|' '/\| *tg[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"±"); print a[1]; exit}' "$raw")
		rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
		printf '%s\t%s\t%s\t%s\t%s\n' \
			"$label" "$size" "${pp:-0}" "${tg:-0}" "${rss:-0}" >>"$OUT"
		echo "$label rep=$rep pp=$pp tg=$tg rss=${rss}kB" >&2
		rm -f "$raw"
	done
done
