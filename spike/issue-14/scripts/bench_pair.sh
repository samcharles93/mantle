#!/usr/bin/env bash
# Measures one mantle container and one llama.cpp GGUF back to back, repeat times.
# Both sides of a pair are sampled within seconds of each other, so noise from a
# busy desktop (or another agent's job) lands on both and cancels in the ratio.
# Usage: bench_pair.sh <repeats> <label> <mcf> <gguf>
set -u

REPEATS="${1:?repeats}"
LABEL="${2:?label}"
MCF="${3:?mcf}"
GGUF="${4:?gguf}"

BIN="${BIN:-/work/wt/mantle-quant14/bin/mantle}"
LLAMA_BENCH="${LLAMA_BENCH:-/work/tmp/llama-cpu/bin/llama-bench}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/out}/pairs.tsv"
STEPS="${STEPS:-64}"
THREADS="${THREADS:-16}"
MANTLE_EXTRA="${MANTLE_EXTRA:-}"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'pair\trep\tengine\tlabel\tpp_tps\ttg_tps\tpeak_rss_kb\n' >"$OUT"
fi

for rep in $(seq 1 "$REPEATS"); do
	# mantle first on odd reps, llama first on even reps, to balance ordering bias.
	order="mantle llama"
	if [ $((rep % 2)) -eq 0 ]; then
		order="llama mantle"
	fi
	for who in $order; do
		raw="$(mktemp)"
		if [ "$who" = "mantle" ]; then
			# shellcheck disable=SC2086
			/usr/bin/time -v "$BIN" benchmark --model "$MCF" --backend cpu \
				--prompt "Explain in one paragraph why the sky appears blue, then list three colours." \
				--steps "$STEPS" --warmup 1 --runs 1 $MANTLE_EXTRA >"$raw" 2>&1
			pp=$(awk '/^Avg/ {print $2; exit}' "$raw")
			tg=$(awk '/^Avg/ {print $3; exit}' "$raw")
			name="$(basename "$MCF" .mcf)"
		else
			/usr/bin/time -v "$LLAMA_BENCH" -m "$GGUF" -p "$STEPS" -n "$STEPS" -r 1 \
				-t "$THREADS" -ngl 0 >"$raw" 2>&1
			pp=$(awk -F'|' '/\| *pp[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"\u00b1"); print a[1]; exit}' "$raw")
			tg=$(awk -F'|' '/\| *tg[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"\u00b1"); print a[1]; exit}' "$raw")
			name="$(basename "$GGUF" .gguf)"
		fi
		rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
		printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
			"$LABEL" "$rep" "$who" "$name" "${pp:-0}" "${tg:-0}" "${rss:-0}" >>"$OUT"
		echo "$LABEL rep=$rep $who $name pp=$pp tg=$tg rss=${rss}kB" >&2
		rm -f "$raw"
	done
done
