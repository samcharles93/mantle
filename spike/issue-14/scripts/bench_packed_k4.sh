#!/usr/bin/env bash
# Three-way CPU comparison on one 4-bit container: a branch build with the packed
# 4-bit matvec (cache disabled), the same build with the int8 cache, and the
# llama.cpp CPU-only baseline. Configs rotate order every repeat so noise from a
# busy box lands on all three.
# Usage: bench_packed_k4.sh <repeats> <mcf> <gguf>
set -u

REPEATS="${1:-3}"
MCF="${2:?mcf}"
GGUF="${3:?gguf}"

BRANCH_BIN="${BRANCH_BIN:-/home/sam/tau-runs/worktrees/quant-quality/bin/mantle}"
PLAIN_BIN="${PLAIN_BIN:-/work/wt/mantle-quant14/bin/mantle}"
LLAMA_BENCH="${LLAMA_BENCH:-/work/tmp/llama-cpu/bin/llama-bench}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/out}/packed-k4.tsv"
STEPS="${STEPS:-64}"
THREADS="${THREADS:-16}"
PROMPT="${PROMPT:-Explain in one paragraph why the sky appears blue, then list three colours.}"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'config\trep\tpp_tps\ttg_tps\tpeak_rss_kb\n' >"$OUT"
fi

run_mantle() {
	local config="$1" bin="$2"
	shift 2
	local raw pp tg rss
	raw="$(mktemp)"
	# shellcheck disable=SC2086
	/usr/bin/time -v "$bin" benchmark --model "$MCF" --backend cpu \
		--prompt "$PROMPT" --steps "$STEPS" --warmup 1 --runs 1 "$@" >"$raw" 2>&1
	if [ $? -ne 0 ]; then
		echo "FAIL $config" >&2
		tail -5 "$raw" >&2
		rm -f "$raw"
		return
	fi
	pp=$(awk '/^Avg/ {print $2; exit}' "$raw")
	tg=$(awk '/^Avg/ {print $3; exit}' "$raw")
	rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
	printf '%s\t%s\t%s\t%s\t%s\n' "$config" "$REP" "$pp" "$tg" "$rss" >>"$OUT"
	echo "$config rep=$REP pp=$pp tg=$tg rss=${rss}kB" >&2
	rm -f "$raw"
}

run_llama() {
	local raw pp tg rss
	raw="$(mktemp)"
	/usr/bin/time -v "$LLAMA_BENCH" -m "$GGUF" -p "$STEPS" -n "$STEPS" -r 1 \
		-t "$THREADS" -ngl 0 >"$raw" 2>&1
	if [ $? -ne 0 ]; then
		echo "FAIL llama" >&2
		tail -5 "$raw" >&2
		rm -f "$raw"
		return
	fi
	pp=$(awk -F'|' '/\| *pp[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"\u00b1"); print a[1]; exit}' "$raw")
	tg=$(awk -F'|' '/\| *tg[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"\u00b1"); print a[1]; exit}' "$raw")
	rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
	printf '%s\t%s\t%s\t%s\t%s\n' "llama-q4km" "$REP" "$pp" "$tg" "$rss" >>"$OUT"
	echo "llama-q4km rep=$REP pp=$pp tg=$tg rss=${rss}kB" >&2
	rm -f "$raw"
}

for REP in $(seq 1 "$REPEATS"); do
	case $((REP % 3)) in
	1) order="packed cached llama" ;;
	2) order="llama packed cached" ;;
	0) order="cached llama packed" ;;
	esac
	for who in $order; do
		case "$who" in
		packed) run_mantle "branch-packed-nocache" "$BRANCH_BIN" --no-quant-cache ;;
		cached) run_mantle "branch-cached" "$BRANCH_BIN" ;;
		llama) run_llama ;;
		esac
	done
done
