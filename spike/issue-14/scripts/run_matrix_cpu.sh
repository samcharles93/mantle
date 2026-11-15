#!/usr/bin/env bash
# Runs the CPU comparison matrix, interleaving mantle with the llama.cpp CPU-only
# baseline so desktop load drift hits both sides of the comparison equally.
# Usage: run_matrix_cpu.sh [repeats]
set -u

REPEATS="${1:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../out}"
BIN="${BIN:-/work/wt/mantle-quant14/bin/mantle}"
MCF_DIR="${MCF_DIR:-/work/models/mcf/probe/lfm3}"
GGUF_DIR="${GGUF_DIR:-/work/models/gguf/liquidai/LFM3-1.7B}"
LOG="$OUT/cpu-matrix.log"

mkdir -p "$OUT"
: >"$LOG"

export STEPS="${STEPS:-96}"
export RUNS="${RUNS:-2}"
export WARMUP="${WARMUP:-1}"
export NPROMPT="${NPROMPT:-96}" NGEN="${NGEN:-96}" THREADS="${THREADS:-16}"

for rep in $(seq 1 "$REPEATS"); do
	echo "=== repeat $rep/$REPEATS $(date +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) ===" | tee -a "$LOG"

	"$HERE/bench_cpu.sh" "$BIN" "$OUT/cpu-mantle.tsv" 1 \
		"$MCF_DIR/LFM3-1.7B.q8.mcf" \
		"$MCF_DIR/LFM3-1.7B.k4.mcf" \
		"$MCF_DIR/LFM3-1.7B.q4.mcf" \
		"/work/models/mcf/LFM3-1.7B.mcf" >>"$LOG" 2>&1

	"$HERE/bench_llamacpp.sh" "$OUT/cpu-llamacpp.tsv" 1 \
		"$GGUF_DIR/LFM3-1.7B-Q8_0.gguf" \
		"$GGUF_DIR/LFM3-1.7B-Q4_K_M.gguf" >>"$LOG" 2>&1
done
echo "done: $OUT" | tee -a "$LOG"
