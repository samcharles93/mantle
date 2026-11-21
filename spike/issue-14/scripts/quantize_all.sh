#!/usr/bin/env bash
# Quantises the LFM3-1.7B bf16 container to every method under test, timing each run.
set -u
BIN="${BIN:-/work/wt/mantle-quant14/bin/mantle}"
SRC="${SRC:-/work/models/mcf/LFM3-1.7B.mcf}"
OUT="${OUT:-/work/models/mcf/probe/lfm3}"
LOGS="$(dirname "$0")/../logs"
mkdir -p "$OUT" "$LOGS"

for m in "$@"; do
	dst="$OUT/LFM3-1.7B.$m.mcf"
	if [ -f "$dst" ]; then
		echo "skip $m (exists)"
		continue
	fi
	echo "=== quantize $m ==="
	/usr/bin/time -v "$BIN" quantize --input "$SRC" --output "$dst" --method "$m" \
		> "$LOGS/quantize.$m.log" 2>&1
	rc=$?
	wall=$(grep -oP 'Elapsed \(wall clock\).*' "$LOGS/quantize.$m.log" | head -1)
	cpu=$(grep -oP 'User time.*' "$LOGS/quantize.$m.log" | head -1)
	rss=$(grep -oP 'Maximum resident set size.*' "$LOGS/quantize.$m.log" | head -1)
	echo "rc=$rc $wall $cpu $rss size=$(stat -c %s "$dst" 2>/dev/null)"
done
