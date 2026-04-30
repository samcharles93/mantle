#!/usr/bin/env bash
set -euo pipefail
usage() { echo "Usage: $0 <baseline.txt> <final.txt> [output.txt]"; exit 1; }
[ $# -lt 2 ] || [ $# -gt 3 ] && usage
BASE="$1"; FINAL="$2"; OUT="${3:-/dev/stdout}"
for f in "$BASE" "$FINAL"; do
  [ -f "$f" ] || { echo "Error: not found: $f"; exit 1; }
  grep -q "CUDA Performance Summary" "$f" || { echo "Error: not a CUDA trace: $f"; exit 1; }
done

extract() { grep -oEm1 "$2" "$1" | grep -oEm1 '[0-9.]+$' | head -n1 || echo "NA"; }

b_tps=$(extract "$BASE" 'gen_tps=[0-9.]+')
a_tps=$(extract "$FINAL" 'gen_tps=[0-9.]+')
b_prompt=$(extract "$BASE" 'prompt_tps=[0-9.]+')
a_prompt=$(extract "$FINAL" 'prompt_tps=[0-9.]+')
b_tokens=$(extract "$BASE" 'Tokens processed: [0-9]+')
a_tokens=$(extract "$FINAL" 'Tokens processed: [0-9]+')
b_cap=$(extract "$BASE" 'Graph captures: [0-9]+')
a_cap=$(extract "$FINAL" 'Graph captures: [0-9]+')
b_launch=$(extract "$BASE" 'Graph launches: [0-9]+')
a_launch=$(extract "$FINAL" 'Graph launches: [0-9]+')
b_fail=$(extract "$BASE" 'Graph failures: [0-9]+')
a_fail=$(extract "$FINAL" 'Graph failures: [0-9]+')
b_sync=$(extract "$BASE" 'Stream syncs: [0-9]+')
a_sync=$(extract "$FINAL" 'Stream syncs: [0-9]+')
b_h2d=$(extract "$BASE" 'H2D bytes: [0-9]+')
a_h2d=$(extract "$FINAL" 'H2D bytes: [0-9]+')
b_d2h=$(extract "$BASE" 'D2H bytes: [0-9]+')
a_d2h=$(extract "$FINAL" 'D2H bytes: [0-9]+')
b_dev_allocs=$(extract "$BASE" 'Device allocs: [0-9]+')
a_dev_allocs=$(extract "$FINAL" 'Device allocs: [0-9]+')
b_dev_mb=$(grep -oEm1 'Device allocs: [0-9]+ \([0-9.]+' "$BASE" | grep -oEm1 '[0-9.]+$' || echo "NA")
a_dev_mb=$(grep -oEm1 'Device allocs: [0-9]+ \([0-9.]+' "$FINAL" | grep -oEm1 '[0-9.]+$' || echo "NA")

delta() { [ "$1" = "NA" ] || [ "$2" = "NA" ] && echo "NA" || awk "BEGIN {printf \"%.4f\", $2-$1}"; }
pct()   { [ "$1" = "NA" ] || [ "$2" = "NA" ] || [ "$1" = "0" ] || [ "$1" = "0.0" ] && echo "NA" || awk "BEGIN {printf \"%.2f\", (($2-$1)/$1)*100}"; }

printf '=== CUDA Trace Comparison ===\nBaseline: %s\nFinal:    %s\n\n' "$BASE" "$FINAL" | tee "$OUT" > /dev/null
printf '%-28s %12s %12s %14s %12s\n' "Metric" "Baseline" "Final" "Delta" "Pct Change" | tee -a "$OUT" > /dev/null
printf '%-28s %12s %12s %14s %12s\n' "--------" "--------" "-----" "-----" "----------" | tee -a "$OUT" > /dev/null

row() {
  local name="$1" b="$2" a="$3"
  printf '%-28s %12s %12s %14s %12s\n' "$name" "$b" "$a" "$(delta "$b" "$a")" "$(pct "$b" "$a")%" | tee -a "$OUT" > /dev/null
}

row "gen_tps"           "$b_tps"        "$a_tps"
row "prompt_tps"        "$b_prompt"     "$a_prompt"
row "tokens_processed"  "$b_tokens"     "$a_tokens"
row "graph_captures"    "$b_cap"        "$a_cap"
row "graph_launches"    "$b_launch"     "$a_launch"
row "graph_failures"    "$b_fail"       "$a_fail"
row "stream_syncs"      "$b_sync"       "$a_sync"
row "h2d_mb"            "$b_h2d"        "$a_h2d"
row "d2h_mb"            "$b_d2h"        "$a_d2h"
row "device_allocs"     "$b_dev_allocs" "$a_dev_allocs"
row "device_mb"         "$b_dev_mb"     "$a_dev_mb"
