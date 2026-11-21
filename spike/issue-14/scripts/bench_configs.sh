#!/usr/bin/env bash
# Runs an arbitrary set of mantle/llama.cpp configs in rotating order so that
# drift on a shared box lands on every config. One line per config in the config
# file, tab separated:
#
#   mantle <label> <binary> <model.mcf> [extra args]
#   llama  <label> <llama-bench> <model.gguf> [extra args]
#
# Usage: bench_configs.sh <repeats> <config-file>
set -u

REPEATS="${1:?repeats}"
CONFIG="${2:?config file}"

OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/out/$(basename "$CONFIG" .tsv).tsv}"
STEPS="${STEPS:-64}"
THREADS="${THREADS:-16}"
PROMPT="${PROMPT:-Explain in one paragraph why the sky appears blue, then list three colours.}"

mapfile -t CONFIGS < <(grep -v '^#' "$CONFIG" | grep -v '^[[:space:]]*$')
COUNT=${#CONFIGS[@]}
if [ "$COUNT" -eq 0 ]; then
	echo "no configs in $CONFIG" >&2
	exit 1
fi

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
	printf 'config\trep\tpp_tps\ttg_tps\tpeak_rss_kb\n' >"$OUT"
fi

run_one() {
	local kind label bin model extra pp tg rss raw
	kind="$1" label="$2" bin="$3" model="$4" extra="${5:-}"
	raw="$(mktemp)"
	if [ "$kind" = "mantle" ]; then
		# shellcheck disable=SC2086
		/usr/bin/time -v "$bin" benchmark --model "$model" --backend cpu \
			--prompt "$PROMPT" --steps "$STEPS" --warmup 1 --runs 1 $extra >"$raw" 2>&1
	else
		# shellcheck disable=SC2086
		/usr/bin/time -v "$bin" -m "$model" -p "$STEPS" -n "$STEPS" -r 1 \
			-t "$THREADS" -ngl 0 $extra >"$raw" 2>&1
	fi
	if [ $? -ne 0 ]; then
		echo "FAIL $label" >&2
		tail -5 "$raw" >&2
		rm -f "$raw"
		return
	fi
	if [ "$kind" = "mantle" ]; then
		pp=$(awk '/^Avg/ {print $2; exit}' "$raw")
		tg=$(awk '/^Avg/ {print $3; exit}' "$raw")
	else
		pp=$(awk -F'|' '/\| *pp[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"\u00b1"); print a[1]; exit}' "$raw")
		tg=$(awk -F'|' '/\| *tg[0-9]+ *\|/ {v=$(NF-1); gsub(/ /,"",v); split(v,a,"\u00b1"); print a[1]; exit}' "$raw")
	fi
	rss=$(sed -n 's/.*Maximum resident set size (kbytes): \([0-9]*\)/\1/p' "$raw" | head -1)
	printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$REP" "$pp" "$tg" "$rss" >>"$OUT"
	echo "$label rep=$REP pp=$pp tg=$tg rss=${rss}kB" >&2
	rm -f "$raw"
}

for REP in $(seq 1 "$REPEATS"); do
	# Rotate the starting config each repeat so ordering bias is spread evenly.
	offset=$(((REP - 1) % COUNT))
	for i in $(seq 0 $((COUNT - 1))); do
		idx=$(((offset + i) % COUNT))
		# shellcheck disable=SC2086
		read -r kind label bin model extra <<<"${CONFIGS[$idx]}"
		run_one "$kind" "$label" "$bin" "$model" "${extra:-}"
	done
done
