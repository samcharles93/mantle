#!/usr/bin/env bash
set -euo pipefail

# Extract deterministic generated model output from a Mantle CUDA trace log.
# Usage: scripts/extract_model_output.sh <trace-file> > output.txt
# Exits with status 2 if no generated output found.

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <trace-log>" >&2
  exit 1
fi

infile="$1"

if [ ! -f "$infile" ]; then
  echo "file not found: $infile" >&2
  exit 1
fi

# Helper: strip ANSI escape sequences
strip_ansi() {
  sed -r "s/\x1B\[[0-9;]*[mKHF]//g"
}

# Known volatile prefixes/fields to remove:
# - leading timestamp + bracketed level: [2026-..] INFO ...
# - log prefixes like 'loading MCF model', 'generation complete', 'Stats: ...'
# We'll capture the block between the first blank line after 'model loaded' and
# the '=== CUDA Performance Summary ===' marker.

awk -v RS="" '/model loaded/ { print; exit }' "$infile" >/dev/null 2>&1 || true


# Locate model loaded line
model_line=$(grep -n "model loaded" "$infile" 2>/dev/null | head -n1 | cut -d: -f1 || true)

# Find start: first non-empty line after the first blank line that follows 'model loaded'
if [ -n "$model_line" ]; then
  start_line=$(awk -v m="$model_line" 'NR>m{ if ($0 ~ /^[[:space:]]*$/) { blank=1; next } if (blank && $0 !~ /^[[:space:]]*$/) { print NR; exit } }' "$infile" || true)
else
  # Fallback: first non-empty line in file
  start_line=$(awk '$0 ~ /./ { print NR; exit }' "$infile" || true)
fi

if [ -z "$start_line" ]; then
  echo "could not locate generated output start" >&2
  exit 2
fi

# Find end: line number of the '=== CUDA Performance Summary ===' marker
end_line=$(grep -n "=== CUDA Performance Summary ===" "$infile" 2>/dev/null | head -n1 | cut -d: -f1 || true)

if [ -z "$end_line" ]; then
  echo "could not locate CUDA Performance Summary marker" >&2
  exit 2
fi

# If start is at or after end, there's no generated content
if [ "$start_line" -ge "$end_line" ]; then
  echo "error: no generated model output found in $infile (start >= end)" >&2
  exit 2
fi

# Extract region (start_line .. end_line-1)
sed -n "${start_line},$((end_line-1))p" "$infile" \
  | sed '1{/^[[:space:]]*$/d}' \
  | # remove ANSI escapes
  perl -pe 's/\e\[[\d;]*[A-Za-z]//g' \
  | # remove bracketed timestamps and level prefixes like [2026-..] INFO
  sed -E 's/^\s*\[[0-9:-TZ. ]+\]\s*//; s/^\s*(INFO|DEBUG|WARN|ERROR)\s*//; s/^[[:space:]]+//g' \
  | # remove lines that are clearly log metadata
  grep -v -E '^(CUDA Device|CUDA graph|CUDA mem preflight|CUDA preload|Stats:|Input tokens\(|Graph launches:|Graph captures:|Tokens processed:|-- |^Device allocs:)' \
  | # collapse multiple blank lines
  awk 'BEGIN{blank=0} /^$/ { if(!blank){ print ""; blank=1 } next } { print; blank=0 }'

# Check whether any non-empty output produced
# We re-run extraction into a temp file to check contents
tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT
sed -n "${start_line},$((end_line-1))p" "$infile" \
  | perl -pe 's/\e\[[\d;]*[A-Za-z]//g' \
  | sed -E 's/^\s*\[[0-9:-TZ. ]+\]\s*//; s/^\s*(INFO|DEBUG|WARN|ERROR)\s*//; s/^[[:space:]]+//g' \
  | grep -v -E '^(CUDA Device|CUDA graph|CUDA mem preflight|CUDA preload|Stats:|Input tokens\(|Graph launches:|Graph captures:|Tokens processed:|-- |^Device allocs:)' \
  | awk 'BEGIN{blank=0} /^$/ { if(!blank){ print ""; blank=1 } next } { print; blank=0 }' > "$tmpfile" || true

if [ ! -s "$tmpfile" ]; then
  echo "error: no generated model output found in $infile" >&2
  exit 2
fi

cat "$tmpfile"
