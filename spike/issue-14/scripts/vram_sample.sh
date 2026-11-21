#!/usr/bin/env bash
# Samples used VRAM while a command runs, then prints the peak. Args: <cmd...>
set -u
out="$(mktemp)"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.1; done >"$out" ) &
sampler=$!
"$@" >/dev/null 2>&1
rc=$?
kill "$sampler" 2>/dev/null
wait "$sampler" 2>/dev/null
peak=$(sort -n "$out" | tail -1)
rm -f "$out"
echo "$peak"
exit $rc
