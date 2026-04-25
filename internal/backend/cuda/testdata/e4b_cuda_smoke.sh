#!/usr/bin/env bash
set -euo pipefail

# e4b_cuda_smoke.sh
# End-to-end regression test for E4B CUDA coherence with long system prompts.

MANTLE_BIN="bin/mantle"
MODEL_PATH="/work/models/mcf/gemma-4-E4B-it.k4.mcf"

# Check if mantle binary exists
if [ ! -f "$MANTLE_BIN" ]; then
    echo "Error: $MANTLE_BIN not found. Please build it first."
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model $MODEL_PATH not found. Skipping test."
    exit 0
fi

SYSTEM_PROMPT="You are Kai, an AI-first ops platform agent engineered to deliver direct, pragmatic, and highly optimized solutions for infrastructure management and Go development."
USER_PROMPT="Hello. I'm Sam, who are you and how are you doing today?"

echo "Running Mantle with E4B model..."
# Run mantle and capture all output
OUTPUT=$(GOEXPERIMENT=simd "$MANTLE_BIN" run -m "$MODEL_PATH" --system "$SYSTEM_PROMPT" --prompt "$USER_PROMPT" --seed 42 -t 0.6 --steps 50 2>&1)

echo "--- RAW OUTPUT ---"
echo "$OUTPUT"
echo "------------------"

# Use python to validate the output
python3 -c '
import sys
import re
from collections import Counter

output = sys.stdin.read()

# Extract generated text
# The generated text is between "model loaded" and "generation complete"
lines = output.split("\n")
text_lines = []
in_text = False
for line in lines:
    # Remove ANSI escape codes
    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    
    if "model loaded" in clean_line:
        in_text = True
        continue
    if "generation complete" in clean_line or "Stats:" in clean_line:
        in_text = False
        continue
    
    # Skip log lines
    if in_text and not re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] (INFO|WARN|ERROR|DEBUG)", clean_line):
        text_lines.append(clean_line)

text = " ".join(text_lines).strip()

print(f"--- EXTRACTED TEXT ---\n{text}\n----------------------")

words = text.split()

# 1. > 20 tokens generated (we approximate with words)
if len(words) <= 20:
    print(f"FAIL: Generated text is too short ({len(words)} words). Expected > 20.")
    sys.exit(1)

# 2. Output does NOT contain repeated system prompt text ("I am Kai") > 3 times
kai_count = text.lower().count("i am kai")
if kai_count > 3:
    print(f"FAIL: Output contains repeated phrase \"I am Kai\" ({kai_count} times).")
    sys.exit(1)

# 3. Detect repetition: if the same phrase appears > 3 times, FAIL
# We check for repeated 5-grams
ngrams = [" ".join(words[i:i+5]).lower() for i in range(len(words)-4)]
if ngrams:
    counts = Counter(ngrams)
    most_common, count = counts.most_common(1)[0]
    if count > 3:
        print(f"FAIL: Output contains repeated phrase \"{most_common}\" ({count} times).")
        sys.exit(1)

# 4. Detect incoherence: if output is a single repeated phrase, FAIL
unique_words = set(w.lower() for w in words)
if len(unique_words) < len(words) * 0.3 and len(words) > 30:
    print(f"FAIL: Output lacks variety (only {len(unique_words)} unique words out of {len(words)}).")
    sys.exit(1)

print("PASS: Output is coherent and non-repetitive.")
sys.exit(0)
' <<< "$OUTPUT"
