# Mantle status (2026-11-07)

Go 1.27.2 with `GOEXPERIMENT=simd`. `go test ./...`, the CUDA-tagged backend tests and `golangci-lint run` all pass.

## MiniCPM5-2B

Checked against `/work/models/mcf/MiniCPM5-2B.mcf` on CPU and CUDA (RTX 3050, 23/42 layers in VRAM).

- Prompt tokenization matches the reference `tokenizer.json`, including the leading `<s>` and ChatML special tokens.
- Stop tokens resolve to `[1 130073]` and generation stops on `<|im_end|>`.
- Output is coherent with and without thinking. `--reasoning-budget 0` renders the template's `enable_thinking=false` block.
- Greedy CPU and CUDA output is identical on short answers. Longer runs diverge only at near-tied logits (0.02 apart on CPU), which is BF16 rounding noise rather than a backend bug. With the default `repeat_penalty=1.1`, ties are created more often.
- Throughput: about 5 tok/s generation on both backends. Prefill is about 10 tok/s on CPU and 65 to 80 tok/s on CUDA. CUDA decode is limited by the 19 layers that don't fit in VRAM.

## Open work

1. DSpark: target hidden-state tap capture is done (`--capture-tap-layers`). Still needed: projector, draft block prediction, batched target verification, accept/reject sampling and rollbackable KV state.
2. Performance against the reference of about 200 tok/s on a 3060 (Q8_0 in llama.cpp).
3. `mantle trace` is CPU-only, so CPU and CUDA logits can't be compared directly.
4. Dependency bumps for existing modules (urfave/cli, echo, golang.org/x/*).
5. Items in `TODO.md`: Gemma3 marked broken, Granite, CI, docs.

CUDA build: `CUDA=1 task build -f`, then run with `LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib`.
