# Mantle status (2026-11-07)

Go 1.27.2 with `GOEXPERIMENT=simd`. `go test ./...`, the CUDA-tagged backend tests and `golangci-lint run` all pass.

## MiniCPM5-2B

Checked against `/work/models/mcf/MiniCPM5-2B.mcf` on CPU and CUDA (RTX 3050, 23/42 layers in VRAM).

- Prompt tokenization matches the reference `tokenizer.json`, including the leading `<s>` and ChatML special tokens.
- Stop tokens resolve to `[1 130073]` and generation stops on `<|im_end|>`.
- Output is coherent with and without thinking. `--reasoning-budget 0` renders the template's `enable_thinking=false` block.
- Greedy CPU and CUDA output is identical on short answers. Longer runs diverge only at near-tied logits (0.02 apart on CPU), which is BF16 rounding noise rather than a backend bug. With the default `repeat_penalty=1.1`, ties are created more often.
- Throughput: about 5 tok/s generation on both backends. Prefill is about 10 tok/s on CPU and 65 to 80 tok/s on CUDA. CUDA decode is limited by the 19 layers that don't fit in VRAM.

Batched CPU prefill now covers per-head q/k norm, so Qwen3-family models use it (Qwen3-0.6B prompt processing 21 to 32 tok/s).

Batched CPU prefill also covers hybrid short-conv + attention layers (LFM2/LFM3 shape): recurrent conv blocks with state carry across chunk boundaries, sliding windows, per-head and per-element attention output gates, sandwich norms, and per-layer zero-tail RoPE. LFM3-1.7B prompt processing went from 7.1 to 17.0 tok/s on a 469-token prompt (default chunk 64; `MANTLE_CPU_MAX_BATCH` tunes the chunk size, about 19 tok/s at 256). Greedy output is identical to the per-token path, and `TestLFM3ReferenceLogits` passes with `PrefillTokens` on the batched path.

Quant-cache matvec was reworked (fused per-block int32 dots + scalar scale combine; see NOTES.md): K4 2048x1024 matvec ~600 to ~200 us, Qwen3-0.6B.k4 CPU decode 4.5 to 13-17 tok/s, bit-identical greedy output.

## LFM3-1.7B (RTX 3050 4GB, CUDA)

- bf16 MCF: 18/24 layers in VRAM, 6 PCIe-managed, decode 9.5 tok/s. Batched CUDA prefill now covers hybrid LFM3 layers (short-conv + sliding window + attn gate + sandwich norms): 469-token prompt prefill 9.7 -> 502 tok/s. Quantized MCFs (q8/k4) still take the per-token quant-kernel path in auto mode (prefill ~30-40 tok/s); dequant mode unlocks the batched path when the dequantized weights fit VRAM.
- q8 MCF (1.85 GB, all 24 layers resident): decode ~44-46 tok/s, prefill ~41-45 tok/s on a short prompt (10 prompt + 87 gen tokens). The decode fused attention pipeline now accepts the standalone LFM3 attn gate (`gate_proj`) on device (gate projection + sigmoid-scale before Wo, device-resident), so gated layers no longer fall back to the per-op host path; `MANTLE_CUDA_TRACE` gained a WaitForResult D2H attribution counter. Decode is GPU-execution-bound (~19 ms/token on the RTX 3050, ~400 small kernels/token plus q8 weight streaming); per-op blocking D2H flushes mostly observe queued GPU work rather than add to it.
- llama.cpp reference on the same GPU (Q8_0 GGUF, ngl 99): pp512 4657 tok/s, tg128 75.3 tok/s. Mantle CUDA prefill is still per-token (the batch gate rejects device ops), hence the large prefill gap; decode gap is 2x.
- CPU: bf16 prefill 31 tok/s (batched), decode 8.5; k4/q8 decode 11.7/11.6 after the fused quant-dot fix (were 4.1/6.9). Quant weights still take the per-token prefill path on CPU (prefill 11-13 tok/s), since the batch GEMM is f32/bf16-only.

## Granite

granite-4.0-350M (granitemoehybrid) runs and matches HF transformers logits to 1e-5 on the last-position top-10 plus greedy continuation (TestGraniteReferenceLogits, MANTLE_GRANITE_MCF). The arch adds a fused gate+up MLP split (shared_mlp.input_linear, per-layer) and the attention_multiplier / logits_scaling / residual_multiplier / embedding_multiplier scalars.

## Quantization design (GSQ/RCO/SLQ)

pkg/mcf/quantization-design.md maps the IST-DASLab methods onto MCF: GSQ outputs are uniform-grid
K-quant-compatible (consumable today, no format change); RCO/SLQ per-layer bitwidth assignments fit
MCF's per-tensor DType (needs `quantize --assignments` + budget verification + QuantInfo provenance);
SLQ non-uniform codebooks need a new `DTypeUQ*` family (0x0040 range) under a v1.1 minor bump, with
CPU LUT kernels (pshufb/vpermb/gather) and CUDA dequant-first. Proposal only — no spec change yet.

## Open work

1. DSpark (CPU first): the draft model loads (`LoadDSparkDraftMCF`), K/V is injected from the target's hidden taps, and a draft block runs with the Markov and confidence heads. The draft decoder is a regular `Instance` (standard Qwen3 layers, loaded with the target's embeddings and LM head through `LoadModelOptions.SharedEmbeddings`/`SharedOutput`), and blocks use the batch path's non-causal mode (`ForwardBlockNonCausal`). On the real pair, drafts match the target's greedy output for 5.1 of 7 tokens per block (`TestDSparkDraftAcceptance`, with `MANTLE_DSPARK_TARGET` and `MANTLE_DSPARK_DRAFT` set). Still needed: the speculative loop (batched verification, accept/reject, KV rollback, sampling at T>0), CLI wiring, then CUDA. The draft MCF was repacked with its `config.json`. Model shape comes from it; speculation on/off, draft length and confidence threshold will be CLI flags.
2. Performance vs llama.cpp on the RTX 3050 (Q8_0, ngl 99): llama-bench pp512 4657 / tg128 75.3 tok/s; Mantle LFM3-1.7B.q8 CUDA is at prefill ~42 / decode ~46 tok/s. Remaining gaps: (a) CUDA batched prefill only covers dense f32/bf16 layers (quant weights fall back to per-token in auto mode; Qwen3-0.6B dense bf16 reaches 448 tok/s prefill, k4 441 in dequant mode); (b) decode ~1.6x behind llama.cpp — now GPU-execution-bound (kernel count + q8 weight streaming), not sync-bound; closing it needs kernel fusion (fewer, larger kernels per layer), and the sandwich-norm device-chaining experiment (decode 44 -> 49 tok/s) still needs a stale-norm-view fix before it can land.
3. `mantle trace` is CPU-only, so CPU and CUDA logits can't be compared directly.
4. Dependency bumps for existing modules (urfave/cli, echo, golang.org/x/*).
5. Items in `TODO.md`: Gemma3 marked broken, Granite, CI, docs.

CUDA build: `CUDA=1 task build -f`, then run with `LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib`.
