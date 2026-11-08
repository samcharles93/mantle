# Mantle status (2026-11-09)

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
- q8 MCF (1.85 GB, all 24 layers resident): decode ~49 tok/s; prefill ~167 tok/s on a 610-token prompt in auto mode (was ~93 after the quant GEMM, ~41 per-token before; the GEMM now uses an int8 dp4a inner product with per-32-block quantized activations, 16.3 -> 7.7 ms per 2048x2048x610 call). The decode fused attention pipeline accepts the standalone LFM3 attn gate (`gate_proj`) on device, and the sandwich (post-branch) norms are device-chained via the pending block result (no per-layer flush). Batched prefill now covers quantized weights in auto mode via a new tiled cached-int8 quant GEMM kernel (`quant_batch.cu`); k4 prefill ~70 tok/s. `MANTLE_CUDA_TRACE` gained a WaitForResult D2H attribution counter. Decode is GPU-execution-bound (~19 ms/token on the RTX 3050); remaining gap to llama.cpp (75 tok/s decode, 4657 tok/s pp512) is kernel count/streaming, not syncs.
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

## LFM3-VL-2B multimodal (2026-11-09)

`LiquidAI/LFM3-VL-2B` packs and runs end to end (text + image). New `lfm3_vl` arch (LFM3 text decoder
under `model.language_model.*`), a host-side vision tower (`VisionTower`: patchifier, embedding norm,
14 hybrid gated-conv/bidirectional-attention layers with axial 2D RoPE and per-frame segments,
multimodal projector) loaded from `model.vision_tower.*` / `model.multi_modal_projector.*`, and a
new optional MCF section 0x0106 carrying `processor_config.json` (image preprocessing params).
`mantle pack`/`quantize` work unchanged (all 424 tensors pack generically; q8 VL verified). Image
input: `mantle run --image file.png ...` (repeatable) and OpenAI-style `input_image` content blocks
(data URL / file path / http URL) in `serve`. The engine expands each `<|image|>` placeholder into
one token per projected vision token and splices the embeddings via `PrefillTokensEmbeds`
(SIMD batch path + CUDA batched plan both support the overrides; per-token `ForwardTokenEmbed`
fallback). Image preprocessing replicates HF bit-for-bit: torchvision uint8 antialias bicubic
(fixed-point int16 weights), smart_resize, normalize, merge-block-major patchify.
Parity vs HF transformers (testdata/lfm3vl_reference.json, `MANTLE_LFM3VL_MCF`): pixel payload exact,
vision features ~2e-6, prompt logits ~2e-5, 8-token greedy + per-step logits match on text, 256x256 and
resized 100x70 image prompts (CPU simd + engine; CUDA output identical). Prompt rendering matches
HF token-for-token including the `<|think|>` generation-prompt tag. CUDA: 89-token image prompt
prefill ~340-410 tok/s (batched path), decode ~10 tok/s on the RTX 3050 (bf16, 18/24 layers in VRAM).
Vision tower runs on the host even under CUDA (once per image; a device path is future work).

## UQ producer CLI (2026-11-09)

`mantle quantize --method uq2..uq8 [--codebooks file.json]` produces `DTypeUQ*` MCFs end to end:
per-256-element super-block codebooks (from the SLQ JSON when provided, else deterministic k-means),
per-32-block f16 scales, packed indices via binary-search nearest level (`mcf.EncodeUQTensor`).
Verified on gemma-3-270m and functiongemma-270m: uq6 layers + q8 embedding (via `--assignments`)
answers correctly; pure uq4 on the embedding degrades (heavy tails + sqrt(hidden) multiplier), which
is the expected per-tensor assignment use case. Unit tests cover codec round-trip, derived and
explicit codebooks, determinism, and bits->dtype mapping.

## LFM3-VL video input (2026-11-09)

Videos are accepted as frame sequences — `mantle run --video <dir-of-frames|single-image>` (repeatable);
mp4 decoding would need a new dependency, so it's out of scope. Each frame is resized with the video
processor's bounds, odd frame counts pad by repeating the last frame, and patches pair frames
(2s, 2s+1) in their temporal slots. The engine expands each `<|video|>` placeholder into per-segment
`<t seconds>` + `<|vision_start|>` + video tokens + `<|vision_end|>` (HF `_calculate_timestamps`
semantics) and splices the vision tower's per-segment embeddings. Parity vs HF on a 4-frame video
(grid [2,16,16], 128 video tokens): pixel payload exact, logits/greedy match (fixture case
video_4frames in TestLFM3VLReferenceLogits / TestLFM3VLEngineImageGeneration).

## DSpark speculative decoding (2026-11-09)

End to end on CPU: `mantle run --draft-model <draft.mcf> [--draft-tokens N] [--draft-confidence F]` (also `MANTLE_DRAFT_MODEL`). Setting `--draft-model` forces the cpu backend, loads the target with hidden taps on the draft's `target_layer_ids`, and pairs the draft (`LoadDSparkDraftMCF`). Each round (`DSparkDraft.Speculate`, internal/backend/simd/dspark_spec.go) drafts `block_size` tokens after the anchor, verifies the whole block against the target in one batched `ForwardTokens`, accepts the longest prefix the target agrees with, rolls the target KV back to the accepted prefix, injects the accepted prefix's taps into the draft cache, and returns the first disagreeing row as the next round's distribution. Greedy (T=0) is exact rejection-free verification; at T>0 the target sample decides, optionally overridden by the confidence head when `--draft-confidence` is set (0 = exact rejection sampling). The repetition-penalty window grows with each accepted draft row, matching the per-token loop. The engine drives it from `Generator.generateSpeculative` (internal/inference/speculative.go); the per-token loop was refactored into `generatePlain` sharing a `streamBuffer`.

Verified: hermetic synthetic tests (spec output == plain greedy, KV rollback, lockstep, guards) plus an env-gated real-pair test (`TestDSparkSpeculateRealPairGreedy`, `MANTLE_DSPARK_TARGET`/`MANTLE_DSPARK_DRAFT`): spec == plain greedy for 48 tokens on two prompts, with and without repeat penalty 1.1. CLI: identical 48-token output with and without `--draft-model`. Acceptance on the real pair is 5.1/7 tokens per block (`TestDSparkDraftAcceptance`). Two real bugs were caught by these tests: the sampler's in-place penalty was applied twice to the returned next-logits row (Speculate now hands disagree a copy and returns a raw copy), and the penalty window initially missed the current round's accepted prefix. Throughput on this 8-core CPU (bf16 MiniCPM5-2B): verify batch ~68 ms/token vs 165 ms/token single, DraftBlock(7) ~100 ms/round, so a round costs ~540 ms — net ~1.3x faster than plain on high-acceptance prompts (france, 4.4 accepted/round), break-even to slower on low-acceptance ones (sky, 2.2/round; dt=7 2.8 tok/s vs plain 5.7). The CUDA draft path (verify batch on device) is where the real win is; still open.

## Open work

1. DSpark on CUDA: the draft machinery is simd-only; the target verify batch and draft block need device paths (then `--draft-model` can stop forcing cpu).
2. Performance vs llama.cpp on the RTX 3050 (Q8_0, ngl 99): llama-bench pp512 4657 / tg128 75.3 tok/s; Mantle LFM3-1.7B.q8 CUDA is at prefill ~167 / decode ~49 tok/s (610-token prompt, auto mode). Remaining gaps: (a) prefill ~28x behind llama.cpp — the batched quant GEMM uses int8 dp4a (~7.7 ms per 2048x2048x610 call); remaining gap is per-layer launch/attention overhead, not the GEMM; (b) decode ~1.5x behind — GPU-execution-bound (kernel count + q8 weight streaming), not sync-bound; closing it needs kernel fusion (fewer, larger kernels per layer).
3. `mantle trace` is CPU-only, so CPU and CUDA logits can't be compared directly.
4. Dependency bumps for existing modules (urfave/cli, echo, golang.org/x/*).
5. Items in `TODO.md`: Gemma3 marked broken, Granite, CI, docs.
6. LFM3-VL/Omni follow-ups: mp4/video-file decoding (currently frame directories only), LFM3-Omni-2B
   (audio encoder + Liquid RVQ codec), a device-side vision tower for large images.

CUDA build: `CUDA=1 task build -f`, then run with `LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib`.
