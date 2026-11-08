# MCF quantization design: GSQ / RCO / SLQ outputs

Status: **proposal** (no on-disk change yet). Nothing here alters `model-container-file.md`;
the spec only changes when a format addition is implemented, in the same commit, versioned per
the Compatibility rule (readers reject unsupported major versions, may ignore unknown minors).

Scope note: Mantle is an execution substrate, not a quantizer. The *search/training* pipelines
(GSQ, RCO, SLQ) stay external; MCF/Mantle only needs to **store and execute their outputs**
efficiently. This document maps each method's output structure onto MCF and proposes the one
new dtype family that requires a format addition.

## 1. What the three methods output (format-relevant facts)

| Method | Output structure | Bitwidth | Grouping | Codebook | Budget |
|---|---|---|---|---|---|
| **GSQ** (Gumbel-Softmax Quant, ISTA-DASLab) | Per-layer learned **integer grid assignments + per-group scales**, serialized as a compressed-tensors/GGUF-K-Quant-compatible checkpoint | 2 bpp default; n-bit and ternary variants | per-group (K-quant-style) | **uniform symmetric integer grid** `{-2^s..2^s-1} × scale` (e.g. `{-2,-1,0,1}×s` at 2-bit; ternary `{-1,0,1}×s`) | none (fixed bits) |
| **RCO** (Riemannian Constrained Optimization) | A **discrete assignment**: one of K bitwidth options per group/layer, chosen to minimize a non-decomposable loss under an **exact total-size budget** (Riemannian projection + Gumbel-STE + exact knapsack DP) | per-group choice among {2..8} | per layer/group | inherits the per-option codebook (uniform INT for the GPTQ database it searches) | exact equality budget B (e.g. total bytes) |
| **SLQ** (Statistically-Lossless Quant) | Per-layer **non-uniform** quant meeting a fidelity target (TL ≥99% benchmark recovery at 3.3–4.7 bpp; DL EAR≥0.99 at 5.0–6.6 bpp): layer-wise GPTQ at each bitwidth → Shapley sensitivity → ILP (or RCO) assignment | **per-layer mixed** 2–8 bpp | group size **128**, **asymmetric** | **non-uniform, learned** codebook (not a uniform grid) | global, via the assignment |

Sources: github.com/IST-DASLab/{GSQ,RCO,SLQ} (READMEs + scripts), arXiv 2604.18556 (GSQ), 2605.00649 (RCO), 2605.02404 (SLQ).

## 2. Mapping onto MCF

MCF already stores a per-tensor `TensorDType` in the fixed 40-byte `TensorIndexEntry`, with a
quant registry (0x10–0x33: int*/q*/k*), a `QuantInfo` section (method metadata), and the
`FlagTensorDataAligned64` 64-byte sub-region alignment rule. That maps onto the three methods as:

### 2.1 GSQ → no format change (P0) — IMPLEMENTED

GSQ payloads are indices into a uniform symmetric grid plus per-group scales — exactly the
shape of MCF's existing `DTypeK2..K6` / `DTypeQ4/Q8` payloads (block 32, super-block 256, f16
scales). GSQ even refines existing GGUF K-quants *in-format*, so:

- **Consume**: GGUF tensor payloads are imported tensor-by-tensor through `pkg/mcf/gguf.go`:
  `mcf.ParseGGUF(r, size)` parses the container (header, metadata, tensor index; offsets are
  relative to the aligned data section), `GGUFFile.TensorPayload` reads a raw payload, and
  `mcf.TranscodeGGUFQuant` converts it to an MCF quant payload. **Q8_0 → `DTypeQ8` is
  value-exact** (GGUF block 32 = f16 scale + 32 int8; MCF stores the same values with scales
  and data in separate regions). Round-trip tests: `pkg/mcf/gguf_test.go` (parse/transcode/
  corruption) and `internal/backend/core/gguf_roundtrip_test.go` (GGUF bytes → MCF tensor →
  `RowTo` equals GGUF dequantization; `MANTLE_GGUF_Q8_0=<file>` runs it against a real file,
  verified on embeddinggemma-300M-Q8_0.gguf).
  **Limitation**: GGUF K-quants (Q2_K..Q6_K) carry a per-superblock min term and Q4_K/Q5_K/Q2_K/Q3_K
  use asymmetric sub-blocks, and Q6_K/Q8_K use int8/f32 scales — MCF's symmetric k*/q* family
  cannot represent those byte-exactly, so K-quant tensors must be dequantized and re-encoded
  (lossy) rather than transcoded. GSQ's own uniform-grid symmetric payloads transcode exactly
  whenever their per-group scales fit the f16/6-bit grid.
- **Decode kernels**: unchanged — existing `dotInt8Int16`/k-quant paths handle the grid.
- Full GGUF→MCF *model* conversion (name mapping, config) is intentionally out of scope:
  Mantle is an execution substrate, not a conversion pipeline.

### 2.2 RCO / SLQ assignments → per-tensor dtype + provenance (P1)

Both methods' defining output is the **per-layer/per-group bitwidth assignment** under a budget.
MCF's per-tensor `DType` already expresses mixed bitwidths across layers (each layer's tensors
carry their own dtype), so no structural change is needed; what is missing is tooling and
provenance:

- `mantle quantize --assignments <file.json>`: an assignment file mapping tensor name/layer →
  bits (the RCO/SLQ search output), plus `--budget-bytes N` verification (the file's exact
  budget guarantee) with a hard error on violation.
- Record the assignment in `QuantInfo` (method = `rco`/`slq`, params = budget, targets) so a
  container is self-describing about how it was produced.

### 2.3 SLQ non-uniform codebooks → new dtype family `UQ` (P2, the only format addition)

SLQ's learned codebooks are not uniform grids, so q*/k* payloads cannot represent them. Proposal:

- Registry: reserve **`0x0040–0x004F`** for the `uq*` family: `DTypeUQ2=0x0040 … DTypeUQ8=0x0047`.
  Payload mirrors the k-quant super-block layout (super-block 256 = 2 groups of 128, block 32),
  but each **super-block carries an explicit codebook** of `2^b` f16 values, followed by per-block
  f16 scales and packed b-bit indices:

  ```
  [Super-block: codebook 2^b × f16 | block0 scale f16 | block0 indices (32×b bits) | block1 … ]
  ```

  All little-endian; every sub-region 64-byte aligned per the existing rule; files containing
  `DTypeUQ*` MUST set `FlagTensorDataAligned64` (same rule as Q*/K*). Codebook per super-block
  keeps locality (a 4-bit UQ super-block codebook is 32 bytes, amortized over 256 weights).
  Asymmetric (SLQ) is expressed by the codebook itself containing the learned levels; no sign bit.

- `QuantInfo` gains method `uq` with params `{bits, group=128, super=256, codebook=f16}`.

- **Versioning**: new dtypes are only meaningful with a reader that knows them. Per the spec's
  Compatibility rule, introduce them under a **minor version bump (v1.1)**; v1.0 readers reject
  the unknown `TensorDType` values when they must materialize those tensors (they already reject
  unknown dtypes), and may ignore the file otherwise. No v1.0 file changes meaning.

## 3. Decode kernel plan

- **CPU (archsimd / Go)**: b ≤ 4 → LUT via `pshufb` on packed nibbles (two 16-entry f16→f32
  expands per 16 weights), matching the existing `dotInt8Int16SIMD` structure; b in 5..8 →
  8-bit indices with `vpermb` on AVX512-VBMI, or `_mm256_i32gather_ps` fallback on AVX2.
  AVX-VNNI has no LUT, so the VNNI path keeps indices in int8 and gathers — benchmark before
  committing to it. Activations stay f32 or int16-quantized as today; the per-block scale folds
  into the existing scale-combine step (see NOTES.md "Performance notes": keep int32 dots in
  lanes, one horizontal reduction per row, scales applied in f32 afterwards).
- **CUDA**: two options — (a) dequantize UQ→F16 at load (the existing `--cuda-weight-mode
  dequant` path pattern; correct immediately, costs VRAM), (b) custom mmq-style kernel with
  codebook in shared memory (LUT in smem, warp-level reduce). Phase (a) first.
- **Benchmarks/tests**: round-trip encode/decode equality, parity vs f32 reference within the
  method's tolerance, and a `BenchmarkMatVecUQ*` alongside `BenchmarkMatVecK4Cached`.

## 4. Phasing

1. **P0** GSQ consumption docs + GGUF Q8_0 → MCF transcode and round-trip tests. No format change. DONE.
2. **P1** `--assignments` + budget verification + QuantInfo provenance (rco/slq). No format change.
3. **P2** `DTypeUQ*` registry reservation, v1.1 spec addition, payload codec + CPU LUT kernels +
   tests/benchmarks, CUDA dequant-mode support. Format change, versioned, spec updated in the
   same commit.
