# Scenario H - what llama.cpp does for CPU prefill that mantle does not

Reference: `/work/ai/llama.cpp` at `f7591a6a0` (verified with `git log -1`). The file names in
the brief are stale for this commit: `ggml-cpu-quants.c` is now `ggml/src/ggml-cpu/quants.c`
plus `arch/x86/quants.c`, `ggml-cpu-repack.cpp` is `ggml/src/ggml-cpu/repack.cpp` plus
`arch/x86/repack.cpp`, and `ggml-threading.cpp` now only holds `ggml_critical_section_*` —
the threadpool lives in `ggml-cpu.c`. Citations below use the current names.

Mantle reference: `/work/wt/mantle-parity-lfm3` at `043ab6a`, read only. No file under either
tree was modified.

## Measurement notes (no model runs)

This scenario is code reading only, so no model was loaded and no `flock
/work/tmp/prefill-measure.lock` wrapper was needed. The only command that touched llama.cpp
artefacts was `llama-bench --help` (no model, no compute). `/proc/loadavg` before the session:
`8.36 6.59 5.70`. Every number below is either quoted from the brief's measured table, quoted
from source, or an arithmetic estimate explicitly labelled `estimate`.

Baseline the estimates are relative to (brief's table, this box): mantle k4 44.3 tok/s prefill
(22.6 ms/token), llama.cpp Q4_K_M 178.7 tok/s (5.60 ms/token), ratio 4.03x. For a 650-token
prompt that is 14.7 s against 3.64 s. LFM3-1.7B k4 weights are ~0.9 GB on disk at ~4.25
bits/param (estimate from param count; `mantle inspect` was not run).

---

## Verified non-differences first (these kill three candidate explanations)

The brief asked me to confirm each claim against mantle. Three of the named suspects are **not**
differences, and should be dropped from the issue:

1. **LM head runs for one position in both.** llama.cpp slices the hidden state with
   `ggml_get_rows(cur, inp_out_ids)` before the final norm and `model.output`
   (`src/models/lfm2.cpp:226-229`, `:250-251`, ids built in `src/llama-graph.cpp:2883-2904`).
   Mantle's prefill calls `FusedRMSNormMatVec` on `p.x[(n-1)*hidden:n*hidden]` only
   (`internal/backend/simd/runtime_batch.go:661-668`, `emitLastRow` set at `:487-508`).
   Both run the 128000x2048 head exactly once per sequence. Not a gap.
2. **Neither reuses the quantised activation across q/k/v.** llama.cpp quantises src1 into
   `params->wdata` once per `GGML_OP_MUL_MAT` (`ggml-cpu.c:1337-1381`, `repack.cpp:4693-4704`)
   and then re-quantises the same rows again for the next op. Mantle does exactly the same
   (`attn_batch.go:46-48` calls `gemmRows` three times, each entering `MatMatQuant` ->
   `getQuantRows`, `matmat_quant.go:113`). So "q8_K activations reused across tensors sharing an
   input row" is **not** something llama.cpp does. (Mantle's own *decode* path does reuse them,
   `attn.go:198-205`, so the batched path is inconsistent with mantle's decode path - but that is
   a mantle-internal regression, not a llama.cpp technique.)
3. **Multi-row vec_dot is not why llama.cpp wins on this CPU.** `type_traits_cpu[GGML_TYPE_Q4_K].nrows`
   is `1` on x86 (`ggml-cpu.c:318-325`); `nrows=2` is gated on `__ARM_FEATURE_MATMUL_INT8`
   (`ggml-cpu.c:249-250`, `:279-283`, `:324-327`). The generic `compute_forward_mul_mat` on this
   box is one row per `vec_dot` (`ggml-cpu.c:1252-1254`), reusing each weight row out of L1 across
   a `blck_0 = blck_1 = 16` tile (`ggml-cpu.c:1211-1212`, `:1244-1258`). llama.cpp's multi-row
   throughput comes from the repack GEMM and from AMX (absent on Alder Lake), not from the
   type-traits `nrows`.

Also worth recording: llama.cpp's Q4_K is *unsigned* 4-bit with a per-block min, and it pays for
that with an extra `bsums` correction term (`arch/x86/quants.c:2126-2144`, the `acc_m` fold). Mantle's
k4 is signed two's complement with no min term (`matvec_packed4.go:11-13`, `:147-157`), so mantle's
storage format is the *cheaper* one. The gap is not format-induced.

---

## Ranked findings

### 1. Register tile: 8 outputs per weight-block load in mantle against 32 in llama.cpp's repacked GEMM

Why: mantle's batched 4-bit kernel (`matMatRangeQuant`, `matmat_quant.go:171-274`) walks two weight
rows at a time and four activation rows at a time, producing 8 outputs per pass over a block
(`quantDot2x4`, `matmat_quant.go:341-376`). Per 32-element block it issues: 2 `abs(w)`, 8
`x*MulSign(w)` (VPSIGNB), 8 `DotProductPairsSaturated` (VPMADDUBSW), 8 `DotProductPairs(ones)`
(VPMADDWD) (`matmat_quant.go:299-301`), then a horizontal-add tree plus float scale fold per 4
outputs (`matmat_quant.go:355-357,375-376`) - roughly 40 vector ops for 8 outputs, i.e. ~5 per
output per block. llama.cpp's repacked AVX2 GEMM holds **four activation rows x eight weight columns
= 32 accumulators** plus 32 min-term accumulators (`arch/x86/repack.cpp:2833-2841`) and feeds them
from one load of the interleaved weight block, reusing four LHS shuffle patterns across eight
columns (`arch/x86/repack.cpp:3074-3120`: four `_mm256_maddubs_epi16` folded by three adds into each
accumulator, 8 accumulators x 2 shuffle patterns), which is ~40-50 vector ops for 32 outputs, i.e.
~1.4 per output per block. Mantle's `abs`/sign trick is also pure overhead llama.cpp does not pay,
because Q4_K weights are unsigned 0..15.

Correction to the brief's hypothesis: on **this** box (AVX2 + AVX-VNNI, no AVX-512) llama.cpp's
Q4_K GEMM does **not** use `VPDPBUSD`. `_mm256_dpbusd_avx_epi32` appears only in
`mul_sum_us8_pairs_acc_int32x8` (`arch/x86/repack.cpp:151-160`) and `mul_sum_i8_pairs_acc_int32x8`
(`:165-173`), which are used by the **GEMV** kernels (n <= 3, e.g. `ggml_gemv_q4_K_8x8_q8_K` at
`:1464`), not by `ggml_gemm_q4_K_8x8_q8_K` (`:2042-3480`). The AVX2 GEMM body is
maddubs+madd-with-int16-accumulators. So "llama.cpp has VNNI and mantle does not" is not the
prefill story; the GEMM micro-kernel shape is. (For completeness: Go's `simd/archsimd` genuinely
cannot emit `VPDPBUSD` - `/usr/local/go/src/simd/archsimd/ops_amd64.go:2061-2093` has only
`DotProductPairs`/`DotProductPairsSaturated`, while `archsimd.X86.AVXVNNI()` reports the feature at
`cpu.go:134-141` - which matters for a rewrite but is not a current-vs-current difference.)

Exposure: estimate 2.0-2.5x of the 4.0x, i.e. ~7-9 s of the 650-token prompt's 14.7 s is
attributable to inner-loop instruction count at equal batch size. This is the single largest
structural difference and the one that a kernel rewrite must close.

Evidence: `internal/backend/simd/matmat_quant.go:299-301,341-376`; `internal/backend/simd/matvec_packed4.go:31-64`
(the single-row sibling, same instruction shape);
`/work/ai/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:2833-2841,3074-3120,1464,151-160`;
`/usr/local/go/src/simd/archsimd/ops_amd64.go:2061-2093`.

### 2. Prompt chunk is 64 positions against llama.cpp's ubatch of 512

Why: `common_params.n_batch = 2048`, `n_ubatch = 512` (`common/common.h:460-461`), confirmed at
runtime by `llama-bench --help` (`-b ... (default: 2048)`, `-ub ... (default: 512)`), and the context
splits every `llama_decode` into ubatches of at most `n_ubatch` (`src/llama-context.cpp:2110`,
`memory->init_batch(*balloc, cparams.n_ubatch, embd_all)`). llama-bench submits up to `n_batch` tokens
per decode (`tools/llama-bench/llama-bench.cpp:2179-2184`). Mantle's chunk is `defaultMaxBatch = 64`
(`runtime_batch.go:35`), chunked at `:487-508`. For a 650-token prompt: 11 mantle chunks against 2
llama.cpp ubatches.

Why it matters less than the 8x ratio suggests: because both kernels already keep a weight block
resident while several activation rows consume it (4 rows in mantle, 4 or 16 in llama.cpp), the
*per-token* weight traffic only scales with chunk size, not the inner loop. At 64 positions mantle
streams 0.9 GB / 64 = 14.1 MB of weights per token = ~624 MB/s at 44.3 tok/s; llama.cpp needs
0.9 GB / 512 = 1.8 MB per token. Both are two orders of magnitude below this box's DRAM bandwidth,
so prefill is *not* DRAM-bound at either setting and going to 512 does not buy a linear win.

What it does buy: 5.5x fewer per-chunk boundaries (per-op barrier, dispatch, RMSNorm/RoPE/softmax
row loops stay the same), fewer scratch slabs, and better L2 residency for the weight block currently
being swept.

Exposure: estimate 1.1-1.35x. It is a real but secondary lever; the 4x is not a batching artefact.

Evidence: `common/common.h:460-461`; `src/llama-context.cpp:2110`; `ggml/src/ggml-cpu/repack.cpp:4711-4716`
(4 chunks per thread over src0 rows, atomic chunk stealing at `:4754-4775`);
`internal/backend/simd/runtime_batch.go:35,487-508`.

### 3. Prefill attention: llama.cpp has a tiled 64x64 CPU flash kernel parallelised over query rows; mantle materialises one score row per head, softmaxes it separately, and runs every head and every row on one goroutine

Why: mantle's `RunAttnHeads` (`attnpool.go:21-118`) loops heads (`:37`) and then writes the whole
score row into `scoresBuf` (`:41-58`), calls `Softmax` on it as a separate pass (`:61`), then does the
weighted V sum (`:63-115`). It is called once per query row from `attn_batch.go:117`
(`RunAttnHeads(&ctx, m.Scratch.Scores, 0, nHead)`) with no worker pool anywhere in the file — no
`go` statement, no channel, no atomic. So prefill attention is single-threaded, and a 64-row chunk
runs 64 sequential per-row attentions.

llama.cpp has two CPU paths, both threaded over query rows with the same atomic chunk stealing
(`ops.cpp:9540-9582`): a streaming online-softmax kernel (`ops.cpp:8834-9080`, running max `M` and
sum `S`, never materialising scores) and, for f32/f16 KV with `n_q >= 64`, a tiled flash kernel with
`GGML_FA_TILE_Q = 64`, `GGML_FA_TILE_KV = 64` (`ggml-cpu/common.h:9-10,104-107`;
`ops.cpp:9080+`). The tiled form computes 64 query rows against each 64-position KV tile, so K/V are
read once per 64 queries instead of once per query.

Exposure: depends on attention's share of prefill (Scenario A/E measure it). Using 15% at 650 tokens
as a placeholder: threading the row loop over 16 workers alone cuts that to ~1-2%, worth ~1.15x;
the K/V-tile reuse in the tiled kernel adds more when the window is long. Mantle also pays a second
pass over the score row for softmax that the online form eliminates.

Evidence: `internal/backend/simd/attnpool.go:21-118`; `internal/backend/simd/attn_batch.go:117`;
`/work/ai/llama.cpp/ggml/src/ggml-cpu/ops.cpp:8834-9080,9540-9582,9080`;
`/work/ai/llama.cpp/ggml/src/ggml-cpu/common.h:9-10`.

### 4. Thread count and topology: llama.cpp sizes itself to the P-cores and never uses the E-cores; mantle always runs `GOMAXPROCS` (16) workers in lockstep

Why: `common_cpu_get_num_math()` (`common/common.cpp:206-249`) detects a hybrid CPU, walks CPUs
pinning to each in turn, skips any that lands on an efficiency core, skips the sibling for
hyper-threading ("hyperthreading isn't useful for linear algebra", `:186-199`), and returns the
P-core count. `llama-bench --help` prints `-t, --threads <n> (default: 4)` on this box, which is
that function's return value: **the 178.7 tok/s reference was produced with 4 threads on 4 P-cores.**
llama.cpp also exposes `--cpu-mask`/`--cpu-strict` and feeds a cpumask into the threadpool
(`common/common.cpp:1771-1787`).

Mantle builds its pool at `max(runtime.GOMAXPROCS(0), 1)` = 16 workers (`matvec.go:105-120`), round-robins
tasks through an extra dispatcher goroutine (`:140-151`), and `MatMatQuant` blocks until *every* one of
the 16 has reported done (`matmat_quant.go:120-131`). Tile stealing at 16-row granularity
(`matmat_quant.go:137,143-151`) means a matmul finishes when the slowest worker finishes; on a hybrid
part, a straggler on an E-core now and then sets the tail. There is no affinity, no core-class
awareness, no way to say "P-cores only".

Exposure: unmeasured, and it is the reason the headline comparison is not apples-to-apples. Per-thread
efficiency: 178.7/4 = 44.7 tok/s/thread for llama.cpp versus 44.3/16 = 2.8 tok/s/thread for mantle, a
~16x per-thread difference. Even if 16 threads should in principle beat 4 by ~2x on this part, the
measured 4x gap means mantle's per-thread work rate is the problem. Directly relevant to Scenario G:
the first thing to try is not `GOMAXPROCS=16` but llama.cpp's default, because the reference number
was produced that way.

Evidence: `common/common.cpp:186-199,206-249`; `llama-bench --help` output (`-t ... (default: 4)`);
`internal/backend/simd/matvec.go:105-120,140-151`; `internal/backend/simd/matmat_quant.go:120-131,137`.

### 5. Weight repacking happens once at load in llama.cpp; mantle decodes the packed nibbles per call

Why: on AVX2 llama.cpp rewrites Q4_K weights into `block_q4_Kx8` at load time — eight rows
interleaved, sub-block major (`repack.cpp:5011-5016` selecting `q4_K_8x8_q8_K`, layout built in
`repack.cpp:3569-3595` `repack_q4_K_to_q4_K_8_bl` + `make_block_q4_Kx8`). The GEMM then reads the
payload directly in exactly the order its registers want. The equivalent activation side is
`ggml_quantize_mat_q8_K_4x8`, an AVX2 kernel that quantises and interleaves four rows at once
(`arch/x86/repack.cpp:290-365`).

Mantle keeps the container's per-row packed payload and re-derives the kernel's order on every call:
`decodePacked4Row` expands each 16-byte block to 32 int8 in scratch (`matvec_packed4.go:147-157`,
called from `matmat_quant.go:218,222`), the activation rows are permuted into pair order into a second
buffer (`matmat_quant.go:48-59`), and the per-pair scale quads are rebuilt (`:60-69`). For packed 4-bit
dtypes mantle also deliberately skips the int8 cache, so this happens on every call
(`quant_cache.go:39-45`).

Exposure: small. One full weight decode per chunk is ~56M 16-byte groups x ~6 vector ops = ~340M
instructions per chunk, ~0.07 s over the 11 chunks of a 650-token prompt (estimate from 4 cores at
~3 IPC, 4.5 GHz) = ~0.5% of 14.7 s, plus L1 traffic that the kernel would pay anyway when reloading
weight blocks per 4-row group. Estimate 0.5-2%.

Evidence: `repack.cpp:5011-5016,3569-3595`; `arch/x86/repack.cpp:290-365`;
`internal/backend/simd/matvec_packed4.go:147-157`; `internal/backend/simd/matmat_quant.go:47-69,218-222`;
`internal/backend/simd/quant_cache.go:39-45`.

### 6. Activation quantisation is a serial scalar loop in mantle; llama.cpp does it cooperatively across threads with an AVX2 four-row kernel

Why: mantle quantises each activation row in `quantizeVecBlocksInto` — a plain Go loop that zeroes the
buffer, finds a per-32-block max, then does `math.Round(float64(v*inv))` per element and writes both
an int8 and an int16 copy (`matvec_quant.go:671-722`). It runs on the calling goroutine, once per
tensor, before any worker is started (`matmat_quant.go:113-115` then `:126-131`), so it is 100% serial.
llama.cpp splits the same work across all `nth` threads by block range (`ggml-cpu.c:1364-1381`), or, on
the repack path, quantises four rows at once with `_mm256_loadu_ps`/`_mm256_cmp_ps` AVX2 code
(`repack.cpp:4696-4701` -> `arch/x86/repack.cpp:290-365`), also thread-parallel, and then hits one
barrier (`ggml-cpu.c:1393`, `repack.cpp:4749`).

Exposure: per 650-token prompt the serial quantisation covers roughly
(sum of activation widths per layer = 26624) x 64 rows x 16 layers x 11 chunks = ~300M scalar element
ops, ~1.5 ns each = ~0.45 s, i.e. ~3% of 14.7 s, all of it with 15 of 16 cores idle. Plus mantle's
extra `q8Pair` permute and `pairScales` rebuild passes (`matmat_quant.go:47-69`). Estimate 3-5%.

Evidence: `internal/backend/simd/matvec_quant.go:671-722`; `internal/backend/simd/matmat_quant.go:47-69,113-131`;
`/work/ai/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1364-1393`.

### 7. Per-op synchronisation: one spin barrier inside a persistent OS-thread pool versus a Go channel round trip per matmul

Why: llama.cpp creates its worker threads once per context (`ggml-cpu.c:3388`), parks them on a
condition variable between graphs (`:3254`, `:3276`) and synchronises inside a graph with an atomic
spin barrier (`ggml_barrier`, `:585-620`, polling `ggml_thread_cpu_relax`), so advancing from one
matmul node to the next costs a few hundred ns and no scheduler involvement. It also uses two
threadpools, `n_threads` for decode and `n_threads_batch` for prefill (`src/llama-context.cpp:2957-2958`).
Mantle's pool is persistent too (`matvec.go:105-160`) but every matmul rents a `done` channel
(`:198`, `matmat_quant.go:120`), pushes N tasks through a dispatcher goroutine that re-sends each to a
per-worker channel (`matvec.go:140-151`), and then collects N done signals. That is ~2N channel
operations plus a goroutine hop per matmul, and it is repeated for every projection of every layer.

Exposure: ~112 matmuls per chunk (7-8 projections x 16 layers) x 11 chunks = ~1250 calls; at 10-30 us
of dispatch cost each that is 12-37 ms of 14.7 s = 0.1-0.3%. Real but rounding-error scale.

Evidence: `ggml-cpu.c:585-620,3254,3276,3388`; `src/llama-context.cpp:2955-2961`;
`internal/backend/simd/matvec.go:105-160,140-151,198`; `internal/backend/simd/matmat_quant.go:120-131`.

### 8. llama.cpp has a dedicated large-batch BF16/F16/F32 GEMM (tinyBLAS); mantle caps its BF16 kernel at 16 rows and falls back to a scalar-converting packed GEMM above that

Why: llama.cpp tries `llamafile_sgemm` first for every mul_mat whose src1 is contiguous f32
(`ggml-cpu.c:1310-1332`), and that dispatches BF16 to a `tinyBLAS<8, __m256, __m256, bf16, bf16, float>`
kernel with an 8-row register tile for any `n >= 2` (`llamafile/sgemm.cpp:3911-3913`, `:4001-4008`).
Mantle's `useSmallBatchBF16` only fires for `n <= smallBatchMaxRows = 16`
(`matmat_bf16.go:11-20`), so the 64-row chunk routes bf16 weights to `GemmParWT` instead, whose own
comment says the packed path "copies every weight tile through a scalar conversion first, which
dominates for small batches" (`matmat_bf16.go:26-29`). This is a plausible explanation for the bf16
row in the brief (11.2 tok/s, 4x below mantle's own k4).

Note this does not touch the Q4_K comparison: `llamafile_sgemm` has no Q4_K case at all
(`grep -c GGML_TYPE_Q4_K llamafile/sgemm.cpp` -> `0`), so llama.cpp's Q4_K prefill runs entirely
through the repack GEMM. Q8_0 *does* take tinyBLAS (`sgemm.cpp:4133-4142`), and Q8_0 is not repacked on
x86 (`repack.cpp:5110-5135` has NEON/RISC-V cases only), which is why Q8_0 prefill (115.2) behaves
differently from Q4_K_M (178.7).

Exposure: unknown for the k4 gap; for bf16 it looks like the dominant term. Estimate 2-4x on bf16
dtype only. Unverified without a bf16 llama.cpp reference number.

Evidence: `ggml-cpu.c:1310-1332`; `llamafile/sgemm.cpp:3911-3913,4001-4008,4133-4142`;
`repack.cpp:5110-5135`; `internal/backend/simd/matmat_bf16.go:11-29`;
`internal/backend/simd/runtime_batch.go:564-582`.

---

## Summary table

| # | Technique | llama.cpp | mantle | Est. share of the 4.0x |
|---|---|---|---|---|
| 1 | GEMM register tile | 4 rows x 8 cols = 32 acc per weight load | 2 rows x 4 cols = 8 acc | 2.0-2.5x |
| 2 | Prefill batch size | ubatch 512 | chunk 64 | 1.1-1.35x |
| 3 | Prefill attention | tiled 64x64 flash, threaded over q rows | materialised score row, serial | ~1.15x at 650 tok |
| 4 | Thread sizing / topology | P-core count (4 here), optional pinning | `GOMAXPROCS` (16), no affinity | confounds the baseline |
| 5 | Weight repacking | once at load (`block_q4_Kx8`) | per-call nibble decode | 0.5-2% |
| 6 | Activation quantisation | threaded, AVX2 4-row | serial scalar loop | 3-5% |
| 7 | Per-op sync | spin barrier | channel round trip per matmul | 0.1-0.3% |
| 8 | Large-batch bf16 GEMM | tinyBLAS 8-row | capped at 16 rows | bf16 only, 2-4x |

Ordering conclusion: the gap is a *kernel efficiency* gap, not a batching gap. Findings 1 and 4 say
that at equal batch size mantle gets roughly a third to a half as much work out of an instruction,
and ~1/16th as much out of a thread-second - though finding 4's per-thread contrast is confounded,
since the reference number came from 4 threads and mantle's from 16. Findings 2, 3, 5, 6, 7 are the
surrounding overheads and together account for well under one of the four missing x.

Confidence: high on the code-level claims (all quoted from the current trees, file:line above) and on
the non-differences in the first section; medium-low on every numeric exposure estimate, because this
scenario ran no models and the per-finding shares are analytic estimates from instruction counts and
byte counts, not measurements.

Could not determine:

- The real split of the 4x between kernel instruction count, attention, and per-op overhead — needs
  Scenario A's pprof on the same 650-token prompt, and Scenario C's microbenchmarks per shape.
- Whether mantle's 16-thread pool actually loses to a P-core-only 4-thread configuration on this box.
  Requires a `GOMAXPROCS` sweep (Scenario G) and, for a true apples-to-apples reference, `llama-bench
  -t 16` against `-t 4`. Not run here (no model runs permitted).
- How much of llama.cpp's repack-GEMM advantage survives at n=64 rather than n=512; llama.cpp has no
  knob to force the non-repacked path for Q4_K without rebuilding with `GGML_CPU_REPACK=OFF`.
- Whether a `VPDPBUSD` form is reachable from Go at all without assembly; the `simd/archsimd` API
  surface has no such op, but I did not audit whether the Go compiler recognises a
  `DotProductPairs(ones)` pattern and folds it (no evidence either way, and `GOEXPERIMENT=simd`
  disassembly was out of scope for a read-only scenario). This only matters if a rewrite goes after
  VNNI, which llama.cpp's own Q4_K prefill GEMM does not use on this box.
- LFM3-1.7B's exact per-layer shapes used in my byte/instruction estimates; I used the Scenario C shape
  list from the brief rather than `mantle inspect` (which would have required running the binary).
- Whether llama.cpp's `n_threads_batch` (prefill) differs from `n_threads` in the reference run;
  `llama-bench` sets both to the same value (`tools/llama-bench/llama-bench.cpp:2169`), so probably not,
  but the table in the brief does not say which flags were used.
