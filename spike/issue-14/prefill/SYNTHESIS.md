# CPU prefill profiling — salvage and synthesis (issue #15)

Nine agents were dispatched on 2026-11-16 02:29. Three delivered reports:
**A** (pprof breakdown), **D** (kernel audit), **H** (llama.cpp reference). Six exceeded
their 30-minute budget: **B**, **C**, **E**, **F**, **G**, **I**. That was a planning
failure on my part, not the agents': all nine were serialised through one measurement
lock (`flock /work/tmp/prefill-measure.lock`) so their numbers could not corrupt each
other, and the queue consumed most of the budget for the six that ran last.

Their reports are missing. Their raw measurements are not, and the usable ones are
recorded below with the command that produced them. Anything a dead agent left that I
could not tie to a completed run is not used.

## The one-line answer

Prefill is not primarily a kernel problem. **72% of k4/q8 prefill wall time is a single
goroutine**, and the quantised GEMM that consumes 86.5% of *CPU* occupies only 28% of
*elapsed* time because it runs across ~12.6 threads. Parallelising the serial work is
worth more than optimising the kernel, and costs no numerics change.

## Ranked findings

### 1. Attention and activation quantisation are never parallelised (48% and 17% of prefill wall)

Why: `batchPrefillChunk.forEachRow` (`runtime_batch.go:608`) is a plain serial loop over
rows and layers. Projections inside it are dispatched to the 16-worker pool, but
attention (`attn_batch.go:126-136` → `attendRow` → `RunAttnHeads` inline at `:117`, one
position at a time) and activation quantisation (`MatMatQuant` calls `getQuantRows` at
`matmat_quant.go:113` *before* dispatching at `:125-130`) both stay on the caller, with
15 of 16 cores idle.

Exposure: 13.40 s of the 18.65 s k4 prefill is single-threaded; attention alone is
8.86 s (48%) and activation quantisation 3.17 s (17%). Parallelising attention across
the same ~12.6 threads takes prefill 18.65 s → ~10.5 s (+78% throughput); adding the
quantisation takes it to ~7.0 s (+167%). At the baseline rate of 44.3 tok/s that is
~79 → ~118 tok/s, i.e. past llama.cpp's Q8_0 (115) and about two-thirds of Q4_K_M (179).

Evidence: `pprof -top` k4 shows the serial subtree as
`13.40s 16.17% (*Instance).PrefillTokens` with children
`9.66s attentionBlock` and `3.17s MatMatQuant`, while the pool's frames
(`newMatVecPool.func1` 68.62 s, `quantJob.run` 66.19 s) are a disjoint goroutine.
Process-wide `/usr/bin/time -v`: `Percent of CPU this job got: 549%`.

### 2. The batched prefill path is worth 3.9x, so any ineligible layer or fallback is expensive

Why: with `MANTLE_CPU_BATCHED_PREFILL=0` the prompt goes through the per-token path.

Exposure: q8 prefill **11.57 and 11.54 tok/s disabled against 44.85 and 44.67 enabled**
(loadavg 9.8-10.0 during both). So the batched path is 3.88x, and every layer that fails
the batch-eligibility gate silently costs that layer 4x on prefill. Also the reason the
naive `--no-quant-cache` measurement in issue #14 was 8.7x slower on decode: it turns the
batched path off too.

Evidence: `/work/tmp/prefill-b/results.txt`, lines `q8-batched-off` and `q8-batched-on`.
The k4 batched-off half of this comparison was killed mid-run (an earlier attempt used a
650-token prompt, which at 1.5 tok/s is 7 minutes for one pass) and never re-measured.

### 3. The exactness contract caps the kernel win at 1.4x, and VNNI is not reachable today

Why: per 32-element block per 2x4 tile the kernel runs ~55 vector instructions. The
cross-lane reduce plus fp32 scale fold happens every 32 elements because
`MatMatQuant` must reproduce `MatVec`'s bits in the same block order.

Exposure, three ceilings on the same kernel: ~39 instructions (1.4x) with unsigned
weights, no per-row `MulSign` and a one-off `-8*sum(x)`; ~31 instructions (1.9x) with
`vpdpbusd`; ~14 instructions (4.1x) with int32 accumulation across a 256-element
superblock and one fp fold per 8 blocks, which is where llama.cpp sits.

`simd/archsimd` exposes no dot product on any 8-bit vector type (`Int8x16` has only
`AddSaturated`/`SubSaturated`; `Uint8x16` has `Add`/`And`/`AES*`/conversions), and no
`golang/go` issue tracks byte VNNI. Go's assembler does accept `VPDPBUSD`, but
`avx_optabs.go:3056-3059` encodes it EVEX-only, which needs AVX512_VNNI + AVX512VL and
SIGILLs on an AVX-VNNI Alder Lake. There is no `avxvnni` symbol in
`cmd/internal/obj/x86`. So the 1.9x row is not reachable from Go today by any supported
route, and the 4.1x requires changing the numerics.

Evidence: `D-kernel-audit.md` findings 1-3; local checks of `go doc simd/archsimd` and
`$GOROOT/src/cmd/internal/obj/x86/avx_optabs.go`.

### 4. bf16 prefill spends ~43% of its CPU moving data rather than multiplying

Why: `useSmallBatchBF16` caps the vectorised kernel at 16 rows (`matmat_bf16.go:13-21`),
so at the default 64-row chunk bf16 goes through `GemmParWT` → `packBTileWT`
(`gemm_batch.go:189`), whose scalar loop assembles each element with two bound-checked
byte loads and a non-inlinable func-value convert (`:209`, `:215-222`). The f32 copy then
doubles inner-loop read width, and `blockUpdateAlpha1SIMDPacked` (`gemm_kernel.go:434`)
accumulates into C in memory with `TileK=16` over k=2048, reloading each C tile 128 times.

Exposure: `packBTileWT` is 64.60 s of bf16's 325.36 s samples (19.85%), and 77.4 s of cum
(23.8%) sits on the four C-tile `LoadFloat32x8` calls (`gemm_kernel.go:475-478`). Removing
the staging copy alone takes bf16 prefill 17.0 → ~21 tok/s.

Evidence: `A-pprof-breakdown.md` finding 2; `D-kernel-audit.md` finding 7.

### 5. Attention is materialised and serialised, against llama.cpp's tiled parallel flash kernel

Why: mantle writes one score row per head, softmaxes it separately, and runs every head
and every row on one goroutine. llama.cpp uses a 64x64 tiled CPU flash kernel
parallelised over query rows.

Exposure: bounded by finding 1. Attention is 11.0% of prefill *CPU* (A) but 48% of prefill
*wall*, because it never leaves the serial goroutine. Parallelising it is the single
cheapest large win in the whole set.

Evidence: `H-llamacpp-reference.md` finding 3; `A-pprof-breakdown.md` finding 1.

### 6. GC and the LM head are not factors (hypotheses eliminated)

Why: prefill asks for one output position only. `runtime_prefill.go:45` passes
`emitLastRow`, and `runtime_batch.go:661-665` runs a single `FusedRMSNormMatVec` over
`p.x[(n-1)*hidden:n*hidden]`. The loop is also effectively allocation-free in steady state.

Exposure: LM head ~0.1% of prefill CPU, embedding gather <0.5%, GC <0.5% (`gcBgMarkWorker`
and `mallocgc` never appear as nodes; the whole k4 profile is 46 nodes). Independently,
GOGC made no difference: one 650-token pass took **45.71 s at default GOGC, 46.61 s at
GOGC=400 and 46.55 s with GOGC=off**, peak RSS 1.21-1.27 GB.

Evidence: `A-pprof-breakdown.md`; `/work/tmp/prefill-I/bench-k4-650-gogc-*.time`.

### 7. Thread scaling is already near its practical limit at 16 threads

Exposure: GOMAXPROCS=16 gives prefill 44.56 tok/s (gen 11.72); GOMAXPROCS=4 gives 31.37
/gen 7.40. Only two points survived, so the parallel-efficiency curve could not be fitted;
what it does show is that going from 4 to 16 threads buys 1.42x, which is far from linear
and consistent with only ~6 threads being busy on average (A's 549% CPU).

Evidence: `/work/tmp/prefill-G/logs/mant650t16fine.out`, `mant650t4.out`.

### 8. Chunk size is real but secondary (not measured)

H estimates 1.1-1.35x from mantle's 64-position chunk against llama.cpp's ubatch of 512.
The sweep that would have tested it (B) timed out, so this remains an estimate from code
reading, not a measurement. It should be measured before changing the default, because
finding 2 shows the fallback penalty is large and finding 5 in D shows a 1-row chunk is
*slower* than the per-row path it replaces (`useBatchQuant` accepts `n >= 1`).

## Recommendation, in order

1. **Parallelise prefill attention over rows** (`attn_batch.go:126-136`,
   `RunAttnHeads`), using the existing worker pool. 48% of prefill wall, no numerics
   change, and the same fix serves the `sliding_attention` layers.
2. **Move activation quantisation inside the worker dispatch** and cache one `QuantVec`
   across q/k/v and across gate/up, as the per-token path already does
   (`attn.go:199-207`, `ffn.go:43-44`). 17% of wall, and 6 quantisations of one block
   where 2 suffice.
3. **Fix `useBatchQuant`'s `n >= 1` acceptance** so a tail chunk takes the per-row path
   instead of paying a full weight decode for one row.
4. **Kernel, within the contract**: unsigned 4-bit codes with a one-off `-8*sum(x)`, no
   per-row `MulSign` (D's 1.4x row).
5. **bf16**: let `MatMatBF16` handle the default 64-row chunk, or remove the func-value
   convert and widen once; accumulate C in registers.
6. Only then consider relaxing bit-exactness for per-256 activation scales, which is the
   4.1x row and a format/numerics decision, not a kernel tweak.

## What this deliberately does not settle

- Chunk-size sweep (`MANTLE_CPU_MAX_BATCH`), and whether raising it past 64 helps.
- Attention's share as a function of prompt length: the O(n^2) term was never fitted,
  because E's sweep produced a single usable point (N=16, prompt_tokens=11, which looks
  like a mis-built prompt) and was discarded.
- GEMM microbenchmarks against the memory roofline: C produced only a header line.
- Per-core utilisation and E-core placement: G's sampler output was not persisted.
- CUDA. Every number here is CPU. The GPU gap is worse (558 against 3643 tok/s) and
  unexamined.

## Provenance

Binary `/work/wt/mantle-parity-lfm3/bin/mantle` (main `043ab6a`, CPU only). Containers
from `/work/models/mcf/parity/lfm3-1.7b/`. All measurements taken on one box with loadavg
between 2 and 12; A's control run reproduced the baseline (45.42 tok/s at 266 tokens,
549% CPU), so the *shares* are sound while individual absolute rates in agent logs are
depressed by contention.
