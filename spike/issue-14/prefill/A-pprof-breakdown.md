# Scenario A — pprof function-level breakdown of prefill (k4, q8, bf16)

Model: LFM3-1.7B, `/work/models/mcf/parity/lfm3-1.7b/LFM3-1.7B.{k4,q8,bf16}.mcf`,
binary `/work/wt/mantle-parity-lfm3/bin/mantle` (main `043ab6a`, CPU only, built
`20261115T142106Z`). Config: 24 layers (13 `conv`, 6 `full_attention`, 5
`sliding_attention`), hidden 2048, 16 heads / 4 KV heads, head_dim 128, FFN 7680,
vocab 128000.

## How the profiles were taken

Prompt (650 tokens after templating), as the brief specifies:

```
PROMPT=$(python3 -c "print(' '.join(['The river had been falling all week and by Friday the gravel bars stood clear.']*40))")
```

Each run, one at a time under the shared lock (`/work/tmp/prefill-A/run.sh`):

```
flock /work/tmp/prefill-measure.lock bash -c \
  './bin/mantle run -m /work/models/mcf/parity/lfm3-1.7b/LFM3-1.7B.k4.mcf \
     --cpuprofile /work/tmp/prefill-A/cpu-k4.pprof -n 4 --temp 0 -p "$PROMPT"'
```

| dtype | loadavg before | loadavg after | prompt_tokens | prompt_tps | gen_tps | profile duration | profile samples |
|---|---|---|---|---|---|---|---|
| k4 | 2.30 3.45 4.72 | 3.82 3.75 4.79 | 650 | 34.85 | 10.46 | 19.32 s | 82.86 s (428.9%) |
| q8 | 3.82 3.75 4.79 | 5.35 4.34 4.91 | 650 | 35.34 | 9.54 | 20.87 s | 84.37 s (404.2%) |
| bf16 | 5.35 4.34 4.91 | 10.77 8.09 6.35 | 650 | 16.98 | 8.07 | 39.25 s | 325.36 s (829.0%) |

One extra control run, to get whole-process CPU% (pprof's sample total alone cannot
distinguish "few threads" from "slow threads"):

```
flock /work/tmp/prefill-measure.lock bash -c '/usr/bin/time -v ./bin/mantle run \
  -m .../LFM3-1.7B.k4.mcf -n 4 --temp 0 -p "$PROMPT16"'      # 16 reps = 266 tokens
```
loadavg 11.62 before, 4.60 after; `prompt_tps=45.42`, `Percent of CPU this job got: 549%`,
`User time 35.28 s`, `Elapsed 6.47 s`, `Maximum resident set size 1187312 kB`.

Analysis commands (no model load, so no lock):

```
GOEXPERIMENT=simd go tool pprof -top -nodecount=40   bin/mantle /work/tmp/prefill-A/cpu-$d.pprof
GOEXPERIMENT=simd go tool pprof -top -cum            bin/mantle /work/tmp/prefill-A/cpu-$d.pprof
GOEXPERIMENT=simd go tool pprof -tree -nodecount=70  bin/mantle /work/tmp/prefill-A/cpu-$d.pprof
GOEXPERIMENT=simd go tool pprof -list=<func>         bin/mantle /work/tmp/prefill-A/cpu-$d.pprof
GOEXPERIMENT=simd go tool pprof -peek=<func>         bin/mantle /work/tmp/prefill-A/cpu-$d.pprof
```

Raw captures: `/work/tmp/prefill-A/{top,topcum,tree,full,list}-*.txt`, profiles
`cpu-{k4,q8,bf16}.pprof`.

**Absolute numbers are depressed by contention** (34.9/35.3 tok/s here against 44.3/46.1
in the brief's table, loadavg 2.3 → 5.4 during those runs). bf16 is the reverse: 16.98
tok/s against the brief's 11.2, because it parallelises better and got more of the box.
Percentages below are shares of CPU, which contention moves much less than rates; the k4
control run at 45.4 tok/s reproduces the brief's rate, so the shape is sound.

Only ~97% of each profile is prefill (k4 18.65 s of 19.32 s wall, q8 18.39 of 20.87,
bf16 38.29 of 39.25), so the tables below divide by the prefill share: k4 96.8% of
samples, q8 94.8%, bf16 98.5%. Without that correction every prefill percentage is
2–4 points lower and nothing reorders.

## Phase table — share of *prefill* CPU, per dtype

| phase (functions) | k4 | q8 | bf16 |
|---|---|---|---|
| **Quantised GEMM path, total** | **86.5%** | **85.8%** | **95.6%** |
| — weight-side kernel (`matMatRangeQuant`/`quantDot2x4`; bf16 `gemmRangeColsWT`/`blockUpdateAlpha1SIMDPacked`) | 82.5% | 82.1% | 75.3% |
| — weight staging / bf16→f32 decode (`packBTileWT`) | n/a | n/a | 20.1% |
| — activation quantisation (`MatMatQuant`→`getQuantRows`→`quantizeVecBlocksInto`) | 4.0% | 3.6% | n/a |
| **Attention** (`RunAttnHeads` = `DotF16` + `accumulateScaledF16` + `Softmax`) | **11.0%** | **11.2%** | **2.8%** |
| **LM head** | **~0.1%** | **~0.1%** | **~0.1%** |
| FFN gate/up activation (`activateFFNRow`, `fastSiluVec`, `fastSigmoidVec`) | 1.4% | 1.5% | <0.5% |
| short-conv block (`shortConvBlock`, 13 of 24 layers) | 0.7% | 0.8% | <0.5% |
| RMSNorm (all sites) | <0.5% | <0.5% | <0.5% |
| softmax (`DefaultOps.Softmax`) | 0.3% | 0.2% | <0.1% |
| embedding gather (`Embeddings.RowTo`) | <0.5% | <0.5% | <0.5% |
| GC (`gcBgMarkWorker`, `mallocgc`) | <0.5% | <0.5% | <0.5% |

"<0.5%" is literal: pprof dropped nodes at cum ≤ 0.41 s for k4 (≤ 1.63 s for bf16), and
**the entire k4 profile is 46 nodes** — RMSNorm, embedding and the collector never appear
as nodes at all. CPU per prompt token: k4 123 ms, q8 123 ms, bf16 493 ms.

The LM head is ~0.1% because prefill asks for one position only:
`runtime_prefill.go:45` passes `emitLastRow`, and `runtime_batch.go:661-665` then runs a
single `FusedRMSNormMatVec` over `p.x[(n-1)*hidden:n*hidden]`. The 2.4–2.9% of
`matVecRange*` in the profiles is the four decode tokens, not prefill.

## Wall-time decomposition — where the serialiser is

pprof samples are on-CPU time, so a single goroutine can only accumulate 13.4 s of
samples in ≥13.4 s of wall. That makes the caller's own frame a direct measure of
single-threaded time:

| dtype | prefill wall | caller goroutine (`PrefillTokens`, serial) | remainder (GEMM windows) | threads in GEMM windows |
|---|---|---|---|---|
| k4 | 18.65 s | 13.40 s (72%) | 5.25 s | ≈12.6 |
| q8 | 18.39 s | 13.34 s (73%) | 5.05 s | ≈13.0 |
| bf16 | 38.29 s | 11.06 s (29%) | 27.23 s | ≈11.1 |

So the quantised GEMM that everyone suspects is **not** what prefill spends its wall time
on at 4 and 8 bit: it runs at ~12.6 threads for 5.25 s of an 18.65 s prefill. Two thirds
of prefill wall is a single goroutine doing attention, activation quantisation and the
FFN elementwise sweep, with 15 of 16 cores idle.

---

## Findings, ranked by exposure

### 1. 72% of k4/q8 prefill wall is one goroutine: attention and activation quantisation are never parallelised

Why: `batchPrefillChunk.forEachRow` (`internal/backend/simd/runtime_batch.go:608`) is a
plain serial loop over rows and layers. The projections inside it (`p.gemmRows`) are
dispatched to the 16-worker pool and waited on (`matmat_quant.go:125-130`), but everything
else stays on the calling goroutine, including two items that are pure compute:

- all of prefill attention: `attn_batch.go:126-136` calls `attendRow(r)` once per row and
  `attendRow` (`attn_batch.go:86-124`) calls `RunAttnHeads(&ctx, ...)` inline at line 117,
  one position at a time, no fan-out over rows, heads or KV blocks.
- all of activation quantisation: `MatMatQuant` calls `getQuantRows` (`matmat_quant.go:113`)
  *before* dispatching (`matmat_quant.go:125-130`), so `quantizeVecBlocksInto` over the
  whole chunk happens on the caller and the pool sits idle while it runs.

Exposure: **13.40 s of the 18.65 s k4 prefill (72%) is single-threaded**, and 13.34 s of
18.39 s for q8. Of that, attention is 8.86 s (48% of prefill wall) and activation
quantisation 3.17 s (17% of prefill wall). Parallelising attention alone across the same
~12.6 threads the GEMM already gets takes prefill from 18.65 s to ~10.5 s (34.9 → ~62
tok/s, +78%); parallelising the activation quantisation too takes it to ~7.0 s (~93 tok/s,
+167%). This is the single largest item in the 4x gap and it is not a kernel problem.

Evidence: `pprof -top` k4 reports the caller's serial prefill subtree as
`13.40s 16.17% ... (*Instance).PrefillTokens` with children
`9.66s 11.66% ... (*batchPrefillChunk).attentionBlock` and `3.17s 3.83% ... MatMatQuant`,
while the pool's frames (`newMatVecPool.func1` 68.62 s, `quantJob.run` 66.19 s) are a
disjoint goroutine. Process-wide, `/usr/bin/time -v` on the 266-token k4 run reports
`Percent of CPU this job got: 549%` against `User time 35.28 s` / `Elapsed 6.47 s`, i.e.
~6 of 16 threads busy on average. bf16 shows what
the same code looks like when the serial share is small (29% of wall → 829% CPU).

### 2. bf16: 20% of prefill CPU is a scalar bf16→f32 staging copy, and another 23% is re-reading the C tile

Why: `projectRows` (`runtime_batch.go:564`) sends bf16 weights to `GemmParWT`, whose
only kernel path is `gemmRangeColsWT` (`gemm_batch.go:93`). That function calls
`packBTileWT` (`gemm_batch.go:189`) for every `(j0,k0)` tile, and for `w.Data == nil`
storage it runs a scalar loop that assembles each bf16 element with two bound-checked
byte loads and converts it (`gemm_batch.go:215-222`), writing an f32 copy into `packB`.
The f32 copy then doubles the bytes the inner kernel reads, and
`blockUpdateAlpha1SIMDPacked` (`gemm_kernel.go:434`) accumulates into `C` in memory rather
than in registers: with `TileK=16` and k=2048, every C tile is loaded and stored 128 times.

Exposure: **`packBTileWT` is 64.60 s of bf16's 325.36 s samples (19.85%), and a further
77.4 s of cum (23.8%) sits on the four C-tile `LoadFloat32x8` calls at
`gemm_kernel.go:475-478`.** Together ~43% of bf16 prefill CPU is spent moving or widening
data around the 32×32×16 f32 GEMM rather than multiplying. Removing just the staging copy
takes bf16 prefill from 17.0 to ~21 tok/s.

Evidence: `pprof -top` bf16
```
    46.62s 14.33% 78.78%     64.60s 19.85%  .../simd.packBTileWT
    26.79s  8.23% 87.01%     26.80s  8.24%  simd/archsimd.LoadFloat32x8 (inline)
    15.10s  4.64% 91.65%     17.84s  5.48%  .../core.BF16ToFloat32
   209.09s 64.26% 64.45%    241.28s 74.16%  .../simd.blockUpdateAlpha1SIMDPacked
```
and `-list=packBTileWT`:
```
    10.83s     10.87s    217:		for kk := 0; kk < kInner; kk++ {
    10.20s     10.26s    219:			u := uint16(w.Raw[off]) | uint16(w.Raw[off+1])<<8
    21.13s     39.01s    220:			dst[kk*width+jj] = decode(u)
```
`-list=blockUpdateAlpha1SIMDPacked`:
```
     2.18s      4.06s    475:			acc0 = archsimd.LoadFloat32x8(cRow[j:])
    47.58s     54.01s    476:			acc1 = archsimd.LoadFloat32x8(cRow[j+8:])
    14.32s     17.26s    477:			acc2 = archsimd.LoadFloat32x8(cRow[j+16:])
    13.35s     18.52s    478:			acc3 = archsimd.LoadFloat32x8(cRow[j+24:])
```

### 3. The quantised inner kernel spends about half its time on the block-scale epilogue, not on the int8 dots

Why: `quantDot2x4` (`matmat_quant.go:341`) folds block scales at 4-lane f32 precision per
32-element block. Per block it does 8 `blockDotInt8` dot-pairs (the actual VNNI work), then
two 4-deep horizontal-add trees (`ConcatAddPairsGrouped` ×4), a `ConvertToFloat32`, two
4-float scale loads, two multiplies and an add per output group. The arithmetic is
correct-by-construction (it reproduces `matVecRangeQuantCached` bit for bit) but the
epilogue is not free.

Exposure: of `quantDot2x4`'s 57.76 s cum in the k4 profile, the epilogue lines account for
**27.6 s (48% of the kernel, 33% of all prefill CPU)**:
scale-vector loads + horizontal-add trees + the f32 scale fold = 1.58 + 2.01 + 2.60 +
2.49 + 1.75 + 9.95 + 7.25 = 27.63 s. The eight int8 dots plus their `MulSign` sign flips
are ~9.4 s of cum on lines 355-358/365-368. A further 5.79 s (7% of prefill CPU) sits on
`for b := range blocks` itself, i.e. loop bookkeeping — the block loop is not unrolled.

Evidence (`pprof -list=quantDot2x4`, k4):
```
    50.95s     57.76s (flat, cum) 69.71% of Total
     5.79s      5.79s    345:	for b := range blocks {
     1.58s      1.58s    351:		ws := archsimd.LoadFloat32x4Array((*[4]float32)(wPairs[4*b : 4*b+4]))
     2.49s      2.49s    359:		h := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))
     2.01s      2.01s    360:		xs := archsimd.LoadFloat32x4Array((*[4]float32)(xPairs01[4*b : 4*b+4]))
     9.95s      9.95s    361:		sum01 = sum01.Add(h.GetLo().Add(h.GetHi()).ConvertToFloat32().Mul(ws.Mul(xs)))
     2.60s      2.60s    370:		xs = archsimd.LoadFloat32x4Array((*[4]float32)(xPairs23[4*b : 4*b+4]))
     7.25s      7.25s    371:		sum23 = sum23.Add(h.GetLo().Add(h.GetHi()).ConvertToFloat32().Mul(ws.Mul(xs)))
```

### 4. Prefill is compute-bound, not bandwidth-bound: 4-bit buys nothing over 8-bit

Why: the k4 container is 906 MiB of weights and the q8 container 1769 MiB, yet at the same
`n=64` chunk the weight-side GEMM costs the same CPU in both. The kernel does one int8
block-dot per weight byte in either case; the pack width only changes how many bytes are
read (`matmat_quant.go:207-227` decodes packed4 rows into scratch first, then runs the same
`quantDot2x4`).

Exposure: k4 `quantJob.run` 66.19 s vs q8 65.71 s (**0.7% difference for 2x the weight
bytes**), and total prefill 80.2 s vs 80.0 s. Eleven chunk passes over 906 MiB is ~10 GB in
5.25 s of wall (~1.9 GB/s) — roughly 15x below what this box streams, so there is no
bandwidth wall here to blame. Corollary: any fix that reduces weight traffic (larger
chunks, repacking) will not move 4-bit prefill on its own; the int8 kernel and the serial
section will.

Evidence: `pprof -top -cum` k4 `66.19s 79.88% (*quantJob).run` (containers per
`/work/models/mcf/parity/lfm3-1.7b/MANIFEST.md`: k4 950,453,616 B, q8 1,855,170,160 B).

### 5. Chunking re-decodes the packed 4-bit weights once per 64-position chunk

Why: prefill splits the prompt into `MaxBatch` chunks (`runtime_batch.go:35`
`defaultMaxBatch = 64`, loop at `runtime_batch.go:487`), and each chunk runs a fresh
`MatMatQuant` per tensor. For packed 4-bit weights `matMatRangeQuant` has no int8 cache to
read, so `rowsAt` (`matmat_quant.go:207-227`) decodes each weight row into scratch
(`decodePacked4Row`, `packed4RowScales`) again. 650 tokens = 11 chunks = 11 decodes of the
whole k4 weight set.

Exposure: **`decodePacked4Row` 1.91 s + `packed4RowScales` 2.41 s = 4.3 s, 5.4% of k4
prefill CPU** (plus part of the 4.47 s flat in `matMatRangeQuant`). q8 pays none of it
because `w.Quant` holds int8 rows; this is the one place k4 is *more* expensive than q8.
Chunk size is Scenario B's lever; the number here is what a 64-position chunk costs.

Evidence: `pprof -top` k4
```
     1.94s  2.34% 83.48%         2s  2.41%  .../simd.packed4RowScales
     1.65s  1.99% 85.47%      1.91s  2.31%  .../simd.decodePacked4Row
```
`pprof -tree` k4 attributes both to `matMatRangeQuant.func2` (`3.67s 4.43%`), the closure
at `matmat_quant.go:218-223`.

### 6. Attention is the only O(n²) phase and it runs at ~1.8 GFLOPS on one core

Why: `RunAttnHeads` (`attnpool.go:21`) materialises one score row per head per position
(`scores := scoresBuf[:winLen]` at `attnpool.go:34`, filled key by key at
`attnpool.go:39-56`), calls `Softmax` separately (`attnpool.go:61`), then accumulates V
position-outer (`attnpool.go:66-80`). Nothing is tiled: each key costs a `DotF16` over
head_dim=128 with per-call constant setup.

Exposure: k4 attention is 8.86 s of 18.65 s wall (48%) and 11.0% of prefill CPU; q8 the
same. FLOPs from the config (11 attention layers, 16 heads, head_dim 128, 6 full + 5
sliding with window 256, sum over positions) give ≈15.9 GFLOP in 8.86 s single-threaded =
**1.8 GFLOPS**, ~3% of one AVX2 core. It is also the phase that grows super-linearly with
prompt length, so its share will be worse at 2k tokens. Softmax is only 2.6% of attention,
so this is score/PV data movement, not the normaliser: `DotF16` 4.10 s (46%),
`accumulateScaledF16` 3.11 s (35%), `Softmax` 0.23 s (2.6%), own flat 1.38 s.

Evidence: `pprof -peek=RunAttnHeads` k4
```
     1.38s  1.67%  1.67%      8.86s 10.69%  .../simd.RunAttnHeads
                                             4.10s 46.28% |   .../simd.DotF16
                                             3.11s 35.10% |   .../simd.accumulateScaledF16
                                             0.23s  2.60% |   .../simd.(*DefaultOps).Softmax
```
The row loop that feeds it is `attn_batch.go:126-136`, on the caller goroutine. Detail
belongs to Scenario E; the attribution is here.

### 7. `newF16Consts` per dot call: 1.7–1.9% of prefill CPU rebuilding the same constants

Why: `dotF16SIMD` builds its f16→f32 constants on every call
(`newF16Consts` → `archsimd.BroadcastUint32x8`, `f16_vec.go`), and attention calls it once
per key per head.

Exposure: k4 `newF16Consts` 1.43 s (1.73%) with `BroadcastUint32x8` 1.27 s beneath it; q8
1.50 s (1.78%). Free win if hoisted, small.

Evidence: `pprof -top` k4 `0.16s 0.19% 96.17% 1.43s 1.73% .../simd.newF16Consts`, child
`1.27s 88.81% ... archsimd.BroadcastUint32x8 (inline)`.

### 8. q8 pays a 2 s build of the int8 weight cache at load (not a prefill cost, but it is q8's only dtype-specific fixed cost)

Why: `loadMat` → `BuildQuantCache` (`internal/backend/core`) expands the q8 container into
int8 rows plus block scales at load when `--no-quant-cache` is off.

Exposure: q8 `model loaded duration=2.03 s` against k4 0.27 s and bf16 0.39 s;
`BuildQuantCache` is 1.64 s, 1.94% of the q8 profile. No effect on prefill tok/s, all of it
on time-to-first-token. In exchange it removes finding 5's decode cost, and the two roughly
cancel (66.19 s vs 65.71 s GEMM).

Evidence: `pprof -top -cum` q8 `1.64s 1.94% ... core.BuildQuantCache (inline)`, log line
`model loaded duration=2.035475337s`.

### 9. What is *not* costing anything (ruled out by this profile)

Why: worth stating explicitly so the other scenarios do not chase them.
- LM head: 0.1% of prefill CPU — `emitLastRow` already restricts it to one position
  (`runtime_prefill.go:45`, `runtime_batch.go:661-665`). Scenario F's arithmetic
  (262 M params × 650 positions) does not apply to this build.
- RMSNorm (24 layers × 2 norms × 650 rows), embedding gather, and the collector never
  appear as nodes: each is <0.5% of prefill CPU. `GOGC` sweeps cannot matter (Scenario I).
- softmax: 0.23 s, 0.3%.

Exposure: 0 tok/s recoverable; the *whole* prefill cost sits in findings 1-6.

---

## Allocations seen in the profile

`matMatRangeQuant` builds a fresh `&quantKernelScratch{}` and calls `ensure` on it whenever
`worker == nil` (`matmat_quant.go:185-196`), i.e. for the serial path and for the 16-row
tail — but in these profiles the collector never showed up, consistent with the scratch
being reused per worker. `MatMatQuant` reuses pooled quant rows (`defer putQuantRows(qr)`,
`matmat_quant.go:114`). Allocation accounting is Scenario I's job; this profile says
allocations are not a prefill cost.

## Confidence:

High on the per-dtype percentages and the function→phase mapping (direct pprof output, all
three dtypes, mutually consistent trees; the k4 control run reproduces the brief's 44-46
tok/s band). Medium-high on the wall-time decomposition in finding 1: it is arithmetic on
CPU seconds and wall clock, not a direct measurement of thread occupancy over time, and the
box was shared (loadavg 2.3-10.8), so the implied "12.6 threads in the GEMM window" could
be a little lower on an idle machine. Medium on the 1.8 GFLOPS attention figure (FLOPs
derived from `config.json` layer/head counts, not from a counter).

## What I could not determine

- Why k4/q8 reach only ~4.0-4.3 cores averaged over the whole run while bf16 reaches 8.3:
  I attribute it to the serial caller phases (finding 1), which explains the shape
  quantitatively, but I did not sample per-thread occupancy (Scenario G owns that).
- Whether the GEMM windows on an idle box really sustain 12.6 of 16 threads, or whether
  part of that remainder is pool dispatch latency (`matvec.go:140-147` round-robins tasks
  through a dispatcher goroutine into per-worker channels of capacity 1).
- The exact split between `MulSign` and the dot product inside `blockDotInt8`: `MulSign`
  is inlined and folded into the caller's line cost, so finding 3's "9.4 s of dots" is an
  upper bound on the pure dot work.
- Whether a wider `n` (chunk 128/256) would reduce the 5.4% packed4 re-decode faster than
  it increases per-chunk scratch (Scenario B).
- Cache and TLB effects: no `perf` counters were collected, so "compute-bound" in finding
  4 is inferred from CPU time being bit-width-independent plus a 15x headroom against
  memory bandwidth, not from a measured IPC.
