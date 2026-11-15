# Scenario D — static audit of the quantised batched GEMM

Scope: code reading only. No model was loaded and no benchmark was run, so no
`flock /work/tmp/prefill-measure.lock` wrapper was needed for this scenario. Every
number below is either (a) a count of operations visible in the source, or (b) simple
arithmetic over the published table in `COMMON.md` (44.3 tok/s k4 = 14.67 s for 650
tokens, 11 chunks of 64) and the published shapes. Nothing here is a measurement I made.

Source tree `/work/wt/mantle-parity-lfm3`, main `043ab6a`, clean tree. Files read in
full: `matmat_quant.go`, `gemm_kernel.go`, `gemm_batch.go`, `gemm_config.go`,
`matmat_bf16.go`, `quant_vec.go`, `matvec_packed4.go`, plus the call sites they need
(`runtime_batch.go:554-692`, `attn_batch.go:38-225`, `matvec.go:1-230`, `gemm.go:1-95`,
`matvec_quant.go:671-725`, `matvec_packed4.go` decode helpers).

## 0. Which kernel actually runs

`projectRows` (`internal/backend/simd/runtime_batch.go:562-580`) is the only dispatch
point for the batched projections. Order of the switch:

1. `useSmallBatchBF16(w,n)` — bf16 weights, `n <= 16` → `MatMatBF16`.
2. `useBatchQuant(w,n)` — quantised weights on AVX2 → `MatMatQuant`.
3. `n > 1 && supportedGemmWeight(w)` → `GemmParWT`.
4. one `MatVec` per row.

`useBatchQuant` (`matmat_quant.go:77-86`) accepts any `DTypeRequiresAligned64` weight
with `C%32==0`, not UQ, on AVX2. For LFM3-1.7B every quantised projection qualifies
(`C` = 2048 or 7680), so **k4 and q8 prefill run 100% through `MatMatQuant`**, including
the 128000×2048 head (via `batchHeadRows`, `runtime_batch.go:590-592`), and k4 never
uses the int8 cache: `quantCacheWantedForLoad` excludes packed 4-bit dtypes
(`quant_cache.go:40-42`), so `batchWeightsPacked(w)` is true and the kernel decodes the
nibble payload itself (`matmat_quant.go:90-92`, `196-231`).

Per layer the batched chunk issues 8 of these matmuls: q, k, v, o (`attn_batch.go:47-49`,
`177`), the attention output gate (`attn_batch.go:145`), up, gate, down
(`attn_batch.go:213-214`, `223`). 24 layers (LFM3-1.7B, per `STATUS.md`) × 8 = 192 per
chunk, × 11 chunks ≈ **2100 `MatMatQuant` calls per 650-token prefill**, plus one head
call.

## 1. Answers to the seven questions

### Q1 — micro-kernel shape, register blocking, arithmetic intensity

The micro-kernel is 2 weight rows × 4 activation rows: `quantDot2x4`
(`matmat_quant.go:341-374`), 8 independent outputs, selected in `matMatRangeQuant`'s
`for ; r+3 < n; r += 4` loop (`matmat_quant.go:234-250`). Tails degrade to
`quantDot2x2` (2×2, `:311-334`) and finally `quantDotRow` (1×1, `:277-297`).

Register blocking: per 32-element block the two weight vectors (`wa`, `we`) and their
absolute values (`aa`, `ae`) are loaded/derived **once** and reused across the four
activation rows (`matmat_quant.go:347-371`). Live 256-bit values in the x0/x1 phase, by
my count: `wa, we, aa, ae, ws, xu, xv, c00, c01, c10, c11, ones, xs, h` = 14, plus the
two accumulators `sum01`, `sum23` = 16. The AVX2 register file is 16 ymm. The code
comment at `matmat_quant.go:335-340` says the two reduce trees "run one after the other
so the live vectors fit in 16 registers", i.e. the author already traded ILP for
registers. There is nothing left for the compiler, so this loop is at (or past) the
spill threshold; the second tree (`sum23`, `:363-371`) shares its temporaries with
`sum01`, forcing a serial dependency between the two halves.

Arithmetic intensity per weight byte (k4, 2 weights/byte, n activation rows:
`4n` flops/byte): n=64 → 256 flops/byte; q8 (1 weight/byte): 128 flops/byte; bf16 via
`GemmParWT`: 64 flops/byte. All three are far above the ~5 flops/byte the box needs to
be compute-bound, so none of these paths can be explained by DRAM bandwidth. The number
that matters is **instructions per MAC**, and it is bad: 55-60 vector instructions per
32-element block (2 weight rows × 4 activation rows = 8 outputs × 32 MACs = 256 MACs),
i.e. **~4.4 MAC per vector instruction** (census in §1-Q1 and finding 1).

*Does the inner loop read weights once per activation row or once per tap?* Neither
exactly: weights are read once **per group of 4 activation rows**. Row pairs are decoded
/ loaded at `matmat_quant.go:225-231` (`rowsAt(i,true)`) and then the `r` loop at `:233`
runs over all n rows, so for n=64 the same weight row pair is re-loaded 16 times per
call — the doc comment at `matmat_quant.go:98-101` ("each weight row is read once and
reused for all n rows while it sits in L1") is true only for n ≤ 4.

### Q2 — are activations quantised per row per call, cost, and reuse across sharing tensors

Yes: `MatMatQuant` calls `getQuantRows(x, n, w.C, batchWeightsPacked(w))`
(`matmat_quant.go:118`), which quantises every one of the n rows
(`matmat_quant.go:35-45`) on the calling goroutine, **before** any work is dispatched.
Nothing is cached across calls: the API takes `x []float32` (`:102`), not a
`*QuantVec`.

The per-token path does cache: `attn.go:124-207` prepares one `QuantVec` and passes it to
the q, k and v matvecs (`project` closure, `attn.go:196-207`), and `ffn.go:28-58` does the
same for gate/up. So the batched path is a **regression** relative to the per-token path
in exactly this respect.

Callers sharing one input row block per chunk per layer: `p.norm` is the source for q, k,
v (`attn_batch.go:47-49`) and the attention gate (`attn_batch.go:145`), then re-normed
and reused for up and gate (`attn_batch.go:213-214`). That is **6 quantisations of the
same 64×2048 block per layer** where 2 would do.

Cost of `quantizeVecBlocksInto` (`matvec_quant.go:671-725`): scalar, per element
`float64` conversion plus `math.Round` plus a clamp (`:712-716`), guarded by an
`if idx >= len(x)` bounds test per element, with three redundant zeroing passes
(`:674-684`) of which the `qx16` one fills a buffer `matmat_quant.go:25` documents as
"unused". Per row it also writes the int16 widened output that nothing reads. n×C
elements per call = 131072 at n=64, C=2048, six times over per layer.

Relative to the matmul: the matmul per call at n=64, R=2048, C=2048 executes roughly
(R/2)×(n/4)×bpr×55 = 1024×16×64×55 ≈ **57.7 M vector instructions**; the quantiser is
~131k element-quantisations ≈ 1.3-2.0 M cycles single-threaded. So the quantiser is a few
percent of one matmul — the problem is that it is *serial*, happens 6× more often than
needed, and is repeated 2100 times per prompt.

### Q3 — allocations on the hot path

- `matmat_quant.go:189` `scratch = &quantKernelScratch{}` — one heap allocation per
  `MatMatQuant` call, but only on the `workers <= 1` path. Not per row, not per pair.
- `matmat_quant.go:35-70` `getQuantRows`/`ensure*Slice` — pooled via `quantRowsPool`
  (`:32`). No per-call `make` in steady state, but `sync.Pool` is drained at every GC, so
  each GC cycle forces a fresh 128 KB `q8` + 128 KB `q8Pair` + 4 KB `q16` + 8 KB
  `pairScales` per pool entry (n=64, C=2048). With ~2100 calls per prompt that is a
  GC-paced allocation stream, not a leak and not free.
- `matmat_bf16.go:211` `tmp := make([]float32, w.R)` — **per call**, on the
  `matMatRangeBF16Fallback` path only (no AVX2 or `RawUint16LE` fails). For the head that
  is 512 KB per call.
- `matmat_bf16.go:84-101` `packActivationGroups` — pooled (`packedActivationPool`), grows
  on demand; buffer is `groups*c8*32` floats = n×C floats, a full copy of the activation
  block per call.
- `gemm_batch.go` / `gemm.go`: none. `packB` is a per-worker persistent buffer
  (`gemm.go:32`), 16 KB.
- `decodePacked4Row`, `packed4RowScales`, `quantDot*`: stack arrays only, no `make`.
- `instance.QuantLayoutForMat` is called **per tile**, not per call
  (`matmat_quant.go:200`), i.e. 128 times per 2048-row matmul; it returns a struct by
  value and does not allocate.

No `make` per row and none per pair anywhere.

### Q4 — weight layout: is the packed nibble payload used directly?

In the batched path, **no**. `batchWeightsPacked` selects the packed branch
(`matmat_quant.go:197-231`): for every weight row pair visited, `rowsAt` calls
`decodePacked4Row` twice and `packed4RowScales` twice, writing an int8 expansion into
`worker.quant.rows` and the pair-order activation (`qr.q8Pair`) is used instead of
`qr.q8`. Only the n=1 matvec path reads the nibbles in place
(`dotBlocksPacked4Int16`, `matvec_packed4.go:33-73`).

Per-call decode work: R rows × bpr blocks; `decodePacked4Row`
(`matvec_packed4.go:147-157`) is ~10 vector ops + one 32 B store per 16 input bytes, so
~640 ops and 2 KB of int8 scratch written per 2048-element row, i.e. **the whole weight
matrix is expanded to int8 once per call** plus `packed4RowScales`' scalar per-block
scale build, which for the k-family does `super * (float32(u6)/32.0)` — a **float
division per block per row** (`matvec_packed4.go:88-96`): 64 divides × 2048 rows = 131k
divides per matmul call.

Size relative to the matmul: ~0.31 vector ops per weight element × 4.2 M elements ≈ 1.3 M
instructions vs 57.7 M for the multiply — **~2.3% at n=64**, but it is a *fixed* cost per
call while the multiply scales with n: **~12% at n=10** (the last chunk of a 650-token
prompt) and **~43% at n=1**.

### Q5 — thread partitioning and per-matmul sync

`MatMatQuant` (`matmat_quant.go:121-131`) takes one `done` channel off
`pool.doneSlots`, sends `workers = min(pool.size, ceil(R/16))` tasks, and blocks on
`workers` replies; `workers` is `GOMAXPROCS` (16 here) for every real width. There is a
**per-matmul channel handshake**, plus the shared dispatcher goroutine
(`matvec.go:138-147`) which round-robins each task to a dedicated worker over a
capacity-1 channel — 16 serialised channel hops per matmul.

Work claiming inside the call is dynamic: tiles of `quantTileRows = 16` weight rows are
handed out by `quantJob.next.Add` (`matmat_quant.go:137`, `150-158`), so a faster P-core
takes more work than an E-core. Granularity is thus *weight rows, 16 at a time*; the
activation-row dimension is never partitioned (each worker sweeps all n rows for each of
its tiles).

Estimated sync cost: 16 sends + 16 receives + 16 dispatcher hops at ~100-300 ns each →
**~5-15 µs per matmul**, ~2100 matmuls → 10-30 ms of a 14.67 s prefill (**~0.2%**). Not a
contributor at n=64; it would matter at n≤4, where a whole matmul is only ~0.9 M
instructions (~0.3 ms).

### Q6 — are 32/32/16 tiles right for widths 2048, 7680, 128000?

They are **not used at all** on this path. `MatMatQuant` never reads `GemmConfig`;
`quantTileRows = 16` (`matmat_quant.go:137`) is the only blocking constant in the kernel,
and `MANTLE_TILE_M/N/K` (`gemm_config.go:38-58`) have no effect on k4/q8 prefill. The
only place tiles apply is `GemmParWT` (bf16/f32), whose `BTransposed` path
(`gemm_batch.go:93-107`) uses `tn=32` columns and `tk=16` (raised to 32 by
`SelectGemmConfig` when k≥192, `gemm_config.go:90-96`).

Which dimension wastes work, argued from the loop structure: the micro-kernel iterates
`for b := range blocks` over the full row (`C/32`, so 64 or 240 blocks) inside the r-loop,
and the r-loop is inside the row-pair loop. So the inner dimension of the trip nest is
**k** (whole row), the middle is **n** (all activation rows), and the outer is **weight
rows (16 per tile)**. Consequences:

- No n-blocking: the activation working set is `n*C` int8 = 128 KB at n=64/C=2048 and
  480 KB for the down projection (C=7680). It is swept once per weight row pair, i.e.
  R/2 = 1024 times per matmul, so the activations are re-read from L2/L3, never held in
  L1 while a tile of weights is applied. Fixing it (16 activation rows resident = 32 KB)
  removes ~460 GB of L2 traffic per prompt; on its own that is worth only ~2%
  (finding 6), but it is a precondition for any faster inner loop, which turns the sweep
  into the bottleneck as soon as the per-element work shrinks.
- The weight dimension is only blocked 16 rows at a time, and within that each row pair
  is re-read n/4 times (Q1). Weight traffic per call = (n/4) × matrix = 16 × 2 MB = 32 MB
  for a 2048² matmul, from L1/L2 rather than DRAM, so it costs bandwidth, not misses.
- Width 7680 or 128000 changes only R, which is the tile-claim direction; nothing about
  the loop structure exploits the wider output. So the 32/32/16 defaults are simply
  irrelevant here, and the "tuned for 256^3" comment (`gemm_config.go:7-8`) describes a
  shard of the code the quantised prefill never enters.

### Q7 — is the batched path taken at n=1..3, and what happens at chunk boundaries

`useBatchQuant` requires only `n >= 1` (`matmat_quant.go:78`), so yes: n=1 takes
`MatMatQuant`. At n=1 the 2×4 loop body never runs; execution falls to
`if r < n { dst[r*ld+i] = quantDotRow(...); dst[r*ld+i+1] = quantDotRow(...) }`
(`matmat_quant.go:261-264`), i.e. one row at a time, weights decoded to int8 first, plus
the full activation pre-pass. That is strictly more work than the plain `MatVec` path,
which reads the packed payload in place (`matvec_packed4.go:114-141`) with no decode.
n=2..3 uses `quantDot2x2` then one `quantDotRow`.

Chunk boundaries: `forEachBatchLogitsEmbeds` (`runtime_batch.go:485-507`) slices the
prompt into `m.MaxBatch` chunks (`m.MaxBatch = cpuMaxBatch()`, `runtime_batch.go:126-127`,
default 64). Each chunk re-enters `forEachRow`, and every matmul in it re-streams and
re-decodes the entire weight matrix. A 650-token prompt is 10×64 + 10, so the last chunk
runs the fixed per-call overheads at n=10 (decode ≈ 12% of kernel work, quantiser
unamortised). The cost of a chunk is therefore almost independent of its n, which is why
`MANTLE_CPU_MAX_BATCH` interacts with the gap (Scenario B's question, but it follows from
the kernel structure).

## 2. Ranked suspected inefficiencies

### 1. The kernel formulation is 2.3x-4x more instructions than the same arithmetic needs, and the bit-exactness contract fixes the floor
Why: per 32-element block per 2×4 tile the kernel executes ~55 vector instructions
before any spill traffic: 9 loads (`matmat_quant.go:347-354`, `360`, `363-364`, `370`), 2
`Abs` (`:349-350`), 8 `MulSign` (`:355-358`, `365-368`), 8
`DotProductPairsSaturated` + 8 `DotProductPairs` via `blockDotInt8` (`:299-301`), 3
nested `ConcatAddPairsGrouped` (`:359`, `:369`), 2 `GetLo`/2 `GetHi`/2 `Add`, 2
`ConvertToFloat32`, 2 `Mul`+2 `Mul`+2 `Add` (`:361`, `:371`), for 256 MACs → **4.4 MAC per
vector instruction**. Three different ceilings apply, and the interesting part is how
close they are to the 4x gap:

- *Same contract, AVX2 only* (unsigned 4-bit weights, one `-8*Σx` correction computed
  once per activation row in `getQuantRows`, per-block scalar dot still reduced every 32
  elements): ~39 instructions per 256 MACs = 6.4 MAC/instr → **1.4x**.
- *Same contract, AVX-VNNI* (add `vpdpbusd`, which fuses the maddubsw+maddwd pair and
  removes the widening vector from the register file): ~31 instructions = 8.3 MAC/instr
  → **1.9x**.
- *Int32 accumulation over a 256-element superblock* (no per-block cross-lane reduce at
  all, one fp fold per 8 blocks): ~14 instructions = 18 MAC/instr → **4.1x**, which lands
  where llama.cpp already is (178.7 tok/s vs 44.3).

The third option breaks `MatMatQuant`'s stated invariant that "every output equals what
MatVec produces for that activation row alone ... the same float32 operations in the same
block order" (`matmat_quant.go:94-101`), because it folds scales for 256 elements instead
of 32. So the exactness contract, not the ISA, is what caps the cheap wins at ~1.9x.
`simd/archsimd` also exposes no byte VNNI dot at all (`go doc simd/archsimd Int32x8` has
`MulSign`/`ConcatAddPairsGrouped`; `Int16x16` only `DotProductPairs`), so option 2 is not
currently expressible.
Exposure: 14.67 s prefill for 650 tokens; MACs = 650 × ~1.3 G layer-matmul parameters
≈ 845 GMAC in 14.67 s = 58 GMAC/s = **~1.3 MAC/cycle/core** at 12 cores / 3.6 GHz, about a
third of what 4.4 MAC/instr allows — so the loop is stall-bound on top of
being instruction-heavy, and the ceilings above are optimistic bounds. Keeping the
contract: **~4-6 s of the 14.67 s recoverable**. Dropping the per-block fold and taking
VNNI: **~10-11 s of the 14.67 s**, i.e. essentially the whole gap.
Evidence: `matmat_quant.go:341-374`; instruction census above; `go doc simd/archsimd
Int32x8` showing no VNNI byte dot; `matmat_quant.go:94-101` for the invariant.

### 2. The horizontal reduce and fp32 scale fold run once per 32-element block, because exactness demands it
Why: `h := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))`
then `h.GetLo().Add(h.GetHi()).ConvertToFloat32()` then
`sum01 = sum01.Add(...Mul(ws.Mul(xs)))` (`matmat_quant.go:359-361`, `369-371`) is executed per
*block*, so a cross-lane permute tree plus an int32→float convert plus an fp32
multiply-fold happens every 32 elements. That is ~15 of the ~55 instructions (27%) and
it is a serial cross-lane dependency chain of ~20+ cycles per block. It is not an
accident: producing the exact bits of `matVecRangeQuantCached` requires the block dot as
a scalar before the fp fold, so the reduce cannot be deferred while the contract holds.
The k-family payload on disk already carries 256-element superblock scales with 6-bit
sub-scales (`packed4RowScales`, `matvec_packed4.go:87-96`) — the structure that would let
you accumulate sub-scale-weighted block dots as exact int32 across 8 blocks
(|dot| ≤ 32×8×127 = 32512, ×63 sub-scale ×8 blocks ≈ 1.6e7, comfortably inside int32) and
fold once per 256 elements. Two things block it today: the kernel expands the 6-bit
sub-scale into a per-32-block fp32 scale (`matvec_packed4.go:95`,
`ws[b] = super * (float32(u6)/32.0)`), and mantle's **activation** scales are per-32
elements (`getQuantRows`, `matmat_quant.go:42-44`), where llama.cpp's `q8_K` carries one
fp scale per 256 plus integer group scales. Uniform 256-element activation scales are the
precondition for the int32 superblock accumulation, and they change the numerics.
Exposure: ~27% of kernel instructions plus the dependency chain → **~4 s of the 14.67 s
if the numerics contract is relaxed**; overlapping finding 1, do not add the two.
Evidence: `matmat_quant.go:359-361`, `:369-371`, `:94-101`; `matvec_packed4.go:87-96`;
`matmat_quant.go:35-45` for the per-32 activation scales.

### 3. Weight sign is folded into the activations once per (row pair, activation row)
Why: `blockDotInt8(aa, xu.MulSign(wa), ones)` (`matmat_quant.go:355-358`, `365-368`)
needs `|w|` unsigned and `x*sign(w)` signed because `DotProductPairsSaturated` is
unsigned×signed. `MulSign` is therefore recomputed for every weight row × activation row
pair — 8 `MulSign` per block per 2×4 tile (~15% of the kernel), and it also chains
`load x → MulSign → maddubsw → maddwd`, lengthening the critical path of every block.
Storing the 4-bit codes unsigned (they are already 0..15 on disk; centring to ±8 happens
in `decodePacked4Row`, `matvec_packed4.go:152-155`) and subtracting `8*Σx` per block —
computed once per activation row in `getQuantRows` — removes all of it.
Exposure: ~15% of kernel instructions plus a shorter critical path per block → **~1.5-2 s
of the 14.67 s** (overlaps finding 1, which counts it inside the 39-instruction AVX2
ceiling; do not add).
Evidence: `matmat_quant.go:299-301`, `:347-371`.

### 4. Activation quantisation is redone per matmul, serially, and 6× per layer more often than needed
Why: `MatMatQuant` takes raw `x []float32` and quantises inside (`matmat_quant.go:102`,
`:118`), so the reuse the per-token path gets from `PrepareQuantVec`
(`attn.go:199-207` caches one `QuantVec` across q/k/v; `ffn.go:43-44` for gate/up) is
unavailable. In the chunked path `p.norm` feeds q, k, v and the attention gate
(`attn_batch.go:47-49`, `145`) and, after re-norming, up and gate
(`attn_batch.go:213-214`): 6 quantisations of one 64×2048 block, when 2 suffice. It also
builds `q8Pair` and `pairScales` every call (`matmat_quant.go:49-70`) — a second full
pass over the activations.
Exposure: n*C = 131072 elements × 6 calls × 24 layers × 11 chunks = 207 M element
quantisations per prompt where 34 M are needed; at ~10-15 cycles/element that is
**0.6-0.9 s of 14.67 s (4-6%)**, all of it single-threaded on the dispatch path, so it
also delays every worker.
Evidence: `matmat_quant.go:35-45`, `:118`; `attn.go:124-207`; `ffn.go:28-58`;
`attn_batch.go:47-49`, `145`, `213-214`.

### 5. Fixed per-call weight work makes short chunks disproportionately expensive
Why: two fixed costs per `MatMatQuant` call, both independent of n: the packed decode
(`decodePacked4Row` + `packed4RowScales` per row pair, `matmat_quant.go:218-223`, called
from `:229-230`) and the
whole-matrix int8 expansion into scratch. At n=64 this is ~2.3% of the multiply, but the
multiply scales with n while the decode does not: ~12% at n=10 (the tail chunk of 650
tokens) and ~43% at n=1, where the kernel degrades to `quantDotRow` per row
(`matmat_quant.go:261-264`) and still pays the decode plus the quantisation pre-pass.
`useBatchQuant` accepting `n >= 1` (`:78`) means a 1-row chunk is *slower* than the
`MatVec` path it replaces.
Exposure: ~8-10% of the cost of the final short chunk, and it makes small
`MANTLE_CPU_MAX_BATCH` settings pay a fixed weight-decode pass per chunk; **~0.2-0.4 s per
650-token prompt** at the default 64 (i.e. ~2%), growing to double-digit percent if the
chunk size is reduced.
Evidence: `matmat_quant.go:196-231`, `:261-264`; `matvec_packed4.go:147-157`, `:87-96`.

### 6. No activation-row blocking: the activation block is re-swept per weight row pair
Why: the loop nest is row pair → all n activation rows → all blocks
(`matmat_quant.go:228-263`). The activation working set is `n*C` int8 (128 KB at n=64,
C=2048; 480 KB for the C=7680 down projection) and is traversed R/2 times (1024 for a
2048-row matmul), so it can never stay in the 32 KB L1: per-weight-pair sweeps land in
L2. Symmetrically the weights of a row pair are re-loaded n/4 times (Q1), contradicting
the doc comment at `matmat_quant.go:98-101`. Per 650-token prefill the activation
re-read volume is roughly (total weight elements / 2) × n = 41 GB per chunk, ~460 GB per
prompt, all of it L2-resident L1 misses.
Exposure: at ~1.4 TB/s aggregate L2 bandwidth that is **~0.3 s of the 14.67 s (~2%)** — real
but small, and it will not be the difference until the instruction count comes down. It
becomes the binding constraint the moment `vpdpbusd` removes the per-element work.
Evidence: `matmat_quant.go:137`, `:150-158`, `:171-272`.

### 7. bf16 prefill decodes the whole weight matrix with a per-element indirect call
Why: with the default chunk of 64 rows, `useSmallBatchBF16` refuses
(`smallBatchMaxRows = 16`, `matmat_bf16.go:13-21`) so bf16 goes through
`GemmParWT` → `packBTileWT` (`gemm_batch.go:189-223`), which decodes every weight element
scalar-wise: two byte loads, a shift and an or per element (`:217-221`) and a call
through a func value, `decode := core.BF16ToFloat32` (`:209`). A call through a func
value cannot be inlined or devirtualised, so that is an indirect call per weight element.
The vectorised widen-once kernel that avoids all of it exists (`MatMatBF16`, bf16→f32 by
`ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()`, `matmat_bf16.go:147-148`) but is
capped at 16 rows — and at n=16 it re-reads the weights once per 4-row group
(`matmat_bf16.go:122-128`), i.e. 4 passes.
Exposure: ~1.3 G weight elements per chunk decoded scalar-ly; at 10-20 cycles/element
over 12 cores that is **0.3-0.6 s per chunk = 3.5-6.5 s of bf16's 58 s prefill
(6-11%)**. It does not explain bf16's 11.2 tok/s on its own; see §4.
Evidence: `gemm_batch.go:189-223`; `matmat_bf16.go:13-21`, `:122-128`, `:143-178`;
`core/dtype.go:72-74`, `:108-140`.

### 8. `quantizeVecBlocksInto` is scalar and does ~40% dead work
Why: `matvec_quant.go:671-725` — scalar per-element `float64` convert, `math.Round`,
clamp (`:712-716`), a per-element `if idx >= len(x)` test, three zeroing passes
(`:674-684`) and a doubling of stores to fill `qx16`, which `matmat_quant.go:25` documents
as unused on this path. Both the round trip through float64 and the int16 output are
avoidable for a batched kernel that only needs int8 + fp32 scale.
Exposure: ~28% of the quantiser's stores plus one of three zero passes are dead; inside
finding 4 that is **~0.1-0.2 s of 14.67 s**.
Evidence: `matvec_quant.go:671-725`; `matmat_quant.go:25`, `:40-45`.

### 9. Per-matmul thread handshake and a serial dispatcher
Why: every `MatMatQuant` acquires a `done` slot, pushes 16 tasks through a shared
capacity-`size*2` channel to a single dispatcher goroutine that round-robins them onto
per-worker capacity-1 channels (`matmat_quant.go:121-131`, `matvec.go:112-155`), then
waits for 16 replies. Workers are counted as `min(pool.size, ceil(R/16))` and are
`GOMAXPROCS` even when the tile count is small; the dispatcher hop serialises dispatch.
Exposure: ~5-15 µs per matmul × ~2100 = **10-30 ms (0.2%)** at n=64; negligible now, but
~2-5% of a matmul once the kernel gets 4x faster, and it is the first thing that will cap
the rewrite at small n.
Evidence: `matmat_quant.go:121-131`; `matvec.go:94-155`; `gemm.go:20-45`.

### 10. Tiling knobs are inert on the quantised path
Why: `MatMatQuant` never reads `GemmConfig`; `quantTileRows` is a package constant
(`matmat_quant.go:137`) and `MANTLE_TILE_M/N/K` only reach `GemmParWT`
(`gemm_config.go:38-58`), which quantised weights never take. Tuning prefill by tile
therefore cannot move the k4/q8 number at all, which will mislead any measurement sweep
that varies them.
Exposure: 0 tok/s directly, but it invalidates a class of experiments and hides the fact
that the only two knobs that matter are chunk size and (via Scenario A/B) which kernel
runs.
Evidence: `matmat_quant.go:102-131`; `gemm_config.go:7-58`; `runtime_batch.go:562-580`.

## 3. What a rewrite must do

0. **Decide the numerics contract first.** `MatMatQuant` currently promises bit-identical
   output to `matVecRangeQuantCached` (`matmat_quant.go:94-101`). Keeping it caps the cheap
   wins at ~1.9x (finding 1). Getting the 4x needs int32 accumulation across a 256-element
   superblock plus one fp32 fold, which changes rounding — and needs the activation
   scales moved from per-32 to per-256 elements (finding 2). Everything else below is
   subordinate to that decision, and it is the one thing a measured rewrite can trade
   away, so it should be an explicit call on the issue, not a side effect.

1. **Use an unsigned-weight × signed-activation dot.** Store/locate 4-bit codes as 0..15,
   compute `Σx` per block once per activation row, and correct with `-8*Σx`. That deletes
   every `MulSign` (finding 3) and is the precondition for `vpdpbusd`-able code. If
   `simd/archsimd` will not expose byte VNNI, say so explicitly and target the AVX2
   `vpmaddubsw + vpmaddwd` floor (~39 instructions per 256 MACs once the sign fold is
   gone) rather than pretending the current 55 is competitive.
2. **Fold scales once per 256 elements, not per 32.** Accumulate sub-scale-weighted block
   dots in int32 across a k4/q4 superblock, then apply `super/32` and the activation scale
   once. k4's on-disk 6-bit sub-scale layout already supports this
   (`matvec_packed4.go:87-96`); the per-block fp32 `wPairs`/`pairScales` interleaving
   (`matmat_quant.go:57-70`, `:231-241`) exists only to serve the per-block fold and
   should go with it.
3. **Shrink the live-vector set.** The 2×4 tile plus two reduce trees needs 16-18 ymm.
   With VNNI there are no widening temporaries, so keep 2 weight rows × 8 activation rows
   with 2-4 int32 accumulators instead of 16 live vectors.
4. **Block activation rows in L1.** Process the activation block in slices of ~16 rows
   (`16×2048 = 32 KB` int8) and sweep many weight row pairs against each slice, so
   weights stream once per slice rather than once per 4 rows, and the activation slice
   stays resident.
5. **Hoist activation quantisation out of the matmul.** Pass a quantised block (the
   `QuantVec`/`quantRows` object) from the chunk, prepared once per normalised block and
   shared by q/k/v/attn-gate, and again by gate/up. Delete the `q8Pair`/`pairScales`
   copies along with the pair-order decode.
6. **Read the packed payload where the register budget allows** and, if a decode is still
   needed, make it a per-call cost paid once per *weight tile* with a vectorised
   conversion, never a per-element indirect call (`gemm_batch.go:209-221`).
7. **Raise or remove `smallBatchMaxRows`** once the bf16 kernel widens weights once per
   call (it currently re-reads them per 4-row group, `matmat_bf16.go:122-128`), and stop
   the packed GEMM from decoding bf16 element-by-element with a func-value call.
8. **Re-derive the equivalence tests** for whatever numerical contract wins: the
   bit-for-bit tests against `matVecRangeQuantCached` exist precisely because this kernel
   was built to be invisible, so a numerics change has to move them to tolerance-based
   comparisons against a reference, and `mantle trace` outputs (per `AGENTS.md`) will
   move with it.

## 4. Confidence and open questions

Confidence: high on the structural facts (loop nest, micro-tile, per-call work, missing
activation reuse, inert tiling knobs, register pressure); medium on the cycle-level
attribution — the instruction census in finding 1 is a lower bound from Go-level SIMD
ops, and the derivation showing ~1.3 MAC/cycle/core (~0.3 IPC) means the loop is
stalled well beyond its issue cost, which static reading cannot explain. The split
between "too many instructions" and "stalls/spills" has to be settled by Scenario A's
pprof and Scenario C's microbenchmarks.

Could not determine statically:

- Whether the compiler actually spills in `quantDot2x4`'s two phases, and
  how `ConcatAddPairsGrouped` lowers (one `vpermd`/`vperm2i128`+`vpaddd`, or more). Both
  change the instruction count materially. A `-gcflags=-S` dump of `quantDot2x4` would
  settle it without running a model.
- The real stall reason: L1/L2 misses per block, load-port saturation, or dependency
  chains through the reduce tree. Needs `perf stat` / pprof on a live prefill.
- What fraction of k4 prefill time the quantised GEMM actually holds (Scenarios A/F), so
  findings 1-3 cannot yet be converted into measured tok/s.
- Why bf16 prefill is 11.2 tok/s when the bf16 path's FLOP roofline and its scalar decode
  together (3.5-6.5 s of the 58 s) do not account for it, and why bf16 prefill is only 1.6x
  its decode rate — that smells like the per-token path rather than `GemmParWT` overhead,
  but the gate in `runtime_batch.go:157-247` does not refuse bf16, so it needs the pprof.
- Whether `MANTLE_CPU_MAX_BATCH` sweeps change which kernel runs (they should not — only
  n changes — but `useSmallBatchBF16`'s 16-row cutoff means chunk sizes ≤16 switch bf16
  to a different kernel, which will show up in Scenario B's sweep as a step, not a slope).
- Actual `GOMAXPROCS` at runtime and the E-core/P-core split's effect on the 16-row tile
  claiming; that is Scenario G's territory.
