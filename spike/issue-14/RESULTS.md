# Issue 14 — quantised performance and size

Hardware: i5-12500H (4P+8E, 16 threads), 62 GB RAM, RTX 3050 Laptop 4 GB (3792 MiB).
Mantle: `discovery/quant-14` at main `1967a8d` (error-minimising q*/k* encoder) plus the
`--no-quant-cache` load option added in this spike. Model: LFM3-1.7B, 1.73 B parameters.

llama.cpp: commit `f7591a6a0`. The CPU baseline is a CPU-only build
(`/work/tmp/llama-cpu`, no CUDA linked, `GGML_NATIVE=ON`, `-march=native`). The
CUDA numbers use the CUDA build at `build/`.

All mantle figures come from `mantle benchmark`; llama figures from `llama-bench`.
Peak RSS from `/usr/bin/time -v`, peak VRAM from 100 ms `nvidia-smi` sampling.

## CPU

Mantle: 1 warmup + 2 runs per invocation, 3 interleaved repeats, median shown.
llama.cpp: `-ngl 0 -t 16`, 2 inner repeats, 3 interleaved repeats. Prompt 96 tokens,
96 generated. Load average moved between 2 and 15 during the run because another
session was working on the same box, so absolute figures are depressed; the two
engines were measured within seconds of each other in every repeat.

| container     | size MiB | % of bf16 | bpw   | prompt tok/s | gen tok/s | peak RSS MiB |
|---------------|----------|-----------|-------|--------------|-----------|--------------|
| mantle bf16   | 3314     | 100%      | 16.07 | 11.2         | 6.81      | 3568 |
| mantle q8     | 1769     | 53.4%     | 8.58  | 31.7         | 9.33      | 4151 |
| mantle q4     | 945      | 28.5%     | 4.58  | 33.0         | 10.01     | 3328 |
| mantle k4     | 906      | 27.3%     | 4.40  | 34.7         | 9.52      | 3292 |
| llama Q8_0    | 1761     | 53.1%     | 8.54  | 71.7         | 11.19     | 1846 |
| llama Q4_K_M  | 1049     | 31.6%     | 5.08  | 126.4        | 16.36     | 1812 |

Ratios (median against median):

| pair          | prefill | decode | RSS   |
|---------------|---------|--------|-------|
| k4 vs Q4_K_M  | 3.6x slower | 1.7x slower | 1.82x |
| q8 vs Q8_0    | 2.3x slower | 1.20x slower | 2.25x |

Tightly paired runs (one mantle invocation and one `llama-bench` run alternating,
5 repeats each, 64 tokens) agree: k4 prefill 34.5 against 116.7, decode 9.66 against
17.68; q8 prefill 35.2 against 72.8, decode 10.16 against 11.75. The quietest
repeat in that set was mantle 38.1 against llama 154.9 prefill and 10.98 against
22.26 decode, so the 4-bit gap is 3.4-4.0x on prefill and 1.8-2.0x on decode.

## CUDA

Mantle: `--backend cuda --cuda-weight-mode quant`, 1 warmup + 2 runs, 2 repeats.
llama.cpp: `-ngl 999`, 2 inner repeats. 64 prompt tokens, 64 generated.

| config                  | size MiB | prompt tok/s | gen tok/s | peak VRAM MiB | peak RSS MiB |
|-------------------------|----------|--------------|-----------|---------------|--------------|
| mantle k4 (quant)       | 906      | 558          | 37.9      | 3129          | 1866 |
| mantle q4 (quant)       | 945      | 565          | 44.3      | 3175          | 1913 |
| mantle q8 (quant)       | 1769     | 535          | 57.6      | 2343          | 4636 |
| llama Q4_K_M            | 1049     | 3643         | 132.5     | 1266          | 1510 |
| llama Q8_0              | 1761     | 3246         | 88.9      | 1902          | 2222 |

| pair          | prefill | decode | VRAM  |
|---------------|---------|--------|-------|
| k4 vs Q4_K_M  | 6.5x slower | 3.5x slower | 2.47x |
| q8 vs Q8_0    | 6.1x slower | 1.54x slower | 1.23x |

`--cuda-weight-mode dequant` cannot run LFM3-1.7B at all on this card: it wants
3798 MiB for the weights alone with 3680 MiB free. The mode difference was measured
on Qwen3-0.6B k4 instead, where both fit:

| mode    | prompt tok/s | gen tok/s | peak VRAM MiB |
|---------|--------------|-----------|---------------|
| quant   | 422-566      | 61.7-66.3 | 1266 |
| dequant | 920-947      | 53.7-55.0 | 1564 |

So dequant buys 1.7x prefill (cuBLAS f16 GEMM) and loses 1.2x decode and 300 MiB.

## Two VRAM findings that fall out of the CUDA table

- 4-bit does not reduce VRAM on CUDA. k4 holds its raw payload (906 MiB) *and* a
  cached-int8 copy built for the batched prefill GEMM (`ensureBatchQuantMat`,
  ~1.95 GB of int8 plus scales), which is why k4 peaks at 3129 MiB against q8's
  2343 MiB for a file half the size. q8 needs only one copy because its payload is
  already in the cached-int8 layout.
- Mantle's decode ordering is inverted against llama.cpp on both devices: mantle q8
  decodes faster than k4 (57.6 against 37.9 on GPU, 9.33 against 9.52 on CPU), while
  llama Q4_K_M decodes faster than Q8_0 (132.5 against 88.9). The k4 sub-scale
  correction costs more than the bytes it saves.

## The packed 4-bit matvec (`run3/quant-quality`, uncommitted)

The branch build replaces the on-the-fly block decode with a matvec that reads the
packed nibbles straight from the payload (`dotBlocksPacked4Int16`: 16 bytes per
32-element block, sign-extended with `(n ^ 8) - 8` in AVX2 lanes, no int8 cache).
Measured quiet (load average 2.2), 64 tokens generated, three configs rotating order
over 3 repeats, medians:

| config                        | prompt tok/s | gen tok/s | peak RSS MiB |
|-------------------------------|--------------|-----------|--------------|
| branch, `--no-quant-cache`    | 18.31        | 15.01     | 1147 |
| branch, cached                | 44.48        | 12.36     | 3293 |
| llama.cpp Q4_K_M              | 170.18       | 24.21     | 1812 |

| comparison      | decode | prefill | RAM |
|-----------------|--------|---------|-----|
| packed vs cached | 1.21x faster | 2.43x slower | 2.87x less |
| packed vs llama.cpp | 1.61x slower | 9.29x slower | 0.63x of llama |
| cached vs llama.cpp | 1.96x slower | 3.83x slower | 1.82x of llama |

Why prefill falls over: the batched prefill GEMM (`matmat_quant.go`) requires
`w.Quant`, so with the cache disabled it falls back to the per-token matvec. The
packed kernel is therefore a decode and memory win and a prefill regression, and the
prefill regression is the same missing piece as finding 3 below: the batched path
needs to read the packed payload too.

This is the first mantle configuration that beats llama.cpp on memory (1147 MiB
against 1812 MiB for a container 10% smaller than Q4_K_M) and the closest on decode
(1.61x against 1.96x). Output text is identical between the packed and cached paths
on the same binary.

## main@043ab6a: packed 4-bit as the default

`043ab6a` makes the packed weights the default for q4/k4/gq*4, extends `MatMatQuant`
to packed weights so batched prefill no longer needs the int8 cache, and stops
building the cache for 4-bit dtypes. Compared against `b42b720` (the packed kernel
merged but the int8 cache still the default) and llama.cpp, quiet window (load 2.7-3.2),
96 tokens generated, 3 repeats rotating order, medians:

| config                       | prompt tok/s | gen tok/s | peak RSS MiB |
|------------------------------|--------------|-----------|--------------|
| main@043ab6a, default (packed) | 43.70      | 13.71     | 1125 |
| main@b42b720, default (cached) | 40.66      | 11.39     | 3293 |
| llama.cpp Q4_K_M             | 162.25       | 19.20     | 1812 |

| comparison          | decode | prefill | RAM |
|---------------------|--------|---------|-----|
| packed vs cached    | 1.20x faster | 1.07x faster | 2.93x less |
| packed vs llama.cpp | 1.40x slower | 3.71x slower | 0.62x of llama |
| cached vs llama.cpp | 1.69x slower | 3.99x slower | 1.86x of llama |

The switch costs nothing on prefill, which was the open risk in the earlier section:
43.70 against the old cached 40.66 in the same window, and 44.5 for the cached path in
the quietest window measured. Decode gains 20% and resident memory drops 2.93x, to the
point where mantle now uses less RAM than llama.cpp for a smaller container.

Decode medians across three windows on this box: 13.71 (load 3), 12.06 and 12.39
(load 4.5-5). The 15.01 recorded for the uncommitted branch kernel came from a
load-2.2 window; measured in the same window as `043ab6a` the branch kernel is 11.35
against 12.39, so `043ab6a` is the faster of the two implementations and the 15.01 was
a quiet-window figure, not a regression since.

## Parity set at main@043ab6a: throughput against llama.cpp

Containers from `/work/models/mcf/parity/lfm3-1.7b/` (see its MANIFEST.md for
provenance), binary `/work/wt/mantle-parity-lfm3/bin/mantle` (main@043ab6a, CPU
only). llama.cpp CPU-only build. 96 tokens generated, 3 repeats rotating order,
medians. Bits per weight against 1,728,801,536 parameters.

| config      | MiB  | bpw  | prompt tok/s | gen tok/s | peak RSS MiB |
|-------------|------|------|--------------|-----------|--------------|
| k4m         | 1015 | 4.93 | 44.33        | 12.55     | 2064 |
| k4          | 906  | 4.40 | 43.75        | 13.90     | 1139 |
| q8          | 1769 | 8.58 | 46.14        | 12.40     | 4151 |
| llama Q4_K_M | 1049 | 5.09 | 178.7       | 24.6      | 1812 |
| llama Q8_0  | 1761 | 8.54 | 115.2        | 16.2      | 1845 |

| pair          | decode      | prefill     | RSS         |
|---------------|-------------|-------------|-------------|
| k4m vs Q4_K_M | 1.96x slower | 4.04x slower | 1.14x llama |
| k4 vs Q4_K_M  | 1.77x slower | 4.10x slower | 0.63x llama |
| q8 vs Q8_0    | 1.31x slower | 2.50x slower | 2.25x llama |

**k4m costs 2x the resident memory of plain k4** (2064 against 1139 MiB) for a file
only 12% bigger. `mantle inspect --quant` shows why: k4m's mixed recipe leaves 18
tensors at k6 (343.24 MiB, the output head plus v_proj/down_proj), and k6 is not in
the packed path, so those tensors get the int8 cache at load while the 136 k4 tensors
do not. The extra 925 MiB is that k6 payload plus its cache. The packed default
(`043ab6a`) covers q4/k4/gq*4 only, so any mixed recipe that upgrades to k6 pays the
cache again. Extending the packed matvec to k6 would remove it.

Decode also reverses on the mixed container: k4m 12.55 against k4 13.90, for +12% file
size. So the mixed recipe currently costs both speed and memory, and the case for it
is quality alone.

## Quality against llama.cpp's own KL

`mantle compare` gives Mantle's KL against its own bf16 container. llama.cpp measures
its own with `llama-perplexity --kl-divergence`, so the two are separate harnesses.
Both were run over the same `eval.txt`, bf16 as reference:

| harness | file | positions | ref PPL |
|---------|------|-----------|---------|
| `mantle compare --max-tokens 512` | eval.txt | 511 | 15.8971 |
| `llama-perplexity -c 256 --chunks 2` | eval.txt | 512 | 15.1374 |

```
llama-perplexity -m LFM3-1.7B-BF16.gguf -f eval.txt -c 256 --chunks 2 \
  --kl-divergence-base llamacpp-bf16-base.kld
llama-perplexity -m LFM3-1.7B-Q4_K_M.gguf -f eval.txt -c 256 --chunks 2 \
  --kl-divergence --kl-divergence-base llamacpp-bf16-base.kld
```

llama.cpp, per chunk and mean:

| file    | KL chunk 1 | KL chunk 2 | mean KL | same top p |
|---------|------------|------------|---------|------------|
| Q8_0    | 0.00151    | 0.00135    | 0.00143 | 97.8% |
| Q4_K_M  | 0.06663    | 0.06038    | 0.06351 | 88.4% |

Mantle, from the parity MANIFEST (511 positions, mean KL and top-1 agreement):

| container | mean KL | top-1 |
|-----------|---------|-------|
| q8        | 0.00257 | 96.28% |
| k6        | 0.01987 | 93.74% |
| k4m       | 0.09898 | 84.15% |
| k4        | 0.17555 | 76.52% |
| gq4       | 0.23952 | 73.58% |

The two harnesses need a calibration before any 4-bit claim, because they disagree on
a container that is essentially lossless: mantle q8 reads 1.80x llama's Q8_0
(0.00257 against 0.00143) and 1.5 points lower on top-1. Applying that offset:

| container | raw KL | adjusted KL | raw top-1 | adjusted top-1 |
|-----------|--------|-------------|-----------|----------------|
| q8        | 0.00257 | 0.00143 | 96.3% | 97.8% |
| k6        | 0.01987 | 0.01104 | 93.7% | 95.2% |
| k4m       | 0.09898 | 0.05501 | 84.2% | 85.7% |
| k4        | 0.17555 | 0.09757 | 76.5% | 78.0% |
| gq4       | 0.23952 | 0.13312 | 73.6% | 75.1% |

Against llama.cpp Q4_K_M (0.06351 KL, 88.4% top-1) at 5.09 bpw:

- **k4m** at 4.93 bpw: adjusted KL 0.0550 is 0.87x Q4_K_M, adjusted top-1 85.7% is 2.7
  points lower. The two metrics disagree in direction, so the defensible claim is that
  k4m sits in Q4_K_M's quality range at 3% smaller size, not that it wins.
- **k4** at 4.40 bpw: adjusted KL 0.0976 is 1.54x worse and top-1 78.0% is 10.4 points
  lower. Plain k4 is not in Q4_K_M's range even after calibration.

Caveats: different harnesses (utilities of the same weights, different tokenizer
pipelines, 511 against 512 positions, one window against two), so the 1.80x offset is
a correction derived from a single near-lossless pair, not a calibrated constant. The
reference perplexities differ by 5% (15.897 against 15.137), which is the size of the
pipeline difference to expect.

## Question 2 — 4-bit payloads without the int8 cache

`--no-quant-cache` decodes q*/k* blocks during the matvec instead of expanding every
weight to int8 plus an f32 scale at load. Generated text is identical either way
(checked on gemma-3-270m k4 and LFM3-1.7B k4). CPU, LFM3-1.7B, 48 tokens generated:

| config       | prompt tok/s | gen tok/s | peak RSS MiB | vs cached |
|--------------|--------------|-----------|--------------|-----------|
| k4 cached    | 34.7         | 9.52      | 3292 | — |
| k4 no cache  | 1.47         | 1.30      | 1148 | prefill 24x slower, decode 7.3x slower, 2.9x less RAM |
| q8 cached    | 31.7         | 9.33      | 4151 | — |
| q8 no cache  | 1.52         | 1.25      | 2033 | prefill 21x slower, decode 7.5x slower, 2.0x less RAM |

Load time, same runs: k4 2.76 s cached against 0.30 s uncached. Building the cache is
most of load time and all of the 3.6x file-to-RSS inflation.

## Quantize timing, LFM3-1.7B

`/usr/bin/time -v`, same box, idle. "old" is main `b206b97` (rounding encoder), "new"
is `1967a8d` (error-minimising, row-parallel).

| method | wall new | CPU new | wall old | CPU old | output  |
|--------|----------|---------|----------|---------|---------|
| q8     | 22.95 s  | 199.8 s | 26.77 s  | 23.9 s  | 1.73 GiB |
| q4     | 39.92 s  | 412.1 s | 38.97 s  | 29.3 s  | 945 MiB |
| k4     | 42.96 s  | 478.1 s | 96.02 s  | 88.4 s  | 906 MiB |

Peak RSS 3.55 GB for all three. The new encoder is 2.2x faster in wall clock on k4 and
costs 5.4x the CPU because the encode is now split across cores. The issue's
"59 s wall, 11 CPU-min" is 43 s wall, 8 CPU-min here.

## Question 4 — uq* payload arithmetic

Per 256-weight super-block: f16 codebook of 2^b levels (2^(b+1) bytes), 8 block scales
(16 bytes), 32b bytes of b-bit indices.

| dtype | bpw   | codebook bytes | codebook share of payload |
|-------|-------|----------------|---------------------------|
| uq2   | 2.75  | 8              | 9.1%   |
| uq3   | 4.00  | 16             | 12.5%  |
| uq4   | 5.50  | 32             | 18.2%  |
| uq5   | 7.50  | 64             | 26.7%  |
| uq6   | 10.50 | 128            | 38.1%  |
| uq7   | 15.50 | 256            | 51.6%  |
| uq8   | 24.50 | 512            | 65.3%  |

k4 is 4.31 bpw and q4 4.50, so uq4 already costs 28% more payload than k4 to carry one
codebook per 256 weights, and uq8 costs 53% more than bf16 (16.07 bpw). Measured on the
shipped gemma-3-270m containers: `uq4` 224 MB and `uq8` 634 MB against bf16 570 MB.

## A trap in benchmarking llama.cpp on a CUDA box

A CUDA-enabled llama.cpp build with `-ngl 0` keeps the weights on CPU but still
offloads prefill operators to the GPU, because `--no-op-offload` defaults to 0. The
same binary and file measured 582 tok/s prefill with `-ngl 0`, 112 tok/s with
`-ngl 0 -nopo 1`, and 126-158 tok/s in a CPU-only build. VRAM stayed at 248 MiB
(context only) either way, so VRAM sampling does not reveal it. My first prefill
comparison was measured this way and overstated the mantle gap by 4-5x; every number
in this document uses the CPU-only build.

## What to fix first

1. **Fused 4-bit dot product on CPU, and route the batched prefill through it.** The
   packed-nibble matvec on `run3/quant-quality` already proves the decode and memory
   half: 1.21x faster decode than the int8 cache at 2.87x less RAM, beating
   llama.cpp's footprint. What is missing is the batched prefill, which still
   requires `w.Quant`, so prefill drops from 44.5 to 18.3 tok/s when the cache is
   off. Fix that path and the cache stops being needed at all, which also removes
   the 3.6x file-to-RSS inflation.
2. **Stop holding two copies of 4-bit weights in VRAM.** k4 peaks at 3129 MiB for a
   906 MiB file because the raw payload and the int8 prefill cache are both resident.
   Either make the batched prefill read the raw payload like the per-token path does,
   or drop the raw copy and accept the cache for both paths.
3. **Prefill throughput**, 3.6x behind llama.cpp on CPU and 6.5x on GPU. The dequant
   experiment shows mantle's f16 GEMM path is 1.7x faster than its quant GEMM at
   prefill time on GPU, so the quant batched path is the weaker one.
4. **Per-tensor recipes.** k4 at 4.40 bpw beats Q4_K_M's 5.08 on size but loses 1.7x
   on decode and 1.8x on RAM, so matching llama.cpp's quality with a *lower* bpw is
   unlikely to close the speed gap; the recipe work is a quality lever only.
5. **uq5..uq8 should go.** Above uq3 the codebook, not the indices, is most of the
   payload (26.7% at uq5, 65.3% at uq8), and uq8 exceeds bf16. uq8's measured 634 MB
   against bf16's 570 MB is the proof. uq2..uq4 keep a reason to exist only if a
   per-tensor codebook (one per tensor rather than one per 256 weights) is added,
   which is a versioned format change and needs its own proposal.

## Not measured

- Quality per bit per recipe: needs KL at every prompt position, not just the last
  token, which the quant-quality session is building.
- Longer-context behaviour (prefill at 2k-8k tokens) and any KV cache effect.
- gemma-4 E2B, the model that broke in #12.
