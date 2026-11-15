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

1. **Fused 4-bit dot product on CPU.** Decode is flat across bit widths (k4 9.52,
   q8 9.33) because both read the int8 cache, and the cache costs 3.6x the file in
   RAM. Removing the cache naively is not the answer (7.3x slower decode), so the
   work is a k4 block-dot kernel on packed payload, which also removes the reason
   for the cache to exist. This single item covers the decode gap, the memory
   inflation and the flat bit-width curve.
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
