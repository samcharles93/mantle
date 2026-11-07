# NOTES and MEMORIES

## Notes

- We don't currently parse `conv_bias` or `block_use_swiglu`, if a model were to set these to true, the inference would be wrong because we'd never load the tensor.
- `DefaultOps` methods wrap package-level functions (e.g. `MatVec`) to satisfy the `Ops` interface for backend swapping, while allowing direct internal reuse of the logic (like in `FusedRMSNormMatVecCPU`) without struct instantiation.

## Performance notes

- Quant-cache matvec (k-quants, the decode hot path): on AVX2-only CPUs (no AVX512/VNNI) a `dotInt8Int16` call per 32-element block costs ~100 ns/block — the call plus horizontal reduction dominate — while a fused loop that computes every block's int32 dot of a row in one pass costs ~5 ns/block. Converting each block's dot to float32 and scaling per block in vector lanes (`ConvertToFloat32`+`MulAdd`) was ~30x slower than the int32 path on this CPU. `matVecRangeQuantCached` therefore uses `dotBlocksInt8Int16` (one tight loop, per-block int32 results) and then applies the per-block scales in a scalar combine, in the same order as before, so results are bit-identical. Measured: K4 2048x1024 matvec ~600 µs → ~200 µs; Qwen3-0.6B.k4 CPU decode ~4.5 → ~13-17 tok/s (BF16 model ~15 tok/s). This is the same principle ggml/llama.cpp's `vec_dot` kernels use: `ggml_vec_dot_q4_K_q8_K` (ggml/src/ggml-cpu/arch/x86/quants.c, AVX2) keeps a single `__m256i sumi` int32 accumulator across a whole 256-element K-superblock, applies the per-sub-block scales in the integer domain via shuffles, and does one horizontal reduction per superblock — it never pays a per-32-block call or per-block reduction. The Go/archsimd port had paid exactly that per-32-block call+reduce. With AVX512/VNNI the per-block f32 conversion is cheap and the gap narrows. Coverage: `TestDotBlocksInt8Int16`, `TestMatVecQuantCachedMatchesUncached`, `BenchmarkMatVecK4Cached` in `matvec_quant_test.go`.

## Memories

- cmd/mantle should stay wiring-focused; reusable CLI behavior belongs in internal/cli/* (currently internal/cli/paths and internal/cli/ux).
