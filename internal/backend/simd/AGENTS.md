# Repository Guidelines

## Scope

This guide applies only to `internal/backend/simd`. This package owns the CPU backend runtime, SIMD/scalar dispatch, loaders, traces, and CPU-side kernel behavior. It must not reach into `internal/backend/cuda`; shared contracts that both backends need belong in `internal/backend/core`.

## Project Structure & Module Organization

- [`cpu.go`](/work/apps/mantle/internal/backend/simd/cpu.go): CPU backend entrypoint used by the parent backend selector.
- [`loader.go`](/work/apps/mantle/internal/backend/simd/loader.go) and [`load_options.go`](/work/apps/mantle/internal/backend/simd/load_options.go): model loading and package-facing options.
- [`runtime*.go`](/work/apps/mantle/internal/backend/simd/runtime.go): token execution, decode, greedy, and prefill paths.
- [`ops*.go`](/work/apps/mantle/internal/backend/simd/ops.go): CPU op dispatch, AVX2/AVX512 fast paths, and runtime/device helpers.
- Kernel/model files such as `attn.go`, `ffn.go`, `gemm.go`, `matvec*.go`, `moe.go`, and `mamba.go`: math and architecture-specific execution.
- `*_test.go` and `ops_bench_test.go`: correctness, parity, and benchmark coverage.

Keep reusable cross-backend types in `core`, not here.

## Build, Test, and Development Commands

Run commands from the repo root with `GOEXPERIMENT=simd`.

- `GOEXPERIMENT=simd go test ./internal/backend/simd`
  Runs SIMD package tests.
- `GOEXPERIMENT=simd go test ./internal/backend/simd -bench .`
  Runs SIMD benchmarks and tests for hot paths.
- `GOEXPERIMENT=simd go test ./internal/backend/simd ./cmd/mantle`
  Good parity-sensitive check before broader validation.
- `GOEXPERIMENT=simd go fmt ./internal/backend/simd`
  Formats this package.

If shared execution contracts changed, also run `GOEXPERIMENT=simd go test ./...`.

## Coding Style & Naming Conventions

Keep dispatch explicit: scalar fallback, AVX2, and AVX512 paths should be easy to trace. Prefer descriptive runtime names such as `prepareTokenRuntimeState`, `bindDefaultOps`, and `SelectGemmConfigWithTiling`. Do not add CUDA conditionals, CUDA imports, or device-memory assumptions here; CPU behavior belongs here, GPU behavior belongs in `cuda`.

## Boundary Notes

There is current boundary debt: [`core_types.go`](/work/apps/mantle/internal/backend/simd/core_types.go), [`interfaces.go`](/work/apps/mantle/internal/backend/simd/interfaces.go), [`load_options.go`](/work/apps/mantle/internal/backend/simd/load_options.go), and [`core_bridge.go`](/work/apps/mantle/internal/backend/simd/core_bridge.go) expose `core` aliases and bridge methods from inside `simd`. Treat these as transitional shims. Do not extend this pattern; move new shared contracts into `core` directly.

The current scan found no direct `simd` import of `internal/backend/cuda`, which is the intended boundary.

## Testing Guidelines

Put tests beside the code they cover and prefer full-output checks for kernels, runtime steps, and parity-sensitive paths. Use standard Go names such as `TestForwardToken`, `TestFlashAttentionDecode`, and `BenchmarkMatVecQ8`.

## Commit & Pull Request Guidelines

Recent history uses concise imperative subjects, commonly `feat:`, `fix:`, and `chore:`. Keep commits scoped to one execution concern: loader behavior, runtime flow, dispatch, or a specific kernel family. PRs should list exact commands run and call out any effect on scalar fallback, SIMD dispatch, or trace parity.
