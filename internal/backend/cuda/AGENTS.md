# Repository Guidelines

## Scope

This guide applies only to `internal/backend/cuda` and `internal/backend/cuda/native`. It covers the CUDA backend runtime, CGO bridge, and native kernel integration. SIMD behavior and the parent `internal/backend` selector are out of scope here.

## Project Structure & Module Organization

- [`cuda.go`](/work/apps/mantle/internal/backend/cuda/cuda.go): backend construction, device checks, preload flow, and runtime lifecycle.
- [`ops.go`](/work/apps/mantle/internal/backend/cuda/ops.go): CUDA ops state, device buffers, weight residency, and graph caches.
- [`errors.go`](/work/apps/mantle/internal/backend/cuda/errors.go): panic-to-error conversion for runtime failures.
- [`native/`](/work/apps/mantle/internal/backend/cuda/native): CGO wrappers, CUDA/cuBLAS bindings, runtime helpers, validation tests, and `.cu` kernels.

Keep high-level backend policy in `cuda.go`, reusable device orchestration in `ops.go`, and raw CUDA interop isolated under `native/`.

## Build, Test, and Development Commands

Run commands from the repo root. CUDA work still requires `GOEXPERIMENT=simd`.

- `GOEXPERIMENT=simd go test -tags=cuda ./internal/backend/cuda`
  Runs CUDA package tests.
- `GOEXPERIMENT=simd go test -tags=cuda ./internal/backend/cuda/native`
  Runs CGO/native runtime and validation tests.
- `GOEXPERIMENT=simd go test -tags=cuda ./internal/backend/cuda/...`
  Good package-level sweep after changes.
- `GOEXPERIMENT=simd go fmt ./internal/backend/cuda/...`
  Formats Go files in this package tree.
- `CUDA=1 task build -f`
  Rebuilds the CUDA-enabled binary and native kernels.

Use `MANTLE_CUDA_TRACE=1` only for debugging; do not leave trace-only behavior coupled to normal execution.

## Coding Style & Naming Conventions

Be explicit about ownership and cleanup of streams, handles, graphs, and device buffers. Keep unsafe or CGO-adjacent code localized and validation-first. Name errors and helpers by the failure mode they represent, for example `cudaExecutionError`, `DeviceCount`, and `ManagedFallbackUsed`.

## Testing Guidelines

Put Go tests next to the relevant wrapper or runtime code. Cover cleanup paths, fallback/error reporting, and native validation behavior. For kernel-facing changes, prefer end-to-end output checks over only asserting internal counters.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, commonly `feat:` and `fix:`. Keep CUDA commits scoped to one concern: runtime lifecycle, memory strategy, graphs, or a specific native op. PRs should list CUDA-specific commands run, required env or toolchain assumptions, and any parity, VRAM, or managed-memory impact.
