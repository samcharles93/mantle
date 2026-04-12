# Repository Guidelines

## Scope

This guide applies only to the `internal/backend` package itself. It does not cover implementation details inside `simd/` or `cuda/`; those subpackages should keep their own package-local guides.

## Project Structure & Module Organization

The top-level backend package is the backend-selection seam for Mantle.

- [`backend.go`](/work/apps/mantle/internal/backend/backend.go): public backend names, normalization, and `Backend` interface.
- [`backend_cpu.go`](/work/apps/mantle/internal/backend/backend_cpu.go): non-CUDA build path and fallback behavior.
- [`backend_cuda.go`](/work/apps/mantle/internal/backend/backend_cuda.go): CUDA-enabled build path and package wiring.
- [`bootstrap/`](/work/apps/mantle/internal/backend/bootstrap): shared bootstrap helpers used when a backend needs common runtime loading.

Keep this package thin. It should route to concrete implementations and expose stable backend selection behavior, not absorb model logic or kernel code.

## Build, Test, and Development Commands

Run commands from the repo root with `GOEXPERIMENT=simd`.

- `GOEXPERIMENT=simd go test ./internal/backend`
  Tests the package boundary and backend selection behavior.
- `GOEXPERIMENT=simd go test ./internal/backend/bootstrap`
  Verifies shared bootstrap helpers.
- `GOEXPERIMENT=simd go test ./internal/backend ./internal/backend/bootstrap`
  Good package-level check before broader validation.
- `GOEXPERIMENT=simd go fmt ./internal/backend ./internal/backend/bootstrap`
  Formats only this package and its helper package.

If you change backend selection or load wiring, also verify `GOEXPERIMENT=simd go build -o bin/mantle ./cmd/mantle`.

## Coding Style & Naming Conventions

Prefer explicit names such as `Normalize`, `newCPU`, `newCUDA`, and `LoadSIMDRuntime`. Keep build-tag behavior obvious and localized. Return precise errors for unsupported backend names or unavailable build modes. Do not hide policy in this package; runtime behavior belongs in the selected backend implementation.

## Testing Guidelines

Add package-local tests beside the code they cover. Focus on normalization, backend dispatch, build-tag behavior, and bootstrap contracts. Use standard Go test naming such as `TestNormalizeRejectsUnknownBackend`.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects, usually with `feat:` or `fix:` prefixes. Keep commits scoped to backend selection or bootstrap wiring only. PRs should state which package boundary changed, list the exact Go commands run, and call out any impact on `auto`, `cpu`, or `cuda` backend selection.
