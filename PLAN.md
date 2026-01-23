# RLVR Inference Library: Plan and Test Strategy

## Overview

The goal of this project is to build a **pure Go** inference-only runtime for language models.  The library should
provide high‑performance tensor kernels (e.g. matrix multiplication) and basic model primitives without
relying on CGO or external C libraries.  We will also add a simple *model‑in‑the‑loop* harness so that
future optimisations can be measured and verified automatically.

Key principles:

1. **Inference only** – We do not implement training; weights are either initialised from seeds or loaded from files.
2. **No allocations in hot paths** – Kernel and decode routines should pre‑allocate all buffers.  Allocations are
   tested explicitly using `testing.AllocsPerRun`.
3. **Portable Go** – The code should compile with the standard Go toolchain; optional architecture‑specific
   enhancements can be added later behind build tags.
4. **Test driven** – Every exported function has a corresponding test that asserts correctness before we
   optimise the implementation.

## Repository Structure

```
rlvr/
  PLAN.md               – this file; outlines goals and tests
  go.mod                – Go module definition
  cmd/rlvr/main.go      – sample program with matmul benchmark and toy inference
  internal/
    tensor/
      mat.go            – tensor matrix type and helper functions
      gemm.go           – blocked and parallel matrix multiplication
    logits/
      sampler.go        – top‑k/top‑p/temperature sampler
    toy/
      model.go          – a tiny language model with embedding and linear head
    … (future kernels: softmax, rmsnorm, etc.)
  testdata/             – placeholder for future weight fixtures
```

## Core Goals

1. **Implement a matrix type** with row‑major storage.  A `Mat` has dimensions `(R, C)`, a stride equal to
   its number of columns, and a single backing slice of `float32` values.

2. **Implement a blocked, parallel GEMM** (general matrix multiplication) routine `GemmPar` that computes
   `C = alpha*A*B + beta*C`.  It should split the output rows into chunks and process them concurrently.

3. **Implement a deterministic sampler** supporting top‑k, top‑p and temperature scaling.  The sampler
   exposes a `Sample` method returning the index of the chosen token from a logits vector.

4. **Implement a tiny toy language model** with an embedding matrix and a linear projection head.  The
   `Forward` method computes logits for a single input token.

5. **Provide a small CLI harness** under `cmd/rlvr` that can benchmark the GEMM kernel and run toy inference
   loops for manual inspection.

6. **Write comprehensive tests**:
   - **Matrix tests**: verify that `NewMat` creates matrices with correct sizes and that `Row` returns
     contiguous slices.  Check that `FillRand` produces values in the expected range.
   - **GEMM tests**: implement a naive reference multiplication and compare the output of `GemmPar` against
     the reference for random inputs.  Use a small absolute tolerance (`1e-3`).
   - **Sampler tests**: verify determinism by instantiating two samplers with the same seed and configuration
     and asserting that sampling the same logits yields the same result.  Also test that greedy and
     temperature modifications behave as expected.
   - **Toy model tests**: seed the model, run `Forward` on a fixed token and compare the output to a
     reference computed via the same naive operations.  Ensures the wiring of embedding and linear projection
     is correct.
   - **Allocation tests**: use `AllocsPerRun` to ensure that critical routines do not allocate
     on the heap during their hot loops (e.g. `GemmPar`, `Sampler.Sample`, `ToyLM.Forward`).

These goals provide a starting point for a verified improvement loop: once these tests pass,
optimisations can be attempted (e.g. blocking sizes, SIMD, quantisation) and benchmark results
can be compared to ensure that changes are both correct and faster.

## RLVR kernel loop (external LLM)

The repo also includes a concrete “model proposes, harness verifies” loop in
`scripts/rlvr_loop.py`.

What it does:

1. Reads the current kernel source (e.g. `internal/tensor/gemm.go`) plus nearby tests.
2. Sends that context to an OpenAI‑compatible model and asks for a **unified diff** patch.
3. Applies the patch in a temporary working copy.
4. Runs `gofmt`, `go test ./...`, and a benchmark gate (e.g. `BenchmarkGemmPar`).
5. If the candidate is correct **and** meets a minimum performance improvement threshold,
   the patch is promoted back into the main working tree and becomes the new baseline.

This gives you a practical RLVR loop: the LLM can suggest changes all day long, but the code
only moves forward when the verifier accepts it.

Typical usage:

```bash
export OPENAI_API_KEY=...

# iterate on GEMM
python scripts/rlvr_loop.py iterate \
  --target internal/tensor/gemm.go \
  --bench ./internal/tensor:BenchmarkGemmPar \
  --model gpt-5.2 \
  --api-endpoint chat
```