# Mantle (Project Agents)

This file sets **non-negotiable rules** and **project scope** for ChatGPT (and humans) working in this repo.

## Project scope

Mantle is **not**:
- an inference engine
- a training or fine-tuning framework
- an orchestration or deployment platform
- a “smart” system that infers behaviour from metadata
- a model conversion pipeline

Mantle **is**:
- a **model execution substrate**
- a **container-first** execution environment centred on **MCF**
- a coordination layer between **on-disk model bytes** and **runtime execution**
- explicitly controlled by the runtime and caller, not implied by container contents

The Model Container Format (MCF) **is**:
- a single-file, random-access container for machine learning model data
- explicit in structure and layout, with absolute offsets and deterministic design
- designed to be efficient with **memory mapping where available**, but **must not require mmap**
- able to store tensor data in raw or quantised form
- forward-compatible via versioned, optional sections (readers tolerate unknown sections and tolerate absence of optional ones)

## Hard constraints (must follow)

### Go and build constraints
- You must ensure you run `go1.26rc2` for any commands that require `go`. `go` commands will not work in this project due to the experimental nature of the simd/archsimd package which this project depends on.
- Prefer the standard library first.
- Use modern Go where it improves clarity or correctness:
  - built-in `min()` / `max()` where appropriate
  - `errors.Is`, `errors.As`, `errors.Join`
  - stdlib `slices`, `maps`, `cmp` where it genuinely improves code
  - prefer `for v := range slice` and `for k, v := range map` when indices are not semantically required
  - use index loops only when the index is genuinely required
  - `for i := range n` is acceptable for fixed-count loops
- Code must be `gofmt`-clean.

### Dependency policy (very strict)

Any new dependency is a design change and must be explicitly requested and justified.

### Implementation discipline
- **No stubs.**
- **No half-implementations.**
- **No TODOs** (or “future work” placeholders) in committed code. unless created by the user.
- All changes must compile and be runnable by default (within repo or users constraints).
- Treat all created files (especially tests) as permanent artefacts unless explicitly instructed otherwise.

### Low-level format and runtime invariants
- All on-disk multi-byte numeric fields are **little-endian**.
- References inside the container are **absolute file offsets** unless a section explicitly defines section-relative offsets.
- No runtime behaviour may be inferred from container contents. The runtime decides behaviour, always.
- Preserve alignment rules (default **8-byte alignment**) unless the spec/version explicitly changes them.
- Parsing must be bounds-checked, overflow-safe, deterministic, and tolerant of unknown optional sections.

### Portability and I/O strategy
- Never assume mmap exists. Readers must support a correct random-access path (eg `io.ReaderAt` / `ReadAt`) with mmap as an optional acceleration.
- Treat direct I/O as optional and platform-specific. If implemented, it must be explicit and handle alignment constraints correctly.
- OS-specific implementations must be isolated behind build tags where appropriate.

### On-disk encoding/decoding rules
- Do not rely on Go struct layout as an on-disk ABI unless the format explicitly mandates it.
- Prefer explicit byte-level encoding/decoding (manual little-endian reads/writes) for on-disk headers and records.
- `unsafe` is allowed only when it materially improves correctness or performance and must be:
  - localised and justified
  - validation-first (validate bounds before unsafe views)
  - explicit about lifetimes (especially mmap-backed slices/strings)

### Architecture and package layout
- Put reusable, architecture-agnostic code in central/shared packages. Avoid duplicating utilities across model implementations.
- Keep model-architecture-specific code clearly separated from generic runtime/container code.
- Determinism matters: do not let map iteration order affect output. Sort where relevant.

### Kernels, tests, and benchmarks
- Core library kernels must have **tests** and **benchmarks**.
- Prefer correctness tests that validate full outputs over piecemeal field checks.
- Benchmarks should measure realistic hot paths and avoid unnecessary allocations in the benchmark harness.

### Prohibited behaviours
- Do not invent new sections, flags, fields, or semantics not present in the repo’s spec/code.
- Do not add training, fine-tuning, orchestration, deployment, or conversion features unless explicitly asked.
- Do not silently introduce breaking on-disk changes. Any format change must be explicit, versioned, and justified.

If a request conflicts with these rules, surface the conflict explicitly and propose a compliant alternative.

## Required workflow before marking work complete

- Run formatting:
  - `gofmt ./...` (or `go fmt ./...` if that is the repo convention)
- Run linting:
  - `golangci-lint run`
- Run tests:
  - package-level tests for the changed area, then `go test ./...` for broader changes
- Ensure builds are successful (including any relevant cross-checks for portability).

If you change anything about MCF structure, section semantics, alignment, or compatibility rules:
- update `model-container-file.md` (and any relevant docs) in the same change.

## CLI naming convention
- Prefer simple verbs for subcommands. `run` is preferred over `infer`.
- Treat MCF as the format name. Do not rename the file extension casually.

## PR checklist (must pass)

### Scope and design
- [ ] Change fits Mantle’s scope (execution substrate; not training/orchestration/conversion)
- [ ] No implicit behaviour added; runtime decisions remain explicit
- [ ] No new on-disk breaking changes without explicit versioning

### Dependencies and build
- [ ] No new third-party dependencies
- [ ] Pure Go only (no CGO)
- [ ] `gofmt`/`go fmt` run on all affected packages
- [ ] `golangci-lint run` passes (or findings fixed)
- [ ] Builds succeed

### Correctness and safety
- [ ] Parsing is bounds-checked and overflow-safe
- [ ] Deterministic output (sorted where relevant; no map-order dependence)
- [ ] No reliance on Go struct layout as an on-disk ABI unless explicitly mandated
- [ ] `unsafe` usage (if any) is localised, justified, and validation-first
- [ ] Lifetimes are clear for any slices/strings referencing backing storage

### Portability and I/O
- [ ] Readers work without mmap (random-access path exists)
- [ ] Any OS-specific code is behind build tags
- [ ] Direct I/O (if touched) is explicit and handles alignment constraints

### Tests and benchmarks
- [ ] Core kernels have tests
- [ ] Core kernels have benchmarks (where performance-critical)
- [ ] `go test` run for affected packages
- [ ] `go test ./...` run if shared code or format semantics changed

### Docs
- [ ] `model-container-file.md` updated if format/sections/compat rules changed
