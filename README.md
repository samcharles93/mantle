# Mantle

Mantle is a **model execution substrate**.

It provides a container-first way to **store**, **load**, **inspect**, and **coordinate** execution of machine learning models using the **Model Container Format (MCF)**, without implying runtime behaviour.

## What this repo is

- **Mantle**: runtime substrate and tooling for model execution
- **MCF**: a single-file, random-access container format for model data (raw or quantised), designed to be efficient with memory mapping where available, but not dependent on it

## Quick start

### Requirements
- Go **1.25**
- No CGO, pure Go only

### Build
```bash
go build ./...
````

## CLI usage

The CLI is organised around simple verbs.

### Run a model

```bash
mantle run -m Qwen3/Qwen3-3B-Instruct
# or with an explicit file
mantle run -m /work/models/Qwen3-3B-Instruct.mcf
```

### Serve a model directory

```bash
mantle serve --models-path /work/models
```

### Inspect a container

```bash
mantle inspect -m /work/models/Qwen3-3B-Instruct.mcf
```

### Pack (build an MCF container)

```bash
mantle pack --in /path/to/model --out /work/models/MyModel.mcf
```

> Note: Exact flags may evolve. Keep verbs stable (`run`, `serve`, `inspect`, `pack`).

## Architecture (high level)

### Mantle layers

* **Model store**: loads model containers from disk (MCF)
* **Container reader**: validates structure and exposes random-access views over sections
* **Execution coordination**: hands model bytes to the selected runtime backend and manages lifetimes and I/O modes

### MCF fundamentals

* Single-file container with **absolute offsets**
* **Little-endian** on-disk fields
* **Optional sections**: readers tolerate unknown sections and tolerate missing optional ones
* Designed for **random access**:

  * buffered I/O (`ReadAt`) as a baseline
  * mmap as an acceleration where supported
  * optional direct I/O is a runtime choice (not a format requirement)

### Current section types (indicative)

* Model info and metadata
* Quantisation info
* Tensor index (names, shapes, dtype, offsets)
* Tensor data

## Contributing

* No stubs, no TODOs, no half-implementations
* Tests and benchmarks are required for core kernels and hot paths
* Keep reusable, architecture-agnostic code centralised for reuse across model families
* Before considering work complete:

  * `gofmt ./...` (or `go fmt ./...`)
  * `golangci-lint run`
  * `go test` for the affected packages (and `go test ./...` for shared/format changes)

See `AGENTS.md` for the full set of hard rules and invariants.
