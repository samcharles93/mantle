# TODO

- [X] Structured logging
  - [X] use log/slog for structured logs
  - [X] create a common log interface
  - [X] create pretty logging function with coloured logs for CLI
  - [X] add log output functionality
  - [X] add log levels
  - [testing] add package, file, timestamps to log outputs

- [X] Test coverage
  - [X] Runtime tests
  - [X] Kernel tests

- [ ] Utilities
  - [ ] ensure tools are still useful, deprecate others

- [ ] Documentation
  - [ ] create basic documentation

- [ ] Devops
  - [X] add CI testing (.github/workflows/ci.yml: fmt, build, vet, go test, golangci-lint, cuda-tag vet)
  - [ ] add github actions for releases
  - [ ] add pre-commit checks

- [ ] Model Container Format (MCF)
  - [ ] ensure bit allocation is safe
  - [ ] are we too restrictive in the format?
    - [ ] investigate alternate modelfile binary formats, how do they differ from mcf?

- [ ] Quantisation
  - [X] Improve inference on quantised models
  - [ ] investigate whether quant methods are reliable/safe
  - [ ] ensure MCF docs and spec is updated for quantisation
  - [ ] GSQ/RCO/SLQ design: see pkg/mcf/quantization-design.md — P0 DONE (GGUF reader + Q8_0→DTypeQ8 exact transcode + round-trip tests in pkg/mcf/gguf.go), P1 DONE (`quantize --assignments` + budget verification + QuantInfo provenance), P2 DONE (`DTypeUQ2..UQ8` under MCF v1.1, codec in pkg/mcf/uq.go, CPU decode+matvec, CUDA dequant-mode; SIMD pshufb LUT specialization is a future opt)

- [ ] Architecture Support
  - [X] Gemma3 text support: verified against HF transformers (testdata/gemma3_reference.json, short + past-512 sliding-window prompts; TestGemma3ReferenceLogits with MANTLE_GEMMA3_MCF). Multimodal (vision) variant still open.
  - [X] Granite support: granitemoehybrid (granite-4.0-350M) verified against HF transformers (TestGraniteReferenceLogits with MANTLE_GRANITE_MCF); fused gate+up split + attention/logits/residual/embedding multipliers wired.
  - [ ] Improve ease of use for adding new architectures

