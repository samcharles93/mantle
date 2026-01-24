We are building a Go LLM Inference library.
We are building in Pure Go only, no CGO.
No stubs are to be created.
No half-implementation.
No TODOs.
Ensure tests and benchmarks are written for core library kernels.
Ensure that all code that can be used with multiple LLM architectures is in a central package allowing reuse.
Ensure you run `go fmt ./...`, `golangci-lint run` and cross-check and remediate issues.
Ensure builds are successful before marking a task off as complete.
Consider all created files, especially tests, to be permanent artifacts unless the user says otherwise.
