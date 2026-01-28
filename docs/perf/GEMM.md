# GEMM Bench Log

Usage

./scripts/benchlog.sh docs/perf/GEMM.md \
  GOEXPERIMENT=simd go1.26rc2 test ./internal/tensor -run=^$ -bench=GemmPar -count=5

./scripts/benchlog.sh docs/perf/GEMM.md \
  GOEXPERIMENT=simd go1.26rc2 test ./internal/tensor -run=^$ -bench=GemmPar -cpuprofile=gemm.out

go1.26rc2 tool pprof -top gemm.out

Use this file to capture GEMM-related performance changes. Keep entries short and consistent.

## Template

```
## YYYY-MM-DD
Change:
- ...

Commands:
GOEXPERIMENT=simd go1.26rc2 test ./internal/tensor -run=^$ -bench=GemmPar -count=5

Results:
- GemmPar: ...
- GemmParSIMD: ...
- GemmParScalar: ...
```
