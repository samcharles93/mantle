#!/bin/bash
# bench_vnni.sh - Compare VPDPBUSD (AVX-VNNI) vs VPMADDWD (AVX2) performance
# This script runs benchmarks to evaluate the performance impact of AddDotProductQuads integration

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== VPDPBUSD Integration Benchmark Suite ==="
echo ""
echo "Environment:"
echo "  Go version: $(go version)"
echo "  Working directory: $(pwd)"
echo ""

# Create output directory
OUTDIR="${DIR}/benchmark_results"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if we have GOEXPERIMENT=simd support
echo "Checking SIMD support..."
if ! go env GOEXPERIMENT | grep -q simd; then
    echo "Warning: GOEXPERIMENT=simd not enabled, some benchmarks may be skipped"
fi
echo ""

echo "=== Phase 1: Correctness Tests ==="
echo "Running unit tests for VPDPBUSD functions..."
GOEXPERIMENT=simd go test -v -run='TestDotUint8Int8|TestConvertInt8ToUint8|TestMatVecQ4WithVPDPBUSD|TestMatVecK4WithVPDPBUSD' -count=3 2>&1 | tee "$OUTDIR/unit_tests_${TIMESTAMP}.log"
echo ""

echo "=== Phase 2: Microbenchmarks (Dot Products) ==="
echo "Running dot product microbenchmarks..."
echo "This compares dotUint8Int8 (VPDPBUSD) vs dotInt8Int16 (VPMADDWD)"
GOEXPERIMENT=simd go test -bench='BenchmarkDot.*SIMD' -benchmem -benchtime=5s -count=5 2>&1 | tee "$OUTDIR/micro_bench_${TIMESTAMP}.log"
echo ""

echo "=== Phase 3: End-to-End MatVec Benchmarks ==="
echo "Running end-to-end quantized matrix-vector multiplication benchmarks..."
echo "This measures full inference performance with cached quantized data"
GOEXPERIMENT=simd go test -bench='BenchmarkMatVecQ4|BenchmarkMatVecK4' -benchmem -benchtime=5s -count=5 2>&1 | tee "$OUTDIR/e2e_bench_${TIMESTAMP}.log"
echo ""

echo "=== Analysis & Results ==="
echo ""
echo "Results saved to: $OUTDIR"
echo "  Unit tests: unit_tests_${TIMESTAMP}.log"
echo "  Microbenchmarks: micro_bench_${TIMESTAMP}.log"
echo "  End-to-end: e2e_bench_${TIMESTAMP}.log"
echo ""

# Try to run benchstat if available
if command -v benchstat &> /dev/null; then
    echo "=== Benchmark Statistics ==="
    echo ""
    echo "Microbench comparison:"
    benchstat "$OUTDIR/micro_bench_${TIMESTAMP}.log" || true
    echo ""
    echo "MatVec comparison:"
    benchstat "$OUTDIR/e2e_bench_${TIMESTAMP}.log" || true
else
    echo "Note: benchstat not installed. Install with: go install golang.org/x/perf/cmd/benchstat@latest"
    echo "Then run: benchstat $OUTDIR/micro_bench_${TIMESTAMP}.log"
fi

echo ""
echo "=== Decision Criteria ==="
echo ""
echo "Performance gains from VPDPBUSD:"
echo "  > 15% speedup : Keep implementation and optimize further"
echo "  5-15% speedup: Keep implementation but monitor complexity"
echo "  < 5% speedup : May not justify the implementation complexity"
echo "  Regression  : Investigate and potentially abandon"
echo ""
echo "Benchmark suite complete. Check logs for detailed results."
