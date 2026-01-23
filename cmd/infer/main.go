package main

import (
    "flag"
    "fmt"
    "runtime"
    "strings"
    "time"

    "infer/internal/logits"
    "infer/internal/tensor"
    "infer/internal/toy"
)

// This command provides two modes:
//  -matmul: run a microbenchmark of our GEMM implementation with configurable sizes.
//  -infer: run a simple toy language model and output random characters.
func main() {
    var (
        mode    = flag.String("mode", "infer", "mode: infer or matmul")
        workers = flag.Int("workers", 0, "number of workers for matmul (0 = GOMAXPROCS)")
        seed    = flag.Int64("seed", 1, "RNG seed")

        // matmul parameters
        m  = flag.Int("m", 512, "rows of A and C")
        n  = flag.Int("n", 512, "columns of B and C")
        k  = flag.Int("k", 512, "columns of A / rows of B")
        it = flag.Int("iters", 10, "number of iterations for matmul benchmark")

        // inference parameters
        prompt = flag.String("prompt", "hello", "prompt string (bytes used as tokens)")
        steps  = flag.Int("steps", 32, "number of tokens to generate")
        vocab  = flag.Int("vocab", 256, "vocabulary size for toy model")
        hidden = flag.Int("hidden", 128, "hidden dimension for toy model")
        temp   = flag.Float64("temp", 0.8, "sampling temperature")
        topK   = flag.Int("topk", 40, "top‑k sampling parameter")
        topP   = flag.Float64("topp", 0.95, "top‑p sampling parameter")
    )
    flag.Parse()
    if *workers <= 0 {
        *workers = runtime.GOMAXPROCS(0)
    }
    switch strings.ToLower(*mode) {
    case "matmul":
        runMatMul(*m, *n, *k, *it, *workers)
    case "infer":
        runInfer(*prompt, *steps, *vocab, *hidden, *seed, float32(*temp), *topK, float32(*topP))
    default:
        panic("unknown mode")
    }
}

// runMatMul performs a simple benchmark of the GEMM implementation.  It
// constructs random matrices A and B, multiplies them repeatedly, and
// reports throughput in GFLOP/s.
func runMatMul(m, n, k, iters, workers int) {
    A := tensor.NewMat(m, k)
    B := tensor.NewMat(k, n)
    C := tensor.NewMat(m, n)
    tensor.FillRand(&A, 1)
    tensor.FillRand(&B, 2)
    // Warm up once
    tensor.GemmPar(&C, &A, &B, 1, 0, workers)
    start := time.Now()
    for i := 0; i < iters; i++ {
        tensor.GemmPar(&C, &A, &B, 1, 0, workers)
    }
    elapsed := time.Since(start)
    // Each GEMM does 2*m*n*k floating point operations
    flops := 2.0 * float64(m) * float64(n) * float64(k) * float64(iters)
    gflops := flops / elapsed.Seconds() / 1e9
    fmt.Printf("matmul: C[%dx%d] = A[%dx%d] * B[%dx%d]\n", m, n, m, k, k, n)
    fmt.Printf("iters=%d workers=%d elapsed=%s throughput=%.2f GF/s\n", iters, workers, elapsed, gflops)
}

// runInfer runs the toy language model for a fixed number of steps and prints
// the resulting bytes to standard output.  The prompt string is interpreted
// as bytes; if vocab<256 then bytes are wrapped modulo vocab.
func runInfer(prompt string, steps, vocab, hidden int, seed int64, temp float32, topK int, topP float32) {
    model := toy.NewToyLM(vocab, hidden, seed)
    s := logits.NewSampler(logits.SamplerConfig{
        Seed:        seed,
        Temperature: temp,
        TopK:        topK,
        TopP:        topP,
    })
    // Convert prompt bytes to token indices
    toks := make([]int, 0, len(prompt)+steps)
    for i := 0; i < len(prompt); i++ {
        toks = append(toks, int(prompt[i])%vocab)
    }
    fmt.Print(prompt)
    for i := 0; i < steps; i++ {
        last := toks[len(toks)-1]
        logitsVec := model.Forward(last)
        next := s.Sample(logitsVec, toks)
        toks = append(toks, next)
        // Print as a byte (mod 256 to wrap into ASCII)
        fmt.Printf("%c", byte(next%256))
    }
    fmt.Println()
}
