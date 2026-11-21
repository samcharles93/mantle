// Simple example: load an MCF model and generate text from a single prompt.
//
// This shows the minimal integration — open a model, run inference, print
// the result. Use it as a starting point or a smoke test for new models.
//
// Usage:
//
//	go run . -m /path/to/model.mcf -p "Hello world"
//
// All sampling parameters are optional and default to the model's own
// generation_config.json values when available.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/samcharles93/mantle/pkg/mantle"
)

func main() {
	modelPath := flag.String("m", "", "path to .mcf model (required)")
	prompt := flag.String("p", "The capital of France is", "prompt text")
	steps := flag.Int("n", 64, "tokens to generate (-1 = unlimited)")
	backend := flag.String("backend", "auto", "auto, cpu, or cuda")
	temp := flag.Float64("temp", 0.8, "sampling temperature")
	topK := flag.Int("top-k", 40, "top-k sampling")
	topP := flag.Float64("top-p", 0.95, "top-p nucleus threshold")
	minP := flag.Float64("min-p", 0.0, "min-p threshold (0 = disabled)")
	penalty := flag.Float64("repeat-penalty", 1.1, "repetition penalty")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "-m is required")
		flag.Usage()
		os.Exit(1)
	}

	ctx := context.Background()

	engine, err := mantle.NewEngine(ctx, *modelPath, mantle.WithBackend(*backend))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	caps := engine.Capabilities()
	fmt.Printf("arch=%s  ctx=%d  vocab=%d  quant=%s\n",
		caps.Arch, caps.MaxContext, caps.VocabSize, caps.Quantization)

	result, err := engine.Generate(ctx, mantle.Request{
		Prompt:        *prompt,
		Steps:         *steps,
		Temperature:   *temp,
		TopK:          *topK,
		TopP:          *topP,
		MinP:          *minP,
		RepeatPenalty: *penalty,
	}, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n%s\n", result.Text)
	fmt.Printf("--- %d tokens in %v (%.1f t/s) ---\n",
		result.Stats.TokensGenerated, result.Stats.Duration, result.Stats.TPS)
}
