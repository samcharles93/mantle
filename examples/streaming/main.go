// Streaming example: real-time token output via the StreamFunc callback.
//
// The StreamFunc receives StreamChunks as the model generates. Each chunk
// carries a Type (text_delta, reasoning_delta, or prompt_echo) and the
// text Delta. Content and reasoning output are printed with distinct
// styling so you can see what the model is doing as it works.
//
// Usage:
//
//	# Basic streaming
//	go run . -m model.mcf -p "Tell me a story"
//
//	# With reasoning model output
//	go run . -m model.mcf -p "Solve: 2+2" --reasoning
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/samcharles93/mantle/pkg/mantle"
)

var (
	modelPath     = flag.String("m", "", "path to .mcf model (required)")
	prompt        = flag.String("p", "Once upon a time", "prompt text")
	steps         = flag.Int("n", 128, "tokens to generate (-1 = unlimited)")
	backend       = flag.String("backend", "auto", "auto, cpu, or cuda")
	temp          = flag.Float64("temp", 0.8, "sampling temperature")
	topP          = flag.Float64("top-p", 0.95, "top-p nucleus threshold")
	topK          = flag.Int("top-k", 40, "top-k sampling")
	showReasoning = flag.Bool("reasoning", false, "print reasoning/thinking output")
)

func main() {
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
	defer func() {
		if err := engine.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close: %v\n", err)
		}
	}()

	caps := engine.Capabilities()
	fmt.Printf("arch=%s  ctx=%d  vocab=%d  quant=%s\n",
		caps.Arch, caps.MaxContext, caps.VocabSize, caps.Quantization)
	fmt.Printf("Prompt: %s\n\n", *prompt)

	var reasoning strings.Builder

	result, err := engine.Generate(ctx, mantle.Request{
		Prompt:      *prompt,
		Steps:       *steps,
		Temperature: *temp,
		TopK:        *topK,
		TopP:        *topP,
	}, func(chunk mantle.StreamChunk) {
		switch chunk.Type {
		case mantle.StreamChunkTextDelta:
			fmt.Print(chunk.Delta)
		case mantle.StreamChunkReasoningDelta:
			reasoning.WriteString(chunk.Delta)
			if *showReasoning {
				fmt.Printf("\033[2m%s\033[0m", chunk.Delta) // dimmed
			}
		case mantle.StreamChunkPromptEcho:
			// Used internally for echo-prompt mode; ignore here.
		}
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\ngenerate: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n--- %d tokens in %v (%.1f t/s, %d prompt) ---\n",
		result.Stats.TokensGenerated, result.Stats.Duration,
		result.Stats.TPS, result.Stats.PromptTokens)

	if reasoning.Len() > 0 {
		fmt.Printf("\n(reasoning: %d chars)\n", reasoning.Len())
	}
}
