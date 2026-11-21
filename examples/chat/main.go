// Chat example: multi-turn conversation using structured messages.
//
// Mantle handles chat templates internally — you provide system/user/assistant
// messages and it renders the correct prompt format for the model's architecture
// (Qwen3, Llama, Gemma, Mistral, etc.).
//
// This example supports interactive mode (no -p flag) for back-and-forth
// conversation, or single-shot mode with -p for one-off queries.
//
// Usage:
//
//	# Single message
//	go run . -m model.mcf -p "What is Go?"
//
//	# With system prompt
//	go run . -m model.mcf -s "You are a poet" -p "Write a haiku"
//
//	# Interactive chat loop
//	go run . -m model.mcf
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/samcharles93/mantle/pkg/mantle"
)

var (
	modelPath = flag.String("m", "", "path to .mcf model (required)")
	prompt    = flag.String("p", "", "user message (empty = interactive mode)")
	system    = flag.String("s", "You are a helpful assistant.", "system prompt")
	steps     = flag.Int("n", 512, "tokens to generate (-1 = unlimited)")
	backend   = flag.String("backend", "auto", "auto, cpu, or cuda")
	temp      = flag.Float64("temp", 0.7, "sampling temperature")
	topP      = flag.Float64("top-p", 0.9, "top-p nucleus threshold")
	topK      = flag.Int("top-k", 40, "top-k sampling")
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
	fmt.Printf("arch=%s  ctx=%d  vocab=%d  quant=%s\n\n",
		caps.Arch, caps.MaxContext, caps.VocabSize, caps.Quantization)

	messages := []mantle.Message{
		{Role: "system", Content: *system},
	}

	if *prompt != "" {
		runOnce(engine, messages, *prompt)
	} else {
		runInteractive(engine, messages)
	}
}

func runOnce(engine *mantle.Engine, msgs []mantle.Message, userMsg string) {
	msgs = append(msgs, mantle.Message{Role: "user", Content: userMsg})

	fmt.Printf("User: %s\n", userMsg)
	result, err := engine.Generate(context.Background(), mantle.Request{
		Messages:    msgs,
		Steps:       *steps,
		Temperature: *temp,
		TopK:        *topK,
		TopP:        *topP,
	}, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Assistant: %s\n", result.Text)
	fmt.Printf("--- %d tokens in %v (%.1f t/s) ---\n",
		result.Stats.TokensGenerated, result.Stats.Duration, result.Stats.TPS)
}

func runInteractive(engine *mantle.Engine, msgs []mantle.Message) {
	fmt.Println("Interactive chat. Type /help for commands, Ctrl-C to exit.")
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		switch input {
		case "":
			continue
		case "/exit", "/quit":
			fmt.Println("goodbye")
			return
		case "/help":
			fmt.Println("/exit  - quit")
			fmt.Println("/clear - reset conversation")
			fmt.Println("/stats - show model capabilities")
			continue
		case "/clear":
			msgs = msgs[:1] // keep system prompt
			engine.ResetContext()
			fmt.Println("conversation reset")
			continue
		case "/stats":
			caps := engine.Capabilities()
			fmt.Printf("arch=%s  ctx=%d  vocab=%d  quant=%s\n",
				caps.Arch, caps.MaxContext, caps.VocabSize, caps.Quantization)
			continue
		}

		msgs = append(msgs, mantle.Message{Role: "user", Content: input})

		result, err := engine.Generate(context.Background(), mantle.Request{
			Messages:    msgs,
			Steps:       *steps,
			Temperature: *temp,
			TopK:        *topK,
			TopP:        *topP,
		}, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
			continue
		}

		fmt.Printf("%s\n", result.Text)
		fmt.Printf("--- %d tokens in %v (%.1f t/s) ---\n",
			result.Stats.TokensGenerated, result.Stats.Duration, result.Stats.TPS)

		msgs = append(msgs, mantle.Message{Role: "assistant", Content: result.Text})
	}
}
