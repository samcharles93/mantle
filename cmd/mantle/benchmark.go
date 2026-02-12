package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

func benchmarkCmd() *cli.Command {
	var (
		warmupRuns int64
		benchRuns  int64
		prompt     string
		steps      int64
	)

	flags := append([]cli.Flag{}, commonModelFlags()...)
	flags = append(flags,
		&cli.Int64Flag{
			Name:        "warmup",
			Usage:       "number of warmup runs",
			Value:       1,
			Destination: &warmupRuns,
		},
		&cli.Int64Flag{
			Name:        "runs",
			Usage:       "number of benchmark runs",
			Value:       3,
			Destination: &benchRuns,
		},
		&cli.StringFlag{
			Name:        "prompt",
			Aliases:     []string{"p"},
			Usage:       "prompt text for benchmarking",
			Value:       "Explain the theory of relativity in simple terms.",
			Destination: &prompt,
		},
		&cli.Int64Flag{
			Name:        "steps",
			Aliases:     []string{"n"},
			Usage:       "number of tokens to generate per run",
			Value:       128,
			Destination: &steps,
		},
	)

	return &cli.Command{
		Name:  "benchmark",
		Usage: "Run standardized performance benchmarks",
		Flags: flags,
		Action: func(ctx context.Context, cmd *cli.Command) error {
			log := logger.FromContext(ctx)

			resolvedModelPath, err := resolveRunModelPath(modelPath, modelsPath, os.Stdin, os.Stderr)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: resolve model: %v", err), 1)
			}
			modelPath = resolvedModelPath

			stat, err := os.Stat(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: stat model path %q: %v", modelPath, err), 1)
			}
			if stat.IsDir() || !strings.HasSuffix(strings.ToLower(modelPath), ".mcf") {
				return cli.Exit("error: benchmark only supports .mcf files", 1)
			}

			log.Info("loading model for benchmark", "path", modelPath)
			loadStart := time.Now()
			loader := inference.Loader{
				Backend: backend,
			}
			loadResult, err := loader.Load(ctx, modelPath, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load model: %v", err), 1)
			}
			defer func() { _ = loadResult.Engine.Close() }()
			loadDuration := time.Since(loadStart)

			// Print system info
			fmt.Println("=== Mantle Benchmark ===")
			fmt.Printf("Model:    %s (%.1f GB)\n", modelPath, float64(stat.Size())/(1024*1024*1024))
			fmt.Printf("Backend:  %s\n", backend)
			fmt.Printf("CPUs:     %d\n", runtime.NumCPU())
			fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
			fmt.Printf("Load:     %s\n", loadDuration.Round(time.Millisecond))
			fmt.Printf("Steps:    %d tokens\n", steps)
			fmt.Printf("Warmup:   %d runs\n", warmupRuns)
			fmt.Printf("Runs:     %d\n", benchRuns)
			fmt.Println()

			seed := int64(42)
			stepsVal := int(steps)
			noTemplate := false
			echoPrompt := false
			opts := inference.RequestOptions{
				Steps:      &stepsVal,
				Seed:       &seed,
				NoTemplate: &noTemplate,
				EchoPrompt: &echoPrompt,
				Messages: []tokenizer.Message{
					{Role: "user", Content: prompt},
				},
			}
			req := inference.ResolveRequest(opts, loadResult.GenerationDefaults)

			// Warmup
			for i := range int(warmupRuns) {
				log.Info("warmup run", "run", i+1)
				_, err := loadResult.Engine.Generate(ctx, &req, nil)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: warmup run %d: %v", i+1, err), 1)
				}
			}

			// Benchmark runs
			type runResult struct {
				PromptTPS float64
				GenTPS    float64
				TPS       float64
				Duration  time.Duration
				Tokens    int
			}
			results := make([]runResult, 0, benchRuns)

			for i := range int(benchRuns) {
				log.Info("benchmark run", "run", i+1)
				result, err := loadResult.Engine.Generate(ctx, &req, nil)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: benchmark run %d: %v", i+1, err), 1)
				}
				results = append(results, runResult{
					PromptTPS: result.Stats.PromptTPS,
					GenTPS:    result.Stats.GenerationTPS,
					TPS:       result.Stats.TPS,
					Duration:  result.Stats.Duration,
					Tokens:    result.Stats.TokensGenerated,
				})
			}

			// Print results
			fmt.Println("=== Results ===")
			fmt.Printf("%-6s %10s %10s %10s %10s %8s\n", "Run", "Prompt", "Gen", "Total", "Duration", "Tokens")
			fmt.Printf("%-6s %10s %10s %10s %10s %8s\n", "---", "tps", "tps", "tps", "", "")

			var sumPrompt, sumGen, sumTPS float64
			for i, r := range results {
				fmt.Printf("%-6d %10.2f %10.2f %10.2f %10s %8d\n",
					i+1, r.PromptTPS, r.GenTPS, r.TPS, r.Duration.Round(time.Millisecond), r.Tokens)
				sumPrompt += r.PromptTPS
				sumGen += r.GenTPS
				sumTPS += r.TPS
			}

			n := float64(len(results))
			fmt.Printf("\n%-6s %10.2f %10.2f %10.2f\n", "Avg", sumPrompt/n, sumGen/n, sumTPS/n)

			// Memory stats
			var mem runtime.MemStats
			runtime.ReadMemStats(&mem)
			fmt.Printf("\nMemory: %.1f MB alloc, %.1f MB sys\n",
				float64(mem.Alloc)/(1024*1024),
				float64(mem.Sys)/(1024*1024))

			return nil
		},
	}
}
