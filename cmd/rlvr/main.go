package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/urfave/cli/v3"
)

const (
	defaultModel   = "gpt-5.2" // or gpt-4o
	defaultBaseURL = "https://api.openai.com/v1"
	planFileName   = "PLAN.md"
)

func main() {
	cmd := &cli.Command{
		Name:  "rlvr",
		Usage: "Run RLVR (Reinforcement Learning / Verification Loop) on inference kernels",
		Commands: []*cli.Command{
			{
				Name:    "iterate",
				Aliases: []string{"i"},
				Usage:   "Iterate on a target file to improve performance",
				Flags: []cli.Flag{
					&cli.StringFlag{Name: "target", Required: true, Usage: "Target file to optimize (relative to repo root)"},
					&cli.StringFlag{Name: "bench", Required: true, Usage: "Benchmark gate (e.g., ./pkg:BenchmarkName)"},
					&cli.IntFlag{Name: "bench-runs", Value: 7, Usage: "Number of benchmark runs (median used)"},
					&cli.FloatFlag{Name: "min-improvement", Value: 0.02, Usage: "Minimum fractional improvement (0.02 = 2%)"},
					&cli.FloatFlag{Name: "min-sigma", Value: 0, Usage: "Require improvement greater than sigma*stderr (0 disables)"},
					&cli.IntFlag{Name: "iterations", Value: 10, Usage: "Number of iterations to attempt"},
					&cli.BoolFlag{Name: "stop-on-win", Usage: "Stop after first accepted improvement"},
					&cli.BoolFlag{Name: "keep-rejects", Usage: "Keep failed candidates in work dir"},
					&cli.BoolFlag{Name: "rebaseline", Usage: "Force re-computation of baseline"},
					&cli.IntFlag{Name: "retry-apply", Value: 3, Usage: "Total attempts per iteration when apply/gofmt fails (min 1)"},
					// LLM Config
					&cli.StringFlag{Name: "model", Value: defaultModel, Usage: "LLM Model name"},
					&cli.StringFlag{Name: "api-key", Sources: cli.EnvVars("OPENAI_API_KEY"), Usage: "API Key"},
					&cli.StringFlag{Name: "base-url", Value: defaultBaseURL, Sources: cli.EnvVars("OPENAI_BASE_URL"), Usage: "API Base URL"},
					&cli.FloatFlag{Name: "temperature", Value: 0.2, Usage: "Sampling temperature"},
					&cli.StringFlag{Name: "reasoning", Value: "med", Usage: "Reasoning effort level (low, med, high)"},
					&cli.IntFlag{Name: "max-tokens", Value: 4000, Usage: "Max generation tokens"},
				},
				Action: runIterate,
			},
		},
	}

	if err := cmd.Run(context.Background(), os.Args); err != nil {
		log.Fatal(err)
	}
}

func runIterate(ctx context.Context, cmd *cli.Command) error {
	rootDir, err := findRepoRoot()
	if err != nil {
		return fmt.Errorf("root finding failed: %w", err)
	}

	workDir := filepath.Join(rootDir, "work")
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return fmt.Errorf("failed to create work dir: %w", err)
	}
	logDir := filepath.Join(workDir, "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return fmt.Errorf("failed to create log dir: %w", err)
	}

	targetRel := cmd.String("target")
	targetPath := filepath.Join(rootDir, targetRel)
	if _, err := os.Stat(targetPath); os.IsNotExist(err) {
		return fmt.Errorf("target file not found: %s", targetPath)
	}

	apiKey := cmd.String("api-key")
	if apiKey == "" {
		return fmt.Errorf("missing API key (set OPENAI_API_KEY or use --api-key)")
	}

	// 1. Baseline
	baselinePath := filepath.Join(workDir, "baseline.json")
	var base *Baseline

	if !cmd.Bool("rebaseline") {
		base, _ = loadBaseline(baselinePath)
	}

	if base == nil {
		fmt.Println("== Computing Baseline ==")
		if err := runGoTestAll(rootDir); err != nil {
			return fmt.Errorf("baseline tests failed: %w", err)
		}
		benchStats, err := runBench(rootDir, cmd.String("bench"), int(cmd.Int("bench-runs")))
		if err != nil {
			return fmt.Errorf("baseline bench failed: %w", err)
		}
		base = &Baseline{BenchStats: benchStats}
		if err := saveBaseline(baselinePath, base); err != nil {
			return err
		}
		for k, v := range benchStats {
			fmt.Printf("Baseline %s: %.1f ns/op\n", k, v.Median)
		}
	}

	// 2. Client Setup
	client := NewClient(apiKey, cmd.String("base-url"), cmd.String("model"), cmd.Float("temperature"), cmd.String("reasoning"), int(cmd.Int("max-tokens")))

	winsDir := filepath.Join(workDir, "wins")
	if err := os.MkdirAll(winsDir, 0755); err != nil {
		return err
	}

	iterations := int(cmd.Int("iterations"))
	for i := 1; i <= iterations; i++ {
		fmt.Printf("\n== Iteration %d/%d ==\n", i, iterations)

		// A. Collect Context
		promptCtx, err := collectContext(rootDir, targetRel)
		if err != nil {
			return fmt.Errorf("failed collecting context: %w", err)
		}

		// B. Build Prompt
		basePrompt := buildPrompt(promptCtx, targetRel, cmd.String("bench"))
		retries := int(cmd.Int("retry-apply"))
		if retries < 1 {
			retries = 1
		}
		lastReject := ""
		for attempt := 0; attempt < retries; attempt++ {
			prompt := basePrompt
			if attempt > 0 {
				prompt = buildRetryPrompt(basePrompt, attempt, lastReject)
			}

			// C. Call LLM
			fmt.Println(">> Calling Model...")
			change, rawReq, err := client.Call(prompt)
			if err != nil {
				if (errors.Is(err, ErrNoToolCall) || errors.Is(err, ErrBadToolArgs) || errors.Is(err, ErrBadToolParse)) && attempt+1 < retries {
					fmt.Printf(">> Retry requested: %v\n", err)
					lastReject = err.Error()
					continue
				}
				fmt.Printf("!! Model call failed: %v\n", err)
				break
			}

			suffix := ""
			if attempt > 0 {
				suffix = fmt.Sprintf("_retry%d", attempt)
			}
			rawPath := filepath.Join(logDir, fmt.Sprintf("iter_%02d_request%s.json", i, suffix))
			_ = os.WriteFile(rawPath, []byte(rawReq), 0644)

			if change.Summary != "" {
				fmt.Printf(">> Summary: %s\n", change.Summary)
			}
			if change.Reasoning != "" {
				fmt.Printf(">> Reasoning: %s\n", change.Reasoning)
			}
			logEntry := map[string]string{
				"summary_of_changes":   change.Summary,
				"reasoning_for_change": change.Reasoning,
				"diff":                 change.Diff,
				"file":                 change.File,
			}
			if data, err := json.MarshalIndent(logEntry, "", "  "); err == nil {
				logPath := filepath.Join(logDir, fmt.Sprintf("iter_%02d_response%s.json", i, suffix))
				_ = os.WriteFile(logPath, data, 0644)
			}

			if change.Diff != "" {
				patchPath := filepath.Join(logDir, fmt.Sprintf("iter_%02d%s.patch", i, suffix))
				_ = os.WriteFile(patchPath, []byte(change.Diff), 0644)
			} else if change.File != "" {
				filePath := filepath.Join(logDir, fmt.Sprintf("iter_%02d%s.file", i, suffix))
				_ = os.WriteFile(filePath, []byte(change.File), 0644)
			}

			// D. Verify
			// We pass the model payload (diff or full file). The verification step decides if it's a diff or file.
			payload := change.Diff
			if payload == "" {
				payload = change.File
			}
			accepted, patchContent, err := verifyCandidate(cmd, rootDir, targetRel, payload, base, workDir)
			if err != nil {
				var rej *RejectError
				if errors.As(err, &rej) && (rej.Kind == RejectApply || rej.Kind == RejectGofmt) && attempt+1 < retries {
					fmt.Printf(">> Retry requested: %s\n", rej.Error())
					lastReject = rej.Error()
					continue
				}
				fmt.Printf("!! Verification process error: %v\n", err)
				break
			}

			if !accepted {
				break
			}

			fmt.Println(">> Candidate ACCEPTED")

			stamp := time.Now().Format("20060102_150405")
			patchPath := filepath.Join(winsDir, fmt.Sprintf("%s.patch", stamp))
			if err := os.WriteFile(patchPath, []byte(patchContent), 0644); err != nil {
				return fmt.Errorf("failed to save winning patch: %w", err)
			}

			fmt.Println(">> Promoting changes to main repo...")
			// Apply the winning patch to main repo
			if err := applyDiff(rootDir, patchContent); err != nil {
				return fmt.Errorf("failed to promote patch: %w", err)
			}

			newBench, err := runBench(rootDir, cmd.String("bench"), int(cmd.Int("bench-runs")))
			if err == nil {
				base.BenchStats = newBench
				_ = saveBaseline(baselinePath, base)
			}

			if cmd.Bool("stop-on-win") {
				fmt.Println("Stopping on first win.")
				return nil
			}
			break
		}
	}

	return nil
}

const (
	RejectApply RejectKind = "apply"
	RejectGofmt RejectKind = "gofmt"
)

type RejectKind string

type RejectError struct {
	Kind RejectKind
	Err  error
}

func (e *RejectError) Error() string {
	return fmt.Sprintf("%s failed: %v", e.Kind, e.Err)
}

func buildRetryPrompt(base string, attempt int, reason string) string {
	if reason == "" {
		return fmt.Sprintf("%s\n\nPrevious attempt failed to apply or format (retry %d). Return a valid unified diff or complete file via propose_change.", base, attempt)
	}
	return fmt.Sprintf("%s\n\nPrevious attempt failed (%s). Retry %d: return a valid unified diff or complete file via propose_change.", base, reason, attempt)
}

// verifyCandidate creates a temp environment, applies the change, and runs tests.
// It returns (accepted, normalized_patch, error).
func verifyCandidate(cmd *cli.Command, rootDir, targetRel, llmOutput string, base *Baseline, workDir string) (bool, string, error) {
	tempDir, err := os.MkdirTemp("", "rlvr_cand_")
	if err != nil {
		return false, "", err
	}

	defer func() {
		if !cmd.Bool("keep-rejects") {
			os.RemoveAll(tempDir)
		} else {
			rejectsDir := filepath.Join(workDir, "rejects")
			_ = os.MkdirAll(rejectsDir, 0755)
			finalPath := filepath.Join(rejectsDir, filepath.Base(tempDir))
			_ = os.Rename(tempDir, finalPath)
			fmt.Printf(">> Kept reject at: %s\n", finalPath)
		}
	}()

	if err := copyRepo(rootDir, tempDir); err != nil {
		return false, "", fmt.Errorf("copy repo failed: %w", err)
	}

	// --- CHANGE: Use SmartApply ---
	// This will handle full files OR diffs and return the unified diff for our records
	appliedPatch, err := SmartApply(tempDir, targetRel, llmOutput)
	if err != nil {
		fmt.Printf(">> Rejected: Apply failed: %v\n", err)
		return false, "", &RejectError{Kind: RejectApply, Err: err}
	}

	// Go Fmt
	if err := runGoFmt(tempDir, []string{targetRel}); err != nil {
		fmt.Printf(">> Rejected: gofmt failed: %v\n", err)
		return false, "", &RejectError{Kind: RejectGofmt, Err: err}
	}

	// Test
	if err := runGoTestAll(tempDir); err != nil {
		fmt.Printf(">> Rejected: Tests failed\n")
		return false, "", nil
	}

	// Bench
	candBench, err := runBench(tempDir, cmd.String("bench"), int(cmd.Int("bench-runs")))
	if err != nil {
		fmt.Printf(">> Rejected: Benchmarks crashed\n")
		return false, "", nil
	}

	ok, msg := isImproved(candBench, base.BenchStats, cmd.Float("min-improvement"), cmd.Float("min-sigma"))
	fmt.Println(msg)

	if !ok {
		fmt.Println(">> Rejected: Not faster")
		return false, "", nil
	}

	return true, appliedPatch, nil
}
