package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/agent"
	clipaths "github.com/samcharles93/mantle/internal/cli/paths"
	"github.com/samcharles93/mantle/internal/hostcaps"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

func doCmd() *cli.Command {
	var (
		workers     int64
		plan        bool
		interactive bool
		verifier    string
		maxSteps    int64
		workspace   string
		stateDir    string
		temp        float64
		topP        float64
		topK        int64
		tooldefsDir string
	)

	flags := append([]cli.Flag{}, commonModelFlags()...)
	flags = append(flags, commonTokenizerFlags()...)
	flags = append(flags,
		&cli.StringFlag{
			Name:        "tooldefs-dir",
			Usage:       "directory of JSON tool definition files",
			Destination: &tooldefsDir,
		},
		&cli.Int64Flag{
			Name:        "workers",
			Aliases:     []string{"j"},
			Usage:       "number of concurrent agents",
			Value:       1,
			Destination: &workers,
		},
		&cli.BoolFlag{
			Name:        "plan",
			Aliases:     []string{"p"},
			Usage:       "force agent to generate a task plan before execution",
			Destination: &plan,
		},
		&cli.BoolFlag{
			Name:        "interactive",
			Aliases:     []string{"i"},
			Usage:       "require user approval before each tool execution",
			Value:       false,
			Destination: &interactive,
		},
		&cli.StringFlag{
			Name:        "verifier",
			Usage:       "shell command that defines success",
			Destination: &verifier,
		},
		&cli.Int64Flag{
			Name:        "max-steps",
			Usage:       "maximum loop iterations per worker",
			Value:       15,
			Destination: &maxSteps,
		},
		&cli.StringFlag{
			Name:        "workspace",
			Aliases:     []string{"w"},
			Usage:       "working directory for the agent",
			Value:       ".",
			Destination: &workspace,
		},
		&cli.StringFlag{
			Name:        "state-dir",
			Usage:       "path for multi-worker locks and task queues",
			Value:       ".mantle_state",
			Destination: &stateDir,
		},
		&cli.Float64Flag{
			Name:        "temp",
			Aliases:     []string{"t"},
			Usage:       "sampling temperature",
			Value:       0.8,
			Destination: &temp,
		},
		&cli.Float64Flag{
			Name:  "top-p",
			Usage: "top_p sampling parameter",
			Value: 0.95,
			Destination: &topP,
		},
		&cli.Int64Flag{
			Name:        "top-k",
			Usage:       "top_k sampling parameter",
			Value:       40,
			Destination: &topK,
		},
	)

	return &cli.Command{
		Name:      "do",
		Usage:     "Execute autonomous agent tasks and workflows",
		ArgsUsage: "<goal>",
		Flags:     flags,
		Action: func(ctx context.Context, c *cli.Command) error {
			log := logger.FromContext(ctx)

			goal := c.Args().First()
			if goal == "" {
				return cli.Exit("error: goal is required as a positional argument\n\nUsage: mantle do [options] <goal>", 1)
			}

			if workers > 1 && !plan {
				plan = true
			}

			// Apply config file defaults.
			cfg := LoadConfig()
			applyAgentConfig(c, cfg, &tooldefsDir)

			// Resolve tooldefs dir: flag > config > default.
			if tooldefsDir == "" {
				if dir, err := os.UserConfigDir(); err == nil {
					tooldefsDir = filepath.Join(dir, "mantle", "tooldefs")
				}
			}

			// Resolve system prompt template: config file path > built-in default.
			sysTmpl := agent.DefaultSystemPrompt
			if cfg.SystemPromptTpl != "" {
				if data, err := os.ReadFile(cfg.SystemPromptTpl); err == nil {
					sysTmpl = string(data)
				} else {
					log.Warn("could not read system prompt template", "path", cfg.SystemPromptTpl, "err", err)
				}
			}

			// Resolve workspace to an absolute path.
			ws, err := filepath.Abs(workspace)
			if err != nil {
				ws, _ = os.Getwd()
			}

			ctx = hostcaps.WithContext(ctx, hostcaps.Detect())

			resolvedModelPath, err := clipaths.ResolveRunModelPath(modelPath, modelsPath, os.Stdin, os.Stderr)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: resolve model: %v", err), 1)
			}
			modelPath = resolvedModelPath

			log.Info("initializing agent", "model", modelPath, "workers", workers, "interactive", interactive)

			loader := inference.Loader{
				TokenizerJSONPath:   tokenizerJSONPath,
				TokenizerConfigPath: tokenizerConfig,
				ChatTemplatePath:    chatTemplate,
				Backend:             backend,
			}

			loadResult, err := loader.Load(ctx, modelPath, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load mcf model: %v", err), 1)
			}
			defer func() { _ = loadResult.Engine.Close() }()

			if workers == 1 {
				return runSoloAgent(ctx, loadResult, goal, interactive, maxSteps, ws, tooldefsDir, sysTmpl, temp, topP, topK)
			}

			return runSwarm(ctx, loadResult, goal, int(workers), stateDir, verifier, maxSteps)
		},
	}
}

func runSoloAgent(ctx context.Context, loadResult *inference.LoadResult, goal string, interactive bool, maxSteps int64, workspace, tooldefsDir, sysTmpl string, temp, topP float64, topK int64) error {
	log := logger.FromContext(ctx)

	registry := agent.NewRegistry()
	registry.Register(&agent.WriteFileTool{Workspace: workspace})
	registry.Register(&agent.ReadFileTool{Workspace: workspace})
	registry.Register(&agent.ExecuteShellTool{Workspace: workspace})
	registry.Register(&agent.ListDirectoryTool{Workspace: workspace})

	// Load JSON-defined tools from tooldefs directory.
	jsonTools, err := agent.LoadToolsFromDir(workspace, tooldefsDir)
	if err != nil {
		log.Warn("failed to load tooldefs", "dir", tooldefsDir, "err", err)
	}
	for _, t := range jsonTools {
		registry.Register(t)
	}

	// Build contextual system prompt.
	tools := registry.List()
	sysPrompt, err := agent.BuildSystemPrompt(sysTmpl, workspace, tools, int(maxSteps))
	if err != nil {
		log.Warn("failed to render system prompt template, using raw template", "err", err)
		sysPrompt = sysTmpl
	}

	// Print header.
	toolNames := make([]string, len(tools))
	for i, t := range tools {
		toolNames[i] = t.Name()
	}
	fmt.Fprintf(os.Stdout, "\033[1mGoal:\033[0m %s\n", goal)
	fmt.Fprintf(os.Stdout, "\033[90mTools: %s  |  max steps: %d\033[0m\n",
		strings.Join(toolNames, ", "), maxSteps)
	if interactive {
		fmt.Fprintf(os.Stdout, "\033[90mMode: interactive (tool approval required)\033[0m\n")
	}

	l := &agent.Loop{
		Engine:      loadResult.Engine,
		Registry:    registry,
		MaxSteps:    int(maxSteps),
		Interactive: interactive,
		Out:         os.Stdout,
		Temperature: float32(temp),
		TopP:        float32(topP),
		TopK:        int(topK),
	}
	l.Messages = append(l.Messages, tokenizer.Message{
		Role:    "system",
		Content: sysPrompt,
	})

	res, err := l.Run(ctx, goal)
	if err != nil {
		return err
	}

	if !res.Success {
		fmt.Fprintf(os.Stdout, "\n\033[31m✗ %s\033[0m\n", res.Output)
	} else if strings.TrimSpace(res.Output) != "" {
		fmt.Fprintf(os.Stdout, "\n\033[1mOutput:\033[0m\n%s\n", res.Output)
	}

	return nil
}

func runSwarm(ctx context.Context, loadResult *inference.LoadResult, goal string, workers int, stateDir string, verifier string, maxSteps int64) error {
	log := logger.FromContext(ctx)

	manager, err := agent.NewStateManager(stateDir)
	if err != nil {
		return fmt.Errorf("create state manager: %w", err)
	}

	swarm := &agent.Swarm{
		Manager:     manager,
		Engine:      loadResult.Engine,
		Workers:     workers,
		MaxSteps:    int(maxSteps),
		Interactive: false,
		Verifier:    verifier,
	}

	log.Info("swarm started", "goal", goal, "workers", workers)
	return swarm.Run(ctx, goal)
}
