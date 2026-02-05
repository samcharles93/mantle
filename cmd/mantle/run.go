package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

func runCmd() *cli.Command {
	var (
		prompt        string
		system        string
		steps         int64
		temp          float64
		topK          int64
		topP          float64
		repeatPenalty float64
		repeatLastN   int64
		seed          int64
		noTemplate    bool
		echoPrompt    bool

		// Optional overrides
		messagesJSON string
		toolsJSON    string
		hfConfigFile string

		// Debug flags
		showConfig bool
		showTokens bool
		// Profiling
		cpuProfile string
		memProfile string
	)

	flags := append([]cli.Flag{}, commonModelFlags()...)
	flags = append(flags, commonTokenizerFlags()...)
	flags = append(flags,
		&cli.StringFlag{
			Name:        "prompt",
			Aliases:     []string{"p"},
			Usage:       "prompt text to tokenize",
			Destination: &prompt,
		},
		&cli.StringFlag{
			Name:        "system",
			Aliases:     []string{"sys"},
			Usage:       "optional system prompt",
			Destination: &system,
		},
		&cli.StringFlag{
			Name:        "messages-json",
			Usage:       "path to JSON chat history (array or {\"messages\": [...]})",
			Destination: &messagesJSON,
		},
		&cli.StringFlag{
			Name:        "tools-json",
			Usage:       "path to JSON tools definition (array or {\"tools\": [...]})",
			Destination: &toolsJSON,
		},
		&cli.Int64Flag{
			Name:        "steps",
			Aliases:     []string{"n", "num-tokens", "num_tokens"},
			Usage:       "number of tokens to generate (default -1 = infinite)",
			Value:       -1,
			Destination: &steps,
		},
		&cli.Float64Flag{
			Name:        "temp",
			Aliases:     []string{"temperature", "t"},
			Usage:       "sampling temperature",
			Value:       0.8,
			Destination: &temp,
		},
		&cli.Int64Flag{
			Name:        "top-k",
			Aliases:     []string{"top_k", "topk"},
			Usage:       "top-k sampling parameter",
			Value:       40,
			Destination: &topK,
		},
		&cli.Float64Flag{
			Name:        "top-p",
			Aliases:     []string{"top_p", "topp"},
			Usage:       "top_p sampling parameter",
			Value:       0.95,
			Destination: &topP,
		},
		&cli.Float64Flag{
			Name:    "min-p",
			Aliases: []string{"min_p", "minp"},
			Usage:   "min_p sampling parameter (0.0 = disabled)",
			Value:   0.05,
		},
		&cli.Float64Flag{
			Name:        "repeat-penalty",
			Aliases:     []string{"repeat_penalty"},
			Usage:       "repetition penalty (1.0 = disabled)",
			Value:       1.1,
			Destination: &repeatPenalty,
		},
		&cli.Int64Flag{
			Name:        "repeat-last-n",
			Aliases:     []string{"repeat_last_n"},
			Usage:       "last n tokens to penalize",
			Value:       64,
			Destination: &repeatLastN,
		},
		&cli.Int64Flag{
			Name:        "seed",
			Usage:       "sampling RNG seed (default -1 = random)",
			Value:       -1,
			Destination: &seed,
		},
		&cli.BoolFlag{
			Name:        "no-template",
			Usage:       "disable chat template rendering",
			Destination: &noTemplate,
		},
		&cli.BoolFlag{
			Name:        "echo-prompt",
			Usage:       "print prompt text before generation",
			Destination: &echoPrompt,
		},
		&cli.StringFlag{
			Name:    "cache-type-k",
			Aliases: []string{"cache_type_k", "ctk"},
			Usage:   "KV cache data type for K (f32, f16)",
			Value:   "f16",
		},
		&cli.StringFlag{
			Name:    "cache-type-v",
			Aliases: []string{"cache_type_v", "ctv"},
			Usage:   "KV cache data type for V (f32, f16)",
			Value:   "f16",
		},
		// Optional overrides
		&cli.StringFlag{
			Name:  "rope-scaling",
			Usage: "RoPE scaling type (linear, yarn, none)",
		},
		&cli.Float64Flag{
			Name:  "rope-scale",
			Usage: "RoPE scaling factor",
		},
		&cli.Float64Flag{
			Name:  "rope-freq-base",
			Usage: "RoPE base frequency",
		},
		&cli.Float64Flag{
			Name:  "rope-freq-scale",
			Usage: "RoPE frequency scaling factor",
		},
		&cli.Int64Flag{
			Name:  "yarn-orig-ctx",
			Usage: "YaRN original context size",
		},
		&cli.Float64Flag{
			Name:  "yarn-ext-factor",
			Usage: "YaRN extrapolation mix factor",
			Value: -1.0,
		},
		&cli.Float64Flag{
			Name:  "yarn-attn-factor",
			Usage: "YaRN attention factor",
			Value: -1.0,
		},
		&cli.Float64Flag{
			Name:  "yarn-beta-slow",
			Usage: "YaRN beta slow",
			Value: -1.0,
		},
		&cli.Float64Flag{
			Name:  "yarn-beta-fast",
			Usage: "YaRN beta fast",
			Value: -1.0,
		},
		&cli.StringFlag{
			Name:        "hf-config",
			Usage:       "explicit path to hf config.json",
			Destination: &hfConfigFile,
		},
		// Debug flags
		&cli.BoolFlag{
			Name:        "show-config",
			Usage:       "print model + tokenizer summary",
			Value:       true,
			Destination: &showConfig,
		},
		&cli.BoolFlag{
			Name:        "show-tokens",
			Usage:       "print prompt token ids",
			Value:       true,
			Destination: &showTokens,
		},
		// Profiling flags
		&cli.StringFlag{
			Name:        "cpuprofile",
			Usage:       "write cpu profile to file",
			Destination: &cpuProfile,
		},
		&cli.StringFlag{
			Name:        "memprofile",
			Usage:       "write memory profile to file",
			Destination: &memProfile,
		},
	)

	return &cli.Command{
		Name:  "run",
		Usage: "Run inference for LLM models",
		Flags: flags,
		Action: func(ctx context.Context, c *cli.Command) error {
			log := logger.FromContext(ctx)

			if cpuProfile != "" {
				f, err := os.Create(cpuProfile)
				if err != nil {
					log.Error("could not create CPU profile", "error", err)
					return cli.Exit("could not create CPU profile", 1)
				}
				defer func() { _ = f.Close() }()
				if err := pprof.StartCPUProfile(f); err != nil {
					log.Error("could not start CPU profile", "error", err)
					return cli.Exit("could not start CPU profile", 1)
				}
				defer pprof.StopCPUProfile()
			}

			if memProfile != "" {
				defer func() {
					f, err := os.Create(memProfile)
					if err != nil {
						log.Error("could not create memory profile", "error", err)
						return
					}
					defer func() { _ = f.Close() }()
					if err := pprof.WriteHeapProfile(f); err != nil {
						log.Error("could not write memory profile", "error", err)
					}
				}()
			}

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
				return cli.Exit("error: mantle run only supports .mcf files", 1)
			}

			loadStart := time.Now()

			log.Info("loading MCF model", "path", modelPath)
			loader := inference.Loader{
				TokenizerJSONPath:   tokenizerJSONPath,
				TokenizerConfigPath: tokenizerConfig,
				ChatTemplatePath:    chatTemplate,
				HFConfigPath:        hfConfigFile,
				Backend:             backend,
			}
			loadResult, err := loader.Load(ctx, modelPath, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load mcf model: %v", err), 1)
			}
			defer func() { _ = loadResult.Engine.Close() }()
			runtimeModel := loadResult.Runtime
			modelCfg := runtimeModel.ModelConfig()
			if modelCfg == nil {
				return cli.Exit("error: backend returned nil model config", 1)
			}
			modelCfg.Config.CacheTypeK = c.String("cache-type-k")
			modelCfg.Config.CacheTypeV = c.String("cache-type-v")

			// RoPE Scaling Overrides
			ropeOverride := false
			if c.IsSet("rope-freq-base") {
				modelCfg.Config.RopeFreqBase = c.Float("rope-freq-base")
				ropeOverride = true
			}
			if c.IsSet("rope-scaling") || c.IsSet("rope-scale") || c.IsSet("yarn-orig-ctx") {
				if modelCfg.Config.RopeScaling == nil {
					modelCfg.Config.RopeScaling = &model.RopeScaling{}
				}
				rs := modelCfg.Config.RopeScaling
				if c.IsSet("rope-scaling") {
					rs.Type = c.String("rope-scaling")
				}
				if c.IsSet("rope-scale") {
					rs.Factor = c.Float("rope-scale")
				}
				if c.IsSet("yarn-orig-ctx") {
					rs.OrigMaxCtx = int(c.Int("yarn-orig-ctx"))
				}
				if c.IsSet("yarn-ext-factor") {
					rs.LowFactor = c.Float("yarn-ext-factor")
				}
				if c.IsSet("yarn-attn-factor") {
					rs.AttentionFactor = c.Float("yarn-attn-factor")
				}
				if c.IsSet("yarn-beta-fast") {
					rs.BetaFast = c.Float("yarn-beta-fast")
				}
				if c.IsSet("yarn-beta-slow") {
					rs.BetaSlow = c.Float("yarn-beta-slow")
				}
				ropeOverride = true
			}

			if ropeOverride {
				runtimeModel.UpdateRoPE()
			}

			genConfig := &modelCfg.Config
			tok := loadResult.Tokenizer
			tokConfig := loadResult.TokenizerConfig
			genDefaults := loadResult.GenerationDefaults

			// Resolve effective chat template once for show-config and rendering.
			effectiveTemplate, templateSource := inference.ResolveChatTemplate(chatTemplate, tokConfig, modelCfg.Arch, loadResult.HFConfigJSON)
			isSet := func(names ...string) bool {
				for _, n := range names {
					if c.IsSet(n) {
						return true
					}
				}
				return false
			}
			tempFromGen := !isSet("temp", "temperature", "t") && genDefaults.Temperature != nil && *genDefaults.Temperature > 0
			topKFromGen := !isSet("top-k", "top_k", "topk") && genDefaults.TopK != nil && *genDefaults.TopK > 0
			topPFromGen := !isSet("top-p", "top_p", "topp") && genDefaults.TopP != nil && *genDefaults.TopP > 0 && *genDefaults.TopP <= 1
			repeatFromGen := !isSet("repeat-penalty", "repeat_penalty") && genDefaults.RepetitionPenalty != nil && *genDefaults.RepetitionPenalty > 0
			samplingSource := func(flagSet bool, fromGen bool) string {
				if flagSet {
					return "flag"
				}
				if fromGen {
					return "generation_config"
				}
				return "default"
			}

			loadDuration := time.Since(loadStart)
			log.Info("model loaded", "duration", loadDuration)

			if seed == -1 {
				seed = time.Now().UnixNano()
			}

			minP := c.Float("min-p")
			repeatLastNVal := int(repeatLastN)
			stepsVal := int(steps)
			seedVal := seed
			baseOpts := inference.RequestOptions{
				Steps:       &stepsVal,
				Seed:        &seedVal,
				MinP:        &minP,
				RepeatLastN: &repeatLastNVal,
				NoTemplate:  &noTemplate,
			}
			if isSet("temp", "temperature", "t") {
				baseOpts.Temperature = &temp
			}
			if isSet("top-k", "top_k", "topk") {
				topKVal := int(topK)
				baseOpts.TopK = &topKVal
			}
			if isSet("top-p", "top_p", "topp") {
				baseOpts.TopP = &topP
			}
			if isSet("repeat-penalty", "repeat_penalty") {
				baseOpts.RepeatPenalty = &repeatPenalty
			}

			previewReq := inference.ResolveRequest(baseOpts, genDefaults)
			temp = previewReq.Temperature
			topK = int64(previewReq.TopK)
			topP = previewReq.TopP
			repeatPenalty = previewReq.RepeatPenalty

			if showConfig {
				log.Info("model config",
					"arch", modelCfg.Arch,
					"blocks", genConfig.BlockCount,
					"embd", genConfig.EmbeddingLength,
					"ffn", genConfig.FFNLength,
					"heads", genConfig.HeadCount,
					"head_dim", genConfig.HeadDim,
					"vocab", genConfig.VocabSize,
					"ctx", genConfig.ContextLength,
				)

				if genConfig.RopeScaling != nil {
					log.Info("rope config",
						"base", genConfig.RopeFreqBase,
						"scaling", genConfig.RopeScaling.Type,
						"factor", genConfig.RopeScaling.Factor,
						"orig_ctx", genConfig.RopeScaling.OrigMaxCtx,
						"low", genConfig.RopeScaling.LowFactor,
						"high", genConfig.RopeScaling.HighFactor,
						"attn", genConfig.RopeScaling.AttentionFactor,
					)
				} else {
					log.Info("rope config", "base", genConfig.RopeFreqBase, "scaling", "none")
				}

				if len(genConfig.HeadCountKV) > 0 {
					log.Info("kv heads", "head_count_kv", genConfig.HeadCountKV)
				}

				log.Info("sampling config",
					"temp", temp,
					"temp_source", samplingSource(isSet("temp", "temperature", "t"), tempFromGen),
					"top_k", topK,
					"top_k_source", samplingSource(isSet("top-k", "top_k", "topk"), topKFromGen),
					"top_p", topP,
					"top_p_source", samplingSource(isSet("top-p", "top_p", "topp"), topPFromGen),
					"repeat_penalty", repeatPenalty,
					"repeat_penalty_source", samplingSource(isSet("repeat-penalty", "repeat_penalty"), repeatFromGen),
				)

				if effectiveTemplate == "" {
					log.Info("chat template", "template", "none")
				} else {
					log.Info("chat template", "source", templateSource)
				}
			}

			var (
				msgs  []tokenizer.Message
				tools []any
			)
			if toolsJSON != "" {
				if !fileExists(toolsJSON) {
					return cli.Exit(fmt.Sprintf("error: tools json not found: %s", toolsJSON), 1)
				}
				loaded, err := tokenizer.LoadToolsJSON(toolsJSON)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load tools json: %v", err), 1)
				}
				tools = loaded
			}

			interactive := false
			if messagesJSON != "" {
				if !fileExists(messagesJSON) {
					return cli.Exit(fmt.Sprintf("error: messages json not found: %s", messagesJSON), 1)
				}
				loaded, err := tokenizer.LoadMessagesJSON(messagesJSON)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load messages json: %v", err), 1)
				}
				msgs = loaded
				if prompt != "" || system != "" {
					log.Warn("messages-json overrides prompt/system flags")
				}
			} else {
				msgs = make([]tokenizer.Message, 0, 10)
				if system != "" {
					msgs = append(msgs, tokenizer.Message{Role: "system", Content: system})
				}
				if prompt != "" {
					msgs = append(msgs, tokenizer.Message{Role: "user", Content: prompt})
				} else {
					interactive = true
					log.Info("interactive mode enabled", "hint", "type /exit to quit")
				}
			}

			scanner := bufio.NewScanner(os.Stdin)

			for {
				// If we need input
				if interactive && (len(msgs) == 0 || msgs[len(msgs)-1].Role != "user") {
					fmt.Print("> ")
					if !scanner.Scan() {
						break
					}
					input := scanner.Text()
					if strings.TrimSpace(input) == "/exit" {
						break
					}
					if strings.TrimSpace(input) == "" {
						continue
					}
					msgs = append(msgs, tokenizer.Message{Role: "user", Content: input})
				}

				rendered, err := inference.RenderPrompt(inference.PromptRenderInput{
					TemplateOverride:    chatTemplate,
					TokenizerConfig:     tokConfig,
					Arch:                modelCfg.Arch,
					HFConfigJSON:        loadResult.HFConfigJSON,
					Messages:            msgs,
					Tools:               tools,
					AddGenerationPrompt: true,
					NoTemplate:          noTemplate,
				})
				if err != nil {
					log.Error("render prompt failed", "error", err)
					break
				}

				if showTokens {
					ids, err := tok.Encode(rendered)
					if err != nil {
						log.Error("encode prompt failed", "error", err)
						break
					}
					log.Debug("input tokens", "count", len(ids), "tokens", joinInts(ids))
				}

				echoPromptVal := echoPrompt && !interactive
				opts := baseOpts
				opts.Messages = msgs
				opts.Tools = tools
				opts.EchoPrompt = &echoPromptVal
				req := inference.ResolveRequest(opts, genDefaults)

				var responseBuilder strings.Builder
				result, err := loadResult.Engine.Generate(ctx, &req, func(s string) {
					fmt.Print(s)
					responseBuilder.WriteString(s)
				})
				if err != nil {
					log.Error("generation failed", "error", err)
					break
				}

				fmt.Println() // Newline after generation
				log.Info("generation complete",
					"tps", result.Stats.TPS,
					"tokens", result.Stats.TokensGenerated,
					"duration", result.Stats.Duration,
				)

				// Append assistant response to history
				msgs = append(msgs, tokenizer.Message{Role: "assistant", Content: responseBuilder.String()})

				if !interactive {
					break
				}
			}
			return nil
		},
	}
}

func joinInts(ids []int) string {
	if len(ids) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteByte('[')
	for i, id := range ids {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(fmt.Sprintf("%d", id))
	}
	b.WriteByte(']')
	return b.String()
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
