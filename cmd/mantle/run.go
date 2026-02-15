package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/reasoning"
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
		reasoningFmt  string
		reasoningBgt  int64

		// Optional overrides
		messagesJSON string
		toolsJSON    string
		hfConfigFile string

		// Streaming options
		streamMode string

		// Debug flags
		showConfig     bool
		showTokens     bool
		rawOutput      bool
		noSWA          bool
		cudaWeightMode string
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
			Category:    "Sampling",
			Aliases:     []string{"temperature", "t"},
			Usage:       "sampling temperature",
			Value:       0.8,
			Destination: &temp,
		},
		&cli.Int64Flag{
			Name:        "top-k",
			Category:    "Sampling",
			Aliases:     []string{"top_k", "topk"},
			Usage:       "top-k sampling parameter",
			Value:       40,
			Destination: &topK,
		},
		&cli.Float64Flag{
			Name:        "top-p",
			Category:    "Sampling",
			Aliases:     []string{"top_p", "topp"},
			Usage:       "top_p sampling parameter",
			Value:       0.95,
			Destination: &topP,
		},
		&cli.Float64Flag{
			Name:     "min-p",
			Category: "Sampling",
			Aliases:  []string{"min_p", "minp"},
			Usage:    "min_p sampling parameter (0.0 = disabled)",
			Value:    0.05,
		},
		&cli.Float64Flag{
			Name:        "repeat-penalty",
			Category:    "Sampling",
			Aliases:     []string{"repeat_penalty"},
			Usage:       "repetition penalty (1.0 = disabled)",
			Value:       1.1,
			Destination: &repeatPenalty,
		},
		&cli.Int64Flag{
			Name:        "repeat-last-n",
			Category:    "Sampling",
			Aliases:     []string{"repeat_last_n"},
			Usage:       "last n tokens to penalize",
			Value:       64,
			Destination: &repeatLastN,
		},
		&cli.Int64Flag{
			Name:        "seed",
			Category:    "Sampling",
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
			Name:        "reasoning-format",
			Usage:       "reasoning extraction mode: auto, none, deepseek, deepseek-legacy",
			Value:       "auto",
			Destination: &reasoningFmt,
		},
		&cli.Int64Flag{
			Name:        "reasoning-budget",
			Usage:       "reasoning budget control: -1 unrestricted, 0 disable thinking in template when supported",
			Value:       -1,
			Destination: &reasoningBgt,
		},
		&cli.StringFlag{
			Name:    "cache-type-k",
			Aliases: []string{"cache_type_k", "ctk"},
			Usage:   "KV cache data type for K (f32, f16, q8_0)",
			Value:   "f16",
		},
		&cli.StringFlag{
			Name:    "cache-type-v",
			Aliases: []string{"cache_type_v", "ctv"},
			Usage:   "KV cache data type for V (f32, f16, q8_0)",
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
		&cli.StringFlag{
			Name:        "stream-mode",
			Usage:       "streaming output mode: instant, smooth, typewriter, quiet",
			Value:       "smooth",
			Destination: &streamMode,
		},
		// Debug flags
		&cli.BoolFlag{
			Name:        "show-config",
			Category:    "Debugging",
			Usage:       "print model + tokenizer summary",
			Value:       false,
			Destination: &showConfig,
		},
		&cli.BoolFlag{
			Name:        "show-tokens",
			Category:    "Debugging",
			Usage:       "print prompt token ids",
			Value:       false,
			Destination: &showTokens,
		},
		&cli.BoolFlag{
			Name:        "raw-output",
			Category:    "Debugging",
			Usage:       "print escaped output chunks without extra formatting",
			Destination: &rawOutput,
		},
		&cli.BoolFlag{
			Name:        "no-swa",
			Usage:       "disable sliding window attention (force full-size KV cache)",
			Destination: &noSWA,
		},
		&cli.StringFlag{
			Name:        "cuda-weight-mode",
			Usage:       "cuda weight loading mode: auto, quant, dequant",
			Value:       "auto",
			Destination: &cudaWeightMode,
		},
		// Profiling flags
		&cli.StringFlag{
			Name:        "cpuprofile",
			Category:    "Profiling",
			Usage:       "write cpu profile to file",
			Destination: &cpuProfile,
		},
		&cli.StringFlag{
			Name:        "memprofile",
			Category:    "Profiling",
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

			// Apply config file defaults for flags not explicitly set
			cfg := LoadConfig()
			applyRunConfig(c, cfg, &modelsPath, &temp, &topK, &topP, &repeatPenalty,
				&steps, &seed, &streamMode)
			if reasoningBgt != -1 && reasoningBgt != 0 {
				return cli.Exit("error: --reasoning-budget must be -1 or 0", 1)
			}
			switch reasoningFmt {
			case "auto", "none", "deepseek", "deepseek-legacy":
			default:
				return cli.Exit("error: --reasoning-format must be one of: auto, none, deepseek, deepseek-legacy", 1)
			}
			switch strings.ToLower(strings.TrimSpace(cudaWeightMode)) {
			case "auto", "quant", "dequant":
				_ = os.Setenv("MANTLE_CUDA_WEIGHT_MODE", strings.ToLower(strings.TrimSpace(cudaWeightMode)))
			default:
				return cli.Exit("error: --cuda-weight-mode must be one of: auto, quant, dequant", 1)
			}
			// Keep verbose CUDA diagnostics gated behind debug mode.
			if debug && os.Getenv("MANTLE_CUDA_TRACE") == "" {
				_ = os.Setenv("MANTLE_CUDA_TRACE", "1")
			}

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
				DisableSWA:          noSWA,
			}
			loader.LoadOptions.CacheTypeK = c.String("cache-type-k")
			loader.LoadOptions.CacheTypeV = c.String("cache-type-v")
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

			// RoPE Scaling Overrides
			ropeOverride := false
			if c.IsSet("rope-freq-base") {
				modelCfg.Config.RopeFreqBase = c.Float("rope-freq-base")
				ropeOverride = true
			}
			if c.IsSet("rope-scaling") || c.IsSet("rope-scale") || c.IsSet("yarn-orig-ctx") {
				if modelCfg.Config.RopeScaling == nil {
					modelCfg.Config.RopeScaling = &simd.RopeScaling{}
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
				return slices.ContainsFunc(names, c.IsSet)
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
			if cfg.MinP != nil && !c.IsSet("min-p") && !c.IsSet("min_p") && !c.IsSet("minp") {
				minP = *cfg.MinP
			}
			repeatLastNVal := int(repeatLastN)
			stepsVal := int(steps)
			seedVal := seed
			baseOpts := inference.RequestOptions{
				Steps:           &stepsVal,
				Seed:            &seedVal,
				MinP:            &minP,
				RepeatLastN:     &repeatLastNVal,
				NoTemplate:      &noTemplate,
				ReasoningFormat: &reasoningFmt,
			}
			reasoningBudgetVal := int(reasoningBgt)
			baseOpts.ReasoningBudget = &reasoningBudgetVal
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
				if genConfig.RopeFreqBaseLocal != 0 && genConfig.RopeFreqBaseLocal != genConfig.RopeFreqBase {
					log.Info("rope local config", "base", genConfig.RopeFreqBaseLocal)
				}

				if len(genConfig.HeadCountKV) > 0 {
					log.Info("kv heads", "head_count_kv", genConfig.HeadCountKV)
				}

				if runtimeModel, ok := loadResult.Runtime.(*simd.Instance); ok {
					swaModes := make([]string, 0, len(runtimeModel.Layers))
					swaCacheLens := make([]int, 0, len(runtimeModel.Layers))
					for _, layer := range runtimeModel.Layers {
						mode := "full"
						if layer.AttnType == "sliding_attention" && layer.AttnWindow > 0 {
							mode = "sliding"
						}
						swaModes = append(swaModes, mode)
						swaCacheLens = append(swaCacheLens, layer.AttnCache.CacheLen)
					}
					log.Info("swa config",
						"layer_mode", swaModes,
						"cache_len", swaCacheLens,
					)
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
					log.Info("interactive mode enabled", "hint", "type /help for commands, /exit to quit")
				}
			}

			for {
				// If we need input
				if interactive && (len(msgs) == 0 || msgs[len(msgs)-1].Role != "user") {
					input, err := readInteractiveLine("> ")
					if err != nil {
						break
					}
					trimmed := strings.TrimSpace(input)
					if trimmed == "/exit" || trimmed == "/quit" {
						break
					}
					if trimmed == "" {
						continue
					}
					if trimmed == "/help" {
						fmt.Println("Commands:")
						fmt.Println("  /help    - show this help")
						fmt.Println("  /clear   - clear conversation history")
						fmt.Println("  /stats   - show generation statistics")
						fmt.Println("  /exit    - exit interactive mode")
						continue
					}
					if trimmed == "/clear" {
						msgs = msgs[:0]
						if system != "" {
							msgs = append(msgs, tokenizer.Message{Role: "system", Content: system})
						}
						loadResult.Engine.ResetContext()
						fmt.Println("Conversation cleared.")
						continue
					}
					if trimmed == "/stats" {
						fmt.Printf("Messages in context: %d\n", len(msgs))
						var mem runtime.MemStats
						runtime.ReadMemStats(&mem)
						fmt.Printf("Memory: %.1f MB alloc, %.1f MB sys\n",
							float64(mem.Alloc)/(1024*1024),
							float64(mem.Sys)/(1024*1024))
						continue
					}
					if strings.HasPrefix(trimmed, "/") {
						fmt.Printf("Unknown command: %s (type /help for commands)\n", trimmed)
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

				// Create streaming writer based on mode
				mode := StreamMode(streamMode)
				if mode != StreamInstant && mode != StreamSmooth && mode != StreamTypewriter && mode != StreamQuiet {
					log.Warn("invalid stream-mode, using 'smooth'", "provided", streamMode)
					mode = StreamSmooth
				}

				if showConfig {
					log.Info("streaming mode", "mode", mode)
					log.Info("reasoning mode", "format", reasoningFmt, "budget", reasoningBgt)
				}

				writer := NewStreamWriter(mode, rawOutput)
				var split reasoning.Splitter
				reasoningOpen := false
				assistantStarted := false
				result, err := loadResult.Engine.Generate(ctx, &req, func(s string) {
					contentDelta, reasoningDelta := split.Push(s)
					if reasoningDelta != "" && reasoningFmt != "none" {
						if !reasoningOpen {
							fmt.Print("\x1b[2m")
							reasoningOpen = true
						}
						fmt.Print(reasoningDelta)
					}
					if contentDelta != "" {
						if reasoningOpen {
							fmt.Print("\x1b[0m")
							reasoningOpen = false
						}
						if !assistantStarted && !rawOutput {
							fmt.Print("\n")
							assistantStarted = true
						}
						writer.Write(contentDelta)
					}
				})
				if reasoningOpen {
					fmt.Print("\x1b[0m")
				}
				streamedText := writer.Flush()
				if err != nil {
					log.Error("generation failed", "error", err)
					break
				}
				responseText := result.Text
				if responseText == "" {
					responseText = streamedText
				}

				if !rawOutput {
					fmt.Println() // Newline after generation
				}
				log.Info("generation complete",
					"tps", result.Stats.TPS,
					"prompt_tps", result.Stats.PromptTPS,
					"gen_tps", result.Stats.GenerationTPS,
					"prompt_tokens", result.Stats.PromptTokens,
					"tokens", result.Stats.TokensGenerated,
					"duration", result.Stats.Duration,
				)

				msgs = append(msgs, tokenizer.Message{Role: "assistant", Content: responseText})

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

func escapeRawOutput(s string) string {
	if s == "" {
		return ""
	}
	var b strings.Builder
	for _, r := range s {
		switch r {
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\\':
			b.WriteString(`\\`)
		default:
			if strconv.IsPrint(r) {
				b.WriteRune(r)
			} else {
				fmt.Fprintf(&b, `\u%04x`, r)
			}
		}
	}
	return b.String()
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
