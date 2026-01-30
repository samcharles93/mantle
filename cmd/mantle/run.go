package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logits"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func runCmd() *cli.Command {
	var (
		modelPath     string
		prompt        string
		system        string
		steps         int64
		temp          float64
		topK          int64
		topP          float64
		repeatPenalty float64
		repeatLastN   int64
		seed          int64
		maxContext    int64
		noTemplate    bool
		echoPrompt    bool

		// Optional overrides
		tokenizerJSON   string
		tokenizerConfig string
		chatTemplate    string
		hfConfigFile    string

		// Debug flags
		showConfig bool
		showTokens bool
		// Profiling
		cpuProfile string
		memProfile string
	)

	return &cli.Command{
		Name:  "run",
		Usage: "Run inference for LLM models",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:        "model",
				Aliases:     []string{"m"},
				Usage:       "path to .mcf file",
				Destination: &modelPath,
			},
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
			&cli.Int64Flag{
				Name:        "max-context",
				Aliases:     []string{"max-ctx", "ctx", "c"},
				Usage:       "max context length",
				Value:       4096,
				Destination: &maxContext,
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
				Name:        "tokenizer-json",
				Usage:       "override path to tokenizer.json",
				Destination: &tokenizerJSON,
			},
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
				Name:        "tokenizer-config",
				Usage:       "override path to tokenizer_config.json",
				Destination: &tokenizerConfig,
			},
			&cli.StringFlag{
				Name:        "chat-template",
				Usage:       "override path to chat_template.jinja",
				Destination: &chatTemplate,
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
		},
		Action: func(ctx context.Context, c *cli.Command) error {
			if cpuProfile != "" {
				f, err := os.Create(cpuProfile)
				if err != nil {
					return cli.Exit(fmt.Sprintf("could not create CPU profile: %v", err), 1)
				}
				defer func() { _ = f.Close() }()
				if err := pprof.StartCPUProfile(f); err != nil {
					return cli.Exit(fmt.Sprintf("could not start CPU profile: %v", err), 1)
				}
				defer pprof.StopCPUProfile()
			}

			if memProfile != "" {
				defer func() {
					f, err := os.Create(memProfile)
					if err != nil {
						fmt.Fprintf(os.Stderr, "could not create memory profile: %v\n", err)
						return
					}
					defer func() { _ = f.Close() }()
					if err := pprof.WriteHeapProfile(f); err != nil {
						fmt.Fprintf(os.Stderr, "could not write memory profile: %v\n", err)
					}
				}()
			}

			resolvedModelPath, err := resolveRunModelPath(modelPath, os.Stdin, os.Stderr)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: resolve model: %v", err), 1)
			}
			modelPath = resolvedModelPath

			stat, err := os.Stat(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: stat model path %q: %v", modelPath, err), 1)
			}

			var (
				m         *model.Instance
				hfTok     *tokenizer.HFTokenizer
				tok       tokenizer.Tokenizer
				tokConfig tokenizer.TokenizerConfig
				genConfig *model.Config
			)

			if stat.IsDir() || !strings.HasSuffix(strings.ToLower(modelPath), ".mcf") {
				return cli.Exit("error: mantle run only supports .mcf files", 1)
			}

			loadStart := time.Now()

			fmt.Printf("Loading MCF model: %s\n", modelPath)

			mcfFile, err := mcfstore.Open(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: open mcf: %v", err), 1)
			}
			defer func() { _ = mcfFile.Close() }()

			cfgBytes := []byte(nil)
			if hfConfigFile != "" {
				cfgBytes, err = os.ReadFile(hfConfigFile)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: read hf config: %v", err), 1)
				}
			} else {
				cfgBytes = mcfFile.SectionData(mcf.SectionHFConfigJSON)
			}
			if len(cfgBytes) == 0 {
				return cli.Exit("error: config.json not found in MCF (use --hf-config to override)", 1)
			}

			m, err = model.LoadModelMCF(mcfFile, cfgBytes, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load mcf model: %v", err), 1)
			}
			m.Config.Config.CacheTypeK = c.String("cache-type-k")
			m.Config.Config.CacheTypeV = c.String("cache-type-v")

			// RoPE Scaling Overrides
			ropeOverride := false
			if c.IsSet("rope-freq-base") {
				m.Config.Config.RopeFreqBase = c.Float("rope-freq-base")
				ropeOverride = true
			}
			if c.IsSet("rope-scaling") || c.IsSet("rope-scale") || c.IsSet("yarn-orig-ctx") {
				if m.Config.Config.RopeScaling == nil {
					m.Config.Config.RopeScaling = &model.RopeScaling{}
				}
				rs := m.Config.Config.RopeScaling
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
				m.UpdateRoPE()
			}

			genConfig = &m.Config.Config

			// Apply generation_config.json defaults when the user did not override flags.
			type hfGenerationConfig struct {
				Temperature       *float64 `json:"temperature"`
				TopK              *int     `json:"top_k"`
				TopP              *float64 `json:"top_p"`
				RepetitionPenalty *float64 `json:"repetition_penalty"`
			}
			var (
				tempFromGen   bool
				topKFromGen   bool
				topPFromGen   bool
				repeatFromGen bool
			)
			isSet := func(names ...string) bool {
				for _, n := range names {
					if c.IsSet(n) {
						return true
					}
				}
				return false
			}
			if genBytes := mcfFile.SectionData(mcf.SectionHFGenerationConfigJSON); len(genBytes) > 0 {
				var hfGen hfGenerationConfig
				if err := json.Unmarshal(genBytes, &hfGen); err != nil {
					fmt.Fprintf(os.Stderr, "warning: parse generation_config.json: %v\n", err)
				} else {
					if !isSet("temp") && hfGen.Temperature != nil && *hfGen.Temperature > 0 {
						temp = *hfGen.Temperature
						tempFromGen = true
					}
					if !isSet("top_k", "topk") && hfGen.TopK != nil && *hfGen.TopK > 0 {
						topK = int64(*hfGen.TopK)
						topKFromGen = true
					}
					if !isSet("top_p", "topp") && hfGen.TopP != nil && *hfGen.TopP > 0 && *hfGen.TopP <= 1 {
						topP = *hfGen.TopP
						topPFromGen = true
					}
					if !isSet("repeat-penalty") && hfGen.RepetitionPenalty != nil && *hfGen.RepetitionPenalty > 0 {
						repeatPenalty = *hfGen.RepetitionPenalty
						repeatFromGen = true
					}
				}
			}

			// Load Tokenizer (Required for MCF)
			if tokenizerJSON != "" && fileExists(tokenizerJSON) {
				tokJSONBytes, err := os.ReadFile(tokenizerJSON)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: read tokenizer.json: %v", err), 1)
				}
				var tokCfgBytes []byte
				if tokenizerConfig != "" && fileExists(tokenizerConfig) {
					tokCfgBytes, err = os.ReadFile(tokenizerConfig)
					if err != nil {
						return cli.Exit(fmt.Sprintf("error: load tokenizer_config.json: %v", err), 1)
					}
				}
				hfTok, err = tokenizer.LoadHFTokenizerBytes(tokJSONBytes, tokCfgBytes)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load tokenizer.json: %v", err), 1)
				}
				if parsedCfg, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSONBytes, tokCfgBytes); err == nil {
					tokConfig = parsedCfg
				} else {
					fmt.Fprintf(os.Stderr, "warning: parse tokenizer_config.json: %v\n", err)
				}
			} else {
				tokJSON := mcfFile.SectionData(mcf.SectionTokenizerJSON)
				if len(tokJSON) == 0 {
					return cli.Exit("error: tokenizer.json not found in MCF (use --tokenizer-json to override)", 1)
				}
				var tokCfg []byte
				if tokenizerConfig != "" && fileExists(tokenizerConfig) {
					tokCfg, err = os.ReadFile(tokenizerConfig)
					if err != nil {
						return cli.Exit(fmt.Sprintf("error: load tokenizer_config.json: %v", err), 1)
					}
				} else {
					tokCfg = mcfFile.SectionData(mcf.SectionTokenizerConfigJSON)
				}

				hfTok, err = tokenizer.LoadHFTokenizerBytes(tokJSON, tokCfg)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load tokenizer.json: %v", err), 1)
				}
				if parsedCfg, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSON, tokCfg); err == nil {
					tokConfig = parsedCfg
				} else {
					fmt.Fprintf(os.Stderr, "warning: parse tokenizer_config.json: %v\n", err)
				}
			}

			tok = hfTok

			// Prefer tokenizer-derived IDs and add-bos behavior; preserve chat_template.
			tokConfig.BOSTokenID = hfTok.BOSID()
			tokConfig.AddBOS = hfTok.AddBOS()
			tokConfig.EOSTokenID = hfTok.EOSID()

			// Resolve effective chat template once for show-config and rendering.
			effectiveTemplate, templateSource := resolveChatTemplate(chatTemplate, tokConfig, m.Config.Arch, cfgBytes)
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
			fmt.Printf("Model loaded in %s\n", loadDuration)

			if showConfig {
				fmt.Fprintf(os.Stderr, "MCF | arch=%s\n", m.Config.Arch)

				fmt.Fprintf(os.Stderr, "blocks=%d embd=%d ffn=%d heads=%d head_dim=%d vocab=%d ctx=%d\n",
					genConfig.BlockCount,
					genConfig.EmbeddingLength,
					genConfig.FFNLength,
					genConfig.HeadCount,
					genConfig.HeadDim,
					genConfig.VocabSize,
					genConfig.ContextLength)
				if genConfig.RopeScaling != nil {
					fmt.Fprintf(
						os.Stderr,
						"rope: base=%g scaling=%s factor=%g orig_ctx=%d low=%g high=%g attn=%g\n",
						genConfig.RopeFreqBase,
						genConfig.RopeScaling.Type,
						genConfig.RopeScaling.Factor,
						genConfig.RopeScaling.OrigMaxCtx,
						genConfig.RopeScaling.LowFactor,
						genConfig.RopeScaling.HighFactor,
						genConfig.RopeScaling.AttentionFactor,
					)
				} else {
					fmt.Fprintf(os.Stderr, "rope: base=%g scaling=none\n", genConfig.RopeFreqBase)
				}

				if len(genConfig.HeadCountKV) > 0 {
					fmt.Fprintf(os.Stderr, "HeadCountKV: %v\n", genConfig.HeadCountKV)
				}

				fmt.Fprintf(os.Stderr, "sampling: temp=%.3g (%s) top_k=%d (%s) top_p=%.3g (%s) repeat_penalty=%.3g (%s)\n",
					temp, samplingSource(c.IsSet("temp"), tempFromGen),
					topK, samplingSource(c.IsSet("top_k") || c.IsSet("topk"), topKFromGen),
					topP, samplingSource(c.IsSet("top_p") || c.IsSet("topp"), topPFromGen),
					repeatPenalty, samplingSource(c.IsSet("repeat-penalty"), repeatFromGen),
				)
				if effectiveTemplate == "" {
					fmt.Fprintln(os.Stderr, "chat_template: none")
				} else {
					fmt.Fprintf(os.Stderr, "chat_template: %s\n", templateSource)
				}
			}

			if seed == -1 {
				seed = time.Now().UnixNano()
			}

			sampler := logits.NewSampler(logits.SamplerConfig{
				Seed:          seed,
				Temperature:   float32(temp),
				TopK:          int(topK),
				TopP:          float32(topP),
				MinP:          float32(c.Float("min-p")),
				RepeatPenalty: float32(repeatPenalty),
				RepeatLastN:   int(repeatLastN),
			})

			stopTokens := []int{tokConfig.EOSTokenID}
			if tokConfig.EOSTokenID < 0 {
				stopTokens = stopTokens[:0]
			}
			// Only add the legacy id=2 fallback when it is actually an end-of-text token
			// for this tokenizer.
			if t, ok := tok.(interface{ TokenString(int) string }); ok {
				token2 := strings.ToLower(strings.TrimSpace(t.TokenString(2)))
				if token2 == "<|endoftext|>" || token2 == "<|end_of_text|>" || token2 == "</s>" {
					if tokConfig.EOSTokenID != 2 && tokConfig.EOSTokenID >= 0 {
						stopTokens = append(stopTokens, 2)
					} else if tokConfig.EOSTokenID < 0 {
						stopTokens = append(stopTokens, 2)
					}
				}
			}

			generator := &inference.Generator{
				Model:         m,
				Sampler:       sampler,
				Tokenizer:     tok,
				StopTokens:    stopTokens,
				ContextTokens: make([]int, 0, int(maxContext)),
			}

			msgs := make([]tokenizer.Message, 0, 10)
			if system != "" {
				msgs = append(msgs, tokenizer.Message{Role: "system", Content: system})
			}

			interactive := false
			if prompt != "" {
				msgs = append(msgs, tokenizer.Message{Role: "user", Content: prompt})
			} else {
				interactive = true
				fmt.Fprintln(os.Stderr, "Interactive mode. Type /exit to quit.")
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

				// Render prompt
				var rendered string
				if !noTemplate && effectiveTemplate != "" {
					// Get info from tokenizer interface if possible
					bosToken := ""
					if t, ok := tok.(interface{ TokenString(int) string }); ok {
						bosToken = t.TokenString(tokConfig.BOSTokenID)
					}
					if s, ok := tokenizer.RenderPromptTemplate(effectiveTemplate, bosToken, tokConfig.AddBOS, msgs, true); ok {
						rendered = s
					}
				}
				if rendered == "" && len(msgs) > 0 {
					// Fallback if no template: just use last content
					rendered = msgs[len(msgs)-1].Content
				}

				if echoPrompt && !interactive {
					fmt.Printf("%s", rendered)
				}

				ids, err := tok.Encode(rendered)
				if err != nil {
					fmt.Fprintln(os.Stderr, "error: encode prompt:", err)
					break
				}

				if showTokens {
					fmt.Fprintf(os.Stderr, "\nInput tokens (%d): %s\n", len(ids), joinInts(ids))
				}

				// Run generation
				var responseBuilder strings.Builder

				_, stats, err := generator.Run(ids, int(steps), func(s string) {
					fmt.Print(s)
					responseBuilder.WriteString(s)
				})
				if err != nil {
					fmt.Fprintln(os.Stderr, "error: generation:", err)
					break
				}

				fmt.Println() // Newline after generation
				fmt.Fprintf(os.Stderr, "Stats: %.2f TPS (%d tokens in %s)\n", stats.TPS, stats.TokensGenerated, stats.Duration)

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

func resolveChatTemplate(override string, cfg tokenizer.TokenizerConfig, arch string, hfConfig []byte) (string, string) {
	template := strings.TrimSpace(override)
	source := ""
	switch {
	case template != "":
		source = "flag"
	case strings.TrimSpace(cfg.ChatTemplate) != "":
		template = cfg.ChatTemplate
		source = "tokenizer_config"
	default:
		if inferred, ok := model.InferChatTemplate(arch, hfConfig); ok {
			template = inferred
			source = "model-default"
		} else {
			return "", "none"
		}
	}

	// If the template looks like a path, load its contents.
	if len(template) < 256 && fileExists(template) {
		if raw, err := os.ReadFile(template); err == nil && len(raw) > 0 {
			template = string(raw)
			source += ":file"
		}
	}
	return template, source
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
