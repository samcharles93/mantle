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
				Required:    true,
			},
			&cli.StringFlag{
				Name:        "prompt",
				Usage:       "prompt text to tokenize",
				Destination: &prompt,
			},
			&cli.StringFlag{
				Name:        "system",
				Usage:       "optional system prompt",
				Destination: &system,
			},
			&cli.Int64Flag{
				Name:        "steps",
				Usage:       "number of tokens to generate (default -1 = infinite)",
				Value:       -1,
				Destination: &steps,
			},
			&cli.Float64Flag{
				Name:        "temp",
				Usage:       "sampling temperature",
				Value:       0.8,
				Destination: &temp,
			},
			&cli.Int64Flag{
				Name:        "topk",
				Usage:       "top-k sampling parameter",
				Value:       40,
				Destination: &topK,
			},
			&cli.Float64Flag{
				Name:        "topp",
				Usage:       "top-p sampling parameter",
				Value:       0.95,
				Destination: &topP,
			},
			&cli.Float64Flag{
				Name:        "repeat-penalty",
				Usage:       "repetition penalty (1.0 = disabled)",
				Value:       1.1,
				Destination: &repeatPenalty,
			},
			&cli.Int64Flag{
				Name:        "repeat-last-n",
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
				Usage:       "max context length (reduce to save memory)",
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
			// Optional overrides
			&cli.StringFlag{
				Name:        "tokenizer-json",
				Usage:       "override path to tokenizer.json",
				Destination: &tokenizerJSON,
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

			stat, err := os.Stat(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: stat model path %q: %v", modelPath, err), 1)
			}

			var (
				m         *model.Model
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
			genConfig = &m.Config.Config

			// Load Tokenizer (Required for MCF)
			if tokenizerJSON != "" && fileExists(tokenizerJSON) {
				hfTok, err := tokenizer.LoadHFTokenizer(tokenizerJSON, tokenizerConfig)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load tokenizer.json: %v", err), 1)
				}
				tok = hfTok
				tokConfig.BOSTokenID = hfTok.BOSID()
				tokConfig.AddBOS = hfTok.AddBOS()
				tokConfig.EOSTokenID = hfTok.EOSID()
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

				hfTok, err := tokenizer.LoadHFTokenizerBytes(tokJSON, tokCfg)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load tokenizer.json: %v", err), 1)
				}
				tok = hfTok
				tokConfig.BOSTokenID = hfTok.BOSID()
				tokConfig.AddBOS = hfTok.AddBOS()
				tokConfig.EOSTokenID = hfTok.EOSID()
			}

			loadDuration := time.Since(loadStart)
			fmt.Printf("Model loaded in %s\n", loadDuration)

			if showConfig {
				fmt.Fprintf(os.Stderr, "MCF | arch=%s\n", m.Config.Arch)

				fmt.Fprintf(os.Stderr, "blocks=%d embd=%d ffn=%d heads=%d vocab=%d ctx=%d\n",
					genConfig.BlockCount,
					genConfig.EmbeddingLength,
					genConfig.FFNLength,
					genConfig.HeadCount,
					genConfig.VocabSize,
					genConfig.ContextLength)

				if len(genConfig.HeadCountKV) > 0 {
					fmt.Fprintf(os.Stderr, "HeadCountKV: %v\n", genConfig.HeadCountKV)
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
				RepeatPenalty: float32(repeatPenalty),
				RepeatLastN:   int(repeatLastN),
			})

			stopTokens := []int{tokConfig.EOSTokenID}
			if tokConfig.EOSTokenID != 2 {
				stopTokens = append(stopTokens, 2) // <|endoftext|> fallback
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
				if !noTemplate {
					// Prefer explicitly provided template, then GGUF embedded/extracted
					template := chatTemplate
					if template == "" {
						template = tokConfig.ChatTemplate
					}

					// Load from file if it looks like a path and exists
					if len(template) < 256 && fileExists(template) {
						if raw, err := os.ReadFile(template); err == nil {
							template = string(raw)
						}
					}

					// Get info from tokenizer interface if possible
					bosToken := ""
					addBOS := false

					if t, ok := tok.(interface{ TokenString(int) string }); ok {
						bosToken = t.TokenString(tokConfig.BOSTokenID)
					}
					addBOS = tokConfig.AddBOS

					if s, ok := tokenizer.RenderPromptTemplate(template, bosToken, addBOS, msgs, true); ok {
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

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
