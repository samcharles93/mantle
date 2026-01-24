package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"sort"
	"strings"
	"time"

	"github.com/urfave/cli/v3"

	"infer/internal/gguf"
	"infer/internal/inference"
	"infer/internal/logits"
	"infer/internal/model"
	"infer/internal/tokenizer"
)

func main() {
	var (
		modelPath       string
		prompt          string
		system          string
		steps           int64
		temp            float64
		topK            int64
		topP            float64
		repeatPenalty   float64
		repeatLastN     int64
		seed            int64
		maxContext      int64
		noTemplate      bool
		echoPrompt      bool

		// Optional overrides
		tokenizerJSON   string
		tokenizerConfig string
		chatTemplate    string
		safetensorsFile string
		hfConfigFile    string

		// Debug flags
		showConfig  bool
		        showTokens  bool
				showKV      bool
				showTensors int64
		
				// Profiling
				cpuProfile string
				memProfile string
			)
		
			cmd := &cli.Command{
				Name:  "infer",
				Usage: "Inference utility for LLM models",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:        "model",
						Usage:       "path to .gguf file OR folder containing safetensors",
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
						Name:        "safetensors-file",
						Usage:       "explicit path to .safetensors file",
						Destination: &safetensorsFile,
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
					&cli.BoolFlag{
						Name:        "show-kv",
						Usage:       "print all GGUF metadata keys",
						Destination: &showKV,
					},
					&cli.Int64Flag{
						Name:        "show-tensors",
						Usage:       "number of tensors to list (0 to skip, -1 for all)",
						Value:       0,
						Destination: &showTensors,
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
						defer f.Close()
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
							defer f.Close()
							// Run GC to get up-to-date statistics
							// runtime.GC() // optional, maybe too intrusive? 
							// Actually usually good for mem profile accuracy to run GC before write.
							// But user asked to avoid GC issues... let's assume standard behavior is fine unless they want debug.
							// We'll write the profile.
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
				isGGUF    bool
				ggufFile  *gguf.File
				genConfig *model.Config
			)

			// Mode detection
			isSafeTensorsDir := stat.IsDir()
			isSafeTensorsFile := !stat.IsDir() && strings.HasSuffix(modelPath, ".safetensors")
			isGGUF = !stat.IsDir() && !isSafeTensorsFile // Default to GGUF if file and not .safetensors

			loadStart := time.Now()

			if isSafeTensorsDir || isSafeTensorsFile || safetensorsFile != "" {
				// --- SafeTensors Path ---
				fmt.Println("Loading SafeTensors model...")

				// Resolve paths
				stPath := safetensorsFile
				cfgPath := hfConfigFile
				tokJSON := tokenizerJSON
				tokCfgPath := tokenizerConfig
			tplPath := chatTemplate

				if isSafeTensorsDir {
					if stPath == "" {
						stPath = filepath.Join(modelPath, "model.safetensors")
					}
					if cfgPath == "" {
						cfgPath = filepath.Join(modelPath, "config.json")
					}
					if tokJSON == "" {
						tokJSON = filepath.Join(modelPath, "tokenizer.json")
					}
					if tokCfgPath == "" {
						tokCfgPath = filepath.Join(modelPath, "tokenizer_config.json")
					}
					if tplPath == "" {
						tplPath = filepath.Join(modelPath, "chat_template.jinja")
					}
				} else if isSafeTensorsFile {
					if stPath == "" {
						stPath = modelPath
					}
					dir := filepath.Dir(stPath)
					if cfgPath == "" {
						cfgPath = filepath.Join(dir, "config.json")
					}
					if tokJSON == "" {
						tokJSON = filepath.Join(dir, "tokenizer.json")
					}
					if tokCfgPath == "" {
						tokCfgPath = filepath.Join(dir, "tokenizer_config.json")
					}
					if tplPath == "" {
						tplPath = filepath.Join(dir, "chat_template.jinja")
					}
				}

				if !fileExists(stPath) {
					return cli.Exit(fmt.Sprintf("error: safetensors file not found: %s", stPath), 1)
				}
				if !fileExists(cfgPath) {
					return cli.Exit(fmt.Sprintf("error: config.json not found: %s", cfgPath), 1)
				}

				// Load Model
			m, err = model.LoadModelSafetensors(stPath, cfgPath, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load safetensors: %v", err), 1)
			}
			genConfig = &m.Config.Config

			// Load Tokenizer (Required for SafeTensors)
			if !fileExists(tokJSON) {
				return cli.Exit(fmt.Sprintf("error: tokenizer.json required for SafeTensors but not found: %s", tokJSON), 1)
			}
			hfTok, err := tokenizer.LoadHFTokenizer(tokJSON, tokCfgPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load tokenizer.json: %v", err), 1)
			}
			tok = hfTok

			// Attempt to load chat template if not provided
			if tplPath != "" && fileExists(tplPath) {
				if raw, err := os.ReadFile(tplPath); err == nil {
					tokConfig.ChatTemplate = string(raw)
				}
			}

			// Fill minimal TokenizerConfig for main loop
			tokConfig.BOSTokenID = hfTok.BOSID()
			tokConfig.AddBOS = hfTok.AddBOS()
			tokConfig.EOSTokenID = hfTok.EOSID()

		} else {
			// --- GGUF Path ---
			fmt.Printf("Loading GGUF model: %s\n", modelPath)

			f, err := gguf.Open(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: open gguf: %v", err), 1)
			}
			ggufFile = f

			cfg, err := model.LoadConfig(f)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load model config: %v", err), 1)
			}
			genConfig = &cfg.Config
			tokConfig = cfg.Tokenizer

			// Load Tokenizer
			// Priority: Explicit JSON > GGUF Embedded
			if tokenizerJSON != "" && fileExists(tokenizerJSON) {
				hfTok, err := tokenizer.LoadHFTokenizer(tokenizerJSON, tokenizerConfig)
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: load tokenizer.json: %v", err), 1)
				}
				tok = hfTok
			} else {
				// Build from GGUF
				gptTok, err := cfg.Tokenizer.BuildGPT2()
				if err != nil {
					return cli.Exit(fmt.Sprintf("error: build tokenizer from GGUF: %v", err), 1)
				}
				tok = gptTok
			}

			// Load Model
			m, err = model.LoadModel(modelPath, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load gguf model: %v", err), 1)
			}
		}

		loadDuration := time.Since(loadStart)
		fmt.Printf("Model loaded in %s\n", loadDuration)

		if showConfig {
			if isGGUF && ggufFile != nil {
				fmt.Fprintf(os.Stderr, "GGUF v%d | tensors=%d | kv=%d\n",
						ggufFile.Header.Version, ggufFile.Header.TensorCount, ggufFile.Header.KVCount)
				fmt.Fprintf(os.Stderr, "arch=%s name=%s quant=%s\n",
						m.Config.Arch,
						mustString(ggufFile.KV, "general.name"),
						mustString(ggufFile.KV, "general.quantization"))
			} else {
				fmt.Fprintf(os.Stderr, "SafeTensors | arch=%s\n", m.Config.Arch)
			}

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

		if isGGUF && showKV && ggufFile != nil {
			fmt.Fprintf(os.Stderr, "Tokenizer Model: %s\n", tokConfig.Model)
			fmt.Fprintf(os.Stderr, "Tokenizer Pre: %s\n", tokConfig.Pre)
			fmt.Fprintf(os.Stderr, "Tokenizer Tokens: %d\n", len(tokConfig.Tokens))
			fmt.Fprintf(os.Stderr, "Tokenizer Merges: %d\n", len(tokConfig.Merges))
			fmt.Fprintf(os.Stderr, "Tokenizer Types: %d\n", len(tokConfig.TokenTypes))

			// Debug: find reserved token
			for i, t := range tokConfig.Tokens {
				if t == "<|reserved_29|>" {
					fmt.Fprintf(os.Stderr, "Debug: <|reserved_29|> is ID %d\n", i)
				}
				if i < 10 {
					fmt.Fprintf(os.Stderr, "Token[%d] = %q\n", i, t)
				}
			}

			fmt.Fprintln(os.Stderr, "\nAll metadata:")
			keys := make([]string, 0, len(ggufFile.KV))
			for k := range ggufFile.KV {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				fmt.Fprintf(os.Stderr, "  %s = %s\n", k, formatValue(ggufFile.KV[k]))
			}
		}

		if showTensors != 0 && isGGUF && ggufFile != nil {
			fmt.Fprintln(os.Stderr, "\nTensors:")
			count := int64(len(ggufFile.Tensors))
			max := showTensors
			if max < 0 || max > count {
				max = count
			}
			for i := int64(0); i < max; i++ {
				t := ggufFile.Tensors[i]
				fmt.Fprintf(os.Stderr, "  %-40s %-6s dims=%s off=%d\n",
						t.Name, t.Type.String(), formatDims(t.Dims), t.Offset)
			}
			if max < count {
				fmt.Fprintf(os.Stderr, "  ... (%d more)\n", count-max)
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
				if len(ids) > 0 {
					fmt.Fprint(os.Stderr, "Decoded input: ")
					for i, id := range ids {
						if i >= 5 {
							fmt.Fprint(os.Stderr, " ...")
							break
						}
						s, _ := tok.Decode([]int{id})
							fmt.Fprintf(os.Stderr, "%q ", s)
						}
					fmt.Fprintln(os.Stderr)
				}
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

	if err := cmd.Run(context.Background(), os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func mustString(kv map[string]gguf.Value, key string) string {
	if s, ok := gguf.GetString(kv, key); ok {
		return s
	}
	return "-"
}

func formatDims(dims []uint64) string {
	if len(dims) == 0 {
		return "[]"
	}
	parts := make([]string, len(dims))
	for i, v := range dims {
		parts[i] = fmt.Sprintf("%d", v)
	}
	return "[" + strings.Join(parts, "x") + "]"
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

func formatValue(v gguf.Value) string {
	switch val := v.Value.(type) {
	case string:
		return val
	case bool:
		if val {
			return "true"
		}
		return "false"
	case gguf.ArrayValue:
		return fmt.Sprintf("array(%s) len=%d", val.ElemType.String(), len(val.Values))
	default:
		return fmt.Sprintf("%v", val)
	}
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
