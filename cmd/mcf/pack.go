package main

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func packCmd() *cli.Command {
	return &cli.Command{
		Name:  "pack",
		Usage: "Pack a Safetensors model directory into a single .mcf file",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "input",
				Aliases:  []string{"in"},
				Usage:    "Model directory containing config/tokenizer/vocab/etc",
				Required: true,
			},
			&cli.StringFlag{
				Name:     "output",
				Aliases:  []string{"out"},
				Usage:    "Output .mcf path",
				Required: true,
			},
			&cli.StringFlag{
				Name:  "model-safetensors",
				Usage: "Optional safetensors file override (relative to --input unless absolute)",
			},
			&cli.IntFlag{
				Name:  "tensor-align",
				Usage: "Alignment (bytes) between tensor payloads in TensorData (0 disables). Typical: 64",
				Value: 64,
			},
			&cli.StringFlag{
				Name:  "cast",
				Usage: "Float casting: keep|f16|bf16",
				Value: "keep",
			},
			&cli.BoolFlag{
				Name:  "no-resources",
				Usage: "Do not embed config/tokenizer/vocab/merges sections",
			},

			// Optional explicit resource overrides (otherwise auto: <input>/<filename>)
			&cli.StringFlag{Name: "config-json", Usage: "Override config.json path"},
			&cli.StringFlag{Name: "generation-config-json", Usage: "Override generation_config.json path"},
			&cli.StringFlag{Name: "tokenizer-json", Usage: "Override tokenizer.json path"},
			&cli.StringFlag{Name: "tokenizer-config-json", Usage: "Override tokenizer_config.json path"},
			&cli.StringFlag{Name: "vocab-json", Usage: "Override vocab.json path"},
			&cli.StringFlag{Name: "merges-txt", Usage: "Override merges.txt path"},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			inDir := cmd.String("input")
			outPath := cmd.String("output")

			modelST := cmd.String("model-safetensors")
			if modelST != "" && !filepath.IsAbs(modelST) {
				modelST = filepath.Join(inDir, modelST)
			}

			opts := mcf.PackOptions{
				InputDir:         inDir,
				ModelSafetensors: modelST,
				OutputPath:       outPath,
				TensorAlign:      cmd.Int("tensor-align"),
				Cast:             cmd.String("cast"),
				IncludeResources: !cmd.Bool("no-resources"),

				ConfigJSONPath:           cmd.String("config-json"),
				GenerationConfigJSONPath: cmd.String("generation-config-json"),
				TokenizerJSONPath:        cmd.String("tokenizer-json"),
				TokenizerConfigJSONPath:  cmd.String("tokenizer-config-json"),
				VocabJSONPath:            cmd.String("vocab-json"),
				MergesTXTPath:            cmd.String("merges-txt"),
			}

			if err := mcf.Pack(opts); err != nil {
				return fmt.Errorf("pack: %w", err)
			}
			_ = ctx
			return nil
		},
	}
}
