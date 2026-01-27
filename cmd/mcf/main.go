// cmd/mcf/main.go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/urfave/cli/v3"
)

func main() {
	app := &cli.Command{
		Name:  "mcf",
		Usage: "Pack and/or Quantize MCF files from Safetensors models",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "input-path",
				Aliases:  []string{"in"},
				Usage:    "Input model directory (preferred) or safetensors file path",
				Required: false,
			},
			&cli.StringFlag{
				Name:     "output-path",
				Aliases:  []string{"out"},
				Usage:    "Output .mcf path",
				Required: false,
			},

			// Optional explicit overrides.
			&cli.StringFlag{
				Name:    "input-safetensors",
				Aliases: []string{"model-safetensors"},
				Usage:   "Optional safetensors file override (relative to --input-path unless absolute)",
			},
			&cli.StringFlag{
				Name:  "input-tokenizer",
				Usage: "Optional tokenizer.json override path",
			},
			&cli.StringFlag{
				Name:  "input-tokenizer_config",
				Usage: "Optional tokenizer_config.json override path",
			},
			&cli.StringFlag{
				Name:  "input-config",
				Usage: "Optional config.json override path",
			},
			&cli.StringFlag{
				Name:  "input-gen_config",
				Usage: "Optional generation_config.json override path",
			},
			&cli.StringFlag{
				Name:  "input-vocab",
				Usage: "Optional vocab.json override path",
			},
			&cli.StringFlag{
				Name:  "input-merges",
				Usage: "Optional merges.txt override path",
			},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			return cli.ShowAppHelp(cmd)
		},
		Commands: []*cli.Command{
			packCmd(),
			quantizeCmd(),
		},
	}

	if err := app.Run(context.Background(), os.Args); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
