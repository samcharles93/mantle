package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/urfave/cli/v3"

	clipaths "github.com/samcharles93/mantle/internal/cli/paths"
	cliux "github.com/samcharles93/mantle/internal/cli/ux"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func showCmd() *cli.Command {
	var (
		modelPath string
	)

	return &cli.Command{
		Name:  "show",
		Usage: "Show high-level metadata for an .mcf model",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:        "model",
				Aliases:     []string{"m"},
				Usage:       "path to .mcf file",
				Destination: &modelPath,
			},
			&cli.StringFlag{
				Name:        "models-path",
				Aliases:     []string{"path"},
				Usage:       "path to directory containing .mcf models",
				Destination: &modelsPath,
			},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			resolved, err := clipaths.ResolveRunModelPath(modelPath, modelsPath, os.Stdin, os.Stderr)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: resolve model: %v", err), 1)
			}
			modelPath = resolved

			stat, err := os.Stat(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: stat model path %q: %v", modelPath, err), 1)
			}

			mf, err := mcf.Open(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: open mcf: %v", err), 1)
			}
			defer func() { _ = mf.Close() }()

			fmt.Printf("Model: %s\n", filepath.Base(modelPath))
			fmt.Printf("Path:  %s\n", modelPath)
			fmt.Printf("Size:  %s\n", cliux.FormatModelSize(stat.Size()))

			cfgBytes := mf.SectionData(mf.Section(mcf.SectionHFConfigJSON))
			if len(cfgBytes) > 0 {
				cfg := parseHFConfig(cfgBytes)
				fmt.Println("\nArchitecture")
				fmt.Printf("  Type:         %s\n", cfg.ModelType)
				if len(cfg.Architectures) > 0 {
					fmt.Printf("  Classes:      %s\n", strings.Join(cfg.Architectures, ", "))
				}

				fmt.Println("\nParameters")
				fmt.Printf("  Layers:       %d\n", cfg.NumLayers)
				fmt.Printf("  Hidden Size:  %d\n", cfg.HiddenSize)
				fmt.Printf("  Attention:    %d heads / %d KV heads\n", cfg.HeadCount, cfg.KVHeads)
				fmt.Printf("  Context:      %d tokens\n", cfg.MaxPosition)
				fmt.Printf("  Vocabulary:   %d tokens\n", cfg.VocabSize)
			}

			quantBytes := mf.SectionData(mf.Section(mcf.SectionQuantInfo))
			if len(quantBytes) > 0 {
				if qi, err := mcf.ParseQuantInfoSection(quantBytes); err == nil {
					if qi.Count() > 0 {
						if r, err := qi.Record(0); err == nil {
							fmt.Println("\nQuantization")
							fmt.Printf("  Method:       %s\n", dtypeName(mcf.TensorDType(r.Method)))
						}
					}
				}
			}

			return nil
		},
	}
}
