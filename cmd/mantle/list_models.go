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
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func listModelsCmd() *cli.Command {
	return &cli.Command{
		Name:    "list-models",
		Aliases: []string{"ls", "models"},
		Usage:   "List available MCF models",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:        "models-path",
				Aliases:     []string{"path"},
				Usage:       "path to directory containing .mcf models",
				Destination: &modelsPath,
			},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			log := logger.FromContext(ctx)

			dir := strings.TrimSpace(modelsPath)
			if dir == "" {
				dir = strings.TrimSpace(os.Getenv(clipaths.EnvMantleModelsDir))
			}
			if dir == "" {
				return cli.Exit("error: --models-path is required unless MANTLE_MODELS_DIR is set", 1)
			}

			models, err := clipaths.DiscoverMCFModels(dir)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: %v", err), 1)
			}
			if len(models) == 0 {
				log.Info("no models found", "path", dir)
				return nil
			}

			table := cliux.NewTable("NAME", "SIZE", "ARCH", "QUANT")
			for _, m := range models {
				name := filepath.Base(m)
				info, err := os.Stat(m)
				if err != nil {
					table.AddRow(name, "?", "?", "?")
					continue
				}
				size := cliux.FormatModelSize(info.Size())

				// Try to get model info from MCF header
				arch := ""
				quant := "none"
				if mf, err := mcf.Open(m); err == nil {
					cfgBytes := mf.SectionData(mf.Section(mcf.SectionHFConfigJSON))
					if len(cfgBytes) > 0 {
						cfg := parseHFConfig(cfgBytes)
						if cfg.ModelType != "" {
							arch = cfg.ModelType
						}
					}

					quantBytes := mf.SectionData(mf.Section(mcf.SectionQuantInfo))
					if len(quantBytes) > 0 {
						if qi, err := mcf.ParseQuantInfoSection(quantBytes); err == nil {
							// For summary, check first tensor's quant method
							if qi.Count() > 0 {
								if r, err := qi.Record(0); err == nil {
									quant = dtypeName(mcf.TensorDType(r.Method))
								}
							}
						}
					}
					_ = mf.Close()
				}

				table.AddRow(name, size, arch, quant)
			}
			fmt.Println(table.String())
			fmt.Printf("%d model(s) found\n", len(models))
			return nil
		},
	}
}
