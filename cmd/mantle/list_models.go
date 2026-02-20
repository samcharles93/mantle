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

			fmt.Printf("Models in %s:\n\n", dir)
			for _, m := range models {
				name := filepath.Base(m)
				info, err := os.Stat(m)
				if err != nil {
					fmt.Printf("  %s\n", name)
					continue
				}
				size := cliux.FormatModelSize(info.Size())

				// Try to get model info from MCF header
				arch := ""
				if mf, err := mcf.Open(m); err == nil {
					cfgBytes := mf.SectionData(mf.Section(mcf.SectionHFConfigJSON))
					if len(cfgBytes) > 0 {
						cfg := parseHFConfig(cfgBytes)
						if cfg.ModelType != "" {
							arch = cfg.ModelType
						}
					}
					_ = mf.Close()
				}

				if arch != "" {
					fmt.Printf("  %-40s %8s  (%s)\n", name, size, arch)
				} else {
					fmt.Printf("  %-40s %8s\n", name, size)
				}
			}
			fmt.Printf("\n%d model(s) found\n", len(models))
			return nil
		},
	}
}
