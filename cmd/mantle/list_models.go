package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/urfave/cli/v3"

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
				dir = strings.TrimSpace(os.Getenv(envMantleModelsDir))
			}
			if dir == "" {
				return cli.Exit("error: --models-path is required unless MANTLE_MODELS_DIR is set", 1)
			}

			models, err := discoverMCFModels(dir)
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
				size := formatModelSize(info.Size())

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

func formatModelSize(bytes int64) string {
	const (
		kb = 1024
		mb = 1024 * kb
		gb = 1024 * mb
	)
	switch {
	case bytes >= gb:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	case bytes >= mb:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	case bytes >= kb:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(kb))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}
