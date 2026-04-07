package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/urfave/cli/v3"

	clipaths "github.com/samcharles93/mantle/internal/cli/paths"
	"github.com/samcharles93/mantle/internal/hf"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func pullCmd() *cli.Command {
	return &cli.Command{
		Name:  "pull",
		Usage: "Pull a model from Hugging Face and pack it into .mcf",
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

			repo := cmd.Args().First()
			if repo == "" {
				return cli.Exit("error: repository name required (e.g., Qwen/Qwen3.5-0.8B-Instruct)", 1)
			}

			// Clean up repo name for local directory
			repoDirName := strings.ReplaceAll(repo, "/", "--")
			tmpDir := filepath.Join(os.TempDir(), "mantle-pull", repoDirName)
			if err := os.MkdirAll(tmpDir, 0755); err != nil {
				return cli.Exit(fmt.Sprintf("pull: failed to create temp dir: %v", err), 1)
			}
			defer os.RemoveAll(tmpDir)

			log.Info("fetching model info", "repo", repo)
			info, err := hf.GetModelInfo(repo)
			if err != nil {
				return cli.Exit(fmt.Sprintf("pull: hf api error: %v", err), 1)
			}

			files := hf.FilterFiles(info)
			if len(files) == 0 {
				return cli.Exit("pull: no suitable files found in repository", 1)
			}

			log.Info("downloading files", "count", len(files))
			for _, rpath := range files {
				log.Info("downloading", "file", rpath)
				err := hf.DownloadFile(repo, rpath, tmpDir, func(current, total int64) {
					// Basic text progress
					if total > 0 && current%(1024*1024*10) == 0 { // log every 10MB
						fmt.Printf("\r  %s: %.1f%% (%d/%d MB)", rpath, float64(current)/float64(total)*100, current/(1024*1024), total/(1024*1024))
					}
				})
				if err != nil {
					return cli.Exit(fmt.Sprintf("pull: download failed: %v", err), 1)
				}
				fmt.Println()
			}

			// Resolve models directory
			dir := strings.TrimSpace(modelsPath)
			if dir == "" {
				dir = strings.TrimSpace(os.Getenv(clipaths.EnvMantleModelsDir))
			}
			if dir == "" {
				cfg := LoadConfig()
				dir = cfg.ModelsDir
			}
			if dir == "" {
				return cli.Exit("error: models directory not configured. set --models-path or MANTLE_MODELS_DIR or use 'mantle config init'", 1)
			}

			// Handle tilde
			if strings.HasPrefix(dir, "~") {
				home, _ := os.UserHomeDir()
				dir = filepath.Join(home, dir[1:])
			}

			outPath := filepath.Join(dir, repoDirName+".mcf")
			if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
				return cli.Exit(fmt.Sprintf("pull: failed to create output dir: %v", err), 1)
			}

			log.Info("packing model", "output", outPath)
			packOpts := mcf.PackOptions{
				InputDir:         tmpDir,
				OutputPath:       outPath,
				Dedup:            true,
				TensorAlign:      64,
				Cast:             "keep",
				IncludeResources: true,
				ProgressEvery:    50,
				Logf: func(format string, args ...any) {
					log.Info(fmt.Sprintf(format, args...))
				},
			}

			if err := mcf.Pack(packOpts); err != nil {
				return cli.Exit(fmt.Sprintf("pull: pack failed: %v", err), 1)
			}

			fmt.Printf("\nSuccessfully pulled and packed model to: %s\n", outPath)
			return nil
		},
	}
}
