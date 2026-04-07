package main

import (
	"context"
	"fmt"
	"os"

	"github.com/urfave/cli/v3"
	"gopkg.in/yaml.v3"

	"github.com/samcharles93/mantle/internal/logger"
)

func configCmd() *cli.Command {
	return &cli.Command{
		Name:  "config",
		Usage: "Manage global mantle configuration",
		Commands: []*cli.Command{
			{
				Name:  "init",
				Usage: "Initialize a default config file in ~/.config/mantle/config.yaml",
				Action: func(ctx context.Context, cmd *cli.Command) error {
					log := logger.FromContext(ctx)
					path := configPath()
					if _, err := os.Stat(path); err == nil {
						return cli.Exit(fmt.Sprintf("config: file already exists at %s", path), 1)
					}

					// Default configuration
					def := Config{
						ModelsDir:     "~/models/mcf",
						Backend:       "auto",
						LogLevel:      "info",
						LogFormat:     "pretty",
						StreamMode:    "smooth",
						ServerAddress: "127.0.0.1:8080",
					}

					// Set pointers for sampling defaults
					temp := 0.8
					topK := 40
					topP := 0.95
					minP := 0.05
					penalty := 1.1
					ctxLen := int64(4096)
					steps := int64(-1)

					def.Temperature = &temp
					def.TopK = &topK
					def.TopP = &topP
					def.MinP = &minP
					def.RepeatPenalty = &penalty
					def.MaxContext = &ctxLen
					def.Steps = &steps

					if err := SaveConfig(def); err != nil {
						return cli.Exit(fmt.Sprintf("config: failed to save: %v", err), 1)
					}

					log.Info("initialized default configuration", "path", path)
					fmt.Printf("Default configuration created at: %s\n", path)
					return nil
				},
			},
			{
				Name:  "show",
				Usage: "Show the currently active configuration",
				Action: func(ctx context.Context, cmd *cli.Command) error {
					path := configPath()
					cfg := LoadConfig()

					fmt.Printf("Configuration Path: %s\n", path)
					if _, err := os.Stat(path); os.IsNotExist(err) {
						fmt.Println("(file does not exist, using built-in defaults)")
					}

					data, err := yaml.Marshal(cfg)
					if err != nil {
						return cli.Exit(fmt.Sprintf("config: failed to marshal: %v", err), 1)
					}
					fmt.Println("\n--- Current Config ---")
					fmt.Println(string(data))
					return nil
				},
			},
			{
				Name:  "path",
				Usage: "Print the path to the config file",
				Action: func(ctx context.Context, cmd *cli.Command) error {
					fmt.Println(configPath())
					return nil
				},
			},
		},
	}
}
