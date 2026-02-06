package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/version"

	"github.com/urfave/cli/v3"
)

func main() {
	app := &cli.Command{
		Name:    "mantle",
		Usage:   "Model execution substrate CLI",
		Version: version.String(),
		Flags:   loggingFlags(),
		Before: func(ctx context.Context, cmd *cli.Command) (context.Context, error) {
			// Initialize logger
			level := logger.ParseLevel(logLevel)
			if debug {
				level = slog.LevelDebug
			}

			var log logger.Logger
			switch logFormat {
			case "json":
				log = logger.JSON(os.Stderr, level)
			case "text":
				log = logger.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
					Level:     level,
					AddSource: true,
				}))
			case "pretty":
				log = logger.Pretty(os.Stderr, level)
			default:
				log = logger.Pretty(os.Stderr, level)
			}

			// Add logger to context
			return logger.WithContext(ctx, log), nil
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			return cli.ShowAppHelp(cmd)
		},
		Commands: []*cli.Command{
			runCmd(),
			packCmd(),
			quantizeCmd(),
			inspectCmd(),
			serveCmd(),
			versionCmd(),
		},
	}

	if err := app.Run(context.Background(), os.Args); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
