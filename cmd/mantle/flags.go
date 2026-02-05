package main

import "github.com/urfave/cli/v3"

var (
	modelPath         string
	modelsPath        string
	maxContext        int64
	backend           string
	tokenizerJSONPath string
	tokenizerConfig   string
	chatTemplate      string
	logLevel          string
	logFormat         string
	debug             bool
)

func commonModelFlags() []cli.Flag {
	return []cli.Flag{
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
		&cli.Int64Flag{
			Name:        "max-context",
			Aliases:     []string{"max-ctx", "ctx", "c"},
			Usage:       "max context length",
			Value:       4096,
			Destination: &maxContext,
		},
		&cli.StringFlag{
			Name:        "backend",
			Usage:       "execution backend (auto, cpu, cuda)",
			Value:       "auto",
			Destination: &backend,
		},
	}
}

func commonTokenizerFlags() []cli.Flag {
	return []cli.Flag{
		&cli.StringFlag{
			Name:        "tokenizer-json",
			Usage:       "override path to tokenizer.json",
			Destination: &tokenizerJSONPath,
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
	}
}

func loggingFlags() []cli.Flag {
	return []cli.Flag{
		&cli.StringFlag{
			Name:        "log-level",
			Usage:       "log level (debug, info, warn, error)",
			Value:       "info",
			Destination: &logLevel,
		},
		&cli.StringFlag{
			Name:        "log-format",
			Usage:       "log format (pretty, json, text)",
			Value:       "pretty",
			Destination: &logFormat,
		},
		&cli.BoolFlag{
			Name:        "debug",
			Usage:       "enable debug logging (shorthand for --log-level=debug)",
			Destination: &debug,
		},
	}
}
