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
			Category:    "Model Configuration",
			Aliases:     []string{"m"},
			Usage:       "path to .mcf file",
			Destination: &modelPath,
		},
		&cli.StringFlag{
			Name:        "models-path",
			Category:    "Model Configuration",
			Aliases:     []string{"path"},
			Usage:       "path to directory containing .mcf models",
			Sources:     cli.EnvVars("MANTLE_MODELS_DIR"),
			Destination: &modelsPath,
		},
		&cli.Int64Flag{
			Name:        "max-context",
			Category:    "Model Configuration",
			Aliases:     []string{"max-ctx", "ctx", "c"},
			Usage:       "max context length",
			Value:       4096,
			Sources:     cli.EnvVars("MANTLE_MAX_CONTEXT"),
			Destination: &maxContext,
		},
		&cli.StringFlag{
			Name:        "backend",
			Category:    "Model Configuration",
			Usage:       "execution backend (auto, cpu, cuda)",
			Value:       "auto",
			Sources:     cli.EnvVars("MANTLE_BACKEND"),
			Destination: &backend,
		},
	}
}

func commonTokenizerFlags() []cli.Flag {
	return []cli.Flag{
		&cli.StringFlag{
			Name:        "tokenizer-json",
			Category:    "Tokenizer Overrides",
			Usage:       "override path to tokenizer.json",
			Destination: &tokenizerJSONPath,
		},
		&cli.StringFlag{
			Name:        "tokenizer-config",
			Category:    "Tokenizer Overrides",
			Usage:       "override path to tokenizer_config.json",
			Destination: &tokenizerConfig,
		},
		&cli.StringFlag{
			Name:        "chat-template",
			Category:    "Tokenizer Overrides",
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
			Sources:     cli.EnvVars("MANTLE_LOG_LEVEL"),
			Destination: &logLevel,
		},
		&cli.StringFlag{
			Name:        "log-format",
			Usage:       "log format (pretty, json, text)",
			Value:       "pretty",
			Sources:     cli.EnvVars("MANTLE_LOG_FORMAT"),
			Destination: &logFormat,
		},
		&cli.BoolFlag{
			Name:        "debug",
			Usage:       "enable debug logging (shorthand for --log-level=debug)",
			Sources:     cli.EnvVars("MANTLE_DEBUG"),
			Destination: &debug,
		},
	}
}
