package main

import (
	"os"
	"path/filepath"

	"github.com/urfave/cli/v3"
	"gopkg.in/yaml.v3"
)

// Config represents the mantle configuration file (~/.config/mantle/config.yaml).
// All fields are pointers so we can distinguish "not set" from zero values.
type Config struct {
	ModelsDir string `yaml:"models_dir"`

	// Sampling defaults
	Temperature   *float64 `yaml:"temperature"`
	TopK          *int64   `yaml:"top_k"`
	TopP          *float64 `yaml:"top_p"`
	MinP          *float64 `yaml:"min_p"`
	RepeatPenalty *float64 `yaml:"repeat_penalty"`
	MaxContext    *int64   `yaml:"max_context"`
	Steps         *int64   `yaml:"steps"`
	Seed          *int64   `yaml:"seed"`

	// Backend
	Backend string `yaml:"backend"`

	// Output
	StreamMode string `yaml:"stream_mode"`
	LogLevel   string `yaml:"log_level"`
	LogFormat  string `yaml:"log_format"`

	// Server
	ServerAddress string `yaml:"server_address"`
}

func configPath() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "mantle", "config.yaml")
}

// applyRunConfig applies config file defaults to run command variables
// when the corresponding CLI flag was not explicitly set.
func applyRunConfig(c *cli.Command, cfg Config,
	modelsPath *string, temp *float64, topK *int64, topP *float64,
	repeatPenalty *float64, steps *int64, seed *int64, streamMode *string,
) {
	if cfg.ModelsDir != "" && !c.IsSet("models-path") {
		*modelsPath = cfg.ModelsDir
	}
	if cfg.Backend != "" && !c.IsSet("backend") {
		backend = cfg.Backend
	}
	if cfg.MaxContext != nil && !c.IsSet("max-context") {
		maxContext = *cfg.MaxContext
	}
	if cfg.Temperature != nil && !c.IsSet("temp") && !c.IsSet("temperature") && !c.IsSet("t") {
		*temp = *cfg.Temperature
	}
	if cfg.TopK != nil && !c.IsSet("top-k") && !c.IsSet("top_k") && !c.IsSet("topk") {
		*topK = *cfg.TopK
	}
	if cfg.TopP != nil && !c.IsSet("top-p") && !c.IsSet("top_p") && !c.IsSet("topp") {
		*topP = *cfg.TopP
	}
	if cfg.RepeatPenalty != nil && !c.IsSet("repeat-penalty") && !c.IsSet("repeat_penalty") {
		*repeatPenalty = *cfg.RepeatPenalty
	}
	if cfg.Steps != nil && !c.IsSet("steps") {
		*steps = *cfg.Steps
	}
	if cfg.Seed != nil && !c.IsSet("seed") {
		*seed = *cfg.Seed
	}
	if cfg.StreamMode != "" && !c.IsSet("stream-mode") {
		*streamMode = cfg.StreamMode
	}
}

// applyServeConfig applies config file defaults to serve command variables.
func applyServeConfig(c *cli.Command, cfg Config, addr *string) {
	if cfg.ModelsDir != "" && !c.IsSet("models-path") {
		modelsPath = cfg.ModelsDir
	}
	if cfg.Backend != "" && !c.IsSet("backend") {
		backend = cfg.Backend
	}
	if cfg.MaxContext != nil && !c.IsSet("max-context") {
		maxContext = *cfg.MaxContext
	}
	if cfg.ServerAddress != "" && !c.IsSet("addr") {
		*addr = cfg.ServerAddress
	}
}

// LoadConfig reads the config file. Returns a zero Config if the file doesn't exist.
func LoadConfig() Config {
	path := configPath()
	if path == "" {
		return Config{}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}
	}
	return cfg
}
