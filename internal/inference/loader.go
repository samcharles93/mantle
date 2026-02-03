package inference

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type Loader struct {
	TokenizerJSONPath   string
	TokenizerConfigPath string
	ChatTemplatePath    string
	HFConfigPath        string
}

type LoadResult struct {
	Engine             Engine
	Model              *model.Instance
	Tokenizer          tokenizer.Tokenizer
	TokenizerConfig    tokenizer.TokenizerConfig
	HFConfigJSON       []byte
	GenerationDefaults GenDefaults
	Arch               string
}

type GenDefaults struct {
	Temperature       *float64
	TopK              *int
	TopP              *float64
	RepetitionPenalty *float64
}

func (l Loader) Load(modelPath string, maxContext int) (*LoadResult, error) {
	if strings.TrimSpace(modelPath) == "" {
		return nil, fmt.Errorf("model path is required")
	}

	mcfFile, err := mcfstore.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = mcfFile.Close() }()

	cfgBytes := mcfFile.SectionData(mcf.SectionHFConfigJSON)
	if l.HFConfigPath != "" {
		override, err := os.ReadFile(l.HFConfigPath)
		if err != nil {
			return nil, fmt.Errorf("load hf config: %w", err)
		}
		if len(override) > 0 {
			cfgBytes = override
		}
	}
	if len(cfgBytes) == 0 {
		return nil, fmt.Errorf("hf config.json not found in MCF (use --hf-config to override)")
	}

	m, err := model.LoadModelMCF(mcfFile, cfgBytes, maxContext)
	if err != nil {
		return nil, err
	}

	genDefaults := parseHFGenerationDefaults(mcfFile.SectionData(mcf.SectionHFGenerationConfigJSON))

	tokJSON, tokCfg, err := l.loadTokenizerBytes(mcfFile)
	if err != nil {
		return nil, err
	}

	hfTok, err := tokenizer.LoadHFTokenizerBytes(tokJSON, tokCfg)
	if err != nil {
		return nil, err
	}

	tokCfgParsed, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSON, tokCfg)
	if err != nil {
		return nil, err
	}
	tokCfgParsed.BOSTokenID = hfTok.BOSID()
	tokCfgParsed.AddBOS = hfTok.AddBOS()
	tokCfgParsed.EOSTokenID = hfTok.EOSID()

	stopTokens := BuildStopTokens(hfTok, tokCfgParsed)

	engine := &EngineImpl{
		model:            m,
		tokenizer:        hfTok,
		tokenizerConfig:  tokCfgParsed,
		arch:             m.Config.Arch,
		hfConfigJSON:     cfgBytes,
		chatTemplatePath: l.ChatTemplatePath,
		stopTokens:       stopTokens,
	}

	return &LoadResult{
		Engine:             engine,
		Model:              m,
		Tokenizer:          hfTok,
		TokenizerConfig:    tokCfgParsed,
		HFConfigJSON:       cfgBytes,
		GenerationDefaults: genDefaults,
		Arch:               m.Config.Arch,
	}, nil
}

func (l Loader) loadTokenizerBytes(mcfFile *mcfstore.File) ([]byte, []byte, error) {
	if l.TokenizerJSONPath != "" {
		tokJSON, err := os.ReadFile(l.TokenizerJSONPath)
		if err != nil {
			return nil, nil, fmt.Errorf("load tokenizer.json: %w", err)
		}
		var tokCfg []byte
		if l.TokenizerConfigPath != "" {
			tokCfg, err = os.ReadFile(l.TokenizerConfigPath)
			if err != nil {
				return nil, nil, fmt.Errorf("load tokenizer_config.json: %w", err)
			}
		}
		return tokJSON, tokCfg, nil
	}

	tokJSON := mcfFile.SectionData(mcf.SectionTokenizerJSON)
	if len(tokJSON) == 0 {
		return nil, nil, fmt.Errorf("tokenizer.json not found in MCF (use --tokenizer-json to override)")
	}
	var tokCfg []byte
	if l.TokenizerConfigPath != "" {
		override, err := os.ReadFile(l.TokenizerConfigPath)
		if err != nil {
			return nil, nil, fmt.Errorf("load tokenizer_config.json: %w", err)
		}
		tokCfg = override
	} else {
		tokCfg = mcfFile.SectionData(mcf.SectionTokenizerConfigJSON)
	}
	return tokJSON, tokCfg, nil
}

func parseHFGenerationDefaults(genBytes []byte) GenDefaults {
	type hfGenerationConfig struct {
		Temperature       *float64 `json:"temperature"`
		TopK              *int     `json:"top_k"`
		TopP              *float64 `json:"top_p"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
	}
	if len(genBytes) == 0 {
		return GenDefaults{}
	}
	var cfg hfGenerationConfig
	if err := json.Unmarshal(genBytes, &cfg); err != nil {
		return GenDefaults{}
	}
	return GenDefaults(cfg)
}
