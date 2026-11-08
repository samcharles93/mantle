package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/samcharles93/mantle/internal/backend"
	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/hostcaps"
	"github.com/samcharles93/mantle/internal/logger"
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
	Backend             string
	DisableSWA          bool
	LoadOptions         core.LoadModelOptions
	UseGraph            bool
	TilingConfig        simd.TilingConfig

	// DSpark speculative decoding. DraftModelPath is a draft MCF; when set,
	// the target captures the draft's target_layer_ids as hidden taps, the
	// CPU backend is forced, and the returned engine speculates. DraftTokens
	// overrides the draft config's block_size (<= 0 uses the config value);
	// DraftConfidence is the sampling-time acceptance threshold (0 disables).
	DraftModelPath  string
	DraftTokens     int
	DraftConfidence float64
}

type LoadResult struct {
	Engine             Engine
	Runtime            core.Runtime
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
	EOSTokenIDs       []int
	DoSample          *bool
}

func (l Loader) Load(ctx context.Context, modelPath string, maxContext int) (*LoadResult, error) {
	log := logger.FromContext(ctx)
	opts := l.LoadOptions
	// propagate UseGraph toggle from Loader to LoadModelOptions
	opts.UseGraph = l.UseGraph
	if opts.HostCaps == nil {
		if caps := hostcaps.FromContext(ctx); caps != nil {
			opts.HostCaps = caps
		} else {
			opts.HostCaps = hostcaps.Detect()
		}
	}
	// Apply tiling configuration
	if l.TilingConfig.TileM != 0 || l.TilingConfig.TileN != 0 || l.TilingConfig.TileK != 0 {
		opts.TilingConfig = core.TilingConfig{
			TileM: l.TilingConfig.TileM,
			TileN: l.TilingConfig.TileN,
			TileK: l.TilingConfig.TileK,
		}
	}

	if strings.TrimSpace(modelPath) == "" {
		return nil, fmt.Errorf("model path is required")
	}
	mcfFile, err := mcfstore.Open(modelPath)
	if err != nil {
		return nil, err
	}
	var draftFile *mcfstore.File
	cleanup := func(err error) (*LoadResult, error) {
		_ = mcfFile.Close()
		if draftFile != nil {
			_ = draftFile.Close()
		}
		return nil, err
	}

	// A DSpark draft fixes the target's hidden-tap layers before the target
	// loads, so open and parse it first.
	var draftCfgBytes []byte
	if l.DraftModelPath != "" {
		df, err := mcfstore.Open(l.DraftModelPath)
		if err != nil {
			return cleanup(fmt.Errorf("open draft model: %w", err))
		}
		draftFile = df
		draftCfgBytes = df.SectionData(mcf.SectionHFConfigJSON)
		if len(draftCfgBytes) == 0 {
			return cleanup(fmt.Errorf("draft MCF %q has no hf config.json", l.DraftModelPath))
		}
		dcfg, err := model.ParseDSparkConfig(draftCfgBytes)
		if err != nil {
			return cleanup(err)
		}
		opts.HiddenTapLayers = dcfg.TargetLayerIDs
	}

	cfgBytes := mcfFile.SectionData(mcf.SectionHFConfigJSON)
	if l.HFConfigPath != "" {
		override, err := os.ReadFile(l.HFConfigPath)
		if err != nil {
			return cleanup(fmt.Errorf("load hf config: %w", err))
		}
		if len(override) > 0 {
			cfgBytes = override
		}
	}
	if len(cfgBytes) == 0 {
		return cleanup(fmt.Errorf("hf config.json not found in MCF (use --hf-config to override)"))
	}
	if l.DisableSWA {
		updated, err := disableSWAConfig(cfgBytes)
		if err != nil {
			return cleanup(fmt.Errorf("disable swa: %w", err))
		}
		cfgBytes = updated
	}

	backendName, err := backend.Normalize(l.Backend)
	if err != nil {
		return cleanup(err)
	}
	if draftFile != nil {
		switch backendName {
		case backend.CUDA:
			return cleanup(fmt.Errorf("dspark speculation is cpu-only: --draft-model cannot be combined with --backend cuda"))
		case backend.Auto:
			backendName = backend.CPU
			log.Info("draft model requested; forcing cpu backend for speculation")
		}
	}

	var runtime backend.Backend
	switch backendName {
	case backend.CPU:
		runtime, err = backend.New(backend.CPU)
		if err != nil {
			return cleanup(err)
		}
		log.Info("backend selected", "backend", "cpu")
	case backend.CUDA:
		runtime, err = backend.New(backend.CUDA)
		if err != nil {
			return cleanup(err)
		}
		log.Info("backend selected", "backend", "cuda")
	case backend.Auto:
		runtime, err = backend.New(backend.CUDA)
		if err != nil {
			log.Debug("CUDA backend not found, trying CPU", "debug", err) // not necessarily an error here, could be intentional
			runtime, err = backend.New(backend.CPU)
			if err != nil {
				return cleanup(err)
			}
			log.Info("backend selected", "backend", "cpu")
		} else {
			log.Info("backend selected", "backend", "cuda")
		}
	default:
		return cleanup(fmt.Errorf("unknown backend %q (expected auto, cpu, or cuda)", backendName))
	}

	modelRuntime, err := runtime.LoadModel(mcfFile, cfgBytes, maxContext, opts)
	if err != nil {
		return cleanup(err)
	}
	if modelRuntime == nil {
		return cleanup(fmt.Errorf("backend %q returned nil model runtime", runtime.Name()))
	}

	var speculator *Speculator
	if draftFile != nil {
		spec, err := loadDSparkSpeculator(modelRuntime, draftFile, draftCfgBytes, maxContext, opts, l.DraftTokens, l.DraftConfidence)
		if err != nil {
			return cleanup(err)
		}
		speculator = spec
	}
	modelConfig := modelRuntime.ModelConfig()
	if modelConfig == nil {
		return cleanup(fmt.Errorf("backend %q returned nil model config", runtime.Name()))
	}

	genDefaults := parseHFGenerationDefaults(mcfFile.SectionData(mcf.SectionHFGenerationConfigJSON))

	tokJSON, tokCfg, err := l.loadTokenizerBytes(mcfFile)
	if err != nil {
		return cleanup(err)
	}

	hfTok, err := tokenizer.LoadHFTokenizerBytes(tokJSON, tokCfg)
	if err != nil {
		return cleanup(err)
	}

	tokCfgParsed, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSON, tokCfg)
	if err != nil {
		return cleanup(err)
	}
	tokCfgParsed.BOSTokenID = hfTok.BOSID()
	tokCfgParsed.AddBOS = hfTok.AddBOS()
	tokCfgParsed.EOSTokenID = hfTok.EOSID()
	effectiveTemplate, templateSource := ResolveChatTemplate(l.ChatTemplatePath, tokCfgParsed, modelConfig.Arch, cfgBytes)
	if strings.TrimSpace(effectiveTemplate) == "" {
		log.Info("chat template resolved", "model_path", modelPath, "source", "none")
	} else {
		log.Info("chat template resolved", "model_path", modelPath, "source", templateSource, "length", len(effectiveTemplate))
	}

	stopTokens := BuildStopTokens(tokCfgParsed, genDefaults.EOSTokenIDs)
	log.Debug("stop tokens resolved", "stop_tokens", stopTokens, "gen_eos_ids", genDefaults.EOSTokenIDs, "tokenizer_eos", tokCfgParsed.EOSTokenID)

	cfgBytesCopy := make([]byte, len(cfgBytes))
	copy(cfgBytesCopy, cfgBytes)

	engine := &EngineImpl{
		model:            modelRuntime,
		tokenizer:        hfTok,
		tokenizerConfig:  tokCfgParsed,
		arch:             modelConfig.Arch,
		hfConfigJSON:     cfgBytesCopy,
		chatTemplatePath: l.ChatTemplatePath,
		stopTokens:       stopTokens,
		mcfFile:          mcfFile,
		draftFile:        draftFile,
		speculator:       speculator,
		useGraph:         l.UseGraph,
	}

	return &LoadResult{
		Engine:             engine,
		Runtime:            modelRuntime,
		Tokenizer:          hfTok,
		TokenizerConfig:    tokCfgParsed,
		HFConfigJSON:       cfgBytesCopy,
		GenerationDefaults: genDefaults,
		Arch:               modelConfig.Arch,
	}, nil
}

// loadDSparkSpeculator pairs a loaded CPU target with its DSpark draft MCF and
// returns the Speculator the engine will drive. The target must already have
// captured the draft's target layers as hidden taps (set from the draft config
// before the target load).
func loadDSparkSpeculator(modelRuntime core.Runtime, draftFile *mcfstore.File, draftCfgBytes []byte, maxContext int, opts core.LoadModelOptions, draftTokens int, draftConfidence float64) (*Speculator, error) {
	si, ok := modelRuntime.(*simd.Instance)
	if !ok {
		return nil, fmt.Errorf("dspark speculation requires the simd cpu backend, got %T", modelRuntime)
	}
	if !si.CanBatchForward(model.DSparkMaxBlockSize + 1) {
		return nil, fmt.Errorf("target model cannot run batched verification blocks of %d tokens", model.DSparkMaxBlockSize+1)
	}
	draftOpts := opts
	draftOpts.HiddenTapLayers = nil
	draftOpts.GpuLayers = 0
	draft, err := simd.LoadDSparkDraftMCF(draftFile, draftCfgBytes, (*core.Instance)(si), maxContext, draftOpts)
	if err != nil {
		return nil, fmt.Errorf("load dspark draft: %w", err)
	}
	tokens := draftTokens
	if tokens <= 0 {
		tokens = draft.Config.BlockSize
	}
	if tokens > draft.Config.BlockSize {
		tokens = draft.Config.BlockSize
	}
	if tokens < 2 {
		return nil, fmt.Errorf("dspark draft block_size %d is below the minimum draft length 2", draft.Config.BlockSize)
	}
	return &Speculator{Draft: draft, Tokens: tokens, Confidence: draftConfidence}, nil
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
		Temperature       *float64        `json:"temperature"`
		TopK              *int            `json:"top_k"`
		TopP              *float64        `json:"top_p"`
		RepetitionPenalty *float64        `json:"repetition_penalty"`
		EOSTokenID        json.RawMessage `json:"eos_token_id"`
		DoSample          *bool           `json:"do_sample"`
	}
	if len(genBytes) == 0 {
		return GenDefaults{}
	}
	var cfg hfGenerationConfig
	if err := json.Unmarshal(genBytes, &cfg); err != nil {
		return GenDefaults{}
	}
	result := GenDefaults{
		Temperature:       cfg.Temperature,
		TopK:              cfg.TopK,
		TopP:              cfg.TopP,
		RepetitionPenalty: cfg.RepetitionPenalty,
		DoSample:          cfg.DoSample,
	}
	if len(cfg.EOSTokenID) > 0 {
		var ids []int
		if err := json.Unmarshal(cfg.EOSTokenID, &ids); err == nil {
			result.EOSTokenIDs = ids
		} else {
			var id int
			if err := json.Unmarshal(cfg.EOSTokenID, &id); err == nil {
				result.EOSTokenIDs = []int{id}
			}
		}
	}
	return result
}

func disableSWAConfig(cfgBytes []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(cfgBytes, &raw); err != nil {
		return nil, err
	}
	raw["sliding_window"] = 0
	raw["sliding_window_pattern"] = 0
	raw["global_attn_every_n_layers"] = 0
	if lt, ok := raw["layer_types"]; ok {
		if list, ok := lt.([]any); ok {
			for i := range list {
				list[i] = "full_attention"
			}
			raw["layer_types"] = list
		}
	}
	out, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	return out, nil
}
