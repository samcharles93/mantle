package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/simd"
	clipaths "github.com/samcharles93/mantle/internal/cli/paths"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/tokenizer"
	"github.com/urfave/cli/v3"
)

func traceCmd() *cli.Command {
	var (
		prompt     string
		system     string
		noTemplate bool
		outPath    string
		summaryOut string
		hfConfig   string
	)

	flags := append([]cli.Flag{}, commonModelFlags()...)
	flags = append(flags, commonTokenizerFlags()...)
	flags = append(flags,
		&cli.StringFlag{
			Name:        "prompt",
			Usage:       "prompt text to trace",
			Destination: &prompt,
		},
		&cli.StringFlag{
			Name:        "system",
			Usage:       "optional system prompt",
			Destination: &system,
		},
		&cli.BoolFlag{
			Name:        "no-template",
			Usage:       "disable chat template rendering",
			Destination: &noTemplate,
		},
		&cli.StringFlag{
			Name:        "out",
			Usage:       "output trace JSON path",
			Value:       "traces/mantle-trace.json",
			Destination: &outPath,
		},
		&cli.StringFlag{
			Name:        "summary-out",
			Usage:       "optional summary JSON path",
			Destination: &summaryOut,
		},
		&cli.StringFlag{
			Name:        "hf-config",
			Usage:       "override path to config.json",
			Destination: &hfConfig,
		},
	)

	return &cli.Command{
		Name:  "trace",
		Usage: "Trace prompt activations for Mantle runtime debugging",
		Flags: flags,
		Action: func(ctx context.Context, c *cli.Command) error {
			log := logger.FromContext(ctx)
			if strings.TrimSpace(prompt) == "" {
				return cli.Exit("error: --prompt is required", 1)
			}
			if !c.IsSet("backend") {
				backend = "cpu"
			}

			resolvedModelPath, err := clipaths.ResolveRunModelPath(modelPath, modelsPath, os.Stdin, os.Stderr)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: resolve model: %v", err), 1)
			}
			modelPath = resolvedModelPath

			stat, err := os.Stat(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: stat model path %q: %v", modelPath, err), 1)
			}
			if stat.IsDir() || !strings.HasSuffix(strings.ToLower(modelPath), ".mcf") {
				return cli.Exit("error: mantle trace only supports .mcf files", 1)
			}

			log.Info("loading MCF model", "path", modelPath)
			loader := inference.Loader{
				TokenizerJSONPath:   tokenizerJSONPath,
				TokenizerConfigPath: tokenizerConfig,
				ChatTemplatePath:    chatTemplate,
				HFConfigPath:        hfConfig,
				Backend:             backend,
			}
			loadResult, err := loader.Load(ctx, modelPath, int(maxContext))
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: load mcf model: %v", err), 1)
			}
			defer func() { _ = loadResult.Engine.Close() }()

			runtimeModel, ok := loadResult.Runtime.(*simd.Instance)
			if !ok {
				return cli.Exit(fmt.Sprintf("error: trace requires CPU SIMD runtime, got %T", loadResult.Runtime), 1)
			}

			messages := make([]tokenizer.Message, 0, 2)
			if system != "" {
				messages = append(messages, tokenizer.Message{Role: "system", Content: system})
			}
			messages = append(messages, tokenizer.Message{Role: "user", Content: prompt})

			rendered, err := inference.RenderPrompt(inference.PromptRenderInput{
				TemplateOverride:    chatTemplate,
				TokenizerConfig:     loadResult.TokenizerConfig,
				Arch:                loadResult.Arch,
				HFConfigJSON:        loadResult.HFConfigJSON,
				Messages:            messages,
				AddGenerationPrompt: true,
				NoTemplate:          noTemplate,
			})
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: render prompt: %v", err), 1)
			}
			tokenIDs, err := loadResult.Tokenizer.Encode(rendered)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: encode prompt: %v", err), 1)
			}

			trace, err := runtimeModel.TraceTokens(tokenIDs)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: trace prompt: %v", err), 1)
			}
			trace.Prompt = prompt
			trace.RenderedPrompt = rendered
			trace.Tokens = make([]string, len(tokenIDs))
			for i, id := range tokenIDs {
				trace.Tokens[i] = loadResult.Tokenizer.TokenString(id)
			}
			if nextToken, err := loadResult.Tokenizer.Decode([]int{trace.NextTokenID}); err == nil {
				trace.NextToken = nextToken
			}

			if err := writeJSON(outPath, trace); err != nil {
				return cli.Exit(fmt.Sprintf("error: write trace: %v", err), 1)
			}
			if summaryOut == "" {
				summaryOut = deriveSummaryPath(outPath)
			}
			if err := writeJSON(summaryOut, buildTraceSummary(modelPath, trace)); err != nil {
				return cli.Exit(fmt.Sprintf("error: write summary: %v", err), 1)
			}

			log.Info("trace saved",
				"trace_path", outPath,
				"summary_path", summaryOut,
				"prompt_tokens", len(tokenIDs),
				"next_token_id", trace.NextTokenID,
				"next_token", trace.NextToken,
			)
			return nil
		},
	}
}

func writeJSON(path string, v any) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("empty output path")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func deriveSummaryPath(tracePath string) string {
	ext := filepath.Ext(tracePath)
	if ext == "" {
		return tracePath + ".summary.json"
	}
	return strings.TrimSuffix(tracePath, ext) + ".summary.json"
}

type traceSummary struct {
	ModelPath               string        `json:"model_path"`
	Prompt                  string        `json:"prompt"`
	RenderedPrompt          string        `json:"rendered_prompt"`
	TokenIDs                []int         `json:"token_ids"`
	Tokens                  []string      `json:"tokens"`
	NextTokenID             int           `json:"next_token_id"`
	NextToken               string        `json:"next_token"`
	InputsEmbeds            tensorStats   `json:"inputs_embeds"`
	HiddenStates            []tensorStats `json:"hidden_states"`
	AttentionOutputs        []tensorStats `json:"attention_outputs,omitempty"`
	FfnOutputs              []tensorStats `json:"ffn_outputs,omitempty"`
	PostFfnHiddenStates     []tensorStats `json:"post_ffn_hidden_states,omitempty"`
	PerLayerResidualOutputs []tensorStats `json:"per_layer_residual_outputs,omitempty"`
	PerLayerInputsRaw       *tensorStats  `json:"per_layer_inputs_raw,omitempty"`
	PerLayerInputsProjected *tensorStats  `json:"per_layer_inputs_projected,omitempty"`
	LastTokenLogits         tensorStats   `json:"last_token_logits"`
}

type tensorStats struct {
	Shape []int   `json:"shape"`
	Mean  float64 `json:"mean"`
	Std   float64 `json:"std"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
	L2    float64 `json:"l2"`
}

func buildTraceSummary(modelPath string, trace *simd.PromptTrace) traceSummary {
	summary := traceSummary{
		ModelPath:           modelPath,
		Prompt:              trace.Prompt,
		RenderedPrompt:      trace.RenderedPrompt,
		TokenIDs:            append([]int(nil), trace.TokenIDs...),
		Tokens:              append([]string(nil), trace.Tokens...),
		NextTokenID:         trace.NextTokenID,
		NextToken:           trace.NextToken,
		InputsEmbeds:        stats2D(trace.InputsEmbeds),
		HiddenStates:        make([]tensorStats, len(trace.HiddenStates)),
		AttentionOutputs:    make([]tensorStats, len(trace.AttentionOutputs)),
		FfnOutputs:          make([]tensorStats, len(trace.FfnOutputs)),
		PostFfnHiddenStates: make([]tensorStats, len(trace.PostFfnHiddenStates)),
		LastTokenLogits:     stats1D(trace.LastTokenLogits),
	}
	for i := range trace.HiddenStates {
		summary.HiddenStates[i] = stats2D(trace.HiddenStates[i])
	}
	for i := range trace.AttentionOutputs {
		summary.AttentionOutputs[i] = stats2D(trace.AttentionOutputs[i])
	}
	for i := range trace.FfnOutputs {
		summary.FfnOutputs[i] = stats2D(trace.FfnOutputs[i])
	}
	for i := range trace.PostFfnHiddenStates {
		summary.PostFfnHiddenStates[i] = stats2D(trace.PostFfnHiddenStates[i])
	}
	if len(trace.PerLayerResidualOutputs) > 0 {
		summary.PerLayerResidualOutputs = make([]tensorStats, len(trace.PerLayerResidualOutputs))
		for i := range trace.PerLayerResidualOutputs {
			summary.PerLayerResidualOutputs[i] = stats2D(trace.PerLayerResidualOutputs[i])
		}
	}
	if len(trace.PerLayerInputsRaw) > 0 {
		stats := stats3D(trace.PerLayerInputsRaw)
		summary.PerLayerInputsRaw = &stats
	}
	if len(trace.PerLayerInputsProjected) > 0 {
		stats := stats3D(trace.PerLayerInputsProjected)
		summary.PerLayerInputsProjected = &stats
	}
	return summary
}

func stats1D(values []float32) tensorStats {
	return statsFlat(values, []int{len(values)})
}

func stats2D(values [][]float32) tensorStats {
	shape := []int{len(values), 0}
	if len(values) > 0 {
		shape[1] = len(values[0])
	}
	flat := make([]float32, 0, shape[0]*shape[1])
	for _, row := range values {
		flat = append(flat, row...)
	}
	return statsFlat(flat, shape)
}

func stats3D(values [][][]float32) tensorStats {
	shape := []int{len(values), 0, 0}
	if len(values) > 0 {
		shape[1] = len(values[0])
	}
	if len(values) > 0 && len(values[0]) > 0 {
		shape[2] = len(values[0][0])
	}
	flat := make([]float32, 0, shape[0]*max(shape[1], 1)*max(shape[2], 1))
	for _, plane := range values {
		for _, row := range plane {
			flat = append(flat, row...)
		}
	}
	return statsFlat(flat, shape)
}

func statsFlat(values []float32, shape []int) tensorStats {
	if len(values) == 0 {
		return tensorStats{Shape: slices.Clone(shape)}
	}
	minV := float64(values[0])
	maxV := minV
	var sum float64
	for _, v := range values {
		fv := float64(v)
		sum += fv
		minV = min(minV, fv)
		maxV = max(maxV, fv)
	}
	mean := sum / float64(len(values))
	var sumSq float64
	for _, v := range values {
		d := float64(v) - mean
		sumSq += d * d
	}
	var l2 float64
	for _, v := range values {
		fv := float64(v)
		l2 += fv * fv
	}
	return tensorStats{
		Shape: slices.Clone(shape),
		Mean:  mean,
		Std:   math.Sqrt(sumSq / float64(len(values))),
		Min:   minV,
		Max:   maxV,
		L2:    math.Sqrt(l2),
	}
}

var _ core.Model = (*simd.Instance)(nil)
