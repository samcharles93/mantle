package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
)

// compareCmd measures how far a model's next-token distributions are from a
// reference model's over the same text: KL divergence at every position,
// top-1 agreement, and both perplexities. It is the quality metric for
// quantized containers (compare a quantized .mcf against its bf16 source).
func compareCmd() *cli.Command {
	var (
		refPath, modelPath, textFile, jsonOut, compareBackend string
		maxTokens                                             int64
	)
	return &cli.Command{
		Name:  "compare",
		Usage: "Compare a model's next-token distributions with a reference model over a text (KL, top-1 agreement, perplexity)",
		Flags: []cli.Flag{
			&cli.StringFlag{Name: "ref", Usage: "reference .mcf (for example the bf16 source)", Required: true, Destination: &refPath},
			&cli.StringFlag{Name: "model", Aliases: []string{"m"}, Usage: ".mcf to evaluate", Required: true, Destination: &modelPath},
			&cli.StringFlag{Name: "text-file", Usage: "plain text to evaluate over (no chat template)", Required: true, Destination: &textFile},
			&cli.Int64Flag{Name: "max-tokens", Usage: "evaluate at most this many tokens of the text", Value: 512, Destination: &maxTokens},
			&cli.StringFlag{Name: "backend", Usage: "backend for both models: cpu|cuda", Value: "cpu", Destination: &compareBackend},
			&cli.StringFlag{Name: "json", Usage: "also write the result as JSON to this path", Destination: &jsonOut},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			log := logger.FromContext(ctx)
			text, err := os.ReadFile(textFile)
			if err != nil {
				return fmt.Errorf("compare: %w", err)
			}
			if maxTokens < 2 {
				return errors.New("compare: --max-tokens must be at least 2")
			}

			log.Info("loading reference", "path", refPath)
			refLogProbs, tokens, err := referenceLogProbs(ctx, refPath, compareBackend, string(text), int(maxTokens))
			if err != nil {
				return fmt.Errorf("compare: reference: %w", err)
			}
			log.Info("loading model", "path", modelPath, "positions", len(refLogProbs))
			stats, err := compareAgainst(ctx, modelPath, compareBackend, string(text), tokens, refLogProbs)
			if err != nil {
				return fmt.Errorf("compare: model: %w", err)
			}

			fmt.Printf("positions      %d\n", stats.Positions)
			fmt.Printf("mean KL        %.5f\n", stats.MeanKL)
			fmt.Printf("median KL      %.5f\n", stats.MedianKL)
			fmt.Printf("p99 KL         %.5f\n", stats.P99KL)
			fmt.Printf("max KL         %.5f\n", stats.MaxKL)
			fmt.Printf("top-1 agree    %.2f%%\n", 100*stats.Top1Agree)
			fmt.Printf("ref PPL        %.4f\n", stats.RefPPL)
			fmt.Printf("model PPL      %.4f\n", stats.ModelPPL)
			if jsonOut != "" {
				raw, err := json.MarshalIndent(stats, "", "  ")
				if err != nil {
					return err
				}
				if err := os.WriteFile(jsonOut, append(raw, '\n'), 0o644); err != nil {
					return fmt.Errorf("compare: %w", err)
				}
			}
			return nil
		},
	}
}

// compareStats summarises per-position divergence from the reference.
type compareStats struct {
	Positions int     `json:"positions"`
	MeanKL    float64 `json:"mean_kl"`
	MedianKL  float64 `json:"median_kl"`
	P99KL     float64 `json:"p99_kl"`
	MaxKL     float64 `json:"max_kl"`
	Top1Agree float64 `json:"top1_agree"`
	RefPPL    float64 `json:"ref_ppl"`
	ModelPPL  float64 `json:"model_ppl"`
}

// positionLogProbs holds a reference model's log-probabilities after one
// prefix, with its argmax.
type positionLogProbs struct {
	logProbs []float32
	argmax   int
}

func loadForCompare(ctx context.Context, path, backendName string, maxContext int) (*inference.LoadResult, error) {
	loader := inference.Loader{Backend: backendName}
	loader.LoadOptions.GpuLayers = -1
	return loader.Load(ctx, path, maxContext)
}

// referenceLogProbs tokenizes text with the reference model, runs it over the
// first maxTokens tokens and keeps the log-probabilities predicting tokens
// 1..n-1.
func referenceLogProbs(ctx context.Context, path, backendName, text string, maxTokens int) ([]positionLogProbs, []int, error) {
	res, err := loadForCompare(ctx, path, backendName, maxTokens+1)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = res.Engine.Close() }()
	tokens, err := res.Tokenizer.Encode(text)
	if err != nil {
		return nil, nil, err
	}
	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}
	if len(tokens) < 2 {
		return nil, nil, errors.New("text has fewer than two tokens")
	}
	res.Runtime.Reset()
	out := make([]positionLogProbs, 0, len(tokens)-1)
	for _, tok := range tokens[:len(tokens)-1] {
		logits, err := res.Runtime.ForwardToken(tok)
		if err != nil {
			return nil, nil, err
		}
		lp := make([]float32, len(logits))
		out = append(out, positionLogProbs{logProbs: lp, argmax: logSoftmaxInto(lp, logits)})
	}
	return out, tokens, nil
}

// compareAgainst runs the candidate over the reference's tokens and scores
// every position against the stored reference distributions.
func compareAgainst(ctx context.Context, path, backendName, text string, tokens []int, ref []positionLogProbs) (compareStats, error) {
	res, err := loadForCompare(ctx, path, backendName, len(tokens)+1)
	if err != nil {
		return compareStats{}, err
	}
	defer func() { _ = res.Engine.Close() }()
	own, err := res.Tokenizer.Encode(text)
	if err != nil {
		return compareStats{}, err
	}
	if len(own) < len(tokens) || !slices.Equal(own[:len(tokens)], tokens) {
		return compareStats{}, errors.New("tokenizers disagree on the text; models are not comparable")
	}
	res.Runtime.Reset()
	kls := make([]float64, 0, len(ref))
	var agree int
	var refNLL, modelNLL float64
	lp := make([]float32, 0)
	for i, tok := range tokens[:len(tokens)-1] {
		logits, err := res.Runtime.ForwardToken(tok)
		if err != nil {
			return compareStats{}, err
		}
		if len(logits) != len(ref[i].logProbs) {
			return compareStats{}, fmt.Errorf("vocab size %d differs from reference %d", len(logits), len(ref[i].logProbs))
		}
		lp = slices.Grow(lp[:0], len(logits))[:len(logits)]
		argmax := logSoftmaxInto(lp, logits)
		kls = append(kls, klDivergence(ref[i].logProbs, lp))
		if argmax == ref[i].argmax {
			agree++
		}
		next := tokens[i+1]
		refNLL -= float64(ref[i].logProbs[next])
		modelNLL -= float64(lp[next])
	}
	return summarizeCompare(kls, agree, refNLL, modelNLL), nil
}

// logSoftmaxInto writes log(softmax(logits)) into dst and returns the argmax.
func logSoftmaxInto(dst, logits []float32) int {
	argmax := 0
	for i, v := range logits {
		if v > logits[argmax] {
			argmax = i
		}
	}
	m := float64(logits[argmax])
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v) - m)
	}
	logZ := m + math.Log(sum)
	for i, v := range logits {
		dst[i] = float32(float64(v) - logZ)
	}
	return argmax
}

// klDivergence returns KL(p || q) for log-probability vectors.
func klDivergence(logP, logQ []float32) float64 {
	var kl float64
	for i, lp := range logP {
		p := math.Exp(float64(lp))
		if p == 0 {
			continue
		}
		kl += p * (float64(lp) - float64(logQ[i]))
	}
	return max(kl, 0)
}

func summarizeCompare(kls []float64, agree int, refNLL, modelNLL float64) compareStats {
	n := len(kls)
	if n == 0 {
		return compareStats{}
	}
	sorted := slices.Clone(kls)
	slices.Sort(sorted)
	var sum float64
	for _, v := range kls {
		sum += v
	}
	pct := func(p float64) float64 {
		return sorted[min(n-1, int(math.Ceil(p*float64(n)))-1)]
	}
	return compareStats{
		Positions: n,
		MeanKL:    sum / float64(n),
		MedianKL:  pct(0.5),
		P99KL:     pct(0.99),
		MaxKL:     sorted[n-1],
		Top1Agree: float64(agree) / float64(n),
		RefPPL:    math.Exp(refNLL / float64(n)),
		ModelPPL:  math.Exp(modelNLL / float64(n)),
	}
}
