package inference

import "github.com/samcharles93/mantle/internal/tokenizer"

type RequestOptions struct {
	Messages []tokenizer.Message
	Tools    []any

	Steps *int
	Seed  *int64

	Temperature   *float64
	TopK          *int
	TopP          *float64
	MinP          *float64
	RepeatPenalty *float64
	RepeatLastN   *int

	NoTemplate *bool
	EchoPrompt *bool

	ReasoningFormat *string
	ReasoningBudget *int
}

func ResolveRequest(opts RequestOptions, defaults GenDefaults) Request {
	req := Request{
		Messages:      opts.Messages,
		Tools:         opts.Tools,
		Steps:         -1,
		Seed:          -1,
		Temperature:   0.8,
		TopK:          40,
		TopP:          0.95,
		MinP:          0.0,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		NoTemplate:    false,
		EchoPrompt:    false,
		ReasoningFormat: "auto",
		ReasoningBudget: -1,
	}

	if defaults.Temperature != nil && *defaults.Temperature > 0 {
		req.Temperature = *defaults.Temperature
	}
	if defaults.TopK != nil && *defaults.TopK > 0 {
		req.TopK = *defaults.TopK
	}
	if defaults.TopP != nil && *defaults.TopP > 0 && *defaults.TopP <= 1 {
		req.TopP = *defaults.TopP
	}
	if defaults.RepetitionPenalty != nil && *defaults.RepetitionPenalty > 0 {
		req.RepeatPenalty = *defaults.RepetitionPenalty
	}

	if opts.Steps != nil {
		req.Steps = *opts.Steps
	}
	if opts.Seed != nil {
		req.Seed = *opts.Seed
	}
	if opts.Temperature != nil {
		req.Temperature = *opts.Temperature
	}
	if opts.TopK != nil {
		req.TopK = *opts.TopK
	}
	if opts.TopP != nil {
		req.TopP = *opts.TopP
	}
	if opts.MinP != nil {
		req.MinP = *opts.MinP
	}
	if opts.RepeatPenalty != nil {
		req.RepeatPenalty = *opts.RepeatPenalty
	}
	if opts.RepeatLastN != nil {
		req.RepeatLastN = *opts.RepeatLastN
	}
	if opts.NoTemplate != nil {
		req.NoTemplate = *opts.NoTemplate
	}
	if opts.EchoPrompt != nil {
		req.EchoPrompt = *opts.EchoPrompt
	}
	if opts.ReasoningFormat != nil {
		req.ReasoningFormat = *opts.ReasoningFormat
	}
	if opts.ReasoningBudget != nil {
		req.ReasoningBudget = *opts.ReasoningBudget
	}

	return req
}
