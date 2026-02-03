package inference

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

type PromptRenderInput struct {
	TemplateOverride    string
	TokenizerConfig     tokenizer.TokenizerConfig
	Arch                string
	HFConfigJSON        []byte
	Messages            []tokenizer.Message
	Tools               []any
	AddGenerationPrompt bool
	NoTemplate          bool
}

func RenderPrompt(input PromptRenderInput) (string, error) {
	if input.NoTemplate {
		return lastUserText(input.Messages)
	}

	effectiveTemplate, _ := ResolveChatTemplate(input.TemplateOverride, input.TokenizerConfig, input.Arch, input.HFConfigJSON)
	if effectiveTemplate == "" {
		return lastUserText(input.Messages)
	}

	bosToken := input.TokenizerConfig.TokenString(input.TokenizerConfig.BOSTokenID)
	rendered, ok, err := tokenizer.RenderPromptTemplate(
		effectiveTemplate,
		bosToken,
		input.TokenizerConfig.TokenString(input.TokenizerConfig.EOSTokenID),
		input.TokenizerConfig.AddBOS,
		input.Messages,
		input.Tools,
		input.AddGenerationPrompt,
		input.Arch,
	)
	if err == nil && ok {
		return rendered, nil
	}
	return lastUserText(input.Messages)
}

func lastUserText(msgs []tokenizer.Message) (string, error) {
	if len(msgs) == 0 {
		return "", nil
	}
	msg := msgs[len(msgs)-1]
	if text, ok := tokenizer.MessageText(msg); ok {
		return text, nil
	}
	return "", fmt.Errorf("prompt content is not a string and no template renderer matched")
}
