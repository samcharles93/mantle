package tokenizer

import "github.com/samcharles93/mantle/internal/tplparser"

// RenderPrompt renders messages using the model's chat template.
func RenderPrompt(cfg TokenizerConfig, arch string, msgs []Message, tools []any, addGenerationPrompt bool) (string, bool, error) {
	return RenderPromptTemplate(cfg.ChatTemplate, cfg.TokenString(cfg.BOSTokenID), cfg.TokenString(cfg.EOSTokenID), cfg.AddBOS, msgs, tools, addGenerationPrompt, arch)
}

// RenderPromptTemplate renders messages using a Jinja template string.
func RenderPromptTemplate(tpl, bosToken, eosToken string, addBOS bool, msgs []Message, tools []any, addGenerationPrompt bool, arch string) (string, bool, error) {
	parserMsgs := make([]tplparser.Message, 0, len(msgs))
	for _, msg := range msgs {
		parserMsgs = append(parserMsgs, tplparser.Message{
			Role:      msg.Role,
			Content:   msg.Content,
			Name:      msg.Name,
			ToolCalls: toTplToolCalls(msg.ToolCalls),
		})
	}
	return tplparser.Render(tplparser.RenderOptions{
		Template:            tpl,
		Arch:                arch,
		BOSToken:            bosToken,
		EOSToken:            eosToken,
		AddBOS:              addBOS,
		AddGenerationPrompt: addGenerationPrompt,
		Messages:            parserMsgs,
		Tools:               tools,
	})
}

func toTplToolCalls(calls []ToolCall) []tplparser.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]tplparser.ToolCall, 0, len(calls))
	for _, call := range calls {
		out = append(out, tplparser.ToolCall{
			ID:   call.ID,
			Type: call.Type,
			Function: tplparser.ToolCallFunction{
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			},
		})
	}
	return out
}
