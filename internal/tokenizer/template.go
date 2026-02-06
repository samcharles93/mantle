package tokenizer

import "github.com/samcharles93/mantle/internal/tplparser"

// RenderPromptTemplate renders messages using a Jinja template string.
func RenderPromptTemplate(tpl, bosToken, eosToken string, addBOS bool, msgs []Message, tools []any, addGenerationPrompt bool, arch string) (string, bool, error) {
	return tplparser.Render(tplparser.RenderOptions{
		Template:            tpl,
		Arch:                arch,
		BOSToken:            bosToken,
		EOSToken:            eosToken,
		AddBOS:              addBOS,
		AddGenerationPrompt: addGenerationPrompt,
		Messages:            msgs,
		Tools:               tools,
	})
}
