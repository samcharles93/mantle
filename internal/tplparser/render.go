package tplparser

import "strings"

// Render returns (output, ok). ok=false means the template is unsupported.
func Render(opts RenderOptions) (string, bool, error) {
	out, ok, err := render(opts)
	if !ok || err != nil {
		return out, ok, err
	}
	return ensureLeadingBOS(out, opts), true, nil
}

func render(opts RenderOptions) (string, bool, error) {
	if opts.Template == "" {
		return renderByArchDefault(opts)
	}
	if out, ok, err := renderByArchDefault(opts); ok || err != nil {
		return out, ok, err
	}
	if out, ok, err := renderByTemplateSignature(opts); ok || err != nil {
		return out, ok, err
	}
	return "", false, nil
}

// ensureLeadingBOS prepends the tokenizer's BOS token when the source template
// emits it as its very first content but the selected renderer did not write it.
//
// Renderers are selected by architecture/signature and re-implement the template
// in Go rather than evaluating the Jinja source, so a template's explicit leading
// BOS has to be honoured separately. MiniCPM5 is the motivating case: it is a
// Llama-family model whose ChatML template starts with "{{- bos_token }}" and
// mentions <tools>, so it dispatches to renderQwen3, which does not emit BOS.
// Templates that do not open with bos_token are left completely untouched.
func ensureLeadingBOS(out string, opts RenderOptions) string {
	if opts.BOSToken == "" || !templateWantsLeadingBOS(opts.Template) {
		return out
	}
	if strings.HasPrefix(out, opts.BOSToken) {
		return out
	}
	return opts.BOSToken + out
}

// templateWantsLeadingBOS reports whether the template's first emitted content is
// its bos_token, e.g. "{{- bos_token }}{%- if tools %}".
func templateWantsLeadingBOS(tpl string) bool {
	t := strings.TrimSpace(tpl)
	if !strings.HasPrefix(t, "{{") {
		return false
	}
	end := strings.Index(t, "}}")
	if end < 0 {
		return false
	}
	expr := strings.TrimSpace(t[2:end])
	expr = strings.TrimSpace(strings.Trim(expr, "-"))
	return expr == "bos_token"
}

func renderByArchDefault(opts RenderOptions) (string, bool, error) {
	switch strings.ToLower(strings.TrimSpace(opts.Arch)) {
	case "lfm2":
		return renderChatML(opts)
	case "gemma":
		return renderGemma3(opts)
	case "gemma3_text", "gemma3", "gemma3n_text", "gemma3n":
		return renderGemma3(opts)
	case "gemma4":
		return renderGemma4(opts)
	case "qwen3", "qwen3_5":
		return renderQwen3(opts)
	case "mistral3":
		return renderMistral3(opts)
	default:
		return "", false, nil
	}
}

func renderByTemplateSignature(opts RenderOptions) (string, bool, error) {
	tpl := opts.Template
	switch {
	case strings.Contains(tpl, "<|turn>") && strings.Contains(tpl, "<turn|>"):
		return renderGemma4(opts)
	case strings.Contains(tpl, "<start_of_turn>") && strings.Contains(tpl, "<start_function_declaration>"):
		return renderGemma3(opts)
	case strings.Contains(tpl, "[SYSTEM_PROMPT]") && strings.Contains(tpl, "[INST]"):
		return renderMistral3(opts)
	case strings.Contains(tpl, "<tools>") || strings.Contains(tpl, "<tool_call>"):
		return renderQwen3(opts)
	case strings.Contains(tpl, "<|im_start|>") && strings.Contains(tpl, "<|im_end|>") && strings.Contains(tpl, "messages"):
		return renderChatML(opts)
	default:
		return "", false, nil
	}
}
