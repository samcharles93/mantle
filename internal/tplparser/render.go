package tplparser

import "strings"

// Render returns (output, ok). ok=false means the template is unsupported.
func Render(opts RenderOptions) (string, bool, error) {
	if opts.Template == "" {
		return renderByArchDefault(opts)
	}
	if out, ok, err := renderByArch(opts); ok || err != nil {
		return out, ok, err
	}
	if out, ok, err := renderByTemplateSignature(opts); ok || err != nil {
		return out, ok, err
	}
	return "", false, nil
}

func renderByArchDefault(opts RenderOptions) (string, bool, error) {
	switch strings.ToLower(strings.TrimSpace(opts.Arch)) {
	case "lfm2":
		return renderLFM2(opts)
	case "gemma":
		return renderFunctionGemma(opts)
	case "qwen3":
		return renderQwen3(opts)
	case "mistral3":
		return renderMistral3(opts)
	default:
		return "", false, nil
	}
}

func renderByArch(opts RenderOptions) (string, bool, error) {
	switch strings.ToLower(strings.TrimSpace(opts.Arch)) {
	case "lfm2":
		return renderLFM2(opts)
	case "gemma":
		return renderFunctionGemma(opts)
	case "qwen3":
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
	case strings.Contains(tpl, "<start_of_turn>") && strings.Contains(tpl, "<start_function_declaration>"):
		return renderFunctionGemma(opts)
	case strings.Contains(tpl, "[SYSTEM_PROMPT]") && strings.Contains(tpl, "[INST]"):
		return renderMistral3(opts)
	case strings.Contains(tpl, "<tools>") || strings.Contains(tpl, "<tool_call>"):
		return renderQwen3(opts)
	case strings.Contains(tpl, "<|im_start|>") && strings.Contains(tpl, "<|im_end|>") && strings.Contains(tpl, "messages"):
		return renderLFM2(opts)
	default:
		return "", false, nil
	}
}
