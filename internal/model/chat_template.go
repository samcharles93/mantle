package model

import "strings"

const lfm2ChatTemplate = `{{ bos_token }}{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}`

// InferChatTemplate returns a runtime-selected chat template when one is not embedded.
// The runtime decides whether to use this fallback.
func InferChatTemplate(arch string, hfConfigJSON []byte) (string, bool) {
	_ = hfConfigJSON
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "lfm2":
		return lfm2ChatTemplate, true
	default:
		return "", false
	}
}
