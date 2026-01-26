package tokenizer

import (
	"strings"
)

// Message represents a chat message for template rendering.
type Message struct {
	Role    string
	Content string
}

// RenderPrompt renders messages using the model's chat template.
func RenderPrompt(cfg TokenizerConfig, msgs []Message, addGenerationPrompt bool) (string, bool) {
	return RenderPromptTemplate(cfg.ChatTemplate, cfg.TokenString(cfg.BOSTokenID), cfg.AddBOS, msgs, addGenerationPrompt)
}

// RenderPromptTemplate renders messages using a Jinja template string.
func RenderPromptTemplate(tpl, bosToken string, addBOS bool, msgs []Message, addGenerationPrompt bool) (string, bool) {
	if tpl == "" {
		return "", false
	}
	// Only handle the LFM2 ChatML-like template for now.
	if !strings.Contains(tpl, "<|im_start|>") || !strings.Contains(tpl, "<|im_end|>") {
		return "", false
	}
	if !strings.Contains(tpl, "messages") {
		return "", false
	}
	return renderLFM2TemplateFromString(tpl, bosToken, addBOS, msgs, addGenerationPrompt), true
}

func renderLFM2TemplateFromString(tpl string, bosToken string, addBOS bool, msgs []Message, addGenerationPrompt bool) string {
	var b strings.Builder

	// The template includes bos_token; avoid duplication if tokenizer already adds BOS.
	if strings.Contains(tpl, "bos_token") && !addBOS && bosToken != "" {
		b.WriteString(bosToken)
	}

	// Extract system prompt if first message is system.
	var systemPrompt string
	if len(msgs) > 0 && msgs[0].Role == "system" {
		systemPrompt = msgs[0].Content
		msgs = msgs[1:]
	}
	if systemPrompt != "" {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(systemPrompt)
		b.WriteString("<|im_end|>\n")
	}

	// Find last assistant index for keep_past_thinking handling.
	lastAssistant := -1
	for i, m := range msgs {
		if m.Role == "assistant" {
			lastAssistant = i
		}
	}

	for i, m := range msgs {
		content := m.Content
		// Apply the template's keep_past_thinking default (false).
		if m.Role == "assistant" && i != lastAssistant {
			if cut := strings.LastIndex(content, "</think>"); cut >= 0 {
				content = strings.TrimSpace(content[cut+len("</think>"):])
			}
		}
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}
	if addGenerationPrompt && strings.Contains(tpl, "add_generation_prompt") {
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String()
}
