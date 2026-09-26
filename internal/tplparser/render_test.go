package tplparser

import (
	"strings"
	"testing"
)

func TestRenderArchDefaultChatML(t *testing.T) {
	t.Parallel()

	out, ok, err := Render(RenderOptions{
		Arch:                "lfm2",
		BOSToken:            "<s>",
		AddBOS:              false,
		AddGenerationPrompt: true,
		Messages: []Message{
			{Role: "user", Content: "hello"},
		},
	})
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !ok {
		t.Fatalf("expected renderer match")
	}
	if !strings.Contains(out, "<|im_start|>user\nhello<|im_end|>\n") {
		t.Fatalf("unexpected output: %q", out)
	}
	if !strings.HasPrefix(out, "<s>") {
		t.Fatalf("expected BOS prefix in output: %q", out)
	}
	if !strings.HasSuffix(out, "<|im_start|>assistant\n") {
		t.Fatalf("expected generation prompt suffix: %q", out)
	}
}

func TestRenderTemplateSignatureFallback(t *testing.T) {
	t.Parallel()

	out, ok, err := Render(RenderOptions{
		Arch:                "unknown",
		Template:            "<|im_start|>{{ messages }}<|im_end|>",
		AddGenerationPrompt: false,
		Messages: []Message{
			{Role: "assistant", Content: "ok"},
		},
	})
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !ok {
		t.Fatalf("expected template signature match")
	}
	if !strings.Contains(out, "<|im_start|>assistant\nok<|im_end|>\n") {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestRenderArchGemma4(t *testing.T) {
	t.Parallel()

	out, ok, err := Render(RenderOptions{
		Arch:                "gemma4",
		BOSToken:            "<bos>",
		AddBOS:              false,
		AddGenerationPrompt: true,
		Messages: []Message{
			{Role: "system", Content: "rules"},
			{Role: "user", Content: "hello"},
			{
				Role:    "assistant",
				Content: "ok",
				ToolCalls: []ToolCall{
					{
						Function: ToolCallFunction{
							Name:      "lookup",
							Arguments: map[string]any{"q": "hi"},
						},
					},
				},
			},
			{Role: "tool", Name: "lookup", Content: map[string]any{"result": "done"}},
		},
	})
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !ok {
		t.Fatalf("expected renderer match")
	}
	if !strings.HasPrefix(out, "<bos><|turn>system\nrules\n<turn|>\n") {
		t.Fatalf("unexpected prefix: %q", out)
	}
	if !strings.Contains(out, "<|turn>model\nok<|tool_call>call:lookup{{q:<escape>hi<escape>}}<tool_call|>\n<turn|>\n") {
		t.Fatalf("missing gemma4 tool call rendering: %q", out)
	}
	if !strings.Contains(out, "<|turn>user\n<|tool_response>response:lookup{result:<escape>done<escape>}<tool_response|>\n<turn|>\n") {
		t.Fatalf("missing gemma4 tool response rendering: %q", out)
	}
	if !strings.HasSuffix(out, "<|turn>model\n") {
		t.Fatalf("expected generation prompt suffix: %q", out)
	}
}

func TestRenderUnsupported(t *testing.T) {
	t.Parallel()

	out, ok, err := Render(RenderOptions{
		Arch:     "unknown",
		Template: "unsupported-template",
		Messages: []Message{{Role: "user", Content: "x"}},
	})
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if ok {
		t.Fatalf("expected ok=false for unsupported template, got true with output %q", out)
	}
	if out != "" {
		t.Fatalf("expected empty output for unsupported template, got %q", out)
	}
}

// MiniCPM5's ChatML template opens with "{{- bos_token }}" and also mentions
// <tools>, so signature dispatch sends it to the Qwen3 renderer, which never
// writes a BOS. The template's explicit leading bos_token has to be honoured
// separately, and a template that does not ask for one must stay untouched.
func TestRenderHonoursLeadingBOSToken(t *testing.T) {
	t.Parallel()

	const bos = "<s>"
	tpl := "{{- bos_token }}{%- if tools %}<tools></tools>{%- endif %}" +
		"{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}" +
		"{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

	out, ok, err := Render(RenderOptions{
		Template:            tpl,
		BOSToken:            bos,
		AddGenerationPrompt: true,
		Messages:            []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("render error: %v", err)
	}
	if !ok {
		t.Fatalf("expected a renderer match")
	}
	if !strings.HasPrefix(out, bos) {
		t.Fatalf("expected BOS prefix, got %q", out)
	}

	// A template that opens with bos_token but dispatches to the ChatML renderer
	// (which already writes it when add_bos_token is false) must not be doubled.
	chatml := "{{- bos_token }}{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
	once, ok, err := Render(RenderOptions{
		Template: chatml,
		BOSToken: bos,
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil || !ok {
		t.Fatalf("render error: %v ok=%v", err, ok)
	}
	if n := strings.Count(once, bos); n != 1 {
		t.Fatalf("expected exactly one BOS, got %d in %q", n, once)
	}

	// A template that does not ask for a leading bos_token must gain none, even
	// though a BOS token is configured.
	plain := "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
	noBOS, ok, err := Render(RenderOptions{
		Template: plain,
		BOSToken: bos,
		AddBOS:   true,
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil || !ok {
		t.Fatalf("render error: %v ok=%v", err, ok)
	}
	if strings.Contains(noBOS, bos) {
		t.Fatalf("template without a leading bos_token must not gain one: %q", noBOS)
	}
}
