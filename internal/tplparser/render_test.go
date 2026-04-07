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
	if !strings.HasPrefix(out, "<bos><|turn>system\nrules<turn|>\n") {
		t.Fatalf("unexpected prefix: %q", out)
	}
	if !strings.Contains(out, "<|turn>model\nok<|tool_call>call:lookup{{q:<escape>hi<escape>}}<tool_call|><turn|>\n") {
		t.Fatalf("missing gemma4 tool call rendering: %q", out)
	}
	if !strings.Contains(out, "<|turn>user\n<|tool_response>response:lookup{result:<escape>done<escape>}<tool_response|><turn|>\n") {
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
