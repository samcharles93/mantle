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
