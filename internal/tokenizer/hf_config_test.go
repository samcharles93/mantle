package tokenizer

import "testing"

func TestParseHFTokenizerConfigBytes(t *testing.T) {
	t.Parallel()

	tokJSON := []byte(`{
		"model":{
			"type":"BPE",
			"vocab":{"<s>":1,"</s>":2,"<unk>":3},
			"merges":[],
			"unk_token":"<unk>"
		},
		"post_processor":{
			"processors":[
				{"type":"TemplateProcessing","special_tokens":{"bos":{"ids":[7]}}}
			]
		}
	}`)
	tokConfig := []byte(`{
		"add_bos_token":false,
		"add_eos_token":true,
		"bos_token":"<s>",
		"eos_token":"</s>",
		"unk_token":"<unk>",
		"chat_template":"{{ messages }}"
	}`)

	cfg, err := ParseHFTokenizerConfigBytes(tokJSON, tokConfig)
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if !cfg.AddBOS {
		t.Fatalf("expected AddBOS=true due to template processing override")
	}
	if !cfg.AddEOS {
		t.Fatalf("expected AddEOS=true")
	}
	if cfg.BOSTokenID != 7 {
		t.Fatalf("unexpected BOS id: got %d want 7", cfg.BOSTokenID)
	}
	if cfg.EOSTokenID != 2 {
		t.Fatalf("unexpected EOS id: got %d want 2", cfg.EOSTokenID)
	}
	if cfg.UNKTokenID != 3 {
		t.Fatalf("unexpected UNK id: got %d want 3", cfg.UNKTokenID)
	}
	if cfg.ChatTemplate != "{{ messages }}" {
		t.Fatalf("unexpected chat template: %q", cfg.ChatTemplate)
	}
}

func TestParseHFTokenizerConfigBytesRejectsUnsupportedModel(t *testing.T) {
	t.Parallel()

	tokJSON := []byte(`{"model":{"type":"WordPiece","vocab":{},"merges":[]}}`)
	_, err := ParseHFTokenizerConfigBytes(tokJSON, nil)
	if err == nil {
		t.Fatalf("expected unsupported tokenizer model error")
	}
}

// TokenString(id) is what prompt rendering uses to inject a template's BOS/EOS
// (internal/inference/prompt.go). It cannot work unless the id -> token table is
// populated, and nothing else fills it in. MiniCPM5 is the motivating case:
// bos_token "<s>" sits at id 0 and its template opens with "{{- bos_token }}", so
// an empty table silently drops the BOS from the rendered prompt.
func TestParseHFTokenizerConfigBytesPopulatesTokenTable(t *testing.T) {
	t.Parallel()

	tokJSON := []byte(`{
		"model":{"type":"BPE","vocab":{"a":2,"b":3},"merges":[]},
		"added_tokens":[
			{"id":0,"content":"<s>","special":true},
			{"id":1,"content":"</s>","special":true}
		]
	}`)
	tokConfig := []byte(`{"bos_token":"<s>","eos_token":"</s>"}`)

	cfg, err := ParseHFTokenizerConfigBytes(tokJSON, tokConfig)
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.BOSTokenID != 0 {
		t.Fatalf("unexpected BOS id: got %d want 0", cfg.BOSTokenID)
	}
	if got := cfg.TokenString(cfg.BOSTokenID); got != "<s>" {
		t.Fatalf("TokenString(BOS): got %q want %q", got, "<s>")
	}
	if got := cfg.TokenString(cfg.EOSTokenID); got != "</s>" {
		t.Fatalf("TokenString(EOS): got %q want %q", got, "</s>")
	}
}
