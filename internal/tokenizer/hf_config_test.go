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
