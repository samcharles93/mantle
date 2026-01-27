package main

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/gguf"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

func loadLFM2Config(f *gguf.File) (*model.ModelConfig, error) {
	arch, err := gguf.MustGetString(f.KV, "general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "lfm2" {
		return nil, fmt.Errorf("unsupported architecture %q", arch)
	}

	blockCount, err := gguf.MustGetUint64(f.KV, "lfm2.block_count")
	if err != nil {
		return nil, err
	}
	emb, err := gguf.MustGetUint64(f.KV, "lfm2.embedding_length")
	if err != nil {
		return nil, err
	}
	ffn, err := gguf.MustGetUint64(f.KV, "lfm2.feed_forward_length")
	if err != nil {
		return nil, err
	}
	headCount, err := gguf.MustGetUint64(f.KV, "lfm2.attention.head_count")
	if err != nil {
		return nil, err
	}

	var headCountKV []int
	if vals, ok := gguf.GetArray[int32](f.KV, "lfm2.attention.head_count_kv"); ok {
		headCountKV = make([]int, len(vals))
		for i, v := range vals {
			headCountKV[i] = int(v)
		}
	}

	rms, _ := gguf.GetFloat64(f.KV, "lfm2.attention.layer_norm_rms_epsilon")
	ropeBase, _ := gguf.GetFloat64(f.KV, "lfm2.rope.freq_base")
	ctxLen, _ := gguf.GetUint64(f.KV, "lfm2.context_length")
	vocab, _ := gguf.GetUint64(f.KV, "lfm2.vocab_size")
	shortConv, _ := gguf.GetUint64(f.KV, "lfm2.shortconv.l_cache")

	cfg := model.Config{
		BlockCount:      int(blockCount),
		EmbeddingLength: int(emb),
		FFNLength:       int(ffn),
		HeadCount:       int(headCount),
		HeadCountKV:     headCountKV,
		RMSEpsilon:      rms,
		RopeFreqBase:    ropeBase,
		ContextLength:   int(ctxLen),
		VocabSize:       int(vocab),
		ShortConvLCache: int(shortConv),
	}

	tok := tokenizer.TokenizerConfig{}
	tok.Model, _ = gguf.GetString(f.KV, "tokenizer.ggml.model")
	tok.Pre, _ = gguf.GetString(f.KV, "tokenizer.ggml.pre")
	tok.AddBOS, _ = gguf.GetBool(f.KV, "tokenizer.ggml.add_bos_token")
	tok.AddEOS, _ = gguf.GetBool(f.KV, "tokenizer.ggml.add_eos_token")
	if v, ok := gguf.GetInt64(f.KV, "tokenizer.ggml.bos_token_id"); ok {
		tok.BOSTokenID = int(v)
	}
	if v, ok := gguf.GetInt64(f.KV, "tokenizer.ggml.eos_token_id"); ok {
		tok.EOSTokenID = int(v)
	}
	if v, ok := gguf.GetInt64(f.KV, "tokenizer.ggml.padding_token_id"); ok {
		tok.PADTokenID = int(v)
	}
	if v, ok := gguf.GetInt64(f.KV, "tokenizer.ggml.unk_token_id"); ok {
		tok.UNKTokenID = int(v)
	}
	if s, ok := gguf.GetString(f.KV, "tokenizer.chat_template"); ok {
		tok.ChatTemplate = s
	}

	if tokens, ok := gguf.GetArray[string](f.KV, "tokenizer.ggml.tokens"); ok {
		tok.Tokens = tokens
	}
	if merges, ok := gguf.GetArray[string](f.KV, "tokenizer.ggml.merges"); ok {
		tok.Merges = merges
	}
	if types, ok := gguf.GetArray[int32](f.KV, "tokenizer.ggml.token_type"); ok {
		tok.TokenTypes = types
	}

	return &model.ModelConfig{
		Arch:      arch,
		Config:    cfg,
		Tokenizer: tok,
	}, nil
}
