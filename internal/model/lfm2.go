package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"infer/internal/gguf"
	"infer/internal/safetensors"
	"infer/internal/tensor"
	"infer/internal/tokenizer"
)

type hfConfig struct {
	BlockDim          int      `json:"block_dim"`
	ConvLCache        int      `json:"conv_L_cache"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	LayerTypes        []string `json:"layer_types"`
	MaxPosition       int      `json:"max_position_embeddings"`
	NormEps           float64  `json:"norm_eps"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	VocabSize         int      `json:"vocab_size"`
	RopeTheta         float64  `json:"rope_theta"`
}

func LoadModel(path string, maxContext int) (*Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cfg, err := LoadConfig(f)
	if err != nil {
		return nil, err
	}

	if maxContext <= 0 || maxContext > cfg.Config.ContextLength {
		maxContext = cfg.Config.ContextLength
	}

	emb, err := tensor.LoadGGUFMat(f, "token_embd.weight")
	if err != nil {
		return nil, err
	}
	outNorm, err := tensor.LoadGGUFVec(f, "token_embd_norm.weight")
	if err != nil {
		return nil, err
	}

	layers := make([]Layer, cfg.Config.BlockCount)
	headKVArr := cfg.Config.HeadCountKV
	if len(headKVArr) == 0 {
		headKVArr = make([]int, cfg.Config.BlockCount)
		for i := range headKVArr {
			headKVArr[i] = cfg.Config.HeadCount
		}
	}

	maxHeadKV := 0
	for _, v := range headKVArr {
		if v > maxHeadKV {
			maxHeadKV = v
		}
	}

	for i := 0; i < cfg.Config.BlockCount; i++ {
		layer := &layers[i]
		layer.HeadKV = headKVArr[i]
		layer.IsRecurrent = layer.HeadKV == 0

		layer.AttnNorm, err = tensor.LoadGGUFVec(f, fmt.Sprintf("blk.%d.attn_norm.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnNorm, err = tensor.LoadGGUFVec(f, fmt.Sprintf("blk.%d.ffn_norm.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnUp, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.ffn_up.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnGate, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.ffn_gate.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnDown, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.ffn_down.weight", i))
		if err != nil {
			return nil, err
		}

		if layer.IsRecurrent {
			layer.ShortConvKernel, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.shortconv.conv.weight", i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvInProj, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.shortconv.in_proj.weight", i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvOutProj, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.shortconv.out_proj.weight", i))
			if err != nil {
				return nil, err
			}
			kernelLen := layer.ShortConvKernel.C
			if kernelLen < 1 {
				return nil, fmt.Errorf("invalid shortconv kernel length for layer %d", i)
			}
			layer.ShortConvState = shortConvState{
				buf:       make([]float32, (kernelLen-1)*cfg.Config.EmbeddingLength),
				kernelLen: kernelLen,
			}
		} else {
			layer.AttnQNorm, err = tensor.LoadGGUFVec(f, fmt.Sprintf("blk.%d.attn_q_norm.weight", i))
			if err != nil {
				return nil, err
			}
			layer.AttnKNorm, err = tensor.LoadGGUFVec(f, fmt.Sprintf("blk.%d.attn_k_norm.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wq, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.attn_q.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wk, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.attn_k.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wv, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.attn_v.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wo, err = tensor.LoadGGUFMat(f, fmt.Sprintf("blk.%d.attn_output.weight", i))
			if err != nil {
				return nil, err
			}
			kvStride := layer.HeadKV * (cfg.Config.EmbeddingLength / cfg.Config.HeadCount)
			layer.AttnCache = attnCache{
				k:        make([]float32, maxContext*kvStride),
				v:        make([]float32, maxContext*kvStride),
				kvStride: kvStride,
			}
		}
	}

	headDim := cfg.Config.EmbeddingLength / cfg.Config.HeadCount
	m := &Model{
		Config:     cfg,
		Embeddings: emb,
		OutputNorm: outNorm,
		Output:     emb,
		Layers:     layers,
		MaxContext: maxContext,
		Pos:        0,
		RMSEpsilon: float32(cfg.Config.RMSEpsilon),
		HeadDim:    headDim,
		HeadCount:  cfg.Config.HeadCount,
		MaxHeadKV:  maxHeadKV,
	}
	m.initScratch()
	m.initRoPE()
	return m, nil
}

func LoadConfig(f *gguf.File) (*ModelConfig, error) {
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

	cfg := Config{
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

	return &ModelConfig{
		Arch:      arch,
		Config:    cfg,
		Tokenizer: tok,
	}, nil
}

func LoadModelSafetensors(modelPath, configPath string, maxContext int) (*Model, error) {
	cfg, err := loadHFConfig(configPath)
	if err != nil {
		return nil, err
	}
	st, err := safetensors.Open(modelPath)
	if err != nil {
		return nil, err
	}

	emb, err := tensor.LoadSafetensorsMat(st, "model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	outNorm, err := tensor.LoadSafetensorsVec(st, "model.embedding_norm.weight")
	if err != nil {
		return nil, err
	}

	blockCount := len(cfg.LayerTypes)
	if blockCount == 0 {
		blockCount = cfg.NumAttentionHeads // fallback, should not happen
	}

	if maxContext <= 0 || maxContext > cfg.MaxPosition {
		maxContext = cfg.MaxPosition
	}

	modelCfg := &ModelConfig{
		Arch: "lfm2",
		Config: Config{
			BlockCount:      blockCount,
			EmbeddingLength: cfg.HiddenSize,
			FFNLength:       inferFFNLength(st, 0),
			HeadCount:       cfg.NumAttentionHeads,
			HeadCountKV:     buildHeadKV(cfg),
			RMSEpsilon:      cfg.NormEps,
			RopeFreqBase:    cfg.RopeTheta,
			ContextLength:   cfg.MaxPosition,
			VocabSize:       cfg.VocabSize,
			ShortConvLCache: cfg.ConvLCache,
		},
	}

	headKVArr := modelCfg.Config.HeadCountKV
	maxHeadKV := 0
	for _, v := range headKVArr {
		if v > maxHeadKV {
			maxHeadKV = v
		}
	}

	layers := make([]Layer, blockCount)
	for i := 0; i < blockCount; i++ {
		layer := &layers[i]
		layer.HeadKV = headKVArr[i]
		layer.IsRecurrent = layer.HeadKV == 0

		layer.AttnNorm, err = tensor.LoadSafetensorsVec(st, fmt.Sprintf("model.layers.%d.operator_norm.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnNorm, err = tensor.LoadSafetensorsVec(st, fmt.Sprintf("model.layers.%d.ffn_norm.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnGate, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.feed_forward.w1.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnDown, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.feed_forward.w2.weight", i))
		if err != nil {
			return nil, err
		}
		layer.FfnUp, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.feed_forward.w3.weight", i))
		if err != nil {
			return nil, err
		}

		if layer.IsRecurrent {
			layer.ShortConvKernel, err = tensor.LoadSafetensorsConvKernel(st, fmt.Sprintf("model.layers.%d.conv.conv.weight", i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvInProj, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.conv.in_proj.weight", i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvOutProj, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.conv.out_proj.weight", i))
			if err != nil {
				return nil, err
			}
			kernelLen := layer.ShortConvKernel.C
			layer.ShortConvState = shortConvState{
				buf:       make([]float32, (kernelLen-1)*cfg.HiddenSize),
				kernelLen: kernelLen,
			}
		} else {
			layer.AttnQNorm, err = tensor.LoadSafetensorsVec(st, fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", i))
			if err != nil {
				return nil, err
			}
			layer.AttnKNorm, err = tensor.LoadSafetensorsVec(st, fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wq, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wk, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wv, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
			if err != nil {
				return nil, err
			}
			layer.Wo, err = tensor.LoadSafetensorsMat(st, fmt.Sprintf("model.layers.%d.self_attn.out_proj.weight", i))
			if err != nil {
				return nil, err
			}
			kvStride := layer.HeadKV * (cfg.HiddenSize / cfg.NumAttentionHeads)
			layer.AttnCache = attnCache{
				k:        make([]float32, maxContext*kvStride),
				v:        make([]float32, maxContext*kvStride),
				kvStride: kvStride,
			}
		}
	}

	headDim := cfg.HiddenSize / cfg.NumAttentionHeads
	m := &Model{
		Config:     modelCfg,
		Embeddings: emb,
		OutputNorm: outNorm,
		Output:     emb,
		Layers:     layers,
		MaxContext: maxContext,
		Pos:        0,
		RMSEpsilon: float32(cfg.NormEps),
		HeadDim:    headDim,
		HeadCount:  cfg.NumAttentionHeads,
		MaxHeadKV:  maxHeadKV,
	}
	m.initScratch()
	m.initRoPE()
	return m, nil
}

func loadHFConfig(path string) (*hfConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg hfConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if cfg.HiddenSize == 0 && cfg.BlockDim > 0 {
		cfg.HiddenSize = cfg.BlockDim
	}
	return &cfg, nil
}

func buildHeadKV(cfg *hfConfig) []int {
	out := make([]int, len(cfg.LayerTypes))
	for i, t := range cfg.LayerTypes {
		if t == "full_attention" {
			out[i] = cfg.NumKeyValueHeads
		} else {
			out[i] = 0
		}
	}
	return out
}

func inferFFNLength(st *safetensors.File, layer int) int {
	name := fmt.Sprintf("model.layers.%d.feed_forward.w1.weight", layer)
	if info, ok := st.Tensor(name); ok && len(info.Shape) >= 1 {
		return info.Shape[0]
	}
	return 0
}

// ForwardToken runs one autoregressive step for the provided token id.
// It returns a logits slice owned by the model (overwritten on next call).
func (m *Model) ForwardToken(tok int) ([]float32, error) {
	if tok < 0 || tok >= m.Config.Config.VocabSize {
		return nil, fmt.Errorf("token id out of range: %d", tok)
	}
	if m.Pos >= m.MaxContext {
		return nil, fmt.Errorf("context length exceeded: %d >= %d", m.Pos, m.MaxContext)
	}

	x := m.scratch.x
	copy(x, m.Embeddings.Row(tok))

	for i := range m.Layers {
		layer := &m.Layers[i]

		// operator norm
		tensor.RMSNorm(m.scratch.tmp, x, layer.AttnNorm, m.RMSEpsilon)

		var opOut []float32
		if layer.IsRecurrent {
			opOut = m.shortconv(layer, m.scratch.tmp)
		} else {
			opOut = m.attention(layer, m.scratch.tmp, m.Pos)
		}
		tensor.Add(x, opOut)

		// ffn
		tensor.RMSNorm(m.scratch.tmp, x, layer.FfnNorm, m.RMSEpsilon)
		ffnOut := m.ffn(layer, m.scratch.tmp)
		tensor.Add(x, ffnOut)
	}

	// output norm
	tensor.RMSNorm(m.scratch.tmp, x, m.OutputNorm, m.RMSEpsilon)
	tensor.MatVec(m.scratch.logits, m.Output, m.scratch.tmp)

	m.Pos++
	return m.scratch.logits, nil
}

func (m *Model) Reset() {
	m.Pos = 0
	for i := range m.Layers {
		layer := &m.Layers[i]
		if layer.AttnCache.k != nil {
			for j := range layer.AttnCache.k {
				layer.AttnCache.k[j] = 0
			}
			for j := range layer.AttnCache.v {
				layer.AttnCache.v[j] = 0
			}
		}
		if layer.ShortConvState.buf != nil {
			for j := range layer.ShortConvState.buf {
				layer.ShortConvState.buf[j] = 0
			}
		}
	}
}

func (m *Model) attention(layer *Layer, x []float32, pos int) []float32 {
	nHead := m.HeadCount
	headDim := m.HeadDim
	kvHeads := layer.HeadKV
	kvStride := layer.AttnCache.kvStride
	if kvHeads <= 0 {
		panic("attention layer without kv heads")
	}

	q := m.scratch.q[:nHead*headDim]
	k := m.scratch.k[:kvStride]
	v := m.scratch.v[:kvStride]
	attnOut := m.scratch.attnOut

	tensor.MatVec(q, layer.Wq, x)
	tensor.MatVec(k, layer.Wk, x)
	tensor.MatVec(v, layer.Wv, x)

	for h := range nHead {
		tensor.RMSNorm(q[h*headDim:(h+1)*headDim], q[h*headDim:(h+1)*headDim], layer.AttnQNorm, m.RMSEpsilon)
	}
	for h := range kvHeads {
		tensor.RMSNorm(k[h*headDim:(h+1)*headDim], k[h*headDim:(h+1)*headDim], layer.AttnKNorm, m.RMSEpsilon)
	}

	tensor.ApplyRoPE(q, nHead, headDim, pos, m.ropeInvFreq)
	tensor.ApplyRoPE(k, kvHeads, headDim, pos, m.ropeInvFreq)

	cacheK := layer.AttnCache.k
	cacheV := layer.AttnCache.v
	copy(cacheK[pos*kvStride:pos*kvStride+kvStride], k)
	copy(cacheV[pos*kvStride:pos*kvStride+kvStride], v)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := range nHead {
		kvHead := h * kvHeads / nHead
		qh := q[h*headDim : (h+1)*headDim]
		scores := m.scratch.scores[:pos+1]
		for t := 0; t <= pos; t++ {
			koff := t*kvStride + kvHead*headDim
			kv := cacheK[koff : koff+headDim]
			scores[t] = tensor.Dot(qh, kv) * scale
		}
		tensor.Softmax(scores)
		out := attnOut[h*headDim : (h+1)*headDim]
		for d := range headDim {
			var sum float32
			for t := 0; t <= pos; t++ {
				voff := t*kvStride + kvHead*headDim + d
				sum += scores[t] * cacheV[voff]
			}
			out[d] = sum
		}
	}

	tensor.MatVec(m.scratch.attnProj, layer.Wo, attnOut[:nHead*headDim])
	return m.scratch.attnProj
}

func (m *Model) shortconv(layer *Layer, x []float32) []float32 {
	embd := m.Config.Config.EmbeddingLength
	tensor.MatVec(m.scratch.scProj, layer.ShortConvInProj, x)
	b := m.scratch.scProj[:embd]
	c := m.scratch.scProj[embd : 2*embd]
	xg := m.scratch.scProj[2*embd:]

	bx := m.scratch.scBx
	for i := range embd {
		bx[i] = b[i] * xg[i]
	}

	kernel := layer.ShortConvKernel
	convOut := m.scratch.scConv
	kernelLen := kernel.C
	state := layer.ShortConvState.buf
	for i := range embd {
		row := kernel.Row(i)
		var sum float32
		for k := 0; k < kernelLen-1; k++ {
			sum += row[k] * state[k*embd+i]
		}
		sum += row[kernelLen-1] * bx[i]
		convOut[i] = sum
	}

	// update state: shift left and append current bx
	if kernelLen > 1 {
		if kernelLen == 2 {
			copy(state, bx)
		} else {
			copy(state, state[embd:])
			copy(state[(kernelLen-2)*embd:], bx)
		}
	}

	for i := range embd {
		m.scratch.tmp2[i] = c[i] * convOut[i]
	}
	tensor.MatVec(m.scratch.tmp, layer.ShortConvOutProj, m.scratch.tmp2)
	return m.scratch.tmp
}

func (m *Model) ffn(layer *Layer, x []float32) []float32 {
	tensor.MatVec(m.scratch.ffnUp, layer.FfnUp, x)
	tensor.MatVec(m.scratch.ffnGate, layer.FfnGate, x)
	for i := range m.scratch.ffnAct {
		m.scratch.ffnAct[i] = tensor.Silu(m.scratch.ffnGate[i]) * m.scratch.ffnUp[i]
	}
	tensor.MatVec(m.scratch.tmp2, layer.FfnDown, m.scratch.ffnAct)
	return m.scratch.tmp2
}

func (m *Model) initScratch() {
	embd := m.Config.Config.EmbeddingLength
	ffn := m.Config.Config.FFNLength
	kv := m.MaxHeadKV * m.HeadDim
	if kv < 1 {
		kv = m.HeadDim
	}
	m.scratch = scratchBuffers{
		x:        make([]float32, embd),
		tmp:      make([]float32, embd),
		tmp2:     make([]float32, embd),
		q:        make([]float32, embd),
		k:        make([]float32, kv),
		v:        make([]float32, kv),
		attnOut:  make([]float32, embd),
		attnProj: make([]float32, embd),
		scores:   make([]float32, m.MaxContext),
		ffnUp:    make([]float32, ffn),
		ffnGate:  make([]float32, ffn),
		ffnAct:   make([]float32, ffn),
		scProj:   make([]float32, embd*3),
		scBx:     make([]float32, embd),
		scConv:   make([]float32, embd),
		logits:   make([]float32, m.Config.Config.VocabSize),
	}
}

func (m *Model) initRoPE() {
	headDim := m.HeadDim
	if headDim == 0 {
		headDim = m.Config.Config.EmbeddingLength / m.Config.Config.HeadCount
	}
	ropeInvFreq := make([]float64, headDim/2)
	for i := 0; i < len(ropeInvFreq); i++ {
		power := float64(2*i) / float64(headDim)
		ropeInvFreq[i] = 1.0 / math.Pow(m.Config.Config.RopeFreqBase, power)
	}
	m.ropeInvFreq = ropeInvFreq
}
