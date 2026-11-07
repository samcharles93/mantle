package simd

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	mcfstore "github.com/samcharles93/mantle/internal/mcfstore"
)

// DSparkConfig is the shape of a DSpark draft checkpoint, read from its
// config.json. These are training-time properties of the weights; whether to
// speculate and how far is decided by the caller, not by this config.
type DSparkConfig struct {
	Architectures    []string `json:"architectures"`
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	DraftVocabSize   int      `json:"draft_vocab_size"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	BlockSize        int      `json:"block_size"`
	MaskTokenID      int      `json:"mask_token_id"`
	TargetLayerIDs   []int    `json:"target_layer_ids"`
	NumTargetLayers  int      `json:"num_target_layers"`
	MarkovRank       int      `json:"markov_rank"`
	MarkovHeadType   string   `json:"markov_head_type"`
	ProjectorType    string   `json:"projector_type"`
	HasConfidence    bool     `json:"enable_confidence_head"`
	HiddenAct        string   `json:"hidden_act"`
	RopeParameters   struct {
		RopeTheta float64 `json:"rope_theta"`
		RopeType  string  `json:"rope_type"`
	} `json:"rope_parameters"`
	SampleFromAnchor *bool `json:"sample_from_anchor"`
}

// ParseDSparkConfig decodes and validates a DSpark draft config.json. Only the
// variant Mantle implements is accepted: a vanilla Markov head, the dspark
// projector, full draft vocabulary, SiLU and default RoPE, drafting from the
// anchor slot.
func ParseDSparkConfig(raw []byte) (*DSparkConfig, error) {
	var c DSparkConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("dspark config: %w", err)
	}
	if !slices.Contains(c.Architectures, "Qwen3DSparkModel") {
		return nil, fmt.Errorf("dspark config: architectures %v is not Qwen3DSparkModel", c.Architectures)
	}
	switch {
	case c.HiddenSize <= 0, c.IntermediateSize <= 0, c.NumLayers <= 0, c.NumHeads <= 0,
		c.NumKVHeads <= 0, c.HeadDim <= 0, c.VocabSize <= 0, c.MarkovRank <= 0:
		return nil, fmt.Errorf("dspark config: non-positive model dimension")
	case c.NumHeads%c.NumKVHeads != 0:
		return nil, fmt.Errorf("dspark config: %d heads not divisible by %d kv heads", c.NumHeads, c.NumKVHeads)
	case c.HeadDim%2 != 0:
		return nil, fmt.Errorf("dspark config: odd head_dim %d", c.HeadDim)
	case c.BlockSize <= 0 || c.BlockSize > smallBatchMaxRows:
		return nil, fmt.Errorf("dspark config: block_size %d outside [1,%d]", c.BlockSize, smallBatchMaxRows)
	case c.MaskTokenID < 0 || c.MaskTokenID >= c.VocabSize:
		return nil, fmt.Errorf("dspark config: mask_token_id %d outside vocab", c.MaskTokenID)
	case len(c.TargetLayerIDs) == 0:
		return nil, fmt.Errorf("dspark config: no target_layer_ids")
	case c.DraftVocabSize != 0 && c.DraftVocabSize != c.VocabSize:
		return nil, fmt.Errorf("dspark config: reduced draft vocabulary is not supported")
	case c.MarkovHeadType != "" && c.MarkovHeadType != "vanilla":
		return nil, fmt.Errorf("dspark config: markov_head_type %q is not supported", c.MarkovHeadType)
	case c.ProjectorType != "" && c.ProjectorType != "dspark":
		return nil, fmt.Errorf("dspark config: projector_type %q is not supported", c.ProjectorType)
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return nil, fmt.Errorf("dspark config: hidden_act %q is not supported", c.HiddenAct)
	case c.RopeParameters.RopeType != "" && c.RopeParameters.RopeType != "default":
		return nil, fmt.Errorf("dspark config: rope_type %q is not supported", c.RopeParameters.RopeType)
	case c.SampleFromAnchor != nil && !*c.SampleFromAnchor:
		return nil, fmt.Errorf("dspark config: bonus-anchor drafting is not supported")
	}
	for i := 1; i < len(c.TargetLayerIDs); i++ {
		if c.TargetLayerIDs[i] <= c.TargetLayerIDs[i-1] {
			return nil, fmt.Errorf("dspark config: target_layer_ids %v must be strictly increasing", c.TargetLayerIDs)
		}
	}
	if c.RMSNormEps <= 0 {
		c.RMSNormEps = 1e-6
	}
	if c.RopeParameters.RopeTheta <= 0 {
		c.RopeParameters.RopeTheta = 10000
	}
	return &c, nil
}

type dsparkLayer struct {
	inNorm, postNorm []float32
	qNorm, kNorm     []float32
	wq, wk, wv, wo   *Mat
	gate, up, down   *Mat
}

// DSparkDraft is a loaded DSpark draft model paired with a target instance.
//
// The draft has no token embedding or LM head of its own: it reads the
// target's. Its attention cache holds one K/V row per target position, built
// from the target's hidden-state taps by Inject. Draft blocks attend over that
// cache plus their own rows without writing to it.
type DSparkDraft struct {
	Config *DSparkConfig

	target *Instance

	fc         *Mat // [hidden, taps*hidden]
	hiddenNorm []float32
	layers     []dsparkLayer
	norm       []float32
	markovW1   *Mat // [vocab, rank]
	markovW2   *Mat // [vocab, rank]
	confW      []float32
	confB      float32

	invFreq []float64
	eps     float32

	// cacheK and cacheV hold [layer][pos*kvDim] for positions [0,ctxLen).
	cacheK, cacheV [][]float32
	ctxLen         int
	maxContext     int

	scratch dsparkScratch
}

type dsparkScratch struct {
	feat, fused, kv                   []float32
	x, h, q, k, v, attn, o            []float32
	gate, up                          []float32
	embd, logits, bias, markov, score []float32
}

// LoadDSparkDraftMCF loads a DSpark draft from an MCF file and pairs it with
// target. configJSON is the draft's config.json. The target must be a dense
// model of matching hidden size and vocabulary, with hidden-tap capture
// enabled on exactly the draft's target layers.
func LoadDSparkDraftMCF(mf *mcfstore.File, configJSON []byte, target *Instance, maxContext int) (*DSparkDraft, error) {
	if mf == nil || target == nil {
		return nil, fmt.Errorf("dspark: nil draft file or target")
	}
	cfg, err := ParseDSparkConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if err := checkDSparkTarget(cfg, target); err != nil {
		return nil, err
	}
	src := mcfSource{mf: mf}
	d := &DSparkDraft{Config: cfg, target: target, maxContext: maxContext, eps: float32(cfg.RMSNormEps)}

	taps := len(cfg.TargetLayerIDs)
	if d.fc, err = loadShapedMat(src, "fc.weight", cfg.HiddenSize, taps*cfg.HiddenSize); err != nil {
		return nil, err
	}
	if d.hiddenNorm, err = loadSizedVec(src, "hidden_norm.weight", cfg.HiddenSize); err != nil {
		return nil, err
	}
	if d.norm, err = loadSizedVec(src, "norm.weight", cfg.HiddenSize); err != nil {
		return nil, err
	}
	qDim, kvDim := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	d.layers = make([]dsparkLayer, cfg.NumLayers)
	for i := range d.layers {
		l := &d.layers[i]
		p := fmt.Sprintf("layers.%d.", i)
		vecs := []struct {
			dst  *[]float32
			name string
			n    int
		}{
			{&l.inNorm, "input_layernorm.weight", cfg.HiddenSize},
			{&l.postNorm, "post_attention_layernorm.weight", cfg.HiddenSize},
			{&l.qNorm, "self_attn.q_norm.weight", cfg.HeadDim},
			{&l.kNorm, "self_attn.k_norm.weight", cfg.HeadDim},
		}
		for _, v := range vecs {
			if *v.dst, err = loadSizedVec(src, p+v.name, v.n); err != nil {
				return nil, err
			}
		}
		mats := []struct {
			dst        **Mat
			name       string
			rows, cols int
		}{
			{&l.wq, "self_attn.q_proj.weight", qDim, cfg.HiddenSize},
			{&l.wk, "self_attn.k_proj.weight", kvDim, cfg.HiddenSize},
			{&l.wv, "self_attn.v_proj.weight", kvDim, cfg.HiddenSize},
			{&l.wo, "self_attn.o_proj.weight", cfg.HiddenSize, qDim},
			{&l.gate, "mlp.gate_proj.weight", cfg.IntermediateSize, cfg.HiddenSize},
			{&l.up, "mlp.up_proj.weight", cfg.IntermediateSize, cfg.HiddenSize},
			{&l.down, "mlp.down_proj.weight", cfg.HiddenSize, cfg.IntermediateSize},
		}
		for _, m := range mats {
			if *m.dst, err = loadShapedMat(src, p+m.name, m.rows, m.cols); err != nil {
				return nil, err
			}
		}
	}
	if d.markovW1, err = loadShapedMat(src, "markov_head.markov_w1.weight", cfg.VocabSize, cfg.MarkovRank); err != nil {
		return nil, err
	}
	if d.markovW2, err = loadShapedMat(src, "markov_head.markov_w2.weight", cfg.VocabSize, cfg.MarkovRank); err != nil {
		return nil, err
	}
	if cfg.HasConfidence {
		w, err := loadShapedMat(src, "confidence_head.proj.weight", 1, cfg.HiddenSize+cfg.MarkovRank)
		if err != nil {
			return nil, err
		}
		d.confW = make([]float32, w.C)
		w.RowTo(d.confW, 0)
		b, err := loadSizedVec(src, "confidence_head.proj.bias", 1)
		if err != nil {
			return nil, err
		}
		d.confB = b[0]
	}
	d.init()
	return d, nil
}

func checkDSparkTarget(cfg *DSparkConfig, target *Instance) error {
	tc := &target.Config.Config
	switch {
	case tc.EmbeddingLength != cfg.HiddenSize:
		return fmt.Errorf("dspark: target hidden %d != draft hidden %d", tc.EmbeddingLength, cfg.HiddenSize)
	case target.Embeddings == nil || target.Embeddings.R != cfg.VocabSize || target.Embeddings.C != cfg.HiddenSize:
		return fmt.Errorf("dspark: target embeddings do not match draft vocab %d x %d", cfg.VocabSize, cfg.HiddenSize)
	case target.Output == nil || target.Output.R != cfg.VocabSize || target.Output.C != cfg.HiddenSize:
		return fmt.Errorf("dspark: target LM head does not match draft vocab %d x %d", cfg.VocabSize, cfg.HiddenSize)
	case cfg.NumTargetLayers != 0 && cfg.NumTargetLayers != len(target.Layers):
		return fmt.Errorf("dspark: draft trained for %d target layers, target has %d", cfg.NumTargetLayers, len(target.Layers))
	case !slices.Equal(target.HiddenTaps.Layers, cfg.TargetLayerIDs):
		return fmt.Errorf("dspark: target taps %v must equal draft target_layer_ids %v", target.HiddenTaps.Layers, cfg.TargetLayerIDs)
	}
	return nil
}

func loadShapedMat(src tensorSource, name string, rows, cols int) (*Mat, error) {
	m, err := loadMat(src, name)
	if err != nil {
		return nil, err
	}
	if m.R != rows || m.C != cols {
		return nil, fmt.Errorf("%s: shape [%d %d], want [%d %d]", name, m.R, m.C, rows, cols)
	}
	return m, nil
}

func loadSizedVec(src tensorSource, name string, n int) ([]float32, error) {
	v, err := loadVec(src, name)
	if err != nil {
		return nil, err
	}
	if len(v) != n {
		return nil, fmt.Errorf("%s: length %d, want %d", name, len(v), n)
	}
	return v, nil
}

func (d *DSparkDraft) init() {
	c := d.Config
	half := c.HeadDim / 2
	d.invFreq = make([]float64, half)
	for i := range d.invFreq {
		d.invFreq[i] = 1 / math.Pow(c.RopeParameters.RopeTheta, float64(2*i)/float64(c.HeadDim))
	}
	kvDim := c.NumKVHeads * c.HeadDim
	d.cacheK = make([][]float32, c.NumLayers)
	d.cacheV = make([][]float32, c.NumLayers)
	for i := range c.NumLayers {
		d.cacheK[i] = make([]float32, d.maxContext*kvDim)
		d.cacheV[i] = make([]float32, d.maxContext*kvDim)
	}
	b, h, q := c.BlockSize, c.HiddenSize, c.NumHeads*c.HeadDim
	taps := len(c.TargetLayerIDs)
	d.scratch = dsparkScratch{
		feat:   make([]float32, smallBatchMaxRows*taps*h),
		fused:  make([]float32, smallBatchMaxRows*h),
		kv:     make([]float32, smallBatchMaxRows*kvDim),
		x:      make([]float32, b*h),
		h:      make([]float32, b*h),
		q:      make([]float32, b*q),
		k:      make([]float32, b*kvDim),
		v:      make([]float32, b*kvDim),
		attn:   make([]float32, b*q),
		o:      make([]float32, b*h),
		gate:   make([]float32, b*c.IntermediateSize),
		up:     make([]float32, b*c.IntermediateSize),
		embd:   make([]float32, b*h),
		logits: make([]float32, b*c.VocabSize),
		bias:   make([]float32, c.VocabSize),
		markov: make([]float32, c.MarkovRank),
		score:  make([]float32, d.maxContext+b),
	}
}

// ContextLen is the number of target positions injected so far.
func (d *DSparkDraft) ContextLen() int { return d.ctxLen }

// Truncate discards injected positions at and after pos, so a rejected
// speculative suffix can be re-injected from the target's accepted state.
func (d *DSparkDraft) Truncate(pos int) {
	d.ctxLen = max(0, min(pos, d.ctxLen))
}

// Inject projects the target's most recent hidden-tap capture into the draft
// cache. The capture's rows are the target positions [pos, pos+rows). pos must
// equal ContextLen, so injection always extends the cache contiguously.
func (d *DSparkDraft) Inject(pos int) error {
	taps := &d.target.HiddenTaps
	rows := taps.Rows
	switch {
	case pos != d.ctxLen:
		return fmt.Errorf("dspark: inject at %d, cache holds %d positions", pos, d.ctxLen)
	case rows <= 0:
		return fmt.Errorf("dspark: no captured target rows to inject")
	case pos+rows > d.maxContext:
		return fmt.Errorf("dspark: inject %d rows at %d exceeds context %d", rows, pos, d.maxContext)
	}
	c := d.Config
	h := c.HiddenSize
	nTaps := len(c.TargetLayerIDs)
	kvDim := c.NumKVHeads * c.HeadDim
	for start := 0; start < rows; start += smallBatchMaxRows {
		n := min(smallBatchMaxRows, rows-start)
		feat := d.scratch.feat[:n*nTaps*h]
		for r := range n {
			for s := range nTaps {
				src := taps.Host[(s*taps.Capacity+start+r)*h : (s*taps.Capacity+start+r+1)*h]
				copy(feat[(r*nTaps+s)*h:(r*nTaps+s+1)*h], src)
			}
		}
		fused := d.scratch.fused[:n*h]
		projectRows(fused, d.fc, feat, n)
		for r := range n {
			RMSNorm(fused[r*h:(r+1)*h], fused[r*h:(r+1)*h], d.hiddenNorm, d.eps)
		}
		for li := range d.layers {
			l := &d.layers[li]
			kv := d.scratch.kv[:n*kvDim]
			projectRows(kv, l.wk, fused, n)
			for r := range n {
				row := kv[r*kvDim : (r+1)*kvDim]
				headRMSNorm(row, l.kNorm, c.NumKVHeads, c.HeadDim, d.eps)
				ApplyRoPE(row, c.NumKVHeads, c.HeadDim, pos+start+r, d.invFreq, 1)
			}
			copy(d.cacheK[li][(pos+start)*kvDim:], kv)
			projectRows(kv, l.wv, fused, n)
			copy(d.cacheV[li][(pos+start)*kvDim:], kv)
		}
	}
	d.ctxLen = pos + rows
	return nil
}

// DraftBlock drafts up to n tokens following anchor, the last committed token,
// which sits at position ContextLen (not yet injected). It returns the drafted
// tokens and, when the checkpoint has a confidence head, each token's
// predicted acceptance probability. Drafting is greedy: each block position
// takes the argmax of the target LM head logits plus the Markov bias chained
// from the previous drafted token.
func (d *DSparkDraft) DraftBlock(anchor, n int) (tokens []int, conf []float32, err error) {
	c := d.Config
	if n <= 0 || n > c.BlockSize {
		return nil, nil, fmt.Errorf("dspark: draft length %d outside [1,%d]", n, c.BlockSize)
	}
	if anchor < 0 || anchor >= c.VocabSize {
		return nil, nil, fmt.Errorf("dspark: anchor token %d outside vocab", anchor)
	}
	pos0 := d.ctxLen
	if pos0+n > d.maxContext {
		return nil, nil, fmt.Errorf("dspark: block at %d exceeds context %d", pos0, d.maxContext)
	}
	h := c.HiddenSize
	s := &d.scratch
	x := s.x[:n*h]
	for i := range n {
		tok := c.MaskTokenID
		if i == 0 {
			tok = anchor
		}
		d.target.Embeddings.RowTo(x[i*h:(i+1)*h], tok)
	}
	for li := range d.layers {
		d.blockLayer(li, n, pos0)
	}
	embd := s.embd[:n*h]
	for i := range n {
		RMSNorm(embd[i*h:(i+1)*h], x[i*h:(i+1)*h], d.norm, d.eps)
	}
	vocab := c.VocabSize
	logits := s.logits[:n*vocab]
	projectRows(logits, d.target.Output, embd, n)

	tokens = make([]int, n)
	if c.HasConfidence {
		conf = make([]float32, n)
	}
	prev := anchor
	for i := range n {
		d.markovW1.RowTo(s.markov, prev)
		MatVec(s.bias, d.markovW2, s.markov)
		col := logits[i*vocab : (i+1)*vocab]
		Add(col, s.bias)
		tokens[i] = argmaxF32(col)
		if c.HasConfidence {
			z := Dot(d.confW[:h], embd[i*h:(i+1)*h]) + Dot(d.confW[h:], s.markov) + d.confB
			conf[i] = float32(1 / (1 + math.Exp(-float64(z))))
		}
		prev = tokens[i]
	}
	return tokens, conf, nil
}

// blockLayer runs draft layer li over the n block rows at positions
// pos0..pos0+n-1. Attention is non-causal: every block row sees the whole
// injected context and every block row.
func (d *DSparkDraft) blockLayer(li, n, pos0 int) {
	c := d.Config
	l := &d.layers[li]
	s := &d.scratch
	h := c.HiddenSize
	hd := c.HeadDim
	qDim, kvDim := c.NumHeads*hd, c.NumKVHeads*hd
	x, hn := s.x[:n*h], s.h[:n*h]
	q, k, v := s.q[:n*qDim], s.k[:n*kvDim], s.v[:n*kvDim]

	for i := range n {
		RMSNorm(hn[i*h:(i+1)*h], x[i*h:(i+1)*h], l.inNorm, d.eps)
	}
	projectRows(q, l.wq, hn, n)
	projectRows(k, l.wk, hn, n)
	projectRows(v, l.wv, hn, n)
	for i := range n {
		qi, ki := q[i*qDim:(i+1)*qDim], k[i*kvDim:(i+1)*kvDim]
		headRMSNorm(qi, l.qNorm, c.NumHeads, hd, d.eps)
		headRMSNorm(ki, l.kNorm, c.NumKVHeads, hd, d.eps)
		ApplyRoPE(qi, c.NumHeads, hd, pos0+i, d.invFreq, 1)
		ApplyRoPE(ki, c.NumKVHeads, hd, pos0+i, d.invFreq, 1)
	}

	attn := s.attn[:n*qDim]
	dsparkAttend(attn, q, k, v, d.cacheK[li], d.cacheV[li], s.score[:d.ctxLen+n], n, d.ctxLen, c.NumHeads, c.NumKVHeads, hd)
	o := s.o[:n*h]
	projectRows(o, l.wo, attn, n)
	Add(x, o)

	for i := range n {
		RMSNorm(hn[i*h:(i+1)*h], x[i*h:(i+1)*h], l.postNorm, d.eps)
	}
	ffn := c.IntermediateSize
	gate, up := s.gate[:n*ffn], s.up[:n*ffn]
	projectRows(gate, l.gate, hn, n)
	projectRows(up, l.up, hn, n)
	for i := range gate {
		gate[i] = Silu(gate[i]) * up[i]
	}
	projectRows(o, l.down, gate, n)
	Add(x, o)
}

// dsparkAttend writes, for each of n block query rows, softmax attention over
// ctx cached positions followed by the n block rows themselves. Heads share
// KV heads in contiguous groups. score needs ctx+n entries.
func dsparkAttend(out, q, k, v, cacheK, cacheV, score []float32, n, ctx, heads, kvHeads, hd int) {
	qDim, kvDim := heads*hd, kvHeads*hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	group := heads / kvHeads
	clear(out[:n*qDim])
	for i := range n {
		for hq := range heads {
			kh := hq / group
			qv := q[i*qDim+hq*hd : i*qDim+(hq+1)*hd]
			for p := range ctx {
				score[p] = Dot(qv, cacheK[p*kvDim+kh*hd:p*kvDim+(kh+1)*hd]) * scale
			}
			for j := range n {
				score[ctx+j] = Dot(qv, k[j*kvDim+kh*hd:j*kvDim+(kh+1)*hd]) * scale
			}
			Softmax(score[:ctx+n])
			o := out[i*qDim+hq*hd : i*qDim+(hq+1)*hd]
			for p := range ctx {
				axpy(o, cacheV[p*kvDim+kh*hd:p*kvDim+(kh+1)*hd], score[p])
			}
			for j := range n {
				axpy(o, v[j*kvDim+kh*hd:j*kvDim+(kh+1)*hd], score[ctx+j])
			}
		}
	}
}

// projectRows computes dst[r] = w · src[r] for n contiguous rows.
func projectRows(dst []float32, w *Mat, src []float32, n int) {
	if useSmallBatchBF16(w, n) {
		MatMatBF16(dst, w, src, n)
		return
	}
	for r := range n {
		MatVec(dst[r*w.R:(r+1)*w.R], w, src[r*w.C:(r+1)*w.C])
	}
}

// headRMSNorm normalises each headDim-wide head of x in place.
func headRMSNorm(x, weight []float32, heads, headDim int, eps float32) {
	for hh := range heads {
		seg := x[hh*headDim : (hh+1)*headDim]
		RMSNorm(seg, seg, weight, eps)
	}
}

func axpy(dst, src []float32, a float32) {
	for i, v := range src {
		dst[i] += a * v
	}
}

func argmaxF32(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}
