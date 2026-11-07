package simd

import (
	"fmt"
	"math"
	"slices"

	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
)

// DSparkDraft is a DSpark draft model paired with a target.
//
// The draft's decoder is an ordinary Instance: standard Qwen3 layers, the
// regular KV cache and the shared batched kernels, loaded with the target's
// embedding and LM head. Only the DSpark parts live here: projecting the
// target's hidden taps into the draft cache, and the Markov and confidence
// heads applied to a drafted block.
//
// The draft cache holds one K/V row per target position. Its Pos is the
// number of injected positions; draft blocks run at Pos and leave it there.
type DSparkDraft struct {
	Config *model.DSparkConfig

	target *core.Instance
	inst   *Instance

	fc         *Mat // [hidden, taps*hidden]
	hiddenNorm []float32
	markovW1   *Mat // [vocab, rank]
	markovW2   *Mat // [vocab, rank]
	confW      []float32
	confB      float32

	feat, fused, keys, vals []float32
	logits, embd            []float32
	bias, markov            []float32
}

// LoadDSparkDraftMCF loads a DSpark draft from an MCF file and pairs it with
// target. configJSON is the draft's config.json. The target must match the
// draft's hidden size and vocabulary and capture hidden taps on exactly the
// draft's target layers. opts selects the draft's own runtime settings, such
// as its KV cache types; its embedding and LM head are always the target's.
func LoadDSparkDraftMCF(mf *mcfstore.File, configJSON []byte, target *core.Instance, maxContext int, opts LoadModelOptions) (*DSparkDraft, error) {
	if mf == nil || target == nil {
		return nil, fmt.Errorf("dspark: nil draft file or target")
	}
	cfg, err := model.ParseDSparkConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if err := checkDSparkTarget(cfg, target); err != nil {
		return nil, err
	}
	hf, err := model.LoadHFConfigBytes(configJSON)
	if err != nil {
		return nil, err
	}
	opts.SharedEmbeddings, opts.SharedOutput = target.Embeddings, target.Output
	opts.HiddenTapLayers = nil
	src := mcfSource{mf: mf}
	inst, err := loadModelFromSource(hf, model.DSparkDraftSpec(), src, maxContext, opts)
	if err != nil {
		return nil, fmt.Errorf("dspark: %w", err)
	}
	if got, want := inst.Config.Config.RopeFreqBase, cfg.RopeParameters.RopeTheta; got != want {
		return nil, fmt.Errorf("dspark: loaded rope base %g, config says %g", got, want)
	}
	if why := inst.batchIneligibleReason(make([]int, cfg.BlockSize)); why != "" {
		return nil, fmt.Errorf("dspark: draft decoder cannot run batched blocks: %s", why)
	}

	d := &DSparkDraft{Config: cfg, target: target, inst: inst}
	h := cfg.HiddenSize
	taps := len(cfg.TargetLayerIDs)
	if d.fc, err = loadShapedMat(src, "fc.weight", h, taps*h); err != nil {
		return nil, err
	}
	if d.hiddenNorm, err = loadSizedVec(src, "hidden_norm.weight", h); err != nil {
		return nil, err
	}
	if d.markovW1, err = loadShapedMat(src, "markov_head.markov_w1.weight", cfg.VocabSize, cfg.MarkovRank); err != nil {
		return nil, err
	}
	if d.markovW2, err = loadShapedMat(src, "markov_head.markov_w2.weight", cfg.VocabSize, cfg.MarkovRank); err != nil {
		return nil, err
	}
	if cfg.HasConfidence {
		w, err := loadShapedMat(src, "confidence_head.proj.weight", 1, h+cfg.MarkovRank)
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

	rows := model.DSparkMaxBlockSize
	d.feat = make([]float32, rows*taps*h)
	d.fused = make([]float32, rows*h)
	d.keys = make([]float32, rows*inst.MaxKVStride)
	d.vals = make([]float32, rows*inst.MaxKVStride)
	d.logits = make([]float32, cfg.BlockSize*cfg.VocabSize)
	d.embd = make([]float32, h)
	d.bias = make([]float32, cfg.VocabSize)
	d.markov = make([]float32, cfg.MarkovRank)
	return d, nil
}

func checkDSparkTarget(cfg *model.DSparkConfig, target *core.Instance) error {
	if target.Config == nil {
		return fmt.Errorf("dspark: target has no config")
	}
	switch tc := &target.Config.Config; {
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

// ContextLen is the number of target positions injected so far.
func (d *DSparkDraft) ContextLen() int { return d.inst.Pos }

// Truncate discards injected positions at and after pos, so a rejected
// speculative suffix can be re-injected from the target's accepted state.
func (d *DSparkDraft) Truncate(pos int) {
	d.inst.Pos = max(0, min(pos, d.inst.Pos))
}

// Inject projects the first rows of the target's most recent hidden-tap
// capture into the draft cache as target positions [pos, pos+rows). pos must
// equal ContextLen, so the cache always extends contiguously. rows may be
// fewer than were captured, e.g. only the accepted prefix of a verification.
func (d *DSparkDraft) Inject(pos, rows int) error {
	taps := &d.target.HiddenTaps
	m := d.inst
	switch {
	case pos != m.Pos:
		return fmt.Errorf("dspark: inject at %d, cache holds %d positions", pos, m.Pos)
	case rows <= 0 || rows > taps.Rows:
		return fmt.Errorf("dspark: inject %d rows, target captured %d", rows, taps.Rows)
	case pos+rows > m.MaxContext:
		return fmt.Errorf("dspark: inject %d rows at %d exceeds context %d", rows, pos, m.MaxContext)
	}
	h := d.Config.HiddenSize
	nTaps := len(d.Config.TargetLayerIDs)
	ops := m.Ops()
	for start := 0; start < rows; start += model.DSparkMaxBlockSize {
		n := min(model.DSparkMaxBlockSize, rows-start)
		feat := d.feat[:n*nTaps*h]
		for r := range n {
			for s := range nTaps {
				src := (s*taps.Capacity + start + r) * h
				copy(feat[(r*nTaps+s)*h:(r*nTaps+s+1)*h], taps.Host[src:src+h])
			}
		}
		fused := d.fused[:n*h]
		projectRows(fused, d.fc, feat, n)
		for r := range n {
			ops.RMSNorm(fused[r*h:(r+1)*h], fused[r*h:(r+1)*h], d.hiddenNorm, m.RMSEpsilon)
		}
		for li := range m.Layers {
			d.injectLayer(&m.Layers[li], fused, n, pos+start)
		}
	}
	m.Pos = pos + rows
	return nil
}

// injectLayer writes K/V for n fused feature rows at positions pos.. into one
// draft layer's cache: the layer's own K/V projections, k_norm and RoPE, with
// no input norm and no norm on V, as the reference does for context rows.
func (d *DSparkDraft) injectLayer(layer *Layer, fused []float32, n, pos int) {
	m := d.inst
	ops := m.Ops()
	stride := layer.HeadKV * layer.HeadDim
	keys, vals := d.keys[:n*stride], d.vals[:n*stride]
	applyRoPE, invFreq, attnScale := batchLayerRoPE(m, layer)

	projectRows(keys, layer.Wk, fused, n)
	projectRows(vals, layer.Wv, fused, n)
	cache := &layer.AttnCache
	for r := range n {
		k := keys[r*stride : (r+1)*stride]
		if len(layer.AttnKNorm) > 0 {
			applyHeadRMSNorm(ops, k, layer.AttnKNorm, m.RMSEpsilon, layer.HeadDim, layer.HeadKV)
		}
		if applyRoPE {
			ops.ApplyRoPE(k, layer.HeadKV, layer.HeadDim, pos+r, invFreq, attnScale)
		}
		cache.EnsurePos(pos + r)
		ops.StoreKV(-1, pos+r, stride,
			cache.K, cache.V, cache.K16, cache.V16,
			cache.KQ8, cache.VQ8, cache.KQ8S, cache.VQ8S,
			k, vals[r*stride:(r+1)*stride])
	}
}

// DraftBlock drafts n tokens following anchor, the last committed token, which
// sits at position ContextLen (not yet injected). It returns the drafted
// tokens and, when the checkpoint has a confidence head, each one's predicted
// acceptance probability. Drafting is greedy: block position i takes the
// argmax of the LM head logits plus the Markov bias chained from token i-1.
func (d *DSparkDraft) DraftBlock(anchor, n int) (tokens []int, conf []float32, err error) {
	c := d.Config
	if n < 2 || n > c.BlockSize {
		return nil, nil, fmt.Errorf("dspark: draft length %d outside [2,%d]", n, c.BlockSize)
	}
	if anchor < 0 || anchor >= c.VocabSize {
		return nil, nil, fmt.Errorf("dspark: anchor token %d outside vocab", anchor)
	}
	block := make([]int, n)
	block[0] = anchor
	for i := 1; i < n; i++ {
		block[i] = c.MaskTokenID
	}
	h, vocab := c.HiddenSize, c.VocabSize
	confIn := make([]float32, n)
	err = d.inst.ForwardBlockNonCausal(block, func(row int, logits, hidden []float32) {
		copy(d.logits[row*vocab:(row+1)*vocab], logits)
		if c.HasConfidence {
			RMSNorm(d.embd, hidden, d.inst.OutputNorm, d.inst.RMSEpsilon)
			confIn[row] = Dot(d.confW[:h], d.embd) + d.confB
		}
	})
	if err != nil {
		return nil, nil, fmt.Errorf("dspark: %w", err)
	}

	tokens = make([]int, n)
	if c.HasConfidence {
		conf = make([]float32, n)
	}
	prev := anchor
	for i := range n {
		d.markovW1.RowTo(d.markov, prev)
		MatVec(d.bias, d.markovW2, d.markov)
		col := d.logits[i*vocab : (i+1)*vocab]
		Add(col, d.bias)
		tokens[i] = argmaxHost(col)
		if c.HasConfidence {
			z := confIn[i] + Dot(d.confW[h:], d.markov)
			conf[i] = float32(1 / (1 + math.Exp(-float64(z))))
		}
		prev = tokens[i]
	}
	return tokens, conf, nil
}
