package audio

import (
	"fmt"
	"math"
)

// fsqBoundEps is the epsilon of LiquidCodecQuantizer._bound
// (modeling_liquid_codec.py:148): half_levels = (levels-1)*(1+eps)/2.
const fsqBoundEps = 1e-3

// FSQBasis returns the mixed-radix basis [prod(levels[:k])] used to pack the
// per-dimension indices of one codebook into a single code:
// for levels [8,7,6,6] this is [1, 8, 56, 336].
func FSQBasis(levels []int) []int {
	basis := make([]int, len(levels))
	prod := 1
	for k := range levels {
		basis[k] = prod
		prod *= levels[k]
	}
	return basis
}

// FSQQuantizer is the finite scalar quantization layer of LiquidCodec
// (modeling_liquid_codec.py:141-190): n_q codebooks, each of codebook_dim
// dimensions over the given levels. An optional linear projection maps
// quantizer_dim latents to the n_q*codebook_dim effective dimensions
// (project_in) and back (project_out); with quantizer_dim == effective dim the
// HF model uses nn.Identity, which is the nil-weight case here.
type FSQQuantizer struct {
	Levels    []int
	Basis     []int
	numBooks  int
	quantDim  int
	halfWidth []int     // levels // 2
	halfLevel []float64 // (levels-1)*(1+eps)/2
	offset    []float64 // 0.5 for even levels, else 0
	shift     []float64 // atanh(offset / halfLevel)

	projectInW  [][]float32 // [effectiveDim][quantDim], nil = identity
	projectInB  []float32
	projectOutW [][]float32 // [quantDim][effectiveDim], nil = identity
	projectOutB []float32
}

// NewFSQQuantizer builds the FSQ core for the given levels (one per codebook
// dimension), numCodebooks codebooks, and quantizer_dim-wide latents.
func NewFSQQuantizer(levels []int, numCodebooks, quantizerDim int) *FSQQuantizer {
	q := &FSQQuantizer{
		Levels:    append([]int(nil), levels...),
		Basis:     FSQBasis(levels),
		numBooks:  numCodebooks,
		quantDim:  quantizerDim,
		halfWidth: make([]int, len(levels)),
		halfLevel: make([]float64, len(levels)),
		offset:    make([]float64, len(levels)),
		shift:     make([]float64, len(levels)),
	}
	for k, level := range levels {
		q.halfWidth[k] = level / 2
		q.halfLevel[k] = float64(level-1) * (1 + fsqBoundEps) / 2
		if level%2 == 0 {
			q.offset[k] = 0.5
		}
		q.shift[k] = math.Atanh(q.offset[k] / q.halfLevel[k])
	}
	return q
}

// EffectiveDim is numCodebooks * len(levels), the width of the projected
// latent (48 for the Omni codec: 12 codebooks x 4 dims).
func (q *FSQQuantizer) EffectiveDim() int {
	return q.numBooks * len(q.Levels)
}

// MaxCode is prod(levels) - 1, the largest code of one codebook (2015).
func (q *FSQQuantizer) MaxCode() int {
	prod := 1
	for _, l := range q.Levels {
		prod *= l
	}
	return prod - 1
}

// SetProjectIn binds the project_in linear layer (nn.Linear(quantizer_dim,
// effective_dim)): weight [effectiveDim][quantizerDim] row-major plus bias.
func (q *FSQQuantizer) SetProjectIn(w [][]float32, b []float32) error {
	eff := q.EffectiveDim()
	if len(w) != eff || (len(w) > 0 && len(w[0]) != q.quantDim) {
		return fmt.Errorf("audio: project_in weight must be [%d][%d], got [%d][%d]", eff, q.quantDim, len(w), len(w[0]))
	}
	if len(b) != 0 && len(b) != eff {
		return fmt.Errorf("audio: project_in bias must have %d entries, got %d", eff, len(b))
	}
	q.projectInW, q.projectInB = w, b
	return nil
}

// SetProjectOut binds the project_out linear layer (nn.Linear(effective_dim,
// quantizer_dim)): weight [quantizerDim][effectiveDim] row-major plus bias.
func (q *FSQQuantizer) SetProjectOut(w [][]float32, b []float32) error {
	eff := q.EffectiveDim()
	if len(w) != q.quantDim || (len(w) > 0 && len(w[0]) != eff) {
		return fmt.Errorf("audio: project_out weight must be [%d][%d], got [%d][%d]", q.quantDim, eff, len(w), len(w[0]))
	}
	if len(b) != 0 && len(b) != q.quantDim {
		return fmt.Errorf("audio: project_out bias must have %d entries, got %d", q.quantDim, len(b))
	}
	q.projectOutW, q.projectOutB = w, b
	return nil
}

// Bound is LiquidCodecQuantizer._bound: the tanh saturating map that compresses
// a latent into the rounding range of its dimension,
//
//	bound(x) = tanh(x + atanh(offset/halfLevel)) * halfLevel - offset
//
// with halfLevel = (level-1)*(1+eps)/2 and offset = 0.5 for even levels.
func (q *FSQQuantizer) Bound(x float64, dim int) float64 {
	return math.Tanh(x+q.shift[dim])*q.halfLevel[dim] - q.offset[dim]
}

// CodeFromIndices packs one codebook's indices into a code:
// code = sum(idx_k * basis_k), in [0, prod(levels)).
func (q *FSQQuantizer) CodeFromIndices(idx []int) int {
	code := 0
	for k, iv := range idx {
		code += iv * q.Basis[k]
	}
	return code
}

// IndicesFromCode is the exact inverse of CodeFromIndices:
// idx_k = (code // basis_k) % levels_k.
func (q *FSQQuantizer) IndicesFromCode(code int) []int {
	idx := make([]int, len(q.Levels))
	for k := range q.Levels {
		idx[k] = (code / q.Basis[k]) % q.Levels[k]
	}
	return idx
}

// EncodeIndices quantizes one codebook's codebook_dim bounded latents to a
// code: idx_k = round-half-even(bound(x_k)) + halfWidth_k (torch.round is
// banker's rounding), then the mixed-radix packing.
func (q *FSQQuantizer) EncodeIndices(vec []float64) int {
	idx := make([]int, len(q.Levels))
	for k, x := range vec {
		idx[k] = int(math.RoundToEven(q.Bound(x, k))) + q.halfWidth[k]
	}
	return q.CodeFromIndices(idx)
}

// Encode runs the full quantizer encode on one frame's latents
// (modeling_liquid_codec.py:161-172): project_in 256->48, reshape to
// [n_q][codebook_dim], bound, round, pack. Returns one code per codebook, each
// in [0, prod(levels)).
func (q *FSQQuantizer) Encode(latents []float32) []int {
	if len(latents) != q.quantDim {
		panic(fmt.Sprintf("audio: FSQ Encode needs %d latents, got %d", q.quantDim, len(latents)))
	}
	eff := q.projectIn(latents)
	dim := len(q.Levels)
	codes := make([]int, q.numBooks)
	for c := 0; c < q.numBooks; c++ {
		vec := eff[c*dim : (c+1)*dim]
		codes[c] = q.EncodeIndices(vec)
	}
	return codes
}

// Decode is the exact inverse decode (modeling_liquid_codec.py:174-183):
// code -> indices -> latents (idx - halfWidth)/halfWidth -> project_out.
// Returns quantizer_dim latents.
func (q *FSQQuantizer) Decode(codes []int) []float32 {
	if len(codes) != q.numBooks {
		panic(fmt.Sprintf("audio: FSQ Decode needs %d codes, got %d", q.numBooks, len(codes)))
	}
	dim := len(q.Levels)
	eff := make([]float64, q.EffectiveDim())
	for c, code := range codes {
		if code < 0 || code > q.MaxCode() {
			panic(fmt.Sprintf("audio: FSQ code %d out of [0, %d]", code, q.MaxCode()))
		}
		idx := q.IndicesFromCode(code)
		for k := range dim {
			hw := float64(q.halfWidth[k])
			eff[c*dim+k] = (float64(idx[k]) - hw) / hw
		}
	}
	return q.projectOut(eff)
}

// projectIn applies y[o] = sum_i W[o][i]*x[i] + b[o] (nn.Linear semantics), or
// the identity when no weights are bound (quantizer_dim == effective dim).
func (q *FSQQuantizer) projectIn(x []float32) []float64 {
	eff := q.EffectiveDim()
	if q.projectInW == nil {
		if len(x) != eff {
			panic(fmt.Sprintf("audio: identity project_in needs %d latents, got %d", eff, len(x)))
		}
		out := make([]float64, eff)
		for i, v := range x {
			out[i] = float64(v)
		}
		return out
	}
	out := make([]float64, eff)
	for o := range eff {
		sum := float64(q.projectInB[o])
		row := q.projectInW[o]
		for i, v := range x {
			sum += float64(row[i]) * float64(v)
		}
		out[o] = sum
	}
	return out
}

// projectOut applies y[o] = sum_i W[o][i]*x[i] + b[o], or the identity when no
// weights are bound.
func (q *FSQQuantizer) projectOut(x []float64) []float32 {
	if q.projectOutW == nil {
		if len(x) != q.quantDim {
			panic(fmt.Sprintf("audio: identity project_out needs %d latents, got %d", q.quantDim, len(x)))
		}
		out := make([]float32, q.quantDim)
		for i, v := range x {
			out[i] = float32(v)
		}
		return out
	}
	out := make([]float32, q.quantDim)
	for o := 0; o < q.quantDim; o++ {
		sum := float64(q.projectOutB[o])
		row := q.projectOutW[o]
		for i, v := range x {
			sum += float64(row[i]) * v
		}
		out[o] = float32(sum)
	}
	return out
}
