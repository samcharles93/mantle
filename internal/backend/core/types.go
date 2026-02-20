package core

import "runtime"

// TilingConfig holds runtime-configurable tiling parameters.
type TilingConfig struct {
	TileM int
	TileN int
	TileK int
}

// QuantCache stores pre-unpacked quantized weights for faster matvec.
type QuantCache struct {
	Q            []int8
	Scales       []float32
	BlocksPerRow int
}

func (qc *QuantCache) validFor(m *Mat) bool {
	if qc == nil || m == nil {
		return false
	}
	if qc.BlocksPerRow <= 0 || m.R <= 0 {
		return false
	}
	blocksPerRow := (m.C + 31) / 32
	if blocksPerRow != qc.BlocksPerRow {
		return false
	}
	totalBlocks, ok := mulInt(m.R, blocksPerRow)
	if !ok {
		return false
	}
	qLen, ok := mulInt(totalBlocks, 32)
	if !ok {
		return false
	}
	if len(qc.Q) < qLen || len(qc.Scales) < totalBlocks {
		return false
	}
	return true
}

// ValidFor reports whether the quant cache matches matrix dimensions/layout.
func (qc *QuantCache) ValidFor(m *Mat) bool {
	return qc.validFor(m)
}

// QuantVec stores transient quantization of input vectors.
type QuantVec struct {
	Q      []int8
	Q16    []int16
	Scales []float32
	Blocks int
}

// AttnCache holds KV cache for attention mechanism.
type AttnCache struct {
	K        []float32
	V        []float32
	K16      []uint16
	V16      []uint16
	KQ8      []int8
	VQ8      []int8
	KQ8S     []float32
	VQ8S     []float32
	KvStride int
	CacheLen int
	Cap      int
}

const kvCachePageSize = 256

// EnsurePos grows cache slices so position pos is addressable.
func (c *AttnCache) EnsurePos(pos int) {
	needed := pos + 1
	if needed <= c.Cap {
		return
	}
	newCap := min(((needed+kvCachePageSize-1)/kvCachePageSize)*kvCachePageSize, c.CacheLen)
	grow := newCap - c.Cap
	stride := c.KvStride
	if c.K != nil {
		c.K = append(c.K, make([]float32, grow*stride)...)
	}
	if c.V != nil {
		c.V = append(c.V, make([]float32, grow*stride)...)
	}
	if c.K16 != nil {
		c.K16 = append(c.K16, make([]uint16, grow*stride)...)
	}
	if c.V16 != nil {
		c.V16 = append(c.V16, make([]uint16, grow*stride)...)
	}
	if c.KQ8 != nil {
		c.KQ8 = append(c.KQ8, make([]int8, grow*stride)...)
	}
	if c.VQ8 != nil {
		c.VQ8 = append(c.VQ8, make([]int8, grow*stride)...)
	}
	if c.KQ8S != nil {
		c.KQ8S = append(c.KQ8S, make([]float32, grow)...)
	}
	if c.VQ8S != nil {
		c.VQ8S = append(c.VQ8S, make([]float32, grow)...)
	}
	c.Cap = newCap
}

// ShortConvState holds state for recurrent attention convolution.
type ShortConvState struct {
	Buf       []float32
	KernelLen int
}

// Ops defines the vector/mat ops used by runtime execution.
type Ops interface {
	MatVec(dst []float32, w *Mat, x []float32)
	MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec)
	RMSNorm(dst, src, weight []float32, eps float32)
	Softmax(x []float32)
	ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32)
	StoreKV(layerIndex, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32)
}

// DefaultOps is a placeholder fallback ops implementation.
type DefaultOps struct{}

func (DefaultOps) MatVec(_ []float32, _ *Mat, _ []float32)                       {}
func (DefaultOps) MatVecWithQuant(_ []float32, _ *Mat, _ []float32, _ *QuantVec) {}
func (DefaultOps) RMSNorm(dst, src, _ []float32, _ float32)                      { copy(dst, src) }
func (DefaultOps) Softmax(_ []float32)                                           {}
func (DefaultOps) ApplyRoPE(_ []float32, _, _, _ int, _ []float64, _ float32) {
}
func (DefaultOps) StoreKV(_ int, _ int, _ int, kDst, vDst []float32, _ []uint16, _ []uint16, _ []int8, _ []int8, _ []float32, _ []float32, k, v []float32) {
	if len(kDst) >= len(k) {
		copy(kDst, k)
	}
	if len(vDst) >= len(v) {
		copy(vDst, v)
	}
}

// bindDefaultOps lazily binds default ops when none are configured.
func (m *Instance) bindDefaultOps() {
	if m == nil {
		return
	}
	if m.ops == nil {
		m.ops = DefaultOps{}
	}
}

// BindDefaultOps binds fallback ops if none are configured.
func (m *Instance) BindDefaultOps() {
	m.bindDefaultOps()
}

type AttnPool struct {
	Size int
}

func AttnWorkersFor(nHead int) int {
	workers := max(runtime.GOMAXPROCS(0), 1)
	if nHead > 0 && workers > nHead {
		workers = nHead
	}
	if workers < 1 {
		return 1
	}
	return workers
}

func NewAttnPool(workers, _ int) *AttnPool {
	if workers < 1 {
		workers = 1
	}
	return &AttnPool{Size: workers}
}

func mulInt(a, b int) (int, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	max := int(^uint(0) >> 1)
	if a > max/b {
		return 0, false
	}
	return a * b, true
}
