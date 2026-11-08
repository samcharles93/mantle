//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
	"unsafe"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// defaultVisionMaxPatches caps the number of patches the device vision tower
// will process in one Encode call. Above it Encode falls back to the host
// tower (logged). 16384 patches = a 128x128 patch grid (2048x2048 px image)
// and roughly 256 MiB of f32 activations, which fits the 4 GiB budget next to
// the preloaded f32 vision weights; the processor maximum (65536 patches =
// 4096x4096 px) would need ~2.5 GiB of activations plus the O(L^2) score
// traffic the tiled kernel avoids but the GEMMs do not, so it stays on host.
const defaultVisionMaxPatches = 16384

// visionMaxPatches returns the device vision patch-count gate, overridable via
// MANTLE_CUDA_VISION_MAX_PATCHES for tuning.
func visionMaxPatches() int {
	if s := os.Getenv("MANTLE_CUDA_VISION_MAX_PATCHES"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return defaultVisionMaxPatches
}

// visionDeviceLayer holds one vision-tower layer's weights on the device (F32).
type visionDeviceLayer struct {
	isAttn bool

	q, k, v, o   native.DeviceBuffer // attention projections
	qNorm, kNorm native.DeviceBuffer // per-head-dim RMSNorm weights

	convIn, convOut native.DeviceBuffer // conv in/out projections
	convKernel      native.DeviceBuffer // [C][k][k]

	opNorm, ffnNorm native.DeviceBuffer
	w1, w2, w3      native.DeviceBuffer
}

// visionDeviceWeights holds the whole vision tower + projector resident on the
// device as F32 (bf16 checkpoint weights are dequantized once at preload so the
// device GEMMs accumulate in f32 exactly like the host oracle).
type visionDeviceWeights struct {
	patchifier native.DeviceBuffer
	embedNorm  native.DeviceBuffer
	layers     []visionDeviceLayer

	projNormW, projNormB native.DeviceBuffer
	proj1, proj2         native.DeviceBuffer
	proj1Bias, proj2Bias native.DeviceBuffer // zero DeviceBuffer when absent

	invFreq []float64 // axial RoPE inverse frequencies, recomputed from cfg
}

// visionEncoder is the device-side core.VisionEncoder for LFM3-VL: it runs the
// patchifier, embedding RMSNorm, the hybrid conv/attention layers with axial
// 2D RoPE, and the multimodal projector on the CUDA device. Images whose patch
// count exceeds visionMaxPatches() fall back to the host tower (logged).
type visionEncoder struct {
	ops    *Ops
	stream native.Stream
	blas   native.BlasHandle

	host core.VisionEncoder // host tower: weight source and fallback oracle
	cfg  *core.VisionConfig
	w    visionDeviceWeights

	maxPatches int

	mu sync.Mutex

	// scratch buffers, grown on demand and freed by close
	pixDev        native.DeviceBuffer
	hDev          native.DeviceBuffer
	normedDev     native.DeviceBuffer
	qDev          native.DeviceBuffer
	kDev          native.DeviceBuffer
	vDev          native.DeviceBuffer
	attnDev       native.DeviceBuffer
	resDev        native.DeviceBuffer
	in3Dev        native.DeviceBuffer
	convTokDev    native.DeviceBuffer
	ffnInDev      native.DeviceBuffer
	upDev         native.DeviceBuffer
	gateDev       native.DeviceBuffer
	downDev       native.DeviceBuffer
	tmp1Dev       native.DeviceBuffer
	outDev        native.DeviceBuffer
	cosDev        native.DeviceBuffer
	sinDev        native.DeviceBuffer
	segStartDev   native.DeviceBuffer
	segLenDev     native.DeviceBuffer
	frameStartDev native.DeviceBuffer
	frameHDev     native.DeviceBuffer
	frameWDev     native.DeviceBuffer
}

// preloadVisionWeights uploads the vision tower + projector of a multimodal
// model to the device as F32 and binds a device visionEncoder on o. Models
// without a vision tower (or with a non-LFM3-VL encoder implementation) are
// left host-only.
func (o *Ops) preloadVisionWeights(m *core.Instance) error {
	if o == nil || m == nil || m.Vision == nil {
		return nil
	}
	tower, ok := m.Vision.(*simd.VisionTower)
	if !ok {
		return nil // unknown encoder implementation: keep the host path
	}
	// Gate on free VRAM: the vision weights are uploaded as F32 (bf16 payloads
	// double), and allocating them when the device is already full would push
	// the allocator into the Unified-Memory fallback, whose page migrations
	// fault under pressure. Keep a headroom for KV cache + scratch.
	need := estimateVisionWeightBytesF32(m)
	const headroom = int64(384) << 20
	if free, _, err := native.MemInfo(); err == nil && free > 0 && free < need+headroom {
		return fmt.Errorf("need ~%d MiB vision weights + %d MiB headroom, free %d MiB",
			need>>20, headroom>>20, free>>20)
	}
	enc, err := newVisionEncoder(o, tower)
	if err != nil {
		return err
	}
	o.vision = enc
	return nil
}

// estimateVisionWeightBytesF32 returns the device bytes the vision tower +
// projector occupy once uploaded as F32 (bf16/f16 raw payloads double).
func estimateVisionWeightBytesF32(m *core.Instance) int64 {
	if m == nil || m.Vision == nil {
		return 0
	}
	tower, ok := m.Vision.(*simd.VisionTower)
	if !ok {
		return 0
	}
	var total int64
	addMat := func(w *core.Mat) {
		if w == nil || w.R <= 0 || w.C <= 0 {
			return
		}
		if w.DType == mcf.DTypeF32 && len(w.Data) > 0 {
			total += int64(len(w.Data)) * 4
			return
		}
		// bf16/f16 raw doubles when expanded to f32; quant payloads expand to
		// full f32 (R*C*4).
		if mcf.DTypeRequiresAligned64(w.DType) {
			total += int64(w.R) * int64(w.C) * 4
			return
		}
		if len(w.Raw) > 0 {
			total += int64(len(w.Raw)) * 2
		}
	}
	addVec := func(v []float32) { total += int64(len(v)) * 4 }
	addMat(tower.Patchifier)
	addVec(tower.EmbedNorm)
	for i := range tower.Layers {
		l := &tower.Layers[i]
		addVec(l.OpNorm)
		addVec(l.FfnNorm)
		addMat(l.W1)
		addMat(l.W2)
		addMat(l.W3)
		if l.IsAttn {
			addMat(l.Q)
			addMat(l.K)
			addMat(l.V)
			addMat(l.O)
			addVec(l.QNorm)
			addVec(l.KNorm)
		} else {
			addMat(l.ConvIn)
			addVec(l.ConvKernel)
			addMat(l.ConvOut)
		}
	}
	addVec(tower.ProjNormW)
	addVec(tower.ProjNormB)
	addMat(tower.Proj1)
	addVec(tower.Proj1Bias)
	addMat(tower.Proj2)
	addVec(tower.Proj2Bias)
	return total
}

// decodeMatF32 decodes a weight matrix to f32 on the host. F32 mats reuse
// their Data slice; bf16/f16/quantized raw payloads are decoded row by row via
// core.Mat.RowTo, which implements exactly the host oracle's decode semantics.
func decodeMatF32(w *core.Mat) (data []float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			data = nil
			err = fmt.Errorf("vision: decode [%d %d] dtype %s failed: %v", w.R, w.C, dtypeString(w.DType), r)
		}
	}()
	n := w.R * w.C
	if w.DType == mcf.DTypeF32 && len(w.Data) >= n {
		return w.Data[:n], nil
	}
	if len(w.Raw) == 0 {
		return nil, fmt.Errorf("vision: empty raw payload for [%d %d] %s", w.R, w.C, dtypeString(w.DType))
	}
	data = make([]float32, n)
	for r := 0; r < w.R; r++ {
		w.RowTo(data[r*w.C:(r+1)*w.C], r)
	}
	return data, nil
}

// uploadMatF32 uploads a weight matrix to the device as F32 (dequantized on
// the host when the checkpoint stores bf16/f16/quant payloads).
func uploadMatF32(w *core.Mat) (native.DeviceBuffer, error) {
	if w == nil || w.R <= 0 || w.C <= 0 {
		return native.DeviceBuffer{}, fmt.Errorf("vision: nil or empty weight matrix")
	}
	data, err := decodeMatF32(w)
	if err != nil {
		return native.DeviceBuffer{}, err
	}
	n := w.R * w.C
	buf, err := native.AllocDevice(int64(n) * 4)
	if err != nil {
		return native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&data[0]), int64(n)*4); err != nil {
		_ = buf.Free()
		return native.DeviceBuffer{}, err
	}
	return buf, nil
}

func uploadVecF32(name string, v []float32) (native.DeviceBuffer, error) {
	if len(v) == 0 {
		return native.DeviceBuffer{}, nil
	}
	buf, err := native.AllocDevice(int64(len(v)) * 4)
	if err != nil {
		return native.DeviceBuffer{}, fmt.Errorf("vision: alloc %s: %w", name, err)
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&v[0]), int64(len(v))*4); err != nil {
		_ = buf.Free()
		return native.DeviceBuffer{}, fmt.Errorf("vision: upload %s: %w", name, err)
	}
	return buf, nil
}

// newVisionEncoder uploads all vision tower weights as F32 and returns the
// device encoder. The host tower is kept as the fallback oracle.
func newVisionEncoder(o *Ops, tower *simd.VisionTower) (enc *visionEncoder, err error) {
	cfg := tower.Config()
	if cfg == nil || cfg.HiddenSize <= 0 || cfg.NumLayers != len(tower.Layers) {
		return nil, fmt.Errorf("vision: inconsistent tower config")
	}
	e := &visionEncoder{
		ops:        o,
		stream:     o.stream,
		blas:       o.blas,
		host:       tower,
		cfg:        cfg,
		maxPatches: visionMaxPatches(),
	}
	// On any upload error, free the partially uploaded weights so a skipped
	// device tower leaves no dangling device allocations.
	defer func() {
		if err != nil {
			e.close()
			enc = nil
		}
	}()
	w := &e.w
	if w.patchifier, err = uploadMatF32(tower.Patchifier); err != nil {
		return nil, fmt.Errorf("vision: patchifier: %w", err)
	}
	if w.embedNorm, err = uploadVecF32("embedding_norm", tower.EmbedNorm); err != nil {
		return nil, err
	}
	w.layers = make([]visionDeviceLayer, len(tower.Layers))
	for i := range tower.Layers {
		l := &tower.Layers[i]
		dl := &w.layers[i]
		dl.isAttn = l.IsAttn
		if dl.opNorm, err = uploadVecF32(fmt.Sprintf("layers.%d.operator_norm", i), l.OpNorm); err != nil {
			return nil, err
		}
		if dl.ffnNorm, err = uploadVecF32(fmt.Sprintf("layers.%d.ffn_norm", i), l.FfnNorm); err != nil {
			return nil, err
		}
		if dl.w1, err = uploadMatF32(l.W1); err != nil {
			return nil, fmt.Errorf("vision: layers.%d.w1: %w", i, err)
		}
		if dl.w2, err = uploadMatF32(l.W2); err != nil {
			return nil, fmt.Errorf("vision: layers.%d.w2: %w", i, err)
		}
		if dl.w3, err = uploadMatF32(l.W3); err != nil {
			return nil, fmt.Errorf("vision: layers.%d.w3: %w", i, err)
		}
		if l.IsAttn {
			if dl.q, err = uploadMatF32(l.Q); err != nil {
				return nil, fmt.Errorf("vision: layers.%d.q: %w", i, err)
			}
			if dl.k, err = uploadMatF32(l.K); err != nil {
				return nil, fmt.Errorf("vision: layers.%d.k: %w", i, err)
			}
			if dl.v, err = uploadMatF32(l.V); err != nil {
				return nil, fmt.Errorf("vision: layers.%d.v: %w", i, err)
			}
			if dl.o, err = uploadMatF32(l.O); err != nil {
				return nil, fmt.Errorf("vision: layers.%d.o: %w", i, err)
			}
			if dl.qNorm, err = uploadVecF32(fmt.Sprintf("layers.%d.q_layernorm", i), l.QNorm); err != nil {
				return nil, err
			}
			if dl.kNorm, err = uploadVecF32(fmt.Sprintf("layers.%d.k_layernorm", i), l.KNorm); err != nil {
				return nil, err
			}
		} else {
			if dl.convIn, err = uploadMatF32(l.ConvIn); err != nil {
				return nil, fmt.Errorf("vision: layers.%d.conv.in_proj: %w", i, err)
			}
			if dl.convKernel, err = uploadVecF32(fmt.Sprintf("layers.%d.conv.spatial_conv", i), l.ConvKernel); err != nil {
				return nil, err
			}
			if dl.convOut, err = uploadMatF32(l.ConvOut); err != nil {
				return nil, fmt.Errorf("vision: layers.%d.conv.out_proj: %w", i, err)
			}
		}
	}
	if w.projNormW, err = uploadVecF32("projector.layer_norm.weight", tower.ProjNormW); err != nil {
		return nil, err
	}
	if w.projNormB, err = uploadVecF32("projector.layer_norm.bias", tower.ProjNormB); err != nil {
		return nil, err
	}
	if w.proj1, err = uploadMatF32(tower.Proj1); err != nil {
		return nil, fmt.Errorf("vision: projector.linear_1: %w", err)
	}
	if w.proj1Bias, err = uploadVecF32("projector.linear_1.bias", tower.Proj1Bias); err != nil {
		return nil, err
	}
	if w.proj2, err = uploadMatF32(tower.Proj2); err != nil {
		return nil, fmt.Errorf("vision: projector.linear_2: %w", err)
	}
	if w.proj2Bias, err = uploadVecF32("projector.linear_2.bias", tower.Proj2Bias); err != nil {
		return nil, err
	}

	// Axial RoPE inverse frequencies, recomputed exactly like the host tower.
	spatialDim := cfg.HeadDim / 2
	w.invFreq = make([]float64, spatialDim/2)
	for i := range w.invFreq {
		w.invFreq[i] = 1.0 / math.Pow(cfg.RopeTheta, float64(2*i)/float64(spatialDim))
	}
	return e, nil
}

// Config implements core.VisionEncoder.
func (e *visionEncoder) Config() *core.VisionConfig { return e.cfg }

// Encode implements core.VisionEncoder on the device. Patch counts above the
// configured gate fall back to the host tower (logged once per call).
func (e *visionEncoder) Encode(pixelValues []float32, grids [][3]int) ([][][]float32, error) {
	cfg := e.cfg
	patchDim := cfg.PatchDim()
	if patchDim <= 0 || len(pixelValues)%patchDim != 0 {
		return nil, fmt.Errorf("vision: pixel payload size %d not a multiple of patch dim %d", len(pixelValues), patchDim)
	}
	numPatches := len(pixelValues) / patchDim
	total := 0
	for _, g := range grids {
		total += g[0] * g[1] * g[2]
	}
	if total != numPatches {
		return nil, fmt.Errorf("vision: grids cover %d patches, payload has %d", total, numPatches)
	}
	if numPatches == 0 {
		return make([][][]float32, len(grids)), nil
	}
	// Size gate: the device tower is used only when the patch count fits both
	// the configured cap and the currently free VRAM (activations are ~75 KiB
	// per patch in f32 at LFM3-VL-2B's shape, plus a headroom for KV/scratch).
	// Anything larger falls back to the host tower (logged).
	gate := e.maxPatches
	if free, _, err := native.MemInfo(); err == nil && free > 0 {
		if mp := e.maxPatchesForFree(free); mp < gate {
			gate = mp
		}
	}
	if numPatches > gate {
		fmt.Fprintf(os.Stderr, "CUDA vision: %d patches exceeds device gate %d (cap %d); running vision tower on host\n", numPatches, gate, e.maxPatches)
		return e.host.Encode(pixelValues, grids)
	}

	e.mu.Lock()
	res, err := e.encodeDevice(pixelValues, grids, numPatches)
	e.mu.Unlock()
	if err != nil && native.IsOOM(err) {
		// Safety net: if the device ran out of memory mid-encode (e.g. the
		// text decoder grew its KV cache between the gate and here), serve
		// the image from the host tower instead of failing the request.
		fmt.Fprintf(os.Stderr, "CUDA vision: device encode OOM (%v); running vision tower on host\n", err)
		return e.host.Encode(pixelValues, grids)
	}
	return res, err
}

// visionEncodeHeadroom is the free VRAM kept aside for KV cache + scratch when
// sizing the device vision gate.
const visionEncodeHeadroom = int64(256) << 20

// activationBytesPerPatch estimates the device scratch the encoder allocates
// per patch (f32) at this model's shape.
func (e *visionEncoder) activationBytesPerPatch() int64 {
	cfg := e.cfg
	h := int64(cfg.HiddenSize)
	per := int64(cfg.PatchDim())                 // pixel payload
	per += 6 * h                                 // h, normed, res, convTok, ffnIn, down
	per += 2 * int64(cfg.NumHeads*cfg.HeadDim)   // q, attnOut
	per += 2 * int64(cfg.NumKVHeads*cfg.HeadDim) // k, v
	per += 3 * h                                 // conv in-proj (B,C,x)
	per += 2 * int64(cfg.IntermediateSize)       // ffn up, gate
	per += 2 * int64(cfg.HeadDim)                // cos/sin tables
	per *= 4
	group := int64(cfg.DownsampleFactor * cfg.DownsampleFactor)
	if group <= 0 {
		group = 1
	}
	per += 4 * (int64(cfg.ProjectorHiddenSize) + int64(cfg.TextHiddenSize)) / group // projector
	return per
}

// maxPatchesForFree returns how many patches fit in freeBytes given the
// per-patch activation footprint, capped at e.maxPatches.
func (e *visionEncoder) maxPatchesForFree(free int64) int {
	bpp := e.activationBytesPerPatch()
	if bpp <= 0 {
		return e.maxPatches
	}
	avail := free - visionEncodeHeadroom
	if avail <= 0 {
		return 0
	}
	return min(int(avail/bpp), e.maxPatches)
}

// visionFrameDesc is one attention/conv segment: the patches of one frame.
type visionFrameDesc struct {
	start, h, w int
}

// encodeDevice runs the full tower on the device.
func (e *visionEncoder) encodeDevice(pixelValues []float32, grids [][3]int, n int) ([][][]float32, error) {
	cfg := e.cfg
	w := &e.w
	stream := e.stream
	hidden := cfg.HiddenSize
	heads := cfg.NumHeads
	kvHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	ffn := cfg.IntermediateSize
	eps := float32(cfg.NormEps)
	patchDim := cfg.PatchDim()

	// Frame segments: each temporal frame of each grid is one segment.
	var frames []visionFrameDesc
	off := 0
	maxFrameTokens := 0
	for _, g := range grids {
		t, gh, gw := g[0], g[1], g[2]
		for range t {
			frames = append(frames, visionFrameDesc{start: off, h: gh, w: gw})
			if gh*gw > maxFrameTokens {
				maxFrameTokens = gh * gw
			}
			off += gh * gw
		}
	}
	nSeg := len(frames)

	// Scratch buffers (grown on demand).
	f4 := func(elems int) int64 { return int64(elems) * 4 }
	if err := ensureVisionBuf(&e.pixDev, f4(n*patchDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.hDev, f4(n*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.normedDev, f4(n*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.qDev, f4(n*heads*headDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.kDev, f4(n*kvHeads*headDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.vDev, f4(n*kvHeads*headDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.attnDev, f4(n*heads*headDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.resDev, f4(n*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.in3Dev, f4(n*3*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.convTokDev, f4(n*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.ffnInDev, f4(n*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.upDev, f4(n*ffn)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.gateDev, f4(n*ffn)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.downDev, f4(n*hidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.cosDev, f4(n*headDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.sinDev, f4(n*headDim)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.segStartDev, f4(nSeg)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.segLenDev, f4(nSeg)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.frameStartDev, f4(nSeg)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.frameHDev, f4(nSeg)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.frameWDev, f4(nSeg)); err != nil {
		return nil, err
	}

	// Upload the pixel payload.
	if err := native.MemcpyH2DAsync(e.pixDev, unsafe.Pointer(&pixelValues[0]), f4(n*patchDim), stream); err != nil {
		return nil, fmt.Errorf("vision: upload pixels: %w", err)
	}

	// Patchifier projection: h = Patchifier @ pix^T, [n, hidden] row-major.
	if err := e.gemmF32(w.patchifier, e.pixDev, patchDim, e.hDev, hidden, n); err != nil {
		return nil, err
	}
	// Embedding RMSNorm over all patch rows (shared weight).
	if err := native.RMSNormBatchedF32(e.hDev, e.hDev, w.embedNorm, eps, hidden, n, stream); err != nil {
		return nil, fmt.Errorf("vision: embedding norm: %w", err)
	}

	// Axial 2D RoPE cos/sin tables (host-built in float64, identical to the
	// host tower), block-major over merge blocks, repeated per temporal frame.
	cosT, sinT := e.buildRoPETables(grids, n)
	if err := native.MemcpyH2DAsync(e.cosDev, unsafe.Pointer(&cosT[0]), f4(n*headDim), stream); err != nil {
		return nil, fmt.Errorf("vision: upload cos: %w", err)
	}
	if err := native.MemcpyH2DAsync(e.sinDev, unsafe.Pointer(&sinT[0]), f4(n*headDim), stream); err != nil {
		return nil, fmt.Errorf("vision: upload sin: %w", err)
	}

	// Segment / frame descriptors.
	segStart := make([]int32, nSeg)
	segLen := make([]int32, nSeg)
	frameStart := make([]int32, nSeg)
	frameH := make([]int32, nSeg)
	frameW := make([]int32, nSeg)
	for i, fr := range frames {
		segStart[i] = int32(fr.start)
		segLen[i] = int32(fr.h * fr.w)
		frameStart[i] = int32(fr.start)
		frameH[i] = int32(fr.h)
		frameW[i] = int32(fr.w)
	}
	if err := native.MemcpyH2DAsync(e.segStartDev, unsafe.Pointer(&segStart[0]), f4(nSeg), stream); err != nil {
		return nil, fmt.Errorf("vision: upload segStart: %w", err)
	}
	if err := native.MemcpyH2DAsync(e.segLenDev, unsafe.Pointer(&segLen[0]), f4(nSeg), stream); err != nil {
		return nil, fmt.Errorf("vision: upload segLen: %w", err)
	}
	if err := native.MemcpyH2DAsync(e.frameStartDev, unsafe.Pointer(&frameStart[0]), f4(nSeg), stream); err != nil {
		return nil, fmt.Errorf("vision: upload frameStart: %w", err)
	}
	if err := native.MemcpyH2DAsync(e.frameHDev, unsafe.Pointer(&frameH[0]), f4(nSeg), stream); err != nil {
		return nil, fmt.Errorf("vision: upload frameH: %w", err)
	}
	if err := native.MemcpyH2DAsync(e.frameWDev, unsafe.Pointer(&frameW[0]), f4(nSeg), stream); err != nil {
		return nil, fmt.Errorf("vision: upload frameW: %w", err)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for li := range w.layers {
		l := &w.layers[li]
		if l.isAttn {
			if err := e.attnLayer(l, n, hidden, heads, kvHeads, headDim, eps, scale, nSeg, maxFrameTokens); err != nil {
				return nil, fmt.Errorf("vision: layer %d attn: %w", li, err)
			}
		} else {
			if err := e.convLayer(l, n, hidden, eps, nSeg, maxFrameTokens, cfg.ConvKernelSize, cfg.MergeSize); err != nil {
				return nil, fmt.Errorf("vision: layer %d conv: %w", li, err)
			}
		}
		// FFN (pre-norm residual MLP with sigmoid gate: up = gate * silu(up)).
		if err := native.RMSNormBatchedF32(e.ffnInDev, e.hDev, l.ffnNorm, eps, hidden, n, stream); err != nil {
			return nil, fmt.Errorf("vision: layer %d ffn norm: %w", li, err)
		}
		if err := e.gemmF32(l.w1, e.ffnInDev, hidden, e.upDev, ffn, n); err != nil {
			return nil, fmt.Errorf("vision: layer %d w1: %w", li, err)
		}
		if err := e.gemmF32(l.w3, e.ffnInDev, hidden, e.gateDev, ffn, n); err != nil {
			return nil, fmt.Errorf("vision: layer %d w3: %w", li, err)
		}
		// SiluMulF32(a, b, out) = silu(a)*b; with a=up, b=gate this is
		// gate * up * sigmoid(up), matching the host mlp exactly.
		if err := native.SiluMulF32(e.upDev, e.gateDev, e.upDev, n*ffn, stream); err != nil {
			return nil, fmt.Errorf("vision: layer %d gate: %w", li, err)
		}
		if err := e.gemmF32(l.w2, e.upDev, ffn, e.downDev, hidden, n); err != nil {
			return nil, fmt.Errorf("vision: layer %d w2: %w", li, err)
		}
		if err := native.AddVectorsF32(e.hDev, e.downDev, n*hidden, stream); err != nil {
			return nil, fmt.Errorf("vision: layer %d ffn residual: %w", li, err)
		}
	}

	// Projector: LayerNorm -> merge down^2 consecutive tokens (a pure reshape
	// of the [n, hidden] buffer) -> linear_1 -> bias+gelu -> linear_2 -> bias.
	down := cfg.DownsampleFactor
	group := down * down
	if group <= 0 {
		group = 1
	}
	merged := n / group
	projHidden := cfg.ProjectorHiddenSize
	textHidden := cfg.TextHiddenSize
	if err := ensureVisionBuf(&e.tmp1Dev, f4(merged*projHidden)); err != nil {
		return nil, err
	}
	if err := ensureVisionBuf(&e.outDev, f4(merged*textHidden)); err != nil {
		return nil, err
	}
	if err := native.LayerNormRowsF32(e.hDev, e.hDev, w.projNormW, w.projNormB, float32(cfg.ProjectorLayerNormEps), n, hidden, stream); err != nil {
		return nil, fmt.Errorf("vision: projector layer norm: %w", err)
	}
	// B = hDev viewed as [merged, group*hidden] row-major == [k x merged]
	// col-major with ldb = group*hidden, so the merge is free.
	if err := e.gemmF32B(w.proj1, e.hDev, group*hidden, group*hidden, e.tmp1Dev, projHidden, merged); err != nil {
		return nil, fmt.Errorf("vision: projector linear_1: %w", err)
	}
	if err := native.GeluExactBiasF32(e.tmp1Dev, w.proj1Bias, merged, projHidden, stream); err != nil {
		return nil, fmt.Errorf("vision: projector gelu: %w", err)
	}
	if err := e.gemmF32(w.proj2, e.tmp1Dev, projHidden, e.outDev, textHidden, merged); err != nil {
		return nil, fmt.Errorf("vision: projector linear_2: %w", err)
	}
	if w.proj2Bias.Ptr() != nil {
		if err := native.AddBiasRowsF32(e.outDev, w.proj2Bias, merged, textHidden, stream); err != nil {
			return nil, fmt.Errorf("vision: projector linear_2 bias: %w", err)
		}
	}

	if err := stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("vision: sync: %w", err)
	}
	out := make([]float32, merged*textHidden)
	if err := native.MemcpyD2H(unsafe.Pointer(&out[0]), e.outDev, f4(merged*textHidden)); err != nil {
		return nil, fmt.Errorf("vision: download output: %w", err)
	}

	res := make([][][]float32, len(grids))
	row := 0
	for gi, g := range grids {
		nTok := cfg.TokensPerImage(g)
		rows := make([][]float32, nTok)
		for i := range nTok {
			rows[i] = out[(row+i)*textHidden : (row+i+1)*textHidden]
		}
		res[gi] = rows
		row += nTok
	}
	return res, nil
}

// attnLayer runs one bidirectional-attention vision layer on the device.
func (e *visionEncoder) attnLayer(l *visionDeviceLayer, n, hidden, heads, kvHeads, headDim int, eps, scale float32, nSeg, maxFrameTokens int) error {
	stream := e.stream
	if err := native.RMSNormBatchedF32(e.normedDev, e.hDev, l.opNorm, eps, hidden, n, stream); err != nil {
		return err
	}
	if err := e.gemmF32(l.q, e.normedDev, hidden, e.qDev, heads*headDim, n); err != nil {
		return err
	}
	if err := e.gemmF32(l.k, e.normedDev, hidden, e.kDev, kvHeads*headDim, n); err != nil {
		return err
	}
	if err := e.gemmF32(l.v, e.normedDev, hidden, e.vDev, kvHeads*headDim, n); err != nil {
		return err
	}
	// Per-head RMSNorm on q [n*heads, headDim] and k [n*kvHeads, headDim].
	if err := native.RMSNormBatchedPairF32(e.qDev, e.qDev, l.qNorm, e.kDev, e.kDev, l.kNorm, eps, headDim, n*heads, n*kvHeads, stream); err != nil {
		return fmt.Errorf("qk norm: %w", err)
	}
	// Axial 2D RoPE.
	if err := native.VisionRoPEF32(e.qDev, e.cosDev, e.sinDev, n, heads, headDim, stream); err != nil {
		return fmt.Errorf("rope q: %w", err)
	}
	if err := native.VisionRoPEF32(e.kDev, e.cosDev, e.sinDev, n, kvHeads, headDim, stream); err != nil {
		return fmt.Errorf("rope k: %w", err)
	}
	// Segment-wise bidirectional attention (tiled, GQA).
	if err := native.VisionAttnSegmentsF32(e.qDev, e.kDev, e.vDev, e.attnDev, e.segStartDev, e.segLenDev, n, nSeg, heads, kvHeads, headDim, scale, stream); err != nil {
		return fmt.Errorf("attention: %w", err)
	}
	// Output projection + residual.
	if err := e.gemmF32(l.o, e.attnDev, heads*headDim, e.resDev, hidden, n); err != nil {
		return err
	}
	return native.AddVectorsF32(e.hDev, e.resDev, n*hidden, stream)
}

// convLayer runs one gated depthwise-2D-conv vision layer on the device.
func (e *visionEncoder) convLayer(l *visionDeviceLayer, n, hidden int, eps float32, nSeg, maxFrameTokens, kSize, merge int) error {
	stream := e.stream
	if err := native.RMSNormBatchedF32(e.normedDev, e.hDev, l.opNorm, eps, hidden, n, stream); err != nil {
		return err
	}
	if err := e.gemmF32(l.convIn, e.normedDev, hidden, e.in3Dev, 3*hidden, n); err != nil {
		return err
	}
	if err := native.VisionConv2dGatedF32(e.in3Dev, l.convKernel, e.convTokDev, e.frameStartDev, e.frameHDev, e.frameWDev, nSeg, n, maxFrameTokens, hidden, kSize, merge, stream); err != nil {
		return fmt.Errorf("conv2d: %w", err)
	}
	if err := e.gemmF32(l.convOut, e.convTokDev, hidden, e.resDev, hidden, n); err != nil {
		return err
	}
	return native.AddVectorsF32(e.hDev, e.resDev, n*hidden, stream)
}

// gemmF32 computes dst[n, rows] = W[rows, k] @ x[n, k]^T (x row-major [n, k],
// dst row-major [n, rows]) with F32 weights/activations and F32 accumulation.
func (e *visionEncoder) gemmF32(w, x native.DeviceBuffer, k int, dst native.DeviceBuffer, rows, n int) error {
	return e.gemmF32B(w, x, k, k, dst, rows, n)
}

// gemmF32B is gemmF32 with an explicit B leading dimension (used by the
// projector merge, where B = the [n, hidden] buffer viewed as
// [merged, group*hidden]).
func (e *visionEncoder) gemmF32B(w, x native.DeviceBuffer, k, ldb int, dst native.DeviceBuffer, rows, n int) error {
	return native.GemmEx(e.blas, native.BlasOpT, native.BlasOpN, rows, n, k, 1.0,
		w, native.BlasF32, k, x, native.BlasF32, ldb, 0.0,
		dst, native.BlasF32, rows, native.BlasComputeF32, native.BlasGemmDefault)
}

// buildRoPETables builds the per-patch axial 2D RoPE cos/sin tables in float64
// exactly like the host tower (simd.VisionTower.Encode).
func (e *visionEncoder) buildRoPETables(grids [][3]int, n int) (cosT, sinT []float32) {
	cfg := e.cfg
	headDim := cfg.HeadDim
	cosT = make([]float32, n*headDim)
	sinT = make([]float32, n*headDim)
	m := cfg.MergeSize
	pos := 0
	for _, g := range grids {
		t, gh, gw := g[0], g[1], g[2]
		hpos := make([]int, gh*gw)
		wpos := make([]int, gh*gw)
		idx := 0
		for bh := 0; bh < gh/m; bh++ {
			for bw := 0; bw < gw/m; bw++ {
				for mh := range m {
					for mw := range m {
						hpos[idx] = bh*m + mh
						wpos[idx] = bw*m + mw
						idx++
					}
				}
			}
		}
		for range t {
			for i := 0; i < gh*gw; i++ {
				hw := make([]float64, headDim/2)
				for j := range e.w.invFreq {
					hw[j] = float64(hpos[i]) * e.w.invFreq[j]
					hw[len(e.w.invFreq)+j] = float64(wpos[i]) * e.w.invFreq[j]
				}
				for d := 0; d < headDim/2; d++ {
					c := float32(math.Cos(hw[d]))
					s := float32(math.Sin(hw[d]))
					cosT[pos*headDim+d] = c
					sinT[pos*headDim+d] = s
					cosT[pos*headDim+headDim/2+d] = c
					sinT[pos*headDim+headDim/2+d] = s
				}
				pos++
			}
		}
	}
	return cosT, sinT
}

// ensureVisionBuf grows buf to at least bytes, preserving nothing (scratch).
func ensureVisionBuf(b *native.DeviceBuffer, bytes int64) error {
	if b.Nbytes() >= bytes && b.Ptr() != nil {
		return nil
	}
	if b.Ptr() != nil {
		if err := b.Free(); err != nil {
			return err
		}
	}
	nb, err := native.AllocDevice(bytes)
	if err != nil {
		return err
	}
	*b = nb
	return nil
}

// close frees all device-resident vision weights and scratch buffers.
func (e *visionEncoder) close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	free := func(b *native.DeviceBuffer) {
		if b.Ptr() != nil {
			_ = b.Free()
			*b = native.DeviceBuffer{}
		}
	}
	free(&e.w.patchifier)
	free(&e.w.embedNorm)
	for i := range e.w.layers {
		l := &e.w.layers[i]
		free(&l.q)
		free(&l.k)
		free(&l.v)
		free(&l.o)
		free(&l.qNorm)
		free(&l.kNorm)
		free(&l.convIn)
		free(&l.convOut)
		free(&l.convKernel)
		free(&l.opNorm)
		free(&l.ffnNorm)
		free(&l.w1)
		free(&l.w2)
		free(&l.w3)
	}
	free(&e.w.projNormW)
	free(&e.w.projNormB)
	free(&e.w.proj1)
	free(&e.w.proj2)
	free(&e.w.proj1Bias)
	free(&e.w.proj2Bias)
	free(&e.pixDev)
	free(&e.hDev)
	free(&e.normedDev)
	free(&e.qDev)
	free(&e.kDev)
	free(&e.vDev)
	free(&e.attnDev)
	free(&e.resDev)
	free(&e.in3Dev)
	free(&e.convTokDev)
	free(&e.ffnInDev)
	free(&e.upDev)
	free(&e.gateDev)
	free(&e.downDev)
	free(&e.tmp1Dev)
	free(&e.outDev)
	free(&e.cosDev)
	free(&e.sinDev)
	free(&e.segStartDev)
	free(&e.segLenDev)
	free(&e.frameStartDev)
	free(&e.frameHDev)
	free(&e.frameWDev)
}
