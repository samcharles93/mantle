//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
	"unsafe"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type Ops struct {
	stream native.Stream
	blas   native.BlasHandle

	mu          sync.Mutex
	weights     map[*simd.Mat]deviceMat
	qweights    map[*simd.Mat]deviceQuantMat
	normWeights map[uintptr]native.DeviceBuffer
	attnCaches  map[*simd.Layer]deviceAttnCache

	xHost native.HostBuffer
	yHost native.HostBuffer
	xDev  native.DeviceBuffer
	yDev  native.DeviceBuffer
	zDev  native.DeviceBuffer
	aDev  native.DeviceBuffer

	xCapBytes int
	yCapBytes int
	xDevBytes int
	yDevBytes int
	zDevBytes int
	aDevBytes int

	kTmpU16 []uint16
	vTmpU16 []uint16
}

type deviceMat struct {
	buf   native.DeviceBuffer
	dtype native.BlasDataType
	rows  int
	cols  int
}

type deviceQuantMat struct {
	format       quantMatFormat
	q            native.DeviceBuffer
	scales       native.DeviceBuffer
	superScales  native.DeviceBuffer
	subScales    native.DeviceBuffer
	rows         int
	cols         int
	blocksPerRow int
}

type deviceAttnCache struct {
	k        native.DeviceBuffer
	v        native.DeviceBuffer
	kvStride int
	cacheLen int
}

type quantMatFormat int

const (
	quantMatFormatCachedInt8 quantMatFormat = iota
	quantMatFormatQ4Raw
	quantMatFormatK4Raw
)

func NewOps(stream native.Stream, blas native.BlasHandle) *Ops {
	return &Ops{
		stream:      stream,
		blas:        blas,
		weights:     make(map[*simd.Mat]deviceMat),
		qweights:    make(map[*simd.Mat]deviceQuantMat),
		normWeights: make(map[uintptr]native.DeviceBuffer),
		attnCaches:  make(map[*simd.Layer]deviceAttnCache),
	}
}

// FusedRMSNormMatVec performs fused RMSNorm + MatVec in a single kernel
// This eliminates one memory roundtrip compared to calling them separately
func (o *Ops) FusedRMSNormMatVec(out []float32, w *simd.Mat, x, normWeight []float32, eps float32) bool {
	if os.Getenv("MANTLE_CUDA_FUSE") == "" {
		return false // Fusion not enabled
	}

	native.RecordRMSNorm()
	native.RecordMatVec()

	n := len(x)
	if len(out) < w.R || len(x) != w.C || len(normWeight) < n {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	rows := w.R
	cols := w.C
	bytes := int64(rows * 4)

	// Ensure buffers
	if err := o.ensureHostVecs(int(bytes), int(bytes)); err != nil {
		return false
	}
	if err := o.ensureDeviceVecs(int(bytes), int(bytes)); err != nil {
		return false
	}

	// Get weight matrix
	wDev, err := o.deviceMat(w)
	if err != nil {
		return false
	}

	// Upload input vector
	xBytes := int64(cols * 4)
	if int(xBytes) > o.xDevBytes {
		if err := o.ensureDeviceVecs(int(xBytes), int(bytes)); err != nil {
			return false
		}
	}
	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), cols), x[:cols])
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
		return false
	}

	// Get norm weight
	normDev, err := o.deviceNormWeight(normWeight)
	if err != nil {
		return false
	}

	// Call fused kernel
	var kernelErr error
	if wDev.dtype == native.BlasBF16 {
		kernelErr = native.FusedRMSNormMatVecBF16(o.yDev, wDev.buf, o.xDev, normDev, eps, rows, cols, o.stream)
	} else {
		kernelErr = native.FusedRMSNormMatVecF32(o.yDev, wDev.buf, o.xDev, normDev, eps, rows, cols, o.stream)
	}

	if kernelErr != nil {
		return false
	}

	// Download result
	if err := o.waitForResult(o.yHost.Ptr(), o.yDev, bytes); err != nil {
		return false
	}

	copy(out, unsafe.Slice((*float32)(o.yHost.Ptr()), rows))
	return true
}

// PreloadModelWeights uploads all model matrices and RMSNorm weights to device memory.
// This removes lazy first-use uploads from the decode hot path.
func (o *Ops) PreloadModelWeights(m *simd.Instance) error {
	if o == nil || m == nil {
		return nil
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	uploadMat := func(name string, mat *simd.Mat) error {
		if mat == nil {
			return nil
		}
		if useQuantKernel() && mat.Quant != nil && mat.Quant.ValidFor(mat) {
			if _, err := o.ensureQuantMat(mat); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			return nil
		}
		if _, err := o.deviceMat(mat); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	}
	uploadNorm := func(name string, weight []float32) error {
		if len(weight) == 0 {
			return nil
		}
		if _, err := o.deviceNormWeight(weight); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	}

	if err := uploadMat("embeddings", m.Embeddings); err != nil {
		return err
	}
	if err := uploadMat("output", m.Output); err != nil {
		return err
	}
	if err := uploadNorm("output_norm", m.OutputNorm); err != nil {
		return err
	}

	for i := range m.Layers {
		layer := &m.Layers[i]
		prefix := fmt.Sprintf("layer[%d]", i)

		if err := uploadNorm(prefix+".attn_norm", layer.AttnNorm); err != nil {
			return err
		}
		if err := uploadNorm(prefix+".post_attn_norm", layer.PostAttnNorm); err != nil {
			return err
		}
		if err := uploadNorm(prefix+".ffn_norm", layer.FfnNorm); err != nil {
			return err
		}
		if err := uploadNorm(prefix+".post_ffn_norm", layer.PostFfnNorm); err != nil {
			return err
		}
		if err := uploadNorm(prefix+".attn_q_norm", layer.AttnQNorm); err != nil {
			return err
		}
		if err := uploadNorm(prefix+".attn_k_norm", layer.AttnKNorm); err != nil {
			return err
		}

		if err := uploadMat(prefix+".wq", layer.Wq); err != nil {
			return err
		}
		if err := uploadMat(prefix+".wk", layer.Wk); err != nil {
			return err
		}
		if err := uploadMat(prefix+".wv", layer.Wv); err != nil {
			return err
		}
		if err := uploadMat(prefix+".wo", layer.Wo); err != nil {
			return err
		}
		if err := uploadMat(prefix+".attn_gate", layer.AttnGate); err != nil {
			return err
		}
		if err := uploadMat(prefix+".shortconv_kernel", layer.ShortConvKernel); err != nil {
			return err
		}
		if err := uploadMat(prefix+".shortconv_in", layer.ShortConvInProj); err != nil {
			return err
		}
		if err := uploadMat(prefix+".shortconv_out", layer.ShortConvOutProj); err != nil {
			return err
		}
		if err := uploadMat(prefix+".ffn_up", layer.FfnUp); err != nil {
			return err
		}
		if err := uploadMat(prefix+".ffn_gate", layer.FfnGate); err != nil {
			return err
		}
		if err := uploadMat(prefix+".ffn_down", layer.FfnDown); err != nil {
			return err
		}
		if layer.MoE != nil {
			if err := uploadMat(prefix+".moe.router", layer.MoE.Router); err != nil {
				return err
			}
			if err := uploadMat(prefix+".moe.shared.up", layer.MoE.Shared.Up); err != nil {
				return err
			}
			if err := uploadMat(prefix+".moe.shared.gate", layer.MoE.Shared.Gate); err != nil {
				return err
			}
			if err := uploadMat(prefix+".moe.shared.down", layer.MoE.Shared.Down); err != nil {
				return err
			}
			for j := range layer.MoE.Experts {
				ex := &layer.MoE.Experts[j]
				exPrefix := fmt.Sprintf("%s.moe.expert[%d]", prefix, j)
				if err := uploadMat(exPrefix+".up", ex.Up); err != nil {
					return err
				}
				if err := uploadMat(exPrefix+".gate", ex.Gate); err != nil {
					return err
				}
				if err := uploadMat(exPrefix+".down", ex.Down); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (o *Ops) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	var err error
	for _, buf := range o.weights {
		if e := buf.buf.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.weights = make(map[*simd.Mat]deviceMat)
	for _, q := range o.qweights {
		if e := q.q.Free(); e != nil && err == nil {
			err = e
		}
		if e := q.scales.Free(); e != nil && err == nil {
			err = e
		}
		if e := q.superScales.Free(); e != nil && err == nil {
			err = e
		}
		if e := q.subScales.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.qweights = make(map[*simd.Mat]deviceQuantMat)

	for _, buf := range o.normWeights {
		if e := buf.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.normWeights = make(map[uintptr]native.DeviceBuffer)
	for _, cache := range o.attnCaches {
		if e := cache.k.Free(); e != nil && err == nil {
			err = e
		}
		if e := cache.v.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.attnCaches = make(map[*simd.Layer]deviceAttnCache)

	if e := o.xDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.yDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.zDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.aDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.xHost.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.yHost.Free(); e != nil && err == nil {
		err = e
	}

	o.xDev = native.DeviceBuffer{}
	o.yDev = native.DeviceBuffer{}
	o.zDev = native.DeviceBuffer{}
	o.aDev = native.DeviceBuffer{}
	o.xHost = native.HostBuffer{}
	o.yHost = native.HostBuffer{}
	o.xCapBytes = 0
	o.yCapBytes = 0
	o.xDevBytes = 0
	o.yDevBytes = 0
	o.zDevBytes = 0
	o.aDevBytes = 0

	return err
}

func (o *Ops) MatVec(dst []float32, w *simd.Mat, x []float32) {
	native.RecordMatVec()
	if w == nil || w.R == 0 || w.C == 0 {
		return
	}
	if len(dst) < w.R || len(x) < w.C {
		panic("matvec shape mismatch")
	}
	if o == nil {
		panic("cuda ops is nil")
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if useQuantKernel() && w.Quant != nil && w.Quant.ValidFor(w) {
		o.matVecQuant(dst, w, x)
		return
	}

	devW, err := o.deviceMat(w)
	if err != nil {
		panic(fmt.Errorf("cuda matvec weight upload failed (r=%d c=%d dtype=%s): %w", w.R, w.C, dtypeString(w.DType), err))
	}

	xBytes := xBufferBytes(devW.dtype, w.C)
	yBytes := int64(w.R) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureHostVecs(int(xBytes), int(yBytes)); err != nil {
		panic(fmt.Errorf("cuda matvec host buffer alloc failed (x=%d y=%d bytes): %w", xBytes, yBytes, err))
	}
	if err := o.ensureDeviceVecs(int(xBytes), int(yBytes)); err != nil {
		panic(fmt.Errorf("cuda matvec device buffer alloc failed (x=%d y=%d bytes): %w", xBytes, yBytes, err))
	}

	if err := fillXBuffer(o.xHost, devW.dtype, x[:w.C]); err != nil {
		panic(err)
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
		panic(fmt.Errorf("cuda matvec H2D copy failed (%d bytes): %w", xBytes, err))
	}

	if err := native.GemmEx(
		o.blas,
		native.BlasOpT,
		native.BlasOpN,
		w.R,
		1,
		w.C,
		1.0,
		devW.buf,
		devW.dtype,
		w.C,
		o.xDev,
		devW.dtype,
		w.C,
		0.0,
		o.yDev,
		native.BlasF32,
		w.R,
		native.BlasComputeF32,
		native.BlasGemmDefault,
	); err != nil {
		panic(fmt.Errorf("cuda matvec gemm failed (m=%d n=1 k=%d): %w", w.R, w.C, err))
	}

	if err := o.waitForResult(o.yHost.Ptr(), o.yDev, yBytes); err != nil {
		panic(fmt.Errorf("cuda matvec result wait failed: %w", err))
	}

	copy(dst[:w.R], unsafe.Slice((*float32)(o.yHost.Ptr()), w.R))
	runtime.KeepAlive(x)
	runtime.KeepAlive(dst)
}

func (o *Ops) matVecQuant(dst []float32, w *simd.Mat, x []float32) {
	qw, err := o.ensureQuantMat(w)
	if err != nil {
		panic(fmt.Errorf("cuda quant matvec weight upload failed (r=%d c=%d dtype=%s): %w", w.R, w.C, dtypeString(w.DType), err))
	}

	xBytes := int64(w.C) * int64(unsafe.Sizeof(float32(0)))
	yBytes := int64(w.R) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureHostVecs(int(xBytes), int(yBytes)); err != nil {
		panic(fmt.Errorf("cuda quant matvec host buffer alloc failed (x=%d y=%d bytes): %w", xBytes, yBytes, err))
	}
	if err := o.ensureDeviceVecs(int(xBytes), int(yBytes)); err != nil {
		panic(fmt.Errorf("cuda quant matvec device buffer alloc failed (x=%d y=%d bytes): %w", xBytes, yBytes, err))
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), w.C), x[:w.C])
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
		panic(fmt.Errorf("cuda quant matvec H2D copy failed (%d bytes): %w", xBytes, err))
	}
	var kernelErr error
	switch qw.format {
	case quantMatFormatQ4Raw:
		kernelErr = native.QuantMatVecQ4F32(qw.q, qw.scales, o.xDev, o.yDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
	case quantMatFormatK4Raw:
		kernelErr = native.QuantMatVecK4F32(qw.q, qw.superScales, qw.subScales, o.xDev, o.yDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
	default:
		kernelErr = native.QuantMatVecInt8BlocksF32(qw.q, qw.scales, o.xDev, o.yDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
	}
	if kernelErr != nil {
		panic(fmt.Errorf("cuda quant matvec kernel failed (rows=%d cols=%d blocks=%d): %w", qw.rows, qw.cols, qw.blocksPerRow, kernelErr))
	}
	if err := o.waitForResult(o.yHost.Ptr(), o.yDev, yBytes); err != nil {
		panic(fmt.Errorf("cuda quant matvec result wait failed: %w", err))
	}

	copy(dst[:w.R], unsafe.Slice((*float32)(o.yHost.Ptr()), w.R))
	runtime.KeepAlive(x)
	runtime.KeepAlive(dst)
}

func (o *Ops) MatVecWithQuant(dst []float32, w *simd.Mat, x []float32, _ *simd.QuantVec) {
	o.MatVec(dst, w, x)
}

func (o *Ops) FFNBlock(layer *simd.Layer, x []float32, out []float32) bool {
	if !useFFNFastPath() {
		return false
	}
	if o == nil || layer == nil || layer.FfnUp == nil || layer.FfnGate == nil || layer.FfnDown == nil {
		return false
	}
	if len(x) < layer.FfnUp.C || len(out) < layer.FfnDown.R {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	upW, err := o.deviceMat(layer.FfnUp)
	if err != nil {
		return false
	}
	gateW, err := o.deviceMat(layer.FfnGate)
	if err != nil {
		return false
	}
	downW, err := o.deviceMat(layer.FfnDown)
	if err != nil {
		return false
	}
	// CUDA FFN fast path currently supports f16/bf16 weights with matching input/output packing.
	if upW.dtype != gateW.dtype || upW.dtype != downW.dtype {
		return false
	}

	interm := upW.rows
	if interm != gateW.rows || upW.cols != len(x) || downW.cols != interm || downW.rows > len(out) {
		return false
	}

	xBytes := xBufferBytes(upW.dtype, len(x))
	intermF32Bytes := int64(interm) * 4
	intermF16Bytes := int64(interm) * 2
	outBytes := int64(downW.rows) * 4

	if err := o.ensureHostVecs(int(xBytes), int(outBytes)); err != nil {
		return false
	}
	if err := o.ensureDeviceVecs(int(xBytes), int(max64Local(intermF32Bytes, outBytes))); err != nil {
		return false
	}
	if err := o.ensureNormTmp(int(max64Local(intermF32Bytes, outBytes))); err != nil {
		return false
	}
	if err := o.ensureActTmp(int(intermF16Bytes)); err != nil {
		return false
	}

	if err := fillXBuffer(o.xHost, upW.dtype, x[:len(x)]); err != nil {
		return false
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
		return false
	}

	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, upW.rows, 1, upW.cols, 1.0, upW.buf, upW.dtype, upW.cols, o.xDev, upW.dtype, upW.cols, 0.0, o.yDev, native.BlasF32, upW.rows, native.BlasComputeF32, native.BlasGemmDefault); err != nil {
		return false
	}
	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, gateW.rows, 1, gateW.cols, 1.0, gateW.buf, gateW.dtype, gateW.cols, o.xDev, gateW.dtype, gateW.cols, 0.0, o.zDev, native.BlasF32, gateW.rows, native.BlasComputeF32, native.BlasGemmDefault); err != nil {
		return false
	}

	if err := native.SiluMulF32(o.zDev, o.yDev, o.yDev, interm, o.stream); err != nil {
		return false
	}

	downInput := o.aDev
	downInputType := downW.dtype
	if downW.dtype == native.BlasBF16 || downW.dtype == native.BlasF32 {
		downInput = o.yDev
		downInputType = native.BlasF32
	} else {
		if err := native.ConvertF32ToF16(o.yDev, o.aDev, interm, o.stream); err != nil {
			return false
		}
	}

	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, downW.rows, 1, downW.cols, 1.0, downW.buf, downW.dtype, downW.cols, downInput, downInputType, downW.cols, 0.0, o.zDev, native.BlasF32, downW.rows, native.BlasComputeF32, native.BlasGemmDefault); err != nil {
		return false
	}
	if err := o.waitForResult(o.yHost.Ptr(), o.zDev, outBytes); err != nil {
		return false
	}
	copy(out[:downW.rows], unsafe.Slice((*float32)(o.yHost.Ptr()), downW.rows))
	return true
}

func (o *Ops) MatVecQKV(q, k, v []float32, wq, wk, wv *simd.Mat, x []float32) bool {
	if !useQKVFastPath() {
		return false
	}
	if o == nil || wq == nil || wk == nil || wv == nil {
		return false
	}
	if len(q) < wq.R || len(k) < wk.R || len(v) < wv.R || len(x) < wq.C || wq.C != wk.C || wq.C != wv.C {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	dq, err := o.deviceMat(wq)
	if err != nil {
		return false
	}
	dk, err := o.deviceMat(wk)
	if err != nil {
		return false
	}
	dv, err := o.deviceMat(wv)
	if err != nil {
		return false
	}
	if dq.dtype != dk.dtype || dq.dtype != dv.dtype {
		return false
	}

	xBytes := xBufferBytes(dq.dtype, wq.C)
	qBytes := int64(dq.rows) * 4
	kBytes := int64(dk.rows) * 4
	vBytes := int64(dv.rows) * 4
	totalOutBytes := qBytes + kBytes + vBytes
	if err := o.ensureHostVecs(int(xBytes), int(totalOutBytes)); err != nil {
		return false
	}
	maxOutBytes := int(max64Local(qBytes, max64Local(kBytes, vBytes)))
	if err := o.ensureDeviceVecs(int(xBytes), maxOutBytes); err != nil {
		return false
	}
	if err := o.ensureNormTmp(int(kBytes)); err != nil {
		return false
	}
	if err := o.ensureActTmp(int(vBytes)); err != nil {
		return false
	}

	if err := fillXBuffer(o.xHost, dq.dtype, x[:wq.C]); err != nil {
		return false
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
		return false
	}

	runOne := func(out native.DeviceBuffer, w deviceMat) bool {
		return native.GemmEx(
			o.blas,
			native.BlasOpT,
			native.BlasOpN,
			w.rows,
			1,
			w.cols,
			1.0,
			w.buf,
			w.dtype,
			w.cols,
			o.xDev,
			w.dtype,
			w.cols,
			0.0,
			out,
			native.BlasF32,
			w.rows,
			native.BlasComputeF32,
			native.BlasGemmDefault,
		) == nil
	}

	if !runOne(o.yDev, dq) || !runOne(o.zDev, dk) || !runOne(o.aDev, dv) {
		return false
	}
	if err := native.MemcpyD2HAsync(o.yHost.Ptr(), o.yDev, qBytes, o.stream); err != nil {
		return false
	}
	if err := native.MemcpyD2HAsync(unsafe.Add(o.yHost.Ptr(), int(qBytes)), o.zDev, kBytes, o.stream); err != nil {
		return false
	}
	if err := native.MemcpyD2HAsync(unsafe.Add(o.yHost.Ptr(), int(qBytes+kBytes)), o.aDev, vBytes, o.stream); err != nil {
		return false
	}
	if err := o.stream.Synchronize(); err != nil {
		return false
	}

	copy(q[:dq.rows], unsafe.Slice((*float32)(o.yHost.Ptr()), dq.rows))
	copy(k[:dk.rows], unsafe.Slice((*float32)(unsafe.Add(o.yHost.Ptr(), int(qBytes))), dk.rows))
	copy(v[:dv.rows], unsafe.Slice((*float32)(unsafe.Add(o.yHost.Ptr(), int(qBytes+kBytes))), dv.rows))
	return true
}

func (o *Ops) AttentionInner(attnOut []float32, layer *simd.Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale float32) bool {
	if !useAttentionInnerFastPath() {
		return false
	}
	if o == nil || layer == nil {
		return false
	}
	if len(attnOut) < nHead*headDim || len(q) < nHead*headDim || len(k) < kvStride || len(v) < kvStride {
		return false
	}
	if pos < 0 || start < 0 || start > pos || kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 {
		return false
	}
	cacheLen := layer.AttnCache.CacheLen
	if cacheLen <= 0 {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	cache, err := o.ensureAttnCache(layer, kvStride, cacheLen)
	if err != nil {
		return false
	}
	if cache.kvStride != kvStride || cache.cacheLen != cacheLen {
		return false
	}

	qBytes := int64(nHead*headDim) * 4
	outBytes := qBytes
	if err := o.ensureHostVecs(int(qBytes), int(outBytes)); err != nil {
		return false
	}
	if err := o.ensureDeviceVecs(int(qBytes), int(outBytes)); err != nil {
		return false
	}
	if err := o.ensureKVU16Tmp(kvStride); err != nil {
		return false
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), nHead*headDim), q[:nHead*headDim])
	for i := range kvStride {
		o.kTmpU16[i] = simd.Float32ToFloat16(k[i])
		o.vTmpU16[i] = simd.Float32ToFloat16(v[i])
	}

	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	kvBytes := int64(kvStride) * 2
	dstOffset := int64(cachePos*kvStride) * 2
	if err := native.MemcpyH2DAsyncAt(cache.k, dstOffset, unsafe.Pointer(&o.kTmpU16[0]), kvBytes, o.stream); err != nil {
		return false
	}
	if err := native.MemcpyH2DAsyncAt(cache.v, dstOffset, unsafe.Pointer(&o.vTmpU16[0]), kvBytes, o.stream); err != nil {
		return false
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), qBytes, o.stream); err != nil {
		return false
	}

	if err := native.AttentionInnerF16CacheF32(o.xDev, cache.k, cache.v, o.yDev, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale, o.stream); err != nil {
		return false
	}
	if err := o.waitForResult(o.yHost.Ptr(), o.yDev, outBytes); err != nil {
		return false
	}
	copy(attnOut[:nHead*headDim], unsafe.Slice((*float32)(o.yHost.Ptr()), nHead*headDim))
	return true
}

func (o *Ops) AttentionInnerProjection(projOut []float32, layer *simd.Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, epsilon float32) bool {
	if !useAttentionInnerFastPath() {
		return false
	}
	if o == nil || layer == nil || layer.Wo == nil {
		return false
	}
	if len(projOut) < layer.Wo.R || len(q) < nHead*headDim || len(k) < kvStride || len(v) < kvStride {
		return false
	}
	if pos < 0 || start < 0 || start > pos || kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 {
		return false
	}
	cacheLen := layer.AttnCache.CacheLen
	if cacheLen <= 0 {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	cache, err := o.ensureAttnCache(layer, kvStride, cacheLen)
	if err != nil {
		return false
	}
	if cache.kvStride != kvStride || cache.cacheLen != cacheLen {
		return false
	}

	qBytes := int64(nHead*headDim) * 4
	projDim := layer.Wo.R
	projBytes := int64(projDim) * 4
	outBytes := qBytes
	if projBytes > outBytes {
		outBytes = projBytes
	}
	if err := o.ensureHostVecs(int(qBytes), int(outBytes)); err != nil {
		return false
	}
	if err := o.ensureDeviceVecs(int(qBytes), int(outBytes)); err != nil {
		return false
	}
	if err := o.ensureKVU16Tmp(kvStride); err != nil {
		return false
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), nHead*headDim), q[:nHead*headDim])
	for i := range kvStride {
		o.kTmpU16[i] = simd.Float32ToFloat16(k[i])
		o.vTmpU16[i] = simd.Float32ToFloat16(v[i])
	}

	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	kvBytes := int64(kvStride) * 2
	dstOffset := int64(cachePos*kvStride) * 2
	if err := native.MemcpyH2DAsyncAt(cache.k, dstOffset, unsafe.Pointer(&o.kTmpU16[0]), kvBytes, o.stream); err != nil {
		return false
	}
	if err := native.MemcpyH2DAsyncAt(cache.v, dstOffset, unsafe.Pointer(&o.vTmpU16[0]), kvBytes, o.stream); err != nil {
		return false
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), qBytes, o.stream); err != nil {
		return false
	}

	if err := native.AttentionInnerF16CacheF32(o.xDev, cache.k, cache.v, o.yDev, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale, o.stream); err != nil {
		return false
	}

	// Projection step: Wo * attnOut (attnOut is in o.yDev)
	// projDim and projBytes already defined above
	if err := o.ensureNormTmp(int(projBytes)); err != nil {
		return false
	}
	// Ensure host buffer for final result
	if err := o.ensureHostVecs(int(qBytes), int(projBytes)); err != nil {
		return false
	}

	if useQuantKernel() && layer.Wo.Quant != nil && layer.Wo.Quant.ValidFor(layer.Wo) {
		qw, err := o.ensureQuantMat(layer.Wo)
		if err != nil {
			return false
		}
		var kernelErr error
		switch qw.format {
		case quantMatFormatQ4Raw:
			kernelErr = native.QuantMatVecQ4F32(qw.q, qw.scales, o.yDev, o.zDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
		case quantMatFormatK4Raw:
			kernelErr = native.QuantMatVecK4F32(qw.q, qw.superScales, qw.subScales, o.yDev, o.zDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
		default:
			kernelErr = native.QuantMatVecInt8BlocksF32(qw.q, qw.scales, o.yDev, o.zDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
		}
		if kernelErr != nil {
			return false
		}
	} else {
		devW, err := o.deviceMat(layer.Wo)
		if err != nil {
			return false
		}
		// gemm: Wo^T * attnOut (Wo is column-major? In BLAS gemm with transposition)
		// Wo dimensions: rows = projDim, cols = nHead*headDim
		// Compute y = Wo^T * x where x is attnOut (size cols)
		// Use GemmEx with OpT, OpN
		if err := native.GemmEx(
			o.blas,
			native.BlasOpT,
			native.BlasOpN,
			devW.rows, // output size
			1,
			devW.cols, // input size
			1.0,
			devW.buf,
			devW.dtype,
			devW.cols,
			o.yDev,
			native.BlasF32,
			devW.cols,
			0.0,
			o.zDev,
			native.BlasF32,
			devW.rows,
			native.BlasComputeF32,
			native.BlasGemmDefault,
		); err != nil {
			return false
		}
	}

	// Copy projection result to host
	if len(layer.PostAttnNorm) > 0 {
		// Upload norm weight if not already present
		wDev, err := o.deviceNormWeight(layer.PostAttnNorm)
		if err != nil {
			return false
		}
		// Ensure temporary buffer for norm result
		if err := o.ensureActTmp(int(projBytes)); err != nil {
			return false
		}
		if err := o.rmsNormDevice(o.zDev, wDev, o.aDev, projDim, epsilon); err != nil {
			return false
		}
		// Copy normalized result to host
		if err := o.waitForResult(o.yHost.Ptr(), o.aDev, projBytes); err != nil {
			return false
		}
	} else {
		// Copy projection result directly
		if err := o.waitForResult(o.yHost.Ptr(), o.zDev, projBytes); err != nil {
			return false
		}
	}
	copy(projOut[:projDim], unsafe.Slice((*float32)(o.yHost.Ptr()), projDim))
	return true
}

func (o *Ops) Softmax(x []float32) {
	if len(x) <= 1 {
		return
	}
	if o == nil {
		panic("cuda ops is nil")
	}
	if !useAttnSoftmaxKernel() {
		simd.Softmax(x)
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	bytes := int64(len(x)) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureHostVecs(int(bytes), int(bytes)); err != nil {
		panic(fmt.Errorf("cuda softmax host buffer alloc failed (%d bytes): %w", bytes, err))
	}
	if err := o.ensureDeviceVecs(int(bytes), int(bytes)); err != nil {
		panic(fmt.Errorf("cuda softmax device buffer alloc failed (%d bytes): %w", bytes, err))
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), len(x)), x)
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), bytes, o.stream); err != nil {
		panic(fmt.Errorf("cuda softmax H2D copy failed (%d bytes): %w", bytes, err))
	}
	if err := native.SoftmaxRowsF32(o.xDev, 1, len(x), o.stream); err != nil {
		panic(fmt.Errorf("cuda softmax kernel failed (cols=%d): %w", len(x), err))
	}
	if err := o.waitForResult(o.yHost.Ptr(), o.xDev, bytes); err != nil {
		panic(fmt.Errorf("cuda softmax result wait failed: %w", err))
	}

	copy(x, unsafe.Slice((*float32)(o.yHost.Ptr()), len(x)))
	runtime.KeepAlive(x)
}

func (o *Ops) RMSNorm(dst, src, weight []float32, eps float32) {
	native.RecordRMSNorm()
	n := len(src)
	if n == 0 {
		return
	}
	if len(dst) < n || len(weight) < n {
		panic("rmsnorm shape mismatch")
	}
	if o == nil {
		panic("cuda ops is nil")
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	bytes := int64(n) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureHostVecs(int(bytes), int(bytes)); err != nil {
		panic(fmt.Errorf("cuda rmsnorm host buffer alloc failed (%d bytes): %w", bytes, err))
	}
	if err := o.ensureDeviceVecs(int(bytes), int(bytes)); err != nil {
		panic(fmt.Errorf("cuda rmsnorm device buffer alloc failed (%d bytes): %w", bytes, err))
	}

	wDev, err := o.deviceNormWeight(weight)
	if err != nil {
		panic(fmt.Errorf("cuda rmsnorm weight upload failed (%d elements): %w", n, err))
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), n), src[:n])
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), bytes, o.stream); err != nil {
		panic(fmt.Errorf("cuda rmsnorm H2D copy failed (%d bytes): %w", bytes, err))
	}

	// Single custom kernel: reduction + rsqrt + weight multiply, no CPU sync
	if err := native.RMSNormF32(o.yDev, o.xDev, wDev, eps, n, o.stream); err != nil {
		panic(fmt.Errorf("cuda rmsnorm kernel failed (n=%d): %w", n, err))
	}

	if err := o.waitForResult(o.yHost.Ptr(), o.yDev, bytes); err != nil {
		panic(fmt.Errorf("cuda rmsnorm result wait failed: %w", err))
	}

	copy(dst[:n], unsafe.Slice((*float32)(o.yHost.Ptr()), n))
}

// RMSNormBatched applies RMSNorm to nHeads contiguous vectors of headDim each,
// all sharing the same weight vector. Launched as a single kernel with one block per head.
func (o *Ops) RMSNormBatched(dst, src, weight []float32, eps float32, headDim, nHeads int) bool {
	totalElems := headDim * nHeads
	if totalElems == 0 || len(src) < totalElems || len(dst) < totalElems || len(weight) < headDim {
		return false
	}
	if o == nil {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	bytes := int64(totalElems) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureHostVecs(int(bytes), int(bytes)); err != nil {
		return false
	}
	if err := o.ensureDeviceVecs(int(bytes), int(bytes)); err != nil {
		return false
	}

	wDev, err := o.deviceNormWeight(weight[:headDim])
	if err != nil {
		return false
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), totalElems), src[:totalElems])
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), bytes, o.stream); err != nil {
		return false
	}

	if err := native.RMSNormBatchedF32(o.yDev, o.xDev, wDev, eps, headDim, nHeads, o.stream); err != nil {
		return false
	}

	if err := o.waitForResult(o.yHost.Ptr(), o.yDev, bytes); err != nil {
		return false
	}

	copy(dst[:totalElems], unsafe.Slice((*float32)(o.yHost.Ptr()), totalElems))
	return true
}

// rmsNormDevice computes RMSNorm on device buffers using a single custom kernel.
// Requires o.mu locked, and buffers allocated.
func (o *Ops) rmsNormDevice(srcDev native.DeviceBuffer, weightDev native.DeviceBuffer, dstDev native.DeviceBuffer, n int, eps float32) error {
	return native.RMSNormF32(dstDev, srcDev, weightDev, eps, n, o.stream)
}

func (o *Ops) ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
	if headDim%2 != 0 {
		panic("headDim must be even for RoPE")
	}
	if attentionFactor == 0 {
		attentionFactor = 1
	}
	half := headDim / 2
	for h := range nHead {
		base := h * headDim
		for i := range half {
			angle := float64(pos) * invFreq[i]
			c := float32(math.Cos(angle)) * attentionFactor
			s := float32(math.Sin(angle)) * attentionFactor
			i0 := base + i
			i1 := base + i + half
			x0 := x[i0]
			x1 := x[i1]
			x[i0] = x0*c - x1*s
			x[i1] = x0*s + x1*c
		}
	}
}

func (o *Ops) StoreKV(_ int, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, _, _ []int8, _, _ []float32, k, v []float32) {
	native.RecordStoreKV()
	if kvStride <= 0 {
		return
	}
	offset := pos * kvStride
	if kDst != nil {
		copy(kDst[offset:], k)
	} else if kDst16 != nil {
		for i, kv := range k {
			kDst16[offset+i] = simd.Float32ToFloat16(kv)
		}
	}
	if vDst != nil {
		copy(vDst[offset:], v)
	} else if vDst16 != nil {
		for i, vv := range v {
			vDst16[offset+i] = simd.Float32ToFloat16(vv)
		}
	}
}

func (o *Ops) deviceMat(w *simd.Mat) (deviceMat, error) {
	if buf, ok := o.weights[w]; ok {
		return buf, nil
	}
	if w.Raw != nil && mcf.DTypeRequiresAligned64(w.DType) {
		dev, err := o.uploadQuantAsF16Device(w)
		if err != nil {
			// Fallback: CPU decode path remains as a compatibility safety net.
			hostF16, decErr := decodeQuantToF16(w)
			if decErr != nil {
				return deviceMat{}, err
			}
			bytes := int64(len(hostF16)) * int64(unsafe.Sizeof(uint16(0)))
			dev, decErr = native.AllocDevice(bytes)
			if decErr != nil {
				return deviceMat{}, err
			}
			if decErr := native.MemcpyH2D(dev, unsafe.Pointer(&hostF16[0]), bytes); decErr != nil {
				_ = dev.Free()
				return deviceMat{}, err
			}
		}
		info := deviceMat{
			buf:   dev,
			dtype: native.BlasF16,
			rows:  w.R,
			cols:  w.C,
		}
		o.weights[w] = info
		runtime.KeepAlive(w)
		return info, nil
	}

	dtype, bytes, hostPtr, err := weightUploadSpec(w)
	if err != nil {
		return deviceMat{}, err
	}
	if bytes == 0 || hostPtr == nil {
		return deviceMat{}, fmt.Errorf("empty weight matrix")
	}

	dev, err := native.AllocDevice(bytes)
	if err != nil {
		return deviceMat{}, err
	}
	if err := native.MemcpyH2D(dev, hostPtr, bytes); err != nil {
		_ = dev.Free()
		return deviceMat{}, err
	}
	info := deviceMat{
		buf:   dev,
		dtype: dtype,
		rows:  w.R,
		cols:  w.C,
	}
	o.weights[w] = info
	runtime.KeepAlive(w)
	return info, nil
}

func (o *Ops) uploadQuantAsF16Device(w *simd.Mat) (native.DeviceBuffer, error) {
	if w == nil || w.Raw == nil || !mcf.DTypeRequiresAligned64(w.DType) {
		return native.DeviceBuffer{}, fmt.Errorf("gpu dequant requires quantized raw matrix")
	}
	totalElems, ok := mulInt(w.R, w.C)
	if !ok {
		return native.DeviceBuffer{}, fmt.Errorf("quant matrix too large")
	}
	outBytes, ok := mulInt(totalElems, 2)
	if !ok {
		return native.DeviceBuffer{}, fmt.Errorf("quant matrix too large")
	}
	out, err := native.AllocDevice(int64(outBytes))
	if err != nil {
		return native.DeviceBuffer{}, err
	}

	switch w.DType {
	case mcf.DTypeQ4:
		v, err := parseQ4PayloadView(w)
		if err != nil {
			_ = out.Free()
			return native.DeviceBuffer{}, err
		}
		qBuf, sBuf, err := o.uploadQ4PayloadBuffers(v)
		if err != nil {
			_ = out.Free()
			return native.DeviceBuffer{}, err
		}
		defer qBuf.Free()
		defer sBuf.Free()
		if err := native.DequantizeQ4ToF16(qBuf, sBuf, out, w.R, v.blocksPerRow, w.C, o.stream); err != nil {
			_ = out.Free()
			return native.DeviceBuffer{}, err
		}
	case mcf.DTypeK4:
		v, err := parseK4PayloadView(w)
		if err != nil {
			_ = out.Free()
			return native.DeviceBuffer{}, err
		}
		qBuf, superBuf, subBuf, err := o.uploadK4PayloadBuffers(v)
		if err != nil {
			_ = out.Free()
			return native.DeviceBuffer{}, err
		}
		defer qBuf.Free()
		defer superBuf.Free()
		defer subBuf.Free()
		if err := native.DequantizeK4ToF16(qBuf, superBuf, subBuf, out, w.R, v.blocksPerRow, w.C, o.stream); err != nil {
			_ = out.Free()
			return native.DeviceBuffer{}, err
		}
	default:
		_ = out.Free()
		return native.DeviceBuffer{}, fmt.Errorf("gpu dequant unsupported dtype: %s", dtypeString(w.DType))
	}

	return out, nil
}

func decodeQuantToF16(w *simd.Mat) ([]uint16, error) {
	if w == nil || w.R == 0 || w.C == 0 {
		return nil, fmt.Errorf("empty quantized weight matrix")
	}
	if w.Raw == nil || !mcf.DTypeRequiresAligned64(w.DType) {
		return nil, fmt.Errorf("decodeQuantToF16 requires quantized raw matrix")
	}
	row := make([]float32, w.C)
	out := make([]uint16, w.R*w.C)
	for r := range w.R {
		w.RowTo(row, r)
		base := r * w.C
		for c, v := range row {
			out[base+c] = simd.Float32ToFloat16(v)
		}
	}
	return out, nil
}

func useQuantKernel() bool {
	return envEnabled("MANTLE_CUDA_QUANT_KERNEL")
}

func useLegacyStreamSync() bool {
	return envEnabled("MANTLE_CUDA_LEGACY_SYNC")
}

func useAttnSoftmaxKernel() bool {
	return envEnabled("MANTLE_CUDA_ATTN_SOFTMAX")
}

func useFFNFastPath() bool {
	return !envEnabled("MANTLE_CUDA_DISABLE_FFN_FASTPATH")
}

func useQKVFastPath() bool {
	return !envEnabled("MANTLE_CUDA_DISABLE_QKV_FASTPATH")
}

func useAttentionInnerFastPath() bool {
	return !envEnabled("MANTLE_CUDA_DISABLE_ATTN_INNER_FASTPATH")
}

func envEnabled(name string) bool {
	v, ok := os.LookupEnv(name)
	if !ok {
		return false
	}
	if b, err := strconv.ParseBool(v); err == nil {
		return b
	}
	switch v {
	case "1", "on", "ON", "yes", "YES", "y", "Y":
		return true
	default:
		return false
	}
}

func (o *Ops) waitForResult(hostDst unsafe.Pointer, devSrc native.DeviceBuffer, bytes int64) error {
	if useLegacyStreamSync() {
		if err := native.MemcpyD2HAsync(hostDst, devSrc, bytes, o.stream); err != nil {
			return err
		}
		if err := o.stream.Synchronize(); err != nil {
			return err
		}
		return nil
	}
	// Blocking D2H copy implicitly waits for producing work and avoids a separate stream sync call.
	return native.MemcpyD2H(hostDst, devSrc, bytes)
}

func (o *Ops) ensureQuantMat(w *simd.Mat) (deviceQuantMat, error) {
	if buf, ok := o.qweights[w]; ok {
		return buf, nil
	}
	if w == nil {
		return deviceQuantMat{}, fmt.Errorf("missing quant matrix")
	}
	if w.DType == mcf.DTypeQ4 && len(w.Raw) > 0 {
		info, err := o.ensureQuantMatQ4Raw(w)
		if err != nil {
			return deviceQuantMat{}, err
		}
		o.qweights[w] = info
		runtime.KeepAlive(w)
		return info, nil
	}
	if w.DType == mcf.DTypeK4 && len(w.Raw) > 0 {
		info, err := o.ensureQuantMatK4Raw(w)
		if err != nil {
			return deviceQuantMat{}, err
		}
		o.qweights[w] = info
		runtime.KeepAlive(w)
		return info, nil
	}
	if w.Quant == nil || !w.Quant.ValidFor(w) {
		return deviceQuantMat{}, fmt.Errorf("missing quant cache")
	}
	qc := w.Quant
	if len(qc.Q) == 0 || len(qc.Scales) == 0 {
		return deviceQuantMat{}, fmt.Errorf("empty quant cache")
	}

	qBytes := int64(len(qc.Q))
	scaleBytes := int64(len(qc.Scales)) * int64(unsafe.Sizeof(float32(0)))
	qBuf, err := native.AllocDevice(qBytes)
	if err != nil {
		return deviceQuantMat{}, err
	}
	if err := native.MemcpyH2D(qBuf, unsafe.Pointer(&qc.Q[0]), qBytes); err != nil {
		_ = qBuf.Free()
		return deviceQuantMat{}, err
	}
	sBuf, err := native.AllocDevice(scaleBytes)
	if err != nil {
		_ = qBuf.Free()
		return deviceQuantMat{}, err
	}
	if err := native.MemcpyH2D(sBuf, unsafe.Pointer(&qc.Scales[0]), scaleBytes); err != nil {
		_ = qBuf.Free()
		_ = sBuf.Free()
		return deviceQuantMat{}, err
	}

	info := deviceQuantMat{
		format:       quantMatFormatCachedInt8,
		q:            qBuf,
		scales:       sBuf,
		rows:         w.R,
		cols:         w.C,
		blocksPerRow: qc.BlocksPerRow,
	}
	o.qweights[w] = info
	runtime.KeepAlive(w)
	return info, nil
}

type q4PayloadView struct {
	blocksPerRow int
	data         []byte
	scales       []byte
}

type k4PayloadView struct {
	blocksPerRow int
	data         []byte
	superScales  []byte
	subScales    []byte
}

func parseQ4PayloadView(w *simd.Mat) (q4PayloadView, error) {
	blocksPerRow, totalBlocks, err := quantBlocks(w.R, w.C)
	if err != nil {
		return q4PayloadView{}, err
	}
	scaleBytes, ok := mulInt(totalBlocks, 2)
	if !ok {
		return q4PayloadView{}, fmt.Errorf("q4 scale region too large")
	}
	dataOff, ok := align64Int(scaleBytes)
	if !ok {
		return q4PayloadView{}, fmt.Errorf("q4 payload layout overflow")
	}
	dataBytes, ok := mulInt(totalBlocks, 16)
	if !ok {
		return q4PayloadView{}, fmt.Errorf("q4 data region too large")
	}
	if dataOff < 0 || dataOff+dataBytes > len(w.Raw) {
		return q4PayloadView{}, fmt.Errorf("q4 payload bounds invalid")
	}
	return q4PayloadView{
		blocksPerRow: blocksPerRow,
		data:         w.Raw[dataOff : dataOff+dataBytes],
		scales:       w.Raw[:scaleBytes],
	}, nil
}

func parseK4PayloadView(w *simd.Mat) (k4PayloadView, error) {
	blocksPerRow, totalBlocks, err := quantBlocks(w.R, w.C)
	if err != nil {
		return k4PayloadView{}, err
	}
	superBlocksPerRow := (blocksPerRow + 7) / 8
	superCount, ok := mulInt(w.R, superBlocksPerRow)
	if !ok {
		return k4PayloadView{}, fmt.Errorf("k4 super scale region too large")
	}
	superBytes, ok := mulInt(superCount, 2)
	if !ok {
		return k4PayloadView{}, fmt.Errorf("k4 super scale region too large")
	}
	subOff, ok := align64Int(superBytes)
	if !ok {
		return k4PayloadView{}, fmt.Errorf("k4 payload layout overflow")
	}
	subBytes := totalBlocks
	dataOff, ok := addInt(subOff, subBytes)
	if !ok {
		return k4PayloadView{}, fmt.Errorf("k4 payload layout overflow")
	}
	dataOff, ok = align64Int(dataOff)
	if !ok {
		return k4PayloadView{}, fmt.Errorf("k4 payload layout overflow")
	}
	dataBytes, ok := mulInt(totalBlocks, 16)
	if !ok {
		return k4PayloadView{}, fmt.Errorf("k4 data region too large")
	}
	if superBytes < 0 || subOff < 0 || dataOff < 0 || dataOff+dataBytes > len(w.Raw) {
		return k4PayloadView{}, fmt.Errorf("k4 payload bounds invalid")
	}
	if subOff+subBytes > len(w.Raw) {
		return k4PayloadView{}, fmt.Errorf("k4 subscale region out of bounds")
	}
	return k4PayloadView{
		blocksPerRow: blocksPerRow,
		data:         w.Raw[dataOff : dataOff+dataBytes],
		superScales:  w.Raw[:superBytes],
		subScales:    w.Raw[subOff : subOff+subBytes],
	}, nil
}

func (o *Ops) uploadQ4PayloadBuffers(v q4PayloadView) (native.DeviceBuffer, native.DeviceBuffer, error) {
	qBuf, err := native.AllocDevice(int64(len(v.data)))
	if err != nil {
		return native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(qBuf, unsafe.Pointer(&v.data[0]), int64(len(v.data))); err != nil {
		_ = qBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	sBuf, err := native.AllocDevice(int64(len(v.scales)))
	if err != nil {
		_ = qBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(sBuf, unsafe.Pointer(&v.scales[0]), int64(len(v.scales))); err != nil {
		_ = qBuf.Free()
		_ = sBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	return qBuf, sBuf, nil
}

func (o *Ops) uploadK4PayloadBuffers(v k4PayloadView) (native.DeviceBuffer, native.DeviceBuffer, native.DeviceBuffer, error) {
	qBuf, err := native.AllocDevice(int64(len(v.data)))
	if err != nil {
		return native.DeviceBuffer{}, native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(qBuf, unsafe.Pointer(&v.data[0]), int64(len(v.data))); err != nil {
		_ = qBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	superBuf, err := native.AllocDevice(int64(len(v.superScales)))
	if err != nil {
		_ = qBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(superBuf, unsafe.Pointer(&v.superScales[0]), int64(len(v.superScales))); err != nil {
		_ = qBuf.Free()
		_ = superBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	subBuf, err := native.AllocDevice(int64(len(v.subScales)))
	if err != nil {
		_ = qBuf.Free()
		_ = superBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(subBuf, unsafe.Pointer(&v.subScales[0]), int64(len(v.subScales))); err != nil {
		_ = qBuf.Free()
		_ = superBuf.Free()
		_ = subBuf.Free()
		return native.DeviceBuffer{}, native.DeviceBuffer{}, native.DeviceBuffer{}, err
	}
	return qBuf, superBuf, subBuf, nil
}

func (o *Ops) ensureQuantMatQ4Raw(w *simd.Mat) (deviceQuantMat, error) {
	v, err := parseQ4PayloadView(w)
	if err != nil {
		return deviceQuantMat{}, err
	}
	qBuf, sBuf, err := o.uploadQ4PayloadBuffers(v)
	if err != nil {
		return deviceQuantMat{}, err
	}

	return deviceQuantMat{
		format:       quantMatFormatQ4Raw,
		q:            qBuf,
		scales:       sBuf,
		rows:         w.R,
		cols:         w.C,
		blocksPerRow: v.blocksPerRow,
	}, nil
}

func (o *Ops) ensureQuantMatK4Raw(w *simd.Mat) (deviceQuantMat, error) {
	v, err := parseK4PayloadView(w)
	if err != nil {
		return deviceQuantMat{}, err
	}
	qBuf, superBuf, subBuf, err := o.uploadK4PayloadBuffers(v)
	if err != nil {
		return deviceQuantMat{}, err
	}

	return deviceQuantMat{
		format:       quantMatFormatK4Raw,
		q:            qBuf,
		superScales:  superBuf,
		subScales:    subBuf,
		rows:         w.R,
		cols:         w.C,
		blocksPerRow: v.blocksPerRow,
	}, nil
}

func quantBlocks(rows, cols int) (int, int, error) {
	if rows <= 0 || cols <= 0 {
		return 0, 0, fmt.Errorf("invalid quant matrix shape")
	}
	blocksPerRow := (cols + 31) / 32
	totalBlocks, ok := mulInt(rows, blocksPerRow)
	if !ok {
		return 0, 0, fmt.Errorf("quant matrix too large")
	}
	return blocksPerRow, totalBlocks, nil
}

func mulInt(a, b int) (int, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > int(^uint(0)>>1)/b {
		return 0, false
	}
	return a * b, true
}

func addInt(a, b int) (int, bool) {
	if a > int(^uint(0)>>1)-b {
		return 0, false
	}
	return a + b, true
}

func align64Int(n int) (int, bool) {
	if n < 0 || n > int(^uint(0)>>1)-63 {
		return 0, false
	}
	return (n + 63) &^ 63, true
}

func weightUploadSpec(w *simd.Mat) (native.BlasDataType, int64, unsafe.Pointer, error) {
	if w.Raw == nil || w.DType == mcf.DTypeF32 {
		if len(w.Data) == 0 {
			return 0, 0, nil, nil
		}
		return native.BlasF32, int64(len(w.Data)) * int64(unsafe.Sizeof(float32(0))), unsafe.Pointer(&w.Data[0]), nil
	}
	switch w.DType {
	case mcf.DTypeBF16:
		if len(w.Raw) == 0 {
			return 0, 0, nil, nil
		}
		return native.BlasBF16, int64(len(w.Raw)), unsafe.Pointer(&w.Raw[0]), nil
	case mcf.DTypeF16:
		if len(w.Raw) == 0 {
			return 0, 0, nil, nil
		}
		return native.BlasF16, int64(len(w.Raw)), unsafe.Pointer(&w.Raw[0]), nil
	default:
		return 0, 0, nil, fmt.Errorf("unsupported weight dtype for cuda backend: %s", dtypeString(w.DType))
	}
}

func (o *Ops) ensureHostVecs(xBytes, yBytes int) error {
	if xBytes > o.xCapBytes {
		if err := o.xHost.Free(); err != nil {
			return err
		}
		buf, err := native.AllocHostPinned(int64(xBytes))
		if err != nil {
			return err
		}
		o.xHost = buf
		o.xCapBytes = xBytes
	}
	if yBytes > o.yCapBytes {
		if err := o.yHost.Free(); err != nil {
			return err
		}
		buf, err := native.AllocHostPinned(int64(yBytes))
		if err != nil {
			return err
		}
		o.yHost = buf
		o.yCapBytes = yBytes
	}
	return nil
}

func (o *Ops) ensureDeviceVecs(xBytes, yBytes int) error {
	if xBytes > o.xDevBytes {
		if err := o.xDev.Free(); err != nil {
			return err
		}
		buf, err := native.AllocDevice(int64(xBytes))
		if err != nil {
			return err
		}
		o.xDev = buf
		o.xDevBytes = xBytes
	}
	if yBytes > o.yDevBytes {
		if err := o.yDev.Free(); err != nil {
			return err
		}
		buf, err := native.AllocDevice(int64(yBytes))
		if err != nil {
			return err
		}
		o.yDev = buf
		o.yDevBytes = yBytes
	}
	return nil
}

func (o *Ops) ensureNormTmp(bytes int) error {
	if bytes > o.zDevBytes {
		if err := o.zDev.Free(); err != nil {
			return err
		}
		buf, err := native.AllocDevice(int64(bytes))
		if err != nil {
			return err
		}
		o.zDev = buf
		o.zDevBytes = bytes
	}
	return nil
}

func (o *Ops) ensureActTmp(bytes int) error {
	if bytes > o.aDevBytes {
		if err := o.aDev.Free(); err != nil {
			return err
		}
		buf, err := native.AllocDevice(int64(bytes))
		if err != nil {
			return err
		}
		o.aDev = buf
		o.aDevBytes = bytes
	}
	return nil
}

func (o *Ops) ensureKVU16Tmp(kvStride int) error {
	if kvStride <= 0 {
		return fmt.Errorf("invalid kv stride")
	}
	if len(o.kTmpU16) < kvStride {
		o.kTmpU16 = make([]uint16, kvStride)
	}
	if len(o.vTmpU16) < kvStride {
		o.vTmpU16 = make([]uint16, kvStride)
	}
	return nil
}

func (o *Ops) ensureAttnCache(layer *simd.Layer, kvStride, cacheLen int) (deviceAttnCache, error) {
	if cache, ok := o.attnCaches[layer]; ok {
		return cache, nil
	}
	if kvStride <= 0 || cacheLen <= 0 {
		return deviceAttnCache{}, fmt.Errorf("invalid attention cache dimensions")
	}
	totalElems, ok := mulInt(kvStride, cacheLen)
	if !ok {
		return deviceAttnCache{}, fmt.Errorf("attention cache too large")
	}
	totalBytes, ok := mulInt(totalElems, 2)
	if !ok {
		return deviceAttnCache{}, fmt.Errorf("attention cache too large")
	}

	kBuf, err := native.AllocDevice(int64(totalBytes))
	if err != nil {
		return deviceAttnCache{}, err
	}
	vBuf, err := native.AllocDevice(int64(totalBytes))
	if err != nil {
		_ = kBuf.Free()
		return deviceAttnCache{}, err
	}

	cache := deviceAttnCache{
		k:        kBuf,
		v:        vBuf,
		kvStride: kvStride,
		cacheLen: cacheLen,
	}
	o.attnCaches[layer] = cache
	return cache, nil
}

func (o *Ops) deviceNormWeight(weight []float32) (native.DeviceBuffer, error) {
	if len(weight) == 0 {
		return native.DeviceBuffer{}, fmt.Errorf("empty rmsnorm weight")
	}
	key := uintptr(unsafe.Pointer(&weight[0]))
	if buf, ok := o.normWeights[key]; ok {
		return buf, nil
	}
	bytes := int64(len(weight)) * int64(unsafe.Sizeof(float32(0)))
	buf, err := native.AllocDevice(bytes)
	if err != nil {
		return native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&weight[0]), bytes); err != nil {
		_ = buf.Free()
		return native.DeviceBuffer{}, err
	}
	o.normWeights[key] = buf
	return buf, nil
}

func xBufferBytes(dtype native.BlasDataType, length int) int64 {
	switch dtype {
	case native.BlasF16, native.BlasBF16:
		return int64(length) * 2
	default:
		return int64(length) * 4
	}
}

func max64Local(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func maxInt3(a, b, c int) int {
	if b > a {
		a = b
	}
	if c > a {
		a = c
	}
	return a
}

func fillXBuffer(buf native.HostBuffer, dtype native.BlasDataType, x []float32) error {
	switch dtype {
	case native.BlasF32:
		copy(unsafe.Slice((*float32)(buf.Ptr()), len(x)), x)
		return nil
	case native.BlasF16:
		dst := unsafe.Slice((*uint16)(buf.Ptr()), len(x))
		for i, v := range x {
			dst[i] = simd.Float32ToFloat16(v)
		}
		return nil
	case native.BlasBF16:
		dst := unsafe.Slice((*uint16)(buf.Ptr()), len(x))
		for i, v := range x {
			dst[i] = bf16FromF32(v)
		}
		return nil
	default:
		return fmt.Errorf("unsupported x buffer dtype %d", dtype)
	}
}

func bf16FromF32(v float32) uint16 {
	u := math.Float32bits(v)
	// Round-to-nearest-even on truncated 16 bits to match SIMD path.
	rnd := uint32(0x7FFF + ((u >> 16) & 1))
	return uint16((u + rnd) >> 16)
}

func dtypeString(dt mcf.TensorDType) string {
	return fmt.Sprintf("0x%04x", uint16(dt))
}
