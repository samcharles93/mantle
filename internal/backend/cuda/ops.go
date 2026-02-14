//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type Ops struct {
	stream native.Stream
	blas   native.BlasHandle

	mu             sync.Mutex
	weights        map[*simd.Mat]deviceMat
	qweights       map[*simd.Mat]deviceQuantMat
	normWeights    map[uintptr]native.DeviceBuffer
	attnCaches     map[*simd.Layer]deviceAttnCache
	f16Dequant     map[*simd.Mat]native.DeviceBuffer  // Cached F16 dequantized weights
	convKernels    map[*simd.Mat]native.DeviceBuffer  // Cached conv kernel weights on device
	convStates     map[*simd.Layer]convDeviceState    // Per-layer conv state on device
	convStateReady map[*simd.Layer]bool               // conv state already resident on device
	quantGraphs    map[quantGraphKey]native.GraphExec // Cached CUDA Graph executables for quant kernels
	ffnQuantGraphs map[ffnQuantGraphKey]native.GraphExec

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

	// persistent device buffer for hidden‑state vector x
	xPersistDev   native.DeviceBuffer
	xPersistBytes int
	xHostRef      []float32 // host slice currently resident in xPersistDev
	xHostPtr      uintptr   // &xHostRef[0] (or 0 if empty)
	xHostLen      int
	xHostDirty    bool // host copy newer than device copy
	xDevDirty     bool // device copy newer than host copy

	// device-resident block output: avoids D2H + H2D round-trip through host
	lastResultDev     native.DeviceBuffer // device buffer holding last block output
	lastResultHostPtr uintptr             // host pointer the result would have been written to
	lastResultLen     int                 // length in float32 elements
	lastResultValid   bool                // unconsumed device result available

	// persistent device mirror for the latest pre-norm output host slice
	// produced by DeviceRMSNorm; used by DeviceMatVec to avoid H2D re-uploads.
	normDev      native.DeviceBuffer
	normDevBytes int
	normHostPtr  uintptr
	normHostLen  int
	normDevValid bool

	kTmpU16     []uint16
	vTmpU16     []uint16
	kTmpQ8      []int8
	vTmpQ8      []int8
	kTmpQ8Scale []float32
	vTmpQ8Scale []float32
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
	kF16      native.DeviceBuffer
	vF16      native.DeviceBuffer
	kQ8       native.DeviceBuffer
	vQ8       native.DeviceBuffer
	kQ8Scales native.DeviceBuffer
	vQ8Scales native.DeviceBuffer
	useQ8K    bool
	useQ8V    bool
	kvStride  int
	cacheLen  int
}

type convDeviceState struct {
	buf  native.DeviceBuffer
	size int // bytes allocated
}

type quantMatFormat int

const (
	quantMatFormatCachedInt8 quantMatFormat = iota
	quantMatFormatQ4Raw
	quantMatFormatK4Raw
)

type quantGraphKey struct {
	format       quantMatFormat
	rows         int
	cols         int
	blocksPerRow int
	qPtr         uintptr
	scalePtr     uintptr
	superPtr     uintptr
	subPtr       uintptr
	xPtr         uintptr
	yPtr         uintptr
}

type ffnQuantGraphKey struct {
	upFormat     quantMatFormat
	gateFormat   quantMatFormat
	downFormat   quantMatFormat
	interm       int
	outRows      int
	upQPtr       uintptr
	upScalePtr   uintptr
	upSuperPtr   uintptr
	upSubPtr     uintptr
	gateQPtr     uintptr
	gateScalePtr uintptr
	gateSuperPtr uintptr
	gateSubPtr   uintptr
	downQPtr     uintptr
	downScalePtr uintptr
	downSuperPtr uintptr
	downSubPtr   uintptr
	xPtr         uintptr
	yPtr         uintptr
	zPtr         uintptr
}

type cudaWeightMode int

const (
	cudaWeightModeAuto cudaWeightMode = iota
	cudaWeightModeQuant
	cudaWeightModeDequant
)

func NewOps(stream native.Stream, blas native.BlasHandle) *Ops {
	return &Ops{
		stream:         stream,
		blas:           blas,
		weights:        make(map[*simd.Mat]deviceMat),
		qweights:       make(map[*simd.Mat]deviceQuantMat),
		normWeights:    make(map[uintptr]native.DeviceBuffer),
		attnCaches:     make(map[*simd.Layer]deviceAttnCache),
		f16Dequant:     make(map[*simd.Mat]native.DeviceBuffer),
		convKernels:    make(map[*simd.Mat]native.DeviceBuffer),
		convStates:     make(map[*simd.Layer]convDeviceState),
		convStateReady: make(map[*simd.Layer]bool),
		quantGraphs:    make(map[quantGraphKey]native.GraphExec),
		ffnQuantGraphs: make(map[ffnQuantGraphKey]native.GraphExec),
	}
}

func (o *Ops) clearNormDeviceView() {
	o.normHostPtr = 0
	o.normHostLen = 0
	o.normDevValid = false
}

// FlushBlockResult flushes a pending device-resident block output to host.
// Exposed for the SIMD runtime to force host visibility before host-only ops.
func (o *Ops) FlushBlockResult() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.flushLastResult()
}

// flushLastResult writes a pending device-resident block result back to host.
// Must be called with o.mu held.
func (o *Ops) flushLastResult() {
	if !o.lastResultValid {
		return
	}
	bytes := int64(o.lastResultLen) * 4
	_ = native.MemcpyD2H(unsafe.Pointer(o.lastResultHostPtr), o.lastResultDev, bytes)
	o.lastResultValid = false
}

// ResetConvStates invalidates device-resident conv state so it will be
// re-uploaded from (zeroed) host buffers on the next call.
func (o *Ops) ResetConvStates() {
	o.mu.Lock()
	defer o.mu.Unlock()
	for k := range o.convStateReady {
		delete(o.convStateReady, k)
	}
	o.lastResultValid = false
}

// FusedRMSNormMatVec performs fused RMSNorm + MatVec in a single kernel
// This eliminates one memory roundtrip compared to calling them separately
func (o *Ops) FusedRMSNormMatVec(out []float32, w *simd.Mat, x, normWeight []float32, eps float32) bool {
	// TODO: Fusion causes illegal memory access errors - needs investigation
	// Enable with MANTLE_CUDA_FUSE=1 for testing
	if os.Getenv("MANTLE_CUDA_FUSE") == "" {
		return false
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

	mode := currentCUDAWeightMode()
	uploadMat := func(name string, mat *simd.Mat) error {
		if mat == nil {
			return nil
		}
		preferQuant := shouldPreferQuantWeights(mat, mode)
		if preferQuant {
			if _, err := o.ensureQuantMat(mat); err != nil {
				if mode == cudaWeightModeQuant {
					return fmt.Errorf(
						"%s: quant preload failed (dtype=%s shape=[%d %d] raw=%d): %w",
						name, dtypeString(mat.DType), mat.R, mat.C, len(mat.Raw), err,
					)
				}
				if cudaTraceEnabled() {
					fmt.Fprintf(
						os.Stderr,
						"CUDA preload fallback %s: quant path failed (dtype=%s shape=[%d %d] raw=%d): %v\n",
						name, dtypeString(mat.DType), mat.R, mat.C, len(mat.Raw), err,
					)
				}
			} else {
				return nil
			}
		}
		if _, err := o.deviceMat(mat); err != nil {
			return fmt.Errorf(
				"%s: device upload failed (mode=%s dtype=%s shape=[%d %d] raw=%d): %w",
				name, cudaWeightModeString(mode), dtypeString(mat.DType), mat.R, mat.C, len(mat.Raw), err,
			)
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

// BeginToken implements simd.DeviceStateOps.
func (o *Ops) BeginToken(x []float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(x) == 0 {
		return
	}
	o.xHostRef = x
	o.xHostLen = len(x)
	o.xHostPtr = uintptr(unsafe.Pointer(&x[0]))
	o.xHostDirty = false
	o.xDevDirty = false
	o.clearNormDeviceView()
	// Ensure persistent device buffer
	bytes := int64(len(x)) * int64(unsafe.Sizeof(float32(0)))
	if int(bytes) > o.xPersistBytes {
		if err := o.xPersistDev.Free(); err != nil {
			// ignore error; leak old buffer
		}
		buf, err := native.AllocDevice(bytes)
		if err != nil {
			panic(fmt.Errorf("cuda: failed to allocate persistent x buffer (%d bytes): %w", bytes, err))
		}
		o.xPersistDev = buf
		o.xPersistBytes = int(bytes)
	}
	// Upload x to device
	if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&x[0]), bytes); err != nil {
		panic(fmt.Errorf("cuda: failed to upload x to device (%d bytes): %w", bytes, err))
	}
}

// EndToken implements simd.DeviceStateOps.
func (o *Ops) EndToken(x []float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.flushLastResult()
	if len(x) == 0 || o.xHostLen == 0 || o.xHostPtr != uintptr(unsafe.Pointer(&x[0])) {
		// Not our buffer, or no persistent buffer
		return
	}
	if o.xDevDirty {
		// Download device copy back to host
		bytes := int64(len(x)) * int64(unsafe.Sizeof(float32(0)))
		if err := native.MemcpyD2H(unsafe.Pointer(&x[0]), o.xPersistDev, bytes); err != nil {
			panic(fmt.Errorf("cuda: failed to download x from device (%d bytes): %w", bytes, err))
		}
		o.xDevDirty = false
	}
	// Clear reference
	o.xHostRef = nil
	o.xHostPtr = 0
	o.xHostLen = 0
	o.xHostDirty = false
	o.clearNormDeviceView()
}

// HostStateDirty implements simd.DeviceStateOps.
func (o *Ops) HostStateDirty(x []float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	// Any host-side update may stale temporary device mirrors.
	o.clearNormDeviceView()
	if len(x) == 0 || o.xHostLen == 0 || len(x) != o.xHostLen {
		return
	}
	if uintptr(unsafe.Pointer(&x[0])) != o.xHostPtr {
		return
	}
	o.xHostDirty = true
	o.xDevDirty = false
}

// SyncHostState implements simd.DeviceStateOps.
func (o *Ops) SyncHostState(x []float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.flushLastResult()
	if len(x) == 0 || o.xHostLen == 0 || len(x) != o.xHostLen {
		return
	}
	if uintptr(unsafe.Pointer(&x[0])) != o.xHostPtr {
		return
	}
	if !o.xDevDirty {
		return
	}
	bytes := int64(len(x)) * int64(unsafe.Sizeof(float32(0)))
	if err := native.MemcpyD2H(unsafe.Pointer(&x[0]), o.xPersistDev, bytes); err != nil {
		panic(fmt.Errorf("cuda: failed to sync x from device (%d bytes): %w", bytes, err))
	}
	o.xDevDirty = false
	o.xHostDirty = false
}

// SyncDeviceSlice ensures host visibility for known device-backed scratch
// slices (persistent x, norm output, pending block result).
func (o *Ops) SyncDeviceSlice(x []float32) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(x) == 0 {
		return
	}
	ptr := uintptr(unsafe.Pointer(&x[0]))
	n := len(x)

	// Pending block output staged for residual add.
	if o.lastResultValid && ptr == o.lastResultHostPtr && n == o.lastResultLen {
		o.flushLastResult()
		return
	}
	// Persistent token state.
	if o.xHostLen == n && ptr == o.xHostPtr && o.xDevDirty {
		bytes := int64(n) * int64(unsafe.Sizeof(float32(0)))
		if err := native.MemcpyD2H(unsafe.Pointer(&x[0]), o.xPersistDev, bytes); err != nil {
			panic(fmt.Errorf("cuda: failed to sync persistent device slice (%d bytes): %w", bytes, err))
		}
		o.xDevDirty = false
		o.xHostDirty = false
		return
	}
	// RMSNorm device view.
	if o.normDevValid && o.normHostLen == n && ptr == o.normHostPtr {
		bytes := int64(n) * int64(unsafe.Sizeof(float32(0)))
		if err := native.MemcpyD2H(unsafe.Pointer(&x[0]), o.normDev, bytes); err != nil {
			panic(fmt.Errorf("cuda: failed to sync norm device slice (%d bytes): %w", bytes, err))
		}
		return
	}
}

// DeviceAdd implements simd.DeviceStateOps.
func (o *Ops) DeviceAdd(dst, src []float32) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.xHostLen == 0 || len(dst) != o.xHostLen || len(src) != o.xHostLen {
		o.flushLastResult()
		return false
	}
	if uintptr(unsafe.Pointer(&dst[0])) != o.xHostPtr {
		o.flushLastResult()
		return false
	}
	bytes := int64(len(dst)) * int64(unsafe.Sizeof(float32(0)))
	if o.xHostDirty {
		if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&dst[0]), bytes); err != nil {
			o.flushLastResult()
			return false
		}
		o.xHostDirty = false
		o.xDevDirty = false
	}

	// Fast path: consume device-resident block output directly (no H2D round-trip)
	if o.lastResultValid && uintptr(unsafe.Pointer(&src[0])) == o.lastResultHostPtr && len(src) == o.lastResultLen {
		if err := native.AddVectorsF32(o.xPersistDev, o.lastResultDev, len(dst), o.stream); err != nil {
			o.flushLastResult()
			return false
		}
		o.lastResultValid = false
		o.xHostDirty = false
		o.xDevDirty = true
		return true
	}

	// Slow path: H2D src from host
	o.flushLastResult()
	if err := o.ensureDeviceVecs(0, int(bytes)); err != nil {
		return false
	}
	if err := native.MemcpyH2D(o.yDev, unsafe.Pointer(&src[0]), bytes); err != nil {
		return false
	}
	if err := native.AddVectorsF32(o.xPersistDev, o.yDev, len(dst), o.stream); err != nil {
		return false
	}
	o.xHostDirty = false
	o.xDevDirty = true
	return true
}

// DeviceRMSNorm implements simd.DeviceStateOps.
func (o *Ops) DeviceRMSNorm(dst, src, weight []float32, eps float32) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.clearNormDeviceView()
	if !o.isDeviceX(src) {
		return false
	}
	if len(dst) != len(src) || len(weight) != len(src) {
		return false
	}
	if o.xHostDirty {
		bytes := int64(o.xHostLen) * int64(unsafe.Sizeof(float32(0)))
		if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&src[0]), bytes); err != nil {
			return false
		}
		o.xHostDirty = false
		o.xDevDirty = false
	}
	if err := o.rmsNormWithDeviceX(dst, weight, eps); err != nil {
		return false
	}
	return true
}

// DeviceMatVec implements simd.DeviceStateOps.
func (o *Ops) DeviceMatVec(dst []float32, w *simd.Mat, x []float32) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(dst) < w.R {
		return false
	}

	var (
		xDev native.DeviceBuffer
		xLen int
	)

	switch {
	case o.isDeviceX(x):
		if o.xHostDirty {
			bytes := int64(o.xHostLen) * int64(unsafe.Sizeof(float32(0)))
			if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&x[0]), bytes); err != nil {
				return false
			}
			o.xHostDirty = false
			o.xDevDirty = false
		}
		xDev = o.xPersistDev
		xLen = o.xHostLen
	case o.isNormDeviceX(x):
		xDev = o.normDev
		xLen = o.normHostLen
	default:
		return false
	}

	if err := o.matVecWithDeviceInput(dst, w, xDev, xLen); err != nil {
		return false
	}
	return true
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

	for _, buf := range o.f16Dequant {
		if e := buf.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.f16Dequant = nil

	for _, exec := range o.quantGraphs {
		if e := exec.Destroy(); e != nil && err == nil {
			err = e
		}
	}
	o.quantGraphs = nil
	for _, exec := range o.ffnQuantGraphs {
		if e := exec.Destroy(); e != nil && err == nil {
			err = e
		}
	}
	o.ffnQuantGraphs = nil

	for _, buf := range o.convKernels {
		if e := buf.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.convKernels = nil

	for _, cs := range o.convStates {
		if e := cs.buf.Free(); e != nil && err == nil {
			err = e
		}
	}
	o.convStates = nil

	for _, cache := range o.attnCaches {
		if e := cache.kF16.Free(); e != nil && err == nil {
			err = e
		}
		if e := cache.vF16.Free(); e != nil && err == nil {
			err = e
		}
		if e := cache.kQ8.Free(); e != nil && err == nil {
			err = e
		}
		if e := cache.vQ8.Free(); e != nil && err == nil {
			err = e
		}
		if e := cache.kQ8Scales.Free(); e != nil && err == nil {
			err = e
		}
		if e := cache.vQ8Scales.Free(); e != nil && err == nil {
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
	if e := o.xPersistDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.normDev.Free(); e != nil && err == nil {
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
	o.xPersistDev = native.DeviceBuffer{}
	o.normDev = native.DeviceBuffer{}
	o.xHost = native.HostBuffer{}
	o.yHost = native.HostBuffer{}
	o.xCapBytes = 0
	o.yCapBytes = 0
	o.xDevBytes = 0
	o.yDevBytes = 0
	o.zDevBytes = 0
	o.aDevBytes = 0
	o.xPersistBytes = 0
	o.normDevBytes = 0
	o.xHostRef = nil
	o.xHostPtr = 0
	o.xHostLen = 0
	o.xHostDirty = false
	o.xDevDirty = false
	o.clearNormDeviceView()

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

	if useQuantKernel() && shouldPreferQuantWeights(w, currentCUDAWeightMode()) {
		o.matVecQuant(dst, w, x)
		return
	}
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

// isDeviceX returns true if x is the slice currently resident in the persistent device buffer.
func (o *Ops) isDeviceX(x []float32) bool {
	if o.xHostLen == 0 || len(x) != o.xHostLen {
		return false
	}
	return uintptr(unsafe.Pointer(&x[0])) == o.xHostPtr
}

// isNormDeviceX returns true when x matches the host slice mirrored in normDev.
func (o *Ops) isNormDeviceX(x []float32) bool {
	if !o.normDevValid || o.normHostLen == 0 || len(x) != o.normHostLen {
		return false
	}
	return uintptr(unsafe.Pointer(&x[0])) == o.normHostPtr
}

// matVecWithDeviceX performs MatVec using x already on device (xPersistDev).
// Requires o.mu locked.
func (o *Ops) matVecWithDeviceX(dst []float32, w *simd.Mat) error {
	return o.matVecWithDeviceInput(dst, w, o.xPersistDev, o.xHostLen)
}

// matVecWithDeviceInput performs MatVec using an already resident device input vector.
// Requires o.mu locked.
func (o *Ops) matVecWithDeviceInput(dst []float32, w *simd.Mat, xDev native.DeviceBuffer, xLen int) error {
	if w == nil || w.R == 0 || w.C == 0 {
		return fmt.Errorf("invalid weight matrix")
	}
	if xLen != w.C {
		return fmt.Errorf("x length mismatch")
	}
	// Use quant path if appropriate
	if useQuantKernel() && shouldPreferQuantWeights(w, currentCUDAWeightMode()) {
		return o.matVecQuantWithDeviceInput(dst, w, xDev)
	}
	if useQuantKernel() && w.Quant != nil && w.Quant.ValidFor(w) {
		return o.matVecQuantWithDeviceInput(dst, w, xDev)
	}
	// Regular dequantized path
	devW, err := o.deviceMat(w)
	if err != nil {
		return err
	}
	yBytes := int64(w.R) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureDeviceVecs(0, int(yBytes)); err != nil {
		return err
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
		xDev,
		native.BlasF32,
		w.C,
		0.0,
		o.yDev,
		native.BlasF32,
		w.R,
		native.BlasComputeF32,
		native.BlasGemmDefault,
	); err != nil {
		return err
	}
	// Copy result to host dst
	if err := o.waitForResult(unsafe.Pointer(&dst[0]), o.yDev, yBytes); err != nil {
		return err
	}
	return nil
}

// matVecQuantWithDeviceX performs quantized MatVec using x already on device.
// Requires o.mu locked.
func (o *Ops) matVecQuantWithDeviceX(dst []float32, w *simd.Mat) error {
	return o.matVecQuantWithDeviceInput(dst, w, o.xPersistDev)
}

// matVecQuantWithDeviceInput performs quantized MatVec with an already resident device input.
// Requires o.mu locked.
func (o *Ops) matVecQuantWithDeviceInput(dst []float32, w *simd.Mat, xDev native.DeviceBuffer) error {
	qw, err := o.ensureQuantMat(w)
	if err != nil {
		return err
	}
	yBytes := int64(w.R) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureDeviceVecs(0, int(yBytes)); err != nil {
		return err
	}
	var kernelErr error
	switch qw.format {
	case quantMatFormatQ4Raw:
		kernelErr = native.QuantMatVecQ4F32(qw.q, qw.scales, xDev, o.yDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
	case quantMatFormatK4Raw:
		kernelErr = native.QuantMatVecK4F32(qw.q, qw.superScales, qw.subScales, xDev, o.yDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
	default:
		kernelErr = native.QuantMatVecInt8BlocksF32(qw.q, qw.scales, xDev, o.yDev, qw.rows, qw.blocksPerRow, qw.cols, o.stream)
	}
	if kernelErr != nil {
		return kernelErr
	}
	if err := o.waitForResult(unsafe.Pointer(&dst[0]), o.yDev, yBytes); err != nil {
		return err
	}
	return nil
}

// rmsNormWithDeviceX performs RMSNorm using src already on device (xPersistDev).
// Requires o.mu locked.
func (o *Ops) rmsNormWithDeviceX(dst, weight []float32, eps float32) error {
	n := len(dst)
	if n == 0 || n != o.xHostLen {
		return fmt.Errorf("size mismatch")
	}
	wDev, err := o.deviceNormWeight(weight)
	if err != nil {
		return err
	}
	bytes := int64(n) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureNormDeviceVec(int(bytes)); err != nil {
		return err
	}
	if err := native.RMSNormF32(o.normDev, o.xPersistDev, wDev, eps, n, o.stream); err != nil {
		return err
	}
	// Keep normalized vector on device; host copy is performed lazily
	// via SyncDeviceSlice when a host fallback path needs it.
	o.normHostPtr = uintptr(unsafe.Pointer(&dst[0]))
	o.normHostLen = n
	o.normDevValid = true
	return nil
}

func (o *Ops) matVecQuant(dst []float32, w *simd.Mat, x []float32) {
	qw, err := o.ensureQuantMat(w)
	if err != nil {
		panic(fmt.Errorf("cuda quant matvec weight upload failed (r=%d c=%d dtype=%s): %w", w.R, w.C, dtypeString(w.DType), err))
	}

	xBytes := int64(w.C) * int64(unsafe.Sizeof(float32(0)))
	yBytes := int64(w.R) * int64(unsafe.Sizeof(float32(0)))
	hostYBytes := int64(0)
	if useLegacyStreamSync() {
		// Legacy mode uses async D2H and requires staging into pinned host memory.
		hostYBytes = yBytes
	}
	if err := o.ensureHostVecs(int(xBytes), int(hostYBytes)); err != nil {
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

	hostDst := unsafe.Pointer(&dst[0])
	if useLegacyStreamSync() {
		hostDst = o.yHost.Ptr()
	}
	if err := o.waitForResult(hostDst, o.yDev, yBytes); err != nil {
		panic(fmt.Errorf("cuda quant matvec result wait failed: %w", err))
	}
	if useLegacyStreamSync() {
		copy(dst[:w.R], unsafe.Slice((*float32)(o.yHost.Ptr()), w.R))
	}
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
	// When weights prefer quant path, use the quantized FFN fast path
	// which runs all 3 projections + SiLU on device using quant kernels.
	mode := currentCUDAWeightMode()
	if mode != cudaWeightModeDequant &&
		(shouldPreferQuantWeights(layer.FfnUp, mode) ||
			shouldPreferQuantWeights(layer.FfnGate, mode) ||
			shouldPreferQuantWeights(layer.FfnDown, mode)) {
		return o.quantFFNBlock(layer, x, out)
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

	xInput := o.xDev
	xInputType := upW.dtype
	devInput, usedDeviceInput, err := o.deviceInputForVector(x[:len(x)], upW.dtype, len(x))
	if err != nil {
		return false
	}
	if usedDeviceInput {
		xInput = devInput.buf
		xInputType = devInput.dtype
	}
	hostXBytes := int(xBytes)
	if usedDeviceInput {
		hostXBytes = 0
	}

	if err := o.ensureHostVecs(hostXBytes, int(outBytes)); err != nil {
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

	if !usedDeviceInput {
		if err := fillXBuffer(o.xHost, upW.dtype, x[:len(x)]); err != nil {
			return false
		}
		if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
			return false
		}
	}

	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, upW.rows, 1, upW.cols, 1.0, upW.buf, upW.dtype, upW.cols, xInput, xInputType, upW.cols, 0.0, o.yDev, native.BlasF32, upW.rows, native.BlasComputeF32, native.BlasGemmDefault); err != nil {
		return false
	}
	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, gateW.rows, 1, gateW.cols, 1.0, gateW.buf, gateW.dtype, gateW.cols, xInput, xInputType, gateW.cols, 0.0, o.zDev, native.BlasF32, gateW.rows, native.BlasComputeF32, native.BlasGemmDefault); err != nil {
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

// quantFFNBlock runs the full FFN (Up, Gate, SiLU, Down) on device using quantized matvec kernels.
// This avoids H2D transfers for intermediate results and keeps the entire FFN computation on-GPU.
func (o *Ops) quantFFNBlock(layer *simd.Layer, x []float32, out []float32) bool {
	if len(x) < layer.FfnUp.C || len(out) < layer.FfnDown.R {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	// Ensure quantized weight matrices on device
	qUp, err := o.ensureQuantMat(layer.FfnUp)
	if err != nil {
		return false
	}
	qGate, err := o.ensureQuantMat(layer.FfnGate)
	if err != nil {
		return false
	}
	qDown, err := o.ensureQuantMat(layer.FfnDown)
	if err != nil {
		return false
	}

	interm := layer.FfnUp.R
	cols := layer.FfnUp.C
	outRows := layer.FfnDown.R

	// Buffer sizes: x input (F32), intermediate (F32), output (F32)
	xBytes := int64(cols) * 4
	intermBytes := int64(interm) * 4
	outBytes := int64(outRows) * 4

	// Try to use device-resident x to avoid H2D copy
	var xInput native.DeviceBuffer
	usedDeviceInput := false
	if o.isNormDeviceX(x) {
		xInput = o.normDev
		usedDeviceInput = true
	} else if o.isDeviceX(x) {
		if o.xHostDirty {
			bytes := int64(o.xHostLen) * 4
			if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&x[0]), bytes); err != nil {
				return false
			}
			o.xHostDirty = false
			o.xDevDirty = false
		}
		xInput = o.xPersistDev
		usedDeviceInput = true
	}

	// Ensure device buffers: xDev for input, yDev for Up result, zDev for Gate/Down result
	if err := o.ensureDeviceVecs(int(xBytes), int(intermBytes)); err != nil {
		return false
	}
	if err := o.ensureNormTmp(int(max64Local(intermBytes, outBytes))); err != nil {
		return false
	}

	if !usedDeviceInput {
		// H2D copy x
		if err := o.ensureHostVecs(int(xBytes), int(outBytes)); err != nil {
			return false
		}
		copy(unsafe.Slice((*float32)(o.xHost.Ptr()), cols), x[:cols])
		if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
			return false
		}
		xInput = o.xDev
	} else {
		if err := o.ensureHostVecs(0, int(outBytes)); err != nil {
			return false
		}
	}

	if useCUDAGraphs() {
		if err := o.runQuantFFNGraph(qUp, qGate, qDown, xInput, interm, outRows, o.stream); err != nil {
			native.RecordGraphFailure()
			if err := o.runQuantFFNDirect(qUp, qGate, qDown, xInput, interm, o.stream); err != nil {
				return false
			}
		}
	} else if err := o.runQuantFFNDirect(qUp, qGate, qDown, xInput, interm, o.stream); err != nil {
		return false
	}

	// Result stays on device — DeviceAdd will consume it directly.
	o.lastResultDev = o.zDev
	o.lastResultHostPtr = uintptr(unsafe.Pointer(&out[0]))
	o.lastResultLen = outRows
	o.lastResultValid = true
	runtime.KeepAlive(x)
	runtime.KeepAlive(out)
	return true
}

// runQuantKernel dispatches the appropriate quantized matvec kernel based on weight format.
func (o *Ops) runQuantKernel(qw deviceQuantMat, xDev, yDev native.DeviceBuffer, stream native.Stream) error {
	if useCUDAGraphs() {
		if err := o.runQuantKernelGraph(qw, xDev, yDev, stream); err == nil {
			return nil
		}
		native.RecordGraphFailure()
	}

	return runQuantKernelDirect(qw, xDev, yDev, stream)
}

func (o *Ops) runQuantKernelGraph(qw deviceQuantMat, xDev, yDev native.DeviceBuffer, stream native.Stream) error {
	key := quantGraphKey{
		format:       qw.format,
		rows:         qw.rows,
		cols:         qw.cols,
		blocksPerRow: qw.blocksPerRow,
		qPtr:         uintptr(qw.q.Ptr()),
		scalePtr:     uintptr(qw.scales.Ptr()),
		superPtr:     uintptr(qw.superScales.Ptr()),
		subPtr:       uintptr(qw.subScales.Ptr()),
		xPtr:         uintptr(xDev.Ptr()),
		yPtr:         uintptr(yDev.Ptr()),
	}

	if exec, ok := o.quantGraphs[key]; ok {
		if err := exec.Launch(stream); err != nil {
			native.RecordGraphFailure()
			return err
		}
		native.RecordGraphLaunch()
		return nil
	}

	if err := stream.BeginCapture(); err != nil {
		return err
	}
	native.RecordGraphCapture()

	kernelErr := runQuantKernelDirect(qw, xDev, yDev, stream)
	if kernelErr != nil {
		_, _ = stream.EndCapture()
		return kernelErr
	}

	graph, err := stream.EndCapture()
	if err != nil {
		return err
	}
	defer func() {
		_ = graph.Destroy()
	}()

	exec, err := graph.Instantiate()
	if err != nil {
		return err
	}
	o.quantGraphs[key] = exec
	if err := exec.Launch(stream); err != nil {
		native.RecordGraphFailure()
		return err
	}
	native.RecordGraphLaunch()
	return nil
}

func runQuantKernelDirect(qw deviceQuantMat, xDev, yDev native.DeviceBuffer, stream native.Stream) error {
	switch qw.format {
	case quantMatFormatQ4Raw:
		return native.QuantMatVecQ4F32(qw.q, qw.scales, xDev, yDev, qw.rows, qw.blocksPerRow, qw.cols, stream)
	case quantMatFormatK4Raw:
		return native.QuantMatVecK4F32(qw.q, qw.superScales, qw.subScales, xDev, yDev, qw.rows, qw.blocksPerRow, qw.cols, stream)
	default:
		return native.QuantMatVecInt8BlocksF32(qw.q, qw.scales, xDev, yDev, qw.rows, qw.blocksPerRow, qw.cols, stream)
	}
}

func (o *Ops) runQuantFFNDirect(qUp, qGate, qDown deviceQuantMat, xInput native.DeviceBuffer, interm int, stream native.Stream) error {
	if err := runQuantKernelDirect(qUp, xInput, o.yDev, stream); err != nil {
		return err
	}
	if err := runQuantKernelDirect(qGate, xInput, o.zDev, stream); err != nil {
		return err
	}
	if err := native.SiluMulF32(o.zDev, o.yDev, o.yDev, interm, stream); err != nil {
		return err
	}
	return runQuantKernelDirect(qDown, o.yDev, o.zDev, stream)
}

func (o *Ops) runQuantFFNGraph(qUp, qGate, qDown deviceQuantMat, xInput native.DeviceBuffer, interm, outRows int, stream native.Stream) error {
	key := ffnQuantGraphKey{
		upFormat:     qUp.format,
		gateFormat:   qGate.format,
		downFormat:   qDown.format,
		interm:       interm,
		outRows:      outRows,
		upQPtr:       uintptr(qUp.q.Ptr()),
		upScalePtr:   uintptr(qUp.scales.Ptr()),
		upSuperPtr:   uintptr(qUp.superScales.Ptr()),
		upSubPtr:     uintptr(qUp.subScales.Ptr()),
		gateQPtr:     uintptr(qGate.q.Ptr()),
		gateScalePtr: uintptr(qGate.scales.Ptr()),
		gateSuperPtr: uintptr(qGate.superScales.Ptr()),
		gateSubPtr:   uintptr(qGate.subScales.Ptr()),
		downQPtr:     uintptr(qDown.q.Ptr()),
		downScalePtr: uintptr(qDown.scales.Ptr()),
		downSuperPtr: uintptr(qDown.superScales.Ptr()),
		downSubPtr:   uintptr(qDown.subScales.Ptr()),
		xPtr:         uintptr(xInput.Ptr()),
		yPtr:         uintptr(o.yDev.Ptr()),
		zPtr:         uintptr(o.zDev.Ptr()),
	}

	if exec, ok := o.ffnQuantGraphs[key]; ok {
		if err := exec.Launch(stream); err != nil {
			native.RecordGraphFailure()
			return err
		}
		native.RecordGraphLaunch()
		return nil
	}

	if err := stream.BeginCapture(); err != nil {
		return err
	}
	native.RecordGraphCapture()
	if err := runQuantKernelDirect(qUp, xInput, o.yDev, stream); err != nil {
		_, _ = stream.EndCapture()
		return err
	}
	if err := runQuantKernelDirect(qGate, xInput, o.zDev, stream); err != nil {
		_, _ = stream.EndCapture()
		return err
	}
	if err := native.SiluMulF32(o.zDev, o.yDev, o.yDev, interm, stream); err != nil {
		_, _ = stream.EndCapture()
		return err
	}
	if err := runQuantKernelDirect(qDown, o.yDev, o.zDev, stream); err != nil {
		_, _ = stream.EndCapture()
		return err
	}

	graph, err := stream.EndCapture()
	if err != nil {
		return err
	}
	defer func() {
		_ = graph.Destroy()
	}()

	exec, err := graph.Instantiate()
	if err != nil {
		return err
	}
	o.ffnQuantGraphs[key] = exec
	if err := exec.Launch(stream); err != nil {
		native.RecordGraphFailure()
		return err
	}
	native.RecordGraphLaunch()
	return nil
}

// ShortConvBlock runs the full ShortConv block (InProj, depthwise conv, OutProj) on device.
// This eliminates H2D/D2H transfers for intermediate proj and conv results.
func (o *Ops) ShortConvBlock(layer *simd.Layer, x []float32, out []float32, embd int) bool {
	if layer == nil || layer.ShortConvInProj == nil || layer.ShortConvOutProj == nil || layer.ShortConvKernel == nil {
		return false
	}
	if layer.ShortConvState.Buf == nil || embd <= 0 {
		return false
	}

	inProj := layer.ShortConvInProj
	outProj := layer.ShortConvOutProj
	kernel := layer.ShortConvKernel
	klen := kernel.C
	state := layer.ShortConvState.Buf

	if len(x) < inProj.C || len(out) < outProj.R {
		return false
	}

	mode := currentCUDAWeightMode()
	if mode != cudaWeightModeDequant &&
		!shouldPreferQuantWeights(inProj, mode) &&
		!shouldPreferQuantWeights(outProj, mode) {
		return false // Only use this fast path for quantized weights
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	// Ensure quantized weights on device
	qIn, err := o.ensureQuantMat(inProj)
	if err != nil {
		return false
	}
	qOut, err := o.ensureQuantMat(outProj)
	if err != nil {
		return false
	}

	// Upload conv kernel weights (cached, F32, embd*klen floats)
	convWDev, err := o.ensureConvKernel(kernel)
	if err != nil {
		return false
	}

	// Ensure conv state device buffer (embd*(klen-1) floats)
	stateLen := embd * (klen - 1)
	stateBytes := int64(stateLen) * 4
	convState, err := o.ensureConvState(layer, int(stateBytes))
	if err != nil {
		return false
	}

	inCols := inProj.C
	projRows := inProj.R // 3*embd
	outRows := outProj.R

	xBytes := int64(inCols) * 4
	projBytes := int64(projRows) * 4
	outBytes := int64(outRows) * 4

	// Try device-resident x
	var xInput native.DeviceBuffer
	usedDeviceInput := false
	if o.isNormDeviceX(x) {
		xInput = o.normDev
		usedDeviceInput = true
	} else if o.isDeviceX(x) {
		if o.xHostDirty {
			bytes := int64(o.xHostLen) * 4
			if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&x[0]), bytes); err != nil {
				return false
			}
			o.xHostDirty = false
			o.xDevDirty = false
		}
		xInput = o.xPersistDev
		usedDeviceInput = true
	}

	// Ensure device buffers
	// yDev: proj output (3*embd), also reused for OutProj output
	// zDev: conv output (embd), also reused for OutProj result
	if err := o.ensureDeviceVecs(int(xBytes), int(max64Local(projBytes, outBytes))); err != nil {
		return false
	}
	if err := o.ensureNormTmp(int(max64Local(int64(embd)*4, outBytes))); err != nil {
		return false
	}

	if !usedDeviceInput {
		if err := o.ensureHostVecs(int(xBytes), int(outBytes)); err != nil {
			return false
		}
		copy(unsafe.Slice((*float32)(o.xHost.Ptr()), inCols), x[:inCols])
		if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
			return false
		}
		xInput = o.xDev
	} else {
		if err := o.ensureHostVecs(0, int(outBytes)); err != nil {
			return false
		}
	}

	// Step 1: InProj(x) → yDev [3*embd floats]
	if err := o.runQuantKernel(qIn, xInput, o.yDev, o.stream); err != nil {
		return false
	}

	// Step 2: H2D conv state (only on first call per layer; state stays on device after that)
	if stateLen > 0 && !o.convStateReady[layer] {
		if err := native.MemcpyH2DAsync(convState.buf, unsafe.Pointer(&state[0]), stateBytes, o.stream); err != nil {
			return false
		}
	}

	// Step 3: ShortConv depthwise kernel: proj(yDev) + convW + state → zDev
	// Conv state is updated in-place on device by the kernel.
	if err := native.ShortConvDepthwise(o.yDev, convWDev, convState.buf, o.zDev, embd, klen, o.stream); err != nil {
		return false
	}

	// Mark conv state as device-resident (skip H2D on subsequent calls)
	o.convStateReady[layer] = true

	// Step 5: OutProj(convOut) → yDev
	// zDev has the conv output (embd F32), use it as input for OutProj
	if err := o.runQuantKernel(qOut, o.zDev, o.yDev, o.stream); err != nil {
		return false
	}

	// Result stays on device — DeviceAdd will consume it directly.
	o.lastResultDev = o.yDev
	o.lastResultHostPtr = uintptr(unsafe.Pointer(&out[0]))
	o.lastResultLen = outRows
	o.lastResultValid = true
	runtime.KeepAlive(x)
	runtime.KeepAlive(out)
	runtime.KeepAlive(state)
	return true
}

// ensureConvKernel uploads F32 conv kernel weights to device, cached per Mat.
func (o *Ops) ensureConvKernel(kernel *simd.Mat) (native.DeviceBuffer, error) {
	if buf, ok := o.convKernels[kernel]; ok {
		return buf, nil
	}
	n := kernel.R * kernel.C
	bytes := int64(n) * 4
	buf, err := native.AllocDevice(bytes)
	if err != nil {
		return native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&kernel.Data[0]), bytes); err != nil {
		buf.Free()
		return native.DeviceBuffer{}, err
	}
	o.convKernels[kernel] = buf
	return buf, nil
}

// ensureConvState ensures a device buffer exists for the layer's conv state.
func (o *Ops) ensureConvState(layer *simd.Layer, bytes int) (convDeviceState, error) {
	if cs, ok := o.convStates[layer]; ok && cs.size >= bytes {
		return cs, nil
	}
	// Free old if exists
	if cs, ok := o.convStates[layer]; ok {
		cs.buf.Free()
		delete(o.convStates, layer)
	}
	buf, err := native.AllocDevice(int64(bytes))
	if err != nil {
		return convDeviceState{}, err
	}
	cs := convDeviceState{buf: buf, size: bytes}
	o.convStates[layer] = cs
	return cs, nil
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

	xInput := o.xDev
	xInputType := dq.dtype
	devInput, usedDeviceInput, err := o.deviceInputForVector(x[:wq.C], dq.dtype, wq.C)
	if err != nil {
		return false
	}
	if usedDeviceInput {
		xInput = devInput.buf
		xInputType = devInput.dtype
	} else {
		if err := fillXBuffer(o.xHost, dq.dtype, x[:wq.C]); err != nil {
			return false
		}
		if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
			return false
		}
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
			xInput,
			xInputType,
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

	qPtr := o.yHost.Ptr()
	kPtr := unsafe.Add(qPtr, int(qBytes))
	vPtr := unsafe.Add(qPtr, int(qBytes+kBytes))
	if useLegacyStreamSync() {
		if err := native.MemcpyD2HAsync(qPtr, o.yDev, qBytes, o.stream); err != nil {
			return false
		}
		if err := native.MemcpyD2HAsync(kPtr, o.zDev, kBytes, o.stream); err != nil {
			return false
		}
		if err := native.MemcpyD2HAsync(vPtr, o.aDev, vBytes, o.stream); err != nil {
			return false
		}
		if err := o.stream.Synchronize(); err != nil {
			return false
		}
	} else {
		if err := native.MemcpyD2H(qPtr, o.yDev, qBytes); err != nil {
			return false
		}
		if err := native.MemcpyD2H(kPtr, o.zDev, kBytes); err != nil {
			return false
		}
		if err := native.MemcpyD2H(vPtr, o.aDev, vBytes); err != nil {
			return false
		}
	}

	copy(q[:dq.rows], unsafe.Slice((*float32)(qPtr), dq.rows))
	copy(k[:dk.rows], unsafe.Slice((*float32)(kPtr), dk.rows))
	copy(v[:dv.rows], unsafe.Slice((*float32)(vPtr), dv.rows))
	return true
}

type deviceInputRef struct {
	buf   native.DeviceBuffer
	dtype native.BlasDataType
}

// deviceInputForVector returns a device-resident input vector when x already
// has a tracked device mirror. If dtype is f16 and the tracked mirror is f32,
// this converts f32->f16 on-device into o.xDev.
// Requires o.mu locked.
func (o *Ops) deviceInputForVector(x []float32, dtype native.BlasDataType, length int) (deviceInputRef, bool, error) {
	var src native.DeviceBuffer
	switch {
	case o.isDeviceX(x):
		if o.xHostDirty {
			bytes := int64(o.xHostLen) * int64(unsafe.Sizeof(float32(0)))
			if err := native.MemcpyH2D(o.xPersistDev, unsafe.Pointer(&x[0]), bytes); err != nil {
				return deviceInputRef{}, false, err
			}
			o.xHostDirty = false
			o.xDevDirty = false
		}
		src = o.xPersistDev
	case o.isNormDeviceX(x):
		src = o.normDev
	default:
		return deviceInputRef{}, false, nil
	}

	switch dtype {
	case native.BlasF32:
		return deviceInputRef{buf: src, dtype: native.BlasF32}, true, nil
	case native.BlasF16:
		xBytes := int(xBufferBytes(native.BlasF16, length))
		if err := o.ensureDeviceVecs(xBytes, 0); err != nil {
			return deviceInputRef{}, false, err
		}
		if err := native.ConvertF32ToF16(src, o.xDev, length, o.stream); err != nil {
			return deviceInputRef{}, false, err
		}
		return deviceInputRef{buf: o.xDev, dtype: native.BlasF16}, true, nil
	default:
		// BF16 and other encodings currently fall back to host upload.
		return deviceInputRef{}, false, nil
	}
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
	needsF16 := !cache.useQ8K || !cache.useQ8V
	needsQ8 := cache.useQ8K || cache.useQ8V
	if needsF16 {
		if err := o.ensureKVU16Tmp(kvStride); err != nil {
			return false
		}
	}
	if needsQ8 {
		if err := o.ensureKVQ8Tmp(kvStride); err != nil {
			return false
		}
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), nHead*headDim), q[:nHead*headDim])
	if !cache.useQ8K {
		for i := range kvStride {
			o.kTmpU16[i] = simd.Float32ToFloat16(k[i])
		}
	} else {
		o.kTmpQ8Scale[0] = quantizeQ8(k, o.kTmpQ8[:kvStride])
	}
	if !cache.useQ8V {
		for i := range kvStride {
			o.vTmpU16[i] = simd.Float32ToFloat16(v[i])
		}
	} else {
		o.vTmpQ8Scale[0] = quantizeQ8(v, o.vTmpQ8[:kvStride])
	}

	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	if cache.useQ8K {
		scaleOffset := int64(cachePos) * int64(unsafe.Sizeof(float32(0)))
		kvBytes := int64(kvStride)
		dstOffset := int64(cachePos * kvStride)
		if err := native.MemcpyH2DAsyncAt(cache.kQ8, dstOffset, unsafe.Pointer(&o.kTmpQ8[0]), kvBytes, o.stream); err != nil {
			return false
		}
		if err := native.MemcpyH2DAsyncAt(cache.kQ8Scales, scaleOffset, unsafe.Pointer(&o.kTmpQ8Scale[0]), int64(unsafe.Sizeof(float32(0))), o.stream); err != nil {
			return false
		}
	} else {
		kvBytes := int64(kvStride) * 2
		dstOffset := int64(cachePos*kvStride) * 2
		if err := native.MemcpyH2DAsyncAt(cache.kF16, dstOffset, unsafe.Pointer(&o.kTmpU16[0]), kvBytes, o.stream); err != nil {
			return false
		}
	}
	if cache.useQ8V {
		scaleOffset := int64(cachePos) * int64(unsafe.Sizeof(float32(0)))
		kvBytes := int64(kvStride)
		dstOffset := int64(cachePos * kvStride)
		if err := native.MemcpyH2DAsyncAt(cache.vQ8, dstOffset, unsafe.Pointer(&o.vTmpQ8[0]), kvBytes, o.stream); err != nil {
			return false
		}
		if err := native.MemcpyH2DAsyncAt(cache.vQ8Scales, scaleOffset, unsafe.Pointer(&o.vTmpQ8Scale[0]), int64(unsafe.Sizeof(float32(0))), o.stream); err != nil {
			return false
		}
	} else {
		kvBytes := int64(kvStride) * 2
		dstOffset := int64(cachePos*kvStride) * 2
		if err := native.MemcpyH2DAsyncAt(cache.vF16, dstOffset, unsafe.Pointer(&o.vTmpU16[0]), kvBytes, o.stream); err != nil {
			return false
		}
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), qBytes, o.stream); err != nil {
		return false
	}

	if cache.useQ8K || cache.useQ8V {
		if err := native.AttentionInnerMixedCacheF32(
			o.xDev,
			cache.kF16,
			cache.vF16,
			cache.kQ8,
			cache.vQ8,
			cache.kQ8Scales,
			cache.vQ8Scales,
			o.yDev,
			cache.useQ8K,
			cache.useQ8V,
			pos,
			start,
			kvStride,
			headDim,
			nHead,
			kvHeads,
			cacheLen,
			scale,
			o.stream,
		); err != nil {
			return false
		}
	} else {
		if err := native.AttentionInnerF16CacheF32(o.xDev, cache.kF16, cache.vF16, o.yDev, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale, o.stream); err != nil {
			return false
		}
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
	needsF16 := !cache.useQ8K || !cache.useQ8V
	needsQ8 := cache.useQ8K || cache.useQ8V
	if needsF16 {
		if err := o.ensureKVU16Tmp(kvStride); err != nil {
			return false
		}
	}
	if needsQ8 {
		if err := o.ensureKVQ8Tmp(kvStride); err != nil {
			return false
		}
	}

	copy(unsafe.Slice((*float32)(o.xHost.Ptr()), nHead*headDim), q[:nHead*headDim])
	if !cache.useQ8K {
		for i := range kvStride {
			o.kTmpU16[i] = simd.Float32ToFloat16(k[i])
		}
	} else {
		o.kTmpQ8Scale[0] = quantizeQ8(k, o.kTmpQ8[:kvStride])
	}
	if !cache.useQ8V {
		for i := range kvStride {
			o.vTmpU16[i] = simd.Float32ToFloat16(v[i])
		}
	} else {
		o.vTmpQ8Scale[0] = quantizeQ8(v, o.vTmpQ8[:kvStride])
	}

	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	if cache.useQ8K {
		scaleOffset := int64(cachePos) * int64(unsafe.Sizeof(float32(0)))
		kvBytes := int64(kvStride)
		dstOffset := int64(cachePos * kvStride)
		if err := native.MemcpyH2DAsyncAt(cache.kQ8, dstOffset, unsafe.Pointer(&o.kTmpQ8[0]), kvBytes, o.stream); err != nil {
			return false
		}
		if err := native.MemcpyH2DAsyncAt(cache.kQ8Scales, scaleOffset, unsafe.Pointer(&o.kTmpQ8Scale[0]), int64(unsafe.Sizeof(float32(0))), o.stream); err != nil {
			return false
		}
	} else {
		kvBytes := int64(kvStride) * 2
		dstOffset := int64(cachePos*kvStride) * 2
		if err := native.MemcpyH2DAsyncAt(cache.kF16, dstOffset, unsafe.Pointer(&o.kTmpU16[0]), kvBytes, o.stream); err != nil {
			return false
		}
	}
	if cache.useQ8V {
		scaleOffset := int64(cachePos) * int64(unsafe.Sizeof(float32(0)))
		kvBytes := int64(kvStride)
		dstOffset := int64(cachePos * kvStride)
		if err := native.MemcpyH2DAsyncAt(cache.vQ8, dstOffset, unsafe.Pointer(&o.vTmpQ8[0]), kvBytes, o.stream); err != nil {
			return false
		}
		if err := native.MemcpyH2DAsyncAt(cache.vQ8Scales, scaleOffset, unsafe.Pointer(&o.vTmpQ8Scale[0]), int64(unsafe.Sizeof(float32(0))), o.stream); err != nil {
			return false
		}
	} else {
		kvBytes := int64(kvStride) * 2
		dstOffset := int64(cachePos*kvStride) * 2
		if err := native.MemcpyH2DAsyncAt(cache.vF16, dstOffset, unsafe.Pointer(&o.vTmpU16[0]), kvBytes, o.stream); err != nil {
			return false
		}
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), qBytes, o.stream); err != nil {
		return false
	}

	if cache.useQ8K || cache.useQ8V {
		if err := native.AttentionInnerMixedCacheF32(
			o.xDev,
			cache.kF16,
			cache.vF16,
			cache.kQ8,
			cache.vQ8,
			cache.kQ8Scales,
			cache.vQ8Scales,
			o.yDev,
			cache.useQ8K,
			cache.useQ8V,
			pos,
			start,
			kvStride,
			headDim,
			nHead,
			kvHeads,
			cacheLen,
			scale,
			o.stream,
		); err != nil {
			return false
		}
	} else {
		if err := native.AttentionInnerF16CacheF32(o.xDev, cache.kF16, cache.vF16, o.yDev, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale, o.stream); err != nil {
			return false
		}
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

	if useQuantKernel() && (shouldPreferQuantWeights(layer.Wo, currentCUDAWeightMode()) || (layer.Wo.Quant != nil && layer.Wo.Quant.ValidFor(layer.Wo))) {
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

	// Keep projection result on device for residual add fast path.
	// Runtime can force host visibility via FlushBlockResult when needed.
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
		o.lastResultDev = o.aDev
	} else {
		o.lastResultDev = o.zDev
	}
	if projDim > 0 {
		o.lastResultHostPtr = uintptr(unsafe.Pointer(&projOut[0]))
		o.lastResultLen = projDim
		o.lastResultValid = true
	}
	runtime.KeepAlive(projOut)
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

func (o *Ops) StoreKV(_ int, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32) {
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
	} else if kDstQ8 != nil {
		storeQ8(k, kDstQ8[offset:offset+kvStride], kDstQ8S, pos)
	}
	if vDst != nil {
		copy(vDst[offset:], v)
	} else if vDst16 != nil {
		for i, vv := range v {
			vDst16[offset+i] = simd.Float32ToFloat16(vv)
		}
	} else if vDstQ8 != nil {
		storeQ8(v, vDstQ8[offset:offset+kvStride], vDstQ8S, pos)
	}
}

func storeQ8(src []float32, dst []int8, scales []float32, pos int) {
	var maxAbs float32
	for _, v := range src {
		if v < 0 {
			if -v > maxAbs {
				maxAbs = -v
			}
		} else if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		scales[pos] = 0
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	scale := maxAbs / 127.0
	scales[pos] = scale
	invScale := 127.0 / maxAbs
	for i, v := range src {
		q := int32(v*invScale + 0.5)
		if v < 0 {
			q = int32(v*invScale - 0.5)
		}
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q)
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
	// Check cache first (2-3% TPS improvement by avoiding redundant GPU dequantization)
	if cached, ok := o.f16Dequant[w]; ok {
		runtime.KeepAlive(w)
		return cached, nil
	}

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
		if cudaTraceSyncEnabled() {
			if err := o.stream.Synchronize(); err != nil {
				_ = out.Free()
				return native.DeviceBuffer{}, err
			}
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
		if cudaTraceSyncEnabled() {
			if err := o.stream.Synchronize(); err != nil {
				_ = out.Free()
				return native.DeviceBuffer{}, err
			}
		}
	default:
		_ = out.Free()
		return native.DeviceBuffer{}, fmt.Errorf("gpu dequant unsupported dtype: %s", dtypeString(w.DType))
	}

	// Cache result before return
	o.f16Dequant[w] = out
	runtime.KeepAlive(w)
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
	if _, ok := os.LookupEnv("MANTLE_CUDA_QUANT_KERNEL"); ok {
		return envEnabled("MANTLE_CUDA_QUANT_KERNEL")
	}
	return currentCUDAWeightMode() != cudaWeightModeDequant
}

func useK4RawKernel() bool {
	// K4 raw CUDA kernels are currently opt-in; default to cached int8 path for stability.
	return envEnabled("MANTLE_CUDA_K4_RAW")
}

func useLegacyStreamSync() bool {
	return envEnabled("MANTLE_CUDA_LEGACY_SYNC")
}

func useCUDAGraphs() bool {
	v, ok := os.LookupEnv("MANTLE_CUDA_GRAPHS")
	if !ok {
		return true
	}
	if b, err := strconv.ParseBool(v); err == nil {
		return b
	}
	switch v {
	case "0", "off", "OFF", "no", "NO", "n", "N":
		return false
	default:
		return true
	}
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

func cudaTraceEnabled() bool {
	return os.Getenv("MANTLE_CUDA_TRACE") != ""
}

func cudaTraceSyncEnabled() bool {
	return envEnabled("MANTLE_CUDA_TRACE_SYNC")
}

func cudaWeightModeString(mode cudaWeightMode) string {
	switch mode {
	case cudaWeightModeQuant:
		return "quant"
	case cudaWeightModeDequant:
		return "dequant"
	default:
		return "auto"
	}
}

func currentCUDAWeightMode() cudaWeightMode {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("MANTLE_CUDA_WEIGHT_MODE")))
	switch v {
	case "quant":
		return cudaWeightModeQuant
	case "dequant":
		return cudaWeightModeDequant
	default:
		return cudaWeightModeAuto
	}
}

func shouldPreferQuantWeights(mat *simd.Mat, mode cudaWeightMode) bool {
	if mat == nil {
		return false
	}
	if !mcf.DTypeRequiresAligned64(mat.DType) || len(mat.Raw) == 0 {
		return false
	}
	switch mode {
	case cudaWeightModeQuant:
		return true
	case cudaWeightModeDequant:
		return false
	default:
		// Memory-safe default: keep quantized payloads on-device unless explicitly overridden.
		return true
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
	if w.DType == mcf.DTypeK4 && len(w.Raw) > 0 && useK4RawKernel() {
		if cudaTraceEnabled() {
			fmt.Fprintf(os.Stderr, "CUDA: attempting K4 raw kernel for matrix %dx%d (raw size=%d)\n", w.R, w.C, len(w.Raw))
		}
		info, err := o.ensureQuantMatK4Raw(w)
		if err != nil {
			if cudaTraceEnabled() {
				fmt.Fprintf(os.Stderr, "CUDA: K4 raw kernel failed: %v\n", err)
			}
			return deviceQuantMat{}, err
		}
		if cudaTraceEnabled() {
			fmt.Fprintf(os.Stderr, "CUDA: K4 raw kernel setup successful\n")
		}
		o.qweights[w] = info
		runtime.KeepAlive(w)
		return info, nil
	}
	if w.Quant == nil && mcf.DTypeRequiresAligned64(w.DType) && len(w.Raw) > 0 {
		qc, err := simd.BuildQuantCache(w)
		if err != nil {
			return deviceQuantMat{}, fmt.Errorf("quant cache build failed for %s matrix [%d x %d]: %w", dtypeString(w.DType), w.R, w.C, err)
		}
		w.Quant = qc
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

func (o *Ops) ensureNormDeviceVec(bytes int) error {
	if bytes > o.normDevBytes {
		if err := o.normDev.Free(); err != nil {
			return err
		}
		buf, err := native.AllocDevice(int64(bytes))
		if err != nil {
			return err
		}
		o.normDev = buf
		o.normDevBytes = bytes
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

func (o *Ops) ensureKVQ8Tmp(kvStride int) error {
	if kvStride <= 0 {
		return fmt.Errorf("invalid kv stride")
	}
	if len(o.kTmpQ8) < kvStride {
		o.kTmpQ8 = make([]int8, kvStride)
	}
	if len(o.vTmpQ8) < kvStride {
		o.vTmpQ8 = make([]int8, kvStride)
	}
	if len(o.kTmpQ8Scale) < 1 {
		o.kTmpQ8Scale = make([]float32, 1)
	}
	if len(o.vTmpQ8Scale) < 1 {
		o.vTmpQ8Scale = make([]float32, 1)
	}
	return nil
}

func quantizeQ8(src []float32, dst []int8) float32 {
	var maxAbs float32
	for _, v := range src {
		if v < 0 {
			if -v > maxAbs {
				maxAbs = -v
			}
		} else if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return 0
	}
	scale := maxAbs / 127.0
	invScale := 127.0 / maxAbs
	for i, v := range src {
		q := int32(v*invScale + 0.5)
		if v < 0 {
			q = int32(v*invScale - 0.5)
		}
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q)
	}
	return scale
}

func (o *Ops) ensureAttnCache(layer *simd.Layer, kvStride, cacheLen int) (deviceAttnCache, error) {
	if cache, ok := o.attnCaches[layer]; ok {
		return cache, nil
	}
	if kvStride <= 0 || cacheLen <= 0 {
		return deviceAttnCache{}, fmt.Errorf("invalid attention cache dimensions")
	}
	if layer == nil {
		return deviceAttnCache{}, fmt.Errorf("nil attention layer")
	}
	totalElems, ok := mulInt(kvStride, cacheLen)
	if !ok {
		return deviceAttnCache{}, fmt.Errorf("attention cache too large")
	}

	useQ8K := layer.AttnCache.KQ8 != nil
	useQ8V := layer.AttnCache.VQ8 != nil

	totalF16Bytes, ok := mulInt(totalElems, 2)
	if !ok {
		return deviceAttnCache{}, fmt.Errorf("attention cache too large")
	}
	totalQ8Bytes := totalElems
	scaleBytes, ok := mulInt(cacheLen, int(unsafe.Sizeof(float32(0))))
	if !ok {
		return deviceAttnCache{}, fmt.Errorf("attention cache too large")
	}

	cache := deviceAttnCache{
		useQ8K:   useQ8K,
		useQ8V:   useQ8V,
		kvStride: kvStride,
		cacheLen: cacheLen,
	}

	alloc := func(dst *native.DeviceBuffer, bytes int) error {
		buf, err := native.AllocDevice(int64(bytes))
		if err != nil {
			return err
		}
		*dst = buf
		return nil
	}
	cleanup := func() {
		_ = cache.kF16.Free()
		_ = cache.vF16.Free()
		_ = cache.kQ8.Free()
		_ = cache.vQ8.Free()
		_ = cache.kQ8Scales.Free()
		_ = cache.vQ8Scales.Free()
	}

	if useQ8K {
		if err := alloc(&cache.kQ8, totalQ8Bytes); err != nil {
			cleanup()
			return deviceAttnCache{}, err
		}
		if err := alloc(&cache.kQ8Scales, scaleBytes); err != nil {
			cleanup()
			return deviceAttnCache{}, err
		}
	} else {
		if err := alloc(&cache.kF16, totalF16Bytes); err != nil {
			cleanup()
			return deviceAttnCache{}, err
		}
	}

	if useQ8V {
		if err := alloc(&cache.vQ8, totalQ8Bytes); err != nil {
			cleanup()
			return deviceAttnCache{}, err
		}
		if err := alloc(&cache.vQ8Scales, scaleBytes); err != nil {
			cleanup()
			return deviceAttnCache{}, err
		}
	} else {
		if err := alloc(&cache.vF16, totalF16Bytes); err != nil {
			cleanup()
			return deviceAttnCache{}, err
		}
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

func estimateModelWeightBytes(m *simd.Instance, mode cudaWeightMode) int64 {
	if m == nil {
		return 0
	}
	var total int64
	addMat := func(mat *simd.Mat) {
		total += estimateMatBytes(mat, mode)
	}
	addNorm := func(v []float32) {
		total += int64(len(v)) * 4
	}

	addMat(m.Embeddings)
	addMat(m.Output)
	addNorm(m.OutputNorm)
	for i := range m.Layers {
		layer := &m.Layers[i]
		addNorm(layer.AttnNorm)
		addNorm(layer.PostAttnNorm)
		addNorm(layer.FfnNorm)
		addNorm(layer.PostFfnNorm)
		addNorm(layer.AttnQNorm)
		addNorm(layer.AttnKNorm)
		addMat(layer.Wq)
		addMat(layer.Wk)
		addMat(layer.Wv)
		addMat(layer.Wo)
		addMat(layer.AttnGate)
		addMat(layer.ShortConvKernel)
		addMat(layer.ShortConvInProj)
		addMat(layer.ShortConvOutProj)
		addMat(layer.FfnUp)
		addMat(layer.FfnGate)
		addMat(layer.FfnDown)
		if layer.MoE != nil {
			addMat(layer.MoE.Router)
			addMat(layer.MoE.Shared.Up)
			addMat(layer.MoE.Shared.Gate)
			addMat(layer.MoE.Shared.Down)
			for j := range layer.MoE.Experts {
				addMat(layer.MoE.Experts[j].Up)
				addMat(layer.MoE.Experts[j].Gate)
				addMat(layer.MoE.Experts[j].Down)
			}
		}
	}
	return total
}

func estimateMatBytes(mat *simd.Mat, mode cudaWeightMode) int64 {
	if mat == nil {
		return 0
	}
	if shouldPreferQuantWeights(mat, mode) {
		if len(mat.Raw) > 0 {
			return int64(len(mat.Raw))
		}
	}
	// Dequantized upload target is fp16 for quant payloads.
	if mcf.DTypeRequiresAligned64(mat.DType) && len(mat.Raw) > 0 {
		return int64(mat.R) * int64(mat.C) * 2
	}
	if len(mat.Raw) > 0 {
		return int64(len(mat.Raw))
	}
	return int64(len(mat.Data)) * 4
}

func dtypeString(dt mcf.TensorDType) string {
	return fmt.Sprintf("0x%04x", uint16(dt))
}
