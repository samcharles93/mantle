//go:build cuda

package native

/*
#cgo LDFLAGS: -L${SRCDIR}/build -lmantle_cuda_kernels -lcudart -lcublas -lstdc++
// Minimal CUDA runtime forward declarations to avoid requiring headers at compile time.
// Linker will still require libcudart when building with the cuda tag.
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);
extern cudaError_t cudaGetDeviceCount(int* count);
extern cudaError_t cudaStreamCreate(cudaStream_t* stream);
extern cudaError_t cudaStreamDestroy(cudaStream_t stream);
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);
extern cudaError_t cudaMalloc(void** ptr, unsigned long long size);
extern cudaError_t cudaMallocManaged(void** ptr, unsigned long long size, unsigned int flags);
extern cudaError_t cudaFree(void* ptr);
extern cudaError_t cudaMemcpy(void* dst, const void* src, unsigned long long size, int kind);
extern cudaError_t cudaMemcpyAsync(void* dst, const void* src, unsigned long long size, int kind, cudaStream_t stream);
extern cudaError_t cudaMallocHost(void** ptr, unsigned long long size);
extern cudaError_t cudaFreeHost(void* ptr);
extern cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device);

#define MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE 1
#define MANTLE_CUDA_MEMCPY_DEVICE_TO_HOST 2
#define MANTLE_CUDA_MEMCPY_DEFAULT 4

typedef struct cublasContext* cublasHandle_t;
typedef int cublasStatus_t;

extern cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
extern cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
extern cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream);
extern cublasStatus_t cublasGemmEx(
	cublasHandle_t handle,
	int transa,
	int transb,
	int m,
	int n,
	int k,
	const void* alpha,
	const void* A,
	int Atype,
	int lda,
	const void* B,
	int Btype,
	int ldb,
	const void* beta,
	void* C,
	int Ctype,
	int ldc,
	int computeType,
	int algo);

extern cublasStatus_t cublasSgemv_v2(
	cublasHandle_t handle,
	int trans,
	int m,
	int n,
	const float* alpha,
	const float* A,
	int lda,
	const float* x,
	int incx,
	const float* beta,
	float* y,
	int incy);
extern cublasStatus_t cublasSdot_v2(
	cublasHandle_t handle,
	int n,
	const float* x,
	int incx,
	const float* y,
	int incy,
	float* result);
extern cublasStatus_t cublasScopy_v2(
	cublasHandle_t handle,
	int n,
	const float* x,
	int incx,
	float* y,
	int incy);
extern cublasStatus_t cublasSscal_v2(
	cublasHandle_t handle,
	int n,
	const float* alpha,
	float* x,
	int incx);
extern cublasStatus_t cublasSdgmm(
	cublasHandle_t handle,
	int mode,
	int m,
	int n,
	const float* A,
	int lda,
	const float* x,
	int incx,
	float* C,
	int ldc);
extern int mantleCudaSoftmaxRowsF32(float* data, int rows, int cols, cudaStream_t stream);
extern int mantleCudaQuantMatVecInt8BlocksF32(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecQ4F32(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecK4F32(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaDequantizeQ4ToF16(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaDequantizeK4ToF16(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaSiluMulF32(
	const float* gate,
	const float* up,
	float* out,
	int n,
	cudaStream_t stream);
extern int mantleCudaConvertF32ToF16(
	const float* in,
	unsigned short* out,
	int n,
	cudaStream_t stream);
extern int mantleCudaAttentionInnerF16CacheF32(
	const float* q,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int pos,
	int start,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	cudaStream_t stream);

static const char* mantleCudaGetErrorString(cudaError_t err) {
	return cudaGetErrorString(err);
}

static int mantleCudaGetDeviceCount(int* out) {
	cudaError_t err = cudaGetDeviceCount(out);
	return (int)err;
}

static int mantleCudaStreamCreate(cudaStream_t* out) {
	cudaError_t err = cudaStreamCreate(out);
	return (int)err;
}

static int mantleCudaStreamDestroy(cudaStream_t stream) {
	cudaError_t err = cudaStreamDestroy(stream);
	return (int)err;
}

static int mantleCudaStreamSynchronize(cudaStream_t stream) {
	cudaError_t err = cudaStreamSynchronize(stream);
	return (int)err;
}

static int mantleCudaMalloc(void** ptr, unsigned long long size) {
	cudaError_t err = cudaMalloc(ptr, size);
	return (int)err;
}

static int mantleCudaMallocManaged(void** ptr, unsigned long long size) {
	// 1 == cudaMemAttachGlobal
	cudaError_t err = cudaMallocManaged(ptr, size, 1u);
	return (int)err;
}

static int mantleCudaFree(void* ptr) {
	cudaError_t err = cudaFree(ptr);
	return (int)err;
}

static int mantleCudaMemcpy(void* dst, const void* src, unsigned long long size, int kind) {
	cudaError_t err = cudaMemcpy(dst, src, size, kind);
	return (int)err;
}

static int mantleCudaMemcpyAsync(void* dst, const void* src, unsigned long long size, int kind, cudaStream_t stream) {
	cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, stream);
	return (int)err;
}

static int mantleCudaMallocHost(void** ptr, unsigned long long size) {
	cudaError_t err = cudaMallocHost(ptr, size);
	return (int)err;
}

static int mantleCudaFreeHost(void* ptr) {
	cudaError_t err = cudaFreeHost(ptr);
	return (int)err;
}

static int mantleCudaDeviceGetAttribute(int* value, int attr, int device) {
	cudaError_t err = cudaDeviceGetAttribute(value, attr, device);
	return (int)err;
}

static int mantleCublasCreate(cublasHandle_t* out) {
	cublasStatus_t st = cublasCreate_v2(out);
	return (int)st;
}

static int mantleCublasDestroy(cublasHandle_t handle) {
	cublasStatus_t st = cublasDestroy_v2(handle);
	return (int)st;
}

static int mantleCublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
	cublasStatus_t st = cublasSetStream_v2(handle, stream);
	return (int)st;
}

static int mantleCublasGemmEx(
	cublasHandle_t handle,
	int transa,
	int transb,
	int m,
	int n,
	int k,
	const void* alpha,
	const void* A,
	int Atype,
	int lda,
	const void* B,
	int Btype,
	int ldb,
	const void* beta,
	void* C,
	int Ctype,
	int ldc,
	int computeType,
	int algo) {
	cublasStatus_t st = cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
	return (int)st;
}

static int mantleCublasSgemv(
	cublasHandle_t handle,
	int trans,
	int m,
	int n,
	const float* alpha,
	const float* A,
	int lda,
	const float* x,
	int incx,
	const float* beta,
	float* y,
	int incy) {
	cublasStatus_t st = cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	return (int)st;
}

static int mantleCublasSdot(
	cublasHandle_t handle,
	int n,
	const float* x,
	int incx,
	const float* y,
	int incy,
	float* result) {
	cublasStatus_t st = cublasSdot_v2(handle, n, x, incx, y, incy, result);
	return (int)st;
}

static int mantleCublasScopy(
	cublasHandle_t handle,
	int n,
	const float* x,
	int incx,
	float* y,
	int incy) {
	cublasStatus_t st = cublasScopy_v2(handle, n, x, incx, y, incy);
	return (int)st;
}

static int mantleCublasSscal(
	cublasHandle_t handle,
	int n,
	const float* alpha,
	float* x,
	int incx) {
	cublasStatus_t st = cublasSscal_v2(handle, n, alpha, x, incx);
	return (int)st;
}

static int mantleCublasSdgmm(
	cublasHandle_t handle,
	int mode,
	int m,
	int n,
	const float* A,
	int lda,
	const float* x,
	int incx,
	float* C,
	int ldc) {
	cublasStatus_t st = cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
	return (int)st;
}

static int mantleCudaSoftmaxRowsF32Wrapper(float* data, int rows, int cols, cudaStream_t stream) {
	return mantleCudaSoftmaxRowsF32(data, rows, cols, stream);
}

static int mantleCudaQuantMatVecInt8BlocksF32Wrapper(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecInt8BlocksF32(q, scales, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaQuantMatVecQ4F32Wrapper(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecQ4F32(qData, scalesF16, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaQuantMatVecK4F32Wrapper(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecK4F32(qData, superScalesF16, subScales, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaDequantizeQ4ToF16Wrapper(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaDequantizeQ4ToF16(qData, scalesF16, outF16, rows, blocksPerRow, cols, stream);
}

static int mantleCudaDequantizeK4ToF16Wrapper(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaDequantizeK4ToF16(qData, superScalesF16, subScales, outF16, rows, blocksPerRow, cols, stream);
}

static int mantleCudaSiluMulF32Wrapper(
	const float* gate,
	const float* up,
	float* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaSiluMulF32(gate, up, out, n, stream);
}

static int mantleCudaConvertF32ToF16Wrapper(
	const float* in,
	unsigned short* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaConvertF32ToF16(in, out, n, stream);
}

static int mantleCudaAttentionInnerF16CacheF32Wrapper(
	const float* q,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int pos,
	int start,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	cudaStream_t stream) {
	return mantleCudaAttentionInnerF16CacheF32(q, cacheK, cacheV, out, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale, stream);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"unsafe"
)

type Stream struct {
	ptr C.cudaStream_t
}

type BlasHandle struct {
	ptr C.cublasHandle_t
}

type DeviceBuffer struct {
	ptr     unsafe.Pointer
	managed bool
}

type HostBuffer struct {
	ptr unsafe.Pointer
}

var managedFallback atomic.Bool

// PerfCounters holds CUDA backend performance statistics.
type PerfCounters struct {
	MatVecCalls   int64
	RMSNormCalls  int64
	StoreKVCalls  int64
	StreamSyncs   int64
	H2DBytes      int64
	D2HBytes      int64
	ManagedAllocs int64
	ManagedBytes  int64
	DeviceAllocs  int64
	DeviceBytes   int64
}

var globalPerfCounters PerfCounters
var perfEnabledOnce sync.Once
var perfEnabledCached bool

// GetPerfCounters returns a copy of current counters.
func GetPerfCounters() PerfCounters {
	return PerfCounters{
		MatVecCalls:   globalPerfCounters.MatVecCalls,
		RMSNormCalls:  globalPerfCounters.RMSNormCalls,
		StoreKVCalls:  globalPerfCounters.StoreKVCalls,
		StreamSyncs:   globalPerfCounters.StreamSyncs,
		H2DBytes:      globalPerfCounters.H2DBytes,
		D2HBytes:      globalPerfCounters.D2HBytes,
		ManagedAllocs: globalPerfCounters.ManagedAllocs,
		ManagedBytes:  globalPerfCounters.ManagedBytes,
		DeviceAllocs:  globalPerfCounters.DeviceAllocs,
		DeviceBytes:   globalPerfCounters.DeviceBytes,
	}
}

// ResetPerfCounters resets all counters to zero.
func ResetPerfCounters() {
	globalPerfCounters = PerfCounters{}
}

// perfEnabled returns true if MANTLE_CUDA_TRACE env var is set.
func perfEnabled() bool {
	perfEnabledOnce.Do(func() {
		perfEnabledCached = os.Getenv("MANTLE_CUDA_TRACE") != ""
	})
	return perfEnabledCached
}

func recordMatVec() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.MatVecCalls, 1)
	}
}

func recordRMSNorm() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.RMSNormCalls, 1)
	}
}

func recordStoreKV() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.StoreKVCalls, 1)
	}
}

func recordStreamSync() {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.StreamSyncs, 1)
	}
}

func recordH2D(bytes int64) {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.H2DBytes, bytes)
	}
}

func recordD2H(bytes int64) {
	if perfEnabled() {
		atomic.AddInt64(&globalPerfCounters.D2HBytes, bytes)
	}
}

func recordDeviceAlloc(bytes int64, managed bool) {
	if !perfEnabled() {
		return
	}
	if managed {
		atomic.AddInt64(&globalPerfCounters.ManagedAllocs, 1)
		atomic.AddInt64(&globalPerfCounters.ManagedBytes, bytes)
	} else {
		atomic.AddInt64(&globalPerfCounters.DeviceAllocs, 1)
		atomic.AddInt64(&globalPerfCounters.DeviceBytes, bytes)
	}
}

// RecordMatVec records a MatVec operation.
func RecordMatVec() { recordMatVec() }

// RecordRMSNorm records an RMSNorm operation.
func RecordRMSNorm() { recordRMSNorm() }

// RecordStoreKV records a StoreKV operation.
func RecordStoreKV() { recordStoreKV() }

type CUDAError struct {
	Code int
	Msg  string
}

func (e *CUDAError) Error() string {
	return fmt.Sprintf("cuda runtime error %d: %s", e.Code, e.Msg)
}

func IsOOM(err error) bool {
	var ce *CUDAError
	return errors.As(err, &ce) && ce.Code == 2
}

func DeviceCount() (int, error) {
	var count C.int
	if err := cudaErr(C.mantleCudaGetDeviceCount(&count)); err != nil {
		return 0, err
	}
	return int(count), nil
}

type DeviceAttribute int

const (
	DevAttrConcurrentManagedAccess                DeviceAttribute = 89  // cudaDevAttrConcurrentManagedAccess
	DevAttrPageableMemoryAccess                   DeviceAttribute = 88  // cudaDevAttrPageableMemoryAccess
	DevAttrPageableMemoryAccessUsesHostPageTables DeviceAttribute = 100 // cudaDevAttrPageableMemoryAccessUsesHostPageTables
)

func DeviceGetAttribute(attr DeviceAttribute, device int) (int, error) {
	var val C.int
	if err := cudaErr(C.mantleCudaDeviceGetAttribute(&val, C.int(attr), C.int(device))); err != nil {
		return 0, err
	}
	return int(val), nil
}

func NewStream() (Stream, error) {
	var stream C.cudaStream_t
	if err := cudaErr(C.mantleCudaStreamCreate(&stream)); err != nil {
		return Stream{}, err
	}
	return Stream{ptr: stream}, nil
}

func (s Stream) Destroy() error {
	if s.ptr == nil {
		return nil
	}
	return cudaErr(C.mantleCudaStreamDestroy(s.ptr))
}

func (s Stream) Ptr() C.cudaStream_t {
	return s.ptr
}

func (s Stream) Synchronize() error {
	if s.ptr == nil {
		return nil
	}
	recordStreamSync()
	return cudaErr(C.mantleCudaStreamSynchronize(s.ptr))
}

func AllocDevice(bytes int64) (DeviceBuffer, error) {
	if bytes <= 0 {
		return DeviceBuffer{}, fmt.Errorf("device alloc size must be > 0")
	}
	var ptr unsafe.Pointer
	if err := cudaErr(C.mantleCudaMalloc((*unsafe.Pointer)(&ptr), C.ulonglong(bytes))); err != nil {
		// Capacity path: fallback to Unified Memory when device allocation OOMs.
		if IsOOM(err) {
			if managedErr := cudaErr(C.mantleCudaMallocManaged((*unsafe.Pointer)(&ptr), C.ulonglong(bytes))); managedErr == nil {
				managedFallback.Store(true)
				recordDeviceAlloc(bytes, true)
				return DeviceBuffer{ptr: ptr, managed: true}, nil
			} else {
				return DeviceBuffer{}, fmt.Errorf("cuda alloc failed (%d bytes): device OOM and managed alloc failed: %w", bytes, managedErr)
			}
		}
		return DeviceBuffer{}, err
	}
	recordDeviceAlloc(bytes, false)
	return DeviceBuffer{ptr: ptr, managed: false}, nil
}

func ResetManagedFallbackFlag() {
	managedFallback.Store(false)
}

func ManagedFallbackUsed() bool {
	return managedFallback.Load()
}

func (b DeviceBuffer) Free() error {
	if b.ptr == nil {
		return nil
	}
	return cudaErr(C.mantleCudaFree(b.ptr))
}

func (b DeviceBuffer) Ptr() unsafe.Pointer {
	return b.ptr
}

func (b DeviceBuffer) Managed() bool {
	return b.managed
}

func AllocHostPinned(bytes int64) (HostBuffer, error) {
	if bytes <= 0 {
		return HostBuffer{}, fmt.Errorf("host alloc size must be > 0")
	}
	var ptr unsafe.Pointer
	if err := cudaErr(C.mantleCudaMallocHost((*unsafe.Pointer)(&ptr), C.ulonglong(bytes))); err != nil {
		return HostBuffer{}, err
	}
	return HostBuffer{ptr: ptr}, nil
}

func (b HostBuffer) Free() error {
	if b.ptr == nil {
		return nil
	}
	return cudaErr(C.mantleCudaFreeHost(b.ptr))
}

func (b HostBuffer) Ptr() unsafe.Pointer {
	return b.ptr
}

func MemcpyH2DAsync(dst DeviceBuffer, src unsafe.Pointer, bytes int64, stream Stream) error {
	if bytes <= 0 {
		return nil
	}
	recordH2D(bytes)
	kind := C.int(C.MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE)
	if dst.managed {
		kind = C.int(C.MANTLE_CUDA_MEMCPY_DEFAULT)
	}
	return cudaErr(C.mantleCudaMemcpyAsync(dst.ptr, src, C.ulonglong(bytes), kind, stream.ptr))
}

func MemcpyH2DAsyncAt(dst DeviceBuffer, dstOffset int64, src unsafe.Pointer, bytes int64, stream Stream) error {
	if bytes <= 0 {
		return nil
	}
	recordH2D(bytes)
	ptr := unsafe.Add(dst.ptr, int(dstOffset))
	kind := C.int(C.MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE)
	if dst.managed {
		kind = C.int(C.MANTLE_CUDA_MEMCPY_DEFAULT)
	}
	return cudaErr(C.mantleCudaMemcpyAsync(ptr, src, C.ulonglong(bytes), kind, stream.ptr))
}

func MemcpyD2HAsync(dst unsafe.Pointer, src DeviceBuffer, bytes int64, stream Stream) error {
	if bytes <= 0 {
		return nil
	}
	recordD2H(bytes)
	kind := C.int(C.MANTLE_CUDA_MEMCPY_DEVICE_TO_HOST)
	if src.managed {
		kind = C.int(C.MANTLE_CUDA_MEMCPY_DEFAULT)
	}
	return cudaErr(C.mantleCudaMemcpyAsync(dst, src.ptr, C.ulonglong(bytes), kind, stream.ptr))
}

func MemcpyH2D(dst DeviceBuffer, src unsafe.Pointer, bytes int64) error {
	if bytes <= 0 {
		return nil
	}
	recordH2D(bytes)
	kind := C.int(C.MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE)
	if dst.managed {
		kind = C.int(C.MANTLE_CUDA_MEMCPY_DEFAULT)
	}
	return cudaErr(C.mantleCudaMemcpy(dst.ptr, src, C.ulonglong(bytes), kind))
}

func MemcpyD2H(dst unsafe.Pointer, src DeviceBuffer, bytes int64) error {
	if bytes <= 0 {
		return nil
	}
	recordD2H(bytes)
	kind := C.int(C.MANTLE_CUDA_MEMCPY_DEVICE_TO_HOST)
	if src.managed {
		kind = C.int(C.MANTLE_CUDA_MEMCPY_DEFAULT)
	}
	return cudaErr(C.mantleCudaMemcpy(dst, src.ptr, C.ulonglong(bytes), kind))
}

func NewBlasHandle(stream Stream) (BlasHandle, error) {
	var handle C.cublasHandle_t
	if err := cublasErr(C.mantleCublasCreate(&handle)); err != nil {
		return BlasHandle{}, err
	}
	if err := cublasErr(C.mantleCublasSetStream(handle, stream.ptr)); err != nil {
		_ = cublasErr(C.mantleCublasDestroy(handle))
		return BlasHandle{}, err
	}
	return BlasHandle{ptr: handle}, nil
}

func (h BlasHandle) Destroy() error {
	if h.ptr == nil {
		return nil
	}
	return cublasErr(C.mantleCublasDestroy(h.ptr))
}

func (h BlasHandle) Ptr() C.cublasHandle_t {
	return h.ptr
}

type BlasDataType int

const (
	BlasF16  BlasDataType = 2  // CUDA_R_16F
	BlasBF16 BlasDataType = 14 // CUDA_R_16BF
	BlasF32  BlasDataType = 0  // CUDA_R_32F
)

type BlasComputeType int

const (
	BlasComputeF32 BlasComputeType = 68 // CUBLAS_COMPUTE_32F
)

type BlasOp int

const (
	BlasOpN BlasOp = 0 // CUBLAS_OP_N
	BlasOpT BlasOp = 1 // CUBLAS_OP_T
)

type BlasGemmAlgo int

const (
	BlasGemmDefault BlasGemmAlgo = -1 // CUBLAS_GEMM_DEFAULT
)

type BlasSideMode int

const (
	BlasSideLeft  BlasSideMode = 0 // CUBLAS_SIDE_LEFT
	BlasSideRight BlasSideMode = 1 // CUBLAS_SIDE_RIGHT
)

func GemmEx(handle BlasHandle, transA, transB BlasOp, m, n, k int, alpha float32, a DeviceBuffer, aType BlasDataType, lda int, b DeviceBuffer, bType BlasDataType, ldb int, beta float32, c DeviceBuffer, cType BlasDataType, ldc int, compute BlasComputeType, algo BlasGemmAlgo) error {
	alphaPtr := unsafe.Pointer(&alpha)
	betaPtr := unsafe.Pointer(&beta)
	return cublasErr(C.mantleCublasGemmEx(
		handle.ptr,
		C.int(transA),
		C.int(transB),
		C.int(m),
		C.int(n),
		C.int(k),
		alphaPtr,
		a.ptr,
		C.int(aType),
		C.int(lda),
		b.ptr,
		C.int(bType),
		C.int(ldb),
		betaPtr,
		c.ptr,
		C.int(cType),
		C.int(ldc),
		C.int(compute),
		C.int(algo),
	))
}

func GemvF32(handle BlasHandle, trans BlasOp, m, n int, alpha float32, a DeviceBuffer, lda int, x DeviceBuffer, incx int, beta float32, y DeviceBuffer, incy int) error {
	alphaPtr := unsafe.Pointer(&alpha)
	betaPtr := unsafe.Pointer(&beta)
	return cublasErr(C.mantleCublasSgemv(
		handle.ptr,
		C.int(trans),
		C.int(m),
		C.int(n),
		(*C.float)(alphaPtr),
		(*C.float)(a.ptr),
		C.int(lda),
		(*C.float)(x.ptr),
		C.int(incx),
		(*C.float)(betaPtr),
		(*C.float)(y.ptr),
		C.int(incy),
	))
}

func DotF32(handle BlasHandle, n int, x DeviceBuffer, incx int, y DeviceBuffer, incy int) (float32, error) {
	var out C.float
	err := cublasErr(C.mantleCublasSdot(
		handle.ptr,
		C.int(n),
		(*C.float)(x.ptr),
		C.int(incx),
		(*C.float)(y.ptr),
		C.int(incy),
		&out,
	))
	return float32(out), err
}

func CopyF32(handle BlasHandle, n int, src DeviceBuffer, incSrc int, dst DeviceBuffer, incDst int) error {
	return cublasErr(C.mantleCublasScopy(
		handle.ptr,
		C.int(n),
		(*C.float)(src.ptr),
		C.int(incSrc),
		(*C.float)(dst.ptr),
		C.int(incDst),
	))
}

func ScalF32(handle BlasHandle, n int, alpha float32, x DeviceBuffer, incx int) error {
	alphaPtr := unsafe.Pointer(&alpha)
	return cublasErr(C.mantleCublasSscal(
		handle.ptr,
		C.int(n),
		(*C.float)(alphaPtr),
		(*C.float)(x.ptr),
		C.int(incx),
	))
}

func DgmmF32(handle BlasHandle, mode BlasSideMode, m, n int, a DeviceBuffer, lda int, x DeviceBuffer, incx int, c DeviceBuffer, ldc int) error {
	return cublasErr(C.mantleCublasSdgmm(
		handle.ptr,
		C.int(mode),
		C.int(m),
		C.int(n),
		(*C.float)(a.ptr),
		C.int(lda),
		(*C.float)(x.ptr),
		C.int(incx),
		(*C.float)(c.ptr),
		C.int(ldc),
	))
}

func SoftmaxRowsF32(buf DeviceBuffer, rows, cols int, stream Stream) error {
	if buf.ptr == nil {
		return fmt.Errorf("softmax buffer is nil")
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("softmax rows/cols must be > 0")
	}
	return cudaErr(C.mantleCudaSoftmaxRowsF32Wrapper((*C.float)(buf.ptr), C.int(rows), C.int(cols), stream.ptr))
}

func QuantMatVecInt8BlocksF32(q, scales, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if q.ptr == nil || scales.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("quant matvec buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("quant matvec rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecInt8BlocksF32Wrapper(
		(*C.schar)(q.ptr),
		(*C.float)(scales.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func QuantMatVecQ4F32(qData, scalesF16, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || scalesF16.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("q4 quant matvec buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("q4 quant matvec rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecQ4F32Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(scalesF16.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func QuantMatVecK4F32(qData, superScalesF16, subScales, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || superScalesF16.ptr == nil || subScales.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("k4 quant matvec buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("k4 quant matvec rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecK4F32Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(superScalesF16.ptr),
		(*C.uchar)(subScales.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func DequantizeQ4ToF16(qData, scalesF16, outF16 DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || scalesF16.ptr == nil || outF16.ptr == nil {
		return fmt.Errorf("q4 dequant buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("q4 dequant rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaDequantizeQ4ToF16Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(scalesF16.ptr),
		(*C.ushort)(outF16.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func DequantizeK4ToF16(qData, superScalesF16, subScales, outF16 DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || superScalesF16.ptr == nil || subScales.ptr == nil || outF16.ptr == nil {
		return fmt.Errorf("k4 dequant buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("k4 dequant rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaDequantizeK4ToF16Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(superScalesF16.ptr),
		(*C.uchar)(subScales.ptr),
		(*C.ushort)(outF16.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func SiluMulF32(gate, up, out DeviceBuffer, n int, stream Stream) error {
	if gate.ptr == nil || up.ptr == nil || out.ptr == nil {
		return fmt.Errorf("silu mul buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("silu mul n must be > 0")
	}
	return cudaErr(C.mantleCudaSiluMulF32Wrapper(
		(*C.float)(gate.ptr),
		(*C.float)(up.ptr),
		(*C.float)(out.ptr),
		C.int(n),
		stream.ptr,
	))
}

func ConvertF32ToF16(in, out DeviceBuffer, n int, stream Stream) error {
	if in.ptr == nil || out.ptr == nil {
		return fmt.Errorf("f32->f16 convert buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("f32->f16 convert n must be > 0")
	}
	return cudaErr(C.mantleCudaConvertF32ToF16Wrapper(
		(*C.float)(in.ptr),
		(*C.ushort)(out.ptr),
		C.int(n),
		stream.ptr,
	))
}

func AttentionInnerF16CacheF32(q, cacheK, cacheV, out DeviceBuffer, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen int, scale float32, stream Stream) error {
	if q.ptr == nil || cacheK.ptr == nil || cacheV.ptr == nil || out.ptr == nil {
		return fmt.Errorf("attention inner buffer is nil")
	}
	if pos < 0 || start < 0 || start > pos {
		return fmt.Errorf("attention inner invalid position/window")
	}
	if kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 || cacheLen <= 0 {
		return fmt.Errorf("attention inner dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaAttentionInnerF16CacheF32Wrapper(
		(*C.float)(q.ptr),
		(*C.ushort)(cacheK.ptr),
		(*C.ushort)(cacheV.ptr),
		(*C.float)(out.ptr),
		C.int(pos),
		C.int(start),
		C.int(kvStride),
		C.int(headDim),
		C.int(nHead),
		C.int(kvHeads),
		C.int(cacheLen),
		C.float(scale),
		stream.ptr,
	))
}

func cublasErr(code C.int) error {
	if code == 0 {
		return nil
	}
	return fmt.Errorf("cublas error %d", int(code))
}

func cudaErr(code C.int) error {
	if code == 0 {
		return nil
	}
	msg := C.GoString(C.mantleCudaGetErrorString(C.cudaError_t(code)))
	return &CUDAError{Code: int(code), Msg: msg}
}
