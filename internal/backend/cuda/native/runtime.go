//go:build cuda

package native

/*
#cgo LDFLAGS: -lcudart -lcublas

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
extern cudaError_t cudaFree(void* ptr);
extern cudaError_t cudaMemcpy(void* dst, const void* src, unsigned long long size, int kind);
extern cudaError_t cudaMemcpyAsync(void* dst, const void* src, unsigned long long size, int kind, cudaStream_t stream);
extern cudaError_t cudaMallocHost(void** ptr, unsigned long long size);
extern cudaError_t cudaFreeHost(void* ptr);

#define MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE 1
#define MANTLE_CUDA_MEMCPY_DEVICE_TO_HOST 2

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
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type Stream struct {
	ptr C.cudaStream_t
}

type BlasHandle struct {
	ptr C.cublasHandle_t
}

type DeviceBuffer struct {
	ptr unsafe.Pointer
}

type HostBuffer struct {
	ptr unsafe.Pointer
}

func DeviceCount() (int, error) {
	var count C.int
	if err := cudaErr(C.mantleCudaGetDeviceCount(&count)); err != nil {
		return 0, err
	}
	return int(count), nil
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
	return cudaErr(C.mantleCudaStreamSynchronize(s.ptr))
}

func AllocDevice(bytes int64) (DeviceBuffer, error) {
	if bytes <= 0 {
		return DeviceBuffer{}, fmt.Errorf("device alloc size must be > 0")
	}
	var ptr unsafe.Pointer
	if err := cudaErr(C.mantleCudaMalloc((*unsafe.Pointer)(&ptr), C.ulonglong(bytes))); err != nil {
		return DeviceBuffer{}, err
	}
	return DeviceBuffer{ptr: ptr}, nil
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
	return cudaErr(C.mantleCudaMemcpyAsync(dst.ptr, src, C.ulonglong(bytes), C.MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE, stream.ptr))
}

func MemcpyD2HAsync(dst unsafe.Pointer, src DeviceBuffer, bytes int64, stream Stream) error {
	if bytes <= 0 {
		return nil
	}
	return cudaErr(C.mantleCudaMemcpyAsync(dst, src.ptr, C.ulonglong(bytes), C.MANTLE_CUDA_MEMCPY_DEVICE_TO_HOST, stream.ptr))
}

func MemcpyH2D(dst DeviceBuffer, src unsafe.Pointer, bytes int64) error {
	if bytes <= 0 {
		return nil
	}
	return cudaErr(C.mantleCudaMemcpy(dst.ptr, src, C.ulonglong(bytes), C.MANTLE_CUDA_MEMCPY_HOST_TO_DEVICE))
}

func MemcpyD2H(dst unsafe.Pointer, src DeviceBuffer, bytes int64) error {
	if bytes <= 0 {
		return nil
	}
	return cudaErr(C.mantleCudaMemcpy(dst, src.ptr, C.ulonglong(bytes), C.MANTLE_CUDA_MEMCPY_DEVICE_TO_HOST))
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
	return fmt.Errorf("cuda runtime error %d: %s", int(code), msg)
}
