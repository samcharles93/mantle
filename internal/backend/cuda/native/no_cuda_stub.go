//go:build !cuda

package native

// This file provides minimal stub types for environments where CUDA build
// tags are not enabled. It allows tools (LSP, editors) to load the package
// without requiring the CUDA toolchain. The real implementations are built
// only when the 'cuda' build tag is active.

import (
	"errors"
	"unsafe"
)

type DeviceBuffer struct {
	ptr unsafe.Pointer
	n   int64
}

func (d DeviceBuffer) Ptr() unsafe.Pointer { return d.ptr }
func (d DeviceBuffer) Nbytes() int64       { return d.n }
func (d DeviceBuffer) Managed() bool       { return false }
func (d *DeviceBuffer) Free() error        { *d = DeviceBuffer{}; return nil }

type HostBuffer struct{ ptr unsafe.Pointer }

func (h HostBuffer) Ptr() unsafe.Pointer { return h.ptr }
func (h *HostBuffer) Free() error        { *h = HostBuffer{}; return nil }

type Stream struct{ ptr unsafe.Pointer }

func (s Stream) Ptr() unsafe.Pointer { return s.ptr }
func (s *Stream) Destroy() error     { *s = Stream{}; return nil }
func (s Stream) Synchronize() error  { return nil }

type BlasHandle struct{}

type BlasDataType int

const (
	BlasF32 BlasDataType = iota
	BlasF16
	BlasBF16
)

type GraphExec struct{}

func (g GraphExec) Destroy() error             { return nil }
func (g GraphExec) Launch(stream Stream) error { return nil }

const (
	BlasOpT         = 0
	BlasOpN         = 1
	BlasComputeF32  = 0
	BlasGemmDefault = 0
)

const (
	MemAdviseSetReadMostly = 1
	MemAdviseSetAccessedBy = 2
)

// Minimal no-op functions used in non-cuda analysis contexts.
func AllocDevice(n int64) (DeviceBuffer, error)                                    { return DeviceBuffer{}, nil }
func AllocManaged(n int64) (DeviceBuffer, error)                                   { return DeviceBuffer{}, nil }
func MemcpyH2D(dev DeviceBuffer, src unsafe.Pointer, n int64) error                { return nil }
func MemcpyH2DAsync(dev DeviceBuffer, src unsafe.Pointer, n int64, s Stream) error { return nil }
func MemcpyD2H(dst unsafe.Pointer, dev DeviceBuffer, n int64) error                { return nil }
func MemcpyD2HAsyncAt(dst unsafe.Pointer, dev DeviceBuffer, off int64, n int64, s Stream) error {
	return nil
}
func MemcpyD2DAsync(dst, src DeviceBuffer, n int64, s Stream) error                { return nil }
func MemcpyD2HAsync(dst unsafe.Pointer, dev DeviceBuffer, n int64, s Stream) error { return nil }
func MemcpyD2HAt(dst unsafe.Pointer, dev DeviceBuffer, off int64, n int64) error   { return nil }
func MemcpyH2DAt(dev DeviceBuffer, src unsafe.Pointer, off int64, n int64) error   { return nil }
func MemInfo() (free int64, total int64, err error)                                { return 0, 0, nil }
func NewStream() (Stream, error)                                                   { return Stream{}, nil }
func MemPrefetchAsync(dev DeviceBuffer, n int64, device int, s Stream) error       { return nil }
func MemAdvise(dev DeviceBuffer, n int64, advice int, device int) error            { return nil }

func RecordFlushIfPending()    {}
func RecordRMSNorm()           {}
func RecordMatVec()            {}
func RecordMatVecCPUFallback() {}
func RecordGraphFailure()      {}
func RecordGraphLaunch()       {}
func RecordGraphCapture()      {}

func RecordAttribFlushLastResult() func() { return func() {} }
func RecordAttribEndToken() func()        { return func() {} }
func RecordAttribSyncHostState() func()   { return func() {} }
func RecordAttribSyncDeviceSlice() func() { return func() {} }

func RoundBF16InPlaceF32(buf DeviceBuffer, n int, s Stream) error                     { return nil }
func ScaleRoundBF16InPlaceF32(buf DeviceBuffer, scale float32, n int, s Stream) error { return nil }
func AddVectorsF32(dst, src DeviceBuffer, n int, s Stream) error                      { return nil }
func ArgMaxF32(dev DeviceBuffer, n int, s Stream) (int, error)                        { return 0, nil }
func LogitSoftcapF32(dev DeviceBuffer, softcap float32, n int, s Stream) error        { return nil }

func ConvertF32ToF16(in, out DeviceBuffer, n int, s Stream) error  { return nil }
func ConvertF32ToBF16(in, out DeviceBuffer, n int, s Stream) error { return nil }

func GemmEx(blas BlasHandle, opA, opB int, m, n, k int, alpha float64, a DeviceBuffer, aType BlasDataType, lda int, b DeviceBuffer, bType BlasDataType, ldb int, beta float64, c DeviceBuffer, cType BlasDataType, ldc int, compute int, algo int) error {
	return nil
}

// Placeholder implementations for fused/quant kernels referenced by ops.go
func FusedRMSNormMatVecBF16(y, w, x, norm DeviceBuffer, eps float32, rows, cols int, s Stream) error {
	return nil
}

func FusedRMSNormMatVecF32(y, w, x, norm DeviceBuffer, eps float32, rows, cols int, s Stream) error {
	return nil
}

func MemcpyD2HAt(dst unsafe.Pointer, dev DeviceBuffer, off int64, n int64) error { return nil }

// Quant/attention kernels (no-op stubs)
func QuantMatVecQ4F32() error         { return nil }
func QuantMatVecK4F32() error         { return nil }
func QuantMatVecInt8BlocksF32() error { return nil }

func AttentionInnerMixedCacheF32(qProj DeviceBuffer, kF16, vF16 DeviceBuffer, kQ8, vQ8 DeviceBuffer, kQ8Scales, vQ8Scales DeviceBuffer, out DeviceBuffer, useQ8K, useQ8V bool, pos, start, kvStride, headDim, nHead, kvHeads, actualCacheLen int, scale, softcap float32, s Stream) error {
	return nil
}

func AttentionInnerF16CacheF32(qProj, kF16, vF16, out DeviceBuffer, pos, start, kvStride, headDim, nHead, kvHeads, actualCacheLen int, scale, softcap float32, s Stream) error {
	return nil
}

func ApplyRoPEInplaceF32(buf DeviceBuffer, invFreq DeviceBuffer, pos int, scale float32, headDim, half, nHead int, s Stream) error {
	return nil
}

func StoreKVQ8RowBroadcast(dst DeviceBuffer, scales DeviceBuffer, src DeviceBuffer, pos, kvStride, blocksPerStride int, s Stream) error {
	return nil
}
func StoreKVF16Row(dst DeviceBuffer, src DeviceBuffer, pos, kvStride int, s Stream) error { return nil }
func MemcpyD2H(dst unsafe.Pointer, dev DeviceBuffer, n int64) error                       { return nil }
