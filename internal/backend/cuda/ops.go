//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type Ops struct {
	stream native.Stream
	blas   native.BlasHandle

	mu      sync.Mutex
	weights map[*simd.Mat]deviceMat

	xHost native.HostBuffer
	yHost native.HostBuffer
	xDev  native.DeviceBuffer
	yDev  native.DeviceBuffer

	xCapBytes int
	yCapBytes int
	xDevBytes int
	yDevBytes int
}

type deviceMat struct {
	buf   native.DeviceBuffer
	dtype native.BlasDataType
	rows  int
	cols  int
}

func NewOps(stream native.Stream, blas native.BlasHandle) *Ops {
	return &Ops{
		stream:  stream,
		blas:    blas,
		weights: make(map[*simd.Mat]deviceMat),
	}
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

	if e := o.xDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.yDev.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.xHost.Free(); e != nil && err == nil {
		err = e
	}
	if e := o.yHost.Free(); e != nil && err == nil {
		err = e
	}
	o.xCapBytes = 0
	o.yCapBytes = 0
	o.xDevBytes = 0
	o.yDevBytes = 0

	return err
}

func (o *Ops) MatVec(dst []float32, w *simd.Mat, x []float32) {
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

	devW, err := o.deviceMat(w)
	if err != nil {
		panic(err)
	}

	xBytes := xBufferBytes(devW.dtype, w.C)
	yBytes := int64(w.R) * int64(unsafe.Sizeof(float32(0)))
	if err := o.ensureHostVecs(int(xBytes), int(yBytes)); err != nil {
		panic(err)
	}
	if err := o.ensureDeviceVecs(int(xBytes), int(yBytes)); err != nil {
		panic(err)
	}

	if err := fillXBuffer(o.xHost, devW.dtype, x[:w.C]); err != nil {
		panic(err)
	}
	if err := native.MemcpyH2DAsync(o.xDev, o.xHost.Ptr(), xBytes, o.stream); err != nil {
		panic(err)
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
		panic(err)
	}

	if err := native.MemcpyD2HAsync(o.yHost.Ptr(), o.yDev, yBytes, o.stream); err != nil {
		panic(err)
	}
	if err := o.stream.Synchronize(); err != nil {
		panic(err)
	}

	copy(dst[:w.R], unsafe.Slice((*float32)(o.yHost.Ptr()), w.R))
	runtime.KeepAlive(x)
	runtime.KeepAlive(dst)
}

func (o *Ops) MatVecWithQuant(dst []float32, w *simd.Mat, x []float32, _ *simd.QuantVec) {
	o.MatVec(dst, w, x)
}

func (o *Ops) deviceMat(w *simd.Mat) (deviceMat, error) {
	if buf, ok := o.weights[w]; ok {
		return buf, nil
	}
	if w.Raw != nil && mcf.DTypeRequiresAligned64(w.DType) {
		return deviceMat{}, fmt.Errorf("cuda backend does not support quantized weights (dtype=%s)", dtypeString(w.DType))
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

func xBufferBytes(dtype native.BlasDataType, length int) int64 {
	switch dtype {
	case native.BlasF16, native.BlasBF16:
		return int64(length) * 2
	default:
		return int64(length) * 4
	}
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
	return uint16(uint32(math.Float32bits(v)) >> 16)
}

func dtypeString(dt mcf.TensorDType) string {
	return fmt.Sprintf("0x%04x", uint16(dt))
}
