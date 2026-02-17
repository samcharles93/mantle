//go:build cuda

package native

import (
	"strings"
	"testing"
	"unsafe"
)

const maxInt32 = 1<<31 - 1

func dummyDeviceBuffer() DeviceBuffer {
	// Use a valid, heap-backed address so stack scanning never sees an invalid pointer value.
	p := new(byte)
	return DeviceBuffer{ptr: unsafe.Pointer(p)}
}

func requireErrContains(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error containing %q, got nil", want)
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("error %q does not contain %q", err.Error(), want)
	}
}

func TestSoftmaxRowsF32Validation(t *testing.T) {
	dummy := dummyDeviceBuffer()
	stream := Stream{}

	requireErrContains(t, SoftmaxRowsF32(DeviceBuffer{}, 1, 1, stream), "native.SoftmaxRowsF32")
	requireErrContains(t, SoftmaxRowsF32(dummy, 0, 1, stream), "native.SoftmaxRowsF32")
	requireErrContains(t, SoftmaxRowsF32(dummy, int(int64(maxInt32)+1), 1, stream), "native.SoftmaxRowsF32")
}

func TestSiluMulF32Validation(t *testing.T) {
	dummy := dummyDeviceBuffer()
	stream := Stream{}

	requireErrContains(t, SiluMulF32(DeviceBuffer{}, dummy, dummy, 1, stream), "native.SiluMulF32")
	requireErrContains(t, SiluMulF32(dummy, dummy, dummy, 0, stream), "native.SiluMulF32")
	requireErrContains(t, SiluMulF32(dummy, dummy, dummy, int(int64(maxInt32)+1), stream), "native.SiluMulF32")
}

func TestAddVectorsF32Validation(t *testing.T) {
	dummy := dummyDeviceBuffer()
	stream := Stream{}

	requireErrContains(t, AddVectorsF32(DeviceBuffer{}, dummy, 1, stream), "native.AddVectorsF32")
	requireErrContains(t, AddVectorsF32(dummy, dummy, 0, stream), "native.AddVectorsF32")
	requireErrContains(t, AddVectorsF32(dummy, dummy, int(int64(maxInt32)+1), stream), "native.AddVectorsF32")
}

func TestShortConvDepthwiseValidation(t *testing.T) {
	dummy := dummyDeviceBuffer()
	stream := Stream{}

	requireErrContains(t, ShortConvDepthwise(DeviceBuffer{}, dummy, dummy, dummy, 1, 1, stream), "native.ShortConvDepthwise")
	requireErrContains(t, ShortConvDepthwise(dummy, dummy, dummy, dummy, 0, 1, stream), "native.ShortConvDepthwise")
	requireErrContains(t, ShortConvDepthwise(dummy, dummy, dummy, dummy, int(int64(maxInt32)+1), 1, stream), "native.ShortConvDepthwise")
}

func TestConvertF32ToF16Validation(t *testing.T) {
	dummy := dummyDeviceBuffer()
	stream := Stream{}

	requireErrContains(t, ConvertF32ToF16(DeviceBuffer{}, dummy, 1, stream), "native.ConvertF32ToF16")
	requireErrContains(t, ConvertF32ToF16(dummy, dummy, 0, stream), "native.ConvertF32ToF16")
	requireErrContains(t, ConvertF32ToF16(dummy, dummy, int(int64(maxInt32)+1), stream), "native.ConvertF32ToF16")
}
