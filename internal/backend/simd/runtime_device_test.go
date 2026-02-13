package simd

import "testing"

type mockDeviceStateOps struct {
	addOK      bool
	addCalls   int
	dirtyCalls int
}

func (m *mockDeviceStateOps) BeginToken(_ []float32) {}

func (m *mockDeviceStateOps) EndToken(_ []float32) {}

func (m *mockDeviceStateOps) HostStateDirty(_ []float32) {
	m.dirtyCalls++
}

func (m *mockDeviceStateOps) SyncHostState(_ []float32) {}

func (m *mockDeviceStateOps) DeviceAdd(dst, src []float32) bool {
	m.addCalls++
	if !m.addOK {
		return false
	}
	for i := range dst {
		dst[i] += src[i]
	}
	return true
}

func (m *mockDeviceStateOps) DeviceRMSNorm(_ []float32, _ []float32, _ []float32, _ float32) bool {
	return false
}

func (m *mockDeviceStateOps) DeviceMatVec(_ []float32, _ *Mat, _ []float32) bool {
	return false
}

func TestAddResidualDeviceSuccessSkipsHostDirty(t *testing.T) {
	ds := &mockDeviceStateOps{addOK: true}
	dst := []float32{1, 2, 3}
	src := []float32{4, 5, 6}

	addResidual(ds, dst, src)

	if ds.addCalls != 1 {
		t.Fatalf("DeviceAdd calls = %d, want 1", ds.addCalls)
	}
	if ds.dirtyCalls != 0 {
		t.Fatalf("HostStateDirty calls = %d, want 0", ds.dirtyCalls)
	}
	want := []float32{5, 7, 9}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAddResidualFallbackMarksHostDirty(t *testing.T) {
	ds := &mockDeviceStateOps{addOK: false}
	dst := []float32{1, 2, 3}
	src := []float32{4, 5, 6}

	addResidual(ds, dst, src)

	if ds.addCalls != 1 {
		t.Fatalf("DeviceAdd calls = %d, want 1", ds.addCalls)
	}
	if ds.dirtyCalls != 1 {
		t.Fatalf("HostStateDirty calls = %d, want 1", ds.dirtyCalls)
	}
	want := []float32{5, 7, 9}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}
