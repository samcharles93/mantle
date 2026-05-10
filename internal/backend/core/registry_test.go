package core

import (
	"reflect"
	"sort"
	"testing"
)

func TestRegisterBackend(t *testing.T) {
	// start fresh for test determinism
	backendMutex.Lock()
	backendRegistry = make(map[string]*BackendInfo)
	backendMutex.Unlock()

	RegisterBackend("cpu", DefaultCapabilities())
	bi := GetBackend("cpu")
	if bi == nil {
		t.Fatalf("expected cpu backend to be registered")
	}
	if bi.Name != "cpu" {
		t.Fatalf("unexpected name: %s", bi.Name)
	}
	if !bi.Capabilities.Supports(CapGraphCompute) {
		t.Fatalf("expected cpu to support graph compute")
	}
}

func TestListBackends(t *testing.T) {
	// reset
	backendMutex.Lock()
	backendRegistry = make(map[string]*BackendInfo)
	backendMutex.Unlock()

	RegisterBackend("b", DefaultCapabilities())
	RegisterBackend("a", SIMDCapabilities())
	names := ListBackends()
	want := []string{"a", "b"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("expected sorted names %v, got %v", want, names)
	}
}

func TestBackendSupports(t *testing.T) {
	backendMutex.Lock()
	backendRegistry = make(map[string]*BackendInfo)
	backendMutex.Unlock()

	RegisterBackend("simd", SIMDCapabilities())
	if !BackendSupports("simd", CapFusedAttention) {
		t.Fatalf("simd should support fused attention")
	}
	if BackendSupports("simd", CapFusedMoE) {
		t.Fatalf("simd should not support MoE by default")
	}
}

func TestRegisterDuplicate(t *testing.T) {
	backendMutex.Lock()
	backendRegistry = make(map[string]*BackendInfo)
	backendMutex.Unlock()

	RegisterBackend("cpu", DefaultCapabilities())
	// update cpu to have SIMD-like caps
	RegisterBackend("cpu", SIMDCapabilities())
	bi := GetBackend("cpu")
	if bi == nil {
		t.Fatalf("cpu missing after update")
	}
	if !bi.Capabilities.Supports(CapFusedFFN) {
		t.Fatalf("expected cpu to now support fused FFN after update")
	}
}

// Ensure ListBackends is deterministic even when map iteration order varies
func TestListBackendsDeterministic(t *testing.T) {
	backendMutex.Lock()
	backendRegistry = make(map[string]*BackendInfo)
	backendMutex.Unlock()

	// insert in reverse order
	RegisterBackend("z", DefaultCapabilities())
	RegisterBackend("y", DefaultCapabilities())
	RegisterBackend("x", DefaultCapabilities())

	names := ListBackends()
	sorted := make([]string, len(names))
	copy(sorted, names)
	sort.Strings(sorted)
	if !reflect.DeepEqual(names, sorted) {
		t.Fatalf("ListBackends not sorted: got %v, want %v", names, sorted)
	}
}
