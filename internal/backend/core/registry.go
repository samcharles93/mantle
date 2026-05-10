package core

import (
	"sort"
	"sync"
)

// BackendInfo describes a registered backend and its capabilities.
type BackendInfo struct {
	Name         string
	Capabilities *Capabilities
}

var (
	// backendRegistry holds registered backends by name. Protect with
	// backendMutex for concurrent access.
	backendMutex    sync.RWMutex
	backendRegistry = make(map[string]*BackendInfo)
)

// RegisterBackend registers or updates a backend's capabilities. It is safe
// to call concurrently.
func RegisterBackend(name string, caps *Capabilities) {
	backendMutex.Lock()
	defer backendMutex.Unlock()
	backendRegistry[name] = &BackendInfo{Name: name, Capabilities: caps}
}

// GetBackend retrieves backend info by name or nil if not found.
func GetBackend(name string) *BackendInfo {
	backendMutex.RLock()
	defer backendMutex.RUnlock()
	if b, ok := backendRegistry[name]; ok {
		return b
	}
	return nil
}

// ListBackends returns a sorted list of registered backend names for
// deterministic iteration.
func ListBackends() []string {
	backendMutex.RLock()
	defer backendMutex.RUnlock()
	names := make([]string, 0, len(backendRegistry))
	for k := range backendRegistry {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// BackendSupports reports whether the named backend supports the given
// capability. Returns false if the backend is not registered.
func BackendSupports(name string, cap OpCapability) bool {
	b := GetBackend(name)
	if b == nil || b.Capabilities == nil {
		return false
	}
	return b.Capabilities.Supports(cap)
}

// Pre-register known backends. This provides a sensible default registry
// without requiring callers to register the common CPU paths.
func init() {
	RegisterBackend("cpu", DefaultCapabilities())
	RegisterBackend("simd", SIMDCapabilities())
}
