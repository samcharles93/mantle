//go:build cuda

package cuda

import (
	"errors"
	"strings"
	"testing"
)

func TestCudaExecutionErrorWrapsError(t *testing.T) {
	err := cudaExecutionError(errors.New("boom"))
	if !strings.Contains(err.Error(), "cuda execution failed") {
		t.Fatalf("unexpected message: %v", err)
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Fatalf("missing wrapped message: %v", err)
	}
}

func TestCudaExecutionErrorValue(t *testing.T) {
	err := cudaExecutionError("panic text")
	if !strings.Contains(err.Error(), "panic text") {
		t.Fatalf("unexpected message: %v", err)
	}
}
