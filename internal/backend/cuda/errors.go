//go:build cuda

package cuda

import (
	"fmt"
	"runtime/debug"
)

func cudaExecutionError(rec any) error {
	stack := string(debug.Stack())
	if recErr, ok := rec.(error); ok {
		return fmt.Errorf("cuda execution failed: %w\n%s", recErr, stack)
	}
	return fmt.Errorf("cuda execution failed: %v\n%s", rec, stack)
}
