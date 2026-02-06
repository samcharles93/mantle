//go:build cuda

package cuda

import "fmt"

func cudaExecutionError(rec any) error {
	if recErr, ok := rec.(error); ok {
		return fmt.Errorf("cuda execution failed: %w", recErr)
	}
	return fmt.Errorf("cuda execution failed: %v", rec)
}
