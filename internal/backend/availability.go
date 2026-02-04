package backend

import "strings"

// Available returns a comma-separated list of available backends.
func Available() string {
	entries := []string{CPU}
	if Has(CUDA) {
		entries = append(entries, CUDA)
	}
	return strings.Join(entries, ",")
}
