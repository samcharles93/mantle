package ux

import "os"

// stdinIsTTY is a small seam for tests.
var stdinIsTTY = isTTY

func isTTY() bool {
	st, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return (st.Mode() & os.ModeCharDevice) != 0
}
