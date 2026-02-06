package mcf

import (
	"os"
)

const (
	mcfAlign = 8
)

func rangesOverlap(a0, a1, b0, b1 uint64) bool {
	// half-open ranges [a0,a1) and [b0,b1)
	return a0 < b1 && b0 < a1
}

func writeFull(f *os.File, p []byte) error {
	for len(p) > 0 {
		n, err := f.Write(p)
		if err != nil {
			return err
		}
		p = p[n:]
	}
	return nil
}
