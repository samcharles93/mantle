//go:build !cuda

package backend

func Has(name string) bool {
	switch name {
	case CPU:
		return true
	default:
		return false
	}
}
