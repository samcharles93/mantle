//go:build cuda

package backend

func Has(name string) bool {
	switch name {
	case CUDA:
		return true
	default:
		return name == CPU
	}
}
