package mcf

import "errors"

var (
	ErrInvalidMagic     = errors.New("invalid MCF magic")
	ErrUnsupportedMajor = errors.New("unsupported MCF major version")
	ErrCorruptFile      = errors.New("corrupt MCF file")
)
