package mcf

import "errors"

var (
	ErrInvalidMagic     = errors.New("invalid MCF magic")
	ErrUnsupportedMajor = errors.New("unsupported MCF major version")
	ErrUnsupportedMinor = errors.New("unsupported MCF minor version")
	ErrCorruptFile      = errors.New("corrupt MCF file")
)
