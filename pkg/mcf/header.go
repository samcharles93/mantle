package mcf

import "unsafe"

const (
	MagicMCF = "MCF\x00"

	// Current Major Version: 1 (Breaking changes only)
	CurrentMajor uint16 = 1

	// Current Minor Version
	CurrentMinor uint16 = 1

	// FlagTensorDataAligned64: REQUIRED for all files using DTypeQ* or DTypeK*
	// This ensures the SoA payload is 64-byte aligned for AVX-512.
	FlagTensorDataAligned64 uint64 = 1 << 0
)

type MCFHeader struct {
	Magic            [4]byte
	Major            uint16
	Minor            uint16
	HeaderSize       uint32
	SectionCount     uint32
	SectionDirOffset uint64
	FileSize         uint64
	Flags            uint64
}

func (h *MCFHeader) Valid() bool {
	if string(h.Magic[:]) != MagicMCF {
		return false
	}
	if h.HeaderSize < uint32(unsafe.Sizeof(MCFHeader{})) {
		return false
	}
	if h.SectionCount == 0 {
		return false
	}
	return true
}

func (h *MCFHeader) Compatible() bool {
	return h.Major == CurrentMajor
}
