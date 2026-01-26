package mcf

import "unsafe"

// MCF global constants must never change.
const (
	// MagicMCF is the file magic for all MCF containers.
	// It is encoded as "MCF\0".
	MagicMCF = "MCF\x00"

	// Current Major Version: Any change indicates a breaking format change.
	CurrentMajor uint16 = 1

	// Current Minor Version: Versions may add new optional sections or fields.
	CurrentMinor uint16 = 0

	// Format level flags
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
