package mcf

import "unsafe"

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
