package mcf

import "encoding/binary"

const (
	MagicMCF = "MCF\x00"
	// mcfHeaderSize is the fixed on-disk size of MCFHeader in bytes.
	mcfHeaderSize = 40

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
	if h.HeaderSize < mcfHeaderSize {
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

func decodeHeader(data []byte) (MCFHeader, bool) {
	if len(data) < mcfHeaderSize {
		return MCFHeader{}, false
	}
	var h MCFHeader
	copy(h.Magic[:], data[0:4])
	h.Major = binary.LittleEndian.Uint16(data[4:6])
	h.Minor = binary.LittleEndian.Uint16(data[6:8])
	h.HeaderSize = binary.LittleEndian.Uint32(data[8:12])
	h.SectionCount = binary.LittleEndian.Uint32(data[12:16])
	h.SectionDirOffset = binary.LittleEndian.Uint64(data[16:24])
	h.FileSize = binary.LittleEndian.Uint64(data[24:32])
	h.Flags = binary.LittleEndian.Uint64(data[32:40])
	return h, true
}

func encodeHeader(dst []byte, h MCFHeader) bool {
	if len(dst) < mcfHeaderSize {
		return false
	}
	copy(dst[0:4], h.Magic[:])
	binary.LittleEndian.PutUint16(dst[4:6], h.Major)
	binary.LittleEndian.PutUint16(dst[6:8], h.Minor)
	binary.LittleEndian.PutUint32(dst[8:12], h.HeaderSize)
	binary.LittleEndian.PutUint32(dst[12:16], h.SectionCount)
	binary.LittleEndian.PutUint64(dst[16:24], h.SectionDirOffset)
	binary.LittleEndian.PutUint64(dst[24:32], h.FileSize)
	binary.LittleEndian.PutUint64(dst[32:40], h.Flags)
	return true
}
