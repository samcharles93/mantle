// Package mcf implements the Model Container File format.
//
// MCF is a single-file, memory-mappable container for machine learning models.
// It describes structure and data only and never implies runtime behaviour.
package mcf

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

type SectionType uint32

const (
	SectionModelInfo   SectionType = 0x0001
	SectionQuantInfo   SectionType = 0x0002
	SectionTensorIndex SectionType = 0x0003
	SectionTensorData  SectionType = 0x0004
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

type MCFSection struct {
	Type    uint32
	Version uint32
	Offset  uint64
	Size    uint64
}
