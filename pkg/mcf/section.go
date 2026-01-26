package mcf

type SectionType uint32

const (
	SectionModelInfo   SectionType = 0x0001
	SectionQuantInfo   SectionType = 0x0002
	SectionTensorIndex SectionType = 0x0003
	SectionTensorData  SectionType = 0x0004
)

type MCFSection struct {
	Type    uint32
	Version uint32
	Offset  uint64
	Size    uint64
}

type Section struct {
	MCFSection
}

func (s *Section) End() uint64 {
	return s.Offset + s.Size
}
