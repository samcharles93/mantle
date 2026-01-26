package mcf

type Section struct {
	MCFSection
}

func (s *Section) End() uint64 {
	return s.Offset + s.Size
}
