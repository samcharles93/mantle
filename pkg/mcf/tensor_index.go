package mcf

type TensorEntry struct {
	NameHash uint64
	DType    uint32
	Rank     uint32
	ShapeOff uint64
	DataOff  uint64
	DataSize uint64
}
