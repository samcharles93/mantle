package mcf

import "encoding/binary"

type SectionType uint32

// mcfSectionSize is the fixed on-disk size of MCFSection in bytes.
const mcfSectionSize = 24

const (
	SectionModelInfo   SectionType = 0x0001
	SectionQuantInfo   SectionType = 0x0002 // Quantization metadata (v1.1)
	SectionTensorIndex SectionType = 0x0003
	SectionTensorData  SectionType = 0x0004

	SectionHFConfigJSON           SectionType = 0x0100
	SectionHFGenerationConfigJSON SectionType = 0x0101
	SectionTokenizerJSON          SectionType = 0x0102
	SectionTokenizerConfigJSON    SectionType = 0x0103
	SectionVocabJSON              SectionType = 0x0104
	SectionMergesTXT              SectionType = 0x0105
	// SectionProcessorConfigJSON optionally embeds a multimodal model's
	// processor_config.json (image/video preprocessing parameters). Absent for
	// text-only models; readers must tolerate its absence.
	SectionProcessorConfigJSON SectionType = 0x0106
	// SectionAudioTokenizer optionally embeds an audio model's speech codec
	// directory (audio_tokenizer/config.json + audio_tokenizer/model.safetensors,
	// e.g. LFM3-Omni's LiquidCodec) as a versioned blob; see audio_tokenizer.go.
	// Absent for models without a codec; readers must tolerate its absence.
	SectionAudioTokenizer SectionType = 0x0107
)

type MCFSection struct {
	Type    uint32
	Version uint32
	Offset  uint64
	Size    uint64
}

func decodeSection(data []byte) (MCFSection, bool) {
	if len(data) < mcfSectionSize {
		return MCFSection{}, false
	}
	return MCFSection{
		Type:    binary.LittleEndian.Uint32(data[0:4]),
		Version: binary.LittleEndian.Uint32(data[4:8]),
		Offset:  binary.LittleEndian.Uint64(data[8:16]),
		Size:    binary.LittleEndian.Uint64(data[16:24]),
	}, true
}

func encodeSection(dst []byte, s MCFSection) bool {
	if len(dst) < mcfSectionSize {
		return false
	}
	binary.LittleEndian.PutUint32(dst[0:4], s.Type)
	binary.LittleEndian.PutUint32(dst[4:8], s.Version)
	binary.LittleEndian.PutUint64(dst[8:16], s.Offset)
	binary.LittleEndian.PutUint64(dst[16:24], s.Size)
	return true
}
