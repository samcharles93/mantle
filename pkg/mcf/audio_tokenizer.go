package mcf

import (
	"encoding/binary"
	"fmt"
)

// Section 0x0107 carries an audio model's speech codec directory
// (audio_tokenizer/config.json + audio_tokenizer/model.safetensors, e.g.
// LFM3-Omni's LiquidCodec) as one optional, versioned blob. Absent for models
// without a codec; readers must tolerate its absence and unknown versions.
//
// Payload layout (version 1), all fields little-endian:
//
//	u32 magic = audioTokenizerMagic ("ATK1")
//	u32 version = AudioTokenizerSectionVersion (1)
//	u64 configSize      // bytes of audio_tokenizer/config.json
//	u64 safetensorsSize // bytes of audio_tokenizer/model.safetensors
//	[configSize bytes of config.json]
//	[safetensorsSize bytes of model.safetensors]
const (
	audioTokenizerMagic          uint32 = 0x314B5441 // "ATK1" in little-endian byte order
	AudioTokenizerSectionVersion uint32 = 1
	audioTokenizerHeaderSize            = 24
)

// EncodeAudioTokenizerSection packs the audio tokenizer's config.json and
// model.safetensors bytes into a section 0x0107 payload.
func EncodeAudioTokenizerSection(configJSON, safetensors []byte) []byte {
	out := make([]byte, audioTokenizerHeaderSize+len(configJSON)+len(safetensors))
	binary.LittleEndian.PutUint32(out[0:4], audioTokenizerMagic)
	binary.LittleEndian.PutUint32(out[4:8], AudioTokenizerSectionVersion)
	binary.LittleEndian.PutUint64(out[8:16], uint64(len(configJSON)))
	binary.LittleEndian.PutUint64(out[16:24], uint64(len(safetensors)))
	copy(out[audioTokenizerHeaderSize:], configJSON)
	copy(out[audioTokenizerHeaderSize+len(configJSON):], safetensors)
	return out
}

// DecodeAudioTokenizerSection parses a section 0x0107 payload into the
// codec's config.json and model.safetensors bytes. All size fields are
// bounds-checked against the payload length with overflow-safe math.
func DecodeAudioTokenizerSection(data []byte) (configJSON, safetensors []byte, err error) {
	if len(data) < audioTokenizerHeaderSize {
		return nil, nil, fmt.Errorf("audio_tokenizer: section payload %d bytes, need at least %d-byte header", len(data), audioTokenizerHeaderSize)
	}
	if magic := binary.LittleEndian.Uint32(data[0:4]); magic != audioTokenizerMagic {
		return nil, nil, fmt.Errorf("audio_tokenizer: bad magic %#x, want %#x", magic, audioTokenizerMagic)
	}
	if v := binary.LittleEndian.Uint32(data[4:8]); v != AudioTokenizerSectionVersion {
		return nil, nil, fmt.Errorf("audio_tokenizer: unsupported payload version %d (want %d)", v, AudioTokenizerSectionVersion)
	}
	configSize := binary.LittleEndian.Uint64(data[8:16])
	stSize := binary.LittleEndian.Uint64(data[16:24])
	if configSize > uint64(len(data)) || stSize > uint64(len(data)) {
		return nil, nil, fmt.Errorf("audio_tokenizer: declared sizes (%d, %d) exceed payload %d bytes", configSize, stSize, len(data))
	}
	if configSize+stSize < configSize || audioTokenizerHeaderSize+configSize+stSize < configSize {
		return nil, nil, fmt.Errorf("audio_tokenizer: size fields overflow")
	}
	total := uint64(audioTokenizerHeaderSize) + configSize + stSize
	if total > uint64(len(data)) {
		return nil, nil, fmt.Errorf("audio_tokenizer: payload truncated: header + %d + %d bytes exceeds %d", configSize, stSize, len(data))
	}
	body := data[audioTokenizerHeaderSize:]
	configJSON = body[:configSize]
	stOff := configSize
	safetensors = body[stOff : stOff+stSize]
	return configJSON, safetensors, nil
}
