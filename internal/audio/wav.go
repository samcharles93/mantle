package audio

import (
	"encoding/binary"
	"fmt"
	"math"
)

// WAV format tags and the sub-format GUID prefixes of WAVE_FORMAT_EXTENSIBLE.
const (
	wavFormatPCM        = 0x0001
	wavFormatIEEEFloat  = 0x0003
	wavFormatExtensible = 0xFFFE
)

// DecodeWAV parses a RIFF/WAVE file and returns mono float32 samples in [-1, 1]
// together with the file's sample rate.
//
// Supported codecs: 16-bit PCM (format 1) and 32-bit IEEE float (format 3),
// including their WAVE_FORMAT_EXTENSIBLE wrappers. Mono passes through; stereo
// is downmixed by averaging the two channels. Any other codec, bit depth, or
// channel count is rejected with a descriptive error.
//
// All parsing is bounds-checked: malformed or truncated files return an error,
// never a panic.
func DecodeWAV(data []byte) (samples []float32, sampleRate int, err error) {
	if len(data) < 12 {
		return nil, 0, fmt.Errorf("audio: not a RIFF/WAVE file: %d bytes is shorter than the 12-byte header", len(data))
	}
	if string(data[0:4]) != "RIFF" {
		return nil, 0, fmt.Errorf("audio: not a RIFF file: bad magic %q", data[0:4])
	}
	if string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("audio: not a WAVE file: bad form type %q", data[8:12])
	}

	var (
		format    uint16
		channels  uint16
		bits      uint16
		haveFmt   bool
		dataParts [][]byte
	)
	off := 12
	for off+8 <= len(data) {
		id := string(data[off : off+4])
		size := binary.LittleEndian.Uint32(data[off+4 : off+8])
		payloadStart := off + 8
		// overflow-safe payload end: size is uint32, payloadStart is int
		if uint64(payloadStart)+uint64(size) > uint64(len(data)) {
			return nil, 0, fmt.Errorf("audio: truncated %q chunk: header promises %d bytes at offset %d, file has %d", id, size, payloadStart, len(data))
		}
		payload := data[payloadStart : payloadStart+int(size)]
		switch id {
		case "fmt ":
			f, err := parseFmtChunk(payload)
			if err != nil {
				return nil, 0, err
			}
			format, channels, bits, haveFmt = f.format, f.channels, f.bits, true
			sampleRate = int(f.sampleRate)
		case "data":
			dataParts = append(dataParts, payload)
		}
		// chunks are word-aligned: odd sizes carry one pad byte
		off = payloadStart + int(size) + int(size&1)
	}
	if !haveFmt {
		return nil, 0, fmt.Errorf("audio: missing fmt chunk")
	}
	if len(dataParts) == 0 {
		return nil, 0, fmt.Errorf("audio: missing data chunk")
	}

	switch format {
	case wavFormatPCM:
		if bits != 16 {
			return nil, 0, fmt.Errorf("audio: unsupported PCM bit depth %d (only 16-bit PCM is supported)", bits)
		}
	case wavFormatIEEEFloat:
		if bits != 32 {
			return nil, 0, fmt.Errorf("audio: unsupported IEEE float bit depth %d (only 32-bit float is supported)", bits)
		}
	default:
		return nil, 0, fmt.Errorf("audio: unsupported codec: format %d with %d-bit samples (only 16-bit PCM and 32-bit IEEE float are supported)", format, bits)
	}
	if channels != 1 && channels != 2 {
		return nil, 0, fmt.Errorf("audio: unsupported channel count %d (only mono and stereo are supported)", channels)
	}

	var total int
	for _, p := range dataParts {
		total += len(p)
	}
	frameBytes := int(channels) * int(bits) / 8
	if total%frameBytes != 0 {
		return nil, 0, fmt.Errorf("audio: data chunk holds %d bytes, not a whole number of %d-byte frames", total, frameBytes)
	}
	nFrames := total / frameBytes
	samples = make([]float32, 0, nFrames)
	for _, p := range dataParts {
		for i := 0; i+frameBytes <= len(p); i += frameBytes {
			switch format {
			case wavFormatPCM:
				v := int16(binary.LittleEndian.Uint16(p[i:]))
				if channels == 2 {
					r := int16(binary.LittleEndian.Uint16(p[i+2:]))
					samples = append(samples, float32(int32(v)+int32(r))/(2*32768.0))
				} else {
					samples = append(samples, float32(v)/32768.0)
				}
			default: // wavFormatIEEEFloat
				v := math.Float32frombits(binary.LittleEndian.Uint32(p[i:]))
				if channels == 2 {
					r := math.Float32frombits(binary.LittleEndian.Uint32(p[i+4:]))
					samples = append(samples, (v+r)/2)
				} else {
					samples = append(samples, v)
				}
			}
		}
	}
	return samples, sampleRate, nil
}

type fmtChunk struct {
	format     uint16
	channels   uint16
	sampleRate uint32
	bits       uint16
}

// parseFmtChunk decodes a "fmt " payload, accepting the plain 16/18-byte form
// and WAVE_FORMAT_EXTENSIBLE (0xFFFE), whose sub-format GUID starts with the
// real format tag.
func parseFmtChunk(p []byte) (fmtChunk, error) {
	var out fmtChunk
	if len(p) < 16 {
		return out, fmt.Errorf("audio: truncated fmt chunk: %d bytes, need at least 16", len(p))
	}
	out.format = binary.LittleEndian.Uint16(p[0:])
	out.channels = binary.LittleEndian.Uint16(p[2:])
	out.sampleRate = binary.LittleEndian.Uint32(p[4:])
	out.bits = binary.LittleEndian.Uint16(p[14:])
	if out.format == wavFormatExtensible {
		// cbSize(2) validBits(2) channelMask(4) subFormat GUID(16)
		if len(p) < 40 {
			return out, fmt.Errorf("audio: truncated extensible fmt chunk: %d bytes, need at least 40", len(p))
		}
		sub := binary.LittleEndian.Uint16(p[24:])
		switch sub {
		case wavFormatPCM, wavFormatIEEEFloat:
			out.format = sub
		default:
			return out, fmt.Errorf("audio: unsupported extensible sub-format %d (only 16-bit PCM and 32-bit IEEE float are supported)", sub)
		}
	}
	if out.sampleRate == 0 {
		return out, fmt.Errorf("audio: invalid sample rate 0")
	}
	return out, nil
}
