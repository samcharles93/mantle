package audio

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// buildWAV assembles a RIFF/WAVE file the way common writers do, including an
// optional odd-sized junk chunk before "data" to exercise chunk padding.
func buildWAV(t *testing.T, format, channels uint16, sampleRate uint32, bits uint16, payload []byte, junkOdd bool) []byte {
	t.Helper()
	out := []byte("RIFF")
	size := uint32(4 + 8 + 16 + 8 + len(payload))
	if junkOdd {
		size += 8 + 3 + 1 // LIST chunk, odd payload 3 bytes + pad
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], size)
	out = append(out, b[:]...)
	out = append(out, "WAVE"...)
	out = append(out, "fmt "...)
	binary.LittleEndian.PutUint32(b[:], 16)
	out = append(out, b[:]...)
	var fmt [16]byte
	binary.LittleEndian.PutUint16(fmt[0:], format)
	binary.LittleEndian.PutUint16(fmt[2:], channels)
	binary.LittleEndian.PutUint32(fmt[4:], sampleRate)
	blockAlign := channels * bits / 8
	binary.LittleEndian.PutUint32(fmt[8:], sampleRate*uint32(blockAlign))
	binary.LittleEndian.PutUint16(fmt[12:], blockAlign)
	binary.LittleEndian.PutUint16(fmt[14:], bits)
	out = append(out, fmt[:]...)
	if junkOdd {
		out = append(out, "LIST"...)
		binary.LittleEndian.PutUint32(b[:], 3)
		out = append(out, b[:]...)
		out = append(out, 'a', 'b', 'c', 0) // 3 payload bytes + 1 pad byte
	}
	out = append(out, "data"...)
	binary.LittleEndian.PutUint32(b[:], uint32(len(payload)))
	out = append(out, b[:]...)
	out = append(out, payload...)
	return out
}

func pcm16Payload(vals ...int16) []byte {
	b := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

func float32Payload(vals ...float32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

func TestDecodeWAVPCM16Mono(t *testing.T) {
	raw := buildWAV(t, 1, 1, 16000, 16, pcm16Payload(0, 16384, -16384, 32767, -32768), false)
	samples, sr, err := DecodeWAV(raw)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}
	if sr != 16000 {
		t.Fatalf("sample rate = %d, want 16000", sr)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768.0, -1}
	if len(samples) != len(want) {
		t.Fatalf("got %d samples, want %d", len(samples), len(want))
	}
	for i := range want {
		if samples[i] != want[i] {
			t.Fatalf("sample %d = %v, want %v", i, samples[i], want[i])
		}
	}
}

func TestDecodeWAVPCM16StereoDownmix(t *testing.T) {
	// interleaved L,R pairs: (32767, 0), (-32768, 0) -> mono (l+r)/2
	payload := pcm16Payload(32767, 0, -32768, 0)
	raw := buildWAV(t, 1, 2, 44100, 16, payload, false)
	samples, sr, err := DecodeWAV(raw)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}
	if sr != 44100 {
		t.Fatalf("sample rate = %d, want 44100", sr)
	}
	want := []float32{32767.0 / 65536.0, -0.5}
	if len(samples) != 2 {
		t.Fatalf("got %d samples, want 2", len(samples))
	}
	for i := range want {
		if samples[i] != want[i] {
			t.Fatalf("sample %d = %v, want %v", i, samples[i], want[i])
		}
	}
}

func TestDecodeWAVFloat32(t *testing.T) {
	for _, ch := range []uint16{1, 2} {
		var payload []byte
		if ch == 1 {
			payload = float32Payload(0.25, -0.75, 1.0)
		} else {
			payload = float32Payload(0.5, -0.5, -1.0, 1.0) // L,R,L,R -> 0, 0
		}
		raw := buildWAV(t, 3, ch, 24000, 32, payload, false)
		samples, sr, err := DecodeWAV(raw)
		if err != nil {
			t.Fatalf("channels %d: DecodeWAV: %v", ch, err)
		}
		if sr != 24000 {
			t.Fatalf("channels %d: sample rate = %d, want 24000", ch, sr)
		}
		if ch == 1 {
			want := []float32{0.25, -0.75, 1.0}
			if len(samples) != len(want) {
				t.Fatalf("got %d samples, want %d", len(samples), len(want))
			}
			for i := range want {
				if samples[i] != want[i] {
					t.Fatalf("sample %d = %v, want %v", i, samples[i], want[i])
				}
			}
		} else {
			if len(samples) != 2 || samples[0] != 0 || samples[1] != 0 {
				t.Fatalf("stereo float downmix = %v, want [0 0]", samples)
			}
		}
	}
}

func TestDecodeWAVSkipsOddJunkChunk(t *testing.T) {
	raw := buildWAV(t, 1, 1, 16000, 16, pcm16Payload(16384), true)
	samples, _, err := DecodeWAV(raw)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}
	if len(samples) != 1 || samples[0] != 0.5 {
		t.Fatalf("samples = %v, want [0.5]", samples)
	}
}

func TestDecodeWAVErrors(t *testing.T) {
	good := buildWAV(t, 1, 1, 16000, 16, pcm16Payload(0), false)
	truncatedData := good[:len(good)-1] // data chunk promises 2 bytes, only 1 present
	badWave := append([]byte(nil), good...)
	copy(badWave[8:], "WAVX")
	cases := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"short header", []byte("RIFF")},
		{"bad riff magic", append([]byte("RIFX"), good[4:]...)},
		{"bad wave magic", badWave},
		{"truncated fmt", good[:16]},
		{"truncated data", truncatedData},
		{"alaw codec", buildWAV(t, 6, 1, 8000, 8, []byte{0x55}, false)},
		{"mp3 codec", buildWAV(t, 85, 1, 44100, 0, nil, false)},
		{"pcm 8-bit", buildWAV(t, 1, 1, 16000, 8, []byte{128}, false)},
		{"pcm 24-bit", buildWAV(t, 1, 1, 16000, 24, []byte{0, 0, 0}, false)},
		{"float64", buildWAV(t, 3, 1, 16000, 64, make([]byte, 8), false)},
		{"three channels", buildWAV(t, 1, 3, 16000, 16, make([]byte, 6), false)},
		{"zero channels", buildWAV(t, 1, 0, 16000, 16, nil, false)},
		{"no data chunk", []byte("RIFF\x14\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00")},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			samples, sr, err := DecodeWAV(tc.data)
			if err == nil {
				t.Fatalf("DecodeWAV succeeded with %d samples, sr %d; want error", len(samples), sr)
			}
		})
	}
}

func TestDecodeWAVRoundTripPCM16(t *testing.T) {
	// float -> pcm16 -> wav -> decode stays within half a quantization step.
	want := make([]float32, 1000)
	for i := range want {
		want[i] = float32(math.Sin(float64(i)/37.0)) * 0.9
	}
	payload := make([]byte, 2*len(want))
	for i, v := range want {
		q := math.Round(float64(v) * 32768.0)
		if q > 32767 {
			q = 32767
		}
		if q < -32768 {
			q = -32768
		}
		binary.LittleEndian.PutUint16(payload[2*i:], uint16(int16(q)))
	}
	raw := buildWAV(t, 1, 1, 16000, 16, payload, false)
	got, _, err := DecodeWAV(raw)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d samples, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1.0/32768.0+1e-7 {
			t.Fatalf("sample %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// TestDecodeWAVHugeChunkSize pins the overflow guard at wav.go: a chunk whose
// declared size is near 2^32 (e.g. 0xFFFFFFFF) must be rejected as truncated
// via the uint64 bounds check, never panic or allocate. Review follow-up P2
// from review-omni-core-rv-omni-b/c: the path was fuzz-verified out-of-tree but
// not pinned by a committed named test.
func TestDecodeWAVHugeChunkSize(t *testing.T) {
	t.Parallel()

	withChunkSize := func(id string, size uint32) []byte {
		out := []byte("RIFF")
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], 4+8+16+8+size)
		out = append(out, b[:]...)
		out = append(out, "WAVE"...)
		out = append(out, "fmt "...)
		binary.LittleEndian.PutUint32(b[:], 16)
		out = append(out, b[:]...)
		var fmt [16]byte
		binary.LittleEndian.PutUint16(fmt[0:], 1) // PCM
		binary.LittleEndian.PutUint16(fmt[2:], 1) // mono
		binary.LittleEndian.PutUint32(fmt[4:], 16000)
		binary.LittleEndian.PutUint32(fmt[8:], 32000)
		binary.LittleEndian.PutUint16(fmt[12:], 2)
		binary.LittleEndian.PutUint16(fmt[14:], 16)
		out = append(out, fmt[:]...)
		out = append(out, id...)
		binary.LittleEndian.PutUint32(b[:], size)
		out = append(out, b[:]...)
		return out // no payload follows
	}

	cases := []struct {
		name string
		data []byte
	}{
		{"data size 0xFFFFFFFF", withChunkSize("data", 0xFFFFFFFF)},
		{"data size 0x80000000", withChunkSize("data", 0x80000000)},
		{"data size 0x7FFFFFFF", withChunkSize("data", 0x7FFFFFFF)},
		{"fmt size 0xFFFFFFFF", func() []byte {
			out := []byte("RIFF")
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], 0xFFFFFFFF)
			out = append(out, b[:]...)
			out = append(out, "WAVE"...)
			out = append(out, "fmt "...)
			binary.LittleEndian.PutUint32(b[:], 0xFFFFFFFF)
			out = append(out, b[:]...)
			return out
		}()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			samples, sr, err := DecodeWAV(tc.data)
			if err == nil {
				t.Fatalf("expected error, got %d samples @ %d Hz", len(samples), sr)
			}
			if !strings.Contains(err.Error(), "truncated") {
				t.Fatalf("expected truncated-chunk error, got %v", err)
			}
		})
	}
}
