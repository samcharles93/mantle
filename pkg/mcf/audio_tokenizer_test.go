package mcf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAudioTokenizerSectionRoundTrip(t *testing.T) {
	t.Parallel()
	config := []byte(`{"model_type":"liquid_codec","fsq_levels":[8,7,6,6]}`)
	st := []byte("\x08\x00\x00\x00\x00\x00\x00\x00{}fake safetensors payload")
	blob := EncodeAudioTokenizerSection(config, st)
	if len(blob) != audioTokenizerHeaderSize+len(config)+len(st) {
		t.Fatalf("blob size %d, want header+%d+%d", len(blob), len(config), len(st))
	}
	if got := binary.LittleEndian.Uint32(blob[0:4]); got != audioTokenizerMagic {
		t.Fatalf("magic %#x, want %#x", got, audioTokenizerMagic)
	}
	if got := binary.LittleEndian.Uint32(blob[4:8]); got != AudioTokenizerSectionVersion {
		t.Fatalf("version %d, want %d", got, AudioTokenizerSectionVersion)
	}
	cfgOut, stOut, err := DecodeAudioTokenizerSection(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(cfgOut, config) || !bytes.Equal(stOut, st) {
		t.Fatalf("round trip mismatch: config %d bytes, st %d bytes", len(cfgOut), len(stOut))
	}
}

func TestDecodeAudioTokenizerSectionErrors(t *testing.T) {
	t.Parallel()
	config := []byte(`{}`)
	st := []byte("st-bytes")
	good := EncodeAudioTokenizerSection(config, st)

	cases := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"short header", good[:audioTokenizerHeaderSize-1]},
		{"bad magic", func() []byte {
			b := append([]byte(nil), good...)
			binary.LittleEndian.PutUint32(b[0:4], 0xDEADBEEF)
			return b
		}()},
		{"bad version", func() []byte { b := append([]byte(nil), good...); binary.LittleEndian.PutUint32(b[4:8], 99); return b }()},
		{"truncated config", good[:audioTokenizerHeaderSize+len(config)-1]},
		{"truncated safetensors", good[:len(good)-1]},
		{"huge config size", func() []byte {
			b := append([]byte(nil), good...)
			binary.LittleEndian.PutUint64(b[8:16], 0xFFFFFFFFFFFFFFF0)
			return b
		}()},
		{"huge safetensors size", func() []byte {
			b := append([]byte(nil), good...)
			binary.LittleEndian.PutUint64(b[16:24], 1<<62)
			return b
		}()},
		{"sizes overflow int", func() []byte {
			b := append([]byte(nil), good...)
			binary.LittleEndian.PutUint64(b[8:16], 1<<62)
			binary.LittleEndian.PutUint64(b[16:24], 1<<62)
			return b
		}()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := DecodeAudioTokenizerSection(tc.data)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
		})
	}
}

func TestAudioTokenizerPayloadAbsent(t *testing.T) {
	t.Parallel()
	// A minimal in-memory MCF without section 0x0107 must yield (nil, nil, nil).
	f := &File{Data: []byte("x"), Header: &MCFHeader{}, Sections: nil}
	cfg, st, err := f.AudioTokenizerPayload()
	if err != nil || cfg != nil || st != nil {
		t.Fatalf("absent section: cfg=%d st=%d err=%v, want nil/nil/nil", len(cfg), len(st), err)
	}
}

func TestAudioTokenizerPayloadCorrupt(t *testing.T) {
	t.Parallel()
	bad := []byte("not an atk1 blob at all........")
	sec := MCFSection{Type: uint32(SectionAudioTokenizer), Version: AudioTokenizerSectionVersion, Offset: 0, Size: uint64(len(bad))}
	f := &File{Data: bad, Header: &MCFHeader{}, Sections: []MCFSection{sec}}
	_, _, err := f.AudioTokenizerPayload()
	if err == nil || !strings.Contains(err.Error(), "audio_tokenizer") {
		t.Fatalf("want audio_tokenizer decode error, got %v", err)
	}
}

// writeTestSafetensors writes a minimal single-tensor safetensors file.
func writeTestSafetensors(t *testing.T, path, name string, payload []byte) {
	t.Helper()
	header := map[string]any{
		name: map[string]any{
			"dtype":        "F32",
			"shape":        []int{len(payload) / 4},
			"data_offsets": []int{0, len(payload)},
		},
	}
	headerRaw, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create safetensors: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerRaw))); err != nil {
		t.Fatalf("write header len: %v", err)
	}
	if _, err := f.Write(headerRaw); err != nil {
		t.Fatalf("write header: %v", err)
	}
	if _, err := f.Write(payload); err != nil {
		t.Fatalf("write payload: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// TestPackAudioTokenizerSection packs a tiny model dir with an audio_tokenizer
// codec directory and checks section 0x0107 round-trips through a reopened
// container; a dir without audio_tokenizer must pack without the section.
func TestPackAudioTokenizerSection(t *testing.T) {
	t.Parallel()

	build := func(t *testing.T, withCodec bool) string {
		dir := t.TempDir()
		writeTestSafetensors(t, filepath.Join(dir, "model.safetensors"), "w", make([]byte, 16))
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"x"}`), 0o644); err != nil {
			t.Fatalf("write config: %v", err)
		}
		if withCodec {
			atkDir := filepath.Join(dir, "audio_tokenizer")
			if err := os.Mkdir(atkDir, 0o755); err != nil {
				t.Fatalf("mkdir: %v", err)
			}
			if err := os.WriteFile(filepath.Join(atkDir, "config.json"), []byte(`{"model_type":"liquid_codec"}`), 0o644); err != nil {
				t.Fatalf("write codec config: %v", err)
			}
			writeTestSafetensors(t, filepath.Join(atkDir, "model.safetensors"), "c", make([]byte, 8))
		}
		out := filepath.Join(t.TempDir(), "out.mcf")
		err := Pack(PackOptions{InputDir: dir, OutputPath: out, IncludeResources: true, Dedup: false})
		if err != nil {
			t.Fatalf("Pack: %v", err)
		}
		return out
	}

	t.Run("with codec", func(t *testing.T) {
		t.Parallel()
		out := build(t, true)
		f, err := Open(out)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		defer func() { _ = f.Close() }()
		sec := f.Section(SectionAudioTokenizer)
		if sec == nil {
			t.Fatalf("section 0x0107 missing after pack")
		}
		if sec.Version != AudioTokenizerSectionVersion {
			t.Fatalf("section version %d, want %d", sec.Version, AudioTokenizerSectionVersion)
		}
		cfg, st, err := f.AudioTokenizerPayload()
		if err != nil {
			t.Fatalf("AudioTokenizerPayload: %v", err)
		}
		if string(cfg) != `{"model_type":"liquid_codec"}` {
			t.Fatalf("config bytes %q", cfg)
		}
		if len(st) != 8+8+len(`{"c":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`) {
			t.Fatalf("safetensors bytes %d", len(st))
		}
	})

	t.Run("without codec", func(t *testing.T) {
		t.Parallel()
		out := build(t, false)
		f, err := Open(out)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		defer func() { _ = f.Close() }()
		if sec := f.Section(SectionAudioTokenizer); sec != nil {
			t.Fatalf("section 0x0107 present for codec-less model")
		}
		cfg, st, err := f.AudioTokenizerPayload()
		if err != nil || cfg != nil || st != nil {
			t.Fatalf("want nil payload, got %d/%d err=%v", len(cfg), len(st), err)
		}
	})

	t.Run("partial codec dir errors", func(t *testing.T) {
		t.Parallel()
		dir := t.TempDir()
		writeTestSafetensors(t, filepath.Join(dir, "model.safetensors"), "w", make([]byte, 16))
		atkDir := filepath.Join(dir, "audio_tokenizer")
		if err := os.Mkdir(atkDir, 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		// config.json present, model.safetensors missing
		if err := os.WriteFile(filepath.Join(atkDir, "config.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		out := filepath.Join(t.TempDir(), "out.mcf")
		err := Pack(PackOptions{InputDir: dir, OutputPath: out, IncludeResources: true})
		if err == nil || !strings.Contains(err.Error(), "audio_tokenizer") {
			t.Fatalf("want audio_tokenizer pack error, got %v", err)
		}
	})
}
