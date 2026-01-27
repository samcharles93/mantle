package mcf

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SafetensorsIndexFile is the standard Hugging Face sharded safetensors index filename.
const SafetensorsIndexFile = "model.safetensors.index.json"

const (
	// A defensive cap; real-world headers are typically in the KBs.
	safetensorsMaxHeaderSize = 256 << 20 // 256 MiB
)

// SafetensorsTensorInfo describes a tensor payload within a single safetensors file.
// Start/End are absolute file offsets (End is exclusive).
//
// DType values follow the safetensors spec, e.g. "F32", "F16", "BF16", "I8", "U8", ...
// Shape is stored as int64 to avoid surprising overflow.
//
// Note: safetensors uses byte offsets relative to the data region; we convert to absolute.
type SafetensorsTensorInfo struct {
	DType string
	Shape []int64
	Start int64
	End   int64
}

func (ti SafetensorsTensorInfo) Size() int64 { return ti.End - ti.Start }

type safetensorsTensorHeader struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// SafetensorsFile provides random access to tensors inside a single safetensors file.
//
// Keep the file open while copying tensors to avoid repeated open/close overhead.
// os.File ReadAt is safe for concurrent use.
type SafetensorsFile struct {
	Path      string
	f         *os.File
	dataStart int64
	Tensors   map[string]SafetensorsTensorInfo

	// Raw metadata (optional, may be nil).
	Metadata json.RawMessage
}

// OpenSafetensorsFile opens and parses a single .safetensors file.
func OpenSafetensorsFile(path string) (*SafetensorsFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	sz, err := fileSize(f)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	if sz < 8 {
		_ = f.Close()
		return nil, fmt.Errorf("safetensors: file too small: %s", path)
	}

	headerLenU64, err := readU64(f)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	if headerLenU64 > safetensorsMaxHeaderSize {
		_ = f.Close()
		return nil, fmt.Errorf("safetensors: header too large (%d bytes): %s", headerLenU64, path)
	}
	headerLen := int64(headerLenU64)
	if 8+headerLen > sz {
		_ = f.Close()
		return nil, fmt.Errorf("safetensors: header exceeds file size: %s", path)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		_ = f.Close()
		return nil, err
	}

	// Header is a JSON map where keys are tensor names (plus optional "__metadata__").
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	dataStart := int64(8) + headerLen

	// Optional metadata.
	meta := raw["__metadata__"]
	delete(raw, "__metadata__")

	tensors := make(map[string]SafetensorsTensorInfo, len(raw))
	for name, msg := range raw {
		var th safetensorsTensorHeader
		if err := json.Unmarshal(msg, &th); err != nil {
			_ = f.Close()
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		if len(th.DataOffsets) != 2 {
			_ = f.Close()
			return nil, fmt.Errorf("safetensors: tensor %q: invalid data_offsets", name)
		}

		startRel, endRel := th.DataOffsets[0], th.DataOffsets[1]
		if startRel < 0 || endRel < 0 || endRel < startRel {
			_ = f.Close()
			return nil, fmt.Errorf("safetensors: tensor %q: invalid offsets", name)
		}

		startAbs := dataStart + startRel
		endAbs := dataStart + endRel
		if startAbs < dataStart || endAbs < startAbs || endAbs > sz {
			_ = f.Close()
			return nil, fmt.Errorf("safetensors: tensor %q: out-of-bounds data range", name)
		}

		if len(th.Shape) == 0 {
			_ = f.Close()
			return nil, fmt.Errorf("safetensors: tensor %q: empty shape", name)
		}
		for _, d := range th.Shape {
			if d <= 0 {
				_ = f.Close()
				return nil, fmt.Errorf("safetensors: tensor %q: invalid dim %d", name, d)
			}
		}

		tensors[name] = SafetensorsTensorInfo{
			DType: th.DType,
			Shape: th.Shape,
			Start: startAbs,
			End:   endAbs,
		}
	}

	return &SafetensorsFile{
		Path:      path,
		f:         f,
		dataStart: dataStart,
		Tensors:   tensors,
		Metadata:  meta,
	}, nil
}

func (sf *SafetensorsFile) Close() error {
	if sf == nil || sf.f == nil {
		return nil
	}
	err := sf.f.Close()
	sf.f = nil
	return err
}

func (sf *SafetensorsFile) Tensor(name string) (SafetensorsTensorInfo, bool) {
	if sf == nil {
		return SafetensorsTensorInfo{}, false
	}
	ti, ok := sf.Tensors[name]
	return ti, ok
}

func (sf *SafetensorsFile) SortedTensorNames() []string {
	if sf == nil {
		return nil
	}
	out := make([]string, 0, len(sf.Tensors))
	for name := range sf.Tensors {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// TensorReader returns a reader over the raw tensor bytes.
func (sf *SafetensorsFile) TensorReader(name string) (*io.SectionReader, SafetensorsTensorInfo, error) {
	if sf == nil || sf.f == nil {
		return nil, SafetensorsTensorInfo{}, errors.New("safetensors: file closed")
	}
	ti, ok := sf.Tensors[name]
	if !ok {
		return nil, SafetensorsTensorInfo{}, fmt.Errorf("safetensors: tensor not found: %s", name)
	}
	if ti.End < ti.Start {
		return nil, SafetensorsTensorInfo{}, fmt.Errorf("safetensors: tensor %q: invalid offsets", name)
	}
	return io.NewSectionReader(sf.f, ti.Start, ti.End-ti.Start), ti, nil
}

// CopyTensorTo streams the raw tensor bytes into dst.
func (sf *SafetensorsFile) CopyTensorTo(dst io.Writer, name string) (int64, SafetensorsTensorInfo, error) {
	r, ti, err := sf.TensorReader(name)
	if err != nil {
		return 0, SafetensorsTensorInfo{}, err
	}
	n, err := io.Copy(dst, r)
	if err != nil {
		return n, SafetensorsTensorInfo{}, fmt.Errorf("safetensors: copy tensor %q: %w", name, err)
	}
	return n, ti, nil
}

// CopyTensorToBuffer streams the raw tensor bytes into dst using the provided buffer.
// If buf is nil, it behaves like CopyTensorTo.
func (sf *SafetensorsFile) CopyTensorToBuffer(dst io.Writer, name string, buf []byte) (int64, SafetensorsTensorInfo, error) {
	r, ti, err := sf.TensorReader(name)
	if err != nil {
		return 0, SafetensorsTensorInfo{}, err
	}
	if buf == nil {
		return sf.CopyTensorTo(dst, name)
	}
	n, err := io.CopyBuffer(dst, r, buf)
	if err != nil {
		return n, SafetensorsTensorInfo{}, fmt.Errorf("safetensors: copy tensor %q: %w", name, err)
	}
	return n, ti, nil
}

// ReadTensor reads the raw tensor bytes into memory.
func (sf *SafetensorsFile) ReadTensor(name string) ([]byte, SafetensorsTensorInfo, error) {
	r, ti, err := sf.TensorReader(name)
	if err != nil {
		return nil, SafetensorsTensorInfo{}, err
	}
	sz := ti.End - ti.Start
	if sz < 0 {
		return nil, SafetensorsTensorInfo{}, fmt.Errorf("safetensors: tensor %q: invalid size", name)
	}
	buf := make([]byte, sz)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, SafetensorsTensorInfo{}, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
	}
	return buf, ti, nil
}

// SafetensorsTensorRef points to a tensor within a sharded model.
type SafetensorsTensorRef struct {
	Name string
	File *SafetensorsFile
	Info SafetensorsTensorInfo
}

// SafetensorsModel provides a unified view of a single safetensors file or a sharded model
// described by model.safetensors.index.json.
type SafetensorsModel struct {
	BasePath string
	Files    map[string]*SafetensorsFile     // key: shard filename (relative)
	Tensors  map[string]SafetensorsTensorRef // key: tensor name
	order    []string                        // cached sorted tensor names
}

func (m *SafetensorsModel) Close() error {
	if m == nil {
		return nil
	}
	var first error
	for _, f := range m.Files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (m *SafetensorsModel) Tensor(name string) (SafetensorsTensorRef, bool) {
	if m == nil {
		return SafetensorsTensorRef{}, false
	}
	tr, ok := m.Tensors[name]
	return tr, ok
}

func (m *SafetensorsModel) SortedTensorNames() []string {
	if m == nil {
		return nil
	}
	if m.order != nil {
		out := make([]string, len(m.order))
		copy(out, m.order)
		return out
	}
	out := make([]string, 0, len(m.Tensors))
	for name := range m.Tensors {
		out = append(out, name)
	}
	sort.Strings(out)
	m.order = out

	out2 := make([]string, len(out))
	copy(out2, out)
	return out2
}

func (m *SafetensorsModel) TensorReader(name string) (*io.SectionReader, SafetensorsTensorRef, error) {
	tr, ok := m.Tensor(name)
	if !ok {
		return nil, SafetensorsTensorRef{}, fmt.Errorf("safetensors: tensor not found: %s", name)
	}
	r, _, err := tr.File.TensorReader(name)
	if err != nil {
		return nil, SafetensorsTensorRef{}, err
	}
	return r, tr, nil
}

func (m *SafetensorsModel) CopyTensorTo(dst io.Writer, name string) (int64, SafetensorsTensorRef, error) {
	r, tr, err := m.TensorReader(name)
	if err != nil {
		return 0, SafetensorsTensorRef{}, err
	}
	n, err := io.Copy(dst, r)
	if err != nil {
		return n, SafetensorsTensorRef{}, fmt.Errorf("safetensors: copy tensor %q: %w", name, err)
	}
	return n, tr, nil
}

// CopyTensorToBuffer streams the raw tensor bytes into dst using the provided buffer.
// If buf is nil, it behaves like CopyTensorTo.
func (m *SafetensorsModel) CopyTensorToBuffer(dst io.Writer, name string, buf []byte) (int64, SafetensorsTensorRef, error) {
	r, tr, err := m.TensorReader(name)
	if err != nil {
		return 0, SafetensorsTensorRef{}, err
	}
	if buf == nil {
		return m.CopyTensorTo(dst, name)
	}
	n, err := io.CopyBuffer(dst, r, buf)
	if err != nil {
		return n, SafetensorsTensorRef{}, fmt.Errorf("safetensors: copy tensor %q: %w", name, err)
	}
	return n, tr, nil
}

// OpenSafetensorsModel opens either:
//   - a single .safetensors file
//   - a directory containing SafetensorsIndexFile
//   - a directory containing one or more *.safetensors (fallback merge)
func OpenSafetensorsModel(path string) (*SafetensorsModel, error) {
	if path == "" {
		return nil, errors.New("safetensors: empty path")
	}

	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	if !st.IsDir() {
		// Single file.
		if !strings.HasSuffix(strings.ToLower(path), ".safetensors") {
			return nil, fmt.Errorf("safetensors: expected .safetensors file: %s", path)
		}
		sf, err := OpenSafetensorsFile(path)
		if err != nil {
			return nil, err
		}
		m := &SafetensorsModel{
			BasePath: path,
			Files:    map[string]*SafetensorsFile{filepath.Base(path): sf},
			Tensors:  make(map[string]SafetensorsTensorRef, len(sf.Tensors)),
		}
		for name, info := range sf.Tensors {
			m.Tensors[name] = SafetensorsTensorRef{Name: name, File: sf, Info: info}
		}
		return m, nil
	}

	// Directory:
	// 1) Prefer the standard HF shard index if present.
	idxPath := filepath.Join(path, SafetensorsIndexFile)
	if _, err := os.Stat(idxPath); err == nil {
		return openSafetensorsIndexModel(path, idxPath)
	}

	// 2) Otherwise require exactly one *.safetensors file in the directory.
	single, err := findSingleSafetensorsInDir(path)
	if err != nil {
		return nil, err
	}
	return OpenSafetensorsModel(single)
}

type safetensorsIndex struct {
	Metadata  map[string]any    `json:"metadata,omitempty"`
	WeightMap map[string]string `json:"weight_map"`
}

func findSingleSafetensorsInDir(dir string) (string, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var matches []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(strings.ToLower(name), ".safetensors") {
			matches = append(matches, filepath.Join(dir, name))
		}
	}
	sort.Strings(matches)
	switch len(matches) {
	case 0:
		return "", fmt.Errorf("safetensors: no .safetensors file and no %s in directory: %s", SafetensorsIndexFile, dir)
	case 1:
		return matches[0], nil
	default:
		return "", fmt.Errorf("safetensors: found %d .safetensors files but no %s in directory: %s", len(matches), SafetensorsIndexFile, dir)
	}
}

func openSafetensorsIndexModel(dir, idxPath string) (*SafetensorsModel, error) {
	b, err := os.ReadFile(idxPath)
	if err != nil {
		return nil, err
	}
	var idx safetensorsIndex
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("safetensors: parse index: %w", err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("safetensors: index has empty weight_map: %s", idxPath)
	}

	// Open each shard referenced in weight_map.
	files := make(map[string]*SafetensorsFile)
	for _, shard := range idx.WeightMap {
		if shard == "" {
			return nil, fmt.Errorf("safetensors: invalid shard name in weight_map")
		}
		if _, ok := files[shard]; ok {
			continue
		}
		full := filepath.Join(dir, shard)
		sf, err := OpenSafetensorsFile(full)
		if err != nil {
			for _, f := range files {
				_ = f.Close()
			}
			return nil, err
		}
		files[shard] = sf
	}

	tensors := make(map[string]SafetensorsTensorRef, len(idx.WeightMap))
	for name, shard := range idx.WeightMap {
		sf := files[shard]
		if sf == nil {
			for _, f := range files {
				_ = f.Close()
			}
			return nil, fmt.Errorf("safetensors: shard %q missing for tensor %q", shard, name)
		}
		info, ok := sf.Tensor(name)
		if !ok {
			for _, f := range files {
				_ = f.Close()
			}
			return nil, fmt.Errorf("safetensors: tensor %q not found in shard %q", name, shard)
		}
		if _, exists := tensors[name]; exists {
			for _, f := range files {
				_ = f.Close()
			}
			return nil, fmt.Errorf("safetensors: duplicate tensor name in weight_map: %q", name)
		}
		tensors[name] = SafetensorsTensorRef{Name: name, File: sf, Info: info}
	}

	return &SafetensorsModel{
		BasePath: dir,
		Files:    files,
		Tensors:  tensors,
	}, nil
}

func fileSize(f *os.File) (int64, error) {
	st, err := f.Stat()
	if err != nil {
		return 0, err
	}
	return st.Size(), nil
}

func readU64(r io.Reader) (uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}
