package mcf

import (
	"errors"
	"io"
	"os"
	"sort"
	"sync"
	"unsafe"
)

// Writer builds an MCF file in a streaming fashion.
type Writer struct {
	f        *os.File
	sections []MCFSection
	seen     map[SectionType]struct{}
	closed   bool

	mu sync.Mutex
}

// NewWriter creates a new MCF writer targeting the given file.
// It truncates the file and reserves space for the header (patched in Finalise()).
func NewWriter(f *os.File) (*Writer, error) {
	if f == nil {
		return nil, errors.New("mcf: nil file")
	}

	// Make sure we always produce a file whose on-disk size matches header.FileSize.
	if err := f.Truncate(0); err != nil {
		return nil, err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	w := &Writer{
		f:    f,
		seen: make(map[SectionType]struct{}),
	}

	// Reserve header bytes (actual bytes, not a seek hole).
	hdrSize := int(unsafe.Sizeof(MCFHeader{}))
	if err := writeZeros(f, hdrSize); err != nil {
		return nil, err
	}

	// Keep the first section aligned (recommended for consumers that may cast payloads).
	if err := alignFile(f, 8); err != nil {
		return nil, err
	}

	return w, nil
}

// WriteSection writes a section payload and records it in the section table.
// Sections may be written in any order. A section type may only be written once.
func (w *Writer) WriteSection(typ SectionType, version uint32, data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("mcf: writer already finalised")
	}
	if _, ok := w.seen[typ]; ok {
		return errors.New("mcf: duplicate section type")
	}

	// Align each section start for clean mmapping and safe casting by consumers.
	if err := alignFile(w.f, 8); err != nil {
		return err
	}

	offset, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	if len(data) > 0 {
		if err := alignFile(w.f, mcfAlign); err != nil {
			return err
		}
	}

	w.sections = append(w.sections, MCFSection{
		Type:    uint32(typ),
		Version: version,
		Offset:  uint64(offset),
		Size:    uint64(len(data)),
	})
	w.seen[typ] = struct{}{}

	return nil
}

// Finalise writes the section directory and patches the header.
// After Finalise, the writer must not be used again.
func (w *Writer) Finalise() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("mcf: writer already finalised")
	}
	w.closed = true

	// Deterministic directory ordering.
	sort.Slice(w.sections, func(i, j int) bool {
		return w.sections[i].Type < w.sections[j].Type
	})

	// Align section directory start.
	if err := alignFile(w.f, 8); err != nil {
		return err
	}

	sectionDirOffset, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	// Write section directory as raw struct bytes (matches the reader's expectations).
	for i := range w.sections {
		if err := writeFull(w.f, structBytes(&w.sections[i])); err != nil {
			return err
		}
	}

	// Compute final file size and truncate to it (critical if target file was reused).
	fileSize, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := w.f.Truncate(fileSize); err != nil {
		return err
	}

	// Build header.
	var header MCFHeader
	copy(header.Magic[:], MagicMCF)
	header.Major = CurrentMajor
	header.Minor = CurrentMinor
	header.HeaderSize = uint32(unsafe.Sizeof(MCFHeader{}))
	header.SectionCount = uint32(len(w.sections))
	header.SectionDirOffset = uint64(sectionDirOffset)
	header.FileSize = uint64(fileSize)
	header.Flags = 0

	// Patch header at start of file (raw bytes, not binary.Write).
	if _, err := w.f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if err := writeFull(w.f, structBytes(&header)); err != nil {
		return err
	}

	return w.f.Sync()
}
