package mcf

import (
	"errors"
	"io"
	"os"
	"sort"
	"sync"
)

const (
	writerPadBufSize  = 4096
	writerCopyBufSize = 1 << 20 // 1 MiB
)

// Writer builds an MCF file in a streaming fashion.
//
// The writer reserves space for the header up-front and patches it during Finalise.
// Use BeginSection for large payloads (eg tensor data) to avoid buffering in memory.
type Writer struct {
	f        *os.File
	sections []MCFSection
	seen     map[SectionType]struct{}
	open     *SectionWriter
	closed   bool

	flags uint64

	padBuf  []byte
	copyBuf []byte

	mu sync.Mutex
}

// SectionWriter streams a section payload directly to the underlying file.
//
// A SectionWriter must be ended (End or Close) before any other section can be written.
// The bytes written (including any padding added via Align) are counted towards the
// section's recorded Size.
type SectionWriter struct {
	w       *Writer
	typ     SectionType
	version uint32
	start   int64
	ended   bool
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
		f:       f,
		seen:    make(map[SectionType]struct{}),
		padBuf:  make([]byte, writerPadBufSize),
		copyBuf: make([]byte, writerCopyBufSize),
	}

	// Reserve fixed header bytes (actual bytes, not a seek hole).
	if err := w.writeZeros(mcfHeaderSize); err != nil {
		return nil, err
	}

	// Keep the first section aligned (recommended for consumers that may cast payloads).
	if err := w.alignTo(mcfAlign); err != nil {
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
	if w.open != nil {
		return errors.New("mcf: section write in progress")
	}
	if _, ok := w.seen[typ]; ok {
		return errors.New("mcf: duplicate section type")
	}

	// Align each section start for clean mmapping and safe casting by consumers.
	if err := w.alignTo(mcfAlign); err != nil {
		return err
	}

	offset, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	if len(data) > 0 {
		if err := writeFull(w.f, data); err != nil {
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

// WriteSectionFromReader copies the section payload from r into the file.
//
// This is useful for moderately large payloads where you don't want to buffer
// the whole section in memory, but you don't need per-item alignment inside
// the section.
func (w *Writer) WriteSectionFromReader(typ SectionType, version uint32, r io.Reader) (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return 0, errors.New("mcf: writer already finalised")
	}
	if w.open != nil {
		return 0, errors.New("mcf: section write in progress")
	}
	if r == nil {
		return 0, errors.New("mcf: nil reader")
	}
	if _, ok := w.seen[typ]; ok {
		return 0, errors.New("mcf: duplicate section type")
	}

	if err := w.alignTo(mcfAlign); err != nil {
		return 0, err
	}

	offset, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}

	buf := w.copyBuf
	if len(buf) == 0 {
		buf = make([]byte, 32*1024)
	}
	written, err := io.CopyBuffer(w.f, r, buf)
	if err != nil {
		return 0, err
	}

	w.sections = append(w.sections, MCFSection{
		Type:    uint32(typ),
		Version: version,
		Offset:  uint64(offset),
		Size:    uint64(written),
	})
	w.seen[typ] = struct{}{}
	return uint64(written), nil
}

func (w *Writer) AddFlags(flags uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("mcf: writer already finalised")
	}
	w.flags |= flags
	return nil
}

// BeginSection begins streaming a section payload directly to the underlying file.
// The returned SectionWriter must be Ended (or Closed) before writing any other section.
func (w *Writer) BeginSection(typ SectionType, version uint32) (*SectionWriter, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil, errors.New("mcf: writer already finalised")
	}
	if w.open != nil {
		return nil, errors.New("mcf: section write in progress")
	}
	if _, ok := w.seen[typ]; ok {
		return nil, errors.New("mcf: duplicate section type")
	}

	if err := w.alignTo(mcfAlign); err != nil {
		return nil, err
	}
	start, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	sw := &SectionWriter{w: w, typ: typ, version: version, start: start}
	w.open = sw
	// Mark as seen immediately: once you start writing bytes for a section type,
	// you cannot safely “undo” it.
	w.seen[typ] = struct{}{}
	return sw, nil
}

// CurrentAbsOffset returns the current absolute file offset.
func (sw *SectionWriter) CurrentAbsOffset() (uint64, error) {
	sw.w.mu.Lock()
	defer sw.w.mu.Unlock()

	if sw.ended {
		return 0, errors.New("mcf: section writer ended")
	}
	if sw.w.open != sw {
		return 0, errors.New("mcf: section writer not active")
	}
	pos, err := sw.w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	return uint64(pos), nil
}

// BytesWritten returns the number of bytes written in this section so far.
func (sw *SectionWriter) BytesWritten() (uint64, error) {
	sw.w.mu.Lock()
	defer sw.w.mu.Unlock()

	if sw.ended {
		return 0, errors.New("mcf: section writer ended")
	}
	if sw.w.open != sw {
		return 0, errors.New("mcf: section writer not active")
	}
	pos, err := sw.w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	if pos < sw.start {
		return 0, errors.New("mcf: invalid file position")
	}
	return uint64(pos - sw.start), nil
}

// TruncateAbs truncates the underlying file back to the given absolute offset,
// and seeks the current file position to that offset.
//
// This is intended for pack-time rollback (eg: dedup). It is only valid while
// the SectionWriter is active and not ended.
func (sw *SectionWriter) TruncateAbs(abs uint64) error {
	sw.w.mu.Lock()
	defer sw.w.mu.Unlock()

	if sw.ended {
		return errors.New("mcf: section writer ended")
	}
	if sw.w.open != sw {
		return errors.New("mcf: section writer not active")
	}

	if abs < uint64(sw.start) {
		return errors.New("mcf: truncate before section start")
	}

	pos, err := sw.w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if abs > uint64(pos) {
		return errors.New("mcf: truncate past current position")
	}

	if err := sw.w.f.Truncate(int64(abs)); err != nil {
		return err
	}
	_, err = sw.w.f.Seek(int64(abs), io.SeekStart)
	return err
}

// Align writes zero padding until the underlying file position is aligned to n bytes.
// This is useful for aligning individual tensor payloads within a TensorData section.
func (sw *SectionWriter) Align(n int) error {
	sw.w.mu.Lock()
	defer sw.w.mu.Unlock()

	if sw.ended {
		return errors.New("mcf: section writer ended")
	}
	if sw.w.open != sw {
		return errors.New("mcf: section writer not active")
	}
	return sw.w.alignTo(int64(n))
}

// Write streams p into the underlying file.
func (sw *SectionWriter) Write(p []byte) (int, error) {
	sw.w.mu.Lock()
	defer sw.w.mu.Unlock()

	if sw.ended {
		return 0, errors.New("mcf: section writer ended")
	}
	if sw.w.open != sw {
		return 0, errors.New("mcf: section writer not active")
	}
	if len(p) == 0 {
		return 0, nil
	}
	if err := writeFull(sw.w.f, p); err != nil {
		return 0, err
	}
	return len(p), nil
}

// End finalises the section and records it in the section directory.
func (sw *SectionWriter) End() error {
	sw.w.mu.Lock()
	defer sw.w.mu.Unlock()

	if sw.ended {
		return errors.New("mcf: section writer already ended")
	}
	if sw.w.open != sw {
		return errors.New("mcf: section writer not active")
	}

	pos, err := sw.w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if pos < sw.start {
		return errors.New("mcf: invalid file position")
	}

	sw.w.sections = append(sw.w.sections, MCFSection{
		Type:    uint32(sw.typ),
		Version: sw.version,
		Offset:  uint64(sw.start),
		Size:    uint64(pos - sw.start),
	})

	sw.w.open = nil
	sw.ended = true
	return nil
}

// Close is an alias for End, allowing use with defer.
func (sw *SectionWriter) Close() error { return sw.End() }

// Finalise writes the section directory and patches the header.
// After Finalise, the writer must not be used again.
func (w *Writer) Finalise() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("mcf: writer already finalised")
	}
	if w.open != nil {
		return errors.New("mcf: section write in progress")
	}
	w.closed = true

	// Deterministic directory ordering.
	sort.Slice(w.sections, func(i, j int) bool {
		return w.sections[i].Type < w.sections[j].Type
	})

	// Align section directory start.
	if err := w.alignTo(mcfAlign); err != nil {
		return err
	}

	sectionDirOffset, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	// Write section directory using explicit little-endian encoding.
	var secBuf [mcfSectionSize]byte
	for i := range w.sections {
		if !encodeSection(secBuf[:], w.sections[i]) {
			return errors.New("mcf: encode section failed")
		}
		if err := writeFull(w.f, secBuf[:]); err != nil {
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
	header.HeaderSize = mcfHeaderSize
	header.SectionCount = uint32(len(w.sections))
	header.SectionDirOffset = uint64(sectionDirOffset)
	header.FileSize = uint64(fileSize)
	header.Flags = w.flags

	// Patch header at start of file
	if _, err := w.f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	var hdrBuf [mcfHeaderSize]byte
	if !encodeHeader(hdrBuf[:], header) {
		return errors.New("mcf: encode header failed")
	}
	if err := writeFull(w.f, hdrBuf[:]); err != nil {
		return err
	}

	return w.f.Sync()
}

func (w *Writer) alignTo(n int64) error {
	if n <= 1 {
		return nil
	}
	pos, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	mod := pos % n
	if mod == 0 {
		return nil
	}
	pad := int(n - mod)
	return w.writeZeros(pad)
}

func (w *Writer) writeZeros(n int) error {
	if n <= 0 {
		return nil
	}
	buf := w.padBuf
	if len(buf) == 0 {
		buf = make([]byte, 4096)
	}
	for n > 0 {
		toWrite := min(n, len(buf))
		if err := writeFull(w.f, buf[:toWrite]); err != nil {
			return err
		}
		n -= toWrite
	}
	return nil
}
