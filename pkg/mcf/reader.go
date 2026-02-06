package mcf

import (
	"fmt"
	"io"
	"os"

	"golang.org/x/sys/unix"
)

type File struct {
	Data     []byte
	Header   *MCFHeader
	Sections []MCFSection
	mmapped  bool
}

// Open maps an MCF file read-only and validates its structure.
// If mmap is unavailable, it falls back to ReadAt-based loading.
// The returned file must be closed to release any mapping.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size64 := stat.Size()
	if size64 < 0 {
		return nil, ErrCorruptFile
	}
	if size64 > int64(int(^uint(0)>>1)) {
		// cannot index this file safely as []byte on this architecture.
		return nil, ErrCorruptFile
	}
	size := int(size64)
	if size < mcfHeaderSize {
		return nil, ErrCorruptFile
	}

	// Prefer mmap where available for zero-copy section slices.
	data, err := unix.Mmap(
		int(f.Fd()),
		0,
		size,
		unix.PROT_READ,
		unix.MAP_SHARED,
	)
	if err == nil {
		mf, parseErr := parseFileData(data, true)
		if parseErr != nil {
			_ = unix.Munmap(data)
			return nil, parseErr
		}
		return mf, nil
	}

	// Fallback path that does not require mmap support.
	data, err = readAllAt(f, size)
	if err != nil {
		return nil, err
	}
	return parseFileData(data, false)
}

// OpenReaderAt loads and validates an MCF from a random-access reader without mmap.
func OpenReaderAt(r io.ReaderAt, size int64) (*File, error) {
	if size < 0 || size > int64(int(^uint(0)>>1)) {
		return nil, ErrCorruptFile
	}
	data, err := readAllAt(r, int(size))
	if err != nil {
		return nil, err
	}
	return parseFileData(data, false)
}

func readAllAt(r io.ReaderAt, size int) ([]byte, error) {
	if size < 0 {
		return nil, ErrCorruptFile
	}
	if size == 0 {
		return []byte{}, nil
	}
	out := make([]byte, size)
	var off int64
	for off < int64(size) {
		n, err := r.ReadAt(out[off:], off)
		off += int64(n)
		if err == nil {
			continue
		}
		if err == io.EOF && off == int64(size) {
			break
		}
		return nil, err
	}
	return out, nil
}

func parseFileData(data []byte, mmapped bool) (*File, error) {
	if len(data) < mcfHeaderSize {
		return nil, ErrCorruptFile
	}
	hdr, ok := decodeHeader(data[:mcfHeaderSize])
	if !ok {
		return nil, ErrCorruptFile
	}
	if !hdr.Valid() {
		return nil, ErrInvalidMagic
	}
	if !hdr.Compatible() {
		return nil, ErrUnsupportedMajor
	}
	if hdr.FileSize != uint64(len(data)) {
		return nil, ErrCorruptFile
	}

	// Basic header sanity. HeaderSize must at least cover the fixed header bytes.
	if hdr.HeaderSize < mcfHeaderSize {
		return nil, ErrCorruptFile
	}
	if uint64(hdr.HeaderSize) > uint64(len(data)) {
		return nil, ErrCorruptFile
	}

	// Section directory bounds check
	secSize := uint64(mcfSectionSize)
	dirSize := uint64(hdr.SectionCount) * secSize
	dirStart := hdr.SectionDirOffset
	dirEnd := dirStart + dirSize

	if dirStart < uint64(hdr.HeaderSize) {
		return nil, ErrCorruptFile
	}
	if dirEnd < dirStart || dirEnd > uint64(len(data)) {
		return nil, ErrCorruptFile
	}

	// Copy and decode the section directory out of file data.
	sections := make([]MCFSection, hdr.SectionCount)
	if hdr.SectionCount > 0 {
		for i := range sections {
			start := int(dirStart) + i*mcfSectionSize
			end := start + mcfSectionSize
			sec, ok := decodeSection(data[start:end])
			if !ok {
				return nil, ErrCorruptFile
			}
			sections[i] = sec
		}
	}

	// Validate section bounds and ensure they do not overlap the section directory.
	for i := range sections {
		s := &sections[i]

		// Basic overflow-safe end calculation
		if s.Size > uint64(len(data)) {
			return nil, fmt.Errorf("%w: section %d size out of range", ErrCorruptFile, i)
		}
		end := s.Offset + s.Size
		if end < s.Offset {
			return nil, fmt.Errorf("%w: section %d offset overflow", ErrCorruptFile, i)
		}
		if end > uint64(len(data)) {
			return nil, fmt.Errorf("%w: section %d out of bounds", ErrCorruptFile, i)
		}
		if s.Offset < uint64(hdr.HeaderSize) {
			return nil, fmt.Errorf("%w: section %d overlaps header", ErrCorruptFile, i)
		}
		if rangesOverlap(s.Offset, end, dirStart, dirEnd) {
			return nil, fmt.Errorf("%w: section %d overlaps section directory", ErrCorruptFile, i)
		}

		// Keep section starts aligned for consumers that use aligned views.
		if (s.Offset % mcfAlign) != 0 {
			return nil, fmt.Errorf("%w: section %d offset not %d-byte aligned", ErrCorruptFile, i, mcfAlign)
		}
	}

	return &File{
		Data:     data,
		Header:   &hdr,
		Sections: sections,
		mmapped:  mmapped,
	}, nil
}

// Close releases file resources and any mmap backing.
func (f *File) Close() error {
	if f == nil {
		return nil
	}
	if f.Data != nil {
		var err error
		if f.mmapped {
			err = unix.Munmap(f.Data)
		}
		f.Data = nil
		f.Header = nil
		f.Sections = nil
		f.mmapped = false
		return err
	}
	f.Header = nil
	f.Sections = nil
	f.mmapped = false
	return nil
}

// Section returns the first section matching the given type, or nil if it does not exist.
func (f *File) Section(t SectionType) *MCFSection {
	for i := range f.Sections {
		if SectionType(f.Sections[i].Type) == t {
			return &f.Sections[i]
		}
	}
	return nil
}

// SectionData returns a zero-copy slice covering the section payload.
// The caller must not retain this slice after File.Close().
func (f *File) SectionData(s *MCFSection) []byte {
	if f == nil || s == nil || f.Data == nil {
		return nil
	}

	start := s.Offset
	end := s.Offset + s.Size

	if end < start || end > uint64(len(f.Data)) {
		return nil
	}

	// Safe because Open() rejects files that don't fit into an int-sized slice.
	return f.Data[int(start):int(end)]
}
