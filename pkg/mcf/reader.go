package mcf

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

type File struct {
	Data     []byte
	Header   *MCFHeader
	Sections []MCFSection
}

// Open maps an MCF file read-only and validates its structure.
// The returned file must be closed to release the mapping.
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
		// cannot mmap into a []byte on this architecture
		return nil, ErrCorruptFile
	}
	size := int(size64)

	hdrSize := int(unsafe.Sizeof(MCFHeader{}))
	if size < hdrSize {
		return nil, ErrCorruptFile
	}

	data, err := unix.Mmap(
		int(f.Fd()),
		0,
		size,
		unix.PROT_READ,
		unix.MAP_SHARED,
	)
	if err != nil {
		return nil, err
	}

	cleanup := func(e error) (*File, error) {
		_ = unix.Munmap(data)
		return nil, e
	}

	// Copy header out of the mmap so File.Header remains valid even after Close().
	var hdr MCFHeader
	copy(structBytes(&hdr), data[:hdrSize])

	if !hdr.Valid() {
		return cleanup(ErrInvalidMagic)
	}
	if !hdr.Compatible() {
		return cleanup(ErrUnsupportedMajor)
	}
	if hdr.FileSize != uint64(size) {
		return cleanup(ErrCorruptFile)
	}

	// Basic header sanity. HeaderSize must at least cover the header struct.
	if hdr.HeaderSize < uint32(hdrSize) {
		return cleanup(ErrCorruptFile)
	}
	if uint64(hdr.HeaderSize) > uint64(len(data)) {
		return cleanup(ErrCorruptFile)
	}

	// Section directory bounds check
	secSize := uint64(unsafe.Sizeof(MCFSection{}))
	dirSize := uint64(hdr.SectionCount) * secSize
	dirStart := hdr.SectionDirOffset
	dirEnd := dirStart + dirSize

	if dirStart < uint64(hdr.HeaderSize) {
		return cleanup(ErrCorruptFile)
	}
	if dirEnd < dirStart || dirEnd > uint64(len(data)) {
		return cleanup(ErrCorruptFile)
	}

	// Copy the section directory out of the mmap (keeps File.Sections valid after Close()).
	sections := make([]MCFSection, hdr.SectionCount)
	if hdr.SectionCount > 0 {
		raw := data[dirStart:dirEnd]
		copy(structSliceBytes(sections), raw)
	}

	// Validate section bounds and ensure they do not overlap the section directory.
	for i := range sections {
		s := &sections[i]

		// Basic overflow-safe end calculation
		if s.Size > uint64(len(data)) {
			return cleanup(fmt.Errorf("%w: section %d size out of range", ErrCorruptFile, i))
		}
		end := s.Offset + s.Size
		if end < s.Offset {
			return cleanup(fmt.Errorf("%w: section %d offset overflow", ErrCorruptFile, i))
		}
		if end > uint64(len(data)) {
			return cleanup(fmt.Errorf("%w: section %d out of bounds", ErrCorruptFile, i))
		}
		if s.Offset < uint64(hdr.HeaderSize) {
			return cleanup(fmt.Errorf("%w: section %d overlaps header", ErrCorruptFile, i))
		}
		if rangesOverlap(s.Offset, end, dirStart, dirEnd) {
			return cleanup(fmt.Errorf("%w: section %d overlaps section directory", ErrCorruptFile, i))
		}

		// Optional but strongly recommended: keep sections aligned for safe casting by consumers.
		// If you want to allow arbitrary alignment, delete these two checks.
		if (s.Offset % mcfAlign) != 0 {
			return cleanup(fmt.Errorf("%w: section %d offset not %d-byte aligned", ErrCorruptFile, i, mcfAlign))
		}
	}

	return &File{
		Data:     data,
		Header:   &hdr,
		Sections: sections,
	}, nil
}

// Close releases the mmap backing this file.
func (f *File) Close() error {
	if f == nil {
		return nil
	}
	if f.Data != nil {
		err := unix.Munmap(f.Data)
		f.Data = nil
		f.Header = nil
		f.Sections = nil
		return err
	}
	f.Header = nil
	f.Sections = nil
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
