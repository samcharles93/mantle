package mcf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"sort"
	"unsafe"
)

// TensorIndexVersion is the on-disk version of the tensor index section payload.
const TensorIndexVersion uint32 = 1

// TensorIndexHeader describes the on-disk layout of the tensor index section.
// Offsets are relative to the start of the section payload.
type TensorIndexHeader struct {
	Version     uint32 // = 1
	Flags       uint32 // TensorIndexFlag*
	TensorCount uint32
	DimsCount   uint32 // total number of uint64 dims in the dims table

	EntriesOff  uint64 // []TensorIndexEntry (TensorCount)
	DimsOff     uint64 // []uint64 (DimsCount)
	StringsOff  uint64 // []byte (StringsSize)
	StringsSize uint64
}

// TensorIndex flags.
const (
	// TensorIndexFlagSortedByName means entries are sorted by raw name bytes ascending.
	// This allows binary-search lookup without building a map.
	TensorIndexFlagSortedByName uint32 = 1 << 0

	// TensorIndexFlagNamesUTF8 indicates names are valid UTF-8 (advisory).
	TensorIndexFlagNamesUTF8 uint32 = 1 << 1
)

// TensorDType identifies the tensor element encoding.
// Keep these stable forever; add new values only.
type TensorDType uint32

const (
	DTypeUnknown TensorDType = iota
	DTypeF32
	DTypeF16
	DTypeBF16
	DTypeF64
	DTypeI8
	DTypeU8
	DTypeI16
	DTypeU16
	DTypeI32
	DTypeU32
	DTypeI64
	DTypeU64

	// Quant/packed encodings can live in a higher range.
	// e.g. 0x1000+... (reserved for future)
)

// TensorIndexEntry is the on-disk fixed-size record for a tensor.
// Name bytes live in the strings table.
// Shape dims live in the dims table.
type TensorIndexEntry struct {
	NameOff uint32 // into strings table
	NameLen uint32 // bytes

	DType TensorDType
	Rank  uint32 // number of dims

	DimOff uint32 // index into dims table (uint64 elements)
	_      uint32 // reserved (padding)

	// DataOff is an absolute file offset (from start of file), not section-relative.
	// This makes slicing data out of the mmap trivial.
	DataOff  uint64
	DataSize uint64
}

// TensorIndex is a parsed view over a tensor index section payload.
// It keeps a reference to the raw section bytes (which usually reference the mmap).
type TensorIndex struct {
	raw []byte
	hdr TensorIndexHeader
}

// TensorIndexRecord is the input to EncodeTensorIndexSection.
type TensorIndexRecord struct {
	Name  string
	DType TensorDType
	Shape []uint64

	// Absolute file offsets into the mapped file:
	DataOff  uint64
	DataSize uint64
}

var errBadTensorIndex = errors.New("mcf: corrupt tensor index section")

// ParseTensorIndexSection validates and returns a view over a tensor index section payload.
// Pass it File.SectionData(File.Section(SectionTensorIndex)).
func ParseTensorIndexSection(sec []byte) (*TensorIndex, error) {
	// Header is fixed-size and little-endian in the file.
	const hdrSize = 48 // binary.Size(TensorIndexHeader{}) if packed; keep constant for stability.

	if len(sec) < hdrSize {
		return nil, ErrCorruptFile
	}

	h := TensorIndexHeader{
		Version:     binary.LittleEndian.Uint32(sec[0:4]),
		Flags:       binary.LittleEndian.Uint32(sec[4:8]),
		TensorCount: binary.LittleEndian.Uint32(sec[8:12]),
		DimsCount:   binary.LittleEndian.Uint32(sec[12:16]),
		EntriesOff:  binary.LittleEndian.Uint64(sec[16:24]),
		DimsOff:     binary.LittleEndian.Uint64(sec[24:32]),
		StringsOff:  binary.LittleEndian.Uint64(sec[32:40]),
		StringsSize: binary.LittleEndian.Uint64(sec[40:48]),
	}

	if h.Version != TensorIndexVersion {
		return nil, ErrUnsupportedMinor
	}

	// Basic structural sanity
	if h.TensorCount == 0 {
		return nil, ErrCorruptFile
	}

	// Bounds check tables (using uint64 arithmetic safely)
	secLen := uint64(len(sec))

	entrySize := uint64(unsafe.Sizeof(TensorIndexEntry{}))
	entriesBytes := uint64(h.TensorCount) * entrySize

	dimsBytes := uint64(h.DimsCount) * 8

	// Entries
	if h.EntriesOff > secLen || h.EntriesOff+entriesBytes > secLen {
		return nil, ErrCorruptFile
	}
	// Dims
	if h.DimsOff > secLen || h.DimsOff+dimsBytes > secLen {
		return nil, ErrCorruptFile
	}
	// Strings
	if h.StringsOff > secLen || h.StringsOff+h.StringsSize > secLen {
		return nil, ErrCorruptFile
	}

	// Validate each entry’s name/shape offsets are within tables.
	for i := uint32(0); i < h.TensorCount; i++ {
		e, err := readTensorIndexEntry(sec, h.EntriesOff, i)
		if err != nil {
			return nil, ErrCorruptFile
		}

		// Name bounds within strings table
		if uint64(e.NameOff)+uint64(e.NameLen) > h.StringsSize {
			return nil, ErrCorruptFile
		}

		// Shape bounds within dims table (DimOff is an index in uint64 elements)
		if e.Rank > 0 {
			end := uint64(e.DimOff) + uint64(e.Rank)
			if end > uint64(h.DimsCount) {
				return nil, ErrCorruptFile
			}
		}
	}

	return &TensorIndex{
		raw: sec,
		hdr: h,
	}, nil
}

func readTensorIndexEntry(sec []byte, entriesOff uint64, i uint32) (TensorIndexEntry, error) {
	entrySize := uint64(unsafe.Sizeof(TensorIndexEntry{}))
	base := entriesOff + uint64(i)*entrySize
	end := base + entrySize
	if end > uint64(len(sec)) {
		return TensorIndexEntry{}, errBadTensorIndex
	}

	b := sec[base:end]

	// Layout matches TensorIndexEntry fields in order (little-endian).
	e := TensorIndexEntry{
		NameOff:  binary.LittleEndian.Uint32(b[0:4]),
		NameLen:  binary.LittleEndian.Uint32(b[4:8]),
		DType:    TensorDType(binary.LittleEndian.Uint32(b[8:12])),
		Rank:     binary.LittleEndian.Uint32(b[12:16]),
		DimOff:   binary.LittleEndian.Uint32(b[16:20]),
		DataOff:  binary.LittleEndian.Uint64(b[24:32]),
		DataSize: binary.LittleEndian.Uint64(b[32:40]),
	}
	return e, nil
}

func (ti *TensorIndex) Count() int {
	return int(ti.hdr.TensorCount)
}

func (ti *TensorIndex) Flags() uint32 {
	return ti.hdr.Flags
}

func (ti *TensorIndex) Entry(i int) (TensorIndexEntry, error) {
	if i < 0 || i >= int(ti.hdr.TensorCount) {
		return TensorIndexEntry{}, ErrCorruptFile
	}
	return readTensorIndexEntry(ti.raw, ti.hdr.EntriesOff, uint32(i))
}

func (ti *TensorIndex) NameBytes(i int) ([]byte, error) {
	e, err := ti.Entry(i)
	if err != nil {
		return nil, err
	}
	strBase := ti.hdr.StringsOff
	off := strBase + uint64(e.NameOff)
	end := off + uint64(e.NameLen)
	if end > strBase+ti.hdr.StringsSize || end > uint64(len(ti.raw)) {
		return nil, ErrCorruptFile
	}
	return ti.raw[off:end], nil
}

func (ti *TensorIndex) Name(i int) (string, error) {
	b, err := ti.NameBytes(i)
	if err != nil {
		return "", err
	}
	if len(b) == 0 {
		return "", nil
	}
	// Zero-copy string view over mmap-backed bytes.
	return unsafe.String(unsafe.SliceData(b), len(b)), nil
}

func (ti *TensorIndex) Shape(i int) ([]uint64, error) {
	e, err := ti.Entry(i)
	if err != nil {
		return nil, err
	}
	if e.Rank == 0 {
		return nil, nil
	}

	out := make([]uint64, 0, e.Rank)
	for d := uint32(0); d < e.Rank; d++ {
		val, err := ti.dimAt(e.DimOff + d)
		if err != nil {
			return nil, err
		}
		out = append(out, val)
	}
	return out, nil
}

func (ti *TensorIndex) dimAt(dimIndex uint32) (uint64, error) {
	if dimIndex >= ti.hdr.DimsCount {
		return 0, ErrCorruptFile
	}
	base := ti.hdr.DimsOff + uint64(dimIndex)*8
	end := base + 8
	if end > uint64(len(ti.raw)) {
		return 0, ErrCorruptFile
	}
	return binary.LittleEndian.Uint64(ti.raw[base:end]), nil
}

// Find returns the entry index for the given tensor name.
// If the index is sorted (TensorIndexFlagSortedByName), this is O(log n).
// Otherwise it's a linear scan.
func (ti *TensorIndex) Find(name string) (int, bool) {
	if ti == nil {
		return -1, false
	}
	key := []byte(name)

	if (ti.hdr.Flags & TensorIndexFlagSortedByName) != 0 {
		n := int(ti.hdr.TensorCount)
		i := sort.Search(n, func(i int) bool {
			nb, err := ti.NameBytes(i)
			if err != nil {
				return true
			}
			return bytes.Compare(nb, key) >= 0
		})
		if i < n {
			nb, err := ti.NameBytes(i)
			if err == nil && bytes.Equal(nb, key) {
				return i, true
			}
		}
		return -1, false
	}

	for i := 0; i < int(ti.hdr.TensorCount); i++ {
		nb, err := ti.NameBytes(i)
		if err != nil {
			return -1, false
		}
		if bytes.Equal(nb, key) {
			return i, true
		}
	}
	return -1, false
}

// TensorData returns a zero-copy view of the tensor payload bytes from the mapped file.
// This assumes entry.DataOff is an absolute file offset.
func (ti *TensorIndex) TensorData(f *File, i int) ([]byte, error) {
	if f == nil || f.Data == nil {
		return nil, ErrCorruptFile
	}
	e, err := ti.Entry(i)
	if err != nil {
		return nil, err
	}

	off := e.DataOff
	end := e.DataOff + e.DataSize
	if end < off || end > uint64(len(f.Data)) {
		return nil, ErrCorruptFile
	}
	return f.Data[off:end], nil
}

// EncodeTensorIndexSection builds a tensor index section payload (v1).
// Records are sorted by name, and the sorted flag is set.
func EncodeTensorIndexSection(records []TensorIndexRecord) ([]byte, error) {
	if len(records) == 0 {
		return nil, errors.New("mcf: tensor index requires at least one record")
	}

	// Copy and sort for determinism.
	recs := make([]TensorIndexRecord, len(records))
	copy(recs, records)
	sort.Slice(recs, func(i, j int) bool { return recs[i].Name < recs[j].Name })

	// Build dims + strings tables, and entries referencing them.
	var (
		dims       []uint64
		stringBlob []byte
		entries    = make([]TensorIndexEntry, 0, len(recs))
	)

	for _, r := range recs {
		if r.Name == "" {
			return nil, errors.New("mcf: tensor name must be non-empty")
		}
		if len(r.Shape) > int(^uint32(0)) {
			return nil, errors.New("mcf: tensor rank too large")
		}

		nameOff := uint32(len(stringBlob))
		nameBytes := []byte(r.Name)
		nameLen := uint32(len(nameBytes))
		stringBlob = append(stringBlob, nameBytes...)

		dimOff := uint32(len(dims))
		dims = append(dims, r.Shape...)

		entries = append(entries, TensorIndexEntry{
			NameOff:  nameOff,
			NameLen:  nameLen,
			DType:    r.DType,
			Rank:     uint32(len(r.Shape)),
			DimOff:   dimOff,
			DataOff:  r.DataOff,
			DataSize: r.DataSize,
		})
	}

	hdr := TensorIndexHeader{
		Version:     TensorIndexVersion,
		Flags:       TensorIndexFlagSortedByName | TensorIndexFlagNamesUTF8,
		TensorCount: uint32(len(entries)),
		DimsCount:   uint32(len(dims)),
	}

	// Layout: header | entries | dims | strings
	hdrSize := uint64(48)
	entrySize := uint64(unsafe.Sizeof(TensorIndexEntry{}))

	hdr.EntriesOff = hdrSize
	hdr.DimsOff = hdr.EntriesOff + entrySize*uint64(len(entries))
	hdr.StringsOff = hdr.DimsOff + uint64(len(dims))*8
	hdr.StringsSize = uint64(len(stringBlob))

	total := hdr.StringsOff + hdr.StringsSize
	if total > uint64(^uint32(0)) && unsafe.Sizeof(uint(0)) == 4 {
		return nil, errors.New("mcf: tensor index too large for 32-bit")
	}

	out := make([]byte, int(total))

	// Header
	binary.LittleEndian.PutUint32(out[0:4], hdr.Version)
	binary.LittleEndian.PutUint32(out[4:8], hdr.Flags)
	binary.LittleEndian.PutUint32(out[8:12], hdr.TensorCount)
	binary.LittleEndian.PutUint32(out[12:16], hdr.DimsCount)
	binary.LittleEndian.PutUint64(out[16:24], hdr.EntriesOff)
	binary.LittleEndian.PutUint64(out[24:32], hdr.DimsOff)
	binary.LittleEndian.PutUint64(out[32:40], hdr.StringsOff)
	binary.LittleEndian.PutUint64(out[40:48], hdr.StringsSize)

	// Entries
	ep := int(hdr.EntriesOff)
	for _, e := range entries {
		// Fixed layout (40 bytes).
		binary.LittleEndian.PutUint32(out[ep+0:ep+4], e.NameOff)
		binary.LittleEndian.PutUint32(out[ep+4:ep+8], e.NameLen)
		binary.LittleEndian.PutUint32(out[ep+8:ep+12], uint32(e.DType))
		binary.LittleEndian.PutUint32(out[ep+12:ep+16], e.Rank)
		binary.LittleEndian.PutUint32(out[ep+16:ep+20], e.DimOff)
		// ep+20..ep+24 reserved/padding
		binary.LittleEndian.PutUint64(out[ep+24:ep+32], e.DataOff)
		binary.LittleEndian.PutUint64(out[ep+32:ep+40], e.DataSize)
		ep += int(entrySize)
	}

	// Dims
	dp := int(hdr.DimsOff)
	for _, d := range dims {
		binary.LittleEndian.PutUint64(out[dp:dp+8], d)
		dp += 8
	}

	// Strings
	copy(out[int(hdr.StringsOff):int(hdr.StringsOff+hdr.StringsSize)], stringBlob)

	return out, nil
}
