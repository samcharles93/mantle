package mcf

import (
	"encoding/binary"
	"errors"
	"math"
	"unsafe"
)

const (
	QuantInfoVersion uint32 = 1

	quantInfoHeaderSize = 8
	quantRecordSize     = 24
)

// QuantDomain defines the reconstruction logic.
type QuantDomain uint8

const (
	DomainWeights     QuantDomain = 0 // Symmetric: Z=0, Range [-max, max]
	DomainActivations QuantDomain = 1 // Asymmetric: Z=Calc, Range [min, max]
)

// QuantInfoHeader is the on-disk header for the QuantInfo payload.
type QuantInfoHeader struct {
	Version     uint32
	RecordCount uint32
}

// QuantRecord is the fixed-size metadata for a single quantized tensor.
// TOTAL SIZE: 24 bytes.
// ALIGNMENT: 8-byte aligned (friendly for 64-bit readers).
type QuantRecord struct {
	TensorIndex uint32 // Maps to Tensor Table

	// Method is cast from TensorDType.
	// We use uint8 to maintain perfect struct packing.
	Method uint8 // Matches TensorDType (e.g., 0x31)

	// Domain controls reconstruction
	Domain uint8 // 0=Weights, 1=Activations

	// Configuration geometry
	BlockSize uint16 // Standard: 32
	SuperSize uint16 // Standard: 256 (k*) or 0 (q*)

	// PADDING: Reserved for future flags (e.g. 'IsSparse').
	// Ensures struct size is exactly 24 bytes.
	Reserved [6]byte

	// Calibration Stats (Reconstruction Source of Truth)
	MinClip float32
	MaxClip float32
}

// QuantInfo is a parsed view over a QuantInfo section payload.
// It keeps a reference to the raw section bytes (which usually reference the mmap).
type QuantInfo struct {
	raw     []byte
	hdr     QuantInfoHeader
	records []QuantRecord
}

var errBadQuantInfo = errors.New("mcf: corrupt quantinfo section")

func init() {
	if unsafe.Sizeof(QuantRecord{}) != quantRecordSize {
		panic("mcf: QuantRecord size must be 24 bytes")
	}
}

// ParseQuantInfoSection validates and returns a view over a QuantInfo section payload.
// Pass it File.SectionData(File.Section(SectionQuantInfo)).
func ParseQuantInfoSection(sec []byte) (*QuantInfo, error) {
	if len(sec) < quantInfoHeaderSize {
		return nil, ErrCorruptFile
	}

	hdr := QuantInfoHeader{
		Version:     binary.LittleEndian.Uint32(sec[0:4]),
		RecordCount: binary.LittleEndian.Uint32(sec[4:8]),
	}
	if hdr.Version != QuantInfoVersion {
		return nil, ErrUnsupportedMinor
	}

	recBytes, ok := mulUint64(uint64(hdr.RecordCount), quantRecordSize)
	if !ok {
		return nil, ErrCorruptFile
	}
	need := uint64(quantInfoHeaderSize) + recBytes
	if need > uint64(len(sec)) {
		return nil, ErrCorruptFile
	}
	if uint64(hdr.RecordCount) > uint64(int(^uint(0)>>1)) {
		return nil, ErrCorruptFile
	}

	var records []QuantRecord
	if hdr.RecordCount > 0 {
		raw := sec[quantInfoHeaderSize:need]
		if uintptr(unsafe.Pointer(&raw[0]))%unsafe.Alignof(QuantRecord{}) != 0 {
			return nil, ErrCorruptFile
		}
		records = unsafe.Slice((*QuantRecord)(unsafe.Pointer(&raw[0])), int(hdr.RecordCount))

		for i := range records {
			r := records[i]
			if !isZeroBytes(r.Reserved[:]) {
				return nil, ErrCorruptFile
			}
			if !isValidDomain(r.Domain) {
				return nil, ErrCorruptFile
			}
			if err := validateQuantRecord(r); err != nil {
				return nil, ErrCorruptFile
			}
		}
	}

	return &QuantInfo{raw: sec, hdr: hdr, records: records}, nil
}

func (qi *QuantInfo) Count() int {
	if qi == nil {
		return 0
	}
	return int(qi.hdr.RecordCount)
}

func (qi *QuantInfo) Record(i int) (QuantRecord, error) {
	if qi == nil || i < 0 || i >= int(qi.hdr.RecordCount) {
		return QuantRecord{}, ErrCorruptFile
	}
	return qi.records[i], nil
}

// Records returns the zero-copy record slice.
// The caller must not retain the slice after File.Close().
func (qi *QuantInfo) Records() []QuantRecord {
	if qi == nil {
		return nil
	}
	return qi.records
}

// EncodeQuantInfoSection builds a QuantInfo section payload (v1).
func EncodeQuantInfoSection(records []QuantRecord) ([]byte, error) {
	if len(records) > int(^uint32(0)) {
		return nil, errors.New("mcf: too many quant records")
	}

	recBytes, ok := mulUint64(uint64(len(records)), quantRecordSize)
	if !ok {
		return nil, errors.New("mcf: quantinfo too large")
	}
	total := uint64(quantInfoHeaderSize) + recBytes
	if total > uint64(int(^uint(0)>>1)) {
		return nil, errors.New("mcf: quantinfo too large for this architecture")
	}

	out := make([]byte, int(total))
	binary.LittleEndian.PutUint32(out[0:4], QuantInfoVersion)
	binary.LittleEndian.PutUint32(out[4:8], uint32(len(records)))

	off := quantInfoHeaderSize
	for _, r := range records {
		if !isZeroBytes(r.Reserved[:]) {
			return nil, errors.New("mcf: quant record reserved bytes must be zero")
		}
		if !isValidDomain(r.Domain) {
			return nil, errors.New("mcf: invalid quant record domain")
		}
		if err := validateQuantRecord(r); err != nil {
			return nil, err
		}

		binary.LittleEndian.PutUint32(out[off+0:off+4], r.TensorIndex)
		out[off+4] = r.Method
		out[off+5] = byte(r.Domain)
		binary.LittleEndian.PutUint16(out[off+6:off+8], r.BlockSize)
		binary.LittleEndian.PutUint16(out[off+8:off+10], r.SuperSize)
		copy(out[off+10:off+16], r.Reserved[:])
		binary.LittleEndian.PutUint32(out[off+16:off+20], math.Float32bits(r.MinClip))
		binary.LittleEndian.PutUint32(out[off+20:off+24], math.Float32bits(r.MaxClip))
		off += quantRecordSize
	}

	return out, nil
}

func validateQuantRecord(r QuantRecord) error {
	method := TensorDType(r.Method)
	switch method {
	case DTypeQ8, DTypeQ4:
		if r.Domain != uint8(DomainWeights) {
			return errBadQuantInfo
		}
		if r.BlockSize != 32 || r.SuperSize != 0 {
			return errBadQuantInfo
		}
	case DTypeK6, DTypeK4, DTypeK3, DTypeK2:
		if r.Domain != uint8(DomainWeights) {
			return errBadQuantInfo
		}
		if r.BlockSize != 32 || r.SuperSize != 256 {
			return errBadQuantInfo
		}
	case DTypeInt8, DTypeInt4:
		if r.BlockSize != 0 || r.SuperSize != 0 {
			return errBadQuantInfo
		}
	default:
		return errBadQuantInfo
	}
	return nil
}

func isValidDomain(d uint8) bool {
	return d == uint8(DomainWeights) || d == uint8(DomainActivations)
}

func isZeroBytes(b []byte) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}

func mulUint64(a, b uint64) (uint64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > ^uint64(0)/b {
		return 0, false
	}
	return a * b, true
}

// DTypeRequiresAligned64 reports whether the dtype requires 64-byte internal payload alignment.
func DTypeRequiresAligned64(dt TensorDType) bool {
	switch dt {
	case DTypeQ8, DTypeQ4, DTypeK6, DTypeK4, DTypeK3, DTypeK2:
		return true
	default:
		return false
	}
}
