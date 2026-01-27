package mcf

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"slices"
)

type dedupKey struct {
	Sum   [32]byte
	DType TensorDType
	Size  uint64
	Rank  uint32
}

type dedupEntry struct {
	Off   uint64
	Shape []uint64
}

type tensorDeduper struct {
	out  *os.File
	buf  []byte
	seen map[dedupKey][]dedupEntry
}

func newTensorDeduper(out *os.File, buf []byte) *tensorDeduper {
	// Need two buffers for range compare. If caller provides too small, we still work.
	if len(buf) < 16*1024 {
		buf = make([]byte, 16*1024)
	}
	return &tensorDeduper{
		out:  out,
		buf:  buf,
		seen: make(map[dedupKey][]dedupEntry),
	}
}

func (d *tensorDeduper) key(dtype TensorDType, shape []uint64, size uint64, sum [32]byte) dedupKey {
	return dedupKey{
		Sum:   sum,
		DType: dtype,
		Size:  size,
		Rank:  uint32(len(shape)),
	}
}

// FindMatch returns an existing offset to reuse if the newly written tensor at newOff
// is byte-identical to a previous tensor with the same key and shape.
//
// The verification is done by comparing bytes in the OUTPUT file:
// [existingOff, existingOff+size) vs [newOff, newOff+size).
func (d *tensorDeduper) FindMatch(newOff uint64, dtype TensorDType, shape []uint64, size uint64, sum [32]byte) (uint64, bool, error) {
	k := d.key(dtype, shape, size, sum)
	cands := d.seen[k]
	for i := range cands {
		if !slices.Equal(cands[i].Shape, shape) {
			continue
		}
		eq, err := compareFileRanges(d.out, cands[i].Off, newOff, size, d.buf)
		if err != nil {
			return 0, false, err
		}
		if eq {
			return cands[i].Off, true, nil
		}
	}
	return 0, false, nil
}

func (d *tensorDeduper) Add(off uint64, dtype TensorDType, shape []uint64, size uint64, sum [32]byte) {
	k := d.key(dtype, shape, size, sum)
	shapeCopy := make([]uint64, len(shape))
	copy(shapeCopy, shape)
	d.seen[k] = append(d.seen[k], dedupEntry{
		Off:   off,
		Shape: shapeCopy,
	})
}

func hashOfBytesWritten(sum []byte) ([32]byte, error) {
	if len(sum) != sha256.Size {
		return [32]byte{}, fmt.Errorf("mcf: invalid sha256 size %d", len(sum))
	}
	var out [32]byte
	copy(out[:], sum)
	return out, nil
}

func compareFileRanges(f *os.File, offA, offB, size uint64, scratch []byte) (bool, error) {
	if size == 0 {
		return true, nil
	}
	half := len(scratch) / 2
	if half == 0 {
		scratch = make([]byte, 16*1024)
		half = len(scratch) / 2
	}
	aBuf := scratch[:half]
	bBuf := scratch[half : half+half]

	var done uint64
	for done < size {
		n := int(min(uint64(len(aBuf)), size-done))

		_, err := f.ReadAt(aBuf[:n], int64(offA+done))
		if err != nil {
			return false, err
		}
		_, err = f.ReadAt(bBuf[:n], int64(offB+done))
		if err != nil {
			return false, err
		}
		if !bytes.Equal(aBuf[:n], bBuf[:n]) {
			return false, nil
		}
		done += uint64(n)
	}
	return true, nil
}
