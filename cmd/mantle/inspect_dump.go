package main

import (
	"fmt"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func printTensorDump(name string, indexBytes, dataBytes []byte) {
	section("Tensor Dump: " + name)
	if len(indexBytes) == 0 {
		fmt.Println("(no tensor index section)")
		return
	}
	idx, err := mcf.ParseTensorIndexSection(indexBytes)
	if err != nil {
		fmt.Printf("(tensor index parse error: %v)\n", err)
		return
	}

	found := -1
	count := idx.Count()
	for i := range count {
		n, err := idx.Name(i)
		if err == nil && n == name {
			found = i
			break
		}
	}
	if found == -1 {
		fmt.Printf("tensor %q not found\n", name)
		return
	}

	entry, err := idx.Entry(found)
	if err != nil {
		fmt.Printf("error reading entry: %v\n", err)
		return
	}

	shape, _ := idx.Shape(found)
	fmt.Printf("Tensor: %s\n", name)
	fmt.Printf("DType: %s\n", dtypeName(entry.DType))
	fmt.Printf("Shape: %v\n", shape)
	fmt.Printf("Offset: %d\n", entry.DataOff)
	fmt.Printf("Size: %d\n", entry.DataSize)

	if uint64(len(dataBytes)) < entry.DataOff+entry.DataSize {
		fmt.Printf("error: data section too small (len=%d) for offset+size=%d\n", len(dataBytes), entry.DataOff+entry.DataSize)
		// Try to read what is available
	}

	maxDump := 64
	avail := int(min(uint64(maxDump), entry.DataSize))
	start := int(entry.DataOff)
	if start >= len(dataBytes) {
		fmt.Println("(data offset out of bounds)")
		return
	}
	end := min(start+avail, len(dataBytes))

	raw := dataBytes[start:end]
	fmt.Printf("Raw Bytes (first %d):\n", len(raw))
	for i, b := range raw {
		fmt.Printf("%02x ", b)
		if (i+1)%16 == 0 {
			fmt.Println()
		}
	}
	fmt.Println()

	// Interpret as u16 if size allows
	if len(raw) >= 2 {
		fmt.Println("As uint16 (le):")
		for i := 0; i+1 < len(raw); i += 2 {
			u := uint16(raw[i]) | uint16(raw[i+1])<<8
			// Approximate float interpretation for simple cases
			// BF16: 1.0 = 0x3f80 (16256)
			// FP16: 1.0 = 0x3c00 (15360)
			valStr := ""
			switch u {
			case 0x3f80:
				valStr = " (BF16=1.0)"
			case 0x3c00:
				valStr = " (FP16=1.0)"
			case 0:
				valStr = " (0.0)"
			}
			fmt.Printf("%04x%s ", u, valStr)
			if (i/2+1)%8 == 0 {
				fmt.Println()
			}
		}
		fmt.Println()
	}
}
