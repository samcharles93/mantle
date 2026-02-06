package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"

	"simd/archsimd"
)

type output struct {
	GoVersion string          `json:"go_version"`
	GoOS      string          `json:"go_os"`
	GoArch    string          `json:"go_arch"`
	CPUs      int             `json:"cpus"`
	Features  map[string]bool `json:"features"`
}

func main() {
	features := map[string]bool{
		"AVX":                archsimd.X86.AVX(),
		"AVX2":               archsimd.X86.AVX2(),
		"FMA":                archsimd.X86.FMA(),
		"AVXAES":             archsimd.X86.AVXAES(),
		"VAES":               archsimd.X86.VAES(),
		"SHA":                archsimd.X86.SHA(),
		"AVX512":             archsimd.X86.AVX512(),
		"AVX512BITALG":       archsimd.X86.AVX512BITALG(),
		"AVX512GFNI":         archsimd.X86.AVX512GFNI(),
		"AVX512VAES":         archsimd.X86.AVX512VAES(),
		"AVX512VBMI":         archsimd.X86.AVX512VBMI(),
		"AVX512VBMI2":        archsimd.X86.AVX512VBMI2(),
		"AVX512VNNI":         archsimd.X86.AVX512VNNI(),
		"AVX512VPCLMULQDQ":   archsimd.X86.AVX512VPCLMULQDQ(),
		"AVX512VPOPCNTDQ":    archsimd.X86.AVX512VPOPCNTDQ(),
		"AVXVNNI":            archsimd.X86.AVXVNNI(),
	}

	out := output{
		GoVersion: runtime.Version(),
		GoOS:      runtime.GOOS,
		GoArch:    runtime.GOARCH,
		CPUs:      runtime.NumCPU(),
		Features:  features,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
}
