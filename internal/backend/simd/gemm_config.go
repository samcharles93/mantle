package simd

import (
	"os"
	"strconv"
)

// Tuned for the benchmark shape (256^3).
const (
	defaultTileM = 32
	defaultTileN = 32
	defaultTileK = 16

	maxTileM = 64
	maxTileN = 64
	maxTileK = 64
)

// MaxTileM returns the maximum TileM value.
func MaxTileM() int { return maxTileM }

// MaxTileN returns the maximum TileN value.
func MaxTileN() int { return maxTileN }

// MaxTileK returns the maximum TileK value.
func MaxTileK() int { return maxTileK }

// CPUHasAVX2 reports whether the CPU supports AVX2.
func CPUHasAVX2() bool { return cpu.HasAVX2 }

// CPUHasAVX512 reports whether the CPU supports AVX-512.
func CPUHasAVX512() bool { return cpu.HasAVX512 }

// TilingConfig holds runtime-configurable tiling parameters.
type TilingConfig struct {
	TileM int
	TileN int
	TileK int
}

// GemmConfig extends tiling with execution mode flags.
type GemmConfig struct {
	TileM int
	TileN int
	TileK int

	UseSIMD    bool
	UsePackedB bool
}

// DefaultTilingConfig returns the default tiling configuration,
// optionally overridden by environment variables.
func DefaultTilingConfig() TilingConfig {
	cfg := TilingConfig{
		TileM: defaultTileM,
		TileN: defaultTileN,
		TileK: defaultTileK,
	}

	// Environment variable overrides
	if v := os.Getenv("MANTLE_TILE_M"); v != "" {
		if m, err := strconv.Atoi(v); err == nil && m > 0 {
			cfg.TileM = m
		}
	}
	if v := os.Getenv("MANTLE_TILE_N"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.TileN = n
		}
	}
	if v := os.Getenv("MANTLE_TILE_K"); v != "" {
		if k, err := strconv.Atoi(v); err == nil && k > 0 {
			cfg.TileK = k
		}
	}

	return cfg
}

// DefaultGemmConfig returns the default GEMM configuration.
func DefaultGemmConfig() GemmConfig {
	tiling := DefaultTilingConfig()
	return GemmConfig{
		TileM:      tiling.TileM,
		TileN:      tiling.TileN,
		TileK:      tiling.TileK,
		UseSIMD:    true,
		UsePackedB: true,
	}
}

// SelectGemmConfig selects a GEMM configuration based on matrix dimensions.
func SelectGemmConfig(m, k, n int) GemmConfig {
	cfg := DefaultGemmConfig()

	switch {
	case k >= 192:
		cfg.TileK = 32
	case k >= 96:
		cfg.TileK = 24
	}

	cfg.TileM = clampTile(cfg.TileM, maxTileM)
	cfg.TileN = clampTile(cfg.TileN, maxTileN)
	cfg.TileK = clampTile(cfg.TileK, maxTileK)

	return cfg
}

// SelectGemmConfigWithTiling selects a GEMM configuration using custom tiling parameters.
func SelectGemmConfigWithTiling(m, k, n int, tiling TilingConfig) GemmConfig {
	cfg := GemmConfig{
		TileM:      tiling.TileM,
		TileN:      tiling.TileN,
		TileK:      tiling.TileK,
		UseSIMD:    true,
		UsePackedB: true,
	}

	// Shape-aware dynamic adjustment
	if k >= 192 && tiling.TileK < 32 {
		cfg.TileK = 32
	} else if k >= 96 && k < 192 && tiling.TileK < 24 {
		cfg.TileK = 24
	}

	// Small matrix optimization
	if m < 64 && n < 64 {
		cfg.TileM = min(cfg.TileM, 16)
		cfg.TileN = min(cfg.TileN, 16)
	}

	cfg.TileM = clampTile(cfg.TileM, maxTileM)
	cfg.TileN = clampTile(cfg.TileN, maxTileN)
	cfg.TileK = clampTile(cfg.TileK, maxTileK)

	return cfg
}

func clampTile(v, max int) int {
	if v < 1 {
		return 1
	}
	if v > max {
		return max
	}
	return v
}

// ToGemmConfig converts TilingConfig to GemmConfig.
func (t *TilingConfig) ToGemmConfig() GemmConfig {
	return GemmConfig{
		TileM:      t.TileM,
		TileN:      t.TileN,
		TileK:      t.TileK,
		UseSIMD:    true,
		UsePackedB: true,
	}
}
