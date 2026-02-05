package simd

// Tuned for the benchmark shape (256^3).
const (
	defaultTileM = 32
	defaultTileN = 32
	defaultTileK = 16

	maxTileM = 64
	maxTileN = 64
	maxTileK = 64
)

type GemmConfig struct {
	TileM int
	TileN int
	TileK int

	UseSIMD    bool
	UsePackedB bool
}

func DefaultGemmConfig() GemmConfig {
	return GemmConfig{
		TileM:      defaultTileM,
		TileN:      defaultTileN,
		TileK:      defaultTileK,
		UseSIMD:    true,
		UsePackedB: true,
	}
}

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

func clampTile(v, max int) int {
	if v < 1 {
		return 1
	}
	if v > max {
		return max
	}
	return v
}
