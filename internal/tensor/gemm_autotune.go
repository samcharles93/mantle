package tensor

import "sync"

type GemmShape struct {
	M int
	K int
	N int
}

type TunedGemm struct {
	Cfg   GemmConfig
	Score float64
}

type GemmAutotuner struct {
	mu    sync.RWMutex
	cache map[GemmShape]TunedGemm
}

func NewGemmAutotuner() *GemmAutotuner {
	return &GemmAutotuner{
		cache: make(map[GemmShape]TunedGemm),
	}
}

func (t *GemmAutotuner) GetConfig(
	shape GemmShape,
	base GemmConfig,
	run func(cfg GemmConfig) float64,
) GemmConfig {
	t.mu.RLock()
	if tuned, ok := t.cache[shape]; ok {
		t.mu.RUnlock()
		return tuned.Cfg
	}
	t.mu.RUnlock()

	bestCfg := base
	bestScore := run(base)

	for _, cfg := range candidateConfigs(base) {
		score := run(cfg)
		if score > bestScore {
			bestCfg = cfg
			bestScore = score
		}
	}

	t.mu.Lock()
	t.cache[shape] = TunedGemm{
		Cfg:   bestCfg,
		Score: bestScore,
	}
	t.mu.Unlock()

	return bestCfg
}

func candidateConfigs(base GemmConfig) []GemmConfig {
	var out []GemmConfig

	for _, tk := range []int{
		base.TileK,
		base.TileK / 2,
		base.TileK * 2,
		24,
		32,
	} {
		if tk <= 0 {
			continue
		}
		cfg := base
		cfg.TileK = clampTile(tk, maxTileK)
		out = append(out, cfg)
	}

	return out
}
