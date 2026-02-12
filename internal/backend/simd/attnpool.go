package simd

import "runtime"

type AttnTask struct {
	Ctx    *AttnContext
	Rs, Re int
	Done   chan struct{}
}

type AttnContext struct {
	Q, CacheK, CacheV    []float32
	CacheK16, CacheV16   []uint16
	CacheKQ8, CacheVQ8   []int8
	CacheKQ8S, CacheVQ8S []float32
	AttnOut              []float32
	Ops                  Ops

	Pos, Start        int
	KvStride, HeadDim int
	NHead, KvHeads    int
	Scale             float32
	CacheLen          int
}

type AttnPool struct {
	Size      int
	Tasks     chan AttnTask
	DoneSlots chan chan struct{}
	Scores    []float32
	MaxCtx    int
}

func AttnWorkersFor(nHead int) int {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if nHead > 0 && workers > nHead {
		workers = nHead
	}
	if workers < 1 {
		return 1
	}
	return workers
}

func NewAttnPool(workers, maxCtx int) *AttnPool {
	if workers < 1 {
		workers = 1
	}
	if maxCtx < 1 {
		maxCtx = 1
	}
	p := &AttnPool{
		Size:      workers,
		Tasks:     make(chan AttnTask, workers*2),
		DoneSlots: make(chan chan struct{}, workers),
		Scores:    make([]float32, workers*maxCtx),
		MaxCtx:    maxCtx,
	}
	for i := 0; i < workers; i++ {
		p.DoneSlots <- make(chan struct{}, 1)
	}
	for i := 0; i < workers; i++ {
		workerID := i
		go func() {
			base := workerID * p.MaxCtx
			scoresBuf := p.Scores[base : base+p.MaxCtx]
			for task := range p.Tasks {
				RunAttnHeads(task.Ctx, scoresBuf, task.Rs, task.Re)
				task.Done <- struct{}{}
			}
		}()
	}
	return p
}

func RunAttnHeads(ctx *AttnContext, scoresBuf []float32, rs, re int) {
	if ctx == nil || rs >= re {
		return
	}
	if ctx.Start < 0 || ctx.Start > ctx.Pos {
		panic("invalid attention window start")
	}
	winLen := ctx.Pos - ctx.Start + 1
	if winLen > len(scoresBuf) {
		panic("attention scores buffer too small")
	}
	cacheLen := ctx.CacheLen
	useRing := cacheLen > 0 && cacheLen < ctx.Pos+1
	scores := scoresBuf[:winLen]
	for h := rs; h < re; h++ {
		kvHead := h * ctx.KvHeads / ctx.NHead
		qh := ctx.Q[h*ctx.HeadDim : (h+1)*ctx.HeadDim]
		for t := ctx.Start; t <= ctx.Pos; t++ {
			cachePos := t
			if useRing {
				cachePos = t % cacheLen
			}
			koff := cachePos*ctx.KvStride + kvHead*ctx.HeadDim
			if ctx.CacheKQ8 != nil {
				kv := ctx.CacheKQ8[koff : koff+ctx.HeadDim]
				scores[t-ctx.Start] = DotQ8(qh, kv, ctx.CacheKQ8S[cachePos]) * ctx.Scale
			} else if ctx.CacheK16 != nil {
				kv := ctx.CacheK16[koff : koff+ctx.HeadDim]
				scores[t-ctx.Start] = DotF16(qh, kv) * ctx.Scale
			} else {
				kv := ctx.CacheK[koff : koff+ctx.HeadDim]
				scores[t-ctx.Start] = Dot(qh, kv) * ctx.Scale
			}
		}
		if ctx.Ops != nil {
			ctx.Ops.Softmax(scores)
		} else {
			Softmax(scores)
		}
		out := ctx.AttnOut[h*ctx.HeadDim : (h+1)*ctx.HeadDim]
		for d := range ctx.HeadDim {
			var sum float32
			if ctx.CacheVQ8 != nil {
				for t := ctx.Start; t <= ctx.Pos; t++ {
					cachePos := t
					if useRing {
						cachePos = t % cacheLen
					}
					voff := cachePos*ctx.KvStride + kvHead*ctx.HeadDim + d
					sum += scores[t-ctx.Start] * float32(ctx.CacheVQ8[voff]) * ctx.CacheVQ8S[cachePos]
				}
			} else if ctx.CacheV16 != nil {
				for t := ctx.Start; t <= ctx.Pos; t++ {
					cachePos := t
					if useRing {
						cachePos = t % cacheLen
					}
					voff := cachePos*ctx.KvStride + kvHead*ctx.HeadDim + d
					sum += scores[t-ctx.Start] * Float16ToFloat32(ctx.CacheV16[voff])
				}
			} else {
				for t := ctx.Start; t <= ctx.Pos; t++ {
					cachePos := t
					if useRing {
						cachePos = t % cacheLen
					}
					voff := cachePos*ctx.KvStride + kvHead*ctx.HeadDim + d
					sum += scores[t-ctx.Start] * ctx.CacheV[voff]
				}
			}
			out[d] = sum
		}
	}
}
