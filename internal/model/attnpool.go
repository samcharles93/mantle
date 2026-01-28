package model

import (
	"runtime"

	"github.com/samcharles93/mantle/internal/tensor"
)

type attnTask struct {
	ctx    *attnContext
	rs, re int
	done   chan struct{}
}

type attnContext struct {
	q, cacheK, cacheV []float32
	attnOut           []float32

	pos, start        int
	kvStride, headDim int
	nHead, kvHeads    int
	scale             float32
}

type attnPool struct {
	size      int
	tasks     chan attnTask
	doneSlots chan chan struct{}
	scores    []float32
	maxCtx    int
}

func attnWorkersFor(nHead int) int {
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

func newAttnPool(workers, maxCtx int) *attnPool {
	if workers < 1 {
		workers = 1
	}
	if maxCtx < 1 {
		maxCtx = 1
	}
	p := &attnPool{
		size:      workers,
		tasks:     make(chan attnTask, workers*2),
		doneSlots: make(chan chan struct{}, workers),
		scores:    make([]float32, workers*maxCtx),
		maxCtx:    maxCtx,
	}
	for i := 0; i < workers; i++ {
		p.doneSlots <- make(chan struct{}, 1)
	}
	for i := 0; i < workers; i++ {
		workerID := i
		go func() {
			base := workerID * p.maxCtx
			scoresBuf := p.scores[base : base+p.maxCtx]
			for task := range p.tasks {
				runAttnHeads(task.ctx, scoresBuf, task.rs, task.re)
				task.done <- struct{}{}
			}
		}()
	}
	return p
}

func (m *Instance) initAttnPool() {
	m.attnPoolOnce.Do(func() {
		m.attnPool = newAttnPool(attnWorkersFor(m.HeadCount), m.MaxContext)
	})
}

func (m *Instance) getAttnPool() *attnPool {
	if m.attnPool == nil {
		m.initAttnPool()
	}
	return m.attnPool
}

func runAttnHeads(ctx *attnContext, scoresBuf []float32, rs, re int) {
	if ctx == nil || rs >= re {
		return
	}
	if ctx.start < 0 || ctx.start > ctx.pos {
		panic("invalid attention window start")
	}
	winLen := ctx.pos - ctx.start + 1
	if winLen > len(scoresBuf) {
		panic("attention scores buffer too small")
	}
	scores := scoresBuf[:winLen]
	for h := rs; h < re; h++ {
		kvHead := h * ctx.kvHeads / ctx.nHead
		qh := ctx.q[h*ctx.headDim : (h+1)*ctx.headDim]
		for t := ctx.start; t <= ctx.pos; t++ {
			koff := t*ctx.kvStride + kvHead*ctx.headDim
			kv := ctx.cacheK[koff : koff+ctx.headDim]
			scores[t-ctx.start] = tensor.Dot(qh, kv) * ctx.scale
		}
		tensor.Softmax(scores)
		out := ctx.attnOut[h*ctx.headDim : (h+1)*ctx.headDim]
		for d := range ctx.headDim {
			var sum float32
			for t := ctx.start; t <= ctx.pos; t++ {
				voff := t*ctx.kvStride + kvHead*ctx.headDim + d
				sum += scores[t-ctx.start] * ctx.cacheV[voff]
			}
			out[d] = sum
		}
	}
}
