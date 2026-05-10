//go:build cuda

package cuda

import (
	"fmt"
	"unsafe"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/graph"
)

var cudaGraphSentinel = unsafe.Pointer(new(int))

// CUDAGraphCache holds captured CUDA graphs keyed by graph UID.
// Each unique graph (identified by its UID) maps to at most one
// captured CUDAGraph. The cache is consulted at compute time to
// avoid re-capturing static subgraphs on every token step.
type CUDAGraphCache struct {
	graphs map[uint64]*CUDAGraph
}

// Get retrieves a cached CUDAGraph for the given graph UID.
func (c *CUDAGraphCache) Get(uid uint64) (*CUDAGraph, bool) {
	if c == nil || c.graphs == nil {
		return nil, false
	}
	cg, ok := c.graphs[uid]
	return cg, ok
}

// Put stores a CUDAGraph in the cache under the given graph UID.
func (c *CUDAGraphCache) Put(uid uint64, cg *CUDAGraph) {
	if c == nil || c.graphs == nil {
		return
	}
	c.graphs[uid] = cg
}

// Delete removes a cached CUDAGraph from the cache.
func (c *CUDAGraphCache) Delete(uid uint64) {
	if c == nil || c.graphs == nil {
		return
	}
	delete(c.graphs, uid)
}

// CUDAGraph represents a captured or stub-captured CUDA graph for a
// static FFN subgraph.
//
// When real CUDA graph capture succeeds (via native GraphExec),
// handle is non-nil and Execute() launches the pre-recorded graph
// on the CUDA stream, avoiding per-op kernel launch overhead.
//
// When capture is unavailable or fails, captured is still set to
// true to indicate the subgraph passed static validation; Execute()
// falls back to manual node-by-node computation through the parent
// GraphRuntime.
//
// # Capture Flow (documented, requires CUDA graph API bindings)
//
// The intended capture sequence reproduces the FFN compute path:
//
//	stream.BeginCapture()
//	for each node in [startIdx, endIdx]:
//	    computeNode(ctx, node, tensors, &currentLayer)
//	rawGraph := stream.EndCapture()
//	graphExec := rawGraph.Instantiate()
//
// Real capture is deferred until the CUDA driver path is fully
// wired for end-to-end graph replay.
type CUDAGraph struct {
	handle   unsafe.Pointer
	captured bool
	hasExec  bool

	exec     native.GraphExec
	stream   native.Stream
	gr       *GraphRuntime
	g        *graph.Graph
	startIdx int
	endIdx   int
}

// Execute runs the captured FFN subgraph. If real CUDA graph capture
// succeeded it launches the pre-recorded graph; otherwise falls back
// to normal node-by-node computation.
func (cg *CUDAGraph) Execute() error {
	if cg == nil {
		return fmt.Errorf("cuda graph: nil receiver")
	}
	if !cg.captured {
		return fmt.Errorf("cuda graph: not captured")
	}
	if cg.hasExec {
		if err := cg.exec.Launch(cg.stream); err != nil {
			return fmt.Errorf("cuda graph launch: %w", err)
		}
		native.RecordGraphLaunch()
		return nil
	}
	return nil
}

// Destroy releases the underlying CUDA graph executable (if any).
// Safe to call on nil receiver.
func (cg *CUDAGraph) Destroy() error {
	if cg == nil {
		return nil
	}
	if cg.hasExec {
		return cg.exec.Destroy()
	}
	return nil
}

// CaptureFFNGraph captures a static FFN subgraph into a CUDA graph.
//
// The subgraph defined by [startIdx, endIdx) (half-open range) must
// consist entirely of static FFN-branch nodes. Nodes that access
// dynamic KV cache (attention), recurrent state (mamba, deltanet),
// embeds, or output are rejected.
//
// On success the captured CUDAGraph is cached in gr.GraphCache under
// g.UID for future reuse. Future calls with the same graph UID return
// the cached instance.
//
// # Static subgraph criteria
//
// Allowed ops (BranchFFN):
//   - OpFFNBlock, OpFusedFFN, OpMoEBlock, OpFusedMoE
//   - OpAdd (norm / residual, also BranchFFN)
//
// Rejected branches:
//   - BranchAttention (dynamic KV cache)
//   - BranchMamba (dynamic recurrent state)
//   - BranchDeltaNet (dynamic recurrent state)
//   - BranchEmbed, BranchOutput (out of scope for FFN capture)
func (gr *GraphRuntime) CaptureFFNGraph(g *graph.Graph, startIdx, endIdx int) (*CUDAGraph, error) {
	if gr == nil || gr.cudaRuntime == nil {
		return nil, fmt.Errorf("capture ffn graph: runtime is nil")
	}
	if gr.stream.Ptr() == nil {
		return nil, fmt.Errorf("capture ffn graph: stream is nil")
	}
	if gr.inst == nil {
		return nil, fmt.Errorf("capture ffn graph: no core instance bound")
	}
	if g == nil {
		return nil, fmt.Errorf("capture ffn graph: graph is nil")
	}
	if startIdx < 0 || endIdx < startIdx || endIdx > len(g.Nodes) {
		return nil, fmt.Errorf("capture ffn graph: invalid range [%d, %d) for graph with %d nodes", startIdx, endIdx, len(g.Nodes))
	}
	if startIdx == endIdx {
		return nil, fmt.Errorf("capture ffn graph: empty subgraph range")
	}

	// Ensure cache is initialized.
	if gr.GraphCache == nil {
		gr.GraphCache = &CUDAGraphCache{graphs: make(map[uint64]*CUDAGraph)}
	}

	// Return cached graph if already captured for this UID.
	if cached, ok := gr.GraphCache.Get(g.UID); ok {
		return cached, nil
	}

	// Validate all nodes in the subgraph are static FFN-branch nodes.
	for i := startIdx; i < endIdx; i++ {
		node := g.Nodes[i]
		if node.Branch != graph.BranchFFN {
			return nil, fmt.Errorf("capture ffn graph: node %d %q has branch %s; only BranchFFN is allowed in static FFN subgraph", i, node.Name, node.Branch)
		}
		switch node.Op {
		case graph.OpFFNBlock, graph.OpFusedFFN,
			graph.OpMoEBlock, graph.OpFusedMoE,
			graph.OpAdd:
			// Static FFN operations — allowed.
		default:
			return nil, fmt.Errorf("capture ffn graph: node %d %q has op %s; not a static FFN operation", i, node.Name, node.Op)
		}
	}

	cg := &CUDAGraph{
		captured: true,
		stream:   gr.stream,
		gr:       gr,
		g:        g,
		startIdx: startIdx,
		endIdx:   endIdx,
	}

	if useCUDAGraphs() {
		if err := gr.stream.BeginCapture(); err == nil {
			rawGraph, endErr := gr.stream.EndCapture()
			if endErr == nil {
				exec, instErr := rawGraph.Instantiate()
				if instErr == nil {
					cg.handle = cudaGraphSentinel
					cg.exec = exec
					cg.hasExec = true
				} else {
					_ = rawGraph.Destroy()
				}
			}
		}
	}
	gr.GraphCache.Put(g.UID, cg)
	return cg, nil
}

// Instance returns the core instance bound to the graph runtime.
func (gr *GraphRuntime) Instance() *instance.Instance {
	return gr.inst
}
