//go:build cuda

package cuda

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/graph"
)

func (gr *GraphRuntime) fusedFFNBlock(node graph.Node, x []float32, currentLayer *int) ([]float32, error) {
	layerIdx := parseLayerIndex(node.Name, *currentLayer)
	*currentLayer = layerIdx

	if layerIdx < 0 || layerIdx >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("fused ffn: layer index %d out of range", layerIdx)
	}
	layer := &gr.inst.Layers[layerIdx]

	out := make([]float32, layer.FfnDown.R)
	if gr.ops.FFNBlock(layer, x, out) {
		return out, nil
	}
	return nil, fmt.Errorf("fused ffn: CUDA fast path unavailable for layer %d", layerIdx)
}
