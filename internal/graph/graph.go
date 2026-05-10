package graph

import "fmt"

// Graph is a flat, ordered sequence of block-level compute nodes.
// The linear order represents the forward-pass execution order.
type Graph struct {
	Nodes []Node
	UID   uint64 // graph identity; used by CUDA graph cache keying
}

// nextTensorID tracks the next available TensorID for allocation.
// It starts at 1 (0 reserved as sentinel/unset).
var nextTensorID TensorID = 1

// AddNode appends a node to the graph and returns the output tensor ID.
func (g *Graph) AddNode(n Node) TensorID {
	g.Nodes = append(g.Nodes, n)
	return n.Output
}

// Validate checks that every node in the graph has valid inputs.
func (g *Graph) Validate() error {
	if len(g.Nodes) == 0 {
		return fmt.Errorf("graph is empty")
	}
	// Collect all output tensor IDs that exist so far in the graph.
	outputs := make(map[TensorID]bool)
	for i, n := range g.Nodes {
		if err := n.Validate(); err != nil {
			return fmt.Errorf("node %d: %w", i, err)
		}
		for _, input := range n.Input {
			// Input 0 is allowed (represents a weight or model parameter, not a computed tensor).
			if input == 0 {
				continue
			}
			// Embedding nodes commonly reference model parameter tensors that are
			// not produced by prior nodes in the graph. Allow unproduced inputs for
			// embed branch to represent weights.
			if n.Branch == BranchEmbed {
				continue
			}
			if !outputs[input] {
				return fmt.Errorf("node %d %q: input tensor %d not produced by any prior node", i, n.Name, input)
			}
		}
		outputs[n.Output] = true
	}
	return nil
}

// Clone creates a deep copy of the graph (used by the fuser).
func (g *Graph) Clone() *Graph {
	if g == nil {
		return nil
	}
	cloned := &Graph{
		Nodes: make([]Node, len(g.Nodes)),
		UID:   g.UID,
	}
	for i, n := range g.Nodes {
		cloned.Nodes[i] = Node{
			Op:     n.Op,
			Branch: n.Branch,
			Name:   n.Name,
			Input:  append([]TensorID(nil), n.Input...),
			Output: n.Output,
			Params: n.Params,
		}
	}
	return cloned
}

// ResetTensorID resets the tensor ID allocator. Call before building a new graph
// to ensure deterministic IDs across builds.
func ResetTensorID() {
	nextTensorID = 1
}

// NewTensorID returns a fresh TensorID for graph construction.
func NewTensorID() TensorID {
	id := nextTensorID
	nextTensorID++
	return id
}
