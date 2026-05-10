package graph

import "fmt"

// NodeBranch identifies which high-level compute dispatch path a node follows.
// It mirrors the existing runtime dispatch: computeAttentionBranch, computeStandardFFNOutput, etc.
type NodeBranch int

const (
	BranchAttention NodeBranch = iota // → computeAttentionBranch()
	BranchFFN                         // → computeStandardFFNOutput() / FFN()
	BranchMamba                       // → Mamba() / ShortConv()
	BranchDeltaNet                    // → DeltaNet()
	BranchEmbed                       // → initializeTokenInput()
	BranchOutput                      // → output norm + projection + softcap
)

// String returns the NodeBranch name.
func (b NodeBranch) String() string {
	switch b {
	case BranchAttention:
		return "Attention"
	case BranchFFN:
		return "FFN"
	case BranchMamba:
		return "Mamba"
	case BranchDeltaNet:
		return "DeltaNet"
	case BranchEmbed:
		return "Embed"
	case BranchOutput:
		return "Output"
	default:
		return fmt.Sprintf("Branch(%d)", int(b))
	}
}

// Node represents a single block-level operation in the computation graph.
type Node struct {
	Op     OpType     // the operation type (fused or non-fused)
	Branch NodeBranch // which dispatch path this node follows
	Name   string     // debug name, e.g. "layer0.attention"
	Input  []TensorID // input tensor slot IDs
	Output TensorID   // output tensor slot ID
	Params NodeParams // operation-specific parameters
}

// Validate checks that the node's basic structure is valid.
func (n *Node) Validate() error {
	if n.Op == OpNone {
		return fmt.Errorf("node %q: OpType is OpNone", n.Name)
	}
	if len(n.Input) == 0 {
		return fmt.Errorf("node %q: Input is empty", n.Name)
	}
	if n.Output == 0 {
		return fmt.Errorf("node %q: Output is zero", n.Name)
	}
	// Validate that Params matches the expected type for this Op/branch.
	// This is a best-effort sanity check, not exhaustive.
	switch n.Branch {
	case BranchAttention:
		if _, ok := n.Params.(AttentionParams); !ok {
			return fmt.Errorf("node %q: BranchAttention requires AttentionParams, got %T", n.Name, n.Params)
		}
	case BranchFFN:
		if _, ok := n.Params.(FFNParams); !ok {
			if _, ok := n.Params.(MoEParams); !ok {
				return fmt.Errorf("node %q: BranchFFN requires FFNParams or MoEParams, got %T", n.Name, n.Params)
			}
		}
	case BranchMamba:
		if _, ok := n.Params.(MambaParams); !ok {
			return fmt.Errorf("node %q: BranchMamba requires MambaParams, got %T", n.Name, n.Params)
		}
	case BranchDeltaNet:
		if _, ok := n.Params.(DeltaNetParams); !ok {
			return fmt.Errorf("node %q: BranchDeltaNet requires DeltaNetParams, got %T", n.Name, n.Params)
		}
	case BranchEmbed:
		if _, ok := n.Params.(EmbedParams); !ok {
			return fmt.Errorf("node %q: BranchEmbed requires EmbedParams, got %T", n.Name, n.Params)
		}
	case BranchOutput:
		if _, ok := n.Params.(OutputParams); !ok {
			return fmt.Errorf("node %q: BranchOutput requires OutputParams, got %T", n.Name, n.Params)
		}
	}
	return nil
}
