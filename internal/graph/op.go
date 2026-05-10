package graph

import "fmt"

type OpType int

const (
	OpNone OpType = iota
	OpAttentionBlock
	OpFFNBlock
	OpMoEBlock
	OpMambaBlock
	OpDeltaNetBlock
	OpEmbed
	OpOutput
	OpAdd
	OpFusedFFN
	OpFusedAttention
	OpFusedNormResidual
	OpFusedMoE
)

func (o OpType) String() string {
	switch o {
	case OpNone:
		return "OpNone"
	case OpAttentionBlock:
		return "OpAttentionBlock"
	case OpFFNBlock:
		return "OpFFNBlock"
	case OpMoEBlock:
		return "OpMoEBlock"
	case OpMambaBlock:
		return "OpMambaBlock"
	case OpDeltaNetBlock:
		return "OpDeltaNetBlock"
	case OpEmbed:
		return "OpEmbed"
	case OpOutput:
		return "OpOutput"
	case OpAdd:
		return "OpAdd"
	case OpFusedFFN:
		return "OpFusedFFN"
	case OpFusedAttention:
		return "OpFusedAttention"
	case OpFusedNormResidual:
		return "OpFusedNormResidual"
	case OpFusedMoE:
		return "OpFusedMoE"
	default:
		return fmt.Sprintf("OpType(%d)", int(o))
	}
}
