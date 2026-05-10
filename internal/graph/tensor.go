package graph

import "fmt"

type TensorID int

type TensorRole int

const (
	RoleInput TensorRole = iota
	RoleWeight
	RoleKVState
	RoleActivation
	RoleOutput
)

func (r TensorRole) String() string {
	switch r {
	case RoleInput:
		return "Input"
	case RoleWeight:
		return "Weight"
	case RoleKVState:
		return "KVState"
	case RoleActivation:
		return "Activation"
	case RoleOutput:
		return "Output"
	default:
		return fmt.Sprintf("Role(%d)", int(r))
	}
}

type DType int

const (
	DTypeF32 DType = iota
	DTypeF16
	DTypeBF16
	DTypeQ8
	DTypeK4
)

func (d DType) String() string {
	switch d {
	case DTypeF32:
		return "F32"
	case DTypeF16:
		return "F16"
	case DTypeBF16:
		return "BF16"
	case DTypeQ8:
		return "Q8"
	case DTypeK4:
		return "K4"
	default:
		return fmt.Sprintf("DType(%d)", int(d))
	}
}
