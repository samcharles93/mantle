package graph

import "testing"

func TestOpTypeString(t *testing.T) {
	tests := []struct {
		op   OpType
		want string
	}{
		{OpNone, "OpNone"},
		{OpAttentionBlock, "OpAttentionBlock"},
		{OpFFNBlock, "OpFFNBlock"},
		{OpMoEBlock, "OpMoEBlock"},
		{OpMambaBlock, "OpMambaBlock"},
		{OpDeltaNetBlock, "OpDeltaNetBlock"},
		{OpEmbed, "OpEmbed"},
		{OpOutput, "OpOutput"},
		{OpAdd, "OpAdd"},
		{OpFusedFFN, "OpFusedFFN"},
		{OpFusedAttention, "OpFusedAttention"},
		{OpFusedNormResidual, "OpFusedNormResidual"},
		{OpFusedMoE, "OpFusedMoE"},
	}
	for _, tt := range tests {
		if got := tt.op.String(); got != tt.want {
			t.Fatalf("OpType.String() = %q; want %q", got, tt.want)
		}
	}
}

func TestOpTypeCount(t *testing.T) {
	if OpNone != 0 {
		t.Fatalf("OpNone expected 0, got %d", OpNone)
	}
	ops := []OpType{
		OpAttentionBlock, OpFFNBlock, OpMoEBlock, OpMambaBlock,
		OpDeltaNetBlock, OpEmbed, OpOutput, OpAdd,
		OpFusedFFN, OpFusedAttention, OpFusedNormResidual, OpFusedMoE,
	}
	if len(ops) != 12 {
		t.Fatalf("expected 12 non-OpNone OpType constants, got %d", len(ops))
	}
	for _, o := range ops {
		if o == 0 {
			t.Fatalf("expected OpType %v to be non-zero", o)
		}
	}
}

func TestDTypeString(t *testing.T) {
	tests := []struct {
		d    DType
		want string
	}{
		{DTypeF32, "F32"}, {DTypeF16, "F16"}, {DTypeBF16, "BF16"},
		{DTypeQ8, "Q8"}, {DTypeK4, "K4"},
	}
	for _, tt := range tests {
		if got := tt.d.String(); got != tt.want {
			t.Fatalf("DType.String() = %q; want %q", got, tt.want)
		}
	}
}

func TestTensorRoleString(t *testing.T) {
	tests := []struct {
		r    TensorRole
		want string
	}{
		{RoleInput, "Input"}, {RoleWeight, "Weight"}, {RoleKVState, "KVState"},
		{RoleActivation, "Activation"}, {RoleOutput, "Output"},
	}
	for _, tt := range tests {
		if got := tt.r.String(); got != tt.want {
			t.Fatalf("TensorRole.String() = %q; want %q", got, tt.want)
		}
	}
}
