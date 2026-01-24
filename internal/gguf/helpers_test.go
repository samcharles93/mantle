package gguf

import (
	"reflect"
	"testing"
)

func TestGetArray(t *testing.T) {
	kv := map[string]Value{
		"strings": {
			Type: TypeArray,
			Value: ArrayValue{
				ElemType: TypeString,
				Values:   []any{"a", "b", "c"},
			},
		},
		"ints": {
			Type: TypeArray,
			Value: ArrayValue{
				ElemType: TypeInt32,
				Values:   []any{int32(1), int32(2), int32(3)},
			},
		},
		"mixed": {
			Type: TypeArray,
			Value: ArrayValue{
				ElemType: TypeString,
				Values:   []any{"a", 1}, // mixed types, invalid for GGUF usually, but checking behavior
			},
		},
		"not_array": {
			Type:  TypeString,
			Value: "hello",
		},
	}

	// Test strings
	strs, ok := GetArray[string](kv, "strings")
	if !ok {
		t.Error("expected ok for strings")
	}
	if !reflect.DeepEqual(strs, []string{"a", "b", "c"}) {
		t.Errorf("got %v, want %v", strs, []string{"a", "b", "c"})
	}

	// Test ints
	ints, ok := GetArray[int32](kv, "ints")
	if !ok {
		t.Error("expected ok for ints")
	}
	if !reflect.DeepEqual(ints, []int32{1, 2, 3}) {
		t.Errorf("got %v, want %v", ints, []int32{1, 2, 3})
	}

	// Test type mismatch
	_, ok = GetArray[int32](kv, "strings")
	if ok {
		t.Error("expected !ok for type mismatch (string array as int32)")
	}

	// Test element mismatch
	_, ok = GetArray[string](kv, "mixed")
	if ok {
		t.Error("expected !ok for mixed element types")
	}

	// Test not an array
	_, ok = GetArray[string](kv, "not_array")
	if ok {
		t.Error("expected !ok for non-array value")
	}

	// Test missing key
	_, ok = GetArray[string](kv, "missing")
	if ok {
		t.Error("expected !ok for missing key")
	}
}
