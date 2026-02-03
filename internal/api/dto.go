package api

import (
	"encoding/json"
	"fmt"
)

type DeleteResponseResp struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

type GetResponseParams struct {
	Include            []string `json:"include,omitempty"`
	IncludeObfuscation bool     `json:"include_obfuscation,omitempty"`
	StartingAfter      string   `json:"starting_after,omitempty"`
	Stream             bool     `json:"stream,omitempty"`
}

type CompactResponseReq struct {
	Model string      `json:"model"`
	Input *InputValue `json:"input,omitempty"`
}

type InputValue struct {
	String *string
	Items  []any
}

func (v *InputValue) UnmarshalJSON(b []byte) error {
	if v == nil {
		return fmt.Errorf("input value: nil receiver")
	}
	if len(b) == 0 || string(b) == "null" {
		*v = InputValue{}
		return nil
	}
	switch b[0] {
	case '"':
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return fmt.Errorf("input value: %w", err)
		}
		v.String = &s
		v.Items = nil
		return nil
	case '[':
		var items []any
		if err := json.Unmarshal(b, &items); err != nil {
			return fmt.Errorf("input value: %w", err)
		}
		v.Items = items
		v.String = nil
		return nil
	default:
		return fmt.Errorf("input value: expected string or array")
	}
}

func (v InputValue) MarshalJSON() ([]byte, error) {
	if v.String != nil {
		return json.Marshal(*v.String)
	}
	if v.Items != nil {
		return json.Marshal(v.Items)
	}
	return []byte("null"), nil
}
