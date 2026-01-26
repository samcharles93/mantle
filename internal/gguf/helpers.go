package gguf

import "fmt"

func GetString(kv map[string]Value, key string) (string, bool) {
	v, ok := kv[key]
	if !ok {
		return "", false
	}
	s, ok := v.Value.(string)
	return s, ok
}

func GetBool(kv map[string]Value, key string) (bool, bool) {
	v, ok := kv[key]
	if !ok {
		return false, false
	}
	b, ok := v.Value.(bool)
	return b, ok
}

func GetUint64(kv map[string]Value, key string) (uint64, bool) {
	v, ok := kv[key]
	if !ok {
		return 0, false
	}
	return asUint64(v.Value)
}

func GetInt64(kv map[string]Value, key string) (int64, bool) {
	v, ok := kv[key]
	if !ok {
		return 0, false
	}
	switch t := v.Value.(type) {
	case int8:
		return int64(t), true
	case int16:
		return int64(t), true
	case int32:
		return int64(t), true
	case int64:
		return t, true
	case uint8:
		return int64(t), true
	case uint16:
		return int64(t), true
	case uint32:
		return int64(t), true
	case uint64:
		return int64(t), true
	default:
		return 0, false
	}
}

func GetFloat64(kv map[string]Value, key string) (float64, bool) {
	v, ok := kv[key]
	if !ok {
		return 0, false
	}
	switch t := v.Value.(type) {
	case float32:
		return float64(t), true
	case float64:
		return t, true
	default:
		return 0, false
	}
}

// GetArray retrieves a slice of type T from the key-value pairs.
// It checks that the value exists, is an array, and that all elements can be asserted to type T.
func GetArray[T any](kv map[string]Value, key string) ([]T, bool) {
	v, ok := kv[key]
	if !ok {
		return nil, false
	}
	arr, ok := v.Value.(ArrayValue)
	if !ok {
		return nil, false
	}
	
	out := make([]T, 0, len(arr.Values))
	for _, item := range arr.Values {
		tItem, ok := item.(T)
		if !ok {
			return nil, false
		}
		out = append(out, tItem)
	}
	return out, true
}

func MustGetString(kv map[string]Value, key string) (string, error) {
	if s, ok := GetString(kv, key); ok {
		return s, nil
	}
	return "", fmt.Errorf("missing or invalid %s", key)
}

func MustGetUint64(kv map[string]Value, key string) (uint64, error) {
	if v, ok := GetUint64(kv, key); ok {
		return v, nil
	}
	return 0, fmt.Errorf("missing or invalid %s", key)
}
