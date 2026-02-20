package utils

import (
	"fmt"
	"strings"
)

func JoinInts(ids []int) string {
	if len(ids) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteByte('[')
	for i, id := range ids {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d", id)
	}
	b.WriteByte(']')
	return b.String()
}
