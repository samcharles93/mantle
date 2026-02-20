package utils

import (
	"fmt"
	"strconv"
	"strings"
)

func EscapeRawOutput(s string) string {
	if s == "" {
		return ""
	}
	var b strings.Builder
	for _, r := range s {
		switch r {
		case '\n':
			fmt.Fprint(&b, "\\n")
		case '\r':
			fmt.Fprint(&b, "\\r")
		case '\t':
			fmt.Fprint(&b, "\\t")
		case '\\':
			fmt.Fprint(&b, "\\\\")
		default:
			if strconv.IsPrint(r) {
				b.WriteRune(r)
			} else {
				fmt.Fprintf(&b, `\u%04x`, r)
			}
		}
	}
	return b.String()
}
