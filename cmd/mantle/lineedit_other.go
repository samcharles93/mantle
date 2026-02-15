//go:build !linux

package main

import (
	"bufio"
	"io"
	"os"
)

func readInteractiveLine(_ string) (string, error) {
	r := bufio.NewReader(os.Stdin)
	s, err := r.ReadString('\n')
	if err != nil && err != io.EOF {
		return "", err
	}
	return trimTrailingNewline(s), nil
}

func trimTrailingNewline(s string) string {
	if len(s) > 0 && s[len(s)-1] == '\n' {
		s = s[:len(s)-1]
	}
	if len(s) > 0 && s[len(s)-1] == '\r' {
		s = s[:len(s)-1]
	}
	return s
}
