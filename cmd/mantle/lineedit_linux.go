//go:build linux

package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/sys/unix"
)

var interactiveHistory []string

func readInteractiveLine(prompt string) (string, error) {
	if !stdinIsTTY() {
		r := bufio.NewReader(os.Stdin)
		s, err := r.ReadString('\n')
		if err != nil && err != io.EOF {
			return "", err
		}
		return trimTrailingNewline(s), nil
	}

	fd := int(os.Stdin.Fd())
	oldState, err := unix.IoctlGetTermios(fd, unix.TCGETS)
	if err != nil {
		return "", err
	}
	newState := *oldState
	newState.Lflag &^= unix.ICANON | unix.ECHO
	newState.Cc[unix.VMIN] = 1
	newState.Cc[unix.VTIME] = 0
	if err := unix.IoctlSetTermios(fd, unix.TCSETS, &newState); err != nil {
		return "", err
	}
	defer func() {
		_ = unix.IoctlSetTermios(fd, unix.TCSETS, oldState)
	}()

	fmt.Print(prompt)
	line := make([]byte, 0, 256)
	cursor := 0
	escState := 0
	var escBuf strings.Builder
	var buf [16]byte
	histPos := len(interactiveHistory)
	histBrowsing := false
	histDraft := ""

	redraw := func() {
		fmt.Printf("\r%s%s", prompt, string(line))
		fmt.Print("\x1b[K")
		if cursor < len(line) {
			fmt.Printf("\r%s%s", prompt, string(line[:cursor]))
		}
	}
	isSpace := func(b byte) bool {
		return b == ' ' || b == '\t'
	}
	moveWordLeft := func() {
		if cursor == 0 {
			return
		}
		for cursor > 0 && isSpace(line[cursor-1]) {
			cursor--
		}
		for cursor > 0 && !isSpace(line[cursor-1]) {
			cursor--
		}
		redraw()
	}
	moveWordRight := func() {
		if cursor >= len(line) {
			return
		}
		for cursor < len(line) && isSpace(line[cursor]) {
			cursor++
		}
		for cursor < len(line) && !isSpace(line[cursor]) {
			cursor++
		}
		redraw()
	}
	deleteWordBack := func() {
		if cursor == 0 {
			return
		}
		start := cursor
		for start > 0 && isSpace(line[start-1]) {
			start--
		}
		for start > 0 && !isSpace(line[start-1]) {
			start--
		}
		line = append(line[:start], line[cursor:]...)
		cursor = start
		redraw()
	}
	deleteWordForward := func() {
		if cursor >= len(line) {
			return
		}
		end := cursor
		for end < len(line) && isSpace(line[end]) {
			end++
		}
		for end < len(line) && !isSpace(line[end]) {
			end++
		}
		line = append(line[:cursor], line[end:]...)
		redraw()
	}
	handleCSI := func(seq string) {
		switch seq {
		case "A": // up
			if len(interactiveHistory) == 0 {
				return
			}
			if !histBrowsing {
				histDraft = string(line)
				histBrowsing = true
				histPos = len(interactiveHistory)
			}
			if histPos > 0 {
				histPos--
				line = append(line[:0], interactiveHistory[histPos]...)
				cursor = len(line)
				redraw()
			}
		case "B": // down
			if !histBrowsing {
				return
			}
			if histPos < len(interactiveHistory)-1 {
				histPos++
				line = append(line[:0], interactiveHistory[histPos]...)
			} else {
				histPos = len(interactiveHistory)
				line = append(line[:0], histDraft...)
				histBrowsing = false
			}
			cursor = len(line)
			redraw()
		case "D":
			if cursor > 0 {
				cursor--
				redraw()
			}
		case "C":
			if cursor < len(line) {
				cursor++
				redraw()
			}
		case "H":
			cursor = 0
			redraw()
		case "F":
			cursor = len(line)
			redraw()
		case "3~":
			if cursor < len(line) {
				line = append(line[:cursor], line[cursor+1:]...)
				redraw()
			}
		case "1;5D", "5D":
			moveWordLeft()
		case "1;5C", "5C":
			moveWordRight()
		case "3;5~":
			deleteWordForward()
		}
	}

	for {
		n, err := os.Stdin.Read(buf[:])
		if err != nil {
			return "", err
		}
		for i := 0; i < n; i++ {
			b := buf[i]
			if escState != 0 {
				switch escState {
				case 1:
					if b == '[' {
						escState = 2
						escBuf.Reset()
					} else if b == 'b' || b == 'B' {
						moveWordLeft() // Alt+b
						escState = 0
					} else if b == 'f' || b == 'F' {
						moveWordRight() // Alt+f
						escState = 0
					} else if b == 127 {
						deleteWordBack() // Alt+Backspace
						escState = 0
					} else {
						escState = 0
					}
				case 2:
					escBuf.WriteByte(b)
					if (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b == '~' {
						handleCSI(escBuf.String())
						escState = 0
					}
				}
				continue
			}

			switch b {
			case 27: // ESC
				escState = 1
			case '\r', '\n':
				fmt.Print("\r\n")
				out := string(line)
				if strings.TrimSpace(out) != "" {
					interactiveHistory = append(interactiveHistory, out)
				}
				return out, nil
			case 3: // Ctrl+C
				fmt.Print("^C\r\n")
				return "", io.EOF
			case 4: // Ctrl+D
				if len(line) == 0 {
					fmt.Print("\r\n")
					return "", io.EOF
				}
			case 127, 8: // backspace
				if cursor > 0 {
					line = append(line[:cursor-1], line[cursor:]...)
					cursor--
					redraw()
				}
			case 1: // Ctrl+A
				cursor = 0
				redraw()
			case 5: // Ctrl+E
				cursor = len(line)
				redraw()
			case 23: // Ctrl+W
				deleteWordBack()
			default:
				if b >= 32 {
					if cursor == len(line) {
						line = append(line, b)
						cursor++
					} else {
						line = append(line, 0)
						copy(line[cursor+1:], line[cursor:])
						line[cursor] = b
						cursor++
					}
					redraw()
				}
			}
		}
	}
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
