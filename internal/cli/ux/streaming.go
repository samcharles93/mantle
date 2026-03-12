package ux

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/samcharles93/mantle/internal/logger"
)

type StreamMode string

const (
	StreamInstant    StreamMode = "instant"
	StreamSmooth     StreamMode = "smooth"
	StreamTypewriter StreamMode = "typewriter"
	StreamQuiet      StreamMode = "quiet"
)

// NormalizeStreamMode parses a user-provided stream mode.
func NormalizeStreamMode(v string) (StreamMode, bool) {
	mode := StreamMode(strings.ToLower(strings.TrimSpace(v)))
	switch mode {
	case StreamInstant, StreamSmooth, StreamTypewriter, StreamQuiet:
		return mode, true
	default:
		return StreamSmooth, false
	}
}

// StreamWriter handles buffered token streaming with configurable modes
type StreamWriter struct {
	mode   StreamMode
	output io.Writer
	buffer *bufio.Writer

	// For batching
	mu            sync.Mutex
	batch         strings.Builder
	lastFlush     time.Time
	flushInterval time.Duration
	batchSize     int // flush after N tokens

	// For quiet mode
	accumulator strings.Builder

	// For raw output mode
	rawOutput bool

	log logger.Logger
}

// NewStreamWriter creates a new streaming output handler
func NewStreamWriter(log logger.Logger, mode StreamMode, rawOutput bool) *StreamWriter {
	w := &StreamWriter{
		mode:          mode,
		output:        os.Stdout,
		buffer:        bufio.NewWriterSize(os.Stdout, 4096),
		flushInterval: 50 * time.Millisecond,
		batchSize:     5,
		lastFlush:     time.Now(),
		rawOutput:     rawOutput,
		log:           log,
	}

	// Start background flusher for smooth mode
	if mode == StreamSmooth {
		go w.backgroundFlusher()
	}

	return w
}

// Write handles a single token from the LLM
func (w *StreamWriter) Write(token string) {
	switch w.mode {
	case StreamInstant:
		w.writeInstant(token)
	case StreamSmooth:
		w.writeSmooth(token)
	case StreamTypewriter:
		w.writeTypewriter(token)
	case StreamQuiet:
		w.writeQuiet(token)
	}
}

// Flush ensures all buffered content is written
func (w *StreamWriter) Flush() string {
	w.mu.Lock()
	defer w.mu.Unlock()

	switch w.mode {
	case StreamQuiet:
		// Return accumulated text for final processing
		result := w.accumulator.String()
		if w.rawOutput {
			// Apply escaping to full output at once
			escaped := escapeRawOutput(result)
			if _, err := fmt.Fprint(w.output, escaped); err != nil {
				w.log.Debug("failed to write output", "error", err)
			}
		} else {
			if _, err := fmt.Fprint(w.output, result); err != nil {
				w.log.Debug("failed to write output", "error", err)
			}
		}
		return result
	case StreamSmooth:
		w.flushBatch()
		return w.accumulator.String()
	default:
		if err := w.buffer.Flush(); err != nil {
			w.log.Debug("failed to flush buffer", "error", err)
		}
		return w.accumulator.String()
	}
}

// writeInstant - original per-token behavior
func (w *StreamWriter) writeInstant(token string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.accumulator.WriteString(token)

	if w.rawOutput {
		escaped := escapeRawOutput(token)
		if _, err := w.buffer.WriteString(escaped); err != nil {
			w.log.Debug("failed to write to buffer", "error", err)
		}
	} else {
		if _, err := w.buffer.WriteString(token); err != nil {
			w.log.Debug("failed to write to buffer", "error", err)
		}
	}
	if err := w.buffer.Flush(); err != nil {
		w.log.Debug("failed to flush buffer", "error", err)
	}
}

// writeSmooth - batched with time-based flushing
func (w *StreamWriter) writeSmooth(token string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.accumulator.WriteString(token)
	w.batch.WriteString(token)

	// Flush if batch size reached
	if w.batch.Len() > 0 {
		elapsed := time.Since(w.lastFlush)
		// Count tokens by spaces (rough approximation)
		tokenCount := strings.Count(w.batch.String(), " ") + 1

		if tokenCount >= w.batchSize || elapsed >= w.flushInterval {
			w.flushBatch()
		}
	}
}

// writeTypewriter - character-by-character output without artificial delay
func (w *StreamWriter) writeTypewriter(token string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.accumulator.WriteString(token)

	// Output character by character
	for _, r := range token {
		if w.rawOutput {
			escaped := escapeRawOutputRune(r)
			if _, err := fmt.Fprint(w.buffer, escaped); err != nil {
				w.log.Debug("failed to write to buffer", "error", err)
			}
		} else {
			if _, err := fmt.Fprintf(w.buffer, "%c", r); err != nil {
				w.log.Debug("failed to write to buffer", "error", err)
			}
		}
		if err := w.buffer.Flush(); err != nil {
			w.log.Debug("failed to flush buffer", "error", err)
		}
	}
}

// writeQuiet - accumulate only, no output until Flush
func (w *StreamWriter) writeQuiet(token string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.accumulator.WriteString(token)
}

// flushBatch writes accumulated batch to output (must hold lock)
func (w *StreamWriter) flushBatch() {
	if w.batch.Len() == 0 {
		return
	}

	text := w.batch.String()
	if w.rawOutput {
		escaped := escapeRawOutput(text)
		if _, err := w.buffer.WriteString(escaped); err != nil {
			w.log.Debug("failed to write to buffer", "error", err)
		}
	} else {
		if _, err := w.buffer.WriteString(text); err != nil {
			w.log.Debug("failed to write to buffer", "error", err)
		}
	}
	if err := w.buffer.Flush(); err != nil {
		w.log.Debug("failed to flush buffer", "error", err)
	}

	w.batch.Reset()
	w.lastFlush = time.Now()
}

// backgroundFlusher periodically flushes batched content
func (w *StreamWriter) backgroundFlusher() {
	ticker := time.NewTicker(w.flushInterval)
	defer ticker.Stop()

	for range ticker.C {
		w.mu.Lock()
		if time.Since(w.lastFlush) >= w.flushInterval && w.batch.Len() > 0 {
			w.flushBatch()
		}
		w.mu.Unlock()
	}
}

// escapeRawOutputRune escapes a single rune for raw output
func escapeRawOutputRune(r rune) string {
	switch r {
	case '\n':
		return `\n`
	case '\r':
		return `\r`
	case '\t':
		return `\t`
	case '\\':
		return `\\`
	default:
		if strconv.IsPrint(r) {
			return string(r)
		}
		return fmt.Sprintf(`\u%04x`, r)
	}
}

func escapeRawOutput(s string) string {
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
