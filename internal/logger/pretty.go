package logger

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"time"
)

// ANSI color codes
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorYellow = "\033[33m"
	colorBlue   = "\033[34m"
	colorGray   = "\033[90m"
	colorCyan   = "\033[36m"
	colorGreen  = "\033[32m"
	colorBold   = "\033[1m"
)

// PrettyHandler is a slog.Handler that formats logs with colors for CLI output.
type PrettyHandler struct {
	opts  slog.HandlerOptions
	w     io.Writer
	mu    sync.Mutex
	group string
	attrs []slog.Attr
}

// NewPrettyHandler creates a new PrettyHandler.
func NewPrettyHandler(w io.Writer, opts *slog.HandlerOptions) *PrettyHandler {
	if opts == nil {
		opts = &slog.HandlerOptions{}
	}
	return &PrettyHandler{
		opts: *opts,
		w:    w,
	}
}

// Enabled reports whether the handler handles records at the given level.
func (h *PrettyHandler) Enabled(_ context.Context, level slog.Level) bool {
	minLevel := slog.LevelInfo
	if h.opts.Level != nil {
		minLevel = h.opts.Level.Level()
	}
	return level >= minLevel
}

// Handle formats and writes a log record.
func (h *PrettyHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Format: [TIME] LEVEL message key=value key=value [source]
	buf := make([]byte, 0, 1024)

	// Timestamp
	buf = append(buf, colorGray...)
	buf = append(buf, '[')
	buf = r.Time.AppendFormat(buf, time.DateTime)
	buf = append(buf, ']')
	buf = append(buf, colorReset...)
	buf = append(buf, ' ')

	// Level with color
	levelColor := levelColor(r.Level)
	buf = append(buf, levelColor...)
	buf = append(buf, colorBold...)
	buf = append(buf, padLevel(r.Level.String())...)
	buf = append(buf, colorReset...)
	buf = append(buf, ' ')

	// Message
	buf = append(buf, r.Message...)

	// Attributes (including handler attrs and record attrs)
	attrs := make([]slog.Attr, 0, len(h.attrs)+r.NumAttrs())
	attrs = append(attrs, h.attrs...)
	r.Attrs(func(a slog.Attr) bool {
		attrs = append(attrs, a)
		return true
	})

	if len(attrs) > 0 {
		buf = append(buf, ' ')
		buf = append(buf, colorCyan...)
		for i, attr := range attrs {
			if i > 0 {
				buf = append(buf, ' ')
			}
			buf = appendAttr(buf, attr, h.group)
		}
		buf = append(buf, colorReset...)
	}

	buf = append(buf, '\n')

	_, err := h.w.Write(buf)
	return err
}

// WithAttrs returns a new handler with additional attributes.
func (h *PrettyHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	newAttrs := make([]slog.Attr, len(h.attrs)+len(attrs))
	copy(newAttrs, h.attrs)
	copy(newAttrs[len(h.attrs):], attrs)

	return &PrettyHandler{
		opts:  h.opts,
		w:     h.w,
		group: h.group,
		attrs: newAttrs,
	}
}

// WithGroup returns a new handler with a group name.
func (h *PrettyHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	newGroup := name
	if h.group != "" {
		newGroup = h.group + "." + name
	}
	return &PrettyHandler{
		opts:  h.opts,
		w:     h.w,
		group: newGroup,
		attrs: h.attrs,
	}
}

// Helper functions

func levelColor(level slog.Level) string {
	switch {
	case level >= slog.LevelError:
		return colorRed
	case level >= slog.LevelWarn:
		return colorYellow
	case level >= slog.LevelInfo:
		return colorBlue
	default:
		return colorGray
	}
}

func padLevel(level string) string {
	// Pad to 5 characters for alignment
	switch len(level) {
	case 4:
		return level + " "
	case 5:
		return level
	default:
		return level
	}
}

func appendAttr(buf []byte, attr slog.Attr, group string) []byte {
	key := attr.Key
	if group != "" {
		key = group + "." + key
	}

	buf = append(buf, key...)
	buf = append(buf, '=')

	switch attr.Value.Kind() {
	case slog.KindString:
		s := attr.Value.String()
		// Quote strings that contain spaces
		if needsQuoting(s) {
			buf = append(buf, '"')
			buf = append(buf, s...)
			buf = append(buf, '"')
		} else {
			buf = append(buf, s...)
		}
	case slog.KindTime:
		buf = attr.Value.Time().AppendFormat(buf, time.RFC3339)
	case slog.KindGroup:
		// Handle group values
		buf = append(buf, '{')
		attrs := attr.Value.Group()
		for i, a := range attrs {
			if i > 0 {
				buf = append(buf, ' ')
			}
			buf = appendAttr(buf, a, "")
		}
		buf = append(buf, '}')
	default:
		buf = append(buf, fmt.Sprint(attr.Value.Any())...)
	}

	return buf
}

func needsQuoting(s string) bool {
	for _, c := range s {
		if c == ' ' || c == '\t' || c == '\n' || c == '"' {
			return true
		}
	}
	return false
}
