package logger

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

func TestDefault(t *testing.T) {
	t.Parallel()
	log := Default()
	if log == nil {
		t.Fatal("Default() returned nil")
	}
	// Should not panic
	log.Info("test message")
	log.Debug("debug message")
	log.Warn("warn message")
	log.Error("error message")
}

func TestJSON(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := JSON(&buf, slog.LevelInfo)
	log.Info("hello", "key", "value")

	output := buf.String()
	if !strings.Contains(output, "hello") {
		t.Fatalf("expected 'hello' in output, got: %s", output)
	}
	if !strings.Contains(output, `"key":"value"`) {
		t.Fatalf("expected key=value in JSON output, got: %s", output)
	}
	if !strings.Contains(output, `"level":"INFO"`) {
		t.Fatalf("expected level INFO in output, got: %s", output)
	}
}

func TestJSONLevelFiltering(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := JSON(&buf, slog.LevelWarn)
	log.Info("should not appear")
	log.Debug("also should not appear")

	if buf.Len() > 0 {
		t.Fatalf("expected no output for info/debug at warn level, got: %s", buf.String())
	}

	log.Warn("should appear")
	if !strings.Contains(buf.String(), "should appear") {
		t.Fatalf("expected warn message in output, got: %s", buf.String())
	}
}

func TestPretty(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := Pretty(&buf, slog.LevelInfo)
	log.Info("test message", "key", "value")

	output := buf.String()
	if !strings.Contains(output, "test message") {
		t.Fatalf("expected 'test message' in output, got: %s", output)
	}
	if !strings.Contains(output, "key=value") {
		t.Fatalf("expected 'key=value' in output, got: %s", output)
	}
}

func TestPrettyDebugLevel(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := Pretty(&buf, slog.LevelDebug)
	log.Debug("debug msg")

	if !strings.Contains(buf.String(), "debug msg") {
		t.Fatalf("expected debug message at debug level, got: %s", buf.String())
	}
}

func TestWith(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := JSON(&buf, slog.LevelInfo)
	childLog := log.With("component", "test")
	childLog.Info("child message")

	output := buf.String()
	if !strings.Contains(output, `"component":"test"`) {
		t.Fatalf("expected component=test in output, got: %s", output)
	}
	if !strings.Contains(output, "child message") {
		t.Fatalf("expected 'child message' in output, got: %s", output)
	}
}

func TestWithGroup(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := JSON(&buf, slog.LevelInfo)
	groupLog := log.WithGroup("mygroup")
	groupLog.Info("grouped message", "field", "val")

	output := buf.String()
	if !strings.Contains(output, "grouped message") {
		t.Fatalf("expected 'grouped message' in output, got: %s", output)
	}
}

func TestFromContextDefault(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	log := FromContext(ctx)
	if log == nil {
		t.Fatal("FromContext with no logger returned nil")
	}
	// Should not panic
	log.Info("from context")
}

func TestContextRoundTrip(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	log := JSON(&buf, slog.LevelInfo)

	ctx := WithContext(context.Background(), log)
	retrieved := FromContext(ctx)

	retrieved.Info("roundtrip test")
	if !strings.Contains(buf.String(), "roundtrip test") {
		t.Fatalf("expected message via context logger, got: %s", buf.String())
	}
}

func TestParseLevel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected slog.Level
	}{
		{"debug", slog.LevelDebug},
		{"info", slog.LevelInfo},
		{"warn", slog.LevelWarn},
		{"warning", slog.LevelWarn},
		{"error", slog.LevelError},
		{"unknown", slog.LevelInfo},
		{"", slog.LevelInfo},
		{"DEBUG", slog.LevelInfo}, // case-sensitive
	}

	for _, tc := range tests {
		result := ParseLevel(tc.input)
		if result != tc.expected {
			t.Errorf("ParseLevel(%q): expected %v, got %v", tc.input, tc.expected, result)
		}
	}
}

func TestPrettyHandlerEnabled(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})

	if h.Enabled(context.Background(), slog.LevelInfo) {
		t.Error("expected info to be disabled at warn level")
	}
	if !h.Enabled(context.Background(), slog.LevelWarn) {
		t.Error("expected warn to be enabled at warn level")
	}
	if !h.Enabled(context.Background(), slog.LevelError) {
		t.Error("expected error to be enabled at warn level")
	}
}

func TestPrettyHandlerWithAttrs(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, nil)

	h2 := h.WithAttrs([]slog.Attr{slog.String("service", "test")})
	logger := slog.New(h2)
	logger.Info("with attrs")

	output := buf.String()
	if !strings.Contains(output, "service=test") {
		t.Fatalf("expected 'service=test' in output, got: %s", output)
	}
}

func TestPrettyHandlerWithGroup(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, nil)

	h2 := h.WithGroup("mygroup")
	logger := slog.New(h2)
	logger.Info("grouped", "key", "val")

	output := buf.String()
	if !strings.Contains(output, "mygroup.key=val") {
		t.Fatalf("expected 'mygroup.key=val' in output, got: %s", output)
	}
}

func TestPrettyHandlerNestedGroups(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, nil)

	h2 := h.WithGroup("a")
	h3 := h2.WithGroup("b")
	logger := slog.New(h3)
	logger.Info("nested", "key", "val")

	output := buf.String()
	if !strings.Contains(output, "a.b.key=val") {
		t.Fatalf("expected 'a.b.key=val' in output, got: %s", output)
	}
}

func TestPrettyHandlerEmptyGroup(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, nil)

	h2 := h.WithGroup("")
	// WithGroup("") should return the same handler
	if h2 != h {
		t.Fatal("WithGroup empty string should return same handler")
	}
}

func TestPrettyQuotesStringsWithSpaces(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, nil)
	logger := slog.New(h)
	logger.Info("test", "msg", "hello world")

	output := buf.String()
	if !strings.Contains(output, `msg="hello world"`) {
		t.Fatalf("expected quoted string with spaces, got: %s", output)
	}
}

func TestPrettyNoQuoteSimpleStrings(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	h := NewPrettyHandler(&buf, nil)
	logger := slog.New(h)
	logger.Info("test", "key", "simple")

	output := buf.String()
	if !strings.Contains(output, "key=simple") {
		t.Fatalf("expected unquoted simple string, got: %s", output)
	}
	if strings.Contains(output, `key="simple"`) {
		t.Fatalf("simple strings should not be quoted, got: %s", output)
	}
}

func TestNeedsQuoting(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected bool
	}{
		{"simple", false},
		{"has space", true},
		{"has\ttab", true},
		{"has\nnewline", true},
		{`has"quote`, true},
		{"", false},
		{"no-special-chars", false},
	}

	for _, tc := range tests {
		result := needsQuoting(tc.input)
		if result != tc.expected {
			t.Errorf("needsQuoting(%q): expected %v, got %v", tc.input, tc.expected, result)
		}
	}
}
