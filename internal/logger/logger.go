package logger

import (
	"context"
	"io"
	"log/slog"
	"os"
)

// Logger is the common interface for logging in Mantle.
// It wraps slog.Logger to allow for dependency injection and testing.
type Logger interface {
	Debug(msg string, args ...any)
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
	With(args ...any) Logger
	WithGroup(name string) Logger
}

// SlogLogger is a Logger implementation that wraps slog.Logger.
type SlogLogger struct {
	logger *slog.Logger
}

// New creates a new Logger with the given handler.
func New(handler slog.Handler) Logger {
	return &SlogLogger{
		logger: slog.New(handler),
	}
}

// Default creates a Logger with default text handler writing to stderr.
func Default() Logger {
	return New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
}

// JSON creates a Logger with JSON handler for production use.
func JSON(w io.Writer, level slog.Level) Logger {
	return New(slog.NewJSONHandler(w, &slog.HandlerOptions{
		AddSource: true,
		Level:     level,
	}))
}

// Pretty creates a Logger with colored pretty output for CLI use.
func Pretty(w io.Writer, level slog.Level) Logger {
	return New(NewPrettyHandler(w, &slog.HandlerOptions{
		AddSource: true,
		Level:     level,
	}))
}

// FromContext retrieves a Logger from the context.
// If no logger is found, returns a default logger.
func FromContext(ctx context.Context) Logger {
	if logger, ok := ctx.Value(loggerKey{}).(Logger); ok {
		return logger
	}
	return Default()
}

// WithContext adds the logger to the context.
func WithContext(ctx context.Context, logger Logger) context.Context {
	return context.WithValue(ctx, loggerKey{}, logger)
}

type loggerKey struct{}

// Implementation of Logger interface

func (l *SlogLogger) Debug(msg string, args ...any) {
	l.logger.Debug(msg, args...)
}

func (l *SlogLogger) Info(msg string, args ...any) {
	l.logger.Info(msg, args...)
}

func (l *SlogLogger) Warn(msg string, args ...any) {
	l.logger.Warn(msg, args...)
}

func (l *SlogLogger) Error(msg string, args ...any) {
	l.logger.Error(msg, args...)
}

func (l *SlogLogger) With(args ...any) Logger {
	return &SlogLogger{
		logger: l.logger.With(args...),
	}
}

func (l *SlogLogger) WithGroup(name string) Logger {
	return &SlogLogger{
		logger: l.logger.WithGroup(name),
	}
}

// ParseLevel converts a string level to slog.Level.
func ParseLevel(level string) slog.Level {
	switch level {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
