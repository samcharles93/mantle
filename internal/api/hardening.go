package api

import (
	"crypto/subtle"
	"fmt"
	"net/http"
	"strings"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
)

// apiV1Path reports whether a request targets the inference API. Static UI
// assets are excluded so a browser can still load the page when auth is on.
func apiV1Path(c *echo.Context) bool {
	return strings.HasPrefix(c.Request().URL.Path, "/v1/")
}

// APIKeyAuth requires the configured key on every /v1 request, accepted as
// "Authorization: Bearer <key>" or "X-API-Key: <key>". An empty key returns nil
// and leaves the API unauthenticated, which is the local-development default.
func APIKeyAuth(key string) echo.MiddlewareFunc {
	key = strings.TrimSpace(key)
	if key == "" {
		return nil
	}
	return middleware.KeyAuthWithConfig(middleware.KeyAuthConfig{
		Skipper:   func(c *echo.Context) bool { return !apiV1Path(c) },
		KeyLookup: "header:" + echo.HeaderAuthorization + ":Bearer ,header:X-API-Key",
		Validator: func(_ *echo.Context, got string, _ middleware.ExtractorSource) (bool, error) {
			return subtle.ConstantTimeCompare([]byte(got), []byte(key)) == 1, nil
		},
		ErrorHandler: func(c *echo.Context, _ error) error {
			return writeError(c, http.StatusUnauthorized, "invalid_request_error",
				"missing or invalid API key", "", "invalid_api_key")
		},
	})
}

// MaxConcurrent rejects /v1 requests with 503 once n are in flight, so a burst
// fails fast instead of queueing behind a busy model indefinitely. n <= 0
// returns nil and applies no cap.
func MaxConcurrent(n int) echo.MiddlewareFunc {
	if n <= 0 {
		return nil
	}
	slots := make(chan struct{}, n)
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c *echo.Context) error {
			if !apiV1Path(c) {
				return next(c)
			}
			select {
			case slots <- struct{}{}:
				defer func() { <-slots }()
				return next(c)
			default:
				c.Response().Header().Set("Retry-After", "1")
				return writeError(c, http.StatusServiceUnavailable, "server_error",
					fmt.Sprintf("server is at its limit of %d concurrent requests", n), "", "server_busy")
			}
		}
	}
}
