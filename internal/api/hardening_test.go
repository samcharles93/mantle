package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/labstack/echo/v5"
)

func newHardenedEcho(t *testing.T, mw ...echo.MiddlewareFunc) *echo.Echo {
	t.Helper()
	e := echo.New()
	for _, m := range mw {
		if m != nil {
			e.Use(m)
		}
	}
	e.GET("/v1/ping", func(c *echo.Context) error { return c.String(http.StatusOK, "pong") })
	e.GET("/static/app.js", func(c *echo.Context) error { return c.String(http.StatusOK, "js") })
	return e
}

func doRequest(t *testing.T, e *echo.Echo, path string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	e.ServeHTTP(rec, req)
	return rec
}

func TestAPIKeyAuthDisabledWithoutKey(t *testing.T) {
	t.Parallel()
	if mw := APIKeyAuth("  "); mw != nil {
		t.Fatal("an empty key must leave the API unauthenticated")
	}
}

func TestAPIKeyAuthProtectsV1Only(t *testing.T) {
	t.Parallel()
	e := newHardenedEcho(t, APIKeyAuth("secret"))

	if rec := doRequest(t, e, "/v1/ping", nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing key: status %d, want 401", rec.Code)
	}
	if rec := doRequest(t, e, "/v1/ping", map[string]string{"Authorization": "Bearer wrong"}); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key: status %d, want 401", rec.Code)
	}
	if rec := doRequest(t, e, "/v1/ping", map[string]string{"Authorization": "Bearer secret"}); rec.Code != http.StatusOK {
		t.Fatalf("bearer key: status %d, want 200", rec.Code)
	}
	if rec := doRequest(t, e, "/v1/ping", map[string]string{"X-API-Key": "secret"}); rec.Code != http.StatusOK {
		t.Fatalf("X-API-Key: status %d, want 200", rec.Code)
	}
	// The embedded web UI stays reachable so a browser can load it.
	if rec := doRequest(t, e, "/static/app.js", nil); rec.Code != http.StatusOK {
		t.Fatalf("static asset: status %d, want 200", rec.Code)
	}
}

func TestMaxConcurrentRejectsWhenFull(t *testing.T) {
	t.Parallel()
	e := newHardenedEcho(t, MaxConcurrent(1))

	running := make(chan struct{})
	release := make(chan struct{})
	e.GET("/v1/slow", func(c *echo.Context) error {
		close(running)
		<-release
		return c.String(http.StatusOK, "done")
	})

	done := make(chan int, 1)
	go func() {
		rec := httptest.NewRecorder()
		e.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/slow", nil))
		done <- rec.Code
	}()
	<-running

	rec := doRequest(t, e, "/v1/ping", nil)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("second request: status %d, want 503", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("Retry-After = %q, want 1", got)
	}

	close(release)
	if code := <-done; code != http.StatusOK {
		t.Fatalf("first request: status %d, want 200", code)
	}
	// The slot is released, so the next request goes through.
	if rec := doRequest(t, e, "/v1/ping", nil); rec.Code != http.StatusOK {
		t.Fatalf("after release: status %d, want 200", rec.Code)
	}
}

func TestMaxConcurrentDisabledAndSkipsNonAPIPaths(t *testing.T) {
	t.Parallel()
	if mw := MaxConcurrent(0); mw != nil {
		t.Fatal("n=0 must disable the cap")
	}
	e := newHardenedEcho(t, MaxConcurrent(1))
	e.GET("/static/slow.js", func(c *echo.Context) error { return c.String(http.StatusOK, "js") })

	running := make(chan struct{})
	release := make(chan struct{})
	e.GET("/v1/slow", func(c *echo.Context) error {
		close(running)
		<-release
		return c.String(http.StatusOK, "done")
	})
	go func() {
		rec := httptest.NewRecorder()
		e.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/slow", nil))
	}()
	<-running
	// Static assets do not consume a generation slot.
	if rec := doRequest(t, e, "/static/slow.js", nil); rec.Code != http.StatusOK {
		t.Fatalf("static request: status %d, want 200", rec.Code)
	}
	close(release)
}
