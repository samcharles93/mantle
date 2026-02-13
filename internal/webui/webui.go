// Package webui provides embedded static files for the Mantle web interface.
package webui

import (
	"embed"
	"io/fs"
	"net/http"
)

//go:embed static/*
var staticFS embed.FS

// StaticFS returns an http.FileSystem for the embedded static files.
func StaticFS() http.FileSystem {
	sub, err := fs.Sub(staticFS, "static")
	if err != nil {
		// This should never happen because we control the embed path
		panic(err)
	}
	return http.FS(sub)
}
