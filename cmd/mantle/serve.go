package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
	"github.com/samcharles93/mantle/internal/api"
	"github.com/urfave/cli/v3"
)

func serveCmd() *cli.Command {
	var (
		addr        string
		readTimeout time.Duration
	)

	return &cli.Command{
		Name:  "serve",
		Usage: "Serve the REST API (Responses API)",
		Flags: append(commonModelFlags(),
			&cli.StringFlag{
				Name:        "addr",
				Usage:       "listen address",
				Value:       "127.0.0.1:8080",
				Destination: &addr,
			},
			&cli.DurationFlag{
				Name:        "read-timeout",
				Usage:       "read timeout",
				Value:       30 * time.Second,
				Destination: &readTimeout,
			},
		),
		Action: func(ctx context.Context, cmd *cli.Command) error {
			store := api.NewResponseStore()
			server := api.NewServer(store)
			e := echo.New()
			e.Use(middleware.RequestLogger())
			e.Use(middleware.Recover())
			server.Register(e)
			fmt.Printf("listening on %s\n", addr)
			sc := echo.StartConfig{
				Address: addr,
				BeforeServeFunc: func(srv *http.Server) error {
					srv.ReadHeaderTimeout = readTimeout
					return nil
				},
			}
			return sc.Start(ctx, e)
		},
	}
}
