package main

import (
	"context"
	"net/http"
	"time"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
	"github.com/samcharles93/mantle/internal/api"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
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
			log := logger.FromContext(ctx)

			store := api.NewResponseStore()
			loader := inference.Loader{
				TokenizerJSONPath:   tokenizerJSONPath,
				TokenizerConfigPath: tokenizerConfig,
				ChatTemplatePath:    chatTemplate,
				Backend:             backend,
			}
			provider := api.NewCachedEngineProvider(api.EngineProviderConfig{
				DefaultModelPath: modelPath,
				ModelsPath:       modelsPath,
				MaxContext:       int(maxContext),
				Loader:           loader,
			})
			service := api.NewInferenceService(provider)
			server := api.NewServer(store, service)
			e := echo.New()
			e.Use(middleware.RequestLogger())
			e.Use(middleware.Recover())
			server.Register(e)
			log.Info("starting server", "address", addr)
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
