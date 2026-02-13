package main

import (
	"context"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
	"github.com/samcharles93/mantle/internal/api"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/webui"
	"github.com/urfave/cli/v3"
)

func serveCmd() *cli.Command {
	var (
		addr           string
		readTimeout    time.Duration
		noWebUI        bool
		cudaWeightMode string
		reasoningFmt   string
		reasoningBgt   int64
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
			&cli.BoolFlag{
				Name:        "no-webui",
				Usage:       "disable embedded web UI",
				Destination: &noWebUI,
			},
			&cli.StringFlag{
				Name:        "cuda-weight-mode",
				Usage:       "cuda weight loading mode: auto, quant, dequant",
				Value:       "auto",
				Destination: &cudaWeightMode,
			},
			&cli.StringFlag{
				Name:        "reasoning-format",
				Usage:       "reasoning extraction mode: auto, none, deepseek, deepseek-legacy",
				Value:       "auto",
				Destination: &reasoningFmt,
			},
			&cli.Int64Flag{
				Name:        "reasoning-budget",
				Usage:       "reasoning budget control: -1 unrestricted, 0 disable thinking in template when supported",
				Value:       -1,
				Destination: &reasoningBgt,
			},
		),
		Action: func(ctx context.Context, cmd *cli.Command) error {
			log := logger.FromContext(ctx)

			// Apply config file defaults for flags not explicitly set
			cfg := LoadConfig()
			applyServeConfig(cmd, cfg, &addr)
			if reasoningBgt != -1 && reasoningBgt != 0 {
				return cli.Exit("error: --reasoning-budget must be -1 or 0", 1)
			}
			switch reasoningFmt {
			case "auto", "none", "deepseek", "deepseek-legacy":
			default:
				return cli.Exit("error: --reasoning-format must be one of: auto, none, deepseek, deepseek-legacy", 1)
			}
			switch strings.ToLower(strings.TrimSpace(cudaWeightMode)) {
			case "auto", "quant", "dequant":
				_ = os.Setenv("MANTLE_CUDA_WEIGHT_MODE", strings.ToLower(strings.TrimSpace(cudaWeightMode)))
			default:
				return cli.Exit("error: --cuda-weight-mode must be one of: auto, quant, dequant", 1)
			}
			// Keep verbose CUDA diagnostics gated behind debug mode.
			if debug && os.Getenv("MANTLE_CUDA_TRACE") == "" {
				_ = os.Setenv("MANTLE_CUDA_TRACE", "1")
			}

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
			service.SetReasoningDefaults(reasoningFmt, int(reasoningBgt))
			server := api.NewServer(store, service)
			e := echo.New()
			e.Use(middleware.RequestLogger())
			e.Use(middleware.Recover())
			server.Register(e)

			// Serve embedded web UI unless disabled
			if !noWebUI {
				// Serve static files from embedded filesystem at /static/*
				e.GET("/static/*", echo.WrapHandler(http.StripPrefix("/static", http.FileServer(webui.StaticFS()))))

				// Serve index.html at root
				e.GET("/", func(c *echo.Context) error {
					fs := webui.StaticFS()
					f, err := fs.Open("index.html")
					if err != nil {
						return echo.ErrNotFound
					}
					defer f.Close()

					stat, err := f.Stat()
					if err != nil {
						return echo.ErrNotFound
					}

					http.ServeContent(c.Response(), c.Request(), "index.html", stat.ModTime(), f)
					return nil
				})

				// SPA fallback - serve index.html for any other unknown route
				e.RouteNotFound("/*", func(c *echo.Context) error {
					// Only serve index.html for GET requests that aren't API routes
					if c.Request().Method == http.MethodGet && !strings.HasPrefix(c.Path(), "/v1/") {
						fs := webui.StaticFS()
						f, err := fs.Open("index.html")
						if err != nil {
							return echo.ErrNotFound
						}
						defer f.Close()

						stat, err := f.Stat()
						if err != nil {
							return echo.ErrNotFound
						}

						http.ServeContent(c.Response(), c.Request(), "index.html", stat.ModTime(), f)
						return nil
					}
					return echo.ErrNotFound
				})
				log.Info("web UI enabled", "address", addr)
			} else {
				log.Info("web UI disabled")
			}

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
