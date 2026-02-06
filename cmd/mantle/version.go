package main

import (
	"context"
	"fmt"

	"github.com/samcharles93/mantle/internal/version"

	"github.com/urfave/cli/v3"
)

func versionCmd() *cli.Command {
	return &cli.Command{
		Name:  "version",
		Usage: "Print version information",
		Action: func(ctx context.Context, cmd *cli.Command) error {
			info := version.Resolve()
			fmt.Printf("version:    %s\n", info.Version)
			if info.Commit != "" {
				fmt.Printf("commit:     %s\n", info.Commit)
			}
			if info.BuildTime != "" {
				fmt.Printf("build time: %s\n", info.BuildTime)
			}
			return nil
		},
	}
}
