package main

import (
	"context"
	"errors"

	"github.com/urfave/cli/v3"
)

func quantizeCmd() *cli.Command {
	return &cli.Command{
		Name:  "quantize",
		Usage: "Quantise an existing .mcf (not wired yet)",
		Action: func(ctx context.Context, cmd *cli.Command) error {
			_ = ctx
			_ = cmd
			return errors.New("quantize: not implemented yet (next step: read MCF + rewrite TensorData/TensorIndex + add QuantInfo)")
		},
	}
}
