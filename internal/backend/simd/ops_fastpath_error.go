package simd

type fastPathErrorConsumer interface {
	ConsumeFastPathError() error
}

func consumeFastPathError(ops Ops) error {
	if ops == nil {
		return nil
	}
	if c, ok := ops.(fastPathErrorConsumer); ok {
		return c.ConsumeFastPathError()
	}
	return nil
}
