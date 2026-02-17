package hostcaps

import "context"

// Snapshot contains host hardware capabilities and effective CUDA policy flags.
type Snapshot struct {
	CPU    CPUFeatures
	CUDA   CUDAFeatures
	Policy CUDAPolicy
}

// CPUFeatures contains CPU ISA capabilities relevant to kernel dispatch.
type CPUFeatures struct {
	HasAVX2       bool
	HasAVX512     bool
	HasFMA        bool
	HasAVXVNNI    bool
	HasAVX512VNNI bool
}

// CUDAFeatures contains runtime CUDA availability for the current host.
type CUDAFeatures struct {
	CompiledWithCUDA bool
	HasCUDADevice    bool
	CUDADeviceCount  int
}

// CUDAPolicy contains effective CUDA policy toggles captured from env.
type CUDAPolicy struct {
	Trace                    bool
	Fuse                     bool
	QuantKernel              bool
	K4Raw                    bool
	LegacySync               bool
	Graphs                   bool
	AttnSoftmax              bool
	DisableFFNFastPath       bool
	DisableQKVFastPath       bool
	DisableAttnInnerFastPath bool
	TraceSync                bool
	WeightMode               string
}

// FromContext returns the host capability snapshot from context, if present.
func FromContext(ctx context.Context) *Snapshot {
	if caps, ok := ctx.Value(contextKey{}).(*Snapshot); ok {
		return caps
	}
	return nil
}

// WithContext stores the host capability snapshot in context.
func WithContext(ctx context.Context, caps *Snapshot) context.Context {
	if caps == nil {
		return ctx
	}
	return context.WithValue(ctx, contextKey{}, caps)
}

type contextKey struct{}
