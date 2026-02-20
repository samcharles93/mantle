package simd

import (
	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/hostcaps"
)

func (m *Instance) asCore() *core.Instance {
	if m == nil {
		return nil
	}
	return (*core.Instance)(m)
}

func (m *Instance) Ops() core.Ops {
	cm := m.asCore()
	if cm == nil {
		return core.DefaultOps{}
	}
	return cm.Ops()
}

func (m *Instance) SetOps(ops core.Ops) {
	cm := m.asCore()
	if cm == nil {
		return
	}
	cm.SetOps(ops)
}

func (m *Instance) SetHostCapabilities(caps *hostcaps.Snapshot) {
	cm := m.asCore()
	if cm == nil {
		return
	}
	cm.SetHostCapabilities(caps)
}

func (m *Instance) setHostCapabilities(caps *hostcaps.Snapshot) {
	m.SetHostCapabilities(caps)
}

func (m *Instance) BindDefaultOps() {
	cm := m.asCore()
	if cm == nil {
		return
	}
	cm.BindDefaultOps()
}

func (m *Instance) GetAttnPool() *core.AttnPool {
	cm := m.asCore()
	if cm == nil {
		return nil
	}
	return cm.GetAttnPool()
}

func (m *Instance) ModelConfig() *core.ModelConfig {
	cm := m.asCore()
	if cm == nil {
		return nil
	}
	return cm.ModelConfig()
}
