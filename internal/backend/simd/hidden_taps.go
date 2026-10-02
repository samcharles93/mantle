package simd

import "fmt"

// Hidden-state tap capture. core.HiddenTaps owns the buffer and its contract;
// this file holds the hooks the decode and batched prefill loops call.
//
// Capture is a read-only side effect: the value stored is the residual stream
// the layer already produced, so enabling taps cannot change any numeric
// result. It is off unless the caller passed HiddenTapLayers at load time; on
// that disabled path every hook below is a single length check.

// beginForwardCapture marks the start of a forward so HiddenTaps.Rows always
// describes the forward just performed rather than a mixture of two.
func (m *Instance) beginForwardCapture() {
	if len(m.HiddenTaps.Layers) != 0 {
		m.HiddenTaps.Rows = 0
	}
}

// captureTapRow records the post-FFN residual output of decoder layer layerIdx
// (or of the embedding when layerIdx == -1) for the current token. The
// single-token path captures one row per forward, so the row is always 0. On
// the device decode path the residual lives on the device, so it forces the
// host copy the debug hook already performs before reading it.
func (rt *tokenRuntimeState) captureTapRow(m *Instance, layerIdx int, x []float32) error {
	if len(m.HiddenTaps.Layers) == 0 {
		return nil
	}
	slot := m.HiddenTaps.SlotFor(layerIdx)
	if slot < 0 {
		return nil
	}
	if rt.ds != nil {
		rt.ds.SyncHostState(rt.x)
		if err := consumeFastPathError(rt.ops); err != nil {
			return fmt.Errorf("hidden tap sync failed: %w", err)
		}
	}
	m.HiddenTaps.StoreRow(slot, 0, x)
	return nil
}

// captureTapBatch records the post-FFN residual of decoder layer layerIdx (or
// the embedding when layerIdx == -1) for every row of a batched chunk. rowBase
// is the chunk's first tap row and x is its [rows, Hidden] block.
func (m *Instance) captureTapBatch(layerIdx, rowBase, rows int, x []float32) {
	t := &m.HiddenTaps
	if len(t.Layers) == 0 {
		return
	}
	slot := t.SlotFor(layerIdx)
	if slot < 0 {
		return
	}
	for r := range rows {
		t.StoreRow(slot, rowBase+r, x[r*t.Hidden:(r+1)*t.Hidden])
	}
}
