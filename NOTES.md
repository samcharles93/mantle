# NOTES and MEMORIES

## Notes

- We don't currently parse `conv_bias` or `block_use_swiglu`, if a model were to set these to true, the inference would be wrong because we'd never load the tensor.
- `DefaultOps` methods wrap package-level functions (e.g. `MatVec`) to satisfy the `Ops` interface for backend swapping, while allowing direct internal reuse of the logic (like in `FusedRMSNormMatVecCPU`) without struct instantiation.

## Memories

- cmd/mantle should stay wiring-focused; reusable CLI behavior belongs in internal/cli/* (currently internal/cli/paths and internal/cli/ux).
