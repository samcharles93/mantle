# CUDA Performance Optimization Roadmap

**Date**: 2026-02-06  
**Status**: In progress (fast‑path regression detected)  
**Goal**: Reduce D2H/H2D copies, eliminate stream syncs, increase TPS for K4‑quantized models.

---

## 1. Current Situation

### Achieved Milestones
- ✅ Sync reduction (stream syncs dropped from ~52k to 0)
- ✅ Attention inner kernel (device‑resident KV cache)
- ✅ Combined attention inner + projection fast‑path (eliminates one D2H copy)
- ✅ Post‑attention norm on device (integrated into fast‑path)
- ✅ Quant‑cache building disabled for CUDA backend (load time ~1.0s)

### Performance Metrics (Before Regression)
- **Gen TPS**: ~43.8
- **Stream syncs**: 0
- **D2H bytes**: ~167 MB (still high)
- **H2D bytes**: ~760 MB (still high)

### Regression (After Rebuild)
- **Stream syncs**: 858 (matches StoreKV calls)
- **Gen TPS**: ~36.8 (drop of ~7 TPS)
- **Root cause**: Combined fast‑path not being used → fallback to CPU path with `StoreKV`.

---

## 2. Immediate Next Step: Diagnose & Fix Fast‑Path Regression

**Objective**: Restore 0 stream syncs before proceeding.

### Subtasks
1. **Add diagnostic logging** to `AttentionInnerProjection` in `internal/backend/cuda/ops.go`
   - Log entry and exit
   - Log which branch is taken (norm vs no‑norm)
   - Log any error from `deviceNormWeight`, `ensureActTmp`, `rmsNormDevice`
2. **Verify interface alignment** between `attentionInnerProjectionFastPath` (in `attention.go`) and CUDA implementation.
3. **Check environment variables** (`MANTLE_CUDA_DISABLE_ATTN_INNER_FASTPATH`, `MANTLE_CUDA_QUANT_KERNEL`).
4. **Test with a minimal model** to isolate whether regression is due to signature change or norm‑weight upload.
5. **If needed, revert norm‑device addition** temporarily to confirm baseline works.
6. **Fix any discovered bug** (e.g., empty `PostAttnNorm` slice, allocation failure).
7. **Rebuild & re‑run benchmark**, confirm stream syncs = 0.

**Expected outcome**: Stream syncs return to 0, TPS back to ~43.8.

---

## 3. Keep Residual Addition on Device

**Objective**: Eliminate D2H copy for the residual addition `Add(x, opOut)` in `runtime.go`.

### Current Flow (CPU)
1. Attention block output (`opOut`) is on host after fast‑path.
2. `Add(x, opOut)` adds `opOut` to `x` (host buffers).
3. Result stays on host for next block.

### Proposed Flow (Device)
- Keep `x` (hidden state) on device across layers.
- After attention block, `opOut` stays on device.
- Perform device‑side vector addition (CUDA kernel or BLAS `Axpy`).
- Result remains on device for FFN block.

### Implementation Steps
1. **Add device buffer for `x`** in CUDA Ops (e.g., `xStateDev`).
2. **Upload initial `x`** at start of generation (once per token).
3. **Modify `runtime.go`** to work with device buffers:
   - Replace host `x` with device pointer where possible.
   - Create fast‑path interface for residual addition (`AddDevice`).
4. **Update attention fast‑path** to write output directly to device buffer (or a temporary device buffer).
5. **Add device‑side addition kernel** (`native.AxpyF32`).
6. **Keep FFN input on device** (requires FFN fast‑path to accept device buffer).
7. **After FFN block, copy final `x` back to host** only when needed (e.g., for logits).

### Risks & Considerations
- **Memory**: Additional device buffer for `x` (size = model dimension).
- **Complexity**: Changes to `runtime.go` must remain compatible with CPU backend.
- **Fallback**: If device addition fails, must have a safe CPU fallback.

**Expected benefit**: Eliminate ~2 D2H copies per layer (attention output and FFN output). Could reduce D2H bytes by ~80%.

---

## 4. Keep FFN Boundary on Device

**Objective**: Single D2H copy at the block boundary (after FFN residual).

### Current Flow
1. FFN block runs on host (or partial device fast‑path).
2. FFN output copied to host.
3. Residual addition on host.
4. Result stays on host for next layer.

### Proposed Flow
- Keep FFN block entirely on device (already have `FFNFastPath`).
- Perform residual addition on device (as above).
- Only copy final `x` back to host **after the entire transformer block** (i.e., after FFN residual).
- This reduces D2H copies from **per‑layer** to **per‑token** (or per‑block if we keep across layers).

### Implementation Steps
1. **Extend FFN fast‑path** to accept device input buffer and write to device output buffer.
2. **Integrate with residual addition** (reuse device‑side addition).
3. **Modify layer loop** in `runtime.go` to keep `x` on device across attention and FFN sub‑blocks.
4. **Add a single D2H copy** after the FFN residual, before the next token (or before logits).
5. **Handle edge cases**: Mamba, MoE, short‑conv (may need separate device paths).

**Expected benefit**: D2H bytes drop from ~167 MB to ~10 MB (estimate). TPS increase of 10‑20%.

---

## 5. Implement Ping‑Pong Buffering

**Objective**: Overlap compute and copy using double‑buffering.

### Current Limitation
- H2D copies (input vectors) block compute.
- D2H copies (output) block next kernel.

### Solution
- Allocate two device buffers for input/output.
- While kernel processes buffer N, copy buffer N+1 to/from host.
- Requires asynchronous streams and careful dependency management.

### Implementation Steps
1. **Add second stream** to CUDA Ops (optional but beneficial).
2. **Duplicate device buffers** (`xDev1`, `xDev2`, `yDev1`, `yDev2`).
3. **Modify `ensureDeviceVecs`** to support double‑buffering.
4. **Implement async H2D copy** before kernel launch.
5. **Implement async D2H copy** after kernel completion.
6. **Use CUDA events** for synchronization between streams.
7. **Integrate with fast‑path kernels** (attention inner, projection, FFN).

**Expected benefit**: Hide copy latency, improve TPS by 5‑10% (bottleneck may shift to compute).

---

## 6. Profile and Evaluate

**Objective**: Quantify each optimization’s impact and identify new bottlenecks.

### Metrics to Track
- **TPS** (prompt & generation)
- **Stream syncs**
- **H2D / D2H bytes**
- **Device memory allocations**
- **Kernel occupancy** (via `nsys`)

### Tools
- `MANTLE_CUDA_TRACE=1` (built‑in trace)
- `nsys profile` (CUDA GPU kernel timeline)
- `nvprof` (if available)
- CPU profiling (`pprof`) to identify remaining host‑side overhead.

### Evaluation Steps
1. **Baseline** (after fixing regression) – record all metrics.
2. **After residual addition** – compare D2H bytes.
3. **After FFN boundary** – compare D2H bytes and TPS.
4. **After ping‑pong buffering** – measure overlap efficiency.
5. **Identify new bottlenecks** (e.g., kernel launch overhead, quantization decode).

---

## 7. Long‑Term / Stretch Goals

### Quant‑Native Kernel Optimization
- Current `MANTLE_CUDA_QUANT_KERNEL=1` causes severe regression (TPS ~6.3).
- Diagnose kernel performance (memory access pattern, occupancy).
- Compare with int8 block kernel using profiling.
- Optimize or replace with better kernel (e.g., using Tensor Cores).

### CUDA Graphs
- Capture entire token generation as a CUDA graph.
- Reduce CGO call overhead.
- Requires all kernels and copies to be graph‑capturable.

### Unified Memory Optimization
- Use CUDA managed memory with concurrent access.
- Avoid explicit H2D/D2H copies altogether (requires careful prefetching).

### Kernel Fusion
- Fuse attention inner + projection + norm + residual into a single kernel.
- Advanced; may require significant kernel redesign.

---

## 8. Dependencies & Constraints

### Hard Constraints (from AGENTS.md)
- No new third‑party dependencies.
- Must work without mmap (random‑access path required).
- Deterministic output, no map‑iteration order dependence.
- All changes must compile and be runnable by default.

### Testing Requirements
- Run `gofmt`, `golangci‑lint`, `go test ./...` after each change.
- Ensure CUDA backend works with all model types (Mamba, MoE, etc.).
- No regression on CPU backend.

---

## 9. Prioritized Task List

1. **Fix fast‑path regression** (critical)
2. **Residual addition on device** (high impact)
3. **FFN boundary on device** (high impact)
4. **Ping‑pong buffering** (moderate impact)
5. **Quant‑native kernel optimization** (if time permits)
6. **CUDA graphs** (stretch)

Each task should be accompanied by:
- A design document (if scope large)
- Implementation in a separate branch
- Performance measurement before/after
- Integration into main after review.

---

## 10. Risk Mitigation

- **Regression risk**: Keep CPU fallback for every fast‑path; test with `MANTLE_CUDA_DISABLE_*` flags.
- **Complexity risk**: Isolate device‑side changes behind interfaces; maintain CPU compatibility.
- **Performance risk**: Profile each step; revert if performance degrades.
- **Maintenance risk**: Document all new device buffers and streams.

---

## 11. Next Actions

1. **Immediate**: Switch to **code mode** to add diagnostic logging and fix fast‑path regression.
2. **After regression fixed**: Re‑evaluate whether to proceed with residual addition or profile first.
3. **Iterate** through the prioritized task list, measuring impact at each step.

**Decision point**: After fixing regression, should we:
- A) Proceed with residual addition on device?
- B) Profile current performance to identify the biggest remaining bottleneck?
- C) Implement ping‑pong buffering first (may be simpler)?

Let’s discuss and decide.