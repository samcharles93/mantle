# Mantle Performance Optimization Plan

## Executive Summary

Mantle currently achieves **~130k ns** per quantized matvec operation (2048×2048). The target is **10x performance improvement** (≤13k ns). This requires both micro-optimizations and architectural changes within project constraints.

### Current Baseline Performance (12th Gen Intel i5-12500H)
| Operation | Time (ns) | Throughput | Speed vs FP32 |
|-----------|-----------|------------|---------------|
| MatVec Q4 (quantized) | 129,624 | ~32M elem/ms | 1.75x faster |
| MatVec K4 (quantized) | 131,560 | ~32M elem/ms | 1.75x faster |
| MatVec FP32 (pool) | 226,537 | ~18M elem/ms | 1.0x (baseline) |
| MatVec BF16 | 196,070 | ~21M elem/ms | 1.15x faster |
| GEMM 256³ | 365,597 | ~45M ops/ms | - |

## Critical Bottlenecks Identified

### 1. Memory Bandwidth Bound Operations
- Quantized matvec: 64-byte alignment but suboptimal SIMD utilization
- BF16/FP16: 64KB lookup tables competing with weight matrices for L2/L3 cache
- KV cache strided access causing poor cache locality in attention

### 2. Allocation Overhead
- **Pre-allocated buffers**: ✅ Implemented - eliminates `make([]int8, ...)` calls in hot loops
- Worker pool buffers now reused across operations

### 3. Suboptimal SIMD Utilization
- Only AVX2 utilized despite 64-byte alignment (AVX-512 compatible)
- Missing SIMD for 2/3/6-bit quantization unpacking
- BF16/FP16 scalar table lookups vs SIMD conversion

### 4. Cache Inefficiency
- Fixed batch size (4) regardless of L1/L2 cache sizes
- GEMM tile sizes (32×32×16) barely fits L1 cache (64KB)
- KV cache access pattern causes stride penalties

## Performance Improvement Roadmap

### Phase 1: Memory Access Optimizations (Estimated 2-3x gain)
- [x] **Pre-allocated matvec buffers** - eliminates GC pressure
- [x] **Cache-aware batching** - adaptive batch sizing based on L1/L2 cache
- [ ] **KV cache layout transposition** - store by head then token for contiguous access
- [ ] **GEMM tile size optimization** - adaptive based on problem dimensions

### Phase 2: SIMD Acceleration (Estimated 3-5x gain)
- [x] **SIMD bit unpacking framework** - stubs for 2/3/6-bit quantization
- [ ] **AVX-512 utilization** for 64-byte aligned quantized payloads
- [ ] **SIMD BF16/FP16 conversion** - replace table lookups with vector shifts
- [ ] **SIMD softmax for attention** - vectorized exp and sum operations
- [ ] **SIMD GEMM tile packing** - accelerate B matrix packing

### Phase 3: Architectural Improvements (Estimated 2-4x gain)
- [ ] **Model parallelism** - split layers across CPU cores
- [ ] **Tensor parallelism** - split individual tensors across cores
- [ ] **Pipeline parallelism** - overlap layer execution
- [ ] **Mmap acceleration** - optional memory mapping for MCF files

### Phase 4: Quantization Specialization (Estimated 1.5-2x gain)
- [ ] **Mixed precision quantization** - per-layer bit selection
- [ ] **Channel-wise scaling** - improve reconstruction accuracy
- [ ] **Asymmetric quantization** for activations (currently weights only)
- [ ] **KV cache quantization** - Q8_0/Q4_0 for attention cache

## Detailed Implementation Plan

### 1. Memory Access Patterns

#### 1.1 KV Cache Optimization
**Current**: `koff = t*ctx.kvStride + kvHead*ctx.headDim` - strided access  
**Optimized**: Transpose layout to `[head][token][dim]` for contiguous access

**Implementation**:
- Add transposition option in attention pool
- Benchmark for sequence lengths 512, 1024, 2048
- Estimated improvement: **2-3x attention speed**

#### 1.2 Cache-Aware GEMM Tiling
**Current**: Fixed 32×32×16 (64KB) - barely fits L1  
**Optimized**: Adaptive (64×64×32) for L2 cache (256KB), smaller for L1-bound problems

**Implementation**:
- Dynamic tile sizing based on `m×n×k` dimensions
- CPU cache size detection
- Estimated improvement: **20-30% GEMM speed**

### 2. SIMD Acceleration

#### 2.1 AVX-512 Quantization
**Current**: 64-byte alignment but only AVX2 used  
**Optimized**: AVX-512 SIMD for aligned quantized payloads

**Implementation**:
- CPU feature detection for AVX-512
- 64-byte vector loads/stores
- Estimated improvement: **2x quantized matvec**

#### 2.2 BF16/FP16 SIMD Conversion
**Current**: 64KB lookup tables causing cache pollution  
**Optimized**: SIMD bit-shift conversion

**Implementation**:
- Replace `bf16ToF32Table` with `vu.ExtendToUint32().ShiftAllLeft(16).AsFloat32x8()`
- Remove 64KB tables
- Estimated improvement: **30-50% BF16/FP16 matvec**

#### 2.3 SIMD Bit Unpacking
**Current**: Scalar bit-by-bit extraction (O(32×bits))  
**Optimized**: SIMD bit manipulation with masks

**Implementation**:
- AVX2 bit extraction for K2/K3/K6
- Unrolled 2-bit extraction (implemented)
- 3/6-bit SIMD with shift-and-mask patterns
- Estimated improvement: **5-10x K2/K3 unpacking**

### 3. Quantization Enhancements

#### 3.1 Mixed Precision
**Current**: Uniform quantization per tensor  
**Optimized**: Per-layer bit selection based on sensitivity

**Implementation**:
- Layer importance analysis
- 4-bit for attention, 8-bit for MLP
- Estimated improvement: **20-30% accuracy at same speed**

#### 3.2 KV Cache Quantization
**Current**: FP32/FP16 KV cache  
**Optimized**: Q8_0/Q4_0 quantization

**Implementation**:
- Add KV cache quantization options
- Per-token scale application
- Estimated improvement: **2x memory bandwidth reduction**

## Constraints & Considerations

### Project Scope Compliance
- ✅ **Not inference engine**: Execution substrate only
- ✅ **Not training framework**: No fine-tuning features
- ✅ **No implicit behavior**: Runtime decisions remain explicit
- ✅ **Portable I/O**: mmap optional, `io.ReaderAt` supported

### Technical Constraints
- **No new dependencies** - All optimizations must use stdlib + `simd/archsimd`
- **Deterministic output** - No map iteration order dependence
- **Bounds-checked parsing** - Overflow-safe, validation-first
- **Portability** - Works without mmap, OS-specific behind build tags

### Performance Targets
| Component | Current | Target (10x) | Required Improvement |
|-----------|---------|---------------|---------------------|
| Quantized matvec | 130k ns | 13k ns | 10× |
| Attention (512 seq) | ~3.4M ns | 340k ns | 10× |
| GEMM 256³ | 366k ns | 37k ns | 10× |
| End-to-end inference | TBD | TBD | 10× |

## Implementation Priority

### High Priority (P0)
1. **SIMD bit unpacking** - 5-10x gain for K2/K3 quantization
2. **KV cache layout** - 2-3x attention speed
3. **BF16/FP16 SIMD** - 1.5-2x matvec speed

### Medium Priority (P1)
4. **Cache-aware GEMM** - 20-30% improvement
5. **AVX-512 utilization** - 2x quantized operations
6. **SIMD softmax** - 3x attention speed

### Low Priority (P2)
7. **Mixed precision quantization** - accuracy/speed tradeoff
8. **KV cache quantization** - memory bandwidth reduction
9. **Model parallelism** - multi-core scaling

## Testing & Validation

### Benchmarks Required
1. **Micro-benchmarks**: Individual kernel performance
2. **Integration tests**: Full model execution
3. **Correctness tests**: Deterministic output verification
4. **Memory tests**: Allocation patterns, cache efficiency

### Success Criteria
- [ ] **10x quantized matvec speed** (≤13k ns)
- [ ] **Zero allocations in hot paths** (confirmed via `-benchmem`)
- [ ] **Correctness preserved** (bit-exact where required)
- [ ] **No new dependencies** (stdlib + archsimd only)
- [ ] **Portability maintained** (works without mmap)

## Risk Assessment

### Technical Risks
1. **AVX-512 availability** - Fallback to AVX2 required
2. **Cache coherence** - Multi-core optimizations may interfere
3. **Numerical stability** - SIMD approximations may affect output

### Mitigation Strategies
- **Feature detection** at runtime with fallbacks
- **Conservative tuning** with performance guards
- **Extensive testing** with tolerance thresholds

## Timeline Estimate

### Week 1-2: Memory Access Optimizations
- Complete KV cache layout
- Implement cache-aware GEMM
- Benchmark memory bandwidth improvements

### Week 3-4: SIMD Acceleration
- Complete SIMD bit unpacking
- Implement BF16/FP16 SIMD
- Add SIMD softmax for attention

### Week 5-6: Architectural Improvements
- AVX-512 utilization
- Model parallelism exploration
- Integration testing

### Week 7-8: Quantization Specialization
- Mixed precision implementation
- KV cache quantization
- Performance validation

## Conclusion

Achieving 10x performance improvement requires addressing multiple bottlenecks simultaneously. The most significant gains will come from:
1. **Memory access patterns** (2-3×)
2. **SIMD utilization** (3-5×) 
3. **Architectural improvements** (2-4×)

Combined, these optimizations should provide the required 10x speedup while maintaining Mantle's core principles of explicit control, portability, and deterministic behavior.

---

*Last Updated: ${DATE}  
Baseline: Quantized matvec ≈130k ns  
Target: Quantized matvec ≤13k ns (10× improvement)*