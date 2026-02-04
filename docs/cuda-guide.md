# CUDA Programming Guide for C++ Developers

This guide distills the CUDA Programming Guide (Release 13.1) into a practical, in-depth roadmap for experienced C++ developers who want to build correct and fast CUDA code. It focuses on the CUDA programming model, unified memory, kernel construction, and performance optimization.

---

## 1. Mental Model: CPU + GPU, Host + Device

CUDA assumes a heterogeneous system:
- **Host**: CPU and its memory.
- **Device**: GPU and its memory.

A CUDA application starts on the CPU. Host code launches **kernels** (functions executed on the GPU) and manages memory movement. Kernel launches are **asynchronous**; you must synchronize to ensure completion.

Key ideas:
- A kernel launch creates many **threads**, organized into **thread blocks** and a **grid**.
- Threads within a block can cooperate via **shared memory** and synchronization.
- Thread blocks can execute in any order. Kernels must not depend on inter-block ordering.

---

## 2. Toolchain and Build Basics

### 2.1 CUDA Toolkit and Driver
You need:
- A supported NVIDIA GPU.
- NVIDIA driver.
- CUDA Toolkit (includes `nvcc`, headers, libraries).

### 2.2 `nvcc` as the Compiler Driver
`nvcc` compiles C++ host code and CUDA device code from `.cu` files.

Basic build:
```bash
nvcc vec_add.cu -o vec_add
```

### 2.3 Compute Capability and Fat Binaries
Each GPU has a **compute capability** (e.g., 8.6), which maps to an SM version (e.g., `sm_86`). You can compile for multiple architectures and include PTX for forward compatibility (JIT compilation at runtime).

Example:
```bash
nvcc vec_add.cu -o vec_add \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86
```

### 2.4 Targeting Architectures and Separate Compilation
Target a specific GPU architecture with `-arch`:
```bash
nvcc main.cu -o app -arch=sm_80
```

Use separate compilation when device code spans multiple translation units:
```bash
nvcc -rdc=true fileA.cu fileB.cu -o app
```

---

## 3. CUDA C++ Essentials

### 3.1 Kernel Declarations
- `__global__` marks a kernel callable from host.
- Kernels return `void`.

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) C[i] = A[i] + B[i];
}
```

### 3.2 Launch Configuration (Triple Chevron)
```cpp
int threads = 256;
int blocks  = (n + threads - 1) / threads;
vecAdd<<<blocks, threads>>>(A, B, C, n);
```

For 2D or 3D:
```cpp
dim3 grid(16, 16);
dim3 block(8, 8);
MatAdd<<<grid, block>>>(A, B, C);
```

### 3.3 Thread and Grid Intrinsics
Inside a kernel:
- `threadIdx` gives thread index inside its block.
- `blockIdx` gives block index inside the grid.
- `blockDim`, `gridDim` give dimensions.

These are 3D vectors: `.x`, `.y`, `.z`.

### 3.4 Function Specifiers
CUDA extends C++ with execution-space annotations:
- `__global__`: Callable from host, runs on device (kernel entry point).
- `__device__`: Callable from device, runs on device.
- `__host__`: Callable from host, runs on host (default).
- `__host__ __device__`: Compiles for both host and device.

---

## 4. Execution Model: Warps and Divergence

Threads are executed in groups of 32 called **warps**. All threads in a warp execute the same instruction. If some threads take one branch and others take another, the warp **diverges**, and execution is serialized.

**Optimization rule**: minimize divergent control flow within a warp.

---

## 5. Memory Model

### 5.1 Memory Types and Scope
| Memory | Scope | Lifetime | Location |
|---|---|---|---|
| Global | Grid | Application | Device DRAM |
| Constant | Grid | Application | Device |
| Shared | Block | Kernel | SM |
| Local | Thread | Kernel | Device (global memory) |
| Registers | Thread | Kernel | SM |

### 5.2 Global Memory
Global memory is the main device memory. It is large but has high latency. Good access patterns are critical for performance.

### 5.3 Shared Memory
Shared memory is on-chip, low latency, block-scoped. It can be used as a user-managed cache. The shared memory pool shares hardware with L1 cache.

Static allocation:
```cpp
__shared__ float tile[32][32];
```

Dynamic allocation:
```cpp
extern __shared__ float smem[];
```

### 5.4 Registers and Local Memory
Registers are fast but limited. If a kernel uses too many registers, occupancy drops and spills go to local memory (which is in global memory).

### 5.5 Constant Memory
Good for small read-only data shared by all threads.

```cpp
__constant__ float coeffs[4];
```

---

## 6. Unified Virtual Addressing and Unified Memory

### 6.1 Unified Virtual Address Space
CUDA uses a unified virtual address space across host and all GPUs in a process. This lets CUDA infer copy direction with `cudaMemcpyDefault` and allows pointer inspection via `cudaPointerGetAttributes()`.

### 6.2 Unified Memory Overview
Unified memory lets you allocate memory once and access it from CPU or GPU:
- Allocate with `cudaMallocManaged` or `__managed__`.
- Driver migrates data to the processor that accesses it.

### 6.3 Simple Unified Memory Example
```cpp
void unifiedMemExample(int n) {
    float *A = nullptr, *B = nullptr, *C = nullptr;

    cudaMallocManaged(&A, n * sizeof(float));
    cudaMallocManaged(&B, n * sizeof(float));
    cudaMallocManaged(&C, n * sizeof(float));

    initArray(A, n);
    initArray(B, n);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
```

### 6.4 Unified Memory Paradigms (Behavior Depends on Platform)
CUDA exposes attributes to determine which unified memory paradigm is available:
- `cudaDevAttrConcurrentManagedAccess`
- `cudaDevAttrPageableMemoryAccess`
- `cudaDevAttrPageableMemoryAccessUsesHostPageTables`

Example query:
```cpp
int concurrent = 0, pageable = 0, hw_coherent = 0;
int dev = 0;
cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev);
cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
cudaDeviceGetAttribute(&hw_coherent,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev);
```

Interpretation:
- **Full unified memory** if `concurrentManagedAccess` is 1.
- If `pageableMemoryAccess` is 1, system allocations are also unified.
- If `pageableMemoryAccessUsesHostPageTables` is 1, coherence is hardware-based.

### 6.5 Unified Memory Performance Control
You can guide placement and access:
- `cudaMemAdviseSetPreferredLocation`
- `cudaMemAdviseSetAccessedBy`
- `cudaMemPrefetchAsync`

Example: prefer host residency and allow device access (direct write):
```cpp
int* data;
size_t bytes = sizeof(int) * 16;
cudaMallocManaged(&data, bytes);

cudaMemLocation host_loc = { .type = cudaMemLocationTypeHost };
cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, host_loc);
cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, host_loc);

kernel<<<1, 16>>>(data);
cudaDeviceSynchronize();
```

### 6.6 Unified Memory Limitations
On some platforms (e.g., Windows or GPUs without concurrent managed access), unified memory has limits:
- No fine-grained GPU page faulting.
- No concurrent CPU/GPU access to managed memory while a kernel is active.
- No oversubscription beyond GPU memory.

In these cases, you must synchronize before CPU access:
```cpp
kernel<<<1,1>>>(...);
cudaDeviceSynchronize();
// Now safe for CPU to access managed memory
```

---

## 7. Explicit Memory Management

Explicit memory gives more control and can improve performance.

```cpp
float *A, *B, *C;
float *dA, *dB, *dC;

cudaMallocHost(&A, n * sizeof(float));
cudaMallocHost(&B, n * sizeof(float));
cudaMallocHost(&C, n * sizeof(float));

cudaMalloc(&dA, n * sizeof(float));
cudaMalloc(&dB, n * sizeof(float));
cudaMalloc(&dC, n * sizeof(float));

cudaMemcpy(dA, A, n * sizeof(float), cudaMemcpyDefault);
cudaMemcpy(dB, B, n * sizeof(float), cudaMemcpyDefault);

vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
cudaDeviceSynchronize();

cudaMemcpy(C, dC, n * sizeof(float), cudaMemcpyDefault);
```

Notes:
- `cudaMallocHost` gives page-locked host memory, required for async transfers.
- `cudaMemcpy` is synchronous; use `cudaMemcpyAsync` for overlap.

---

## 8. Synchronization and Streams

Kernel launches are asynchronous. Common sync methods:
- `cudaDeviceSynchronize()` for full device sync.
- `cudaStreamSynchronize(stream)` for stream-specific.

CUDA streams let you overlap kernel execution and memory transfers.

Example with streams:
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(dA, A, bytes, cudaMemcpyDefault, stream);
vecAdd<<<blocks, threads, 0, stream>>>(dA, dB, dC, n);
cudaMemcpyAsync(C, dC, bytes, cudaMemcpyDefault, stream);

cudaStreamSynchronize(stream);
```
Note: Asynchronous host-device copies require pinned host memory (`cudaMallocHost`).

---

## 9. Kernel Optimization Essentials

### 9.1 Coalesced Global Memory Access
Global memory transactions happen in 32-byte segments. You want threads in a warp to access data in the same segment.

Rule of thumb:
- **Consecutive threads should access consecutive addresses**.

This is one of the most important performance considerations.

Data layout tip:
- Prefer **Structure of Arrays (SoA)** over **Array of Structures (AoS)** when possible so adjacent threads read adjacent elements.

### 9.2 Shared Memory to Improve Coalescing
A naive matrix transpose reads coalesced but writes strided. Using shared memory as a tile buffer fixes this.

Sketch (simplified):
```cpp
__shared__ float tile[32][32];

// Load from global to shared (coalesced)
tile[threadIdx.x][threadIdx.y] = a[...];
__syncthreads();

// Store from shared to global (coalesced)
c[...] = tile[threadIdx.y][threadIdx.x];
```

### 9.3 Avoid Shared Memory Bank Conflicts
Shared memory is split into banks. Access patterns that map multiple threads to the same bank cause conflicts.

Common fix for transpose tiles:
```cpp
__shared__ float tile[32][33]; // +1 padding avoids bank conflicts
```

### 9.4 Control Divergence
Minimize branch divergence within warps. Refactor conditionals or split kernels if necessary.

### 9.5 Manage Register Pressure
Registers are per-thread and limited per SM. Excess registers reduce occupancy and can spill to local memory.

Options:
- Reduce temporary variables.
- Use `-maxrregcount` if needed, but beware of spills.

### 9.6 Launch Bounds and Occupancy Hints
Use launch bounds to hint desired occupancy and cap register usage:
```cpp
__global__ void __launch_bounds__(256, 4) myKernel(...) {
    // Max 256 threads per block, min 4 blocks per SM
}
```

### 9.7 Use Constant Memory for Small Read-Only Data
Constant memory is cached and efficient when all threads read the same values.

### 9.8 Be Careful with Atomics
Atomics serialize access. Use sparingly or redesign algorithms to reduce contention.

---

## 10. Example: End-to-End Vector Add (Unified Memory + Coalesced Access)

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    for (int i = 0; i < n; ++i) {
        A[i] = 1.0f; B[i] = 2.0f;
    }

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();

    printf("C[0] = %f\n", C[0]);

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
```

Build:
```bash
nvcc vec_add_um.cu -o vec_add_um
```

---

## 11. Practical Checklist for New Kernels

1. **Define the parallel mapping** (thread index to data index).
2. **Pick block size** (start with 128-256 threads).
3. **Ensure coalesced global memory access**.
4. **Use shared memory when it improves data reuse**.
5. **Avoid warp divergence in hot paths**.
6. **Keep register usage reasonable**.
7. **Prefer unified memory for simplicity, explicit memory for peak performance**.
8. **Profile and iterate** (Nsight tools or timing kernels manually).

---

## 12. When to Use Unified vs Explicit Memory

**Unified memory** is great for:
- Rapid prototyping.
- Irregular access patterns.
- Easier correctness.

**Explicit memory** is best for:
- High-performance production paths.
- Overlapping transfers and compute.
- Controlling where data lives.

---

## 13. Next Steps

- Move from basic kernels to **tiling**, **shared-memory staging**, and **streamed pipelines**.
- Explore **CUDA Graphs** for repeated kernel launch sequences.
- Learn architecture-specific tuning (occupancy, instruction throughput).

---

## 14. Transformer Inference Requirements (CUDA-Centric)

Transformer inference is dominated by a small set of GPU-heavy kernels and strict memory constraints. Understanding these requirements helps you design kernels, layouts, and execution strategies that scale.

### 14.1 Core Compute Kernels
Inference is primarily a sequence of:
- **GEMMs** (Q/K/V projections, output projection, MLP).
- **Softmax + attention** (often fused).
- **LayerNorm / RMSNorm** and pointwise ops.

CUDA implications:
- Most time goes to GEMM-like workloads. Use layouts and strides that allow coalesced access and friendly alignment for Tensor Cores when possible.
- Fuse pointwise ops (bias, activation, residual, layernorm) to reduce memory bandwidth pressure.

### 14.2 Memory Footprint and Bandwidth Pressure
Inference performance is frequently **memory-bandwidth bound**, especially for small batch sizes or long context lengths.

Key memory consumers:
- **Weights** (static, large, reused).
- **Activations** (intermediate, per layer).
- **KV cache** (dominant for long sequences).

KV cache rough size (bytes):
```
2 * num_layers * num_heads * head_dim * sequence_length * batch_size * bytes_per_element
```
The factor `2` is for K and V. For long contexts, KV cache bandwidth and residency often dominate.

### 14.3 Sequence Length, Batch Size, and Shapes
Inference can be:
- **Prefill** (process prompt): high arithmetic intensity, large GEMMs.
- **Decode** (token-by-token): small GEMMs, latency sensitive, often memory bound.

Practical CUDA impacts:
- Decode kernels are launch-overhead sensitive. Prefer fused kernels and CUDA Graphs if the shape is stable.
- Prefill benefits from larger batch sizes and sequence lengths to better utilize the GPU.

### 14.4 Precision and Quantization
Common choices:
- **FP16/BF16** for Tensor Core acceleration.
- **INT8/FP8** for lower memory bandwidth and higher throughput.

CUDA considerations:
- Align data layouts for vectorized loads and Tensor Core friendly shapes (e.g., multiples of 8 or 16 in inner dimensions).
- Quantization reduces bandwidth but may add dequantization overhead. Fuse dequantization into GEMMs when possible.

### 14.5 KV Cache Layout and Access
KV cache access patterns must be coalesced. A typical layout is:
```
[layer][token][head][head_dim]
```
or
```
[layer][head][token][head_dim]
```
Choose a layout so threads in a warp access contiguous `head_dim` values for a given token and head.

### 14.6 Unified vs Explicit Memory for Inference
Unified memory is usually not ideal for high-throughput inference because page migration can introduce latency.

Recommendations:
- Use explicit device memory for weights, KV cache, and activations.
- Use pinned host memory for input/output staging.
- For large models, consider streaming weights or partitioning across GPUs.

### 14.7 Fused QKV + Bias + RoPE Pattern
Q, K, and V projections are often fused into one GEMM and then post-processed. A common pattern is a kernel that applies bias and rotary positional embedding (RoPE) in a single pass over Q/K.

Sketch (structure only):
```cpp
__global__ void qkv_bias_rope(
    const half* qkv, const half* bias,
    half* q_out, half* k_out, half* v_out,
    int heads, int head_dim, int seq_len) {
    int token = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    // Layout: [token][3][head][head_dim]
    int base = ((token * 3 + 0) * heads + h) * head_dim + d;
    half q = qkv[base + 0 * heads * head_dim];
    half k = qkv[base + 1 * heads * head_dim];
    half v = qkv[base + 2 * heads * head_dim];

    q = __hadd(q, bias[(0 * heads + h) * head_dim + d]);
    k = __hadd(k, bias[(1 * heads + h) * head_dim + d]);
    v = __hadd(v, bias[(2 * heads + h) * head_dim + d]);

    // Apply RoPE to q and k in-place (omitted math)
    // rope(q, k, token, d, head_dim);

    q_out[(token * heads + h) * head_dim + d] = q;
    k_out[(token * heads + h) * head_dim + d] = k;
    v_out[(token * heads + h) * head_dim + d] = v;
}
```

CUDA impact:
- Fusing reduces memory traffic and kernel launches.
- Make `head_dim` a multiple of 8 or 16 for vectorized loads.

### 14.8 KV Cache Update Kernel Pattern
During decode, you typically write one token per layer to the KV cache.

Example layout:
```
[layer][token][head][head_dim]
```

Update kernel sketch:
```cpp
__global__ void kv_cache_write(
    const half* k_src, const half* v_src,
    half* k_cache, half* v_cache,
    int layer, int token, int heads, int head_dim, int seq_len) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int src = (h * head_dim + d);
    int dst = ((layer * seq_len + token) * heads + h) * head_dim + d;
    k_cache[dst] = k_src[src];
    v_cache[dst] = v_src[src];
}
```

CUDA impact:
- Arrange threads so `d` is the fastest-changing index for coalescing.
- Keep blocks sized to `head_dim` with `blockDim.x` a multiple of 32.

### 14.9 Attention Tiling Sketch (Shared Memory)
Scaled dot-product attention benefits from tiling K/V into shared memory to reduce global traffic.

Sketch (structure only):
```cpp
// For each query block:
// 1) Load a K tile into shared memory
// 2) Compute Q*K^T for that tile
// 3) Accumulate softmax stats
// 4) Load V tile and accumulate output
```

CUDA impact:
- Use shared memory to reuse K/V across multiple queries.
- Use warp-level primitives for softmax reduction (max and sum).

### 14.10 Decode vs Prefill Kernel Strategy
Practical strategy split:
- Prefill: favor large GEMMs and batch multiple sequences for throughput.
- Decode: favor fused kernels and CUDA Graphs to reduce launch overhead.

If decode shape is stable (same batch size, head counts, head_dim), capture the execution in a CUDA Graph and replay per token.

### 14.11 Fused MLP Kernel Pattern (GEMM + Bias + Activation)
Transformer MLP blocks are typically:
```
Y = activation(X * W1 + b1)
Z = Y * W2 + b2
```

The biggest wins come from fusing **bias + activation** into the output of the first GEMM, and then feeding the result into the second GEMM without materializing intermediate buffers when possible.

High-level pattern:
1. GEMM1: `X * W1` (often large and compute-heavy).
2. Fuse bias + activation in-place (e.g., GELU or SiLU).
3. GEMM2: `Y * W2`.

CUDA implications:
- Use tensor-core friendly dimensions for GEMM1 and GEMM2.
- Fuse bias + activation as an epilogue kernel (or use a library that supports fused epilogues).
- If you must launch a separate kernel, keep it bandwidth-efficient and coalesced.

Sketch (activation-only kernel):
```cpp
__global__ void bias_gelu(half* y, const half* b, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        half v = __hadd(y[i], b[i % stride]);
        // Approx GELU (placeholder)
        // y[i] = gelu(v);
        y[i] = v;
    }
}
```

### 14.12 Attention Softmax Walkthrough (Warp-Level)
Softmax is the numerically sensitive, bandwidth-heavy part of attention. The core steps for each query are:
1. Compute dot products with K.
2. Find max for numerical stability.
3. Subtract max, exponentiate, sum.
4. Normalize and accumulate weighted V.

Warp-level pattern (for a single query row):
```cpp
// Assume one warp handles one query row
float max_val = -INFINITY;
for (int j = lane; j < k_len; j += warpSize) {
    float v = dot(q, k[j]);
    max_val = max(max_val, v);
}
max_val = warpReduceMax(max_val);

float sum = 0.0f;
for (int j = lane; j < k_len; j += warpSize) {
    float v = expf(dot(q, k[j]) - max_val);
    sum += v;
}
sum = warpReduceSum(sum);

for (int j = lane; j < k_len; j += warpSize) {
    float p = expf(dot(q, k[j]) - max_val) / sum;
    // accumulate p * v[j] into output
}
```

CUDA implications:
- Use warp-level reductions for max and sum to avoid block-wide syncs.
- Cache tiles of K/V into shared memory for reuse across multiple queries.
- Use vectorized loads when reading K and V (e.g., `half2` or `float4`).

### 14.13 Fused Attention Kernel Outline (Shared Memory Tiling)
At a high level, fused attention kernels combine QK^T, softmax, and PV into one kernel to reduce intermediate writes.

Sketch (structure only):
```cpp
// Block handles a tile of queries.
// Shared memory stores a tile of K and V.
extern __shared__ half smem[];
half* Ktile = smem;
half* Vtile = smem + K_TILE_ELEMS;

for (int kt = 0; kt < k_len; kt += K_TILE) {
    // 1) Load K/V tiles into shared memory (coalesced)
    // 2) Compute Q*K^T for this tile
    // 3) Update running max and sum for softmax
}

// 4) Normalize and accumulate output using V tiles
```

CUDA implications:
- Keep K/V tiles small enough for shared memory, large enough for reuse.
- Use warp-level reductions for softmax max/sum.
- Avoid shared memory bank conflicts by padding.

### 14.14 KV Cache Quantization and Layout Tradeoffs
Quantizing KV cache reduces bandwidth and footprint but adds dequantization cost.

Common strategies:
- INT8 with per-head or per-group scale/zero-point.
- FP8 or INT4 for extreme memory reduction (more complex).

Tradeoffs:
- Smaller cache improves throughput for long contexts.
- Dequantization can be fused into attention math to minimize overhead.
- Choose layouts that preserve coalescing and alignment for vectorized loads.

Layout tips:
- Keep `head_dim` contiguous as the fastest-changing index.
- Store scale/zero-point in a parallel array aligned to the same tile structure.

### 14.15 Fused Attention Pseudocode (Warp-Level Softmax + Tiling)
The following pseudocode illustrates a fused attention kernel structure that avoids writing intermediate QK scores to global memory. It uses shared memory tiles and warp-level reductions for softmax.

```cpp
// Pseudocode (structure only)
for each block of queries (Q_tile):
    // Initialize running max and sum for softmax
    max[q] = -INF
    sum[q] = 0
    out[q, :] = 0

    for each K/V tile:
        load K_tile, V_tile into shared memory
        __syncthreads()

        // Compute Q * K^T for this tile
        for each query q in Q_tile:
            for each key k in K_tile:
                score = dot(Q[q], K[k])
                // Update running max and sum (online softmax)
                new_max = max(max[q], score)
                sum[q] = sum[q] * exp(max[q] - new_max) + exp(score - new_max)
                max[q] = new_max

        // Accumulate output with V_tile
        for each query q in Q_tile:
            for each key k in K_tile:
                p = exp(score - max[q]) / sum[q]
                out[q, :] += p * V[k, :]

        __syncthreads()
```

CUDA implications:
- Use warp-level reductions for `max` and `sum` to avoid block-wide stalls.
- Keep Q in registers, K/V in shared memory.
- Avoid recomputing `exp(score - max)` more than needed by caching partials in registers where possible.

### 14.16 Shared Memory Layout Notes (Attention Tiling)
When tiling K/V into shared memory, layout determines both coalescing and bank conflicts.

Guidelines:
- Use a layout where the **head_dim** is contiguous for each key/value to enable vectorized loads.
- Pad shared memory tiles if head_dim is a multiple of 32 to avoid bank conflicts.
- Arrange tiles so that threads in a warp read contiguous elements when loading K/V.

Example shared memory layout (conceptual):
```
Ktile[TileK][HeadDim + PAD]
Vtile[TileK][HeadDim + PAD]
```
Use `PAD = 1` when `HeadDim` is a multiple of 32 to reduce conflicts.

---

## 15. Performance Optimization, Measurement, and Profiling

Optimization is only useful when driven by measurement. CUDA kernels can appear correct but be far from optimal without profiling.

### 15.1 Measure with CUDA Events
CUDA events provide low-overhead GPU timing:
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

### 15.2 Host Timing vs GPU Timing
- CPU timers (`std::chrono`) measure **enqueue time** unless you synchronize.
- CUDA events measure **actual device execution time**.

Rule: Use CUDA events for kernel timing, and synchronize when you need host-visible completion.

### 15.3 Profiling Tools
Common workflow:
- **Nsight Systems** for end-to-end timeline (CPU + GPU overlap).
- **Nsight Compute** for kernel-level metrics (occupancy, memory throughput, instruction mix).

Use these to answer:
- Is the kernel compute-bound or memory-bound?
- Are there stalls due to memory latency or low occupancy?
- Are global memory loads coalesced?
- Are shared memory bank conflicts present?

### 15.4 Profiling Command Recipes
Basic Nsight Systems run:
```bash
nsys profile -o profile_report ./your_app
```

Kernel-level profiling with Nsight Compute:
```bash
ncu --set full --target-processes all ./your_app
```

Profile a single kernel by name:
```bash
ncu --kernel-name vecAdd --launch-skip 5 --launch-count 10 ./your_app
```

### 15.5 Build Flags for Profiling
Recommended build flags for profiling and readable traces:
```bash
nvcc -O3 -lineinfo your_app.cu -o your_app
```

Notes:
- `-lineinfo` preserves source mapping with minimal overhead.
- `-G` enables device debug but heavily reduces performance; use only for debugging, not profiling.
- `-Xptxas=-v` prints register usage and shared memory stats at compile time.

### 15.6 Microbenchmark Harness Pattern
Use a warmup phase and repeated timing to reduce variance:
```cpp
// Warmup
for (int i = 0; i < 10; ++i) {
    kernel<<<grid, block>>>(...);
}
cudaDeviceSynchronize();

// Timed
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
for (int i = 0; i < 100; ++i) {
    kernel<<<grid, block>>>(...);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
float avg_us = (ms * 1000.0f) / 100.0f;
```

### 15.7 Optimization Checklist
Focus on the biggest wins first:
- Coalesce global memory access.
- Use shared memory to increase reuse.
- Reduce global memory traffic via fusion.
- Minimize divergent branches within warps.
- Tune block size to balance occupancy and register usage.
- Avoid excessive atomics or serialize them by design.

### 15.8 Validate Optimizations
After every optimization:
- Re-measure kernel runtime.
- Compare output correctness.
- Monitor register usage and occupancy.

### 15.9 Understand Roofline Limits
Speedups are bounded by:
- **Memory bandwidth** for data-heavy kernels.
- **Compute throughput** for math-heavy kernels.

If measured performance is near the hardware ceiling, further micro-optimizations likely produce diminishing returns.

### 15.10 Profiling Playbook (What to Measure and Why)
Use this as a quick triage guide for common kernel types.

**GEMM / MatMul kernels**
- Check: Tensor Core utilization, occupancy, achieved FLOPs.
- Nsight Compute metrics: `sm__throughput.avg.pct_of_peak_sustained_active`, `sm__pipe_tensor_active.avg.pct_of_peak_sustained_active`.
- If low: verify alignment, layout, and that dimensions are multiples of 8/16 where required.

**Attention / Softmax kernels**
- Check: memory throughput vs compute, shared memory usage, bank conflicts.
- Metrics: `dram__throughput.avg.pct_of_peak_sustained_active`, `l1tex__t_bytes.sum`, shared memory conflict counters.
- If slow: reduce global memory traffic via tiling; use warp-level reductions.

**Elementwise / Activation kernels**
- Check: launch overhead and memory bandwidth.
- If small workloads dominate: fuse ops or use CUDA Graphs to reduce launch cost.

**KV cache read/write**
- Check: coalescing and L2 hit rates.
- Metrics: `l2_tex__t_bytes.sum`, `l2_tex__hit_rate`.
- If poor: re-evaluate layout so `head_dim` is contiguous.

**General checklist**
- Are global loads coalesced?
- Is shared memory padded to avoid bank conflicts?
- Is occupancy limited by registers or shared memory?

---

### References
This guide is based on the CUDA Programming Guide (Release 13.1).
