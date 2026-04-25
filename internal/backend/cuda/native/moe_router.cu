#include <cuda_runtime.h>
#include <float.h>

extern "C" {

// MoE router kernel.
// Inputs:
//   raw[numExperts]        — router logits (post-matvec, in-place)
//   bias[numExperts]       — additive selection bias (may be nullptr if biasLen==0)
//   biasLen                — number of valid bias entries (rest treated as 0)
//   k                      — number of experts to select (k <= 8)
//   routeScale             — final weight scaling factor
//   numExperts             — total experts
// Outputs:
//   raw                    — overwritten with sigmoid(raw)        (no bias added)
//   idx[k]                 — selected expert indices (deterministic, lower-index wins ties)
//   weights[k]             — normalized weights: (raw[idx[j]] / sum_raw_selected) * routeScale
//
// Semantics matches simd/moe.go:
//   raw[i]   = sigmoid(raw[i])
//   sel[i]   = raw[i] + (i < biasLen ? bias[i] : 0)
//   top-k argmax over sel[] with lower-index tiebreak
//   denom    = sum(raw[idx[j]] for j in 0..k-1); if denom==0 -> denom=1
//   w[j]     = raw[idx[j]] / denom * routeScale   (or 0 if idx<0)
//
// Single block. Threads == numExperts (rounded to next pow2, capped at 1024).

__global__ void moe_router_f32_kernel(
    float* __restrict__ raw,
    const float* __restrict__ bias,
    int biasLen,
    int k,
    float routeScale,
    int numExperts,
    int* __restrict__ idxOut,
    float* __restrict__ weightsOut
) {
    extern __shared__ float smem[];
    // Layout:
    //   smem[0 .. numExperts-1]            : sel scores (mutable; -INF marks selected)
    //   smem[numExperts .. 2*numExperts-1] : raw scores (post-sigmoid, immutable)
    //   smem[2*numExperts .. ]             : reduction scratch (val pairs)
    float* sel = smem;
    float* sig = smem + numExperts;
    float* redVal = smem + 2 * numExperts;
    int* redIdx = (int*)(redVal + blockDim.x);

    int tid = threadIdx.x;

    // Phase 1: sigmoid(raw) into both sig[] and raw[]; sel[] = sig + bias
    for (int i = tid; i < numExperts; i += blockDim.x) {
        float v = raw[i];
        float s = 1.0f / (1.0f + __expf(-v));
        sig[i] = s;
        raw[i] = s;
        float b = (bias != nullptr && i < biasLen) ? bias[i] : 0.0f;
        sel[i] = s + b;
    }
    __syncthreads();

    // Phase 2: iterative top-K argmax with lower-index tiebreak.
    for (int j = 0; j < k; ++j) {
        // Each thread scans its strided slice; track local best (max value, lowest index).
        float bestV = -FLT_MAX;
        int   bestI = -1;
        for (int i = tid; i < numExperts; i += blockDim.x) {
            float v = sel[i];
            if (v > bestV || (v == bestV && (bestI < 0 || i < bestI))) {
                bestV = v;
                bestI = i;
            }
        }
        redVal[tid] = bestV;
        redIdx[tid] = bestI;
        __syncthreads();

        // Tree reduction: max value with lower-index tiebreak.
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float vR = redVal[tid + s];
                int   iR = redIdx[tid + s];
                float vL = redVal[tid];
                int   iL = redIdx[tid];
                bool takeR = (vR > vL) || (vR == vL && iR >= 0 && (iL < 0 || iR < iL));
                if (takeR) {
                    redVal[tid] = vR;
                    redIdx[tid] = iR;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            int chosen = redIdx[0];
            idxOut[j] = chosen;
            if (chosen >= 0 && chosen < numExperts) {
                sel[chosen] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    // Phase 3: weight normalization. denom = sum(sig[idx[j]]) over j with idx>=0.
    if (tid == 0) {
        float denom = 0.0f;
        for (int j = 0; j < k; ++j) {
            int id = idxOut[j];
            if (id >= 0 && id < numExperts) {
                denom += sig[id];
            }
        }
        if (denom == 0.0f) denom = 1.0f;
        for (int j = 0; j < k; ++j) {
            int id = idxOut[j];
            if (id < 0 || id >= numExperts) {
                weightsOut[j] = 0.0f;
            } else {
                weightsOut[j] = (sig[id] / denom) * routeScale;
            }
        }
    }
}

static int nextPow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

int mantleCudaMoERouterF32(
    float* raw,
    const float* bias,
    int biasLen,
    int k,
    float routeScale,
    int numExperts,
    int* idxOut,
    float* weightsOut,
    cudaStream_t stream
) {
    if (numExperts <= 0 || k <= 0) {
        return cudaErrorInvalidValue;
    }
    int blockSize = nextPow2(numExperts);
    if (blockSize > 1024) blockSize = 1024;
    if (blockSize < 32) blockSize = 32;
    size_t shBytes = sizeof(float) * (size_t)numExperts * 2
                   + sizeof(float) * (size_t)blockSize
                   + sizeof(int)   * (size_t)blockSize;
    moe_router_f32_kernel<<<1, blockSize, shBytes, stream>>>(
        raw, bias, biasLen, k, routeScale, numExperts, idxOut, weightsOut);
    return cudaGetLastError();
}

} // extern "C"
