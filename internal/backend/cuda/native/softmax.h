#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cudaStream_t;

int mantleCudaSoftmaxRowsF32(float* data, int rows, int cols, cudaStream_t stream);
int mantleCudaQuantMatVecInt8BlocksF32(
    const int8_t* q,
    const float* scales,
    const float* x,
    float* y,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream);
int mantleCudaQuantMatVecQ4F32(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    const float* x,
    float* y,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream);
int mantleCudaQuantMatVecK4F32(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    const float* x,
    float* y,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream);
int mantleCudaDequantizeQ4ToF16(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    uint16_t* outF16,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream);
int mantleCudaDequantizeK4ToF16(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    uint16_t* outF16,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream);
int mantleCudaSiluMulF32(
    const float* gate,
    const float* up,
    float* out,
    int n,
    cudaStream_t stream);
int mantleCudaConvertF32ToF16(
    const float* in,
    uint16_t* out,
    int n,
    cudaStream_t stream);
int mantleCudaAttentionInnerF16CacheF32(
    const float* q,
    const uint16_t* cacheK,
    const uint16_t* cacheV,
    float* out,
    int pos,
    int start,
    int kvStride,
    int headDim,
    int nHead,
    int kvHeads,
    int cacheLen,
    float scale,
    cudaStream_t stream);
int mantleCudaAttentionInnerMixedCacheF32(
    const float* q,
    const uint16_t* cacheKF16,
    const uint16_t* cacheVF16,
    const int8_t* cacheKQ8,
    const int8_t* cacheVQ8,
    const float* cacheKScales,
    const float* cacheVScales,
    float* out,
    int useQ8K,
    int useQ8V,
    int pos,
    int start,
    int kvStride,
    int headDim,
    int nHead,
    int kvHeads,
    int cacheLen,
    float scale,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
