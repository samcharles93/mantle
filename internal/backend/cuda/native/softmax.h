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

#ifdef __cplusplus
}
#endif

#endif
