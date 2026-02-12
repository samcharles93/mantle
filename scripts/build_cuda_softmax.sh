#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../internal/backend/cuda/native/build"

mkdir -p "${BUILD_DIR}"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "$COMPUTE_CAP" ]; then
	COMPUTE_CAP="86"
fi

echo "Detected compute capability: sm_${COMPUTE_CAP}"
echo "Compiling CUDA kernels..."
nvcc -O3 -lineinfo -arch=sm_${COMPUTE_CAP} -c "${SCRIPT_DIR}/../internal/backend/cuda/native/softmax.cu" -o "${BUILD_DIR}/softmax.o"
nvcc -O3 -lineinfo -arch=sm_${COMPUTE_CAP} -c "${SCRIPT_DIR}/../internal/backend/cuda/native/fused_rmsnorm_matvec.cu" -o "${BUILD_DIR}/fused_rmsnorm_matvec.o"
nvcc -O3 -lineinfo -arch=sm_${COMPUTE_CAP} -c "${SCRIPT_DIR}/../internal/backend/cuda/native/rmsnorm.cu" -o "${BUILD_DIR}/rmsnorm.o"
ar rcs "${BUILD_DIR}/libmantle_cuda_kernels.a" "${BUILD_DIR}/softmax.o" "${BUILD_DIR}/fused_rmsnorm_matvec.o" "${BUILD_DIR}/rmsnorm.o"

echo "CUDA kernels build complete: ${BUILD_DIR}/libmantle_cuda_kernels.a"
