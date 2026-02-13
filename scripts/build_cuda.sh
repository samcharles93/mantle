#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../internal/backend/cuda/native/build"

mkdir -p "${BUILD_DIR}"

DETECTED_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
if [ -z "${DETECTED_CAP}" ]; then
	DETECTED_CAP="86"
fi

if [ -n "${MANTLE_CUDA_ARCH:-}" ]; then
	OVERRIDE_CAP="${MANTLE_CUDA_ARCH#sm_}"
	OVERRIDE_CAP="${OVERRIDE_CAP//[^0-9]/}"
	if [ -n "${OVERRIDE_CAP}" ]; then
		DETECTED_CAP="${OVERRIDE_CAP}"
	fi
fi

SUPPORTED_CAPS="$(
	nvcc --help 2>/dev/null \
		| grep -oE 'sm_[0-9]+' \
		| sed 's/sm_//' \
		| sort -nu \
		| tr '\n' ' '
)"

if [ -z "${SUPPORTED_CAPS}" ]; then
	echo "Failed to detect supported SM targets from nvcc; defaulting to sm_${DETECTED_CAP}." >&2
	TARGET_CAP="${DETECTED_CAP}"
else
	TARGET_CAP=""
	for cap in ${SUPPORTED_CAPS}; do
		if [ "${cap}" -le "${DETECTED_CAP}" ]; then
			TARGET_CAP="${cap}"
		fi
	done
	if [ -z "${TARGET_CAP}" ]; then
		TARGET_CAP="$(printf '%s\n' ${SUPPORTED_CAPS} | tail -n1)"
	fi
fi

ARCH_FLAGS=(-gencode "arch=compute_${TARGET_CAP},code=sm_${TARGET_CAP}")
if [ "${TARGET_CAP}" -lt "${DETECTED_CAP}" ]; then
	# Include PTX for forward JIT on newer GPUs when nvcc lacks native SASS support.
	ARCH_FLAGS+=(-gencode "arch=compute_${TARGET_CAP},code=compute_${TARGET_CAP}")
fi

echo "Detected compute capability: sm_${DETECTED_CAP}"
if [ "${TARGET_CAP}" != "${DETECTED_CAP}" ]; then
	echo "nvcc does not support sm_${DETECTED_CAP}; using sm_${TARGET_CAP} (+PTX forward-compat)." >&2
fi
echo "Compiling CUDA kernels with: ${ARCH_FLAGS[*]}"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/softmax.cu" -o "${BUILD_DIR}/softmax.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/fused_rmsnorm_matvec.cu" -o "${BUILD_DIR}/fused_rmsnorm_matvec.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/rmsnorm.cu" -o "${BUILD_DIR}/rmsnorm.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/add_vectors.cu" -o "${BUILD_DIR}/add_vectors.o"
ar rcs "${BUILD_DIR}/libmantle_cuda_kernels.a" "${BUILD_DIR}/softmax.o" "${BUILD_DIR}/fused_rmsnorm_matvec.o" "${BUILD_DIR}/rmsnorm.o" "${BUILD_DIR}/add_vectors.o"

echo "CUDA kernels build complete: ${BUILD_DIR}/libmantle_cuda_kernels.a"
