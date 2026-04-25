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
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/shortconv.cu" -o "${BUILD_DIR}/shortconv.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/round_bf16.cu" -o "${BUILD_DIR}/round_bf16.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/scale_round_bf16.cu" -o "${BUILD_DIR}/scale_round_bf16.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/attn_fused.cu" -o "${BUILD_DIR}/attn_fused.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/mamba_depthwise_conv.cu" -o "${BUILD_DIR}/mamba_depthwise_conv.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/mamba_activation.cu" -o "${BUILD_DIR}/mamba_activation.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/mamba_ssm_scan.cu" -o "${BUILD_DIR}/mamba_ssm_scan.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/mamba_dt.cu" -o "${BUILD_DIR}/mamba_dt.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/rmsnorm_gated.cu" -o "${BUILD_DIR}/rmsnorm_gated.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/deltanet_l2norm.cu" -o "${BUILD_DIR}/deltanet_l2norm.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/deltanet_recurrent.cu" -o "${BUILD_DIR}/deltanet_recurrent.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/moe_router.cu" -o "${BUILD_DIR}/moe_router.o"
nvcc -O3 -lineinfo "${ARCH_FLAGS[@]}" -c "${SCRIPT_DIR}/../internal/backend/cuda/native/moe_accumulate.cu" -o "${BUILD_DIR}/moe_accumulate.o"
ar rcs "${BUILD_DIR}/libmantle_cuda_kernels.a" "${BUILD_DIR}/softmax.o" "${BUILD_DIR}/fused_rmsnorm_matvec.o" "${BUILD_DIR}/rmsnorm.o" "${BUILD_DIR}/add_vectors.o" "${BUILD_DIR}/shortconv.o" "${BUILD_DIR}/round_bf16.o" "${BUILD_DIR}/scale_round_bf16.o" "${BUILD_DIR}/attn_fused.o" "${BUILD_DIR}/mamba_depthwise_conv.o" "${BUILD_DIR}/mamba_activation.o" "${BUILD_DIR}/mamba_ssm_scan.o" "${BUILD_DIR}/mamba_dt.o" "${BUILD_DIR}/rmsnorm_gated.o" "${BUILD_DIR}/deltanet_l2norm.o" "${BUILD_DIR}/deltanet_recurrent.o" "${BUILD_DIR}/moe_router.o" "${BUILD_DIR}/moe_accumulate.o"

echo "CUDA kernels build complete: ${BUILD_DIR}/libmantle_cuda_kernels.a"
