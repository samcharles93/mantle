#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="${SCRIPT_DIR}/../internal/backend/cuda/native"
BUILD_DIR="${NATIVE_DIR}/build"

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

# Host compiler selection: nvcc rejects GCC newer than the maximum its
# host_config.h advertises. When the default g++ is too new, fall back to the
# newest installed supported g++-N. MANTLE_CUDA_HOST_COMPILER overrides this.
HOST_FLAGS=()
if [ -n "${MANTLE_CUDA_HOST_COMPILER:-}" ]; then
	HOST_FLAGS=(-ccbin "${MANTLE_CUDA_HOST_COMPILER}")
elif command -v g++ >/dev/null 2>&1; then
	NVCC_REAL="$(readlink -f "$(command -v nvcc)")"
	HOST_CONFIG="$(dirname "${NVCC_REAL}")/../include/crt/host_config.h"
	if [ ! -f "${HOST_CONFIG}" ]; then
		HOST_CONFIG="/usr/local/cuda/include/crt/host_config.h"
	fi
	MAX_GCC_MAJOR="$(
		grep -oE 'gcc versions later than [0-9]+' "${HOST_CONFIG}" 2>/dev/null \
			| grep -oE '[0-9]+' \
			| head -1
	)"
	DEFAULT_GXX_MAJOR="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
	if [ -n "${MAX_GCC_MAJOR}" ] && [ -n "${DEFAULT_GXX_MAJOR}" ] \
		&& [ "${DEFAULT_GXX_MAJOR}" -gt "${MAX_GCC_MAJOR}" ]; then
		SELECTED_GXX=""
		for ((v = MAX_GCC_MAJOR; v >= 10; v--)); do
			if command -v "g++-${v}" >/dev/null 2>&1; then
				SELECTED_GXX="$(command -v "g++-${v}")"
				break
			fi
		done
		if [ -z "${SELECTED_GXX}" ]; then
			echo "error: nvcc supports host GCC <= ${MAX_GCC_MAJOR}, but default g++ is major ${DEFAULT_GXX_MAJOR} and no g++-N fallback (10..${MAX_GCC_MAJOR}) is installed." >&2
			echo "Install a supported g++ or set MANTLE_CUDA_HOST_COMPILER=/path/to/g++." >&2
			exit 1
		fi
		echo "Default g++ major (${DEFAULT_GXX_MAJOR}) exceeds nvcc's supported maximum (${MAX_GCC_MAJOR}); using ${SELECTED_GXX}." >&2
		HOST_FLAGS=(-ccbin "${SELECTED_GXX}")
	fi
fi

KERNELS=(
	softmax
	fused_rmsnorm_matvec
	rmsnorm
	add_vectors
	shortconv
	round_bf16
	scale_round_bf16
	attn_fused
	mamba_depthwise_conv
	mamba_activation
	mamba_ssm_scan
	mamba_dt
	rmsnorm_gated
	deltanet_l2norm
	deltanet_recurrent
	moe_router
	moe_accumulate
)

echo "Detected compute capability: sm_${DETECTED_CAP}"
if [ "${TARGET_CAP}" != "${DETECTED_CAP}" ]; then
	echo "nvcc does not support sm_${DETECTED_CAP}; using sm_${TARGET_CAP} (+PTX forward-compat)." >&2
fi
echo "Compiling CUDA kernels with: ${ARCH_FLAGS[*]} ${HOST_FLAGS[*]:-}"

OBJECTS=()
for k in "${KERNELS[@]}"; do
	nvcc -O3 -lineinfo "${HOST_FLAGS[@]}" "${ARCH_FLAGS[@]}" -c "${NATIVE_DIR}/${k}.cu" -o "${BUILD_DIR}/${k}.o"
	OBJECTS+=("${BUILD_DIR}/${k}.o")
done

ar rcs "${BUILD_DIR}/libmantle_cuda_kernels.a" "${OBJECTS[@]}"

echo "CUDA kernels build complete: ${BUILD_DIR}/libmantle_cuda_kernels.a"
