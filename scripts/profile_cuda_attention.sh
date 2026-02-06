#!/usr/bin/env bash
set -e

export MANTLE_CUDA_TRACE=1

echo "Running profiling with CUDA tracing enabled..."
echo "Set MANTLE_CUDA_TRACE=1 to enable detailed timing"
echo ""

cat <<'EOT'
To profile attention timing:

1. Export the environment variable:
   export MANTLE_CUDA_TRACE=1

2. Run your mantle inference command:
   mantle run <your-model> --prompt "test prompt"

3. Timing statistics are collected automatically.

4. Example output will show:
   - Attention score computation time
   - Softmax kernel time
   - F32->F16 conversion time (if KV cache is F16)
   - Value computation time
   - Total attention time
EOT
