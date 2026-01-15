#!/usr/bin/env bash
set -euo pipefail
# NIXL reads the environment variables.
source /data/setup-nixl.sh

export HOST_IP=$(ifconfig ens10f0np0.674 | awk '/inet / {print $2}')
export SGL_DS3_LOAD_SHARE_NORM=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_FLASHINFER_FP4_GEMM_BACKEND=cutlass
export SGLANG_ENABLE_JIT_DEEPGEMM='false'

python3 -m sglang.launch_server \
  --model-path "/data/dsv31-agent-1003-bt-rl-1026-mtp-2-force-thinkFP4/" \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13 \
  --host "${HOST_IP}" \
  --port 12347 \
  --quantization modelopt_fp4 \
  --watchdog-timeout 10000 \
  --tp 4 \
  --attention-backend trtllm_mla \
  --chunked-prefill-size -1 \
  --enable-symm-mem \
  --moe-dense-tp-size 1 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --served-model-name "cursor/dsv31-gb200-tgl-test-02" \
  --enable-metrics \
  --enable-cache-report \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --max-running-requests 32 \
  --log-requests \
  --log-requests-level 0 \
  --pp-size 2 \
  --mem-fraction-static 0.86 \
  --disable-flashinfer-autotune \
  --enable-inc-tokenizer
