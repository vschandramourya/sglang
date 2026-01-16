#!/usr/bin/env bash
set -euo pipefail
# NIXL reads the environment variables.
# source /data/setup-nixl.sh

export HOST_IP=$(ifconfig ens10f0np0.674 | awk '/inet / {print $2}')
export SGL_DS3_LOAD_SHARE_NORM=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0

python3 -m sglang.launch_server \
  --model-path "/data/dsv31-agent-1003-bt-rl-1026-mtp-2-force-thinkFP4/" \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13" \
  --host "${HOST_IP}" \
  --port 12347 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --watchdog-timeout 10000 \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --served-model-name "cursor/dsv31-gb200-tgl-test-02" \
  --enable-metrics \
  --enable-cache-report \
  --enable-flashinfer-allreduce-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --max-running-requests 32 \
  --log-requests \
  --log-requests-level 0 \
  --chunked-prefill-size 131072 \
  --mem-fraction-static 0.75 \
  --enable-inc-tokenizer \
  --enable-trtllm-mla-fp8-prefill \
  --enable-mla-k-transform-kernel \
  --hicache-ratio 3.0 \
  --enable-hierarchical-cache \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --schedule-policy lpm
