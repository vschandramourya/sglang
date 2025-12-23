#!/bin/bash
set -euo pipefail

# =========================
# Environment variables
# =========================
export SGL_DS3_LOAD_SHARE_NORM="1"
export SGLANG_ENABLE_SPEC_V2="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP="1"

# =========================
# Launch SGLang server
# =========================
python3 -m sglang.launch_server \
  --model-path /data/dsv31-agent-1003-bt-rl-1026-mtp-2-force-thinkFP4/ \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend sglang \
  --tp 4 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --served-model-name cursor/dsv31-b200-tgl-test-02 \
  --host 0.0.0.0 \
  --port 12345 \
  --enable-metrics \
  --enable-flashinfer-allreduce-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --max-running-requests 32 \
  --chunked-prefill-size 98304 \
  --max-prefill-tokens 98304 \
  --mem-fraction-static 0.75 \
  --log-requests \
  --log-requests-level 0 \
  --enable-inc-tokenizer
