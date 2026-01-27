#!/usr/bin/env bash
set -euo pipefail

export HOST_IP=$(ifconfig ens10f0np0 | awk '/inet / {print $2}')

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.168.9.72:12349 \
  --decode http://10.168.9.72:12350 \
  --host ${HOST_IP} \
  --port 8091 \
  --prefill-policy cache_aware
