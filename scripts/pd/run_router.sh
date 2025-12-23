#!/usr/bin/env bash
set -euo pipefail

export HOST_IP=$(ifconfig ens10f0np0 | awk '/inet / {print $2}')

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.168.9.69:12347 \
  --prefill http://10.168.9.69:12346  \
  --decode http://10.168.9.66:12346 \
  --host ${HOST_IP} \
  --port 8000 \
  --prefill-policy cache_aware

  