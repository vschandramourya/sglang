#!/bin/bash

PREFILL_HOST="0.0.0.0:12345"

OUTPUT_DIR="/home/msrinivasa/tgl/scripts/pd/sim/new"

echo "Starting profile on Prefill Worker ($PREFILL_HOST)..."
curl -X POST http://${PREFILL_HOST}/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "'"$OUTPUT_DIR"'",
    "num_steps": 12,
    "start_step": 2,
    "activities": ["GPU", "CPU"]
  }'
echo ""

echo "Profiling initiated on prefill worker."
