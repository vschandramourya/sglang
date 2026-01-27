#!/bin/bash

PREFILL_HOST="10.168.9.72:12349"

OUTPUT_DIR="/scratch/msrinivasa/sim"

echo "Starting profile on Prefill Worker ($PREFILL_HOST)..."
curl -X POST http://${PREFILL_HOST}/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "'"$OUTPUT_DIR"'",
    "num_steps": 5,
    "start_step": 2,
    "activities": ["GPU", "CPU"]
  }'
echo ""

echo "Profiling initiated on prefill worker."
