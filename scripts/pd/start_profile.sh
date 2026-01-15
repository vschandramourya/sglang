#!/bin/bash

HOST=${1:-"10.168.9.69:12345"}

curl -X POST http://${HOST}/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/scratch/ywang/sim",
    "num_steps": 5,
    "start_step": 2,
    "activities": ["GPU", "CPU"]
  }'
