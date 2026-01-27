export HOST_IP=10.173.2.69

export SGLANG_TORCH_PROFILER_DIR=/scratch/workspaces/jiejing/workdir/tgl_pd/yubo_test/profile


export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_12:1,mlx5_13:1
export UCX_TLS=rc,cuda

export UCX_LOG_LEVEL=info
export UCX_LOG_COMPONENT=stats
#export UCX_NET_DEVICES=ens9f0np0,ens10f0np0,ens10f1np1
#export UCX_TLS=tcp,cuda_ipc,rc,cuda_copy,cma

#export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13
#export NCCL_IB_GID_INDEX=0

export IB_NET_HOST=10.168.9.70

export MOONCAKE_MASTER=${IB_NET_HOST}:50051

#export CUDA_VISIBLE_DEVICES=4,5,6,7
export MOONCAKE_TE_META_DATA_SERVER="http://${IB_NET_HOST}:9900/metadata" \
export MOONCAKE_PROTOCOL="rdma"
#export MOONCAKE_DEVICE="mlx5_0"
export MOONCAKE_DEVICE='{"0": "mlx5_0", "1": "mlx5_1", "2": "mlx5_2", "3": "mlx5_3"}'

MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb" \
SGL_DS3_LOAD_SHARE_NORM=1 \
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1 \
python3 -m sglang.launch_server \
  --model-path /data/dsv31-agent-1003-bt-rl-1026-mtp-2-force-thinkFP4/ \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13 \
  --host ${HOST_IP} \
  --port 12347 \
  --nnodes 1 \
  --node-rank 0 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --watchdog-timeout 10000 \
  --tp 4 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --served-model-name cursor/dsv31-gb200-tgl-test-02 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --enable-metrics \
  --enable-flashinfer-allreduce-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --max-running-requests 32 \
  --mem-fraction-static 0.80 \
  --log-requests \
  --log-requests-level 0 \
  --enable-trtllm-mla-fp8-prefill \
  --enable-request-time-stats-logging \
  --enable-hierarchical-cache \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend mooncake \
  --enable-inc-tokenizer | tee pre-prefill_mooncake_log.log 2>&1
