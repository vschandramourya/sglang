#etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 &

#sleep 2

#python -m mooncake.http_metadata_server --host 0.0.0.0 --port 9900 &

#mooncake_master -enable_http_metadata_server -http_metadata_server_host 10.173.2.70 -http_metadata_server_port 9900 -eviction_high_watermark_ratio=0.80
#mooncake_master    -enable_http_metadata_server -http_metadata_server_host 0.0.0.0 -http_metadata_server_port 9900 -eviction_high_watermark_ratio=0.80 -enable_ha -etcd_endpoints http://0.0.0.0:2379
#mooncake_master    -enable_http_metadata_server -rpc_listen_addr 10.168.9.70:50051  -http_metadata_server_host 0.0.0.0 -http_metadata_server_port 9900 -eviction_high_watermark_ratio=0.80 -enable_ha -etcd_endpoints http://0.0.0.0:2379  &

mooncake_master \
  -rpc_address 10.168.9.70 \
  -rpc_port 50051 \
  -enable_http_metadata_server \
  -http_metadata_server_host 10.168.9.70 \
  -http_metadata_server_port 9900 \
  -eviction_high_watermark_ratio=0.80

sleep 10

#python -m mooncake.http_metadata_server --host 0.0.0.0 --port 9900 &
#sleep 1
#mooncake_master     -eviction_high_watermark_ratio=0.80


#mooncake_worker \
#  --transport rdma \
#  --bind 0.0.0.0:60051 \
#  --master_addr localhost:50051 \
#  --http_metadata_server_port 9900 \
#  --memory_size $((64 * 1024 * 1024 * 1024))


#mooncake_master \
#  -enable_http_metadata_server \
#  -http_metadata_server_host 10.168.9.70 \
#  -http_metadata_server_port 9900 \
#  -eviction_high_watermark_ratio=0.80


MOONCAKE_LOCAL_HOSTNAME="10.168.9.70" \
MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE" \
MOONCAKE_MASTER="10.168.9.70:50051" \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="100gb" \
MOONCAKE_LOCAL_BUFFER_SIZE=0 \
python -m mooncake.mooncake_store_service --port=8081 &
