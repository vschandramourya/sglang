python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://10.173.2.70:12347 \
    --prefill http://10.173.2.69:12347 \
    --decode http://10.173.2.70:22356 \
    --host 10.173.2.70 --port 8000 \
    --prefill-policy cache_aware \
    --pre-prefill-url http://10.173.2.69:12347 \
    --health-check-interval-secs 10 \
    --log-level debug | tee router_log.log
