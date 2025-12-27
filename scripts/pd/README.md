## PD Development Guide

> Current PD is caught up with commit d252d7ab

### PD Local Setup

The PD setup looks like the following:

```
+--------------------------------------------+
| Eval -> Tproxy -> Router -> Prefill       |
|                      \-> Decode           |
+--------------------------------------------+
```

> If you are only trying to run sglang bench serving, you don't need tproxy. You only need it when you want to replay a online traffic.

First time running:
Run the following script:
`
setup.sh
`

It should create three containers: router, prefill, decode.

We also change your local bash so you can easily execute into your container

```
dx ywang-prefill
```

You need to modify the router config, as it is always changing. Simply exec into your container and locate your
scripts. Your scripts can be found in $PD_SCRIPTS_DIR. If you want to use a centralized code source to control the behavior of your server. Copy it under `/data/<your_username>/`:
```
export PYTHONPATH=/data/<username>/tgl/python/
```

To start the prefill node:

```
$PD_SCRIPTS_DIR/run_prefill.sh
```

To start the decode node:
```
$PD_SCRIPTS_DIR/run_decode.sh
```

Be careful, you need to modify your script to limit the GPU you use for each instance. The default script is good for local 1P1D.

Network interfaces:

`ifconfig` gives the following information

```
docker0 inet 172.17.0.1 ... # This is docker bridge networks.

ens10f0np0: 10.173.2.69/16 (MTU 9000)

ens9f0np0: 10.174.2.69/16 (MTU 9000)

ens10f1np1: 10.179.33.69/20 (MTU 9000, but much lower RX/TX so far)

ens10f0np0.674: 10.168.9.69/21 (VLAN sub-interface; MTU 9000)
```


> We should use ens10f0np0.674's IP 10.168.9.69 for our LLM communication. Use 10.173.2.69 for accepting request traffic.

### Useful Tips
Use /data/ib-traffic-monitor to monitor the traffic through IB.

To run simulation tool, check out https://github.com/togethercomputer/sim/blob/dev locally and run:

```
python3 sim_tgl.py   --server http://0.0.0.0:12345 --model cursor/dsv31-gb200-tgl-test-02     --system-prompt-len 5000     --initial-qps 0.1     --max-qps 0.8   --max-inflight 32     --ramp-duration 10     --sustain-duration 600     --window 30     --gpus 4     --generation-length-mean 559     --generation-length-median 306     --acc-len 1.0   --mtp-overhead-factor 1.0     --max-prompt-tokens 215000     --poisson  --poisson-shape 2     --new-session-rate 0.05  --session-decay-lambda 0.005     --initial-sessions 7  --random-seed 123  --initial-prefix-mean 44000 --initial-prefix-median 39000 --new-tokens-mean 5500 --new-tokens-median 1400 --dashboard-mode --name cursor-fast-v2-exp-tp4-hicache-tgl-0.5qps
```

Then run the dashboard with

```
python3 dashboard.py --data-dir benchmarks
```

Then portforward with:
```
ssh -L 8050:10.168.9.69:8050  ywang@research-dev-b200-04.cloud.together.ai
```