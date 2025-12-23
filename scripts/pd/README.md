## PD Development Guide

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
scripts. You can find them when you run setup

```
Your scripts can be found in ...
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