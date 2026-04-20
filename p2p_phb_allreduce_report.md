# NCCL benchmark report: `5kp_p2p_phb_allreduce_raw_data.txt`

## 1. 测试命令

### AR OFF（1）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP OFF（2）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP ON（3）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

## 2. 测试环境与拓扑

- **硬件拓扑**：两台 **6KD GPU** 服务器，MPI 跨机运行；每节点 **8 GPU**（`mpirun -npernode 8`），合计 **16 GPU / 16 ranks**。
- **集合通信用例**：`all_reduce_perf`（NCCL Tests）。
- **日志中出现的节点名**：R6KD-CX8aaS-GPU-14、R6KD-CX8aaS-GPU-15。
- **说明**：原始日志中 CASE1 标题可能写作 “AN OFF”，本报告按 **AR OFF** 理解与对齐。

## 3. Out-of-place algbw 对比（基准：AR OFF）

下表仅统计 **out-of-place** 列中的 **algbw（GB/s）**。相对 **AR OFF** 的差异百分比为：`(algbw_case - algbw_AR_OFF) / algbw_AR_OFF * 100%`；正值表示相对基准更快，负值表示更慢。

| Message size | AR OFF algbw (GB/s) | AR ON DDP OFF algbw (GB/s) | vs AR OFF | AR ON DDP ON algbw (GB/s) | vs AR OFF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 0.03 | 0.03 | 0% | 0.03 | 0% |
| 2 KiB | 0.06 | 0.06 | 0% | 0.06 | 0% |
| 4 KiB | 0.1 | 0.1 | 0% | 0.1 | 0% |
| 8 KiB | 0.19 | 0.16 | -15.79% | 0.19 | 0% |
| 16 KiB | 0.38 | 0.39 | +2.63% | 0.36 | -5.26% |
| 32 KiB | 0.44 | 0.55 | +25% | 0.55 | +25% |
| 64 KiB | 0.72 | 0.67 | -6.94% | 0.65 | -9.72% |
| 128 KiB | 0.63 | 0.68 | +7.94% | 0.66 | +4.76% |
| 256 KiB | 0.71 | 0.73 | +2.82% | 0.67 | -5.63% |
| 512 KiB | 0.76 | 0.75 | -1.32% | 0.47 | -38.16% |
| 1 MiB | 0.78 | 0.79 | +1.28% | 0.81 | +3.85% |
| 2 MiB | 5.54 | 5.3 | -4.33% | 5.58 | +0.72% |
| 4 MiB | 5.91 | 5.84 | -1.18% | 6.03 | +2.03% |
| 8 MiB | 6.19 | 6.05 | -2.26% | 6.23 | +0.65% |
| 16 MiB | 17.75 | 16.17 | -8.9% | 15.92 | -10.31% |
| 32 MiB | 21.18 | 19.6 | -7.46% | 21.2 | +0.09% |
| 64 MiB | 23.93 | 23.07 | -3.59% | 23.91 | -0.08% |
| 128 MiB | 25.52 | 25.48 | -0.16% | 25.45 | -0.27% |
| 256 MiB | 26.13 | 26.2 | +0.27% | 26.15 | +0.08% |
| 512 MiB | 26.39 | 25.9 | -1.86% | 26.4 | +0.04% |
| 1 GiB | 26.44 | 13.32 | -49.62% | 26.43 | -0.04% |
| 2 GiB | 26.51 | 13.53 | -48.96% | 26.51 | 0% |
| 4 GiB | 26.54 | 8.94 | -66.31% | 26.53 | -0.04% |
| 8 GiB | 26.58 | 8.23 | -69.04% | 26.57 | -0.04% |
