# NCCL benchmark report: `5kp_p2p_disable_alltoall_raw_data.txt`

## 1. 测试命令

### AR OFF（1）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP OFF（2）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP ON（3）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

## 2. 测试环境与拓扑

- **硬件拓扑**：两台 **6KD GPU** 服务器，MPI 跨机运行；每节点 **8 GPU**（`mpirun -npernode 8`），合计 **16 GPU / 16 ranks**。
- **集合通信用例**：`alltoall_perf`（NCCL Tests）。
- **日志中出现的节点名**：R6KD-CX8aaS-GPU-14、R6KD-CX8aaS-GPU-15。
- **说明**：原始日志中 CASE1 标题可能写作 “AN OFF”，本报告按 **AR OFF** 理解与对齐。

## 3. Out-of-place algbw 对比（基准：AR OFF）

下表仅统计 **out-of-place** 列中的 **algbw（GB/s）**。相对 **AR OFF** 的差异百分比为：`(algbw_case - algbw_AR_OFF) / algbw_AR_OFF * 100%`；正值表示相对基准更快，负值表示更慢。

| Message size | AR OFF algbw (GB/s) | AR ON DDP OFF algbw (GB/s) | vs AR OFF | AR ON DDP ON algbw (GB/s) | vs AR OFF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 0.02 | 0.02 | 0% | 0.02 | 0% |
| 2 KiB | 0.04 | 0.04 | 0% | 0.04 | 0% |
| 4 KiB | 0.09 | 0.09 | 0% | 0.09 | 0% |
| 8 KiB | 0.18 | 0.15 | -16.67% | 0.18 | 0% |
| 16 KiB | 0.36 | 0.35 | -2.78% | 0.35 | -2.78% |
| 32 KiB | 0.71 | 0.64 | -9.86% | 0.7 | -1.41% |
| 64 KiB | 1.4 | 1.38 | -1.43% | 1.38 | -1.43% |
| 128 KiB | 2.72 | 2.66 | -2.21% | 2.68 | -1.47% |
| 256 KiB | 4.18 | 4.01 | -4.07% | 5.12 | +22.49% |
| 512 KiB | 8.21 | 6.97 | -15.1% | 6.97 | -15.1% |
| 1 MiB | 12.08 | 9.65 | -20.12% | 11.66 | -3.48% |
| 2 MiB | 18.75 | 16.78 | -10.51% | 18.54 | -1.12% |
| 4 MiB | 26.58 | 23.17 | -12.83% | 24.94 | -6.17% |
| 8 MiB | 33.45 | 1.72 | -94.86% | 1.93 | -94.23% |
| 16 MiB | 40.38 | 37.83 | -6.32% | 39.73 | -1.61% |
| 32 MiB | 45.36 | 43.04 | -5.11% | 45.22 | -0.31% |
| 64 MiB | 47.71 | 10.81 | -77.34% | 34.8 | -27.06% |
| 128 MiB | 48.64 | 19.2 | -60.53% | 10.51 | -78.39% |
| 256 MiB | 49.19 | 15.03 | -69.45% | 17.55 | -64.32% |
| 512 MiB | 49.42 | 25.03 | -49.35% | 19.74 | -60.06% |
| 1 GiB | 49.69 | 40.09 | -19.32% | 30.26 | -39.1% |
| 2 GiB | 49.68 | 39.74 | -20.01% | 35.44 | -28.66% |
| 4 GiB | 49.8 | 48.99 | -1.63% | 46.46 | -6.71% |
| 8 GiB | 49.83 | 49.41 | -0.84% | 49.58 | -0.5% |
