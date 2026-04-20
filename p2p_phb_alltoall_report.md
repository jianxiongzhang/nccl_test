# NCCL benchmark report: `5kp_p2p_phb_alltoall_raw_data.txt`

## 1. 测试命令

### AR OFF（1）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP OFF（2）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP ON（3）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
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
| 1 KiB | 0.03 | 0.03 | 0% | 0.03 | 0% |
| 2 KiB | 0.07 | 0.06 | -14.29% | 0.07 | 0% |
| 4 KiB | 0.14 | 0.13 | -7.14% | 0.13 | -7.14% |
| 8 KiB | 0.27 | 0.27 | 0% | 0.22 | -18.52% |
| 16 KiB | 0.52 | 0.51 | -1.92% | 0.5 | -3.85% |
| 32 KiB | 1.01 | 0.96 | -4.95% | 0.96 | -4.95% |
| 64 KiB | 1.88 | 1.88 | 0% | 1.79 | -4.79% |
| 128 KiB | 3.57 | 3.49 | -2.24% | 3.48 | -2.52% |
| 256 KiB | 5.84 | 5.34 | -8.56% | 6.4 | +9.59% |
| 512 KiB | 11.04 | 9.5 | -13.95% | 10.85 | -1.72% |
| 1 MiB | 15.45 | 13.78 | -10.81% | 15.11 | -2.2% |
| 2 MiB | 24.14 | 21.96 | -9.03% | 0.44 | -98.18% |
| 4 MiB | 26.87 | 22.96 | -14.55% | 26.86 | -0.04% |
| 8 MiB | 32.42 | 30.03 | -7.37% | 31.5 | -2.84% |
| 16 MiB | 40.95 | 38.55 | -5.86% | 40.31 | -1.56% |
| 32 MiB | 43.03 | 42.05 | -2.28% | 42.25 | -1.81% |
| 64 MiB | 43.92 | 9.78 | -77.73% | 44.06 | +0.32% |
| 128 MiB | 44.14 | 15.69 | -64.45% | 10.25 | -76.78% |
| 256 MiB | 42.45 | 16.08 | -62.12% | 19.94 | -53.03% |
| 512 MiB | 41.62 | 28.3 | -32% | 19.7 | -52.67% |
| 1 GiB | 37.59 | 36.73 | -2.29% | 29.35 | -21.92% |
| 2 GiB | 43.58 | 38.05 | -12.69% | 39.57 | -9.2% |
| 4 GiB | 43.34 | 40.59 | -6.35% | 40.21 | -7.22% |
| 8 GiB | 43.72 | 42.84 | -2.01% | 44.12 | +0.91% |
