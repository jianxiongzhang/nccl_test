# NCCL benchmark report: `5kp_p2p_disable_plugin_alltoall_raw_data.txt`

## 1. 测试命令

### AR OFF（1）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP OFF（2）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP ON（3）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
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
| 2 KiB | 0.05 | 0.05 | 0% | 0.05 | 0% |
| 4 KiB | 0.09 | 0.09 | 0% | 0.09 | 0% |
| 8 KiB | 0.18 | 0.18 | 0% | 0.18 | 0% |
| 16 KiB | 0.36 | 0.36 | 0% | 0.29 | -19.44% |
| 32 KiB | 0.71 | 0.71 | 0% | 0.71 | 0% |
| 64 KiB | 1.42 | 1.43 | +0.7% | 1.41 | -0.7% |
| 128 KiB | 2.7 | 2.72 | +0.74% | 2.7 | 0% |
| 256 KiB | 5.26 | 4 | -23.95% | 3.97 | -24.52% |
| 512 KiB | 8.47 | 5.73 | -32.35% | 6.84 | -19.24% |
| 1 MiB | 12.7 | 10.59 | -16.61% | 10.71 | -15.67% |
| 2 MiB | 21.35 | 18.12 | -15.13% | 18.03 | -15.55% |
| 4 MiB | 25.63 | 22.96 | -10.42% | 22.83 | -10.92% |
| 8 MiB | 33.47 | 30.42 | -9.11% | 29 | -13.36% |
| 16 MiB | 2.37 | 36.81 | +1453.16% | 36.69 | +1448.1% |
| 32 MiB | 11.4 | 1.02 | -91.05% | 1.93 | -83.07% |
| 64 MiB | 2.45 | 2.28 | -6.94% | 1.4 | -42.86% |
| 128 MiB | 1.13 | 1.91 | +69.03% | 2.08 | +84.07% |
| 256 MiB | 3 | 3.66 | +22% | 3.8 | +26.67% |
| 512 MiB | 5.92 | 6.72 | +13.51% | 5.98 | +1.01% |
| 1 GiB | 10.82 | 11.64 | +7.58% | 11.95 | +10.44% |
| 2 GiB | 22.41 | 20.98 | -6.38% | 22.04 | -1.65% |
| 4 GiB | 31.58 | 34.36 | +8.8% | 34.94 | +10.64% |
| 8 GiB | 42.51 | 43.94 | +3.36% | 43.5 | +2.33% |
