# NCCL benchmark report: `5kp_p2p_disable_plugin_allreduce_raw_data.txt`

## 1. 测试命令

### AR OFF（1）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP OFF（2）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### AR ON DDP ON（3）
```
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
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
| 1 KiB | 0.02 | 0.02 | 0% | 0.02 | 0% |
| 2 KiB | 0.04 | 0.04 | 0% | 0.04 | 0% |
| 4 KiB | 0.07 | 0.02 | -71.43% | 0.06 | -14.29% |
| 8 KiB | 0.12 | 0.12 | 0% | 0.12 | 0% |
| 16 KiB | 0.22 | 0.22 | 0% | 0.22 | 0% |
| 32 KiB | 0.41 | 0.39 | -4.88% | 0.4 | -2.44% |
| 64 KiB | 0.65 | 0.63 | -3.08% | 0.64 | -1.54% |
| 128 KiB | 1.44 | 1.4 | -2.78% | 1.38 | -4.17% |
| 256 KiB | 2.6 | 2.54 | -2.31% | 2.53 | -2.69% |
| 512 KiB | 4.11 | 4.06 | -1.22% | 4.07 | -0.97% |
| 1 MiB | 5.3 | 5.5 | +3.77% | 5.45 | +2.83% |
| 2 MiB | 6.74 | 6.66 | -1.19% | 6.44 | -4.45% |
| 4 MiB | 7.55 | 7.61 | +0.79% | 7.34 | -2.78% |
| 8 MiB | 18.46 | 18.31 | -0.81% | 14.59 | -20.96% |
| 16 MiB | 19.8 | 19.78 | -0.1% | 17.21 | -13.08% |
| 32 MiB | 19.56 | 19.47 | -0.46% | 18.93 | -3.22% |
| 64 MiB | 20.28 | 20 | -1.38% | 20.09 | -0.94% |
| 128 MiB | 25.47 | 23.28 | -8.6% | 24.05 | -5.58% |
| 256 MiB | 25.46 | 24.47 | -3.89% | 24.34 | -4.4% |
| 512 MiB | 25.49 | 25.29 | -0.78% | 25.28 | -0.82% |
| 1 GiB | 25.65 | 25.37 | -1.09% | 25.29 | -1.4% |
| 2 GiB | 25.68 | 25.33 | -1.36% | 25.4 | -1.09% |
| 4 GiB | 25.75 | 25.39 | -1.4% | 25.39 | -1.4% |
| 8 GiB | 25.85 | 25.42 | -1.66% | 25.42 | -1.66% |
