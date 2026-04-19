# 6KD NCCL `alltoall_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_LEVEL=PHB`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `alltoall_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置；本组数据在 **`NCCL_P2P_LEVEL=PHB`** 下采集。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `alltoall_perf` 参数保持一致；均设置 **`-x NCCL_P2P_LEVEL=PHB`**。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `alltoall_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| P2P | **`NCCL_P2P_LEVEL=PHB`**（在 PHB 拓扑范围内允许 P2P） |
| 其他 | `NCCL_SHM_DISABLE=1`；UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备 |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.03 | 0.03 | 0.00% | 0.03 | 0.00% |
| 2048 | 0.06 | 0.07 | +16.67% | 0.06 | 0.00% |
| 4096 | 0.12 | 0.13 | +8.33% | 0.13 | +8.33% |
| 8192 | 0.25 | 0.26 | +4.00% | 0.25 | 0.00% |
| 16384 | 0.49 | 0.42 | -14.29% | 0.49 | 0.00% |
| 32768 | 0.92 | 0.95 | +3.26% | 0.94 | +2.17% |
| 65536 | 1.73 | 1.83 | +5.78% | 1.79 | +3.47% |
| 131072 | 3.27 | 3.26 | -0.31% | 3.38 | +3.36% |
| 262144 | 5.52 | 5.36 | -2.90% | 6.24 | +13.04% |
| 524288 | 9.39 | 8.90 | -5.22% | 9.56 | +1.81% |
| 1048576 | 13.71 | 12.42 | -9.41% | 13.55 | -1.17% |
| 2097152 | 19.88 | 19.94 | +0.30% | 22.61 | +13.73% |
| 4194304 | 24.54 | 21.90 | -10.76% | 26.28 | +7.09% |
| 8388608 | 29.84 | 29.33 | -1.71% | 27.96 | -6.30% |
| 16777216 | 38.94 | 34.27 | -12.00% | 39.28 | +0.87% |
| 33554432 | 42.32 | 40.22 | -4.96% | 42.31 | -0.02% |
| 67108864 | 43.12 | 42.02 | -2.55% | 43.22 | +0.23% |
| 134217728 | 43.63 | 43.20 | -0.99% | 43.86 | +0.53% |
| 268435456 | 43.72 | 43.77 | +0.11% | 43.39 | -0.75% |
| 536870912 | 43.93 | 43.81 | -0.27% | 43.48 | -1.02% |
| 1073741824 | 43.88 | 44.24 | +0.82% | 43.10 | -1.78% |
| 2147483648 | 43.66 | 44.37 | +1.62% | 43.34 | -0.73% |
| 4294967296 | 43.78 | 44.47 | +1.58% | 43.20 | -1.32% |
| 8589934592 | 43.76 | 44.62 | +1.97% | 43.14 | -1.42% |

**简要观察（out-of-place algbw）**

- **中小消息（约 16KB–2MB）**：相对 AR OFF，**AR ON DDP OFF** 在 **16KB、512KB–1MB、4M、16M** 等步长上出现 **负向** 偏差；**AR ON DDP ON** 在 **256KB、2M、4M** 等步长上 **明显高于** 基准（例如 **2M** 约 **+13.7%**），但在 **8M** 步长相对基准 **偏低**（约 **-6.3%**）。
- **大消息（约 32MB 及以上）**：三种配置整体落在 **~43–44.6 GB/s** 区间；**AR ON DDP OFF** 在 **≥1GB** 消息上相对基准多为 **小幅正向**（约 **+0.8%–+2.0%**）；**AR ON DDP ON** 在 **≥512MB** 的多个步长上相对基准略 **低**（约 **-0.7% 至 -1.8%**）。

---

## 4. 原始数据附录

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:04:20+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid    819 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid    820 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid    821 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid    822 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid    823 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid    824 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid    825 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid    826 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid    845 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid    846 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid    847 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid    848 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid    849 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid    850 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid    851 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid    852 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:00814] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:00814] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    37.69    0.03    0.03       0    32.02    0.03    0.03    N/A
        2048            32     float    none      -1    33.34    0.06    0.06       0    33.59    0.06    0.06    N/A
        4096            64     float    none      -1    35.05    0.12    0.11       0    34.47    0.12    0.11    N/A
        8192           128     float    none      -1    32.90    0.25    0.23       0    32.28    0.25    0.24    N/A
       16384           256     float    none      -1    33.50    0.49    0.46       0    32.64    0.50    0.47    N/A
       32768           512     float    none      -1    35.64    0.92    0.86       0    36.84    0.89    0.83    N/A
       65536          1024     float    none      -1    37.83    1.73    1.62       0    38.71    1.69    1.59    N/A
      131072          2048     float    none      -1    40.09    3.27    3.07       0    40.00    3.28    3.07    N/A
      262144          4096     float    none      -1    47.47    5.52    5.18       0    55.19    4.75    4.45    N/A
      524288          8192     float    none      -1    55.85    9.39    8.80       0    58.78    8.92    8.36    N/A
     1048576         16384     float    none      -1    76.50   13.71   12.85       0    75.94   13.81   12.95    N/A
     2097152         32768     float    none      -1   105.48   19.88   18.64       0    95.09   22.06   20.68    N/A
     4194304         65536     float    none      -1   170.93   24.54   23.01       0   162.46   25.82   24.20    N/A
     8388608        131072     float    none      -1   281.08   29.84   27.98       0   262.46   31.96   29.96    N/A
    16777216        262144     float    none      -1   430.88   38.94   36.50       0   436.38   38.45   36.04    N/A
    33554432        524288     float    none      -1   792.84   42.32   39.68       0   817.96   41.02   38.46    N/A
    67108864       1048576     float    none      -1  1556.34   43.12   40.42       0  1575.48   42.60   39.93    N/A
   134217728       2097152     float    none      -1  3076.04   43.63   40.91       0  3080.79   43.57   40.84    N/A
   268435456       4194304     float    none      -1  6140.36   43.72   40.98       0  6127.41   43.81   41.07    N/A
   536870912       8388608     float    none      -1  12220.9   43.93   41.19       0  12277.4   43.73   41.00    N/A
  1073741824      16777216     float    none      -1  24470.9   43.88   41.14       0  24522.5   43.79   41.05    N/A
  2147483648      33554432     float    none      -1  49186.9   43.66   40.93       0  49094.7   43.74   41.01    N/A
  4294967296      67108864     float    none      -1  98104.6   43.78   41.04       0  98336.5   43.68   40.95    N/A
  8589934592     134217728     float    none      -1   196284   43.76   41.03       0   196614   43.69   40.96    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.1462 
#
# Collective test concluded: alltoall_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:05:27+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   1153 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   1154 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   1155 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   1156 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   1157 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   1158 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   1159 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   1160 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   1255 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   1256 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   1257 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   1258 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   1259 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   1260 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   1261 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   1262 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:01148] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:01148] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    35.30    0.03    0.03       0    31.55    0.03    0.03    N/A
        2048            32     float    none      -1    31.47    0.07    0.06       0    31.37    0.07    0.06    N/A
        4096            64     float    none      -1    32.09    0.13    0.12       0    31.61    0.13    0.12    N/A
        8192           128     float    none      -1    31.81    0.26    0.24       0    31.65    0.26    0.24    N/A
       16384           256     float    none      -1    39.14    0.42    0.39       0    34.25    0.48    0.45    N/A
       32768           512     float    none      -1    34.36    0.95    0.89       0    34.24    0.96    0.90    N/A
       65536          1024     float    none      -1    35.90    1.83    1.71       0    36.34    1.80    1.69    N/A
      131072          2048     float    none      -1    40.15    3.26    3.06       0    39.01    3.36    3.15    N/A
      262144          4096     float    none      -1    48.92    5.36    5.02       0    61.06    4.29    4.02    N/A
      524288          8192     float    none      -1    58.88    8.90    8.35       0    63.65    8.24    7.72    N/A
     1048576         16384     float    none      -1    84.45   12.42   11.64       0    80.92   12.96   12.15    N/A
     2097152         32768     float    none      -1   105.15   19.94   18.70       0   105.91   19.80   18.56    N/A
     4194304         65536     float    none      -1   191.55   21.90   20.53       0   184.45   22.74   21.32    N/A
     8388608        131072     float    none      -1   286.00   29.33   27.50       0   290.41   28.89   27.08    N/A
    16777216        262144     float    none      -1   489.60   34.27   32.13       0   452.61   37.07   34.75    N/A
    33554432        524288     float    none      -1   834.23   40.22   37.71       0   844.56   39.73   37.25    N/A
    67108864       1048576     float    none      -1  1597.03   42.02   39.39       0  1596.54   42.03   39.41    N/A
   134217728       2097152     float    none      -1  3107.23   43.20   40.50       0  3124.62   42.95   40.27    N/A
   268435456       4194304     float    none      -1  6133.39   43.77   41.03       0  6110.96   43.93   41.18    N/A
   536870912       8388608     float    none      -1  12254.8   43.81   41.07       0  12207.5   43.98   41.23    N/A
  1073741824      16777216     float    none      -1  24269.9   44.24   41.48       0  24258.9   44.26   41.50    N/A
  2147483648      33554432     float    none      -1  48402.9   44.37   41.59       0  48359.2   44.41   41.63    N/A
  4294967296      67108864     float    none      -1  96590.7   44.47   41.69       0  96679.6   44.42   41.65    N/A
  8589934592     134217728     float    none      -1   192496   44.62   41.83       0   193297   44.44   41.66    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.7226 
#
# Collective test concluded: alltoall_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:06:35+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   1487 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   1488 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   1489 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   1490 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   1491 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   1492 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   1493 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   1494 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   1665 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   1666 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   1667 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   1668 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   1669 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   1670 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   1671 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   1672 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:01482] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:01482] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    37.68    0.03    0.03       0    32.06    0.03    0.03    N/A
        2048            32     float    none      -1    31.95    0.06    0.06       0    32.06    0.06    0.06    N/A
        4096            64     float    none      -1    32.38    0.13    0.12       0    32.15    0.13    0.12    N/A
        8192           128     float    none      -1    32.65    0.25    0.24       0    31.84    0.26    0.24    N/A
       16384           256     float    none      -1    33.46    0.49    0.46       0    32.51    0.50    0.47    N/A
       32768           512     float    none      -1    34.94    0.94    0.88       0    34.83    0.94    0.88    N/A
       65536          1024     float    none      -1    36.60    1.79    1.68       0    38.16    1.72    1.61    N/A
      131072          2048     float    none      -1    38.74    3.38    3.17       0    39.44    3.32    3.12    N/A
      262144          4096     float    none      -1    42.01    6.24    5.85       0    51.50    5.09    4.77    N/A
      524288          8192     float    none      -1    54.81    9.56    8.97       0    54.16    9.68    9.08    N/A
     1048576         16384     float    none      -1    77.39   13.55   12.70       0    80.25   13.07   12.25    N/A
     2097152         32768     float    none      -1    92.74   22.61   21.20       0    95.24   22.02   20.64    N/A
     4194304         65536     float    none      -1   159.60   26.28   24.64       0   157.81   26.58   24.92    N/A
     8388608        131072     float    none      -1   299.99   27.96   26.22       0   299.33   28.02   26.27    N/A
    16777216        262144     float    none      -1   427.17   39.28   36.82       0   433.26   38.72   36.30    N/A
    33554432        524288     float    none      -1   793.02   42.31   39.67       0   803.66   41.75   39.14    N/A
    67108864       1048576     float    none      -1  1552.55   43.22   40.52       0  1567.27   42.82   40.14    N/A
   134217728       2097152     float    none      -1  3060.20   43.86   41.12       0  3050.73   44.00   41.25    N/A
   268435456       4194304     float    none      -1  6185.87   43.39   40.68       0  6111.66   43.92   41.18    N/A
   536870912       8388608     float    none      -1  12348.4   43.48   40.76       0  12244.1   43.85   41.11    N/A
  1073741824      16777216     float    none      -1  24911.6   43.10   40.41       0  24518.1   43.79   41.06    N/A
  2147483648      33554432     float    none      -1  49548.0   43.34   40.63       0  49136.2   43.70   40.97    N/A
  4294967296      67108864     float    none      -1  99422.5   43.20   40.50       0  98234.3   43.72   40.99    N/A
  8589934592     134217728     float    none      -1   199105   43.14   40.45       0   196463   43.72   40.99    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.153 
#
# Collective test concluded: alltoall_perf
#
```
