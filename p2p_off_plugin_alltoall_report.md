# 6KD NCCL `alltoall_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_DISABLE=1`，`NCCL_NET_PLUGIN=spcx`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `alltoall_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置；网络侧启用 **Spectrum-X NCCL 插件**（**`NCCL_NET_PLUGIN=spcx`**），并显式将 Spectrum-X 插件库加入 **`LD_LIBRARY_PATH`**。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

**数据说明**：CASE1（AR OFF）在 **32MB 及以上**多个步长的 **out-of-place `algbw` 已显著塌陷到约 0.4–7.7 GB/s**（同时 **16MB** 的 **in-place** 时间出现异常放大），因此表格在该区间的百分比会呈现极端数值；解读时建议结合 **原始日志时间列**与是否复现实验。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `alltoall_perf` 参数保持一致；均包含 **`NCCL_NET_PLUGIN=spcx`**、**`NCCL_P2P_DISABLE=1`** 以及指向 HPC-X 包内 **`nccl_spectrum-x_plugin/lib`** 的 **`LD_LIBRARY_PATH`** 前缀。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `alltoall_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**；`LD_LIBRARY_PATH` 包含 **`.../hpcx-v2.26-.../nccl_spectrum-x_plugin/lib`** |
| 其他 | **`NCCL_P2P_DISABLE=1`**；`NCCL_SHM_DISABLE=1`；UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备 |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.02 | 0.02 | 0.00% | 0.01 | -50.00% |
| 2048 | 0.04 | 0.04 | 0.00% | 0.04 | 0.00% |
| 4096 | 0.07 | 0.09 | +28.57% | 0.08 | +14.29% |
| 8192 | 0.17 | 0.17 | 0.00% | 0.18 | +5.88% |
| 16384 | 0.34 | 0.35 | +2.94% | 0.26 | -23.53% |
| 32768 | 0.69 | 0.69 | 0.00% | 0.69 | 0.00% |
| 65536 | 1.32 | 1.37 | +3.79% | 1.36 | +3.03% |
| 131072 | 2.56 | 2.63 | +2.73% | 2.59 | +1.17% |
| 262144 | 5.01 | 3.79 | -24.35% | 3.84 | -23.35% |
| 524288 | 8.11 | 6.47 | -20.22% | 6.57 | -19.00% |
| 1048576 | 11.32 | 10.42 | -7.95% | 10.48 | -7.42% |
| 2097152 | 18.38 | 17.13 | -6.80% | 17.09 | -7.02% |
| 4194304 | 23.39 | 20.98 | -10.30% | 21.43 | -8.38% |
| 8388608 | 31.91 | 26.47 | -17.05% | 26.58 | -16.70% |
| 16777216 | 36.54 | 0.19 | -99.48% | 0.19 | -99.48% |
| 33554432 | 0.96 | 0.76 | -20.83% | 0.42 | -56.25% |
| 67108864 | 1.26 | 0.89 | -29.37% | 0.70 | -44.44% |
| 134217728 | 1.33 | 1.38 | +3.76% | 1.12 | -15.79% |
| 268435456 | 2.60 | 2.78 | +6.92% | 2.51 | -3.46% |
| 536870912 | 4.72 | 4.45 | -5.72% | 5.52 | +16.95% |
| 1073741824 | 7.74 | 16.78 | +116.80% | 13.27 | +71.45% |
| 2147483648 | 14.83 | 31.04 | +109.31% | 30.79 | +107.62% |
| 4294967296 | 29.68 | 35.00 | +17.92% | 33.17 | +11.76% |
| 8589934592 | 39.80 | 36.46 | -8.39% | 36.46 | -8.39% |

**简要观察（out-of-place algbw）**

- **≤8MB**：三种配置整体处于 **~0.02–32 GB/s** 量级；相对 AR OFF，**AR ON DDP OFF** 在 **256KB–8MB** 多为 **负向**；**AR ON DDP ON** 在 **16KB** 因 **`algbw` 显示为 0.01** 导致百分比幅度很大（不代表吞吐真实提升/下降，更多是打印精度与噪声）。
- **16MB**：CASE1 的 **out-of-place** 仍显示 **~36.5 GB/s**，但 **CASE2/3** 的时间跃迁到 **~89–90 ms** 且 **`algbw`≈0.19 GB/s**，表现为 **严重性能退化**（相对 CASE1 约 **-99%**）。
- **32MB–512MB**：CASE1 基准本身已处于 **低吞吐区**（约 **0.4–7.7 GB/s**）；**AR ON** 在该区间与 **512MB–2GB** 的形态与 CASE1 **不一致**（例如 **1GB** 附近 **AR ON** 明显高于 CASE1 的塌陷区）。
- **≥4GB**：**AR ON** 在 **2G** 步长相对 CASE1 **更高**；**8G** 步长三者接近（约 **36.5 GB/s**），**AR ON** 略低于 CASE1（约 **-8.4%**）。

---

## 4. 原始数据附录

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:16:31+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   4384 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   4385 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   4386 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   4387 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   4388 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   4389 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   4390 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   4391 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   5047 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   5048 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   5049 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   5050 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   5051 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   5052 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   5053 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   5054 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:04379] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:04379] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    61.85    0.02    0.02       0    47.04    0.02    0.02    N/A
        2048            32     float    none      -1    56.91    0.04    0.03       0    49.19    0.04    0.04    N/A
        4096            64     float    none      -1    55.26    0.07    0.07       0    46.91    0.09    0.08    N/A
        8192           128     float    none      -1    47.35    0.17    0.16       0    46.67    0.18    0.16    N/A
       16384           256     float    none      -1    48.81    0.34    0.31       0    47.00    0.35    0.33    N/A
       32768           512     float    none      -1    47.63    0.69    0.64       0    47.36    0.69    0.65    N/A
       65536          1024     float    none      -1    49.76    1.32    1.23       0    48.04    1.36    1.28    N/A
      131072          2048     float    none      -1    51.22    2.56    2.40       0    51.01    2.57    2.41    N/A
      262144          4096     float    none      -1    52.31    5.01    4.70       0    73.37    3.57    3.35    N/A
      524288          8192     float    none      -1    64.63    8.11    7.61       0    79.19    6.62    6.21    N/A
     1048576         16384     float    none      -1    92.63   11.32   10.61       0    94.73   11.07   10.38    N/A
     2097152         32768     float    none      -1   114.13   18.38   17.23       0   103.71   20.22   18.96    N/A
     4194304         65536     float    none      -1   179.32   23.39   21.93       0   189.67   22.11   20.73    N/A
     8388608        131072     float    none      -1   262.88   31.91   29.92       0   261.06   32.13   30.12    N/A
    16777216        262144     float    none      -1   459.13   36.54   34.26       0  70123.2    0.24    0.22    N/A
    33554432        524288     float    none      -1  34787.6    0.96    0.90       0  81153.3    0.41    0.39    N/A
    67108864       1048576     float    none      -1  53140.7    1.26    1.18       0  73344.9    0.91    0.86    N/A
   134217728       2097152     float    none      -1   100829    1.33    1.25       0  90898.7    1.48    1.38    N/A
   268435456       4194304     float    none      -1   103234    2.60    2.44       0   125675    2.14    2.00    N/A
   536870912       8388608     float    none      -1   113859    4.72    4.42       0   110986    4.84    4.53    N/A
  1073741824      16777216     float    none      -1   138791    7.74    7.25       0   152069    7.06    6.62    N/A
  2147483648      33554432     float    none      -1   144824   14.83   13.90       0   139516   15.39   14.43    N/A
  4294967296      67108864     float    none      -1   144716   29.68   27.82       0   174083   24.67   23.13    N/A
  8589934592     134217728     float    none      -1   215847   39.80   37.31       0   205790   41.74   39.13    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.64628 
#
# Collective test concluded: alltoall_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:18:27+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   4864 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   4865 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   4866 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   4867 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   4868 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   4869 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   4870 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   4871 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   5603 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   5604 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   5605 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   5606 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   5607 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   5608 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   5609 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   5610 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:04859] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:04859] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    64.19    0.02    0.01       0    47.96    0.02    0.02    N/A
        2048            32     float    none      -1    49.55    0.04    0.04       0    46.93    0.04    0.04    N/A
        4096            64     float    none      -1    46.94    0.09    0.08       0    46.55    0.09    0.08    N/A
        8192           128     float    none      -1    47.09    0.17    0.16       0    46.36    0.18    0.17    N/A
       16384           256     float    none      -1    47.22    0.35    0.33       0    46.68    0.35    0.33    N/A
       32768           512     float    none      -1    47.51    0.69    0.65       0    48.34    0.68    0.64    N/A
       65536          1024     float    none      -1    47.79    1.37    1.29       0    47.52    1.38    1.29    N/A
      131072          2048     float    none      -1    49.85    2.63    2.47       0    49.99    2.62    2.46    N/A
      262144          4096     float    none      -1    69.25    3.79    3.55       0    73.14    3.58    3.36    N/A
      524288          8192     float    none      -1    81.06    6.47    6.06       0    78.86    6.65    6.23    N/A
     1048576         16384     float    none      -1   100.66   10.42    9.77       0   102.58   10.22    9.58    N/A
     2097152         32768     float    none      -1   122.41   17.13   16.06       0   134.62   15.58   14.60    N/A
     4194304         65536     float    none      -1   199.91   20.98   19.67       0   192.35   21.81   20.44    N/A
     8388608        131072     float    none      -1   316.89   26.47   24.82       0  4133.70    2.03    1.90    N/A
    16777216        262144     float    none      -1  90448.4    0.19    0.17       0  86418.0    0.19    0.18    N/A
    33554432        524288     float    none      -1  43943.3    0.76    0.72       0  84871.7    0.40    0.37    N/A
    67108864       1048576     float    none      -1  75045.7    0.89    0.84       0  65936.3    1.02    0.95    N/A
   134217728       2097152     float    none      -1  97335.4    1.38    1.29       0  76642.4    1.75    1.64    N/A
   268435456       4194304     float    none      -1  96508.3    2.78    2.61       0  88828.3    3.02    2.83    N/A
   536870912       8388608     float    none      -1   120658    4.45    4.17       0  77253.7    6.95    6.52    N/A
  1073741824      16777216     float    none      -1  63996.6   16.78   15.73       0  44333.9   24.22   22.71    N/A
  2147483648      33554432     float    none      -1  69190.5   31.04   29.10       0  72648.5   29.56   27.71    N/A
  4294967296      67108864     float    none      -1   122720   35.00   32.81       0   129516   33.16   31.09    N/A
  8589934592     134217728     float    none      -1   235591   36.46   34.18       0   242039   35.49   33.27    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.22904 
#
# Collective test concluded: alltoall_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:20:17+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   5344 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   5345 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   5346 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   5347 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   5348 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   5349 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   5350 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   5351 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   6159 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   6160 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   6161 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   6162 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   6163 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   6164 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   6165 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   6166 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:05339] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:05339] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    71.11    0.01    0.01       0    47.05    0.02    0.02    N/A
        2048            32     float    none      -1    47.08    0.04    0.04       0    46.95    0.04    0.04    N/A
        4096            64     float    none      -1    50.57    0.08    0.08       0    47.21    0.09    0.08    N/A
        8192           128     float    none      -1    46.75    0.18    0.16       0    46.69    0.18    0.16    N/A
       16384           256     float    none      -1    64.14    0.26    0.24       0    47.10    0.35    0.33    N/A
       32768           512     float    none      -1    47.65    0.69    0.64       0    47.32    0.69    0.65    N/A
       65536          1024     float    none      -1    48.05    1.36    1.28       0    48.99    1.34    1.25    N/A
      131072          2048     float    none      -1    50.61    2.59    2.43       0    49.87    2.63    2.46    N/A
      262144          4096     float    none      -1    68.32    3.84    3.60       0    84.54    3.10    2.91    N/A
      524288          8192     float    none      -1    79.75    6.57    6.16       0    78.14    6.71    6.29    N/A
     1048576         16384     float    none      -1   100.05   10.48    9.83       0   114.02    9.20    8.62    N/A
     2097152         32768     float    none      -1   122.71   17.09   16.02       0   120.93   17.34   16.26    N/A
     4194304         65536     float    none      -1   195.69   21.43   20.09       0   192.11   21.83   20.47    N/A
     8388608        131072     float    none      -1   315.64   26.58   24.92       0   310.97   26.98   25.29    N/A
    16777216        262144     float    none      -1  89138.0    0.19    0.18       0  82657.0    0.20    0.19    N/A
    33554432        524288     float    none      -1  79962.2    0.42    0.39       0  92740.5    0.36    0.34    N/A
    67108864       1048576     float    none      -1  95230.6    0.70    0.66       0  65313.5    1.03    0.96    N/A
   134217728       2097152     float    none      -1   119759    1.12    1.05       0  98259.8    1.37    1.28    N/A
   268435456       4194304     float    none      -1   106998    2.51    2.35       0  87207.2    3.08    2.89    N/A
   536870912       8388608     float    none      -1  97191.4    5.52    5.18       0   100721    5.33    5.00    N/A
  1073741824      16777216     float    none      -1  80887.2   13.27   12.44       0  47885.0   22.42   21.02    N/A
  2147483648      33554432     float    none      -1  69741.1   30.79   28.87       0  70716.9   30.37   28.47    N/A
  4294967296      67108864     float    none      -1   129467   33.17   31.10       0   121460   35.36   33.15    N/A
  8589934592     134217728     float    none      -1   235584   36.46   34.18       0   236517   36.32   34.05    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.62695 
#
# Collective test concluded: alltoall_perf
#
```
