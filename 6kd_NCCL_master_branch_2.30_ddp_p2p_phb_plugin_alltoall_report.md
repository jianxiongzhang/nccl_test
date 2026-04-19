# 6KD NCCL `alltoall_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_LEVEL=PHB`，`NCCL_NET_PLUGIN=spcx`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `alltoall_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置；本组同时启用 **Spectrum-X NCCL 插件**（**`NCCL_NET_PLUGIN=spcx`**）与 **`NCCL_P2P_LEVEL=PHB`**，并在 **`LD_LIBRARY_PATH`** 中前置 **`nccl_spectrum-x_plugin/lib`**。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

**数据说明**：CASE1（AR OFF）在 **16MB–512MB** 区间出现 **`algbw` 断崖**（与 **8MB 及以下**的高吞吐不连续）；该区间百分比会极端化，解读需结合 **time 列**与是否复测。CASE2 在 **32MB** 同时出现 **out-of-place 低吞吐**与 **in-place 高吞吐**（**914 µs / 36.70 GB/s**），提示不同路径/缓冲形态差异显著。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `alltoall_perf` 参数保持一致；均包含 **`NCCL_NET_PLUGIN=spcx`**、**`NCCL_P2P_LEVEL=PHB`** 以及 Spectrum-X 插件库的 **`LD_LIBRARY_PATH`** 前缀。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `alltoall_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**；`LD_LIBRARY_PATH` 包含 **`.../nccl_spectrum-x_plugin/lib`** |
| P2P | **`NCCL_P2P_LEVEL=PHB`** |
| 其他 | `NCCL_SHM_DISABLE=1`；UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备 |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.02 | 0.03 | +50.00% | 0.03 | +50.00% |
| 2048 | 0.07 | 0.06 | -14.29% | 0.06 | -14.29% |
| 4096 | 0.13 | 0.13 | 0.00% | 0.13 | 0.00% |
| 8192 | 0.28 | 0.27 | -3.57% | 0.27 | -3.57% |
| 16384 | 0.53 | 0.53 | 0.00% | 0.53 | 0.00% |
| 32768 | 1.00 | 1.00 | 0.00% | 1.01 | +1.00% |
| 65536 | 1.89 | 1.88 | -0.53% | 1.92 | +1.59% |
| 131072 | 3.48 | 3.37 | -3.16% | 3.49 | +0.29% |
| 262144 | 6.28 | 5.03 | -19.90% | 5.28 | -15.92% |
| 524288 | 10.69 | 8.30 | -22.36% | 9.09 | -15.06% |
| 1048576 | 14.53 | 13.05 | -10.19% | 13.20 | -9.15% |
| 2097152 | 23.19 | 19.60 | -15.48% | 19.87 | -14.32% |
| 4194304 | 25.63 | 22.37 | -12.72% | 22.27 | -13.11% |
| 8388608 | 31.85 | 29.14 | -8.51% | 29.89 | -6.12% |
| 16777216 | 1.58 | 0.30 | -81.01% | 1.14 | -27.85% |
| 33554432 | 3.41 | 1.02 | -70.09% | 0.93 | -72.73% |
| 67108864 | 1.50 | 0.62 | -58.67% | 0.70 | -53.33% |
| 134217728 | 1.54 | 1.12 | -27.27% | 1.08 | -29.87% |
| 268435456 | 2.71 | 2.81 | +3.69% | 2.37 | -12.55% |
| 536870912 | 5.39 | 6.05 | +12.24% | 5.48 | +1.67% |
| 1073741824 | 10.75 | 11.24 | +4.56% | 9.88 | -8.09% |
| 2147483648 | 17.89 | 22.80 | +27.45% | 28.43 | +58.91% |
| 4294967296 | 31.11 | 33.26 | +6.91% | 33.26 | +6.91% |
| 8589934592 | 38.30 | 36.63 | -4.36% | 36.41 | -4.94% |

**简要观察（out-of-place algbw）**

- **≤8MB**：相对 AR OFF，**AR ON** 两种配置在 **256KB–8MB** 多为 **负向**（约 **-6% 至 -22%**）；更小消息上差异较小（**1024B** 因 **`algbw` 打印精度**出现 **+50%** 这类幅度，需谨慎解读）。
- **16MB–512MB（CASE1 基准已塌陷）**：**AR ON DDP OFF** 在 **16MB** 相对基准更差（约 **-81%**）；**AR ON DDP ON** 在 **16MB** 相对基准约 **-28%**。**2GB** 步长上 **AR ON** 明显高于 CASE1 的 **~17.9 GB/s** 基准（约 **+27%–+59%**）。
- **≥4GB**：三种配置回到 **~33–38 GB/s** 量级；**AR ON** 相对 CASE1 在 **4G** 约 **+6.9%**，在 **8G** 约 **-4% 至 -5%**。

---

## 4. 原始数据附录

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:11:29+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   2948 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   2949 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   2950 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   2951 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   2952 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   2953 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   2954 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   2955 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   3383 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   3384 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   3385 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   3386 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   3387 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   3388 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   3389 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   3390 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:02943] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:02943] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    55.66    0.02    0.02       0    29.21    0.04    0.03    N/A
        2048            32     float    none      -1    29.69    0.07    0.06       0    29.51    0.07    0.07    N/A
        4096            64     float    none      -1    30.39    0.13    0.13       0    29.71    0.14    0.13    N/A
        8192           128     float    none      -1    29.69    0.28    0.26       0    29.00    0.28    0.26    N/A
       16384           256     float    none      -1    30.86    0.53    0.50       0    29.90    0.55    0.51    N/A
       32768           512     float    none      -1    32.92    1.00    0.93       0    32.68    1.00    0.94    N/A
       65536          1024     float    none      -1    34.60    1.89    1.78       0    35.48    1.85    1.73    N/A
      131072          2048     float    none      -1    37.62    3.48    3.27       0    51.63    2.54    2.38    N/A
      262144          4096     float    none      -1    41.72    6.28    5.89       0    46.73    5.61    5.26    N/A
      524288          8192     float    none      -1    49.02   10.69   10.03       0    63.96    8.20    7.68    N/A
     1048576         16384     float    none      -1    72.16   14.53   13.62       0    73.72   14.22   13.34    N/A
     2097152         32768     float    none      -1    90.41   23.19   21.75       0    96.23   21.79   20.43    N/A
     4194304         65536     float    none      -1   163.62   25.63   24.03       0   157.00   26.71   25.04    N/A
     8388608        131072     float    none      -1   263.35   31.85   29.86       0   268.41   31.25   29.30    N/A
    16777216        262144     float    none      -1  10642.5    1.58    1.48       0  29073.1    0.58    0.54    N/A
    33554432        524288     float    none      -1  9845.58    3.41    3.20       0  12376.0    2.71    2.54    N/A
    67108864       1048576     float    none      -1  44747.1    1.50    1.41       0  69049.0    0.97    0.91    N/A
   134217728       2097152     float    none      -1  87119.5    1.54    1.44       0   100641    1.33    1.25    N/A
   268435456       4194304     float    none      -1  99106.7    2.71    2.54       0  95238.8    2.82    2.64    N/A
   536870912       8388608     float    none      -1  99571.1    5.39    5.05       0  91569.8    5.86    5.50    N/A
  1073741824      16777216     float    none      -1  99880.1   10.75   10.08       0  98929.6   10.85   10.18    N/A
  2147483648      33554432     float    none      -1   120020   17.89   16.77       0   113043   19.00   17.81    N/A
  4294967296      67108864     float    none      -1   138053   31.11   29.17       0   140429   30.58   28.67    N/A
  8589934592     134217728     float    none      -1   224292   38.30   35.90       0   211843   40.55   38.01    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 9.04853 
#
# Collective test concluded: alltoall_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:13:05+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   3426 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   3427 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   3428 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   3429 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   3430 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   3431 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   3432 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   3433 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   3937 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   3938 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   3939 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   3940 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   3941 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   3942 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   3943 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   3944 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:03421] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:03421] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    38.30    0.03    0.03       0    29.53    0.03    0.03    N/A
        2048            32     float    none      -1    34.89    0.06    0.06       0    32.28    0.06    0.06    N/A
        4096            64     float    none      -1    30.89    0.13    0.12       0    30.01    0.14    0.13    N/A
        8192           128     float    none      -1    30.08    0.27    0.26       0    29.59    0.28    0.26    N/A
       16384           256     float    none      -1    30.93    0.53    0.50       0    30.57    0.54    0.50    N/A
       32768           512     float    none      -1    32.92    1.00    0.93       0    32.80    1.00    0.94    N/A
       65536          1024     float    none      -1    34.88    1.88    1.76       0    36.23    1.81    1.70    N/A
      131072          2048     float    none      -1    38.84    3.37    3.16       0    39.08    3.35    3.14    N/A
      262144          4096     float    none      -1    52.07    5.03    4.72       0    57.68    4.55    4.26    N/A
      524288          8192     float    none      -1    63.20    8.30    7.78       0    59.88    8.76    8.21    N/A
     1048576         16384     float    none      -1    80.37   13.05   12.23       0    81.97   12.79   11.99    N/A
     2097152         32768     float    none      -1   106.99   19.60   18.38       0   108.77   19.28   18.08    N/A
     4194304         65536     float    none      -1   187.53   22.37   20.97       0   182.35   23.00   21.56    N/A
     8388608        131072     float    none      -1   287.84   29.14   27.32       0   292.59   28.67   26.88    N/A
    16777216        262144     float    none      -1  55373.8    0.30    0.28       0  52157.5    0.32    0.30    N/A
    33554432        524288     float    none      -1  32930.9    1.02    0.96       0   914.29   36.70   34.41    N/A
    67108864       1048576     float    none      -1   108198    0.62    0.58       0  77371.1    0.87    0.81    N/A
   134217728       2097152     float    none      -1   119377    1.12    1.05       0  94871.9    1.41    1.33    N/A
   268435456       4194304     float    none      -1  95538.1    2.81    2.63       0  92311.1    2.91    2.73    N/A
   536870912       8388608     float    none      -1  88747.1    6.05    5.67       0  93812.8    5.72    5.37    N/A
  1073741824      16777216     float    none      -1  95532.7   11.24   10.54       0  88408.3   12.15   11.39    N/A
  2147483648      33554432     float    none      -1  94171.7   22.80   21.38       0  71455.4   30.05   28.18    N/A
  4294967296      67108864     float    none      -1   129135   33.26   31.18       0   128207   33.50   31.41    N/A
  8589934592     134217728     float    none      -1   234524   36.63   34.34       0   232613   36.93   34.62    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 9.48105 
#
# Collective test concluded: alltoall_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:14:43+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   3904 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   3905 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   3906 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   3907 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   3908 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   3909 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   3910 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   3911 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   4491 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   4492 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   4493 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   4494 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   4495 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   4496 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   4497 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   4498 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:03899] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:03899] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    38.09    0.03    0.03       0    29.67    0.03    0.03    N/A
        2048            32     float    none      -1    32.95    0.06    0.06       0    29.97    0.07    0.06    N/A
        4096            64     float    none      -1    30.68    0.13    0.13       0    29.97    0.14    0.13    N/A
        8192           128     float    none      -1    29.84    0.27    0.26       0    29.35    0.28    0.26    N/A
       16384           256     float    none      -1    30.73    0.53    0.50       0    30.57    0.54    0.50    N/A
       32768           512     float    none      -1    32.30    1.01    0.95       0    32.53    1.01    0.94    N/A
       65536          1024     float    none      -1    34.18    1.92    1.80       0    34.19    1.92    1.80    N/A
      131072          2048     float    none      -1    37.51    3.49    3.28       0    41.86    3.13    2.94    N/A
      262144          4096     float    none      -1    49.68    5.28    4.95       0    56.88    4.61    4.32    N/A
      524288          8192     float    none      -1    57.67    9.09    8.52       0    64.93    8.07    7.57    N/A
     1048576         16384     float    none      -1    79.46   13.20   12.37       0    84.76   12.37   11.60    N/A
     2097152         32768     float    none      -1   105.54   19.87   18.63       0   111.46   18.82   17.64    N/A
     4194304         65536     float    none      -1   188.37   22.27   20.87       0   201.49   20.82   19.52    N/A
     8388608        131072     float    none      -1   280.65   29.89   28.02       0   292.56   28.67   26.88    N/A
    16777216        262144     float    none      -1  14752.8    1.14    1.07       0  15120.6    1.11    1.04    N/A
    33554432        524288     float    none      -1  36059.3    0.93    0.87       0  4870.96    6.89    6.46    N/A
    67108864       1048576     float    none      -1  96479.4    0.70    0.65       0  77664.8    0.86    0.81    N/A
   134217728       2097152     float    none      -1   124723    1.08    1.01       0  94946.2    1.41    1.33    N/A
   268435456       4194304     float    none      -1   113468    2.37    2.22       0   132539    2.03    1.90    N/A
   536870912       8388608     float    none      -1  97938.8    5.48    5.14       0  99936.1    5.37    5.04    N/A
  1073741824      16777216     float    none      -1   108667    9.88    9.26       0  72051.6   14.90   13.97    N/A
  2147483648      33554432     float    none      -1  75538.9   28.43   26.65       0  72025.3   29.82   27.95    N/A
  4294967296      67108864     float    none      -1   129135   33.26   31.18       0   129751   33.10   31.03    N/A
  8589934592     134217728     float    none      -1   235901   36.41   34.14       0   238305   36.05   33.79    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.95948 
#
# Collective test concluded: alltoall_perf
#
```
