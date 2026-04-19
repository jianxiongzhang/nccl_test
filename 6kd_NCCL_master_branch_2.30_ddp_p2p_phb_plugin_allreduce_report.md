# 6KD NCCL `all_reduce_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_LEVEL=PHB`，`NCCL_NET_PLUGIN=spcx`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `all_reduce_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置；本组同时启用 **Spectrum-X NCCL 插件**（**`NCCL_NET_PLUGIN=spcx`**）与 **`NCCL_P2P_LEVEL=PHB`**，并在 **`LD_LIBRARY_PATH`** 中前置 **`nccl_spectrum-x_plugin/lib`**。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

**数据说明**：**AR ON** 两种配置在 **8MB** 步长出现 **out-of-place 时间 ~47–48 ms** 且 **`algbw`≈0.18 GB/s** 的 **断崖式退化**（相对 **AR OFF** 约 **-96%**）；该点建议结合 **NCCL/插件版本、拓扑与复现实验**单独排查。CASE1 日志中还出现一次 **`ORTE_ERROR_LOG: Data unpack would read past end of buffer`**（见原始数据附录）。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `all_reduce_perf` 参数保持一致；均包含 **`NCCL_NET_PLUGIN=spcx`**、**`NCCL_P2P_LEVEL=PHB`** 以及 Spectrum-X 插件库的 **`LD_LIBRARY_PATH`** 前缀。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `all_reduce_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**；`LD_LIBRARY_PATH` 包含 **`.../nccl_spectrum-x_plugin/lib`** |
| P2P | **`NCCL_P2P_LEVEL=PHB`** |
| 其他 | `NCCL_SHM_DISABLE=1`；UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备 |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.03 | 0.03 | 0.00% | 0.03 | 0.00% |
| 2048 | 0.06 | 0.06 | 0.00% | 0.06 | 0.00% |
| 4096 | 0.10 | 0.10 | 0.00% | 0.10 | 0.00% |
| 8192 | 0.18 | 0.17 | -5.56% | 0.17 | -5.56% |
| 16384 | 0.36 | 0.35 | -2.78% | 0.35 | -2.78% |
| 32768 | 0.59 | 0.53 | -10.17% | 0.58 | -1.69% |
| 65536 | 0.71 | 0.64 | -9.86% | 0.70 | -1.41% |
| 131072 | 0.55 | 0.59 | +7.27% | 0.62 | +12.73% |
| 262144 | 0.59 | 0.63 | +6.78% | 0.68 | +15.25% |
| 524288 | 0.81 | 0.87 | +7.41% | 0.89 | +9.88% |
| 1048576 | 0.81 | 0.83 | +2.47% | 0.83 | +2.47% |
| 2097152 | 5.29 | 5.05 | -4.54% | 4.97 | -6.05% |
| 4194304 | 5.84 | 5.54 | -5.14% | 5.56 | -4.79% |
| 8388608 | 4.18 | 0.18 | -95.69% | 0.18 | -95.69% |
| 16777216 | 16.98 | 15.98 | -5.89% | 15.98 | -5.89% |
| 33554432 | 20.62 | 19.62 | -4.85% | 19.61 | -4.90% |
| 67108864 | 23.43 | 22.82 | -2.60% | 22.87 | -2.39% |
| 134217728 | 25.18 | 24.80 | -1.51% | 24.94 | -0.95% |
| 268435456 | 26.15 | 26.03 | -0.46% | 26.01 | -0.54% |
| 536870912 | 26.23 | 26.14 | -0.34% | 26.15 | -0.31% |
| 1073741824 | 26.39 | 26.28 | -0.42% | 26.30 | -0.34% |
| 2147483648 | 26.44 | 26.35 | -0.34% | 26.31 | -0.49% |
| 4294967296 | 26.47 | 26.39 | -0.30% | 26.37 | -0.38% |
| 8589934592 | 26.51 | 26.38 | -0.49% | 26.36 | -0.57% |

**简要观察（out-of-place algbw）**

- **8MB**：**AR ON DDP OFF** 与 **AR ON DDP ON** 相对 **AR OFF** 均约 **-96%**（`algbw` 从 **4.18** 掉到 **0.18 GB/s**），与 **~47–48 ms** 量级的延迟尖峰一致。
- **中小消息（约 128KB–1MB）**：相对 AR OFF，**AR ON** 在多个步长上 **`algbw` 更高**（约 **+2%–+16%**）；**32KB–64KB** 区间 **AR ON DDP OFF** 偏 **慢**（约 **-10%**），**AR ON DDP ON** 更接近基准。
- **≥16MB**：三种配置整体回到 **~16–26.5 GB/s**；相对基准多为 **小幅负向**（多数在 **-0.3% 至 -6%** 量级，**16MB** 附近约 **-6%**）。

---

## 4. 原始数据附录

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:10:49+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
[R6KD-CX8aaS-GPU-11:02705] [[23664,0],0] ORTE_ERROR_LOG: Data unpack would read past end of buffer in file util/show_help.c at line 501
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   2710 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   2711 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   2712 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   2713 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   2714 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   2715 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   2716 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   2717 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   3107 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   3108 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   3109 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   3110 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   3111 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   3112 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   3113 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   3114 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:02705] 14 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:02705] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    37.38    0.03    0.05       0    33.89    0.03    0.06       0
        2048           512     float     sum      -1    36.51    0.06    0.11       0    36.12    0.06    0.11       0
        4096          1024     float     sum      -1    41.71    0.10    0.18       0    41.70    0.10    0.18       0
        8192          2048     float     sum      -1    45.15    0.18    0.34       0    43.27    0.19    0.36       0
       16384          4096     float     sum      -1    45.18    0.36    0.68       0    40.71    0.40    0.75       0
       32768          8192     float     sum      -1    55.88    0.59    1.10       0    60.14    0.54    1.02       0
       65536         16384     float     sum      -1    91.81    0.71    1.34       0    99.74    0.66    1.23       0
      131072         32768     float     sum      -1   236.29    0.55    1.04       0   240.64    0.54    1.02       0
      262144         65536     float     sum      -1   440.71    0.59    1.12       0   425.40    0.62    1.16       0
      524288        131072     float     sum      -1   644.02    0.81    1.53       0   618.36    0.85    1.59       0
     1048576        262144     float     sum      -1  1288.70    0.81    1.53       0  1299.13    0.81    1.51       0
     2097152        524288     float     sum      -1   396.23    5.29    9.92       0   379.91    5.52   10.35       0
     4194304       1048576     float     sum      -1   718.71    5.84   10.94       0   728.00    5.76   10.80       0
     8388608       2097152     float     sum      -1  2005.46    4.18    7.84       0  4856.79    1.73    3.24       0
    16777216       4194304     float     sum      -1   988.09   16.98   31.84       0   998.85   16.80   31.49       0
    33554432       8388608     float     sum      -1  1626.98   20.62   38.67       0  1639.57   20.47   38.37       0
    67108864      16777216     float     sum      -1  2863.84   23.43   43.94       0  2850.03   23.55   44.15       0
   134217728      33554432     float     sum      -1  5331.10   25.18   47.21       0  5314.50   25.26   47.35       0
   268435456      67108864     float     sum      -1  10265.1   26.15   49.03       0  10275.7   26.12   48.98       0
   536870912     134217728     float     sum      -1  20471.5   26.23   49.17       0  20453.3   26.25   49.22       0
  1073741824     268435456     float     sum      -1  40692.8   26.39   49.47       0  40707.5   26.38   49.46       0
  2147483648     536870912     float     sum      -1  81209.7   26.44   49.58       0  81239.7   26.43   49.56       0
  4294967296    1073741824     float     sum      -1   162229   26.47   49.64       0   162262   26.47   49.63       0
  8589934592    2147483648     float     sum      -1   323997   26.51   49.71       0   324082   26.51   49.70       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.5682 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:12:22+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   3188 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   3189 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   3190 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   3191 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   3192 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   3193 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   3194 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   3195 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   3661 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   3662 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   3663 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   3664 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   3665 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   3666 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   3667 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   3668 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:03183] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:03183] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    35.63    0.03    0.05       0    34.00    0.03    0.06       0
        2048           512     float     sum      -1    36.87    0.06    0.10       0    36.87    0.06    0.10       0
        4096          1024     float     sum      -1    41.20    0.10    0.19       0    40.98    0.10    0.19       0
        8192          2048     float     sum      -1    48.72    0.17    0.32       0    43.18    0.19    0.36       0
       16384          4096     float     sum      -1    46.15    0.35    0.67       0    44.90    0.36    0.68       0
       32768          8192     float     sum      -1    61.79    0.53    0.99       0    59.96    0.55    1.02       0
       65536         16384     float     sum      -1   102.86    0.64    1.19       0    97.61    0.67    1.26       0
      131072         32768     float     sum      -1   222.00    0.59    1.11       0   216.60    0.61    1.13       0
      262144         65536     float     sum      -1   414.36    0.63    1.19       0   419.00    0.63    1.17       0
      524288        131072     float     sum      -1   605.72    0.87    1.62       0   638.33    0.82    1.54       0
     1048576        262144     float     sum      -1  1257.52    0.83    1.56       0  1344.99    0.78    1.46       0
     2097152        524288     float     sum      -1   414.96    5.05    9.48       0   416.88    5.03    9.43       0
     4194304       1048576     float     sum      -1   757.36    5.54   10.38       0   752.74    5.57   10.45       0
     8388608       2097152     float     sum      -1  47785.5    0.18    0.33       0  58280.3    0.14    0.27       0
    16777216       4194304     float     sum      -1  1050.16   15.98   29.95       0  1054.92   15.90   29.82       0
    33554432       8388608     float     sum      -1  1709.90   19.62   36.79       0  1727.68   19.42   36.42       0
    67108864      16777216     float     sum      -1  2940.38   22.82   42.79       0  2941.82   22.81   42.77       0
   134217728      33554432     float     sum      -1  5410.98   24.80   46.51       0  5400.63   24.85   46.60       0
   268435456      67108864     float     sum      -1  10312.4   26.03   48.81       0  10307.8   26.04   48.83       0
   536870912     134217728     float     sum      -1  20540.3   26.14   49.01       0  20516.3   26.17   49.07       0
  1073741824     268435456     float     sum      -1  40865.0   26.28   49.27       0  40825.5   26.30   49.31       0
  2147483648     536870912     float     sum      -1  81508.1   26.35   49.40       0  81445.3   26.37   49.44       0
  4294967296    1073741824     float     sum      -1   162764   26.39   49.48       0   163079   26.34   49.38       0
  8589934592    2147483648     float     sum      -1   325615   26.38   49.46       0   328901   26.12   48.97       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.0081 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:14:01+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   3666 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   3667 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   3668 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   3669 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   3670 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   3671 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   3672 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   3673 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   4215 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   4216 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   4217 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   4218 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   4219 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   4220 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   4221 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   4222 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:03661] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:03661] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    37.40    0.03    0.05       0    34.12    0.03    0.06       0
        2048           512     float     sum      -1    37.21    0.06    0.10       0    36.43    0.06    0.11       0
        4096          1024     float     sum      -1    41.60    0.10    0.18       0    41.28    0.10    0.19       0
        8192          2048     float     sum      -1    47.87    0.17    0.32       0    41.87    0.20    0.37       0
       16384          4096     float     sum      -1    46.70    0.35    0.66       0    41.44    0.40    0.74       0
       32768          8192     float     sum      -1    56.13    0.58    1.09       0    57.28    0.57    1.07       0
       65536         16384     float     sum      -1    93.64    0.70    1.31       0    95.89    0.68    1.28       0
      131072         32768     float     sum      -1   210.76    0.62    1.17       0   212.04    0.62    1.16       0
      262144         65536     float     sum      -1   386.00    0.68    1.27       0   387.21    0.68    1.27       0
      524288        131072     float     sum      -1   586.66    0.89    1.68       0   593.47    0.88    1.66       0
     1048576        262144     float     sum      -1  1256.12    0.83    1.57       0  1150.69    0.91    1.71       0
     2097152        524288     float     sum      -1   421.74    4.97    9.32       0   408.57    5.13    9.62       0
     4194304       1048576     float     sum      -1   753.80    5.56   10.43       0   747.48    5.61   10.52       0
     8388608       2097152     float     sum      -1  47675.0    0.18    0.33       0  22071.5    0.38    0.71       0
    16777216       4194304     float     sum      -1  1050.00   15.98   29.96       0  1042.16   16.10   30.18       0
    33554432       8388608     float     sum      -1  1710.79   19.61   36.78       0  1709.31   19.63   36.81       0
    67108864      16777216     float     sum      -1  2934.79   22.87   42.87       0  2940.73   22.82   42.79       0
   134217728      33554432     float     sum      -1  5382.58   24.94   46.75       0  5392.23   24.89   46.67       0
   268435456      67108864     float     sum      -1  10321.7   26.01   48.76       0  10292.6   26.08   48.90       0
   536870912     134217728     float     sum      -1  20532.6   26.15   49.03       0  20510.1   26.18   49.08       0
  1073741824     268435456     float     sum      -1  40829.7   26.30   49.31       0  40856.7   26.28   49.28       0
  2147483648     536870912     float     sum      -1  81620.6   26.31   49.33       0  81519.0   26.34   49.39       0
  4294967296    1073741824     float     sum      -1   162861   26.37   49.45       0   162957   26.36   49.42       0
  8589934592    2147483648     float     sum      -1   325860   26.36   49.43       0   325396   26.40   49.50       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.0758 
#
# Collective test concluded: all_reduce_perf
#
```
