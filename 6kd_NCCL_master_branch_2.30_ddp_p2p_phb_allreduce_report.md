# 6KD NCCL `all_reduce_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_LEVEL=PHB`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `all_reduce_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置；本组数据在 **`NCCL_P2P_LEVEL=PHB`** 下采集（允许在 PHB 拓扑范围内使用 P2P，与 `NCCL_P2P_DISABLE=1` 系列不同）。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `all_reduce_perf` 参数保持一致；均设置 **`-x NCCL_P2P_LEVEL=PHB`**。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `all_reduce_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| P2P | **`NCCL_P2P_LEVEL=PHB`**（在 PCI/NUMA 拓扑的 PHB 范围内允许 P2P） |
| 其他 | `NCCL_SHM_DISABLE=1`；UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备 |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.03 | 0.03 | 0.00% | 0.03 | 0.00% |
| 2048 | 0.06 | 0.06 | 0.00% | 0.06 | 0.00% |
| 4096 | 0.10 | 0.10 | 0.00% | 0.10 | 0.00% |
| 8192 | 0.17 | 0.17 | 0.00% | 0.18 | +5.88% |
| 16384 | 0.37 | 0.38 | +2.70% | 0.37 | 0.00% |
| 32768 | 0.42 | 0.54 | +28.57% | 0.53 | +26.19% |
| 65536 | 0.65 | 0.70 | +7.69% | 0.65 | 0.00% |
| 131072 | 0.66 | 0.66 | 0.00% | 0.66 | 0.00% |
| 262144 | 0.75 | 0.76 | +1.33% | 0.69 | -8.00% |
| 524288 | 0.75 | 0.71 | -5.33% | 0.81 | +8.00% |
| 1048576 | 0.72 | 0.82 | +13.89% | 0.82 | +13.89% |
| 2097152 | 5.30 | 5.08 | -4.15% | 5.31 | +0.19% |
| 4194304 | 5.74 | 5.58 | -2.79% | 5.69 | -0.87% |
| 8388608 | 5.91 | 5.93 | +0.34% | 5.87 | -0.68% |
| 16777216 | 17.14 | 17.20 | +0.35% | 17.35 | +1.22% |
| 33554432 | 20.85 | 20.62 | -1.10% | 20.71 | -0.67% |
| 67108864 | 23.62 | 23.75 | +0.55% | 23.77 | +0.63% |
| 134217728 | 25.35 | 25.41 | +0.24% | 25.32 | -0.12% |
| 268435456 | 26.08 | 26.18 | +0.38% | 26.12 | +0.15% |
| 536870912 | 26.23 | 26.20 | -0.11% | 26.28 | +0.19% |
| 1073741824 | 26.33 | 26.34 | +0.04% | 26.38 | +0.19% |
| 2147483648 | 26.39 | 26.39 | 0.00% | 26.40 | +0.04% |
| 4294967296 | 26.38 | 26.47 | +0.34% | 26.45 | +0.27% |
| 8589934592 | 26.45 | 26.50 | +0.19% | 26.50 | +0.19% |

**简要观察（out-of-place algbw）**

- **中小消息（约 32KB–1MB）**：相对 AR OFF，**AR ON DDP OFF** 在 **32KB、64KB、1MB** 等步长上 **algbw 更高**；但在 **256KB、2MB** 等步长上出现 **回落**。**AR ON DDP ON** 在 **8KB** 与 **512KB–1MB** 区间相对基准 **更高**，而在 **256KB** 步长相对基准 **更低**，整体呈强尺寸依赖（该区间同时受 P2P/多路径/算法切换影响）。
- **大消息（约 16MB 及以上）**：三种配置收敛到约 **17–26.5 GB/s** 量级，差异多在 **±1%** 以内；尾部 **8GB** 步长上 **AR ON** 两种配置相对基准约 **+0.2%** 量级。

---

## 4. 原始数据附录

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:03:40+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid    653 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid    654 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid    655 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid    656 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid    657 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid    658 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid    659 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid    660 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid    641 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid    642 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid    643 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid    644 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid    645 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid    646 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid    647 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid    648 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    36.95    0.03    0.05       0    34.17    0.03    0.06       0
        2048           512     float     sum      -1    36.47    0.06    0.11       0    36.25    0.06    0.11       0
        4096          1024     float     sum      -1    41.34    0.10    0.19       0    41.01    0.10    0.19       0
        8192          2048     float     sum      -1    46.91    0.17    0.33       0    42.81    0.19    0.36       0
       16384          4096     float     sum      -1    44.01    0.37    0.70       0    42.21    0.39    0.73       0
       32768          8192     float     sum      -1    77.31    0.42    0.79       0    76.94    0.43    0.80       0
       65536         16384     float     sum      -1   100.50    0.65    1.22       0   114.79    0.57    1.07       0
      131072         32768     float     sum      -1   199.03    0.66    1.23       0   192.53    0.68    1.28       0
[R6KD-CX8aaS-GPU-11:00648] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:00648] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
      262144         65536     float     sum      -1   348.64    0.75    1.41       0   331.31    0.79    1.48       0
      524288        131072     float     sum      -1   702.28    0.75    1.40       0   705.70    0.74    1.39       0
     1048576        262144     float     sum      -1  1463.07    0.72    1.34       0  1362.59    0.77    1.44       0
     2097152        524288     float     sum      -1   395.41    5.30    9.94       0   392.99    5.34   10.01       0
     4194304       1048576     float     sum      -1   731.06    5.74   10.76       0   722.95    5.80   10.88       0
     8388608       2097152     float     sum      -1  1419.73    5.91   11.08       0  1432.04    5.86   10.98       0
    16777216       4194304     float     sum      -1   978.58   17.14   32.15       0   977.19   17.17   32.19       0
    33554432       8388608     float     sum      -1  1609.30   20.85   39.09       0  1614.94   20.78   38.96       0
    67108864      16777216     float     sum      -1  2840.76   23.62   44.29       0  2835.53   23.67   44.38       0
   134217728      33554432     float     sum      -1  5294.64   25.35   47.53       0  5286.45   25.39   47.60       0
   268435456      67108864     float     sum      -1  10292.6   26.08   48.90       0  10300.9   26.06   48.86       0
   536870912     134217728     float     sum      -1  20464.1   26.23   49.19       0  20430.1   26.28   49.27       0
  1073741824     268435456     float     sum      -1  40773.2   26.33   49.38       0  40764.1   26.34   49.39       0
  2147483648     536870912     float     sum      -1  81362.5   26.39   49.49       0  81400.1   26.38   49.47       0
  4294967296    1073741824     float     sum      -1   162826   26.38   49.46       0   162454   26.44   49.57       0
  8589934592    2147483648     float     sum      -1   324761   26.45   49.59       0   324641   26.46   49.61       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.827 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:04:47+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid    987 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid    988 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid    989 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid    990 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid    991 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid    992 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid    993 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid    994 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   1051 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   1052 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   1053 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   1054 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   1055 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   1056 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   1057 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   1058 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:00982] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:00982] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    36.78    0.03    0.05       0    34.77    0.03    0.06       0
        2048           512     float     sum      -1    36.97    0.06    0.10       0    36.98    0.06    0.10       0
        4096          1024     float     sum      -1    41.93    0.10    0.18       0    41.79    0.10    0.18       0
        8192          2048     float     sum      -1    47.82    0.17    0.32       0    41.88    0.20    0.37       0
       16384          4096     float     sum      -1    43.04    0.38    0.71       0    43.05    0.38    0.71       0
       32768          8192     float     sum      -1    60.64    0.54    1.01       0    61.86    0.53    0.99       0
       65536         16384     float     sum      -1    94.28    0.70    1.30       0    96.54    0.68    1.27       0
      131072         32768     float     sum      -1   197.31    0.66    1.25       0   186.20    0.70    1.32       0
      262144         65536     float     sum      -1   344.17    0.76    1.43       0   345.45    0.76    1.42       0
      524288        131072     float     sum      -1   742.88    0.71    1.32       0   746.45    0.70    1.32       0
     1048576        262144     float     sum      -1  1277.06    0.82    1.54       0  1334.56    0.79    1.47       0
     2097152        524288     float     sum      -1   412.46    5.08    9.53       0   410.85    5.10    9.57       0
     4194304       1048576     float     sum      -1   751.54    5.58   10.46       0   746.38    5.62   10.54       0
     8388608       2097152     float     sum      -1  1414.02    5.93   11.12       0  1463.09    5.73   10.75       0
    16777216       4194304     float     sum      -1   975.23   17.20   32.26       0   975.81   17.19   32.24       0
    33554432       8388608     float     sum      -1  1627.55   20.62   38.66       0  1619.69   20.72   38.84       0
    67108864      16777216     float     sum      -1  2826.07   23.75   44.52       0  2842.29   23.61   44.27       0
   134217728      33554432     float     sum      -1  5281.73   25.41   47.65       0  5294.02   25.35   47.54       0
   268435456      67108864     float     sum      -1  10253.8   26.18   49.09       0  10249.1   26.19   49.11       0
   536870912     134217728     float     sum      -1  20490.3   26.20   49.13       0  20402.4   26.31   49.34       0
  1073741824     268435456     float     sum      -1  40761.0   26.34   49.39       0  40711.7   26.37   49.45       0
  2147483648     536870912     float     sum      -1  81364.5   26.39   49.49       0  81287.5   26.42   49.53       0
  4294967296    1073741824     float     sum      -1   162271   26.47   49.63       0   162379   26.45   49.59       0
  8589934592    2147483648     float     sum      -1   324127   26.50   49.69       0   324305   26.49   49.66       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.8229 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:05:55+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_LEVEL=PHB -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   1321 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   1322 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   1323 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   1324 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   1325 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   1326 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   1327 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   1328 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   1461 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   1462 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   1463 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   1464 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   1465 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   1466 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   1467 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   1468 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:01316] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:01316] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    36.83    0.03    0.05       0    34.08    0.03    0.06       0
        2048           512     float     sum      -1    36.80    0.06    0.10       0    37.01    0.06    0.10       0
        4096          1024     float     sum      -1    41.12    0.10    0.19       0    41.50    0.10    0.19       0
        8192          2048     float     sum      -1    45.13    0.18    0.34       0    42.59    0.19    0.36       0
       16384          4096     float     sum      -1    44.12    0.37    0.70       0    42.59    0.38    0.72       0
       32768          8192     float     sum      -1    61.35    0.53    1.00       0    65.28    0.50    0.94       0
       65536         16384     float     sum      -1   100.96    0.65    1.22       0   102.11    0.64    1.20       0
      131072         32768     float     sum      -1   200.11    0.66    1.23       0   201.01    0.65    1.22       0
      262144         65536     float     sum      -1   381.78    0.69    1.29       0   398.95    0.66    1.23       0
      524288        131072     float     sum      -1   649.81    0.81    1.51       0   643.88    0.81    1.53       0
     1048576        262144     float     sum      -1  1282.58    0.82    1.53       0  1308.07    0.80    1.50       0
     2097152        524288     float     sum      -1   395.05    5.31    9.95       0   391.86    5.35   10.03       0
     4194304       1048576     float     sum      -1   736.92    5.69   10.67       0   728.71    5.76   10.79       0
     8388608       2097152     float     sum      -1  1429.00    5.87   11.01       0  1408.48    5.96   11.17       0
    16777216       4194304     float     sum      -1   966.86   17.35   32.54       0   971.12   17.28   32.39       0
    33554432       8388608     float     sum      -1  1619.88   20.71   38.84       0  1608.97   20.85   39.10       0
    67108864      16777216     float     sum      -1  2823.70   23.77   44.56       0  2833.58   23.68   44.41       0
   134217728      33554432     float     sum      -1  5300.52   25.32   47.48       0  5284.17   25.40   47.62       0
   268435456      67108864     float     sum      -1  10277.2   26.12   48.97       0  10273.1   26.13   48.99       0
   536870912     134217728     float     sum      -1  20425.5   26.28   49.28       0  20432.3   26.28   49.27       0
  1073741824     268435456     float     sum      -1  40701.7   26.38   49.46       0  40742.9   26.35   49.41       0
  2147483648     536870912     float     sum      -1  81349.6   26.40   49.50       0  81335.4   26.40   49.51       0
  4294967296    1073741824     float     sum      -1   162383   26.45   49.59       0   162298   26.46   49.62       0
  8589934592    2147483648     float     sum      -1   324115   26.50   49.69       0   324150   26.50   49.69       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.8702 
#
# Collective test concluded: all_reduce_perf
#
```
