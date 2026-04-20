# 6KD NCCL `all_reduce_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_DISABLE=1`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `all_reduce_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `all_reduce_perf` 参数保持一致。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `all_reduce_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| 网络 | UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备；`NCCL_SHM_DISABLE=1`；**`NCCL_P2P_DISABLE=1`**（禁用 GPU P2P，走网络路径） |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.02 | 0.02 | 0.00% | 0.02 | 0.00% |
| 2048 | 0.04 | 0.03 | -25.00% | 0.03 | -25.00% |
| 4096 | 0.06 | 0.06 | 0.00% | 0.06 | 0.00% |
| 8192 | 0.11 | 0.11 | 0.00% | 0.11 | 0.00% |
| 16384 | 0.19 | 0.20 | +5.26% | 0.19 | 0.00% |
| 32768 | 0.25 | 0.36 | +44.00% | 0.38 | +52.00% |
| 65536 | 0.61 | 0.58 | -4.92% | 0.60 | -1.64% |
| 131072 | 1.25 | 1.14 | -8.80% | 1.31 | +4.80% |
| 262144 | 1.86 | 1.77 | -4.84% | 2.32 | +24.73% |
| 524288 | 3.57 | 2.59 | -27.45% | 3.70 | +3.64% |
| 1048576 | 4.55 | 3.54 | -22.20% | 4.62 | +1.54% |
| 2097152 | 5.30 | 4.57 | -13.77% | 5.51 | +3.96% |
| 4194304 | 5.98 | 5.20 | -13.04% | 6.09 | +1.84% |
| 8388608 | 17.42 | 15.82 | -9.18% | 17.32 | -0.57% |
| 16777216 | 19.57 | 19.98 | +2.10% | 20.00 | +2.20% |
| 33554432 | 19.22 | 19.66 | +2.29% | 19.80 | +3.02% |
| 67108864 | 19.90 | 20.15 | +1.26% | 20.12 | +1.11% |
| 134217728 | 22.71 | 22.83 | +0.53% | 22.38 | -1.45% |
| 268435456 | 21.66 | 21.89 | +1.06% | 21.33 | -1.52% |
| 536870912 | 20.12 | 19.95 | -0.84% | 19.82 | -1.49% |
| 1073741824 | 19.83 | 20.20 | +1.87% | 19.79 | -0.20% |
| 2147483648 | 19.81 | 20.22 | +2.07% | 19.75 | -0.30% |
| 4294967296 | 19.90 | 20.29 | +1.96% | 19.79 | -0.55% |
| 8589934592 | 19.91 | 20.18 | +1.36% | 19.74 | -0.85% |

**简要观察（out-of-place algbw）**

- **中小消息**（约 32KB–2MB）：相对 AR OFF，**AR ON DDP OFF** 在多个步长上出现明显 **负向** 偏差；**AR ON DDP ON** 在部分步长（如 256KB、512KB）反而 **高于** 基准，可能与 IB 乱序/匹配/预投递等路径更匹配当前流量形态有关。
- **大消息**（约 16MB 及以上）：三种配置整体处于相近量级；**AR ON DDP OFF** 在多数大步长上略 **高于** 基准（约 +1%–+2%）；**AR ON DDP ON** 在最大几步长上相对基准略 **低**（约 -0.2% 至 -0.9%）。

---

## 4. 原始数据附录

以下为三次运行日志中的完整相关段落（含设备枚举、表头与数据行），便于复核。

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:07:03+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   1655 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   1656 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   1657 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   1658 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   1659 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   1660 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   1661 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   1662 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   1871 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   1872 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   1873 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   1874 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   1875 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   1876 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   1877 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   1879 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:01650] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:01650] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    60.47    0.02    0.03       0    55.77    0.02    0.03       0
        2048           512     float     sum      -1    58.31    0.04    0.07       0    58.41    0.04    0.07       0
        4096          1024     float     sum      -1    64.76    0.06    0.12       0    64.52    0.06    0.12       0
        8192          2048     float     sum      -1    73.96    0.11    0.21       0    68.22    0.12    0.23       0
       16384          4096     float     sum      -1    85.26    0.19    0.36       0    72.67    0.23    0.42       0
       32768          8192     float     sum      -1   132.57    0.25    0.46       0   131.04    0.25    0.47       0
       65536         16384     float     sum      -1   107.47    0.61    1.14       0   107.01    0.61    1.15       0
      131072         32768     float     sum      -1   104.79    1.25    2.35       0    99.26    1.32    2.48       0
      262144         65536     float     sum      -1   140.67    1.86    3.49       0   144.49    1.81    3.40       0
      524288        131072     float     sum      -1   146.75    3.57    6.70       0   243.32    2.15    4.04       0
     1048576        262144     float     sum      -1   230.58    4.55    8.53       0   231.75    4.52    8.48       0
     2097152        524288     float     sum      -1   395.73    5.30    9.94       0   384.30    5.46   10.23       0
     4194304       1048576     float     sum      -1   701.43    5.98   11.21       0   687.95    6.10   11.43       0
     8388608       2097152     float     sum      -1   481.51   17.42   32.67       0   490.24   17.11   32.08       0
    16777216       4194304     float     sum      -1   857.30   19.57   36.69       0   840.84   19.95   37.41       0
    33554432       8388608     float     sum      -1  1745.86   19.22   36.04       0  1704.56   19.69   36.91       0
    67108864      16777216     float     sum      -1  3372.14   19.90   37.31       0  3410.14   19.68   36.90       0
   134217728      33554432     float     sum      -1  5909.09   22.71   42.59       0  5912.64   22.70   42.56       0
   268435456      67108864     float     sum      -1  12394.1   21.66   40.61       0  12615.5   21.28   39.90       0
   536870912     134217728     float     sum      -1  26682.6   20.12   37.73       0  26968.3   19.91   37.33       0
  1073741824     268435456     float     sum      -1  54136.6   19.83   37.19       0  53941.1   19.91   37.32       0
  2147483648     536870912     float     sum      -1   108393   19.81   37.15       0   108243   19.84   37.20       0
  4294967296    1073741824     float     sum      -1   215807   19.90   37.32       0   215897   19.89   37.30       0
  8589934592    2147483648     float     sum      -1   431537   19.91   37.32       0   432914   19.84   37.20       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.9974 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:08:18+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   1991 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   1992 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   1993 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   1994 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   1995 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   1996 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   1997 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   1998 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   2283 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   2284 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   2285 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   2286 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   2287 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   2288 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   2289 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   2290 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:01986] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:01986] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    62.36    0.02    0.03       0    56.79    0.02    0.03       0
        2048           512     float     sum      -1    60.18    0.03    0.06       0    59.56    0.03    0.06       0
        4096          1024     float     sum      -1    67.49    0.06    0.11       0    66.73    0.06    0.12       0
        8192          2048     float     sum      -1    75.59    0.11    0.20       0    69.68    0.12    0.22       0
       16384          4096     float     sum      -1    81.24    0.20    0.38       0    71.26    0.23    0.43       0
       32768          8192     float     sum      -1    90.73    0.36    0.68       0    88.93    0.37    0.69       0
       65536         16384     float     sum      -1   113.48    0.58    1.08       0   112.11    0.58    1.10       0
      131072         32768     float     sum      -1   115.48    1.14    2.13       0   115.22    1.14    2.13       0
      262144         65536     float     sum      -1   148.33    1.77    3.31       0   152.84    1.72    3.22       0
      524288        131072     float     sum      -1   202.31    2.59    4.86       0   201.79    2.60    4.87       0
     1048576        262144     float     sum      -1   295.88    3.54    6.64       0   289.05    3.63    6.80       0
     2097152        524288     float     sum      -1   459.39    4.57    8.56       0   472.46    4.44    8.32       0
     4194304       1048576     float     sum      -1   806.82    5.20    9.75       0   767.97    5.46   10.24       0
     8388608       2097152     float     sum      -1   530.33   15.82   29.66       0   529.36   15.85   29.71       0
    16777216       4194304     float     sum      -1   839.88   19.98   37.45       0   851.57   19.70   36.94       0
    33554432       8388608     float     sum      -1  1706.35   19.66   36.87       0  1702.97   19.70   36.94       0
    67108864      16777216     float     sum      -1  3331.20   20.15   37.77       0  3310.43   20.27   38.01       0
   134217728      33554432     float     sum      -1  5879.05   22.83   42.81       0  5882.79   22.82   42.78       0
   268435456      67108864     float     sum      -1  12263.2   21.89   41.04       0  12174.5   22.05   41.34       0
   536870912     134217728     float     sum      -1  26917.4   19.95   37.40       0  26678.6   20.12   37.73       0
  1073741824     268435456     float     sum      -1  53150.1   20.20   37.88       0  53197.4   20.18   37.85       0
  2147483648     536870912     float     sum      -1   106197   20.22   37.92       0   106098   20.24   37.95       0
  4294967296    1073741824     float     sum      -1   211677   20.29   38.04       0   211916   20.27   38.00       0
  8589934592    2147483648     float     sum      -1   425662   20.18   37.84       0   423787   20.27   38.01       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.8746 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:09:33+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   2374 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   2375 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   2376 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   2377 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   2378 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   2379 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   2381 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   2382 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   2695 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   2696 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   2697 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   2698 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   2699 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   2700 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   2701 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   2702 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:02369] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:02369] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    62.15    0.02    0.03       0    57.80    0.02    0.03       0
        2048           512     float     sum      -1    59.94    0.03    0.06       0    58.83    0.03    0.07       0
        4096          1024     float     sum      -1    64.36    0.06    0.12       0    64.24    0.06    0.12       0
        8192          2048     float     sum      -1    74.69    0.11    0.21       0    68.55    0.12    0.22       0
       16384          4096     float     sum      -1    85.78    0.19    0.36       0    70.11    0.23    0.44       0
       32768          8192     float     sum      -1    86.92    0.38    0.71       0    85.44    0.38    0.72       0
       65536         16384     float     sum      -1   109.69    0.60    1.12       0   108.19    0.61    1.14       0
      131072         32768     float     sum      -1   100.09    1.31    2.46       0   113.92    1.15    2.16       0
      262144         65536     float     sum      -1   112.84    2.32    4.36       0   116.33    2.25    4.23       0
      524288        131072     float     sum      -1   141.79    3.70    6.93       0   141.66    3.70    6.94       0
     1048576        262144     float     sum      -1   227.16    4.62    8.66       0   228.42    4.59    8.61       0
     2097152        524288     float     sum      -1   380.58    5.51   10.33       0   382.44    5.48   10.28       0
     4194304       1048576     float     sum      -1   688.55    6.09   11.42       0   669.10    6.27   11.75       0
     8388608       2097152     float     sum      -1   484.30   17.32   32.48       0   484.69   17.31   32.45       0
    16777216       4194304     float     sum      -1   838.66   20.00   37.51       0   815.17   20.58   38.59       0
    33554432       8388608     float     sum      -1  1694.67   19.80   37.12       0  1711.52   19.61   36.76       0
    67108864      16777216     float     sum      -1  3334.85   20.12   37.73       0  3383.05   19.84   37.19       0
   134217728      33554432     float     sum      -1  5997.33   22.38   41.96       0  5961.18   22.52   42.22       0
   268435456      67108864     float     sum      -1  12585.0   21.33   39.99       0  12694.3   21.15   39.65       0
   536870912     134217728     float     sum      -1  27083.3   19.82   37.17       0  27228.0   19.72   36.97       0
  1073741824     268435456     float     sum      -1  54263.7   19.79   37.10       0  54815.5   19.59   36.73       0
  2147483648     536870912     float     sum      -1   108741   19.75   37.03       0   109301   19.65   36.84       0
  4294967296    1073741824     float     sum      -1   217041   19.79   37.10       0   217697   19.73   36.99       0
  8589934592    2147483648     float     sum      -1   435250   19.74   37.00       0   435909   19.71   36.95       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 19.1041 
#
# Collective test concluded: all_reduce_perf
#
```
