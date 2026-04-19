# 6KD NCCL `alltoall_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_DISABLE=1`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `alltoall_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `alltoall_perf` 参数保持一致。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `alltoall_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| 网络 | UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备；`NCCL_SHM_DISABLE=1`；**`NCCL_P2P_DISABLE=1`**（禁用 GPU P2P） |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.02 | 0.02 | 0.00% | 0.02 | 0.00% |
| 2048 | 0.04 | 0.04 | 0.00% | 0.04 | 0.00% |
| 4096 | 0.09 | 0.08 | -11.11% | 0.08 | -11.11% |
| 8192 | 0.17 | 0.17 | 0.00% | 0.16 | -5.88% |
| 16384 | 0.34 | 0.34 | 0.00% | 0.33 | -2.94% |
| 32768 | 0.68 | 0.68 | 0.00% | 0.65 | -4.41% |
| 65536 | 1.31 | 1.35 | +3.05% | 1.31 | 0.00% |
| 131072 | 2.53 | 2.60 | +2.77% | 2.53 | 0.00% |
| 262144 | 4.14 | 3.84 | -7.25% | 4.80 | +15.94% |
| 524288 | 8.33 | 5.84 | -29.89% | 6.47 | -22.33% |
| 1048576 | 10.32 | 8.66 | -16.09% | 10.12 | -1.94% |
| 2097152 | 17.80 | 16.63 | -6.57% | 18.05 | +1.40% |
| 4194304 | 24.95 | 21.72 | -12.95% | 24.79 | -0.64% |
| 8388608 | 31.26 | 29.56 | -5.44% | 32.05 | +2.53% |
| 16777216 | 37.73 | 34.60 | -8.30% | 38.02 | +0.77% |
| 33554432 | 44.04 | 42.17 | -4.25% | 41.54 | -5.68% |
| 67108864 | 47.37 | 46.11 | -2.66% | 47.23 | -0.30% |
| 134217728 | 47.98 | 47.71 | -0.56% | 47.99 | +0.02% |
| 268435456 | 48.33 | 48.74 | +0.85% | 47.46 | -1.80% |
| 536870912 | 48.81 | 49.32 | +1.05% | 48.26 | -1.13% |
| 1073741824 | 48.86 | 49.53 | +1.37% | 48.21 | -1.33% |
| 2147483648 | 48.99 | 49.60 | +1.24% | 48.09 | -1.84% |
| 4294967296 | 48.70 | 49.64 | +1.93% | 48.15 | -1.13% |
| 8589934592 | 48.76 | 49.68 | +1.89% | 48.09 | -1.37% |

**简要观察（out-of-place algbw）**

- **中消息**（约 256KB–1MB）：相对 AR OFF，**AR ON DDP OFF** 在 **512KB** 附近出现显著回落（约 **-30%**）；**AR ON DDP ON** 在该区间同样偏慢，但在 **256KB** 步长上反而明显高于基准（约 **+16%**），呈现强烈的尺寸依赖。
- **大消息**（约 32MB 及以上）：三种配置整体接近 **48–50 GB/s** 量级；**AR ON DDP OFF** 在多数最大步长上略 **快于** 基准（约 **+1%–+2%**）；**AR ON DDP ON** 在 **256MB–8GB** 区间相对基准略 **慢**（约 **-1% 至 -2%**），与 **AR ON DDP OFF** 的尾部趋势不同。

---

## 4. 原始数据附录

以下为三次运行日志中的完整相关段落（含设备枚举、表头与数据行），便于复核。

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:07:52+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   1823 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   1824 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   1825 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   1826 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   1827 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   1828 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   1829 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   1830 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   2077 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   2078 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   2079 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   2080 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   2081 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   2082 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   2083 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   2084 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:01818] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:01818] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    58.93    0.02    0.02       0    46.91    0.02    0.02    N/A
        2048            32     float    none      -1    46.73    0.04    0.04       0    47.56    0.04    0.04    N/A
        4096            64     float    none      -1    47.82    0.09    0.08       0    49.05    0.08    0.08    N/A
        8192           128     float    none      -1    47.41    0.17    0.16       0    47.65    0.17    0.16    N/A
       16384           256     float    none      -1    47.87    0.34    0.32       0    47.47    0.35    0.32    N/A
       32768           512     float    none      -1    48.46    0.68    0.63       0    48.05    0.68    0.64    N/A
       65536          1024     float    none      -1    50.09    1.31    1.23       0    48.35    1.36    1.27    N/A
      131072          2048     float    none      -1    51.84    2.53    2.37       0    50.46    2.60    2.44    N/A
      262144          4096     float    none      -1    63.30    4.14    3.88       0    68.42    3.83    3.59    N/A
      524288          8192     float    none      -1    62.91    8.33    7.81       0    63.57    8.25    7.73    N/A
     1048576         16384     float    none      -1   101.56   10.32    9.68       0   102.42   10.24    9.60    N/A
     2097152         32768     float    none      -1   117.80   17.80   16.69       0   102.11   20.54   19.25    N/A
     4194304         65536     float    none      -1   168.12   24.95   23.39       0   163.58   25.64   24.04    N/A
     8388608        131072     float    none      -1   268.37   31.26   29.30       0   262.04   32.01   30.01    N/A
    16777216        262144     float    none      -1   444.61   37.73   35.38       0   438.00   38.30   35.91    N/A
    33554432        524288     float    none      -1   761.92   44.04   41.29       0   773.25   43.39   40.68    N/A
    67108864       1048576     float    none      -1  1416.60   47.37   44.41       0  1394.85   48.11   45.10    N/A
   134217728       2097152     float    none      -1  2797.44   47.98   44.98       0  2820.18   47.59   44.62    N/A
   268435456       4194304     float    none      -1  5554.51   48.33   45.31       0  5637.83   47.61   44.64    N/A
   536870912       8388608     float    none      -1  10999.3   48.81   45.76       0  11242.1   47.76   44.77    N/A
  1073741824      16777216     float    none      -1  21976.7   48.86   45.80       0  22419.3   47.89   44.90    N/A
  2147483648      33554432     float    none      -1  43836.4   48.99   45.93       0  44781.4   47.95   44.96    N/A
  4294967296      67108864     float    none      -1  88196.3   48.70   45.65       0  89491.1   47.99   44.99    N/A
  8589934592     134217728     float    none      -1   176166   48.76   45.71       0   178913   48.01   45.01    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.3044 
#
# Collective test concluded: alltoall_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:09:07+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   2206 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   2207 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   2208 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   2209 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   2210 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   2211 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   2212 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   2213 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   2489 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   2490 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   2491 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   2492 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   2493 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   2494 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   2496 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   2498 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:02201] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:02201] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    60.27    0.02    0.02       0    46.79    0.02    0.02    N/A
        2048            32     float    none      -1    47.59    0.04    0.04       0    47.00    0.04    0.04    N/A
        4096            64     float    none      -1    48.20    0.08    0.08       0    46.98    0.09    0.08    N/A
        8192           128     float    none      -1    47.62    0.17    0.16       0    47.33    0.17    0.16    N/A
       16384           256     float    none      -1    48.04    0.34    0.32       0    47.59    0.34    0.32    N/A
       32768           512     float    none      -1    48.53    0.68    0.63       0    47.66    0.69    0.64    N/A
       65536          1024     float    none      -1    48.49    1.35    1.27       0    48.55    1.35    1.27    N/A
      131072          2048     float    none      -1    50.40    2.60    2.44       0    49.71    2.64    2.47    N/A
      262144          4096     float    none      -1    68.24    3.84    3.60       0    70.90    3.70    3.47    N/A
      524288          8192     float    none      -1    89.82    5.84    5.47       0    75.46    6.95    6.51    N/A
     1048576         16384     float    none      -1   121.14    8.66    8.11       0   121.11    8.66    8.12    N/A
     2097152         32768     float    none      -1   126.09   16.63   15.59       0   123.92   16.92   15.87    N/A
     4194304         65536     float    none      -1   193.11   21.72   20.36       0   211.76   19.81   18.57    N/A
     8388608        131072     float    none      -1   283.74   29.56   27.72       0   310.59   27.01   25.32    N/A
    16777216        262144     float    none      -1   484.85   34.60   32.44       0   473.66   35.42   33.21    N/A
    33554432        524288     float    none      -1   795.70   42.17   39.53       0   771.68   43.48   40.76    N/A
    67108864       1048576     float    none      -1  1455.51   46.11   43.23       0  1456.35   46.08   43.20    N/A
   134217728       2097152     float    none      -1  2813.14   47.71   44.73       0  2834.97   47.34   44.38    N/A
   268435456       4194304     float    none      -1  5507.69   48.74   45.69       0  5583.68   48.08   45.07    N/A
   536870912       8388608     float    none      -1  10885.0   49.32   46.24       0  11001.9   48.80   45.75    N/A
  1073741824      16777216     float    none      -1  21677.5   49.53   46.44       0  21950.9   48.92   45.86    N/A
  2147483648      33554432     float    none      -1  43294.7   49.60   46.50       0  43754.4   49.08   46.01    N/A
  4294967296      67108864     float    none      -1  86517.5   49.64   46.54       0  87429.9   49.12   46.05    N/A
  8589934592     134217728     float    none      -1   172920   49.68   46.57       0   174658   49.18   46.11    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.7291 
#
# Collective test concluded: alltoall_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:10:22+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1

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
#  Rank  0 Group  0 Pid   2542 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   2543 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   2544 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   2545 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   2546 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   2547 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   2548 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   2549 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   2901 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   2902 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   2903 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   2904 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   2905 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   2906 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   2907 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   2908 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:02537] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:02537] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024            16     float    none      -1    62.75    0.02    0.02       0    49.39    0.02    0.02    N/A
        2048            32     float    none      -1    49.15    0.04    0.04       0    49.33    0.04    0.04    N/A
        4096            64     float    none      -1    49.28    0.08    0.08       0    49.07    0.08    0.08    N/A
        8192           128     float    none      -1    49.89    0.16    0.15       0    49.12    0.17    0.16    N/A
       16384           256     float    none      -1    49.63    0.33    0.31       0    49.48    0.33    0.31    N/A
       32768           512     float    none      -1    50.32    0.65    0.61       0    50.83    0.64    0.60    N/A
       65536          1024     float    none      -1    50.16    1.31    1.22       0    50.19    1.31    1.22    N/A
      131072          2048     float    none      -1    51.74    2.53    2.38       0    51.36    2.55    2.39    N/A
      262144          4096     float    none      -1    54.63    4.80    4.50       0    61.11    4.29    4.02    N/A
      524288          8192     float    none      -1    80.98    6.47    6.07       0    80.85    6.48    6.08    N/A
     1048576         16384     float    none      -1   103.63   10.12    9.49       0    97.43   10.76   10.09    N/A
     2097152         32768     float    none      -1   116.19   18.05   16.92       0   116.08   18.07   16.94    N/A
     4194304         65536     float    none      -1   169.18   24.79   23.24       0   216.20   19.40   18.19    N/A
     8388608        131072     float    none      -1   261.73   32.05   30.05       0   253.30   33.12   31.05    N/A
    16777216        262144     float    none      -1   441.27   38.02   35.64       0   465.15   36.07   33.81    N/A
    33554432        524288     float    none      -1   807.67   41.54   38.95       0   742.11   45.21   42.39    N/A
    67108864       1048576     float    none      -1  1421.02   47.23   44.27       0  1395.11   48.10   45.10    N/A
   134217728       2097152     float    none      -1  2796.77   47.99   44.99       0  2828.53   47.45   44.49    N/A
   268435456       4194304     float    none      -1  5655.60   47.46   44.50       0  5644.22   47.56   44.59    N/A
   536870912       8388608     float    none      -1  11123.8   48.26   45.25       0  11409.6   47.05   44.11    N/A
  1073741824      16777216     float    none      -1  22273.2   48.21   45.19       0  22703.6   47.29   44.34    N/A
  2147483648      33554432     float    none      -1  44652.3   48.09   45.09       0  45342.5   47.36   44.40    N/A
  4294967296      67108864     float    none      -1  89203.8   48.15   45.14       0  90551.3   47.43   44.47    N/A
  8589934592     134217728     float    none      -1   178626   48.09   45.08       0   181120   47.43   44.46    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.9275 
#
# Collective test concluded: alltoall_perf
#
```
