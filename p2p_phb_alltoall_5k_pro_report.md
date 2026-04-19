# NCCL AllToAll 性能测试报告（P2P = PHB）

**报告主题：** `alltoall_perf`，跨两台 **5K Pro** GPU 服务器，在 **`NCCL_P2P_LEVEL=PHB`** 前提下对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 三种配置的 **out-of-place algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志节点为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`；拓扑按测试计划记为两台 **5K Pro**。

---

## 1. 测试命令

三次运行除 NCCL IB 开关外保持一致；共性：`NCCL_SHM_DISABLE=1`、`UCX_TLS=ib`、**`NCCL_P2P_LEVEL=PHB`**、`alltoall_perf -b 1k -e 8G -f 2 -g 1`。

### 1.1 AR OFF（基准）

```bash
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=0 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 AR ON / DDP OFF

```bash
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 AR ON / DDP ON

```bash
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

完整 `mpirun` 单行命令见第 4 节「原始数据」各 CASE 的 `COMMAND:` 行。

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台 **5K Pro** GPU 服务器（日志：`R6KD-CX8aaS-GPU-14`、`R6KD-CX8aaS-GPU-15`） |
| 进程/GPU | 每节点 8 进程，共 **16 GPU**；`alltoall_perf` **每 rank 1 GPU**（`-g 1`） |
| 集合通信 | **alltoall_perf** |
| P2P | **`NCCL_P2P_LEVEL=PHB`** |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*`） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllToAll**；跨机经 **bond0 / IB**。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.03 | 0.02 | −33.3% | 0.02 | −33.3% |
| 2048 | 0.06 | 0.05 | −16.7% | 0.06 | 0.0% |
| 4096 | 0.12 | 0.12 | 0.0% | 0.12 | 0.0% |
| 8192 | 0.25 | 0.24 | −4.0% | 0.25 | 0.0% |
| 16384 | 0.49 | 0.48 | −2.0% | 0.48 | −2.0% |
| 32768 | 0.94 | 0.92 | −2.1% | 0.92 | −2.1% |
| 65536 | 1.82 | 1.76 | −3.3% | 1.24 | −31.9% |
| 131072 | 3.32 | 3.34 | +0.6% | 3.34 | +0.6% |
| 262144 | 5.71 | 4.63 | −18.9% | 6.15 | +7.7% |
| 524288 | 10.31 | 9.32 | −9.6% | 8.95 | −13.2% |
| 1048576 | 14.84 | 12.42 | −16.3% | 12.70 | −14.4% |
| 2097152 | 23.31 | 20.34 | −12.7% | 23.25 | −0.3% |
| 4194304 | 25.73 | 21.89 | −14.9% | 24.64 | −4.2% |
| 8388608 | 32.31 | 28.44 | −12.0% | 31.01 | −4.0% |
| 16777216 | 39.06 | 35.71 | −8.6% | 38.99 | −0.2% |
| 33554432 | 42.39 | 16.61 | −60.8% | 42.62 | +0.5% |
| 67108864 | 43.81 | 31.17 | −28.9% | 26.12 | −40.4% |
| 134217728 | 44.32 | 43.12 | −2.7% | 44.51 | +0.4% |
| 268435456 | 44.02 | 23.35 | −47.0% | 27.42 | −37.7% |
| 536870912 | 44.16 | 21.97 | −50.2% | 20.37 | −53.9% |
| 1073741824 | 42.72 | 27.28 | −36.2% | 26.21 | −38.7% |
| 2147483648 | 43.77 | 31.61 | −27.8% | 28.17 | −35.6% |
| 4294967296 | 43.26 | 38.43 | −11.2% | 35.06 | −19.0% |
| 8589934592 | 43.16 | 40.87 | −5.3% | 41.40 | −4.1% |

**摘要观察（out-of-place algbw）：**

- **AR ON / DDP OFF** 在 **32 MiB** 出现 **约 −61%** 的断崖式回落，**256 MiB–1 GiB** 亦显著低于基准；**128 MiB** 与基准接近（约 **−3%**）；**2 GiB–8 GiB** 为温和负 Delta。
- **AR ON / DDP ON** 在 **64 MiB、64 MiB–128 MiB 区间的一侧（64 MiB）** 明显偏弱（**−32%**）；**256 MiB–1 GiB** 同样大幅低于基准；**128 MiB、32 MiB** 与基准基本持平或略优。
- **小消息**（**1 KiB**）三种配置打印精度下 algbw 均为 **0.02–0.03 GB/s**，百分比波动大，宜结合 **time (us)** 与多次重复实验解读。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:02:38+00:00
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
[R6KD-CX8aaS-GPU-14:10724] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:10724] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  10729 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  10730 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  10731 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  10732 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  10733 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  10734 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  10735 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  10736 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  12555 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  12556 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  12557 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  12558 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  12559 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  12560 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  12561 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  12562 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    38.67    0.03    0.02       0    32.09    0.03    0.03    N/A
        2048            32     float    none      -1    35.10    0.06    0.05       0    32.68    0.06    0.06    N/A
        4096            64     float    none      -1    33.39    0.12    0.11       0    32.01    0.13    0.12    N/A
        8192           128     float    none      -1    32.56    0.25    0.24       0    32.04    0.26    0.24    N/A
       16384           256     float    none      -1    33.52    0.49    0.46       0    32.87    0.50    0.47    N/A
       32768           512     float    none      -1    34.78    0.94    0.88       0    34.22    0.96    0.90    N/A
       65536          1024     float    none      -1    36.06    1.82    1.70       0    35.81    1.83    1.72    N/A
      131072          2048     float    none      -1    39.52    3.32    3.11       0    38.44    3.41    3.20    N/A
      262144          4096     float    none      -1    45.91    5.71    5.35       0    52.31    5.01    4.70    N/A
      524288          8192     float    none      -1    50.87   10.31    9.66       0    55.69    9.41    8.83    N/A
     1048576         16384     float    none      -1    70.67   14.84   13.91       0    86.00   12.19   11.43    N/A
     2097152         32768     float    none      -1    89.98   23.31   21.85       0    92.51   22.67   21.25    N/A
     4194304         65536     float    none      -1   163.00   25.73   24.12       0   156.85   26.74   25.07    N/A
     8388608        131072     float    none      -1   259.65   32.31   30.29       0   266.63   31.46   29.50    N/A
    16777216        262144     float    none      -1   429.55   39.06   36.62       0   424.39   39.53   37.06    N/A
    33554432        524288     float    none      -1   791.62   42.39   39.74       0   800.92   41.89   39.28    N/A
    67108864       1048576     float    none      -1  1531.66   43.81   41.08       0  2802.24   23.95   22.45    N/A
   134217728       2097152     float    none      -1  3028.55   44.32   41.55       0  3017.07   44.49   41.71    N/A
   268435456       4194304     float    none      -1  6097.84   44.02   41.27       0  6031.64   44.50   41.72    N/A
   536870912       8388608     float    none      -1  12157.7   44.16   41.40       0  12012.3   44.69   41.90    N/A
  1073741824      16777216     float    none      -1  25131.5   42.72   40.05       0  25868.5   41.51   38.91    N/A
  2147483648      33554432     float    none      -1  49058.7   43.77   41.04       0  51164.6   41.97   39.35    N/A
  4294967296      67108864     float    none      -1  99284.8   43.26   40.56       0   103946   41.32   38.74    N/A
  8589934592     134217728     float    none      -1   199011   43.16   40.47       0   193528   44.39   41.61    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.9533 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:04:28+00:00
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
[R6KD-CX8aaS-GPU-14:11058] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:11058] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  11063 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  11064 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  11065 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  11066 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  11067 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  11068 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  11069 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  11070 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  12965 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  12966 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  12967 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  12968 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  12969 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  12970 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  12971 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  12972 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    56.53    0.02    0.02       0    32.72    0.03    0.03    N/A
        2048            32     float    none      -1    37.25    0.05    0.05       0    34.03    0.06    0.06    N/A
        4096            64     float    none      -1    34.19    0.12    0.11       0    33.45    0.12    0.11    N/A
        8192           128     float    none      -1    33.73    0.24    0.23       0    32.95    0.25    0.23    N/A
       16384           256     float    none      -1    34.43    0.48    0.45       0    33.62    0.49    0.46    N/A
       32768           512     float    none      -1    35.69    0.92    0.86       0    36.09    0.91    0.85    N/A
       65536          1024     float    none      -1    37.22    1.76    1.65       0    37.15    1.76    1.65    N/A
      131072          2048     float    none      -1    39.30    3.34    3.13       0    40.33    3.25    3.05    N/A
      262144          4096     float    none      -1    56.59    4.63    4.34       0    55.98    4.68    4.39    N/A
      524288          8192     float    none      -1    56.24    9.32    8.74       0    59.80    8.77    8.22    N/A
     1048576         16384     float    none      -1    84.43   12.42   11.64       0    93.28   11.24   10.54    N/A
     2097152         32768     float    none      -1   103.11   20.34   19.07       0   105.04   19.97   18.72    N/A
     4194304         65536     float    none      -1   191.62   21.89   20.52       0   208.26   20.14   18.88    N/A
     8388608        131072     float    none      -1   295.01   28.44   26.66       0   290.57   28.87   27.06    N/A
    16777216        262144     float    none      -1   469.84   35.71   33.48       0  4682.84    3.58    3.36    N/A
    33554432        524288     float    none      -1  2020.19   16.61   15.57       0   847.67   39.58   37.11    N/A
    67108864       1048576     float    none      -1  2153.30   31.17   29.22       0  1593.06   42.13   39.49    N/A
   134217728       2097152     float    none      -1  3112.98   43.12   40.42       0  3552.87   37.78   35.42    N/A
   268435456       4194304     float    none      -1  11496.3   23.35   21.89       0  10216.6   26.27   24.63    N/A
   536870912       8388608     float    none      -1  24442.0   21.97   20.59       0  25663.6   20.92   19.61    N/A
  1073741824      16777216     float    none      -1  39365.2   27.28   25.57       0  36628.1   29.31   27.48    N/A
  2147483648      33554432     float    none      -1  67941.7   31.61   29.63       0  79182.8   27.12   25.43    N/A
  4294967296      67108864     float    none      -1   111774   38.43   36.02       0   128464   33.43   31.34    N/A
  8589934592     134217728     float    none      -1   210196   40.87   38.31       0   201466   42.64   39.97    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 15.9641 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:06:23+00:00
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
[R6KD-CX8aaS-GPU-14:11392] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:11392] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  11397 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  11398 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  11399 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  11400 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  11401 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  11402 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  11403 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  11404 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  13375 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  13376 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  13377 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  13378 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  13379 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  13380 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  13381 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  13382 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    51.46    0.02    0.02       0    32.61    0.03    0.03    N/A
        2048            32     float    none      -1    33.06    0.06    0.06       0    32.84    0.06    0.06    N/A
        4096            64     float    none      -1    33.39    0.12    0.11       0    32.56    0.13    0.12    N/A
        8192           128     float    none      -1    33.20    0.25    0.23       0    32.66    0.25    0.24    N/A
       16384           256     float    none      -1    34.40    0.48    0.45       0    33.42    0.49    0.46    N/A
       32768           512     float    none      -1    35.54    0.92    0.86       0    35.33    0.93    0.87    N/A
       65536          1024     float    none      -1    52.99    1.24    1.16       0    36.51    1.80    1.68    N/A
      131072          2048     float    none      -1    39.19    3.34    3.14       0    39.14    3.35    3.14    N/A
      262144          4096     float    none      -1    42.62    6.15    5.77       0    50.19    5.22    4.90    N/A
      524288          8192     float    none      -1    58.61    8.95    8.39       0    51.73   10.14    9.50    N/A
     1048576         16384     float    none      -1    82.56   12.70   11.91       0    78.22   13.40   12.57    N/A
     2097152         32768     float    none      -1    90.19   23.25   21.80       0  4781.60    0.44    0.41    N/A
     4194304         65536     float    none      -1   170.24   24.64   23.10       0   156.36   26.82   25.15    N/A
     8388608        131072     float    none      -1   270.49   31.01   29.07       0   267.65   31.34   29.38    N/A
    16777216        262144     float    none      -1   430.25   38.99   36.56       0   424.95   39.48   37.01    N/A
    33554432        524288     float    none      -1   787.30   42.62   39.96       0   797.55   42.07   39.44    N/A
    67108864       1048576     float    none      -1  2569.22   26.12   24.49       0  1529.98   43.86   41.12    N/A
   134217728       2097152     float    none      -1  3015.34   44.51   41.73       0  9753.98   13.76   12.90    N/A
   268435456       4194304     float    none      -1  9791.01   27.42   25.70       0  8767.30   30.62   28.70    N/A
   536870912       8388608     float    none      -1  26350.1   20.37   19.10       0  21515.3   24.95   23.39    N/A
  1073741824      16777216     float    none      -1  40973.2   26.21   24.57       0  44665.7   24.04   22.54    N/A
  2147483648      33554432     float    none      -1  76237.6   28.17   26.41       0  65473.0   32.80   30.75    N/A
  4294967296      67108864     float    none      -1   122511   35.06   32.87       0   107108   40.10   37.59    N/A
  8589934592     134217728     float    none      -1   207466   41.40   38.82       0   200355   42.87   40.19    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.05 
#
# Collective test concluded: alltoall_perf
#
```

---

