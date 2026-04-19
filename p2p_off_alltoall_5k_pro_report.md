# NCCL AllToAll 性能测试报告（P2P 关闭）

**报告主题：** `alltoall_perf`，跨两台 **5K Pro** GPU 服务器，对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 三种配置下 **out-of-place** 的 **algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志中节点标识为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`，此处环境与拓扑按测试计划记为两台 **5K Pro**；若与资产命名不一致，请以机房/CMDB 记录为准。

---

## 1. 测试命令

三条用例除 NCCL IB 相关开关外保持一致；共性：`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`、`UCX_TLS=ib`、`alltoall_perf -b 1k -e 8G -f 2 -g 1`。

### 1.1 AR OFF（基准）

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=0 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 AR ON / DDP OFF

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 AR ON / DDP ON

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

完整 `mpirun` 单行命令见文末「原始数据」各 CASE 的 `COMMAND:` 行。

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台 **5K Pro** GPU 服务器（日志主机名：`R6KD-CX8aaS-GPU-14`、`R6KD-CX8aaS-GPU-15`） |
| 进程/GPU | 每节点 8 进程，共 **16 GPU**；`alltoall_perf` **每 rank 1 GPU**（`-g 1`） |
| 集合通信 | **alltoall_perf**（NCCL collective benchmark） |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*` 列表） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |
| P2P | `NCCL_P2P_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllToAll**；跨机流量走 **bond0 / IB**。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。极小数值行的百分比仅作形式参考。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.02 | 0.02 | 0.0% | 0.02 | 0.0% |
| 2048 | 0.04 | 0.04 | 0.0% | 0.04 | 0.0% |
| 4096 | 0.08 | 0.08 | 0.0% | 0.08 | 0.0% |
| 8192 | 0.16 | 0.17 | +6.3% | 0.17 | +6.3% |
| 16384 | 0.32 | 0.33 | +3.1% | 0.30 | −6.3% |
| 32768 | 0.64 | 0.65 | +1.6% | 0.58 | −9.4% |
| 65536 | 1.25 | 1.30 | +4.0% | 1.30 | +4.0% |
| 131072 | 2.42 | 2.53 | +4.5% | 2.19 | −9.5% |
| 262144 | 4.07 | 3.79 | −6.9% | 3.81 | −6.4% |
| 524288 | 6.61 | 6.71 | +1.5% | 7.99 | +20.9% |
| 1048576 | 11.01 | 9.05 | −17.8% | 11.52 | +4.6% |
| 2097152 | 17.38 | 16.53 | −4.9% | 20.35 | +17.1% |
| 4194304 | 23.22 | 21.49 | −7.5% | 24.04 | +3.5% |
| 8388608 | 11.51 | 29.31 | +154.6% | 32.95 | +186.3% |
| 16777216 | 36.96 | 34.49 | −6.7% | 37.90 | +2.5% |
| 33554432 | 43.65 | 42.20 | −3.3% | 6.46 | −85.2% |
| 67108864 | 47.33 | 34.51 | −27.1% | 18.39 | −61.1% |
| 134217728 | 48.01 | 8.69 | −81.9% | 36.80 | −23.4% |
| 268435456 | 48.55 | 19.37 | −60.1% | 14.36 | −70.4% |
| 536870912 | 48.85 | 28.48 | −41.7% | 23.25 | −52.4% |
| 1073741824 | 42.77 | 28.41 | −33.6% | 27.54 | −35.6% |
| 2147483648 | 48.39 | 25.11 | −48.1% | 30.20 | −37.6% |
| 4294967296 | 47.61 | 34.03 | −28.5% | 36.48 | −23.4% |
| 8589934592 | 47.58 | 42.82 | −10.0% | 43.70 | −8.2% |

**摘要观察（out-of-place algbw）：**

- **中小消息**（约 **1 MiB–4 MiB**）上，**AR ON / DDP ON** 在 **2 MiB** 相对基准有明显 **提升**（约 **+17%**）；**1 MiB** 在 **AR ON / DDP OFF** 上相对基准偏弱（约 **−18%**）。
- **8 MiB**：基准 **11.51 GB/s** 相对前后尺寸偏低，疑似单点测量噪声；两种 AR 配置均显著更高（表中表现为大幅正 Delta），建议 **复测** 该点并结合 `NCCL_DEBUG`/拓扑确认。
- **大体量**（约 **32 MiB–512 MiB** 及 **1 GiB**）：**AR ON / DDP OFF** 与 **AR ON / DDP ON** 均出现 **大幅低于基准** 的区段（例如 **128 MiB** 的 DDP OFF 约 **−82%**；**32 MiB** 的 DDP ON 约 **−85%**），适合作为后续调参与稳定性排查的重点区间。
- **8 GiB** 附近：两种 AR 配置略低于基准（约 **−8%～−10%**），差距相对温和。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:08:26+00:00
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
[R6KD-CX8aaS-GPU-14:11728] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:11728] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  11733 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  11734 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  11735 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  11736 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  11737 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  11738 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  11739 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  11740 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  13787 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  13788 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  13789 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  13790 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  13791 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  13792 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  13793 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  13794 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    65.96    0.02    0.01       0    50.54    0.02    0.02    N/A
        2048            32     float    none      -1    51.75    0.04    0.04       0    50.86    0.04    0.04    N/A
        4096            64     float    none      -1    51.26    0.08    0.07       0    50.59    0.08    0.08    N/A
        8192           128     float    none      -1    50.91    0.16    0.15       0    50.79    0.16    0.15    N/A
       16384           256     float    none      -1    51.38    0.32    0.30       0    50.96    0.32    0.30    N/A
       32768           512     float    none      -1    51.59    0.64    0.60       0    51.44    0.64    0.60    N/A
       65536          1024     float    none      -1    52.47    1.25    1.17       0    51.84    1.26    1.19    N/A
      131072          2048     float    none      -1    54.08    2.42    2.27       0    54.51    2.40    2.25    N/A
      262144          4096     float    none      -1    64.40    4.07    3.82       0    71.07    3.69    3.46    N/A
      524288          8192     float    none      -1    79.36    6.61    6.19       0    64.77    8.09    7.59    N/A
     1048576         16384     float    none      -1    95.27   11.01   10.32       0    98.47   10.65    9.98    N/A
     2097152         32768     float    none      -1   120.65   17.38   16.30       0   104.48   20.07   18.82    N/A
     4194304         65536     float    none      -1   180.63   23.22   21.77       0   169.80   24.70   23.16    N/A
     8388608        131072     float    none      -1   728.58   11.51   10.79       0   307.53   27.28   25.57    N/A
    16777216        262144     float    none      -1   453.90   36.96   34.65       0   444.36   37.76   35.40    N/A
    33554432        524288     float    none      -1   768.74   43.65   40.92       0   740.32   45.32   42.49    N/A
    67108864       1048576     float    none      -1  1417.86   47.33   44.37       0  1405.20   47.76   44.77    N/A
   134217728       2097152     float    none      -1  2795.50   48.01   45.01       0  2800.49   47.93   44.93    N/A
   268435456       4194304     float    none      -1  5529.44   48.55   45.51       0  5609.64   47.85   44.86    N/A
   536870912       8388608     float    none      -1  10990.4   48.85   45.80       0  11122.5   48.27   45.25    N/A
  1073741824      16777216     float    none      -1  25107.1   42.77   40.09       0  22198.8   48.37   45.35    N/A
  2147483648      33554432     float    none      -1  44383.2   48.39   45.36       0  45630.7   47.06   44.12    N/A
  4294967296      67108864     float    none      -1  90216.6   47.61   44.63       0  96573.2   44.47   41.69    N/A
  8589934592     134217728     float    none      -1   180556   47.58   44.60       0   178782   48.05   45.04    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.4972 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:10:24+00:00
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
[R6KD-CX8aaS-GPU-14:12064] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:12064] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12069 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  12070 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  12071 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  12072 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  12073 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  12074 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  12075 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  12076 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  14199 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  14200 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  14201 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  14202 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  14203 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  14204 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  14205 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  14206 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    62.62    0.02    0.02       0    49.46    0.02    0.02    N/A
        2048            32     float    none      -1    48.65    0.04    0.04       0    48.61    0.04    0.04    N/A
        4096            64     float    none      -1    49.53    0.08    0.08       0    49.36    0.08    0.08    N/A
        8192           128     float    none      -1    48.97    0.17    0.16       0    49.64    0.17    0.15    N/A
       16384           256     float    none      -1    49.95    0.33    0.31       0    49.92    0.33    0.31    N/A
       32768           512     float    none      -1    50.05    0.65    0.61       0   127.53    0.26    0.24    N/A
       65536          1024     float    none      -1    50.25    1.30    1.22       0    55.12    1.19    1.11    N/A
      131072          2048     float    none      -1    51.78    2.53    2.37       0    52.96    2.47    2.32    N/A
      262144          4096     float    none      -1    69.24    3.79    3.55       0    86.28    3.04    2.85    N/A
      524288          8192     float    none      -1    78.14    6.71    6.29       0    77.17    6.79    6.37    N/A
     1048576         16384     float    none      -1   115.84    9.05    8.49       0   109.65    9.56    8.97    N/A
     2097152         32768     float    none      -1   126.86   16.53   15.50       0   124.08   16.90   15.85    N/A
     4194304         65536     float    none      -1   195.14   21.49   20.15       0   187.53   22.37   20.97    N/A
     8388608        131072     float    none      -1   286.22   29.31   27.48       0   281.20   29.83   27.97    N/A
    16777216        262144     float    none      -1   486.46   34.49   32.33       0  2568.51    6.53    6.12    N/A
    33554432        524288     float    none      -1   795.13   42.20   39.56       0   770.68   43.54   40.82    N/A
    67108864       1048576     float    none      -1  1944.54   34.51   32.35       0  2429.62   27.62   25.89    N/A
   134217728       2097152     float    none      -1  15436.7    8.69    8.15       0  5483.75   24.48   22.95    N/A
   268435456       4194304     float    none      -1  13855.8   19.37   18.16       0  5858.73   45.82   42.95    N/A
   536870912       8388608     float    none      -1  18851.1   28.48   26.70       0  19148.1   28.04   26.29    N/A
  1073741824      16777216     float    none      -1  37793.1   28.41   26.64       0  56713.9   18.93   17.75    N/A
  2147483648      33554432     float    none      -1  85510.8   25.11   23.54       0  70923.2   30.28   28.39    N/A
  4294967296      67108864     float    none      -1   126222   34.03   31.90       0   121749   35.28   33.07    N/A
  8589934592     134217728     float    none      -1   200623   42.82   40.14       0   188907   45.47   42.63    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 15.4133 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:12:29+00:00
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
[R6KD-CX8aaS-GPU-14:12401] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:12401] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12406 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  12407 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  12408 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  12409 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  12410 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  12411 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  12412 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  12413 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  14611 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  14612 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  14613 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  14614 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  14615 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  14616 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  14617 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  14618 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    65.94    0.02    0.01       0    49.13    0.02    0.02    N/A
        2048            32     float    none      -1    49.12    0.04    0.04       0    49.83    0.04    0.04    N/A
        4096            64     float    none      -1    49.15    0.08    0.08       0    48.84    0.08    0.08    N/A
        8192           128     float    none      -1    48.84    0.17    0.16       0    48.72    0.17    0.16    N/A
       16384           256     float    none      -1    55.18    0.30    0.28       0    55.62    0.29    0.28    N/A
       32768           512     float    none      -1    56.88    0.58    0.54       0    49.31    0.66    0.62    N/A
       65536          1024     float    none      -1    50.55    1.30    1.22       0    49.98    1.31    1.23    N/A
      131072          2048     float    none      -1    59.73    2.19    2.06       0    51.70    2.54    2.38    N/A
      262144          4096     float    none      -1    68.77    3.81    3.57       0    61.13    4.29    4.02    N/A
      524288          8192     float    none      -1    65.63    7.99    7.49       0    78.69    6.66    6.25    N/A
     1048576         16384     float    none      -1    91.02   11.52   10.80       0    91.45   11.47   10.75    N/A
     2097152         32768     float    none      -1   103.05   20.35   19.08       0    99.28   21.12   19.80    N/A
     4194304         65536     float    none      -1   174.50   24.04   22.53       0   165.83   25.29   23.71    N/A
     8388608        131072     float    none      -1   254.59   32.95   30.89       0  2567.98    3.27    3.06    N/A
    16777216        262144     float    none      -1   442.62   37.90   35.54       0   432.70   38.77   36.35    N/A
    33554432        524288     float    none      -1  5192.03    6.46    6.06       0  2842.46   11.80   11.07    N/A
    67108864       1048576     float    none      -1  3650.18   18.39   17.24       0  1396.17   48.07   45.06    N/A
   134217728       2097152     float    none      -1  3647.24   36.80   34.50       0  2794.40   48.03   45.03    N/A
   268435456       4194304     float    none      -1  18693.9   14.36   13.46       0  11739.7   22.87   21.44    N/A
   536870912       8388608     float    none      -1  23090.7   23.25   21.80       0  16077.6   33.39   31.31    N/A
  1073741824      16777216     float    none      -1  38988.6   27.54   25.82       0  30175.5   35.58   33.36    N/A
  2147483648      33554432     float    none      -1  71111.2   30.20   28.31       0  68824.3   31.20   29.25    N/A
  4294967296      67108864     float    none      -1   117749   36.48   34.20       0   109447   39.24   36.79    N/A
  8589934592     134217728     float    none      -1   196553   43.70   40.97       0   185930   46.20   43.31    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 15.8748 
#
# Collective test concluded: alltoall_perf
#
```

---

*对比表数据取自各 case 日志 **out-of-place** 列 **algbw**；CASE1 标题为日志原文 “AN OFF”，对应 **AR OFF**。*
