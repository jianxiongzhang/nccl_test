# NCCL AllReduce 性能测试报告（P2P 关闭）

**报告主题：** `all_reduce_perf`，跨两台 6KD GPU 服务器，对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 三种配置下 **out-of-place** 的 **algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为 100%，其余两种配置给出绝对值及相对基准的百分比变化（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

---

## 1. 测试命令

以下三条命令分别对应三个用例；除 NCCL IB 相关开关外，其余 `mpirun`/NCCL/UCX 环境变量保持一致。共性要点：`NCCL_P2P_DISABLE=1`（禁用 P2P）、`NCCL_SHM_DISABLE=1`、`UCX_TLS=ib`、`all_reduce_perf -b 1k -e 8G -f 2 -g 1`。

### 1.1 AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot \
  -mca plm_rsh_args "-p 3456" --mca pml ucx --mca btl ^openib \
  -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 \
  -x NCCL_NET_PLUGIN=none -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH,NET,TUNING \
  -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 \
  -x NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_0,mlx5_1,mlx5_6,mlx5_7,mlx5_4,mlx5_5 \
  -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 \
  -x NCCL_P2P_DISABLE=1 \
  -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:... \
  -x NCCL_IB_ADAPTIVE_ROUTING=0 \
  -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
  /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 AR ON / DDP OFF

与 1.1 相同，仅将自适应路由打开：

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0
```

### 1.3 AR ON / DDP ON

在 1.2 基础上将 DDP 相关 IB 特性一并打开：

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1
```

> 说明：完整单行命令见文末「原始数据」各 CASE 的 `COMMAND:` 字段（与机上日志一致）。

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台 **6KD** GPU 服务器：`R6KD-CX8aaS-GPU-14`、`R6KD-CX8aaS-GPU-15` |
| 进程/GPU | 每节点 8 进程，共 **16 GPU**；`all_reduce_perf` **每 rank 1 GPU**（`-g 1`） |
| 集合通信 | **all_reduce_perf**（NCCL collective benchmark） |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*` 列表） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |
| P2P | `NCCL_P2P_DISABLE=1`（本报告标题所指的 P2P 关闭场景） |

拓扑简述：**双机、每机 8×GPU，16 rank 全互联 AllReduce**；跨机流量走 **bond0 / IB**。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**algbw 单位：GB/s。**  
**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。基准为 0 或极小值时百分比参考意义有限，表中仍按公式给出。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.02 | 0.02 | 0.0% | 0.02 | 0.0% |
| 2048 | 0.03 | 0.03 | 0.0% | 0.03 | 0.0% |
| 4096 | 0.06 | 0.06 | 0.0% | 0.06 | 0.0% |
| 8192 | 0.11 | 0.11 | 0.0% | 0.11 | 0.0% |
| 16384 | 0.20 | 0.19 | −5.0% | 0.19 | −5.0% |
| 32768 | 0.25 | 0.36 | +44.0% | 0.35 | +40.0% |
| 65536 | 0.60 | 0.58 | −3.3% | 0.58 | −3.3% |
| 131072 | 1.28 | 1.12 | −12.5% | 1.30 | +1.6% |
| 262144 | 1.82 | 1.75 | −3.8% | 0.12 | −93.4% |
| 524288 | 3.53 | 2.56 | −27.5% | 3.61 | +2.3% |
| 1048576 | 4.66 | 3.60 | −22.7% | 4.45 | −4.5% |
| 2097152 | 5.42 | 4.52 | −16.6% | 5.40 | −0.4% |
| 4194304 | 5.94 | 0.96 | −83.8% | 6.11 | +2.9% |
| 8388608 | 17.98 | 8.32 | −53.7% | 17.86 | −0.7% |
| 16777216 | 20.00 | 8.17 | −59.1% | 20.93 | +4.7% |
| 33554432 | 19.85 | 19.56 | −1.5% | 20.06 | +1.1% |
| 67108864 | 20.15 | 19.31 | −4.2% | 12.51 | −37.9% |
| 134217728 | 22.93 | 15.62 | −31.9% | 21.24 | −7.4% |
| 268435456 | 21.45 | 21.75 | +1.4% | 20.92 | −2.5% |
| 536870912 | 19.77 | 17.75 | −10.2% | 17.31 | −12.4% |
| 1073741824 | 19.92 | 20.30 | +1.9% | 19.60 | −1.6% |
| 2147483648 | 19.73 | 20.32 | +3.0% | 19.68 | −0.3% |
| 4294967296 | 19.80 | 20.19 | +2.0% | 19.61 | −1.0% |
| 8589934592 | 19.74 | 20.25 | +2.6% | 19.72 | −0.1% |

**摘要观察（out-of-place algbw）：**

- **AR ON / DDP OFF** 在 **4 MiB、8 MiB、16 MiB** 等区间出现 **显著回落**（例如 4 MiB 约 **−84%**、8 MiB 约 **−54%**），与基准相比波动大；大消息（约 **256 MiB–8 GiB**）多数接近或略优于基准。
- **AR ON / DDP ON** 在 **64 MiB、256 KiB** 等点出现 **明显异常低谷**（64 MiB 约 **−38%**，256 KiB 约 **−93%**），其余区间与基准接近或互有胜负，需在相同条件下 **复测** 以区分噪声与稳定退化。

---

## 4. 原始数据（日志摘录）

以下为三次运行完整日志文本，便于审计与复现（CASE1 标题在日志中为 “AN OFF”，结合命令可知为 **AR OFF**）。

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:07:16+00:00
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
[R6KD-CX8aaS-GPU-14:11560] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:11560] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  11565 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  11566 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  11567 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  11568 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  11569 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  11570 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  11571 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  11572 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  13581 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  13582 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  13583 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  13584 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  13585 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  13586 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  13587 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  13589 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    63.73    0.02    0.03       0    57.93    0.02    0.03       0
        2048           512     float     sum      -1    60.50    0.03    0.06       0    59.82    0.03    0.06       0
        4096          1024     float     sum      -1    66.96    0.06    0.11       0    66.24    0.06    0.12       0
        8192          2048     float     sum      -1    76.73    0.11    0.20       0    70.37    0.12    0.22       0
       16384          4096     float     sum      -1    80.71    0.20    0.38       0    71.52    0.23    0.43       0
       32768          8192     float     sum      -1   132.79    0.25    0.46       0   130.12    0.25    0.47       0
       65536         16384     float     sum      -1   109.70    0.60    1.12       0   107.20    0.61    1.15       0
      131072         32768     float     sum      -1   102.30    1.28    2.40       0   100.74    1.30    2.44       0
      262144         65536     float     sum      -1   144.10    1.82    3.41       0   143.61    1.83    3.42       0
      524288        131072     float     sum      -1   148.59    3.53    6.62       0   147.86    3.55    6.65       0
     1048576        262144     float     sum      -1   224.82    4.66    8.75       0   235.74    4.45    8.34       0
     2097152        524288     float     sum      -1   387.00    5.42   10.16       0   382.52    5.48   10.28       0
     4194304       1048576     float     sum      -1   706.30    5.94   11.13       0   682.19    6.15   11.53       0
     8388608       2097152     float     sum      -1   466.52   17.98   33.71       0   466.08   18.00   33.75       0
    16777216       4194304     float     sum      -1   839.02   20.00   37.49       0   849.07   19.76   37.05       0
    33554432       8388608     float     sum      -1  1690.54   19.85   37.22       0  1690.01   19.85   37.23       0
    67108864      16777216     float     sum      -1  3329.65   20.15   37.79       0  3279.07   20.47   38.37       0
   134217728      33554432     float     sum      -1  5853.77   22.93   42.99       0  5883.09   22.81   42.78       0
   268435456      67108864     float     sum      -1  12516.0   21.45   40.21       0  12512.3   21.45   40.23       0
   536870912     134217728     float     sum      -1  27154.3   19.77   37.07       0  27159.0   19.77   37.06       0
  1073741824     268435456     float     sum      -1  53915.3   19.92   37.34       0  54336.1   19.76   37.05       0
  2147483648     536870912     float     sum      -1   108861   19.73   36.99       0   108570   19.78   37.09       0
  4294967296    1073741824     float     sum      -1   216944   19.80   37.12       0   217275   19.77   37.06       0
  8589934592    2147483648     float     sum      -1   435186   19.74   37.01       0   433339   19.82   37.17       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 19.1617 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:09:14+00:00
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
[R6KD-CX8aaS-GPU-14:11896] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:11896] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  11901 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  11902 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  11903 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  11904 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  11905 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  11906 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  11907 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  11908 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  13993 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  13994 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  13995 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  13996 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  13997 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  13998 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  13999 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  14000 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    66.03    0.02    0.03       0    57.88    0.02    0.03       0
        2048           512     float     sum      -1    59.72    0.03    0.06       0    59.34    0.03    0.06       0
        4096          1024     float     sum      -1    66.34    0.06    0.12       0    65.74    0.06    0.12       0
        8192          2048     float     sum      -1    77.09    0.11    0.20       0    70.15    0.12    0.22       0
       16384          4096     float     sum      -1    86.98    0.19    0.35       0    71.81    0.23    0.43       0
       32768          8192     float     sum      -1    90.92    0.36    0.68       0    89.66    0.37    0.69       0
       65536         16384     float     sum      -1   113.44    0.58    1.08       0   112.05    0.58    1.10       0
      131072         32768     float     sum      -1   116.85    1.12    2.10       0   116.30    1.13    2.11       0
      262144         65536     float     sum      -1   149.92    1.75    3.28       0   154.05    1.70    3.19       0
      524288        131072     float     sum      -1   205.14    2.56    4.79       0   204.37    2.57    4.81       0
     1048576        262144     float     sum      -1   291.07    3.60    6.75       0   284.75    3.68    6.90       0
     2097152        524288     float     sum      -1   463.58    4.52    8.48       0   452.53    4.63    8.69       0
     4194304       1048576     float     sum      -1  4355.19    0.96    1.81       0   757.17    5.54   10.39       0
     8388608       2097152     float     sum      -1  1008.13    8.32   15.60       0   516.03   16.26   30.48       0
    16777216       4194304     float     sum      -1  2054.75    8.17   15.31       0   847.64   19.79   37.11       0
    33554432       8388608     float     sum      -1  1715.23   19.56   36.68       0  6840.75    4.91    9.20       0
    67108864      16777216     float     sum      -1  3475.57   19.31   36.20       0  3613.29   18.57   34.82       0
   134217728      33554432     float     sum      -1  8590.44   15.62   29.30       0  5861.61   22.90   42.93       0
   268435456      67108864     float     sum      -1  12342.0   21.75   40.78       0  15660.4   17.14   32.14       0
   536870912     134217728     float     sum      -1  30248.1   17.75   33.28       0  27187.7   19.75   37.03       0
  1073741824     268435456     float     sum      -1  52903.6   20.30   38.06       0  53603.7   20.03   37.56       0
  2147483648     536870912     float     sum      -1   105680   20.32   38.10       0   105789   20.30   38.06       0
  4294967296    1073741824     float     sum      -1   212703   20.19   37.86       0   211780   20.28   38.03       0
  8589934592    2147483648     float     sum      -1   424241   20.25   37.96       0   422831   20.32   38.09       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 16.7303 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:11:18+00:00
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
[R6KD-CX8aaS-GPU-14:12232] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:12232] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12237 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  12238 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  12239 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  12240 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  12241 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  12242 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  12243 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  12244 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  14405 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  14406 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  14407 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  14408 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  14409 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  14410 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  14411 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  14412 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    63.36    0.02    0.03       0    57.42    0.02    0.03       0
        2048           512     float     sum      -1    59.86    0.03    0.06       0    59.41    0.03    0.06       0
        4096          1024     float     sum      -1    65.42    0.06    0.12       0    64.85    0.06    0.12       0
        8192          2048     float     sum      -1    75.85    0.11    0.20       0    69.25    0.12    0.22       0
       16384          4096     float     sum      -1    87.01    0.19    0.35       0    71.71    0.23    0.43       0
       32768          8192     float     sum      -1    92.69    0.35    0.66       0    86.93    0.38    0.71       0
       65536         16384     float     sum      -1   113.07    0.58    1.09       0   108.36    0.60    1.13       0
      131072         32768     float     sum      -1   100.97    1.30    2.43       0   100.60    1.30    2.44       0
      262144         65536     float     sum      -1  2164.05    0.12    0.23       0   117.46    2.23    4.18       0
      524288        131072     float     sum      -1   145.43    3.61    6.76       0   145.48    3.60    6.76       0
     1048576        262144     float     sum      -1   235.88    4.45    8.34       0   240.78    4.35    8.17       0
     2097152        524288     float     sum      -1   388.07    5.40   10.13       0   383.90    5.46   10.24       0
     4194304       1048576     float     sum      -1   686.98    6.11   11.45       0   679.96    6.17   11.57       0
     8388608       2097152     float     sum      -1   469.56   17.86   33.50       0   468.11   17.92   33.60       0
    16777216       4194304     float     sum      -1   801.55   20.93   39.25       0  1193.29   14.06   26.36       0
    33554432       8388608     float     sum      -1  1672.77   20.06   37.61       0  1685.37   19.91   37.33       0
    67108864      16777216     float     sum      -1  5365.74   12.51   23.45       0  3893.19   17.24   32.32       0
   134217728      33554432     float     sum      -1  6319.93   21.24   39.82       0  6064.90   22.13   41.49       0
   268435456      67108864     float     sum      -1  12830.8   20.92   39.23       0  18872.9   14.22   26.67       0
   536870912     134217728     float     sum      -1  31014.6   17.31   32.46       0  30912.3   17.37   32.56       0
  1073741824     268435456     float     sum      -1  54778.4   19.60   36.75       0  54278.4   19.78   37.09       0
  2147483648     536870912     float     sum      -1   109094   19.68   36.91       0   109573   19.60   36.75       0
  4294967296    1073741824     float     sum      -1   219024   19.61   36.77       0   217761   19.72   36.98       0
  8589934592    2147483648     float     sum      -1   435564   19.72   36.98       0   436511   19.68   36.90       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.8894 
#
# Collective test concluded: all_reduce_perf
#
```

---

*报告生成说明：对比表仅使用各 case 日志中 **out-of-place** 列的 **algbw**；百分比为相对 **AR OFF** 的相对变化。*
