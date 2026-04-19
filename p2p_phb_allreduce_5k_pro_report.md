# NCCL AllReduce 性能测试报告（P2P = PHB）

**报告主题：** `all_reduce_perf`，跨两台 **5K Pro** GPU 服务器，在 **`NCCL_P2P_LEVEL=PHB`** 前提下对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 三种配置的 **out-of-place algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志节点为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`；环境与拓扑按测试计划记为两台 **5K Pro**，若与资产命名不一致请以 CMDB 为准。

---

## 1. 测试命令

三次运行除 NCCL IB 开关外保持一致；共性包括：`NCCL_SHM_DISABLE=1`、`UCX_TLS=ib`、**`NCCL_P2P_LEVEL=PHB`**、`all_reduce_perf -b 1k -e 8G -f 2 -g 1`。

### 1.1 AR OFF（基准）

```bash
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=0 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 AR ON / DDP OFF

```bash
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 AR ON / DDP ON

```bash
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 \
/workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

完整 `mpirun` 单行命令见第 4 节「原始数据」各 CASE 的 `COMMAND:` 行。

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台 **5K Pro** GPU 服务器（日志：`R6KD-CX8aaS-GPU-14`、`R6KD-CX8aaS-GPU-15`） |
| 进程/GPU | 每节点 8 进程，共 **16 GPU**；`all_reduce_perf` **每 rank 1 GPU**（`-g 1`） |
| 集合通信 | **all_reduce_perf** |
| P2P | **`NCCL_P2P_LEVEL=PHB`**（允许同一 PCIe 主机桥接域内的 P2P，按 NCCL 语义筛选路径） |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*`） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllReduce**；跨机经 **bond0 / IB**。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.03 | 0.03 | 0.0% | 0.03 | 0.0% |
| 2048 | 0.05 | 0.05 | 0.0% | 0.05 | 0.0% |
| 4096 | 0.10 | 0.10 | 0.0% | 0.10 | 0.0% |
| 8192 | 0.18 | 0.15 | −16.7% | 0.16 | −11.1% |
| 16384 | 0.35 | 0.34 | −2.9% | 0.34 | −2.9% |
| 32768 | 0.44 | 0.53 | +20.5% | 0.51 | +15.9% |
| 65536 | 0.73 | 0.66 | −9.6% | 0.55 | −24.7% |
| 131072 | 0.68 | 0.64 | −5.9% | 0.71 | +4.4% |
| 262144 | 0.82 | 0.71 | −13.4% | 0.73 | −11.0% |
| 524288 | 0.76 | 0.67 | −11.8% | 0.35 | −53.9% |
| 1048576 | 0.82 | 0.75 | −8.5% | 0.87 | +6.1% |
| 2097152 | 5.41 | 5.19 | −4.1% | 5.55 | +2.6% |
| 4194304 | 5.85 | 5.68 | −2.9% | 5.88 | +0.5% |
| 8388608 | 6.11 | 5.88 | −3.8% | 6.14 | +0.5% |
| 16777216 | 17.73 | 4.34 | −75.5% | 13.10 | −26.1% |
| 33554432 | 21.30 | 9.82 | −53.9% | 21.29 | −0.05% |
| 67108864 | 23.99 | 14.97 | −37.6% | 24.07 | +0.3% |
| 134217728 | 25.63 | 20.31 | −20.6% | 19.84 | −22.6% |
| 268435456 | 26.20 | 18.08 | −31.0% | 13.16 | −49.8% |
| 536870912 | 26.35 | 23.03 | −12.6% | 20.63 | −21.7% |
| 1073741824 | 26.45 | 26.09 | −1.4% | 26.45 | 0.0% |
| 2147483648 | 26.51 | 26.51 | 0.0% | 26.52 | +0.04% |
| 4294967296 | 26.54 | 26.54 | 0.0% | 26.53 | −0.04% |
| 8589934592 | 26.58 | 26.58 | 0.0% | 26.49 | −0.3% |

**摘要观察（out-of-place algbw）：**

- **AR ON / DDP OFF** 在 **16 MiB–64 MiB** 一带相对基准 **显著偏低**（16 MiB 约 **−76%**，32 MiB 约 **−54%**，64 MiB 约 **−38%**），**128 MiB–512 MiB** 亦低于基准；**1 GiB 及以上** 与基准几乎一致。
- **AR ON / DDP ON** 在 **16 MiB、256 MiB、128 MiB–512 MiB** 等点相对基准 **偏弱**；**32 MiB、64 MiB、1 GiB–4 GiB** 与基准接近或略优/略差；**8 GiB** 略低于基准约 **0.3%**。
- **512 KiB** 的 DDP ON 行 **0.35 GB/s** 相对前后尺寸异常，建议 **复测** 并结合 `NCCL_DEBUG`/图算法日志排查是否为噪声或路径切换。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:01:37+00:00
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
[R6KD-CX8aaS-GPU-14:10558] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:10558] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  10563 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  10564 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  10565 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  10566 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  10567 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  10568 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  10569 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  10570 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  12351 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  12352 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  12353 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  12354 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  12355 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  12356 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  12357 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  12358 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    37.89    0.03    0.05       0    35.31    0.03    0.05       0
        2048           512     float     sum      -1    37.85    0.05    0.10       0    37.86    0.05    0.10       0
        4096          1024     float     sum      -1    42.41    0.10    0.18       0    42.16    0.10    0.18       0
        8192          2048     float     sum      -1    44.98    0.18    0.34       0    41.98    0.20    0.37       0
       16384          4096     float     sum      -1    47.21    0.35    0.65       0    43.21    0.38    0.71       0
       32768          8192     float     sum      -1    74.99    0.44    0.82       0    76.67    0.43    0.80       0
       65536         16384     float     sum      -1    89.27    0.73    1.38       0    92.13    0.71    1.33       0
      131072         32768     float     sum      -1   192.34    0.68    1.28       0   176.53    0.74    1.39       0
      262144         65536     float     sum      -1   321.35    0.82    1.53       0   334.98    0.78    1.47       0
      524288        131072     float     sum      -1   694.20    0.76    1.42       0   675.37    0.78    1.46       0
     1048576        262144     float     sum      -1  1271.30    0.82    1.55       0  1322.22    0.79    1.49       0
     2097152        524288     float     sum      -1   387.81    5.41   10.14       0   387.70    5.41   10.14       0
     4194304       1048576     float     sum      -1   716.81    5.85   10.97       0   695.75    6.03   11.30       0
     8388608       2097152     float     sum      -1  1373.28    6.11   11.45       0  1376.62    6.09   11.43       0
    16777216       4194304     float     sum      -1   946.44   17.73   33.24       0   947.64   17.70   33.20       0
    33554432       8388608     float     sum      -1  1575.43   21.30   39.93       0  1575.51   21.30   39.93       0
    67108864      16777216     float     sum      -1  2797.57   23.99   44.98       0  2785.21   24.09   45.18       0
   134217728      33554432     float     sum      -1  5237.35   25.63   48.05       0  5248.61   25.57   47.95       0
   268435456      67108864     float     sum      -1  10243.7   26.20   49.13       0  10246.7   26.20   49.12       0
   536870912     134217728     float     sum      -1  20378.4   26.35   49.40       0  20347.4   26.39   49.47       0
  1073741824     268435456     float     sum      -1  40598.2   26.45   49.59       0  40571.9   26.47   49.62       0
  2147483648     536870912     float     sum      -1  81000.5   26.51   49.71       0  81010.2   26.51   49.70       0
  4294967296    1073741824     float     sum      -1   161851   26.54   49.76       0   161894   26.53   49.74       0
  8589934592    2147483648     float     sum      -1   323195   26.58   49.83       0   323115   26.58   49.85       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.0721 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:03:27+00:00
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
[R6KD-CX8aaS-GPU-14:10892] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:10892] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  10897 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  10898 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  10899 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  10900 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  10901 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  10902 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  10903 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  10904 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  12761 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  12762 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  12763 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  12764 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  12765 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  12766 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  12767 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  12768 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    37.70    0.03    0.05       0    35.39    0.03    0.05       0
        2048           512     float     sum      -1    38.04    0.05    0.10       0    38.37    0.05    0.10       0
        4096          1024     float     sum      -1    42.22    0.10    0.18       0    41.88    0.10    0.18       0
        8192          2048     float     sum      -1    52.95    0.15    0.29       0    43.01    0.19    0.36       0
       16384          4096     float     sum      -1    47.49    0.34    0.65       0    45.03    0.36    0.68       0
       32768          8192     float     sum      -1    61.43    0.53    1.00       0    62.58    0.52    0.98       0
       65536         16384     float     sum      -1   100.02    0.66    1.23       0   102.64    0.64    1.20       0
      131072         32768     float     sum      -1   203.99    0.64    1.20       0   204.90    0.64    1.20       0
      262144         65536     float     sum      -1   369.93    0.71    1.33       0   327.04    0.80    1.50       0
      524288        131072     float     sum      -1   779.04    0.67    1.26       0   778.83    0.67    1.26       0
     1048576        262144     float     sum      -1  1397.04    0.75    1.41       0  1410.40    0.74    1.39       0
     2097152        524288     float     sum      -1   403.90    5.19    9.74       0  4223.27    0.50    0.93       0
     4194304       1048576     float     sum      -1   738.07    5.68   10.66       0   730.76    5.74   10.76       0
     8388608       2097152     float     sum      -1  1426.61    5.88   11.03       0  1395.79    6.01   11.27       0
    16777216       4194304     float     sum      -1  3865.59    4.34    8.14       0   953.91   17.59   32.98       0
    33554432       8388608     float     sum      -1  3417.05    9.82   18.41       0  4713.61    7.12   13.35       0
    67108864      16777216     float     sum      -1  4483.59   14.97   28.06       0  2801.74   23.95   44.91       0
   134217728      33554432     float     sum      -1  6608.42   20.31   38.08       0  5340.76   25.13   47.12       0
   268435456      67108864     float     sum      -1  14846.3   18.08   33.90       0  13879.0   19.34   36.26       0
   536870912     134217728     float     sum      -1  23313.3   23.03   43.18       0  22640.9   23.71   44.46       0
  1073741824     268435456     float     sum      -1  41159.5   26.09   48.91       0  40633.2   26.43   49.55       0
  2147483648     536870912     float     sum      -1  80998.6   26.51   49.71       0  81013.2   26.51   49.70       0
  4294967296    1073741824     float     sum      -1   161818   26.54   49.77       0   161826   26.54   49.76       0
  8589934592    2147483648     float     sum      -1   323184   26.58   49.84       0   323126   26.58   49.84       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.8737 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:05:21+00:00
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
[R6KD-CX8aaS-GPU-14:11226] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:11226] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  11231 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  11232 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  11233 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  11234 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  11235 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  11236 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  11237 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  11238 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  13171 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  13172 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  13173 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  13174 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  13175 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  13176 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  13177 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  13178 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    37.35    0.03    0.05       0    35.00    0.03    0.05       0
        2048           512     float     sum      -1    37.40    0.05    0.10       0    37.40    0.05    0.10       0
        4096          1024     float     sum      -1    41.93    0.10    0.18       0    41.57    0.10    0.18       0
        8192          2048     float     sum      -1    52.56    0.16    0.29       0    42.32    0.19    0.36       0
       16384          4096     float     sum      -1    48.01    0.34    0.64       0    45.17    0.36    0.68       0
       32768          8192     float     sum      -1    64.27    0.51    0.96       0    69.81    0.47    0.88       0
       65536         16384     float     sum      -1   119.59    0.55    1.03       0   118.41    0.55    1.04       0
      131072         32768     float     sum      -1   185.54    0.71    1.32       0   191.82    0.68    1.28       0
      262144         65536     float     sum      -1   358.59    0.73    1.37       0   371.47    0.71    1.32       0
      524288        131072     float     sum      -1  1513.16    0.35    0.65       0   605.74    0.87    1.62       0
     1048576        262144     float     sum      -1  1202.98    0.87    1.63       0  1243.23    0.84    1.58       0
     2097152        524288     float     sum      -1   377.61    5.55   10.41       0   379.24    5.53   10.37       0
     4194304       1048576     float     sum      -1   713.46    5.88   11.02       0   707.98    5.92   11.11       0
     8388608       2097152     float     sum      -1  1366.54    6.14   11.51       0  1411.42    5.94   11.14       0
    16777216       4194304     float     sum      -1  1281.09   13.10   24.55       0   949.93   17.66   33.12       0
    33554432       8388608     float     sum      -1  1575.79   21.29   39.93       0  2696.02   12.45   23.34       0
    67108864      16777216     float     sum      -1  2788.08   24.07   45.13       0  2886.66   23.25   43.59       0
   134217728      33554432     float     sum      -1  6764.31   19.84   37.20       0  5242.78   25.60   48.00       0
   268435456      67108864     float     sum      -1  20402.6   13.16   24.67       0  15514.2   17.30   32.44       0
   536870912     134217728     float     sum      -1  26018.7   20.63   38.69       0  20818.4   25.79   48.35       0
  1073741824     268435456     float     sum      -1  40598.0   26.45   49.59       0  41230.9   26.04   48.83       0
  2147483648     536870912     float     sum      -1  80980.9   26.52   49.72       0  80986.2   26.52   49.72       0
  4294967296    1073741824     float     sum      -1   161870   26.53   49.75       0   161775   26.55   49.78       0
  8589934592    2147483648     float     sum      -1   324288   26.49   49.67       0   323056   26.59   49.86       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 19.1424 
#
# Collective test concluded: all_reduce_perf
#
```

---

*表内数据取自各 case 日志 **out-of-place** 列 **algbw**；CASE1 标题 “AN OFF” 对应 **AR OFF**。*
