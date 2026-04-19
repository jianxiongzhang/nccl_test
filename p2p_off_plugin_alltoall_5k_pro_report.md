# NCCL AllToAll 性能测试报告（P2P 关闭 + Spectrum-X 网络插件）

**报告主题：** `alltoall_perf`，跨两台 **5K Pro** GPU 服务器，在 **`NCCL_P2P_DISABLE=1`** 且 **`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件 + 对应 `LD_LIBRARY_PATH`）条件下，对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 的 **out-of-place algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志节点为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`；拓扑按测试计划记为两台 **5K Pro**。

---

## 1. 测试命令

三次运行除 NCCL IB 相关开关外保持一致；共性要点：

- **`NCCL_NET_PLUGIN=spcx`**
- **`NCCL_P2P_DISABLE=1`**
- **`NCCL_SHM_DISABLE=1`**，**`UCX_TLS=ib`**，`alltoall_perf -b 1k -e 8G -f 2 -g 1`
- **`LD_LIBRARY_PATH`** 含 **`nccl_spectrum-x_plugin/lib`**（见原始 `COMMAND:`）

### 1.1 AR OFF（基准）

```bash
-x NCCL_NET_PLUGIN=spcx ... \
-x NCCL_P2P_DISABLE=1 \
-x NCCL_IB_ADAPTIVE_ROUTING=0 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/alltoall_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 AR ON / DDP OFF

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0
```

### 1.3 AR ON / DDP ON

```bash
-x NCCL_IB_ADAPTIVE_ROUTING=1 \
-x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1
```

完整 `mpirun` 单行命令见第 4 节「原始数据」各 CASE 的 `COMMAND:` 行。

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台 **5K Pro** GPU 服务器（日志：`R6KD-CX8aaS-GPU-14`、`R6KD-CX8aaS-GPU-15`） |
| 进程/GPU | 每节点 8 进程，共 **16 GPU**；`alltoall_perf` **每 rank 1 GPU**（`-g 1`） |
| 集合通信 | **alltoall_perf** |
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件） |
| P2P | **`NCCL_P2P_DISABLE=1`** |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*`） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllToAll**；跨机经 **bond0 / IB**，并由 **spcx** 插件参与网络路径/策略。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。极小数值行的百分比波动大，仅作参考。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.01 | 0.01 | 0.0% | 0.01 | 0.0% |
| 2048 | 0.04 | 0.04 | 0.0% | 0.04 | 0.0% |
| 4096 | 0.09 | 0.09 | 0.0% | 0.08 | −11.1% |
| 8192 | 0.17 | 0.17 | 0.0% | 0.17 | 0.0% |
| 16384 | 0.34 | 0.32 | −5.9% | 0.34 | 0.0% |
| 32768 | 0.68 | 0.68 | 0.0% | 0.67 | −1.5% |
| 65536 | 1.35 | 1.36 | +0.7% | 1.32 | −2.2% |
| 131072 | 2.58 | 2.58 | 0.0% | 2.39 | −7.4% |
| 262144 | 4.22 | 3.81 | −9.7% | 3.80 | −10.0% |
| 524288 | 8.11 | 6.64 | −18.1% | 6.59 | −18.7% |
| 1048576 | 11.24 | 10.43 | −7.2% | 10.28 | −8.5% |
| 2097152 | 20.33 | 16.90 | −16.9% | 15.00 | −26.2% |
| 4194304 | 23.93 | 21.35 | −10.8% | 21.35 | −10.8% |
| 8388608 | 32.24 | 22.26 | −31.0% | 0.41 | −98.7% |
| 16777216 | 1.71 | 4.62 | +170.2% | 0.18 | −89.5% |
| 33554432 | 1.48 | 0.43 | −70.9% | 0.41 | −72.3% |
| 67108864 | 0.84 | 0.63 | −25.0% | 0.75 | −10.7% |
| 134217728 | 1.30 | 1.24 | −4.6% | 1.44 | +10.8% |
| 268435456 | 2.57 | 2.77 | +7.8% | 3.05 | +18.7% |
| 536870912 | 4.90 | 6.75 | +37.8% | 4.72 | −3.7% |
| 1073741824 | 8.10 | 14.16 | +74.8% | 11.26 | +39.0% |
| 2147483648 | 11.86 | 32.79 | +176.5% | 32.84 | +176.9% |
| 4294967296 | 25.80 | 36.05 | +39.7% | 36.67 | +42.1% |
| 8589934592 | 36.45 | 36.23 | −0.6% | 36.17 | −0.8% |

**摘要观察（out-of-place algbw）：**

- **AR OFF 基准**在 **8 MiB–128 MiB** 区间出现 **极低 algbw**（与 **4 MiB** 的 **~24 GB/s** 相比不连续），更像 **路径/插件状态切换或测量异常**；解读后续百分比时应结合 **time (µs)** 与 **复测**。
- **AR ON / DDP OFF**：**2 MiB–4 MiB** 相对基准多为 **负 Delta**；**512 MiB–2 GiB** 显著 **优于** 基准（例如 **1 GiB** 约 **+75%**，**2 GiB** 约 **+177%**）；**8 MiB** 相对 **−31%**。
- **AR ON / DDP ON**：趋势与 CASE2 类似，但 **8 MiB** 的 **0.41 GB/s**、**16 MiB** 的 **0.18 GB/s** 等点 **异常偏低**，建议 **优先复测** 并对照 **in-place** 与 **`NCCL_DEBUG`**。
- **8 GiB** 三种配置 **几乎持平**（约 **36 GB/s**）。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:21:31+00:00
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
[R6KD-CX8aaS-GPU-14:14245] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:14245] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  14250 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  14251 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  14252 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  14253 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  14254 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  14255 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  14256 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  14257 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  16757 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  16758 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  16759 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  16760 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  16761 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  16762 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  16763 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  16764 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    69.17    0.01    0.01       0    48.31    0.02    0.02    N/A
        2048            32     float    none      -1    47.53    0.04    0.04       0    47.69    0.04    0.04    N/A
        4096            64     float    none      -1    47.40    0.09    0.08       0    47.57    0.09    0.08    N/A
        8192           128     float    none      -1    47.61    0.17    0.16       0    47.44    0.17    0.16    N/A
       16384           256     float    none      -1    48.25    0.34    0.32       0    47.69    0.34    0.32    N/A
       32768           512     float    none      -1    48.37    0.68    0.64       0    48.43    0.68    0.63    N/A
       65536          1024     float    none      -1    48.63    1.35    1.26       0    48.56    1.35    1.27    N/A
      131072          2048     float    none      -1    50.85    2.58    2.42       0    51.93    2.52    2.37    N/A
      262144          4096     float    none      -1    62.05    4.22    3.96       0    60.15    4.36    4.09    N/A
      524288          8192     float    none      -1    64.67    8.11    7.60       0    63.24    8.29    7.77    N/A
     1048576         16384     float    none      -1    93.25   11.24   10.54       0    89.12   11.77   11.03    N/A
     2097152         32768     float    none      -1   103.15   20.33   19.06       0   101.17   20.73   19.43    N/A
     4194304         65536     float    none      -1   175.27   23.93   22.44       0   166.36   25.21   23.64    N/A
     8388608        131072     float    none      -1   260.21   32.24   30.22       0  5176.42    1.62    1.52    N/A
    16777216        262144     float    none      -1  9839.55    1.71    1.60       0  49967.2    0.34    0.31    N/A
    33554432        524288     float    none      -1  22735.5    1.48    1.38       0  30357.0    1.11    1.04    N/A
    67108864       1048576     float    none      -1  79839.1    0.84    0.79       0   100370    0.67    0.63    N/A
   134217728       2097152     float    none      -1   103114    1.30    1.22       0  95184.6    1.41    1.32    N/A
   268435456       4194304     float    none      -1   104294    2.57    2.41       0   105451    2.55    2.39    N/A
   536870912       8388608     float    none      -1   109572    4.90    4.59       0   115675    4.64    4.35    N/A
  1073741824      16777216     float    none      -1   132630    8.10    7.59       0   116486    9.22    8.64    N/A
  2147483648      33554432     float    none      -1   181106   11.86   11.12       0   164435   13.06   12.24    N/A
  4294967296      67108864     float    none      -1   166449   25.80   24.19       0   157494   27.27   25.57    N/A
  8589934592     134217728     float    none      -1   235696   36.45   34.17       0   210205   40.86   38.31    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 7.39538 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:24:10+00:00
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
[R6KD-CX8aaS-GPU-14:14734] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:14734] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  14739 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  14740 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  14741 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  14742 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  14743 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  14744 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  14745 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  14746 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  17313 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  17314 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  17315 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  17316 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  17317 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  17318 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  17319 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  17320 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    77.79    0.01    0.01       0    47.35    0.02    0.02    N/A
        2048            32     float    none      -1    47.73    0.04    0.04       0    47.04    0.04    0.04    N/A
        4096            64     float    none      -1    47.46    0.09    0.08       0    47.38    0.09    0.08    N/A
        8192           128     float    none      -1    47.08    0.17    0.16       0    46.87    0.17    0.16    N/A
       16384           256     float    none      -1    51.65    0.32    0.30       0    47.22    0.35    0.33    N/A
       32768           512     float    none      -1    48.26    0.68    0.64       0    47.71    0.69    0.64    N/A
       65536          1024     float    none      -1    48.30    1.36    1.27       0    48.42    1.35    1.27    N/A
      131072          2048     float    none      -1    50.87    2.58    2.42       0    50.27    2.61    2.44    N/A
      262144          4096     float    none      -1    68.72    3.81    3.58       0    73.99    3.54    3.32    N/A
      524288          8192     float    none      -1    78.90    6.64    6.23       0    78.41    6.69    6.27    N/A
     1048576         16384     float    none      -1   100.57   10.43    9.77       0    99.70   10.52    9.86    N/A
     2097152         32768     float    none      -1   124.06   16.90   15.85       0   118.79   17.65   16.55    N/A
     4194304         65536     float    none      -1   196.43   21.35   20.02       0   189.62   22.12   20.74    N/A
     8388608        131072     float    none      -1   376.90   22.26   20.87       0   313.63   26.75   25.08    N/A
    16777216        262144     float    none      -1  3633.32    4.62    4.33       0  58473.7    0.29    0.27    N/A
    33554432        524288     float    none      -1  78638.9    0.43    0.40       0  37403.4    0.90    0.84    N/A
    67108864       1048576     float    none      -1   106876    0.63    0.59       0   103661    0.65    0.61    N/A
   134217728       2097152     float    none      -1   108267    1.24    1.16       0  84216.8    1.59    1.49    N/A
   268435456       4194304     float    none      -1  97047.0    2.77    2.59       0  93543.0    2.87    2.69    N/A
   536870912       8388608     float    none      -1  79505.8    6.75    6.33       0  76611.6    7.01    6.57    N/A
  1073741824      16777216     float    none      -1  75838.8   14.16   13.27       0  45337.8   23.68   22.20    N/A
  2147483648      33554432     float    none      -1  65484.3   32.79   30.74       0  69925.2   30.71   28.79    N/A
  4294967296      67108864     float    none      -1   119141   36.05   33.80       0   123058   34.90   32.72    N/A
  8589934592     134217728     float    none      -1   237069   36.23   33.97       0   236350   36.34   34.07    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.8641 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:26:40+00:00
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
[R6KD-CX8aaS-GPU-14:15214] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:15214] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  15219 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  15220 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  15221 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  15222 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  15223 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  15224 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  15225 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  15226 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  17869 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  17870 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  17871 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  17872 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  17873 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  17874 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  17875 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  17876 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    68.91    0.01    0.01       0    48.50    0.02    0.02    N/A
        2048            32     float    none      -1    49.29    0.04    0.04       0    48.12    0.04    0.04    N/A
        4096            64     float    none      -1    48.62    0.08    0.08       0    48.15    0.09    0.08    N/A
        8192           128     float    none      -1    47.52    0.17    0.16       0    47.71    0.17    0.16    N/A
       16384           256     float    none      -1    48.58    0.34    0.32       0    48.54    0.34    0.32    N/A
       32768           512     float    none      -1    48.90    0.67    0.63       0    49.92    0.66    0.62    N/A
       65536          1024     float    none      -1    49.52    1.32    1.24       0    80.00    0.82    0.77    N/A
      131072          2048     float    none      -1    54.81    2.39    2.24       0    51.40    2.55    2.39    N/A
      262144          4096     float    none      -1    69.04    3.80    3.56       0    74.33    3.53    3.31    N/A
      524288          8192     float    none      -1    79.55    6.59    6.18       0    79.37    6.61    6.19    N/A
     1048576         16384     float    none      -1   101.96   10.28    9.64       0  1119.33    0.94    0.88    N/A
     2097152         32768     float    none      -1   139.85   15.00   14.06       0   137.03   15.30   14.35    N/A
     4194304         65536     float    none      -1   196.49   21.35   20.01       0   193.71   21.65   20.30    N/A
     8388608        131072     float    none      -1  20553.3    0.41    0.38       0  7095.64    1.18    1.11    N/A
    16777216        262144     float    none      -1  91786.2    0.18    0.17       0  84629.0    0.20    0.19    N/A
    33554432        524288     float    none      -1  81449.8    0.41    0.39       0  80308.9    0.42    0.39    N/A
    67108864       1048576     float    none      -1  89802.1    0.75    0.70       0   114999    0.58    0.55    N/A
   134217728       2097152     float    none      -1  93234.6    1.44    1.35       0  92610.1    1.45    1.36    N/A
   268435456       4194304     float    none      -1  88012.3    3.05    2.86       0  85744.2    3.13    2.93    N/A
   536870912       8388608     float    none      -1   113828    4.72    4.42       0  65963.7    8.14    7.63    N/A
  1073741824      16777216     float    none      -1  95357.2   11.26   10.56       0  45682.3   23.50   22.04    N/A
  2147483648      33554432     float    none      -1  65382.6   32.84   30.79       0  69474.2   30.91   28.98    N/A
  4294967296      67108864     float    none      -1   117118   36.67   34.38       0   123278   34.84   32.66    N/A
  8589934592     134217728     float    none      -1   237515   36.17   33.91       0   238460   36.02   33.77    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 7.48116 
#
# Collective test concluded: alltoall_perf
#
```

---

*表内数据取自各 case 日志 **out-of-place** 列 **algbw**；CASE1 标题 “AN OFF” 对应 **AR OFF**。*
