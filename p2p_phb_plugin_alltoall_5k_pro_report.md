# NCCL AllToAll 性能测试报告（P2P = PHB + Spectrum-X 网络插件）

**报告主题：** `alltoall_perf`，跨两台 **5K Pro** GPU 服务器，在 **`NCCL_P2P_LEVEL=PHB`**、**`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件 + 对应 `LD_LIBRARY_PATH`）条件下，对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 的 **out-of-place algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志节点为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`；拓扑按测试计划记为两台 **5K Pro**。

---

## 1. 测试命令

三次运行除 NCCL IB 相关开关外保持一致；共性要点：

- **`NCCL_NET_PLUGIN=spcx`**
- **`NCCL_P2P_LEVEL=PHB`**
- **`NCCL_SHM_DISABLE=1`**，**`UCX_TLS=ib`**，`alltoall_perf -b 1k -e 8G -f 2 -g 1`
- **`LD_LIBRARY_PATH`** 含 **`nccl_spectrum-x_plugin/lib`**（见原始 `COMMAND:`）

### 1.1 AR OFF（基准）

```bash
-x NCCL_NET_PLUGIN=spcx ... \
-x NCCL_P2P_LEVEL=PHB \
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
| P2P | **`NCCL_P2P_LEVEL=PHB`** |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*`） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllToAll**；跨机经 **bond0 / IB**，并由 **spcx** 插件参与网络路径/策略。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。极小基准值会导致百分比极大，表中 **8 MiB** 行即属此类，应结合 **time (µs)** 与复测解读。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.03 | 0.02 | −33.3% | 0.02 | −33.3% |
| 2048 | 0.07 | 0.07 | 0.0% | 0.07 | 0.0% |
| 4096 | 0.13 | 0.13 | 0.0% | 0.13 | 0.0% |
| 8192 | 0.26 | 0.27 | +3.8% | 0.27 | +3.8% |
| 16384 | 0.49 | 0.52 | +6.1% | 0.53 | +8.2% |
| 32768 | 0.97 | 0.98 | +1.0% | 0.99 | +2.1% |
| 65536 | 1.82 | 1.85 | +1.6% | 1.90 | +4.4% |
| 131072 | 3.01 | 3.41 | +13.3% | 3.49 | +15.9% |
| 262144 | 6.04 | 5.23 | −13.4% | 5.31 | −12.1% |
| 524288 | 10.33 | 8.86 | −14.2% | 9.22 | −10.7% |
| 1048576 | 14.68 | 13.53 | −7.8% | 11.56 | −21.3% |
| 2097152 | 22.87 | 19.74 | −13.7% | 20.13 | −12.0% |
| 4194304 | 26.02 | 22.71 | −12.7% | 22.30 | −14.3% |
| 8388608 | 1.48 | 28.59 | +1832% | 29.04 | +1862% |
| 16777216 | 1.61 | 0.57 | −64.6% | 1.50 | −6.8% |
| 33554432 | 41.45 | 1.70 | −95.9% | 1.34 | −96.8% |
| 67108864 | 0.65 | 0.89 | +36.9% | 0.74 | +13.8% |
| 134217728 | 1.47 | 1.31 | −10.9% | 1.29 | −12.2% |
| 268435456 | 2.70 | 2.41 | −10.7% | 2.95 | +9.3% |
| 536870912 | 5.62 | 4.88 | −13.2% | 6.00 | +6.8% |
| 1073741824 | 10.80 | 11.29 | +4.5% | 10.55 | −2.3% |
| 2147483648 | 18.65 | 26.45 | +41.8% | 29.97 | +60.7% |
| 4294967296 | 26.63 | 32.91 | +23.6% | 33.03 | +24.0% |
| 8589934592 | 34.83 | 34.18 | −1.9% | 35.82 | +2.8% |

**摘要观察（out-of-place algbw）：**

- **AR OFF 基准**在 **8 MiB / 16 MiB / 64 MiB / 128 MiB** 等处 **algbw 极低**，与 **4 MiB（约 26 GB/s）**、**32 MiB（约 41 GB/s）** 不连续；对这些行求 **Δ%** 时会出现 **极大正/负百分比**，更适合作为 **稳定性/异常点** 排查而非直接性能排名。
- **中小消息（约 128 KiB–1 MiB）**：**AR ON** 两种配置在 **128 KiB** 相对基准 **更好**（约 **+13%～+16%**）；**256 KiB–2 MiB** 多为 **负 Delta**。
- **8 MiB**：相对 **AR OFF 的 1.48 GB/s**，两种 AR 配置 **显著更高**（表中显示为 **+1800% 量级**）；若排除基准异常，与 **CASE2/CASE3 的 ~28–29 GB/s** 更宜与 **4 MiB** 或复测后的 **AR OFF 8 MiB** 对齐比较。
- **32 MiB**：两种 AR 配置相对 **41.45 GB/s** 基准 **约 −96%**，建议 **复测**。
- **2 GiB–4 GiB**：两种 AR 配置相对基准 **明显提升**（约 **+24%～+61%**）；**8 GiB** 与基准 **接近**（约 **±3%**）。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:14:23+00:00
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
[R6KD-CX8aaS-GPU-14:12807] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:12807] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12812 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  12813 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  12814 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  12815 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  12816 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  12817 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  12818 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  12819 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  15093 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  15094 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  15095 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  15096 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  15097 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  15098 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  15099 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  15100 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    40.78    0.03    0.02       0    30.98    0.03    0.03    N/A
        2048            32     float    none      -1    30.55    0.07    0.06       0    30.06    0.07    0.06    N/A
        4096            64     float    none      -1    30.75    0.13    0.12       0    30.42    0.13    0.13    N/A
        8192           128     float    none      -1    30.93    0.26    0.25       0    30.23    0.27    0.25    N/A
       16384           256     float    none      -1    33.17    0.49    0.46       0    31.74    0.52    0.48    N/A
       32768           512     float    none      -1    33.89    0.97    0.91       0    33.78    0.97    0.91    N/A
       65536          1024     float    none      -1    35.95    1.82    1.71       0    35.37    1.85    1.74    N/A
      131072          2048     float    none      -1    43.50    3.01    2.82       0    38.79    3.38    3.17    N/A
      262144          4096     float    none      -1    43.37    6.04    5.67       0    50.43    5.20    4.87    N/A
      524288          8192     float    none      -1    50.77   10.33    9.68       0    52.43   10.00    9.37    N/A
     1048576         16384     float    none      -1    71.44   14.68   13.76       0    73.56   14.25   13.36    N/A
     2097152         32768     float    none      -1    91.68   22.87   21.44       0    95.11   22.05   20.67    N/A
     4194304         65536     float    none      -1   161.19   26.02   24.39       0   156.60   26.78   25.11    N/A
     8388608        131072     float    none      -1  5686.63    1.48    1.38       0   267.82   31.32   29.36    N/A
    16777216        262144     float    none      -1  10427.5    1.61    1.51       0  9819.78    1.71    1.60    N/A
    33554432        524288     float    none      -1   809.44   41.45   38.86       0  39706.0    0.85    0.79    N/A
    67108864       1048576     float    none      -1   103605    0.65    0.61       0  86402.1    0.78    0.73    N/A
   134217728       2097152     float    none      -1  91092.4    1.47    1.38       0  95629.5    1.40    1.32    N/A
   268435456       4194304     float    none      -1  99272.0    2.70    2.54       0  89791.0    2.99    2.80    N/A
   536870912       8388608     float    none      -1  95578.5    5.62    5.27       0  86511.6    6.21    5.82    N/A
  1073741824      16777216     float    none      -1  99415.3   10.80   10.13       0  93659.3   11.46   10.75    N/A
  2147483648      33554432     float    none      -1   115136   18.65   17.49       0   109291   19.65   18.42    N/A
  4294967296      67108864     float    none      -1   161268   26.63   24.97       0   143870   29.85   27.99    N/A
  8589934592     134217728     float    none      -1   246641   34.83   32.65       0   238714   35.98   33.74    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.9909 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:16:43+00:00
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
[R6KD-CX8aaS-GPU-14:13286] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:13286] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  13291 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  13292 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  13293 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  13294 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  13295 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  13296 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  13297 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  13299 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  15647 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  15648 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  15649 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  15650 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  15651 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  15652 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  15653 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  15654 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    49.94    0.02    0.02       0    30.35    0.03    0.03    N/A
        2048            32     float    none      -1    30.02    0.07    0.06       0    29.92    0.07    0.06    N/A
        4096            64     float    none      -1    30.81    0.13    0.12       0    30.93    0.13    0.12    N/A
        8192           128     float    none      -1    30.62    0.27    0.25       0    29.70    0.28    0.26    N/A
       16384           256     float    none      -1    31.73    0.52    0.48       0    30.68    0.53    0.50    N/A
       32768           512     float    none      -1    33.57    0.98    0.92       0    32.58    1.01    0.94    N/A
       65536          1024     float    none      -1    35.48    1.85    1.73       0    34.64    1.89    1.77    N/A
      131072          2048     float    none      -1    38.38    3.41    3.20       0    37.99    3.45    3.23    N/A
      262144          4096     float    none      -1    50.16    5.23    4.90       0    52.31    5.01    4.70    N/A
      524288          8192     float    none      -1    59.16    8.86    8.31       0    59.40    8.83    8.27    N/A
     1048576         16384     float    none      -1    77.49   13.53   12.69       0    80.37   13.05   12.23    N/A
     2097152         32768     float    none      -1   106.26   19.74   18.50       0   106.76   19.64   18.42    N/A
     4194304         65536     float    none      -1   184.70   22.71   21.29       0   178.57   23.49   22.02    N/A
     8388608        131072     float    none      -1   293.39   28.59   26.80       0   306.45   27.37   25.66    N/A
    16777216        262144     float    none      -1  29577.1    0.57    0.53       0  26881.7    0.62    0.59    N/A
    33554432        524288     float    none      -1  19726.7    1.70    1.59       0  10598.3    3.17    2.97    N/A
    67108864       1048576     float    none      -1  75668.8    0.89    0.83       0  78608.2    0.85    0.80    N/A
   134217728       2097152     float    none      -1   102572    1.31    1.23       0  78548.9    1.71    1.60    N/A
   268435456       4194304     float    none      -1   111393    2.41    2.26       0  99145.3    2.71    2.54    N/A
   536870912       8388608     float    none      -1   109959    4.88    4.58       0  94323.3    5.69    5.34    N/A
  1073741824      16777216     float    none      -1  95122.9   11.29   10.58       0  76286.6   14.08   13.20    N/A
  2147483648      33554432     float    none      -1  81189.8   26.45   24.80       0  72417.2   29.65   27.80    N/A
  4294967296      67108864     float    none      -1   130494   32.91   30.86       0   129598   33.14   31.07    N/A
  8589934592     134217728     float    none      -1   251298   34.18   32.05       0   251502   34.15   32.02    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.84861 
#
# Collective test concluded: alltoall_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:19:03+00:00
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
[R6KD-CX8aaS-GPU-14:13765] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:13765] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  13770 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  13771 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  13772 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  13773 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  13774 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  13775 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  13776 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  13777 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  16201 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  16202 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  16203 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  16204 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  16205 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  16206 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  16207 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  16208 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    59.75    0.02    0.02       0    30.20    0.03    0.03    N/A
        2048            32     float    none      -1    30.00    0.07    0.06       0    29.94    0.07    0.06    N/A
        4096            64     float    none      -1    30.76    0.13    0.12       0    30.25    0.14    0.13    N/A
        8192           128     float    none      -1    30.66    0.27    0.25       0    30.11    0.27    0.26    N/A
       16384           256     float    none      -1    31.16    0.53    0.49       0    30.63    0.53    0.50    N/A
       32768           512     float    none      -1    33.05    0.99    0.93       0    32.19    1.02    0.95    N/A
       65536          1024     float    none      -1    34.48    1.90    1.78       0    33.98    1.93    1.81    N/A
      131072          2048     float    none      -1    37.60    3.49    3.27       0    37.50    3.49    3.28    N/A
      262144          4096     float    none      -1    49.33    5.31    4.98       0    55.29    4.74    4.44    N/A
      524288          8192     float    none      -1    56.86    9.22    8.65       0    57.84    9.06    8.50    N/A
     1048576         16384     float    none      -1    90.69   11.56   10.84       0    87.24   12.02   11.27    N/A
     2097152         32768     float    none      -1   104.18   20.13   18.87       0   106.67   19.66   18.43    N/A
     4194304         65536     float    none      -1   188.08   22.30   20.91       0   180.30   23.26   21.81    N/A
     8388608        131072     float    none      -1   288.83   29.04   27.23       0   338.09   24.81   23.26    N/A
    16777216        262144     float    none      -1  11206.7    1.50    1.40       0  11381.2    1.47    1.38    N/A
    33554432        524288     float    none      -1  25006.8    1.34    1.26       0  19073.6    1.76    1.65    N/A
    67108864       1048576     float    none      -1  90679.0    0.74    0.69       0  45442.0    1.48    1.38    N/A
   134217728       2097152     float    none      -1   103819    1.29    1.21       0   105134    1.28    1.20    N/A
   268435456       4194304     float    none      -1  91127.0    2.95    2.76       0   110299    2.43    2.28    N/A
   536870912       8388608     float    none      -1  89432.2    6.00    5.63       0   108881    4.93    4.62    N/A
  1073741824      16777216     float    none      -1   101787   10.55    9.89       0  60640.2   17.71   16.60    N/A
  2147483648      33554432     float    none      -1  71664.3   29.97   28.09       0  68174.8   31.50   29.53    N/A
  4294967296      67108864     float    none      -1   130048   33.03   30.96       0   128093   33.53   31.43    N/A
  8589934592     134217728     float    none      -1   239796   35.82   33.58       0   242007   35.49   33.28    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 8.99947 
#
# Collective test concluded: alltoall_perf
#
```

---

*表内数据取自各 case 日志 **out-of-place** 列 **algbw**；CASE1 标题 “AN OFF” 对应 **AR OFF**。*
