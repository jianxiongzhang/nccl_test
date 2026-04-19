# NCCL AllReduce 性能测试报告（P2P = PHB + Spectrum-X 网络插件）

**报告主题：** `all_reduce_perf`，跨两台 **5K Pro** GPU 服务器，在 **`NCCL_P2P_LEVEL=PHB`**、**`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件 + 对应 `LD_LIBRARY_PATH`）条件下，对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 的 **out-of-place algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志节点为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`；拓扑按测试计划记为两台 **5K Pro**。

---

## 1. 测试命令

三次运行除 NCCL IB 相关开关外保持一致；共性要点：

- **`NCCL_NET_PLUGIN=spcx`**
- **`NCCL_P2P_LEVEL=PHB`**
- **`NCCL_SHM_DISABLE=1`**，**`UCX_TLS=ib`**，`all_reduce_perf -b 1k -e 8G -f 2 -g 1`
- **`LD_LIBRARY_PATH`** 含 **`nccl_spectrum-x_plugin/lib`**（见原始 `COMMAND:`）

### 1.1 AR OFF（基准）

```bash
-x NCCL_NET_PLUGIN=spcx ... \
-x NCCL_P2P_LEVEL=PHB \
-x NCCL_IB_ADAPTIVE_ROUTING=0 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
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
| 进程/GPU | 每节点 8 进程，共 **16 GPU**；`all_reduce_perf` **每 rank 1 GPU**（`-g 1`） |
| 集合通信 | **all_reduce_perf** |
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件） |
| P2P | **`NCCL_P2P_LEVEL=PHB`** |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*`） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllReduce**；跨机经 **bond0 / IB**，并由 **spcx** 插件参与网络路径/策略。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.03 | 0.03 | 0.0% | 0.03 | 0.0% |
| 2048 | 0.06 | 0.05 | −16.7% | 0.05 | −16.7% |
| 4096 | 0.10 | 0.10 | 0.0% | 0.10 | 0.0% |
| 8192 | 0.16 | 0.17 | +6.3% | 0.18 | +12.5% |
| 16384 | 0.36 | 0.33 | −8.3% | 0.35 | −2.8% |
| 32768 | 0.57 | 0.53 | −7.0% | 0.51 | −10.5% |
| 65536 | 0.63 | 0.67 | +6.3% | 0.63 | 0.0% |
| 131072 | 0.58 | 0.62 | +6.9% | 0.57 | −1.7% |
| 262144 | 0.66 | 0.67 | +1.5% | 0.62 | −6.1% |
| 524288 | 0.83 | 0.89 | +7.2% | 0.82 | −1.2% |
| 1048576 | 0.83 | 0.88 | +6.0% | 0.79 | −4.8% |
| 2097152 | 5.42 | 5.12 | −5.5% | 5.22 | −3.7% |
| 4194304 | 5.90 | 5.64 | −4.4% | 5.69 | −3.6% |
| 8388608 | 6.05 | 0.38 | −93.7% | 0.16 | −97.4% |
| 16777216 | 17.56 | 13.59 | −22.6% | 16.38 | −6.7% |
| 33554432 | 21.01 | 20.03 | −4.7% | 20.20 | −3.9% |
| 67108864 | 23.95 | 23.31 | −2.7% | 23.16 | −3.3% |
| 134217728 | 25.60 | 25.26 | −1.3% | 19.11 | −25.4% |
| 268435456 | 26.17 | 18.72 | −28.5% | 22.46 | −14.2% |
| 536870912 | 22.82 | 26.25 | +15.0% | 22.07 | −3.3% |
| 1073741824 | 26.46 | 26.41 | −0.2% | 26.15 | −1.2% |
| 2147483648 | 26.51 | 26.49 | −0.1% | 26.50 | −0.04% |
| 4294967296 | 26.53 | 26.54 | +0.04% | 26.14 | −1.5% |
| 8589934592 | 26.52 | 26.56 | +0.2% | 26.58 | +0.2% |

**摘要观察（out-of-place algbw）：**

- **8 MiB**：两种 AR 配置均出现 **断崖式回落**（约 **−94%～−97%**），与前后尺寸不连续，建议 **复测** 并对照 **in-place** 与 **`NCCL_DEBUG`**。
- **AR ON / DDP OFF**：**256 MiB** 相对基准约 **−28.5%**；**512 MiB** 相对基准约 **+15%**；**16 MiB** 约 **−22.6%**。
- **AR ON / DDP ON**：**128 MiB** 约 **−25%**；**256 MiB** 约 **−14%**；**4 GiB** 约 **−1.5%**；其余大消息与基准 **基本持平**。
- **约 1 GiB–8 GiB**：多数 **Δ%** 在 **±1%** 量级，三种配置收敛。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:13:22+00:00
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
[R6KD-CX8aaS-GPU-14:12569] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:12569] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12574 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  12575 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  12576 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  12577 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  12578 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  12579 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  12580 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  12581 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  14817 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  14818 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  14819 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  14820 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  14821 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  14822 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  14823 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  14824 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    35.49    0.03    0.05       0    33.42    0.03    0.06       0
        2048           512     float     sum      -1    36.52    0.06    0.11       0    36.57    0.06    0.11       0
        4096          1024     float     sum      -1    41.59    0.10    0.18       0    40.92    0.10    0.19       0
        8192          2048     float     sum      -1    50.05    0.16    0.31       0    42.97    0.19    0.36       0
       16384          4096     float     sum      -1    45.23    0.36    0.68       0    43.70    0.37    0.70       0
       32768          8192     float     sum      -1    57.71    0.57    1.06       0    60.96    0.54    1.01       0
       65536         16384     float     sum      -1   104.39    0.63    1.18       0   106.14    0.62    1.16       0
      131072         32768     float     sum      -1   225.03    0.58    1.09       0   225.07    0.58    1.09       0
      262144         65536     float     sum      -1   399.15    0.66    1.23       0   414.55    0.63    1.19       0
      524288        131072     float     sum      -1   628.88    0.83    1.56       0   634.15    0.83    1.55       0
     1048576        262144     float     sum      -1  1261.54    0.83    1.56       0  1242.73    0.84    1.58       0
     2097152        524288     float     sum      -1   386.93    5.42   10.16       0   385.97    5.43   10.19       0
     4194304       1048576     float     sum      -1   710.64    5.90   11.07       0   710.45    5.90   11.07       0
     8388608       2097152     float     sum      -1  1387.48    6.05   11.34       0  8916.79    0.94    1.76       0
    16777216       4194304     float     sum      -1   955.41   17.56   32.93       0   959.51   17.49   32.78       0
    33554432       8388608     float     sum      -1  1596.69   21.01   39.40       0  1587.98   21.13   39.62       0
    67108864      16777216     float     sum      -1  2801.78   23.95   44.91       0  2804.72   23.93   44.86       0
   134217728      33554432     float     sum      -1  5243.34   25.60   48.00       0  5254.19   25.54   47.90       0
   268435456      67108864     float     sum      -1  10257.0   26.17   49.07       0  10245.4   26.20   49.13       0
   536870912     134217728     float     sum      -1  23521.4   22.82   42.80       0  20376.0   26.35   49.40       0
  1073741824     268435456     float     sum      -1  40579.7   26.46   49.61       0  42045.6   25.54   47.88       0
  2147483648     536870912     float     sum      -1  80996.5   26.51   49.71       0  81023.8   26.50   49.70       0
  4294967296    1073741824     float     sum      -1   161891   26.53   49.74       0   161842   26.54   49.76       0
  8589934592    2147483648     float     sum      -1   323879   26.52   49.73       0   323213   26.58   49.83       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 20.6323 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:15:41+00:00
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
[R6KD-CX8aaS-GPU-14:13048] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:13048] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  13053 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  13054 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  13055 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  13056 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  13057 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  13058 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  13059 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  13060 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  15371 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  15372 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  15373 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  15374 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  15375 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  15376 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  15377 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  15378 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    38.03    0.03    0.05       0    34.89    0.03    0.06       0
        2048           512     float     sum      -1    39.51    0.05    0.10       0    37.91    0.05    0.10       0
        4096          1024     float     sum      -1    42.21    0.10    0.18       0    41.43    0.10    0.19       0
        8192          2048     float     sum      -1    47.27    0.17    0.32       0    44.01    0.19    0.35       0
       16384          4096     float     sum      -1    49.85    0.33    0.62       0    42.91    0.38    0.72       0
       32768          8192     float     sum      -1    62.23    0.53    0.99       0    61.60    0.53    1.00       0
       65536         16384     float     sum      -1    97.94    0.67    1.25       0    99.88    0.66    1.23       0
      131072         32768     float     sum      -1   210.37    0.62    1.17       0   207.97    0.63    1.18       0
      262144         65536     float     sum      -1   389.98    0.67    1.26       0   389.18    0.67    1.26       0
      524288        131072     float     sum      -1   588.33    0.89    1.67       0   602.00    0.87    1.63       0
     1048576        262144     float     sum      -1  1186.48    0.88    1.66       0  1172.83    0.89    1.68       0
     2097152        524288     float     sum      -1   409.22    5.12    9.61       0   403.35    5.20    9.75       0
     4194304       1048576     float     sum      -1   743.30    5.64   10.58       0  1209.76    3.47    6.50       0
     8388608       2097152     float     sum      -1  21804.9    0.38    0.72       0  32107.1    0.26    0.49       0
    16777216       4194304     float     sum      -1  1234.24   13.59   25.49       0  2308.78    7.27   13.63       0
    33554432       8388608     float     sum      -1  1675.08   20.03   37.56       0  5559.00    6.04   11.32       0
    67108864      16777216     float     sum      -1  2878.67   23.31   43.71       0  3264.58   20.56   38.54       0
   134217728      33554432     float     sum      -1  5313.97   25.26   47.36       0  5829.35   23.02   43.17       0
   268435456      67108864     float     sum      -1  14340.8   18.72   35.10       0  12022.3   22.33   41.87       0
   536870912     134217728     float     sum      -1  20450.1   26.25   49.22       0  22459.3   23.90   44.82       0
  1073741824     268435456     float     sum      -1  40660.9   26.41   49.51       0  40575.5   26.46   49.62       0
  2147483648     536870912     float     sum      -1  81058.6   26.49   49.67       0  81077.7   26.49   49.66       0
  4294967296    1073741824     float     sum      -1   161812   26.54   49.77       0   161750   26.55   49.79       0
  8589934592    2147483648     float     sum      -1   323437   26.56   49.80       0   322998   26.59   49.86       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.4536 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:18:00+00:00
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
[R6KD-CX8aaS-GPU-14:13527] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:13527] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  13532 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  13533 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  13534 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  13535 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  13536 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  13537 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  13538 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  13539 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  15925 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  15926 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  15927 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  15928 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  15929 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  15930 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  15931 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  15932 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    36.38    0.03    0.05       0    34.59    0.03    0.06       0
        2048           512     float     sum      -1    38.25    0.05    0.10       0    37.91    0.05    0.10       0
        4096          1024     float     sum      -1    41.97    0.10    0.18       0    41.90    0.10    0.18       0
        8192          2048     float     sum      -1    45.03    0.18    0.34       0    43.55    0.19    0.35       0
       16384          4096     float     sum      -1    47.22    0.35    0.65       0    45.13    0.36    0.68       0
       32768          8192     float     sum      -1    63.79    0.51    0.96       0    63.18    0.52    0.97       0
       65536         16384     float     sum      -1   103.88    0.63    1.18       0   103.54    0.63    1.19       0
      131072         32768     float     sum      -1   229.14    0.57    1.07       0   237.31    0.55    1.04       0
      262144         65536     float     sum      -1   420.94    0.62    1.17       0   425.63    0.62    1.15       0
      524288        131072     float     sum      -1   643.28    0.82    1.53       0   626.34    0.84    1.57       0
     1048576        262144     float     sum      -1  1330.45    0.79    1.48       0  1312.21    0.80    1.50       0
     2097152        524288     float     sum      -1   401.68    5.22    9.79       0   404.44    5.19    9.72       0
     4194304       1048576     float     sum      -1   736.95    5.69   10.67       0   731.09    5.74   10.76       0
     8388608       2097152     float     sum      -1  51138.6    0.16    0.31       0  22411.0    0.37    0.70       0
    16777216       4194304     float     sum      -1  1023.98   16.38   30.72       0  1026.61   16.34   30.64       0
    33554432       8388608     float     sum      -1  1661.47   20.20   37.87       0  1672.95   20.06   37.61       0
    67108864      16777216     float     sum      -1  2897.32   23.16   43.43       0  2882.22   23.28   43.66       0
   134217728      33554432     float     sum      -1  7023.72   19.11   35.83       0  9849.83   13.63   25.55       0
   268435456      67108864     float     sum      -1  11951.6   22.46   42.11       0  11459.6   23.42   43.92       0
   536870912     134217728     float     sum      -1  24322.4   22.07   41.39       0  24730.8   21.71   40.70       0
  1073741824     268435456     float     sum      -1  41056.4   26.15   49.04       0  43785.8   24.52   45.98       0
  2147483648     536870912     float     sum      -1  81031.3   26.50   49.69       0  81065.2   26.49   49.67       0
  4294967296    1073741824     float     sum      -1   164299   26.14   49.01       0   161830   26.54   49.76       0
  8589934592    2147483648     float     sum      -1   323220   26.58   49.83       0   323221   26.58   49.83       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.8688 
#
# Collective test concluded: all_reduce_perf
#
```

---

*对比表仅使用各 case 日志 **out-of-place** 列 **algbw**；CASE1 标题 “AN OFF” 对应 **AR OFF**。*
