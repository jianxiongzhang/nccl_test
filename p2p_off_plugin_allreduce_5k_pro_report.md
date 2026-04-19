# NCCL AllReduce 性能测试报告（P2P 关闭 + Spectrum-X 网络插件）

**报告主题：** `all_reduce_perf`，跨两台 **5K Pro** GPU 服务器，在 **`NCCL_P2P_DISABLE=1`** 且 **`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件 + 对应 `LD_LIBRARY_PATH`）条件下，对比 **AR OFF**、**AR ON / DDP OFF**、**AR ON / DDP ON** 的 **out-of-place algbw（GB/s）**。  
**基准：** 以 **AR OFF** 的 out-of-place algbw 为参照，其余两种配置给出绝对值及相对变化百分比（正数表示高于基准，负数表示低于基准）。

**测试日期：** 2026-04-19（日志时间戳为 UTC）

> **主机名说明：** 日志节点为 `R6KD-CX8aaS-GPU-14` / `R6KD-CX8aaS-GPU-15`；拓扑按测试计划记为两台 **5K Pro**。

---

## 1. 测试命令

三次运行除 NCCL IB 相关开关外保持一致；共性要点：

- **`NCCL_NET_PLUGIN=spcx`**
- **`NCCL_P2P_DISABLE=1`**
- **`NCCL_SHM_DISABLE=1`**，`**UCX_TLS=ib**`，`all_reduce_perf -b 1k -e 8G -f 2 -g 1`
- **`LD_LIBRARY_PATH`** 包含 NCCL 与 **HPC-X Spectrum-X 插件** 路径（见原始 `COMMAND:`）

### 1.1 AR OFF（基准）

```bash
-x NCCL_NET_PLUGIN=spcx ... \
-x NCCL_P2P_DISABLE=1 \
-x NCCL_IB_ADAPTIVE_ROUTING=0 \
-x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 \
/workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 AR ON / DDP OFF

将自适应路由打开，其余 IB 特性保持与 AR OFF 相同：

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
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**（Spectrum-X 插件；`LD_LIBRARY_PATH` 含 `nccl_spectrum-x_plugin/lib`） |
| P2P | **`NCCL_P2P_DISABLE=1`** |
| NCCL / CUDA | NCCL **2.30.3+cuda13.0**；nccl-tests **2.18.3** |
| 网络 | IB（`UCX_TLS=ib`），`NCCL_SOCKET_IFNAME=bond0`，多 HCA（`mlx5_*`） |
| 共享内存 | `NCCL_SHM_DISABLE=1` |

拓扑简述：**双机、每机 8×GPU，16 rank AllReduce**；跨机流量经 **bond0 / IB**，并由 **spcx** 插件参与网络路径/策略。

---

## 3. Out-of-place algbw 对比表（相对 AR OFF）

**Δ%** 计算：`((algbw_配置 − algbw_AR_OFF) / algbw_AR_OFF) × 100%`。极小数值行的百分比仅作形式参考。

| Size (B) | AR OFF algbw (基准) | AR ON DDP OFF algbw | vs 基准 Δ% | AR ON DDP ON algbw | vs 基准 Δ% |
|----------|---------------------|---------------------|------------|--------------------|------------|
| 1024 | 0.02 | 0.01 | −50.0% | 0.02 | 0.0% |
| 2048 | 0.04 | 0.03 | −25.0% | 0.03 | −25.0% |
| 4096 | 0.06 | 0.06 | 0.0% | 0.06 | 0.0% |
| 8192 | 0.11 | 0.10 | −9.1% | 0.10 | −9.1% |
| 16384 | 0.20 | 0.17 | −15.0% | 0.20 | 0.0% |
| 32768 | 0.39 | 0.38 | −2.6% | 0.37 | −5.1% |
| 65536 | 0.61 | 0.59 | −3.3% | 0.60 | −1.6% |
| 131072 | 1.35 | 1.32 | −2.2% | 1.31 | −3.0% |
| 262144 | 2.42 | 2.36 | −2.5% | 2.37 | −2.1% |
| 524288 | 3.86 | 3.77 | −2.3% | 3.53 | −8.6% |
| 1048576 | 5.33 | 4.94 | −7.3% | 5.19 | −2.6% |
| 2097152 | 6.09 | 6.54 | +7.4% | 6.45 | +5.9% |
| 4194304 | 6.57 | 7.40 | +12.6% | 7.10 | +8.1% |
| 8388608 | 18.21 | 18.01 | −1.1% | 18.03 | −1.0% |
| 16777216 | 17.29 | 16.77 | −3.0% | 1.97 | −88.6% |
| 33554432 | 16.18 | 7.51 | −53.6% | 15.42 | −4.7% |
| 67108864 | 16.01 | 15.25 | −4.8% | 12.43 | −22.4% |
| 134217728 | 20.84 | 13.67 | −34.4% | 13.66 | −34.5% |
| 268435456 | 20.62 | 13.18 | −36.1% | 13.98 | −32.2% |
| 536870912 | 18.21 | 18.46 | +1.4% | 18.56 | +1.9% |
| 1073741824 | 18.45 | 18.46 | +0.1% | 18.39 | −0.3% |
| 2147483648 | 18.54 | 18.62 | +0.4% | 18.57 | +0.2% |
| 4294967296 | 18.59 | 18.65 | +0.3% | 18.53 | −0.3% |
| 8589934592 | 18.71 | 18.61 | −0.5% | 18.52 | −1.0% |

**摘要观察（out-of-place algbw）：**

- **约 2–8 MiB**：**AR ON** 两种配置相对基准多为 **小幅正增益**（最高约 **+13%** 在 **4 MiB**）。
- **中消息（约 16 MiB–256 MiB）**：**AR ON / DDP OFF** 在 **32 MiB** 出现 **约 −54%** 断崖；**128 MiB–256 MiB** 亦显著低于基准。**AR ON / DDP ON** 在 **16 MiB** 的 out-of-place 点 **1.97 GB/s** 相对 **17.29 GB/s** 异常偏低（约 **−89%**），强烈建议 **复测** 并对照 in-place 与 `NCCL_DEBUG`。
- **大消息（约 512 MiB–8 GiB）**：三种配置 **接近持平**，差异多在 **±2%** 以内。

---

## 4. 原始数据（日志摘录）

```
========================================
CASE1: AN OFF
Started: 2026-04-19T11:20:18+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
[R6KD-CX8aaS-GPU-14:14005] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:14005] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  14010 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  14011 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  14012 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  14013 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  14014 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  14015 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  14016 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  14017 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  16479 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  16480 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  16481 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  16482 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  16483 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  16484 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  16485 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  16486 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    61.09    0.02    0.03       0    53.35    0.02    0.04       0
        2048           512     float     sum      -1    57.54    0.04    0.07       0    57.20    0.04    0.07       0
        4096          1024     float     sum      -1    63.91    0.06    0.12       0    62.97    0.07    0.12       0
        8192          2048     float     sum      -1    73.85    0.11    0.21       0    68.20    0.12    0.23       0
       16384          4096     float     sum      -1    82.55    0.20    0.37       0    69.05    0.24    0.44       0
       32768          8192     float     sum      -1    83.33    0.39    0.74       0    83.78    0.39    0.73       0
       65536         16384     float     sum      -1   107.75    0.61    1.14       0   105.44    0.62    1.17       0
      131072         32768     float     sum      -1    96.86    1.35    2.54       0    97.65    1.34    2.52       0
      262144         65536     float     sum      -1   108.14    2.42    4.55       0   111.79    2.34    4.40       0
      524288        131072     float     sum      -1   135.87    3.86    7.23       0   135.86    3.86    7.24       0
     1048576        262144     float     sum      -1   196.78    5.33    9.99       0   199.19    5.26    9.87       0
     2097152        524288     float     sum      -1   344.51    6.09   11.41       0   346.80    6.05   11.34       0
     4194304       1048576     float     sum      -1   638.57    6.57   12.32       0  1638.94    2.56    4.80       0
     8388608       2097152     float     sum      -1   460.61   18.21   34.15       0   460.46   18.22   34.16       0
    16777216       4194304     float     sum      -1   970.26   17.29   32.42       0   860.87   19.49   36.54       0
    33554432       8388608     float     sum      -1  2073.48   16.18   30.34       0  1996.05   16.81   31.52       0
    67108864      16777216     float     sum      -1  4192.51   16.01   30.01       0  4275.32   15.70   29.43       0
   134217728      33554432     float     sum      -1  6438.86   20.84   39.08       0  6315.36   21.25   39.85       0
   268435456      67108864     float     sum      -1  13016.7   20.62   38.67       0  13241.2   20.27   38.01       0
   536870912     134217728     float     sum      -1  29476.9   18.21   34.15       0  28970.9   18.53   34.75       0
  1073741824     268435456     float     sum      -1  58201.5   18.45   34.59       0  57551.2   18.66   34.98       0
  2147483648     536870912     float     sum      -1   115855   18.54   34.75       0   114677   18.73   35.11       0
  4294967296    1073741824     float     sum      -1   230992   18.59   34.86       0   228743   18.78   35.21       0
  8589934592    2147483648     float     sum      -1   459108   18.71   35.08       0   459602   18.69   35.04       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.8413 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T11:22:55+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
[R6KD-CX8aaS-GPU-14:14488] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:14488] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  14498 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  14499 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  14500 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  14501 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  14502 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  14503 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  14504 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  14505 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  17035 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  17036 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  17037 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  17038 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  17039 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  17040 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  17041 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  17042 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    68.96    0.01    0.03       0    58.64    0.02    0.03       0
        2048           512     float     sum      -1    61.75    0.03    0.06       0    60.52    0.03    0.06       0
        4096          1024     float     sum      -1    69.15    0.06    0.11       0    67.95    0.06    0.11       0
        8192          2048     float     sum      -1    78.87    0.10    0.19       0    74.30    0.11    0.21       0
       16384          4096     float     sum      -1    96.58    0.17    0.32       0    72.63    0.23    0.42       0
       32768          8192     float     sum      -1    86.58    0.38    0.71       0    85.59    0.38    0.72       0
       65536         16384     float     sum      -1   111.05    0.59    1.11       0   108.49    0.60    1.13       0
      131072         32768     float     sum      -1    99.49    1.32    2.47       0  3610.92    0.04    0.07       0
      262144         65536     float     sum      -1   110.85    2.36    4.43       0   126.88    2.07    3.87       0
      524288        131072     float     sum      -1   138.92    3.77    7.08       0   138.30    3.79    7.11       0
     1048576        262144     float     sum      -1   212.32    4.94    9.26       0   199.68    5.25    9.85       0
     2097152        524288     float     sum      -1   320.54    6.54   12.27       0  2551.37    0.82    1.54       0
     4194304       1048576     float     sum      -1   566.51    7.40   13.88       0   553.41    7.58   14.21       0
     8388608       2097152     float     sum      -1   465.74   18.01   33.77       0   465.02   18.04   33.82       0
    16777216       4194304     float     sum      -1  1000.65   16.77   31.44       0   819.06   20.48   38.41       0
    33554432       8388608     float     sum      -1  4470.03    7.51   14.07       0  2921.68   11.48   21.53       0
    67108864      16777216     float     sum      -1  4399.21   15.25   28.60       0  5119.76   13.11   24.58       0
   134217728      33554432     float     sum      -1  9820.63   13.67   25.63       0  6880.38   19.51   36.58       0
   268435456      67108864     float     sum      -1  20370.2   13.18   24.71       0  13659.9   19.65   36.85       0
   536870912     134217728     float     sum      -1  29081.6   18.46   34.61       0  29212.8   18.38   34.46       0
  1073741824     268435456     float     sum      -1  58172.0   18.46   34.61       0  58634.8   18.31   34.34       0
  2147483648     536870912     float     sum      -1   115344   18.62   34.91       0   116457   18.44   34.58       0
  4294967296    1073741824     float     sum      -1   230304   18.65   34.97       0   231185   18.58   34.83       0
  8589934592    2147483648     float     sum      -1   461512   18.61   34.90       0   461327   18.62   34.91       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 16.424 
#
# Collective test concluded: all_reduce_perf
#

========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T11:25:26+00:00
========================================
COMMAND:
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl \^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1

--------------------------------------------------------------------------
Two mutually-exclusive MCA variables were specified.  This can result
in undefined behavior, such as ignoring the components that the MCA
variables are supposed to affect.

  1st MCA variable: btl_tcp_if_include
    Source of value: environment
  2nd MCA variable: btl_tcp_if_exclude
    Source of value: file (/opt/hpcx/ompi/etc/openmpi-mca-params.conf:97)
--------------------------------------------------------------------------
[R6KD-CX8aaS-GPU-14:14974] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-14:14974] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid  14979 on R6KD-CX8aaS-GPU-14 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  1 Group  0 Pid  14980 on R6KD-CX8aaS-GPU-14 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank  2 Group  0 Pid  14981 on R6KD-CX8aaS-GPU-14 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank  3 Group  0 Pid  14982 on R6KD-CX8aaS-GPU-14 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank  4 Group  0 Pid  14983 on R6KD-CX8aaS-GPU-14 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank  5 Group  0 Pid  14984 on R6KD-CX8aaS-GPU-14 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank  6 Group  0 Pid  14985 on R6KD-CX8aaS-GPU-14 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank  7 Group  0 Pid  14986 on R6KD-CX8aaS-GPU-14 device  7 [0000:f9:00] NVIDIA Graphics Device
#  Rank  8 Group  0 Pid  17591 on R6KD-CX8aaS-GPU-15 device  0 [0000:06:00] NVIDIA Graphics Device
#  Rank  9 Group  0 Pid  17592 on R6KD-CX8aaS-GPU-15 device  1 [0000:09:00] NVIDIA Graphics Device
#  Rank 10 Group  0 Pid  17593 on R6KD-CX8aaS-GPU-15 device  2 [0000:76:00] NVIDIA Graphics Device
#  Rank 11 Group  0 Pid  17594 on R6KD-CX8aaS-GPU-15 device  3 [0000:79:00] NVIDIA Graphics Device
#  Rank 12 Group  0 Pid  17595 on R6KD-CX8aaS-GPU-15 device  4 [0000:86:00] NVIDIA Graphics Device
#  Rank 13 Group  0 Pid  17596 on R6KD-CX8aaS-GPU-15 device  5 [0000:89:00] NVIDIA Graphics Device
#  Rank 14 Group  0 Pid  17597 on R6KD-CX8aaS-GPU-15 device  6 [0000:f6:00] NVIDIA Graphics Device
#  Rank 15 Group  0 Pid  17598 on R6KD-CX8aaS-GPU-15 device  7 [0000:f9:00] NVIDIA Graphics Device
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    65.40    0.02    0.03       0    58.72    0.02    0.03       0
        2048           512     float     sum      -1    62.63    0.03    0.06       0    61.47    0.03    0.06       0
        4096          1024     float     sum      -1    68.96    0.06    0.11       0    68.12    0.06    0.11       0
        8192          2048     float     sum      -1    78.91    0.10    0.19       0    72.92    0.11    0.21       0
       16384          4096     float     sum      -1    82.80    0.20    0.37       0    73.74    0.22    0.42       0
       32768          8192     float     sum      -1    87.45    0.37    0.70       0    85.12    0.38    0.72       0
       65536         16384     float     sum      -1   109.60    0.60    1.12       0   107.80    0.61    1.14       0
      131072         32768     float     sum      -1   100.40    1.31    2.45       0    99.70    1.31    2.46       0
      262144         65536     float     sum      -1   110.48    2.37    4.45       0   119.24    2.20    4.12       0
      524288        131072     float     sum      -1   148.41    3.53    6.62       0   137.07    3.82    7.17       0
     1048576        262144     float     sum      -1   202.07    5.19    9.73       0   201.85    5.19    9.74       0
     2097152        524288     float     sum      -1   325.29    6.45   12.09       0   324.69    6.46   12.11       0
     4194304       1048576     float     sum      -1   590.61    7.10   13.32       0   556.15    7.54   14.14       0
     8388608       2097152     float     sum      -1   465.37   18.03   33.80       0   464.85   18.05   33.84       0
    16777216       4194304     float     sum      -1  8506.65    1.97    3.70       0   854.17   19.64   36.83       0
    33554432       8388608     float     sum      -1  2175.51   15.42   28.92       0  2051.15   16.36   30.67       0
    67108864      16777216     float     sum      -1  5400.57   12.43   23.30       0  4806.61   13.96   26.18       0
   134217728      33554432     float     sum      -1  9824.13   13.66   25.62       0  6516.05   20.60   38.62       0
   268435456      67108864     float     sum      -1  19197.0   13.98   26.22       0  13651.4   19.66   36.87       0
   536870912     134217728     float     sum      -1  28932.3   18.56   34.79       0  29169.4   18.41   34.51       0
  1073741824     268435456     float     sum      -1  58394.6   18.39   34.48       0  58702.8   18.29   34.30       0
  2147483648     536870912     float     sum      -1   115661   18.57   34.81       0   116102   18.50   34.68       0
  4294967296    1073741824     float     sum      -1   231744   18.53   34.75       0   231531   18.55   34.78       0
  8589934592    2147483648     float     sum      -1   463733   18.52   34.73       0   463923   18.52   34.72       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 16.5583 
#
# Collective test concluded: all_reduce_perf
#
```

---

*对比表仅使用各 case 日志 **out-of-place** 列 **algbw**；CASE1 标题 “AN OFF” 对应 **AR OFF**。*
