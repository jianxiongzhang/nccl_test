# 6KD NCCL `all_reduce_perf` 测试报告（master / NCCL 2.30，`NCCL_P2P_DISABLE=1`，`NCCL_NET_PLUGIN=spcx`）

**报告日期**：2026-04-19  
**测试对象**：跨两台 6KD（RTX 6000D）服务器的 NCCL `all_reduce_perf`，对比 **AR OFF**、**AR ON + DDP OFF**、**AR ON + DDP ON** 三种配置；网络侧启用 **Spectrum-X NCCL 插件**（**`NCCL_NET_PLUGIN=spcx`**），并显式将 Spectrum-X 插件库加入 **`LD_LIBRARY_PATH`**。  
**指标**：仅统计 **out-of-place** 列中的 **algbw（GB/s）**；相对 **AR OFF** 的差异以百分比表示：\(\Delta\% = 100 \times (\mathrm{algbw}_\mathrm{case} - \mathrm{algbw}_\mathrm{AR\ OFF}) / \mathrm{algbw}_\mathrm{AR\ OFF}\)。正值表示更快（更高带宽），负值表示更慢。

---

## 1. 测试命令

三份用例除 InfiniBand 自适应路由与 DDP 相关开关外，其余 MPI / NCCL / `all_reduce_perf` 参数保持一致；均包含 **`NCCL_NET_PLUGIN=spcx`**、**`NCCL_P2P_DISABLE=1`** 以及指向 HPC-X 包内 **`nccl_spectrum-x_plugin/lib`** 的 **`LD_LIBRARY_PATH`** 前缀。

### 1.1 CASE1：AR OFF（基准，`NCCL_IB_ADAPTIVE_ROUTING=0`，DDP 相关关闭）

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=0 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.2 CASE2：AR ON，DDP OFF

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=0 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=0 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

### 1.3 CASE3：AR ON，DDP ON

```bash
mpirun --allow-run-as-root --hostfile /workspace/hostfile -npernode 8 --bind-to none --map-by slot -mca plm_rsh_args -p\ 3456 --mca pml ucx --mca btl ^openib -mca btl_tcp_if_include bond0 -mca oob_tcp_if_include bond0 -x NCCL_NET_PLUGIN=spcx -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=GRAPH\,NET\,TUNING -x NCCL_SHM_DISABLE=1 -x UCX_TLS=ib -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5_2\,mlx5_3\,mlx5_0\,mlx5_1\,mlx5_6\,mlx5_7\,mlx5_4\,mlx5_5 -x NCCL_NETDEVS_POLICY=MAX:1 -x NCCL_BUFFSIZE=16777216 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_P2P_DISABLE=1 -x LD_LIBRARY_PATH=/workspace/nccl/build/lib:/workspace/hpcx-v2.26-gcc-doca_ofed-ubuntu22.04-cuda13-x86_64/nccl_spectrum-x_plugin/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 -x NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 -x NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 /workspace/nccl-tests/build/all_reduce_perf -b 1k -e 8G -f 2 -g 1
```

---

## 2. 测试环境与拓扑

| 项目 | 说明 |
|------|------|
| 节点 | 两台服务器：`R6KD-CX8aaS-GPU-11`、`R6KD-CX8aaS-GPU-12`（日志中的 **6KD** 平台） |
| GPU | 每节点 8× **NVIDIA RTX 6000D**；MPI 总 rank 数 **16**（`npernode 8` × 2 节点） |
| 集合通信 | **NCCL** `all_reduce_perf`（`nccl-tests` 2.18.3 头文件/库版本 23003；运行时 **NCCL 2.30.3+cuda13.0**） |
| 网络插件 | **`NCCL_NET_PLUGIN=spcx`**；`LD_LIBRARY_PATH` 包含 **`.../hpcx-v2.26-.../nccl_spectrum-x_plugin/lib`** |
| 其他 | **`NCCL_P2P_DISABLE=1`**；`NCCL_SHM_DISABLE=1`；UCX `UCX_TLS=ib`；`NCCL_SOCKET_IFNAME=bond0`；`NCCL_IB_HCA` 列出 mlx5 设备 |
| MPI | Open MPI + **PML UCX**；日志曾提示 `btl_tcp_if_include` 与 `btl_tcp_if_exclude` 同时生效（来自 HPC-X 默认配置），属环境告警 |

---

## 3. Out-of-place `algbw` 对比（相对 AR OFF）

| size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw | vs AR OFF | AR ON DDP ON algbw | vs AR OFF |
|---------:|--------------------:|--------------------:|----------:|-------------------:|----------:|
| 1024 | 0.02 | 0.02 | 0.00% | 0.02 | 0.00% |
| 2048 | 0.04 | 0.03 | -25.00% | 0.03 | -25.00% |
| 4096 | 0.06 | 0.06 | 0.00% | 0.06 | 0.00% |
| 8192 | 0.11 | 0.11 | 0.00% | 0.11 | 0.00% |
| 16384 | 0.19 | 0.19 | 0.00% | 0.20 | +5.26% |
| 32768 | 0.40 | 0.39 | -2.50% | 0.39 | -2.50% |
| 65536 | 0.62 | 0.61 | -1.61% | 0.61 | -1.61% |
| 131072 | 1.36 | 1.33 | -2.21% | 1.33 | -2.21% |
| 262144 | 2.42 | 2.40 | -0.83% | 2.41 | -0.41% |
| 524288 | 3.89 | 3.89 | 0.00% | 3.61 | -7.20% |
| 1048576 | 5.22 | 4.91 | -5.94% | 5.01 | -4.02% |
| 2097152 | 6.50 | 6.47 | -0.46% | 6.42 | -1.23% |
| 4194304 | 7.37 | 7.34 | -0.41% | 7.36 | -0.14% |
| 8388608 | 17.30 | 17.12 | -1.04% | 17.35 | +0.29% |
| 16777216 | 19.68 | 20.31 | +3.20% | 20.60 | +4.67% |
| 33554432 | 15.53 | 19.36 | +24.66% | 16.46 | +5.99% |
| 67108864 | 15.75 | 15.97 | +1.40% | 15.89 | +0.89% |
| 134217728 | 22.09 | 21.51 | -2.63% | 21.49 | -2.72% |
| 268435456 | 20.81 | 20.53 | -1.35% | 20.59 | -1.06% |
| 536870912 | 18.58 | 18.10 | -2.58% | 18.08 | -2.69% |
| 1073741824 | 18.45 | 18.34 | -0.60% | 18.17 | -1.52% |
| 2147483648 | 18.64 | 18.43 | -1.13% | 18.47 | -0.91% |
| 4294967296 | 18.66 | 18.54 | -0.64% | 18.26 | -2.14% |
| 8589934592 | 18.82 | 18.50 | -1.70% | 18.57 | -1.33% |

**简要观察（out-of-place algbw）**

- **32MB 步长**：相对 AR OFF，**AR ON DDP OFF** 的 **algbw 提升幅度最大**（约 **+24.7%**）；**AR ON DDP ON** 在该步长亦 **高于** 基准（约 **+6.0%**），与两侧用例在 **16MB** 附近整体抬升相一致。
- **512KB–1MB**：相对基准多为 **负向或接近持平**；其中 **512KB** 在 **AR ON DDP ON** 下约 **-7.2%**。
- **≥128MB**：三种配置整体落在 **~18–22 GB/s** 量级；多数大步长上 **AR ON** 相对基准略 **低**（约 **-0.6% 至 -2.7%**），**8GB** 步长上 **AR ON DDP ON** 的 **4G** 点相对基准约 **-2.1%**。

---

## 4. 原始数据附录

### 4.1 CASE1：AR OFF

```
========================================
CASE1: AN OFF
Started: 2026-04-19T03:15:39+00:00
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
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   4144 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   4145 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   4146 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   4147 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   4148 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   4149 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   4150 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   4151 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   4769 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   4770 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   4771 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   4772 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   4773 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   4774 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   4775 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   4776 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:04139] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:04139] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    61.99    0.02    0.03       0    53.35    0.02    0.04       0
        2048           512     float     sum      -1    57.34    0.04    0.07       0    59.50    0.03    0.06       0
        4096          1024     float     sum      -1    64.88    0.06    0.12       0    65.46    0.06    0.12       0
        8192          2048     float     sum      -1    75.29    0.11    0.20       0    70.31    0.12    0.22       0
       16384          4096     float     sum      -1    85.43    0.19    0.36       0    71.17    0.23    0.43       0
       32768          8192     float     sum      -1    82.22    0.40    0.75       0    84.18    0.39    0.73       0
       65536         16384     float     sum      -1   105.63    0.62    1.16       0   102.91    0.64    1.19       0
      131072         32768     float     sum      -1    96.05    1.36    2.56       0    95.84    1.37    2.56       0
      262144         65536     float     sum      -1   108.45    2.42    4.53       0   110.62    2.37    4.44       0
      524288        131072     float     sum      -1   134.64    3.89    7.30       0   136.28    3.85    7.21       0
     1048576        262144     float     sum      -1   200.81    5.22    9.79       0   191.31    5.48   10.28       0
     2097152        524288     float     sum      -1   322.53    6.50   12.19       0   329.32    6.37   11.94       0
     4194304       1048576     float     sum      -1   569.48    7.37   13.81       0   565.13    7.42   13.92       0
     8388608       2097152     float     sum      -1   484.95   17.30   32.43       0   478.42   17.53   32.88       0
    16777216       4194304     float     sum      -1   852.45   19.68   36.90       0   836.77   20.05   37.59       0
    33554432       8388608     float     sum      -1  2161.15   15.53   29.11       0  1755.92   19.11   35.83       0
    67108864      16777216     float     sum      -1  4259.65   15.75   29.54       0  3892.54   17.24   32.33       0
   134217728      33554432     float     sum      -1  6074.68   22.09   41.43       0  5972.13   22.47   42.14       0
   268435456      67108864     float     sum      -1  12901.0   20.81   39.01       0  13007.8   20.64   38.69       0
   536870912     134217728     float     sum      -1  28896.0   18.58   34.84       0  28750.3   18.67   35.01       0
  1073741824     268435456     float     sum      -1  58211.5   18.45   34.59       0  56858.7   18.88   35.41       0
  2147483648     536870912     float     sum      -1   115201   18.64   34.95       0   114760   18.71   35.09       0
  4294967296    1073741824     float     sum      -1   230205   18.66   34.98       0   229832   18.69   35.04       0
  8589934592    2147483648     float     sum      -1   456414   18.82   35.29       0   456421   18.82   35.29       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.4247 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.2 CASE2：AR ON DDP OFF

```
========================================
CASE2: AR ON DDP OFF
Started: 2026-04-19T03:17:34+00:00
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
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   4624 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   4625 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   4626 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   4627 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   4628 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   4629 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   4630 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   4631 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   5325 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   5326 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   5327 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   5328 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   5329 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   5330 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   5331 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   5332 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:04619] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:04619] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    63.21    0.02    0.03       0    56.17    0.02    0.03       0
        2048           512     float     sum      -1    60.08    0.03    0.06       0    58.86    0.03    0.07       0
        4096          1024     float     sum      -1    67.05    0.06    0.11       0    66.22    0.06    0.12       0
        8192          2048     float     sum      -1    76.71    0.11    0.20       0    70.32    0.12    0.22       0
       16384          4096     float     sum      -1    86.40    0.19    0.36       0    71.83    0.23    0.43       0
       32768          8192     float     sum      -1    85.07    0.39    0.72       0    83.69    0.39    0.73       0
       65536         16384     float     sum      -1   107.38    0.61    1.14       0   106.96    0.61    1.15       0
      131072         32768     float     sum      -1    98.33    1.33    2.50       0    98.13    1.34    2.50       0
      262144         65536     float     sum      -1   109.37    2.40    4.49       0   113.18    2.32    4.34       0
      524288        131072     float     sum      -1   134.77    3.89    7.29       0   142.89    3.67    6.88       0
     1048576        262144     float     sum      -1   213.61    4.91    9.20       0   201.62    5.20    9.75       0
     2097152        524288     float     sum      -1   324.05    6.47   12.13       0   323.52    6.48   12.15       0
     4194304       1048576     float     sum      -1   571.18    7.34   13.77       0   578.90    7.25   13.59       0
     8388608       2097152     float     sum      -1   489.93   17.12   32.10       0   485.11   17.29   32.42       0
    16777216       4194304     float     sum      -1   825.95   20.31   38.09       0   818.91   20.49   38.41       0
    33554432       8388608     float     sum      -1  1733.43   19.36   36.29       0  1709.38   19.63   36.81       0
    67108864      16777216     float     sum      -1  4201.62   15.97   29.95       0  4084.47   16.43   30.81       0
   134217728      33554432     float     sum      -1  6238.86   21.51   40.34       0  6187.31   21.69   40.67       0
   268435456      67108864     float     sum      -1  13077.9   20.53   38.49       0  13092.7   20.50   38.44       0
   536870912     134217728     float     sum      -1  29660.5   18.10   33.94       0  29327.4   18.31   34.32       0
  1073741824     268435456     float     sum      -1  58538.4   18.34   34.39       0  58633.7   18.31   34.34       0
  2147483648     536870912     float     sum      -1   116523   18.43   34.56       0   116893   18.37   34.45       0
  4294967296    1073741824     float     sum      -1   231684   18.54   34.76       0   231697   18.54   34.76       0
  8589934592    2147483648     float     sum      -1   464397   18.50   34.68       0   463607   18.53   34.74       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.3696 
#
# Collective test concluded: all_reduce_perf
#
```

### 4.3 CASE3：AR ON DDP ON

```
========================================
CASE3: AR ON DDP ON
Started: 2026-04-19T03:19:24+00:00
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
# nccl-tests version 2.18.3 nccl-headers=23003 nccl-library=23003
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid   5104 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid   5105 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid   5106 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid   5107 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid   5108 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid   5109 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid   5110 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid   5111 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid   5881 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid   5882 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid   5883 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid   5884 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid   5885 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid   5886 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid   5888 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid   5889 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.3+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
[R6KD-CX8aaS-GPU-11:05099] 15 more processes have sent help message help-mca-var.txt / mutually-exclusive-vars
[R6KD-CX8aaS-GPU-11:05099] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
        1024           256     float     sum      -1    61.21    0.02    0.03       0    57.47    0.02    0.03       0
        2048           512     float     sum      -1    59.33    0.03    0.06       0    58.82    0.03    0.07       0
        4096          1024     float     sum      -1    66.24    0.06    0.12       0    66.46    0.06    0.12       0
        8192          2048     float     sum      -1    75.41    0.11    0.20       0    70.11    0.12    0.22       0
       16384          4096     float     sum      -1    82.96    0.20    0.37       0    71.97    0.23    0.43       0
       32768          8192     float     sum      -1    84.76    0.39    0.72       0    83.71    0.39    0.73       0
       65536         16384     float     sum      -1   108.19    0.61    1.14       0   105.08    0.62    1.17       0
      131072         32768     float     sum      -1    98.54    1.33    2.49       0    98.77    1.33    2.49       0
      262144         65536     float     sum      -1   108.96    2.41    4.51       0   111.50    2.35    4.41       0
      524288        131072     float     sum      -1   145.08    3.61    6.78       0   137.15    3.82    7.17       0
     1048576        262144     float     sum      -1   209.25    5.01    9.40       0   200.66    5.23    9.80       0
     2097152        524288     float     sum      -1   326.71    6.42   12.04       0   321.06    6.53   12.25       0
     4194304       1048576     float     sum      -1   569.64    7.36   13.81       0   556.10    7.54   14.14       0
     8388608       2097152     float     sum      -1   483.50   17.35   32.53       0   479.88   17.48   32.78       0
    16777216       4194304     float     sum      -1   814.39   20.60   38.63       0   832.66   20.15   37.78       0
    33554432       8388608     float     sum      -1  2039.12   16.46   30.85       0  2118.00   15.84   29.70       0
    67108864      16777216     float     sum      -1  4224.32   15.89   29.79       0  4168.52   16.10   30.19       0
   134217728      33554432     float     sum      -1  6244.33   21.49   40.30       0  6167.99   21.76   40.80       0
   268435456      67108864     float     sum      -1  13036.7   20.59   38.61       0  13054.7   20.56   38.55       0
   536870912     134217728     float     sum      -1  29702.3   18.08   33.89       0  29361.8   18.28   34.28       0
  1073741824     268435456     float     sum      -1  59104.3   18.17   34.06       0  58108.2   18.48   34.65       0
  2147483648     536870912     float     sum      -1   116275   18.47   34.63       0   116347   18.46   34.61       0
  4294967296    1073741824     float     sum      -1   235199   18.26   34.24       0   230650   18.62   34.91       0
  8589934592    2147483648     float     sum      -1   462635   18.57   34.81       0   463208   18.54   34.77       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 18.126 
#
# Collective test concluded: all_reduce_perf
#
```
