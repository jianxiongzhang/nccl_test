# NCCL `run_zjx.sh` 对比报告（out-of-place algbw）
**日期：** 2026-04-15

**主机：** `ssh R6KD-CX8aaS-GPU-11 "/root/xuyj/run_zjx.sh"`（各次在远端 `export` 后执行）

**说明：** 你原文第三次写为「AR ON **DDF** ON」，本报告按 **AR ON DDP ON** 命名（与 `OOO_RQ` / `RECEIVER_SIDE_MATCHING` / `PREPOST_RECEIVE_WORK_REQUESTS` 均为 1 的配置一致）。

**delta：** 相对 **AR OFF** 的差异百分比：`((测量值 − AR OFF) / AR OFF) × 100%`（正为提升，负为下降）。

---

## 环境变量（三组）

| 标签 | NCCL_IB_ADAPTIVE_ROUTING | NCCL_IB_OOO_RQ | NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME | NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS |
|------|--------------------------|----------------|----------------------------------------|---------------------------------------|
| AR OFF（基准） | 0 | 0 | 0 | 0 |
| AR ON DDP OFF | 1 | 0 | 0 | 0 |
| AR ON DDP ON | 1 | 1 | 1 | 1 |

---

## AllReduce（`all_reduce_perf`）

| Size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw (GB/s) | delta | AR ON DDP ON algbw (GB/s) | delta |
|----------|---------------------|---------------------------|-------|--------------------------|-------|
| 1024 | 0.02 | 0.02 | +0.00% | 0.02 | +0.00% |
| 2048 | 0.03 | 0.03 | +0.00% | 0.03 | +0.00% |
| 4096 | 0.06 | 0.06 | +0.00% | 0.06 | +0.00% |
| 8192 | 0.11 | 0.11 | +0.00% | 0.11 | +0.00% |
| 16384 | 0.21 | 0.21 | +0.00% | 0.20 | -4.76% |
| 32768 | 0.38 | 0.39 | +2.63% | 0.39 | +2.63% |
| 65536 | 0.60 | 0.61 | +1.67% | 0.61 | +1.67% |
| 131072 | 1.30 | 1.32 | +1.54% | 1.32 | +1.54% |
| 262144 | 2.31 | 2.31 | +0.00% | 2.21 | -4.33% |
| 524288 | 3.64 | 3.61 | -0.82% | 3.56 | -2.20% |
| 1048576 | 4.53 | 4.51 | -0.44% | 4.40 | -2.87% |
| 2097152 | 5.51 | 5.54 | +0.54% | 5.53 | +0.36% |
| 4194304 | 6.29 | 6.26 | -0.48% | 6.37 | +1.27% |
| 8388608 | 17.21 | 17.05 | -0.93% | 17.27 | +0.35% |
| 16777216 | 20.34 | 20.53 | +0.93% | 20.29 | -0.25% |
| 33554432 | 19.54 | 19.73 | +0.97% | 19.42 | -0.61% |
| 67108864 | 20.23 | 20.14 | -0.44% | 19.93 | -1.48% |
| 134217728 | 25.10 | 25.08 | -0.08% | 25.06 | -0.16% |
| 268435456 | 25.23 | 25.15 | -0.32% | 25.16 | -0.28% |
| 536870912 | 25.15 | 25.08 | -0.28% | 25.06 | -0.36% |
| 1073741824 | 25.22 | 25.19 | -0.12% | 25.17 | -0.20% |
| 2147483648 | 25.34 | 25.35 | +0.04% | 25.31 | -0.12% |
| 4294967296 | 25.51 | 25.54 | +0.12% | 25.50 | -0.04% |
| 8589934592 | 25.64 | 25.66 | +0.08% | 25.65 | +0.04% |

**Avg bus bandwidth（AllReduce 段）：** AR OFF 21.8363，AR ON DDP OFF 21.821，AR ON DDP ON 21.8035。

---

## AllToAll（`alltoall_perf`）

| Size (B) | AR OFF algbw (GB/s) | AR ON DDP OFF algbw (GB/s) | delta | AR ON DDP ON algbw (GB/s) | delta |
|----------|---------------------|---------------------------|-------|--------------------------|-------|
| 1024 | 0.02 | 0.02 | +0.00% | 0.02 | +0.00% |
| 2048 | 0.04 | 0.04 | +0.00% | 0.04 | +0.00% |
| 4096 | 0.08 | 0.09 | +12.50% | 0.09 | +12.50% |
| 8192 | 0.17 | 0.17 | +0.00% | 0.17 | +0.00% |
| 16384 | 0.33 | 0.34 | +3.03% | 0.34 | +3.03% |
| 32768 | 0.66 | 0.68 | +3.03% | 0.67 | +1.52% |
| 65536 | 1.32 | 1.32 | +0.00% | 1.35 | +2.27% |
| 131072 | 2.48 | 2.58 | +4.03% | 2.46 | -0.81% |
| 262144 | 4.88 | 4.96 | +1.64% | 4.91 | +0.61% |
| 524288 | 8.15 | 8.24 | +1.10% | 8.21 | +0.74% |
| 1048576 | 11.69 | 11.59 | -0.86% | 11.39 | -2.57% |
| 2097152 | 20.24 | 20.80 | +2.77% | 20.38 | +0.69% |
| 4194304 | 25.27 | 25.26 | -0.04% | 24.97 | -1.19% |
| 8388608 | 33.16 | 32.65 | -1.54% | 32.49 | -2.02% |
| 16777216 | 39.52 | 39.52 | +0.00% | 39.41 | -0.28% |
| 33554432 | 45.23 | 45.20 | -0.07% | 45.20 | -0.07% |
| 67108864 | 47.52 | 48.03 | +1.07% | 47.99 | +0.99% |
| 134217728 | 48.52 | 48.57 | +0.10% | 48.60 | +0.16% |
| 268435456 | 48.50 | 48.69 | +0.39% | 48.67 | +0.35% |
| 536870912 | 48.61 | 48.78 | +0.35% | 48.78 | +0.35% |
| 1073741824 | 48.73 | 48.87 | +0.29% | 48.88 | +0.31% |
| 2147483648 | 48.73 | 48.91 | +0.37% | 48.84 | +0.23% |
| 4294967296 | 48.79 | 48.93 | +0.29% | 48.86 | +0.14% |
| 8589934592 | 48.84 | 48.96 | +0.25% | 48.91 | +0.14% |

**Avg bus bandwidth（AllToAll 段）：** AR OFF 22.7759，AR ON DDP OFF 22.8112，AR ON DDP ON 22.7233。

---

## 简要结论

- **AllReduce：** 大消息（约 128 MiB 及以上）三组 out-of-place algbw 均在约 **25.0～25.7 GB/s**，delta 多在 **±1%** 量级；中小消息存在一定波动。
- **AllToAll：** 大消息稳定在约 **48.5～49.0 GB/s**；个别中等消息尺寸下 **AR ON DDP ON** 相对 **AR OFF** 略低，见表。

---

## 附录：原始数据（完整 nccl-tests 终端输出）

### AR OFF

```text
# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1809753 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1809754 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1809755 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1809756 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1809757 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1809758 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1809759 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1809760 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1205361 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1205362 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1205363 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1205364 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1205365 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1205366 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1205367 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1205368 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    60.02    0.02    0.03       0    56.69    0.02    0.03       0
        2048           512     float     sum      -1    59.22    0.03    0.06       0    58.72    0.03    0.07       0
        4096          1024     float     sum      -1    64.55    0.06    0.12       0    64.50    0.06    0.12       0
        8192          2048     float     sum      -1    72.34    0.11    0.21       0    69.08    0.12    0.22       0
       16384          4096     float     sum      -1    76.95    0.21    0.40       0    69.53    0.24    0.44       0
       32768          8192     float     sum      -1    86.04    0.38    0.71       0    85.78    0.38    0.72       0
       65536         16384     float     sum      -1   109.64    0.60    1.12       0   106.99    0.61    1.15       0
      131072         32768     float     sum      -1   100.78    1.30    2.44       0    99.38    1.32    2.47       0
      262144         65536     float     sum      -1   113.54    2.31    4.33       0   117.11    2.24    4.20       0
      524288        131072     float     sum      -1   144.03    3.64    6.83       0   145.36    3.61    6.76       0
     1048576        262144     float     sum      -1   231.47    4.53    8.49       0   227.17    4.62    8.65       0
     2097152        524288     float     sum      -1   380.62    5.51   10.33       0   376.19    5.57   10.45       0
     4194304       1048576     float     sum      -1   666.92    6.29   11.79       0   652.50    6.43   12.05       0
     8388608       2097152     float     sum      -1   487.34   17.21   32.27       0   487.40   17.21   32.27       0
    16777216       4194304     float     sum      -1   824.71   20.34   38.14       0   827.89   20.27   38.00       0
    33554432       8388608     float     sum      -1  1717.46   19.54   36.63       0  1700.38   19.73   37.00       0
    67108864      16777216     float     sum      -1  3317.67   20.23   37.93       0  3342.67   20.08   37.64       0
   134217728      33554432     float     sum      -1  5346.37   25.10   47.07       0  5356.19   25.06   46.98       0
   268435456      67108864     float     sum      -1  10640.4   25.23   47.30       0  10688.1   25.12   47.09       0
   536870912     134217728     float     sum      -1  21344.7   25.15   47.16       0  21364.4   25.13   47.12       0
  1073741824     268435456     float     sum      -1  42574.0   25.22   47.29       0  42585.3   25.21   47.28       0
  2147483648     536870912     float     sum      -1  84763.5   25.34   47.50       0  84694.9   25.36   47.54       0
  4294967296    1073741824     float     sum      -1   168393   25.51   47.82       0   168571   25.48   47.77       0
  8589934592    2147483648     float     sum      -1   335016   25.64   48.08       0   335323   25.62   48.03       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.8363 
#
# Collective test concluded: all_reduce_perf
#

# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1809977 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1809978 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1809979 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1809980 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1809981 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1809982 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1809983 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1809985 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1205591 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1205592 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1205593 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1205594 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1205595 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1205596 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1205597 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1205598 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    62.98    0.02    0.02       0    48.34    0.02    0.02    N/A
        2048            32     float    none      -1    48.24    0.04    0.04       0    48.03    0.04    0.04    N/A
        4096            64     float    none      -1    48.96    0.08    0.08       0    48.22    0.08    0.08    N/A
        8192           128     float    none      -1    48.47    0.17    0.16       0    48.78    0.17    0.16    N/A
       16384           256     float    none      -1    49.38    0.33    0.31       0    48.82    0.34    0.31    N/A
       32768           512     float    none      -1    49.43    0.66    0.62       0    48.72    0.67    0.63    N/A
       65536          1024     float    none      -1    49.78    1.32    1.23       0    49.14    1.33    1.25    N/A
      131072          2048     float    none      -1    52.86    2.48    2.32       0    52.68    2.49    2.33    N/A
      262144          4096     float    none      -1    53.76    4.88    4.57       0    59.63    4.40    4.12    N/A
      524288          8192     float    none      -1    64.34    8.15    7.64       0    62.88    8.34    7.82    N/A
     1048576         16384     float    none      -1    89.67   11.69   10.96       0    89.54   11.71   10.98    N/A
     2097152         32768     float    none      -1   103.60   20.24   18.98       0   100.17   20.94   19.63    N/A
     4194304         65536     float    none      -1   166.01   25.27   23.69       0   162.84   25.76   24.15    N/A
     8388608        131072     float    none      -1   252.97   33.16   31.09       0   249.32   33.65   31.54    N/A
    16777216        262144     float    none      -1   424.56   39.52   37.05       0   408.84   41.04   38.47    N/A
    33554432        524288     float    none      -1   741.93   45.23   42.40       0   731.99   45.84   42.97    N/A
    67108864       1048576     float    none      -1  1412.34   47.52   44.55       0  1392.05   48.21   45.20    N/A
   134217728       2097152     float    none      -1  2766.42   48.52   45.48       0  2762.89   48.58   45.54    N/A
   268435456       4194304     float    none      -1  5534.52   48.50   45.47       0  5541.13   48.44   45.42    N/A
   536870912       8388608     float    none      -1  11045.0   48.61   45.57       0  11084.7   48.43   45.41    N/A
  1073741824      16777216     float    none      -1  22033.1   48.73   45.69       0  22116.1   48.55   45.52    N/A
  2147483648      33554432     float    none      -1  44071.4   48.73   45.68       0  44237.5   48.54   45.51    N/A
  4294967296      67108864     float    none      -1  88035.8   48.79   45.74       0  88429.4   48.57   45.53    N/A
  8589934592     134217728     float    none      -1   175891   48.84   45.78       0   176995   48.53   45.50    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.7759 
#
# Collective test concluded: alltoall_perf
#
```

### AR ON DDP OFF

```text
# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1808538 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1808539 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1808540 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1808541 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1808542 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1808543 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1808544 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1808545 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1204268 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1204269 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1204270 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1204271 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1204272 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1204273 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1204274 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1204275 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    58.64    0.02    0.03       0    55.77    0.02    0.03       0
        2048           512     float     sum      -1    59.64    0.03    0.06       0    59.00    0.03    0.07       0
        4096          1024     float     sum      -1    63.64    0.06    0.12       0    63.08    0.06    0.12       0
        8192          2048     float     sum      -1    72.24    0.11    0.21       0    68.23    0.12    0.23       0
       16384          4096     float     sum      -1    78.11    0.21    0.39       0    69.32    0.24    0.44       0
       32768          8192     float     sum      -1    85.11    0.39    0.72       0    83.86    0.39    0.73       0
       65536         16384     float     sum      -1   106.91    0.61    1.15       0   106.61    0.61    1.15       0
      131072         32768     float     sum      -1    99.34    1.32    2.47       0    99.40    1.32    2.47       0
      262144         65536     float     sum      -1   113.43    2.31    4.33       0   118.11    2.22    4.16       0
      524288        131072     float     sum      -1   145.04    3.61    6.78       0   146.24    3.59    6.72       0
     1048576        262144     float     sum      -1   232.30    4.51    8.46       0   226.65    4.63    8.67       0
     2097152        524288     float     sum      -1   378.52    5.54   10.39       0   384.47    5.45   10.23       0
     4194304       1048576     float     sum      -1   670.23    6.26   11.73       0   648.37    6.47   12.13       0
     8388608       2097152     float     sum      -1   491.97   17.05   31.97       0   487.41   17.21   32.27       0
    16777216       4194304     float     sum      -1   817.36   20.53   38.49       0   822.56   20.40   38.24       0
    33554432       8388608     float     sum      -1  1700.49   19.73   37.00       0  1719.82   19.51   36.58       0
    67108864      16777216     float     sum      -1  3332.56   20.14   37.76       0  3328.61   20.16   37.80       0
   134217728      33554432     float     sum      -1  5352.59   25.08   47.02       0  5407.90   24.82   46.54       0
   268435456      67108864     float     sum      -1  10675.4   25.15   47.15       0  10682.1   25.13   47.12       0
   536870912     134217728     float     sum      -1  21408.4   25.08   47.02       0  21361.1   25.13   47.12       0
  1073741824     268435456     float     sum      -1  42621.6   25.19   47.24       0  42649.8   25.18   47.20       0
  2147483648     536870912     float     sum      -1  84703.7   25.35   47.54       0  84690.3   25.36   47.54       0
  4294967296    1073741824     float     sum      -1   168146   25.54   47.89       0   168557   25.48   47.78       0
  8589934592    2147483648     float     sum      -1   334728   25.66   48.12       0   335528   25.60   48.00       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.821 
#
# Collective test concluded: all_reduce_perf
#

# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1808848 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1808849 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1808850 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1808851 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1808852 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1808853 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1808854 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1808855 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1204579 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1204580 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1204581 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1204582 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1204583 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1204584 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1204585 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1204586 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    57.97    0.02    0.02       0    53.22    0.02    0.02    N/A
        2048            32     float    none      -1    48.91    0.04    0.04       0    47.16    0.04    0.04    N/A
        4096            64     float    none      -1    47.58    0.09    0.08       0    48.53    0.08    0.08    N/A
        8192           128     float    none      -1    47.53    0.17    0.16       0    48.74    0.17    0.16    N/A
       16384           256     float    none      -1    48.14    0.34    0.32       0    47.81    0.34    0.32    N/A
       32768           512     float    none      -1    48.46    0.68    0.63       0    48.23    0.68    0.64    N/A
       65536          1024     float    none      -1    49.49    1.32    1.24       0    48.46    1.35    1.27    N/A
      131072          2048     float    none      -1    50.79    2.58    2.42       0    49.76    2.63    2.47    N/A
      262144          4096     float    none      -1    52.86    4.96    4.65       0    72.48    3.62    3.39    N/A
      524288          8192     float    none      -1    63.65    8.24    7.72       0    62.94    8.33    7.81    N/A
     1048576         16384     float    none      -1    90.44   11.59   10.87       0    88.79   11.81   11.07    N/A
     2097152         32768     float    none      -1   100.83   20.80   19.50       0    98.45   21.30   19.97    N/A
     4194304         65536     float    none      -1   166.06   25.26   23.68       0   162.58   25.80   24.19    N/A
     8388608        131072     float    none      -1   256.90   32.65   30.61       0   249.00   33.69   31.58    N/A
    16777216        262144     float    none      -1   424.51   39.52   37.05       0   408.85   41.04   38.47    N/A
    33554432        524288     float    none      -1   742.37   45.20   42.37       0   730.81   45.91   43.04    N/A
    67108864       1048576     float    none      -1  1397.10   48.03   45.03       0  1386.47   48.40   45.38    N/A
   134217728       2097152     float    none      -1  2763.24   48.57   45.54       0  2779.90   48.28   45.26    N/A
   268435456       4194304     float    none      -1  5512.81   48.69   45.65       0  5534.89   48.50   45.47    N/A
   536870912       8388608     float    none      -1  11006.4   48.78   45.73       0  11051.1   48.58   45.54    N/A
  1073741824      16777216     float    none      -1  21972.4   48.87   45.81       0  22090.9   48.61   45.57    N/A
  2147483648      33554432     float    none      -1  43903.5   48.91   45.86       0  44265.7   48.51   45.48    N/A
  4294967296      67108864     float    none      -1  87770.5   48.93   45.88       0  88500.5   48.53   45.50    N/A
  8589934592     134217728     float    none      -1   175430   48.96   45.90       0   177182   48.48   45.45    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.8112 
#
# Collective test concluded: alltoall_perf
#
```

### AR ON DDP ON

```text
# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1809167 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1809168 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1809169 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1809170 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1809171 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1809172 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1809173 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1809175 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1204827 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1204828 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1204829 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1204830 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1204831 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1204832 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1204833 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1204834 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    58.58    0.02    0.03       0    56.09    0.02    0.03       0
        2048           512     float     sum      -1    60.02    0.03    0.06       0    58.58    0.03    0.07       0
        4096          1024     float     sum      -1    63.95    0.06    0.12       0    63.37    0.06    0.12       0
        8192          2048     float     sum      -1    72.30    0.11    0.21       0    67.48    0.12    0.23       0
       16384          4096     float     sum      -1    80.59    0.20    0.38       0    69.56    0.24    0.44       0
       32768          8192     float     sum      -1    84.88    0.39    0.72       0    83.44    0.39    0.74       0
       65536         16384     float     sum      -1   107.62    0.61    1.14       0   107.75    0.61    1.14       0
      131072         32768     float     sum      -1    99.59    1.32    2.47       0    99.03    1.32    2.48       0
      262144         65536     float     sum      -1   118.68    2.21    4.14       0   120.52    2.18    4.08       0
      524288        131072     float     sum      -1   147.36    3.56    6.67       0   144.85    3.62    6.79       0
     1048576        262144     float     sum      -1   238.39    4.40    8.25       0   232.48    4.51    8.46       0
     2097152        524288     float     sum      -1   379.13    5.53   10.37       0   372.46    5.63   10.56       0
     4194304       1048576     float     sum      -1   658.15    6.37   11.95       0   653.95    6.41   12.03       0
     8388608       2097152     float     sum      -1   485.81   17.27   32.38       0   488.14   17.18   32.22       0
    16777216       4194304     float     sum      -1   827.04   20.29   38.04       0   827.23   20.28   38.03       0
    33554432       8388608     float     sum      -1  1727.48   19.42   36.42       0  1705.54   19.67   36.89       0
    67108864      16777216     float     sum      -1  3367.83   19.93   37.36       0  3323.90   20.19   37.86       0
   134217728      33554432     float     sum      -1  5356.04   25.06   46.99       0  5350.23   25.09   47.04       0
   268435456      67108864     float     sum      -1  10671.2   25.16   47.17       0  10668.2   25.16   47.18       0
   536870912     134217728     float     sum      -1  21421.9   25.06   46.99       0  21354.9   25.14   47.14       0
  1073741824     268435456     float     sum      -1  42654.8   25.17   47.20       0  42617.0   25.20   47.24       0
  2147483648     536870912     float     sum      -1  84859.4   25.31   47.45       0  84797.8   25.32   47.48       0
  4294967296    1073741824     float     sum      -1   168440   25.50   47.81       0   168340   25.51   47.84       0
  8589934592    2147483648     float     sum      -1   334826   25.65   48.10       0   334958   25.64   48.08       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.8035 
#
# Collective test concluded: all_reduce_perf
#

# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1809474 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1809475 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1809476 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1809477 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1809478 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1809479 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1809480 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1809481 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1205140 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1205141 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1205142 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1205143 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1205144 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1205145 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1205146 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1205147 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    63.16    0.02    0.02       0    47.38    0.02    0.02    N/A
        2048            32     float    none      -1    49.67    0.04    0.04       0    48.75    0.04    0.04    N/A
        4096            64     float    none      -1    48.00    0.09    0.08       0    47.31    0.09    0.08    N/A
        8192           128     float    none      -1    48.19    0.17    0.16       0    47.44    0.17    0.16    N/A
       16384           256     float    none      -1    48.19    0.34    0.32       0    47.86    0.34    0.32    N/A
       32768           512     float    none      -1    48.71    0.67    0.63       0    48.44    0.68    0.63    N/A
       65536          1024     float    none      -1    48.68    1.35    1.26       0    48.57    1.35    1.26    N/A
      131072          2048     float    none      -1    53.32    2.46    2.30       0    60.75    2.16    2.02    N/A
      262144          4096     float    none      -1    53.38    4.91    4.60       0    58.57    4.48    4.20    N/A
      524288          8192     float    none      -1    63.88    8.21    7.69       0    68.43    7.66    7.18    N/A
     1048576         16384     float    none      -1    92.09   11.39   10.67       0    88.78   11.81   11.07    N/A
     2097152         32768     float    none      -1   102.92   20.38   19.10       0   101.57   20.65   19.36    N/A
     4194304         65536     float    none      -1   167.98   24.97   23.41       0   177.28   23.66   22.18    N/A
     8388608        131072     float    none      -1   258.19   32.49   30.46       0   250.38   33.50   31.41    N/A
    16777216        262144     float    none      -1   425.71   39.41   36.95       0   408.62   41.06   38.49    N/A
    33554432        524288     float    none      -1   742.40   45.20   42.37       0   731.39   45.88   43.01    N/A
    67108864       1048576     float    none      -1  1398.38   47.99   44.99       0  1387.08   48.38   45.36    N/A
   134217728       2097152     float    none      -1  2761.89   48.60   45.56       0  2761.70   48.60   45.56    N/A
   268435456       4194304     float    none      -1  5515.84   48.67   45.62       0  5540.37   48.45   45.42    N/A
   536870912       8388608     float    none      -1  11005.7   48.78   45.73       0  11055.0   48.56   45.53    N/A
  1073741824      16777216     float    none      -1  21966.1   48.88   45.83       0  22087.8   48.61   45.57    N/A
  2147483648      33554432     float    none      -1  43973.8   48.84   45.78       0  44167.9   48.62   45.58    N/A
  4294967296      67108864     float    none      -1  87905.4   48.86   45.81       0  88470.9   48.55   45.51    N/A
  8589934592     134217728     float    none      -1   175614   48.91   45.86       0   177061   48.51   45.48    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.7233 
#
# Collective test concluded: alltoall_perf
#
```
