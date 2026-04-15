# NCCL 性能对比报告（out-of-place algbw）

**测试日期**: 2026-04-15  
**节点**: `R6KD-CX8aaS-GPU-11`（launcher），集合通信跨 `R6KD-CX8aaS-GPU-11` 与 `R6KD-CX8aaS-GPU-12` 共 16× NVIDIA RTX 6000D  
**工具**: nccl-tests 2.18.2，NCCL 2.30.1+cuda13.0，`float`，验证开启  

**脚本（执行顺序）**

1. `ssh R6KD-CX8aaS-GPU-11 "/root/jianxiong/run_ar_off.sh"` — **基准**
2. `ssh R6KD-CX8aaS-GPU-11 "/root/jianxiong/run_ar_on.sh"`
3. `ssh R6KD-CX8aaS-GPU-11 "/root/jianxiong/run_ar_on_ddp_on.sh"`

**指标说明**: 下表中的 **algbw** 均取自 nccl-tests 表头中 **out-of-place** 列（单位 GB/s）。**vs 基准** 为相对第一次运行的比例：\(\text{algbw}_{\text{本次}} / \text{algbw}_{\text{run\_ar\_off}} \times 100\%\)。若基准 algbw 为 0（打印为 0.00），百分比按 0 分母记为 **N/A** 或 **0.0%**（与 nccl-tests 四舍五入一致）。

---

## 1. AllReduce（`all_reduce_perf`，redop `sum`）

脚本名称以 `run_ar_*` 为主，**AllReduce** 为核心对比项。开启 `run_ar_on` 后，中小消息与中等规模仍接近基准；自约 **512 MiB** 起 out-of-place algbw 明显低于基准，最大消息约降至 **37–38%**。叠加 `run_ar_on_ddp_on` 后，小消息与部分区间进一步恶化，大消息稳定在基准约 **38–43%**，但中间存在 **524288 B、2097152 B** 等异常低谷（可能与调度/争用或测试抖动有关，详见原始日志）。

### 1.1 out-of-place algbw 汇总表（相对 `run_ar_off.sh`）

| size (B) | 基准 `run_ar_off` algbw (GB/s) | `run_ar_on` algbw | vs 基准 | `run_ar_on_ddp_on` algbw | vs 基准 |
|---------:|--------------------------------:|------------------:|--------:|--------------------------:|--------:|
| 1024 | 0.02 | 0.02 | 100.0% | 0.00 | 0.0% |
| 2048 | 0.04 | 0.03 | 75.0% | 0.01 | 25.0% |
| 4096 | 0.06 | 0.06 | 100.0% | 0.02 | 33.3% |
| 8192 | 0.12 | 0.10 | 83.3% | 0.04 | 33.3% |
| 16384 | 0.20 | 0.17 | 85.0% | 0.09 | 45.0% |
| 32768 | 0.25 | 0.35 | 140.0% | 0.16 | 64.0% |
| 65536 | 0.62 | 0.59 | 95.2% | 0.19 | 30.6% |
| 131072 | 1.31 | 1.11 | 84.7% | 0.39 | 29.8% |
| 262144 | 2.33 | 1.77 | 76.0% | 0.71 | 30.5% |
| 524288 | 3.66 | 2.69 | 73.5% | 0.23 | 6.3% |
| 1048576 | 4.59 | 3.63 | 79.1% | 1.75 | 38.1% |
| 2097152 | 5.61 | 4.60 | 82.0% | 0.16 | 2.9% |
| 4194304 | 6.33 | 4.65 | 73.5% | 1.49 | 23.5% |
| 8388608 | 17.29 | 14.74 | 85.3% | 6.44 | 37.2% |
| 16777216 | 20.22 | 11.95 | 59.1% | 1.91 | 9.4% |
| 33554432 | 19.38 | 18.57 | 95.8% | 3.83 | 19.8% |
| 67108864 | 20.23 | 17.38 | 85.9% | 4.09 | 20.2% |
| 134217728 | 25.12 | 21.27 | 84.7% | 5.06 | 20.1% |
| 268435456 | 25.21 | 24.14 | 95.8% | 9.67 | 38.4% |
| 536870912 | 25.30 | 13.33 | 52.7% | 10.77 | 42.6% |
| 1073741824 | 25.39 | 12.03 | 47.4% | 10.63 | 41.9% |
| 2147483648 | 25.46 | 10.98 | 43.1% | 9.80 | 38.5% |
| 4294967296 | 25.57 | 9.73 | 38.1% | 9.92 | 38.8% |
| 8589934592 | 25.76 | 9.66 | 37.5% | 9.86 | 38.3% |

**Avg bus bandwidth（AllReduce 段 nccl-tests 汇总）**

| 配置 | Avg bus bandwidth |
|------|-------------------:|
| `run_ar_off.sh` | 21.8903 |
| `run_ar_on.sh` | 14.5426 |
| `run_ar_on_ddp_on.sh` | 6.93999 |

---

## 2. AllToAll（`alltoall_perf`，附带）

同一脚本在 AllReduce 之后运行 **AllToAll**。前段消息 **out-of-place algbw** 与基准接近；`run_ar_on` 在最大两条 size 上出现 **断崖式下降**（约 **11.7% / 10.7%** 相对基准）。`run_ar_on_ddp_on` 在中大消息上整体明显偏低，最大消息相对基准约 **13.5%**。

| size (B) | 基准 algbw | `run_ar_on` | vs 基准 | `run_ar_on_ddp_on` | vs 基准 |
|---------:|-----------:|------------:|--------:|-------------------:|--------:|
| 1024 | 0.02 | 0.02 | 100.0% | 0.00 | 0.0% |
| 2048 | 0.04 | 0.04 | 100.0% | 0.01 | 25.0% |
| 4096 | 0.08 | 0.08 | 100.0% | 0.03 | 37.5% |
| 8192 | 0.17 | 0.17 | 100.0% | 0.05 | 29.4% |
| 16384 | 0.33 | 0.34 | 103.0% | 0.10 | 30.3% |
| 32768 | 0.66 | 0.67 | 101.5% | 0.20 | 30.3% |
| 65536 | 1.31 | 1.31 | 100.0% | 0.40 | 30.5% |
| 131072 | 2.53 | 2.52 | 99.6% | 0.79 | 31.2% |
| 262144 | 3.98 | 3.85 | 96.7% | 1.58 | 39.7% |
| 524288 | 8.12 | 6.88 | 84.7% | 3.05 | 37.6% |
| 1048576 | 11.46 | 9.49 | 82.8% | 5.37 | 46.9% |
| 2097152 | 19.74 | 16.38 | 83.0% | 10.04 | 50.9% |
| 4194304 | 23.74 | 22.47 | 94.7% | 10.83 | 45.6% |
| 8388608 | 32.51 | 30.39 | 93.5% | 14.13 | 43.5% |
| 16777216 | 39.58 | 37.20 | 94.0% | 16.80 | 42.4% |
| 33554432 | 44.56 | 42.76 | 96.0% | 15.90 | 35.7% |
| 67108864 | 47.54 | 46.16 | 97.1% | 16.19 | 34.1% |
| 134217728 | 48.44 | 48.22 | 99.5% | 5.10 | 10.5% |
| 268435456 | 48.81 | 49.24 | 100.9% | 9.30 | 19.1% |
| 536870912 | 48.80 | 49.62 | 101.7% | 4.05 | 8.3% |
| 1073741824 | 48.83 | 49.14 | 100.6% | 4.10 | 8.4% |
| 2147483648 | 48.91 | 48.25 | 98.7% | 4.06 | 8.3% |
| 4294967296 | 49.01 | 5.71 | 11.7% | 4.28 | 8.7% |
| 8589934592 | 49.04 | 5.26 | 10.7% | 6.64 | 13.5% |

---

## 3. 原始测试数据

以下为三次 SSH 命令的完整标准输出（含设备枚举、in-place 列及 nccl-tests 页脚）。

### 3.1 `run_ar_off.sh`

```
# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1815904 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1815905 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1815906 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1815907 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1815908 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1815909 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1815910 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1815911 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1209929 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1209930 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1209931 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1209932 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1209933 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1209934 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1209935 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1209936 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    58.59    0.02    0.03       0    55.13    0.02    0.03       0
        2048           512     float     sum      -1    57.80    0.04    0.07       0    57.83    0.04    0.07       0
        4096          1024     float     sum      -1    64.62    0.06    0.12       0    63.85    0.06    0.12       0
        8192          2048     float     sum      -1    70.96    0.12    0.22       0    68.35    0.12    0.22       0
       16384          4096     float     sum      -1    81.01    0.20    0.38       0    70.58    0.23    0.44       0
       32768          8192     float     sum      -1   129.69    0.25    0.47       0   129.89    0.25    0.47       0
       65536         16384     float     sum      -1   106.29    0.62    1.16       0   105.13    0.62    1.17       0
      131072         32768     float     sum      -1   100.01    1.31    2.46       0    98.66    1.33    2.49       0
      262144         65536     float     sum      -1   112.49    2.33    4.37       0   118.45    2.21    4.15       0
      524288        131072     float     sum      -1   143.27    3.66    6.86       0   141.60    3.70    6.94       0
     1048576        262144     float     sum      -1   228.49    4.59    8.60       0   227.72    4.60    8.63       0
     2097152        524288     float     sum      -1   373.91    5.61   10.52       0   371.26    5.65   10.59       0
     4194304       1048576     float     sum      -1   662.27    6.33   11.87       0   652.36    6.43   12.06       0
     8388608       2097152     float     sum      -1   485.04   17.29   32.43       0   483.40   17.35   32.54       0
    16777216       4194304     float     sum      -1   829.59   20.22   37.92       0   831.83   20.17   37.82       0
    33554432       8388608     float     sum      -1  1731.56   19.38   36.33       0  1707.78   19.65   36.84       0
    67108864      16777216     float     sum      -1  3317.21   20.23   37.93       0  3375.29   19.88   37.28       0
   134217728      33554432     float     sum      -1  5344.02   25.12   47.09       0  5337.99   25.14   47.14       0
   268435456      67108864     float     sum      -1  10647.4   25.21   47.27       0  10605.4   25.31   47.46       0
   536870912     134217728     float     sum      -1  21222.6   25.30   47.43       0  21198.3   25.33   47.49       0
  1073741824     268435456     float     sum      -1  42284.3   25.39   47.61       0  42302.3   25.38   47.59       0
  2147483648     536870912     float     sum      -1  84349.8   25.46   47.74       0  84401.2   25.44   47.71       0
  4294967296    1073741824     float     sum      -1   167966   25.57   47.94       0   167509   25.64   48.08       0
  8589934592    2147483648     float     sum      -1   333522   25.76   48.29       0   333543   25.75   48.29       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 21.8903 
#
# Collective test concluded: all_reduce_perf
#

# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1816146 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1816147 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1816148 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1816149 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1816150 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1816151 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1816152 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1816154 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1210168 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1210169 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1210170 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1210171 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1210172 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1210173 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1210174 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1210175 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    60.53    0.02    0.02       0    48.42    0.02    0.02    N/A
        2048            32     float    none      -1    48.82    0.04    0.04       0    48.42    0.04    0.04    N/A
        4096            64     float    none      -1    48.79    0.08    0.08       0    56.98    0.07    0.07    N/A
        8192           128     float    none      -1    48.91    0.17    0.16       0    48.60    0.17    0.16    N/A
       16384           256     float    none      -1    49.16    0.33    0.31       0    48.67    0.34    0.32    N/A
       32768           512     float    none      -1    49.92    0.66    0.62       0    59.08    0.55    0.52    N/A
       65536          1024     float    none      -1    50.10    1.31    1.23       0    51.18    1.28    1.20    N/A
      131072          2048     float    none      -1    51.73    2.53    2.38       0    51.44    2.55    2.39    N/A
      262144          4096     float    none      -1    65.85    3.98    3.73       0    77.50    3.38    3.17    N/A
      524288          8192     float    none      -1    64.60    8.12    7.61       0    66.01    7.94    7.45    N/A
     1048576         16384     float    none      -1    91.53   11.46   10.74       0    87.43   11.99   11.24    N/A
     2097152         32768     float    none      -1   106.23   19.74   18.51       0   102.75   20.41   19.13    N/A
     4194304         65536     float    none      -1   176.70   23.74   22.25       0   169.09   24.81   23.26    N/A
     8388608        131072     float    none      -1   258.05   32.51   30.48       0   254.38   32.98   30.92    N/A
    16777216        262144     float    none      -1   423.84   39.58   37.11       0   409.35   40.98   38.42    N/A
    33554432        524288     float    none      -1   752.99   44.56   41.78       0   738.35   45.45   42.60    N/A
    67108864       1048576     float    none      -1  1411.65   47.54   44.57       0  1405.97   47.73   44.75    N/A
   134217728       2097152     float    none      -1  2770.80   48.44   45.41       0  2763.68   48.56   45.53    N/A
   268435456       4194304     float    none      -1  5499.77   48.81   45.76       0  5543.76   48.42   45.39    N/A
   536870912       8388608     float    none      -1  11001.4   48.80   45.75       0  11047.9   48.60   45.56    N/A
  1073741824      16777216     float    none      -1  21990.2   48.83   45.78       0  22108.9   48.57   45.53    N/A
  2147483648      33554432     float    none      -1  43902.5   48.91   45.86       0  44178.8   48.61   45.57    N/A
  4294967296      67108864     float    none      -1  87639.2   49.01   45.94       0  88365.9   48.60   45.57    N/A
  8589934592     134217728     float    none      -1   175163   49.04   45.97       0   176698   48.61   45.58    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.6343 
#
# Collective test concluded: alltoall_perf
#
```

### 3.2 `run_ar_on.sh`

（与终端捕获一致，exit code 0）

```
# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1816727 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1816728 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1816729 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1816730 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1816731 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1816732 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1816733 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1816734 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1210625 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1210626 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1210627 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1210628 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1210629 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1210630 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1210631 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1210632 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1    58.06    0.02    0.03       0    55.79    0.02    0.03       0
        2048           512     float     sum      -1    59.79    0.03    0.06       0    58.17    0.04    0.07       0
        4096          1024     float     sum      -1    64.26    0.06    0.12       0    68.83    0.06    0.11       0
        8192          2048     float     sum      -1    84.57    0.10    0.18       0    75.18    0.11    0.20       0
       16384          4096     float     sum      -1    94.64    0.17    0.32       0    71.20    0.23    0.43       0
       32768          8192     float     sum      -1    94.79    0.35    0.65       0    92.18    0.36    0.67       0
       65536         16384     float     sum      -1   110.64    0.59    1.11       0   116.71    0.56    1.05       0
      131072         32768     float     sum      -1   118.20    1.11    2.08       0   117.89    1.11    2.08       0
      262144         65536     float     sum      -1   147.95    1.77    3.32       0   146.37    1.79    3.36       0
      524288        131072     float     sum      -1   194.96    2.69    5.04       0   187.37    2.80    5.25       0
     1048576        262144     float     sum      -1   288.54    3.63    6.81       0   282.14    3.72    6.97       0
     2097152        524288     float     sum      -1   456.23    4.60    8.62       0   453.25    4.63    8.68       0
     4194304       1048576     float     sum      -1   902.33    4.65    8.72       0   984.38    4.26    7.99       0
     8388608       2097152     float     sum      -1   569.20   14.74   27.63       0   514.56   16.30   30.57       0
    16777216       4194304     float     sum      -1  1403.93   11.95   22.41       0   968.48   17.32   32.48       0
    33554432       8388608     float     sum      -1  1806.82   18.57   34.82       0  1752.57   19.15   35.90       0
    67108864      16777216     float     sum      -1  3860.89   17.38   32.59       0  3807.43   17.63   33.05       0
   134217728      33554432     float     sum      -1  6310.48   21.27   39.88       0  5937.67   22.60   42.38       0
   268435456      67108864     float     sum      -1  11119.3   24.14   45.27       0  11034.4   24.33   45.61       0
   536870912     134217728     float     sum      -1  40275.9   13.33   24.99       0  54318.5    9.88   18.53       0
  1073741824     268435456     float     sum      -1  89264.2   12.03   22.55       0  89676.5   11.97   22.45       0
  2147483648     536870912     float     sum      -1   195602   10.98   20.59       0   211971   10.13   19.00       0
  4294967296    1073741824     float     sum      -1   441337    9.73   18.25       0   436189    9.85   18.46       0
  8589934592    2147483648     float     sum      -1   889642    9.66   18.10       0   867267    9.90   18.57       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 14.5426 
#
# Collective test concluded: all_reduce_perf
#

# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1817294 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1817295 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1817296 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1817297 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1817298 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1817299 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1817300 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1817301 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1211154 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1211155 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1211156 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1211157 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1211158 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1211159 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1211160 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1211161 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1    51.58    0.02    0.02       0    49.30    0.02    0.02    N/A
        2048            32     float    none      -1    48.42    0.04    0.04       0    48.28    0.04    0.04    N/A
        4096            64     float    none      -1    48.83    0.08    0.08       0    47.92    0.09    0.08    N/A
        8192           128     float    none      -1    48.39    0.17    0.16       0    48.30    0.17    0.16    N/A
       16384           256     float    none      -1    48.77    0.34    0.31       0    48.36    0.34    0.32    N/A
       32768           512     float    none      -1    49.22    0.67    0.62       0    48.86    0.67    0.63    N/A
       65536          1024     float    none      -1    50.15    1.31    1.23       0    49.02    1.34    1.25    N/A
      131072          2048     float    none      -1    52.06    2.52    2.36       0    51.25    2.56    2.40    N/A
      262144          4096     float    none      -1    68.05    3.85    3.61       0    72.41    3.62    3.39    N/A
      524288          8192     float    none      -1    76.24    6.88    6.45       0    75.22    6.97    6.53    N/A
     1048576         16384     float    none      -1   110.54    9.49    8.89       0   110.46    9.49    8.90    N/A
     2097152         32768     float    none      -1   128.04   16.38   15.36       0   123.78   16.94   15.88    N/A
     4194304         65536     float    none      -1   186.65   22.47   21.07       0   176.96   23.70   22.22    N/A
     8388608        131072     float    none      -1   276.02   30.39   28.49       0   276.59   30.33   28.43    N/A
    16777216        262144     float    none      -1   450.97   37.20   34.88       0   441.22   38.02   35.65    N/A
    33554432        524288     float    none      -1   784.75   42.76   40.09       0   774.88   43.30   40.60    N/A
    67108864       1048576     float    none      -1  1453.71   46.16   43.28       0  1438.29   46.66   43.74    N/A
   134217728       2097152     float    none      -1  2783.55   48.22   45.20       0  2765.39   48.53   45.50    N/A
   268435456       4194304     float    none      -1  5451.11   49.24   46.17       0  5415.44   49.57   46.47    N/A
   536870912       8388608     float    none      -1  10819.7   49.62   46.52       0  10746.8   49.96   46.83    N/A
  1073741824      16777216     float    none      -1  21851.9   49.14   46.07       0  22905.8   46.88   43.95    N/A
  2147483648      33554432     float    none      -1  44506.1   48.25   45.24       0   195095   11.01   10.32    N/A
  4294967296      67108864     float    none      -1   751606    5.71    5.36       0   894315    4.80    4.50    N/A
  8589934592     134217728     float    none      -1  1634307    5.26    4.93       0  1657530    5.18    4.86    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.8976 
#
# Collective test concluded: alltoall_perf
#
```

### 3.3 `run_ar_on_ddp_on.sh`

（与终端捕获一致，exit code 0）

```
# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1816826 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1816827 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1816828 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1816829 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1816830 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1816831 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1816832 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1816833 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1210703 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1210704 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1210705 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1210706 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1210707 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1210708 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1210709 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1210710 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024           256     float     sum      -1   670.90    0.00    0.00       0   169.29    0.01    0.01       0
        2048           512     float     sum      -1   181.31    0.01    0.02       0   177.85    0.01    0.02       0
        4096          1024     float     sum      -1   186.98    0.02    0.04       0   179.71    0.02    0.04       0
        8192          2048     float     sum      -1   189.81    0.04    0.08       0   194.14    0.04    0.08       0
       16384          4096     float     sum      -1   190.33    0.09    0.16       0   191.76    0.09    0.16       0
       32768          8192     float     sum      -1   205.69    0.16    0.30       0   204.17    0.16    0.30       0
       65536         16384     float     sum      -1   350.25    0.19    0.35       0   349.97    0.19    0.35       0
      131072         32768     float     sum      -1   340.38    0.39    0.72       0   339.14    0.39    0.72       0
      262144         65536     float     sum      -1   368.09    0.71    1.34       0   370.43    0.71    1.33       0
      524288        131072     float     sum      -1  2315.88    0.23    0.42       0   390.34    1.34    2.52       0
     1048576        262144     float     sum      -1   600.20    1.75    3.28       0  8054.85    0.13    0.24       0
     2097152        524288     float     sum      -1  13002.3    0.16    0.30       0  5855.61    0.36    0.67       0
     4194304       1048576     float     sum      -1  2811.85    1.49    2.80       0  2762.08    1.52    2.85       0
     8388608       2097152     float     sum      -1  1302.84    6.44   12.07       0  2341.89    3.58    6.72       0
    16777216       4194304     float     sum      -1  8773.11    1.91    3.59       0  3947.18    4.25    7.97       0
    33554432       8388608     float     sum      -1  8761.73    3.83    7.18       0  4382.51    7.66   14.36       0
    67108864      16777216     float     sum      -1  16415.8    4.09    7.67       0  32650.5    2.06    3.85       0
   134217728      33554432     float     sum      -1  26524.2    5.06    9.49       0  22047.7    6.09   11.41       0
   268435456      67108864     float     sum      -1  27755.8    9.67   18.13       0  32484.1    8.26   15.49       0
   536870912     134217728     float     sum      -1  49840.6   10.77   20.20       0  52752.2   10.18   19.08       0
  1073741824     268435456     float     sum      -1   100994   10.63   19.93       0   107929    9.95   18.65       0
  2147483648     536870912     float     sum      -1   219095    9.80   18.38       0   220217    9.75   18.28       0
  4294967296    1073741824     float     sum      -1   433010    9.92   18.60       0   436345    9.84   18.46       0
  8589934592    2147483648     float     sum      -1   871305    9.86   18.49       0   619282   13.87   26.01       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 6.93999 
#
# Collective test concluded: all_reduce_perf
#

# nccl-tests version 2.18.2 nccl-headers=23001 nccl-library=23001
# Collective test starting: alltoall_perf
# nThread 1 nGpus 1 minBytes 1024 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1817374 on R6KD-CX8aaS-GPU-11 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  1 Group  0 Pid 1817375 on R6KD-CX8aaS-GPU-11 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank  2 Group  0 Pid 1817376 on R6KD-CX8aaS-GPU-11 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank  3 Group  0 Pid 1817377 on R6KD-CX8aaS-GPU-11 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank  4 Group  0 Pid 1817378 on R6KD-CX8aaS-GPU-11 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank  5 Group  0 Pid 1817379 on R6KD-CX8aaS-GPU-11 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank  6 Group  0 Pid 1817380 on R6KD-CX8aaS-GPU-11 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank  7 Group  0 Pid 1817382 on R6KD-CX8aaS-GPU-11 device  7 [0000:f9:00] NVIDIA RTX 6000D
#  Rank  8 Group  0 Pid 1211254 on R6KD-CX8aaS-GPU-12 device  0 [0000:06:00] NVIDIA RTX 6000D
#  Rank  9 Group  0 Pid 1211255 on R6KD-CX8aaS-GPU-12 device  1 [0000:09:00] NVIDIA RTX 6000D
#  Rank 10 Group  0 Pid 1211256 on R6KD-CX8aaS-GPU-12 device  2 [0000:76:00] NVIDIA RTX 6000D
#  Rank 11 Group  0 Pid 1211257 on R6KD-CX8aaS-GPU-12 device  3 [0000:79:00] NVIDIA RTX 6000D
#  Rank 12 Group  0 Pid 1211258 on R6KD-CX8aaS-GPU-12 device  4 [0000:86:00] NVIDIA RTX 6000D
#  Rank 13 Group  0 Pid 1211259 on R6KD-CX8aaS-GPU-12 device  5 [0000:89:00] NVIDIA RTX 6000D
#  Rank 14 Group  0 Pid 1211260 on R6KD-CX8aaS-GPU-12 device  6 [0000:f6:00] NVIDIA RTX 6000D
#  Rank 15 Group  0 Pid 1211261 on R6KD-CX8aaS-GPU-12 device  7 [0000:f9:00] NVIDIA RTX 6000D
NCCL version 2.30.1+cuda13.0
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
        1024            16     float    none      -1   347.61    0.00    0.00       0   161.59    0.01    0.01    N/A
        2048            32     float    none      -1   161.40    0.01    0.01       0   160.80    0.01    0.01    N/A
        4096            64     float    none      -1   161.88    0.03    0.02       0   162.08    0.03    0.02    N/A
        8192           128     float    none      -1   161.67    0.05    0.05       0   161.91    0.05    0.05    N/A
       16384           256     float    none      -1   162.05    0.10    0.09       0   162.11    0.10    0.09    N/A
       32768           512     float    none      -1   162.92    0.20    0.19       0   162.22    0.20    0.19    N/A
       65536          1024     float    none      -1   162.58    0.40    0.38       0   162.45    0.40    0.38    N/A
      131072          2048     float    none      -1   165.41    0.79    0.74       0   164.01    0.80    0.75    N/A
      262144          4096     float    none      -1   166.37    1.58    1.48       0   166.93    1.57    1.47    N/A
      524288          8192     float    none      -1   172.14    3.05    2.86       0  2391.22    0.22    0.21    N/A
     1048576         16384     float    none      -1   195.17    5.37    5.04       0   193.74    5.41    5.07    N/A
     2097152         32768     float    none      -1   208.98   10.04    9.41       0  2306.87    0.91    0.85    N/A
     4194304         65536     float    none      -1   387.25   10.83   10.15       0   388.49   10.80   10.12    N/A
     8388608        131072     float    none      -1   593.73   14.13   13.25       0   587.65   14.27   13.38    N/A
    16777216        262144     float    none      -1   998.70   16.80   15.75       0   994.47   16.87   15.82    N/A
    33554432        524288     float    none      -1  2110.27   15.90   14.91       0  1680.02   19.97   18.72    N/A
    67108864       1048576     float    none      -1  4145.46   16.19   15.18       0  13421.5    5.00    4.69    N/A
   134217728       2097152     float    none      -1  26293.3    5.10    4.79       0  30913.5    4.34    4.07    N/A
   268435456       4194304     float    none      -1  28869.7    9.30    8.72       0  66084.0    4.06    3.81    N/A
   536870912       8388608     float    none      -1   132552    4.05    3.80       0   135899    3.95    3.70    N/A
  1073741824      16777216     float    none      -1   261863    4.10    3.84       0   246081    4.36    4.09    N/A
  2147483648      33554432     float    none      -1   529323    4.06    3.80       0   444457    4.83    4.53    N/A
  4294967296      67108864     float    none      -1  1003950    4.28    4.01       0   852435    5.04    4.72    N/A
  8589934592     134217728     float    none      -1  1293091    6.64    6.23       0   177114   48.50   45.47    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 5.56077 
#
# Collective test concluded: alltoall_perf
#
```

---

*报告由自动化抓取 nccl-tests 输出整理；百分比按打印的 algbw 数值计算，未做更高精度反推。*
