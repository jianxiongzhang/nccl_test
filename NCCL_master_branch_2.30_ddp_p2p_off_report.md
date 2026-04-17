# NCCL master branch 2.30 — DDP P2P off — `all_reduce_perf` / `alltoall_perf` report

## 1. Test commands

Commands were launched from `R6KD-CX8aaS-GPU-11` and execute the bundled scripts on that host (which in turn drive the 16-GPU job across both nodes).

| Case | Command |
| --- | --- |
| AR OFF | `ssh R6KD-CX8aaS-GPU-11 "/root/jianxiong/script/run_ar_off.sh"` |
| AR ON, DDP OFF | `ssh R6KD-CX8aaS-GPU-11 "/root/jianxiong/script/run_ar_on.sh"` |
| AR ON, DDP ON | `ssh R6KD-CX8aaS-GPU-11 "/root/jianxiong/script/run_ar_on_ddp_on.sh"` |

Each script runs `nccl-tests` collectives `all_reduce_perf` and `alltoall_perf` with the parameters shown in the raw logs (for example `minBytes 1024`, `maxBytes 8589934592`, step factor 2, `warmup iters: 1`, `iters: 20`, validation enabled).

## 2. Test environment and topology

- **Hardware**: two **NVIDIA RTX 6000D** servers (6KD-class hosts in the naming pattern), **8 GPUs per node**, **16 ranks total** (`nGpus 1`, one process per GPU).
- **Hosts**: `R6KD-CX8aaS-GPU-11` (ranks 0–7) and `R6KD-CX8aaS-GPU-12` (ranks 8–15). PCIe bus IDs per GPU are listed in the raw logs (for example `0000:06:00` … `0000:f9:00` on each node).
- **NCCL**: runtime reports **`NCCL version 2.30.3+cuda13.0`**. The `nccl-tests` banner shows headers/library build **`23003`** (see raw logs).
- **Collectives under test**: `all_reduce_perf` (float sum) and `alltoall_perf` (float, `redop none`). The comparison tables below use the **out-of-place** `algbw` column only.

## 3. Comparison tables (out-of-place `algbw`, GB/s)

Baseline is **AR OFF**. **Δ%** is `((algbw_case − algbw_AR_OFF) / algbw_AR_OFF) × 100`. Positive Δ% means higher algorithm bandwidth than AR OFF; negative means lower.

In the tables below, the numeric **Δ vs AR OFF** cells are **percentage points** relative to AR OFF (for example, `44.00` means **+44.00%**, and `−4.84` means **−4.84%**).

### 3.1 `all_reduce_perf` (out-of-place `algbw`)

| size (B) | AR OFF | AR ON DDP OFF | Δ vs AR OFF (%) | AR ON DDP ON | Δ vs AR OFF (%) |
| ---:| ---:| ---:| ---:| ---:| ---:|
| 1024 | 0.02 | 0.02 | 0.00 | 0.02 | 0.00 |
| 2048 | 0.03 | 0.03 | 0.00 | 0.03 | 0.00 |
| 4096 | 0.06 | 0.06 | 0.00 | 0.06 | 0.00 |
| 8192 | 0.11 | 0.11 | 0.00 | 0.11 | 0.00 |
| 16384 | 0.22 | 0.22 | 0.00 | 0.22 | 0.00 |
| 32768 | 0.25 | 0.36 | 44.00 | 0.39 | 56.00 |
| 65536 | 0.62 | 0.59 | −4.84 | 0.61 | −1.61 |
| 131072 | 1.32 | 1.14 | −13.64 | 1.31 | −0.76 |
| 262144 | 2.34 | 1.93 | −17.52 | 2.30 | −1.71 |
| 524288 | 3.70 | 2.88 | −22.16 | 3.69 | −0.27 |
| 1048576 | 4.53 | 3.71 | −18.10 | 4.63 | 2.21 |
| 2097152 | 5.53 | 4.65 | −15.91 | 5.56 | 0.54 |
| 4194304 | 6.26 | 5.38 | −14.06 | 6.34 | 1.28 |
| 8388608 | 17.35 | 16.18 | −6.74 | 17.13 | −1.27 |
| 16777216 | 20.24 | 20.39 | 0.74 | 20.35 | 0.54 |
| 33554432 | 19.62 | 19.02 | −3.06 | 19.69 | 0.36 |
| 67108864 | 19.86 | 19.60 | −1.31 | 19.98 | 0.60 |
| 134217728 | 25.18 | 23.99 | −4.73 | 25.08 | −0.40 |
| 268435456 | 25.32 | 24.41 | −3.59 | 25.19 | −0.51 |
| 536870912 | 25.29 | 25.32 | 0.12 | 25.15 | −0.55 |
| 1073741824 | 25.40 | 25.31 | −0.35 | 25.25 | −0.59 |
| 2147483648 | 25.51 | 25.33 | −0.71 | 25.35 | −0.63 |
| 4294967296 | 25.68 | 25.32 | −1.40 | 25.54 | −0.55 |
| 8589934592 | 25.77 | 25.32 | −1.75 | 25.64 | −0.50 |

**Avg bus bandwidth (`nccl-tests` summary):** AR OFF **21.8894** GB/s; AR ON DDP OFF **21.1929** GB/s; AR ON DDP ON **21.8501** GB/s.

### 3.2 `alltoall_perf` (out-of-place `algbw`)

| size (B) | AR OFF | AR ON DDP OFF | Δ vs AR OFF (%) | AR ON DDP ON | Δ vs AR OFF (%) |
| ---:| ---:| ---:| ---:| ---:| ---:|
| 1024 | 0.02 | 0.02 | 0.00 | 0.02 | 0.00 |
| 2048 | 0.04 | 0.04 | 0.00 | 0.04 | 0.00 |
| 4096 | 0.09 | 0.09 | 0.00 | 0.09 | 0.00 |
| 8192 | 0.17 | 0.17 | 0.00 | 0.17 | 0.00 |
| 16384 | 0.34 | 0.33 | −2.94 | 0.34 | 0.00 |
| 32768 | 0.68 | 0.67 | −1.47 | 0.67 | −1.47 |
| 65536 | 1.35 | 1.31 | −2.96 | 1.30 | −3.70 |
| 131072 | 2.60 | 2.59 | −0.38 | 2.60 | 0.00 |
| 262144 | 4.03 | 3.86 | −4.22 | 4.97 | 23.33 |
| 524288 | 8.20 | 6.78 | −17.32 | 8.14 | −0.73 |
| 1048576 | 12.09 | 9.56 | −20.93 | 11.80 | −2.40 |
| 2097152 | 20.73 | 16.51 | −20.36 | 20.56 | −0.82 |
| 4194304 | 25.58 | 22.65 | −11.45 | 25.55 | −0.12 |
| 8388608 | 32.97 | 30.65 | −7.04 | 33.27 | 0.91 |
| 16777216 | 39.72 | 37.28 | −6.14 | 39.58 | −0.35 |
| 33554432 | 44.77 | 42.95 | −4.07 | 45.24 | 1.05 |
| 67108864 | 47.66 | 46.29 | −2.87 | 48.03 | 0.78 |
| 134217728 | 48.68 | 48.17 | −1.05 | 48.42 | −0.53 |
| 268435456 | 48.79 | 49.22 | 0.88 | 48.71 | −0.16 |
| 536870912 | 48.84 | 49.71 | 1.78 | 48.64 | −0.41 |
| 1073741824 | 48.99 | 49.95 | 1.96 | 48.91 | −0.16 |
| 2147483648 | 48.96 | 50.07 | 2.27 | 49.00 | 0.08 |
| 4294967296 | 49.03 | 50.06 | 2.10 | 48.99 | −0.08 |
| 8589934592 | 49.15 | 50.10 | 1.93 | 49.02 | −0.26 |

**Avg bus bandwidth (`nccl-tests` summary):** AR OFF **22.8307** GB/s; AR ON DDP OFF **22.3033** GB/s; AR ON DDP ON **22.8494** GB/s.

## 4. Raw logs (appendix)

Full console output for reproducibility (includes device listing, per-size `time` / `busbw`, and in-place columns).

### Case 1 — AR OFF

```log
PLACEHOLDER_CASE1_AR_OFF
```

### Case 2 — AR ON, DDP OFF

```log
PLACEHOLDER_CASE2_AR_ON_DDP_OFF
```

### Case 3 — AR ON, DDP ON

```log
PLACEHOLDER_CASE3_AR_ON_DDP_ON
```
