# TD.2a — llama3.2:1b decode attribution on `DEV=NV:NAK` (RTX 3090, AG02/USB4)

Worktree: `tinygrad-dock`, detached at fork master `b37d80fc9`. Machine: MacBook Pro M3 Pro, dock
live (TD.1 first light passed 2026-08-24). Model: `llama3.2:1b` (dense, **16 transformer layers**
— confirmed below by per-kernel invocation counts, all recurring kernels fire exactly 16x or 32x).

No `tinygrad/` source was touched. All measurements use existing env-var instrumentation
(`DEBUG=2`, `JIT_BATCH_SIZE=1`, `JITBEAM=2`) — no diff to disclose.

## TL;DR verdict

HCQGraph **does** engage — there is no per-kernel socket round trip in the steady-state decode
path. But it **does not matter much**: only ~3-4 ms/token (~3%) of the 133 ms is host
dispatch/socket overhead that graphing removes. The other **~97% (≈130 ms) is genuine on-GPU
kernel-execution time**, reported by real hardware signal timestamps, identical whether the step
is graphed or run kernel-by-kernel. Four recurring per-layer GGUF-quantized gemv/reduce kernel
shapes (all suffixed `_32`, the Q-block axis) account for **88% of the step** and run at
**1–34 GB/s effective and 17–136 "GFLOPS"** on a 936 GB/s / >10 TFLOPS card — i.e. under 5% of the
card's real throughput. `JITBEAM=2` finds a ~19x faster kernel selection for the same ASTs
(7.5 → 142 tok/s), proving this is a **kernel-selection/codegen quality problem on the NV/NAK
path**, not a transport, dispatch, or graph-formation problem. None of the prepared transport
levers (T2.1/T2.2/T2.3/drain_every) touch this; the closest prepared lever is **T1.10** (MATVEC for
quantized gemvs), which was implemented and measured on METAL only — its NV/NAK-side counterpart
does not exist yet and is the real gap.

---

## 1. Does HCQGraph engage?

**Yes**, but it fragments into 4 sub-graphs per token instead of 1, because `JIT_BATCH_SIZE`
(`tinygrad/engine/jit.py:32-59`) defaults to 32 and **doubles after every flush** rather than
sizing to the whole step (`max_batch_size *= 2` at `jit.py:42`). Evidence, `DEBUG=2` during JIT
capture (warmup):

```
JIT GRAPHing batch with 32 kernels
JIT GRAPHing batch with 64 kernels
JIT GRAPHing batch with 128 kernels
JIT GRAPHing batch with 8 kernels
```

32+64+128+8 = 232 kernels — matches `jit execs 232 calls` seen under `JIT_BATCH_SIZE=1` and the
sum of the per-kernel table below. At replay, `DEBUG=2` shows one line **per graph batch**, not
per kernel (`ast.arg == "graph"` collapses `get_call_kernels` to a single `(dev, call)` —
`tinygrad/engine/realize.py:29-33`), each carrying a real device-measured wall time
(`tinygrad/runtime/graph/hcq.py:298-301`, `wait=True` path):

```
*** NV   1  batched 32   tm  18.09ms/  18.09ms ( 89 GFLOPS  6|26  GB/s)
*** NV   2  batched 64   tm  32.85ms/  50.94ms ( 97 GFLOPS  6|29  GB/s)
*** NV   3  batched 128  tm  71.67ms/ 122.61ms (100 GFLOPS  7|29  GB/s)
*** NV   4  batched 8    tm  10.19ms/ 132.80ms (396 GFLOPS 24|115 GB/s)
                                              (wall for the step: 136.64 ms)
```

Sum of the 4 batch times = 132.80 ms ≈ the step's wall time (136.64 ms) — only ~3.8 ms (~3%) of
host overhead outside GPU execution. **No per-kernel socket dispatch is happening in steady
state.** (This also rules out the "graph not forming" and "every kernel dispatches individually
over USB4" hypotheses from the task brief.)

Confirmed by direct comparison: forcing `JIT_BATCH_SIZE=1` (no graphing at all — every kernel gets
its own `NVCommandQueue`, its own 3-socket-write submit `ops_nv.py:121-133`, and, under `DEBUG=2`,
a forced `dev.synchronize()` after every kernel) gives:

- Wall time for the same steady-state step: **163.31 ms**
- Sum of real GPU kernel-execution time (`sig_en - sig_st` hardware timestamps): **133.10 ms**
- Host dispatch/socket/sync overhead: **30.2 ms** (≈18.5% of wall — this is what graphing removes)

Graphed vs ungraphed wall time: 136.64 ms vs 163.31 ms → **graphing saves ~26 ms/token (~16%)**.
That's real and worth having, but small next to the ~130 ms floor underneath it.

## 2. Where do the 133 ms go — per-kernel table

Table below is the **ungraphed** (`JIT_BATCH_SIZE=1`) capture of the second (steady-state) decode
step, `DEBUG=2`, so every row is one kernel with a real GPU-signal-measured execution time (not a
graph aggregate). Total = 133.10 ms, matching the graphed wall time to within noise — confirming
kernel execution time is unchanged by whether the step is graphed.

| kernel (shape signature) | count/step | avg time | total time | % of step | throughput |
|---|---:|---:|---:|---:|---|
| `r_64_32_4_8_2_2_2_32` (2/layer) | 32 | 1624 µs | 52.0 ms | **39.1%** | 136 GFLOPS, 9\|34 GB/s |
| `r_4_8_2_8_4_8_2_2_2_32` (1/layer) | 16 | 2483 µs | 39.7 ms | **29.9%** | 22 GFLOPS, 1\|6 GB/s |
| `r_2_8_16_2_2_8_2_2_2_32` (1/layer) | 16 | 914 µs | 14.6 ms | 11.0% | 17 GFLOPS, 1\|6 GB/s |
| `r_2_2_4_2_16_8_2_2_2_32` (1/layer) | 16 | 686 µs | 11.0 ms | 8.3% | 29 GFLOPS, 1\|14 GB/s |
| `r_1336_32_3_8_2_2_2_32` (lm_head, 1/token) | 1 | 7763 µs | 7.76 ms | 5.8% | 466 GFLOPS, 28\|135 GB/s |
| `r_128_8_4_4_32_2_2_2_4` (1/layer) | 16 | 328 µs | 5.25 ms | 3.9% | 601 GFLOPS, 43\|212 GB/s |
| `r_128_8_4_4_8_2_2_2_4` (1/layer) | 16 | 85 µs | 1.35 ms | 1.0% | 577 GFLOPS, 41\|210 GB/s |
| everything else (RoPE/reshape/KV-write/argmax, ~120 tiny kernels) | ~120 | <15 µs each | ~1.1 ms | 0.8% | trivial |
| **Total (232 kernels)** | | | **133.10 ms** | 100% | |

The recurring per-layer count is always **16 or 32**, i.e. exactly 1x or 2x per transformer layer
— confirms the model has 16 layers and that these are the attention/FFN weight-read kernels, not
one-off setup. All four dominant shapes end in `_32` — the GGUF quantization-block axis (Q4_K/Q6_K
style dequant fused into the gemv, per `llm/gguf.py`) — matching T1.10's documented finding that
"GGUF block substructure splits the row axis into multiple global axes" and that these are "the
dominant decode kernels for Q4_K models." **The top 4 shapes alone are 88.3% of the step**, and
they run at 1–9 "GFLOPS-equivalent" GB/s bandwidth vs the card's 936 GB/s — under 4% efficiency.
By contrast the two smaller `r_128_8_4_4_...` kernels (rows 6–7) hit 577–601 GFLOPS / 41–43 GB/s —
proof this is a per-shape codegen problem, not a hard ceiling (some shapes get decent code, the
big ones don't).

**Split: kernel execution vs dispatch/sync/copy overhead** (steady-state step):

| | ungraphed (JIT_BATCH_SIZE=1) | graphed (default) |
|---|---:|---:|
| wall time | 163.31 ms | 136.64 ms |
| GPU kernel-execution time (signal timestamps) | 133.10 ms (81.5%) | ~132.80 ms (97.2%) |
| host dispatch/socket/sync overhead | 30.21 ms (18.5%) | ~3.84 ms (2.8%) |

Kernel execution dominates in both cases; graphing just removes the per-kernel dispatch tax.

## 3. JITBEAM=2 datum

One run, same model/device, `--benchmark 3 --warmup`, decode tok/s only (steady-state, last 2 of 3
lines — the first line in any `--benchmark` run is consistently faster/still-warming and is not
representative, see command log below):

```
 15.30 ms,  65.37 tok/s   (first step, not steady state)
  6.93 ms, 144.20 tok/s
  7.05 ms, 141.83 tok/s
```

**No-BEAM baseline: 133 ms/token, 7.52 tok/s. JITBEAM=2: ~7.0 ms/token, ~142 tok/s — a ~19x
speedup**, from BEAM finding a different tile/thread configuration for the exact same kernel ASTs
(same ops, same GGUF weights, same device). For context: this JITBEAM=2 number is **~3.8x faster
than the 5070 Ti external reference** (37.6 tok/s, same dock class) and **~6x faster than local
METAL** (~24 tok/s). BEAM search + 3 benchmark iterations took 4m52s wall (fits comfortably in
budget); results cache into `~/Library/Caches/tinygrad/cache.db` per the design doc, so this is a
one-time cost, not a per-run tax.

## 4. Biggest lever

**Kernel-selection/codegen quality for the GGUF-quantized (`_32`-block) per-layer gemvs on the
NV/NAK path** — not any of the transport levers. Evidence chain:

1. Graph engagement is real and correctly collapses dispatch (§1) — T2.1 (_copyout parallel),
   T2.2 (batched PTE writes), T2.3 (_REMOTE_SIZING knobs), and `drain_every>1` (T2.5) all target
   **host/socket dispatch overhead**, which is already only ~3-4 ms/token (2.8%) once graphed.
   Even a perfect transport (0 ms overhead) would only buy back that ~4 ms, taking 133 ms → ~129 ms
   — noise next to the actual gap.
2. 88% of the step is 4 recurring quantized-weight gemv/reduce kernel shapes running at 1-9
   "GFLOPS" / 1-34 GB/s — under 5% of the card's real throughput (§2).
3. `JITBEAM=2` recovers ~19x on the *identical* kernel ASTs with *zero* transport change (§3) —
   this isolates the cause to kernel implementation choice, not the tunnel, not the driver, not
   graph formation.
4. This is the **NV-side gap that T1.10 already named and fixed for METAL only**
   (`NV_LLM_DESIGN.md` §3.4, `TASKS.md` T1.10): "quantized gemvs miss MATVEC for two deeper
   reasons: the weight operand is a whole dequant expression... and GGUF block substructure splits
   the row axis into multiple global axes." T1.10's heuristic fix lives in
   `codegen/opt/heuristic.py` and was measured/verified on METAL; whether/how it fires for NV's
   renderer stack (`[CUDARenderer, PTXRenderer, NVCCRenderer, NAKRenderer]`, NAK in our case, which
   the design doc already flags as exposing **no tensor cores**, `ops_nv.py:634`) is untested. The
   BEAM result says the heuristic path is leaving a huge amount on the table specifically here.

**Recommended next step (not done in this task — STOP condition, measurement only):** re-run T1.10
verification/microbenchmark on `DEV=NV:NAK` directly (not just METAL) to see whether the MATVEC
heuristic fires at all for these `_32`-block shapes on NV, and if not, why (renderer-specific guard
gap vs. genuinely different optimal tiling). The **cheap, no-code-change interim win** is exactly
what the design doc's WS1 item 3 already proposed: default `JITBEAM=2` for the NV `llm` CLI path
(or ship a pre-populated beam cache for the workhorse models) — it alone would take this box from
7.5 → ~142 tok/s with no source changes.

---

## Exact commands used

```bash
cd tinygrad-dock   # worktree, relative paths only for repo files

# baseline repro (known-good, reconfirmed)
PYTHONPATH=. DEV=NV:NAK /Users/artur/Documents/tinygrad/.venv/bin/python \
  -m tinygrad.llm -m llama3.2:1b --benchmark 5 --warmup

# Q1/Q2: graphed steady state, DEBUG=2, 3 decode steps (2nd/3rd are steady-state)
PYTHONPATH=. DEV=NV:NAK DEBUG=2 /Users/artur/Documents/tinygrad/.venv/bin/python \
  -m tinygrad.llm -m llama3.2:1b --benchmark 3 --warmup

# Q1/Q2: ungraphed per-kernel table + dispatch-overhead isolation, 2 decode steps
PYTHONPATH=. DEV=NV:NAK JIT_BATCH_SIZE=1 DEBUG=2 /Users/artur/Documents/tinygrad/.venv/bin/python \
  -m tinygrad.llm -m llama3.2:1b --benchmark 2 --warmup

# Q3: JITBEAM=2 datum
PYTHONPATH=. DEV=NV:NAK JITBEAM=2 /Users/artur/Documents/tinygrad/.venv/bin/python \
  -m tinygrad.llm -m llama3.2:1b --benchmark 3 --warmup
```

No `tinygrad/` files were modified for this task (`git status`/`git diff --stat` clean throughout);
all attribution came from existing `DEBUG`/`JIT_BATCH_SIZE`/`JITBEAM` env-var hooks and manual
log parsing (ad hoc, not committed).

## Caveat / open thread (not chased further, out of scope for this task)

Within any single `--benchmark N` run, decode step **i=0 is consistently ~1.4-1.5x faster** than
steady state (e.g. 90-96 ms vs 133-137 ms in separate repro runs) before settling. Cause not
determined (not context-length growth — it's a step change, not a ramp). Flagged for whoever picks
up per-token variance next; does not affect the steady-state attribution above since all numbers
here are taken from the 2nd/3rd step onward.
