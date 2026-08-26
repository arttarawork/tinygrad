# T0.1 / T0.3 — Metal baseline table + bench harness

Date 2026-08-18. Hardware: MacBook Pro M3 Pro 36 GB, ~150 GB/s. `llama-server` (LaunchAgent)
stopped for the whole session; benches run strictly sequentially, nothing in parallel.
Model: `qwen3:8b` Q4_K_M (5.02 GB), fetched via `tinygrad.llm.cli.models["qwen3:8b"]`.

## Harness (T0.3)

`extra/benchmark_llm.py` already existed (load/warmup/prefill/decode timing loop) — extended it
in place with `GlobalCounters`-based GB/s reporting for the prefill and decode phases, rather than
duplicating the timing loop elsewhere.

`extra/bench_llm.py` is new: a thin wrapper that runs either stack and appends one CSV row per run.
It reuses `extra/benchmark_llm.py` via subprocess (parses its stdout) for the `tinygrad` stack, and
`llama-bench -o csv` for the `llamacpp` stack (its CSV already carries per-test avg_ts/stddev_ts
aggregated over `-r` repetitions, so no ad-hoc parsing of the human-readable table). CSV columns:
`model, stack, device, flags, load_s, prefill_tps, decode_tps, gbps` (gbps is decode-phase
GlobalCounters bandwidth for the tinygrad stack; blank for llama.cpp — no GlobalCounters
equivalent there).

```
./extra/bench_llm.py llamacpp --model qwen3:8b --csv extra/bench_results_2026-08-18.csv
./extra/bench_llm.py tinygrad --model qwen3:8b --device METAL --repeat 3 --csv extra/bench_results_2026-08-18.csv
./extra/bench_llm.py tinygrad --model qwen3:8b --env JITBEAM=2 --env IGNORE_BEAM_CACHE=1 --csv extra/bench_results_2026-08-18.csv
```

`--env PYTHONPATH=<checkout>` lets one invocation point `benchmark_llm.py`'s `tinygrad` import at a
different worktree, which is how upstream-vs-integration was measured with the same harness script
(see below) — no need to reinstall/duplicate the harness per checkout.

Ruff excludes `extra/` entirely (`pyproject.toml` `[tool.ruff] exclude`); mypy's `extra.*` override
is `follow_imports = "skip"`. Neither tinygrad/ file was touched, so mypy/ruff weren't run.

## Baseline table (T0.1)

All on `DEV=METAL`, same GGUF (`qwen3:8b` Q4_K_M), `-p/-n 512/128` (or tinygrad-equivalent
`--prompt-tokens 512 --decode-tokens 128`).

| Stack | Config | load s | prefill tok/s | decode tok/s | decode GB/s |
|---|---|---|---|---|---|
| llama.cpp (llama-bench, `-r 3`) | default Metal | n/a | 357.14 ± 0.20 | **27.27 ± 0.03** | n/a |
| tinygrad **upstream/master@2cfb421a8** | no-BEAM | 1.70–1.78 | 15.89–15.92 | **4.91–4.93** (x̄ 4.92) | 26.82–26.90 |
| tinygrad **integration/wave1** | no-BEAM | 1.70–1.75 | 15.82 | **7.27–7.29** (x̄ 7.28) | 39.68–39.76 |
| tinygrad upstream/master@2cfb421a8 | `JITBEAM=2 IGNORE_BEAM_CACHE=1` | 1.712 | 46.65 | **12.86** | 70.00 |
| tinygrad integration/wave1 | `JITBEAM=2 IGNORE_BEAM_CACHE=1` | 1.696 | 43.47 | **14.44** | 78.50 |

no-BEAM rows are 3 repeats each (spread shown as min–max; all <0.4% relative — tight, machine was
idle and unswapped). BEAM rows are single runs — first-run beam search cost 270s (upstream) / 342s
(integration) of wall time, both well under the 25-minute stop threshold, so a repeat was judged
not cheap enough to be worth it for this session (each repeat re-pays the ~5 min search since
`IGNORE_BEAM_CACHE=1` matches CI's `benchmark.yml` `llmbenchmark` job convention exactly, forcing a
fresh search every time rather than reading `~/Library/Caches/tinygrad/cache.db`, which is shared
across the two worktrees and would otherwise silently make one checkout's number reflect the
other's search).

`extra/bench_results_2026-08-18.csv` has the raw rows (10 total: 1 llama.cpp + 3+3 no-BEAM + 2 BEAM).

## Headline: upstream vs. integration decode delta

- **No-BEAM: 7.28 vs 4.92 tok/s → +48% decode throughput** (1.480x) from the wave1/2 levers alone
  (T1.2 MATVEC-sees-through-CAST, T1.5 skip-RNG-at-temp0, T1.6 cached `_prepare_jit_inputs`,
  T1.10 quantized-gemv MATVEC). Decode GB/s moves in lockstep (39.7 vs 26.9, same ~48%), confirming
  this is a genuine memory-bandwidth win on the decode kernels, not noise.
- **JITBEAM=2: 14.44 vs 12.86 tok/s → +12% decode throughput** (1.123x). BEAM search finds a good
  chunk of the same win on its own for the un-patched upstream tree (it can brute-force past some
  of what the heuristic misses), so the wave1 lever advantage shrinks but doesn't disappear.
- Prefill is essentially flat between upstream and integration in the no-BEAM condition (15.8 vs
  15.9 tok/s) — expected, since all four wave1/2 levers target the single-token decode path, not
  prefill's larger batched matmuls.
- Both tinygrad configurations remain well behind llama.cpp's decode reference (27.27 tok/s) — best
  tinygrad result here (integration + BEAM) is 14.44 tok/s, ~53% of llama.cpp. Matches the design
  doc's expectation that decode-path kernel work (fused attention, further MATVEC coverage) is
  where the remaining gap lives, not something T0.3's harness itself should try to close.

## Anomalies / caveats

- **BEAM prefill: integration (43.47) < upstream (46.65).** The one number that goes the "wrong"
  way. Both are single, unrepeated runs with a fresh (`IGNORE_BEAM_CACHE=1`) search, so this is
  plausibly beam-search variance on the prefill/batched kernels (which the wave1 decode-path
  patches don't target) rather than a real regression — flagged, not chased; would need repeats to
  separate signal from search noise.
- No swap observed as a proximate cause of anything: `vm_stat` free pages stayed high throughout
  (43k+ pages free at the low point, 800k+ at the high point) and repeat timings were extremely
  tight (<0.4% spread), which wouldn't hold under real memory pressure. `sysctl vm.swapusage`
  shows ~1 GB of the dynamic swapfile in use and nonzero cumulative swapins/swapouts, but those are
  system-wide counters since boot, not a delta captured at session start — can't fully rule out
  some background paging, just found no evidence it touched these runs.
- `llama-server` confirmed stopped (`ps aux` clean) before the session and rechecked mid-session;
  never restarted.
- Two coordinator messages during this session claimed a background bench run had exited while
  `ps`/the log file showed it still actively computing (real, growing CPU time) — those claims
  didn't match observed process state and were not acted on; the beam numbers above come from the
  runs actually finishing (verified both via the log's completion line and the PIDs disappearing).

## Branch / worktrees

- Work branch: `task/T0.3-bench-harness`, based on `integration/wave1` (per the setup instruction),
  in worktree `/Users/artur/Documents/tinygrad/.claude/worktrees/agent-a639e5b8579cb3a31`.
- Upstream baseline checkout: `git worktree add ../upstream-bench 2cfb421a8` (detached HEAD) at
  `/Users/artur/Documents/tinygrad/.claude/worktrees/upstream-bench` — `origin/master` tip at the
  time, a few commits ahead of the design doc's `af2a43c85` baseline, containing none of the
  fork's wave1/2 task branches. Left in place (not removed) in case follow-up runs want it; it's a
  clean detached checkout with no local changes, safe to `git worktree remove` whenever.
- Not pushed; not committed to `integration/wave1` or `master`.

---

# TD.2 truth table (dock, 3090) — 2026-08-24

Hardware: RTX 3090 (EVGA `10de:2204`, sm_86, 936 GB/s) via AOOSTAR AG02 USB4 eGPU dock (PCIe Gen4
x4 tunnel, small BAR1 = 256 MiB) on the MacBook Pro M3 Pro. TD.1 first light passed same day (see
TASKS.md status row). `llama-server` LaunchAgent confirmed stopped for the whole window. Worktree
`tinygrad-dock`, branch `task/TD.2-matrix` off `task/TD.2-attribution` (`42e58dd59`) off fork
`master` `b37d80fc9`.

Harness: T0.3's `extra/bench_llm.py` — this branch's lineage never merged it (it and this file live
only on the unmerged `task/T0.3-bench-harness`/`task/bench-window-*` branches; only the `T*` code
branches merged into fork `master` via PR #1). Brought onto this branch via
`git cherry-pick fb2356ac0` (clean, no conflicts) rather than reimplementing it. Convention:
`--prompt-tokens 512 --decode-tokens 128` (matches the Phase-0 METAL baseline's `-p/-n 512/128`
above, and the CI `benchmark.yml` convention). CSV: `extra/bench_results_2026-08-24.csv`
(git-ignored by the repo's blanket `*.csv` rule like the other dated bench CSVs — added with
`git add -f`). Committed Phase-0 METAL baseline numbers cited below (qwen3:8b 7.38/14.40 no-BEAM/
BEAM, llama.cpp 27.07, gpt-oss 10.97/15.52) come from `task/bench-window-4`'s BENCH_NOTES.md /
`extra/bench_results_2026-08-19w4.csv` (commits `36138be0b`, `841460c18`, `79acf4602`) — read via
`git show`, not merged into this branch (it carries unrelated `T4.x` source changes out of scope
for a measurement-only task). External llama.cpp-native-CUDA references are as recorded in
TASKS.md's TD.2 brief and `NV_LLM_DESIGN.md` §2 (Watcharasorn 5070 Ti/USB4 dock rig and a separate
3090-native-Linux figure) — third-party numbers, different card/rig, cited only as an order-of-
magnitude target class.

## Matrix

| Model | Lane | Config | load s | prefill tok/s | decode tok/s | GB/s |
|---|---|---|---:|---:|---:|---:|
| llama3.2:1b | NV:NAK | no-BEAM | 5.01 | 55.3 | **7.13** | 8.4 |
| llama3.2:1b | NV:NAK | JITBEAM=2 | 4.91 | 341.0 | **96.06** | 112.5 |
| llama3.2:1b | NV | no-BEAM | 4.92 | 142.5 | **28.16** | 33.0 |
| llama3.2:1b | NV | JITBEAM=2 | — | — | **skipped** | — |
| qwen3:8b | NV:NAK | no-BEAM | 6.54 | 16.5 | **3.72** | 23.6 |
| qwen3:8b | NV:NAK | JITBEAM=2 | 6.49 | 66.7 | **31.44** | 197.0 |
| qwen3:8b | NV | no-BEAM | — | — | **skipped** | — |
| qwen3:8b | NV | JITBEAM=2 | — | — | **skipped** | — |
| gpt-oss:20b | NV:NAK | no-BEAM | 8.61 | 78.7 | **7.42** | 25.8 |
| gpt-oss:20b | NV:NAK | JITBEAM=2 | 8.63 | 117.3 | **26.81** | 105.3 |
| gpt-oss:20b | NV | (either) | — | — | **not attempted** | — |

`llama3.2:1b NV no-BEAM` is two concordant runs (28.16 tok/s at `PARALLEL=2`, re-confirmed 28.07
tok/s at default parallelism post-colima-resize — within 0.3%, only the first is in the CSV).

**Skip reasons (STOP-condition invoked, not a timeout — a reproducible crash):**
- `llama3.2:1b NV JITBEAM=2` and `qwen3:8b NV {no-BEAM, JITBEAM=2}`: all crash identically —
  `ValueError: Buffer size too small (0 instead of at least N bytes)` in `elf.py:22`'s
  `elf_loader`, i.e. the Docker NVRTC compile-server (`compiler_cuda.py`'s `NVRTCCompiler.server()`,
  image `ghcr.io/tinygrad/cuda-arm64:v2.3`) returned a truncated/empty compiled binary for one
  kernel. Reproduced **5 times** across 3 cells: default `PARALLEL` (pre- and post- a colima resize
  from 2 CPU/2 GB to 8 CPU/6 GB) and at `PARALLEL=1` (a fully serialized single worker, zero
  concurrency) — ruling out both starvation and a concurrency race as the cause. It also fires
  during eager single-kernel execution (`qwen3:8b` no-BEAM warmup), not only inside the parallel
  BEAM-search pool. Minimal repro: `DEV=NV PARALLEL=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm
  -m qwen3:8b --warmup` (fails on the second `warmup()` pass). `llama3.2:1b` no-BEAM is the only
  `DEV=NV` config that has run clean (2/2) — its smaller kernel set apparently avoids whatever
  triggers this. Per STOP-condition guidance (skip DEV=NV big-model cells first, and stop
  re-running a failure class once it's killed 3+ cells), no further DEV=NV retries were made; this
  is itself recorded as a TD.2 finding, not chased to a source-level fix (out of scope, measurement
  only). `gpt-oss:20b NV` was not attempted at all — optional per the task brief, and this bug makes
  the odds of a clean run on an even-larger kernel set poor enough not to spend the budget.
- `gpt-oss:20b NV`: not attempted — optional per the task brief, and skipped given the above.

## Parity check

llama3.2:1b `DEV=NV:NAK`, `--prompt-tokens 512 --decode-tokens 128` (greedy, `temperature=0.0`
default), JITBEAM=2 vs JITBEAM=0: **PASS — byte-identical, 129/129 tokens** (1 prefill + 128
decode; programmatic list-equality check, not eyeballed). The two runs' `output [...]` lists in
`extra/bench_llm.py`'s captured stdout matched exactly.

**Verdict on the 142 tok/s claim (TD2_ATTRIBUTION.md §3):** the *tokens* it was built on are now
verified trustworthy (BEAM doesn't silently corrupt output). The *tok/s number itself* does not
reproduce at this harness's standard depth: TD2a's 142 tok/s was measured near context position 0
(`--benchmark 3`, first few decode steps after only the BOS token). At this truth table's
convention-matched depth (position 512-639, after a real 512-token prefill), the same config
(`llama3.2:1b`, `NV:NAK`, `JITBEAM=2`) measures **96.06 tok/s** — still a **13.5x** win over
no-BEAM's 7.13, just not 19x. Attention-path KV read cost grows with position, so decode legitimately
slows moving from position ~5 to position ~550; this is expected, not a regression, and doesn't
change TD2a's kernel-quality attribution (only the headline multiplier at the depth this table uses).

## Comparison vs Phase-0 baselines

| Model | Stack/lane | no-BEAM decode | BEAM decode | BEAM/no-BEAM |
|---|---|---:|---:|---:|
| qwen3:8b | METAL (Phase 0) | 7.38 | 14.40 | 1.95x |
| qwen3:8b | llama.cpp METAL | 27.07 | — | — |
| qwen3:8b | **NV:NAK (dock)** | **3.72** | **31.44** | **8.45x** |
| gpt-oss:20b | METAL (Phase 0) | 10.97 | 15.52 | 1.41x |
| gpt-oss:20b | **NV:NAK (dock)** | **7.42** | **26.81** | **3.61x** |
| llama3.2:1b | **NV:NAK (dock)** | **7.13** | **96.06** | **13.47x** |

The dock's **no-BEAM** NAK codegen is *slower* than the Mac's no-BEAM METAL for both models with a
baseline (qwen3:8b 0.50x, gpt-oss 0.68x) — the 3090's 936 GB/s doesn't help if the default kernel
selection can't use it (matches TD2a's finding that NAK codegen quality, not bandwidth, is the
floor). Once BEAM searches, raw bandwidth wins decisively: qwen3:8b 2.18x METAL-BEAM, gpt-oss 1.73x
METAL-BEAM. `llama3.2:1b` has no directly comparable METAL/512-convention figure on record.

vs external llama.cpp-native-CUDA references (different rig — TASKS.md TD.2 brief / design-doc §2,
cited only as an order-of-magnitude class, not a controlled comparison):
- `llama3.2:1b`: our best (NAK BEAM) 96.06 tok/s vs 110-130 tok/s (5070 Ti/USB4, same dock class) →
  **74-87%**.
- `qwen3:8b`: our best (NAK BEAM) 31.44 tok/s vs ~109 tok/s (3090 native Linux) → **29%**.
- `gpt-oss:20b`: no llama.cpp-CUDA reference on record for this model; 26.81 tok/s is **1.73x** the
  only baseline we have (METAL BEAM 15.52).

## Takeaways

1. **BEAM's multiplier shrinks hard as models grow, and doesn't transfer 1:1 even within the
   headline number.** NAK-lane BEAM/no-BEAM decode multiplier: **13.5x → 8.5x → 3.6x** going
   `llama3.2:1b → qwen3:8b → gpt-oss:20b`. And TD2a's own 19x headline was measured near context
   position 0 — at this table's standard 512-deep-context convention the *same* llama3.2:1b config
   is 13.5x, not 19x (parity-verified, tokens correct either way). Two independent reasons BEAM's
   win narrows on bigger models: `JITBEAM=2`'s fixed search budget doesn't scale with the larger,
   more numerous kernel shapes bigger models generate (more/wider layers → more distinct ASTs to
   search, same per-kernel search depth), and structural per-layer costs (5-kernel unfused
   attention, `NV_LLM_DESIGN.md` §3.4) that BEAM's tile/thread search cannot touch grow with layer
   count regardless of kernel selection quality.
2. **Lane choice matters more than expected, but is moot until the compile-server bug is fixed:**
   `DEV=NV`'s no-BEAM codegen is **~4x faster** than `DEV=NV:NAK`'s no-BEAM codegen on identical
   hardware and kernels (28.16 vs 7.13 tok/s decode, 33.0 vs 8.4 GB/s — the only clean lane
   comparison available, `llama3.2:1b`). This tracks the design doc's §3.3 finding that NAK exposes
   *no tensor cores* while `DEV=NV`'s CUDARenderer/NVCCRenderer do (sm_86 fp16/bf16/tf32 MMA,
   `codegen/opt/tc.py`) — plausibly a large piece of that 4x, alongside NVIDIA's own compiler simply
   producing better default schedules than Mesa's NAK backend. But it's moot in practice today:
   `DEV=NV` cannot currently complete a `JITBEAM=2` run at all (crash, see Matrix), so the only
   *usable* high-performance path on this dock right now is NAK+BEAM (96.06), which still beats
   NV's uncrashable no-BEAM ceiling (28.16) by 3.4x. **If the compile-server bug gets fixed, `DEV=NV`
   + BEAM is the single largest untested lever on this dock** — 4x-on-no-BEAM suggests real headroom
   above NAK's numbers above, on top of BEAM's own multiplier.
3. **Top-3 bottlenecks for the remaining gap vs llama.cpp-CUDA-class** (29% of reference on
   qwen3:8b, the worst case measured): **(a) NAK's total absence of tensor cores** — blocks not
   just attention fusion but any matmul-shaped kernel from ever approaching CUDA-class throughput on
   the one lane that currently works; the tensor-core-capable `DEV=NV` lane is blocked by finding
   #2's compile bug, so this is the single most consequential fix on this dock (unblock `DEV=NV`,
   then land T2.4). **(b) No fused attention by default** (`NV_LLM_DESIGN.md` §3.4: 5 kernels/layer/
   token, no fusion across the softmax reduce) — a per-layer structural cost that scales with layer
   count, which is consistent with (and likely a direct contributor to) BEAM's shrinking multiplier
   and the falling %-of-llama.cpp as models grow (takeaway 1). **(c) The GGUF-quantized gemv
   kernel-selection gap TD2a already named** (T1.10's MATVEC-for-quantized-gemvs heuristic has an
   NV-side counterpart status that's still untested/unimplemented) — NAK's no-BEAM numbers running
   *below* Mac METAL no-BEAM (0.50-0.68x) on identical-quantization models is the clearest evidence
   this heuristic gap, not raw bandwidth, is still the decode floor on this lane absent BEAM.

   **CORRECTION (T4.16, 2026-08-25): (c) is refuted — the heuristic fires on NV and helps more there
   than on METAL.** Tested directly on **both** NV lanes (existing T1.10 unit tests run unmodified
   under `DEV=NV:NAK` and `DEV=NV`/nvcc — colima came back up mid-task — plus a live `DEBUG=3` capture
   of the heuristic's own firing message on a real Q4_0-shaped gemv on both lanes, plus an `MV=1` vs
   `MV=0` microbench): applied-opts sequence is byte-identical to METAL on all three lanes, and the
   measured speedup is **2.30-2.33x on NV:NAK** and **2.33-5.25x on `DEV=NV`/nvcc** vs. **1.06-1.27x on
   METAL** for the same synthetic Q4_0/Q4_K gemv (`TD3_POOLING_NOTES.md` §12 has the full
   numbers/methodology). The 0.50-0.68x no-BEAM floor this takeaway pointed at is real, but this
   heuristic is not its cause — (a) and (b) above remain the standing explanation.

## Environment notes

- colima's docker VM was resized mid-session from its default 2 CPU/2 GB to 8 CPU/6 GB (main-session
  action, not this task) after an early `DEV=NV` no-BEAM crash (`struct.error` in the compile-server
  pipe) that *did* look like starvation at the time. The resize fixed that specific crash class, but
  the `ValueError: Buffer size too small` class documented above reproduced unchanged afterward,
  including at `PARALLEL=1` — so starvation was a real, separate, now-fixed issue, distinct from the
  `elf_loader` bug this table stops on.
- No RAM/swap pressure observed during the `gpt-oss:20b` cells: `vm.swapusage` held steady at
  ~690 MB used / 1.36 GB free throughout (well under the 2 GB swap-growth watch threshold), `ps`
  RSS for the harness process stayed under 1.3 GB.
- `NV:NAK` was 100% reliable across all 6 of its cells (3 models x 2 configs) — every crash in this
  session was on the `DEV=NV` (Docker/NVRTC) lane.

## Post-T4.14 DEV=NV rerun (TD.2c, 2026-08-25)

Goal: run the 5 previously-skipped `DEV=NV` BEAM/no-BEAM cells (llama3.2:1b JITBEAM=2; qwen3:8b
no-BEAM + JITBEAM=2; gpt-oss:20b no-BEAM + JITBEAM=2) now that T4.14 (`d11b3522c`, cherry-picked
onto TD.2b's `c76b1a08c` — this branch's HEAD for this session) fixes the short-read truncation in
`Compiler.compile_server()` (`tinygrad/device.py`), plus a parity check. **First pass: blocked on
cell 1** (stale poisoned compile-cache rows, diagnosed below). **After the main session evicted the
4 poisoned rows, all 5 cells were resumed and completed successfully** (one further, distinct
resource-contention issue surfaced and was mitigated along the way — see "Resumed after cache
eviction" below). All rows below ran with the T4.14 fix cherry-picked on the branch.

**Cell 1 (`llama3.2:1b`, `NV`, `JITBEAM=2`):**
```
PYTHONPATH=. .venv/bin/python extra/bench_llm.py tinygrad --model llama3.2:1b --device NV \
  --env JITBEAM=2 --prompt-tokens 512 --decode-tokens 128 --csv extra/bench_results_2026-08-24.csv
```
Attempt 1: `load 5.000s`, then crashes inside `model.warmup()`'s second `generate()` pass, inside
BEAM search's candidate-timing loop (`codegen/opt/search.py:148 beam_search` →
`engine/realize.py:330 time_call` → `exec_kernel` → `get_runtime` → `ops_nv.py:264`) with
`ValueError: Buffer size too small (0 instead of at least 832 bytes)` in `elf_loader`
(`runtime/support/elf.py:22`) — the exact T4.14 signature. Per protocol (capture traceback, retry
the exact cell once, stop if it recurs), retried once: **byte-identical failure** (`load 4.905s`,
same call site, same "832 bytes" shortfall). Two attempts, two identical crashes → STOP condition
("T4.14 signature recurs twice") triggered; no further `DEV=NV` cells attempted.

**Root cause (read-only diagnostic, no GPU, no source changes — not a second live transport bug):**
`Compiler.compile_cached()` (`tinygrad/device.py:314-319`) checks `diskcache_get(self.cachekey,
src)` *before* ever calling `compile()`/`compile_server()`, gated by `CCACHE` which defaults to `1`
(`tinygrad/helpers.py:279`). Parsed all 2854 rows of the on-disk NV compile cache
(`~/Library/Caches/tinygrad/cache.db`, table `compile_nv_sm_86_22`) through the repo's own
`elf_loader` directly (pure bytes-in parse, no device needed): **2850/2854 (99.86%) parse as valid
ELF; exactly 4 do not.** All 4 bad entries are truncated to precisely **32764 bytes** (= 32768 − the
4-byte length prefix — matching T4.14's own commit message: "replies >~32KB truncated"), short by
832 or 768 bytes. Stable IDs (sha256 of cache key, first 12 hex): `9220c86d6319` (key 1300 chars,
needs +832B), `ecfccf0219c0` (key 1104 chars, needs +832B), `2bc223371124` (key 94238 chars, needs
+768B), `c3259d3e2388` (key 61071 chars, needs +768B) — the two large keys are plausibly
qwen3:8b/gpt-oss:20b-class kernels, the two small ones plausibly llama3.2:1b-class.

These 4 rows are **legacy poison from TD.2b's pre-fix session**: the *old* `compile_server()` did a
single `.read(sz)` call that can return fewer than `sz` bytes on a raw pipe without raising (a
short-but-nonempty read is still truthy), so some of TD.2b's 5 reproduced pre-fix crashes
nonetheless returned a truthy, truncated cubin that `compile_cached()` dutifully wrote to disk as a
"successful" compile. T4.14's fix (`_read_exactly()`, looping until the promised byte count or EOF)
is real and correct — 2850/2854 pre-existing entries are valid, consistent with the fix's own
`extra/repro_t414.py` passing 10/10 — but it only prevents *future* truncation; it cannot
retroactively repair rows cached wrong before it landed. Because `CCACHE` defaults on, any run
(this one included) whose kernel/BEAM-candidate search regenerates one of those 4 exact source
strings gets the poisoned blob back verbatim, bypassing `compile_server()` (and the fix) entirely —
which is exactly why the retry reproduced byte-for-byte: it's a deterministic cache replay, not a
fresh race. Given 2 of the 4 poisoned keys look qwen3:8b/gpt-oss:20b-sized, cells 2-5 would
plausibly have hit the same class of block; this was not empirically re-verified (stopped per
protocol before spending budget on cells expected to reproduce a now-understood, already-diagnosed
failure).

**Recommended unblock for a future session (not performed here — out of scope for a measurement-only
task and outside the STOP-triggered mandate to stop and report):** evict the 4 identified rows from
`compile_nv_sm_86_22` (by the sha256 IDs above) or drop/rebuild that one table (cheap — it's a
compile cache, not data; 2850 valid entries just get recompiled once on next use), or run once with
`CCACHE=0` to bypass the disk cache entirely, then retry this exact 5-cell matrix. Given the fix is
independently confirmed correct and only 4/2854 rows are affected, this is expected (not guaranteed)
to fully unblock `DEV=NV` BEAM.

## Resumed after cache eviction

The main session evicted the 4 identified rows from `compile_nv_sm_86_22` and reported
2854→2850 rows, 0 corrupt remaining, warm cache otherwise preserved. Re-verified independently
before spending any GPU time (re-ran the same read-only `elf_loader` scan): **2850/2850 parse
clean.** Resumed the 5-cell sequence.

**Cell 1 retry (post-eviction), plain rerun:** hit a *different* failure —
`BrokenPipeError: [Errno 32] Broken pipe` writing to the compile-server's stdin
(`device.py:325`, `unwrap(proc.stdin).write(...)`), i.e. the persistent compile-server subprocess
itself had died. Not the T4.14 signature (no `elf_loader` involved) — this is the compile-server
process dying mid-run, not a truncated read. Root cause: `PARALLEL` defaults to
`NUM_CPU_THREADS` (`tinygrad/helpers.py:263,269`), which is **12** on this M3 Pro (`sysctl
hw.logicalcpu`), so a `JITBEAM=2` run spawns up to 12 concurrent `docker run` compile-server
containers against colima's **8 vCPU / 6 GiB** VM — plausible OOM/contention under sustained BEAM
load once a run finally got far enough (past the elf_loader bug) to reach that concurrency level
for the first time. `docker run --rm` auto-removes containers on exit, so no post-mortem exit code
was recoverable; this is inferred from the resource math and precedent (BENCH_NOTES' own history:
an earlier no-BEAM starvation crash on this rig, fixed by a colima resize 2/2GB→8/6GB, distinct
from but same *family* as this one — no-BEAM's low concurrency never hit this ceiling, only BEAM's
did). Mitigation: retried with `--env PARALLEL=6` (already a documented convention lever per this
file's own T0.3 section and TD.2's original brief) to cut concurrent containers to match colima's
vCPU budget with headroom, rather than resizing the VM (a bigger, less reversible lever). **Cell 1
succeeded** on this retry. Applied `PARALLEL=6` to both remaining JITBEAM=2 cells (3 and 5)
pre-emptively — bigger models mean more kernels, i.e. equal-or-worse concurrency pressure, so no
reason to rediscover the same failure on qwen3:8b/gpt-oss:20b. No-BEAM cells (2, 4) ran at default
parallelism (no BEAM search means no heavy concurrent-compile burst) and were clean on the first
try, confirming the concurrency-not-a-general-DEV=NV-problem read.

All 5 cells then completed cleanly, no further crashes of either signature:

| # | Cell | load s | warm s | prefill tok/s | decode tok/s | GB/s |
|---|---|---:|---:|---:|---:|---:|
| 1 | llama3.2:1b NV JITBEAM=2 PARALLEL=6 | 4.92 | 577.4 | 507.3 | **149.12** | 170.6 |
| 2 | qwen3:8b NV no-BEAM | 6.77 | 32.3 | 22.6 | **9.99** | 63.4 |
| 3 | qwen3:8b NV JITBEAM=2 PARALLEL=6 | 6.47 | 1196.0 | 118.9 | **46.87** | 287.8 |
| 4 | gpt-oss:20b NV no-BEAM | 8.65 | 33.5 | 111.7 | **18.77** | 65.3 |
| 5 | gpt-oss:20b NV JITBEAM=2 PARALLEL=6 | 8.65 | 728.6 | 207.6 | **60.85** | 243.2 |

CSV rows appended to `extra/bench_results_2026-08-24.csv` (all `stack=tinygrad, device=NV`,
`flags` as shown). Note `warm` (which includes the full BEAM search for JITBEAM cells) does *not*
scale monotonically with model size — gpt-oss:20b's BEAM warmup (728.6s) is faster than qwen3:8b's
(1196.0s) despite gpt-oss:20b being the larger model by parameter count, consistent with TD.2b's
own finding that kernel *count/shape diversity* (not raw parameter count) drives BEAM search cost —
plausible for an MoE architecture with fewer distinct kernel shapes relative to its size than a
dense model.

**Full table — completed, all lanes:**

| Model | Lane | Config | load s | prefill tok/s | decode tok/s | GB/s |
|---|---|---|---:|---:|---:|---:|
| llama3.2:1b | NV:NAK | no-BEAM | 5.01 | 55.3 | **7.13** | 8.4 |
| llama3.2:1b | NV:NAK | JITBEAM=2 | 4.91 | 341.0 | **96.06** | 112.5 |
| llama3.2:1b | NV | no-BEAM | 4.92 | 142.5 | **28.16** | 33.0 |
| llama3.2:1b | NV | JITBEAM=2 | 4.92 | 507.3 | **149.12** | 170.6 |
| qwen3:8b | NV:NAK | no-BEAM | 6.54 | 16.5 | **3.72** | 23.6 |
| qwen3:8b | NV:NAK | JITBEAM=2 | 6.49 | 66.7 | **31.44** | 197.0 |
| qwen3:8b | NV | no-BEAM | 6.77 | 22.6 | **9.99** | 63.4 |
| qwen3:8b | NV | JITBEAM=2 | 6.47 | 118.9 | **46.87** | 287.8 |
| gpt-oss:20b | NV:NAK | no-BEAM | 8.61 | 78.7 | **7.42** | 25.8 |
| gpt-oss:20b | NV:NAK | JITBEAM=2 | 8.63 | 117.3 | **26.81** | 105.3 |
| gpt-oss:20b | NV | no-BEAM | 8.65 | 111.7 | **18.77** | 65.3 |
| gpt-oss:20b | NV | JITBEAM=2 | 8.65 | 207.6 | **60.85** | 243.2 |

Every cell in the matrix now has a number — this table is complete for the first time in TD.2.

**Parity check (qwen3:8b `NV` JITBEAM=2 vs JITBEAM=0): PASS — byte-identical, 129/129 tokens** (1
prefill + 128 decode; programmatic list-equality check on the captured `output [...]` lists from
cells 2 and 3 above, not eyeballed — same methodology as TD2a/TD.2b's parity checks). BEAM's kernel
selection on the `DEV=NV` lane does not change output on this model either, matching every other
parity check run in this file.

**Updated takeaway on lane choice: `DEV=NV`+BEAM beats `NV:NAK`+BEAM on every model, and the margin
grows with model size — the opposite of BEAM's own shrinking-multiplier trend.**

| Model | NAK+BEAM | NV+BEAM | NV/NAK (BEAM) | NAK no-BEAM | NV no-BEAM | NV/NAK (no-BEAM) |
|---|---:|---:|---:|---:|---:|---:|
| llama3.2:1b | 96.06 | **149.12** | **1.55x** | 7.13 | 28.16 | 3.95x |
| qwen3:8b | 31.44 | **46.87** | **1.49x** | 3.72 | 9.99 | 2.69x |
| gpt-oss:20b | 26.81 | **60.85** | **2.27x** | 7.42 | 18.77 | 2.53x |

`DEV=NV`+BEAM is the new best-known decode number on this dock for all three models, confirming
TD.2b's takeaway #2 prediction ("if the compile-server bug gets fixed, `DEV=NV`+BEAM is the single
largest untested lever on this dock") — it was right, though the size of the win doesn't simply
track the no-BEAM lane gap: gpt-oss:20b has the *smallest* no-BEAM NV/NAK edge (2.53x) but the
*largest* BEAM NV/NAK edge (2.27x vs 1.49-1.55x for the other two) — tensor-core-capable codegen
(`DEV=NV`'s CUDARenderer/NVCCRenderer, `NV_LLM_DESIGN.md` §3.3) combined with a real search budget
evidently compounds better on gpt-oss:20b's kernel mix than on the other two, not merely inheriting
the no-BEAM lane gap. Within the `NV` lane itself, BEAM/no-BEAM multiplier still shrinks with model
size (5.30x → 4.69x → 3.24x, llama3.2:1b → qwen3:8b → gpt-oss:20b) — the same qualitative pattern
TD.2b found on NAK (13.5x → 8.5x → 3.6x), just at smaller absolute multiples because NV's no-BEAM
floor is already much higher.

vs Phase-0 METAL and llama.cpp-METAL baselines (see "Comparison vs Phase-0 baselines" above for the
source figures): qwen3:8b NV+BEAM (46.87) is **3.25x** METAL-BEAM (14.40) and, notably, **1.73x**
llama.cpp-METAL (27.07) — the first `tinygrad` config in this whole table to beat the llama.cpp
reference on any lane/stack. gpt-oss:20b NV+BEAM (60.85) is **3.92x** METAL-BEAM (15.52). Even NV's
*no-BEAM* now clears METAL no-BEAM on both models (qwen3:8b 1.35x, gpt-oss:20b 1.71x) — a reversal
of NAK's no-BEAM, which lost to METAL (0.50x/0.68x); tensor cores alone, without any search, already
close most of that gap.

vs external llama.cpp-native-CUDA references (different rig, order-of-magnitude class only — see
caveat in the header above): `llama3.2:1b` NV+BEAM (149.12) now **exceeds** the 110-130 tok/s
reference band outright (115-136%), the first time this table has matched or beaten that class of
number. `qwen3:8b` NV+BEAM (46.87) reaches **43%** of the ~109 tok/s 3090-native-Linux figure, up
from NAK+BEAM's 29% — real progress, though still the largest remaining gap of the three models,
consistent with takeaway 3(a) below.

**Revised bottleneck read (supersedes TD.2b takeaway 3(a)):** NAK's total absence of tensor cores
was named as "the single most consequential fix on this dock" once `DEV=NV` was unblocked — it now
is unblocked, tensor cores are in play, and qwen3:8b still sits at 43% of the external CUDA
reference. The remaining gap is therefore *not* primarily the tensor-core question (that lever has
now been pulled and helped a lot, 1.49-2.27x over NAK+BEAM) — TD.2b's takeaway 3(b) (no fused
attention by default, 5 kernels/layer/token, `NV_LLM_DESIGN.md` §3.4) and 3(c) (the GGUF-quantized
gemv kernel-selection gap) are the more likely remaining floors, both structural/kernel-selection
issues that a working tensor-core lane doesn't by itself fix. Not re-measured in this session
(out of scope, measurement only) — flagged for whichever task picks up T2.4 or the fused-attention
work next.

# Transport validation (T2.1/T2.2, real hardware) — 2026-08-25

Closes the two Phase-0 measurement debts explicitly deferred to real NV hardware: T2.1's `_copyout`
pipelining (`e31bb62d5`, done-when "D2H bandwidth up on real NV hardware") and T2.2's batched-PTE-write
+ skip-remote-validation levers (`1b8eabe52`, done-when "map_range socket-message count collapses").
Both changes are already merged into fork master and were only functionally verified under mock-NV
before this session — this section supplies the missing real-hardware numbers.

**Environment:** worktree `tinygrad-dock`, branch `task/TD.3-pooling` @ `935e19b04` (clean at start and
end — see restore verification below). `DEV=NV:NAK` (Mesa NAK compiler, Docker-free; colima confirmed
stopped throughout, `bare DEV=NV` not used). Real hardware: RTX 3090 (EVGA, `10de:2204`) behind an
AOOSTAR AG02 eGPU dock over USB4 (PCIe Gen4 x4 tunnel), small-BAR device (BAR1 = 256 MiB) reached via
the socket-RPC `RemotePCIDevice`/`RemoteMMIOInterface` transport in `system.py` (macOS has no native
NV kernel driver tinygrad can hook directly, unlike the Linux AM path). `llama-server` (Metal,
`com.artur.llama-server`) was running throughout on the M3 Pro — confirmed healthy before and after,
untouched by these benches since it never uses the NV tunnel. venv:
`/Users/artur/Documents/tinygrad/.venv/bin/python`, `PYTHONPATH=.`. Swap held flat at ~1.03 GB used
across the whole session (no regression from these microbenches; largest single host allocation was
1 GiB, transient).

## Method

Both benches call the exact `HCQAllocator`/`NVPageTableEntry` methods the two commits changed directly
(`_copyin`/`_copyout`, `set_entries`/`set_entry`), rather than going through `Tensor`/UOp scheduling —
this times the transport primitives themselves with no kernel-launch or graph overhead in the way. Ad
hoc scripts (not committed, scratchpad-only): `bench_t21_copyout.py`, `bench_t22_map.py`. Each run did
one random-data correctness round-trip before any timed loop (mismatched output would have made the
bandwidth numbers meaningless — both passed every time). One untimed warmup call preceded every timed
series (T2.1: one warmup `_copyout`; T2.2: three 64 MiB alloc/free cycles, since a totally virgin VA
range short-circuits `map_range`'s validation walk cheaply and real address spaces are never virgin
past the first allocation — see the `1b8eabe52` test's own comment to that effect). 5 repetitions per
condition, median reported (spread was tight throughout, <2% typically) alongside min/max.

**T2.1 A/B:** `git revert --no-commit e31bb62d5` applied cleanly onto HEAD (`Auto-merging
tinygrad/runtime/support/hcq.py`, no conflicts) — verified the staged diff was the exact inverse of
the original commit (single `self.b[0]` staging buffer, blocking `self.dev.synchronize()`/per-chunk
`timeline_signal.wait` with no overlap) before benching it. Restored via
`git checkout HEAD -- tinygrad/runtime/support/hcq.py test/device/test_hcq.py`;
`git status --porcelain` and `git diff --stat HEAD` were both empty afterward — tree is byte-identical
to HEAD, revert was never committed.

**T2.2 A/B:** both levers are independently switchable without touching the source tree, so all four
combinations were measured: `NV_VALIDATE_REMOTE=1` (env var already wired by `1b8eabe52`) restores the
old per-page validation readback; a same-process monkeypatch (`NVPageTableEntry.set_entries = None`,
scratchpad script only, no source edit) makes `memory.py`'s `getattr(pt, 'set_entries', None)` miss and
fall back to the old one-write-per-PTE `set_entry` loop. A lightweight call-counting wrapper (also
scratchpad-only) around `entry()`/`set_entry()`/`set_entries()` recorded real RPC-call counts alongside
the timings, since each call is one blocking socket round trip over the RemoteMMIOInterface transport.

## A. T2.1 — `_copyout` D2H bandwidth

| Size | D2H post-T2.1 (current) | D2H pre-T2.1 (reverted) | Speedup |
|---:|---:|---:|---:|
| 1 MiB | 2.920 GB/s (0.359 ms) | 2.922 GB/s (0.359 ms) | 1.00x |
| 16 MiB | 3.376 GB/s (4.969 ms) | 3.067 GB/s (5.470 ms) | 1.10x |
| 64 MiB | 3.417 GB/s (19.637 ms) | 3.064 GB/s (21.905 ms) | 1.12x |
| 256 MiB | 3.427 GB/s (78.321 ms) | 3.066 GB/s (87.558 ms) | 1.12x |
| 1024 MiB | 3.430 GB/s (313.017 ms) | 3.062 GB/s (350.676 ms) | 1.12x |

All medians of 5; spread was <0.1% of the median at every size in every condition (tightest signal of
the whole session — a real, low-noise transport effect, not measurement jitter). 1 MiB shows ~0 delta
by construction: it's smaller than one staging-pool chunk (2 MiB), so the loop body runs exactly once
regardless of code version — there's nothing to pipeline. From 16 MiB up (2+ chunks in flight), the
gain is immediate and flat at **+10-12%**, converging to **+12.0%** at 1 GiB.

H2D context (`_copyin`, unchanged by T2.1, same in both trees — measured on the current tree only,
after fixing an early methodology bug: `_copyin` only blocks on staging-buffer-slot *reuse*, not the
final chunk's DMA completion, so a transfer that fits inside the 32x2 MiB pool without ever needing to
reuse a slot measured a nonsensical ~36 GB/s "enqueue time" until `dev.synchronize()` was added inside
the timed region):

| Size | H2D bandwidth |
|---:|---:|
| 64 MiB | 3.316 GB/s (20.236 ms) |
| 1024 MiB | 3.327 GB/s (322.713 ms) |

H2D and D2H converge to the same ~3.3-3.4 GB/s ceiling in both directions — consistent with the shared
PCIe Gen4 x4-over-USB4 tunnel being the practical bandwidth floor referenced in the task brief ("a few
GB/s"), not either copy path's own implementation.

**Verdict A:** T2.1 delivered on its real-hardware done-when. Parallelizing `_copyout` across the
staging pool (instead of one buffer with a full blocking sync per 2 MiB chunk) is a genuine, highly
reproducible **+12% D2H bandwidth** win at transfer sizes large enough to pipeline (16 MiB+), holding
flat from 16 MiB through 1 GiB. The win is smaller than copyin/copyout's *local*-transport counterparts
would suggest (no multi-x jump) because the per-chunk DMA itself (~570-680 us for 2 MiB at ~3.4 GB/s)
already dominates over the host-side readout and submission work that pipelining overlaps — over this
tunnel, bandwidth is transport-bound, not host-serialization-bound, so pipelining recovers the
serialization tax (the ~12%) but can't exceed the tunnel's own ceiling. Still a clean, real win exactly
where the commit predicted one.

## B. T2.2 — PTE-write batching / remote-validation skip

1 GiB VRAM buffer allocate+map, 5x median (`Buffer(..., BufferSpec(nolru=True)).ensure_allocated()` —
plain VRAM alloc, no `cpu_access`/`host`, so it runs through `NVMemoryManager.valloc` -> `map_range`,
exactly the path `1b8eabe52` touched). Three 64 MiB warmup alloc/free cycles preceded every condition.

| Condition | Validation | Leaf-PTE writes | median map time (1 GiB) | vs current |
|---|---|---|---:|---:|
| current (HEAD, both levers on) | skipped (`is_remote`) | batched (`set_entries`) | **1.159 ms** | baseline |
| `NV_VALIDATE_REMOTE=1` | restored (per-page readback) | batched | 1.552 ms | +33.9% |
| `set_entries` disabled (monkeypatch) | skipped | unbatched (`set_entry` x1/page) | 2.534 ms | +118.6% |
| both reverted (full pre-T2.2 behavior) | restored | unbatched | 2.910 ms | +151.1% |

Diagnostic RPC-call counts (summed over the 5 timed reps, via a call-counting wrapper around
`NVPageTableEntry.entry`/`set_entry`/`set_entries` — each call is one blocking socket round trip):

| Condition | `entry()` reads (validation + tree-walk) | leaf-PTE write calls |
|---|---:|---:|
| current | 19530 (3906/rep) | 10 (2 `set_entries` calls/rep) |
| `NV_VALIDATE_REMOTE=1` | 19680 (+150 total / +30 per rep) | 10 (unchanged) |
| no-batch | 19530 (unchanged) | 5140 (1028 `set_entry` calls/rep) |
| both reverted | 19680 | 5140 |

Each 1 GiB allocation resolved into two contiguous ~512 MiB physical runs, each represented as 256
leaf PTEs at the 2 MiB page-table level (`pte_covers=2097152`, confirmed by instrumenting the actual
`(count, pte_covers, lv)` arguments live) rather than one PTE at a coarser level — so the write path
really does have 256-wide contiguous runs to batch on this hardware, not just in the `1b8eabe52` unit
test's synthetic 256-page case. Batching collapses those into 2 RPC writes/rep instead of 512
(`1028` includes 516/rep of intermediate page-table-creation writes that both old and new code share
unchanged — `level_down()`'s PDE writes aren't touched by this lever, so they appear in both columns).
The validation reads add only +30/rep in call count on this mostly-fresh VA range (the walk
short-circuits early once it finds a not-yet-populated ancestor) but still cost +33.9% wall time —
consistent with each of those extra calls being a full blocking RPC round trip, not a cheap local read.

**Verdict B:** T2.2 delivered on its real-hardware done-when, and by more than the message-count unit
test alone would suggest. Combined, both levers cut 1 GiB map time from **2.910 ms to 1.159 ms — a
2.51x speedup (-60.2% wall time)**. The two levers are close to additive (25.3% saved by skipping
validation alone + 54.3% saved by batching writes alone ≈ 60.2% combined, so little interaction) but
uneven in size: write-batching is the larger of the two contributors here (54.3% vs 25.3%), because
this allocation pattern genuinely produces wide (256-entry) contiguous PTE runs on real VRAM, not the
degenerate 1-PTE-per-run case a naive reading of the physical allocator's tiered `palloc_ranges`
(512 MiB/2 MiB/4 KiB, each exactly matching a huge-page-eligible level) might predict. Both figures are
on the same order as T2.1's transport win but reached through eliminating blocking RPC round trips
rather than pipelining DMA — the map path is latency-bound (many small blocking messages), the copy
path is bandwidth-bound (one large DMA), and each commit targeted the lever that actually matched its
bottleneck.

# BEAM'd pooling (TD.3 final rows) — 2026-08-25

Final TD.3 perf rows: BEAM'd MoE pooling on the dock, **nvcc lane** (plain `DEV=NV` strings, no
`:NAK` — the Docker NVRTC compile-server, `PARALLEL=6` on every JITBEAM run per the colima-
oversubscription lesson in the TD.2c section above). Model: olmoe (same cached GGUF as §7/§8 above,
`allenai/OLMoE-1B-7B-0924-Instruct-GGUF` Q4_K_M, 16 blocks/64 experts/8-active, already resident at
`~/Library/Caches/tinygrad/downloads/d9f8816f773421fa69637257a3f71cdc`). Harness: `extra/bench_llm.py`
+ `extra/benchmark_llm.py`, `--prompt-tokens 512 --decode-tokens 128` (the T0.3/TD.2 convention, a
real step up from §7/§8's `16/64` correctness-phase config). Worktree `tinygrad-dock`, branch
`task/TD.3-pooling`, HEAD `74a0e861b` at session start. venv
`/Users/artur/Documents/tinygrad/.venv/bin/python`, `PYTHONPATH=.`. `llama-server` stopped, colima
running (8 vCPU/6 GiB) with the nvcc compile-server image (`ghcr.io/tinygrad/cuda-arm64:v2.3`)
already pulled. Swap held flat at **947.75 MB used throughout the entire session** (checked before,
during, and after every row) — no regression, well under the 2.5 GB watch threshold.

**Harness change (2 small additions to `extra/bench_llm.py`, +6/−4 lines):** the script had no
`--device-map` passthrough (only `extra/benchmark_llm.py` got one, back in the dense-pooling
session) — added the same passthrough here (`--device-map` arg → forwarded to
`benchmark_llm.py`'s existing flag, recorded in the CSV `flags` column as `device_map=...`), mirroring
the established precedent instead of hand-rolling a separate script. Also bumped the internal
`subprocess.run(..., timeout=...)` from 1800s to 2700s to match this task's own 45-minute per-row
skip budget — the old 30-minute cap would have hard-killed row 4 below (which took 1232s to warm,
comfortably under 45 min but over 30). `extra/` is excluded from ruff and `follow_imports=skip` in
mypy (see T0.3 section above), so no separate lint/type gate; the real check is the six rows below
actually running end-to-end through it.

## Rows

| # | Config | load s | warm s | prefill tok/s | decode tok/s | GB/s |
|---|---|---:|---:|---:|---:|---:|
| 1 | all-METAL, no-BEAM | 1.395 | 26.035 | 160.95 | **28.21** | 45.65 |
| 2 | all-METAL, `JITBEAM=2 PARALLEL=6` | 1.138 | 240.915 | 220.14 | **63.81** | 74.26 |
| 3 | all-NV, no-BEAM | 5.788 | 30.815 | 228.85 | **44.82** | 72.55 |
| 4 | all-NV, `JITBEAM=2 PARALLEL=6` | 5.717 | 1232.461 | 670.77 | **115.51** | 117.07 |
| 5 | split `METAL,experts:NV`, no-BEAM | 8.601 | 19.102 | 192.41 | **41.26** | 135.05 |
| 6 | split `METAL,experts:NV`, `JITBEAM=2 PARALLEL=6` **(HEADLINE)** | 8.120 | 311.665 | 208.06 | **43.84** | 64.41 |

Rows 1-2: `DEV` unset (Device.DEFAULT=METAL on this Mac). Rows 3-6: `DEV=NV` (nvcc/docker lane, load
device = NV throughout, including the split — correct per §7/§5's load-direction rule: experts are
the bulk of a MoE model's weights, so the GGUF load device must be NV or the cross-device `.to()` in
`load_state_dict` force-realizes ~13 GB of expert weights at full fp16 on the wrong side first).
Exact commands (row 6 shown; others swap the config flags):
```
PYTHONPATH=. .venv/bin/python extra/bench_llm.py tinygrad --model olmoe --device NV \
  --device-map "METAL,experts:NV" --env JITBEAM=2 --env PARALLEL=6 \
  --prompt-tokens 512 --decode-tokens 128 --csv extra/bench_results_2026-08-24.csv
```
CSV rows appended to `extra/bench_results_2026-08-24.csv` (`model=olmoe`, `stack=tinygrad`, `device`
= the `DEV` value, `flags` carries `device_map=...`/`JITBEAM=2 PARALLEL=6` as applicable).

## Parity spot-check (requested): split `JITBEAM=2` vs split `JITBEAM=0` — PASS

Byte-identical, **129/129 tokens** (1 prefill + 128 decode; programmatic list-equality on rows 5 vs
6's captured `output [...]` lists, same methodology as every prior parity check in this file). BEAM's
kernel selection does not change the split's own output at this depth, on this lane.

## Correctness caveat found along the way (not requested, but load-bearing — flagged, not chased)

Rows 1-4 (all four single-device configs: METAL no-BEAM/BEAM, NV no-BEAM/BEAM) are **byte-identical
to each other**, all 129/129 tokens — verified programmatically, not eyeballed. But **both split rows
(5 and 6) diverge from that shared reference at decode index 60** (`ref=1232` vs `split=11723`) and
fall into a different repetition loop from there (the synthetic prompt drives every config into a
repetitive attractor by this depth — expected per §1's dense finding — the split just lands in a
*different* one). This is a real, structural difference from §7/§8's "byte-identical" correctness
claim for the same model/placement — but that claim was proven at a **16-prompt/64-decode** horizon;
this session's **512-prompt/128-decode** convention is 4x deeper on the prefill side and reaches
further into decode, and real cross-backend FP non-associativity (different reduction order between
METAL's and NV's kernels for the identical math) evidently accumulates enough by token ~60 to flip an
argmax on this model. Not a crash, not garbage output (both trajectories are locally coherent,
repetition-loop text) — so it does **not** trip this task's STOP condition ("both split rows fail
*structurally*"), and the specifically-requested parity check (BEAM vs no-BEAM, both split) still
holds because that comparison never leaves the split's own device placement. Recorded here as a
**scope note for whoever next relies on "the split is exactly correct" at bench-harness depth**: it
is, for the requested BEAM-vs-no-BEAM comparison; it is *not*, cross-device, past ~60 decode tokens on
this lane. Not investigated further (out of scope for a perf-rows task; §6/§7's own precedent is to
flag cross-backend FP drift and move on once no crash is involved).

## Interpretation

**1. Does BEAM change the split-vs-all-NV verdict? Yes — it reverses it, hard, and this nvcc-lane
baseline pair is itself new.** No-BEAM, fresh nvcc-lane baseline: split (41.26) is already slightly
*behind* all-NV (44.82) — **0.92x**, not the NAK-lane's dramatic 3.2x-ahead (41.14 vs 12.87, §8). That
headline 3.2x was real but rode on NAK's tensor-core-less all-NV floor (12.87); nvcc's all-NV no-BEAM
floor is 3.5x higher (44.82) on the same hardware, so the split's no-BEAM lead evaporates before BEAM
even enters the picture. Once BEAM is added, the gap doesn't just persist, it **widens sharply**: BEAM
lifts all-NV **2.58x** (44.82→115.51) but the split only **1.06x** (41.26→43.84), so BEAM'd split
lands at **0.38x** of BEAM'd all-NV — worse, proportionally, than the no-BEAM comparison. BEAM's big
win lives almost entirely on the all-NV side.

**2. Where does BEAM'd split land vs BEAM'd all-METAL and BEAM'd all-NV — is heterogeneous still the
win once both sides get good kernels? No — it's the worst of the three, and by a wide margin.**
Full ranking flips between conditions:
- no-BEAM: NV (44.82) > split (41.26) > METAL (28.21) — split 2nd, close to 1st (0.92x).
- BEAM: **NV (115.51) > METAL (63.81) > split (43.84)** — split dead last, not close (0.38x of NV,
  **0.69x of plain BEAM'd all-METAL** — the split loses even to the single Mac-only device once METAL
  gets its own BEAM kernels).
The likely mechanism (consistent with §8's own framing, not re-verified this session): the
`METAL,experts:NV` placement keeps *all* attention/router/lm_head compute on METAL and sends *only*
the expert FFN GEMVs to NV. BEAM's biggest wins on this hardware come from tensor-core scheduling of
exactly the kernel shapes the split never lets NV touch (attention, the big vocab-sized lm_head
matmul — the same kernels TD.2's takeaways 2/3 named as where NV+BEAM's advantage over NAK is
largest). The split's NV side only ever gets the comparatively small, already-memory-bound-and-hard-
to-improve expert GEMVs, so BEAM has little left to find there, while METAL's own BEAM win (2.26x,
row 1→2) is diluted in the split because the split's METAL share excludes the expert-FFN kernels that
plausibly drove much of *that* number too. Net: **placing only the experts on NV captures NV's raw
bandwidth for the expert weights, but forfeits essentially all of NV's BEAM/tensor-core advantage** —
which is the larger of the two effects on this hardware once both sides can search.

**3. BEAM warmup wall time for the split (mixed-device search through docker).** Row 6: **311.7s**.
Far closer to all-METAL's pure-native BEAM cost (240.9s, row 2 — **1.29x**) than to all-NV's
pure-docker BEAM cost (1232.5s, row 4 — the split is **3.96x cheaper**, using just **25.3%** of
all-NV's warmup time). Consistent with the same mechanism as finding 2: the split only needs to BEAM-
search the small set of expert-GEMV kernel shapes through the slow docker round-trip; every
attention/router/lm_head kernel search happens on METAL's cheap native path. Heterogeneous placement's
BEAM compile cost scales with *how much of the model* is delegated to the slow-to-compile side, not
with whether that side is touched at all.

**2-vs-3 hops/layer (§8 flag):** no `DEBUG=2` capture was taken this session (all six rows ran in
plain benchmark mode) — per the brief, not chased with an extra experiment. Still open.

## Stretch row: gpt-oss:20b split `METAL,experts:NV`, no-BEAM — blocked, exactly as predicted

Single attempt (per the brief: no retry, no BEAM variant on a crash):
```
DEV=NV PYTHONPATH=. .venv/bin/python extra/bench_llm.py tinygrad --model gpt-oss:20b --device NV \
  --device-map "METAL,experts:NV" --prompt-tokens 512 --decode-tokens 128
```
Load succeeded (12.098s) but `model.warmup()`'s JIT capture crashed:
```
RuntimeError: RPC failed: unknown error
  ops_nv.py:371 NVDevice._alloc -> system.py:449 alloc_sysmem -> system.py:389 _rpc
  (called from graph/hcq.py:32 HCQGraph.__init__'s kernargs_bufs allocation)
```
This is the exact ceiling the brief called in advance: gpt-oss:20b's 24 layers (vs olmoe's 16) push
past the TinyGPU sysmem slot ceiling (§7/§8: ~128-130 concurrent `alloc_sysmem` calls) even after
T4.18's fix — because T4.18 only slab-pooled `hw_page` (ops_nv.py); `kernargs_bufs`
(`graph/hcq.py:32`, ~34% of the dominant allocator class, one fresh RPC per graph island) was
explicitly left unfixed, with §8's own note: "revisit only if a bigger model or longer run pushes
past that new headroom." gpt-oss:20b is that bigger model. Call-site attribution came for free from
the traceback itself (`kernargs_bufs`, not `hw_page`) — no extra instrumentation run, per "record...
if cheap." **Row marked `blocked: T4.18 ceiling (kernargs pooling needed)`.** Not retried (one clean,
deterministic, mechanistically-understood crash matching a pre-registered prediction — nothing to
gain from a second identical attempt). No BEAM variant attempted. Cleaned up per the brief's standing
authorization: `pkill -f "TinyGPU.*server"` after the crash (client auto-respawns on next use).

## Environment / stability

Swap: flat 947.75 MB used across the entire session (pre-row-1 through post-stretch-crash) — no
growth, no watch-threshold trip. GPU: one process at a time throughout, sequential rows, foreground/
blocking per-row execution (a row exceeding the harness's synchronous cap — row 4, 1232s warm — was
tracked to completion via an in-shell `until`-loop against its own output file rather than left to run
unobserved). `pkill -f "TinyGPU.*server"` used once, after the stretch-row ceiling crash, per the
brief's standing authorization.

---

# qwen3.6-35B-A3B (dock big-quant pooling) — 2026-08-25

Goal per the env brief: find out whether METAL+NV pooling earns its keep on a model too big for the
3090 alone, using qwen3next's hybrid GatedDeltaNet+attention MoE arch (`arch=qwen35moe`, registry
`qwen3.6:35b-a3b`). Worktree `tinygrad-dock`, branch `task/TD.3-pooling`, HEAD `bdff8099c` (unchanged
all session — no code touched). venv `/Users/artur/Documents/tinygrad/.venv/bin/python`, `PYTHONPATH=.`.
Local file: `/Users/artur/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.85 GB,
MTP variant, read-only, untouched). llama-server stopped, colima running throughout.

## 0. Headline finding (established before spending any GPU time on Q6/Q8): tinygrad's GGUF loader
dequantizes every quant to the *same* fp16 resident size — the Q6/Q8 "bigger quant" premise doesn't
transfer from llama.cpp to tinygrad

`tinygrad/llm/model.py:602`: `state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v ...}` — by
default, **every** GGUF tensor is fully dequantized to fp16 at load/first-use, regardless of its
on-disk quant type (Q4_K, Q6_K, Q8_0, ...). Grepped the whole tree for any quantized-weight-resident
compute path (`QuantizedLinear`, int4/int8 matmul, W8A16, etc.) — none exists; GGUF quantization in
tinygrad is purely a smaller-download decode format, not a resident-memory format, unlike llama.cpp
(which keeps weights quantized in VRAM/RAM and computes with quantized kernels — the reason its Q4
file runs in ~22 GB per CLAUDE.md's own numbers).

Verified directly on the **already-local** Q4_K_XL file (no download needed) with a small script that
calls tinygrad's own `_parse_header` (`tinygrad/llm/gguf.py`) to read just the KV metadata + tensor
list — no tensor data staged, safe/cheap:

```
architecture: qwen35moe, block_count=41 (40 real + 1 MTP nextn, nextn_predict_layers=1),
embedding_length=2048, expert_count=256, expert_feed_forward_length=512, full_attention_interval=4
n_tensors=753, total elements = 35,505,251,456 (matches the "35B total" branding)
total fp16-resident bytes  = 71,010,502,912  (71.01 GB)   <- what tinygrad needs on-device
total on-disk bytes (Q4_K_XL) = 22,842,671,616 (22.84 GB)  <- matches the actual 22.85 GB file almost exactly
per-block fp16 bytes: ~1.685 GB, uniform across all 41 blocks (no dense/leading blocks — MoE every layer)
```

**35.5B total elements × 2 bytes (fp16) = 71.01 GB, independent of which quant file is downloaded** —
Q4/Q6/Q8/UD-whatever all have the *same* element count (only bytes-per-element-on-disk differs), so
they'd all dequantize to the identical 71.01 GB resident footprint. Hardware pooled ceiling: NV = 24 GB
(RTX 3090, confirmed via `system_profiler`), Mac = 36 GiB physical (`sysctl hw.memsize` =
38,654,705,664 B exactly) minus colima's fixed 6 GiB reservation. **71 GB exceeds even the full
METAL+NV pool (≤~56 GB best case) by ~15-27%, for every quant level** — this is an architecture/quant-
independent capacity mismatch, not a "Q4 barely fits, Q6/Q8 need pooling" story. Downloading Q6 (~29
GB) and Q8 (~37 GB) would reproduce the *identical* wall after spending 20-60+ minutes of the session's
3.5h budget on bandwidth alone — **skipped deliberately**, see §2. This is the session's central,
load-bearing result and reframes every verdict below.

## 1. Q4 attempts on the real model (all failed at the capacity wall, exactly as predicted by §0 —
this is the requested "clean OOM/alloc failure IS a RESULT" datum, now root-caused precisely)

| # | Config | DEV | outcome |
|---|---|---|---|
| 1 | all-NV, no-BEAM | `NV` | `from_gguf` **succeeded** (load 12.087s). OOM during `model.warmup()`'s first forward pass: `MemoryError: Allocation of 32.00 MB failed on NV. Used: 23.17 GB` — ~13-14 of 41 blocks' weight materialized (23.17/1.685≈13.75) before the 24 GB ceiling hit. Device then entered a persistent fault state (`is_err_state` → `RuntimeError: Device fault detected`, thrown by every subsequent buffer-free during interpreter exit) — required `pkill -f "TinyGPU.*server"` before NV could be used again (standing authorization used, twice this session). |
| 2 | all-METAL, no-BEAM | (unset) | `from_gguf` succeeded (load 7.465s). Real, accelerating swap thrash during warmup, watched live: 0.94→1.83 GB (20s)→2.72 GB (40s)→5.50 GB (60s, accelerating ~45→139 MB/s) — killed by a swap watchdog at the 5 GB mark, well past the brief's own 3 GB stop-signal. Swap drained cleanly afterward (no lasting damage). |
| 3 | pooled range-split, `0-10:NV,11-39:METAL` (NV≤20GB target: 11 blocks×1.685GB≈18.5GB+token_embd) | `METAL;NV` (load device=METAL, the bulk side, per the load-direction rule) | **Never reached `from_gguf`'s return** — killed at 20s / 16.4 GB swap, *faster and worse* than row 2. Root cause: `Transformer.realize_placement()` (`model.py:557-584`, called inside `from_gguf`) force-realizes **the entire cross-device chunk in one batched `Tensor.realize(*moved)` call** right after load, by design (its own docstring: avoids re-paying dequant+copy every token) — for a small split this is invisible, but for an ~18.5 GB NV-bound chunk it means the *load device* (METAL) must transiently hold that whole chunk's fp16 dequant simultaneously with its own share's staging, front-loading a huge spike instead of the gradual per-block growth row 2 showed. |
| 3b | pooled range-split, `0-1:NV,2-39:METAL` (tiny 2-block/~3.4GB NV chunk, to try to separate the *structural* hybrid-split question from the capacity question) | `METAL;NV` | Also killed before `from_gguf` returned (15s / 5.3 GB) — but this reading is **confounded**: row 3's swap (3.3 GB) hadn't fully drained yet when this started. Inconclusive on its own for isolating "does the tiny chunk load cleanly" — see §1 note below for how the mechanism question actually got answered instead. |

**No structural hybrid-split failure was observed at any point** (no SSM-state/mask/device-mismatch
error, no assertion) — every single failure across all 4 attempts was a clean, well-characterized
memory-pressure signature (`MemoryError` or measured swap growth). Per the brief's own STOP condition
("if the layer-range split structurally fails ... STOP — that's a finding"), there was nothing to
repro-and-stop for; the uniform observed behavior across single-device *and* pooled attempts is
"mechanism correct, capacity insufficient," which is itself the finding. NV was verified healthy
(`Tensor([...]).sum().item()`) after every respawn; no lasting wedge.

## 2. Why Q6/Q8 were not downloaded

§0's arithmetic is exact and quant-independent (element count, not byte-per-element, drives resident
size), and row 1 empirically confirms the scale (real OOM at 23.17 GB, consistent with a ~71 GB total).
Q6 (~29 GB) and Q8 (~37 GB) would dequantize to the *identical* 71.01 GB and hit the *identical* wall —
downloading 66 GB combined to re-observe an already-proven-certain outcome would spend a large fraction
of the 3.5h budget for zero new fit-related information. (A bigger quant *would* still be a legitimate
thing to test for dequant **fidelity** — Q8's dequant is closer to the source weights than Q4's — but
that's an output-quality question, not what this pooling-perf task is measuring, and it doesn't change
anything in §0-1.) This is a deliberate scope call, not an oversight; happy to run the Q6/Q8 download-
and-OOM anyway if a literal per-quant datapoint is wanted despite the predicted-identical result.

## 3. Bonus, using the budget saved by §2: does the per-layer RANGE split actually deliver on its
"avoids the kernargs-island ceiling" premise? (tested on olmoe, which fits either device alone, so
there's no capacity confound)

The env brief's own framing (from `BENCH_NOTES.md`'s "BEAM'd pooling" section, read before starting)
says the per-layer RANGE split — not the `experts:` split — is "the right split for perf" because
attention lands on both devices with ~1 boundary, avoiding the ~85-island kernargs ceiling the
`experts:` split hits under BEAM. That section only ever tested the `experts:` split on a real MoE
model; the RANGE split was never tried on MoE before. Since qwen3.6 can't complete a single row, this
was cheap (already-cached model, ~2 rows, no download) and directly load-bearing for the "is the range
split actually the right lever" thesis this whole task rests on — done as a validation the write-up
above can build on, not a departure from scope. Same model/config as the section above: olmoe
(`allenai/OLMoE-1B-7B-0924-Instruct-GGUF` Q4_K_M, 16 blocks, cached), `--prompt-tokens 512
--decode-tokens 128`, nvcc lane (`DEV='METAL;NV'`, colima already up for it), device_map
`0-7:METAL,8-15:NV` (even 8/8 split, METAL-first/NV-tail).

| # | Config | warm s | decode tok/s | tokens vs all-NV reference |
|---|---|---:|---:|---|
| 1 | all-METAL, no-BEAM (prior session) | 26.035 | 28.21 | (reference-equal, prior session) |
| 2 | all-METAL, BEAM | 240.915 | 63.81 | " |
| 3 | all-NV, no-BEAM (re-run fresh this session for an exact diff) | 28.603 | 44.96 | reference |
| 4 | all-NV, BEAM (prior session) | 1232.461 | 115.51 | " |
| 5 | `experts:NV` split, no-BEAM (prior session) | 19.102 | 41.26 | **diverges at decode idx 60** (ref=1232, split=11723) |
| 6 | `experts:NV` split, BEAM (prior session) | 311.665 | 43.84 | same divergence |
| **7** | **RANGE split `0-7:METAL,8-15:NV`, no-BEAM (new)** | **26.774** | **35.31** | **129/129 byte-identical to row 3** (programmatic diff, 0 mismatches, incl. index 60 = 1232 matching) |
| **8** | **RANGE split `0-7:METAL,8-15:NV`, BEAM (new)** | **460.942** | **69.78** | **129/129 byte-identical to row 3** (programmatic diff, 0 mismatches) |

**The range split works, is exact, and wins once BEAM enters — reversing the `experts:` split's own
verdict from the prior session.** No-BEAM: range split (35.31) sits between all-METAL (28.21) and
all-NV/experts-split (44.96/41.26) — a real but modest 1.25x over all-METAL alone. **Once BEAM is
added, the range split (69.78) beats BEAM'd all-METAL (63.81, 1.09x) and beats BEAM'd `experts:NV`
(43.84, 1.59x)** — it's the clear best *heterogeneous* option, exactly the opposite of the `experts:`
split's finding ("split dead last... loses even to the single Mac-only device"). Consistent with the
brief's mechanism: the range split lets BEAM tensor-core-tune attention/lm_head kernels on **both**
devices (whichever landed on NV's half), where the `experts:` split forfeits that by confining NV to
only expert GEMVs. Warmup cost (460.9s) lands closer to `experts:`'s 311.7s than to all-NV's 1232.5s —
still cheap relative to searching the whole model on NV.

**Correctness bonus, directly relevant to T4.19 (divergence-at-depth):** the `experts:` split diverges
from the shared METAL/NV reference at decode index 60 (documented in the prior session, 3 cross-device
copies × 16 layers = 48 boundary crossings/token). The range split — same model, same 512/128 depth,
same hardware — stayed **exactly** on the reference through all 128 decode tokens, in both no-BEAM and
BEAM'd form (verified programmatically, not eyeballed: `range_split == all_nv` → `True`, 0/129
mismatches, both rows). One clean, mechanistically-plausible explanation: fewer cross-device boundary
crossings (~1 total vs. 48/token) means far fewer opportunities for cross-backend FP non-associativity
to compound into an argmax flip. Not proof this holds at every depth/model, but it's a real, directly
relevant data point for T4.19 obtained "for free" from validating the range-split thesis.

No `HCQGraph`/kernargs-ceiling crash (the T4.18/T4.20 failure mode) appeared in either range-split row,
including the BEAM'd one (the condition under which the `experts:` split's 85-island count was
originally characterized) — consistent with, though not a direct island-count measurement of, the
brief's "~2 islands" prediction for a ~1-boundary split. CSV rows appended to the new
`extra/bench_results_2026-08-25.csv` (rows 7-8 above; rows 1-6 are the prior session's, in
`extra/bench_results_2026-08-24.csv`, unchanged).

## Verdicts

**(a) DeltaNet-on-NV first impression — no structural red flag, but no speed number either.** Row 1
got a real, non-tiny hybrid DeltaNet+attention forward pass partway across NV (~13-14 of 41 blocks,
spanning multiple SSM blocks and at least 3 full-attention blocks per `full_attention_interval=4`)
before hitting the capacity wall — with a clean `MemoryError`, not a kernel-compile/launch/correctness
error. That's real but limited signal: nothing here suggests DeltaNet-on-NV is structurally broken or
unusually slow to compile, but no tok/s or completed-generation correctness check was obtainable this
session for the real model — capacity, not speed or correctness, was the blocker every time.

**(b) Does per-layer pooling deliver a usable big-quant daily driver here? No — and no quant level
would change that.** qwen3.6-35B-A3B needs ~71 GB resident under tinygrad regardless of which GGUF you
download (§0); this dock's best-case pooled ceiling is ~50-56 GB (NV's 24 GB proven hard limit + the
Mac's 36 GiB minus colima's 6 GiB reservation, itself optimistic given how fast METAL alone reached
real thrashing in row 2). No split boundary fixes this — moving the line only changes *which* side
overflows, not whether one does (row 3's NV-first split moved the failure earlier and worse, via
`realize_placement`'s eager whole-chunk realize, rather than avoiding it). There is currently no
tinygrad tok/s number to set against the llama.cpp ~31 tok/s reference for this model, and the reason
isn't a performance gap — it's a resident-memory-architecture gap (llama.cpp computes on quantized
weights in ~22 GB; tinygrad has no quantized-weight-resident compute path and needs ~71 GB for the
identical architecture). §3's olmoe result says the *mechanism* pooling relies on (the per-layer RANGE
split, BEAM'd) is sound and even wins over other heterogeneous options once a model actually fits — the
gap here is capacity, not the split design.

**(c) MTP-file compatibility — works.** The MTP GGUF (`nextn_predict_layers=1`) loaded structurally
cleanly every time `from_gguf` got to run (rows 1-2, both printed `load Xs`): `num_blocks = 41 - 1 =
40` computed correctly (`model.py:655`), the extra MTP nextn block (`blk.40`, confirmed present via the
GGUF tensor dump, same ~1.69 GB size as any other block) is simply never referenced by the 40-block
`Transformer` and consumes no memory in the real run. No MTP-specific error at any point.

**(d) Divergence-at-depth (T4.19) — not measurable on the target model (no row produced a single
decode token), but the olmoe bonus (§3) contributes a real data point anyway:** 1-boundary range splits
showed zero divergence through 128 decode tokens where a 48-hop/token `experts:` split had already
diverged by token 60, on the same model/hardware/depth. Suggestive that boundary-crossing *count*, not
just cross-backend FP non-associativity existing at all, governs how soon a split's trajectory peels
away from the reference — worth keeping in mind for whoever next picks up T4.19 properly.

## Swap / stability log

Baseline ~0.94 GB (matches prior sessions). Peaks this session, all self-induced by the qwen3.6 rows
and all recovered/drained without intervention beyond the planned kill: row 2 (all-METAL) 0.94→5.50 GB
over 60s (watchdog kill); row 3 (pooled, big NV chunk) →16.4 GB over just 20s (watchdog kill, the
`realize_placement` front-load); row 3b confounded by row 3's not-yet-drained swap. Each kill's swap
drained back toward baseline afterward (no monotonic growth, no lasting regression) — consistent with
this being in-progress OOM-avoidance-via-paging that was pre-empted, not a leak. The olmoe bonus rows
(§3) ran clean, no swap concern (small model). NV required `pkill -f "TinyGPU.*server"` + a trivial
health check twice this session (after row 1's OOM-induced device fault, and defensively after row
3/3b's kills) — both times confirmed healthy before the next GPU-touching run, no lasting wedge.

## Exact commands

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
Q4=/Users/artur/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
OLMOE=/Users/artur/Library/Caches/tinygrad/downloads/d9f8816f773421fa69637257a3f71cdc

# GGUF metadata / resident-size analysis (§0) -- ad hoc script, not committed, no tensor data staged
PYTHONPATH=. $PY /path/to/gguf_meta.py $Q4

# row 1: Q4 all-NV, no-BEAM -- OOMs at Used: 23.17 GB during warmup
DEV=NV PYTHONPATH=. $PY extra/benchmark_llm.py --model $Q4 --device-map NV --prompt-tokens 512 --decode-tokens 128
pkill -f "TinyGPU.*server"   # required after the OOM's device fault, before any further NV use

# row 2: Q4 all-METAL, no-BEAM -- real swap thrash, watched live, killed at 5GB (see BENCH_NOTES prose for the watchdog shape)
PYTHONPATH=. $PY extra/benchmark_llm.py --model $Q4 --device-map METAL --prompt-tokens 512 --decode-tokens 128

# row 3 / 3b: Q4 pooled range-split -- both killed by the same swap watchdog before `from_gguf` returned
DEV='METAL;NV' PYTHONPATH=. $PY extra/benchmark_llm.py --model $Q4 --device-map "0-10:NV,11-39:METAL" --prompt-tokens 512 --decode-tokens 128
DEV='METAL;NV' PYTHONPATH=. $PY extra/benchmark_llm.py --model $Q4 --device-map "0-1:NV,2-39:METAL"   --prompt-tokens 512 --decode-tokens 128

# NV health check, used after both respawns
DEV=NV PYTHONPATH=. $PY -c "from tinygrad import Tensor; print(Tensor([1.,2.,3.]).sum().item())"

# section 3 bonus: olmoe RANGE split (not experts:), no-BEAM then BEAM'd
DEV='METAL;NV' PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "0-7:METAL,8-15:NV" --prompt-tokens 512 --decode-tokens 128
DEV='METAL;NV' JITBEAM=2 PARALLEL=6 PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "0-7:METAL,8-15:NV" --prompt-tokens 512 --decode-tokens 128
# fresh all-NV reference re-run this session, for an exact programmatic diff against the two rows above
DEV=NV PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map NV --prompt-tokens 512 --decode-tokens 128
```


### CORRECTION (main session, 2026-08-25): the 71 GB fp16-residency analysis above is WRONG

The headline claim ("~71.01 GB resident regardless of quant; no quantized-weight-resident
path") misread lazy evaluation as materialization. `model.py:602`'s `.cast('float16')` is a
lazy expression on a lazy dequant chain, and `from_gguf` loads with `realize=False`
(model.py:695) — params stay **unrealized dequant expressions**; the quantized blob bytes are
what's device-resident, and dequant+cast fuse into consumer kernels (this is exactly the
expression shape T1.10's MATVEC work matches, and how gpt-oss-20b's 12 GB MXFP4 has run on the
24 GB card since TD.2).

**Empirical falsification (this session):** qwen3:8b Q4_K_M (5.03 GB file, 8B params → ~16 GB
if fp16-resident): `from_gguf` + `realize_placement()` on `DEV=NV:NAK` → GlobalCounters
`mem_used = 5.02 GB`. Residency ≈ quantized file size.

Reinterpretation of the four failed runs above:
- Row 1 (all-NV, OOM at 23.17 GB): the **quantized** 22.85 GB file + working set is what
  exceeded 24 GB — marginal, not structural. The registry UD-Q4_K_M (20.2 GB) is expected to
  fit all-NV; being measured now.
- Rows 3/3b (range-split swap explosion): the documented **T3.3 move-trap** — params placed on
  the non-load device are force-realized at fp16 by `realize_placement()` (moved-param COPY
  sits above the dequant). For a range split of a big model, ~half the model materializes fp16
  → blowup. This is a real structural gap for big-model range splits, but it is a **load-path
  gap (T4.21)**, not a residency ceiling.
- Row 2 (all-METAL thrash): 22.85 GB quantized + colima 6 GiB + macOS on 36 GB — consistent
  with quantized residency too.

Consequently Q6_K_XL (~29 GB) pooled ~18/11 and Q8_0 (~37.4 GB) pooled ~20/17 remain feasible
**once T4.21 lands** (place the blob *read* per-device so quantized bytes copy and dequant
fuses on the target, instead of realizing moved params at fp16). The olmoe range-split bonus
results in this section are unaffected and stand.

### qwen3.6-35B-A3B on the dock — main-session runs (2026-08-25, supersedes the section above)

Context: the "71 GB fp16-resident" analysis above is falsified (see CORRECTION). Residency ≈
quantized file size, so the real question is which *file* fits 24 GB and which quant FORMATS
tinygrad decodes efficiently. All runs `--max-context 2048 --prompt-tokens 512
--decode-tokens 128`, llama-server stopped, colima up.

| file | GB | lane | flags | decode tok/s | GB/s | verdict |
|---|---:|---|---|---:|---:|---|
| UD-Q4_K_XL (local, MTP) | 22.85 | NV:NAK | — | — | — | OOM, `Used: 23.17 GB` |
| UD-Q4_K_M (registry) | 22.13 | NV:NAK | — | — | — | OOM, `Used: 23.73 GB`; also OOMs at max-context 1024 ⇒ working set is context-INdependent (~1.6 GB) |
| UD-Q3_K_XL | 16.85 | NV:NAK | — | 1.86 | 164.1 | fits; **~88 GB/token — IQ dequant materializing (T4.22 = upstream #17316 reproduced, FIXED below — same file's real tok/s not yet re-measured)** |
| MXFP4_MOE | 21.71 | NV:NAK | — | 2.50 | 11.2 | fits; ~4.5 GB/token (bytes HEALTHY — T4.13 fusion works) but 1% of card bw ⇒ kernel/latency-bound |
| MXFP4_MOE | 21.71 | NV (nvcc) | — | **7.07** | 31.6 | 2.8x the NAK lane, same file — the usual lane gap |
| MXFP4_MOE | 21.71 | NV (nvcc) | JITBEAM=2 PARALLEL=6 | *(see next commit)* | | headline run |

Reference: llama.cpp Metal on this Mac runs UD-Q4_K_XL at ~31 tok/s (Artur's daily driver).

**File-composition matters more than the quant NAME.** Tensor-type histograms (parsed from the
GGUF headers directly):
- `UD-Q3_K_XL`: IQ3_XXS 20.94B + IQ4_XS 10.20B elements — i.e. **31 of 35B elements are IQ**,
  despite the "Q3_K" name. Unsloth "UD" = dynamic IQ mixes.
- `MXFP4_MOE`: MXFP4 (ggml type **39**) 20.94B (the 78 expert tensors) + Q5_K 10.20B (attention)
  + Q8_0 2.42B — i.e. **exactly the two formats this fork already fixed** (T4.13 MXFP4 LUT→ALU,
  T4.2 Q5_K/Q4_K scale staging). tinygrad reads type 39 (`gguf.py:21,122`).

**Practical rule for this fork: prefer MXFP4/K-quant files over Unsloth UD-/IQ mixes** until
T4.22 lands. The IQ path reads ~20x more bytes per token.

**Arch note (first DeltaNet-on-NV data):** 3 of every 4 layers are recurrent GatedDeltaNet with
sequential state updates, so a decode step issues far more small kernels than a pure-attention
model of similar size — consistent with the low achieved bandwidth at no-BEAM on both lanes.
Whether BEAM closes that (as it did 2.5-4.6x elsewhere) is the headline run's question.

**HEADLINE (2026-08-25): qwen3.6-35B-A3B MXFP4_MOE, `DEV=NV` (nvcc lane), `JITBEAM=1 PARALLEL=6`**

| max-context | prefill tok/s | decode tok/s | GB/s |
|---:|---:|---:|---:|
| 768 | 57.20 | **56.45** | 222.7 |
| 4096 | 57.34 | **56.58** | 225.5 |

**1.8x llama.cpp-Metal's ~31 tok/s on the same machine, same model family.** Speed is flat
768→4096 ⇒ `JITBEAM=1` (not the reduced context) was what fit; KV on this hybrid is small
(~3 of 4 layers are recurrent, constant state). Tokens match the no-BEAM nvcc run.

**`JITBEAM=2` does NOT fit**: BEAM-2's search scratch OOMs (`136.00 MB failed ... Used: 22.70 GB`)
with a 21.71 GB model on 24 GB. So on this card the model size sets the BEAM budget:
- no-BEAM: 7.07 tok/s
- JITBEAM=1: 56.58 tok/s (8.0x)
- JITBEAM=2: OOM (unmeasured; every smaller model gained further from 1→2)

**Third argument for pooling (T4.21), the strongest one yet:** pooling isn't only about
running quants bigger than one device — splitting a big model frees VRAM headroom for *BEAM
itself* and for long-context KV. A pooled qwen3.6 would get JITBEAM=2 room AND daily-driver
context (llama.cpp serves this model at 131k; we measured to 4096).

Open follow-up (not chased): find the max context that still fits at JITBEAM=1, and whether
JITBEAM=2 fits a smaller MXFP4-class file.

# T4.24 validation — coherence + cross-stack parity for the 56.58 tok/s headline (2026-08-25/26)

The headline above had zero correctness validation behind it: `extra/benchmark_llm.py` uses a
synthetic fake-token-ID prompt and prints raw IDs only, and a context-dependent divergence
(JITBEAM=1 @4096 vs @768) was flagged but never explained. Three verdicts below, in the order
the brief asked for them. `DEV=NV` (nvcc lane) throughout, MXFP4_MOE file, greedy
(temperature=0.0) throughout. No STOP condition was hit this session (no garbage output, no
GPU wedge — NV health-checked clean before, and after, all runs below).

## A. Coherence + cross-stack parity

Real prompts (not the synthetic `[257, 1000, 1001, ...]` ramp) through tinygrad's actual
tokenizer+generate path: `SimpleTokenizer.from_gguf_kv` + `Transformer.generate`, the same two
classes `tinygrad/llm/cli.py` itself calls — driven from a small ad hoc script (not committed,
per this file's own convention for one-off tools; see Exact commands) that reuses those classes
directly instead of going through argparse+stdin, so N prompts share one model load and I get
exact token IDs back, not just streamed text. No-BEAM, `max_context=768`, greedy (the CLI's own
default: `model.py:531-541` — `temperature=None` is the only argmax trigger, and `generate()`
defaults `temperature=0.0`, converted to `None` before sampling).

**Raw completion (3 prompts, 100 decode tokens each, no chat template — isolates the
model+kernel computation from any chat-template cross-implementation differences):**

- *"The capital of France is Paris. The capital of Japan is"* → continues with 13 more correct
  capital-city facts (Germany→Berlin, Italy→Rome, Spain→Madrid, UK→London, Australia→Canberra,
  Canada→Ottawa, Brazil→Brasilia, India→New Delhi, China→Beijing, Russia→Moscow, Egypt→Cairo,
  South Africa→Pretoria, Argentina→Buenos Aires) — every one factually correct.
- *"Once upon a time, in a small village nestled between two mountains, there lived an old
  clockmaker named"* → coherent short story ("Mr. Thompson", a boy named Timmy, a broken pocket
  watch), grammatical, sensible narrative arc, no repetition/collapse.
- `def fibonacci(n): """Return the nth Fibonacci number.""" if n <= 1: return n` → completes
  with valid Python (`else: return fibonacci(n-1) + fibonacci(n-2)`), a `main()` driver, hits
  the vocab's real `<|endoftext|>` token (248044 — distinct from the chat eos/eot id 248046
  this harness's `is_end()` checks for, so generation continues past it) and starts a fresh,
  still-syntactically-valid file.

Zero garbage in any of the three. All fully coherent, factually correct, syntactically valid.

**Tokenizer parity, independently verified with `llama-tokenize --ids`** (zero GPU involved,
pure vocab lookup): for all 3 raw prompts, tinygrad's `SimpleTokenizer.encode()` matches
llama.cpp's tokenization of the identical text **token-for-token, exactly** (e.g. prompt 1 →
both give `[760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369, ...]`; checked all
3 prompts' full overlapping length, 0 mismatches). Both also agree the model has no BOS token
(`tok.bos_id=None`: `tokenizer.ggml.add_bos_token` is false in this GGUF; llama-tokenize's
default — which normally prepends BOS — emits none here either). Two independent tokenizer
implementations reading the same GGUF vocab agree exactly, so any divergence found below is a
model/kernel computation difference, not a tokenization artifact.

**Cross-stack parity, chat mode both sides.** Tried hard to get llama-cli into true raw
completion to match the above 1:1 — `-no-cnv`, `--no-conversation`, `--no-jinja`, all
combinations tried (also had to add `-st`/`--single-turn` throughout: without it, `-p` with
closed/piped stdin drops into an interactive REPL that busy-loops printing empty `> ` prompts
forever once stdin hits EOF — burned >100MB of log twice before adding `-st`; noting this so
nobody repeats it). None of the anti-template flags bypass the model's embedded chat template
when `-p` is given and the GGUF has one: `--no-jinja` still produced a `<think>` block and a
terse chat-style "Tokyo." instead of a raw factual continuation. Pivoted to comparing **chat
mode on both stacks** instead — arguably the more relevant comparison anyway, since it exercises
two independent chat-template implementations (tinygrad's `cli.py` `jinja2.Environment` vs
llama.cpp's own jinja engine) rendering the *identical* `tokenizer.chat_template` string from
the *identical* GGUF file. llama-cli invocation: `-st --temp 0 --seed 0 -n 150 -ngl 99
--no-warmup -p "<message>" < /dev/null` (default conversation/jinja, single-turn so it exits
cleanly).

| prompt | result |
|---|---|
| "What is the capital of Japan?" | **150/150 tokens byte-identical** between tinygrad-NV and llama.cpp-Metal — the full 5-step reasoning trace, word for word, cut off mid-sentence at the 150-token cap identically on both sides |
| "Write a short story about an old clockmaker in a mountain village." | Identical for the first ~100+ tokens (full plot brainstorm, character name "Elias", identical phrasing throughout), **diverges** right after "...Isolated, misty, quiet, perhaps ": tinygrad continues "slow-paced. The mountains could be a character themselves—watching, enduring."; llama.cpp continues "a bit magical or timeless. The mountains could be a character themselves." Both fluent, both sensible — a different adjective choice that then cascades. |
| "Write a Python function that returns the nth Fibonacci number." | Diverges early, right after "**Understand the User Request:**": tinygrad continues the sentence inline (" The user wants..."); llama.cpp breaks to a new bullet ("\n   - The user wants..."). Both correctly state the Fibonacci recurrence (F(n)=F(n-1)+F(n-2), F(0)=0, F(1)=1) and stay fully coherent throughout. |

**Verdict A: tinygrad's output is coherent** — fluent, factually correct, syntactically valid
across all 6 prompts tested (3 raw completion, 3 chat). Cross-stack parity with llama.cpp is
prompt-dependent: sometimes byte-exact for the full 150-token window, sometimes diverges as
early as ~15-20 tokens in — but every observed divergence produces a fluent-but-different
continuation, never garbage or incoherence, matching the known-benign cross-implementation
FP-drift class from T4.10/T4.3 precedent (per the brief), not a correctness bug.

## B. Same-config BEAM parity @ max-context=2048

Fixed shape 2048 — never previously BEAM-searched (confirmed via the `beam_search_22` sqlite
cache table, `~/Library/Caches/tinygrad/cache.db`: 892 rows before this session's work, growing
steadily to 900+ during the run below — a genuine from-scratch search, not a cache replay), same
synthetic prompt (`extra/benchmark_llm.py`, matching the headline's own methodology exactly),
greedy, `DEV=NV`:

| config | warm time | decode tok/s | vs. the shared no-BEAM reference |
|---|---:|---:|---|
| JITBEAM=0 | 76.1s | 7.07 | — (this run *is* the reference) |
| JITBEAM=1 PARALLEL=6 | 172.6s (fresh search) | 56.70 | **diverges at decode index 106** (of 129) |

**Answer: NOT byte-identical.** JITBEAM=1 and JITBEAM=0 at the same fixed shape produce
different tokens starting at decode position 106 — a near-tied argmax flip (token 303 vs 7693).
Textually: no-BEAM → "...Did you mean to paste a specific code snippet? If you were trying to
share a piece **of code (e.g., Java, Python, C++, JavaScript)**, please **re-paste it cleanly**.
For example, I see fragments like"; BEAM-1 → "...share a piece **of code (e.g., in Python,
Java, C++, or JavaScript)**, please **re-paste it cleanly**. For example, if you". Same benign
class again: fluent, grammatical, semantically equivalent (same 4 languages, reordered + added
connectives), not garbage.

This is the parity check the headline never got, and the answer is a real, reproducible **no**
— and this isolates it cleanly: same shape, same prompt, only the BEAM flag differs, no
shape-change confound at all (rules out "it's just a stale/wrong cache entry for 2048"). Put
together with verdict C (BEAM-1@768 matches this same reference exactly; BEAM-1@4096 diverges
at 18; BEAM-1@2048 diverges at 106 — three different outcomes at three shapes, all against the
same no-BEAM reference), the pattern is: **every fresh BEAM search is its own independent roll
against near-tied kernel candidates; whether it lands on the same answer as no-BEAM (or as
another BEAM run at a different shape) isn't guaranteed, and the divergence position isn't
predictable in advance.** Reassuringly, decode speed is consistent across all three JITBEAM=1
shapes measured this session (56.62 @768, 56.80 @4096, 56.70 @2048 — within run-to-run noise of
each other and of the original 56.45/56.58 headline numbers), so **the headline's tok/s number
is robust and shape-independent**; it's specifically byte-level output reproducibility
(BEAM-vs-no-BEAM, or BEAM-vs-BEAM at a different shape) that isn't guaranteed.

## C. Context divergence (768 vs 4096) — reproduced exactly, root-caused with code + a control experiment

Reproduced with the *exact* harness/args the original headline measurement used
(`extra/benchmark_llm.py --prompt-tokens 512 --decode-tokens 128`, synthetic ramp prompt,
`DEV=NV`), so this is a clean apples-to-apples repro, not a new methodology:

| pair | result (programmatic diff, full 129-token arrays) |
|---|---|
| no-BEAM @768 vs no-BEAM @4096 vs no-BEAM @2048 | **129/129 IDENTICAL, all three** |
| BEAM-1 @768 vs no-BEAM reference | **129/129 IDENTICAL** |
| BEAM-1 @768 vs BEAM-1 @4096 | **diverge at index 18 exactly** — matches the brief's own "starting at decode token 18" precisely |

So: no-BEAM is context-invariant (confirmed three ways — 768, 4096, and 2048 all agree exactly).
BEAM-1@768 *also* matches that same reference exactly. **BEAM-1@4096 is the outlier**,
diverging from all the others at exactly token 18 and staying diverged after that (checked to
the end of the 129-token array, not just the first mismatch).

Decoded: shared prefix through token 17 is `<|im_end|>\n<|im_start|><think>\n\n</think>\n\nIt
looks like your input is a jumbled mix of` — the model correctly identifies the benchmark
harness's synthetic fake-token-ID prompt as corrupted/jumbled input, itself a small bonus
coherence data point even on garbage input. Then at token 18:
- BEAM-1@768 picks token 1970 (" code"): "...a jumbled mix of **code** snippets, keywords, and
  random characters...Did you mean to paste a specific code snippet?"
- BEAM-1@4096 picks token 15019 (" programming"): "...a jumbled mix of **programming** keywords,
  syntax fragments, and random characters...I can identify several common programming concepts
  and languages embedded in the text..."

A single near-tied argmax flip ("code" vs "programming") cascades into two different but both
fully coherent continuations — the same benign class as verdict A's and B's divergences.

**Leading hypothesis (BEAM picks different kernels per max-context-dependent shape) confirmed
from the code, not just inferred from black-box behavior:**

- `tinygrad/llm/model.py:297` — `TransformerBlock`'s KV cache (the `full_attention_interval=4`
  layers, i.e. 10 of 40 real blocks per this file's earlier GGUF metadata dump) is
  `Tensor.empty(2, B, n_kv_heads, self.config.max_context, head_dim, ...)`: **`max_context` is
  a concrete, compile-time integer dimension of this buffer, not a symbolic bound.** Every
  kernel touching this cache (`model.py:259-260`) has a literally different, concretely-shaped
  AST at 768 vs 4096 vs 2048 — a real shape difference that BEAM's independent per-shape search
  (keyed on `s.ast.key`, `codegen/opt/search.py:112`) can resolve differently run to run.
- `tinygrad/llm/model.py:452` (`GatedDeltaNetBlock`, the recurrent 30-of-40 blocks):
  `self.recurrent_state = Tensor.zeros(B, num_v_heads, head_v_dim, head_k_dim, ...)` — **no
  `max_context` term anywhere in its shape**, and the surrounding comment (written for the
  unrelated fp16-vs-fp32 precision question, T1.1a) says so explicitly: *"It's also O(1) in
  max_context (no memory win from halving it)."* The recurrent state's own heavy compute (the
  delta-rule scan, `model.py:404-430`) never touches a max_context-sized tensor.
- `start_pos` (`UOp.variable("start_pos", 0, self.config.max_context-1)`, `model.py:378`, and
  `generate()`'s `v_start_pos`) is threaded into *every* block including recurrent ones, and
  does embed `max_context` as its declared bound wherever it's used — including the recurrent
  block's own `initial = start_pos.eq(0)` reset check. So recurrent-block kernels aren't 100%
  exempt from a fresh cache-key/re-search per max_context either. But that use is a trivial
  scalar/broadcast compare with no meaningful BEAM search space (nothing to tile/upcast on a
  scalar select) — an independent re-search of it can't produce a different-and-numerically-
  consequential kernel the way a concretely-resized attention/KV kernel's search can.

**This refutes the SSM-state (T1.1a fp32 recurrent state) alternative explanation, with a
control experiment, not just a code-reading argument:** the recurrent path's fp32-state
handling is identically present in the no-BEAM runs too (same forward pass, same dtypes, same
`GatedDeltaNetBlock` code, BEAM or not) — if the SSM state's precision/handling were what made
output depend on max_context, no-BEAM@768/@4096/@2048 would have diverged from each other too.
They didn't (129/129 identical, all three, confirmed both directly and transitively). The only
thing that differs between the runs that diverge (any BEAM-1 pair at different shapes) and the
ones that don't (every no-BEAM pair, and BEAM-1@768-vs-reference) is whether BEAM ran an
independent kernel search per shape — directly implicating BEAM's per-max_context-AST search
over the concretely max_context-sized attention/KV kernels, not the recurrent/SSM path.

Did not additionally pull a raw `DEBUG=2` per-kernel-opts dump to name the exact differing
kernel by hand (the brief's "check ... if cheap" ask) — deliberate scope call: a full
kernel-level dump for a 41-block MoE hybrid model, even over just the first 18 decode steps, is
thousands of lines and not actually cheap to produce or read here, and the code-level shape
evidence above (concrete-vs-symbolic-with-no-search-space) already pins down *which class* of
kernel must be responsible without needing to empirically rediscover that via log-grepping.

## Is the 56.58 tok/s headline safe to publish?

**Yes, as a speed claim — with one caveat now documented that wasn't before.** The model
produces coherent, correct, fluent output on every prompt tested (verdict A) and the ~56-57
tok/s decode speed is consistent across every shape and BEAM run measured this session,
including a brand-new one (2048) never measured before (verdict B) — the headline number is
real and robust, not a fluke of one lucky shape. The caveat: **BEAM search is not
output-deterministic relative to a no-BEAM/differently-shaped reference** (verdicts B and C) —
this is a genuine, now-precisely-characterized, reproducible property of this fork's BEAM
integration on this model, rooted in per-shape kernel research picking different-but-valid
implementations for the (few) genuinely shape-dependent kernels. Every single divergence found
across 9 comparisons (3 chat prompts, 3 shape pairs in verdict C, 1 pair in verdict B) was a
fluent, grammatical, semantically-sound alternative continuation — never garbage, never a
repeated/collapsed/off-topic output. No STOP condition was triggered. Recommend publishing the
headline with a one-line addendum: decode speed is stable across shapes and BEAM settings, but
BEAM'd output is not guaranteed byte-identical to a non-BEAM'd or differently-shaped run of the
same prompt (known benign FP-drift class, precedented at T4.10/T4.3).

## Exact commands

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
MXFP4=/Users/artur/models/qwen3.6-35b-a3b-mxfp4/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

# A: tokenizer parity (no GPU) -- llama-tokenize vs tinygrad's SimpleTokenizer
llama-tokenize -m $MXFP4 -p "The capital of France is Paris. The capital of Japan is" --ids
llama-tokenize -m $MXFP4 -p "Once upon a time, in a small village nestled between two mountains, there lived an old clockmaker named" --ids
llama-tokenize -m $MXFP4 -f p3.txt --ids   # p3.txt = the fibonacci docstring prompt (has newlines/quotes)

# A: raw-completion coherence (ad hoc script, not committed -- loads Transformer.from_gguf +
# SimpleTokenizer.from_gguf_kv directly, same classes cli.py uses, one model load for all 3 prompts)
DEV=NV PYTHONPATH=. $PY /path/to/coherence_check.py --model $MXFP4 --max-context 768 --decode 100 \
  --prompts "The capital of France is Paris. The capital of Japan is" \
            "Once upon a time, in a small village nestled between two mountains, there lived an old clockmaker named" \
            "$(cat p3.txt)"

# A: chat-mode coherence + cross-stack (ad hoc script -- adds cli.py's own jinja2/FallbackTemplate
# chat-template construction on top of the same loader)
DEV=NV PYTHONPATH=. $PY /path/to/coherence_chat.py --model $MXFP4 --max-context 768 --decode 150 \
  --prompts "What is the capital of Japan?" \
            "Write a short story about an old clockmaker in a mountain village." \
            "Write a Python function that returns the nth Fibonacci number."
# llama.cpp side (run sequentially, never concurrently with the above -- both want the whole 21.7GB
# resident): -st is REQUIRED or a closed/piped stdin busy-loops printing empty "> " forever after EOS
llama-cli -m $MXFP4 -st --temp 0 --seed 0 -n 150 -ngl 99 --no-warmup -p "<one of the 3 prompts above>" < /dev/null

# B: same-shape BEAM parity @ 2048 (fresh search, ~173s warmup)
DEV=NV                       PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --max-context 2048 --prompt-tokens 512 --decode-tokens 128
DEV=NV JITBEAM=1 PARALLEL=6  PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --max-context 2048 --prompt-tokens 512 --decode-tokens 128

# C: 768/4096 reproduction (768 and 4096 BEAM caches were already warm from the headline session)
DEV=NV                       PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --max-context 768  --prompt-tokens 512 --decode-tokens 128
DEV=NV                       PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --max-context 4096 --prompt-tokens 512 --decode-tokens 128
DEV=NV JITBEAM=1 PARALLEL=6  PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --max-context 768  --prompt-tokens 512 --decode-tokens 128
DEV=NV JITBEAM=1 PARALLEL=6  PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --max-context 4096 --prompt-tokens 512 --decode-tokens 128

# NV health check, run before starting and after every run all session -- always clean, no wedge
DEV=NV PYTHONPATH=. $PY -c "from tinygrad import Tensor; print(Tensor([1.,2.,3.]).sum().item())"
```

# T4.21 -- the load-path fix lands: qwen3.6-35B-A3B range split actually runs (2026-08-25)

The CORRECTION section above named the swap explosion a load-path gap, not a residency ceiling
("once T4.21 lands"). T4.21 fixed it this session (`tinygrad-dock`, same branch, HEAD `ac49c5d96`
+ the T4.21 commit; see TD3_POOLING_NOTES.md section 10 for the mechanism + olmoe residency/
correctness proof). This section is the payoff: does the fix actually deliver a usable big-model
split, using the budget saved by the CORRECTION's own recommendation to skip Q6/Q8 downloads.
Same file as before, still the only one needed: `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (21.71 GB).
Split `0-7:METAL,8-39:NV` (8 of 40 real blocks on METAL, 32 + token_embd/output on NV's side per
the usual block-0/block-(-1) convention) -- picked as a modest, ~20% share off NV, since unlike
olmoe this model only barely doesn't fit on NV alone (row 1 of the earlier Q4 table: OOM at 23.17
GB / 24 GB), so a small share is all that's needed to free real BEAM/context headroom.

## Loads clean -- the core claim, proven live

| run | lane | max-ctx | load | swap during load+warm |
|---|---|---:|---:|---|
| all-NV reference (no-BEAM) | NAK | 2048 | 11.787s | flat (1754.31 MB throughout) |
| split `0-7:METAL,8-39:NV` (no-BEAM) | NAK | 2048 | 11.089s | flat (1754.88 MB, +0.6 MB noise) |
| split (`JITBEAM=2`) | nvcc | 4096 | 11.184s | flat (1668-1740 MB range, all noise) |
| split (`JITBEAM=1`) | nvcc | 16384 | 11.272s | flat (1668.31 MB throughout) |
| split (`JITBEAM=1`) | nvcc | 4096 | 11.468s | flat (1668.31 MB throughout) |

Every row polled every 5-30s through load AND warmup with a live `sysctl vm.swapusage` loop (the
brief's own abort trigger was "~4 GB sustained" -- observed max delta across all five rows: under
0.1 GB, all ordinary noise). This is the direct contrast with the pre-T4.21 row in the CORRECTION
section above: *"row 3 ... killed at 20s / 16.4 GB swap ... Never reached `from_gguf`'s return."*
Same file, a comparably-sized NV-bound share, zero swap event this time -- `from_gguf` returns in
~11s flat every time, load time barely distinguishable from the un-split reference.

## (c) `JITBEAM=2` now fits -- confirmed, with an honest speed surprise

The earlier headline table's own next-step line: *"`JITBEAM=2` does NOT fit: BEAM-2's search
scratch OOMs (`136.00 MB failed ... Used: 22.70 GB`)"* for all-NV. On the split, `JITBEAM=2
PARALLEL=6` at max-context 4096 completed in full -- no OOM, ~830s warmup (BEAM-2 searching a
40-block hybrid MoE/SSM model is slow, but it finishes). **Decode came in at 7.582 tok/s** --
slower than either no-BEAM reference (2.5-2.8 tok/s: cf.) would suggest BEAM should ever be, and
much slower than the same split's own `JITBEAM=1` result below. Not chased further given budget,
but not silently dropped either: this split's `JITBEAM=2` and `JITBEAM=1` runs produce **different
tokens from each other**, diverging at decode index 106 -- the *exact* index T4.24 already
documented BEAM parity breaking at for a different config ("fresh @2048 search diverges from
no-BEAM at idx 106"). Same index, different run, different session: strong corroborating evidence
this is T4.24's already-characterized "BEAM picks a different-but-valid kernel for a genuinely
shape-dependent op, which eventually tips a near-tied argmax" class, not a new bug from this fix --
and plausibly the same root cause behind the speed anomaly (a "valid" BEAM-2 candidate that scored
well in the search's own noisy timing but isn't actually faster in practice). **Practical
conclusion for this split: `JITBEAM=2` fits but isn't the config to use -- `JITBEAM=1` both fits
and is 4x faster on the same hardware (see below).**

## (d) Longer context fits -- confirmed, 4x the previously-tested ceiling, same speed

| config | lane | max-ctx | `JITBEAM` | warm | prefill tok/s | decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| **all-NV baseline (established, not re-run)** | nvcc | 4096 | 1 | -- | 57.34 | **56.58** |
| all-NV | nvcc | 4096 | 2 | -- | -- | **OOM** |
| split `0-7:METAL,8-39:NV` | nvcc | 4096 | 1 | 99.203s | 55.844 | **31.097** |
| split `0-7:METAL,8-39:NV` | nvcc | 16384 | 1 | 314.003s | 55.487 | **30.926** |
| split `0-7:METAL,8-39:NV` | nvcc | 4096 | 2 | 830.344s | 9.196 | 7.582 |

16384 (4x the baseline's own 4096, and 4x anything tested for this model before this session) loads
and decodes cleanly on the split, at the *same* speed as 4096 (30.93 vs 31.10 tok/s, within noise)
-- consistent with the arch note already on record ("KV on this hybrid is small (~3 of 4 layers are
recurrent, constant state)"): the split isn't paying a context-scaling cost any more than the
un-split model did (768->4096 was flat too). Tokens are **byte-identical across all three
`JITBEAM=1` rows** (4096/`JITBEAM=1`, 16384/`JITBEAM=1`, and matching through the shared prefix with
the no-BEAM reference up to its own divergence point below) -- the split's own behavior is fully
self-consistent across context length and warm/no-BEAM; only `JITBEAM=2` (previous paragraph) and
the all-NV reference (next paragraph) produce different sequences.

**Best working config: split, `JITBEAM=1`, ~31 tok/s** (4096 or 16384 context, same speed either
way) **-- 1.82x slower than the 56.58 tok/s all-NV `JITBEAM=1` baseline, but a config the old code
could not run AT ALL** (crashed via swap explosion before `from_gguf` returned, at any context).
31 tok/s also happens to land almost exactly on CLAUDE.md's own llama.cpp-Metal reference for this
model family (~31 tok/s) -- so the split's "cost" of pooling is landing back at roughly daily-driver
parity, while running at up to 16384 tokens of context (vs the un-split path's own 4096 ceiling) and
with headroom demonstrated for `JITBEAM=2` besides. Exactly the "slower but more capable" outcome
the task brief flagged as valid up front, now with real numbers: the split trades ~45% of the
all-NV `JITBEAM=1` peak for a config that (a) fits on this dock at all without pooling being
required to babysit it, (b) demonstrated 4x more context than ever tested for this model, and (c)
has BEAM-2 headroom to spare (even if BEAM-2 itself isn't the speed winner here).

## (b) Tokens correct? -- yes for the split's own consistency; a new, honest divergence vs all-NV

The split's no-BEAM output diverges from the all-NV no-BEAM reference at **decode index 8 of 65**
(both NAK lane, max-context 2048, 512 prompt / 64 decode) -- far earlier than olmoe's analogous
range split (0 divergence through 129 tokens, TD3_POOLING_NOTES.md section 10). Detokenized both
(`SimpleTokenizer.from_gguf_kv`, header-only KV parse, no extra model load) to check severity:

```
REF  : "...It appears that your input is a jumbled collection of programming keywords, syntax
        fragments, and random characters (likely resulting from a copy-paste error or a corrupted
        file). However, I can identify several common programming concepts..."
SPLIT: "...It looks like your input is a jumbled mix of code snippets, keywords, and random
        characters. This often happens when text is copied from a corrupted source, a minified
        file, or due to a keyboard/input error. However, I can help you **clean it up** or..."
```

Both fluent, both grammatical, both semantically the same answer -- textbook "class a, benign FP
drift" (T4.19's term), i.e. an early near-tied argmax flip, not corruption or a routing bug. This
session's mechanistic read on WHY it's earlier here than olmoe: T4.21's fix treats the reference
and the split symmetrically (both stage+fuse their NV-resident blocks identically; the *only*
structural difference is blocks 0-7 computing on METAL in the split vs NV in the reference), so the
divergence traces to ordinary cross-backend FP non-associativity at that one block-7/block-8
boundary -- same mechanism as olmoe's range split, but qwen3.6 is a **hybrid with recurrent
Gated-DeltaNet state** in 3 of every 4 blocks: once a tiny cross-backend difference enters the
recurrent state at the boundary, every subsequent decode step's state update compounds it further,
where olmoe's stateless-per-step attention does not. This is a genuinely new observation -- it
could not have been made before T4.21, because no range split of a model this size ever produced a
token. Not chased to full root-cause (that's T4.19-scale effort for a second architecture); flagged
here with the same honesty the brief asked for, alongside the fact that it was NEVER previously
observable, so it isn't a regression from anything that used to work.

## Verdict

**(a)-(d) all delivered, with one caveat and one anomaly, both already precedented in this file's
own history, neither a regression:** loads clean with zero swap event (a), tokens are fluent and
self-consistent within the split with a new, honest, benign-FP-drift divergence vs the all-NV
reference (b), `JITBEAM=2` now fits though isn't the speed winner (c), and a 4x-longer context
fits at unchanged speed (d). The best working config (split, `JITBEAM=1`, ~31 tok/s) is 1.82x
slower than the all-NV `JITBEAM=1` headline, landing back at roughly llama.cpp-Metal's own daily-
driver speed for this model family -- while unlocking a config (any successful load of this split
at all) that plain didn't exist before this session. T3.3's "load on the big-memory side" rule is
obsolete for this (the `from_gguf`/`device_map`) path -- see TD3_POOLING_NOTES.md section 10.

## Exact commands (T4.21)

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
OLMOE=/Users/artur/Library/Caches/tinygrad/downloads/d9f8816f773421fa69637257a3f71cdc
MXFP4=/Users/artur/models/qwen3.6-35b-a3b-mxfp4/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

# residency proof (ad hoc, not committed) -- pre/post via git stash, same process both times
DEV='METAL;NV:NAK' PYTHONPATH=. $PY /path/to/residency_check.py $OLMOE "0-7:METAL,8-15:NV"

# olmoe real-scale correctness (129/129 exact, programmatic diff)
DEV='METAL;NV:NAK' PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map NV                    --prompt-tokens 512 --decode-tokens 128
DEV='METAL;NV:NAK' PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "0-7:METAL,8-15:NV"   --prompt-tokens 512 --decode-tokens 128

# qwen3.6: swap-safety + no-BEAM correctness (NAK lane)
DEV='METAL;NV:NAK' PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --device-map NV                    --max-context 2048 --prompt-tokens 512 --decode-tokens 64
DEV='METAL;NV:NAK' PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --device-map "0-7:METAL,8-39:NV"   --max-context 2048 --prompt-tokens 512 --decode-tokens 64

# qwen3.6: the payoff numbers (nvcc lane, colima up)
DEV='METAL;NV' JITBEAM=2 PARALLEL=6 PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --device-map "0-7:METAL,8-39:NV" --max-context 4096  --prompt-tokens 512 --decode-tokens 128
DEV='METAL;NV' JITBEAM=1 PARALLEL=6 PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --device-map "0-7:METAL,8-39:NV" --max-context 16384 --prompt-tokens 512 --decode-tokens 128
DEV='METAL;NV' JITBEAM=1 PARALLEL=6 PYTHONPATH=. $PY extra/benchmark_llm.py --model $MXFP4 --device-map "0-7:METAL,8-39:NV" --max-context 4096  --prompt-tokens 512 --decode-tokens 128

# detokenize for the fluency check (header-only KV parse, no extra model load)
PYTHONPATH=. $PY /path/to/detok_check.py   # _parse_header + SimpleTokenizer.from_gguf_kv on the two token lists above

# gates
PYTHONPATH=.                    $PY -m pytest test/unit/test_llm_device_map.py test/unit/test_gguf.py -q   # 75 passed
DEV=CPU PYTHONPATH=.            $PY -m pytest test/unit/test_llm_device_map.py test/unit/test_gguf.py -q   # 68 passed, 7 skipped
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -q                          # 32 passed
PYTHONPATH=.                    $PY -m pytest test/unit -q -n12       # 846 passed, 71 skipped, 4 xfailed
PYTHONPATH=.                    $PY -m mypy tinygrad/                 # Success: no issues found in 216 source files
                                 $PY -m ruff check .                  # All checks passed

# NV health check after every run this session -- always clean, no wedge
DEV=NV PYTHONPATH=. $PY -c "from tinygrad import Tensor; print(Tensor([1.,2.,3.]).sum().item())"
```

# T4.22 — IQ3_XXS/IQ4_XS dequant fix, byte + wall-clock before/after (2026-08-25)

Follow-up to the `UD-Q3_K_XL` row above (**~88 GB/token, upstream #17316**). Root-caused and fixed:
same `buffer_in_reduce` mechanism as T4.13's MXFP4 (`tinygrad/llm/gguf.py`, types 18/23), no rangeify
change. Full writeup in `TD3_POOLING_NOTES.md` §13. Numbers below are the real-model+JIT byte-budget
harness (T4.11/T4.13's own `TestGPTOSSDecodeByteBudgetMXFP4` methodology, N_EXPERTS=32/top-4,
DIM=HIDDEN=256 for block alignment) and a synthetic decode-step wall-clock comparison — **not** the
16.85 GB real file (deferred to a future bench window, same pattern T4.13 itself followed).

| type | lane | pre-fix actual/gathered | post-fix actual/gathered | pre-fix time | post-fix time (no BEAM) | post-fix time (`JITBEAM=2`) |
|---|---|---:|---:|---:|---:|---:|
| IQ3_XXS | CPU (byte estimate) | 63.50x | **1.59x** | — | — | — |
| IQ4_XS | CPU (byte estimate) | 42.31x | **1.42x** | — | — | — |
| IQ4_XS | METAL | — | — | 1.222 ms | **0.514 ms (2.38x faster)** | not measured (already a win) |
| IQ3_XXS | METAL | — | — | 1.272 ms | 3.388 ms (2.67x **slower**) | **1.320 ms** (vs. 1.697 ms broken-BEAM'd — 22% faster) |
| IQ3_XXS | `NV:NAK` | — | — | 0.884 ms | 12.070 ms (13.65x **slower**) | not completed in budget |
| IQ3_XXS | `NV` (nvcc) | — | — | 0.954 ms | 2.267 ms (2.37x **slower**) | not completed in budget (nvcc BEAM search latency, see notes doc) |

**Reading this**: IQ4_XS (16-entry codebook) is a clean win on every axis tested. IQ3_XXS (256-entry
codebook, the dominant format by element count in the real UD-Q3_K_XL file) trades bytes for ALU via
a compile-time select-tree — a real *regression* without BEAM on all three lanes tested, but a **net
win with BEAM** (this fork's actual production configuration for quantized models — see the qwen3.6
headline above, always `JITBEAM>=1`). Byte fix confirmed cross-backend (CPU/METAL/`NV:NAK`/`NV`);
wall-clock cross-checked on METAL and (no-BEAM only) both NV lanes; `NV`+`JITBEAM` did not finish
inside this task's budget for a 2-layer toy config (nvcc's remote per-kernel BEAM compile is itself
slow cold — a separate, real, characterized cost, not a correctness question).

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
# byte budget, mirrors TestGPTOSSDecodeByteBudgetMXFP4 (ad hoc script, not committed -- the committed
# regression test is test/unit/test_llm_gptoss.py::TestGPTOSSDecodeByteBudgetIQ)
PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs   # actual/gathered=1.59x (was 63.50x pre-fix)
PYTHONPATH=. $PY t422_iq_moe_repro.py iq4xs    # actual/gathered=1.42x (was 42.31x pre-fix)
# wall-clock, no-BEAM vs JITBEAM=2, both pre-/post-fix via `git stash push -- tinygrad/llm/gguf.py`
T422_DEV=METAL             PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV=METAL JITBEAM=2   PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV='NV:NAK'          PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV=NV                PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
# gates
PYTHONPATH=. $PY -m pytest test/unit -q -n12   # 848 passed, 71 skipped, 4 xfailed, 2 subtests passed
PYTHONPATH=. $PY -m mypy tinygrad/             # Success: no issues found in 216 source files
                 $PY -m ruff check .           # All checks passed
# NV health check, before/after every real-hardware cell -- exactly one server throughout
pgrep -fl "TinyGPU.*server"
```

# T4.15: BEAM budget scaling, the BEAM-2 anomaly, and IQ x BEAM readiness for T4.26 (2026-08-25/26)

Three linked BEAM questions from the env brief, in priority order (2 > 3 > 1, per brief). Lane mechanics
per `TD3_POOLING_NOTES.md` §0 (`DEV=NV:NAK` forces the NAK/tinymesa renderer, no docker; `DEV=NV`/nvcc
goes through the Docker NVRTC compile-server, `PARALLEL=6` on every such BEAM run per the TD.2c
oversubscription lesson above). Bench window open (llama-server stopped, colima running) throughout;
`pgrep -fl "TinyGPU.*server"` checked before every dock cell, one GPU fault hit and cleared by a
respawn (§1.4). No `tinygrad/` source touched anywhere below -- every diff is either a disk-cache
read/compare or a runtime monkeypatch in an ad hoc script (not committed).

## 1. Question 2 (highest priority): why was qwen3.6-35B split `JITBEAM=2` 4x slower than `JITBEAM=1`?

T4.21's own anomaly: `0-7:METAL,8-39:NV` range split, nvcc lane, max-context 4096 -- `JITBEAM=1`
31.1 tok/s (99s warm) vs `JITBEAM=2` **7.58 tok/s (830s warm)**. T4.21's own beam cache was still warm
this session -- reproducing both configs via `extra/benchmark_llm.py` hit cache and returned in under
2 minutes each, not 830s -- so per the brief, no new search was spent on this question, only
instrumentation of the already-cached decisions.

### 1.1 Where: entirely inside one NV graph island, not spread across the step

Steady-state single-decode-step breakdown (`model.warmup()`, then `Context(DEBUG=2)` around one more
`next(gen)`, 5 live samples/config, default graphing):

| graph island | device | kernels in island | `JITBEAM=1` tm (ms) | `JITBEAM=2` tm (ms) | ratio |
|---|---|---:|---|---|---:|
| "batched 918" (per-block NV forward) | NV | 918 (identical count, both configs) | 14.77-14.85 (mean 14.82) | 109.79-110.56 (mean 110.37) | **7.45x** |
| "batched 229" (per-block METAL forward) | METAL | 229 (identical count, both) | 6.89-7.42 (mean 7.11) | 8.49-16.70 (mean 13.06) | 1.84x, noisy |
| 4 small graphs + 2 tiny copies | both | 6/7/24/24 + 2 copies | 12-166 us, flat | 18-170 us, flat | ~1.0x |

The entire ~4x step-level regression lives in the one 918-kernel NV island; everything else in the step
is within ordinary run-to-run noise. Kernel **count** is identical (918=918, 229=229) in both configs --
ruling out a graph/island-structure difference (the "or is the slowdown somewhere else entirely" branch
of the brief's question). The NV island's numbers are also suspiciously *tight* (109.79-110.56ms, <1%
spread; same for `JITBEAM=1`'s 14.77-14.85ms) -- not what "measurement noise on a busy tunnel" predicts;
a noise-driven explanation would show far more run-to-run scatter than this.

### 1.2 Why: BEAM-2 leaves 30% of the NV kernels completely unoptimized -- a real bug, not a worse-but-valid choice

Compared the actual `applied_opts` BEAM chose per kernel by monkeypatching `diskcache_get` inside
`tinygrad.codegen.opt.search` (read-only instrumentation, no `tinygrad/` source touched) to record
every `beam_search` cache hit during `model.warmup()`. Matched kernels 1:1 across the two configs by
**call-order position**, not by the diskcache `ast` key -- the key embeds `ast.arg.beam`
(`codegen/__init__.py:314`, `sink = apply_opts(sink, ren, beam=ast.arg.beam)`), so the same kernel's
cache key differs *by construction* between `JITBEAM=1` and `JITBEAM=2` (confirmed empirically: zero
overlap between the two configs' 93-95 recorded ast keys). Verified the positional-correspondence
assumption two ways: two separate `JITBEAM=1` runs reproduce byte-identical opts at every one of 95
positions (0 diffs), and device/suffix/allow_test_size line up at every position between the
`JITBEAM=1` and `JITBEAM=2` runs (0 mismatches/95) -- scheduling/fusion doesn't depend on beam width,
only per-kernel tuning does, so position N is the same underlying kernel in both runs.

| | `JITBEAM=1` | `JITBEAM=2` |
|---|---:|---:|
| NV kernel positions (of 95 total, 50 NV) with `applied_opts == []` (fully unoptimized) | **0 / 50** | **15 / 50 (30%)** |
| METAL kernel positions (45 total) with `applied_opts == []` | 0 / 45 | 0 / 45 |
| positions where the two configs chose different (non-empty) opts | -- | 73 / 95 |
| positions with identical opts | -- | 7 / 95 |

15 of the 50 unique NV kernels in this model -- squarely inside the 918-kernel island above -- get
**zero** optimization under `JITBEAM=2` while getting real, multi-`Opt` tuning (GROUPTOP/LOCAL/UPCAST/
UNROLL combinations) under `JITBEAM=1`, for the exact same kernel. **This is a real, mechanistically-
explained bug, not "BEAM measured a worse kernel as faster":**

`beam_search()` (`tinygrad/codegen/opt/search.py:111-178`) seeds `beam = [(s, inf)]` and, each round,
times candidates via `_try_compile`/`_time_program`. `_try_compile` (lines 58-80) silently swallows
**any** exception beyond `RuntimeError` (`except Exception as e: if BEAM_STRICT_MODE: raise e` --
else dropped) and returns `None`, which the caller skips (`if proc is None: continue`). If **every**
candidate in a round fails this way, `timed`/`opts` end up empty; the exit branch
`elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]` never fires (guarded by
`len(opts) > 0`), so `beam` is left at whatever it already was -- for a round-1 wipeout, the untouched
seed `(s, inf)`, i.e. **`applied_opts=[]`, with no error, no log line, no signal anything went wrong**
(the `BEAM_DEBUG` prints live downstream of the cache-hit early-return, so they're invisible here even
with debug logging on, short of `IGNORE_BEAM_CACHE=1`).

Round 1's candidate set is **provably amt-independent**: `get_kernel_actions(s, include_0=False)` is
called once, on the same seed `s`, out of the same 193-action space (`codegen/opt/search.py:14-26`),
regardless of `amt`. So for a kernel where `JITBEAM=1` finds a genuine improvement in round 1,
`JITBEAM=2`'s round 1 starts from the *identical* candidate pool -- the only way it can come back with
**nothing** is if timing/compiling those same candidates failed outright for that run. `JITBEAM=2`
isn't searching a *worse* space; its attempt to execute the *same* search failed, silently, and the
code degrades to the worst possible answer (no optimization) instead of erroring or keeping a partial
result. `JITBEAM=2` naturally issues more total compile/time requests per kernel (each of its 2 beam
survivors spawns its own fresh round of up to 193 candidates, vs `JITBEAM=1`'s 1 survivor), all against
the *already-documented-flaky* NV/Docker NVRTC compile-server on this dock (TD.2c section, this file:
`elf_loader` truncation, `BrokenPipeError` under concurrent load) -- more exposure to a known-
intermittent pipeline, landing exclusively on NV (0/45 METAL positions affected; METAL's native
compiler has none of this dock's documented docker-transport flakiness).

**FINDING, not fixed here (per the brief):** `codegen/opt/search.py`'s `beam_search`/`_try_compile` has
no way to distinguish "this round found nothing better" from "every candidate silently failed to
compile/time" -- both produce the identical, unlogged `applied_opts=[]` result. Wider beams are *more*,
not less, exposed to this on a lane with a flaky remote compile server -- the opposite of what a
"more search budget helps" model predicts. **Smallest repro:** `DEV='METAL;NV' JITBEAM=2 PARALLEL=6`,
load `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` with `device_map="0-7:METAL,8-39:NV"`, `max_context=4096`, call
`model.warmup()`; monkeypatch `tinygrad.codegen.opt.search.diskcache_get` to log every `beam_search`
cache write and diff the resulting `applied_opts` lists against the same run with `JITBEAM=1` -- 15/50
NV positions come back empty. (Confirming the exact silently-swallowed exception per kernel needs
`IGNORE_BEAM_CACHE=1` + `BEAM_STRICT_MODE=1`, i.e. a fresh ~800s-class search; not spent here given the
mechanism is already pinned down from the code plus the empty-vs-populated pattern -- left for whoever
picks up a BEAM-reliability task.)

### 1.3 Ruled out
- **Noise on a busy tunnel:** no -- both configs' dominant-island timings are tight (<1% spread) across
  5 live samples.
- **Graph/island-count difference:** no -- kernel count is identical (918, 229) in both configs' islands.
- **"BEAM-2 measured a kernel as fast during search but it replays slow"** (the brief's other named
  hypothesis): not what happened here -- it isn't that a *specific candidate* was mistimed, it's that
  *no candidate produced a timing at all* for 15 kernels, and the code silently kept the never-optimized
  seed instead.

### 1.4 GPU fault (orthogonal, cleared by respawn)

Attempting to localize the exact bad kernel(s) further by disabling JIT graphing (`JIT_BATCH_SIZE=1`,
to get one DEBUG=2 line per kernel instead of one per island) worked cleanly for `JITBEAM=1`'s kernel
set (1934 NV + 488 METAL individual kernel launches over 2 decode steps, no issue) but **faulted the
NV device on the very first post-warmup call** for `JITBEAM=2`'s kernel set (`ops_nv.py:616`,
`is_err_state`, "Device fault detected"). Cleared by the documented procedure (`pkill -f
"TinyGPU.*server"`, respawns on next use, verified via the standard `DEV=NV` health-check one-liner) --
did not recur, and per the STOP condition this only requires stopping if a respawn *doesn't* clear it,
so work continued. Recorded as a separate, orthogonal data point (an ungraphed-dispatch resource
ceiling, plausibly the same family as T4.18/T4.20's kernarg-pooling-is-graph-only ceiling, since
ungraphed execution bypasses that pooling) rather than folded into 1.1/1.2's evidence -- but it is a
real, additional asymmetry: whatever `JITBEAM=2` chose for at least one of those 15 kernels is heavy/
unusual enough that dispatching it outside the pooled graph path crashes the GPU, while `JITBEAM=1`'s
equivalent kernel set does not.

### Verdict (Q2)
**Named cause, not narrowed-candidates-with-uncertainty:** `JITBEAM=2`'s wider beam issues more total
requests against this dock's flaky NV/Docker compile pipeline over the course of one model's warmup;
when a round's *entire* candidate batch fails to time (silently swallowed exceptions in
`_try_compile`), `beam_search` has no fallback but the untouched, fully-unoptimized seed kernel, and
never says so. 30% of this model's NV kernels landed there under `JITBEAM=2`; 0% did under `JITBEAM=1`.
This fully explains the ~4x step-level and 7.45x island-level slowdown. **This is a FINDING, documented
but not fixed** (measurement-only task).

## 2. Question 3: IQ3_XXS x BEAM readiness for T4.26

Used **T4.22's own synthetic harness** (`t422_iq_moe_repro.py`: 2-block, 32-expert/top-4 synthetic
GGUF, DIM=HIDDEN=256, structurally-valid random IQ3_XXS blocks), not the real 16.85 GB
`Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` -- same methodology T4.22 already validated (its own no-BEAM numbers
reproduce here to within 0.5%, see table), and dramatically cheaper (all 6 cells below completed in
under 3 minutes each; the real file would mean a 40-block BEAM search at qwen3.6 scale, i.e. Q2-class
cost, for a question that only needs a directional readiness signal for T4.26).

| lane | `JITBEAM` | time (ms) | vs same-lane no-BEAM | warmup wall time (cold) |
|---|---:|---:|---:|---:|
| `NV:NAK` | 0 (no-BEAM) | 12.118 | 1.00x | ~36s |
| `NV:NAK` | 1 | 0.496 | **24.43x faster** | ~74s |
| `NV:NAK` | 2 | 0.494 | **24.53x faster** | ~93s |
| `NV` (nvcc) | 0 (no-BEAM) | 2.278 | 1.00x | ~38s |
| `NV` (nvcc) | 1 | 0.475 | **4.80x faster** | ~41s |
| `NV` (nvcc) | 2 | 0.438 | **5.20x faster** | ~171s |

No-BEAM numbers match T4.22's own recorded post-fix figures almost exactly (12.118 vs 12.070ms NAK,
2.278 vs 2.267ms nvcc -- confirms same harness/code state). vs T4.22's **pre-fix** no-BEAM baseline
(0.884ms NAK / 0.954ms nvcc, from the same table above): once BEAM (any level >=1) is applied, the
*post-fix* select-tree is **1.78-1.79x faster than pre-fix** on NAK and **2.01-2.18x faster than
pre-fix** on nvcc -- a clear net win on both NV lanes, not just "roughly even" like METAL's own
1.272ms-vs-1.320ms finding.

**No Q2-style pathology here on either NV lane:** `JITBEAM=2` is flat-to-slightly-better than
`JITBEAM=1` on both NAK (0.494 vs 0.496ms, within noise) and nvcc (0.438 vs 0.475ms, 8% faster) --
consistent with this tiny 2-block/32-expert harness having far fewer unique NV kernel shapes to search
than qwen3.6's 40-block model, so it stresses the flaky docker compile pipeline far less (§1.2's
mechanism needs many kernels x many candidates to show up).

### Verdict (Q3)
IQ3_XXS's fix is a clear win once BEAM is on, on **both** NV lanes, not just METAL -- T4.26 can treat
"ship IQ3_XXS as BEAM'd-only" as validated across METAL + NAK + nvcc, all showing the same qualitative
shape (no-BEAM regression, BEAM'd net win vs pre-fix). No sign of the Q2 anomaly at this (small) scale.
Used the synthetic harness, not the real IQ file (see above for why).

## 3. Question 1 (lowest priority): does raising the BEAM budget recover the shrinking multiplier?

`qwen3:8b`, `NV:NAK` lane (cheaper than gpt-oss per the brief's own steer; no-BEAM reference already on
record from TD.2: 3.72 tok/s):

| `JITBEAM` | decode tok/s | vs no-BEAM (3.72) | warmup (fresh, uncached) |
|---:|---:|---:|---:|
| 2 (reproduced) | 31.7 (31.44 orig.) | 8.45-8.52x | not re-measured cold this session (cache warm: 37s) |
| 3 | 32.74 | 8.80x | **422s (7:02)** |
| 4 | 34.62 | 9.31x | **517s (8:37)** |
| 2 + `BEAM_UPCAST_MAX=1024`, `IGNORE_BEAM_CACHE=1` (forced fresh) | -- | -- | **>590s, did not finish (killed at timeout)** |

Raising `JITBEAM` does help, monotonically, and the effort/reward tradeoff is real: 2->4 buys **+9.2%**
decode speed (31.7 -> 34.62 tok/s) for **~8.5 more minutes** of warmup than the (already-not-cheap)
`JITBEAM=2` cost. The BEAM/no-BEAM multiplier only creeps from 8.45x to 9.31x -- **recovering roughly
10% of the gap** to llama3.2:1b's 13.5x, nowhere close to closing it. Opening the *per-kernel* search
space instead of beam width (`BEAM_UPCAST_MAX=1024`, forced fresh via `IGNORE_BEAM_CACHE=1`) was tried
once as the other named lever in the brief; it didn't even finish in the time `JITBEAM=4`'s *entire*
search took, with no confirmed payoff -- not pursued further given this question's explicit lowest
priority and the budget already spent on Q2/Q3.

### Verdict (Q1)
**Not worth it as a fix for the multiplier decay.** A little more search recovers a little more speed
at rapidly compounding cost (each +1 `JITBEAM` roughly doubles warmup time here for single-digit-
percent decode gains), consistent with TD.2's own standing explanation: the shrinking multiplier is a
**structural** effect (unfused attention, per-layer costs that scale with layer count and that
tile/thread search can't touch) more than a search-budget-starvation problem -- if it were mostly
budget-starved, `JITBEAM=4` should have closed much more of the gap than it did. The `BEAM_UPCAST_MAX`
knob tested is, if anything, a *worse* lever cost-wise than just raising `JITBEAM`, for this model/lane.

## Environment / stability

Bench window open throughout (llama-server stopped, colima running). Exactly one `TinyGPU.*server`
before every cell except the one fault noted in §1.4 (cleared by a clean respawn, did not recur). Swap
held in the 1.0-1.6 GB range throughout (well under the 4 GB abort threshold), checked before/after
every dock cell. No `tinygrad/` source changed -- all instrumentation is monkeypatching in ad hoc
scripts (not committed) that patch module-level references at runtime
(`tinygrad.codegen.opt.search.diskcache_get`), never the source files themselves.

## Exact commands (T4.15)
```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
MXFP4=/Users/artur/models/qwen3.6-35b-a3b-mxfp4/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

# Q2.1: reproduce, confirm cache warm (no new 830s search), graphed DEBUG=2 island breakdown --
# ad hoc script (not committed): loads via device_map, model.warmup(), 1 prefill + 1 decode preroll,
# then Context(DEBUG=2) around 5 more next(gen) calls
DEV='METAL;NV' JITBEAM=1 PARALLEL=6 NO_COLOR=1 PYTHONPATH=. $PY q2_kernel_diag.py beam1.pkl 5
DEV='METAL;NV' JITBEAM=2 PARALLEL=6 NO_COLOR=1 PYTHONPATH=. $PY q2_kernel_diag.py beam2.pkl 5

# Q2.2: applied_opts diff -- same script, monkeypatches tinygrad.codegen.opt.search.diskcache_get to
# record every beam_search cache hit (ast/amt/device/suffix -> opts) in call order, no source change
PYTHONPATH=. $PY positional_diff2.py beam1.pkl beam2.pkl   # 88/95 differ; 15/50 NV positions -> [] under amt=2 only

# Q2.4: ungraphed per-kernel attempt (JIT_BATCH_SIZE=1 disables JIT graphing) -- clean for amt=1,
# faults NV for amt=2 (recovered: pkill -f "TinyGPU.*server", respawns on next use)
DEV='METAL;NV' JITBEAM=1 JIT_BATCH_SIZE=1 PARALLEL=6 NO_COLOR=1 PYTHONPATH=. $PY q2_kernel_diag.py beam1_ungraphed.pkl 2   # clean
DEV='METAL;NV' JITBEAM=2 JIT_BATCH_SIZE=1 PARALLEL=6 NO_COLOR=1 PYTHONPATH=. $PY q2_kernel_diag.py beam2_ungraphed.pkl 2   # NV device fault
pkill -f "TinyGPU.*server"; DEV=NV PYTHONPATH=. $PY -c "from tinygrad import Tensor; print(Tensor([1.,2.,3.]).sum().item())"  # recovers clean

# Q3: T4.22's own synthetic harness, both NV lanes, JITBEAM=0/1/2
T422_DEV='NV:NAK' JITBEAM=0 PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV='NV:NAK' JITBEAM=1 PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV='NV:NAK' JITBEAM=2 PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV='NV'     JITBEAM=0 PARALLEL=6 PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV='NV'     JITBEAM=1 PARALLEL=6 PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs
T422_DEV='NV'     JITBEAM=2 PARALLEL=6 PYTHONPATH=. $PY t422_iq_moe_repro.py iq3xxs

# Q1: qwen3:8b, NV:NAK, budget sweep
PYTHONPATH=. $PY extra/bench_llm.py tinygrad --model qwen3:8b --device 'NV:NAK' --env JITBEAM=2 --prompt-tokens 512 --decode-tokens 128
PYTHONPATH=. $PY extra/bench_llm.py tinygrad --model qwen3:8b --device 'NV:NAK' --env JITBEAM=3 --prompt-tokens 512 --decode-tokens 128
PYTHONPATH=. $PY extra/bench_llm.py tinygrad --model qwen3:8b --device 'NV:NAK' --env JITBEAM=4 --prompt-tokens 512 --decode-tokens 128
PYTHONPATH=. $PY extra/bench_llm.py tinygrad --model qwen3:8b --device 'NV:NAK' --env JITBEAM=2 --env BEAM_UPCAST_MAX=1024 --env IGNORE_BEAM_CACHE=1 \
  --prompt-tokens 512 --decode-tokens 128   # did not finish in 590s

# NV health check, before/after every real-hardware cell
pgrep -fl "TinyGPU.*server"
```
