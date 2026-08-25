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
`Compiler.compile_server()` (`tinygrad/device.py`), plus a parity check. **Result: blocked on cell
1.** Cells 2-5 and the parity check were not attempted.

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

**Full table — unchanged from TD.2b, now with a root-caused (not merely "crashes") reason for the
open cells:**

| Model | Lane | Config | load s | prefill tok/s | decode tok/s | GB/s |
|---|---|---|---:|---:|---:|---:|
| llama3.2:1b | NV:NAK | no-BEAM | 5.01 | 55.3 | **7.13** | 8.4 |
| llama3.2:1b | NV:NAK | JITBEAM=2 | 4.91 | 341.0 | **96.06** | 112.5 |
| llama3.2:1b | NV | no-BEAM | 4.92 | 142.5 | **28.16** | 33.0 |
| llama3.2:1b | NV | JITBEAM=2 | — | — | **blocked (poisoned cache row, root-caused above)** | — |
| qwen3:8b | NV:NAK | no-BEAM | 6.54 | 16.5 | **3.72** | 23.6 |
| qwen3:8b | NV:NAK | JITBEAM=2 | 6.49 | 66.7 | **31.44** | 197.0 |
| qwen3:8b | NV | no-BEAM | — | — | **not attempted (blocked by cell-1 STOP)** | — |
| qwen3:8b | NV | JITBEAM=2 | — | — | **not attempted (blocked by cell-1 STOP)** | — |
| gpt-oss:20b | NV:NAK | no-BEAM | 8.61 | 78.7 | **7.42** | 25.8 |
| gpt-oss:20b | NV:NAK | JITBEAM=2 | 8.63 | 117.3 | **26.81** | 105.3 |
| gpt-oss:20b | NV | (either) | — | — | **not attempted (blocked by cell-1 STOP)** | — |

**Parity check (qwen3:8b `NV` JITBEAM=2 vs JITBEAM=0):** not performed — depends on a completed
qwen3:8b `NV` JITBEAM=2 run, which never happened this session.

**Updated takeaway on lane choice:** still moot, unchanged from TD.2b. `DEV=NV`+BEAM still cannot
complete a run on this dock, so whether it beats NAK+BEAM (96.06 / 31.44 / 26.81 tok/s decode on
llama3.2:1b / qwen3:8b / gpt-oss:20b) remains **unmeasured** — this session neither confirms nor
denies the "4x-on-no-BEAM suggests real headroom" projection from TD.2b's takeaway #2. NAK+BEAM
remains the only lane that has ever produced a complete, usable high-throughput number on this dock.
The T4.14 fix itself is vindicated (99.86% of the existing NV compile cache is valid, matching its
own repro's 10/10), so the remaining blocker is narrowly scoped (4 known rows) and well-understood,
not an open question about the fix's correctness — unlike TD.2b's finding, this is no longer "an
unfixed compiler bug," it is "a fixed bug whose blast radius includes a few stale cache entries that
outlived it."
