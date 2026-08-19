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

# BENCH WINDOW 2026-08-19 — waves 3-5 re-bench, T1.1b, T4.4, T4.3

Date 2026-08-19. Same hardware/protocol as above (`llama-server` stopped for the whole session,
sequential, nothing in parallel). Branch `task/bench-window-2` = `integration/wave1` (`3e0df0fc7`)
+ `task/T0.3-bench-harness` merged in (clean merge, harness untouched). `integration/wave1` now
carries waves 1-5: T1.2/T1.10 MATVEC (fp16+quant), T1.5 temp-0 RNG skip, T1.6 jit-input cache,
T1.1a fp16 KV (default), T1.8b tuned attn, T4.5 force-realize, T2.5 sync-amortize (`drain_every`
default 1), T1.9 streaming GGUF load, T4.6 KV-prealloc cap (`DEFAULT_MAX_CONTEXT=8192`), T4.2 Q4_K
2x dequant staging, T2.3 remote-tuning knobs, T1.7 PCONTIG (landed as a dead-end/no-op — see below).

## A. Integrated re-bench (headline)

Same protocol as T0.1: `qwen3:8b` Q4_K_M, METAL, `-p/-n 512/128`, 3 no-BEAM repeats + 1
`JITBEAM=2 IGNORE_BEAM_CACHE=1` run, plus one llama-bench reference re-check.

| Stack | Config | load s | prefill tok/s | decode tok/s | decode GB/s |
|---|---|---|---|---|---|
| llama.cpp (ref, 08-18) | default Metal | n/a | 357.14 ± 0.20 | 27.27 ± 0.03 | n/a |
| llama.cpp (ref, **08-19 re-check**) | default Metal | n/a | 356.97 ± 0.20 | **27.07 ± 0.03** | n/a |
| tinygrad upstream (08-18, `2cfb421a8`) | no-BEAM | 1.70–1.78 | 15.89–15.92 | 4.91–4.93 | 26.82–26.90 |
| tinygrad integration (08-18) | no-BEAM | 1.70–1.75 | 15.82 | 7.27–7.29 | 39.68–39.76 |
| tinygrad **integration-now (08-19)** | no-BEAM | 2.12–2.44 | **14.97–14.98** | **7.37–7.39** (x̄ 7.38) | **46.76–46.89** |
| tinygrad upstream (08-18) | BEAM | 1.712 | 46.65 | 12.86 | 70.00 |
| tinygrad integration (08-18) | BEAM | 1.696 | 43.47 | 14.44 | 78.50 |
| tinygrad **integration-now (08-19)** | BEAM | 2.133 | **46.01** | **14.40** | **94.09** |

- **Decode vs the 08-18 no-BEAM reference: 7.38 vs 7.28 tok/s, +1.4%** — small but real, and decode
  GB/s jumped much more (46.8 vs 39.7, +18%): consistent with T4.2's Q4_K gemv finding (414→201 µs,
  ~2x) — the *bandwidth-reported* work per decode step dropped/changed shape, but wall-clock tok/s
  barely moved because Q4_K's gemv was ALU-bound, not membw-bound, at that point (T4.2's own note:
  "now FASTER than Q4_0" but the model's overall decode step is not 100% Q4_K-gemv-dominated —
  attention + other layers still gate wall time). So waves 3-5 mostly show up as a GB/s accounting
  change plus a small tok/s gain, not the "should show strongly" 2x some hoped for from T4.2 alone.
- **BEAM: 14.40 vs 14.44 tok/s — flat**, within noise of the single 08-18 run.
- **Prefill dropped: 14.98 vs 15.82 tok/s no-BEAM (-5.3%), 46.01 vs 43.47 tok/s BEAM (+5.9%)** — see
  §C below; the no-BEAM prefill regression is new since 08-18 and is a candidate side effect of one
  of the wave3-5 levers (T4.6's `DEFAULT_MAX_CONTEXT=8192` changes KV buffer shape/indexing even
  when the prompt is short; T1.9's streaming load changes how weight tensors land in memory — not
  bisected further, out of scope for this pass, flagged for a future task).
- **llama.cpp reference re-check: 27.07 vs 27.27 tok/s (-0.7%), prefill 356.97 vs 357.14 — stable**,
  confirms the 08-18 reference point and that today's machine state is comparable.
- Both tinygrad configs still well behind llama.cpp (best result 14.40 tok/s BEAM = 53% of 27.07).

Raw rows appended to `extra/bench_results_2026-08-18.csv` (still the working file — new rows are
tagged `2026-08-19 task/bench-window-2@integration-wave1-now`).

## B. T1.1b — fp16 KV decode delta

Default (fp16 KV, T1.1a) vs `KV_F32=1` escape hatch, no-BEAM, `qwen3:8b` Q4_K_M METAL. Short (`-p
512 -n 128`, 3 repeats each — the "default fp16" repeats are the part-A no-BEAM rows above, same
config) and one long-context pair (`--max-context 8192 --prompt-tokens 4096 --decode-tokens 128`,
run directly against `extra/benchmark_llm.py` since `bench_llm.py`'s wrapper doesn't expose
`--max-context`; single run each side — the point is the qualitative KV-share-of-bytes shift, not
noise-hunting a 4.5-minute-per-run config).

| Context | KV dtype | decode tok/s | decode GB/s |
|---|---|---|---|
| 512+128 (short) | fp16 (default) | 7.35–7.39 (x̄ 7.38) | 46.76–46.89 (x̄ 46.82) |
| 512+128 (short) | f32 (`KV_F32=1`) | 7.35 (all 3) | 49.14–49.15 |
| 4096+128 (long) | fp16 (default) | **5.11** | 43.45 |
| 4096+128 (long) | f32 (`KV_F32=1`) | **5.00** | 54.65 |

- **Short context: no measurable tok/s delta** (7.38 vs 7.35, within the <0.1 tok/s repeat spread)
  — expected, KV at ≤640 tokens is a rounding error next to the ~5 GB Q4_K weight set re-read every
  decode step.
- **Long context: fp16 KV is +2.2% decode tok/s** (5.11 vs 5.00) — small in wall-clock terms, but
  the *GB/s delta is large and in the opposite direction* (43.45 vs 54.65, f32 reports **+25.8%**
  more bytes/sec for barely less wall time) — exactly what halving the KV bytes read per step
  should look like from `GlobalCounters`: f32 KV pushes more bytes through the same wall-clock
  window while the actual token-per-second win is muted because KV bytes are still a minority share
  of total per-step traffic at 4096 tokens (attention compute + the weight re-read still dominate;
  a rough back-of-envelope for qwen3:8b's GQA (n_kv_heads×head_dim) KV state at 4096 tokens ×32
  layers is on the order of 0.5–1 GB fp16-vs-f32 delta, vs. ~5 GB of weight traffic per step).
- Net: **T1.1a's fp16-KV default is a real, if modest, win that grows with context** — at 4096
  tokens it's already visible in both tok/s and GB/s; the design doc's expectation that the win
  "shows more at long context" is directionally confirmed, but doesn't dominate wall-clock time
  until KV cache size approaches weight-read size (would need a much longer context or a model
  with a much smaller weight footprint to see fp16 KV swing tok/s by more than a couple percent).
- Both long-context rows also show prefill dropping to ~10.4 tok/s from ~15.0 tok/s at 512 tokens
  (expected — larger attention cost as context grows during prefill), and are consistent with each
  other within 0.3% (10.45 vs 10.42) — KV dtype doesn't materially affect prefill, as expected
  (prefill's KV writes, not the growing-context reads that dominate decode).
