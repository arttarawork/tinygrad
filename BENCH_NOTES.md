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
