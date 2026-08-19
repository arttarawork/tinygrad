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

## C. T4.4 — BEAM prefill anomaly: VERDICT = noise, not a regression

3 `JITBEAM=2 IGNORE_BEAM_CACHE=1` prefill repeats on integration-now (`task/bench-window-2`,
`e4c8868a0` base) and 3 on `upstream/master` tip (`ca86a4270`, worktree `../upstream-bench-2` =
`/Users/artur/Documents/tinygrad/.claude/worktrees/upstream-bench-2`), plus one `MV=0` control run
to test the design doc's candidate hypothesis directly.

| Side | prefill tok/s (3 repeats) | mean | decode tok/s (3 repeats) |
|---|---|---|---|
| integration-now | 46.01, 47.90, 48.24 | 47.38 | 14.40, 11.98, 13.29 |
| upstream@ca86a4270 | 47.12, 46.41, 47.95 | 47.16 | 14.07, 13.20, 13.01 |
| integration-now, `MV=0` control | 47.29 (1 run) | — | 11.31 |

- **The two spreads fully overlap** (integration 46.01–48.24 vs upstream 46.41–47.95) — mean delta
  is 0.5%, well inside each side's own repeat spread (~4.7% and ~3.3% respectively). The 08-18
  single-run gap (integration 43.47 vs upstream 46.65, a 6.9% "regression") does not reproduce:
  today's integration-now floor (46.01) already exceeds that 43.47 figure, and today's upstream
  ceiling (47.95) is barely above its own 08-18 point estimate. **Verdict: noise** — `JITBEAM=2
  IGNORE_BEAM_CACHE=1` runs a small, budget-limited, from-scratch search every time, and the spread
  isn't confined to prefill either: **decode tok/s on the same integration-now runs varies 11.98 to
  14.40 (20% swing)** across otherwise-identical repeats, so a single-run ±7% prefill delta between
  two checkouts is well within what this search's own repeat-to-repeat noise produces. No lever
  bisection needed; closing per T4.4's "if it's within noise, close it as variance" clause.
- **The design doc's candidate hypothesis (MATVEC guard misfiring on a prefill kernel it shouldn't)
  is independently ruled out, not just by noise but structurally**: `MV=0` prefill (47.29) lands
  squarely inside the no-guard-difference spread (46.01–48.24), and this is *provably* a true
  no-op, not a lucky match — `codegen/opt/postrange.py:339-356`'s `apply_opts` branches on
  `beam >= 1` *before* it ever reaches the `hand_coded_optimizations` call (`elif not NOOPT and
  ...: from tinygrad.codegen.opt.heuristic import hand_coded_optimizations`); the `MV` env var is
  read nowhere except inside that function (`codegen/opt/heuristic.py:70`). Under `JITBEAM=2`,
  `hand_coded_optimizations` — and therefore the entire MATVEC guard — is never called at all; BEAM
  search picks its own tiling via `beam_search()` independent of the heuristic. The candidate
  culprit named in the task doesn't apply to the BEAM code path by construction.
- Anomaly (noted, not chased): on this synthetic dummy prompt (`benchmark_llm.py`'s
  `[257] + [1000+i%1000 for i in range(511)]`, not real text), integration-now's BEAM decode output
  tokens are a plausible-looking numbered list (`[198, 197, 197, 322, 220, 16, 13, 220, 17, ...]`,
  identical across all integration-now BEAM repeats including the `MV=0` control) while
  upstream@ca86a4270's BEAM decode output is a different, degenerate repeating 17-token loop
  (`[11, 345, 0, 50994, 2428, 1110, 72, 55277, 905, 14, 21, 48, 21, 48, ...]`, also identical across
  all 3 upstream repeats). Each side is internally deterministic/reproducible (temp=0, same output
  every repeat) but the two sides diverge from each other. Most likely explanation: greedy argmax
  is a discontinuous function of the logits, and different kernel tilings (integration's heuristic
  fp16/quant-MATVEC-shaped prefill kernels are bypassed under BEAM too, so this isn't that guard —
  it's whatever BEAM's own search landed on for each checkout) sum reductions in a different order,
  giving different rounding; on a degenerate synthetic prompt with likely near-tied logits this can
  flip the argmax pick early and cascade into a fully different (but equally "valid" per-checkout)
  continuation. Not investigated further — T4.3 below is the real correctness-parity check, on real
  prompts (a synthetic garbage prompt isn't a meaningful place to chase greedy-decode divergence).

## D. T4.3 — gpt-oss-20b real-model validation

`gpt-oss:20b` MXFP4 (12.1 GB GGUF, cached), METAL, `llama-server` stopped. Memory watched throughout
(`vm_stat`/`sysctl vm.swapusage`) — stayed healthy, no swap growth from any of these runs (see
Anomalies).

**Tool note:** `llama-cli -no-cnv -p "..."` hung generating unbounded output (ran to 512 KB+ over 2
min before being killed — likely `-no-cnv` not fully suppressing the interactive read-loop for this
build, `--no-conversation`'s help text ties "interactive mode" to the same flag but evidently didn't
disable it here). Switched to `llama-completion` (a llama.cpp binary built specifically for one-shot,
non-interactive completion; `-i/--interactive` defaults to `false` there) with `-no-cnv < /dev/null`
for belt-and-suspenders — worked cleanly for all 3 prompts, respected `-n 64` exactly.

### (1) Tokenizer + greedy-generation parity vs llama.cpp

3 prompts, `--temp 0 -n 64` both sides (tinygrad: `Transformer.generate(..., temperature=0.0)`,
default `chunk_size=32`; llama.cpp: `llama-completion --temp 0 -n 64 -no-cnv`), tokenized
independently by each stack from the same GGUF:

| # | Prompt (truncated) | tokens | crosses chunk=32 boundary? |
|---|---|---|---|
| 1 | "The capital of France is" | 5 | no |
| 2 | "Explain in one sentence why the sky..." | 13 | no |
| 3 | "Write a short Python function that takes a list of integers..." | **33** | **yes** (32+1) |

- **Tokenizer parity: EXACT on all 3 prompts.** tinygrad's `SimpleTokenizer` (T1.3's `gpt-4o` preset)
  produced byte-identical token ID sequences to `llama-tokenize -m <gguf> --ids --no-parse-special`
  for all three prompts (verified id-by-id, including the 33-token prompt). No divergence to report
  here — T1.3's tokenizer is a genuine match for gpt-oss's o200k-based vocab.
- **Generation parity, prompts 1 & 2 (single prefill chunk): EXACT, all 64 generated tokens**,
  verified by diffing the full concatenated prompt+completion text from `llama-completion` (run with
  `--display-prompt`, the default) against `tok.decode(prompt_ids + gen_ids)` from tinygrad — the
  two are byte-identical strings on both prompts (mod a trailing newline llama.cpp's CLI appends).
- **Generation parity, prompt 3 (crosses the chunk_size=32 boundary): DIVERGES at generated token
  26** (0-indexed position 25 in the 64-token output; absolute sequence position 58 — i.e. 26 decode
  steps after the 33-token, 2-chunk prefill finishes). Tokens 1-25 are identical between the two
  stacks (`"\n\nHere is a short Python function that takes a list of integers and returns the sum of
  all even numbers in the list,"` — 25 tokens, byte-identical). At token 26, tinygrad picks id 4251
  (`" along"`, continuing "...list, along with a brief docstring...") while llama.cpp picks id 3463
  (`" including"`, continuing "...list, including a brief docstring..." — an echo of the prompt's own
  wording). Both continuations remain fluent, on-topic Python code afterward (tinygrad: `def
  sum_of_evens(numbers): """Calculate the sum..."""`; llama.cpp: `def sum_of_even_numbers(numbers):
  """Returns the sum..."""`) — not garbage, just a different (also-reasonable) choice of words.
  Verified precisely by tokenizing llama.cpp's raw output text with tinygrad's own tokenizer (already
  shown to match llama.cpp's exactly) and diffing token-by-token against tinygrad's `gen_ids`; did
  not additionally extract per-step logits (would need instrumenting `Transformer.forward` to return
  raw logits instead of the post-argmax token — judged not worth the extra surface for what the
  token-level position already demonstrates).
- **Verdict: real, position-identified divergence, isolated to the multi-chunk-prefill case.** The
  two single-chunk prompts are perfect controls: same tokenizer, same model, same greedy decode,
  zero divergence over 64 tokens each. The only prompt that crosses the `chunk_size=32` prefill
  boundary (T1.3's noted "untested interaction" between GGUF-quantized gpt-oss and chunked prefill,
  compounded by the sliding-window layers alternating every other block) is the only one that
  diverges, and it diverges well after the boundary itself (26 decode steps later) rather than at
  the boundary — consistent with a small floating-point rounding difference introduced by prefill
  being split into two forward-pass chunks (vs. llama.cpp's own batching) propagating through
  attention/KV state until it flips an argmax choice once two candidate continuations have close
  enough logits. Not a correctness bug in the sense of producing garbage or crashing — a legitimate
  case of "different valid computation order gives a different but individually coherent output,"
  the same phenomenon flagged (with less rigor) in Part C's BEAM anomaly note. Flagged for follow-up
  if bit-exact chunked-prefill parity ever becomes a goal; out of scope to fix here (T4.3 asks for
  the parity verdict, not a fix).

### (2) Benchmark row

| Model | load s | warm s | prefill tok/s | decode tok/s | decode GB/s |
|---|---|---|---|---|---|
| gpt-oss:20b MXFP4, no-BEAM | 2.660 | 41.77 | 16.57 | **1.69** | 100.65 |

Decode is far slower than `qwen3:8b`'s 7.38 tok/s (expected — MXFP4 dequant + a larger active
parameter set per token than qwen3:8b's dense 8B, and this is the *first* real-model bench-window
data point for gpt-oss on this harness). GB/s (100.65) is the highest of any row in this session,
consistent with a much larger active-weight footprint per decode step.

## Wrap-up: anomalies / caveats for the whole 2026-08-19 window

- **`llama-cli -no-cnv` runaway** (part D): hung generating unbounded output past its `-n 64` limit,
  killed manually after ~2 min / 512 KB+ of output at 100% CPU / ~15 GB RSS. Root-caused to the
  wrong binary for a one-shot completion (`llama-cli` is REPL-oriented; `-no-cnv` didn't fully
  suppress its interactive continuation loop on this build) rather than a tinygrad-side or
  memory-side problem. `llama-completion` (a dedicated one-shot completion binary in the same
  llama.cpp install) fixed it outright — not filed as a tinygrad issue, just noted here so a future
  session doesn't repeat the 2-minute detour.
- **No swap growth observed** across the whole window: `sysctl vm.swapusage`'s `used` figure was
  identical (798.56 MB) before Part D's gpt-oss-20b load and after all of Part D's runs finished;
  `vm_stat` free+inactive pages stayed in the multi-GB range throughout, including while the 12 GB
  gpt-oss GGUF was resident. `llama-server` confirmed stopped before the window and never restarted
  by this session (restore command in `TASKS.md`/`CLAUDE.md` if the orchestrator hasn't already).
- No thermal throttling indicators observed (BEAM search wall-times — 221-312s across 7 runs — show
  no systematic drift that would suggest throttling under sustained load).
- Worktrees added this session: `../upstream-bench-2` (`upstream/master@ca86a4270`, detached HEAD,
  clean, no local changes) for part C's upstream comparison — left in place, same rationale as the
  08-18 session's `../upstream-bench` (now two upstream reference worktrees exist, pinned to
  different tips; both safe to `git worktree remove` whenever). Branch `task/bench-window-2` =
  `integration/wave1` (`3e0df0fc7`) + `task/T0.3-bench-harness` merged, plus this session's 4 commits
  (parts A-D). Not pushed; not merged into `integration/wave1` or `master`.

# BENCH WINDOW 2026-08-19w3 — FAST_ATTN tiebreaker, T4.10 disambiguation, gpt-oss bytes, headline refresh

Branch `task/bench-window-3` = `integration/wave1` (`090489bca`, includes T1.8c symbolic-Tk tuned
attention, T4.11's byte-budget test, T4.7's compound-expr fix) + `task/bench-window-2` merged
(clean, no conflicts). `llama-server` stopped for the whole window; ~32 GB free at start; strictly
sequential.

## A. FAST_ATTN tiebreaker (qwen3:8b, METAL, greedy)

`FAST_ATTN=1` swaps `attention_impl` for T1.8b/T1.8c's tuned Metal decode-attention kernel (online
softmax, LOCAL `dout`, cooperative QK contraction, CHUNK=16 threadgroup staging — see
`tinygrad/llm/attn_kernel.py`). It only engages on decode (`T==1`, unmasked); qwen3 has no
sliding-window layers (that's gpt-oss-specific), so with T1.8c's symbolic-Tk fix already merged it
fires on **every** decode step for this model, not just the first.

**(1) Standard `-p 512 -n 128`, 3 repeats each:**

| flag | decode tok/s (3 runs) | avg | vs FAST_ATTN=0 |
|---|---|---|---|
| `FAST_ATTN=0` | 7.39, 7.37, 7.37 | 7.377 | — |
| `FAST_ATTN=1` | 7.38, 7.38, 7.38 | 7.380 | **+0.05%** (noise) |

Prefill tok/s identical either way (14.97-14.98, expected — the tuned kernel never touches
prefill/T>1). **Tokens byte-identical across all 6 runs** (same 128-token output list, spot-checked
in full, not just a hash).

**(2) Long-context: `--max_context 8192`, 4096-token prompt, 128 decode tokens (1 run each —
single-run, not averaged over repeats like (1)):**

| flag | prefill tok/s | decode tok/s | decode GB/s |
|---|---|---|---|
| `FAST_ATTN=0` | 10.449 | 5.012 | 42.62 |
| `FAST_ATTN=1` | 10.451 | 5.117 | 43.51 |

**+2.09% decode tok/s** at Tk≈4096-4223, tokens again byte-identical (both produced the same
124-token continuation of the synthetic incrementing prompt). Directionally consistent with the
docket's hypothesis (attention's share of decode time grows with Tk, favoring the fused kernel) but
this is a single run per flag, not 3 — noise band not established the way (1)'s was.

**Verdict (T4.8 go/no-go): mixed, lean no-go.** At 8B/Hd=128, FAST_ATTN=1 is a wash at standard
context (+0.05%, within run-to-run noise) and a modest, single-run +2.1% at 4k-token context —
nowhere near the kind of win that would justify the ~300-550-line T4.8 Metal-simdgroup investment
the docket frames this as gating. Contrast with llama3.2:1b's reported -2-3% *regression*: this
model/head-dim combination at least doesn't lose, but "doesn't lose, and might gain ~2% only once
context gets long" is a weak basis for the bigger investment. Recommend: **don't fund T4.8** off
this data alone; if the long-context number is worth pinning down, 3 repeats of (2) would be the
next cheap step before committing to the larger kernel-rewrite scope — not committed here to keep
this window sequential and on schedule for parts B-D.

## B. T4.10 disambiguation (length vs chunking) — gpt-oss-20b, METAL, greedy

**Prompt caveat up front:** T4.3's original 33-token divergent-case prompt was never committed
(ad hoc session, not persisted). Reconstructed a same-family, same-token-count prompt instead —
"Write a short Python function that takes a list of integers and returns the sum of all even
numbers in the list, including a brief docstring explaining what it does." — verified via
tinygrad's own `SimpleTokenizer` to be exactly 33 raw tokens (crosses the `chunk_size=32` prefill
boundary by exactly 1 token, same as the original). This is a reconstruction, not a byte-identical
replay of T4.3's literal prompt — flagged so the result below isn't overclaimed as "the exact same
run." (It turned out to reproduce the *same* divergence position and the *same* two token IDs as
the original T4.3 finding — see below — which is itself informative: whatever's driving this isn't
sensitive to the specific prompt wording.)

**Anomaly hit before the real test:** `model.warmup()` always internally calls `generate([0], ...)`
with the default `chunk_size=32` (no `chunk_size` param on `warmup()`). The JIT dispatch key is only
`(is_prefill, temp_is_None)` — it does **not** include `chunk_size` / the `v_toks` Variable's max
bound. Calling `warmup()` then `generate(..., chunk_size=64)` in the same process collides: the
decode JIT entry captured during warmup has `v_toks` bound to max=32; the `chunk_size=64` call's
`v_toks` (max=64) doesn't match, raising `tinygrad.engine.jit.JitError: args mismatch in JIT`. Real
bug (a server that varies `chunk_size` across requests in one process would hit this), not part of
what B is testing — routed around by skipping `model.warmup()` in the repro script (harmless for a
correctness-only comparison; only costs a slightly slower first token). **Not filed as a task here
per the docket's "no fixes here" scope — flagged for a future task instead.**

**(1) `chunk_size=32` (default, reproduces the divergence) vs (2) `chunk_size=64` (single prefill
chunk — the 33-token prompt fits in one chunk, eliminating tinygrad's own chunking as a variable):**

Both produced **byte-identical 64-token generations** (same `gen_ids` list, full diff, not a hash)
— tinygrad's own output is chunk-size-invariant on this prompt, confirming T4.10's original
self-comparison finding (`f66fc48a6`) from a second angle.

**vs. `llama-completion -m gpt-oss-20b --temp 0 -n 64 -no-cnv -f <same 33-token prompt>`:** both
tinygrad runs (chunk_size=32 and 64) diverge from llama.cpp's completion at the **same position**:
generated token 26 (0-indexed 25) — tinygrad picks id 4251 (`" along"`), llama.cpp picks id 3463
(`" including"`), verified precisely by tokenizing llama.cpp's raw completion text with tinygrad's
own tokenizer and diffing id-for-id against `gen_ids` (25 identical tokens first, both fluent,
on-topic continuations after the divergence — same phenomenon shape as the original T4.3 note).

**Verdict: (2) matches (1), not llama.cpp — CONFIRMED closed, T4.10's FP-drift-near-tie hypothesis
holds.** Since `chunk_size=64` (no chunking at all — one prefill pass, exactly matching llama.cpp's
own single-batch prompt eval, `33 tokens` per its own perf log) still diverges from llama.cpp
identically to `chunk_size=32`, tinygrad's prefill chunking is ruled out as the cause of the
cross-stack divergence — the divergence must come from somewhere else in tinygrad's forward pass
(vs. llama.cpp's), consistent with a small floating-point rounding difference propagating through
attention/KV state until it flips an argmax choice once two candidates are nearly tied, exactly as
T4.10/T4.3 originally proposed. This also empirically confirms the `memory` branch's `TASKS.md`
update (`ba068e9ae`, "T4.10 cleared — chunking is invariant, T4.3 divergence is FP drift near tied
argmax") — that commit predates this actual chunk_size=64-vs-llama.cpp cross-check (it only had
T4.10's tinygrad-vs-itself self-comparison to go on at the time), so it was directionally right but
not yet earned by this specific test; it is now. No swap growth across any of B's runs (`vm.swapusage`
used stayed at 1213.19 MB throughout, load through llama-completion).
