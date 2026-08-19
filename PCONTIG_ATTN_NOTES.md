# T1.7 — PCONTIG online-softmax fusion for decode attention: analysis

Branch: `task/T1.7-pcontig-attn` off `integration/wave1`. No production code changed (see
"Fix vs document" below) — this is the STOP-condition exit: 3 distinct structural blockers found,
one of them severe enough (silent wrong numerics) that no amount of small guard-fixing is
appropriate here.

**Bottom line up front: dead end.** `PCONTIG`'s fusion mechanism does not just fail to help decode
attention — it produces **numerically wrong output** on the exact shapes tinygrad's own test suite
claims to validate, the existing tests that were supposed to catch this are silently defeated by an
unrelated caching bug (`SCACHE`), and the one thing that's supposed to be the whole point of T1.7
(surviving a *symbolic* Tk after the first decode token) crashes outright at Metal shader-compile
time for a structural reason (threadgroup memory sized by a Variable's static upper bound). Do not
wire this into `tinygrad/llm`. See Recommendation.

## 1. State map — where the code lives today (re-located from the `af2a43c85` baseline)

The old baseline had this around `rangeify.py:264-282`; it's still there, lightly shifted:

| Mechanism | File:line | Gate | What it does |
|---|---|---|---|
| Skip bufferize-removal cutoff | `tinygrad/schedule/rangeify.py:256` | `not (PCONTIG > 2)` | Normally, a bufferize touching >3 distinct buffers is kept (not fused). `PCONTIG>2` lifts this cap. |
| "Online softmax" local-bufferize | `tinygrad/schedule/rangeify.py:267-282` | `PCONTIG > 2` | When a reduce directly touches a raw buffer (`buffer_in_reduce`), and the output/input size ratio looks favorable (`out_in_ratio < 10`), partitions ranges into a `AddrSpace.LOCAL` (Metal `threadgroup`) bufferize + index — this is the actual "keep a partial result on-chip and re-read it" flash-attention-style move. |
| Consumer-range merging | `tinygrad/schedule/indexing.py:259` | `PCONTIG` (truthy, i.e. ≥1) | Lets ranges with the same underlying index but different valids merge instead of forcing a new realize. |
| Ending-ranges realize suppression | `tinygrad/schedule/indexing.py:276` | `PCONTIG > 1` (i.e. ≥2) | Normally, a reduce/elementwise op whose input ranges "ended" earlier than its own forces a new realize boundary; this gate skips that when `PCONTIG>1`. |
| `PCONTIG` ContextVar | `tinygrad/helpers.py:270` | default `0` | Off unless explicitly set. Not touched by this branch. |

**Naming trap for future readers:** every doc (`CLAUDE.md`, `NV_LLM_DESIGN.md`, `TASKS.md`) calls
the flag "`PCONTIG>2`" and treats `PCONTIG=2` as "the flash-attention path is on." The code's actual
`> 2` comparisons require `PCONTIG=3` to reach `rangeify.py`'s local-bufferize branch. `PCONTIG=2`
only activates `indexing.py`'s two milder mechanisms — and, confusingly, that alone is *already
enough* to fully fuse simple ≤3-buffer attention/matmul-chain expressions into 1 kernel (because
`rangeify.py:256`'s buffer-count cap never blocks a ≤3-buffer graph in the first place, regardless
of `PCONTIG`). I did not fully bisect which of the three mechanisms produces which specific wrong
answer at each level — see §6.

Existing test coverage, also re-located (structure unchanged from what the baseline referenced):
- `test/backend/test_rangeify.py`: `TestDoubleMatmul` (22 tests, `a@b@c` fusion, `PCONTIG=2`),
  `TestPcontig.{test_flash_attention, test_flash_attention_bw, test_flash_attention_opt}` (the
  named "flash attention" test, tiny shape `BS,HEADS,SEQLEN,EMB = 4,2,16,8`).
- `test/backend/test_softmax_fusion.py`: 8 tests marked `@unittest.skip("needs RANGEIFY>1")`
  (`test_fuse_norm`, `test_fuse_argmax`, `test_fuse_softmax`, `test_fuse_softmax_dtype`,
  `test_double_gemm`, `test_attention_kernel_count`, `test_flash_attention`, `test_auto_softmax`) —
  `RANGEIFY` isn't a real `ContextVar` anymore (only `DEBUG_RANGEIFY` is), this is a stale name for
  what's now `PCONTIG`.

## 2. Methodology finding that had to be fixed first: `SCACHE` makes same-process PCONTIG-vs-baseline comparisons vacuous

Before any of the numbers below meant anything, I had to find and route around this. Worth flagging
prominently because it silently defeats the project's *existing* correctness tests, independent of
anything in this task.

`schedule_cache: dict[bytes, UOp] = {}` at `tinygrad/schedule/__init__.py:111`, gated by
`SCACHE = ContextVar("SCACHE", 1)` (`tinygrad/helpers.py:279`, **default ON**), keyed only by
`function.key` — a pure structural hash of the graph. It does **not** include `PCONTIG` (or any
other scheduling-affecting `ContextVar`) in the key.

Every existing PCONTIG test follows the pattern "compute under `Context(PCONTIG=2, ...)`, then
compute a comparison under a different `Context` in the *same process*, assert they match." Because
the schedule cache is keyed structurally, the second call transparently **reuses the first call's
cached (fused) schedule** — so the "comparison" is the fused kernel checked against itself, not
against an independently-scheduled baseline. Proof:

```
$ PYTHONPATH=. SCACHE=0 pytest test/backend/test_rangeify.py -q
FAILED TestDoubleMatmul::test_baseline (+ 19 more of its 22 tests)
FAILED TestPcontig::test_flash_attention - AssertionError: 72.45880889892578 not less than or equal to 1e-06
FAILED TestPcontig::test_flash_attention_bw
FAILED TestPcontig::test_flash_attention_opt
24 failed, 5 passed, 7 skipped
```
vs. the same suite under default `SCACHE=1`: **29 passed, 7 skipped** — all green, all vacuous.

All numbers in this document use isolated fresh subprocesses (no shared cache at all, strongest
guarantee) or explicit `SCACHE=0` (verified equivalent to fresh-process results, see below), plus
independent numpy ground truth wherever "correct" is claimed, never a same-process PCONTIG-vs-default
comparison.

## 3. Blocker 1 — PCONTIG's fusion is numerically wrong on multi-pass reduce patterns (concrete shapes, no symbolics involved yet)

Verified three independent ways: (a) two fully isolated subprocesses with the same seed compared to
each other, (b) the same, cross-checked against a numpy reference, (c) forcing the repo's own test
suite honest via `SCACHE=0` (§2).

| Case | Shape | PCONTIG | Kernels | vs. ground truth (numpy or PCONTIG=0) |
|---|---|---|---|---|
| Single reduce (argmax) | `(50,50)` | 2 | fused | **correct** (diff = 0) |
| Double matmul `a@b@c` | `(32,32)×3` | 2 | fused (per un-skip run) | **WRONG** — max abs diff 2.98 vs numpy |
| Plain softmax→matmul (`fa()`, the exact `TestPcontig` shape) | `(4,2,16,8)` | 2, 3 | 4→1 | **WRONG** — max abs diff 0.53; in-suite `mse=72.46` (limit `1e-6`) |
| `Tensor.softmax` (`NOOPT=1`, matches `TestSoftmaxFusion.test_softmax`) | `(32,10)` | 2 | — | **WRONG** — 100% of elements mismatched, max abs diff 2.34 |
| `Tensor.softmax` backward | `(32,10)` | 2 | — | **WRONG** — max abs diff 38.6 (values that should be ~1e-7) |
| Decode SDPA, GQA 32/8, Hd=128, Tk=512 | `(1,32,1,128)`×`(1,8,512,128)` | 2, 3 | 4→1 | **WRONG** — max abs diff 0.52 |
| Decode SDPA, GQA 32/8, Hd=128, Tk=2048 | same, Tk=2048 | 2, 3 | 4→1 | **WRONG** — max abs diff 0.51 |

The one case that's genuinely correct (`argmax`) is a *single* reduce with no re-read of its own
result. Every broken case shares the "flash-attention shape": compute a reduce, stash the partial
result, do more work that reads it back for a second pass (softmax's max→sub→exp→sum→div; chained
matmul's intermediate; SDPA's QK^T→softmax→@V). That's exactly the pattern `rangeify.py:267-282`'s
local-bufferize branch exists to accelerate — the bug is centered on the mechanism most relevant to
this task's goal, not a tangential one.

Repro (isolated, no ambiguity): `scratchpad/_isolated_fa.py` reproduces the `TestPcontig` shape in
two separate `python` processes with the same seed; `scratchpad/_check_double_gemm.py` and
`scratchpad/_check_argmax.py` cross-check against numpy. (Scratch scripts, not committed — see
"What I did not commit" below.)

## 4. Blocker 2 — concrete decode shapes blow Metal's threadgroup-memory budget, even at the milder PCONTIG=1

```
q=(1,32,1,128) k,v=(1,8,512,128) GQA, PCONTIG=1:
RuntimeError: Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)
```
65536 = `H(32) × Tk(512) × 4 bytes` **exactly**. This is not a Tk-sized tile — it's the full
`(H, Tk)` score matrix materialized into on-chip `threadgroup` memory with no chunking at all. At
`PCONTIG=1` this happens through `indexing.py:259`'s range-merge alone (the "milder" mechanism, not
even the dedicated online-softmax branch). `PCONTIG=0` (baseline) and `PCONTIG=2/3` (which happen to
route this particular shape through a different, still-wrong-but-non-crashing path, per §3) don't
hit this specific crash — so the failure mode is shape- and level-dependent, not monotonic with
`PCONTIG`'s numeric value.

## 5. Blocker 3 — symbolic Tk (the decision-critical question): fails to compile, doesn't degrade gracefully

This is the fact T1.8b flagged as calibrating: does fusion survive the JIT's symbolic `Tk` after
the first decode token? Built the exact construction `tinygrad/llm/model.py` uses
(`UOp.variable("start_pos", 0, max_context-1).bind(start_pos)` at `model.py:376,704,726`, feeding a
`k[..., 0:start_pos+T, :]`-style slice) as `vtk = UOp.variable("Tk", 1, 8192).bind(512)`,
`k = kfull[:, :, 0:vtk, :]`.

| PCONTIG | Result |
|---|---|
| 0 (baseline) | OK — 5 kernels (one more than the concrete case's 4, for the extra symbolic-bound slice) |
| 1 | `RuntimeError: Compiler encountered an internal error` (Metal pipeline-state creation) |
| 2 | same |
| 3 | same |

**Verdict: no, fusion does not survive a symbolic reduce range — and it fails harder than a graceful
fallback.** Traced the exact cause by dumping the generated MSL source before the runtime pipeline
call (`scratchpad/_diag_symbolic.py`):

```c
threadgroup __attribute__((aligned(16))) float buf4[8192];
threadgroup __attribute__((aligned(16))) float buf5[8192];
threadgroup __attribute__((aligned(16))) float buf6[8192];
...
for (int Lidx1 = 0; Lidx1 < data4_; Lidx1++) { ... }   // data4_ = the runtime Tk value (512)
```

The three `threadgroup` arrays are sized by the **Variable's static upper bound** (8192, the `max`
passed to `UOp.variable("Tk", 1, 8192)`), not the runtime-bound value (512) that the loop actually
iterates to. `3 × 8192 × 4 bytes = 96 KB` — 3x over Metal's 32 KB/threadgroup limit, guaranteed,
regardless of what Tk actually is at runtime. Offline `metal` compilation of this source succeeds
(the AIR/MTLB binary in the diagnostic dump shows it); it's Apple's runtime
`newComputePipelineStateWithFunction` that rejects it, surfacing as the opaque "Compiler encountered
an internal error" instead of the cleaner "Threadgroup memory size exceeds..." message §4 got for a
smaller (but still oversized) concrete case.

This is architectural, not a happens-to-be-too-big-this-time accident: for any realistic `max_context` (e.g.
qwen3.6's 131072 per `CLAUDE.md`), sizing local memory by the bound's static maximum requests ~1.5 MB
of on-chip memory. There is no tiling/chunking budget logic in `remove_bufferize`'s local-bufferize
branch at all — it partitions *which* ranges go local, not *how much fits*.

## 6. Un-skipping `test_softmax_fusion.py`'s `"needs RANGEIFY>1"` tests

Under `PCONTIG=2` (no other flags): 13 passed, 3 failed (`test_fuse_norm` FAILED with an
`assert_allclose` mismatch; `test_softmax`/`test_softmax_bw`, not in the skip list, also FAILED —
already covered in §3's table).

**Caveat that matters more than the pass count:** most of the "13 passed" go through `_test_fuse`,
which computes the same expression twice *in the same context* and compares — it never checks
against ground truth or against a genuinely different `PCONTIG` value, so it cannot by construction
catch the class of bug in §3. I independently ground-truth-checked one of the "passing" cases —
`test_double_gemm` — and it is **actually wrong** (§3's table, max diff 2.98 vs numpy), despite
"passing" here. Do not read the 13 passes as 13 correctness confirmations; read them as "13 cases
where PCONTIG=2's fused schedule was internally self-consistent," which argmax-style single reduces
and this double-matmul case both were, for different reasons (one is genuinely right, one is
consistently the same wrong answer both times). `test_attention_kernel_count` is the one exception
worth trusting as-is — it only asserts a kernel *count* via `check_schedule`, not values.

## 7. Illustrative timing (NOT validated — the compared output is wrong)

Given §3, a proper kernel-time table (as the task's step 3 asks for "if it works on both
concrete and symbolic") doesn't apply — fusion doesn't produce correct output on either. For
context on whether this would even be worth someone else's time to debug and fix later, one
directional data point anyway (min-of-20, `SCACHE=0` so this includes Python re-scheduling overhead
each call — not a clean steady-state kernel-time number, treat as a rough signal only):

```
Tk=512:  PCONTIG=0 (correct, 4 kernels) 8918us | PCONTIG=2 (WRONG output, 1 kernel) 7898us | 1.13x
Tk=2048: PCONTIG=0 (correct, 4 kernels) 8999us | PCONTIG=2 (WRONG output, 1 kernel) 12179us | 0.74x (SLOWER)
```
Even setting correctness aside, there's no clear win here — roughly break-even at Tk=512, worse at
Tk=2048. This tracks with T1.8b's finding that the existing 4-kernel chain already sits near ~46% of
bandwidth at Tk=8192 (the "honest prize is ~2x, not 5x" framing) — a kernel-count win from 4→1 isn't
automatically a wall-clock win once you account for the loss of parallelism / occupancy from cramming
everything into one giant single-threadgroup kernel with no tiling.

## 8. Fix vs. document

**No production code changed.** Everything found here is structural, matching the task's own
guardrail ("a wrong 'fix' here corrupts numerics silently"):
- The §3 correctness bug lives inside `remove_bufferize`'s substitution logic
  (`rangeify.py:267-282`, the `is_pcontig`/`is_subs` partition and `pm_gate_substitute` call) and/or
  `indexing.py`'s range-merge (`:259`,`:276`) — genuinely unclear which, and not a "guard is one
  comparison too strict" situation; it's a wrong-value bug in a rewrite rule.
- The §4/§5 threadgroup-sizing issue needs an actual occupancy-aware local-memory budget (real
  tiling), not a guard tweak.
- The §2 `SCACHE` cache-key gap is real and separately worth fixing, but "make the schedule cache
  key context-aware" is a design decision about a perf-critical, heavily-used caching mechanism
  (it's what makes `tinygrad/llm`'s `@function`-based JIT reuse cheap) — not small.

All are flagged for Artur to triage/upstream at his discretion, not touched here.

## 9. Recommendation

**Dead end for T1.7 as currently implemented — do not wire PCONTIG into `tinygrad/llm`.** This
isn't "still experimental, needs more validation" — it's "produces silently wrong logits" if pointed
at production decode attention today, on top of not even reaching the symbolic-Tk case it would need
to reach for real serving (T1.8b's finding #1: Tk goes symbolic after token 1).

Given T1.8b already closed the `custom_kernel` route (symbolic-shape assertion, no BEAM, no warp
primitives) and this closes the `PCONTIG` route (wrong numerics on concrete shapes, hard compile
failure on symbolic ones), **both currently-known fused-attention tracks for Metal decode are
blocked by upstream-shaped gaps**, not by anything fixable in this fork with a small diff:
1. `custom_kernel` needs symbolic-shape support in tinygrad's custom-kernel path (T1.8b blocker 1).
2. `PCONTIG`'s local-bufferize needs a real tiling/chunking budget instead of sizing by the full (or
   worst-case) extent (this task's blocker 3), plus someone upstream fixing the §3 correctness bug
   in `remove_bufferize`.

Concretely, redirect effort to the levers that are already yielding real, verified wins: **T1.10**
(quantized MATVEC, already landed 4x on Q4_0/Q6_K), **T4.2** (Q4_K dequant, already landed 2x),
**T4.4** (BEAM prefill anomaly, still open, small filler). The existing 4-kernel SDPA chain stays the
production path; per T1.8b it's already at ~46% of bandwidth at Tk=8192, so the remaining ~2x gap vs
llama.cpp is a kernel-tuning problem on that chain (or waiting on upstream for either blocked track),
not a fusion problem this fork can currently solve.

Two things worth a word to Artur for possible upstream reporting (not filed by me, per this fork's
PR-prep discipline — small, hand-verified, named-hardware only):
- The `SCACHE` cache-key gap (§2): it makes the *entire* PCONTIG test suite currently green for the
  wrong reason. That's a real, cheaply-reproducible finding independent of T1.7's goal.
- The correctness bug itself (§3): precise, minimal, ground-truth-verified repros exist
  (`scratchpad/_isolated_fa.py`, `_check_double_gemm.py`) if someone wants to open an issue.

## What I did not commit

The exploration/repro scripts referenced above live in this session's scratchpad
(`/private/tmp/claude-501/.../scratchpad/`), not in the repo — they're throwaway diagnostics, not
a durable test suite, and none of them belong in `tinygrad/` or `test/`. If this analysis needs to
be reproduced later, the shapes/seeds/commands are fully specified inline above.
