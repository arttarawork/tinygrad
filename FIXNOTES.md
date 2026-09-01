# T4.73: WY prefill->decode garbage-token bug -- diagnosis (no fork-local fix found)

Status: **diagnosed, not fixed**. The STOP condition in this task's brief applies: the evidence below shows
this is a tinygrad-core jit capture/replay issue, not a `model.py` usage-pattern one, and not a WY-specific
one at all. No patch is included. `test/unit/test_gdn_scan_parity.py::TestGDNScanJitReplayRegression` tracks
it as `@unittest.expectedFailure` so the suite stays green and a future core fix will visibly need to flip it.

## 0. The bug, restated, and step-1 verification

`PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device CPU --chunks 4`
still reproduces exactly as briefed:
```
FIRST-DIVERGENCE step=decode#2 tensor=logits[0] ref=0.008032575249671936 got=-0.0022140454966574907
```
`--chunks 1` and `--chunks 2` still PASS. This was re-verified as step 1, unchanged.

## 1. Root cause: what it is NOT (all directly proven), what it IS (characterized), what's still open

### 1a. Ruled out, with proof

- **Not `gdn_scan_wy`'s math, not anything WY-specific.** `extra/wy_boundary_repro.py` now has a `--impl
  {wy,loop}` flag (added this task). `--impl loop` reproduces the **identical** divergence, byte for byte:
  ```
  $ PYTHONPATH=. .../python extra/wy_boundary_repro.py --device CPU --chunks 4 --impl wy
  FIRST-DIVERGENCE (impl=wy)   step=decode#2 tensor=logits[0] ref=0.008032575249671936 got=-0.0022140454966574907
  $ PYTHONPATH=. .../python extra/wy_boundary_repro.py --device CPU --chunks 4 --impl loop
  FIRST-DIVERGENCE (impl=loop) step=decode#2 tensor=logits[0] ref=0.008032575249671936 got=-0.0022140454966574907
  ```
  Verified via `tinygrad.llm.model.gdn_last_scan_impl` (the T4.69b dispatch-recording list) that `--impl loop`
  really does force every prefill/decode call onto the per-token loop (never `gdn_scan_wy`) for this whole
  run. The per-token loop **does not call `_gdn_tri_inverse`, does not build `a_bar`/`kkt`/`qk`, and shares
  no gdn_scan_wy code at all** -- so the bug cannot live in WY's triangular-solve/cumprod math. It must live
  in code the two impls share: the `_attention` preamble (conv window, `pad_to`, symbolic slicing) or
  something entirely outside `GatedDeltaNetBlock`.

- **Not `model.py`'s symbolic `pad_to`/window-slicing usage (the brief's suspects 1 and 2).** Both suspects
  predict the corruption should be visible as wrong VALUES fed into the scan (wrong padding -> wrong
  `alpha`/`beta`/`k`/`v` at the boundary -> visibly wrong `recurrent_state`/`conv_state` right after the
  mismatched chunk). Directly measured (`fingerprint_tensors`, both blocks) at every step of the `--chunks 4`
  run: `recurrent_state` diff is a constant `4.28e-6`, `conv_state` diff is exactly `0`, at **every single
  step including immediately after the mismatched chunk** -- i.e. no different from ordinary float
  non-associativity. Only the **returned logits** are wrong, and only at **one specific call**. A wrong-
  padding bug cannot produce "state always correct, output wrong exactly once."

- **Not `schedule/memory.py`'s arena/buffer-reuse planner.** `NO_MEMORY_PLANNER=1
  PYTHONPATH=. .../python extra/wy_boundary_repro.py --device CPU --chunks 4` reproduces the identical
  divergence. `NO_MEMORY_PLANNER=1` makes `memory_plan_rewrite` a no-op (`schedule/memory.py:21`, every
  internal buffer keeps its own dedicated allocation, no `TLSFAllocator` suballocation/reuse at all) -- so
  this rules out the "stale bytes from an aliased/reused scratch buffer" theory as the mechanism, at least at
  tinygrad's own memory-planning layer.

- **Not specific to `GatedDeltaNetBlock` paired with an attention block.** A stack of **two
  `GatedDeltaNetBlock`s** (no `TransformerBlock`/attention/RoPE/KV-cache at all) reproduces it too, through
  the identical chunked-prefill-then-decode jit sequence. So `cache_kv`/RoPE/causal masking are not
  ingredients.

- **Not something a lone block can trigger.** The same jit pattern (prefill chunks 32,32,32,29, then 3
  decode steps) driven through a **single** `GatedDeltaNetBlock` called directly (no `Transformer`, no
  `embd`/`norm`/`out_proj`, raw feature vectors in and out) does **not** reproduce it -- diffs stay at
  float-noise level (`1e-7`..`0`) at every step. Needs at least two stacked blocks. A simplified two-
  `GatedDeltaNetBlock` stack that ALSO drops `embd`/`norm`/`out_proj` (returning one raw tensor instead of
  the real `forward()`'s `(argmax_id, logits)` tuple) also does **not** reproduce it. So the trigger needs
  something about the fuller call shape (2+ stateful blocks, plus the head/output-projection machinery
  and/or the 2-tensor return) -- not narrowed further within this task's budget.

### 1b. Characterized (all directly measured)

- **The wrong value appears at exactly one call: the DECODE jit key's own first CAPTURE** (its 2nd-ever
  invocation, `_TinyJit` `cnt: 0->1`), which happens to be the call right after chunked prefill finishes.
  Verified by monkey-patching `GatedDeltaNetBlock._attention` to log every call that actually runs Python
  (captures/eager calls only -- replays never re-run `_attention`'s Python at all, confirmed against
  `engine/jit.py`'s `_TinyJit.__call__`): only 4 Python-level calls happen for `--chunks 4`, at
  `start_pos={0,32,125,126}`; the `start_pos=126` call (decode's own `cnt==1` capture) is where the
  divergence's step (`decode#2`) is produced.
- **It self-corrects one step later.** `decode#3`'s (jit) logits are back to float-noise-level agreement
  with reference. Even more tellingly: `decode#2`'s WRONG jit logits (`[-0.0022140455, 0.0023561937,
  -0.0025964978, -0.028997280, 6.1558261e-05, ...]`) numerically match **reference's `decode#3`** logits
  (`[-0.0022140502, 0.0023562356, -0.0025964624, -0.028997287, 6.1556260e-05, ...]`) to ~1e-6 -- i.e. jit's
  decode#2 computed something that looks like it read state "one step ahead" of where it should have been,
  not a random/algorithmic wrongness. This is a capture-time transient, not a persistent corruption (matches
  1a's "state buffers stay correct" finding).
- **Needs a REPLAY at a mismatched bind value, but not merely one.** `--chunks 3` (prefill chunks 32,32,29:
  capture at 32, then ONE replay, immediately mismatched at 29) passes clean -- even re-checked at
  `--decode-steps 12` (no delayed divergence). `--chunks 4` (32,32,32,29: capture at 32, then a MATCHING
  replay at 32, THEN a mismatched replay at 29) fails. An exact-multiple prompt (4 chunks of 32, i.e. two
  replays, both matching, mismatch never happens) passes clean. So the trigger needs >=1 matching replay
  *and then* a mismatched one -- a single mismatched replay right after capture is safe.
- **Needs `chunk_size == 32` specifically.** `--chunks 4 --chunk-size {8,12,16,24}` all PASS, with the exact
  same replay shape (capture, matching replay, mismatched replay) merely scaled down. Only `chunk_size=32`
  (also tested at `--chunks 5`) fails. This is a concrete-size-dependent trigger, not a general "any
  chunking" one -- and 32 is not an arbitrary probe value, **it is `gdn_chunk_for`'s own auto-selected
  default chunk width on METAL/NV/CUDA** (`tinygrad/llm/model.py:23`, `return 32 if dev... in ("METAL", "NV",
  "CUDA") else 1`) -- i.e. this is the width the pooled server actually runs at today, not an edge case.

### 1c. Leading (unconfirmed) core loci -- for whoever picks this up next

Both are consistent with everything in 1a/1b; neither was confirmed further within this task's CPU/model.py-
usage-pattern scope (confirming either needs tracing actual lowered UOp graphs, which is core-internals work
beyond "verify a model.py usage pattern"):

1. **`_TinyJit.__call__`'s capture-time double execution** (`tinygrad/engine/jit.py`, the `cnt == 1` branch,
   ~line 266-296): on the very first capture of any jit key, `ret = self.fxn(*args, **kwargs)` runs the
   function EAGERLY first (to discover the graph), and its result is realized; then the graph is lowered
   (`jit_lower`) and `ret = self.captured(input_buf_uops, var_vals)` re-executes the just-built
   `CapturedJit` a SECOND time with the same inputs, and this second result is what's actually returned
   (`CapturedJit.__call__`, `engine/jit.py` ~line 184: `return self.ret`, where `self.ret` are the same
   Tensor objects the eager pass already computed and realized). Since this "capture" mechanism runs for
   EVERY jit key's own first capture, it must be either (a) fine in general and only exposed by some very
   specific prior state, or (b) papered over by simple cases (constant re-computation on already-correct
   values is a no-op) and only visible when something upstream (the 32-wide mismatched-replay chunk) leaves
   a transient side effect the second execution reads differently than the first. This fits 1b's "self-
   corrects one step later" observation (a one-time double-read/write at the capture boundary) better than
   any hypothesis involving persistent buffer corruption.
2. **`codegen/__init__.py`'s `to_program_cache`** (keyed by `to_program_key(ast, renderer)`): compiled
   kernels are cached and reused across structurally-identical `Ops.FUNCTION`/`Ops.CALL` bodies from
   DIFFERENT call sites (this is `@function(precompile=True)`'s whole point, per the existing
   `REPORT.md` SS1 analysis this task did not need to redo). Two stacked `GatedDeltaNetBlock`s compiled in
   the same process, with a size-32 axis, are exactly the kind of "structurally similar small kernels from
   different contexts" scenario where an insufficiently-specific cache key could serve the wrong compiled
   program to the wrong call site. Not confirmed; would need to inspect `to_program_key`'s actual hash
   inputs against the specific kernels this repro compiles.

Both point at **tinygrad core** (`engine/jit.py` and/or `codegen/`), not at `tinygrad/llm/model.py`'s usage
of symbolic shapes. Per the STOP condition, this task does not patch core.

## 2. Why no CPU suite ever caught this

Two independent, compounding reasons, both directly confirmed by reading the existing tests:

1. **`test_gdn_scan_parity.py`'s own helpers never go through `TinyJit` at all**, by explicit, documented
   design (`run_attention` calls `block._attention(...)` directly -- see the file's own header comment: "no
   JIT/TinyJit -- that's the whole point of this harness, it isolates the scan from generate()'s JIT
   machinery"). No amount of chunk-count tuning in that file's existing tests could ever have found a
   capture/replay bug -- the file structurally never captures or replays anything. That's also why the new
   regression test (below) cannot reuse those helpers and instead drives a real chunked `TinyJit`, borrowed
   from `extra/wy_boundary_repro.py`.
2. **The one place that DOES exercise real jit-captured chunked GDN prefill,
   `test/unit/test_llm_server.py::TestRecurrentChunkedPrefill`, happens to sit in the safe zone on both axes
   this task found:** `test_chunked_matches_one_token_prefill` uses `GDN_CHUNK=4` with a 9-token prompt ("4
   tokens at chunk 4 -> 4+4+1"), i.e. **3** total prefill chunks (capture at 4, then ONE replay, immediately
   mismatched at the trailing 1-token remainder) -- exactly the `--chunks 3` shape this task proved passes
   clean (no *prior matching* replay before the mismatch) -- **and** at chunk width 4, not 32. Neither
   `GDN_CHUNK=4` nor a single post-capture mismatched replay is enough to trigger this bug (1b). On top of
   that, `gdn_chunk_for`'s own device-aware default (`model.py:22-24`, tested directly by
   `test_auto_chunk_is_device_aware`) returns **1** for CPU when `GDN_CHUNK` isn't set explicitly -- so any
   CPU test that exercises `generate()`'s real auto-chunking without an explicit override never even attempts
   a >1-token chunk, let alone width 32, regardless of prompt length. Both of these are why extending
   coverage here needed a purpose-built, non-default-chunk-width harness (`wy_boundary_repro.py`,
   `T4.76`-era work, reused as-is by this task's regression test) rather than anything already in CI.

## 3. Fix rationale: why nothing was patched

- The WY math (`gdn_scan_wy`, `_gdn_tri_inverse`) is innocent -- proven identical-under-loop (1a).
- `model.py`'s symbolic `pad_to`/window-slicing usage in `GatedDeltaNetBlock._attention` is innocent -- state
  buffers stay correct at every step; a padding bug cannot produce "state fine, one output wrong" (1a).
- A defensive `.contiguous()` on `_attention`'s return value (tried, monkey-patched, `--chunks 4` under both
  `--impl wy` and forced-loop) made **no difference** -- same byte-identical divergence. Not committed
  (no-op change).
- The one model.py-adjacent lever that *would* structurally prevent ANY mismatched-bind replay -- keying the
  prefill jit cache by the exact bound `toks` value instead of `chunk_size` -- was considered and rejected:
  `Transformer.__call__`'s own T4.12 comment (`model.py`, just above `__call__`) explains this was already
  tried and deliberately reverted, because it defeats the entire point of the symbolic-Variable replay
  design (one capture serving every chunk width up to `chunk_size`, not a fresh capture per exact remainder
  length) and would reintroduce the N-times-recompilation cost T4.69a's whole WY effort was measured against.
  It would also only mask THIS specific trigger shape while leaving whatever the real core mechanism is
  intact for other call shapes (e.g. `redo_ids`/speculative-decode's own rollback replay path, `model.py`
  ~line 1632, does its own mismatched-width rebind and was not exercised by this task at all).
- No other narrower, model.py-local change was found that fixes the observed corruption without either (a)
  broadly changing `generate()`'s chunking/keying contract, or (b) blindly perturbing memory layout in a way
  that might dodge this exact toy geometry without addressing the actual defect at production scale (48
  heads, head_dim=128) -- which this task did not have the hardware/scope to validate either way. Per the
  STOP condition, that risk was not taken.

## 4. What's committed

- `extra/wy_boundary_repro.py`: added `--impl {wy,loop}` (default `wy`) so "loop reproduces this too" is a
  standing, re-runnable command instead of a one-off patch. No other behavior change (default `--impl wy` is
  byte-identical to before).
- `test/unit/test_gdn_scan_parity.py`: `TestGDNScanJitReplayRegression` (`@unittest.expectedFailure`) --
  reuses `extra/wy_boundary_repro.py`'s harness at the exact proven-failing parameters (`--chunks 4
  --chunk-size 32`, two-block stack, seed 5). Confirmed failing on this branch (`pytest -k JitReplay` ->
  `1 xfailed`); will need `@unittest.expectedFailure` removed once a real core fix lands, at which point it
  becomes a normal regression guard.
- This file.

No changes to `tinygrad/llm/model.py` or any core file.

## 5. Gates (this branch, CPU/NULL only, as required)

```
CHECK_OOB=1 DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest \
  test/unit/test_gdn_scan_parity.py test/unit/test_attention.py test/unit/test_spec_decode.py -x -q
# -> 49 passed, 9 skipped, 1 xfailed
/Users/artur/Documents/tinygrad/.venv/bin/python -m mypy tinygrad/        # -> Success: no issues found in 219 source files
/Users/artur/Documents/tinygrad/.venv/bin/python -m ruff check .          # -> All checks passed!
```
`serve.py` was not touched -- the `SPEC=2 DEV=NULL .../test/null/` gate does not apply.

## 6. Exact commands for Artur's hardware revalidation

Nothing here has changed the serve stack, so the ORIGINAL serve-side reproduction guidance still stands
exactly as briefed: any 1-chunk prompt after warmup under `GDN_SCAN_IMPL=2`, e.g.
```
WY_TRACE=1 DEV=NV JITBEAM=2 PARALLEL=2 BEAM_TIMEOUT_SEC=30 GDN_CHUNK=32 GDN_SCAN_IMPL=2 \
  PYTHONPATH=. python -m tinygrad.llm -m <the Qwen3.8-27B GGUF> --device-map <the METAL+NV map> \
  --max_context <ctx> --serve 8081
```
then drive one request and read the `[WY_TRACE]` lines. **New, given section 1a's finding: also worth one
run with `GDN_SCAN_IMPL=1` (loop) at the same `GDN_CHUNK=32`** -- if this task's CPU characterization
transfers to hardware, decode should still go wrong (garbage/NaN) even with WY fully disabled, which would
be the hardware-side confirmation that this was never a WY-only bug. If loop turns out clean on real
hardware where WY isn't, that would be new information this task's CPU repro did not predict and is worth
flagging back here.

CPU-side reproduction/characterization commands (no hardware needed, all re-run by this task):
```
# the original bug, still reproduces exactly as briefed:
PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device CPU --chunks 4
# not WY-specific (T4.73's new flag):
PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device CPU --chunks 4 --impl loop
# not the memory planner:
NO_MEMORY_PLANNER=1 PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device CPU --chunks 4
# safe zone (all PASS): a single mismatched replay with no prior match, and every chunk_size but 32
PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device CPU --chunks 3 --decode-steps 12
PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device CPU --chunks 4 --chunk-size 16
# the regression test:
CHECK_OOB=1 DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest test/unit/test_gdn_scan_parity.py -k JitReplay -v
```
