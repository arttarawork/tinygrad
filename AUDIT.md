# T4.73 — WY prefill→decode corruption: bug audit + candidate fixes

Scope: bug audit + candidate fixes only. I cannot open METAL/NV (hard rule) and could not reproduce the
bug on CPU (confirmed again here — see §6's second bullet). Everything below is static/structural analysis
of `tinygrad/llm/model.py`, `tinygrad/engine/jit.py`, `tinygrad/schedule/memory.py`, plus one CPU-side
detector (`extra/wy_alias_audit.py`). Hardware confirmation is out of scope for this task by design.

## 0. The bug, restated from the T4.72 ladder (TASKS.md, 2026-08-31)

Serving qwen3.8-27B on the pooled METAL+NV split with `GDN_SCAN_IMPL=2` (the chunkwise WY scan,
`gdn_scan_wy` in `model.py`): chunked WY prefill completes at full speed and correct throughput, then the
**first decode step** — which T4.69b's gate correctly routes to the LOOP impl at `T_pad==1` — samples an
**out-of-vocab token id** (e.g. 248320 vs vocab ~152k), which is a tokenizer `KeyError`, not merely "a wrong
but in-range token". Eliminations already done on hardware (T4.72 round E and earlier):
- Persists with `BEAM=0`/`JITBEAM=0` → **not a miscompiled searched kernel**, it's graph-level.
- Does **not** reproduce on CPU single-device.
- Does **not** reproduce on CPU multi-device (`CPU:0`/`CPU:1`, the full `test_llm_device_map` suite passes
  under `GDN_SCAN_IMPL=2`).
- The LOOP impl (`GDN_SCAN_IMPL=1`) is clean on the *same* real hardware, same head-group split.
- MTP is exonerated (round B reproduces the bug with MTP off; `serve.py` never routes through
  `speculative_generate` unless `--mtp` is set and the model has an `mtp_head` — see §5).

So: real-device-specific (METAL and/or NV), WY-path-specific, at the prefill→decode boundary, not a BEAM
artifact. An out-of-vocab (not merely wrong) token id is itself a strong clue: `forward()`'s greedy path is
`self.output(...).argmax(-1, ...)` over a fixed `vocab_size`-wide axis — argmax over a correctly-shaped
tensor can *never* return an index ≥ vocab_size no matter how wrong the logit *values* are. Getting a
literal out-of-range integer back on the host points at a **buffer-identity/aliasing** failure (the host
ends up reading bytes that were never a valid vocab-size logits row, or a corrupted-shape read), not a pure
numerical error in the state math — exactly the class of bug this audit was asked to hunt for.

## 1. Every WY-path tensor that crosses a replay boundary or feeds an in-place store

`gdn_scan_wy` (model.py:83-142, current numbering with all three candidates applied) returns
`(final_state, out.transpose(1, 2))`. Classifying both, and everything downstream to the two in-place
stores in `GatedDeltaNetBlock._attention`'s else-branch tail (model.py 584-690, current numbering):

| Tensor | Where it's built | UOp shape (pre-patch) | View or computed? | Crosses a boundary? |
|---|---|---|---|---|
| `final_state` | `gdn_scan_wy` return #1: `a_bar[:, :, T-1, :].unsqueeze(-1) * (state + u.transpose(-1,-2) @ k)` | top-level op = `MUL` (elementwise) | **Computed**, not a bare view — but its *dependency graph* threads through a SHRINK (`a_bar[:,:,T-1,:]`), a RESHAPE (`unsqueeze`), a PERMUTE (`u.transpose`), and a MATMUL, none of which are present at all in LOOP's equivalent (see §2) | **YES** — this is `run_scan`'s `state` (G≤1) or one `group_states[i]` (G>1), which ultimately feeds `self.recurrent_state`'s `.store()`, read by the *next*, differently-shaped jit capture (decode) |
| `out.transpose(1, 2)` | `gdn_scan_wy` return #2 | `PERMUTE` | **Bare view** over `out`'s buffer (confirmed empirically: `extra/wy_alias_audit.py` Part A prints `op=Ops.PERMUTE`, `has_buffer_identity()=False` on the unpatched function) | Feeds `run_scan`'s `stacked`/`g_outs`, consumed only *within this same call* (feeds `core`, this call's own logits) — not read by a later capture, but SPEC_NOTES.md §3's exact pattern (a returned movement op) |
| `group_states[i]` (G>1 path) | `g_state.contiguous()` immediately after each group's `run_scan` call (model.py, T4.62) | `CONTIGUOUS` | Materialized (T4.62 already does this) | No — local to this call, but its OWN input (`final_state` above) is view-heavy |
| `state` (the value stored into `recurrent_state`) | G≤1: `run_scan`'s raw return, **no `.contiguous()`**. G>1: `group_states[0].cat(*group_states[1:], dim=1)`, **no `.contiguous()` on the cat itself** (only the pre-cat per-group pieces are contiguous) | `CAT` (G>1) or the raw scan return (G≤1) | Lazy composed expression, not yet a forced kernel boundary, until candidate-3 | **YES — this is the actual store source.** `self.recurrent_state.uop.store(state.cast(...).uop)`: read by the next jit capture (decode, or the next prefill chunk) |
| `stacked` (this call's own output) | G≤1: raw `run_scan` return. G>1: `group_outs[0].cat(...)`. Either way: **`stacked.contiguous()` is explicitly called** right before `core` | `CONTIGUOUS` (forced) | Materialized | No — feeds `core`, this call's own logits only |
| `conv_state` store source | `conv_window[:, T:T+K-1].cast(self.conv_state.dtype)` | `CAST` over a SHRINK of `win` (a fresh per-call buffer from `Tensor.zeros(...)`) | Computed (cast forces a real op) | Yes, but **identical for LOOP and WY** (built before `run_scan` is even dispatched) — ruled out as the differentiator, see §4 |
| `self.recurrent_state` / `self.conv_state` themselves (the target buffers) | `Tensor.zeros(...).clone()` in `_init_state`, called once, **outside** the `@function(precompile=True)` trace (`FFNBlock.__call__` calls `_init_state` before defining `_run`) | `AFTER(RESHAPE(BUFFER), STORE(...))` | Owned, persistent buffer identity, stable for the model's lifetime | Confirmed empirically **held** (protected from arena reuse) in every capture, both impls — `extra/wy_alias_audit.py` Part B |

**The one clean asymmetry in the current (pre-T4.73) code**: `core`'s source (`stacked`) gets an explicit,
unconditional `.contiguous()` right before crossing into `Tensor(...).uop.after(state_store)`. `state` —
the value that actually needs to survive into a *different, later* jit capture — does not get the
equivalent treatment; it relies on `cat()` (G>1) or a bare `run_scan` return (G≤1) plus `.cast()` alone to
already be "fully its own buffer" by the time `.store()` reads it. Both `core` and `state` cross a boundary
that matters (one within-call, one cross-call — cross-call is the *more* dangerous of the two), yet only one
of them got the defensive treatment. That asymmetry is candidate 3.

## 2. The exact loop-vs-WY graph-shape difference feeding the store

**LOOP's returned `state`** (the per-token python-unrolled recurrence, model.py's `run_scan` else-branch):
every step is `state = s1 + delta * k[:, :, t]` — a straight-line chain of `T_pad` elementwise ADD/MUL/SUM
ops, **all at the exact same shape as `recurrent_state` itself** `(B,H,V,K)`. No matmul, no transpose, no
slice, no cat appears anywhere in the chain that produces `state`. It's structurally identical in *kind* to
counting a `for` loop of "state = state + adjustment" — deep, but uniform and narrow.

**WY's returned `final_state`**: `a_bar[:, :, T-1, :].unsqueeze(-1) * (state + u.transpose(-1, -2) @ k)`.
Even *after* candidate-1 wraps this in `.contiguous()`, the value that gets materialized still has, in its
immediate dependency fan-in: a SHRINK+RESHAPE of `a_bar` (itself a `cumprod` over the whole chunk), a
PERMUTE of `u`, a MATMUL against `k`, and `u` itself comes from `_gdn_tri_inverse(m) @ rhs` — a
Neumann-series doubling loop of `ceil(log2(C))` (C,C)-shaped matmuls (`m`, `n_pow`, `p`), plus `kkt`/`qk`
(both (B,H,T,T)) and `rhs`/`v_tilde` (both (B,H,T,V)). None of these shapes or op kinds (matmul, (T,T)
matrices, triangular solve) exist anywhere in LOOP's graph at all. Measured directly (both impls, real
48-head geometry, `extra/wy_alias_audit.py` Part B, at first capture, all three candidates applied): **WY's
prefill capture registers 50 distinct schedule-visible buffers vs LOOP's 44 (+6, ~14%)** — a real but *modest* difference, not the order
of magnitude I initially expected; `GDN_HEAD_GROUPS`'s per-group `.contiguous()` calls already bound the
blow-up. The qualitative difference (matmul/(T,T)-shaped intermediates vs a uniform elementwise chain) is
the more important finding than the raw count.

## 3. Why CPU masks it, ranked

1. **(Leading) Real-device allocator/async-timing hazard invisible to CPU by construction.**
   `tinygrad/schedule/memory.py`'s `memory_plan_rewrite` (the buffer-donation/arena-reuse pass) and
   `tinygrad/engine/jit.py`'s `held_bufs` computation are **backend-agnostic Python** — confirmed via
   `git log --oneline -- tinygrad/schedule/memory.py tinygrad/engine/jit.py`: neither file has been touched
   by *any* T4.6x fork work (jit.py's last fork-side change is T4.33's upstream-sync merge); this is stable,
   shared upstream plumbing that runs identically regardless of `DEV`. A *pure* liveness-interval bug in
   that pass would misplan buffers on CPU too. Since it provably doesn't, the actual hazard must live at a
   layer this generic Python pass cannot see or control: a real device's own memory allocator recycling a
   physical buffer's address for a new allocation as soon as its *host-side* liveness bookkeeping says it's
   free, racing against that buffer's *actual* asynchronous kernel writes still in flight on the device
   queue. `engine/realize.py`'s `run_linear(wait=False)` (referenced directly in `generate()`'s own T2.5
   comments) confirms realize/dispatch is async-by-design on real backends. CPU's runtime has no comparable
   async command queue or address-recycling allocator to race against — there is nothing for a missing
   fence to be missing *from*. This is consistent with the bug being METAL/NV-specific and *not*
   reproducing even on CPU multi-device (`CPU:0`/`CPU:1`): multi-device alone doesn't introduce real
   asynchronous, independently-clocked hardware queues the way an actual GPU backend does.
2. **Graph-complexity/exposure multiplier (secondary, not sufficient alone).** WY's heavier intermediate
   set (§2) means more schedule-visible buffers and more buffer-donation *events* per capture, per GDN
   block, and this fork's forward pass captures the **entire model** as one `TinyJit(self.forward)` linear
   program (`Transformer.forward` loops over all blocks; the JIT wraps that whole method, keyed only by
   `(is_prefill, greedy, chunk_size, spec)` — model.py:1009-1054, current numbering). More reuse events raise the odds of
   tripping *any* latent real-device timing hazard, without needing the hazard itself to be WY-specific in
   nature. I cannot confirm or rule out a specific cross-block collision (a WY block's intermediate reusing
   a *different* block's buffer address) — the per-block body is wrapped in `@function(precompile=True)`
   (a `Ops.FUNCTION`/`CallInfo`-based mechanism, `tinygrad/function.py`, distinct from `TinyJit`), and I did
   not fully chase how far the outer capture's `memory_plan_rewrite` sees into vs. around a `FUNCTION`
   node's internals — flagged as unresolved, see §6.
3. **(Checked, ruled out)** `generate()`'s own documented aliasing fix for `drain_every>1`
   (model.py:1352-1357, current numbering: *"the JIT reuses its output buffer across replays... a not-yet-drained `out` would
   silently alias data the next chained step's replay overwrites"* — literally the same bug class this audit
   was asked to hunt for, already found and fixed once in this exact file) does **not** apply here:
   `tinygrad/llm/serve.py:129` calls `model.generate(ids, temperature=temperature)` with no `drain_every`
   argument, i.e. the default `drain_every=1`, which drains (host-reads) every sampled token immediately,
   before any further replay can touch its buffer — this is symmetric for LOOP and WY and was already
   correct before T4.69a. Confirmed by reading the exact call site; ruled out as the differentiator.

## 4. `conv_state` — checked and ruled out as the differentiator

`conv_state`'s store source (`conv_window[:, T:T+K-1].cast(...)`) is built entirely *before* `run_scan` is
even dispatched (model.py 605-611, current numbering) — completely independent of `GDN_SCAN_IMPL`. If this were the
corruption site, LOOP would be corrupted too, which hardware disproves. Not a candidate target.

## 5. `speculative_generate` / MTP

Exonerated by the T4.72 ladder itself (round B reproduces with MTP off) and independently by reading
`serve.py`: `use_spec = self.server.mtp and model.mtp_head is not None` gates the speculative path entirely
behind `--mtp`, which the failing repro didn't set. Not investigated further here.

## 6. What I could not resolve (honest gaps)

- **Whether the outer whole-model `TinyJit(self.forward)` capture's `memory_plan_rewrite` treats each
  block's `@function(precompile=True)`-wrapped body (`Ops.FUNCTION`) as opaque (its own internal buffers
  never enter the outer linear program's `si.src[1:]` at all) or transparent (fully visible/flattened).**
  I read `tinygrad/function.py`'s `_function.__call__` and `UOp.call`'s `Ops.FUNCTION` construction far
  enough to see that implicit buffer inputs (weights, `recurrent_state`, `conv_state`) get folded into the
  `FUNCTION` node's own call-site argument list (`graph_rewrite(uret, pm_ctx, ...)` in `function.py`), which
  is enough for `held_bufs`'s *global*, `all_tensors`-based computation to still catch them regardless (and
  `extra/wy_alias_audit.py` Part B empirically confirms `recurrent_state`/`conv_state` are held in my
  simplified single-block harness) — but I did not verify whether a `FUNCTION` node's *internal* scratch
  buffers get their own independent, isolated memory-planning pass at precompile time, or whether they're
  visible to (and thus can collide within) the enclosing whole-model capture's arena. This bears directly on
  hypothesis §3.2 (cross-block collision) and I'm flagging it rather than guessing further — it would need
  either a deeper read of `engine/realize.py`'s `compile_linear`/`link_linear` or a hardware-side experiment
  (e.g. does the corruption still occur with `GDN_HEAD_GROUPS=1` forced, removing the extra per-block
  buffer traffic, while keeping `GDN_SCAN_IMPL=2`? That single A/B would be informative and needs no code
  change — it's a pure env-var experiment the coordinator could run before or alongside the candidates
  below).
- **I cannot reproduce the corruption itself anywhere**, by design/hard-rule (no METAL/NV) and by ladder
  history (no CPU repro, single or multi-device). Everything above is inference from code structure plus one
  CPU-side structural detector (`extra/wy_alias_audit.py`), not a confirmed root cause. Treat §3's ranking as
  a prioritized *search order* for the hardware A/B, not a proof.

## 7. Ranked hypotheses (search-order for the hardware A/B)

1. **Highest confidence for "this is the fix": the missing `.contiguous()` on `state` before its
   `self.recurrent_state.store()`** (§1's "one clean asymmetry", candidate 3). It's the one concrete,
   demonstrable, LOOP-vs-WY-irrelevant-but-currently-asymmetric spot, it's on the *exact* active code path
   for the reported config (48 heads → `GDN_HEAD_GROUPS` auto-splits to G=2), and it directly targets the
   value that crosses into the next jit capture.
2. **Defensive, lower marginal confidence (G≤1 geometries; general hardening): `.contiguous()` on
   `final_state` inside `gdn_scan_wy`** (candidate 1). For the *active* G=2 config this mostly overlaps with
   candidate 3 (each `group_states[i]` is already `.contiguous()`'d at the call site before candidate 3 even
   applies) — its independent value is the G≤1 path (e.g. qwen3.6-35B's 32-head geometry, where `state` from
   `run_scan` currently gets no `.contiguous()` at all before candidate 3's fix, and none before that either)
   and matches SPEC_NOTES.md §3's established "materialize inside the traced function" convention.
3. **Defensive, general hardening, not demonstrated to be live today: `.contiguous()` on
   `out.transpose(1,2)`** (candidate 2). Closes the literal "bare movement-op view returned across a
   function boundary" pattern SPEC_NOTES.md §3 already found and fixed once elsewhere in this file — but
   every *current* caller of `gdn_scan_wy` already forces its own `.contiguous()` on this value before use
   (`g_outs.contiguous()` in the head-group split, or `stacked.contiguous()` regardless of G), so under
   today's call graph this specific view never actually escapes unmaterialized. Worth an A/B slot mainly as
   insurance against a future call site (or a currently-unexamined interaction) that consumes it before that
   `.contiguous()` fires.
4. **Unresolved, needs a hardware experiment, not a code candidate**: cross-block buffer-arena interaction
   under the whole-model single TinyJit capture (§6). Suggested experiment: `GDN_SCAN_IMPL=2
   GDN_HEAD_GROUPS=1` (if the model's real geometry allows forcing G=1) vs the auto G=2 — if forcing G=1
   changes the failure, that's strong evidence for a graph-complexity/collision effect over a purely
   WY-math-shape effect at a *fixed* group; if the failure persists identically, that argues *against*
   hypothesis §3.2 and further isolates the store-site asymmetry (§7.1) as the cause.

## 8. Candidates (separate commits, smallest-first; all in `tinygrad/llm/model.py`, all one-line
defensive `.contiguous()` insertions, all no-ops on CPU, all independently cherry-pickable since each
touches disjoint lines)

| Commit | SHA | What | Targets |
|---|---|---|---|
| `T4.73-candidate-1` | `9317feac8` | `.contiguous()` on `gdn_scan_wy`'s `final_state` before return | The value that becomes (or feeds, via `cat`) the recurrent-state store source, at its origin |
| `T4.73-candidate-2` | `4daa39a6d` | `.contiguous()` on `gdn_scan_wy`'s `out.transpose(1, 2)` before return | The literal bare-view return case (SPEC_NOTES.md §3's exact pattern) |
| `T4.73-candidate-3` | `17084ea43` | `.contiguous()` on `state` (in `_attention`, both the G≤1 and G>1 paths, one call site) immediately before `.cast(self.recurrent_state.dtype)` and the in-place `.store()` | The concrete LOOP-vs-WY asymmetry: `core`/`stacked` already got this treatment, `state` didn't |

Each commit is stacked on the previous (candidate-2's diff sits on top of candidate-1, candidate-3 on top of
both) but touches disjoint lines, so any one is cleanly `git cherry-pick`-able onto a plain
`GDN_SCAN_IMPL=2` base if the coordinator wants to A/B just one at a time. Every candidate keeps all four
gate test files green (`test_gdn_scan_parity.py`, `test_attention.py`, `test_spec_decode.py`,
`test_mtp_load.py`: 56 passed, 9 skipped, unchanged from baseline at every step, verified cumulatively after
each commit), plus mypy (`Success: no issues found in 220 source files`) and ruff (`All checks passed!`).

**Suggested hardware A/B order**: try candidate-3 alone first (cherry-pick just that commit onto a clean
`GDN_SCAN_IMPL=2` config) since it's the highest-confidence, most targeted fix for the exact reported
config; if the corruption persists, add candidates 1 and 2 (either together or one at a time — they're
independent lines); if it *still* persists with all three applied, the bug is not in this file's tensor
materialization at all, and the search should move to the `@function(precompile=True)`/`Ops.FUNCTION`
cross-block question in §6, or to the real-device allocator/copy-engine layer directly (outside model.py
entirely).

## 9. `extra/wy_alias_audit.py` — CPU-side detector, findings

Two parts (see the script's own docstring for full method/honesty notes):

- **Part A** (graph-shape, no scheduling): confirms empirically, on the actual `gdn_scan_wy` function, that
  `final_state.uop.has_buffer_identity()` and `out.transpose(1,2).uop.has_buffer_identity()` are both
  `False` — i.e. both are genuine lazy expressions the memory planner is free to fuse/reorder, not
  already-safe buffers. (`has_buffer_identity()` is the exact predicate `Tensor.contiguous()` itself checks
  to decide whether to insert a real materialization boundary or no-op — `tinygrad/mixin/elementwise.py`.)
  This is stable whether candidates 1/2 are applied or not (a `.contiguous()`-wrapped value is still
  `Ops.CONTIGUOUS`, which also reads `False` here — only the reported `op=` field changes).
- **Part B** ("under the JIT" buffer-identity tracking, the DELIVER-list ask): drives a real 48-head
  `GatedDeltaNetBlock` through actual `TinyJit`-wrapped prefill-shaped (T_pad=4) then decode-shaped (T_pad=1)
  calls — warmup/capture/replay, mirroring `engine/jit.py`'s own `cnt` 0/1/≥2 stages — once per impl.
  `engine/jit.py`'s `memory_plan_rewrite` is monkeypatched to record, per real capture, whether
  `self.recurrent_state`'s and `self.conv_state`'s own buffers were ever arena-reuse-eligible ("flagged as
  an intermediate"). **Result: CLEAR in every capture, both impls — both buffers were `held` (never
  plannable) every time.** This rules out "the state buffer's own identity gets stolen by the generic
  planner" (on CPU's planner logic, which — per §3 — is the same code that would run on METAL/NV). It does
  **not** clear METAL/NV: real allocator pooling and async completion-timing races are outside what this
  backend-agnostic Python pass can expose on *any* device, CPU included — that is precisely the mechanism
  §3 ranks as the leading explanation, and by nature it cannot be observed this way. Also reports the §2
  buffer-count delta (WY 50 vs LOOP 44 distinct schedule-visible buffers at first capture, same geometry,
  all three candidates applied) plus a kernel-count delta (WY 42 vs LOOP 36 kernels for the prefill
  capture); the decode capture that follows is byte-identical in both counts regardless of which impl the
  preceding prefill used (24 kernels/32 buffers either way) — expected, since T4.69b's gate always routes
  T_pad==1 to LOOP, and a nice internal sanity check that the harness reflects the model's real dispatch.

Honest bottom line for the detector: it found no smoking gun (and could not have, per its own stated
limits), but it did (a) empirically confirm the graph-shape claims candidates 1/2 rely on, and (b) rule out
one entire hypothesis class (generic-planner buffer-identity theft) with a real, running check rather than
by argument alone.

## 10. If none of this is it

A well-argued non-answer: the corruption site may simply not be visible to anything CPU can run, including
this detector — real GPU command-queue/copy-engine timing (§3.1) is the kind of bug that, by definition,
only a real device with real asynchronous execution can expose or confirm. The three candidates are the
correct-shaped, minimal, safe things to try first because they harden the *one* concrete asymmetry this
audit found in the graph itself; if the coordinator's hardware A/B shows no change with all three applied,
that itself is a valuable, informative result (it would redirect the search to `engine/realize.py`'s
linking/copy-engine layer, or to the specific METAL/NV backend's buffer-allocator implementation, neither of
which this task's scope reaches).
