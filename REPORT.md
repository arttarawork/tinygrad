# T4.76: Ops.FUNCTION / cross-capture analysis of the WY prefill -> decode garbage-token bug

Scope: source-level analysis only (this worktree touched CPU/NULL exclusively -- see the branch's own
constraints). Every claim below is either a direct citation (file:line, read and quoted/paraphrased) or
explicitly flagged as inference/unconfirmed. Deliverables A (`tinygrad/llm/model.py`'s `WY_TRACE`) and B
(`extra/wy_boundary_repro.py`) exist because several of the open questions below cannot be settled from
source alone -- see each one's "confirms" line.

Restating the bug (not re-litigated, taken as given): serving Qwen3.8-27B on METAL+NV with
`GDN_SCAN_IMPL=2` (WY), the FIRST decode step after WY prefill samples out-of-vocab id 248320 -- identical
across two prompt lengths, four builds/configs (including `GDN_HEAD_GROUPS=1`, no head-group split at all),
and BEAM=0. Zero device faults. CPU is clean, single-device and multi-device (`CPU:0,CPU:1`) alike, under
the same `IMPL=2`. Prefill itself looks correct (28 tok/s).

## 1. What `@function(precompile=True)` does to a block's subgraph inside the whole-model TinyJit capture

`FFNBlock.__call__` (`tinygrad/llm/model.py:391-395`) wraps a **freshly-defined closure** `_run(x, start_pos)`
with `@function(precompile=True, allow_implicit=True)` on every single call -- not once per block object, once
per *invocation*. That decorator is `_function` (`tinygrad/function.py:32-105`). Its `__call__`
(`function.py:43-90`):

1. Runs `self.fxn(*args, **kwargs)` (`function.py:56`) -- i.e. traces `_run`'s python body once, eagerly,
   with device usage disallowed unless `DEVICE_IN_FUNCTION_BUG=1` (`function.py:53`, the "nothing may touch a
   device" constraint `FFNBlock._init_state`'s own comment at `model.py:485-488` warns about).
2. Substitutes the two **explicit** args (`x`, `start_pos`) with `PARAM` placeholders (`function.py:67-68`).
3. Runs a bottom-up `graph_rewrite` with `pm_ctx` (`function.py:15-19`) that finds every remaining bare
   `Ops.BUFFER` (or an `AFTER`/`CONTIGUOUS` whose backward slice reaches a `BUFFER` but no `PARAM`) and
   promotes it to an **implicit** input, appending it to the same `call_uops` list (`add_to_ctx`,
   `function.py:9-13`, mutating `ctx[0]` in place). For `GatedDeltaNetBlock._attention`
   (`model.py:574-671`) this is exactly how `self.recurrent_state.uop`, `self.conv_state.uop`, and every
   block weight get captured -- bare `BUFFER` nodes, discovered and promoted here, one PARAM slot each.
   `allow_implicit=True` (set by `FFNBlock.__call__`) is what lets this succeed instead of raising
   (`function.py:75-79`).
4. Builds `uret.call(*call_uops, ...)` (`function.py:81-82`) -> `UOp.call` (`uop/ops.py:1208-1218`): a
   value-producing body becomes `Ops.FUNCTION` with `src = (TUPLE-wrapped body,) + call_uops` -- i.e. the
   **concrete UOp identity of every implicit capture, as of THIS trace**, is baked into `FUNCTION`'s own
   `src` tuple at the call site.

**Does `precompile=True` inline the body into the surrounding graph?** No -- `resolve_function`
(`tinygrad/schedule/prepare.py:86-103`) is the inlining path (`earliest_rewrites`,
`schedule/prepare.py:117-119`), and its very first line is `if c.arg.precompile: return None`
(`prepare.py:87`) -- i.e. it explicitly **skips** any `FUNCTION` node built with `precompile=True`, leaving it
un-inlined at that stage. Instead, `transform_precompiled_call` (`tinygrad/tensor.py:109-142`, wired in via
`pm_early_transform_tensor_graph` at `tensor.py:145-147`) rewrites it into an **opaque `Ops.CALL`**: it
allocates a fresh output buffer per call site (`outs = tuple(r.empty_like() for r in resolved)`,
`tensor.py:117`), rewrites the traced body into a `SINK` of stores into that buffer plus PARAM targets
(`tensor.py:120-132`), and produces `new_call = UOp(Ops.CALL, src=(fxn, *input_buffers, *outs), arg=c.arg)`
(`tensor.py:135`) -- `input_buffers` being exactly the explicit+implicit `call_uops` from step 3/4 above,
carried through unchanged (`tensor.py:112`). The recurrent-state STORE inside `_attention`
(`state_store = self.recurrent_state.uop.store(...)`, `model.py:670`, threaded into the return value via
`core = Tensor(stacked.contiguous().uop.after(state_store))`, `model.py:619` shows the same pattern for the
conv-carry) is not specially extracted here -- it rides along as an ordinary interior node of the ONE traced
return expression, the same way `Ops.AFTER` is respected anywhere else in this codebase.

**Does this SINK-bodied `CALL` get its own isolated buffer arena, or does it join the whole-model plan?**
It joins the whole plan. A `SINK`, once normally scheduled, becomes its own nested `LINEAR`; that nested
`LINEAR` gets **flattened** into the caller's `LINEAR` by `pm_flatten_linear`
(`tinygrad/engine/realize.py:236-239`), which is applied inside the *main* schedule-creation pipeline
(`tinygrad/schedule/__init__.py:117`, `... ])+pm_flatten_linear`) -- not just the `pm_validate`-only path.
By the time `memory_plan_rewrite` runs -- **once**, over the entire captured `big_linear`
(`tinygrad/engine/jit.py:281`, `linear = jit_lower(big_linear, held_bufs, input_buf_uops)` at line 292) -- a
block's internal kernels are ordinary entries in the *same flat list* as every other block's kernels and the
embedding/output projection, each with its own concrete buffer arguments visible to
`_collect_bufs`/`_can_plan` (`tinygrad/schedule/memory.py:7-16`).

**Answering the two sub-questions directly:**
- *Does each block's buffer planning get isolated?* No. `memory_plan_rewrite`'s arena reuse is scoped only
  by `(device, is-this-a-copy-buffer)` (`schedule/memory.py:37`, `_key`), never by "which block" or "which
  call site." `precompile=True`'s actual, intended benefit is a **compile-cache** hit (the same structural
  key hits `to_program_cache`, avoiding N-times recompilation across identical blocks/calls) plus skipping
  python-level retracing cost inside `_function.__call__` for that structure -- not memory isolation.
- *Do FUNCTION-internal intermediates participate in the whole-capture arena reuse?* Yes, once lowered to
  concrete kernels they are ordinary flattened `LINEAR` entries and are exactly as eligible for
  `memory_plan_rewrite`'s suballocation as anything else -- **except** whatever is in `held_bufs`
  (`engine/jit.py:291`, see SS2), which is the one thing that keeps `recurrent_state`/`conv_state`/`cache_kv`
  out of the reused arena in the first place.

## 2. Cross-capture buffer aliasing: what decode reads that prefill wrote, and what guarantees visibility

**Enumerated buffers**, all persistent, `hasattr`-gated-once-allocated Tensor attributes that are *never*
reassigned as a python attribute after first allocation (only mutated in place via `.uop.store(...)`):

| Buffer | Owner | Allocated | Written by prefill | Read by decode |
|---|---|---|---|---|
| `recurrent_state` | `GatedDeltaNetBlock` | `_init_state`, `model.py:688-689` | `model.py:670` (`state_store`) | `model.py:619` (`state = Tensor(self.recurrent_state.uop.after(conv_state_store))`) |
| `conv_state` | `GatedDeltaNetBlock` | `_init_state`, `model.py:680` | `model.py:601` (`conv_state_store`) | `model.py:592` (`win = win.after(win[...].store(conv_state.cast(...)...))`) |
| `cache_kv` / `cache_k` | `TransformerBlock` / `MLATransformerBlock` | `_init_state`, `model.py:489-492` / `543-544` | `model.py:447-448` / `529-530` | same lines, subsequent-position slice read |

(No other cross-call buffer exists for these block types; weights are read-only, never written after load.)

**Identity, at the python level, is not the weak link.** Since these attributes are never reassigned, a
FRESH trace of `_run` at decode time reads `self.recurrent_state.uop` and gets the *same* UOp/buffer identity
prefill's trace captured -- this is the identical mechanism `cache_kv` already relies on for ordinary
multi-step decode, exercised correctly by every existing chunked/sequential parity test and by the pooled
server's own continuation behavior. Source reading finds no bug in *this* part.

**Where the guarantee is thinner -- two distinct gaps, both real, neither provable from source alone:**

**(a) `held_bufs` is a snapshot, not an invariant.** `held_bufs = set(buffers) | {u for tref in
list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if u.op is Ops.BUFFER}`
(`engine/jit.py:291`) is computed **once**, at *that TinyJit's own* `cnt==1` capture moment
(`_TinyJit.__call__`, `engine/jit.py:266-297`), from whatever is reachable through the weakref registry
`all_tensors` *at that instant*. It is what keeps `_can_plan` (`schedule/memory.py:12-16`) from ever
aliasing `recurrent_state` into the reused arena. This is correct as long as the block object (hence its
Tensor attributes) is reachable at that moment -- true for the ordinary case (the live `Transformer`/
`LLMServer` holds it) -- but it is a best-effort, GC/weakref-timing-dependent snapshot, not a proof; source
reading cannot rule out a transient window (e.g. interaction with T4.67's `snapshot_state`/`restore_state`,
which *does* mutate these same tensors via `.assign()` -- checked: `Tensor.assign`,
`tensor.py:436-455`, uses the identical `self.uop.after(self.uop.store(x.uop))` idiom, i.e. it is
identity-preserving by construction, so this specific concern is **ruled out**, not open).

**(b) Cross-capture ordering/visibility on a real async device is enforced only by convention, not by an
explicit sync.** `CapturedJit.__call__` (`engine/jit.py:184-188`) calls `run_linear(self.linear, var_vals,
input_uops=concrete, jit=True)` with **no `wait=True`** -- `run_linear`'s own default is `wait=False`
(`engine/realize.py:318`). Dispatch is fire-and-forget from the host's perspective. What makes the NEXT
external call (decode) see prefill's finished state is: (i) prefill's OWN `.realize()` (`generate()`'s `out =
...realize()`, current text `model.py` `generate()`) forcing everything in `state_store`'s `.after()`-
dependency chain to be *scheduled* before whatever `.realize()` actually blocks on; (ii) the target device's
command queue executing in submission order thereafter. (i) is a property of the captured graph's own
topology -- verified structurally sound for the LOOP path (used identically by `cache_kv`, tested); *not
independently re-verified for WY's specific graph shape* by this task (CPU-clean is consistent with, but does
not prove, (i) holding for WY, since (ii) never runs on CPU the way it does on a real device -- see SS1's
CPU-vs-GPU scheduling difference and SS3 below). (ii) is unconditionally true only *within one device's own
queue*; prefill and decode are two **independently linked** (`CapturedJit.linear`, `engine/jit.py:169-170`)
programs, each built by its own capture, each individually passed through `graph_split_rewrite`
(`engine/jit.py:32-59`) -- a **GPU-only** hardware-command-graph batching pass, gated by `dev.graph is not
None` (`jit.py:48`; CPU structurally never takes it). Whether the two independently-graph-batched programs
(prefill's, decode's) are guaranteed ordered relative to each other on the SAME queue, or whether crossing a
device_map device boundary needs (and gets) an explicit fence/copy that WY's shape might route around, is a
scheduling/runtime property this task's source reading could not settle without instrumented hardware data
-- exactly deliverable A/B's job.

## 3. Ranked shortlist of mechanisms (max 3)

All three are consistent with every given fact (prompt-independence, same id across 4 configs, BEAM=0-clean,
CPU-clean including multi-device CPU). None could be confirmed or ruled out from source alone.

### #1 (top-ranked): decode's captured graph is a replay from an UNRELATED capture history, and the
real prefill it's chained after is a fresh, never-before-planned eager trace -- a lifecycle mismatch that
only a real async device (never CPU) can expose.

This is the most source-grounded candidate, built from a chain of confirmed facts, not speculation:

- Every `--serve` run calls `model.warmup()` unconditionally (`if args.warmup or args.serve: ...
  model.warmup()`, `tinygrad/llm/cli.py:193-194`).
- `Transformer.warmup()` (`model.py:1262-1265`) runs `generate([0], temperature=t)` -- a **1-token** dummy
  prompt -- twice per temperature.
- `Transformer.__call__`'s `is_prefill = bool(resolve(tokens.shape[1] != 1))` (`model.py:1070`).
  `resolve(x, default=True)` (`tinygrad/uop/ops.py:54-58`) returns the *concrete* boolean the moment
  `x.simplify()` degenerates to `vmin==vmax` -- true whenever the CURRENT bound chunk size is exactly 1,
  regardless of the underlying Variable's wider declared range. For a 1-token prompt, `generate()`'s own
  chunk loop (`n_toks = min(chunk_size, virtual_len - start_pos)`) always binds `nt` to 1 -- so
  `is_prefill` resolves **False** at every step of `warmup()`'s own dummy sequence.
- Therefore `warmup()` **only ever populates the decode jit key** `(False, True, None, False)` (and its
  sampled sibling) -- it never touches `(True, True, chunk_size, False)`, the key any real (>1-token) prompt's
  prefill actually uses. Confirmed independently by building deliverable B: `run_jit`'s own warmup phase, run
  with a 1-token dummy exactly like `Transformer.warmup()`, populates only the decode key
  (`extra/wy_boundary_repro.py`, `run_jit`'s docstring and `--warmup` path).
- Consequence for the real bug: by the time the real, bug-triggering prompt arrives, **decode's TinyJit is
  already at `cnt>=2`** -- every call to it is a pure replay (`_TinyJit.__call__`, `engine/jit.py:298-304`,
  no python of `forward()`/`_attention()` runs at all) of a graph captured (`held_bufs` snapshotted, memory
  planned, `graph_split_rewrite`'d) against warmup's own, unrelated 1-token history -- while the real prompt's
  own prefill key has **never been touched before**, so it is a fresh `cnt==0` "jit ignore" eager call
  (`engine/jit.py:260-265`) -- no capture, no memory plan, no hardware-graph batching at all for THIS
  specific call. The prefill-then-decode boundary the bug lives at is, mechanically, the very first moment
  these two independently-lifecycled TinyJit objects -- one totally fresh, one already replaying an
  already-cold capture from different data -- are chained together for the first time with real (not dummy)
  data flowing through the shared persistent buffers between them.
- This cleanly explains the CPU/GPU split too: `graph_split_rewrite`'s hardware-graph batching
  (`engine/jit.py:26-59`) is **only reachable when `dev.graph is not None`** (`jit.py:48,50`) -- true for
  METAL/NV, structurally false for CPU (checked: no CPU code path sets `.graph`). So "decode replays a
  pre-built hardware command graph immediately after an eager, unbatched prefill dispatch" is a call
  sequence CPU can never construct, regardless of anything else -- not "didn't happen to trigger," but
  "cannot happen on this device class."
- **Confirms via A/B:** on real hardware, `WY_TRACE=1` should show `step=prefill(last)` with a sane, finite
  per-block state fingerprint (a fresh eager trace, nothing exotic) immediately followed by `step=decode#1`
  already reading garbage -- i.e. the corruption is visible at the FIRST read, not built up over decode
  steps. For B: `extra/wy_boundary_repro.py --device NV/METAL` **without** `--warmup` (decode's key captured
  fresh, from the SAME real-data history as the prefill immediately before it) is predicted to PASS or fail
  differently than the **same command with `--warmup`** (decode pre-captured against unrelated dummy data,
  exactly mirroring production's `Transformer.warmup()`). A divergence that appears only with `--warmup` --
  or is markedly worse with it -- would be strong, first-of-its-kind confirmation of this exact mechanism.

### #2: WY's specific dependency-graph shape (heavier transpose/permute pattern feeding the same
`.after()`-chained state store) trips a `schedule/prepare.py` rewrite rule differently than the loop's
simpler per-step chain, corrupting the STORE's effective target/value at lowering time.

`gdn_scan_wy` (`model.py:83-127`) computes `final_state` via `state + u.transpose(-1,-2) @ k` from a
value (`state`) that itself derives from `self.recurrent_state.uop.after(conv_state_store)`
(`model.py:619`) through several transposes and matmuls the loop's per-step closure
(`model.py:626-668`) never constructs in that shape. `fix_store_hazard`
(`tinygrad/schedule/prepare.py:52-59`) exists precisely to detect "the store's own target is reachable
through an unsafe reordering op (PERMUTE/FLIP/certain SHRINKs) in its source" and insert a defensive
`.contiguous()` when it fires -- a real, general-purpose correctness rule whose trigger condition is a
structural property of the graph, and WY's shape is the most structurally different input this rule has
ever seen for this exact store. If it fires (or fails to fire) differently for WY than for the loop, in a
way that redirects `state_store`'s effective target away from the "real" `recurrent_state` buffer, or drops
part of the `.after()` chain during a rewrite tuned/tested only against the loop's simpler shape, this would
plausibly explain a WY-specific, deterministic (same garbage id, not a search artifact), buffer-identity-
flavored corruption localized to exactly this store.

**This does not by itself explain CPU-clean** -- `earliest_rewrites` runs identically on every device --
unless the actual trigger additionally depends on a device-specific downstream choice (a renderer/codegen
simplification, or the specific real-scale buffer sizes at 48 heads x head_dim=128, which the CPU parity
suite exercises at a different, smaller scale in places -- flagged, not confirmed). Ranked below #1 because
it requires an *additional*, unconfirmed device-dependent ingredient to fit CPU-clean at all, where #1's
CPU exemption is structural and unconditional.

**Confirms via A/B:** if this is the mechanism, `WY_TRACE`'s prefill(last) line should *already* show a bad
(NaN or wildly-scaled) `recurrent_state`/`conv_state` fingerprint right after prefill -- the corruption would
be in the STORE's own value/target, baked in before decode ever runs, not something decode's read introduces.
This is the single cleanest discriminator between #2 and #1/#3: **prefill's own fingerprint sane vs. already
bad** is the one-bit answer deliverable A was built to extract in one hardware run.

### #3: `held_bufs`/arena-protection has a real, narrow gap specific to the real 48-head/head_dim=128
scale that neither the CPU parity suite (smaller synthetic geometries in places) nor this task's own
CPU-only deliverable B (deliberately small head_dim, per the brief) exercises.

Source reading finds the *design* of `held_bufs` (SS2a) sound and finds no bug in its stated logic -- but it
is exactly the mechanism the task's own framing names ("cross-capture state store"), and this task's own
attempt to build a maximally faithful CPU repro (deliverable B) could not use the real head_dim (128) without
risking the CI-crash class the GATES explicitly warn about (`grouped-c32`/`c64`-class deep LOOP chains --
module docstring, `extra/wy_boundary_repro.py`), so the real-scale case remains, honestly, untested by
anything in this task. If a buffer larger than some threshold is handled differently by `TLSFAllocator`
(`schedule/memory.py:5,48`) or by a real device's allocator underneath it in a way that only manifests at
real-model scale, this task cannot rule it out. Ranked last because it requires an unconfirmed scale-dependent
bug in a mechanism whose written logic is otherwise sound, versus #1's fully-confirmed structural setup and
#2's at-least-plausible-without-new-assumptions trigger.

**Confirms via A/B:** `WY_TRACE`'s per-block fingerprint would show `recurrent_state`'s `amax`/`sum` at
decode-time resembling the scale of some OTHER tensor in the model (an aliased activation) rather than
NaN or its own expected small magnitude -- "structured wrong values," in the brief's own words, not a
nan-flood. Deliverable B run at `--head-dim 128 --heads 48` (the real scale, needing `--chunks`/`--chunk-size`
tuned down to stay CI-safe per the module's own warning, or run only on real hardware where the CPU-clang
crash class doesn't apply) would be the direct test -- not attempted by this task.

## Honesty: what source reading alone cannot settle

- Whether `held_bufs`'s snapshot timing ever actually misses `recurrent_state`/`conv_state` in a real
  server's object graph -- the *design* is sound; a live-process timing gap is not provable or disprovable
  from source.
- Whether `graph_split_rewrite`'s hardware-graph batching genuinely fails to order two independently-captured
  programs (prefill's, decode's) correctly on a real device queue, or whether device_map's cross-device
  boundary (if the affected GDN block sits on METAL with a downstream/upstream NV block, or vice versa) drops
  a needed fence specifically for WY's shape -- this is runtime/driver behavior, not a static graph property.
- Whether `fix_store_hazard` (or any other `earliest_rewrites` rule) actually fires differently for WY's
  graph than the loop's at the real 48-head/head_dim=128 scale -- would need to trace the actual rewritten
  graph for that exact geometry, which requires running it (this task did not, staying CPU/NULL-only, and the
  CPU parity suite's own geometries don't necessarily match in every dimension).
- Whether mechanism #1 (the warmup/history-mismatch) is sufficient on its own, or only a necessary
  precondition that additionally needs #2 or #3's graph/buffer specifics to actually corrupt data -- source
  reading establishes the STRUCTURAL SETUP (two mismatched TinyJit lifecycles meeting for the first time)
  with certainty, but not that this setup alone, absent a second ingredient, is sufficient to produce garbage.
  Deliverable B's `--warmup` flag is built specifically to start separating these on real hardware.

## Exact commands for hardware runs

**WY_TRACE=1 on the real serve line** (add `WY_TRACE=1` to the environment prefix already documented in
`~/CLAUDE.md`'s "Pooled model" `pooled-serve.sh` invocation, or export it in the shell before calling
`python -m tinygrad.llm` directly with `--serve`/`--mtp` as normal):

```
WY_TRACE=1 DEV=NV JITBEAM=2 PARALLEL=2 BEAM_TIMEOUT_SEC=30 GDN_CHUNK=32 GDN_SCAN_IMPL=2 \
  PYTHONPATH=. python -m tinygrad.llm -m <the Qwen3.8-27B GGUF> --device-map <the METAL+NV map> \
  --max_context <ctx> --serve 8081
```

Then drive one request with the repro prompt (20-tok or 23-tok) and read the `[WY_TRACE] ...` lines from
stdout/the server log -- one line for `step=prefill(last)` and one each for `step=decode#1/2/3`.

**`wy_boundary_repro.py` on METAL/NV** (never run by this task -- CPU-only was verified here):

```
DEV=METAL PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device METAL
DEV=NV    PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device NV
# ranked mechanism #1's direct test -- run both and compare:
DEV=NV    PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device NV
DEV=NV    PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device NV --warmup
```

If a `FIRST-DIVERGENCE` appears on hardware, sanity-check it against `extra/wy_boundary_repro.py`'s own
tolerance/seed note before concluding anything: the real bug's signature is a PROMPT-INDEPENDENT,
CONFIG-INDEPENDENT constant garbage id (not a seed-sensitive near-tie flip with both sides' values small and
plausible -- see below).

## Side-finding from building deliverable B (NOT the T4.76 bug -- flagged separately on purpose)

While calibrating deliverable B's tolerance, an unrelated CPU-visible artifact turned up: an untrained,
randomly-weighted tiny model occasionally shows a real (not ULP-level) argmax flip at a SPECIFIC decode step
when driven through TinyJit versus called eagerly, for reasons this task could not fully root-cause. It was
run down as far as scope allowed and **ruled out** as WY-related or capture-phase-related:

- Reproduces identically with `GDN_SCAN_IMPL` forced to `GDN_SCAN_LOOP` on both the "reference" and "jit"
  paths -- not WY-specific.
- Reproduces identically after explicitly warming both jit keys to `cnt>=2` first (i.e. with every compared
  call already a pure replay, no capture involved in the comparison) -- not capture/replay-phase-specific.
- Reproduces identically with a GDN-only stack (no attention block at all) -- not attention-related.
- Traced to ~1e-6-level state differences (the SAME order of magnitude `test_gdn_scan_parity.py` already
  documents as ordinary chunk-boundary float non-associativity) amplified roughly 10^5x through this
  UNTRAINED toy model's own unnormalized random weights -- consistent with, though not proven to be, the
  well-known sensitivity of small random (non-trained) recurrent/deep nets to tiny perturbations, which a
  real trained model's much larger, better-separated logits would never expose.

This is why `extra/wy_boundary_repro.py` pins `--seed 5` and `--rtol/--atol 1e-3` (`test_attention.py`'s own
`TestGatedDeltaNetBlock` convention) rather than a tighter default -- documented in full, with the specific
ruled-out checks listed, in the script's own module docstring and `run_jit`'s docstring. It is flagged here,
separately, because it touches the exact same "eager vs. TinyJit" territory as this report's mechanisms and
should not be confused with the real bug if it resurfaces on hardware: the real bug's id is constant across
prompts and configs; this artifact flips based on which near-tied token wins by a hair, with both sides'
logit values small and mutually plausible.
