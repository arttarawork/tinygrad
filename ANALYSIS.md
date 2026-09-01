# T4.74 — snapshot_state NV device-fault analysis

Status: analysis + CHECK_OOB harness + 3 hardening candidates, all CPU/NULL-verified. Cannot reproduce on
hardware (task scope forbids opening METAL/NV). Verdict below is a ranked, evidence-based elimination, not a
hardware-confirmed root cause — see "Hardware A/B sequence" for how to close the gap.

## 0. The fault, restated precisely

Ladder round C (T4.72, `td5` pinned at `295659c42`, `--state-cache` on, plain `generate()` loop, no WY/MTP):
the NV device faulted (`is_err_state`, "PCI bus-master now cleared", recoverable via fresh client —
`runtime/ops_nv.py:52-60`'s `_fault_recovery_hint`) during a 5.9k-token request, at the post-prefill boundary
— exactly where `Handler.run_model` (`tinygrad/llm/serve.py:133-137`) calls `self.server.store_snapshot(ids)`
on the first token `generate()` yields:

```python
for next_id in gen:
  if len(out) == 0:
    ...
    if self.server.state_cache_mb > 0: self.server.store_snapshot(ids)   # serve.py:137
```

`store_snapshot` (`serve.py:240-252`) calls `self.model.snapshot_state()` once per distinct `ids` tuple.
`Transformer.snapshot_state()` (`tinygrad/llm/model.py:1240`) clones every block's live state and realizes it
all in one call:

```python
for b in self.blk:
  if isinstance(b, GatedDeltaNetBlock): blocks.append({"conv_state": b.conv_state.clone(), "recurrent_state": b.recurrent_state.clone()})
  elif isinstance(b, MLATransformerBlock): blocks.append({"cache_k": b.cache_k[:, :, :pos, :].clone()})
  elif isinstance(b, TransformerBlock): blocks.append({"cache_kv": b.cache_kv[:, :, :, :pos, :].clone()})
  ...
Tensor.realize(*(t for bs in blocks for t in bs.values()))
```

Round A (same 5.9k prefill, `--state-cache` OFF) was clean. Round A and round C's `generate()` calls are
byte-identical up to the fault point — state-cache adds nothing to `generate()`/the forward pass itself, only
this one extra call after the first token. So whatever is wrong is intrinsic to `snapshot_state()`'s own added
device work, not an interaction the forward pass already had a chance to exercise safely. Real shapes (given):
B=1, n_kv_heads=4, head_dim=256, pos=5872, max_context=65536, fp16, 16 attention (`TransformerBlock`) blocks
on NV for the qwen3.8-27B pooled map.

## 1. Ranked hypotheses

### #1 (top-ranked): a burst of fresh, never-before-exercised allocation + kernel dispatch

`snapshot_state()` is architecturally unique in the serving path: it's the **only** place that issues a large
batch of brand-new device allocations and a kernel shape this session has never run, back-to-back, in one
non-JIT `Tensor.realize()` call, immediately after a heavy prefill.

Evidence, precise:

- **Shape audit** (`extra/snapshot_oob_audit.py`, real shapes): each attention block's clone is a fresh
  ~24.1 MB allocation (`(2,1,4,5872,256)` fp16 = 24,059,904 bytes; ×16 blocks ≈ 367 MB, matching the task's
  "~370 MB"). All 16 blocks' clone kernels share **one compiled AST** (`kernel identity across all 16 blocks:
  SAME AST object` — verified via UOp hash-consing identity, not just equal keys), so this is one compile, but
  **16 separate `Ops.CALL` dispatches and 16 separate fresh allocations**, all issued inside one
  `Tensor.realize()` with no draining in between (`run_linear`, `tinygrad/engine/realize.py:318-322`, iterates
  `linear.src` and dispatches each call with `wait=False` unless `DEBUG>=2` — nothing forces a wait between the
  16 calls).
- **Never warmed.** `Transformer.warmup()` (`model.py:1220-1223`) only exercises `generate()`'s own decode/
  prefill kernels (`UOp.variable`-bound `start_pos`/`toks`, so ONE compile serves every future position/length).
  `snapshot_state()`'s clone uses a **concrete** `pos` (`len(self._cached_tokens)`, a plain Python int, not a
  bound Variable) — so its output shape, and therefore its AST cache key, is different for every distinct
  context length a snapshot is ever taken at. `warmup()` never calls it. **The first live `store_snapshot()`
  call of a session is therefore this kernel's first-ever compile and first-ever execution on the real device**,
  mid-request, right after the JIT-graph-heavy prefill — this matches the fault's actual circumstance (T4.72's
  round C was a fresh-client session; the fault landed on essentially the first snapshot taken).
- Structural note, not yet a fix: because `pos` is baked in concretely, **almost every distinct request length
  under `--state-cache` re-triggers a fresh compile** of this kernel (a new prefix length → a new output shape
  → a new AST key) — this isn't a one-time startup cost, it recurs across a session as context grows. This
  reframes T4.72's own conclusion ("faults are not search-correlated at all; they are the T4.74 store path") as
  consistent with "novel-kernel-at-novel-shape" being the recurring trigger, not a static one-off.
- Investigated and **refuted**: fresh **BEAM search** specifically. `snapshot_state()`'s `Tensor.realize()` is
  an ordinary eager call (not inside a `@function(precompile=True)` JIT capture), so its kernel compiles via
  `run_linear` → `compile_linear(linear, validate=..., input_uops=...)` (`realize.py:318-320`) with no explicit
  `beam=` argument, meaning `beam_val = BEAM.value` (`realize.py:310`) — the **plain** `BEAM` ContextVar
  (`tinygrad/helpers.py:232`, default `0`). The pooled-serve ritual (`~/CLAUDE.md`) sets `JITBEAM=2`, not bare
  `BEAM=`; `JITBEAM` is read **only** inside the JIT-capture compile path (`tinygrad/engine/jit.py:72`:
  `compile_linear(linear, beam=getenv("JITBEAM", BEAM.value))`), which governs the model's own forward-pass
  kernels, never plain `.realize()` calls like this one. So under the documented launch config, `BEAM.value==0`
  here and `compile_linear`'s `if beam_val >= 1: graph_rewrite(linear, pm_beam, ...)` (`realize.py:310`) never
  fires — **no BEAM search runs for this kernel at all.** This means the historically-dominant fork fault
  mechanism (T4.29/34/47/48/49/50: a bad candidate hit *during a live BEAM search*; T4.53: the actual named
  culprit was a data-dependent MoE `weight[sel]` gather, unrelated to this code) does **not** directly apply
  here in its documented, precise form. What remains is the weaker but still real claim: a first-ever compile
  *of the plain heuristic kernel* for a large, high-rank (5D), strided-source copy — a shape nothing else in
  this codebase generates (see §3) — landing on a device already stressed by the JITBEAM-searched prefill work
  that just preceded it, inside a burst of 16 such dispatches plus fresh allocations. This is necessarily a
  process-of-elimination conclusion, not a proven mechanism: it's what's left once (a) is refuted by CHECK_OOB
  and (c) is weakened by the drain argument below, not something I can watch fault on this machine.

### #2: interaction with the (allegedly) in-flight generation stream — weakened, not eliminated

Task hypothesis (c). `generate()`'s drain loop (`model.py`, `generate()`'s per-step `pending`/`drain_every`
block) only yields a token after `Tensor.cat(*pending, dim=1).tolist()` — a **host readback**, which cannot
return before the underlying device write completes. `Handler.run_model` calls `model.generate(ids,
temperature=temperature)` with no `drain_every=` override (`serve.py:129`), so `drain_every=1` (the default):
every single decode/prefill step is individually drained (synced) before `generate()` ever yields. By the time
`serve.py`'s `for next_id in gen:` loop body runs `store_snapshot(ids)` (`serve.py:137`), the Python generator
`gen` is **suspended** (no further device dispatch happens until it's resumed) and the *last* step's token was
already read back to host — which, on a strictly in-order HCQ queue, implies everything queued before that
readback's dependency chain has completed. Two things keep this from being a clean refutation: (1) I can't
verify from source alone whether NV's readback wait is a full queue drain or a narrower per-buffer wait (that's
in `runtime/ops_nv.py`'s HCQ signal plumbing — `NVDevice`/`NVComputeQueue`/`NVCopyQueue`, `ops_nv.py:106-268` —
skimmed, not executed, per the task's hard rule; the signal/timeline-value scheme *looks* like a monotonic
per-queue counter that should make this safe, but "looks safe from reading" is not "verified on hardware");
(2) this argument is specific to the plain `generate()` path that actually faulted (round C) — it does not
extend to `speculative_generate()` without separate checking (out of scope here; round D showed MTP clean
anyway). Net: demoted from the task's original framing, but candidate 1 below is cheap enough to ship as
insurance regardless of how this resolves.

### #3: OOB-class access pattern — refuted by CHECK_OOB, with a precise reason it doesn't just mean "safe"

Task hypothesis (a). **CHECK_OOB (z3-backed) proves every LOAD/STORE in `snapshot_state()`'s and
`restore_state()`'s clone/assign graphs in-bounds, at the exact real shapes** (§2). Beyond the empirical
result, there's a structural reason this was always likely to hold: `pos` is a concrete Python int, not a
`UOp.variable`-bound symbolic value — there are **no free variables at all** in the index expressions
`type_verify`'s `validate_index` (`tinygrad/uop/spec.py:10-33`) has to reason about, so the *fast path*
(`0<=idx.vmin and idx.vmax<sz`, pure interval arithmetic) almost certainly settles it without ever invoking z3
— a strictly easier proof obligation than the model's own everyday decode-step slicing (`start_pos:UOp`-bound,
genuinely symbolic, and *that* already passes CHECK_OOB in production). This is also categorically different
from this fork's actual precedent for an "OOB-class" fault: T4.53 named the real culprit as the MoE expert
router's `weight[sel]` — a **data-dependent gather** (`sel` is a runtime tensor value, not a static bound).
z3 can still analyze such gathers (via a loaded value's inferred `[vmin,vmax]`), but the proof obligation is
fundamentally weaker there (it depends on whatever range the *producing* computation is known to stay within,
not a static shape fact) — and a gather's hardware access pattern (scattered, data-dependent addresses) is a
different animal from a strided-but-fully-static copy. `cache_kv[:, :, :, :pos, :]` is a `SHRINK` (confirmed:
`extra/snapshot_oob_audit.py`'s `classify()` finds a `Ops.SHRINK` node on the index path), not a gather — and
the exact same *shape family* (a growing-prefix shrink of `cache_kv` along the position axis) is already
exercised on **every single forward pass, every layer, every token** via `assigned_kv[0,:,:,0:start_pos+T,:]`
(`model.py`, `TransformerBlock._attention`) — if this access pattern itself were fault-prone, ordinary
generation would fault constantly, not just the snapshot path. What IS genuinely novel is not the *logical*
access pattern but the *kernel shape* built purely around it (see #1) — CHECK_OOB proves the math is right; it
cannot prove a backend codegen bug for a kernel shape nothing else in the codebase generates. That residual
risk is folded into hypothesis #1, not kept as a separate "OOB" claim.

## 2. CHECK_OOB verdict (from `extra/snapshot_oob_audit.py`)

Run both ways; identical verdict:

```
$ CHECK_OOB=1 DEV=NULL PYTHONPATH=. .venv/bin/python extra/snapshot_oob_audit.py
$ CHECK_OOB=1 DEV=CPU  PYTHONPATH=. .venv/bin/python extra/snapshot_oob_audit.py   # real execution, 3.3s wall
```

```
full cache_kv shape=(2, 1, 4, 65536, 256)  per-block clone shape=(2, 1, 4, 5872, 256)  dtype=dtypes.half  (24.1 MB/block)

--- single attention-block clone (snapshot_state's per-block op): 1 CALL(s) ---
  [0] kernel-shaped (SINK, needs codegen)

--- ALL 16 attention blocks, ONE batched realize (production's exact idiom): 16 CALL(s) ---
  [0..15] kernel-shaped (SINK, needs codegen)
  kernel identity across all 16 blocks: SAME AST object -- ONE compile/BEAM search serves every block
PASS  model.snapshot_state() (16 attention blocks, real shape, pos=5872): CHECK_OOB proved every LOAD/STORE in bounds (z3-backed, real shapes)
PASS  model.restore_state(snap) (slice-assign mirror of #3): CHECK_OOB proved every LOAD/STORE in bounds (z3-backed, real shapes)

CHECKPOINT conv_state shape=(1, 3, 64)  recurrent_state shape=(1, 4, 8, 8)  (both position-independent, O(1) in max_context)

--- speculative_generate's CHECKPOINT (GDN conv/recurrent whole-buffer clone): 2 CALL(s) ---
  [0] kernel-shaped (SINK, needs codegen)
  [1] kernel-shaped (SINK, needs codegen)
PASS  CHECKPOINT clone realize: CHECK_OOB proved every LOAD/STORE in bounds (z3-backed, real shapes)

ALL CHECKS PASSED
```

Mechanism note (for anyone re-verifying this): `CHECK_OOB=1` alone is sufficient with `SPEC` at its default
(`SPEC=1`) — `type_verify` runs automatically inside the normal scheduling pipeline whenever `SPEC` is truthy
(`tinygrad/schedule/__init__.py:133`: `if SPEC: type_verify(function, spec_tensor)`; also
`tinygrad/codegen/__init__.py:289,394`), and `validate_index` only does the real bounds check (fast-path or
z3) when `CHECK_OOB` is also set (`uop/spec.py:16`: `if not CHECK_OOB or is_image_shape(...): return True`).
No `SPEC=2` needed for this (that only adds a stricter *dtype*-consistency check on every UOp construction,
`uop/ops.py:196-198`, orthogonal to bounds-checking) — the harness confirms both give the same PASS.

A side discovery while building the harness, worth recording: scheduling the **same Python `Tensor` object**
through two separate `schedule_linear()`/`realize()` calls in one process can misreport the CALL count on the
second call (an apparent "fusion" that a clean N=1..16 sweep, each built and scheduled exactly once, showed was
not real — every N gave exactly N separate CALLs). The harness's methodology (fresh clones, scheduled once)
avoids this; flagging it in case anyone else builds a similar audit script on this codebase.

## 3. Key differences vs. `speculative_generate`'s CHECKPOINT (the non-faulting analog)

`speculative_generate`'s GDN checkpoint (`model.py:1536-1540`, comment "(b) CHECKPOINT"):

```python
gdn_snap = [(b, b.conv_state.clone(), b.recurrent_state.clone()) for b in gdn_blocks]
if gdn_snap: Tensor.realize(*(s for _, c, r in gdn_snap for s in (c, r)))
```

This idiom has run live on hardware repeatedly (T4.72 round D: MTP clean, zero faults) without incident.
Differences from `snapshot_state`'s attention-block clone, all confirmed by the harness:

| | CHECKPOINT (GDN) | `snapshot_state` (attention) |
|---|---|---|
| Slicing | **None** — whole buffer, every time | `[:, :, :, :pos, :]` — a `SHRINK` view |
| Size @ real shapes | tiny, fixed (e.g. `(1,3,64)` / `(1,4,8,8)` in the harness's synthetic GDN config; O(1) in `max_context` by construction) | ~24.1 MB/block, ×16 ≈ 367 MB |
| Shape stability | **Position-independent** — same shape (hence same AST/cache key) at every call, for the life of the process | **Position-dependent** — a new `pos` is a new shape, hence a new AST, on essentially every distinct request length |
| Warmed? | No (`warmup()` doesn't call `speculative_generate` either), but its shape never changes, so ANY first real `--mtp` request compiles it once, permanently | No, and shape-instability means it's effectively never fully warmed — recurs per distinct length |
| Batch size in one realize | 1 GDN block × 2 tensors in the observed 3.8/GDN mix (small; scales with `ssm_layers` count) | All 16 attention blocks in one call |
| Schedules as | `Ops.SINK` (kernel), confirmed — **not** `Ops.COPY` even for this trivial whole-buffer case | `Ops.SINK` (kernel), same op-class as CHECKPOINT |

The last row matters: I initially expected a same-device whole-buffer `.clone()` might use the cheap
`Ops.COPY` buffer-transfer path (`engine/realize.py`'s `exec_copy`) instead of compute-kernel codegen. Verified
empirically that it does **not** — same-device `.clone()` always goes through kernel codegen in this tinygrad
version, regardless of contiguity. `Ops.COPY` only appears for a **cross-device** transfer (verified: a
per-plane, fully-concrete-outer-index `[:pos, :]` sub-slice — contiguous once the leading axes are fixed to
concrete ints — schedules as `Ops.COPY` when moved via `.to()` to a *different* device, with zero kernel
involved). This is exactly what makes candidate 3 (below) work: it's not "avoid touching a SHRINK", it's
"avoid the same-device compute-kernel path entirely by forcing a genuinely cross-device transfer of
already-contiguous sub-pieces." The task's other suggested variant, "contiguous-staged clones"
(`.contiguous()` before `.clone()`), was evaluated and **not** implemented: `.contiguous()` on a non-trivial
same-device view still needs a kernel to materialize (same conclusion as above), so it would not change the
kernel-codegen risk profile at all — same op class, same shape, just a different Python spelling. That's why
candidate 3 is a real cross-device round-trip, not a `.contiguous()` shim.

## 4. Hardening candidates (separate commits, smallest-first)

All three keep `restore_state()` untouched (T4.72: "the snapshot RESTORE worked even mid-fault-era" — no
evidence it needs hardening, and it's a different op — an `ASSIGN` into a pre-existing live buffer, not a
fresh allocation — so it doesn't share hypothesis #1's mechanism).

- **`7e2600b0c`** — `extra/snapshot_oob_audit.py` (the harness itself; not a hardening patch).
- **`156c1e51b` — T4.74-candidate-1: sync device(s) before `snapshot_state`'s clone burst.**
  One `Device[dev].synchronize()` per distinct device, before the clone loop. Addresses hypothesis #2 as cheap
  insurance regardless of how the drain-order question above resolves; a no-op on an already-idle queue.
  Zero behavior change to the snapshot's contents.
- **`b9afb832e` — T4.74-candidate-2: per-block realize instead of one batched realize.**
  `Tensor.realize(*blocks[-1].values())` moved inside the per-block loop instead of one combined call at the
  end. Addresses hypothesis #1 directly: bounds the fresh-allocation/fresh-dispatch burst to one block
  (~24 MB) instead of all 16 (~367 MB) at once, and — if a hardware fault recurs — pinpoints which block's
  clone it lands on (useful bisection data even if this exact fix doesn't hold). Still exactly one `.clone()`
  per block; only the realize granularity changes.
- **`d1681cdce` — T4.74-candidate-3: opt-in host-routed copy for the attention clone (`SNAPSHOT_VIA_COPY`,
  default 0/off, byte-identical when off).**
  Routes the attention-block clone through `2*B*n_kv_heads` (=8 at the real shapes) per-plane
  `.to("CPU").to(dev)` round-trips instead of the single strided compute kernel — eliminating that kernel's
  first-ever live compile+execution from the picture entirely (§3). This is the most direct test of hypothesis
  #1's "novel kernel shape" facet, at a real but modest cost (~367 MB over a host round-trip per snapshot;
  cheap next to the multi-second prefill that already ran — the pooled dock's own measured D2H numbers put
  this around ~0.1-0.2s). Deliberately opt-in: this is a real performance trade, unlike candidates 1-2, and
  should only ship if the hardware A/B below actually implicates the kernel path.

Gates run after **every** candidate (all green): `CHECK_OOB=1 DEV=CPU pytest test/unit/test_state_cache.py
test/unit/test_spec_decode.py test/unit/test_mtp_load.py` (24 passed for candidates 1-2, 25 for candidate 3 —
one new test added), `test/unit/test_llm_device_map.py test/unit/test_llm_server.py
test/unit/test_llm_sampling.py` (60 passed / 7 skipped), `SPEC=2 DEV=NULL pytest test/null/` (1532 passed / 81
skipped / 16 xfailed), `mypy tinygrad/` ("Success: no issues found in 219 source files"), `ruff check .` ("All
checks passed!"). `extra/snapshot_oob_audit.py` re-run and green after candidates 2 and 3 (its schedule/CHECK_OOB
assertions are agnostic to realize granularity and to `SNAPSHOT_VIA_COPY`, so it doubles as a regression check
for both).

## 5. Hardware A/B sequence for the coordinator

Cheapest-information-first, each a fresh-client short probe on the exact round-C repro (5.9k-token prefill,
`--state-cache` on, plain loop decode, no WY/MTP — `td5` pinned at `295659c42` reproduces the stack; the
candidates are on top of fork master `07e6d67cf`, so re-pin/rebase `td5` onto a candidate commit for the test):

1. **Baseline re-confirm** (no candidate): re-run round C exactly as T4.72 did, on `07e6d67cf` with no T4.74
   commits, to confirm the fault still reproduces on the current master before attributing anything to a fix
   (T4.72's repro was on the older `295659c42`; master has since gained T4.55-71). If it does NOT reproduce
   here, something in T4.55-71 already changed the picture and the remaining steps need re-scoping first.
2. **`156c1e51b` alone** (candidate 1, sync). If this alone fixes it: hypothesis #2 was real despite the
   drain-order argument in §1 — the async-decode-race framing was right, or NV's per-buffer wait genuinely
   isn't a full drain. Cheap enough to just keep regardless of outcome.
3. **`b9afb832e`** (candidates 1+2, per-block realize). If this fixes it (and #2 alone didn't): hypothesis #1's
   burst-size framing was right — the fault is about *how much* fresh allocation/dispatch lands on the queue at
   once, not the kernel shape itself. Try shrinking further (e.g. group size < 16 but > 1) only if useful for
   a perf/safety tradeoff; per-block is already the smallest useful grouping.
4. **`d1681cdce` with `SNAPSHOT_VIA_COPY=1`** (candidates 1+2+3). If THIS is what it takes: the kernel shape
   itself (or its first-ever compile) was the actual trigger, not just burst size — §1's "never-warmed novel
   kernel" mechanism, most directly confirmed. This is the strongest, most specific result and would justify
   keeping candidate 3 as the standing state-cache default (flip `SNAPSHOT_VIA_COPY`'s default to 1) rather
   than leaving it opt-in.
5. **If none of 2-4 fix it**: the mechanism is something this analysis didn't identify from source alone (most
   likely candidate: the NV HCQ signal/queue semantics genuinely don't guarantee what §1's "#2 weakened"
   argument assumes, in a way no amount of code-reading settles) — collect `BEAM_LAUNCH_LOG`/fault evidence the
   way T4.50/53 did and treat it as its own bisection task rather than iterating further on these three.

Each step isolates one variable relative to the last (1 → 1+2 → 1+2+3), so a fix appearing at step N and
persisting through later steps is attributable to what step N added, not to cumulative changes.
