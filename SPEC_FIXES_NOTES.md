# T4.66d — fixing the two measured fixed costs in MTP speculative decoding

Follow-up to SPEC_PROFILE_NOTES.md (T4.66c's read-the-code audit). That audit ranked candidates from source
alone, with no hardware access; this task started from a REAL hardware run (Qwen3.8-27B pooled METAL+NV, k=3,
steady-state `SPEC_TRACE` lines) that confirmed candidate #1 (the draft embed hop) and exposed a cost
`SPEC_TRACE` labelled `state_assign_ms` (T4.66b's REDO-free FIXUP) that the audit had reasoned should be
dispatch-only/near-zero but that measured ~19,500ms on every partial accept. Both fixes below are implemented,
tested (CPU/NULL only, no hardware access from this worktree either), and committed on
`task/T4.66d-spec-fixes`.

Real model numbers used throughout this doc (read from `/Users/artur/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf`'s
header only — `_parse_header`, no tensor data staged, same technique `test_mtp_load.py::TestMTPLoadRealMetadata`
uses — this is a plain file read, no `DEV=METAL/NV`, consistent with the CPU/NULL-only grant):

| field | value |
|---|---|
| `vocab_size` | 248,320 (matches the task's "248k" figure almost exactly) |
| `dim` (embedding_length) | 5,120 |
| real transformer blocks | 64 (+1 nextn/MTP block) |
| `full_attention_interval` | 4 → 16 full-attention blocks, **48 GatedDeltaNet blocks** (matches the task's "48 GDN blocks" exactly) |
| `token_embd.weight` / `output.weight` | ggml_type=8 = **Q8_0** (34 bytes / 32 elements = 1.0625 bytes/element), shape (vocab, dim) |
| ssm config | conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=6144 → head_k_dim=head_v_dim=128, num_v_heads=48, num_k_heads=16 |

## Deliverable 1 — the state-assign fix (19,500ms → ~1,800ms on partial accept)

### What the cost was

`speculative_generate`'s FIXUP step (`m < k_eff`) built 48×2=96 `.assign()` nodes — one `recurrent_state`
and one `conv_state` write per GDN block — sourced from `state_track[:, :, m]` / `conv_window[:, m+1:...]`,
slices of the per-position captures `GatedDeltaNetBlock._attention(capture=True)` produces, then dispatched
all 96 in one `Tensor.realize(*assigns)` call.

Reading `engine/realize.py`: `Tensor.realize()` always calls `run_linear(..., wait=False)` (never threads a
`wait=True` unless `DEBUG>=2`), and `run_linear`'s `ctx.wait` flows into `exec_kernel`'s `rt(..., wait=ctx.wait)`
— so on paper this dispatch is async, exactly as T4.66c's audit assumed (SS(d): "this call SCHEDULES the fixup
and returns without blocking"). FIXUP's assigns are also confirmed device-LOCAL (same-device source and
destination — `state_track`/`conv_window` are produced on the same device as their owning block), so they lower
to plain elementwise kernels (`exec_kernel`), never the cross-device `exec_copy` path candidate #1 indicts
(`dest.device.split(":")[0] == src.device.split(":")[0]` is true here). By the numbers alone this should be
cheap: each assign moves ~3 MiB (the `recurrent_state` slice, `(1,48,128,128)` fp32) or less (`conv_state`,
~120 KiB) — at ~400ms/block-pair that's a laughably slow ~7.5 KB/ms, ruling out bandwidth as the driver. The
measured 19.5s can only be dispatch/sync overhead, not data volume — confirming T4.66c's own framing ("the
phase is labeled dispatch-only, so the dispatch itself is blocking") as the right diagnosis, even though the
exact HCQ-level mechanism (why *this* same-device assign blocks when a plain same-device `.clone()`/`.assign()`
elsewhere in this same file does not — see below) needs a real hardware profiler this CPU/NULL-only worktree
doesn't have to pin further.

One structural fact IS confirmed from source and matters for the fix: `state_track` is not a small tensor.
`capture=True` forces the per-token loop and retains state at all `T_pad` (=32, `gdn_chunk_for`'s default)
padded positions, not just the ~4 real ones (SPEC_PROFILE_NOTES SS3(c), the "T_pad tax"). Per GDN block that's
`state_track`: `(B=1, H=48, T_pad=32, V=128, K=128)` = 25,165,824 fp32 elements ≈ **96 MiB**, materialized via a
32-way `.stack()` — **on every single VERIFY call**, not just ones that end up partially accepted. Across all
48 GDN blocks that's **≈4.5 GiB of extra tensor construction per VERIFY call** (`conv_window`'s own capture
adds another ~33 MiB total, negligible by comparison). This is a real, over-and-above-what-T4.66c-flagged cost
that removing `capture` (below) eliminates as a side effect.

### Options considered

**(A) Batch harder.** The code already IS one batched `Tensor.realize()` call for all 96 assigns — that's
already "option A" in its literal form. Batching the *Python call* doesn't reduce the *kernel/dispatch count*:
`run_linear`'s `for call in linear.src: ...` still iterates and dispatches each of the 96 `CALL` nodes
individually (elementwise ops on 96 different destination buffers don't fuse — there's no shared computation to
fuse across unrelated buffers). The only way to actually cut the dispatch count below 96 would be restructuring
GDN state storage from 48 separate per-block buffers into one unified multi-block buffer (one gather + one
store instead of 96) — a real model-architecture change (touches `GatedDeltaNetBlock`'s per-block
`conv_state`/`recurrent_state` fields, `snapshot_state`/`restore_state`, every block-indexed access site) that
crosses into exactly the "tinygrad-core / architecture change" territory this task's STOP condition names.
Documented here, not implemented.

**(B) Revert to CHECKPOINT + REDO for partial accepts.** Chosen. Pre-T4.66b's design: clone every GDN block's
`conv_state`/`recurrent_state` (native-sized, NOT the T_pad-inflated capture stack) *before* VERIFY runs;
on partial accept, restore the clones and redo exactly the `m+1` confirmed-correct tokens as one small extra
forward (`spec=True`, same jit key as VERIFY — never a fresh capture). Given directly in the task: this measured
~1.8s total on real hardware vs. 19.5s for (A)'s already-batched form — a ~10x difference neither a CPU dispatch
count nor the async-dispatch reasoning above predicted, which is exactly T4.66c's own point about candidate #1
generalized: **the CPU/NULL dispatch-count proxy that motivated T4.66b's original change is structurally blind
to same-device sync costs on real HCQ hardware, the same way it was blind to cross-device copy costs.**
T4.66b's own commit message measured "25→13 spec forwards, zero REDO" on a CPU/mock harness as unqualified
progress; that metric was real but insufficient — it counted dispatches, not what each dispatch costs on the
actual multi-device stack.

Why (B) beats (A) even though (A) requires zero *additional* code: (A) is already implemented and still costs
19.5s. There is no "try harder at batching" lever left within the current state-storage layout — the 96-kernel
count is a floor, not a chosen inefficiency. (B)'s one extra forward is a completely different shape of work
(one more `TransformerBlock`/`GatedDeltaNetBlock` stack pass over `m+1≤k` tokens, sharing VERIFY's own warm jit
capture) instead of 96 small per-buffer dispatches, and it's independently, directly measured to be ~10x
cheaper on the actual target hardware. Per the task's own framing, that measurement wins over the CPU-only
reasoning.

### What changed (`tinygrad/llm/model.py`)

- `Transformer.forward`: dropped the `isinstance(block, GatedDeltaNetBlock) and spec` special case and the
  `gdn_extra` list — every block (GDN or not) now takes the plain 2-arg call again, and `spec=True` returns
  `(logits, hidden)` — a 2-tuple, like pre-T4.66b — instead of `(logits, hidden, *gdn_extra)`. This is also what
  recovers the ~4.5 GiB/call capture-stack construction and the "always loop, never WY" restriction on VERIFY's
  own GDN scan (T4.66c's candidate #2's "forced loop" half — not its "T_pad=32 vs ~4 real tokens" width half,
  which is a separate, still-open lever tied to `chunk_size`/`v_toks`'s shared-Variable design, out of this
  task's scope).
- `GatedDeltaNetBlock._attention`/`__call__`'s own `capture`/`spec` parameters are **untouched** — "keep the
  capture path only where it's free" per the task: the mechanism stays, tested directly by
  `test_gdn_scan_parity.py::TestGDNScanCapture` (which calls `_attention(..., capture=True)` directly, never
  through `forward()`), just unreached from the hot path now. Available for a future, better-designed use.
- `speculative_generate`: CHECKPOINT (`gdn_snap = [(b, b.conv_state.clone(), b.recurrent_state.clone()) ...]`,
  one batched `Tensor.realize()`) runs unconditionally right before VERIFY, timed inside `verify_dispatch_ms`
  (chronologically honest — it really does happen before VERIFY, and like that phase's own buf/bind setup it's
  real host CPU + dispatch-only device work). On `m == k_eff` (full accept): unchanged, `h_last =
  verify_h[:, m:m+1]` — T4.66b's per-position hidden capture is free and orthogonal to the expensive part, so
  it stays; no FIXUP. On `m < k_eff` (partial accept): restore `gdn_snap`, redo `chunk_ids[:m+1]` as one more
  `spec=True` forward, take `h_last` from its own last position. Timed under the existing `state_assign_ms`
  label (unchanged name — SPEC_TRACE format stays stable for the hardware invocation below).

### Why h_last never needed a REDO (only GDN state does)

Worth stating explicitly since it shapes the fix: attention-based hidden state at position `m` only ever
depends on tokens `0..m` (causal masking) — it's unaffected by whatever wrong draft tokens follow at
`m+1..k_eff`, on *any* accept length, full or partial. That's true independent of GDN capture and is why
`verify_h[:, m:m+1]` staying correct on partial accept was never in question — the REDO exists **only** to
rebuild GDN's `conv_state`/`recurrent_state`, which (unlike attention KV, or unlike per-position hidden) are
O(1) read-modify-write accumulators with no position-indexed slice to fall back to. The REDO's own `h_last`
output is used anyway (simpler, matches the proven pre-T4.66b shape exactly) rather than trusting
`verify_h[:, m:m+1]` on the partial-accept branch too — either is correct; this just changes nothing rather
than introducing a new code path to reason about for a phase that's about to be recomputed anyway.

## Deliverable 2 — the draft embed hop (~850ms → predicted tens of ms for k=3)

### What the cost was

`MTPHead.draft`: `owner.token_embd(tok_ids.to(owner.token_embd.weight.device)).float().to(dev)`. `tok_ids` is
always produced on `dev` (= `self.block.device` = `dmap[-1]`, where `from_gguf`'s MTP branch places ALL of
`mtp_head` — its source is either `tok_last`, sliced from `verify_ids_tensor` which comes from `owner.output`
(fixed at `dmap[-1]`), or this same method's own previous return, `owner.output(...)`, also `dmap[-1]`) while
`owner.token_embd` lives on `dmap[0]` (`Transformer.__init__` places it on the first block's device). So every
call pays: `tok_ids` hops `dmap[-1]→dmap[0]`, the embedded vector hops back `dmap[0]→dmap[-1]`. Two real
cross-device syncs, `k_eff` times per iteration — SPEC_PROFILE_NOTES.md's own leading candidate (#1), and the
task's hardware run confirms it: ~850ms/3 calls ≈ 283ms/call.

### Options evaluated (with real byte sizes)

**Move `mtp_head` to `owner.token_embd`'s device instead** — disqualified on a correctness-of-approach
argument, not just cost: `tok_ids`'s home device is pinned by **`owner.output`** (the main model's output
layer, fixed at `dmap[-1]` by `Transformer.__init__`, never moved by this fix — moving it would hurt the main
model's own forward), not by `mtp_head`. Moving `mtp_head` to `dmap[0]` does nothing to `tok_ids`'s origin — it
would still need to hop `dmap[-1]→dmap[0]` for the lookup exactly as today, and now the *output* step
(`shared_head_norm(x).to(owner.output.weight.device)`, currently a no-op since mtp_head and output share a
device) becomes a **new** real hop of the same tiny `(B,1,dim)` shape. Net: same hop count, just relocated —
this option doesn't achieve "zero cross-device copies" at all, independent of any memory argument.

**Copy the embedding table onto `mtp_head`'s device (chosen).** Byte cost, from the real header:
`vocab_size × dim` = 248,320 × 5,120 = 1,271,398,400 elements. `token_embd.weight` is resident at Q8_0 (34
bytes / 32 elements = 1.0625 bytes/element, confirmed via `ggml_type=8` in the real GGUF header) — **not** a
dequantized fp32 table (this fork never keeps a full-precision copy of a quantized tensor resident; dequant is
fused into the consuming kernel). Full copy: 1,271,398,400 / 32 × 34 = 1,350,860,800 bytes ≈ **1.26 GiB**
(1.35 GB decimal) — exactly the same size as `output.weight` (identical shape, identical quant), which the
target device (`dmap[-1]`) already hosts today. Against CLAUDE.md's own "~3.4 GB headroom on NV at 192k
context" this is real (≈37% of remaining headroom) but bounded and known, not "the whole embedding table
blows the budget" — a `.to()` on an already-quantized resident tensor replicates it byte-for-byte, so there's
no new precision decision to make either. **This wins**: it's the only option that actually removes both hops
(once `tok_ids`'s home device and the lookup device are the same, no hop is needed on either side), it's a
one-time, lazily-paid cost (a model that never drafts never allocates it), and the ~1.26 GiB is small next to
the ~283ms/call × k_eff removed on *every single iteration* of a session (unlike Deliverable 1's fix, which
only pays off on the ~1-in-4-ish partial-accept iterations per the hardware fact's own 43/53 full-chain ratio).

**Documented, not implemented** (ponytail: mark the ceiling, not build for it speculatively): if a future
config ever makes the full copy genuinely too much (a much wider `dim`, an unquantized embedding table, or a
context length that already leaves less headroom — CLAUDE.md's own 256k map is called out as "near the
METAL/unified-RAM edge"), the escape hatch is a lower-precision cast of the local copy (e.g. force fp16/int8
regardless of the source dtype) rather than a dynamic per-id remote gather — the bytes actually moved per call
are already tiny (a few KB), so a *partial* remote fetch buys nothing; the cost is the hop itself, not its size.

### What changed (`tinygrad/llm/model.py`)

`MTPHead.draft` lazily builds `self._local_token_embd` (an `nn.Embedding.__new__(nn.Embedding)` instance with
`.weight = owner.token_embd.weight.to(dev).realize()`, built once — `hasattr` check, same idiom
`GatedDeltaNetBlock._init_state` already uses for `conv_state`/`recurrent_state`) and looks up through that
instead of `owner.token_embd`. `.to(dev)` is `tinygrad.tensor.Tensor.to`'s own documented no-op when the
device already matches — so in any device-map regime where `token_embd` and `mtp_head` already share a device
(including every existing single-device test), this is a true no-op: `local.weight is owner.token_embd.weight`,
zero extra memory, byte-identical computation. Token identity holds: same table (a verbatim copy, or literally
the same object when already colocated), same ids, same lookup function (`Embedding.__call__`'s own
`USE_ATOMICS` dispatch, preserved by constructing a real `Embedding` instance rather than reimplementing the
lookup by hand) — just no hop. `mtp_head` is not touched by `Transformer.snapshot_state`/`restore_state`
(deliberately excluded already — see `snapshot_state`'s own docstring), so this cache never bloats serve.py's
session-state LRU.

## Predicted new per-iteration budget

Using the task's own measured OLD numbers as the baseline and crediting only what's directly attributable to
each fix (conservative — real hardware may do better, see the upside notes):

| phase | OLD (full accept) | OLD (partial accept) | NEW (full accept) | NEW (partial accept) |
|---|---|---|---|---|
| `draft_ms` (k=3) | ~850 | ~850 | **~tens of ms** (candidate #3's own "tens of ms total, not dominant" residual — `_function` retrace + one small block forward per call, the only draft cost left once both hops are gone) | same |
| `verify_dispatch_ms` | ~800 | ~800 | ~800 + small CHECKPOINT overhead (48 native-sized clones, hypothesized cheap per the byte-size argument above — **flagged for hardware confirmation**, see invocation below) | same |
| `accept_ms` | ~107 | ~107 | ~107 (conservative floor; plausibly improves too — capture's removal drops ~4.5 GiB/call of construction and lifts VERIFY's forced-loop restriction, both inside the replayed program `accept_ms` waits on — unquantified from source alone) | same |
| `state_assign_ms` | 0 | ~19,500 | 0 (unchanged) | **~1,800** (CHECKPOINT+REDO, per the task's own given historical measurement) |
| **total/iter** | **~1,757** | **~21,257** | **~937-957** | **~2,737-2,757** |

Sanity check against the hardware fact's own aggregate (53 iterations, 192 tokens, 0.74 tok/s, "43/53 full
chains" — a related but not necessarily identical run, so this is an order-of-magnitude check, not a
reproduction): 43×1,757 + 10×21,257 ≈ 288.1s, vs. the observed 260s — same order of magnitude. Applying the
same 43/10 split to the NEW numbers: 43×957 + 10×2,757 ≈ 68.7s for the same 192 tokens ≈ **2.8 tok/s**, roughly
a **3.8x** aggregate improvement from these two fixes alone, before crediting `accept_ms`'s unquantified
upside.

**Honest limit**: 2.8 tok/s (predicted) is still well under the ~7 tok/s plain-decode baseline on this
hardware — these fixes remove the 10x regression and the dominant per-call draft tax, they don't by themselves
make speculative decoding beat plain decode here. The residual gap is almost entirely `verify_dispatch_ms` +
`accept_ms` (~900ms/iter, present on *every* iteration, full or partial) against plain decode's ~143ms/token —
SPEC_PROFILE_NOTES.md's candidate #2 (T_pad=32 padding a ~4-real-token chunk, independent of the
forced-loop restriction this task's Deliverable 1 change happens to lift) is the next lever, out of this
task's scope.

## Hardware invocation

Same shape as SPEC_PROFILE_NOTES.md §9, re-run after this task's changes:

```
SPEC_TRACE=1 SPEC_STATS=1 DEV=NV ... python -m tinygrad.llm --serve 8081 --mtp ...
```

(same flags as the T4.66b/T4.66c measurement runs — `SPEC_TRACE`/`SPEC_STATS` are `ContextVar`s read at
generator-construction/loop time, an env var suffices same as `MTP`/`GDN_CHUNK`/etc.) against the same k=3
prompt. Read the printed `[SPEC_TRACE]` lines: `state_assign_ms` should now read ~0 on full accept (unchanged)
and roughly 1,000-2,500ms on partial accept (confirms Deliverable 1; anything still near 19,500 means the
CHECKPOINT/REDO revert didn't land as expected). `draft_ms` should drop to a small fraction of its old ~850ms
value regardless of accept type (confirms Deliverable 2) — `draft_ms - draft_dispatch_ms` should now be small
too (candidate #1's own "confirms via" signal, now near-zero instead of large-positive). If
`verify_dispatch_ms` grows by more than a few tens of ms relative to its old ~800ms baseline, the CHECKPOINT
clone (native-sized, not capture-stack-derived) is more expensive on real hardware than this analysis
predicts — worth its own follow-up trace if so, since nothing else in this task's changes touches that phase.

## Tests

- `test/unit/test_spec_decode.py`: all existing token-identity tests unchanged and green, including
  `test_forced_partial_accept_matches_generate_gdn` (forces a genuine mid-chain partial accept and checks
  greedy output still matches `generate()` exactly — this is the test that would catch a CHECKPOINT/REDO
  correctness regression) and `TestSpecTrace` (SPEC_TRACE format/invariants, unchanged — `state_assign_ms`
  still measures the same chronological phase, just different work inside it).
- New: `TestMTPDraftDeviceLocalEmbed.test_no_cross_device_copy_once_warm` — CPU:0/CPU:1 device map (token_embd
  on CPU:0, mtp_head on CPU:1, the non-colocated/interesting case), asserts the draft path's own schedule
  (built via `Tensor.linear_with_vars` — `draft()` is never jitted, so there's no `.captured.linear` the way
  `test_llm_device_map.py`'s tests read one) contains zero `Ops.COPY` calls once the lazy cache is warm.
- `test/unit/test_gdn_scan_parity.py::TestGDNScanCapture` unchanged and green — it calls
  `GatedDeltaNetBlock._attention(..., capture=True)` directly, never through `forward()`, so it's unaffected by
  `forward()` no longer requesting capture.
- Gates run: `CHECK_OOB=1 DEV=CPU pytest test/unit/test_spec_decode.py test/unit/test_gdn_scan_parity.py
  test/unit/test_llm_device_map.py -x -q` → 49 passed, 7 skipped (METAL/NV-only tests, expected on CPU);
  `SPEC=2 DEV=NULL pytest test/null/ -x -q` → 1532 passed, 81 skipped, 16 xfailed, 2 subtests passed (matches
  T4.66b's own baseline count, no regression); `mypy tinygrad/` → no issues; `ruff check .` → all checks
  passed.

## CPU-proxy dispatch counts (before/after) — and why they're the wrong metric here

Analytical, from reading the code (no hardware access from this worktree either) — the same method
SPEC_PROFILE_NOTES.md itself used, and the same one T4.66b's own commit message used for its "25→13 spec
forwards" claim:

- **DRAFT, per call**: unchanged call count (still `k_eff` eager `_function(precompile=True)` calls), but each
  call's schedule now contains 0 cross-device `COPY` nodes instead of 2 (6 removed per iteration at k=3) — this
  is the one count in this task where the CPU proxy and the hardware measurement agree in direction (both say
  "better"), because unlike Deliverable 1, this fix's mechanism (fewer COPY nodes) is exactly what a
  same-device CPU/NULL backend can still observe accurately in relative terms, even though the *absolute*
  per-hop cost the audit is chasing (candidate #1's "not a cheap async dispatch" finding) is real-hardware-only.
- **FIXUP/state-assign, per partial-accept iteration**: T4.66b took this from 2 "spec forwards" (VERIFY + REDO)
  to 1 (VERIFY only, zero REDO) — measured on a CPU/mock harness as "25→13 spec forwards, zero REDO" in that
  commit's own words. **T4.66d reverts this specific count**: back to 2 spec forwards per partial-accept
  iteration (VERIFY + REDO), plus one `Tensor.realize()` bundling 96 small assign/clone nodes both before
  VERIFY (CHECKPOINT, now unconditional — a NEW recurring cost the CPU proxy would also flag, though it's the
  same small shape T4.66b's own removed CHECKPOINT step used to pay every iteration too) and, on partial accept
  only, again after (RESTORE). **By the CPU dispatch-count metric alone, this change is a regression** — it is
  the same metric, measured the same way, that made T4.66b's original change look like unqualified progress.
  The task's own hardware numbers (19,500ms → ~1,800ms) show the metric was measuring the wrong thing: on
  CPU/NULL, `dest.device.split(":")[0] == src.device.split(":")[0]` is trivially true (or there's only one
  device), so the real cost this task fixes — whatever makes a same-device assign sourced from a freshly
  `.stack()`-built, T_pad-inflated capture tensor different from a same-device clone/assign of a stable,
  natively-sized persistent buffer on real HCQ hardware — is exactly as invisible to this proxy as candidate
  #1's cross-device copy cost was. This is the same lesson SPEC_PROFILE_NOTES.md's candidate #1 already drew,
  now confirmed to generalize beyond cross-device copies specifically: **a CPU/NULL dispatch-count proxy is not
  a substitute for a real-hardware trace whenever the thing being optimized is a sync/dispatch cost rather than
  a compute cost**, and this task's whole premise (a hardware-measured audit correcting a CPU-only-validated
  optimization) is itself the evidence for that.
