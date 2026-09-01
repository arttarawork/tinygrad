# T4.66c — first-principles cost audit: where does ~4.9s/iter go?

Hardware fact this task exists to explain (2026-09-01, Qwen3.8-27B pooled METAL+NV, k=3): warm speculative
decode = 0.74 tok/s (260s / 192 tok, 53 iterations ~= 4.9s/iter) vs 7 tok/s plain, despite excellent acceptance
(avg 3.60/verify, 43/53 full chains) and a CPU dispatch-count proxy showing strict improvement (25->13 spec
forwards, zero REDO). Also: prefill in the same round ran 13 tok/s vs 22 without MTP. This document is a
read-the-code audit (no hardware access from this worktree -- CPU/NULL only, see CLAUDE.md's T4.66c grant) of
every host<->device round-trip, schedule build, jit key, cross-device hop, and eager call in the warm-iteration
path, ranked by predicted cost, each tagged with which `SPEC_TRACE` phase (see `speculative_generate` in
`tinygrad/llm/model.py`, and its `SPEC_TRACE` ContextVar docstring) would confirm or kill it on real hardware.

All line numbers below are against this worktree's `tinygrad/llm/model.py` post-T4.66c (SPEC_TRACE added).

## 0. The device-placement fact pattern that makes this interesting

`Transformer.__init__` (`model.py:1017-1050`) places each block per `device_map`, `token_embd` on the FIRST
block's device (`dmap[0]`), `output_norm`/`output` on the LAST block's device (`dmap[-1]`). `from_gguf`'s MTP
branch (`model.py:1330-1334`) places `model.mtp_head` -- `enorm`, `hnorm`, `eh_proj`, `block`, `shared_head_norm`,
ALL of it -- on `last_dev = model.blk[-1].device`, i.e. the SAME device as `output`/`output_norm`. Every device
map on record for this fork (CLAUDE.md's "Pooled model" section: `0-2:METAL,3:NV,...,29-39:NV`) puts block 0 on
METAL and the last block on NV, so in the current serving configuration:

- `token_embd` lives on **METAL** (`dmap[0]`).
- `mtp_head` (all of it) and `output`/`output_norm` live on **NV** (`dmap[-1]`).
- Every OTHER block lives on whichever of METAL/NV the map assigns it (alternating in chunks).

This is a fact about the CURRENT map, not a law: if a future map ever collapsed `dmap[0] == dmap[-1]`, candidate
#1 below (the draft-chain embedding hop) would vanish on its own. A one-line hardware check
(`model.token_embd.weight.device`, `model.mtp_head.block.device`) confirms which regime a given run is in.

## 1. Phase-by-phase walk of one warm iteration (k=3, full accept: draft 3 -> verify 4 -> accept -> emit 4)

Reading `speculative_generate`'s main loop (`model.py:1611` on): `SPEC_TRACE`'s phase boundaries were chosen to
line up with this walk, so the phase names below are literally the timer names in the code.

### (a) DRAFT -- `SPEC_TRACE` phase `draft_ms` / `draft_dispatch_ms`

`for i in range(k_eff): dlogits = self.mtp_head.draft(self, h_last, dtok, v_draft_pos[i].bind(dpos))` then
`dtok = dlogits.argmax(-1)` (greedy). Per call, `MTPHead.draft` (`model.py:968-981`) does:

1. `tok_ids.to(owner.token_embd.weight.device)` -- `tok_ids` (== `dtok`, always produced by a PREVIOUS call's
   `owner.output(...)`, which lives on `dmap[-1]`=NV -- see next line's device, or by `tok_last` for i=0, ALSO
   on NV, see ACCEPT below) moves **NV -> METAL**. Lazy (`.to()` builds a `COPY` UOp, doesn't execute yet --
   `tensor.py:543-548`).
2. `owner.token_embd(...)` -- embedding lookup, on METAL, cheap (one row of a big table).
3. `.float().to(dev)` where `dev = self.block.device` (`model.py:980`) == NV (mtp_head's own placement) --
   moves the embedded vector **METAL -> NV**. Lazy, same as above.
4. `h.to(dev)` -- `h` is `h_last`, already on NV (it's `verify_h`'s slice, and `verify_h` comes from `forward()`'s
   block loop ending on `self.blk[-1].device` == NV) -- **no-op**, not a real hop.
5. `self.block(x, start_pos)` -- ONE `TransformerBlock`/`MLATransformerBlock` forward (see SS5(a) below for why
   it's exactly one, never the 64-block main model), entirely on NV.
6. `owner.output(self.shared_head_norm(x).to(owner.output.weight.device))` -- `.to(NV)` is a no-op (`x` already
   on NV from step 5); the final vocab projection also runs on NV.

So **every DRAFT call makes exactly 2 real cross-device COPYs** (steps 1 and 3), both through the
METAL<->NV boundary, in the current device-placement regime (SS0). None of this is `TinyJit`-wrapped (SS3(a)
below) -- step 5's block call retraces its Python body on every single invocation.

None of steps 1-6 call `.tolist()/.numpy()/.item()/.realize()` -- the whole DRAFT call chain stays lazy. The
FIRST real sync in the DRAFT phase is `chunk_ids = (tok_last.cat(*draft_tensors, dim=1) ...).tolist()`
(`model.py:1652`, greedy) -- one host pull that resolves the ENTIRE merged schedule: all k_eff draft calls
(their block-forward kernels AND their 2*k_eff cross-device copies) plus the final `cat`. `draft_ms` (wraps this
whole span) is therefore the one number reflecting DRAFT's real device time; `draft_dispatch_ms` (sum of each
call's own wall-clock, no sync inside) is Python/graph-construction overhead only -- see SS3(a).

(Sampled/temperature>0 path differs: each drafted id needs a REAL `.numpy()` pull inside the loop, so there
`draft_dispatch_ms` already includes device time per call -- documented in the `SPEC_TRACE` ContextVar
docstring. The hardware fact this task explains doesn't state a temperature, but greedy is `serve.py`'s default
and the more likely path; the audit below assumes greedy unless noted.)

### (b) VERIFY dispatch -- `SPEC_TRACE` phase `verify_dispatch_ms`

`buf = Tensor(chunk_ids + [0]*(chunk_size-n), device=dev)` where `dev = self.blk[0].device` == METAL
(`model.py:1531`) -- builds a FRESH tensor from a Python list (unlike `generate()`, which chains `out` device-side
and only re-homes it once per step, see SS6) -- host-side list/array construction + upload to METAL. Then
`self(buf[:, :nt], sp, None, spec=True)` -- one call into `Transformer.__call__` (`model.py:1154`), which
`setdefault`s a `TinyJit(self.forward)` keyed `(is_prefill, greedy, chunk_size, spec)` (SS3(b) -- always the SAME
key here, warm after the first VERIFY ever run) and replays it. A REPLAY of an already-captured `TinyJit` does
NOT re-run `forward()`'s Python body -- it replays a fixed, pre-linearized program of kernel-launch and copy
`ExecItem`s (`engine/jit.py`), so none of `forward()`'s 64-block Python loop, nor any block's
`@function(precompile=True)` retracing, happens here. This line only DISPATCHES that program -- no
`.tolist()/.numpy()/.item()/.realize()` is called on its result, so `verify_dispatch_ms` is dispatch-only by
construction: it measures how long it takes to build the input buffer and hand the replay to the runtime, not
how long that replay takes to execute.

### (c) ACCEPT -- `SPEC_TRACE` phase `accept_ms`

`verify_ids_tensor = verify_logits.argmax(-1)` (dispatch), then
`verify_ids = verify_ids_tensor.pad_to((1, chunk_size)).tolist()[0][:n]` (`model.py:1683`) -- **the real sync**:
this `.tolist()` is what actually waits for VERIFY's replayed program (SS(b) above) to finish executing on the
device, INCLUDING every cross-device hop inside the main model's own 64-block loop (`x = x.to(block.device)` at
every block boundary where the map changes device -- ~13-14 such hops for the CLAUDE.md map's 8 alternations,
shared with plain decode, see SS6). Then the host-only `while m < k_eff and ...: m += 1` compare loop (list
indexing, free), then `h_last = verify_h[:, m:m+1].contiguous()` (a cheap dispatch, no further sync -- `.contiguous()`
materializes a view into its own buffer but does not force realize, see SPEC_NOTES.md SS3). `accept_ms` is
therefore the one number that reflects VERIFY's actual end-to-end device time (dispatch delay from (b) plus
execution plus every cross-device hop the main-model forward makes).

### (d) FIXUP / state-assign -- `SPEC_TRACE` phase `state_assign_ms`

`if m < k_eff:` (never true on a full accept, per this section's own k=3-full-accept walk -- reads ~0 here,
included below for completeness since T4.66c times the block unconditionally): per-GDN-block
`.assign()` calls (device-local, no cross-device hop -- `state_track`/`conv_window` are produced on the SAME
device as their owning block, see `model.py:748-763`) then `Tensor.realize(*assigns)`. `Tensor.realize()`
defaults to `wait=False` (`tensor.py:420-425` -> `engine/realize.py:318`'s `run_linear(..., wait=False)` default)
-- this call SCHEDULES the fixup and returns without blocking; its actual device cost lands wherever the NEXT
real sync happens to fall (this same iteration's `accept_ms` already happened, so it's the FOLLOWING iteration's
`draft_ms` or `accept_ms`). `state_assign_ms` is therefore dispatch-only, always -- exactly like SS(b).

### total_ms

Independently measured (`iter_t0` at the very top of the loop body to just before the print). Because the four
phases above are sequential, non-overlapping sub-windows of this same span, `draft_ms + verify_dispatch_ms +
accept_ms + state_assign_ms <= total_ms` is a hard invariant on the underlying floats (only defeated by
`%.2f`-rounding noise, see the unit test) -- any run where the gap is NOT tiny means a host round-trip escaped
this list (see SS2 for the ones already hunted down; if this ever happens, it's the next thing to trace).

## 2. Every host<->device round-trip in the loop (`.tolist()`/`.numpy()`/`.item()`/`.realize()`)

Hunted by reading the whole loop body top to bottom (`model.py:1611-1747`):

| call | line (approx) | forces a wait? | phase |
|---|---|---|---|
| `chunk_ids = (...).tolist()` | ~1652 (greedy) | YES -- resolves the whole DRAFT schedule | `draft_ms` |
| `dlogits[:, -1, :].numpy()` | ~1660 (sampled draft only) | YES, per drafted id | `draft_dispatch_ms` (sampled) |
| `verify_ids_tensor.pad_to(...).tolist()` | ~1683 (greedy) | YES -- resolves VERIFY | `accept_ms` |
| `verify_logits.pad_to(...).numpy()` | ~1697 (sampled accept) | YES -- resolves VERIFY | `accept_ms` |
| `tok_last.item()` (prefill anchor only, once per whole call) | ~1596 | YES, but once per REQUEST not per iteration | (outside the loop; not per-iter) |
| `Tensor.realize(*assigns)` | ~1718 | **NO** -- `wait=False` default | `state_assign_ms` (dispatch-only) |
| `h_last = verify_h[:, m:m+1].contiguous()` | ~1707 | **NO** -- `.contiguous()` doesn't realize | inside `accept_ms`, dispatch-only tail |
| `buf = Tensor(chunk_ids + [0]*..., device=dev)` | ~1668 | host-side list->array->device construction, not a device WAIT, but real host CPU work | `verify_dispatch_ms` |

That's it -- every `.tolist()/.numpy()/.item()/.realize()` in the loop is accounted for. Two real device syncs
per iteration (`draft_ms`'s tail, `accept_ms`'s head), one dispatch-only `realize()` whose true cost is deferred,
one host-side buffer build. This matches SPEC_NOTES.md's own claim ("exactly two `.tolist()` pulls") for the
greedy path, refined here with exactly where each one's cost actually lands.

## 3. Every schedule build / jit key touched

**(a) DRAFT: never jitted.** `MTPHead.draft` -> `self.block(x, start_pos)` -> (for a plain `TransformerBlock`,
the common case) `FFNBlock.__call__` (`model.py:440-445`), which builds a **fresh** `@function(precompile=True)`
closure and calls it on every invocation. Reading `tinygrad/function.py:43-90`'s `_function.__call__`: it always
does `ret = self.fxn(*args, **kwargs)` (line 56) -- i.e. it **reruns the Python body** (builds a new UOp graph
through `self._attention`/`self._feed_forward` etc.), then `get_state_dict`, `dedup`, a `graph_rewrite` pass
(`"get_implicit_inputs"`), `renumber_invalid_outputs`, and `uret.call(..., precompile=True)`. `precompile=True`
caches the COMPILED KERNEL/CALL target (so the actual GPU kernel isn't recompiled every call), but nothing
caches the PYTHON-side retracing -- confirmed by reading the function, and already the express subject of
SPEC_NOTES.md SS7's "not done" item ("`_function` never caches across calls the way `TinyJit` does"). This is
paid **k_eff times per iteration**, unconditionally, on real hardware and on CPU/mock alike -- it's genuine
Python overhead, small in absolute terms (tinygrad's own UOp/graph_rewrite passes are typically low-single-digit
milliseconds for a graph this size) but not zero, and it is what `draft_dispatch_ms` (greedy mode) measures.

**(b) VERIFY: one stable jit key, proven never to retrace.** `Transformer.__call__` (`model.py:1154-1163`) keys
`self.jit` on `(is_prefill, temperature is None, chunk_size, spec)`. For every VERIFY call: `spec=True` (fixed),
`temperature is None` i.e. greedy (fixed per `speculative_generate` call), `chunk_size` (fixed per call, a
python int closed over the whole generator). The only question is `is_prefill = bool(resolve(tokens.shape[1] !=
1))` (`model.py:1155`) -- does THIS vary with `n` (1..k+1)? Read `resolve()` (`uop/ops.py:54-58`): for a
non-concrete-bool UOp, it returns `bool(sx.vmin)` only if `sx.vmin == sx.vmax`, else the `default` (which
`resolve()`'s own signature defaults to `True`, and `model.py:1155` doesn't override it). `tokens.shape[1]` here
is `nt = v_toks.bind(n)` -- and `UOp.bind()` (`uop/ops.py:995-1001`) returns `self.after(self.store(uval))`, an
`AFTER` node whose `_min_max` is defined (`uop/ops.py:1102`) to equal `self.src[0]._min_max` -- i.e. the
ORIGINAL Variable's range, `[1, chunk_size]`, REGARDLESS of what concrete value it's bound to. So
`nt != 1`'s own min/max straddle both `False` (when `nt` could be 1) and `True` (when `nt` could be >1) --
`sx.vmin != sx.vmax`, so `resolve()` can't prove either way and falls to `default=True`. **`is_prefill` is
`True` unconditionally for every VERIFY call, independent of `n`/`k_eff`/`m` — the jit key never changes, so
VERIFY never retraces across iterations for varying k_eff or accepted length.** (This was already claimed,
untested-by-me, in SPEC_NOTES.md SS4; this is an independent code-level re-derivation of why it's actually
true, not just asserted.) One-time cost only: the FIRST VERIFY call of a `speculative_generate` run (and the
prefill loop's own final `spec=True` chunk, which shares this exact key) pays a real `TinyJit` capture (plus, on
real hardware, a possible cold BEAM search per CLAUDE.md's own notes on kernel cache warmth) -- not a per-iteration
cost on a warm server.

**(c) The recurrent (GDN) scan's own hidden schedule inflation.** `chunk_size` for `speculative_generate` (no
override from `serve.py`, which calls it with defaults -- SPEC_NOTES.md SS6's `Handler.run_model` snippet passes
only `k`/`temperature`) resolves to `min(32, gdn_chunk_for(device))` then `max(that, k+1)`
(`model.py:1528-1529`); `gdn_chunk_for` (`model.py:22-24`) auto-selects **32** on METAL/NV/CUDA. So VERIFY's
declared/padded token-axis width (`T_pad = x.max_shape[1]`, `model.py:656`) is **32**, not `k_eff+1` (~4) --
`v_toks`'s Variable was deliberately given range `[1, chunk_size]` so ONE jit slot serves every VERIFY length
(SS(b) above, and SPEC_NOTES.md SS4), but the price is that the recurrent scan's per-token Python-unrolled loop
(`GatedDeltaNetBlock._attention`, `for t in range(T_pad):` at `model.py:725`) runs **32 iterations every VERIFY
call, for every GDN block**, even though only k_eff+1 (~4) of those steps carry real tokens. This is NOT new to
`spec=True` -- any chunked-prefill call sharing this Variable pays it too -- but VERIFY pays it on EVERY
iteration (once per ~3.6 emitted tokens, per the hardware fact's own avg accept length), not once per whole
prompt. `capture=True` (VERIFY's own flag, `_attention`'s docstring at `model.py:621-655`) makes this worse in a
second, independent way: it forces the plain per-token LOOP -- never the WY chunked form, never the AMD fused
kernel -- regardless of `GDN_SCAN_IMPL` (`model.py:718`'s `use_wy = ... and not capture`), and ADDS a `.stack()`
retention of the per-step state at all 32 positions (`states.append(state)`, `model.py:729`), not just the ~4
real ones. T4.69b's own measurement (cited in SPEC_NOTES.md SS7) shows WY roughly +25-30% over the loop at
prefill-sized (5.9k-16.2k token) chunks -- i.e. the loop is the SLOWER of the two at any chunk with real work in
it, and `capture=True` structurally can't use the faster one. This is the **T_pad tax**: an ~8x width overhead
(32 padded steps vs ~4 real ones) on top of forcing the intrinsically slower scan implementation, paid by EVERY
GDN block -- most blocks in this fork's qwen3.5/qwen3.6-family hybrid architectures, per `from_gguf`'s
`ssm_layers` pattern (`model.py:1240`); exact count for the Qwen3.8-27B checkpoint in the hardware fact isn't in
this worktree's docs, but the mechanism doesn't depend on the exact number -- on every single VERIFY call. It
shows up inside `accept_ms` (VERIFY's real device time), not as a separate phase -- there's no
per-block sub-timer in this task's scope, but SS8's ranking below gives it a predicted magnitude and a way to
test it in isolation.

## 4. Every cross-device hop, counted

Per outer iteration, k_eff=3, full accept, current device-placement regime (SS0):

- **DRAFT: 2 hops x k_eff = 6** (`MTPHead.draft`'s token-embedding round trip, SS1(a) steps 1+3). All folded
  into `draft_ms`'s tail sync -- see SS5/SS8 for why these are not "free async dispatches."
- **VERIFY: however many block-to-block device transitions the map has.** `forward()`'s block loop
  (`xin = x.to(block.device)`, inside `Transformer.forward`, called once per block) hops every time consecutive
  blocks in `self.blk` sit on different devices -- illustrating with the ONE fully-documented map in this repo
  (CLAUDE.md's 192k pooled map for the 35B model, `0-2:METAL,3:NV,4-6:METAL,7:NV,...`, 40 blocks, 8
  alternations): that's **~14 hops** per full-model forward. The Qwen3.8-27B checkpoint in the hardware fact
  almost certainly uses a different (not-yet-documented-in-this-worktree) map, but the STRUCTURE is the same --
  count `sum(1 for a,b in zip(dmap, dmap[1:]) if a != b)` for whatever map that run actually used to get the
  real number. This is IDENTICAL machinery to plain `generate()`'s decode step (same `forward()`, same loop) --
  it is not a NEW cost `speculative_generate` introduces, it's the map's fixed per-forward tax, paid by VERIFY
  (once per iteration, over k_eff+1 tokens) at roughly the same per-hop cost as plain decode pays per TOKEN
  (once per token). Shows up inside `accept_ms`.
- **FIXUP: 0.** `state_track`/`conv_window` are produced on the SAME device as their owning GDN block
  (`model.py:748-763` returns them alongside `out`, no `.to()` in between), and `block.recurrent_state`/
  `conv_state` live on that same block's device -- the `.assign()` calls in the FIXUP loop
  (`model.py:1712-1715`) never cross devices.
- **`buf`/prefill-tensor construction: 0 additional device-class hops** -- these are built directly with
  `device=dev` (`model.py:1531`'s `dev = self.blk[0].device`, METAL), matching `token_embd`'s device, so no
  `.to()` is needed once `self(...)` starts.

**Answering the task's specific question:** the draft head (`mtp_head`, all of it) lives ENTIRELY on one device
(NV, SS0) -- `MTPHead.draft` itself runs no cross-device compute hop for its own block forward. The hop is
narrower and sneakier than "the draft head spans two devices": it's `owner.token_embd` (borrowed from the main
model, living on the OTHER end of the map) that forces exactly 2 copies per call, k_eff times per iteration.

## 5. Special-attention answers

**(a) Does `MTPHead.draft` run the full 64-block model or only the nextn block?** Read `MTPHead.draft`
(`model.py:968-981`): it calls `self.block(x, start_pos)` exactly once, where `self.block` is a SINGLE
`TransformerBlock` (or `MLATransformerBlock`) instance -- the same class the main model uses for one block, built
once in `MTPHead.__init__` (`model.py:961`) and never looped. **Definitively: only the nextn block, never the
64-block main model.** The (comparatively large) fixed cost per draft call is NOT "64 blocks of compute" -- it's
(i) the 2 cross-device copies (SS4) and (ii) the never-jitted `_function(precompile=True)` Python retrace (SS3a)
on ONE small block. Both are real but neither is "a whole extra forward pass through the model."

**(b) Does VERIFY's `spec=True` jit key retrace when k_eff or accepted length varies?** No -- proven in SS3(b)
above by tracing `resolve()`+`UOp.bind()`'s `_min_max` propagation, not just citing the design intent: `nt`'s
range stays `[1, chunk_size]` after `.bind()`, so `is_prefill` can never be proven `False` and always resolves
to the safe default `True`, for every `n` from 1 to k+1. The jit key `(True, True, chunk_size, True)` is
identical across every VERIFY call in a `speculative_generate` run.

**(c) Do the capture-mode extra outputs force per-iteration device->host copies or schedule rebuilds?** No host
copies: `gdn_extra` (the `state_track`/`conv_window` tuple) is only ever touched by `.assign()` inside the SAME
device's FIXUP step (SS4) -- never `.numpy()/.tolist()/.item()`'d. No schedule rebuilds either: VERIFY's capture
path is baked into the ONE jit capture from SS3(b) (the graph is bigger because `capture=True` adds `.stack()`
nodes per GDN block, but that bigger graph is captured once and replayed, same as the plain graph would be). The
REAL cost capture-mode adds is the T_pad tax from SS3(c): every GDN block computes and retains state at all 32
padded positions (not ~4), and is forced onto the slower per-token-loop scan instead of WY -- extra DEVICE
compute+memory-bandwidth work baked into the one-time-captured, per-call-replayed VERIFY schedule, not a NEW
sync or a NEW schedule per call.

**(d) The MTP=1 prefill sag (13 vs 22 tok/s, "the same round") -- what does loading mtp_head change for
non-spec calls?** Three candidate mechanisms, in descending likelihood:

1. **`speculative_generate`'s own prefill loop forces `spec=True` (hence `capture=True` on every GDN block) on
   its FINAL chunk** (`model.py:1544`: `prefill_result = ... self(t[:, sp:sp+nt], sp, None, spec=True)`) --
   purely to recover `h_last` for the first draft anchor. This chunk pays the FULL SS3(c) T_pad tax (forced
   loop, no WY, full 32-wide retention) that a plain `generate()` prefill chunk of the same size never would
   (plain prefill's `spec=False` chunks are free to use WY per `GDN_SCAN_IMPL`'s auto-resolution, and T4.69b's own
   note says WY specifically wins at prefill-sized chunks, ~+25-30%). If the benchmarked prompt is short enough
   that a meaningful fraction of its chunks are the (necessarily-`spec=True`) LAST one, this alone could produce
   a visible aggregate slowdown. This is a real, structural, ALREADY-PRESENT-IN-THE-CODE effect, not a mystery --
   it's the cost of getting `h_last` the SS3(c)-described way, paid once per `speculative_generate` CALL (not
   per outer iteration), on exactly the chunks that need it.
2. **Memory pressure from mtp_head's own weights sitting on NV.** `mtp_head` adds one extra block's worth of
   parameters (`enorm`, `hnorm`, `eh_proj`, `block`, `shared_head_norm`) plus its own attention KV cache (sized
   off the SAME `max_context` as the main model, per `MTPHead.draft`'s docstring, `model.py:975-976`) onto
   whichever device holds `dmap[-1]` -- NV in the current map, which CLAUDE.md's own numbers already show
   running close to the 3090's 24GB ceiling (20.6GB static at 192k context, BEFORE mtp_head). `LRUAllocator.alloc`
   (`device.py`, `class LRUAllocator`) falls back to `free_cache()` + retry on `MemoryError` -- if mtp_head's
   extra footprint pushes NV close enough to the edge that this fallback starts firing during PREFILL's normal
   buffer churn (chunked prefill allocates/frees a lot of scratch), every such retry is a real, if hard-to-size-
   from-code-alone, slowdown. This is the "memory pressure?" hypothesis the task names -- plausible, but its
   magnitude can't be predicted from source alone; a hardware check (compare `Device["NV"].allocator`'s free/used
   before vs. after loading mtp_head, or just diff prefill tok/s with `MTP=0` vs `MTP=1` but WITHOUT ever calling
   `speculative_generate`, isolating this from candidate 1) would settle it directly.
3. **A cold/uncached kernel variant for the `spec=True` prefill-tail shape.** The `spec=True` jit key
   (`is_prefill=True, greedy, chunk_size, spec=True`) is DIFFERENT from every earlier prefill chunk's
   `spec=False` key -- if this specific hardware round was the first time THIS chunk_size/shape combination's
   `spec=True` variant was ever traced (even with a warm BEAM cache for the individual kernels, `TinyJit`'s OWN
   capture -- schedule linking, buffer planning -- still happens once per NEW key), that one-time capture cost
   could land inside the SAME wall-clock window a bench harness attributes to "prefill." Distinguishing this from
   candidate 1 needs either a second back-to-back run (capture cost shouldn't recur) or a timestamp around the
   specific chunk that trips `spec=True`.

Candidate 1 is the only one already fully confirmed by reading this codebase; 2 and 3 are real, plausible, and
each independently testable in one hardware run, but not something `SPEC_TRACE` (scoped to the OUTER per-token
loop, not the prefill loop, per this task's own deliverable boundary) instruments directly -- flagging that
gap here rather than silently expanding SPEC_TRACE's scope.

## 6. Comparison to plain decode's per-token path

`generate()`'s decode step (`model.py:1391-1410`): build `sp, nt` (both concrete/degenerate here, `nt` bound to
1), ONE `self(...)` call (a `TinyJit(self.forward)` replay, same jit machinery as VERIFY, keyed
`(False, greedy, None, False)` -- `is_prefill=False` since a decode call's `tokens` shape really is a concrete
1, so `resolve()` proves it directly this time, no `default` fallback needed), `.realize()` (dispatch-only,
`wait=False`), and -- critically -- **one explicit device hop already in the design**:
`if out.device != t.device: out = out.to(t.device).realize()` (`model.py:1406`), because the JIT capture is
keyed to a SPECIFIC input device and the chained `out` (which lands on `dmap[-1]` after a forward) must be
normalized back to `dmap[0]` before the next replay. This is EXACTLY one cross-device copy per decode step
(output-device -> first-block-device), same METAL<->NV boundary as everything else here, and it's already
priced into the 7 tok/s (~143ms/token) baseline this whole audit measures against. Then ONE host sync
(`Tensor.cat(*pending, dim=1).tolist()`, `model.py:1420`, batched every `drain_every` steps -- 1 by default) --
so plain decode's per-token cost is: **one jit replay (with the map's ~14 in-forward device hops, T=1-shaped, no
T_pad tax since T=1 is concrete not symbolic) + one chained cross-device hop + one `.tolist()`.** No
`_function(precompile=True)` Python retracing outside the jit (nothing in `generate()`'s loop calls a bare block
directly), no capture-mode T_pad inflation (T_pad=1 for a concrete decode-shaped call, not 32), no per-step
MTPHead calls at all.

Speculative decode's VERIFY step is structurally the SAME shape of work (one jit replay of `forward()`, same
per-block device hops) MINUS T=1's advantage (T_pad=32 padding, forced loop scan, capture retention -- SS3(c))
PLUS an entirely new category of work plain decode never pays at all: the DRAFT chain's k_eff never-jitted,
2-cross-device-hop-per-call `MTPHead.draft` invocations. **The mystery time is not "VERIFY is a slower version of
one decode step" -- VERIFY should cost roughly a small multiple of one decode step's ~143ms (a few hundred ms,
inflated by T_pad=32's ~8x GDN-scan tax on top of a >1-token forward). The mystery time is DRAFT: a category of
cost (never-jitted eager calls with their OWN cross-device round trips) that has no analogue in plain decode's
per-token path at all, run k_eff=3 times per iteration.**

## 7. Every non-jit eager call in the loop

Exhaustive list of everything in the per-iteration path that is NOT a `TinyJit` replay:

- `k_eff = min(...)` -- pure Python int arithmetic.
- The k_eff `self.mtp_head.draft(...)` calls -- SS3(a): bare `@function(precompile=True)` closures, retraced
  every call.
- `dlogits.argmax(-1)` / the sampled-path `_softmax_np`+`rng.choice` -- lazy Tensor op (greedy) or real host numpy
  (sampled), per drafted position.
- `tok_last.cat(*draft_tensors, dim=1)` (greedy) -- one lazy cat, folded into the k_eff draft calls' merged
  schedule, resolved at the same final `.tolist()`.
- `buf = Tensor(chunk_ids + [0]*..., device=dev).reshape(...)` -- eager host-side list/array build + device
  upload, not inside any jit.
- The greedy/sampled ACCEPT math (`verify_ids_tensor.pad_to(...)`, the `while m < k_eff` compare loop,
  `accepted = chunk_ids[1:1+m] + [...]`) -- all host-side Python once the sync above lands the data.
- `h_last = verify_h[:, m:m+1].contiguous()` -- one eager lazy op (materializes a view, doesn't realize).
- The FIXUP `.assign()` calls + `Tensor.realize(*assigns)` -- eager, dispatch-only.

Everything ELSE in the iteration (the whole 40-block main-model forward, both in VERIFY and in plain decode) runs
inside the ONE `TinyJit(self.forward)` replay per call -- no Python body re-execution, no `_function` retracing,
just a fixed program of kernel launches and copies. This is the single clearest structural asymmetry between
DRAFT (all eager, k_eff times) and VERIFY (one jit replay): DRAFT pays Python-and-copy overhead PROPORTIONAL TO
k, VERIFY doesn't scale with anything except the (fixed, k-independent) T_pad tax.

## 8. Ranked candidates for the missing ~4.75s/iter

Ranked by predicted magnitude (high to low), each with the `SPEC_TRACE` phase that confirms or kills it.

**#1 -- Cross-device COPY between METAL and NV is a genuine, non-trivial HOST-BLOCKING sync, not a cheap async
dispatch, and DRAFT pays 2*k_eff of them per iteration, all invisible until `draft_ms`'s tail sync.**
Predicted magnitude: **largest single candidate, plausibly seconds, not milliseconds.** Evidence, read directly
from `tinygrad/engine/realize.py` and `tinygrad/runtime/support/hcq.py` (outside this worktree's own `llm/`
code, but load-bearing for this claim so traced anyway):
`exec_copy` (`engine/realize.py:170-180`) only takes the fast `_transfer` (true peer-to-peer) path when
`dest.device.split(":")[0] == src.device.split(":")[0]` -- FALSE for METAL<->NV, so it falls to
`_copyout`/`_copyin`, which have **no `wait=` parameter at all** (unlike `exec_kernel`, which threads
`wait=ctx.wait` through for async dispatch) -- they are unconditionally executed inline, synchronously, at
schedule-execution time. On the NV side specifically (`HCQAllocator._copyin`/`_copyout`,
`tinygrad/runtime/support/hcq.py:581-624`): every chunk of a copy submits a hardware SDMA copy that first WAITS
on `self.dev.timeline_signal` for `self.dev.timeline_value - 1` (i.e. "whatever was the most recently submitted
piece of work on this device, kernels included"), then the HOST itself blocks in `_drain()`
(`self.dev.timeline_signal.wait(self.b_timeline[buf_idx])`) until that copy's own completion signal lands, before
the bytes can be read into/out of the staging buffer. In effect: **the first cross-device copy encountered
while executing a merged schedule forces the host to wait for essentially everything dispatched so far on that
device to actually finish** -- not just its own producing kernel. `MTPHead.draft`'s 2 copies per call, times
k_eff=3, means the DRAFT phase's merged schedule contains 6 such barriers, interleaved with (otherwise tiny)
single-block-forward kernel dispatches -- turning what LOOKS like "queue k_eff async calls, sync once at the
end" into "dispatch a little, block, dispatch a little more, block again," six times, before `draft_ms`'s own
`.tolist()` is even reached. This is qualitatively different from, and much more expensive than, ordinary
kernel-launch dispatch overhead, and it is **structurally invisible to a CPU/mock dispatch-count proxy**: on
CPU (or a single-device mock), `dest.device.split(":")[0] == src.device.split(":")[0]` is trivially true (or
there's only one device at all), so NONE of this code path ever executes -- the exact same "25->13 spec
forwards, zero REDO" measurement that looked like a strict win on CPU pays zero cost for this on CPU, by
construction, while paying it in full on the real METAL+NV hardware. This is the leading explanation for "a
dispatch-count proxy can't see it."
**Confirms via:** `draft_ms - draft_dispatch_ms` (large positive gap => hidden tail-sync cost, consistent with
this) on a hardware run; a definitive confirm would add (out of this task's scope, a natural T4.66d follow-up)
a print of `model.token_embd.weight.device` vs `model.mtp_head.block.device` to confirm the hop exists at all in
that run's device map, and/or a version of `MTPHead.draft` that skips the embedding hop (e.g. draft-token
embedding done once on `dev` via a cached/co-located copy of the embedding table) as an A/B.

**#2 -- The T_pad tax: VERIFY's GDN scan runs `chunk_size` (32) padded steps instead of k_eff+1 (~4), forced
onto the slower per-token loop (never WY) by `capture=True`.**
Predicted magnitude: **meaningful, likely hundreds of ms, probably not the whole gap alone.** SS3(c) derives
this precisely: an ~8x width overhead on every GDN block's scan (most blocks, in this fork's qwen3.5/qwen3.6-
family hybrid architectures -- see `from_gguf`'s `ssm_layers` pattern, `model.py:1240`), paid on every VERIFY
call (once per ~3.6 emitted tokens), and T4.69b's own
measurement (SPEC_NOTES.md SS7) puts the loop-vs-WY gap at ~25-30% at real (prefill-sized) chunk widths -- so
this candidate is "~8x too much scan work, on the slower of two known implementations," not a constant-factor
tweak. Ceiling on its contribution: VERIFY only runs ONCE per iteration (not k_eff times like candidate #1), so
even a several-hundred-ms hit here is smaller in aggregate than #1's 6-barriers-per-iteration story, unless the
per-barrier cost in #1 turns out to be much smaller than predicted.
**Confirms via:** `accept_ms` (this is where VERIFY's real device time lands) compared against a rough
plain-decode-equivalent baseline (~143ms/tok from the 7 tok/s baseline, scaled for a >1-token forward) -- if
`accept_ms` is many multiples of that baseline, this candidate (or #1's shared-forward hops, priced in equally
to both) is live. Isolating it from #1 needs a same-hardware run with `GDN_HEAD_GROUPS`/`GDN_SCAN_IMPL` forced
differently, or (cleanest) comparing `accept_ms` against plain `generate()`'s own per-forward cost on a
same-sized (k+1-token) chunk with `spec=False` -- out of this task's instrumentation scope but a natural
follow-up bench.

**#3 -- `_function(precompile=True)` Python-side retracing, k_eff times per iteration, never cached (unlike
`TinyJit`).**
Predicted magnitude: **small, likely tens of ms total, not the dominant term.** SS3(a): confirmed by reading
`tinygrad/function.py:43-90` -- `self.fxn(*args, **kwargs)` reruns every call, plus `get_state_dict`/`dedup`/
`graph_rewrite`/`renumber_invalid_outputs`. This is real, measurable, PURE overhead (no compute value), but
tinygrad's own graph-construction passes over a single small block are not typically multi-hundred-ms operations
-- this is the kind of cost that shows up clearly in a profiler but rarely dominates a multi-second gap on its
own. Included because the task asked to rank it, and because it's REAL work an actual `TinyJit`-wrapped
draft-chain (SPEC_NOTES.md SS7's "not done" design) would eliminate.
**Confirms via:** `draft_dispatch_ms` directly (it's defined to measure exactly this, in greedy mode) --
compare its magnitude (should be tens of ms for k_eff=3) against `draft_ms`'s total; if `draft_dispatch_ms` is
already a big fraction of `draft_ms`, this candidate is bigger than predicted and #1 correspondingly smaller.

**#4 -- Memory pressure / allocator churn from mtp_head's extra footprint on NV (also SS5(d)'s candidate 2 for
the prefill sag).**
Predicted magnitude: **unknown from source alone -- could be anywhere from negligible to significant depending
on how close to the 3090's ceiling the specific model/context combination sits.** `LRUAllocator.alloc`
(`tinygrad/device.py`, `class LRUAllocator`) falls back to `free_cache()` + retry on `MemoryError` -- if NV is
tight enough that this fires repeatedly during the iteration loop's normal buffer churn (draft-chain block
outputs, VERIFY's capture-mode state tensors), every retry is real, unpredictable-from-source latency. Not
specific to DRAFT or VERIFY -- would show up as elevated, possibly erratic, `draft_ms`/`accept_ms` together, or
(more distinctively) as a slowdown that WORSENS over the course of a long generation as fragmentation grows,
rather than a flat per-iteration constant.
**Confirms via:** total_ms trend across iterations within one `SPEC_TRACE=1` run (flat vs. drifting upward) plus
(outside this task's scope) an out-of-band memory-usage sample on NV during the run.

**#5 -- `buf` construction's fresh host-list-to-device-tensor build every VERIFY call, instead of chaining a
device-side tensor like `generate()` does.**
Predicted magnitude: **small.** `buf` is built on METAL (`dev = self.blk[0].device`), and METAL's unified memory
makes small host->device uploads cheap; this is real (unlike `generate()`'s device-side `out` chaining,
`speculative_generate` round-trips `chunk_ids` through a Python list every iteration even though the ids came
from a `.tolist()` that already happened for the ACCEPT compare) but a handful of ints is not a bandwidth-bound
operation.
**Confirms via:** `verify_dispatch_ms` (this line lives inside that phase) -- if unexpectedly large relative to
just "queue one jit replay," this is bigger than predicted.

## 9. Hardware invocation for the next round

Add `SPEC_TRACE=1` to the standing T4.66b hardware line (CLAUDE.md's pooled-server invocation, `--mtp` serving
path): run the pooled server with `SPEC_TRACE=1 SPEC_STATS=1 DEV=NV ... python -m tinygrad.llm --serve 8081
--mtp ...` (same flags as the existing T4.66b measurement run, `SPEC_TRACE`/`SPEC_STATS` are both `ContextVar`s
read at generator-construction/loop time, so an env var suffices same as `MTP`/`GDN_CHUNK`/etc.) and re-run the
exact same k=3 prompt. Read the printed `[SPEC_TRACE]` lines: if `draft_ms` dominates and
`draft_ms - draft_dispatch_ms` is the bulk of it, candidate #1 is confirmed as the primary cost; if `accept_ms`
dominates instead, candidate #2 (or the shared per-forward device hops from SS6, common to plain decode too) is
the story instead; if neither phase sums close to the externally-observed ~4.9s/iter, the gap is OUTSIDE this
function entirely (serve.py, HTTP, scheduler contention) and this instrumentation will have proven that just as
usefully.
