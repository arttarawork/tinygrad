# T4.66e — native-width VERIFY: giving speculative decode's verify/redo their own jit key

Follow-up to SPEC_PROFILE_NOTES.md (T4.66c's read-the-code audit, candidate #2: "the T_pad tax") and
SPEC_FIXES_NOTES.md (T4.66d, which fixed candidate #2's *forced-loop* half by dropping `capture=True`
from the hot path, but explicitly left candidate #2's *width* half — VERIFY padding every k_eff+1 (~4)
real tokens out to the prefill loop's own `chunk_size` (32) — "a separate, still-open lever tied to
chunk_size/v_toks's shared-Variable design, out of this task's scope"). This task closes that lever.
Base: `2f74b7fb7` (T4.66b+c+d). CPU/NULL only, per this worktree's grant — no hardware access.

## Deliverable 1 — native-width VERIFY

### The T_pad derivation, traced exactly (as the task asked)

`GatedDeltaNetBlock._attention` (`model.py:656`): `T_pad = x.max_shape[1]`. `Tensor.max_shape`
(`tinygrad/mixin/movement.py:50-53`) is `to_max_shape(self.shape)`, and `to_max_shape`
(`tinygrad/uop/ops.py:1782`) replaces every symbolic (UOp) shape entry with `int(x.vmax)`. `x`'s token
axis, for any `spec=True` call, is `nt = v_toks.bind(n)` (pre-T4.66e) or `nt = v_toks_verify.bind(n)`
(post-T4.66e) — `UOp.bind()` returns `self.after(self.store(uval))`, an AFTER node whose own `_min_max`
(`uop/ops.py:1102`) is defined to equal the ORIGINAL Variable's declared range, regardless of the
concrete `uval` it was bound to. So `T_pad` is not "how many real tokens this call has" — it is
`v_toks.vmax` (or `v_toks_verify.vmax`), the Variable's own **declared maximum**, always, independent of
`n`. Pre-T4.66e, VERIFY and REDO both bound the SAME `v_toks` the prefill loop uses
(`UOp.variable("toks", 1, chunk_size)`, `chunk_size` defaulting to 32) — so `T_pad` was 32 for a k=3
VERIFY call that only ever has 4 real tokens, an 8x pad. Post-T4.66e, VERIFY/REDO bind a SEPARATE
`v_toks_verify = UOp.variable("toks", 1, verify_chunk)` (`model.py:1602`, `verify_chunk =
next_power2(k + 1)`, `model.py:1601`) — so `T_pad` for a VERIFY/REDO call is now `verify_chunk` (4 for
the default k=3), not 32. This is the exact mechanism `GatedDeltaNetBlock._attention`'s own per-token
loop (`for t in range(T_pad):`, `model.py:725`) and `gdn_scan_wy`'s chunked form both key off — see SS(a)
below for confirmation this actually changes what runs.

### (a) A distinct jit key falls out automatically

`Transformer.__call__` (`model.py:1185-1186`):
```python
chunk_size = next((cast(int, v.vmax) for v in tokens.uop.variables() if v.expr == "toks"), None) if is_prefill else None
return self.jit.setdefault((is_prefill, temperature is None, chunk_size, spec), TinyJit(self.forward))(...)
```
`chunk_size` here is *derived from whatever Variable named "toks" the call's own tokens tensor carries*
— not a parameter threaded down from the caller. Since VERIFY/REDO's tokens now carry `v_toks_verify`
(declared range `[1, verify_chunk]`), this expression reports `verify_chunk`, giving the key
`(True, True, verify_chunk, True)` — automatically distinct from the prefill tail's
`(True, True, chunk_size, True)` whenever `verify_chunk != chunk_size` (true for every realistic
combination: `chunk_size` defaults to 32 and `verify_chunk` is `next_power2(k+1)`, e.g. 4 for k=3; the
two could only coincide if a caller explicitly passed a `chunk_size` argument small enough to already
equal `verify_chunk`, in which case sharing the slot is harmless — they're already the same shape).
`test_llm_server.py::test_mixed_chunk_size_no_recapture_storm` already establishes this `chunk_size`
tuple slot is exactly how `generate()`'s own chunk_size variants stay separate; this is the same
mechanism, not a new one. Confirmed directly: `TestSpecWarmup.test_warmup_captures_spec_keys_when_mtp_present`
(added below) asserts `model.jit` contains a key with `chunk_size==verify_chunk` after warmup.

### (b) Own Variable range → T_pad = VERIFY_CHUNK

Follows directly from the trace above — `v_toks_verify`'s declared range is `[1, verify_chunk]`, so
`T_pad` for any call carrying it is `verify_chunk`. Confirmed directly (not just derived): new test
`TestVerifyNativeWidth.test_verify_forward_gdn_scan_uses_verify_chunk_width` monkeypatches
`GatedDeltaNetBlock._attention` to record `x.max_shape[1]` (the exact expression production code uses)
on every call, drives one full outer `speculative_generate` iteration past the prefill, and asserts
`verify_chunk` (4, for k=3) appears and 32 never does.

### (c) No bind-mismatch with the draft-pos Variables or the prefill key's v_toks — with one real exception found and fixed

**Draft-pos Variables (`v_draft_pos`):** no interaction at all — DRAFT never touches "toks" (it uses
`v_draft_pos[i]`, a completely separate name/Variable per drafted position), so nothing here changes
that mechanism.

**The prefill key's own `v_toks`:** untouched, still built exactly as before (`model.py:1582`,
unchanged range `[1, chunk_size]`). Renaming `v_toks_verify` to a name other than "toks" and re-running
the reproduction below still crashed with the identical values under the new name — proving the hazard
below is NOT about `v_toks_verify` colliding with `v_toks`'s name; it is entirely internal to
`v_toks_verify`'s own two use sites (VERIFY, REDO). Reusing the name "toks" for `v_toks_verify` is safe
and is what makes (a) above work automatically — kept.

**The real exception — found by running the gates, not anticipated up front:** giving VERIFY/REDO their
OWN, previously-cold jit key exposes a genuine T4.66a-class bind-mismatch that the shared-key,
long-since-warmed pre-T4.66e design never hit. `test_draft_reuses_schedule_across_positions` failed with
`RuntimeError: bind mismatch on toks, 1 != 3` the first time this task's Deliverable 1 change was tested
against the full gate — root-caused and fixed; see the dedicated section below, since the mechanism is
subtle, the fix took two failed attempts to find, and it's exactly the kind of thing a future task
touching this code needs to know about.

## The bind-mismatch bug: root cause, two things that didn't work, what did

**Symptom:** `test_draft_reuses_schedule_across_positions` (k=2, a partial accept on the very first outer
iteration) crashed inside DRAFT's own `.tolist()` sync, one full outer iteration after the partial
accept, with `RuntimeError: bind mismatch on toks, 1 != 3` — a scheduler-level consistency check
(`schedule/__init__.py`'s `create_linear_with_vars`) refusing to build a schedule that needs "toks" to
simultaneously mean two different concrete numbers.

**Root cause, pinned down by instrumenting `UOp.bind` and `_TinyJit.__call__`'s own `cnt` (engine/jit.py):**
a `TinyJit` key's first two calls are qualitatively different from every call after
(`_TinyJit.__call__`, `engine/jit.py:258-306`): call 1 (`cnt=0`) just runs `forward()` eagerly — no jit
bookkeeping, a completely ordinary lazy Tensor call. Call 2 (`cnt=1`) is the real CAPTURE (traces
`forward()` again, this time under `capturing`, builds `self.captured`). From call 2 onward, EVERY call
(capture or replay) returns the exact same frozen `self.captured.ret` Python object — only the
underlying buffer's data changes per replay. Pre-T4.66e, VERIFY/REDO shared the prefill tail's
already-warm key (cnt already >=1 by the time the outer loop starts, since prefill's own tail call is
what pays cnt=0), so this distinction never mattered. Post-T4.66e, `v_toks_verify`'s key starts cold
every process lifetime: VERIFY is always this key's `cnt=0` (eager) call; REDO, the first time this key
ever sees a partial accept, is `cnt=1` (the capture). On that partial accept, `verify_ids_tensor` (hence
`tok_last = verify_ids_tensor[:, m:m+1]`, kept lazy on purpose — "like generate()'s own `out`
chaining", `model.py`, ACCEPT step) is tied to VERIFY's own eager-call graph ("toks"=n), while `h_last`
(from `redo_h[:, -1:]`) is tied to REDO's own capture-call graph ("toks"=m+1) — two INDEPENDENT lazy
graphs pinned to two different concrete "toks" values. Both feed the very next outer iteration's DRAFT
chain (`tok_last` as `dtok`, `h_last` reused throughout DRAFT) — so that iteration's DRAFT sync must
resolve both together, and the scheduler correctly refuses to guess which "toks" applies.

**Two plausible fixes that do NOT work (tried and measured, not just reasoned about):**
1. **`.realize()` the lazy slice before REDO runs.** Does not help: `Tensor.realize()`
   (`tensor.py:420-425`) calls `run_linear(...)` (dispatches the compute) but never reassigns `self.uop`
   — confirmed directly by printing `tok_last.uop.has_buffer_identity()` immediately after
   `.realize()`: still `False`. The tensor's own graph, "toks" bind node and all, is unchanged, so a
   LATER, independent schedule walk (the next iteration's DRAFT sync) still re-encounters it.
2. **`.contiguous()` the lazy slice** (the pattern `h_last` itself already uses, and SPEC_NOTES.md §3's
   own fix for a related-but-different bug — silent buffer aliasing across replays, not this
   scheduler-level crash). Does not help either: the copy kernel `.contiguous()` inserts still needs
   "toks" to compute the source offset into `verify_ids_tensor`'s own `(1, verify_chunk)`-shaped buffer
   — the dependency travels straight through the copy. Confirmed by trying it: the identical crash,
   values only swapped (`bind mismatch on toks, 3 != 1`).

**What actually works:** rebuild `tok_last` from data already sitting on the host, with NO ancestry in
`verify_ids_tensor` at all — exactly the pattern the *sampled* branch already uses unconditionally
(`tok_last = Tensor([[accepted[-1]]], dtype="int32", device=dev)`). `accepted[-1]` was already computed
a few lines earlier (host-side) and always equals `verify_ids[m]`/what the lazy slice would have held.
Applied only on the greedy, partial-accept branch (`model.py:1848`, `if greedy: tok_last =
Tensor([[accepted[-1]]], dtype="int32", device=dev)`) — the far more common full-accept path's `tok_last`
stays exactly as lazy as it always was (both `tok_last` and `h_last` there come from the SAME VERIFY
call, so they can never disagree). `h_last` needs no equivalent fix: once REDO's capture exists, every
future VERIFY/REDO call (this iteration's REDO included) returns that same captured object, and it
always gets drained by the very next DRAFT sync (a required `mtp_head.draft` input) before any later
VERIFY/REDO call could rebind it again — the same pattern that already makes `h_last` safe pre-T4.66e,
and safe here from the second partial accept of this key's lifetime onward. `tok_last` was the one value
this pattern didn't cover, because it comes from a DIFFERENT call (VERIFY) than `h_last` (REDO). Cost:
one tiny `(1,1)` host->device upload, only on the partial-accept branch (already paying ~1.8s for the
REDO forward itself, per SPEC_FIXES_NOTES.md's own measurement — this is noise against that).

Full reasoning, with the exact line numbers, is written in-line at both `v_toks_verify`'s own definition
(`model.py:1583-1602`) and at the fix site (`model.py:1812-1848`).

## Deliverable 2 — spec-path warmup

`Transformer.warmup()` (`model.py:1369-1392`) only ever drove `generate()`'s own jit keys — the T4.65 gap
SPEC_PROFILE_NOTES.md/SPEC_FIXES_NOTES.md both flagged (recommended fix in T4.72), since
`speculative_generate`'s spec=True keys (the prefill tail's, and now VERIFY/REDO's own dedicated
`verify_chunk` one) were only ever first touched at REQUEST time. Added: when `self.mtp_head is not
None`, `for _ in range(2): list(zip(range(2), self.speculative_generate([0])))` — mirroring the
EXISTING `generate()` loop immediately above it, both in shape and in reason: per the `cnt`
mechanics above, a key's first-ever call doesn't actually capture anything — calling
`speculative_generate` only ONCE would leave every spec key exactly as un-captured as never warming it
at all, silently moving the real capture cost onto the first live request. Two calls, two tokens each
(prefill anchor + one full outer iteration): the prefill-tail key is hit once per call (so twice total,
guaranteeing cnt reaches the capture on the second); the VERIFY/REDO key is hit at least once per outer
iteration, and either a partial accept within the first call's own iteration captures it right there
(VERIFY then REDO = 2 calls in one pass) or, on a full accept, the second call's own VERIFY provides the
second hit instead — either way, by the time `warmup()` returns, `model.jit[key].captured` is set for
every spec key. Confirmed directly by `TestSpecWarmup.test_warmup_captures_spec_keys_when_mtp_present`.
REDO needed no separate warming beyond this: it always reuses VERIFY's exact jit key (see
`speculative_generate`'s own REDO comment, `model.py:1798-1799`), so once that key has been called twice
by any combination of VERIFY/REDO calls, both replay the same captured slot. 4 tokens total is still
"a few" — the same order of magnitude as `generate()`'s own 4-call warmup loop just above it.

`TestSpecWarmup.test_warmup_without_mtp_head_adds_no_spec_keys` guards the other direction: a plain
`Transformer` (`mtp_head` unset, matching every pre-T4.66e caller) must see zero spec=True keys after
warmup — the new branch is fully inert when MTP was never turned on.

## Deliverable 3 — tests

All in `test/unit/test_spec_decode.py`:
- Every pre-existing test in the file (token-identity, forced-mismatch, forced-perfect,
  schedule-reuse, state-integrity, sampled, GDN-specific, `SPEC_TRACE`, `MTPDraftDeviceLocalEmbed`) stays
  green unchanged — including `test_forced_partial_accept_matches_generate_gdn`, the GDN-specific
  forced-partial-accept test, and `test_matches_generate_several_prompts_and_k`/`_gdn` (greedy spec ==
  plain, real random drafts, both k=1 and k=3).
- `TestVerifyNativeWidth.test_verify_forward_gdn_scan_uses_verify_chunk_width` — the T_pad shape
  assertion the task asked for, via a monkeypatch spy on `GatedDeltaNetBlock._attention` (no existing
  hook exposes T_pad directly; `gdn_last_scan_impl` only records LOOP-vs-WY, not the width) recording
  `x.max_shape[1]` — the identical expression production code uses. Runs under `Context(GDN_CHUNK=32)` so
  the CPU test device's own `gdn_chunk_for()` auto-narrowing (T4.55: 1 off METAL/NV/CUDA) doesn't collapse
  `chunk_size` down to `verify_chunk` and hide the very gap under test.
- `TestSpecWarmup.test_warmup_captures_spec_keys_when_mtp_present` and
  `test_warmup_without_mtp_head_adds_no_spec_keys` — the warmup assertion the task asked for (jit dict
  keys populated, and actually `captured`, when `mtp_head` is present; nothing added when it isn't).

Gates run (this worktree's venv, CPU/NULL only): `CHECK_OOB=1 DEV=CPU pytest test/unit/test_spec_decode.py
test/unit/test_gdn_scan_parity.py test/unit/test_llm_server.py -x -q` -> 59 passed (56 pre-existing + 3
new); `SPEC=2 DEV=NULL pytest test/null/ -x -q` -> 1532 passed, 81 skipped, 16 xfailed, 2 subtests passed
(identical to T4.66d's own baseline count — no regression); `mypy tinygrad/` -> no issues found in 219
source files; `ruff check .` -> all checks passed.

## Predicted per-iteration budget (k=3, VERIFY_CHUNK=4)

Building on SPEC_FIXES_NOTES.md's own post-T4.66d prediction table (its own numbers, not re-derived
here) and applying only this task's own lever (VERIFY/REDO's GDN-scan width, `T_pad`: 32 -> 4):

| phase | post-T4.66d (measured/predicted) | post-T4.66e (predicted) |
|---|---|---|
| `draft_ms` (k=3) | ~tens of ms (T4.66d's own residual, unaffected by this task) | unchanged -- this task never touches DRAFT |
| `verify_dispatch_ms` | ~800 (T4.66d's own carried-forward number; SPEC_PROFILE_NOTES.md's candidate #2 called this "meaningful, likely hundreds of ms, probably not the whole gap alone" -- i.e. NOT 100% T_pad-scan cost) | **~100-800**, see the math and caveat below |
| `accept_ms` | ~107 | unchanged -- this task never touches ACCEPT's own mechanics (VERIFY's forward gets cheaper, but `accept_ms` measures the SYNC/compare step around it, not remeasured here) |
| `state_assign_ms` (partial accept only) | ~1,800 (T4.66d's own CHECKPOINT+REDO number) | ~1,800 + a sub-millisecond `(1,1)` host upload (the bind-mismatch fix's own tiny cost -- noise) |
| **total/iter (full accept)** | **~900-950** | **~200-900**, see caveat |

**The verify_dispatch_ms math, and its honest limit:** the task's own formula, `800ms x
VERIFY_CHUNK/32 = 800 x 4/32 = 100ms`, is an OPTIMISTIC bound: it assumes the entire measured
`verify_dispatch_ms` scales linearly with `T_pad`, which would only be true if the GDN scan's
per-token loop were the WHOLE cost. SPEC_PROFILE_NOTES.md's own candidate #2 write-up already
disclaims this ("probably not the whole gap alone") — some of `verify_dispatch_ms` is candidate #1's
per-forward cross-device hops (shared with plain decode, `T_pad`-independent), jit dispatch overhead, and
CHECKPOINT clone construction (T4.66d's own addition, sized off the NATIVE state, not `T_pad`) — none of
which shrink with `verify_chunk`. So the honest predicted range is **100ms (if the scan totally
dominates) to 800ms (if it barely contributes)** -- the true number depends on how much of that 800ms
this fork's real target checkpoint (Qwen3.8-27B, 48 GDN blocks per SPEC_FIXES_NOTES.md's own
header-derived count) actually spends inside the per-token loop specifically, which only a hardware trace
(below) can settle. Given 48 GDN blocks each running a T_pad-deep unrolled Python loop is a lot of
repeated small work, the scan plausibly dominates and the real number should land closer to the lower
end -- but this is a prediction, not a measurement, exactly like SPEC_FIXES_NOTES.md's own
`accept_ms`/CHECKPOINT-overhead entries were.

**Why `next_power2(k+1)` and not a hard floor of 8 (or any other fixed width):** no alignment
requirement in this codebase forces one. `GatedDeltaNetBlock._attention`'s per-token loop
(`for t in range(T_pad):`) has no divisibility constraint on `T_pad` at all. `gdn_scan_wy`'s triangular
inverse (`_gdn_tri_inverse`, `model.py:54-82`) is built via Neumann-series doubling over
`(c-1).bit_length()` steps and its own docstring states this holds "regardless of whether C is a power
of 2" — no padding or power-of-two bookkeeping needed. The one real `%32` alignment assert in this
codebase, `kernels/amd.py:441` (`assert T_pad % BLOCK_M == 0`), belongs to the fused RDNA3 (AMD) scan
kernel, gated by `amd_custom_kernels_supported()` (`kernels/amd.py:26-34`), which returns `False` for
every device whose name doesn't start with `"AMD"` — unreachable on METAL/NV (this fork's actual
target) and moot regardless, since AMD is fully descoped for this fork (CLAUDE.md). `next_power2` (from
`tinygrad/helpers.py`, already used elsewhere in the codebase — `ops_qcom.py`, `mlxdev.py` — reused here
rather than hand-rolled) is a tidy, conventional choice, not a forced one; for k=3 it gives exactly 4,
matching the task's own worked example.

## Hardware invocation for the next round

Same shape as SPEC_PROFILE_NOTES.md §9 / SPEC_FIXES_NOTES.md's own "Hardware invocation", re-run after
this task's changes:

```
SPEC_TRACE=1 SPEC_STATS=1 DEV=NV ... python -m tinygrad.llm --serve 8081 --mtp ...
```

against the same k=3 prompt used for T4.66b/c/d's own measurements. Read the printed `[SPEC_TRACE]`
lines: `verify_dispatch_ms` should drop from its old ~800ms baseline toward somewhere in the 100-800ms
range predicted above -- the closer to 100ms, the more of the old cost really was the T_pad-scan share;
if it barely moves, the per-forward cross-device hops (candidate #1's OTHER half — the ones plain decode
also pays, unrelated to MTP) or the CHECKPOINT clone dominate instead, both out of this task's own scope.
`draft_ms`/`accept_ms`/`state_assign_ms` should be unchanged from T4.66d's own measurements (this task
never touches those phases). A one-time confirmation worth taking alongside the trace: print
`model.jit.keys()` right after the server's own warmup call (or watch for a mid-request capture stall on
the very first speculative request) to confirm `(True, True, verify_chunk, True)` is already captured
before the first real request lands, per Deliverable 2.
