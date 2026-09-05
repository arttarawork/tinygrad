# T4.73d — WY chunked GDN scan: content x length residual (root cause + fix)

Base: fork master `2d38cada5` + T4.73c (`b9b316e3b`, `37775d769`). Branch
`task/T4.73d-wy-content-residual`. CPU/NULL only throughout (no METAL/NV touched from this worktree;
all hardware runs were driven by the coordinator on the pooled METAL+NV server and handed back as
`WY_TRACE`/`WY_DUMP_DIR`/`WY_DUMP_AMAX` captures). Continues T4.73c's fix (a_bar fp32 underflow at a single
chunk, head 42) with the LAST residual: a specific paragraph repeated many times still NaN-floods
`GDN_SCAN_IMPL=2` (WY) past ~130-180 chunks, while `GDN_SCAN_IMPL=1` (the loop) handles the same prompt
cleanly.

## The established fact (recap, T4.73d PHASE 1/2A)

Hardware (pooled METAL+NV, `GDN_SCAN_IMPL=2`, `GDN_CHUNK=32`): one 265-char paragraph ("The history of
metallurgy spans thousands of years...") repeated x128 (5,782 tokens, ~181 chunks) NaN-floods at the first
decode step; x96 (~136 chunks) is clean. The loop impl handles the full x128 (and longer) prompt fine.

Per-chunk instrumentation (`WY_TRACE=2`, this task's own PHASE 1 deliverable) traced the real failure to
**block 44** of qwen3.8-27B, NOT the block first seen non-finite (block 45, which only inherits block 44's
already-broken output — block 44 itself overflows straight to `Inf` with zero `NaN`s of its own, which is
why PHASE 1's nan-count-only trigger missed it; fixed in PHASE 2A, see below). Block 44's
`recurrent_state` abs-max on the x128 prompt: chunk 1 `2.27` → chunk 2 `31.3` → chunk 4 `664` → chunk 8
`4.9e4` → chunk 16 `1.8e8` → chunk 32 `1.2e13` → chunk 64 `3.1e22` → chunk 96 `1.6e29` → chunk 128 `3.1e37`
→ chunk 131 `1.41e38` (finite, just under fp32 max) → chunk 132 `Inf`. The **same content run under
`GDN_SCAN_IMPL=1`** keeps block 44's abs-max flat at `0.67-0.71` across every chunk — a contraction, as the
per-token delta rule always is for `beta` in `(0,1)`. Baseline (unrelated short prompts): block 44's abs-max
is `~0.65`. This rules out "slow drift in a legitimately large state" (`state_in` itself is never huge going
into any single chunk until very late — see below) and confirms the mechanism is specific to the WY chunked
*formulation*, not real model dynamics.

### PHASE 2A: instrument fix (non-finite trigger, WY_DUMP_AMAX)

`_wy_chunk_trace`'s original culprit trigger checked NaN count only (`s_nan or c_nan`). A block that
overflows straight to `Inf` (zero `NaN`s of its own) never tripped it — only the first DOWNSTREAM block to
turn that `Inf` into actual `NaN`s got reported. Fixed: `_wy_chunk_fp` now returns `(nan, inf, amax)`
instead of `(nan, amax)`, and the trigger fires on `nan or inf` for either tensor. Also added
`WY_DUMP_AMAX=<threshold>`: a second, independent, latched trigger that fires on the first chunk where some
block's state/conv abs-max crosses the threshold **while still fully finite** — the non-finite trigger can
only ever capture a block *after* it breaks (or a downstream victim); this one captures the still-finite
inputs a per-chunk amplification needs to be measured from. Dumps to
`preoverflow_chunk<idx>_blk<i>.safetensors` (`culprit_*` stays non-finite-only).

## Root cause

`GDN_DUMP_AMAX=5` on `extra/t473d_payloads/p8k_x4.json` (the paragraph repeated x4, 202 tokens) caught
block 44 crossing the threshold at **chunk 3** (`preoverflow_chunk3_blk44.safetensors`, `state_in` abs-max
`31.29` — chunk 2's real output — fully finite; `recurrent_state_out` abs-max `217.09` — chunk 3's real
output, a **~x7 gain in one chunk**). Only **head 25** of 48 has any real magnitude in this dump (every
other head's `state_in`/`recurrent_state_out` abs-max is well under 1) — the same "one outlier head, whole
head at once" pattern as T4.73c's head 42, but a different mechanism.

Loading the dump into `extra/wy_content_amplification_repro.py` and calling `gdn_scan_wy` directly (no
device, JIT, or full-model machinery) reproduces the amplification on plain CPU immediately: from
`state_in` abs-max `31.3`, `gdn_scan_wy` (pre-fix) gives abs-max `268.9`; the sequential loop, from the
*same* `state_in`, gives `1.20`. (The real hardware value was `217.1`, not `268.9` — expected, not a
discrepancy: this computation is, per below, catastrophically ill-conditioned, so CPU vs. NV/METAL kernel
fusion/summation-order differences shift the exact garbage value while both are equally wrong relative to
the loop's `1.20`.)

Head 25's real inputs for this chunk: `beta` uniformly high (`0.87`-`0.99` across all 32 steps — see the
literal values hardcoded in the new regression test) and `k` (keys) extremely near-collinear (real pairwise
cosine similarity: mean `0.96`, max `0.999`, over the 32x32 off-diagonal pairs) — exactly what "one
paragraph repeated" produces: nearly every token's key vector points almost the same direction within a
32-token window. Both ingredients matter (see "Ruled out" below).

### Isolating the exact expression

Recomputing every `gdn_scan_wy` intermediate (non-kda path, `tinygrad/llm/model.py`) in float64 (exact
reference) vs. float32 (production dtype) on head 25's real `state_in`/`q`/`k`/`v`/`beta`/`alpha`:

| intermediate | max&#124;f64&#124; | max&#124;f32&#124; | max&#124;f32-f64&#124; |
|---|---|---|---|
| `a_bar` | 0.9999 | 0.9999 | 3.0e-8 |
| `m = beta*kkt` | 0.980 | 0.980 | 3.8e-7 |
| **`_gdn_tri_inverse(m)`** | **1.0** | **4.2** | **4.2** |
| `rhs` | 31.2 | 31.2 | 8.4e-6 |
| `w = tri_inverse(m) @ rhs` | 29.0 | 120.9 | 120.9 |
| `final_state` | 1.20 | 69.7 | 69.1 |

Every intermediate matches float64 to plain float32 rounding (~1e-6 to 1e-8) — **except** `_gdn_tri_inverse`
itself, which the *exact* (float64, and independently a float64 forward-substitution reference) computation
puts at `max|.|=1.0`, but the production float32 doubling code puts at `max|.|=4.2` — already wrong before
anything downstream touches it. `w` and `final_state` inherit that error multiplicatively and it's gone by
the time it reaches the recurrent state.

### The mechanism: Neumann-series doubling's catastrophic cancellation

`_gdn_tri_inverse` (pre-fix) computed `(I+m)^-1 = sum_{i=0}^{C-1} n^i` (`n := -m`, nilpotent since `n^C==0`
exactly for a `C`-wide strictly-lower-triangular matrix) via **doubling**: `(P, n^k) -> (P + n^k@P, n^k@n^k)`
for `ceil(log2(C))` steps, terminating because `n^k` becomes exactly `0` once `k>=C`. This is *exact in
infinite precision* — T4.69a's original derivation is correct — but tracing the actual float32 values
through head 25's real (`max|m|=0.985`) matrix shows why it isn't safe in finite precision:

```
step 1/5: max|n_pow|=28        max|p|=1        (n^2, partial sum through n^1)
step 2/5: max|n_pow|=3,403     max|p|=358      (n^4, partial sum through n^3)
step 3/5: max|n_pow|=1,377,330 max|p|=335,274  (n^8, partial sum through n^7)
step 4/5: max|n_pow|=68,166,000 max|p|=35,035,700  (n^16, partial sum through n^15)
step 5/5: max|n_pow|=0 (exact, n^32==0)  max|p|=1.0 (f64) / 4.0 (f32)  <- final answer
```

Repeated squaring of a matrix whose entries are close to 1 in magnitude (high `beta` times near-1
key-cosine-similarity) drives the intermediate power `n_pow` up by 1-2 orders of magnitude *per step* —
here to `6.8e7` after 4 steps — even though the **true final answer is `~1.0`**: nilpotency forces total
cancellation on the last step, and float32's ~7 significant decimal digits cannot represent a
cancellation from `3.5e7` down to `1.0` (that needs ~8 digits just for the leading digit of the answer to
be right). Float64 has enough digits (15-16) to survive the same cancellation at this magnitude — confirmed
above (float64 doubling matches the exact forward-substitution reference) — but float64 is not viable for
production: `tinygrad/renderer/cstyle.py`'s `MetalRenderer.supported_dtypes()` explicitly excludes
`dtypes.double`, and METAL is one of the two target devices this whole optimization exists for (same
constraint FIXNOTES_T473C.md already hit and ruled out for a different part of this same function family).

## The fix

Replace the doubling algorithm with **block-recursive halving** — the *other* exact way to invert a
unit-lower-triangular matrix, and (per `_gdn_tri_inverse`'s own pre-existing docstring) literally the
**first draft T4.69a tried**, before switching to doubling purely to lower the UOp/kernel count (a real but,
compared to a hard correctness bug, strictly smaller concern):

```
(I+m)^-1 = [[Inv11, 0], [-Inv22 @ M21 @ Inv11, Inv22]]     (m = [[M11, 0], [M21, M22]], 2x2 block split)
Inv11 := (I+M11)^-1, Inv22 := (I+M22)^-1                    (recursive, same identity, down to C=1: (I+0)^-1=I)
```

This is exact for any unit-lower-triangular matrix by the standard block-triangular-inverse identity, and
critically **never forms an intermediate larger than the true (bounded) answer**: `Inv11`/`Inv22` are
themselves valid, bounded triangular inverses by induction, and `Inv21` is a product of already-bounded
matrices — there is no large-magnitude value to catastrophically cancel away. Verified on head 25's real
`m`: halving (float32) matches the float64 forward-substitution reference to `1.5e-7` (plain float32
machine precision), vs. doubling's `4.0` absolute error on a true value of `1.0`.

`c==1` (no `T=1` GDN_CHUNK actually reaches this — `run_scan` gates `T_pad==1` to the loop — but a chunk can
recurse down to 1-wide leaves internally) returns `Tensor.ones_like(m)`: `m`'s diagonal (hence every 1x1
leaf) is exactly `0` by the strictly-lower-triangular precondition, so `(I+0)^-1 = I`, a 1x1 identity.
Non-power-of-2 `c` splits as evenly as possible each level (`lo = c//2`) — handled exactly, same as the
retired doubling code's own claim ("no padding or odd/even-split bookkeeping needed"), though in production
`c` is always the concrete `GDN_CHUNK` (32) since `T_pad` pads every chunk (including the last, partial one)
to the full declared width.

Applies uniformly to both `gdn_scan_wy` call sites (kda and non-kda) since it's a property of the shared
`_gdn_tri_inverse` helper, not something specific to either branch — no conditional/targeted gating needed,
since halving is provably at least as accurate as doubling everywhere (it's the same exact math via a
better-conditioned route), not a special-case-only patch.

**Known tradeoff, not addressed here**: more total kernels (`O(C)` recursive-call nodes, each its own small
matmuls/concats) than doubling's `O(log2 C)` big uniform matmuls — exactly the concern that made T4.69a
choose doubling originally. Not re-measured in this task (CPU/NULL-only, no BEAM/perf harness run) — flagged
for the coordinator's hardware validation pass. If it regresses BEAM/compile time meaningfully at
`GDN_CHUNK=32`/`GDN_HEAD_GROUPS` in production, the next lever is a hybrid (halving only below some fixed,
small block size, doubling above it, e.g. halve down to 8-wide leaves then finish with 3 doubling steps) —
not implemented since the plain full-halving fix above is already verified exact and no perf regression has
actually been measured yet.

## Ruled out

- **Slow single-block state accumulation**: block 44's own `state_in` was checked directly from the dump —
  fully finite, abs-max `31.3` (chunk 2's real output), nowhere near fp32 range. The growth is a genuine
  per-chunk multiplicative gain (`~x7` in the one chunk captured), not a large value that was already
  present and merely carried forward.
- **A single expression in isolation, at a zero or arbitrary-magnitude carried-in state**: exploratory CPU
  probes (see the PHASE 2A commit) found a *single* `gdn_scan_wy` call from state0=0, or from a large FIXED
  (not accumulated) state0 up to `1e10`, stays exact vs. the loop even at `beta` up to `0.999`, for
  *randomly generated* (non-collinear) keys. The bug needed the real (or realistically reconstructed)
  near-collinear key structure to manifest, which is exactly what `_gdn_tri_inverse`'s `m` matrix depends
  on and nothing else in the formula does.
- **Chaining identical chunk content**: repeating the exact same synthetic chunk every "chunk" (a naive
  model of "steady repeated-paragraph regime") did NOT reproduce growth even at `beta=0.999` — the real
  bug needs chunk-to-chunk content variation (real consecutive 32-token windows of a repeating paragraph
  are highly self-similar but never bit-identical), consistent with the tri-inverse producing a
  content-dependent (not just magnitude-dependent) error each chunk.
- **float64 promotion**: ruled out for production (METAL has no `dtypes.double`), same as T4.73c.

## Verification

`extra/wy_content_amplification_repro.py`: loads the real dump
(`extra/t473d_payloads/dumps/preoverflow_chunk3_blk44.safetensors`, **not committed** — a ~9MB run-specific
artifact, same convention as `extra/blk0_real.safetensors` before it; regenerate via a traced hardware/local
run with `WY_TRACE=2 WY_DUMP_DIR=... WY_DUMP_AMAX=5` on the x4-repeat prompt), reproduces the amplification
end-to-end on plain CPU with the retired doubling code inlined for direct side-by-side comparison, and
confirms the fix matches the loop from the real `state_in`. Output (abbreviated):

```
current (halving) gdn_scan_wy: absmax=1.20169   head 25 absmax=1.20169
retired (doubling) gdn_scan_wy: absmax=268.948   head 25 absmax=268.948   (real hardware observed 217.092)
loop (ground truth) from the SAME state_in: absmax=1.20169
...
forward-substitution (f64 ground truth): max|inv|=1
halving (CURRENT fix, f32):  max|inv|=1        max|diff vs f64 ref|=1.5e-07
doubling (RETIRED, f32):     max|inv|=6.3       max|diff vs f64 ref|=6.3        <- catastrophic cancellation
FIX CONFIRMED: current gdn_scan_wy matches the loop from this real, previously-amplifying state_in.
```

**New regression tests**: `test/unit/test_gdn_scan_parity.py::TestGDNScanHighBetaCollinearAmplification`
hardcodes head 25's real 32 `beta` values (chunk 3, blk44, the x4 prompt) onto synthetic near-collinear keys
(a base unit vector plus small per-step noise at the real `K=128` dim, reproducing the real ~0.96 mean
pairwise cosine similarity without embedding the literal real key matrix — `_gdn_tri_inverse` only ever
sees `m = beta * strictly_lower(k @ k.T)`, never `k` itself, so matching `m`'s statistics is what matters).
No GGUF or dump file needed in CI (same minimal-real-numbers pattern as T4.73c's
`TestGDNScanRealWeightUnderflow`, which hardcoded 2 scalars). Two tests:
- `test_tri_inverse_matches_forward_substitution_reference`: `_gdn_tri_inverse(m)` vs. an independent
  float64 forward-substitution ground truth, `rtol=atol=1e-3`.
- `test_full_scan_matches_loop_from_zero_state`: full `gdn_scan_wy` vs. the sequential loop, from a **zero**
  carried-in state (the bug is visible even here — `rhs` collapses to `beta*v` but the tri-inverse it's
  multiplied by is already wrong regardless of the carried state), `rtol=atol=1e-2`.

Manually verified both ways: `git stash push -- tinygrad/llm/model.py` (reverting only the fix, keeping the
new tests) → **both FAIL** (max absolute difference `1.85` / relative difference `>5000x` on the tri-inverse
test); `git stash pop` (fix restored) → **both PASS**.

**Gates** (`CHECK_OOB=1 DEV=CPU`, venv, from the worktree):
- `pytest test/unit/test_gdn_scan_parity.py test/unit/test_attention.py test/unit/test_llm_server.py -x -q`
  — all green (see the commit message for the exact pass/skip counts).
- `mypy tinygrad/` — clean (219 files).
- `ruff check .` — clean.
- `pylint --disable=all -e W0311 -e C0303 --jobs=0 --indent-string='  ' --recursive=y .` (CI whitespace
  lint) — clean on every file this task touches (pre-existing violations in the untracked, un-committed
  `extra/t473d_payloads/` staging directory are out of scope — not touched by this change, not part of any
  commit).

## Not run (hardware validation, coordinator's side)

Full-model, multi-hundred-chunk validation on the real x128 prompt (and the originally-reported x96-clean /
x128-broken boundary) was NOT re-run from this worktree (CPU/NULL-only rule; the pooled server was up and
serving the standing Q8 model throughout this task). The coordinator's own re-run of the traced x128 (or
longer) prompt against this fix is the remaining validation step — expected result: block 44's
`recurrent_state` abs-max trajectory stays flat (matching the loop's `0.67-0.71` baseline) instead of the
pre-fix exponential growth to `Inf` at chunk 132.
