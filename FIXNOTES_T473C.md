# T4.73c — WY chunked GDN scan: NaN blowup at real trained weights (root cause + fix)

Base: fork master `2d38cada5`. Branch `task/T4.73c-wy-numerics`. CPU/NULL only throughout (no METAL/NV
touched). Continues T4.73's finding ("WY graph bug") with a hardware-confirmed real-weight repro and a fix.

## The established fact (recap)

`extra/bug1_gguf_ab.py`, run against the real `Qwen3.8-27B-Q8_0.gguf` on pure CPU:
`GDN_SCAN_IMPL=2` (`gdn_scan_wy`) → `recurrent_state` NaN starting at blk0, **exactly one head's worth**
(16384 = 786432/48 elements) → cascades downstream, argmax collapses to a constant out-of-vocab id.
`GDN_SCAN_IMPL=1` (the per-token loop, identical math run sequentially) → clean. Same chunk width (32),
same inputs, same fp32 scan dtype. A separate random-init weight-scale sweep found a NaN cliff around
scale 0.8 that hits **both** impls (a different, mundane instability, not this bug) — the real-weight
failure is **WY-only**, pointing at WY's own algebra, not the recurrence itself.

## Root cause

`gdn_scan_wy` (`tinygrad/llm/model.py`) computes, per chunk (`Abar_i` = cumulative product of per-step
decay `alpha` from the chunk's start through step `i` inclusive):

```
a_bar   = alpha.cumprod(axis=2)                    # Abar_i
v_tilde = v / a_bar                                # <-- the culprit expression
rhs     = beta * (v_tilde - k @ state.T)
u       = tri_inverse(m) @ rhs
out     = a_bar * (q @ state.T + qk @ u)
final_state = a_bar[-1] * (state + u.T @ k)
```

`v_tilde = v / a_bar` is mathematically the "decay-normalize the state, do a plain delta rule, fold the
decay back in by multiplying by `a_bar` again" trick (see the derivation still in `gdn_scan_wy`'s
docstring). It is exact in infinite precision, because `a_bar` cancels: `a_bar_i * (v_j / a_bar_j)`
recovers a bounded value whenever the two `a_bar`s are close. It is **not exact in float32** once `a_bar`
underflows to a literal `0.0`: `v / 0.0` is `Inf` (or overflows to `Inf` slightly before hitting exact 0,
since the quotient exceeds float32's range first), and `0.0 * Inf` (the later re-multiply, or the causal
mask itself — see below) is `NaN`, not `0`. The per-token loop never forms `1/a_bar` — it only ever does
`state * alpha`, which safely shrinks toward 0 — so it stays clean regardless of how small `alpha` gets.

### Real numbers (extracted from `blk.0` of the real GGUF, see Extraction below)

Non-kda architecture (`arch=qwen35`): `alpha` is a **per-head scalar** (last dim 1, broadcasts across all
128 value-channels of a head) — `ssm_a` has shape `(48,)`, one learned scale per head, not per-channel.
Driving the real embedded prompt (ids 1000..1031) through blk0's real weights at chunk 32
(`extra/wy_numerics_repro.py`):

- `alpha` (per-step decay) ranges from `5.44e-4` to `1.0` across all 48 heads/32 steps.
- **Head 42** is the outlier: learned `ssm_a[42] = -0.337646`, `ssm_dt.bias[42] = 14.875`. The
  next-most-aggressive head (26) has `ssm_a = -0.131226` — under half the magnitude. `softplus(dt_bias) *
  ssm_a` alone (i.e. even ignoring the input-dependent term) gives head 42 a baseline per-step
  `alpha ≈ 6.6e-3`; real activations push the observed minimum to `5.44e-4`.
- Head 42's cumulative product (`a_bar`) underflows to an **exact** float32 `0.0` at chunk position 17 of
  32: `a_bar[16] = 1.121e-44` (a denormal) → `a_bar[17] = 0.0`.
- `v_tilde` at head 42 is non-finite for 2171/4096 (t, channel) entries from that point on (some entries
  turn `Inf` slightly before the exact-zero point too, once the quotient alone exceeds float32's ~3.4e38
  max).
- `recurrent_state` ends up NaN in **all 16384** entries of head 42 (all of `V×K = 128×128`), matching
  `bug1_gguf_ab.py`'s hardware finding exactly, both at `GDN_HEAD_GROUPS=1` and at the production default
  (`GDN_HEAD_GROUPS=0` → auto → 2 groups of 24 for 48 heads).

### Why ONE head, and why the whole (32, 5120) output, not just the tail

`alpha` is a per-**head** scalar for this (non-kda) architecture, so a head's decay is identical across
all its value-channels — when `a_bar` underflows, it does so for the **entire head at once** (all 128
channels), not a subset. Different heads specialize in different memory horizons via their own learned
`ssm_a`/`ssm_dt.bias`; head 42 is trained to forget aggressively enough that its cumulative decay crosses
float32's representable range within a single 32-token chunk, while every other head (next-closest:
`ssm_a` less than half the magnitude) stays many orders of magnitude away from that edge over the same 32
steps. This is real, learned parameter diversity, not an artifact.

The NaN reaches **every** output position (not just the ones at/after the underflow), because the causal
masks in `gdn_scan_wy` (`* strict_lower`, `* lower_incl`) are plain elementwise multiplies by a 0/1 mask,
not a structural slice. `_gdn_tri_inverse(m)`'s off-causal entries are exact `0.0` (proven bit-exact in its
own docstring), which is correct — but `0.0 * Inf = NaN`, not `0.0`, in IEEE-754. So the very first matmul
that mixes an `Inf`-containing row of `rhs` through `_gdn_tri_inverse(m)`'s exactly-zero off-causal entries
already turns **every** row of the pseudo-value matrix `u` non-finite, including rows that are causally
"before" the underflow and mathematically should be completely unaffected. From there it propagates through
`qk @ u` and the shared `ssm_out` projection (which mixes all 48 heads' channels at every time position) to
the entire `(32, 5120)` block output. Confirmed directly: pre-fix, `gdn_scan_wy`'s own `out` (before
`ssm_norm`/`ssm_out`) is non-finite at head 42, **all 32** time positions, not just positions ≥17.

### Ruled out

- **`_gdn_tri_inverse`'s Neumann doubling**: independently verified well-conditioned on the exact real
  `beta`/`k` values with a synthetic, non-decay-involving `rhs` — `max|inv| = 1.0`, finite. Not the culprit.
- **`pad_to` / padded-tail garbage**: the repro uses `T=32` real (non-padded) tokens with no symbolic
  chunking involved at all, and reproduces the bug identically — padding is not a factor.
- **float64 promotion** (a candidate fix): checked `tinygrad/renderer/cstyle.py`'s
  `MetalRenderer.supported_dtypes()` — it explicitly excludes `dtypes.double`. Metal is one of the two
  target devices this optimization exists for, so a float64-based fix is a non-starter for production,
  independent of whether it would work on CPU.

## The fix (non-kda only; kda branch untouched)

Never form `a_bar` as a lone divisor. Keep decay in **log space** and only ever use it two ways:

1. A **pairwise ratio** `exp(G_i − G_j)` for `i ≥ j` (bounded in `(0, 1]` — `G`, the cumulative log-decay,
   is non-increasing under decay — computed directly from the log-space difference, never as a quotient of
   two independently-exponentiated numbers). The invalid (`j > i`) direction is forced to **exactly** `0`
   by masking the exponent to `-1e30` **before** `exp`, not by clamping the ratio afterward — this keeps
   `_gdn_tri_inverse`'s strictly-lower-triangular precondition exact, unchanged.
2. The **absolute** `a_bar = exp(G_i)` used strictly as a **multiplier** (never a divisor) — safe to
   underflow to `0.0`, which now correctly means "fully decayed", never `NaN`.

This is an exact reformulation, not an approximation: folding decay into the coupling matrix instead of
into `v` (and undoing it later) is the standard chunked-linear-attention/DeltaNet trick. Hand-verified (and
confirmed in pure numpy, float64 vs. float32, before touching tinygrad) that the new pseudo-value `w`
relates to the old tilde-form pseudo-value `u` by `w_j = Abar_j * u_j` pointwise, so both forms agree
exactly in infinite precision:

```
g            = log(max(alpha, 1e-30))                          # per-step, always finite
G            = cumsum(g)                                       # cumulative log-decay
ratio_incl   = where(j<=i, exp(G_i-G_j), 0)                     # masked BEFORE exp
ratio_strict = where(j<i,  exp(G_i-G_j), 0)
a_bar        = exp(G)                                           # multiplier only, never divided

kkt   = (k @ k.T) * ratio_strict                                 # decay folded in directly (was undecayed)
m     = beta * kkt
rhs   = beta * (v - a_bar * (k @ state.T))                       # v used RAW -- no division
w     = tri_inverse(m) @ rhs
qk    = (q @ k.T) * ratio_incl                                   # decay folded in directly
out   = a_bar * (q @ state.T) + qk @ w                           # a_bar multiplies ONLY the S0 carry term
final_state = a_bar[-1] * state + (ratio_incl[-1,:] * w).T @ k   # same split
```

kda (per-value-channel `alpha`) is left **byte-identical** to before: its per-channel decay would need a
genuinely per-channel `(T,T,V)` coupling/tri-solve to fold in this way (untested cost), and it is not
established broken (`test_attention.py`'s `test_varied_chunk_sizes_match_decode(kda=True)` only exercises
small random weights, which don't hit this failure mode). The branch is on `alpha.shape[-1] == 1`.

### A bug found and fixed while getting the tests green

First implementation clamped the pairwise log-diff to `<= 0` (`diff.minimum(0)`), reasoning that decay
means `alpha <= 1` so `G` is non-increasing. That's true for the **real** model (all 48 measured heads had
`alpha <= 1`) but **not** guaranteed by `test_gdn_scan_parity.py`'s random-init weights — `ssm_a` there is
drawn from `randn() * 0.1`, unconstrained in sign, so some random heads have `alpha > 1` (genuine growth).
The clamp silently capped that growth, diverging from the loop's ground truth by up to ~0.4% of state
magnitude — caught by the existing `test_small_chunk_sizes_match_sequential` /
`test_large_chunk_matches_sequential` / `test_continuation_from_nonzero_start_pos` tests going red (0.13%
to 6.5% of elements outside `rtol=atol=1e-4`, confirmed via a captured-inputs isolation script that a
double-precision numpy port of the *unclamped* formula agrees with the sequential loop to `~1e-10`, while
the *clamped* one does not). Fix: don't clamp the valid-region diff at all — only the invalid (`j>i`)
region needs forcing to exact zero, which the `-1e30`-before-`exp` sentinel already does regardless of
growth or decay. All 6 pre-existing tests pass after removing the clamp.

## Verification

**Single-block real-weight repro** (`extra/wy_numerics_repro.py`, built from `extra/blk0_real.safetensors`
— see `extra/extract_blk0_real.py` for the targeted, ~123MB, read-only extraction that never loads the full
29GB GGUF):

| | pre-fix | post-fix |
|---|---|---|
| WY (`GDN_HEAD_GROUPS=1`) `recurrent_state` NaNs | 16384 | **0** |
| LOOP (`GDN_HEAD_GROUPS=1`) `recurrent_state` NaNs | 0 | 0 |
| WY (auto, `GDN_HEAD_GROUPS=0`→2, production default) NaNs | 16384 | **0** |
| LOOP (auto) NaNs | 0 | 0 |
| WY vs. LOOP `recurrent_state`, `max\|diff\|` | n/a (NaN) | 5.94e-3 (magnitude ~20 heads) |
| WY vs. LOOP `output`, `max\|diff\|` | n/a (NaN) | 1.22e-4 |

The residual ~5.9e-3 absolute / ~3e-4 relative gap is **not** at head 42 (the underflowing head, whose
state is correctly near-zero from full decay) — it's concentrated on ordinary heads (40, 8, 24) with state
magnitude ~20-23, consistent with float32 addition reassociating differently between the WY chunked
matmul/triangular-solve form and the sequential per-token loop at REAL (much larger than random-init)
accumulated magnitudes — the same class of gap `test_gdn_scan_parity.py`'s own docstring already
anticipated for WY generally (measured there at ~1e-7 for its O(1) random-init scale; O(20) real state
naturally reassociates more visibly, still nowhere near correctness-threatening).

**Gates** (`CHECK_OOB=1 DEV=CPU`, venv, from the worktree):
- `pytest test/unit/test_gdn_scan_parity.py test/unit/test_attention.py -x -q` — all pass (7 in the parity
  file including the new regression test below, 33 passed / 9 skipped in test_attention.py; skips
  pre-existing, unrelated).
- `mypy tinygrad/` — clean (219 files).
- `ruff check .` — clean.

**New regression test**: `test/unit/test_gdn_scan_parity.py::TestGDNScanRealWeightUnderflow` hardcodes the
two real scalars that make head 42 an outlier (`ssm_a[42] = -0.337646`, `ssm_dt.bias[42] = 14.875`) onto an
otherwise-random `"38"`-geometry (48-head) block — no GGUF needed in CI. Runs in ~13s, serial-safe.
Manually verified both ways: `git stash push -- tinygrad/llm/model.py` (reverting only the fix, keeping the
new test) → **fails** with non-finite WY output/state; `git stash pop` (fix restored) → **passes**, WY
finite and within `rtol=atol=1e-2` of the loop (looser than the module's `1e-4` — justified above; this is
a real-magnitude-derived test, not the random-init suite, which stays at its existing `1e-4` untouched).

## Full-model validation (not run — pooled server was up; CPU/NULL-only rule)

```
IMPL=2 PYTHONPATH=. <venv>/bin/python extra/bug1_gguf_ab.py   # expect: clean states (0 NaNs), was NaN-FLOOD (16384) pre-fix
IMPL=1 PYTHONPATH=. <venv>/bin/python extra/bug1_gguf_ab.py   # unaffected either way -- clean states (sanity check)
```

Check `curl -s -m 2 http://localhost:8081/v1/models` returns nothing (server down) before running — the
full driver needs ~27GB and must not run while the pooled server holds the model.
