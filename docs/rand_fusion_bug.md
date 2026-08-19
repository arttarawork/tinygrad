# Suspected bug: `Tensor.rand_like` fused into a JIT'd Gumbel-argmax chain (unconfirmed)

Status: **investigated, not reproduced**, at commit `af2a43c85` (2026-08-18). Filed as
characterization work, not a fix. See `test/unit/test_rand_fusion_bug.py` (skipped) and
`extra/rand_fusion_bug_repro.py` for the runnable artifacts.

## Report

Found during unrelated device-map work on `tinygrad/llm/model.py`'s `Transformer.generate()`.
Original repro scripts were lost; this doc + the committed test/script are a reconstruction.

**Claim:** at `temperature=0`, `generate()` can emit a non-greedy token (a token other than
`argmax(logits)`). Trigger conditions as reported:

- `temperature` is a `Tensor` argument to the JIT'd `forward()`, which
  `tinygrad/engine/jit.py:_prepare_jit_inputs` (line 209) always force-realizes before the call.
- The prefill graph is symbolic-shaped (`UOp.variable`-bound sequence length, per
  `Transformer.generate()`, `tinygrad/llm/model.py:466-478`), replayed via `TinyJit` across many
  bound values.
- `Tensor.rand_like(logits)` (`tinygrad/llm/model.py:364`, the Gumbel-max sampling expression)
  fuses into the same kernel as the downstream `argmax`.
- Evidence cited: forcing `u.realize()` on the rand output "fixes" it; splitting `logits` out as
  a separate JIT output does not; the scaled top-2 logit gap was measured at ~1.5e10 (temp
  clamped to 1e-12, so `logits/temp` amplifies any real gap by ~1e12) — legitimate Gumbel noise
  (bounded, typically a few units) cannot flip an argmax across a gap that large, so the
  conclusion was that the fused `rand` values themselves must be numerically wrong, not just a
  valid-but-unlucky draw.
- Reported on both METAL and CPU; described as sequence-dependent, because `Tensor.manual_seed`
  resets `Tensor._device_seeds`/`_device_rng_counters` to fresh lazy tensors, which changes what
  the scheduler decides to fuse.

## Environment

- tinygrad `af2a43c85` (branch `task/rand-fusion-bug-repro`, forked from
  `arttarawork/tinygrad`'s `memory` branch).
- MacBook Pro M3 Pro, 36 GB, METAL backend (`Device.DEFAULT == METAL`); also tested with
  `DEV=CPU`.
- `PYTHONPATH=. .venv/bin/python`; default (no `BEAM`) for the main sweeps, plus a smaller
  `BEAM=2` sweep (see below).

## What was tried

Two independent lines of differential testing, both against tiny random-weight models (no
downloads), covering the reported trigger shape (symbolic JIT + realized-Tensor temperature +
`rand_like` feeding a Gumbel-argmax):

### 1. Decision-level: does `generate()` at `temp=0` ever disagree with eager greedy argmax?

Using the real `tinygrad.llm.model.Transformer` (not a stand-in), across:
- plain attention-only (`TransformerBlock`), MoE-enabled, and SSM/hybrid
  (`GatedDeltaNetBlock`) configs,
- many seeds, prompt lengths, and `chunk_size` settings chosen specifically to force multiple
  distinct-shaped prefill calls through one `TinyJit` (i.e. force the JIT past first-call
  "ignore" and second-call "capture" into >=3rd-call "replay" with a *different* bound shape than
  what was captured — the scenario the report's "symbolic prefill graph fuses" describes),
  including repeated `generate()` calls on the same model to extend the JIT's call count,
- both METAL and `DEV=CPU`.

**~1,180 trials total, 0 mismatches.** Every generated token at `temperature=0` matched an
independently (eagerly, no JIT, no RNG) computed `argmax(logits)` for the same context.

### 2. Value-level: are the fused RNG values themselves correct, independent of whether they'd flip an argmax?

This is the more sensitive test, and the one that matters most given the report's own claim that
the corruption is large ("wrong, not merely noisy"). A direct argmax-agreement check can only
detect corruption large enough to flip a decision; it says nothing about smaller corruption, and
depends on the test's logits happening to have a close-enough top-2 gap.

Method: reimplemented `threefry2x32` in numpy (mirrors `tinygrad/codegen/decomp/op.py:48-60`
exactly) and `RandMixin.random_bits`/`_bits_to_rand`
(`tinygrad/mixin/rand.py:12-37`) to compute the *reference* uniform(0,1) samples tinygrad
*should* produce for a given `(key, counter)` pair. Verified this reference against tinygrad's
own eager, non-symbolic `Tensor.rand_like` output first (exact bitwise match, confirming the
reference implementation itself is correct) before using it as ground truth.

Then, inside a `TinyJit`'d `forward(x, temperature)` with:
- `x` sliced by a `UOp.variable`-bound symbolic length (so the upstream logits computation is
  symbolic-shaped, matching the prefill graph),
- `temperature` a realized `Tensor` input (forced by `_prepare_jit_inputs`),
- `u = Tensor.rand_like(logits, contiguous=False)` (disabling the default `contiguous=True` to
  maximize the chance the scheduler fuses `u`'s computation into the same kernel as the
  Gumbel/argmax expression that consumes it — i.e. actively trying to hit the reported trigger,
  not just hoping for it),

the function returns `z` (the pre-argmax, per-vocab Gumbel-perturbed logits) instead of the
final `argmax`, so the actual `u` values used inside the fused kernel can be recovered exactly
via the inverse Gumbel transform (`u = exp(-exp(logits/temp - z))`) and compared element-wise
against the reference threefry output for the *exact* `(key, counter)` that
`Tensor._next_counter` handed to that call.

This was run across dozens of calls with randomly varying bound shapes (forcing JIT
capture-then-replay-at-different-shape), at `temperature=1` (chosen so `logits/temp` and the
Gumbel term are the same order of magnitude — see Note on methodology below).

**80/80 calls matched the reference to ~1e-6 (float32 rounding only).** No divergence found.

### 3. `BEAM=2` (kernel search)

Kernel search can pick a different fused/split schedule than the default heuristics, so it was
run separately (decision-level check only, real `Transformer`, chunked prefill forcing
capture-then-replay): **6/6 trials, 0 mismatches.** (Kept small — `BEAM` autotuning is slow per
kernel — but it does confirm the bug isn't specific to the non-BEAM scheduling path.)

## Note on methodology: a self-inflicted false positive along the way

Early attempts at the value-level check appeared to show *100% divergence* — every single call,
including the very first (plain eager, no JIT, no symbolic shape at all) — which would have
disproven the reference implementation, not confirmed the bug. Root cause: the inverse-Gumbel
formula was transcribed with a sign error (`u = exp(-exp(z - logits/temp))` instead of the
correct `u = exp(-exp(logits/temp - z))`). Fixing the sign made the eager case match exactly,
and re-running the full JIT/symbolic sweep with the corrected formula produced the 80/80 match
above.

This is called out explicitly because it is exactly the shape of mistake that produces a
convincing-looking "RNG is corrupted" signal from a debugging harness bug rather than a real
tinygrad bug — and because at `temperature=0` specifically, `logits/temp` is scaled by up to
1e12, which destroys all float32 precision of the Gumbel term when reconstructing `z`
numerically (`z` at that scale *is* `logits/temp`, to float32 precision — the actual random
contribution is unrepresentable). An argmax-only check at `temp=0` doesn't have this problem
(it only reads the final integer decision), but any attempt to inspect *values* at `temp=0`
needs to either work at `temp>0` (as done here — RNG values are temperature-independent, only
the argmax decision is affected by temperature) or use higher precision. This doesn't rule out
that the original report made a similar harness mistake, but it isn't possible to confirm or
rule that out without the lost original script.

## Conclusion / hypothesis

No divergence was found despite deliberately trying to hit the reported trigger conditions
(symbolic JIT shape, realized-Tensor temperature, `contiguous=False` rand to encourage fusion,
capture-then-replay-at-a-different-shape, both decision-level and direct-value-level checks, on
both METAL and CPU, across attention/MoE/SSM model variants). Two honest possibilities:

1. **The bug is real but needs conditions this investigation didn't hit.** `BEAM=2` was tried
   (see above) and also showed no divergence, though only at small scale (autotuning is slow).
   Remaining candidates: the full 35B-parameter model's actual scale/dtype (real GGUF weights,
   float16 throughout, MTP head) rather than tiny random weights; a specific
   `graph_split_rewrite`/Metal-command-buffer-graph batching interaction visible only at higher
   kernel counts (kernel names like `r_(start_pos+toks)_(start_pos+toks)` were observed during
   this investigation, confirming multi-variable symbolic kernels *do* get compiled and fused in
   these toy configs — so the general fusion machinery is exercised, just not shown to misbehave).
2. **The original observation was itself a measurement artifact**, analogous to the sign error
   found in this investigation's own harness (see above) — plausible precisely because it is so
   easy to construct a convincing false positive when reconstructing values from a Gumbel-argmax
   expression at `temperature=0`, where the amplification factor is large enough to hide real
   bugs *and* manufacture fake ones.

Given the priority was root-cause narrowing over blind fixing, and no reproducible divergence
exists to root-cause, no upstream issue is being filed from this investigation. The committed
test and script are left as regression/investigation probes for the fork, per the task, rather
than as proof of a bug.

## Repro artifacts

- `extra/rand_fusion_bug_repro.py` — standalone, <40-line, no `tinygrad/llm` import. Symbolic
  JIT graph + realized scalar temperature + `rand_like` fused into argmax, vs. a `--control` mode
  that inserts `u.realize()`. Currently prints `0 mismatches` for both modes.
- `test/unit/test_rand_fusion_bug.py` — two `@unittest.skip`'d tests: the ground-truth threefry
  differential check (methodology above), and a decision-level check against the real
  `tinygrad.llm.model.Transformer`. Both pass when the skip is removed (verified via
  `TestCase.<method>.__wrapped__` during this investigation).
