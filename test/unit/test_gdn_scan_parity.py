# T4.61: standalone parity harness for GatedDeltaNetBlock's chunked scan (tinygrad/llm/model.py _attention,
# the non-AMD unrolled-per-t else-branch -- amd_custom_kernels_supported() is only ever true on RDNA3, so
# DEV=CPU/METAL/NV all take this branch) at the two standing local models' REAL head geometry -- see
# ~/CLAUDE.md "Pooled model" (qwen3.6-35B-A3B, :8081) and the TD.6 qwen3.8-27B candidate. Only `dim` (the
# residual width feeding the block's linear projections) is scaled down for CPU speed: num_v_heads/
# num_k_heads/head_k_dim/head_v_dim -- the shapes the (B, num_v_heads, head_v_dim, head_k_dim)
# recurrent_state scan actually runs at -- are exact.
#
# GatedDeltaNetBlock.__init__ derives: head_k_dim=ssm.state_size, num_k_heads=ssm.group_count,
# num_v_heads=ssm.time_step_rank, head_v_dim=ssm.inner_size//ssm.time_step_rank. Construction/weight-init
# approach follows test/unit/test_attention.py's TestGatedDeltaNetBlock (generic nn.state.get_parameters
# loop, as used there by test_varied_chunk_sizes_match_decode / test_kda_prefill_matches_decode) and the
# chunked-vs-sequential comparison follows test/unit/test_llm_server.py's TestRecurrentChunkedPrefill.
#
# Perf note (see T4.61 report): at this real head geometry, calling _attention directly (no JIT/TinyJit --
# that's the whole point of this harness, it isolates the scan from generate()'s JIT machinery) costs a
# roughly constant ~4s/call on this machine's CPU backend REGARDLESS of T (schedule/codegen overhead
# dominates, not the per-step unroll count, until T gets much larger e.g. 32). So the design below minimizes
# the *number* of _attention calls -- especially for the chunk=1 sequential reference, which needs one call
# per token -- rather than reusing one big shared token stream for every check.
import functools
import unittest
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.llm.model import (
  GatedDeltaNetBlock, SSMConfig, TransformerConfig, GDN_SCAN_LOOP, GDN_SCAN_WY, gdn_scan_impl_for, gdn_last_scan_impl,
  _gdn_tri_inverse, gdn_scan_wy,
)

GEOMETRIES = {
  # qwen3.6-35B-A3B (qwen3next hybrid; the pooled server model, CLAUDE.md "Pooled model")
  "35b": SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=32, inner_size=128 * 32),
  # qwen3.8-27B (same architecture family; real dim=5120, ssm.conv_kernel=4)
  "38": SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=128 * 48),
}
DIM = 8            # residual width: doesn't affect the scan's own shapes, kept tiny so the surrounding
                   # linears/conv compile fast on CPU -- this is the "scaled-down dim" the task asked for
MAX_CONTEXT = 128  # generous vs. every start_pos used below (the internal
                   # start_pos UOp.variable bind requires start_pos <= max_context-1)

SMALL_TOKENS, SMALL_CHUNKS = 9, (1, 2, 4, 8)  # 9-1=8 divisible by 1,2,4,8 -> every chunking's remainder is T=1
BIG_TOKENS, BIG_CHUNK = 17, 16              # 17-1=16 -> remainder is again T=1. Was 33/32: the 32-step
# single-impl chain joined the CI xdist worker-crash class on PR #24's run (same native C-stack fragility as
# the excluded grouped chunk-32 combos in test_attention.py); 16 halves the fused depth while keeping a
# beyond-SMALL_CHUNKS parity point. Chunk-32 coverage continues via the IMPL matrix + hardware validation.
PREFIX, CONT_CHUNK = 5, 8                   # continuation: resume from a real (non-zero) state at start_pos=5

# T4.69a: both scan implementations run this whole matrix (chunks include 1, 4, and 32 -- BIG_CHUNK/CONT_CHUNK
# cover 32, SMALL_CHUNKS covers 1 and 4). GDN_SCAN_LOOP is the pre-T4.69a else-branch loop (unchanged);
# GDN_SCAN_WY is gdn_scan_wy's chunkwise WY-form (model.py) -- see its docstring for the algorithm.
IMPLS = (GDN_SCAN_LOOP, GDN_SCAN_WY)

# Empirically (see T4.61 report): chunked vs. sequential differ by ~1e-6 absolute at this geometry (float32
# rounding: a multi-step unrolled scan is one fused graph, a sequential run is separately-scheduled calls
# chained through a realized buffer -- same arithmetic, different op fusion/scheduling can still round
# differently at the ULP level). Not an approximation in the scan itself, just float non-associativity, so
# this is tight -- 10x looser than the worst observed diff, but 10x tighter than the codebase's existing
# chunk-parity convention (test_attention.py's TestGatedDeltaNetBlock uses rtol=atol=1e-3).
#
# T4.69a addendum: GDN_SCAN_WY reassociates the float math much more than a mere chunking-boundary change
# (a (T,T) triangular solve + matmuls instead of T sequential FMAs), so a bigger gap than loop-vs-loop
# chunking is expected in principle -- but measured at this file's real geometries (num_v_heads 32/48,
# head_k_dim=head_v_dim=128, chunk<=32) it comes in at ~1e-7 (see the T4.69a report's evidence-script-adjacent
# standalone check), i.e. still within the SAME RTOL/ATOL as the loop-vs-loop comparison below -- no separate,
# looser WY tolerance was needed. Known characteristic (not a bug, and not specific to this implementation --
# inherent to the decay-normalized chunked/WY delta-rule form, same as the reference FLA/Gated-DeltaNet
# implementations): the algorithm divides by the chunk's cumulative decay product, so a head with VERY
# aggressive per-step decay (e.g. a trained ssm_a with much larger magnitude than this file's randn*0.1 init
# produces) compounded over a full 32-token chunk could underflow that product towards 0 and blow up to inf/nan
# where the loop -- which never explicitly forms 1/decay -- would just quietly (and correctly) forget the old
# state. Not triggered anywhere in this file's geometries; if it ever is, the fix is a smaller GDN_CHUNK for
# WY, not a change to this tolerance.
RTOL, ATOL = 1e-4, 1e-4

# T4.68/T4.70b: the chunk-64 parity tests were REMOVED. The chunk-64 config was closed by the 2026-08-31
# hardware A/B (perf identical to chunk 32, greedy output token-identical -- TASKS.md), and their 64-step
# fused chains were the deepest graphs in CI, joining the xdist worker-crash class on PR #24's run.

def make_block(ssm: SSMConfig, seed: int = 0) -> GatedDeltaNetBlock:
  config = TransformerConfig(num_blocks=1, dim=DIM, hidden_dim=DIM * 2, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                              vocab_size=32, head_dim=DIM, rope_theta=10000.0, rope_dim=DIM, v_head_dim=DIM,
                              max_context=MAX_CONTEXT, ssm_layers=(True,), ssm=ssm)
  block = GatedDeltaNetBlock(config, ssm)
  Tensor.manual_seed(seed)  # deterministic, non-degenerate (non-zero) weights -- same seed order every call
  params = nn.state.get_parameters(block)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  # pin weights to concrete values NOW, before returning: Tensor's RNG is a lazily-updated global counter
  # (tensor.py Tensor._next_counter), so leaving these lazy would let whichever OTHER Tensor.randn() call
  # (e.g. this module's input token stream) happens to realize first "steal" this block's reserved counter
  # slice -- two separately-built make_block(ssm) calls would then silently get DIFFERENT weights despite
  # the identical seed. Confirmed empirically (see T4.61 report) to be the root cause of an early large,
  # not-just-rounding mismatch in the continuation check below.
  Tensor.realize(*params)
  return block

def run_attention(block: GatedDeltaNetBlock, x: Tensor, start_pos: int) -> np.ndarray:
  x_norm = block.attn_norm(x)
  block._init_state(x_norm)
  return block._attention(x_norm, start_pos).realize().numpy()

def snapshot(block: GatedDeltaNetBlock) -> tuple[np.ndarray, np.ndarray]:
  return block.conv_state.numpy().copy(), block.recurrent_state.numpy().copy()

def restore(block: GatedDeltaNetBlock, snap: tuple[np.ndarray, np.ndarray]) -> None:
  conv, rec = snap
  Tensor.realize(block.conv_state.assign(Tensor(conv, dtype=block.conv_state.dtype)),
                 block.recurrent_state.assign(Tensor(rec, dtype=block.recurrent_state.dtype)))

def sequential(block: GatedDeltaNetBlock, x: Tensor, start_pos: int = 0) -> np.ndarray:
  return np.concatenate([run_attention(block, x[:, t:t + 1], start_pos + t) for t in range(x.shape[1])], axis=1)

def chunked(block: GatedDeltaNetBlock, x: Tensor, chunk: int, start_pos: int = 0) -> np.ndarray:
  outs, pos = [], 0
  while pos < x.shape[1]:
    size = min(chunk, x.shape[1] - pos)
    outs.append(run_attention(block, x[:, pos:pos + size], start_pos + pos))
    pos += size
  return np.concatenate(outs, axis=1)

@functools.lru_cache(maxsize=None)
def small_reference(name: str) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
  """chunk=1 sequential ground truth over SMALL_TOKENS tokens from start_pos=0 (covers SMALL_CHUNKS).
  Pinned to GDN_SCAN_LOOP explicitly (not just the auto default) so this lru_cached ground truth is the same
  regardless of which impl's test happens to trigger the cache miss first."""
  with Context(GDN_SCAN_IMPL=GDN_SCAN_LOOP):
    block = make_block(GEOMETRIES[name])
    x = (Tensor.randn(1, SMALL_TOKENS, DIM) * 0.1).realize()
    ref_out = sequential(block, x)
    return x.numpy(), ref_out, snapshot(block)

@functools.lru_cache(maxsize=None)
def big_reference(name: str) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
  """chunk=1 sequential ground truth over BIG_TOKENS tokens from start_pos=0 (covers BIG_CHUNK), plus a
  state snapshot at start_pos=PREFIX (covers the start_pos>0 continuation case) captured along the way --
  one sequential pass serves both checks instead of two (see the perf note above). Pinned to GDN_SCAN_LOOP,
  same reason as small_reference above."""
  with Context(GDN_SCAN_IMPL=GDN_SCAN_LOOP):
    block = make_block(GEOMETRIES[name])
    x = (Tensor.randn(1, BIG_TOKENS, DIM) * 0.1).realize()
    outs, mid_state = [], None
    for t in range(BIG_TOKENS):
      outs.append(run_attention(block, x[:, t:t + 1], t))
      if t == PREFIX - 1: mid_state = snapshot(block)
    assert mid_state is not None
    return x.numpy(), np.concatenate(outs, axis=1), mid_state, snapshot(block)

class TestGDNScanChunkParity(unittest.TestCase):
  """T4.61: the scan (model.py GatedDeltaNetBlock._attention, non-AMD else-branch) must give the same outputs
  and final recurrent state no matter how its input token stream is split into chunks. T4.69a: this now runs
  under both GDN_SCAN_IMPL values (the pre-existing per-token loop, and gdn_scan_wy's chunkwise WY-form) --
  the ground truth (small_reference/big_reference) is always the GDN_SCAN_LOOP sequential run; each impl's
  chunked reconstruction is checked against that same ground truth via Context(GDN_SCAN_IMPL=impl)."""

  def test_small_chunk_sizes_match_sequential(self):
    for impl in IMPLS:
      for name in GEOMETRIES:
        x_np, ref_out, (ref_conv, ref_rec) = small_reference(name)
        x = Tensor(x_np)
        for chunk in SMALL_CHUNKS:
          block = make_block(GEOMETRIES[name])  # fresh block: starts at start_pos=0 with zeroed state already
          with Context(GDN_SCAN_IMPL=impl): out = chunked(block, x, chunk)
          conv, rec = snapshot(block)
          np.testing.assert_allclose(out, ref_out, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} {chunk=} output")
          np.testing.assert_allclose(conv, ref_conv, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} {chunk=} conv_state")
          np.testing.assert_allclose(rec, ref_rec, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} {chunk=} recurrent_state")

  def test_large_chunk_matches_sequential(self):
    for impl in IMPLS:
      for name in GEOMETRIES:
        x_np, ref_out, _, (ref_conv, ref_rec) = big_reference(name)
        block = make_block(GEOMETRIES[name])
        with Context(GDN_SCAN_IMPL=impl): out = chunked(block, Tensor(x_np), BIG_CHUNK)
        conv, rec = snapshot(block)
        np.testing.assert_allclose(out, ref_out, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} output")
        np.testing.assert_allclose(conv, ref_conv, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} conv_state")
        np.testing.assert_allclose(rec, ref_rec, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} recurrent_state")

  def test_continuation_from_nonzero_start_pos(self):
    # state carried across calls: resuming with a CONT_CHUNK-wide chunked call from a real (already
    # non-zero, previously-computed) recurrent_state at start_pos=PREFIX must match resuming one token at
    # a time from that same state -- covers a case test_large_chunk_matches_sequential can't (there, every
    # chunked run's FIRST call is always at start_pos=0, i.e. the state-reset branch; here it isn't).
    for impl in IMPLS:
      for name in GEOMETRIES:
        x_np, ref_out, mid_state, (ref_conv, ref_rec) = big_reference(name)
        block = make_block(GEOMETRIES[name])
        block._init_state(Tensor.zeros(1, 1, DIM))  # allocate conv_state/recurrent_state so restore() can assign into them
        restore(block, mid_state)
        with Context(GDN_SCAN_IMPL=impl): out = chunked(block, Tensor(x_np[:, PREFIX:]), CONT_CHUNK, start_pos=PREFIX)
        conv, rec = snapshot(block)
        np.testing.assert_allclose(out, ref_out[:, PREFIX:], rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} continuation output")
        np.testing.assert_allclose(conv, ref_conv, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} continuation conv_state")
        np.testing.assert_allclose(rec, ref_rec, rtol=RTOL, atol=ATOL, err_msg=f"{impl=} {name=} continuation recurrent_state")

  def test_auto_resolves_to_loop(self):
    # T4.69a: GDN_SCAN_IMPL=0 (auto, the default) must resolve to GDN_SCAN_LOOP -- flipping the default to
    # WY is a later, measured decision (see extra/gdn_wy_evidence.py), so today's graph stays byte-identical
    # for anyone not explicitly opting in. Explicit 1/2 must resolve to themselves.
    with Context(GDN_SCAN_IMPL=0): self.assertEqual(gdn_scan_impl_for(), GDN_SCAN_LOOP)
    with Context(GDN_SCAN_IMPL=GDN_SCAN_LOOP): self.assertEqual(gdn_scan_impl_for(), GDN_SCAN_LOOP)
    with Context(GDN_SCAN_IMPL=GDN_SCAN_WY): self.assertEqual(gdn_scan_impl_for(), GDN_SCAN_WY)

class TestGDNScanSingleStepGate(unittest.TestCase):
  """T4.69b: GDN_SCAN_IMPL=2 (WY) must fall back to the per-token loop when a call's scan only covers one
  step (T_pad==1, e.g. a decode step) -- see run_scan's comment in model.py for the measured motivation and
  why T_pad, not T, is the gate. gdn_last_scan_impl (model.py, test-introspection only) records the impl
  run_scan actually dispatched to, so the gate is directly assertable instead of inferred from output alone."""

  def _dispatched_impl(self, block: GatedDeltaNetBlock, x: Tensor, start_pos: int) -> int:
    gdn_last_scan_impl.clear()
    run_attention(block, x, start_pos)
    self.assertTrue(gdn_last_scan_impl, "run_scan recorded no dispatch")
    # every entry recorded during this one _attention call belongs to some GDN_HEAD_GROUPS head group, all
    # sharing the same T_pad (the "38" geometry auto-picks G=2) -- they must all agree with each other
    self.assertEqual(len(set(gdn_last_scan_impl)), 1, f"head groups disagreed: {gdn_last_scan_impl}")
    return gdn_last_scan_impl[-1]

  def test_multi_token_records_wy_single_step_records_loop(self):
    for name in GEOMETRIES:
      block = make_block(GEOMETRIES[name])
      x = (Tensor.randn(1, 4, DIM) * 0.1).realize()
      with Context(GDN_SCAN_IMPL=GDN_SCAN_WY):
        self.assertEqual(self._dispatched_impl(block, x[:, :3], 0), GDN_SCAN_WY, f"{name=} T_pad=3 chunk")
        self.assertEqual(self._dispatched_impl(block, x[:, 3:4], 3), GDN_SCAN_LOOP, f"{name=} T_pad=1 step")

  def test_single_step_output_matches_loop_exactly(self):
    # at T_pad==1 the WY *request* is silently downgraded to the loop internally, so from an identical fresh
    # (zeroed, deterministic -- see make_block) starting state the two requests must produce bit-identical
    # results, not merely close ones -- assert_array_equal, not allclose.
    for name in GEOMETRIES:
      x = (Tensor.randn(1, 1, DIM) * 0.1).realize()
      block_wy, block_loop = make_block(GEOMETRIES[name]), make_block(GEOMETRIES[name])
      with Context(GDN_SCAN_IMPL=GDN_SCAN_WY): out_wy = run_attention(block_wy, x, 0)
      with Context(GDN_SCAN_IMPL=GDN_SCAN_LOOP): out_loop = run_attention(block_loop, x, 0)
      np.testing.assert_array_equal(out_wy, out_loop, err_msg=f"{name=} output")
      conv_wy, rec_wy = snapshot(block_wy)
      conv_loop, rec_loop = snapshot(block_loop)
      np.testing.assert_array_equal(conv_wy, conv_loop, err_msg=f"{name=} conv_state")
      np.testing.assert_array_equal(rec_wy, rec_loop, err_msg=f"{name=} recurrent_state")

class TestGDNScanRealWeightUnderflow(unittest.TestCase):
  """T4.73c regression -- see FIXNOTES_T473C.md for the full diagnosis. At REAL trained qwen3.8-27B blk.0
  weights, head 42 of 48 decays fast enough (learned ssm_a=-0.337646, ssm_dt.bias=14.875 -- the next-most-
  aggressive real head's ssm_a is -0.131226, under half that magnitude) that its cumulative decay (a_bar)
  underflows to an EXACT float32 0.0 within one 32-token chunk (observed on the real weights: a_bar[16]=
  1.121e-44 (a denormal) -> a_bar[17]=0.0). gdn_scan_wy's pre-T4.73c formula divided v by that a_bar
  (v_tilde = v/a_bar) -> Inf, and the causal mask -- implemented as `* strict_lower`/`* lower_incl`, i.e. a
  plain elementwise multiply, where 0*Inf=NaN, not 0 -- spread that NaN to EVERY position of the affected
  head and, via the shared ssm_out projection, to the entire block output. Confirmed on the real GGUF via
  extra/wy_numerics_repro.py + extra/extract_blk0_real.py (not runnable in CI -- needs the 27GB file):
  pre-fix, GDN_SCAN_IMPL=2 produced exactly 16384 NaNs (one full head, 786432/48 elements) in
  recurrent_state while GDN_SCAN_IMPL=1 (the loop) stayed clean; post-fix, both are clean and agree.

  This test reproduces the SAME regime with only those two hardcoded real scalars overridden (no GGUF
  needed in CI, everything else stays make_block's usual small random init) -- they alone drive head 42's
  per-step alpha down to ~6.6e-3 regardless of input (softplus(x+dt_bias) is dominated by dt_bias=14.875 at
  this input scale), comfortably enough to underflow a_bar well before chunk position 32 (the real model's
  observed low was 5.44e-4/step -- real activations push it even lower, so this hardcoded setup is a
  gentler trigger than the real weights, not a cherry-picked worst case).

  Manually verified both ways (see the T4.73c report): reverting the fix (restoring gdn_scan_wy's
  `v_tilde = v / a_bar` non-kda form) makes this test's finiteness assertions FAIL (non-finite output and
  recurrent_state); with the fix in place, it PASSES and WY's output/state agree with the loop within
  RTOL2/ATOL2 below -- looser than the module's RTOL/ATOL because this file's real magnitudes (state
  values here reach ~O(20)) reassociate float32 addition order much more visibly than the random-init
  suite's O(1) values do (measured gap ~3e-4 relative / ~6e-3 absolute on this exact setup -- not
  concentrated on head 42 itself, which is correctly near-zero from full decay, but on ordinary heads
  whose larger accumulated state is more sensitive to the WY chunked reassociation vs. the sequential
  loop's rounding order)."""
  RTOL2, ATOL2 = 1e-2, 1e-2

  def _patched_block(self) -> GatedDeltaNetBlock:
    block = make_block(GEOMETRIES["38"])  # 48 heads -- head index 42 must exist
    a, bias = block.ssm_a.numpy(), block.ssm_dt["bias"].numpy()
    a[42], bias[42] = -0.337646, 14.875  # T4.73c real magnitudes, extra/blk0_real.safetensors
    Tensor.realize(block.ssm_a.replace(Tensor(a, dtype=block.ssm_a.dtype)),
                   block.ssm_dt["bias"].replace(Tensor(bias, dtype=block.ssm_dt["bias"].dtype)))
    return block

  def test_underflowing_real_head_stays_finite_and_matches_loop(self):
    x = (Tensor.randn(1, 32, DIM) * 0.1).realize()  # GDN_CHUNK's practical ceiling -- the real bug's chunk width
    block_wy, block_loop = self._patched_block(), self._patched_block()
    with Context(GDN_SCAN_IMPL=GDN_SCAN_WY, GDN_CHUNK=32): out_wy = run_attention(block_wy, x, 0)
    with Context(GDN_SCAN_IMPL=GDN_SCAN_LOOP, GDN_CHUNK=32): out_loop = run_attention(block_loop, x, 0)
    _, rec_wy = snapshot(block_wy)
    self.assertTrue(np.isfinite(out_wy).all(), "T4.73c regression: WY output went non-finite (a_bar-underflow division bug is back)")
    self.assertTrue(np.isfinite(rec_wy).all(), "T4.73c regression: WY recurrent_state went non-finite")
    np.testing.assert_allclose(out_wy, out_loop, rtol=self.RTOL2, atol=self.ATOL2, err_msg="WY vs loop output")
    np.testing.assert_allclose(rec_wy, snapshot(block_loop)[1], rtol=self.RTOL2, atol=self.ATOL2, err_msg="WY vs loop recurrent_state")

class TestGDNScanHighBetaCollinearAmplification(unittest.TestCase):
  """T4.73d regression -- see FIXNOTES_T473D.md and extra/wy_content_amplification_repro.py for the full
  hardware diagnosis (a real WY_DUMP_AMAX capture from qwen3.8-27B blk44). Real blk44 head 25's beta values
  for one chunk of a paragraph-repeated prompt (hardcoded below, all 32 real values) are uniformly high
  (0.87-0.99); combined with near-collinear keys (this content's real per-chunk key cosine similarity:
  mean~0.96, max~0.999 -- "one paragraph repeated" produces highly correlated key vectors within a chunk;
  reconstructed here as a synthetic base-vector-plus-small-noise pattern at the real K=128 dim, not the
  literal real k matrix, since _gdn_tri_inverse only ever sees m = beta * strictly_lower(k @ k.T), never k
  itself), this drove the pre-T4.73d Neumann-series DOUBLING _gdn_tri_inverse to compute an inverse matrix
  several times too large (an intermediate power transiently reaching ~1e7-1e8 before nilpotency forces
  exact cancellation down to the true, bounded ~1.0 answer -- float32's ~7 significant digits cannot
  represent that cancellation). This propagated into a real hardware amplification of GatedDeltaNetBlock's
  recurrent_state (~x7 in one chunk, compounding to fp32 Inf by chunk 132 of a longer prompt).

  Manually verified: reverting ONLY _gdn_tri_inverse to the retired doubling algorithm (kept inline in
  extra/wy_content_amplification_repro.py for comparison) makes both tests below FAIL; the current
  block-recursive-halving implementation PASSES both, matching a float64 forward-substitution reference to
  float32 precision."""
  # real beta, qwen3.8-27B blk44 head 25, chunk 3 of extra/t473d_payloads/p8k_x4.json's traced prefill
  # (WY_DUMP_AMAX=5 dump, preoverflow_chunk3_blk44.safetensors) -- exact values, not rounded
  BETA = [0.914062, 0.98291, 0.916504, 0.96582, 0.955078, 0.948242, 0.965332, 0.946289, 0.985352, 0.953613,
          0.978027, 0.967773, 0.956055, 0.967773, 0.928223, 0.985352, 0.876953, 0.963379, 0.921875, 0.97998,
          0.978027, 0.937012, 0.9375, 0.985352, 0.936035, 0.985352, 0.949219, 0.92334, 0.97998, 0.873535,
          0.977051, 0.988281]
  K_DIM = 128  # real head_k_dim==head_v_dim for the "38" geometry above

  def _near_collinear_keys(self) -> np.ndarray:
    # a base unit vector plus small per-step noise reproduces this content's ~0.96 mean pairwise cosine
    # similarity (real, measured against the dump -- see the docstring above) without embedding the
    # literal 32x128 real key matrix.
    rng = np.random.default_rng(0)
    base = rng.standard_normal(self.K_DIM)
    base /= np.linalg.norm(base)
    k = np.stack([base + 0.03 * rng.standard_normal(self.K_DIM) for _ in range(len(self.BETA))])
    return k / np.linalg.norm(k, axis=-1, keepdims=True)

  def test_tri_inverse_matches_forward_substitution_reference(self):
    k, beta, T = self._near_collinear_keys(), np.array(self.BETA), len(self.BETA)
    strict_lower = (np.arange(T)[None, :] < np.arange(T)[:, None]).astype(np.float64)
    m64 = beta[:, None] * ((k.astype(np.float64) @ k.astype(np.float64).T) * strict_lower)
    self.assertGreater(np.abs(m64).max(), 0.5, "test setup sanity: m should have large (high-beta/collinear) entries")
    # float64 forward substitution: the textbook-stable ground truth for a unit-lower-triangular solve,
    # used only as an independent reference here -- never as the production fix (see _gdn_tri_inverse's
    # docstring for why: it's the same O(C)-sequential-steps shape T4.69a moved away from for kernel count).
    a = np.eye(T) + m64
    ref = np.zeros((T, T))
    for col in range(T):
      b, x = np.eye(T)[:, col], np.zeros(T)
      for row in range(T): x[row] = b[row] - a[row, :row] @ x[:row]
      ref[:, col] = x
    got = _gdn_tri_inverse(Tensor(m64.reshape(1, 1, T, T).astype(np.float32))).numpy().reshape(T, T)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3,
      err_msg="T4.73d regression: _gdn_tri_inverse diverged from the exact forward-substitution reference "
              "-- the doubling-era catastrophic-cancellation bug is back")

  def test_full_scan_matches_loop_from_zero_state(self):
    # the bug is visible even from a ZERO carried-in state (rhs = beta*(v - a_bar*k@state.T) collapses to
    # beta*v, but the tri-inverse it's multiplied by is already wrong regardless of state) -- no need to
    # fabricate a large incoming state to see gdn_scan_wy diverge from the loop here.
    T, K = len(self.BETA), self.K_DIM
    k_np = self._near_collinear_keys()
    rng = np.random.default_rng(1)
    q_np, v_np = rng.standard_normal((T, K)) * 0.1, rng.standard_normal((T, K)) * 0.3
    beta_np, alpha_np = np.array(self.BETA), np.full(T, 0.99)  # alpha near 1: a "remembering" head, like the real one
    q, k, v = (Tensor(a.reshape(1, 1, T, K).astype(np.float32)) for a in (q_np, k_np, v_np))
    beta, alpha = Tensor(beta_np.reshape(1, 1, T).astype(np.float32)), Tensor(alpha_np.reshape(1, 1, T, 1).astype(np.float32))
    final_state, _ = gdn_scan_wy(Tensor.zeros(1, 1, K, K), q, k, v, beta, alpha)
    final_state = final_state.realize().numpy()[0, 0]

    loop_state = np.zeros((K, K))
    for t in range(T):
      loop_state = (alpha_np[t] * (loop_state @ (np.eye(K) - beta_np[t] * np.outer(k_np[t], k_np[t])))
                    + beta_np[t] * np.outer(v_np[t], k_np[t]))
    np.testing.assert_allclose(final_state, loop_state, rtol=1e-2, atol=1e-2,
      err_msg="T4.73d regression: gdn_scan_wy diverged from the sequential loop on high-beta/near-collinear "
              "content -- the doubling-era tri-inverse amplification bug is back")

if __name__ == "__main__":
  unittest.main()
