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
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig, GDN_SCAN_LOOP, GDN_SCAN_WY, gdn_scan_impl_for

GEOMETRIES = {
  # qwen3.6-35B-A3B (qwen3next hybrid; the pooled server model, CLAUDE.md "Pooled model")
  "35b": SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=32, inner_size=128 * 32),
  # qwen3.8-27B (same architecture family; real dim=5120, ssm.conv_kernel=4)
  "38": SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=128 * 48),
}
DIM = 8            # residual width: doesn't affect the scan's own shapes, kept tiny so the surrounding
                   # linears/conv compile fast on CPU -- this is the "scaled-down dim" the task asked for
MAX_CONTEXT = 64   # generous vs. every start_pos used below (the internal start_pos UOp.variable bind
                   # requires start_pos <= max_context-1)

SMALL_TOKENS, SMALL_CHUNKS = 9, (1, 2, 4, 8)  # 9-1=8 divisible by 1,2,4,8 -> every chunking's remainder is T=1
BIG_TOKENS, BIG_CHUNK = 33, 32              # 33-1=32 -> remainder is again T=1
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

if __name__ == "__main__":
  unittest.main()
