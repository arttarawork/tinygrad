import unittest
import numpy as np
from tinygrad import Tensor, UOp, TinyJit

# ***** reference threefry2x32, mirroring tinygrad/codegen/decomp/op.py:threefry2x32 exactly *****
# used to compute ground-truth uniform(0,1) samples from a known (key, counter) pair, independent
# of tinygrad's own scheduler/codegen, so we can tell "wrong RNG values" apart from "correct RNG
# values that merely produce a different (but still valid) sample".

def _rotl(v, r):
  v = v.astype(np.uint32)
  return ((v << np.uint32(r)) | (v >> np.uint32(32 - r))).astype(np.uint32)

def _threefry2x32_np(x0, x1, key0, key1):
  x0, x1, key0, key1 = (a.astype(np.uint32) for a in (x0, x1, key0, key1))
  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  ks = [key1, key0 ^ key1 ^ np.uint32(0x1BD11BDA), key0]
  xr = [x0 + ks[2], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]:
      x0n = xr[0] + xr[1]
      xr[1] = (x0n ^ _rotl(xr[1], r)).astype(np.uint32)
      xr[0] = x0n.astype(np.uint32)
    xr = [(xr[0] + ks[i % 3]).astype(np.uint32), (xr[1] + ks[(i + 1) % 3] + np.uint32(i + 1)).astype(np.uint32)]
  return xr[0], xr[1]

def _reference_rand(seed_key, counter, num):
  # mirrors tinygrad/mixin/rand.py: RandMixin.random_bits + _bits_to_rand, for a float32 output
  key0, key1 = np.uint32(seed_key[0]), np.uint32(seed_key[1])
  c_low, c_high = np.uint32(counter[0]), np.uint32(counter[1])
  nk0, nk1 = _threefry2x32_np(np.array([c_low]), np.array([c_high]), np.array([key0]), np.array([key1]))
  half = -(-num // 2)
  counts0 = np.arange(half, dtype=np.uint32)
  counts1 = counts0 + np.uint32(half)
  lo, hi = _threefry2x32_np(counts0, counts1, np.full(half, nk0[0], dtype=np.uint32), np.full(half, nk1[0], dtype=np.uint32))
  bits = np.concatenate([lo, hi])[:num]
  uint_bits = bits >> np.uint32(32 - 23)             # nmant=23 for float32
  combined = (uint_bits | np.uint32(0x3F800000)).astype(np.uint32)   # OR with bit pattern of 1.0f
  return combined.view(np.float32) - 1.0

class TestRandFusionBug(unittest.TestCase):
  """
  Bug (reported, not confirmed at this baseline -- see docs/rand_fusion_bug.md):
  tinygrad/llm/model.py Transformer.generate() at temperature=0 was observed to occasionally emit
  a non-greedy token. Suspected trigger: `temperature` is a Tensor JIT input, which
  `engine/jit.py:_prepare_jit_inputs` always force-realizes; when this is combined with a
  symbolic-shaped prefill graph (TinyJit + UOp.variable, replayed across many bound shapes) and
  `Tensor.rand_like` fuses into the Gumbel-argmax sampling chain (model.py:364), the resulting
  uniform samples were reported to be wrong (not just noisy -- the observed top-2 logit gap after
  scaling by 1/temperature was ~1.5e10, far larger than legitimate Gumbel noise could flip).

  This test checks the strongest form of that claim directly: it compares the actual float32
  values tinygrad produces for a `rand_like` call fused (contiguous=False) into a symbolic-shaped,
  JIT-captured-and-replayed graph with a realized-Tensor `temperature` input, against an
  independent reference threefry2x32 implementation seeded with the exact (key, counter) that
  `Tensor._next_counter` handed to that call. If tinygrad's fused RNG is "wrong, not merely
  noisy" (per the report), this is the check that would catch it -- an argmax-level check can
  only catch corruption large enough to flip a decision, but a direct value diff catches any
  corruption at all.

  Status: extensively swept (this test, extra/rand_fusion_bug_repro.py, and manual investigation
  covering the real tinygrad/llm/model.py Transformer -- attention-only, MoE, and SSM/GatedDeltaNet
  configs, on both METAL and CPU) at commit af2a43c85 and found NO divergence: fused rand values
  match the reference to float32 rounding (~1e-6), and generate() at temperature=0 always matches
  an independently-computed eager greedy argmax. Left here, skipped, as a documented regression
  probe / investigation aid for the fork -- not a confirmed repro. See docs/rand_fusion_bug.md.
  """
  @unittest.skip("could not reproduce at af2a43c85 after extensive sweeps; see docs/rand_fusion_bug.md")
  def test_fused_rand_matches_reference_threefry_under_symbolic_jit(self):
    VOCAB, DIM, MAXLEN = 32, 16, 128
    Tensor.manual_seed(0)
    W = Tensor.randn(DIM, VOCAB).realize()
    full = Tensor.randn(1, MAXLEN, DIM).realize()

    def forward(x: Tensor, temperature: Tensor) -> Tensor:
      logits = (x[:, -1:] @ W)[:, -1, :]
      u = Tensor.rand_like(logits, contiguous=False)          # fusion candidate: NOT force-realized
      z = logits / temperature.maximum(1e-12) - (u.maximum(1e-12).log().neg()).log()
      return z                                                 # return pre-argmax so u is recoverable

    jit_forward = TinyJit(forward)
    dev = full.device
    seed_np = Tensor._device_seeds[dev].numpy().astype(np.uint32)
    c = Tensor._device_rng_counters[dev].numpy().astype(np.uint64)
    counter64 = int(c[0]) | (int(c[1]) << 32)   # ground truth starting point, read once before any jit call

    temp_val = 1.0   # temp=1 keeps logits/temp and the gumbel term the same magnitude (float32-safe to invert);
    temp = Tensor([temp_val]).realize()  # RNG values are temperature-independent, so this transfers to temp=0
    v_len = UOp.variable("T", 1, MAXLEN)

    import random
    random.seed(3)
    for T in (random.randint(1, MAXLEN) for _ in range(40)):
      counter_np = np.array([counter64 & 0xffffffff, (counter64 >> 32) & 0xffffffff], dtype=np.uint32)
      vt = v_len.bind(T)
      z = jit_forward(full[:, :vt], temp).numpy()[0]
      counter64 = (counter64 + VOCAB) & ((1 << 64) - 1)

      logits = ((full[:, T - 1:T] @ W)[:, -1, :]).numpy()[0]
      u_actual = np.exp(-np.exp(logits / temp_val - z))
      u_ref = _reference_rand(seed_np, counter_np, VOCAB)
      np.testing.assert_allclose(u_actual, u_ref, atol=1e-3, err_msg=f"fused rand diverged from reference threefry at T={T}")

  @unittest.skip("could not reproduce at af2a43c85 after extensive sweeps; see docs/rand_fusion_bug.md")
  def test_generate_temperature_zero_is_greedy(self):
    # decision-level check against the real tinygrad/llm/model.py Transformer, mirroring the
    # exact reported scenario (generate() with a chunked/symbolic prefill).
    from tinygrad.llm.model import Transformer, TransformerConfig
    Tensor.manual_seed(0)
    cfg = TransformerConfig(num_blocks=2, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                             vocab_size=32, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=64)
    model = Transformer(cfg)

    def greedy_token(tokens, start_pos):
      x = model.token_embd(Tensor([tokens])).float()
      for block in model.blk: x = block(x, start_pos)
      logits = model.output(model.output_norm(x[:, -1:]))[:, -1, :]
      return int(logits.argmax(-1).item())

    tokens = list(range(1, 21))
    gen = model.generate(tokens, chunk_size=6, temperature=0.0)   # chunk_size < len(tokens) forces JIT capture+replay
    for step, tok in enumerate(gen):
      if step >= 6: break
      ctx = tokens[:-1]
      self.assertEqual(tok, greedy_token(ctx, 0), f"generate() diverged from greedy argmax at step {step}")

if __name__ == '__main__':
  unittest.main()
