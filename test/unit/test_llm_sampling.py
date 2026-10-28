import unittest
from tinygrad import Tensor, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig

def _tiny_model():
  Tensor.manual_seed(42)
  model = Transformer(TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                                        vocab_size=100, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32))
  Tensor.realize(*nn.state.get_parameters(model))
  return model

class TestGreedySampling(unittest.TestCase):
  def _forward_linear(self, temperature):
    linear, _ = _tiny_model().forward(Tensor([[1]], dtype="int32"), 0, temperature).linear_with_vars()
    return linear

  def test_greedy_has_no_rng_kernels(self):
    self.assertFalse(any(u.op is Ops.THREEFRY for u in self._forward_linear(None).toposort()))
    self.assertTrue(any(u.op is Ops.THREEFRY for u in self._forward_linear(Tensor([0.7])).toposort()))

  def test_greedy_matches_old_temp0(self):
    # old temp=0 path (logits/1e-12 - gumbel noise) must pick the same token as plain argmax (up to exact-tie logits)
    model = _tiny_model()
    tok = Tensor([[1]], dtype="int32")
    self.assertEqual(model.forward(tok, 0, None).item(), model.forward(tok, 0, Tensor([0.0])).item())

  def test_no_recapture_across_temperature_switch(self):
    model = _tiny_model()
    def take(n, temp): return [t for _, t in zip(range(n), model.generate([1, 2, 3], temperature=temp))]
    a = take(4, 0.0)
    caps = {k: j.captured for k, j in model.jit.items()}
    self.assertIsNotNone(caps[(False, False, None, False)])  # greedy rollout was jitted
    take(4, 0.7)
    b = take(4, 0.0)
    self.assertEqual(a, b)  # greedy is deterministic and unaffected by sampled runs
    # the sampled run captured its own jits and did not recapture the greedy ones
    # (jit entries are now created lazily, T4.12 -- a key absent from `caps` just wasn't hit yet)
    for k, j in model.jit.items():
      if caps.get(k) is not None: self.assertIs(j.captured, caps[k])
    self.assertIsNotNone(model.jit[(False, False, None, False)].captured)

if __name__ == '__main__':
  unittest.main()


class TestMinPSampling(unittest.TestCase):
  """T4.85: min-p drops the tail below min_p x top probability; the top token always survives; min_p=0 leaves the tail reachable."""
  def test_tail_never_sampled_with_min_p(self):
    from tinygrad.llm.model import sample_logits
    logits = Tensor([[6.0, 5.9, -4.0, -4.0]]).expand(64, 4).contiguous()   # two near-ties on top, two tokens ~1e-4 of the top
    ids = sample_logits(logits, Tensor([1.0]), min_p=0.5).numpy().ravel().tolist()
    self.assertTrue(set(ids) <= {0, 1}, ids)
    self.assertEqual(sample_logits(logits, Tensor([1.0]), min_p=1.0).numpy().ravel().tolist(), [0]*64)   # only the top survives
    ids0 = sample_logits(Tensor([[0.0, 0.0, 0.0, 0.0]]).expand(256, 4).contiguous(), Tensor([1.0]), min_p=0.0).numpy().ravel().tolist()
    self.assertEqual(set(ids0), {0, 1, 2, 3})   # uniform logits, no cut: every token reachable


class TestMinPMaskLogDomain(unittest.TestCase):
  """T4.125c: `scaled >= scaled.max() + log(min_p)` (the mask sample_logits uses now) must select exactly the same
  tokens as the softmax-based mask it replaced, `probs >= probs.max() * min_p` (softmax is monotone in `scaled`
  and its normalizer cancels out of the ratio -- these are the same test by construction, not by luck) -- checked
  on random logits that include ties, -inf entries (what a presence penalty or a vocab mask produces) and both
  very small and very large temperatures."""
  def test_new_mask_matches_old_softmax_mask(self):
    import math
    import numpy as np
    rng = np.random.default_rng(125)
    for min_p in (1e-6, 0.1, 0.5, 0.9, 1.0):
      for temperature in (1e-3, 0.6, 1.0, 5.0, 100.0):
        logits_np = rng.normal(size=(16, 37)).astype(np.float32)
        logits_np[:, 2] = -np.inf                 # a masked-out token (presence penalty / vocab mask)
        logits_np[0, 3] = logits_np[0, 4] = 10.0   # an explicit tie at the top of row 0
        logits = Tensor(logits_np)
        temp_t = Tensor([temperature])
        scaled = logits / temp_t.maximum(1e-12)
        probs = scaled.softmax(-1)
        old_mask = (probs >= probs.max(-1, keepdim=True) * min_p).numpy()
        new_mask = (scaled >= scaled.max(-1, keepdim=True) + math.log(min_p)).numpy()
        np.testing.assert_array_equal(old_mask, new_mask, err_msg=f"min_p={min_p} temperature={temperature}")
        # the top token always survives, every row, under both masks
        top = logits_np.argmax(-1)
        self.assertTrue(old_mask[np.arange(16), top].all())
        self.assertTrue(new_mask[np.arange(16), top].all())
        # the tied pair at row 0's top both survive (a tie is not accidentally broken by the log-domain rewrite)
        self.assertTrue(old_mask[0, 3] and old_mask[0, 4] and new_mask[0, 3] and new_mask[0, 4])
