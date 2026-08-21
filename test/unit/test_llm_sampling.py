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
    self.assertIsNotNone(caps[(False, True, None)])  # greedy rollout was jitted
    take(4, 0.7)
    b = take(4, 0.0)
    self.assertEqual(a, b)  # greedy is deterministic and unaffected by sampled runs
    # the sampled run captured its own jits and did not recapture the greedy ones
    # (jit entries are now created lazily, T4.12 -- a key absent from `caps` just wasn't hit yet)
    for k, j in model.jit.items():
      if caps.get(k) is not None: self.assertIs(j.captured, caps[k])
    self.assertIsNotNone(model.jit[(False, False, None)].captured)

if __name__ == '__main__':
  unittest.main()
