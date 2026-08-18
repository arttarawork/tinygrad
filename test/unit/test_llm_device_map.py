import unittest
from tinygrad import Tensor, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig, parse_device_map

TEST_CONFIG = TransformerConfig(num_blocks=4, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                                head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)

class TestParseDeviceMap(unittest.TestCase):
  def test_ranges(self): self.assertEqual(parse_device_map("0-1:CPU:0,2-3:CPU:1", 4), ["CPU:0", "CPU:0", "CPU:1", "CPU:1"])
  def test_single_index(self): self.assertEqual(parse_device_map("0:CPU:1,1-3:CPU:0", 4), ["CPU:1", "CPU:0", "CPU:0", "CPU:0"])
  def test_even_split(self): self.assertEqual(parse_device_map("CPU:0,CPU:1", 4), ["CPU:0", "CPU:0", "CPU:1", "CPU:1"])
  def test_dict(self): self.assertEqual(parse_device_map({0: "CPU:0", 1: "CPU:1"}, 2), ["CPU:0", "CPU:1"])
  def test_gap_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-1:CPU:0", 4)

class TestDeviceMapModel(unittest.TestCase):
  def _generate(self, model, prompt, n):
    # reseed so the sampling rand stream is identical for both models (and stays greedy at temperature=0)
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt))
    return [next(gen) for _ in range(n)]

  def test_split_matches_single_device(self):
    # NOTE: single-device ref uses the trivial map so both models run on CPU regardless of the default device
    ref, split = Transformer(TEST_CONFIG, device_map="CPU:0"), Transformer(TEST_CONFIG, device_map="0-1:CPU:0,2-3:CPU:1")
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)

    # weights landed on the mapped devices ("CPU:0" canonicalizes to "CPU")
    self.assertEqual([b.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])
    self.assertEqual(split.token_embd.weight.device, "CPU")     # consumed before the first block
    self.assertEqual(split.output.weight.device, "CPU:1")       # consumed after the last block

    # identical outputs over prefill + several decode steps, twice (second prompt exercises JIT replay)
    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 6), self._generate(split, prompt, 6))

    # lazily-created per-block state followed the activations to the mapped devices
    self.assertEqual([b.cache_kv.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])
    self.assertEqual([b.freqs_cis.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])

    # the JIT captured the mixed-device trace: only COPY calls touch two devices, compute kernels are single-device
    rollout_jit = split.jit[(False, True)]  # (is_prefill, greedy): generate() defaults to temperature=0
    self.assertIsNotNone(rollout_jit.captured)
    copies = []
    for call in rollout_jit.captured.linear.src:
      devs = set(u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None)
      if call.src[0].op is Ops.COPY: copies.append((devs, call.src[0].arg))
      else: self.assertLessEqual(len(devs), 1, f"compute kernel with mixed-device buffers: {devs}")
    self.assertIn(({"CPU", "CPU:1"}, "CPU:1"), copies)  # the block-boundary activation hop

if __name__ == '__main__':
  unittest.main()
