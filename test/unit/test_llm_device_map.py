import unittest
from tinygrad import Tensor, nn, Device
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, parse_device_map

TEST_CONFIG = TransformerConfig(num_blocks=4, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                                head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)

# recurrent (Gated-DeltaNet) blocks build helper tensors themselves (reset flag, conv window) instead of only consuming
# pre-placed weights, so device_map needs a separate case to exercise that code path (F1/F2/F7)
SSM_TEST_CONFIG = TransformerConfig(num_blocks=4, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                                    head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=32,
                                    ssm=SSMConfig(conv_kernel=4, state_size=4, group_count=2, time_step_rank=2, inner_size=8),
                                    ssm_layers=(True, True, True, True))

class TestParseDeviceMap(unittest.TestCase):
  def test_ranges(self): self.assertEqual(parse_device_map("0-1:CPU:0,2-3:CPU:1", 4), ["CPU:0", "CPU:0", "CPU:1", "CPU:1"])
  def test_single_index(self): self.assertEqual(parse_device_map("0:CPU:1,1-3:CPU:0", 4), ["CPU:1", "CPU:0", "CPU:0", "CPU:0"])
  def test_even_split(self): self.assertEqual(parse_device_map("CPU:0,CPU:1", 4), ["CPU:0", "CPU:0", "CPU:1", "CPU:1"])
  def test_dict(self): self.assertEqual(parse_device_map({0: "CPU:0", 1: "CPU:1"}, 2), ["CPU:0", "CPU:1"])
  def test_gap_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-1:CPU:0", 4)
  def test_whitespace_after_comma(self):
    # a stray space (e.g. "a, b") must not get misdetected as the even-split form and produce garbage devices
    self.assertEqual(parse_device_map("0-1:CPU:0, 2-3:CPU:1", 4), ["CPU:0", "CPU:0", "CPU:1", "CPU:1"])
  def test_mixed_form_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-1:CPU:0,CPU:1", 4)
  def test_range_out_of_bounds_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-7:CPU:0", 4)
  def test_overlap_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-2:CPU:0,1-3:CPU:1", 4)
  def test_dict_gap_raises(self):
    with self.assertRaises(AssertionError): parse_device_map({0: "CPU:0", 2: "CPU:1"}, 3)
  def test_even_split_too_many_devices_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("CPU:0,CPU:1,CPU:2", 2)

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

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestDeviceMapMetalCPU(unittest.TestCase):
  """T3.2: the cross-BACKEND rehearsal for Metal+NV pooling. Same shape as TestDeviceMapModel's
  CPU:0/CPU:1 split, but the seam now crosses an actual backend boundary (METAL<->CPU)."""
  def _generate(self, model, prompt, n):
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt))
    return [next(gen) for _ in range(n)]

  def test_split_matches_single_device(self):
    ref, split = Transformer(TEST_CONFIG, device_map="METAL"), Transformer(TEST_CONFIG, device_map="0-1:METAL,2-3:CPU:0")
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)

    self.assertEqual([b.device for b in split.blk], ["METAL", "METAL", "CPU", "CPU"])
    self.assertEqual(split.token_embd.weight.device, "METAL")  # consumed before the first block
    self.assertEqual(split.output.weight.device, "CPU")        # consumed after the last block

    # identical outputs over prefill + several decode steps, twice (second prompt exercises JIT replay)
    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 6), self._generate(split, prompt, 6))

    # lazily-created per-block state followed the activations to the mapped devices
    self.assertEqual([b.cache_kv.device for b in split.blk], ["METAL", "METAL", "CPU", "CPU"])
    self.assertEqual([b.freqs_cis.device for b in split.blk], ["METAL", "METAL", "CPU", "CPU"])

    # JIT-capture characterization (T3.2 objective 2): the mixed METAL/CPU trace captures end-to-end
    # (no eager fallback). METAL segments still batch into a graph (CUSTOM_FUNCTION "graph", since
    # MetalDevice.graph=MetalGraph); CPU segments stay ungraphed/sequential (CPUDevice.graph=None,
    # from HCQ2Compiled's default). A graph batch can never span devices -- GraphRunner.supports_uop
    # requires call.src[0].op is Ops.PROGRAM, which a cross-device COPY never satisfies -- so the
    # boundary hop always lands as a standalone COPY call between the two per-backend islands.
    rollout_jit = split.jit[(False, True)]
    self.assertIsNotNone(rollout_jit.captured)
    copies, graphed_devs, ungraphed_devs = [], set(), set()
    for call in rollout_jit.captured.linear.src:
      ast = call.src[0]
      if ast.op is Ops.COPY:
        devs = set(u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None)
        copies.append((devs, ast.arg))
      elif ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph":
        batch_devs = set()
        for inner in ast.src[0].src: batch_devs |= set(u.device for u in inner.toposort() if u.op is Ops.BUFFER and u.device is not None)
        self.assertEqual(len(batch_devs), 1, f"graphed batch spans devices: {batch_devs}")  # graphing never crosses backends
        graphed_devs |= batch_devs
      else:
        devs = set(u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None)
        self.assertLessEqual(len(devs), 1, f"ungraphed compute kernel with mixed-device buffers: {devs}")
        ungraphed_devs |= devs
    self.assertIn(({"METAL", "CPU"}, "CPU"), copies)  # the block-boundary activation hop
    self.assertIn("METAL", graphed_devs)              # METAL kernels got graph-batched
    self.assertIn("CPU", ungraphed_devs)               # CPU kernels stayed eager/sequential

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestDeviceMapMetalCPURecurrent(unittest.TestCase):
  def _generate(self, model, prompt, n, temperature=0.0):
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt), temperature=temperature)
    return [next(gen) for _ in range(n)]

  def test_recurrent_split_matches_single_device(self):
    ref, split = Transformer(SSM_TEST_CONFIG, device_map="METAL"), Transformer(SSM_TEST_CONFIG, device_map="0-1:METAL,2-3:CPU:0")
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)
    self.assertEqual([b.device for b in split.blk], ["METAL", "METAL", "CPU", "CPU"])

    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 6), self._generate(split, prompt, 6))

    self._generate(split, [5, 6, 7, 8], 4, temperature=0.8)  # exercise the sampled JIT variant too

    self.assertEqual([b.conv_state.device for b in split.blk], ["METAL", "METAL", "CPU", "CPU"])
    self.assertEqual([b.recurrent_state.device for b in split.blk], ["METAL", "METAL", "CPU", "CPU"])

class TestDeviceMapRecurrentModel(unittest.TestCase):
  def _generate(self, model, prompt, n, temperature=0.0):
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt), temperature=temperature)
    return [next(gen) for _ in range(n)]

  def test_recurrent_split_matches_single_device(self):
    ref, split = Transformer(SSM_TEST_CONFIG, device_map="CPU:0"), Transformer(SSM_TEST_CONFIG, device_map="0-1:CPU:0,2-3:CPU:1")
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)
    self.assertEqual([b.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])

    # identical outputs over prefill + several decode steps, twice (second prompt exercises JIT replay): F1/F2 crash on first forward/decode
    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 6), self._generate(split, prompt, 6))

    # nonzero temperature exercises the sampled JIT variant and temp's placement on the last block's device (F7)
    self._generate(split, [5, 6, 7, 8], 4, temperature=0.8)

    # lazily-created recurrent state followed the activations to the mapped devices (F1)
    self.assertEqual([b.conv_state.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])
    self.assertEqual([b.recurrent_state.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])

if __name__ == '__main__':
  unittest.main()
