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

# tiny MoE config (T3.3: sub-block expert placement) -- dim/hidden_dim/num_experts follow test_llm_moe.py's pattern
MOE_TEST_CONFIG = TransformerConfig(num_blocks=4, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                                    head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32,
                                    num_experts=4, num_experts_per_tok=2)

class TestParseDeviceMap(unittest.TestCase):
  def test_ranges(self): self.assertEqual(parse_device_map("0-1:CPU:0,2-3:CPU:1", 4), (["CPU:0", "CPU:0", "CPU:1", "CPU:1"], None))
  def test_single_index(self): self.assertEqual(parse_device_map("0:CPU:1,1-3:CPU:0", 4), (["CPU:1", "CPU:0", "CPU:0", "CPU:0"], None))
  def test_even_split(self): self.assertEqual(parse_device_map("CPU:0,CPU:1", 4), (["CPU:0", "CPU:0", "CPU:1", "CPU:1"], None))
  def test_dict(self): self.assertEqual(parse_device_map({0: "CPU:0", 1: "CPU:1"}, 2), (["CPU:0", "CPU:1"], None))
  def test_gap_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-1:CPU:0", 4)
  def test_whitespace_after_comma(self):
    # a stray space (e.g. "a, b") must not get misdetected as the even-split form and produce garbage devices
    self.assertEqual(parse_device_map("0-1:CPU:0, 2-3:CPU:1", 4), (["CPU:0", "CPU:0", "CPU:1", "CPU:1"], None))
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

  # --- "experts:<device>" segment (T3.3: sub-block MoE expert placement) ---
  def test_experts_segment_ranges(self):
    self.assertEqual(parse_device_map("0-1:CPU:0,2-3:CPU:1,experts:CPU:2", 4), (["CPU:0", "CPU:0", "CPU:1", "CPU:1"], "CPU:2"))
  def test_experts_segment_even_split(self):
    self.assertEqual(parse_device_map("CPU:0,experts:CPU:1", 4), (["CPU:0"] * 4, "CPU:1"))
  def test_experts_segment_first(self):
    # order shouldn't matter
    self.assertEqual(parse_device_map("experts:CPU:1,0-3:CPU:0", 4), (["CPU:0"] * 4, "CPU:1"))
  def test_experts_dict_key(self):
    self.assertEqual(parse_device_map({0: "CPU:0", 1: "CPU:0", "experts": "CPU:1"}, 2), (["CPU:0", "CPU:0"], "CPU:1"))
  def test_no_experts_segment_is_none(self):
    self.assertEqual(parse_device_map("0-3:CPU:0", 4), (["CPU:0"] * 4, None))
    self.assertEqual(parse_device_map({0: "CPU:0"}, 1), (["CPU:0"], None))
  def test_experts_only_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("experts:CPU:1", 4)
  def test_experts_missing_device_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-3:CPU:0,experts:", 4)
  def test_experts_duplicate_raises(self):
    with self.assertRaises(AssertionError): parse_device_map("0-3:CPU:0,experts:CPU:1,experts:CPU:2", 4)

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
    split.realize_placement()  # T4.5: force-realize the placement COPYs before any JIT capture sees them

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

    # the JIT captured the mixed-device trace: only COPY calls touch two devices, compute kernels are single-device.
    # T4.5 regression: with realize_placement() run before capture, the ONLY copy left in the graph is the real
    # per-token block-boundary activation hop -- none of the (up to 21, pre-fix) load-time weight-placement copies
    # leak into the JIT and replay every step.
    rollout_jit = split.jit[(False, True, None)]  # (is_prefill, greedy, chunk_size-if-prefill): generate() defaults to temperature=0
    self.assertIsNotNone(rollout_jit.captured)
    copies = []
    for call in rollout_jit.captured.linear.src:
      devs = set(u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None)
      if call.src[0].op is Ops.COPY: copies.append((devs, call.src[0].arg))
      else: self.assertLessEqual(len(devs), 1, f"compute kernel with mixed-device buffers: {devs}")
    self.assertEqual(copies, [({"CPU", "CPU:1"}, "CPU:1")])  # exactly the block-boundary activation hop, nothing else

class TestRealizePlacement(unittest.TestCase):
  """T4.5: Transformer.realize_placement() -- the shared home for from_gguf's force-realize fix, now
  available to manual load_state_dict callers too, plus the stranded-param footgun guard."""

  def test_no_device_map_is_noop(self):
    # zero overhead / harmless when device_map wasn't given: no _placed_devices to check against, nothing to realize
    model = Transformer(TEST_CONFIG)
    self.assertIsNone(model._placed_devices)
    model.realize_placement()  # must not raise

  def test_stranded_param_raises(self):
    # the footgun from T3.3's report: a hand-built weight assigned without device= silently lands on
    # Device.DEFAULT instead of following its block's device_map placement. CPU:97/CPU:98 are indices no
    # real Device.DEFAULT ever resolves to, so this holds regardless of what backend runs the test.
    model = Transformer(TEST_CONFIG, device_map="CPU:97,CPU:98")
    w = model.blk[0].attn_q.weight
    assert w is not None
    model.blk[0].attn_q.weight = Tensor.randn(*w.shape)  # BUG: missing device=w.device -> strands on Device.DEFAULT
    with self.assertRaises(AssertionError):
      model.realize_placement()

  def test_correctly_placed_param_does_not_raise(self):
    # the same footgun scenario, done right (device=w.device): must not raise
    model = Transformer(TEST_CONFIG, device_map="CPU:97,CPU:98")
    w = model.blk[0].attn_q.weight
    assert w is not None
    model.blk[0].attn_q.weight = Tensor.randn(*w.shape, device=w.device)
    model.realize_placement()  # must not raise

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
    rollout_jit = split.jit[(False, True, None)]
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

def _randomize_experts(model, seed=7):
  # ExpertWeights.__init__ zero-inits (like all model.py weights); randomize so the placement test is a real
  # exact-output check, not a trivially-all-zero one (test_llm_moe.py's block-level tests do the same by hand).
  # NOTE 1: device= must match the already-placed weight's device -- Tensor.randn defaults to Device.DEFAULT,
  # and a bare re-assignment (unlike .to_()/.replace()) would silently strand the new weight there instead.
  # NOTE 2: .realize() each one immediately -- 12 sequential Tensor.randn calls (4 blocks x 3 tensors) all
  # chain lazily onto the global threefry counter; left unrealized, that chain plus device_map's own lazy
  # placement COPY compounds into a UOp graph deep enough to blow Python's recursion limit inside .key's
  # hashing the first time generate() forces a realize (RecursionError, not obviously from either cause
  # alone -- found while building this test, unrelated to the T3.3 hop-placement code itself).
  Tensor.manual_seed(seed)
  for block in model.blk:
    for name in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
      w = getattr(block, name).weight
      getattr(block, name).weight = Tensor.randn(*w.shape, device=w.device).realize()

class TestDeviceMapMoEExperts(unittest.TestCase):
  """T3.3: sub-block placement -- attention+norms+KV on the block device, routed-expert weights
  (ffn_{gate,up,down}_exps) on a separate 'experts:<device>' device, with activations hopping around
  the three ExpertWeights calls inside the @function-traced block body (not at the block boundary)."""
  def _generate(self, model, prompt, n):
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt))
    return [next(gen) for _ in range(n)]

  def test_experts_split_matches_unsplit_homogeneous(self):
    ref = Transformer(MOE_TEST_CONFIG, device_map="CPU:0")
    _randomize_experts(ref)
    split = Transformer(MOE_TEST_CONFIG, device_map="0-1:CPU:0,2-3:CPU:1,experts:CPU:2")
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)
    split.realize_placement()

    # experts landed on the separate device regardless of which block-device range they're in; the router
    # (ffn_gate_inp) and everything else stayed with the block
    self.assertEqual([b.ffn_gate_exps.weight.device for b in split.blk], ["CPU:2"] * 4)
    self.assertEqual([b.ffn_up_exps.weight.device for b in split.blk], ["CPU:2"] * 4)
    self.assertEqual([b.ffn_down_exps.weight.device for b in split.blk], ["CPU:2"] * 4)
    self.assertEqual([b.ffn_gate_inp.weight.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])
    self.assertEqual([b.attn_norm.weight.device for b in split.blk], ["CPU", "CPU", "CPU:1", "CPU:1"])

    # identical outputs over prefill + several decode steps, twice (second prompt exercises JIT replay)
    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 6), self._generate(split, prompt, 6))

    # JIT captured the mid-block hop: only COPY calls touch the experts device, compute kernels are single-device
    rollout_jit = split.jit[(False, True, None)]
    self.assertIsNotNone(rollout_jit.captured)
    expert_copies = []
    for call in rollout_jit.captured.linear.src:
      devs = set(u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None)
      if call.src[0].op is Ops.COPY:
        if "CPU:2" in devs: expert_copies.append((devs, call.src[0].arg))
      else: self.assertLessEqual(len(devs), 1, f"compute kernel with mixed-device buffers: {devs}")
    # per MoE layer: h and sel hop IN (2 copies -- sel must travel with h since self.weight[sel] indexes
    # the remote weight buffer), x_down hops OUT (1 copy) = 3, not the naively-budgeted 2 (in+out as one
    # hop each). No extra copies sneak in from the gather/probs path (probs/scores never leave the block
    # device -- x_down is brought back to it before combining with probs).
    self.assertEqual(len(expert_copies), 4 * 3, f"expected 3 expert-device copies/layer (h, sel, x_down): {expert_copies}")
    self.assertEqual(sum(1 for _, dst in expert_copies if dst == "CPU:2"), 4 * 2)  # 2 "in" copies/layer land on CPU:2
    self.assertEqual(sum(1 for devs, dst in expert_copies if dst != "CPU:2"), 4)   # 1 "out" copy/layer lands back off it

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestDeviceMapMoEExpertsMetalCPU(unittest.TestCase):
  """T3.3 cross-backend: experts on CPU, attention+router+everything else on METAL."""
  def _generate(self, model, prompt, n):
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt))
    return [next(gen) for _ in range(n)]

  def test_experts_on_cpu_rest_on_metal(self):
    ref = Transformer(MOE_TEST_CONFIG, device_map="METAL")
    _randomize_experts(ref)
    split = Transformer(MOE_TEST_CONFIG, device_map="METAL,experts:CPU")
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)
    split.realize_placement()

    self.assertEqual([b.ffn_gate_exps.weight.device for b in split.blk], ["CPU"] * 4)
    self.assertEqual([b.attn_norm.weight.device for b in split.blk], ["METAL"] * 4)

    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 6), self._generate(split, prompt, 6))

    rollout_jit = split.jit[(False, True, None)]
    self.assertIsNotNone(rollout_jit.captured)
    expert_copies = 0
    for call in rollout_jit.captured.linear.src:
      if call.src[0].op is Ops.COPY:
        devs = set(u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None)
        if "CPU" in devs and "METAL" in devs: expert_copies += 1
    self.assertEqual(expert_copies, 4 * 3)  # same 3-copies/layer shape as the homogeneous CPU:0/CPU:2 case

if __name__ == '__main__':
  unittest.main()
