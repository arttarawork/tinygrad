import unittest
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, parse_device_map, parse_tp_spec, _tp_split_sizes, _split_tp_keys
from test.unit.test_mtp_load import _build_tiny_qwen35_gguf, _gguf_tensor

# T4.70b: Megatron-style FFN tensor-parallel (dense blocks only). Tiny configs mirror test_llm_device_map.py's
# patterns (CPU:0/CPU:1 splits, token-identical generate() assertions, jit-key/device checks) -- hidden_dim=128
# (a multiple of 32, GGUF Q8_0's block width) leaves room for both an even (64/64) and an uneven (96/32) split.
DENSE_CONFIG = TransformerConfig(num_blocks=2, dim=32, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                                  vocab_size=50, head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=64)

# recurrent (Gated-DeltaNet) block: TP only touches ffn_gate/up/down -- attention/GDN/norms stay per-block-device
# (T4.70b's scope guard) -- this config exercises that composition the same way test_llm_device_map.py's own
# SSM_TEST_CONFIG does for plain device_map.
SSM_CONFIG = TransformerConfig(num_blocks=2, dim=32, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                                vocab_size=50, head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=64,
                                ssm=SSMConfig(conv_kernel=4, state_size=4, group_count=2, time_step_rank=2, inner_size=8),
                                ssm_layers=(True, True))

# tiny MoE config (T3.3's pattern) -- tp: + num_experts>0 must raise at construction (T4.70b scope guard)
MOE_CONFIG = TransformerConfig(num_blocks=2, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                                vocab_size=50, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32,
                                num_experts=4, num_experts_per_tok=2)

class TestParseTpSpec(unittest.TestCase):
  def test_basic(self):
    self.assertEqual(parse_tp_spec("0-1:CPU:0,tp:CPU:0=0.5,CPU:1=0.5"), (("CPU:0", 0.5), ("CPU:1", 0.5)))
  def test_three_way_order_preserved(self):
    self.assertEqual(parse_tp_spec("CPU:0,tp:CPU:1=0.2,CPU:0=0.5,CPU:2=0.3"), (("CPU:1", 0.2), ("CPU:0", 0.5), ("CPU:2", 0.3)))
  def test_absent_is_none(self):
    self.assertIsNone(parse_tp_spec("0-1:CPU:0,2-3:CPU:1"))
    self.assertIsNone(parse_tp_spec("CPU:0"))
  def test_dict_form_is_none(self):
    self.assertIsNone(parse_tp_spec({0: "CPU:0", 1: "CPU:1"}))
  def test_fractions_must_sum_to_one(self):
    with self.assertRaises(AssertionError): parse_tp_spec("CPU:0,tp:CPU:0=0.5,CPU:1=0.4")
  def test_malformed_pair_raises(self):
    with self.assertRaises(AssertionError): parse_tp_spec("CPU:0,tp:CPU:0")
  def test_parse_device_map_ignores_and_drops_tp_segment(self):
    # parse_device_map's own return contract is UNCHANGED (still the plain 2-tuple) whether or not a tp:
    # segment is present -- existing callers/tests that never pass one see zero behavior change (T4.70b
    # hard rule: byte-identical default path, proven by test_llm_device_map.py running unmodified).
    self.assertEqual(parse_device_map("0-1:CPU:0,2-3:CPU:1,tp:CPU:0=0.5,CPU:1=0.5", 4),
                     (["CPU:0", "CPU:0", "CPU:1", "CPU:1"], None))
    self.assertEqual(parse_device_map("CPU:0,experts:CPU:2,tp:CPU:0=0.5,CPU:1=0.5", 4), (["CPU:0"] * 4, "CPU:2"))

class TestTpSplitSizes(unittest.TestCase):
  def test_even_split(self): self.assertEqual(_tp_split_sizes(128, (("CPU:0", 0.5), ("CPU:1", 0.5))), [64, 64])
  def test_uneven_fraction(self): self.assertEqual(_tp_split_sizes(128, (("CPU:0", 0.75), ("CPU:1", 0.25))), [96, 32])
  def test_rounds_to_nearest_block_of_32(self):
    # 96 * 0.5 = 48, not a multiple of 32 -- must round, and the two shards must still sum to 96 exactly
    sizes = _tp_split_sizes(96, (("CPU:0", 0.5), ("CPU:1", 0.5)))
    self.assertEqual(sum(sizes), 96)
    self.assertTrue(all(s % 32 == 0 for s in sizes))
  def test_three_way_even(self):
    self.assertEqual(_tp_split_sizes(96, (("CPU:0", 1/3), ("CPU:1", 1/3), ("CPU:2", 1/3))), [32, 32, 32])
  def test_single_shard_is_the_whole_dim(self):
    self.assertEqual(_tp_split_sizes(128, (("CPU:0", 1.0),)), [128])
  def test_unaligned_hidden_dim_raises(self):
    with self.assertRaises(AssertionError): _tp_split_sizes(100, (("CPU:0", 0.5), ("CPU:1", 0.5)))

class TestFFNTensorParallel(unittest.TestCase):
  """(a)/(b)/(c)/(e): token-identical generate() vs an unsplit reference, uneven-fraction shapes, and
  per-device weight placement -- same construction idiom as test_llm_device_map.py's TestDeviceMapModel
  (a full-size reference model realized eagerly, its state_dict sliced via _split_tp_keys and loaded into
  a device_map'd model with a 'tp:' segment, then realize_placement() before any JIT capture)."""
  def _generate(self, model, prompt, n, temperature=0.0):
    Tensor.manual_seed(42)
    gen = model.generate(list(prompt), temperature=temperature)
    return [next(gen) for _ in range(n)]

  def _tp_model(self, config, tp_str, block_dm="CPU:0"):
    ref = Transformer(config, device_map=block_dm)
    Tensor.realize(*nn.state.get_parameters(ref))  # see test_llm_device_map.py's own comment: force-realize
                                                    # ref's still-lazy weights before load_state_dict shares them
    dm = f"{block_dm},tp:{tp_str}"
    split = Transformer(config, device_map=dm)
    sd = nn.state.get_state_dict(ref)
    tp = parse_tp_spec(dm)
    assert tp is not None
    _split_tp_keys(sd, tp, config.num_blocks)
    nn.state.load_state_dict(split, sd, verbose=False, realize=False)
    split.realize_placement()
    return ref, split

  def test_token_identical_several_prompts(self):
    ref, split = self._tp_model(DENSE_CONFIG, "CPU:0=0.5,CPU:1=0.5")
    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 16), self._generate(split, prompt, 16))

  def test_uneven_fraction_shapes_and_output_match(self):
    ref, split = self._tp_model(DENSE_CONFIG, "CPU:0=0.75,CPU:1=0.25")
    self.assertEqual([lin.weight.shape[0] for lin in split.blk[0].ffn_gate_tp], [96, 32])
    self.assertEqual([lin.weight.shape[0] for lin in split.blk[0].ffn_up_tp], [96, 32])
    self.assertEqual([lin.weight.shape[1] for lin in split.blk[0].ffn_down_tp], [96, 32])
    for prompt in ([1, 2, 3, 4], [14, 15, 16]):
      self.assertEqual(self._generate(ref, prompt, 16), self._generate(split, prompt, 16))

  def test_weights_land_per_device_no_full_size_tensor(self):
    # (c): the memory point of the design -- each shard is smaller than the full hidden_dim, on its OWN
    # device, and no full-size ffn_gate/up/down attribute exists alongside the shards.
    _, split = self._tp_model(DENSE_CONFIG, "CPU:0=0.5,CPU:1=0.5")
    for block in split.blk:
      self.assertFalse(hasattr(block, "ffn_gate"))
      self.assertFalse(hasattr(block, "ffn_up"))
      self.assertFalse(hasattr(block, "ffn_down"))
      self.assertEqual([lin.weight.device for lin in block.ffn_gate_tp], ["CPU", "CPU:1"])
      self.assertEqual([lin.weight.device for lin in block.ffn_up_tp], ["CPU", "CPU:1"])
      self.assertEqual([lin.weight.device for lin in block.ffn_down_tp], ["CPU", "CPU:1"])
      for lin in block.ffn_gate_tp + block.ffn_up_tp:
        self.assertEqual(lin.weight.shape, (64, DENSE_CONFIG.dim))  # half of hidden_dim=128, never the full 128
      for lin in block.ffn_down_tp:
        self.assertEqual(lin.weight.shape, (DENSE_CONFIG.dim, 64))

  def test_tp_with_moe_raises(self):
    # (d): a clear error at construction, not a silent no-op
    with self.assertRaises(AssertionError):
      Transformer(MOE_CONFIG, device_map="CPU:0,tp:CPU:0=0.5,CPU:1=0.5")

  def test_combined_with_differing_block_device_map(self):
    # (e): the block device_map (attention/norms/GDN -- CPU:0 for both blocks) and the tp devices
    # (CPU:1/CPU:2) are entirely disjoint -- confirms tp shards follow their OWN spec, not their block's.
    ref = Transformer(DENSE_CONFIG, device_map="CPU:0")
    Tensor.realize(*nn.state.get_parameters(ref))
    dm = "0-1:CPU:0,tp:CPU:1=0.5,CPU:2=0.5"
    split = Transformer(DENSE_CONFIG, device_map=dm)
    sd = nn.state.get_state_dict(ref)
    tp = parse_tp_spec(dm)
    assert tp is not None
    _split_tp_keys(sd, tp, DENSE_CONFIG.num_blocks)
    nn.state.load_state_dict(split, sd, verbose=False, realize=False)
    split.realize_placement()

    self.assertEqual([b.attn_norm.weight.device for b in split.blk], ["CPU", "CPU"])
    self.assertEqual([b.device for b in split.blk], ["CPU", "CPU"])
    for block in split.blk:
      self.assertEqual([lin.weight.device for lin in block.ffn_gate_tp], ["CPU:1", "CPU:2"])
      self.assertEqual([lin.weight.device for lin in block.ffn_down_tp], ["CPU:1", "CPU:2"])

    for prompt in ([5, 6, 7, 8], [1, 2, 3]):
      self.assertEqual(self._generate(ref, prompt, 16), self._generate(split, prompt, 16))

  def test_recurrent_block_composes_with_tp(self):
    # GatedDeltaNetBlock's attention/GDN state stays per-block-device; only its inherited FFNBlock
    # ffn_gate/up/down get TP-sharded -- exercises the "block's other parts keep their own device" guarantee.
    ref, split = self._tp_model(SSM_CONFIG, "CPU:0=0.5,CPU:1=0.5")
    for prompt in ([5, 6, 7, 8], [9, 10, 11, 12, 13]):
      self.assertEqual(self._generate(ref, prompt, 16), self._generate(split, prompt, 16))
    self.assertEqual([b.conv_state.device for b in split.blk], ["CPU", "CPU"])  # GDN state: whole, per-block device

class TestTpWithMTP(unittest.TestCase):
  """(f): the stacks compose -- FFN-TP on the main model's dense blocks, loaded via the REAL from_gguf path
  (not the synthetic direct-construction helper above), combined with an MTP head (whole-device, out of
  TP's scope) and speculative_generate. Reuses test_mtp_load.py's tiny synthetic qwen35-shaped GGUF builder
  (same one test_spec_decode.py itself builds on) -- hidden=64 keeps the FFN a multiple of 32."""
  def _load(self, device_map, seed=0, hidden=64, dim=32, num_blocks=2, max_context=64):
    with Context(MTP=1):
      model, _ = Transformer.from_gguf(_gguf_tensor(_build_tiny_qwen35_gguf(seed=seed, hidden=hidden, dim=dim, num_blocks=num_blocks)),
                                       max_context=max_context, device_map=device_map, realize=False)
    return model

  def test_mtp_head_stays_whole_device_not_tp_sharded(self):
    dm = "0-1:CPU:0,tp:CPU:0=0.5,CPU:1=0.5"
    model = self._load(device_map=dm)
    self.assertIsNotNone(model.mtp_head)
    assert model.mtp_head is not None
    self.assertTrue(hasattr(model.mtp_head.block, "ffn_gate"))       # whole-device dense FFN, unsplit
    self.assertFalse(hasattr(model.mtp_head.block, "ffn_gate_tp"))   # never sharded -- out of T4.70b's scope
    self.assertTrue(hasattr(model.blk[0], "ffn_gate_tp"))            # the MAIN model's blocks ARE sharded
    self.assertEqual([lin.weight.device for lin in model.blk[0].ffn_gate_tp], ["CPU", "CPU:1"])

  def test_speculative_generate_token_identical_with_tp(self):
    # a fresh model PER (seed, prompt) -- mirrors test_spec_decode.py's own _load()-per-run pattern exactly
    # (generate()/speculative_generate() are stateful across calls on one instance; reusing an instance
    # across unrelated prompts is a separate, pre-existing state-management question this task doesn't
    # touch, so tests here sidestep it the same way the existing spec-decode tests already do).
    dm = "0-1:CPU:0,tp:CPU:0=0.5,CPU:1=0.5"
    for seed, prompt in enumerate(([1, 2, 3], [7])):
      ref = self._load(device_map="CPU:0", seed=seed)
      split = self._load(device_map=dm, seed=seed)
      ref_out = [v for _, v in zip(range(16), ref.generate(list(prompt), temperature=0.0))]
      split_out = [v for _, v in zip(range(16), split.speculative_generate(list(prompt), k=3, temperature=0.0))]
      self.assertEqual(split_out, ref_out, f"{seed=} {prompt=}")

if __name__ == '__main__':
  unittest.main()
