import os, unittest
from tinygrad import Tensor, GlobalCounters, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, kv_int8_block, snapshot_nbytes

ATTN_CFG = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                             head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32)
MLA_CFG = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=1, norm_eps=1e-5, vocab_size=100,
                            head_dim=16, rope_theta=10000.0, rope_dim=8, v_head_dim=16, max_context=32, kv_lora_rank=16)
SSM_CFG = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                            head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=32,
                            ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32),
                            ssm_layers=(True,))

def _run_block(cfg:TransformerConfig):
  model = Transformer(cfg)
  h = model.token_embd(Tensor([[1, 2, 3]], dtype='int32')).float()
  for block in model.blk: h = block(h, 0)
  h.realize()
  return model

class _KVF32TestCase(unittest.TestCase):
  """Base: KV_F32 is read via tinygrad's @functools.cache'd getenv(), so tests must clear that cache
  after mutating os.environ (matching how the flag is meant to be set: once, before the process starts)."""
  def setUp(self):
    self._old_env:str = os.environ.get("KV_F32", "")
    self._had_env = "KV_F32" in os.environ

  def tearDown(self):
    if self._had_env: os.environ["KV_F32"] = self._old_env
    else: os.environ.pop("KV_F32", None)
    getenv.cache_clear()  # type: ignore[attr-defined]

  def _set_kv_f32(self, val:str|None):
    if val is None: os.environ.pop("KV_F32", None)
    else: os.environ["KV_F32"] = val
    getenv.cache_clear()  # type: ignore[attr-defined]

class TestKVDtypeFlag(_KVF32TestCase):
  """KV caches default to fp16; KV_F32=1 reverts every flag-gated cache to dtypes.default_float."""
  def test_attention_cache_kv_default_fp16(self):
    self._set_kv_f32(None)
    model = _run_block(ATTN_CFG)
    self.assertEqual(model.blk[0].cache_kv.dtype, dtypes.float16)

  def test_attention_cache_kv_f32_escape(self):
    self._set_kv_f32("1")
    model = _run_block(ATTN_CFG)
    self.assertEqual(model.blk[0].cache_kv.dtype, dtypes.default_float)

  def test_mla_cache_k_default_fp16(self):
    self._set_kv_f32(None)
    model = _run_block(MLA_CFG)
    self.assertEqual(model.blk[0].cache_k.dtype, dtypes.float16)

  def test_mla_cache_k_f32_escape(self):
    self._set_kv_f32("1")
    model = _run_block(MLA_CFG)
    self.assertEqual(model.blk[0].cache_k.dtype, dtypes.default_float)

  def test_ssm_conv_state_follows_flag(self):
    self._set_kv_f32(None)
    model = _run_block(SSM_CFG)
    self.assertEqual(model.blk[0].conv_state.dtype, dtypes.float16)
    self._set_kv_f32("1")
    model = _run_block(SSM_CFG)
    self.assertEqual(model.blk[0].conv_state.dtype, dtypes.default_float)

  def test_ssm_recurrent_state_always_fp32(self):
    # recurrent_state accumulates error across the whole generation (read-modify-write every decode
    # step), unlike the write-once/read-many KV caches -- it is NOT flag-gated, always dtypes.default_float,
    # regardless of KV_F32 (evidence: tiny random-weight config, isolated fp16 recurrent_state flipped
    # 7/320 greedy tokens vs 0/320 for every KV-cache-like buffer; see model.py's GatedDeltaNetBlock._init_state)
    for val in (None, "0", "1"):
      self._set_kv_f32(val)
      model = _run_block(SSM_CFG)
      self.assertEqual(model.blk[0].recurrent_state.dtype, dtypes.default_float, f"KV_F32={val!r}")

class TestKVDtypeKernelCount(_KVF32TestCase):
  """The fp16 cast at cache write/read must fuse into the existing kernels, not add new ones."""
  def _kernel_count(self, cfg:TransformerConfig) -> int:
    model = Transformer(cfg)
    h = model.token_embd(Tensor([[1, 2, 3]], dtype='int32')).float()
    GlobalCounters.reset()
    for block in model.blk: h = block(h, 0)
    h.realize()
    return GlobalCounters.kernel_count

  def _assert_kernel_count_unchanged(self, cfg:TransformerConfig):
    # precompute_freqs_cis is @functools.cache'd (model.py): whichever model construction is first to
    # realize a given (rope_dim, max_context, theta, device) combo in this process pays for computing
    # it, every later same-args model reuses the already-realized tensor for free. Warm it once,
    # outside the measured pair, so the fp16 vs fp32 comparison below isn't just measuring cache order.
    self._kernel_count(cfg)
    n_fp16 = self._kernel_count(cfg)
    self._set_kv_f32("1")
    n_fp32 = self._kernel_count(cfg)
    self.assertEqual(n_fp16, n_fp32)

  def test_attention_kernel_count_unchanged(self):
    self._set_kv_f32(None)
    self._assert_kernel_count_unchanged(ATTN_CFG)

  def test_mla_kernel_count_unchanged(self):
    self._set_kv_f32(None)
    self._assert_kernel_count_unchanged(MLA_CFG)

  def test_ssm_kernel_count_unchanged(self):
    self._set_kv_f32(None)
    self._assert_kernel_count_unchanged(SSM_CFG)

# T6.1: KV_INT8=1 -- int8 KV cache for TransformerBlock's standard attention path (see model.py's
# kv_int8_block/_init_state/_attention). MLA/GDN are untouched by this flag (out of scope -- model.py's
# module comment above kv_int8_block explains why) so they get no tests here.
class _KVInt8TestCase(unittest.TestCase):
  """Same save/restore/cache_clear dance as _KVF32TestCase above -- KV_INT8 is a second, independent
  @functools.cache'd getenv() flag."""
  def setUp(self):
    self._old_env:str = os.environ.get("KV_INT8", "")
    self._had_env = "KV_INT8" in os.environ

  def tearDown(self):
    if self._had_env: os.environ["KV_INT8"] = self._old_env
    else: os.environ.pop("KV_INT8", None)
    getenv.cache_clear()  # type: ignore[attr-defined]

  def _set_kv_int8(self, val:str|None):
    if val is None: os.environ.pop("KV_INT8", None)
    else: os.environ["KV_INT8"] = val
    getenv.cache_clear()  # type: ignore[attr-defined]

def _seeded_model(cfg:TransformerConfig, seed:int=42) -> Transformer:
  # matches test_state_cache.py's _tiny_model: manual_seed then construct gives reproducible nn.Linear/
  # RMSNorm random init, and get_parameters+realize forces it all concrete before any forward pass.
  Tensor.manual_seed(seed)
  model = Transformer(cfg)
  Tensor.realize(*nn.state.get_parameters(model))
  return model

def _clone_model(cfg:TransformerConfig, ref:Transformer) -> Transformer:
  """A second, independently-constructed model carrying ref's exact weights (nn.state copy, not reliance
  on RNG draw order) -- so a KV_INT8 run and a default run being compared start from identical weights."""
  model = Transformer(cfg)
  nn.state.load_state_dict(model, nn.state.get_state_dict(ref), verbose=False)
  return model

def _forward_steps(model:Transformer, prompt:list[int], decode:list[int], chunk:int) -> list[Tensor]:
  """Chunked prefill (prompt split into `chunk`-sized pieces, each its own block() call at the running
  start_pos) + one-token decode steps for each id in `decode`. Returns every step's realized block-stack
  output (attention+FFN) -- exactly where KV-cache quantization error would show up."""
  outs, pos = [], 0
  for i in range(0, len(prompt), chunk):
    piece = prompt[i:i+chunk]
    h = model.token_embd(Tensor([piece], dtype='int32')).float()
    for block in model.blk: h = block(h, pos)
    outs.append(h.realize())
    pos += len(piece)
  for tok in decode:
    h = model.token_embd(Tensor([[tok]], dtype='int32')).float()
    for block in model.blk: h = block(h, pos)
    outs.append(h.realize())
    pos += 1
  return outs

class TestKVInt8Flag(_KVInt8TestCase):
  """KV_INT8 defaults off (cache_kv stays whatever KV_F32 would give it, no cache_kv_scale at all); when
  set, cache_kv is int8 and cache_kv_scale exists with one scale per kv_int8_block(head_dim)-wide chunk."""
  def test_attention_cache_kv_default_not_int8(self):
    self._set_kv_int8(None)
    model = _run_block(ATTN_CFG)
    self.assertNotEqual(model.blk[0].cache_kv.dtype, dtypes.int8)
    self.assertFalse(hasattr(model.blk[0], "cache_kv_scale"))

  def test_attention_cache_kv_int8_shape_and_scale(self):
    self._set_kv_int8("1")
    model = _run_block(ATTN_CFG)
    block = model.blk[0]
    self.assertEqual(block.cache_kv.dtype, dtypes.int8)
    blk = kv_int8_block(ATTN_CFG.head_dim)  # ATTN_CFG.head_dim=8 < 32 -> single whole-head_dim block
    self.assertEqual(blk, ATTN_CFG.head_dim)
    self.assertEqual(block.cache_kv_scale.shape, (2, 1, ATTN_CFG.n_kv_heads, ATTN_CFG.max_context, ATTN_CFG.head_dim // blk))
    self.assertEqual(block.cache_kv_scale.dtype, dtypes.float16)

class TestKVInt8Numerics(_KVInt8TestCase):
  """Numeric agreement between KV_INT8=1 and the fp16 default, identical weights (_clone_model) and inputs."""
  def test_attention_output_close_to_fp16(self):
    prompt, decode = [1, 2, 3, 4, 5], [6, 7, 8]  # 2 prefill chunks (3+2) + 3 decode steps
    self._set_kv_int8(None)
    ref_model = _seeded_model(ATTN_CFG)
    ref_outs = _forward_steps(ref_model, prompt, decode, chunk=3)

    self._set_kv_int8("1")
    int8_model = _clone_model(ATTN_CFG, ref_model)
    int8_outs = _forward_steps(int8_model, prompt, decode, chunk=3)

    # measured (5 seeds, random-normal-ish activations from tiny uniform-init weights): max relative error
    # 0.31%-0.69%, consistent with per-8-element (ATTN_CFG.head_dim=8, one whole-head_dim block) symmetric
    # absmax int8 quant (~1/127 = 0.8% worst-case per-element, damped by the attention softmax averaging
    # several quantized positions together). 1e-2 gives real headroom above that without being loose enough
    # to hide a broken quantize/dequantize.
    max_rel = 0.0
    for ref, got in zip(ref_outs, int8_outs):
      diff = (got.float() - ref.float()).abs()
      denom = ref.float().abs().max().item()
      max_rel = max(max_rel, diff.max().item() / denom if denom > 1e-9 else diff.max().item())
    self.assertLess(max_rel, 1e-2, f"KV_INT8 vs fp16 max relative error {max_rel}")

  def test_generate_greedy_ids_match_fp16(self):
    # measured: greedy ids agree exactly across 7 seeds (1,3,7,11,42,100,123) on this config -- the
    # quantization error above is far below the smallest top-1/top-2 logit margins this tiny model produces.
    # seed=42 (below) matches test_state_cache.py's _tiny_model default and gives varied (non-degenerate) ids.
    prompt = [1, 2, 3, 4, 5, 6]
    self._set_kv_int8(None)
    ref_model = _seeded_model(ATTN_CFG)
    ref_ids = [t for _, t in zip(range(6), ref_model.generate(list(prompt), temperature=0.0))]

    self._set_kv_int8("1")
    int8_model = _clone_model(ATTN_CFG, ref_model)
    int8_ids = [t for _, t in zip(range(6), int8_model.generate(list(prompt), temperature=0.0))]

    self.assertEqual(int8_ids, ref_ids, f"KV_INT8 greedy ids {int8_ids} diverged from fp16 {ref_ids}")

class TestKVInt8SnapshotRestore(_KVInt8TestCase):
  """snapshot_state/restore_state must carry cache_kv_scale alongside cache_kv (T6.1), the same way
  test_state_cache.py's TestSnapshotRestoreRoundTrip covers the plain fp16 cache_kv round trip."""
  def test_round_trip_carries_scale(self):
    self._set_kv_int8("1")
    model = _seeded_model(ATTN_CFG)
    next(model.generate([1, 2, 3], temperature=0.0))  # prefill boundary, like test_state_cache.py's _prime
    snap = model.snapshot_state()
    self.assertIn("cache_kv_scale", snap["blocks"][0])
    self.assertEqual(snap["blocks"][0]["cache_kv_scale"].dtype, dtypes.float16)

    next(model.generate([9, 9, 9, 9], temperature=0.0))  # unrelated traffic clobbers the live cache + scale
    model.restore_state(snap)
    full = [1, 2, 3, 4, 5]
    got = [v for _, v in zip(range(4), model.generate(list(full), temperature=0.0))]
    model._cached_tokens = []  # force a truly cold re-prefill on the SAME model/weights for comparison
    cold = [v for _, v in zip(range(4), model.generate(list(full), temperature=0.0))]
    self.assertEqual(got, cold)

class TestKVInt8SnapshotNbytes(_KVInt8TestCase):
  """snapshot_nbytes(int8 cache) ~ half of fp16's for the same context -- exact ratio is 0.5 + 1/blk
  (int8 payload halves the 2-byte fp16 payload; the fp16 scale tensor adds 2/blk back per element)."""
  def _assert_ratio(self, cfg:TransformerConfig):
    self._set_kv_int8(None)
    ref_model = _seeded_model(cfg, seed=1)
    next(ref_model.generate([1, 2, 3], temperature=0.0))
    n_fp16 = snapshot_nbytes(ref_model.snapshot_state())

    self._set_kv_int8("1")
    int8_model = _clone_model(cfg, ref_model)
    next(int8_model.generate([1, 2, 3], temperature=0.0))
    n_int8 = snapshot_nbytes(int8_model.snapshot_state())

    blk = kv_int8_block(cfg.head_dim)
    self.assertAlmostEqual(n_int8 / n_fp16, 0.5 + 1.0 / blk, places=6)
    return n_fp16, n_int8

  def test_clean_32_block_about_half(self):
    # head_dim=32 divides evenly by kv_int8_block's 32 -- the realistic ratio (~0.53, matching this fork's
    # real qwen configs, e.g. NV_LLM_DESIGN.md's 35B@192k 3.75GiB->~2GiB and 27B 64KB/token->~34KB/token).
    cfg = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                            vocab_size=100, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)
    n_fp16, n_int8 = self._assert_ratio(cfg)
    self.assertLess(n_int8, n_fp16 * 0.6)
    self.assertGreater(n_int8, n_fp16 * 0.45)

  def test_tiny_head_dim_fallback_block_still_shrinks(self):
    # ATTN_CFG.head_dim=8 < 32 -> kv_int8_block falls back to one whole-head_dim block (ratio 0.625, still
    # a real reduction) -- covers kv_int8_block's other branch, not just the clean-32 case above.
    n_fp16, n_int8 = self._assert_ratio(ATTN_CFG)
    self.assertLess(n_int8, n_fp16)

if __name__ == "__main__":
  unittest.main()
