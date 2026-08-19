import os, unittest
from tinygrad import Tensor, GlobalCounters, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig

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

if __name__ == "__main__":
  unittest.main()
