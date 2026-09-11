"""T4.101: context-memory accounting. Confirms two things measured on the 27B pooled server (memory grows
~30 KB/token max_context under KV_INT4, ~46 KB/token under KV_INT8, while the KV cache itself is only
18/35 KB/token -- see HANDOFF/TASKS T4.101 for the full report):

1. precompute_freqs_cis's module-level @functools.cache already shares ONE rope table across every
   attention-type block on the same device -- main attention blocks AND the MTP nextn block (itself a
   TransformerBlock, see MTPHead) alike. This was suspect #1 in the investigation; it turns out to already
   be true today (nothing to fix), so this file exists to guard the invariant against a future regression
   (e.g. someone dropping the @functools.cache while refactoring).
2. The only PERSISTENT buffers that scale with max_context are cache_kv/cache_kv_scale (real per-block state,
   one allocation per attention-type block) and the single shared freqs_cis table above -- their byte formulas
   match hand computation exactly. GatedDeltaNetBlock's conv_state/recurrent_state are O(1) in max_context
   (see its own _init_state comment) and never touched here.
3. The actual dominant unexplained term is NOT a per-instance attribute at all: it's the prefill attention
   score-pipeline arena (T5.6, test_llm_prefill_arena.py) -- (B,H,T,Tk_max) fp32 scratch that the JIT memory
   planner reuses across the whole prefill trace (NOT multiplied by attention-block count) and that is
   completely independent of the KV cache dtype (KV_INT4/KV_INT8 dequantize back to float before the shared
   SDPA call). This file's last test locks in that scaling behavior.
"""
import os, unittest
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.helpers import getenv
from tinygrad.llm.model import (Transformer, TransformerConfig, SSMConfig, MTPHead, TransformerBlock,
                                GatedDeltaNetBlock, precompute_freqs_cis, kv_int8_block)

# num_blocks=8, attention every 4th block (indices 3, 7) -- same ~1-in-4 ratio as the real 65-block/16-attention
# hybrid interleave, just tiny (test/unit/test_state_cache.py's GDN_CFG pattern).
CFG = TransformerConfig(num_blocks=8, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                        vocab_size=50, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=64,
                        ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32),
                        ssm_layers=(True, True, True, False, True, True, True, False))
ATTN_IDX = 3  # first attention block

def _build(max_context:int) -> Transformer:
  cfg = replace(CFG, max_context=max_context)
  Tensor.manual_seed(1)
  model = Transformer(cfg)
  # MTPHead is normally attached by from_gguf under MTP=1 -- mtp_cfg mirrors that branch exactly (qk_norm=
  # head_dim for an ssm config, block_cls=TransformerBlock since MTP/nextn is architecturally always a plain
  # attention block, never GatedDeltaNet) -- same pattern test_spec_decode.py uses for its tiny MTP models.
  model.mtp_head = MTPHead(replace(cfg, qk_norm=cfg.head_dim), TransformerBlock)
  return model

def _forward(model:Transformer, ntok:int=5) -> Tensor:
  h = model.token_embd(Tensor([[1] * ntok], dtype="int32")).float()
  for blk in model.blk: h = blk(h, 0)
  # draft() runs mtp_head.block's own __call__, allocating ITS OWN cache_kv/freqs_cis the same way a main
  # block's first forward does (FFNBlock._init_state) -- needed so the tests below can inspect it.
  model.mtp_head.draft(model, h[:, -1:], Tensor([[2]], dtype="int32"), 0).realize()
  return h

class TestRopeTableSharing(unittest.TestCase):
  def test_shared_across_main_blocks_and_mtp(self):
    model = _build(64)
    _forward(model)
    attn_idxs = [i for i, b in enumerate(model.blk) if isinstance(b, TransformerBlock)]
    self.assertGreaterEqual(len(attn_idxs), 2, "config should include at least 2 attention blocks")
    tables = [model.blk[i].freqs_cis for i in attn_idxs]
    for t in tables[1:]: self.assertIs(t, tables[0], "every attention block's freqs_cis must be the same object")
    self.assertIs(model.mtp_head.block.freqs_cis, tables[0], "MTP's own attention block must share it too")

  def test_gdn_blocks_never_allocate_a_rope_table(self):
    model = _build(64)
    _forward(model)
    for i, b in enumerate(model.blk):
      if isinstance(b, GatedDeltaNetBlock): self.assertFalse(hasattr(b, "freqs_cis"), f"block {i} is GDN, has no positions")

  def test_different_devices_get_independent_tables(self):
    # the other side of the invariant: sharing is keyed by device too, so a real multi-device deployment
    # (blocks split across METAL/NV) does NOT alias tables across devices.
    t_a = precompute_freqs_cis(8, 64, 10000.0, device="CPU:0")
    t_b = precompute_freqs_cis(8, 64, 10000.0, device="CPU:1")
    self.assertIsNot(t_a, t_b)

class TestMaxContextByteFormulas(unittest.TestCase):
  """cache_kv shape is (2, B, n_kv_heads, max_context, head_dim); freqs_cis is (max_context, rope_dim) fp32 --
  see TransformerBlock/MLATransformerBlock._init_state and precompute_freqs_cis."""
  def test_cache_kv_bytes_match_formula_for_main_and_mtp_blocks(self):
    itemsize = 2  # fp16 default, kv_cache_dtype()
    for mc in (64, 128):
      model = _build(mc)
      _forward(model)
      expected = 2 * 1 * CFG.n_kv_heads * mc * CFG.head_dim * itemsize
      self.assertEqual(model.blk[ATTN_IDX].cache_kv.nbytes(), expected)
      self.assertEqual(model.mtp_head.block.cache_kv.nbytes(), expected, "MTP's own cache follows the same formula")

  def test_cache_kv_bytes_per_token_delta(self):
    m64, m128 = _build(64), _build(128)
    _forward(m64)
    _forward(m128)
    delta = m128.blk[ATTN_IDX].cache_kv.nbytes() - m64.blk[ATTN_IDX].cache_kv.nbytes()
    self.assertEqual(delta / (128 - 64), 2 * 1 * CFG.n_kv_heads * CFG.head_dim * 2)

  def test_freqs_cis_bytes_match_formula(self):
    for mc in (64, 128):
      model = _build(mc)
      _forward(model)
      self.assertEqual(model.blk[ATTN_IDX].freqs_cis.nbytes(), mc * CFG.rope_dim * 4)  # fp32

  def test_kv_int_scale_and_packed_cache_match_formula(self):
    """KV_INT4/KV_INT8 (T6.1/T4.100): cache_kv_scale is the SAME fp16 layout for both flags -- one scale per
    kv_int8_block(head_dim)-wide chunk; only cache_kv's own packing differs (uint8 nibble-pairs vs int8)."""
    old = {f: (os.environ.get(f, ""), f in os.environ) for f in ("KV_INT4", "KV_INT8")}
    try:
      for flag, packed_head_dim in (("KV_INT4", CFG.head_dim // 2), ("KV_INT8", CFG.head_dim)):
        os.environ[flag] = "1"
        getenv.cache_clear()  # type: ignore[attr-defined]
        model = _build(64)
        _forward(model)
        blk = model.blk[ATTN_IDX]
        blk_width = kv_int8_block(CFG.head_dim)
        expected_scale = 2 * 1 * CFG.n_kv_heads * 64 * (CFG.head_dim // blk_width) * 2  # fp16 scale
        self.assertEqual(blk.cache_kv_scale.nbytes(), expected_scale)
        self.assertEqual(blk.cache_kv.nbytes(), 2 * 1 * CFG.n_kv_heads * 64 * packed_head_dim * 1)
        os.environ.pop(flag, None)
        getenv.cache_clear()  # type: ignore[attr-defined]
    finally:
      for flag, (val, had) in old.items():
        if had: os.environ[flag] = val
        else: os.environ.pop(flag, None)
      getenv.cache_clear()  # type: ignore[attr-defined]

class TestPrefillArenaScalesWithMaxContext(unittest.TestCase):
  """T4.101 reconciliation: the residual ~10-12 KB/token neither rope-table duplication (already one shared
  copy, see TestRopeTableSharing) nor MTP's own cache (small, already in the formula above) explains is the
  prefill attention score-pipeline arena (T5.6): (B,H,T,Tk_max) fp32 scratch, ~2 planner slots, reused across
  the WHOLE prefill trace (not multiplied by attention-block count) and untouched by KV_INT4/KV_INT8 (dequant
  happens before the shared SDPA call)."""
  def _arena_bytes(self, max_context:int, num_blocks:int) -> int:
    from test.unit.test_llm_prefill_arena import capture_arenas
    from test.unit.test_llm_server import TEST_CONFIG
    cfg = replace(TEST_CONFIG, max_context=max_context, num_blocks=num_blocks)
    def run():
      g = Transformer(cfg).generate(list(range(1, 41)), chunk_size=32)
      for _ in range(3): next(g)
    return max(capture_arenas(run)[0])

  def test_arena_grows_linearly_with_max_context_independent_of_block_count(self):
    from test.unit.test_llm_server import TEST_CONFIG
    a1024_1blk = self._arena_bytes(1024, num_blocks=1)
    a2048_1blk = self._arena_bytes(2048, num_blocks=1)
    a2048_4blk = self._arena_bytes(2048, num_blocks=4)
    self.assertEqual(a2048_1blk, a2048_4blk, "the score arena is a shared/reused scratch pool, not one per attention block")
    per_token = (a2048_1blk - a1024_1blk) / 1024
    predicted = 2 * TEST_CONFIG.n_heads * 32 * 4  # 2 planner slots x (B,H,T,Tk_max) fp32, T=chunk_size=32
    self.assertLess(abs(per_token - predicted) / predicted, 0.10, f"{per_token=} vs {predicted=}")

if __name__ == '__main__':
  unittest.main()
