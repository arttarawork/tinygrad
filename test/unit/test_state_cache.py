import unittest
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, snapshot_nbytes, snapshot_matches, kv_cache_dtype

# T4.67: snapshot_state/restore_state round-trip tests. Tiny synthetic models built directly from a
# TransformerConfig (test_llm_sampling.py's _tiny_model pattern) -- no GGUF needed, only that snapshot+restore
# reproduces the model's OWN state exactly. ATTN_CFG is attention-only (plain TransformerBlock everywhere);
# GDN_CFG mixes one GatedDeltaNetBlock with one attention block (test_llm_server.py's SSM_CFG shape) so a
# single round-trip test exercises both block kinds' snapshot/restore code paths at once.
ATTN_CFG = TransformerConfig(num_blocks=2, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                             vocab_size=32, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=64)
GDN_CFG = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                            vocab_size=32, head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=64,
                            ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32),
                            ssm_layers=(True, False))

def _tiny_model(cfg:TransformerConfig, seed:int=42) -> Transformer:
  Tensor.manual_seed(seed)
  model = Transformer(cfg)
  Tensor.realize(*nn.state.get_parameters(model))
  return model

def _prime(model:Transformer, tokens:list[int]) -> None:
  """Runs exactly one generate() step so the model's caches + _cached_tokens reflect having fully prefilled
  `tokens` -- the same 'prefill just completed' boundary serve.py snapshots at (model.py's generate(): after
  one decode step, _cached_tokens == the fed prefix, since the just-sampled token hasn't been fed back in yet)."""
  next(model.generate(list(tokens), temperature=0.0))

def _cold(model:Transformer, tokens:list[int], n:int) -> list[int]:
  """A truly cold continuation on the SAME model/weights: resetting _cached_tokens (not the cache buffers
  themselves) forces get_start_pos back to 0, so the next generate() call re-prefills `tokens` from scratch,
  overwriting every live position regardless of prior content -- the same idiom test_llm_server.py's
  test_kv_cache_resume_matches_fresh uses to get a 'fresh' comparison run without a second model instance."""
  model._cached_tokens = []
  return [v for _, v in zip(range(n), model.generate(list(tokens), temperature=0.0))]

class TestSnapshotRestoreRoundTrip(unittest.TestCase):
  def _round_trip(self, cfg:TransformerConfig):
    model = _tiny_model(cfg)
    prefix, full = [1, 2, 3], [1, 2, 3, 4, 5]
    _prime(model, prefix)
    snap = model.snapshot_state()
    _prime(model, [9, 9, 9, 9])  # unrelated traffic clobbers _cached_tokens and the live caches/GDN state
    model.restore_state(snap)
    got = [v for _, v in zip(range(6), model.generate(list(full), temperature=0.0))]
    self.assertEqual(got, _cold(model, full, 6))

  def test_round_trip_attention_only(self): self._round_trip(ATTN_CFG)
  def test_round_trip_gdn_hybrid(self): self._round_trip(GDN_CFG)

  def _longer_prompt_after_restore(self, cfg:TransformerConfig):
    # a bigger extension than the round-trip test above -- proves the TAIL prefill (not just a trivial
    # 1-2 token continuation) reconstructs the true state, not just something close enough to look right.
    model = _tiny_model(cfg, seed=7)
    prefix, full = [2, 3, 4], [2, 3, 4, 5, 6, 7, 8]
    _prime(model, prefix)
    snap = model.snapshot_state()
    _prime(model, [9, 9, 9, 9, 9])
    model.restore_state(snap)
    got = [v for _, v in zip(range(5), model.generate(list(full), temperature=0.0))]
    self.assertEqual(got, _cold(model, full, 5))

  def test_longer_prompt_after_restore_attention_only(self): self._longer_prompt_after_restore(ATTN_CFG)
  def test_longer_prompt_after_restore_gdn_hybrid(self): self._longer_prompt_after_restore(GDN_CFG)

  def test_restore_rejected_on_mismatched_prefix_falls_back_cold(self):
    # snapshot_matches is pure token-list comparison, independent of block type -- one config covers it;
    # GDN vs attention block-state correctness is already covered by the round-trip tests above.
    model = _tiny_model(ATTN_CFG, seed=3)
    prefix = [1, 2, 3]
    _prime(model, prefix)
    snap = model.snapshot_state()
    mismatched = [1, 2, 9, 9, 9]  # diverges from `prefix` at index 2 -- not an extension of it
    self.assertFalse(snapshot_matches(snap, mismatched))
    # a correct caller never calls restore_state here (see serve.py's find_snapshot) -- plain generate()
    # (whatever unrelated partial-prefix reuse get_start_pos finds on its own) must still match a cold run.
    got = [v for _, v in zip(range(6), model.generate(list(mismatched), temperature=0.0))]
    self.assertEqual(got, _cold(model, mismatched, 6))

class TestSnapshotNbytes(unittest.TestCase):
  def test_matches_hand_computed_size_and_grows_with_longer_prefix(self):
    model = _tiny_model(ATTN_CFG, seed=1)
    _prime(model, [1, 2, 3])
    self.assertEqual(len(model._cached_tokens), 3)
    n3 = snapshot_nbytes(model.snapshot_state())
    # cache_kv shape is (2, B=1, n_kv_heads, pos, head_dim) per attention block -- see TransformerBlock._init_state
    per_block = 2 * 1 * ATTN_CFG.n_kv_heads * 3 * ATTN_CFG.head_dim * kv_cache_dtype().itemsize
    self.assertEqual(n3, per_block * ATTN_CFG.num_blocks)
    self.assertGreater(n3, 0)

    _prime(model, [1, 2, 3, 4, 5])
    n5 = snapshot_nbytes(model.snapshot_state())
    self.assertGreater(n5, n3)

  def test_gdn_block_contributes_a_position_independent_amount(self):
    # conv_state/recurrent_state are fixed-size accumulators, not position-indexed (GatedDeltaNetBlock._init_state)
    # -- unlike an attention block's KV slice, a longer prefix must NOT grow the GDN block's own contribution.
    model = _tiny_model(GDN_CFG, seed=1)
    _prime(model, [1, 2])
    gdn_bytes = sum(snapshot_nbytes(bs) for bs in model.snapshot_state()["blocks"] if "conv_state" in bs)
    self.assertGreater(gdn_bytes, 0)
    _prime(model, [1, 2, 3, 4, 5])
    gdn_bytes2 = sum(snapshot_nbytes(bs) for bs in model.snapshot_state()["blocks"] if "conv_state" in bs)
    self.assertEqual(gdn_bytes, gdn_bytes2)

if __name__ == '__main__':
  unittest.main()
