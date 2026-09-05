import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.llm.model import (Transformer, TransformerConfig, SSMConfig, VisionInput, vision_positions, mrope_freqs_cis,
                                precompute_freqs_cis)

# one GDN block + one attention block (the qwen3.5/3.6 hybrid shape), tiny -- the same config test_llm_server's recurrent suites use
CFG = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100, head_dim=16,
                        rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=64,
                        ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32), ssm_layers=(True, False))
PROMPT = list(range(1, 10))
# recorded on master 2d38cada5 (pre-T5.3) with this config, seed 7, GDN_CHUNK=4: greedy ids of generate(PROMPT); the smallest top-1/top-2
# logit margin along the run is 0.059, far above CPU float noise, so this is a stable fingerprint of the untouched text path
MASTER_IDS = [54, 6, 91, 16, 35, 75]
IMG = (1, 4, 4)  # a 4x4-patch image -> 2x2 = 4 visual tokens

def fresh() -> Transformer:
  Tensor.manual_seed(7)
  return Transformer(CFG)

def run(model:Transformer, ids:list[int], n:int, chunk:int, vision:VisionInput|None=None) -> list[int]:
  with Context(GDN_CHUNK=chunk): return [t for _, t in zip(range(n), model.generate(list(ids), vision=vision))]

def embeds(n:int, seed:int=3) -> Tensor:
  return Tensor(np.random.default_rng(seed).standard_normal((n, CFG.dim), dtype=np.float32))

class TestTextPathUnchanged(unittest.TestCase):
  def test_greedy_ids_match_master(self):
    self.assertEqual(run(fresh(), PROMPT, 6, 4), MASTER_IDS)

  def test_snapshot_round_trip_matches_master(self):
    a = fresh()
    with Context(GDN_CHUNK=4):
      ga = a.generate(list(PROMPT))
      first, snap = next(ga), a.snapshot_state()
      self.assertEqual(snap["rope_delta"], 0)
      b = fresh()
      next(b.generate([0]))
      b.restore_state(snap)
      self.assertEqual([first] + [t for _, t in zip(range(5), b.generate(list(PROMPT) + [first]))], MASTER_IDS)

  def test_text_calls_carry_no_vision_args(self):
    seen = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      seen.append(kwargs)
      return Tensor([[42]])
    with patch.object(Transformer, "__call__", mock_call):
      list(zip(range(3), fresh().generate(list(PROMPT))))
    self.assertTrue(all(k == {} for k in seen), seen)

class TestVisionPositions(unittest.TestCase):
  def test_one_image(self):
    pos3, mask, delta = vision_positions([(3, 4, IMG)], 10, 0, 0)
    self.assertEqual(pos3[:3].tolist(), [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    self.assertEqual(pos3[3:7].tolist(), [[3, 3, 3], [3, 3, 4], [3, 4, 3], [3, 4, 4]])  # (p, p+hi, p+wi), row-major over the 2x2 grid
    self.assertEqual(pos3[7:].tolist(), [[5, 5, 5], [6, 6, 6], [7, 7, 7]])              # text resumes at 3 + max(4,4)//2
    self.assertEqual(mask.tolist(), [False] * 3 + [True] * 4 + [False] * 3)
    self.assertEqual(delta, 2)  # 4 visual tokens spanned 2 rope positions

  def test_wide_image_and_two_images(self):
    # 4x8 patches -> 2x4 = 8 tokens over rope span max(4,8)//2 = 4; then a 4x4 image; deltas add up
    pos3, mask, delta = vision_positions([(1, 8, (1, 4, 8)), (10, 4, IMG)], 15, 0, 0)
    self.assertEqual(pos3[1:9].tolist(), [[1, 1 + h, 1 + w] for h in range(2) for w in range(4)])
    self.assertEqual(pos3[9].tolist(), [5, 5, 5])                     # 1 + 4
    self.assertEqual(pos3[10:14].tolist(), [[6, 6 + h, 6 + w] for h in range(2) for w in range(2)])
    self.assertEqual(pos3[14].tolist(), [8, 8, 8])                    # 6 + 2
    self.assertEqual(delta, (8 - 4) + (4 - 2))
    self.assertEqual(mask.sum(), 12)

  def test_reused_prefix_and_boundary_inside_span(self):
    spans = [(1, 4, IMG), (8, 4, IMG)]
    full, _, delta_full = vision_positions(spans, 14, 0, 0)
    # resuming at 6 with the first image's deficit (2) reproduces the tail of the full plan exactly
    tail, mask, delta_tail = vision_positions(spans, 14, 6, 2)
    self.assertEqual(tail.tolist(), full[6:].tolist())
    self.assertEqual(mask.tolist(), [False, False, True, True, True, True, False, False])
    self.assertEqual(delta_tail, delta_full)
    with self.assertRaises(AssertionError): vision_positions(spans, 14, 3, 0)  # boundary inside the first image

class TestMRope(unittest.TestCase):
  def test_text_positions_reproduce_the_table(self):
    table = precompute_freqs_cis(CFG.rope_dim, CFG.max_context, CFG.rope_theta).numpy()
    p = np.array([0, 1, 7, 40, 63], np.int32)
    rows = mrope_freqs_cis(Tensor(np.stack([p, p, p], -1)[None]), CFG).numpy()
    np.testing.assert_array_equal(rows, table[p])

  def test_pair_i_takes_axis_i_mod_3(self):
    pos3 = np.array([[[5, 9, 2]]], np.int32)
    rows = mrope_freqs_cis(Tensor(pos3), CFG).numpy()[0]
    n = CFG.rope_dim // 2
    inv = 1.0 / (CFG.rope_theta ** (np.arange(0, CFG.rope_dim, 2, dtype=np.float32) / CFG.rope_dim))
    ang = np.array([pos3[0, 0, i % 3] for i in range(n)], np.float32) * inv
    np.testing.assert_allclose(rows, np.concatenate([np.cos(ang), np.sin(ang)]), rtol=1e-5, atol=1e-6)

class TestImagePrompt(unittest.TestCase):
  IDS = [1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7, 8]  # a 4-token image run at 3..6 (placeholder ids), text either side
  VIS = [(3, 4, IMG)]

  def test_chunked_matches_one_shot(self):
    v = VisionInput(self.VIS, embeds(4))
    chunked, one_shot = run(fresh(), self.IDS, 5, 4, v), run(fresh(), self.IDS, 5, 16, v)
    self.assertEqual(chunked, one_shot)
    self.assertNotEqual(chunked, run(fresh(), self.IDS, 5, 4))  # the visual rows change the output (not silently a text run)

  def test_side_inputs_and_decode_rope_start(self):
    calls = []
    orig = Transformer.__call__
    def spy(self, tokens, start_pos, temperature, **kwargs):
      calls.append((start_pos, kwargs))
      return orig(self, tokens, start_pos, temperature, **kwargs)
    v = VisionInput(self.VIS, embeds(4))
    with patch.object(Transformer, "__call__", spy): run(fresh(), self.IDS, 3, 4, v)
    pos3, mask, delta = vision_positions(self.VIS, len(self.IDS), 0, 0)
    prefill = [(sp, kw) for sp, kw in calls if "vis_e" in kw]
    self.assertEqual([sp.unbind()[1] for sp, _ in prefill], [0, 4, 8])                   # bound start_pos of the three 4-wide chunks
    for sp, kw in prefill:
      s = sp.unbind()[1]
      self.assertEqual(kw["vis_pos"].numpy()[0, :4].tolist(), pos3[s:s+4].tolist())
      self.assertEqual(kw["vis_m"].numpy()[0, :4].tolist(), mask[s:s+4].tolist())
      rows = kw["vis_e"].numpy()[0]
      if s == 0:
        np.testing.assert_array_equal(rows[3], v.embeds.numpy()[0])
        self.assertEqual(rows[:3].any(), False)
      if s == 4:
        np.testing.assert_array_equal(rows[:3], v.embeds.numpy()[1:])
        self.assertEqual(rows[3:].any(), False)
    decode = [(sp, kw) for sp, kw in calls if "rope_start" in kw]
    self.assertEqual([(sp.unbind()[1], kw["rope_start"].unbind()[1]) for sp, kw in decode], [(12, 12 - delta), (13, 13 - delta)])

  def test_snapshot_carries_rope_delta(self):
    v = VisionInput(self.VIS, embeds(4))
    a = fresh()
    with Context(GDN_CHUNK=4):
      ga = a.generate(list(self.IDS), vision=v)
      first, snap = next(ga), a.snapshot_state()
      self.assertEqual(snap["rope_delta"], 2)
      a_rest = [next(ga) for _ in range(4)]
      b = fresh()
      next(b.generate([0]))
      b.restore_state(snap)
      self.assertEqual(b._rope_delta, 2)
      # the continuation extends the cached image prefix: the same ids, no re-supplied images needed
      self.assertEqual([t for _, t in zip(range(4), b.generate(list(self.IDS) + [first]))], a_rest)

  def test_warmup_vision(self):
    m = fresh()
    with Context(GDN_CHUNK=4): m.warmup(vision=True)
    self.assertEqual(run(m, PROMPT, 6, 4), MASTER_IDS)  # a text request after the vision warmup is the plain text path again

if __name__ == "__main__": unittest.main()
