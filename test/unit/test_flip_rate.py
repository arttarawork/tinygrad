import unittest, tempfile, os, time
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import Transformer
from test.unit.test_llm_server import TEST_CONFIG
from test.unit.test_state_cache import GDN_CFG
from extra.flip_rate import teacher_forced_run, run_dump, compare_runs, reset_recurrent_state, _top1_and_gap

# T4.98e2: teacher_forced_run used to drive Transformer.forward() eagerly every step (no JIT reuse across
# steps/prompts -- ~5x slower than served decode). _eager_teacher_forced_run below is that ORIGINAL
# implementation, frozen here (test-only, not in extra/flip_rate.py itself -- production code carries just
# the one, JIT'd implementation) as the reference TestFlipRateJITParity pins the new path to.
#
# .realize() on every step (NOT part of the original T4.98e code) is a deliberate fix, not a behavior
# preserved from before: TransformerBlock._attention's cache write (model.py, "we don't want to change
# self.cache_kv") never reassigns self.cache_kv itself, only a local `.after()`-chained view of it -- so
# calling forward() repeatedly with nothing forcing intermediate realization (T4.98e's actual behavior,
# multi-chunk prompts only -- every existing flip_rate test's prompt fits one default 32-wide chunk, so this
# was never exercised before) lets a later step's schedule miss an earlier step's cache write (confirmed:
# without realize, a 3-chunk prefill still argmaxes correctly, but the FIRST decode step after it silently
# diverges from what an equivalent single wide-chunk prefill gives; single-chunk parity restored by adding
# .realize() to each step, no other change). teacher_forced_run's own __call__/TinyJit path never hits this
# -- TinyJit forces exactly this kind of per-step execution already (Tensor.realize() on the first two calls
# to any jit key, then every replay dispatches its precompiled kernels immediately via run_linear) -- so this
# is the fair eager baseline for parity, not a change to what's being tested.
def _eager_step_logits(model:Transformer, ids:list[int], start_pos:int) -> Tensor:
  t = Tensor([ids], dtype="int32", device=model.blk[0].device)
  logits, _ = model.forward(t, start_pos, None, spec=True)
  return logits[:, -1, :].realize()

def _eager_teacher_forced_run(model:Transformer, prompt_ids:list[int], n_tokens:int, chunk_size:int=32,
                               force_ids:list[int]|None=None) -> tuple[list[int], list[float]]:
  reset_recurrent_state(model)
  pos = 0
  while pos < len(prompt_ids):
    n = min(chunk_size, len(prompt_ids) - pos)
    logits = _eager_step_logits(model, prompt_ids[pos:pos + n], pos)
    pos += n
  argmax_ids: list[int] = []
  gaps: list[float] = []
  for i in range(n_tokens):
    idx, gap = _top1_and_gap(logits)
    argmax_ids.append(idx)
    gaps.append(gap)
    if i == n_tokens - 1: break
    logits = _eager_step_logits(model, [force_ids[i] if force_ids is not None else idx], pos)
    pos += 1
  return argmax_ids, gaps

class TestFlipRateCore(unittest.TestCase):
  def _model(self, seed:int=0) -> Transformer:
    Tensor.manual_seed(seed)
    return Transformer(TEST_CONFIG)

  def test_self_forced_zero_flips(self):
    # forcing a model on its OWN free-running sequence must reproduce that exact sequence -- a sanity check
    # on the forcing plumbing itself, not just a quality claim.
    model = self._model()
    ref_ids, ref_gap = teacher_forced_run(model, [1, 2, 3, 4], n_tokens=6)
    other_ids, _ = teacher_forced_run(model, [1, 2, 3, 4], n_tokens=6, force_ids=ref_ids)
    total, flips, _ = compare_runs(np.array([ref_ids]), np.array([other_ids]), np.array([ref_gap]))
    self.assertEqual(total, 6)
    self.assertEqual(flips, 0)

  def test_perturbed_weight_flips(self):
    model = self._model()
    ref_ids, ref_gap = teacher_forced_run(model, [1, 2, 3, 4], n_tokens=8)
    # perturb the LM head: directly reshuffles argmax over the 100-way vocab, the most direct lever to flip
    Tensor.manual_seed(1)
    model.output.weight.assign(model.output.weight + Tensor.randn(*model.output.weight.shape) * 5.0).realize()
    other_ids, _ = teacher_forced_run(model, [1, 2, 3, 4], n_tokens=8, force_ids=ref_ids)
    total, flips, mean_gap = compare_runs(np.array([ref_ids]), np.array([other_ids]), np.array([ref_gap]))
    self.assertEqual(total, 8)
    self.assertGreater(flips, 0)
    self.assertGreaterEqual(mean_gap, 0)

  def test_run_dump_multi_prompt_shapes(self):
    model = self._model()
    result = run_dump(model, [[1, 2, 3], [4, 5]], n_tokens=4)
    self.assertEqual(len(result["argmax_ids"]), 2)
    self.assertEqual(len(result["argmax_ids"][0]), 4)
    self.assertEqual(len(result["top2_gap"][1]), 4)

class TestFlipRateCompare(unittest.TestCase):
  def test_compare_synthetic_npz(self):
    with tempfile.TemporaryDirectory() as d:
      ref_path, other_path = os.path.join(d, "ref.npz"), os.path.join(d, "other.npz")
      np.savez(ref_path, argmax_ids=np.array([[1, 2, 3], [4, 5, 6]]), top2_gap=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
      np.savez(other_path, argmax_ids=np.array([[1, 9, 3], [4, 5, 9]]), top2_gap=np.array([[0.1, 1.5, 0.3], [0.4, 0.5, 2.0]]))
      ref, other = np.load(ref_path), np.load(other_path)
      total, flips, mean_gap = compare_runs(ref["argmax_ids"], other["argmax_ids"], other["top2_gap"])
      self.assertEqual(total, 6)
      self.assertEqual(flips, 2)
      self.assertAlmostEqual(mean_gap, (1.5 + 2.0) / 2)

class TestFlipRateJITParity(unittest.TestCase):
  """T4.98e2: teacher_forced_run now drives Transformer.__call__ (JIT'd, generate()'s own bound-Variable
  prefill/decode chunking) instead of calling forward() eagerly every step. These pin the new path to the
  frozen eager reference above -- both must decode the identical argmax stream. Every test's prefill chunk
  is padded (prompt not a multiple of chunk_size -- see model.py's GatedDeltaNetBlock._attention comment on
  why padded steps are exact no-ops); most also use a small explicit chunk_size against a longer prompt so
  prefill spans >1 chunk (real chunk, then a padded remainder), not just one padded chunk."""

  def _pair(self, cfg) -> tuple[Transformer, Transformer]:
    Tensor.manual_seed(5)
    ref = Transformer(cfg)
    Tensor.manual_seed(5)
    new = Transformer(cfg)
    return ref, new

  def test_matches_eager_reference_attention_only(self):
    ref_model, new_model = self._pair(TEST_CONFIG)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    ref_ids, _ = _eager_teacher_forced_run(ref_model, prompt, n_tokens=10, chunk_size=3)
    new_ids, _ = teacher_forced_run(new_model, prompt, n_tokens=10, chunk_size=3)
    self.assertEqual(new_ids, ref_ids)

  def test_matches_eager_reference_recurrent(self):
    # GDN_CFG (test_state_cache.py): one GatedDeltaNetBlock + one attention block -- the recurrent-state path
    # _eager_teacher_forced_run's plain forward() and teacher_forced_run's JIT'd __call__ must agree on too.
    ref_model, new_model = self._pair(GDN_CFG)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    ref_ids, _ = _eager_teacher_forced_run(ref_model, prompt, n_tokens=10, chunk_size=3)
    new_ids, _ = teacher_forced_run(new_model, prompt, n_tokens=10, chunk_size=3)
    self.assertEqual(new_ids, ref_ids)

  def test_matches_eager_reference_forced(self):
    # teacher forcing onto a fixed id sequence -- exercises the force_ids branch of both paths together.
    ref_model, new_model = self._pair(GDN_CFG)
    force_ids = [5, 9, 12, 3, 1, 44, 2, 17]
    ref_ids, _ = _eager_teacher_forced_run(ref_model, [2, 4, 6], n_tokens=8, force_ids=force_ids)
    new_ids, _ = teacher_forced_run(new_model, [2, 4, 6], n_tokens=8, force_ids=force_ids)
    self.assertEqual(new_ids, ref_ids)

  def test_matches_eager_reference_multi_chunk_prefill(self):
    # prompt longer than chunk_size -- exercises >1 prefill chunk (a real, then a padded remainder chunk).
    ref_model, new_model = self._pair(TEST_CONFIG)
    prompt = list(range(1, 15))
    ref_ids, _ = _eager_teacher_forced_run(ref_model, prompt, n_tokens=6, chunk_size=4)
    new_ids, _ = teacher_forced_run(new_model, prompt, n_tokens=6, chunk_size=4)
    self.assertEqual(new_ids, ref_ids)

class TestFlipRateJITSpeed(unittest.TestCase):
  def test_jit_path_at_least_3x_faster_per_token_after_warmup(self):
    # T4.98e2's actual perf claim: measured on the served 27B over the dock, this rewrite's target is a
    # >=3x per-token speedup once both jit families (prefill, decode) are past their one-time capture --
    # reproduced here on CPU with the tiny model instead. n_tokens is decode-step-heavy on purpose (24 >> the
    # 2-call capture cost each family pays once) so the ratio reflects steady-state replay, not capture noise.
    n_tokens, warmup_prompts, timed_prompts = 24, 2, 4
    prompts = [[(i * 7 + j) % 90 + 1 for j in range(3)] for i in range(warmup_prompts + timed_prompts)]

    Tensor.manual_seed(11)
    eager_model = Transformer(TEST_CONFIG)
    t0 = time.perf_counter()
    for p in prompts[warmup_prompts:]: _eager_teacher_forced_run(eager_model, p, n_tokens)
    eager_s = time.perf_counter() - t0

    Tensor.manual_seed(11)
    jit_model = Transformer(TEST_CONFIG)
    for p in prompts[:warmup_prompts]: teacher_forced_run(jit_model, p, n_tokens)  # capture both jit families, excluded below
    t0 = time.perf_counter()
    for p in prompts[warmup_prompts:]: teacher_forced_run(jit_model, p, n_tokens)
    jit_s = time.perf_counter() - t0

    eager_per_tok, jit_per_tok = eager_s / (timed_prompts * n_tokens), jit_s / (timed_prompts * n_tokens)
    speedup = eager_per_tok / jit_per_tok
    print(f"[T4.98e2] eager {eager_per_tok*1e3:.3f} ms/tok, jit-post-warmup {jit_per_tok*1e3:.3f} ms/tok, speedup {speedup:.1f}x")
    self.assertGreaterEqual(speedup, 3.0, f"jit path only {speedup:.1f}x faster, need >=3x")

if __name__ == "__main__":
  unittest.main()
