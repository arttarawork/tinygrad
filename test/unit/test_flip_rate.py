import unittest, tempfile, os
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import Transformer
from test.unit.test_llm_server import TEST_CONFIG
from extra.flip_rate import teacher_forced_run, run_dump, compare_runs

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

if __name__ == "__main__":
  unittest.main()
