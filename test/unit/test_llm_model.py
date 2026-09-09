import unittest
from tinygrad import Tensor
from tinygrad.llm.model import Transformer
from test.unit.test_llm_server import TEST_CONFIG

class TestPresencePenalty(unittest.TestCase):
  """T4.92: presence_penalty (OpenAI-style, binary presence not frequency) on Transformer.generate -- see model.py's own
  comments (forward()'s read side, generate()'s reset/update side) for the mechanism. TEST_CONFIG's tiny random-weight
  model degenerates into repeating one token under greedy decoding at these fixed seeds -- a clean unpenalized-vs-
  penalized contrast without needing to hand-construct weights or a logit bias."""

  def test_zero_penalty_is_byte_identical_and_adds_no_jit_family(self):
    # omitted presence_penalty (every pre-T4.92 caller) vs an explicit 0.0 must generate identical ids, allocate no
    # mask, and add no new jit key shape -- the default path is untouched.
    Tensor.manual_seed(2)
    omitted = [t for _, t in zip(range(10), Transformer(TEST_CONFIG).generate([1, 2, 3], temperature=0.0))]
    Tensor.manual_seed(2)
    m2 = Transformer(TEST_CONFIG)
    explicit_zero = [t for _, t in zip(range(10), m2.generate([1, 2, 3], temperature=0.0, presence_penalty=0.0))]
    self.assertEqual(omitted, explicit_zero)
    self.assertFalse(hasattr(m2, "penalty_mask"))
    self.assertTrue(all(len(k) == 4 for k in m2.jit), m2.jit.keys())

  def test_large_penalty_breaks_a_greedy_repeat_loop(self):
    # seed=2 is fully degenerate under plain greedy (repeats the same token forever) -- a large penalty must stop
    # it repeating at all within this window (vocab_size=100 leaves plenty of room for 20 unique picks).
    Tensor.manual_seed(2)
    unpenalized = [t for _, t in zip(range(20), Transformer(TEST_CONFIG).generate([1, 2, 3], temperature=0.0))]
    self.assertEqual(len(set(unpenalized)), 1, "fixture assumption: this seed must degenerate under plain greedy")
    Tensor.manual_seed(2)
    penalized = [t for _, t in zip(range(20), Transformer(TEST_CONFIG).generate([1, 2, 3], temperature=0.0, presence_penalty=1e6))]
    self.assertEqual(len(set(penalized)), len(penalized), "no repeats once presence penalty is active")

  def test_mask_marks_exactly_the_generated_ids_not_the_prompt(self):
    Tensor.manual_seed(2)
    model = Transformer(TEST_CONFIG)
    prompt, k = [1, 2, 3], 5
    generated = [t for _, t in zip(range(k), model.generate(prompt, temperature=0.0, presence_penalty=1e6))]
    marked = set(model.penalty_mask.numpy()[0].nonzero()[0].tolist())
    self.assertEqual(marked, set(generated))
    self.assertFalse(any(p in marked for p in prompt if p not in generated))  # prompt-only ids never marked

  def test_second_generate_call_starts_with_cleared_mask(self):
    Tensor.manual_seed(3)
    model = Transformer(TEST_CONFIG)
    list(t for _, t in zip(range(5), model.generate([1, 2, 3], temperature=0.0, presence_penalty=1e6)))
    self.assertGreater(model.penalty_mask.numpy().sum(), 0)
    first = next(model.generate([4, 5, 6], temperature=0.0, presence_penalty=1e6))
    mask = model.penalty_mask.numpy()
    self.assertEqual(mask.sum(), 1.0)  # only the one token just generated in THIS call -- run 1's marks are gone
    self.assertEqual(mask[0, first], 1.0)

if __name__ == '__main__':
  unittest.main()
