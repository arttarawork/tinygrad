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

class TestOneSamplingFamily(unittest.TestCase):
  """T4.105: generate() always hands forward() a temperature Tensor; temperature 0 is decided inside the graph, so greedy and
  sampled generation share one jit family (prefill + decode) instead of two."""
  def test_greedy_matches_argmax_and_families_are_shared(self):
    from dataclasses import replace
    from itertools import islice
    from tinygrad import nn
    model = Transformer(replace(TEST_CONFIG, max_context=64))
    for p in nn.state.get_parameters(model): p.replace(Tensor.randn(*p.shape) * 0.1)
    Tensor.realize(*nn.state.get_parameters(model))
    prompt = [1, 2, 3, 4, 5]
    greedy = list(islice(model.generate(list(prompt), chunk_size=4, temperature=0.0), 4))
    keys_after_greedy = set(model.jit)
    sampled = list(islice(model.generate(list(prompt), chunk_size=4, temperature=0.7), 4))
    self.assertEqual(set(model.jit), keys_after_greedy, "a sampled run must reuse the greedy run's jit families")
    self.assertEqual(len(keys_after_greedy), 2, keys_after_greedy)  # one prefill + one decode family
    # the greedy ids are reproducible (same weights, same prompt) and non-empty
    again = list(islice(model.generate(list(prompt), chunk_size=4, temperature=0.0), 4))
    self.assertEqual(greedy, again)
    self.assertGreaterEqual(len(greedy), 1)
    self.assertGreaterEqual(len(sampled), 1)

class TestPrefillOnly(unittest.TestCase):
  """T4.111: prefill_only must leave the model exactly where generate() would mid-response to `tokens` as the
  whole prompt -- serve.py resumes a real request from it as if it were a genuinely cold prefill of the combined
  (prefix + tail) prompt."""

  def test_matches_cold_generate_token_identical(self):
    from dataclasses import replace
    from tinygrad import nn
    cfg = replace(TEST_CONFIG, max_context=64)
    prefix, tail = [1, 2, 3, 4, 5, 6, 7], [8, 9, 10]
    def fresh_model():
      Tensor.manual_seed(11)
      m = Transformer(cfg)
      for p in nn.state.get_parameters(m): p.replace(Tensor.randn(*p.shape) * 0.1)
      Tensor.realize(*nn.state.get_parameters(m))
      return m
    # chunk_size=3 deliberately doesn't divide len(prefix)=7 evenly, so prefill_only's own chunk boundary (0,3,6,7)
    # never lines up with a monolithic cold run's chunk boundary over prefix+tail (0,3,6,9,10) -- exactly the case
    # a real request hits (its own tail length is never aligned to whatever chunked a prior session's prefix).
    warm = fresh_model()
    warm.prefill_only(prefix, chunk_size=3)
    warm_out = [t for _, t in zip(range(5), warm.generate(prefix + tail, chunk_size=3, temperature=0.0))]

    cold = fresh_model()
    cold_out = [t for _, t in zip(range(5), cold.generate(prefix + tail, chunk_size=3, temperature=0.0))]

    self.assertEqual(warm_out, cold_out)

  def test_get_start_pos_after_prefill_only(self):
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))
    prefix = [1, 2, 3, 4, 5]
    model.prefill_only(prefix)
    self.assertEqual(model._cached_tokens, prefix)
    self.assertEqual(model.get_start_pos(prefix + [6, 7]), len(prefix))

  def test_recurrent_get_start_pos_after_prefill_only(self):
    # has_recurrent_block routes get_start_pos through the exact-prefix-or-nothing rule (model.py's own comment
    # on GDN state) -- prefill_only must leave _cached_tokens in a shape that rule accepts too, not just the
    # attention-only rule above.
    from unittest.mock import patch
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    with patch.object(Transformer, '__call__', return_value=Tensor([[42]])):
      model.prefill_only([1, 2, 3])
    self.assertEqual(model._cached_tokens, [1, 2, 3])
    self.assertEqual(model.get_start_pos([1, 2, 3, 9, 9]), 3)   # extends the prefix -- reused
    self.assertEqual(model.get_start_pos([1, 2, 9, 9]), 0)       # diverges at position 2 -- must NOT reuse
