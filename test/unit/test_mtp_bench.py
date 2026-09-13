import math, unittest
from extra.mtp_bench import fill_context, bench_plain, bench_spec, parse_spec_stats, format_stats, deterministic_prompt
from test.unit.test_spec_decode import _load, _load_gdn

# T4.94 prep: extra/mtp_bench.py's pure parts (string parsing) plus its model-driving parts against
# test_spec_decode.py's existing tiny in-memory GGUF fixtures (_load: attention-only, _load_gdn: one real
# GatedDeltaNetBlock + one attention block, both with a real MTPHead built the same way from_gguf's own MTP
# branch does) -- no real GGUF file needed, so this task's STOP condition (ship without a test if
# speculative_generate can't be driven without one) never triggers.

class TestParseSpecStats(unittest.TestCase):
  def test_parses_a_real_summary_line(self):
    line = ("[SPEC_STATS] iters=5 emitted=12 drafted=15 avg_accept_len=2.40 drafts_per_token=1.25 "
            "accept_len_hist={0:1, 1:2, 3:2}")
    got = parse_spec_stats(line)
    self.assertEqual(got, {"iters": 5, "emitted": 12, "drafted": 15, "avg_accept_len": 2.4,
                           "drafts_per_token": 1.25, "accept_len_hist": {0: 1, 1: 2, 3: 2}})

  def test_finds_the_line_inside_other_captured_stdout(self):
    text = "some warmup noise\n[SPEC_STATS] iters=1 emitted=1 drafted=0 avg_accept_len=1.00 " \
           "drafts_per_token=0.00 accept_len_hist={0:1}\nmore noise"
    got = parse_spec_stats(text)
    self.assertIsNotNone(got)
    self.assertEqual(got["accept_len_hist"], {0: 1})

  def test_none_when_never_printed(self):
    self.assertIsNone(parse_spec_stats(""))
    self.assertIsNone(parse_spec_stats("some unrelated stdout\n"))

class TestFormatStats(unittest.TestCase):
  def test_none_stats_says_never_printed(self):
    self.assertIn("never printed", format_stats(None))

  def test_real_stats_render_avg_and_hist(self):
    stats = {"avg_accept_len": 2.0, "drafts_per_token": 1.5, "accept_len_hist": {1: 1, 0: 2}}
    out = format_stats(stats)
    self.assertIn("avg_accept_len=2.00", out)
    self.assertIn("accept_len_hist={0:2, 1:1}", out)  # sorted by accept length, not insertion order

class TestDeterministicPrompt(unittest.TestCase):
  def test_length_and_disjoint_offsets(self):
    a, b = deterministic_prompt(5, offset=0), deterministic_prompt(5, offset=1)
    self.assertEqual((len(a), len(b)), (5, 5))
    self.assertTrue(set(a).isdisjoint(b))

class TestFillContext(unittest.TestCase):
  def test_fills_cache_and_get_start_pos_reuses_it(self):
    model = _load(seed=0)
    fill_context(model, [1, 2, 3], chunk_size=8)
    self.assertEqual(model._cached_tokens, [1, 2, 3])
    self.assertEqual(model.get_start_pos([1, 2, 3, 4]), 3)  # exact-prefix extension reuses the whole fill

  def test_empty_tokens_is_a_no_op(self):
    model = _load(seed=0)
    before = list(model._cached_tokens)
    fill_context(model, [])
    self.assertEqual(model._cached_tokens, before)

class TestBenchPlainAndSpec(unittest.TestCase):
  """Drives the real bench_plain/bench_spec against a tiny model -- timings are meaningless at this scale,
  but every code path (fill, prefill timing, decode timing, SPEC_STATS scraping, greedy-equality check) is
  the same one extra/mtp_bench.py's CLI runs on a real GGUF."""

  def _assert_sane_timing(self, result:dict) -> None:
    for key in ("prefill_tok_s", "decode_tok_s"):
      self.assertTrue(math.isfinite(result[key]) and result[key] > 0, f"{key}={result[key]}")

  def test_plain_and_spec_agree_on_greedy_output_no_fill(self):
    prompt = [1, 2, 3]
    plain = bench_plain(_load(seed=0), [], prompt, decode_tokens=4, chunk_size=8)
    spec = bench_spec(_load(seed=0), [], prompt, decode_tokens=4, k=2, chunk_size=8)
    self._assert_sane_timing(plain)
    self._assert_sane_timing(spec)
    self.assertEqual(plain["output"], spec["output"])
    self.assertIsNotNone(spec["stats"])
    self.assertGreater(spec["stats"]["iters"], 0)

  def test_plain_and_spec_agree_on_greedy_output_with_fill(self):
    fill, prompt = [4, 5], [1, 2, 3]
    plain = bench_plain(_load(seed=1), fill, prompt, decode_tokens=3, chunk_size=8)
    spec = bench_spec(_load(seed=1), fill, prompt, decode_tokens=3, k=1, chunk_size=8)
    self.assertEqual(plain["output"], spec["output"])

  def test_spec_matches_plain_on_a_gdn_model(self):
    # exercises the ACCEPT/FIXUP path with a real GatedDeltaNetBlock in the mix (_load's model has none --
    # see test_spec_decode.py's own comment on why _load_gdn exists), same greedy-identity guarantee.
    prompt = [1, 2, 3]
    plain = bench_plain(_load_gdn(seed=0), [], prompt, decode_tokens=6, chunk_size=8)
    spec = bench_spec(_load_gdn(seed=0), [], prompt, decode_tokens=6, k=3, chunk_size=8)
    self.assertEqual(plain["output"], spec["output"])

if __name__ == "__main__":
  unittest.main()
