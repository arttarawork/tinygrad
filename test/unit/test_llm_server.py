import contextlib, io, pathlib, re, tempfile, unittest
import numpy as np
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.nn.state import get_state_dict, safe_load
from tinygrad.schedule import schedule_cache
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.serve import StreamRouter, splice_ids
from tinygrad.llm.cli import SimpleTokenizer, FallbackTemplate

TEST_CONFIG = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                           norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)
V_START_POS = UOp.variable("start_pos", 0, TEST_CONFIG.max_context-1)
V_TOKS = UOp.variable("toks", 1, 32)  # 32 is the default chunk_size in generate

class TestTransformerGenerate(unittest.TestCase):
  def test_warmup(self):
    model, calls = Transformer(TEST_CONFIG), []
    def generate(tokens, temperature=0.0, **kwargs):
      calls.append((tokens, temperature))
      yield from (1, 2)
    with patch.object(model, "generate", generate): model.warmup()
    # warms both the greedy (temperature=0.0) and sampled (temperature>0) jit pairs
    self.assertEqual(calls, [([0], 0.0), ([0], 0.0), ([0], 1.0), ([0], 1.0)])

  def test_warmup_then_generate_with_default_chunk(self):
    # warmup must not capture JIT graphs that generate()'s default chunk_size then rejects
    model = Transformer(TEST_CONFIG)
    model.warmup()
    self.assertIsInstance(next(model.generate([5, 6, 7, 8])), int)

  def test_warmup_captures_sampled_jit(self):
    # a nonzero-temperature request must not pay a mid-request JIT capture (F4)
    model = Transformer(TEST_CONFIG)
    model.warmup()
    for key, jit in model.jit.items(): self.assertIsNotNone(jit.captured, f"jit[{key}] wasn't warmed")

  def test_warmup_then_generate_different_chunk_size(self):
    # T4.12 regression: warmup() always runs at chunk_size=32 internally; a later generate() at a
    # DIFFERENT chunk_size used to hit warmup's prefill jit whose captured "toks" Variable was bound
    # to range 1..32, raising JitError("args mismatch in JIT") on the range-64 slice. Must now work,
    # and produce the same tokens as an unwarmed model (same weights via a fixed seed; greedy decode
    # is otherwise deterministic).
    from dataclasses import replace
    cfg = replace(TEST_CONFIG, max_context=128)
    Tensor.manual_seed(1337)
    warmed = Transformer(cfg)
    warmed.warmup()
    out_warmed = [t for _, t in zip(range(5), warmed.generate(list(range(1, 6)), chunk_size=64))]

    Tensor.manual_seed(1337)
    fresh = Transformer(cfg)  # no warmup() -- baseline
    out_fresh = [t for _, t in zip(range(5), fresh.generate(list(range(1, 6)), chunk_size=64))]

    self.assertEqual(out_warmed, out_fresh)

  def test_mixed_chunk_size_no_recapture_storm(self):
    # T4.12: alternating chunk_size (32 -> 64 -> 32) must not grow the jit dict per-call, and a chunk_size
    # seen before must hit the SAME already-captured jit object, not recapture from scratch.
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=128))

    def run(tokens, chunk_size):
      gen = model.generate(list(tokens), chunk_size=chunk_size)
      for _ in range(3): next(gen)
      model._cached_tokens = []

    run(range(1, 6), 32)
    run(range(6, 11), 32)  # 2nd use of chunk_size=32 -> captures
    prefill_32 = model.jit[(True, True, 32, False)]
    self.assertIsNotNone(prefill_32.captured)

    run(range(20, 25), 64)  # 1st use of chunk_size=64
    run(range(25, 30), 32)  # 3rd use of chunk_size=32 -> must reuse, not recapture

    self.assertIs(model.jit[(True, True, 32, False)], prefill_32)  # same object, no fresh capture
    # bounded to exactly the variants actually used: prefill@32, decode, prefill@64 -- no per-call growth
    self.assertEqual(set(model.jit.keys()), {(True, True, 32, False), (False, True, None, False), (True, True, 64, False)})

  def test_recurrent_warmup_unchanged(self):
    # T4.12: recurrent models force chunk_size=1 in generate() (get_start_pos/generate's ssm branch), so
    # every call is decode-shaped -- warmup() must keep producing only the 2 decode jit variants, no
    # prefill/chunk_size proliferation and no double-capture from the new chunk_size keying.
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    with Context(GDN_CHUNK=1): model.warmup()
    self.assertEqual(set(model.jit.keys()), {(False, True, None, False), (False, False, None, False)})
    for key, jit in model.jit.items(): self.assertIsNotNone(jit.captured, f"jit[{key}] wasn't warmed")

  def test_generate_at_boundary_yields_one_token(self):
    # prompt len == max_context - 1 leaves room for exactly one generated token -- must succeed
    model = Transformer(TEST_CONFIG)
    self.assertIsInstance(next(model.generate(list(range(TEST_CONFIG.max_context - 1)))), int)

  def test_generate_prompt_fills_context_raises(self):
    # T4.6: prompt len == max_context (zero room to generate) must fail loudly, naming max_context,
    # not silently yield nothing (the old `while virtual_len < max_context` behavior)
    model = Transformer(TEST_CONFIG)
    with self.assertRaisesRegex(AssertionError, f"max_context={TEST_CONFIG.max_context}"):
      next(model.generate(list(range(TEST_CONFIG.max_context))))

  def test_generate_prompt_exceeds_context_raises(self):
    # past max_context must also raise the same clear assert, not an opaque reshape shape-mismatch
    model = Transformer(TEST_CONFIG)
    with self.assertRaisesRegex(AssertionError, f"max_context={TEST_CONFIG.max_context}"):
      next(model.generate(list(range(TEST_CONFIG.max_context + 5))))

  def test_first_recurrent_generate_before_state_init(self):
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    with patch.object(Transformer, '__call__', return_value=Tensor([[42]])):
      self.assertEqual(next(model.generate([0])), 42)

  def test_recurrent_live_state_reuse(self):
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    model._cached_tokens = [1, 2, 3, 4, 5]
    self.assertEqual(model.get_start_pos([1, 2, 3, 4, 5, 42, 10]), 5)
    calls = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      calls.append((tokens.shape, start_pos))
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call):
      next(model.generate([1, 2, 3, 4, 5, 42, 10]))
    # resumes from the reused state at position 5 and consumes the 2 new tokens (one chunk or two decode steps)
    self.assertEqual(calls[0][1], V_START_POS.bind(5))
    def ntok(shape): return shape[1] if isinstance(shape[1], int) else shape[1].unbind()[1]
    self.assertEqual(sum(ntok(c[0]) for c in calls), 2)

  def test_recurrent_divergent_prompt_restarts(self):
    model, calls = Transformer(TEST_CONFIG), []
    model.has_recurrent_block, model._cached_tokens = True, [1, 2, 9]
    def mock_call(self, tokens, start_pos, temperature):
      calls.append(start_pos)
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call): next(model.generate([1, 2, 10, 11]))
    self.assertEqual(calls[0], V_START_POS.bind(0))

  def test_template_starts_reasoning(self):
    router = StreamRouter(reasoning=True)
    self.assertEqual(list(router.route("reasoning</think>answer")),
                     [("reasoning_content", "reasoning"), ("content", "answer")])

  def test_kv_cache_reuse(self):
    """Test that generate reuses the KV cache when tokens extend the cached prefix."""
    model = Transformer(TEST_CONFIG)

    captured_inputs = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      captured_inputs.append((tokens.shape, start_pos))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation: prefill 5 tokens + 1 decode
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens)
      next(gen)  # prefill
      next(gen)  # decode

      # second call extends the conversation — cached prefix should be reused
      captured_inputs.clear()
      tokens = [1, 2, 3, 4, 5, 42, 42, 10, 11, 12]
      gen = model.generate(tokens)
      next(gen)

    # should process tokens[6:] = [42, 10, 11, 12] since first 6 have cached k/v
    self.assertEqual(captured_inputs, [((1, V_TOKS.bind(4)), V_START_POS.bind(6))])

  def test_kv_cache_invalidation(self):
    """Test that generate invalidates the KV cache when tokens diverge from the cached prefix."""
    model = Transformer(TEST_CONFIG)

    captured_inputs = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      captured_inputs.append((tokens.shape, start_pos))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation
      gen = model.generate([1, 2, 3, 4, 5])
      next(gen)

      # completely different prompt — KV cache should be invalidated
      captured_inputs.clear()
      gen = model.generate([10, 20, 30])
      next(gen)

    # should process all 3 tokens from start
    self.assertEqual(captured_inputs, [((1, V_TOKS.bind(3)), V_START_POS.bind(0))])

  def test_two_prompts_schedule_cache(self):
    """Third prompt should hit the schedule cache, not miss (first two warm up both jits: prefill + decode)."""
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))

    # first two prompts warm up both jits (prefill + decode)
    ids = list(range(1, 6))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    ids += list(range(10, 15))
    gen = model.generate(ids)
    for _ in range(3): next(gen)
    cache_size_after_warmup = len(schedule_cache)

    # third prompt should reuse the same schedule cache entries, not create new ones
    ids += list(range(20, 25))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    self.assertEqual(cache_size_after_warmup, len(schedule_cache),
      f"third prompt added {len(schedule_cache) - cache_size_after_warmup} new schedule cache entries (expected 0)")

  def test_chunked_prefill(self):
    """When prompt > chunk_size, all chunks should be prefill"""
    from tinygrad.uop.ops import resolve
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))

    def get_prefill_flags(tokens, chunk_size):
      is_prefill = []
      def mock_call(self, tokens, start_pos, temperature, **kwargs):
        is_prefill.append(resolve(tokens.shape[1] != 1))
        return Tensor([[42]])
      with patch.object(Transformer, '__call__', mock_call):
        gen = model.generate(tokens, chunk_size=chunk_size)
        for _ in range(3): next(gen)
      model._cached_tokens = []
      return is_prefill

    # 8 tokens, chunk_size=4 -> 2 prefill chunks
    self.assertEqual(get_prefill_flags(list(range(8)), 4), [True, True, False, False])
    # 9 tokens, chunk_size=4 -> 3 prefill chunks (4+4+1)
    self.assertEqual(get_prefill_flags(list(range(9)), 4), [True, True, True, False, False])
    # 4 tokens, chunk_size=4 -> 1 prefill chunk
    self.assertEqual(get_prefill_flags(list(range(4)), 4), [True, False, False])

  def test_chunked_prefill_kv_cache_matches_single_chunk(self):
    config = TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=1000000, rope_dim=4, qk_norm=4, v_head_dim=4, max_context=16)
    def model():
      m = Transformer(config)
      rng = np.random.RandomState(1234)
      for t in get_state_dict(m).values():
        t.assign(Tensor(rng.uniform(-1, 1, t.shape).astype(np.float32))).realize()
      return m
    def prefill(m, chunk_size):
      gen = m.generate(list(range(1, 9)), chunk_size=chunk_size, temperature=0.0)
      next(gen)
      return [b.cache_kv.numpy() for b in m.blk]
    for g, r in zip(prefill(model(), 4), prefill(model(), 8)):
      np.testing.assert_allclose(g[:, :, :, :8, :], r[:, :, :, :8, :], atol=1e-5)

  def test_kv_cache_resume_matches_fresh(self):
    model = Transformer(TEST_CONFIG)

    # generate 2 tokens, then abandon
    prompt = list(range(1, 6))
    gen = model.generate(list(prompt))
    out1, out2 = next(gen), next(gen)

    # resume with conversation history + new user tokens appended
    extended = prompt + [out1, out2, 10, 11, 12]
    gen = model.generate(list(extended))
    resumed_out = [next(gen) for _ in range(3)]

    # compare against fresh generation (no cache) of the same prompt
    model._cached_tokens = []
    gen = model.generate(list(extended))
    fresh_out = [next(gen) for _ in range(3)]

    self.assertEqual(fresh_out, resumed_out)

  def test_temperature_zero_is_greedy(self):
    """Temperature 0 (or near 0) should produce deterministic output."""
    model = Transformer(TEST_CONFIG)
    tokens = list(range(1, 6))
    results = [list(zip(range(5), model.generate(list(tokens)))) for _ in range(3)]
    # all runs should produce the same tokens
    self.assertEqual(results[0], results[1])
    self.assertEqual(results[1], results[2])

  def test_temperature_high_produces_variety(self):
    """High temperature should produce different outputs across runs."""
    model = Transformer(TEST_CONFIG)
    tokens = list(range(1, 6))
    runs = set()
    for _ in range(5):
      gen = model.generate(list(tokens), temperature=2.0)
      out = tuple(next(gen) for _ in range(10))
      runs.add(out)
    # with temperature=2.0, we should see at least 2 distinct outputs across 5 runs
    self.assertGreater(len(runs), 1, "high temperature should produce varied outputs")

  def test_recurrent_temperature_high_produces_variety(self):
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    outputs = {model.forward(Tensor([[1]]), 0, Tensor([2.0])).item() for _ in range(5)}
    self.assertGreater(len(outputs), 1)

  def test_temperature_passed_to_forward(self):
    """Temperature from generate should be passed through to __call__."""
    model = Transformer(TEST_CONFIG)
    captured_temps = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      captured_temps.append(float(temperature.item()))
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call):
      gen = model.generate([1, 2, 3], temperature=0.6)
      next(gen)
    self.assertAlmostEqual(captured_temps[-1], 0.6, places=5)

SSM_CFG = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100, head_dim=16,
                            rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=64,
                            ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32), ssm_layers=(True, False))

class TestRecurrentChunkedPrefill(unittest.TestCase):
  # T4.55: GDN_CHUNK>1 runs the unrolled T_pad scan for a whole chunk on devices without a fused scan kernel
  def _run(self, chunk:int, prompt:list[int], n:int=4):
    Tensor.manual_seed(7)
    m = Transformer(SSM_CFG)
    with Context(GDN_CHUNK=chunk): out = [t for _, t in zip(range(n), m.generate(list(prompt)))]
    return out, [b.recurrent_state.numpy() for b in m.blk if hasattr(b, "recurrent_state")], m

  def test_chunked_matches_one_token_prefill(self):
    prompt = list(range(1, 10))  # 9 tokens at chunk 4 -> 4+4+1, two chunk boundaries plus a partial chunk
    (o1, s1, _), (o4, s4, m4) = self._run(1, prompt), self._run(4, prompt)
    self.assertEqual(o1, o4)
    for a, b in zip(s1, s4): np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    self.assertIn((True, True, 4, False), m4.jit)  # the prefill jit really was captured at the chunk width

  def test_auto_chunk_is_device_aware(self):
    # auto (GDN_CHUNK=0): 32 only on the GPU backends it was measured on; CPU keeps the one-token-per-step prefill (x86 clang 18
    # crashes on the unrolled 32-step scan kernel -- CI's Test LLM job); an explicit GDN_CHUNK wins everywhere
    from tinygrad.llm.model import gdn_chunk_for
    with Context(GDN_CHUNK=0):
      self.assertEqual([gdn_chunk_for(d) for d in ("METAL", "NV", "NV:1", "CUDA", ("METAL", "NV"))], [32, 32, 32, 32, 32])
      self.assertEqual([gdn_chunk_for(d) for d in ("CPU", "CPU:1", "NULL", "AMD", ("CPU", "METAL"))], [1, 1, 1, 1, 1])
    with Context(GDN_CHUNK=16): self.assertEqual([gdn_chunk_for(d) for d in ("CPU", "METAL")], [16, 16])

  def test_warmup_captures_chunked_prefill(self):
    m = Transformer(SSM_CFG)
    with Context(GDN_CHUNK=4): m.warmup()
    self.assertEqual(set(m.jit.keys()), {(False, True, None, False), (False, False, None, False), (True, True, 4, False), (True, False, 4, False)})
    for key, jit in m.jit.items(): self.assertIsNotNone(jit.captured, f"jit[{key}] wasn't warmed")

CHUNK_LINE_RE = re.compile(
  r"^\[WY_TRACE\] chunk=(\d+) start_pos=(\d+) (blk\d+:state\(nan=\d+ inf=\d+ amax=[^)]+\)/conv\(nan=\d+ inf=\d+ amax=[^)]+\)\s*)+$")

class TestWYTraceChunkInstrumentation(unittest.TestCase):
  """T4.73d PHASE 1: WY_TRACE=2 must print one [WY_TRACE] chunk=... line per PREFILL CHUNK (not just the
  last, like WY_TRACE=1 -- see model.py's _wy_chunk_trace), and WY_DUMP_DIR must save the first non-finite
  chunk's GDN scan inputs to a loadable safetensors file. A poisoned token embedding (one vocab row set to
  NaN, or to a large finite value for the pre-overflow trigger below) gives a deterministic, fast way to
  make a SPECIFIC prefill chunk's GDN state go non-finite/large, without needing the real (numerically
  delicate, content x length dependent) WY bug this instrumentation exists to chase -- NaN/large values
  propagate through the block's linear/conv/scan math regardless of chunk content.

  PHASE 2: real hardware found a block (blk44) that overflows straight to Inf with no NaN of its own -- only
  a DOWNSTREAM block consuming its Inf output produced actual NaNs, and the original nan-count-only trigger
  reported that downstream victim as "the culprit" instead. test_inf_only_block_is_the_reported_culprit
  covers the fix (nan/inf counted separately, trigger on either); test_preoverflow_amax_trigger_fires_while_
  still_finite covers the new WY_DUMP_AMAX trigger this real case also motivated (the non-finite trigger can
  only ever capture a block AFTER it breaks, never the still-finite inputs a per-chunk amplification needs
  to be measured from)."""
  NAN_TOKEN = 50

  def setUp(self):
    from tinygrad.llm.model import wy_dump_amax_done, wy_dump_done
    wy_dump_done.clear()       # module-level "dumped once already" latches -- reset so test order/xdist
    wy_dump_amax_done.clear()  # worker reuse can't wedge either of them

  def _run(self, prompt:list[int], chunk:int, poison_value:float|None, dump_dir:str = "", dump_amax:float = 0.0) -> str:
    Tensor.manual_seed(7)
    m = Transformer(SSM_CFG)
    if poison_value is not None:
      w = m.token_embd.weight.numpy()
      w[self.NAN_TOKEN] = poison_value
      Tensor.realize(m.token_embd.weight.assign(Tensor(w, dtype=m.token_embd.weight.dtype)))
    buf = io.StringIO()
    with Context(GDN_CHUNK=chunk, WY_TRACE=2, WY_DUMP_DIR=dump_dir, WY_DUMP_AMAX=dump_amax), contextlib.redirect_stdout(buf):
      list(zip(range(1), m.generate(list(prompt))))  # only need to reach the end of prefill, not decode
    return buf.getvalue()

  def test_per_chunk_lines_parse_one_per_chunk(self):
    prompt = list(range(1, 10))  # 9 tokens at chunk=3 -> 3 prefill chunks, start_pos 0/3/6
    lines = [ln for ln in self._run(prompt, chunk=3, poison_value=None).splitlines() if ln.startswith("[WY_TRACE] chunk=")]
    self.assertEqual(len(lines), 3, lines)
    for i, line in enumerate(lines):
      match = CHUNK_LINE_RE.match(line)
      self.assertIsNotNone(match, line)
      self.assertEqual(int(match[1]), i + 1, line)  # 1-based chunk index
      self.assertEqual(int(match[2]), i * 3, line)  # start_pos advances by the chunk width

  def test_culprit_chunk_detected_and_dumped(self):
    prompt = [1, 2, 3, 4, self.NAN_TOKEN, 6, 7, 8, 9]  # the NaN token lands in the 2nd of 3 chunks
    with tempfile.TemporaryDirectory() as d:
      log = self._run(prompt, chunk=3, poison_value=np.nan, dump_dir=d)
      chunk_lines = [ln for ln in log.splitlines() if ln.startswith("[WY_TRACE] chunk=")]
      self.assertEqual(len(chunk_lines), 3, chunk_lines)
      self.assertRegex(chunk_lines[0], r"state\(nan=0 inf=0 amax=", "chunk 1 (before the NaN token) must be clean")
      self.assertNotRegex(chunk_lines[1], r"state\(nan=0 inf=0 ", "chunk 2 (contains the NaN token) must go non-finite")
      culprit_lines = [ln for ln in log.splitlines() if ln.startswith("[WY_TRACE] culprit ")]
      self.assertEqual(len(culprit_lines), 1, "must dump at most once, on the FIRST non-finite chunk")
      self.assertTrue(culprit_lines[0].startswith("[WY_TRACE] culprit chunk=2 blk=0 head="), culprit_lines[0])
      dumped = list(pathlib.Path(d).glob("culprit_chunk2_blk0.safetensors"))
      self.assertEqual(len(dumped), 1, list(pathlib.Path(d).iterdir()))
      tensors = safe_load(str(dumped[0]))
      for name in ("state_in", "q", "k", "v", "beta", "alpha", "recurrent_state_out", "conv_state_out"):
        self.assertIn(name, tensors)
      self.assertEqual(tensors["state_in"].shape, tensors["recurrent_state_out"].shape)  # (B,H,V,K), both sides
      self.assertTrue(np.isnan(tensors["recurrent_state_out"].numpy()).any())  # the actual (buggy) observed state

  def test_inf_only_block_is_flagged_non_finite(self):
    # T4.73d PHASE 2 regression: _wy_chunk_fp must count inf independently of nan -- a block with inf=0/
    # nan>0 or nan=0/inf>0 must BOTH trip the "non-finite" culprit trigger (real hardware hit exactly the
    # nan=0/inf>0 case, which the original nan-count-only check silently passed through).
    from tinygrad.llm.model import _wy_chunk_fp
    finite = Tensor([1.0, 2.0, -3.5])
    nan_only = Tensor([1.0, float("nan"), 3.0])
    inf_only = Tensor([1.0, float("inf"), 3.0])
    self.assertEqual(_wy_chunk_fp(finite), (0, 0, 3.5))
    self.assertEqual(_wy_chunk_fp(nan_only)[:2], (1, 0))
    self.assertEqual(_wy_chunk_fp(inf_only)[:2], (0, 1))
    self.assertEqual(_wy_chunk_fp(inf_only)[2], float("inf"))  # abs-max still surfaces the inf

  def test_preoverflow_amax_trigger_fires_while_still_finite(self):
    # no poisoning needed: this tiny random-init model's conv_state (raw attn_qkv projections -- ssm_a/
    # ssm_conv1d.weight default to all-zero, which keeps recurrent_state frozen at exactly 0 here, but
    # conv_state doesn't go through either) already has a nonzero baseline amax every chunk -- a threshold
    # comfortably below that baseline is enough to exercise the trigger mechanism itself (fire once, on the
    # first qualifying chunk, only while every tensor involved is still fully finite).
    prompt = list(range(1, 10))
    with tempfile.TemporaryDirectory() as d:
      log = self._run(prompt, chunk=3, poison_value=None, dump_dir=d, dump_amax=1.0)
      self.assertEqual([ln for ln in log.splitlines() if ln.startswith("[WY_TRACE] culprit ")], [],
                        "must not fire the non-finite trigger -- this prompt never goes non-finite")
      preoverflow_lines = [ln for ln in log.splitlines() if ln.startswith("[WY_TRACE] preoverflow ")]
      self.assertEqual(len(preoverflow_lines), 1, "must dump at most once, on the FIRST chunk crossing the threshold")
      match = re.match(r"^\[WY_TRACE\] preoverflow chunk=(\d+) blk=(\d+) amax=([^ ]+) threshold=1$", preoverflow_lines[0])
      self.assertIsNotNone(match, preoverflow_lines[0])
      self.assertGreaterEqual(float(match[3]), 1.0)
      dumped = list(pathlib.Path(d).glob(f"preoverflow_chunk{match[1]}_blk{match[2]}.safetensors"))
      self.assertEqual(len(dumped), 1, list(pathlib.Path(d).iterdir()))
      tensors = safe_load(str(dumped[0]))
      for name in ("state_in", "q", "k", "v", "beta", "alpha", "recurrent_state_out", "conv_state_out"):
        self.assertTrue(np.isfinite(tensors[name].numpy()).all(), f"{name} must be fully finite (pre-overflow)")

def _byte_tok() -> SimpleTokenizer:
  # byte-level vocab (no merges) in the GPT-2 byte-encoder alphabet: printable ASCII maps to itself, 'Ġ' = space, 'Ċ' = newline
  normal = {chr(b): b for b in range(33, 127)} | {'Ġ': 32, 'Ċ': 10}
  # qwen: eos = eot = <|im_end|>
  return SimpleTokenizer(normal, {"<|im_start|>": 200, "<|im_end|>": 201, "<|endoftext|>": 202}, "qwen2", bos_id=None, eos_id=201, eot_id=201)

class TestSpliceIds(unittest.TestCase):
  def setUp(self):
    self.tok, self.tmpl = _byte_tok(), FallbackTemplate(_byte_tok())
    self.render = lambda msgs, gen: self.tmpl.render(messages=msgs, add_generation_prompt=gen)
    self.hist = [{"role":"system","content":"be brief"}, {"role":"user","content":"hi"}]
    self.prev_rendered = self.render(self.hist, True)
    self.prev_ids = self.tok.encode(self.prev_rendered)
    self.gen = self.tok.encode("ok then")  # what the model generated, up to (not including) its <|im_end|>
    self.last = (self.prev_rendered, self.prev_ids, len(self.hist), self.gen)

  def _splice(self, msgs):
    return splice_ids(self.last, self.render(msgs, True), msgs, self.render, self.tok)

  def test_splices_generated_ids(self):
    msgs = self.hist + [{"role":"assistant","content":"ok   then"}, {"role":"user","content":"more"}]  # client copy re-rendered with other whitespace
    ids = self._splice(msgs)
    self.assertIsNotNone(ids)
    self.assertEqual(ids[:len(self.prev_ids)+len(self.gen)], self.prev_ids + self.gen)  # the model's own ids, not a re-tokenization
    self.assertEqual(self.tok.decode(ids[len(self.prev_ids)+len(self.gen):]), "<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n")
    self.assertNotEqual(ids, self.tok.encode(self.render(msgs, True)))  # a plain encode would have taken the client's whitespace

  def test_tool_call_turn_without_content(self):
    last = (*self.last[:3], self.tok.encode('<tool_call>{"name":"f"}</tool_call>'))
    msgs = self.hist + [{"role":"assistant","content":None,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
                        {"role":"tool","content":"42","tool_call_id":"c1"}]
    ids = splice_ids(last, self.render(msgs, True), msgs, self.render, self.tok)
    self.assertEqual(ids[:len(self.prev_ids)+len(last[3])], self.prev_ids + last[3])
    self.assertTrue(self.tok.decode(ids[len(self.prev_ids)+len(last[3]):]).startswith("<|im_end|>\n<|im_start|>tool\n42<|im_end|>\n"))

  def test_edited_reply_falls_back(self):
    self.assertIsNone(self._splice(self.hist + [{"role":"assistant","content":"something else"}, {"role":"user","content":"more"}]))

  def test_changed_history_falls_back(self):
    msgs = [{"role":"system","content":"be verbose"}] + self.hist[1:] + [{"role":"assistant","content":"ok then"}, {"role":"user","content":"more"}]
    self.assertIsNone(self._splice(msgs))

  def test_empty_end_marker_falls_back(self):
    # a tokenizer whose end-of-turn token decodes to "" (test doubles do this) must not splice: "" would match at the end of any turn
    from unittest.mock import Mock
    tok = Mock(eos_id=999, eot_id=None, decode=Mock(return_value=""), encode=Mock(return_value=[7]))
    msgs = self.hist + [{"role":"assistant","content":None}, {"role":"user","content":"more"}]
    self.assertIsNone(splice_ids(self.last, self.render(msgs, True), msgs, self.render, tok))

  def test_no_assistant_turn_falls_back(self):
    self.assertIsNone(self._splice(self.hist + [{"role":"user","content":"more"}]))

if __name__ == '__main__':
  unittest.main()
