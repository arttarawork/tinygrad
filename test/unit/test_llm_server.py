import unittest
import numpy as np
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.nn.state import get_state_dict
from tinygrad.schedule import schedule_cache
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.serve import StreamRouter, splice_ids, lmstudio_models_payload, template_kwargs, LLMServer
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
    # T4.97: </think> before <tool_call> still routes through content mode first, byte-identical to before change A
    router = StreamRouter(reasoning=True)
    out = list(router.route('reasoning</think>ok <tool_call>{"name":"f"}</tool_call>', final=True))
    self.assertEqual(out, [("reasoning_content", "reasoning"), ("content", "ok ")])
    self.assertEqual((router.mode, router.buf), ("tool", '<tool_call>{"name":"f"}</tool_call>'))

  def test_tool_call_inside_think_block_ends_reasoning(self):
    # T4.97: a well-formed tool call emitted INSIDE the think block, with no </think> at all, must still end up
    # routed for the final tool-call parse, not swallowed as reasoning_content (2026-09-08 production incident).
    import re
    from tinygrad.llm.serve import parse_tool_call
    router = StreamRouter(reasoning=True)
    chunks = ["let me call the tool. ", "<tool_c", 'all>{"name":"f","arguments":{}}</tool_call>']  # mid-tag split
    out = [d for c in chunks for d in router.route(c)]
    out += list(router.route("", final=True))
    self.assertEqual(out, [("reasoning_content", "let me call the tool. ")])  # never leaked as reasoning_content
    self.assertEqual(router.mode, "tool")
    self.assertTrue(router.buf.startswith("<tool_call>"))
    calls = [parse_tool_call(m.group(1)) for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", router.buf, re.DOTALL)]
    self.assertEqual(calls, [("f", {})])

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

class TestLMStudioShim(unittest.TestCase):
  # T4.80: LM Studio's native probe endpoints (Hermes's /reasoning command) + reasoning_effort -> enable_thinking

  def test_template_kwargs_no_overrides(self):
    self.assertEqual(template_kwargs({}), {"preserve_thinking": True})

  def test_template_kwargs_honors_chat_template_kwargs(self):
    self.assertEqual(template_kwargs({"chat_template_kwargs": {"enable_thinking": False}}),
                     {"preserve_thinking": True, "enable_thinking": False})

  def test_reasoning_effort_none_overrides_chat_template_kwargs(self):
    body = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}}
    self.assertEqual(template_kwargs(body)["enable_thinking"], False)

  def test_reasoning_effort_overrides_to_thinking_on(self):
    self.assertEqual(template_kwargs({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}})["enable_thinking"], True)
    self.assertEqual(template_kwargs({"reasoning_effort": "medium"})["enable_thinking"], True)

  def test_reasoning_effort_case_and_whitespace_insensitive(self):
    self.assertEqual(template_kwargs({"reasoning_effort": "NONE "})["enable_thinking"], False)

  def test_template_kwargs_effort_levels(self):
    # T4.91: minimal/low/medium/high/xhigh/max all map onto the Qwen3.8 template's low/medium/xhigh vocabulary; an
    # unrecognized string falls back to medium (the template raises on anything it doesn't recognize, so an unmapped
    # value must never reach it), and an absent reasoning_effort is a no-op (covered by test_template_kwargs_no_overrides).
    cases = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "garbage": "medium"}
    for effort, level in cases.items():
      with self.subTest(effort=effort):
        self.assertEqual(template_kwargs({"reasoning_effort": effort}),
                          {"preserve_thinking": True, "enable_thinking": True, "reasoning_effort": level})

  def test_template_kwargs_none_and_off_disable_thinking(self):
    # thinking off means no reasoning_effort key at all -- the template never reads it then (T4.91).
    self.assertEqual(template_kwargs({"reasoning_effort": "none"}), {"preserve_thinking": True, "enable_thinking": False})
    self.assertEqual(template_kwargs({"reasoning_effort": "off"}), {"preserve_thinking": True, "enable_thinking": False})

  def test_template_kwargs_explicit_reasoning_effort_wins(self):
    # a client's own chat_template_kwargs.reasoning_effort beats the one we derive from the top-level effort (T4.91).
    body = {"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "medium"}}
    self.assertEqual(template_kwargs(body), {"preserve_thinking": True, "enable_thinking": True, "reasoning_effort": "medium"})

  def test_lmstudio_models_payload_shape(self):
    self.assertEqual(lmstudio_models_payload("tiny", 32)["models"], [{
      "key": "tiny", "id": "tiny", "object": "model", "type": "llm", "max_context_length": 32,
      "capabilities": {"reasoning": {"allowed_options": ["none", "minimal", "low", "medium", "high", "xhigh"]}},
      "loaded_instances": [{"id": "tiny", "config": {"context_length": 32}}],
    }])

  def test_lmstudio_and_openai_probe_endpoints_over_http(self):
    import threading, time, json, types, urllib.request, urllib.error
    server = LLMServer(("127.0.0.1", 0), model=types.SimpleNamespace(max_context=32), model_name="tiny", tok=None, template=None)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
      with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/models", timeout=5) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(resp.read()), lmstudio_models_payload("tiny", 32))
      load_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/models/load", data=b"{}",
                                        headers={"Content-Type": "application/json"})
      with urllib.request.urlopen(load_req, timeout=5) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(resp.read()), {"status": "loaded", "model": "tiny", "context_length": 32})
      with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
        self.assertEqual(json.loads(resp.read()), {"object": "list", "data": [{"id": "tiny", "object": "model"}]})
      nope_req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/nope", data=b"{}",
                                        headers={"Content-Type": "application/json"})
      with self.assertRaises(urllib.error.HTTPError) as cm:
        urllib.request.urlopen(nope_req, timeout=5)
      self.assertEqual(cm.exception.code, 404)
    finally:
      server.shutdown()
      server.server_close()

class TestReasoningEffortStandInTemplate(unittest.TestCase):
  """T4.91: render through a small jinja2 template that mirrors the Qwen3.8 template's reasoning_effort CONTRACT (accepts
  low/medium/xhigh, raises otherwise, injects distinguishable text per level) -- independent of that template's actual
  prose, which TestReasoningEffortRealTemplate below checks directly against the GGUF (where, unlike here, 'medium'
  turns out to inject no text at all)."""
  TEMPLATE = (
    "{%- if enable_thinking is defined and enable_thinking is false -%}\n"
    "[thinking off]\n"
    "{%- else -%}\n"
    "{%- set e = reasoning_effort|default('xhigh') -%}\n"
    "{%- if e == 'high' -%}{%- set e = 'xhigh' -%}{%- endif -%}\n"
    "{%- if e not in ('low', 'medium', 'xhigh') -%}\n"
    "{{ raise_exception('Unexpected reasoning effort ' ~ e) }}\n"
    "{%- endif -%}\n"
    "[effort:{{ e }}]\n"
    "{%- endif -%}"
  )

  @classmethod
  def setUpClass(cls):
    import jinja2
    env = jinja2.Environment()
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    cls.template = env.from_string(cls.TEMPLATE)

  def _render(self, reasoning_effort):
    return self.template.render(**template_kwargs({"reasoning_effort": reasoning_effort}))

  def test_low_medium_high_render_distinct_text(self):
    self.assertIn("[effort:low]", self._render("low"))
    self.assertIn("[effort:medium]", self._render("medium"))
    self.assertIn("[effort:xhigh]", self._render("high"))  # Hermes "high" -> template "xhigh"

  def test_none_disables_thinking_with_no_effort_text(self):
    out = self._render("none")
    self.assertIn("[thinking off]", out)
    self.assertNotIn("[effort:", out)

  def test_garbage_effort_never_reaches_the_template(self):
    self._render("garbage")  # our own mapping already sanitized it to "medium" -- must not raise

import os
GGUF_PATH = "/Users/artur/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"

@unittest.skipUnless(os.path.exists(GGUF_PATH), "qwen3.8-27b GGUF not present on this machine")
class TestReasoningEffortRealTemplate(unittest.TestCase):
  """T4.91: render the model's OWN chat template (text loaded from the GGUF header, no weights -- see gguf_load) with our
  derived kwargs. Verified against the template source first: it accepts reasoning_effort in {low, medium, xhigh}
  (raises otherwise; 'high' is its own internal alias for xhigh) but only 'xhigh' and 'low' actually inject an
  instruction sentence into the system prompt -- 'medium' is silently accepted and adds no text."""

  @classmethod
  def setUpClass(cls):
    import pathlib, json, jinja2
    from tinygrad.llm.gguf import gguf_load
    kv, _ = gguf_load(pathlib.Path(GGUF_PATH))
    env = jinja2.Environment()
    env.filters['tojson'] = lambda obj, **kw: json.dumps(obj, **kw)
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    env.globals['strftime_now'] = lambda fmt: ""
    env.globals['bos_token'], env.globals['eos_token'] = "", ""
    cls.template = env.from_string(kv['tokenizer.chat_template'])

  def _render(self, reasoning_effort):
    kwargs = template_kwargs({"reasoning_effort": reasoning_effort})
    return self.template.render(messages=[{"role": "user", "content": "hi"}], tools=None, add_generation_prompt=True, **kwargs)

  def test_low_injects_low_instruction(self):
    self.assertIn("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                  "conclusion without unnecessary elaboration.", self._render("low"))

  def test_high_injects_xhigh_instruction(self):
    self.assertIn("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                  "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity "
                  "in the final answer.", self._render("high"))

  def test_medium_injects_nothing(self):
    self.assertNotIn("Reasoning effort is set to", self._render("medium"))

  def test_thinking_off_renders_closed_think_block(self):
    out = self._render("none")
    self.assertNotIn("Reasoning effort is set to", out)
    self.assertTrue(out.rstrip().endswith("<think>\n\n</think>"))

if __name__ == '__main__':
  unittest.main()

class TestDefaultTemperature(unittest.TestCase):
  """T4.85: an omitted temperature means DEFAULT_TEMPERATURE (0 = greedy, as before); an explicit one always wins."""
  def test_omitted_uses_default_explicit_wins(self):
    import tinygrad.llm.serve as srv
    self.assertEqual(srv.request_temperature({}), 0.0)
    with patch.object(srv, "DEFAULT_TEMPERATURE", 0.6):
      self.assertEqual(srv.request_temperature({}), 0.6)
      self.assertEqual(srv.request_temperature({"temperature": 0}), 0.0)
      self.assertEqual(srv.request_temperature({"temperature": 1.1}), 1.1)

class TestThinkingBudget(unittest.TestCase):
  """T4.88: past the budget the server closes the think block itself and continues the same request from the cached prefix."""
  def test_budget_closes_think_block_and_continues(self):
    from types import SimpleNamespace
    import tinygrad.llm.serve as srv
    from tinygrad.llm.serve import Handler
    # a char-level tokenizer: id == ord(ch); 0 ends the stream
    class Tok:
      def encode(self, s): return [ord(c) for c in s]
      def is_end(self, i): return i == 0
      def stream_decoder(self):
        return lambda i=None: "" if i is None else chr(i)
    calls = []
    def generate(ids, temperature=0.0, vision=None):
      calls.append(list(ids))
      if len(calls) == 1:                       # the model thinks forever
        for c in "think " * 100: yield ord(c)
      else:                                     # after the forced close it answers
        for c in "42": yield ord(c)
        yield 0
    model = SimpleNamespace(get_start_pos=lambda ids: 0, generate=generate, mtp_head=None, max_context=4096)
    h = Handler.__new__(Handler)
    h.server = SimpleNamespace(model=model, tok=Tok(), mtp=False, spec_k=1, state_cache_mb=0, vision=None, last=None)
    ids = [ord(c) for c in "q"]
    out = list(h.run_model(ids, "m", reasoning=True, think_budget=12))
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in out if c["choices"])
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in out if c["choices"])
    self.assertEqual(len(calls), 2)
    self.assertTrue(calls[1][-len(srv.THINK_CLOSE):] == [ord(c) for c in srv.THINK_CLOSE])   # the second generate continues from ids+out+close
    self.assertIn("Considering the limited time", reasoning)
    self.assertLess(len(reasoning), 12 + len(srv.THINK_CLOSE) + 8)                          # the runaway think was cut at the budget
    self.assertEqual(content.strip(), "42")
    self.assertEqual(srv.thinking_budget({"reasoning_effort": "low"}), 0)                    # THINK_BUDGET is 0 in tests: unlimited
    with patch.object(srv, "THINK_BUDGET", 4096):
      efforts = ("minimal", "low", "medium", "high", "xhigh")
      self.assertEqual([srv.thinking_budget({"reasoning_effort": e}) for e in efforts], [512, 1024, 4096, 16384, 0])

class TestReasoningLoopBreaker(unittest.TestCase):
  """T4.89: a sentence cycle inside the think block gets a decisive nudge; after LOOP_NUDGES nudges the block is closed."""
  def test_cycle_is_nudged_then_closed(self):
    from types import SimpleNamespace
    import tinygrad.llm.serve as srv
    from tinygrad.llm.serve import Handler
    class Tok:
      def encode(self, s): return [ord(c) for c in s]
      def is_end(self, i): return i == 0
      def stream_decoder(self):
        return lambda i=None: "" if i is None else chr(i)
    cycle = "Actually, the simplest is: keep two files, zip them. A single self-contained file is more portable. "
    close = [ord(c) for c in srv.THINK_CLOSE]
    calls = []
    def generate(ids, temperature=0.0, vision=None):
      calls.append(list(ids))
      if ids[-len(close):] == close:            # the think block was closed for us: answer
        for c in "42": yield ord(c)
        yield 0
      else:                                     # otherwise keep circling (the server nudges us out of it)
        while True:
          for c in cycle: yield ord(c)
    model = SimpleNamespace(get_start_pos=lambda ids: 0, generate=generate, mtp_head=None, max_context=4096)
    h = Handler.__new__(Handler)
    h.server = SimpleNamespace(model=model, tok=Tok(), mtp=False, spec_k=1, state_cache_mb=0, vision=None, last=None)
    with patch.object(srv, "LOOP_REPEATS", 3), patch.object(srv, "LOOP_NUDGES", 2):
      out = list(h.run_model([ord("q")], "m", reasoning=True))
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in out if c["choices"])
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in out if c["choices"])
    self.assertEqual(len(calls), 4)                                        # initial + 2 nudges + the close
    self.assertEqual(reasoning.count(srv.LOOP_NUDGE.strip()), 2)
    self.assertIn("Considering the limited time", reasoning)
    self.assertLess(reasoning.count("keep two files"), 3 * 3 + 3)         # each round was cut at the third repeat
    self.assertEqual(content.strip(), "42")

  def test_detector_ignores_short_and_resets(self):
    from tinygrad.llm.serve import LoopDetector
    d = LoopDetector(3)
    self.assertIsNone(d.feed("ok. ok. ok. ok. "))                            # <6 words never counts
    for _ in range(2): self.assertIsNone(d.feed("The same long sentence said again and again. "))
    self.assertEqual(d.feed("the same long  sentence said AGAIN and again.\n"), "the same long sentence said again and again.")
    for _ in range(2): self.assertIsNone(d.feed("The same long sentence said again and again. "))   # counts reset after a hit

  def test_detector_ignores_code_ish_sentences(self):
    # T4.97: repeated code/formula lines (2026-09-08: an iterated code edit, an arithmetic checklist) are legitimate
    # repetition, not an anxious loop -- a line carrying '=', ';', '{', '}', a backtick, or '</' never counts.
    from tinygrad.llm.serve import LoopDetector
    d = LoopDetector(3)
    code = "let offtick = i*t16 + t16 - 1;\n"                                # >=6 words, has '=' and ';'
    for _ in range(4): self.assertIsNone(d.feed(code))                      # never fires, however many times repeated
    prose = "Actually, the simplest is: keep two files, zip them.\n"        # a real loop sentence -- no skip chars
    for _ in range(2): self.assertIsNone(d.feed(prose))
    self.assertEqual(d.feed(prose), "actually, the simplest is: keep two files, zip them.")   # still fires at LOOP_REPEATS

  def test_detector_mixed_stream_fires_on_prose_not_code(self):
    from tinygrad.llm.serve import LoopDetector
    d = LoopDetector(3)
    code = "let offtick = i*t16 + t16 - 1;\n"
    prose = "Actually, the simplest is: keep two files, zip them.\n"
    hits = [d.feed(code + prose) for _ in range(3)]                          # code line never counts, prose does
    self.assertEqual(hits, [None, None, "actually, the simplest is: keep two files, zip them."])

class TestKeepAlive(unittest.TestCase):
  """T4.90: empty-delta heartbeats while the generator is silent; a hung-up client stops generation."""
  def _handler(self, wfile):
    from tinygrad.llm.serve import Handler
    h = Handler.__new__(Handler)
    h.wfile, h.send_response, h.send_header, h.end_headers = wfile, lambda *a: None, lambda *a: None, lambda: None
    return h

  def test_heartbeats_fill_the_silence(self):
    import io, json, time
    import tinygrad.llm.serve as srv
    tmpl = {"id":"x", "object":"chat.completion.chunk", "created":1, "model":"m"}
    def gen():
      yield {"choices":[{"index":0, "delta":{"role":"assistant", "content":""}, "finish_reason":None}], **tmpl}
      time.sleep(0.45)                                                      # the "prefill"
      yield {"choices":[{"index":0, "delta":{"content":"hi"}, "finish_reason":None}], **tmpl}
    w = io.BytesIO()
    with patch.object(srv, "KEEPALIVE_SEC", 0.1): self._handler(w).stream_json(gen())
    lines = [json.loads(l[6:]) for l in w.getvalue().decode().split("\n\n") if l.startswith("data: ") and l != "data: [DONE]"]
    beats = [l for l in lines if l["choices"][0]["delta"] == {}]
    self.assertGreaterEqual(len(beats), 2)
    self.assertEqual(beats[0]["model"], "m")                                 # heartbeats carry the reply's template keys
    self.assertEqual([l["choices"][0]["delta"].get("content") for l in lines if l["choices"][0]["delta"]], ["", "hi"])
    self.assertTrue(w.getvalue().endswith(b"data: [DONE]\n\n"))

  def test_hung_up_client_stops_generation(self):
    import io, time
    import tinygrad.llm.serve as srv
    class Gone(io.BytesIO):
      def __init__(self):
        super().__init__()
        self.n = 0
      def write(self, b):
        self.n += 1
        if self.n > 1: raise BrokenPipeError()                              # the client left after the first chunk
        return super().write(b)
    closed, produced = [], []
    def gen():
      try:
        yield {"choices":[{"index":0, "delta":{"role":"assistant", "content":""}, "finish_reason":None}], "model":"m"}
        time.sleep(0.4)                                                      # silence: the heartbeat hits the dead socket
        for i in range(50):
          produced.append(i)
          yield {"choices":[{"index":0, "delta":{"content":"x"}, "finish_reason":None}], "model":"m"}
      finally: closed.append(True)
    with patch.object(srv, "KEEPALIVE_SEC", 0.1): self._handler(Gone()).stream_json(gen())
    self.assertEqual(closed, [True])
    self.assertLessEqual(len(produced), 1)                                   # noticed before the first real token, at most one slipped

  def test_disabled_falls_back_to_plain_stream(self):
    import io
    import tinygrad.llm.serve as srv
    w = io.BytesIO()
    def gen(): yield {"choices":[{"index":0, "delta":{"content":"a"}, "finish_reason":None}], "model":"m"}
    with patch.object(srv, "KEEPALIVE_SEC", 0): self._handler(w).stream_json(gen())
    self.assertEqual(w.getvalue().count(b"data: "), 2)                        # the chunk + [DONE], no heartbeats

class TestStreamLog(unittest.TestCase):
  """T4.83: STREAM_LOG appends the streamed text live, field-tagged, and rotates once past 8 MB."""
  def test_fields_and_rotation(self):
    import os, tempfile
    from tinygrad.llm.serve import StreamLog
    with tempfile.TemporaryDirectory() as d:
      p = os.path.join(d, "s.log")
      with open(p, "w") as f: f.write("x" * 9_000_000)   # oversized leftover -> rotated away first
      log = StreamLog(p, "tiny in:0+3")
      for field, text in (("reasoning_content", "let me "), ("reasoning_content", "think"), ("content", "42")): log.write(field, text)
      log.close()
      self.assertTrue(os.path.exists(p + ".1"))
      body = open(p).read()
      self.assertIn("tiny in:0+3 =====", body)
      self.assertIn("--- reasoning_content ---\nlet me think\n--- content ---\n42", body)

class TestStateCacheOOM(unittest.TestCase):
  """T5.7: a failed snapshot must never abort the request; oversized sequences are skipped; eviction happens before allocation."""
  def _server(self, mb):
    from types import SimpleNamespace
    from tinygrad.llm.serve import LLMServer
    srv = LLMServer(("127.0.0.1", 0), model=SimpleNamespace(max_context=64, snapshot_state=lambda: {}), model_name="tiny", tok=None, template=None,
                    state_cache_mb=mb)
    self.addCleanup(srv.server_close)
    return srv
  def test_one_shot_image_requests_are_not_cached(self):
    srv = self._server(1024)
    srv.store_snapshot([1, 2, 3], True)   # T5.7c: an image request never lands in the cache
    self.assertEqual(len(srv.snapshots), 0)
    srv.store_snapshot([1, 2, 3])
    self.assertEqual(len(srv.snapshots), 1)

  def test_memoryerror_is_swallowed_and_cache_cleared(self):
    srv = self._server(1)
    srv.snapshots[(1, 2)] = {"t": Tensor.zeros(4)}
    def boom(): raise MemoryError("Allocation of 77.06 MB failed on NV")
    srv.model.snapshot_state = boom
    srv.store_snapshot([1, 2, 3, 4])   # must not raise
    self.assertEqual(len(srv.snapshots), 0)
  def test_size_estimate_separates_fixed_state_from_per_token_kv(self):
    # T5.7b: a 4-token snapshot with 1 MB of GDN state and 4 KB/token of KV must predict ~1 MB + n*4 KB, not (1 MB+16 KB)*n/4
    from tinygrad.llm.model import snapshot_nbytes_for
    snap = {"tokens": [1, 2, 3, 4], "pos": 4, "blocks": [{"recurrent_state": Tensor.zeros(256 * 1024)},              # 1 MB fixed (fp32)
                                                          {"cache_kv": Tensor.zeros(2, 1, 1, 4, 512)}]}                 # 4 KB/token (fp32)
    self.assertEqual(snapshot_nbytes_for(snap, 4), 1024 * 1024 + 4 * 4096)
    self.assertEqual(snapshot_nbytes_for(snap, 4000), 1024 * 1024 + 4000 * 4096)
    self.assertLess(snapshot_nbytes_for(snap, 4000), 20 * 1024 * 1024)   # the naive extrapolation would say ~1 GB
  def test_oversized_sequence_is_skipped(self):
    srv = self._server(1)   # 1 MB cap
    # 1 MB of KV for 8 tokens = 128 KB/token
    srv.snapshots[tuple(range(8))] = {"pos": 8, "tokens": list(range(8)), "blocks": [{"cache_kv": Tensor.zeros(256 * 1024)}]}
    calls = []
    srv.model.snapshot_state = lambda: calls.append(1) or {"t": Tensor.zeros(1)}
    srv.store_snapshot(list(range(64)))   # ~8 MB predicted > cap -> skipped without allocating
    self.assertEqual(calls, [])
    self.assertIn(tuple(range(8)), srv.snapshots)
  def test_evicts_before_allocating(self):
    srv = self._server(1)
    # 768 KB of KV for 8 tokens
    srv.snapshots[tuple(range(8))] = {"pos": 8, "tokens": list(range(8)), "blocks": [{"cache_kv": Tensor.zeros(192 * 1024)}]}
    srv.model.snapshot_state = lambda: {"t": Tensor.zeros(192 * 1024)}
    srv.store_snapshot(list(range(8, 16)))   # predicted 768 KB; 768 + 768 > 1 MB -> the old one is evicted first
    self.assertEqual(list(srv.snapshots.keys()), [tuple(range(8, 16))])

  def test_reference_to_oldest_snapshot_is_released_before_reallocating(self):
    # T4.96: the oldest snapshot used to stay referenced (bound to a local name) through the evict loop and the
    # snapshot_state() call right after it, so evicting it from self.snapshots never actually freed its device
    # buffers. A snapshot_state that raises MemoryError while any snapshot it previously handed out is still
    # referenced (checked via weakref, after a gc.collect()) reproduces exactly that failure mode.
    import gc, weakref
    class Snap(dict): pass  # a plain dict doesn't support weakref
    srv = self._server(1)   # 1 MB cap -- fits exactly one 768 KB snapshot
    refs: list[weakref.ref] = []
    def snapshot_state():
      gc.collect()
      if any(r() is not None for r in refs): raise MemoryError("Allocation of 768.00 KB failed on NV")
      snap = Snap(blocks=[{"recurrent_state": Tensor.zeros(192 * 1024)}])   # 768 KB, fixed cost (T5.7b)
      refs.append(weakref.ref(snap))
      return snap
    srv.model.snapshot_state = snapshot_state
    srv.store_snapshot(list(range(8)))
    srv.store_snapshot(list(range(16)))   # 768 KB + 768 KB > 1 MB cap -> evicts the first; it must not still be referenced
    self.assertEqual(list(srv.snapshots.keys()), [tuple(range(16))])

class TestSpliceCacheCopy(unittest.TestCase):
  """T4.95: Transformer.generate()/speculative_generate() append every token they yield straight into the `ids` list
  they're given (real behavior -- see model.py's `tokens.append(int(v)); ...; yield tokens[-1]`). run_model must pass
  a COPY, or that mutation leaks into do_POST's record (server.last, read by splice_ids as the next turn's prev_ids)
  and into inject()'s `ids + out` resume prompt, doubling this turn's own output. The fakes below mutate their `ids`
  argument like the real ones do -- the other fakes in this file don't, which is why they never caught this."""
  def _tok(self):
    class Tok:
      def encode(self, s): return [ord(c) for c in s]
      def is_end(self, i): return i == 0
      def stream_decoder(self):
        return lambda i=None: "" if i is None else chr(i)
    return Tok()

  def test_prompt_list_untouched_and_last_is_pure(self):
    from types import SimpleNamespace
    from tinygrad.llm.serve import Handler
    def generate(ids, temperature=0.0, vision=None):
      for c in "42\0":                      # two real tokens then EOS (id 0)
        ids.append(ord(c))
        model._cached_tokens = ids[:-1]
        yield ord(c)
    model = SimpleNamespace(get_start_pos=lambda ids: 0, generate=generate, mtp_head=None, max_context=4096)
    h = Handler.__new__(Handler)
    server = SimpleNamespace(model=model, tok=self._tok(), mtp=False, spec_k=1, state_cache_mb=0, vision=None, last=None)
    h.server = server
    ids = [ord("q")]
    record = ("<rendered>", ids, 1)          # do_POST builds record from the SAME `ids` object it passes to run_model
    list(h.run_model(ids, "m", record=record))
    self.assertEqual(ids, [ord("q")])                       # the caller's prompt list is untouched by generate()
    self.assertEqual(server.last[1], [ord("q")])             # prev_ids for the next splice_ids call is the pure prompt
    self.assertEqual(server.last[3], [ord("4"), ord("2")])   # generated ids, tracked separately (out) -- EOS excluded

  def test_speculative_path_prompt_also_untouched(self):
    from types import SimpleNamespace
    from tinygrad.llm.serve import Handler
    def speculative_generate(ids, k=3, temperature=0.0):
      for c in "42\0":
        ids.append(ord(c))
        model._cached_tokens = ids[:-1]
        yield ord(c)
    model = SimpleNamespace(get_start_pos=lambda ids: 0, speculative_generate=speculative_generate, mtp_head=object(), max_context=4096)
    h = Handler.__new__(Handler)
    server = SimpleNamespace(model=model, tok=self._tok(), mtp=True, spec_k=3, state_cache_mb=0, vision=None, last=None)
    h.server = server
    ids = [ord("q")]
    record = ("<rendered>", ids, 1)
    list(h.run_model(ids, "m", record=record))
    self.assertEqual(ids, [ord("q")])
    self.assertEqual(server.last[1], [ord("q")])

  def test_inject_resumes_without_duplicating_prior_output(self):
    from types import SimpleNamespace
    import tinygrad.llm.serve as srv
    from tinygrad.llm.serve import Handler
    calls = []
    def generate(ids, temperature=0.0, vision=None):
      calls.append(list(ids))
      if len(calls) == 1:                   # the model thinks forever, mutating its `ids` arg like the real generate()
        for c in "think " * 100:
          ids.append(ord(c))
          model._cached_tokens = ids[:-1]
          yield ord(c)
      else:                                 # after the forced close it answers
        for c in "42\0":
          ids.append(ord(c))
          model._cached_tokens = ids[:-1]
          yield ord(c)
    model = SimpleNamespace(get_start_pos=lambda ids: 0, generate=generate, mtp_head=None, max_context=4096)
    h = Handler.__new__(Handler)
    h.server = SimpleNamespace(model=model, tok=self._tok(), mtp=False, spec_k=1, state_cache_mb=0, vision=None, last=None)
    list(h.run_model([ord("q")], "m", reasoning=True, think_budget=12))
    self.assertEqual(len(calls), 2)
    # the resumed prompt is prompt + (everything generated before the budget hit) + THINK_CLOSE, each token once
    self.assertEqual(calls[1], [ord("q")] + [ord(c) for c in "think think "] + [ord(c) for c in srv.THINK_CLOSE])
class TestBoundarySnapshot(unittest.TestCase):
  """T4.96: only a request that did NOT extend the live cache (cold, or resumed from a snapshot) stores one --
  a tool-loop step that merely extended it must not evict the snapshot that bridges the next real boundary."""
  def _tok(self):
    class Tok:
      def encode(self, s): return [ord(c) for c in s]
      def is_end(self, i): return i == 0
      def stream_decoder(self): return lambda i=None: "" if i is None else chr(i)
    return Tok()

  def test_boundary_only_requests_store_a_snapshot(self):
    from types import SimpleNamespace
    from tinygrad.llm.serve import Handler
    def generate(ids, temperature=0.0, vision=None): yield 0   # ends immediately -- one store opportunity per request
    calls = []
    model = SimpleNamespace(get_start_pos=lambda ids: 0, generate=generate, mtp_head=None, max_context=4096)
    server = SimpleNamespace(model=model, tok=self._tok(), mtp=False, spec_k=1, state_cache_mb=1, vision=None, last=None,
                              find_snapshot=lambda ids: None, store_snapshot=lambda *a: calls.append(a))
    h = Handler.__new__(Handler)
    h.server = server
    list(h.run_model([1, 2, 3], "m"))                  # cold (get_start_pos always 0) -- a boundary request
    self.assertEqual(len(calls), 1)
    model.get_start_pos = lambda ids: 1                 # extends the live cache -- not a boundary
    calls.clear()
    list(h.run_model([1, 2, 3], "m"))
    self.assertEqual(calls, [])                          # a tool-loop-style continuation must not store

  def test_reference_to_restored_snapshot_is_released_before_next_store(self):
    # T4.96: run_model's own `snap := find_snapshot(ids)` walrus is a generator-frame local -- generator frames
    # don't drop locals across yields, so it used to live for the whole request. A snapshot restored early in a
    # request was then still referenced when that same request's store_snapshot tried to evict and replace it.
    import gc, weakref
    from tinygrad.llm.serve import Handler
    class Snap(dict): pass
    class FakeModel:
      mtp_head, max_context = None, 4096
      def __init__(self): self.cached, self.script = [], []
      def get_start_pos(self, ids):
        return len(self.cached) if self.cached and len(self.cached) < len(ids) and ids[:len(self.cached)] == self.cached else 0
      def restore_state(self, snap): self.cached = list(snap["tokens"])
      def generate(self, ids, temperature=0.0, vision=None):
        self.cached = list(ids)
        for t in self.script:
          yield t
          self.cached.append(t)
        yield 0
    model = FakeModel()
    refs: list[weakref.ref] = []
    def snapshot_state():
      gc.collect()
      if any(r() is not None for r in refs): raise MemoryError("Allocation of 768.00 KB failed on NV")
      snap = Snap(tokens=list(model.cached), blocks=[{"recurrent_state": Tensor.zeros(192 * 1024)}])
      refs.append(weakref.ref(snap))
      return snap
    model.snapshot_state = snapshot_state
    srv = LLMServer(("127.0.0.1", 0), model=model, model_name="tiny", tok=self._tok(), template=None, state_cache_mb=1)
    self.addCleanup(srv.server_close)
    h = Handler.__new__(Handler)
    h.server = srv

    model.script = [9, 8]                       # request 1 "thinks" for two tokens -- the live cache ends past [1,2,3]
    list(h.run_model([1, 2, 3], "m"))            # cold prefill -- snapshot A stored at the [1,2,3] boundary
    self.assertEqual(list(srv.snapshots.keys()), [(1, 2, 3)])

    model.script = []                            # request 2 (e.g. reasoning stripped from the rendered prompt):
    list(h.run_model([1, 2, 3, 7], "m"))         # extends A, not the (diverged) live cache -- resumes from A, stores B
    self.assertEqual(list(srv.snapshots.keys()), [(1, 2, 3, 7)])   # A was released before B's allocation
