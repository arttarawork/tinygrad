import contextlib, io, re, unittest
from unittest import mock
from dataclasses import replace
import numpy as np
from typing import cast
from tinygrad import Tensor, nn, UOp, TinyJit
from tinygrad.helpers import Context
from tinygrad.schedule import schedule_cache
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import (
  Transformer, TransformerConfig, TransformerBlock, GatedDeltaNetBlock, MTPHead, SSMConfig, spec_accept_tensor, sample_logits, spec_check_ids,
  spec_k_clamped,
)
from test.unit.test_mtp_load import _build_tiny_qwen35_gguf, _gguf_tensor, VOCAB, DIM, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM
from test.unit.test_llm_device_map import TEST_CONFIG

# T4.64: speculative_generate must be TOKEN-IDENTICAL to generate(temperature=0.0) regardless of draft
# quality (correctness never depends on the MTP head guessing right -- only the accept-iteration count
# does). Reuses test_mtp_load.py's tiny synthetic qwen35-shaped GGUF builder: full_attention_interval=1
# there makes every real block a plain TransformerBlock (no GatedDeltaNet), so this model never exercises
# any of the GDN-specific ACCEPT-step code at all (see _load_gdn below, used by the tests that need to).

def _load(seed:int=0, max_context:int=64) -> Transformer:
  with Context(MTP=1):
    model, _ = Transformer.from_gguf(_gguf_tensor(_build_tiny_qwen35_gguf(seed=seed)), max_context=max_context,
                                     device_map=None, realize=False)
  return model

# T4.66b: a version of _load() whose real blocks include a GatedDeltaNetBlock -- _load()'s GGUF-backed model
# above never has one (see its own comment), so none of the existing tests in this file ever exercised
# GatedDeltaNetBlock._attention's `capture` path or speculative_generate's per-GDN-block ACCEPT-step
# reconstruction at all. Built directly (no GGUF -- test_gdn_scan_parity.py's make_block already establishes
# this is the simpler way to get a small real GatedDeltaNetBlock for tests) with 2 real blocks (0: GDN, 1:
# plain attention -- exercises forward()'s per-block isinstance(..., GatedDeltaNetBlock) branch both ways in
# one model) plus an MTPHead built the same way from_gguf's own MTP branch builds one (mtp_cfg = replace(config,
# qk_norm=config.head_dim) since config.ssm is set, block_cls=TransformerBlock -- MTPHead's own block is always
# attention-only, never GatedDeltaNet, regardless of the owning model's architecture).
def _load_gdn(seed:int=0, max_context:int=64) -> Transformer:
  ssm = SSMConfig(conv_kernel=4, state_size=4, group_count=1, time_step_rank=2, inner_size=8)
  config = TransformerConfig(num_blocks=2, dim=DIM, hidden_dim=HIDDEN, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                              norm_eps=1e-5, vocab_size=VOCAB, head_dim=HEAD_DIM, rope_theta=10000.0,
                              rope_dim=HEAD_DIM, v_head_dim=HEAD_DIM, max_context=max_context,
                              attn_output_gate=True, ssm_layers=(True, False), ssm=ssm)
  model = Transformer(config)
  model.mtp_head = MTPHead(replace(config, qk_norm=config.head_dim), TransformerBlock)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(model)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  Tensor.realize(*params)  # pin weights now -- lazy RNG counter race, see test_gdn_scan_parity.py's make_block
  assert any(isinstance(b, GatedDeltaNetBlock) for b in model.blk) and not all(isinstance(b, GatedDeltaNetBlock) for b in model.blk)
  return model

def _run(model:Transformer, prompt:list[int], n:int, spec:bool, k:int=3) -> list[int]:
  gen = model.speculative_generate(list(prompt), k=k) if spec else model.generate(list(prompt), temperature=0.0)
  return [v for _, v in zip(range(n), gen)]

def _wrong_logits(target:int) -> Tensor: return Tensor([[[100.0 if i == target else -100.0 for i in range(VOCAB)]]])

# T4.71: sized for CI -- the fork's 2-core ubuntu runners run these model tests ~3x slower than the M3 they
# were written on, and the full matrix (k x prompts x N_GEN, a fresh model PAIR per combo) blew both the
# Unit Tests and SPEC=2 job walls once merged. k in {1,3} + one multi-token and one single-token prompt +
# 16 tokens exercise every code path the bigger matrix did (k=2 covers nothing k=1/k=3 don't).
PROMPTS = ([1, 2, 3], [7])
N_GEN = 16

class TestSpeculativeGenerate(unittest.TestCase):
  def test_matches_generate_several_prompts_and_k(self):
    # the core gate: identical output to plain greedy generate(), for every k in {1,2,3} and several prompts
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        ref = _run(_load(seed=seed), prompt, N_GEN, spec=False)
        got = _run(_load(seed=seed), prompt, N_GEN, spec=True, k=k)
        self.assertEqual(got, ref, f"{k=} {seed=} {prompt=}")

  def test_forced_mismatch_still_matches_generate(self):
    # a draft that is ALWAYS wrong -- a fixed constant, ignoring its real inputs entirely -- forces m=0 on
    # essentially every iteration (VOCAB=11, so the real greedy continuation coincidentally landing on the
    # same forced id is rare). Output must still be token-identical: this exercises the m=0/rollback path
    # (GDN checkpoint restore + re-forward) every single iteration, not just the accept path test (a) mostly
    # sees. calls>0 confirms drafting (and therefore the rollback it forces) actually ran.
    wrong_logits = _wrong_logits(VOCAB - 1)
    calls = 0
    def fake_draft(owner, h, tok_ids, start_pos):
      nonlocal calls
      calls += 1
      return wrong_logits
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        ref = _run(_load(seed=seed), prompt, N_GEN, spec=False)
        model = _load(seed=seed)
        model.mtp_head.draft = fake_draft  # instance attribute shadows the class method -- see draft's own
                                            # 4-arg call site in speculative_generate: no self to rebind
        calls = 0
        got = _run(model, prompt, N_GEN, spec=True, k=k)
        self.assertEqual(got, ref, f"{k=} {seed=} {prompt=}")
        self.assertGreater(calls, 0, f"{k=} {seed=} {prompt=}")

  def test_forced_perfect_matches_generate_and_full_accepts(self):
    # a draft that always returns the TRUE next token (peeked from a reference generate() run) must full-
    # accept every iteration. draft() is called exactly k times per outer iteration (see speculative_generate's
    # DRAFT step), so total calls / k pins down the iteration count without needing any new instrumentation
    # on the model: it must equal exactly the number of iterations a run of pure (k+1)-token full accepts
    # would need to cover N_GEN tokens -- proving every iteration actually fully accepted (any partial accept
    # would have needed more, smaller iterations, inflating this count).
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        lookahead = N_GEN + k + 4  # margin: the last iteration's drafts may peek a few positions past N_GEN
        ref = _run(_load(seed=seed), prompt, lookahead, spec=False)
        ref_tokens = list(prompt) + ref  # ref_tokens[i] is the true token AT position i

        calls = 0
        def fake_draft(owner, h, tok_ids, start_pos, _ref=ref_tokens):
          nonlocal calls
          calls += 1
          # T4.66: speculative_generate now passes a bound Variable here, not a plain int (see SPEC_NOTES.md
          # §4) -- unbind() recovers the concrete position for this host-side lookup either way.
          pos = start_pos if isinstance(start_pos, int) else start_pos.unbind()[1]
          return _wrong_logits(_ref[pos + 1])  # "wrong" only in name -- here it's the true next token

        model = _load(seed=seed)
        model.mtp_head.draft = fake_draft
        calls = 0
        got = _run(model, prompt, N_GEN, spec=True, k=k)
        self.assertEqual(got, ref[:N_GEN], f"{k=} {seed=} {prompt=}")
        expected_iters = -(-N_GEN // (k + 1))  # ceil(N_GEN / (k+1))
        self.assertEqual(calls, k * expected_iters, f"{k=} {seed=} {prompt=}")

  def test_draft_reuses_schedule_across_positions(self):
    # T4.66: a drafted position never repeats across a speculative_generate() run. Before this task,
    # MTPHead.draft's start_pos reached it as a plain python int (dpos) -- see SPEC_NOTES.md's old §4 --
    # so every DRAFT call baked a fresh literal into mtp_head.block's @function(precompile=True) trace,
    # growing tinygrad's schedule/program caches without bound over a long generation (confirmed by
    # temporarily reverting the fix locally: schedule_cache grew by a steady +2 EVERY subsequent token,
    # 7 -> 72 over 30 tokens on this same tiny model/settings, never stabilizing -- vs. the fixed code's
    # 7 -> 17 -> 19 then flat for the remaining 27). speculative_generate now binds start_pos to a
    # Variable (v_draft_pos) instead, so the SAME compiled schedule replays at every drafted position --
    # this test fails on the old int-path behavior and passes on the fixed one.
    model = _load(seed=0, max_context=64)
    gen = model.speculative_generate([1, 2, 3], k=2)
    list(zip(range(4), gen))  # warm up: prefill plus this run's first drafted/verified/(maybe) redone shapes
    size_after_warmup = len(schedule_cache)
    list(zip(range(20), gen))  # 20 more tokens' worth of draft/verify/redo at positions never seen before
    self.assertEqual(size_after_warmup, len(schedule_cache),
      f"drafting at new positions added {len(schedule_cache) - size_after_warmup} schedule cache entries "
      "(expected 0 -- MTPHead.draft's start_pos must replay one compiled schedule, not retrace per position)")

  def test_state_integrity_continues_like_a_fresh_generate(self):
    # after speculative_generate stops (mid- or between-iteration), a plain generate() on the SAME model
    # continuing the SAME growing token list (serve.py's splice/_cached_tokens path) must produce exactly
    # what a from-scratch generate() over the whole thing would have -- proving speculative_generate leaves
    # no stale/incorrect state (KV cache, GDN state, _cached_tokens) behind, whichever accept path it took.
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        model_a = _load(seed=seed)
        tokens_so_far = list(prompt)
        got_first = [v for _, v in zip(range(12), model_a.speculative_generate(tokens_so_far, k=k))]
        got_second = [v for _, v in zip(range(6), model_a.generate(tokens_so_far, temperature=0.0))]

        full_ref = _run(_load(seed=seed), prompt, 18, spec=False)
        self.assertEqual(got_first + got_second, full_ref, f"{k=} {seed=} {prompt=}")

  def test_sampled_state_integrity_continues_like_a_fresh_generate(self):
    # T4.65: temperature>0 output isn't reference-comparable token-for-token (it's genuinely random), but
    # the STATE speculative_generate leaves behind must be exactly as if the sampled tokens had really been
    # part of the prompt all along. Verified by comparing a follow-up greedy generate() (continuing model_a
    # in place -- exercising the KV/GDN-cache-reuse path) against a from-scratch greedy generate() on a
    # FRESH model with the same weights (same seed), given the resulting prefix as its prompt: if
    # speculative_generate had left any stale/incorrect state behind, these would diverge.
    Tensor.manual_seed(12345)  # T4.132: every draw is on device
    for k in (1, 2, 3):
      for seed, prompt in enumerate(PROMPTS):
        model_a = _load(seed=seed)
        tokens_so_far = list(prompt)
        got_first = [v for _, v in zip(range(15), model_a.speculative_generate(tokens_so_far, k=k, temperature=1.0))]
        self.assertEqual(len(got_first), 15, f"{k=} {seed=} {prompt=}")
        self.assertTrue(all(0 <= tid < VOCAB for tid in got_first), f"{k=} {seed=} {prompt=}")
        self.assertEqual(len(tokens_so_far), len(prompt) + 15, f"{k=} {seed=} {prompt=}")
        # T4.132: _cached_tokens is the COMMITTED state -- tokens[:-1] at a batch boundary, up to k tokens further inside one
        # (the drafts VERIFY already fed; see speculative_generate's yield loop) -- always extending tokens[:-1]
        cached = model_a._cached_tokens
        self.assertEqual(cached[:len(tokens_so_far) - 1], tokens_so_far[:-1], f"{k=} {seed=} {prompt=}")
        self.assertLessEqual(len(cached) - (len(tokens_so_far) - 1), k, f"{k=} {seed=} {prompt=}")

        # snapshot BEFORE calling generate() on either model: generate() mutates its `tokens` list argument
        # in place (appends as it yields, same as speculative_generate) -- model_a and model_b must each
        # get their OWN copy, or model_a's own continuation would silently extend the list model_b then
        # (wrongly) treats as its prompt.
        prefix_snapshot = list(tokens_so_far)
        got_second = [v for _, v in zip(range(8), model_a.generate(tokens_so_far, temperature=0.0))]

        model_b = _load(seed=seed)
        ref_continuation = [v for _, v in zip(range(8), model_b.generate(prefix_snapshot, temperature=0.0))]
        self.assertEqual(got_second, ref_continuation, f"{k=} {seed=} {prompt=}")

  # --- T4.66b: GDN-specific tests (_load_gdn) -- every test above uses _load()'s model, which has NO
  # GatedDeltaNetBlock at all (see _load's own comment), so none of them ever exercised
  # GatedDeltaNetBlock._attention's `capture` path or speculative_generate's per-GDN-block ACCEPT-step
  # reconstruction (the REDO replacement) at all. See test_gdn_scan_parity.py's TestGDNScanCapture for a
  # lower-level check that capture=True's returned per-position state matches sequential ground truth at
  # every position directly; the tests below check the same thing end-to-end, through speculative_generate.

  def test_matches_generate_several_prompts_and_k_gdn(self):
    # same core gate as test_matches_generate_several_prompts_and_k above, but on a model that actually has a
    # GatedDeltaNetBlock -- exercises the real (non-forced) mtp_head.draft against real per-position GDN
    # state capture/reconstruction, complementing the forced-deterministic test below.
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        ref = _run(_load_gdn(seed=seed), prompt, N_GEN, spec=False)
        got = _run(_load_gdn(seed=seed), prompt, N_GEN, spec=True, k=k)
        self.assertEqual(got, ref, f"{k=} {seed=} {prompt=}")

  def test_forced_partial_accept_matches_generate_gdn(self):
    # T4.66b's core correctness gate: forces a REAL mid-chain partial accept (0 < m < k_eff) against a model
    # that has a GatedDeltaNetBlock, deterministically -- exercising the new REDO-free reconstruction (a
    # capture-based recurrent_state/conv_state fixup read off the verify call itself) specifically, not just
    # the m==0 case test_forced_mismatch_still_matches_generate above already covers (m==0 never needs to read
    # a MIDDLE position out of state_track -- position 0 is also the fresh-checkpoint-adjacent case, and was
    # already exercisable pre-T4.66b via the old restore-then-redo path too). fake_draft is engineered to be
    # right about the FIRST drafted position of every chain and wrong about every later one (a fixed
    # off-vocabulary-ish constant, VOCAB-1) -- with k_eff constant at k (max_context=64 is generous relative
    # to N_GEN here, so the boundary-degrade case in k_eff's own definition never triggers), every iteration's
    # m is deterministically 1, which is genuinely "0 < m < k_eff" whenever k>=2. Checked via SPEC_STATS
    # (closing the generator to force its finally block to fire now, not whenever GC gets to it) rather than
    # merely inferred from the call count, so this test would fail loudly if that engineering ever stopped
    # working (e.g. a future change to draft/verify's position bookkeeping).
    for k in (2, 3):
      for seed, prompt in enumerate(PROMPTS):
        lookahead = N_GEN + k + 4  # margin: the last iteration's drafts may peek a few positions past N_GEN
        ref = _run(_load_gdn(seed=seed), prompt, lookahead, spec=False)
        ref_tokens = list(prompt) + ref

        calls = 0
        def fake_draft(owner, h, tok_ids, start_pos, _ref=ref_tokens):
          nonlocal calls
          pos = start_pos if isinstance(start_pos, int) else start_pos.unbind()[1]
          first_in_chain = calls % k == 0
          calls += 1
          return _wrong_logits(_ref[pos + 1] if first_in_chain else VOCAB - 1)

        model = _load_gdn(seed=seed)
        model.mtp_head.draft = fake_draft
        calls = 0
        buf = io.StringIO()
        with Context(SPEC_STATS=1), contextlib.redirect_stdout(buf):
          gen = model.speculative_generate(list(prompt), k=k)
          got = [v for _, v in zip(range(N_GEN), gen)]
          gen.close()  # forces speculative_generate's try/finally to run (and print) right now
        self.assertEqual(got, ref[:N_GEN], f"{k=} {seed=} {prompt=}")
        self.assertGreater(calls, 0, f"{k=} {seed=} {prompt=}")

        stats_line = buf.getvalue()
        hist_match = re.search(r"accept_len_hist=\{([^}]*)\}", stats_line)
        self.assertIsNotNone(hist_match, f"{k=} {seed=} {prompt=}: SPEC_STATS never printed a histogram: {stats_line!r}")
        hist = {}
        for pair in hist_match.group(1).split(", ") if hist_match.group(1) else []:
          acc_len, count = pair.split(":")
          hist[int(acc_len)] = int(count)
        self.assertIn(1, hist, f"{k=} {seed=} {prompt=}: expected a genuine mid-chain partial accept (m=1) in {hist}")
        self.assertEqual(hist.get(0, 0), 0, f"{k=} {seed=} {prompt=}: m==0 happened, fake_draft's position-0 wasn't honored: {hist}")

  def test_state_integrity_continues_like_a_fresh_generate_gdn(self):
    # same invariant as test_state_integrity_continues_like_a_fresh_generate above, but on the GDN model --
    # that test's base model has no GatedDeltaNetBlock at all, so it never actually checked that
    # speculative_generate leaves correct GDN (conv_state/recurrent_state) state behind, only KV/_cached_tokens.
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        model_a = _load_gdn(seed=seed)
        tokens_so_far = list(prompt)
        got_first = [v for _, v in zip(range(12), model_a.speculative_generate(tokens_so_far, k=k))]
        got_second = [v for _, v in zip(range(6), model_a.generate(tokens_so_far, temperature=0.0))]

        full_ref = _run(_load_gdn(seed=seed), prompt, 18, spec=False)
        self.assertEqual(got_first + got_second, full_ref, f"{k=} {seed=} {prompt=}")

  def test_stop_mid_batch_then_generate_matches_fresh_gdn(self):
    # T4.132: a caller that stops INSIDE an accepted batch (serve.py at a stop token) must still continue correctly: the
    # recurrent state is already ahead of what it has seen (VERIFY fed the whole batch), _cached_tokens says so, and the
    # follow-up generate() re-prefills instead of resuming a state that is 1..m tokens ahead. A perfect draft at k=3 makes
    # every batch 4 tokens: 1 (anchor) + 4 + 2 stops two tokens into the second batch.
    k = 3
    for seed, prompt in enumerate(PROMPTS):
      ref = _run(_load_gdn(seed=seed), prompt, 24, spec=False)
      ref_tokens = list(prompt) + ref
      def fake_draft(owner, h, tok_ids, start_pos, _ref=ref_tokens):
        pos = start_pos if isinstance(start_pos, int) else start_pos.unbind()[1]
        return _wrong_logits(_ref[pos + 1])
      model = _load_gdn(seed=seed)
      model.mtp_head.draft = fake_draft
      toks = list(prompt)
      got = [v for _, v in zip(range(7), model.speculative_generate(toks, k=k))]
      self.assertEqual(got, ref[:7], f"{seed=} {prompt=}")
      self.assertEqual(model._cached_tokens, ref_tokens[:len(prompt) + 8], "committed through the batch's last draft, 2 past the yields")
      got_more = [v for _, v in zip(range(6), model.generate(toks, temperature=0.0))]
      self.assertEqual(got + got_more, ref[:13], f"{seed=} {prompt=}")

class TestSpecTrace(unittest.TestCase):
  """T4.66c: SPEC_TRACE=0 (default, zero overhead) is exercised implicitly by every test above running
  unchanged with it never set -- this only covers the ON path: one line per iteration, every line parses,
  and its phases (each a real sub-interval of the same iteration, timed with time.perf_counter()) never
  exceed the iteration's independently-measured total. See SPEC_TRACE's own ContextVar docstring in
  model.py for what each phase means and which ones are dispatch-only vs. end at a real host sync."""

  TRACE_RE = re.compile(
    r"^\[SPEC_TRACE\] iter=(\d+) k_eff=(\d+) m=(\d+) draft_ms=([\d.]+) draft_dispatch_ms=([\d.]+) "
    r"verify_dispatch_ms=([\d.]+) accept_ms=([\d.]+) state_assign_ms=([\d.]+) total_ms=([\d.]+)$")

  def test_trace_lines_parse_and_phases_sum_to_total(self):
    model = _load_gdn(seed=0)  # a real (tiny) GDN block + MTPHead -- exercises the FIXUP/state_assign path too
    buf = io.StringIO()
    with Context(SPEC_TRACE=1), contextlib.redirect_stdout(buf):
      got = [v for _, v in zip(range(8), model.speculative_generate([1, 2, 3], k=3))]
    self.assertEqual(len(got), 8)
    lines = [line for line in buf.getvalue().splitlines() if line.startswith("[SPEC_TRACE]")]
    self.assertGreater(len(lines), 0, "SPEC_TRACE=1 printed no trace lines")
    for i, line in enumerate(lines, start=1):
      m_ = self.TRACE_RE.match(line)
      self.assertIsNotNone(m_, f"trace line didn't parse: {line!r}")
      iter_n, k_eff, m, draft_ms, draft_dispatch_ms, verify_dispatch_ms, accept_ms, state_assign_ms, total_ms = m_.groups()
      self.assertEqual(int(iter_n), i)  # one line per iteration, in order, no skips/dupes
      self.assertLessEqual(int(m), int(k_eff))
      # +0.02: draft_dispatch_ms <= draft_ms is a hard invariant on the underlying floats (draft_dispatch_ms
      # sums strict sub-intervals of the window draft_ms measures), but each is independently %.2f-rounded,
      # which can push them to adjacent buckets in the wrong direction by up to 0.01ms each.
      self.assertLessEqual(float(draft_dispatch_ms), float(draft_ms) + 0.02,
                            f"draft_dispatch_ms (dispatch-only) exceeded draft_ms (dispatch+sync): {line!r}")
      phases = float(draft_ms) + float(verify_dispatch_ms) + float(accept_ms) + float(state_assign_ms)
      total = float(total_ms)
      # the 4 phases are sequential, non-overlapping sub-intervals of the SAME iteration total_ms independently
      # measures (iter_t0 to just before this line prints) -- phases <= total on the underlying floats is a
      # hard invariant (perf_counter is monotonic), never just approximate; the only slack asserted here is
      # each value's OWN independent %.2f rounding (up to 0.005ms each, 5 values -> 0.025ms, rounded up to
      # 0.05ms), not measurement noise. The only untimed cost is host-only glue between phases (k_eff's min(),
      # n=len(chunk_ids), the stats_on block), which should be tiny next to real tensor-op time.
      self.assertLessEqual(phases, total + 0.05, f"phases summed to more than total_ms: {line!r}")
      self.assertGreaterEqual(phases, total * 0.5, f"untimed residual ate more than half of total_ms: {line!r}")

class TestMTPDraftDeviceLocalEmbed(unittest.TestCase):
  """T4.66d: MTPHead.draft's embedding lookup used to hop tok_ids from mtp_head's device (dmap[-1], where
  tok_ids is always produced -- see MTPHead.draft's own docstring) to token_embd's device (dmap[0]) and hop
  the embedded vector back -- SPEC_PROFILE_NOTES.md's leading candidate for DRAFT's own per-call cost,
  confirmed dominant (~280ms/call) on real METAL+NV hardware (SPEC_FIXES_NOTES.md). Rebuilds the same
  two-device shape TestDeviceMapModel (test_llm_device_map.py) uses -- CPU:0/CPU:1 standing in for METAL/NV,
  token_embd landing on CPU:0 (dmap[0]) and mtp_head on CPU:1 (dmap[-1]), the interesting (non-colocated)
  case -- and checks the draft path's own schedule for cross-device COPY calls once the fix's lazy
  local-embedding-table cache is warm. draft() is never jitted (SPEC_PROFILE_NOTES SS3(a)), so there's no
  `.captured.linear` to read the way TestDeviceMapModel does -- build the schedule directly via
  Tensor.linear_with_vars instead (schedule_linear's own "no vars" assert is a needless risk here: a plain-int
  start_pos is documented to bake in as a literal, never a bound Variable, but linear_with_vars needs no such
  assumption to hold)."""

  def test_no_cross_device_copy_once_warm(self):
    config = replace(TEST_CONFIG, num_blocks=2)
    model = Transformer(config, device_map="CPU:0,CPU:1")
    model.mtp_head = MTPHead(config, TransformerBlock)
    last_dev = model.blk[-1].device
    for p in nn.state.get_parameters(model.mtp_head): p.to_(last_dev)
    Tensor.realize(*nn.state.get_parameters(model))

    # the interesting case: embedding and head start on DIFFERENT devices, mirroring the real pooled map
    # (CLAUDE.md: token_embd on dmap[0], mtp_head entirely on dmap[-1])
    self.assertEqual(model.token_embd.weight.device, "CPU")
    self.assertEqual(model.mtp_head.block.device, "CPU:1")
    self.assertNotEqual(model.token_embd.weight.device, model.mtp_head.block.device)

    h = Tensor.randn(1, 1, config.dim, device=last_dev)
    tok_ids = Tensor([[3]], dtype="int32", device=last_dev)

    # warm call: builds + realizes the one-time local-embedding-table copy (a real, expected COPY) plus this
    # call's own compute -- not what's under test, just get past it.
    model.mtp_head.draft(model, h, tok_ids, 0).realize()
    self.assertTrue(hasattr(model.mtp_head, "_local_token_embd"), "warm call should have built the local cache")

    # the SECOND call reuses the warm cache (the lazy hasattr check short-circuits, so the cache-build COPY
    # never re-enters the graph) -- inspect its schedule (don't realize -- only the plan matters here) for
    # cross-device copies, the same technique TestDeviceMapModel uses on a JIT-captured graph.
    out = model.mtp_head.draft(model, h, tok_ids, 1)
    linear, _ = out.linear_with_vars()
    copies = [call for call in linear.src if call.src[0].op is Ops.COPY]
    self.assertEqual(copies, [], f"draft() should perform zero cross-device copies once warm: {copies}")

class TestSpecDeviceMap(unittest.TestCase):
  """T4.66f: speculative_generate on a REAL multi-device map -- blk[0] and mtp_head/output on DIFFERENT
  devices (CPU:0/CPU:1 standing in for the real hardware's METAL/NV split, same convention
  TestMTPDraftDeviceLocalEmbed/TestDeviceMapModel use elsewhere in this file). Every OTHER speculative-
  decode test in this file uses _load()/_load_gdn() (device_map=None -- a single device, where the bug
  below collapses to a no-op: `dev` and the head's device are the same string) or
  TestMTPDraftDeviceLocalEmbed (multi-device, but calls MTPHead.draft directly once -- never
  speculative_generate's own loop across outer iterations). Neither could have caught a tensor
  speculative_generate itself builds landing on the wrong one of the two devices.

  Bug (pre-fix): a partial accept (m < k_eff) rebuilt tok_last as `Tensor(..., device=dev)` where
  `dev = self.blk[0].device` -- but the DRAFT chain's own tensors (draft_tensors, chained off
  MTPHead.draft's return) are always on `owner.output.weight.device` instead (forced by draft()'s own
  last line, `return owner.output(...)`). Those are the same string on a single device, so every test
  above passes either way; on a real split they're not, and the NEXT outer iteration's greedy
  `tok_last.cat(*draft_tensors, dim=1)` fuses the two into one kernel with no copy between them --
  RuntimeError: all buffers must be on the same device (schedule/__init__.py's assert_all_same_devices).
  Reproduced pre-fix (message: "all buffers must be on the same device: ['CPU', 'CPU:1']"), fixed by
  building that rebuild (and the sampled-path one right above it) on `self.output.weight.device`."""

  def _load_split(self, config:TransformerConfig, seed:int) -> Transformer:
    # ref: single device, real varied weights (mirrors _load_gdn's own randomize-then-realize pattern).
    ref = Transformer(config)
    ref.mtp_head = MTPHead(config, TransformerBlock)
    Tensor.manual_seed(seed)
    for p in nn.state.get_parameters(ref): p.replace(Tensor.randn(*p.shape) * 0.1)
    Tensor.realize(*nn.state.get_parameters(ref))

    # split: same weights (via load_state_dict, which hops each tensor to ITS OWN param's already-placed
    # device -- nn/state.py:214), but blk[0] and blk[-1]/mtp_head/output on different devices -- the
    # from_gguf-mirroring MTP placement TestMTPDraftDeviceLocalEmbed also uses (mtp_head always lands on
    # the LAST block's device, matching output -- see from_gguf's own MTP branch).
    split = Transformer(config, device_map="CPU:0,CPU:1")
    split.mtp_head = MTPHead(config, TransformerBlock)
    for p in nn.state.get_parameters(split.mtp_head): p.to_(split.blk[-1].device)
    nn.state.load_state_dict(split, nn.state.get_state_dict(ref), verbose=False, realize=False)
    split.realize_placement()
    # the whole point of this test: a REAL split, not an incidental same-device map
    self.assertNotEqual(split.blk[0].device, split.output.weight.device)
    self.assertEqual(split.output.weight.device, split.mtp_head.block.device)
    return split

  def test_partial_accept_then_next_iteration_matches_generate(self):
    # a draft that's always wrong forces a partial accept (m=0 -- the exact shape SPEC_TRACE showed on
    # real hardware) on essentially every iteration, same trick as
    # test_forced_mismatch_still_matches_generate above (vocab_size=100 here, even less prone to a
    # coincidental real-token match than that test's VOCAB=11). Built explicitly on
    # owner.output.weight.device, matching the real MTPHead.draft's own contract (its last line forces
    # this device on every return) -- a device-less Tensor(...) here would default to Device.DEFAULT,
    # which happens to equal blk[0]'s device under DEV=CPU and would silently mask the very bug this
    # test exists to catch.
    config = replace(TEST_CONFIG, num_blocks=2)
    def fake_draft(owner, h, tok_ids, start_pos):
      wrong_id = config.vocab_size - 1
      return Tensor([[[100.0 if i == wrong_id else -100.0 for i in range(config.vocab_size)]]],
                    device=owner.output.weight.device)
    for k in (1, 3):
      for seed, prompt in enumerate(PROMPTS):
        ref = _run(self._load_split(config, seed), prompt, N_GEN, spec=False)
        model = self._load_split(config, seed)
        model.mtp_head.draft = fake_draft
        got = _run(model, prompt, N_GEN, spec=True, k=k)  # pre-fix: RuntimeError at the 2nd outer iteration
        self.assertEqual(got, ref, f"{k=} {seed=} {prompt=}")

class TestSpecAcceptTensor(unittest.TestCase):
  """spec_accept_tensor (T4.132: on device, batched; T4.65's host-side numpy spec_accept before it) -- Leviathan et al.'s
  speculative-sampling accept/reject/resample test speculative_generate's temperature>0 path wires in. The uniforms are explicit
  inputs, so every case below is deterministic; the statistical test batches its trials over the leading dim."""

  @staticmethod
  def _logits(*rows:list[float]) -> Tensor:
    # log p (clipped: a 0 becomes exp(-60), ~1e-26) as the logits, so softmax(logits) recovers p -- (1, len(rows), V)
    return Tensor(np.log(np.array(rows, dtype=np.float64)).clip(-60, None).astype(np.float32)).unsqueeze(0)

  def test_identical_distributions_always_accept(self):
    # q==p exactly -> accept_prob=min(1,p/q)=1 everywhere -> ALWAYS accept, regardless of the draw (u in [0,1) is always < 1).
    # Stress it with the draw pushed right up against 1.
    probs = [0.1, 0.5, 0.1, 0.2, 0.1]
    q, p = self._logits(probs, probs), self._logits(probs, probs, probs)  # k=2, +1 bonus row
    m, final = spec_accept_tensor(Tensor([[1, 3]]), q, p, 1.0, Tensor([[0.999999999, 0.999999999]]), Tensor.full((1, 3, 5), 0.5))
    self.assertEqual(m.item(), 2)
    self.assertEqual(final.shape, (1, 3))
    self.assertEqual(final.tolist()[0][2], 1)  # the bonus: equal noise everywhere -> the Gumbel-max is argmax p_k (0.5 at id 1)

  def test_excluded_token_always_rejects_with_correct_resample_support(self):
    # q puts mass on token 0, p excludes it entirely (p[0]=0) -> ratio=0 -> ALWAYS reject regardless of the random draw --
    # and the resample must be distributed EXACTLY as normalize(max(0, p-q)), not just "excludes token 0"
    q0 = [0.5, 0.3, 0.1, 0.1, 0.0]
    p0 = [0.0, 0.4, 0.2, 0.2, 0.2]
    B = 20000
    Tensor.manual_seed(7)
    q, p = self._logits(q0).expand(B, 1, 5), self._logits(p0, [0.2] * 5).expand(B, 2, 5)  # k=1; the bonus row is unused here
    m, final = spec_accept_tensor(Tensor.zeros(B, 1, dtype="int32"), q, p, 1.0, Tensor.full((B, 1), 1e-9), Tensor.rand(B, 2, 5))
    self.assertEqual(m.numpy().max(), 0)  # smallest-possible draw -- still rejects (the ratio is exactly 0)
    residual = np.clip(np.array(p0) - np.array(q0), 0, None)
    expected = residual / residual.sum()  # the docstring's exact formula, computed independently here
    self.assertEqual(expected[0], 0.0)  # token 0 (excluded by p) must never get resample mass
    counts = np.bincount(final.numpy()[:, 0], minlength=5) / B
    self.assertEqual(counts[0], 0.0)
    np.testing.assert_allclose(counts, expected, atol=0.02)

  def test_zero_drafts_samples_bonus_directly(self):
    # k_eff=0 (speculative_generate's max_context boundary degrades to this, see SPEC_NOTES.md's position ledger): no drafts
    # to test at all, straight to a bonus sample from p[0] -- a flat p, so the Gumbel-max picks the largest uniform (id 3)
    m, final = spec_accept_tensor(Tensor.zeros(1, 0, dtype="int32"), Tensor.zeros(1, 0, 5), self._logits([0.2] * 5), 1.0,
                                  Tensor.zeros(1, 0), Tensor([[[0.1, 0.2, 0.3, 0.9, 0.4]]]))
    self.assertEqual((m.tolist(), final.tolist()), ([0], [[3]]))

  def test_min_p_truncates_p_and_q(self):
    # T4.132: min_p (generate()'s MIN_P tail cut) keeps only ids with p >= min_p * max(p), on p and q alike: a draft outside p's
    # cut is always rejected (lp = -inf), the resample is distributed as the truncated residual, token 4 (q's mode) never emitted
    p0, q0 = [0.50, 0.30, 0.15, 0.04, 0.01], [0.01, 0.04, 0.15, 0.30, 0.50]
    B = 20000
    Tensor.manual_seed(5)
    q, p = self._logits(q0).expand(B, 1, 5), self._logits(p0, p0).expand(B, 2, 5)
    m, final = spec_accept_tensor(Tensor.full((B, 1), 4, dtype="int32"), q, p, 1.0, Tensor.full((B, 1), 1e-9), Tensor.rand(B, 2, 5), min_p=0.1)
    self.assertEqual(m.numpy().max(), 0)  # draft 4: p=0.01 < 0.1*0.5 -> cut from p -> rejected even with the smallest draw
    def cut(d): return (t := np.where(np.array(d) >= 0.1 * max(d), d, 0.0)) / t.sum()
    residual = np.clip(cut(p0) - cut(q0), 0, None)
    counts = np.bincount(final.numpy()[:, 0], minlength=5) / B
    np.testing.assert_allclose(counts, residual / residual.sum(), atol=0.02)

  def test_statistical_marginal_matches_p(self):
    # the theorem spec_accept_tensor's whole correctness rests on (Leviathan et al., Theorem 1): if draft_ids are drawn by
    # ancestral sampling from q, the emitted token at that drafted position is marginally distributed as p -- REGARDLESS of q
    # (draft quality only ever affects the accept RATE). Checked empirically over 20k trials on a 5-token vocab (one batched call),
    # since this is exactly the guarantee the sampled speculative-decoding path (and its "distribution-equal to
    # generate(temperature=t)" claim) relies on. The drafts come from sample_logits, the draw speculative_generate itself uses.
    Tensor.manual_seed(42)
    B = 20000
    q = [0.10, 0.50, 0.05, 0.30, 0.05]
    p = [0.40, 0.10, 0.20, 0.05, 0.25]
    q_logits, p_logits = self._logits(q).expand(B, 1, 5), self._logits(p, [0.2] * 5).expand(B, 2, 5)  # k=1, + an unused bonus row
    drafts = sample_logits(q_logits[:, 0], Tensor([1.0]))  # (B,1): ancestral draft samples d ~ q
    m, final = spec_accept_tensor(drafts, q_logits, p_logits, 1.0, Tensor.rand(B, 1), Tensor.rand(B, 2, 5))
    emitted = (m.reshape(B, 1) == 1).where(drafts, final[:, :1])  # the token AT the drafted position: the draft if accepted, else the correction
    empirical = np.bincount(emitted.numpy().ravel(), minlength=5) / B
    np.testing.assert_allclose(empirical, p, atol=0.02)  # loose: ~7 sigma even at p's largest entry (0.40)

class TestOneSyncPerIteration(unittest.TestCase):
  """T4.132: the loop's only host round-trip per outer iteration is ACCEPT's readback, greedy and sampled alike (before: the greedy
  chain synced after DRAFT and again after VERIFY, the sampled one pulled a vocab-wide row to the host per drafted token)."""

  def _syncs_per_iteration(self, temperature:float) -> tuple[int, int]:
    model = _load_gdn(seed=0, max_context=64)  # a real GDN block: the partial-accept restore + REDO path runs too
    syncs: list[str] = []
    orig = {name: getattr(Tensor, name) for name in ("tolist", "numpy", "item")}
    def wrap(name):
      def f(t, *a, **kw):
        syncs.append(name)
        return orig[name](t, *a, **kw)
      return f
    for name in orig: setattr(Tensor, name, wrap(name))
    try:
      buf = io.StringIO()
      with Context(SPEC_STATS=1), contextlib.redirect_stdout(buf):
        gen = model.speculative_generate([1, 2, 3], k=3, temperature=temperature)
        next(gen)  # prefill + the anchor's own .item()
        del syncs[:]
        for _ in range(14): next(gen)
        gen.close()  # SPEC_STATS prints: iters = completed outer iterations (an iteration's syncs all precede its yields)
    finally:
      for name, f in orig.items(): setattr(Tensor, name, f)
    iters = int(cast(re.Match, re.search(r"iters=(\d+)", buf.getvalue())).group(1))
    return len(syncs), iters

  def test_greedy(self):
    syncs, iters = self._syncs_per_iteration(0.0)
    self.assertGreater(iters, 3)
    self.assertEqual(syncs, iters)

  def test_sampled(self):
    Tensor.manual_seed(3)
    syncs, iters = self._syncs_per_iteration(1.0)
    self.assertGreater(iters, 3)
    self.assertEqual(syncs, iters)

def _spec_keys(model:Transformer) -> list[tuple]:
  return [key for key in model.jit if key[3]]  # spec=True is the 4th (is_prefill, greedy, chunk_size, spec) slot

def _verify_key(width:int, redo:bool=False) -> tuple:
  """T4.132: the jit key of a concrete width-`width` VERIFY call (Transformer.__call__: a 1-token call is decode-shaped, a wider
  one stores its width negated in the chunk_size slot so it never collides with a symbolic chunk's "toks" bound); redo=True is the
  partial-accept REDO family of that width (its graph restores the GDN checkpoint first)."""
  return (width > 1, True, -width if width > 1 else None, True) + (("redo",) if redo else ())

class TestVerifyConcreteWidth(unittest.TestCase):
  """T4.132: VERIFY (and REDO) run at CONCRETE widths -- k_eff+1 and m+1, each width its own jit family keyed on the width
  (Transformer.__call__) -- so the static-token-count NV kernels cover them. T4.66e's one symbolic family (a "toks" Variable
  bound at next_power2(k+1), T_pad = the bound) is gone: GatedDeltaNetBlock._attention sees a plain int token axis equal to
  the call's real width, never the prefill chunk_size (32) nor a padded bound."""

  def test_verify_forward_runs_at_its_own_concrete_width(self):
    k = 3
    model = _load_gdn(seed=0, max_context=64)
    widths: list = []
    orig_attention = GatedDeltaNetBlock._attention
    def spy(self, x, start_pos, capture=False):
      widths.append(x.shape[1])
      return orig_attention(self, x, start_pos, capture=capture)
    GatedDeltaNetBlock._attention = spy
    try:
      # GDN_CHUNK=32 pins the prefill chunk width to the 32 the real NV map auto-selects, so a (symbolic) prefill chunk
      # width could still leak into the loop's calls and be caught below
      with Context(GDN_CHUNK=32):
        gen = model.speculative_generate([1, 2, 3], k=k)
        next(gen)  # prefill only -- its own tail chunk legitimately runs the GDN scan at the symbolic chunk_size=32
        widths.clear()  # only care about what happens from here: the outer loop's own VERIFY/REDO calls
        next(gen)  # one full outer iteration: DRAFT + VERIFY (+ REDO on a partial accept), both spec=True
    finally:
      GatedDeltaNetBlock._attention = orig_attention
    self.assertGreater(len(widths), 0, "VERIFY's forward should have run the GDN block's _attention at least once")
    self.assertTrue(all(isinstance(w, int) and 1 <= w <= k + 1 for w in widths), f"not concrete / out of 1..k+1: {widths}")
    self.assertIn(k + 1, widths, f"the first iteration's VERIFY is at the full width k_eff+1={k+1}, saw {widths}")
    loop_keys = [key for key in _spec_keys(model) if key[2] != 32]
    allowed = {_verify_key(w) for w in range(1, k + 2)} | {_verify_key(w, redo=True) for w in range(1, k + 1)}
    self.assertTrue(all(key in allowed for key in loop_keys), loop_keys)

  def test_k_above_the_decode_kernels_is_clamped(self):
    # T4.132: k+1 > NV_SMALL_T would run VERIFY on the generic dequant+matmul path (k=4: ~1.4 s per iteration on the card)
    self.assertEqual([spec_k_clamped(k) for k in (1, 3, 4, 7)], [1, 3, 3, 3])
    model = _load_gdn(seed=0, max_context=64)
    self.assertEqual(_run(model, [1, 2, 3], 8, spec=True, k=5), _run(_load_gdn(seed=0, max_context=64), [1, 2, 3], 8, spec=False))
    self.assertTrue(all(key[2] is None or key[2] >= -4 for key in _spec_keys(model)), _spec_keys(model))

  def test_each_width_is_one_family_captured_once_and_replayed(self):
    # widths 1..k+1 driven directly: the second call of a width captures its family, every later call replays it (cnt grows,
    # no new key), and a different width never shares the family
    k = 3
    model = _load_gdn(seed=0, max_context=64)
    v_sp = UOp.variable("start_pos", 0, 63)
    for w in range(1, k + 2):
      for _ in range(3): model(Tensor([[1] * w], dtype="int32"), v_sp.bind(w), None, spec=True)
      self.assertIn(_verify_key(w), model.jit)
      self.assertIsNotNone(model.jit[_verify_key(w)].captured)
      self.assertEqual(model.jit[_verify_key(w)].cnt, 3)
    self.assertEqual(sorted(_spec_keys(model), key=str), sorted({_verify_key(w) for w in range(1, k + 2)}, key=str))

class TestSpecWarmup(unittest.TestCase):
  """T4.66e deliverable 2: Transformer.warmup() only ever drove generate()'s own jit keys pre-T4.66e --
  speculative_generate's spec=True keys (the prefill tail's and VERIFY/REDO's) were captured at REQUEST time
  instead (the T4.65 gap; recommended fix in T4.72). See warmup()'s own comment for why each key needs calling
  TWICE (engine/jit.py _TinyJit.__call__: a key's first-ever call just runs eagerly, no capture at all -- the
  capture that actually pays off on every later replay is the SAME key's second call). T4.132: every concrete
  VERIFY/REDO width 1..spec_k+1 is its own family and warmup drives each directly."""

  def test_warmup_captures_spec_keys_when_mtp_present(self):
    model = _load_gdn(seed=0, max_context=64)
    self.assertEqual(model.jit, {}, "nothing should be captured before warmup runs")
    model.warmup(spec_k=3)
    spec_keys = _spec_keys(model)
    for w in range(1, 3 + 2): self.assertIn(_verify_key(w), spec_keys, f"VERIFY width {w} not warmed: {spec_keys}")
    for w in range(1, 3 + 1): self.assertIn(_verify_key(w, redo=True), spec_keys, f"REDO width {w} not warmed: {spec_keys}")
    for key in spec_keys:
      self.assertIsNotNone(model.jit[key].captured, f"jit[{key}] was called but never actually captured (still cnt<=1)")
    # T4.132 step 6: the loop's own graphs -- every draft step, the greedy and sampled accept families at n = k+1, the MTP fill/rewrite
    for key in [("draft", i) for i in range(3)] + [("accept", 4, True), ("accept", 4, False)]:
      self.assertIn(key, model.spec_jit, f"{key} not warmed: {sorted(model.spec_jit, key=str)}")
    self.assertTrue(any(key[0] == "mtp_prefill" for key in model.spec_jit), sorted(model.spec_jit, key=str))
    for key, jit in model.spec_jit.items(): self.assertIsNotNone(cast(TinyJit, jit).captured, f"spec_jit[{key}] never captured")
    self.assertEqual(model._cached_tokens, [], "warmup's direct verify calls advanced the live state: nothing may resume from it")

  def test_warmup_without_mtp_head_adds_no_spec_keys(self):
    # a plain Transformer (mtp_head unset, same as _load()'s own model MINUS its Context(MTP=1) --
    # _load() always sets mtp_head, so it can't stand in for "pre-T4.66e" here) mirrors every model built
    # without MTP=1: warmup() must stay exactly as it was -- generate()'s own greedy+sampled keys only,
    # never touching speculative_generate at all.
    model = Transformer(TEST_CONFIG)
    self.assertIsNone(model.mtp_head)
    model.warmup()
    self.assertFalse(any(key[3] for key in model.jit), f"no mtp_head means no spec=True key should exist: {list(model.jit)}")

class TestPerfectDraftFullAcceptance(unittest.TestCase):
  """T4.66g: the task this fixes started from a real-hardware regression (Qwen3.8-27B pooled, k=3): the
  accept-length histogram went from {0:6, 2:4, 3:44} (T4.66b+c, 81% full accepts) to {0:6, 1:5, 2:59}
  (T4.66d+e+f, 0% full accepts) -- the LAST drafted position is never accepted, even though token-identity
  output is still correct (rejection is always safe, so no assertEqual(got, ref)-style test catches an
  acceptance-RATE regression). This is the oracle the task's own METHOD section asks for: force the draft
  chain to be exactly the model's own true continuation (peeked from a reference generate() run via
  MTPHead.draft's 4-arg call site -- an instance-attribute override, the same trick
  test_forced_perfect_matches_generate_and_full_accepts already uses on the non-GDN model). With a
  provably-correct draft, VERIFY must accept the FULL k-token chain every iteration, or the bug is in the
  verify/accept plumbing itself (the comparison loop, verify_chunk padding/indexing, a tok_last rebuild),
  not in draft quality -- exactly the "pin the bug to verify/accept plumbing" split the task's SUSPECTS
  list needed.

  Bisected as instructed (git checkout tinygrad/llm/model.py at 11aac9efc/T4.66b, 2f74b7fb7/T4.66d,
  c7ed888ba/T4.66e, and HEAD/2c444d7e1 T4.66f, re-running this exact test unchanged each time): full
  acceptance every iteration at ALL FOUR commits, k in {1,2,3}, both PROMPTS, both _load_gdn seeds, a
  head-group-split (G>1) GDN config, AND a real CPU:0/CPU:1 split mirroring the hardware's METAL/NV split
  (TestSpecDeviceMap's own _load_split pattern) -- this test PASSES at every one of them, both before and
  after this task's own fix. That rules OUT the verify/accept comparison logic (suspects 1/2: T4.66e's
  verify_chunk/next_power2 padding, T4.66e/f's tok_last rebuilds) as the regression's cause: a wrong
  index/pad there would corrupt the EMITTED token itself (verify_ids[m] directly becomes accepted[-1] on
  the very iteration it goes wrong), which would break token identity with generate() immediately -- but
  the hardware evidence says token identity holds for the whole 70-iteration run. Also rules out suspect 3
  (T4.66d's device-local embedding copy: confirmed bit-identical to the pre-T4.66d cross-device lookup,
  same token ids, same weights) and suspect 4 (the per-position draft_pos Variables: unchanged in every
  commit from T4.66b's own baseline onward, so it cannot explain a regression relative to that baseline).

  See T4.66g's own commit message for where the actual regression was found instead: not an indexing bug
  in the comparison logic at all (which is why this oracle can never fail on it -- forcing the draft's own
  OUTPUT structurally can't expose a bug that only degrades how well an UNFORCED draft predicts), but a
  state leak in warmup() itself (TestWarmupResetsMTPCache below)."""

  def test_perfect_draft_always_fully_accepts(self):
    for k in (1, 2, 3):
      for seed, prompt in enumerate(PROMPTS):
        N_GEN = 16
        lookahead = N_GEN + k + 4
        ref = _run(_load_gdn(seed=seed), prompt, lookahead, spec=False)
        ref_tokens = list(prompt) + ref

        def fake_draft(owner, h, tok_ids, start_pos, _ref=ref_tokens):
          pos = start_pos if isinstance(start_pos, int) else start_pos.unbind()[1]
          return _wrong_logits(_ref[pos + 1])  # "wrong" only in name -- the true next token, always

        model = _load_gdn(seed=seed)
        model.mtp_head.draft = fake_draft
        buf = io.StringIO()
        with Context(SPEC_STATS=1), contextlib.redirect_stdout(buf):
          gen = model.speculative_generate(list(prompt), k=k)
          got = [v for _, v in zip(range(N_GEN), gen)]
          gen.close()
        self.assertEqual(got, ref[:N_GEN], f"{k=} {seed=} {prompt=}")

        stats_line = buf.getvalue()
        hist_match = re.search(r"accept_len_hist=\{([^}]*)\}", stats_line)
        self.assertIsNotNone(hist_match, f"{k=} {seed=} {prompt=}: SPEC_STATS never printed a histogram: {stats_line!r}")
        hist = {}
        for pair in hist_match.group(1).split(", ") if hist_match.group(1) else []:
          acc_len, count = pair.split(":")
          hist[int(acc_len)] = int(count)
        self.assertEqual(set(hist), {k}, f"{k=} {seed=} {prompt=}: a provably-correct draft chain was still "
                          f"not fully accepted every iteration -- accept_len_hist={hist}")

class TestWarmupResetsMTPCache(unittest.TestCase):
  """T4.66g: warmup()'s MTP-warming addition (T4.66e) calls speculative_generate([0]) (twice, see warmup's
  own comment) to pre-capture the VERIFY/REDO jit key. Unlike the main model's own blocks (self.blk), whose
  warmup residue in cache_kv/recurrent_state is harmless (a real request's own prefill legitimately
  rewrites position 0 onward -- GatedDeltaNetBlock's own `initial` reset, and attention's cache simply
  being overwritten at the positions it writes -- before ever attending to it), mtp_head.block is NEVER
  "prefilled": draft() only ever runs from a real request's own start_pos onward (see
  speculative_generate's DRAFT step). Left unreset, warmup's own K/V at low positions (0..~3, from the
  nonsense prompt=[0]) is NEVER revisited/overwritten by a real request and gets attended to by every
  draft() call for the rest of the process's life (causal attention has no window) -- this is exactly the
  "rejection is always safe, so token-identity tests can't see it" blind spot the task's own hardware
  evidence describes: it degrades the MTP head's own next-token guess (draft quality / acceptance rate)
  without ever touching the main model's own state or its output."""

  def test_warmup_leaves_no_cache_on_mtp_block(self):
    model = _load_gdn(seed=0, max_context=64)
    model.warmup()
    # T4.66i: the cache object survives (same identity for every graph that captured it) but holds no warmup K/V
    self.assertTrue(hasattr(model.mtp_head.block, "cache_kv"))
    self.assertEqual(float(model.mtp_head.block.cache_kv.float().abs().max().numpy()), 0.0,
                     "warmup must leave mtp_head.block's cache all-zero: draft() must not attend to the dummy run's K/V")

  def test_stale_warmup_cache_measurably_changes_a_later_draft_call(self):
    # isolates JUST the cache leak: same h/tok_ids/start_pos both times (a position far from warmup's own
    # 0..~3, so it never overlaps with what gets rewritten), the only difference is whether mtp_head.block's
    # own cache still holds warmup's nonsense-prompt K/V at those low positions or has been reset.
    model = _load_gdn(seed=0, max_context=64)
    model.warmup()  # the fix under test: mtp_head.block has no cache right now (see the test above)
    dim = model.mtp_head.block.attn_norm.weight.shape[0]
    h, tok_ids = Tensor.randn(1, 1, dim), Tensor([[3]], dtype="int32")
    clean = model.mtp_head.draft(model, h, tok_ids, 20).numpy()

    # re-pollute EXACTLY like a pre-fix warmup() would have left it (the same dummy calls, minus the reset)
    for _ in range(2): list(zip(range(2), model.speculative_generate([0])))
    polluted = model.mtp_head.draft(model, h, tok_ids, 20).numpy()  # same h/tok_ids/start_pos -- only the cache differs

    self.assertFalse(np.allclose(clean, polluted, atol=1e-5),
                      "mtp_head.block's cache content at low positions had no measurable effect on a "
                      "draft() call at position 20 -- either the pollution isn't real or this stopped "
                      "catching it")

class TestWarmupAnchorShift(unittest.TestCase):
  """T4.66h: T4.66g's own task flagged (but explicitly left out of scope) a DIFFERENT anomaly found while
  isolating its MTP-cache fix: calling the FULL warmup() before a short real prompt could shift the MAIN
  model's own first (anchor) token, even though mtp_head.block's cache leak (T4.66g's own fix, above) is
  unrelated and per-block state (recurrent_state/conv_state/cache_kv) was bit-identical right after the real
  prefill either way.

  T4.66g's own prime suspect was get_start_pos/_cached_tokens splicing a real prompt onto warmup's dummy id
  (0) -- DISPROVEN here: the exact "prompt starts with 0, so it splices onto warmup's own residual state"
  collision (confirmed via direct get_start_pos() inspection: it returns a nonzero start_pos in exactly this
  case) still matches a fresh model's own anchor, at every seed/prompt tried, before AND after this fix --
  get_start_pos's own `tokens[:len(cached)] == cached` guard makes that reuse correct-by-construction
  (identical in kind to snapshot_state/restore_state's own deliberate cross-session prefix reuse), not a bug.
  Prompts that never touch id 0 at all reproduce the real bug just as reliably (see the test below), so it has
  nothing to do with matching warmup's own dummy token.

  The real mechanism: speculative_generate's own prefill-tail anchor read, `tok_all[:, -1:, :]` (and its
  hidden-state twin, `prefill_result[1][:, -1:]`) -- a NEGATIVE index on a Tensor returned from a REPLAYED jit
  call. Per engine/jit.py's _TinyJit.__call__ (already documented in REDO's own comment further up in
  model.py), only a jit key's SECOND-ever call actually captures; every call from then on (capture included)
  returns the SAME frozen Tensor object, whose lazy shape still reports THAT capture call's own bound length
  -- confirmed by monkeypatching Transformer.__call__ to print the returned tensor's shape UOp directly: after
  warmup (which captures this exact key at nt=1, from its own un-skippable `speculative_generate([0])` dummy
  call -- see warmup()'s own T4.66e comment), a real 4-token prompt's first use of this SAME key (is_prefill=
  True, keyed on chunk_size/verify_chunk, never on this call's own bound `nt`) is a replay whose reported
  length is still frozen at 1 -- `tok_all.shape[1]`'s bound CONST reads 1, not 4. `-1:` on that reports
  "length 1, so position 0" and silently returns warmup's OWN single-token anchor instead of the real prompt's
  own last position; the underlying BUFFER holds fully correct per-position data throughout (confirmed
  per-position with concrete, non-negative indices) -- only the negative index's length-dependent resolution
  is stale. The fix: slice with a host-known int, never a jit-returned tensor's own reported length -- `n_toks`
  (the plain Python int already tracking the width of the chunk that produced tok_all/h_last) replaces both
  `-1:` reads. (VERIFY's own `.pad_to(...)`-then-`[:n]` read, which this fix originally cited as the safe idiom,
  had the same hole one layer deeper -- see TestStaleReplayShape, T4.66j.)

  Production exposure: tinygrad/llm/cli.py's `main()` calls `model.warmup()` unconditionally before serving
  any request, and serves real prompts through speculative_generate() whenever MTP is enabled -- there is no
  length-1 guard anywhere on that path, so any --mtp deployment hits this on its first real request whenever
  that request's own final prefill chunk has more than one real token (the overwhelmingly common case for any
  real chat prompt/template, unlike this file's own deliberately tiny/adversarial PROMPTS)."""

  def test_warmup_then_speculative_generate_matches_fresh_anchor(self):
    # neither "prompt starts with warmup's own dummy id 0" nor "prompt avoids id 0 entirely" is special --
    # the bug (pre-fix) fires on ANY prompt whose final prefill chunk has more than 1 real token, once warmup
    # has captured the prefill-tail key at width 1. gdn_chunk=1 is bare-CPU's own auto-selected width (T4.55);
    # 32 is what METAL/NV auto-select (gdn_chunk_for) and what the original hardware regression ran under.
    for gdn_chunk in (1, 32):
      for seed, prompt in enumerate(([0, 1, 2], [1, 2, 3], [5, 9, 3, 2], [7], [0])):
        with self.subTest(gdn_chunk=gdn_chunk, prompt=prompt):
          with Context(GDN_CHUNK=gdn_chunk):
            ref = _run(_load_gdn(seed=seed), prompt, 1, spec=True)
            warm = _load_gdn(seed=seed)
            warm.warmup()
            got = _run(warm, prompt, 1, spec=True)
          self.assertEqual(got, ref, f"{gdn_chunk=} {prompt=}: warmup() shifted speculative_generate's own "
                            "anchor (first emitted) token relative to a fresh model")

  def test_warmup_plain_path_alone_does_not_shift_the_anchor(self):
    # isolates the OTHER half of warmup() (generate()'s own greedy/sampled jit pairs, which run even without
    # mtp_head) from the spec/MTP half fixed above: this half was never broken -- generate()'s own `-1` slice
    # (forward()'s `x[:, -1:]`) lives INSIDE the jit-captured graph itself, not a host-side read of a replayed
    # call's reported shape, so it re-derives correctly from whatever "toks" is bound to on every replay (see
    # this class's own docstring). Standing guard against ever reintroducing an analogous bug on this path.
    for gdn_chunk in (1, 32):
      for seed, prompt in enumerate(([0, 1, 2], [1, 2, 3], [5, 9, 3, 2])):
        with self.subTest(gdn_chunk=gdn_chunk, prompt=prompt):
          with Context(GDN_CHUNK=gdn_chunk):
            ref = _run(_load_gdn(seed=seed), prompt, 1, spec=False)
            warm = _load_gdn(seed=seed)
            warm.mtp_head = None  # isolate: plain generate()-only warmup, no speculative_generate([0])
            warm.warmup()
            got = _run(warm, prompt, 1, spec=False)
          self.assertEqual(got, ref, f"{gdn_chunk=} {prompt=}: plain warmup() shifted generate()'s own anchor")

def _split(ref:Transformer) -> Transformer:
  """A CPU:0/CPU:1 device-map copy of `ref` (mtp_head on the last block's device), weights shared."""
  cfg = ref.blk[0].config
  m = Transformer(cfg, device_map="CPU:0,CPU:1")
  m.mtp_head = MTPHead(replace(cfg, qk_norm=cfg.head_dim), TransformerBlock)
  for p in nn.state.get_parameters(m.mtp_head): p.to_(m.blk[-1].device)
  nn.state.load_state_dict(m, nn.state.get_state_dict(ref), verbose=False, realize=False)
  m.realize_placement()
  return m

def _spec_call(m:Transformer, ids:list[int], nt:int, bound:int) -> tuple[Tensor, Tensor]:
  """One direct spec=True main-model call at width `nt` under a "toks" Variable bounded by `bound` -- i.e. the
  jit key (True, True, bound, True) speculative_generate's VERIFY/REDO use when verify_chunk == bound."""
  v_sp, v_t = UOp.variable("start_pos", 0, m.max_context - 1), UOp.variable("toks", 1, bound)
  t = Tensor([ids[:nt] + [0] * (bound - nt)], dtype="int32", device=m.blk[0].device)
  return cast(tuple[Tensor, Tensor], m(t[:, :v_t.bind(nt)], v_sp.bind(0), None, spec=True))

class TestWarmupSplitTokenIdentity(unittest.TestCase):
  """T4.66i: the guard that would have flagged the T4.66e..g hardware garbage -- speculative_generate AFTER a
  full warmup() (plain + spec dummies) on a real CPU:0/CPU:1 split must be token-identical to a fresh model's
  plain generate(). NOTE: whether this can fail on CPU depends on what width warmup()'s own dummy run happens
  to capture the VERIFY/REDO key at -- here its second-ever call is VERIFY at the full k+1 (confirmed: the
  captured "toks" bound reads 4), so the stale-length reads below never bite; TestStaleReplayShape forces a
  width-1 capture instead."""
  def test_warmup_then_spec_on_split_matches_fresh_generate(self):
    ref = _load_gdn(seed=0, max_context=64)
    for prompt in ([5, 2, 3, 4], [0, 2, 3, 4]):
      want = _run(_split(ref), prompt, 8, spec=False)
      m = _split(ref)
      m.warmup()
      self.assertEqual(_run(m, prompt, 8, spec=True), want, f"{prompt=}")

class TestStaleReplayShape(unittest.TestCase):
  """T4.66j: the hardware run after T4.66h/i had the anchor right but the essay laced with id 0 ("!" in Qwen's
  vocab) at accept boundaries, accept length capped at 2. Same mechanism as T4.66h, one reader further down: a
  replayed jit key returns its CAPTURE call's frozen Tensors, whose symbolic "toks" length stays bound to the
  capture width (REDO on warmup()'s first partial accept captures VERIFY's key at m+1 = 1..k), while the buffer
  underneath is sized to the bound and holds all n real positions. VERIFY's `argmax(-1).pad_to((1, verify_chunk))
  .tolist()[0][:n]` padded FROM that stale length, so every position >= the capture width read back as a pad
  ZERO -- emitted as the bonus/correction token and never equal to a draft. Fixed by reading VERIFY's outputs
  through `verify_logits[:, :n]` (a concrete-shaped shrink bounded by the host-known n; batch 1 keeps its data
  offsets free of "toks") -- both greedy ids and the sampled path's per-position distributions.

  Both tests force the condition deterministically: capture the k=3 VERIFY/REDO key (True, True, 4, True) at
  width 1 with two direct calls (cnt=0 eager, cnt=1 capture -- engine/jit.py), then use it at width 4."""

  def test_verify_after_width1_capture_matches_fresh_generate(self):
    ref = _load_gdn(seed=0, max_context=64)
    prompt = [5, 2, 3, 4]
    want = _run(_split(ref), prompt, 12, spec=False)
    self.assertNotIn(0, want)  # so a pad zero leaking into the emitted stream is visible below, not just a mismatch
    m = _split(ref)
    for _ in range(2): Tensor.realize(*_spec_call(m, prompt, nt=1, bound=4))
    self.assertIsNotNone(m.jit[(True, True, 4, True)].captured, "VERIFY/REDO key must be captured (at width 1) before the real run")
    got = _run(m, prompt, 12, spec=True, k=3)  # pre-fix: [7, 1, 7, 1, 7, 1, 7, 0, 2, 2, 2, 2] -- id 0, then divergence
    self.assertNotIn(0, got, f"{got=}: a pad zero reached the emitted stream")
    self.assertEqual(got, want)

  def test_replayed_outputs_read_through_host_known_width(self):
    # the mechanism itself, for both spec=True outputs: after a width-1 capture, a width-4 replay's [:, :4] view
    # matches a fresh (eager, correctly-shaped) call at every position; the frozen shape's negative index does not.
    ref = _load_gdn(seed=0, max_context=64)
    prompt = [5, 2, 3, 4]
    m, fresh = _split(ref), _split(ref)
    for _ in range(2): Tensor.realize(*_spec_call(m, prompt, nt=1, bound=4))
    r_logits, r_h = _spec_call(m, prompt, nt=4, bound=4)  # replay: reported length stays at the capture's 1
    f_logits, f_h = _spec_call(fresh, prompt, nt=4, bound=4)  # cnt=0: eager, shape bound to this call's own 4
    np.testing.assert_allclose(r_logits[:, :4].numpy(), f_logits[:, :4].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r_h[:, :4].numpy(), f_h[:, :4].numpy(), rtol=1e-5, atol=1e-6)
    self.assertEqual(r_logits[:, :4].argmax(-1).tolist(), f_logits[:, :4].argmax(-1).tolist())
    # the stale reported length, made visible: -1: on the replay is position 0, not the real last position
    self.assertGreater(float(np.abs(r_h[:, -1:].numpy() - f_h[:, 3:4].numpy()).max()), 1e-3)

class TestSpecKVConsistency(unittest.TestCase):
  """T4.132: after a speculative run the main model's state must be exactly the state of the tokens it yielded: every attention
  block's KV at positions 0..len(tokens)-2 and every GDN block's conv/recurrent state equal a from-scratch forward over tokens[:-1]
  (tok_last, the last yield, is not fed yet). blk0's K/V at a position are a function of that position's token alone, so this is the
  direct check that VERIFY/REDO fed the ids the loop reported. Caught: REDO's input built as a `buf[:, :m+1]` view of VERIFY's chunk
  buffer -- a concrete shrink view handed to a TinyJit is baked at capture (`.contiguous()` folds into the view, nothing is
  parametrized), so every later REDO re-fed the capture call's ids. The token-identity tests never saw it: their tiny models repeat
  one token, so the baked id was the right id."""

  def _assert_state_of(self, model:Transformer, ref:Transformer, toks:list[int], label:str):
    L = len(toks) - 1
    Tensor.realize(*_spec_call(ref, toks, nt=L, bound=L))  # eager (cnt=0), concrete: the state after exactly tokens[:-1]
    for i, (a, b) in enumerate(zip(model.blk, ref.blk)):
      if isinstance(a, GatedDeltaNetBlock):
        np.testing.assert_allclose(a.conv_state.numpy(), b.conv_state.numpy(), rtol=1e-2, atol=1e-2, err_msg=f"{label} blk{i} conv_state")
        np.testing.assert_allclose(a.recurrent_state.numpy(), b.recurrent_state.numpy(), rtol=1e-2, atol=1e-2, err_msg=f"{label} blk{i} recurrent")
      else:
        np.testing.assert_allclose(a.cache_kv[:, :, :, :L].float().numpy(), b.cache_kv[:, :, :, :L].float().numpy(), rtol=1e-2, atol=1e-2,
                                   err_msg=f"{label} blk{i} cache_kv")

  def test_sampled_run_leaves_the_state_of_its_tokens(self):
    # the sampled path's token stream varies (greedy on these tiny models repeats one id); k=1 so most iterations REDO at width 1.
    # token_embd.weight is realized on both sides: MTPHead._block_in realizes it in place on the first draft (a numerics-only shift)
    for seed in (0, 1):
      Tensor.manual_seed(seed)
      model, ref = _load(seed=seed), _load(seed=seed)
      model.token_embd.weight.realize(), ref.token_embd.weight.realize()
      toks = [7]
      list(zip(range(15), model.speculative_generate(toks, k=1, temperature=1.0)))
      self.assertGreater(len(set(toks)), 3, f"{toks=}: the sampled stream repeated itself, this run proves nothing")
      self._assert_state_of(model, ref, toks, f"{seed=} {toks=}")

  def test_greedy_redo_with_an_alternating_stream_gdn(self):
    # deterministic: _load_gdn(seed=0) on [5, 2, 3, 4] alternates two ids (TestStaleReplayShape's sequence); an always-wrong draft
    # forces m=0 and a width-1 REDO every iteration, so the REDO inputs alternate -- a baked REDO input re-feeds one of them
    for k in (1, 3):
      model = _load_gdn(seed=0)
      model.mtp_head.draft = lambda owner, h, tok_ids, start_pos: _wrong_logits(VOCAB - 1)
      toks = [5, 2, 3, 4]
      list(zip(range(12), model.speculative_generate(toks, k=k)))
      self.assertGreater(len(set(toks[4:])), 1, f"{toks=}")
      self._assert_state_of(model, _load_gdn(seed=0), toks, f"{k=} {toks=}")

class TestMTPPrefill(unittest.TestCase):
  """T4.66l: mtp_head.block's KV cache holds the prompt's MTP context (MTPHead.prefill, run chunk by chunk beside the
  main prefill) and every accepted position's TRUE-hidden entry (the accept-time rewrite) -- the semantics the nextn
  head was trained on: the step whose inputs are (h_t, tok_{t+1}) lives in slot t+1. Before this, the prompt slots
  were all-zero (MTP_CACHE_NOTES.md section 6: the honest b..j baseline was 8% full chains on correct text)."""

  def test_prefill_matches_chained_draft(self):
    # the T-wide prefill must leave exactly the cache T single-step draft() calls (the pre-existing, trusted
    # definition of one MTP step) leave for the same (h_t, tok_{t+1}) inputs at slots t+1
    ref = _load_gdn(seed=0, max_context=64)
    prompt = [5, 2, 3, 4, 9, 1, 7]
    n = len(prompt)
    a, b = _split(ref), _split(ref)
    _, h = _spec_call(a, prompt, nt=n, bound=n)  # eager (cnt=0): concrete (1, n, D)
    v_t, v_p = UOp.variable("mtp_toks", 1, 8), UOp.variable("mtp_pos", 0, 63)
    a.mtp_head.prefill(a, h[:, :n - 1], prompt[1:], 1, v_t, v_p).realize()
    v_d = UOp.variable("draft_pos_0", 0, 63)
    for t in range(n - 1): b.mtp_head.draft(b, h[:, t:t + 1], Tensor([[prompt[t + 1]]], dtype="int32"), v_d.bind(t + 1)).realize()
    ca, cb = (m.mtp_head.block.cache_kv[:, :, :, :n + 1, :].float().numpy() for m in (a, b))
    np.testing.assert_allclose(ca[:, :, :, 1:n], cb[:, :, :, 1:n], rtol=1e-4, atol=1e-6)
    self.assertEqual(float(np.abs(ca[:, :, :, 0]).max()), 0.0)  # slot 0 (step -1) is never written
    self.assertEqual(float(np.abs(ca[:, :, :, n]).max()), 0.0)  # slot n (step n-1 needs tok_n: the anchor, draft()'s job)
    self.assertGreater(float(np.abs(ca[:, :, :, 1:n]).max()), 0.0)

  def test_generate_maintains_cache_and_snapshot_round_trips(self):
    # after chunked prefill + draft chains with a mix of accept lengths (0..k, exercising both the full-accept and
    # the REDO path's rewrite), every slot 1..len(tokens)-2 must match a one-shot prefill over the final token
    # sequence on a fresh model; then snapshot_state/restore_state must carry the MTP cache too.
    k = 3
    ref = _load_gdn(seed=0, max_context=64)
    prompt = [5, 2, 3, 4, 9]
    want = _run(_split(ref), prompt, N_GEN + k + 4, spec=False)
    ref_tokens = list(prompt) + want
    m = _split(ref)
    orig, calls = m.mtp_head.draft, 0
    def fake_draft(owner, h, tok_ids, start_pos, _ref=ref_tokens):
      nonlocal calls
      orig(owner, h, tok_ids, start_pos).realize()  # the real draft still runs, for its K/V side effect (slot start_pos)
      chain, step = divmod(calls, k)
      calls += 1
      pos = start_pos.unbind()[1]
      # chain i accepts i mod (k+1) drafts; on out_dev, where the real draft's logits live (split model: see out_dev in model.py)
      return _wrong_logits(_ref[pos + 1] if step < chain % (k + 1) else VOCAB - 1).to(owner.output.weight.device)
    m.mtp_head.draft = fake_draft
    buf = io.StringIO()
    with Context(SPEC_STATS=1), contextlib.redirect_stdout(buf):
      got = _run(m, prompt, N_GEN, spec=True, k=k)
    self.assertEqual(got, want[:N_GEN])
    hist = buf.getvalue().split("accept_len_hist=")[-1]
    self.assertTrue(all(f"{i}:" in hist for i in range(k + 1)), buf.getvalue())
    tokens = list(prompt) + got
    L = len(tokens)
    r = _split(ref)
    _, h = _spec_call(r, tokens, nt=L - 1, bound=L - 1)
    v_t, v_p = UOp.variable("mtp_toks", 1, L), UOp.variable("mtp_pos", 0, 63)
    r.mtp_head.prefill(r, h[:, :L - 2], tokens[1:L - 1], 1, v_t, v_p).realize()
    live = m.mtp_head.block.cache_kv[:, :, :, :L, :].float().numpy()
    oracle = r.mtp_head.block.cache_kv[:, :, :, :L, :].float().numpy()
    np.testing.assert_allclose(live[:, :, :, 1:L - 1], oracle[:, :, :, 1:L - 1], rtol=1e-3, atol=1e-4)
    # snapshot/restore round trip carries the MTP cache (live slice [:pos])
    snap = m.snapshot_state()
    self.assertIn("mtp_cache_kv", snap)
    self.assertEqual(int(snap["mtp_cache_kv"].shape[3]), snap["pos"])
    m.mtp_head.block.cache_kv.assign(Tensor.zeros_like(m.mtp_head.block.cache_kv)).realize()
    m.restore_state(snap)
    back = m.mtp_head.block.cache_kv[:, :, :, :L, :].float().numpy()
    np.testing.assert_array_equal(back[:, :, :, :snap["pos"]], live[:, :, :, :snap["pos"]])

  def test_prefill_only_fills_the_mtp_cache(self):
    # T4.132: with an MTP head loaded prefill_only runs speculative_generate's prefill, so the draft block's cache holds the
    # prompt's context (slots 1..n-1) exactly as a speculative_generate call's own prefill leaves it -- what a --mtp server's
    # prefix snapshot (serve.py, T4.111) carries -- and _cached_tokens is the prompt, as with generate()
    prompt = [5, 2, 3, 4, 9, 1, 7]
    a, b = _load_gdn(seed=0), _load_gdn(seed=0)
    a.prefill_only(prompt)
    self.assertEqual(a._cached_tokens, prompt)
    next(b.speculative_generate(list(prompt), k=1))
    n = len(prompt)
    ca, cb = (m.mtp_head.block.cache_kv[:, :, :, :n, :].float().numpy() for m in (a, b))
    np.testing.assert_array_equal(ca, cb)
    self.assertGreater(float(np.abs(ca[:, :, :, 1:n]).max()), 0.0)

def _load_gdn_varied(seed:int=0, max_context:int=128) -> Transformer:
  """T4.132 (10-30): a tiny GDN+attention model whose greedy stream is NOT one repeating id. _load()/_load_gdn()'s models emit
  [1, 1, ...] / [1, 7, 1, 7, ...]: a VERIFY forward that read an UNWRITTEN input buffer still produced those ids, so every
  token-identity test above passed while the card produced garbage. Six blocks in the 27B's pattern (GDN, GDN, GDN, attention,
  GDN, GDN), dim 64, vocab 512, weights at 0.5 scale -- the first 12 greedy ids are all different."""
  ssm = SSMConfig(conv_kernel=4, state_size=16, group_count=2, time_step_rank=4, inner_size=64)
  config = TransformerConfig(num_blocks=6, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=512, head_dim=16,
                             rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=max_context, attn_output_gate=True,
                             ssm_layers=(True, True, True, False, True, True), ssm=ssm)
  model = Transformer(config)
  model.mtp_head = MTPHead(replace(config, qk_norm=config.head_dim), TransformerBlock)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(model)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.5)
  Tensor.realize(*params)
  return model

VARIED_PROMPT = [5, 2, 3, 4, 1, 7, 3, 9, 11, 2, 40, 100, 7]

class TestSpecForwardMatchesDecode(unittest.TestCase):
  """T4.132 (10-30, the card bisect): forward(spec=True) -- VERIFY/REDO -- was wrong at EVERY concrete width, T=1 included, while
  generate() was right, independent of every kernel knob. GatedDeltaNetBlock.checkpoint returned the block's input as
  AFTER(x, <snapshot assigns>). In this scheduler (schedule/__init__.py create_schedule) a reader of an AFTER state depends only
  on that AFTER's OWN kernels (its srcs[1:] that are CALL/END); an AFTER wrapped around an input with no kernels of its own has
  none, so the block lost its dependency on x's producer (it read an unwritten buffer) and the snapshot ordering edge hung off
  nothing. The checkpoint now orders the block's STATE reads after the copies (Tensor.assign's STORE+AFTER idiom on the buffer
  being snapshotted). These tests compare rows of logits on a model whose greedy ids vary, with the probe's own eager calls."""

  def setUp(self):
    self.model = _load_gdn_varied()
    self.ref = _run(self.model, VARIED_PROMPT, 12, spec=False)  # plain greedy: the ground truth
    self.assertGreaterEqual(len(set(self.ref)), 8, f"{self.ref}: this model's greedy stream must vary for these tests to mean anything")
    self.first = next(iter(self.model.generate(list(VARIED_PROMPT), temperature=0.0)))  # state = the prompt, _cached_tokens = the prompt
    self.assertEqual(self.first, self.ref[0])
    self.snap = self.model.snapshot_state()
    self.pos = self.snap["pos"]
    self.gdn = [b for b in self.model.blk if isinstance(b, GatedDeltaNetBlock)]

  def _spec_rows(self, ids:list[int], pos:int) -> np.ndarray:  # eager, no jit: the card probe's own call (spec_bisect.py)
    logits, _ = cast(tuple[Tensor, Tensor], self.model.forward(Tensor([ids]), pos, None, spec=True))
    return logits.float().numpy()[0]

  def test_t1_chain_and_every_width_match_generate(self):
    self.model.restore_state(self.snap)
    chain, cur = [], self.first
    for i in range(4):
      cur = int(self._spec_rows([cur], self.pos + i)[-1].argmax())
      chain.append(cur)
    self.assertEqual(chain, self.ref[1:5], "T=1 spec forwards fed one token at a time must reproduce generate()'s ids")
    chunk = [self.first] + self.ref[1:4]  # what a VERIFY of three correct drafts feeds; the right verdicts are ref[1:5]
    for T in range(1, 5):
      self.model.restore_state(self.snap)
      rows = self._spec_rows(chunk[:T], self.pos)
      self.assertFalse(np.isnan(rows).any(), f"T={T}: NaN logits")
      self.assertEqual(rows.argmax(-1).tolist(), self.ref[1:1 + T], f"T={T}: VERIFY rows disagree with generate()")

  def test_width4_rows_equal_the_t1_chain_logits(self):
    chunk = [self.first] + self.ref[1:4]
    want = []
    for i in range(4):
      self.model.restore_state(self.snap)
      if i: self._spec_rows(chunk[:i], self.pos)  # advance the state through the first i tokens, then one more
      want.append(self._spec_rows([chunk[i]], self.pos + i)[-1])
    self.model.restore_state(self.snap)
    rows = self._spec_rows(chunk, self.pos)
    for i in range(4): np.testing.assert_allclose(rows[i], want[i], rtol=1e-3, atol=1e-3, err_msg=f"row {i}")

  def test_checkpoint_orders_the_copy_first_and_leaves_plain_buffers(self):
    # the forward with the checkpoint == the forward without it (bit-exact: same kernels, only ordering edges differ); spec_snap is
    # the pre-call state; the live state tensors are plain buffers again afterwards (no AFTER view lingers into the next call)
    chunk = [self.first] + self.ref[1:4]
    orig = GatedDeltaNetBlock.checkpoint
    GatedDeltaNetBlock.checkpoint = lambda self, x, restore=False: None  # type: ignore[method-assign]
    try:
      self.model.restore_state(self.snap)
      want_rows = self._spec_rows(chunk, self.pos)
      want_post = [(b.conv_state.numpy(), b.recurrent_state.numpy()) for b in self.gdn]
    finally: GatedDeltaNetBlock.checkpoint = orig
    self.model.restore_state(self.snap)
    pre = [(b.conv_state.numpy(), b.recurrent_state.numpy()) for b in self.gdn]
    rows = self._spec_rows(chunk, self.pos)
    np.testing.assert_array_equal(rows, want_rows)
    for i, (b, (pc, pr), (qc, qr)) in enumerate(zip(self.gdn, pre, want_post)):
      self.assertTrue(b.conv_state.uop.has_buffer_identity() and b.recurrent_state.uop.has_buffer_identity(), f"gdn {i}: an AFTER view lingers")
      assert b.spec_snap is not None
      np.testing.assert_array_equal(b.spec_snap[0].numpy(), pc, err_msg=f"gdn {i} conv snapshot")
      np.testing.assert_array_equal(b.spec_snap[1].numpy(), pr, err_msg=f"gdn {i} recurrent snapshot")
      np.testing.assert_array_equal(b.conv_state.numpy(), qc, err_msg=f"gdn {i} conv after")
      np.testing.assert_array_equal(b.recurrent_state.numpy(), qr, err_msg=f"gdn {i} recurrent after")
    # RESTORE (speculative_generate's partial-accept path) puts the pre-call state back exactly
    Tensor.realize(*(x for b in self.gdn for x in (b.conv_state.assign(cast(tuple, b.spec_snap)[0]),
                                                    b.recurrent_state.assign(cast(tuple, b.spec_snap)[1]))))
    for i, (b, (pc, pr)) in enumerate(zip(self.gdn, pre)):
      np.testing.assert_array_equal(b.conv_state.numpy(), pc, err_msg=f"gdn {i} conv restored")
      np.testing.assert_array_equal(b.recurrent_state.numpy(), pr, err_msg=f"gdn {i} recurrent restored")

  def test_speculative_generate_matches_generate_on_varied_ids(self):
    for k in (1, 3):
      self.assertEqual(_run(_load_gdn_varied(), VARIED_PROMPT, 12, spec=True, k=k), self.ref, f"{k=}")

  def test_nv_decode_kernels_see_the_checkpointed_state(self):
    # the card path: GatedDeltaNetBlock through nv.gdn_decode's two kernels (gate mocked open, CPU form) with the checkpoint active,
    # at concrete widths 1 and 3, against the generic block with the same checkpoint -- output, both states, both snapshots
    from test.unit.test_llm_nv_scan import tiny_block, decode_gate, TINY_DIM
    fused, generic = tiny_block(), tiny_block()
    with decode_gate(): fused.prepare_nv_decode()
    self.assertIsNotNone(fused.nv_decode_weights)
    xs = Tensor.randn(1, 8, TINY_DIM).realize()
    pos = 0
    for T in (1, 3, 1, 3):
      pre = {}
      for name, block in (("fused", fused), ("generic", generic)):
        block._init_state(xs)
        pre[name] = (block.conv_state.numpy(), block.recurrent_state.numpy())
        block.checkpoint(xs[:, pos:pos + T])
        with decode_gate() if name == "fused" else contextlib.nullcontext():
          pre[name] += (cast(Tensor, block(xs[:, pos:pos + T].contiguous(), pos)).numpy(),)
      np.testing.assert_allclose(pre["fused"][2], pre["generic"][2], rtol=1e-3, atol=1e-5, err_msg=f"{pos=} {T=} output")
      for name, block in (("fused", fused), ("generic", generic)):
        assert block.spec_snap is not None
        np.testing.assert_array_equal(block.spec_snap[0].numpy(), pre[name][0], err_msg=f"{name} {pos=} {T=} conv snapshot")
        np.testing.assert_array_equal(block.spec_snap[1].numpy(), pre[name][1], err_msg=f"{name} {pos=} {T=} recurrent snapshot")
        self.assertFalse(np.array_equal(block.recurrent_state.numpy(), pre[name][1]), f"{name} {pos=} {T=}: the state did not advance")
      np.testing.assert_array_equal(fused.conv_state.numpy(), generic.conv_state.numpy(), err_msg=f"{pos=} {T=} conv")
      np.testing.assert_allclose(fused.recurrent_state.numpy(), generic.recurrent_state.numpy(), rtol=1e-5, atol=1e-7,
                                 err_msg=f"{pos=} {T=} recurrent")
      pos += T

class TestSpecDebugAndIdGuard(unittest.TestCase):
  """T4.132 (10-30): SPEC_DEBUG=1 prints one stderr line per iteration with the verify rows' NaN/inf counts, the verify argmax ids,
  the chunk and the emitted ids; an id outside [0, vocab) (an argmax over a NaN row reads back as the row width) raises instead of
  being emitted -- greedy and sampled, anchor and loop."""
  DEBUG_RE = re.compile(r"^\[SPEC_DEBUG\] start_pos=(\d+) k_eff=(\d+) m=(\d+) chunk=(\[[^\]]*\]) verify_argmax=(\[[^\]]*\]) final=(\[[^\]]*\]) "
                        r"nan_per_row=(\[[^\]]*\]) inf_per_row=(\[[^\]]*\])$")

  def test_debug_lines_parse_and_are_clean_on_a_healthy_model(self):
    model = _load_gdn_varied()
    err = io.StringIO()
    with Context(SPEC_DEBUG=1), contextlib.redirect_stderr(err):
      got = [v for _, v in zip(range(9), model.speculative_generate(list(VARIED_PROMPT), k=3))]
    lines = [line for line in err.getvalue().splitlines() if line.startswith("[SPEC_DEBUG]")]
    self.assertGreater(len(lines), 0)
    emitted = []
    for line in lines:
      m_ = self.DEBUG_RE.match(line)
      self.assertIsNotNone(m_, f"debug line didn't parse: {line!r}")
      _, k_eff, m, chunk, verify_argmax, final, nan_rows, inf_rows = m_.groups()
      chunk_l, verify_l, final_l = eval(chunk), eval(verify_argmax), eval(final)  # noqa: S307 -- ints lists the test itself printed
      self.assertEqual(len(chunk_l), int(k_eff) + 1)
      self.assertEqual(verify_l, final_l, "greedy: the emitted candidates are the verify argmax ids")
      self.assertEqual(eval(nan_rows), [0] * (int(k_eff) + 1))
      self.assertEqual(eval(inf_rows), [0] * (int(k_eff) + 1))
      emitted += chunk_l[1:1 + int(m)] + [final_l[int(m)]]
    self.assertEqual(emitted[:len(got) - 1], got[1:])  # the lines describe exactly what was yielded after the anchor

  def test_nan_verify_logits_raise_instead_of_emitting_ids(self):
    for temperature in (0.0, 1.0):
      model = _load_gdn_varied()
      orig = Transformer.__call__
      def nan_verify(self, tokens, start_pos, temperature, spec=False, *args, **kwargs):
        out = orig(self, tokens, start_pos, temperature, spec, *args, **kwargs)
        if spec and isinstance(tokens.shape[1], int) and tokens.shape[1] > 1:  # the loop's VERIFY (concrete, wider than one token)
          return out[0] * float("nan"), out[1]
        return out
      err = io.StringIO()
      with mock.patch.object(Transformer, "__call__", nan_verify), Context(SPEC_DEBUG=1), contextlib.redirect_stderr(err):
        gen = model.speculative_generate(list(VARIED_PROMPT), k=3, temperature=temperature)
        next(gen)  # the anchor comes from the prefill tail: still fine
        with self.assertRaisesRegex(RuntimeError, "outside vocab 512") as ctx: next(gen)
      self.assertIn("NaN logits upstream", str(ctx.exception))
      lines = [line for line in err.getvalue().splitlines() if line.startswith("[SPEC_DEBUG]")]
      self.assertEqual(len(lines), 1, err.getvalue())
      self.assertIn("nan_per_row=[512, 512, 512, 512]", lines[0], f"{temperature=}: {lines[0]!r}")

  def test_spec_check_ids(self):
    spec_check_ids([0, 5, 511], 512, "ok")
    for bad in ([512], [-1], [3, 512, 4]):
      with self.assertRaisesRegex(RuntimeError, "outside vocab 512"): spec_check_ids(bad, 512, "bad")

class TestSpecLaunchCensus(unittest.TestCase):
  """T4.132 step 6: once warm, one speculative iteration is k_eff draft replays + VERIFY + the accept replay + (full accept) the MTP
  rewrite replay or (partial) the REDO replay that restores the GDN checkpoint inside its graph -- and a handful of 1-2 kernel eager
  launches (the next anchor and h_last uploaded/copied as jit inputs, the chunk cat, REDO's / the rewrite's input prep). Before: ~93
  eagerly dispatched draft launches, 96 eager restore assigns per partial accept, the accept math and the rewrite trace per iteration."""

  def _census(self, temperature:float) -> list[list[tuple[str, int]]]:
    import tinygrad.tensor as tensor_mod, tinygrad.engine.jit as jit_mod
    log: list[tuple[str, int]] = []
    orig_t, orig_j = tensor_mod.run_linear, jit_mod.run_linear
    def spy(name, orig):
      def f(linear, *a, **kw):
        log.append((name, sum(1 for c in linear.src if c.op is Ops.CALL)))
        return orig(linear, *a, **kw)
      return f
    tensor_mod.run_linear, jit_mod.run_linear = spy("eager", orig_t), spy("replay", orig_j)
    try:
      model = _load_gdn_varied()
      gen = model.speculative_generate(list(VARIED_PROMPT), k=3, temperature=temperature)
      for _ in range(9): next(gen)  # prefill + the first iterations: traces and captures
      per_yield = []
      for _ in range(12):
        n0 = len(log)
        next(gen)
        per_yield.append(log[n0:])
      gen.close()
    finally: tensor_mod.run_linear, jit_mod.run_linear = orig_t, orig_j
    return [p for p in per_yield if p]  # a yield inside an accepted batch launches nothing

  def test_warm_iteration_is_replays_plus_tiny_eager_launches(self):
    for temperature in (0.0, 1.0):
      iters = self._census(temperature)
      self.assertGreater(len(iters), 3)
      for launches in iters:
        eager, replays = [k for t, k in launches if t == "eager"], [k for t, k in launches if t == "replay"]
        self.assertLessEqual(len(eager), 5, f"{temperature=}: too many eager launches in one iteration: {launches}")
        self.assertLessEqual(max(eager), 2, f"{temperature=}: an eager launch of more than 2 kernels (a restore? a draft chain?): {launches}")
        # drafts (3) + VERIFY + accept, then the REDO (partial accept) and/or the MTP rewrite (m > 0)
        self.assertIn(len(replays), (6, 7), f"{temperature=}: {launches}")
        self.assertGreaterEqual(min(replays[:3]), 10, f"{temperature=}: the first three replays are the draft steps: {launches}")

if __name__ == '__main__':
  unittest.main()
