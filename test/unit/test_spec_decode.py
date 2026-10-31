import contextlib, io, os, re, tempfile, unittest
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
  spec_k_clamped, linear_rows,
)
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT
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

def _wrong_logits_v(target:int, vocab:int) -> Tensor: return Tensor([[[100.0 if i == target else -100.0 for i in range(vocab)]]])
def _wrong_logits(target:int) -> Tensor: return _wrong_logits_v(target, VOCAB)

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

    # T4.142: a draft vocabulary lives on the head's device too -- the slice, the id map and full_vocab's gather add no hop
    model.mtp_head.set_draft_vocab(model, list(range(0, config.vocab_size, 2)))
    out = model.mtp_head.draft(model, h, tok_ids, 1)
    self.assertEqual(out.shape, (1, 1, config.vocab_size))
    linear, _ = out.linear_with_vars()
    copies = [call for call in linear.src if call.src[0].op is Ops.COPY]
    self.assertEqual(copies, [], f"draft() under a draft vocabulary should perform zero cross-device copies: {copies}")

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

  def _syncs_per_iteration(self, temperature:float, draft_vocab:list[int]|None=None) -> tuple[int, int]:
    model = _load_gdn(seed=0, max_context=64)  # a real GDN block: the partial-accept restore + REDO path runs too
    if draft_vocab is not None: model.mtp_head.set_draft_vocab(model, draft_vocab)  # T4.142
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

  def test_draft_vocab_adds_no_sync(self):
    # T4.142: the subset pick, the id map and the full-vocab q are gathers inside the captured chain -- still one readback per iteration
    for temperature in (0.0, 1.0):
      Tensor.manual_seed(3)
      syncs, iters = self._syncs_per_iteration(temperature, draft_vocab=list(range(0, VOCAB, 2)))
      self.assertGreater(iters, 3)
      self.assertEqual(syncs, iters, f"{temperature=}")

def _spec_keys(model:Transformer) -> list[tuple]:
  return [key for key in model.jit if key[3]]  # spec=True is the 4th (is_prefill, greedy, chunk_size, spec) slot

def _verify_key(width:int) -> tuple:
  """T4.132: the jit key of a concrete width-`width` VERIFY call (Transformer.__call__: a 1-token call is decode-shaped, a wider
  one stores its width negated in the chunk_size slot so it never collides with a symbolic chunk's "toks" bound)."""
  return (width > 1, True, -width if width > 1 else None, True)

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
    allowed = {_verify_key(w) for w in range(1, k + 2)}
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
      self.assertEqual(model.jit[_verify_key(w)].cnt, 4)  # T4.132: captured on its first call (cnt starts at 1)
    self.assertEqual(sorted(_spec_keys(model), key=str), sorted({_verify_key(w) for w in range(1, k + 2)}, key=str))

class TestSpecWarmup(unittest.TestCase):
  """T4.66e deliverable 2: Transformer.warmup() only ever drove generate()'s own jit keys pre-T4.66e --
  speculative_generate's spec=True keys (the prefill tail's and VERIFY/REDO's) were captured at REQUEST time
  instead (the T4.65 gap; recommended fix in T4.72). See warmup()'s own comment for why each key needs calling
  TWICE (engine/jit.py _TinyJit.__call__: a key's first-ever call just runs eagerly, no capture at all -- the
  capture that actually pays off on every later replay is the SAME key's second call). T4.132: every concrete
  VERIFY width 1..spec_k+1 is its own family and warmup drives each directly, and the partial-accept adopt family."""

  def test_warmup_captures_spec_keys_when_mtp_present(self):
    model = _load_gdn(seed=0, max_context=64)
    self.assertEqual(model.jit, {}, "nothing should be captured before warmup runs")
    model.warmup(spec_k=3)
    spec_keys = _spec_keys(model)
    for w in range(1, 3 + 2): self.assertIn(_verify_key(w), spec_keys, f"VERIFY width {w} not warmed: {spec_keys}")
    for key in spec_keys:
      self.assertIsNotNone(model.jit[key].captured, f"jit[{key}] was called but never actually captured (still cnt<=1)")
    # T4.132: the loop's own graphs -- the draft chain and the greedy and sampled accept families at every k_eff 1..3 (n = k_eff+1), the
    # partial-accept adopt, the prompt's MTP fill
    for key in [("chain", i) for i in (1, 2, 3)] + [("accept", n, g) for n in (2, 3, 4) for g in (True, False)] + [("adopt",)]:
      self.assertIn(key, model.spec_jit, f"{key} not warmed: {sorted(model.spec_jit, key=str)}")
    self.assertTrue(any(key[0] == "mtp_prefill" for key in model.spec_jit), sorted(model.spec_jit, key=str))
    for key, jit in model.spec_jit.items(): self.assertIsNotNone(cast(TinyJit, jit).captured, f"spec_jit[{key}] never captured")
    self.assertEqual(model._cached_tokens, [], "warmup's direct verify calls advanced the live state: nothing may resume from it")
    # T4.132: the prompt prefill replays generate()'s own families -- no spec=True prefill (symbolic chunk) family exists
    self.assertEqual([key for key in spec_keys if isinstance(key[2], int) and key[2] > 0], [])

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
    a.mtp_head.prefill(a, h[:, :n - 1], prompt[1:], 1, v_t, v_p)
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
    r.mtp_head.prefill(r, h[:, :L - 2], tokens[1:L - 1], 1, v_t, v_p)
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

  def _gdn_states(self) -> list[tuple[np.ndarray, np.ndarray]]: return [(b.conv_state.numpy(), b.recurrent_state.numpy()) for b in self.gdn]

  def test_adopt_equals_the_state_of_the_accepted_tokens(self):
    # T4.132: a partial accept of m drafts adopts VERIFY's per-token slot m (Transformer._adopt_step) -- bit for bit the GDN state a fresh
    # forward of exactly the m+1 accepted tokens leaves (what T4.66d's REDO recomputed), at every m; at m == 0 (one fed token) also
    # bit for bit plain decode's. (m >= 1 differs from plain decode by fp16 conv rounding: a multi-token run keeps its conv window in
    # fp32, decode rounds conv_state between steps -- generate() vs any VERIFY chunk, unchanged by T4.132.) The live state tensors are
    # plain buffers again afterwards.
    v_m = UOp.variable("spec_m", 0, 2)
    for m in range(3):
      chunk = [self.first] + self.ref[1:1 + m] + [(self.ref[1 + m] + 1) % 512] + [7] * (2 - m)  # d_m is wrong: accept length m
      self.model.restore_state(self.snap)
      rows = self._spec_rows(chunk, self.pos)
      self.assertEqual(rows.argmax(-1).tolist()[:m + 1], self.ref[1:2 + m], f"{m=}: VERIFY rows")
      self.model._adopt_step(v_m.bind(m))
      got = self._gdn_states()
      self.model.restore_state(self.snap)
      self._spec_rows(chunk[:m + 1], self.pos)
      for i, ((gc, gr), (wc, wr)) in enumerate(zip(got, self._gdn_states())):
        np.testing.assert_array_equal(gc, wc, err_msg=f"{m=} gdn {i} conv")
        np.testing.assert_array_equal(gr, wr, err_msg=f"{m=} gdn {i} recurrent")
      if m == 0:
        self.model.restore_state(self.snap)
        Tensor.realize(*cast(tuple[Tensor, Tensor], self.model.forward(Tensor([[self.first]]), self.pos, None)))  # (token, hidden)
        for i, ((gc, gr), (wc, wr)) in enumerate(zip(got, self._gdn_states())):
          np.testing.assert_array_equal(gc, wc, err_msg=f"gdn {i} conv vs decode")
          np.testing.assert_array_equal(gr, wr, err_msg=f"gdn {i} recurrent vs decode")
      for b in self.gdn: self.assertTrue(b.conv_state.uop.has_buffer_identity() and b.recurrent_state.uop.has_buffer_identity())

  def test_verify_output_and_final_state_unchanged_by_the_track(self):
    # the per-token stores are a pure addition: VERIFY's rows and final state == the same forward with the track stores stripped (the
    # per-token loop scan either way: capture forces it, GDN_SCAN_IMPL=1 picks it for the plain call -- WY reassociates the state math)
    chunk = [self.first] + self.ref[1:4]
    self.model.restore_state(self.snap)
    rows, post = self._spec_rows(chunk, self.pos), self._gdn_states()
    self.model.restore_state(self.snap)
    orig = GatedDeltaNetBlock.__call__
    with mock.patch.object(GatedDeltaNetBlock, "__call__", lambda self, x, sp, rope=None, spec=False: orig(self, x, sp, rope)), \
         Context(GDN_SCAN_IMPL=1):
      np.testing.assert_array_equal(self._spec_rows(chunk, self.pos), rows)
    for i, ((gc, gr), (wc, wr)) in enumerate(zip(self._gdn_states(), post)):
      np.testing.assert_array_equal(gc, wc, err_msg=f"gdn {i} conv")
      np.testing.assert_array_equal(gr, wr, err_msg=f"gdn {i} recurrent")

  def test_speculative_generate_matches_generate_on_varied_ids(self):
    for k in (1, 3):
      self.assertEqual(_run(_load_gdn_varied(), VARIED_PROMPT, 12, spec=True, k=k), self.ref, f"{k=}")

  def test_nv_decode_kernels_track_every_token(self):
    # the card path: GatedDeltaNetBlock(spec=True) through nv.gdn_decode's two kernels (gate mocked open, CPU form) at concrete widths
    # 2..4: slot t == the state the same kernels leave after a fresh run of the first t+1 tokens (bit for bit), and the fused block's
    # output, final states and slots match the generic block's (the per-token loop) at the existing fused-vs-generic tolerances
    from test.unit.test_llm_nv_scan import tiny_block, decode_gate, TINY_DIM
    fused, generic, fresh = tiny_block(), tiny_block(), tiny_block()
    with decode_gate():
      for b in (fused, fresh): b.prepare_nv_decode()
    self.assertIsNotNone(fused.nv_decode_weights)
    xs = Tensor.randn(1, 12, TINY_DIM).realize()
    pos = 0
    for T in (2, 4, 3):
      x = xs[:, pos:pos + T].contiguous().realize()
      pre = []
      for b in (fused, generic, fresh): b._init_state(x)
      for b in (fused, generic): pre.append((b.conv_state.numpy(), b.recurrent_state.numpy()))
      with decode_gate(): out_f = cast(Tensor, fused(x, pos, None, True)).numpy()
      out_g = cast(Tensor, generic(x, pos, None, True)).numpy()
      with self.subTest(T=T, pos=pos):
        np.testing.assert_allclose(out_f, out_g, rtol=1e-3, atol=1e-5)
        np.testing.assert_array_equal(fused.conv_state.numpy(), generic.conv_state.numpy())
        np.testing.assert_allclose(fused.recurrent_state.numpy(), generic.recurrent_state.numpy(), rtol=1e-5, atol=1e-7)
        ft, gt = cast(tuple, fused.spec_track), cast(tuple, generic.spec_track)
        for t in range(T - 1):
          np.testing.assert_array_equal(ft[0][t].numpy(), gt[0][t].numpy(), err_msg=f"slot {t} conv fused vs generic")
          np.testing.assert_allclose(ft[1][t].numpy(), gt[1][t].numpy(), rtol=1e-5, atol=1e-7, err_msg=f"slot {t} recurrent")
          for b, (c0, r0) in ((fresh, pre[0]),):
            b.conv_state.assign(Tensor(c0)).realize()
            b.recurrent_state.assign(Tensor(r0)).realize()
            with decode_gate(): cast(Tensor, b(x[:, :t + 1].contiguous(), pos)).realize()
            np.testing.assert_array_equal(ft[0][t].numpy(), b.conv_state.numpy(), err_msg=f"slot {t} conv vs a {t+1}-token run")
            np.testing.assert_array_equal(ft[1][t].numpy(), b.recurrent_state.numpy(), err_msg=f"slot {t} recurrent vs a {t+1}-token run")
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
  """T4.132: once warm, one speculative iteration schedules NOTHING eagerly -- the draft chain replay, VERIFY, the accept replay (the
  test, the readback, the next anchor and h_last gathered on device, the MTP cache rewrite) and on a partial accept the adopt replay.
  Step 6 had 3-4 eager launches per iteration (h_last's copy, the anchor upload, the chunk cat, REDO's / the rewrite's input prep:
  ~20 ms each on the card's host) and k + 3..4 replays; before step 6, ~93 eagerly dispatched draft launches per iteration."""

  def _census(self, temperature:float, draft_vocab:list[int]|None=None) -> list[list[tuple[str, int]]]:
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
      if draft_vocab is not None: model.mtp_head.set_draft_vocab(model, draft_vocab)  # T4.142
      gen = model.speculative_generate(list(VARIED_PROMPT), k=3, temperature=temperature)
      for _ in range(12): next(gen)  # prefill + the first iterations: traces and captures
      per_yield = []
      for _ in range(16):
        n0 = len(log)
        next(gen)
        per_yield.append(log[n0:])
      gen.close()
    finally: tensor_mod.run_linear, jit_mod.run_linear = orig_t, orig_j
    return [p for p in per_yield if p]  # a yield inside an accepted batch launches nothing

  def test_warm_iteration_is_replays_only(self):
    # T4.142: and with a draft vocabulary -- the chain stays one captured graph
    for temperature, draft_vocab in ((0.0, None), (1.0, None), (0.0, list(range(0, 512, 2))), (1.0, list(range(0, 512, 2)))):
      iters, tag = self._census(temperature, draft_vocab), f"{temperature=} vocab={draft_vocab is not None}"
      self.assertGreater(len(iters), 3)
      for launches in iters:
        self.assertEqual([k for t, k in launches if t == "eager"], [], f"{tag}: an eager schedule in a warm iteration: {launches}")
        # the chain, VERIFY, the accept (+ the adopt on a partial accept)
        self.assertIn(len(launches), (3, 4), f"{tag}: {launches}")
      self.assertTrue(any(len(launches) == 4 for launches in iters), f"{tag}: no partial accept exercised the adopt: {iters}")

class TestSpecTrackSlots(unittest.TestCase):
  """T4.132: spec_track holds spec_k_clamped(k) slots (k=2: 2 per GDN block, ~300 MB at 27B geometry instead of ~460); a later request
  at a larger k grows it and drops the graphs bound to the old buffers (VERIFY families, adopt), which recapture -- still token-identical."""
  def test_slots_follow_k_and_grow_safely(self):
    model = _load_gdn_varied()
    model.warmup(spec_k=2)
    gdn = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
    self.assertEqual({cast(tuple, b.spec_track)[1].shape[0] for b in gdn}, {2})
    ref = _run(_load_gdn_varied(), VARIED_PROMPT, 12, spec=False)
    self.assertEqual(_run(model, VARIED_PROMPT, 12, spec=True, k=2), ref)
    self.assertEqual(_run(model, VARIED_PROMPT, 12, spec=True, k=3), ref)
    self.assertEqual({cast(tuple, b.spec_track)[1].shape[0] for b in gdn}, {3})

class TestSpecCaptureFirst(unittest.TestCase):
  """T4.132 (262k OOM, 10-30): every family the loop replays captures on its FIRST call -- an eager first call schedules every
  intermediate unplanned at once (VERIFY: all attention blocks' partials). The chain and the accept test capture unpruned (their sampled
  draws need Tensor.rand's counter advance in every replay), so their one-time work must already be done (MTPHead.ready): a 1-token
  prompt never runs MTPHead.prefill, and the chain's capture would otherwise hold the MTP block cache's zero-fill and replay it."""
  def test_families_capture_first_and_the_mtp_cache_survives(self):
    for temperature in (0.0, 1.0):
      model = _load_gdn_varied()
      gen = model.speculative_generate([7], k=3, temperature=temperature)
      for _ in range(12): next(gen)
      gen.close()
      for key, jit in [*model.spec_jit.items(), *((k, j) for k, j in model.jit.items() if k[3])]:
        self.assertIsNotNone(cast(TinyJit, jit).captured, f"{temperature=}: {key} not captured")
      calls = {key: cast(TinyJit, j).cnt for key, j in model.spec_jit.items()}
      self.assertTrue(all(c >= 2 for c in calls.values()), f"captured on the first call: cnt starts at 1 ({calls})")
      kv = cast(Tensor, model.mtp_head.block.cache_kv).float().numpy()
      written = len(model._cached_tokens)
      self.assertTrue(all(np.abs(kv[:, :, :, s]).max() > 0 for s in range(1, written)), f"{temperature=}: an MTP cache slot was wiped")

  def test_sampled_replays_draw_fresh_numbers(self):
    model = _load_gdn_varied()
    gen = model.speculative_generate(list(VARIED_PROMPT), k=3, temperature=1.0)
    for _ in range(4): next(gen)
    dev = cast(str, model.output.weight.device)
    before = Tensor._device_rng_counters[dev].tolist()
    for _ in range(6): next(gen)
    self.assertNotEqual(Tensor._device_rng_counters[dev].tolist(), before, "the chain/accept replays did not advance the RNG counter")
    gen.close()

class TestLiveGraphsAfterSpec(unittest.TestCase):
  """T4.132 (10-30 card): plain decode slowed 27.0 -> 34.6 ms/token after warmup(spec_k=3) in the same process. Every eager realize
  walks every live Tensor's graph (tensor.py _apply_map_to_tensors), and the MTP block's output -- returned by the ("accept", n, greedy)
  and ("mtp_prefill", bound) jits -- stayed a live lazy GETTUPLE graph of the whole precompiled block for as long as each jit lived
  (on this model ~420 nodes each, 2124 live nodes in all vs 219; ~57k on the card). Those outputs are now realized inside their graphs
  and dropped, and generate() reads each decoded token straight from the jit's output buffer (no per-token eager copy kernel)."""
  def _live(self) -> dict[int, int]:
    import gc
    from tinygrad.tensor import all_tensors
    gc.collect()
    return {id(t): len(t.uop.toposort()) for r in list(all_tensors) if (t := r()) is not None}

  def _decode_walks(self, model:Transformer) -> tuple[int, int]:
    import tinygrad.tensor as tensor_mod
    calls, nodes = [0], [0]
    orig_apply, orig_topo = tensor_mod._apply_map_to_tensors, UOp.topovisit
    def apply(m, name):
      calls[0] += 1
      return orig_apply(m, name)
    def topo(self, visitor, cache):
      n0 = len(cache)
      r = orig_topo(self, visitor, cache)
      nodes[0] += len(cache) - n0
      return r
    gen = model.generate(list(VARIED_PROMPT), temperature=0.0)
    for _ in range(2): next(gen)
    tensor_mod._apply_map_to_tensors, UOp.topovisit = apply, topo
    try:
      for _ in range(8): next(gen)
    finally: tensor_mod._apply_map_to_tensors, UOp.topovisit = orig_apply, orig_topo
    gen.close()
    return calls[0], nodes[0]

  def test_spec_warmup_leaves_no_lazy_graphs(self):
    model = _load_gdn_varied()
    for temperature in (0.0, 1.0):
      for _ in range(2): list(zip(range(2), model.generate([0], temperature=temperature)))
    self.assertEqual(self._decode_walks(model), (0, 0), "a plain decode step schedules nothing eagerly (no live-graph walk)")
    before = self._live()
    model.warmup(spec_k=3)
    new = {k: n for k, n in self._live().items() if k not in before}
    self.assertLessEqual(max(new.values()), 20, "a Tensor the spec warmup left alive still carries a lazy graph")
    self.assertLess(sum(new.values()), 600, f"{len(new)} new live tensors, {sum(new.values())} nodes")
    self.assertEqual(self._decode_walks(model), (0, 0))

class TestDraftChainHidden(unittest.TestCase):
  """T4.136: under SPEC_CHAIN_H=1 (default) the draft chain feeds step i+1 step i's own block output (MTPHead.draft_step's x) instead of
  reusing h_last (0); the knob is read when the ("chain", k_eff) family captures. Draft quality only: the token stream never moves."""
  K = 2

  def _chain(self, model:Transformer, h:np.ndarray, chain_h:int) -> list[np.ndarray]:
    """each step's logits from the ("chain", K) family's REPLAY (captured under chain_h, replayed under the other value: read at capture
    only; the replay rewrites the same slots from the same inputs)."""
    model.mtp_head.ready(model, self.K)
    dev = model.output.weight.device
    Tensor.rand(1, device=dev).realize()  # the device's RNG state, as speculative_generate sets it up before the first capture
    args = (Tensor(h, device=dev).realize(), Tensor([[3]], dtype="int32", device=dev).realize(), Tensor([0.0], device=dev).realize())
    pos = [UOp.variable(f"draft_pos_{i}", 0, model.max_context - 1).bind(5 + i) for i in range(self.K)]
    for v in (chain_h, 1 - chain_h):
      with Context(SPEC_CHAIN_H=v): out = cast(tuple[Tensor, ...], model._spec_replay(("chain", self.K), model._draft_chain, *args, *pos))
    self.assertEqual(cast(TinyJit, model.spec_jit[("chain", self.K)]).cnt, 3)  # captured on the first call, replayed on the second
    return [q.numpy() for q in out[2:]]

  def test_draft_temperature_scale_scales_the_draw_and_q(self):
    # T4.145: under SPEC_DRAFT_TEMP_SCALE=s the draft draws at temp*s and hands the accept logits/s (softmax((logits/s)/temp) == the draw's);
    # 1.0 leaves both untouched
    from unittest.mock import patch
    import tinygrad.llm.model as mm
    from tinygrad.llm.model import sample_logits, MIN_P
    m = _load_gdn(seed=0)
    m.mtp_head.ready(m, 2)
    dev = m.output.weight.device
    h = Tensor(np.random.default_rng(0).standard_normal((1, 1, DIM)).astype(np.float32), device=dev)
    tok, v = Tensor([[3]], dtype="int32", device=dev), UOp.variable("draft_pos_0", 0, m.max_context - 1).bind(5)
    lg, _ = m.mtp_head.draft_step(m, h, tok, v)
    with patch.object(mm, "SPEC_DRAFT_TEMP_SCALE", 0.5):
      Tensor.manual_seed(1)
      d, q, _ = m._draft_step(h, tok, v, Tensor([1.0], device=dev))
      np.testing.assert_allclose(q.numpy(), lg.numpy() / 0.5, rtol=1e-6, atol=1e-6)
      Tensor.manual_seed(1)
      d2 = sample_logits(lg[:, -1, :], Tensor([0.5], device=dev), MIN_P)  # the same draw at temp * s
      self.assertEqual(int(d.item()), int(d2.item()))
    with patch.object(mm, "SPEC_DRAFT_TEMP_SCALE", 1.0):
      _, q1, _ = m._draft_step(h, tok, v, Tensor([1.0], device=dev))
      np.testing.assert_allclose(q1.numpy(), lg.numpy(), rtol=1e-6, atol=1e-6)

  def test_chain_feeds_each_step_the_heads_own_hidden(self):
    # split models (head on CPU:1, blk[0] on CPU), so the captured graph's copies are visible too
    ref = _load_gdn(seed=0)
    h = np.random.default_rng(0).standard_normal((1, 1, DIM)).astype(np.float32)
    chained_m, reused_m = _split(ref), _split(ref)
    chained, reused = self._chain(chained_m, h, 1), self._chain(reused_m, h, 0)
    # the eager oracle: draft_step twice on a fresh copy, the first step's x fed back as the second step's h
    m = _split(ref)
    m.mtp_head.ready(m, self.K)
    dev, v = m.output.weight.device, [UOp.variable(f"draft_pos_{i}", 0, m.max_context - 1) for i in range(self.K)]
    lg0, x = m.mtp_head.draft_step(m, Tensor(h, device=dev), Tensor([[3]], dtype="int32", device=dev), v[0].bind(5))
    Tensor.realize(lg0, x)
    self.assertEqual(x.shape, (1, 1, DIM))
    self.assertEqual(x.device, m.mtp_head.block.device)
    lg1, _ = m.mtp_head.draft_step(m, x, lg0.argmax(-1), v[1].bind(6))
    np.testing.assert_allclose(chained[0], reused[0], rtol=1e-6, atol=1e-6)  # step 1 takes h_last either way
    np.testing.assert_allclose(chained[0], lg0.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(chained[1], lg1.numpy(), rtol=1e-5, atol=1e-5)
    self.assertGreater(float(np.abs(chained[1] - reused[1]).max()), 1e-3, "step 2 took the same hidden under both values")
    # x stays on the head's device: the captured chain's only copy is VERIFY's chunk to blk[0]'s device, either value
    for model in (chained_m, reused_m):
      copies = []
      for call in cast(TinyJit, model.spec_jit[("chain", self.K)]).captured.linear.src:
        devs = {u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None}
        if call.src[0].op is Ops.COPY: copies.append(call.src[0].arg)
        else: self.assertLessEqual(len(devs), 1, f"compute kernel with mixed-device buffers: {devs}")
      self.assertEqual(copies, [model.blk[0].device])

  def test_both_values_match_generate(self):
    # correctness must not depend on draft quality: greedy speculative_generate == generate() under either hidden
    for load, prompt in ((_load_gdn_varied, VARIED_PROMPT), (_load, [1, 2, 3])):
      ref = _run(load(), prompt, N_GEN, spec=False)
      for chain_h in (0, 1):
        with Context(SPEC_CHAIN_H=chain_h):
          self.assertEqual(_run(load(), prompt, N_GEN, spec=True, k=3), ref, f"{load.__name__} SPEC_CHAIN_H={chain_h}")

# --- T4.142: the draft vocabulary (SPEC_DRAFT_VOCAB, MTPHead.set_draft_vocab) ---

def _quant_linear(ggml_type:int, out_f:int, in_f:int, seed:int=0) -> Linear:
  """a Linear holding a random ggml-quantized weight the way from_gguf loads one: the lazy dequant view of a realized byte buffer, cast
  to fp16 (random bytes, one finite fp16 scale per block)"""
  elems, nbytes = _GGML_QUANT[ggml_type]
  rng = np.random.default_rng(seed)
  blocks = rng.integers(0, 256, size=(out_f * in_f // elems, nbytes), dtype=np.uint8)
  d = rng.uniform(0.01, 0.1, size=len(blocks)).astype(np.float16).view(np.uint8).reshape(-1, 2)
  if ggml_type == 14: blocks[:, -2:] = d  # Q6_K: d is the block's last 2 bytes
  else: blocks[:, :2] = d                 # Q4_0: its first
  lin = Linear.__new__(Linear)
  lin.in_features, lin.out_features, lin.bias = in_f, out_f, None
  lin.weight = ggml_data_to_tensor(Tensor(blocks.reshape(-1)).realize(), out_f * in_f, ggml_type).reshape(out_f, in_f).cast("float16")
  return lin

def _load_with_vocab(ids:list[int], seed:int=0) -> Transformer:
  """_load() with SPEC_DRAFT_VOCAB pointing at an .npy of `ids` -- the server's path (from_gguf reads it once, at load)"""
  with tempfile.TemporaryDirectory() as d:
    np.save(path := os.path.join(d, "vocab.npy"), np.array(ids, dtype=np.int32))
    with Context(SPEC_DRAFT_VOCAB=path): return _load(seed=seed)

def _with_vocab(model:Transformer, ids:list[int]) -> Transformer:
  cast(MTPHead, model.mtp_head).set_draft_vocab(model, ids)
  return model

def _accept_hist(text:str) -> dict[int, int]:
  hist = cast(re.Match, re.search(r"accept_len_hist=\{([^}]*)\}", text)).group(1)
  return {int(a): int(c) for a, c in (pair.split(":") for pair in hist.split(", "))} if hist else {}

def _table_draft(model:Transformer, ref_tokens:list[int]) -> None:
  """a perfect draft that still goes through the draft vocabulary: an instance draft_step (not draft -- so the chain stays the captured
  family and its picks map through draft_ids) whose (1,1,S) logits peak at the slice column of the true next token, read on device from
  a table by the bound draft position (every replay reads its own), and are flat when that token is off the set (column 0 wins greedy;
  sampled, a uniform pick over the set) -- a forced miss."""
  mh, dev = cast(MTPHead, model.mtp_head), model.output.weight.device
  S, inv = cast(Tensor, mh.draft_ids).shape[0], cast(Tensor, mh.draft_inv)
  table = Tensor(ref_tokens + [0] * model.max_context, dtype="int32", device=dev).realize()
  cols = Tensor(list(range(S)), dtype="int32", device=dev).realize()
  def draft_step(owner, h, tok_ids, start_pos):
    col = inv[table[start_pos + 1:start_pos + 2]]
    return (cols == col).where(100.0, -100.0).float().reshape(1, 1, S), h
  mh.draft_step = draft_step

class TestDraftVocab(unittest.TestCase):
  """T4.142: the draft steps run a vocabulary-trimmed copy of the output head (draft_output: the head's rows at draft_ids, in its own ggml
  format), pick among its S columns, map the pick through draft_ids and hand the sampled accept q over the full vocab, -inf off the set.
  A token outside the set is never drafted -- a wasted draft, never a wrong token: the stream must stay generate()'s."""

  def test_linear_rows_gathers_raw_blocks_in_the_heads_format(self):
    ids = [0, 3, 4, 5, 17, 40, 63]
    for ggml_type in (14, 2):  # Q6_K (the 27B's output head), Q4_0
      elems, nbytes = _GGML_QUANT[ggml_type]
      lin = _quant_linear(ggml_type, 64, 512, seed=ggml_type)
      sub = linear_rows(lin, Tensor(ids, dtype="int32"))
      self.assertEqual((sub.in_features, sub.out_features, sub.bias), (512, len(ids), None))
      np.testing.assert_array_equal(sub.weight.numpy(), lin.weight.numpy()[ids])  # dequantized: exactly the head's rows
      x = Tensor.randn(1, 1, 512)
      np.testing.assert_allclose(sub(x).numpy(), lin(x).numpy()[..., ids], rtol=1e-5, atol=1e-5)
      # what set_quantized sees on the card: the same format over a buffer of exactly S rows of raw blocks -- no dequantized copy
      probe = Linear.__new__(Linear)
      probe.set_quantized(sub.weight)
      self.assertEqual(probe.ggml_type, ggml_type)
      self.assertEqual(probe.weight.shape, (len(ids) * 512 // elems * nbytes,))
      self.assertEqual(probe.weight.uop.buf_uop.buffer.base.nbytes, len(ids) * 512 // elems * nbytes)
      # a head that already ran set_quantized (on the card prepare_dense_weights does) gathers from its pre-quantization graph's bytes
      lin.set_quantized(lin.weight)
      self.assertEqual(lin.ggml_type, ggml_type)
      np.testing.assert_array_equal(linear_rows(lin, Tensor(ids, dtype="int32")).weight.numpy(), sub.weight.numpy())

  def test_from_gguf_builds_the_slice_and_unset_changes_nothing(self):
    ids = [0, 2, 4, 6, 8, 10]
    model, plain = _load_with_vocab([10, 4, 0, 2, 8, 6, 4]), _load()  # unsorted, a duplicate: normalized to the sorted set
    mh = cast(MTPHead, model.mtp_head)
    self.assertEqual(cast(Tensor, mh.draft_ids).tolist(), ids)
    self.assertEqual(cast(Tensor, mh.draft_inv).tolist(), [ids.index(i) if i in ids else len(ids) for i in range(VOCAB)])
    np.testing.assert_array_equal(cast(Linear, mh.draft_output).weight.numpy(), model.output.weight.numpy()[ids])
    pm = cast(MTPHead, plain.mtp_head)
    self.assertEqual((pm.draft_output, pm.draft_ids, pm.draft_inv), (None, None, None))
    self.assertNotIn("draft_output", vars(pm))
    # draft(): still (1,1,VOCAB) -- the full head's logits on the set, exactly -inf off it. Loose: this f32 GGUF's dense slice is realized
    # fp16, while the full head's lazy f32 -> fp16 cast fuses into its matmul without the rounding (a quantized head's slice is the same
    # lazy dequant view as the head -- test_linear_rows_gathers_raw_blocks_in_the_heads_format)
    h, tok = Tensor.randn(1, 1, DIM).realize(), Tensor([[3]], dtype="int32")
    got, want = mh.draft(model, h, tok, 0).numpy()[0, 0], pm.draft(plain, h, tok, 0).numpy()[0, 0]
    np.testing.assert_allclose(got[ids], want[ids], rtol=5e-3, atol=5e-3)
    self.assertTrue(np.isneginf(np.delete(got, ids)).all())
    with self.assertRaises(AssertionError): _with_vocab(_load(), [0, VOCAB])  # an id outside the vocab

  def test_half_vocab_matches_generate(self):
    # correctness never depends on draft quality: greedy, and sampled at a temperature low enough that p is one-hot (each emitted token
    # is then generate()'s greedy one whatever the drafts were), match generate() token for token with half the vocabulary drafted
    cases = ((lambda: _with_vocab(_load_gdn_varied(), list(range(0, 512, 2))), _load_gdn_varied, VARIED_PROMPT),
             (lambda: _load_with_vocab(list(range(0, VOCAB, 2))), _load, [1, 2, 3]),
             (lambda: _with_vocab(_load_gdn(), list(range(1, VOCAB, 2))), _load_gdn, [7]))
    for load_vocab, load, prompt in cases:
      ref = _run(load(), prompt, N_GEN, spec=False)
      for temperature in (0.0, 1e-4):
        Tensor.manual_seed(1)
        model = load_vocab()
        got = [v for _, v in zip(range(N_GEN), model.speculative_generate(list(prompt), k=3, temperature=temperature))]
        self.assertEqual(got, ref, f"{load.__name__} {temperature=}")

  def test_sampled_state_integrity_with_half_vocab(self):
    # temperature 1: not reference-comparable token for token, but every id is in the vocab and the state left behind is exactly the
    # emitted tokens' (a greedy continuation matches a fresh model's) -- test_sampled_state_integrity_continues_like_a_fresh_generate's check
    Tensor.manual_seed(7)
    for k in (1, 3):
      model, toks = _with_vocab(_load_gdn_varied(), list(range(0, 512, 2))), list(VARIED_PROMPT)
      got = [v for _, v in zip(range(12), model.speculative_generate(toks, k=k, temperature=1.0))]
      self.assertTrue(all(0 <= t < 512 for t in got), f"{k=} {got}")
      prefix = list(toks)
      self.assertEqual([v for _, v in zip(range(6), model.generate(toks, temperature=0.0))],
                       [v for _, v in zip(range(6), _load_gdn_varied().generate(prefix, temperature=0.0))], f"{k=}")

  def test_forced_misses_off_the_set_still_match_generate(self):
    # a perfect draft through the slice (_table_draft) with the true token of the second iteration's first draft taken OUT of the set:
    # the first iteration fully accepts (the slice's picks map back to the true ids), that one misses at once (m=0 -- greedy: column 0,
    # sampled: the residual p - q = p emits the off-set token), and the stream is still generate()'s -- greedy and sampled (p one-hot)
    k = 3
    ref = _run(_load_gdn_varied(), VARIED_PROMPT, N_GEN + k + 4, spec=False)
    excluded = ref[k + 2]  # anchor ref[0]; iteration 1 drafts ref[1..k] and emits ref[1..k+1]; iteration 2's first draft is ref[k+2]
    self.assertNotIn(excluded, ref[1:k + 1], "pick a model whose first chain is all in the set")
    for temperature in (0.0, 1e-4):
      model = _with_vocab(_load_gdn_varied(), [i for i in range(512) if i != excluded])
      _table_draft(model, list(VARIED_PROMPT) + ref)
      Tensor.manual_seed(2)
      buf = io.StringIO()
      with Context(SPEC_STATS=1), contextlib.redirect_stdout(buf):
        gen = model.speculative_generate(list(VARIED_PROMPT), k=k, temperature=temperature)
        got = [v for _, v in zip(range(N_GEN), gen)]
        gen.close()
      self.assertEqual(got, ref[:N_GEN], f"{temperature=}")
      hist = _accept_hist(buf.getvalue())
      self.assertIn(k, hist, f"{temperature=}: no full accept -- the slice's picks did not map back to the true ids: {hist}")
      self.assertIn(0, hist, f"{temperature=}: the off-set token was not a forced miss: {hist}")
      self.assertIsNotNone(cast(TinyJit, model.spec_jit[("chain", k)]).captured, "the chain ran as its captured family")

  def test_restricted_q_is_zero_off_the_set_and_keeps_the_marginal(self):
    # the sampled accept with q over a draft vocabulary: full_vocab's -inf columns are exactly 0 after the softmax (no NaN), and the token
    # at a drafted position is still marginally p -- including the mass p puts on ids the draft can never propose (only the residual emits
    # them). Drafts are drawn as _draft_step draws them: the Gumbel-max over the S columns (min_p relative to the subset's max), mapped
    # through the ids. min_p cuts p and q alike (T4.132), here with q's lowest column below the cut.
    V, ids, B = 6, [1, 2, 4], 20000
    mh = MTPHead.__new__(MTPHead)
    mh.draft_ids, mh.draft_inv = Tensor(ids, dtype="int32"), Tensor([ids.index(i) if i in ids else len(ids) for i in range(V)], dtype="int32")
    q_sub, p = [0.1, 0.6, 0.3], [0.15, 0.30, 0.05, 0.20, 0.10, 0.20]
    def cut(d, min_p): return (t := np.where(np.array(d) >= min_p * max(d), d, 0.0)) / t.sum()
    for min_p in (0.0, 0.3):
      Tensor.manual_seed(11)
      q_logits = Tensor(np.log(np.array(q_sub, dtype=np.float32))).reshape(1, 1, 3).expand(B, 1, 3)
      q_full = mh.full_vocab(q_logits)  # (B,1,V)
      soft = q_full[0, 0].softmax(-1).numpy()
      self.assertFalse(np.isnan(soft).any())
      np.testing.assert_allclose(soft[ids], q_sub, rtol=1e-5)
      self.assertEqual(soft[[0, 3, 5]].tolist(), [0.0, 0.0, 0.0])
      drafts = cast(Tensor, mh.draft_ids)[sample_logits(q_logits[:, 0], Tensor([1.0]), min_p)]  # (B,1) ids from the set
      p_logits = Tensor(np.log(np.array([p, [1 / V] * V], dtype=np.float32))).unsqueeze(0).expand(B, 2, V)  # k=1 + an unused bonus row
      m, final = spec_accept_tensor(drafts, q_full, p_logits, 1.0, Tensor.rand(B, 1), Tensor.rand(B, 2, V), min_p=min_p)
      emitted = (m.reshape(B, 1) == 1).where(drafts, final[:, :1]).numpy().ravel()
      self.assertTrue(set(np.unique(drafts.numpy()).tolist()) <= set(ids))
      self.assertTrue(((emitted >= 0) & (emitted < V)).all() and ((final.numpy() >= 0) & (final.numpy() < V)).all())
      np.testing.assert_allclose(np.bincount(emitted, minlength=V) / B, cut(p, min_p), atol=0.02, err_msg=f"{min_p=}")

  def test_split_chain_stays_one_graph_on_the_heads_device(self):
    # a CPU:0/CPU:1 split: the slice, the id map and full_vocab's gather live on the head's device -- the captured chain's only copy is
    # VERIFY's chunk to blk[0]'s device, no compute kernel mixes devices, and the stream is generate()'s
    ref = _load_gdn(seed=0)
    want = _run(_split(ref), [1, 2, 3], N_GEN, spec=False)
    m = _with_vocab(_split(ref), list(range(0, VOCAB, 2)))
    mh = cast(MTPHead, m.mtp_head)
    self.assertEqual({cast(Tensor, mh.draft_ids).device, cast(Tensor, mh.draft_inv).device, cast(Linear, mh.draft_output).weight.device},
                     {m.output.weight.device})
    self.assertNotEqual(m.blk[0].device, m.output.weight.device)
    self.assertEqual(_run(m, [1, 2, 3], N_GEN, spec=True, k=3), want)
    copies = []
    for call in cast(TinyJit, m.spec_jit[("chain", 3)]).captured.linear.src:
      devs = {u.device for u in call.toposort() if u.op is Ops.BUFFER and u.device is not None}
      if call.src[0].op is Ops.COPY: copies.append(call.src[0].arg)
      else: self.assertLessEqual(len(devs), 1, f"compute kernel with mixed-device buffers: {devs}")
    self.assertEqual(copies, [m.blk[0].device])

if __name__ == '__main__':
  unittest.main()
