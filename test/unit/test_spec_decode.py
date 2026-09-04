import contextlib, io, re, unittest
from dataclasses import replace
from unittest.mock import Mock
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import Context, next_power2
from tinygrad.schedule import schedule_cache
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import (
  Transformer, TransformerConfig, TransformerBlock, GatedDeltaNetBlock, MTPHead, SSMConfig, spec_accept,
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
    rng = np.random.default_rng(12345)
    for k in (1, 2, 3):
      for seed, prompt in enumerate(PROMPTS):
        model_a = _load(seed=seed)
        tokens_so_far = list(prompt)
        got_first = [v for _, v in zip(range(15), model_a.speculative_generate(tokens_so_far, k=k, temperature=1.0, rng=rng))]
        self.assertEqual(len(got_first), 15, f"{k=} {seed=} {prompt=}")
        self.assertTrue(all(0 <= tid < VOCAB for tid in got_first), f"{k=} {seed=} {prompt=}")
        self.assertEqual(len(tokens_so_far), len(prompt) + 15, f"{k=} {seed=} {prompt=}")
        # bookkeeping invariant generate() maintains too (see its own tokens.append/self._cached_tokens lines)
        self.assertEqual(model_a._cached_tokens, tokens_so_far[:-1], f"{k=} {seed=} {prompt=}")

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

class TestSpecAccept(unittest.TestCase):
  """spec_accept is pure host-side numpy (no model/Tensor involved) -- Leviathan et al.'s speculative-
  sampling accept/reject/resample test T4.65 wires into speculative_generate's temperature>0 path."""

  def test_identical_distributions_always_accept(self):
    # q==p exactly -> accept_prob=min(1,p/q)=1 everywhere -> ALWAYS accept, regardless of the random draw
    # (rng.random() in [0,1) is always < 1.0). Stress it with the draw pushed right up against 1.
    probs = np.array([0.1, 0.5, 0.1, 0.2, 0.1])
    q_probs, p_probs = np.stack([probs, probs]), np.stack([probs, probs, probs])  # k_eff=2, +1 bonus row
    rng = Mock()
    rng.random = Mock(return_value=0.999999999)
    rng.choice = Mock(return_value=4)  # only consulted for the bonus token, on full accept
    accepted, m = spec_accept([1, 3], q_probs, p_probs, rng)
    self.assertEqual((accepted, m), ([1, 3, 4], 2))
    rng.choice.assert_called_once()
    args, kwargs = rng.choice.call_args
    self.assertEqual(args[0], 5)
    np.testing.assert_allclose(kwargs["p"], probs)  # bonus sampled straight from p_probs[k_eff] (== probs here)

  def test_excluded_token_always_rejects_with_correct_resample_support(self):
    # q puts mass on token 0, p excludes it entirely (p[0]=0) -> ratio=0 -> ALWAYS reject regardless of the
    # random draw (rng.random() in [0,1) is never < 0) -- and the resample distribution passed to rng.choice
    # must equal EXACTLY normalize(max(0, p-q)), not just "excludes token 0".
    q0 = np.array([0.5, 0.3, 0.1, 0.1, 0.0])
    p0 = np.array([0.0, 0.4, 0.2, 0.2, 0.2])
    q_probs, p_probs = q0[None, :], np.stack([p0, np.full(5, 0.2)])  # k_eff=1; bonus row unused here
    rng = Mock()
    rng.random = Mock(return_value=1e-9)  # smallest-possible draw -- still must reject (ratio is exactly 0)
    rng.choice = Mock(return_value=4)
    accepted, m = spec_accept([0], q_probs, p_probs, rng)
    self.assertEqual((accepted, m), ([4], 0))
    residual = np.clip(p0 - q0, 0, None)
    expected = residual / residual.sum()  # the docstring's exact formula, computed independently here
    self.assertEqual(expected[0], 0.0)  # token 0 (excluded by p) must never get resample mass
    args, kwargs = rng.choice.call_args
    self.assertEqual(args[0], 5)
    np.testing.assert_allclose(kwargs["p"], expected)

  def test_zero_drafts_samples_bonus_directly(self):
    # k_eff=0 (speculative_generate's max_context boundary degrades to this, see SPEC_NOTES.md's position
    # ledger): no drafts to test at all, straight to a bonus sample from p_probs[0] -- rng.random() (the
    # accept test) is never even called.
    p0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    rng = Mock()
    rng.choice = Mock(return_value=3)
    accepted, m = spec_accept([], np.empty((0, 5)), p0[None, :], rng)
    self.assertEqual((accepted, m), ([3], 0))
    rng.random.assert_not_called()

  def test_statistical_marginal_matches_p(self):
    # the theorem spec_accept's whole correctness rests on (Leviathan et al., Theorem 1): if draft_ids are
    # drawn by ancestral sampling from q, the emitted token at that drafted position is marginally
    # distributed as p -- REGARDLESS of q (draft quality only ever affects the accept RATE). Checked
    # empirically over ~20k trials on a 5-token vocab, since this is exactly the guarantee the sampled
    # speculative-decoding path (and its "distribution-equal to generate(temperature=t)" claim) relies on.
    rng = np.random.default_rng(42)
    q = np.array([0.10, 0.50, 0.05, 0.30, 0.05])
    p = np.array([0.40, 0.10, 0.20, 0.05, 0.25])
    q_probs = q[None, :]                                    # k_eff=1
    p_probs = np.stack([p, np.full(5, 0.2)])                # + an (unused-here) valid bonus row
    n_trials = 20000
    counts = np.zeros(5)
    for _ in range(n_trials):
      d = int(rng.choice(5, p=q))                           # ancestral draft sample: d ~ q
      accepted, _m = spec_accept([d], q_probs, p_probs, rng)
      counts[accepted[0]] += 1                               # accepted[0] is the token AT the drafted position
    empirical = counts / n_trials
    np.testing.assert_allclose(empirical, p, atol=0.02)      # loose: ~7 sigma even at p's largest entry (0.40)

class TestVerifyNativeWidth(unittest.TestCase):
  """T4.66e: VERIFY (and REDO) get their own dedicated jit key + Variable (v_toks_verify, "verify_chunk" =
  next_power2(k+1)) instead of sharing the prefill loop's wide (chunk_size, default 32) v_toks -- see
  model.py's comments at v_toks_verify's own definition, and SPEC_VERIFY_NOTES.md for the full T_pad
  derivation this test checks end to end (x.max_shape[1] inside GatedDeltaNetBlock._attention, which is
  exactly the "toks" Variable's declared range once bound -- see uop/ops.py's to_max_shape/UOp.bind)."""

  def test_verify_forward_gdn_scan_uses_verify_chunk_width(self):
    k = 3
    model = _load_gdn(seed=0, max_context=64)
    t_pads: list[int] = []
    orig_attention = GatedDeltaNetBlock._attention
    def spy(self, x, start_pos, capture=False):
      t_pads.append(x.max_shape[1])  # the exact expression _attention itself uses to compute T_pad
      return orig_attention(self, x, start_pos, capture=capture)
    GatedDeltaNetBlock._attention = spy
    try:
      # GDN_CHUNK=32 pins the prefill chunk width to the same 32 CLAUDE.md's real METAL/NV device maps
      # auto-select (gdn_chunk_for's own default there) -- on the bare-CPU test device, gdn_chunk_for()
      # would otherwise auto-narrow the PREFILL chunk to 1 (T4.55's non-GPU fallback), which happens to
      # collapse chunk_size down to k+1 too and hide the very gap (32 vs verify_chunk) this test exists
      # to check.
      with Context(GDN_CHUNK=32):
        gen = model.speculative_generate([1, 2, 3], k=k)
        next(gen)  # prefill only -- its own tail chunk legitimately runs the GDN scan at chunk_size=32
        t_pads.clear()  # only care about what happens from here: the outer loop's own VERIFY/REDO calls
        next(gen)  # one full outer iteration: DRAFT + VERIFY (+ REDO on a partial accept), both spec=True
    finally:
      GatedDeltaNetBlock._attention = orig_attention
    verify_chunk = next_power2(k + 1)
    self.assertGreater(len(t_pads), 0, "VERIFY's forward should have run the GDN block's _attention at least once")
    self.assertIn(verify_chunk, t_pads, f"expected a GDN scan at VERIFY_CHUNK={verify_chunk}, saw {sorted(set(t_pads))}")
    self.assertNotIn(32, t_pads, "VERIFY/REDO must not still pad the GDN scan to the prefill chunk_size (32)")

class TestSpecWarmup(unittest.TestCase):
  """T4.66e deliverable 2: Transformer.warmup() only ever drove generate()'s own jit keys pre-T4.66e --
  speculative_generate's spec=True keys (the prefill tail's, and VERIFY/REDO's own dedicated verify_chunk
  one) were captured at REQUEST time instead (the T4.65 gap; recommended fix in T4.72). See warmup()'s own
  comment for why each key needs calling TWICE (engine/jit.py _TinyJit.__call__: a key's first-ever call
  just runs eagerly, no capture at all -- the capture that actually pays off on every later replay is the
  SAME key's second call)."""

  def test_warmup_captures_spec_keys_when_mtp_present(self):
    model = _load_gdn(seed=0, max_context=64)
    self.assertEqual(model.jit, {}, "nothing should be captured before warmup runs")
    model.warmup()
    spec_keys = [key for key in model.jit if key[3]]  # spec=True is the 4th (is_prefill, greedy, chunk_size, spec) slot
    self.assertGreater(len(spec_keys), 0, "warmup should capture at least one spec=True jit key once mtp_head is set")
    verify_chunk = next_power2(3 + 1)  # speculative_generate's own default k=3, matching warmup's own self.speculative_generate([0]) call
    self.assertTrue(any(key[2] == verify_chunk for key in spec_keys),
                    f"expected a captured spec key at chunk_size=verify_chunk={verify_chunk}, got {spec_keys}")
    for key in spec_keys:
      self.assertIsNotNone(model.jit[key].captured, f"jit[{key}] was called but never actually captured (still cnt<=1)")

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
    for attr in ("cache_kv", "cache_k"):
      self.assertFalse(hasattr(model.mtp_head.block, attr),
                        f"warmup should leave mtp_head.block exactly as unused (no {attr}) as before it "
                        "ever ran speculative decoding -- draft() rebuilds it fresh (all-zero) on first real use")

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

if __name__ == '__main__':
  unittest.main()
