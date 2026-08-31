import unittest
from unittest.mock import Mock
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.schedule import schedule_cache
from tinygrad.llm.model import Transformer, spec_accept
from test.unit.test_mtp_load import _build_tiny_qwen35_gguf, _gguf_tensor, VOCAB

# T4.64: speculative_generate must be TOKEN-IDENTICAL to generate(temperature=0.0) regardless of draft
# quality (correctness never depends on the MTP head guessing right -- only the accept-iteration count
# does). Reuses test_mtp_load.py's tiny synthetic qwen35-shaped GGUF builder: full_attention_interval=1
# there makes every real block a plain TransformerBlock (no GatedDeltaNet), so this model exercises the
# speculative_generate GDN-checkpoint/restore code as an always-empty (no-op) list -- see SPEC_NOTES.md.

def _load(seed:int=0, max_context:int=64) -> Transformer:
  with Context(MTP=1):
    model, _ = Transformer.from_gguf(_gguf_tensor(_build_tiny_qwen35_gguf(seed=seed)), max_context=max_context,
                                     device_map=None, realize=False)
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

if __name__ == '__main__':
  unittest.main()
