import unittest
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer
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

PROMPTS = ([1, 2, 3], [4, 5], [1, 2, 3, 4, 5, 6], [7])
N_GEN = 32

class TestSpeculativeGenerate(unittest.TestCase):
  def test_matches_generate_several_prompts_and_k(self):
    # the core gate: identical output to plain greedy generate(), for every k in {1,2,3} and several prompts
    for k in (1, 2, 3):
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
    for k in (1, 2, 3):
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
    for k in (1, 2, 3):
      for seed, prompt in enumerate(PROMPTS):
        lookahead = N_GEN + k + 4  # margin: the last iteration's drafts may peek a few positions past N_GEN
        ref = _run(_load(seed=seed), prompt, lookahead, spec=False)
        ref_tokens = list(prompt) + ref  # ref_tokens[i] is the true token AT position i

        calls = 0
        def fake_draft(owner, h, tok_ids, start_pos, _ref=ref_tokens):
          nonlocal calls
          calls += 1
          return _wrong_logits(_ref[start_pos + 1])  # "wrong" only in name -- here it's the true next token

        model = _load(seed=seed)
        model.mtp_head.draft = fake_draft
        calls = 0
        got = _run(model, prompt, N_GEN, spec=True, k=k)
        self.assertEqual(got, ref[:N_GEN], f"{k=} {seed=} {prompt=}")
        expected_iters = -(-N_GEN // (k + 1))  # ceil(N_GEN / (k+1))
        self.assertEqual(calls, k * expected_iters, f"{k=} {seed=} {prompt=}")

  def test_state_integrity_continues_like_a_fresh_generate(self):
    # after speculative_generate stops (mid- or between-iteration), a plain generate() on the SAME model
    # continuing the SAME growing token list (serve.py's splice/_cached_tokens path) must produce exactly
    # what a from-scratch generate() over the whole thing would have -- proving speculative_generate leaves
    # no stale/incorrect state (KV cache, GDN state, _cached_tokens) behind, whichever accept path it took.
    for k in (1, 2, 3):
      for seed, prompt in enumerate(PROMPTS):
        model_a = _load(seed=seed)
        tokens_so_far = list(prompt)
        got_first = [v for _, v in zip(range(20), model_a.speculative_generate(tokens_so_far, k=k))]
        got_second = [v for _, v in zip(range(10), model_a.generate(tokens_so_far, temperature=0.0))]

        full_ref = _run(_load(seed=seed), prompt, 30, spec=False)
        self.assertEqual(got_first + got_second, full_ref, f"{k=} {seed=} {prompt=}")

if __name__ == '__main__':
  unittest.main()
