import unittest
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig

ATTN_CFG = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                             vocab_size=100, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32)
SSM_CFG = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                            head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=32,
                            ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32),
                            ssm_layers=(True,))

def _run(cfg:TransformerConfig, prompt:list[int], n:int, drain_every:int=1, temperature:float=0.0) -> list[int]:
  Tensor.manual_seed(42)
  model = Transformer(cfg)
  Tensor.realize(*nn.state.get_parameters(model))
  gen = model.generate(list(prompt), temperature=temperature, drain_every=drain_every)
  return [t for _, t in zip(range(n), gen)]

class TestSyncAmortize(unittest.TestCase):
  """T2.5: generate() batches `drain_every` decode steps between host round-trips instead of syncing every
  sampled token. Correctness requirement: identical token stream regardless of drain_every (only host-visible
  timing changes)."""

  def test_drain_every_1_matches_no_arg_default(self):
    # drain_every's default (1) must be indistinguishable from omitting it -- the pre-T2.5 call signature
    self.assertEqual(_run(ATTN_CFG, [1, 2, 3], 6), _run(ATTN_CFG, [1, 2, 3], 6, drain_every=1))

  def test_greedy_token_stream_identical_n1_vs_n4(self):
    self.assertEqual(_run(ATTN_CFG, [1, 2, 3], 10, drain_every=1), _run(ATTN_CFG, [1, 2, 3], 10, drain_every=4))

  def test_recurrent_token_stream_identical_n1_vs_n4(self):
    # has_recurrent_block forces chunk_size=1 (unrelated to drain_every) -- both knobs must compose correctly
    self.assertEqual(_run(SSM_CFG, [1, 2, 3], 10, drain_every=1), _run(SSM_CFG, [1, 2, 3], 10, drain_every=4))

  def test_chunked_prefill_composes_with_drain(self):
    # prompt longer than chunk_size forces multiple prefill chunks before the first sampled token
    long_prompt = list(range(1, 12))
    n1 = _run(ATTN_CFG, long_prompt, 8, drain_every=1)
    n4 = _run(ATTN_CFG, long_prompt, 8, drain_every=4)
    self.assertEqual(n1, n4)

  def test_drain_flushes_before_context_limit(self):
    # max_context - prompt_len isn't a multiple of drain_every: the last partial batch must still flush
    from dataclasses import replace
    tight_cfg = replace(ATTN_CFG, max_context=10)
    prompt = [1, 2, 3]
    n1 = _run(tight_cfg, prompt, 20, drain_every=1)  # 20 > available room; generator stops at max_context
    n4 = _run(tight_cfg, prompt, 20, drain_every=4)
    self.assertEqual(n1, n4)
    self.assertEqual(len(n1), tight_cfg.max_context - len(prompt))  # nothing stranded in `pending`

  def test_eos_mid_drain_window_truncates_identically(self):
    # simulate the caller-side is_end() loop (cli.py/serve.py): break the moment a "stop token" is yielded.
    # With drain_every=4, that stop token can land mid-batch -- the already-computed tokens after it in that
    # batch must never be appended/yielded (up to drain_every-1 wasted device steps, but no leaked host state).
    ref = _run(ATTN_CFG, [1, 2, 3], 8, drain_every=1)
    eos_tok = ref[2]  # a token that will be sampled partway through generation

    def _consume(drain_every:int) -> tuple[list[int], list[int]]:
      Tensor.manual_seed(42)
      model = Transformer(ATTN_CFG)
      Tensor.realize(*nn.state.get_parameters(model))
      out = []
      for tok in model.generate([1, 2, 3], drain_every=drain_every):
        out.append(tok)
        if tok == eos_tok: break
      return out, list(model._cached_tokens)

    out1, cached1 = _consume(1)
    out4, cached4 = _consume(4)
    self.assertEqual(out1, out4)
    self.assertEqual(cached1, cached4)
    self.assertEqual(out1[-1], eos_tok)  # sanity: the loop actually stopped on the stop token

  def test_streaming_yields_one_token_at_a_time_in_order(self):
    # even batched internally, next() must still hand back exactly one token per call, in generation order
    Tensor.manual_seed(42)
    model = Transformer(ATTN_CFG)
    Tensor.realize(*nn.state.get_parameters(model))
    gen = model.generate([1, 2, 3], drain_every=4)
    seen = [next(gen) for _ in range(9)]
    self.assertEqual(len(seen), 9)
    self.assertTrue(all(isinstance(t, int) for t in seen))
    self.assertEqual(seen, _run(ATTN_CFG, [1, 2, 3], 9, drain_every=1))

if __name__ == '__main__':
  unittest.main()
