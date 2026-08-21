import unittest
from dataclasses import replace
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig

# mirror gpt-oss's alternating sliding/full pattern + GQA + attn_sinks (T1.3), on a tiny dense config
BASE_CFG = TransformerConfig(num_blocks=4, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                             vocab_size=200, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8,
                             max_context=128, attn_sinks=True)
SLIDING_CFG = replace(BASE_CFG, sliding_window=4, sliding_layers=(True, False, True, False))
NOWINDOW_CFG = replace(BASE_CFG, sliding_window=0, sliding_layers=())

def _run(cfg:TransformerConfig, prompt:list[int], n:int, chunk_size:int, seed:int=42) -> list[int]:
  Tensor.manual_seed(seed)
  model = Transformer(cfg)
  Tensor.realize(*nn.state.get_parameters(model))
  gen = model.generate(list(prompt), chunk_size=chunk_size, temperature=0.0)
  return [t for _, t in zip(range(n), gen)]

class TestSlidingWindowChunkedPrefill(unittest.TestCase):
  """T4.10: T4.3 found gpt-oss-20b's real generation diverges from llama.cpp only on a prompt long
  enough to cross the chunk_size=32 prefill boundary (single-chunk prompts matched llama.cpp exactly).
  The suspicion was a chunking x sliding-window mask bug: the mask in TransformerBlock._attention is
  built from `start_pos` (absolute) and a local chunk row index, so an off-by-chunk error would show
  up as a token-stream difference between single-chunk and multi-chunk prefill of the SAME prompt --
  no external (llama.cpp) reference needed for that comparison, since single-chunk-prefill was already
  proven to match llama.cpp bit-for-bit in T4.3.

  Result: self-consistent at every chunk size / prompt length / seed tried (including the exact
  33-token / chunk_size=32 shape of T4.3's diverging prompt, generated 20 tokens past its token-26
  divergence point) -- tinygrad's own chunked and single-chunk prefill produce an IDENTICAL greedy
  token stream. This rules out the mask-construction hypothesis; see T4.10's report for the next-
  suspect writeup (get_start_pos/_reusable_prefix_len, also cleared) and the standing hypothesis for
  the real vs-llama.cpp divergence (ordinary cross-implementation fp drift compounding over decode
  steps until a near-tied argmax flips -- not caused by chunking)."""

  def test_sliding_window_chunked_matches_single_chunk(self):
    for seed in (42, 7):
      for plen in (11, 33, 34):  # 33/34 straddle the real chunk_size=32 boundary, like T4.3's repro
        prompt = list(range(1, plen + 1))
        ref = _run(SLIDING_CFG, prompt, 20, chunk_size=max(64, plen + 1), seed=seed)
        for cs in (2, 4, 8, 16, 32):
          if cs >= plen: continue
          out = _run(SLIDING_CFG, prompt, 20, chunk_size=cs, seed=seed)
          self.assertEqual(out, ref, f"{seed=} {plen=} {cs=}")

  def test_no_window_chunking_matches_single_chunk(self):
    """Control: sliding_window=0 must also be chunk-invariant. Proves any future failure in the test
    above is window-specific, not a general chunked-prefill regression."""
    for plen in (11, 33):
      prompt = list(range(1, plen + 1))
      ref = _run(NOWINDOW_CFG, prompt, 20, chunk_size=max(64, plen + 1))
      for cs in (2, 8, 16):
        if cs >= plen: continue
        out = _run(NOWINDOW_CFG, prompt, 20, chunk_size=cs)
        self.assertEqual(out, ref, f"{plen=} {cs=}")

if __name__ == '__main__':
  unittest.main()
