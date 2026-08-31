import unittest
from unittest.mock import Mock, patch
from tinygrad import Tensor

# T4.67: cross-session state cache (LLMServer.snapshots/find_snapshot/store_snapshot, wired into
# Handler.run_model). Mirrors test_llm_server.py's mock-model/mock-tokenizer harness -- see there for the
# established pattern -- but the model mock is backed by a small stateful fake (_FakeCache) instead of a
# dumb constant, so get_start_pos/snapshot_state/restore_state behave like the real Transformer's exact-
# prefix rule (model.py's get_start_pos/snapshot_matches) without needing any real compute.

class _FakeCache:
  """Minimal stand-in for Transformer's get_start_pos/snapshot_state/restore_state. Real (tiny) Tensors in
  snapshot_state's return so snapshot_nbytes' real arithmetic (imported by serve.py) drives the LRU test."""
  def __init__(self, block_nbytes:int=1024):
    self.tokens: list[int] = []
    self._n = max(1, block_nbytes // 4)  # float32 elements -> ~block_nbytes when snapshot_nbytes sums it
  def get_start_pos(self, ids:list[int]) -> int:
    return len(self.tokens) if self.tokens and len(self.tokens) < len(ids) and ids[:len(self.tokens)] == self.tokens else 0
  def snapshot_state(self) -> dict:
    return {"tokens": list(self.tokens), "blocks": [{"cache_kv": Tensor.zeros(self._n).contiguous().realize()}]}
  def restore_state(self, snap:dict) -> None:
    self.tokens = list(snap["tokens"])

class TestLLMServerStateCache(unittest.TestCase):
  @staticmethod
  def _server(state_cache_mb:int, block_nbytes:int=1024*1024):
    fake = _FakeCache(block_nbytes)
    mock_tok = Mock()
    mock_tok.encode = Mock(return_value=[1, 2, 3])
    mock_tok.decode = Mock(return_value="Hello")
    mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    mock_tok.preset = "llama3"
    mock_tok.bos_id, mock_tok.eos_id, mock_tok.eot_id = 1, 999, None
    mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    mock_model = Mock()
    mock_model.max_context = 1000
    mock_model.mtp_head = None
    mock_model.get_start_pos = Mock(side_effect=fake.get_start_pos)
    mock_model.snapshot_state = Mock(side_effect=fake.snapshot_state)
    mock_model.restore_state = Mock(side_effect=fake.restore_state)
    def fake_generate(ids, **kwargs):
      fake.tokens = list(ids)  # simulates _cached_tokens == ids once prefill completes (see model.py's generate())
      yield from (300, 301, 999)
    mock_model.generate = Mock(side_effect=fake_generate)

    from tinygrad.llm.cli import FallbackTemplate
    from tinygrad.llm.serve import LLMServer
    server = LLMServer(('127.0.0.1', 0), mock_model, "test-model", mock_tok, FallbackTemplate(mock_tok), state_cache_mb=state_cache_mb)
    return server, mock_model, fake

  def test_longest_matching_snapshot_wins_and_prefills_only_the_tail(self):
    from tinygrad.llm.serve import Handler
    server, mock_model, fake = self._server(state_cache_mb=64)
    short_prefix, long_prefix = [1, 2], [1, 2, 3, 4]
    list(Handler.run_model(Mock(server=server), short_prefix, "test"))   # stores (1,2)
    list(Handler.run_model(Mock(server=server), long_prefix, "test"))    # live-splices onto (1,2), then stores (1,2,3,4)
    list(Handler.run_model(Mock(server=server), [50, 51], "test"))       # unrelated session clobbers the live cache

    ids = long_prefix + [5, 6]  # exactly extends BOTH stored candidates -- the longer one must win
    with patch("tinygrad.llm.serve.stderr_log") as log, patch("tinygrad.llm.serve.colored", side_effect=lambda text, _color: text):
      list(Handler.run_model(Mock(server=server), ids, "test"))
    in_line = next(c.args[0] for c in log.call_args_list if c.args[0].startswith("in:"))
    self.assertIn(f"in:{4:5d} +{2:5d}", in_line)  # restored to the 4-token boundary -> only [5, 6] (2 tokens) prefill
    self.assertEqual(mock_model.restore_state.call_args[0][0]["tokens"], long_prefix)

  def test_lru_evicts_oldest_snapshot_once_over_the_cap(self):
    from tinygrad.llm.serve import Handler
    # each stored snapshot is ~1 MB (block_nbytes default); a 2 MB cap fits exactly 2 of them
    server, mock_model, fake = self._server(state_cache_mb=2, block_nbytes=1024*1024)
    list(Handler.run_model(Mock(server=server), [1], "test"))
    list(Handler.run_model(Mock(server=server), [2], "test"))
    self.assertEqual(list(server.snapshots.keys()), [(1,), (2,)])  # both fit under the cap
    list(Handler.run_model(Mock(server=server), [3], "test"))
    self.assertEqual(list(server.snapshots.keys()), [(2,), (3,)])  # (1,) was least-recently-used -> evicted

  def test_cap_zero_never_touches_snapshot_state_or_restore(self):
    from tinygrad.llm.serve import Handler
    server, mock_model, fake = self._server(state_cache_mb=0)
    list(Handler.run_model(Mock(server=server), [1, 2, 3, 4], "test"))
    list(Handler.run_model(Mock(server=server), [50, 51], "test"))  # unrelated -- the live splice also misses
    mock_model.snapshot_state.assert_not_called()
    mock_model.restore_state.assert_not_called()
    self.assertEqual(len(server.snapshots), 0)
    self.assertEqual(mock_model.get_start_pos.call_count, 2)  # exactly once per request -- no post-restore recheck

if __name__ == '__main__':
  unittest.main()
