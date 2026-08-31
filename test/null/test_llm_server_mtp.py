import unittest, threading, time
from unittest.mock import Mock

# T4.65: --mtp routing (Handler.run_model's use_spec = self.server.mtp and model.mtp_head is not None).
# Mirrors test_llm_server.py's mock-model/mock-tokenizer harness -- see there for the established pattern.

class TestLLMServerMTP(unittest.TestCase):
  """--mtp on, model.mtp_head set: a mocked speculative round-trip, both greedy (temperature absent/0) and
  sampled (temperature>0), through the real HTTP path (do_POST -> run_model -> speculative_generate)."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="Hello")
    cls.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    cls.mock_tok.preset = "llama3"
    cls.mock_tok.bos_id = 1
    cls.mock_tok.eos_id = 999
    cls.mock_tok.eot_id = None
    cls.mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    cls.mock_model = Mock()
    cls.mock_model.max_context = 4
    cls.mock_model.get_start_pos = Mock(return_value=0)
    cls.mock_model.mtp_head = Mock()  # non-None: "the loaded model has an mtp_head"
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    cls.mock_model.speculative_generate = Mock(side_effect=lambda ids, **kwargs: iter([400, 401, 999]))

    from tinygrad.llm.cli import FallbackTemplate
    from tinygrad.llm.serve import LLMServer

    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "test-model", cls.mock_tok, FallbackTemplate(cls.mock_tok),
                           mtp=True, spec_k=2)
    cls.port = cls.server.server_address[1]
    cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.server_thread.start()
    time.sleep(0.1)

    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.port}/v1", api_key="test")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def setUp(self):
    self.mock_model.generate.reset_mock()
    self.mock_model.speculative_generate.reset_mock()

  def test_greedy_request_routes_through_speculative_generate(self):
    # temperature absent -> body.get("temperature", 0.0) -> 0.0 -> speculative_generate's own greedy path
    resp = self.client.chat.completions.create(model="test-model", messages=[{"role":"user", "content":"Hi"}], stream=False)
    self.mock_model.speculative_generate.assert_called_once()
    self.mock_model.generate.assert_not_called()
    _, kwargs = self.mock_model.speculative_generate.call_args
    self.assertEqual(kwargs["k"], 2)          # server's spec_k
    self.assertEqual(kwargs["temperature"], 0.0)
    self.assertEqual(resp.choices[0].finish_reason, "stop")
    self.assertIsNotNone(resp.choices[0].message.content)

  def test_sampled_request_routes_through_speculative_generate_with_temperature(self):
    resp = self.client.chat.completions.create(model="test-model", messages=[{"role":"user", "content":"Hi"}],
                                               stream=False, temperature=1.0)
    self.mock_model.speculative_generate.assert_called_once()
    self.mock_model.generate.assert_not_called()
    _, kwargs = self.mock_model.speculative_generate.call_args
    self.assertEqual(kwargs["k"], 2)
    self.assertEqual(kwargs["temperature"], 1.0)  # the request's own temperature, passed through unchanged
    self.assertEqual(resp.choices[0].finish_reason, "stop")

class TestLLMServerMTPFallback(unittest.TestCase):
  """When --mtp is absent, or present but the model has no mtp_head, routing must fall back to plain
  generate() -- byte-identical to serving without --mtp at all (T4.65 hard rule). Direct run_model calls
  (test_llm_server.py's test_interrupted_stream_logs_tokens style) instead of a full HTTP server: these are
  gating checks, not round-trips, so the lighter idiom is enough."""

  @staticmethod
  def _make_server(mtp:bool, mtp_head):
    mock_tok = Mock()
    mock_tok.encode = Mock(return_value=[200, 201, 202])
    mock_tok.decode = Mock(return_value="Hello")
    mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    mock_tok.preset = "llama3"
    mock_tok.bos_id, mock_tok.eos_id, mock_tok.eot_id = 1, 999, None
    mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    mock_model = Mock()
    mock_model.max_context = 4
    mock_model.get_start_pos = Mock(return_value=0)
    mock_model.mtp_head = mtp_head
    mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    mock_model.speculative_generate = Mock(side_effect=lambda ids, **kwargs: iter([400, 401, 999]))

    from tinygrad.llm.cli import FallbackTemplate
    from tinygrad.llm.serve import LLMServer
    server = LLMServer(('127.0.0.1', 0), mock_model, "test-model", mock_tok, FallbackTemplate(mock_tok), mtp=mtp, spec_k=2)
    return server, mock_model

  def test_mtp_flag_off_uses_plain_generate(self):
    server, mock_model = self._make_server(mtp=False, mtp_head=Mock())
    from tinygrad.llm.serve import Handler
    list(Handler.run_model(Mock(server=server), [200, 201, 202], "test"))
    mock_model.generate.assert_called_once()
    mock_model.speculative_generate.assert_not_called()

  def test_mtp_flag_on_but_no_mtp_head_uses_plain_generate(self):
    server, mock_model = self._make_server(mtp=True, mtp_head=None)
    from tinygrad.llm.serve import Handler
    list(Handler.run_model(Mock(server=server), [200, 201, 202], "test"))
    mock_model.generate.assert_called_once()
    mock_model.speculative_generate.assert_not_called()

  def test_default_server_has_mtp_off(self):
    # LLMServer's own default (no mtp= kwarg at all) -- exactly what every pre-T4.65 caller still gets.
    from tinygrad.llm.cli import FallbackTemplate
    from tinygrad.llm.serve import LLMServer, Handler
    mock_tok = Mock()
    mock_tok.is_end = Mock(return_value=True)
    mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    mock_model = Mock()
    mock_model.max_context = 4
    mock_model.get_start_pos = Mock(return_value=0)
    mock_model.mtp_head = Mock()
    mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    mock_model.speculative_generate = Mock(side_effect=lambda ids, **kwargs: iter([400, 401, 999]))
    server = LLMServer(('127.0.0.1', 0), mock_model, "test-model", mock_tok, FallbackTemplate(mock_tok))  # no mtp kwarg
    self.assertFalse(server.mtp)
    list(Handler.run_model(Mock(server=server), [200, 201, 202], "test"))
    mock_model.generate.assert_called_once()
    mock_model.speculative_generate.assert_not_called()

if __name__ == '__main__':
  unittest.main()
