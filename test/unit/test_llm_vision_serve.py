# T5.4 (VISION_DESIGN.md section 2.1/2.4): server/CLI vision wiring -- pure-function unit tests plus a couple of
# HTTP-level 400/capability checks in the T4.80 TestLMStudioShim style (test/unit/test_llm_server.py). No real
# model/encoder: stubs only.
import base64, unittest
from types import SimpleNamespace
from tinygrad.llm.image import hash_ids, image_hash
from tinygrad.llm.serve import (ImageError, IMAGE_PLACEHOLDER, LLMServer, expand_image_pads, extract_images,
                                lmstudio_models_payload, load_image)

VOCAB = 151936  # Qwen3.6's real vocab size -- big enough that hash_ids' modulo behaves like it does in production

class TestExpandImagePads(unittest.TestCase):
  def test_single_image_all_hash_no_pad(self):
    digest = image_hash(b"img-a")
    out, spans = expand_image_pads([1, 2, 999, 3], 999, [(digest, 6, (1, 4, 4))], VOCAB)
    run = hash_ids(digest, VOCAB)[:6]  # min(8, 6) == 6: every id in the run is hash-derived, no pad left
    self.assertEqual(out, [1, 2, *run, 3])
    self.assertEqual(spans, [(2, 6, (1, 4, 4))])

  def test_single_image_eight_hash_then_pad(self):
    digest = image_hash(b"img-b")
    out, spans = expand_image_pads([999], 999, [(digest, 12, (1, 8, 8))], VOCAB)
    self.assertEqual(out, hash_ids(digest, VOCAB)[:8] + [999] * 4)  # min(8, 12) == 8: 8 hash ids then 4 pad ids
    self.assertEqual(spans, [(0, 12, (1, 8, 8))])

  def test_two_images_in_order_offsets_account_for_first_expansion(self):
    d1, d2 = image_hash(b"one"), image_hash(b"two")
    out, spans = expand_image_pads([1, 999, 2, 999, 3], 999, [(d1, 6, (1, 4, 4)), (d2, 12, (1, 8, 8))], VOCAB)
    run1, run2 = hash_ids(d1, VOCAB)[:6], hash_ids(d2, VOCAB)[:8] + [999] * 4
    self.assertEqual(out, [1, *run1, 2, *run2, 3])
    self.assertEqual(spans, [(1, 6, (1, 4, 4)), (1 + len(run1) + 1, 12, (1, 8, 8))])

  def test_zero_images_ids_unchanged(self):
    ids = [1, 2, 3]
    out, spans = expand_image_pads(ids, 999, [], VOCAB)
    self.assertEqual(out, ids)
    self.assertEqual(spans, [])

  def test_more_pad_occurrences_than_images_raises(self):
    with self.assertRaises(ImageError):
      expand_image_pads([999, 999], 999, [(image_hash(b"x"), 4, (1, 4, 4))], VOCAB)

  def test_fewer_pad_occurrences_than_images_raises(self):
    with self.assertRaises(ImageError):
      expand_image_pads([1, 2], 999, [(image_hash(b"x"), 4, (1, 4, 4))], VOCAB)

  def test_hash_ids_placement_uses_original_bytes(self):
    data = b"hello image bytes"
    out, _ = expand_image_pads([999], 999, [(image_hash(data), 6, (1, 4, 4))], VOCAB)
    self.assertEqual(out[:6], hash_ids(image_hash(data), VOCAB)[:6])

class TestExtractImages(unittest.TestCase):
  def test_text_and_data_url_and_text_flattened_in_order(self):
    payload = base64.b64encode(b"fake-png-bytes").decode()
    messages = [{"role": "user", "content": [
      {"type": "text", "text": "before "},
      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
      {"type": "text", "text": " after"},
    ]}]
    images = extract_images(messages)
    self.assertEqual(messages[0]["content"], f"before {IMAGE_PLACEHOLDER} after")
    self.assertEqual(images, [b"fake-png-bytes"])

  def test_bare_string_content_untouched(self):
    messages = [{"role": "user", "content": "just text"}]
    self.assertEqual(extract_images(messages), [])
    self.assertEqual(messages[0]["content"], "just text")

  def test_bare_string_image_url_value(self):
    payload = base64.b64encode(b"abc").decode()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/png;base64,{payload}"}]}]
    self.assertEqual(extract_images(messages), [b"abc"])

  def test_unsupported_url_scheme_raises(self):
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ftp://host/pic.png"}}]}]
    with self.assertRaises(ImageError):
      extract_images(messages)

class TestLoadImage(unittest.TestCase):
  def test_garbage_bytes_is_image_error_not_a_raw_pil_exception(self):
    with self.assertRaises(ImageError):
      load_image(b"not an image, just garbage bytes" * 20, 1_000_000)

class TestLmstudioVisionCapability(unittest.TestCase):
  def test_no_vision_key_by_default(self):
    self.assertNotIn("vision", lmstudio_models_payload("tiny", 32)["models"][0]["capabilities"])

  def test_vision_true_adds_capability(self):
    self.assertIs(lmstudio_models_payload("tiny", 32, vision=True)["models"][0]["capabilities"]["vision"], True)

class _HTTPCase(unittest.TestCase):
  # shared live-server harness, T4.80 TestLMStudioShim style (test/unit/test_llm_server.py)
  def _run(self, server, method, path, body=None):
    import json, threading, time, urllib.error, urllib.request
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
      data = json.dumps(body).encode() if body is not None else None
      req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                   headers={"Content-Type": "application/json"} if data else {})
      try:
        with urllib.request.urlopen(req, timeout=5) as resp: return resp.status, json.loads(resp.read())
      except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
    finally:
      server.shutdown()
      server.server_close()

class TestVisionUnavailable(_HTTPCase):
  def test_image_request_without_mmproj_is_400_vision_unavailable(self):
    tok = SimpleNamespace(encode=lambda s: [123] if s == "<|image_pad|>" else [1])
    server = LLMServer(("127.0.0.1", 0), model=SimpleNamespace(max_context=32), model_name="tiny", tok=tok, template=None, vision=None)
    body = {"model": "tiny", "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}]}
    status, resp = self._run(server, "POST", "/v1/chat/completions", body)
    self.assertEqual(status, 400)
    self.assertEqual(resp["error"]["code"], "vision_unavailable")

class TestModelsEndpointVisionCapability(_HTTPCase):
  def test_get_models_reports_vision_true_when_encoder_loaded(self):
    server = LLMServer(("127.0.0.1", 0), model=SimpleNamespace(max_context=32), model_name="tiny", tok=None, template=None,
                       vision=SimpleNamespace())  # do_GET only checks `is not None`, never calls the encoder
    status, payload = self._run(server, "GET", "/api/v1/models")
    self.assertEqual(status, 200)
    self.assertIs(payload["models"][0]["capabilities"]["vision"], True)

if __name__ == '__main__':
  unittest.main()
