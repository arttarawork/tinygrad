from __future__ import annotations
import collections, json, os, pathlib, re, time, typing, uuid
from typing import TYPE_CHECKING
from tinygrad import Tensor
from tinygrad.helpers import DEBUG, colored, getenv, stderr_log
from tinygrad.llm.image import DEFAULT_MAX_PIXELS, hash_ids, image_hash, n_visual_tokens, preprocess
from tinygrad.llm.model import VisionInput, snapshot_matches, snapshot_nbytes
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
if TYPE_CHECKING:
  import numpy as np
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.vision import VisionEncoder

# T4.65: k for --mtp speculative decoding (LLMServer.spec_k's default) -- see cli.py's --mtp flag.
SPEC_TOKENS = getenv("SPEC_TOKENS", 3)
# T4.67: MB cap for the cross-session prefill state cache (LLMServer.snapshots) when enabled -- see
# LLMServer.store_snapshot and cli.py's --state-cache flag. Mirrors SPEC_TOKENS/--mtp above: this constant is
# only the MAGNITUDE: LLMServer.__init__'s own default is 0 (off), so a caller who doesn't ask for this
# (e.g. every existing test/null/test_llm_server*.py, constructing LLMServer with no state_cache_mb= kwarg)
# gets byte-identical pre-T4.67 behavior -- snapshot_state/restore_state are never called (see Handler.run_model).
STATE_CACHE_MB = getenv("STATE_CACHE_MB", 2048)

def parse_tool_call(s:str) -> tuple[str, typing.Any]|None:
  s = s.strip()
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  if (fm := re.match(r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", s, re.DOTALL)):
    args = {}
    for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", fm.group(2), re.DOTALL):
      value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2))
      try: args[pm.group(1)] = json.loads(value)
      except json.JSONDecodeError: args[pm.group(1)] = value
    return fm.group(1), args
  return None

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass

# T5.4 (VISION_DESIGN.md section 2.1/2.4): OpenAI image_url content parts. See cli.py's --mmproj flag for how a VisionEncoder gets
# attached to LLMServer, and Handler.do_POST below for how these helpers fit into the request path.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"  # literal text -- the chat template sees plain text; the resulting
                                                                     # image_pad ids get expanded post-tokenize by expand_image_pads
IMAGE_URL_TIMEOUT_S = 10
IMAGE_URL_MAX_BYTES = 20 * 1024 * 1024

class ImageError(Exception):
  """A client-facing image request error: Handler.do_POST catches this and answers 400 with .code as the OpenAI-style error code
  (e.g. "vision_unavailable"; "invalid_image" is the default)."""
  def __init__(self, message:str, code:str = "invalid_image"):
    super().__init__(message)
    self.code = code

def image_error_response(message:str, code:str) -> dict:
  return {"error": {"message": message, "type": "invalid_request_error", "code": code}}

def _fetch_image_url(url:str) -> bytes:
  """data:<mime>;base64,<payload> -> base64-decode; http(s) -> fetch with a timeout and a size cap; anything else is a client error."""
  if url.startswith("data:"):
    header, sep, payload = url.partition(",")
    if not sep or "base64" not in header: raise ImageError(f"unsupported data url (need base64): {url[:50]!r}")
    import base64
    try: return base64.b64decode(payload)
    except Exception as e: raise ImageError(f"bad base64 image data: {e}") from e
  if url.startswith(("http://", "https://")):
    import urllib.request
    try:
      with urllib.request.urlopen(url, timeout=IMAGE_URL_TIMEOUT_S) as resp: data = resp.read(IMAGE_URL_MAX_BYTES + 1)
    except Exception as e: raise ImageError(f"could not fetch image url {url[:80]!r}: {e}") from e
    if len(data) > IMAGE_URL_MAX_BYTES: raise ImageError(f"image at {url[:80]!r} exceeds {IMAGE_URL_MAX_BYTES} bytes")
    return data
  raise ImageError(f"unsupported image url scheme: {url[:50]!r}")

def extract_images(messages:list[dict]) -> list[bytes]:
  """Flattens every message's list-form `content` (OpenAI content parts) to a plain string IN PLACE: a `text` part contributes its
  text, an `image_url` part (a {"url": ...} dict or a bare url string) contributes IMAGE_PLACEHOLDER at that position and its bytes
  are fetched and appended to the returned list, in prompt order (VISION_DESIGN.md section 2.1). A non-list `content` (the common,
  text-only case) is untouched -- text-only requests keep behaving byte-identically. Raises ImageError (-> 400) on an unsupported
  content-part type or image url scheme."""
  images: list[bytes] = []
  for m in messages:
    content = m.get("content")
    if not isinstance(content, list): continue
    parts: list[str] = []
    for c in content:
      if c.get("type") == "text": parts.append(c.get("text", ""))
      elif c.get("type") == "image_url":
        url = c["image_url"]["url"] if isinstance(c["image_url"], dict) else c["image_url"]
        images.append(_fetch_image_url(url))
        parts.append(IMAGE_PLACEHOLDER)
      else: raise ImageError(f"unsupported content part type: {c.get('type')!r}")
    m["content"] = "".join(parts)
  return images

def load_image(data:bytes, max_pixels:int) -> tuple[np.ndarray, tuple[int, int, int]]:
  """image.py's preprocess(), with a bad image (corrupt bytes -> PIL OSError, extreme aspect ratio -> ValueError) turned into an
  ImageError (-> 400 invalid_image) instead of propagating into an unhandled 500."""
  try: return preprocess(data, max_pixels=max_pixels)
  except (ValueError, OSError) as e: raise ImageError(f"invalid image: {e}") from e

def expand_image_pads(ids:list[int], pad_id:int, images:list[tuple[bytes, int, tuple[int, int, int]]],
                      vocab_size:int) -> tuple[list[int], list[tuple[int, int, tuple[int, int, int]]]]:
  """Expands each `pad_id` occurrence in `ids` into a run of n ids, one occurrence per (digest, n, grid) in `images`, in order
  (VISION_DESIGN.md section 2.1): the first min(8, n) ids are hash_ids(digest, vocab_size) -- so two prompts differing only in image
  content differ in token ids too, which is what get_start_pos and the T4.67 state cache key their reuse decision on -- the rest of
  the run stays pad_id (the model never embeds these ids as text; T5.3's VisionInput.spans is how it knows to replace them). Returns
  the expanded ids and each image's (offset, n, grid) span into them, in id order. Raises ImageError if the pad-occurrence count in
  `ids` doesn't match len(images)."""
  out: list[int] = []
  spans: list[tuple[int, int, tuple[int, int, int]]] = []
  it = iter(images)
  for tid in ids:
    if tid != pad_id:
      out.append(tid)
      continue
    img = next(it, None)
    if img is None: raise ImageError("more image placeholders in the prompt than images")
    digest, n, grid = img
    k = min(8, n)
    spans.append((len(out), n, grid))
    out += hash_ids(digest, vocab_size)[:k] + [pad_id] * (n - k)
  if next(it, None) is not None: raise ImageError("fewer image placeholders in the prompt than images")
  return out, spans

def lmstudio_models_payload(model_name:str, max_context:int, vision:bool = False) -> dict:
  # T4.80: LM Studio's native GET /api/v1/models shape -- Hermes's /reasoning command only offers effort
  # levels for a model whose provider answers this probe with a non-empty capabilities.reasoning.allowed_options.
  # T5.4: capabilities.vision=True (only set when True -- omitted, not False, otherwise) lets Hermes's vision auxiliary
  # task route to this server; see cli.py's --mmproj.
  capabilities: dict[str, typing.Any] = {"reasoning": {"allowed_options": ["none", "minimal", "low", "medium", "high", "xhigh"]}}
  if vision: capabilities["vision"] = True
  return {"models": [{"key":model_name, "id":model_name, "object":"model", "type":"llm", "max_context_length":max_context,
                      "capabilities": capabilities,
                      "loaded_instances": [{"id":model_name, "config": {"context_length":max_context}}]}]}

def template_kwargs(body:dict) -> dict:
  # chat_template_kwargs (e.g. {"enable_thinking": false}) go to the template, as llama-server does; a top-level
  # reasoning_effort (LM Studio's /reasoning knob) overrides enable_thinking when present -- see T4.80.
  kwargs = {"preserve_thinking": True, **(body.get("chat_template_kwargs") or {})}
  if isinstance(effort := body.get("reasoning_effort"), str): kwargs["enable_thinking"] = effort.strip().lower() != "none"
  return kwargs

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .buf for the final parse
  def __init__(self, reasoning:bool=False):
    self.buf = ""
    self.mode = "reasoning" if reasoning else "undecided"  # output inside a think block is sent as reasoning_content
  def split(self, tag:str, final:bool) -> tuple[str, bool]:
    # split buf on the first full tag, holding back a partial tag at the end unless final
    if tag in self.buf:
      before, self.buf = self.buf.split(tag, 1)
      return before, True
    hold = max((i for i in range(1, min(len(self.buf), len(tag))+1) if tag.startswith(self.buf[-i:])), default=0) if not final else 0
    emit, self.buf = self.buf[:len(self.buf)-hold], self.buf[len(self.buf)-hold:]
    return emit, False
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.buf += piece
    if self.mode == "undecided":  # decide whether the output starts with a think block
      if not final and len(self.buf) < len("<think>") and "<think>".startswith(self.buf): return
      self.mode, self.buf = ("reasoning", self.buf[len("<think>"):]) if self.buf.startswith("<think>") else ("content", self.buf)
    if self.mode == "reasoning":
      emit, done = self.split("</think>", final)
      if emit: yield "reasoning_content", emit
      if not done: return
      self.mode = "content"
    if self.mode == "tool": return
    emit, found = self.split("<tool_call>", final)
    if emit: yield "content", emit
    if found: self.mode, self.buf = "tool", "<tool_call>" + self.buf

def splice_ids(last:tuple[str, list[int], int, list[int]], rendered:str, messages:list[dict],
               render:typing.Callable[[list[dict], bool], str], tok:SimpleTokenizer) -> list[int]|None:
  """Tokenize a follow-up request by splicing the model's own generated ids in place of the client's re-rendered assistant turn.
  Clients re-render that turn (tool_calls as JSON, think blocks trimmed, whitespace) so re-tokenizing its text rarely reproduces the ids
  the KV/recurrent state was built on -- and a recurrent model reuses its state only for an exact token-prefix extension (get_start_pos).
  The template's end-of-turn marker is a special token, so encoding from it onward is boundary-exact. None = fall back to a plain encode."""
  prev_rendered, prev_ids, n, gen = last
  if len(messages) <= n or messages[n].get("role") != "assistant" or not rendered.startswith(prev_rendered): return None
  upto = render(messages[:n+1], False)  # history through our assistant turn, as the client's copy of it re-renders
  if not (upto.startswith(prev_rendered) and rendered.startswith(upto)): return None
  content = messages[n].get("content")
  def norm(t:str) -> str: return " ".join(t.split())
  if isinstance(content, str) and norm(content) and norm(content) not in norm(tok.decode(gen)): return None  # the client edited our reply
  # the end-of-turn marker must decode to real text (a special token): an empty marker would 'match' at the end of any turn
  turn, ends = upto[len(prev_rendered):], [e for e in (tok.decode([t]) for t in (tok.eos_id, tok.eot_id) if t is not None) if e]
  if not ends or (idx := max(turn.rfind(e) for e in ends)) < 0: return None
  return prev_ids + gen + tok.encode(turn[idx:] + rendered[len(upto):])

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path == "/api/v1/models":
      payload = lmstudio_models_payload(self.server.model_name, self.server.model.max_context, self.server.vision is not None)
      self.send_data(json.dumps(payload).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, record:tuple[str, list[int], int]|None=None, vision:VisionInput|None=None):
    model, tok = self.server.model, self.server.tok
    prompt_tokens = len(ids)
    cache_start_pos = model.get_start_pos(ids)
    # T4.67: (a) the splice/live-cache path above found nothing to reuse -- before falling back to a fully cold
    # prefill (c), try the cross-session state cache (b): the longest snapshot whose ids exactly prefix this
    # request's. Skipped entirely when the cache is off (state_cache_mb<=0), so snapshot_state/restore_state are
    # then never called -- byte-identical to pre-T4.67 behavior.
    if cache_start_pos == 0 and self.server.state_cache_mb > 0 and (snap := self.server.find_snapshot(ids)) is not None:
      model.restore_state(snap)
      cache_start_pos = model.get_start_pos(ids)
    # T5.4: " img:{images}/{visual tokens}" after the in: field, only when this request actually carries images.
    img_field = f" img:{len(vision.spans)}/{sum(n for _, n, _ in vision.spans)}" if vision is not None and vision.spans else ""
    stderr_log(f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}{img_field}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = pt = time.perf_counter()
    dec = tok.stream_decoder()
    router = StreamRouter(reasoning)
    def log_stats(interrupted:bool=False):
      et = time.perf_counter()
      total = f"total:{et-st:6.2f}s"
      stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
                 f"out:{len(out):5d}  {colored('--', 'BLACK')}  {colored(total, 'red') if interrupted else total}\n")
    completed = False
    # T4.65: --mtp routes chat completions through MTP speculative decoding when the loaded model actually
    # has an mtp_head (Transformer.from_gguf under MTP=1) -- absent either condition, this is model.generate,
    # byte-identical to before --mtp existed. speculative_generate(temperature=temperature) already picks
    # its own greedy (temperature<=0) vs sampled (>0) path internally, so no extra branching is needed here.
    # T5.4: speculative_generate has no vision plumbing -- an image request always takes the plain generate() path.
    use_spec = self.server.mtp and model.mtp_head is not None and vision is None
    gen = model.speculative_generate(ids, k=self.server.spec_k, temperature=temperature) if use_spec \
      else model.generate(ids, temperature=0.0 if vision is not None else temperature, vision=vision)  # T5.5: image requests are greedy --
      # only the greedy vision jit family is warmed (each extra prefill family costs ~0.78 GB on the 3090, see model.VISION_CHUNK)
    try:
      yield chunk({"role":"assistant", "content":""})
      for next_id in gen:
        if len(out) == 0:
          stderr_log(f"prefill:{(prompt_tokens-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
          # T4.67: prefill for `ids` just completed (model._cached_tokens now covers exactly `ids` -- same
          # boundary generate()/speculative_generate() themselves just set) -- park it for a later session.
          if self.server.state_cache_mb > 0: self.server.store_snapshot(ids)
        if tok.is_end(next_id): break
        out.append(next_id)
        for field, delta in router.route(dec(next_id)): yield chunk({field:delta})
        if max_tokens is not None and len(out) >= max_tokens:
          finish_reason = "length"
          break
      for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
      tool_calls: list[dict] = []
      for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", router.buf, re.DOTALL):
        if (parsed := parse_tool_call(m.group(1))) is None:
          stderr_log(f"failed to parse tool call: {m.group(1)[:200]}")
          yield chunk({"content":m.group(0)})  # don't silently drop output the client can't use
        else:
          name, args = parsed
          tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                             "function":{"name":name, "arguments":args if isinstance(args, str) else json.dumps(args)}})
      if tool_calls:
        yield chunk({"tool_calls":tool_calls})
        if finish_reason == "stop": finish_reason = "tool_calls"
      completed = True
      if record is not None: self.server.last = (*record, out)  # what the model state now holds, for splice_ids on the next turn
      yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
      if include_usage:
        yield {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                        "total_tokens": prompt_tokens + len(out)}, **tmpl}
      log_stats()
    except GeneratorExit:
      if not completed: log_stats(interrupted=True)
      raise

  def do_POST(self):
    request_st = time.perf_counter()
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  ")
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # render and tokenize
      normalize_messages(body["messages"])
      # T5.4 (VISION_DESIGN.md section 2.1/2.4): image_url content parts. extract_images flattens body["messages"] content lists to
      # plain text IN PLACE -- a no-op for the common text-only body, which is untouched from here on -- collecting raw image bytes.
      loaded: list[tuple[bytes, np.ndarray, tuple[int, int, int]]] = []
      try:
        images = extract_images(body["messages"])
        if images and (self.server.vision is None or self.server.image_pad_id is None):
          raise ImageError("start the server with --mmproj", code="vision_unavailable")
        for data in images:
          patches, grid = load_image(data, self.server.vision_max_pixels)
          loaded.append((image_hash(data), patches, grid))
      except ImageError as e:
        return self.send_data(json.dumps(image_error_response(str(e), e.code)).encode(), status_code=400)
      kwargs = template_kwargs(body)
      def render(messages:list[dict], add_generation_prompt:bool) -> str:
        return self.server.template.render(messages=messages, tools=body.get("tools"), add_generation_prompt=add_generation_prompt, **kwargs)
      rendered = render(body["messages"], True)
      ids: list[int]
      vision_input: VisionInput|None = None
      record: tuple[str, list[int], int]|None = None
      if loaded:
        # image prompts skip splice_ids (the splice cache stays text-only -- expanded ids already carry the image identity for the
        # T4.67 state cache) and pass no record= either; run_model forces the plain generate() path when vision is set.
        enc, pad_id = self.server.vision, self.server.image_pad_id
        assert enc is not None and pad_id is not None  # guaranteed by the vision_unavailable check above
        vocab_size = int(self.server.model.token_embd.weight.shape[0])
        try:
          ids, spans = expand_image_pads(self.server.tok.encode(rendered), pad_id,
                                         [(digest, n_visual_tokens(grid), grid) for digest, _, grid in loaded], vocab_size)
        except ImageError as e:
          return self.send_data(json.dumps(image_error_response(str(e), e.code)).encode(), status_code=400)
        embed_list = [enc(Tensor(patches, device=enc.device), grid) for _, patches, grid in loaded]
        vision_input = VisionInput(spans, Tensor.cat(*embed_list, dim=0).realize())
      else:
        ids = (splice_ids(self.server.last, rendered, body["messages"], render, self.server.tok) if self.server.last else None) \
          or self.server.tok.encode(rendered)
        record = (rendered, ids, len(body["messages"]))
      think = f"think:{'on' if kwargs['enable_thinking'] else 'off'}  {colored('--', 'BLACK')}  " if "enable_thinking" in kwargs else ""
      stderr_log(f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  {think}")
      if len(ids) >= self.server.model.max_context:
        stderr_log(f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                              reasoning=rendered.rstrip().endswith("<think>"), record=record, vision=vision_input)
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, reasoning, tool_calls, finish_reason = [], [], [], "stop"
        for c in chunks:
          if not c["choices"]: continue
          choice = c["choices"][0]
          if (delta := choice.get("delta", {})):
            if delta.get("content"): out.append(delta["content"])
            if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
            tool_calls += [{k:v for k, v in tc.items() if k != "index"} for tc in delta.get("tool_calls", [])]
          if choice.get("finish_reason"): finish_reason = choice["finish_reason"]
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(out) or None}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if tool_calls: message["tool_calls"] = tool_calls
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
    elif self.path == "/api/v1/models/load":  # T4.80: LM Studio's load probe -- a no-op, the model is always loaded
      self.send_data(json.dumps({"status":"loaded", "model":self.server.model_name, "context_length":self.server.model.max_context}).encode())
    else:
      # a clean 404, not an exception: local tooling probes other servers' APIs on this port (Ollama's /api/show and
      # friends), and raising here tears down the connection and spams a traceback per probe
      self.send_data(json.dumps({"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}}).encode(),
                     status_code=404)

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any,
               mtp:bool=False, spec_k:int=SPEC_TOKENS, state_cache_mb:int=0, vision:VisionEncoder|None=None):
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    self.mtp, self.spec_k = mtp, spec_k  # T4.65: --mtp/SPEC_TOKENS -- see Handler.run_model's use_spec
    self.last: tuple[str, list[int], int, list[int]]|None = None  # (rendered prompt, ids, message count, generated ids) of the last completed request
    # T4.67: cross-session state cache -- self.last above only ever remembers ONE (the most recent) sequence;
    # this keyed, MB-capped, LRU dict lets a later request reuse ANY previously-snapshotted sequence whose
    # tokens it exactly extends, not just the immediately preceding one. Default 0 = off (byte-identical to
    # pre-T4.67 -- see Handler.run_model); pass state_cache_mb=STATE_CACHE_MB (or cli.py's --state-cache) to enable.
    self.state_cache_mb = state_cache_mb
    self.snapshots: collections.OrderedDict[tuple[int, ...], dict] = collections.OrderedDict()  # insertion/touch order == LRU order
    # T5.4: --mmproj wiring (VISION_DESIGN.md section 2.4) -- vision is None unless cli.py's --mmproj loaded a VisionEncoder.
    # image_pad_id is resolved ONCE here, not per-request: None (no --mmproj, or "<|image_pad|>" isn't a single token in this
    # tokenizer) makes every image_url request 400 vision_unavailable in do_POST, same as vision being None does.
    self.vision = vision
    self.vision_max_pixels = int(os.environ.get("VISION_MAX_PIXELS") or DEFAULT_MAX_PIXELS)
    try: pad_ids = tok.encode("<|image_pad|>") if tok is not None else []
    except Exception: pad_ids = []
    # a tok stub whose .encode isn't configured for real ids (test/null/*'s bare Mock()) degrades to vision off, not a crash --
    # same outcome as a real tokenizer where "<|image_pad|>" isn't exactly one id.
    self.image_pad_id = pad_ids[0] if isinstance(pad_ids, list) and len(pad_ids) == 1 else None
    super().__init__(server_address, Handler)

  def find_snapshot(self, ids:list[int]) -> dict|None:
    """The longest stored snapshot whose tokens are an exact prefix of `ids` (snapshot_matches), touched for
    LRU on a hit. None if no stored snapshot applies."""
    best_key = max((k for k in self.snapshots if snapshot_matches(self.snapshots[k], ids)), key=len, default=None)
    if best_key is None: return None
    self.snapshots.move_to_end(best_key)
    return self.snapshots[best_key]

  def store_snapshot(self, ids:list[int]) -> None:
    """Snapshot self.model's current state -- assumed to have just finished prefilling exactly `ids` (see
    Handler.run_model) -- under key tuple(ids), LRU-evicting the oldest entries to stay under state_cache_mb
    (always keeping at least the just-stored entry, even if it alone exceeds the cap)."""
    key = tuple(ids)
    if key in self.snapshots:
      self.snapshots.move_to_end(key)
      return
    self.snapshots[key] = self.model.snapshot_state()
    cap = self.state_cache_mb * 1024 * 1024
    total = sum(snapshot_nbytes(s) for s in self.snapshots.values())
    while total > cap and len(self.snapshots) > 1:
      total -= snapshot_nbytes(self.snapshots.popitem(last=False)[1])
