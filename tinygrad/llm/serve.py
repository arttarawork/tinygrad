from __future__ import annotations
import collections, json, os, pathlib, re, threading, time, typing, uuid
from typing import TYPE_CHECKING
from tinygrad import Tensor
from tinygrad.helpers import DEBUG, colored, getenv, stderr_log
from tinygrad.llm.image import DEFAULT_MAX_PIXELS, hash_ids, image_hash, n_visual_tokens, preprocess
from tinygrad.llm.model import VisionInput, snapshot_matches, snapshot_nbytes, snapshot_nbytes_for
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler, filter_keys
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

# T4.85: Hermes (and most agent clients) send no `temperature`; an absent value used to mean 0 = greedy decoding, and Qwen's
# thinking mode under greedy decoding degenerates into endless repetition (2026-09-07: 12k chars of reasoning, three sentences
# repeated 11x). DEFAULT_TEMPERATURE (env, default 0 = byte-identical to before) is what an omitted temperature means; the
# standing recipe sets 0.6 (Qwen's thinking-mode recommendation). An explicit temperature in the request always wins.
DEFAULT_TEMPERATURE = getenv("DEFAULT_TEMPERATURE", 0.0)
# T4.92: Qwen's model card recommends DIFFERENT samplers per mode (thinking: temperature 1.0/top-p 0.95; non-thinking:
# temperature 0.7/top-p 0.8) -- one DEFAULT_TEMPERATURE for both modes was always a compromise. DEFAULT_TEMPERATURE_THINK
# (env, default: falls back to DEFAULT_TEMPERATURE -- so a server that never sets it keeps today's single-default
# behavior, byte-identical) is what an omitted temperature means for a THINKING request only; a non-thinking request keeps
# meaning DEFAULT_TEMPERATURE, same as before T4.92. An explicit request temperature always wins, in either mode.
DEFAULT_TEMPERATURE_THINK = getenv("DEFAULT_TEMPERATURE_THINK", DEFAULT_TEMPERATURE)
def request_temperature(body:dict, thinking:bool) -> float:
  return float(body.get("temperature", DEFAULT_TEMPERATURE_THINK if thinking else DEFAULT_TEMPERATURE))

# T4.92: Qwen's model card also recommends presence_penalty=1.5 in non-thinking mode ("to reduce endless repetitions"; thinking
# mode wants 0 -- penalizing reused reasoning vocabulary mid-thought would fight the model's own working-through-it style).
# PRESENCE_PENALTY (env, default 0 = byte-identical to before this existed -- see model.py's Transformer.generate) is what a
# non-thinking request gets when it sends no presence_penalty of its own; a thinking request never gets the env default. A
# request's own `presence_penalty` always wins in EITHER mode (0 explicitly disables it, same as omitting it in non-thinking).
PRESENCE_PENALTY = getenv("PRESENCE_PENALTY", 0.0)
def request_presence_penalty(body:dict, thinking:bool) -> float:
  if "presence_penalty" in body: return float(body["presence_penalty"])
  return 0.0 if thinking else PRESENCE_PENALTY

# T4.88: thinking budget. Qwen's thinking mode overthinks (2026-09-07: 45k reasoning tokens, 4 h, for one turn); Qwen's own
# recipe is to cap the think block and force it closed with a short "answer now" sentence, then let the model continue from
# its cached prefix. THINK_BUDGET (env, default 0 = unlimited) is the cap at reasoning_effort=medium; low/minimal get a
# quarter/eighth, high 4x, xhigh unlimited. The plain generate() path only (MTP's speculative loop is not spliced).
THINK_BUDGET = getenv("THINK_BUDGET", 0)
THINK_CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
def thinking_budget(body:dict) -> int:
  if not THINK_BUDGET: return 0
  e = str(body.get("reasoning_effort", "medium")).strip().lower()
  return {"minimal": THINK_BUDGET // 8, "low": THINK_BUDGET // 4, "high": THINK_BUDGET * 4, "xhigh": 0}.get(e, THINK_BUDGET)

# T4.89: reasoning loop breaker. The 2026-09-07 captures show Qwen's "anxious" thinking is literal sentence cycles (one 44k-char
# think block said "actually, the simplest is: keep two files, zip them." 11 times, oscillating with the other option), not long
# productive thinking. When a sentence of >=6 words recurs LOOP_REPEATS times inside the think block the server injects a decisive
# sentence in the model's own voice and continues; after LOOP_NUDGES nudges it closes the think block (THINK_CLOSE). LOOP_REPEATS=0 disables.
LOOP_REPEATS, LOOP_NUDGES = getenv("LOOP_REPEATS", 3), getenv("LOOP_NUDGES", 3)
LOOP_NUDGE = ("\n\nI notice I've been going back and forth over the same point. It's settled: I'll go with the approach I already have, "
              "stop re-checking it, and move on to the next step.\n\n")
class LoopDetector:
  """Counts normalized reasoning sentences (>=6 words) as they stream; feed() returns the sentence that just hit `repeats`, then resets."""
  def __init__(self, repeats:int): self.repeats, self.buf, self.counts = repeats, "", collections.Counter[str]()
  def feed(self, delta:str) -> str|None:
    self.buf += delta
    while (m := re.search(r"[.!?]\s|\n", self.buf)) is not None:
      sent, self.buf = " ".join(self.buf[:m.end()].lower().split()), self.buf[m.end():]
      if len(sent.split()) < 6: continue
      self.counts[sent] += 1
      if self.counts[sent] >= self.repeats:
        self.buf, self.counts = "", collections.Counter()
        return sent
    return None

class StreamLog:
  """T4.83: STREAM_LOG=<path> appends every request's streamed text as it is generated (reasoning and content, flushed per
  token) so `tail -f` or the LAN viewer page (~/.hermes/stream-viewer) shows the thinking LIVE -- Hermes itself only shows
  reasoning once the turn completes. One rotation at 8 MB keeps the file small."""
  def __init__(self, path:str, header:str):
    if os.path.exists(path) and os.path.getsize(path) > 8_000_000: os.replace(path, path + ".1")
    self.f, self.field = open(path, "a", encoding="utf-8"), ""
    self.f.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {header} =====\n")
    self.f.flush()
  def write(self, field:str, text:str) -> None:
    if field != self.field:
      self.f.write(f"\n--- {field} ---\n")
      self.field = field
    self.f.write(text)
    self.f.flush()
  def close(self) -> None: self.f.close()

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

# T4.90: heartbeat period (s) for streamed replies; 0 disables. The 2026-09-08 session loss started with Hermes's stream
# watchdog dropping a request during a 2 h prefill silence and the server not noticing the hang-up until its first token.
KEEPALIVE_SEC = getenv("KEEPALIVE_SEC", 30)

class Handler(HTTPRequestHandler):
  def stream_json(self, source):
    """viz's stream_json plus a heartbeat: while the generator is silent (prefill, a buffered tool call) a helper thread
    writes an empty-delta chunk every KEEPALIVE_SEC, so clients' stale-stream watchdogs stay quiet and a hung-up client
    is noticed within ~KEEPALIVE_SEC instead of at the first real token (generation then stops). The generator itself
    stays on this thread -- METAL objects must (T4.84); the helper only touches the socket."""
    if not KEEPALIVE_SEC: return super().stream_json(source)
    lock, st = threading.Lock(), typing.cast(dict[str, typing.Any], {"last": time.monotonic(), "done": False, "gone": False, "tmpl": None})
    def write(d:dict):
      self.wfile.write(f"data: {json.dumps(filter_keys(d))}\n\n".encode("utf-8"))
      self.wfile.flush()
      st["last"] = time.monotonic()
    def beat():
      while not st["done"]:
        time.sleep(min(1.0, KEEPALIVE_SEC))
        if st["done"] or st["tmpl"] is None or time.monotonic() - st["last"] < KEEPALIVE_SEC: continue
        with lock:
          if st["done"]: break
          try: write({**st["tmpl"], "choices": [{"index":0, "delta":{}, "finish_reason":None}]})
          except OSError:
            st["gone"] = True
            break
    t = threading.Thread(target=beat, daemon=True)
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      t.start()
      for r in source:
        if st["gone"]: break
        with lock: write(r)
        if st["tmpl"] is None: st["tmpl"] = {k:v for k, v in r.items() if k != "choices"}
      if not st["gone"]:
        with lock:
          self.wfile.write("data: [DONE]\n\n".encode("utf-8"))
          self.wfile.flush()
    except (BrokenPipeError, ConnectionResetError): pass
    finally:
      st["done"] = True
      source.close()
      if t.is_alive(): t.join(timeout=2)

  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path == "/api/v1/models":
      payload = lmstudio_models_payload(self.server.model_name, self.server.model.max_context, self.server.vision is not None)
      self.send_data(json.dumps(payload).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, record:tuple[str, list[int], int]|None=None, vision:VisionInput|None=None, think_budget:int=0,
                presence_penalty:float=0.0):
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
    slog = StreamLog(p, f"{model_name} in:{cache_start_pos}+{len(ids)-cache_start_pos}{img_field}") if (p := os.environ.get("STREAM_LOG")) else None
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
    # T4.92: presence_penalty is plumbed to the plain generate() path only -- speculative_generate's accept/resample math
    # (Leviathan et al., see model.py's spec_accept) has its own correctness proof that a logit bias would need to thread
    # through carefully, out of scope here, so a --mtp request never applies one (T4.92's own task note).
    use_spec = self.server.mtp and model.mtp_head is not None and vision is None
    gen = model.speculative_generate(ids, k=self.server.spec_k, temperature=temperature) if use_spec \
      else model.generate(ids, temperature=0.0 if vision is not None else temperature, vision=vision, presence_penalty=presence_penalty)
      # T5.5: image requests are greedy -- only the greedy vision jit family is warmed (each extra prefill family costs
      # ~0.78 GB on the 3090, see model.VISION_CHUNK)
    try:
      yield chunk({"role":"assistant", "content":""})
      it = iter(gen)
      def inject(text:str):
        # T4.88/T4.89: feed `text` as if the model wrote it, then continue the same request from ids+out (the live cached
        # prefix -- only `text`'s few tokens prefill).
        nonlocal gen, it
        gen.close()
        for t in tok.encode(text):
          out.append(t)
          for field, delta in router.route(dec(t)):
            if slog is not None: slog.write(field, delta)
            yield chunk({field:delta})
        # T4.92: presence_penalty carries over -- generate() resets its own mask per call, so without this an active
        # penalty would silently vanish for the rest of the response after a loop-breaker/think-budget injection.
        gen = model.generate(ids + out, temperature=0.0 if vision is not None else temperature, vision=vision, presence_penalty=presence_penalty)
        it = iter(gen)
      loops, nudges = (LoopDetector(LOOP_REPEATS) if LOOP_REPEATS and not use_spec else None), 0
      while (next_id := next(it, None)) is not None:
        if len(out) == 0:
          stderr_log(f"prefill:{(prompt_tokens-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
          # T4.67: prefill for `ids` just completed (model._cached_tokens now covers exactly `ids` -- same
          # boundary generate()/speculative_generate() themselves just set) -- park it for a later session.
          if self.server.state_cache_mb > 0: self.server.store_snapshot(ids, vision is not None)
        if tok.is_end(next_id): break
        out.append(next_id)
        hit = None
        for field, delta in router.route(dec(next_id)):
          if slog is not None: slog.write(field, delta)
          yield chunk({field:delta})
          if loops is not None and field == "reasoning_content": hit = loops.feed(delta) or hit
        if max_tokens is not None and len(out) >= max_tokens:
          finish_reason = "length"
          break
        if hit is not None and router.mode == "reasoning":
          nudges += 1
          msg = f"reasoning loop x{LOOP_REPEATS} ({hit[:48]!r}) -- nudge {nudges}/{LOOP_NUDGES}"
          stderr_log(f"{colored(msg, 'yellow')}  {colored('--', 'BLACK')}  ")
          yield from inject(LOOP_NUDGE if nudges <= LOOP_NUDGES else THINK_CLOSE)
        elif think_budget and not use_spec and router.mode == "reasoning" and len(out) >= think_budget:
          # T4.88: budget hit -- close the think block with Qwen's own "answer now" sentence. Once per request.
          stderr_log(f"{colored(f'think budget {think_budget} hit -- closing the think block', 'yellow')}  {colored('--', 'BLACK')}  ")
          yield from inject(THINK_CLOSE)
          think_budget = 0
      for field, delta in router.route(dec(), final=True):
        if slog is not None: slog.write(field, delta)
        yield chunk({field:delta})
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
    finally:
      if slog is not None: slog.close()

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
      thinking = rendered.rstrip().endswith("<think>")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=request_temperature(body, thinking),
                              reasoning=thinking, record=record, vision=vision_input, think_budget=thinking_budget(body),
                              presence_penalty=request_presence_penalty(body, thinking))
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

  def store_snapshot(self, ids:list[int], one_shot:bool=False) -> None:
    """Snapshot self.model's current state -- assumed to have just finished prefilling exactly `ids` (see
    Handler.run_model) -- under key tuple(ids), LRU-evicting the oldest entries to stay under state_cache_mb
    (always keeping at least the just-stored entry, even if it alone exceeds the cap)."""
    # T5.7c (2026-09-06): image requests are one-shot auxiliary calls (screenshot analysis) that are never continued, yet each
    # snapshot carries the ~150 MB fixed DeltaNet state; five of them ate the 3090's headroom, the 46k-token session snapshot
    # then OOM'd and the whole cache was dropped -> a 41-minute re-prefill. Don't cache them.
    if one_shot: return
    key = tuple(ids)
    if key in self.snapshots:
      self.snapshots.move_to_end(key)
      return
    cap = self.state_cache_mb * 1024 * 1024
    # T5.7: a snapshot is allocated on the model's devices and the 3090 has only a few hundred MB of headroom once the vision
    # jit families are up. 2026-09-05: a 39k-token Telegram session's ~0.8 GB snapshot OOM'd on NV *before the first token was
    # streamed*, Hermes saw an empty response and re-ran the 4-minute prefill four times. So: (1) predict the size from an
    # existing snapshot's bytes-per-token and skip sequences that could never fit the cap; (2) evict BEFORE allocating so the
    # old and new snapshots never coexist; (3) an allocation failure drops the cache and the request continues uncached.
    if self.snapshots:
      s0 = next(iter(self.snapshots.values()))
      if (est := snapshot_nbytes_for(s0, len(ids))) > cap:
        stderr_log(f"{colored(f'state cache: skip {len(ids)}-token snapshot (~{est>>20} MB > cap)', 'yellow')}  {colored('--', 'BLACK')}  ")
        return
      total = sum(snapshot_nbytes(v) for v in self.snapshots.values())
      while total + est > cap and self.snapshots: total -= snapshot_nbytes(self.snapshots.popitem(last=False)[1])
    try: snap = self.model.snapshot_state()
    except MemoryError as e:
      self.snapshots.clear()
      stderr_log(f"{colored(f'state cache: snapshot dropped ({str(e)[:50]}), cache cleared', 'yellow')}  {colored('--', 'BLACK')}  ")
      return
    self.snapshots[key] = snap
    total = sum(snapshot_nbytes(v) for v in self.snapshots.values())
    while total > cap and len(self.snapshots) > 1:
      total -= snapshot_nbytes(self.snapshots.popitem(last=False)[1])
