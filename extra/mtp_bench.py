"""T4.94 prep: bench tool for MTP speculative decoding on the card build (see HANDOFF_2026-09-12.md sec 5/
TASKS.md T4.94). Loads a GGUF with MTP=1 (a Context override -- see model.py's MTP ContextVar docstring: it's
read once inside Transformer.from_gguf, so the override only needs to wrap that one call, not stay open for
the rest of the process), then measures (a) plain model.generate() decode tok/s, (b) model.speculative_generate()
at each --k, and (c) whether the two agree token-for-token (they must on greedy -- see speculative_generate's
own docstring/SPEC_NOTES.md). Mirrors extra/benchmark_llm.py's own prefill/decode timing split (GlobalCounters
+ perf_counter around next(gen), first token counted as prefill/time-to-first-token, the rest as decode).

--fill N first primes the live cache (main model KV/GDN state AND the MTP head's own KV cache) to N tokens via
one speculative_generate priming call, chunked at 128 tokens (extra/flip_rate.py's teacher_forced_run uses the
same bound-Variable prefill mechanics, chunk_size=32 there -- widened here for a fast large fill) so the
orchestrator can measure at a given filled-context depth (HANDOFF_2026-09-12 sec 6: decode speed depends on
FILLED context, not window size). Reusing speculative_generate itself for the fill (rather than a hand-rolled
prefill loop through Transformer.__call__ directly) is deliberate: it's the only thing that also primes the MTP
head's own attention block cache for the filled region (T4.66l's mtp_head.prefill, called from
speculative_generate's own chunked-prefill loop only -- plain generate() never touches mtp_head at all), which
matters for a representative accept-rate reading at a given fill depth.

Usage (see this task's report for the exact card command lines):
  PYTHONPATH=. python extra/mtp_bench.py --model qwen.gguf --max-context 262144 --fill 2048 \\
    --prompt-tokens 64 --decode-tokens 64 --k 1,2,3
"""
from __future__ import annotations
import argparse, contextlib, io, re, time
from tinygrad import Context
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.model import Transformer

FILL_CHUNK = 128  # wider than generate()'s own default chunk_size=32 -- just for a fast large fill, unrelated
                   # to the --chunk-size the TIMED prefill below uses (that one matches production/benchmark_llm.py)

SPEC_STATS_RE = re.compile(
  r"\[SPEC_STATS\] iters=(\d+) emitted=(\d+) drafted=(\d+) avg_accept_len=([\d.]+) drafts_per_token=([\d.]+) "
  r"accept_len_hist=\{([^}]*)\}")

def parse_spec_stats(text:str) -> dict|None:
  """Pulls model.py's [SPEC_STATS] summary line (SPEC_STATS ContextVar -- printed once, in a try/finally, when
  speculative_generate's generator ends, see its own docstring) out of captured stdout. There is no
  programmatic accessor for it: the histogram/counters are plain local variables inside the generator
  function, never returned or stored on the model -- exposing them without touching model.py's own logic (this
  task is audit-only for model.py/serve.py) means reading the print it already emits. Returns None if the
  line never printed (e.g. the generator was closed before its first outer DRAFT/VERIFY iteration)."""
  m = SPEC_STATS_RE.search(text)
  if m is None: return None
  hist: dict[int, int] = {}
  if m.group(6):
    for pair in m.group(6).split(", "):
      acc_len, count = pair.split(":")
      hist[int(acc_len)] = int(count)
  return {"iters": int(m.group(1)), "emitted": int(m.group(2)), "drafted": int(m.group(3)),
          "avg_accept_len": float(m.group(4)), "drafts_per_token": float(m.group(5)), "accept_len_hist": hist}

def deterministic_prompt(n:int, offset:int=0) -> list[int]:
  """Same shape as extra/benchmark_llm.py's own synthetic prompt (a fixed first token + a repeating ramp).
  `offset` gives --fill and the timed prompt disjoint token streams (the ramp is 1000 wide, so a 100000
  stride keeps any two small offsets from ever overlapping), so a fill+prompt concatenation isn't just the
  same short cycle repeated end to end."""
  base = 1000 + 100_000 * offset
  return [257 + 100_000 * offset] + [base + i % 1000 for i in range(n - 1)]

def fill_context(model:Transformer, tokens:list[int], chunk_size:int=FILL_CHUNK) -> None:
  """Primes model's live cache to `tokens` (see module docstring for why this goes through
  speculative_generate rather than generate()). Stops before the outer DRAFT/VERIFY loop ever runs: one
  next() pulls exactly the prefill-phase anchor token, and closing right after discards it cleanly (dropping
  it from the timed run, same as a fresh cache would never have had it) -- speculative_generate's own
  try/finally is a no-op here since SPEC_STATS is never turned on for this call. Empty `tokens` is a no-op:
  speculative_generate's prefill loop assumes prompt_len>=1 (see its own comment) and there's nothing to
  prime anyway."""
  if not tokens: return
  gen = model.speculative_generate(list(tokens), k=1, chunk_size=chunk_size)
  next(gen)
  gen.close()

def bench_plain(model:Transformer, fill_tokens:list[int], prompt_tokens:list[int], decode_tokens:int,
                 chunk_size:int) -> dict:
  """model.generate() baseline: mirrors extra/benchmark_llm.py's own timing split exactly."""
  fill_context(model, fill_tokens)
  gen = model.generate(fill_tokens + prompt_tokens, chunk_size=chunk_size)
  GlobalCounters.reset()
  st = time.perf_counter()
  out = [next(gen)]
  pt = time.perf_counter()
  prefill_tok_s, prefill_gb_s = len(prompt_tokens) / (pt - st), GlobalCounters.global_mem / (pt - st) / 1e9
  GlobalCounters.reset()
  for _ in range(decode_tokens): out.append(next(gen))
  et = time.perf_counter()
  decode_tok_s = decode_tokens / (et - pt) if decode_tokens else float("nan")
  decode_gb_s = GlobalCounters.global_mem / (et - pt) / 1e9 if decode_tokens else float("nan")
  return {"prefill_tok_s": prefill_tok_s, "prefill_gb_s": prefill_gb_s,
          "decode_tok_s": decode_tok_s, "decode_gb_s": decode_gb_s, "output": out}

def bench_spec(model:Transformer, fill_tokens:list[int], prompt_tokens:list[int], decode_tokens:int, k:int,
               chunk_size:int) -> dict:
  """model.speculative_generate() at a given k: same timing split as bench_plain (directly comparable tok/s),
  plus the tree's own SPEC_STATS accept-length histogram/drafts-per-token (see parse_spec_stats)."""
  fill_context(model, fill_tokens)
  with Context(SPEC_STATS=1):
    gen = model.speculative_generate(fill_tokens + prompt_tokens, k=k, chunk_size=chunk_size)
    GlobalCounters.reset()
    st = time.perf_counter()
    out = [next(gen)]
    pt = time.perf_counter()
    prefill_tok_s, prefill_gb_s = len(prompt_tokens) / (pt - st), GlobalCounters.global_mem / (pt - st) / 1e9
    GlobalCounters.reset()
    for _ in range(decode_tokens): out.append(next(gen))
    et = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): gen.close()  # forces the try/finally SPEC_STATS print to fire now
  decode_tok_s = decode_tokens / (et - pt) if decode_tokens else float("nan")
  decode_gb_s = GlobalCounters.global_mem / (et - pt) / 1e9 if decode_tokens else float("nan")
  return {"prefill_tok_s": prefill_tok_s, "prefill_gb_s": prefill_gb_s, "decode_tok_s": decode_tok_s,
          "decode_gb_s": decode_gb_s, "output": out, "stats": parse_spec_stats(buf.getvalue())}

def format_stats(stats:dict|None) -> str:
  if stats is None: return "SPEC_STATS never printed (0 outer iterations)"
  hist_str = ", ".join(f"{acc}:{cnt}" for acc, cnt in sorted(stats["accept_len_hist"].items()))
  return (f"avg_accept_len={stats['avg_accept_len']:.2f} drafts_per_token={stats['drafts_per_token']:.2f} "
          f"accept_len_hist={{{hist_str}}}")

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model", required=True, help="path to gguf model")
  parser.add_argument("--max-context", type=int, default=8192, help="max context length (default: %(default)s)")
  parser.add_argument("--device-map", default=None, help="per-block device placement, same syntax as tinygrad.llm.cli")
  parser.add_argument("--prompt-tokens", type=int, default=64, help="new tokens to prefill and time (default: %(default)s)")
  parser.add_argument("--decode-tokens", type=int, default=64, help="decode steps to time per configuration (default: %(default)s)")
  parser.add_argument("--k", default="1,2,3", help="comma-separated draft lengths to bench (default: %(default)s)")
  parser.add_argument("--fill", type=int, default=0, help="tokens to pre-fill the cache with before timing (default: %(default)s)")
  parser.add_argument("--chunk-size", type=int, default=32, help="prefill chunk width for the timed run (default: %(default)s)")
  args = parser.parse_args()
  ks = [int(x) for x in args.k.split(",") if x]

  st = time.perf_counter()
  with Context(MTP=1):  # must be set before from_gguf -- see MTP's own ContextVar docstring in model.py
    model, _ = Transformer.from_gguf(args.model, args.max_context, device_map=args.device_map)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)

  st = time.perf_counter()
  model.warmup()  # mtp_head is not None -> this also warms speculative_generate's own jit keys (T4.66e)
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)

  fill_tokens = deterministic_prompt(args.fill, offset=0) if args.fill else []
  prompt_tokens = deterministic_prompt(args.prompt_tokens, offset=1)

  plain = bench_plain(model, fill_tokens, prompt_tokens, args.decode_tokens, args.chunk_size)
  print(f"[plain]     fill={args.fill} prompt={args.prompt_tokens} decode={args.decode_tokens} "
        f"prefill={plain['prefill_tok_s']:.2f}tok/s/{plain['prefill_gb_s']:.2f}GB/s "
        f"decode={plain['decode_tok_s']:.2f}tok/s/{plain['decode_gb_s']:.2f}GB/s", flush=True)

  for k in ks:
    spec = bench_spec(model, fill_tokens, prompt_tokens, args.decode_tokens, k, args.chunk_size)
    match = spec["output"] == plain["output"]
    print(f"[spec k={k}]  fill={args.fill} prompt={args.prompt_tokens} decode={args.decode_tokens} "
          f"prefill={spec['prefill_tok_s']:.2f}tok/s/{spec['prefill_gb_s']:.2f}GB/s "
          f"decode={spec['decode_tok_s']:.2f}tok/s/{spec['decode_gb_s']:.2f}GB/s "
          f"{format_stats(spec['stats'])} match_plain_greedy={match}", flush=True)

if __name__ == "__main__": main()
