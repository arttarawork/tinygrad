"""T4.98e: teacher-forced top-1 flip rate between two runs of a model -- the quality metric for judging a
quant/kernel change (WY vs loop scan, a new BEAM winner, KV_INT8, ...) against a reference run.

`dump` greedily decodes --tokens new tokens per prompt (optionally teacher-forced onto a reference dump's own
sequence, so positions line up instead of compounding divergence) and records each step's argmax token id and
top-2 logit gap into an .npz. `compare` diffs two dumps positionally: how often did the candidate's argmax
differ from the reference's, and how confident (top-2 gap) was it when it did.

Usage:
  PYTHONPATH=. DEV=CPU python extra/flip_rate.py dump --model ref.gguf --prompts prompts.jsonl --out ref.npz --tokens 64
  PYTHONPATH=. DEV=CPU python extra/flip_rate.py dump --model other.gguf --prompts prompts.jsonl --out other.npz --tokens 64 --force ref.npz
  PYTHONPATH=. DEV=CPU python extra/flip_rate.py compare ref.npz other.npz

prompts.jsonl: one {"messages": [{"role": "user", "content": "..."}, ...]} per line.
"""
from __future__ import annotations
import argparse, json, time
from typing import cast, TYPE_CHECKING
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
from tinygrad.llm.cli import SimpleTokenizer, FallbackTemplate
if TYPE_CHECKING: import jinja2

# ******** core: model + already-tokenized ids only, no GGUF/tokenizer -- what the tests drive directly ********

def reset_recurrent_state(model:Transformer) -> None:
  """Zero every GDN block's O(1) recurrent accumulators so the next prompt starts clean. Unlike the
  position-indexed attention/MLA KV caches (safe to just start overwriting from start_pos=0 -- causal
  masking means a fresh prompt only ever reads positions it just wrote itself, see
  Transformer.get_start_pos's own comment on this), GatedDeltaNetBlock's conv_state/recurrent_state persist
  regardless of start_pos and would otherwise leak the previous prompt's state into this one. No-op on an
  attention-only model, or before any GDN block has run its first forward (nothing allocated yet to reset)."""
  for b in model.blk:
    if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "recurrent_state"):
      b.conv_state.assign(Tensor.zeros_like(b.conv_state)).realize()
      b.recurrent_state.assign(Tensor.zeros_like(b.recurrent_state)).realize()

def _spec_step(model:Transformer, tokens:Tensor, sp:UOp, n:int) -> Tensor:
  """One Transformer.__call__(spec=True) step -- returns the (1, vocab) logits at the n-th (host-known,
  1-indexed) real position. T4.98e2: this goes through __call__ (JIT'd, keyed on (is_prefill, greedy,
  chunk_size, spec) -- see __call__'s own comment), not forward() directly, so a chunk/decode shape seen
  before replays a captured graph instead of retracing forward() from scratch every step (measured ~5x
  slower that way -- the whole point of this rewrite). `n` is passed explicitly rather than read off
  tokens' own shape: a replayed JIT output's lazy shape can still report an EARLIER call's bound width for
  a negative index (model.py's T4.66h comment covers the exact same hazard in speculative_generate's own
  prefill-tail read), so this always indexes by the width THIS call actually asked for, like T4.66h's fix."""
  full, _ = cast(tuple[Tensor, Tensor], model(tokens, sp, None, spec=True))
  return full[:, n - 1, :]

def _top1_and_gap(logits:Tensor) -> tuple[int, float]:
  vals = logits[0].numpy()
  top1 = int(vals.argmax())
  gap = float(vals[top1] - np.partition(vals, -2)[-2]) if len(vals) > 1 else float("inf")
  return top1, gap

def teacher_forced_run(model:Transformer, prompt_ids:list[int], n_tokens:int, chunk_size:int=32,
                        force_ids:list[int]|None=None) -> tuple[list[int], list[float]]:
  """Prefill prompt_ids in chunk_size-wide chunks, then take n_tokens greedy decode steps, both through
  Transformer.__call__ with the same bound-Variable slicing generate() uses for its own prefill/decode loop
  (T4.98e2 -- see _spec_step): one Tensor of max_context tokens, v_start_pos/v_toks bound per step. This
  reuses generate()'s served JIT graphs (captured once per (prefill, decode) family across the whole run,
  replayed every later step/prompt) instead of retracing forward() eagerly every step. Each step feeds back
  either this model's own just-computed argmax (force_ids=None, free-running) or force_ids[i] (teacher
  forcing onto a reference run's sequence -- module docstring). Returns (argmax id, top-2 logit gap) per step."""
  assert prompt_ids, "prompt_ids must be non-empty"
  assert force_ids is None or len(force_ids) == n_tokens, f"force_ids must have {n_tokens} entries, got {len(force_ids)}"
  reset_recurrent_state(model)
  dev = model.blk[0].device
  v_start_pos = UOp.variable("start_pos", 0, model.max_context - 1)
  v_toks = UOp.variable("toks", 1, chunk_size)
  t = Tensor(prompt_ids + [0] * (model.max_context - len(prompt_ids)), dtype="int32", device=dev).reshape(1, model.max_context)
  pos = 0
  logits: Tensor|None = None
  while pos < len(prompt_ids):
    n = min(chunk_size, len(prompt_ids) - pos)
    sp, nt = v_start_pos.bind(pos), v_toks.bind(n)
    logits = _spec_step(model, t[:, sp:sp + nt], sp, n)
    pos += n
  argmax_ids: list[int] = []
  gaps: list[float] = []
  for i in range(n_tokens):
    idx, gap = _top1_and_gap(cast(Tensor, logits))
    argmax_ids.append(idx)
    gaps.append(gap)
    if i == n_tokens - 1: break
    tok = force_ids[i] if force_ids is not None else idx
    logits = _spec_step(model, Tensor([[tok]], dtype="int32", device=dev), v_start_pos.bind(pos), 1)
    pos += 1
  return argmax_ids, gaps

def run_dump(model:Transformer, prompts_ids:list[list[int]], n_tokens:int, chunk_size:int=32,
             force_ids_list:list[list[int]]|None=None) -> dict[str, list]:
  """teacher_forced_run over each already-tokenized prompt. Saves nothing to disk -- the CLI's dump command
  below handles rendering messages -> ids (needs a tokenizer + chat template) and np.savez'ing the result;
  this is the pure part tests drive directly with an in-memory model, skipping GGUF loading entirely."""
  assert force_ids_list is None or len(force_ids_list) == len(prompts_ids)
  argmax_ids, gaps = [], []
  for i, ids in enumerate(prompts_ids):
    a, g = teacher_forced_run(model, ids, n_tokens, chunk_size, force_ids_list[i] if force_ids_list is not None else None)
    argmax_ids.append(a)
    gaps.append(g)
  return {"argmax_ids": argmax_ids, "top2_gap": gaps}

def compare_runs(ref_ids:np.ndarray, other_ids:np.ndarray, other_gap:np.ndarray) -> tuple[int, int, float]:
  """Positional flip count between two dumps' argmax arrays (same shape: prompts x tokens). other_gap is the
  CANDIDATE run's own top-2 gap, read only at the positions where it flipped relative to ref: a small gap there
  means the flip was a near-tie (noise), a large one means the candidate confidently picked a different token."""
  assert ref_ids.shape == other_ids.shape, f"shape mismatch: ref {ref_ids.shape} vs other {other_ids.shape}"
  flip_mask = ref_ids != other_ids
  total, flips = int(ref_ids.size), int(flip_mask.sum())
  mean_gap = float(other_gap[flip_mask].mean()) if flips else 0.0
  return total, flips, mean_gap

# ******** CLI: GGUF/tokenizer loading, chat-template rendering, npz I/O ********

def build_template(tok:SimpleTokenizer, kv:dict) -> jinja2.Template|FallbackTemplate:
  """Mirrors tinygrad/llm/cli.py main()'s template selection exactly (jinja2 if available and the GGUF carries
  one, else FallbackTemplate's plain role-header concatenation) -- duplicated here rather than imported since
  cli.py doesn't factor this out into a function, and cli.py is library code this task must not touch."""
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if (ct := kv.get('tokenizer.chat_template')) is not None:
    try:
      import jinja2
      env = jinja2.Environment()
      env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
      env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
      env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
      env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
      env.globals['eos_token'] = tok.decode([tok.eos_id])
      template = env.from_string(ct)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")
  return template

def load_prompts(path:str) -> list[list[dict]]:
  with open(path) as f: return [json.loads(line)["messages"] for line in f if line.strip()]

def cmd_dump(args:argparse.Namespace) -> None:
  model, kv = Transformer.from_gguf(args.model, args.max_context, device_map=args.device_map)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  template = build_template(tok, kv)
  prompts_ids = [tok.encode(template.render(messages=m, add_generation_prompt=True)) for m in load_prompts(args.prompts)]
  force_ids_list = None
  if args.force:
    ref = np.load(args.force)
    assert ref["argmax_ids"].shape == (len(prompts_ids), args.tokens), \
      f"--force {args.force} has shape {ref['argmax_ids'].shape}, expected ({len(prompts_ids)}, {args.tokens}) " \
      "-- dump it from the same --prompts file with the same --tokens"
    force_ids_list = ref["argmax_ids"].tolist()
  result = run_dump(model, prompts_ids, args.tokens, args.chunk_size, force_ids_list)
  np.savez(args.out, argmax_ids=np.array(result["argmax_ids"], dtype=np.int64), top2_gap=np.array(result["top2_gap"], dtype=np.float32))
  print(f"wrote {args.out}: {len(prompts_ids)} prompts x {args.tokens} tokens")

def cmd_compare(args:argparse.Namespace) -> None:
  ref, other = np.load(args.ref), np.load(args.other)
  total, flips, mean_gap = compare_runs(ref["argmax_ids"], other["argmax_ids"], other["top2_gap"])
  print(f"positions={total} flips={flips} flip_rate={flips / total:.4f} mean_top2_gap_at_flips={mean_gap:.4f}")

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)

  d = sub.add_parser("dump", help="greedy (or teacher-forced) decode over a prompt set, recording argmax id + top-2 gap per step")
  d.add_argument("--model", required=True, help="path to gguf model")
  d.add_argument("--prompts", required=True, help='jsonl file, one {"messages": [...]} per line')
  d.add_argument("--out", required=True, help="output .npz path")
  d.add_argument("--tokens", type=int, default=64, help="decode steps per prompt (default: %(default)s)")
  d.add_argument("--max-context", type=int, default=8192, help="max context length (default: %(default)s)")
  d.add_argument("--chunk-size", type=int, default=32, help="prefill chunk width (default: %(default)s)")
  d.add_argument("--device-map", default=None, help="per-block device placement, same syntax as tinygrad.llm.cli")
  d.add_argument("--force", default=None, help="reference .npz (from a prior dump) to teacher-force this run onto")

  c = sub.add_parser("compare", help="positional flip rate between two dumps")
  c.add_argument("ref")
  c.add_argument("other")

  args = parser.parse_args()
  (cmd_dump if args.cmd == "dump" else cmd_compare)(args)

if __name__ == "__main__": main()
