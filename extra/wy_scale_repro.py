"""T4.73a: parameterized reproduction matrix for bug-1 (the serve-only WY replay NaN-flood -- see
HANDOFF_2026-09-01.md §2). Never a standalone replica loop: every run here goes through the REAL
Transformer.warmup() + Transformer.generate() call sequence tinygrad/llm/serve.py actually uses --
warmup() burns the prefill jit's cnt 0+1 (both temperatures, twice each, via generate([0])), so the
first REAL prompt's prefill call is already a JIT REPLAY, exactly like the hardware trace.

Starting point this supersedes: an earlier tiny CPU/METAL real-generate check (2 GDN blocks, dim 32,
head_dim 8, prompt 20) came back clean -- but generate() caps chunk_size via gdn_chunk_for(device),
which returns 1 on CPU unless GDN_CHUNK overrides it. That check silently ran chunk_size=1 (no chunked
prefill at all). This script's first axis (--chunk, matrix Phase 1) is that same tiny config with
GDN_CHUNK actually in effect -- the untested ingredient.

Axes (one at a time, escalating -- see build_matrix()): chunk width, GDN geometry (tiny vs qwen3.8-27B's
real 48-head/128-dim shape), block count, weight dtype (from_gguf's HALF=1 casts every loaded weight to
float16 post-dequant -- see model.py from_gguf's `state_dict = {k:v.cast('float16') if getenv("HALF",1)...}`),
interleaved attention blocks, and a second differing-length prompt.

Detection: after warmup() + a real prompt's prefill + 3 decode steps (4 pulled tokens total), count NaNs
in every GDN block's recurrent_state/conv_state. Bug-1 floods them (block 0 partially, later blocks fully,
per the hardware trace); a clean run has 0 NaNs everywhere. GDN_SCAN_IMPL=1 (loop) runs as a control at
every config -- bug-1 is WY-only, so a config where the loop ALSO floods is a DIFFERENT bug (flagged, not
claimed as bug-1). gdn_last_scan_impl (model.py, test-introspection global) records whether run_scan's
Python body actually ran during the checked stage -- a JIT replay never re-enters it, so an empty list
there is corroborating evidence for "this was a replay", independent of the NaN count itself.

Run one config:
  DEV=CPU PYTHONPATH=. .venv/bin/python extra/wy_scale_repro.py --chunk 32 --real-geometry --blocks 16 --dtype f16
Run the escalation matrix (see RESULTS.md for the recorded grid):
  DEV=CPU PYTHONPATH=. .venv/bin/python extra/wy_scale_repro.py --matrix
"""
import argparse, os
assert os.environ.get("DEV") == "CPU", (
  "run with DEV=CPU -- this worktree is CPU/NULL-only (task hard rule), and NULL doesn't run real "
  "arithmetic (no NaNs to find). Never DEV=METAL/NV from this worktree.")

from dataclasses import dataclass
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.model import (
  Transformer, TransformerConfig, SSMConfig, GDN_SCAN_LOOP, GDN_SCAN_WY, gdn_last_scan_impl,
)

IMPL_NAME = {GDN_SCAN_LOOP: "loop", GDN_SCAN_WY: "wy"}

# qwen3.8-27B's real GDN geometry == test/unit/test_gdn_scan_parity.py GEOMETRIES["38"], the shape
# bug-1 was hardware-confirmed on (num_v_heads=48, head_k_dim=head_v_dim=128, num_k_heads=16).
REAL_SSM = SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=128 * 48)

@dataclass(frozen=True)
class Config:
  chunk: int = 32              # GDN_CHUNK -- the effective prefill chunk width (serve: 32 or 16)
  heads: int = 4                # tiny-geometry num_v_heads (ignored when real_geometry=True)
  head_dim: int = 8             # tiny-geometry head_k_dim==head_v_dim (ignored when real_geometry=True)
  kv_heads: int = 2             # tiny-geometry num_k_heads (ignored when real_geometry=True)
  blocks: int = 2               # number of GDN blocks (+ interleaved attention, see interleave_attn)
  dim: int = 32                 # residual width feeding the linear projections (not a scan shape)
  dtype: str = "f32"            # "f32" | "f16" -- f16 mimics from_gguf's default HALF=1 weight cast
  weight_scale: float = 0.1     # matches test_gdn_scan_parity.py's make_block (randn * 0.1)
  real_geometry: bool = False   # True: use REAL_SSM, ignore heads/head_dim/kv_heads
  interleave_attn: bool = False # every 4th block is real attention (qwen3.8's pattern), rest GDN
  prompt_len: int = 20
  second_prompt_len: int = 0    # 0 = skip; else a second, unrelated (non-extending) prompt afterward
  seed: int = 0

def ssm_for(cfg: Config) -> SSMConfig:
  if cfg.real_geometry: return REAL_SSM
  assert cfg.heads % cfg.kv_heads == 0, f"{cfg.heads=} must be divisible by {cfg.kv_heads=}"
  return SSMConfig(conv_kernel=4, state_size=cfg.head_dim, group_count=cfg.kv_heads,
                    time_step_rank=cfg.heads, inner_size=cfg.head_dim * cfg.heads)

def build_model(cfg: Config) -> Transformer:
  ssm_layers = tuple(not (cfg.interleave_attn and i % 4 == 3) for i in range(cfg.blocks))
  max_context = cfg.prompt_len + cfg.second_prompt_len + 16
  tconfig = TransformerConfig(num_blocks=cfg.blocks, dim=cfg.dim, hidden_dim=cfg.dim * 2, n_heads=2, n_kv_heads=2,
                               norm_eps=1e-5, vocab_size=32, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8,
                               max_context=max_context, ssm=ssm_for(cfg), ssm_layers=ssm_layers)
  Tensor.manual_seed(cfg.seed)
  model = Transformer(tconfig)
  params = nn.state.get_parameters(model)
  for p in params: p.replace(Tensor.randn(*p.shape) * cfg.weight_scale)
  # pin weights NOW: Tensor's RNG is a lazily-realized global counter (test_gdn_scan_parity.py's
  # make_block) -- leaving these lazy risks a later randn() (e.g. warmup's sampled path) stealing this
  # counter slice, which would make "seed everything" not actually reproducible run-to-run.
  Tensor.realize(*params)
  if cfg.dtype == "f16":
    for p in params: p.replace(p.cast(dtypes.float16).contiguous())
    Tensor.realize(*params)
  return model

def gdn_blocks(model: Transformer) -> list[tuple[int, object]]:
  return [(i, b) for i, b in enumerate(model.blk) if hasattr(b, "recurrent_state")]

def nan_report(model: Transformer) -> dict[int, dict]:
  report = {}
  for i, b in gdn_blocks(model):
    rec, conv = b.recurrent_state.numpy(), b.conv_state.numpy()
    report[i] = {"rec_nan": int(np.isnan(rec).sum()), "rec_n": rec.size,
                 "conv_nan": int(np.isnan(conv).sum()), "conv_n": conv.size}
  return report

def is_flooded(report: dict[int, dict]) -> bool:
  return any(v["rec_nan"] or v["conv_nan"] for v in report.values())

def run_stage(model: Transformer, prompt: list[int]) -> dict:
  gdn_last_scan_impl.clear()
  toks = [t for _, t in zip(range(4), model.generate(prompt, temperature=0.0))]  # prefill output + 3 decode steps
  return {"tokens": toks, "nan": nan_report(model), "python_ran": bool(gdn_last_scan_impl)}

def run_config(cfg: Config, impl: int) -> dict:
  model = build_model(cfg)
  with Context(GDN_CHUNK=cfg.chunk, GDN_SCAN_IMPL=impl):
    model.warmup()
    prompt1 = [(i % 30) + 1 for i in range(cfg.prompt_len)]
    result = {"prompt1": run_stage(model, prompt1)}
    if cfg.second_prompt_len:
      prompt2 = [(i % 30) + 2 for i in range(cfg.second_prompt_len)]  # unrelated -- a fresh, non-extending request
      result["prompt2"] = run_stage(model, prompt2)
  return result

def describe(cfg: Config) -> str:
  geom = "real38(h48,d128,kv16)" if cfg.real_geometry else f"tiny(h{cfg.heads},d{cfg.head_dim},kv{cfg.kv_heads})"
  extra = (["attn/4"] if cfg.interleave_attn else []) + ([f"+p2={cfg.second_prompt_len}"] if cfg.second_prompt_len else [])
  return (f"chunk={cfg.chunk} geom={geom} blocks={cfg.blocks} dtype={cfg.dtype} scale={cfg.weight_scale} "
          f"prompt={cfg.prompt_len}" + "".join(f" {e}" for e in extra))

def format_result(impl: int, result: dict) -> str:
  parts = []
  for stage in ("prompt1", "prompt2"):
    if stage not in result: continue
    r = result[stage]
    worst = max((max(v["rec_nan"], v["conv_nan"]) for v in r["nan"].values()), default=0)
    tag = "NAN-FLOOD" if is_flooded(r["nan"]) else "PASS"
    parts.append(f"{stage}={tag}(tok={r['tokens']} worst_nan={worst} python_ran={r['python_ran']})")
  return f"{IMPL_NAME[impl]}: " + " ".join(parts)

def run_and_print(cfg: Config, label: str = "") -> dict[int, dict]:
  results = {impl: run_config(cfg, impl) for impl in (GDN_SCAN_LOOP, GDN_SCAN_WY)}
  wy_flooded = any(is_flooded(results[GDN_SCAN_WY][s]["nan"]) for s in results[GDN_SCAN_WY])
  loop_flooded = any(is_flooded(results[GDN_SCAN_LOOP][s]["nan"]) for s in results[GDN_SCAN_LOOP])
  verdict = "BOTH-IMPLS-BROKEN(not bug-1)" if (wy_flooded and loop_flooded) else \
            "REPRO(wy-only)" if wy_flooded else "clean"
  print(f"[{label or describe(cfg)}] {describe(cfg)} :: {verdict}")
  print(f"    {format_result(GDN_SCAN_LOOP, results[GDN_SCAN_LOOP])}")
  print(f"    {format_result(GDN_SCAN_WY, results[GDN_SCAN_WY])}")
  return results

def build_matrix() -> list[tuple[str, Config]]:
  m = []
  # Phase 1: chunk width alone, tiny geometry, 2 blocks, f32 (the untested ingredient vs. the starting point)
  for chunk in (8, 16, 32): m.append((f"P1-chunk{chunk}", Config(chunk=chunk)))
  # Phase 2: real GDN geometry (qwen3.8-27B shape), 2 blocks, f32 -- serve uses chunk 32 or 16, both implicated
  for chunk in (16, 32): m.append((f"P2-real-chunk{chunk}", Config(chunk=chunk, real_geometry=True)))
  # Phase 3: block count, real geometry, chunk=32 -- capped at 8 to keep every matrix row under the 60s/config
  # budget (measured ~5s/block here: 2 blocks ~10s, 8 blocks ~45s, 16 blocks ~84s/impl -- 16 was spot-checked
  # manually instead, see RESULTS.md, since it alone would blow the budget for a script that's re-run whole)
  m.append(("P3-real-blocks8", Config(chunk=32, real_geometry=True, blocks=8)))
  # Phase 4: weight dtype f16 (from_gguf's default HALF=1 cast), real geometry, 8 blocks, chunk=32
  m.append(("P4-real-8blk-f16", Config(chunk=32, real_geometry=True, blocks=8, dtype="f16")))
  # Phase 5: interleaved attention (qwen3.8's every-4th pattern) on top of phase 4's config
  m.append(("P5-real-8blk-f16-attn", Config(chunk=32, real_geometry=True, blocks=8, dtype="f16", interleave_attn=True)))
  # Phase 6: second, differing-length prompt (exact serve prompt/warmup pattern) on top of phase 4's config
  m.append(("P6-real-8blk-f16-p2", Config(chunk=32, real_geometry=True, blocks=8, dtype="f16", second_prompt_len=13)))
  return m

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--matrix", action="store_true", help="run the built-in escalation matrix (ignores other axis flags)")
  ap.add_argument("--chunk", type=int, default=32)
  ap.add_argument("--heads", type=int, default=4)
  ap.add_argument("--head-dim", type=int, default=8)
  ap.add_argument("--kv-heads", type=int, default=2)
  ap.add_argument("--blocks", type=int, default=2)
  ap.add_argument("--dim", type=int, default=32)
  ap.add_argument("--dtype", choices=("f32", "f16"), default="f32")
  ap.add_argument("--weight-scale", type=float, default=0.1)
  ap.add_argument("--real-geometry", action="store_true")
  ap.add_argument("--interleave-attn", action="store_true")
  ap.add_argument("--prompt-len", type=int, default=20)
  ap.add_argument("--second-prompt-len", type=int, default=0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--label", default="")
  args = ap.parse_args()

  if args.matrix:
    for label, cfg in build_matrix(): run_and_print(cfg, label)
    return
  cfg = Config(chunk=args.chunk, heads=args.heads, head_dim=args.head_dim, kv_heads=args.kv_heads,
               blocks=args.blocks, dim=args.dim, dtype=args.dtype, weight_scale=args.weight_scale,
               real_geometry=args.real_geometry, interleave_attn=args.interleave_attn,
               prompt_len=args.prompt_len, second_prompt_len=args.second_prompt_len, seed=args.seed)
  run_and_print(cfg, args.label)

if __name__ == "__main__":
  main()
