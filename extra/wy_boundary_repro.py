"""T4.76 deliverable B: standalone repro of generate()'s EXACT jit-capture STRUCTURE, not the scan math.

Plain math parity for gdn_scan_wy already passes on CPU (test/unit/test_gdn_scan_parity.py) at every chunk
size and at this same real head geometry. What has never been isolated is serve()'s exact CAPTURE SEQUENCE:
a multi-chunk WY prefill under one whole-model TinyJit, immediately followed by single-token decode steps
under a SEPARATE whole-model TinyJit (the T4.69b gate keeps decode on the loop impl regardless -- T_pad==1
there always). This script drives a tiny hybrid stack -- the REAL GatedDeltaNetBlock (48 v-heads, the real
qwen3.8-27B head count -- see REPORT.md "THE BUG") plus one REAL attention TransformerBlock, small dims
otherwise -- through that exact sequence, replicating Transformer.__call__'s 4-tuple jit key and
Transformer.generate()'s start_pos/toks Variable-binding discipline (tinygrad/llm/model.py) by hand, since
generate() itself is bound to the full Transformer class (real GGUF-shaped token_embd/output, no seam to
swap in a miniature stack -- see REPORT.md's "drive via the closest public path" note). Compares full
logits + per-block state fingerprints, step by step, against a no-jit, GDN_SCAN_IMPL=1 (loop) reference run
built from the SAME two block objects' class but instantiated fresh (identical seed -> identical weights,
same trick test_gdn_scan_parity.py's make_block relies on) so no persistent state leaks between the two runs.

Expected result on CPU: PASS (the bug is GPU-only -- see REPORT.md's ranked mechanisms; all three point at
device-async or hardware-graph-capture specifics that are structurally absent on CPU). PASS here is not a
clean bill of health for the real bug -- this script's job is to be the smallest thing that CAN fail on
METAL/NV, not to already fail on CPU.

Chunk sizes: the jit/wy path chunks its prefill at --chunk-size (default 32, matching production's GDN_CHUNK
-- WY's chunkwise form is a log-depth doubling, not a T-deep unrolled chain, so this is not the "deep chain"
class the GATES warning below is about). The no-jit reference chunks its OWN (mathematically chunk-invariant,
per test_gdn_scan_parity.py) prefill at a fixed, small 8 instead, specifically to avoid ever compiling a
grouped (48 heads -> GDN_HEAD_GROUPS auto-splits to 2) LOOP-impl graph deeper than the {1,2,4,8} the repo's
own CI-safety note (tinygrad/CLAUDE.md gates section) calls out as safe -- LOOP's T_pad-deep unrolled chain,
not WY's, is what "grouped-c32 ... deep chains natively crash CI workers" refers to (T4.55/T4.68 history).

Tolerance/seed note (found while building this on CPU, see REPORT.md's honesty-over-confidence section): an
UNTRAINED, randomly-weighted tiny model is occasionally locally chaotic -- a ~1e-6 eager-vs-JIT float
non-associativity difference (same character as test_gdn_scan_parity.py's own documented chunk-boundary
noise) in one decode step's state can amplify, through this toy model's own unnormalized random weights,
into a visibly different next-step argmax on SOME seeds. Confirmed unrelated to WY specifically (reproduces
identically with GDN_SCAN_IMPL forced to LOOP on BOTH paths) and unrelated to TinyJit's trace/capture/replay
phase (reproduces identically after explicitly warming both jit keys to cnt>=2 first, i.e. with every
compared call a pure replay) -- it is a property of THIS geometry's random weights being locally
ill-conditioned at a handful of (seed, step) points, not of the capture structure under test. --seed 5 is
verified stable at the tolerance below across this script's DEFAULT geometry only (chunks/decode-steps vary
freely, still green -- but e.g. --dim 32 reopens a divergence at this same seed: pick a different --seed if
a non-default geometry ever needs one). A real hardware run that hits a FIRST-DIVERGENCE should be checked
against this signature before concluding anything -- the real bug's signature (REPORT.md) is a
PROMPT-INDEPENDENT, CONFIG-INDEPENDENT constant garbage id, not a seed-sensitive
near-tie flip with both sides' values small and plausible.

Invocations for Artur's own later hardware runs (this task only ran the CPU one below):
  DEV=METAL PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device METAL
  DEV=NV    PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device NV
  # more prefill chunks, still the default (verified-stable) geometry otherwise:
  DEV=NV    PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py --device NV --chunks 3
  # a wider --dim needs its own --seed search first (see the tolerance/seed note above) -- don't assume 5 still works
This task's own CPU verification:
  PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_boundary_repro.py
"""
import argparse, random, sys
from typing import Callable, cast
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.helpers import Context
from tinygrad.uop.ops import resolve
from tinygrad.nn.state import get_parameters
from tinygrad.llm.model import TransformerConfig, SSMConfig, GatedDeltaNetBlock, TransformerBlock, Linear, GDN_SCAN_LOOP, GDN_SCAN_WY

REF_LOOP_CHUNK = 8  # safe LOOP-impl prefill chunk width for the reference run only -- see module docstring

def build_config(args:argparse.Namespace, max_context:int) -> tuple[TransformerConfig, SSMConfig]:
  ssm = SSMConfig(conv_kernel=4, state_size=args.head_dim, group_count=args.heads, time_step_rank=args.heads,
                   inner_size=args.head_dim * args.heads)
  cfg = TransformerConfig(num_blocks=2, dim=args.dim, hidden_dim=args.dim * 2, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                           vocab_size=args.vocab, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8,
                           max_context=max_context, ssm_layers=(True, False))
  return cfg, ssm

def build_stack(cfg:TransformerConfig, ssm:SSMConfig, seed:int, device:str) -> list[GatedDeltaNetBlock|TransformerBlock]:
  """Real GatedDeltaNetBlock + real TransformerBlock, randomly weighted (GatedDeltaNetBlock zero-inits several
  GGUF-loaded-only fields -- ssm_conv1d/ssm_dt/ssm_a -- so leaving them at 0 would make q/k/v trivially zero
  and hide any real WY-vs-loop divergence; get_parameters walks those the same as every Linear, so one
  blanket randomize-and-replace loop covers everything, same idiom test_gdn_scan_parity.py's make_block uses).
  Two calls with the SAME seed give BYTE-IDENTICAL weights (manual_seed fully resets Tensor's RNG counter),
  which is how this script gets a reference stack and a jit stack that start from the same weights without
  needing a state_dict clone.

  ssm_a's SIGN is then pinned negative (decay = exp(softplus(..)*ssm_a) <= 1): this script runs a real prompt
  plus several decode steps, much longer than test_gdn_scan_parity.py's few-token checks, so a channel left
  free to randomly land with decay > 1 amplifies WY-vs-loop's ordinary ~1e-4 rounding difference (see
  RTOL/ATOL below) exponentially over steps -- confirmed empirically while building this script: with ssm_a
  unpinned, decode#2 diverges by ~0.07 (a real-looking divergence that vanishes once ssm_a is pinned), purely
  from float non-associativity blowing up through an unstable recurrence, not from the capture structure."""
  blocks: list[GatedDeltaNetBlock|TransformerBlock] = [GatedDeltaNetBlock(cfg, ssm), TransformerBlock(cfg)]
  Tensor.manual_seed(seed)
  params = get_parameters(blocks)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  for block in blocks:
    if isinstance(block, GatedDeltaNetBlock): block.ssm_a.replace(-block.ssm_a.abs())
  for p in params: p.to_(device)
  Tensor.realize(*params)
  return blocks

def build_head(cfg:TransformerConfig, seed:int, device:str) -> tuple[nn.Embedding, nn.RMSNorm, Linear]:
  """Stateless (no cross-call buffers), so unlike build_stack this is built ONCE and shared by both the
  reference and jit/wy runs -- no seed-matching needed, they're the literal same objects."""
  embd, norm, out = nn.Embedding(cfg.vocab_size, cfg.dim), nn.RMSNorm(cfg.dim, cfg.norm_eps), Linear(cfg.dim, cfg.vocab_size, bias=False)
  Tensor.manual_seed(seed)
  params = get_parameters([embd, norm, out])
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  for p in params: p.to_(device)
  Tensor.realize(*params)
  return embd, norm, out

def forward(blocks:list, embd:nn.Embedding, norm:nn.RMSNorm, out_proj:Linear, tokens:Tensor, start_pos) -> tuple[Tensor, Tensor]:
  """Mirrors Transformer.forward (tinygrad/llm/model.py) exactly, except it always returns (argmax, FULL
  logits) instead of only the argmax for greedy -- forward()'s spec=True branch already proves a 2-tuple of
  Tensors round-trips through TinyJit correctly, so this reuses that same return shape rather than needing
  WY_TRACE's separate persistent-buffer trick (deliverable A): here the logits ARE the traced function's own
  output, so a JIT REPLAY naturally recomputes them fresh every call with no extra plumbing."""
  x = embd(tokens.to(embd.weight.device)).float()
  for block in blocks: x = block(x.to(block.device), start_pos)
  logits = out_proj(norm(x[:, -1:]))[:, -1, :]
  return logits.argmax(-1, keepdim=True), logits

def fingerprint_tensors(block) -> dict[str, Tensor]:
  if isinstance(block, GatedDeltaNetBlock): return {"recurrent_state": block.recurrent_state, "conv_state": block.conv_state}
  if isinstance(block, TransformerBlock): return {"cache_kv": block.cache_kv}
  return {}

def first_mismatch(a:list[float], b:list[float], rtol:float, atol:float) -> int|None:
  for i, (x, y) in enumerate(zip(a, b)):
    xn, yn = x != x, y != y  # NaN != NaN
    if xn or yn:
      if xn != yn: return i  # one side NaN, the other not: a real divergence
      continue                # both NaN: agree qualitatively, not a NEW divergence here
    if abs(x - y) > atol + rtol * abs(y): return i
  return None

def compare_step(step:str, ref_blocks:list, jit_blocks:list, ref_out:tuple[Tensor, Tensor], jit_out:tuple[Tensor, Tensor],
                  rtol:float, atol:float) -> str|None:
  ref_argmax, ref_logits_t = ref_out
  jit_argmax, jit_logits_t = jit_out
  ref_logits, jit_logits = cast(list[float], ref_logits_t.flatten().tolist()), cast(list[float], jit_logits_t.flatten().tolist())
  if (i := first_mismatch(ref_logits, jit_logits, rtol, atol)) is not None:
    return f"step={step} tensor=logits[{i}] ref={ref_logits[i]!r} got={jit_logits[i]!r}"
  ref_id, jit_id = int(ref_argmax.item()), int(jit_argmax.item())
  if ref_id != jit_id: return f"step={step} tensor=argmax ref={ref_id} got={jit_id}"
  for bi, (rb, jb) in enumerate(zip(ref_blocks, jit_blocks)):
    jt_map = fingerprint_tensors(jb)
    for name, rt in fingerprint_tensors(rb).items():
      rl, jl = cast(list[float], rt.flatten().tolist()), cast(list[float], jt_map[name].flatten().tolist())
      if (i := first_mismatch(rl, jl, rtol, atol)) is not None:
        return f"step={step} block={bi} tensor={name}[{i}] ref={rl[i]!r} got={jl[i]!r}"
  return None

def run_reference(blocks:list, embd, norm, out_proj, tokens:list[int], n_decode:int) -> list[tuple[Tensor, Tensor]]:
  """No JIT anywhere, GDN_SCAN_IMPL=1 (loop) throughout -- the "known good" this script's jit/wy run is
  checked against. Chunks its own prefill at REF_LOOP_CHUNK (mathematically a no-op for the loop impl -- see
  module docstring -- purely to keep the unrolled T_pad shallow)."""
  steps = []
  with Context(GDN_SCAN_IMPL=GDN_SCAN_LOOP):
    dev = blocks[0].device
    t = Tensor([tokens], dtype="int32", device=dev)
    pos = 0
    argmax_id = out = None
    while pos < len(tokens):
      size = min(REF_LOOP_CHUNK, len(tokens) - pos)
      argmax_id, out = forward(blocks, embd, norm, out_proj, t[:, pos:pos + size], pos)
      Tensor.realize(argmax_id, out)
      pos += size
    assert argmax_id is not None and out is not None
    steps.append((argmax_id, out))
    for _ in range(n_decode):
      argmax_id, out = forward(blocks, embd, norm, out_proj, argmax_id, pos)
      Tensor.realize(argmax_id, out)
      steps.append((argmax_id, out))
      pos += 1
  return steps

def run_jit(blocks:list, embd, norm, out_proj, tokens:list[int], chunk_size:int, max_context:int,
            n_decode:int) -> list[tuple[Tensor, Tensor]]:
  """Replicates Transformer.__call__ + Transformer.generate()'s exact jit discipline (tinygrad/llm/model.py):
  the same 4-tuple jit key (is_prefill, greedy, chunk_size, spec) keyed dict of TinyJit(forward), the same
  v_start_pos/v_toks bound UOp Variables, the same `t[:, sp:sp+nt] if is_prefill_call else out` chaining of a
  decode step's input from the previous step's own (already-realized) output tensor. GDN_SCAN_IMPL=2 (WY)
  active for the whole run: the T4.69b gate (T_pad>1) keeps decode on the loop impl regardless, exactly as
  in production -- see run_scan in model.py."""
  jit_cache: dict[tuple[bool, bool, int|None, bool], Callable[..., tuple[Tensor, Tensor]]] = {}
  def call(tok:Tensor, start_pos) -> tuple[Tensor, Tensor]:
    is_prefill = bool(resolve(tok.shape[1] != 1))
    cs = next((cast(int, v.vmax) for v in tok.uop.variables() if v.expr == "toks"), None) if is_prefill else None
    key = (is_prefill, True, cs, False)
    if key not in jit_cache: jit_cache[key] = TinyJit(lambda tok, sp: forward(blocks, embd, norm, out_proj, tok, sp))
    return jit_cache[key](tok.contiguous(), start_pos)

  v_start_pos = UOp.variable("start_pos", 0, max_context - 1)
  v_toks = UOp.variable("toks", 1, chunk_size)
  dev = blocks[0].device
  t = Tensor(tokens + [0] * (max_context - len(tokens)), dtype="int32", device=dev).reshape(1, max_context)
  prompt_len = len(tokens)
  start_pos, virtual_len = 0, prompt_len
  out: Tensor|None = None
  steps: list[tuple[Tensor, Tensor]] = []
  with Context(GDN_SCAN_IMPL=GDN_SCAN_WY):
    while len(steps) < 1 + n_decode:
      n_toks = min(chunk_size, virtual_len - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      is_prefill_call = start_pos < prompt_len or out is None
      argmax_id, logits = call(t[:, sp:sp + nt] if is_prefill_call else cast(Tensor, out), sp)
      Tensor.realize(argmax_id, logits)
      start_pos += n_toks
      out = argmax_id
      if start_pos < virtual_len: continue  # more prompt chunks still to consume
      steps.append((argmax_id, logits))
      virtual_len += 1
  return steps

def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--device", default="CPU", help="tinygrad device string (CPU, METAL, NV, ...) -- CPU only was verified by T4.76")
  p.add_argument("--dim", type=int, default=16, help="residual stream width")
  p.add_argument("--heads", type=int, default=48, help="GDN num_v_heads == num_k_heads (48 = the real qwen3.8-27B count)")
  p.add_argument("--head-dim", type=int, default=8, help="GDN head_k_dim == head_v_dim (kept small; heads is the real dim)")
  p.add_argument("--vocab", type=int, default=32, help="vocab size")
  p.add_argument("--chunks", type=int, default=2, help="prefill chunks for the jit/wy path (prompt_len = chunks*chunk_size - 3)")
  p.add_argument("--chunk-size", type=int, default=32, help="jit/wy prefill chunk width -- production's GDN_CHUNK")
  p.add_argument("--decode-steps", type=int, default=3, help="decode steps to compare (matches deliverable A's WY_TRACE)")
  p.add_argument("--seed", type=int, default=5, help="verified-stable default -- see the tolerance note below")
  p.add_argument("--rtol", type=float, default=1e-3, help="test_attention.py's own TestGatedDeltaNetBlock convention")
  p.add_argument("--atol", type=float, default=1e-3)
  args = p.parse_args()

  prompt_len = max(1, args.chunks) * args.chunk_size - 3
  max_context = prompt_len + args.decode_steps + 8
  cfg, ssm = build_config(args, max_context)

  ref_blocks = build_stack(cfg, ssm, args.seed, args.device)
  jit_blocks = build_stack(cfg, ssm, args.seed, args.device)  # same seed -> identical weights, see build_stack
  embd, norm, out_proj = build_head(cfg, args.seed, args.device)

  random.seed(args.seed)
  tokens = [random.randrange(args.vocab) for _ in range(prompt_len)]

  ref_steps = run_reference(ref_blocks, embd, norm, out_proj, tokens, args.decode_steps)
  jit_steps = run_jit(jit_blocks, embd, norm, out_proj, tokens, args.chunk_size, max_context, args.decode_steps)
  assert len(ref_steps) == len(jit_steps) == 1 + args.decode_steps

  for i, (ref_out, jit_out) in enumerate(zip(ref_steps, jit_steps)):
    step_name = "prefill(last)" if i == 0 else f"decode#{i}"
    if (divergence := compare_step(step_name, ref_blocks, jit_blocks, ref_out, jit_out, args.rtol, args.atol)) is not None:
      print(f"FIRST-DIVERGENCE {divergence}")
      return 1
  print(f"PASS ({1 + args.decode_steps} steps, device={args.device}, heads={args.heads}, head_dim={args.head_dim}, "
        f"chunk_size={args.chunk_size}, prompt_len={prompt_len})")
  return 0

if __name__ == "__main__":
  sys.exit(main())
