"""T4.69a evidence: does gdn_scan_wy (model.py) lower the chunk-32 GatedDeltaNet scan to fewer, matmul-shaped
kernels instead of GatedDeltaNetBlock._attention's else-branch per-token loop (one T_pad-deep unrolled chain
fused into a single huge kernel)?

Builds the else-branch chunk-32 scan graph at the REAL qwen3.8-27B geometry gdn_headgroup_evidence.py (T4.62)
used -- num_v_heads=48, head_v_dim=128, head_k_dim=128 -- once under GDN_SCAN_IMPL=1 (today's per-token loop)
and once under GDN_SCAN_IMPL=2 (gdn_scan_wy), both at GDN_HEAD_GROUPS=1 (pinned, not auto: num_v_heads=48
would otherwise auto-split into G=2 head groups -- see gdn_head_groups_for -- which would confound a
loop-vs-wy comparison with a head-splitting effect this script isn't measuring). For every kernel in each
schedule, prints its linearized UOp count via the exact mechanism codegen/opt/search.py's _try_compile
checks against BEAM_UOPS_MAX (default 3000): to_program(ast, renderer) then len(prg.src[1].src) -- same
mechanism gdn_headgroup_evidence.py uses, for direct comparability with that T4.62 evidence.

Run with: DEV=NULL PYTHONPATH=. <venv>/bin/python extra/gdn_wy_evidence.py
(DEV=NULL: no real device is opened, per the hard rule against touching METAL/NV. BEAM/JITBEAM are asserted
unset below, so no BEAM search runs either -- apply_opts takes its default, non-searched path, same caveat
gdn_headgroup_evidence.py documents: this does NOT reproduce a real GPU's BEAM-searched kernel choices, only
this sandbox's default lowering.)
"""
import os
assert os.environ.get("DEV") == "NULL", "run with DEV=NULL -- see module docstring"
assert not os.environ.get("BEAM") and not os.environ.get("JITBEAM"), "BEAM/JITBEAM must stay unset (T4.62 hard rule, kept here)"

from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Context
from tinygrad.codegen import to_program
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig, GDN_SCAN_LOOP, GDN_SCAN_WY

# the three "REAL dims" gdn_headgroup_evidence.py's T4.62 spec names for qwen3.8-27B
NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM = 48, 128, 128
CHUNK = 32  # GDN_CHUNK's measured-good chunk width (model.py) -- this is "the chunk-32 scan graph"
NUM_K_HEADS = 8  # arbitrary (any divisor of NUM_V_HEADS): q/k are repeat-expanded to NUM_V_HEADS before the
                 # scan, same as gdn_headgroup_evidence.py -- doesn't affect either impl's UOp count
BEAM_UOPS_MAX = 3000  # tinygrad/codegen/opt/search.py: getenv("BEAM_UOPS_MAX", 3000)

def make_block() -> GatedDeltaNetBlock:
  ssm = SSMConfig(conv_kernel=4, state_size=HEAD_K_DIM, group_count=NUM_K_HEADS, time_step_rank=NUM_V_HEADS,
                   inner_size=HEAD_V_DIM * NUM_V_HEADS)
  config = TransformerConfig(num_blocks=1, dim=128, hidden_dim=256, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                              vocab_size=32, head_dim=128, rope_theta=10000.0, rope_dim=128, v_head_dim=128,
                              max_context=CHUNK, ssm_layers=(True,), ssm=ssm)
  return GatedDeltaNetBlock(config, ssm)

def schedule_kernels(out:Tensor) -> list[tuple[int, UOp]]:
  """[(uop_count, ast), ...] for every kernel in out's schedule -- identical mechanism to
  gdn_headgroup_evidence.py's schedule_kernels, duplicated (not imported) so this script stays a
  self-contained, directly-runnable piece of evidence."""
  linear, var_vals = out.linear_with_vars()
  assert set(var_vals) <= {"start_pos"}, f"expected only the bound start_pos var, got {var_vals}"
  renderer = Device[Device.DEFAULT].renderer
  kernels = []
  for call in linear.src:
    ast = call.src[0]
    if ast.op is not Ops.SINK: continue  # skip any non-kernel (e.g. CUSTOM_FUNCTION/graph) call
    if ast.arg is None: ast = ast.replace(arg=KernelInfo())
    prg = to_program(ast, renderer)
    kernels.append((len(prg.src[1].src), ast))
  return kernels

def run(impl:int) -> list[tuple[int, UOp]]:
  block = make_block()
  x = Tensor.zeros(1, CHUNK, block.config.dim, dtype=dtypes.float32)
  x_norm = block.attn_norm(x)
  block._init_state(x_norm)
  with Context(GDN_HEAD_GROUPS=1, GDN_SCAN_IMPL=impl):  # pin head-groups to G=1: isolate the scan-impl effect
    out = block._attention(x_norm, 0)
  return schedule_kernels(out)

if __name__ == "__main__":
  print(f"geometry: num_v_heads={NUM_V_HEADS} head_v_dim={HEAD_V_DIM} head_k_dim={HEAD_K_DIM} chunk={CHUNK} "
        f"(num_k_heads={NUM_K_HEADS}, arbitrary -- collapses via repeat before the scan; GDN_HEAD_GROUPS=1 pinned)\n")

  results = {}
  for impl, label in ((GDN_SCAN_LOOP, "GDN_SCAN_IMPL=1 (loop)"), (GDN_SCAN_WY, "GDN_SCAN_IMPL=2 (wy)")):
    kernels = run(impl)
    results[impl] = kernels
    counts = [n for n, _ in kernels]
    n_big, _ = max(kernels, key=lambda x: x[0])
    print(f"{label}: {len(kernels)} kernels, uop counts={counts}")
    print(f"  largest kernel: {n_big} uops -- {'OVER' if n_big >= BEAM_UOPS_MAX else 'under'} "
          f"the BEAM_UOPS_MAX={BEAM_UOPS_MAX} cap\n")

  loop_kernels, wy_kernels = results[GDN_SCAN_LOOP], results[GDN_SCAN_WY]
  loop_big, wy_big = max(n for n, _ in loop_kernels), max(n for n, _ in wy_kernels)
  print(f"SUMMARY: loop={len(loop_kernels)} kernels (largest {loop_big} uops) vs "
        f"wy={len(wy_kernels)} kernels (largest {wy_big} uops).")
  print(f"  kernel COUNT: {'wy has fewer' if len(wy_kernels) < len(loop_kernels) else 'wy does NOT have fewer'} "
        f"kernels than the loop ({len(wy_kernels)} vs {len(loop_kernels)}).")
  under_cap = {"loop": loop_big < BEAM_UOPS_MAX, "wy": wy_big < BEAM_UOPS_MAX}
  which_under = "both" if all(under_cap.values()) else "neither" if not any(under_cap.values()) else \
    next(name for name, ok in under_cap.items() if ok) + " alone"
  print(f"  largest kernel: wy's is {'smaller' if wy_big < loop_big else 'NOT smaller'} than the loop's "
        f"({wy_big} vs {loop_big} uops) -- {which_under} under BEAM_UOPS_MAX={BEAM_UOPS_MAX}. "
        f"Report this honestly either way, not massaged.")

  print(f"""
CAVEAT -- what this does and doesn't show (read before trusting the numbers above, same caveat class as
gdn_headgroup_evidence.py's T4.62 evidence): this measures the real _attention scan graph's schedule/lowering
with NO BEAM search and NO real GPU renderer (DEV=NULL only -- both forbidden here). The loop's single huge
kernel is fused by its own final .contiguous() regardless of BEAM. wy's shape is a handful of same-size
(B*H,CHUNK,CHUNK)/(B*H,CHUNK,V) matmuls (the Neumann-doubling triangular inverse is ceil(log2(CHUNK)) steps
on the FULL matrix, not a shrinking recursion -- see _gdn_tri_inverse's docstring) -- DEV=NULL's default
(non-searched) scheduler is free to fuse a same-shape matmul chain into fewer, BIGGER kernels than the loop's
one dominant kernel, which is consistent with what's printed above (fewer kernels overall, but the single
biggest one is bigger, from late-stage fusion of the triangular-solve output with the O/state computation
that follows it -- the shared trailing uop-counts across both impls' lists above are the (impl-independent)
ssm_norm/output-gate/output-projection tail, confirming the size difference sits in the scan-specific kernels
just before that tail, not downstream of it). Whether a REAL GPU renderer's BEAM search would keep those
matmuls fused this way, split them differently, or land under the cap is exactly what this script cannot see
without violating the hard rule against opening METAL/NV -- BEAM_UOPS_MAX exists to bound BEAM SEARCH
CANDIDATE cost specifically (T4.62's comment), so crossing it here, under a scheduler that never runs BEAM at
all, does not by itself predict a real BEAM-searched run would also cross it. What IS directly confirmed here,
without a GPU: the actual kernel count and per-kernel UOp size of whatever the default scheduler produces for
each impl at this real geometry.""")
