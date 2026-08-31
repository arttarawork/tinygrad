"""T4.62 evidence: does splitting the GatedDeltaNet scan into GDN_HEAD_GROUPS head groups change the lowered
chunk-32 scan kernel(s) the way the uop-cap argument for the change needs?

Builds the else-branch chunk-32 scan graph (GatedDeltaNetBlock._attention, non-AMD path) for the REAL
qwen3.8-27B geometry named in the T4.62 spec -- num_v_heads=48, head_v_dim=128, head_k_dim=128 -- once at
GDN_HEAD_GROUPS=1 (today's behavior: one un-grouped scan) and once at GDN_HEAD_GROUPS=2 (T4.62's split),
schedules each, and for every kernel in the schedule prints its linearized UOp count via the exact mechanism
tinygrad/codegen/opt/search.py's _try_compile checks against BEAM_UOPS_MAX (default 3000): to_program(ast,
renderer) then len(prg.src[1].src).

Run with: DEV=NULL PYTHONPATH=. <venv>/bin/python extra/gdn_headgroup_evidence.py
(DEV=NULL: no real device is opened, per T4.62's hard rule against touching METAL/NV; BEAM/JITBEAM are
asserted unset below, so no BEAM search runs either -- apply_opts takes its default, non-searched path.)

HONEST RESULT (read before trusting the printed numbers): under DEV=NULL's default optimizer, NEITHER
GDN_HEAD_GROUPS=1 NOR =2 crosses BEAM_UOPS_MAX for this geometry -- this script does NOT reproduce the
production METAL/NV, BEAM-searched cap-crossing numerically (see the CAVEAT printed at the end, and the
T4.62 completion report, for why and for what this script confirms instead).
"""
import os
assert os.environ.get("DEV") == "NULL", "run with DEV=NULL -- see module docstring"
assert not os.environ.get("BEAM") and not os.environ.get("JITBEAM"), "BEAM/JITBEAM must stay unset (T4.62 hard rule)"

from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Context
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig

# the three "REAL dims" the T4.62 spec names for qwen3.8-27B
NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM = 48, 128, 128
CHUNK = 32  # GDN_CHUNK's measured-good chunk width (model.py) -- this is "the chunk-32 scan graph"
NUM_K_HEADS = 8  # not one of the spec's "REAL dims"; q/k are repeat-expanded to NUM_V_HEADS before the
                 # group split (GatedDeltaNetBlock._attention, before the else-branch), so this choice
                 # doesn't affect the scan's UOp count -- any divisor of NUM_V_HEADS gives the same numbers
BEAM_UOPS_MAX = 3000  # tinygrad/codegen/opt/search.py: getenv("BEAM_UOPS_MAX", 3000)

def make_block() -> GatedDeltaNetBlock:
  ssm = SSMConfig(conv_kernel=4, state_size=HEAD_K_DIM, group_count=NUM_K_HEADS, time_step_rank=NUM_V_HEADS,
                   inner_size=HEAD_V_DIM * NUM_V_HEADS)
  config = TransformerConfig(num_blocks=1, dim=128, hidden_dim=256, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                              vocab_size=32, head_dim=128, rope_theta=10000.0, rope_dim=128, v_head_dim=128,
                              max_context=CHUNK, ssm_layers=(True,), ssm=ssm)
  return GatedDeltaNetBlock(config, ssm)

def schedule_kernels(out:Tensor) -> tuple[list[tuple[int, UOp]], Renderer]:
  """[(uop_count, ast), ...] for every kernel in out's schedule, via the same to_program(...).src[1].src
  BEAM_UOPS_MAX checks against (search.py's _try_compile). No BEAM search: KernelInfo.beam defaults to 0
  (module-level assert above backs this with the env check), so apply_opts takes its default path."""
  # var_vals always has a bound "start_pos" (GatedDeltaNetBlock._attention binds it even for a concrete int
  # start_pos, so the reset-at-position-0 check stays a runtime value) -- a bound scalar, not a shape-varying
  # symbolic dim, so T_pad's unrolled loop is still fully concrete. Nothing else expected here.
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
  return kernels, renderer

def run(groups:int) -> tuple[list[tuple[int, UOp]], Renderer]:
  block = make_block()
  x = Tensor.zeros(1, CHUNK, block.config.dim, dtype=dtypes.float32)
  x_norm = block.attn_norm(x)
  block._init_state(x_norm)
  with Context(GDN_HEAD_GROUPS=groups):
    out = block._attention(x_norm, 0)
  return schedule_kernels(out)

if __name__ == "__main__":
  print(f"geometry: num_v_heads={NUM_V_HEADS} head_v_dim={HEAD_V_DIM} head_k_dim={HEAD_K_DIM} chunk={CHUNK} "
        f"(num_k_heads={NUM_K_HEADS}, arbitrary -- collapses via repeat before the group split)\n")

  for groups in (1, 2):
    kernels, renderer = run(groups)
    counts = [n for n, _ in kernels]
    n_big, ast_big = max(kernels, key=lambda x: x[0])
    print(f"GDN_HEAD_GROUPS={groups}: {len(kernels)} kernels, uop counts={counts}")
    print(f"  largest kernel: {n_big} uops -- {'OVER' if n_big >= BEAM_UOPS_MAX else 'under'} "
          f"the BEAM_UOPS_MAX={BEAM_UOPS_MAX} cap")
    # Confirm *why* this kernel is the relevant one: its schedule should have a num_v_heads-sized axis (the
    # axis GDN_HEAD_GROUPS narrows), and that axis should be upcastable -- i.e. eligible for exactly the
    # vectorization opts (search.py's UPCAST/UNROLL/etc `actions`) a real BEAM search would explore.
    head_axis_size = NUM_V_HEADS if groups <= 1 else -(-NUM_V_HEADS // groups)
    k = Scheduler(ast_big, renderer)
    matches = [i for i, sz in enumerate(k.full_shape) if sz == head_axis_size and i in k.upcastable_dims]
    print(f"  full_shape={k.full_shape}")
    print(f"  head-sized ({head_axis_size}) upcastable axis at position(s) {matches}"
          + ("" if matches else "  <-- NOT FOUND, see CAVEAT"))
    print()

  print(f"""CAVEAT -- what this does and doesn't show (see it BEFORE trusting the numbers above):
This measures the real _attention scan graph's schedule/lowering with NO BEAM search (forbidden by T4.62's
hard rules) and NO real GPU renderer (DEV=NULL only, also forbidden). Under NULL's default, non-searched
optimizer neither GDN_HEAD_GROUPS=1 nor =2 crosses BEAM_UOPS_MAX here: this sandbox's default lowering does
not reproduce the exact METAL/NV, BEAM-searched cap-crossing that motivates T4.62 (that needs the real
renderer's has_local=True heuristics and its 54-candidate search, which this script cannot touch without
violating the hard rules). Separately probing a hand-picked UPCAST of the confirmed num_v_heads axis (one
of search.py's actual `actions`, applied directly via Scheduler.apply_opt -- not a search, no device use)
showed uop count scaling with the UPCAST factor, not with the axis's total extent, and tinygrad refuses to
fully-upcast (amt=0) any axis over 16 elements regardless of whether it's 24 or 48 -- so even that probe
doesn't reproduce an H=32-vs-H=48-style threshold by itself; the real BEAM search's blowup likely comes
from a *combination* of opts (UPCAST+LOCAL+GROUP together) across its 54 candidates, which is exactly the
search this script must not run.
What IS directly confirmed here, without a GPU: (1) GDN_HEAD_GROUPS structurally changes the schedule --
GDN_HEAD_GROUPS=2 produces roughly double the kernel count (separate kernels per group) instead of one
full-width kernel set; (2) num_v_heads is a real, distinctly-sized, upcastable schedule axis in the
dominant kernel, shrinking from {NUM_V_HEADS} at G=1 to {-(-NUM_V_HEADS//2)} at G=2 -- exactly the axis a
head-vectorizing BEAM candidate would upcast, and exactly the axis this change narrows. Caveat on fusion:
nothing here stops the scheduler from still fusing unrelated per-group work (e.g. two groups' conv/gating
slices) back together where it legally can -- the only fusion boundary this change *forces* is each
group's own state/output .contiguous(), which is what produced the roughly-doubled kernel count above.""")
