"""T4.73 evidence: a CPU-side aliasing DETECTOR for the WY prefill->decode corruption bug (AUDIT.md).

THE BUG (can't be reproduced here -- see AUDIT.md): on the real METAL+NV pooled split, serving qwen3.8-27B
with GDN_SCAN_IMPL=2 (gdn_scan_wy, model.py), a chunked WY prefill completes fine, then the FIRST decode
step samples an out-of-vocab token id -- i.e. something in the recurrent GDN state (or a value near it)
gets corrupted crossing from the prefill jit capture into the decode jit capture. CPU (incl. multi-device
CPU:0/CPU:1) never reproduces it; the working hypothesis is a real-device buffer-donation/allocator
interaction that CPU's allocator masks (AUDIT.md §"why CPU masks it").

This script cannot exercise that real-device path at all (DEV=CPU only, per the hard rule against opening
METAL/NV). What it CAN do, honestly:

  Part A -- graph-shape check (exact, no scheduling): gdn_scan_wy's two return values are inspected via
  UOp.has_buffer_identity(), the exact predicate Tensor.contiguous() itself checks to decide whether to
  insert a real materialization boundary (Ops.CONTIGUOUS) or no-op -- see tinygrad/mixin/elementwise.py.
  False (the expected/reported result on this file's ORIGINAL, pre-T4.73 code, where final_state is a bare
  elementwise MUL and out.transpose(1,2) a bare PERMUTE view) means both are lazy expressions the memory
  planner is free to fuse into whatever consumes them next -- i.e. candidates 1/2's .contiguous() calls are
  real materialization points, not no-ops. If candidates 1/2 are ALREADY applied when this runs, gdn_scan_wy
  itself now returns .contiguous()-wrapped (Ops.CONTIGUOUS) values -- has_buffer_identity() is still False
  for those too (CONTIGUOUS isn't RESHAPE/BUFFER/PARAM), so this check's PASS/FAIL is stable either way; only
  the printed `op=` field changes (MUL/PERMUTE pre-patch vs CONTIGUOUS post-patch) to show which state you're
  looking at.

  Part B -- "under the JIT" buffer-identity tracking (the DELIVER-list ask): a real (tiny) GatedDeltaNetBlock
  at the actual qwen3.8-27B geometry (48 heads -> GDN_HEAD_GROUPS auto-splits into G=2, the SAME code path
  the bug report's config takes) is driven through TinyJit-wrapped prefill-shaped (T_pad=4) then
  decode-shaped (T_pad=1) calls -- warmup, capture, replay, matching engine/jit.py's own cnt 0/1/>=2 stages --
  once under GDN_SCAN_IMPL=2 (WY) and once under =1 (LOOP), each on a fresh block. engine/jit.py's
  memory_plan_rewrite (the buffer-donation/arena-reuse pass -- see AUDIT.md) is monkeypatched to record, per
  capture: how many distinct buffers it saw, how many were `held` (protected from arena reuse -- see
  _TinyJit.__call__'s held_bufs computation) vs `plannable` (arena-reuse-eligible), and -- the actual
  "flag an output buffer later reused as an intermediate" check -- whether self.recurrent_state's and
  self.conv_state's OWN buffers were ever plannable (i.e. NOT held) in either impl's capture. Those two
  buffers are exactly the ones that must survive from one jit capture into the next (prefill -> decode); if
  either is ever plannable, that is a real, direct bug (the planner could donate its physical memory to some
  in-capture scratch value). This part uses ONLY the same public helpers memory_plan_rewrite itself uses
  (_collect_bufs, _can_plan from tinygrad.schedule.memory) -- it does not reimplement or guess at the
  allocator's own interval/arena logic, only re-derives the same held/plannable split for reporting.

  HONEST LIMITS: memory_plan_rewrite is backend-agnostic Python (same code runs under CPU/METAL/NV -- see
  AUDIT.md's git-log check: this file is untouched fork-local upstream plumbing), so Part B running clean on
  CPU does NOT clear METAL/NV -- it can only clear (or indict) the *planner's own bookkeeping*, which is
  identical across devices. It cannot see real-device allocator pooling, async command-queue timing, or
  buffer-copy-engine races, which is exactly the class of thing this bug is suspected to be (AUDIT.md ranks
  this the leading "why CPU masks it" explanation). A clean Part B result narrows the hypothesis space
  (rules out "the state buffer's own identity gets stolen by the generic planner") without proving the real
  device is safe.

Run with: CHECK_OOB=1 DEV=CPU PYTHONPATH=. <venv>/bin/python extra/wy_alias_audit.py
"""
import os
assert os.environ.get("DEV") == "CPU", "run with DEV=CPU -- see module docstring"

from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
import tinygrad.engine.jit as jit_mod
from tinygrad.schedule.memory import _collect_bufs, _can_plan
from tinygrad.llm.model import (
  GatedDeltaNetBlock, SSMConfig, TransformerConfig, GDN_SCAN_LOOP, GDN_SCAN_WY, gdn_scan_wy,
)

def _buf_of(t:Tensor) -> UOp:
  """Unwrap RESHAPE/UNSHARD/MSELECT/DETACH/AFTER down to the underlying Ops.BUFFER -- same spine
  UOp.has_buffer_identity(after_ok=True) walks, but returning the buffer UOp itself (for `in held_bufs`/
  `in plannable` membership tests) instead of a bool. self.recurrent_state/self.conv_state are built via
  Tensor.zeros(...).clone(), which is an eager AFTER(RESHAPE(BUFFER), STORE(...)) plan, not yet a bare
  BUFFER -- this is what actually needs unwrapping here."""
  u = t.uop
  while u.op in (Ops.RESHAPE, Ops.UNSHARD, Ops.MSELECT, Ops.DETACH, Ops.AFTER): u = u.src[0]
  assert u.op is Ops.BUFFER, f"expected to land on Ops.BUFFER, got {u.op} -- re-check tinygrad internals"
  return u

# the real qwen3.8-27B geometry (same numbers as gdn_wy_evidence.py / gdn_headgroup_evidence.py):
# num_v_heads=48 -> gdn_head_groups_for(48) auto-splits into G=2, the exact head-group config the T4.72
# bug ladder's failing run used. dim kept tiny so the surrounding linears/conv compile fast on CPU.
NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, NUM_K_HEADS = 48, 8, 8, 8
DIM, MAX_CONTEXT, CHUNK = 8, 64, 4  # CHUNK>1 (so T_pad>1 triggers WY) but shallow (no deep unrolled chains)

def make_block(seed:int) -> GatedDeltaNetBlock:
  ssm = SSMConfig(conv_kernel=4, state_size=HEAD_K_DIM, group_count=NUM_K_HEADS, time_step_rank=NUM_V_HEADS,
                   inner_size=HEAD_V_DIM * NUM_V_HEADS)
  config = TransformerConfig(num_blocks=1, dim=DIM, hidden_dim=DIM * 2, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                              vocab_size=32, head_dim=DIM, rope_theta=10000.0, rope_dim=DIM, v_head_dim=DIM,
                              max_context=MAX_CONTEXT, ssm_layers=(True,), ssm=ssm)
  block = GatedDeltaNetBlock(config, ssm)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(block)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  Tensor.realize(*params)  # pin now -- Tensor's lazy RNG counter footgun, see test_gdn_scan_parity.py
  return block

# ---------------------------------------------------------------------------------------------------------
# Part A: is gdn_scan_wy's raw output already "safe" (has_buffer_identity), or a bare lazy expression?
# ---------------------------------------------------------------------------------------------------------
def part_a() -> bool:
  print("=== Part A: gdn_scan_wy output graph-shape check (no scheduling; result is stable whether or not "
        "candidates 1/2 are currently applied -- see module docstring) ===")
  B, H, T, V, K = 1, 4, CHUNK, 6, 6
  state = Tensor.randn(B, H, V, K) * 0.1
  q, k = Tensor.randn(B, H, T, K) * 0.1, Tensor.randn(B, H, T, K) * 0.1
  v = Tensor.randn(B, H, T, V) * 0.1
  beta = Tensor.randn(B, H, T).sigmoid()
  alpha = Tensor.randn(B, H, T, V).sigmoid()  # kda-shaped alpha (V-wide); non-kda (1-wide) has the same op shape
  final_state, out_t = gdn_scan_wy(state, q, k, v, beta, alpha)
  fs_has_id, out_has_id = final_state.uop.has_buffer_identity(), out_t.uop.has_buffer_identity()
  print(f"  final_state.uop.has_buffer_identity()        = {fs_has_id}  (op={final_state.uop.op})")
  print(f"  out.transpose(1,2).uop.has_buffer_identity()  = {out_has_id}  (op={out_t.uop.op})")
  ok = not fs_has_id and not out_has_id
  print(f"  -> both are lazy (not already-safe) expressions: {ok} "
        f"({'candidates 1/2 are real materialization points, not no-ops' if ok else 'UNEXPECTED -- re-check AUDIT.md reasoning'})")
  return ok

# ---------------------------------------------------------------------------------------------------------
# Part B: under a real TinyJit prefill->decode transition, is recurrent_state/conv_state EVER plannable
# (i.e. arena-reuse-eligible, "flagged as an intermediate") in either impl's capture?
# ---------------------------------------------------------------------------------------------------------
CAPTURES: list[dict] = []          # one entry per memory_plan_rewrite call this script triggers
_current_label = [""]
_current_bufs: list[tuple[UOp, UOp]|tuple[None, None]] = [(None, None)]  # (recurrent_state, conv_state) buf UOps

_real_memory_plan_rewrite = jit_mod.memory_plan_rewrite
def _instrumented_memory_plan_rewrite(linear, held_bufs=None):
  hb = held_bufs if held_bufs is not None else set()
  seen: dict[UOp, int] = {}
  for i, si in enumerate(linear.src):
    for src in si.src[1:]:
      for b in _collect_bufs(src):
        seen[b] = i  # last index is all we need here; membership is what we report
  plannable = [b for b in seen if _can_plan(b, hb)]
  rec_buf, conv_buf = _current_bufs[0]
  CAPTURES.append(dict(
    label=_current_label[0], n_calls=len(linear.src), n_distinct_bufs=len(seen), n_held=len(seen) - len(plannable),
    n_plannable=len(plannable),
    rec_state_plannable=(rec_buf in plannable) if rec_buf is not None else None,
    conv_state_plannable=(conv_buf in plannable) if conv_buf is not None else None,
    rec_state_held=(rec_buf in hb) if rec_buf is not None else None,
    conv_state_held=(conv_buf in hb) if conv_buf is not None else None,
  ))
  return _real_memory_plan_rewrite(linear, held_bufs)

def run_sequence(impl:int, label:str) -> None:
  block = make_block(seed=0)
  v_start = UOp.variable("start_pos", 0, MAX_CONTEXT - 1)

  @TinyJit
  def prefill_step(x:Tensor, start_pos:UOp) -> Tensor:
    xn = block.attn_norm(x)
    block._init_state(xn)
    return block._attention(xn, start_pos)

  @TinyJit
  def decode_step(x:Tensor, start_pos:UOp) -> Tensor:
    xn = block.attn_norm(x)
    block._init_state(xn)
    return block._attention(xn, start_pos)

  with Context(GDN_SCAN_IMPL=impl, GDN_HEAD_GROUPS=0):  # auto head-groups: 48 heads -> G=2, the bug's config
    pos = 0
    for i in range(3):  # cnt 0 (warmup/ignore) -> 1 (CAPTURE, instrumented) -> 2 (replay)
      x = (Tensor.randn(1, CHUNK, DIM) * 0.1).realize()
      _current_label[0] = f"{label} prefill call#{i}"
      if i == 0:  # first call initializes conv_state/recurrent_state -- grab their buffer identity now
        block._init_state(block.attn_norm(x))
        _current_bufs[0] = (_buf_of(block.recurrent_state), _buf_of(block.conv_state))
      prefill_step(x, v_start.bind(pos)).realize()
      pos += CHUNK
    for i in range(3):
      x = (Tensor.randn(1, 1, DIM) * 0.1).realize()
      _current_label[0] = f"{label} decode call#{i}"
      decode_step(x, v_start.bind(pos)).realize()
      pos += 1

def part_b() -> bool:
  print("\n=== Part B: recurrent_state/conv_state buffer-identity tracking across a real prefill->decode JIT transition ===")
  jit_mod.memory_plan_rewrite = _instrumented_memory_plan_rewrite
  try:
    run_sequence(GDN_SCAN_WY, "WY")
    run_sequence(GDN_SCAN_LOOP, "LOOP")
  finally:
    jit_mod.memory_plan_rewrite = _real_memory_plan_rewrite

  flagged = []
  for c in CAPTURES:
    print(f"  {c['label']:22s} kernels={c['n_calls']:3d} bufs={c['n_distinct_bufs']:3d} held={c['n_held']:3d} plannable={c['n_plannable']:3d} "
          f"rec_state:{'HELD' if c['rec_state_held'] else 'plannable' if c['rec_state_held'] is False else 'n/a'} "
          f"conv_state:{'HELD' if c['conv_state_held'] else 'plannable' if c['conv_state_held'] is False else 'n/a'}")
    if c["rec_state_plannable"] or c["conv_state_plannable"]: flagged.append(c["label"])

  wy_prefill = [c for c in CAPTURES if c["label"] == "WY prefill call#1"][0]
  loop_prefill = [c for c in CAPTURES if c["label"] == "LOOP prefill call#1"][0]
  print(f"\n  graph-complexity delta at capture time (prefill, same geometry): WY {wy_prefill['n_distinct_bufs']} "
        f"distinct buffers vs LOOP {loop_prefill['n_distinct_bufs']} "
        f"({'+' if wy_prefill['n_distinct_bufs'] >= loop_prefill['n_distinct_bufs'] else ''}"
        f"{wy_prefill['n_distinct_bufs'] - loop_prefill['n_distinct_bufs']})")

  if flagged:
    print(f"  FLAGGED: recurrent_state/conv_state was arena-reuse-eligible in: {flagged} -- this would be a direct,\n"
          "  confirmed planner bug (the state buffer's own physical memory could be donated to scratch). Investigate model.py's\n"
          "  _init_state / held_bufs interaction before looking anywhere else.")
  else:
    print("  CLEAR: recurrent_state and conv_state were HELD (never arena-reuse-eligible) in every capture, both impls.\n"
          "  This rules out \"the state buffer's own identity gets stolen\" on CPU's planner logic (shared, device-agnostic\n"
          "  code -- see AUDIT.md). It does NOT clear METAL/NV: real allocator pooling / async completion timing is outside\n"
          "  what this generic, backend-agnostic pass can expose on any device, CPU included.")
  return not flagged

if __name__ == "__main__":
  ok_a = part_a()
  ok_b = part_b()
  print(f"\n=== summary: Part A clean={ok_a}, Part B clear={ok_b} -- see AUDIT.md for the full ranked hypothesis list ===")
  assert ok_a, "Part A: gdn_scan_wy's outputs unexpectedly already have buffer identity -- re-check reasoning above"
  assert ok_b, "Part B: recurrent_state/conv_state was arena-reuse-eligible somewhere -- see FLAGGED output above"
