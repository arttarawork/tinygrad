#!/usr/bin/env python3
"""T4.74 fault-path analysis: DEV=NULL/CPU-only harness that builds Transformer.snapshot_state()/restore_state()'s
EXACT clone/assign graphs at the real qwen3.8-27B pooled-map shapes (B=1, n_kv_heads=4, head_dim=256, pos=5872,
max_context=65536, fp16 -- the shapes named in the T4.74 task) and runs them through tinygrad's normal schedule +
CHECK_OOB (z3) verification pipeline (tinygrad/uop/spec.py's validate_index, wired into every schedule via
tinygrad/schedule/__init__.py's `if SPEC: type_verify(...)` -- SPEC defaults to 1, so CHECK_OOB=1 alone is enough,
no SPEC=2 needed; see ANALYSIS.md), then contrasts the schedule shape against speculative_generate's CHECKPOINT
idiom (whole-buffer GDN clone, no position slicing). Never touches METAL/NV -- asserted below.

Run: CHECK_OOB=1 DEV=NULL PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/snapshot_oob_audit.py
(DEV=CPU also works; NULL is faster -- NullAllocator._alloc is a no-op, see tinygrad/runtime/ops_null.py, so the
256 MB/block full cache_kv buffers this harness declares cost nothing real to "allocate".)
"""
import sys
from tinygrad import Tensor, Device
from tinygrad.helpers import Context, CHECK_OOB, SPEC
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, snapshot_matches

# HARD RULE: this tool audits schedules, it never executes against real hardware.
assert Device.DEFAULT in ("NULL", "CPU"), f"refusing to run on DEV={Device.DEFAULT} -- this audit is DEV=NULL/CPU only"

# the real T4.74 shapes: qwen3.8-27B pooled METAL+NV map, snapshot taken right after a 5872-token prefill.
B, KV_HEADS, HEAD_DIM, MAX_CTX, POS = 1, 4, 256, 65536, 5872

def _attn_config(num_blocks:int) -> TransformerConfig:
  # dim/hidden_dim/vocab are toy-sized -- irrelevant to cache_kv's shape, which depends only on n_kv_heads/
  # head_dim/max_context (see TransformerBlock._init_state). n_heads==n_kv_heads (MHA) is the simplest valid
  # config at these cache dimensions -- GQA head-count ratios don't change cache_kv's shape either.
  return TransformerConfig(num_blocks=num_blocks, dim=32, hidden_dim=64, n_heads=KV_HEADS, n_kv_heads=KV_HEADS,
                           norm_eps=1e-5, vocab_size=32, head_dim=HEAD_DIM, rope_theta=10000.0, rope_dim=HEAD_DIM,
                           v_head_dim=HEAD_DIM, max_context=MAX_CTX)

def _seed_attn_cache(model:Transformer, pos:int) -> None:
  """Allocates every block's cache_kv at its real (production) shape via the block's own _init_state -- no forward
  pass needed: snapshot_state only ever reads cache_kv's shape/buffer and len(_cached_tokens), never how they were
  populated. Faking `pos` this way (instead of an actual pos-token generate() prefill) is deliberate: a real
  5872-token CPU/NULL forward pass would be slow for no benefit here, since NULL's execution is a no-op anyway
  (runtime/ops_null.py) -- no real numbers ever flow. This harness audits SHAPES and INDEX BOUNDS, not values;
  test/unit/test_state_cache.py already covers round-trip VALUE correctness at small scale."""
  x = Tensor.zeros(B, 1, model.blk[0].config.dim)
  for b in model.blk: b._init_state(x)
  # realize NOW: production's cache_kv is always an already-materialized Buffer by the time snapshot_state clones
  # it (populated via real .store() writes over prior forward passes -- see TransformerBlock._attention). Leaving
  # it as a lazy Tensor.zeros(...) graph here would make the clone's schedule also include "materialize the zero
  # fill", which is not part of the real fault-path graph and would misreport the CALL count below.
  Tensor.realize(*(b.cache_kv for b in model.blk))
  model._cached_tokens = list(range(pos))

def _gdn_config() -> TransformerConfig:
  # small/synthetic dims: CHECKPOINT's shape doesn't depend on context length at all (see ANALYSIS.md), so there's
  # no "real dimensions" to match here the way there is for cache_kv -- one GDN block is enough to characterize it.
  return TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=32,
                           head_dim=16, rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=MAX_CTX,
                           ssm=SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=4, inner_size=32),
                           ssm_layers=(True,))

def classify(call:UOp) -> str:
  ast = call.src[0]
  kind = "COPY (raw buffer transfer, no codegen/BEAM)" if ast.op is Ops.COPY else f"kernel-shaped ({ast.op.name}, needs codegen)"
  shrink = any(u.op is Ops.SHRINK for u in ast.toposort())
  return f"{kind}{'  [index derives from a SHRINK view]' if shrink else ''}"

def report_schedule(tensors:list[Tensor], label:str) -> UOp:
  linear = tensors[0].schedule_linear(*tensors[1:])
  print(f"\n--- {label}: {len(linear.src)} CALL(s) ---")
  for i, call in enumerate(linear.src): print(f"  [{i}] {classify(call)}")
  return linear

def verified(fn, label:str) -> bool:
  try:
    with Context(CHECK_OOB=1):
      fn()
    print(f"PASS  {label}: CHECK_OOB proved every LOAD/STORE in bounds (z3-backed, real shapes)")
    return True
  except Exception as e:
    print(f"FAIL  {label}: {type(e).__name__}: {e}")
    return False

def main() -> int:
  print(f"DEV={Device.DEFAULT}  ambient CHECK_OOB={CHECK_OOB.value}  SPEC={SPEC.value} (>=1 required for type_verify to run at all)")
  ok = True
  NUM_ATTN_BLOCKS = 16  # T4.74's real map: ALL 16 attention blocks live on NV for the qwen3.8-27B pooled split

  # 1) isolate the exact per-block op snapshot_state() performs: cache_kv[:, :, :, :pos, :].clone()
  model = Transformer(_attn_config(num_blocks=NUM_ATTN_BLOCKS))
  _seed_attn_cache(model, POS)
  mb = (2*B*KV_HEADS*POS*HEAD_DIM*2) / 1e6  # fp16 = 2 bytes; matches T4.74's "~370 MB / 16 blocks" ~= 23 MB/block
  print(f"\nfull cache_kv shape={tuple(model.blk[0].cache_kv.shape)}  per-block clone shape="
        f"{tuple((model.blk[0].cache_kv[:, :, :, :POS, :]).shape)}  dtype={model.blk[0].cache_kv.dtype}  ({mb:.1f} MB/block)")
  # NOTE: each `.clone()` below is a fresh Tensor object scheduled exactly once -- schedule_linear/realize on the
  # SAME tensor object a second time (e.g. reusing a tensor across two separate report_schedule calls) was found,
  # while building this harness, to sometimes report a misleadingly-fused CALL count on a second scheduling; a
  # clean sweep (num_blocks=1,2,3,4,8,16, each scheduled exactly once) confirmed the real, reproducible behavior
  # used below: N blocks -> N separate CALLs, linear, no fusion at any tested N.
  report_schedule([model.blk[0].cache_kv[:, :, :, :POS, :].clone()], "single attention-block clone (snapshot_state's per-block op)")

  # 2) the SAME op on all 16 blocks, batched in ONE realize -- production's own
  #    `Tensor.realize(*(t for bs in blocks for t in bs.values()))` idiom, at the real block count
  all_clones = [b.cache_kv[:, :, :, :POS, :].clone() for b in model.blk]
  linear_all = report_schedule(all_clones, f"ALL {NUM_ATTN_BLOCKS} attention blocks, ONE batched realize (production's exact idiom)")
  kernel_calls = [c for c in linear_all.src if c.src[0].op is not Ops.COPY]
  shared = len(kernel_calls) == NUM_ATTN_BLOCKS and all(c.src[0] is kernel_calls[0].src[0] for c in kernel_calls)
  print(f"  kernel identity across all {NUM_ATTN_BLOCKS} blocks: "
        f"{'SAME AST object -- ONE compile/BEAM search serves every block' if shared else 'DIFFERENT ASTs'}")

  # 3) the real call: Transformer.snapshot_state(), CHECK_OOB-verified end to end at the real shapes and block count
  ok &= verified(lambda: model.snapshot_state(), f"model.snapshot_state() ({NUM_ATTN_BLOCKS} attention blocks, real shape, pos={POS})")

  # 4) restore_state's mirror-image slice-assign, same shapes (the fault's inverse op; T4.72 noted this leg is
  #    the one that kept working "even mid-fault-era" -- worth its own verdict, not just assumed safe by symmetry)
  snap = model.snapshot_state()
  ok &= verified(lambda: model.restore_state(snap), "model.restore_state(snap) (slice-assign mirror of #3)")
  assert snapshot_matches(snap, list(range(POS)) + [999]), "snapshot_matches sanity check"

  # 5) contrast: speculative_generate's CHECKPOINT idiom -- GDN whole-buffer clone, NO position slicing at all
  gdn_model = Transformer(_gdn_config())
  gx = Tensor.zeros(B, 1, gdn_model.blk[0].config.dim)
  gdn_model.blk[0]._init_state(gx)
  gdn_block = gdn_model.blk[0]
  Tensor.realize(gdn_block.conv_state, gdn_block.recurrent_state)  # see _seed_attn_cache's comment: match production's already-materialized state
  ck_clone = (gdn_block.conv_state.clone(), gdn_block.recurrent_state.clone())
  print(f"\nCHECKPOINT conv_state shape={tuple(gdn_block.conv_state.shape)}  "
        f"recurrent_state shape={tuple(gdn_block.recurrent_state.shape)}  (both position-independent, O(1) in max_context)")
  report_schedule(list(ck_clone), "speculative_generate's CHECKPOINT (GDN conv/recurrent whole-buffer clone)")
  ok &= verified(lambda: Tensor.realize(*ck_clone), "CHECKPOINT clone realize")

  print(f"\n{'ALL CHECKS PASSED' if ok else 'AT LEAST ONE CHECK FAILED'}")
  return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())
