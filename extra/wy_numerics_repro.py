#!/usr/bin/env python3
"""T4.73c: seconds-fast, CPU-only repro + bisection for the WY chunked GDN scan's NaN blowup at REAL
qwen3.8-27B blk.0 weights (see extra/extract_blk0_real.py for how those weights were pulled, and
extra/bug1_gguf_ab.py for the ORIGINAL hardware+full-model finding this reproduces at the single-block
level: GDN_SCAN_IMPL=2 -> recurrent_state NaN starting at blk0, exactly one head's worth (16384 =
786432/48) of NaNs; GDN_SCAN_IMPL=1 (same math, per-token loop) -> clean).

Builds ONE real GatedDeltaNetBlock (non-kda: arch=qwen35 has attn_gate/ssm_alpha, not ssm_g_a/ssm_f_a),
loads blk.0's real weights, and drives it with the real embedded prompt rows (ids 1000..1031) through
_attention at chunk 32 -- no full Transformer, no tokenizer, no generate()/JIT.

Diagnostic mechanism: gdn_scan_wy is monkeypatched (module-level function swap, not an edit to model.py)
so every call's real (state,q,k,v,beta,alpha) inputs -- exactly what _attention's run_scan closure builds
-- are captured and independently inspected before delegating to the real implementation, unchanged.

Run: PYTHONPATH=. <venv>/bin/python extra/wy_numerics_repro.py
"""
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.nn.state import safe_load, load_state_dict
import tinygrad.llm.model as M
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig, GDN_SCAN_LOOP, GDN_SCAN_WY, _gdn_tri_inverse

WEIGHTS = "extra/blk0_real.safetensors"

# real qwen3.8-27B blk.0 geometry (extract_blk0_real.py's printed kv -- matches test_gdn_scan_parity.py's
# existing "38" GEOMETRIES entry exactly)
SSM = SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=128 * 48)
DIM = 5120
NORM_EPS = 9.999999974752427e-07

def make_real_block() -> GatedDeltaNetBlock:
  config = TransformerConfig(num_blocks=1, dim=DIM, hidden_dim=DIM * 2, n_heads=1, n_kv_heads=1, norm_eps=NORM_EPS,
                              vocab_size=32, head_dim=DIM, rope_theta=10000.0, rope_dim=DIM, v_head_dim=DIM,
                              max_context=64, ssm_layers=(True,), ssm=SSM)
  block = GatedDeltaNetBlock(config, SSM)
  loaded = {k: v.to("CPU") for k, v in safe_load(WEIGHTS).items()}  # safe_load's tensors are DISK-backed; CPU-only rule
  x = loaded.pop("token_embd_rows")
  # strict=False: this state dict deliberately omits ffn_*/post_attention_norm.weight (FFNBlock-only,
  # unused by _attention/_init_state -- see extract_blk0_real.py) -- those stay at their random __init__
  # values, which is fine, this repro never calls the FFN path.
  load_state_dict(block, loaded, strict=False, verbose=False)
  return block, x

def nan_inf_report(name: str, t: Tensor) -> tuple[int, int]:
  a = t.float().numpy()
  n_nan, n_inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
  if n_nan or n_inf: print(f"    {name}: shape={a.shape} NaN={n_nan} Inf={n_inf}")
  return n_nan, n_inf

def diagnose(state: Tensor, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, alpha: Tensor) -> None:
  """Re-derives gdn_scan_wy's own intermediates from the SAME real inputs it was just called with, to find
  the first non-finite value's exact expression and head. Pure numpy from here -- read-only, no tinygrad
  graph side effects, doesn't touch the real call this wraps."""
  B, H, T, _ = q.shape
  a = alpha.float().numpy()               # (B,H,T,1) -- non-kda: one scalar per head per step
  a_bar = np.cumprod(a, axis=2)           # exactly what gdn_scan_wy computes (same op, same dtype/order)
  v_np = v.float().numpy()
  print(f"  alpha: per-head per-step decay, min={a.min():.6g} max={a.max():.6g}")
  # per head: first chunk position (if any) where the CUMULATIVE product underflows to an exact float32 0
  zero_head, zero_pos = None, None
  for h in range(H):
    ah = a_bar[0, h, :, 0]
    nz = np.nonzero(ah == 0.0)[0]
    if len(nz):
      print(f"  head {h}: a_bar underflows to EXACT 0.0 at chunk pos {nz[0]} (of {T}); "
            f"a_bar[pos-1]={ah[nz[0]-1] if nz[0] else 1.0:.3e} -> a_bar[pos]=0.0 "
            f"(per-step alpha there = {a[0,h,nz[0],0]:.3e})")
      if zero_head is None: zero_head, zero_pos = h, nz[0]
    else:
      print(f"  head {h}: a_bar never underflows, min={ah.min():.3e}")
  if zero_head is None:
    print("  no head's a_bar underflows to exact 0.0 in this chunk -- WY should be numerically clean here")
    return
  print(f"\n  FIRST culprit: head {zero_head}, chunk pos {zero_pos} (a_bar==0.0 exactly from here to chunk end)")
  v_tilde = v_np / a_bar
  bad = ~np.isfinite(v_tilde[0, zero_head])
  print(f"  v_tilde = v/a_bar at head {zero_head}: non-finite at {bad.sum()}/{bad.size} (t,v) entries"
        f" (all t>=pos with a_bar==0, since v is generically nonzero there: {v_np[0,zero_head,zero_pos].round(3)[:4]}...)")

  # cross-check the tri-inverse hypothesis independently, at the SAME real beta/k magnitudes: does
  # _gdn_tri_inverse's Neumann doubling blow up even with a well-scaled (no a_bar involved) rhs?
  Tn = q.shape[2]
  idx = np.arange(Tn)
  strict_lower = (idx.reshape(1, Tn) < idx.reshape(Tn, 1)).astype(np.float32)
  k_np, beta_np = k.float().numpy(), beta.float().numpy()
  kkt = np.einsum("bhtd,bhsd->bhts", k_np, k_np) * strict_lower
  m = beta_np[..., None] * kkt
  m_t = Tensor(m.astype(np.float32))
  inv = _gdn_tri_inverse(m_t).numpy()
  print(f"  m=beta*kkt (decay-FREE, as gdn_scan_wy builds it): max|m|={np.abs(m).max():.3e}")
  print(f"  _gdn_tri_inverse(m) alone (no v/a_bar involved): max|.|={np.abs(inv).max():.3e}, "
        f"finite={np.isfinite(inv).all()} -- {'BLOWS UP TOO' if not np.isfinite(inv).all() else 'stays well-conditioned'}")

_orig_gdn_scan_wy = M.gdn_scan_wy
_calls = []
def _wrapped(state, q, k, v, beta, alpha):
  # these are lazy intermediates inside _attention's bigger fused graph -- realize them once so both the
  # diagnostic .numpy() calls below AND the real (unchanged) computation see the same materialized values
  state, q, k, v, beta, alpha = (t.realize() for t in (state, q, k, v, beta, alpha))
  _calls.append(1)
  print(f"\n--- gdn_scan_wy call #{len(_calls)}: q/k/v shape={tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)} ---")
  diagnose(state, q, k, v, beta, alpha)
  final_state, out = _orig_gdn_scan_wy(state, q, k, v, beta, alpha)
  final_state, out = final_state.realize(), out.realize()
  out_np = out.float().numpy()  # (B,T,H,V) per gdn_scan_wy's return contract
  H = out_np.shape[2]
  for h in range(H):
    bad_t = np.nonzero(~np.isfinite(out_np[0, :, h, :]).all(axis=-1))[0]
    if len(bad_t): print(f"  gdn_scan_wy's own OUT (pre ssm_norm/out_gate/ssm_out): head {h} non-finite at "
                          f"time positions {bad_t.tolist()} (of {out_np.shape[1]})")
  return final_state, out

def run(impl: int, head_groups: int, label: str) -> tuple[int, int, np.ndarray, np.ndarray]:
  block, x = make_real_block()
  x = x.reshape(1, x.shape[0], DIM).float()
  print(f"\n=== {label}: GDN_SCAN_IMPL={impl} GDN_HEAD_GROUPS={head_groups} T={x.shape[1]} ===")
  with Context(GDN_SCAN_IMPL=impl, GDN_CHUNK=32, GDN_HEAD_GROUPS=head_groups):
    x_norm = block.attn_norm(x)
    block._init_state(x_norm)
    out = block._attention(x_norm, 0).realize()
  n_nan_out, n_inf_out = nan_inf_report("output", out)
  n_nan_state, n_inf_state = nan_inf_report("recurrent_state", block.recurrent_state)
  print(f"  VERDICT: {'NaN-FLOOD' if n_nan_state or n_inf_state else 'clean'} "
        f"(recurrent_state NaN={n_nan_state} Inf={n_inf_state}, total elements={block.recurrent_state.numel()})")
  return n_nan_state, n_inf_state, out.float().numpy(), block.recurrent_state.float().numpy()

if __name__ == "__main__":
  M.gdn_scan_wy = _wrapped
  print("############ diagnostic pass: GDN_HEAD_GROUPS=1 (one call, all 48 heads, easy head indexing) ############")
  n_nan_wy, _, wy_out_g1, wy_state_g1 = run(GDN_SCAN_WY, 1, "WY (diagnostic, G=1)")
  M.gdn_scan_wy = _orig_gdn_scan_wy  # stop diagnosing -- confirm the loop stays clean, and check auto (G=2, production) too

  print("\n############ confirmation passes (no diagnostics, matching bug1_gguf_ab.py's production config) ############")
  n_nan_loop_g1, _, loop_out_g1, loop_state_g1 = run(GDN_SCAN_LOOP, 1, "LOOP (G=1)")
  n_nan_wy_auto, _, wy_out_auto, wy_state_auto = run(GDN_SCAN_WY, 0, "WY (AUTO head groups = production default, G=2 for 48 heads)")
  n_nan_loop_auto, _, loop_out_auto, loop_state_auto = run(GDN_SCAN_LOOP, 0, "LOOP (AUTO head groups)")

  print("\n############ WY vs LOOP tolerance (test_gdn_scan_parity.py's own convention: rtol=atol=1e-4) ############")
  for label, wy_a, loop_a in (("G=1 output", wy_out_g1, loop_out_g1), ("G=1 recurrent_state", wy_state_g1, loop_state_g1),
                              ("auto output", wy_out_auto, loop_out_auto), ("auto recurrent_state", wy_state_auto, loop_state_auto)):
    close = np.allclose(wy_a, loop_a, rtol=1e-4, atol=1e-4)
    print(f"  {label}: max|diff|={np.abs(wy_a - loop_a).max():.3e}  allclose(rtol=atol=1e-4)={close}")

  print("\n############ SUMMARY (post-T4.73c-fix expectation: all four rows clean) ############")
  print(f"  WY   G=1:   recurrent_state NaNs = {n_nan_wy}")
  print(f"  LOOP G=1:   recurrent_state NaNs = {n_nan_loop_g1}")
  print(f"  WY   auto:  recurrent_state NaNs = {n_nan_wy_auto}")
  print(f"  LOOP auto:  recurrent_state NaNs = {n_nan_loop_auto}")
  ok = n_nan_wy == 0 and n_nan_loop_g1 == 0 and n_nan_wy_auto == 0 and n_nan_loop_auto == 0
  print(f"  {'FIX CONFIRMED' if ok else 'STILL BROKEN'}: WY matches LOOP (both clean) at the single-block level.")
  if not ok:
    print("  (pre-T4.73c-fix, WY showed exactly 16384 NaNs = one full head (head 42/48) here -- see FIXNOTES_T473C.md)")
