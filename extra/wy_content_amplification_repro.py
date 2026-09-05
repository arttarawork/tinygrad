#!/usr/bin/env python3
"""T4.73d: seconds-fast, CPU-only repro + bisection for the WY chunked GDN scan's per-chunk STATE
AMPLIFICATION on real qwen3.8-27B blk.44 content (the "content x length residual" left after T4.73c fixed
the single-chunk head-42 underflow -- see FIXNOTES_T473C.md for that earlier, different bug, and
FIXNOTES_T473D.md for this one's full diagnosis).

Real-hardware finding (pooled METAL+NV, GDN_SCAN_IMPL=2, one specific paragraph repeated many times):
blk44's recurrent_state amax grows smoothly chunk over chunk (2.27 -> 31.3 -> 217 -> 664 -> 3744 -> 21117 -> ...
-> 1.41e38 -> Inf by chunk 132 of a longer prompt) while GDN_SCAN_IMPL=1 (the loop, same math run
sequentially) keeps the SAME content's amax flat at ~0.65-0.71 -- a contraction, as the per-token delta rule
recurrence always is for beta in (0,1). This script loads the REAL captured inputs for the chunk where that
growth was first caught red-handed with fully finite operands (T4.73d PHASE 2's WY_DUMP_AMAX=5 trigger on
model.py's generate(), chunk 3 of the x4-repeated-paragraph prompt: state_in absmax 31.3 -- chunk 2's
output -- in, recurrent_state_out absmax 217.1 -- chunk 3's real, hardware-observed output -- out, a ~x7
gain in one chunk) and reproduces the SAME ~x7 gain from a plain CPU call to gdn_scan_wy, with NO device,
JIT, or full-model machinery involved.

Bisection mechanism (see the printed output below and FIXNOTES_T473D.md): head 25 is the only head with any
real magnitude in this dump (beta 0.87-0.99, real key cosine similarity ~0.96 mean/0.999 max -- extremely
near-collinear, exactly the "same paragraph repeated" content this task named as the leading suspect).
Comparing every gdn_scan_wy intermediate in float64 (exact reference) against float32 (production dtype)
isolates the DIVERGENCE to exactly one term: _gdn_tri_inverse's Neumann-series DOUBLING computes an inverse
matrix with max|.|~4.2 where the true (float64, and forward-substitution) answer is max|.|~1.0 -- a ~4x
error from float32 catastrophically canceling an intermediate power (n_pow) that transiently reaches
~6.8e7 before nilpotency (n^32==0 exactly) forces it back down to the true, bounded answer. Every other
intermediate (a_bar, m, rhs) matches float64 to ~1e-6, i.e. plain float32 rounding, not an instability.

T4.73d's fix (tinygrad/llm/model.py's _gdn_tri_inverse) replaces the doubling algorithm with block-recursive
halving -- exact in infinite precision (same as doubling), but never forms an intermediate larger than the
true (bounded) answer, so there is no catastrophic cancellation to lose float32 precision to. This script
demonstrates BOTH algorithms side by side (the retired doubling code is inlined below, not reachable from
model.py any more) against the real dumped inputs, and against a float64 forward-substitution ground truth.

Needs the dump from a WY_DUMP_AMAX hardware/local run (T4.73d PHASE 2, model.py's WY_TRACE=2 WY_DUMP_DIR=...
WY_DUMP_AMAX=...): extra/t473d_payloads/dumps/preoverflow_chunk3_blk44.safetensors (not committed -- a
~9MB, run-specific artifact, like extra/blk0_real.safetensors before it; regenerate by re-running the traced
prefill, see FIXNOTES_T473D.md). CPU/NULL only throughout, no GGUF or pooled-server dependency (unlike
extract_blk0_real.py's family, this dump is the WY_TRACE instrumentation's OWN output, not a fresh GGUF read).

Run: PYTHONPATH=. <venv>/bin/python extra/wy_content_amplification_repro.py
"""
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from tinygrad.llm.model import gdn_scan_wy, _gdn_tri_inverse

DUMP = "extra/t473d_payloads/dumps/preoverflow_chunk3_blk44.safetensors"

def _gdn_tri_inverse_doubling_RETIRED(m: Tensor) -> Tensor:
  """The pre-T4.73d _gdn_tri_inverse, kept here ONLY for this repro's side-by-side comparison -- see
  tinygrad/llm/model.py's current docstring and FIXNOTES_T473D.md for why it was replaced."""
  c = m.shape[-1]
  idx = Tensor.arange(c, dtype=dtypes.int32).clone(m.device)
  eye = (idx.reshape(c, 1) == idx.reshape(1, c)).float()
  p, n_pow = Tensor.zeros_like(m) + eye, -m
  for _ in range((c - 1).bit_length()):
    p = p + n_pow @ p
    n_pow = n_pow @ n_pow
  return p

def forward_sub_f64(m64: np.ndarray) -> np.ndarray:
  """Textbook-stable ground truth for (I+m)^-1 (unit-lower-triangular solve), float64, used only as an
  independent reference -- never as the production fix (see _gdn_tri_inverse's docstring for why: it's the
  same O(C) sequential-steps shape the T4.69a report moved away from for kernel-count reasons)."""
  n = m64.shape[0]
  A = np.eye(n) + m64
  x_cols = np.zeros((n, n))
  for col in range(n):
    b = np.eye(n)[:, col]
    x = np.zeros(n)
    for row in range(n): x[row] = b[row] - A[row, :row] @ x[:row]
    x_cols[:, col] = x
  return x_cols

def loop_ref(state0: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
  """Per-token delta-rule recurrence, matching GatedDeltaNetBlock._attention's run_scan loop branch exactly
  (model.py) -- the ground truth every gdn_scan_wy chunked form must agree with."""
  S = state0.copy()
  for i in range(k.shape[0]):
    ki, vi, bi, ai = k[i], v[i], beta[i], alpha[i]
    S = ai * (S @ (np.eye(k.shape[1]) - bi * np.outer(ki, ki))) + bi * np.outer(vi, ki)
  return S

if __name__ == "__main__":
  d = {name: t.to("CPU") for name, t in safe_load(DUMP).items()}
  state_in, q, k, v, beta, alpha = (d[n].float() for n in ("state_in", "q", "k", "v", "beta", "alpha"))
  rec_out_hw = d["recurrent_state_out"].numpy()
  print(f"loaded {DUMP}: state_in absmax={np.abs(state_in.numpy()).max():.6g}  "
        f"hardware recurrent_state_out absmax={np.abs(rec_out_hw).max():.6g}")

  heads = [h for h in range(state_in.shape[1]) if np.abs(state_in.numpy()[0, h]).max() > 1 or np.abs(rec_out_hw[0, h]).max() > 1]
  print(f"heads with any real magnitude this chunk: {heads}")
  H = heads[0]
  k25 = k.numpy()[0, H]
  sims = k25 @ k25.T
  np.fill_diagonal(sims, 0)
  print(f"head {H}: beta range [{beta.numpy()[0,H].min():.4g},{beta.numpy()[0,H].max():.4g}]  "
        f"key pairwise cosine similarity: mean|.|={np.abs(sims).mean():.4g} max|.|={np.abs(sims).max():.4g} "
        f"(near-collinear -- this content is one paragraph repeated)")

  print("\n=== full-block gdn_scan_wy (CURRENT, halving) vs the retired doubling code vs the loop ===")
  final_fixed, _ = gdn_scan_wy(state_in, q, k, v, beta, alpha)
  final_fixed = final_fixed.realize().numpy()
  print(f"  current (halving) gdn_scan_wy: absmax={np.abs(final_fixed).max():.6g}  "
        f"head {H} absmax={np.abs(final_fixed[0, H]).max():.6g}")

  import tinygrad.llm.model as M
  saved = M._gdn_tri_inverse
  M._gdn_tri_inverse = _gdn_tri_inverse_doubling_RETIRED
  final_doubling, _ = gdn_scan_wy(state_in, q, k, v, beta, alpha)
  final_doubling = final_doubling.realize().numpy()
  M._gdn_tri_inverse = saved
  print(f"  retired (doubling) gdn_scan_wy:  absmax={np.abs(final_doubling).max():.6g}  "
        f"head {H} absmax={np.abs(final_doubling[0, H]).max():.6g}  "
        f"(real hardware observed {np.abs(rec_out_hw[0, H]).max():.6g} on NV/METAL -- same ballpark, device/"
        f"kernel-fusion rounding differences shift the EXACT value in an ill-conditioned computation like this)")

  loop_state = loop_ref(state_in.numpy()[0, H].astype(np.float64), q.numpy()[0, H].astype(np.float64),
                         k.numpy()[0, H].astype(np.float64), v.numpy()[0, H].astype(np.float64),
                         beta.numpy()[0, H].astype(np.float64), alpha.numpy()[0, H, :, 0].astype(np.float64))
  print(f"  loop (ground truth) from the SAME state_in: absmax={np.abs(loop_state).max():.6g}")

  print(f"\n=== isolating _gdn_tri_inverse on head {H}'s real m = beta * strictly_lower(k @ k.T) ===")
  T = k.shape[2]
  idx = np.arange(T)
  strict_lower = (idx[None, :] < idx[:, None]).astype(np.float64)
  m64 = beta.numpy()[0, H][:, None].astype(np.float64) * ((k25.astype(np.float64) @ k25.astype(np.float64).T) * strict_lower)
  ref = forward_sub_f64(m64)
  inv_halving = _gdn_tri_inverse(Tensor(m64.reshape(1, 1, T, T).astype(np.float32))).numpy().reshape(T, T)
  inv_doubling = _gdn_tri_inverse_doubling_RETIRED(Tensor(m64.reshape(1, 1, T, T).astype(np.float32))).numpy().reshape(T, T)
  print(f"  max|m|={np.abs(m64).max():.6g}")
  print(f"  forward-substitution (f64 ground truth): max|inv|={np.abs(ref).max():.6g}")
  print(f"  halving (CURRENT fix, f32):               max|inv|={np.abs(inv_halving).max():.6g}  "
        f"max|diff vs f64 ref|={np.abs(inv_halving.astype(np.float64)-ref).max():.3e}")
  print(f"  doubling (RETIRED, f32):                   max|inv|={np.abs(inv_doubling).max():.6g}  "
        f"max|diff vs f64 ref|={np.abs(inv_doubling.astype(np.float64)-ref).max():.3e}  <- catastrophic cancellation")

  print("\n=== VERDICT ===")
  ok = np.abs(final_fixed[0, H]).max() < 5 and np.allclose(final_fixed[0, H], loop_state, rtol=1e-2, atol=1e-2)
  print(f"  {'FIX CONFIRMED' if ok else 'STILL BROKEN'}: current gdn_scan_wy matches the loop from this real, "
        f"previously-amplifying state_in (was {np.abs(final_doubling[0,H]).max():.4g} before the fix).")
