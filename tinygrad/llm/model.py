from __future__ import annotations
import enum, functools, itertools, math, pathlib, re, sys, time
from dataclasses import dataclass, replace
from typing import Callable, cast, TYPE_CHECKING
if TYPE_CHECKING: import numpy as np  # T4.65 CI fix: tinygrad's core stays numpy-free at import time -- the minimal
# CI lanes (Test LLM, SPEC=2) have no numpy installed. The sampled speculative path imports it lazily inside the
# functions that actually need it; annotations are lazy via `from __future__ import annotations`.
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.dtype import DType
from tinygrad.device import Buffer
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.kernels.nv import gated_delta_prefill as nv_gated_delta_prefill, nv_custom_kernels_supported, nv_decode_kernel_supported, \
  gdn_decode as nv_gdn_decode, GDN_DECODE_TOKENS
from tinygrad.llm.kernels.nv_quant import nv_quant_supported, NV_QUANT_TYPES, add_rmsnorm_q8, register_fused_q8, split_k_partials, ffn_act_q8, \
  clear_step_caches
from tinygrad.llm.kernels.nv_dense import nv_f16_gemv_raw
from tinygrad.llm.kernels.nv_attn import MAX_PREFILL_ROWS, nv_decode_attention, nv_decode_attention_ok, nv_prefill_attention, nv_verify_attention
from tinygrad.llm.kernels.nv_gdn_prep import gdn_prefill as nv_gdn_prefill, gdn_post as nv_gdn_post, GDN_NV_PREP
from tinygrad.llm.kernels.nv_attn_prep import NV_ATTN_PREP, attn_prep, attn_prep_geometry_ok
from tinygrad.llm.gguf import gguf_load, ggml_data_to_tensor
from tinygrad.uop.ops import resolve, Ops
from tinygrad.helpers import ContextVar, DEBUG, GlobalCounters
import traceback

# T4.55: prefill chunk width for recurrent models on devices without a fused scan kernel (see generate). 0 = auto: 32 on the GPU
# backends it was measured on -- METAL (qwen3.5:0.8b: 77 -> 168 tok/s no-BEAM, 99 -> 346 BEAM'd) and NV (the METAL+NV pooled
# qwen3.6-35B Q8_0: 46 -> 158-173 tok/s BEAM'd, decode unchanged) -- and 1 (the pre-T4.55 one-token-per-step prefill) elsewhere:
# the unrolled 32-step scan is one huge kernel, and x86 clang 18 crashes compiling it (CI's CPU "Test LLM" job on qwen3.5:0.8b).
# 64 falls off a cliff even on METAL (28 tok/s). Set GDN_CHUNK explicitly to override either way.
# T4.68: 64 is explicit-only (never auto-selected above); CPU scan-parity now covers both geometries (test_gdn_scan_parity.py).
GDN_CHUNK = ContextVar("GDN_CHUNK", 0)
# T4.124a: serving prefill width -- the default (32) reproduces every pre-T4.124a chunk_size/gdn_chunk_for call byte-identically
# (min(32, MAX_PREFILL_ROWS=64) == 32). Raise it (e.g. 64/128/256) only on devices whose fused NV scan and prefill-attention
# kernel (nv_attn.py, gated on MAX_PREFILL_ROWS below) both stay linear in chunk width; gdn_chunk_for below gates that.
# generate()/prefill_only() fall back to this whenever their own caller doesn't pass an explicit chunk_size.
PREFILL_CHUNK = ContextVar("PREFILL_CHUNK", 32)
# T5.5: prefill chunk width for IMAGE prompts. The image-prompt jit family is a second capture whose planned arena is the attention
# score pipeline (2 x (B,H,T,Tk_max) fp32 after T5.6) -- it scales with the chunk width, and on the pooled qwen3.6-35B @192k every
# extra prefill family costs ~0.78 GB of the 3090's headroom at width 32. 16 halves that for the few hundred visual tokens of a request.
VISION_CHUNK = ContextVar("VISION_CHUNK", 16)
def gdn_chunk_for(device:str|tuple[str, ...]|None) -> int:
  if GDN_CHUNK.value > 0: return GDN_CHUNK.value
  # T4.124a: the fused NV scan has no width cliff (kernels/nv.py), and nv_attn.py's prefill-attention kernel handles up to
  # MAX_PREFILL_ROWS rows regardless of M -- cap PREFILL_CHUNK there so a wider request never silently falls off the NV
  # attention kernel onto the generic path (which OOMs at long context, see nv_attn.py / model.py's _nv_attn).
  if nv_custom_kernels_supported(device): return min(PREFILL_CHUNK.value, MAX_PREFILL_ROWS)
  dev = (device[0] if isinstance(device, tuple) else device) or Device.DEFAULT
  return 32 if dev.split(":")[0] in ("METAL", "NV", "CUDA") else 1

# T4.62: on wide-head geometries (e.g. qwen3.8-27B, num_v_heads=48) the unrolled per-t scan in
# GatedDeltaNetBlock._attention's else-branch -- one python loop building the whole chunk's graph, fused into
# a single kernel by its final .contiguous() -- lowers to more UOps than BEAM_UOPS_MAX (codegen/opt/search.py)
# for every BEAM candidate; narrower geometries (e.g. qwen3.6-35B's 32 heads) fit fine. The scan has no
# cross-head dependency (each head's (V,K) state slice evolves independently), so split it into G sequential
# head groups, each narrow enough to lower under the cap on its own -- see gdn_headgroup_evidence.py.
GDN_HEAD_GROUPS = ContextVar("GDN_HEAD_GROUPS", 0)
def gdn_head_groups_for(num_v_heads:int) -> int:
  if GDN_HEAD_GROUPS.value > 0: return GDN_HEAD_GROUPS.value
  return -(-num_v_heads // 32)  # auto: smallest G with ceil(num_v_heads/G) <= 32 == ceil(num_v_heads/32)

# T4.69a: chunkwise WY-form scan (gdn_scan_wy, below) as a device-agnostic alternative to
# GatedDeltaNetBlock._attention's else-branch per-token loop -- same math (reference: that loop, and
# kernels/amd.py's gated_delta_prefill, whose per-step algebra was checked against it -- see the T4.69a
# report), expressed as O(log T_pad) matmul-shaped ops instead of a T_pad-deep python-unrolled chain.
# 0 (auto, default): the loop -- flipping the default to WY is a later, measured decision (perf evidence:
# extra/gdn_wy_evidence.py); auto must stay byte-identical to pre-T4.69a behavior. 1: force the loop.
# 2: force gdn_scan_wy. Selection happens inside GatedDeltaNetBlock._attention's run_scan closure, so a
# GDN_HEAD_GROUPS-sliced call composes with it for free (each group independently dispatches).
GDN_SCAN_IMPL = ContextVar("GDN_SCAN_IMPL", 0)
GDN_SCAN_LOOP, GDN_SCAN_WY = 1, 2
# T4.69c (2026-09-04): auto (0) now selects the chunkwise WY form for chunked prefill -- decode/T_pad==1 still takes the
# loop via run_scan's T4.69b gate. Flipped only after T4.73c (a_bar underflow) + T4.73d (tri-inverse cancellation) made WY
# greedy-identical to the loop on real weights end-to-end on hardware; measured 27 tok/s vs 22-23 at 5.9k on qwen3.8-27B.
# GDN_SCAN_IMPL=1 restores the loop everywhere.
def gdn_scan_impl_for() -> int: return GDN_SCAN_IMPL.value if GDN_SCAN_IMPL.value > 0 else GDN_SCAN_WY
# T4.69b: test-introspection only, never read inside the model itself -- run_scan (below) appends the impl it
# actually dispatched to (post single-step gate), so a test can assert WY vs LOOP was chosen without reaching
# into a kernel count. Read via gdn_last_scan_impl[-1].
gdn_last_scan_impl: list[int] = []

def _gdn_tri_inverse(m:Tensor) -> Tensor:
  """(I+m)^-1 for a strictly-lower-triangular (..., C, C) m (zero diagonal and above), via block-recursive
  halving: split m into 2x2 blocks [[M11,0],[M21,M22]] (M11, M22 strictly-lower-triangular on their own,
  M21 the full below-diagonal coupling block), recurse on each half, and assemble
    (I+m)^-1 = [[Inv11, 0], [-Inv22 @ M21 @ Inv11, Inv22]]   (Inv11 := (I+M11)^-1, Inv22 := (I+M22)^-1)
  which is exact for any unit-lower-triangular matrix (standard block-triangular inverse identity), by
  induction down to the C=1 base case ((I+0)^-1 = I, a scalar 1). log2(C)-deep recursion; C need not be a
  power of 2 (odd sizes just split as evenly as possible each level -- gdn_scan_wy's T_pad is always a
  concrete int, see the assert below, but never assumed to be a power of 2).

  T4.73d: this REPLACES a Neumann-series DOUBLING implementation (P_2k = P_k + n^k@P_k, n := -m, using
  n^C == 0 exactly to terminate in ceil(log2 C) steps on the one full-size matrix, no shrinking submatrices
  or concats -- see FIXNOTES_T473D.md for the retired code and the full derivation) that is EXACT in
  infinite precision but catastrophically ill-conditioned in float32 once m's entries get large (beta close
  to 1, as trained heads with near-collinear/repeated keys within a chunk produce): repeatedly squaring
  n (n_pow = n_pow @ n_pow) drives its magnitude up by orders of magnitude EACH doubling step (measured on
  a real qwen3.8-27B blk44 chunk: max|n_pow| 28 -> 3.4e3 -> 1.4e6 -> 6.8e7 over 4 steps) even though the
  running partial sum p, and the final exact answer, both stay near-unity (the huge n^k terms cancel back
  down to ~1) -- float32's ~7 significant decimal digits are nowhere near enough to represent that
  cancellation, producing a WRONG (too-large) inverse (measured: max|inv| ~4.2, true value ~1.0) that then
  amplifies every downstream chunk's carried recurrent_state by a real, hardware-observed factor (~x7 that
  chunk, compounding across chunks to fp32 overflow). Block-recursive halving never forms an intermediate
  matrix larger than the true (bounded, since (I+m)^-1 for unit-lower-triangular m is itself well-scaled)
  answer -- Inv11/Inv22 are themselves valid, bounded triangular inverses by induction, and Inv21 is a
  product of bounded matrices, so there is no large-magnitude intermediate to cancel away. Verified on the
  same real blk44 chunk (test_gdn_scan_parity.py's TestGDNScanHighBetaCollinearAmplification): matches a
  float64 forward-substitution reference to ~1.5e-7 (float32 machine precision), vs. the old doubling
  code's ~4.0 absolute error on a true value of ~1.0.

  This was ALREADY the very first (T4.69a) draft, dropped at the time only because it lowers to more UOps
  (O(C) total recursive-call nodes, each its own small matmuls/concats, vs. doubling's O(log2 C) big
  uniform matmuls) -- a real but strictly smaller concern than the correctness bug above; not established
  broken until this task, so no defensive fallback is needed for the doubling path -- see FIXNOTES_T473D.md
  for the full before/after kernel-count discussion if that regresses BEAM/compile time in practice."""
  c = m.shape[-1]
  assert isinstance(c, int), "gdn_scan_wy always builds m from a concrete (padded-to-T_pad) chunk size"
  if c == 1: return Tensor.ones_like(m)  # (I+0)^-1 = I -- m's diagonal (hence every 1x1 block) is exactly 0
  lo = c // 2
  m11, m22, m21 = m[..., :lo, :lo], m[..., lo:, lo:], m[..., lo:, :lo]
  inv11, inv22 = _gdn_tri_inverse(m11), _gdn_tri_inverse(m22)
  inv21 = -(inv22 @ m21 @ inv11)
  top = inv11.cat(Tensor.zeros_like(m[..., :lo, lo:]), dim=-1)     # (..., lo, C): [Inv11 | 0]
  bottom = inv21.cat(inv22, dim=-1)                                 # (..., C-lo, C): [Inv21 | Inv22]
  return top.cat(bottom, dim=-2)                                    # (..., C, C)

def gdn_scan_wy(state:Tensor, q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor) -> tuple[Tensor, Tensor]:
  """Chunkwise WY-form gated delta rule scan: the whole (B,H,T,*) chunk in one shot via matmuls plus one
  (T,T) triangular solve, instead of _attention's else-branch per-token loop closure (T-deep python-unrolled
  chain). Both compute the same per-token recurrence (checked against kernels/amd.py's gated_delta_prefill,
  which implements it as a hardware loop with an algebraically-reassociated output term, NOT this chunked
  form -- see the T4.69a report for why "port the algorithm" ended up meaning "derive the standard
  chunked/WY delta-rule parallelization instead of amd.py's structural shape", with amd.py and the loop
  supplying the reference MATH only): per token i (S_{-1} = state entering the call),
    S_i = diag(alpha_i) @ S_{i-1} @ (I - beta_i k_i k_i^T) + beta_i v_i k_i^T,   out_i = S_i @ q_i

  Derivation (chunk of C=T tokens, S0 := state entering the chunk): let Abar_i = prod_{j<=i} alpha_j
  (row-wise, i.e. per value-channel, cumulative decay through step i inclusive). Substituting the
  decay-normalized state S~_i := diag(Abar_i)^-1 @ S_i turns the GATED recurrence into the plain (ungated)
  delta rule in (S~, v~ := v/Abar): S~_i = S~_{i-1} + beta_i (v~_i - S~_{i-1} k_i) k_i^T. Unrolling
  S~_i = S0 + sum_{j<=i} u_j k_j^T turns that into a strictly-lower-triangular linear system for the C
  "pseudo-value" rows u_j (u_i + beta_i sum_{j<i} (k_j.k_i) u_j = beta_i(v~_i - S0 k_i)), i.e. in matrix form
    (I + Beta @ tril(K K^T, -1)) @ U = Beta @ (V~ - K @ S0^T)      -- solved via _gdn_tri_inverse above.
  Output and final state then fold the decay back in:
    O   = Abar * (Q @ S0^T + tril(Q K^T, 0) @ U)                  -- (C,V), row i = out_i
    S_C = Abar[-1] * (S0 + U^T @ K)                                -- state leaving the chunk
  `alpha` may be a per-head scalar (last dim 1, broadcasts across V -- non-kda) or per-value-channel (last
  dim V -- kda). The tilde form above (dividing v by the cumulative Abar) is used AS-IS for kda, unchanged
  since T4.69a -- test_attention.py's test_varied_chunk_sizes_match_decode(kda=True) still exercises it.

  T4.73c: for non-kda, dividing by Abar (a product accumulated from the CHUNK START) is numerically unsafe
  at real trained weights -- a fast-decaying head's Abar can underflow past float32's range within one
  chunk, and multiplying back by Abar afterward does not cancel a division-by-0/overflow (see
  FIXNOTES_T473C.md for the full real-weight diagnosis: qwen3.8-27B blk.0 head 42/48). Fixed below by an
  EXACT reformulation (not an approximation -- the new pseudo-value w relates to the old u by w_j = Abar_j
  * u_j pointwise) that never divides by Abar: every place decay appears is either a bounded PAIRWISE ratio
  exp(G_i-G_j) for i>=j (in (0,1], since the cumulative log-decay G is non-increasing -- masked to exactly
  0 for invalid (j>i) pairs by masking the exponent BEFORE exp, so it can never overflow either), or the
  absolute exp(G_i) used strictly as a MULTIPLIER (safe to underflow to 0 -- that just means "fully
  decayed", never a divisor). Only non-kda's scalar-per-head alpha makes this ratio foldable into a plain
  (T,T) matrix shared across V channels; kda's per-channel Abar would need a (T,T,V) coupling/tri-solve
  (untested cost, and kda is not established broken), so that path is untouched below.

  Shapes (all float32; T_pad is the WHOLE chunk -- one WY block, no further sub-chunking needed since
  GDN_CHUNK's practical ceiling (32) is already tiny for a log-depth triangular solve; padded steps are
  beta=0/alpha=1 no-ops exactly as in the loop, so trailing padding never perturbs O or S_C for real tokens):
  state (B,H,V,K); q,k (B,H,T,K); v (B,H,T,V); beta (B,H,T); alpha (B,H,T,V|1). Returns
  (final_state (B,H,V,K), stacked_outputs (B,T,H,V)) -- T-before-H, matching the loop's return contract
  (outs[0].stack(*outs[1:], dim=1))."""
  T = q.shape[2]
  idx = Tensor.arange(T, dtype=dtypes.int32).clone(q.device)
  row, col = idx.reshape(T, 1), idx.reshape(1, T)
  strict_lower_b, lower_incl_b = col < row, col <= row

  if alpha.shape[-1] == 1:  # non-kda: T4.73c safe path (see docstring addendum + FIXNOTES_T473C.md)
    g = alpha[..., 0].float().maximum(1e-30).log()                      # (B,H,T) per-step log-decay, always finite
    # (this floor is per-STEP, not cumulative -- only engages if a single raw step's alpha underflows to
    # exact float32 0 on its own, which real weights here never do (observed min 5.44e-4); a defensive
    # floor, not the fix -- the fix is never dividing by the CUMULATIVE product at all, see below)
    G = g.cumsum(axis=2)                                                # (B,H,T) cumulative log-decay through step i
    # NOT clamped to <=0: alpha (hence diff=G_i-G_j for the valid j<=i region) is a GATE, not guaranteed
    # <=1 -- e.g. this file's own random-init tests draw ssm_a from randn() (can be positive, i.e. genuine
    # per-step growth, alpha>1) -- clamping would silently cap real growth and diverge from the loop, which
    # applies no such cap. Only the INVALID (j>i) region needs forcing to exactly 0, via the -1e30 sentinel
    # below (masking BEFORE exp, not clamping diff), so _gdn_tri_inverse's bit-exact triangularity holds.
    diff = G.unsqueeze(-1) - G.unsqueeze(-2)                             # (B,H,T,T) G_i-G_j, valid region only
    ratio_incl = lower_incl_b.where(diff, -1e30).exp()                  # (B,H,T,T) exp(G_i-G_j), j<=i; else exactly 0
    ratio_strict = strict_lower_b.where(diff, -1e30).exp()              # same, j<i only (feeds the tri-solve)
    a_bar = G.exp().unsqueeze(-1)                                       # (B,H,T,1) Abar -- ONLY ever a multiplier below
    kkt = (k @ k.transpose(-1, -2)) * ratio_strict                      # (B,H,T,T), decay folded in directly
    m = beta.unsqueeze(-1) * kkt                                        # row i scaled by beta_i (Beta @ T)
    rhs = beta.unsqueeze(-1) * (v - a_bar * (k @ state.transpose(-1, -2)))  # (B,H,T,V) -- v used RAW, no division
    w = _gdn_tri_inverse(m) @ rhs                                       # (B,H,T,V) pseudo-values (undecayed form)
    qk = (q @ k.transpose(-1, -2)) * ratio_incl                         # (B,H,T,T), decay folded in directly
    out = a_bar * (q @ state.transpose(-1, -2)) + qk @ w                # (B,H,T,V) -- a_bar multiplies only the S0 term
    tail_ratio = ratio_incl[:, :, T - 1:T, :].transpose(-1, -2)         # (B,H,T,1): exp(G_{T-1}-G_j) per row j
    final_state = a_bar[:, :, T - 1, :].unsqueeze(-1) * state + (tail_ratio * w).transpose(-1, -2) @ k  # (B,H,V,K)
    return final_state, out.transpose(1, 2)                            # (B,T,H,V)

  # kda: unchanged since T4.69a (per-value-channel alpha; not established broken -- see docstring addendum)
  strict_lower, lower_incl = strict_lower_b.float(), lower_incl_b.float()
  a_bar = alpha.cumprod(axis=2)                                          # (B,H,T,V|1), decay through step i
  v_tilde = v / a_bar                                                    # decay-normalized values
  kkt = (k @ k.transpose(-1, -2)) * strict_lower                         # (B,H,T,T), strictly lower
  m = beta.unsqueeze(-1) * kkt                                           # row i scaled by beta_i (Beta @ T)
  rhs = beta.unsqueeze(-1) * (v_tilde - k @ state.transpose(-1, -2))     # (B,H,T,V)
  u = _gdn_tri_inverse(m) @ rhs                                          # (B,H,T,V) pseudo-values
  qk = (q @ k.transpose(-1, -2)) * lower_incl                            # (B,H,T,T), lower incl. diagonal
  out = a_bar * (q @ state.transpose(-1, -2) + qk @ u)                   # (B,H,T,V)
  final_state = a_bar[:, :, T-1, :].unsqueeze(-1) * (state + u.transpose(-1, -2) @ k)  # (B,H,V,K)
  return final_state, out.transpose(1, 2)                                # (B,T,H,V)

# T4.63: qwen3.5-family GGUFs carry a DeepSeek-style MTP ("nextn") block beyond the main num_blocks
# (nextn_predict_layers, already excluded from num_blocks -- see from_gguf). 0 (default): today's
# behavior, byte-identical -- gguf.py parses the block, from_gguf drops it (unused-weights warning).
# 1: build + load it as Transformer.mtp_head (see MTPHead) -- loading + forward only; T4.64 wires it
# into speculative decoding.
MTP = ContextVar("MTP", 0)

# T4.66b: per-request speculative-decode instrumentation for speculative_generate -- 0 (default): zero
# overhead, nothing recorded, nothing printed (only a cheap `if` guard per outer iteration). 1: accumulate an
# accept-length histogram (m -> iteration count) and a drafts-per-emitted-token ratio, printed once when the
# generator's main loop ends -- either it runs out (max_context) or the caller closes/abandons it (a request
# ending, GeneratorExit) -- via a try/finally wrapping the loop, so both paths are covered without serve.py
# needing to know anything about this. This is the knob the hardware run uses to find the best `k` for a
# given draft-head quality: a low average accept length means k is too big (wasted draft compute), the
# histogram's shape shows whether raising/lowering k would help.
SPEC_STATS = ContextVar("SPEC_STATS", 0)

# T4.66c: per-iteration wall-clock PHASE breakdown for speculative_generate -- 0 (default): zero overhead, not
# even a time.perf_counter() call, just one cheap `if` per outer iteration (same contract as SPEC_STATS above).
# 1: prints one line per outer iteration with wall-clock ms for each phase below. Built to name where real
# wall-clock goes on actual hardware (METAL+NV pooled, k=3: ~4.9s/iter observed) that a dispatch-count proxy
# can't see -- see SPEC_PROFILE_NOTES.md for the full first-principles audit this instrumentation exists to test.
#
# SYNC SEMANTICS (read before trusting a number here -- tinygrad dispatch is async, so wall-clock measured at a
# non-sync point is queuing time, not execution time; every phase below says which kind it is):
#   draft_ms           DISPATCH-ONLY since T4.132 (both modes): the draft-chain replay (Transformer._draft_chain, all k_eff steps and
#                       VERIFY's input in one graph; its first two calls trace/capture); no .tolist()/.numpy()/.item() -- the
#                       drafted ids stay on device, their device time lands in accept_ms.
#   draft_dispatch_ms  == draft_ms since T4.132 (one replay; kept so the line format stays).
#   verify_dispatch_ms DISPATCH-ONLY. The self(...) call that launches the main-model forward (a jit replay once
#                       warm); nothing here calls .tolist()/.numpy()/.item(), so this never actually waits for the
#                       forward to run -- it only measures how long it takes to QUEUE that work.
#   accept_ms          the iteration's ONE REAL SYNC (T4.132): the accept-test jit replay (Transformer._accept_step) and its
#                       readback -- [m, chunk ids, final ids] -- which waits for the draft chain, VERIFY and the accept math to
#                       finish; then the host-side assembly of the accepted ids. The one number that reflects the iteration's
#                       actual device time (draft chain + VERIFY + accept).
#   state_assign_ms    DISPATCH-ONLY, always. Tensor.realize() defaults to wait=False (engine/realize.py's
#                       run_linear), so this SCHEDULES the partial-accept adopt replay (T4.132: spec_track slot m -> the live GDN
#                       state) but does not block -- its device cost lands at the next iteration's accept_ms. 0 at a full accept.
#                       (The MTP cache rewrite runs inside the accept replay since T4.132.)
#   total_ms           Independently measured wall-clock for the whole iteration body (top of the loop to just
#                       before this line prints), NOT computed as the sum of the phases above. Compare the two:
#                       a gap means some host round-trip in the loop escaped the phases above. Also compare
#                       total_ms against an externally-observed per-iteration time (e.g. a bench harness's
#                       wall-clock between emitted tokens) -- that separates "the time is IN this function" from
#                       "the time is somewhere else entirely" (serve.py, HTTP, scheduler contention).
SPEC_TRACE = ContextVar("SPEC_TRACE", 0)

# T4.132 (10-30): what VERIFY produced, per iteration, to stderr -- NaN/inf counts per row of the verify logits, the verify argmax
# ids, the chunk (tok_last + the drafts), the accept length and the emitted ids -- so a card probe shows WHERE garbage enters (a wrong
# draft is harmless; a NaN verify row or an id >= vocab is not). 0 (default): one `if` per iteration, nothing else; 1: one extra host
# sync per iteration (the logits readback), after the loop's own.
SPEC_DEBUG = ContextVar("SPEC_DEBUG", 0)

# T4.136 (10-31): what the draft chain feeds MTP step i+1 as its hidden. 1 (default): step i's own block output (MTPHead.draft_step's x,
# before shared_head_norm) -- DeepSeek-style recursive MTP (vLLM's MTP proposer for the Qwen3-Next family does the same); step 0 still
# takes the main model's hidden. 0: every step reuses the main model's hidden at start_pos-1 (T4.132's chain). Read once, when the
# ("chain", k_eff) family captures (Transformer._draft_chain): replays keep the value that family was captured with -- on the server that
# is warmup's, so an A/B needs a restart per value. Draft quality only: correctness never depends on it.
SPEC_CHAIN_H = ContextVar("SPEC_CHAIN_H", 1)

# T4.142 (10-31): a vocabulary-trimmed output head for the MTP draft steps. "" (default): the draft steps run the owner's full head,
# byte-identical. A path to an .npy of token ids (draft_vocab_32k.npy: 32768 ids covering > 99 % of what the 27B emits): from_gguf
# builds MTPHead.draft_output = the head's rows at those ids in the head's own ggml format (MTPHead.set_draft_vocab) and every draft
# step picks among them -- on the card the full head is 1.2 ms of a ~2.85 ms draft step (~1.1 GB of Q6_K read), a 32k slice ~0.16 ms.
# A token outside the set is never drafted: a wasted draft (VERIFY emits the token), never a wrong one. VERIFY and plain decode keep
# the full head. Read once, at load.
SPEC_DRAFT_VOCAB = ContextVar("SPEC_DRAFT_VOCAB", "")

def spec_check_ids(ids:list[int], vocab:int, where:str) -> None:
  """T4.132 (10-30): an id outside [0, vocab) can only come from NaN logits (an argmax over a NaN row read back as the row width on
  the card) -- never emit it; the server reports the error instead of streaming ids the tokenizer has no bytes for."""
  if (bad := [v for v in ids if not 0 <= v < vocab]):
    raise RuntimeError(f"speculative_generate: {where}: id(s) {bad} outside vocab {vocab} -- NaN logits upstream (SPEC_DEBUG=1 shows the rows)")

def fresh_copy(x:Tensor) -> Tensor:
  """T4.132: x copied into a buffer of its own, for a TinyJit input. A jit input's signature is its view over its base buffer; `.contiguous()`
  folds away on a view that is already buffer-backed (a jit output's full-width slice, a padded tensor that needed no pad), which hands the
  jit a different view structure from one call to the next ("args mismatch") -- or a view whose baked offset the capture freezes."""
  return Tensor.empty(*x.shape, dtype=x.dtype, device=x.device).assign(x)

def kv_cache_dtype() -> DType:
  """Attention/MLA KV cache dtype: fp16 by default (halves the cache that scales with max_context --
  the dominant decode-memory cost). KV_F32=1 reverts to dtypes.default_float, e.g. to isolate an accuracy
  regression. Does NOT apply to GatedDeltaNetBlock's recurrent state -- see its _init_state for why."""
  return dtypes.default_float if getenv("KV_F32", 0) else dtypes.float16

# T6.1: KV_INT8=1 -- int8 KV cache for TransformerBlock's *standard* attention path only (byte-identical to
# today when unset). Out of scope: MLATransformerBlock.cache_k (a single already-compressed latent cache --
# a different quantization problem) and GatedDeltaNetBlock's conv_state/recurrent_state (small O(1)
# accumulators, not the thing that scales with max_context). Independent of KV_F32 -- that flag keeps
# governing cache_k/conv_state as before; it's meaningless for cache_kv once this is on (int8 either way).
# Per-(position, kv-head) symmetric absmax scales, kv_int8_block(head_dim)-wide blocks along head_dim
# (llama.cpp's q8_0 KV granularity), one scale per block for K and one for V -- see TransformerBlock's
# _init_state (allocation) and _attention (the quantize-on-write/dequantize-on-read idiom).
# T4.100: KV_INT4=1 reuses this exact same block width for its own absmax scale ("the int8-style scale
# layout") -- only cache_kv's own per-element packing differs between the two flags, see _attention.
def kv_int8_block(head_dim:int) -> int:
  """Quantization block width along head_dim: 32 (every real model config in this fork has head_dim a
  multiple of 32) when it divides evenly, else the whole head_dim as a single block -- keeps tiny test
  configs (e.g. head_dim=8) exact and reshape-safe without a stricter assert no real model would ever hit."""
  return 32 if head_dim % 32 == 0 else head_dim

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

def rope_inv_freq(dim: int, theta: float = 10000.0, yarn_factor: float = 1.0, yarn_orig_ctx: int = 0, yarn_beta_fast: float = 32.0,
                  yarn_beta_slow: float = 1.0, yarn_attn_factor: float|None = None) -> tuple[Tensor, float]:
  """(inv_freq (dim//2,), attn_scale): the per-frequency-pair rotation rates precompute_freqs_cis's table is built from, split out (T5.3)
  so the image-prompt path (mrope_freqs_cis) rotates with exactly the same numbers."""
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  attn_scale = 1.0
  if yarn_factor > 1.0:
    # YaRN NTK-by-parts frequency interpolation (https://arxiv.org/abs/2309.00071), verified against
    # transformers' _compute_yarn_parameters (modeling_rope_utils.py): low/high bound the "correction
    # range" (in rotary-dim index space) where we blend extrapolated (raw) and interpolated (/factor) freqs.
    def find_dim(num_rot:float) -> float: return (dim * math.log(yarn_orig_ctx / (num_rot * 2 * math.pi))) / (2 * math.log(theta))
    low = float(max(math.floor(find_dim(yarn_beta_fast)), 0))
    high = float(min(math.ceil(find_dim(yarn_beta_slow)), dim // 2 - 1))
    if low == high: high += 0.001
    extrap_factor = 1.0 - ((Tensor.arange(dim // 2) - low) / (high - low)).clamp(0, 1)
    inv_freq = inv_freq / yarn_factor * (1 - extrap_factor) + inv_freq * extrap_factor
    attn_scale = yarn_attn_factor if yarn_attn_factor is not None else (0.1 * math.log(yarn_factor) + 1.0 if yarn_factor > 1 else 1.0)
  return inv_freq, attn_scale

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None, yarn_factor: float = 1.0,
                          yarn_orig_ctx: int = 0, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0,
                          yarn_attn_factor: float|None = None) -> Tensor:
  inv_freq, attn_scale = rope_inv_freq(dim, theta, yarn_factor, yarn_orig_ctx, yarn_beta_fast, yarn_beta_slow, yarn_attn_factor)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * inv_freq.unsqueeze(dim=0)
  return (freqs.cos() * attn_scale).cat(freqs.sin() * attn_scale, dim=-1).clone(device)

def mrope_freqs_cis(pos3:Tensor, config:TransformerConfig) -> Tensor:
  """T5.3: interleaved M-RoPE rows (T, rope_dim) = cos|sin for per-token (t, h, w) positions `pos3` (1, T, 3) int32 -- Qwen3.5/3.6's
  `mrope_interleaved` with section [11, 11, 10] over the 32 pairs of rope_dim 64: frequency pair i rotates by the position of axis i % 3
  (t, h, w). A text token carries (p, p, p) and gets exactly precompute_freqs_cis's row p (same int32 -> float32 product, same cos/sin).
  Computed from the positions, not gathered from the max_context table: nothing to prove in-bounds, no (T, 3, max_context) one-hot."""
  inv_freq, attn_scale = rope_inv_freq(config.rope_dim, config.rope_theta, config.yarn_factor, config.yarn_orig_ctx, config.yarn_beta_fast,
                                       config.yarn_beta_slow, config.yarn_attn_factor)
  n = config.rope_dim // 2
  ang = pos3[0].float().unsqueeze(-1) * inv_freq.reshape(1, 1, n)                                            # (T, 3, n): every axis
  axis = (Tensor.arange(n) % 3).reshape(1, 1, n) == Tensor.arange(3).reshape(1, 3, 1)  # constants, device-free like pairwise_topk's
  ang = axis.where(ang, 0.0).sum(axis=1)                                                                       # (T, n): pair i <- axis i%3
  return (ang.cos() * attn_scale).cat(ang.sin() * attn_scale, dim=-1)

@functools.cache
def positions(n:int, device:str|None=None) -> Tensor: return Tensor.arange(n, dtype=dtypes.int32).clone(device).realize()

def causal_mask(pos:Tensor, T:int|UOp, start_pos:int|UOp, dtype:DType, sliding_window:int=0) -> Tensor:
  """(1,1,T,start_pos+T) additive mask over the KV cache: 0 where query row i may see key j (j <= start_pos+i; and j > start_pos+i-window
  when sliding), -inf elsewhere. Built from slices of a realized position buffer: the Tensor.full(...).triu(...) form it replaces ran
  arange(start_pos+T), and for a symbolic length that lowers to the single-stage cumsum -- an N x N reduce with no inputs, per
  attention layer per prefill chunk (T4.58, 3090: 27 us at N=64, 3.5 ms at 4k, 48 ms at 16k -- the whole long-context prefill
  cost of the pooled qwen3.6-35B). `pos` comes from FFNBlock._positions: max_context + the chunk variable's declared max entries,
  because the SYMBOLIC bound of start_pos+T is (max_context-1)+chunk_size -- past a max_context-sized buffer -- and CHECK_OOB's z3
  check (on in CI) rejects a load it cannot prove in-bounds even though generate() never runs there."""
  d = pos[:start_pos+T].reshape(1, 1, 1, start_pos+T) - (pos[:T] + Tensor(start_pos, device=pos.device)).reshape(1, 1, T, 1)  # key pos - query pos
  mask = (d > 0).where(float("-inf"), 0.0)
  if sliding_window: mask = mask + (d <= -sliding_window).where(float("-inf"), 0.0)
  return mask.cast(dtype)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int, bias:bool=False):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    if bias: self.bias = Tensor.zeros(num_experts, out_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    out = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return out + self.bias[sel] if hasattr(self, 'bias') else out

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  attn_out_bias: bool = False
  router_bias: bool = False
  moe_bias: bool = False
  attn_sinks: bool = False
  sliding_window: int = 0
  sliding_layers: tuple[bool, ...] = ()
  clamp_swiglu: bool = False
  swiglu_limit: float = 7.0
  swiglu_alpha: float = 1.702
  yarn_factor: float = 1.0
  yarn_orig_ctx: int = 0
  yarn_beta_fast: float = 32.0
  yarn_beta_slow: float = 1.0
  yarn_attn_factor: float|None = None
  # T4.70b: FFN tensor-parallel shard spec -- (device, fraction) pairs, ordered, fractions summing to 1.0;
  # () (default) is the byte-identical no-TP path. Set by Transformer.__init__ from device_map's "tp:"
  # segment (see parse_tp_spec) -- never touched by from_gguf's own `config` local (so MTPHead's inner
  # block, built from that object, stays whole-device -- see from_gguf's MTP branch). Lives on
  # TransformerConfig (like max_context, another construction-time-not-architecture knob) rather than as
  # a separate FFNBlock.__init__ parameter because it changes the STRUCTURE FFNBlock.__init__ builds
  # (N shard Linears instead of one), unlike device_map's experts_dev (a pure post-construction placement
  # override that never needed to reach FFNBlock.__init__ at all).
  tp: tuple[tuple[str, float], ...] = ()

# T4.132: the NV decode fusions below (the fused norm/residual/activation kernels, one block per token) serve a concrete run of up to
# this many tokens -- nv_q8_linear's dp4a gemv batch cap (kernels/nv_quant.py: tokens > 4 take the gemm/generic path), the same
# run the GDN decode kernels cover (GDN_DECODE_TOKENS) -- i.e. a decode step (1) or speculative decoding's VERIFY (<= k+1).
NV_SMALL_T = 4

@functools.cache
def spec_k_clamped(k:int) -> int:
  """T4.132: speculative_generate's draft count, capped so VERIFY (k+1 tokens) stays on the NV_SMALL_T-token decode kernels -- at k=4 the
  width-5 VERIFY fell to the generic dequant+matmul path (~1.4 s per iteration on the card). Warns once per k."""
  if k > NV_SMALL_T - 1: print(f"warning: speculative k={k} clamped to {NV_SMALL_T - 1} (VERIFY = k+1 tokens; decode kernels cover {NV_SMALL_T})")
  return min(k, NV_SMALL_T - 1)

def nv_fused_norm(norm:nn.RMSNorm, weight:Tensor|None, x:Tensor, *consumers:Linear) -> Tensor:
  """norm(x) -- T4.125b: on the NV decode path (concrete T <= NV_SMALL_T, `weight` = norm's fp32 copy prepared at load, every consumer
  an NV-quantized Linear) the RMSNorm and the consumers' shared q8 quantize run as ONE add_rmsnorm_q8 kernel (no residual add):
  register_fused_q8 hands that quantize the kernel's (q, scale), so the RMSNorm graph returned here is never read -- _ffn_output's
  trick at the attention blocks' attn_norm (q/k/v) and at output_norm (the head). Anything else: plain norm(x), byte-identical."""
  T, dim = x.shape[1], x.shape[-1]
  if not (isinstance(T, int) and T <= NV_SMALL_T and isinstance(dim, int) and dim % 256 == 0 and weight is not None
          and nv_quant_supported(weight.device) and all(c.ggml_type in NV_QUANT_TYPES for c in consumers)): return norm(x)
  tokens = T * cast(int, x.shape[0])
  _, q, scale, _ = add_rmsnorm_q8(x.reshape(tokens, dim), weight, norm.eps)
  register_fused_q8(normed := norm(x), tokens, dim, q, scale)
  return normed

class FFNBlock:
  freqs_cis: Tensor  # set by the attention/MLA subclasses' _init_state (GDN blocks never touch it)
  nv_norm_weights: tuple[Tensor, Tensor]|None = None  # T4.125b: (attn_norm, ffn_norm) fp32 copies, set by prepare_nv_norms
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.router_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.moe_bias)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.moe_bias)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim, bias=config.moe_bias)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    elif config.tp:
      # T4.70b: Megatron-style FFN tensor-parallel, dense blocks only (the MoE branch above is out of
      # scope -- Transformer.__init__ asserts num_experts==0 whenever tp is set, before any block is
      # built). ffn_gate/ffn_up are COLUMN-parallel: shard d owns OUTPUT rows [lo:hi) of the logical
      # (hidden_dim, dim) weight, so each device computes a full h_d = silu(gate_d(x)) * up_d(x) from its
      # own copy of x with zero cross-device traffic in between. ffn_down is ROW-parallel: shard d owns
      # the matching INPUT columns [lo:hi) of its (dim, hidden_dim) weight (same split points, so its
      # input width matches h_d's), producing a (dim,)-wide PARTIAL sum that _feed_forward adds across
      # shards after one hop back to the block's own device -- see _tp_split_sizes for the 32-alignment
      # rounding and tinygrad/llm/gguf.py's ggml_data_to_tensor for why it's needed (GGUF quant blocks are
      # 32-wide along a tensor's last/input axis). No full-size ffn_gate/ffn_up/ffn_down is ever built here
      # (unlike experts_dev, which only overrides PLACEMENT after ExpertWeights is built at full size --
      # tp changes STRUCTURE, so it has to happen here, at construction, not as a later override).
      sizes = _tp_split_sizes(config.hidden_dim, config.tp)
      self.ffn_gate_tp = [Linear(config.dim, sz, bias=False) for sz in sizes]
      self.ffn_up_tp   = [Linear(config.dim, sz, bias=False) for sz in sizes]
      self.ffn_down_tp = [Linear(sz, config.dim, bias=False) for sz in sizes]
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      # routed experts may live on a different device than attention/router (device_map "experts:<dev>"); the
      # weights never move (placed once at load), only these activations hop, around the three ExpertWeights
      # calls. sel must travel with h: self.weight[sel] indexes the (now-remote) weight buffer with it, so both
      # operands of that gather need to be on the same device. .to() is a no-op when already co-located.
      expert_dev = self.ffn_gate_exps.weight.device
      h, sel = h.to(expert_dev), sel.to(expert_dev)
      gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      if self.config.clamp_swiglu:
        # gpt-oss clamped swiglu: gate*sigmoid(alpha*gate) * (up+1), both branches clamped
        gate, up = gate.clamp(max_=self.config.swiglu_limit), up.clamp(-self.config.swiglu_limit, self.config.swiglu_limit)
        act = gate * (self.config.swiglu_alpha * gate).sigmoid() * (up + 1)
      else:
        act = gate.silu() * up
      x_down = self.ffn_down_exps(sel, act.contiguous()).to(x.device)  # (B, T, k, D), hop back to the block device
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    if hasattr(self, 'ffn_gate_tp'):
      # T4.70b Megatron FFN-TP forward: one broadcast of x per shard device (column-parallel gate/up
      # consume it directly, no further cross-device traffic to compute h_d), then one partial-sum hop
      # back per shard (row-parallel down already produced the full (B,T,dim) shape, just a PARTIAL sum)
      # -- the hidden state h_d itself (the whole point of splitting hidden_dim) never leaves its device.
      primary = self.device
      partials = []
      # named g_lin/u_lin/d_lin, not gate/up/down: those names are already bound (to Tensors) in the MoE
      # branch above -- mypy infers one type per function-scoped variable name across ALL branches, so
      # reusing them here (now Linear-typed) would read as a type conflict even though the branches are
      # mutually exclusive at runtime.
      for g_lin, u_lin, d_lin in zip(self.ffn_gate_tp, self.ffn_up_tp, self.ffn_down_tp):
        xd = x.to(g_lin.weight.device)  # .to() is a no-op when x is already on this shard's device
        h_d = g_lin(xd).silu().contiguous() * u_lin(xd)
        partials.append(d_lin(h_d).to(primary))
      return functools.reduce(lambda a, b: a + b, partials)
    # T4.123c: on the NV custom-quant decode path (T==1, concrete) the gate/up outputs already come out of
    # nv_q8_linear's own partial-sum kernel (kernels/nv_quant.py), so the .contiguous() below only forces a
    # SECOND launch (SiLU*up) instead of letting the scheduler fuse it into the one that follows -- 64 extra
    # launches/step on the 27B (T4.123 profile, P3). Every other path (generic dequant+matmul, AMD, prefill,
    # symbolic T) keeps the contiguous: this task must not change their kernels or numerics.
    T = x.shape[1]
    if (isinstance(T, int) and T <= NV_SMALL_T and nv_quant_supported(self.ffn_gate.weight.device)
        and self.ffn_gate.ggml_type in NV_QUANT_TYPES and self.ffn_up.ggml_type in NV_QUANT_TYPES):
      gate, up = self.ffn_gate(x), self.ffn_up(x)
      # T4.127a: when ffn_down will actually read it through nv_q8_linear (same build-only-if-consumed guard
      # nv_fused_norm/_ffn_output use) and both gemvs' real split-K partials are there (T4.125b -- always true
      # here, ffn_gate/up are bias=False, checked defensively like _ffn_output's own split_k_partials lookup),
      # fuse SiLU(gate)*up with ffn_down's q8-quantize into one kernel instead of a generic combine + q8_quantize_nv.
      gate_partials, up_partials = split_k_partials(gate), split_k_partials(up)
      if self.ffn_down.ggml_type in NV_QUANT_TYPES and gate_partials is not None and up_partials is not None:
        tokens, hidden = T * cast(int, x.shape[0]), self.ffn_gate.out_features
        q, scale = ffn_act_q8(gate_partials, up_partials, tokens, hidden)
        register_fused_q8(act := gate.silu() * up, tokens, hidden, q, scale)
        return self.ffn_down(act)
      return self.ffn_down(gate.silu() * up)
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  def _ffn_output(self, x:Tensor, attn_out:Tensor) -> Tensor:
    """x + attn_out -> ffn_norm -> _feed_forward -- the shared extraction point every __call__/_run uses (T4.123e,
    design doc section 3, P4): on the NV decode path (concrete T==1, dense FFN, both ffn_gate/ffn_up already
    NV-quantized -- the same gate test_llm_nv_quant.py's T4.123c fusion uses below) the residual-add (x+attn_out)
    and ffn_norm's RMSNorm+quantize collapse into ONE add_rmsnorm_q8 kernel instead of ~4 separate ones (a
    generic residual-sum, RMSNorm's sum-of-squares + apply, q8_quantize_nv). The kernel sums however many partials it's
    given: T4.125b passes the producing gemv's real split-K partials (nv_quant.py's split_k_partials, so attn_out's own
    summing reduce is never scheduled), else attn_out.reshape(...,1) as a one-wide "split" (T4.123e's form).
    `self.ffn_norm(h)` below is called completely normally afterwards
    (building nn.RMSNorm's ordinary lazy graph, the exact Tensor _feed_forward keeps using) but it is NEVER
    actually scheduled: register_fused_q8 (kernels/nv_quant.py) makes the q8_quantize call inside
    ffn_gate/ffn_up's nv_q8_linear (shared between the two since T4.123a) return the kernel's own (q, scale)
    instead of re-deriving them, so the RMSNorm graph's only readers never read its VALUE -- dead code, never
    lowered. Every other path (MoE, tensor-parallel FFN, symbolic T, non-NV, ggml_type not yet probed or not
    NV-quantized, norm weights not prepared by prepare_nv_norms) keeps the original two-step byte-identical form."""
    T, dim = x.shape[1], self.config.dim
    if (isinstance(T, int) and T <= NV_SMALL_T and dim % 256 == 0 and not hasattr(self, 'ffn_gate_exps') and not hasattr(self, 'ffn_gate_tp')
        and nv_quant_supported(self.ffn_gate.weight.device) and self.ffn_gate.ggml_type in NV_QUANT_TYPES
        and self.ffn_up.ggml_type in NV_QUANT_TYPES and self.nv_norm_weights is not None):
      tokens = T * cast(int, x.shape[0])
      # T4.125b: the producing gemv's own split-K partials (ssm_out/attn_output via nv_q8_linear), summed in-kernel -- attn_out's
      # summing reduce is then never read; anything else (a generic Linear) still folds in as one C=1 "split"
      partials = split_k_partials(attn_out)
      new_residual, q, scale, _ = add_rmsnorm_q8(x.reshape(tokens, dim), self.nv_norm_weights[1], self.ffn_norm.eps,
                                                   partials=attn_out.reshape(tokens, dim, 1) if partials is None else partials)
      h = new_residual.reshape(x.shape)
      normed = self.ffn_norm(h)
      register_fused_q8(normed, tokens, dim, q, scale)
      return h + self._feed_forward(normed)
    h = x + attn_out
    return h + self._feed_forward(self.ffn_norm(h))

  def prepare_nv_norms(self) -> None:
    """T4.125b: realized fp32 copies of the RMSNorm weights add_rmsnorm_q8 reads on the NV decode path -- made once at load
    (from_gguf), never inside a JIT capture (prepare_nv_decode's idiom). The loaded weights are lazy GGUF views (F32 bytes, then
    from_gguf's blanket .cast(float16)) the fused kernel re-converted on EVERY decode step (one E_40_32_4 per site, 10-28 profile).
    .float() folds that cast chain back to the file's fp32 values -- the numbers nn.RMSNorm's generic path multiplies by. Leaves
    nv_norm_weights None (both sites keep the generic graph) off the NV gate."""
    if self.attn_norm.weight is None or self.ffn_norm.weight is None or not nv_quant_supported(self.attn_norm.weight.device): return
    self.nv_norm_weights = (self.attn_norm.weight.float().contiguous().realize(), self.ffn_norm.weight.float().contiguous().realize())

  @property
  def device(self) -> str|tuple[str, ...]|None:
    assert self.attn_norm.weight is not None
    return self.attn_norm.weight.device

  def _positions(self, x:Tensor) -> Tensor:
    # max_context + this call's chunk max (x.max_shape[1]: the chunk variable's declared max, stable across JIT replays) makes
    # every symbolic slice in causal_mask provably in-bounds; cached per size + device, realized by _init_state (see there)
    return positions(self.config.max_context + x.max_shape[1], x.device)

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor|None=None) -> Tensor: raise NotImplementedError
  def _normed_for_attention(self, x:Tensor) -> Tensor: return self.attn_norm(x)  # T4.125b: TransformerBlock/GatedDeltaNetBlock fuse it

  def _rope_freqs(self, rope:int|UOp|Tensor, T:int|UOp) -> Tensor:
    # T5.3: `rope` is either this chunk's first ROPE position (an int/Variable: a slice of the table -- every text request, and decode
    # steps after images via generate()'s rope_start Variable) or a (1, T, 3) int32 tensor of per-token (t, h, w) positions (the
    # image-prompt chunks). start_pos keeps indexing the KV cache; only the rotation angle moves.
    if isinstance(rope, Tensor): return mrope_freqs_cis(rope, self.config)
    return self.freqs_cis[rope:rope+T]

  def __call__(self, x: Tensor, start_pos: int|UOp, rope:int|UOp|Tensor|None=None):
    self._init_state(x)
    if rope is None: rope = start_pos  # text: rope position == cache position (the pre-T5.3 graph, same UOp in both slots)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor):
      return self._ffn_output(x, self._attention(self._normed_for_attention(x), start_pos, rope)).contiguous()
    return _run(x, start_pos, rope)

# --- attention-override hook (T1.8) -----------------------------------------------------------
# Pluggable seam for a fused/tuned attention kernel to replace the standard (multi-kernel) SDPA
# expression below. Swap this module attribute (e.g. `tinygrad.llm.model.attention_impl = my_fn`)
# to route TransformerBlock's *standard* attention path through a custom implementation. Two
# paths intentionally do NOT go through this hook and are unaffected by swapping it:
#   - the manual attention-sinks softmax (gpt-oss, config.attn_sinks) in TransformerBlock itself
#   - MLATransformerBlock / GatedDeltaNetBlock, which compute attention inline, not via SDPA
# Signature matches Tensor.scaled_dot_product_attention's shape contract:
#   q:(B,H,T,Hd)  k,v:(B,KvH,Tk,Hd)  mask:(1,1,T,Tk)|None  ->  (B,H,T,Hd)
# A custom impl that only handles some shapes (e.g. decode-only, T==1, unmasked) must fall back
# to `_sdpa_default` itself for everything else (prefill, sliding-window masks, etc).
# T4.103a: the JIT memory planner's scratch arena for this call (T5.6: ~2 x (B,H,T,Tk_max) fp32, the whole
# planned scratch of a prefill jit family) scales with H = n_heads, not the per-head work -- on a wide-head,
# few-kv-head geometry (qwen3.8-27B: H=24, n_kv_heads=4) that arena dwarfs the actual attention math. Splitting
# the score pipeline into G groups of q heads (each group calling the same SDPA math on its own q/k/v slice,
# same shared mask) and concatenating the outputs doesn't change any number computed -- softmax and the two
# matmuls are per-(batch,head) independent, no cross-head reduction. SDPA_HEAD_GROUPS=0 = n_kv_heads (inferred from k's
# own head-count axis -- one KV head per group, no repeat_interleave copy needed since a size-1 batch dim
# already broadcasts in matmul); SDPA_HEAD_GROUPS=1 takes the identical, untouched single-call path (today's
# behavior, byte-identical). The kernels still loop over start_pos+T (mask/k/v slicing is unchanged) --
# grouping only slices the *head* axis, never introduces a static KV tile.
# Plain independent groups aren't enough: they have no data dependency on each other, and memory_plan_rewrite
# (schedule/memory.py) assigns buffer lifetimes from POSITION IN THE LINEARIZED KERNEL LIST -- the linearizer
# is free to interleave independent groups' kernels (e.g. every group's QK^T before any group's softmax),
# which keeps several groups' score buffers concurrently live (measured: only ~1.33-1.59x smaller at G=2..4,
# not ~1/G). A per-group `.realize()` fixes that (measured ~3.98x at G=4) but is not viable here: `_attention`
# always runs inside TransformerBlock.__call__'s `@function(precompile=True)` trace (`_run`, above), which sets
# ALLOW_DEVICE_USAGE=0 (tinygrad/function.py) specifically to forbid mid-trace device compilation -- a
# `.realize()` there raises `AssertionError: usage of device CPU disallowed` against real Transformer.generate().
# The fix that IS legal in that trace: an explicit ORDERING EDGE via UOp.after (no device touched -- pure graph
# construction, same idiom `_attention` already uses for the KV-cache write above, and the hand-written kernels
# in tinygrad/llm/kernels/*.py use for their `Tensor(out.after(kernel))` results). For group i>0, round-trip
# group i-1's OUTPUT (small, (B,H/G,T,Hd) -- NOT the score buffer, so this scratch doesn't scale with context)
# through a fresh buffer via `.assign()` (itself a lazy STORE, no realize), then gate group i's q slice on that
# buffer's AFTER: `.after()` doesn't change the q slice's VALUE, only its position in the linearized kernel
# list, so group i's first kernel (QK^T, which reads this q) cannot be scheduled before group i-1's last kernel
# has run -- collapsing the two groups' liveness windows to non-overlapping. Verified: bit-exact vs the
# unchained version, and ~3.93-3.98x smaller at G=4 through both a raw JIT harness and the real
# `Transformer.generate()` path (see test_llm_prefill_arena.py's test_group_count_scales_arena_down).
# Measured on the 3090 (09-11, chain 19, Q4_0 card map): 4 groups = flips unchanged, int8 context ceilings +16k tokens each, but prefill
# 119 -> 109 tok/s and (before the T>4 gate below) decode 23.4 -> 17.6 -- the dependency chain serializes the groups and each SDPA kernel
# is 4x smaller. So: OPT-IN (default 1 = the single call), and grouping only applies to prefill chunks (T > 4) where the arena is
# large; a decode step's score row (H x 1 x Tk) is ~25 MB at 262k and not worth the split.
SDPA_HEAD_GROUPS = ContextVar("SDPA_HEAD_GROUPS", 1)
def sdpa_head_groups_for(n_kv_heads:int) -> int: return n_kv_heads if SDPA_HEAD_GROUPS.value == 0 else SDPA_HEAD_GROUPS.value  # 0 = one per KV head

def _sdpa_default(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None) -> Tensor:
  H, KvH = cast(int, q.shape[1]), cast(int, k.shape[1])  # head counts are concrete
  G = sdpa_head_groups_for(KvH)
  T = q.shape[2]  # a bound Variable for prefill chunks (group), a concrete int for decode steps (never split)
  if G <= 1 or (isinstance(T, int) and T <= 4): return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  assert H % G == 0 and KvH % G == 0, f"SDPA_HEAD_GROUPS={G} must divide both n_heads={H} and n_kv_heads={KvH}"
  hg, kg = H // G, KvH // G
  outs, prev = [], None
  for i in range(G):
    qg = q[:, i*hg:(i+1)*hg]
    if prev is not None:
      scratch = Tensor.empty(*prev.shape, dtype=prev.dtype, device=prev.device).assign(prev)
      qg = Tensor(qg.uop.after(scratch.uop))
    o = qg.scaled_dot_product_attention(k[:, i*kg:(i+1)*kg], v[:, i*kg:(i+1)*kg], attn_mask=mask, enable_gqa=True)
    outs.append(o)
    prev = o
  return outs[0].cat(*outs[1:], dim=1)

attention_impl: Callable[[Tensor, Tensor, Tensor, Tensor|None], Tensor] = _sdpa_default

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_out_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _normed_for_attention(self, x:Tensor) -> Tensor:  # T4.125b: q/k/v are q8 consumers -- see nv_fused_norm
    return nv_fused_norm(self.attn_norm, None if self.nv_norm_weights is None else self.nv_norm_weights[0], x, self.attn_q, self.attn_k, self.attn_v)

  nv_qk_norm_weights: tuple[Tensor, Tensor]|None = None  # T4.137: (attn_q_norm, attn_k_norm) fp32 copies for attn_prep, set by prepare_nv_norms
  def prepare_nv_norms(self) -> None:
    super().prepare_nv_norms()
    # T4.137: the per-head q/k RMSNorm weights' fp32 copies (the same .float() of the lazy GGUF views nn.RMSNorm multiplies by), made once at
    # load like attn_norm/ffn_norm's -- read by the decode path's attn_prep kernel (_nv_attn_prep), same NV gate as the parent's
    if self.nv_norm_weights is not None and self.config.qk_norm == self.config.head_dim and self.config.attn_output_gate:
      assert self.attn_q_norm.weight is not None and self.attn_k_norm.weight is not None  # nn.RMSNorm types the weight Optional (mypy)
      self.nv_qk_norm_weights = (self.attn_q_norm.weight.float().contiguous().realize(), self.attn_k_norm.weight.float().contiguous().realize())

  def _attention(self, x:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor|None=None) -> Tensor:
    if rope is None: rope = start_pos  # T5.3 callers pass the rope position explicitly; direct callers (tests) get the pre-T5.3 default
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    B, T, _ = x.shape
    # T4.137: on the NV decode path the q/k norms, rope, the gate split and the KV_INT4 store below are ONE kernel (attn_prep_nv)
    if (prep := self._nv_attn_prep(q, k, v, start_pos, rope)) is not None: q, gate = prep[0], prep[1]
    else:
      if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)
      if self.config.attn_output_gate:
        qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
        q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
      q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
      k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
      v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
      if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

      freqs_cis = self._rope_freqs(rope, T)
      q = apply_rope(q[..., :self.config.rope_dim], freqs_cis).cat(q[..., self.config.rope_dim:], dim=-1)
      k = apply_rope(k[..., :self.config.rope_dim], freqs_cis).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    # cast to the cache's dtype at write (a no-op when KV_F32=1); cast back up to the activation dtype at
    # read, so attention compute always runs at x's precision regardless of what the cache stores
    if self.cache_kv.dtype == dtypes.int8:
      # T6.1 KV_INT8: symmetric absmax int8 quant, `blk`-wide blocks along head_dim (see kv_int8_block).
      # Two independent buffers (cache_kv, cache_kv_scale), each with its own .uop.after(store) so the JIT
      # sees both writes ordered before their own reads -- same dependency idiom as the plain-cast path
      # below, just doubled since there are now two buffers instead of one. AMD's fused flash-attention
      # kernel (below, in the else branch) expects a real-valued cache and is never reached here -- KV_INT8
      # always takes the dequant+attention_impl path, a correct (if not fused-kernel-accelerated) fallback
      # on the already-descoped AMD/hybrid combo (CLAUDE.md: AMD is not a target).
      blk = kv_int8_block(self.config.head_dim)
      blocks = Tensor.stack(k, v).reshape(2, B, self.config.n_kv_heads, T, self.config.head_dim // blk, blk)
      scale = (blocks.abs().max(axis=-1, keepdim=True) / 127.0).maximum(1e-8)           # (2,B,KvH,T,Hd//blk,1)
      quant = (blocks / scale).round().clip(-127, 127).cast(dtypes.int8).reshape(2, B, self.config.n_kv_heads, T, self.config.head_dim)
      assigned_kv = Tensor(self.cache_kv.uop.after(
        self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(quant.uop)))
      assigned_scale = Tensor(self.cache_kv_scale.uop.after(
        self.cache_kv_scale[:, :, :, start_pos:start_pos+T, :].uop.store(scale.squeeze(-1).cast(self.cache_kv_scale.dtype).uop)))
      def _dequant(idx:int) -> Tensor:
        qv, sv = assigned_kv[idx, :, :, 0:start_pos+T, :], assigned_scale[idx, :, :, 0:start_pos+T, :]
        return (qv.cast(x.dtype).reshape(*qv.shape[:-1], qv.shape[-1] // blk, blk) * sv.cast(x.dtype).unsqueeze(-1)).reshape(qv.shape)
      nv_fast = self._nv_attn(q, assigned_kv, assigned_scale, start_pos, T)
      if nv_fast is None: k, v = _dequant(0), _dequant(1)
    elif self.cache_kv.dtype == dtypes.uint8:
      # T4.100 KV_INT4: same kv_int8_block(head_dim)-wide absmax-scale layout as KV_INT8 above (cache_kv_scale
      # is byte-identical between the two flags -- shape, dtype, allocation, all shared, see _init_state).
      # Only cache_kv's own encoding differs: two adjacent head_dim elements pack into one stored byte, so
      # cache_kv's last dim is head_dim//2 (_init_state again). Quant: q = round(x/scale) clipped to the full
      # 4-bit two's-complement range [-8,7] (scale is picked the same way as KV_INT8's /127.0, just /7.0 for
      # the smaller range); nibble = q+8 in [0,15] -- an unsigned OFFSET/biased encoding (ggml's own Q4_0
      # convention) rather than raw two's-complement-in-a-nibble, so packing only ever needs
      # bitwise_or/lshift/rshift/bitwise_and on non-negative small ints, never shift-of-a-negative-value
      # (backend-dependent for signed types). Dequant undoes this exactly: unpack the two nibbles, subtract
      # the 8 bias, multiply by scale -- algebraically round(x/scale)*scale, same as KV_INT8 (per-element
      # error <= scale/2, the same round-to-nearest guarantee any such quantizer gives -- see
      # test_llm_kv_dtype.py's direct pack/unpack round-trip test).
      blk, hd = kv_int8_block(self.config.head_dim), self.config.head_dim
      if prep is not None: assigned_kv, assigned_scale = prep[2], prep[3]  # T4.137: attn_prep_nv already stored this run's rows
      else:
        blocks = Tensor.stack(k, v).reshape(2, B, self.config.n_kv_heads, T, hd // blk, blk)
        scale = (blocks.abs().max(axis=-1, keepdim=True) / 7.0).maximum(1e-8)               # (2,B,KvH,T,Hd//blk,1)
        nibble = ((blocks / scale).round().clip(-8, 7) + 8).cast(dtypes.uint8).reshape(2, B, self.config.n_kv_heads, T, hd // 2, 2)
        packed = nibble[..., 0].bitwise_or(nibble[..., 1].lshift(4))                        # (2,B,KvH,T,Hd//2) uint8, [0,15]|[0,15]<<4
        assigned_kv = Tensor(self.cache_kv.uop.after(
          self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(packed.uop)))
        assigned_scale = Tensor(self.cache_kv_scale.uop.after(
          self.cache_kv_scale[:, :, :, start_pos:start_pos+T, :].uop.store(scale.squeeze(-1).cast(self.cache_kv_scale.dtype).uop)))
      def _dequant(idx:int) -> Tensor:
        pv, sv = assigned_kv[idx, :, :, 0:start_pos+T, :], assigned_scale[idx, :, :, 0:start_pos+T, :]
        lo, hi = pv.bitwise_and(0xF), pv.rshift(4)                                        # unpacked nibbles, both [0,15]
        qv = Tensor.stack(lo, hi, dim=-1).reshape(*pv.shape[:-1], hd).cast(x.dtype) - 8    # interleaved back to (...,Hd), debiased
        return (qv.reshape(*qv.shape[:-1], qv.shape[-1] // blk, blk) * sv.cast(x.dtype).unsqueeze(-1)).reshape(qv.shape)
      nv_fast = self._nv_attn(q, assigned_kv, assigned_scale, start_pos, T)
      if nv_fast is None: k, v = _dequant(0), _dequant(1)
    else:
      assigned_kv = Tensor(self.cache_kv.uop.after(
        self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)))
      # on RDNA3, hybrid models use custom flash attention kernels directly on the (compressed) KV cache
      if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
        attn = flash_attention(q, assigned_kv, start_pos+T)
        attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
        return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
      nv_fast = self._nv_attn(q, assigned_kv, None, start_pos, T)
      if nv_fast is None:
        k = assigned_kv[0, :, :, 0:start_pos+T, :].cast(x.dtype)
        v = assigned_kv[1, :, :, 0:start_pos+T, :].cast(x.dtype)

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    # sliding-window layers also need a mask at T==1 (decode), to drop cache entries older than the window
    mask = causal_mask(self._positions(x), T, start_pos, x.dtype, self.config.sliding_window) \
      if resolve(T != 1) or self.config.sliding_window else None

    if hasattr(self, "attn_sinks"):
      # attention sinks: a learned per-head logit that only contributes to the softmax denominator
      # (no value contribution), competing with real keys for probability mass. Can't express this via
      # scaled_dot_product_attention's attn_mask (that only biases real score entries), so do it manually.
      KV, R = self.config.n_kv_heads, self.config.n_heads // self.config.n_kv_heads
      qg, kg, vg = q.reshape(B, KV, R, T, self.config.head_dim), k.unsqueeze(2), v.unsqueeze(2)
      scores = (qg.cast(dtypes.float32) @ kg.cast(dtypes.float32).transpose(-1, -2)) * (1.0 / self.config.head_dim ** 0.5)
      if mask is not None: scores = scores + mask.unsqueeze(1)
      sink = self.attn_sinks["weight"].reshape(1, KV, R, 1, 1).cast(dtypes.float32)
      m = scores.max(-1, keepdim=True).maximum(sink)
      e = (scores - m).exp()
      w = (e / (e.sum(-1, keepdim=True) + (sink - m).exp())).cast(x.dtype)
      attn = (w @ vg.cast(x.dtype)).reshape(B, self.config.n_heads, T, self.config.head_dim)
    else:
      attn = nv_fast if nv_fast is not None else attention_impl(q, k, v, mask)      # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _nv_attn_prep(self, q:Tensor, k:Tensor, v:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]|None:
    """T4.137: (q (B,H,T,Hd) fp32, gate (B,T,H*Hd), cache_kv, cache_kv_scale after this run's store) from the projections by kernels/
    nv_attn_prep.py's attn_prep -- a concrete run of T <= NV_SMALL_T tokens (decode, VERIFY) on a KV_INT4 cache where the NV attention
    kernels run (nv_decode_attention_ok), per-head q/k norms with load-time weights (prepare_nv_norms), the output gate, a text rope
    position. None = the generic graph (prefill chunks, image rope rows, NV_ATTN_PREP=0, every other config)."""
    cfg, T = self.config, q.shape[1]
    if not (NV_ATTN_PREP.value and isinstance(T, int) and T <= NV_SMALL_T and self.nv_qk_norm_weights is not None and not isinstance(rope, Tensor)
            and self.cache_kv.dtype == dtypes.uint8 and attn_prep_geometry_ok(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim)
            and nv_decode_attention_ok(q.device, cfg.head_dim, self.cache_kv)): return None
    return attn_prep(q, k, v, *self.nv_qk_norm_weights, self.freqs_cis, self.cache_kv, self.cache_kv_scale, start_pos, rope, cfg.n_heads,
                     (self.attn_q_norm.eps, self.attn_k_norm.eps))

  def _nv_attn(self, q:Tensor, assigned_kv:Tensor, assigned_scale:Tensor|None, start_pos:int|UOp, T:int|UOp) -> Tensor|None:
    """T4.107a/b: the NV split-KV kernels (kernels/nv_attn.py, NV_CUSTOM_ATTN=1) on the cache as stored -- the decode kernel for a
    plain T==1 step, the prefill kernel for a chunk of up to MAX_PREFILL_ROWS tokens (T symbolic or int); None = take the generic
    dequant + attention_impl path (sliding windows, attention sinks, other devices, wider chunks)."""
    if self.config.sliding_window or hasattr(self, "attn_sinks"): return None
    if not nv_decode_attention_ok(q.device, self.config.head_dim, self.cache_kv): return None
    if isinstance(T, int) and T == 1: return nv_decode_attention(q, assigned_kv, assigned_scale, start_pos+T).cast(q.dtype)
    # T4.134: a speculative VERIFY run (2..4 concrete tokens) on the decode kernel, whose cost barely moves with the fill
    if (verify := nv_verify_attention(q, assigned_kv, assigned_scale, start_pos+T)) is not None: return verify.cast(q.dtype)
    if q.max_shape[2] > MAX_PREFILL_ROWS: return None
    return nv_prefill_attention(q, assigned_kv, assigned_scale, start_pos+T, start_pos).cast(q.dtype)

  def _init_state(self, x:Tensor):
    # every call, not just the first: a new chunk size needs its own (larger) position buffer, and it must be realized HERE --
    # __call__ runs _init_state before the @function(precompile=True) trace, inside which nothing may touch a device
    self._positions(x)
    if not hasattr(self, "cache_kv"):
      int8 = bool(getenv("KV_INT8", 0))
      int4 = bool(getenv("KV_INT4", 0))  # T4.100: wins if both are set somehow (arbitrary but deterministic tie-break)
      if int4: assert self.config.head_dim % 2 == 0, f"KV_INT4 packs pairs of head_dim elements per byte, {self.config.head_dim=} is odd"
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      # T4.100 KV_INT4: cache_kv's last dim is head_dim//2 -- two 4-bit values packed per stored byte, see _attention
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context,
                                   self.config.head_dim // 2 if int4 else self.config.head_dim,
                                   dtype=dtypes.uint8 if int4 else dtypes.int8 if int8 else kv_cache_dtype(), device=x.device)
      if int8 or int4:
        # T6.1 KV_INT8 / T4.100 KV_INT4: one absmax scale per kv_int8_block(head_dim)-wide chunk of the
        # UNPACKED head_dim, same leading shape as cache_kv, same fp16 dtype for both flags (KV_INT4 "implies
        # the int8-style scale layout" -- see kv_int8_block's own comment). Zeroed like cache_kv itself: a
        # position never written is never read either (see _attention's 0:start_pos+T reads), so leftover-
        # zero scale there is inert (0 dequants to 0 regardless of scale) -- same reasoning as cache_kv's own
        # zero-fill above.
        self.cache_kv_scale = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context,
                                           self.config.head_dim // kv_int8_block(self.config.head_dim),
                                           dtype=dtypes.float16, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device,
        yarn_factor=self.config.yarn_factor, yarn_orig_ctx=self.config.yarn_orig_ctx, yarn_beta_fast=self.config.yarn_beta_fast,
        yarn_beta_slow=self.config.yarn_beta_slow, yarn_attn_factor=self.config.yarn_attn_factor)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor|None=None) -> Tensor:
    if rope is None: rope = start_pos  # T5.3 callers pass the rope position explicitly; direct callers (tests) get the pre-T5.3 default
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self._rope_freqs(rope, T))
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self._rope_freqs(rope, T))

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    # cast to the cache's dtype at write, back up to x's dtype at read -- see TransformerBlock._attention
    k = Tensor(self.cache_k.uop.after(
      self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.cast(self.cache_k.dtype).uop)))[:, :, 0:start_pos+T, :].cast(x.dtype)
    v = k[..., :self.config.kv_lora_rank]

    mask = causal_mask(self._positions(x), T, start_pos, x.dtype) if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    self._positions(x)  # see TransformerBlock._init_state
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim,
                                  dtype=kv_cache_dtype(), device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device,
        yarn_factor=self.config.yarn_factor, yarn_orig_ctx=self.config.yarn_orig_ctx, yarn_beta_fast=self.config.yarn_beta_fast,
        yarn_beta_slow=self.config.yarn_beta_slow, yarn_attn_factor=self.config.yarn_attn_factor)

class GatedDeltaNetBlock(FFNBlock):
  nv_decode_weights: tuple[Tensor, Tensor, Tensor, Tensor]|None = None  # T4.123f+g, set by prepare_nv_decode
  nv_beta_alpha_weight: Tensor|None = None  # T4.127b: ssm_beta+ssm_alpha concatenated fp16 weight, set by prepare_nv_decode
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _normed_for_attention(self, x:Tensor) -> Tensor:
    """attn_norm(x), fused with its own quantize-equivalent at the NV decode path (T4.123e) -- see FFNBlock's
    _ffn_output docstring for the general idea. GDN's own _attention calls `x = x.half()` as its very first
    operation UNCONDITIONALLY, in every branch (the fast nv_gdn_decode path and the generic per-token-loop
    path alike) -- every attn_norm consumer here (attn_qkv/attn_gate/ssm_beta/ssm_alpha, or ssm_g_a/ssm_f_a for
    kda) reads that fp16 copy, not an int8-quantized one (T4.98h's nv_dense f16 gemv, not nv_q8_linear). So
    add_rmsnorm_q8's want_fp16 output IS exactly the value _attention's `x.half()` would have produced, and
    Tensor.cast is a true no-op when the dtype already matches (tinygrad/mixin/dtype.py: `return self if
    self.dtype == dt else ...`) -- feeding _attention this already-fp16 Tensor in place of attn_norm(x) makes
    its internal `.half()` a no-op, so _attention needs no change at all (this task's GDN restriction).
    T4.125b: attn_qkv/attn_gate ARE q8 consumers (Q4_0 on the 27B: nv_q8_linear quantized this fp16 copy again, one
    q8_quantize_nv per GDN block per step) -- register_fused_q8 hands them the kernel's own (q, scale), quantized from
    the fp32 normed value instead of its fp16 rounding (a small numerics change); ssm_beta/ssm_alpha keep the fp16 copy."""
    T, dim = x.shape[1], self.config.dim
    if (isinstance(T, int) and T <= NV_SMALL_T and dim % 256 == 0 and self.nv_norm_weights is not None
        and nv_quant_supported(self.nv_norm_weights[0].device)):
      tokens = T * cast(int, x.shape[0])
      _, q, scale, fp16_out = add_rmsnorm_q8(x.reshape(tokens, dim), self.nv_norm_weights[0], self.attn_norm.eps, want_fp16=True)
      assert fp16_out is not None
      register_fused_q8(normed := fp16_out.reshape(x.shape), tokens, dim, q, scale)
      return normed
    return self.attn_norm(x)

  def prepare_nv_decode(self) -> None:
    """T4.123f+g: realized fp32 copies of the four small weights kernels/nv.py's gdn_decode reads -- made once at load (from_gguf), never
    inside a JIT capture: the loaded tensors are lazy GGUF byte views a custom kernel would re-copy on every step. Leaves
    nv_decode_weights None (decode keeps the generic graph + fused scan) off the gate or outside the kernels' geometry.
    T4.127b: ssm_beta/ssm_alpha's weights concatenated into one (2*num_v_heads, dim) fp16 buffer -- same dtype convention as
    nv_dense.py's _fp16_copy (every nv_f16_gemv input is fp16 regardless of the GGUF dtype) -- so _attention's decode branch
    runs ONE gemv (nv_f16_gemv_raw) instead of two. Left None (falls back to the two separate calls, _attention's own
    dispatch, below) when config.dim isn't a multiple of 128: nv_f16_gemv_raw's kernel needs that (nv_dense.py's
    nv_dense_eligible check, which the ORIGINAL per-Linear self.ssm_beta(x)/self.ssm_alpha(x) calls apply themselves --
    this gemv doesn't go through Linear.__call__, so the same check has to happen here).
    T4.138: also under the prefill gate (nv_custom_kernels_supported): _nv_prefill reads the same weights. Each path still checks its
    own gate at call time."""
    dev = self.ssm_a.device
    if hasattr(self, "ssm_g_a") or self.ssm_norm.weight is None or not (nv_decode_kernel_supported(dev) or nv_custom_kernels_supported(dev)): return
    if self.head_k_dim % 32 or self.head_v_dim % 8 or self.head_v_dim > 256 or self.conv_channels % 32: return  # 32-lane rows, <= 1024 threads
    self.nv_decode_weights = cast(tuple[Tensor, Tensor, Tensor, Tensor], tuple(t.float().contiguous().realize() for t in
      (self.ssm_conv1d["weight"], self.ssm_norm.weight, self.ssm_dt["bias"], self.ssm_a)))
    if self.config.dim % 128 == 0:
      self.nv_beta_alpha_weight = self.ssm_beta.weight.cat(self.ssm_alpha.weight, dim=0).cast(dtypes.float16).contiguous().realize()

  def _attention(self, x:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor|None=None, capture:bool=False) -> Tensor:
    # rope: unused, GDN has no positions (T5.3); kept so FFNBlock.__call__ can pass it positionally to every block type.
    """capture=True (speculative_generate's VERIFY: a concrete run of 2..GDN_DECODE_TOKENS tokens, see __call__) also writes the
    conv window and the recurrent state after each of the call's first T-1 tokens into self.spec_track[0][t] / [1][t] -- a partial
    accept of m < T-1 drafts makes slot m the live state (Transformer._adopt_step) instead of re-running the accepted tokens (T4.132;
    replaces T4.66d's CHECKPOINT + REDO forward). The output and the final state are those of capture=False. On the card the two
    gdn_decode kernels store the slots from values they already hold; elsewhere the per-token loop retains them (capture forces the
    loop: WY and the fused prefill scan only ever form the run's last state). Slot t's conv rows are the window rows t+1..t+K-1 rounded
    to conv_state's dtype -- what conv_state_store keeps for the final state."""
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos, device=x.device).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # T4.123f+g: a decode token on the card is two kernels from the four projections (conv + conv_state shift, then norms/beta/alpha/
    # scan/ssm_norm/gate) instead of 13 small generic kernels around the fused scan -- kernels/nv.py's gdn_decode. Same gate as the
    # fused decode scan below; prefill (symbolic or T > GDN_DECODE_TOKENS) never comes here. T4.132: a concrete run of up to
    # GDN_DECODE_TOKENS tokens (speculative decoding's VERIFY) takes the same two kernels, the tokens sequential inside them.
    if not symbolic and T <= GDN_DECODE_TOKENS and self.nv_decode_weights is not None and nv_decode_kernel_supported(x.device):
      x = x.half()
      # T4.127b: ssm_beta(x)/ssm_alpha(x) -- two f16 gemvs on the same input -- fused into one over their load-time
      # concatenated weight (prepare_nv_decode) when that's ready; gdn_decode reads beta/alpha out of the one result
      # (beta at [:heads], alpha at [heads:]). Falls back to the two separate calls (dim not a multiple of 128 --
      # prepare_nv_decode's own comment) with the same (B,T,2*heads) shape via one concat.
      beta_alpha = (nv_f16_gemv_raw(self.nv_beta_alpha_weight, x.reshape(B*T, self.config.dim), 2*self.num_v_heads).reshape(B, T, -1)
                    if self.nv_beta_alpha_weight is not None else self.ssm_beta(x).cat(self.ssm_alpha(x), dim=-1))
      z = nv_gdn_decode(self.attn_qkv(x), self.attn_gate(x), beta_alpha, self.conv_state, self.recurrent_state,
                        *self.nv_decode_weights, start_pos, k_heads=self.num_k_heads, qk_eps=1e-6, norm_eps=self.ssm_norm.eps,
                        track=self.spec_track if capture else None)
      assert not capture or T - 1 <= cast(tuple[Tensor, Tensor], self.spec_track)[1].shape[0], f"{T=}: more tokens than spec_track slots"
      return self.ssm_out(z)

    # T4.138: every other call on the card's fused-scan path (a prefill chunk: symbolic T, padded to T_pad) -- see _nv_prefill.
    # GDN_NV_PREP=0: the generic glue below around the same fused scan (A/B, rollback)
    if (GDN_NV_PREP.value and not capture and not is_kda and self.nv_decode_weights is not None and self.head_v_dim % 32 == 0
        and nv_custom_kernels_supported(x.device)):
      return self._nv_prefill(x, start_pos, T, T_pad)

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp).
    # T4.123b: when T is concrete and equals T_pad (decode, or any non-symbolic call), the two stores below
    # write every row ((conv_kernel-1) + T == (conv_kernel-1) + T_pad) -- skip the zero-fill, it's dead weight.
    # Symbolic prefill chunks can have T < T_pad (the remainder chunk), so the padded rows must stay zero.
    win_fully_written = not symbolic and T == T_pad
    win_ctor = Tensor.empty if win_fully_written else Tensor.zeros
    win = win_ctor(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels, device=x.device).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    state_track: Tensor|None = None
    # T4.66b: `capture` also excludes the fused path -- it has no per-position state to expose (see the
    # docstring above), so capturing forces the plain loop below regardless of device.
    # T4.98g: on NV, the fused kernel also serves T_pad==1 decode (see nv.py's GDN_NV_FUSED_DECODE) -- gated
    # independently of GDN_NV_FUSED (prefill's own gate) so decode fusion doesn't need prefill fusion on too.
    nv_decode = T_pad == 1 and nv_decode_kernel_supported(x.device)
    fused = gated_delta_prefill if amd_custom_kernels_supported(x.device) else \
            nv_gated_delta_prefill if nv_custom_kernels_supported(x.device) or nv_decode else None
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and fused is not None and not capture:
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3 / NV sm_70+, T6.2)
      core = fused(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      state = initial.where(0, state.float())

      def run_scan(state:Tensor, q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor) -> tuple[Tensor, Tensor, Tensor|None]:
        # T4.69a: GDN_SCAN_IMPL picks the per-token loop (default/auto) or the chunkwise WY-form gdn_scan_wy
        # (device-agnostic, matmul-shaped -- see its docstring). Dispatch lives here, inside the per-group
        # call site below, so a GDN_HEAD_GROUPS split composes with it for free.
        # T4.69b: WY amortizes one (T_pad,T_pad) triangular solve plus several matmuls across T_pad steps --
        # at T_pad==1 there's nothing to amortize, just that overhead around a single-step recurrence the loop
        # already computes directly. Measured on the 2026-08-31 GPU battery (pooled qwen3.8-27B): IMPL=2
        # improved chunked prefill (22->28 tok/s @5.9k, 21->27 @16.2k) but made token-by-token decode
        # pathologically slow. Gate on T_pad, not T: T is a bound symbolic Variable during chunked prefill
        # (isinstance(T, UOp)) even for a single-real-token remainder chunk, but that chunk is still padded out
        # to the full declared chunk_size -- T_pad, already concrete (max_shape/to_max_shape always returns
        # plain ints) either way, is the one quantity that means "how many steps this call's scan processes"
        # for both the symbolic/padded prefill path and a plain concrete T=1 decode step.
        # T4.66b: `capture` also forces the loop (never WY) -- see _attention's own docstring for why.
        use_wy = gdn_scan_impl_for() == GDN_SCAN_WY and T_pad > 1 and not capture
        gdn_last_scan_impl.append(GDN_SCAN_WY if use_wy else GDN_SCAN_LOOP)
        if use_wy: return (*gdn_scan_wy(state, q, k, v, beta, alpha), None)
        q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
        alpha = alpha.unsqueeze(-1)
        outs: list[Tensor] = []
        states: list[Tensor]|None = [] if capture else None
        for t in range(T_pad):
          s1 = state * alpha[:, :, t]  # decay the state
          delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
          state = s1 + delta * k[:, :, t]
          outs.append((state * q[:, :, t]).sum(-1))
          if states is not None: states.append(state)  # T4.66b: free retention -- `state` is already computed above
        track = states[0].stack(*states[1:], dim=2) if states else None  # (B,H,T_pad,V,K)
        return state, outs[0].stack(*outs[1:], dim=1), track

      # T4.62: split the scan into G sequential head groups when the geometry needs it (see
      # gdn_head_groups_for above) -- each group's own .contiguous() forces a separate, narrower kernel,
      # then a single cat reassembles the full state/output. G=1 (unchanged geometries) skips all of this
      # and lowers to exactly the same graph as before.
      if (G := gdn_head_groups_for(self.num_v_heads)) <= 1:
        state, stacked, state_track = run_scan(state, q, k, v, beta, alpha)
      else:
        gsize = -(-self.num_v_heads // G)  # ceil(H/G): only the last group is smaller when H % G != 0
        group_states, group_outs, group_tracks = [], [], []
        for lo in range(0, self.num_v_heads, gsize):
          hi = min(lo + gsize, self.num_v_heads)
          g_state, g_outs, g_track = run_scan(state[:, lo:hi], q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], beta[:, lo:hi], alpha[:, lo:hi])
          group_states.append(g_state.contiguous())
          group_outs.append(g_outs.contiguous())
          if capture: group_tracks.append(cast(Tensor, g_track).contiguous())
        state = group_states[0].cat(*group_states[1:], dim=1) if len(group_states) > 1 else group_states[0]
        stacked = group_outs[0].cat(*group_outs[1:], dim=2) if len(group_outs) > 1 else group_outs[0]
        if capture: state_track = group_tracks[0].cat(*group_tracks[1:], dim=1) if len(group_tracks) > 1 else group_tracks[0]

      # store the updated recurrent state in place, then read the stacked outputs after the write
      stores = [self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)]
      if capture and T > 1:  # T4.132: the per-token states of slots 0..T-2 (see the docstring)
        conv_t, rec_t = cast(tuple[Tensor, Tensor], self.spec_track)
        assert state_track is not None and isinstance(T, int) and T - 1 <= rec_t.shape[0], f"{T=} {rec_t.shape=}"
        # one store per slot: a single permuted store of all slots became a CPU kernel whose buffer slots had a gap (the T-th state
        # unread) plus a THREAD core-id var, declared out of order with the runtime's argument order -- a segfault at the 35B's head
        # geometry (test_gdn_scan_parity.py)
        stores += [rec_t[t].uop.store(state_track[:, :, t].cast(rec_t.dtype).uop) for t in range(T-1)]
        stores += [conv_t[t].uop.store(conv_window[:, t+1:t+self.ssm_conv_kernel].cast(conv_t.dtype).uop) for t in range(T-1)]
      core = Tensor(stacked.contiguous().uop.after(*stores))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _nv_prefill(self, x:Tensor, start_pos:UOp, T:int|UOp, T_pad:int) -> Tensor:
    """T4.138: a prefill chunk with the glue around the fused scan as nv_gdn_prep's kernels (the generic graph above: ~15 kernels
    before the scan, ~287 us per block at 128 tokens on the card, and 4 after it, ~43 us). The projections run at the padded width T_pad
    on one contiguous fp16 x (nv_wmma_linear then pads nothing and the f16 gemv copies nothing; zero rows past T); the toks Variable
    masks the rest. The post-scan kernel also quantizes z for ssm_out (register_fused_q8: its gemm reads those, not z)."""
    B, dim = cast(int, x.shape[0]), self.config.dim
    x = x.half()
    # contiguous in its 2-D form: the f16 gemv's own .contiguous() then folds (a reshaped CONTIGUOUS would be copied again, E_5120_32_4)
    x_flat = (x.pad_to((B, T_pad, dim)) if isinstance(T, UOp) else x).reshape(B*T_pad, dim).contiguous()
    x_pad = x_flat.reshape(B, T_pad, dim)
    beta_alpha = (nv_f16_gemv_raw(self.nv_beta_alpha_weight, x_flat, 2*self.num_v_heads)
                  if self.nv_beta_alpha_weight is not None else self.ssm_beta(x_pad).cat(self.ssm_alpha(x_pad), dim=-1))
    conv_w, norm_w, dt_bias, ssm_a = cast(tuple[Tensor, Tensor, Tensor, Tensor], self.nv_decode_weights)
    core = nv_gdn_prefill(self.attn_qkv(x_pad), beta_alpha, self.conv_state, self.recurrent_state, conv_w, dt_bias, ssm_a, start_pos, T,
                          k_heads=self.num_k_heads, qk_eps=1e-6)
    z, q, scale, sums = nv_gdn_post(core, self.attn_gate(x_pad), norm_w, T, self.ssm_norm.eps)
    register_fused_q8(z, B*T_pad, self.num_v_heads*self.head_v_dim, q, scale, sums)
    out = self.ssm_out(z)
    return out[:, :T] if isinstance(T, UOp) else out

  spec_track: tuple[Tensor, Tensor]|None = None  # T4.132: (conv (S,B,K-1,C), recurrent (S,B,H,V,K)) -- a VERIFY run's per-token states
  def __call__(self, x:Tensor, start_pos:int|UOp, rope:int|UOp|Tensor|None=None, spec:int=0) -> Tensor:
    # rope (T5.3): passed positionally by Transformer.forward to every block; GDN has no positions, so it is ignored here -- it must
    # still be a named parameter, or forward()'s third argument would land in `spec`.
    # T4.132: spec=True (Transformer.forward's concrete multi-token spec call, i.e. VERIFY) also fills spec_track through _attention's
    # `capture` -- stores into buffers made here, before the precompiled trace (nothing in it may touch a device), on first use (a model
    # that never speculates never pays for them). spec = the slot count (Transformer.spec_slots: spec_k_clamped(k)); True = the most.
    self._init_state(x)
    if spec: self._init_spec_track(GDN_DECODE_TOKENS - 1 if spec is True else spec)
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp) -> Tensor:
      attn_out = self._attention(self._normed_for_attention(x), start_pos, capture=bool(spec))
      return self._ffn_output(x, attn_out).contiguous()
    return _run(x, start_pos)

  def _init_spec_track(self, S:int=GDN_DECODE_TOKENS - 1) -> None:
    if self.spec_track is None:
      conv, rec = self.conv_state, self.recurrent_state
      self.spec_track = (Tensor.empty(S, *conv.shape, dtype=conv.dtype, device=conv.device),
                         Tensor.empty(S, *rec.shape, dtype=rec.dtype, device=rec.device))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      # conv_state only ever holds the last (conv_kernel-1) input rows (already upcast to fp32 on read via
      # win.dtype in _attention above) and doesn't scale with max_context, but it's free to halve too: an
      # evidence run (tiny random-weight config, 5 prompts x 64 decode steps, isolated from recurrent_state
      # by keeping recurrent_state fp32) showed 0/320 greedy-token divergences, max logit delta ~0.003 --
      # flag-gate it like the KV caches.
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, dtype=kv_cache_dtype(), device=x.device).clone()
      # recurrent_state is NOT flag-gated -- always fp32, unlike the write-once/read-many KV caches above it
      # is read-modify-written every decode step (decay + delta rule), so fp16 error compounds across the
      # whole generation instead of staying local to one position. Evidence (same tiny config/harness, fp16
      # recurrent_state isolated from conv_state -- conv_state fp32): 7/320 greedy tokens flipped (2 of 5
      # prompts affected) with max logit delta ~2.9 -- vs 0/320 and ~0.003 for every KV-cache-like buffer
      # above (TransformerBlock.cache_kv, MLATransformerBlock.cache_k, this block's own conv_state). It's
      # also O(1) in max_context (no memory win from halving it), so there's no upside to offset that risk.
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim,
                                          dtype=dtypes.default_float, device=x.device).clone()

def parse_device_map(dm:str|dict[int|str,str], num_blocks:int) -> tuple[list[str], str|None]:
  """Per-block device: "0-15:CPU:0,16-31:CPU:1" (inclusive ranges), "CPU:0,CPU:1" (even split), or {block_idx: device}.
  An optional "experts:<device>" segment (str form) or "experts" key (dict form) routes MoE routed-expert weight
  tensors (ffn_{gate,up,down}_exps) to a separate device, independent of their block's device. The router
  (ffn_gate_inp) always stays with its block. Returns (per-block devices, experts device or None).

  A str `dm` may also carry a trailing "tp:<device>=<fraction>[,<device>=<fraction>...]" segment requesting
  FFN tensor-parallel sharding -- see parse_tp_spec, which parses it from this same string. This function
  only strips it out before doing its own block/experts parsing (same idea as "experts:", but tp's own
  device=fraction list is itself comma-separated, so unlike experts: it can't be isolated as a single
  comma-split token -- it must be the LAST segment in `dm`) and otherwise ignores its content entirely: it
  never appears in this function's own return value, which stays the plain 2-tuple above in every case,
  tp: segment or not -- existing callers that only ever pass a plain (no-tp) `dm` see zero change here."""
  if isinstance(dm, dict):
    experts_dev = dm.get("experts")
    blocks = {k: v for k, v in dm.items() if k != "experts"}
    assert all(i in blocks for i in range(num_blocks)), f"device_map must cover all {num_blocks} blocks: {dm}"
    return [blocks[i] for i in range(num_blocks)], experts_dev

  dm = re.split(r',\s*tp:', dm, maxsplit=1)[0]
  parts = [s.strip() for s in dm.split(",")]
  experts_parts = [p for p in parts if p.startswith("experts:")]
  assert len(experts_parts) <= 1, f"device_map has more than one 'experts:' segment: {dm}"
  experts_dev = experts_parts[0].split(":", 1)[1] if experts_parts else None
  assert experts_dev != "", f"device_map 'experts:' segment needs a device: {dm}"
  block_dm = ",".join(p for p in parts if not p.startswith("experts:"))
  assert block_dm, f"device_map has no block segments (only 'experts:'): {dm}"

  segs = [s.strip().split(":", 1) for s in block_dm.split(",")]
  indexed = [len(s) == 2 and s[0].replace("-", "").isdigit() for s in segs]
  if not any(indexed):
    # no free-memory query exists on Device/Allocator, so auto placement splits evenly by block count
    devs = [s.strip() for s in block_dm.split(",")]
    assert len(devs) <= num_blocks, f"device_map lists {len(devs)} devices for only {num_blocks} blocks: {dm}"
    return [devs[i * len(devs) // num_blocks] for i in range(num_blocks)], experts_dev
  assert all(indexed), f"device_map mixes indexed ('lo[-hi]:device') and plain segments: {dm}"
  out: list[str|None] = [None] * num_blocks
  for rng, dev in segs:
    lo_s, _, hi_s = rng.partition("-")
    lo, hi = int(lo_s), int(hi_s or lo_s)
    assert 0 <= lo <= hi < num_blocks, f"device_map range {rng} out of bounds for {num_blocks} blocks: {dm}"
    assert all(out[i] is None for i in range(lo, hi+1)), f"device_map range {rng} overlaps a previous range: {dm}"
    for i in range(lo, hi+1): out[i] = dev
  assert all(out), f"device_map must cover all {num_blocks} blocks: {dm}"
  return out, experts_dev  # type: ignore[return-value]

def parse_tp_spec(dm:str|dict[int|str,str]) -> tuple[tuple[str, float], ...]|None:
  """Parses the optional "tp:<device>=<fraction>[,<device>=<fraction>...]" segment from a device_map
  string requesting FFN tensor-parallel sharding (T4.70b) -- see parse_device_map, which strips this same
  segment out before doing its own block/experts parsing, and FFNBlock.__init__/`_feed_forward`, which
  consume the result via TransformerConfig.tp. Str form only: a dict-form device_map has no room for a
  multi-pair sub-list (same asymmetry as "experts:<device>" being a string SEGMENT vs the dict form's
  plain "experts" KEY) -- returns None for a dict `dm`.

  Returns an ordered tuple of (device, fraction) pairs -- insertion order IS shard order 0,1,2,...
  (_tp_split_sizes, FFNBlock, and _split_tp_keys all zip over this same order to line shard index up with
  split position) -- or None when no tp: segment is present. Fractions must sum to 1.0 (a small float
  tolerance covers e.g. three 1/3 literals).

  The tp: segment must be the LAST one in `dm`: its own device=fraction list is itself comma-separated, so
  (unlike "experts:<device>", a single token) it can't be isolated by a top-level comma-split -- splitting
  on the first ",tp:" and keeping everything after it is the only unambiguous way to find its extent."""
  if not isinstance(dm, str): return None
  parts = re.split(r',\s*tp:', dm, maxsplit=1)
  if len(parts) == 1: return None
  pairs = []
  for seg in parts[1].split(","):
    dev, eq, frac_s = seg.strip().partition("=")
    assert eq and dev and frac_s, f"device_map 'tp:' segment needs '<device>=<fraction>' pairs: {dm}"
    pairs.append((dev, float(frac_s)))
  total = sum(f for _, f in pairs)
  assert abs(total - 1.0) < 1e-6, f"device_map 'tp:' fractions must sum to 1.0, got {total}: {dm}"
  return tuple(pairs)

def _tp_split_sizes(total:int, tp:tuple[tuple[str, float], ...]) -> list[int]:
  """Splits an FFN's hidden_dim (`total`) into len(tp) contiguous shard widths proportional to each
  device's fraction, each rounded to a multiple of 32. Every shard boundary uses the SAME split points for
  both ffn_gate/up (column-parallel: split their OUTPUT/hidden dim) and ffn_down (row-parallel: split its
  INPUT/hidden dim to match) -- see FFNBlock.__init__. Output-dim slices of a stored GGUF tensor are always
  block-aligned regardless of the cut point (each output row is whole 32-wide quant blocks -- ggml packs a
  Q8_0 block as 32 contiguous elements along a tensor's LAST/input axis, never spanning rows -- see
  tinygrad/llm/gguf.py's ggml_data_to_tensor); ffn_down's input-dim slice is the one that actually NEEDS
  the 32-rounding below, since that axis carries the block structure -- sharing the same split points is
  what makes ffn_down's shard width match ffn_gate/up's h_d width in the first place.

  Only the LAST shard's fraction is implicit (whatever remains after the others round) -- every other
  shard's fraction rounds independently to the nearest block, so shards always sum to exactly `total` even
  though independently-rounded fractions might not (a discrepancy the last shard silently absorbs)."""
  assert total % 32 == 0, f"tp split needs an FFN hidden_dim ({total}) that's already a multiple of 32"
  bounds = [0]
  for _, frac in tp[:-1]: bounds.append(min(total, bounds[-1] + round(total * frac / 32) * 32))
  bounds.append(total)
  sizes = [hi - lo for lo, hi in zip(bounds, bounds[1:])]
  assert all(s > 0 and s % 32 == 0 for s in sizes), \
    f"tp split of hidden_dim={total} over fractions {[f for _, f in tp]} produced a non-positive or unaligned shard: {sizes}"
  return sizes

def _split_tp_keys(state_dict:dict[str, Tensor], tp:tuple[tuple[str, float], ...], num_blocks:int) -> None:
  """Splits each dense block's ffn_{gate,up,down}.weight into per-shard tensors, in place -- the FFN-TP
  counterpart to _rename_mtp_keys below (same trick: pure state-dict key/tensor surgery, no model object
  touched, called from from_gguf right before nn.state.load_state_dict). ffn_gate/up are column-parallel:
  shard d gets OUTPUT rows [lo:hi) of the (hidden_dim, dim)-shaped stored tensor. ffn_down is row-parallel:
  shard d gets the matching INPUT columns [lo:hi) of its (dim, hidden_dim)-shaped stored tensor -- see
  _tp_split_sizes for why both use the same split points. New keys (blk.{i}.ffn_{gate,up,down}_tp.{d}.weight)
  match FFNBlock's `ffn_{gate,up,down}_tp: list[Linear]` attributes -- nn.state's list-attribute key
  numbering (get_state_dict) lands on exactly these paths, so a plain load_state_dict finds them.

  Skips any block with no ffn_gate.weight key (an MoE block, addressed via ffn_gate_exps/... instead) --
  unreachable via from_gguf/Transformer.__init__ in practice (a tp spec together with num_experts>0
  anywhere in the config is rejected at Transformer construction, before this ever runs), but this
  function only has the state_dict's own keys to go by, no config, so it stays defensive rather than
  assuming that guard was necessarily the caller."""
  for i in range(num_blocks):
    gate_k, up_k, down_k = f"blk.{i}.ffn_gate.weight", f"blk.{i}.ffn_up.weight", f"blk.{i}.ffn_down.weight"
    if gate_k not in state_dict: continue
    gate, up, down = state_dict.pop(gate_k), state_dict.pop(up_k), state_dict.pop(down_k)
    sizes = _tp_split_sizes(cast(int, gate.shape[0]), tp)  # a loaded GGUF weight's shape is always concrete, never symbolic
    lo = 0
    for d, sz in enumerate(sizes):
      hi = lo + sz
      state_dict[f"blk.{i}.ffn_gate_tp.{d}.weight"] = gate[lo:hi]
      state_dict[f"blk.{i}.ffn_up_tp.{d}.weight"] = up[lo:hi]
      state_dict[f"blk.{i}.ffn_down_tp.{d}.weight"] = down[:, lo:hi]
      lo = hi

def _rename_mtp_keys(state_dict:dict[str, Tensor], num_blocks:int) -> None:
  """Renames blk.{num_blocks}.* (the nextn/MTP block a qwen3.5-family GGUF carries beyond the main
  num_blocks) onto MTPHead's attribute paths, in place -- see MTPHead's docstring for the forward
  semantics. blk.{n}.nextn.X -> mtp_head.X (the 4 MTP-specific tensors); every other blk.{n}.X ->
  mtp_head.block.X (X unchanged: the inner block's attribute names already match a main block's, e.g.
  attn_q, ffn_gate -- from_gguf's post_attention_norm->ffn_norm rename runs first and applies here too).
  Pure key manipulation -- never touches a tensor's value, so this is cheap to exercise against just a
  real GGUF's tensor-name list (no data realized) as well as against a real load.
  # ponytail: only the first nextn layer (blk.{num_blocks}) is modeled -- nextn_predict_layers>1 (not
  # seen in any real qwen3.5-family checkpoint so far) would need mtp_head to be a list of these."""
  prefix = f"blk.{num_blocks}."
  for k in [k for k in state_dict if k.startswith(prefix)]:
    rest = k[len(prefix):]
    new_k = f"mtp_head.{rest[len('nextn.'):]}" if rest.startswith("nextn.") else f"mtp_head.block.{rest}"
    state_dict[new_k] = state_dict.pop(k)

def linear_rows(lin:Linear, ids:Tensor) -> Linear:
  """T4.142: a bias-free Linear holding rows `ids` ((S,) int, on lin's device) of lin's weight, in lin's own storage. A ggml-quantized
  weight (from_gguf's lazy dequant view of the GGUF bytes, or set_quantized's packed view of them) gathers whole rows of its RAW blocks
  -- row r of an (out, in) ggml tensor is the in/block_elems consecutive blocks at r * row_bytes (Q6_K at in=5120: 20 x 210 bytes) --
  into one realized byte buffer (a single byte-gather kernel, S x row_bytes; nothing is dequantized) and rebuilds the same dequant view
  over it (gguf.py ggml_data_to_tensor), so set_quantized finds the same format and the NV decode gemv runs the slice like any head.
  A dense weight -- or a format set_quantized does not know (then dequantized, those S rows only) -- gathers its rows, realized.
  Linear.__new__: no random init (it would advance the default device's RNG counter)."""
  assert lin.bias is None, "linear_rows: the output head has no bias"
  probe = Linear.__new__(Linear)  # set_quantized's format probe, on the pre-quantization graph if lin already ran it
  probe.set_quantized(lin.weight if lin.dequant_weight is None else lin.dequant_weight)
  out, S = Linear.__new__(Linear), cast(int, ids.shape[0])
  out.in_features, out.out_features, out.bias = lin.in_features, S, None
  if probe.ggml_type is None: out.weight = lin.weight[ids].contiguous().realize()
  else:
    raw = cast(Tensor, probe.packed_bytes if probe.packed_bytes is not None else probe.weight)  # uint8: the whole tensor's blocks
    rows = raw.reshape(lin.out_features, -1)[ids].reshape(-1).contiguous().realize()
    out.weight = ggml_data_to_tensor(rows, S * lin.in_features, probe.ggml_type).reshape(S, lin.in_features) \
      .cast(cast(Tensor, probe.dequant_weight).dtype)
  return out

class MTPHead:
  """DeepSeek-style multi-token-prediction ("nextn") head (qwen3.5-family GGUFs -- see from_gguf's
  MTP=1 branch and _rename_mtp_keys for how blk.{num_blocks}.* lands here). llama.cpp-compatible
  forward: eh_proj(concat(enorm(embed(next_tok)), hnorm(last_hidden))) -> `block` (a real attention
  block, built the same way the main Transformer builds one for this arch/config -- see from_gguf;
  it keeps its OWN KV cache over drafted positions, separate from the main model's blocks) ->
  shared_head_norm -> the OWNING Transformer's output head. That last step is output-norm-free: this
  head has its own learned final norm (shared_head_norm) instead of reusing the main model's
  output_norm, same as llama.cpp's DeepSeek-MTP implementation. One token of lookahead per call;
  T4.64 chains calls (start_pos, start_pos+1, ...), feeding each call's own sampled token back in
  (T4.136: and its block output as the next call's hidden -- draft_step, SPEC_CHAIN_H).

  `draft` takes the owning Transformer as an explicit argument rather than storing it on self: this
  object hangs off Transformer.mtp_head, and nn/state.py's get_state_dict recurses through every plain
  attribute with no cycle guard -- a stored back-reference would make it walk
  model -> mtp_head -> owner -> model -> ... forever the moment anything asks for this model's
  state/parameters (load_state_dict, realize_placement, get_parameters, ...)."""
  # T4.142: the draft vocabulary (set_draft_vocab; class-level None = the owner's full head, and nothing extra in get_state_dict)
  draft_output:Linear|None = None
  draft_ids:Tensor|None = None
  draft_inv:Tensor|None = None
  def __init__(self, config:TransformerConfig, block_cls:type[FFNBlock]=TransformerBlock):
    self.enorm, self.hnorm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.eh_proj = Linear(2 * config.dim, config.dim, bias=False)
    self.block = block_cls(config)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)

  def set_draft_vocab(self, owner:Transformer, ids:list[int]) -> None:
    """T4.142 (SPEC_DRAFT_VOCAB, from_gguf): restrict the draft steps' output head to the token ids `ids` -- draft_output = the owner's
    head's rows at those ids (linear_rows: its own ggml format, realized once on the head's device), draft_ids (S,) the sorted ids and
    draft_inv (vocab,) each id's column in the slice (S off the set; full_vocab), both int32 on the same device. The owner's output is
    untouched (VERIFY, plain decode). On a card, call it before prepare_dense_weights (from_gguf does) so the slice's format probe runs
    at load, outside any capture."""
    vocab, dev = owner.output.out_features, owner.output.weight.device
    ids = sorted(set(ids))
    assert ids and ids[0] >= 0 and ids[-1] < vocab, f"draft vocabulary ids must lie in [0, {vocab}): got {ids[0]}..{ids[-1]}"
    inv = [len(ids)] * vocab
    for s, i in enumerate(ids): inv[i] = s
    self.draft_ids, self.draft_inv = (Tensor(x, dtype=dtypes.int32, device=dev).realize() for x in (ids, inv))
    self.draft_output = linear_rows(owner.output, self.draft_ids)

  def full_vocab(self, logits:Tensor) -> Tensor:
    """T4.142: draft_step's logits (B,T,S) over the draft vocabulary -> (B,T,vocab), -inf off the set: the q spec_accept_tensor tests
    (softmax is exactly 0 there, so a token outside the set is only ever emitted by the residual resample, p - q = p). One gather by
    draft_inv from the slice padded with a -inf column -- a single vocab-wide elementwise kernel. The identity without a draft vocab."""
    if self.draft_inv is None: return logits
    return logits.pad(((0, 0), (0, 0), (0, 1)), value=float("-inf"))[:, :, self.draft_inv]

  def draft(self, owner:Transformer, h:Tensor, tok_ids:Tensor, start_pos:int|UOp) -> Tensor:
    """h: (B,1,dim) the main model's last-layer hidden state (pre-output-norm) at position t -- or, T4.136, the draft
    chain's own: under SPEC_CHAIN_H=1 step i+1 of Transformer._draft_chain takes step i's block output (draft_step's x).
    tok_ids: (B,1) int, the token actually at position t+1. Returns (B,1,vocab) draft logits
    predicting position t+2. `block`'s KV cache advances with start_pos across calls, same as a
    main block's (see FFNBlock._init_state/__call__) -- so calling this again at start_pos+1
    continues the same drafted sequence instead of overwriting it. T4.136: = draft_step(...)[0]; the chain calls
    draft_step (for x too) unless this method is patched on the INSTANCE -- test_spec_decode.py's oracles return
    logits only, and the chain reuses h for them. T4.142: under a draft vocabulary, full_vocab(draft_step(...)[0]) --
    still (B,1,vocab), -inf off the set.

    start_pos accepts a bound UOp Variable, not just a plain int (T4.66): TransformerBlock/
    MLATransformerBlock already handle one end-to-end (the same way the main model's own decode step
    does), so passing one here lets mtp_head.block's @function(precompile=True) trace replay across
    drafted positions instead of retracing per position -- see speculative_generate's `v_draft_pos`
    and SPEC_NOTES.md §4. test_mtp_load.py exercises both calling conventions directly.

    T4.66d: the embedding lookup used to be `owner.token_embd(tok_ids.to(owner.token_embd.weight.device))`
    -- SPEC_PROFILE_NOTES.md's leading candidate for DRAFT's own per-call cost, confirmed dominant on real
    METAL+NV hardware (~280ms/call, see SPEC_FIXES_NOTES.md): tok_ids is always produced on `dev` (either
    `tok_last`, from the main model's own output layer, or this method's own previous return -- both pinned
    to `owner.output`'s device regardless of where mtp_head itself lives), so hopping it to
    owner.token_embd's device and hopping the embedded vector back is two real cross-device syncs, every
    call. A lazily-built, device-local copy of the whole embedding table (self._local_token_embd, built once
    -- a model that never drafts never pays for it) makes tok_ids' home device and the lookup device the
    SAME one: zero hops for this step, ever again. Considered moving mtp_head to owner.token_embd's device
    instead (the other option this task named): doesn't work -- tok_ids' home device is pinned by
    owner.output (unmoved, still needed by the main model's own forward), so that would only relocate this
    same hop to the `owner.output.weight.device` line below, not remove it. See SPEC_FIXES_NOTES.md for the
    byte-size accounting (this fork always serves a quantized GGUF, so the copy is the quantized size, not a
    dequantized fp32 one -- `.to()` replicates whatever owner.token_embd.weight is resident as, verbatim, no
    new precision decision here)."""
    return self.full_vocab(self.draft_step(owner, h, tok_ids, start_pos)[0])

  def draft_step(self, owner:Transformer, h:Tensor, tok_ids:Tensor, start_pos:int|UOp) -> tuple[Tensor, Tensor]:
    """T4.136: draft() plus the block's output x (B,1,dim) on self.block's device, before shared_head_norm -- the hidden the draft
    chain's next step takes (SPEC_CHAIN_H). Both lazy: in the chain's graph x feeds the next step's hnorm directly (same device).
    T4.142: under a draft vocabulary the logits are draft_output's, (B,1,S) -- column s is token draft_ids[s] (full_vocab maps back)."""
    x = self.block(self._block_in(owner, h, tok_ids), start_pos)
    head = owner.output if self.draft_output is None else self.draft_output
    return head(self.shared_head_norm(x).to(owner.output.weight.device)), x

  def _block_in(self, owner:Transformer, h:Tensor, tok_ids:Tensor) -> Tensor:
    """eh_proj(concat(enorm(embed(tok_ids)), hnorm(h))) on self.block's device -- (B,T,dim) f32 from h (B,T,dim) and
    tok_ids (B,T); the shared front half of draft() (T=1) and prefill() (T symbolic). See draft()'s T4.66d note on
    _local_token_embd."""
    dev = self.block.device
    e = self.local_embedding(owner)(tok_ids.to(dev)).float()
    return self.eh_proj(self.enorm(e).cat(self.hnorm(h.to(dev).float()), dim=-1))

  def local_embedding(self, owner:Transformer) -> nn.Embedding:
    """the owner's token embedding on self.block's device (T4.66d), copied once on first use. T4.132: on the same device the owner's own
    (a lazy dequant view of the GGUF bytes, gathered per row like the main model's lookup) -- realizing it was the whole fp16 table,
    248320 x 5120 = 2.4 GiB on the card, the 262k window's OOM"""
    if owner.token_embd.weight.device == self.block.device: return owner.token_embd
    if not hasattr(self, "_local_token_embd"):
      local = nn.Embedding.__new__(nn.Embedding)
      local.weight = owner.token_embd.weight.to(self.block.device).realize()
      self._local_token_embd = local
    return self._local_token_embd

  def ready(self, owner:Transformer, max_width:int) -> None:
    """T4.132: everything a draft step / the accept's cache rewrite (widths 1..max_width) would create or realize on its first call, done
    now: those graphs capture on their FIRST call without pruning (their sampled draws need Tensor.rand's counter advance in every
    replay), so first-call one-time work would replay every iteration -- the KV cache's zero-fill above all (a 1-token prompt never runs
    MTPHead.prefill, which otherwise creates it). The block's cache, scale and rope table, the position buffers, the local embedding."""
    dev = self.block.device
    self.block._init_state(Tensor.empty(1, 1, owner.token_embd.weight.shape[-1], device=dev))
    Tensor.realize(*(t for a in ("cache_kv", "cache_kv_scale", "freqs_cis") if isinstance(t := getattr(self.block, a, None), Tensor)))
    for w in range(1, max_width + 1): positions(self.block.config.max_context + w, dev)
    self.local_embedding(owner)

  def prefill(self, owner:Transformer, h:Tensor, tok_next:list[int], start_pos:int, v_toks:UOp, v_pos:UOp) -> None:
    """T4.66l: fill `block`'s KV cache for T consecutive MTP steps at once -- what draft() does for one step, minus
    the head. h: (1,T,dim) the main model's pre-output_norm hidden at positions p..p+T-1; tok_next: the T tokens
    actually at positions p+1..p+T; start_pos = p+1, the cache slot of the first step (draft()'s own
    convention: the step whose inputs are (h_t, tok_{t+1}) lives in slot t+1 -- slot 0 is never written). This is the
    context the nextn block was trained to attend over; before T4.66l the prompt slots stayed all-zero (P zero keys
    diluting every draft's attention -- MTP_CACHE_NOTES.md section 2/6). Returns the block's (B,T,dim) output; callers
    realize it only for the K/V store it carries. T is symbolic (v_toks, bounded by the prefill chunk width; v_pos
    bounds the slot) so every width 1..bound replays ONE precompiled trace -- the same reason draft() takes a
    Variable start_pos (see there); the pad below is what gives the symbolic slice a concrete buffer to view.
    h's shape must be concrete: callers slice a replayed key's frozen output with host-known widths (`[:, :n]`,
    T4.66h/j) before passing it here. Both symbolic-width views are shrinks of CONTIGUOUS concrete buffers -- the
    main prefill's own `tokens.contiguous()` pattern (Transformer.__call__): a pad folded straight into a
    symbolic-range kernel is a gated load the range simplifier rejects (codegen/simplify.py mark_gated, `.val` on
    the bound PARAM -- reproduced on the very first norm kernel), so h's pad is materialized by its own concrete
    copy kernel first (one small kernel per distinct T) and the token row is built padded on the host."""
    B, T, D = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
    assert B == 1 and len(tok_next) == T, f"{B=} {len(tok_next)=} {T=}"
    bound, dev = cast(int, v_toks.vmax), self.block.device
    hs = fresh_copy(h.to(dev).float().pad_to((B, bound, D)))[:, :v_toks.bind(T)]
    ts = Tensor([tok_next + [0] * (bound - T)], dtype="int32", device=dev)[:, :v_toks.bind(T)]
    # T4.132: the block's precompiled trace was re-run in python on every call (the 27B's full-accept state_assign_ms, ~0.5 s on the
    # card); one TinyJit per `bound` replays it -- the two symbolic views above are its inputs (shrinks of realized buffers, the
    # variables unbound in the input signature), so every width 1..bound and every slot replay one graph
    owner._spec_replay(("mtp_prefill", bound), functools.partial(self._prefill_body, owner), hs, ts, v_pos.bind(start_pos), prune=True)

  def _prefill_body(self, owner:Transformer, hs:Tensor, ts:Tensor, pos:UOp) -> None:
    # T4.132: realized for its K/V store and dropped, never a jit output: a returned block output stays the precompiled call's lazy
    # GETTUPLE graph for as long as the jit lives, and every later realize walks every live Tensor's graph (_apply_map_to_tensors) --
    # on the card ~57k nodes per plain decode step after warmup(spec_k=3) (27.0 -> 34.6 ms/token)
    cast(Tensor, self.block(self._block_in(owner, hs, ts), pos)).realize()

def spec_accept_tensor(draft_ids:Tensor, q_logits:Tensor, p_logits:Tensor, temperature:float|Tensor, u_accept:Tensor, u_sample:Tensor,
                       min_p:float=0.0) -> tuple[Tensor, Tensor]:
  """Leviathan et al. ("Fast Inference from Transformers via Speculative Decoding", 2023) speculative-sampling accept/resample
  test -- on device, batched over a leading dim (T4.132; T4.65's host-side numpy form before it) -- speculative_generate's
  temperature>0 path (see SPEC_NOTES.md section 6 for how this wires into the position ledger).

  draft_ids (B,k) int: the drafted ids. THE CALLER must have drawn them by ancestral sampling from q (draft_ids[i] ~
  softmax(q_logits[i]/temperature)) -- this function only implements the accept/reject/resample test; the proof below assumes it.
  q_logits (B,k,V): the draft logits at each drafted position. p_logits (B,k+1,V): the verify (main-model) logits at each of the
  k+1 verify positions, row k being the "bonus" position. u_accept (B,k), u_sample (B,k+1,V): uniforms in [0,1) -- Tensor.rand
  (seeded for a test).

  Returns (m, final): m (B,) int in 0..k is how many drafted tokens are accepted; final (B,k+1) int holds, at i < k, the token
  emitted if the first rejection is at i -- a sample from normalize(max(0, p_i-q_i)), the "residual" distribution -- and at k the
  bonus, a fresh sample from p_k. The caller emits draft_ids[:m] + [final[m]] (m+1 tokens). Accept drafted token i with probability
  min(1, p_i[d]/q_i[d]): u < min(1, p/q) <=> log u < log p - log q (u < 1 always), so both sides are log-softmaxes -- nothing is
  normalized over the vocabulary on the host. Every sample is a Gumbel-max draw, argmax(log r + g) with g = -log(-log u), a sample
  from r/sum(r) (sample_logits' own transform); log 0 = -inf keeps an excluded token out, and an all-zero residual row (p == q
  exactly) falls back to argmax p, as the numpy form did. min_p > 0 (T4.132: generate()'s MIN_P tail cut, sample_logits) is applied
  to p and q alike before the softmax, so the emitted marginal is generate(temperature, MIN_P)'s truncated p (the proof holds for
  any p/q; the caller must have drawn draft_ids from the same truncated q).

  Proof this reproduces p exactly (Leviathan et al., Theorem 1): accept x with probability min(1,p(x)/q(x)),
  so P(emit x, accepted) = q(x)*min(1,p(x)/q(x)) = min(p(x),q(x)). P(reject) = 1 - sum_y min(p(y),q(y)), and
  sum_y max(0,p(y)-q(y)) = sum_y [p(y) - min(p(y),q(y))] = 1 - sum_y min(p(y),q(y)) = P(reject) too, so
  normalize(max(0,p-q)) is a valid distribution and P(emit x, rejected) = P(reject) * max(0,p(x)-q(x))/P(reject)
  = max(0,p(x)-q(x)). Total: P(emit x) = min(p(x),q(x)) + max(0,p(x)-q(x)) = p(x) (either p(x)<=q(x), giving
  p(x)+0, or p(x)>q(x), giving q(x)+(p(x)-q(x))) -- true for every x regardless of q, which is why draft
  quality only ever affects the acceptance RATE, never correctness. test_spec_decode.py's statistical test
  checks this empirically since it's the theorem the whole sampled path's correctness rests on."""
  B, k = cast(int, draft_ids.shape[0]), cast(int, draft_ids.shape[1])
  def gumbel(u:Tensor) -> Tensor: return u.maximum(1e-12).log().neg().log().neg()
  def log_dist(logits:Tensor) -> Tensor:  # log softmax(logits/temperature), the min-p tail cut first (sample_logits' own mask)
    scaled = logits / temperature
    if min_p > 0: scaled = (scaled >= scaled.max(-1, keepdim=True) + math.log(min_p)).where(scaled, float("-inf"))
    return scaled.log_softmax(-1)
  lp = log_dist(p_logits)                                                                  # (B,k+1,V)
  bonus = (lp[:, k] + gumbel(u_sample[:, k])).argmax(-1, keepdim=True)                      # (B,1) ~ p_k
  if k == 0: return Tensor.zeros(B, dtype=dtypes.int32, device=draft_ids.device), bonus
  lq = log_dist(q_logits)                                                                  # (B,k,V)
  idx = draft_ids.unsqueeze(-1)
  # <= not <: a fp32 uniform can round to exactly 1.0 (log 0) and p == q (lp - lq == 0) must accept it as any u < 1 would
  accept = u_accept.maximum(1e-12).log() <= lp[:, :k].gather(-1, idx).squeeze(-1) - lq.gather(-1, idx).squeeze(-1)  # (B,k)
  m = accept.cast(dtypes.int32).cumprod(-1).sum(-1)                                        # the first rejection's index, or k
  residual = (lp[:, :k].exp() - lq.exp()).relu()                                           # (B,k,V) max(0, p-q)
  resample = (residual.log() + gumbel(u_sample[:, :k])).argmax(-1)                         # (B,k) ~ residual/sum(residual)
  correction = (residual.sum(-1) > 0).where(resample, lp[:, :k].argmax(-1))
  # .contiguous() before the cat: an argmax over rows holding -inf fused straight into a cat/stack came back out of range (id == V)
  # on CPU (tinygrad, 10-29) -- materialized first, the ids are right (TestSpecAcceptTensor pins the resample distribution)
  return m, correction.contiguous().cat(bonus.contiguous(), dim=-1)

MIN_P = getenv("MIN_P", 0.0)  # T4.85: min-p tail cut for sampled decoding (0 = off, byte-identical to the plain Gumbel-max sampler)

def sample_logits(logits:Tensor, temperature:Tensor, min_p:float=0.0) -> Tensor:
  """One sampled token id per row of `logits` (B, vocab) -> (B, 1). Gumbel-max: argmax(logits/temp - log(-log(u))) is a sample
  from softmax(logits/temp). min_p > 0 first drops every token whose probability is below min_p x the top token's (the tail
  that makes a 0.6-temperature sample emit garbage); there is no sort here (the tensor library has none), so this is the
  sort-free stand-in for top-k/top-p. The top token always survives, so the masked argmax is always defined.
  T4.125c: `probs >= probs.max() * min_p` is the same test as `scaled >= scaled.max() + log(min_p)` (softmax is monotone
  in scaled and its normalizer cancels out of the ratio), so the mask is computed directly on `scaled` -- no exp/sum/divide
  over the whole vocab."""
  scaled = logits / temperature.maximum(1e-12)
  if min_p > 0:
    scaled = (scaled >= scaled.max(-1, keepdim=True) + math.log(min_p)).where(scaled, float("-inf"))
  return (scaled - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

# T4.99: the device snapshot_state's clones are parked on. "" (default) leaves them on the live buffers' own
# device, byte-identical to pre-T4.99; e.g. "CPU" frees the pooled map's device memory for weights/KV and uses
# host RAM instead (restore_state copies each one back to its live buffer's device before the assign, see there).
# On the pooled map the METAL share IS host memory already, so keep STATE_CACHE_MB modest (2048) there too --
# only an all-on-card layout (every block on one discrete GPU) has host RAM to spare for a much bigger cap.
STATE_CACHE_DEVICE = getenv("STATE_CACHE_DEVICE", "")

class Transformer:
  def __init__(self, config:TransformerConfig, device_map:str|dict[int|str,str]|None=None):
    # T4.70b: FFN tensor-parallel spec, parsed from device_map's optional "tp:" segment (see parse_tp_spec)
    # BEFORE dense_config/config get their usual replace()s below, so config.tp reaches every block's
    # _cfg(i) config the same way every other field does -- no separate constructor parameter needed (see
    # TransformerConfig.tp's own comment for why this lives on the config instead). tp is None (the
    # overwhelmingly common case, and every existing caller) leaves `config` completely untouched here --
    # byte-identical to pre-T4.70b behavior. MoE/shared-expert FFN sharding is explicitly out of scope
    # (Megatron-shape column/row-parallel dense FFN only) -- raise here, at construction, rather than
    # silently no-op or build something half-wired.
    tp = parse_tp_spec(device_map) if device_map is not None else None
    if tp is not None:
      assert config.num_experts == 0, "device_map 'tp:' (FFN tensor-parallel) doesn't support MoE models " \
        "(num_experts>0 anywhere in the config) -- dense-FFN-only, see T4.70b's scope guard"
      config = replace(config, tp=tp)
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    def _cfg(i:int) -> TransformerConfig:
      cfg = dense_config if i < config.leading_dense_blocks else config
      # sliding_layers marks which layers use the windowed attention span; others get the full-context mask
      if config.sliding_layers and not config.sliding_layers[i]: cfg = replace(cfg, sliding_window=0)
      return cfg
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(_cfg(i), config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(_cfg(i))
                               for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.mtp_head: MTPHead|None = None  # set by from_gguf when MTP=1 and the GGUF has a nextn block (T4.63)
    self.nv_output_norm_weight: Tensor|None = None  # T4.125b: output_norm's fp32 copy for nv_fused_norm, set by from_gguf
    self.max_context = config.max_context
    # set by the device_map branch below; None means "no device_map", the fast/no-op path for realize_placement()
    self._placed_devices: frozenset[str]|None = None
    if device_map is not None:
      dmap, experts_dev = parse_device_map(device_map, config.num_blocks)
      for block, dev in zip(self.blk, dmap):
        for p in nn.state.get_parameters(block): p.to_(dev)
      # token_embd feeds the first block; output_norm/output consume the last block's activations
      for p in nn.state.get_parameters(self.token_embd): p.to_(dmap[0])
      for p in nn.state.get_parameters([self.output_norm, self.output]): p.to_(dmap[-1])
      # routed-expert weights (not the router, which stays with attention) move to a separate device on
      # top of the per-block placement above -- overrides ffn_{gate,up,down}_exps' dev for every MoE block
      if experts_dev is not None:
        for block in self.blk:
          if hasattr(block, 'ffn_gate_exps'):
            for p in nn.state.get_parameters([block.ffn_gate_exps, block.ffn_up_exps, block.ffn_down_exps]): p.to_(experts_dev)
      # T4.70b: FFN tensor-parallel shards move to THEIR tp device on top of the per-block placement above
      # (same override pattern as experts_dev, one level deeper: len(tp) shards per block instead of one
      # device) -- the generic per-block loop just moved every ffn_{gate,up,down}_tp[d] to the block's OWN
      # device like any other param, undoing the whole point of splitting them across devices.
      if tp is not None:
        for block in self.blk:
          if hasattr(block, 'ffn_gate_tp'):
            for gate, up, down, (dev2, _) in zip(block.ffn_gate_tp, block.ffn_up_tp, block.ffn_down_tp, tp):
              for p in nn.state.get_parameters([gate, up, down]): p.to_(dev2)
      # canonicalized so it compares equal to p.device later (e.g. "CPU:0" -> "CPU") -- realize_placement()'s footgun guard
      self._placed_devices = frozenset(Device.canonicalize(d) for d in dmap + ([experts_dev] if experts_dev is not None else [])
                                       + ([d for d, _ in tp] if tp is not None else []))
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # T5.3: rope-position deficit at the end of _cached_tokens -- sum over the images in it of (visual tokens - rope span), 0 for text
    self._rope_delta: int = 0
    # we specialize the JIT for prefill/rollout, sampled/greedy, spec (T4.64) and vision (T5.3); prefill also keys on
    # chunk_size (T4.12) -- created lazily in __call__ since chunk_size isn't known until generate() picks one
    self.jit: dict[tuple, Callable[..., Tensor|tuple[Tensor, ...]]] = {}
    # T4.132: speculative_generate's own captured graphs -- ("chain", k_eff) the draft chain, ("accept", n, greedy) the accept test
    # + readback + the next chain's inputs + the MTP cache rewrite, ("adopt",) a partial accept's GDN state, ("mtp_prefill", bound)
    # the prompt's MTP cache fill (MTPHead.prefill) -- kept apart from self.jit, whose keys are Transformer.__call__'s families
    self.spec_jit: dict[tuple, Callable[..., Tensor|tuple[Tensor, ...]|None]] = {}
    self.spec_slots = 0  # T4.132: GatedDeltaNetBlock.spec_track's slot count, see _spec_track_slots

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, spec:bool=False, rope_start:int|UOp|None=None,
              vis_e:Tensor|None=None, vis_m:Tensor|None=None, vis_pos:Tensor|None=None,
              presence_penalty:Tensor|None=None) -> Tensor|tuple[Tensor, ...]:
    # contract: temperature=None is the ONLY greedy trigger. it's a python-level check (not a value check) because it
    # picks which jit variant gets captured (with or without RNG kernels) -- a Tensor of value 0.0 still takes the
    # sampled path below. callers must normalize temp<=0 to None themselves (generate() already does this)
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()  # (B, T, D)
    # T5.3: rope_start (a bound Variable from generate()) is the rope position of tokens[0] when it differs from the cache position
    # (decode after images); None keeps the pre-T5.3 graph exactly. vis_e/vis_m/vis_pos make this an image-prompt chunk: (1, chunk, D)
    # visual embeddings replace the token embeddings where the (1, chunk) mask is set, and the (1, chunk, 3) int32 (t, h, w) rope
    # positions ride to the attention blocks (GDN blocks ignore them). All three are chunk-max sized; T slices them like tokens.
    rope: int|UOp|Tensor = start_pos if rope_start is None else rope_start
    if vis_e is not None:
      assert vis_m is not None and vis_pos is not None, "image-prompt chunk needs vis_e, vis_m and vis_pos together"
      T = tokens.shape[1]
      x = vis_m[:, :T].to(x.device).unsqueeze(-1).where(vis_e[:, :T].to(x.device).float(), x)
      rope = vis_pos[:, :T]
    # T4.132: a concrete multi-token spec call is speculative_generate's VERIFY: every GDN block also stores its state after each of the
    # run's first T-1 tokens (GatedDeltaNetBlock.spec_track), which a partial accept adopts (_adopt_step) -- no checkpoint, no REDO
    # forward. The symbolic prefill chunks (and T == 1) have no partial accept and skip it.
    track = spec and isinstance(T_in := tokens.shape[1], int) and T_in > 1
    slots = self.spec_slots or GDN_DECODE_TOKENS - 1  # see _spec_track_slots
    # activations hop devices at block boundaries (.to is a no-op when the device matches)
    for block in self.blk:
      xin, r = x.to(block.device), rope.to(block.device) if isinstance(rope, Tensor) else rope
      x = cast(Tensor, block(xin, start_pos, r, slots) if track and isinstance(block, GatedDeltaNetBlock) else block(xin, start_pos, r))
    if spec:
      # T4.64: speculative_generate's verify/prefill-tail path needs (a) the model's per-position output (to
      # check a chained MTP draft against what the main model actually says there) and (b) the pre-output_norm
      # hidden state (to seed the next draft chain -- the same `x` the default path below slices before
      # norm+head). Applying output_norm+output to every position instead of just the last is strictly more
      # work, never less, and this whole branch is behind a flag every EXISTING caller leaves False -- so the
      # default path below stays byte-for-byte what it was.
      # T4.65: this used to return the per-position ARGMAX id (greedy-only). It now returns the raw per-position
      # LOGITS instead -- speculative_generate derives the greedy id from them (argmax, moved to the caller,
      # same value either way) and, for its sampled-acceptance path (temperature>0), the full softmax
      # distributions Leviathan et al.'s accept/resample test needs (see spec_accept below, SPEC_NOTES.md §6).
      # forward() itself still never samples here: it always traces greedy (enforced by the assert below), since
      # sampled acceptance is pure host-side numpy math over these logits -- no RNG kernel to capture either way.
      assert temperature is None, "spec=True always traces greedy -- speculative_generate's sampled path (T4.65) " \
        "computes acceptance host-side from these logits, it never needs forward() itself to sample"
      # no .contiguous() on the logits (unlike x below): self.output(...) is a fresh matmul over the whole
      # (unsliced) x, real computed output that owns its buffer, not a view into a larger scratch buffer --
      # same reasoning SPEC_NOTES.md §3 already gives for why the old argmax result never needed this either.
      # .contiguous(): x is a zero-copy VIEW into its own buffer (an offset/width depending on the bound
      # `toks` variable) -- fine for the default path below, which consumes it inline in the SAME jit call
      # that produced it, but this hidden state is carried OUT of this call (into speculative_generate's next-
      # iteration draft, or its ACCEPT step's position-m slice) and read back only after further replays of
      # this same jit graph have reused x's buffer for a new chunk. Without materializing it into its own
      # buffer here (same as the logits already are, being real computed output rather than a view), that
      # later read sees whatever the intervening replays overwrote instead of this call's own hidden state --
      # confirmed by a byte-level cache_kv diff between with/without .contiguous() on this line (SPEC_NOTES.md).
      # T4.66b: returns the FULL per-position hidden (B,T,D), not just x[:, -1:] (B,1,D) -- speculative_generate's
      # full-accept ACCEPT step needs the hidden state at whichever position m turns out to be the accept length
      # (m==k_eff there, so this is just the last position, but the slice-by-m expression is shared with T4.66b's
      # since-reverted partial-accept path -- see SPEC_NOTES.md/SPEC_FIXES_NOTES.md); T4.132's accept replay gathers
      # row m on device. Callers that only ever wanted the last position (the prefill tail) slice `[:, -1:]` themselves.
      # T4.132: the head's norm fused with its q8 quantize at a concrete small width (nv_fused_norm gates; symbolic T = plain norm)
      return self.output(nv_fused_norm(self.output_norm, self.nv_output_norm_weight, x, self.output)), x.contiguous()  # (B,T,vocab), (B,T,D)
    # only run the output projection on the last token
    logits = self.output(nv_fused_norm(self.output_norm, self.nv_output_norm_weight, x[:, -1:], self.output))[:, -1, :]  # T4.125b
    # T4.92: presence penalty -- self.penalty_mask is a (1,vocab) 0/1 buffer, reset in place at the start of generate() and
    # updated in its eager decode loop (never inside this jit'd graph -- see generate()'s own comment); a plain READ here is
    # exactly like reading cache_kv, no AFTER-chaining needed. presence_penalty is a Tensor (not a python float), same reason
    # temperature is: its MAGNITUDE can then vary across replays of this same captured graph -- only "is it None" (part of
    # __call__'s jit key) may ever change which graph gets captured. None (every pre-T4.92 caller) skips this entirely.
    if presence_penalty is not None: logits = logits - self.penalty_mask.to(logits.device) * presence_penalty.to(logits.device)
    # greedy: temperature None (explicit-greedy callers, the spec path) -> plain argmax, no RNG kernels. generate() always passes a
    # Tensor (T4.105), so its graph carries both the argmax and the Gumbel sampler and picks at runtime: temperature <= 0 = greedy.
    # The unused branch costs a vocab-sized argmax/softmax per token (~1 % of a decode step); it buys one jit family instead of two.
    greedy = logits.argmax(-1, keepdim=True)
    if temperature is None: tok = greedy
    else:
      temperature = temperature.to(logits.device)
      tok = (temperature > 0).reshape(1, 1).where(sample_logits(logits, temperature, MIN_P), greedy)
    # T4.132: with an MTP head every plain call also returns its per-position pre-norm hidden (B,T,D) -- MTPHead.prefill's input -- so
    # speculative_generate's prompt prefill replays generate()'s own families (the prefill family captured first and pruned: the T4.128
    # memory path; at a 1-token chunk width, CPU's, the decode family) instead of keeping two more 128-wide spec families. The token is
    # unchanged; x is already the last block's materialized output, so the cost is that buffer staying a held jit output (2.6 MB per
    # 128-token chunk at 27B geometry, 20 KB per decode step).
    return (tok, x.contiguous()) if self.mtp_head is not None else tok

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, spec:bool=False, rope_start:int|UOp|None=None,
               vis_e:Tensor|None=None, vis_m:Tensor|None=None, vis_pos:Tensor|None=None,
               presence_penalty:Tensor|None=None) -> Tensor|tuple[Tensor, ...]:
    is_prefill = bool(resolve(tokens.shape[1] != 1))
    # T4.12: a prefill/first-chunk call's `tokens` is a symbolic slice carrying the "toks" Variable, whose bound
    # range IS chunk_size -- captured into the jit graph. Keying only on (is_prefill, greedy) let a later
    # generate() at a different chunk_size replay a jit captured with the wrong Variable range -> JitError
    # ("args mismatch in JIT"). Decode/rollout steps feed a plain (1,1) tensor (no bound Variable, chunk_size-
    # independent), so only the prefill variants need the extra key; None keeps the rollout jit singular.
    # T4.64: `spec` also joins the key -- it picks a python branch inside forward() that changes the traced
    # graph (and its return arity), so a spec=True call must never replay a spec=False capture or vice versa.
    # T5.3: `vision` joins the key too -- an image request's prompt chunks carry three extra input tensors and its decode steps a second
    # bound Variable (rope_start), so its captures can never replay a text capture or vice versa; text requests pass neither and keep
    # their pre-T5.3 keys and graphs untouched.
    # T4.92: `presence_penalty` (is not None) joins vision in one (vision_on, penalty_on) pair, appended only when at least one is
    # active -- plain text/no-penalty keys stay the pre-T5.3 4-tuples. The two flags must stay POSITIONAL together, never each its
    # own independently-omitted single-True marker: omitting either on its own collapses "vision only" and "penalty only" onto the
    # identical (...,True) key, so a penalty-only decode step would replay a vision capture (wrong inputs) or vice versa.
    # T4.132: a CONCRETE multi-token call (speculative_generate's VERIFY at width k_eff+1 <= k+1) keys on its own width, stored
    # NEGATED so it can never share a family with a symbolic chunk whose "toks" bound happens to equal it (on CPU the spec prefill
    # chunk is widened to exactly k+1). Every width is its own family with a static token count, so the <=4-token gemv, the prefill
    # attention tile and the small-T GDN decode kernels all cover it (a symbolic width covered none of them: TASKS.md T4.94's audit).
    chunk_size = next((cast(int, v.vmax) for v in tokens.uop.variables() if v.expr == "toks"), -cast(int, tokens.shape[1])) if is_prefill else None
    vision_on, penalty_on = rope_start is not None or vis_e is not None, presence_penalty is not None
    key = (is_prefill, temperature is None, chunk_size, spec) + ((vision_on, penalty_on) if vision_on or penalty_on else ())
    if key not in self.jit and DEBUG >= 1:  # T5.5: each new jit family costs a planned arena on its devices -- name them as they appear
      mem = " ".join(f"{d}={v/2**30:.2f}GiB" for d, v in sorted(GlobalCounters.mem_used_per_device.items()))
      print(f"[jit family] {key}  from {traceback.extract_stack()[-3].name}  rope_delta={getattr(self, '_rope_delta', 0)}  mem_used: {mem}")
    if key not in self.jit:
      # T4.132: VERIFY = a spec call at a concrete width (speculative_generate's); a symbolic-width spec call is extra/flip_rate.py's
      # teacher-forced logits, which perturbs weights between runs -- pruning would freeze the weight-derived kernels, so it stays as was
      verify = spec and isinstance(tokens.shape[1], int)
      self.jit[key] = jit = TinyJit(self._forward_step, prune=(is_prefill and not spec) or verify)
      # T4.128: a prefill family captures on its FIRST call. TinyJit's default first call runs the forward eagerly, and an eager
      # schedule can't plan the buffers custom kernels write into (Tensor.empty outputs count as held), so every block's partials
      # and q8 activations are alive at once -- at 262k that pass was the card's memory peak (within ~50 MB of OOM, 10-28/29). The
      # capture plans them; prune=True runs the input-independent first-call work (KV cache zero-fill) once instead of every replay.
      # T4.132: the VERIFY families too (each concrete width): VERIFY's eager first call holding every attention block's partials at
      # once OOMed at 262k (10-30). Their old exemption (fp16-ulp differences between captured-first and eager graphs, the KV
      # zero-fill being replayed) no longer holds: test_generate_maintains_cache_and_snapshot_round_trips passes with them captured
      # first, and prune runs the zero-fill once.
      if (is_prefill and not spec) or verify: jit.cnt = 1
    return self.jit[key](
      tokens.contiguous(), start_pos, temperature, spec, rope_start, vis_e, vis_m, vis_pos, presence_penalty)

  def _forward_step(self, *args) -> Tensor|tuple[Tensor, ...]:
    # T4.126: nv_quant.py's per-step sharing caches (q8 activations, fused q8, split-K partials) only serve calls inside ONE
    # forward, but their entries keep that step's buffers alive: at a 128-token prefill chunk that pinned ~180 MB of eager-pass
    # activations while TinyJit allocated the capture's arenas, which no longer fit at 262k (10-28 card run). Drop them here,
    # before TinyJit realizes/captures -- the returned graph keeps everything it still needs.
    try: return self.forward(*args)
    finally: clear_step_caches()

  def realize_placement(self):
    """Call once, right after loading weights into a device_map'd model (from_gguf does this for you) --
    e.g. `nn.state.load_state_dict(model, state_dict, realize=False); model.realize_placement()` for manual loaders.
    No-op (single attribute check) when device_map wasn't passed to __init__.

    load_state_dict's assignment is a `.to(param.device)` per tensor (nn/state.py:214, nothing layered after
    it); when that device differs from the load source, `.to` returns a NEW tensor whose OUTERMOST op is an
    unrealized COPY of the entire upstream expression -- e.g. a GGUF param whose dequant was built on the load
    device, or a manually-placed weight built on whatever device it happened to be constructed on. Left lazy,
    that COPY gets captured into the JIT trace and re-executed (recomputing everything upstream of it, AND
    the copy) every token (measured: 21 spurious COPYs/step on a 2-block METAL/CPU split test model -> 1 real
    one, once realized here).

    Only params whose top-level op IS that COPY pay this, so only those get eagerly realized below (T4.21):
    a `gguf.py`-loaded param placed via device_map has its raw quantized blob staged directly on the target
    device (T4.21's loader-side fix) with the dequant built ON TOP of that -- `.to(param.device)` is then a
    no-op (device already matches, no COPY node at all), so it never reaches this method's `moved` list in a
    state that needs realizing; forcing it anyway would eagerly materialize the FULL dequantized (e.g. fp16)
    size instead of leaving it fused into the consuming matmul like a same-device param -- fine for a small
    moved share, a multi-GB blowup for a big-model range split (T4.21's bug). Checking the top-level op instead
    of blanket-realizing every moved param is what makes that distinction cheaply, with no UOp-graph rewrite:
    a REAL cross-device COPY (this method's actual job) is always the outermost op right after `load_state_dict`
    (see nn/state.py:214) -- if it isn't there, there's nothing left for this method to do for that param.

    Also a footgun guard: a hand-built weight assigned without device= (e.g. Tensor.randn(...), which defaults to
    Device.DEFAULT) silently strands itself off the map instead of following its block's placement. Assert (not
    warn) -- this runs once right after load, not in a hot loop, and a silently-stranded weight is a correctness
    bug (wrong-device compute or a surprise COPY every step), not something to let slide."""
    if self._placed_devices is None: return
    params = nn.state.get_parameters(self)
    moved = [p for p in params if p.device != Device.DEFAULT]
    to_realize = [p for p in moved if p.uop.op is Ops.COPY]
    for p in to_realize: p.replace(p.contiguous())
    # Tensor.realize(*to_realize) is `to_realize[0].realize(*to_realize[1:])` -- with to_realize empty (no
    # moved param, or T4.21's loader already placed every moved param's blob with no COPY left to realize)
    # that's a bare Tensor.realize() call with no `self` bound at all: TypeError, not skip-nothing-to-do. Guard it.
    if to_realize: Tensor.realize(*to_realize)
    # params are always single-device here (llm/model.py never shards a weight) -- cast satisfies canonicalize's str|None signature
    stray = sorted(set(Device.canonicalize(cast(str, p.device)) for p in params) - self._placed_devices)
    assert not stray, f"device_map: param(s) landed on {stray}, outside the configured device_map {sorted(self._placed_devices)} " \
      "(a hand-built tensor is probably missing device=)"

  # T4.6: _init_state pre-allocates the whole KV cache (shape ~ max_context) on the first forward, before a
  # single token is generated. A caller who doesn't pass max_context used to get the model's NATIVE context
  # (up to 131072) here -- 2.2-5.3x the model's own weight size in KV cache nobody asked for (T1.9's finding,
  # measured: llama3.2:1b +4.3GB, qwen3:8b +6.07GB). Default to something sane instead; a caller that really
  # wants the model's full native context can still ask for it explicitly, either with max_context=<big N>
  # (min()'d against native below, so overshooting is harmless) or max_context=None (bypasses the cap entirely).
  DEFAULT_MAX_CONTEXT = 8192

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=DEFAULT_MAX_CONTEXT, *,
                device_map:str|dict[int|str,str]|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # gguf_load streams per-tensor (T1.9); no need to force the whole file onto the default device first.
    # T4.21: device_map threaded straight in, so cross-device tensors stage their raw blob on the target
    # device and dequant there -- see the T4.21 comment in gguf.py's _gguf_parse for the mechanism.
    kv, state_dict = gguf_load(gguf, device_map)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)

    ld = kv.get(f'{arch}.leading_dense_block_count', 0)
    # gpt-oss: llama.cpp doesn't write a per-layer pattern key for this; the alternation (even=sliding,
    # odd=full, starting at layer 0) is architecture-fixed, same as the HF config's `layer_types` and
    # examples/mlperf/models/gpt_oss.py's `sliding = i % 2 == 0`. GGUF only carries the window *size*.
    sliding_layers = tuple(i % 2 == 0 for i in range(kv[f'{arch}.block_count'])) if arch == 'gpt-oss' else ()
    rope_scaling = kv.get(f'{arch}.rope.scaling.type')
    yarn_factor = kv[f'{arch}.rope.scaling.factor'] if rope_scaling == 'yarn' else 1.0
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      # gpt-oss routes with softmax computed only over the selected top-k logits (no GGUF key for this; the HF
      # model has no configurable score_function, it's baked into the arch, same as examples/mlperf/models/gpt_oss.py)
      expert_gating_func=ExpertGating(
        kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX_WEIGHT if arch == 'gpt-oss' else ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=ld,
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{ld}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if ld else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{ld}.exp_probs_b.bias" in state_dict,
      attn_out_bias='blk.0.attn_output.bias' in state_dict,
      router_bias=f"blk.{ld}.ffn_gate_inp.bias" in state_dict,
      moe_bias=f"blk.{ld}.ffn_gate_exps.bias" in state_dict,
      attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_layers=sliding_layers,
      # gpt-oss's clamped-swiglu limit/alpha aren't GGUF metadata either (no config knob upstream); fixed
      # constants confirmed against the HF gpt-oss config (swiglu_limit=7.0) and examples/mlperf/models/gpt_oss.py
      clamp_swiglu=arch == 'gpt-oss', swiglu_limit=7.0, swiglu_alpha=1.702,
      yarn_factor=yarn_factor, yarn_orig_ctx=kv.get(f'{arch}.rope.scaling.original_context_length', 0),
      yarn_beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0), yarn_beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0),
      yarn_attn_factor=kv.get(f'{arch}.rope.scaling.yarn_attn_factor'))
    model = Transformer(config, device_map)  # pre-placed params make load_state_dict load each weight to its mapped device
    # T4.70b: split each dense block's ffn_{gate,up,down}.weight into the per-device shards `model` itself
    # already built (Transformer.__init__, from the same device_map) -- mirrors _rename_mtp_keys below:
    # pure state_dict key/tensor surgery, done before load_state_dict ever sees it. Re-parses device_map
    # (parse_tp_spec is cheap/pure) rather than threading Transformer's own already-parsed tp back out --
    # `model` doesn't otherwise expose it, and adding a getter just for this one internal call isn't worth it.
    if device_map is not None and (tp := parse_tp_spec(device_map)) is not None: _split_tp_keys(state_dict, tp, config.num_blocks)
    if MTP.value and kv.get(f'{arch}.nextn_predict_layers', 0) > 0:
      # mirror Transformer.__init__'s own full-attention block config (MTP/nextn is architecturally always a
      # plain attention block -- DeepSeek-MTP semantics -- positioned after every real block: never one of
      # leading_dense_blocks, never sliding (gpt-oss, the only sliding_layers arch, has no MTP block)
      mtp_cfg = replace(config, qk_norm=config.head_dim) if config.ssm else config
      block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
      model.mtp_head = MTPHead(mtp_cfg, block_cls)
      last_dev = model.blk[-1].device  # gguf.py's device-map clamp already stages blk.{num_blocks}.* here too
      for p in nn.state.get_parameters(model.mtp_head): p.to_(last_dev)
      _rename_mtp_keys(state_dict, config.num_blocks)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    else:
      # device_map's cross-device placements MUST realize -- see Transformer.realize_placement's docstring.
      # No-op when device_map is None (nothing moved off Device.DEFAULT to begin with).
      model.realize_placement()
    # T4.142: the draft steps' vocabulary-trimmed head (SPEC_DRAFT_VOCAB) -- before prepare_dense_weights, which probes its format too
    if model.mtp_head is not None and SPEC_DRAFT_VOCAB.value:
      import numpy as np
      model.mtp_head.set_draft_vocab(model, np.load(SPEC_DRAFT_VOCAB.value).tolist())
    from tinygrad.llm.kernels.nv_dense import prepare_dense_weights  # local: mirrors amd.py's import discipline
    prepare_dense_weights(model)  # T4.98h: the fp16 head copies must exist before the first (@function, JIT-captured) call
    for blk in model.blk:
      blk.prepare_nv_norms()  # T4.125b: same reason
      if isinstance(blk, GatedDeltaNetBlock): blk.prepare_nv_decode()  # T4.123f+g: same reason
    if model.mtp_head is not None: model.mtp_head.block.prepare_nv_norms()
    if model.output_norm.weight is not None and nv_quant_supported(model.output_norm.weight.device):
      model.nv_output_norm_weight = model.output_norm.weight.float().contiguous().realize()  # T4.125b: as prepare_nv_norms
    return model, kv

  def warmup(self, vision:bool=False, spec_k:int=3):
    # warm both the greedy and sampled jit pairs, so a request doesn't pay a mid-request capture for whichever it hits first
    for temperature in (0.0, 1.0):
      for _ in range(2): list(zip(range(2), self.generate([0], temperature=temperature)))
    # T4.66e: the above only ever calls generate() -- speculative_generate's own keys (VERIFY's and its spec_jit graphs; T4.132: its
    # prompt prefill replays generate()'s families just warmed above, no spec=True prefill family exists) were otherwise first captured at
    # REQUEST time (the T4.65 gap SPEC_PROFILE_NOTES.md/SPEC_FIXES_NOTES.md both flagged, recommended fix in
    # T4.72). Only when MTP is actually active: `for _ in range(2):`, exactly mirroring the generate() loop
    # just above and for the identical reason (engine/jit.py _TinyJit.__call__): a jit key's FIRST-EVER call
    # (cnt=0) just runs forward() eagerly, no capture at all; the CAPTURE that actually pays off on every
    # later replay only happens on that SAME key's SECOND call (cnt=1) -- so calling speculative_generate
    # ONCE would leave every spec key exactly as uncaptured as never warming it at all, silently moving the
    # real capture cost onto the first live request. Each of the 2 calls pulls 2 tokens (prefill anchor + one
    # full outer iteration: DRAFT touches every mtp_head.block position variable) -- 4 tokens total, cheap.
    if self.mtp_head is not None:
      spec_k = spec_k_clamped(spec_k)  # T4.132: only the widths speculative_generate can reach
      self._spec_track_slots(spec_k)  # sized for spec_k before the dummy runs' narrower VERIFYs allocate it
      # T4.66l: a 2-token dummy prompt (not [0]) so the MTP prefill's own precompiled trace (MTPHead.prefill -- one
      # trace for every prompt-chunk width) is warmed here too, not on the first real request.
      # T4.132: both temperatures, like generate()'s loop above -- the draft-chain families serve both, the accept family is per
      # (n, greedy) -- and every k_eff 1..spec_k (each its own chain/VERIFY/accept family; k_eff < k near max_context): each run's
      # first iteration is one full chain + VERIFY + accept, the second run captures them.
      for temperature in (0.0, 1.0):
        for kk in range(1, spec_k + 1):
          for _ in range(2): list(zip(range(2), self.speculative_generate([0, 0], k=kk, temperature=temperature)))
      # T4.132: VERIFY is one concrete-width family per width 1..spec_k+1 (k_eff+1); width 1 (k_eff == 0, the max_context boundary)
      # isn't reached by the runs above -- drive every width directly, twice. spec_k is the server's SPEC_TOKENS (cli.py), clamped.
      v_sp = UOp.variable("start_pos", 0, self.max_context - 1)
      for w in range(1, spec_k + 2):
        for _ in range(2): self(Tensor([[0] * w], dtype="int32", device=self.blk[0].device), v_sp.bind(1), None, spec=True)
      # T4.132: the partial-accept adopt family (spec_track slot -> live GDN state; the VERIFY drives above made spec_track)
      if self.has_recurrent_block:
        for _ in range(2): self._spec_replay(("adopt",), self._adopt_step, UOp.variable("spec_m", 0, GDN_DECODE_TOKENS - 2).bind(0))
      self._cached_tokens = []  # those calls advanced the live state past what _cached_tokens describes: nothing may resume from it
      # T4.66g: unlike the main model's own blocks above (whose warmup residue in cache_kv/recurrent_state
      # is harmless -- a real request's own prefill legitimately rewrites position 0 onward before ever
      # attending to it), mtp_head.block is never "prefilled": draft() only ever runs from a real request's
      # own start_pos onward (see speculative_generate's DRAFT step), so this dummy run's own K/V at low
      # positions (0..~3, from the nonsense prompt=[0]) is NEVER revisited/overwritten by a real request.
      # Left in place, every draft() call for the rest of this process's life attends back to it (causal
      # attention has no window) -- confirmed (test_spec_decode.py's TestWarmupResetsMTPCache) to measurably
      # change a later draft() call's own logits with everything else (h/tok_ids/start_pos) held identical,
      # isolating this from the model's other state entirely. This is exactly the "draft quality quietly
      # degrades, but rejection is always safe so token-identity tests never catch it" shape: the main
      # model's own state (and therefore every emitted token) is untouched, only the MTP head's own
      # next-token guess is. Reset mtp_head.block's KV cache so it
      # rebuilds fresh (all-zero, via _init_state's own hasattr-gated allocation, same object identity
      # `@function(precompile=True)` doesn't pin -- see MTPHead.draft/FFNBlock.__call__'s own docstrings on
      # why it retraces per call) on the first real request, same as if warmup had never run speculative
      # decoding at all. freqs_cis is NOT reset: it's a pure function of config (rope_dim/max_context/
      # rope_theta/yarn), never session data, so nothing to pollute.
      # T4.66i: zero it IN PLACE instead of delattr+recreate -- same object identity for every captured graph and
      # precompiled CALL that references it, no free/realloc on the device (a recycled allocation is never
      # zero-filled on METAL/NV), and no freqs_cis churn; the intent (no warmup K/V left for real drafts to
      # attend to) is unchanged.
      for attr in ("cache_kv", "cache_k"):
        if hasattr(self.mtp_head.block, attr): (c:=getattr(self.mtp_head.block, attr)).assign(Tensor.zeros_like(c)).realize()
    if vision:
      # T5.3: the image-prompt family too (chunked prefill with the side inputs, then decode with rope_start) -- a 2-chunk dummy prompt
      # carrying one 4-token span (grid (1, 4, 4) merges to 2x2), so the first real image request replays instead of capturing
      dummy = VisionInput([(1, 4, (1, 4, 4))], Tensor.zeros(4, self.token_embd.weight.shape[1]))
      # T5.5: greedy only -- image requests are always decoded greedily (serve.py forces it), so the sampled vision pair is never hit;
      # each extra prefill family costs ~0.78 GB on the 3090 (see VISION_CHUNK), and the standing map has room for exactly one.
      for _ in range(2): list(zip(range(2), self.generate([0] * 38, temperature=0.0, vision=dummy)))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  # T4.67: cross-session state cache. get_start_pos/_cached_tokens above only ever remember ONE sequence -- this
  # model instance's own most recent one (the "splice" path). snapshot_state/restore_state let a caller (serve.py)
  # park SEVERAL sequences' complete state off to the side (keyed by token-id tuple, see snapshot_matches) and
  # swap one back in when a new request's ids exactly extend it, instead of re-prefilling a byte-identical shared
  # prefix (e.g. a system prompt) from scratch just because the live cache currently holds some OTHER session's
  # state. Device-side only (Tensor.clone/.assign, one batched realize each, same idiom as speculative_generate's
  # own GDN CHECKPOINT) -- no host round-trip for cache contents, only for the tiny token-id list. T4.99: unless
  # STATE_CACHE_DEVICE names a different device (e.g. "CPU"), in which case the clones DO round-trip through it
  # (restore_state copies each one back to its live buffer's device -- which never moves -- before the assign).
  def snapshot_state(self) -> dict:
    """Snapshot this model's complete sequence state at its current token boundary (len(self._cached_tokens)):
    every GDN block's conv_state/recurrent_state (whole tensors -- O(1) accumulators, not position-indexed) and
    every attention/MLA block's KV cache LIVE slice only (never the full max_context-sized buffer). Precondition
    (always true the way serve.py calls this -- right after a generate()/speculative_generate() step yields its
    first token, i.e. prefill just completed): at least one forward pass has already run on this model, so every
    block's cache buffers exist. restore_state is the inverse; snapshot_matches tells a caller whether it's even
    valid to call that for a given token list. T4.66l: mtp_head.block's own KV cache (live slice, same rule) is
    included too when MTP is loaded -- since the MTP prefill it holds the prompt's context for the draft head, and a
    restore without it would leave the drafts attending over whatever the last request left there (quality only,
    never correctness -- a wrong draft costs iterations, not tokens; the one slot the restored request still cannot
    fill is `pos` itself, step pos-1, whose hidden state is not recomputed on a cache hit)."""
    pos = len(self._cached_tokens)
    dev = STATE_CACHE_DEVICE or None  # T4.99: None here means "clone stays on its source's own device" (see Tensor.clone)
    blocks: list[dict] = []
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock): blocks.append({"conv_state": b.conv_state.clone(dev), "recurrent_state": b.recurrent_state.clone(dev)})
      elif isinstance(b, MLATransformerBlock): blocks.append({"cache_k": b.cache_k[:, :, :pos, :].clone(dev)})
      elif isinstance(b, TransformerBlock):
        bs: dict = {"cache_kv": b.cache_kv[:, :, :, :pos, :].clone(dev)}
        if hasattr(b, "cache_kv_scale"): bs["cache_kv_scale"] = b.cache_kv_scale[:, :, :, :pos, :].clone(dev)  # T6.1 KV_INT8 / T4.100 KV_INT4
        blocks.append(bs)
      else: raise TypeError(f"snapshot_state: unhandled block type {type(b).__name__}")
    snap: dict = {"tokens": list(self._cached_tokens), "pos": pos, "blocks": blocks, "rope_delta": self._rope_delta}
    # ponytail: TransformerBlock-shaped MTP block only (every nextn head this fork loads); an MLA-shaped one would need its cache_k here.
    if self.mtp_head is not None and isinstance(self.mtp_head.block, TransformerBlock) and hasattr(self.mtp_head.block, "cache_kv"):
      snap["mtp_cache_kv"] = self.mtp_head.block.cache_kv[:, :, :, :pos, :].clone(dev)
      if hasattr(self.mtp_head.block, "cache_kv_scale"):  # T6.1 KV_INT8 / T4.100 KV_INT4
        snap["mtp_cache_kv_scale"] = self.mtp_head.block.cache_kv_scale[:, :, :, :pos, :].clone(dev)
    # T4.112 (09-13): one realize PER BLOCK, not one batched realize -- the planner folded all 32 attention-block clone temporaries
    # (the contiguous NV-side slices before the copy to STATE_CACHE_DEVICE) into a single 405 MB buffer at 21.6k tokens (1.1 GB at
    # 60k), which does not fit beside a 262144-token KV cache on the 3090 and turned every snapshot into an allocator OOM flush
    # (41 s on the wire). Per block the temporary is one slice (~25 MB) and always fits; the extra realizes cost milliseconds.
    for bs in blocks: Tensor.realize(*bs.values())
    if (mtp := [snap[key] for key in ("mtp_cache_kv", "mtp_cache_kv_scale") if key in snap]): Tensor.realize(*mtp)
    return snap

  def restore_state(self, snap:dict) -> None:
    """Inverse of snapshot_state: writes every cloned buffer back into this model's LIVE (preallocated) caches
    via the slice-assign idiom (`buf[...].assign(...)`, batched-realized -- generalizing speculative_generate's
    GDN-checkpoint restore to attention/MLA KV slices too), then restores _cached_tokens so get_start_pos/
    generate/speculative_generate see exactly the state snapshot_state captured -- a continuation is then
    TOKEN-IDENTICAL to an uncached run over the same full token sequence. Pure mechanical apply, no validation:
    the caller must only call this when snap is actually safe to apply (see snapshot_matches) -- restoring a
    snapshot whose tokens are NOT an exact prefix of what comes next would silently produce wrong output, same
    as get_start_pos's own recurrent-reuse rule would if a caller bypassed it. T4.99: each buffer is `.to()`d onto
    its LIVE counterpart's device first (a no-op when snapshot_state left it there already, i.e. STATE_CACHE_DEVICE
    unset) -- the live caches themselves never move, only where snapshot_state parked the clone."""
    assert len(snap["blocks"]) == len(self.blk), "restore_state: snapshot block count doesn't match this model"
    # T4.135 (10-30): one realize PER BLOCK, as snapshot_state does (T4.112) -- one batched realize planned every block's NV-side copy
    # of the snapshot into a single arena the size of the whole snapshot (3.06 GB for a 167k-token conversation), which does not fit
    # beside the 262k KV cache + MTP: every restore of a long session OOM'd (Hermes, 10-30 19:03). Per block it is one slice.
    for b, bs in zip(self.blk, snap["blocks"]):
      assigns: list[Tensor] = []
      if isinstance(b, GatedDeltaNetBlock):
        assigns.append(b.conv_state.assign(bs["conv_state"].to(b.conv_state.device)))
        assigns.append(b.recurrent_state.assign(bs["recurrent_state"].to(b.recurrent_state.device)))
      elif isinstance(b, MLATransformerBlock): assigns.append(b.cache_k[:, :, :bs["cache_k"].shape[2], :].assign(bs["cache_k"].to(b.cache_k.device)))
      elif isinstance(b, TransformerBlock):
        assigns.append(b.cache_kv[:, :, :, :bs["cache_kv"].shape[3], :].assign(bs["cache_kv"].to(b.cache_kv.device)))
        if (bscale := bs.get("cache_kv_scale")) is not None:  # T6.1 KV_INT8 / T4.100 KV_INT4
          assigns.append(b.cache_kv_scale[:, :, :, :bscale.shape[3], :].assign(bscale.to(b.cache_kv_scale.device)))
      else: raise TypeError(f"restore_state: unhandled block type {type(b).__name__}")
      Tensor.realize(*assigns)
    assigns = []
    if (mk := snap.get("mtp_cache_kv")) is not None and self.mtp_head is not None:  # T4.66l -- see snapshot_state
      mtp_block = cast(TransformerBlock, self.mtp_head.block)
      assigns.append(mtp_block.cache_kv[:, :, :, :mk.shape[3], :].assign(mk.to(mtp_block.cache_kv.device)))
      if (mks := snap.get("mtp_cache_kv_scale")) is not None:  # T6.1 KV_INT8 / T4.100 KV_INT4
        assigns.append(mtp_block.cache_kv_scale[:, :, :, :mks.shape[3], :].assign(mks.to(mtp_block.cache_kv_scale.device)))
    if assigns: Tensor.realize(*assigns)
    self._cached_tokens = list(snap["tokens"])
    self._rope_delta = snap.get("rope_delta", 0)  # T5.3 (pre-T5.3 snapshots: text only, 0)

  # T4.111: run generate()'s own chunked prefill over `tokens` and stop, discarding the one token it samples once
  # the prefill completes -- serve.py calls this once per distinct FIXED PREFIX (the system prompt + tool schemas,
  # byte-stable across every session that starts the same way) so store_snapshot can pin it under that prefix's own
  # key. T4.67's find_snapshot already resumes from ANY stored snapshot that's an exact token prefix of a request's
  # ids -- this just gives it one to find before any session-specific request ever completes, so a brand-new session
  # sharing only the prefix skips straight to prefilling its own tail instead of re-prefilling the whole thing cold.
  def prefill_only(self, tokens:list[int], chunk_size:int|None=None) -> None:
    """Leaves this model exactly where generate() would leave it mid-response to `tokens` as the whole prompt:
    self._cached_tokens == list(tokens), so get_start_pos(tokens + more) == len(tokens) and a following
    generate(tokens + more) is TOKEN-IDENTICAL to a cold generate(tokens + more) -- same chunked prefill loop, same
    JIT families, same bound-Variable slicing (this IS generate(), see below). No separate mirror of the loop: this
    runs generate() itself, greedy (temperature is irrelevant -- the one token it samples is thrown away), and stops
    right after its first yield. Up to that point generate() has done nothing but chunked prefill (the `continue`
    branch, every chunk through the last); the last chunk's forward call also produces the first decode logit as an
    ordinary side effect of processing that chunk -- no extra forward call is spent to get it, and discarding the
    token it turns into costs nothing further. At the yield, generate() has already set
    `self._cached_tokens = tokens[:-1]` on ITS OWN (copied) token list, which by construction is exactly the ORIGINAL
    `tokens` we were given (the only thing ever appended past it is the one sampled token now being thrown away).
    No separate recurrent-state reset is needed either: generate()'s own start_pos==0 device-side branch
    (Transformer._attention's `initial = start_pos.eq(0)` mask -- the same one a genuinely cold prefill hits) already
    zeros a GatedDeltaNetBlock's conv_state/recurrent_state whenever `tokens` doesn't already extend this model's
    CURRENT _cached_tokens (get_start_pos returns 0 in exactly that case); if it does happen to extend it, reusing
    that live state is exactly correct, not a bug -- get_start_pos's own prefix-reuse rule, unchanged here.
    T4.132: with an MTP head loaded this runs speculative_generate's own prefill instead (the same loop, plus MTPHead.prefill
    per chunk filling the draft block's cache -- the context every later draft attends over, and part of snapshot_state), so a
    --mtp server's prefix snapshots (serve.py, T4.111) restore a complete state; its anchor yield sets _cached_tokens the same way."""
    it = self.speculative_generate(list(tokens), k=1, chunk_size=chunk_size) if self.mtp_head is not None \
      else self.generate(list(tokens), chunk_size=chunk_size, temperature=0.0)
    next(it, None)
    it.close()

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0, drain_every:int=1, vision:VisionInput|None=None,
               presence_penalty:float=0.0):
    """chunk_size (T4.124a): None (the default, every pre-T4.124a caller) resolves to PREFILL_CHUNK.value (itself 32 by default) --
    byte-identical to the old chunk_size=32 default. Pass an explicit width to override PREFILL_CHUNK for one call.
    vision (T5.3): the prompt's images -- see VisionInput. None (every pre-T5.3 caller) is the byte-identical text path.
    drain_every: batch this many decode steps between host round-trips (T2.5 sync amortization) instead of
    syncing every sampled token. drain_every=1 (default) is byte-identical to the pre-T2.5 behavior -- every
    generate()-caller and test that assumes one .item()/next() per decode step keeps working unchanged. Pass
    drain_every=2..4 from a real serving loop to amortize the sync (streaming still yields one token at a time,
    just in bursts -- see the NOTE below the drain block for EOS handling).
    presence_penalty (T4.92, OpenAI-style, binary presence not frequency): <=0 (the default) is byte-identical to before this
    existed -- no buffer allocated, no extra op anywhere. >0 subtracts `presence_penalty` from the logit of every token id
    already emitted BY THIS generate() CALL (never the prompt, never a previous call -- see self.penalty_mask's reset below
    and forward()'s own comment for the read side) before the greedy/sampled decision, every step."""
    chunk_size = chunk_size or PREFILL_CHUNK.value
    # T4.6: a prompt that already fills (or overflows) max_context has zero room to generate into. Without this,
    # `while virtual_len < self.max_context` below never runs and this silently yields nothing (len==max_context),
    # or the `t = Tensor(...).reshape(1, self.max_context)` a few lines down throws an opaque shape-mismatch
    # (len>max_context) -- neither names the actual problem. serve.py has its own (HTTP-level) version of this
    # check before it ever calls generate(); this is the equivalent guard for every other caller (CLI, library).
    assert len(tokens) < self.max_context, \
      f"prompt has {len(tokens)} tokens but max_context={self.max_context} leaves no room to generate " \
      "-- raise it via --max_context (cli.py) or Transformer.from_gguf(max_context=...)"
    # T4.55: recurrent blocks have a fused scan kernel only on RDNA3 (llm/kernels/amd.py); everywhere else the unrolled T_pad scan
    # in GatedDeltaNetBlock._attention handles a chunk, but generate() historically pinned chunk_size=1 (one token per step, i.e.
    # prefill at decode speed -- 46 tok/s on the pooled qwen3.6-35B). GDN_CHUNK caps the chunk width for those devices instead.
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device):
      chunk_size = min(chunk_size, gdn_chunk_for(self.token_embd.weight.device))
    if vision is not None: chunk_size = min(chunk_size, VISION_CHUNK.value)  # T5.5: see VISION_CHUNK
    drain_every = max(1, drain_every)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    # create helper tensors on the block devices that consume them, so device_map'd models don't replay a cross-device copy every step
    # T4.105: always a Tensor -- temperature 0 is decided INSIDE the captured graph (forward's where-select), so greedy and sampled
    # requests share one jit family (the greedy family's own planned arena was ~1.6 GB at a 131k window on the 3090)
    temp = Tensor([temperature], device=self.blk[-1].device)
    # T4.92: penalty_mask must be reset IN PLACE, never reassigned -- a jit family already captured with presence_penalty active
    # baked in a read of THIS buffer object; reassigning self.penalty_mask here would silently detach every future replay of
    # that graph from the reset (same reasoning as T4.66i's in-place MTP cache-reset, see its comment). hasattr guards the
    # very first call ever (nothing to reset yet); `.clone()` on the fresh zeros matches conv_state/recurrent_state's own
    # init idiom (_init_state above) for the same reason -- a bare Tensor.zeros() is a broadcast view, not an assignable buffer.
    penalty = Tensor([presence_penalty], device=self.blk[-1].device) if presence_penalty > 0 else None
    if penalty is not None:
      if not hasattr(self, "penalty_mask"):
        self.penalty_mask = Tensor.zeros(1, self.token_embd.weight.shape[0], dtype=dtypes.float32, device=self.blk[-1].device).clone()
      else: self.penalty_mask.assign(Tensor.zeros_like(self.penalty_mask)).realize()
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32", device=self.blk[0].device).reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    # T5.3: image prompts (and text prompts extending a cached image prefix) get a host-side plan: per-position (t,h,w) rope ids + image
    # mask + encoder rows for the chunks still to prefill, and the rope deficit the decode steps run at. Text-only stays plan-less.
    if vision is not None:
      for off, n, _ in vision.spans:  # never resume inside an image span: re-prefill it whole (its rows ride with its chunks)
        if off < start_pos < off + n: start_pos = off
    if start_pos == 0: self._rope_delta = 0
    elif self._rope_delta and start_pos != len(self._cached_tokens):
      start_pos = 0  # ponytail: partial-prefix reuse (attention-only models) behind images -- the stored deficit is the full prefix's; re-prefill
    plan = _VisionPlan(vision, prompt_len, start_pos, self._rope_delta, chunk_size, int(self.token_embd.weight.shape[1]), t.device) \
      if vision is not None or self._rope_delta else None
    if plan is not None:
      v_rope_start = UOp.variable("rope_start", 0, self.max_context-1)
      self._rope_delta = plan.delta
    # T2.5: a sampled token already chains device-side -- `out` feeds the next step's input directly (below),
    # no host round-trip needed for the compute itself. .item() was only ever needed for host bookkeeping/
    # streaming (tokens list, _cached_tokens, yield). So launch up to `drain_every` decode steps back-to-back
    # (each still its own realize(), all async-dispatched -- see engine/realize.py's run_linear(wait=False)) and
    # read them all back with ONE host copy instead of one .item() sync per token. `pending` holds the
    # not-yet-host-read device tensors; `virtual_len` tracks what len(tokens) *will* be once pending is drained,
    # so the chunk/prefill arithmetic below is untouched by the deferral -- only the host-visible
    # append/_cached_tokens/yield timing moves.
    pending: list[Tensor] = []
    virtual_len = len(tokens)
    while virtual_len < self.max_context:
      n_toks = min(chunk_size, virtual_len - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      inp = t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out
      if plan is None: res = self(inp, sp, temp, presence_penalty=penalty)
      elif start_pos < prompt_len:
        e, m, p3 = plan.chunk(start_pos, n_toks)
        res = self(inp, sp, temp, vis_e=e, vis_m=m, vis_pos=p3, presence_penalty=penalty)
      else: res = self(inp, sp, temp, rope_start=v_rope_start.bind(start_pos - plan.delta), presence_penalty=penalty)
      out = (res[0] if isinstance(res, tuple) else res).realize()  # T4.132: an MTP model's prefill chunk also returns its hidden
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < virtual_len: continue
      # move the sampled token once, back to t's device, so the next step's input matches the JIT's prefill-captured device.
      if out.device != t.device: out = out.to(t.device).realize()
      # when draining is deferred (drain_every>1), `out` outlives the JIT replay that produced it -- the JIT reuses its
      # output buffer across replays (engine/jit.py's memory_plan_rewrite), so a not-yet-drained `out` would silently
      # alias data the *next* chained step's replay overwrites. Snapshot it into a private buffer to break that alias.
      # drain_every==1 (the default) drains immediately below, before any further replay can touch the buffer -- no
      # snapshot needed there, so this stays a pre-T2.5-identical zero-extra-op path.
      elif drain_every > 1: out = out.clone().realize()
      pending.append(out)
      # T4.92: mark this token in the presence mask -- reached only for an actually-generated token (the `continue` above
      # skips every still-consuming-the-prompt step, so prompt tokens never enter the mask). In-place assign, device-side,
      # no host round-trip (out is never .item()'d here) -- see forward()'s read side and the reset comment above.
      if penalty is not None:
        onehot = out.to(self.penalty_mask.device).reshape(1).one_hot(int(self.penalty_mask.shape[1])).cast(self.penalty_mask.dtype)
        self.penalty_mask.assign(self.penalty_mask.maximum(onehot)).realize()
      virtual_len += 1
      # drain once the batch fills, or generation is about to stop (don't strand tokens on-device)
      if len(pending) >= drain_every or virtual_len >= self.max_context:
        # NOTE on EOS: the caller checks for a stop token per yielded value (see cli.py/serve.py's is_end loop)
        # and stops pulling from this generator. Because we yield one token at a time from the drained batch
        # below, a stop token anywhere in the batch makes the caller break before the extras after it are ever
        # appended/yielded -- self._cached_tokens and the returned `tokens` list end exactly at what was
        # actually consumed. The only cost is up to drain_every-1 already-computed, now-wasted device steps.
        # T4.132: a lone jit output (drain_every 1, the server's) is read straight from its buffer: `.tolist()` scheduled a contiguous
        # copy kernel per token -- an eager realize, whose _apply_map_to_tensors walks every live Tensor's graph
        drained = Tensor.cat(*pending, dim=1) if len(pending) > 1 else pending[0]
        # (a memoryview of the int32 ids, not Buffer.numpy(): the minimal CI lanes have no numpy -- see the imports)
        vals = cast(Buffer, drained.uop.buffer).as_memoryview().cast("i").tolist() if drained.uop.has_buffer_identity() and \
          isinstance(drained.device, str) and drained.dtype == dtypes.int32 else cast(list[list[int]], drained.tolist())[0]
        for v in vals:
          tokens.append(int(v))
          self._cached_tokens = tokens[:-1]
          yield tokens[-1]
        pending = []

  # T4.132: the speculative loop's graphs as TinyJits (self.spec_jit), so a steady-state iteration schedules nothing eagerly: the draft
  # chain, VERIFY, the accept test (+ the next iteration's inputs + the MTP cache rewrite), and on a partial accept the adopt -- 3-4
  # replays and the one readback. Step 6's form (one replay per draft step, the chunk cat, the anchor upload, h_last's copy and the
  # rewrite's input prep as eager realizes) cost ~20 ms of host time per eager realize on the card's host (10-30 phase table).
  def _draft_step(self, h:Tensor, tok:Tensor, pos:UOp, temp:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """One MTP draft step: (h, the token at pos) -> the drafted id (B,1), its logits (B,1,vocab) and the head block's output x (B,1,dim,
    T4.136: MTPHead.draft_step's, the next step's hidden under SPEC_CHAIN_H; an instance-patched draft -- the eager oracles, logits
    only -- hands h back instead). Greedy or sampled is decided inside the graph from `temp` (forward()'s T4.105 where-select), so one
    family serves both; the sampled draw is sample_logits' Gumbel-max over softmax(logits/temp) with MIN_P -- the ancestral q
    spec_accept_tensor's proof assumes.
    T4.142: under a draft vocabulary (MTPHead.set_draft_vocab) the step's logits are the slice's (B,1,S): the pick (argmax / the same
    Gumbel-max + min_p, relative to the subset's max) runs over the S columns and maps through draft_ids, and the q handed to the accept
    is full_vocab's (B,1,vocab), -inf off the set -- the softmax over the subset, exactly the distribution the draft was drawn from. Two
    gathers on the head's device inside the same graph: no sync, no copy. An instance-patched draft returns full-vocab logits: unmapped."""
    assert self.mtp_head is not None
    mh, ids = self.mtp_head, None
    if "draft" in vars(mh): logits, x = mh.draft(self, h, tok, pos), h
    else: (logits, x), ids = mh.draft_step(self, h, tok, pos), mh.draft_ids
    greedy = logits.argmax(-1)
    d = (temp > 0).reshape(1, 1).where(sample_logits(logits[:, -1, :], temp, MIN_P), greedy)
    return (d, logits, x) if ids is None else (ids[d], mh.full_vocab(logits), x)

  def _draft_chain(self, h:Tensor, tok:Tensor, temp:Tensor, *pos:UOp) -> tuple[Tensor, ...]:
    """T4.132: the draft chain as one graph -- len(pos) MTP steps (_draft_step), each fed the previous step's id at its own bound position
    (draft_pos_i: distinct names, so every step binds once per replay). Returns VERIFY's chunk [tok, d0..] (1,n) on the head's device,
    the same chunk on the first block's device (VERIFY's input; the same tensor on one device) and each step's logits (the sampled
    accept's q). One family per k_eff = len(pos).
    T4.136: step 0 takes `h` (the main model's hidden at start_pos-1); under SPEC_CHAIN_H=1 every later step takes the previous step's
    block output x -- a lazy edge inside this one graph, on the head's device (the block's own), so no realize and no copy is added.
    SPEC_CHAIN_H is read here, i.e. once, when the ("chain", k_eff) family captures; its replays keep that value."""
    chain_h = bool(SPEC_CHAIN_H.value)
    ids, qs = [tok], []
    for p in pos:
      d, lg, x = self._draft_step(h, ids[-1], p, temp)
      if chain_h: h = x
      ids.append(d)
      qs.append(lg)
    chunk = ids[0].cat(*ids[1:], dim=1).contiguous()
    return (chunk, chunk.to(self.blk[0].device).contiguous(), *qs)

  def _accept_step(self, greedy:bool, verify_logits:Tensor, verify_h:Tensor, chunk:Tensor, temp:Tensor, mtp_pos:UOp,
                   *q_logits:Tensor) -> tuple[Tensor, ...]:
    """The accept test (speculative_generate's ACCEPT) as one graph. Returns the (2n+1,) int readback [m, chunk ids, final ids] -- final[i]
    is the id emitted if the accept length is i (greedy: VERIFY's argmax; sampled: spec_accept_tensor's residual resample / bonus) --
    then, gathered at m on device (T4.132), the next iteration's draft-chain inputs: the anchor final[m] (1,1) and h_last =
    verify_h[:, m] (1,1,D); the graph also realizes the MTP cache rewrite below. One family per (n, greedy): the sampled one draws its
    uniforms inside the graph.
    The MTP rewrite (T4.66l) refills mtp_head.block's cache slots start_pos+1..start_pos+k_eff (mtp_pos = start_pos+1) from the TRUE
    per-position hidden: the draft chain wrote slot start_pos+i from (its hidden, chunk[i]) -- the right token, but that hidden is position
    start_pos-1's (h_last) at step 0 and, T4.136 (SPEC_CHAIN_H=1), the head's own previous block output at every step i >= 1 (h_last
    again under 0) -- while slot t+1 belongs to (h_t, tok_{t+1}) = (verify_h[:, j], chunk[j+1]) for t = start_pos+j.
    T4.132: all k_eff slots unconditionally (step 6 rewrote the m accepted ones, which needed m on the host): for j >= m the pair is a
    rejected draft's, but those slots are >= the next iteration's start_pos' = start_pos+m+1, and MTPHead.draft at position p writes
    slot p (the block's store) before attending to slots <= p, so the next chain overwrites each before anything reads it (the
    attention-KV argument of SPEC_NOTES.md section 2). At the k_eff-wide concrete width, on the decode kernels -- not step 6's replay of
    the prompt's 128-wide MTP prefill family (19 ms on the card)."""
    n = cast(int, chunk.shape[1])
    k_eff, drafts = n - 1, chunk[:, 1:]
    if greedy:
      final = verify_logits.argmax(-1)  # (1, n) ids
      m_t = (final[:, :k_eff] == drafts).cast(dtypes.int32).cumprod(-1).sum(-1) if k_eff else Tensor.zeros(1, dtype=dtypes.int32, device=chunk.device)
    else:
      vocab = cast(int, verify_logits.shape[-1])
      q = Tensor.stack(*[ql[:, -1, :] for ql in q_logits], dim=1) if q_logits else verify_logits[:, :0]  # (1, k_eff, vocab)
      m_t, final = spec_accept_tensor(drafts, q, verify_logits, temp, Tensor.rand(1, k_eff, device=cast(str, chunk.device)),
                                      Tensor.rand(1, n, vocab, device=cast(str, chunk.device)), min_p=MIN_P)
    # .contiguous(): at n == 2, m is a bool cast to int, and the gather fused onto it came out as an INDEX by the bool (codegen
    # verification error on CPU) -- materialized, the gather reads an int
    m_t = m_t.cast(dtypes.int32).contiguous()
    anchor = final.cast(dtypes.int32).gather(1, m_t.reshape(1, 1)).contiguous()  # (1,1)
    D = cast(int, verify_h.shape[-1])
    h_next = verify_h.gather(1, m_t.to(verify_h.device).reshape(1, 1, 1).expand(1, 1, D)).contiguous()  # (1,1,D)
    outs: tuple[Tensor, ...] = (m_t.reshape(1).cat(chunk[0], final[0]), anchor, h_next)
    if k_eff:
      assert self.mtp_head is not None
      # realized here, not returned -- see MTPHead._prefill_body: a returned block output keeps its whole lazy graph alive
      cast(Tensor, self.mtp_head.block(self.mtp_head._block_in(self, verify_h[:, :k_eff], drafts), mtp_pos)).realize()
    return outs

  def _spec_track_slots(self, k:int) -> None:
    """T4.132: every GDN block's spec_track holds k slots -- the widest VERIFY (k+1 tokens) has k partial accepts -- not the most the
    kernels allow (k=2: 2 x 151 MB at 27B geometry instead of 3). Growing it reallocates the buffers on the next VERIFY, so the graphs
    that captured the old ones (every concrete-width VERIFY family and the adopt family) are dropped and recapture. Called outside any
    capture: speculative_generate and warmup, before their VERIFYs."""
    if k <= self.spec_slots: return
    if self.spec_slots:
      for b in self.blk:
        if isinstance(b, GatedDeltaNetBlock): b.spec_track = None
      for key in [key for key in self.jit if key[3] and isinstance(key[2], int) and key[2] < 0]: del self.jit[key]
      self.spec_jit.pop(("adopt",), None)
    self.spec_slots = k

  def _adopt_step(self, m:UOp) -> None:
    """T4.132: a partial accept's GDN state -- every GDN block's live conv/recurrent state <- slot m of its spec_track (the state VERIFY
    left after token m). One graph for every m (a bound Variable)."""
    Tensor.realize(*(x for b in self.blk if isinstance(b, GatedDeltaNetBlock) and b.spec_track is not None
                     for x in (b.conv_state.assign(b.spec_track[0][m]), b.recurrent_state.assign(b.spec_track[1][m]))))

  def _spec_replay(self, key:tuple, fxn:Callable, *args, prune:bool=False):
    """T4.132: speculative_generate's graphs, one TinyJit per key in self.spec_jit, each captured on its FIRST call (the prefill families'
    T4.128 rule: an eager first call schedules every intermediate unplanned at once -- the card's memory peak). prune=True runs the first
    call's one-time work once (MTPHead.prefill: the MTP block's KV zero-fill on the first prompt). The chain and the accept test do NOT
    prune: pruning kept Tensor.rand's counter advance out of the replay and every sampled draw repeated (T4.139: engine/jit.py's
    prune_linear keeps it now -- the plain prefill family's first-token draw had that bug); their one-time work is done beforehand
    instead (MTPHead.ready, the RNG state in speculative_generate). adopt has no Tensor inputs."""
    if key not in self.spec_jit:
      self.spec_jit[key] = jit = TinyJit(fxn, prune=prune)
      jit.cnt = 1
    return self.spec_jit[key](*args)

  def speculative_generate(self, tokens:list[int], k:int=3, chunk_size:int|None=None, temperature:float=0.0):
    """T4.64/T4.65: MTP speculative decoding -- drafts up to `k` tokens per iteration via self.mtp_head and
    verifies all of them in one extra main-model forward instead of paying one forward per token. See
    SPEC_NOTES.md for the full position ledger, the causal-mask argument for why neither the main model's
    nor the MTP block's attention KV cache ever needs a snapshot/restore (only the main model's GatedDeltaNet
    state does), and §6 for the sampled-acceptance math this docstring summarizes.

    temperature<=0 (the default): TOKEN-IDENTICAL to generate(temperature=0.0) -- every verify id is a plain
    argmax over forward(spec=True)'s per-position logits, exactly reproducing T4.64's behavior (which used to
    compute that same argmax INSIDE forward() -- see forward()'s spec branch). Draft quality never affects
    correctness here, only how many iterations it takes (see test_spec_decode.py's forced-mismatch/
    forced-perfect tests).

    temperature>0 (T4.65): distribution-preserving SAMPLED speculative decoding (Leviathan et al., "Fast
    Inference from Transformers via Speculative Decoding", 2023 -- see spec_accept_tensor). Draft tokens are
    drawn by ancestral sampling from the draft model's own softmax(logits/temperature) ("q"), then accepted or
    resampled against the main model's softmax(logits/temperature) ("p") so the marginal distribution of
    every emitted token equals p exactly (test_spec_decode.py verifies this empirically -- it's the theorem
    the whole path relies on). The output is distribution-equal to generate(temperature=t) but NOT
    sequence-equal to any particular generate() run -- it's a different (also p-distributed) sample, not a
    reproduction. T4.132: every draw is on device (Tensor.rand -- seed with Tensor.manual_seed), so the
    sampled path has the greedy path's one host sync per iteration.

    Mirrors generate()'s own bookkeeping exactly (the `tokens` list, self._cached_tokens, one token
    yielded at a time) -- generate() itself is untouched by this; see Transformer.__call__'s optional
    `spec=` return path.

    Partial accepts (T4.132): VERIFY's GDN blocks store their state after every token (GatedDeltaNetBlock.spec_track -- on the
    card from inside the gdn_decode kernels) and a partial accept of m drafts adopts slot m as the live state (_adopt_step, one
    replay of ~150 MB of copies at 27B geometry). T4.66b read the same per-position states out of a captured `.stack()` and
    `.assign()`ed them eagerly per block (~19.5 s per partial accept on METAL+NV); T4.66d replaced that with CHECKPOINT + a REDO
    forward of the accepted tokens (43-79 ms on the card, 10-30); the adopt replay replaces both. A full accept needs nothing
    (VERIFY's in-place state write is already right); h_last is verify_h[:, m] either way. A warm iteration is the draft-chain
    replay, VERIFY, the accept replay (+ the next anchor/h_last + the MTP cache rewrite) and the one readback, plus the adopt replay
    on a partial accept -- nothing scheduled eagerly (test_spec_decode.py's TestSpecLaunchCensus). SPEC_STATS (a ContextVar) turns
    on per-request accept-length instrumentation.
    """
    assert self.mtp_head is not None, "speculative_generate needs an MTP-enabled model (Transformer.from_gguf under MTP=1)"
    assert k >= 1, "k=0 has nothing to draft -- use generate() directly"
    k = spec_k_clamped(k)
    self._spec_track_slots(k)
    assert len(tokens) < self.max_context, \
      f"prompt has {len(tokens)} tokens but max_context={self.max_context} leaves no room to generate " \
      "-- raise it via Transformer.from_gguf(max_context=...)"
    # temperature<=0 is greedy, same convention generate() itself uses (see there)
    greedy = temperature <= 0
    # T4.132: chunk_size None resolves to PREFILL_CHUNK like generate() (T4.124a) -- the server's prefill chunk, so a --mtp
    # request's prompt prefills in the same chunks (and the same warmed spec prefill family) as a plain one; explicit widths
    # (mtp_bench.py, tests) are unchanged.
    chunk_size = chunk_size or PREFILL_CHUNK.value
    # T4.55: same recurrent-chunk cap generate() applies (see there). T4.132: exactly generate()'s width (no more widening to k+1, which
    # only the old MTP rewrite through the prefill family needed), so the prompt prefill replays generate()'s own family.
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device):
      chunk_size = min(chunk_size, gdn_chunk_for(self.token_embd.weight.device))
    dev = self.blk[0].device
    # T4.66f: where the DRAFT chain's own tokens live -- MTPHead.draft's return is always forced onto
    # owner.output.weight.device (see its own last line), so draft_tensors below is ALWAYS this device,
    # never `dev`. tok_last must match it whenever it's rebuilt from host data (the ACCEPT step below,
    # both branches) -- full accept already gets this right for free (verify_ids_tensor lives here too,
    # same self.output call) but the two explicit `Tensor([[accepted[-1]]], ...)` rebuilds used to hardcode
    # `dev` instead, splitting `tok_last`/`draft_tensors` across devices with no hop between them. Harmless
    # on any single-device config (dev==out_dev there) and on the sampled path alone (chunk_ids there is
    # built via host .item(), never .cat()) -- but greedy's `tok_last.cat(*draft_tensors, dim=1)` a few
    # lines below fuses them into one kernel with no copy inserted, which crashes on a real multi-device
    # map (assert_all_same_devices) the FIRST time a partial accept (m<k_eff) is followed by another
    # iteration -- see test_spec_decode.py's TestSpecDeviceMap for the reproduction.
    out_dev = self.output.weight.device
    temp_t = Tensor([temperature], device=out_dev)  # T4.132: the sampled path's draws (sample_logits) run on device, from this

    v_start_pos = UOp.variable("start_pos", 0, self.max_context - 1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # T4.132: VERIFY runs at CONCRETE widths (k_eff+1, each width 1..k+1 its own jit family -- Transformer.__call__
    # keys a concrete multi-token call on its width) instead of T4.66e's one symbolic family bound at next_power2(k+1): a symbolic
    # width kept every hand-written NV kernel off VERIFY (the <=4-token gemv, the prefill attention tile, the GDN decode kernels
    # all need a static token count -- TASKS.md T4.94's audit), which is why the loop lost 16-27x to plain decode. The T4.66e/j
    # "toks" bind-mismatch and stale-capture-width hazards (two calls of one family at different widths) are gone with it: every
    # family only ever sees its own width, so a replay's output shape is always this call's real shape.
    # T4.66: a drafted position never repeats across a speculative_generate() run, so passing MTPHead.draft's
    # start_pos as a plain python int (pre-T4.66) baked a fresh literal into mtp_head.block's
    # @function(precompile=True) trace every DRAFT call -- confirmed (test_spec_decode.py's schedule-cache
    # regression test) to grow schedule_cache/to_program_cache without bound, one entry-set per drafted
    # position ever seen. Binding a Variable instead replays ONE compiled schedule at every position, the
    # same trick v_start_pos already gives the main model's VERIFY calls below.
    # A separate Variable (set) from v_start_pos, not a reuse of it: TransformerBlock never re-wraps start_pos
    # in its own Variable (unlike GatedDeltaNetBlock, which self-converts -- see its _attention), so whatever
    # name/bounds the caller binds is exactly what reaches mtp_head.block's compiled kernel. Bounds match
    # v_start_pos's: mtp_head.block's cache_kv/freqs_cis are sized off the SAME config.max_context as the main
    # model (from_gguf's MTP branch builds mtp_cfg from the owning Transformer's own config), and k_eff already
    # caps drafted positions at start_pos+k_eff <= max_context-2 (see k_eff's definition below) -- safely
    # inside [0, max_context-1] with room to spare.
    #
    # T4.66b: k DISTINCT Variables (draft_pos_0..k-1), not one name rebound k_eff times per iteration -- this
    # is what lets the per-step .realize() T4.66 needed (see the DRAFT loop below) go away. With one shared
    # name, k_eff different bound values could never coexist in ONE merged schedule ("bind mismatch on
    # draft_pos, i != i+1" -- reproduced while building T4.66, see SPEC_NOTES.md §4), so every drafted token
    # had to be individually realized/dispatched before the next .bind() of that same name was even created.
    # k distinct names have no such conflict (each is bound at most once per iteration), so all k_eff draft
    # calls now merge into ONE schedule (T4.132: one draft-chain replay) -- same as
    # VERIFY already did. Costs k (not 1) schedule_cache/to_program_cache entries instead of 1, still
    # bounded/warmed-once per process (k defaults to 3 -- SPEC_TOKENS): the same "one entry-set per NAME,
    # reused at every position" property T4.66 established, just k of them. Index i means "the i-th draft step
    # of THIS iteration" and gets reused with a different concrete dpos across MANY outer iterations, exactly
    # how the old single name was already reused across iterations --
    # test_draft_reuses_schedule_across_positions' own "flat after warmup" assertion still holds, just with a
    # bigger (k-entry, not 1-entry) flat plateau.
    v_draft_pos = [UOp.variable(f"draft_pos_{i}", 0, self.max_context - 1) for i in range(k)]
    # T4.66l: the MTP block's own prefill width and slot Variables (see MTPHead.prefill): one precompiled trace serves every
    # prompt-chunk width 1..chunk_size. T4.132: mtp_pos is also the accept replay's rewrite slot (_accept_step, a concrete width).
    v_mtp_toks, v_mtp_pos = UOp.variable("mtp_toks", 1, chunk_size), UOp.variable("mtp_pos", 0, self.max_context - 1)
    v_adopt = UOp.variable("spec_m", 0, GDN_DECODE_TOKENS - 2)  # T4.132: the accepted-draft count a partial accept adopts (_adopt_step)

    # --- prefill: generate()'s own chunking loop (same v_start_pos/v_toks vars, same zero-padded token buffer, same get_start_pos
    # cache reuse) -- generate() itself is never called; this is a self-contained copy of just its chunking loop. T4.132: every chunk
    # replays generate()'s OWN prefill family (same key: temperature Tensor on the last block's device, no penalty, the same chunk
    # width on the card), which with an MTP head also returns the chunk's per-position hidden (forward's tail): each chunk fills the
    # MTP block's cache from it, the last chunk's token is the anchor and its last hidden row h_last. Before, every chunk ran a
    # spec=True family (logits for every position): two more 128-wide families, each run eagerly on its first call -- the T4.128
    # memory peak, OOM at 262k on the card (10-30).
    prompt_len = len(tokens)
    t = Tensor(tokens + [0] * (self.max_context - prompt_len), dtype="int32", device=dev).reshape(1, self.max_context)
    temp_pf = Tensor([temperature], device=self.blk[-1].device)  # generate()'s temperature input (its device: part of the family's key)
    start_pos = self.get_start_pos(tokens)
    # T5.3: this path has no rope_start/vision plumbing -- a cached prefix that contains images would decode at the wrong rope positions
    if start_pos > 0 and self._rope_delta: start_pos = 0
    if start_pos == 0: self._rope_delta = 0
    tok_last: Tensor|None = None
    h_last: Tensor|None = None
    while start_pos < prompt_len:
      n_toks = min(chunk_size, prompt_len - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      tok_pf, hidden = cast(tuple[Tensor, Tensor], self(t[:, sp:sp + nt], sp, temp_pf))
      if start_pos + n_toks >= prompt_len:
        # T4.66h: index the last REAL position with the host-known `n_toks` (positive slice), never `-1:` -- a replayed family's
        # frozen output reports its capture call's width (TestWarmupAnchorShift). fresh_copy: a jit input of its own.
        tok_last, h_last = tok_pf, fresh_copy(hidden[:, n_toks - 1:n_toks])
      # T4.66l: MTP prefill -- fill mtp_head.block's cache for this chunk's MTP steps t = start_pos.. from (h_t, tok_{t+1}) into slots
      # t+1 (MTPHead.prefill). The last prompt position's step needs tok_{prompt_len} -- the anchor -- so that one is the first draft
      # step's, not ours: hence the prompt_len-1 clip. Host-known widths only (`[:, :n_mtp]`, T4.66h/j); dispatched before the next
      # chunk's replay of the same family overwrites `hidden` (same device queue).
      n_mtp = min(n_toks, prompt_len - 1 - start_pos)
      if n_mtp > 0:
        self.mtp_head.prefill(self, hidden[:, :n_mtp], tokens[start_pos + 1:start_pos + 1 + n_mtp], start_pos + 1, v_mtp_toks, v_mtp_pos)
      start_pos += n_toks
    # get_start_pos always returns < prompt_len, so the loop above always runs >=1 time; the assert is only for mypy's narrowing.
    assert tok_last is not None and h_last is not None
    # the first post-prompt token: the prefill family's own greedy / sampled (temperature, MIN_P) draw at the last position -- exactly
    # the token generate() itself would yield first. It is about to become chunk[0]/the draft anchor, but it is ALSO real output and is
    # emitted now. A buffer of the prefill family: the first chain reads it before any later prefill replay could overwrite it.
    vocab = self.output.out_features
    anchor = int(tok_last.item())
    spec_check_ids([anchor], vocab, f"anchor at position {prompt_len}")
    tokens.append(anchor)
    self._cached_tokens = tokens[:-1]
    # T4.132: the chain / accept graphs capture on their first call without pruning -- their one-time work happens here, eagerly, once
    self.mtp_head.ready(self, k)
    if (rng_dev := cast(str, out_dev)) not in Tensor._device_seeds:
      Tensor.realize(*Tensor._next_counter(rng_dev, 0), Tensor._device_rng_counters[rng_dev])
    yield tokens[-1]

    stats_on = bool(SPEC_STATS.value)
    accept_hist: dict[int, int] = {}
    stats_iters = stats_drafted = stats_emitted = 0
    trace_on = bool(SPEC_TRACE.value)  # T4.66c: see SPEC_TRACE's own ContextVar docstring for the phase list/sync semantics
    trace_iter = 0
    debug_on = bool(SPEC_DEBUG.value)  # T4.132: see SPEC_DEBUG
    try:
      while len(tokens) < self.max_context:
        if trace_on: iter_t0 = time.perf_counter()
        k_eff = min(k, self.max_context - 1 - len(tokens))

        # (a) DRAFT: chain the MTP step k_eff times. Step 0 takes h_last (the main model's hidden at start_pos-1, the one
        # anchor into its context); T4.136: under SPEC_CHAIN_H=1 (default) every later step takes the previous step's own
        # block output (MTPHead.draft_step's x, before shared_head_norm -- DeepSeek-style recursive MTP), under 0 h_last is
        # reused UNCHANGED (T4.132's chain). tok_ids and start_pos advance per call, and `block`'s own KV cache (see
        # MTPHead.draft's docstring) carries the chain's positional continuity from call to call. This never touches
        # the MAIN model's state -- mtp_head.block is always attention-only, never GatedDeltaNet (see
        # from_gguf's MTP branch) -- so drafting alone never touches the main model's GDN state; only verify does.
        # Draft quality has zero effect on correctness (see the forced-mismatch test in test_spec_decode.py): the hidden
        # each step takes moves only the acceptance rate.
        #
        # T4.132: the drafted ids stay on device -- the greedy argmax, or (temperature>0) a Gumbel-max sample from
        # softmax(logits/temperature) (sample_logits: the exact ancestral draw spec_accept_tensor's proof assumes; the
        # draft logits are kept for its accept test) -- and feed VERIFY's input: no host sync in this phase. The whole chain
        # is ONE replay (_draft_chain, family ("chain", k_eff)) whose inputs are buffers already on device: h_last and the
        # anchor, the previous accept replay's outputs (the prefill's on the first iteration), and its outputs are VERIFY's
        # input chunk and the draft logits. A draft patched onto the MTPHead INSTANCE (test_spec_decode.py's forced-accept
        # oracles read the position in python) cannot be replayed: the chain runs eagerly then, reusing h_last (they return logits only).
        n = k_eff + 1
        if trace_on: draft_t0 = time.perf_counter()
        if k_eff:
          pos = [v_draft_pos[i].bind(start_pos + i) for i in range(k_eff)]
          if "draft" in vars(self.mtp_head): Tensor.realize(*(drafted := self._draft_chain(h_last, tok_last, temp_t, *pos)))
          else: drafted = cast(tuple[Tensor, ...], self._spec_replay(("chain", k_eff), self._draft_chain, h_last, tok_last, temp_t, *pos))
          chunk, buf, q_logits = drafted[0], drafted[1], drafted[2:] if not greedy else ()  # greedy reads no q
        else: chunk, buf, q_logits = tok_last, tok_last.to(dev).realize(), ()  # the max_context boundary: one plain step (once per request)
        if trace_on: draft_ms = (time.perf_counter() - draft_t0) * 1000

        # (b) VERIFY: one main-model forward of the whole chunk [tok_last, d0..d_{k_eff-1}] at the current
        # start_pos, returning the per-position logits (what the real model actually predicts from every one of
        # those positions) and the per-position hidden state. GatedDeltaNet conv/recurrent state is a single
        # read-modify-written accumulator, not a position-indexed cache, so once VERIFY mixes a wrong draft token into
        # it there's no slice to discard (unlike attention KV -- the mask argument in SPEC_NOTES.md). T4.132: VERIFY's
        # GDN blocks also store their state after each token (spec_track; on the card from inside the gdn_decode
        # kernels) and a partial accept adopts the right one below. VERIFY is the width-n family (concrete, its own jit
        # key -- see Transformer.__call__).
        if trace_on: verify_t0 = time.perf_counter()
        verify_logits, verify_h = cast(tuple[Tensor, Tensor], self(buf, v_start_pos.bind(start_pos), None, spec=True))  # (1,n,vocab), (1,n,D)
        if trace_on: verify_dispatch_ms = (time.perf_counter() - verify_t0) * 1000

        # (c) ACCEPT, on device (T4.132): how many of the k_eff drafts hold up (m, 0..k_eff) and the one extra token
        # this iteration emits beyond them -- `final[i]` is what gets emitted if the accept length is i: greedy, the
        # main model's own id at position i (the bonus at i == k_eff, the correction below it); temperature>0,
        # spec_accept_tensor's residual resample / bonus sample. Then the iteration's ONE host sync reads back
        # [m, the chunk's ids, final] -- 2n+1 ints -- and the accepted tokens are assembled on the host. verify_ids[i]
        # predicts the position AFTER chunk[i] was fed, so it is compared with draft d_i == chunk[i+1] (chunk[0] is
        # tok_last, already confirmed real); m = the longest matching prefix. The same replay (_accept_step, family
        # ("accept", n, greedy)) gathers the next anchor final[m] and h_last = verify_h[:, m] (VERIFY's hidden at the last
        # accepted position; causal -- row m never saw a draft past it) into its own output buffers, the next chain's
        # inputs, and rewrites the MTP block's cache slots from VERIFY's hidden (see _accept_step).
        if trace_on: accept_t0 = time.perf_counter()
        acc_args = (verify_logits, verify_h, chunk, temp_t, v_mtp_pos.bind(start_pos + 1), *q_logits)
        accepted_t = cast(tuple[Tensor, ...], self._spec_replay(("accept", n, greedy), functools.partial(self._accept_step, greedy), *acc_args)
                          if k_eff else self._accept_step(greedy, *acc_args))
        tok_last, h_last = accepted_t[1], accepted_t[2]
        readback = cast(list[int], accepted_t[0].tolist())  # THE sync: draft chain + VERIFY + the accept test
        m, chunk_ids, final_ids = readback[0], readback[1:1 + n], readback[1 + n:]
        assert 0 <= m <= k_eff and len(chunk_ids) == len(final_ids) == n, f"{m=} {k_eff=} {n=}"
        if debug_on:  # T4.132: a second sync, debug only -- what VERIFY actually produced for this chunk
          lg = verify_logits.float()[0]
          print(f"[SPEC_DEBUG] start_pos={start_pos} k_eff={k_eff} m={m} chunk={chunk_ids} verify_argmax={lg.argmax(-1).tolist()} final={final_ids} "
                f"nan_per_row={(lg != lg).sum(-1).tolist()} inf_per_row={(lg.abs() == float('inf')).sum(-1).tolist()}", file=sys.stderr, flush=True)
        spec_check_ids(chunk_ids + final_ids, vocab, f"start_pos={start_pos} m={m} chunk={chunk_ids} final={final_ids}")
        accepted = chunk_ids[1:1 + m] + [final_ids[m]]  # d0..d_{m-1} plus the bonus/correction token: m+1 total
        if trace_on: accept_ms = (time.perf_counter() - accept_t0) * 1000

        # T4.132: a partial accept (m < k_eff) also folded the wrong drafts d_m.. into every GDN block's state; each block's VERIFY
        # stored its state after every token (spec_track), so slot m -- the state after exactly the accepted tokens -- becomes the live
        # state: one replay of the adopt family (~150 MB of copies at the 27B's geometry), instead of T4.66d's CHECKPOINT + a REDO
        # forward of the m+1 accepted tokens (the card's phase table: 43-79 ms per partial accept). Attention KV needs nothing (the
        # discarded positions start_pos+m+1.. are never read before the next VERIFY overwrites them -- SPEC_NOTES.md section 2).
        # Dispatch-only (state_assign_ms); the device queue orders it before the next VERIFY.
        if trace_on: assign_t0 = time.perf_counter()
        if m < k_eff and self.has_recurrent_block: self._spec_replay(("adopt",), self._adopt_step, v_adopt.bind(m))
        if trace_on: state_assign_ms = (time.perf_counter() - assign_t0) * 1000

        if stats_on:
          accept_hist[m] = accept_hist.get(m, 0) + 1
          stats_iters += 1
          stats_drafted += k_eff
          stats_emitted += m + 1

        if trace_on:
          trace_iter += 1
          total_ms = (time.perf_counter() - iter_t0) * 1000
          print(f"[SPEC_TRACE] iter={trace_iter} k_eff={k_eff} m={m} "
                f"draft_ms={draft_ms:.2f} draft_dispatch_ms={draft_ms:.2f} "
                f"verify_dispatch_ms={verify_dispatch_ms:.2f} accept_ms={accept_ms:.2f} "
                f"state_assign_ms={state_assign_ms:.2f} total_ms={total_ms:.2f}")

        start_pos += m + 1
        # T4.132: the state is committed through accepted[:-1] BEFORE the first yield (VERIFY fed the drafts d0..d_{m-1};
        # accepted[-1] is the model's own prediction, not fed yet) and _cached_tokens says so at once -- not tokens[:-1] per
        # yield, which a caller stopping at a stop token inside this batch (serve.py's is_end) would read as "the state is
        # exactly my tokens" and resume a recurrent state that is 1..m tokens ahead of it. get_start_pos reuses the live
        # cache only when the next prompt extends _cached_tokens exactly; otherwise it re-prefills (or serve.py restores a
        # state-cache snapshot) -- test_spec_decode.py's test_stop_mid_batch_then_generate_matches_fresh_gdn.
        committed = tokens + accepted[:-1]
        for v in accepted:
          tokens.append(v)
          self._cached_tokens = committed
          yield tokens[-1]
    finally:
      # T4.66b: SPEC_STATS instrumentation -- see its own ContextVar docstring. Fires once, whichever way this
      # generator ends: the loop above exhausting max_context, or the caller closing/abandoning it (a request
      # ending -- serve.py's Handler.run_model drops `gen` when its own generator ends or is GC'd, which
      # raises GeneratorExit here at whatever yield this was suspended on).
      if stats_on and stats_iters:
        hist_str = ", ".join(f"{acc}:{cnt}" for acc, cnt in sorted(accept_hist.items()))
        print(f"[SPEC_STATS] iters={stats_iters} emitted={stats_emitted} drafted={stats_drafted} "
              f"avg_accept_len={stats_emitted/stats_iters:.2f} drafts_per_token={stats_drafted/stats_emitted:.2f} "
              f"accept_len_hist={{{hist_str}}}")

@dataclass(frozen=True)
class VisionInput:
  """T5.3: the visual side of one prompt. spans: per image in id order, (offset of its run in the id list, visual token count n,
  (t, gh, gw) PATCH grid) with n == (gh//2)*(gw//2) (t must be 1 -- video is out of scope); embeds: the encoder's (sum of n, D) rows in
  the same order (T5.2). The ids at a span's positions are placeholders the model never embeds (T5.4 puts hash-derived ids there so the
  sequence stays cache-keyable); vision_start/vision_end are ordinary text tokens outside the span."""
  spans: list[tuple[int, int, tuple[int, int, int]]]
  embeds: Tensor

def vision_positions(spans:list[tuple[int, int, tuple[int, int, int]]], prompt_len:int, start_pos:int,
                     delta:int) -> tuple[np.ndarray, np.ndarray, int]:
  """HF get_rope_index for ids[start_pos:prompt_len]: (pos3 (n, 3) int32 (t, h, w) rope ids, mask (n,) bool = visual positions, the rope
  deficit at prompt_len). Text tokens count up from start_pos - delta; an image span at rope position p takes (p, p+hi, p+wi) over its
  merged gh//2 x gw//2 grid row-major, and the text after it resumes at p + max(gh, gw)//2 -- so each image adds n - max(gh, gw)//2 to
  the deficit. Spans fully inside the reused prefix are `delta`'s business (skipped); a boundary inside a span is a caller bug."""
  import numpy as np
  n = prompt_len - start_pos
  pos3, mask = np.zeros((n, 3), np.int32), np.zeros(n, bool)
  p, i = start_pos - delta, start_pos  # rope position of ids[i]
  for off, cnt, (gt, gh, gw) in sorted(spans):
    if off + cnt <= start_pos: continue
    assert off >= start_pos, f"vision_positions: reuse boundary {start_pos} inside the image span at {off}+{cnt}"
    assert gt == 1 and cnt == (gh // 2) * (gw // 2), f"vision_positions: span {(off, cnt, (gt, gh, gw))} is not one image of (gh//2)*(gw//2) tokens"
    pos3[i-start_pos:off-start_pos] = np.arange(p, p + off - i)[:, None]  # the text run before the span
    p, i = p + off - i, off
    hi, wi = np.meshgrid(np.arange(gh // 2), np.arange(gw // 2), indexing="ij")
    pos3[i-start_pos:i-start_pos+cnt] = p + np.stack([np.zeros_like(hi), hi, wi], -1).reshape(cnt, 3)
    mask[i-start_pos:i-start_pos+cnt] = True
    p, i = p + max(gh, gw) // 2, i + cnt
  pos3[i-start_pos:] = np.arange(p, p + prompt_len - i)[:, None]  # the trailing text
  return pos3, mask, prompt_len - (p + prompt_len - i)

class _VisionPlan:
  """T5.3, generate()'s per-request image bookkeeping: vision_positions over the chunks still to prefill, the encoder rows on the host,
  and chunk() -- the three chunk-max-sized side inputs (vis_e, vis_m, vis_pos: see Transformer.forward) for the chunk at [sp, sp+n)."""
  def __init__(self, vision:VisionInput|None, prompt_len:int, start_pos:int, delta:int, chunk:int, dim:int, device:str|tuple[str, ...]|None):
    import numpy as np
    self.spans = sorted(vision.spans) if vision is not None else []
    self.start, self.width, self.dim, self.device = start_pos, chunk, dim, device
    self.pos3, self.mask, self.delta = vision_positions(self.spans, prompt_len, start_pos, delta)
    self.rows = vision.embeds.numpy().astype(np.float32) if vision is not None else None
    if self.rows is not None: assert self.rows.shape == (sum(c for _, c, _ in self.spans), dim), f"embeds {self.rows.shape} vs spans/dim"
  def chunk(self, sp:int, n:int) -> tuple[Tensor, Tensor, Tensor]:
    import numpy as np
    e, m, p3 = np.zeros((1, self.width, self.dim), np.float32), np.zeros((1, self.width), bool), np.zeros((1, self.width, 3), np.int32)
    lo = sp - self.start
    m[0, :n], p3[0, :n] = self.mask[lo:lo+n], self.pos3[lo:lo+n]
    base = 0
    for off, cnt, _ in self.spans:
      a, b = max(off, sp), min(off + cnt, sp + n)  # the span's overlap with this chunk
      if a < b: e[0, a-sp:b-sp] = cast(np.ndarray, self.rows)[base + a - off:base + b - off]
      base += cnt
    return Tensor(e, device=self.device), Tensor(m, device=self.device), Tensor(p3, device=self.device)

def snapshot_nbytes(snap:dict) -> int:
  """Total device-buffer bytes a Transformer.snapshot_state() dict holds: recursively sums every Tensor's
  size*itemsize (tokens/pos are plain python and cost nothing here). serve.py's state-cache LRU sizes its
  MB cap against this."""
  def walk(x:object) -> int:
    if isinstance(x, Tensor): return math.prod(int(d) for d in x.shape) * x.dtype.itemsize
    if isinstance(x, dict): return sum(walk(v) for v in x.values())
    if isinstance(x, (list, tuple)): return sum(walk(v) for v in x)
    return 0
  return walk(snap)

def snapshot_nbytes_for(snap:dict, n_tokens:int) -> int:
  """Predict the bytes a snapshot of an n_tokens-long sequence would take, from an existing snapshot: KV-cache slices
  (`cache_kv`/`cache_k`/`cache_kv_scale`, one row per token) scale with the length; GDN conv/recurrent states (and
  everything else) are fixed-size accumulators and count once. A plain nbytes*n/len(tokens) extrapolation charges the
  fixed states per token and predicts ~29 GB for a 39k-token session (T5.7b) -- then serve.py skips snapshots that
  would fit. T4.100: `cache_kv_scale` (T6.1 KV_INT8 / T4.100 KV_INT4's per-block absmax scales, see kv_int8_block)
  is position-indexed exactly like `cache_kv` itself -- it used to fall into the `else` branch below and get
  charged as a one-off fixed cost instead of scaling with n_tokens, silently under-predicting int8/int4 KV sessions."""
  pos = max(int(snap.get("pos", len(snap.get("tokens", [])))), 1)
  per_token, fixed = 0.0, 0
  for blk in snap.get("blocks", []):
    for k, v in blk.items():
      if not isinstance(v, Tensor): continue
      nb = math.prod(int(d) for d in v.shape) * v.dtype.itemsize
      if k in ("cache_kv", "cache_k", "cache_kv_scale"): per_token += nb / pos
      else: fixed += nb
  # MTP cache slice (T4.66l) and any other per-token entries at the top level
  for k, v in snap.items():
    if k != "blocks" and isinstance(v, Tensor): per_token += math.prod(int(d) for d in v.shape) * v.dtype.itemsize / pos
  return int(fixed + per_token * n_tokens)

def snapshot_matches(snap:dict, tokens:list[int]) -> bool:
  """True iff snap's cached tokens are a strict, exact prefix of `tokens` -- the same recurrent exact-prefix
  rule get_start_pos enforces (see there): only then is Transformer.restore_state(snap) safe to apply before
  continuing with `tokens`. restore_state itself performs no such check -- callers (serve.py's snapshot
  lookup, tests) must call this first."""
  cached = snap["tokens"]
  return bool(cached) and len(cached) < len(tokens) and tokens[:len(cached)] == cached
