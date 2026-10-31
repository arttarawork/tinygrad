"""T4.137: an attention block's decode-step glue as ONE NV kernel (attn_prep_nv) -- everything TransformerBlock._attention does between
the q/k/v projections and the attention kernel, for a concrete run of T <= 4 tokens (a decode step, a speculative VERIFY run) on a
KV_INT4 cache: the projections' split-K partial sums, the per-head q/k RMSNorm, rope on the first rope_dim dims, the output-gate split,
and the int4 KV store (per-32-dim absmax scale, nibbles) into cache_kv / cache_kv_scale in place.

Why: the generic graph ran it as 9 launches per attention block per step (10-31 card profile: q/k partial sums, the two norms' sums of
squares, q norm+rope, k/v sum+norm+rope, the absmax, the scale store, the nibble store), each 5.5-6.8 us of launch latency for a few KB
of data -- 0.85 ms of a 26 ms decode step over the 27B's 16 attention blocks, and the same again inside every VERIFY.

Layout: block = (token row, kv head), G warps (the GQA q heads of that kv head), lane = AxisType.WARP (threadIdx.x, never split: the
shuffles below pair HARDWARE lanes). Lane l owns the contiguous dims l*PER .. l*PER+PER-1 (PER = head_dim/32) of each vector, so
  - a 32-dim quant block is the GL = 32/PER lanes l//GL: its absmax is log2(GL) xor shuffles, its scale the one store of lane l%GL == 0
  - a lane's PER/2 packed bytes are its own PER nibbles (low nibble = even dim): no shuffle, no gate
  - rope pairs dim i < rope_dim/2 with i + rope_dim/2, which lives RP = rope_dim/2/PER lanes away: one xor shuffle per element
  - the RMS sum of squares is a full xor butterfly (every lane ends with the head's total)
Warp w handles q head kvh*G + w and its output gate; every warp of the block also computes the block's k and v head (a few hundred
redundant FLOPs), and since those cache stores don't depend on the warp the lowering issues them from warp 0 only (`lidx1==0`, pinned by
the render test) -- no hand-written warp gate, which the index lowering would drop anyway (schedule/indexing.py's reshape rewrite:
symbolic.drop_and_clauses keeps only gate clauses whose ranges appear in the gated index component).

Numerics follow the generic graph op for op (the same UOp mixin ops: x/s is x*(1/s), round is half-to-even, nn.RMSNorm's
x * (mean(x^2) + eps).sqrt().reciprocal() * weight, apply_rope's x1*cos - x2*sin / x2*cos + x1*sin, model.py's int4 encoding); the
partials are summed left to right from 0 as the generic reduce does. Only the RMS sum of squares runs in a different order (lane
partials + butterfly vs the generic reduce's), so q/k can differ from the generic path in the last ulp and a k nibble can flip on an
exact rounding tie -- the flip gate's level. Gate: model.py's TransformerBlock._nv_attn_prep (NV_ATTN_PREP, default 1; 0 = the
generic graph)."""
from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import _unbind
from tinygrad.llm.kernels.nv_attn import _shfl, WARP_SIZE
from tinygrad.llm.kernels.nv_quant import split_k_partials

NV_ATTN_PREP = ContextVar("NV_ATTN_PREP", 1)  # 0 = the generic graph (A/B); read in model.py's TransformerBlock._nv_attn_prep only

def attn_prep_geometry_ok(n_heads:int, n_kv_heads:int, head_dim:int, rope_dim:int) -> bool:
  """the lane layout above fits: whole packed bytes per lane (64 | head_dim), PER dividing 32 (whole quant blocks per lane group), at most
  32 warps per block, and rope halves made of whole lanes a power of two apart (the xor partner)"""
  per = head_dim // WARP_SIZE
  if head_dim % 64 or WARP_SIZE % per or n_heads % n_kv_heads or n_heads // n_kv_heads > 32: return False
  if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2 or (rope_dim // 2) % per: return False
  rp = rope_dim // 2 // per
  return rp & (rp - 1) == 0

def _butterfly(val:UOp, lane:UOp, offsets:tuple[int, ...], maximum:bool=False) -> UOp:
  for off in offsets:
    other = _shfl(val, lane ^ off)
    val = val.maximum(other) if maximum else val + other
  return val

@functools.cache
def _attn_prep_kernel(q_out:UOp, gate_out:UOp, cache_kv:UOp, cache_scale:UOp, q_part:UOp, k_part:UOp, v_part:UOp, q_norm:UOp, k_norm:UOp,
                      freqs:UOp, *bound:UOp, start_pos:int|UOp, rope_pos:int|UOp, eps_q:float, eps_k:float) -> UOp:
  """q_out (B,H,T,D) fp32 = rope(rmsnorm(q) * q_norm); gate_out (B*T, H*D) fp32 = the gate half of attn_q; cache_kv (2,B,KVH,N,D/2) uint8 and
  cache_scale (2,B,KVH,N,D/32) fp16 rows start_pos+t <- int4(rope(rmsnorm(k) * k_norm)) / int4(v). q_part/k_part/v_part are the projections'
  split-K partials (B*T, F, C), q's columns per head [q (D) | gate (D)] (attn_output_gate's interleave); freqs is the whole rope table
  (max_context, rope_dim) = cos | sin, read at row rope_pos+t. The bound variables ride last (`bound`, never loaded: see attn_prep)."""
  B, H, T, D = cast(tuple[int, int, int, int], q_out.shape)
  _, _, KVH, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  G, PER = H // KVH, D // WARP_SIZE
  half, rows = cast(int, freqs.shape[1]) // 2, cast(int, freqs.shape[0])
  RP, GL = half // PER, WARP_SIZE // PER  # rope partner distance in lanes; lanes per 32-dim quant block
  block = UOp.range(B * T * KVH, 0, AxisType.GLOBAL)
  warp = UOp.range(G, 1, AxisType.LOCAL) if G > 1 else UOp.const(0, dtypes.weakint)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  row, kvh = block // KVH, block % KVH  # row = b*T + t, the projections' token row
  b, t = row // T, row % T
  head = kvh * G + warp
  dims = [lane * PER + j for j in range(PER)]
  def summed(part:UOp, col:UOp) -> UOp:  # left to right from 0: the generic reduce's order (add_rmsnorm_q8's partials note)
    return sum((part[row, col, c].load().float() for c in range(cast(int, part.shape[-1]))), UOp.const(0, dtypes.float32))
  def rms_norm(xs:list[UOp], weight:UOp, eps:float) -> list[UOp]:
    ss = _butterfly(sum((x * x for x in xs), UOp.const(0, dtypes.float32)), lane, (16, 8, 4, 2, 1))
    inv = ((ss * (1 / D)) + eps).sqrt().reciprocal()  # nn.RMSNorm: x * (mean(x^2) + eps).rsqrt(), then * weight
    return [x * inv * weight[d].load() for x, d in zip(xs, dims)]
  # rope rows past the table (a start_pos the caller never runs: VERIFY's variable is declared to max_context-1) clamp instead of
  # reading out of bounds; the same for the cache rows below
  rope_row = (rope_pos + t).minimum(rows - 1)
  def rope(ys:list[UOp]) -> list[UOp]:
    out = []
    for j, y in enumerate(ys):
      partner = _shfl(y, lane ^ RP)  # every lane shuffles (full mask); lanes past the rope dims keep y
      i = (lane % RP) * PER + j      # the pair's frequency: dim i of the first half, i + half of the second
      cos, sin = freqs[rope_row, i].load().float(), freqs[rope_row, half + i].load().float()
      out.append((lane < RP).where(y * cos - partner * sin, (lane < 2 * RP).where(y * cos + partner * sin, y)))
    return out
  q = rope(rms_norm([summed(q_part, head * 2 * D + d) for d in dims], q_norm, eps_q))
  gate = [summed(q_part, head * 2 * D + D + d) for d in dims]
  k = rope(rms_norm([summed(k_part, kvh * D + d) for d in dims], k_norm, eps_k))
  v = [summed(v_part, kvh * D + d) for d in dims]
  stores = [q_out[b, head, t, d].store(x) for x, d in zip(q, dims)] + [gate_out[row, head * D + d].store(x) for x, d in zip(gate, dims)]
  pos = (start_pos + t).minimum(N - 1)
  offsets = tuple(GL >> s for s in range(1, GL.bit_length()))  # GL/2 .. 1: the GL lanes of this lane's quant block
  for kv, xs in enumerate((k, v)):
    # model.py's KV_INT4 store: scale = absmax/7 (min 1e-8), nibble = round(x/scale) clipped to [-8,7] + 8, low nibble = even dim
    amax = _butterfly(functools.reduce(lambda a, x: a.maximum(x.abs()), xs[1:], xs[0].abs()), lane, offsets, maximum=True)
    scale = (amax / 7.0).maximum(1e-8)
    nib = [((x / scale).round().clip(-8, 7) + 8).cast(dtypes.uint8) for x in xs]
    stores += [cache_kv[kv, b, kvh, pos, lane * (PER // 2) + i].store(nib[2 * i].bitwise_or(nib[2 * i + 1].lshift(4))) for i in range(PER // 2)]
    stores.append(cache_scale[kv, b, kvh, pos, (lane // GL).valid((lane % GL).eq(0))].store(scale.cast(cache_scale.dtype)))
  ends = (lane, warp, block) if G > 1 else (lane, block)
  return UOp.group(*stores).end(*ends).sink(arg=KernelInfo(name="attn_prep_nv", opts_to_apply=()))

def attn_prep(q:Tensor, k:Tensor, v:Tensor, q_norm:Tensor, k_norm:Tensor, freqs:Tensor, cache_kv:Tensor, cache_scale:Tensor,
              start_pos:int|UOp, rope:int|UOp, n_heads:int, eps:tuple[float, float]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """q (B,T,H*2*D), k/v (B,T,KVH*D): attn_q/attn_k/attn_v's outputs -- their split-K partials when nv_q8_linear made them (the summing
  reduces are then never scheduled), else the tensors themselves as one-wide partials. q_norm/k_norm: fp32 copies made at load
  (TransformerBlock.prepare_nv_norms). Returns (q (B,H,T,D) fp32, gate (B,T,H*D) fp32, cache_kv and cache_scale after the in-place store)."""
  B, T = cast(tuple[int, int], q.shape[:2])
  D = cast(int, cache_kv.shape[4]) * 2
  parts = [p if (p := split_k_partials(x)) is not None else x.reshape(B * T, x.shape[-1], 1) for x in (q, k, v)]
  q_out = Tensor.empty(B, n_heads, T, D, dtype=dtypes.float32, device=q.device)
  gate = Tensor.empty(B * T, n_heads * D, dtype=dtypes.float32, device=q.device)
  # the kernel reads start_pos/rope as PARAMs by name (kernel_var); the bound variables themselves ride along as the LAST CALL srcs so the
  # schedule binds them (kernels/nv.py gdn_decode's idiom: a precompiled @function drops arguments no kernel references)
  binds = [Tensor(u) for u in dict.fromkeys(u for x in (start_pos, rope) if isinstance(x, UOp) for u in x.toposort() if u.is_bound_var)]
  fxn = functools.partial(_attn_prep_kernel, start_pos=_unbind(start_pos), rope_pos=_unbind(rope), eps_q=eps[0], eps_k=eps[1])
  q_out, gate, kv, scale = Tensor.custom_kernel(q_out, gate, cache_kv, cache_scale, *parts, q_norm, k_norm, freqs, *binds, fxn=fxn)[:4]
  return q_out, gate.reshape(B, T, n_heads * D), kv, scale
