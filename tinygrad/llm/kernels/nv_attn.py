"""T4.107a: NV split-KV ("flash-decoding") attention for the T==1 decode step, reading the KV cache AS STORED (KV_INT4 packed
nibbles + fp16 scales, KV_INT8 + scales, or plain fp16) -- no dequantized k/v copies, no generic reduce kernels.

Why: on the card path decode costs ~44 ms + 4.5 ms per 1k tokens of filled context (HANDOFF_2026-09-12 §6): the GDN blocks are
O(1)/token, the slope is the 16 attention layers' generic SDPA kernels (register spills, a serial reduce over the whole context).
Design = amd.py's _amd_flash_attention_decode_partial (RDNA3, fp16 cache), kept separate so that kernel stays untouched (the
nv.py idiom): grid = (batch x kv_head) x (chunk slot), one 32-lane warp per block; each lane owns a contiguous head_dim/32 run of
the head (one 4-byte int4 word per key), the warp walks its chunks of BLOCK_N keys KEY_GROUP at a time, dequantizes K/V in
registers, dots every GQA query head of the kv head against them (q loaded once per block, warp_reduce = __shfl_xor_sync), and
keeps the online-softmax (running max, sum, PER-lane acc) per query head. Chunk slots x keys are STATIC in max_context, but the
GLOBAL range over slots is the symbolic min(valid chunks, slots) and every key beyond valid_kv_len is a masked load (no traffic),
so the work tracks the FILLED context, and the bound start_pos reaches the kernel as a PARAM (kernel_var) like the AMD path. The
(m, l, acc) partials are merged with plain tensor ops (_merge_partials). Gate: NV_CUSTOM_ATTN=1 (default 0 = byte-identical
generic path everywhere) -- validate on the card with extra/nv_attn_validate_real.py before turning it on in a recipe.

T4.107b: the same skeleton for a PREFILL chunk (T <= 64 query tokens whose K/V are already in the cache). The remaining slope after
T4.107a is prefill: 8.8 ms + 0.95 ms per 1k filled tokens per new token, because _sdpa_default materialises the (chunk x filled)
score matrix per head through generic kernels (the T4.101 arena). Like amd.py's flash_attention the symbolic chunk is padded to
its static width and the causal limit comes in as q_start (row r sees keys <= q_start + r; garbage rows are sliced off);
grid = (batch x kv head x row tile) x chunk slot, NV_ATTN_QROWS query rows per block (1 = exactly the decode kernel's register
footprint, per row; 2/4 share each K/V load across rows at the price of registers -- a card-side knob). No tensor cores (a
warp-shuffle dot per key, as decode); memory is O(tile) -- the planned partials are (B,H,T_pad,64 slots,D) fp32 instead of
2 x (B,H,T,max_context) fp32. Masked rows/slots use a finite running max (SENTINEL) so a slot with no visible key never
produces exp2(-inf - -inf) = NaN."""
from __future__ import annotations
import functools, math
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve
from tinygrad.llm.kernels.amd import _unbind
from tinygrad.llm.kernels.nv import warp_reduce, _nv_device_ok

NV_CUSTOM_ATTN = ContextVar("NV_CUSTOM_ATTN", 0)
NV_ATTN_QROWS = ContextVar("NV_ATTN_QROWS", 1)  # T4.107b: query rows per prefill block (register budget vs K/V re-reads; tune on the card)
NV_ATTN_WMMA = ContextVar("NV_ATTN_WMMA", 0)  # T4.113: the tensor-core prefill kernel below (0 = the T4.107b warp-shuffle kernel)
NV_ATTN_WMMA_DSPLIT = ContextVar("NV_ATTN_WMMA_DSPLIT", 1)  # T4.113: head_dim halves per block (2 = half the O registers, QK^T twice); card knob
WARP_SIZE, LOG2E = 32, math.log2(math.e)
BLOCK_N, MAX_CHUNKS, KEY_GROUP = 128, 256, 4  # keys per chunk; partial slots per (batch, q head); keys per loop step (one rescale per group)
PREFILL_CHUNKS, MAX_PREFILL_ROWS, SENTINEL = 64, 256, -1e30  # prefill: slots at T_pad<=32; widest chunk handled (T4.124b); finite "no key yet" max
WMMA_M, WMMA_N, WMMA_K = 16, 8, 16  # mma.sync.aligned.m16n8k16 fp16 in / fp32 out (nv_gemm.py's fragment map)
WMMA_ARG = ((WMMA_N, WMMA_M, WMMA_K), 'NV', WARP_SIZE)

def nv_decode_attention_ok(device:str|tuple[str, ...]|None, head_dim:int, cache_kv:Tensor) -> bool:
  """The decode kernel covers this layer: NV (sm_70+), a head_dim each lane can own a contiguous run of (32 | head_dim; 64 | head_dim
  for the packed int4 cache so a lane never straddles a byte), a cache length in whole chunks, and a cache dtype it can dequantize."""
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV" or not NV_CUSTOM_ATTN.value: return False
  N = cache_kv.shape[3]
  if not isinstance(N, int) or N % BLOCK_N or head_dim % (64 if cache_kv.dtype == dtypes.uint8 else 32): return False
  if cache_kv.dtype not in (dtypes.uint8, dtypes.int8, dtypes.float16): return False
  return _nv_device_ok(device)

def _reg(shape:tuple[int, ...], slot:int, value:float, *deps:UOp) -> UOp:
  ret = UOp.placeholder(shape, dtypes.float, slot=slot, addrspace=AddrSpace.REG)
  for dep in deps: ret = ret.after(dep)  # (re)initialise inside these ranges: a LOOP-form grid (the CPU test) reuses the registers per block
  return ret.after(ret.store(ret.const_like(value)))

def kv_value(cache_kv:UOp, scale:UOp|None, kv:int, b:UOp|int, h:UOp|int, key:UOp, d:UOp) -> UOp:
  """Element d of K (kv=0) / V (kv=1) at `key`, as stored by model.py's _attention: uint8 = two 4-bit values per byte (low nibble =
  even element, biased by 8) times the fp16 absmax scale of its kv_int8_block; int8 = value times the same scale; else a plain cast.
  Pure index math on the placeholders (no lane assumptions), so the CPU test can drive it with GLOBAL ranges."""
  if cache_kv.dtype == dtypes.uint8:
    assert scale is not None
    blk = cast(int, cache_kv.shape[-1]) * 2 // cast(int, scale.shape[-1])
    nibble = (cache_kv[kv, b, h, key, d // 2].load().cast(dtypes.uint32) >> ((d % 2) * 4)) & 0xF
    return (nibble.cast(dtypes.float) - 8.0) * scale[kv, b, h, key, d // blk].load().float()
  if cache_kv.dtype == dtypes.int8:
    assert scale is not None
    blk = cast(int, cache_kv.shape[-1]) // cast(int, scale.shape[-1])
    return cache_kv[kv, b, h, key, d].load().float() * scale[kv, b, h, key, d // blk].load().float()
  return cache_kv[kv, b, h, key, d].load().float()

@functools.cache
def _decode_partial_kernel(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *scale:UOp, valid_kv_len:int|UOp, max_kv_len:int) -> UOp:
  valid_kv_len = _unbind(valid_kv_len)
  sc = scale[0] if scale else None
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  G, PER, chunks = H // H_KV, D // WARP_SIZE, cast(int, out.shape[2])
  assert M == 1 and H % H_KV == 0 and PER * WARP_SIZE == D and max_kv_len == N and N % BLOCK_N == 0 and BLOCK_N % KEY_GROUP == 0
  block_bhkv = UOp.range(B * H_KV, 0, AxisType.GLOBAL)
  valid_chunks = (valid_kv_len + BLOCK_N - 1) // BLOCK_N
  group_count = min(valid_chunks, chunks) if isinstance(valid_chunks, int) else valid_chunks.minimum(chunks)
  block_n, lane = UOp.range(group_count, 1, AxisType.GLOBAL), UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  dims = tuple(lane * PER + j for j in range(PER))  # this lane's contiguous run of the head
  q_heads = tuple(kv_head * G + head for head in range(G))
  qs = tuple(tuple(q[b, q_head, 0, d].load().float() * (1 / math.sqrt(D)) for d in dims) for q_head in q_heads)  # once per block
  acc, row_max, row_sum = _reg((G, PER), 0, 0), _reg((G,), 1, -math.inf), _reg((G,), 2, 0)
  groups_per_chunk = BLOCK_N // KEY_GROUP
  steps = ((valid_chunks + group_count - 1) // group_count) * groups_per_chunk
  offset = UOp.range(steps, 100, AxisType.REDUCE)
  chunk = block_n + (offset // groups_per_chunk) * group_count
  keys = tuple(chunk * BLOCK_N + (offset % groups_per_chunk) * KEY_GROUP + i for i in range(KEY_GROUP))
  valid = tuple(key < valid_kv_len for key in keys)
  kvals, vvals = (tuple(tuple(kv_value(cache_kv, sc, kv, b, kv_head, key.valid(ok), d) for d in dims) for key, ok in zip(keys, valid))
                  for kv in range(2))
  updates:list[UOp] = []
  for head in range(G):
    scores = tuple(warp_reduce(sum((qv * k for qv, k in zip(qs[head], key_kvals)), UOp.const(0, dtypes.float)), full_wave=True)
                   for key_kvals in kvals)
    prev_acc, prev_max, prev_sum = acc.after(offset)[head], row_max.after(offset)[head], row_sum.after(offset)[head]
    new_max = functools.reduce(lambda a, vs: a.maximum(vs[0].where(vs[1], UOp.const(-math.inf, dtypes.float))), zip(valid, scores), prev_max)
    alpha = ((prev_max - new_max) * LOG2E).exp2()
    betas = tuple(ok.where(((score - new_max) * LOG2E).exp2(), UOp.const(0, dtypes.float)) for ok, score in zip(valid, scores))
    updates += [acc[head].store(prev_acc * alpha + sum((UOp.stack(*value) * beta for value, beta in zip(vvals, betas)), acc[head].const_like(0))),
                row_sum[head].store(prev_sum * alpha + sum(betas, UOp.const(0, dtypes.float))), row_max[head].store(new_max)]
  update = UOp.group(*updates).end(offset)
  acc, row_max, row_sum = acc.after(update), row_max.after(update), row_sum.after(update)
  stores = [out[b, q_head, block_n, d].store(acc[head, j]) for head, q_head in enumerate(q_heads) for j, d in enumerate(dims)] + \
    [stats[b, q_head.valid(lane.eq(0)), block_n, i].store(x[head]) for head, q_head in enumerate(q_heads) for i, x in enumerate((row_max, row_sum))]
  return UOp.group(*stores).end(lane, block_n, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial_nv", opts_to_apply=()))

def _merge_partials(partial:Tensor, stats:Tensor, valid_kv_len:int|UOp) -> Tensor:
  """(...,D) attention from the live chunk slots' (acc, m, l) along the second-to-last dim of partial (B,H,slots,D) for decode,
  (B,H,rows,slots,D) for prefill: out = sum_s acc_s e^(m_s-m) / sum_s l_s e^(m_s-m). Plain tensor ops. A slot a row never saw
  (prefill: causal limit below the slot) carries (m=SENTINEL, l=0, acc=0) and weighs exactly 0."""
  chunks = cast(int, partial.shape[-2])
  live = (valid_kv_len + BLOCK_N - 1) // BLOCK_N
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  partial, stats = partial[..., :live, :], stats[..., :live, :]
  weights = ((stats[..., 0] - stats[..., 0].max(-1, keepdim=True)) * LOG2E).exp2()
  return (partial * weights.unsqueeze(-1)).sum(-2) / (stats[..., 1] * weights).sum(-1, keepdim=True)

def nv_decode_attention(q:Tensor, cache_kv:Tensor, cache_kv_scale:Tensor|None, valid_kv_len:int|UOp) -> Tensor:
  """q (B,H,1,D) against the cache (already written through for this step, so the AFTER chain orders the kernel behind the store);
  valid_kv_len = start_pos+1 stays bound at the graph level. Returns (B,H,1,D) in float32. T4.129: a KV_INT4 cache on sm_80+ takes
  the tensor-core kernel (_decode_mma_kernel) unless NV_ATTN_DEC_MMA=0."""
  if NV_ATTN_DEC_MMA.value and _decode_mma_ok(q, cache_kv, cache_kv_scale):
    return _decode_mma_attention(q, cache_kv, cast(Tensor, cache_kv_scale), valid_kv_len)
  B, H, D = cast(int, cache_kv.shape[1]), cast(int, q.shape[1]), cast(int, q.shape[3])
  N = cast(int, cache_kv.shape[3])
  chunks = min(MAX_CHUNKS, N // BLOCK_N)
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  fxn = functools.partial(_decode_partial_kernel, valid_kv_len=valid_kv_len, max_kv_len=N)
  srcs = (partial, stats, q.contiguous(), cache_kv) + ((cache_kv_scale,) if cache_kv_scale is not None else ())
  partial, stats = Tensor.custom_kernel(*srcs, fxn=fxn)[:2]
  return _merge_partials(partial, stats, valid_kv_len).unsqueeze(2)

# ******** T4.129: KV_INT4 decode on the tensor cores ********
# The T4.107a kernel above reached ~159 GB/s at a 131k fill (951 us per layer for 151 MB): every key cost a 32-lane shuffle-reduce
# dot per GQA head (G = 6), a long serial chain per key at ~12 one-warp blocks per SM. Here the G heads of a kv head are the rows
# of int8 mma tiles instead, so a key costs ~1/16 of an mma for S and ~1/16 for P.V, and the per-key work is a handful of
# fragment unpacks. Knobs are read in _decode_mma_attention only (never inside the cached builder):
NV_ATTN_DEC_MMA = ContextVar("NV_ATTN_DEC_MMA", 1)       # 0 = the T4.107a kernel for KV_INT4 too (A/B)
NV_ATTN_DEC_SLOTS = ContextVar("NV_ATTN_DEC_SLOTS", 256)  # partial slots per (batch, q head) at most (= warps per kv head)
NV_ATTN_DEC_WARPS = ContextVar("NV_ATTN_DEC_WARPS", 4)    # warps (each its own slot) per block
NV_ATTN_DEC_KTILES = ContextVar("NV_ATTN_DEC_KTILES", 1)  # 16-key tiles per loop step (one softmax rescale per step); divides 8
# T4.134: VERIFY runs only -- head_dim parts per slot (adjacent blocks: each keeps D/parts of O and reads its share of V, S is
# recomputed per part); divides head_dim/64. O^T is D/16 x 4 fp32 per row group per lane: 192 registers at VERIFY(3), G = 6. sm_86
# cubins (nvrtc lane, 10-30; registers + spill stack bytes): T == 1 194 + 0 (never split); VERIFY(3) at 1: 255 + 1120, 2: 255 + 160,
# 4: 250 + 0; VERIFY(2) at 1: 255 + 112, 2: 224 + 0; VERIFY(4) at 2: 255 + 272, 4: 255 + 80. A card knob (1 / 2 / 4)
# T4.144 diagnosis (nvrtc 12.8 lane, sm_86, 4 warps, 262144 window; STL/LDL total, then inside the key loop = per 16-key step):
#   T == 1:              194 regs +   0 B stack, STL/LDL   0 /   0, loop   0 /   0 (16 IMMA, 16 HMMA)
#   VERIFY(3) DSPLIT 2:  255 regs + 160 B stack, STL/LDL  46 /  44, loop  20 /  40 (32 IMMA, 24 HMMA)
#   VERIFY(4) DSPLIT 2:  255 regs + 272 B stack, STL/LDL  76 /  73, loop  25 /  68
#   VERIFY(3) DSPLIT 1:  255 regs + 1120 B,      STL/LDL 268 / 249, loop 181 / 244;  VERIFY(4) DSPLIT 4: 255 + 80 B, loop 6 / 19
# Where the state sits (VERIFY(3), DSPLIT 2): resident across the loop are the O^T accumulators (3 row groups x 8 dim tiles x 4 =
# 96), the q A fragments (2 m-tiles x 8 blocks x 4 words, 16 of them the zero half = 48) and the q scales (3 x 8 = 24): 168 of 255
# before the step's K words, K scales and S partials. ptxas parks 18 of the read-only q values in local memory before the loop and
# swaps 17 O registers out (STL) for the S phase and back (LDL.LU) for P.V every step -- live state, not an ordering artefact.
# After (T4.144: VERIFY's q words/scales in shared memory, the prologue split over the block's warps): VERIFY(3) DSPLIT 2 249 regs +
# 0 B (3464 SASS instructions per warp, was 8304); VERIFY(4) DSPLIT 2 255 + 16 B (3 loop-invariant index words, one LDL each per
# step; 3424 instructions); VERIFY(2) DSPLIT 2 189 + 0; DSPLIT 4: VERIFY(3) 195 + 0, VERIFY(4) 170 + 0; DSPLIT 1 still spills
# (VERIFY(3) 736 B, VERIFY(4) 936 B). T == 1 unchanged (194 + 0, 3640 instructions).
NV_ATTN_DEC_DSPLIT = ContextVar("NV_ATTN_DEC_DSPLIT", 2)
DEC_IMMA_ARG = ((WMMA_N, WMMA_M, 32), 'CUDA', WARP_SIZE)  # m16n8k32 s8 -> s32 ('CUDA': the PYTHON emulator's fragment model; the
DEC_HMMA_ARG = ((WMMA_N, WMMA_M, WMMA_K), 'CUDA', WARP_SIZE)  # renderer ignores the string) and m16n8k16 f16 -> f32
MAGIC = 0x4B400000  # bitcast(MAGIC + c) = 1.5*2**23 + c exactly for |c| < 2**22 (T4.126): int32 -> fp32 without an I2F

def _shfl(val:UOp, src_lane:UOp) -> UOp:
  """T4.129: __shfl_sync from hardware lane src_lane (the kernel's lane range is AxisType.WARP: lane IS the hardware lane)."""
  return UOp(Ops.CUSTOM, src=(val, src_lane.cast(dtypes.int32)), arg=("__shfl_sync(0xffffffff, {0}, {1})", dtypes.float))

def _decode_mma_ok(q:Tensor, cache_kv:Tensor, cache_kv_scale:Tensor|None) -> bool:
  # KV_INT4 (uint8 nibbles) with 32-wide scale blocks, G <= 8 heads per kv head (rows of one m-tile), D in 64/128/256 (each of the
  # 8 mma row groups owns D/8 dims inside ONE scale block), sm_80+ (the int8 mma)
  from tinygrad.llm.kernels.nv_gemm import _nv_wmma_ok
  H, D, H_KV, N = q.shape[1], q.shape[3], cache_kv.shape[2], cache_kv.shape[3]
  if cache_kv.dtype != dtypes.uint8 or cache_kv_scale is None or not all(isinstance(x, int) for x in (H, D, H_KV, N)): return False
  if D not in (64, 128, 256) or cache_kv_scale.shape[-1] * 32 != D or H % H_KV or not 2 <= H // H_KV <= 8 or N % BLOCK_N: return False
  return _nv_wmma_ok(q.device)

@functools.cache
def _decode_mma_kernel(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, scale:UOp, *, valid_kv_len:int|UOp, max_kv_len:int, warps:int,
                       k_tiles:int, tokens:int=1, d_split:int=1) -> UOp:
  """T4.129: _decode_partial_kernel's contract -- (acc, m, l) per (batch, q head, slot) for _merge_partials -- for a KV_INT4 cache, on
  the tensor cores. Grid: (batch x kv head) x slot blocks of `warps` warps; each warp is one slot and walks chunks slot, slot+S,
  slot+2S, ... of KS = 16*k_tiles keys (S = live slots = min(ceil(valid/KS), out.shape[2])); lane is AxisType.WARP (threadIdx.x,
  never split), g = lane//4, t = lane%4 (nv_gemm's fragment maps). Per 16-key tile:

  S = Q.K^T on mma.m16n8k32 s8 (two 8-key n-tiles x one mma per 32-dim scale block): A = the G heads' q, quantized to int8 per
  (head, block) in the prologue (rows g; rows g+8 and heads >= G are zero), B = the int4 K words as int8 (nibble-8, T4.126's
  unsign). Lane (g, t) gives word 4*blk+t of key g: low nibbles = k 4t..4t+3, high nibbles = k 16+4t.., and A takes q's dims in the
  same order (dims 32*blk+8t+2i / +1), so k never leaves the block and the int32 C is that block's exact dot. The block's scales
  join in fp32: S += float(C) * sq[head, blk] * sk[key, blk], float(C) via MAGIC (folded into the mma's C init). sq carries
  log2(e)/sqrt(D): S is in the exp2 domain.
  The online softmax runs on C's (row g = head, cols 2t, 2t+1 = keys) -- per head over the tile by two xor shuffles -- and the
  same registers ARE the B fragment of O^T += V^T.P^T on mma.m16n8k16 f16 (B[k][n] = P[key k][head n]: k 2t, 2t+1 from n-tile 0,
  2t+8, 2t+9 from n-tile 1, n = g), so P never moves. M = 16 dims per m-tile: row g -> dim D/8*g + 2j (low nibble of byte j of
  row g's D/16-byte run), row g+8 -> the high nibble, so a lane reads one D/16-byte run of V per key and its rows sit in one
  scale block: A = (nibble-8)*sv in fp16 (the prefill kernel's V rounding). C of O^T is (dims, heads 2t/2t+1): the rescale
  needs alpha of heads 2t and 2t+1, two shuffles from lanes 8t, 8t+4. Stored m is natural-log (the merge's convention).
  Numerics vs the T4.107a kernel: q rounded to int8 per 32-block (llama.cpp's q8 x q4 K); P and V*sv rounded to fp16 for the PV
  mma (fp32 accumulate); softmax state in fp32.
  T4.134: tokens = T > 1 is a speculative VERIFY run. q/out/stats arrive as (B, H*T, ...) views (_decode_mma_attention), head-major,
  so a kv head's G = heads x T ROWS are this kernel's "heads" (G <= 32): row group k = rows 8k..8k+7 is S m-tile k//2's rows g
  (k even) / g+8 (k odd) -- the A fragment's g+8 half T == 1 leaves zero -- and the P.V n-tile k (P never moves: C rows g+8 of an
  S tile are the B fragment of the next n-tile). K/V words, their dequant and the int8 B fragments are shared by every row group.
  Row r is token r % T and sees keys < valid_kv_len - (T-1) + r % T: one compare per (row group, score) replaces the shared key
  mask (a select, no branch); the loads keep the shared valid_kv_len mask. d_split > 1 splits a slot's head_dim over that many
  adjacent blocks (NV_ATTN_DEC_DSPLIT); T == 1 at d_split 1 builds exactly the T4.129 graph.
  T4.144: T > 1 quantizes q cooperatively (the block's warps split the (row group, block) units) into shared memory and the key loop
  reads the A words and scales from there: same values, one __syncthreads() after the prologue, none in the loop."""
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  G, slots, KS, blocks, run = H // H_KV, cast(int, out.shape[2]), 16 * k_tiles, D // 32, D // 64  # run = V words per row group
  RG = -(-G // 8)  # T4.134: 8-row groups (1 at T == 1)
  assert M == 1 and 2 <= G <= 32 and G % tokens == 0 and H % H_KV == 0 and D in (64, 128, 256) and max_kv_len == N and N % KS == 0
  assert run % d_split == 0, f"d_split={d_split} must divide head_dim/64 = {run} (whole V words per part)"
  JD = D // 16 // d_split  # O^T dim tiles per block
  assert cache_kv.dtype == dtypes.uint32 and cache_kv.shape[4] * 8 == D, "the int4 cache arrives as uint32 words (_decode_mma_attention)"
  live = (valid_kv_len + KS - 1) // KS
  S = min(live, slots) if isinstance(live, int) else live.minimum(slots)
  per_bh = (S + warps - 1) // warps
  block = UOp.range(B * H_KV * per_bh * d_split, 0, AxisType.GLOBAL)
  cell, dpart = (block // d_split, block % d_split) if d_split > 1 else (block, None)  # T4.134: a slot's head_dim parts are adjacent blocks
  warp = UOp.range(warps, 1, AxisType.LOCAL) if warps > 1 else UOp.const(0, dtypes.weakint)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  bh, slot = cell // per_bh, ((cell % per_bh) * warps + warp).minimum(S - 1)  # surplus warps redo the last slot: same values
  b, kvh = bh // H_KV, bh % H_KV
  g, t = lane // 4, lane % 4
  # prologue: this lane's A fragments (head g, dims 32*blk+8t..+7) and the per-(head, block) scales, q8_quantize's rounding
  from tinygrad.llm.kernels.nv_gemm import _i8x4
  from tinygrad.llm.kernels.nv_quant import _q8_pack
  rows = [g] + [g + 8 * k for k in range(1, RG)]  # T4.134: this lane's row in each row group
  heads = [kvh * G + row for row in rows]
  head_oks = [row < G for row in rows]
  # A masked index component is gated on the `dim` (or key) component, never on the head: the index lowering substitutes what a gate
  # pins (oh < 2 pins t = 0) into the gated component and then drops a gate whose ranges no longer appear in it
  # (schedule/indexing.py's reshape rewrite, pm_drop_and_clauses) -- a head-gated store rendered unconditional at G in {2, 3}
  zeros = (UOp.const(0, dtypes.int8),) * 4
  def quant(head:UOp, head_ok:UOp, blk:UOp|int) -> tuple[UOp, UOp, UOp]:
    # (even-dim word, odd-dim word, the scale S multiplies by) of this lane's 8 dims of (head, block)
    qv = [q[b, head, 0, (blk * 32 + 8 * t + i).valid(head_ok)].load().float() for i in range(8)]
    amax = functools.reduce(lambda a, x: a.maximum(x), (v.abs() for v in qv))
    amax = amax.maximum(_shfl(amax, lane ^ 1))
    amax = amax.maximum(_shfl(amax, lane ^ 2))
    sq = (amax / 127).maximum(1e-8)
    # sq * log2(e)/sqrt(D) as its own max(): as sq * const the symbolic pass pulls the constant out of every product (one more FMUL per term)
    return _q8_pack(tuple(qv[0::2]), sq), _q8_pack(tuple(qv[1::2]), sq), (amax * (LOG2E / math.sqrt(D) / 127)).maximum(1e-8 * LOG2E / math.sqrt(D))
  if tokens == 1:
    qfrags:list[list[tuple[tuple[UOp, ...], tuple[UOp, ...]]]] = []  # per row group, per block: (even dims, odd dims) as int8
    sqs:list[list[UOp]] = []
    for head, head_ok in zip(heads, head_oks):
      qfrags.append([])
      sqs.append([])
      for blk in range(blocks):
        even, odd, sqv = quant(head, head_ok, blk)
        qfrags[-1].append((_i8x4(even), _i8x4(odd)))
        sqs[-1].append(sqv)
  else:
    # T4.144: VERIFY keeps the q words and scales in shared memory instead of registers -- resident across the key loop they cost 72
    # registers at 3 row groups, which pushed the O accumulators into local memory every step (the diagnosis above). The warps of a
    # block share (batch, kv head, head_dim part), hence q: warp w quantizes units (row group, block) w, w + warps, ... (surplus warps
    # redo the last unit: same values, same addresses), add_raw_barrier puts one __syncthreads() after the stores, and the key loop
    # loads each word where its mma needs it (the .after(step) below keeps the loads inside the loop). Same words, same scales:
    # the arithmetic is the registers' bit for bit. Layout [unit][parity][lane] / [unit][lane]: a warp's load is 32 consecutive words.
    units = RG * blocks
    qbuf = UOp.placeholder((units, 2, WARP_SIZE), dtypes.uint32, slot=100, addrspace=AddrSpace.LOCAL)
    sbuf = UOp.placeholder((units, WARP_SIZE), dtypes.float, slot=101, addrspace=AddrSpace.LOCAL)
    pstores = []
    for i in range(-(-units // warps)):
      u = warp + i * warps if units % warps == 0 else (warp + i * warps).minimum(units - 1)
      row = g + 8 * (u // blocks)
      even, odd, sqv = quant(kvh * G + row, row < G, u % blocks)
      pstores += [qbuf[u, 0, lane].store(even), qbuf[u, 1, lane].store(odd), sbuf[u, lane].store(sqv)]
    qbuf, sbuf = qbuf.after(*pstores), sbuf.after(*pstores)
  row_max, row_sum = _reg((RG,), 1, SENTINEL, block), _reg((RG,), 2, 0, block)
  # O^T m-tile j x row group k: (dim rows g, g+8) x (rows 2t, 2t+1 of the group)
  o_acc = [tuple(_reg((4,), 10 + k * JD + j, 0, block) for j in range(JD)) for k in range(RG)]
  steps = (live + S - 1) // S
  step = UOp.range(steps, 100, AxisType.REDUCE)
  if tokens > 1:
    qb, sb = qbuf.after(step), sbuf.after(step)
    qfrags = [[(_i8x4(qb[k * blocks + blk, 0, lane].load()), _i8x4(qb[k * blocks + blk, 1, lane].load())) for blk in range(blocks)]
              for k in range(RG)]
    sqs = [[sb[k * blocks + blk, lane].load() for blk in range(blocks)] for k in range(RG)]
  def half(k:int, blk:int, odd:int) -> tuple[UOp, ...]: return qfrags[k][blk][odd] if k < RG else zeros
  # m-tile mt's A: rows g = row group 2mt, rows g+8 = row group 2mt+1 (zero past the last group: always at T == 1)
  afrags = [[UOp.stack(*half(2 * mt, blk, 0), *half(2 * mt + 1, blk, 0), *half(2 * mt, blk, 1), *half(2 * mt + 1, blk, 1))
             for blk in range(blocks)] for mt in range(-(-RG // 2))]
  base = (slot + step * S) * KS
  def unsign(x:UOp) -> UOp: return (x + 0x78787878) ^ 0x80808080  # 4 nibbles (one per byte) -> 4 int8 nibble-8, T4.126
  scores:list[list[UOp]] = [[] for _ in range(RG)]
  oks, tiles = [], []
  for kt in range(k_tiles):
    keys = [base + 16 * kt + 8 * nt + 2 * t + i for nt in range(2) for i in range(2)]  # this lane's C keys = its PV k 2t, 2t+1, +8, +9
    key_ok = [key < valid_kv_len for key in keys]
    for nt in range(2):
      kb = base + 16 * kt + 8 * nt + g  # this lane's B (K) key
      kb = kb.valid(kb < valid_kv_len)
      sks = [[scale[0, b, kvh, key.valid(ok), blk].load().float() for blk in range(blocks)]
             for key, ok in zip(keys[2*nt:2*nt+2], key_ok[2*nt:2*nt+2])]
      s = [[UOp.const(0, dtypes.float), UOp.const(0, dtypes.float)] for _ in range(RG)]
      for blk in range(blocks):
        word = cache_kv[0, b, kvh, kb, 4 * blk + t].load()
        bfrag = UOp.stack(*_i8x4(unsign(word & 0x0F0F0F0F)), *_i8x4(unsign((word >> 4) & 0x0F0F0F0F)))
        for mt, afrag in enumerate(afrags):
          c = (UOp.wmma(afrag[blk], bfrag, UOp.const((0,) * 4, dtypes.int32), *DEC_IMMA_ARG) + MAGIC).bitcast(dtypes.float32)
          # (c - 1.5*2**23) * d, never (c + s) - 1.5*2**23: the symbolic pass reassociates a constant out of an add chain
          for k in range(2 * mt, min(2 * mt + 2, RG)):  # C rows g (c0, c1) = row group 2mt, rows g+8 (c2, c3) = 2mt+1
            s[k] = [s[k][i] + (c[2 * (k % 2) + i] - 12582912.0) * (sqs[k][blk] * sks[i][blk]) for i in range(2)]
      for k in range(RG): scores[k] += s[k]
    oks += key_ok
    tiles.append(keys)
  # T4.134: the score mask per row group -- row r (token r % T) sees keys < valid_kv_len - (T-1) + r % T; T == 1: the load mask
  row_oks = [oks] if tokens == 1 else \
    [[key < valid_kv_len + (row % tokens - (tokens - 1)) for key in (key for keys in tiles for key in keys)] for row in rows]
  ps, alphas, row_stores = [], [], []
  for k in range(RG):
    prev_max, prev_sum = row_max.after(step)[k].load(), row_sum.after(step)[k].load()
    tile_max = functools.reduce(lambda a, x: a.maximum(x), (ok.where(sc, SENTINEL) for sc, ok in zip(scores[k], row_oks[k])))
    tile_max = tile_max.maximum(_shfl(tile_max, lane ^ 1))
    tile_max = tile_max.maximum(_shfl(tile_max, lane ^ 2))  # row g's max over the step's keys (the 4 lanes of row g)
    new_max = prev_max.maximum(tile_max)
    alpha = (prev_max - new_max).exp2()
    ps.append([ok.where((sc - new_max).exp2(), 0.0) for sc, ok in zip(scores[k], row_oks[k])])
    new_sum = prev_sum * alpha + functools.reduce(lambda a, x: a + x, ps[k])  # this lane's keys only: the 4 lanes are summed at the end
    alphas.append((_shfl(alpha, 8 * t), _shfl(alpha, 8 * t + 4)))  # O^T's columns are rows 2t, 2t+1 = S rows of lanes 8t, 8t+4
    row_stores += [row_max[k].store(new_max), row_sum[k].store(new_sum)]
  # V: per tile key, this lane's run of row group g (D/16 bytes) and its block's scale, as fp16
  vw = run * g if dpart is None else run * g + run // d_split * dpart  # T4.134: this block's words of the run
  vruns = [[[cache_kv[1, b, kvh, key.valid(ok), vw + w].load() for w in range(run // d_split)] for key, ok in zip(keys, key_ok)]
           for keys, key_ok in zip(tiles, [oks[4*kt:4*kt+4] for kt in range(k_tiles)])]
  svs = [[scale[1, b, kvh, key.valid(ok), (D // 8) * g // 32].load() for key, ok in zip(keys, oks[4*kt:4*kt+4])] for kt, keys in enumerate(tiles)]
  def vval(kt:int, k:int, j:int, high:int) -> UOp:
    # dim row of byte j (high nibble: row g+8) of tile key k, (nibble - 8) * sv in fp16: 0x6400|n is 1024+n exactly
    nib = (vruns[kt][k][j // 4] >> (8 * (j % 4) + 4 * high)) & 0xF
    return ((nib | 0x6400).cast(dtypes.uint16).bitcast(dtypes.float16) - 1032.0) * svs[kt][k]
  pfrags = [[UOp.stack(*(p.cast(dtypes.float16) for p in ps[k][4*kt:4*kt+4])) for kt in range(k_tiles)] for k in range(RG)]
  o_stores = []
  for j in range(JD):
    # A e0..e7 = (rows g, g+8) x (k 2t, 2t+1, 2t+8, 2t+9) in the m16n8k16 order: A[g][2t], A[g][2t+1], A[g+8][2t], A[g+8][2t+1], ...
    vfrags = [UOp.stack(*(vval(kt, 2 * kh + kk, j, high) for kh in range(2) for high in range(2) for kk in range(2))) for kt in range(k_tiles)]
    for k in range(RG):  # T4.134: one V^T fragment, every row group's P
      prev = o_acc[k][j].after(step)
      o = UOp.stack(*(prev[i].load() * alphas[k][i % 2] for i in range(4)))
      for kt in range(k_tiles): o = UOp.wmma(vfrags[kt], pfrags[k][kt], o, *DEC_HMMA_ARG)
      o_stores.append(o_acc[k][j].store(o))
  update = UOp.group(*o_stores, *row_stores).end(step)
  stores = []
  for k in range(RG):
    m_fin, l_fin = row_max.after(update)[k].load(), row_sum.after(update)[k].load()
    l_fin = l_fin + _shfl(l_fin, lane ^ 1)
    l_fin = l_fin + _shfl(l_fin, lane ^ 2)
    # head keeps g (G >= 2: `g < G` pins nothing, eq is not parsed), so the gate survives. T4.134: a last row group of ONE row
    # (G = 9: `g + 8 < 9` pins g = 0) takes eq
    stat_head = heads[k].valid(t.eq(0) & (g.eq(0) if G - 8 * k == 1 else head_oks[k]))
    stores += [stats[b, stat_head, slot, 0].store(m_fin * (1 / LOG2E)), stats[b, stat_head, slot, 1].store(l_fin)]
  for k in range(RG):
    for j, acc in enumerate(o_acc[k]):
      vals = acc.after(update)
      for i in range(4):
        oh = 2 * t + i % 2 + 8 * k if k else 2 * t + i % 2
        dim = (D // 8) * g + 2 * j + i // 2 if dpart is None else (D // 8) * g + 2 * j + i // 2 + 2 * JD * dpart
        stores.append(out[b, kvh * G + oh, slot, dim.valid(oh < G)].store(vals[i].load()))
  ends = (lane, warp, block) if warps > 1 else (lane, block)
  return UOp.group(*stores).end(*ends).sink(arg=KernelInfo(name="flash_decode_mma_nv", opts_to_apply=()))

def _decode_mma_attention(q:Tensor, cache_kv:Tensor, cache_kv_scale:Tensor, valid_kv_len:int|UOp) -> Tensor:
  B, H, D, N = cast(int, cache_kv.shape[1]), cast(int, q.shape[1]), cast(int, q.shape[3]), cast(int, cache_kv.shape[3])
  T = cast(int, q.shape[2])  # T4.134: 1, or a VERIFY run (nv_verify_attention): the kernel's rows are heads x tokens, head-major
  k_tiles = NV_ATTN_DEC_KTILES.value
  assert BLOCK_N % (16 * k_tiles) == 0, f"NV_ATTN_DEC_KTILES={k_tiles} must divide {BLOCK_N // 16}"
  slots = min(NV_ATTN_DEC_SLOTS.value, N // (16 * k_tiles))
  partial = Tensor.empty(B, H * T, slots, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H * T, slots, 2, dtype="float32", device=q.device)
  # Tensor.custom_kernel's call by hand so the int4 cache's parameter is uint32 words over the same uint8 buffer (a kernel argument
  # is a pointer: no view, no copy -- a Tensor-level bitcast of the stored cache would schedule a full copy every step)
  # T4.137: a q that is already a kernel's output buffer (attn_prep_nv's (B,H,T,D)) goes in as a view: .contiguous() after the reshape
  # would copy it (a reshape of a buffer only folds BEFORE the contiguous); a lazy q is materialized in this layout as before
  qv = q.reshape(B, H * T, 1, D)
  srcs = (partial, stats, q.contiguous() if T == 1 else qv if q.uop.op is Ops.AFTER else qv.contiguous(), cache_kv, cache_kv_scale)
  params = [UOp.placeholder_like(s.uop, slot=i) for i, s in enumerate(srcs)]
  params[3] = UOp.placeholder((2, B, cast(int, cache_kv.shape[2]), N, D // 8), dtypes.uint32, slot=3)
  d_split = max(NV_ATTN_DEC_DSPLIT.value, 1) if T > 1 else 1
  if (D // 64) % d_split: d_split = 1  # T4.134: whole V words per part (head_dim 64: no split)
  kernel = _decode_mma_kernel(*params, valid_kv_len=valid_kv_len, max_kv_len=N, warps=max(NV_ATTN_DEC_WARPS.value, 1), k_tiles=k_tiles,
                              tokens=T, d_split=d_split)
  call = kernel.call(*(s.uop for s in srcs))
  partial, stats = Tensor(partial.uop.after(call)), Tensor(stats.uop.after(call))
  # live slots = ceil(valid / KS): _merge_partials counts ceil(valid' / BLOCK_N), so hand it valid' = valid * BLOCK_N / KS
  out = _merge_partials(partial, stats, valid_kv_len * (BLOCK_N // (16 * k_tiles)))
  return out.unsqueeze(2) if T == 1 else out.reshape(B, H, T, D)

# ******** T4.134: speculative VERIFY runs on the decode kernel ********
# VERIFY (2..k+1 query tokens) took the prefill kernel: every query head padded to a 16-row m-tile, 64 slots -- its cost grew with the
# fill (card, 10-30: VERIFY(3) 29.1 ms at 2k, ~60 at 131k vs plain decode 27.0 / 32.0). The decode kernel above takes the run as
# G*T rows per kv head instead (2 m-tiles at G = 6, T = 3), keeping its split-KV slots, so the cost tracks decode's.
NV_ATTN_DEC_MMA_T = ContextVar("NV_ATTN_DEC_MMA_T", 1)  # 1 = runs of 2..MAX_VERIFY_TOKENS on _decode_mma_kernel; 0 = nv_prefill_attention (A/B)
MAX_VERIFY_TOKENS = 4  # G*T <= 32 rows (two m-tiles) at the model's G = 6 .. 8

def nv_verify_attention(q:Tensor, cache_kv:Tensor, cache_kv_scale:Tensor|None, valid_kv_len:int|UOp) -> Tensor|None:
  """q (B,H,T,D) for a concrete 2 <= T <= MAX_VERIFY_TOKENS run whose keys are already stored (valid_kv_len = start_pos + T; token t
  sees keys < start_pos + t + 1, nv_prefill_attention's causal limit) on the decode mma kernel; (B,H,T,D) float32, or None when the
  run, the cache or the device doesn't fit it (the caller takes nv_prefill_attention)."""
  T = q.shape[2]
  if not NV_ATTN_DEC_MMA_T.value or not isinstance(T, int) or not 2 <= T <= MAX_VERIFY_TOKENS: return None
  if not _decode_mma_ok(q, cache_kv, cache_kv_scale): return None
  return _decode_mma_attention(q, cache_kv, cast(Tensor, cache_kv_scale), valid_kv_len)

# ******** T4.107b: prefill chunks ********

@functools.cache
def _prefill_partial_kernel(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *scale:UOp, valid_kv_len:int|UOp, q_start:int|UOp, max_kv_len:int,
                            q_rows:int, lanes:int=WARP_SIZE) -> UOp:
  """The decode kernel over q_rows query rows of a (statically padded) chunk: block = (batch, kv head, row tile) x chunk slot; row r
  of the chunk sees keys <= q_start + r (and < valid_kv_len, the load mask). `lanes` is the warp (32: one lane per head_dim/32
  run, warp_reduce for the dots); lanes=1 is the same graph with one lane owning the whole head and no shuffles -- the
  device-agnostic form the CPU test runs against numpy."""
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start)
  sc = scale[0] if scale else None
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  G, PER, chunks = H // H_KV, D // lanes, cast(int, out.shape[3])
  assert M % q_rows == 0 and H % H_KV == 0 and PER * lanes == D and max_kv_len == N and N % BLOCK_N == 0 and BLOCK_N % KEY_GROUP == 0
  tiles = M // q_rows
  valid_chunks = (valid_kv_len + BLOCK_N - 1) // BLOCK_N
  group_count = min(valid_chunks, chunks) if isinstance(valid_chunks, int) else valid_chunks.minimum(chunks)
  # one flattened grid range (blocks x live chunk slots); the CPU form (lanes=1) runs it as a plain loop -- the CPU renderer has no
  # global dims, and a symbolic slot count inside a second loop range linearised into a >10k-deep chain
  block = UOp.range(B * H_KV * tiles * group_count, 0, AxisType.GLOBAL if lanes > 1 else AxisType.LOOP)
  block_bhr, block_n = block // group_count, block % group_count
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL) if lanes > 1 else None
  bh, tile = block_bhr // tiles, block_bhr % tiles
  b, kv_head = bh // H_KV, bh % H_KV
  rows = tuple(tile * q_rows + i for i in range(q_rows))
  dims = tuple((lane * PER + j) if lane is not None else j for j in range(PER))  # this lane's contiguous run of the head
  q_heads = tuple(kv_head * G + head for head in range(G))
  qs = tuple(tuple(tuple(q[b, q_head, row, d].load().float() * (1 / math.sqrt(D)) for d in dims) for q_head in q_heads) for row in rows)
  acc, row_max, row_sum = _reg((q_rows, G, PER), 0, 0, block), _reg((q_rows, G), 1, SENTINEL, block), _reg((q_rows, G), 2, 0, block)  # per block
  groups_per_chunk = BLOCK_N // KEY_GROUP
  steps = ((valid_chunks + group_count - 1) // group_count) * groups_per_chunk
  offset = UOp.range(steps, 100, AxisType.REDUCE)
  chunk = block_n + (offset // groups_per_chunk) * group_count
  keys = tuple(chunk * BLOCK_N + (offset % groups_per_chunk) * KEY_GROUP + i for i in range(KEY_GROUP))
  loadable = tuple(key < valid_kv_len for key in keys)
  kvals, vvals = (tuple(tuple(kv_value(cache_kv, sc, kv, b, kv_head, cast(UOp, key).valid(ok), cast(UOp, d)) for d in dims)
                        for key, ok in zip(keys, loadable)) for kv in range(2))
  def dot(qv:tuple[UOp, ...], kv:tuple[UOp, ...]) -> UOp:
    s = sum((a * k for a, k in zip(qv, kv)), UOp.const(0, dtypes.float))
    return warp_reduce(s, full_wave=True) if lanes == WARP_SIZE else s
  updates:list[UOp] = []
  for r, row in enumerate(rows):
    valid = tuple(ok & (key <= q_start + row) for key, ok in zip(keys, loadable))  # the causal limit of this query row
    for head in range(G):
      scores = tuple(dot(qs[r][head], key_kvals) for key_kvals in kvals)
      prev_acc, prev_max, prev_sum = acc.after(offset)[r, head], row_max.after(offset)[r, head], row_sum.after(offset)[r, head]
      new_max = functools.reduce(lambda a, vs: a.maximum(vs[0].where(vs[1], UOp.const(SENTINEL, dtypes.float))), zip(valid, scores), prev_max)
      alpha = ((prev_max - new_max) * LOG2E).exp2()
      betas = tuple(ok.where(((score - new_max) * LOG2E).exp2(), UOp.const(0, dtypes.float)) for ok, score in zip(valid, scores))
      new_acc = prev_acc * alpha + sum((UOp.stack(*value) * beta for value, beta in zip(vvals, betas)), acc[r, head].const_like(0))
      updates += [acc[r, head].store(new_acc), row_sum[r, head].store(prev_sum * alpha + sum(betas, UOp.const(0, dtypes.float))),
                  row_max[r, head].store(new_max)]
  update = UOp.group(*updates).end(offset)
  acc, row_max, row_sum = acc.after(update), row_max.after(update), row_sum.after(update)
  stores = [out[b, q_head, row, block_n, d].store(acc[r, head, j]) for r, row in enumerate(rows) for head, q_head in enumerate(q_heads)
            for j, d in enumerate(dims)]
  stores += [stats[b, q_head.valid(lane.eq(0)) if lane is not None else q_head, row, block_n, i].store(x[r, head])
             for r, row in enumerate(rows) for head, q_head in enumerate(q_heads) for i, x in enumerate((row_max, row_sum))]
  ends = (lane, block) if lane is not None else (block,)
  return UOp.group(*stores).end(*ends).sink(arg=KernelInfo(name="flash_prefill_partial_nv", opts_to_apply=()))

# ******** T4.113: prefill chunks on the tensor cores ********

def _shfl_xor(val:UOp, offset:int, maximum:bool=False) -> UOp:
  """Butterfly step over hardware lanes L ^ offset (warp_reduce's CUSTOM, one offset). With the (4, 8) lane split of _wmma fragments
  (hardware L = 4*lane_hi + lane_lo) offsets 1 and 2 combine the four lanes that share a fragment row."""
  other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
  return val.maximum(other) if maximum else val + other

@functools.cache
def _prefill_wmma_kernel(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *scale:UOp, valid_kv_len:int|UOp, q_start:int|UOp, max_kv_len:int,
                         d_split:int=1) -> UOp:
  """_prefill_partial_kernel's contract (same partial/stats layout, same q_start/valid_kv_len masks, same chunk slots) with the
  arithmetic on mma.sync.aligned.m16n8k16: one warp per (batch, QUERY head, 16-row tile, head_dim part) x chunk slot. Per 128-key
  chunk: S = Q.K^T as 16 n-tiles of 8 keys (Q fp16 A fragments, K dequantised straight into fp16 B fragments), the online softmax per
  row in fp32 (chunk max/sum over the 4 lanes of a fragment row by two xor shuffles), then O += P.V with P cast to fp16 as the A
  operand -- the C layout of two adjacent S tiles IS the A layout of a 16-key k-step (nv_gemm.py's fragment map: rows lane_hi/+8,
  cols 2*lane_lo/+1 and +8/+9), so P never leaves the registers. d_split > 1 gives each block a D/d_split slice of the output
  (O accumulators / d_split at the price of computing S d_split times). The k-steps of S and the key-steps of P.V are RANGE loops
  (static register files, ~85 KB of CUDA): fully unrolled, the lowered program could not be pickled back from the compile worker pool
  (C-stack overflow, 2026-09-16). Renders only on the CUDA renderer (no CPU form)."""
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start)
  sc = scale[0] if scale else None
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  G, chunks, DC = H // H_KV, cast(int, out.shape[3]), D // d_split
  assert M % WMMA_M == 0 and H % H_KV == 0 and D % WMMA_K == 0 and DC * d_split == D and DC % WMMA_N == 0 and max_kv_len == N \
    and N % BLOCK_N == 0 and BLOCK_N % WMMA_K == 0
  tiles, k_steps, s_tiles, o_tiles = M // WMMA_M, D // WMMA_K, BLOCK_N // WMMA_N, DC // WMMA_N
  valid_chunks = (valid_kv_len + BLOCK_N - 1) // BLOCK_N
  group_count = min(valid_chunks, chunks) if isinstance(valid_chunks, int) else valid_chunks.minimum(chunks)
  block = UOp.range(B * H * tiles * d_split * group_count, 0, AxisType.GLOBAL)
  block_rest, block_n = block // group_count, block % group_count
  block_rest, dpart = block_rest // d_split, block_rest % d_split
  bh, tile = block_rest // tiles, block_rest % tiles
  b, q_head = bh // H, bh % H
  kv_head, dbase = q_head // G, dpart * DC
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  lane_hi, lane_lo = lane % 8, lane // 8  # nv_gemm._wmma_layout_nv's split: lane_hi = fragment row/col (0-7), lane_lo = k/n sub-index (0-3)
  rows = (tile * WMMA_M + lane_hi, tile * WMMA_M + lane_hi + 8)  # this lane's two fragment rows
  o_acc = tuple(_reg((4,), 10 + n, 0, block) for n in range(o_tiles))  # this block's O fragments, one (4,) register file per n-tile
  row_max, row_sum = _reg((2,), 1, SENTINEL, block), _reg((2,), 2, 0, block)
  steps = (valid_chunks + group_count - 1) // group_count
  offset = UOp.range(steps, 100, AxisType.REDUCE)
  key_base = (block_n + offset * group_count) * BLOCK_N
  def kval(kv:int, key:UOp, d:UOp|int) -> UOp: return kv_value(cache_kv, sc, kv, b, kv_head, key, cast(UOp, d)).cast(dtypes.float16)
  def vval(key:UOp, dim_base:UOp|int) -> UOp:
    """V[key][dim_base + lane_hi] as fp16 -- the B fragment's n index is lane_hi, so the int4 nibble select must not put a div/mod of
    the lane into an index (that re-splits the lane range, nv_gemm._byte's lesson): byte = dim_base/2 + (lane_hi >> 1), nibble by
    (lane_hi & 1); the scale block of dim_base + lane_hi is dim_base's (dim_base is a multiple of 8, blocks are 32 wide)."""
    if cache_kv.dtype == dtypes.uint8:
      assert sc is not None
      blk = cast(int, cache_kv.shape[-1]) * 2 // cast(int, sc.shape[-1])
      byte = cache_kv[1, b, kv_head, key, dim_base // 2 + (lane_hi >> 1)].load().cast(dtypes.uint32)
      nibble = (byte >> ((lane_hi & 1).cast(dtypes.uint32) * 4)) & 0xF
      return ((nibble.cast(dtypes.float) - 8.0) * sc[1, b, kv_head, key, dim_base // blk].load().float()).cast(dtypes.float16)
    if cache_kv.dtype == dtypes.int8:
      assert sc is not None
      blk = cast(int, cache_kv.shape[-1]) // cast(int, sc.shape[-1])
      v = cache_kv[1, b, kv_head, key, dim_base + lane_hi].load().float() * sc[1, b, kv_head, key, dim_base // blk].load().float()
      return v.cast(dtypes.float16)
    return cache_kv[1, b, kv_head, key, dim_base + lane_hi].load().cast(dtypes.float16)
  # S = Q.K^T: A = Q (rows x head_dim), B = K^T (head_dim x keys). The k-steps are a RANGE (not unrolled): the 16 n-tile accumulators
  # stay static register files while the linear program stays short -- fully unrolled (512 mma + ~4k loads per chunk) the lowered
  # PROGRAM UOp was too deep to pickle across the compile worker pool (C-stack overflow in the worker; 2026-09-16 card run).
  s_acc = tuple(_reg((4,), 50 + j, 0, offset) for j in range(s_tiles))  # zeroed per chunk
  ks = UOp.range(k_steps, 101, AxisType.REDUCE)
  k0 = ks * WMMA_K + 2 * lane_lo
  afrag = UOp.stack(*(q[b, q_head, row, k0 + dk].load().cast(dtypes.float16) for row in rows for dk in (0, 1)),
                    *(q[b, q_head, row, k0 + 8 + dk].load().cast(dtypes.float16) for row in rows for dk in (0, 1)))
  s_stores = []
  for j in range(s_tiles):
    key = key_base + j * WMMA_N + lane_hi
    key_ok = key.valid(key < valid_kv_len)
    bfrag = UOp.stack(kval(0, key_ok, k0), kval(0, key_ok, k0 + 1), kval(0, key_ok, k0 + 8), kval(0, key_ok, k0 + 9))
    s_stores.append(s_acc[j].store(UOp.wmma(afrag, bfrag, s_acc[j].after(ks), *WMMA_ARG)))
  s_done = UOp.group(*s_stores).end(ks)
  s_vals = [acc.after(s_done) for acc in s_acc]
  # masked scores (this lane: rows[0] at c0/c1, rows[1] at c2/c3; keys 2*lane_lo, +1 of each 8-key tile), chunk max per row
  scale_log2 = LOG2E / math.sqrt(D)
  scores, oks = [], []
  for j, sv in enumerate(s_vals):
    for i in range(4):
      key, row = key_base + j * WMMA_N + 2 * lane_lo + (i % 2), rows[i // 2]
      ok = (key < valid_kv_len) & (key <= q_start + row)
      oks.append(ok)
      scores.append(ok.where(sv[i].load() * scale_log2, UOp.const(SENTINEL, dtypes.float)))
  prev_max, prev_sum = row_max.after(offset), row_sum.after(offset)
  new_max, alphas, ps = [], [], []
  for r in range(2):
    m = functools.reduce(lambda a, s: a.maximum(s), [s for idx, s in enumerate(scores) if idx % 4 // 2 == r], UOp.const(SENTINEL, dtypes.float))
    m = _shfl_xor(_shfl_xor(m, 1, maximum=True), 2, maximum=True)
    nm = prev_max[r].load().maximum(m)
    new_max.append(nm)
    alphas.append((prev_max[r].load() - nm).exp2())  # scores already carry log2(e): the running max is in the exp2 domain
  for idx, (s, ok) in enumerate(zip(scores, oks)):
    ps.append(ok.where((s - new_max[idx % 4 // 2]).exp2(), UOp.const(0, dtypes.float)))
  lane_sums = [functools.reduce(lambda a, p: a + p, [p for idx, p in enumerate(ps) if idx % 4 // 2 == r], UOp.const(0, dtypes.float))
               for r in range(2)]
  # P parked in a (16 tiles x 4) register file so the key-step RANGE below can index it; O rescaled by alpha once per chunk
  p_reg = UOp.placeholder((s_tiles, 4), dtypes.float, slot=90, addrspace=AddrSpace.REG).after(offset)
  p_ready = UOp.group(*(p_reg[j, i].store(ps[j * 4 + i]) for j in range(s_tiles) for i in range(4)))
  p_reg = p_reg.after(p_ready)
  rescaled = []
  for n in range(o_tiles):
    prev = o_acc[n].after(offset)
    rescaled.append(o_acc[n].store(UOp.stack(*(prev[i].load() * alphas[i // 2] for i in range(4)))))
  rescale = UOp.group(*rescaled)
  # O += P.V over a RANGE of 16-key steps: A = P (rows x 16 keys) from two adjacent S tiles, B = V (16 keys x 8 dims) per output n-tile
  t = UOp.range(BLOCK_N // WMMA_K, 102, AxisType.REDUCE)
  afrag = UOp.stack(*(p_reg[2 * t + tt, i].load().cast(dtypes.float16) for tt in (0, 1) for i in range(4)))
  keys = tuple(key_base + t * WMMA_K + 2 * lane_lo + dk for dk in (0, 1, 8, 9))
  keys_ok = tuple(key.valid(key < valid_kv_len) for key in keys)
  o_stores = []
  for n in range(o_tiles):
    bfrag = UOp.stack(*(vval(key, dbase + n * WMMA_N) for key in keys_ok))
    o_stores.append(o_acc[n].store(UOp.wmma(afrag, bfrag, o_acc[n].after(rescale, t), *WMMA_ARG)))
  o_done = UOp.group(*o_stores).end(t)
  updates = [o_done] + [row_max[r].store(new_max[r]) for r in range(2)] + \
    [row_sum[r].store(prev_sum[r].load() * alphas[r] + lane_sums[r]) for r in range(2)]
  update = UOp.group(*updates).end(offset)
  o_acc, row_max, row_sum = tuple(acc.after(update) for acc in o_acc), row_max.after(update), row_sum.after(update)
  sums = [_shfl_xor(_shfl_xor(row_sum[r].load(), 1), 2) for r in range(2)]  # the four lanes of a row hold partial sums
  stores = []
  for n, acc in enumerate(o_acc):
    d0 = dbase + n * WMMA_N + 2 * lane_lo
    stores += [out[b, q_head, rows[0], block_n, d0].store(acc[0].load()), out[b, q_head, rows[0], block_n, d0 + 1].store(acc[1].load()),
               out[b, q_head, rows[1], block_n, d0].store(acc[2].load()), out[b, q_head, rows[1], block_n, d0 + 1].store(acc[3].load())]
  stats_head = q_head.valid(lane_lo.eq(0) & dpart.eq(0)) if d_split > 1 else q_head.valid(lane_lo.eq(0))
  # T4.130 fix (10-29): row_max is in the exp2 domain (scores carry log2 e) but _merge_partials weighs slots as exp2((m_s - m) * LOG2E),
  # i.e. it expects the natural-domain max the decode/T4.107b kernels store -- stored unconverted (09-16 .. 10-29) every slot below the
  # row max was under-weighted (w^1.44) on any prompt longer than one chunk
  stores += [stats[b, stats_head, rows[r], block_n, 0].store(row_max[r].load() / LOG2E) for r in range(2)]
  stores += [stats[b, stats_head, rows[r], block_n, 1].store(sums[r]) for r in range(2)]
  return UOp.group(*stores).end(lane, block).sink(arg=KernelInfo(name="flash_prefill_wmma_nv", opts_to_apply=()))

# ******** T4.130: prefill chunks, each K/V tile dequantised once for the whole GQA group ********

NV_ATTN_GQA = ContextVar("NV_ATTN_GQA", 1)  # T4.130: inside NV_ATTN_WMMA=1, the shared-tile kernel below (0 = the T4.113 kernel, byte-identical)
NV_ATTN_GQA_WARPS = ContextVar("NV_ATTN_GQA_WARPS", 8)  # T4.130: m-tiles (one warp each) sharing a staged K/V tile; 10-29 card: 8 best
NV_ATTN_GQA_KEYS = ContextVar("NV_ATTN_GQA_KEYS", 32)  # T4.130: keys per staged tile (16 or 32: K + V tiles are 4*keys*head_dim bytes); card knob
# the CUDA renderer ignores the wmma device string; the PYTHON emulator keys its NVIDIA fragment model on 'CUDA' (nv_gemm.IMMA_ARG)
GQA_WMMA_ARG = ((WMMA_N, WMMA_M, WMMA_K), 'CUDA', WARP_SIZE)
SMEM_BYTES = 48 * 1024  # static __shared__ cap

def _kv_halves(cache_kv:UOp, sc:UOp|None, kv:int, b:UOp, h:UOp, key:UOp, d0:UOp, n:int) -> list[UOp]:
  """K/V elements d0 .. d0+n-1 at `key` as fp16 (d0 a multiple of n, n a multiple of 8, the run inside one scale block). The int4
  cache arrives as uint32 WORDS (nv_prefill_attention binds the uint8 buffer to a uint32 parameter: 8 elements per word, element e =
  bits 4e..4e+3 -- the coalescer merges a run into one vector load, it never merges uint8) and is dequantised in fp16 with no
  int->float conversion: bitcast(0x6400 | nibble) = 1024 + nibble exactly, minus 1032 = nibble - 8 exactly, times the fp16 scale = ONE
  rounding of the exact product -- the same bits as kv_value's fp32 (nibble - 8) * scale cast to fp16. The other caches take
  kv_value as is."""
  if cache_kv.dtype != dtypes.uint32: return [kv_value(cache_kv, sc, kv, b, h, key, d0 + i).cast(dtypes.float16) for i in range(n)]
  assert sc is not None
  scale = sc[kv, b, h, key, d0 // (cast(int, cache_kv.shape[-1]) * 8 // cast(int, sc.shape[-1]))].load()
  out = []
  for w in range(n // 8):
    word = cache_kv[kv, b, h, key, d0 // 8 + w].load()
    out += [((((word >> (4 * e)) & 0xF) | 0x6400).cast(dtypes.uint16).bitcast(dtypes.float16) - 1032.0) * scale for e in range(8)]
  return out

@functools.cache
def _prefill_gqa_kernel(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *scale:UOp, valid_kv_len:int|UOp, q_start:int|UOp, max_kv_len:int,
                        warps:int, keys:int) -> UOp:
  """_prefill_wmma_kernel's contract and arithmetic (m16n8k16, fp32 online softmax per row, P kept in registers as the P.V A operand)
  with the K/V work shared across the GQA group. The G query heads of a kv head x the chunk's rows are G*T/16 query m-tiles; a block
  = `warps` of them (one warp each) x a chunk slot. Per `keys`-key tile, every thread of the block dequantises a share of the tile
  into shared memory ONCE (fp16), then each warp runs S = Q.K^T, the softmax and O += P.V for its own m-tile from there: the int4
  loads and dequant are shared by `warps` m-tiles instead of redone per query head (T4.113: 6 heads x 8 row tiles = 48x per kv head).
  Q's A fragments are loaded once per block into registers (16 x 4 at head_dim 256), O is 32 n-tiles x 4 fp32 per lane.

  Shared memory holds the tiles in FRAGMENT order: each 32-lane slot group is one mma operand (K: per 8-key n-tile x 16-dim k-step,
  4 fp16 per lane = b0 b1; V: per 16-key k-step x 8-dim n-tile, the same with keys and dims swapped -- the P.V B operand is V's
  transpose, which the dequant writes directly), so a fragment read is one 8-byte load per lane over 256 contiguous bytes. The slot
  of lane (g, t) inside its group is rotated by a function of the operand (K: t + n-tile + k-step pair mod 4, V: g + n-tile/2 mod 8)
  -- a permutation, so reads stay conflict-free, and it spreads the dequant threads' stores (which all write the same (g, t) slot of
  different groups) over the banks. lane is AxisType.WARP (threadIdx.x, never split), the warp a LOCAL range (_linear_imma_nv's
  layout); the barriers come from add_raw_barrier (stores -> loads of a tile) and add_war_barrier (end of the tile loop)."""
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start)
  sc = scale[0] if scale else None
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  G, chunks, W, KT = H // H_KV, cast(int, out.shape[3]), warps, keys
  KS, OT, NT, TT = D // WMMA_K, D // WMMA_N, KT // WMMA_N, KT // WMMA_K  # k-steps of S, n-tiles of O, S n-tiles / P.V k-steps per tile
  mtiles = G * (M // WMMA_M)
  assert M % WMMA_M == 0 and H % H_KV == 0 and D % 32 == 0 and max_kv_len == N and N % BLOCK_N == 0 and BLOCK_N % KT == 0 \
    and KT % WMMA_K == 0 and mtiles % W == 0 and 4 * KT * D <= SMEM_BYTES
  valid_chunks = (valid_kv_len + BLOCK_N - 1) // BLOCK_N
  group_count = min(valid_chunks, chunks) if isinstance(valid_chunks, int) else valid_chunks.minimum(chunks)
  mgroups = mtiles // W
  block = UOp.range(B * H_KV * group_count * mgroups, 0, AxisType.GLOBAL)
  # the m-groups of one (kv head, slot) are ADJACENT blocks: launched together they stream the same K/V tiles, so the group's
  # 2nd..12th reads (model geometry: 48 m-tiles / 4 warps) hit L2 instead of DRAM (T4.113 ordered the slots fastest)
  block_rest, mgroup = block // mgroups, block % mgroups
  bh, block_n = block_rest // group_count, block_rest % group_count
  b, kv_head = bh // H_KV, bh % H_KV
  warp, lane = UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  g, t = lane // 4, lane % 4  # fragment row/col (0-7), k/n sub-index (0-3): the hardware lane is 4*g + t
  mtile = mgroup * W + warp
  q_head = kv_head * G + mtile % G
  rows = ((mtile // G) * WMMA_M + g, (mtile // G) * WMMA_M + g + 8)  # this lane's two fragment rows
  # Q's A fragments for all k-steps, once per block (registers): a0 a1 = row g, cols 2t/+1; a2 a3 = row g+8; a4..a7 = cols +8/+9
  afrags = [UOp.stack(*(q[b, q_head, rows[r], ks * WMMA_K + 2 * t + c8 + c].load().cast(dtypes.float16)
                        for c8 in (0, 8) for r in (0, 1) for c in (0, 1))) for ks in range(KS)]
  o_acc = tuple(_reg((4,), 10 + n, 0, block) for n in range(OT))
  row_max, row_sum = _reg((2,), 1, SENTINEL, block), _reg((2,), 2, 0, block)
  k_tile = UOp.placeholder((KT * D,), dtypes.float16, slot=100, addrspace=AddrSpace.LOCAL)
  v_tile = UOp.placeholder((KT * D,), dtypes.float16, slot=101, addrspace=AddrSpace.LOCAL)
  steps = (valid_chunks + group_count - 1) // group_count
  offset, sub = UOp.range(steps, 100, AxisType.REDUCE), UOp.range(BLOCK_N // KT, 101, AxisType.REDUCE)
  key0 = (block_n + offset * group_count) * BLOCK_N + sub * KT
  threads, tid = W * WARP_SIZE, warp * WARP_SIZE + lane
  def tasks(total:int):
    for i in range(-(-total // threads)):  # surplus threads redo the last task: same values, same addresses, no gate
      yield (tid + i * threads).minimum(total - 1) if total % threads else tid + i * threads
  # rotations, not XORs: CHECK_OOB's z3 pass (CI) has no XOR with a constant; the bank argument only needs a permutation per group
  def k_slot(j, ks, gg, tt): return (j * KS + ks) * WARP_SIZE + 4 * gg + (tt + j + NT * (ks // 2)) % 4
  def v_slot(kstep, n, gg, tt): return (kstep * OT + n) * WARP_SIZE + 4 * ((gg + n // 2) % 8) + tt
  stores = []
  # K: a task = (key, 32-dim scale block) -> the b0/b1 pairs of 4 lanes x 2 k-steps of its n-tile
  for task in tasks(KT * D // 32):
    kn, blk = task % KT, task // KT
    key = key0 + kn
    vals = _kv_halves(cache_kv, sc, 0, b, kv_head, key.valid(key < valid_kv_len), blk * 32, 32)
    for kso in range(2):
      for tt in range(4):
        slot = k_slot(kn // 8, blk * 2 + kso, kn % 8, tt)
        stores += [k_tile[slot * 4 + i].store(vals[16 * kso + 2 * tt + o]) for i, o in enumerate((0, 1, 8, 9))]
  # V: a task = (the 4 keys 2t/+1/+8/+9 of a 16-key step, 16 dims) -> b0/b1 of 8 lanes x 2 n-tiles, V transposed on the way
  for task in tasks(KT // 4 * D // 16):
    quad, half = task % (KT // 4), task // (KT // 4)
    kstep, tq = quad // 4, quad % 4
    vrows: list[list[UOp]] = []
    for dk in (0, 1, 8, 9):
      key = key0 + kstep * WMMA_K + 2 * tq + dk
      vrows.append(_kv_halves(cache_kv, sc, 1, b, kv_head, key.valid(key < valid_kv_len), half * 16, 16))
    for np_ in range(2):
      for gg in range(8):
        slot = v_slot(kstep, half * 2 + np_, gg, tq)
        stores += [v_tile[slot * 4 + i].store(vrows[i][8 * np_ + gg]) for i in range(4)]
  k_tile, v_tile = k_tile.after(*stores), v_tile.after(*stores)
  # S = Q.K^T over the tile's NT n-tiles (8 keys each), all k-steps unrolled (static register indices for the Q fragments)
  zero = UOp.const((0.0,) * 4, dtypes.float)
  s_vals = [zero] * NT
  for ks in range(KS):
    for j in range(NT):
      bfrag = UOp.stack(*(k_tile[k_slot(j, ks, g, t) * 4 + i].load() for i in range(4)))
      s_vals[j] = UOp.wmma(afrags[ks], bfrag, s_vals[j], *GQA_WMMA_ARG)
  # masked scores (rows[0] at c0/c1, rows[1] at c2/c3; keys 2t, 2t+1 of each 8-key n-tile), online softmax as _prefill_wmma_kernel
  scale_log2 = LOG2E / math.sqrt(D)
  scores, oks = [], []
  for j in range(NT):
    for i in range(4):
      key, row = key0 + j * WMMA_N + 2 * t + (i % 2), rows[i // 2]
      ok = (key < valid_kv_len) & (key <= q_start + row)
      oks.append(ok)
      scores.append(ok.where(s_vals[j][i] * scale_log2, UOp.const(SENTINEL, dtypes.float)))
  prev_max, prev_sum = row_max.after(sub), row_sum.after(sub)
  new_max, alphas = [], []
  for r in range(2):
    m = functools.reduce(lambda a, s: a.maximum(s), [s for idx, s in enumerate(scores) if idx % 4 // 2 == r], UOp.const(SENTINEL, dtypes.float))
    nm = prev_max[r].load().maximum(_shfl_xor(_shfl_xor(m, 1, maximum=True), 2, maximum=True))
    new_max.append(nm)
    alphas.append((prev_max[r].load() - nm).exp2())
  ps = [ok.where((s - new_max[idx % 4 // 2]).exp2(), UOp.const(0, dtypes.float)) for idx, (s, ok) in enumerate(zip(scores, oks))]
  lane_sums = [functools.reduce(lambda a, p: a + p, [p for idx, p in enumerate(ps) if idx % 4 // 2 == r], UOp.const(0, dtypes.float))
               for r in range(2)]
  # O = O * alpha + P.V: P's A fragment for k-step tt = S n-tiles 2tt and 2tt+1 (the C layout of two adjacent tiles)
  pfrags = [UOp.stack(*(ps[(2 * tt + jj) * 4 + i].cast(dtypes.float16) for jj in (0, 1) for i in range(4))) for tt in range(TT)]
  updates = []
  for n in range(OT):
    prev = o_acc[n].after(sub)
    acc = UOp.stack(*(prev[i].load() * alphas[i // 2] for i in range(4)))
    for tt in range(TT):
      acc = UOp.wmma(pfrags[tt], UOp.stack(*(v_tile[v_slot(tt, n, g, t) * 4 + i].load() for i in range(4))), acc, *GQA_WMMA_ARG)
    updates.append(o_acc[n].store(acc))
  updates += [row_max[r].store(new_max[r]) for r in range(2)] + [row_sum[r].store(prev_sum[r].load() * alphas[r] + lane_sums[r]) for r in range(2)]
  update = UOp.group(*updates).end(sub, offset)
  o_acc, row_max, row_sum = tuple(acc.after(update) for acc in o_acc), row_max.after(update), row_sum.after(update)
  sums = [_shfl_xor(_shfl_xor(row_sum[r].load(), 1), 2) for r in range(2)]  # the four lanes of a row hold partial sums
  out_stores = []
  for n, acc in enumerate(o_acc):
    d0 = n * WMMA_N + 2 * t
    out_stores += [out[b, q_head, rows[i // 2], block_n, d0 + i % 2].store(acc[i].load()) for i in range(4)]
  stats_head = q_head.valid(t.eq(0))
  # the running max lives in the exp2 domain (scores carry log2 e); _merge_partials weighs slots as exp2((m_s - m) * LOG2E), the
  # decode/T4.107b kernels' NATURAL-domain max -- so store m / LOG2E (the T4.113 kernel stores its exp2-domain max unconverted,
  # which that merge mis-weighs whenever a row's live slots have different maxes; emulator-checked, see test_llm_nv_attn)
  out_stores += [stats[b, stats_head, rows[r], block_n, i].store(x) for r in range(2) for i, x in enumerate((row_max[r].load() / LOG2E, sums[r]))]
  return UOp.group(*out_stores).end(block, warp, lane).sink(arg=KernelInfo(name="flash_prefill_gqa_nv", opts_to_apply=()))

def _gqa_geometry(H:int, H_KV:int, T_pad:int, D:int) -> dict[str, int]|None:
  """T4.130: the shared-tile kernel's shape from the NV_ATTN_GQA_* knobs (read here, never inside the cached builder), or None (-> the
  T4.113 kernel) when NV_ATTN_GQA is off or head_dim isn't whole 32-wide scale blocks. warps = the largest divisor of the m-tile
  count (G * T_pad/16) <= the knob; keys = the knob rounded down to a power of two in [16, 128] whose K + V tiles fit SMEM_BYTES."""
  if not NV_ATTN_GQA.value or D % 32 or H % H_KV: return None
  mtiles = H // H_KV * T_pad // WMMA_M
  warps = max(w for w in range(1, max(NV_ATTN_GQA_WARPS.value, 1) + 1) if mtiles % w == 0)
  keys = WMMA_K
  while keys * 2 <= min(NV_ATTN_GQA_KEYS.value, BLOCK_N) and 4 * keys * 2 * D <= SMEM_BYTES: keys *= 2
  return {"warps": warps, "keys": keys} if 4 * keys * D <= SMEM_BYTES else None

def nv_prefill_attention(q:Tensor, cache_kv:Tensor, cache_kv_scale:Tensor|None, valid_kv_len:int|UOp, q_start:int|UOp,
                         lanes:int=WARP_SIZE) -> Tensor:
  """q (B,H,T,D) for a prefill chunk whose keys are already stored (valid_kv_len = q_start + T); row r attends keys <= q_start + r.
  T may be a bound Variable: the queries are padded to the chunk's static width (padded rows attend every valid key and are
  sliced off), exactly amd.py's flash_attention. Returns (B,H,T,D) float32. `lanes` is only for the CPU test (lanes=1).
  NV_ATTN_WMMA=1 (T4.113) takes the tensor-core kernel when head_dim allows it (16-row tiles); otherwise the T4.107b kernel.
  T4.130: under NV_ATTN_WMMA=1, NV_ATTN_GQA=1 (default) takes the shared-tile kernel (_gqa_geometry); 0 = the T4.113 kernel."""
  B, H, T, D = q.shape
  d_split = NV_ATTN_WMMA_DSPLIT.value
  wmma = bool(NV_ATTN_WMMA.value) and lanes == WARP_SIZE and cast(int, D) % WMMA_K == 0 and cast(int, D) % (WMMA_N * d_split) == 0
  q_rows = WMMA_M if wmma else NV_ATTN_QROWS.value
  T_pad = -(-q.max_shape[2] // q_rows) * q_rows
  if resolve(T != T_pad): q = q.pad_to((B, H, T_pad, D))
  N = cast(int, cache_kv.shape[3])
  # T4.124b: slots scale down as T_pad widens (64 at T_pad<=32, byte-identical to before; 32/16/8 at 64/128/256) so the partial
  # buffer stays ~48 MiB at every width instead of growing with T_pad -- see nv_prefill_attention's own docstring and the design
  # doc's section 3.3. The kernels need no change: _prefill_partial_kernel/_prefill_wmma_kernel read `chunks` back off out.shape[3].
  chunks = min(PREFILL_CHUNKS * 32 // max(T_pad, 32), N // BLOCK_N)
  partial = Tensor.empty(B, H, T_pad, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, T_pad, chunks, 2, dtype="float32", device=q.device)
  srcs = (partial, stats, q.contiguous(), cache_kv) + ((cache_kv_scale,) if cache_kv_scale is not None else ())
  if wmma and (gqa := _gqa_geometry(cast(int, H), cast(int, cache_kv.shape[2]), T_pad, cast(int, D))) is not None:
    # T4.130: Tensor.custom_kernel's call by hand so the int4 cache's parameter can be uint32 words over the same uint8 buffer (a
    # kernel argument is a pointer: no view, no copy -- a Tensor-level bitcast of the stored cache schedules a full copy)
    params = [UOp.placeholder_like(s.uop, slot=i) for i, s in enumerate(srcs)]
    if cache_kv.dtype == dtypes.uint8: params[3] = UOp.placeholder((*cache_kv.shape[:-1], cast(int, cache_kv.shape[-1]) // 4), dtypes.uint32, slot=3)
    kernel = _prefill_gqa_kernel(*params, valid_kv_len=valid_kv_len, q_start=q_start, max_kv_len=N, **gqa).call(*(s.uop for s in srcs))
    partial, stats = Tensor(partial.uop.after(kernel)), Tensor(stats.uop.after(kernel))
  else:
    if wmma: fxn = functools.partial(_prefill_wmma_kernel, valid_kv_len=valid_kv_len, q_start=q_start, max_kv_len=N, d_split=d_split)
    else: fxn = functools.partial(_prefill_partial_kernel, valid_kv_len=valid_kv_len, q_start=q_start, max_kv_len=N, q_rows=q_rows, lanes=lanes)
    partial, stats = Tensor.custom_kernel(*srcs, fxn=fxn)[:2]
  out = _merge_partials(partial, stats, valid_kv_len)
  return out if resolve(T == T_pad) else out[:, :, :T]
