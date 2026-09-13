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
generic path everywhere) -- validate on the card with extra/nv_attn_validate_real.py before turning it on in a recipe."""
from __future__ import annotations
import functools, math
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import _unbind
from tinygrad.llm.kernels.nv import warp_reduce, _nv_device_ok

NV_CUSTOM_ATTN = ContextVar("NV_CUSTOM_ATTN", 0)
WARP_SIZE, LOG2E = 32, math.log2(math.e)
BLOCK_N, MAX_CHUNKS, KEY_GROUP = 128, 256, 4  # keys per chunk; partial slots per (batch, q head); keys per loop step (one rescale per group)

def nv_decode_attention_ok(device:str|tuple[str, ...]|None, head_dim:int, cache_kv:Tensor) -> bool:
  """The decode kernel covers this layer: NV (sm_70+), a head_dim each lane can own a contiguous run of (32 | head_dim; 64 | head_dim
  for the packed int4 cache so a lane never straddles a byte), a cache length in whole chunks, and a cache dtype it can dequantize."""
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV" or not NV_CUSTOM_ATTN.value: return False
  N = cache_kv.shape[3]
  if not isinstance(N, int) or N % BLOCK_N or head_dim % (64 if cache_kv.dtype == dtypes.uint8 else 32): return False
  if cache_kv.dtype not in (dtypes.uint8, dtypes.int8, dtypes.float16): return False
  return _nv_device_ok(device)

def _reg(shape:tuple[int, ...], slot:int, value:float) -> UOp:
  ret = UOp.placeholder(shape, dtypes.float, slot=slot, addrspace=AddrSpace.REG)
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
  """(B,H,1,D) attention from the live chunk slots' (acc, m, l): out = sum_s acc_s e^(m_s-m) / sum_s l_s e^(m_s-m). Plain tensor ops."""
  chunks = cast(int, partial.shape[2])
  live = (valid_kv_len + BLOCK_N - 1) // BLOCK_N
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  partial, stats = partial[:, :, :live], stats[:, :, :live]
  weights = ((stats[..., 0] - stats[..., 0].max(2, keepdim=True)) * LOG2E).exp2()
  return ((partial * weights.unsqueeze(-1)).sum(2) / (stats[..., 1] * weights).sum(2, keepdim=True)).unsqueeze(2)

def nv_decode_attention(q:Tensor, cache_kv:Tensor, cache_kv_scale:Tensor|None, valid_kv_len:int|UOp) -> Tensor:
  """q (B,H,1,D) against the cache (already written through for this step, so the AFTER chain orders the kernel behind the store);
  valid_kv_len = start_pos+1 stays bound at the graph level. Returns (B,H,1,D) in float32."""
  B, H, D = cast(int, cache_kv.shape[1]), cast(int, q.shape[1]), cast(int, q.shape[3])
  N = cast(int, cache_kv.shape[3])
  chunks = min(MAX_CHUNKS, N // BLOCK_N)
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  fxn = functools.partial(_decode_partial_kernel, valid_kv_len=valid_kv_len, max_kv_len=N)
  srcs = (partial, stats, q.contiguous(), cache_kv) + ((cache_kv_scale,) if cache_kv_scale is not None else ())
  partial, stats = Tensor.custom_kernel(*srcs, fxn=fxn)[:2]
  return _merge_partials(partial, stats, valid_kv_len)
