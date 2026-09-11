"""T4.98c: NV port of amd.py's quantized-linear decode kernel (Q4_K/Q5_K/Q6_K/IQ4_XS) plus formats amd.py doesn't
have (Q8_0, Q4_0; T4.98i adds Q4_1, IQ4_NL) -- CUDA-C (sm_70+) intrinsics mirroring amd.py's RDNA3 ones (__dp4a for sudot4,
__byte_perm for v_perm_b32, __shfl_xor_sync for ds_swizzle via kernels/nv.py's warp_reduce). Same q8-activation
design as amd.py: q8_quantize turns the activation into int8 words + a per-32-group float scale, group_dot does
one 32-lane warp per (token, output, chunk-of-32-groups) with a dp4a dot per group, warp_reduce sums the lanes.
Decode-only (tokens<=4): the WMMA gemm path (amd.py's _q5_linear_f16_wmma_kernel/_iq4_linear_f16_wmma_kernel
equivalent) is T4.98f. Dispatch: amd.py's Linear.__call__ calls nv_q8_linear when NV_CUSTOM_QUANT=1 (default 0)
on an NV device with a supported ggml_type; nv_q8_linear returns None for anything it doesn't cover (big batches,
symbolic token counts, unsupported formats) and the caller falls back to the generic dequant+matmul path."""
from __future__ import annotations
import functools, math
from typing import Callable, cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.amd import (
  Linear, Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, Q4_0, Q4_1, IQ4_NL, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q4_WORDS, Q5_WORDS, Q6_BYTES, IQ4_WORDS,
  _half, _q5_scales, _iq4_scales, IQ_GRID_SIZES)
from tinygrad.llm.kernels.nv import warp_reduce, _nv_device_ok, iq_grid

NV_CUSTOM_QUANT = ContextVar("NV_CUSTOM_QUANT", 0)  # 0 = never take this path, even on NV (default: byte-identical everywhere)
NV_QUANT_TYPES = (Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, Q4_0, Q4_1, IQ4_NL)

def nv_quant_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV" or not NV_CUSTOM_QUANT.value: return False
  return _nv_device_ok(device)  # CUDA C renderer + sm_70+: dp4a needs sm_61+, full-mask __shfl_*_sync needs Volta+

# ******** CUDA intrinsics mirroring amd.py's AMDGCN ones ********

def _nv_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  # __dp4a(int,int,int) is signed x signed x accumulate -- matches amd's sudot4(true,a,true,b,c) exactly (T4.98c)
  return UOp(Ops.CUSTOMI, src=(a.int(), b.int(), c), arg=("__dp4a({}, {}, {})", dtypes.int32))

def _nv_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  # AMD v_perm_b32(S0,S1,sel): sel is 4 BYTE-lanes (0-15 each), idx 0-3 <- S1 (2nd arg), 4-7 <- S0 (1st arg) --
  # hardware-verified against test/amd/hw/test_vop3.py::TestPermMore.test_v_perm_b32_select_high_bytes (S0=
  # 0x03020100, S1=0x07060504, sel picks indices 4,5,6,7 -> result reconstructs S0 byte-for-byte). CUDA's
  # __byte_perm(x,y,s) (prmt.b32) is the opposite: idx 0-3 <- x (1st arg), 4-7 <- y (2nd arg) -- so a/b swap
  # below. It also only reads 4 NIBBLES packed into sel's low 16 bits, not 4 bytes spread across all 32 like
  # AMD's -- repack the byte-lanes amd.py's callers build (_iq4_bytes: each lane already a 0-15 value, high
  # nibble zero) into that layout. T4.98c
  a32, b32, sel = a.cast(dtypes.uint32), b.cast(dtypes.uint32), selectors.cast(dtypes.uint32)
  nv_sel = sum(((sel >> (4*i)) & (0xf << (4*i)) for i in range(4)), UOp.const(0, dtypes.uint32))
  return UOp(Ops.CUSTOMI, src=(b32, a32, nv_sel), arg=("__byte_perm({}, {}, {})", dtypes.uint32))

def _nv_load(ptr:UOp, lanes:int|None=None) -> UOp:
  assert ptr.op is Ops.INDEX
  if lanes is None:
    # amd's __builtin_nontemporal_load is a cache-bypass hint with no direct CUDA equivalent wired here; __ldg is
    # a possible perf lever to try once this runs on real hardware (T4.98c) -- a plain load is correct either way
    return ptr.load()
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(dtype=ptr.dtype)

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  # verbatim amd.py's _iq4_bytes with _nv_byte_perm in place of _amd_byte_perm (T4.98c)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _nv_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _nv_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _nv_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

# ******** q8-activation quant linear (verbatim amd.py structure; warp_reduce is the only device-specific piece) ********

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str) -> UOp:
  chunks = (group_count+31)//32
  token_output_chunk, lane = UOp.range(out.shape[0]*out_features*chunks, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, output, chunk = token_output_chunk // (out_features*chunks), (token_output_chunk//chunks) % out_features, token_output_chunk % chunks
  group = lane+chunk*32
  value = group_dot(token, output, group) if group_count % 32 == 0 else \
    (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token_output_chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group//groups, token_group%groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane*4+i].float() for i in range(4))
  word = sum(((v/group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8)
              for i,v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  stores = (q[token, group, lane.valid(lane < 8)].store(word), scale[token, group.valid(lane.eq(0))].store(group_scale))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize_nv", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  q, scale = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:2]
  return q, scale

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, grid:UOp|None=None, *, out_features:int, in_features:int, ggml_type:int) -> UOp:
  # grid: the IQ3 codebook (nv.py iq_grid) as a 5th kernel input for IQ3_XXS/IQ3_S (T4.98k), None for every other format
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    xwords = _nv_load(xq[token, group, 0], 8)
    if ggml_type in (Q4_0, Q8_0, Q4_1, IQ4_NL):
      # 32-weight blocks: one ggml block per activation group (no block/subgroup split). raw is a uint8 view
      # (set_quantized): none of these block widths (34/18/20/18 bytes) are 4-byte aligned, so no uint32 word
      # view here (T4.98c; T4.98i adds Q4_1/IQ4_NL)
      block_bytes = {Q8_0: 34, Q4_1: 20}.get(ggml_type, 18)  # Q4_0 and IQ4_NL are both 18
      base = (output * group_count + group) * block_bytes
      d = _half(raw[base].cast(dtypes.uint16) | (raw[base+1].cast(dtypes.uint16) << 8))
      qs_base = base + (4 if ggml_type == Q4_1 else 2)  # Q4_1 has an extra f16 m before qs (Q4_0/Q8_0/IQ4_NL don't)
      if ggml_type == Q8_0:
        # gguf.py ggml_type==8: value = d * qs[i], qs already signed int8 in element order -- no unpack needed
        dot = UOp.const(0, dtypes.int32)
        for word_idx in range(8):
          qbytes = _nv_load(raw[qs_base + word_idx*4], 4)
          word = sum((qbytes[i].cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
          dot = _nv_dp4a(word, xwords[word_idx], dot)
        return dot.float() * d * xd[token, group]
      if ggml_type == IQ4_NL:
        # gguf.py ggml_type==20: value = d * kvalues_iq4nl[q] -- Q4_0's exact byte layout and nibble order
        # (elements 0-15 = low nibbles of qs[0:16], 16-31 = high nibbles of the SAME qs[0:16]) but each nibble
        # indexes the 16-entry kvalues_iq4nl LUT instead of q-8, and there's no sub-block scale to apply.
        # _iq4_bytes (IQ4_XS's LUT trick below) turns a raw packed byte-word straight into a dp4a-ready word of
        # signed LUT values, so no separate nibble-unpack step is needed here (T4.98i)
        loads = [_nv_load(raw[qs_base + g*4], 4) for g in range(4)]
        dot = UOp.const(0, dtypes.int32)
        for word_idx in range(8):
          word = sum((loads[word_idx % 4][i].cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
          dot = _nv_dp4a(_iq4_bytes(word, 4*(word_idx//4)), xwords[word_idx], dot)
        return dot.float() * xd[token, group] * d
      # Q4_0 (gguf.py ggml_type==2, value = d*(q-8)) and Q4_1 (ggml_type==3, value = d*q+m) share the same nibble
      # unpack: q_to_uint8(.,4)'s order is elements 0-15 = low nibbles of qs[0:16], elements 16-31 = high nibbles
      # of the SAME qs[0:16]. Q4_1's m is a constant added to every weight in the block, so over the dot product
      # it becomes m * sum(activation) -- same dp4a-of-ones qsum trick that folds Q4_0's -8 in, just added with
      # coefficient m instead of subtracted with coefficient 8*d (T4.98i)
      if ggml_type == Q4_1: m = _half(raw[base+2].cast(dtypes.uint16) | (raw[base+3].cast(dtypes.uint16) << 8))
      loads = [_nv_load(raw[qs_base + g*4], 4) for g in range(4)]
      dot, qsum = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        shifted = (loads[word_idx % 4] >> 4) if word_idx >= 4 else loads[word_idx % 4]
        word = sum(((shifted[i] & 0x0f).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
        dot, qsum = _nv_dp4a(word, xwords[word_idx], dot), _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      if ggml_type == Q4_1: return (dot.float()*d + qsum.float()*m) * xd[token, group]
      return (dot.float() - 8*qsum.float()) * d * xd[token, group]
    # 256-weight superblocks: verbatim amd.py's group_dot (_nv_dp4a/_nv_load/_iq4_bytes in place of the _amd_ ones)
    block, subgroup = group // 8, group % 8
    if ggml_type in (Q4_K, Q5_K):
      base = (output * in_features//GGML_BLOCK_SIZE + block) * (Q4_WORDS if ggml_type == Q4_K else Q5_WORDS)
      qs_base, dot, qsum = base + (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8, UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (raw[qs_base+word_idx] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K: word |= ((raw[base+4+word_idx] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        dot, qsum = _nv_dp4a(word, xwords[word_idx], dot), _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      return (dot.float()*d*scale - qsum.float()*dmin*minimum) * xd[token, group]
    if ggml_type == IQ4_XS:
      base = (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        packed = _nv_load(raw[base + 2 + subgroup*4 + word_idx%4])
        dot = _nv_dp4a(_iq4_bytes(packed, 4*(word_idx//4)), xwords[word_idx], dot)
      d, scale = _iq4_scales(raw, base, subgroup)
      return dot.float() * xd[token, group] * d * scale
    base = (output*in_features//GGML_BLOCK_SIZE+block)*Q6_BYTES
    dots = [UOp.const(0, dtypes.int32)] * 2
    for word_idx in range(8):
      pos, within = subgroup*32 + word_idx*4, (subgroup*32 + word_idx*4)%128
      low = _nv_load(raw[base + (pos//128)*64 + within%64], 4) >> ((within//64)*4).cast(dtypes.uint8)
      high = _nv_load(raw[base + 128 + (pos//128)*32 + within%32], 4) >> ((within//32)*2).cast(dtypes.uint8)
      quant = ((low & 15) | ((high & 3) << 4)).bitcast(dtypes.int8) - 32
      word = sum((quant[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
      dots[word_idx//4] = _nv_dp4a(word, xwords[word_idx], dots[word_idx//4])
    scales = [raw[base + 192 + subgroup*2+i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    dbits = raw[base+208].cast(dtypes.uint16) | (raw[base+209].cast(dtypes.uint16) << 8)
    return (dots[0].float()*scales[0] + dots[1].float()*scales[1]) * xd[token, group] * _half(dbits)
  names = {Q4_K: "linear_q4_k_nv", Q5_K: "linear_q5_k_nv", IQ4_XS: "linear_iq4_xs_nv", Q6_K: "linear_q6_nv",
           Q8_0: "linear_q8_0_nv", Q4_0: "linear_q4_0_nv", Q4_1: "linear_q4_1_nv", IQ4_NL: "linear_iq4_nl_nv"}
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type])

def nv_q8_linear(layer:Linear, x:Tensor) -> Tensor|None:
  """amd.py's q8_linear decode branch, ported to NV, restricted to small batches (tokens<=4): the WMMA gemm path
  amd.py takes for bigger/aligned batches isn't ported here (T4.98f). Returns None when it doesn't cover this
  call (big batch, symbolic token count, unsupported format) so Linear.__call__ (amd.py) can fall back."""
  if layer.ggml_type not in NV_QUANT_TYPES: return None
  numel = x.numel()
  if not isinstance(numel, int): return None  # symbolic token count: no padded-chunk path ported yet (T4.98f)
  tokens = numel // layer.in_features
  if tokens > 4: return None  # decode-only for now
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  def run(fxn:Callable[..., UOp], out:UOp, *srcs:UOp) -> Tensor:
    all_srcs = (out,)+srcs
    params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
    kernel = fxn(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
    result = Tensor(out.after(kernel))
    if len(result.shape) == 3: result = result.sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
  xq, xd = q8_quantize(x, tokens, in_features)
  decode = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
  out = Tensor.empty(tokens, out_features, (in_features+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  grids = (iq_grid(layer.ggml_type, cast(str, x.device)).uop,) if layer.ggml_type in IQ_GRID_SIZES else ()  # T4.98k: IQ3 codebook input
  return run(decode, out, raw, xq.uop, xd.uop, *grids)
