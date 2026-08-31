import functools, io, pathlib, re, struct
from typing import Any, Callable

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.nn.state import TensorIO

# ggml packs each iq grid entry as N bytes (N=4 for uint32 grids, N=8 for uint64 grids) in a single word. See ggml-common.h.
@functools.lru_cache(None)
def _ggml_iq_grid(device: str, grid: tuple[int, ...], grid_shape: tuple[int, int]) -> Tensor:
  values = [float((w >> (8*i)) & 0xFF) for w in grid for i in range(grid_shape[1])]
  return Tensor(values, dtype=dtypes.float32, device=device).reshape(grid_shape)

# native types {ggml_type: dtype}
_GGML_NATIVE = {0: dtypes.float32, 1: dtypes.float16, 24: dtypes.int8, 25: dtypes.int16,
                26: dtypes.int32, 27: dtypes.int64, 28: dtypes.float64, 30: dtypes.bfloat16}

# quant types {ggml_type: (number of elements, number of bytes)}
_GGML_QUANT = {2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24), 8:(32,34),
               10:(256,84), 11:(256,110), 12:(256,144), 13:(256,176), 14:(256,210),
               16:(256,66), 17:(256,74), 18:(256,98), 19:(256,50), 20:(32,18), 21:(256,110), 22:(256,82), 23:(256,136),
               29:(256,56), 39:(32,17), 41:(128,18)}

def _ggml_nbytes(n: int, ggml_type: int) -> int:
  """Exact on-disk byte length of a GGUF tensor's data blob (mirrors the slicing ggml_data_to_tensor
  does internally), computed from its element count and type alone -- lets the loader stage just this
  tensor's byte range instead of the whole file."""
  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None: return dtype.itemsize * n
  if (nb := _GGML_QUANT.get(ggml_type)) is not None: return (n // nb[0]) * nb[1]
  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 24),
  int16 (id: 25), int32 (id: 26), int64 (id: 27), float64 (id: 28), bfloat16 (id: 30)
  Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q5_0 (id: 6),
  Q5_1 (id: 7), Q8_0 (id: 8), Q2_K (id: 10), Q3_K (id: 11), Q4_K (id: 12), Q5_K (id: 13),
  Q6_K (id: 14), IQ2_XXS (id: 16), IQ2_XS (id: 17), IQ3_XXS (id: 18), IQ1_S (id: 19),
  IQ4_NL (id: 20), IQ3_S (id: 21), IQ2_S (id: 22), IQ4_XS (id: 23), IQ1_M (id: 29), MXFP4 (id: 39), Q1_0 (id: 41)
  """
  # https://github.com/ggerganov/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/include/ggml.h#L356

  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None:
    return t[:dtype.itemsize * n].contiguous().bitcast(dtype)

  def q_to_uint8(t: Tensor, b: int) -> Tensor:
    # TODO: rewrite with arange?
    shift_tensor, bitmask = Tensor.const(tuple(2**(i*b) for i in range(8//b)), t.dtype), 0xff >> (8 - b)
    return t.unsqueeze(-1).div(shift_tensor, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

  def select_const(idx: Tensor, vals, lo: int = 0):
    # Selects vals[idx] via a compile-time balanced binary decision tree (nested .where() over
    # Python float constants) instead of a Tensor-indexed gather. A gather reads its table through
    # a buffer-accessing REDUCE (Tensor.__getitem__'s one-hot-sum, mixin/op.py) that rangeify's
    # buffer_in_reduce refuses to fuse into a consuming reduce -- same class of bug as ggml_type==39's
    # MXFP4 LUT below, for genuinely arbitrary (non-bit-decomposable) codebooks: IQ3_XXS's 256-entry
    # grid and IQ4_XS's 16-entry kvalues_iq4nl (T4.22). Bit-exact vs the gather form (verified
    # exhaustively over the full code space for both tables); cheap for small tables like these --
    # not a general gather replacement.
    if len(vals) == 1: return vals[0]
    mid = lo + len(vals)//2
    return (idx < mid).where(select_const(idx, vals[:len(vals)//2], lo), select_const(idx, vals[len(vals)//2:], mid))

  if (nelements_nbytes := _GGML_QUANT.get(ggml_type)) is not None:
    from tinygrad.runtime.autogen import ggml_common as _ggml
    blocks = t[:(n//nelements_nbytes[0])*nelements_nbytes[1]].reshape((-1, nelements_nbytes[1])).contiguous()
    if ggml_type == 2: return (q_to_uint8(blocks[:,2:], 4).bitcast(dtypes.int8) - 8) * blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
    if ggml_type == 3:
      d, m = (blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [ 0, 2 ])
      return q_to_uint8(blocks[:,4:], 4).bitcast(dtypes.int8) * d + m
    if ggml_type in (6, 7):
      d = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
      qh_off = 2 if ggml_type == 6 else 4
      qh = q_to_uint8(blocks[:,qh_off:qh_off+4], 1).reshape((-1, 8, 4)).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      q = q_to_uint8(blocks[:,qh_off+4:], 4).bitcast(dtypes.int8) + qh * 16
      return q * d + (blocks[:,2:4].bitcast(dtypes.float16).cast(dtypes.float32) if ggml_type == 7 else -16 * d)
    if ggml_type == 8: return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].bitcast(dtypes.int8)
    # Q2_K: 256 elements per 84-byte block (scales:16, qs:64, d:2, dmin:2)
    if ggml_type == 10:
      d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [80, 82])
      sc = blocks[:, :16]
      q = q_to_uint8(blocks[:, 16:80].reshape((-1, 2, 32)), 2).reshape((-1, 16, 16))
      return (d * sc.bitwise_and(0xF).unsqueeze(-1) * q - dmin * sc.rshift(4).unsqueeze(-1)).flatten(-2)
    # Q3_K: 256 elements per 110-byte block (hmask:32, qs:64, scales:12, d:2)
    if ggml_type == 11:
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1)
      sc = q_to_uint8(blocks[:,96:104], 4).bitwise_or(q_to_uint8(blocks[:,104:108], 2).lshift(4)).bitcast(dtypes.int8) - 32
      q = q_to_uint8(blocks[:,32:96].reshape((-1, 2, 32)), 2).reshape((-1, 16, 16))
      qh = q_to_uint8(blocks[:,:32], 1).reshape((-1, 16, 16))
      return (d * sc.unsqueeze(-1) * (q.bitcast(dtypes.int8) - qh.bitwise_xor(1).lshift(2).bitcast(dtypes.int8))).flatten(-2)
     # Q4_K: 256 elements per 144-byte block (d:2, dmin:2, scales:12, qs:128)
     # Q5_K: 256 elements per 176-byte block (d:2, dmin:2, scales:12, qh:32, qs:128)
    if ggml_type in (12, 13):
      d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
      s = blocks[:,4:16]  # 12 bytes: 6-bit scales[0-3], 6-bit mins[0-3], high bits[4-7]
      sc = s[:,0:4].bitwise_and(63).cat(s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4)), dim=-1)
      mn = s[:,4:8].bitwise_and(63).cat(s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4)), dim=-1)
      # stage the per-(block,sub-block) d*scale / dmin*min products as their own small contiguous tensor (8
      # values/block, reused by 32 weights each) instead of leaving them fused into the consuming matmul: without
      # this, rangeify doesn't hoist the fp16->fp32 d/dmin unpack out of the reduce's innermost per-weight loop
      # (confirmed via the generated METAL source - unlike sc/mn, which correctly do get hoisted to the sub-block
      # loop), so it gets redone once per output weight (32x/block) instead of once per block. Realizing here trades
      # that for one cheap extra kernel (reads ~same 12+4 scale bytes/block, writes 2 floats/sub-block) that only
      # runs once at weight-load time, since the result is reused unchanged across every subsequent decode token.
      # Q4_K 4096x4096 gemv on METAL: ~410us -> ~205us kernel time (bit-exact vs the fused form; T4.2).
      dsc, dminmn = (d * sc.unsqueeze(-1)).contiguous(), (dmin * mn.unsqueeze(-1)).contiguous()
      qs_off = 48 if ggml_type == 13 else 16
      q = Tensor.stack((qs:=blocks[:,qs_off:qs_off+128].reshape(-1,4,32)).bitwise_and(0xF), qs.rshift(4), dim=2).reshape(-1,8,32)
      if ggml_type == 13: q = q + q_to_uint8(blocks[:,16:48], 1).reshape(-1, 8, 32) * 16
      return (dsc * q - dminmn).flatten(-2)
    if ggml_type == 14:
      xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
      scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32)
      return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
    if ggml_type == 18:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scale_words = blocks[:, 66:98].bitcast(dtypes.uint32)
      db = d * (scale_words.rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.5
      sign_idx = scale_words.unsqueeze(-1).rshift(Tensor.const((0, 7, 14, 21), dtypes.uint32)).bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
      # even_signs[i] == i | (parity_bit(i) << 7): ggml stores only 7 sign bits/group and derives the
      # 8th from parity (even_signs was a real Tensor+gather -- a buffer-reading REDUCE inside the
      # dequant expression, same class of bug as T4.13's MXFP4 LUT; see the ggml_type==39 comment
      # above). Computed via the standard XOR-fold SWAR parity trick instead: bit-exact for all 128
      # values (verified), pure ALU, no buffer.
      px = sign_idx ^ sign_idx.rshift(4)
      px = px ^ px.rshift(2)
      px = px ^ px.rshift(1)
      even_signs_sign_idx = sign_idx.bitwise_or(px.bitwise_and(1).lshift(7))
      signs = (q_to_uint8(even_signs_sign_idx.reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      # iq3xxs_grid is a genuine 256-entry codebook (not bit-decomposable like the parity above) --
      # select_const dodges the same buffer_in_reduce issue without a formula (see its docstring).
      # flat (256*4,), matching _ggml_iq_grid's own unpack order
      grid_vals = tuple(float((w >> (8*i)) & 0xFF) for w in _ggml.iq3xxs_grid for i in range(4))
      code = blocks[:, 2:66].cast(dtypes.int32)
      # (-1,64,4): degroup the 4 interleaved sub-values per code
      grid4 = Tensor.stack(*[select_const(code, grid_vals[c::4]) for c in range(4)], dim=-1)
      grid = grid4.reshape((-1, 8, 4, 8))
      return (db * grid * signs).flatten(-3)
    # IQ2_XXS: 256 elements per 66-byte block (d:2, qs:64). 8 groups of 32: 4 grid bytes + packed signs/scale.
    if ggml_type == 16:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      qs_u32 = blocks[:, 2:].bitcast(dtypes.uint32).reshape((-1, 8, 2))
      db = d * (qs_u32[:, :, 1].rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.25
      sign_idx = qs_u32[:, :, 1].unsqueeze(-1).rshift(Tensor.const((0, 7, 14, 21), dtypes.uint32))
      sign_idx = sign_idx.bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
      even_signs = Tensor([i | (0x80 if i.bit_count() % 2 else 0) for i in range(128)], dtype=dtypes.uint8, device=t.device)
      signs = (q_to_uint8(even_signs[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq2xxs_grid, (256, 8))[blocks[:, 2:].reshape((-1, 8, 8))[:, :, :4]].reshape((-1, 8, 4, 8))
      return (db * grid * signs).flatten(-3)
    # IQ2_XS: 256 elements per 74-byte block (d:2, qs:64 as uint16, scales:8)
    if ggml_type == 17:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      db = d * (q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 4).reshape((-1, 16)).cast(dtypes.float32) + 0.5).reshape((-1, 16, 1, 1)) * 0.25
      qs = blocks[:, 2:66].bitcast(dtypes.uint16)
      sign_idx = qs.rshift(9).cast(dtypes.int32)
      even_signs = Tensor([i | (0x80 if i.bit_count() % 2 else 0) for i in range(128)], dtype=dtypes.uint8, device=t.device)
      signs = (q_to_uint8(even_signs[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 16, 2, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq2xs_grid, (512, 8))[qs.bitwise_and(511)].reshape((-1, 16, 2, 8))
      return (db * grid * signs).flatten(-3)
    # IQ1_S: 256 elements per 50-byte block (d:2, qs:32, qh:16). grid bytes are int8 {-1,0,1}.
    if ggml_type == 19:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      qh = blocks[:, 34:50].bitcast(dtypes.uint16)
      dl = d * (qh.rshift(12).bitwise_and(7).cast(dtypes.float32) * 2 + 1).reshape((-1, 8, 1, 1))
      delta = (qh.bitwise_and(0x8000) == 0).where(0.125, -0.125).reshape((-1, 8, 1, 1))
      qh_hi = qh.unsqueeze(-1).rshift(Tensor.const((0, 3, 6, 9), dtypes.uint16)).bitwise_and(7).lshift(8)
      q = blocks[:, 2:34].cast(dtypes.uint16) + qh_hi.reshape((-1, 32))
      grid = _ggml_iq_grid(t.device, _ggml.iq1s_grid, (2048, 8))[q].reshape((-1, 8, 4, 8))
      grid = (grid > 127).where(grid - 256, grid)
      return (dl * (grid + delta)).flatten(-3)
    if ggml_type == 20:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)
      return d * Tensor(list(_ggml.kvalues_iq4nl), dtype=dtypes.float32, device=t.device)[q_to_uint8(blocks[:, 2:], 4)]
    if ggml_type == 21:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scales = (1 + 2 * q_to_uint8(blocks[:, 106:110].reshape((-1, 4, 1)), 4).reshape((-1, 8))).cast(dtypes.float32).reshape((-1, 8, 1, 1))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 1).reshape((-1, 64)).cast(dtypes.uint16)
      signs = (q_to_uint8(blocks[:, 74:106].reshape((-1, 32, 1)), 1).reshape((-1, 256)) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      q = blocks[:, 2:66].cast(dtypes.uint16) + qh.lshift(8)
      return (d * scales * _ggml_iq_grid(t.device, _ggml.iq3s_grid, (512, 4))[q].reshape((-1, 8, 4, 8)) * signs).flatten(-3)
    if ggml_type == 22:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      db = d * (q_to_uint8(blocks[:, 74:82].reshape((-1, 8, 1)), 4).reshape((-1, 16)).cast(dtypes.float32) + 0.5).reshape((-1, 16, 1, 1)) * 0.25
      signs = (q_to_uint8(blocks[:, 34:66].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 16, 2, 8))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 2).reshape((-1, 32)).cast(dtypes.uint16)
      q = blocks[:, 2:34].cast(dtypes.uint16) + qh.lshift(8)
      return (db * _ggml_iq_grid(t.device, _ggml.iq2s_grid, (1024, 8))[q].reshape((-1, 16, 2, 8)) * signs).flatten(-3)
    if ggml_type == 23:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1))
      scale_shifts = Tensor.const((0, 2, 4, 6, 8, 10, 12, 14), dtypes.uint16)
      scales_l = Tensor.stack((sl:=blocks[:, 4:8]).bitwise_and(0xF), sl.rshift(4), dim=2).reshape((-1, 8))
      scales_h = blocks[:, 2:4].bitcast(dtypes.uint16).unsqueeze(-1).rshift(scale_shifts).bitwise_and(0x03).reshape((-1, 8)).cast(dtypes.uint8)
      scales = (scales_l.bitwise_or(scales_h.lshift(4)).bitcast(dtypes.int8) - 32).cast(dtypes.float32).reshape((-1, 8, 1))
      q = (qs:=blocks[:, 8:].reshape((-1, 8, 16))).bitwise_and(0xF).cat(qs.rshift(4), dim=2)
      # kvalues_iq4nl is a genuine 16-entry codebook (not bit-decomposable like MXFP4's E2M1 below) --
      # select_const dodges the same buffer_in_reduce issue without a formula (see its docstring).
      return (d * scales * select_const(q, _ggml.kvalues_iq4nl)).flatten(-2)
    # IQ1_M: 256 elements per 56-byte block (qs:32, qh:16, scales:8). f16 scale packed in high nibbles.
    if ggml_type == 29:
      sc16 = blocks[:, 48:56].bitcast(dtypes.uint16)
      d = sc16.bitwise_and(0xF000).rshift(Tensor.const((12, 8, 4, 0), dtypes.uint16))
      d = d[:, 0:1].bitwise_or(d[:, 1:2]).bitwise_or(d[:, 2:3]).bitwise_or(d[:, 3:4])
      d = d.bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1, 1))
      scales = sc16.unsqueeze(-1).rshift(Tensor.const((0, 3, 6, 9), dtypes.uint16)).bitwise_and(7)
      dl = d * (scales.cast(dtypes.float32) * 2 + 1).reshape((-1, 8, 2, 1, 1))
      qh_n = Tensor.stack(blocks[:, 32:48].bitwise_and(0x0F), blocks[:, 32:48].rshift(4), dim=-1).reshape((-1, 32))
      q = blocks[:, :32].cast(dtypes.uint16) + qh_n.bitwise_and(7).cast(dtypes.uint16).lshift(8)
      delta = (qh_n.bitwise_and(0x08) == 0).where(0.125, -0.125).reshape((-1, 8, 2, 2, 1))
      grid = _ggml_iq_grid(t.device, _ggml.iq1s_grid, (2048, 8))[q].reshape((-1, 8, 2, 2, 8))
      grid = (grid > 127).where(grid - 256, grid)
      return (dl * (grid + delta)).flatten(-4)
    if ggml_type == 39:
      # e8m0 block scale and the E2M1 4-bit value are computed via ALU bit-ops instead of Tensor-indexed
      # LUT gathers (the original form: `lut_tensor[codes]`). A gather reads a real buffer through a
      # REDUCE, and rangeify's remove_bufferize refuses to fuse a bufferize point whose expression
      # contains a buffer-reading REDUCE (schedule/rangeify.py's buffer_in_reduce check) into ANY
      # consumer that itself indexes the result -- e.g. MoE's `weight[sel]` expert gather. That forced
      # the ENTIRE dequantized tensor (every expert, not just the k selected) to materialize every
      # decode step instead of just the gathered rows (T4.13; T4.2 hit an analogous but cheaper issue
      # with Q4_K's scale unpack). Both replacements below are bit-exact vs the LUT form (verified for
      # all 256 e8m0 byte values and all 16 four-bit codes) and touch no buffer but `blocks` itself.
      e = blocks[:, 0].cast(dtypes.uint32)
      d = (e == 0).where(0x00200000, (e == 1).where(0x00400000, (e - 1) * 0x00800000)).bitcast(dtypes.float32).unsqueeze(-1)
      codes = q_to_uint8(blocks[:, 1:17], 4).cast(dtypes.int32)
      mant, exp, sign = codes.bitwise_and(1), codes.rshift(1).bitwise_and(3), codes.rshift(3).bitwise_and(1)
      mag = (exp == 0).where(mant, (2 + mant).lshift((exp - 1).maximum(0)))
      fp4_val = mag.cast(dtypes.float32) * (1 - 2 * sign.cast(dtypes.float32))
      return (fp4_val * d).flatten(-2)[:n]
    if ggml_type == 41:
      d = blocks[:,:2].bitcast(dtypes.float16)
      bits = q_to_uint8(blocks[:,2:], 1).reshape(-1, 8, 16).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      return d * (bits * 2 - 1)
  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

def _read_unpack(fmt: str, n: int, r:io.BufferedIOBase): return struct.unpack(fmt, r.read(n))[0]
def read_str(r:io.BufferedIOBase): return str(r.read(read_uint64(r)), "utf-8")
def read_arr(r:io.BufferedIOBase):
  item_reader, n = readers[read_int32(r)], read_uint64(r)
  return [item_reader(r) for _ in range(n)]

readers: dict[int, Callable[[io.BufferedIOBase], Any]] = { 8: read_str, 9: read_arr,
  **{ t: functools.partial(_read_unpack, "<"+f, nb) for t,f,nb in \
    [ (0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8) ] } }
read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

_HEADER_CHUNK = 16 * 1024 * 1024  # generous prefix for magic+KV+tensor-infos (real headers measured ~6-8MB)
_STAGE_BATCH = 64 * 1024 * 1024   # merge adjacent tensors into one disk->device copy up to this size, to
                                   # amortize per-copy dispatch overhead over the many small tensors (norms,
                                   # biases, small embeddings) real GGUFs have alongside the big matmul weights

def _parse_header(header: Tensor) -> tuple[dict, list, int]:
  r = io.BufferedReader(TensorIO(header), 1_000_000)
  magic, version, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")

  kv_data = {}
  for _ in range(n_kv):
    k, typ = read_str(r), read_int32(r)
    kv_data[k] = readers[typ](r)

  t_infos = [ (read_str(r), tuple(read_uint64(r) for _ in range(read_uint32(r))), read_int32(r), read_uint64(r)) for _ in range(n_tensors) ]
  return kv_data, t_infos, r.tell()

def _gguf_parse(tensor: Tensor, device_map:str|dict[int|str,str]|None=None) -> tuple[dict, dict[str, Tensor]]:
  # [T1.9] Only a small header prefix gets realized to parse KV metadata + tensor infos -- not the whole
  # (multi-GB) file. Tensor DATA is staged in bounded batches (_STAGE_BATCH) below instead of one whole-file
  # blob: no single allocation is ever bigger than one batch, which matters under memory pressure (a
  # fragmented allocator can satisfy many small requests where one big contiguous one fails -- see commit
  # message) and keeps per-tensor dequant fusing into its eventual consumer exactly as before.
  size = tensor.shape[0]
  chunk = min(size, _HEADER_CHUNK)
  while True:
    header = tensor[:chunk].to(None).realize()
    try:
      kv_data, t_infos, pos = _parse_header(header)
      if chunk < size and pos >= chunk: raise ValueError("header may have been truncated by the chunk boundary")
    except (struct.error, IndexError, ValueError):
      if chunk >= size: raise
      chunk = min(size, chunk * 2)
      continue
    break

  alignment = kv_data.get("general.alignment", 32)
  data_start = round_up(pos, alignment)

  # [T4.21] When a device_map is given, stage each tensor's raw (still-quantized) blob straight onto ITS
  # placed device instead of always Device.DEFAULT, so the dequant expression built on top of it runs where
  # the bytes already live -- no COPY sits above the dequant. Left as the pre-T4.21 Device.DEFAULT-only
  # behavior (dev_for=None) when device_map is None, so single-device loads are untouched.
  #
  # Transformer.realize_placement() (llm/model.py) force-realizes params moved off Device.DEFAULT, once,
  # right after load, to stop the JIT from recapturing a dequant+COPY every token (see its docstring). Before
  # this fix, "moved off Device.DEFAULT" meant the COPY wrapped the *finished* dequant (built on the load
  # device, at fp16), so realizing it force-materialized the full fp16 size on the LOAD device before the
  # bytes ever reached the target -- fine for T3.3's small moved share, a multi-GB swap spike for a big-model
  # range split. Building the dequant ON the target device from the start (this function) means that same
  # forced realize now just computes it locally there -- cheap, and exactly the residency T1.9 intended.
  #
  # Mirrors the block/experts placement Transformer.__init__ computes from the same parse_device_map() output
  # (model.py) -- by GGUF tensor name here, since this runs before any Transformer/nn.Module exists to read
  # placement back off of. Imported locally: model.py imports gguf_load at module scope, so a top-level
  # import here would cycle; by the time anything actually calls gguf_load, model.py is fully initialized.
  dev_for: Callable[[str], str]|None = None
  if device_map is not None:
    from tinygrad.llm.model import parse_device_map
    arch = kv_data.get('general.architecture')
    # num_blocks must match from_gguf's own formula exactly (real blocks, MTP nextn excluded). A later
    # multi-part-split file may carry partial/no KV (only split 0 is guaranteed the full tensor listing) --
    # fall back to Device.DEFAULT staging for it rather than KeyError on a rare, untested combination.
    num_blocks = kv_data[f'{arch}.block_count'] - kv_data.get(f'{arch}.nextn_predict_layers', 0) if arch is not None \
      and f'{arch}.block_count' in kv_data else None
    if num_blocks is not None:
      dmap, experts_dev = parse_device_map(device_map, num_blocks)
      def _dev_for(name: str) -> str:
        if not name.startswith("blk."): return dmap[0] if name == "token_embd.weight" else dmap[-1]
        if experts_dev is not None and any(f".ffn_{w}_exps." in name for w in ("gate", "up", "down")): return experts_dev
        idx = int(name.split(".", 2)[1])
        # the MTP nextn block beyond num_blocks: unreferenced when model.py's MTP=0 (default) -- dropped
        # after load with a warning -- but consumed into Transformer.mtp_head when MTP=1 (T4.63). Either
        # way it lands on the LAST block's device: model.py's MTP=1 path places mtp_head there too (see
        # from_gguf), so this clamp already stages its blob exactly where that load will look for it.
        return dmap[idx] if idx < len(dmap) else dmap[-1]
      dev_for = _dev_for

  # sort by on-disk offset and greedily merge adjacent tensors (bounded by _STAGE_BATCH, and -- when
  # device_map is active -- sharing a target device) into one disk->device copy each, instead of one copy
  # per tensor. Each tensor's dequant graph still starts from its own VIEW of the (already-realized) batch,
  # so per-tensor fusion is unaffected -- this only changes how many COPY ops the loader issues and which
  # device(s) they land on, not the fusion itself.
  infos = sorted(((name, dims, typ, off, _ggml_nbytes(prod(dims), typ)) for name, dims, typ, off in t_infos), key=lambda x: x[3])

  def flush(batch: list[tuple[str, tuple[int, ...], int, int, int]]) -> dict[str, Tensor]:
    lo, hi = batch[0][3], batch[-1][3] + batch[-1][4]
    # realize the raw (still-quantized) batch bytes right away, same as the pre-T1.9 whole-file realize
    # did -- this is required regardless (DISK can't run the dequant ALU ops) and keeps the scheduler
    # from tangling hundreds of small COPYs into the same schedule as the per-tensor dequant graphs below.
    staged = tensor[data_start + lo:data_start + hi].to(dev_for(batch[0][0]) if dev_for else None).realize()
    return {name: ggml_data_to_tensor(staged[off - lo:off - lo + nbytes], prod(dims), typ).reshape(*reversed(dims))
            for name, dims, typ, off, nbytes in batch}

  state_dict: dict[str, Tensor] = {}
  batch: list[tuple[str, tuple[int, ...], int, int, int]] = []
  for info in infos:
    if batch and (info[3] + info[4] - batch[0][3] > _STAGE_BATCH or (dev_for is not None and dev_for(info[0]) != dev_for(batch[0][0]))):
      state_dict.update(flush(batch))
      batch = []
    batch.append(info)
  if batch: state_dict.update(flush(batch))
  return kv_data, state_dict

def _gguf_split_paths(path: pathlib.Path, kv: dict) -> list[pathlib.Path]:
  if (total := kv.get('split.count', 1)) <= 1: return [path]
  if kv.get('split.no', 0) != 0: raise ValueError(f"multi-part GGUF must be loaded from the first split, got split.no={kv['split.no']}")
  if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(path))): raise ValueError(f"first split path must end with -00001-of-NNNNN.gguf: {path}")
  return [pathlib.Path(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total+1)]

def gguf_load(fn: Tensor|str|pathlib.Path, device_map:str|dict[int|str,str]|None=None) -> tuple[dict, dict[str, Tensor]]:
  """
  Loads a .gguf file, returning the `kv_data` and `state_dict`. Multi-part splits are auto-merged when loaded by path.

  ```python
  import pathlib
  from tinygrad import Device, Tensor
  from tinygrad.llm.gguf import gguf_load

  gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(Device.DEFAULT)
  kv_data, state_dict = gguf_load(gguf_tensor)
  ```

  NOTE: The provided tensor must be on a device that supports execution.

  `device_map` (same syntax as `tinygrad.llm.model.Transformer`/`parse_device_map`) places each tensor's raw
  blob directly on its mapped device instead of `Device.DEFAULT` -- see the T4.21 comment in `_gguf_parse`.
  """
  kv, sd = _gguf_parse(fn if isinstance(fn, Tensor) else Tensor(pathlib.Path(fn)), device_map)
  if kv.get('split.count', 1) <= 1: return kv, sd
  if isinstance(fn, Tensor): raise ValueError("multi-part GGUF requires a path argument (got Tensor)")
  for pp in _gguf_split_paths(pathlib.Path(fn), kv)[1:]: sd.update(_gguf_parse(Tensor(pp), device_map)[1])
  return kv, sd
