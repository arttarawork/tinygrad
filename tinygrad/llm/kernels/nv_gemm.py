"""T4.98f: NV tensor-core (WMMA) gemm for Q8_0/Q4_0 prefill chunks (tokens > 4; nv_quant.py's dp4a gemv keeps
tokens <= 4). Applicability check (step 1) found tinygrad's OWN automatic tensor-core optimizer already lowers
this exact fused-dequant kernel (Q8_0 or Q4_0 dequant expression as the B operand of a matmul) onto CUDA's
mma.sync.aligned.m16n8k16 WHEN both matmul operands are literally dtypes.half at the top -- true only for
GatedDeltaNet's own `x = x.half()` projections (model.py:802) today, NOT for the dense attention/FFN path,
whose activations stay dtypes.default_float (fp32 unless DEFAULT_FLOAT=HALF) so the weight's HALF=1 dequant
cast gets promoted back up to fp32 and no fp16/bf16 tensor core matches (the one fp32-input core, tf32, is
gated behind ALLOW_TF32=0 by default). Report has the full evidence trail. This module hand-writes the B
(weight) fragment's dequant instead of relying on that automatic path, for three reasons: (1) it works
regardless of the caller's activation dtype (casts x to fp16 itself, like nv_dense.py's gemv does), (2) it
reuses ONE dequantized B fragment across every M-tile (token) in `token_tile`, which the generic per-kernel
TC application does not structurally guarantee, and (3) it is BEAM-free -- correct by construction rather
than by a search that may or may not find this shape.

Fragment layout: derived empirically from tinygrad's OWN compiler (codegen/opt/tc.py's cuda_81616 TensorCore
definition, m16n8k16 fp16-in/fp32-out; renderer/cstyle.py's CUDARenderer.render_kernel, which emits the raw
`asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" ...)` and treats the a/b/acc UOp operands as opaque
per-lane register files in whatever order the kernel builds them) -- NOT hand-derived from the PTX ISA guide.
Method: built a plain (M=16,K=48,N=8) fp16 matmul (one M-tile, one N-tile, 3 K-tiles -- no block-level tiling
to obscure the per-lane formula), forced `Opt(OptOps.TC, 0, (-1,0,1))`, rendered for a bare `sm_86` target via
`CUDARenderer(DEV.target("NV", arch="sm_86"))` (no device opened), and read the generated index arithmetic
back into these closed-form per-lane formulas (cross-checked a second time against the SAME kernel with a
real Q8_0 dequant as the B operand, which reproduced byte-identical fragment index math around a
`d * (int8)qs` expression in place of the plain buffer load). Both renders are on branch task/T4.98f-nv-gemm's
report; the derivation is NOT copied from NVIDIA's PTX ISA doc, though it agrees with it.

Per-lane fragment map (lane_hi = lane//4 in [0,8), lane_lo = lane%4 in [0,4); k0 = 2*lane_lo, relative to a
16-wide K-tile -- the K dimension covers 2 such tiles per 32-wide Q8_0/Q4_0 block, `half` in {0,1} below):

           A (MxK), row-major            B (KxN), "col" fragment          C/D (MxN) accumulator
  row = lane_hi (+8 for a1/a3)      col = lane_hi (fixed per lane)     row = lane_hi (+8 for c2/c3)
  col = k0 (+1 for the pair),       row = k0 (+1), and k0+8 (+1)       col = 2*lane_lo (+1 for c1/c3)
        and k0+8 (+1)                 for the two b0/b1 pairs

    a0 = A[lane_hi,   k0], A[lane_hi,   k0+1]        b0 = B[k0,   lane_hi], B[k0+1, lane_hi]
    a1 = A[lane_hi+8, k0], A[lane_hi+8, k0+1]        b1 = B[k0+8, lane_hi], B[k0+9, lane_hi]
    a2 = A[lane_hi,   k0+8], A[lane_hi,   k0+9]
    a3 = A[lane_hi+8, k0+8], A[lane_hi+8, k0+9]      c0=C[lane_hi,  2*lane_lo] c1=C[lane_hi,  2*lane_lo+1]
                                                      c2=C[lane_hi+8,2*lane_lo] c3=C[lane_hi+8,2*lane_lo+1]

Note B's "col" (the output/weight-row index, lane_hi) and C's "col" (2*lane_lo) are DIFFERENT lane splits of
the same logical N axis -- that is how the ISA actually wires it (cross-checked twice, see the report); do not
"simplify" this to reuse one variable for both, that would silently transpose/scramble the output.
"""
from __future__ import annotations
import functools
from typing import cast, Callable
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import Linear, Q4_0, Q8_0, Q4_1, IQ4_NL, Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_GROUP_SIZE, QUANT_SIZES, _half
from tinygrad.runtime.autogen import ggml_common as _ggml
from tinygrad.helpers import ContextVar
from tinygrad.llm.kernels.nv import _nv_device_ok

NV_WMMA = ContextVar("NV_WMMA", 1)  # inside NV_CUSTOM_QUANT=1: 0 = keep the generic prefill matmul (A/B lever for the bench)

WMMA_M, WMMA_N, WMMA_K, WARP_SIZE = 16, 8, 16, 32  # mma.sync.aligned.m16n8k16, fp16 in / fp32 accumulate
WMMA_ARG = ((WMMA_N, WMMA_M, WMMA_K), 'NV', WARP_SIZE)  # UOp.wmma's `dims` is (N,M,K), matching TensorCore.dims

@functools.cache
def _nv_wmma_ok(device:str|tuple[str, ...]|None) -> bool:
  # cuda_81616 (m16n8k16, codegen/opt/tc.py) is only in cuda_sm80+ -- sm_75 has just cuda_8168_f16 (m16n8k8, a
  # different fragment layout this file doesn't implement). This fork's one real target is sm_86 (RTX 3090);
  # this is a cheap guard so a lower arch falls through to the generic path instead of a ptxas shape error.
  if isinstance(device, tuple): device = device[0]
  if device is None or not _nv_device_ok(device): return False
  with Context(ALLOW_DEVICE_USAGE=1):
    arch = getattr(getattr(Device[device].renderer, "target", None), "arch", "")  # the renderer carries the target (nv.py does the same)
  return arch.startswith("sm_") and int(arch[3:]) >= 80

def _wmma_layout_nv(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(WMMA_N*output_tiles), 1)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  # The mma fragments are keyed on the HARDWARE warp lane L = threadIdx.x + blockDim.x*threadIdx.y (x fastest), with
  # g = L//4 the A/B row/col index (0-7) and t = L%4 the k/n sub-index (0-3). The lowerer splits this 32-wide LOCAL
  # range by the div/mod structure it sees and hands the MOST significant digit to threadIdx.x: written as lane//4 and
  # lane%4 it rendered as an (8, 4) block with L = lane//4 + 8*(lane%4) -- a consistent scramble (rel err ~1.0 on the
  # 3090, 2026-09-11 probe: a lane bit-permutation fit it exactly). Written as lane%8 / lane//8 it renders as a (4, 8)
  # block, lidx0 = lane//8 (x), lidx1 = lane%8 (y), so L = lane//8 + 4*(lane%8) = 4*lane_hi + lane_lo: the identity the
  # fragments need. test_llm_nv_gemm pins that render. nv_quant/nv_dense never noticed: their lane use is a shuffle
  # reduction, which is permutation-invariant.
  lane_hi, lane_lo = lane % 8, lane // 8  # lane_hi = g: A's/B's shared row/col index (0-7); lane_lo = t: the k/n sub-index (0-3)
  output_bases = tuple((output_block*output_tiles+tile)*WMMA_N for tile in range(output_tiles))
  input_rows = tuple(token_block*token_tile + mt*WMMA_M + lane_hi for mt in range(token_tile // WMMA_M))
  return token_block, output_block, lane, lane_hi, lane_lo, output_bases, input_rows

def _quant_linear_wmma_nv(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, layout, dequant4, name:str) -> UOp:
  x = x.reshape(out.shape[0], in_features)
  token_block, output_block, lane, lane_hi, lane_lo, output_bases, input_rows = layout
  output_tiles, m_tiles = len(output_bases), len(input_rows)
  group_count = in_features // Q8_GROUP_SIZE
  accs = tuple(tuple(UOp.placeholder((4,), dtypes.float32, slot=ot*m_tiles+tile, addrspace=AddrSpace.REG)
                     for tile in range(m_tiles)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(group_count, 3, AxisType.REDUCE)
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    k0 = group*Q8_GROUP_SIZE + half*WMMA_K + 2*lane_lo
    afrags = tuple(UOp.stack(
        x[row,   k0].cast(dtypes.float16), x[row,   k0+1].cast(dtypes.float16),
        x[row+8, k0].cast(dtypes.float16), x[row+8, k0+1].cast(dtypes.float16),
        x[row,   k0+8].cast(dtypes.float16), x[row,   k0+9].cast(dtypes.float16),
        x[row+8, k0+8].cast(dtypes.float16), x[row+8, k0+9].cast(dtypes.float16),
      ) for row in input_rows)
    kk = half*WMMA_K + 2*lane_lo  # offset within the current 32-wide ggml block, 0..31
    for output_tile, output_base in enumerate(output_bases):
      bfrag = UOp.stack(*dequant4(raw, output_base + lane_hi, group_count, group, kk))
      for tile, afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs, output_values in zip(accs, wmma_accs)
                       for acc, value in zip(output_accs, output_values))).end(group)
  stores = []
  for output_base, output_accs in zip(output_bases, accs):
    n0 = output_base + 2*lane_lo
    for row, acc in zip(input_rows, output_accs):
      c0, c1, c2, c3 = (acc.after(update)[i].load() for i in range(4))
      stores += [out[row, n0].store(c0.cast(out.dtype)), out[row, n0+1].store(c1.cast(out.dtype)),
                 out[row+8, n0].store(c2.cast(out.dtype)), out[row+8, n0+1].store(c3.cast(out.dtype))]
  return UOp.group(*stores).end(token_block, output_block, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

# ******** per-format B-fragment decoders (T4.98f: Q8_0/Q4_0; T4.98j: Q4_1, IQ4_NL and the 256-weight K-quants) ********
# Each returns the 4 fp16 weight values a lane's B fragment needs -- elements kk, kk+1, kk+8, kk+9 of the 32-wide K group
# `group` of output row `output_row` -- decoded straight from the packed bytes with the exact math of gguf.py's
# ggml_data_to_tensor. The 4 lanes sharing an output row recompute the same block scales; that redundancy is cheap next
# to the mma and keeps every decoder a few lines of scalar byte arithmetic.

def _byte(raw:UOp, b:UOp) -> UOp:
  # packed byte b as int32. raw is always a BYTE view (amd.py's packed_bytes for the word-view formats): extracting bytes
  # from uint32 words needs b//4 and b%4 on a lane-dependent index, which makes the lowerer re-split the lane range
  # and breaks the (4, 8) mapping _wmma_layout_nv relies on (T4.98j, caught by test_render_pins_the_lane_split)
  assert raw.dtype.itemsize == 1, raw.dtype
  return raw[b].cast(dtypes.int32)

def _half_at(raw:UOp, b:UOp) -> UOp: return _half(_byte(raw, b) | (_byte(raw, b + 1) << 8))  # little-endian f16 at bytes b, b+1

def _select_const(idx:UOp, vals:tuple[int, ...], lo:int=0) -> UOp:
  # vals[idx] as a compile-time balanced where-tree (gguf.py's select_const idiom): the 16-entry kvalues_iq4nl codebook
  if len(vals) == 1: return UOp.const(float(vals[0]), dtypes.float32)
  mid = len(vals) // 2
  return (idx < lo + mid).where(_select_const(idx, vals[:mid], lo), _select_const(idx, vals[mid:], lo + mid))

def _nibble(byte:UOp, high:UOp) -> UOp: return high.where(byte >> 4, byte & 0xF)

def _q8_0_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 8: value = d * qs[i], qs already signed int8 in element order (gguf.py, ref_q8_0)
  base = (output_row * group_count + group) * 34
  d = _half_at(raw, base)
  return tuple((d * raw[base+2+(kk+o)].bitcast(dtypes.int8).float()).cast(dtypes.float16) for o in (0, 1, 8, 9))

def _q4_0_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 2: value = d*(nibble-8); q_to_uint8(.,4)'s order is elements 0-15 = low nibbles of qs[0:16],
  # elements 16-31 = high nibbles of the SAME qs[0:16] (gguf.py, ref_q4_0)
  base = (output_row * group_count + group) * 18
  d = _half_at(raw, base)
  def elem(o):
    k = kk + o
    return (d * (_nibble(_byte(raw, base + 2 + (k % 16)), k >= 16) - 8).float()).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _q4_1_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 3: 20-byte block d, m, qs[16]; value = d*nibble + m, same nibble order as Q4_0 (T4.98i's decode gemv)
  base = (output_row * group_count + group) * 20
  d, m = _half_at(raw, base), _half_at(raw, base + 2)
  def elem(o):
    k = kk + o
    return (d * _nibble(_byte(raw, base + 4 + (k % 16)), k >= 16).float() + m).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _iq4_nl_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 20: Q4_0's 18-byte layout, value = d * kvalues_iq4nl[nibble] (no sub-block scale)
  base = (output_row * group_count + group) * 18
  d = _half_at(raw, base)
  def elem(o):
    k = kk + o
    return (d * _select_const(_nibble(_byte(raw, base + 2 + (k % 16)), k >= 16), _ggml.kvalues_iq4nl)).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _kquant_base(output_row:UOp, group_count:int, group:UOp, block_bytes:int) -> tuple[UOp, UOp]:
  # 256-weight super-blocks: 8 consecutive 32-wide groups share one block; returns (block byte base, sub-block j in 0..7)
  return (output_row * (group_count // 8) + group // 8) * block_bytes, group % 8

def _q4q5_k_scales(raw:UOp, base:UOp, j:UOp) -> tuple[UOp, UOp]:
  # Q4_K/Q5_K: d, dmin then 12 scale bytes: 6-bit sc[0-3] (bytes 0-3), 6-bit mn[0-3] (bytes 4-7), and for j>=4 the low
  # nibbles/high nibbles of bytes 8-11 topped up with the two spare high bits of bytes j-4 / j (gguf.py ggml_type 12/13)
  d, dmin = _half_at(raw, base), _half_at(raw, base + 2)
  def s(i): return _byte(raw, base + 4 + i)
  low = j < 4
  sc = low.where(s(j) & 63, (s(j + 4) & 0xF) | ((s(j - 4) >> 6) << 4))
  mn = low.where(s(j + 4) & 63, (s(j + 4) >> 4) | ((s(j) >> 6) << 4))
  return d * sc.float(), dmin * mn.float()

def _q4_k_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 12 (144 bytes): sub-block j's 32 elements are the low (j even) / high (j odd) nibbles of qs[(j//2)*32 : +32]
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[Q4_K])
  dsc, dmn = _q4q5_k_scales(raw, base, j)
  high = (j % 2).ne(0)
  def elem(o): return (dsc * _nibble(_byte(raw, base + 16 + (j // 2) * 32 + (kk + o)), high).float() - dmn).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _q5_k_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 13 (176 bytes): Q4_K's layout with 32 qh bytes at 16 (bit j of qh[i] is element i's 5th bit) and qs at 48
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[Q5_K])
  dsc, dmn = _q4q5_k_scales(raw, base, j)
  high = (j % 2).ne(0)
  def elem(o):
    i = kk + o
    q = _nibble(_byte(raw, base + 48 + (j // 2) * 32 + i), high) | (((_byte(raw, base + 16 + i) >> j) & 1) << 4)
    return (dsc * q.float() - dmn).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _q6_k_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 14 (210 bytes): ql[128] (two 64-byte halves, low nibbles then high nibbles), qh[64] (two 32-byte halves, 2-bit
  # pairs), 16 int8 scales per 16 elements, d at 208; value = d * ((ql | qh<<4) - 32) * scale (gguf.py ggml_type 14)
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[Q6_K])
  d = _half_at(raw, base + 208)
  h, q4 = j // 4, j % 4            # 128-element half, 32-element quarter within it (qh pair index)
  high = (q4 // 2).ne(0)           # elements 64-127 of the half are the high nibbles
  def elem(o):
    i = kk + o
    lo = _nibble(_byte(raw, base + h * 64 + (q4 % 2) * 32 + i), high)
    hi = (_byte(raw, base + 128 + h * 32 + i) >> (2 * q4)) & 3
    scale = ((_byte(raw, base + 192 + j * 2 + i // 16) ^ 0x80) - 0x80).float()  # int8 from the unsigned byte
    return (d * ((lo | (hi << 4)) - 32).float() * scale).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _iq4_xs_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 23 (136 bytes): d, 16-bit scales_h, 4 bytes scales_l, qs[128]; sub-block j = 16 bytes (low nibbles = elements
  # 0-15, high = 16-31), scale = ((scales_l[j] | scales_h[j]<<4) - 32), value = d * scale * kvalues_iq4nl[q]
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[IQ4_XS])
  d = _half_at(raw, base)
  scales_h = _byte(raw, base + 2) | (_byte(raw, base + 3) << 8)
  sl = _nibble(_byte(raw, base + 4 + j // 2), (j % 2).ne(0))
  scale = ((sl | (((scales_h >> (2 * j)) & 3) << 4)) - 32).float()
  def elem(o):
    k = kk + o
    return (d * scale * _select_const(_nibble(_byte(raw, base + 8 + j * 16 + (k % 16)), k >= 16), _ggml.kvalues_iq4nl)).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

# ggml_type -> (fragment decoder, K alignment the decoder needs, kernel name)
WMMA_FORMATS: dict[int, tuple[Callable, int, str]] = {
  Q8_0: (_q8_0_dequant4, 32, "linear_q8_0_wmma_nv"), Q4_0: (_q4_0_dequant4, 32, "linear_q4_0_wmma_nv"),
  Q4_1: (_q4_1_dequant4, 32, "linear_q4_1_wmma_nv"), IQ4_NL: (_iq4_nl_dequant4, 32, "linear_iq4_nl_wmma_nv"),
  Q4_K: (_q4_k_dequant4, 256, "linear_q4_k_wmma_nv"), Q5_K: (_q5_k_dequant4, 256, "linear_q5_k_wmma_nv"),
  Q6_K: (_q6_k_dequant4, 256, "linear_q6_k_wmma_nv"), IQ4_XS: (_iq4_xs_dequant4, 256, "linear_iq4_xs_wmma_nv")}

def _token_tile_for(tokens:int) -> int: return 64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else WMMA_M
def _output_tiles_for(out_features:int) -> int: return 4 if out_features % (WMMA_N*4) == 0 else 1

@functools.cache
def _wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int) -> UOp:
  dequant4, _, name = WMMA_FORMATS[ggml_type]
  layout = _wmma_layout_nv(out, out_features, _token_tile_for(cast(int, out.shape[0])), _output_tiles_for(out_features))
  return _quant_linear_wmma_nv(out, raw, x, out_features, in_features, layout, dequant4, name)

def nv_wmma_linear(layer:Linear, x:Tensor) -> Tensor|None:
  """WMMA gemm for every format in WMMA_FORMATS at prefill (tokens > 4, multiple of 16 -- or symbolic, padded to
  `x.max_shape` the way amd.py's q8_linear pads for a symbolic chunk size). Returns None when it doesn't cover this
  call (unsupported format, an out_features/in_features that isn't tile/block-aligned, tokens<=4, or an arch below
  sm_80) so Linear.__call__ (amd.py) falls back to the generic dequant+matmul path or nv_quant.py's decode gemv."""
  fmt = WMMA_FORMATS.get(layer.ggml_type) if layer.ggml_type is not None else None
  if not NV_WMMA.value or fmt is None: return None
  out_features, in_features = layer.out_features, layer.in_features
  if out_features % WMMA_N != 0 or in_features % fmt[1] != 0: return None
  numel = x.numel()
  symbolic = not isinstance(numel, int)
  tokens = x.max_shape[-2] if symbolic else numel // in_features
  if tokens <= 4 or tokens % WMMA_M != 0: return None  # tokens<=4: nv_quant.py's decode gemv covers it
  if not _nv_wmma_ok(x.device): return None
  x_pad = x.pad_to(x.max_shape) if symbolic else x  # concrete shape from here on
  xh = x_pad.cast(dtypes.float16).contiguous().reshape(tokens, in_features)
  raw = (layer.packed_bytes if layer.packed_bytes is not None else layer.weight).uop.buf_uop  # always the byte view, see _byte
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xh.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  kernel = _wmma_kernel(*params, out_features=out_features, in_features=in_features, ggml_type=layer.ggml_type).call(*all_srcs)
  result = Tensor(out.after(kernel)).reshape(*x_pad.shape[:-1], out_features)
  if symbolic: result = result.shrink(tuple((0, s) for s in (*x.shape[:-1], out_features)))
  return result if layer.bias is None else result + layer.bias
