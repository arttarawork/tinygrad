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
import functools, math
from typing import cast, Callable
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import (Linear, Q4_0, Q8_0, Q4_1, IQ4_NL, Q4_K, Q5_K, Q6_K, IQ4_XS, Q3_K, IQ3_XXS, IQ3_S,
                                      IQ_GRID_SIZES, Q8_GROUP_SIZE, QUANT_SIZES, Q5_WORDS, _half)
from tinygrad.runtime.autogen import ggml_common as _ggml
from tinygrad.helpers import ContextVar
from tinygrad.llm.kernels.nv import _nv_device_ok, iq_grid
from tinygrad.llm.kernels.nv_quant import q8_quantize

NV_WMMA = ContextVar("NV_WMMA", 1)  # inside NV_CUSTOM_QUANT=1: 0 = keep the generic prefill matmul (A/B lever for the bench)
NV_WMMA_TOKEN_TILE = ContextVar("NV_WMMA_TOKEN_TILE", 64)  # T4.124c: card knob -- more/fewer token-tile warps per block (PB.3)
# T4.126: Q4_0 prefill on the int8 tensor cores (q8 activations, decode's quantizer) instead of the fp16 WMMA kernel. On by
# default since the 10-28 card run (128-token chunk 233.5 -> 156.2 ms; gates 11/1152 and 12/1152 at near-ties vs the fp16 path);
# NV_IMMA=0 restores the fp16 kernel. T4.128: the same switch takes Q4_1 (ffn_down of 8 blocks) and Q5_K (GDN ssm_out) onto the
# same kernel (card gates pending); NV_IMMA=0 = all three on fp16. The shape knobs are card-sweepable and, like
# NV_WMMA_TOKEN_TILE, read only in nv_wmma_linear (never in the cached builder):
NV_IMMA = ContextVar("NV_IMMA", 1)
NV_IMMA_TOKEN_TILE = ContextVar("NV_IMMA_TOKEN_TILE", 64)  # tokens per block (largest power of two <= knob dividing the chunk)
NV_IMMA_WARPS = ContextVar("NV_IMMA_WARPS", 4)            # warps per block, sharing one staged activation tile
NV_IMMA_OUT_TILES = ContextVar("NV_IMMA_OUT_TILES", 2)    # 8-output n-tiles per warp (each A fragment feeds this many mmas)
NV_IMMA_STAGE = ContextVar("NV_IMMA_STAGE", 4)            # 32-groups per shared-memory stage (barrier pair); 0 = no staging (10-28: 4 best)
# T4.128: 1 = double-buffered stage tile, the next stage's activation fills overlapping this stage's mmas (one barrier per stage);
# 0 = the T4.126 kernel, byte-identical. Card A/B pending; with NV_IMMA_STAGE=0 (no tile) it has nothing to do
NV_IMMA_PIPE = ContextVar("NV_IMMA_PIPE", 0)
# T4.131: occupancy. The T4.126 kernel takes 196 registers (ptxas, nvcc 12.8 sm_86) = 2 blocks of 4 warps per SM; min blocks per SM
# through __launch_bounds__(threads, n): 3 -> 158 registers, 0 spills; 4 -> 128 registers, 144 B spills. 0 = today's render
NV_IMMA_MINBLOCKS = ContextVar("NV_IMMA_MINBLOCKS", 3)  # 10-29 card: 0 140.6, 3 138.2, 4 178.9 (spills) ms/chunk
# T4.131: 1 = the next stage's weight words are loaded into registers one stage ahead (their global-load latency, exposed at the
# top of every step today, hides behind this stage's mmas); 0 = today's kernel, byte-identical. Costs one register per word read
NV_IMMA_WPREFETCH = ContextVar("NV_IMMA_WPREFETCH", 0)
# T4.131: split-K for the low-N / high-K shapes. The 5120-output layers get 160 blocks for the 3090's 82 SMs (one wave at 2 blocks
# per SM) and each walks every stage step serially (136 for ffn_down). splitk blocks per output tile each walk steps/splitk
# stages into their own fp32 partial, (splitk, tokens, out) summed by the caller (NOT bit-identical to one pass: fp32 order).
# 1 = off (today's kernel); 0 = auto (the smallest divisor of the step count that gives ~2 waves of resident blocks, else the
# largest <= 8; the wide layers keep one pass); n = that split wherever it divides the step count
NV_IMMA_SPLITK = ContextVar("NV_IMMA_SPLITK", 1)

WMMA_M, WMMA_N, WMMA_K, WARP_SIZE = 16, 8, 16, 32  # mma.sync.aligned.m16n8k16, fp16 in / fp32 accumulate
WMMA_ARG = ((WMMA_N, WMMA_M, WMMA_K), 'NV', WARP_SIZE)  # UOp.wmma's `dims` is (N,M,K), matching TensorCore.dims
# T4.126: mma.sync.aligned.m16n8k32 s8 x s8 -> s32 (one 32-wide Q4_0/q8 group per K). The CUDA renderer ignores the device
# string; the PYTHON emulator keys its NVIDIA fragment model on "CUDA", which lets the IMMA kernel run there unchanged.
IMMA_K = 32
IMMA_ARG = ((WMMA_N, WMMA_M, IMMA_K), 'CUDA', WARP_SIZE)

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

# ******** T4.98k: the three 3-bit formats (Q3_K, IQ3_XXS, IQ3_S) -- 256-weight blocks, same _kquant_base sub-block
# j = group % 8. Two (IQ3_XXS/IQ3_S) index a codebook (nv.py iq_grid, an extra kernel input bound like raw/x) by an
# 8-bit "code" covering 4 weights (code index e//4, e = 32*j+i the block-relative element, i = kk+o the group-
# relative one); the grid word's byte e%4 is the magnitude. Both e//4 and e%4 divide/mod the lane-dependent `i` by
# 4 -- the exact hazard _wmma_layout_nv's docstring warns about (kk = half*WMMA_K + 2*lane_lo; lane_lo is a LOCAL-
# range UOp) -- so _div4_mod4 below builds them from `kk` alone via comparisons/subtraction (never // or % on an
# expression containing lane_lo), and the per-o `o//4`/`o%4` terms are plain Python ints (o is a compile-time
# constant, unrolled by the elem() loop), safe to add in directly. Ditto for i//8/i%8 (IQ3_XXS's sign sub-group,
# IQ3_S's sign byte): built from a (kk>=16) comparison plus the compile-time o//8, never dividing kk/i themselves.
# All three decoders' exact per-element formulas were cross-checked against gguf.py's ggml_data_to_tensor (numpy
# port of its q_to_uint8/reshape pipeline, 1000 random 256-weight blocks + all-0x00/all-0xFF edge blocks, every one
# of the 256 elements/block) in the T4.98k report's verify_iq3.py -- notably IQ3_S's scale-nibble and qh byte/bit
# split turned out BYTE-major (nibble(j) = j%2==0 ? low : high nibble of byte(106+j//2); qh bit = c%8 of byte
# (66+c//8)), not the field-major grouping a literal reading of ggml's C reference suggests once gguf.py's own
# q_to_uint8 is fed a (-1, N, 1)-reshaped input (Q3_K and IQ3_XXS never reshape that way, so they stay field-major
# and match the naive reading).

def _div4_mod4(kk:UOp) -> tuple[UOp, UOp]:
  # (kk//4, kk%4) via a comparison/subtraction where-tree on kk itself -- kk only ever takes the 8 values
  # {0,2,4,6,16,18,20,22} in this kernel (half*WMMA_K + 2*lane_lo, half in {0,1}, lane_lo in 0..3), so this never
  # applies // or % to an expression containing lane_lo (comparisons/where/subtraction don't split the LOCAL range)
  hi = kk >= 16
  base = kk - hi.where(16, 0)  # 0, 2, 4 or 6, regardless of half
  return hi.where(4, 0) + (base >= 4).where(1, 0), (base.eq(2) | base.eq(6)).where(2, 0)

def _grid_byte(grid:UOp, idx:UOp, byte_in_word:UOp) -> UOp:
  # byte `byte_in_word` (0..3) of the codebook's uint32 word at `idx` -- grid is the IQ3 codebook (nv.py iq_grid),
  # an extra kernel input bound the same way `raw` is (see nv_wmma_linear/_wmma_kernel)
  return (grid[idx].cast(dtypes.int32) >> (8 * byte_in_word)) & 0xFF

def _q3_k_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 11 (110 bytes): hmask[32] at 0, qs[64] at 32, scales[12] at 96, d (f16) at 108 (gguf.py ggml_type==11,
  # field-major -- no (-1,N,1) reshape trick here). e = 32*j+i (i = kk+o) needs the scale index s = 2*j + (i>=16):
  # s is only ever compared/added below, and the real ggml s//4 (nibble-pair shift) / s%4 (byte offset) split is
  # rewritten purely from j (a REDUCE sub-block index, not lane-dependent -- safe to divide) plus the (i>=16) flag
  # folded in through .where() branches, never a // or % on an expression containing kk.
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[Q3_K])
  d = _half_at(raw, base + 108)
  p, half_j, j_par = j % 4, j // 2, j % 2
  def elem(o):
    i = kk + o
    q2 = (_byte(raw, base + 32 + 32 * (j // 4) + i) >> (2 * p)) & 3
    qh = (_byte(raw, base + i) >> j) & 1
    hi = i >= 16
    # scale index s = 2*j+hi (0..15) is only ever used through this byte offset -- 2*p+hi is tightly bounded [0,7]
    # on BOTH .where() branches (unlike addressing directly by the wide-ranging s, which reads up to base+96+15,
    # one byte past the 110-byte block on the last row/group: CHECK_OOB's z3 index validator caught it, T4.98k) --
    # and the s<8 nibble selector, which reduces to j<4 (independent of hi -- verified by truth table, j in 0..7)
    lo4_byte = _byte(raw, base + 96 + hi.where(2 * p + 1, 2 * p))
    low4 = (j < 4).where(lo4_byte & 0xF, lo4_byte >> 4)
    high2 = (_byte(raw, base + 104 + hi.where(2 * j_par + 1, 2 * j_par)) >> (2 * half_j)) & 3
    sc = (((low4 | (high2 << 4)) ^ 0x80) - 0x80).float() - 32
    return (d * sc * (q2 + 4 * qh - 4).float()).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _iq3_xxs_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp, *, grid:UOp):
  # ggml type 18 (98 bytes): d (f16) at 0, qs[64] at 2 (8-bit grid code per 4 weights), 8 little-endian uint32
  # scale/sign words at 66, one per 32-wide sub-block j (gguf.py ggml_type==18, field-major). See the module-level
  # comment above for the e//4 / e%4 / i//8 / i%8 restructuring (_div4_mod4, kk_hi2 below).
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[IQ3_XXS])
  d = _half_at(raw, base)
  w = _byte(raw, base+66+4*j) | (_byte(raw, base+67+4*j) << 8) | (_byte(raw, base+68+4*j) << 16) | (_byte(raw, base+69+4*j) << 24)
  scale = (w >> 28) & 0xF
  db = d * (scale.float() + 0.5) * 0.5
  kk_div4, kk_mod4 = _div4_mod4(kk)
  kk_hi2 = (kk >= 16).where(2, 0)
  def elem(o):
    i = kk + o
    code_idx = 8 * j + kk_div4 + o // 4       # e//4 (0..63): o is compile-time, safe to // /% directly
    byte_in_word = kk_mod4 + o % 4            # e%4
    magnitude = _grid_byte(grid, _byte(raw, base + 2 + code_idx), byte_in_word)
    k_sub = kk_hi2 + o // 8                   # i//8 (0..3): the 8-weight sign sub-group
    v = i - 8 * k_sub                         # i%8, via subtraction instead of a raw % on i
    s7 = (w >> (7 * k_sub)) & 0x7F
    par = s7 ^ (s7 >> 4)  # XOR-fold parity (bit0), same idiom as gguf.py's even_signs derivation
    par = par ^ (par >> 2)
    par = par ^ (par >> 1)
    sign = 1 - 2 * (((s7 | ((par & 1) << 7)) >> v) & 1)
    return (db * magnitude.float() * sign.float()).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _iq3_s_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp, *, grid:UOp):
  # ggml type 21 (110 bytes): d (f16) at 0, qs[64] at 2, qh[8] at 66, signs[32] at 74, scales[4] at 106 (gguf.py
  # ggml_type==21). scale/qh/signs all go through gguf.py's (-1,N,1)-reshaped q_to_uint8 calls -- BYTE-major, see
  # the module-level comment above. code index (e//4) and grid-byte (e%4): same _div4_mod4 restructuring as
  # IQ3_XXS; since the resulting per-o delta is always < 8, the qh code's c//8 collapses to j exactly (no division
  # of a lane-dependent value needed) and c%8 IS that delta.
  base, j = _kquant_base(output_row, group_count, group, QUANT_SIZES[IQ3_S])
  d = _half_at(raw, base)
  nib = _nibble(_byte(raw, base + 106 + j // 2), (j % 2).ne(0))
  scale = (1 + 2 * nib).float()
  kk_div4, kk_mod4 = _div4_mod4(kk)
  kk_hi2 = (kk >= 16).where(2, 0)
  def elem(o):
    i = kk + o
    delta = kk_div4 + o // 4                  # e//4 mod 8 == c%8 (c//8 == j, since delta is always < 8)
    c = 8 * j + delta                          # code index e//4 (0..63)
    byte_in_word = kk_mod4 + o % 4             # e%4
    qh_bit = (_byte(raw, base + 66 + j) >> delta) & 1
    # + not | : the operands are bit-disjoint (qs_byte < 256, qh_bit<<8 in {0,256}) so the value is identical, but
    # tinygrad's vmin/vmax for a non-bool OR is unbounded (always dtype.min/max) while ADD gets a tight interval --
    # an unbounded index forces CHECK_OOB's z3 prover, which then crashes on the dynamic-amount shift in `delta`'s
    # ancestry (AttributeError: 'ArithRef' object has no attribute 'as_long'); ADD keeps idx providably in [0,511)
    # and the fast vmin/vmax check passes without ever reaching z3 (T4.98k).
    idx = _byte(raw, base + 2 + c) + (qh_bit << 8)
    magnitude = _grid_byte(grid, idx, byte_in_word)
    k_sub = kk_hi2 + o // 8                    # i//8 (0..3): sign byte sub-index
    sign_bit = (_byte(raw, base + 74 + 4 * j + k_sub) >> (i - 8 * k_sub)) & 1
    sign = 1 - 2 * sign_bit
    return (d * scale * magnitude.float() * sign.float()).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

# ggml_type -> (fragment decoder, K alignment the decoder needs, kernel name)
WMMA_FORMATS: dict[int, tuple[Callable, int, str]] = {
  Q8_0: (_q8_0_dequant4, 32, "linear_q8_0_wmma_nv"), Q4_0: (_q4_0_dequant4, 32, "linear_q4_0_wmma_nv"),
  Q4_1: (_q4_1_dequant4, 32, "linear_q4_1_wmma_nv"), IQ4_NL: (_iq4_nl_dequant4, 32, "linear_iq4_nl_wmma_nv"),
  Q4_K: (_q4_k_dequant4, 256, "linear_q4_k_wmma_nv"), Q5_K: (_q5_k_dequant4, 256, "linear_q5_k_wmma_nv"),
  Q6_K: (_q6_k_dequant4, 256, "linear_q6_k_wmma_nv"), IQ4_XS: (_iq4_xs_dequant4, 256, "linear_iq4_xs_wmma_nv"),
  Q3_K: (_q3_k_dequant4, 256, "linear_q3_k_wmma_nv"), IQ3_XXS: (_iq3_xxs_dequant4, 256, "linear_iq3_xxs_wmma_nv"),
  IQ3_S: (_iq3_s_dequant4, 256, "linear_iq3_s_wmma_nv")}

def _token_tile_for(tokens:int, knob:int=64) -> int:
  # T4.124c: the largest of {knob, 32, 16} that is <= knob and divides tokens; knob=64 (NV_WMMA_TOKEN_TILE's default)
  # reproduces the original fixed rule (64 if tokens%64==0 else 32 if tokens%32==0 else WMMA_M) exactly.
  for tile in sorted({t for t in (knob, 32, 16) if t <= knob}, reverse=True):
    if tokens % tile == 0: return tile
  return WMMA_M
def _output_tiles_for(out_features:int) -> int: return 4 if out_features % (WMMA_N*4) == 0 else 1

# ******** T4.126: int8 tensor-core (IMMA) prefill gemm -- q8 activations x int8-unpacked weights (T4.128: Q4_1, Q5_K) ********

def _i8x4(word:UOp) -> tuple[UOp, ...]:
  # the 4 signed bytes of a uint32, little-endian (element 4j+i of a packed fragment register is byte i). The renderer packs
  # them back into the mma's .b32 operand through the fragment struct; nvcc folds the round trip (see _linear_imma_nv)
  return tuple((word >> (8*i)).cast(dtypes.int8) for i in range(4))

def _int_times_half(c:UOp, d:UOp) -> UOp:
  # T4.126: float(c) * d for an int32 c with |c| < 2**22 and an fp16-valued fp32 d, as ONE fused multiply-add and no I2F
  # (sm_86 issues I2F at 16/clk/SM vs 128 for FFMA -- four conversions per lane per mma would cap the kernel at half the
  # int8 tensor rate). f = bitcast(c + 0x4B400000) is 1.5*2**23 + c exactly; 1.5*2**23 * d is exact in fp32 (2 + 11
  # significant bits), so fma(f, d, -1.5*2**23*d) rounds the exact c*d once: the same bits as cvt.rn.f32.s32 then FMUL. A
  # Q4_0 x q8 group dot is |c| <= 32*8*127 = 32512. The codegen's pm_wmma_add folds the +0x4B400000 into the mma's C operand
  # (WMMA(a,b,0)+k -> WMMA(a,b,k)): the tensor core does the integer add.
  # T4.128: also exact for Q5_K's d*sc (fp16 d times a 6-bit sc: <= 17 significant bits, 19 after the 1.5*2**23) and |c| <=
  # 32*31*127 = 125984; Q4_1's d is fp16 and |c| <= 32*15*127.
  return (c + 0x4B400000).bitcast(dtypes.float32) * d + d * -12582912.0

# T4.128: the per-format halves of the IMMA kernel. imma_b(raw, n, groups, group, t) -> the lane's two B registers (uint32, 4 int8
# each): weight elements 4t..4t+3 and 16+4t..16+4t+3 of output row n's 32-group `group` (the fragment map in _linear_imma_nv).
# imma_scales(raw, n, groups, group) -> (D, M) with weight = D*q + M for that group's q in the B registers: the group adds
# D*xd*dot(q, x8) + M*xs to the output (xs = xd*sum(x8), q8_quantize's sums) -- M is None for Q4_0, whose -8 is folded into q.
# Every index here is lane-dependent only through n and t, and the WARP lane range is never split, so the word views' //, %
# and shifts are safe (unlike the fp16 kernel's LOCAL lane, _byte).

def _q4_0_imma_b(raw:UOp, n:UOp, groups:int, group:UOp, t:UOp) -> tuple[UOp, UOp]:
  # u16 view, 9 per 18-byte block (d, qs[0:16]): elements 0-15 are the low nibbles of qs[0:16] and 16-31 the high nibbles of the
  # same bytes (gguf.py), so qs[4t..4t+3]'s low nibbles are b0 and its high nibbles b1; nibble-8 per byte is (n + 0x78) ^ 0x80
  # (no carry: n + 0x78 <= 0x87)
  base16 = (n*groups + group)*9
  qs = raw[base16 + 1 + 2*t].cast(dtypes.uint32) | (raw[base16 + 2 + 2*t].cast(dtypes.uint32) << 16)
  def unsign(nib:UOp) -> UOp: return (nib + 0x78787878) ^ 0x80808080
  return unsign(qs & 0x0F0F0F0F), unsign((qs >> 4) & 0x0F0F0F0F)

def _q4_0_imma_scales(raw:UOp, n:UOp, groups:int, group:UOp) -> tuple[UOp, UOp|None]: return _half(raw[(n*groups + group)*9]), None

def _q4_1_imma_b(raw:UOp, n:UOp, groups:int, group:UOp, t:UOp) -> tuple[UOp, UOp]:
  # u32 view (amd.py packed_u32), 5 words per 20-byte block: d | m << 16, then qs[0:16] in Q4_0's nibble order; q = 0..15 as is
  qs = raw[(n*groups + group)*5 + 1 + t]
  return qs & 0x0F0F0F0F, (qs >> 4) & 0x0F0F0F0F

def _q4_1_imma_scales(raw:UOp, n:UOp, groups:int, group:UOp) -> tuple[UOp, UOp|None]:
  dm = raw[(n*groups + group)*5]
  return _half(dm), _half(dm >> 16)

def _q5_k_imma_b(raw:UOp, n:UOp, groups:int, group:UOp, t:UOp) -> tuple[UOp, UOp]:
  # u32 view (Q5_K's self.weight), 44 words per 176-byte super-block: d | dmin << 16, scales[12] (words 1-3), qh[32] (4-11),
  # qs[128] (12-43). Sub-block j's element i is nibble (j odd: high) of qs[(j//2)*32 + i] with bit j of qh[i] on top (gguf.py,
  # _q5_k_dequant4): q = 0..31, four elements per word -- nv_quant's decode gemv builds the same words
  base, j = _kquant_base(n, groups, group, Q5_WORDS)
  shift, bit = ((j % 2)*4).cast(dtypes.uint32), j.cast(dtypes.uint32)
  def q(r:int) -> UOp:
    return ((raw[base + 12 + (j//2)*8 + 4*r + t] >> shift) & 0x0F0F0F0F) | (((raw[base + 4 + 4*r + t] >> bit) & 0x01010101) << 4)
  return q(0), q(1)

def _q5_k_imma_scales(raw:UOp, n:UOp, groups:int, group:UOp) -> tuple[UOp, UOp|None]:
  # _q4q5_k_scales on words: byte j%4 of scales[0:4] / [4:8] / [8:12] (words 1, 2, 3); j < 4: sc, m = the low 6 bits of the first
  # two, else the low / high nibble of the third topped up with the first two's spare high bits. D = d*sc, M = -dmin*m (both exact).
  # The minus is dmin's fp16 sign bit flipped, not a *-1: the codegen turns acc + (x*-1)*y into acc - x*y, which nvcc still fuses
  # (FFMA with a negated operand) but the PYTHON emulator rounds x*y first -- the two would then disagree in the last bit
  base, j = _kquant_base(n, groups, group, Q5_WORDS)
  sh = ((j % 4)*8).cast(dtypes.uint32)
  dm, lo_sc, lo_mn, hi = raw[base], raw[base + 1] >> sh, raw[base + 2] >> sh, raw[base + 3] >> sh
  low = j < 4
  sc = low.where(lo_sc & 63, (hi & 0xF) | (((lo_sc >> 6) & 3) << 4))
  mn = low.where(lo_mn & 63, ((hi >> 4) & 0xF) | (((lo_mn >> 6) & 3) << 4))
  return _half(dm) * sc.float(), _half((dm >> 16) ^ 0x8000) * mn.float()

# ggml_type -> (B-register builder, epilogue scales, kernel name); the formats with an M need q8_quantize's sums
IMMA_FORMATS: dict[int, tuple[Callable, Callable, str]] = {
  Q4_0: (_q4_0_imma_b, _q4_0_imma_scales, "linear_q4_0_imma_nv"), Q4_1: (_q4_1_imma_b, _q4_1_imma_scales, "linear_q4_1_imma_nv"),
  Q5_K: (_q5_k_imma_b, _q5_k_imma_scales, "linear_q5_k_imma_nv")}

class _RegRaw:
  """T4.131 (NV_IMMA_WPREFETCH=1): a stand-in for `raw` in the imma_b/imma_scales calls. Record mode (regs and raw): every raw[i]
  becomes a global load stored into the next slot of the register array `regs`; replay mode (regs only): the same call sequence
  hands the slots back, in order, as register loads; count mode (raw only) just counts the reads to size the array. The format
  functions' call order is the whole contract -- the index arithmetic (Q5_K's sub-block shift and bit) is recomputed on replay
  from the stage being computed, only the words come from the registers."""
  def __init__(self, regs:UOp|None, raw:UOp|None):
    self.regs, self.raw, self.k = regs, raw, 0
    self.stores: list[UOp] = []
    self.loads: list[UOp] = []
  def __getitem__(self, idx:UOp) -> UOp:
    k, self.k = self.k, self.k + 1
    if self.raw is None:
      self.loads.append(ld := cast(UOp, self.regs)[k].load())
      return ld
    if self.regs is not None: self.stores.append(self.regs[k].store(self.raw[idx]))
    return self.raw[idx]

@functools.cache
def _linear_imma_nv(out:UOp, raw:UOp, xq:UOp, xd:UOp, xsum:UOp|None=None, *, ggml_type:int, out_features:int, in_features:int,
                    token_tile:int, warps:int, out_tiles:int, stage:int, pipe:int=0, minblocks:int=0, wprefetch:int=0, splitk:int=1) -> UOp:
  """T4.126: Q4_0 prefill gemm on the int8 tensor cores (mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32): one mma per
  (16-token m-tile, 8-output n-tile, 32-wide group) from a zero int32 accumulator -- exact int dot -- then
  acc += (float(c) * d_w[n]) * xd[m] in fp32, llama.cpp MMQ's numerics. Inputs: out (tokens, out_features) fp32, raw = the
  Q4_0 weight's uint16 view (amd.py packed_u16; 9 u16 per 18-byte block: d, then qs[0:16]), xq/xd = nv_quant.q8_quantize's
  (tokens, groups, 8) uint32 words (4 int8 per word, element 4w+i = byte i) and (tokens, groups) fp32 scales.
  T4.128: the same kernel for Q4_1 and Q5_K -- IMMA_FORMATS supplies the B registers and the per-output (D, M); with an M the
  group adds M[n] * xsum[m] after the D term (one more FFMA per output element), xsum = q8_quantize(sums=True)'s third output,
  staged like xd.

  T4.131 splitk > 1: a third grid axis (fastest) of splitk blocks per output tile, each walking steps/splitk consecutive stages
  from a zero accumulator into out[k] of a (splitk, tokens, out_features) partials buffer; the caller sums over k.

  Layout: grid (tokens/token_tile, out_features/(8*out_tiles*warps)) with the token blocks fastest, block = `warps` warps x 32
  lanes. lane is its own AxisType.WARP range (never split by the lowerer, mark_range_mod skips WARP: renders as threadIdx.x, 32)
  and warp an AxisType.LOCAL range (threadIdx.y; a stage whose fill index needs warp%k splits it into y/z, a consistent
  relabelling of the warps), so the hardware lane IS `lane` -- no lane%8/lane//8 trick needed here (_wmma_layout_nv). Each warp
  owns out_tiles n-tiles and every m-tile of the token tile: B is decoded once per (n-tile, group) and reused over the m-tiles,
  A once per (m-tile, group) and reused over the n-tiles. The REDUCE loop steps `stage` groups at a time through a
  shared-memory copy of the block's activations (stage 0 = no staging, fragments straight from global). T4.128 pipe=1: two
  copies of that tile, the loop's own fills prefetching the next stage into the other copy while it computes on this one.

  Fragment map for m16n8k32 .s8 (g = lane//4 in [0,8), t = lane%4; k relative to the 32-wide group). Derived like T4.98f:
  a forced-TC fp8 matmul (16x64 @ 64x8, codegen/opt/tc.py's cuda_81632_f8 = the same m16n8k32 8-bit fragment shape),
  Opt(OptOps.TC, 0, (-1,0,1)), rendered for sm_89 via CUDARenderer(DEV.target("NV", arch="sm_89")): lidx0 = threadIdx.x (32),
  g = lidx0>>2, t = lidx0&3; A loaded as four 4-byte runs A[g][4t..], A[g+8][4t..], A[g][16+4t..], A[g+8][16+4t..] (in that
  register order), B as B[4t..4t+3][g] then B[16+4t..16+4t+3][g], C stored as C[g][2t..2t+1] and C[g+8][2t..2t+1]. Agrees with
  the PTX ISA's m16n8k32 8-bit layout (groupID/threadID_in_group) and with ops_python's (8,16,32) emulator model, which
  test_llm_nv_gemm's _imma_probe runs against numpy:

           A (MxK = tokens x k)            B (KxN = k x outputs)          C/D (MxN) int32
    a0 = A[g,   4t..4t+3]              b0 = B[4t..4t+3,    g]         c0 = C[g,   2t]   c1 = C[g,   2t+1]
    a1 = A[g+8, 4t..4t+3]              b1 = B[16+4t..+3,   g]         c2 = C[g+8, 2t]   c3 = C[g+8, 2t+1]
    a2 = A[g,   16+4t..16+4t+3]
    a3 = A[g+8, 16+4t..16+4t+3]        (byte i of a register = the i-th k of its run)

  So a0..a3 are xq words t, t, t+4, t+4 of tokens g, g+8, g, g+8 -- the quantizer's packing IS the fragment, no unpack. b0/b1
  come from ONE 32-bit qs word in Q4_0/Q4_1 (low and high nibbles, _q4_0_imma_b). C's n index (2t, 2t+1) is a different lane
  split than B's (g) -- the scales the epilogue needs are those of outputs 2t and 2t+1, not of the row this lane decoded."""
  imma_b, imma_scales, name = IMMA_FORMATS[ggml_type]
  groups, S = in_features // Q8_GROUP_SIZE, max(stage, 1)  # S groups per REDUCE iteration
  assert (groups // S) % splitk == 0, f"split-K {splitk} must divide the {groups // S} stage steps"
  steps = groups // S // splitk  # T4.131: stage steps per block (every step of the one pass when splitk == 1)
  # output_block gets the lower axis id so the token blocks of one output block are adjacent in the launch order (gidx0 is the
  # LAST global axis): they run together and share their weight rows through L2 instead of each streaming them from DRAM
  output_block, token_block = UOp.range(out_features // (WMMA_N*out_tiles*warps), 0), UOp.range(out.shape[-2] // token_tile, 1)
  warp, lane = UOp.range(warps, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  g, t = lane // 4, lane % 4
  t0 = token_block*token_tile
  n0s = tuple(((output_block*warps + warp)*out_tiles + o)*WMMA_N for o in range(out_tiles))
  r0s = tuple(mt*WMMA_M for mt in range(token_tile // WMMA_M))  # m-tile rows within the token tile
  accs = [[UOp.placeholder((4,), dtypes.float32, slot=o*len(r0s)+mt, addrspace=AddrSpace.REG) for mt in range(len(r0s))]
          for o in range(out_tiles)]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  zero = UOp.const((0,)*4, dtypes.int32)
  step = UOp.range(steps, 3, AxisType.REDUCE)
  # T4.131: the split-K axis is a global range with the highest axis id (the global dims sort by id, the last one is gidx0): the
  # splits of one output tile launch adjacently, its token blocks `splitk` launches apart, still within one wave
  kblk = UOp.range(splitk, 4) if splitk > 1 else None
  def stage_of(st): return st if kblk is None else kblk*steps + st  # this block's stage `st` as an index into every stage
  prefetch:list[UOp] = []
  if stage:
    # T4.126: the block's activation tile for these S groups (token_tile x 8S words + token_tile x S scales) staged once in
    # shared memory by all warps, then read by every warp: the warps differ only in their n-tiles, so global activation traffic
    # drops `warps`-fold and a token's S groups are one contiguous 32S-byte run (its S scales 4S bytes) instead of scattered
    # 4-byte reads. Row strides 8S+4 words / S+1 floats keep the fragment reads conflict-free: lane (g,t) reads word
    # g*(8S+4) + t (+const), bank 4*(g*(2S+1) mod 8) + t, a bijection over the 32 lanes since 2S+1 is odd; the scale read
    # g*(S+1) (+const) hits 8 distinct banks (S+1 odd) or, S odd, 2-way at worst. The LOCAL stores make add_raw_barrier put a
    # __syncthreads() before the reads, and add_war_barrier one at the end of the step loop (codegen/__init__.py).
    wstride, dstride, threads, copies = 8*S + 4, S + 1, warps*WARP_SIZE, 2 if pipe else 1
    xs_buf = UOp.placeholder((copies*token_tile*wstride,), dtypes.uint32, slot=100, addrspace=AddrSpace.LOCAL)
    xd_buf = UOp.placeholder((copies*token_tile*dstride,), dtypes.float32, slot=101, addrspace=AddrSpace.LOCAL)
    tiles = [(xs_buf, wstride, 8*S, 4, lambda at, r, c: xq[t0 + r, at*S + c // 8, c % 8]),
             (xd_buf, dstride, S, 1, lambda at, r, c: xd[t0 + r, at*S + c])]
    if xsum is not None:  # T4.128: the sums ride along with the scales (same layout, same fill index)
      tiles.append((sum_buf := UOp.placeholder((copies*token_tile*dstride,), dtypes.float32, slot=102, addrspace=AddrSpace.LOCAL), dstride,
                    S, 1, lambda at, r, c: xsum[t0 + r, at*S + c]))
    def fill(at, copy) -> UOp:  # stage `at`'s tiles into tile copy `copy` (None: the one copy of the unpipelined kernel)
      fills = []
      # each thread fills 4 consecutive words (a 16-byte-aligned run in both buffers: 8S and 8S+4 are multiples of 4), which the
      # coalescer turns into one uint4 LDG + one uint4 STS; the scales go one float per thread
      for buf, stride, per_row, vec, load in tiles:
        total = token_tile*per_row//vec
        for i in range(-(-total // threads)):
          idx = warp*WARP_SIZE + lane + i*threads
          if total % threads: idx = idx.minimum(total - 1)  # surplus threads rewrite the last element with its own value: no gate
          r, c0 = idx // (per_row//vec), idx % (per_row//vec) * vec
          base = r * stride + c0 if copy is None else copy*(token_tile*stride) + r*stride + c0
          fills += [buf[base + j].store(load(at, r, c0 + j)) for j in range(vec)]
      return UOp.group(*fills)
    if pipe:
      # T4.128: the prologue fills copy 0 with stage 0 before the loop (its barrier lands outside it); step then reads copy step&1
      # and stores stage step+1 into the other copy. Nothing in the step reads those stores, so the linearizer places them last
      # (stores late), after the mmas: the global loads issue at the top and their latency hides behind the step's tensor work.
      # add_war_barrier's one __syncthreads() at the end of the step orders both hazards (next step's reads after these stores,
      # these stores after the previous step's reads of that copy) -- one barrier per step instead of two. The last step's
      # prefetch is clamped to the last stage: a copy nobody reads again, no gate. The copy index is step & 1, NOT step % 2: a
      # RANGE % c makes pm_split_ranges split the loop into (n/2, 2) ranges whose 3-tuple args add_war_barrier's r.arg[1] check
      # doesn't recognise -- the end-of-step barrier silently vanished (a WAR race; test_llm_nv_gemm pins the barriers)
      ready, cur = fill(stage_of(0), 0), step & 1
      prefetch.append(fill(stage_of((step + 1).minimum(steps - 1)), 1 - (step & 1)))
    else: ready, cur = fill(stage_of(step), None), None
    xs, xds = xs_buf.after(ready), xd_buf.after(ready)
    def at_copy(i, stride:int): return i if cur is None else cur*(token_tile*stride) + i
    def word(row, w:int, s:int): return xs[at_copy(row*wstride + s*8 + w, wstride)]
    def scale(row, s:int): return xds[at_copy(row*dstride + s, dstride)]
    if xsum is not None:
      sums = sum_buf.after(ready)
      def gsum(row, s:int): return sums[at_copy(row*dstride + s, dstride)]
  else:  # stage 0: every warp loads its fragments straight from global (the pre-staging kernel, kept as a card A/B)
    def word(row, w:int, s:int): return xq[t0 + row, stage_of(step), w]
    def scale(row, s:int): return xd[t0 + row, stage_of(step)]
    def gsum(row, s:int): return cast(UOp, xsum)[t0 + row, stage_of(step)]
  def wside(rr, at) -> list[tuple[UOp, tuple]]:
    # the weight side of stage `at` read through raw view rr (raw itself, or T4.131's _RegRaw): per (s, n-tile) the B fragment and
    # (D, M) for outputs 2t and 2t+1 -- this call order is _RegRaw's record/replay contract
    vals = []
    for s in range(S):
      group = at*S + s
      for n0 in n0s:
        bfrag = UOp.stack(*(b for word32 in imma_b(rr, n0 + g, groups, group, t) for b in _i8x4(word32)))
        vals.append((bfrag, tuple(imma_scales(rr, n0 + 2*t + j, groups, group) for j in (0, 1))))
    return vals
  if wprefetch:
    # T4.131: the weight words of stage step+1 (every raw[] read imma_b/imma_scales make: Q4_0 2 qs + 2 d per (group, n-tile)) are
    # loaded one stage ahead into a register array that lives across the loop back-edge, so their global-load latency hides behind
    # this stage's mmas (T4.128's pipe did that for the activation tile only; the weights were the latency left exposed at the top
    # of every step). The prologue loads stage 0; the step replays the array for its own stage, then refills it with the next
    # stage's words (the last step reloads the last stage: a harmless re-read, no gate, as pipe's fill). The refill stores go
    # through a view `after` the replay loads, so the linearizer can never put a refill ahead of the read it would clobber
    probe = _RegRaw(None, raw)
    wside(probe, stage_of(step.const_like(0)))
    wregs = UOp.placeholder((probe.k,), raw.dtype, slot=103, addrspace=AddrSpace.REG)  # the raw view's word type (u16 or u32)
    first = _RegRaw(wregs, raw)
    wside(first, stage_of(step.const_like(0)))
    wr = wregs.after(UOp.group(*first.stores)).after(step)
    replay = _RegRaw(wr, None)
    wvals = wside(replay, stage_of(step))
    ahead = _RegRaw(wr.after(*replay.loads), raw)
    wside(ahead, stage_of((step + 1).minimum(steps - 1)))
    prefetch += ahead.stores
  else: wvals = wside(raw, stage_of(step))
  running = [[acc.after(step) for acc in row] for row in accs]
  for s in range(S):
    afrags = tuple(UOp.stack(*(b for r in range(4) for b in _i8x4(word(r0 + g + 8*(r % 2), t + 4*(r // 2), s)))) for r0 in r0s)
    xss = tuple(UOp.stack(*(scale(r0 + g + 8*(i // 2), s) for i in range(4))) for r0 in r0s)  # C's m index: g for c0/c1, g+8 for c2/c3
    if xsum is not None: sss = tuple(UOp.stack(*(gsum(r0 + g + 8*(i // 2), s) for i in range(4))) for r0 in r0s)
    for o in range(out_tiles):
      bfrag, dm = wvals[s*out_tiles + o]
      dws = UOp.stack(*(dm[i % 2][0] for i in range(4)))  # C's n index: 2t for c0/c2, 2t+1 for c1/c3
      mws = None if dm[0][1] is None else UOp.stack(*(dm[i % 2][1] for i in range(4)))
      for mt, afrag in enumerate(afrags):
        # acc += (c * d_w) * xd: two FFMAs per element. Not c*(d_w*xd): the C renderer strips a same-op child's parens, so the
        # card would compile it as (c*d_w)*xd anyway and round differently from this graph on the emulator
        running[o][mt] = running[o][mt] + _int_times_half(UOp.wmma(afrag, bfrag, zero, *IMMA_ARG), dws) * xss[mt]
        if mws is not None: running[o][mt] = running[o][mt] + mws * sss[mt]  # T4.128: + M[n] * xsum[m], the third FFMA
  update = UOp.group(*(acc.store(value) for row, values in zip(accs, running) for acc, value in zip(row, values)), *prefetch).end(step)
  stores = []
  for n0, row in zip(n0s, accs):
    for r0, acc in zip(r0s, row):
      c0, c1, c2, c3 = (acc.after(update)[i].load() for i in range(4))
      m = t0 + r0 + g
      o_ = out if kblk is None else out[kblk]  # T4.131: this split's partial
      stores += [o_[m, n0 + 2*t].store(c0), o_[m, n0 + 2*t + 1].store(c1), o_[m + 8, n0 + 2*t].store(c2), o_[m + 8, n0 + 2*t + 1].store(c3)]
  # T4.131: minblocks -> __launch_bounds__(threads, minblocks) (KernelInfo.min_blocks, the CUDA renderer); 0 = today's KernelInfo
  ranges = (token_block, output_block, warp, lane) if kblk is None else (kblk, token_block, output_block, warp, lane)
  return UOp.group(*stores).end(*ranges).sink(arg=KernelInfo(name=name, opts_to_apply=(), min_blocks=minblocks))

@functools.cache
def _wmma_kernel(out:UOp, raw:UOp, x:UOp, grid:UOp|None=None, *, out_features:int, in_features:int, ggml_type:int,
                  token_tile:int) -> UOp:
  # grid: the IQ3 codebook (nv.py iq_grid) as a 4th kernel input for IQ3_XXS/IQ3_S (T4.98k); the decoder receives it by keyword.
  # token_tile is a required plain-int keyword (the caller computes it from NV_WMMA_TOKEN_TILE via _token_tile_for) so it joins
  # this functools.cache key -- never read the ContextVar in here, a cache hit would then serve a stale tile for a later call
  # with a different knob value, and no local default here means a caller can never silently mis-tile a different `tokens`.
  dequant4, _, name = WMMA_FORMATS[ggml_type]
  if ggml_type in IQ_GRID_SIZES: dequant4 = functools.partial(dequant4, grid=grid)
  layout = _wmma_layout_nv(out, out_features, token_tile, _output_tiles_for(out_features))
  return _quant_linear_wmma_nv(out, raw, x, out_features, in_features, layout, dequant4, name)

def _imma_weight(layer:Linear) -> Tensor|None:
  # T4.128: the aligned view each IMMA format reads (amd.py set_quantized): Q4_0's u16 view, Q4_1's u32 view, Q5_K's word view
  return {Q4_0: layer.packed_u16, Q4_1: layer.packed_u32, Q5_K: layer.weight}.get(cast(int, layer.ggml_type))

def _imma_geometry(layer:Linear, tokens:int) -> dict[str, int]|None:
  # T4.126: the IMMA kernel's shape arguments from the NV_IMMA_* knobs, or None (-> the fp16 WMMA kernel) when NV_IMMA is off,
  # the layer isn't an IMMA format (T4.128: Q4_0, Q4_1, Q5_K), or out_features doesn't split into whole blocks. tokens % 16 == 0
  # and in_features % 32 (Q5_K: 256) == 0 hold here.
  warps, out_tiles = NV_IMMA_WARPS.value, NV_IMMA_OUT_TILES.value
  if not NV_IMMA.value or _imma_weight(layer) is None or layer.out_features % (WMMA_N*out_tiles*warps): return None
  token_tile = WMMA_M
  while token_tile*2 <= NV_IMMA_TOKEN_TILE.value and tokens % (token_tile*2) == 0: token_tile *= 2
  groups = layer.in_features // Q8_GROUP_SIZE
  stage = NV_IMMA_STAGE.value and math.gcd(NV_IMMA_STAGE.value, groups)  # must divide the groups; 0 = off
  scale_rows = 1 if layer.ggml_type == Q4_0 else 2  # T4.128: the sums buffer is a second scale-sized tile
  copies = 2 if NV_IMMA_PIPE.value and stage else 1  # T4.128: the pipelined kernel holds two copies of the stage tile
  # static __shared__ is capped at 48 KB: shrink the stage rather than hand nvrtc a kernel it rejects mid-serve
  while stage > 1 and copies*token_tile*((8*stage + 4) + scale_rows*(stage + 1))*4 > 48*1024: stage = math.gcd(stage // 2, groups)
  return {"token_tile": token_tile, "warps": warps, "out_tiles": out_tiles, "stage": stage, "pipe": int(copies == 2),
          "minblocks": NV_IMMA_MINBLOCKS.value, "wprefetch": NV_IMMA_WPREFETCH.value,
          "splitk": _imma_splitk(groups // max(stage, 1), layer.out_features // (WMMA_N*out_tiles*warps) * (tokens // token_tile))}

def _imma_splitk(steps:int, blocks:int) -> int:
  # T4.131: NV_IMMA_SPLITK's rule for a kernel of `blocks` (output tiles x token blocks) each walking `steps` stage steps. 0 = auto:
  # below two waves of the 3090's resident blocks (82 SMs x NV_IMMA_MINBLOCKS, or ptxas's 2 blocks per SM at the defaults) take
  # the smallest divisor of the steps that reaches two waves, else the largest <= 8 (128 tokens at the defaults: 5120-out 160
  # blocks -> 4, 10240-out 320 -> 2, 17408-out 544 -> 1); n = n wherever it divides the steps (1 = off)
  if (knob := NV_IMMA_SPLITK.value) != 0: return knob if knob > 1 and steps % knob == 0 else 1
  if blocks >= (target := 2 * 82 * (NV_IMMA_MINBLOCKS.value or 2)): return 1
  candidates = [k for k in range(1, 9) if steps % k == 0]
  return next((k for k in candidates if blocks * k >= target), candidates[-1])

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
  if (imma := _imma_geometry(layer, cast(int, tokens))) is not None:
    # T4.126: q8 activations (shared across the Linears reading one x, nv_quant's T4.123a cache) x the aligned weight view;
    # T4.128: Q4_1/Q5_K also take the quantizer's group sums
    acts = q8_quantize(x_pad, cast(int, tokens), in_features, sums=layer.ggml_type != Q4_0)
    # T4.131: split-K writes (splitk, tokens, out) fp32 partials, summed here by the generic reduce (its order: p0 + p1 + ... as the
    # reduce loop runs them, from a zero accumulator -- a different fp32 order than the one-pass kernel's serial accumulation, so
    # not bit-identical to it; each partial is bit-exact for its own groups)
    out = Tensor.empty(*((imma["splitk"], tokens) if imma["splitk"] > 1 else (tokens,)), out_features, dtype=dtypes.float32, device=x.device).uop
    all_srcs = (out, cast(Tensor, _imma_weight(layer)).uop.buf_uop, *(a.uop for a in acts))
    params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
    kernel = _linear_imma_nv(*params, ggml_type=layer.ggml_type, out_features=out_features, in_features=in_features, **imma).call(*all_srcs)
    result = Tensor(out.after(kernel))
    if imma["splitk"] > 1: result = result.sum(0)
    result = result.reshape(*x_pad.shape[:-1], out_features)
    if symbolic: result = result.shrink(tuple((0, s) for s in (*x.shape[:-1], out_features)))
    return result if layer.bias is None else result + layer.bias
  xh = x_pad.cast(dtypes.float16).contiguous().reshape(tokens, in_features)
  raw = (layer.packed_bytes if layer.packed_bytes is not None else layer.weight).uop.buf_uop  # always the byte view, see _byte
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device).uop
  grids = (iq_grid(layer.ggml_type, cast(str, x.device)).uop,) if layer.ggml_type in IQ_GRID_SIZES else ()  # T4.98k: IQ3 codebook input
  all_srcs = (out, raw, xh.uop, *grids)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  token_tile = _token_tile_for(cast(int, tokens), NV_WMMA_TOKEN_TILE.value)  # read the knob here, never inside the cached builder
  kernel = _wmma_kernel(*params, out_features=out_features, in_features=in_features, ggml_type=layer.ggml_type,
                         token_tile=token_tile).call(*all_srcs)
  result = Tensor(out.after(kernel)).reshape(*x_pad.shape[:-1], out_features)
  if symbolic: result = result.shrink(tuple((0, s) for s in (*x.shape[:-1], out_features)))
  return result if layer.bias is None else result + layer.bias
