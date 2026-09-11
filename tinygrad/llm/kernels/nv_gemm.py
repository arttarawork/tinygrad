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
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import Linear, Q4_0, Q8_0, Q8_GROUP_SIZE, _half  # Q8_GROUP_SIZE==32 doubles here as
                                                                                # the ggml block width of Q4_0/Q8_0
from tinygrad.llm.kernels.nv import _nv_device_ok

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
    arch = getattr(getattr(Device[device], "target", None), "arch", "")
  return arch.startswith("sm_") and int(arch[3:]) >= 80

def _wmma_layout_nv(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(WMMA_N*output_tiles), 1)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  lane_hi, lane_lo = lane // 4, lane % 4  # lane_hi: A's/B's shared row/col index (0-7); lane_lo: the k/n sub-index (0-3)
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

def _q8_0_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 8: value = d * qs[i], qs already signed int8 in element order (gguf.py, ref_q8_0)
  base = (output_row * group_count + group) * 34
  d = _half(raw[base].cast(dtypes.uint16) | (raw[base+1].cast(dtypes.uint16) << 8))
  return tuple((d * raw[base+2+(kk+o)].bitcast(dtypes.int8).float()).cast(dtypes.float16) for o in (0, 1, 8, 9))

def _q4_0_dequant4(raw:UOp, output_row:UOp, group_count:int, group:UOp, kk:UOp):
  # ggml type 2: value = d*(nibble-8); q_to_uint8(.,4)'s order is elements 0-15 = low nibbles of qs[0:16],
  # elements 16-31 = high nibbles of the SAME qs[0:16] (gguf.py, ref_q4_0)
  base = (output_row * group_count + group) * 18
  d = _half(raw[base].cast(dtypes.uint16) | (raw[base+1].cast(dtypes.uint16) << 8))
  def elem(o):
    k = kk + o
    byte = raw[base + 2 + (k % 16)]
    nib = (k >= 16).where(byte >> 4, byte & 0x0F)
    return (d * (nib.cast(dtypes.int32) - 8).float()).cast(dtypes.float16)
  return tuple(elem(o) for o in (0, 1, 8, 9))

def _token_tile_for(tokens:int) -> int: return 64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else WMMA_M
def _output_tiles_for(out_features:int) -> int: return 4 if out_features % (WMMA_N*4) == 0 else 1

@functools.cache
def _q8_0_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int) -> UOp:
  layout = _wmma_layout_nv(out, out_features, _token_tile_for(out.shape[0]), _output_tiles_for(out_features))
  return _quant_linear_wmma_nv(out, raw, x, out_features, in_features, layout, _q8_0_dequant4, "linear_q8_0_wmma_nv")

@functools.cache
def _q4_0_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int) -> UOp:
  layout = _wmma_layout_nv(out, out_features, _token_tile_for(out.shape[0]), _output_tiles_for(out_features))
  return _quant_linear_wmma_nv(out, raw, x, out_features, in_features, layout, _q4_0_dequant4, "linear_q4_0_wmma_nv")

def nv_wmma_linear(layer:Linear, x:Tensor) -> Tensor|None:
  """WMMA gemm for Q8_0/Q4_0 prefill (tokens > 4, multiple of 16 -- or symbolic, padded to `x.max_shape` the way
  amd.py's q8_linear pads for a symbolic chunk size). Returns None when it doesn't cover this call (unsupported
  format, an out_features/in_features that isn't tile-aligned, tokens<=4, or an arch below sm_80) so
  Linear.__call__ (amd.py) falls back to the generic dequant+matmul path or nv_quant.py's decode gemv."""
  if layer.ggml_type not in (Q4_0, Q8_0): return None
  out_features, in_features = layer.out_features, layer.in_features
  if out_features % WMMA_N != 0 or in_features % Q8_GROUP_SIZE != 0: return None
  numel = x.numel()
  symbolic = not isinstance(numel, int)
  tokens = x.max_shape[-2] if symbolic else numel // in_features
  if tokens <= 4 or tokens % WMMA_M != 0: return None  # tokens<=4: nv_quant.py's decode gemv covers it
  if not _nv_wmma_ok(x.device): return None
  x_pad = x.pad_to(x.max_shape) if symbolic else x  # concrete shape from here on
  xh = x_pad.cast(dtypes.float16).contiguous().reshape(tokens, in_features)
  raw = layer.weight.uop.buf_uop
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xh.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  fxn = _q8_0_wmma_kernel if layer.ggml_type == Q8_0 else _q4_0_wmma_kernel
  kernel = fxn(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel)).reshape(*x_pad.shape[:-1], out_features)
  if symbolic: result = result.shrink(tuple((0, s) for s in (*x.shape[:-1], out_features)))
  return result if layer.bias is None else result + layer.bias
