"""T4.98h: NV decode-only fp16 gemv for small DENSE (non-ggml-quantized) Linear layers -- e.g. GatedDeltaNet's
ssm_alpha/ssm_beta head projections ((5120->48) in Qwen3.8-27B, bias=False). A PROFILE=1 decode trace
(2026-09-10) showed these running as a generic reduce kernel (r_48_320_4_4) at 0.249ms/call, 96 calls/token =
24ms/token (16% of decode) for 0.5MB of fp16 weight (~2GB/s). Mirrors upstream's _amd_f16_gemv_kernel/f16_gemv
(commit a0a901c8e, "faster qwen 3.8", RDNA3): one 32-lane warp per (token, output row), each lane accumulating a
4-wide fp16 chunk of the row per step, fp32 accumulate, warp_reduce (kernels/nv.py) sums the 32 lanes. Bias is
added after the kernel (like nv_quant.py's nv_q8_linear) instead of threaded into it -- ssm_alpha/beta are
bias=False so this is unexercised, but it keeps both NV decode kernels' calling convention the same.

Dispatch lives in amd.py's Linear.__call__, behind the existing NV_CUSTOM_QUANT gate (no second env var) -- see
nv_dense_eligible's docstring for why the shape/dtype check is split out from the token-count check nv_f16_gemv
does internally: they feed two different decisions in Linear.__call__."""
from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.helpers import ALLOW_DEVICE_USAGE
from tinygrad.engine.realize import capturing
from tinygrad.dtype import dtypes, least_upper_dtype
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import Linear, IQ_GRID_SIZES
from tinygrad.llm.kernels.nv import warp_reduce, iq_grid

WARP_SIZE, VAL_CHUNK = 32, 4

def nv_dense_eligible(layer:Linear) -> bool:
  """Shape/dtype check only: stable for the lifetime of a Linear instance (its weight shape/dtype never
  changes), unlike the token count nv_f16_gemv checks below (which varies call to call). amd.py's Linear.__call__
  uses this one to decide whether to keep the layer's nv_supported flag alive across calls -- gating that
  decision on token count instead would permanently kill the decode-time optimization the first time this layer
  sees a >4-token (e.g. prefill-chunk) call."""
  return layer.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and layer.out_features <= 2048 and layer.in_features % 128 == 0

@functools.cache
def _nv_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *, in_features:int, out_features:int, tokens:int) -> UOp:
  # one warp per (token, output row); each of the 32 lanes accumulates a 4-wide fp16 chunk of the row per step
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  per = in_features // (WARP_SIZE * VAL_CHUNK)
  assert per * WARP_SIZE * VAL_CHUNK == in_features
  w, x = w.reshape((out_features, per, WARP_SIZE*VAL_CHUNK)), x.reshape((tokens, per, WARP_SIZE*VAL_CHUNK))
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(VAL_CHUNK):
      acc = acc + w[out_row, i, lane*VAL_CHUNK+j].load().float() * x[token, i, lane*VAL_CHUNK+j].load().float()
  total = warp_reduce(acc, full_wave=True)
  return out[token, out_row.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token, out_row, lane).sink(
    arg=KernelInfo(name="linear_f16_gemv_nv", opts_to_apply=()))

def _fp16_copy(layer:Linear) -> Tensor:
  # realize once: casting+copying a lazy (ggml-load-chain-or-not) weight view on every decode step would cost as
  # much as the matmul it replaces. Always fp16 regardless of the original dtype -- fp32 weights lose precision
  # here, the same trade the quantized paths already make
  return layer.weight.cast(dtypes.float16).contiguous().realize()

def prepare_dense_weights(model) -> int:
  """Make the fp16 copies nv_f16_gemv reads, once, at load time (Transformer.from_gguf calls this after placement).
  A layer that turns out to be ggml-quantized (set_quantized finds a format) belongs to nv_quant.py, not here --
  the probe mirrors Linear.__call__'s order. Returns the number of layers prepared."""
  from tinygrad import nn
  from tinygrad.llm.kernels.nv_quant import nv_quant_supported  # local: nv_quant imports amd, which nv_dense's caller imports
  n = 0
  for layer in cast(dict[str, Linear], nn.state.get_state_dict(model, tensor_type=Linear)).values():  # typed as Tensors by get_state_dict
    if layer.dense_weight is not None or not layer.use_custom_quant: continue
    if not nv_quant_supported(layer.weight.device): continue
    if not nv_dense_eligible(layer):  # a quantized layer: only the IQ3 codebook needs preparing
      if layer.ggml_type is None: layer.set_quantized(layer.weight)
      if layer.ggml_type in IQ_GRID_SIZES: iq_grid(layer.ggml_type, cast(str, layer.weight.device))
      continue
    if layer.ggml_type is None: layer.set_quantized(layer.weight)
    if layer.ggml_type is None:
      layer.dense_weight = _fp16_copy(layer)
      n += 1
    elif layer.ggml_type in IQ_GRID_SIZES: iq_grid(layer.ggml_type, cast(str, layer.weight.device))  # T4.98k: codebook realized now, not in a capture
  return n

def nv_f16_gemv(layer:Linear, x:Tensor) -> Tensor|None:
  """Decode-only (tokens<=4, like nv_quant.py's nv_q8_linear) fp16 gemv for a small dense Linear layer. Returns
  None when it doesn't cover this call (not shape-eligible, bigger batch, or a symbolic token count) so
  Linear.__call__ (amd.py) falls back to the generic matmul."""
  if not nv_dense_eligible(layer): return None
  numel = x.numel()
  if not isinstance(numel, int): return None  # symbolic token count: no padded path ported (matches nv_quant.py)
  tokens = numel // layer.in_features
  if tokens > 4: return None  # decode-only
  out_dtype = least_upper_dtype(x.dtype, layer.weight.dtype)  # matches nn.Linear's own x.dot(weight) promotion
  if layer.dense_weight is None:
    # the first call of a served model happens inside a @function context (ALLOW_DEVICE_USAGE=0) under JIT capture:
    # a .realize() there asserts ("usage of device NV disallowed", every NV_CUSTOM_QUANT=1 model load on 09-11) and
    # would be recorded into the graph anyway. prepare_dense_weights (called at load) makes the copy; without it,
    # fall through to the generic matmul rather than do eager work here
    if not ALLOW_DEVICE_USAGE.value or capturing: return None
    layer.dense_weight = _fp16_copy(layer)
  weight = layer.dense_weight
  xh = x.contiguous() if x.dtype == dtypes.float16 else x.cast(dtypes.float16).contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=out_dtype, device=x.device)
  fxn = functools.partial(_nv_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens)
  result = Tensor.custom_kernel(out, weight.reshape(-1), xh.reshape(tokens, layer.in_features), fxn=fxn)[0]
  result = result.reshape(*x.shape[:-1], layer.out_features)
  return result if layer.bias is None else result + layer.bias
