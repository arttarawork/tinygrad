"""T4.98h: NV fp16 gemv for small DENSE (non-ggml-quantized) Linear layers -- e.g. GatedDeltaNet's
ssm_alpha/ssm_beta head projections ((5120->48) in Qwen3.8-27B, bias=False). A PROFILE=1 decode trace
(2026-09-10) showed these running as a generic reduce kernel (r_48_320_4_4) at 0.249ms/call, 96 calls/token =
24ms/token (16% of decode) for 0.5MB of fp16 weight (~2GB/s). Mirrors upstream's _amd_f16_gemv_kernel/f16_gemv
(commit a0a901c8e, "faster qwen 3.8", RDNA3): one 32-lane warp per (token, output row), each lane accumulating a
4-wide fp16 chunk of the row per step, fp32 accumulate, warp_reduce (kernels/nv.py) sums the 32 lanes. Bias is
added after the kernel (like nv_quant.py's nv_q8_linear) instead of threaded into it -- ssm_alpha/beta are
bias=False so this is unexercised, but it keeps both NV decode kernels' calling convention the same.

T4.122b: the same gemv also covers PREFILL chunks (tokens > 4, including a symbolic chunk-width token count) --
a PROFILE dump (2026-10-27) showed these same two Linear layers compiling to a generic tensor-core kernel at
prefill (`r_32_2_2_2_2_3_320_2`, global_size (1,1,1): ONE thread block doing the whole 32x5120 @ 5120x96
product) at 535us/call, 96 calls/chunk = 51ms of a 205ms chunk. The one-warp-per-(token,output row) layout below
needs no change to cover this: it was always launched with `tokens*out_features` blocks (48 blocks already at
tokens=1), so a wider or symbolic `tokens` just launches more of the same tiny block -- unlike nv_gemm.py's WMMA
path, there's no cross-token work to tile/amortize here, every (token, output row) pair is independent. A
symbolic token count is padded to the chunk bound (`x.max_shape`) and the output sliced back down, exactly as
nv_gemm.py's `nv_wmma_linear` does for its own symbolic prefill chunks -- safe here too because padding adds
whole extra *rows* (one warp's full gemv over zeros, never a partial one), never touches a valid row's math.

Dispatch lives in amd.py's Linear.__call__, behind the existing NV_CUSTOM_QUANT gate (no second env var) -- see
nv_dense_eligible's docstring for why the shape/dtype check is split out from the (now token-count-independent)
checks nv_f16_gemv does internally: they feed two different decisions in Linear.__call__."""
from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.helpers import ALLOW_DEVICE_USAGE, ContextVar
from tinygrad.engine.realize import capturing
from tinygrad.dtype import AddrSpace, dtypes, least_upper_dtype
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import Linear, IQ_GRID_SIZES
from tinygrad.llm.kernels.nv import warp_reduce, iq_grid

WARP_SIZE, VAL_CHUNK = 32, 4
# T4.138 (step 4): a prefill call (tokens > 4) gives each warp a tile of NV_F16_GEMV_TOKENS tokens x NV_F16_GEMV_ROWS output rows: a
# lane's 4 weight halves serve every token of the tile and its 4 activation halves every row. One warp per (token, row) re-read a
# 10 KB weight row and a 10 KB token row per output (126 MB of L2 traffic for ssm_beta|ssm_alpha at 128 tokens -- 47 us per 48 rows on
# the card, 10-31 profile); an 8 x 8 tile reads 1/8 of that. Each (token, row)'s lane sum is the same FMA sequence in the same order,
# so the outputs are bit-identical to the 1 x 1 kernel. Card knobs, read in the dispatchers (uncached) and passed into the cache key;
# a value that doesn't divide the call's tokens/rows falls back to 1. Decode (tokens <= 4) keeps its kernel byte for byte.
NV_F16_GEMV_TOKENS, NV_F16_GEMV_ROWS = ContextVar("NV_F16_GEMV_TOKENS", 8), ContextVar("NV_F16_GEMV_ROWS", 8)

def _gemv_tile(tokens:int, out_features:int) -> dict[str, int]:
  if tokens <= 4: return {}
  t, r = NV_F16_GEMV_TOKENS.value, NV_F16_GEMV_ROWS.value
  return {"tile_tokens": t if t > 1 and tokens % t == 0 else 1, "tile_rows": r if r > 1 and out_features % r == 0 else 1}

def nv_dense_eligible(layer:Linear) -> bool:
  """Shape/dtype check only: stable for the lifetime of a Linear instance (its weight shape/dtype never
  changes), unlike the token count nv_f16_gemv checks below (which varies call to call). amd.py's Linear.__call__
  uses this one to decide whether to keep the layer's nv_supported flag alive across calls -- gating that
  decision on token count instead would permanently kill the decode-time optimization the first time this layer
  sees a >4-token (e.g. prefill-chunk) call."""
  return layer.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and layer.out_features <= 2048 and layer.in_features % 128 == 0

@functools.cache
def _nv_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *, in_features:int, out_features:int, tokens:int, tile_tokens:int=1, tile_rows:int=1) -> UOp:
  # one warp per (token, output row) -- T4.138: per (tile_tokens tokens, tile_rows rows) --; each of the 32 lanes accumulates a 4-wide
  # fp16 chunk of the row per step. 1 x 1 builds exactly the graph it always did
  assert tokens % tile_tokens == 0 and out_features % tile_rows == 0
  token, out_row = UOp.range(tokens // tile_tokens, 0, AxisType.GLOBAL), UOp.range(out_features // tile_rows, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  per = in_features // (WARP_SIZE * VAL_CHUNK)
  assert per * WARP_SIZE * VAL_CHUNK == in_features
  w, x = w.reshape((out_features, per, WARP_SIZE*VAL_CHUNK)), x.reshape((tokens, per, WARP_SIZE*VAL_CHUNK))
  if tile_tokens == 1 and tile_rows == 1:
    acc = UOp.const(0, dtypes.float32)
    for i in range(per):
      for j in range(VAL_CHUNK):
        acc = acc + w[out_row, i, lane*VAL_CHUNK+j].load().float() * x[token, i, lane*VAL_CHUNK+j].load().float()
    total = warp_reduce(acc, full_wave=True)
    return out[token, out_row.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token, out_row, lane).sink(
      arg=KernelInfo(name="linear_f16_gemv_nv", opts_to_apply=()))
  # T4.138: the K steps as a REDUCE loop, the tile's accumulators a register file (the fused scan's idiom) -- unrolled like the 1 x 1
  # kernel, all 40 x (tokens + rows) loads were hoisted ahead of the math and an 8 x 8 tile spilled 6.4 KB per thread (ptxas sm_86).
  # Per (token, row) the FMAs still run i-major, j-minor from 0: the 1 x 1 kernel's sequence
  toks, rows = tuple(token*tile_tokens + m for m in range(tile_tokens)), tuple(out_row*tile_rows + r for r in range(tile_rows))
  pairs = tuple((t, o) for t in toks for o in rows)
  regs = UOp.placeholder((len(pairs),), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  regs = regs.after(regs.store(UOp.stack(*(UOp.const(0, dtypes.float32),) * len(pairs))))
  i = UOp.range(per, 3, AxisType.REDUCE)
  accs = [regs.after(i)[k].load() for k in range(len(pairs))]
  for j in range(VAL_CHUNK):
    ws, xs = {o: w[o, i, lane*VAL_CHUNK+j].load().float() for o in rows}, {t: x[t, i, lane*VAL_CHUNK+j].load().float() for t in toks}
    accs = [a + ws[o] * xs[t] for a, (t, o) in zip(accs, pairs)]
  step = regs.store(UOp.stack(*accs)).end(i)
  stores = [out[t, o.valid(lane.eq(0))].store(warp_reduce(regs.after(step)[k].load(), full_wave=True).cast(out.dtype))
            for k, (t, o) in enumerate(pairs)]
  return UOp.group(*stores).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv_nv", opts_to_apply=()))

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
  """Decode AND prefill fp16 gemv for a small dense Linear layer: one warp per (token, output row), unchanged
  from T4.98h for every call shape. A concrete tokens<=4 call takes exactly the path/kernel/launch it always
  did (nothing below depends on a token-count cap, that cap used to be the whole gate). A symbolic token count
  (prefill's bound `toks` Variable) is padded to the chunk bound the same way nv_gemm.py's `nv_wmma_linear`
  pads for its own symbolic prefill chunks: run the kernel at the padded concrete token count, then shrink the
  output back down to x's real (possibly still-symbolic) token count (T4.122b). Returns None only when it
  doesn't cover this call at all (not shape/dtype-eligible, or the weight copy isn't ready yet) so
  Linear.__call__ (amd.py) falls back to the generic matmul."""
  if not nv_dense_eligible(layer): return None
  numel = x.numel()
  symbolic = not isinstance(numel, int)
  tokens = x.max_shape[-2] if symbolic else numel // layer.in_features
  out_dtype = least_upper_dtype(x.dtype, layer.weight.dtype)  # matches nn.Linear's own x.dot(weight) promotion
  if layer.dense_weight is None:
    # the first call of a served model happens inside a @function context (ALLOW_DEVICE_USAGE=0) under JIT capture:
    # a .realize() there asserts ("usage of device NV disallowed", every NV_CUSTOM_QUANT=1 model load on 09-11) and
    # would be recorded into the graph anyway. prepare_dense_weights (called at load) makes the copy; without it,
    # fall through to the generic matmul rather than do eager work here
    if not ALLOW_DEVICE_USAGE.value or capturing: return None
    layer.dense_weight = _fp16_copy(layer)
  weight = layer.dense_weight
  x_pad = x.pad_to(x.max_shape) if symbolic else x  # concrete shape from here on (mirrors nv_gemm.py's nv_wmma_linear)
  # T4.125b: a reshaped fp16 kernel output (add_rmsnorm_q8's want_fp16 copy at GDN's attn_norm site) already is a whole contiguous
  # buffer, but .contiguous() only short-circuits a plain BUFFER, not an AFTER -- it copied it again every step (E_40_32_4_1)
  xh = x_pad if x_pad.dtype == dtypes.float16 and x_pad.uop.has_buffer_identity(after_ok=True) else \
    x_pad.contiguous() if x_pad.dtype == dtypes.float16 else x_pad.cast(dtypes.float16).contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=out_dtype, device=x.device)
  fxn = functools.partial(_nv_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens,
                          **_gemv_tile(cast(int, tokens), layer.out_features))  # concrete after the pad (mypy: int|UOp)
  result = Tensor.custom_kernel(out, weight.reshape(-1), xh.reshape(tokens, layer.in_features), fxn=fxn)[0]
  result = result.reshape(*x_pad.shape[:-1], layer.out_features)
  if symbolic: result = result.shrink(tuple((0, s) for s in (*x.shape[:-1], layer.out_features)))
  return result if layer.bias is None else result + layer.bias

def nv_f16_gemv_raw(weight:Tensor, x:Tensor, out_features:int) -> Tensor:
  """T4.127b: _nv_f16_gemv_kernel called directly on a weight that isn't one Linear's own dense_weight -- no
  ggml/bias/eligibility checks, for a caller that already has its own prepared fp16 weight and knows this is a
  small bias=False dense gemv (GatedDeltaNetBlock's ssm_beta+ssm_alpha concatenated weight, prepare_nv_decode).
  x must already be fp16, 2D (tokens, in_features); weight 2D (out_features, in_features)."""
  tokens, in_features = cast(int, x.shape[0]), cast(int, x.shape[1])
  xh = x if x.uop.has_buffer_identity(after_ok=True) else x.contiguous()  # T4.125b: nv_f16_gemv's own dance, fp16-only here
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float16, device=x.device)
  fxn = functools.partial(_nv_f16_gemv_kernel, in_features=in_features, out_features=out_features, tokens=tokens, **_gemv_tile(tokens, out_features))
  return Tensor.custom_kernel(out, weight.reshape(-1), xh.reshape(tokens, in_features), fxn=fxn)[0]
