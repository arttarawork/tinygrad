"""Fused Gated-DeltaNet prefill scan for NV (CUDA-C, sm_70+). Same math + lane/register mapping as amd.py's gated_delta_prefill;
the only device-specific piece is the 32-lane reduce (ds_swizzle on RDNA3 -> __shfl_xor_sync here). Design: NV_FUSED_SCAN_DESIGN.md."""
from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.amd import kernel_var

GDN_NV_FUSED = ContextVar("GDN_NV_FUSED", 1)  # 0 = never take the fused path, even on NV (A/B vs the loop/WY scan)
# T4.98g: the SAME kernel also serves the T_pad==1 decode step (its token loop is a REDUCE range sized `tokens`;
# tokens==1 is just one iteration of the identical readout+state-update math -- see model.py's dispatch). A
# separate gate, default 0 (byte-identical), independent of GDN_NV_FUSED: the pooled recipe leaves that one 0
# under BEAM_CACHE_ONLY for reasons specific to PREFILL's neighbouring kernels (CLAUDE.md), unrelated to decode.
GDN_NV_FUSED_DECODE = ContextVar("GDN_NV_FUSED_DECODE", 0)

@functools.cache
@functools.cache
def iq_grid(ggml_type:int, device:str) -> Tensor:
  """The IQ3 codebook (iq3xxs_grid: 256 words, iq3s_grid: 512 words; 4 uint8 magnitudes per uint32 word) as a realized
  uint32 tensor on `device` -- the IQ3 decoders in nv_quant.py/nv_gemm.py take it as an extra kernel input and index it
  by code (one 1-2 KB table, L1-resident). Realized here ONCE per device: call from prepare_dense_weights (load time),
  never first inside a @function/JIT capture (T4.98h's lesson)."""
  from tinygrad.runtime.autogen import ggml_common as _ggml
  from tinygrad.llm.kernels.amd import IQ3_XXS, IQ3_S
  words = {IQ3_XXS: _ggml.iq3xxs_grid, IQ3_S: _ggml.iq3s_grid}[ggml_type]
  with Context(ALLOW_DEVICE_USAGE=1): return Tensor(list(words), dtype=dtypes.uint32, device=device).realize()

def _nv_device_ok(device:str) -> bool:
  from tinygrad.renderer.cstyle import CUDARenderer  # PTX=1 renders no Ops.CUSTOM: only the C renderer can emit the shuffle
  with Context(ALLOW_DEVICE_USAGE=1):
    ren = Device[device].renderer
    arch = getattr(getattr(ren, "target", None), "arch", "")
  return isinstance(ren, CUDARenderer) and arch.startswith("sm_") and int(arch[3:]) >= 70  # full-mask __shfl_*_sync = Volta+ (mock-NV is sm_35)

def nv_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV" or not GDN_NV_FUSED.value: return False
  return _nv_device_ok(device)

def nv_decode_kernel_supported(device:str|tuple[str, ...]|None) -> bool:
  # T4.98g: same shape as nv_custom_kernels_supported, keyed on GDN_NV_FUSED_DECODE instead -- kept as its own
  # small function (not a shared helper) so the well-tested prefill gate above stays untouched, same idiom this
  # file already uses to keep the AMD kernel untouched (see the module docstring).
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV" or not GDN_NV_FUSED_DECODE.value: return False
  return _nv_device_ok(device)

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

# verbatim amd._gated_delta_prefill_kernel except for warp_reduce and the kernel name (kept separate so the AMD kernel stays untouched)
@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.placeholder((row_tile*key_dim//32,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float()) for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.REDUCE)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    av, bv = alpha[bh, token, row if alpha_dim > 1 else 0].load(), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k*av) * bv
    updates += [x*av + delta*y for x,y in zip(previous, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q*av + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill_nv", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and state.shape == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, value_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=_gated_delta_prefill_kernel)[0]
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  assert start_pos.uop.is_bound_var
  call = _gated_delta_prefill_kernel(*params, kernel_var(start_pos.uop.src[0])).call(*contig)
  return Tensor(contig[0].after(call))
