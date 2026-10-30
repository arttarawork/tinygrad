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

# ******** T4.123f+g: the whole GatedDeltaNet decode step (T == 1) in two launches ********
# The generic decode graph ran 13 small kernels (~6 us each regardless of size) around the fused scan: the attn_qkv partial sum, the
# conv window in and out of conv_state, conv+SiLU for v and -- recomputed per consumer -- for q/k, their l2 norms, the repeated q/k,
# kq, beta, alpha, ssm_norm's rms and the gated output. Here: two kernels, behind the same GDN_NV_FUSED_DECODE gate.
#  1. _gdn_conv_decode_kernel: one thread per conv channel -- sums attn_qkv's split-K partials, runs the taps + SiLU into conv_out and
#     shifts conv_state in place. Every conv_state element is read and written by one thread only, so the shift is race-free.
#     It cannot live in the scan kernel: a k-head's q/k channels feed every v-head mapped to it (3 blocks at 27B geometry), and one
#     block overwriting conv_state rows that the others still read needs a grid-wide barrier.
#  2. _gdn_decode_kernel: one block per (batch, v-head) = value_dim/row_tile warps x row_tile value rows (the old scan's lane map), so
#     the block owns the whole head. Each warp l2-normalizes its k-head's q/k from conv_out (three warp_reduces over the same 2x128
#     floats in every warp -- L1 hits), forms kq, beta, alpha, runs the gated-delta update of its rows, then ssm_norm's rms over the head
#     is a block reduction (one shared float per warp, one barrier), times the norm weight and SiLU(attn_gate, its split-K partials
#     summed in-kernel), stored fp16 for ssm_out. Row tile 8, not the old scan's 4: 4 means 1024-thread blocks, whose 64-register cap
#     spills 24 B/thread (ptxas sm_86); 8 = 512 threads at 27B geometry, 128 registers, no stack frame, the same state bytes in
#     flight per SM. 48 such blocks on 48 of the 82 SMs instead of 1536 one-warp blocks: the 6.3 MB state read+write is 131 KB per
#     busy SM, far below one SM's load rate, so the scan part should stay DRAM-bound (~7 us) -- the card profile decides.
# lanes=1 is the CPU form of both (one lane owns the whole head, no shuffles, no shared memory) for the CPU tests, as in nv_attn.py.

def _lane(lanes:int, axis:int) -> UOp: return UOp.range(lanes, axis, AxisType.LOCAL) if lanes > 1 else UOp.const(0, dtypes.weakint)

def _added(xs) -> UOp: return functools.reduce(lambda a, b: a + b, xs)  # left to right, the order the generic graph sums in

# T4.132: the two kernels also serve a short run of tokens (speculative decoding's VERIFY, k_eff+1 <= GDN_DECODE_TOKENS) processed
# sequentially inside the kernel: the conv window shifts one row per token and the gated-delta state stays in registers across the
# tokens (one state read + one write for the whole run, as for a single token); per token the rms block reduction gets its own LOCAL
# slots. tokens == 1 builds the exact graph it always did (the row helpers index with `b` alone there, not `b*1 + 0`): the T == 1
# render is pinned by test_llm_nv_scan.py's md5.
GDN_DECODE_TOKENS = 4
# T4.132: a speculative VERIFY run (tokens > 1) may also hand both kernels a `track` buffer: the conv window / recurrent state after each
# of its first tokens-1 tokens, (S, B, ...) with S >= tokens-1 -- what a partial accept of m < tokens-1 drafts adopts as the live state
# (model.py's GatedDeltaNetBlock.spec_track / Transformer._adopt_step) instead of re-running the accepted tokens. Written from values the
# kernels already hold (the window rows, the state in registers); omitted, the graphs are unchanged.

def _token_row(b:UOp, tokens:int): return (lambda t: b) if tokens == 1 else (lambda t: b*tokens + t)  # (batch b, token t) -> flat row

@functools.cache
def _gdn_conv_decode_kernel(conv_out:UOp, conv_state:UOp, qkv:UOp, weight:UOp, *track_bound:UOp, start_pos:UOp, lanes:int=32,
                            tokens:int=1) -> UOp:
  """conv_out (B*tokens,C) fp32: row (b,t) = SiLU(sum_i weight[c,i] * window[t+i]) with window = [conv_state rows (zero at start_pos 0) |
  sum_p qkv[(b,t),c,p] per token]; conv_state (B,taps-1,C) <- window[tokens:] in place. T4.132: track (S,B,taps-1,C) <- window[t+1:t+taps]
  for t < tokens-1, the conv state after token t. The bound start_pos rides last (see gdn_decode): a scalar src, not a buffer slot."""
  *track, _ = track_bound
  batch, history, channels = conv_state.shape
  parts = qkv.shape[-1]
  assert all(isinstance(s, int) for s in (batch, history, channels, parts)) and channels % lanes == 0 and weight.shape == (channels, history+1)
  batch, history, channels, parts = cast(tuple[int, int, int, int], (batch, history, channels, parts))
  block, lane = UOp.range(batch*channels//lanes, 0), _lane(lanes, 1)
  b, c = block // (channels//lanes), (block % (channels//lanes))*lanes + lane
  row = _token_row(b, tokens)
  initial = start_pos.eq(0)
  window = [initial.where(0, conv_state[b, i, c].load().float()) for i in range(history)] + \
           [_added(qkv[row(t), c, p].load() for p in range(parts)) for t in range(tokens)]
  outs = [conv_out[row(t), c].store(_added(x*weight[c, i].load() for i,x in enumerate(window[t:t+history+1])).silu()) for t in range(tokens)]
  # the old rows all feed the outs, so storing the shifted window through conv_state.after(outs) orders every load before every store
  shifts = [conv_state.after(*outs)[b, i, c].store(window[tokens+i].cast(conv_state.dtype)) for i in range(history)]
  shifts += [tr[t, b, i, c].store(window[t+1+i].cast(tr.dtype)) for tr in track for t in range(tokens-1) for i in range(history)]
  ends = tuple(r for r in (lane, block) if r.op is Ops.RANGE)
  return UOp.group(*outs, *shifts).end(*ends).sink(arg=KernelInfo(name="gdn_conv_decode_nv", opts_to_apply=()))

@functools.cache
def _gdn_decode_kernel(z:UOp, state:UOp, conv_out:UOp, beta_alpha:UOp, dt_bias:UOp, ssm_a:UOp, gate:UOp, norm_weight:UOp, *track_bound:UOp,
                       start_pos:UOp, k_heads:int, qk_eps:float, norm_eps:float, lanes:int=32, row_tile:int=8, tokens:int=1) -> UOp:
  """model.py's decode graph from conv_out on, per (b, v-head h): q/k = conv_out's k-head h % k_heads l2-normalized (q * key_dim^-0.5),
  v = conv_out's v channels, beta = sigmoid(beta) in its own dtype, alpha = exp(softplus(alpha + dt_bias) * ssm_a), the fused scan's
  one-token step on state (B,H,V,K) in place (zeroed at start_pos 0), then z (B,H*V) = rmsnorm(core) * norm_weight * SiLU(sum_p gate).
  T4.127b: beta_alpha (B, 2*heads) is ssm_beta(x)/ssm_alpha(x) run as ONE f16 gemv over their concatenated weight
  (model.py's GatedDeltaNetBlock.prepare_nv_decode/nv_f16_gemv_raw) -- beta at column h, alpha at column heads+h,
  plain indexing into the one buffer (no copy kernel).
  T4.132: `tokens` sequential rows (conv_out/beta_alpha/gate/z are (B*tokens, ...)): the state runs through the tokens in registers
  and is stored once; each token's rms reduction uses its own LOCAL slots. track (S,B,H,V,K) <- the state after token t, t < tokens-1."""
  *track, _ = track_bound
  batch, heads, value_dim, key_dim = state.shape
  parts = gate.shape[-1]
  assert all(isinstance(s, int) for s in (batch, heads, value_dim, key_dim, parts)) and heads % k_heads == 0
  batch, heads, value_dim, key_dim, parts = cast(tuple[int, int, int, int, int], (batch, heads, value_dim, key_dim, parts))
  assert key_dim % lanes == 0 and value_dim % row_tile == 0 and value_dim // row_tile <= 32
  waves, q_dim = value_dim // row_tile if lanes > 1 else 1, k_heads*key_dim
  beta_alpha = beta_alpha.reshape(batch*tokens, 2*heads)
  bh, lane = UOp.range(batch*heads, 0), _lane(lanes, 1)
  wave = UOp.range(waves, 2, AxisType.LOCAL) if waves > 1 else UOp.const(0, dtypes.weakint)
  b, h = bh // heads, bh % heads
  row_of = _token_row(b, tokens)
  k_head = h % k_heads  # model.py .repeat()s the k-heads along the head axis: v-head h reads k-head h % k_heads
  cols = tuple(lane + i*lanes for i in range(key_dim//lanes))
  rows = tuple(wave*(value_dim//waves) + i for i in range(value_dim//waves))
  def dot(xs, ys) -> UOp:
    s = sum((x*y for x,y in zip(xs, ys)), UOp.const(0, dtypes.float32))
    return warp_reduce(s, full_wave=True) if lanes > 1 else s
  def l2(xs:tuple[UOp, ...]) -> tuple[UOp, ...]:  # Tensor.normalize: x / max(||x||_2, eps)
    den = dot(xs, xs).sqrt().maximum(qk_eps)
    return tuple(x / den for x in xs)
  initial = start_pos.eq(0)
  previous = {row: tuple(initial.where(0, state[b, h, row, c].load().float()) for c in cols) for row in rows}
  partial = UOp.placeholder((tokens*waves,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL) if waves > 1 else None
  z_stores = []
  for t in range(tokens):
    r = row_of(t)
    q = tuple(x * key_dim**-0.5 for x in l2(tuple(conv_out[r, k_head*key_dim + c].load() for c in cols)))
    k = l2(tuple(conv_out[r, q_dim + k_head*key_dim + c].load() for c in cols))
    kq = dot(q, k)
    bv = beta_alpha[r, h].load().sigmoid().float()
    av = ((beta_alpha[r, heads+h].load().float() + dt_bias[h].load()).softplus() * ssm_a[h].load()).exp()
    outs = []
    for row in rows:  # _gated_delta_prefill_kernel's step at tokens == 1
      prev = previous[row]
      delta = (conv_out[r, 2*q_dim + h*value_dim + row].load() - dot(prev, k)*av) * bv
      previous[row] = tuple(x*av + delta*y for x,y in zip(prev, k))
      outs.append(dot(prev, q)*av + delta*kq)
      if t < tokens-1: z_stores += [tr[t, b, h, row, c].store(x) for tr in track for x,c in zip(previous[row], cols)]
    squares = _added(o*o for o in outs)  # this warp's rows (warp_reduce leaves every lane the same sums)
    if partial is not None:
      slot = (wave if tokens == 1 else t*waves + wave).valid(lane.eq(0))
      partial = partial.after(partial[slot].store(squares).barrier())
      squares = _added(partial[w if tokens == 1 else t*waves + w].load() for w in range(waves))
    inv = (squares * (1/value_dim) + norm_eps).rsqrt()  # nn.RMSNorm: x * (mean(x^2) + eps).rsqrt(), then * weight
    for row,o in zip(rows, outs):
      gated = o*inv*norm_weight[row].load() * _added(gate[r, h*value_dim + row, p].load() for p in range(parts)).silu()
      z_stores.append(z[r, (h*value_dim + row).valid(lane.eq(0))].store(gated.cast(z.dtype)))
  stores = [state[b, h, row, c].store(x.cast(state.dtype)) for row in rows for x,c in zip(previous[row], cols)] + z_stores
  return UOp.group(*stores).end(*(r for r in (wave, lane, bh) if r.op is Ops.RANGE)).sink(arg=KernelInfo(name="gdn_decode_nv", opts_to_apply=()))

def _split_k_partials(t:Tensor) -> Tensor:
  """t (B,1,F) as (B,F,P): nv_q8_linear returns its split-K partials summed lazily (`Tensor(out.after(call)).sum(-1).reshape(...)`), so
  handing a kernel the partials folds that sum's own launch into it. Matched structurally -- the rebuilt sum IS t.uop (hash-consing)
  -- so any other producer (a generic matmul, a biased Linear) comes back as (B,F,1) and is realized as usual."""
  u = t.uop
  for _ in range(6):
    if u.op is Ops.AFTER and len(u.shape) == 3 and Tensor(u).sum(-1).reshape(t.shape).uop is t.uop: return Tensor(u)
    if not u.src: break
    u = u.src[0]
  return t.reshape(-1, t.shape[-1], 1)  # T4.132: (B*T, F, 1) for a T-token run too

def gdn_decode(qkv:Tensor, out_gate:Tensor, beta_alpha:Tensor, conv_state:Tensor, state:Tensor, conv_weight:Tensor, norm_weight:Tensor,
               dt_bias:Tensor, ssm_a:Tensor, start_pos:UOp, k_heads:int, qk_eps:float, norm_eps:float,
               track:tuple[Tensor, Tensor]|None=None) -> Tensor:
  """One GatedDeltaNet decode token (T == 1) -- T4.132: or a run of T <= GDN_DECODE_TOKENS tokens, sequential inside the kernels -- from
  the projections: qkv = attn_qkv(x), out_gate = attn_gate(x), beta_alpha = ssm_beta(x) and ssm_alpha(x) run as ONE f16 gemv (T4.127b,
  model.py's prepare_nv_decode/nv_f16_gemv_raw) over their concatenated weight -- (B,T,2*heads), beta at [:heads], alpha at [heads:].
  Updates conv_state and state in place; returns z (B,T,H*V) fp16, ssm_out's input. The four weights must be realized buffers
  (GatedDeltaNetBlock.prepare_nv_decode): lazy GGUF byte views would be re-copied every step. T4.132: track = (conv (S,B,taps-1,C),
  state (S,B,H,V,K)) buffers also receive the conv window / recurrent state after each of the first T-1 tokens (S >= T-1)."""
  batch, heads, value_dim, _ = state.shape
  tokens = cast(int, qkv.shape[1])
  assert 1 <= tokens <= GDN_DECODE_TOKENS, f"{tokens=}"
  channels, device = conv_state.shape[2], conv_state.device
  lanes = 32 if isinstance(device, str) and device.split(":")[0] == "NV" else 1  # off NV only the CPU tests get here
  assert start_pos.is_bound_var
  # the kernels read start_pos as a PARAM by name (kernel_var); the bound variable itself rides along as the LAST CALL src (`_bound`,
  # never loaded) so the schedule binds it -- a precompiled @function drops arguments no kernel references, and the generic graph
  # this replaces got the binding from its conv-window kernel. T4.132: last -- it is no buffer slot, so a track buffer after it went unbound
  var, bound = kernel_var(start_pos.src[0]), Tensor(start_pos)
  conv_out = Tensor.empty(batch*tokens, channels, dtype=dtypes.float32, device=device)
  conv_track, state_track = ((track[0],), (track[1],)) if track is not None and tokens > 1 else ((), ())
  conv_out = Tensor.custom_kernel(conv_out, conv_state, _split_k_partials(qkv), conv_weight, *conv_track, bound,
                                  fxn=functools.partial(_gdn_conv_decode_kernel, start_pos=var, lanes=lanes, tokens=tokens))[0]
  z = Tensor.empty(batch*tokens, heads*value_dim, dtype=dtypes.float16, device=device)
  z = Tensor.custom_kernel(z, state, conv_out, beta_alpha.reshape(batch*tokens, 2*heads), dt_bias, ssm_a,
                           _split_k_partials(out_gate), norm_weight, *state_track, bound, fxn=functools.partial(_gdn_decode_kernel, start_pos=var, k_heads=k_heads,
                           qk_eps=qk_eps, norm_eps=norm_eps, lanes=lanes, tokens=tokens))[0]
  return z.reshape(batch, tokens, heads*value_dim)
