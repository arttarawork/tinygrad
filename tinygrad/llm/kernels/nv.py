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
# T4.146: gdn_decode folds the beta_alpha f16 gemv into its conv kernel (1, default) or runs nv_dense's own launch (0, A/B) -- bit-identical
GDN_DECODE_GEMV = ContextVar("GDN_DECODE_GEMV", 1)
# T4.146: gdn_decode_nv also emits ssm_out's q8 activations (1, default; read in model.py) or ssm_out quantizes z itself (0, A/B) -- bit-identical
GDN_DECODE_Q8 = ContextVar("GDN_DECODE_Q8", 1)

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

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False, offsets:tuple[int, ...]|None=None) -> UOp:
  # T4.138: `offsets` reduces a sub-warp group (nv_quant's prefill quantizer: (4, 2, 1) over 8 lanes); the lane must be the hardware lane
  for offset in (offsets if offsets is not None else (16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

# verbatim amd._gated_delta_prefill_kernel except for warp_reduce and the kernel name (kept separate so the AMD kernel stays untouched).
# T4.138: lanes=1 is the CPU form (one thread owns a row tile's whole key dim, no shuffle -- gdn_decode's convention) for the CPU tests of
# nv_gdn_prep.gdn_prefill; lanes=32 builds exactly the graph it always did.
@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None,
                                lanes:int=32) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % lanes == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), _lane(lanes, 1)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*lanes for i in range(key_dim//lanes))
  current = UOp.placeholder((row_tile*key_dim//lanes,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float()) for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.REDUCE)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  reduce = (lambda x: warp_reduce(x, full_wave=True)) if lanes > 1 else (lambda x: x)
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//lanes+i].load() for i in range(key_dim//lanes))
    av, bv = alpha[bh, token, row if alpha_dim > 1 else 0].load(), beta[bh, token].load()
    state_k = reduce(sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32)))
    state_q = reduce(sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32)))
    delta = (v[bh, token, row].load() - state_k*av) * bv
    updates += [x*av + delta*y for x,y in zip(previous, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q*av + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//lanes+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(*(r for r in (lane, bh_row) if r.op is Ops.RANGE)).sink(
    arg=KernelInfo(name="gated_delta_prefill_nv", opts_to_apply=()))

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
#     block overwriting conv_state rows that the others still read needs a grid-wide barrier. (T4.146 re-checked the alternatives: one
#     block per k-head puts the 6.3 MB state stream on 16 SMs; a last-arriver atomic counter needs atomics + fences the codegen and the
#     PYTHON emulator don't model; a ring-buffered conv_state changes the state-cache layout.) T4.146: it also runs the beta_alpha f16 gemv
#     (gemv=True), one launch fewer per GDN block.
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

def f16_dot_lane(w:UOp, x:UOp, row:UOp, tok:UOp, lane:UOp|int, per:int) -> UOp:
  """lane `lane`'s fp32 sum of nv_dense's one-warp f16 gemv: w (rows, per, 128), x (tokens, per, 128) fp16, the lane's 4 halves of each
  128-wide step, i-major j-minor from 0 -- T4.146: shared by nv_dense's 1 x 1 kernel and _gdn_conv_decode_kernel's folded gemv"""
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(4): acc = acc + w[row, i, lane*4+j].load().float() * x[tok, i, lane*4+j].load().float()
  return acc

def _lane_tree(vals:list[UOp]) -> UOp:  # warp_reduce(full_wave)'s butterfly over 32 lane values for the CPU form: v[l] + v[l ^ off]
  for off in (16, 8, 4, 2, 1): vals = [vals[l] + vals[l ^ off] for l in range(32)]
  return vals[0]

@functools.cache
def _gdn_conv_decode_kernel(conv_out:UOp, conv_state:UOp, qkv:UOp, weight:UOp, *rest:UOp, start_pos:UOp, lanes:int=32,
                            tokens:int=1, gemv:bool=False) -> UOp:
  """conv_out (B*tokens,C) fp32: row (b,t) = SiLU(sum_i weight[c,i] * window[t+i]) with window = [conv_state rows (zero at start_pos 0) |
  sum_p qkv[(b,t),c,p] per token]; conv_state (B,taps-1,C) <- window[tokens:] in place. T4.132: track (S,B,taps-1,C) <- window[t+1:t+taps]
  for t < tokens-1, the conv state after token t. The bound start_pos rides last (see gdn_decode): a scalar src, not a buffer slot.
  T4.146: gemv=True also runs model.py's beta_alpha gemv (ssm_beta | ssm_alpha over their concatenated fp16 weight, T4.127b) in this
  launch: rest starts with beta_alpha (B*tokens, 2H) fp16 out, x (B*tokens, D) fp16 and the weight (2H, D) fp16. Block i's warp also forms
  output i (B*tokens*2H <= blocks: 96 / 288 of 320 at 27B geometry, 1 / 3 tokens), exactly nv_dense's 1 x 1 gemv warp (f16_dot_lane + the
  5-shuffle butterfly + fp16), so beta_alpha is the separate linear_f16_gemv_nv launch's bit for bit; surplus blocks redo the last output
  (the same bits to the same address). Its 80 half4 loads per lane stay hoisted: 188-196 registers of the one-warp block's 255, no spill
  (nvrtc lane, sm_86; two outputs per block -- 4 tokens -- spilled 344 B, so gdn_decode keeps the separate gemv there). Not the conv
  itself into gdn_decode: see the section comment."""
  beta_alpha = xh = ba_w = None
  if gemv: (beta_alpha, xh, ba_w), rest = rest[:3], rest[3:]
  *track, _ = rest
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
  if gemv:  # T4.146: see the docstring
    assert beta_alpha is not None and xh is not None and ba_w is not None  # the gemv's optional srcs (mypy)
    rows, out_features, dim = cast(int, beta_alpha.shape[0]), cast(int, beta_alpha.shape[1]), cast(int, xh.shape[-1])
    per, n_out = dim // 128, rows * out_features
    assert dim % 128 == 0 and ba_w.shape == (out_features * dim,) and n_out <= batch*channels//lanes
    w3, x3 = ba_w.reshape(out_features, per, 128), xh.reshape(rows, per, 128)
    o = block if n_out == batch*channels//lanes else block.minimum(n_out - 1)
    tok, row = o // out_features, o % out_features
    total = warp_reduce(f16_dot_lane(w3, x3, row, tok, lane, per), full_wave=True) if lanes > 1 else \
      _lane_tree([f16_dot_lane(w3, x3, row, tok, ln, per) for ln in range(32)])
    shifts.append(beta_alpha[tok, row.valid(lane.eq(0))].store(total.cast(beta_alpha.dtype)))
  ends = tuple(r for r in (lane, block) if r.op is Ops.RANGE)
  return UOp.group(*outs, *shifts).end(*ends).sink(arg=KernelInfo(name="gdn_conv_decode_nv", opts_to_apply=()))

@functools.cache
def _gdn_decode_kernel(z:UOp, state:UOp, conv_out:UOp, beta_alpha:UOp, dt_bias:UOp, ssm_a:UOp, gate:UOp, norm_weight:UOp, *rest:UOp,
                       start_pos:UOp, k_heads:int, qk_eps:float, norm_eps:float, lanes:int=32, row_tile:int=8, tokens:int=1, q8:bool=False) -> UOp:
  """model.py's decode graph from conv_out on, per (b, v-head h): q/k = conv_out's k-head h % k_heads l2-normalized (q * key_dim^-0.5),
  v = conv_out's v channels, beta = sigmoid(beta) in its own dtype, alpha = exp(softplus(alpha + dt_bias) * ssm_a), the fused scan's
  one-token step on state (B,H,V,K) in place (zeroed at start_pos 0), then z (B,H*V) = rmsnorm(core) * norm_weight * SiLU(sum_p gate).
  T4.127b: beta_alpha (B, 2*heads) is ssm_beta(x)/ssm_alpha(x) run as ONE f16 gemv over their concatenated weight
  (model.py's GatedDeltaNetBlock.prepare_nv_decode/nv_f16_gemv_raw) -- beta at column h, alpha at column heads+h,
  plain indexing into the one buffer (no copy kernel).
  T4.132: `tokens` sequential rows (conv_out/beta_alpha/gate/z are (B*tokens, ...)): the state runs through the tokens in registers
  and is stored once; each token's rms reduction uses its own LOCAL slots. track (S,B,H,V,K) <- the state after token t, t < tokens-1.
  T4.146: q8=True (card form, value_dim % 32 == 0) also writes ssm_out's q8 activations of z -- rest starts with xq (B*tokens, H*V/32, 8) uint32 and
  xd (B*tokens, H*V/32) fp32, q8_quantize(z)'s decode layout. A q8 group (32 rows of a head) spans 32/row_tile warps, so each token's fp16
  z rows are staged in shared memory behind a second barrier; then warp g quantizes the head's group g exactly as q8_quantize_nv's decode
  kernel does (the max over its 32 lanes, lanes < 8 pack a word -- surplus warps redo the last group: the same bits, the same addresses),
  from the fp16 values (a load in between: the codegen folds a cast(half).float() pair back to fp32). Bit-identical to quantizing z."""
  xq = xd = None
  if q8: (xq, xd), rest = rest[:2], rest[2:]
  *track, _ = rest
  batch, heads, value_dim, key_dim = state.shape
  parts = gate.shape[-1]
  assert all(isinstance(s, int) for s in (batch, heads, value_dim, key_dim, parts)) and heads % k_heads == 0
  batch, heads, value_dim, key_dim, parts = cast(tuple[int, int, int, int, int], (batch, heads, value_dim, key_dim, parts))
  assert key_dim % lanes == 0 and value_dim % row_tile == 0 and value_dim // row_tile <= 32 and (not q8 or (lanes > 1 and value_dim % 32 == 0))
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
  z_stores: list[UOp] = []
  for t in range(tokens):
    r = row_of(t)
    # T4.140: token t reads its inputs only after token t-1's stores. Unordered, the linearizer hoisted every load to the top and sank
    # every store to the bottom, so all T tokens' gate partials (40 floats per token), outputs and track rows were live at once: under
    # the 128-register cap of 512 threads ptxas (nvrtc lane, sm_86, with track) needed stack frames of 0 / 304 / 584 / 1368 B per thread
    # at T = 1 / 2 / 3 / 4 (~4 KB of spill traffic per thread at T = 4, ~100 MB per launch: the card's 14.5 / 51.6 / 130 us). Ordered,
    # one token's inputs are live at a time: 0 / 24 / 0 / 0 B, 128 registers. The arithmetic is untouched (bit-identical on the
    # emulator); t == 0 has no stores before it, so T == 1 keeps its exact graph (md5-pinned).
    co, ba, gt = (x.after(*z_stores) for x in (conv_out, beta_alpha, gate))
    q = tuple(x * key_dim**-0.5 for x in l2(tuple(co[r, k_head*key_dim + c].load() for c in cols)))
    k = l2(tuple(co[r, q_dim + k_head*key_dim + c].load() for c in cols))
    kq = dot(q, k)
    bv = ba[r, h].load().sigmoid().float()
    av = ((ba[r, heads+h].load().float() + dt_bias[h].load()).softplus() * ssm_a[h].load()).exp()
    outs = []
    for row in rows:  # _gated_delta_prefill_kernel's step at tokens == 1
      prev = previous[row]
      delta = (co[r, 2*q_dim + h*value_dim + row].load() - dot(prev, k)*av) * bv
      previous[row] = tuple(x*av + delta*y for x,y in zip(prev, k))
      outs.append(dot(prev, q)*av + delta*kq)
      if t < tokens-1: z_stores += [tr[t, b, h, row, c].store(x) for tr in track for x,c in zip(previous[row], cols)]
    squares = _added(o*o for o in outs)  # this warp's rows (warp_reduce leaves every lane the same sums)
    if partial is not None:
      slot = (wave if tokens == 1 else t*waves + wave).valid(lane.eq(0))
      partial = partial.after(partial[slot].store(squares).barrier())
      squares = _added(partial[w if tokens == 1 else t*waves + w].load() for w in range(waves))
    inv = (squares * (1/value_dim) + norm_eps).rsqrt()  # nn.RMSNorm: x * (mean(x^2) + eps).rsqrt(), then * weight
    zh = [(o*inv*norm_weight[row].load() * _added(gt[r, h*value_dim + row, p].load() for p in range(parts)).silu()).cast(z.dtype)
          for row,o in zip(rows, outs)]
    z_stores += [z[r, (h*value_dim + row).valid(lane.eq(0))].store(x) for row, x in zip(rows, zh)]
    if q8:
      assert xq is not None and xd is not None  # the q8 outputs' optional srcs (mypy)
      z_stores += _z_q8(xq, xd, rows, zh, t, r, h, value_dim, tokens, lane, wave, waves)
  stores = [state[b, h, row, c].store(x.cast(state.dtype)) for row in rows for x,c in zip(previous[row], cols)] + z_stores
  return UOp.group(*stores).end(*(r for r in (wave, lane, bh) if r.op is Ops.RANGE)).sink(arg=KernelInfo(name="gdn_decode_nv", opts_to_apply=()))

def _z_q8(xq:UOp, xd:UOp, rows, zh:list[UOp], t:int, r:UOp, h:UOp, value_dim:int, tokens:int, lane:UOp, wave:UOp, waves:int) -> list[UOp]:
  """T4.146: _gdn_decode_kernel's q8 epilogue for token t (see its docstring): the stores of z's q8 words and scales. The card form only:
  in the CPU form one thread owns a head, and its unrolled 32-wide group maxima and packs made the scheduler of a whole speculative
  iteration (test_llm_nv_quant's census) run for minutes -- gdn_decode leaves ssm_out's own quantize there."""
  from tinygrad.llm.kernels.nv_quant import _q8_pack  # nv_quant imports this module
  groups = value_dim // 32
  zbuf = UOp.placeholder((tokens*value_dim,), dtypes.float16, slot=1, addrspace=AddrSpace.LOCAL)
  zb = zbuf.after(UOp.group(*(zbuf[(t*value_dim + row).valid(lane.eq(0))].store(x) for row, x in zip(rows, zh))).barrier())
  g = wave.minimum(groups - 1) if waves > groups else wave
  gscale = (warp_reduce(zb[t*value_dim + g*32 + lane].load().float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word = _q8_pack(tuple(zb[t*value_dim + g*32 + lane.minimum(7)*4 + i].load().float() for i in range(4)), gscale)
  return [xq[r, h*groups + g, lane.valid(lane < 8)].store(word), xd[r, (h*groups + g).valid(lane.eq(0))].store(gscale)]

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

def _decode_lanes(device) -> int: return 32 if isinstance(device, str) and device.split(":")[0] == "NV" else 1  # off NV only the CPU tests

def gdn_decode(qkv:Tensor, out_gate:Tensor, beta_alpha:Tensor|tuple[Tensor, Tensor], conv_state:Tensor, state:Tensor, conv_weight:Tensor,
               norm_weight:Tensor, dt_bias:Tensor, ssm_a:Tensor, start_pos:UOp, k_heads:int, qk_eps:float, norm_eps:float,
               track:tuple[Tensor, Tensor]|None=None, q8:bool=False) -> Tensor|tuple[Tensor, Tensor, Tensor]:
  """One GatedDeltaNet decode token (T == 1) -- T4.132: or a run of T <= GDN_DECODE_TOKENS tokens, sequential inside the kernels -- from
  the projections: qkv = attn_qkv(x), out_gate = attn_gate(x), beta_alpha = ssm_beta(x) and ssm_alpha(x) run as ONE f16 gemv (T4.127b,
  model.py's prepare_nv_decode/nv_f16_gemv_raw) over their concatenated weight -- (B,T,2*heads), beta at [:heads], alpha at [heads:].
  Updates conv_state and state in place; returns z (B,T,H*V) fp16, ssm_out's input. The four weights must be realized buffers
  (GatedDeltaNetBlock.prepare_nv_decode): lazy GGUF byte views would be re-copied every step. T4.132: track = (conv (S,B,taps-1,C),
  state (S,B,H,V,K)) buffers also receive the conv window / recurrent state after each of the first T-1 tokens (S >= T-1).
  T4.146: beta_alpha may be the gemv's inputs instead, (x (B*T, D) fp16, the concatenated weight (2H, D) fp16): the conv kernel forms it
  (GDN_DECODE_GEMV, one output per conv block), else nv_dense's linear_f16_gemv_nv launch. q8=True (head_v_dim % 32 == 0) returns
  (z, xq, xd) -- z's q8 activations from the scan kernel's epilogue, for register_fused_q8 (ssm_out's input) -- on the card; the CPU
  form returns z alone (see _z_q8)."""
  batch, heads, value_dim, _ = state.shape
  tokens = cast(int, qkv.shape[1])
  assert 1 <= tokens <= GDN_DECODE_TOKENS, f"{tokens=}"
  channels, device = conv_state.shape[2], conv_state.device
  lanes = _decode_lanes(device)
  assert start_pos.is_bound_var
  # the kernels read start_pos as a PARAM by name (kernel_var); the bound variable itself rides along as the LAST CALL src (`_bound`,
  # never loaded) so the schedule binds it -- a precompiled @function drops arguments no kernel references, and the generic graph
  # this replaces got the binding from its conv-window kernel. T4.132: last -- it is no buffer slot, so a track buffer after it went unbound
  var, bound = kernel_var(start_pos.src[0]), Tensor(start_pos)
  conv_out = Tensor.empty(batch*tokens, channels, dtype=dtypes.float32, device=device)
  conv_track, state_track = ((track[0],), (track[1],)) if track is not None and tokens > 1 else ((), ())
  gemv: tuple[Tensor, ...] = ()
  if isinstance(beta_alpha, tuple):  # T4.146: the gemv folded into the conv kernel (one output per block); its x as nv_f16_gemv_raw takes it
    x, w = beta_alpha
    if GDN_DECODE_GEMV.value and batch*tokens*2*heads <= batch*channels//lanes:
      gemv = (Tensor.empty(batch*tokens, 2*heads, dtype=dtypes.float16, device=device), x if x.uop.has_buffer_identity(after_ok=True) else
              x.contiguous(), w.reshape(-1))
    else:
      from tinygrad.llm.kernels.nv_dense import nv_f16_gemv_raw  # nv_dense imports this module
      beta_alpha = nv_f16_gemv_raw(w, x, 2*cast(int, heads))
  outs = Tensor.custom_kernel(conv_out, conv_state, _split_k_partials(qkv), conv_weight, *gemv, *conv_track, bound,
                              fxn=functools.partial(_gdn_conv_decode_kernel, start_pos=var, lanes=lanes, tokens=tokens, gemv=bool(gemv)))
  conv_out, beta_alpha = outs[0], outs[4] if gemv else cast(Tensor, beta_alpha)
  z = Tensor.empty(batch*tokens, heads*value_dim, dtype=dtypes.float16, device=device)
  assert not q8 or value_dim % 32 == 0, f"q8 groups must not span heads ({value_dim=})"
  q8 = q8 and lanes > 1
  xq8 = (Tensor.empty(batch*tokens, heads*value_dim//32, 8, dtype=dtypes.uint32, device=device),
         Tensor.empty(batch*tokens, heads*value_dim//32, dtype=dtypes.float32, device=device)) if q8 else ()
  res = Tensor.custom_kernel(z, state, conv_out, beta_alpha.reshape(batch*tokens, 2*heads), dt_bias, ssm_a,
                             _split_k_partials(out_gate), norm_weight, *xq8, *state_track, bound,
                             fxn=functools.partial(_gdn_decode_kernel, start_pos=var, k_heads=k_heads, qk_eps=qk_eps, norm_eps=norm_eps, lanes=lanes,
                                                   tokens=tokens, q8=q8))
  z = res[0].reshape(batch, tokens, heads*value_dim)
  return (z, res[8], res[9]) if q8 else z
