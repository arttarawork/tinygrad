"""T4.138: the GatedDeltaNet PREFILL chunk's generic glue as NV kernels (the decode step has kernels/nv.py's gdn_decode since T4.123f+g).

Step 0 -- the map. Every generic kernel of one 128-token prefill chunk, identified from the schedule of ONE GatedDeltaNetBlock call at the
27B geometry (dim 5120, hidden 17408, 48 v-heads / 16 k-heads of 128, conv 4 taps over 10240 channels; symbolic `toks` <= 128) with the
NV gates mocked open on CPU, each generic kernel named by the sm_86 hand-coded optimizer -- the names the card's 10-31 profile shows
(logs/t4136/prof-2048/summary.txt; test_llm_nv_gdn_prep.py's census pins this list). ms/chunk = launches x us from that profile; bytes
at 128 tokens; floor = bytes / 936 GB/s, or the ~5 us launch floor.

 kernel (launches/chunk x us)   ms    op in model.py (GatedDeltaNetBlock._attention unless noted)             bytes moved         floor
 -- pre-scan (between attn_qkv/attn_gate/ssm_beta/ssm_alpha and gated_delta_prefill_nv) --------------------------------------------
 r_2_8_8_16_3_32_4  (48 x 124)  5.93  kq = (q*k).sum(-1) in nv.gated_delta_prefill: q, k RECOMPUTED inline per v-head      ~2.2 MB   5 us
                                      from the conv window (4-tap conv + SiLU, / the two norms below, * K^-0.5)
 r_128_16_16_8      (96 x 40)   3.85  the q and k L2 norms (sum of SiLU(conv)^2 over 128 per token x k-head), conv      1.1 MB each  5 us
                                      recomputed from the window
 E_16_16_8_8_16_3   (96 x 12)   1.17  q (* K^-0.5) and k normalized, .repeat'ed to 48 v-heads, materialized (1,48,T,128)  4.2 MB each  4.5 us
 E_48_2_16_16_8_4   (48 x 18)   0.85  v = SiLU(conv) of the 6144 v channels, materialized (1,48,T,128) fp32               6.3 MB    6.8 us
 E_28toks2A8029_128 (48 x 15)   0.73  attn_qkv's output (toks x 10240 fp32) copied into the window win[:, 3:3+T]          10.5 MB   11 us
 E_6_2_8_16_4       (96 x 7)    0.65  beta = sigmoid(ssm_beta(x)), alpha = exp(softplus(ssm_alpha(x)+dt_bias)*ssm_a),     6144 el.  5 us
                                      padded past T (beta 0, alpha 1)
 E_320_32_3         (96 x 6)    0.59  conv_state -> win[:, :3] (fp16 -> fp32), and conv_state <- win[:, T:T+3]           0.2 MB    5 us
 E_10480_32_4       (pruned)    --    the window's zero fill (1,131,10240): a write-only init, so the pruned prefill family runs it ONCE
                                      at capture -- the padded rows of a remainder chunk then read whatever the arena left there
 linear_f16_gemv_nv (98.6 x 47) 4.60  ssm_beta, ssm_alpha (5120 -> 48 fp16) at 128 tokens: one warp per (token, row)    1.8 MB DRAM  2 us
                                      re-reads a 10 KB weight row + 10 KB token row -- 126 MB of L2 traffic per launch
 E_5120_32_4        (49 x 9)    0.44  nv_f16_gemv's .contiguous() copy of the padded fp16 x (its input)                    2.6 MB    2.8 us
 sum, GDN pre-scan glue: ~287 us per block = 13.8 ms/chunk (the two f16 gemvs + the copy: another 4.8 ms)
 -- post-scan (gated_delta_prefill_nv's core -> ssm_out) -----------------------------------------------------------------------------
 r_8_48_8_4_4_16    (48 x 10)   0.47  ssm_norm's sum of squares over V per (token, v-head)                                3.1 MB    3.4 us
 E_32_6_2_8_16_4_4  (48 x 14)   0.65  z = ssm_norm(core) * SiLU(attn_gate) -> fp16                                        7.9 MB    8.4 us
 E_16_96_8_16_4     (48 x 7)    0.35  z[:, :T] padded back to 128 rows for ssm_out's q8 quantize (nv_wmma_linear)         3.1 MB    3.4 us
 q8_quantize_nv     (48 of 263) --    ssm_out's q8 activations (Q5_K on the card: + the group sums)                        2.6 MB    2.8 us
 -- every block's norm sites, quantizers and FFN (not this task's steps 1/2) --------------------------------------------------------
 q8_quantize_nv     (263 x 17)  4.57  q8 activations of every IMMA gemm, 4 per block: the attn input (5120 fp16), ssm_out/ 2.1/2.5/   avg
                                      attn_output (6144/... fp16), ffn input (5120 fp32), ffn_down (17408 fp32)        3.4/11.4 MB 5.2 us
 r_toks_16_320      (128 x 19)  2.37  RMSNorm sum of squares over 5120 per token: attn_norm and ffn_norm (FFNBlock)       2.6-5.2 MB 3-6 us
 E_4_80_8_16_4_4    (130 x 10)  1.26  RMSNorm apply + pad to 128 rows (the q8 input): attn_norm -> fp16 (GDN, also the    2.6-5.2 MB 3-6 us
                                      f16 gemv's input), ffn_norm -> fp32 with h = x + attn_out recomputed
 E_16_272_8_16_4    (65 x 37)   2.39  SiLU(gate) * up padded to 128 rows for ffn_down's q8 quantize (FFNBlock)            26.7 MB   28.5 us
 E_28toks2A13629_128 (64 x 25)  1.57  ffn_gate(x).silu().contiguous() (_feed_forward's "TODO: remove the need for this     17.8 MB   19 us
                                      contiguous")
 E_28toks2A4029_128 (65 x 15)   0.97  the block output x + attn_out + ffn_out (.contiguous())                              10.5 MB   11 us
"""
from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.amd import kernel_var
from tinygrad.llm.kernels.nv import _gated_delta_prefill_kernel
from tinygrad.llm.kernels.nv_quant import _q8_pack

# ******** step 1: the pre-scan glue as two kernels ********
# gdn_prep_nv -- one warp per (batch, tile of PREP_TILE tokens, k-head): its lanes own the k-head's 128 q and 128 k channels and the
#   v channels of the three v-heads that read it (model.py .repeat()s k-head h % k_heads), 4 + 4 + 12 per lane. Per token of the tile:
#   the 4-tap conv over the window [conv_state | qkv rows < toks | zero] (taps in the generic graph's order) + SiLU, q and k
#   l2-normalized (q * K^-0.5) by warp reductions, kq = q.k, beta = sigmoid(beta_proj), alpha = exp(softplus(alpha_proj + dt_bias) *
#   ssm_a) (beta 0 / alpha 1 past toks: exact no-op steps, the generic pad) -- stored as gated_delta_prefill_nv's inputs: q, k (each
#   v-head's copy), v, kq, beta, alpha. Reads qkv once (5.2 MB) and writes 9.4 MB: ~16 us of DRAM at 128 tokens for what took ~287 us.
#   No window buffer: the zero padding past toks is a select on the toks variable, so no one-time zero fill can go stale.
# gdn_conv_state_nv -- conv_state <- the window rows toks..toks+K-2, one thread per channel (reads its old rows, then writes them). Its
#   own kernel because the prep kernel's tile 0 reads the old rows: a store gated to tile 0 does not survive the index lowering (a gate
#   clause whose ranges are not in the gated component is dropped -- symbolic.py drop_and_clauses), and any other writer would race the
#   reads. Ordered after the prep kernel by taking its conv_state output; the scan's state input carries it into the graph.
PREP_TILE = 4  # tokens per prep warp: >= conv_kernel-1 (only tile 0 reads conv_state); a chunk it doesn't divide is one tile
FULL_WARP = (16, 8, 4, 2, 1)

def lanes_for(device:str|tuple[str, ...]|None) -> int:
  return 32 if isinstance(device, str) and device.split(":")[0] == "NV" else 1  # off NV only the CPU tests get here (lanes=1, nv.py's form)

def _warp_lane(lanes:int) -> UOp: return UOp.range(lanes, -1, AxisType.WARP) if lanes > 1 else UOp.const(0, dtypes.weakint)

def xor_reduce(val:UOp, offsets:tuple[int, ...], maximum:bool=False) -> UOp:
  """butterfly over hardware lanes lane ^ off (the lane is an AxisType.WARP range: threadIdx.x, never split by the lowerer). Every lane
  ends with the same bits: each step is a+b (or max) on both partners, commutative -- so a store of the result needs no lane gate."""
  for off in offsets:
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {off})", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _added(xs) -> UOp: return functools.reduce(lambda a, b: a + b, xs)  # left to right, the order the generic graph sums in

@functools.cache
def _gdn_prep_kernel(q:UOp, k:UOp, v:UOp, kq:UOp, beta:UOp, alpha:UOp, conv_state:UOp, qkv:UOp, beta_alpha:UOp, conv_w:UOp, dt_bias:UOp,
                     ssm_a:UOp, *bound:UOp, start_pos:UOp, toks:UOp|int, k_heads:int, qk_eps:float, lanes:int=32, tile:int=PREP_TILE) -> UOp:
  """q/k (B,H,T,K), v (B,H,T,V), kq/beta/alpha (B,H,T) fp32 out; conv_state (B,taps-1,C) read; qkv (B,T,C) = attn_qkv(x) at the padded
  chunk width T; beta_alpha (B,T,2H) fp16 = ssm_beta(x) | ssm_alpha(x); conv_w (C,taps), dt_bias (H,), ssm_a (H,) fp32. The bound
  start_pos (and toks, when symbolic) ride last (gdn_decode's convention): scalar srcs, never loaded."""
  batch, heads, tokens, key_dim = cast(tuple[int, int, int, int], q.shape)
  value_dim, (_, history, channels) = cast(int, v.shape[-1]), cast(tuple[int, int, int], conv_state.shape)
  q_dim, rep, tiles = k_heads*key_dim, heads // k_heads, tokens // tile
  assert tokens % tile == 0 and (tile >= history or tiles == 1) and heads % k_heads == 0 and channels == 2*q_dim + heads*value_dim
  assert key_dim % lanes == 0 and value_dim % lanes == 0
  bt, kh, lane = UOp.range(batch*tiles, 0), UOp.range(k_heads, 1), _warp_lane(lanes)
  b, t0 = bt // tiles, (bt % tiles) * tile
  first, initial = (bt % tiles).eq(0), start_pos.eq(0)
  cols = tuple(lane + i*lanes for i in range(key_dim // lanes))
  rows = tuple(lane + i*lanes for i in range(value_dim // lanes))
  vheads = tuple(kh + j*k_heads for j in range(rep))  # model.py .repeat()s the k-heads: v-head h reads k-head h % k_heads
  chans = [kh*key_dim + c for c in cols] + [q_dim + kh*key_dim + c for c in cols] + [2*q_dim + h*value_dim + r for h in vheads for r in rows]
  window: dict[tuple[int, int], UOp] = {}
  def row(ci:int, j:int) -> UOp:  # window row t0+j of channel chans[ci]: [conv_state (zero at start_pos 0) | qkv rows < toks | zero]
    if (ci, j) not in window:
      c, r = chans[ci], t0 + (j - history)
      real = (r < toks).where(qkv[b, r.maximum(0) if j < history else r, c].load().float(), 0.0)
      # the history rows: tile 0 only (tile >= history). The load gate may not survive lowering (an in-bounds read either way; the select
      # drops it) -- gdn_conv_state_nv, which overwrites these rows, runs after this kernel
      window[(ci, j)] = real if j >= history else first.where(initial.where(0.0, conv_state[b, j, c.valid(first)].load().float()), real)
    return window[(ci, j)]
  taps = [tuple(conv_w[c, i].load() for i in range(history + 1)) for c in chans]
  def dot(xs, ys) -> UOp:
    s = _added(x*y for x, y in zip(xs, ys))
    return xor_reduce(s, FULL_WARP) if lanes > 1 else s
  def l2(xs:list[UOp]) -> tuple[UOp, ...]:  # Tensor.normalize: x / max(||x||_2, eps)
    den = dot(xs, xs).sqrt().maximum(qk_eps)
    return tuple(x / den for x in xs)
  nq, nv = len(cols), len(rows)
  stores = []
  for tl in range(tile):
    t = t0 + tl
    conv = [_added(row(ci, tl + i) * w for i, w in enumerate(taps[ci])).silu() for ci in range(len(chans))]
    qn = tuple(x * key_dim**-0.5 for x in l2(conv[:nq]))
    kn = l2(conv[nq:2*nq])
    kqv, ok = dot(qn, kn), t < toks
    for j, h in enumerate(vheads):
      stores += [q[b, h, t, c].store(x) for c, x in zip(cols, qn)] + [k[b, h, t, c].store(x) for c, x in zip(cols, kn)]
      stores += [v[b, h, t, r].store(x) for r, x in zip(rows, conv[2*nq + j*nv:2*nq + (j+1)*nv])]
      bv = ok.where(beta_alpha[b, t, h].load().sigmoid().float(), 0.0)
      av = ok.where(((beta_alpha[b, t, heads + h].load().float() + dt_bias[h].load()).softplus() * ssm_a[h].load()).exp(), 1.0)
      stores += [kq[b, h, t].store(kqv), beta[b, h, t].store(bv), alpha[b, h, t].store(av)]  # the same bits in every lane: no gate
  ends = tuple(r for r in (lane, kh, bt) if r.op is Ops.RANGE)
  return UOp.group(*stores).end(*ends).sink(arg=KernelInfo(name="gdn_prep_nv", opts_to_apply=()))

@functools.cache
def _gdn_conv_state_kernel(conv_state:UOp, qkv:UOp, *bound:UOp, start_pos:UOp, toks:UOp|int, lanes:int=32) -> UOp:
  """conv_state (B,taps-1,C) <- window rows toks..toks+taps-2 of [conv_state (zero at start_pos 0) | qkv rows]: the generic graph's
  conv_window[:, T:T+taps-1] store. One thread per channel: its old rows are loaded before any of its stores (every stored value is a
  select over those loads, and the store goes through conv_state.after(them))."""
  batch, history, channels = cast(tuple[int, int, int], conv_state.shape)
  assert channels % lanes == 0
  blk, lane = UOp.range(batch*channels // lanes, 0), _warp_lane(lanes)
  b, c = blk // (channels // lanes), (blk % (channels // lanes))*lanes + lane
  initial = start_pos.eq(0)
  old = [initial.where(0.0, conv_state[b, m, c].load().float()) for m in range(history)]
  new = []
  for i in range(history):
    w = toks + i  # the window row that becomes row i; toks >= 1, so w > i: never a row this thread already stored
    if isinstance(w, int): new.append(old[w] if w < history else qkv[b, w - history, c].load().float())
    else: new.append(functools.reduce(lambda acc, m: w.eq(m).where(old[m], acc), range(history), qkv[b, (w - history).maximum(0), c].load().float()))
  stores = [conv_state.after(*old)[b, i, c].store(x.cast(conv_state.dtype)) for i, x in enumerate(new)]
  ends = tuple(r for r in (lane, blk) if r.op is Ops.RANGE)
  return UOp.group(*stores).end(*ends).sink(arg=KernelInfo(name="gdn_conv_state_nv", opts_to_apply=()))

def _call(fxn, *srcs:UOp) -> UOp: return fxn(*(UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs))).call(*srcs)

def gdn_prep(qkv:Tensor, beta_alpha:Tensor, conv_state:Tensor, conv_w:Tensor, dt_bias:Tensor, ssm_a:Tensor, start_pos:UOp, toks:int|UOp, heads:int,
             key_dim:int, value_dim:int, k_heads:int, qk_eps:float) -> tuple[tuple[Tensor, ...], UOp]:
  """The two pre-scan kernels: ((q, k, v, kq, beta, alpha) -- gated_delta_prefill_nv's inputs --, the conv_state store's CALL). Nothing
  reads conv_state downstream, so a caller must carry that CALL into its graph (gdn_prefill hangs it on the scan's state input)."""
  batch, tokens, _ = cast(tuple[int, int, int], qkv.shape)
  device, lanes, history = conv_state.device, lanes_for(conv_state.device), cast(int, conv_state.shape[1])
  assert start_pos.is_bound_var and (isinstance(toks, int) or toks.is_bound_var)
  sp, tk = kernel_var(start_pos.src[0]), (kernel_var(toks.src[0]) if isinstance(toks, UOp) else toks)
  bound = (Tensor(start_pos),) + ((Tensor(toks),) if isinstance(toks, UOp) else ())
  tile = PREP_TILE if tokens % PREP_TILE == 0 and PREP_TILE >= history else tokens
  f32 = dtypes.float32
  outs = [Tensor.empty(batch, heads, tokens, d, dtype=f32, device=device) for d in (key_dim, key_dim, value_dim)] + \
         [Tensor.empty(batch, heads, tokens, dtype=f32, device=device) for _ in range(3)]
  prep = functools.partial(_gdn_prep_kernel, start_pos=sp, toks=tk, k_heads=k_heads, qk_eps=qk_eps, lanes=lanes, tile=tile)
  res = Tensor.custom_kernel(*outs, conv_state, qkv, beta_alpha.reshape(batch, tokens, 2*heads), conv_w, dt_bias, ssm_a, *bound, fxn=prep)
  conv_call = _call(functools.partial(_gdn_conv_state_kernel, start_pos=sp, toks=tk, lanes=lanes), res[6].uop, qkv.uop, *(x.uop for x in bound))
  return tuple(res[:6]), conv_call

def gdn_prefill(qkv:Tensor, beta_alpha:Tensor, conv_state:Tensor, state:Tensor, conv_w:Tensor, dt_bias:Tensor, ssm_a:Tensor, start_pos:UOp,
                toks:int|UOp, k_heads:int, qk_eps:float) -> Tensor:
  """A GatedDeltaNet prefill chunk from its projections to the scan output: qkv = attn_qkv(x) (B,T,C) and beta_alpha = ssm_beta(x) |
  ssm_alpha(x) (B,T,2H) fp16 at the PADDED width T (rows >= toks are ignored), toks = the real token count (model.py's bound "toks"
  Variable, or an int). Updates conv_state and state in place (zeroed first at start_pos 0); returns core (B,H,T,V) fp32 -- rows >= toks
  are the padding's (finite, sliced off by the caller). conv_w/dt_bias/ssm_a: GatedDeltaNetBlock.prepare_nv_decode's fp32 copies."""
  batch, heads, value_dim, key_dim = cast(tuple[int, int, int, int], state.shape)
  (q, k, v, kq, beta, alpha), conv_call = gdn_prep(qkv, beta_alpha, conv_state, conv_w, dt_bias, ssm_a, start_pos, toks, heads, key_dim,
                                                   value_dim, k_heads, qk_eps)
  core = Tensor.empty(batch, heads, cast(int, qkv.shape[1]), value_dim, dtype=dtypes.float32, device=conv_state.device)
  scan = functools.partial(_gated_delta_prefill_kernel, start_pos=kernel_var(start_pos.src[0]), lanes=lanes_for(conv_state.device))
  # the scan's state input carries the conv_state store into the graph (model.py's generic path hangs it there too)
  return Tensor(core.uop.after(_call(scan, core.uop, q.uop, k.uop, v.uop, beta.uop, alpha.uop, state.uop.after(conv_call), kq.uop)))

# ******** step 2: the post-scan glue as one kernel ********
# gdn_post_nv -- one warp per (batch, token, v-head) at V = 128 (128/V heads per warp below it); each lane owns one q8 word of the head's
#   row, 4 consecutive values. ssm_norm's rms over the head row (a warp reduction), * norm weight * SiLU(attn_gate) -> fp16 z (zero past
#   toks: the rows nv_wmma_linear's pad zeroed), then ssm_out's q8 activations of that fp16 z -- the max over each 32-group's 8 lanes,
#   the packed word, the scale and the group sum, q8_quantize's exact formulas, so bit-identical to quantizing z. register_fused_q8 hands
#   them to ssm_out's IMMA gemm (sums for the card's Q5_K). Replaces ssm_norm's reduce, the gated elementwise, z's pad copy and ssm_out's
#   q8_quantize_nv: ~43 us per block at 128 tokens on the card for ~7 MB of traffic (~8 us).

def _offsets(n:int) -> tuple[int, ...]: return tuple(1 << i for i in reversed(range(n.bit_length() - 1)))  # xor butterfly over n lanes

@functools.cache
def _gdn_post_kernel(z:UOp, xq:UOp, xd:UOp, xs:UOp, core:UOp, gate:UOp, norm_w:UOp, *bound:UOp, toks:UOp|int, norm_eps:float,
                     lanes:int=32) -> UOp:
  """z (B,T,H*V) fp16, xq (B*T,H*V/32,8) uint32, xd/xs (B*T,H*V/32) fp32 out (q8_quantize(z, sums=True)'s layout); core (B,H,T,V) fp32 =
  the scan's output, gate (B,T,H*V) = attn_gate(x), norm_w (V,) fp32 = ssm_norm's weight."""
  batch, heads, tokens, value_dim = cast(tuple[int, int, int, int], core.shape)
  per = 4 if lanes > 1 else value_dim  # values per thread: one q8 word, or the CPU form's whole head row
  hl = value_dim // per  # threads per head row
  hpw = lanes // hl if lanes > 1 else 1  # heads per warp
  assert value_dim % 32 == 0 and hl & (hl - 1) == 0 and (lanes == 1 or (lanes % hl == 0 and heads % hpw == 0))
  w, lane = UOp.range(batch*tokens*heads // hpw, 0), _warp_lane(lanes)
  bt, hw = w // (heads // hpw), w % (heads // hpw)
  b, t = bt // tokens, bt % tokens
  h, li = (hw*hpw + lane // hl, lane % hl) if hpw > 1 else (hw, lane)
  base, row = h*value_dim + li*per, b*tokens + t  # this thread's first feature of z's row, and that row
  vals = [core[b, h, t, li*per + j].load() for j in range(per)]
  sq = _added(x*x for x in vals)
  if hl > 1: sq = xor_reduce(sq, _offsets(hl))
  inv, ok = (sq * (1/value_dim) + norm_eps).rsqrt(), t < toks  # nn.RMSNorm: x * (mean(x^2) + eps).rsqrt(), then * weight
  zh = [ok.where(x*inv*norm_w[li*per + j].load() * gate[b, t, base + j].load().float().silu(), 0.0).cast(dtypes.float16) for j, x in enumerate(vals)]
  stores = [z[b, t, base + j].store(x) for j, x in enumerate(zh)]
  # quantize the STORED fp16 z (q8_quantize's input): the codegen folds a cast(half).float() pair back to the fp32 value, so read the
  # thread's own stores back (an L1 hit) -- this keeps the q8 words bit-identical to the old path's q8_quantize_nv of z
  zf = [z.after(*stores)[b, t, base + j].load().float() for j in range(per)]
  def amax(xs) -> UOp: return functools.reduce(lambda a, x: a.maximum(x), (x.abs() for x in xs))
  def qsum(xs, scale) -> UOp: return _added((x/scale).round().clip(-127, 127) for x in xs)  # integers <= 32*127: exact in any order
  if per == 4:  # one word per lane, a 32-group = 8 adjacent lanes (the WARP lane is the hardware lane: xor 4/2/1 stays in the group)
    scale = (xor_reduce(amax(zf), (4, 2, 1), maximum=True) / 127).maximum(1e-8)
    g = base // 32
    stores += [xq[row, g, li % 8].store(_q8_pack(tuple(zf), scale)), xd[row, g].store(scale),
               xs[row, g].store(scale * xor_reduce(qsum(zf, scale), (4, 2, 1)))]  # the same bits in the group's 8 lanes: no gate
  else:
    for gi in range(per // 32):
      vs = zf[32*gi:32*gi + 32]
      scale, g = (amax(vs) / 127).maximum(1e-8), base // 32 + gi
      stores += [xq[row, g, wi].store(_q8_pack(tuple(vs[4*wi:4*wi + 4]), scale)) for wi in range(8)]
      stores += [xd[row, g].store(scale), xs[row, g].store(scale * qsum(vs, scale))]
  return UOp.group(*stores).end(*(r for r in (lane, w) if r.op is Ops.RANGE)).sink(arg=KernelInfo(name="gdn_post_nv", opts_to_apply=()))

def gdn_post(core:Tensor, gate:Tensor, norm_w:Tensor, toks:int|UOp, norm_eps:float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """(z (B,T,H*V) fp16 = (ssm_norm(core) * SiLU(gate)) with rows >= toks zeroed, and q8_quantize(z, sums=True)'s (q, scale, sums)):
  core = gdn_prefill's output, gate = attn_gate(x) at the padded width, norm_w = prepare_nv_decode's fp32 ssm_norm weight."""
  batch, heads, tokens, value_dim = cast(tuple[int, int, int, int], core.shape)
  features, device = heads*value_dim, core.device
  tk, bound = (kernel_var(toks.src[0]), (Tensor(toks),)) if isinstance(toks, UOp) else (toks, ())
  outs = [Tensor.empty(batch, tokens, features, dtype=dtypes.float16, device=device),
          Tensor.empty(batch*tokens, features // 32, 8, dtype=dtypes.uint32, device=device)] + \
         [Tensor.empty(batch*tokens, features // 32, dtype=dtypes.float32, device=device) for _ in range(2)]
  fxn = functools.partial(_gdn_post_kernel, toks=tk, norm_eps=norm_eps, lanes=lanes_for(device))
  z, xq, xd, xs = Tensor.custom_kernel(*outs, core, gate.reshape(batch, tokens, features), norm_w, *bound, fxn=fxn)[:4]
  return z, xq, xd, xs
