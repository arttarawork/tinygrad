"""T4.98c: NV port of amd.py's quantized-linear decode kernel (Q4_K/Q5_K/Q6_K/IQ4_XS) plus formats amd.py doesn't
have (Q8_0, Q4_0; T4.98i adds Q4_1, IQ4_NL) -- CUDA-C (sm_70+) intrinsics mirroring amd.py's RDNA3 ones (__dp4a for sudot4,
__byte_perm for v_perm_b32, __shfl_xor_sync for ds_swizzle via kernels/nv.py's warp_reduce). Same q8-activation
design as amd.py: q8_quantize turns the activation into int8 words + a per-32-group float scale, group_dot does
one 32-lane warp per (token, output, chunk-of-32-groups) with a dp4a dot per group, warp_reduce sums the lanes.
Decode-only (tokens<=4): the WMMA gemm path (amd.py's _q5_linear_f16_wmma_kernel/_iq4_linear_f16_wmma_kernel
equivalent) is T4.98f. Dispatch: amd.py's Linear.__call__ calls nv_q8_linear when NV_CUSTOM_QUANT=1 (default 0)
on an NV device with a supported ggml_type; nv_q8_linear returns None for anything it doesn't cover (big batches,
symbolic token counts, unsupported formats) and the caller falls back to the generic dequant+matmul path."""
from __future__ import annotations
import functools, math
from typing import Callable, cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.amd import (
  Linear, Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, Q4_0, Q4_1, IQ4_NL, Q3_K, IQ3_XXS, IQ3_S, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q4_WORDS, Q5_WORDS,
  Q6_BYTES, IQ4_WORDS, _half, _q5_scales, _iq4_scales, IQ_GRID_SIZES)
from tinygrad.llm.kernels.nv import warp_reduce, _nv_device_ok, iq_grid

NV_CUSTOM_QUANT = ContextVar("NV_CUSTOM_QUANT", 0)  # 0 = never take this path, even on NV (default: byte-identical everywhere)
NV_QUANT_TYPES = (Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, Q4_0, Q4_1, IQ4_NL, Q3_K, IQ3_XXS, IQ3_S)  # T4.98k adds the three 3-bit formats
# T4.125d: output rows per warp in the decode gemv (1, 2, or default 4) -- _decode_linear falls back to 1 when it
# doesn't divide out_features. Read only in nv_q8_linear (uncached) and passed into _quant_decode_kernel's cache
# key as rows_per_warp -- never read inside a @functools.cache'd kernel builder (nv_gemm.py's NV_WMMA_TOKEN_TILE
# comment explains why: a cache hit would then silently serve a stale R for a later call with a different knob).
NV_GEMV_ROWS = ContextVar("NV_GEMV_ROWS", 4)  # 10-28 card sweep: R=1 28.6, R=2 28.0, R=4 27.9 ms/step (Q6_K head 1.62 -> 1.20 ms)

def nv_quant_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV" or not NV_CUSTOM_QUANT.value: return False
  return _nv_device_ok(device)  # CUDA C renderer + sm_70+: dp4a needs sm_61+, full-mask __shfl_*_sync needs Volta+

# ******** CUDA intrinsics mirroring amd.py's AMDGCN ones ********

def _nv_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  # __dp4a(int,int,int) is signed x signed x accumulate -- matches amd's sudot4(true,a,true,b,c) exactly (T4.98c)
  return UOp(Ops.CUSTOMI, src=(a.int(), b.int(), c), arg=("__dp4a({}, {}, {})", dtypes.int32))

def _nv_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  # AMD v_perm_b32(S0,S1,sel): sel is 4 BYTE-lanes (0-15 each), idx 0-3 <- S1 (2nd arg), 4-7 <- S0 (1st arg) --
  # hardware-verified against test/amd/hw/test_vop3.py::TestPermMore.test_v_perm_b32_select_high_bytes (S0=
  # 0x03020100, S1=0x07060504, sel picks indices 4,5,6,7 -> result reconstructs S0 byte-for-byte). CUDA's
  # __byte_perm(x,y,s) (prmt.b32) is the opposite: idx 0-3 <- x (1st arg), 4-7 <- y (2nd arg) -- so a/b swap
  # below. It also only reads 4 NIBBLES packed into sel's low 16 bits, not 4 bytes spread across all 32 like
  # AMD's -- repack the byte-lanes amd.py's callers build (_iq4_bytes: each lane already a 0-15 value, high
  # nibble zero) into that layout. T4.98c
  a32, b32, sel = a.cast(dtypes.uint32), b.cast(dtypes.uint32), selectors.cast(dtypes.uint32)
  nv_sel = sum(((sel >> (4*i)) & (0xf << (4*i)) for i in range(4)), UOp.const(0, dtypes.uint32))
  return UOp(Ops.CUSTOMI, src=(b32, a32, nv_sel), arg=("__byte_perm({}, {}, {})", dtypes.uint32))

def _nv_load(ptr:UOp, lanes:int|None=None) -> UOp:
  assert ptr.op is Ops.INDEX
  if lanes is None:
    # amd's __builtin_nontemporal_load is a cache-bypass hint with no direct CUDA equivalent wired here; __ldg is
    # a possible perf lever to try once this runs on real hardware (T4.98c) -- a plain load is correct either way
    return ptr.load()
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(dtype=ptr.dtype)

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  # verbatim amd.py's _iq4_bytes with _nv_byte_perm in place of _amd_byte_perm (T4.98c)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _nv_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _nv_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _nv_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

# ******** q8-activation quant linear (verbatim amd.py structure; warp_reduce is the only device-specific piece) ********

def _decode_linear(out:UOp, xq:UOp, xd:UOp, out_features:int, group_count:int, group_dot, name:str, rows_per_warp:int=1) -> UOp:
  # T4.125d: R output rows per warp amortise the per-group activation load (xwords/xd) over R weight dots instead of
  # one -- a lane's 18-byte Q4_0 weight block is smaller than the 32 B of activation words + scale it loads to dot
  # it, so at R=1 the gemv moves more activation bytes than weight bytes per dot (llama.cpp's mmvq does the same
  # amortisation over several rows). R rows are a plain Python loop producing R values/warp_reduces/stores, grouped
  # before .end() -- NOT a new LOCAL range: the lane range stays exactly the 32-wide one this function always built,
  # because warp_reduce's butterfly shuffle combines whichever 32 threads the hardware warp maps these lanes to
  # regardless of R (nv-custom-kernel-lane-split memory note: shuffle reductions here are lane-identity-agnostic).
  R = rows_per_warp if out_features % rows_per_warp == 0 else 1  # fall back when R doesn't divide out_features
  row_groups = out_features // R
  chunks = (group_count+31)//32
  token_output_chunk, lane = UOp.range(out.shape[0]*row_groups*chunks, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, output, chunk = token_output_chunk // (row_groups*chunks), (token_output_chunk//chunks) % row_groups, token_output_chunk % chunks
  group = lane+chunk*32
  safe_group = group if group_count % 32 == 0 else group.minimum(group_count-1)
  # the activation (xwords/xs) depends only on (token, group), never on the output row -- load it once per lane and
  # reuse it for all R rows below, instead of reloading it inside group_dot once per row (R=1: identical load, same
  # clamped safe_group the original inline ternary passed into group_dot, just hoisted one level up).
  xwords, xs = _nv_load(xq[token, safe_group, 0], 8), xd[token, safe_group]
  def value(row:UOp) -> UOp:
    dot = group_dot(token, row, safe_group, xwords, xs)
    return dot if group_count % 32 == 0 else (group < group_count).where(dot, UOp.const(0, dtypes.float32))
  rows = tuple(output if R == 1 else output*R + r for r in range(R))  # R==1: no extra output*1+0 node -- bit-for-bit the old row index
  totals = tuple(warp_reduce(value(row), full_wave=True) for row in rows)
  stores = tuple(out[token, row, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)) for row, total in zip(rows, totals))
  return UOp.group(*stores).end(token_output_chunk, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, x:UOp, sums:UOp|None=None, *, tokens:int, in_features:int) -> UOp:
  # T4.128: `sums` (optional 4th param, an output after the input so the 3-param kernel and its render stay as they were) gets
  # xd * sum(q8) per (token, group) -- llama.cpp q8_1's `s`, the activation side of Q4_1's +m and Q5_K's -dmin*m terms
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group//groups, token_group%groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane*4+i].float() for i in range(4))
  word = sum(((v/group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8)
              for i,v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  stores = (q[token, group, lane.valid(lane < 8)].store(word), scale[token, group.valid(lane.eq(0))].store(group_scale))
  if sums is not None:
    # T4.128: each lane's own element quantized exactly as the packing does, summed by one more warp reduce (integers <= 32*127:
    # exact in fp32 in any order), times the scale once
    qsum = warp_reduce((x[token, group, lane].float()/group_scale).round().clip(-127, 127), full_wave=True)
    stores += (sums[token, group.valid(lane.eq(0))].store(group_scale * qsum),)
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize_nv", opts_to_apply=()))

# T4.123a: several Linears in one forward step quantize the SAME activation (FFN gate+up, GDN attn_gate/attn_qkv,
# attention q/k/v all read one norm output) -- q8_quantize used to run the quantize kernel once per Linear even
# though the result is identical. Share it, keyed on x.uop (the live graph node for this call's activation).
# A plain `functools.cache` (unbounded) was rejected: a cache VALUE here is a (q, scale) pair whose CALL node has
# x.uop as a literal src (the kernel genuinely reads x's buffer), so the cache would hold a live strong path back
# to its own key (cache -> value -> kernel -> x.uop) -- a cycle rooted in a module-global, which neither simple
# refcounting nor a weakref (e.g. WeakKeyDictionary) can break, since the dict is itself always reachable. The
# fix is a HARD BOUND: `lru_cache(maxsize=...)` evicts the oldest entries once the cache overflows. The Linears that share
# an input are called back to back (q/k/v, gate/up, attn_gate..attn_qkv), so a handful of entries gets every share; a
# big bound would pin up to that many old activation graphs (and the buffers an eager/capture run realized for them) on a
# card that runs within ~1 GB of full at 262k context.
@functools.lru_cache(maxsize=16)
def _q8_quantize_cached(x_uop:UOp, tokens:int, in_features:int, sums:bool) -> tuple[UOp, ...]:
  x = Tensor(x_uop)
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  extra = [Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)] if sums else []
  outs = Tensor.custom_kernel(q, scale, x, *extra, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))
  return tuple(t.uop for t in outs[:2] + outs[3:])

def q8_quantize(x:Tensor, tokens:int, in_features:int, sums:bool=False) -> tuple[Tensor, ...]:
  """(q, scale), or with sums=True (q, scale, sums) -- T4.128: the IMMA Q4_1/Q5_K prefill kernels need the per-group sums. The
  variant is its own cache entry rather than always computed: the plain kernel (every decode quantize, the Q4_0 prefill one) keeps
  its render, and no activation of the 27B feeds both kinds of consumer (Q4_1 ffn_down and Q5_K ssm_out read their own inputs), so
  the split never costs a second launch. The fused (decode-only) entries carry no sums and are skipped for a sums request."""
  if not sums and (hit := _fused_q8.get((x.uop, tokens, in_features))) is not None: return Tensor(hit[0]), Tensor(hit[1])
  return tuple(Tensor(u) for u in _q8_quantize_cached(x.uop, tokens, in_features, sums))

# T4.123e: model.py's _ffn_output/GatedDeltaNetBlock's own norm-site helper call self.ffn_norm(h)/self.attn_norm(x)
# completely normally (building nn.RMSNorm's ordinary lazy graph -- the exact Tensor object every consumer Linear
# keeps using unchanged) but the fused add_rmsnorm_q8 kernel already computed that norm's (q, scale) itself, in
# the SAME kernel as the residual-add. register_fused_q8 lets the q8_quantize call every consumer's nv_q8_linear
# makes on that exact Tensor (shared across consumers since T4.123a -- FFN gate+up, GDN attn_gate/attn_qkv,
# attention q/k/v all read one norm output) return those already-computed tensors instead of re-deriving them
# with a second q8_quantize_nv kernel -- so the RMSNorm graph built above is dead code (nothing ever reads its
# VALUE) and is never scheduled. Same cycle-safety rationale as _q8_quantize_cached above (a cache value holds a
# strong path back to its own key): bounded by simple FIFO eviction, not unbounded -- entries are read back
# within the same block's _run body, microseconds after being registered, so eviction order barely matters here
# (unlike _q8_quantize_cached's true LRU, tuned for sharing across back-to-back calls).
_FUSED_Q8_MAXSIZE = 16
_fused_q8: dict[tuple[UOp, int, int], tuple[UOp, UOp]] = {}

def register_fused_q8(x:Tensor, tokens:int, in_features:int, q:Tensor, scale:Tensor) -> None:
  key = (x.uop, tokens, in_features)
  if key not in _fused_q8 and len(_fused_q8) >= _FUSED_Q8_MAXSIZE: _fused_q8.pop(next(iter(_fused_q8)))
  _fused_q8[key] = (q.uop, scale.uop)

# T4.125b: nv_q8_linear's result is its kernel's split-K partials (tokens, out_features, C) summed by a generic reduce -- one
# r_40_32_4_6 per ssm_out/attn_output per step. add_rmsnorm_q8 sums partials in-kernel, so _ffn_output (model.py) fetches the
# producing gemv's un-summed partials here and the summed graph is never read (dead code, as with register_fused_q8 above). Same
# bounded FIFO: the lookup comes right after the producing call.
_split_k: dict[UOp, UOp] = {}

def _register_split_k(result:Tensor, partials:Tensor) -> None:
  if result.uop not in _split_k and len(_split_k) >= _FUSED_Q8_MAXSIZE: _split_k.pop(next(iter(_split_k)))
  _split_k[result.uop] = partials.uop

def split_k_partials(result:Tensor) -> Tensor|None: return None if (p := _split_k.get(result.uop)) is None else Tensor(p)

def clear_step_caches() -> None:
  """T4.126: drop the per-step sharing caches above (Transformer._forward_step calls this after every forward) -- their entries
  hold a step's activation buffers alive past the step, which a 128-token prefill capture at 262k cannot afford."""
  _q8_quantize_cached.cache_clear()
  _fused_q8.clear()
  _split_k.clear()

# ******** T4.123d: fused residual-add + RMSNorm + q8-quantize decode kernel (design doc section 3, P4 kernel
# half) -- wired into model.py's FFNBlock._ffn_output / GatedDeltaNetBlock._normed_for_attention by T4.123e
# (register_fused_q8 above is that wiring's other half). At each of the 129 norm sites per decode token the
# residual stream used to go through 3-4 separate ~6us kernels (a generic residual-sum, r_16_320's one-warp
# RMSNorm sum-of-squares, E_40_32_4's apply, q8_quantize_nv): this does all of it in one kernel, one block per
# token -- at the sites T4.123e's two gates cover (dense ffn_norm both block types, GDN's attn_norm). ********

def _q8_pack(values:tuple[UOp, ...], group_scale:UOp) -> UOp:
  # verbatim _q8_quantize_kernel's word-packing formula (the exact rounding/layout this task must match): 4
  # signed-int8 lanes packed into one uint32 word, each lane independently round-half-to-even + clip(-127,127).
  return sum(((v/group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8)
              for i,v in enumerate(values)), UOp.const(0, dtypes.uint32))

@functools.cache
def _add_rmsnorm_q8_kernel(*params:UOp, tokens:int, dim:int, eps:float, C:int, want_fp16:bool, threads:int) -> UOp:
  """params, in order: new_residual, q, scale, residual, weight, [partials if C>0], [fp16_out if want_fp16] --
  the variadic *params (keyed by position, not by a keyword-with-default that would shift when an optional src
  is omitted) is what lets Tensor.custom_kernel's one-placeholder-per-src convention carry two independent
  optional tensors (nv_quant.py's other kernels only ever have one optional src, e.g. _quant_decode_kernel's
  IQ3 grid, which a trailing `=None` kwarg handles fine -- two would collide positionally).

  One block (AxisType.GLOBAL) per token, `threads` (a multiple of 32 dividing dim, <= 1024) threads per block.
  Phase 1: every thread computes `per_thread` (=dim/threads) strided elements of residual+sum(partials),
  writes them to the new_residual output AND stages them in a block-shared buffer (AddrSpace.LOCAL) -- the
  LOCAL store is what makes the codegen pipeline's add_raw_barrier pass (tinygrad/codegen/__init__.py) insert
  a __syncthreads() wherever this buffer is read back below, the same mechanism r_16_320's auto-scheduled
  buf2/__syncthreads() uses and attn_kernel.py's local_buf/`.after(produce_done)` idiom hand-builds explicitly.
  Each thread also reduces its own elements' sum-of-squares via warp_reduce (one shuffle-tree per warp) and
  stashes its warp's total in a second LOCAL buffer, one slot per warp.
  Phase 2: every thread re-reads all `threads/32` per-warp sums (now safely visible post-barrier) and computes
  rms = sqrt(mean+eps) then 1/rms -- that exact two-op order, not a fused rsqrt, to match nn.RMSNorm._norm's
  `.rsqrt()` decomposition (tensor.py: `self.sqrt().reciprocal()`) bit-for-bit. Each of the `threads/32` warps
  then owns `per_thread` (== dim/32/warps exactly, given the dim%threads==0 assert -- see below) contiguous
  32-wide q8 groups and quantizes its normed slice with _q8_pack -- the identical warp-per-group / lane<8-packs-4 layout
  _q8_quantize_kernel uses, just reading the LOCAL-staged (not global) activation and multiplying in
  inv_rms*weight first. An optional fp16 copy of the normed value (the GDN block's `x.half()` f16-gemv input,
  design table row for E_40_32_4) is written from the same per-lane value, no recompute."""
  assert threads % 32 == 0 and dim % threads == 0, f"threads={threads} must be a multiple of 32 dividing dim={dim}"
  warps = threads // 32
  # groups_per_warp (= groups/warps, groups=dim/32) is ALWAYS == per_thread (= dim/threads) given the assert
  # above: dim=m*threads=m*32*warps => dim/32=m*warps => (dim/32)/warps=m=dim/threads. One variable, two uses.
  per_thread = dim // threads
  new_residual, q, scale, residual, weight = params[:5]
  i = 5
  partials = params[i] if C > 0 else None
  if C > 0: i += 1
  fp16_out = params[i] if want_fp16 else None

  # warp (0..warps-1) is AxisType.LOCAL, lane (the 32 hardware lanes) is the dedicated AxisType.WARP (axis_id
  # -1, same reserved id amd.py's _amd_flash_attention/codegen/opt/postrange.py's tensor-core path use) --
  # NOT one size-`threads` AxisType.LOCAL range split via //32/%32: add_gpudims is free to decompose a single
  # wide LOCAL range into (threadIdx.x, threadIdx.y) in whatever shape it likes (e.g. 256 -> (8,32)), which
  # does NOT line up with CUDA's actual hardware warp boundary (threadIdx.x fastest, warps = contiguous
  # linearized-thread-id chunks of 32) unless the 32-wide axis is specifically pinned to the warp slot via
  # AxisType.WARP -- confirmed by rendering this kernel both ways (the first draft's split render showed
  # local_size=(8,32) with our "lane" value NOT equal to the hardware lane, which would make warp_reduce's
  # __shfl_xor_sync combine the wrong 32 threads).
  tok = UOp.range(tokens, 0, AxisType.GLOBAL)
  warp, lane = UOp.range(warps, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid = warp * 32 + lane
  val_buf = UOp.placeholder((dim,), dtypes.float32, slot=50, addrspace=AddrSpace.LOCAL)
  wsum_buf = UOp.placeholder((warps,), dtypes.float32, slot=51, addrspace=AddrSpace.LOCAL)

  # phase 1: residual add (+ optional partial sums), staged into val_buf for phase 2's re-read. Unrolled over
  # per_thread (a compile-time int, 5 at the standing dim=5120/threads=1024) the same way _q8_quantize_kernel
  # unrolls its 4-wide word pack -- no RANGE/REG machinery needed since each thread's per_thread elements are
  # independent (no loop-carried state).
  vals, writes = [], []
  for r in range(per_thread):
    d = tid + r*threads
    val = residual[tok, d].load().float()
    # T4.125b: partials first, residual last -- renders as p0+...+p(C-1)+residual, exactly the generic reduce's sum and then the
    # residual add it replaces (the C renderer strips a same-op child's parens, so residual+(p0+...) compiled as (residual+p0)+...)
    if C: val = sum((cast(UOp, partials)[tok, d, c].load().float() for c in range(C)), UOp.const(0, dtypes.float32)) + val
    vals.append(val)
    writes += [new_residual[tok, d].store(val), val_buf[d].store(val)]
  warp_sq = warp_reduce(sum((v*v for v in vals), UOp.const(0, dtypes.float32)), full_wave=True)
  writes.append(wsum_buf[warp.valid(lane.eq(0))].store(warp_sq))
  phase1 = UOp.group(*writes)

  # phase 1b: cross-warp combine -- every thread re-reads all `warps` shared partials post-barrier (same
  # redundant-per-thread combine r_16_320's own 16-wide loop does for its one warp)
  vb, wsum = val_buf.after(phase1), wsum_buf.after(phase1)
  total = sum((wsum[w].load() for w in range(warps)), UOp.const(0, dtypes.float32))
  inv_rms = ((total * (1.0 / dim)) + eps).sqrt().reciprocal()

  # phase 2: normed = val_buf * inv_rms * weight, then _q8_quantize_kernel's exact per-32-group pack
  stores = []
  for g in range(per_thread):  # groups_per_warp == per_thread, see above
    group = warp * per_thread + g
    base = group * Q8_GROUP_SIZE
    own = vb[base + lane].load() * inv_rms * weight[base + lane].load().float()
    group_scale = (warp_reduce(own.abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
    word_lane = lane.minimum(7)
    xs = tuple(vb[base + word_lane*4 + j].load() * inv_rms * weight[base + word_lane*4 + j].load().float() for j in range(4))
    stores += [q[tok, group, lane.valid(lane < 8)].store(_q8_pack(xs, group_scale)), scale[tok, group.valid(lane.eq(0))].store(group_scale)]
    if want_fp16: stores.append(cast(UOp, fp16_out)[tok, base + lane].store(own.cast(dtypes.float16)))
  return UOp.group(*stores).end(lane, warp, tok).sink(arg=KernelInfo(name="add_rmsnorm_q8_nv", opts_to_apply=()))

def add_rmsnorm_q8(residual:Tensor, weight:Tensor, eps:float, partials:Tensor|None=None, want_fp16:bool=False,
                    threads:int|None=None) -> tuple[Tensor, Tensor, Tensor, Tensor|None]:
  """Thin wrapper (T4.123d) around _add_rmsnorm_q8_kernel: residual (tokens,dim) fp32, optional partials
  (tokens,dim,C) fp32 (the previous projection's un-summed split-K partials -- summed in-kernel), weight (dim,)
  the RMSNorm weight. Returns (new_residual, q, scale, fp16_copy_or_None); q/scale are nv_q8_linear-ready
  (same (tokens,groups,8) uint32 / (tokens,groups) float32 shape _q8_quantize_cached produces). Wired into
  FFNBlock._ffn_output / GatedDeltaNetBlock._normed_for_attention's decode path by T4.123e."""
  tokens, dim = cast(tuple[int, int], residual.shape)
  # T4.125b: default = the widest block (<= 1024 threads): 1024 at dim=5120, i.e. 5 elements per thread and 5 q8 groups per warp
  # instead of 256 threads' 20 each -- the 256-thread kernel was latency-bound (12.7 us per launch on the card, 10-28 profile)
  if threads is None: threads = next((t for t in range(1024, 31, -32) if dim % t == 0), 32)
  assert threads % 32 == 0 and dim % threads == 0, f"threads={threads} must be a multiple of 32 dividing dim={dim}"
  groups = dim // Q8_GROUP_SIZE
  C = 0 if partials is None else cast(int, partials.shape[-1])
  dev = residual.device
  new_residual = Tensor.empty(tokens, dim, dtype=dtypes.float32, device=dev)
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=dev)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=dev)
  fp16_out = Tensor.empty(tokens, dim, dtype=dtypes.float16, device=dev) if want_fp16 else None
  # order must match _add_rmsnorm_q8_kernel's *params unpacking exactly: fixed outputs+inputs, then partials
  # (if any), then fp16_out (if any) LAST -- fp16_out is conceptually an output but must sit positionally
  # after the optional partials input, since the kernel keys optionals by position, not name (see its docstring)
  srcs = [new_residual, q, scale, residual, weight]
  if partials is not None: srcs.append(partials)
  if fp16_out is not None: srcs.append(fp16_out)
  fxn = functools.partial(_add_rmsnorm_q8_kernel, tokens=tokens, dim=dim, eps=eps, C=C, want_fp16=want_fp16, threads=threads)
  results = srcs[0].custom_kernel(*srcs[1:], fxn=fxn)
  return results[0], results[1], results[2], (results[-1] if want_fp16 else None)

# ******** T4.127a: FFN activation (SiLU(gate)*up) + q8-quantize decode kernel -- fuses FFNBlock._feed_forward's
# T4.123c combine step with the q8_quantize_nv that follows it (ffn_down's input) into one kernel, reading the
# gate/up gemvs' own un-summed split-K partials (split_k_partials below) instead of a materialized activation
# buffer. Same warp-per-32-group layout/packing as _q8_quantize_kernel (via _q8_pack), wired into model.py's
# _feed_forward by register_fused_q8 -- the exact pattern nv_fused_norm/_ffn_output already use. ********

@functools.cache
def _ffn_act_q8_kernel(q:UOp, scale:UOp, gate:UOp, up:UOp, *, tokens:int, hidden_dim:int, C:int) -> UOp:
  """act = silu(sum_c gate[c]) * sum_c up[c], per element, then _q8_quantize_kernel's exact pack -- gate/up are
  nv_q8_linear's un-summed split-K partials (tokens, hidden_dim, C), summed in index order (matching the generic
  reduce this replaces); SiLU as Tensor.silu lowers it (x * sigmoid(x), tinygrad/mixin/elementwise.py)."""
  groups = hidden_dim // Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group // groups, token_group % groups
  gate, up = gate.reshape(tokens, groups, 32, C), up.reshape(tokens, groups, 32, C)
  def act(e:UOp) -> UOp:
    gate_sum = sum((gate[token, group, e, c].float() for c in range(C)), UOp.const(0, dtypes.float32))
    up_sum = sum((up[token, group, e, c].float() for c in range(C)), UOp.const(0, dtypes.float32))
    return gate_sum.silu() * up_sum
  group_scale = (warp_reduce(act(lane).abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(act(word_lane*4+i) for i in range(4))
  stores = (q[token, group, lane.valid(lane < 8)].store(_q8_pack(xs, group_scale)),
            scale[token, group.valid(lane.eq(0))].store(group_scale))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="ffn_act_q8_nv", opts_to_apply=()))

def ffn_act_q8(gate_partials:Tensor, up_partials:Tensor, tokens:int, hidden_dim:int) -> tuple[Tensor, Tensor]:
  """Thin wrapper (T4.127a) around _ffn_act_q8_kernel -- see FFNBlock._feed_forward's NV decode branch (model.py).
  Returns (q, scale) in the same (tokens, groups, 8) uint32 / (tokens, groups) float32 shape q8_quantize/
  add_rmsnorm_q8 produce, ready for register_fused_q8."""
  C = cast(int, gate_partials.shape[-1])
  groups = hidden_dim // Q8_GROUP_SIZE
  dev = gate_partials.device
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=dev)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=dev)
  fxn = functools.partial(_ffn_act_q8_kernel, tokens=tokens, hidden_dim=hidden_dim, C=C)
  q, scale = Tensor.custom_kernel(q, scale, gate_partials, up_partials, fxn=fxn)[:2]
  return q, scale

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, grid:UOp|None=None, *, out_features:int, in_features:int, ggml_type:int,
                          rows_per_warp:int=1) -> UOp:
  # grid: the IQ3 codebook (nv.py iq_grid) as a 5th kernel input for IQ3_XXS/IQ3_S (T4.98k), None for every other format
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp, xwords:UOp, xs:UOp) -> UOp:
    # T4.125d: xwords (the activation words) and xs (its scale) are hoisted into _decode_linear and passed in --
    # loaded once per lane, shared across the R rows one warp now produces, instead of reloaded here per row.
    def _word32_u16(idx16:UOp) -> UOp:
      # T4.125a: raw is set_quantized's uint16 view here (Q4_0/IQ4_NL/Q6_K only) -- two u16 loads reconstruct the
      # same little-endian 32-bit word the byte view's four single-byte loads built (GPU loads are little-endian
      # on both AMD and NV, verified T4.98c), halving the load-instruction count. Every caller's idx16 is derived
      # from a byte offset proven even (block widths 18/210 are both even), so the division is always exact.
      return raw[idx16].cast(dtypes.uint32) | (raw[idx16+1].cast(dtypes.uint32) << 16)
    if ggml_type in (Q4_0, IQ4_NL):
      # 32-weight blocks, 18 bytes, 2-byte aligned -- u16 view (T4.125a)
      base16 = (output * group_count + group) * 9  # 18 bytes / 2
      d = _half(raw[base16])
      qs_base16 = base16 + 1
      words = [_word32_u16(qs_base16 + g*2) for g in range(4)]  # the same raw 4-byte qs words the byte view built
      if ggml_type == IQ4_NL:
        # gguf.py ggml_type==20: value = d * kvalues_iq4nl[q] -- Q4_0's exact byte layout and nibble order
        # (elements 0-15 = low nibbles of qs[0:16], 16-31 = high nibbles of the SAME qs[0:16]) but each nibble
        # indexes the 16-entry kvalues_iq4nl LUT instead of q-8, and there's no sub-block scale to apply.
        # _iq4_bytes (IQ4_XS's LUT trick below) turns a raw packed byte-word straight into a dp4a-ready word of
        # signed LUT values, so no separate nibble-unpack step is needed here (T4.98i)
        dot = UOp.const(0, dtypes.int32)
        for word_idx in range(8):
          dot = _nv_dp4a(_iq4_bytes(words[word_idx % 4], 4*(word_idx//4)), xwords[word_idx], dot)
        return dot.float() * xs * d
      # Q4_0 (gguf.py ggml_type==2, value = d*(q-8)): q_to_uint8(.,4)'s order is elements 0-15 = low nibbles of
      # qs[0:16], elements 16-31 = high nibbles of the SAME qs[0:16] -- the -8 offset folds into a dp4a-of-ones
      # qsum trick (T4.98i). Per-byte-lane extraction from the reassembled word: (word >> (i*8+shift)) & 0x0f ==
      # the byte view's (byte_i >> shift) & 0x0f, since shift <= 4 never reaches into the next byte's bits.
      dot, qsum = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        shift = 4 if word_idx >= 4 else 0
        word = sum((((words[word_idx % 4] >> (i*8 + shift)) & 0x0f).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
        dot, qsum = _nv_dp4a(word, xwords[word_idx], dot), _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      return (dot.float() - 8*qsum.float()) * d * xs
    if ggml_type in (Q8_0, Q4_1):
      # 32-weight blocks, byte view (34/20-byte blocks; Q4_1's 20 bytes ARE 4-byte aligned but kept on the byte
      # view for kernel-structure parity with Q4_0 -- T4.98i). Only Q4_0/IQ4_NL/Q6_K moved to the u16 view (T4.125a)
      block_bytes = {Q8_0: 34, Q4_1: 20}[ggml_type]
      base = (output * group_count + group) * block_bytes
      d = _half(raw[base].cast(dtypes.uint16) | (raw[base+1].cast(dtypes.uint16) << 8))
      qs_base = base + (4 if ggml_type == Q4_1 else 2)  # Q4_1 has an extra f16 m before qs (Q4_0/Q8_0/IQ4_NL don't)
      if ggml_type == Q8_0:
        # gguf.py ggml_type==8: value = d * qs[i], qs already signed int8 in element order -- no unpack needed
        dot = UOp.const(0, dtypes.int32)
        for word_idx in range(8):
          qbytes = _nv_load(raw[qs_base + word_idx*4], 4)
          word = sum((qbytes[i].cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
          dot = _nv_dp4a(word, xwords[word_idx], dot)
        return dot.float() * d * xs
      # Q4_1 (gguf.py ggml_type==3, value = d*q+m): same nibble unpack as Q4_0, q_to_uint8(.,4)'s order is
      # elements 0-15 = low nibbles of qs[0:16], elements 16-31 = high nibbles of the SAME qs[0:16]. m is a
      # constant added to every weight in the block, so over the dot product it becomes m * sum(activation) --
      # same dp4a-of-ones qsum trick Q4_0 uses for its -8 bias, just added with coefficient m (T4.98i)
      m = _half(raw[base+2].cast(dtypes.uint16) | (raw[base+3].cast(dtypes.uint16) << 8))
      loads = [_nv_load(raw[qs_base + g*4], 4) for g in range(4)]
      dot, qsum = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        shifted = (loads[word_idx % 4] >> 4) if word_idx >= 4 else loads[word_idx % 4]
        word = sum(((shifted[i] & 0x0f).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
        dot, qsum = _nv_dp4a(word, xwords[word_idx], dot), _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      return (dot.float()*d + qsum.float()*m) * xs
    # 256-weight superblocks: verbatim amd.py's group_dot (_nv_dp4a/_nv_load/_iq4_bytes in place of the _amd_ ones)
    block, subgroup = group // 8, group % 8
    if ggml_type in (Q4_K, Q5_K):
      base = (output * in_features//GGML_BLOCK_SIZE + block) * (Q4_WORDS if ggml_type == Q4_K else Q5_WORDS)
      qs_base, dot, qsum = base + (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8, UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (raw[qs_base+word_idx] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K: word |= ((raw[base+4+word_idx] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        dot, qsum = _nv_dp4a(word, xwords[word_idx], dot), _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      return (dot.float()*d*scale - qsum.float()*dmin*minimum) * xs
    if ggml_type == IQ4_XS:
      base = (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        packed = _nv_load(raw[base + 2 + subgroup*4 + word_idx%4])
        dot = _nv_dp4a(_iq4_bytes(packed, 4*(word_idx//4)), xwords[word_idx], dot)
      d, scale = _iq4_scales(raw, base, subgroup)
      return dot.float() * xs * d * scale
    if ggml_type == Q3_K:
      # gguf.py ggml_type==11: 110-byte block (hmask[32], qs[64], scales[12], d f16). Each element e (i=e%32 in this
      # group) is q2 + 4*qh - 4 (q2 = 2-bit field p=subgroup%4 of qs byte at row=subgroup//4; qh = bit `subgroup` of
      # hmask[i]) -- packs straight into int8 like every other format. The 6-bit sub-block scale sc(k) (k=0..15,
      # int8, -32 offset) covers 16 elements, two per 32-group (k=2*subgroup, 2*subgroup+1), so this needs the same
      # two-dot/two-scale split as Q6_K below. sc's own two bit-fields (low nibble, byte 96+k or 96+(k-8); high 2
      # bits, byte 104+k%4) are read with k clamped to each branch's valid range (k.minimum(7)/.maximum(8)) so the
      # unused branch's address never walks past the 110-byte block (T4.98k)
      base = (output * in_features//GGML_BLOCK_SIZE + block) * 110
      row, p = subgroup // 4, (subgroup % 4).cast(dtypes.uint8)
      dots = [UOp.const(0, dtypes.int32)] * 2
      for word_idx in range(8):
        q2_bytes = _nv_load(raw[base + 32 + row*32 + word_idx*4], 4)
        qh_bytes = _nv_load(raw[base + word_idx*4], 4)
        q2 = (q2_bytes >> (2*p)) & 3
        qh = (qh_bytes >> subgroup.cast(dtypes.uint8)) & 1
        quant = (q2 | (qh << 2)).bitcast(dtypes.int8) - 4
        word = sum((quant[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
        dots[word_idx//4] = _nv_dp4a(word, xwords[word_idx], dots[word_idx//4])
      def _q3_scale(half:int) -> UOp:
        k = 2*subgroup + half
        k_lo, k_hi = k.minimum(7), k.maximum(8)
        low4 = (k < 8).where(raw[base + 96 + k_lo].cast(dtypes.uint8) & 0xf, (raw[base + 88 + k_hi].cast(dtypes.uint8) >> 4) & 0xf)
        high2 = (raw[base + 104 + k % 4].cast(dtypes.uint8) >> (2*(k//4)).cast(dtypes.uint8)) & 3
        return ((low4 | (high2 << 4)).bitcast(dtypes.int8) - 32).float()
      d = _half(raw[base+108].cast(dtypes.uint16) | (raw[base+109].cast(dtypes.uint16) << 8))
      return (dots[0].float()*_q3_scale(0) + dots[1].float()*_q3_scale(1)) * xs * d
    if ggml_type == IQ3_XXS:
      assert grid is not None, "IQ3 codebook input missing (nv.py iq_grid)"  # also narrows the Optional for mypy
      # gguf.py ggml_type==18: 98-byte block (d f16, qs[64] one code/4 weights, 8 uint32 scale+sign words at 66).
      # One db (no two-scale split -- unlike Q3_K/Q6_K, the whole 32-group shares one scale) and a codebook gather:
      # code c=subgroup*8+word_idx indexes qs, `grid[qs[c]]` (grid = nv.py iq_grid's 256-entry uint32 table, byte
      # `lane` of the word is the magnitude); the parity-derived sign bit per lane negates the magnitude via the
      # (mag^mask)-mask trick (mask = 0 or all-ones) before packing into the dp4a word (T4.98k)
      base = (output * in_features//GGML_BLOCK_SIZE + block) * 98
      d = _half(raw[base].cast(dtypes.uint16) | (raw[base+1].cast(dtypes.uint16) << 8))
      sbytes = _nv_load(raw[base + 66 + subgroup*4], 4)
      sword = sum((sbytes[i].cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
      db = d * ((sword >> 28).float() + 0.5) * 0.5
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        code = raw[base + 2 + subgroup*8 + word_idx].cast(dtypes.uint32)
        gword = grid[code.cast(dtypes.int32)]
        s7 = (sword >> (7*(word_idx//2))) & 0x7f
        px = s7 ^ (s7 >> 4)
        px = px ^ (px >> 2)
        px = px ^ (px >> 1)
        s8 = s7 | ((px & 1) << 7)
        v0 = (word_idx & 1) * 4
        lanes = []
        for l in range(4):
          mag = (gword >> (8*l)) & 0xff
          mask = UOp.const(0, dtypes.uint32) - ((s8 >> (v0+l)) & 1)
          lanes.append(((mag ^ mask) - mask).cast(dtypes.uint8))
        word = sum((lanes[l].cast(dtypes.uint32) << (l*8) for l in range(4)), UOp.const(0, dtypes.uint32))
        dot = _nv_dp4a(word, xwords[word_idx], dot)
      return dot.float() * db * xs
    if ggml_type == IQ3_S:
      assert grid is not None, "IQ3 codebook input missing (nv.py iq_grid)"  # also narrows the Optional for mypy
      # gguf.py ggml_type==21: 110-byte block (d f16, qs[64], qh[8] one high-bit byte per group, signs[32],
      # scales[4] nibble-packed). idx = qs[c] | (bit `word_idx` of qh[subgroup] << 8) into the 512-entry grid (same
      # magnitude/sign packing as IQ3_XXS); scale(subgroup) is 1+2*nib, nib the low/high nibble (subgroup%2) of
      # scales byte subgroup//2 -- byte-major order (q_to_uint8 on a (-1,4,1)-reshaped input keeps the byte index
      # outer and only bit-splits within each byte), verified against gguf.py's ggml_data_to_tensor directly since
      # it does NOT match a naive low-nibbles-then-high-nibbles reading (T4.98k)
      base = (output * in_features//GGML_BLOCK_SIZE + block) * 110
      d = _half(raw[base].cast(dtypes.uint16) | (raw[base+1].cast(dtypes.uint16) << 8))
      nib = (raw[base + 106 + subgroup//2].cast(dtypes.uint8) >> ((subgroup%2)*4).cast(dtypes.uint8)) & 0xf
      scale = (1 + 2*nib).float()
      qh_byte = raw[base + 66 + subgroup].cast(dtypes.uint32)
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        code = raw[base + 2 + subgroup*8 + word_idx].cast(dtypes.uint32)
        qh_bit = (qh_byte >> word_idx) & 1
        # + not | : disjoint bit ranges (code is 0..255, qh_bit<<8 only ever sets bit 8) so they're numerically
        # identical, but the z3 OOB checker (CHECK_OOB=1) has no rule for OR between two non-constant ArithRefs
        # inside an index expression -- T4.98k, caught by test_renders_and_compiles_for_sm86
        gword = grid[(code + (qh_bit << 8)).cast(dtypes.int32)]
        sign_byte = raw[base + 74 + subgroup*4 + word_idx//2].cast(dtypes.uint32)
        v0 = (word_idx & 1) * 4
        lanes = []
        for l in range(4):
          mag = (gword >> (8*l)) & 0xff
          mask = UOp.const(0, dtypes.uint32) - ((sign_byte >> (v0+l)) & 1)
          lanes.append(((mag ^ mask) - mask).cast(dtypes.uint8))
        word = sum((lanes[l].cast(dtypes.uint32) << (l*8) for l in range(4)), UOp.const(0, dtypes.uint32))
        dot = _nv_dp4a(word, xwords[word_idx], dot)
      return dot.float() * d * scale * xs
    # T4.125a: raw is the u16 view here too (210-byte blocks are even); base16 is base//2, exactly (210 is even).
    base16 = (output*in_features//GGML_BLOCK_SIZE+block)*(Q6_BYTES//2)
    dots = [UOp.const(0, dtypes.int32)] * 2
    for word_idx in range(8):
      pos, within = subgroup*32 + word_idx*4, (subgroup*32 + word_idx*4)%128
      # the two 4-byte windows below (ggml's ql/qh fields) always start at even byte offsets (within%64 and
      # within%32 are multiples of 4; every block stride added to them is even too) -- _word32_u16 reconstructs
      # the same word the byte view's four single-byte loads built, from two u16 loads. The pre-shift trick
      # below matches the byte view's per-byte-lane shift: shift<=4 (low) / <=2 (high) never crosses a byte.
      low = _word32_u16(base16 + (pos//128)*32 + (within%64)//2) >> ((within//64)*4).cast(dtypes.uint32)
      high = _word32_u16(base16 + 64 + (pos//128)*16 + (within%32)//2) >> ((within//32)*2).cast(dtypes.uint32)
      quants = [(((low >> (i*8)) & 15) | (((high >> (i*8)) & 3) << 4)).cast(dtypes.uint8).bitcast(dtypes.int8) - 32 for i in range(4)]
      word = sum((quants[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
      dots[word_idx//4] = _nv_dp4a(word, xwords[word_idx], dots[word_idx//4])
    scale_word = raw[base16 + 96 + subgroup]
    scales = [((scale_word >> (8*i)) & 0xff).cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    return (dots[0].float()*scales[0] + dots[1].float()*scales[1]) * xs * _half(raw[base16 + 104])
  names = {Q4_K: "linear_q4_k_nv", Q5_K: "linear_q5_k_nv", IQ4_XS: "linear_iq4_xs_nv", Q6_K: "linear_q6_nv",
           Q8_0: "linear_q8_0_nv", Q4_0: "linear_q4_0_nv", Q4_1: "linear_q4_1_nv", IQ4_NL: "linear_iq4_nl_nv",
           Q3_K: "linear_q3_k_nv", IQ3_XXS: "linear_iq3_xxs_nv", IQ3_S: "linear_iq3_s_nv"}
  return _decode_linear(out, xq, xd, out_features, group_count, group_dot, names[ggml_type], rows_per_warp=rows_per_warp)

def nv_q8_linear(layer:Linear, x:Tensor) -> Tensor|None:
  """amd.py's q8_linear decode branch, ported to NV, restricted to small batches (tokens<=4): the WMMA gemm path
  amd.py takes for bigger/aligned batches isn't ported here (T4.98f). Returns None when it doesn't cover this
  call (big batch, symbolic token count, unsupported format) so Linear.__call__ (amd.py) can fall back."""
  if layer.ggml_type not in NV_QUANT_TYPES: return None
  numel = x.numel()
  if not isinstance(numel, int): return None  # symbolic token count: no padded-chunk path ported yet (T4.98f)
  tokens = numel // layer.in_features
  if tokens > 4: return None  # decode-only for now
  # T4.125a: Q4_0/IQ4_NL/Q6_K read their packed weight through set_quantized's uint16 view (half the load
  # instructions _quant_decode_kernel's group_dot issued through the byte view) -- every other format unchanged
  weight = cast(Tensor, layer.packed_u16) if layer.ggml_type in (Q4_0, IQ4_NL, Q6_K) else layer.weight
  raw, out_features, in_features = weight.uop.buf_uop, layer.out_features, layer.in_features
  def run(fxn:Callable[..., UOp], out:UOp, *srcs:UOp) -> Tensor:
    all_srcs = (out,)+srcs
    params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
    kernel = fxn(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
    result = Tensor(out.after(kernel))
    partials = result  # T4.125b
    if len(result.shape) == 3: result = result.sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    if layer.bias is None: _register_split_k(result, partials)  # T4.125b: see split_k_partials
    return result if layer.bias is None else result + layer.bias
  xq, xd = q8_quantize(x, tokens, in_features)
  # T4.125d: read NV_GEMV_ROWS here (uncached) and pass it into the cache key -- never inside _quant_decode_kernel
  decode = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type, rows_per_warp=NV_GEMV_ROWS.value)
  out = Tensor.empty(tokens, out_features, (in_features+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  grids = (iq_grid(layer.ggml_type, cast(str, x.device)).uop,) if layer.ggml_type in IQ_GRID_SIZES else ()  # T4.98k: IQ3 codebook input
  return run(decode, out, raw, xq.uop, xd.uop, *grids)
