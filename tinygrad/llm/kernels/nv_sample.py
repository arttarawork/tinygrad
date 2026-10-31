"""T4.146: the head's token pick as NV kernels -- forward()'s greedy argmax and sample_logits' Gumbel-max draw (with the min_p mask) over
the vocabulary in 3 launches (2 greedy-only) instead of the generic graph's ~11: on the card (10-31 profile, decode step) the head
gemv's split-K sum (r_1940_32_4_5), five vocab-wide reduces (r_2_32_4_970, 57 us each: 256 threads walking 970 logits apiece) and their
small tails cost ~0.33 ms of a 25 ms step, and the speculative chain pays a draw per draft step.

  head_max_nv   blocks of THREADS threads x PER_THREAD logits: the head gemv's partials summed (left to right from 0, the generic reduce's
                order) and stored as the logits the draw reads; each block's (max, first index of it) -> (rows, nb)
  head_draw_nv  the same blocks: the max of the scaled logits from head_max_nv's block maxima (x -> x * (1/t) is monotone for t > 0,
                so the block maxima scaled are the scaled blocks' maxima, and a max is exact in any order), sample_logits' ops on the
                same values -- s = logits / t, s >= max(s) + log(min_p) or -inf, s - log(-log(max(u, 1e-12))) with u =
                Tensor.rand_like(logits) -- and each block's (max, first index)
  head_pick_nv  one block per row: the block candidates -> the greedy id and the drawn id; temperature > 0 picks, as forward() does
A candidate beats another with a larger value, or the same value and a smaller index, so ties go to the first index as tinygrad's argmax
((x == max) * reversed arange) does; a NaN never wins and a row without a non-NaN value gives the row width, as there. The uniforms are
Tensor.rand_like(logits)'s, so the RNG counter advances exactly as sample_logits' (T4.139). Layout: warp = LOCAL range, lane = the WARP
range (threadIdx.x, never split: the xor shuffles pair hardware lanes). Gate: model.py (NV_SAMPLE, default 1; 0 = the generic graph)."""
from __future__ import annotations
import functools, math
from typing import cast
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.nv_quant import nv_quant_supported

NV_SAMPLE = ContextVar("NV_SAMPLE", 1)  # 0 = the generic argmax / sample_logits graph (A/B); read in model.py at trace time
THREADS, PER_THREAD = 256, 8  # 2048 logits per block: 122 blocks at the 27B's 248320
WARPS = THREADS // 32

def nv_sample_ok(logits:Tensor) -> bool:
  vocab = logits.shape[-1]
  return bool(NV_SAMPLE.value and len(logits.shape) == 2 and isinstance(vocab, int) and isinstance(logits.shape[0], int) and vocab < 2**24
              and -(-vocab // (THREADS*PER_THREAD)) <= THREADS and nv_quant_supported(logits.device))

def _xor(val:UOp, off:int) -> UOp: return UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {off})", dtypes.float))

def _pick(a:tuple[UOp, UOp], b:tuple[UOp, UOp]) -> tuple[UOp, UOp]:
  take = (a[0] < b[0]) | (a[0].eq(b[0]) & (b[1] < a[1]))  # b beats a: a larger value, or the same value at a smaller index
  return take.where(b[0], a[0]), take.where(b[1], a[1])

def _cand(v:UOp, idx:UOp, vocab:int, ok:UOp|None=None) -> tuple[UOp, UOp]:
  bad = v.ne(v) if ok is None else v.ne(v) | ok.logical_not()  # NaN (or past the vocabulary): the (-inf, vocab) sentinel
  return bad.where(-math.inf, v), bad.where(float(vocab), idx.cast(dtypes.float32))

def _block_best(cands:list[tuple[UOp, UOp]], warp:UOp, lane:UOp, slot:int) -> tuple[UOp, UOp]:
  """the block's best (value, index): each thread's candidates, the warp's xor butterfly (every lane ends with the same pair: the order
  is total), one shared slot per warp, a barrier, every thread the same over the WARPS slots"""
  v, i = functools.reduce(_pick, cands)
  for off in (16, 8, 4, 2, 1): v, i = _pick((v, i), (_xor(v, off), _xor(i, off)))
  sv, si = (UOp.placeholder((WARPS,), dtypes.float32, slot=slot + k, addrspace=AddrSpace.LOCAL) for k in range(2))
  done = UOp.group(sv[warp.valid(lane.eq(0))].store(v), si[warp.valid(lane.eq(0))].store(i)).barrier()
  sv, si = sv.after(done), si.after(done)
  return functools.reduce(_pick, [(sv[w].load(), si[w].load()) for w in range(WARPS)])

def _grid(rows:int, nb:int):
  blk, warp, lane = UOp.range(rows*nb, 0), UOp.range(WARPS, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  return blk // nb, blk % nb, warp, lane, warp*32 + lane, (blk, warp, lane)

def _elements(b:UOp, tid:UOp, vocab:int):
  """(index, in-vocabulary gate or None) of a thread's PER_THREAD logits: strided by THREADS, so a warp's loads are contiguous"""
  for e in range(PER_THREAD):
    idx = b*(THREADS*PER_THREAD) + e*THREADS + tid
    yield idx, (None if vocab % (THREADS*PER_THREAD) == 0 else idx < vocab)

@functools.cache
def _head_max_kernel(bmax:UOp, bidx:UOp, *rest:UOp, rows:int, vocab:int, nb:int, C:int) -> UOp:
  """bmax/bidx (rows, nb) fp32 out; rest = (logits out (rows, vocab), the head gemv's partials (rows, vocab, C)) or (logits (rows, vocab),)"""
  row, b, warp, lane, tid, ends = _grid(rows, nb)
  cands, stores = [], []
  for idx, ok in _elements(b, tid, vocab):
    g = idx if ok is None else idx.valid(ok)
    if C:
      out, parts = rest
      v = sum((parts[row, g, c].load() for c in range(C)), UOp.const(0, dtypes.float32))
      stores.append(out[row, g].store(v))
    else: v = rest[0][row, g].load().float()
    cands.append(_cand(v, idx, vocab, ok))
  v, i = _block_best(cands, warp, lane, slot=0)
  return UOp.group(*stores, bmax[row, b].store(v), bidx[row, b].store(i)).end(*ends).sink(arg=KernelInfo(name="head_max_nv", opts_to_apply=()))

@functools.cache
def _head_draw_kernel(bv:UOp, bi:UOp, logits:UOp, u:UOp, temp:UOp, bmax:UOp, *, rows:int, vocab:int, nb:int, min_p:float) -> UOp:
  """bv/bi (rows, nb) fp32 out: each block's best of sample_logits' Gumbel-perturbed (masked) scaled logits; logits/u (rows, vocab), temp (1,),
  bmax (rows, nb) = head_max_nv's block maxima of the logits"""
  row, b, warp, lane, tid, ends = _grid(rows, nb)
  tp = temp[0].load().maximum(1e-12)
  thr = None
  if min_p > 0:  # sample_logits: scaled.max(-1) + log(min_p), scaled = logits / tp
    # each block maximum scaled first, then the max of those: the threshold's add then reads a shuffled max, never the product itself, so
    # nvcc cannot contract it into an fma (one rounding) -- the generic graph adds log(min_p) to a reduced max too
    m = bmax[row, tid.minimum(nb - 1)].load() / tp
    if nb < THREADS: m = (tid < nb).where(m, -math.inf)
    thr = _block_best([(m, UOp.const(0, dtypes.float32))], warp, lane, slot=0)[0] + math.log(min_p)
  cands = []
  for idx, ok in _elements(b, tid, vocab):
    g = idx if ok is None else idx.valid(ok)
    s = logits[row, g].load() / tp
    if thr is not None: s = (s >= thr).where(s, -math.inf)
    cands.append(_cand(s - u[row, g].load().maximum(1e-12).log().neg().log(), idx, vocab, ok))
  v, i = _block_best(cands, warp, lane, slot=2)
  return UOp.group(bv[row, b].store(v), bi[row, b].store(i)).end(*ends).sink(arg=KernelInfo(name="head_draw_nv", opts_to_apply=()))

@functools.cache
def _head_pick_kernel(tok:UOp, bmax:UOp, bidx:UOp, *rest:UOp, rows:int, nb:int, vocab:int) -> UOp:
  """tok (rows, 1) int32 out: the greedy id from head_max_nv's candidates; rest = (bv, bi, temp): the drawn id, picked when temp > 0"""
  row, _, warp, lane, tid, ends = _grid(rows, 1)
  def best(vs:UOp, ix:UOp, slot:int) -> UOp:
    j = tid.minimum(nb - 1)
    cand = (vs[row, j].load(), ix[row, j].load())
    if nb < THREADS: cand = ((tid < nb).where(cand[0], -math.inf), (tid < nb).where(cand[1], float(vocab)))
    return _block_best([cand], warp, lane, slot)[1]
  ids = best(bmax, bidx, 0)
  if rest:
    bv, bi, temp = rest
    ids = (temp[0].load() > 0).where(best(bv, bi, 2), ids)
  return tok[row, 0].store(ids.cast(dtypes.int32)).end(*ends).sink(arg=KernelInfo(name="head_pick_nv", opts_to_apply=()))

def nv_sample(logits:Tensor, temperature:Tensor|None, min_p:float, partials:Tensor|None=None) -> Tensor:
  """(rows, 1) int32 token ids for logits (rows, vocab): logits.argmax(-1) when temperature is None, else forward()'s
  (temperature > 0).where(sample_logits(logits, temperature, min_p), argmax) -- the same ids. partials: the head gemv's split-K partials
  (rows, vocab, C) of these logits (split_k_partials), summed in head_max_nv instead of by their own reduce."""
  rows, vocab = cast(int, logits.shape[0]), cast(int, logits.shape[1])
  nb, dev = -(-vocab // (THREADS*PER_THREAD)), logits.device
  def empty(*shape, dtype=dtypes.float32) -> Tensor: return Tensor.empty(*shape, dtype=dtype, device=dev)
  C = 0 if partials is None else cast(int, partials.shape[-1])
  assert partials is None or partials.shape == (rows, vocab, C), f"{partials.shape=} {logits.shape=}"
  srcs = (empty(rows, vocab), partials) if partials is not None else (logits,)
  outs = Tensor.custom_kernel(empty(rows, nb), empty(rows, nb), *srcs, fxn=functools.partial(_head_max_kernel, rows=rows, vocab=vocab, nb=nb, C=C))
  bmax, bidx, lg = outs[0], outs[1], (outs[2] if partials is not None else logits)
  pick = functools.partial(_head_pick_kernel, rows=rows, nb=nb, vocab=vocab)
  if temperature is None: return Tensor.custom_kernel(empty(rows, 1, dtype=dtypes.int32), bmax, bidx, fxn=pick)[0]
  temp = temperature.reshape(1)
  u = Tensor.rand_like(logits)  # sample_logits' own draw: the same uniforms, the same RNG counter advance
  bv, bi = Tensor.custom_kernel(empty(rows, nb), empty(rows, nb), lg, u, temp, bmax,
                                fxn=functools.partial(_head_draw_kernel, rows=rows, vocab=vocab, nb=nb, min_p=min_p))[:2]
  return Tensor.custom_kernel(empty(rows, 1, dtype=dtypes.int32), bmax, bidx, bv, bi, temp, fxn=pick)[0]
