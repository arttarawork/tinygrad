"""T1.8b: tuned Metal decode-attention kernel, opt-in via FAST_ATTN=1 (see tinygrad/llm/cli.py).

Builds on T1.8's attention_impl hook and naive decode-only custom_kernel proof (tinygrad/llm/model.py,
test/unit/test_attention.py). Three changes over the naive kernel, in order of measured impact:

  1. online softmax (one pass): m/l/acc are updated together per Tk step instead of three separate
     max/sum/weighted-output passes, so K is read once instead of three times.
  2. `dout` (the output head-dim index) is declared AxisType.LOCAL instead of the default WEAK/GLOBAL.
     This is the single biggest lever: without it, Metal dispatches one thread per threadgroup (verified
     by inspecting the rendered source -- no `lid` use at all), wasting 31/32 SIMD lanes. Making dout LOCAL
     turns each (b, h) into one threadgroup of Hd cooperating threads.
  3. cooperative QK contraction: each of the Hd threads computes ONE multiply-add (its own
     Q[d]*K[j,d]) into a threadgroup-shared (LOCAL addrspace) buffer; all Hd threads then redundantly sum
     that Hd-element buffer (cheap threadgroup-SRAM reads) to get the shared score, instead of every
     thread independently re-reading all of K[j,:] from global memory (an Hd-fold redundant global-memory
     read in the naive kernel). V is already read exactly once per thread by construction (each thread
     owns one output column) -- untouched.
  4. chunking: threadgroup_barrier is the dominant remaining cost (~2 barriers/Tk-step without it).
     CHUNK Tk positions are staged into a (CHUNK, Hd) shared buffer per barrier pair instead of 1,
     amortizing barrier overhead ~CHUNK-fold. Measured optimum around CHUNK=16-32 on M3 Pro; larger
     chunks cost more LOCAL memory and start hurting occupancy.

ponytail: the CHUNK-sized threadgroup buffer needs Tk to be known at kernel-*build* time (Python-level
`Tk // CHUNK` / remainder unrolling) -- that's fine because of the FAST_ATTN gate below, not despite it.

HARD LIMITER (found while wiring this up, applies equally to T1.8's original naive kernel): real decode
runs through `Transformer.generate` -> `TinyJit`, which promotes the growing KV-cache slice length (Tk)
to a symbolic UOp Variable after the *first* decode step (verified live: `model.generate(...)` on a
1-token prompt -- decode step 0 is concrete, step 1 is already symbolic and crashes without the gate
below). `Tensor.custom_kernel` hard-asserts `all_int(self.shape)` (tinygrad/uop/ops.py,
`placeholder_like`) and refuses any symbolic shape outright. So *no* custom_kernel-based attention_impl
-- tuned or naive -- can run on the real multi-token decode hot path without this gate; it would crash by
the second generated token. The `isinstance(k.shape[2], int)` check below is therefore not a defensive
nicety, it's the difference between "opt-in speedup" and "opt-in crash". See NV_LLM_DESIGN.md / TASKS.md
T1.8b entry for the full writeup: this kernel is measurably faster than the naive one at fixed shapes,
but with the JIT symbolic promotion in the way, it only actually engages for the first decode step (and
for any concrete-shape, non-JIT call, e.g. direct prefill-style invocations) -- it's shipped anyway,
flag-gated, because it's correct and a real improvement over the naive kernel at every shape that CAN
reach it; unblocking the JIT ceiling itself is future work (T2.4-adjacent), out of scope here.
"""
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.llm import model

CHUNK = 16  # threadgroup-shared-buffer rounds; near-optimal at Hd in {64,128} on M3 Pro (measured T1.8b)


def _make_tuned_attn_kernel(chunk:int):
  def _tuned_attn_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
    B, H, _, Hd = Q.shape
    KvH, Tk = K.shape[1], K.shape[2]
    Hd = int(Hd)  # head_dim is always concrete (unlike Tk, which the FAST_ATTN gate guarantees is concrete too)
    R, scale = H // KvH, Hd ** -0.5

    b, h, dout = UOp.range(B, 0), UOp.range(H, 1), UOp.range(Hd, 2, axis_type=AxisType.LOCAL)
    kvh = h // R

    m = UOp.placeholder((1,), dtypes.float32, 10, addrspace=AddrSpace.REG)
    l = UOp.placeholder((1,), dtypes.float32, 11, addrspace=AddrSpace.REG)
    acc = UOp.placeholder((1,), dtypes.float32, 12, addrspace=AddrSpace.REG)
    m = m.after(b, h, dout)[0].set(float("-inf"))
    l = l.after(m)[0].set(0.0)
    acc = acc.after(l)[0].set(0.0)

    q_val = Q[b, h, 0, dout].cast(dtypes.float32)
    n_full = Tk // chunk

    if n_full > 0:
      local_buf = UOp.placeholder((chunk, Hd), dtypes.float32, 50, addrspace=AddrSpace.LOCAL)
      r = UOp.range(n_full, 3, axis_type=AxisType.REDUCE)

      # produce: stage `chunk` positions' Q*K partials (one multiply-add per thread per position)
      cw = UOp.range(chunk, 101, axis_type=AxisType.LOOP)
      j_w = r * chunk + cw
      store_produce = local_buf[cw, dout].after(m, l, acc, r).store(q_val * K[b, kvh, j_w, dout].cast(dtypes.float32))
      produce_done = store_produce.end(cw)

      # consume: `chunk` scores, carrying the online-softmax recurrence forward across rounds
      cr = UOp.range(chunk, 102, axis_type=AxisType.REDUCE)
      j_r = r * chunk + cr
      d2 = UOp.range(Hd, 100, axis_type=AxisType.REDUCE)
      qk = UOp.placeholder((1,), dtypes.float32, 13, addrspace=AddrSpace.REG)
      qk = qk.after(produce_done, cr)[0].set(0.0)
      qk = qk[0].set(qk.after(d2)[0] + local_buf.after(produce_done)[cr, d2], end=d2)
      score = qk[0] * scale

      m_old, l_old, acc_old = m.after(r, cr)[0], l.after(r, cr)[0], acc.after(r, cr)[0]
      m_new_val = m_old.maximum(score)
      corr = (m_old - m_new_val).exp()
      l_new_val = l_old * corr + (score - m_new_val).exp()
      acc_new_val = acc_old * corr + (score - m_new_val).exp() * V[b, kvh, j_r, dout].cast(dtypes.float32)
      grouped = UOp.group(m[0].store(m_new_val), l[0].store(l_new_val), acc[0].store(acc_new_val)).end(cr, r)
      m, l, acc = m.after(grouped), l.after(grouped), acc.after(grouped)

    # tail: Tk % chunk leftover positions, unrolled at kernel-build time (Tk is a concrete int here --
    # guaranteed by the FAST_ATTN gate in tuned_decode_attention, which falls back for symbolic Tk). No
    # cooperative LOCAL-buffer staging here (each thread does its own full Hd contraction, Hd-fold
    # redundant global reads like the naive kernel) -- not worth the complexity for <= chunk-1 positions.
    for j in range(n_full * chunk, Tk):
      j_const = UOp.const(j)
      dsum = UOp.range(Hd, 200 + j, axis_type=AxisType.REDUCE)
      qk_t = UOp.placeholder((1,), dtypes.float32, 14, addrspace=AddrSpace.REG)
      qk_t = qk_t.after(m, l, acc)[0].set(0.0)
      qk_t = qk_t[0].set(qk_t.after(dsum)[0] + Q[b, h, 0, dsum].cast(dtypes.float32) * K[b, kvh, j_const, dsum].cast(dtypes.float32), end=dsum)
      score = qk_t[0] * scale

      m_old, l_old, acc_old = m[0], l[0], acc[0]
      m_new_val = m_old.maximum(score)
      corr = (m_old - m_new_val).exp()
      l_new_val = l_old * corr + (score - m_new_val).exp()
      acc_new_val = acc_old * corr + (score - m_new_val).exp() * V[b, kvh, j_const, dout].cast(dtypes.float32)
      grouped = UOp.group(m[0].store(m_new_val), l[0].store(l_new_val), acc[0].store(acc_new_val)).end()
      m, l, acc = m.after(grouped), l.after(grouped), acc.after(grouped)

    out = acc[0] / l[0]
    store = O[b, h, 0, dout].store(out.cast(O.dtype))
    return store.end(dout, h, b).sink(arg=KernelInfo(name=f"tuned_attn_{B}_{H}_{Tk}_{Hd}", opts_to_apply=()))
  return _tuned_attn_kernel

_tuned_kernel_fxn = _make_tuned_attn_kernel(CHUNK)


def tuned_decode_attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None) -> Tensor:
  """attention_impl-compatible tuned decode kernel. Falls back to the default SDPA path for anything it
  doesn't handle: prefill/rollout (T>1), masked decode (e.g. sliding-window), and -- critically -- a
  symbolic (JIT-promoted) Tk, which Tensor.custom_kernel cannot express at all. See module docstring."""
  if mask is not None or q.shape[2] != 1 or not isinstance(k.shape[2], int):
    return model._sdpa_default(q, k, v, mask)
  O = Tensor.empty(q.shape, dtype=q.dtype, device=q.device)
  return Tensor.custom_kernel(O, q, k, v, fxn=_tuned_kernel_fxn)[0]
