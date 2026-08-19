"""T1.8b/T1.8c: tuned Metal decode-attention kernel, opt-in via FAST_ATTN=1 (see tinygrad/llm/cli.py).

Builds on T1.8's attention_impl hook and naive decode-only custom_kernel proof (tinygrad/llm/model.py,
test/unit/test_attention.py). Four changes over the naive kernel, in order of measured impact (T1.8b):

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

T1.8c: chunking is now entirely kernel-side and symbolic-Tk-safe. The round count is `ceildiv(Tk, CHUNK)`
(tinygrad.helpers.ceildiv on a UOp: `(Tk + CHUNK - 1) // CHUNK` when Tk isn't a known-positive Python int --
see its source) fed straight into a REDUCE `UOp.range`, so it's just as valid when Tk is a bound Variable as
when it's a concrete int -- no Python-level `Tk // CHUNK`/remainder branching at all, unlike T1.8b's
`n_full = Tk // chunk; if n_full > 0: ...; for j in range(n_full*chunk, Tk): ...` (which needed a concrete
int and raised "eval failed to be a single number" the moment Tk was a bound Variable -- T4.7's JIT-promoted
KV-cache length after the first decode step). The last chunking round can run past the true Tk when Tk isn't
a multiple of CHUNK (always possible now that Tk is dynamic): those positions clamp their K/V read index to
`Tk - 1` (always a physically valid index into the real, live cache -- see HARD LIMITER below for why an
*unclamped* out-of-range read would be unsafe) via `(pos < Tk).where(pos, Tk - 1)`, and get masked out of the
online-softmax update with `valid.where(score, -inf)` instead -- `exp(-inf - m) == 0` drops their
contribution to both the running sum and the weighted output exactly like a real tail loop would, with no
separate code path. This is the in-kernel tail guard the old Python tail loop used to provide; there is now
only one code path, chunked, for every Tk.

HARD LIMITER, now RESOLVED for this kernel (previously the blocker T1.8b hit and T1.8c was scoped to fix):
real decode runs through `Transformer.generate` -> `TinyJit`, which promotes the growing KV-cache slice
length (Tk) to a symbolic UOp Variable after the *first* decode step (verified live: `model.generate(...)`
on a 1-token prompt -- decode step 0 is concrete, step 1 is already symbolic). `Tensor.custom_kernel` used to
hard-assert `all_int(self.shape)` outright; T4.7 (tinygrad/uop/ops.py `placeholder_like`) taught it to accept
a bound Variable on a REDUCE-only dim instead, by allocating the underlying buffer at its static max extent
and SHRINKing to the live (symbolic) extent, converting the bound value to its kernel-side PARAM form. T4.7's
own tests covered that machinery only via a *directly*-bound-Variable shape (`k_full[:, :, :tk_var]`); real
decode's actual Tk is a *compound* expression, `start_pos + T` (an ADD of a bound Variable and a constant),
which `placeholder_like`'s original `to_kernel_param` didn't recurse into -- it left the raw bound-var BUFFER
node embedded inside the ADD, which downstream codegen (`codegen/late/coalesce.py` memory coalescing) can't
render ("memory coalescing should be on INDEX, not Ops.BUFFER"), reproduced with a minimal probe mirroring
model.py's real Tk construction. Found and fixed alongside this task (T4.7-adjacent, not the chunking
question T1.8c was scoped around): `to_kernel_param` now recurses into any UOp's `.src` when the node itself
isn't a bound var/Variable, so a bound var nested anywhere inside shape arithmetic gets PARAM-ified, not just
a bare top-level one. Small (4 lines), root-cause (fixes every compound symbolic-shape expression, not just
Tk), and necessary: without it, no custom-kernel attention_impl -- tuned or naive -- can reach a *second*
real decode step at all, chunked or not. See NV_LLM_DESIGN.md / TASKS.md T1.8c entry for the full writeup.
"""
from tinygrad import Tensor, dtypes, Device
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import ceildiv
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.llm import model

CHUNK = 16  # threadgroup-shared-buffer rounds; near-optimal at Hd in {64,128} on M3 Pro (measured T1.8b)


def _make_tuned_attn_kernel(chunk:int):
  def _tuned_attn_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
    B, H, _, Hd = Q.shape
    KvH, Tk = K.shape[1], K.shape[2]
    Hd = int(Hd)  # head_dim is always concrete; only Tk (the KV length) is ever symbolic
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

    # T1.8c: round count is ceildiv(Tk, chunk) -- symbolic-safe (a UOp expression when Tk is a bound
    # Variable, a plain int when Tk is concrete) fed directly into a REDUCE range. No Python branch on Tk.
    n_chunks = ceildiv(Tk, chunk)
    local_buf = UOp.placeholder((chunk, Hd), dtypes.float32, 50, addrspace=AddrSpace.LOCAL)
    r = UOp.range(n_chunks, 3, axis_type=AxisType.REDUCE)

    # produce: stage `chunk` positions' Q*K partials (one multiply-add per thread per position). the last
    # round may run past Tk (whenever Tk isn't a multiple of chunk) -- those lanes clamp their read to
    # Tk - 1 (always physically valid: the real, live Tk, never an assumed static bound) and get masked out
    # below instead of skipped, replacing the old Python-unrolled tail loop with one code path.
    cw = UOp.range(chunk, 101, axis_type=AxisType.LOOP)
    j_w = r * chunk + cw
    j_w_safe = (j_w < Tk).where(j_w, Tk - 1)
    store_produce = local_buf[cw, dout].after(m, l, acc, r).store(q_val * K[b, kvh, j_w_safe, dout].cast(dtypes.float32))
    produce_done = store_produce.end(cw)

    # consume: `chunk` scores, carrying the online-softmax recurrence forward across rounds
    cr = UOp.range(chunk, 102, axis_type=AxisType.REDUCE)
    j_r = r * chunk + cr
    valid_r = j_r < Tk
    j_r_safe = valid_r.where(j_r, Tk - 1)
    d2 = UOp.range(Hd, 100, axis_type=AxisType.REDUCE)
    qk = UOp.placeholder((1,), dtypes.float32, 13, addrspace=AddrSpace.REG)
    qk = qk.after(produce_done, cr)[0].set(0.0)
    qk = qk[0].set(qk.after(d2)[0] + local_buf.after(produce_done)[cr, d2], end=d2)
    # invalid (tail-padding) lanes score -inf: exp(-inf - m_new) == 0, so they can't move m/l/acc at all.
    score = valid_r.where(qk[0] * scale, float("-inf"))

    m_old, l_old, acc_old = m.after(r, cr)[0], l.after(r, cr)[0], acc.after(r, cr)[0]
    m_new_val = m_old.maximum(score)
    corr = (m_old - m_new_val).exp()
    l_new_val = l_old * corr + (score - m_new_val).exp()
    acc_new_val = acc_old * corr + (score - m_new_val).exp() * V[b, kvh, j_r_safe, dout].cast(dtypes.float32)
    grouped = UOp.group(m[0].store(m_new_val), l[0].store(l_new_val), acc[0].store(acc_new_val)).end(cr, r)
    m, l, acc = m.after(grouped), l.after(grouped), acc.after(grouped)

    out = acc[0] / l[0]
    store = O[b, h, 0, dout].store(out.cast(O.dtype))
    return store.end(dout, h, b).sink(arg=KernelInfo(name=f"tuned_attn_{B}_{H}_{Tk}_{Hd}", opts_to_apply=()))
  return _tuned_attn_kernel

_tuned_kernel_fxn = _make_tuned_attn_kernel(CHUNK)


def tuned_decode_attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None) -> Tensor:
  """attention_impl-compatible tuned decode kernel. Handles decode (T==1) at any Tk, concrete or symbolic
  (T1.8c). Falls back to the default SDPA path for anything it doesn't handle: prefill/rollout (T>1) and
  masked decode (e.g. sliding-window) -- those need a real per-query mask, which this decode-only kernel
  (one query row per (b,h) threadgroup, unconditional over Tk modulo the Tk-tail guard above) doesn't carry --
  and any device whose renderer can't do LOCAL (dout is AxisType.LOCAL, local_buf is AddrSpace.LOCAL; a
  renderer with has_local=False, e.g. CPU's Clang/LLVM/X86, has no threadgroup/shared-memory model for this
  kernel to target at all -- without this gate, codegen's add_gpudims hits the has_threads branch with no
  GLOBAL/THREAD range to size "core_id" from and crashes with an IndexError on an empty global_shape)."""
  if mask is not None or q.shape[2] != 1:
    return model._sdpa_default(q, k, v, mask)
  dev = q.device if isinstance(q.device, str) else q.device[0]
  if not Device[dev].renderer.has_local:
    return model._sdpa_default(q, k, v, mask)
  O = Tensor.empty(q.shape, dtype=q.dtype, device=q.device)
  return Tensor.custom_kernel(O, q, k, v, fxn=_tuned_kernel_fxn)[0]
