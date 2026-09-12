# T5.6: the prefill attention's score pipeline (qk, exp(qk-max), exp/sum -- each (B,H,T,Tk_max) fp32 at the SYMBOLIC max length) is
# the whole planned scratch of a prefill jit family; the memory planner used to give the three pairwise-overlapping buffers three
# slots because its TLSF rounds a request up to the next bucket before searching and so never reuses a freed block of exactly the
# requested size. With bucket-aligned sizing they take two. Numerics are untouched: the planner only places buffers.
import unittest
from dataclasses import replace
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops
import tinygrad.engine.jit as jitmod
from tinygrad.llm.model import Transformer, _sdpa_default
from test.unit.test_llm_server import TEST_CONFIG

def capture_arenas(fn):
  """Run fn(); return the planner's arena sizes (bytes) of every jit capture it triggered (the int8 arena buffers it substitutes in)."""
  arenas, orig = [], jitmod.memory_plan_rewrite
  def spy(linear, held_bufs=None):
    ret = orig(linear, held_bufs)
    # the planner hands kernels SHRINK/BITCAST views into int8 arena buffers -- collect the arenas underneath
    bufs = [u for si in ret.src for src in si.src[1:] for u in src.toposort() if u.op is Ops.BUFFER and u.dtype == dtypes.int8]
    arenas.append(sorted({u.max_numel() for u in bufs}, reverse=True))
    return ret
  jitmod.memory_plan_rewrite = spy
  try: fn()
  finally: jitmod.memory_plan_rewrite = orig
  return arenas

class TestPrefillArena(unittest.TestCase):
  def test_score_pipeline_plans_two_slots(self):
    N, KvH, Hd, H, T = 8192, 2, 32, 2, 32
    cache = Tensor.zeros(2, 1, KvH, N, Hd, dtype=dtypes.float16).contiguous().realize()
    v_sp, v_t = UOp.variable("start_pos", 0, N-1), UOp.variable("toks", 1, T)
    @TinyJit
    def attn(q, kv, start_pos, Tv):
      kvb = Tensor(cache.uop.after(cache[:, :, :, start_pos:start_pos+Tv, :].uop.store(kv.cast(cache.dtype).uop)))
      k, v = kvb[0, :, :, 0:start_pos+Tv, :].cast(dtypes.float32), kvb[1, :, :, 0:start_pos+Tv, :].cast(dtypes.float32)
      return q.scaled_dot_product_attention(k, v, enable_gqa=True).realize()
    def run():
      for i in range(2):
        Tb = v_t.bind(T)
        attn(Tensor.rand(1, H, T, Hd)[:, :, :Tb], Tensor.rand(2, 1, KvH, T, Hd)[:, :, :, :Tb], v_sp.bind(i*T), Tb)
    arenas = capture_arenas(run)
    self.assertEqual(len(arenas), 1)
    slot = H * T * (N + T) * 4  # one (B,H,T,Tk_max) fp32 score buffer
    self.assertLess(max(arenas[0]), 2.2 * slot, f"arena {max(arenas[0])/1e6:.2f} MB: the three score buffers should share two slots")
    self.assertGreaterEqual(max(arenas[0]), 2 * slot)

  def test_group_count_scales_arena_down(self):
    # T4.103a: SDPA_HEAD_GROUPS splits the score pipeline into G groups of query heads (default: one KV head
    # per group), each group calling the same SDPA math on its own q/k/v slice with no data dependency on any
    # other group. A group's OWN score buffer is H/G heads wide instead of H, so if the planner fully drained
    # one group (freed its buffers) before starting the next, the whole arena would be ~1/G of the G=1 arena.
    # It does NOT: memory_plan_rewrite (schedule/memory.py) assigns buffer lifetimes from POSITION IN THE
    # LINEARIZED KERNEL LIST, and the linearizer is free to interleave independent groups' kernels (e.g. every
    # group's QK^T before any group's softmax) since nothing but the final `cat` forces an order -- so several
    # groups' score buffers end up concurrently live. Measured law (H=8, KvH=4, Hd=T=32, N=8192, this test's
    # shape): G=1 -> 2.03 slots (the T5.6 pairwise-overlap packing), G=2 -> 3.05 slots, G=4 -> 5.11 slots (a
    # "slot" = one group's own (B,H/G,T,Tk_max) fp32 buffer) -- i.e. slot-count grows ~G+1, not staying at 2,
    # which caps the REAL reduction at ~2x/(1+1/G) regardless of G: ratio(G=2)=1.33x, ratio(G=4)=1.59x here.
    # The only way found to force true per-group draining is a `.realize()` at the end of each group (verified
    # separately to bring G=4 to ~3.98x, i.e. genuinely ~1/4) -- but `_attention` always runs inside
    # TransformerBlock.__call__'s `@function(precompile=True)` trace (model.py's `_run`), which sets
    # ALLOW_DEVICE_USAGE=0 (tinygrad/function.py) specifically to forbid mid-trace device compilation; a
    # `.realize()` there raises `AssertionError: usage of device CPU disallowed` (reproduced directly against
    # Transformer.generate()), i.e. it would break real generation, not just this test. So the shipped grouping
    # keeps the plain loop + cat (no realize): still real, byte-identical-at-G=1, and here ~1.59x smaller at
    # the default G=4 -- just not the ~1/4 a naive per-group-slot count would suggest.
    N, KvH, Hd, H, T = 8192, 4, 32, 8, 32
    def run(groups):
      cache = Tensor.zeros(2, 1, KvH, N, Hd, dtype=dtypes.float16).contiguous().realize()
      v_sp, v_t = UOp.variable("start_pos", 0, N-1), UOp.variable("toks", 1, T)
      @TinyJit
      def attn(q, kv, start_pos, Tv):
        kvb = Tensor(cache.uop.after(cache[:, :, :, start_pos:start_pos+Tv, :].uop.store(kv.cast(cache.dtype).uop)))
        k, v = kvb[0, :, :, 0:start_pos+Tv, :].cast(dtypes.float32), kvb[1, :, :, 0:start_pos+Tv, :].cast(dtypes.float32)
        with Context(SDPA_HEAD_GROUPS=groups): return _sdpa_default(q, k, v, None).realize()
      for i in range(2):
        Tb = v_t.bind(T)
        attn(Tensor.rand(1, H, T, Hd)[:, :, :Tb], Tensor.rand(2, 1, KvH, T, Hd)[:, :, :, :Tb], v_sp.bind(i*T), Tb)
    arenas1, arenas4 = capture_arenas(lambda: run(1)), capture_arenas(lambda: run(4))
    self.assertEqual((len(arenas1), len(arenas4)), (1, 1))
    slot1, slot4 = max(arenas1[0]), max(arenas4[0])
    self.assertLess(slot4, slot1, f"1-group {slot1/1e6:.2f} MB vs 4-group {slot4/1e6:.2f} MB: grouping must shrink the arena")
    ratio = slot1 / slot4
    self.assertGreater(ratio, 1.4, f"1-group {slot1/1e6:.2f} MB vs 4-group {slot4/1e6:.2f} MB, ratio {ratio:.2f}")
    self.assertLess(ratio, 1.8, f"1-group {slot1/1e6:.2f} MB vs 4-group {slot4/1e6:.2f} MB, ratio {ratio:.2f}")

  def test_model_prefill_arena_is_two_score_slots(self):
    # the tiny attention model's prefill family: (B,H,T,Tk_max) fp32 slots with H=2, T=32 -> ~2 slots + small stuff, not 3
    cfg = replace(TEST_CONFIG, max_context=8192)
    def run():
      g = Transformer(cfg).generate(list(range(1, 41)), chunk_size=32)
      for _ in range(3): next(g)
    arenas = capture_arenas(run)
    slot = cfg.n_heads * 32 * (cfg.max_context + 32) * 4
    self.assertLess(max(arenas[0]), 2.3 * slot, f"prefill arena {max(arenas[0])/1e6:.2f} MB vs slot {slot/1e6:.2f} MB")

  def test_planned_size_is_block_aligned_and_reusable(self):
    # the sizing must (a) keep every offset block_size-aligned (LLVM kernel args are declared `align 32`; x86 aligned vector
    # loads fault otherwise) and (b) let a freed block satisfy an equal request across the whole size range
    from tinygrad.schedule.memory import planned_size
    from tinygrad.runtime.support.memory import TLSFAllocator
    for n in list(range(0, 70000, 7)) + [2105344, 402718720, 1 << 30]:
      self.assertEqual(planned_size(n) % 256, 0, n)
      self.assertGreaterEqual(planned_size(n), n)
    for n in (300, 4096, 8200, 2105344, 402718720):
      sz, t = planned_size(n), TLSFAllocator(planned_size(n) * 6, block_size=256, lv2_cnt=32)
      a, b = t.alloc(sz), t.alloc(sz)
      t.free(a)
      c = t.alloc(sz)
      self.assertEqual(c, a, f"n={n}: a freed equal-sized block must be reused")
      self.assertEqual((a | b | c) % 256, 0)

  def test_planner_does_not_change_numerics(self):
    # the planner only decides WHERE intermediates live: outputs must equal the unplanned run bit for bit
    def ids(no_planner:int) -> list[int]:
      with Context(NO_MEMORY_PLANNER=no_planner):
        Tensor.manual_seed(7)
        m = Transformer(replace(TEST_CONFIG, max_context=256))
        for p in m.blk[0].__dict__.values():
          if isinstance(p, Tensor) and p.requires_grad is None and p.ndim == 2: p.assign(Tensor.randn(*p.shape) * 0.05).realize()
        g = m.generate(list(range(1, 45)), chunk_size=32)
        return [next(g) for _ in range(6)]
    self.assertEqual(ids(1), ids(0))

if __name__ == "__main__": unittest.main()
