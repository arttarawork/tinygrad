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
    # per group), each group calling the same SDPA math on its own q/k/v slice. A group's OWN score buffer is
    # H/G heads wide instead of H, so IF the planner fully drains one group (frees its buffers) before starting
    # the next, the whole arena is ~1/G of the G=1 arena. It does NOT do that on its own: memory_plan_rewrite
    # (schedule/memory.py) assigns buffer lifetimes from POSITION IN THE LINEARIZED KERNEL LIST, and with no
    # data dependency between groups the linearizer is free to interleave their kernels (e.g. every group's
    # QK^T before any group's softmax), keeping several groups' score buffers concurrently live -- measured
    # only ~1.33-1.59x smaller at G=2..4 that way, not ~1/G. _sdpa_default fixes this with an explicit ordering
    # edge (UOp.after, see its docstring): group i>0's q slice is gated on a fresh, output-sized scratch buffer
    # holding group i-1's result, so group i's first kernel cannot be scheduled before group i-1's last kernel
    # has run -- collapsing the groups' liveness windows to non-overlapping, with no `.realize()` (device
    # execution) anywhere. Measured here (H=8, KvH=4, Hd=T=32, N=8192): ratio ~3.9-4.0x at the default G=4,
    # i.e. genuinely ~1/G -- assert a wide-ish band since TLSF bucket rounding moves it a few percent run to run.
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
    arenas1, arenas2, arenas4 = capture_arenas(lambda: run(1)), capture_arenas(lambda: run(2)), capture_arenas(lambda: run(4))
    self.assertEqual((len(arenas1), len(arenas2), len(arenas4)), (1, 1, 1))
    slot1, slot2, slot4 = max(arenas1[0]), max(arenas2[0]), max(arenas4[0])
    self.assertLess(slot4, slot2, f"2-group {slot2/1e6:.2f} MB vs 4-group {slot4/1e6:.2f} MB: more groups must shrink further")
    self.assertLess(slot2, slot1, f"1-group {slot1/1e6:.2f} MB vs 2-group {slot2/1e6:.2f} MB: grouping must shrink the arena")
    ratio2, ratio4 = slot1 / slot2, slot1 / slot4
    self.assertGreater(ratio2, 1.75, f"1-group {slot1/1e6:.2f} MB vs 2-group {slot2/1e6:.2f} MB, ratio {ratio2:.2f}")
    self.assertLess(ratio2, 2.25, f"1-group {slot1/1e6:.2f} MB vs 2-group {slot2/1e6:.2f} MB, ratio {ratio2:.2f}")
    self.assertGreater(ratio4, 3.5, f"1-group {slot1/1e6:.2f} MB vs 4-group {slot4/1e6:.2f} MB, ratio {ratio4:.2f}")
    self.assertLess(ratio4, 4.3, f"1-group {slot1/1e6:.2f} MB vs 4-group {slot4/1e6:.2f} MB, ratio {ratio4:.2f}")

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

class TestSharedArena(unittest.TestCase):
  """T4.105: JIT_SHARED_ARENA=1 -- every capture on a device aliases one planned arena (grown to the largest capture)."""
  def _two_jits(self):
    from tinygrad import TinyJit
    w1, w2, w3 = (Tensor.randn(256, 256).realize(), Tensor.randn(256, 64).realize(), Tensor.randn(8, 16).realize())
    @TinyJit
    def big(x:Tensor) -> Tensor: return ((x @ w1).relu() @ w2).sum(-1, keepdim=True).contiguous()   # two matmuls: a (64,256) intermediate
    @TinyJit
    def small(x:Tensor) -> Tensor: return ((x[:, :8] @ w3).relu() @ w3.T).sum(-1, keepdim=True).contiguous()
    def ref_big(x): return ((x @ w1).relu() @ w2).sum(-1, keepdim=True).numpy()
    def ref_small(x): return ((x[:, :8] @ w3).relu() @ w3.T).sum(-1, keepdim=True).numpy()
    return big, small, ref_big, ref_small

  def test_off_by_default_distinct_arenas(self):
    from tinygrad.schedule import memory as mem
    big, small, _, _ = self._two_jits()
    x = Tensor.randn(64, 256).realize()
    arenas = capture_arenas(lambda: [big(x).realize(), big(x).realize(), small(x).realize(), small(x).realize()])
    self.assertEqual(mem.JIT_SHARED_ARENA.value, 0)
    self.assertEqual(len(arenas), 2)
    self.assertTrue(all(len(a) >= 1 for a in arenas), arenas)  # both captures plan at least one arena

  def test_shared_arena_aliases_captures_and_keeps_results(self):
    import numpy as np
    from tinygrad import Context
    from tinygrad.schedule import memory as mem
    big, small, ref_big, ref_small = self._two_jits()
    x = Tensor.randn(64, 256).realize()
    rb, rs = ref_big(x), ref_small(x)
    mem._shared_arenas.clear()
    with Context(JIT_SHARED_ARENA=1):
      for _ in range(2):
        big(x).realize()
        small(x).realize()   # capture big first, then small
      shared = dict(mem._shared_arenas)
      for _ in range(3):   # alternate replays: both graphs' intermediates alias the same memory, results stay right
        np.testing.assert_allclose(big(x).numpy(), rb, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(small(x).numpy(), rs, rtol=1e-4, atol=1e-3)
    mem._shared_arenas.clear()
    self.assertEqual(len(shared), 1, shared)  # one (device, lane) key on CPU, one arena for both captures

  def test_shared_arena_grows_to_the_largest_capture(self):
    from tinygrad import Context
    from tinygrad.schedule import memory as mem
    big, small, _, _ = self._two_jits()
    x = Tensor.randn(64, 256).realize()
    mem._shared_arenas.clear()
    with Context(JIT_SHARED_ARENA=1):
      for _ in range(2): small(x).realize()   # small first: allocates a small shared arena
      first = next(iter(mem._shared_arenas.values())).max_numel()
      for _ in range(2): big(x).realize()     # bigger: replaces the shared arena with a larger one
      second = next(iter(mem._shared_arenas.values())).max_numel()
    mem._shared_arenas.clear()
    self.assertGreater(second, first)
