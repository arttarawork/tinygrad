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
from tinygrad.llm.model import Transformer
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

  def test_model_prefill_arena_is_two_score_slots(self):
    # the tiny attention model's prefill family: (B,H,T,Tk_max) fp32 slots with H=2, T=32 -> ~2 slots + small stuff, not 3
    cfg = replace(TEST_CONFIG, max_context=8192)
    def run():
      g = Transformer(cfg).generate(list(range(1, 41)), chunk_size=32)
      for _ in range(3): next(g)
    arenas = capture_arenas(run)
    slot = cfg.n_heads * 32 * (cfg.max_context + 32) * 4
    self.assertLess(max(arenas[0]), 2.3 * slot, f"prefill arena {max(arenas[0])/1e6:.2f} MB vs slot {slot/1e6:.2f} MB")

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
