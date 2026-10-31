"""T4.144: attn_finish_nv (kernels/nv_attn.py) -- the merge of the decode/VERIFY attention kernel's split-KV partials, the output gate and
attn_output's q8 activations as one kernel. EXECUTED on the PYTHON emulator (warp shuffle swapped for a shared exchange, test_llm_nv_attn's
_emulated_shfl) against (a) the generic path it replaces -- _merge_partials, `attn * gate.sigmoid()`, q8_quantize -- bit for bit, and
(b) fp64 attention through the real dispatch (the decode mma kernel on the emulator, its registered partials); the sm_86 render; the
decode-step census (merge + gate + q8 -> one kernel) and the routing gates. No GPU, no nvrtc."""
import contextlib, math, pickle, re, sys, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, dtypes, Context
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.llm.kernels import nv_attn, nv_quant, nv_gemm
from tinygrad.llm.kernels.amd import _unbind
from tinygrad.llm.kernels.nv_attn import BLOCK_N, LOG2E, SENTINEL
from test.unit.test_llm_nv_attn import _emulated_shfl, _lower_sm86, _dequant_int4_np, _q8_round_np
from test.unit.test_llm_nv_quant import _emulated_warp_reduce_one_warp

@contextlib.contextmanager
def _emulated():
  # the PYTHON emulator has no CUDA intrinsics: both kernels' shuffles become shared exchanges; the builders' caches bypassed
  with patch.object(nv_attn, "_shfl", _emulated_shfl(1)), patch.object(nv_attn, "_attn_finish_kernel", nv_attn._attn_finish_kernel.__wrapped__), \
       patch.object(nv_quant, "warp_reduce", _emulated_warp_reduce_one_warp), \
       patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__):
    nv_quant.clear_step_caches()
    try: yield
    finally: nv_quant.clear_step_caches()

def _generic(partial:Tensor, stats:Tensor, valid, gate:Tensor, T:int) -> tuple[np.ndarray, np.ndarray]:
  """what the served graph ran before T4.144: the merge, model.py's transpose + `attn * gate.sigmoid()`, nv_q8_linear's q8_quantize"""
  B, R, _, D = partial.shape
  attn = nv_attn._merge_partials(partial, stats, valid).reshape(B, R // T, T, D).transpose(1, 2).reshape(B, T, -1)
  q, scale = nv_quant.q8_quantize(attn * gate.sigmoid(), B * T, R // T * D)
  return q.numpy(), scale.numpy()

def _dequant_q8(q:np.ndarray, scale:np.ndarray) -> np.ndarray:
  return q.view(np.int8).reshape(*scale.shape, 32).astype(np.float64) * scale[..., None]

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestFinishOnEmulator(unittest.TestCase):
  def test_matches_the_generic_path_bit_for_bit(self):
    # synthetic partials in the decode kernel's layout (rows head*T + t): slot maxima spread over +-40 (weights from 1 down to underflow),
    # slots a VERIFY row never saw (m = SENTINEL / log2 e, l = 0, acc = 0), live counts that end mid-step and below the slot count
    # (stale garbage past `live` must not be read), symbolic and concrete fills, T = 1..4, G = 6 like the model and odd ones, merge steps
    # of 8 (the default), 3 and 1 slots
    rng = np.random.default_rng(0)
    for B, H, D, T, slots, start, symbolic, unroll in ((1, 6, 64, 1, 32, 200, True, 8), (1, 24, 256, 3, 256, 1500, True, 8),
                                                       (1, 4, 128, 4, 16, 37, False, 3), (1, 2, 64, 2, 64, 1023, True, 3),
                                                       (2, 3, 64, 1, 8, 2, False, 8), (1, 6, 64, 4, 256, 4093, True, 1)):
      with self.subTest(B=B, H=H, D=D, T=T, slots=slots, start=start, symbolic=symbolic, unroll=unroll):
        R = H * T
        m = rng.uniform(-40, 40, (B, R, slots)).astype(np.float32)
        l = rng.uniform(0.5, 30, (B, R, slots)).astype(np.float32)
        acc = rng.standard_normal((B, R, slots, D)).astype(np.float32) * 5
        empty = rng.random((B, R, slots)) < 0.1
        empty[..., 0] = False  # every row sees the first key (an all-empty row is 0/0 in both paths)
        m[empty], l[empty], acc[empty] = SENTINEL / LOG2E, 0, 0
        gate = (rng.standard_normal((B, T, H * D)) * 2).astype(np.float32)
        partial, stats, g = (Tensor(a, device="PYTHON").contiguous().realize() for a in (acc, np.stack([m, l], -1), gate))
        # the decode mma kernel hands the merge valid * BLOCK_N / 16 (16-key slots)
        valid = (UOp.variable("start_pos", 0, 262143 - T).bind(start) + T) * (BLOCK_N // 16) if symbolic else (start + T) * (BLOCK_N // 16)
        with _emulated(), Context(NV_ATTN_FINISH_UNROLL=unroll):
          q, scale = nv_attn.attn_finish(partial, stats, valid, g, T)
          names = [u.arg.name for u in q.uop.toposort() if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
          got_q, got_scale = q.numpy(), scale.numpy()
          ref_q, ref_scale = _generic(partial, stats, valid, g, T)
        self.assertEqual(names, ["attn_finish_nv"])
        np.testing.assert_array_equal(got_scale.view(np.uint32), ref_scale.view(np.uint32))
        np.testing.assert_array_equal(got_q, ref_q)

  def test_through_the_attention_dispatch(self):
    # the served chain: the decode mma kernel (T == 1) / VERIFY (T = 2..4) on the emulator registers its partials, attn_finish reads them;
    # against the generic merge + gate + q8 bit for bit, and against fp64 attention x sigmoid(gate) to the q8 rounding (half a step)
    rng = np.random.default_rng(1)
    worst = 0.0
    for kvh, G, Dh, N, T, start, slots, warps, d_split in ((1, 6, 64, 128, 1, 99, 4, 2, 1), (1, 6, 128, 128, 3, 98, 2, 2, 2),
                                                           (2, 3, 64, 128, 4, 13, 2, 1, 1), (1, 6, 64, 256, 2, 200, 4, 4, 1)):
      with self.subTest(kvh=kvh, G=G, D=Dh, T=T, start=start, slots=slots, warps=warps, d_split=d_split):
        Hq, valid = kvh * G, start + T
        raw = rng.integers(0, 256, size=(2, 1, kvh, N, Dh // 2), dtype=np.uint8)
        sc = rng.uniform(0.01, 0.2, size=(2, 1, kvh, N, Dh // 32)).astype(np.float16)
        q_np = (rng.standard_normal((1, Hq, T, Dh)) * 2).astype(np.float16)
        gate_np = (rng.standard_normal((1, T, Hq * Dh)) * 2).astype(np.float32)
        kv, sct, q, gate = (Tensor(a, device="PYTHON").contiguous().realize() for a in (raw, sc, q_np, gate_np))
        vl = UOp.variable("start_pos", 0, N - T).bind(start) + T
        with patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True), patch.object(nv_attn, "_decode_mma_kernel", nv_attn._decode_mma_kernel.__wrapped__), \
             Context(NV_ATTN_DEC_SLOTS=slots, NV_ATTN_DEC_WARPS=warps, NV_ATTN_DEC_DSPLIT=d_split), _emulated():
          with patch.object(nv_attn, "_shfl", _emulated_shfl(warps)):  # the attention kernel's warps
            out = nv_attn.nv_decode_attention(q, kv, sct, vl) if T == 1 else nv_attn.nv_verify_attention(q, kv, sct, vl)
          merge = nv_quant.attn_merge_inputs(out)
          self.assertIsNotNone(merge)
          got_q, got_scale = (x.numpy() for x in nv_attn.attn_finish(*merge, gate, T))
          ref_q, ref_scale = _generic(*merge, gate, T)
        np.testing.assert_array_equal(got_scale.view(np.uint32), ref_scale.view(np.uint32))
        np.testing.assert_array_equal(got_q, ref_q)
        kvd = _dequant_int4_np(raw[:, 0], sc[:, 0], Dh)
        ref = np.zeros((T, Hq, Dh))
        for h in range(Hq):
          for t in range(T):
            lim = valid - (T - 1) + t
            s = kvd[0, h // G, :lim] @ _q8_round_np(q_np[0, h, t].astype(np.float32)).astype(np.float64) / math.sqrt(Dh)
            p = np.exp(s - s.max())
            ref[t, h] = p @ kvd[1, h // G, :lim] / p.sum()
        ref = ref.reshape(T, Hq * Dh) / (1 + np.exp(-gate_np[0].astype(np.float64)))
        deq = _dequant_q8(got_q, got_scale).reshape(T, Hq * Dh)
        step = np.repeat(got_scale, 32, axis=-1).reshape(T, Hq * Dh)
        err = np.abs(deq - ref)
        worst = max(worst, float(err.max()))
        # q8 rounding (half a step) + the attention kernel's own fp16/int8 error (test_llm_nv_attn's 2e-3)
        self.assertTrue((err <= 0.5 * step + 2e-3 * (1 + np.abs(ref))).all(), f"max err {err.max():.3e}")
    print(f"\nattn_finish through the dispatch: max |dequant(q8) - fp64 attention x sigmoid(gate)| {worst:.3e}")

class TestRender(unittest.TestCase):
  def _sink(self, T:int, H:int=24, D:int=256, slots:int=256):
    params = [UOp.placeholder((T, H * D // 32, 8), dtypes.uint32, slot=0), UOp.placeholder((T, H * D // 32), dtypes.float32, slot=1),
              UOp.placeholder((1, H * T, slots, D), dtypes.float32, slot=2), UOp.placeholder((1, H * T, slots, 2), dtypes.float32, slot=3),
              UOp.placeholder((T, H * D), dtypes.float32, slot=4)]
    valid = (UOp.variable("start_pos", 0, 262144 - T) + T) * (BLOCK_N // 16)
    return nv_attn._attn_finish_kernel(*params, valid=_unbind(valid), T=T)  # attn_finish unbinds before the cached builder

  def test_render_for_sm86(self):
    # one-warp blocks with the lane unsplit on threadIdx.x (the shuffles pair hardware lanes), the slot count from the bound position,
    # the unrolled merge step (8 slots), and the compile pool's pickle limit
    for T in (1, 3, 4):
      with self.subTest(T=T):
        prg, src = _lower_sm86(self._sink(T))
        self.assertIn("attn_finish_nv", src)
        self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)
        self.assertNotIn("lidx1", src)
        self.assertIn("__launch_bounds__(32)", src)
        self.assertIn("const int data5_", src)  # the bound position: ceil(valid / 16) live slots
        self.assertIn("__shfl_sync(0xffffffff", src)
        # the quantizer divides the gated x as stored (one reciprocal of the group scale), as q8_quantize reading x from memory does --
        # not the symbolic pass's fold num * (1/(den * (1 + e^-g) * scale)), a second rounding of the value q8 rounds
        self.assertEqual(len(re.findall(r"\(1/alu\d+\)", src)), 1)
        self.assertEqual(src.count("for (int Ridx"), 2)  # the row max, the weighted sums
        self.assertEqual(src.count("exp2("), 8 + 4)  # the weights of NV_ATTN_FINISH_UNROLL (8) slots + the 4 features' sigmoids
        self.assertLess(len(src), 150_000)
        self.assertLess(len(pickle.dumps(prg)), 2_000_000)

class TestRoutingAndCensus(unittest.TestCase):
  """graph level on CPU with every NV gate mocked open (test_llm_nv_attn_prep's census): a KV_INT4 decode step / VERIFY run on the decode
  attention kernel ends in attn_finish_nv instead of the merge's reduces, the gate's elementwise and q8_quantize_nv; NV_ATTN_FINISH=0,
  NV_ATTN_PREP=0 (no fp32 gate), and VERIFY on the prefill kernel (NV_ATTN_DEC_MMA_T=0) keep the generic tail."""
  def _census(self, T:int=1, finish:int=1, **ctx) -> list[str]:
    from test.unit.test_llm_nv_attn_prep import _block, TestRoutingAndCensus as Prep
    with Context(**ctx): return Prep._census(Prep(), _block(), T, finish=finish)

  def test_decode_and_verify_take_one_kernel(self):
    for T in (1, 3, 4):
      with self.subTest(T=T):
        on, off = self._census(T), self._census(T, finish=0)
        print(f"\nT={T}: {len(off)} kernels -> {len(on)}")
        self.assertEqual(on.count("attn_finish_nv"), 1)
        self.assertNotIn("attn_finish_nv", off)
        self.assertEqual(off.count("q8_quantize_nv") - on.count("q8_quantize_nv"), 1)  # attn_output's quantize
        tail = [k for k in off if k not in on]
        self.assertTrue(tail and all(k.startswith(("reduce", "elementwise", "q8_quantize_nv")) for k in tail), tail)
        self.assertLessEqual(len(on), len(off) - 3, "merge (>= 2 reduces) + gate/q8 -> one kernel")

  def test_generic_tails_unchanged(self):
    self.assertNotIn("attn_finish_nv", self._census(1, NV_ATTN_PREP=0), "no attn_prep: the gate is the generic graph's")
    self.assertNotIn("attn_finish_nv", self._census(3, NV_ATTN_DEC_MMA_T=0), "VERIFY on the prefill kernel: no registered partials")

if __name__ == "__main__": unittest.main()
