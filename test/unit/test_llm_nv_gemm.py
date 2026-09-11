"""T4.98f: NV WMMA (mma.sync m16n8k16) prefill gemm for Q8_0/Q4_0 -- graph builds for the real shapes, the sm_86 render pins the
lane-range split the fragment mapping relies on, and the dispatch gates. Numerics are hardware-only (t4x/wmma_probe.py)."""
import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels import nv_gemm

def _sink(ggml_type_kernel, tokens:int, out_features:int, in_features:int, block_bytes:int):
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32).uop
  raw = Tensor.empty(out_features * in_features // 32 * block_bytes, dtype=dtypes.uint8).uop
  x = Tensor.empty(tokens, in_features, dtype=dtypes.float16).uop
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate((out, raw, x)))
  return ggml_type_kernel(*params, out_features=out_features, in_features=in_features)

class TestKernelGraphAndRender(unittest.TestCase):
  SHAPES = ((16, 16, 32), (32, 32, 64), (64, 48, 128), (32, 6144, 5120))  # (tokens, out, in): 1-4 M tiles, 1/4 N tiles, 1-160 K groups

  def test_graph_builds(self):
    for kern, bb in ((nv_gemm._q8_0_wmma_kernel, 34), (nv_gemm._q4_0_wmma_kernel, 18)):
      for tokens, o, i in self.SHAPES:
        with self.subTest(kernel=kern.__name__, tokens=tokens, out=o):
          sink = _sink(kern, tokens, o, i, bb)
          self.assertIs(sink.op, Ops.SINK)
          self.assertTrue(any(u.op is Ops.WMMA for u in sink.toposort()))

  def test_render_pins_the_lane_split(self):
    # the fragment identity in _wmma_layout_nv assumes the 32-lane LOCAL range renders as lidx0=threadIdx.x (4) = lane//8 and
    # lidx1=threadIdx.y (8) = lane%8, i.e. hardware lane = 4*(lane%8) + lane//8; if the lowerer ever changes that split this
    # must fail before a wrong kernel reaches the card (see _wmma_layout_nv's comment and the 2026-09-11 probe)
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for kern, bb in ((nv_gemm._q8_0_wmma_kernel, 34), (nv_gemm._q4_0_wmma_kernel, 18)):
      for tokens, o, i in self.SHAPES:
        with self.subTest(kernel=kern.__name__, tokens=tokens, out=o):
          prg = to_program(_sink(kern, tokens, o, i, bb), renderer)
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", src)
          self.assertIn("int lidx0 = threadIdx.x; /* 4 */", src)
          self.assertIn("int lidx1 = threadIdx.y; /* 8 */", src)
          self.assertEqual(src.count("threadIdx.z"), 0)

class TestGates(unittest.TestCase):
  def test_wmma_gate_closed_off_nv(self):
    for dev in ("CPU", "NULL", None, ("CPU", "CPU:1")): self.assertFalse(nv_gemm._nv_wmma_ok(dev), dev)

  def test_tile_helpers(self):
    self.assertEqual([nv_gemm._token_tile_for(t) for t in (16, 32, 48, 64, 128)], [16, 32, 16, 64, 64])
    self.assertEqual([nv_gemm._output_tiles_for(o) for o in (8, 16, 32, 48)], [1, 1, 4, 1])

if __name__ == "__main__": unittest.main()
