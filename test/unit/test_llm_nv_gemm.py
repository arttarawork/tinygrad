"""T4.98f/T4.98j: NV WMMA (mma.sync m16n8k16) prefill gemm for Q8_0/Q4_0/Q4_1/IQ4_NL and the K-quants -- graph builds
for the real shapes, the sm_86 render pins the lane-range split the fragment mapping relies on, and the dispatch gates.
Numerics are hardware-only (t4x/wmma_probe.py, nv_gemm_validate.py)."""
import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels import nv_gemm
from tinygrad.llm.kernels.amd import Q8_0, Q4_0, Q4_1, IQ4_NL, Q4_K, Q5_K, Q6_K, IQ4_XS, QUANT_SIZES
from tinygrad.llm.kernels.nv_quant import NV_CUSTOM_QUANT  # importing nv_quant registers the ContextVar the dispatch test sets

BLOCK = {Q8_0: (32, 34), Q4_0: (32, 18), Q4_1: (32, 20), IQ4_NL: (32, 18), **{t: (256, QUANT_SIZES[t]) for t in (Q4_K, Q5_K, Q6_K, IQ4_XS)}}

def _sink(ggml_type:int, tokens:int, out_features:int, in_features:int):
  weights, block_bytes = BLOCK[ggml_type]
  dt = dtypes.uint8  # the kernel always gets a byte view (amd.py packed_bytes for the word-view formats)
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32).uop
  raw = Tensor.empty(out_features * in_features // weights * block_bytes // dt.itemsize, dtype=dt).uop
  x = Tensor.empty(tokens, in_features, dtype=dtypes.float16).uop
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate((out, raw, x)))
  return nv_gemm._wmma_kernel(*params, out_features=out_features, in_features=in_features, ggml_type=ggml_type)

def _shapes(ggml_type:int):  # (tokens, out, in): 1-4 M tiles, 1/4 N tiles, several K blocks; K a multiple of the format's block
  k = nv_gemm.WMMA_FORMATS[ggml_type][1]
  return ((16, 16, k), (32, 32, 2 * k), (64, 48, 4 * k), (32, 6144, 5120))

class TestKernelGraphAndRender(unittest.TestCase):
  def test_graph_builds_for_every_format(self):
    for ggml_type in nv_gemm.WMMA_FORMATS:
      for tokens, o, i in _shapes(ggml_type):
        with self.subTest(ggml_type=ggml_type, tokens=tokens, out=o):
          sink = _sink(ggml_type, tokens, o, i)
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
    for ggml_type in nv_gemm.WMMA_FORMATS:
      for tokens, o, i in _shapes(ggml_type)[:3]:
        with self.subTest(ggml_type=ggml_type, tokens=tokens, out=o):
          prg = to_program(_sink(ggml_type, tokens, o, i), renderer)
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", src)
          self.assertIn("int lidx0 = threadIdx.x; /* 4 */", src)
          self.assertIn("int lidx1 = threadIdx.y; /* 8 */", src)
          self.assertEqual(src.count("threadIdx.z"), 0)

class TestGates(unittest.TestCase):
  def test_dispatch_stays_generic_off_nv(self):
    # CPU: nv_quant_supported is False, so Linear.__call__ never reaches nv_wmma_linear; and called directly it returns None
    import numpy as np
    from tinygrad import nn, Context
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels.amd import Linear
    rng = np.random.default_rng(0)
    blocks = [np.float16(0.5).tobytes() + rng.integers(-4, 5, 32).astype(np.int8).tobytes() for _ in range(16)]
    packed = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16 * 32, Q8_0).reshape(16, 32)
    lin = Linear(32, 16, bias=False)
    nn.state.load_state_dict(lin, {"weight": decoded}, verbose=False, realize=False)
    x = Tensor.randn(1, 32, 32, dtype=dtypes.float16)
    with Context(NV_CUSTOM_QUANT=1):
      self.assertEqual(NV_CUSTOM_QUANT.value, 1)
      np.testing.assert_allclose(lin(x).numpy(), (x.float() @ decoded.T).numpy(), rtol=1e-3, atol=1e-3)
      lin.set_quantized(lin.weight)
      self.assertEqual(lin.ggml_type, Q8_0)
      self.assertIsNone(nv_gemm.nv_wmma_linear(lin, x))          # CPU: the sm_80 gate is closed
      with Context(NV_WMMA=0): self.assertIsNone(nv_gemm.nv_wmma_linear(lin, x))

  def test_wmma_gate_closed_off_nv(self):
    for dev in ("CPU", "NULL", None, ("CPU", "CPU:1")): self.assertFalse(nv_gemm._nv_wmma_ok(dev), dev)

  def test_tile_helpers(self):
    self.assertEqual([nv_gemm._token_tile_for(t) for t in (16, 32, 48, 64, 128)], [16, 32, 16, 64, 64])
    self.assertEqual([nv_gemm._output_tiles_for(o) for o in (8, 16, 32, 48)], [1, 1, 4, 1])

if __name__ == "__main__": unittest.main()
