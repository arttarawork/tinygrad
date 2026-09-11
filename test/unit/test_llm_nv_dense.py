"""T4.98h: the NV dense-linear decode gemv (tinygrad/llm/kernels/nv_dense.py) for small non-ggml-quantized
Linear layers (e.g. GatedDeltaNet's ssm_alpha/ssm_beta head projections, (5120->48) in Qwen3.8-27B).

Everything here runs WITHOUT a GPU (no DEV=NV, no Device["NV"] opened): kernel UOp graphs are built directly
and, when the nvrtc lane is reachable, rendered+compiled for sm_86 through a CUDARenderer built from a bare
Target spec (mirrors test_llm_nv_quant.py's pattern) -- never via an actual NV device. The dispatch gate (which
needs a real NV device to open for real) is exercised by monkeypatching nv_quant.nv_quant_supported and
nv_dense.nv_f16_gemv, exactly as test_llm_nv_quant.py's own docstring describes for that class of test."""
import unittest
from unittest import mock
import numpy as np
from tinygrad import Tensor, UOp, Context, dtypes, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.kernels import nv_dense, nv_quant  # module objects: mock.patch.object targets below, and
                                                      # importing nv_quant registers its NV_CUSTOM_QUANT ContextVar
                                                      # (Context(NV_CUSTOM_QUANT=1) KeyErrors otherwise -- amd.py's
                                                      # Linear.__call__ only imports nv_quant lazily, on first call)
from tinygrad.llm.kernels.nv_dense import nv_dense_eligible, nv_f16_gemv, _nv_f16_gemv_kernel, WARP_SIZE, VAL_CHUNK

# ******** pure-numpy transcription of the kernel's lane/accumulation math ********

def _numpy_gemv(w:np.ndarray, x:np.ndarray, in_features:int) -> np.ndarray:
  """Transcribes _nv_f16_gemv_kernel: w (out_features,in_features), x (tokens,in_features) -> (tokens,out_features),
  fp32 accumulate. Walks in_features the same way the kernel's w.reshape((out,per,128))[..., lane*4+j] indexing
  does -- element i*128 + lane*4 + j for i in range(per), lane in range(32), j in range(4) -- so a transposition
  bug in that indexing would show up here as a wrong (or non-bijective) covering of in_features, not just get
  papered over by re-deriving x @ w.T directly."""
  out_features, tokens, per = w.shape[0], x.shape[0], in_features // (WARP_SIZE*VAL_CHUNK)
  wf = w.astype(np.float32).reshape(out_features, per, WARP_SIZE*VAL_CHUNK)
  xf = x.astype(np.float32).reshape(tokens, per, WARP_SIZE*VAL_CHUNK)
  lane_sums = np.zeros((tokens, out_features, WARP_SIZE), dtype=np.float32)
  for lane in range(WARP_SIZE):
    for j in range(VAL_CHUNK):
      lane_sums[:, :, lane] += np.einsum("oi,ti->to", wf[:, :, lane*VAL_CHUNK+j], xf[:, :, lane*VAL_CHUNK+j])
  return lane_sums.sum(-1)  # warp_reduce: sum the 32 per-lane partials

class TestLaneAccumulationMath(unittest.TestCase):
  def test_matches_reference_matmul(self):
    rng = np.random.default_rng(0)
    for in_features, out_features, tokens in ((5120, 48, 1), (5120, 48, 4), (128, 8, 3), (256, 16, 1)):
      with self.subTest(in_features=in_features, out_features=out_features, tokens=tokens):
        w = rng.standard_normal((out_features, in_features)).astype(np.float32)
        x = rng.standard_normal((tokens, in_features)).astype(np.float32)
        np.testing.assert_allclose(_numpy_gemv(w, x, in_features), x @ w.T, rtol=1e-3, atol=1e-3)

# ******** kernel graph construction + sm_86 render/compile ********

class TestKernelGraphAndRender(unittest.TestCase):
  """(in_features, out_features) shapes: the real Qwen3.8-27B ssm_alpha/ssm_beta shape plus a couple of others,
  all satisfying the in_features % 128 == 0 / out_features <= 2048 dispatch gate."""
  SHAPES = ((5120, 48), (2560, 128), (128, 8))

  @staticmethod
  def _build(in_features:int, out_features:int, tokens:int=1, out_dtype=dtypes.float16):
    out = UOp.placeholder((tokens, out_features), out_dtype, slot=0)
    w = UOp.placeholder((out_features*in_features,), dtypes.float16, slot=1)
    x = UOp.placeholder((tokens, in_features), dtypes.float16, slot=2)
    return _nv_f16_gemv_kernel(out, w, x, in_features=in_features, out_features=out_features, tokens=tokens)

  def test_graph_builds_for_qwen_shape_and_others(self):
    for in_features, out_features in self.SHAPES:
      for tokens in (1, 4):
        with self.subTest(in_features=in_features, out_features=out_features, tokens=tokens):
          sink = self._build(in_features, out_features, tokens)
          self.assertEqual(sink.op, Ops.SINK)

  def test_graph_builds_for_fp32_output_dtype(self):
    # least_upper_dtype(fp16, fp32) in nv_f16_gemv can pick fp32 as the store dtype (an fp32-weight layer)
    sink = self._build(5120, 48, out_dtype=dtypes.float32)
    self.assertEqual(sink.op, Ops.SINK)

  def test_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    from tinygrad.device import CompileError
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    for in_features, out_features in self.SHAPES:
      with self.subTest(in_features=in_features, out_features=out_features):
        sink = self._build(in_features, out_features)
        try: prg = to_program(sink, renderer)
        except CompileError: raise
        except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn("__shfl_xor_sync(0xffffffff", src)
        self.assertNotIn("amdgcn", src)

# ******** nv_dense_eligible: the shape/dtype gate ********

class TestEligibility(unittest.TestCase):
  @staticmethod
  def _linear(in_features:int, out_features:int, dtype=dtypes.float16) -> Linear:
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": Tensor.randn(out_features, in_features, dtype=dtype)}, verbose=False, realize=False)
    return linear

  def test_real_qwen_shape_is_eligible(self):
    self.assertTrue(nv_dense_eligible(self._linear(5120, 48)))

  def test_bfloat16_and_float32_weights_are_eligible_too(self):
    self.assertTrue(nv_dense_eligible(self._linear(5120, 48, dtypes.bfloat16)))
    self.assertTrue(nv_dense_eligible(self._linear(5120, 48, dtypes.float32)))

  def test_out_features_over_2048_is_not_eligible(self):
    self.assertFalse(nv_dense_eligible(self._linear(5120, 4096)))

  def test_in_features_not_a_multiple_of_128_is_not_eligible(self):
    self.assertFalse(nv_dense_eligible(self._linear(5121, 48)))

# ******** nv_f16_gemv: token-count gating + the realize-once weight cache ********

class TestNvF16Gemv(unittest.TestCase):
  IN_FEATURES, OUT_FEATURES = 5120, 48

  def _linear(self) -> Linear:
    linear = Linear(self.IN_FEATURES, self.OUT_FEATURES, bias=False)
    nn.state.load_state_dict(linear, {"weight": Tensor.randn(self.OUT_FEATURES, self.IN_FEATURES, dtype=dtypes.float16)},
                              verbose=False, realize=False)
    return linear

  def test_returns_none_for_more_than_4_tokens(self):
    linear = self._linear()
    self.assertIsNone(nv_f16_gemv(linear, Tensor.randn(5, self.IN_FEATURES, dtype=dtypes.float16)))

  def test_returns_none_for_symbolic_token_count(self):
    linear = self._linear()
    data = np.random.default_rng(0).standard_normal((4, self.IN_FEATURES)).astype(np.float16)
    sym = Tensor(data).contiguous()[:UOp.variable("tokens", 1, 4).bind(2)]
    self.assertIsNone(nv_f16_gemv(linear, sym))

  def test_returns_none_when_not_shape_eligible(self):
    linear = Linear(5121, 8, bias=False)  # in_features not a multiple of 128
    nn.state.load_state_dict(linear, {"weight": Tensor.randn(8, 5121, dtype=dtypes.float16)}, verbose=False, realize=False)
    self.assertIsNone(nv_f16_gemv(linear, Tensor.randn(1, 5121, dtype=dtypes.float16)))

  def test_returns_the_right_shape_for_decode(self):
    linear = self._linear()
    out = nv_f16_gemv(linear, Tensor.randn(1, self.IN_FEATURES, dtype=dtypes.float16))
    self.assertIsNotNone(out)
    self.assertEqual(out.shape, (1, self.OUT_FEATURES))

  def test_dense_weight_realized_once_and_reused(self):
    linear = self._linear()
    nv_f16_gemv(linear, Tensor.randn(1, self.IN_FEATURES, dtype=dtypes.float16))
    first = linear.dense_weight
    self.assertIsNotNone(first)
    nv_f16_gemv(linear, Tensor.randn(2, self.IN_FEATURES, dtype=dtypes.float16))
    self.assertIs(linear.dense_weight, first)

# ******** Linear.__call__ dispatch: gate open/closed, and the disable-forever trap ********

class TestDispatch(unittest.TestCase):
  IN_FEATURES, OUT_FEATURES = 5120, 48

  def _dense_linear(self) -> Linear:
    linear = Linear(self.IN_FEATURES, self.OUT_FEATURES, bias=False)
    nn.state.load_state_dict(linear, {"weight": Tensor.randn(self.OUT_FEATURES, self.IN_FEATURES, dtype=dtypes.float16)},
                              verbose=False, realize=False)
    return linear

  def test_no_custom_kernel_without_env(self):
    linear = self._dense_linear()
    with mock.patch.object(nv_dense, "nv_f16_gemv") as mock_gemv:
      out = linear(Tensor.randn(1, self.IN_FEATURES, dtype=dtypes.float16))
    mock_gemv.assert_not_called()
    self.assertTrue(np.isfinite(out.numpy()).all())
    self.assertIsNone(linear.ggml_type)

  def test_no_custom_kernel_on_cpu_even_with_env(self):
    linear = self._dense_linear()
    with Context(NV_CUSTOM_QUANT=1), mock.patch.object(nv_dense, "nv_f16_gemv") as mock_gemv:
      out = linear(Tensor.randn(1, self.IN_FEATURES, dtype=dtypes.float16))
    mock_gemv.assert_not_called()  # CPU isn't NV: nv_quant_supported stays False regardless of the env
    self.assertTrue(np.isfinite(out.numpy()).all())

  def test_dispatches_to_nv_f16_gemv_when_nv_and_env_are_both_on(self):
    linear = self._dense_linear()
    sentinel = Tensor.zeros(1, self.OUT_FEATURES)
    with Context(NV_CUSTOM_QUANT=1), \
         mock.patch.object(nv_quant, "nv_quant_supported", return_value=True), \
         mock.patch.object(nv_dense, "nv_f16_gemv", return_value=sentinel) as mock_gemv:
      out = linear(Tensor.randn(1, self.IN_FEATURES, dtype=dtypes.float16))
    mock_gemv.assert_called_once()
    self.assertIs(out, sentinel)

  def test_use_custom_quant_survives_a_bigger_than_decode_call(self):
    # a >4-token prefill-shaped call for a gemv-eligible dense layer must NOT permanently disable
    # use_custom_quant -- otherwise every later 1-token decode call for the same layer instance would
    # silently fall back to the generic matmul for the rest of the process's life (T4.98h)
    linear = self._dense_linear()
    with Context(NV_CUSTOM_QUANT=1), mock.patch.object(nv_quant, "nv_quant_supported", return_value=True):
      big = linear(Tensor.randn(8, self.IN_FEATURES, dtype=dtypes.float16))  # tokens=8>4: nv_f16_gemv returns None internally
      self.assertTrue(np.isfinite(big.numpy()).all())
      self.assertTrue(linear.use_custom_quant)  # must not have been permanently disabled by the big call
      sentinel = Tensor.zeros(1, self.OUT_FEATURES)
      with mock.patch.object(nv_dense, "nv_f16_gemv", return_value=sentinel) as mock_gemv:
        out = linear(Tensor.randn(1, self.IN_FEATURES, dtype=dtypes.float16))  # a later 1-token decode call
      mock_gemv.assert_called_once()
      self.assertIs(out, sentinel)

if __name__ == "__main__": unittest.main()
