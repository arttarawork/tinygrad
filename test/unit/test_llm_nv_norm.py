"""T4.123d: the fused residual-add + RMSNorm + q8-quantize decode kernel (tinygrad/llm/kernels/nv_quant.py's
_add_rmsnorm_q8_kernel / add_rmsnorm_q8), design doc section 3 (P4 kernel half). This file only covers the
kernel itself; the model.py wiring (T4.123e: FFNBlock._ffn_output, GatedDeltaNetBlock._normed_for_attention,
nv_quant.py's register_fused_q8) is tested in test_llm_nv_quant.py instead.

Everything here runs WITHOUT a GPU (no DEV=NV, no Device["NV"] opened), mirroring test_llm_nv_quant.py's and
test_llm_nv_dense.py's own docstrings: kernel UOp graphs are built directly (bypassing Tensor.custom_kernel for
the graph-shape/render checks, exactly as those files' TestKernelGraphAndRender classes do) and, when the
nvrtc lane is reachable, rendered+compiled for sm_86 through a CUDARenderer built from a bare Target spec --
never via an actual NV device. The kernel uses AxisType.LOCAL/AddrSpace.LOCAL (a real threadgroup, >1 warp),
which CPU's renderer doesn't support at all (attn_kernel.py's own docstring: "a renderer with has_local=False
... has no threadgroup/shared-memory model"), so unlike the single-warp kernels in those two files this one
can never be realized on CPU even for a toy shape -- math parity is checked with a NumPy transcription of the
intended add -> RMSNorm -> q8 pipeline instead (same idea as test_llm_nv_dense.py's _numpy_gemv), using
test_llm_amd.py's own q8_quantize reference formula (scale=max(abs)/127 clamped 1e-8, q=clip(rint(x/scale))).
T4.125b adds TestKernelRunsOnPythonEmulator: the kernel itself executed on tinygrad's PYTHON emulator device (which does
model a threadgroup), with only the warp shuffle swapped for an equivalent shared-memory exchange."""
import itertools, sys, unittest
from unittest import mock
import numpy as np
from tinygrad import Tensor, UOp, dtypes, Context
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops
from tinygrad.llm.kernels import nv_quant
from tinygrad.llm.kernels.amd import Q8_GROUP_SIZE
from tinygrad.llm.kernels.nv_quant import _add_rmsnorm_q8_kernel, _q8_pack, add_rmsnorm_q8

# ******** pure-numpy reference: add -> RMSNorm -> q8, transcribing the kernel's own arithmetic ********

def np_add_rmsnorm_q8(residual:np.ndarray, partials:np.ndarray|None, weight:np.ndarray, eps:float):
  """residual (tokens,dim), partials (tokens,dim,C) or None, weight (dim,). Returns (new_residual, q (int8,
  tokens,groups,32), scale (tokens,groups), normed (tokens,dim) fp32) -- q/scale use EXACTLY _q8_quantize_kernel's
  formula (test_llm_amd.py's TestQ8Quantize.test_values_and_scales reference): scale=max(abs)/127 clamped at
  1e-8, q=clip(rint(x/scale),-127,127). RMSNorm computed as sqrt-then-reciprocal (nn.RMSNorm._norm's .rsqrt()
  decomposition, tensor.py: self.sqrt().reciprocal()), matching the kernel's op order bit-for-bit at fp64/fp32
  numpy precision (reduction-order differences vs the kernel's warp/cross-warp tree are within float tolerance,
  same spirit as test_llm_nv_dense.py's _numpy_gemv not replicating the GPU's lane-summation order either)."""
  tokens, dim = residual.shape
  new_residual = residual + (partials.sum(-1) if partials is not None else 0.0)
  rms = np.sqrt((new_residual.astype(np.float64)**2).mean(-1, keepdims=True) + eps)
  inv_rms = 1.0 / rms
  normed = (new_residual * inv_rms * weight).astype(np.float32)
  groups = dim // Q8_GROUP_SIZE
  g = normed.reshape(tokens, groups, Q8_GROUP_SIZE)
  scale = np.maximum(np.max(np.abs(g), axis=-1, keepdims=True) / 127, 1e-8)
  q = np.clip(np.rint(g / scale), -127, 127).astype(np.int8)
  return new_residual.astype(np.float32), q, scale[..., 0], normed

class TestNumpyReferenceMatchesQ8QuantizeFormula(unittest.TestCase):
  """Sanity-check the reference itself against test_llm_amd.py's own q8_quantize formula before trusting it
  to stand in for the kernel (which can't run without a GPU, see the module docstring)."""
  def test_pure_norm_no_partials_matches_manual_rmsnorm_plus_quantize(self):
    rng = np.random.default_rng(0)
    tokens, dim = 2, 64
    residual = rng.standard_normal((tokens, dim)).astype(np.float32)
    weight = rng.standard_normal(dim).astype(np.float32)
    eps = 1e-6
    new_residual, q, scale, normed = np_add_rmsnorm_q8(residual, None, weight, eps)
    np.testing.assert_array_equal(new_residual, residual)
    expected_normed = residual * (1.0 / np.sqrt((residual**2).mean(-1, keepdims=True) + eps)) * weight
    np.testing.assert_allclose(normed, expected_normed, rtol=1e-5, atol=1e-6)
    groups = dim // Q8_GROUP_SIZE
    g = expected_normed.reshape(tokens, groups, Q8_GROUP_SIZE)
    expected_scale = np.maximum(np.max(np.abs(g), axis=-1) / 127, 1e-8)
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-6)
    expected_q = np.clip(np.rint(g / expected_scale[..., None]), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(q, expected_q)

  def test_partials_are_summed_before_the_norm(self):
    rng = np.random.default_rng(1)
    tokens, dim, C = 1, 32, 5
    residual = rng.standard_normal((tokens, dim)).astype(np.float32)
    partials = rng.standard_normal((tokens, dim, C)).astype(np.float32)
    weight = np.ones(dim, dtype=np.float32)
    new_residual, _, _, _ = np_add_rmsnorm_q8(residual, partials, weight, 1e-6)
    np.testing.assert_allclose(new_residual, residual + partials.sum(-1), rtol=1e-5, atol=1e-6)

class TestKernelGraphAndRender(unittest.TestCase):
  """Build the kernel's UOp graph directly (bypassing Tensor.custom_kernel/add_rmsnorm_q8's Tensor plumbing,
  exactly like test_llm_nv_quant.py's/test_llm_nv_dense.py's TestKernelGraphAndRender) and, where the nvrtc
  lane is reachable, render+compile it for sm_86."""
  TOKENS, DIM, THREADS = 1, 5120, 256  # the standing Qwen3.8-27B norm-site shape

  @staticmethod
  def _placeholders(tokens:int, dim:int, C:int, want_fp16:bool) -> list[UOp]:
    groups = dim // Q8_GROUP_SIZE
    params = [
      UOp.placeholder((tokens, dim), dtypes.float32, slot=0),          # new_residual
      UOp.placeholder((tokens, groups, 8), dtypes.uint32, slot=1),     # q
      UOp.placeholder((tokens, groups), dtypes.float32, slot=2),       # scale
      UOp.placeholder((tokens, dim), dtypes.float32, slot=3),          # residual
      UOp.placeholder((dim,), dtypes.float32, slot=4),                 # weight
    ]
    if C: params.append(UOp.placeholder((tokens, dim, C), dtypes.float32, slot=5))      # partials
    if want_fp16: params.append(UOp.placeholder((tokens, dim), dtypes.float16, slot=6))  # fp16_out
    return params

  def _build(self, tokens:int|None=None, dim:int|None=None, threads:int|None=None, C:int=0, want_fp16:bool=False,
             eps:float=1e-6, split:int=1) -> UOp:
    tokens, dim, threads = tokens or self.TOKENS, dim or self.DIM, threads or self.THREADS
    params = self._placeholders(tokens, dim, C, want_fp16)
    return _add_rmsnorm_q8_kernel(*params, tokens=tokens, dim=dim, eps=eps, C=C, want_fp16=want_fp16, threads=threads, split=split)

  def test_graph_builds_for_every_partials_and_fp16_combo(self):
    for C in (0, 6):
      for want_fp16 in (False, True):
        with self.subTest(C=C, want_fp16=want_fp16):
          sink = self._build(C=C, want_fp16=want_fp16)
          self.assertEqual(sink.op, Ops.SINK)

  def test_graph_builds_for_multiple_tokens(self):
    # one block per token: tokens>1 must build the same shape of graph (tok is the AxisType.GLOBAL range)
    for tokens in (1, 4):
      with self.subTest(tokens=tokens):
        sink = self._build(tokens=tokens)
        self.assertEqual(sink.op, Ops.SINK)

  def test_uses_a_real_threadgroup_not_a_single_warp(self):
    # the whole point of this task vs. q8_quantize_nv's one warp: >=256 threads, a block-level (not warp-level)
    # reduction. The 32 hardware lanes must be their OWN dedicated AxisType.WARP range (axis_id -1, same
    # convention amd.py's _amd_flash_attention/codegen/opt/postrange.py's tensor-core path use) -- NOT derived
    # via //32 and %32 from one big AxisType.LOCAL range, which add_gpudims is free to decompose into
    # (threadIdx.x, threadIdx.y) in a shape that does NOT line up with CUDA's real hardware warp boundary
    # (confirmed by an earlier draft of this kernel: a single 256-wide LOCAL range rendered as
    # local_size=(8,32), which does not preserve warp_reduce's __shfl_xor_sync grouping -- see nv_quant.py's
    # comment at the warp/lane declaration).
    sink = self._build()
    nodes = list(sink.toposort())
    local_ranges = [u for u in nodes if u.op is Ops.RANGE and u.arg[-1] is AxisType.LOCAL]
    warp_ranges = [u for u in nodes if u.op is Ops.RANGE and u.arg[-1] is AxisType.WARP]
    self.assertEqual(len(local_ranges), 1, "the warp-group axis (0..threads/32-1)")
    self.assertEqual(len(warp_ranges), 1, "the dedicated 32-lane axis -- NOT a //32 split of one big LOCAL range")
    self.assertEqual(local_ranges[0].src[0].arg, self.THREADS // 32)
    self.assertEqual(warp_ranges[0].src[0].arg, 32)
    self.assertGreater(self.THREADS, 32)
    local_bufs = {u.arg.slot for u in nodes if u.op is Ops.BUFFER and u.arg.addrspace is AddrSpace.LOCAL}
    self.assertEqual(len(local_bufs), 2, "val_buf + wsum_buf")

  def test_bad_threads_not_multiple_of_32_raises(self):
    with self.assertRaises(AssertionError):
      self._build(threads=100)

  def test_threads_not_dividing_dim_raises(self):
    # 384 % 32 == 0 (passes the first half of the assert) but 5120 % 384 != 0
    with self.assertRaises(AssertionError):
      self._build(dim=5120, threads=384)

  def test_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    from tinygrad.device import CompileError
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    # T4.125b: 1024 = the default at dim 5120; T4.137: split 5 = its default dim/threads blocks per token
    for C, want_fp16, (threads, split) in itertools.product((0, 6), (False, True), ((256, 1), (1024, 1), (1024, 5))):
      with self.subTest(C=C, want_fp16=want_fp16, threads=threads, split=split):
        sink = self._build(C=C, want_fp16=want_fp16, threads=threads, split=split)
        try: prg = to_program(sink, renderer)
        except CompileError: raise
        except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn(f"__launch_bounds__({threads})", src)
        self.assertIn("__shfl_xor_sync(0xffffffff", src)   # warp_reduce (both the sq and the max-abs reduce)
        self.assertIn("__syncthreads()", src)               # the whole point: add_raw_barrier fired on val_buf/wsum_buf
        self.assertIn("__shared__", src)
        self.assertNotIn("amdgcn", src)

class TestQ8Pack(unittest.TestCase):
  """_q8_pack is a 1:1 lift of _q8_quantize_kernel's word-packing formula. Its round+clip+cast-to-int8 stage
  (the numerically meaningful part -- the signed value ggml's dequant expects) is checked here by running the
  UOp graph through tinygrad's CPU/NULL interpreter via Tensor(...).item() -- no AxisType.LOCAL/RANGE involved
  in this sub-expression, so unlike the rest of this kernel it CAN actually execute without a GPU -- against
  the same reference test_llm_amd.py's TestQ8Quantize.test_values_and_scales uses for the real q8_quantize_nv
  kernel (scale=max(abs)/127 clamped 1e-8, q=clip(rint(x/scale),-127,127)).

  NOT checked the same way: the final packed uint32 WORD (int8 -> uint8 -> uint32, 4 lanes shifted+summed).
  Found while writing this test: tinygrad's CPU/NULL interpreter SIGN-extends an int8->uint8->uint32 cast
  chain for negative source values instead of zero-extending it -- e.g. UOp.const(-127,dtypes.int8)
  .cast(dtypes.uint8).cast(dtypes.uint32) evaluates via .item() to 4294967169 (0xFFFFFF81), not 129 -- a
  pre-existing bit-reinterpretation bug in that interpreter's own constant handling, unrelated to this task:
  it would hit _q8_quantize_kernel's OWN identical cast chain exactly the same way, if anything had ever run
  it through this interpreter path (nothing in this repo does -- test_llm_amd.py skips without real hardware,
  test_llm_nv_quant.py never executes past graph-build/render). It does NOT affect the real target: CUDA's
  unsigned-to-unsigned widening is well-defined zero-extension (the dumped q8_quantize_nv.txt kernel's own
  generated C -- `(uint)((unsigned char)((signed char)alu))` -- is correct by the C standard), and
  _quant_decode_kernel's read side never re-derives these bytes through a cast chain at all, only raw memory
  loads. So the packed WORD's layout is checked structurally here (uint32 dtype) and exercised for real by
  TestKernelGraphAndRender's sm86 compile instead of by CPU execution."""
  def test_round_clip_to_int8_matches_q8_quantize_reference(self):
    rng = np.random.default_rng(2)
    for _ in range(50):
      scale = float(rng.uniform(0.01, 5.0))
      v = float(rng.uniform(-300, 300))
      chain = (UOp.const(v, dtypes.float32) / UOp.const(scale, dtypes.float32)).round().clip(-127, 127).cast(dtypes.int8)
      got = int(Tensor(chain).item())
      expected = int(np.clip(np.rint(np.float32(v) / np.float32(scale)), -127, 127).astype(np.int8))
      self.assertEqual(got, expected, f"v={v} scale={scale}")

  def test_returns_a_uint32_expression_from_four_inputs(self):
    xs = tuple(UOp.const(float(i), dtypes.float32) for i in range(4))
    word = _q8_pack(xs, UOp.const(1.0, dtypes.float32))
    self.assertEqual(word.dtype, dtypes.uint32)

class TestWrapperBuildsExpectedShapes(unittest.TestCase):
  """add_rmsnorm_q8 (the thin Tensor-level wrapper) must produce the right shapes/dtypes without ever
  realizing the graph (CPU/NULL can't run AddrSpace.LOCAL at all -- see the module docstring): Tensor shape/
  dtype are graph metadata, available without a device opening or a kernel ever running."""
  DIM = 5120

  def test_shapes_without_partials_or_fp16(self):
    residual = Tensor.empty(1, self.DIM, dtype=dtypes.float32)
    weight = Tensor.empty(self.DIM, dtype=dtypes.float32)
    new_residual, q, scale, fp16_out = add_rmsnorm_q8(residual, weight, eps=1e-6)
    groups = self.DIM // Q8_GROUP_SIZE
    self.assertEqual(new_residual.shape, (1, self.DIM))
    self.assertEqual(new_residual.dtype, dtypes.float32)
    self.assertEqual(q.shape, (1, groups, 8))
    self.assertEqual(q.dtype, dtypes.uint32)
    self.assertEqual(scale.shape, (1, groups))
    self.assertEqual(scale.dtype, dtypes.float32)
    self.assertIsNone(fp16_out)

  def test_shapes_with_partials_and_fp16(self):
    tokens, C = 4, 6
    residual = Tensor.empty(tokens, self.DIM, dtype=dtypes.float32)
    partials = Tensor.empty(tokens, self.DIM, C, dtype=dtypes.float32)
    weight = Tensor.empty(self.DIM, dtype=dtypes.float32)
    new_residual, q, scale, fp16_out = add_rmsnorm_q8(residual, weight, eps=1e-6, partials=partials, want_fp16=True)
    groups = self.DIM // Q8_GROUP_SIZE
    self.assertEqual(new_residual.shape, (tokens, self.DIM))
    self.assertEqual(q.shape, (tokens, groups, 8))
    self.assertEqual(scale.shape, (tokens, groups))
    self.assertIsNotNone(fp16_out)
    assert fp16_out is not None
    self.assertEqual(fp16_out.shape, (tokens, self.DIM))
    self.assertEqual(fp16_out.dtype, dtypes.float16)

  def test_bad_threads_dim_combo_raises(self):
    residual = Tensor.empty(1, self.DIM, dtype=dtypes.float32)
    weight = Tensor.empty(self.DIM, dtype=dtypes.float32)
    with self.assertRaises(AssertionError):
      add_rmsnorm_q8(residual, weight, eps=1e-6, threads=384)

  def test_default_threads_is_the_widest_block(self):
    # T4.125b: <= 1024 threads, a multiple of 32 dividing dim -- 1024 at the standing 5120 (the 256 default was latency-bound).
    # T4.137: dim/threads blocks per token (NV_NORM_SPLIT, default 1), one with NV_NORM_SPLIT=0
    for dim, threads in ((5120, 1024), (256, 256), (2560, 640)):
      for split in (1, 0):
        with self.subTest(dim=dim, split=split), Context(NV_NORM_SPLIT=split):
          new_residual = add_rmsnorm_q8(Tensor.empty(2, dim, dtype=dtypes.float32), Tensor.empty(dim, dtype=dtypes.float32), eps=1e-6)[0]
          ranges = [u for u in new_residual.uop.toposort() if u.op is Ops.CALL for u in u.src[0].toposort() if u.op is Ops.RANGE]
          self.assertEqual([u.src[0].arg for u in ranges if u.arg[-1] is AxisType.LOCAL], [threads // 32])
          self.assertEqual([u.src[0].arg for u in ranges if u.arg[-1] is AxisType.GLOBAL], [2 * (dim // threads if split else 1)])

_emu_slot = itertools.count(60)
def _emulated_warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  """warp_reduce's xor butterfly with each __shfl_xor_sync(v, off) replaced by a block-shared exchange: thread tid writes v at [tid]
  and reads [tid ^ off] -- the same partner a 32-lane xor shuffle reads, since tid = warp*32 + lane. The PYTHON emulator runs the
  whole block in lockstep (barriers are no-ops), so the store is always seen."""
  assert full_wave
  rngs = {u.arg[-1]: u for u in val.toposort() if u.op is Ops.RANGE}
  lane, warp = rngs[AxisType.WARP], rngs[AxisType.LOCAL]
  tid = warp * 32 + lane
  for off in (16, 8, 4, 2, 1):
    buf = UOp.placeholder((warp.src[0].arg * 32,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
    partner = tid + off - (tid // off) % 2 * (2 * off)  # tid ^ off for a power-of-two off, spelled so CHECK_OOB's z3 can bound it
    other = buf.after(buf[tid].store(val))[partner].load()
    val = val.maximum(other) if maximum else val + other
  return val

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestKernelRunsOnPythonEmulator(unittest.TestCase):
  """T4.125b: the kernel itself EXECUTED (not a transcription), on tinygrad's PYTHON emulator device -- every thread mapping,
  both phases, the barrier-ordered shared buffers, the q8 packing -- with only the warp shuffle swapped for an equivalent shared
  exchange (_emulated_warp_reduce; the emulator has no CUDA intrinsics). Compared against np_add_rmsnorm_q8. The standing shape:
  dim 5120 at the default 1024 threads with C=6 split-K partials (ssm_out/attn_output), plus the old 256-thread layout."""
  def _run(self, tokens:int, dim:int, C:int, threads:int|None, want_fp16:bool):
    rng = np.random.default_rng(dim + C)
    residual, weight = rng.standard_normal((tokens, dim)).astype(np.float32), rng.standard_normal(dim).astype(np.float32)
    partials = rng.standard_normal((tokens, dim, C)).astype(np.float32) if C else None
    with mock.patch.object(nv_quant, "warp_reduce", _emulated_warp_reduce), \
         mock.patch.object(nv_quant, "_add_rmsnorm_q8_kernel", nv_quant._add_rmsnorm_q8_kernel.__wrapped__):  # bypass the cache
      outs = add_rmsnorm_q8(Tensor(residual, device="PYTHON"), Tensor(weight, device="PYTHON"), 1e-6, want_fp16=want_fp16, threads=threads,
                            partials=None if partials is None else Tensor(partials, device="PYTHON"))
      new_residual, q, scale = (t.numpy() for t in outs[:3])
      fp16 = None if outs[3] is None else outs[3].numpy()
    return residual, partials, weight, new_residual, q, scale, fp16

  def _check(self, tokens:int, dim:int, C:int, threads:int|None, want_fp16:bool):
    residual, partials, weight, new_residual, q, scale, fp16 = self._run(tokens, dim, C, threads, want_fp16)
    ref_residual, ref_q, ref_scale, ref_normed = np_add_rmsnorm_q8(residual, partials, weight, 1e-6)
    # partials summed left to right, then the residual: the same order numpy's 6-element sum + add uses -> exact
    np.testing.assert_array_equal(new_residual, ref_residual)
    np.testing.assert_allclose(scale, ref_scale, rtol=1e-6)
    np.testing.assert_array_equal(q.view(np.int8).reshape(ref_q.shape), ref_q)
    if want_fp16: np.testing.assert_allclose(fp16, ref_normed.astype(np.float16), rtol=1e-3, atol=1e-3)

  def test_standing_shape_1024_threads_with_partials(self): self._check(2, 5120, 6, None, True)
  def test_256_threads_without_partials(self): self._check(1, 256, 0, 256, False)

  def test_split_blocks_are_bit_identical_to_one_block(self):
    # T4.137: NV_NORM_SPLIT (default 1) runs dim/threads = 5 blocks per token at the standing shape, each repeating phase 1 and doing a fifth
    # of phase 2 -- every output equal to the one-block kernel's (NV_NORM_SPLIT=0), bit for bit; VERIFY's 3 tokens = 15 blocks
    for C, want_fp16 in ((6, False), (0, True)):
      with self.subTest(C=C, want_fp16=want_fp16):
        with Context(NV_NORM_SPLIT=1): split = self._run(3, 5120, C, None, want_fp16)[3:]
        with Context(NV_NORM_SPLIT=0): one = self._run(3, 5120, C, None, want_fp16)[3:]
        for a, b in zip(split, one):
          if a is not None: np.testing.assert_array_equal(a, b)

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestPrefillWidthOnPythonEmulator(unittest.TestCase):
  """T4.141: the kernel at a prefill chunk's padded width, executed on the emulator: the residual is the chunk's (1, toks, dim) view of a
  (1, tokens, dim) buffer whose rows past toks hold NaN (stale arena memory), the partials come in the IMMA gemm's (C, tokens, dim)
  layout. The chunk's rows match the numpy pipeline, every padded row is the generic zero pad's (residual 0, q 0, scale 1e-8, fp16 0)."""
  TOKENS, TOKS, DIM = 16, 11, 256

  def _run(self, C:int, want_fp16:bool=False, q8_fp16:bool=False, threads:int|None=64):
    rng = np.random.default_rng(C)
    residual = rng.standard_normal((1, self.TOKENS, self.DIM)).astype(np.float32)
    residual[:, self.TOKS:] = np.nan
    weight = rng.standard_normal(self.DIM).astype(np.float32)
    partials = rng.standard_normal((C, self.TOKENS, self.DIM)).astype(np.float32)
    with mock.patch.object(nv_quant, "warp_reduce", _emulated_warp_reduce), \
         mock.patch.object(nv_quant, "_add_rmsnorm_q8_kernel", nv_quant._add_rmsnorm_q8_kernel.__wrapped__):
      view = Tensor(residual, device="PYTHON")[:, :UOp.variable("toks", 1, self.TOKENS).bind(self.TOKS)]
      outs = add_rmsnorm_q8(view, Tensor(weight, device="PYTHON"), 1e-6, partials=Tensor(partials, device="PYTHON"), want_fp16=want_fp16,
                            threads=threads, q8_fp16=q8_fp16)
      got = [None if t is None else t.numpy() for t in outs]
    return residual[0], partials, weight, got

  def test_chunk_rows_match_numpy_and_padded_rows_are_the_zero_pad(self):
    n = self.TOKS
    for C, threads in ((3, 64), (1, 256)):  # 64 threads: 4 elements per thread, 4 blocks per token (NV_NORM_SPLIT)
      with self.subTest(C=C, threads=threads):
        residual, partials, weight, (new_residual, q, scale, _) = self._run(C, threads=threads)
        ref_residual, ref_q, ref_scale, _ = np_add_rmsnorm_q8(residual[:n], partials[:, :n].transpose(1, 2, 0), weight, 1e-6)
        np.testing.assert_array_equal(new_residual[:n], ref_residual)
        np.testing.assert_array_equal(q[:n].view(np.int8).reshape(ref_q.shape), ref_q)
        np.testing.assert_allclose(scale[:n], ref_scale, rtol=1e-6)
        np.testing.assert_array_equal(new_residual[n:], 0)
        np.testing.assert_array_equal(q[n:], 0)
        np.testing.assert_array_equal(scale[n:], np.float32(1e-8))

  def test_q8_fp16_quantizes_the_fp16_copy(self):
    # GDN's prefill attn_norm site: q/scale must be q8_quantize OF the fp16 copy (what _nv_prefill's projections read), not of the fp32
    # value -- the late codegen folds a plain cast(half).float() back to fp32, so the kernel stages the rounded values in shared memory
    from test.unit.test_llm_nv_gemm import _emulated_lane_reduce
    n = self.TOKS
    residual, partials, weight, (_, q, scale, fp16) = self._run(2, want_fp16=True, q8_fp16=True)
    ref_normed = np_add_rmsnorm_q8(residual[:n], partials[:, :n].transpose(1, 2, 0), weight, 1e-6)[3]
    np.testing.assert_allclose(fp16[:n].astype(np.float32), ref_normed.astype(np.float16).astype(np.float32), rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(fp16[n:], 0)
    with mock.patch.object(nv_quant, "warp_reduce", _emulated_lane_reduce), \
         mock.patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__):
      ref_q, ref_scale = (t.numpy() for t in nv_quant.q8_quantize(Tensor(fp16, device="PYTHON"), self.TOKENS, self.DIM))
    np.testing.assert_array_equal(q, ref_q)
    np.testing.assert_array_equal(scale, ref_scale)

class TestPrefillWidthRender(unittest.TestCase):
  """T4.141: the sm_86 source (no nvrtc lane: test_llm_nv_scan._render_cuda) of the prefill kernel at the 27B shape -- 128 padded tokens x
  5 blocks of 1024 threads, ssm_out/attn_output's 4 split-K partials in the IMMA layout -- and the decode renders, md5-pinned at
  c685345f4 (before T4.141): the prefill variant must leave them byte-identical."""
  DECODE_MD5 = {(1, 0, False): "ddc3ffdc7546feef07aa2144bd9b3f91", (1, 5, True): "8c7127624561d0ceb175dc802de5a74a",
                (4, 5, False): "3cb0ffc632af9591d401b8da86857fa4", (4, 0, True): "cfaa4b5a10f8ba643aef0f543e61653e"}

  @staticmethod
  def _source(residual:Tensor, partials:Tensor|None=None, want_fp16:bool=False, q8_fp16:bool=False) -> str:
    from test.unit.test_llm_nv_scan import _render_cuda
    q = add_rmsnorm_q8(residual, Tensor.empty(5120), 1e-6, partials=partials, want_fp16=want_fp16, q8_fp16=q8_fp16)[1]
    return _render_cuda(next(u.src[0] for u in q.uop.toposort() if u.op is Ops.CALL))

  def test_decode_renders_unchanged(self):
    import hashlib
    for (tokens, C, want_fp16), digest in self.DECODE_MD5.items():
      with self.subTest(tokens=tokens, C=C, want_fp16=want_fp16):
        src = self._source(Tensor.empty(tokens, 5120), Tensor.empty(tokens, 5120, C) if C else None, want_fp16)
        self.assertEqual(hashlib.md5(src.encode()).hexdigest(), digest)

  def test_prefill_kernel_at_the_27b_shape(self):
    view = Tensor.empty(1, 128, 5120)[:, :UOp.variable("toks", 1, 128).bind(77)]
    for want_fp16 in (False, True):
      with self.subTest(want_fp16=want_fp16):
        src = self._source(view, Tensor.empty(4, 128, 5120), want_fp16, q8_fp16=want_fp16)
        self.assertIn("__launch_bounds__(1024)", src)
        self.assertRegex(src, r"const int data\d+_\) \{")  # the chunk width is the last argument: one kernel serves every remainder chunk
        self.assertEqual(src.count("__syncthreads()"), 2 if want_fp16 else 1)  # q8_fp16: the staged fp16 values' own barrier
        self.assertEqual("half" in src, want_fp16)

if __name__ == "__main__": unittest.main()
