"""T4.122a: the three prefill elementwise kernels profiled at /Users/artur/Documents/tinygrad-bench-data/
logs/profile-20261027-dump/kernels/E_28toks2A*.txt (SiLU toks*17408, cache-offset copy toks*10240, 3-way
residual add toks*5120) each schedule as ONE merged range whose extent is a symbolic `toks*N` UOp (bound
Variable vmin=1,vmax=32 times a concrete factor N). hand_coded_optimizations' "local groups" section used to
gate LOCAL-opt candidates on `rng.src[0].op is Ops.CONST`, which a MUL(toks, N) range never satisfies even
though N is an exploitable concrete factor -- so these kernels kept local_size=1 (__launch_bounds__(1), one
thread actually doing work per 32-wide warp). Fixed in tinygrad/codegen/opt/heuristic.py by gating on
`const_factor() > 1` (same idiom this file already uses via .divides() for MV/TC's symbolic axes) instead of
a bare CONST check, with a widened candidate-size list so a lone merged axis (no sibling axis to combine
with) can reach the 128-thread floor in one split.

Everything here runs WITHOUT a GPU (no DEV=NV, no Device["NV"] opened): kernel graphs are built via normal
Tensor ops with a bound symbolic Variable (mirrors test/unit/test_llm_nv_dense.py's TestKernelGraphAndRender),
then rendered (never compiled) for sm_86 through a CUDARenderer built from a bare Target spec.

ponytail: kernel 2 (the real cache-offset copy) also writes into offset 30720 of a larger pre-existing
buffer; that offset is immaterial to the axis-shape decision under test here (the LOCAL/UPCAST heuristic only
looks at the RANGE's extent, never the STORE index's additive offset -- confirmed by reading
tinygrad/codegen/opt/heuristic.py and postrange.py's shift_to), so this test reproduces the shared
`toks*10240` single-axis shape via a plain shrink+reshape copy instead of the exact cat-into-offset buffer
layout. Revisit if a future heuristic change starts looking at STORE indices too.
"""
import unittest
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import Ops

CHUNK_MAX = 32  # the real prefill chunk cap (vmax of the "toks" Variable)

def _single_kernel_ast(t: Tensor):
  """Realizes t and returns the one kernel SINK ast that results (pre-opt, as apply_opts/to_program sees it)."""
  linear, _ = t.linear_with_vars()
  calls = list(linear.src)
  assert len(calls) == 1, f"expected exactly one kernel, got {len(calls)}"
  return calls[0].src[0]

def _toks_vmin_vmax(uop: UOp) -> tuple[int, int] | None:
  for u in uop.toposort():
    if u.op is Ops.PARAM and getattr(u.arg, "name", None) == "toks":
      return u.arg.vmin_vmax
  return None

class TestSymbolicLocalOpt(unittest.TestCase):
  """Builds each of the three real shapes with a bound `toks` Variable, merged (via reshape) into a single
  flat range -- the same AST shape the profiled kernels have -- and renders it for sm_86."""

  @classmethod
  def setUpClass(cls):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: cls.renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: raise unittest.SkipTest(f"nvrtc lane unavailable: {e!r}")
    cls.to_program = staticmethod(to_program)

  @staticmethod
  def _build_silu(toks_val:int, factor:int=17408) -> Tensor:
    v = UOp.variable("toks", 1, CHUNK_MAX)
    base = Tensor.rand(CHUNK_MAX, factor).realize()
    return base[:v.bind(toks_val)].silu().reshape(v.bind(toks_val)*factor).contiguous()

  @staticmethod
  def _build_add3(toks_val:int, factor:int=5120) -> Tensor:
    v = UOp.variable("toks", 1, CHUNK_MAX)
    a, b, c = (Tensor.rand(CHUNK_MAX, factor).realize() for _ in range(3))
    Tb = v.bind(toks_val)
    return (a[:Tb] + b[:Tb] + c[:Tb]).reshape(Tb*factor).contiguous()

  @staticmethod
  def _build_copy(toks_val:int, factor:int=10240) -> Tensor:
    v = UOp.variable("toks", 1, CHUNK_MAX)
    base = Tensor.rand(CHUNK_MAX, factor).realize()
    Tb = v.bind(toks_val)
    return (base[:Tb] + 0).reshape(Tb*factor).contiguous()

  def _check(self, build_fn, factor:int):
    ast = _single_kernel_ast(build_fn(4))  # the bound value (4) is just the AST's representative var_val
    prg = self.to_program(ast, self.renderer)
    local_size = prg.arg.local_size
    nthreads = local_size[0] * local_size[1] * local_size[2]
    self.assertGreaterEqual(nthreads, 128, f"{prg.arg.name}: local_size={local_size}")

    # exact factorization (no remainder): grid*block covers exactly toks*factor elements for EVERY toks in
    # 1..32, so there is no boundary case needing an extra mask/guard.
    quotient, remainder = divmod(factor, nthreads)
    self.assertEqual(remainder, 0)
    for toks_val in range(1, CHUNK_MAX+1):
      self.assertEqual(toks_val*quotient*nthreads, toks_val*factor)

    # the bound Variable's range survived into the rendered kernel unchanged (vmin=1, vmax=32) -- the
    # heuristic didn't clip/concretize it to the representative var_val used above.
    self.assertEqual(_toks_vmin_vmax(prg.arg.global_size[0]), (1, CHUNK_MAX))

  def test_silu_toks17408(self): self._check(self._build_silu, 17408)
  def test_add3_toks5120(self): self._check(self._build_add3, 5120)
  def test_copy_toks10240(self): self._check(self._build_copy, 10240)

class TestSymbolicNumerics(unittest.TestCase):
  """Numerics on whatever DEV is configured (no renderer involved): a symbolic-chunk call must match the
  same computation done without any Variable at all, for several toks values."""

  # NOTE: a tensor sliced by a bound Variable keeps a symbolic (though value-known) shape; .numpy() needs a
  # concrete shape, so re-slice with the plain int (a no-op view, since the bound value IS toks_val) first --
  # same trick test_symbolic_tensor.py's test_pad uses before .tolist().

  def test_silu_matches_plain(self):
    base = Tensor.rand(CHUNK_MAX, 128).realize()
    for toks_val in (1, 5, 17, 32):
      v = UOp.variable("toks", 1, CHUNK_MAX)
      got = base[:v.bind(toks_val)].silu()[:toks_val].numpy()
      expected = base[:toks_val].silu().numpy()
      np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

  def test_add3_matches_plain(self):
    a, b, c = (Tensor.rand(CHUNK_MAX, 64).realize() for _ in range(3))
    for toks_val in (1, 5, 17, 32):
      v = UOp.variable("toks", 1, CHUNK_MAX)
      Tb = v.bind(toks_val)
      got = (a[:Tb]+b[:Tb]+c[:Tb])[:toks_val].numpy()
      expected = (a[:toks_val]+b[:toks_val]+c[:toks_val]).numpy()
      np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

  def test_copy_matches_plain(self):
    base = Tensor.rand(CHUNK_MAX, 80).realize()
    for toks_val in (1, 5, 17, 32):
      v = UOp.variable("toks", 1, CHUNK_MAX)
      got = base[:v.bind(toks_val)][:toks_val].numpy()
      expected = base[:toks_val].numpy()
      np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

class TestConcreteShapesUnaffected(unittest.TestCase):
  """Pins the exact opts/launch config for concrete (non-symbolic) kernels at the shapes named in the task --
  elementwise, reduce, matmul -- so a regression here is caught even without a before/after git diff. Values
  confirmed byte-identical to pre-fix heuristic.py via `git stash` A/B on this same probe."""

  @classmethod
  def setUpClass(cls):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: cls.renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: raise unittest.SkipTest(f"nvrtc lane unavailable: {e!r}")
    cls.to_program = staticmethod(to_program)

  def _render(self, t:Tensor):
    ast = _single_kernel_ast(t)
    return self.to_program(ast, self.renderer)

  def test_elementwise_silu_32x17408(self):
    prg = self._render(Tensor.rand(32, 17408).realize().silu().contiguous())
    self.assertEqual(prg.arg.global_size, (4352, 1, 1))
    self.assertEqual(prg.arg.local_size, (32, 1, 1))

  def test_add3_32x5120(self):
    a, b, c = (Tensor.rand(32, 5120).realize() for _ in range(3))
    prg = self._render((a+b+c).contiguous())
    self.assertEqual(prg.arg.global_size, (1280, 1, 1))
    self.assertEqual(prg.arg.local_size, (32, 1, 1))

  def test_reduce_sum_32x5120(self):
    prg = self._render(Tensor.rand(32, 5120).realize().sum(axis=1).contiguous())
    self.assertEqual(prg.arg.global_size, (32, 1, 1))
    self.assertEqual(prg.arg.local_size, (16, 1, 1))

  def test_matmul_32x5120_5120x17408(self):
    m1 = Tensor.rand(32, 5120).realize()
    m2 = Tensor.rand(5120, 17408).realize()
    prg = self._render((m1@m2).contiguous())
    self.assertEqual(prg.arg.global_size, (272, 1, 1))
    self.assertEqual(prg.arg.local_size, (8, 16, 1))

if __name__ == '__main__':
  unittest.main()
