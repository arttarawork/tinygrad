import io, contextlib, unittest, signal, time
from unittest import mock
from tinygrad import Device, Tensor
from tinygrad.codegen.opt.postrange import Scheduler, args_from_ast
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt import search as search_mod
from tinygrad.helpers import Context

# T4.27: beam_search used to silently return the untouched, unoptimized seed kernel whenever every
# candidate in a round failed to compile/time (_try_compile swallows the exception and returns None).
# These build a real (but hardware-free) ast/Scheduler on the NULL device -- a real renderer/"compiler"
# that never touches a GPU and allocates nothing -- then fake compile failures via a monkeypatched
# _try_compile to exercise beam_search's own control flow without needing real (flaky) hardware.
def _build_seed():
  dev = Device["NULL"]
  with Context(ALLOW_DEVICE_USAGE=1):
    ast = (Tensor.rand(16, 16, device="NULL") + Tensor.rand(16, 16, device="NULL")).schedule_linear().src[-1].src[0]
  s = Scheduler(ast, dev.renderer)
  rawbufs, var_vals = args_from_ast(ast, "NULL")
  return s, rawbufs, var_vals

class TestBeamSearchFailureDiagnostics(unittest.TestCase):
  def setUp(self): self.real_try_compile = search_mod._try_compile
  def tearDown(self): search_mod._try_compile = self.real_try_compile

  def test_all_candidates_fail_warns_and_falls_back(self):
    s, rawbufs, var_vals = _build_seed()
    n_candidates = len(search_mod.get_kernel_actions(s, include_0=False))
    search_mod._try_compile = lambda x: (x[0], None, "BrokenPipeError")
    buf = io.StringIO()
    with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0), contextlib.redirect_stdout(buf):
      result = search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    out = buf.getvalue()
    # loud: names the swallowed exception class and the real tried/failed counts
    self.assertIn("WARNING", out)
    self.assertIn("BrokenPipeError", out)
    self.assertIn(f"{n_candidates} tried", out)
    self.assertIn(f"{n_candidates} failed", out)
    # doesn't silently hand back the untouched seed: falls back to the same heuristics used when BEAM==0
    self.assertIsNot(result, s)
    self.assertEqual(result.applied_opts, hand_coded_optimizations(s.copy()).applied_opts)

  def test_partial_failure_stays_quiet_and_still_optimizes(self):
    s, rawbufs, var_vals = _build_seed()
    real = self.real_try_compile
    def partial_fail(x):
      idx, _cand = x
      if idx < 2: return idx, None, "RuntimeError"  # pretend the first two candidates hit a flaky compile
      return real(x)
    search_mod._try_compile = partial_fail
    buf = io.StringIO()
    with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0), contextlib.redirect_stdout(buf):
      result = search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    out = buf.getvalue()
    self.assertNotIn("WARNING", out)  # some candidates still worked: nothing anomalous to report
    self.assertNotEqual(result.applied_opts, [])

def _forced_getenv(overrides:dict):
  # BEAM_TIMEOUT_SEC/BEAM_UOPS_MAX are read via bare getenv() in search.py, not a Context-visible ContextVar,
  # and getenv() itself is @functools.cache'd process-wide -- an os.environ override only takes effect if it
  # happens to be the very first call with that (key, default) pair in the whole test process, which isn't
  # reliable inside a shared pytest run (T4.27's own partial-failure test above already primed the real
  # defaults). Patching the search_mod.getenv name that _try_compile actually looks up sidesteps both problems.
  real = search_mod.getenv
  def fake(key, default=0): return overrides[key] if key in overrides else real(key, default)
  return fake

# T4.46: _try_compile used to report every failure cause -- the BEAM_TIMEOUT_SEC alarm, the BEAM_UOPS_MAX cap,
# and a genuine compile/runtime error -- as indistinguishable names (RuntimeError, or worse: TimeoutException,
# which isn't even a RuntimeError, so it silently missed the "except RuntimeError" branch's own DEBUG tracing)
# in beam_search's WARNING/Counter. Both forced failure causes now raise a dedicated RuntimeError subclass so
# the Counter names the real cause; T4.27's tests above are untouched by this (their kernel never approaches
# either limit).
class TestBeamFailureCauseNames(unittest.TestCase):
  def test_uop_cap_surfaces_beam_uop_limit(self):
    s, rawbufs, var_vals = _build_seed()
    with mock.patch.object(search_mod, "getenv", _forced_getenv({"BEAM_UOPS_MAX": 1})):
      buf = io.StringIO()
      with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0), contextlib.redirect_stdout(buf):
        search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    out = buf.getvalue()
    self.assertIn("WARNING", out)
    self.assertIn("BeamUopLimit", out)
    self.assertNotIn("RuntimeError", out)  # the whole point: no longer lumped under the generic name

  def test_alarm_timeout_surfaces_beam_compile_timeout(self):
    if not hasattr(signal, "alarm"): self.skipTest("signal.alarm unavailable on this platform")
    s, rawbufs, var_vals = _build_seed()
    def one_candidate(si, include_0=False, max_up=None): return {0: si.copy()}  # keep the round to 1 forced-slow compile
    def slow_to_program(*a, **kw):
      time.sleep(2)  # BEAM_TIMEOUT_SEC is forced to 1 below; SIGALRM must interrupt this well before it returns
      raise AssertionError("alarm should have interrupted the slow compile")  # pragma: no cover
    with mock.patch.object(search_mod, "getenv", _forced_getenv({"BEAM_TIMEOUT_SEC": 1})), \
         mock.patch.object(search_mod, "get_kernel_actions", one_candidate), \
         mock.patch.object(search_mod, "to_program", slow_to_program):
      buf = io.StringIO()
      with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0), contextlib.redirect_stdout(buf):
        search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    out = buf.getvalue()
    self.assertIn("WARNING", out)
    self.assertIn("BeamCompileTimeout", out)
    self.assertNotIn("TimeoutException", out)  # old name: an Exception, not even a RuntimeError -- see NOTES

if __name__ == "__main__":
  unittest.main()
