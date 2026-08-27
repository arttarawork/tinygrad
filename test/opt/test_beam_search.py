import io, contextlib, unittest, signal, time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.codegen.opt.postrange import Scheduler, args_from_ast
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt import search as search_mod
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import Context
from tinygrad.renderer import Renderer
from tinygrad.helpers import Target

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
# T4.48 (T4.47_RCA.md): a BEAM candidate abandoned for being slow (search.py's own device-side timeout,
# BEAM_DEV_TIMEOUT) and a candidate that FAULTS the GPU raise the identical RuntimeError ("Wait timeout: ...",
# hcq.py:296) and were previously swallowed identically -- nothing ever checked whether the device was actually
# faulted before grinding out the remaining rounds. These fake _time_program (the runtime/timing call, not
# _try_compile's compile-time one exercised above) to inject that RuntimeError, and fake dev.iface to simulate
# an actually-faulted device without touching real hardware.
class TestBeamSearchFaultSurfacing(unittest.TestCase):
  def setUp(self): self.real_time_program = search_mod._time_program
  def tearDown(self): search_mod._time_program = self.real_time_program

  def test_wait_timeout_named_beam_device_timeout(self):
    # coordinator fold-in: hcq.py's "Wait timeout: ..." RuntimeError used to collapse onto the same bare
    # "RuntimeError" Counter/WARNING entry as every other cause. NULL has no .iface, so _device_faulted's
    # duck-typed check is a true no-op here -- this exercises the naming alone, not the F1 abort.
    s, rawbufs, var_vals = _build_seed()
    search_mod._time_program = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Wait timeout: 1 ms! (the signal is not set to 5, but 4)"))
    buf = io.StringIO()
    with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0), contextlib.redirect_stdout(buf):
      search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    out = buf.getvalue()
    self.assertIn("BeamDeviceTimeout", out)
    self.assertNotIn("'RuntimeError':", out)

  def test_fault_after_runtime_error_aborts_with_candidate_opts(self):
    s, rawbufs, var_vals = _build_seed()
    expected_first = next(iter(search_mod.get_kernel_actions(s, include_0=False).values()))
    dev = Device["NULL"]
    fake_iface = SimpleNamespace(dev_impl=SimpleNamespace(is_err_state=False), sleep=lambda tm: None)
    state = {"raised": False}
    def fake_time_program(*a, **kw):
      if state["raised"]: raise AssertionError("must abort immediately, not time a second candidate")
      state["raised"] = True
      fake_iface.dev_impl.is_err_state = True  # the candidate just timed out faulted the device
      raise RuntimeError("Wait timeout: 1 ms! (the signal is not set to 5, but 4)")
    search_mod._time_program = fake_time_program
    with patch.object(dev, "iface", fake_iface, create=True):
      with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0):
        with self.assertRaises(RuntimeError) as ctx:
          search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    msg = str(ctx.exception)
    self.assertIn("fault", msg.lower())
    self.assertIn(str(expected_first.applied_opts), msg)  # names the actual failing candidate, not a generic message

  def test_runtime_error_without_fault_continues_search(self):
    # negative control: the same injected RuntimeError on a device that never faults (NULL: no .iface at all,
    # so _device_faulted degrades to a no-op) must NOT abort -- search keeps going and still optimizes, exactly
    # like pre-fix behavior for the non-faulted case.
    s, rawbufs, var_vals = _build_seed()
    real = self.real_time_program
    state = {"n": 0}
    def flaky_time_program(*a, **kw):
      state["n"] += 1
      if state["n"] <= 2: raise RuntimeError("Wait timeout: 1 ms! (the signal is not set to 5, but 4)")
      return real(*a, **kw)
    search_mod._time_program = flaky_time_program
    with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0):
      result = search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    self.assertGreater(state["n"], 2)  # search kept calling _time_program past the injected failures
    self.assertNotEqual(result.applied_opts, [])

# T4.48 F3 (T4.47_RCA.md): beam_search must synchronize the timing device before returning, to bound the
# lifetime of any candidate it abandoned-but-never-cancelled during the search (see TestBeamSearchFaultSurfacing
# above). NULL's Compiled.synchronize() is a real no-op, so these patch the instance method to count/fake calls.
class TestBeamSearchSyncsBeforeReturn(unittest.TestCase):
  def test_synchronize_called_before_return(self):
    s, rawbufs, var_vals = _build_seed()
    calls = []
    with patch.object(Device["NULL"], "synchronize", lambda: calls.append(1)):
      with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0):
        search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    self.assertEqual(calls, [1], "beam_search must call dev.synchronize() exactly once before returning")

  def test_synchronize_fault_gets_clear_search_message(self):
    s, rawbufs, var_vals = _build_seed()
    def raiser(): raise RuntimeError("Device fault detected.")
    with patch.object(Device["NULL"], "synchronize", raiser):
      with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=0):
        with self.assertRaises(RuntimeError) as ctx:
          search_mod.beam_search(s, rawbufs, var_vals, 1, disable_cache=True)
    msg = str(ctx.exception)
    self.assertIn("search", msg.lower())
    self.assertIn("Device fault detected.", msg)  # original error preserved, not swallowed into an opaque one

class TestSecondGroupReduceDeniedOnNV(unittest.TestCase):
  # T4.53: two INDEPENDENT reduce axes (not one axis re-split) both turned into GROUP_REDUCE reproducibly
  # faulted NV silicon (2/2 repros). A kernel with a real ast/2 reduce dims (NULL device -- no hardware),
  # opt-applied through a fake renderer so this needs no live NV/Metal device either.
  def _two_reduce_ast(self):
    with Context(ALLOW_DEVICE_USAGE=1):
      a, b = Tensor.rand(16, 16, device="NULL"), Tensor.rand(16, 16, device="NULL")
      return (a.sum(0) + b.sum(1)).schedule_linear().src[-1].src[0]

  def test_second_group_denied_on_nv(self):
    ast = self._two_reduce_ast()
    s = Scheduler(ast, Renderer(target=Target(device="NV")))
    s.apply_opt(Opt(OptOps.GROUP, axis=0, arg=0))  # 1st GROUP_REDUCE: fine
    with self.assertRaises(KernelOptError):
      s.apply_opt(Opt(OptOps.GROUP, axis=0, arg=0))  # 2nd, independent one: denied

  def test_second_group_still_allowed_off_nv(self):
    ast = self._two_reduce_ast()
    s = Scheduler(ast, Renderer(target=Target(device="METAL")))  # same has_local capability, different device
    s.apply_opt(Opt(OptOps.GROUP, axis=0, arg=0))
    s.apply_opt(Opt(OptOps.GROUP, axis=0, arg=0))  # doesn't raise -- guard is NV-only

if __name__ == "__main__":
  unittest.main()
