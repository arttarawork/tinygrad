import io, contextlib, unittest
from types import SimpleNamespace
from unittest.mock import patch
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

if __name__ == "__main__":
  unittest.main()
