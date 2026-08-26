import io, contextlib, unittest
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

def _key_for(s, amt=1, allow_test_size=True): return {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size,
                                                        "device": s.ren.target.device, "suffix": s.ren.suffix}

class _FakeDiskCache:
  # in-memory stand-in for diskcache_get/_put, monkeypatched into search_mod so these tests never touch the
  # real (multi-GB, hardware-measured) beam_search disk cache
  def __init__(self): self.store: dict = {}
  def _k(self, table, key): return (table, tuple(sorted(key.items())) if isinstance(key, dict) else key)
  def get(self, table, key): return self.store.get(self._k(table, key))
  def put(self, table, key, val, prepickled=False):
    self.store[self._k(table, key)] = val
    return val

# T4.39: beam_search used to unconditionally diskcache_put beam[0][0].applied_opts, even when beam[0]
# was still the untouched inf-scored seed (the T4.27 fallback path above) -- so a later call would hit
# the cache and silently replay an empirically-unvalidated kernel forever, with none of T4.27's WARNINGs
# (they only fire during the search that produced the bad result). Fix: skip the cache write whenever the
# final chosen score is inf (covers both the fallback-to-seed case and a candidate whose only timing
# attempt hit the _time_program AssertionError path -- neither was ever actually measured as fast).
class TestBeamSearchDoesNotCacheUnvalidatedResults(unittest.TestCase):
  def setUp(self):
    self.real_try_compile = search_mod._try_compile
    self.real_get, self.real_put = search_mod.diskcache_get, search_mod.diskcache_put
    self.cache = _FakeDiskCache()
    search_mod.diskcache_get, search_mod.diskcache_put = self.cache.get, self.cache.put

  def tearDown(self):
    search_mod._try_compile = self.real_try_compile
    search_mod.diskcache_get, search_mod.diskcache_put = self.real_get, self.real_put

  def test_total_failure_is_not_cached_and_replay_call_resarches(self):
    s, rawbufs, var_vals = _build_seed()
    key = _key_for(s)
    calls = 0
    def fail_all(x):
      nonlocal calls
      calls += 1
      return x[0], None, "BrokenPipeError"
    search_mod._try_compile = fail_all
    with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=1):
      buf1 = io.StringIO()
      with contextlib.redirect_stdout(buf1):
        result1 = search_mod.beam_search(s, rawbufs, var_vals, 1)
      self.assertIn("WARNING", buf1.getvalue())
      self.assertGreater(calls, 0)
      # the core T4.39 assertion: a total-failure fallback result must NOT reach the disk cache
      self.assertIsNone(self.cache.get("beam_search", key))
      self.assertEqual(result1.applied_opts, hand_coded_optimizations(s.copy()).applied_opts)

      # a second call for the identical ast/key must re-search (and re-warn), not silently replay
      calls_before = calls
      buf2 = io.StringIO()
      with contextlib.redirect_stdout(buf2):
        result2 = search_mod.beam_search(s, rawbufs, var_vals, 1)
      self.assertGreater(calls, calls_before)  # _try_compile really ran again -> proves it wasn't a cache hit
      self.assertIn("WARNING", buf2.getvalue())
      self.assertEqual(result2.applied_opts, hand_coded_optimizations(s.copy()).applied_opts)
      self.assertIsNone(self.cache.get("beam_search", key))  # still nothing persisted after the replay attempt

  def test_successful_search_is_still_cached_and_replayed(self):
    # positive control: a real, validly-timed winner must still be cached (T4.39 doesn't touch this path)
    s, rawbufs, var_vals = _build_seed()
    key = _key_for(s)
    with Context(ALLOW_DEVICE_USAGE=1, PARALLEL=0, CACHELEVEL=1):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        result1 = search_mod.beam_search(s, rawbufs, var_vals, 1)
      self.assertNotIn("WARNING", buf.getvalue())
      cached = self.cache.get("beam_search", key)
      self.assertIsNotNone(cached)
      self.assertEqual(cached, result1.applied_opts)

      # second call must replay from the cache, not re-search: prove it by making a re-search blow up
      def boom(x): raise AssertionError("beam_search should not re-search on a cache hit")
      search_mod._try_compile = boom
      result2 = search_mod.beam_search(s, rawbufs, var_vals, 1)
      self.assertEqual(result2.applied_opts, result1.applied_opts)

if __name__ == "__main__":
  unittest.main()
