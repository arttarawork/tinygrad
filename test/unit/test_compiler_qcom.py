# T4.43: compiler_qcom.py's non-aarch64 path (every dev/CI host -- Darwin arm64 reports
# platform.machine() == 'arm64', not 'aarch64', and x86_64 hosts obviously aren't 'aarch64' either)
# had the identical bug T4.31 fixed for compiler_cuda.py's OSX docker path: QCOMCompiler.__init__
# spawned a fresh qemu/docker compile-server subprocess on every construction, and Renderer.__reduce__
# (renderer/__init__.py) reconstructs a fresh QCOMCompiler on every multiprocessing unpickle -- one
# per BEAM candidate handed to a worker (codegen/opt/search.py's pool.imap_unordered). Fixed by
# routing __init__/compile() through the same process-local _server_cache (now shared in
# compiler_server_cache.py, see test_compiler_cuda.py for its direct unit tests) instead of calling
# self.server()/self.compile_server() directly, and by no longer killing the (now shared) process in
# __del__.
#
# These tests drive the real QCOMCompiler class -- construction, real pickle round trips, __del__,
# compile() -- with subprocess.Popen faked out and network/host-arch probes patched. No QCOM
# hardware, network fetch, docker, or qemu required.
import gc, pathlib, pickle, unittest
from unittest.mock import patch
from tinygrad.device import CompileError, CompileTransportError
from tinygrad.runtime.support import compiler_qcom as cq
from tinygrad.runtime.support import compiler_server_cache as csc

class _FakeProc:
  """stands in for a subprocess.Popen: alive until killed."""
  def __init__(self, *a, **kw): self._alive = True
  def poll(self): return None if self._alive else 1
  def kill(self): self._alive = False

class TestQCOMCompilerServerCache(unittest.TestCase):
  def setUp(self):
    csc._server_cache.clear()
    self.spawn_count = 0
    def fake_popen(*a, **kw):
      self.spawn_count += 1
      return _FakeProc()
    self.enterContext(patch("subprocess.Popen", side_effect=fake_popen))
    self.enterContext(patch.object(cq, "fetch", return_value=pathlib.Path("/fake/rootfs")))
    self.enterContext(patch.object(cq.shutil, "which", return_value=None))  # force the docker (not qemu) branch, deterministically
    self.enterContext(patch.object(cq.platform, "machine", return_value="x86_64"))

  def test_unpickle_reuses_cached_server(self):
    # the actual T4.43 bug: N unpickles of the same (class, arch) compiler must share ONE spawned
    # server, not spawn one per unpickle -- exactly what Renderer.__reduce__ triggers once per BEAM
    # candidate in beam_search's worker pool.
    c = cq.QCOMCompiler("a630")
    self.assertEqual(self.spawn_count, 1)
    for _ in range(5):
      c = pickle.loads(pickle.dumps(c))
    self.assertEqual(self.spawn_count, 1)

  def test_del_does_not_kill_cached_server(self):
    # __del__ used to unconditionally kill self.compiler_process -- fatal once that process is
    # shared: a GC'd candidate's __del__ would kill the cache out from under the next reuse, which
    # would silently degrade the fix above back to one-spawn-per-candidate.
    c = cq.QCOMCompiler("a630")
    proc = c.compiler_process
    del c
    gc.collect()
    self.assertIsNone(proc.poll())  # still alive after the instance was collected

  def test_compile_retries_once_on_transport_error(self):
    calls = []
    def fake_compile_server(src, proc):
      calls.append(proc)
      if len(calls) == 1: raise CompileTransportError("simulated dead transport")
      return b"compiled"
    c = cq.QCOMCompiler("a630")
    with patch.object(cq.Compiler, "compile_server", side_effect=fake_compile_server):
      self.assertEqual(c.compile("src"), b"compiled")
    self.assertEqual(self.spawn_count, 2)  # exactly one respawn after the simulated dead transport

  def test_compile_does_not_retry_genuine_compile_error(self):
    # must never paper over a real compile error by retrying it
    c = cq.QCOMCompiler("a630")
    with patch.object(cq.Compiler, "compile_server", side_effect=CompileError("bad source")):
      with self.assertRaises(CompileError):
        c.compile("src")
    self.assertEqual(self.spawn_count, 1)  # no respawn attempted

if __name__ == "__main__":
  unittest.main()
