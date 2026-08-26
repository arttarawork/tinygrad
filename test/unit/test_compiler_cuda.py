# T4.31: the OSX docker NV compile path (compiler_cuda.py) spawns one persistent `docker run`
# subprocess in NVRTCCompiler/NVPTXCompiler.__init__. Renderer.__reduce__ (renderer/__init__.py)
# rebuilds that Compiler from scratch on every multiprocessing unpickle -- cheap for every other
# backend, but on OSX each unpickle used to spawn a brand-new container. beam_search sends one
# unpickle per candidate to the shared worker pool (codegen/opt/search.py's pool.imap_unordered),
# so a real search spawned hundreds of abandoned containers within seconds and eventually wedged the
# colima/docker transport. These tests drive the fix's actual logic (_get_server/_compile_with_retry)
# through a duck-typed fake compiler -- no OSX, Docker, or GPU required.
# T4.43: compiler_qcom.py's non-aarch64 path had the identical bug shape, so this cache now lives in
# compiler_server_cache.py and is shared by both -- see test_compiler_qcom.py for the QCOM-side proof.
import unittest
from tinygrad.device import CompileError, CompileTransportError
from tinygrad.runtime.support import compiler_server_cache as cc

class _FakeProc:
  """stands in for a subprocess.Popen: alive until killed."""
  def __init__(self, pid:int):
    self.pid, self._alive = pid, True
  def poll(self): return None if self._alive else 1
  def kill(self): self._alive = False

class FakeCompiler:
  """duck-types what _get_server/_compile_with_retry touch on a real Compiler: .server(), .compiler_process, .compile_server()."""
  spawn_count = 0
  def __init__(self, fail_times:int=0):
    self.fail_times = fail_times  # simulate this many consecutive dead-transport compiles before succeeding
  def server(self, cmd:str, arch:str, *args) -> _FakeProc:
    FakeCompiler.spawn_count += 1
    return _FakeProc(FakeCompiler.spawn_count)
  def compile_server(self, src:str, proc) -> bytes:
    if self.fail_times > 0:
      self.fail_times -= 1
      raise CompileTransportError("simulated dead transport")
    return b"compiled"

class AlwaysBadCompiler(FakeCompiler):
  def compile_server(self, src:str, proc) -> bytes: raise CompileError("Compilation Error")

class TestServerCache(unittest.TestCase):
  def setUp(self):
    cc._server_cache.clear()
    FakeCompiler.spawn_count = 0

  def test_reused_across_constructions(self):
    # the actual T4.31 bug: two separate (freshly-unpickled) Compiler objects of the same class/
    # arch/args must share ONE spawned process, not spawn one each.
    p1 = cc._get_server(FakeCompiler(), "cmd", "sm_86", True)
    p2 = cc._get_server(FakeCompiler(), "cmd", "sm_86", True)
    self.assertIs(p1, p2)
    self.assertEqual(FakeCompiler.spawn_count, 1)

  def test_different_key_gets_different_server(self):
    c = FakeCompiler()
    p_ptx, p_cubin = cc._get_server(c, "cmd", "sm_86", True), cc._get_server(c, "cmd", "sm_86", False)
    p_other_arch = cc._get_server(c, "cmd", "sm_90", True)
    self.assertIsNot(p_ptx, p_cubin)
    self.assertIsNot(p_ptx, p_other_arch)
    self.assertEqual(FakeCompiler.spawn_count, 3)

  def test_dead_server_is_respawned_not_reused(self):
    c = FakeCompiler()
    p1 = cc._get_server(c, "cmd", "sm_86", True)
    p1.kill()
    p2 = cc._get_server(c, "cmd", "sm_86", True)
    self.assertIsNot(p1, p2)
    self.assertEqual(FakeCompiler.spawn_count, 2)

class TestCompileWithRetry(unittest.TestCase):
  def setUp(self):
    cc._server_cache.clear()
    FakeCompiler.spawn_count = 0

  def test_succeeds_first_try_no_respawn(self):
    c = FakeCompiler(fail_times=0)
    c.compiler_process = cc._get_server(c, "cmd", "sm_86", True)
    self.assertEqual(cc._compile_with_retry(c, "src", "cmd", "sm_86", True), b"compiled")
    self.assertEqual(FakeCompiler.spawn_count, 1)

  def test_retries_once_on_transport_error(self):
    c = FakeCompiler(fail_times=1)
    c.compiler_process = cc._get_server(c, "cmd", "sm_86", True)
    self.assertEqual(cc._compile_with_retry(c, "src", "cmd", "sm_86", True), b"compiled")
    self.assertEqual(FakeCompiler.spawn_count, 2)  # exactly one respawn after the simulated dead transport

  def test_does_not_retry_or_respawn_on_genuine_compile_error(self):
    # must never paper over a real compile error by retrying it
    c = AlwaysBadCompiler()
    c.compiler_process = cc._get_server(c, "cmd", "sm_86", True)
    with self.assertRaises(CompileError):
      cc._compile_with_retry(c, "src", "cmd", "sm_86", True)
    self.assertEqual(FakeCompiler.spawn_count, 1)  # no respawn attempted

if __name__ == "__main__":
  unittest.main()
