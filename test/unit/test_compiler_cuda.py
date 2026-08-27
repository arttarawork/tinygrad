# T4.31: the OSX docker NV compile path (compiler_cuda.py) spawns one persistent `docker run`
# subprocess in NVRTCCompiler/NVPTXCompiler.__init__. Renderer.__reduce__ (renderer/__init__.py)
# rebuilds that Compiler from scratch on every multiprocessing unpickle -- cheap for every other
# backend, but on OSX each unpickle used to spawn a brand-new container. beam_search sends one
# unpickle per candidate to the shared worker pool (codegen/opt/search.py's pool.imap_unordered),
# so a real search spawned hundreds of abandoned containers within seconds and eventually wedged the
# colima/docker transport. These tests drive the fix's actual logic (_get_server/_compile_with_retry)
# through a duck-typed fake compiler -- no OSX, Docker, or GPU required.
import unittest, struct
from tinygrad.device import CompileError, CompileTransportError, Compiler
from tinygrad.runtime.support import compiler_cuda as cc

class _FakeProc:
  """stands in for a subprocess.Popen: alive until killed."""
  def __init__(self, pid:int):
    self.pid, self._alive, self.killed, self.waited = pid, True, False, False
  def poll(self): return None if self._alive else 1
  def kill(self): self._alive, self.killed = False, True
  def wait(self): self.waited = True

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

  def test_evicted_process_is_reaped_on_retry(self):
    # T4.45: _compile_with_retry used to _server_cache.pop() the dead entry and drop it on the floor --
    # never kill()/wait()-ed, so a process that's still alive at the OS level (e.g. the docker CLI
    # client up but the container inside it dead) leaked until atexit's _reap_servers ran.
    c = FakeCompiler(fail_times=1)
    c.compiler_process = cc._get_server(c, "cmd", "sm_86", True)
    dead_proc = c.compiler_process
    self.assertEqual(cc._compile_with_retry(c, "src", "cmd", "sm_86", True), b"compiled")
    self.assertTrue(dead_proc.killed)
    self.assertTrue(dead_proc.waited)

  def test_already_dead_evicted_process_is_still_reaped_not_double_killed(self):
    # the evicted process may have already exited on its own (poll() != None) -- must still wait() it
    # (avoid a zombie) but must NOT call kill() again on an already-reclaimed pid.
    c = FakeCompiler(fail_times=1)
    c.compiler_process = cc._get_server(c, "cmd", "sm_86", True)
    dead_proc = c.compiler_process
    dead_proc._alive = False  # simulate the process having already exited on its own
    self.assertEqual(cc._compile_with_retry(c, "src", "cmd", "sm_86", True), b"compiled")
    self.assertFalse(dead_proc.killed)
    self.assertTrue(dead_proc.waited)

  def test_does_not_retry_or_respawn_on_genuine_compile_error(self):
    # must never paper over a real compile error by retrying it
    c = AlwaysBadCompiler()
    c.compiler_process = cc._get_server(c, "cmd", "sm_86", True)
    with self.assertRaises(CompileError):
      cc._compile_with_retry(c, "src", "cmd", "sm_86", True)
    self.assertEqual(FakeCompiler.spawn_count, 1)  # no respawn attempted

class _ShortWriteProc:
  """simulates a raw unbuffered pipe (bufsize=0) whose write() legitimately accepts at most `chunk`
  bytes per call -- captures everything actually delivered so a test can tell "looped until the whole
  message got through" apart from "returned after a single short write". Its read() hands back a fixed
  canned reply regardless of what/how much was written, so the test never blocks on a real deadlock
  (T4.42 NOTES.md Sec3b: the real deadlock is the server never getting the rest of the message and the
  client then blocking on the read that follows -- this fake's read is not gated on the write at all,
  by design, so the test stays a plain, fast unit test of the write loop)."""
  def __init__(self, reply:bytes, chunk:int, pid:int=9001):
    self.stdin = self.stdout = self
    self.reply, self.chunk, self.pid = reply, chunk, pid
    self.written, self.write_calls = b"", 0
    self._reply_pos = 0
  def write(self, data:bytes) -> int:
    self.write_calls += 1
    n = min(self.chunk, len(data))
    self.written += data[:n]
    return n
  def read(self, n:int) -> bytes:
    end = min(self._reply_pos + n, len(self.reply))
    ret, self._reply_pos = self.reply[self._reply_pos:end], end
    return ret

class TestCompileServerWrite(unittest.TestCase):
  def test_short_write_delivers_every_byte(self):
    # T4.45: compile_server()'s outbound write ignored write()'s return value -- a legitimate short
    # write (bufsize=0 pipes can return short per Python's own subprocess docs) silently sent only a
    # prefix of the message. Proven failing pre-fix: with the single unlooped `.write(data)` call, only
    # the first `chunk` bytes ever reach the fake, so `proc.written` falls short of `expected_wire`.
    src = "x" * 5000  # large enough that chunk=7 needs hundreds of write() calls to deliver it all
    payload = b"ok"
    proc = _ShortWriteProc(struct.pack("I", len(payload)) + payload, chunk=7)
    self.assertEqual(Compiler().compile_server(src, proc), payload)
    expected_wire = struct.pack("I", len(src.encode())) + src.encode()
    self.assertEqual(proc.written, expected_wire)

  def test_full_single_write_unchanged(self):
    # negative control: a healthy write that fully lands in one call -- must still take exactly one
    # write() call (the loop's first iteration exhausts `data` and returns), same as before the fix.
    src = "small"
    payload = b"ok"
    proc = _ShortWriteProc(struct.pack("I", len(payload)) + payload, chunk=10_000)
    self.assertEqual(Compiler().compile_server(src, proc), payload)
    expected_wire = struct.pack("I", len(src.encode())) + src.encode()
    self.assertEqual(proc.written, expected_wire)
    self.assertEqual(proc.write_calls, 1)

if __name__ == "__main__":
  unittest.main()
