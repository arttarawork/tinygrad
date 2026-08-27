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
from tinygrad.helpers import Context
import tinygrad.device as device_mod

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

# T4.42: does a flaky compile-server round trip ever let a bad/truncated/empty blob -- or a poisoned
# success -- reach the *persistent* compile cache (Compiler.compile_cached's diskcache_put, device.py:320)?
# T4.14 closed silent read truncation (_read_exactly loops-or-raises, never returns a partial buffer) and
# T4.31 closed the bare-exception/no-retry-classification gap (CompileTransportError, retry-once with a
# fresh process). These tests drive the REAL Compiler.compile_cached()/compile_server() (device.py) and the
# REAL _compile_with_retry()/_get_server() (compiler_cuda.py) against fake transports that fail every way
# the wire protocol can fail, with an in-memory fake diskcache standing in for the real one -- never touches
# ~/Library/Caches/tinygrad/cache.db (same pattern as test/opt/test_beam_search.py's T4.39 tests).

class _FakeCompileProc:
  # same double as test/null/test_device.py's TestCompileServer (T4.14/T4.31) -- duplicated here since
  # test/unit and test/null are independent modules and this task only runs test/unit + test/opt.
  # simulates a raw unbuffered pipe: read(n) hands back at most `chunk` bytes regardless of n.
  def __init__(self, reply:bytes, chunk:int, pid:int=4242):
    self.stdin = self.stdout = self
    self.reply, self.chunk, self.pos, self.pid = reply, chunk, 0, pid
  def write(self, data): pass
  def read(self, n:int) -> bytes:
    end = min(self.pos + min(self.chunk, n), len(self.reply))
    ret, self.pos = self.reply[self.pos:end], end
    return ret

class _DeadPipeProc:
  # simulates a compile-server process whose stdin pipe is already closed (the server died)
  def __init__(self, pid:int=4242):
    self.stdin = self.stdout = self
    self.pid = pid
  def write(self, data): raise BrokenPipeError(32, "Broken pipe")
  def read(self, n:int) -> bytes: raise AssertionError("should never read after a failed write")

class _FakeDiskCache:
  # in-memory stand-in for diskcache_get/_put, monkeypatched into tinygrad.device -- these tests must
  # never touch the real (multi-GB, hardware-populated) persistent compile cache.
  def __init__(self): self.store: dict = {}
  def get(self, table, key): return self.store.get((table, key))
  def put(self, table, key, val, prepickled=False):
    self.store[(table, key)] = val
    return val

class _SingleProcCompiler(Compiler):
  """real Compiler.compile_cached()/compile_server(), wired to one fake transport, no retry -- exercises
  device.py's compile_cached/compile_server directly (not compiler_cuda.py's retry wrapper)."""
  def __init__(self, proc, cachekey:str):
    self.proc = proc
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes: return self.compile_server(src, self.proc)

class _RetryCompiler(Compiler):
  """real Compiler.compile_cached(), wired through the real _compile_with_retry/_get_server
  (compiler_cuda.py), against a queue of fake transports -- exercises T4.31's retry-once path."""
  def __init__(self, procs:list, cachekey:str):
    self._procs = list(procs)
    self.compiler_process = self._procs.pop(0)
    super().__init__(cachekey)
  def server(self, cmd, arch, *args): return self._procs.pop(0)
  def compile(self, src:str) -> bytes: return cc._compile_with_retry(self, src, "fake-cmd", "fake-arch")

class _FakeDiskCacheTestCase(unittest.TestCase):
  # shared setUp/tearDown: monkeypatch tinygrad.device's diskcache_get/_put to an in-memory fake, and
  # force CCACHE on (so Compiler.__init__ actually sets a cachekey -- otherwise compile_cached's cache
  # branch is a no-op regardless of what device.py does, and every assertion below would pass vacuously).
  def setUp(self):
    cc._server_cache.clear()
    self.cache = _FakeDiskCache()
    self.real_get, self.real_put = device_mod.diskcache_get, device_mod.diskcache_put
    device_mod.diskcache_get, device_mod.diskcache_put = self.cache.get, self.cache.put
    self.ctx = Context(CCACHE=1)
    self.ctx.__enter__()

  def tearDown(self):
    self.ctx.__exit__(None, None, None)
    device_mod.diskcache_get, device_mod.diskcache_put = self.real_get, self.real_put
    cc._server_cache.clear()

class TestCompileCachedTransportFlakes(_FakeDiskCacheTestCase):
  def test_eof_mid_body_not_cached(self):
    payload = b"x" * 100
    proc = _FakeCompileProc(struct.pack("I", len(payload) + 50) + payload, chunk=3)  # promises 50 bytes it never sends
    c = _SingleProcCompiler(proc, cachekey="t442_eof")
    with self.assertRaises(CompileTransportError):
      c.compile_cached("src_eof")
    self.assertEqual(self.cache.store, {})  # the core T4.42 assertion: nothing reached the disk cache

  def test_broken_pipe_not_cached(self):
    proc = _DeadPipeProc()
    c = _SingleProcCompiler(proc, cachekey="t442_brokenpipe")
    with self.assertRaises(CompileTransportError):
      c.compile_cached("src_brokenpipe")
    self.assertEqual(self.cache.store, {})

  def test_genuine_compile_error_not_cached(self):
    # a 0-byte reply is compileserver.py's own signal for "compilation failed" (its except-branch sets
    # lib = b""), not a transport death -- must stay a plain CompileError and still never get cached.
    proc = _FakeCompileProc(struct.pack("I", 0), chunk=3)
    c = _SingleProcCompiler(proc, cachekey="t442_compileerr")
    with self.assertRaises(CompileError) as ctx:
      c.compile_cached("src_bad")
    self.assertNotIsInstance(ctx.exception, CompileTransportError)
    self.assertEqual(self.cache.store, {})

  def test_successful_compile_is_cached_and_replayed(self):
    # positive control: proves the harness really would observe a cache write if compile_cached ever made
    # one -- the empty stores above aren't just an artifact of this test class never calling diskcache_put.
    payload = b"a real compiled lib"
    proc = _FakeCompileProc(struct.pack("I", len(payload)) + payload, chunk=4)
    c = _SingleProcCompiler(proc, cachekey="t442_ok")
    self.assertEqual(c.compile_cached("src_ok"), payload)
    self.assertEqual(self.cache.store[("t442_ok", "src_ok")], payload)
    # a second call for the same src must replay the cache -- prove it by making a re-read blow up
    proc.read = lambda n: (_ for _ in ()).throw(AssertionError("should not touch the transport on a cache hit"))
    self.assertEqual(c.compile_cached("src_ok"), payload)

class TestCompileCachedRetryPath(_FakeDiskCacheTestCase):
  def test_retry_success_is_cached_with_correct_payload(self):
    # T4.31's retry-once-with-a-fresh-process path: the first (dead) attempt must not contaminate what
    # ultimately gets cached -- only the second (healthy, independent) round trip's exact payload may land.
    payload = b"good after respawn"
    dead, healthy = _DeadPipeProc(pid=1), _FakeCompileProc(struct.pack("I", len(payload)) + payload, chunk=5, pid=2)
    c = _RetryCompiler([dead, healthy], cachekey="t442_retry_ok")
    self.assertEqual(c.compile_cached("src_retry_ok"), payload)
    self.assertEqual(self.cache.store[("t442_retry_ok", "src_retry_ok")], payload)

  def test_retry_exhausted_not_cached(self):
    # both attempts dead: _compile_with_retry gives up after one retry (T4.31, never loops/swallows) --
    # the exception must still propagate all the way through compile_cached with nothing cached.
    c = _RetryCompiler([_DeadPipeProc(pid=1), _DeadPipeProc(pid=2)], cachekey="t442_retry_fail")
    with self.assertRaises(CompileTransportError):
      c.compile_cached("src_retry_fail")
    self.assertEqual(self.cache.store, {})
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
