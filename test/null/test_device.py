#!/usr/bin/env python
import unittest, os, subprocess, struct, socket, threading, time
from unittest.mock import patch
from tinygrad import Tensor
from tinygrad.device import Device, Compiler, CompileError, CompileTransportError, enumerate_devices_str
from tinygrad.helpers import diskcache_get, diskcache_put, getenv, Context, Target, WIN, OSX, DEV
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.system import RemotePCIDevice

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    self.assertEqual(Device.canonicalize(None), Device.DEFAULT)
    self.assertEqual(Device.canonicalize("CPU"), "CPU")
    self.assertEqual(Device.canonicalize("cpu"), "CPU")
    self.assertEqual(Device.canonicalize("CL"), "CL")
    self.assertEqual(Device.canonicalize("CL:0"), "CL")
    self.assertEqual(Device.canonicalize("cl:0"), "CL")
    self.assertEqual(Device.canonicalize("CL:1"), "CL:1")
    self.assertEqual(Device.canonicalize("cl:1"), "CL:1")
    self.assertEqual(Device.canonicalize("CL:2"), "CL:2")
    self.assertEqual(Device.canonicalize("disk:/dev/shm/test"), "DISK:/dev/shm/test")
    self.assertEqual(Device.canonicalize("disk:000.txt"), "DISK:000.txt")

  def test_getitem_not_exist(self):
    with self.assertRaises(ModuleNotFoundError):
      Device["TYPO"]

  @unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
  def test_nonexistent_renderer(self):
    with self.assertRaisesRegex(RuntimeError, "has no renderer"):
      with Context(DEV="CPU:TYPO"): Device[Device.DEFAULT].renderer
    with self.assertRaisesRegex(RuntimeError, "did you mean: 'CLANG'"):
      with Context(DEV="CPU:CLANGJIT"): Device[Device.DEFAULT].renderer

  @unittest.skipIf(Device.DEFAULT != "AMD", "only run on AMD")
  def test_nonexistent_iface(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device[Device.DEFAULT].iface'],
                            env={**os.environ, "DEV":"USA+AMD"}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"did you mean: 'USB'", result.stderr)

  @unittest.skipIf(Device.DEFAULT != "AMD", "only run on AMD")
  def test_dev_id_out_of_range(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device[Device.DEFAULT]'],
                            env={**os.environ, "DEV":":99+AMD"}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"invalid visibility filter", result.stderr)

  def test_lowercase_canonicalizes(self):
    device = Device.DEFAULT
    with Context(DEV=device.lower()):
      self.assertEqual(Device.canonicalize(None), device)

  def test_set_device_default_raises(self):
    with self.assertRaisesRegex(AttributeError, "setting Device.DEFAULT is deprecated"):
      Device.DEFAULT = "CPU"

  def test_old_device_env_raises(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device.DEFAULT'],
                            env={**os.environ, "CPU": "1", "DEV": ""}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"deprecated", result.stderr)

  def test_old_renderer_env_raises(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device[Device.DEFAULT].renderer'],
                            env={**os.environ, "DEV": "CPU", "CPU_LLVM": "1"}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"deprecated", result.stderr)

  @unittest.skipIf(WIN, "skipping windows test") # TODO: subprocess causes memory violation?
  def test_env_overwrite_default_compiler(self):
    if Device.DEFAULT == "CPU":
      from tinygrad.runtime.support.compiler_cpu import ClangCompiler
      from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
      try: _, _ = CPULLVMCompiler(), ClangCompiler()
      except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

      imports = ("from tinygrad import Device; from tinygrad.runtime.support.compiler_cpu import ClangCompiler; "
                 "from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler")
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, CPULLVMCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU:LLVM"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, ClangCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, ClangCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU:CLANG"})
    elif Device.DEFAULT == "AMD":
      from tinygrad.runtime.support.compiler_amd import HIPCompiler
      from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
      try: _, _ = HIPCompiler(Device[Device.DEFAULT].arch), AMDLLVMCompiler(Device[Device.DEFAULT].arch)
      except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

      imports = ("from tinygrad import Device; from tinygrad.runtime.support.compiler_amd import HIPCompiler; "
                 "from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler")
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, AMDLLVMCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD:LLVM"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD:HIP"})
    else: self.skipTest("only run on CPU/AMD")

  @unittest.skipIf(WIN, "skipping windows test")
  def test_env_online(self):
    from tinygrad.runtime.support.compiler_cpu import ClangCompiler
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
    try: _, _ = CPULLVMCompiler(), ClangCompiler()
    except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

    with Context(DEV="CPU:LLVM"):
      inst = Device["CPU"].compiler
      self.assertIsInstance(Device["CPU"].compiler, CPULLVMCompiler)
    with Context(DEV="CPU"):
      self.assertIsInstance(Device["CPU"].compiler, ClangCompiler)
    with Context(DEV="CPU:LLVM"):
      self.assertIsInstance(Device["CPU"].compiler, CPULLVMCompiler)
      assert inst is Device["CPU"].compiler  # cached

  @unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
  def test_compiler_autodetect_fallback(self):
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler

    try: CPULLVMCompiler()
    except Exception as e: self.skipTest(f"skipping: LLVM not available: {e}")

    dev = Device["CPU"]
    dev.cached_renderer.clear()
    with patch("tinygrad.renderer.cstyle.ClangRenderer.__init__", side_effect=RuntimeError("broken")):
      self.assertIsInstance(dev.renderer.compiler, CPULLVMCompiler)

  def test_dev_contextvar(self):
    orig_dev = Device.DEFAULT
    with Context(DEV="CPU"): self.assertEqual(Tensor.empty(1).device, "CPU")
    with Context(DEV="NULL"): self.assertEqual(Tensor.empty(1).device, "NULL")
    self.assertEqual(Tensor.empty(1).device, orig_dev)

class TestDevVar(unittest.TestCase):
  def test_parse(self):
    for d, t in [("AMD", Target(device="AMD", renderer="")), ("AMD:LLVM", Target(device="AMD", renderer="LLVM")),
                 (":LLVM", Target(device="", renderer="LLVM")), ("AMD::gfx1100", Target(device="AMD", arch="gfx1100")),
                 ("AMD:LLVM:gfx1100", Target(device="AMD", renderer="LLVM", arch="gfx1100")), ("::gfx1100", Target(arch="gfx1100")),
                 ("USB+", Target(interface="USB")), ("USB+AMD", Target(device="AMD", interface="USB")),
                 ("PCI:0+AMD", Target(device="AMD", interface="PCI", indices="0")), (":0+AMD", Target(device="AMD", indices="0")),
                 ("PCI:0,1+AMD", Target(device="AMD", interface="PCI", indices="0,1")),
                 ("QCOM;USB+AMD", [Target(device="QCOM"), Target(device="AMD", interface="USB")])]:
      with Context(DEV=d):
        self.assertEqual(DEV.value, t if isinstance(t, list) else [t])
        self.assertEqual(str(DEV), d)

  def test_target(self):
    with Context(DEV="CPU"): self.assertEqual(DEV.target("CPU"), Target("CPU"))
    with Context(DEV="CPU:LLVM"): self.assertEqual(DEV.target("CPU"), Target("CPU", "LLVM"))
    with Context(DEV=":LLVM"): self.assertEqual(DEV.target("CPU"), Target("CPU", "LLVM"))
    with Context(DEV="AMD:LLVM"): self.assertEqual(DEV.target("CPU"), Target("CPU"))
    with Context(DEV=""): self.assertEqual(DEV.target("CPU"), Target("CPU"))
    with Context(DEV="QCOM:IR3;AMD:LLVM"):
      self.assertEqual(DEV.target("QCOM"), Target("QCOM", "IR3"))
      self.assertEqual(DEV.target("AMD"), Target("AMD", "LLVM"))
      self.assertEqual(DEV.target("CPU"), Target("CPU"))

  def test_dev_arch_override(self):
    with Context(DEV="NULL::gfx1100"):
      self.assertEqual(Device["NULL"].renderer.target.arch, "gfx1100")

class MockCompiler(Compiler):
  def __init__(self, key): super().__init__(key)
  def compile(self, src) -> bytes: return src.encode()

class TestCompiler(unittest.TestCase):
  def test_compile_cached(self):
    diskcache_put("key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(CCACHE=1):
      self.assertEqual(MockCompiler("key").compile_cached("123"), str.encode("123"))
      self.assertEqual(diskcache_get("key", "123"), str.encode("123"))

  def test_compile_cached_disabled(self):
    diskcache_put("disabled_key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(CCACHE=0):
      self.assertEqual(MockCompiler("disabled_key").compile_cached("123"), str.encode("123"))
      self.assertIsNone(diskcache_get("disabled_key", "123"))

  def test_device_compile(self):
    getenv.cache_clear()
    with Context(CCACHE=0):
      a = Tensor([0.,1.], device=Device.DEFAULT).realize()
      (a + 1).realize()

class _FakeCompileProc:
  # simulates a raw unbuffered pipe: read(n) hands back at most `chunk` bytes regardless of n
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

class TestCompileServer(unittest.TestCase):
  def test_fragmented_reassembly(self):
    payload = bytes((i * 7) % 256 for i in range(5000))
    proc = _FakeCompileProc(struct.pack("I", len(payload)) + payload, chunk=2)
    self.assertEqual(Compiler().compile_server("src", proc), payload)

  def test_eof_mid_body_raises(self):
    # T4.31: a dead/short-lived server (EOF mid-reply) is a transport failure, not a compile failure --
    # must raise the more specific CompileTransportError (still an instance of CompileError) so a
    # caller can tell "respawn and retry" apart from "this source genuinely failed to compile".
    payload = b"x" * 100
    proc = _FakeCompileProc(struct.pack("I", len(payload) + 50) + payload, chunk=3)  # promises 50 bytes it never sends
    self.assertRaisesRegex(CompileTransportError, "got 100 of 150 bytes", Compiler().compile_server, "src", proc)

  def test_broken_pipe_raises_transport_error(self):
    # T4.31: compile_server() used to let a bare BrokenPipeError from a dead compile-server process
    # propagate uncaught, killing the whole run with no indication of which transport died.
    proc = _DeadPipeProc(pid=1234)
    self.assertRaisesRegex(CompileTransportError, "pid=1234", Compiler().compile_server, "src", proc)

  def test_genuine_compile_error_is_not_a_transport_error(self):
    # T4.31: a real compile failure always completes the wire protocol (a 0-byte reply) -- must stay
    # plain CompileError, never CompileTransportError, so callers never retry a genuinely bad source.
    proc = _FakeCompileProc(struct.pack("I", 0), chunk=3)
    with self.assertRaises(CompileError) as ctx:
      Compiler().compile_server("src", proc)
    self.assertNotIsInstance(ctx.exception, CompileTransportError)

@unittest.skipIf(WIN, "AF_UNIX fd-passing (SCM_RIGHTS) isn't supported on Windows")
class TestRemotePCIDeviceRPC(unittest.TestCase):
  # RemotePCIDevice._rpc's has_fd branch rides an SCM_RIGHTS fd on a 17-byte reply that can arrive
  # fragmented under load; drive it over a real AF_UNIX socketpair so recvmsg()/cmsg semantics are real.
  @staticmethod
  def _serve(server:socket.socket, sizes:list[int], fd:int, fd_frag:int):
    header, pos = struct.pack('<BQQ', 0, 0, 0), 0
    server.recv(33)  # drain the fixed 33-byte request header (no payload in these tests)
    for i, sz in enumerate(sizes):
      anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack('i', fd))] if i == fd_frag else []
      server.sendmsg([header[pos:pos + sz]], anc)
      pos += sz
      time.sleep(0.05)  # force the client to see this as a genuinely separate recvmsg() delivery

  def _run(self, sizes:list[int], fd_frag:int):
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    r_fd, w_fd = os.pipe()
    t = threading.Thread(target=self._serve, args=(server, sizes, r_fd, fd_frag), daemon=True)
    t.start()
    try:
      _, _, _, fd = RemotePCIDevice._rpc(client, 0, 0, has_fd=True)
      os.write(w_fd, b"hello")
      self.assertEqual(os.read(fd, 5), b"hello")  # proves the received fd is the same pipe, not garbage
      os.close(fd)
    finally:
      t.join(timeout=2)
      client.close()
      server.close()
      os.close(r_fd)
      os.close(w_fd)

  def test_fd_on_first_fragment(self): self._run(sizes=[5, 12], fd_frag=0)
  def test_fd_on_delayed_fragment(self): self._run(sizes=[10, 7], fd_frag=1)

  def test_eof_mid_message_raises(self):
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    def serve():
      server.recv(33)
      server.sendmsg([struct.pack('<BQQ', 0, 0, 0)[:8]])  # 8 of 17 bytes, then hang up -- no fd ever sent
      server.close()
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
      self.assertRaisesRegex(RuntimeError, "got 8 of 17 bytes", RemotePCIDevice._rpc, client, 0, 0, has_fd=True)
    finally:
      t.join(timeout=2)
      client.close()

  def test_failed_rpc_with_no_fd_raises_rpc_failed(self):
    # a failing has_fd RPC (status != 0) legitimately carries no fd -- must surface as the normal
    # "RPC failed" error, not get misread as a dropped fd (this is what real olmoe-scale NV allocation hits)
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    def serve():
      server.recv(33)
      server.sendmsg([struct.pack('<BQQ', 1, 0, 0)])  # status=1, zero-length error body, no cmsg
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
      self.assertRaisesRegex(RuntimeError, "RPC failed", RemotePCIDevice._rpc, client, 0, 0, has_fd=True)
    finally:
      t.join(timeout=2)
      client.close()
      server.close()

@unittest.skip("this test is broken if you have tinymesa installed")
@unittest.skipIf(OSX and 'libclang' in DLL._loaded_, "MTLCompiler can't be loaded after libclang on OSX")
class TestRunAsModule(unittest.TestCase):
  def test_module_runs(self):
    cpu_line = [l for l in enumerate_devices_str() if "CPU" in l][0]
    self.assertIn("PASS", cpu_line, f"expected CPU to PASS, got: {cpu_line}")

if __name__ == "__main__":
  unittest.main()
