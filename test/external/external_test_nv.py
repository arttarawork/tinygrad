import os, shutil, struct, subprocess, tempfile, threading, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.helpers import getenv
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry
from tinygrad.runtime.support.am.amdev import AMPageTableEntry
from tinygrad.runtime.support.system import APLRemotePCIDevice, RemotePCIDevice
from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, PCIIface, _fault_recovery_hint

class CountingMMIOInterface:
  """Fake MMIOInterface: one __getitem__/__setitem__ call == one socket-RPC message in the real RemoteMMIOInterface (system.py), so
  counting Python-level calls here is a faithful proxy for counting messages over the wire. `is_remote` feeds the same flag the real
  RemoteMMIOInterface carries, which memory.py/nvdev.py use to decide whether to skip the blocking validation readback."""
  def __init__(self, buf:bytearray, addr:int, nbytes:int, fmt='B', stats:dict|None=None, is_remote=False):
    self.buf, self.addr, self.nbytes, self.fmt, self.is_remote = buf, addr, nbytes, fmt, is_remote
    self.el_sz = struct.calcsize(fmt)
    self.stats = stats if stats is not None else {'w': 0, 'r': 0}
  def __len__(self): return self.nbytes // self.el_sz
  def __getitem__(self, k):
    self.stats['r'] += 1
    idxs = range(*k.indices(len(self))) if isinstance(k, slice) else [k]
    vals = [struct.unpack_from('<'+self.fmt, self.buf, self.addr + i*self.el_sz)[0] for i in idxs]
    return vals if isinstance(k, slice) else vals[0]
  def __setitem__(self, k, v):
    self.stats['w'] += 1
    idxs = range(*k.indices(len(self))) if isinstance(k, slice) else [k]
    vals = v if isinstance(k, slice) else [v]
    for i, val in zip(idxs, vals): struct.pack_into('<'+self.fmt, self.buf, self.addr + i*self.el_sz, val)
  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return CountingMMIOInterface(self.buf, self.addr+offset, (self.nbytes-offset) if size is None else size, fmt or self.fmt,
                                  stats=self.stats, is_remote=self.is_remote)

class FakeNVMemoryManager(NVMemoryManager):
  def on_range_mapped(self): pass # real impl pokes an MMU-invalidate register; irrelevant to counting PTE reads/writes

class FakeNVDev:
  """Minimal NVDev stand-in for exercising real page-table code with no hardware: NVDev.include() just loads static bitfield
  definitions (pure Python, no I/O), and vram is a counting fake instead of a real BAR1 mapping."""
  def __init__(self, is_remote=False):
    self.smi_dev, self.is_booting, self.mmu_ver = False, True, 3
    NVDev.include(self, "dev_mmu", "gh100")
    self.pte_t, self.pde_t, self.dual_pde_t = self.NV_MMU_VER3_PTE, self.NV_MMU_VER3_PDE, self.NV_MMU_VER3_DUAL_PDE
    self.stats = {'w': 0, 'r': 0}
    vram_size = 64 << 20
    self.vram = CountingMMIOInterface(bytearray(vram_size), 0, vram_size, stats=self.stats, is_remote=is_remote)
    self.mm = FakeNVMemoryManager(self, vram_size, boot_size=(1 << 20), pt_t=NVPageTableEntry, va_bits=56, va_shifts=[12, 21, 29, 38, 47, 56],
                                   va_base=0, palloc_ranges=[(4 << 10, 4 << 10)], reserve_ptable=False)
    self.is_booting = False

def n_pages(n): return n * 0x1000

# 2MB-aligned so warm()/small/big all land in one leaf page table (512 * 4KB entries) sharing the same already-created ancestor chain.
WARM_VA = 0x40000000

def warm(dev):
  # Establishes the PDE chain down to a leaf PT. A totally virgin VA range short-circuits the validation walk at a high (huge-page)
  # level instead of visiting per-page entries -- real address spaces are never virgin like that past the first-ever allocation, so
  # this mirrors the steady-state repeat-allocation case the validation pass actually costs latency on.
  dev.mm.map_range(WARM_VA, n_pages(1), paddrs=[(0x1000000, n_pages(1))], aspace=AddrSpace.PHYS)

def measure(dev, vaddr, n, paddr_base):
  before = dict(dev.stats)
  dev.mm.map_range(vaddr, n_pages(n), paddrs=[(paddr_base, n_pages(n))], aspace=AddrSpace.PHYS)
  return {'w': dev.stats['w'] - before['w'], 'r': dev.stats['r'] - before['r']}

class TestNVPTEBatching(unittest.TestCase):
  def test_write_batching_collapses_with_range_size(self):
    # Before this change: one MMIO write per PTE (N writes for N pages). After: one packed slice write per contiguous run, so the
    # write count stays flat (1) whether the range is 16 pages or 256 pages.
    dev = FakeNVDev(is_remote=False)
    warm(dev)
    small = measure(dev, WARM_VA + n_pages(16), 16, paddr_base=0x21000000)
    big = measure(dev, WARM_VA + n_pages(64), 256, paddr_base=0x22000000)
    assert small['w'] == 1, f"16-page map cost {small['w']} writes, expected 1 batched write"
    assert big['w'] == 1, f"256-page map cost {big['w']} writes, expected 1 batched write (unbatched would be 256)"

  def test_validation_reads_flat_when_remote_but_scale_when_local(self):
    # Tree descent itself (unrelated to this change) costs a fixed, non-zero number of reads per map_range call, on both passes --
    # so compare N=16 vs N=256 within each mode and look at the DELTA, which cancels that fixed cost and isolates the per-page cost.
    remote, local = FakeNVDev(is_remote=True), FakeNVDev(is_remote=False)
    for dev in (remote, local): warm(dev)

    r16 = measure(remote, WARM_VA + n_pages(16), 16, paddr_base=0x21000000)
    r256 = measure(remote, WARM_VA + n_pages(64), 256, paddr_base=0x22000000)
    assert r256['r'] == r16['r'], "remote validation reads must not grow with page count -- they should be skipped entirely"

    l16 = measure(local, WARM_VA + n_pages(16), 16, paddr_base=0x21000000)
    l256 = measure(local, WARM_VA + n_pages(64), 256, paddr_base=0x22000000)
    assert l256['r'] - l16['r'] == 240, "local (safe-by-default) validation should cost exactly one extra read per extra page"
    assert l16['r'] > r16['r'], "remote must do strictly fewer reads than local for the same range"

  def test_remote_validation_can_be_forced_back_on(self):
    getenv.cache_clear() # getenv is @functools.cache'd; must clear before the env var takes effect
    os.environ['NV_VALIDATE_REMOTE'] = '1'
    try:
      dev = FakeNVDev(is_remote=True)
      warm(dev)
      f16 = measure(dev, WARM_VA + n_pages(16), 16, paddr_base=0x21000000)
      f256 = measure(dev, WARM_VA + n_pages(64), 256, paddr_base=0x22000000)
      assert f256['r'] - f16['r'] == 240, "NV_VALIDATE_REMOTE=1 should force per-page validation back on even when remote"
    finally:
      del os.environ['NV_VALIDATE_REMOTE']
      getenv.cache_clear()

  def test_am_page_table_entry_has_no_batch_path(self):
    # AM's PT entry class intentionally has no set_entries: memory.py's map_range falls back to the old per-entry write loop for it,
    # so AM's write/validation semantics are untouched by this change. If this ever fires, that guarantee silently broke.
    assert not hasattr(AMPageTableEntry, 'set_entries')

def fake_iface(is_remote:bool|None):
  """Minimal stand-in for NVDevice.iface: `is_remote=None` mimics NVKIface/MOCKIface (no dev_impl at all -- Linux driver, mock, always
  local); True/False mimics PCIIface backed by a Remote/local PCIDevice (dev_impl.vram is the BAR1 MMIOInterface T2.2 tagged)."""
  return SimpleNamespace() if is_remote is None else SimpleNamespace(dev_impl=SimpleNamespace(vram=SimpleNamespace(is_remote=is_remote)))

class TestNVRemoteSizingSkeleton(unittest.TestCase):
  """T2.3: NVDevice has zero tuned remote values yet (no dock) -- this only proves the SELECTION works, mirroring AMD's is_usb()-keyed
  knobs (ops_amd.py:980-994). _LOCAL_SIZING and _REMOTE_SIZING hold identical numbers today; tuning later only edits _REMOTE_SIZING."""
  def test_no_dev_impl_iface_is_local(self):
    # NVKIface / MOCKIface (Linux driver, mock-NV): no dev_impl.vram at all -- must resolve to local, never raise.
    assert NVDevice.is_remote(SimpleNamespace(iface=fake_iface(None))) is False

  def test_local_pcidevice_iface_is_local(self):
    # PCIIface can back a local (non-Remote) PCIDevice too -- checked on the concrete vram view, not iface type.
    assert NVDevice.is_remote(SimpleNamespace(iface=fake_iface(False))) is False

  def test_remote_pcidevice_iface_is_remote(self):
    assert NVDevice.is_remote(SimpleNamespace(iface=fake_iface(True))) is True

  def test_sizing_selection_path_flips_on_remoteness(self):
    # Mirrors the exact selection NVDevice.__init__ performs. Values are equal right now (asserted below) -- what must hold is the
    # PATH: a local iface binds the _LOCAL_SIZING object, a remote one binds the distinct _REMOTE_SIZING object, by identity.
    local_dev, remote_dev = SimpleNamespace(iface=fake_iface(None)), SimpleNamespace(iface=fake_iface(True))
    local_sizing = NVDevice._REMOTE_SIZING if NVDevice.is_remote(local_dev) else NVDevice._LOCAL_SIZING
    remote_sizing = NVDevice._REMOTE_SIZING if NVDevice.is_remote(remote_dev) else NVDevice._LOCAL_SIZING
    assert local_sizing is NVDevice._LOCAL_SIZING and local_sizing is not NVDevice._REMOTE_SIZING
    assert remote_sizing is NVDevice._REMOTE_SIZING and remote_sizing is not NVDevice._LOCAL_SIZING
    assert NVDevice._LOCAL_SIZING == NVDevice._REMOTE_SIZING, "values must be identical pre-dock -- a no-op skeleton, not a tuned one"
    assert NVDevice._LOCAL_SIZING is not NVDevice._REMOTE_SIZING, "but distinct objects, so tuning later is a values-only edit"

class TestNVBindRemoteBatching(unittest.TestCase):
  """T2.3's one gated exception: NVCommandQueue.bind() writes the queue into a device buffer one 32-bit word at a time. If that
  buffer is RPC-backed (T2.2's is_remote), that's len(_q) separate socket sendalls -- the same pathology T2.2 batched away for PTE
  writes. Local (incl. NVK-Linux, where is_remote is always False) keeps the original per-word loop, byte-for-byte."""
  def _bind_and_count(self, is_remote:bool, n_words:int=200) -> dict:
    mmio = CountingMMIOInterface(bytearray(n_words * 4), 0, n_words * 4, fmt='B', is_remote=is_remote)
    hw_page = SimpleNamespace(cpu_view=lambda: mmio, size=n_words * 4)
    # is_remote() False models the pre-T4.18 hw_page path: these tests parameterize the BUFFER VIEW's
    # is_remote (T2.2/T2.3's write batching), which is orthogonal to T4.18's device-level slab.
    fake_dev = SimpleNamespace(allocator=SimpleNamespace(alloc=lambda size, options: hw_page, free=lambda *a, **k: None),
                               is_remote=lambda: False)

    q = NVCommandQueue()
    q._q = list(range(0xd000, 0xd000 + n_words))
    q.bind(fake_dev)
    assert list(q._q[:]) == list(range(0xd000, 0xd000 + n_words)), "bound queue content must be unchanged regardless of write path"
    return mmio.stats

  def test_local_buffer_keeps_per_word_writes(self):
    stats = self._bind_and_count(is_remote=False, n_words=200)
    assert stats['w'] == 200, f"local bind() should cost 200 individual writes (unchanged behavior), got {stats['w']}"

  def test_remote_buffer_collapses_to_one_bulk_write(self):
    stats = self._bind_and_count(is_remote=True, n_words=200)
    assert stats['w'] == 1, f"remote bind() should collapse to 1 bulk slice write regardless of queue length, got {stats['w']}"

class TestNVBindHwPageSlab(unittest.TestCase):
  """T4.18: on the remote/APL transport the wire protocol has no unmap verb, so every bind()'s hw_page permanently consumes one of
  ~128 sysmem slots -- a cross-device MoE graph (~85 islands) exhausts them mid-capture. Remote bind() must therefore suballocate
  from ONE slab instead of allocating per bind; local/NVK keeps one alloc per bind, byte-for-byte."""
  def _bind_n(self, is_remote:bool, n_binds:int=32, n_words:int=64) -> int:
    allocs = []
    def fake_alloc(size, options):
      allocs.append(size)
      buf = bytearray(size)
      page:SimpleNamespace = SimpleNamespace(size=size, base=None,
        cpu_view=lambda buf=buf: CountingMMIOInterface(buf, 0, len(buf), fmt='B', is_remote=is_remote))
      # .base mirrors HCQBuffer.offset()'s parent link -- bind()'s __del__ uses it to skip freeing slab suballocations
      page.offset = lambda off, sz, buf=buf, page=page: SimpleNamespace(size=sz, base=page,
        cpu_view=lambda: CountingMMIOInterface(buf, off, sz, fmt='B', is_remote=is_remote))
      return page
    fake_dev = SimpleNamespace(allocator=SimpleNamespace(alloc=fake_alloc, free=lambda *a, **k: None),
                               is_remote=lambda: is_remote, _hwq_slab=None, _hwq_bump=None)
    for _ in range(n_binds):
      q = NVCommandQueue()
      q._q = list(range(0xd000, 0xd000 + n_words))
      q.bind(fake_dev)
      assert list(q._q[:]) == list(range(0xd000, 0xd000 + n_words)), "bound queue content must survive slab suballocation"
    return len(allocs)

  def test_remote_binds_share_one_slab(self):
    # pre-T4.18 this was 32 allocations = 32 permanently-held sysmem slots
    assert (n:=self._bind_n(is_remote=True, n_binds=32)) == 1, f"32 remote binds should consume ONE slab allocation, got {n}"

  def test_local_binds_allocate_per_bind(self):
    assert (n:=self._bind_n(is_remote=False, n_binds=32)) == 32, f"local bind() must keep one alloc per bind (unchanged), got {n}"

class TestNVKernargsPoolSlab(unittest.TestCase):
  """T4.20: HCQGraph.__init__ (graph/hcq.py:32) allocates one kernargs buffer per graph island per device,
  freed in __del__ -- but T4.18 already proved a remote sysmem free is a client-side no-op (no unmap verb),
  so every island permanently burns one of the ~128 sysmem slots. A many-island graph (an experts: split
  measured ~85 islands) can exhaust them. Pool kernargs the same way T4.18 pooled hw_page: one never-freed
  slab, bump-suballocated; local/NVK keeps one alloc per island, byte-for-byte."""
  def _alloc_free_n(self, is_remote:bool, n:int=40, size:int=256) -> int:
    allocs = []
    def fake_alloc(sz, options):
      allocs.append(sz)
      page:SimpleNamespace = SimpleNamespace(size=sz, base=None)
      page.offset = lambda off, osz, page=page: SimpleNamespace(size=osz, base=page)
      return page
    fake_dev = SimpleNamespace(allocator=SimpleNamespace(alloc=fake_alloc, _alloc=fake_alloc, _free=lambda *a, **k: None),
                               is_remote=lambda: is_remote, _kernargs_slab=None, _kernargs_bump=None)
    bufs = [NVDevice.alloc_kernargs(fake_dev, size) for _ in range(n)]
    for b in bufs: NVDevice.free_kernargs(fake_dev, b)
    return len(allocs)

  def test_remote_kernargs_share_one_slab(self):
    # pre-T4.20 this was 40 allocations = 40 permanently-held sysmem slots (freeing them is a remote no-op)
    assert (n:=self._alloc_free_n(is_remote=True, n=40)) == 1, f"40 remote kernargs allocs should consume ONE slab allocation, got {n}"

  def test_local_kernargs_allocate_per_call(self):
    assert (n:=self._alloc_free_n(is_remote=False, n=40)) == 40, f"local kernargs alloc must keep one alloc per call (unchanged), got {n}"

class TestNVFaultRecoveryHint(unittest.TestCase):
  """T4.23: an NV device fault (is_err_state) is genuinely GSP/hardware-reported (support/nv/ip.py sets it only from real
  NV_VGPU_MSG_EVENT_OS_ERROR_LOG/MMU_FAULT_QUEUED messages) and NV never sets can_recover (hcq.py) -- there is no safe
  in-process reset, so the raised error must name the out-of-band fix instead of a bare message. Remote-only: local
  NVK/mock have no TinyGPU.app server to respawn, so their message must stay byte-for-byte unchanged."""
  def test_hint_names_the_fix_when_remote(self):
    hint = _fault_recovery_hint(SimpleNamespace(is_remote=lambda: True))
    assert "pkill -f 'TinyGPU.*server'" in hint, hint

  def test_hint_is_empty_when_local(self):
    assert _fault_recovery_hint(SimpleNamespace(is_remote=lambda: False)) == ""

  def _fake_sleep_self(self, is_remote:bool):
    stat_q = SimpleNamespace(read_resp=lambda: iter(()))
    return SimpleNamespace(dev_impl=SimpleNamespace(gsp=SimpleNamespace(stat_q=stat_q), is_err_state=True),
                            dev=SimpleNamespace(is_remote=lambda: is_remote))

  def test_pciiface_sleep_raises_with_hint_when_remote(self):
    with self.assertRaises(RuntimeError) as ctx: PCIIface.sleep(self._fake_sleep_self(True), 200)
    assert "pkill -f 'TinyGPU.*server'" in str(ctx.exception), ctx.exception

  def test_pciiface_sleep_message_unchanged_when_local(self):
    with self.assertRaises(RuntimeError) as ctx: PCIIface.sleep(self._fake_sleep_self(False), 200)
    assert str(ctx.exception) == "Device fault detected.", ctx.exception

class TestAPLRemotePCIDeviceSpawn(unittest.TestCase):
  """T4.25: APLRemotePCIDevice.__init__'s connect-or-spawn loop (support/system.py) had no cross-process
  lock, so simultaneous constructors could each fail their first connect() and each spawn a TinyGPU.app
  server. Live repro on this dock: racing 10 real TinyGPU.app processes to bind() one fresh socket path
  left only 4 with a clean 'bind: File exists'/'Address already in use' exit -- the other 6 landed in
  accept() on a socket a losing peer's own stale-socket recovery had unlinked out from under them, and hung
  there forever (confirmed with lsof + sample). Two of the three originally-suspected defects did NOT
  reproduce on this platform and are deliberately left unaddressed: respawning over a genuinely stale
  (dead-server) socket works fine as-is (TinyGPU.app unlinks and rebinds it itself: inode changes, client
  connects, zero lingering -- verified live, twice); and retrying connect() on the same AF_UNIX SOCK_STREAM
  socket after ECONNREFUSED/ENOENT works fine on macOS, unlike Linux (verified directly, no EINVAL). The
  fix actually needed is just a flock(LOCK_EX) around the whole connect-or-spawn sequence, serializing
  spawns so the race above can never start. Drives the real subprocess/socket/flock path with a fake
  APP_PATH (a trivial bind+listen / hang script); only RemotePCIDevice.__init__ (the unrelated
  device-ownership handshake that follows, not part of this fix) is stubbed out."""

  def setUp(self):
    self.tmpdir = tempfile.mkdtemp(prefix="t425_")
    self.sock_path = os.path.join(self.tmpdir, "tinygpu.sock")
    self.procs:list[subprocess.Popen] = []

  def tearDown(self):
    for p in self.procs:
      try:
        p.kill()
        p.wait(timeout=2)
      except Exception:
        pass
    shutil.rmtree(self.tmpdir, ignore_errors=True)

  def _script(self, name:str, body:str) -> str:
    path = os.path.join(self.tmpdir, name)
    with open(path, "w") as f: f.write("#!/usr/bin/env python3\n" + body)
    os.chmod(path, 0o755)
    return path

  def _fake_popen(self):
    real_popen = subprocess.Popen
    def counting(*a, **k):
      p = real_popen(*a, **k)
      self.procs.append(p)
      return p
    return counting

  def _sock_getenv(self, k, d=None): return self.sock_path if k == "APL_REMOTE_SOCK" else d

  def test_concurrent_construction_spawns_exactly_once(self):
    # listen() backlog must comfortably exceed the thread count below: nothing ever calls accept(), so every
    # successful connect() just occupies one backlog slot permanently -- too small a backlog looks like a
    # (suppressed, retried, eventually-timing-out) connection refusal and is a test-fixture bug, not a fix bug.
    server = self._script("server.py", "import socket, sys, time\n"
                                        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                                        "s.bind(sys.argv[-1])\n"
                                        "s.listen(64)\n"
                                        "time.sleep(30)\n")

    class _FakeAPL(APLRemotePCIDevice):
      APP_PATH = server
      def ensure_app(self): pass

    results:list = []
    def worker():
      try:
        _FakeAPL("NV", "0000:00:00.0")
        results.append("ok")
      except Exception as e:
        results.append(e)

    with patch("subprocess.Popen", side_effect=self._fake_popen()), \
         patch("tinygrad.runtime.support.system.getenv", side_effect=self._sock_getenv), \
         patch.object(RemotePCIDevice, "__init__", lambda self, *a, **k: None):
      threads = [threading.Thread(target=worker) for _ in range(6)]
      for t in threads: t.start()
      for t in threads: t.join(timeout=10)

    assert results == ["ok"] * 6, f"all 6 concurrent constructions should succeed, got {results}"
    assert len(self.procs) == 1, f"6 constructors racing an empty socket dir should spawn exactly ONE server, got {len(self.procs)}"

  def test_failed_spawn_is_killed_not_orphaned(self):
    hang = self._script("hang.py", "import time\ntime.sleep(60)\n")  # never binds -> connect() never succeeds

    class _FakeAPL(APLRemotePCIDevice):
      APP_PATH = hang
      def ensure_app(self): pass

    with patch("subprocess.Popen", side_effect=self._fake_popen()), \
         patch("tinygrad.runtime.support.system.getenv", side_effect=self._sock_getenv), \
         patch("time.sleep", lambda s: None), \
         patch.object(RemotePCIDevice, "__init__", lambda self, *a, **k: None):
      with self.assertRaisesRegex(RuntimeError, "Failed to connect"):
        _FakeAPL("NV", "0000:00:00.0")

    assert len(self.procs) == 1
    self.procs[0].wait(timeout=2)
    assert self.procs[0].poll() is not None, "a spawn that never connects must be killed, not orphaned"

if __name__ == "__main__":
  unittest.main()
