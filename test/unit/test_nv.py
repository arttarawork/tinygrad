import contextlib, io, itertools, os, shutil, socket, struct, subprocess, tempfile, threading, time, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.helpers import getenv, Context
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry
from tinygrad.runtime.support.nv.ip import NV_FLCN
from tinygrad.runtime.support.am.amdev import AMPageTableEntry
from tinygrad.runtime.support.system import APLRemotePCIDevice, RemotePCIDevice
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, PCIIface, NVSignal, DispatchRing, _fault_recovery_hint
from tinygrad.runtime.support.hcq import HCQAllocatorBase, HCQBuffer, _dev_already_faulted

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
    # T4.70d: the skeleton is no longer a no-op -- remote kernargs_size is tuned (16MB starved under FFN-TP's
    # dispatch burst; the per-alloc fallback then hit the TinyGPU tunnel's per-client map cap -- see ops_nv.py).
    # Everything else must stay identical; assert the exact intended divergence, nothing more.
    assert NVDevice._REMOTE_SIZING["kernargs_size"] == 64 << 20 and NVDevice._LOCAL_SIZING["kernargs_size"] == 16 << 20
    assert {k: v for k, v in NVDevice._LOCAL_SIZING.items() if k != "kernargs_size"} == \
           {k: v for k, v in NVDevice._REMOTE_SIZING.items() if k != "kernargs_size"}, "only kernargs_size may diverge"
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

class _FakePCIConfig:
  """In-memory PCI config space: just enough of PCIDevice's read_config/write_config_flush (T4.37) plus pcibus/map_bar
  (T4.40b, for driving the real NVDev.__init__) for the bus-master tests."""
  def __init__(self, command:int=pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER):
    self.space = {pci.PCI_COMMAND: command}
    self.pcibus = "0000:66:00.0"
  def read_config(self, offset:int, size:int) -> int: return self.space.get(offset, 0)
  def write_config_flush(self, offset:int, value:int, size:int) -> None: self.space[offset] = value
  def map_bar(self, *a, **k): return None

def _fake_pciiface_self(is_remote:bool) -> SimpleNamespace:
  """Minimal stand-in for a PCIIface `self`, faulted: enough of dev_impl/pci_dev for sleep()'s quiesce-then-raise path."""
  stat_q = SimpleNamespace(read_resp=lambda: iter(()))
  return SimpleNamespace(pci_dev=_FakePCIConfig(), dev=SimpleNamespace(is_remote=lambda: is_remote),
                          dev_impl=SimpleNamespace(gsp=SimpleNamespace(stat_q=stat_q), is_err_state=True))

class TestNVFaultRecoveryHint(unittest.TestCase):
  """T4.23: an NV device fault (is_err_state) is genuinely GSP/hardware-reported (support/nv/ip.py sets it only from real
  NV_VGPU_MSG_EVENT_OS_ERROR_LOG/MMU_FAULT_QUEUED messages) and NV never sets can_recover (hcq.py) -- there is no safe
  in-process reset, so the raised error must name the out-of-band fix instead of a bare message. Remote-only: local
  NVK/mock have no TinyGPU.app server to respawn, so their message must stay byte-for-byte unchanged.

  T4.37: the fix named here used to be `pkill -f 'TinyGPU.*server'` -- doing exactly that under a live fault is what
  DMA-panicked the host on 2026-08-26 (T4.36), so the hint was rewritten to point at a fresh client's controlled reset
  instead. See TestNVQuiesceOnFault below for the bus-master-clearing mechanism that makes that safe."""
  def test_hint_names_the_fix_when_remote(self):
    hint = _fault_recovery_hint(SimpleNamespace(is_remote=lambda: True))
    assert "fresh client" in hint.lower() and "bus-master" in hint.lower(), hint
    assert "pkill -f 'TinyGPU" not in hint, "must no longer tell the user to pkill the server -- that caused T4.36"

  def test_hint_is_empty_when_local(self):
    assert _fault_recovery_hint(SimpleNamespace(is_remote=lambda: False)) == ""

  def test_pciiface_sleep_raises_with_hint_when_remote(self):
    with self.assertRaises(RuntimeError) as ctx: PCIIface.sleep(_fake_pciiface_self(True), 200)
    assert "fresh client" in str(ctx.exception).lower(), ctx.exception

  def test_pciiface_sleep_message_unchanged_when_local(self):
    with self.assertRaises(RuntimeError) as ctx: PCIIface.sleep(_fake_pciiface_self(False), 200)
    assert str(ctx.exception) == "Device fault detected.", ctx.exception

class TestNVQuiesceOnFault(unittest.TestCase):
  """T4.37: the 2026-08-26 DART panic (T4.36) happened because the NV fault path raised without ever clearing PCI
  bus-master -- a faulted GPU's GSP firmware keeps DMAing into host-sysmem queues, and killing the server (T4.23's old
  recovery hint) tore those mappings down while the GPU could still reach them. These assert the fault path clears
  MASTER (idempotently, and without touching other PCI_COMMAND bits) before it raises, and that backends with no
  pci_dev at all (NVKIface/MOCKIface) are provably untouched."""

  def test_pciiface_sleep_clears_bus_master_on_fault(self):
    fake = _fake_pciiface_self(True)
    cmd = fake.pci_dev.read_config(pci.PCI_COMMAND, 2)
    assert cmd & pci.PCI_COMMAND_MASTER and cmd & pci.PCI_COMMAND_MEMORY, "fixture should start with MASTER+MEMORY set"
    with self.assertRaises(RuntimeError): PCIIface.sleep(fake, 200)
    cmd = fake.pci_dev.read_config(pci.PCI_COMMAND, 2)
    assert not (cmd & pci.PCI_COMMAND_MASTER), "MASTER must be cleared after a fault"
    assert cmd & pci.PCI_COMMAND_MEMORY, "only MASTER should be touched, not the rest of PCI_COMMAND"

  def test_pciiface_sleep_clear_is_idempotent(self):
    fake = _fake_pciiface_self(True)
    with self.assertRaises(RuntimeError): PCIIface.sleep(fake, 200)
    # is_err_state never clears itself and MASTER is already off -- a second sleep() (e.g. a stale poller) must still
    # raise cleanly, not blow up because the bit was already clear.
    with self.assertRaises(RuntimeError) as ctx: PCIIface.sleep(fake, 200)
    assert not (fake.pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER)
    assert "fresh client" in str(ctx.exception).lower(), ctx.exception

  def _fake_hang_self(self, is_nvd:bool):
    pci_dev = _FakePCIConfig()
    no_fault = SimpleNamespace(mmuFault=SimpleNamespace(valid=False), smErrorStateArray=[])
    iface = SimpleNamespace(pci_dev=pci_dev, rm_control=lambda *a, **k: no_fault)
    return SimpleNamespace(iface=iface, debugger=None, debug_channel=0, is_remote=lambda: True, is_nvd=lambda: is_nvd), pci_dev

  def test_on_device_hang_clears_bus_master_when_nvd(self):
    fake, pci_dev = self._fake_hang_self(is_nvd=True)
    with self.assertRaises(RuntimeError): NVDevice.on_device_hang(fake)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), "MASTER must be cleared after a hang"

  def test_on_device_hang_leaves_non_pciiface_untouched(self):
    # NVKIface/MOCKIface have no pci_dev at all -- is_nvd() must gate the write so this path is never touched for them.
    fake, pci_dev = self._fake_hang_self(is_nvd=False)
    with self.assertRaises(RuntimeError): NVDevice.on_device_hang(fake)
    assert pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER, "must be untouched when not is_nvd()"

  def test_device_fini_clears_bus_master_when_faulted(self):
    pci_dev = _FakePCIConfig()
    fake = SimpleNamespace(pci_dev=pci_dev, dev_impl=SimpleNamespace(fini=lambda: None, is_err_state=True))
    PCIIface.device_fini(fake)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER)

  def test_device_fini_clears_bus_master_even_if_fini_raises(self):
    # Seen live (T4.37 step A): fini() -> rpc_unloading_guest_driver() times out against a dead GSP and raises BEFORE the
    # clear -- the clear must still run (try/finally) or a quiet exit with a live fault leaves a bus-mastering GPU.
    pci_dev = _FakePCIConfig()
    def fini_raises(): raise RuntimeError("Timeout waiting for RPC response for command 47")
    fake = SimpleNamespace(pci_dev=pci_dev, dev_impl=SimpleNamespace(fini=fini_raises, is_err_state=True))
    with self.assertRaisesRegex(RuntimeError, "command 47"): PCIIface.device_fini(fake)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), "MASTER must be cleared even when fini() raises"

  def test_device_fini_clears_bus_master_when_fini_raises_without_err_state(self):
    # T4.40b (closes an A1 hole): is_err_state is set only by a GSP-delivered fault event (support/nv/ip.py) -- a wedge
    # that never delivered one but whose fini() unload RPC itself times out/raises must still clear MASTER. Pre-fix this
    # exited with MASTER on: the old `finally: if is_err_state` guard never fires when is_err_state is False.
    pci_dev = _FakePCIConfig()
    def fini_raises(): raise TimeoutError("Timeout waiting for RPC response for command 47")
    fake = SimpleNamespace(pci_dev=pci_dev, dev_impl=SimpleNamespace(fini=fini_raises, is_err_state=False))
    with self.assertRaisesRegex(TimeoutError, "command 47"): PCIIface.device_fini(fake)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), \
      "MASTER must be cleared when fini() raises even without is_err_state"

  def test_device_fini_leaves_bus_master_when_clean(self):
    pci_dev = _FakePCIConfig()
    fake = SimpleNamespace(pci_dev=pci_dev, dev_impl=SimpleNamespace(fini=lambda: None, is_err_state=False))
    PCIIface.device_fini(fake)
    assert pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER, "a clean exit must not touch bus-master"

def _fake_dev_for_free(sync_calls:list, is_err_state:bool, raise_on_sync:bool=False) -> SimpleNamespace:
  """Minimal HCQCompiled-device stand-in for HCQAllocatorBase._free's per-buffer loop: enough of
  iface.dev_impl.is_err_state for _dev_already_faulted, plus a synchronize() spy. raise_on_sync mirrors the
  real fault path (ops_nv.py:655's "Device fault detected") so a broken guard shows up as a raised exception,
  not just a miscounted call."""
  def synchronize(timeout=None):
    sync_calls.append(1)
    if raise_on_sync: raise RuntimeError("Device fault detected. NV device is in an unrecoverable fault state.")
  return SimpleNamespace(iface=SimpleNamespace(dev_impl=SimpleNamespace(is_err_state=is_err_state)), synchronize=synchronize)

class TestNVFastFaultedTeardown(unittest.TestCase):
  """T4.52 (T4.47_RCA.md follow-up; T4.50 named the storm anatomy): once is_err_state is set, bus-master is
  already cleared (T4.37/T4.40b) and the mappings die with the process regardless, so a per-buffer
  synchronize() during teardown -- LRU free_cache, Buffer.__del__, HCQCompiled.finalize's cache sweep, all of
  which funnel through HCQAllocatorBase._free -- is pointless. Pre-fix it re-raised "Device fault detected"
  once per still-cached buffer: an observed ~120-buffer echo storm, ~490s of teardown. These prove _free skips
  the sync (and thus the raise) once the fault is already known, while a healthy device's teardown -- and the
  first, not-yet-discovered fault on any device -- is untouched."""

  def _fake_allocator(self, do_free_calls:list) -> SimpleNamespace:
    return SimpleNamespace(_do_free=lambda buf, options: do_free_calls.append((buf, options)))

  def test_free_skips_sync_on_already_faulted_device(self):
    sync_calls:list = []
    dev = _fake_dev_for_free(sync_calls, is_err_state=True, raise_on_sync=True)
    buf = HCQBuffer(va_addr=0x1000, size=0x1000, owner=dev)
    do_free_calls:list = []
    HCQAllocatorBase._free(self._fake_allocator(do_free_calls), buf, None) # must not raise
    assert sync_calls == [], "a known-faulted device must not be synchronized per buffer"
    assert do_free_calls == [(buf, None)], "client-side bookkeeping (_do_free) must still run"

  def test_free_skips_sync_for_n_buffers_zero_echoes(self):
    # T4.50's storm anatomy: ~120 near-identical per-buffer echoes for one fault. Proves the guard holds across
    # a whole teardown sweep (LRU free_cache's shape: many buffers freed off one faulted device in a row), not
    # just a single buffer.
    sync_calls:list = []
    dev = _fake_dev_for_free(sync_calls, is_err_state=True, raise_on_sync=True)
    do_free_calls:list = []
    alloc = self._fake_allocator(do_free_calls)
    for i in range(120): HCQAllocatorBase._free(alloc, HCQBuffer(va_addr=0x1000 * i, size=0x1000, owner=dev), None)
    assert sync_calls == [], f"expected zero sync RPCs across 120 buffer frees on a faulted device, got {len(sync_calls)}"
    assert len(do_free_calls) == 120, "every buffer must still get its client-side free"

  def test_free_still_synchronizes_healthy_device(self):
    # Negative control: a healthy device's teardown must be byte-identical to pre-fix -- synchronize() still
    # runs exactly once per buffer.
    sync_calls:list = []
    dev = _fake_dev_for_free(sync_calls, is_err_state=False)
    buf = HCQBuffer(va_addr=0x1000, size=0x1000, owner=dev)
    do_free_calls:list = []
    HCQAllocatorBase._free(self._fake_allocator(do_free_calls), buf, None)
    assert sync_calls == [1], "a healthy device must still be synchronized exactly once per buffer"
    assert do_free_calls == [(buf, None)]

  def test_free_still_raises_on_first_undiscovered_fault(self):
    # The first fault surfacing must stay loud: is_err_state False (not yet discovered) means the guard must
    # not fire, so synchronize() runs and a genuine failure there still raises out of _free normally.
    sync_calls:list = []
    dev = _fake_dev_for_free(sync_calls, is_err_state=False, raise_on_sync=True)
    buf = HCQBuffer(va_addr=0x1000, size=0x1000, owner=dev)
    do_free_calls:list = []
    with self.assertRaises(RuntimeError):
      HCQAllocatorBase._free(self._fake_allocator(do_free_calls), buf, None)
    assert sync_calls == [1], "an undiscovered fault must still be probed via a real synchronize() call"
    assert do_free_calls == [], "the buffer's client-side bookkeeping never runs once synchronize() raises (pre-existing _free ordering, unchanged)"

  def test_dev_already_faulted_true_when_err_state_set(self):
    assert _dev_already_faulted(SimpleNamespace(iface=SimpleNamespace(dev_impl=SimpleNamespace(is_err_state=True))))

  def test_dev_already_faulted_false_when_healthy(self):
    assert not _dev_already_faulted(SimpleNamespace(iface=SimpleNamespace(dev_impl=SimpleNamespace(is_err_state=False))))

  def test_dev_already_faulted_false_with_no_iface(self):
    # CPU and other non-HCQ backends have no `iface` attribute at all -- must degrade to False, never AttributeError.
    assert not _dev_already_faulted(SimpleNamespace())

  def test_dev_already_faulted_false_with_no_dev_impl(self):
    # NVKIface/MOCKIface-shaped: has `iface` but no `dev_impl` (only PCIIface sets one) -- must degrade to False.
    assert not _dev_already_faulted(SimpleNamespace(iface=SimpleNamespace()))

class TestNVQuiesceOnInitFailure(unittest.TestCase):
  """T4.40b (RCA T4.40_RCA.md fix 40-1 / mechanism A2): _early_ip_init() sets PCI bus-master partway through
  NVDev.__init__, before flcn/gsp ever boot (nvdev.py:113-137, MASTER set at :127). A boot-time RPC timeout raising
  anywhere after that point -- the confirmed real trigger is a get_available_devices() probe during pytest collection
  -- left GSP-RM armed with bus-master on and never unloaded, because a device whose __init__ raises is never
  registered (device.py never calls fini()/device_fini() for it): the armed session lingers until process exit tears
  down its DMA mappings. This is panic 2's best-fit mechanism.

  These drive the REAL NVDev.__init__ (not a copy) against a fake pci_dev with _early_ip_init/_early_mmu_init/flcn/gsp
  stubbed out -- proving the try/except placement across both seams (inside _early_ip_init, e.g. wait_for_reset()'s
  wait_cond; and after it returns, in init_sw()/init_hw()), not re-testing _early_ip_init's own register decode logic
  (already exercised by TestNVPTEBatching's FakeNVDev)."""

  def _fake_target(self, pci_dev, early_ip_init, flcn=None, gsp=None) -> SimpleNamespace:
    fake = SimpleNamespace()
    fake._early_ip_init = early_ip_init
    fake._early_mmu_init = lambda: None
    fake.flcn = flcn if flcn is not None else SimpleNamespace(init_sw=lambda: None, init_hw=lambda: None)
    fake.gsp = gsp if gsp is not None else SimpleNamespace(init_sw=lambda: None, init_hw=lambda: None)
    return fake

  def _set_master(self, pci_dev): pci_dev.write_config_flush(pci.PCI_COMMAND, pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)

  def test_init_clears_bus_master_when_early_ip_init_raises_after_master_set(self):
    # Mirrors the real seam exactly: nvdev.py sets MASTER partway through _early_ip_init, then calls wait_for_reset()
    # (a wait_cond, same 10s-timeout-then-raise primitive as the RPC waits) before that method returns to __init__.
    pci_dev = _FakePCIConfig()
    def early_ip_init_raises_after_master():
      self._set_master(pci_dev)
      raise TimeoutError("waiting for reset")
    fake = self._fake_target(pci_dev, early_ip_init_raises_after_master)
    with self.assertRaisesRegex(TimeoutError, "waiting for reset"): NVDev.__init__(fake, pci_dev)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), \
      "MASTER must be cleared when _early_ip_init raises after setting it (a narrower fix that only wraps the code " \
      "AFTER _early_ip_init() returns would miss this and fail here)"

  def test_init_clears_bus_master_when_a_later_boot_step_raises(self):
    # init_sw()/init_hw() run AFTER _early_ip_init() returns -- the other seam the fix must cover (RCA's "confirmed
    # real trigger": a boot-time RPC timeout during this phase).
    pci_dev = _FakePCIConfig()
    def early_ip_init_ok(): self._set_master(pci_dev)
    def init_hw_raises(): raise TimeoutError("Timeout waiting for RPC response for command 12")
    fake = self._fake_target(pci_dev, early_ip_init_ok, gsp=SimpleNamespace(init_sw=lambda: None, init_hw=init_hw_raises))
    with self.assertRaisesRegex(TimeoutError, "command 12"): NVDev.__init__(fake, pci_dev)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), \
      "MASTER must be cleared when a later boot step (init_sw/init_hw) raises"

  def test_init_leaves_bus_master_set_when_boot_succeeds(self):
    # Negative control (the #16536 guard): a healthy boot must leave MASTER exactly as _early_ip_init set it, and
    # must not raise.
    pci_dev = _FakePCIConfig()
    def early_ip_init_ok(): self._set_master(pci_dev)
    fake = self._fake_target(pci_dev, early_ip_init_ok)
    NVDev.__init__(fake, pci_dev)
    assert pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER, "a healthy boot must not touch bus-master"
    assert fake.is_booting is False, "healthy __init__ must still run to completion (not swallowed by the new except)"

class _RegProbe:
  """Throwaway stand-in for resolving a register's real NVReg (address + fields) through NVDev's own
  include()/reg(), independent of any live mmio/pci_dev -- so fixtures below never hand-copy offsets out
  of the autogen tables."""
  include, reg = NVDev.include, NVDev.reg

def _reg(group:str, arch:str, name:str):
  probe = _RegProbe()
  probe.include(group, arch)
  return probe.reg(name)

class _FakeIpInitDev:
  """Drives the REAL NVDev._early_ip_init() (T4.40c) against a fake BAR0 register file + fake PCI config
  space. Reuses NVDev's own include()/reg()/wreg()/rreg() unmodified -- they're pure Python dict bookkeeping
  over self.mmio, nothing to fake there -- and fakes only the two real I/O boundaries: self.mmio (BAR0
  MMIO, see _EventBar0) and self.pci_dev (config space + the FLR, see _FakeFLRPCIConfig)."""
  include, reg, wreg, rreg = NVDev.include, NVDev.reg, NVDev.wreg, NVDev.rreg
  def __init__(self, pci_dev, mmio):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, mmio

class _FakeFLRPCIConfig(_FakePCIConfig):
  """_FakePCIConfig plus an FLR hook and event-ordered write_config_flush, so a test can assert MASTER
  isn't set until after the verify step's register reads (see _EventBar0, which appends into the same
  `events` list). reset() is a pure log entry: T4.40c's fix does not -- and per the RCA, cannot yet --
  assume anything about what an FLR itself does to a wedged core; that unknown is exactly what the fix
  works around instead of relying on."""
  def __init__(self, events:list, mmio, command:int=pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER):
    super().__init__(command)
    self.events, self._mmio = events, mmio
  def write_config_flush(self, offset:int, value:int, size:int) -> None:
    self.events.append(("cfg", offset, value))
    super().write_config_flush(offset, value, size)
  def reset(self) -> None: self.events.append(("flr",))
  def map_bar(self, *a, **k): return self._mmio

class _EventBar0:
  """Word-addressed (matches NVDev.rreg/wreg's `addr // 4`) fake BAR0 register file. Every access appends
  to the same ordered `events` list _FakeFLRPCIConfig writes into, so a test can assert real happens-before
  ordering between MMIO reads and PCI_COMMAND config writes.

  Models the RCA's M-C hypothesis instead of just asserting it: riscv CPUCTL.active_stat (bit 7) reads back
  1 (active) until the GSP falcon engine reset actually completes -- NV_PGSP_FALCON_ENGINE reset=1 then
  reset=0, ip.py NV_FLCN.reset()'s own unmodified primitive -- then reads 0. That proves the fix's verify
  step is watching a real consequence of calling reset(), not a register that merely starts at 0.
  `wedged=True` makes the engine reset have no effect on it at all: a core neither the FLR nor a falcon
  engine reset brings down (T4.40_RCA.md Sec5's open worst case)."""
  def __init__(self, events:list, wedged:bool=False):
    self.events, self.wedged, self.words, self.engine_reset_done = events, wedged, {}, False
    self.engine_addr = self.cpuctl_addr = -1  # filled in by _make_fake_hw once addresses are resolved

  def __getitem__(self, idx):
    if idx == self.cpuctl_addr: val = 0 if (self.engine_reset_done and not self.wedged) else (1 << 7)
    else: val = self.words.get(idx, 0)
    self.events.append(("r", idx, val))
    return val

  def __setitem__(self, idx, val):
    self.words[idx] = val
    if idx == self.engine_addr and (val & 1) == 0: self.engine_reset_done = True
    self.events.append(("w", idx, val))

def _poke(mmio:_EventBar0, r, raw:int|None=None, **fields):
  mmio.words[(r.base + r.off) // 4] = raw if raw is not None else r.encode(**fields)

def _make_fake_hw(wedged:bool) -> tuple[list, _FakeFLRPCIConfig, _EventBar0]:
  """Common fixture for TestNVHaltVerifyBeforeMaster: primes exactly the registers _early_ip_init()'s new
  code path touches -- GA102/non-COT chip identity (so self.flcn is a real NV_FLCN, matching the actual
  target hardware), WPR2 up (a previous GSP-RM session -- the FLR branch this fix changes), and
  wait_for_reset()'s post-boot scratch sentinel (unrelated to this fix, but still on the unconditional tail
  of _early_ip_init() that every test here runs through whenever it doesn't raise first)."""
  events:list = []
  mmio = _EventBar0(events, wedged=wedged)
  pci_dev = _FakeFLRPCIConfig(events, mmio)

  eng, cpuctl = _reg("dev_gsp", "ga102", "NV_PGSP_FALCON_ENGINE"), _reg("dev_riscv_pri", "ga102", "NV_PRISCV_RISCV_CPUCTL").with_base(0x00110000)
  mmio.engine_addr, mmio.cpuctl_addr = (eng.base + eng.off) // 4, (cpuctl.base + cpuctl.off) // 4

  _poke(mmio, _reg("nv_ref", "", "NV_PMC_BOOT_42"), architecture=0x17, implementation=2)          # GA102 (non-COT, exercises NV_FLCN)
  _poke(mmio, _reg("dev_fb", "tu102", "NV_PFB_PRI_MMU_WPR2_ADDR_HI"), val=1)                       # a previous session -> take the FLR branch
  _poke(mmio, _reg("dev_gc6_island", "ga102", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK"), read_protection_level0=1)
  _poke(mmio, _reg("dev_gc6_island", "ga102", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05")[0], raw=0xff)
  return events, pci_dev, mmio

def _fast_timeout_clock():
  """time.perf_counter() replacement that advances the mocked clock 50ms (0.05s -- perf_counter() returns
  SECONDS; wait_cond does int(perf_counter()*1000) to get ms) per call: a wait_cond() that never meets its
  condition hits the (unmodified, real) 10s default timeout in ~200 fast calls instead of a real 10s
  busy-spin -- while still letting every loop iteration actually run at least once (wait_cond's `val` is
  only bound inside the loop body; a clock that jumps straight past the timeout on the first check would
  skip the body entirely and crash on an unrelated UnboundLocalError instead of raising the TimeoutError
  under test)."""
  counter = itertools.count(0, 0.05)
  return lambda: next(counter)

class TestNVHaltVerifyBeforeMaster(unittest.TestCase):
  """T4.40c (RCA T4.40_RCA.md Sec5 "M-C" / Sec6 fix 3): _early_ip_init() cleared MASTER for the FLR, then
  unconditionally set it back on a blind 0.1s sleep -- nothing ever verified the *previous* GSP-RM's riscv
  core actually halted, so a core an FLR doesn't halt would be live, bus-mastering, during the ~1-1.5s
  init_sw()/init_hw() preamble that follows (RCA Sec2). The fix reuses ip.py's NV_FLCN.reset() (pure MMIO)
  and verifies NV_PRISCV_RISCV_CPUCTL.active_stat == 0 before MASTER goes back on.

  These drive the REAL NVDev._early_ip_init()/NVDev.__init__ (not copies) against a fake register file that
  models the halt as a genuine consequence of calling reset() (see _EventBar0), not an assumption."""

  def test_master_not_set_before_core_verified_halted(self):
    # Proven failing pre-fix: pre-T4.40c code never reads CPUCTL at all before setting MASTER, so
    # `halted_reads` below would be empty and the first assert would fail.
    events, pci_dev, mmio = _make_fake_hw(wedged=False)
    dev = _FakeIpInitDev(pci_dev, mmio)
    NVDev._early_ip_init(dev)

    master_writes = [i for i,e in enumerate(events) if e[0] == "cfg" and e[1] == pci.PCI_COMMAND and e[2] & pci.PCI_COMMAND_MASTER]
    halted_reads = [i for i,e in enumerate(events) if e[0] == "r" and e[1] == mmio.cpuctl_addr and e[2] == 0]
    assert halted_reads, "fixture/fix bug: the halt-verify step never observed active_stat == 0"
    assert master_writes, "the healthy path must still set MASTER"
    assert min(master_writes) > min(halted_reads), \
      "MASTER must not be set until AFTER the verify step has observed the previous core halted"

  def test_healthy_fake_boots_exactly_as_before_master_ends_set(self):
    # Negative control (the #16536 guard): a core that halts as soon as the code does the right thing about
    # it must still boot straight through _early_ip_init(), ending with MASTER set, same as pre-fix.
    events, pci_dev, mmio = _make_fake_hw(wedged=False)
    dev = _FakeIpInitDev(pci_dev, mmio)
    NVDev._early_ip_init(dev)  # must not raise
    assert pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER, "a healthy boot must end with MASTER set"
    assert isinstance(dev.flcn, NV_FLCN) and dev.flcn.falcon == 0x00110000, "fixture bug: not exercising the NV_FLCN (GA102) path"

  def test_never_halting_core_raises_and_master_stays_clear(self):
    # T4.40_RCA.md Sec5's open worst case: a core neither the FLR nor a falcon engine reset brings down.
    events, pci_dev, mmio = _make_fake_hw(wedged=True)
    dev = _FakeIpInitDev(pci_dev, mmio)
    with patch("time.perf_counter", side_effect=_fast_timeout_clock()):
      with self.assertRaisesRegex(TimeoutError, "did not halt"):
        NVDev._early_ip_init(dev)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), \
      "MASTER must stay clear when the previous core never halts"
    assert not any(e[0] == "cfg" and e[1] == pci.PCI_COMMAND and e[2] & pci.PCI_COMMAND_MASTER for e in events), \
      "MASTER must never have been set at all on this path"

  def test_composes_with_t440b_wrap_master_stays_clear_through_full_init(self):
    # T4.40b wraps the whole NVDev.__init__ body in `try/except BaseException: clear MASTER; raise`. Drive
    # the REAL NVDev.__init__ (not just _early_ip_init) to prove the two fixes compose: no double-clear
    # breakage (the inherited AND-with-~MASTER clear is idempotent whether or not this fix already cleared
    # it), and the exception still propagates to the caller instead of being swallowed.
    events, pci_dev, mmio = _make_fake_hw(wedged=True)
    with patch("time.perf_counter", side_effect=_fast_timeout_clock()):
      with self.assertRaisesRegex(TimeoutError, "did not halt"):
        NVDev(pci_dev)
    assert not (pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER), \
      "MASTER must stay clear after NVDev.__init__ propagates the halt-verify timeout"

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
  device-ownership handshake that follows, not part of this fix) is stubbed out.

  T4.40a extends this fixture: the fix under test shells out to pgrep/ps for liveness detection, so a
  patched Popen call is no longer synonymous with "spawned a server" -- self.procs (below) tracks every
  child for teardown, self.spawned_servers tracks only real [APP_PATH, "server", sock_path] spawns made
  while subprocess.Popen was patched, i.e. spawns the code under test actually made."""

  def setUp(self):
    self.tmpdir = tempfile.mkdtemp(prefix="t425_")
    self.sock_path = os.path.join(self.tmpdir, "tinygpu.sock")
    self.procs:list[subprocess.Popen] = []
    self.spawned_servers:list[subprocess.Popen] = []

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
      if isinstance(p.args, list) and len(p.args) >= 2 and p.args[1] == "server": self.spawned_servers.append(p)
      return p
    return counting

  def _sock_getenv(self, k, d=None): return self.sock_path if k == "APL_REMOTE_SOCK" else d

  def _getenv_with(self, **overrides):
    """Like _sock_getenv, plus fixed values for other keys (T4.40a: PYTEST_XDIST_WORKER / NV_NO_SPAWN)."""
    def fn(k, d=None): return self.sock_path if k == "APL_REMOTE_SOCK" else overrides.get(k, d)
    return fn

  def _pgrep_sees(self, app_path:str, timeout=2.0) -> bool:
    # fork() makes the pid visible immediately but execve() into app_path is not synchronous with Popen()
    # returning -- poll briefly instead of assuming the first pgrep call already sees the post-exec argv.
    deadline = time.time() + timeout
    while time.time() < deadline:
      if subprocess.run(["pgrep", "-f", f"{app_path} server"], stdout=subprocess.PIPE, text=True).stdout.strip(): return True
      time.sleep(0.01)
    return False

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
    n = len(self.spawned_servers)
    assert n == 1, f"6 constructors racing an empty socket dir should spawn exactly ONE server, got {n}"

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

    assert len(self.spawned_servers) == 1
    self.spawned_servers[0].wait(timeout=2)
    assert self.spawned_servers[0].poll() is not None, "a spawn that never connects must be killed, not orphaned"

  def test_live_unresponsive_server_blocks_spawn_and_names_the_situation(self):
    """T4.40a (i): a server that's alive but wedged (e.g. bound to a faulted GSP session, HANDOFF_2026-08-26.md
    section 2) must never get a sibling spawned next to it -- two servers on one device is the precondition
    both 2026-08-26 host panics share. Simulate it directly: a real process matching "<APP_PATH> server",
    spawned BEFORE construction (so it looks exactly like a leftover from an earlier session), that never
    binds the socket -- every connect() attempt genuinely fails."""
    hang = self._script("hang.py", "import time\ntime.sleep(60)\n")  # never binds -> connect() never succeeds
    wedged = subprocess.Popen([hang, "server", self.sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self.procs.append(wedged)
    assert self._pgrep_sees(hang), "fixture bug: pre-spawned fake server never became visible to pgrep"

    class _FakeAPL(APLRemotePCIDevice):
      APP_PATH = hang
      def ensure_app(self): pass

    with patch("subprocess.Popen", side_effect=self._fake_popen()), \
         patch("tinygrad.runtime.support.system.getenv", side_effect=self._sock_getenv), \
         patch("time.sleep", lambda s: None), \
         patch.object(RemotePCIDevice, "__init__", lambda self, *a, **k: None):
      with self.assertRaisesRegex(RuntimeError, "is alive but not accepting connections"):
        _FakeAPL("NV", "0000:00:00.0")

    assert len(self.spawned_servers) == 0, f"a pre-existing live server must block the spawn entirely, got {len(self.spawned_servers)} new spawn(s)"

  def test_xdist_never_spawns_next_to_a_wedged_server_either(self):
    """T4.40a: the xdist/NV_NO_SPAWN branch must also recognize a genuinely wedged server (not just "absent")
    -- this is T4.38's addendum scenario verbatim: a serial pytest run opening NV against a server left over
    from a previously-faulted session."""
    hang = self._script("hang.py", "import time\ntime.sleep(60)\n")
    wedged = subprocess.Popen([hang, "server", self.sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self.procs.append(wedged)
    assert self._pgrep_sees(hang), "fixture bug: pre-spawned fake server never became visible to pgrep"

    class _FakeAPL(APLRemotePCIDevice):
      APP_PATH = hang
      def ensure_app(self): pass

    with patch("subprocess.Popen", side_effect=self._fake_popen()), \
         patch("tinygrad.runtime.support.system.getenv", side_effect=self._getenv_with(PYTEST_XDIST_WORKER="gw0")), \
         patch.object(RemotePCIDevice, "__init__", lambda self, *a, **k: None):
      with self.assertRaisesRegex(RuntimeError, "is alive but not accepting connections"):
        _FakeAPL("NV", "0000:00:00.0")

    assert len(self.spawned_servers) == 0, f"xdist must never spawn next to a wedged server either, got {len(self.spawned_servers)} spawn(s)"

  def test_xdist_or_no_spawn_never_spawns_and_raises_immediately(self):
    """T4.40a (iii): a probe / xdist worker must never spawn, with no server present in either trigger mode."""
    for label, overrides in [("xdist", {"PYTEST_XDIST_WORKER": "gw0"}), ("NV_NO_SPAWN", {"NV_NO_SPAWN": 1})]:
      with self.subTest(label=label):
        class _FakeAPL(APLRemotePCIDevice):
          APP_PATH = "/nonexistent/T440a_test_fake_app"  # never spawned, never pgrep-matched -- must stay that way
          def ensure_app(self): pass

        with patch("subprocess.Popen", side_effect=self._fake_popen()), \
             patch("tinygrad.runtime.support.system.getenv", side_effect=self._getenv_with(**overrides)), \
             patch.object(RemotePCIDevice, "__init__", lambda self, *a, **k: None):
          with self.assertRaisesRegex(RuntimeError, "no TinyGPU server process found"):
            _FakeAPL("NV", "0000:00:00.0")

    assert len(self.spawned_servers) == 0, f"a probe/xdist worker must never spawn, got {len(self.spawned_servers)} spawn(s)"

  def test_xdist_worker_connects_to_a_responsive_server_without_spawning(self):
    """T4.40a (iv): xdist worker + a responsive pre-existing server -> connects fine, spawns nothing."""
    server = self._script("server.py", "import socket, sys, time\n"
                                        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                                        "s.bind(sys.argv[-1])\n"
                                        "s.listen(64)\n"
                                        "time.sleep(30)\n")
    running = subprocess.Popen([server, "server", self.sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self.procs.append(running)
    deadline = time.time() + 2.0
    while True:
      probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      try:
        probe.connect(self.sock_path)
        probe.close()
        break
      except (ConnectionRefusedError, FileNotFoundError):
        probe.close()
        if time.time() > deadline: raise
        time.sleep(0.01)

    class _FakeAPL(APLRemotePCIDevice):
      APP_PATH = server
      def ensure_app(self): pass

    with patch("subprocess.Popen", side_effect=self._fake_popen()), \
         patch("tinygrad.runtime.support.system.getenv", side_effect=self._getenv_with(PYTEST_XDIST_WORKER="gw0")), \
         patch.object(RemotePCIDevice, "__init__", lambda self, *a, **k: None):
      _FakeAPL("NV", "0000:00:00.0")  # must not raise

    assert len(self.spawned_servers) == 0, f"an xdist worker must never spawn even while waiting, got {len(self.spawned_servers)} new spawn(s)"

class _FakeSleepIface:
  """Counts calls to sleep(tm) the way NVSignal._sleep makes them -- a faithful proxy for "one drain", whether
  that's a local stat_q memory read (PCIIface, direct) or one RPC round trip (remote): CountingMMIOInterface
  above already establishes that convention for MMIO reads/writes, this is the same idea for iface.sleep()."""
  def __init__(self): self.calls:list[int] = []
  def sleep(self, tm:int): self.calls.append(tm)

def _fake_nv_signal(iface) -> NVSignal:
  sig = NVSignal.__new__(NVSignal)  # skip HCQSignal.__init__: no real buffer needed, _sleep only reads self.owner
  sig.owner, sig.should_return = SimpleNamespace(iface=iface), False  # should_return=False: keeps __del__ a no-op on this bufferless fake
  return sig

class TestNVSignalSleepFaultDrain(unittest.TestCase):
  """T4.48 F2 (T4.47_RCA.md): NVSignal._sleep used to drain the GSP status queue (owner.iface.sleep) only once
  time_spent_since_last_sleep_ms had accumulated past 200ms. BEAM's device-side per-candidate timeout
  (BEAM_DEV_TIMEOUT) is typically 1-3ms -- far under that gate -- so a fault from an earlier abandoned
  candidate (search.py's beam_search abandons slow candidates by design, T4.48 F1) could sit undrained,
  invisible, for an entire search. Fixed to also drain on the first _sleep() call of every wait, detected as
  "elapsed time just went backwards" since hcq.py's wait() resets its timer at the start of every wait() and
  again on every progress event within one."""
  def test_drains_on_first_sleep_of_a_wait(self):
    # pre-fix: 0 > 200 is False, so the old body never called iface.sleep() at all here.
    iface = _FakeSleepIface()
    _fake_nv_signal(iface)._sleep(0)
    assert iface.calls == [0], f"must drain (a probe, not a sleep -- passes 0) on the first _sleep() of a wait, got {iface.calls}"

  def test_drains_at_most_once_across_a_whole_sub_200ms_wait(self):
    # simulates one wait()'s busy-spin progression that never crosses the 200ms gate: must NOT drain per iteration.
    iface = _FakeSleepIface()
    sig = _fake_nv_signal(iface)
    for elapsed in (0, 1, 2, 5, 10, 50, 100, 150, 199): sig._sleep(elapsed)
    assert len(iface.calls) == 1, f"expected exactly one drain across a whole sub-200ms wait, got {len(iface.calls)}"

  def test_still_drains_past_200ms_unchanged(self):
    iface = _FakeSleepIface()
    sig = _fake_nv_signal(iface)
    sig._sleep(0)    # new: first-call probe
    sig._sleep(250)  # pre-existing: long-wait drain, unchanged
    assert iface.calls == [0, 200], f"first-call probe passes 0, long-wait drain still passes 200 unchanged, got {iface.calls}"

  def test_drains_again_when_a_new_wait_starts(self):
    # a signal (e.g. the device's timeline_signal) is reused across many wait() calls over its life -- each
    # fresh wait (elapsed resetting back toward 0) must get its own first-call drain, not just the first ever.
    iface = _FakeSleepIface()
    sig = _fake_nv_signal(iface)
    for elapsed in (0, 5, 10): sig._sleep(elapsed)  # wait #1, ends without ever hitting 200ms
    assert len(iface.calls) == 1
    for elapsed in (0, 5, 10): sig._sleep(elapsed)  # wait #2, same signal instance (mirrors timeline_signal reuse)
    assert len(iface.calls) == 2, f"a fresh wait must get its own first-call drain, got {iface.calls}"

  def test_noop_without_owner(self):
    sig = NVSignal.__new__(NVSignal)
    sig.owner, sig.should_return = None, False
    sig._sleep(0)  # must not raise

def _fake_hang_self_with_control(is_err_state:bool, rm_control, error_state=None, dispatch_ring=None) -> SimpleNamespace:
  """Like TestNVQuiesceOnFault's _fake_hang_self, plus the is_err_state/error_state/dispatch_ring knobs T4.78's
  on_device_hang forensics change reads. iface intentionally has no `dev_impl` for is_err_state=None (the
  NVKIface/MOCKIface shape); is_err_state True/False models a PCIIface with dev_impl.is_err_state set."""
  pci_dev = _FakePCIConfig()
  iface_kwargs = dict(pci_dev=pci_dev, rm_control=rm_control)
  if is_err_state is not None: iface_kwargs['dev_impl'] = SimpleNamespace(is_err_state=is_err_state)
  fake = SimpleNamespace(iface=SimpleNamespace(**iface_kwargs), debugger=None, debug_channel=0, is_remote=lambda: True, is_nvd=lambda: True)
  if error_state is not None: fake.error_state = error_state
  if dispatch_ring is not None: fake.dispatch_ring = dispatch_ring
  return fake

class TestNVOnDeviceHangForensics(unittest.TestCase):
  """T4.78 (T475_NV_FAULT_RCA.md S5m1): a GSP-RM already known-faulted (is_err_state) has never once answered
  another RPC (5/5 observed) -- pre-fix, on_device_hang's own forensic RPCs burned the default 10s timeout on
  each of up to two calls and then raised a BRAND NEW RuntimeError, discarding synchronize()'s original "Device
  fault detected" exception entirely. These prove: (1) a confirmed-wedged device gets a short forensic timeout;
  (2) a healthy/not-yet-confirmed hang keeps the full timeout (forensics still has a chance to succeed); (3) a
  forensic RPC timeout is caught and reported as one clear line, never left to propagate/stall; (4) the ORIGINAL
  exception (self.error_state -- the same object synchronize() is about to re-raise) survives with its type and
  primary message intact, the forensic report merely appended, never replacing it."""

  def test_wedged_device_gets_short_forensic_timeout(self):
    seen = []
    def rm_control(obj, cmd, params, timeout=10000):
      seen.append(timeout)
      raise RuntimeError("Timeout waiting for RPC response for command 76")
    fake = _fake_hang_self_with_control(is_err_state=True, rm_control=rm_control, error_state=RuntimeError("Device fault detected."))
    NVDevice.on_device_hang(fake)  # must not raise: error_state is present, so it's enriched in place and returns
    assert seen == [2000], f"a confirmed-wedged device must use the short (~2s) forensic timeout, got {seen}"

  def test_healthy_hang_keeps_default_forensic_timeout(self):
    seen = []
    no_fault = SimpleNamespace(mmuFault=SimpleNamespace(valid=False), smErrorStateArray=[])
    def rm_control(obj, cmd, params, timeout=10000):
      seen.append(timeout)
      return no_fault
    fake = _fake_hang_self_with_control(is_err_state=False, rm_control=rm_control, error_state=RuntimeError("some other hang"))
    NVDevice.on_device_hang(fake)
    assert seen == [10000], f"a not-yet-confirmed-wedged hang should keep the full forensic timeout, got {seen}"

  def test_forensic_timeout_reports_one_clear_line_instead_of_propagating(self):
    def rm_control(obj, cmd, params, timeout=10000): raise RuntimeError("Timeout waiting for RPC response for command 76")
    orig = RuntimeError("Device fault detected. NV device is in an unrecoverable fault state.")
    fake = _fake_hang_self_with_control(is_err_state=True, rm_control=rm_control, error_state=orig)
    NVDevice.on_device_hang(fake)  # must not raise the RPC timeout -- it must be caught, not left to stall/propagate
    assert "GSP-RM unresponsive (standard post-fault wedge)" in str(orig)

  def test_original_exception_object_type_and_message_preserved(self):
    def rm_control(obj, cmd, params, timeout=10000): raise RuntimeError("Timeout waiting for RPC response for command 76")
    orig = RuntimeError("Device fault detected. NV device is in an unrecoverable fault state.")
    fake = _fake_hang_self_with_control(is_err_state=True, rm_control=rm_control, error_state=orig)
    NVDevice.on_device_hang(fake)
    assert fake.error_state is orig, "must enrich the SAME exception synchronize() is about to re-raise, not a new one"
    assert type(orig) is RuntimeError
    assert str(orig).startswith("Device fault detected. NV device is in an unrecoverable fault state."), \
      f"primary message must survive unchanged as a prefix, got {orig}"
    assert str(orig) != "Device fault detected. NV device is in an unrecoverable fault state.", "context must be appended, not left off"

  def test_no_original_exception_falls_back_to_a_fresh_one(self):
    # Mirrors TestNVQuiesceOnFault's pre-existing fixture shape (no error_state at all) -- on_device_hang called
    # standalone (not via synchronize()'s except block) must still raise, unchanged from the pre-T4.78 behavior.
    no_fault = SimpleNamespace(mmuFault=SimpleNamespace(valid=False), smErrorStateArray=[])
    fake = _fake_hang_self_with_control(is_err_state=False, rm_control=lambda *a, **k: no_fault)
    with self.assertRaises(RuntimeError) as ctx: NVDevice.on_device_hang(fake)
    assert "fresh client" in str(ctx.exception).lower(), ctx.exception

  def test_dispatch_ring_dumped_before_returning(self):
    dumps = []
    ring = SimpleNamespace(dump=lambda: dumps.append(1))
    no_fault = SimpleNamespace(mmuFault=SimpleNamespace(valid=False), smErrorStateArray=[])
    fake = _fake_hang_self_with_control(is_err_state=False, rm_control=lambda *a, **k: no_fault,
                                         error_state=RuntimeError("Device fault detected."), dispatch_ring=ring)
    NVDevice.on_device_hang(fake)
    assert dumps == [1], "the dispatch ring must be dumped exactly once before the exception propagates"

  def test_no_dispatch_ring_attribute_is_untouched(self):
    # NVKIface/MOCKIface-vintage fakes (like TestNVQuiesceOnFault's) predate dispatch_ring entirely -- must
    # degrade to a no-op, never AttributeError.
    no_fault = SimpleNamespace(mmuFault=SimpleNamespace(valid=False), smErrorStateArray=[])
    fake = _fake_hang_self_with_control(is_err_state=None, rm_control=lambda *a, **k: no_fault)
    with self.assertRaises(RuntimeError): NVDevice.on_device_hang(fake)  # must not raise AttributeError instead

class TestNVDispatchRing(unittest.TestCase):
  """T4.78 (T475_NV_FAULT_RCA.md S5m1): NV_DISPATCH_RING's bookkeeping factored out as a plain class with no
  device access at all, so it's exercisable here without hardware -- the hooks that feed it (NVComputeQueue.exec,
  NVCopyQueue.copy, NVCommandQueue._submit_to_gpfifo) are one-line call sites into this."""

  def test_add_assigns_monotonic_sequence_numbers(self):
    ring = DispatchRing(8)
    for i in range(3): ring.add('kernel', f'k{i}', [(0x1000 + i, 64)])
    assert [e[0] for e in ring.entries] == [0, 1, 2]

  def test_ring_keeps_only_last_n_most_recent_last(self):
    ring = DispatchRing(3)
    for i in range(5): ring.add('kernel', f'k{i}', [])
    # oldest of the 5 (k0, k1) must have fallen off; survivors ordered oldest-first / most-recent-last.
    assert [e[2] for e in ring.entries] == ['k2', 'k3', 'k4']
    assert ring.seq == 5, "the sequence counter must keep counting past the ring's capacity, not wrap/reset"

  def test_drain_appends_staged_entries_and_clears_the_staging_list(self):
    ring = DispatchRing(8)
    staged = [('kernel', 'k0', [(0x1000, 64)]), ('copy', 'NV->NV:1', [(0x2000, 128), (0x3000, 128)])]
    expected = list(staged)  # drain() mutates `staged` in place -- snapshot before, so this compares content, not identity
    ring.drain(staged)
    assert [(e[1], e[2], e[3]) for e in ring.entries] == expected
    assert staged == [], "drain must clear the caller's staging list"

  def test_drain_is_a_noop_on_an_empty_staging_list(self):
    ring = DispatchRing(8)
    ring.drain([])
    assert len(ring.entries) == 0 and ring.seq == 0

  def test_dump_prints_each_entry_most_recent_last(self):
    ring = DispatchRing(2)
    ring.add('kernel', 'first', [(0x1000, 64)])
    ring.add('copy', 'NV->NV:1', [(0x2000, 128)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ring.dump()
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert "first" in lines[0] and "NV->NV:1" in lines[1], f"must print most-recent-last, got {lines}"

class TestNVDispatchRingSubmitChokePoint(unittest.TestCase):
  """T4.78: NVCommandQueue._submit_to_gpfifo is the one choke point shared by compute/copy/video _submit() --
  drives the real method (not a copy) against a minimal fake gpfifo/device to prove it drains any staged
  NV_DISPATCH_RING dispatches into dev.dispatch_ring and clears the staging list either way."""

  def _fake_submit(self, dispatch_ring):
    q = NVCommandQueue()
    q._q = []
    q._dispatches = [('kernel', 'k0', [(0x1000, 64)])]
    q.hw_page = SimpleNamespace(va_addr=0x2000, size=0, base=None)
    # is_remote()/allocator satisfy NVCommandQueue.__del__ (mirrors TestNVBindRemoteBatching's fixture) -- otherwise
    # GC tears this fake down with an AttributeError ignored-in-__del__ warning, unrelated to what's under test here.
    dev = SimpleNamespace(dispatch_ring=dispatch_ring, gpu_mmio={}, synchronize=lambda: None,
                          is_remote=lambda: False, allocator=SimpleNamespace(alloc=lambda *a, **k: None, free=lambda *a, **k: None))
    q.binded_device = dev
    gpfifo = SimpleNamespace(ring={}, put_value=0, entries_count=8, gpput={}, token=0x77)
    return q, dev, gpfifo

  def test_submit_drains_staged_dispatches_into_the_ring(self):
    ring = DispatchRing(8)
    q, dev, gpfifo = self._fake_submit(ring)
    q._submit_to_gpfifo(dev, gpfifo)
    assert [(e[1], e[2]) for e in ring.entries] == [('kernel', 'k0')]
    assert q._dispatches == [], "the queue's staging list must be cleared once drained"

  def test_submit_without_a_ring_still_clears_staging_and_does_not_raise(self):
    q, dev, gpfifo = self._fake_submit(None)  # dispatch_ring=None: NV_DISPATCH_RING was off at device-init time
    q._submit_to_gpfifo(dev, gpfifo)  # must not raise (e.g. AttributeError/NoneType has no attribute 'drain')
    assert q._dispatches == []

class TestNVEagerDrainForensics(unittest.TestCase):
  """T4.78 NV_EAGER_DRAIN: after each dispatch submission, synchronously wait for completion so a fault
  attributes to exactly one dispatch -- implemented via the existing wait/sleep machinery (dev.synchronize(),
  which itself polls iface.sleep() internally), not a new draining mechanism. Drives the real
  NVCommandQueue._submit_to_gpfifo to prove the guard: off (default) never synchronizes -- no behavior change --
  and on synchronizes exactly once per submission."""

  def _fake_submit(self):
    q = NVCommandQueue()
    q._q = []
    q.hw_page = SimpleNamespace(va_addr=0x2000, size=0, base=None)
    sync_calls:list = []
    dev = SimpleNamespace(dispatch_ring=None, gpu_mmio={}, synchronize=lambda: sync_calls.append(1),
                          is_remote=lambda: False, allocator=SimpleNamespace(alloc=lambda *a, **k: None, free=lambda *a, **k: None))
    q.binded_device = dev
    gpfifo = SimpleNamespace(ring={}, put_value=0, entries_count=8, gpput={}, token=0x77)
    return q, dev, gpfifo, sync_calls

  def test_eager_drain_off_never_synchronizes(self):
    q, dev, gpfifo, sync_calls = self._fake_submit()
    with Context(NV_EAGER_DRAIN=0):
      q._submit_to_gpfifo(dev, gpfifo)
    assert sync_calls == [], "NV_EAGER_DRAIN=0 (default) must never synchronize -- no behavior change when off"

  def test_eager_drain_on_synchronizes_once_per_submit(self):
    q, dev, gpfifo, sync_calls = self._fake_submit()
    with Context(NV_EAGER_DRAIN=1):
      q._submit_to_gpfifo(dev, gpfifo)
    assert sync_calls == [1], f"NV_EAGER_DRAIN=1 must synchronize exactly once per submission, got {sync_calls}"

if __name__ == "__main__":
  unittest.main()
