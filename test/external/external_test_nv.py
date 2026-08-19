import os, struct, unittest
from tinygrad.helpers import getenv
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry
from tinygrad.runtime.support.am.amdev import AMPageTableEntry

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

if __name__ == "__main__":
  unittest.main()
