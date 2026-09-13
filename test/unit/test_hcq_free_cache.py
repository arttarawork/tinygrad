# T4.112: HCQAllocatorBase.free_cache synchronizes each device once, not once per cached buffer (an OOM flush of hundreds of
# cached buffers on the 3090 cost one device round trip each -- 41 s per state-cache snapshot at the 262144-token window).
import unittest
from tinygrad.runtime.support.hcq import HCQAllocatorBase

class _Dev:
  def __init__(self): self.syncs = 0
  def synchronize(self): self.syncs += 1

class _Buf:
  def __init__(self, dev):
    self.mapped_devs, self.mappings = [dev], {}

class _Alloc(HCQAllocatorBase):
  def __init__(self, dev):
    self.dev, self.freed = dev, 0
    super().__init__(dev, batch_cnt=0)  # no copy buffers: _alloc is never needed
  def _alloc(self, size, options): return _Buf(self.dev)
  def _do_free(self, buf, options): self.freed += 1

class TestFreeCacheSyncsOnce(unittest.TestCase):
  def test_one_sync_per_device_for_many_cached_buffers(self):
    dev = _Dev()
    a = _Alloc(dev)
    for i in range(50): a.cache[(1024 * (i + 1), None)].append(_Buf(dev))
    a.free_cache()
    self.assertEqual(a.freed, 50)
    self.assertEqual(dev.syncs, 1)
    self.assertFalse(a._synced_for_free_cache)
  def test_plain_free_still_syncs_per_buffer(self):
    dev = _Dev()
    a = _Alloc(dev)
    a._free(_Buf(dev))
    a._free(_Buf(dev))
    self.assertEqual(dev.syncs, 2)
  def test_empty_cache_is_a_no_op(self):
    dev = _Dev()
    a = _Alloc(dev)
    a.free_cache()
    self.assertEqual((dev.syncs, a.freed), (0, 0))

if __name__ == "__main__": unittest.main()
