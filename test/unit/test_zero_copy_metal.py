"""T3.4 spike: prove the METAL<->CPU zero-copy aliasing primitive in isolation, outside the llm pipeline.

Established semantics (full writeup in the task report):
  - Metal buffers are always allocated MTLResourceStorageModeShared (MetalAllocator._alloc, ops_metal.py) --
    ordinary host-visible unified memory on Apple Silicon. `.contents()` gives a stable, process-lifetime
    host pointer for the buffer.
  - CPU's BufferSpec.external_ptr genuinely wraps a raw host address (CPUAllocator._alloc, ops_cpu.py:179).
    Metal's external_ptr does NOT: MetalAllocator._alloc (ops_metal.py:157) does
    `metal.MTLBuffer(options.external_ptr)`, and MTLBuffer is an objc.Spec/id_ subclass (autogen/metal.py:25,
    support/objc.py) -- it re-interprets the int as an existing Objective-C `id<MTLBuffer>` and message-sends
    on it later (.contents() in _as_buffer, .release() in _free). Handing it a raw CPU-owned pointer (from
    mmap/malloc, no valid isa) would message-send on non-object memory -- undefined behaviour / a hard crash,
    not a Python exception. NOT reproduced here on purpose (it would take the whole process down); this is a
    static-analysis-only finding, documented instead of demonstrated.
  - So aliasing only works one direction: METAL owns the allocation, CPU borrows its pointer. Allocate
    normally on METAL, read the host pointer via the allocator's _as_buffer(), and wrap that raw address as a
    CPU tensor via Tensor.from_blob (which uses CPU's genuine external_ptr=raw-address path -- no
    message-sending involved on either side).
  - Coherency is real cache-coherent unified memory (no didModifyRange:/.Managed-storage dance -- that API is
    for Intel Macs' discrete-VRAM path and was never used here). ORDERING still needs an explicit wait: a
    write dispatched through a device's own async execution model (a Metal command buffer, or CPU's HCQ2
    worker ring, ops_cpu.py) is only guaranteed visible to the other side after that *writer's*
    Device.synchronize() -- not the reader's. Both tests below establish this directly.
"""
import ctypes, unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import mv_address

def _metal_host_ptr(buf:Buffer) -> int:
  return mv_address(Device["METAL"].allocator._as_buffer(buf._buf))

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required")
class TestZeroCopyPrimitive(unittest.TestCase):
  def test_raw_pointer_aliasing_is_the_same_memory(self):
    # no tinygrad kernels involved at all -- just prove the host pointer behind a Metal shared-storage buffer
    # is a stable, ordinary address that ctypes can read/write directly, independent of any Tensor machinery.
    buf = Buffer("METAL", 64, dtypes.uint8, preallocate=True)
    ptr = _metal_host_ptr(buf)
    ctypes.memmove(ptr, bytes(range(64)), 64)
    ptr2 = _metal_host_ptr(buf)  # re-derived from scratch (fresh _as_buffer() call)
    self.assertEqual(ptr, ptr2)  # stable for the buffer's lifetime
    self.assertEqual(bytes((ctypes.c_uint8 * 64).from_address(ptr2)), bytes(range(64)))
    buf.deallocate()

  def test_metal_write_visible_to_cpu_alias_after_metal_sync(self):
    m = Tensor.zeros(256, dtype=dtypes.float32, device="METAL").contiguous().realize()
    ptr = _metal_host_ptr(m.uop.buffer)
    c = Tensor.from_blob(ptr, (256,), dtype=dtypes.float32, device="CPU")  # aliases the SAME host memory

    m.assign(m + 1).realize()      # dispatches a Metal kernel; realize() does not block on completion
    Device["METAL"].synchronize()  # the writer's device must drain before the reader (CPU alias) can trust it
    self.assertEqual(c.tolist(), [1.0] * 256)

    m.assign(m + 41).realize()     # second write, over the JIT-free eager path -- same buffer, same address
    Device["METAL"].synchronize()
    self.assertEqual(c.tolist(), [42.0] * 256)

  def test_cpu_alias_write_visible_to_metal_after_cpu_sync(self):
    m = Tensor.zeros(256, dtype=dtypes.float32, device="METAL").contiguous().realize()
    ptr = _metal_host_ptr(m.uop.buffer)
    c = Tensor.from_blob(ptr, (256,), dtype=dtypes.float32, device="CPU")

    c.assign(c + 7).realize()    # dispatches through CPU's own HCQ2 async worker ring (ops_cpu.py), not sync
    Device["CPU"].synchronize()  # required: the WRITER's (CPU) sync gates visibility, not the reader's
    self.assertEqual(m.tolist(), [7.0] * 256)  # read straight off the METAL tensor, not through the alias

if __name__ == "__main__":
  unittest.main()
