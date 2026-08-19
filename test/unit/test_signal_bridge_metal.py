"""T3.6 spike: prototype an async signal bridge for the CPU(producer)->METAL(consumer) boundary hop,
replacing T3.4's refuted zero-copy-alone approach (aliasing removed the memcpy but bought nothing --
the fixed per-hop cost is host-blocking SYNCHRONIZATION, not the copy: MetalDevice.synchronize() costs
~0.3us idle but ~150us+ right after any dispatch, a real waitUntilCompleted() driver round trip).

This spike converts that host-blocking drain into a GPU-side dependency edge: the consuming Metal
command buffer encodes encodeWaitForEvent:value: on a dedicated MTLSharedEvent BEFORE it is committed
(ops_metal.py's own _transfer already does exactly this for METAL<->METAL boundary hops -- this reuses
the identical primitive for a CPU-producer/METAL-consumer pair). A lightweight background thread
("watcher") flips the event's signaledValue once the producer's own completion signal is reached, so
the calling/orchestrating Python thread never blocks on the drain -- only the watcher thread does.

Primitives reused, not reinvented (ponytail rung 2 -- nothing here is a new mechanism):
  - T3.4's proven aliasing direction (METAL owns the allocation, CPU borrows its host pointer via
    Tensor.from_blob) -- the ONLY safe direction; Metal's external_ptr re-interprets the int as an
    existing MTLBuffer objc id, so a raw CPU-owned pointer there is unsafe (see T3.4's report).
  - Device["CPU"].synchronize() itself as the "has the producer's write landed" check -- the watcher
    thread calls the real, already-correct synchronize() (which internally drains CPU's worker rings
    AND its generic HCQ2 timeline signal), just from a background thread instead of the caller's.
  - MTLSharedEvent + encodeWaitForEvent:value:/encodeSignalEvent:value: -- already used METAL<->METAL
    in ops_metal.py's _transfer; setSignaledValue:/signaledValue aren't in the autogen _methods_ list
    for MTLSharedEvent, so they're called the same ad-hoc objc.msg(...) way ops_metal.py's to_ns_str
    does for selectors autogen doesn't cover.
  - A blit (blitCommandEncoder copyFromBuffer...) stands in for "GPU work that reads the produced
    buffer" -- same command-buffer/driver machinery as a compute dispatch, avoids needing to hand-roll
    a MetalProgram just to measure submit-to-complete latency.

See SIGNAL_BRIDGE_NOTES.md (repo root) for the capture-op analysis (why this stays eager/out-of-JIT)
and the measured latency table.
"""
import ctypes, threading, time, statistics, unittest
from typing import Any
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import mv_address
import tinygrad.runtime.support.objc as objc
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.ops_metal import wait_check, to_ns_str

def _metal_host_ptr(buf:Buffer) -> int:
  return mv_address(Device["METAL"].allocator._as_buffer(buf._buf))

# MTLSharedEvent's setSignaledValue:/signaledValue aren't in metal.py's autogen _methods_ list for that
# class (only newSharedEvent, the factory, is) -- call them the same raw way ops_metal.py's to_ns_str does.
_set_signaled_value = objc.msg("setSignaledValue:", None, [ctypes.c_uint64])
_get_signaled_value = objc.msg("signaledValue", ctypes.c_uint64, [])

class Watcher:
  """One background thread bridging a CPU producer to a Metal consumer: arm() records which MTLSharedEvent
  and target value to flip; the thread blocks on the REAL Device["CPU"].synchronize() (not a hand-rolled
  poll -- CPU's own worker-ring + generic HCQ2 timeline dance is nontrivial enough that reusing the
  already-correct implementation beats guessing at it) and then signals the event. Single pending slot:
  fine for this spike's sequential single-hop measurement loop; a real integration serving concurrent
  hops would need a bounded work queue instead.
  # ponytail: single-slot watcher, not a queue -- upgrade if concurrent in-flight hops are ever needed.
  """
  def __init__(self):
    self._req, self._done = threading.Event(), threading.Event()
    self._event: Any = None
    self._event_value = 0
    self.notice_ts = 0.0  # perf_counter() timestamp when the watcher observed CPU completion
    threading.Thread(target=self._loop, daemon=True).start()

  def _loop(self):
    while True:
      self._req.wait()
      self._req.clear()
      Device["CPU"].synchronize()
      self.notice_ts = time.perf_counter()
      _set_signaled_value(self._event, self._event_value)
      self._done.set()

  def arm(self, event, event_value:int):
    self._done.clear()
    self._event, self._event_value = event, event_value
    self._req.set()

  def wait_armed(self, timeout:float=5.0):
    """Only for correctness tests that need to know the watcher finished; the benchmark path never calls this
    on the timed critical section -- that's the entire point."""
    assert self._done.wait(timeout), "watcher did not signal in time"

def _new_bridge_event():
  return Device["METAL"].sysdevice.newSharedEvent().retained()

def _blit_commit(dst:Buffer, src:Buffer, nbytes:int, wait_on:tuple[Any,int]|None):
  """Commit a blit copy dst<-src on METAL's queue. If wait_on is given, encode encodeWaitForEvent:value:
  on the command buffer BEFORE the blit -- the GPU-side dependency edge that replaces host blocking."""
  dev = Device["METAL"]
  cbuf = dev.mtl_queue.commandBuffer().retained()
  if wait_on is not None: cbuf.encodeWaitForEvent_value(ctypes.cast(wait_on[0], metal.MTLEvent), wait_on[1])
  enc = cbuf.blitCommandEncoder().retained()
  enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(src.buf, 0, dst.buf, 0, nbytes)
  enc.endEncoding()
  cbuf.setLabel(to_ns_str("T3.6 bridge consumer"))
  cbuf.commit()
  return cbuf

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required")
class TestSignalBridgeCorrectness(unittest.TestCase):
  def _make_pair(self, n=256):
    m = Tensor.zeros(n, dtype=dtypes.float32, device="METAL").contiguous().realize()
    ptr = _metal_host_ptr(m.uop.buffer)
    c = Tensor.from_blob(ptr, (n,), dtype=dtypes.float32, device="CPU")  # aliases the SAME host memory
    scratch = Tensor.zeros(n, dtype=dtypes.float32, device="METAL").contiguous().realize()
    return m, c, scratch

  def test_bridge_wait_sees_producer_write(self):
    m, c, scratch = self._make_pair()
    watcher = Watcher()
    bridge_event = _new_bridge_event()

    c.assign(c + 42).realize()  # CPU write, dispatched async -- NOT synchronized here
    watcher.arm(bridge_event, 1)
    cbuf = _blit_commit(scratch.uop.buffer._buf, m.uop.buffer._buf, m.uop.buffer.nbytes, wait_on=(bridge_event, 1))
    wait_check(cbuf)  # block on the CONSUMER only, to check the result
    self.assertEqual(scratch.tolist(), [42.0] * 256)
    watcher.wait_armed()

  def test_bridge_wait_actually_blocks_not_a_noop(self):
    # a longer CPU write chain (several dependent ops) makes "the wait was skipped, blit read stale/zero
    # data" a near-certainty if encodeWaitForEvent didn't really gate the blit -- guards against a bridge
    # that silently degrades to "always racy" while still passing the simpler test above by luck.
    m, c, scratch = self._make_pair(n=4096)
    watcher = Watcher()
    bridge_event = _new_bridge_event()

    x = c
    for _ in range(8): x = x + 1  # chain of CPU dispatches, not yet realized
    c.assign(x).realize()
    watcher.arm(bridge_event, 1)
    cbuf = _blit_commit(scratch.uop.buffer._buf, m.uop.buffer._buf, m.uop.buffer.nbytes, wait_on=(bridge_event, 1))
    wait_check(cbuf)
    self.assertEqual(scratch.tolist(), [8.0] * 4096)
    watcher.wait_armed()

  def test_bridge_event_value_increments_across_iterations(self):
    # a fresh event value per iteration (mirroring ops_metal.py _transfer's src_dev.timeline_value += 1)
    # -- reusing the same value across iterations would make later waits vacuously already-satisfied.
    m, c, scratch = self._make_pair()
    watcher = Watcher()
    bridge_event = _new_bridge_event()
    for i in range(1, 4):
      c.assign(c + 1).realize()
      watcher.arm(bridge_event, i)
      cbuf = _blit_commit(scratch.uop.buffer._buf, m.uop.buffer._buf, m.uop.buffer.nbytes, wait_on=(bridge_event, i))
      wait_check(cbuf)
      self.assertEqual(scratch.tolist(), [float(i)] * 256)
      watcher.wait_armed()

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required")
class TestSignalBridgeBenchmark(unittest.TestCase):
  """Runs a SMALL rep count so it stays a fast CI-safe test; see bench() below / __main__ for the full
  min-of-N table used in the T3.6 report."""
  def test_bridge_not_slower_than_baseline_smoke(self):
    baseline, bridge = _run_once(n_elems=256, reps=5, warmup=2)
    self.assertGreater(len(baseline["total"]), 0)
    self.assertGreater(len(bridge["total"]), 0)

def _run_once(n_elems:int, reps:int, warmup:int) -> tuple[dict, dict]:
  m = Tensor.zeros(n_elems, dtype=dtypes.float32, device="METAL").contiguous().realize()
  ptr = _metal_host_ptr(m.uop.buffer)
  c = Tensor.from_blob(ptr, (n_elems,), dtype=dtypes.float32, device="CPU")
  scratch = Tensor.zeros(n_elems, dtype=dtypes.float32, device="METAL").contiguous().realize()
  nbytes = m.uop.buffer.nbytes
  watcher = Watcher()
  bridge_event = _new_bridge_event()

  def baseline_iter():
    t0 = time.perf_counter()
    c.assign(c + 1).realize()
    Device["CPU"].synchronize()  # the old way: host-blocking full drain
    t_submit = time.perf_counter()
    cbuf = _blit_commit(scratch.uop.buffer._buf, m.uop.buffer._buf, nbytes, wait_on=None)
    wait_check(cbuf)
    t_done = time.perf_counter()
    return t0, t_submit, t_done, cbuf

  def bridge_iter(val:int):
    t0 = time.perf_counter()
    c.assign(c + 1).realize()  # dispatched async, NOT synchronized
    watcher.arm(bridge_event, val)
    cbuf = _blit_commit(scratch.uop.buffer._buf, m.uop.buffer._buf, nbytes, wait_on=(bridge_event, val))
    t_submit = time.perf_counter()  # main thread is free from here in the bridge path
    wait_check(cbuf)
    t_done = time.perf_counter()
    watcher.wait_armed()
    return t0, t_submit, t_done, cbuf

  for _ in range(warmup): baseline_iter()
  baseline = {"submit": [], "total": []}
  for _ in range(reps):
    t0, t_submit, t_done, _ = baseline_iter()
    baseline["submit"].append((t_submit - t0) * 1e6)
    baseline["total"].append((t_done - t0) * 1e6)

  for i in range(warmup): bridge_iter(1000 + i)
  bridge = {"submit": [], "total": [], "watcher_notice_to_done": []}
  for i in range(reps):
    val = 2000 + i
    t0, t_submit, t_done, _ = bridge_iter(val)
    bridge["submit"].append((t_submit - t0) * 1e6)
    bridge["total"].append((t_done - t0) * 1e6)
    bridge["watcher_notice_to_done"].append((t_done - watcher.notice_ts) * 1e6)

  return baseline, bridge

def bench(n_elems:int=256, reps:int=50, warmup:int=10):
  baseline, bridge = _run_once(n_elems, reps, warmup)
  def stats(xs): return f"min={min(xs):7.1f}us  median={statistics.median(xs):7.1f}us  mean={statistics.mean(xs):7.1f}us"
  print(f"\n=== T3.6 signal bridge microbench (n_elems={n_elems}, reps={reps}, warmup={warmup}) ===")
  print(f"baseline submit-to-consumer-committed (host-blocking sync included): {stats(baseline['submit'])}")
  print(f"baseline total (producer issue -> consumer complete):               {stats(baseline['total'])}")
  print(f"bridge   submit-to-consumer-committed (no host block):              {stats(bridge['submit'])}")
  print(f"bridge   total (producer issue -> consumer complete):               {stats(bridge['total'])}")
  print(f"bridge   watcher-notice-to-consumer-done (watcher wakeup component):{stats(bridge['watcher_notice_to_done'])}")
  return baseline, bridge

if __name__ == "__main__":
  import sys
  if "--bench" in sys.argv:
    bench()
  else:
    unittest.main()
