"""
T4.49 Phase 1.2 -- structural fake proof (GPU-free, runs under the project's established MOCKNV
recipe -- TASKS.md:86 -- real HCQCompiled/HCQProgram/HWQueue code, gpuocelot PTX-emulated engine,
no real GPU needed).

Reproduces, at the raw HWQueue level, exactly what BEAM's abandonment does (search.py:154-160):
submit a candidate whose completion signal is deliberately withheld ("abandoned"/still-"running"),
then -- WITHOUT waiting for it or cancelling it -- submit the next candidate and time it.

Claims under test (see T4.49_NOTES.md Phase 1.1 for the full source chain):
  1. HCQProgram.__call__ always chains a new queue's exec behind a `wait(timeline_signal,
     timeline_value-1)` on the *same* per-device GPFIFO (ops_nv.py NVComputeQueue._submit ->
     dev.compute_gpfifo, one ring, processed strictly in order). So candidate N+1's timestamp/exec
     commands cannot be processed by the GPU before candidate N's completion signal fires --
     independent of whether the HOST waited for N.
  2. On NV, HCQCompiled.can_recover is always False (never set anywhere in ops_nv.py), so
     HCQCompiled.synchronize() ALWAYS discards the caller's `timeout` argument (BEAM's
     early_stop*3 ms) and falls back to HCQDEV_WAIT_TIMEOUT_MS (default 30000ms). BEAM's
     device-timeout abandonment is therefore not a ms-scale, routine event on NV -- it requires a
     candidate to hang ~30s.
  3. Consequence of #1: the abandoned candidate's execution time does NOT leak into the next
     candidate's *measured* duration (sig_en - sig_st) -- refuting the literal contamination
     hypothesis. Consequence of chaining without host sync: the NEXT candidate's own wait can
     itself spuriously time out purely because it is queued behind the still-unresolved abandoned
     one (a "cascade"), burning wall time without corrupting the recorded number.

Run (HCQDEV_WAIT_TIMEOUT_MS must be set BEFORE interpreter start: tinygrad's getenv() is
@functools.cache'd, so mutating os.environ mid-script after the first read is silently inert):
  OCELOT_PATH=<venv>/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX HCQDEV_WAIT_TIMEOUT_MS=300 \
  PYTHONPATH=<this repo> <venv>/bin/python extra/t449/fake_proof_wait_chain.py
"""
import time
from tinygrad import Device
from tinygrad.helpers import getenv

def main():
  d0 = Device["NV"]
  print(f"[setup] iface={type(d0.iface).__name__} can_recover={d0.can_recover} "
        f"HCQDEV_WAIT_TIMEOUT_MS={getenv('HCQDEV_WAIT_TIMEOUT_MS', 30000)}")
  assert d0.can_recover is False, "expected can_recover=False on NV (all interfaces) -- rerun aborted, assumption broken"
  assert getenv("HCQDEV_WAIT_TIMEOUT_MS", 30000) == 300, "set HCQDEV_WAIT_TIMEOUT_MS=300 in the shell env before launching (see docstring)"

  # ---- Claim 2: HCQCompiled.synchronize() ignores the caller's timeout when can_recover=False ----
  # Show synchronize(timeout=50) -- a stand-in for BEAM's early_stop*3 ms -- actually times out on
  # HCQDEV_WAIT_TIMEOUT_MS (300, pinned above), not the 50 we passed in.
  sig = d0.new_signal()
  d0.hw_compute_queue_t().wait(sig, 1).signal(d0.timeline_signal, (tgt:=d0.next_timeline())).submit(d0)
  t0 = time.perf_counter()
  try:
    d0.synchronize(timeout=50)  # BEAM would pass ~early_stop*3, e.g. a handful of ms
    raise SystemExit("FAIL: synchronize() returned without the signal ever firing")
  except RuntimeError as e:
    dt, msg = time.perf_counter() - t0, str(e)
  print(f"[claim 2] synchronize(timeout=50) actually raised after {dt*1e3:.0f} ms (not ~50 ms): {msg}")
  assert dt > 0.2, f"FAIL: raised too fast ({dt*1e3:.0f} ms) -- the passed timeout=50 was NOT ignored as predicted"
  sig.value = 1
  d0.timeline_signal.wait(tgt)  # drain so the timeline is consistent for what follows

  # ---- Claims 1 & 3: candidate A "hangs" (never signals); candidate B is launched right after
  # with no sync/cancellation, exactly as beam_search's except-continue does. ----
  fake_signal = d0.new_signal()  # models A's completion condition; withheld (models a slow/hung kernel)

  qA = d0.hw_compute_queue_t().wait(fake_signal, 1)
  qA.signal(d0.timeline_signal, (a_target:=d0.next_timeline())).submit(d0)
  print(f"[A] submitted candidate A (target={a_target}), gated on fake_signal>=1 (withheld)")

  # host "abandons" A exactly like beam_search: a short wait that times out, then continue --
  # no cancellation, no sync. (Direct HCQSignal.wait with an explicit timeout, modeling what BEAM
  # would get IF can_recover let the short timeout through -- claim 2 already showed NV doesn't.)
  t0 = time.perf_counter()
  try: d0.timeline_signal.wait(a_target, timeout=50)
  except RuntimeError: pass
  print(f"[A] abandoned after {(time.perf_counter()-t0)*1e3:.0f} ms host wait; A's kernel is still 'running' (fake_signal unresolved)")

  # candidate B: launched immediately, chained (wait on timeline_value-1 == a_target) per
  # HCQProgram.__call__'s own pattern. Use timestamp() commands (no real exec needed -- the
  # question is purely about wait/signal/timestamp ordering, which is exec-independent).
  sig_st, sig_en = d0.new_signal(), d0.new_signal()
  t_launch0 = time.perf_counter()
  qB = d0.hw_compute_queue_t().wait(d0.timeline_signal, d0.timeline_value - 1)
  qB.timestamp(sig_st).timestamp(sig_en).signal(d0.timeline_signal, (b_target:=d0.next_timeline())).submit(d0)
  t_launch1 = time.perf_counter()
  print(f"[B] submitted candidate B (target={b_target}) immediately after abandoning A: submit took {(t_launch1-t_launch0)*1e3:.2f} ms (non-blocking)")

  # B must NOT have executed yet: its own wait targets a_target's successor, and the GPFIFO is FIFO,
  # so B's timestamp commands cannot have been processed while A is still gating the queue.
  assert sig_st.value == 0 and sig_en.value == 0, "FAIL: B's timestamps fired before A resolved -- overlap!"
  print("[B] confirmed: sig_st/sig_en still unset (0) -- B has not executed while A is still pending")

  # B's OWN wait, issued right after (models the next _time_program call's dev.synchronize) --
  # cascade check: does B spuriously time out purely because it's queued behind A?
  t0 = time.perf_counter()
  try:
    d0.timeline_signal.wait(b_target, timeout=50)
    raise SystemExit("FAIL: B's wait succeeded while A was still withheld -- chaining assumption wrong")
  except RuntimeError:
    dt_cascade = time.perf_counter() - t0
  print(f"[cascade] B's own wait ALSO timed out after {dt_cascade*1e3:.0f} ms, though B's kernel never ran -- "
        f"a healthy candidate queued right after a hung one is at risk of the same false 'Wait timeout'")

  # Now let A finally "finish" (simulates the hung/slow kernel eventually completing) and drain.
  t_resolve = time.perf_counter()
  fake_signal.value = 1
  d0.timeline_signal.wait(b_target, timeout=5000)  # generous -- proves it drains once A is free, not stuck forever
  t_drain = time.perf_counter()
  print(f"[drain] once A resolved, B drained {(t_drain - t_resolve)*1e3:.1f} ms later "
        f"(total wall cost since B was submitted: {(t_drain - t_launch0)*1e3:.1f} ms)")

  # The measured value: B's own duration must be tiny (near-zero -- no real exec, just two
  # back-to-back timestamps) and NOT include A's hang. This is the actual "does it contaminate the
  # NUMBER" question.
  b_dur_us = float(sig_en.timestamp - sig_st.timestamp)
  print(f"[claim 3] B's measured duration (sig_en - sig_st) = {b_dur_us:.2f} us -- "
        f"does NOT include A's withheld runtime (A hung for {(t_resolve - t_launch0)*1e3:.0f} ms of wall time)")
  assert b_dur_us < 50_000, f"FAIL: B's measured duration ({b_dur_us} us) looks contaminated by A's hang"

  print("\nVERDICT: no overlap, no measured-value contamination (claims 1/3 hold); can_recover=False makes "
        "BEAM_DEV_TIMEOUT a no-op on NV at ms-scale (claim 2 holds); a genuinely hung candidate DOES risk "
        "cascading false 'Wait timeout' drops on immediately-following candidates, at real wall-clock cost, "
        "without corrupting any recorded timing number.")

if __name__ == "__main__":
  main()
