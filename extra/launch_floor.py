"""T4.98e: per-kernel JIT launch-cost floor.

Builds N tiny, unfusable kernels (N independent 32-float buffers, each `b.assign(b + 1.0)`), captures them
in one TinyJit graph, warms it, then times replays -- the per-launch dispatch cost a real serving graph
(hundreds of kernels/decode step) pays regardless of kernel size, isolated from any actual compute.

Usage: PYTHONPATH=. DEV=CPU python extra/launch_floor.py [--n 50,200,600] [--reps 20]
"""
from __future__ import annotations
import argparse, time
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, dtypes

def build_step(n:int):
  bufs = [Tensor.zeros(32, dtype=dtypes.float32).contiguous().realize() for _ in range(n)]
  @TinyJit
  def step() -> None:
    for b in bufs: b.assign(b + 1.0).realize()
  return step

def measure(n:int, reps:int) -> tuple[float, float, int]:
  """Warm an n-kernel jit graph, then time `reps` replays. Returns (ms_per_replay, us_per_launch, kernels_per_replay)."""
  step = build_step(n)
  for _ in range(2): step()  # cnt 0->1 (ignore, eager) then 1->2 (capture + first exec) -- every call from here is a pure replay
  Device[Device.DEFAULT].synchronize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(reps): step()
  Device[Device.DEFAULT].synchronize()
  ms_per_replay = (time.perf_counter() - st) * 1e3 / reps
  us_per_launch = ms_per_replay * 1e3 / n
  return ms_per_replay, us_per_launch, GlobalCounters.kernel_count // reps

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--n", default="50,200,600", help="comma-separated kernel counts to measure (default: %(default)s)")
  parser.add_argument("--reps", type=int, default=20, help="replays timed per N (default: %(default)s)")
  args = parser.parse_args()
  for n in (int(x) for x in args.n.split(",")):
    ms, us, kernels = measure(n, args.reps)
    print(f"N={n:5d}  {ms:9.4f} ms/replay  {us:8.3f} us/launch  kernels/replay={kernels}")
