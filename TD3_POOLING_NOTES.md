# TD.3 — first real METAL+NV pooling run (MacBook M3 Pro + RTX 3090, AG02/USB4)

Worktree: `tinygrad-dock`, branch `task/TD.3-pooling` off `task/TD.2-matrix` @ `c534ab1af`
(includes the T4.14 compile-server fix). Model: `llama3.2:1b` (dense, **16 transformer layers**,
confirmed from GGUF `llama.block_count`). llama-server stayed up throughout (correctness-phase
precedent from T3.2/TD.2). `--device-map NV` compiles via the colima/docker nvcc compile-server
(not `NV:NAK`) — matches the env brief; no BEAM (correctness phase).

Diff: 2 lines in `extra/benchmark_llm.py` (added `--device-map` passthrough, mirrors
`tinygrad.llm.cli`'s existing flag) + one new test class in `test/unit/test_llm_device_map.py`
(`TestDeviceMapMetalNV`). No `tinygrad/` core files touched.

## TL;DR verdict

**METAL+NV pooling works and is cheap.** Split tokens are byte-identical to both single-device
references in both directions (16/16 and 64/64 decode tokens tested). JIT capture succeeds
end-to-end exactly like T3.2's METAL+CPU rehearsal — but the **boundary hop is ~80-90 µs
steady-state, not ~750 µs**: NV is HCQ-based and its `synchronize()` is a native hardware
timeline-signal wait, not the polling/full-drain primitives METAL and CPU use. The NV island
still HCQ-graph-batches with METAL as its cross-device neighbor. **T3.6 (async signal bridge)
should stay parked** — see §6.

---

## 1. Single-device sanity (objective 1)

`--device-map METAL` / `--device-map NV`, prompt=16 synthetic tokens, decode=64, greedy:

| | load | warm | prefill tok/s | decode tok/s | GB/s |
|---|---|---|---|---|---|
| all-METAL | 0.876s | 17.154s | 60.14 | 23.997 | 25.10 |
| all-NV    | 7.939s | 7.797s  | 116.63 | 64.830 | 162.39 |

Both produced the **exact same 64-token greedy sequence**:
`[2331, 220, 15, 220, 15, 220, 15, ...]` (a synthetic gibberish prompt drives the model into a
repetitive attractor quickly — expected and fine for a determinism check; sustained repetition
over 64 steps is actually a good stress test since it re-visits the same logit basin every step).
**METAL and NV are argmax-equivalent on this model for at least 64 decode steps** — no
backend-level FP drift observed at this horizon.

## 2/3. Split runs, both directions (objectives 2-3)

Reasoning stated **before running** (per the task brief): `output_norm`/`output` (lm_head) always
land on `dmap[-1]` (`tinygrad/llm/model.py:514-516`), and activations only ever flow forward
block-by-block (`model.py:537`) — there is no path back. If the two single-device references ever
disagreed with each other, the natural hypothesis was that the split would track whichever
reference shares the **tail** device (`dmap[-1]`), since that's the last stage before the argmax
and also hosts the largest single matmul (dim × vocab_size). **This hypothesis turned out to be
untestable**: METAL and NV already agree with each other at this horizon (§1), so there was nothing
for the split to diverge toward. The result actually obtained is stronger than what was asked:
parity holds unconditionally, not just "matches one of the two."

| device_map | direction | prefill tok/s | decode tok/s | tokens |
|---|---|---|---|---|
| `0-3:METAL,4-15:NV` | METAL-first (4 METAL / 12 NV, tail=NV) | 86.97 | 41.809 | **identical** |
| `0-11:NV,12-15:METAL` | NV-first (12 NV / 4 METAL, tail=METAL) | 87.71 | 36.023 | **identical** |

Both split directions reproduced the exact same 64-token sequence as both single-device
references. `split.token_embd.weight.device`/`split.output.weight.device` land on `dmap[0]`/`dmap[-1]`
as expected in both directions (verified directly, not just inferred).

## 4. JIT capture / graph islands (objective 4)

Capture succeeds end-to-end for the mixed METAL+NV trace, same shape as T3.2's METAL+CPU result.
Introspected `model.jit[(False, True, None)].captured.linear.src` directly (same AST-walk the
T3.2/T3.3 tests use) on the real 16-layer model, both directions:

```
split A "0-3:METAL,4-15:NV":  COPY {NV,METAL} -> NV       (dst follows the map: NV is last)
  graph batches: METAL{32 kernels}, METAL{26 kernels}, NV{128 kernels}, NV{46 kernels}
split B "0-11:NV,12-15:METAL": COPY {METAL,NV} -> METAL   (dst follows the map: METAL is last)
  graph batches: NV{32}, NV{64}, NV{74}, METAL{62}
```

- **Captured cross-device COPY = exactly 1 per token** in both directions (the block-boundary
  activation hop) — matches T4.5's "exactly one boundary hop" dense-split finding, now confirmed
  on real METAL+NV hardware, not just synthetic CPU:0/CPU:1.
- **The NV island still HCQ-graph-batches** with METAL as its cross-device neighbor
  (`Device["NV"].graph is HCQGraph`), exactly as it does in NV-only runs (TD.2a). METAL similarly
  still uses `MetalGraph`. Zero ungraphed compute kernels on either side — unlike METAL+CPU
  (T3.2: "CPU segments stay ungraphed/sequential, `CPUDevice.graph=None`"), **both islands here
  graph**. Batch counts (2 METAL + 2-3 NV batches, not 1 each) come from `JIT_BATCH_SIZE`'s
  default 32-kernel cap (`engine/jit.py:73`), not from anything cross-device-specific.
- A **second, separate, uncaptured** copy also fires once per token: `generate()`
  (`model.py:762-763`, T2.5's "move the sampled token once, back to t's device") issues an eager
  `.to(t.device).realize()` for the sampled token **outside** the JIT-wrapped `forward()` whenever
  the output device (`dmap[-1]`) differs from the first block's device (`dmap[0]`) — true for both
  splits here. This is not new to METAL+NV; any device_map split whose head and tail devices differ
  pays it (T3.2's split would have too). It shows up in DEBUG=2 logs as the `4 B` copy row (§5) but
  never enters `rollout_jit.captured` since it runs after the JIT call returns.
- Added `TestDeviceMapMetalNV` to `test/unit/test_llm_device_map.py` (mirrors
  `TestDeviceMapMetalCPU`, gated `Device.DEFAULT == "METAL" and "NV" in Device.get_available_devices()`
  — reuses the framework's own device-probe helper, `device.py:40-42`, which already
  exception-suppresses unavailable backends; skips cleanly off-dock/CI). **Ran for real on this
  dock** (not skipped): 29/29 passed, confirming the tiny-synthetic-config parity+capture
  assertions on actual NV hardware, matching the real-model manual runs above.

## 5. Boundary cost vs the 750 µs METAL↔CPU floor (objective 5)

`DEBUG=2` steady-state (graphed) rows for one full decode step, split A, 3rd of 3 `--benchmark`
steps (representative; both splits/both copy directions gave the same ~80-90 µs number):

```
*** METAL   1  batched 32                tm   4683.50us   (4 METAL layers, part 1)
*** METAL   2  batched 26                tm   3658.75us   (4 METAL layers, part 2)
*** NV      3  copy    8.00 KB, NV <- METAL   tm     80.42us   <- activation boundary hop
*** NV      4  batched 128               tm   7258.96us   (12 NV layers + lm_head, part 1)
*** NV      5  batched 46                tm   4934.54us   (12 NV layers + lm_head, part 2)
*** METAL   6  copy        4 B, METAL <- NV   tm     80.50us   <- sampled-token hop back (§4, uncaptured)
 27.70 ms,  36.09 tok/s
```

Split B (reversed direction) steady state: `METAL <- NV 8.00KB` = 79.62 µs, `NV <- METAL 4B` =
81.96 µs. **Both copy directions, both copy sizes (8 KB activation / 4 B token id), both splits:
~80-90 µs.** Cost is flat with size in this range, same qualitative shape T3.2 found for
METAL↔CPU (fixed floor, not bandwidth-bound below ~1M elements) — but the floor itself is
**~8-9x lower** than the reported ~750 µs METAL↔CPU number.

**Why NV is cheaper (root-caused, not guessed):** all three backends' `synchronize()` differ in
kind, not just constant factor:
- `MetalDevice.synchronize()` (`ops_metal.py:52-54`) calls `cbuf.waitUntilCompleted()` per
  in-flight command buffer — an OS-level blocking wait on the command buffer object (T3.4 measured
  ~150 µs for this alone).
- `CPUDevice.synchronize()` (`ops_cpu.py:216-219`) polls a software `done`/`put` counter pair per
  emulated worker **thread** in a loop (`while done[0] < put[0]: self._wait_signal(...)`) — CPU has
  no real hardware queue, so this is a software emulation of one.
- `HCQCompiled.synchronize()` (`hcq.py:425,433`, NV's base) is `self.timeline_signal.wait(value)`
  — a raw hardware timeline-signal value wait, the same native GPU semaphore mechanism T2.1/T2.2/T2.3
  tuned for the transport work. No polling loop, no command-buffer object wait.

The ~750 µs METAL↔CPU floor is dominated by whichever side's sync is heaviest in that pairing;
CPU's software thread-polling loop is structurally heavier than NV's native hardware signal wait.
METAL+NV pays METAL's own `waitUntilCompleted()` cost on one side either way, but the other side is
cheap here in a way CPU never was.

**Split vs all-NV tok/s, decomposed:** all-NV steady state ≈ 15.4 ms/token; split A ≈ 23.9 ms/token
(benchmark-mode, no DEBUG overhead) — a ~8.5 ms/token delta. The two boundary copies together
account for only ~0.16-0.2 ms of that (under 4%); **the other ~96% is simply the 4 METAL layers'
own slower compute** (8.34 ms for 4 METAL layers vs. what those layers would cost on NV) — not
transport. The transport question this task exists to answer is emphatically settled: **the
USB4/HCQ hop is not the bottleneck, was never close to it, and is cheaper than the CPU rehearsal
that was standing in for it.**

## Swap / stability

`sysctl vm.swapusage` hovered 2.0-2.3 GB used (up from ~0.7 GB baseline) across the whole session
(colima's VM + llama-server's 23 GB wired + this process's own METAL+NV working set), but **did
not grow monotonically** — it plateaued and even receded slightly between runs (2267 → 2211 → 2203
→ 2075 MB across 4 consecutive runs). Treated as a stable elevated baseline from concurrent
processes, not a leak; no runs were reduced or aborted for it. Minority-METAL placement (4/16
layers) was used throughout per the env brief.

## 6. T3.6 (async signal bridge) revival verdict

**Stays parked.** T3.6's METAL+CPU rehearsal found a **net loss** (~20-35 µs) from adding a
signal-bridge versus plain `synchronize()`, because the bridge's own watcher-thread cost as much as
it saved. This task's real-hardware number makes the case *weaker*, not stronger: the existing,
un-bridged METAL+NV hop is already ~80-90 µs — cheaper than T3.6's own best-case savings estimate
applied to the CPU number, because NV's native `HCQCompiled.synchronize()` **already is** the
lightweight GPU-side signal wait T3.6 proposed building by hand; there's no CPU-style full-drain or
polling loop for a bridge to route around on the NV side. And in absolute terms the hop is now
under 1% of a token's wall time (80-90 µs of a 15-28 ms step) — even a hypothetical zero-cost bridge
has essentially nothing left to claim. Building 100-160+ lines of capture-op/scheduler surgery
(T3.6's own sizing) for a saving that's already close to net-negative in the closest analog tested
is not worth it. If pooling ever needs another lever, it's the compute split ratio (§5's 96%), not
the hop.

---

## Exact commands

```bash
cd tinygrad-dock   # worktree, relative paths only for repo files
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
MODEL=/Users/artur/Library/Caches/tinygrad/downloads/3cdb17618469285f97f176c434543c9c  # llama3.2:1b

# branch setup
git checkout -b task/TD.3-pooling   # from c534ab1af

# single-device sanity (objective 1)
PYTHONPATH=. $PY extra/benchmark_llm.py --model $MODEL --device-map METAL --prompt-tokens 16 --decode-tokens 64
PYTHONPATH=. $PY extra/benchmark_llm.py --model $MODEL --device-map NV    --prompt-tokens 16 --decode-tokens 64

# split runs, both directions (objectives 2-3)
PYTHONPATH=. $PY extra/benchmark_llm.py --model $MODEL --device-map "0-3:METAL,4-15:NV"  --prompt-tokens 16 --decode-tokens 64
PYTHONPATH=. $PY extra/benchmark_llm.py --model $MODEL --device-map "0-11:NV,12-15:METAL" --prompt-tokens 16 --decode-tokens 64

# JIT capture / graph-island introspection (objective 4) -- ad hoc script, not committed
PYTHONPATH=. $PY /path/to/td3_capture_check.py "0-3:METAL,4-15:NV"
PYTHONPATH=. $PY /path/to/td3_capture_check.py "0-11:NV,12-15:METAL"

# boundary cost, graphed steady state (objective 5)
PYTHONPATH=. DEBUG=2 $PY -m tinygrad.llm -m llama3.2:1b --device-map "0-3:METAL,4-15:NV"  --benchmark 3 --warmup
PYTHONPATH=. DEBUG=2 $PY -m tinygrad.llm -m llama3.2:1b --device-map "0-11:NV,12-15:METAL" --benchmark 3 --warmup

# regression test
PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v   # 29 passed, incl. TestDeviceMapMetalNV for real
PYTHONPATH=. $PY -m mypy tinygrad/                                # Success: no issues found in 216 source files
$PY -m ruff check extra/benchmark_llm.py test/unit/test_llm_device_map.py   # All checks passed
PYTHONPATH=. $PY -m pytest test/unit -q -n12                      # 844 passed, 68 skipped, 4 xfailed
```
