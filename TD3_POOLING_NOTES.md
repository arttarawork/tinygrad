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

## 7. MoE shape (TD.3-moe): routed experts on NV, rest on METAL

Reproduces T3.3's MoE placement cross-backend on real hardware: `experts:NV` (RTX 3090) + everything
else on METAL. Worktree/branch unchanged (`tinygrad-dock`, `task/TD.3-pooling`). Model: olmoe
(`allenai/OLMoE-1B-7B-0924-Instruct-GGUF` Q4_K_M, 16 blocks, 64 experts, 8 active/token, dim=2048,
already in the fetch cache: `.../downloads/d9f8816f773421fa69637257a3f71cdc`, 4,213,512,672 bytes).

**TL;DR: the MoE placement logic itself is fully correct cross-backend at real scale (exact tokens,
3 copies/layer confirmed) — but a new, structural bug in the NV eGPU transport (independent of T4.14,
undiscovered until this session) blocks the real olmoe split from ever reaching graphed/steady-state
execution. Correctness is proven; the steady-state performance number is a well-grounded prediction,
not a direct measurement, as a result.**

### 0. Mechanism correction: "NV:NAK" is a `DEV=` target string, not a device_map string

The env brief's phrasing ("use the NV:NAK device string") doesn't map onto `device_map`/`Tensor(device=...)`
literally — confirmed empirically before writing anything:

```
>>> Tensor([1,2,3], device='NV:NAK')
ValueError: invalid literal for int() with base 10: 'NAK'   # ops_nv.py's _select_iface parses
                                                              # the segment after ":" as a device INDEX
```

The real mechanism (already established by TD.1/TD.2a/TD.2b/TD.2c, just not spelled out in the TD.3
brief): `DEV` is a global renderer/default-device override parsed by `Target.parse` (`helpers.py:193-230`),
keyed by device NAME, e.g. `DEV=NV:NAK` sets `Device.DEFAULT="NV"` **and** forces NV's renderer to
`NAKRenderer` (tinymesa, no docker) for the whole process. `device_map`/`Tensor(device=...)` strings stay
plain `"NV"` throughout. To keep `Device.DEFAULT=METAL` (needed for this file's `Device.DEFAULT=="METAL"`
test-skip convention) while still routing NV compiles through NAK, use the multi-target form:
`DEV='METAL;NV:NAK'` (semicolon-separated `Target` list; verified both forms directly — `Device.DEFAULT`
and `Device['NV'].renderer` come out exactly as expected, no docker touched). Quote it — bash treats
unquoted `;` as a command separator. Running the full `test_llm_device_map.py` file with
`DEV='METAL;NV:NAK'` (including the pre-existing, docker-authored `TestDeviceMapMetalNV`) is 31/31 green
with colima stopped, confirming this is the correct, permanent invocation for this dock.

### 1. Tiny synthetic MoE, both directions (objective 1) — PASS

Added `TestDeviceMapMoEExpertsMetalNV` to `test/unit/test_llm_device_map.py`, mirroring
`TestDeviceMapMoEExpertsMetalCPU` (same `MOE_TEST_CONFIG`: 4 blocks, 4 experts, k=2). Run with
`DEV='METAL;NV:NAK'`:

- `test_experts_on_nv_rest_on_metal` (the production shape): exact tokens vs an all-METAL reference
  over 2 prompts x 6 generate steps each (prefill + JIT replay), **and** the captured-COPY hop count
  is exactly `4*3=12` (h, sel in; x_down out — see §3).
- `test_experts_on_metal_rest_on_nv` (reverse, cheap per the brief): exact tokens vs an all-NV
  reference, same 2 prompts. No capture introspection here (objective 1 only asked for "cheap").

31/31 in the file (29 pre-existing + 2 new), mypy + ruff clean. Swap unaffected (1066->1067 MB).

### 2. olmoe production shape vs both references (objective 2) — PASS (tokens), blocked (graphed perf)

`extra/benchmark_llm.py --device-map "METAL,experts:NV" --prompt-tokens 16 --decode-tokens 64`,
greedy, no BEAM:

| run | DEV | load device | load | warm | decode tok/s | 64 tokens |
|---|---|---|---|---|---|---|
| all-METAL | (none) | METAL | 1.488s | 32.347s | 2.195 | `[604, 253, 4376, 273, 253, 31142, ...]` |
| all-NV | `NV:NAK` | NV | 5.763s | 27.080s | 12.894 | **identical** |
| split (experts:NV, rest METAL) | `NV:NAK` | NV | 7.899s | 16.830s | 0.864 (eager, see §6) | **identical** |

All three produce the **exact same 64-token greedy sequence** — olmoe is argmax-equivalent
METAL/NV/split at this horizon, same as the dense T3.3/TD.3 findings, now confirmed for the real MoE
production shape too. The split run only completes in eager mode (`JIT=0`, see §6) because the graphed
path hits a structural transport bug before producing a single token — correctness is proven, the
benchmark-mode tok/s above is **not** comparable to the graphed references (eager pays full per-kernel
Python dispatch every step; not a boundary-cost measurement).

### 3. Hop count (objective 3) — 3/layer confirmed by mechanism + tiny/live introspection; NOT re-verified live on real olmoe over NV (blocked)

`FFNBlock._feed_forward` (`tinygrad/llm/model.py:167-180`) is the entire mechanism: `h.to(expert_dev),
sel.to(expert_dev)` (2 "in" copies) then `x_down = ...to(x.device)` (1 "out" copy) — generic Tensor
ops, zero scale- or backend-specific branching. Confirmed exactly `3/layer` two independent ways this
session:
- Tiny-config captured-COPY introspection (§1): `12 = 4*3`, both directions.
- Live `DEBUG=2` steady-state trace on the same tiny config (§4 below): exactly 12 copy lines, in the
  predicted h(64B)/sel(8B)/x_down(128B) pattern, repeating once per block.

Also ran the tiny `MOE_TEST_CONFIG` at olmoe's **real depth** (`num_blocks=16`, still-synthetic
weights, `DEV=NV:NAK`) purely to rule out a topology-depth trigger for §6's bug — it passed cleanly
(16 layers x 3 = 48 copies implied by the same code path, no crash), which is corroborating but not a
substitute for §3's actual ask: the captured-COPY count could not be taken directly on real olmoe over
NV, because JIT capture never completes there (§6). Given the mechanism has no size- or depth-dependent
branch, `48 = 16*3` for real olmoe follows from the code, not from a fresh introspection run.

### 4. Boundary cost (objective 4) — measured on tiny (real dock hardware), predicted for olmoe

Live `DEBUG=2` graphed steady-state, one decode step, tiny `MOE_TEST_CONFIG` (`METAL,experts:NV`,
`DEV='METAL;NV:NAK'`), 12 copies across 4 layers:

| hop | payload | timings (4 layers) | avg |
|---|---|---|---|
| h in (NV<-METAL) | 64 B | 70.75, 70.25, 67.96, 66.29 us | 68.8 us |
| sel in (NV<-METAL) | 8 B | 66.29, 66.04, 64.33, 66.29 us | 65.7 us |
| x_down out (METAL<-NV) | 128 B | 63.12, 61.79, 63.58, 61.88 us | 62.6 us |

All 12 cluster in **62-71 us**, average **65.7 us** — same flat-floor character T3.2/TD.3 found (cost
doesn't track payload size at these scales: 8 B and 128 B cost the same as 64 B), and in the same
regime as TD.3's dense 80-90 us/hop finding (a bit lower here, still same order of magnitude; both
numbers are well under 1M elements, T3.2's bandwidth-bound threshold). Real olmoe's actual hop
payloads (h~8 KB, sel~32 B, x_down~64 KB per token) are *also* comfortably inside that flat-floor
regime, so this tiny measurement extrapolates directly: **predicted olmoe boundary cost = 48 hops x
~65-90 us = ~3.1-4.3 ms/token** — consistent with (same order as, low end of) the task's ~4-5 ms/token
prediction. This is a **prediction grounded in a real on-dock measurement of the exact hop shape**, not
a direct end-to-end measurement on olmoe itself: §6's bug blocks the graphed split from running at all,
so the TD.3-dense-style "split tok/s vs all-NV tok/s, decompose the delta, attribute the remainder to
METAL compute share" exercise could not be completed for the real model this session.

### 5. Load-direction rule (objective 5) — holds by code argument + partial live evidence; wrong direction not forced live

Rule (T3.3, CPU): the GGUF load device (`Device.DEFAULT` when `gguf_load()` runs) must be the
big-memory/bulk side, or the cross-device `.to()` in `load_state_dict`/`realize_placement` force-realizes
the moved tensors at full fp16 before the copy, losing fused dequant. With experts on NV (the bulk of a
MoE model's weights), the load device must be NV.

- **Correct direction, live**: `DEV=NV:NAK` (`Device.DEFAULT=NV`) load of real olmoe with
  `experts:NV` (no motion — same-device, fused) + the small METAL share moving (force-realize, but
  small) completed in **7.899s** — comparable to the all-NV reference's 5.763s, no anomaly. §2's exact
  tokens confirm correctness too.
- **Wrong direction, NOT run live**: `Device.DEFAULT=METAL` would force the ~12.88 GB fp16 expert
  weights (`64 experts x 2048 x 1024 x 3 tensors x 16 layers x 2 bytes`, computed directly from this
  GGUF's own metadata — matches T3.3's CPU-measured "~13 GB for olmoe" almost exactly, same model) to
  materialize on METAL/host RAM before copying to NV. Deliberately not forced this session: llama-server
  already holds ~17-23 GB resident on a 36 GB machine, so a real +13 GB transient sits close to the
  edge, and §6 found a new NV-transport fragility that could compound unpredictably under a heavier
  real-scale load — compounding an already-understood risk with a newly-found unknown one is exactly
  the "don't fight it" case the brief calls out, just encountered one step earlier (at the memory-risk
  decision) rather than after seeing the blowup.
- **Code-path argument for why the CPU-proven mechanism transfers**: `Transformer.realize_placement`
  (`model.py:557-584`) and `nn.state.load_state_dict`'s cross-device line (`v.replace(state_dict[k].to(v.device))`,
  `nn/state.py:213`) contain **zero backend-specific branches** — generic `Tensor.to()`/`.contiguous()`/
  `.realize()` calls throughout. T3.3's CPU-measured blowup is a property of this shared, backend-agnostic
  code path, not of CPU specifically.
- **Verdict: rule holds.** High confidence from the code-path argument + matching analytic sizing +
  the correct direction's clean, fast, exact-token real-olmoe run; the wrong-direction blowup itself
  was reasoned about rather than re-triggered live, by choice, for the risk reasons above.

### 6. Structural finding: NV eGPU transport drops an ancillary FD under real-scale mixed-graph load (new — not T4.14, different call site)

Real olmoe's graphed split (`DEV=NV:NAK`, `--device-map "METAL,experts:NV"`, default JIT) crashes
**deterministically (2/2)** during `model.warmup()`'s first JIT capture/replay, after a successful
7.9s load:

```
IndexError: list index out of range
  File "tinygrad/runtime/support/system.py", line 379, in _rpc
    fd = struct.unpack('<i', anc[0][2][:4])[0]      # anc came back empty -- no fd delivered
  File "tinygrad/runtime/support/system.py", line 441, in alloc_sysmem   (APLRemotePCIDevice)
  File "tinygrad/runtime/ops_nv.py", line 107, in bind                   (NVComputeQueue.bind)
  File "tinygrad/runtime/graph/hcq.py", line 217, in __init__            (HCQGraph, first-use queue bind)
  File "tinygrad/engine/realize.py", line 136, in get_graph_runtime
```

`RemotePCIDevice._rpc`'s `has_fd=True` branch (`system.py:377-379`) does a single, non-retrying
`sock.recvmsg(17, socket.CMSG_LEN(4))` with no short-read handling — unlike its sibling `_recvall`
(same file, `:368-372`), which correctly loops until it has all requested bytes. On a long-lived,
heavily-reused stream socket (the TinyGPU.app RPC channel lives for the whole process), SCM_RIGHTS
ancillary data is only delivered attached to the specific `recvmsg()` call that reads the *first* byte
of the sender's `sendmsg()` — any prior short/misaligned read anywhere on that socket can silently
strand a later call's ancillary FD. This is the same bug **class** T4.14 already fixed (short-read
truncation), just a different call site never exercised by any prior TD session (nobody had previously
combined `Device.DEFAULT=NV` load-direction with a real-scale mixed METAL+NV graph until this task).

**Isolated with a targeted repro matrix** (all `device_map="METAL,experts:NV"`-shaped):

| scenario | Device.DEFAULT | weights | result |
|---|---|---|---|
| tiny MoE, 4 layers | METAL | synthetic | pass (committed test) |
| tiny MoE, 4 layers | **NV** | synthetic | pass |
| tiny MoE, **16 layers** (olmoe's real depth) | **NV** | synthetic | pass |
| real **dense** llama3.2:1b, 16 layers | **NV** | real GGUF | pass |
| real **olmoe**, 16 layers, MoE | **NV** | real GGUF | **crash, 2/2** |

Neither `Device.DEFAULT=NV` alone, nor 16-layer interleaving depth alone, nor real-GGUF scale alone
(dense) reproduces it — only real-scale **MoE** GGUF loading (many large per-tensor transfers feeding
straight into a mixed-graph capture) does. Consistent with (not proven as) a load-time framing desync
that only surfaces on the next `has_fd=True` call, i.e. the graph's queue-bind.

**Workaround found (diagnostic, not a fix): `JIT=0`** disables `TinyJit` capture entirely (falls back
to eager per-op dispatch, `engine/jit.py:260`), which avoids `HCQGraph` construction altogether and
produced §2's byte-identical, correct split tokens. Not usable for steady-state perf measurement (pure
Python dispatch overhead per kernel).

**Not fixed** — per the task's STOP condition: this is `tinygrad/runtime/support/system.py` socket-
protocol surgery (a different subsystem from `model.py`, and a different bug from T4.14), well outside
a MoE-placement-verification task's scope. Documented here as a candidate follow-up: harden
`RemotePCIDevice._rpc`'s `has_fd` branch with the same retry discipline `_recvall` already has.

### Exact commands (MoE)

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
OLMOE=/Users/artur/Library/Caches/tinygrad/downloads/d9f8816f773421fa69637257a3f71cdc

# tiny synthetic, both directions (objective 1) -- committed regression test
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v -k MoEExpertsMetalNV
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v   # full file, 31/31

# olmoe references + production split (objective 2)
PYTHONPATH=.          $PY extra/benchmark_llm.py --model $OLMOE --device-map METAL --prompt-tokens 16 --decode-tokens 64
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map NV    --prompt-tokens 16 --decode-tokens 64
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "METAL,experts:NV" --prompt-tokens 16 --decode-tokens 64  # crashes graphed (see S6)
JIT=0 DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "METAL,experts:NV" --prompt-tokens 16 --decode-tokens 64  # eager workaround -- exact tokens

# boundary-cost measurement (objective 4) -- ad hoc script, not committed, same pattern as TD.3's capture-check
DEV='METAL;NV:NAK' PYTHONPATH=. $PY /path/to/moe_hop_cost.py   # builds tiny MOE_TEST_CONFIG w/ device_map="METAL,experts:NV",
                                                                 # warms up, wraps one steady-state decode step in Context(DEBUG=2)

# gates
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v   # 31 passed
PYTHONPATH=.        $PY -m mypy tinygrad/                                          # Success: no issues found in 216 source files
$PY -m ruff check test/unit/test_llm_device_map.py                                 # All checks passed
```

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
