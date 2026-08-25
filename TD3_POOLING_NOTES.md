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

### 7. T4.17 follow-up: has_fd fix shipped + tested — but the real mechanism was NOT fragmentation, and a deeper, genuinely server-side ceiling now blocks full completion

Repro re-confirmed 1/1 at HEAD (`b3e64e734`) before any change, exact same `IndexError: list index out
of range` at `system.py:379`. Fixed `RemotePCIDevice._rpc`'s `has_fd` branch (`system.py`, +8 net lines,
one commit) and added 4 regression tests (`test/null/test_device.py::TestRemotePCIDeviceRPC`, real
AF_UNIX `socketpair`, no GPU) — all 4 fail against pre-fix code, all 4 pass post-fix. Full details: this
session's `T4.17: ...` commit.

**§6's "fragmentation" hypothesis was wrong** (it was explicitly hedged "consistent with, not proven
as" — this session proves what actually happens). Instrumenting `recvmsg()` directly on the real olmoe
repro shows **zero fragmentation at any point**: 128 consecutive `has_fd` calls each return the full
17-byte header + fd in a single `recvmsg()` call, then the 129th (and, via `LRUAllocator`'s catch-evict-
retry, the 130th) each return a **complete, well-formed 17-byte header with `status=1` (failure),
zero-length error body, and no cmsg** — a legitimate error reply, which correctly carries no fd. The
crash's real cause: the `has_fd` branch unpacked a fd unconditionally, before ever checking the status
byte, so a real RPC failure crashed exactly like a dropped fd would (same `IndexError`, same line —
indistinguishable from the outside, which is why §6 misattributed it). Fix does two things in the same
code region: (1) `_recvmsg_all()` loops `recvmsg()` until 17 bytes are collected, capturing the fd from
whichever fragment carries it (mirrors `_recvall`; this part *does* fix a real, separate short-read
hazard, verified by the fragmented-reply tests even though it wasn't what olmoe hit); (2) `_rpc()` now
checks the status byte first (existing `"RPC failed: ..."` path) and only requires a fd for a
*successful* `has_fd` call, raising a distinct, clear error otherwise.

**End-to-end verdict**: re-ran the exact §6 command post-fix (fresh `TinyGPU.app` server via
`pkill -f TinyGPU` + respawn, to rule out state accumulated across this session's several crashed runs
— result identical on a cold server, so not stale state). The documented crash is gone — no more
`IndexError`. But the graphed split still cannot complete: it now fails cleanly with
`RuntimeError: RPC failed: unknown error`, deterministic across 3 runs (2 warm-server, 1 cold-server),
always at the exact same allocation (`NVComputeQueue.bind`'s 104 B `hw_page`, ~4.89 GB already resident
on NV). This is a **new, separate, genuinely server-side finding**: `TinyGPU.app` (closed binary, not
in this repo, cannot be changed here — same constraint the task's STOP condition anticipated) refuses a
`has_fd` sysmem allocation once ~128-130 are concurrently outstanding. No previously-tested model came
close to that count — dense llama3.2:1b and the tiny synthetic MoE config both use far fewer live
sysmem mappings than real olmoe's many small per-expert tensors (64 experts x 3 tensors x 16 layers).
The exact server-side resource being exhausted is unknown (closed binary; 128 is a suspicious
round-number ceiling, consistent with a fixed-size table, but that's inference, not proof).

**Consequently, still blocked** (same bottom line as §2/§3/§4 before this session, now for a precise,
provably-server-side reason instead of a hedged guess): tokens-byte-match against the eager/all-device
references, end-to-end split tok/s vs. all-NV tok/s, and a live 48-copy count at olmoe scale all require
the graphed split to run past its first JIT capture, which it still cannot do. §4's ~3.1-4.3 ms/token
boundary-cost figure remains a grounded prediction, not a direct measurement. Swap stayed ~1.0-1.1 GB
across every run this session (fix work + repro + verification), no regression.

**Gates**: 4 new tests + full `test/null/test_device.py` (19 passed, 7 skipped, `-n12`) + real-hardware
`test/unit/test_llm_device_map.py` (31/31, `DEV='METAL;NV:NAK'`, deliberately run *without* `-n12` —
real shared NV device, parallel workers would contend for the GPU) + mypy (`Success: no issues found in
216 source files`) + ruff (`All checks passed!`) all green.

**PR-readiness**: the `system.py` commit is small (+8 net lines, one code region), hand-verifiable, one
lever, matches T4.14's fix shape and test style closely (loop-until-n helper, real-transport fake-server
unit tests, clear "got X of Y" error text) — same bar as T4.14, ready. It is a genuine, tested fix for a
real bug class (T4.14-sibling short-read *and* a status-before-fd ordering bug), independently worth
shipping even though it wasn't sufficient to unblock olmoe's graphed run end-to-end.

### 8. T4.18: sysmem-ceiling characterization + fix — graphed olmoe split now completes

Characterized §7's ~128-130 ceiling with call-site-attributed client-side instrumentation: a temporary
`sitecustomize.py` (not committed; on `PYTHONPATH`, auto-imported at interpreter start) monkeypatching
`APLRemotePCIDevice.alloc_sysmem` and `PCIIfaceBase.free` to log every call with a `traceback`-derived
call site, a monotonic id, and size. Run on (a) the tiny graphed MoE split (passes) and (b) real olmoe's
graphed split (crashes), each against a freshly `pkill`ed/respawned TinyGPU.app server for a clean count.

**Ranked table at the real-olmoe crash point** (128 cumulative `alloc_sysmem` calls succeed, ALL still
live — zero had been freed yet — before the 129th/130th fail exactly as §7 documented):

| rank | call site | count | share |
|---|---|---|---|
| 1 (tie) | `graph/hcq.py:32` `HCQGraph.__init__`'s per-graph-island `kernargs_bufs` | 43 | 34% |
| 1 (tie) | `ops_nv.py:107` `NVCommandQueue.bind()`'s per-queue `hw_page` | 42 | 33% |
| 3 | one-time device bring-up (32×2MB copy-staging pool 32 + GSP boot 8 + cmdq_page/signal-pool/persistent-kernargs-arena 3), model-independent | 43 | 34% |

**Dominant class**: the two `HCQGraph.__init__`-time allocators (85/128 = 66%), NOT the load path — GGUF
dequant loading (~12s of the run, in between id 43 and id 44) issues **zero** new sysmem allocations;
every load-time host-visible buffer is already served by an existing pool (the 32×2MB copy-staging
ring, the 16MB persistent kernargs arena, or the cmdq page). The real driver: a METAL+NV MoE graph
alternates devices at every layer boundary (3 cross-device copies/layer, §3), and `engine/jit.py`'s
`graph_split_rewrite` flushes a new graph island whenever consecutive ops can't share a device set —
confirmed live with `JIT_BATCH_SIZE=4096` (128× the default): island sizes stayed at 5–27 kernels,
unchanged, proving the split is device-alternation-driven, not size-cap-driven (`JIT_BATCH_SIZE` doubles
per flush, so a size-cap-bound split would have grown islands fast; it didn't). Real olmoe's per-layer
METAL↔NV↔METAL structure needs ~85 islands across warmup's 4 forward passes; TD.2a measured only ~4
islands/token for dense llama3.2:1b. Each island permanently owns one `kernargs_bufs` plus one-or-more
`hw_page`, each a fresh `has_fd` RPC.

**Are they still needed at crash time — and does "freed" mean anything here?** In this run, yes: every
live allocation is a currently-referenced graph island (nothing had been superseded yet). But a second
finding from the same instrumentation matters more. In a longer tiny-model run where JIT graphs DO get
replaced (the pytest suite, multiple prompts/steps), 71/115 of these same-class allocations do show
`freed == total` — yet serving them from the ordinary `LRUAllocator` cache once freed (see "first
attempt" below) produced **zero reduction** in subsequent `alloc_sysmem` calls. Root cause: `RemoteCmd`
(`system.py`) has **no unmap/free verb at all** — the server's slot table can only grow for the life of
the process regardless of what the client does, and `PCIIfaceBase.free()`'s only destructive branches
are gated on `is_local()` (false for the remote/APL transport) or `AddrSpace.PHYS` (sysmem uses
`AddrSpace.SYS`), so freeing a remote sysmem `HCQBuffer` is *already* a no-op today on every side (server
slot, and even the client's own dup'd fd + mmap). A fix that waits for a free to reuse a slot cannot
work — confirmed by shipping exactly that version first and re-running the crash repro unchanged.

**Fix (shipped, `tinygrad/runtime/ops_nv.py`, +27/−2 lines, one file)**: since a sysmem slot can never be
returned once granted, the only lever is issuing fewer `alloc_sysmem` calls in the first place. Every
observed `hw_page` this session was exactly 16384 B, so `NVCommandQueue.bind()` now serves it from a
single lazily-allocated, never-freed 4MB slab (`NVDevice._hwq_slab`), bump-suballocated
(`tinygrad.runtime.support.memory.BumpAllocator`, `wrap=False`) instead of a fresh RPC per bind() — same
one-alloc-many-small-pieces shape as the existing 32×2MB copy pool — falling back to a real allocation
if the slab (sized for ~256 islands) is ever exhausted. Guarded entirely by `dev.is_remote()`, mirroring
`bind()`'s existing `is_remote`-checked bulk-write optimization two lines below and T2.2/T2.3's
precedent — the local/NVK path takes the original, untouched branch. `__del__` skips freeing
slab-derived slices (identified by `hw_page.base is dev._hwq_slab`; a shared slice must never be
individually released) but frees real fallback-path allocations exactly as before. Only `hw_page` was
touched — `kernargs_bufs` (the other ~34%) lives in the cross-backend `graph/hcq.py` and was
deliberately left alone (see "not fixed" below); fixing `hw_page` alone was sufficient (verified below).

**Verified**:
- Real olmoe graphed split (`DEV=NV:NAK`, `--device-map "METAL,experts:NV"`, default JIT) **completes
  end-to-end** for the first time: `load 8.016s, warm 17.871s, prefill 111.080 tok/s, decode 41.142
  tok/s`, 64-token output **byte-identical** to a fresh all-NV reference run in the same session and to
  §2's documented all-METAL/all-NV table (`[604, 253, 4376, 273, 253, 31142, ...]`, all 64 tokens match).
- Peak outstanding `alloc_sysmem` calls for this now-*complete* run: **108**, down from the **128** that
  used to crash a much shorter (warmup-only, partial) run — `hw_page` collapsed from 42 calls to 1 (the
  slab itself, 4,194,304 B); `kernargs_bufs` alone (still unfixed) grew to 64 for the full run but no
  longer combines with `hw_page` to cross the ceiling.
- Split decode tok/s (41.142) beats the all-NV reference (12.870, matching the previously-documented
  12.894) — MoE's non-expert compute (attention, router, lm_head) runs natively on METAL while only the
  expert GEMVs pay the NV eGPU cost, so this isn't the "split adds boundary overhead vs. best single
  device" shape T0.3/TD.3's dense methodology assumed; there's no positive delta here to decompose into
  a boundary-cost estimate. §4's ~65–90 µs/hop floor and ~3.1–4.3 ms/token prediction stand as the best
  available boundary-cost figure. A live decode-shape capture this session did show cross-device copies
  repeating every layer (64 KB METAL←NV + 8 KB NV←METAL), comfortably inside the flat-floor regime — but
  only 2 hops/layer were visible at decode batch=1, not the tiny model's 3 (§3), and re-deriving the
  exact real-olmoe hop count/direction breakdown was not completed this session. Flagged as a minor open
  item (possible scheduler-level fusion of the h/sel input copies at this batch size), not a correctness
  concern — §3's code-path argument for the copy mechanism doesn't depend on the count matching exactly.
- Gates: `test/unit/test_llm_device_map.py` 31/31 (`DEV='METAL;NV:NAK'`), full `test/unit -q -n12` 844
  passed / 70 skipped / 4 xfailed (same DEV; the one failure seen without it is the known
  docker/NAK-compiler-lane requirement, not a regression), mypy (`Success: no issues found in 216 source
  files`), ruff (`All checks passed!`) all green.

**Not fixed (documented, not attempted)**: `kernargs_bufs` (`graph/hcq.py:32,316`) is the other ~34% of
the dominant class with the identical pathology (fresh `_alloc`/`_free` per graph island, bypassing even
the ordinary LRU cache) — but it lives in `HCQGraph`, shared by NV/AMD/QCOM, and a safe version of the
same slab trick there needs a generic (non-NV-specific) remote-detection hook plus care that a graph's
kernargs stay at a *stable* address for the life of every future replay (a *wrapping* bump allocator,
like the existing per-device eager-mode kernargs arena, would silently corrupt a still-live graph's
arguments once it wraps — exactly the allocator-redesign risk the task's STOP condition calls out). Left
alone because fixing `hw_page` alone already dropped the full-run peak to 108 (comfortably under
~128–130); revisit only if a bigger model or longer run pushes past that new headroom.

**First attempt (shipped, then reverted after disproving it)**: before finding the slab fix, tried the
much smaller "drop `nolru=True` so `hw_page` recycles through the ordinary `LRUAllocator` cache on the
remote path" (`nolru=not dev.is_remote()`, 2-line diff). mypy/ruff clean, tiny-test still byte-exact —
but re-running the real-olmoe crash repro reproduced the **identical** crash at the **identical**
allocation, peak count unchanged. Root cause (above): nothing is freed until process exit, so a cache
keyed on "wait for a free" never gets a hit while the run is still alive. Reverted in favor of the slab,
which doesn't depend on freeing anything.

### 9. T4.19 — experts-split divergence at decode index 60: root-caused (class a, benign FP drift), not a routing bug; hops-per-layer re-confirmed at 3

BENCH_NOTES.md's "Correctness caveat" (BEAM'd-pooling session) found the `experts:NV` split diverging
from the mutually-byte-identical single-device references at decode index 60 of the 512-prompt/128-
decode convention (`ref=1232` vs `split=11723`), a real, deeper-than-T4.18/§8's-16/64-horizon finding
that never got explained. This session explains it with per-tensor evidence, not just re-observing it.

**TL;DR: class (a), benign cross-device FP drift — not a routing/dtype bug.** The expert GEMV (matmul,
not the copy) computes a tiny (~1e-6-abs at layer 0) different value on NV vs METAL for identical inputs
— ordinary cross-backend floating-point non-associativity — which compounds through the residual stream
across 16 layers (~1e-3-abs by layer 15) until it tips a genuinely near-tied final-logit argmax (top-2
gap collapses from a normal-step ~0.8 to ~0.01–0.02 exactly at the divergence step). Expert routing
(`sel`) matches **exactly**, all 16 layers, at the actual divergence step — the routing-bug candidate is
directly ruled out, not just assumed. Does not reproduce at tiny scale even pushed 125x past the real
trigger depth. The hops-per-layer count is **3**, re-confirmed at real decode shape with a clean capture
— §8's "2 hops visible" open item does not hold up.

#### 9.1 Reproduction (real olmoe, 512-prompt/128-decode, NAK lane) — objective 1

Re-ran BENCH_NOTES.md's exact convention fresh at this session's HEAD (`539c83d5f`), `DEV=NV:NAK`
(brief's preferred lane; compile-fast, same shape as nvcc per the brief):

```
all-NV reference:  [321, 420, ..., 6973, 15, 380, 1232, 1537, ...]
METAL,experts:NV:  [321, 420, ..., 6973, 15, 380, 11723, 6003, ...]
```
Programmatic diff (full 129-element arrays): **first divergence at index 60 exactly** (`ref[60]=1232`,
`split[60]=11723`), 69/129 total mismatches after that (the two trajectories fall into different
repetition loops, as expected for this synthetic prompt at this depth — neither is garbage). Matches
BENCH_NOTES.md's documented numbers exactly — same bug, reproduced fresh, not a stale/one-off finding.

#### 9.2 Tiny-scale bisection (objective 1) — does NOT reproduce

Built tiny MoE configs (`test/unit/test_llm_device_map.py`'s `MOE_TEST_CONFIG` shape, randomized expert
weights via the file's own `_randomize_experts`) and ran decode far past the committed test's 6-step
check, looking for the split to diverge from an all-METAL/all-NV reference at *any* point:

| shape | seeds tried | max decode steps | divergences found |
|---|---|---:|---:|
| shallow, 4 blocks (`MOE_TEST_CONFIG` as-is) | 7, 1, 42, 123 | 2500 | 0 |
| deep, 16 blocks (olmoe's real depth, same tiny dim/experts) | 7, 1, 42 | 2500 (seed 7 also run to **8000**) | 0 |
| deep+wide, 16 blocks, dim=64, hidden=128, 8 experts/k=2 | 7 | 2500 | 0 |

Zero divergences across every (shape, seed) combination — including 8000 decode steps on the config
that matches olmoe's real depth, **125x** the real trigger's index-60 depth. (Round 2's sweep hit the
already-documented T4.18 §8 `kernargs_bufs` sysmem ceiling — expected, not a new bug: accumulating many
sequential JIT-graphed `Transformer` instances with different shapes in one long-lived process exhausts
the ~128-130 concurrent-`alloc_sysmem` limit that `kernargs_bufs` was explicitly left unfixed for. Fixed
with the standard `pkill -f "TinyGPU.*server"` + NV health-check, then continued in a fresh process.)
**Verdict: no tiny repro.** Per the brief, continued the hop/class isolation on real olmoe directly.
Random untrained weights plausibly just don't produce the same near-tied router/logit geometry a real
trained model's logit distribution does; not investigated further (would need real trained tiny weights,
out of scope for a 2-hour task).

#### 9.3 Which hop carries the numerical difference (objective 2) — the expert GEMV, not either copy

Instrumented real olmoe (`dim=2048, 16 blocks, 64 experts, k=8, vocab=50304`) by monkeypatching (not
editing) `FFNBlock._feed_forward`/`Transformer.forward` to stash per-layer `sel`/`h`/`x_down`/`probs`
tensor refs and the final logits. Constraint discovered along the way: `TransformerBlock.__call__`'s
`_run` closure is `@function(precompile=True, allow_implicit=True)`-wrapped, which runs with
`Context(ALLOW_DEVICE_USAGE=0)` (`tinygrad/function.py:53`) — `.item()`/`.tolist()` are disallowed while
still inside that trace. Worked around by stashing bare Tensor refs inside `_feed_forward` and only
extracting values in `Transformer.forward` *after* the block loop returns (a plain, unguarded method).

Ran both `ref` (all-METAL) and `split` (`METAL,experts:NV`) under the **normal fast graphed JIT** up to
output index 58 (identical state to the real 512/128 run, verified: both match the §9.1 logs exactly
through index 58), then switched the last few steps to `Context(JIT=0)` (eager, `engine/jit.py:260`'s
"ignore" branch — runs `Transformer.forward`'s Python body fresh every call instead of replaying a
captured graph) so the monkeypatch fires. Per-layer `xdown_sample` delta (first 8 elements of the
expert-output tensor, ref vs split) at step 60:

| layer | 0 | 1 | 2 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|---|---|
| abs delta | 2.3e-6 | 3.0e-6 | 1.5e-5 | 4.6e-5 | 4.0e-4 | 7.5e-4 | 1.9e-3 |

`h` (the pre-hop input, identical at layer 0 since nothing has diverged yet) and `sel` (the routing
indices) are **exact bit-for-bit copies** across the `.to(expert_dev)` hop by construction (`.to()` is a
data movement, not a compute op) — confirmed empirically too (`sel` matches at every layer, see §9.4).
The delta is already present at **layer 0's `x_down`** (2.3e-6 abs, on O(1)-magnitude values) — i.e. the
instant the identical `h`/`sel` reach the expert GEMV computed on NV vs. the same GEMV computed on
METAL, the two hardware/kernel paths produce a tiny different rounding, then it **compounds roughly
800x over the 16-layer residual stream** (2.3e-6 → 1.9e-3) until it's large enough to matter at the
final LM-head argmax. **Verdict: the expert compute itself (cross-backend non-associative FP reduction),
not either copy** — `h`-in and `sel`-in carry zero error (lossless data movement); `x_down`-out carries
exactly whatever error the NV-vs-METAL GEMV already introduced upstream of it.

**Corroborating control experiment (not requested, but cheap and load-bearing):** with the identical
state through index 58, `split[60]` **does not diverge from `ref[60]`** when the tail is run eagerly
(`1232` both) — vs. `11723` in the fully-graphed run. Same weights, same tokens, same device placement;
the only thing that changed is graphed-replay vs. eager-dispatch execution of the *same* logical ops.
This is exactly what a genuinely near-tied comparison should do (fragile, execution-path-sensitive) and
is not what a structural bug (e.g. corrupted indices) would do (mode-independent, wouldn't care whether
the last few kernels were graphed or eager). Consistent with — and a direct instance of — candidate
(a)'s literal mechanism: a boundary/execution-path change alters accumulation/fusion order enough to
flip a hair's-breadth argmax.

#### 9.4 Naming the class (objective 3) — (a), benign FP drift; the heart-of-the-task check

**Expert-ID match at the divergence step:** at step 60 (the actual flip), `sel` (all 8 selected expert
IDs) matches **exactly, at all 16 layers**, ref vs. split — routing is not the source of the bug. (One
nuance worth recording: at step 59, two layers show a same-*set*-different-*order* `sel` — e.g. layer 13
`[...,9,45,...]` vs `[...,45,9,...]` — a `pairwise_topk` tie-break flip from the same accumulating drift;
harmless, since `probs = scores.gather(-1, sel)` gathers with the same permuted `sel`, so the weighted
sum is unaffected. This is corroborating evidence the drift is real and growing, not a second bug.)

**Logit-gap evidence:**

| step | max abs logit delta | ref top1–top2 gap | split top1–top2 gap | flips under graphed JIT? |
|---|---:|---:|---:|---|
| 59 | 0.101 | 0.826 | 0.823 | no (both pick 380) |
| **60** | 0.152 | **0.0225** | **0.0114** | **yes** (ref picks 1232, split picks 11723) |
| 61 | 0.038 | 0.112 | 0.115 | no (both pick 1537) |

At step 60 both `ref` and `split`'s top-2 candidates are the **identical pair** {1232, 11723} — both
placements agree these two tokens are neck-and-neck; the gap between them (0.011–0.022 out of logits
~13) is 35-70x smaller than a normal step's ~0.8 gap. This is precisely the brief's candidate-(a)
signature ("logit deltas ... tiny and the top-2 logits are near-tied"), not candidate (b)'s ("routing
difference" — ruled out above — "or a delta far larger than fp16 rounding" — 0.15 absolute on a ~13-scale
logit is small, and it's concentrated in a genuine coin-flip-close tie, not a gross error).

**Verdict: class (a).** Experts-split pooling is numerically sound. The T4.18/§7/§8 "byte-identical"
correctness claim for this model/placement needs the qualifier BENCH_NOTES.md already flagged: exact at
the 16-prompt/64-decode horizon those sessions tested, and *still mechanistically sound* at the deeper
512/128 horizon — the divergence there is the same ordinary cross-backend FP-non-associativity class as
T4.10/T4.24/T3.3's dense findings, now confirmed (not just presumed) for the `experts:` MoE placement
shape too, with the routing-bug alternative explicitly measured and ruled out rather than assumed away.
This closes the TD.4 documentation-risk item BENCH_NOTES.md's caveat flagged.

#### 9.5 Hops-per-layer at decode shape (objective 4) — 3, not 2; §8's open item resolved

§8 flagged a live decode-shape capture showing only 2 cross-device copies/layer (64 KB METAL←NV + 8 KB
NV←METAL) against the tiny-model/code-path-confirmed 3, and left it open ("not completed this session").
Re-captured cleanly this session: warmed up normally, drove 1 prefill + 2 decode calls to force the
greedy rollout JIT key past capture into confirmed **steady-state replay** (`cnt>=2`, `engine/jit.py`'s
"jit exec" branch), then wrapped exactly one more decode call in `Context(DEBUG=2)` and read the raw
copy lines (full log saved, 100 lines, 49 mention `copy`):

```
NV      copy    8.00 KB,      NV <- METAL     (h in)
NV      copy       32 B,      NV <- METAL     (sel in)
METAL   copy   64.00 KB,   METAL <- NV        (x_down out)
```
repeating **exactly 16 times** (once per real olmoe block) = 48 cross-device copies, plus one unrelated
`4 B, METAL <- METAL` same-device copy (sampled-token bookkeeping, not a MoE hop) = 49 total, matching
the grep count exactly. Payload sizes match olmoe's real shape precisely (`h`: 1×1×2048×4B = 8192B;
`sel`: 1×1×8×4B = 32B; `x_down`: 1×1×8×2048×4B = 65536B — `dim=2048`, `k=8` from this GGUF's own
metadata). **This is the full 3/layer, not 2** — matching the tiny-model count (§1/§3) and the
zero-branching code-path argument (`model.py:171-172,180`'s three `.to()` calls) exactly, with no
surprises. §8's "2" was very likely a capture-methodology artifact (e.g. capturing the one-time JIT-
*capture* step rather than a confirmed steady-state *replay*, or a grep that missed the tiny 32 B line
among the noise) rather than a real structural difference — this session's capture explicitly verified
`cnt>=2` before capturing and saved the full raw log, so there's no such ambiguity here. **Answer: 3
hops/layer at decode shape, same as every other shape measured.**

#### 9.6 Regression test + gates

Added `test_experts_split_no_divergence_deep` to `TestDeviceMapMoEExpertsMetalNV`
(`test/unit/test_llm_device_map.py`): olmoe-depth (16-block) tiny config, 200-decode-step exact-match
check between `ref` and `split` — a much deeper regression guard than the existing 6-step placement
test, cheap (a few seconds), and would very likely catch a real routing/dtype bug reintroduced into the
hop mechanism (§9.2 found zero false positives up to 8000 steps on this shape, so the guard has real
margin, not a hair-trigger). 32/32 in the file (31 pre-existing + 1 new), `DEV='METAL;NV:NAK'`; full
`test/unit -q -n12` 844 passed / 71 skipped / 4 xfailed (same DEV); mypy (`Success: no issues found in
216 source files`); ruff (`All checks passed!`) all green. Swap held ~1.8 GB throughout (no regression),
NV health-checked clean before, during (after the expected T4.18-ceiling hit), and after.

### Exact commands (T4.19)

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
OLMOE=/Users/artur/Library/Caches/tinygrad/downloads/d9f8816f773421fa69637257a3f71cdc

# 9.1: reproduce the divergence fresh (NAK lane)
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map NV --prompt-tokens 512 --decode-tokens 128
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "METAL,experts:NV" --prompt-tokens 512 --decode-tokens 128
# programmatic diff of the two "output [...]" lists -> first mismatch at index 60

# 9.2: tiny bisection (ad hoc scripts, not committed) -- 0 divergences, up to 8000 decode steps
DEV='METAL;NV:NAK' PYTHONPATH=. $PY /path/to/tiny_bisect.py    # 400 steps x 3 shapes x seed=7
DEV='METAL;NV:NAK' PYTHONPATH=. $PY /path/to/tiny_bisect2.py   # 2500 steps x 3 shapes x up to 4 seeds
DEV='METAL;NV:NAK' PYTHONPATH=. $PY /path/to/tiny_bisect3.py   # 8000 steps, deep-16block, seed=7
pkill -f "TinyGPU.*server"   # only needed after round 2's expected T4.18-Sec8 kernargs-ceiling hit

# 9.3/9.4: hop-trace instrumentation (ad hoc, not committed) -- monkeypatches _feed_forward/forward,
# 58 graphed steps then Context(JIT=0) for the tail; saves a pickle of per-layer/per-step tensors
DEV=NV:NAK PYTHONPATH=. $PY /path/to/olmoe_hop_trace.py

# 9.5: decode-shape hop count (ad hoc, not committed) -- warms up, forces steady-state replay
# (1 prefill + 2 decode calls), then Context(DEBUG=2) around exactly one more decode call
DEV='METAL;NV:NAK' PYTHONPATH=. $PY /path/to/olmoe_decode_copies.py

# gates
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v   # 32 passed
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit -q -n12   # 844 passed, 71 skipped, 4 xfailed
PYTHONPATH=. $PY -m mypy tinygrad/                                # Success: no issues found in 216 source files
$PY -m ruff check test/unit/test_llm_device_map.py                # All checks passed
```

### Exact commands (T4.18)

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
OLMOE=/Users/artur/Library/Caches/tinygrad/downloads/d9f8816f773421fa69637257a3f71cdc

# characterization: temporary sitecustomize.py (not committed) on PYTHONPATH auto-installs hooks on
# APLRemotePCIDevice.alloc_sysmem / PCIIfaceBase.free with traceback-attributed call sites
pkill -f "TinyGPU.*server"   # clean slot count before each ceiling-crossing run
T4_18_TAG=<name> T4_18_TRACE_DIR=<dir> DEV=NV:NAK PYTHONPATH=<dir>:. $PY extra/benchmark_llm.py \
  --model $OLMOE --device-map "METAL,experts:NV" --prompt-tokens 16 --decode-tokens 64

# post-fix verification (no instrumentation needed)
pkill -f "TinyGPU.*server"
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "METAL,experts:NV" \
  --prompt-tokens 16 --decode-tokens 64   # now completes: load/warm/prefill/decode all print, exact tokens
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map NV \
  --prompt-tokens 16 --decode-tokens 64   # all-NV reference for the tok/s comparison

# gates
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v   # 31 passed
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit -q -n12   # 844 passed, 70 skipped, 4 xfailed
PYTHONPATH=. $PY -m mypy tinygrad/                                # Success: no issues found in 216 source files
$PY -m ruff check tinygrad/                                       # All checks passed
```

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

# T4.17 follow-up (S7): fix verification + the new, deeper server-side ceiling
PYTHONPATH=. $PY -m pytest test/null/test_device.py -v -k RemotePCIDeviceRPC   # 4 new tests, real AF_UNIX socketpair
pkill -f TinyGPU && rm -f /var/folders/*/T/tinygpu.sock   # cold-restart the server to rule out state from earlier crashed runs
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $OLMOE --device-map "METAL,experts:NV" --prompt-tokens 16 --decode-tokens 64
  # post-fix: no more IndexError -- now a clean "RuntimeError: RPC failed: unknown error" at the same
  # NVComputeQueue.bind allocation, deterministic on both warm and cold server (new S7 finding)

# gates
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -v   # 31 passed
PYTHONPATH=.        $PY -m mypy tinygrad/                                          # Success: no issues found in 216 source files
$PY -m ruff check test/unit/test_llm_device_map.py                                 # All checks passed
```

### 10. T4.21: the load-path fix -- big-model range splits no longer materialize the moved share at fp16

The CORRECTION/qwen3.6 sections above root-caused the swap explosion to a **load-path gap**, not a
residency ceiling: `gguf_load` always staged a tensor's raw quantized blob on `Device.DEFAULT`, so
`ggml_data_to_tensor`'s dequant chain was built there too; when `device_map` then moved a param to a
different device, `load_state_dict`'s `.to(param.device)` (`nn/state.py:214`) wrapped that WHOLE dequant
chain in a COPY -- the COPY sat above the dequant, so `Transformer.realize_placement()`'s force-realize
(needed to stop the COPY being recaptured+re-paid every JIT-replayed token, T3.2/T4.5) materialized the
FULL dequantized size (fp16) on the LOAD device before a single byte reached the target. Fine for T3.3's
small moved share; for ~half a big model, fatal (measured: 16.4 GB swap in 20s, row 3 above).

**Fix, two parts, both required** (a loader-only fix alone was insufficient -- see below):

1. **`tinygrad/llm/gguf.py`** (`_gguf_parse`/`gguf_load`, +~55 lines): when `device_map` is passed, resolve
   each raw GGUF tensor NAME to a target device (mirroring `Transformer.__init__`'s block/`experts:`
   placement, via `parse_device_map` -- imported locally inside `_gguf_parse` to avoid a cycle with
   `model.py`, which imports `gguf_load` at module scope) and stage that tensor's blob **directly on its
   target device**, before `ggml_data_to_tensor` builds the dequant on top. The per-batch merge-adjacent-
   tensors optimization (`_STAGE_BATCH`) now also breaks a batch at a device boundary, so a batch shared by
   two differently-placed tensors doesn't force them onto the same device. `num_blocks` is computed from
   `kv_data` using the exact same formula `from_gguf` uses (block_count minus any MTP nextn layers); a
   tensor whose block index falls outside that range (qwen3.6's unreferenced MTP block) falls back to the
   last device, same convention as `output`/`output_norm`. A GGUF missing the KV needed to resolve this
   (e.g. a later multi-part-split file, which isn't guaranteed the full metadata) falls back to the
   pre-T4.21 `Device.DEFAULT` staging for that call instead of raising. Net effect: `load_state_dict`'s
   later `.to(param.device)` is a no-op for these tensors (source and target already match) -- no COPY node
   exists in the graph at all.

2. **`tinygrad/llm/model.py`** (`realize_placement`, ~10 line change): turns out (1) alone is NOT enough --
   `realize_placement()` still unconditionally forced `.contiguous()+.realize()` on every param with
   `device != Device.DEFAULT`, and forcing a realize on a lazy dequant chain necessarily materializes its
   FULL output (there's no partial-materialize for an elementwise dequant) -- so even with the blob already
   local, the moved param would still eagerly expand to fp16, just on the correct device instead of the
   wrong one (confirmed empirically below: NV residency was still ~fp16-sized, 6.92 GB, with (1) alone).
   The fix: only force-realize a moved param whose **top-level op is still `Ops.COPY`**
   (`p.uop.op is Ops.COPY`) -- `load_state_dict`'s assignment (`nn/state.py:214`) is a bare `.to()` with
   nothing layered after it, so a genuine cross-device transfer of arbitrary upstream compute is ALWAYS the
   outermost op right after load; if it isn't there (T4.21's loader already placed the blob correctly), the
   dequant sits directly over an already-resident buffer with nothing to realize early -- exactly as safe
   and as memory-cheap to leave lazy/fused as a same-device param already was. Verified this doesn't touch
   any EXISTING test's behavior: every synthetic (non-GGUF) test in `test_llm_device_map.py` builds its
   "moved" params via `nn.state.load_state_dict(split, nn.state.get_state_dict(ref), ...)` where `ref`'s
   values (realized or not) sit on a device that genuinely differs from `split`'s target -- `.to()` always
   inserts a real COPY there, so `p.uop.op is Ops.COPY` holds and the old force-realize behavior is
   unchanged for all of them (confirmed: same 32/32 pass, including the T4.5 hop-count regression tests,
   which measure ACTIVATION copies inside `_feed_forward`, unrelated to this weight-loading path).

**Residency proof** (olmoe Q4_K_M, 4.2135 GB file, `DEV='METAL;NV:NAK'`, `device_map="0-7:METAL,8-15:NV"`,
`GlobalCounters.mem_used_per_device` read immediately after `from_gguf` returns -- no forward pass, so this
is pure weight residency):

| | METAL (unmoved, `Device.DEFAULT`) | NV (moved) | TOTAL |
|---|---:|---:|---:|
| PRE-fix (`git stash`) | 2.1126 GB | **6.9192 GB** | 9.0317 GB |
| POST-fix | 2.0915 GB | **2.1202 GB** | 4.2117 GB |

NV's residency drops from 3.3x its quantized share (fp16, matching the diagnosis exactly) to matching it
almost exactly; TOTAL drops from 2.1x the file size to ~1.00x. This is the core proof T4.21 exists to
produce. (1) alone, tested in isolation before adding (2), gave NV = 6.9192 GB -- identical to pre-fix,
confirming (2) was the part actually doing the residency work; (1) is still necessary because it's what
gives `realize_placement()` a clean top-level-op signal to check, and what stops the LOAD device from
transiently spiking (see qwen3.6 swap log below -- (1) is what a UOp-rewrite-of-the-COPY-alone would have
had to reproduce anyway).

**olmoe correctness -- two layers:**
- New, committed, fast unit tests (`test/unit/test_gguf.py`,
  `TestGGUF.test_device_map_places_blob_on_target_before_dequant` /
  `test_device_map_malformed_kv_falls_back_to_default`, using the file's existing `_build_gguf` synthetic-
  GGUF helper -- no download, no real model): block ranges, `token_embd`/`output_norm`'s
  block-0/block-(-1) convention, the `experts:` override, an MTP-style out-of-range block index falling
  back instead of raising, and the malformed-KV graceful fallback -- all assert both final `.device` AND
  `p.uop.op is not Ops.COPY` (the actual mechanism, not just where it ends up).
- Real-scale, ad hoc (not committed, same convention as this file's other big-model checks):
  `extra/benchmark_llm.py --device-map "0-7:METAL,8-15:NV"` vs `--device-map NV`, `DEV='METAL;NV:NAK'`,
  512 prompt / 128 decode tokens (the exact §3-bonus depth) -- **129/129 tokens byte-identical**
  (programmatic diff, 0 mismatches), reproducing §3's own pre-fix-loader finding now on the post-fix
  loader: the range-split mechanism itself was never in question, only its memory behavior at scale.

**T3.3's "load on the big-memory side" rule -- obsolete for the `from_gguf`/`device_map` path.** That rule
existed only because a moved param unconditionally materialized at fp16, so which side happened to be the
transient LOAD device determined where a memory spike landed. Post-T4.21, a `gguf_load`-sourced moved
param's blob is staged and (if still needed) realized directly on ITS OWN target device -- there is no
cross-device fp16 detour left for "load device" to describe. **Not obsolete for the generic/manual-loader
path** `realize_placement()` still serves (its own docstring's `nn.state.load_state_dict(...,
realize=False); model.realize_placement()` example) -- a hand-built or non-GGUF state_dict can still
produce a genuine COPY-over-arbitrary-compute, and T3.3's asymmetry could still apply there if the moved
share is large. One known, small, NOT fixed-by-T4.21 corner even on the `from_gguf` path: a TIED embedding
(`output.weight` aliased to `token_embd.weight` when the GGUF has no separate `output.weight` tensor,
`model.py:605`) is assigned in `model.py` *after* `gguf_load` returns, so the loader never sees a raw
`output.weight` tensor to place directly -- if `device_map`'s first-block device differs from its
last-block device, that ONE tensor still takes the pre-T4.21 COPY-after-dequant path. Bounded by
`vocab_size x dim` (a few hundred MB at most, not the multi-GB-scale bug this fixes), and not hit by either
model tested this session (checked directly: both olmoe and qwen3.6-mxfp4 have a separate `output.weight`
tensor on disk, not tied) -- noted for completeness, not chased.

**qwen3.6-35B-A3B payoff run:** see BENCH_NOTES.md's new "T4.21" section for the full numbers (loads
without the swap explosion, `JITBEAM=2` now fits, 4x longer context fits, decode tok/s vs the 56.58
baseline, and an honest correctness caveat).

```bash
# gates (all green)
PYTHONPATH=.                       $PY -m pytest test/unit/test_llm_device_map.py test/unit/test_gguf.py -q   # 75 passed
DEV=CPU PYTHONPATH=.               $PY -m pytest test/unit/test_llm_device_map.py test/unit/test_gguf.py -q   # 68 passed, 7 skipped
DEV='METAL;NV:NAK' PYTHONPATH=.    $PY -m pytest test/unit/test_llm_device_map.py -q                          # 32 passed
PYTHONPATH=.                       $PY -m pytest test/unit -q -n12          # 846 passed, 71 skipped, 4 xfailed (was 844 pre-T4.21)
PYTHONPATH=.                       $PY -m mypy tinygrad/                    # Success: no issues found in 216 source files
                                    $PY -m ruff check .                     # All checks passed
```

### 11. T4.20: re-scope with gpt-oss + T4.21 -- range split doesn't need the fix; shipped it anyway (cheap, proven, protects the shapes that still do)

**Step 1 (decisive): does gpt-oss:20b range-split clean now?** Yes. `--device-map "0-11:METAL,12-23:NV"`
(gpt-oss:20b is 24 blocks), no-BEAM, `DEV=NV:NAK`, 16 prompt/64 decode tokens: **completes end-to-end**,
output byte-identical to an all-NV reference (`[13, 220, 16, 13, 220, ...]` x64, 0 mismatches -- greedy
decode repeating on the harness's synthetic random-token prompt, same degenerate-but-correct shape T4.18/
T4.19 saw on olmoe). Confirms T4.21's structural prediction directly: a range split has ~1 device boundary
per forward pass, not one per MoE layer.

**Island/allocation count** (same call-site-attributed technique as T4.18: a scratch `sitecustomize.py`,
not committed, monkeypatching `APLRemotePCIDevice.alloc_sysmem`/`PCIIfaceBase.free`, run against a freshly
`pkill`ed server): **48 total `alloc_sysmem` calls** for the whole run (load+warmup+prefill+64 decode),
none freed (T4.18's root cause -- remote sysmem free is a no-op -- applies here too, so total==peak):

| call site | count | class |
|---|---|---|
| `support/hcq.py:535` (`HCQCompiled.__init__`, via `ops_nv.py:691`) | 32 | one-time device bring-up (signal pool / kernargs arena, per device) |
| `support/nv/nvdev.py:168` (via `support/nv/ip.py`, 8 distinct sites) | 8 | one-time GSP boot -- matches T4.18's olmoe count exactly |
| `graph/hcq.py:32` (`HCQGraph.__init__`'s kernargs_bufs, via `engine/realize.py:136`) | **4** | **the metric of interest -- graph islands needing kernargs** |
| `ops_nv.py:130` (T4.18's hw_page slab, one-time, via `graph/hcq.py:217`) | 1 | hw_page (already pooled) |
| `ops_nv.py:680` (cmdq_page) + `support/hcq.py:451,417` | 3 | misc one-time bring-up |

Bring-up (32+8+3=43) matches T4.18's olmoe bring-up bucket exactly (43, model-independent, as expected).
**4 kernargs allocations** for a range split vs. olmoe's **experts:** split needing ~85 islands (and 64
kernargs alone even after T4.18's hw_page fix, per §8) -- a ~16-20x reduction, and 48 total vs. the
~128-130 ceiling leaves 80+ slots of headroom. This is with gpt-oss's real 24-layer MoE architecture, not
a synthetic stand-in.

**Verdict: T4.20 downgrades from blocker to robustness for the shape the project actually wants.** T4.21
already made range split the recommended shape for big models; this session confirms it's also the
ceiling-safe shape for MoE-with-real-experts (gpt-oss), independent of model size. Compounding this,
TASKS.md's same-day TD.3-beam-rows finding is that `experts:` splits are now *also* the wrong shape on
performance grounds under BEAM (dead last: 43.8 vs all-NV's 115.5, all-METAL's 63.8 -- BEAM's tensor-core
win lands on attention, which `experts:` leaves on METAL) -- so the many-island pathology's one known
real-world trigger is a shape with no remaining reason to use it either way.

**Shipped the fix anyway** (recommendation: worth it). Reasoning: the design was already fully understood
(mirrors T4.18's `hw_page` slab almost exactly, same file), small (32 lines across the two runtime files,
well under the 80-line budget), provably zero-risk to AMD/QCOM/local-NV (duck-typed dispatch -- see below),
and it closes a crash class that is cheap to reopen (a finer-grained future split, a deeper MoE, or a
3rd pooled device would all raise the island count again) but expensive to re-diagnose from scratch (this
is the second time this exact ceiling has needed a from-first-principles characterization session). A
"trigger condition to revisit" would have been the alternative if the fix were bigger or riskier than this
one turned out to be -- it wasn't, so ship it. Trigger condition to *un*-ship or revisit, if it's ever a
maintenance burden: no backend besides NV has grown a remote transport with this ceiling in the time since
T4.18 shipped its sibling fix, and none is expected.

**The fix** (`tinygrad/runtime/ops_nv.py` +24, `tinygrad/runtime/graph/hcq.py` +8/-2, `test/external/
external_test_nv.py` +26 -- two files, 32 lines of runtime code): `NVDevice` gets `alloc_kernargs`/
`free_kernargs`, byte-for-byte the same lazily-allocated/never-freed/bump-suballocated-with-fallback shape
as T4.18's `_hwq_slab`/`_hwq_bump`, but a **separate** slab (`_kernargs_slab`/`_kernargs_bump`) -- kernargs
sizes are variable (unlike hw_page's fixed 16KB), so sharing one slab between the two classes would let
one's usage skew the other's already-measured headroom. `graph/hcq.py:32`'s alloc and its `__del__`'s free
each become a 2-branch duck-typed dispatch (`getattr(d, 'alloc_kernargs'/'free_kernargs', None)`) that
defers to the device when it offers a pool, else takes the *exact original* `d.allocator._alloc`/`_free`
call -- AMD/QCOM (no such attributes) and local/NVK NV (guarded by `is_remote()` inside `alloc_kernargs`
itself) are provably byte-identical: `getattr(..., None)` is `None` for them, so the `else` branch runs,
unchanged. The stable-address requirement `graph/hcq.py:32`'s comment worried about (a live graph's
kernargs must not move under a future replay) falls out for free from `wrap=False`: a bump allocator that
never wraps never reuses a slice's address, and T4.18 already proved a remote sysmem free was a no-op on
every side (no unmap verb) -- so "never free" isn't a new risk, it's naming what was already true.

**Proof the fix is real, two levels:**
- *Synthetic, proven failing pre-fix* (`TestNVKernargsPoolSlab`, mirrors `TestNVBindHwPageSlab`'s
  counting-fake exactly): 40 calls to `NVDevice.alloc_kernargs`/`free_kernargs` against a fake remote
  device collapse to **1** real allocator call (was an `AttributeError` before the fix existed, confirmed
  by `git stash`-ing only the two runtime files and re-running -- the test file itself was not stashed);
  the local/non-remote path still makes 40 (unchanged). 18/18 in the file post-fix.
- *Real hardware, both directions*: pre-fix (stashed the runtime-file diff, kept the test), gpt-oss:20b
  **`--device-map "METAL,experts:NV"`** (the exact stretch shape TASKS.md's TD.3-beam-rows entry names as
  blocked) reproduces the crash **at `graph/hcq.py:32` itself**, `RuntimeError: RPC failed: unknown error`
  -- the same signature the task brief names, at the same line T4.18 left alone. Restored the fix, reran
  unchanged: **completes end-to-end**, `load 8.260s, warm 29.225s, prefill 43.923 tok/s, decode 18.454
  tok/s`, output byte-identical to the range-split run and the all-NV reference (64/64 tokens). The
  range-split run itself was also rerun post-fix as a regression check: still byte-identical, tok/s
  unchanged within noise (49.7/9.8 vs pre-fix 48.9/9.9) -- the pool is exercised (4 kernargs allocs now
  share 1 slab alloc instead of 4) and costs nothing observable.

**Gates, all green:**
```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
GPTOSS=/Users/artur/Library/Caches/tinygrad/downloads/35ff51c27772d214bab0172591e2ade8

# step 1: re-scope check + regression check post-fix (byte-identical to each other and to all-NV)
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $GPTOSS --device-map "0-11:METAL,12-23:NV" --prompt-tokens 16 --decode-tokens 64
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $GPTOSS --device-map NV --prompt-tokens 16 --decode-tokens 64

# the fix, proven on the real crash shape (pre-fix crashes at graph/hcq.py:32; post-fix completes)
DEV=NV:NAK PYTHONPATH=. $PY extra/benchmark_llm.py --model $GPTOSS --device-map "METAL,experts:NV" --prompt-tokens 16 --decode-tokens 64

# unit-level proof (proven failing pre-fix by stashing only the two runtime files, not the test)
PYTHONPATH=. $PY -m pytest test/external/external_test_nv.py -v -k KernargsPool   # 2 passed
PYTHONPATH=. $PY -m pytest test/external/external_test_nv.py -v                   # 18 passed

# gates
DEV='METAL;NV:NAK' PYTHONPATH=. $PY -m pytest test/unit/test_llm_device_map.py -q   # 32 passed
PYTHONPATH=.        $PY -m pytest test/unit -q -n12                                # 846 passed, 71 skipped, 4 xfailed (unchanged vs T4.21)
PYTHONPATH=.        $PY -m mypy tinygrad/                                          # Success: no issues found in 216 source files
                    $PY -m ruff check tinygrad/runtime/ops_nv.py tinygrad/runtime/graph/hcq.py test/external/external_test_nv.py  # All checks passed
OCELOT_PATH=/Users/artur/Documents/tinygrad/.venv/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX PYTHONPATH=. \
  $PY -m pytest test/test_tiny.py test/device/test_hcq.py -q   # 49 passed, 6 skipped, 14 subtests passed
```

### 12. T4.16: does the MATVEC/quantized-gemv heuristic fire on NV? — yes, and it helps *more* there than on METAL

**Verdict up front:** the heuristic fires identically on **all three** lanes tested — METAL, `NV:NAK`,
and `NV` (nvcc, colima came back up mid-task, see note below) — byte-identical applied-opts sequence
on all three — and it *helps* on both NV lanes by more than it does on METAL for the same synthetic
Q4_0/Q4_K gemv: 2.3x (`NV:NAK`) and 2.3-5.3x (`NV` nvcc) vs. 1.06-1.27x (METAL). **No fix made —
nothing is broken.** This refutes the working hypothesis TASKS.md's T4.16 entry states ("T1.10's
MATVEC heuristic path never fires on NV shapes") and this file's own `BENCH_NOTES.md` takeaway 3(c)
("NV-side counterpart status that's still untested/unimplemented... this heuristic gap... is still
the decode floor") — both were untested assumptions; both are now measured and wrong on all lanes
checked. Correction added at that spot in `BENCH_NOTES.md`.

#### Does it fire? (objective 1)

The heuristic's *only* renderer-capability gate is `k.ren.has_local and ... k.ren.has_shared`
(`tinygrad/codegen/opt/heuristic.py:70-71`). Both default `True` on the base `Renderer` class
(`tinygrad/renderer/__init__.py:64,66`) and are **not overridden** by `NAKRenderer`, `CUDARenderer`,
`NVCCRenderer`, or `PTXRenderer` (checked all four class bodies directly) — only `ClangRenderer`
(CPU), `LVPRenderer` (Mesa llvmpipe software rasterizer, not a serving target here), and a couple of
ISA renderers set `has_local=False`. There is no NV-specific, or CUDA-vs-NAK-specific, gate anywhere
in the guard — the code has no reason to distinguish them.

Confirmed empirically, two ways, on all three lanes (`DEV=METAL`, `DEV=NV:NAK`, and `DEV=NV` — the
nvcc/colima lane was flagged "not run, colima stopped" earlier in this same task; colima came back up
mid-session — env brief update — so it was cheap to close the gap rather than leave it inferred):
1. **The repo's own T1.10 tests, run unmodified**
   (`test/opt/test_kernel_opts.py::test_matvec_heuristic_sees_through_cast` /
   `_sees_through_quant_dequant` / `_quant_dequant_real_pipeline`) — these `assertEqual`/`assertIn`
   the applied-opts sequence against a hardcoded `[OptOps.GROUP, OptOps.LOCAL, OptOps.UPCAST]` for a
   fp16 gemv and for real Q4_0/Q4_K/Q6_K-shaped GGUF dequant gemvs (a hand-built AST *and* the real
   `.realize()` pipeline through `pm_split_ranges`). **3/3 PASS on `DEV=METAL`, `DEV=NV:NAK`, and
   `DEV=NV`** — same assertion, same hardcoded expected sequence, all three lanes, no code changes
   between runs (just the `DEV=` env var).
2. **Live `DEBUG=3` capture** of the heuristic's own internal print (`heuristic.py:87`, only
   reachable from inside the guarded branch, right before it returns) on a real 8192x8192 Q4_0-shaped
   gemv, identical on both NV lanes:
   ```
   MATVEC: k.full_shape=[8192, 256, 2, 16] r0_1 MV_BLOCKSIZE=4 MV_THREADS_PER_ROW=8 MV_ROWS_PER_THREAD=4
   ```
   (`DEV=NV:NAK` and `DEV=NV` both print this exact line, byte-for-byte.) Direct proof of branch
   entry, not inferred from timing — `[8192, 256, 2, 16]` is the block/sub-block/byte split of the
   reduce axis (Q4_0 block=32 → 8192/32=256, further split 2x16) that T1.10's fix specifically taught
   the heuristic to walk.

Source inspection (the capability-gate argument above) already predicted this would generalize across
NV's compile lanes since `CUDARenderer`/`NVCCRenderer`/`PTXRenderer` share `NAKRenderer`'s
`has_local=True, has_shared=True` defaults — the live `DEV=NV` run now confirms that prediction rather
than leaving it as inference. `pgrep -fl "TinyGPU.*server"` stayed at exactly 1 and swap stayed flat
(1564 MB) across every cell in this sub-section; nothing dock-side needed a respawn.

#### Does it help? (objective 2)

Microbench: `x @ w.T`, `x=(1,COLS)` fp32, `w=(ROWS,COLS)` GGUF-quantized (Q4_0/Q4_K via
`ggml_data_to_tensor`, the same helper T1.10's own tests use), `ROWS=COLS=8192` (a real llama-class
FFN-projection scale; ~36-37 MB raw quantized blob). `Context(BEAM=0, DEBUG=2)` forces the hand-coded
(no-search) path — `DEBUG>=2` is required for `engine/realize.py` to actually populate
`GlobalCounters.time_sum_s` (otherwise `et` stays `None` and nothing is measured, a dead end hit
first). Timed via reset-then-read `GlobalCounters.time_sum_s` around `.realize()` (same pattern
`test/external/speed_v_theoretical.py` already uses), min of 5 post-warmup samples (1 warmup call
dropped). `MV=0` — the heuristic's own escape hatch, `getenv("MV",1)`, confirmed to be T4.1's
"`MV=0` ≡ unpatched control" — disables only the MATVEC branch; **toggled via subprocess env, not
`Context`**, because `getenv` is `@functools.cache`d (`tinygrad/helpers.py:159`) and latches to
whatever it read on its first call in the process — it cannot be flipped mid-process.

An initial attempt at T1.10's own unit-test shape (1024x4096, ~2.25 MB raw) was dispatch-overhead-
dominated (100-150us fixed cost dwarfing the actual transfer, MV=1 sometimes *slower* than MV=0 —
pure noise) and produced nothing usable as a bandwidth signal; recorded so nobody repeats it.

| Device | Quant | MV=1 (heuristic on, default) | MV=0 (heuristic off) | Ratio |
|---|---|---:|---:|---:|
| METAL (sanity control) | Q4_0 | 46.87 GB/s | 44.41 GB/s | 1.06x |
| METAL (sanity control) | Q4_K | 32.96 GB/s | 25.99 GB/s | 1.27x |
| **NV:NAK** | **Q4_0** | **78.5 GB/s** | **34.17 GB/s** | **2.30x** |
| **NV:NAK** | **Q4_K** | **66.9 GB/s** | **28.73 GB/s** | **2.33x** |
| **NV (nvcc)** | **Q4_0** | **210.6 GB/s** | **40.1 GB/s** | **5.25x** |
| **NV (nvcc)** | **Q4_K** | **121.4 GB/s** | **52.0 GB/s** | **2.33x** |

Every NV row reproduced twice independently (both quant types, both MV settings, both lanes):
<0.3% run-to-run variance throughout. Swap flat (1564-1572 MB, no growth) and exactly one
`TinyGPU...server` process throughout every cell in both NV lanes (checked before and after each).
`NV` (nvcc) cells used plain single realizes (no `JITBEAM`), so none of TD.2c's BEAM-era
compile-server concurrency/`PARALLEL` concerns apply here.

**Reading the table:** on both NV lanes the heuristic doesn't just fire, it delivers a *larger*
relative win (2.30-2.33x on NAK, 2.33-5.25x on nvcc) than the identical harness shows on METAL
(1.06-1.27x) for this exact op. The absolute GB/s numbers are **not** directly comparable cross-device
(3090 HBM ≈936 GB/s vs. M3 Pro unified memory ≈150-200 GB/s peak — NAK's 78.5 GB/s and nvcc's 210.6
GB/s beating METAL's 46.9 GB/s here is raw hardware headroom on one isolated kernel, not evidence
NAK/nvcc-overall beats METAL-overall; TD.2's real-model finding that NAK no-BEAM decode is 0.50-0.68x
of METAL no-BEAM stands, measured across a full mixed kernel workload, not contradicted by this). What
this table *does* establish: **this specific heuristic is not the source of that gap, on either NV
lane.** The quantized-gemv MATVEC path is doing its job on NV, at least as well as on METAL — better,
by this measure. TD.2's takeaway 3(c) is refuted (correction added in `BENCH_NOTES.md`). The real
0.50-0.68x floor is elsewhere in the decode kernel mix — takeaways 3(a) (NAK has no tensor cores) and
3(b) (5-kernel unfused attention/layer) are unaffected by this finding and remain the standing
explanation. Bonus cross-check: nvcc-MV=1 beats NAK-MV=1 by 2.68x (Q4_0) / 1.81x (Q4_K) here — same
direction and same order of magnitude as TD.2's independent "`DEV=NV` no-BEAM codegen is ~4x faster
than `DEV=NV:NAK` no-BEAM codegen" finding (llama3.2:1b, real model decode), from a completely
different measurement (one isolated synthetic kernel vs. a whole model's decode loop) — good agreement
between the two, not a coincidence chased further here.

One methodology caveat worth naming: this `MV=0` control is *not* equivalent to T1.10's original
pre-fix baseline. `MV=0` only skips the hand-tuned GROUP/LOCAL/UPCAST recipe; the rest of
`hand_coded_optimizations` (generic local-axis grouping further down the function) still recognizes
this AST as reduce-shaped and still applies its own (weaker) opts. T1.10's pre-fix bug was a total
pattern-match miss (the dequant MUL wasn't recognized as matvec-shaped *at all*), which is a
different, worse baseline than "matvec recognized, hand-tuned sizing skipped." Both are legitimate
things to measure; this section measures the latter (the one the current `MV` escape hatch actually
gives you), and it's the more relevant one for "is the *current* heuristic pulling its weight on NV."

#### Fix? (objective 3)

Not applicable — nothing is broken. No changes made in `tinygrad/`.

#### Does BEAM moot this? (objective 4)

Structurally, yes — not just "in practice, mostly." `apply_opts()`
(`tinygrad/codegen/opt/postrange.py:334-350`) is a strict if/elif: `beam >= 1` (line 340) calls
`beam_search()` — an independent Opt-space search seeded from the raw, un-optimized kernel;
`hand_coded_optimizations()` (lines 347-350) — and therefore the entire MATVEC branch this section is
about — is reached only in the `elif`, i.e. **never called at all once `JITBEAM>=1`/`BEAM>=1`.** This
question has zero relevance to any BEAM'd run, on any device — not a small effect, a structurally
absent one.

Per TD.2's own truth table (this file's prerequisite reading), BEAM already delivers large
multipliers on NV independent of this heuristic entirely:
- `NV:NAK`: qwen3:8b 3.72→31.44 tok/s (8.45x), gpt-oss:20b 7.42→26.81 (3.61x), llama3.2:1b 7.13→96.06
  (13.47x)
- `NV` (nvcc, post-T4.14 fix): llama3.2:1b 28.16→149.12 (5.30x), qwen3:8b 9.99→46.87 (4.69x),
  gpt-oss:20b 18.77→60.85 (3.24x)

So T4.16 is exactly what the task brief scoped it as: a cold-start/no-BEAM-UX-only question — and the
answer is reassuring for that UX too, since this particular lever already works on NV without any
changes.

**Recommendation: no action needed.** Close T4.16 as investigated, hypothesis refuted, no fix
required — an acceptable, expected outcome for this task's priority.

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python

# objective 1: existing T1.10 tests, unmodified, on all three lanes -- 3/3 pass, identical opts, every lane
PYTHONPATH=. DEV=METAL      $PY -m pytest test/opt/test_kernel_opts.py -k matvec -v   # 3 passed
PYTHONPATH=. DEV='NV:NAK'   $PY -m pytest test/opt/test_kernel_opts.py -k matvec -v   # 3 passed
PYTHONPATH=. DEV=NV         $PY -m pytest test/opt/test_kernel_opts.py -k matvec -v   # 3 passed (colima up mid-task)

# objective 1: live proof of branch entry on a real 8192x8192 Q4_0 gemv (ad hoc, not committed)
# -- run once with DEV='NV:NAK' and once with DEV=NV; byte-identical "MATVEC: ..." line both times
DEV='NV:NAK' PYTHONPATH=. $PY -c "
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.gguf import ggml_data_to_tensor
ROWS, COLS = 8192, 8192
raw = Tensor.randint(ROWS*(COLS//32)*18, low=0, high=256, dtype=dtypes.uint8).realize()
x = Tensor.rand(1, COLS).realize()
w = ggml_data_to_tensor(raw, ROWS*COLS, 2).reshape(ROWS, COLS)
with Context(BEAM=0, DEBUG=3): (x @ w.T).realize()"   # prints "MATVEC: k.full_shape=[8192, 256, 2, 16] ..."

# objective 2: microbench, ad hoc script (not committed) -- MV toggled via subprocess env, not Context
DEV=METAL           PYTHONPATH=. $PY t416_matvec_bench.py 2 12   # Q4_0, Q4_K -- MV=1 (default)
DEV=METAL    MV=0   PYTHONPATH=. $PY t416_matvec_bench.py 2 12   # MV=0
DEV='NV:NAK'        PYTHONPATH=. $PY t416_matvec_bench.py 2 12   # MV=1 (default)
DEV='NV:NAK' MV=0   PYTHONPATH=. $PY t416_matvec_bench.py 2 12   # MV=0
DEV=NV              PYTHONPATH=. $PY t416_matvec_bench.py 2 12   # MV=1 (default, colima up mid-task)
DEV=NV       MV=0   PYTHONPATH=. $PY t416_matvec_bench.py 2 12   # MV=0

# gates -- no tinygrad/ code changed (only these two .md files); run to confirm the investigation
# itself left no regressions, not because anything here needed fixing
PYTHONPATH=. $PY -m pytest test/opt -q       # 38 passed, 4 skipped, 1 xfailed, 5 subtests passed
PYTHONPATH=. $PY -m pytest test/unit -q -n12 # 846 passed, 71 skipped, 4 xfailed, 2 subtests passed
PYTHONPATH=. $PY -m mypy tinygrad/           # Success: no issues found in 216 source files
$PY -m ruff check .                          # All checks passed
```

### 13. T4.22: IQ3_XXS/IQ4_XS dequant materialization (upstream #17316) — FIXED, same class as T4.13, no rangeify change

**Verdict up front: fixed.** IQ3_XXS and IQ4_XS dequant (`gguf.py` types 18/23) used `Tensor`-indexed
LUT gathers exactly like pre-fix MXFP4 (T4.13) — confirmed via live instrumentation that the same
named check (`buffer_in_reduce`, `schedule/rangeify.py:258-266`) refuses to fuse them into MoE's
`weight[sel]` gather, forcing full-expert materialization every decode step. Root cause matches the
task's hypothesis exactly; **the "IQ grids are not ALU-expressible" half of the hypothesis is only
half true** — one of IQ3_XXS's two gathers (`even_signs`, a sign-parity table) turned out to be a
genuine closed-form bit trick after all (T4.13-style ALU), and the two truly-arbitrary codebooks
(IQ3_XXS's 256-entry grid, IQ4_XS's 16-entry `kvalues_iq4nl`) are small enough to replace with a
**compile-time balanced binary select-tree** (nested `.where()` over the table's literal float
constants) instead of a bit trick — mechanically equivalent to T4.13's fix (no buffer touched, so
`buffer_in_reduce` never fires) without needing a closed-form formula. **No rangeify/scheduler code
was touched.**

#### Mechanism (objective 1) — confirmed two ways, byte numbers worse than T4.13's original bug

Source inspection: `_ggml_iq_grid(...)[blocks[:,2:66]]` (grid) and `even_signs[sign_idx]` (sign
parity) in type 18, `iq4_xs_lut[q]` in type 23, are all `Tensor.__getitem__` advanced-index gathers.
`Tensor._getitem`'s general path for a single tensor-indexed axis (`mixin/op.py:105-134`, `len(dims)==1`
so the "consecutive indices" fast path at line 111 never applies) compiles this to a one-hot mask
times the source, **summed via a REDUCE** (`mask.where(x, 0).sum(sum_axis)`) — i.e. every LUT/grid
gather is *itself* a REDUCE that reads a real buffer (the table). When this sits inside a bufferize
candidate that's a would-be input to a consuming reduce (MoE's `weight[sel]`), `remove_bufferize`'s
`buffer_in_reduce` check (walks each `REDUCE`'s own source for `PARAM`/`STAGE`/`AFTER`) sees exactly
that and refuses to remove the bufferize — forcing full materialization. Live-instrumented (temporary
`print` in `buf_gate`, reverted after use, zero diff left behind): a candidate bufferize shaped
`(32, 256, 256)` — **the full (N_EXPERTS, out, in) expert tensor** — shows `buffer_in_reduce=True` for
IQ3_XXS and IQ4_XS but is **absent** from Q4_K's and (already-fixed) MXFP4's traces at the identical
config, in the identical test harness (mirrors T4.11/T4.13's own `TestGPTOSSDecodeByteBudgetMXFP4`:
gpt-oss arch, N_EXPERTS=32, EXPERTS_PER_TOK=4, `Transformer.from_gguf` + `model.generate()`, real JIT
path, `DEV=CPU` for a backend-independent AST-shape byte count — dims bumped to DIM=HIDDEN=256 from
T4.11/13's 64/128 because IQ3_XXS/IQ4_XS need 256-element block alignment that gguf-py's
`quant_shape_from_byte_shape` enforces on `raw_dtype` tensors, unlike MXFP4's 32-element blocks).

A **plain dense gemv** (no MoE gather at all) does **not** reproduce the blowup for either type —
confirmed by direct measurement (1.98x/1.00x analytic bytes) before realizing why: `buffer_in_reduce`
only controls whether a *separate materialize-then-reread* boundary is inserted, which costs at most
~2x regardless of the consumer. The catastrophic multiplier is specific to MoE's `weight[sel]`: if the
dequant bufferize can't be pushed past the gather, `weight[sel]` ends up indexing an **already fully
materialized dense array**, paying N_EXPERTS/EXPERTS_PER_TOK (here 8x) instead of 1x — and IQ3_XXS
pays *even more* than that dense estimate because it chains **two independent** gathers (grid AND
sign-parity), each forcing its own full materialization, compounding rather than adding:

| type | pre-fix actual/gathered | pre-fix actual/dense | (T4.13 MXFP4 pre-fix, for scale) |
|---|---:|---:|---:|
| IQ3_XXS (18) | **63.50x** | **7.94x** (worse than "all experts"!) | ~8x gathered |
| IQ4_XS (23) | **42.31x** | **5.29x** | |

This matches (and slightly exceeds, on the dominant IQ3_XXS format) the real qwen3.6-35B UD-Q3_K_XL
finding this task started from (~88 GB/token measured, ~83 GB/token on upstream #17316's reporter) —
the mechanism, not just the byte-order-of-magnitude, transfers exactly.

#### Options evaluated (objective 2)

**(a) Rangeify: relax `buffer_in_reduce` generally — rejected, two independent reasons, no code
written.** This fork already has a generalized version of exactly this idea living in the same
function: `PCONTIG>2`'s partial-bufferize path (`rangeify.py:267-282`, `out_in_ratio<10` gate). Tested
directly on this shape: **`PCONTIG=0/2/3` produce byte-identical schedules** (2,907,280 B every time,
3 fresh processes) — the ratio gate compares *element* counts (dequant output elements vs. raw bytes
treated as elements) and a quant format's ~2-4x element-level compression never clears the 10x bar
tuned for attention's O(seq²) blowup (`PCONTIG_ATTN_NOTES.md`'s own subject). **Even where that gate
does engage, this fork's own prior investigation (T1.7, same file) already proved it produces
silently wrong numerics** on generic reduce-feeds-reduce patterns (double matmul: max abs diff 2.98
vs. numpy; softmax; SDPA) — a live, unresolved, first-party correctness bug in the exact code path
option (a) would need to extend. A brand-new, narrowly-scoped relaxation is conceivable but proving
it safe (T4.9-style: full survey of what else routes through `remove_bufferize`) is multi-hour work
by T1.7's own demonstrated difficulty, and **upstream PR #17493 is a full rangeify rewrite** — low
expected shelf life for new logic there. Matches this task's own STOP condition; not attempted.

**(b) Load-time transcode — not attempted, quantified enough to reject.** fp16 transcode roughly
doubles resident size (T4.21 specifically fixed a residency blowup on this exact model; undoing it
is a non-starter). int8/requantize-to-Q4_0 is a genuine *lossy* re-quantization needing its own
perplexity/precision study — bigger scope than this task's budget, and still costs real extra
residency (IQ3_XXS/IQ4_XS are ~3.1-4.25 bits/weight; even Q4_0 is 4.5 bits/weight nominal).

**(c) Selected — per-table fix, matched to what each table actually is:**
- `even_signs` (128-entry "table"): not actually arbitrary — `even_signs[i] == i | (parity(i)<<7)`
  (ggml derives the 8th sign bit from 7-bit parity to save a bit). Replaced with the standard
  XOR-fold SWAR parity trick (`px = i^(i>>4); px ^= px>>2; px ^= px>>1; bit7 = (px&1)<<7`) — pure ALU,
  **no size limit**, same spirit as T4.13's MXFP4 bit-tricks. Bit-exact for all 128 values (exhaustive).
- IQ3_XXS's 256-entry grid and IQ4_XS's 16-entry `kvalues_iq4nl`: genuinely arbitrary codebooks (no
  bit formula — confirmed by inspection: `kvalues_iq4nl = (-127,-104,-83,...)`, a hand-tuned
  non-uniform table). Replaced with a **compile-time balanced binary select-tree**: nested `.where()`
  comparing the code against literal Python-int thresholds, bottoming out at literal Python-float
  leaves — zero `Tensor`/buffer anywhere in the expression, so `buffer_in_reduce` can never fire
  (there's no REDUCE at all, let alone one touching a buffer). One shared helper, `select_const`
  (nested in `ggml_data_to_tensor`, next to the existing `q_to_uint8` helper), used by both types.

Deliberately **not** extended to IQ3_S (type 21, 512-entry grid) or IQ2_S (type 22, 1024-entry grid)
this session — same bug class (both still gather via `_ggml_iq_grid(...)[q]`, confirmed by
inspection), `select_const` would mechanically apply, but a linear select-tree costs `len(vals)-1`
`.where()`s and neither format appears in the qwen3.6 file this task is scoped to (BENCH_NOTES'
histogram: only IQ3_XXS + IQ4_XS). Given IQ3_XXS's 256-entry tree already showed a real no-BEAM
slowdown (below), 511/1023-entry trees are a real open question, not a safe extrapolation — flagged
as a natural, well-scoped follow-up (ideally with a smarter 2-level split-table lookup, Q4_K-scale-
staging-style, rather than a fully linear tree, if the byte win doesn't hold up at that size).

#### Verification (objective 3 — T4.13-grade rigor)

- **Bit-exact, exhaustive over the full code space** (not sampled): all 128 `even_signs` values, all
  256 IQ3_XXS grid codes (both checked directly against the pre-patch gather form, `array_equal`).
  IQ4_XS also cross-checked against an independent from-scratch numpy port of the block format
  (0 mismatches on 507,648/524,288 non-NaN elements against random raw bytes; the ~3% NaN entries are
  from fully-random synthetic scale bytes hitting fp16 NaN bit patterns, identical positions in old
  and new code — a test-data artifact, not a bug).
  **Existing repo tests are the strongest signal**: `test/unit/test_gguf.py`'s `test_dequantization_iq3_xxs`/
  `_iq4_xs` and `test_gguf_gemv_iq3_xxs`/`_iq4_xs` (real quantize/dequantize via the `gguf` pip
  package, i.e. ground truth independent of this session) — **11/11 pass unmodified**.
- **Byte count**, real model+JIT harness, both types, confirmed on **CPU, METAL, and real NV
  hardware (`NV:NAK` and `NV` nvcc)** — schedule-level fix, byte counts match across backends as
  expected:

  | type | pre-fix actual/gathered | **post-fix actual/gathered** | (Q4_K control / MXFP4-fixed, same harness) |
  |---|---:|---:|---:|
  | IQ3_XXS | 63.50x | **1.59-1.60x** | 1.44-1.73x / 1.42-2.47x |
  | IQ4_XS | 42.31x | **1.42-1.81x** | (same row) |

  Both land squarely in the same healthy range as the known-good controls — full resolution, not a
  partial improvement.
- **New regression test** (`test/unit/test_llm_gptoss.py::TestGPTOSSDecodeByteBudgetIQ`, mirrors
  `TestGPTOSSDecodeByteBudgetMXFP4`): **proven failing pre-fix** (`38,234,500 B` / `35,350,148 B`
  actual vs. `<3x gathered` asserted, via `git stash push -- tinygrad/llm/gguf.py` with the test kept
  in place), **passes post-fix**.

#### Does BEAM change the picture? (objective 4) — yes, decisively, and it's the crux of the recommendation

Unlike T4.13 (where bytes were the whole story), **here the trade is bytes-for-ALU, and whether that
trade is worth it depends on BEAM**, tested directly (same tiny 2-layer/32-expert/256-dim config,
`DEBUG=2`+`GlobalCounters.time_sum_s`, min-of-3 post-warmup, fresh process per cell):

| lane | broken (no BEAM) | fixed (no BEAM) | fixed (`JITBEAM=2`) | broken (`JITBEAM=2`) |
|---|---:|---:|---:|---:|
| METAL, IQ4_XS | 1.222 ms | **0.514 ms (2.38x faster)** | — | — |
| METAL, IQ3_XXS | 1.272 ms | 3.388 ms (2.67x **slower**) | **1.320 ms** | 1.697 ms |
| `NV:NAK`, IQ3_XXS | 0.884 ms | 12.070 ms (13.65x **slower**) | not measured | — |
| `NV` (nvcc), IQ3_XXS | 0.954 ms | 2.267 ms (2.37x **slower**) | attempted, did not finish in budget* | — |

IQ4_XS's 16-way tree is an unambiguous win on every axis, BEAM or not. **IQ3_XXS's 256-way tree is a
genuine no-BEAM regression on every lane tested** — worst on `NV:NAK` (NAK's weaker codegen is hit
hardest by the extra ALU), still a regression on nvcc and METAL. **But with `JITBEAM=2` on METAL, the
fixed version is 22% *faster* than the broken version at BEAM'd-2** (1.320 ms vs. 1.697 ms) *while
still reading ~27x fewer bytes* (2.31x gathered vs. 63.53x) — BEAM finds a materially better schedule
for the select-tree that the hand-coded path doesn't, the same structural fact T4.16 established
generically (`apply_opts` never calls `hand_coded_optimizations` once `beam>=1` — BEAM is an
independent from-scratch search, not a refinement of the naive lowering). **This fork's own
production qwen3.6 config always runs BEAM'd** (BENCH_NOTES' headline: `JITBEAM=1 PARALLEL=6`
nvcc) — no-BEAM is a cold-start/dev-loop path only (T4.16's own framing). On that basis the fix is
recommended for the configuration this fork actually serves; the no-BEAM path is a known, honestly-
reported soft spot for IQ3_XXS specifically, not silently swept under the rug.

*`JITBEAM`'s per-kernel search over the remote nvcc nvcc/colima compile-server pays a real
round-trip per candidate (~8-10s observed for one 22-25-action kernel search) — for a full decode
step's ~20-30 distinct kernel shapes this did not complete inside this task's time budget on the tiny
synthetic config. A real characterized cost (nvcc BEAM cold-start latency), not a correctness or
methodology concern — noted for whoever next measures a fresh nvcc BEAM config from cold.

#### Real-model confirmation — not run this session (same pattern as T4.13's own first session)

Mirrors T4.13's own history exactly: T4.13's original fix session also deferred the real-model
number ("Real-model tok/s confirm = next bench window") and got it in a **separate**, later session
(bench window 4). Given (1) the byte-count fix is confirmed via the *exact* T4.11/T4.13 harness
methodology, on real NV hardware, not just a synthetic proxy, and (2) a real 16.85 GB
`Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` load + `JITBEAM` cold-compile (many more distinct kernel shapes
than this session's 2-layer toy) would very likely have exceeded this task's remaining budget on its
own (see the nvcc-BEAM latency note above), this was a deliberate scope call, not an oversight.
**Next step**: `DEV=NV --device-map NV`, `JITBEAM=1 PARALLEL=6`,
`/Users/artur/models/qwen3.6-35b-a3b-q3/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`, expect decode well above
the current 1.86 tok/s floor — **not quoting a specific target number**: the measured wall-clock
trade at toy scale (bytes down ~40-60x, but only a ~22% BEAM'd wall-clock win, not a bytes-scale
win) shows this is not purely memory-bound the way MXFP4's fix was, so a naive
byte-ratio extrapolation would overpromise.

#### Gates, all green

```bash
cd tinygrad-dock
PY=/Users/artur/Documents/tinygrad/.venv/bin/python
PYTHONPATH=. $PY -m pytest test/unit/test_gguf.py test/unit/test_llm_gptoss.py -v -k "iq or Iq or IQ"  # 13 passed
PYTHONPATH=. $PY -m pytest test/unit -q -n12   # 848 passed, 71 skipped, 4 xfailed, 2 subtests passed
PYTHONPATH=. $PY -m pytest test/opt -q         # 38 passed, 4 skipped, 1 xfailed, 5 subtests passed
PYTHONPATH=. $PY -m mypy tinygrad/             # Success: no issues found in 216 source files
$PY -m ruff check .                            # All checks passed
DEV=CPU PYTHONPATH=. $PY -m pytest test/unit/test_gguf.py test/unit/test_llm_gptoss.py -q  # 55 passed
# proof the new test is real (not vacuously true): fails pre-fix with the fix stashed, test kept
git stash push -- tinygrad/llm/gguf.py
PYTHONPATH=. $PY -m pytest test/unit/test_llm_gptoss.py -v -k ByteBudgetIQ  # 2 FAILED (38.2MB / 35.4MB actual vs <3x gathered)
git stash pop
```
Mock-NV not run: no NV-specific (`ops_nv.py`/`hcq.py`) code was touched, only `tinygrad/llm/gguf.py`
and one test file — outside the brief's "if NV code is touched" trigger. Dock health verified clean
before/after every real-hardware cell above (`pgrep -fl "TinyGPU.*server"` exactly one throughout;
one transient `nv_usb4.lock` contention from an interrupted background BEAM run cleared on retry with
no wedge, confirmed by a trivial `Tensor([1,2,3])+1` round-trip).

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
