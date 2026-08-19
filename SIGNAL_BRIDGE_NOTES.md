# T3.6 — async signal bridge: capture-op analysis + eager-prototype measurements

Scoped spike, branch `task/T3.6-signal-bridge` off `integration/wave1`. Replaces T3.4's refuted
zero-copy-alone approach (aliasing removed the memcpy but bought nothing — T3.4 isolated the fixed
per-hop cost as *synchronization*, not the copy). This spike asks: does converting the host-blocking
drain into a GPU-side dependency edge (`MTLSharedEvent` `waitForEvent`/`signalEvent`, already used
METAL↔METAL in `ops_metal.py`'s `_transfer`) actually save wall-clock time, and what would it take to
make that JIT-capturable?

**Verdict up front:** capture-op integration sized at 100–150+ lines of scheduler/JIT surgery across
4 files — over the task's ~80-line STOP budget, not built. The eager prototype (CPU-producer /
METAL-consumer, exactly as scoped) shows **no win — a small, consistent net loss** (~20–35 µs) versus
`Device.synchronize()`, for reasons the measurements below pin down precisely. Nothing wired into the
split-model pipeline (objective 3's gate — "only if (2) shows a real win" — was not met). This is a
clean refutation, in the same family as T3.4/T1.4/T4.10.

## Part 1 — capture-op analysis (decision-critical, done first as instructed)

### Where a cross-backend COPY lives today in a captured graph

T3.2's finding, re-confirmed by reading the code directly: a cross-device `Ops.COPY` is a **standalone
call**, never part of any graphed batch. `graph_split_rewrite` (`tinygrad/engine/jit.py:32-59`) walks
the linear call sequence and groups only calls where `graph_t.supports_uop(...)` returns true.
`GraphRunner.supports_uop` (`engine/jit.py:150-152`) requires `Ops.PROGRAM`; `MultiGraphRunner`
(`engine/jit.py:155-159`, used by `HCQGraph`) additionally allows `Ops.COPY` but only when all
devices are the *same HCQ type* — never applicable to METAL, whose graph is plain `GraphRunner`
(`MetalGraph`, `runtime/graph/metal.py:10`). So a METAL↔CPU boundary COPY always breaks out of any
graph batch into its own `exec_copy` call (`engine/realize.py:157-168`), executed between two
per-backend graph islands — exactly what T3.2 measured.

### How `HCQGraph` handles cross-device waits WITHIN the HCQ world

`runtime/graph/hcq.py` pre-plans the **entire** multi-device dependency graph as one object that owns
every involved device's hardware queues (`self.comp_queues`, `self.copy_queues`, one per device/queue
pair). Cross-device ordering is a **first-class HW-queue instruction**: `enqueue_queue.wait(sig, val)`
(line 168), where `sig` is a per-device timeline signal and `val` is a **rebindable `UOp.variable`**
(`timeline_sig_<dev>`, `timeline_var_<dev>`, lines 154-156). At `__call__` (replay) time, these
variables are re-resolved fresh from live device state (`dev.timeline_value - 1`, line 276;
`dev.signal_t` addresses, line 277) via `hcq_var_vals`, and fed into the pre-built queue instruction
stream. This works because HCQGraph owns *all* the queues involved and can bake the wait as a native
instruction in the consuming device's own hardware queue ahead of time.

### How `MetalGraph` handles dependencies — it doesn't

`runtime/graph/metal.py` is a **single-device** `MTLIndirectCommandBuffer`
(`MTLIndirectCommandTypeConcurrentDispatch`) with purely intra-buffer ordering via `setBarrier()`
(line 48). There is no cross-device or cross-call wait primitive anywhere in it — no analog to
HCQGraph's signal-variable rebinding. Each `__call__` builds and commits exactly **one** fresh
`command_buffer` (line 75) for the whole batch.

### The favorable structural fact this analysis turned up

Stage A's mixed METAL/CPU trace is **not** one HCQGraph-style pre-scheduled object. It's
`run_linear`'s flat, freshly-Python-interpreted-every-replay call sequence (`CapturedJit.__call__` →
`run_linear`, `engine/jit.py:184-188`), where each call type dispatches through a small
pattern-matched `exec_*` function (`engine/realize.py:259`) that receives a live `ExecContext`. Only
the METAL portion is wrapped in a sub-object (`Ops.CUSTOM_FUNCTION arg="graph"`, one call node). This
means a **new standalone call type** sitting between the CPU producer's calls and the METAL graph's
call in the linear sequence can read live signal state in Python **at each replay** with no
capture-time value-freezing problem — much simpler than replicating HCQGraph's ahead-of-time
machinery. Confirmed concretely: `HCQ2Compiled.signal("timeline")` / `signal("value",1)` +
`_wait_signal` (`support/hcq2.py:598-615`) are already the exact host-visible counters and polling
loop `Device["CPU"].synchronize()` uses — a hypothetical `exec_wait_signal` could just read them live,
no new bookkeeping needed for *that* part.

### New call type, or a property on COPY?

**New standalone call type**, not a COPY property. The wait must gate the *next* graph batch's own
command buffer — a different call node in the linear sequence than wherever the wait would be
scheduled. A property on `Ops.COPY` can't reach forward into a sibling `Ops.CUSTOM_FUNCTION arg="graph"`
call's own `MetalGraph.__call__` invocation without the same cross-call plumbing a standalone op needs
anyway — so COPY-as-property buys nothing and is a worse fit conceptually (COPY means "move data";
here, with T3.4's aliasing primitive, there is no data to move for the CPU→METAL direction — only a
wait is needed. Making a no-op copy carry a wait obligation is a stranger data model than a `wait_signal`
call that simply supersedes the COPY node for that direction).

### Sizing (why this STOPs)

| Piece | File | Est. lines | Why |
|---|---|---|---|
| Carve-out in `assert_all_same_devices` | `schedule/__init__.py:149-151,179` | ~10-15 | a wait op inherently references 2 devices (consumer + producer signal), same shape as the existing `copy_kernel_to_copy_uop` carve-out |
| New `exec_wait_signal` + `ExecContext` field to hand a pending `(event, value)` to the next graph call | `engine/realize.py` | ~15-20 | small in isolation — reads live CPU signal state, no blocking |
| `GraphRunner.__call__` interface change so `MetalGraph` can receive+consume the pending wait before `commit()` | `engine/jit.py` + `runtime/graph/metal.py` (+ `runtime/graph/hcq.py` to keep the shared interface consistent) | ~25-35 | touches a shared base-class signature implemented by two very different graph runners |
| Watcher-thread subsystem: lifecycle (start/stop with JIT capture/replay/teardown), single-vs-multi-slot safety, thread-safety of concurrent replays | new module | ~30-50 | genuinely new subsystem, not a small extension of anything existing |
| Block-boundary seam rework in `llm/model.py` to replace the CPU→METAL COPY with alias+wait for that direction only (mirrors T3.4's already-built, never-merged buffer-identity registry) | `llm/model.py` | ~30-40 (reuse of T3.4's registry, adapted) | needed so there's an aliased buffer for the wait to gate in the first place |

Total ≈ **110–160 lines**, spread across scheduler, JIT replay, two graph runners, and the LLM app —
clears the task's "STOP before any scheduler surgery >~80 lines" bar by a wide margin, and the pieces
are not independently choppable (the watcher subsystem and the `ExecContext` plumbing are both
required for even the smallest useful version). **Recommendation: do not build the capturable version.**
This matches T4.8's own sizing precedent (warp-reduce, ~300-550 lines, "scoped, deferred").

## Part 2 — eager prototype (measured, outside the JIT)

File: `test/unit/test_signal_bridge_metal.py` (gated `skipUnless(Device.DEFAULT=="METAL")`, zero
production-code changes — the whole spike is additive). Design combines two already-proven primitives
rather than inventing anything:

- **T3.4's aliasing primitive** (the only safe direction: METAL owns the allocation, CPU borrows its
  host pointer via `Tensor.from_blob` — Metal's `external_ptr` re-interprets the int as an existing
  objc id, so a raw CPU pointer there is unsafe).
- **`MTLSharedEvent`** `encodeWaitForEvent:value:` on the consumer's command buffer before `commit()`
  (identical mechanism to `ops_metal.py`'s `_transfer`, extended to a dedicated bridge event).
  `setSignaledValue:`/`signaledValue` aren't in the Metal autogen `_methods_` list for
  `MTLSharedEvent` — called via the same raw `objc.msg(...)` pattern `ops_metal.py` already uses for
  `to_ns_str`.
- **A background watcher thread** that calls the real `Device["CPU"].synchronize()` (not a hand-rolled
  poll — CPU's worker-ring + generic HCQ2 timeline interaction is nontrivial enough that reusing the
  already-correct implementation beats guessing at it) and then flips the event.

4 correctness tests pass (proving the wait genuinely gates visibility — including a race-sensitive
variant with a chained 8-op CPU write, and an event-value-increments-across-iterations test guarding
against a stale-value false pass) + a benchmark harness (`bench()`, `--bench` flag).

### Latency table (interleaved baseline/bridge, n=256 f32 elements, 100 reps, 10 warmup, min-of-run not cherry-picked)

| Metric | baseline (`Device["CPU"].synchronize()`) | bridge (watcher + `waitForEvent`) | delta |
|---|---|---|---|
| submit (producer-issue → consumer command buffer committed) | median 320.3 µs / min 302.6 µs | median 355.7 µs / min 335.3 µs | **bridge +35 µs (worse)** |
| total (producer-issue → consumer result ready) | median 473.6 µs / min 453.5 µs | median 492.4 µs / min 466.1 µs | **bridge +19 µs (worse)** |

Non-interleaved run (blocked, 50 reps) showed the same direction and magnitude — this is not
measurement-order noise.

### Component breakdown (why)

| Component | Cost |
|---|---|
| `encodeWaitForEvent_value` + commit vs plain commit | ~1.4 µs (13.2 → 14.6 µs median) — negligible |
| bare `threading.Event` arm→watcher-wakeup round trip | ~3.8 µs median — negligible |
| **CPU dispatch alone** (`.assign().realize()`, no sync at all) | **277-300 µs** |
| CPU dispatch + `Device["CPU"].synchronize()` | 280-303 µs — **barely more than dispatch alone** |
| blit commit + wait_check, isolated | 156-254 µs (GPU dispatch/driver, not device-specific to this bridge) |
| 20,000-iteration Python loop, with vs without a concurrent watcher `synchronize()` in flight | 932 µs vs 980 µs — **no GIL-contention penalty found** (ruled out as the explanation) |

Two things fall out of this: (1) **eager-mode dispatch is dominated by Python-side scheduling
overhead** (~280-300 µs per `.realize()` call — full schedule/lower/cache-lookup from scratch, no
JIT amortization), which swamps the actual device-sync cost this bridge targets; for this shape,
CPU's own `synchronize()` is *already nearly free* by the time `.realize()` returns — there's barely
an async gap left to bridge. (2) The ~20-35 µs the bridge loses is real and reproducible, but it isn't
GIL contention (directly measured, ruled out) — most likely plain OS thread-wake/scheduling cost of
running a second live thread, which every arm/notify cycle pays regardless of how small the actual
wait is.

### The structural case for the bridge — and why it doesn't apply to *this* direction

T3.4 characterized the expensive part of `MetalDevice.synchronize()` as a **full-queue drain**:
`for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)` (`ops_metal.py:52-54`) — it drains
*everything* currently in flight, not just the one buffer a caller actually needs. I tested this
directly (dispatch K unrelated small METAL kernels, then time `Device["METAL"].synchronize()`):

| K unrelated in-flight kernels | `Device["METAL"].synchronize()` |
|---|---|
| 0 | median 168.0 µs (min 142.8 µs — matches T3.4's own reported ~150 µs almost exactly) |
| 4 | median 376.5 µs |
| 16 | median 406.8 µs |
| 64 | median 1134.4 µs |

This confirms the drain cost scales with unrelated in-flight depth — the real structural argument for
a *targeted* wait over a full drain. But it cuts against the direction this spike was scoped to test:
the expensive drain lives on **`MetalDevice.synchronize()`**, invoked when METAL is the side that must
certify its own queue — not on `Device["CPU"].synchronize()`, which this prototype's baseline
correctly uses for a CPU-producer hop and which is cheap regardless (per the component breakdown
above). The theoretical win lives in the **METAL-producer / CPU-consumer** direction, or in
METAL-consumer scenarios where METAL itself has unrelated eager work in flight — not in the specific
CPU-producer/METAL-consumer shape the task scoped for the first prototype (reasonably: it's the
direction T3.4's aliasing primitive is actually safe in).

Even there, the ceiling is capped in the case that matters most: the **production JIT-graphed
steady-state decode loop** already collapses METAL's own per-step work into **one** `MetalGraph` ICB
submit per replay (`runtime/graph/metal.py:75-96` — one `command_buffer` per `__call__`), so
`mtl_buffers_in_flight` sits near K≈0-1 in steady state, not K=16-64. That puts the realistic
production ceiling near the K=0 floor (~150 µs) — of which the watcher mechanism's own ~20-35 µs
overhead (measured above) would eat a nontrivial slice, on top of the >100-line capture-op cost from
Part 1 to make it reachable from the JIT loop at all.

## Watcher-latency reality check

The task asked specifically: if watcher wakeup (~50-100 µs) eats the win, measure and say so. Measured
reality is more specific than that guess: the *wakeup* itself is cheap (~4 µs bare round trip), and
there's no GIL-contention penalty — but the **net effect of running a live second thread at all is a
consistent ~20-35 µs loss** even before counting any capture-op integration cost, and it's paid on
every hop regardless of whether that hop would have benefited from a targeted wait. A background
Python thread is not free even when it does nothing expensive — this sizes the answer to "does the
watcher need to be native": for the CPU-producer direction, a **native (non-Python) watcher probably
still wouldn't clear the >80-line capture-op bar or beat the K≈0-1 production ceiling by enough to
matter**; it would only be worth revisiting if a future workload puts real unrelated in-flight depth
on the METAL side in steady state (contradicting the current MetalGraph-batches-to-one-ICB assumption).

## What got wired

Nothing. Objective 3 ("wire it into ONE seam... behind a default-off flag") is explicitly gated on
objective 2 showing a real win; it didn't. Zero `tinygrad/` production files touched — the entire
spike is `test/unit/test_signal_bridge_metal.py` plus this notes file.

## Gates

- `mypy tinygrad/`: clean (215 files, unaffected — no production code touched).
- `ruff check .`: clean on the new file (project convention: mypy scopes to `tinygrad/` only, per
  `TASKS.md`'s own gate list — matches T3.4/other spike precedent).
- `pytest test/unit -k "device_map or llm or signal_bridge" -n12`: 97 passed.

## Branch

`task/T3.6-signal-bridge`, off `integration/wave1`. Not merged (like T3.4, this is a refutation kept
as evidence + reusable microbench harness, not a landed feature).
