# T4.75 — Unified NV fault-mechanism RCA (all five 08-27→09-01 faults)

Fable-fork analysis, 2026-09-01. Code refs = fork master `07e6d67cf` (the tree that produced faults #3-#5; faults #1-#2 ran
earlier masters whose runtime files match at every cited line). Evidence: `t460/fault{3,4,5}_evidence.txt`,
`~/.hermes/logs/pooled-serve.log` (segments identified below), `t455/pooled_q8_c32_p16384_ctx131072_altmap_128k.log` (fault #1),
`t455/pooled_m_q8_c32_p2048_ctx65536_altmap_q38beam.log` (fault #2), `t455/ulog_128k_fault.txt` + `t459/ulog_q38_fault.txt`
(DEXT view), TASKS rows T4.23/34/36/37/40/47/50/53/72, `T4.47_RCA.md`, `T4.40_RCA.md`. Read-only analysis; nothing changed.

## §0 Verdict in one paragraph

**All five faults are the SAME event pipeline with different surfacing frames, and the pipeline is not the mystery — the
fault *sources* are.** In every one of the five, the GSP delivered a real fault event into the host-resident status queue
(`is_err_state` set — log-proven below, *including* faults #1/#2, correcting the TD.5/TD.6 "GSP RM went silent, no event"
framing), after which GSP-RM stopped servicing RPCs (the cmd-76 / cmd-47 timeouts are downstream echoes of one wedge, not
independent failures), bus-master was cleared (T4.37/T4.40b containment — worked 5/5, zero panic-shaped log entries), and the
next fresh client's FLR + halt-verify + reboot recovered the device (5-for-5). The evidence-backed common denominator of the
sources: **every fault occurred during the first-ever NV execution of an unvetted kernel/copy population** (192k attention
family; the 48-head/dim-5120 Qwen3.8 family; the spec-decode family + state-cache clone family at total VRAM exhaustion; the
state-cache clone family alone; the sharded-FFN TP family) — while warm-cache serving of proven populations has *never*
faulted, across weeks. That is the T4.53 class generalized: T4.53 proved one concrete NV-renderer-specific OOB kernel and
denied one combo; it did not close the class. One **new, cheaply-testable secondary source** falls out of the code: the
client's physical allocator plausibly overlaps GSP-RM's top-of-VRAM reservation by ~100-130 MB (§4.B), which would make
*total-exhaustion* rounds (fault #3 hit literal `Can't allocate 4096 bytes`) clobber live RM state. Experiment E1 (a
read-only register readout) settles it exactly.

## §1 The fault lifecycle as the code implements it (shared by all five)

**Steady-state serving does almost nothing that can wedge the RM.** After boot, GSP RPCs are *rare*: submits are GPFIFO ring
writes + a doorbell MMIO write (`_submit_to_gpfifo`, ops_nv.py:170-182 — socket messages to the TinyGPU server, no GSP RPC),
signal polling is a local mmap read (APL `alloc_sysmem` returns a shared-fd mapping, system.py:500-506; `HCQSignal.value`,
hcq.py:260), and vidmem alloc/free/map is entirely client-side (`mm.valloc`/PTE writes over BAR1, memory.py:247-289,
nvdev.py:52-60). The GSP's *only* standing host contact: it DMAs RPC responses and **fault events** into its message queues,
which live in **host sysmem** (`init_rm_args`, ip.py:364-387; the T4.36-era finding).

1. **Event delivery.** A device-side fault (MMU fault, RC error, RM-internal error) makes GSP-RM enqueue
   `NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED` (4101) or `OS_ERROR_LOG` (4102) into `stat_q`. Nothing interrupts us — the client
   only sees it when it *drains*.
2. **Drain sites.** `read_resp` (ip.py:62-85) sets `is_err_state` on those two functions (ip.py:74; `OS_ERROR_LOG`
   additionally prints `GSP LOG:` — **zero such prints in any of the five logs ⇒ all five were `MMU_FAULT_QUEUED`-class,
   i.e. genuine memory-access faults, not RM log complaints**). Drains happen only in `PCIIface.sleep()` (ops_nv.py:649-655),
   reached from `NVSignal._sleep` on the first poll of every wait and every 200 ms thereafter (ops_nv.py:33-48, the T4.48-F2
   cadence), and inside any `wait_resp` loop.
3. **Raise + containment.** `sleep()` sees `is_err_state`, clears PCI bus-master (`write_config_flush`, the T4.37 quiesce),
   and raises `Device fault detected` — which is why every fault's *first* traceback ends at ops_nv.py:655 regardless of
   what actually faulted. `synchronize()` parks the exception in `error_state` (hcq.py:428-440) so all later syncs re-raise.
4. **The wedge (every time).** `synchronize`'s except-path calls `on_device_hang()` (ops_nv.py:859-881) → its very first
   `rm_control` (READ_ALL_SM_ERROR_STATES) hits `wait_resp`'s 10 s timeout → **`Timeout waiting for RPC response for command
   76`**. At exit, `device_fini` → `fini_hw` → `rpc_unloading_guest_driver` → **cmd 47 timeout** (ip.py:522, 611-614, 87-91);
   the try/finally still clears MASTER (ops_nv.py:633-647). This cmd-76→cmd-47 pair appears in **all five** logs and in
   T4.37 step A and the T4.50 capture: **a faulted GSP-RM on this stack has never once answered another RPC.** The wedge is
   a *constant post-fault property*, not a separate mechanism to hunt.
5. **Recovery.** Next client: WPR2-up detected → clear MASTER → FLR → falcon engine reset + `active_stat==0` verify
   (T4.40c, nvdev.py:145-173) → full GSP reboot. 5-for-5 clean, DEXT logs show only the clear + finalize-mappings
   (`ulog_128k_fault.txt`, `ulog_q38_fault.txt`) — the panic-era host-safety problem is closed and stays closed.

**Why the RM wedges** (analysis, not verifiable from here): tinygrad registers no MMU fault buffer, never acks RC/fault
events, and services no interrupts — GSP-RM's RC recovery path stalls waiting on a handshake the driverless polled
environment never provides. Treat as black-box: post-fault, the RM is gone until reset. (The `on_device_hang` TODO about
`CLEAR_ALL_SM_ERROR_STATES` cannot work — the RM stops answering before we could send it.)

## §2 The five faults, normalized

| # | date | workload (first-ever NV execution of…) | surfacing frame (all end at ops_nv.py:655) | memory state at fault | RPC echoes |
|---|------|------|------|------|------|
| 1 | 08-27 17:43 | 192k-ctx attention kernel family (TD.5 run 1 warmup, fresh BEAM) | `warmup → generate → JIT capture → get_runtime → NVProgram.__init__:346 synchronize` | no OOM (0 MemoryErrors) | cmd 76 → cmd 47 |
| 2 | 08-28 11:09 | whole Qwen3.8 geometry (48-v-head scan, dim 5120; TD.6 BEAM run 1, 1h45m in) | identical to #1 (NVProgram.__init__ sync during warmup capture) | no OOM | cmd 76 → cmd 47 |
| 3 | 08-31 19:19 | spec-decode (`--mtp`) family + state-cache clone family (phase 4; MTP-budget map) | request-path `synchronize`/graph waits; **preceded by `KeyError: 248320` (the T4.73 WY corruption) on the first spec request, two clean WY prefills (25/24 tok/s), then 12 MemoryErrors ending at `Can't allocate 4096 bytes` = total TLSF exhaustion** | **at the absolute VRAM wall** (`Used: 21.71 GB` + full fragmentation) | cmd 76 + cmd 47 (×4), 16 fault echo lines |
| 4 | 08-31 20:23 | state-cache clone family alone (T4.72 round C, loop scan, plain map) | first decode graph after the 5.9k prefill's `store_snapshot`: `exec_copy → _copyout → _drain` wait (hcq.py:643/626) | **zero memory pressure** (0 MemoryErrors in segment) | echoes only |
| 5 | 08-31 21:43 | sharded-FFN TP family (T4.70d warmup, `tp:NV=0.70`) | warmup JIT compile at 350/351 kernels → `synchronize` (hcq.py:436) | no OOM logged (21.7 GB static budget) | cmd 76 → cmd 47 **during UNLOADING_GUEST_DRIVER teardown** |

Corrections this table forces onto the record:
- **TD.5/TD.6's "RPC-47 timeout, GSP went silent, no event" framing is wrong.** Both logs open with the standard
  `Device fault detected` raise out of a drain (f1 lines 3-15, f2 lines 15-27) and end with
  `NV synchronization failed before finalizing: Device fault detected` — `error_state` held the *evented* fault. The
  cmd-47 line was merely the last exception printed. All five faults are the same evented class.
- **Fault #3 is a compound**: the T4.73 WY corruption (KeyError, *not* a device fault) fired first; the device fault came
  later, after the state-cache had driven the allocator to literal 4 KB exhaustion. T4.72's re-attribution (store path, not
  MTP) stands, with the wall as an additional distinguishing feature vs fault #4's pressure-free store fault.
- The "storm counts" (109/123 fault lines in the T4.29/T4.34 era, 16 lines in #3) remain what T4.50 proved: **one fault +
  deterministic per-buffer teardown echoes**, now mostly suppressed by T4.52's echo-skip.

## §3 The common denominator, stated precisely

Not "fresh searches" (disproven: ~9 clean fresh searches), not "heavy copies" (the 29-37 GB model loads are the heaviest
copy workload we run and have never faulted), not memory pressure alone (T4.23 proved clean OOM never faults; T4.34 proved
storms at 13 GB free). The invariant that survives all data:

> **Every fault occurred while the NV device was executing kernels (or device-side copy programs) from a population that
> had never run on NV silicon before. No fault has ever occurred executing an already-vetted population.**

Fresh BEAM searches, warmup captures of new model geometries, the first spec-decode requests, the first state-cache stores,
and the first TP warmup are all instances of the same exposure. Warm-cache serving replays survivors. This is exactly the
T4.53 lesson scaled up: BEAM's search space (and, equivalently, first-run heuristic lowerings of a new geometry) contains
rare kernels that are **miscompiled/marginal specifically on the NV backend** (T4.53: byte-tight bounds in the AST, Metal
renders the same AST clean ⇒ NV-renderer/shared-mem-specific; root cause parked as T4.54). One combo was denied; the class
was not closed, and every new kernel family re-rolls the dice.

### Ranked fault-source hypotheses

**A. NV-backend OOB/marginal generated kernels (T4.53 class) — PRIMARY; proven once, strongly correlated 5/5.**
Covers #1/#2/#5 cleanly (pure new-family warmups, no memory pressure) and is consistent with #3/#4 (the clone/spec kernels
are themselves first-run). MMU_FAULT_QUEUED with no GSP LOG matches an OOB access exactly. Prior probability boosted by the
proven instance and by Metal-vs-NV render divergence.

**B. Physical-allocator overlap with the GSP-RM reservation at top-of-VRAM — NEW, UNVERIFIED, cheaply decidable.**
`_early_mmu_init` gives the memory manager `vram_size - 64 MB` (nvdev.py:198-199); `pa_allocator` spans `[~2 MB,
vram_size-64 MB)` (memory.py:181-184). But the WPR meta we hand the GSP (ip.py:434-455) reserves, from the top: 1 MB VGA +
1 MB FRTS + bootloader + the radix3 GSP image (~36-64 MB) + a 129 MB (`0x8100000`) GSP heap + 1 MB non-WPR heap ⇒
**~170-195 MB**, i.e. the client's allocator plausibly overlaps the RM's region by **~105-130 MB**. TLSF hands out low
addresses first, so the overlap is touched only near exhaustion — precisely the wall-cluster (fault #3 at `Can't allocate
4096 bytes`; the T4.30-cell-4/T4.35 era faults at ≤1 GB headroom). Writes into the WPR proper are hardware-dropped (reads
return garbage — silent corruption); writes into the non-WPR RM heap clobber live RM state (wedge). Firmware files aren't
cached locally so the image-size term is a bound, not a number — **E1 below reads the true WPR bounds off the registers and
settles this in one minute of a healthy window.** If confirmed, it is also a standing silent-corruption hazard for *any*
at-the-wall run, faulting or not. (It cannot explain #4: zero pressure there.)

**C. A genuine addressing/aliasing bug in the snapshot-store path (T4.74's hunt) — REQUIRED for #4 unless A covers it.**
Round C is deterministic: same 5.9k prefill, cache off = clean (round A), cache on = fault (2/2: #3's segment + #4). The
store runs ~60 first-ever clone kernels + slice reads of live KV/conv/recurrent buffers right at the prefill boundary
(model.py:1240-1258, serve.py:137) — either one of those kernels is an (A)-class OOB on NV, or the store's interaction with
in-flight decode state (clone of a buffer the JIT graph is still writing?) is a real race. T4.74's audit + candidates are
the right instrument; the E3 experiment below is its hardware harness.

**D. USB4/tunnel integrity under sustained load — RANKED LOW.** MMIO/cfg RPCs keep working during every wedge (the MASTER
clear itself succeeds; DEXT logs show clean cfg traffic), model loads push far more bulk data than any faulting phase, and
recovery never needs a dock power-cycle. No datum requires it; only E5 would resurrect it.

**E. Power/thermal transients (eGPU PSU under BEAM-style launch churn) — RANKED LOW.** Deterministic A/B behavior of the
state-cache rounds and the zero-fault record of equally-hot warm workloads argue against a stochastic electrical cause;
same E5 catch-all.

## §4 Reconciliation with the T4.47/T4.50-era RCA

The old RCA is **confirmed and extended, not contradicted**:
- T4.47's chain (abandoned/faulting candidate → event sits undrained → surfaces from an unrelated frame) is exactly what
  #1/#2/#5 show — with F2's drain-on-wait-start now bounding the latency to one wait, which is why the surfacing frames are
  now *near* the culprit (program-construction syncs during the same capture) instead of minutes later.
- T4.50's storm anatomy (1 fault + teardown echoes) reproduces in #3's 16 lines; T4.52's skip killed the 490 s grind.
- T4.53 named and denied one culprit combo; #2 faulting on an all-new geometry and #5 on the TP family are the class
  outliving the single deny — as T4.53 itself predicted by parking T4.54 (the renderer root cause) rather than closing it.
- What IS new versus that era: (i) proof that *all* faults are evented (the "silent RPC-timeout class" never existed);
  (ii) the wedge understood as a constant (cmd-76/47 echoes carry no independent information — stop treating "command 47"
  as a signature worth bisecting); (iii) the B-overlap hypothesis; (iv) the store-path determinism (C) — the first faulting
  workload that is *replayable on demand*, which is worth more than every stochastic datum we have.

## §5 Ranked systemic mitigations

1. **Fault forensics that survive the wedge (do first — GPU-free, ~15 lines).** `on_device_hang` burns 10 s on an RPC that
   has failed 100% of the time and reports nothing. Change: short per-call timeout (2 s) + catch → report "GSP-RM
   unresponsive (the standard post-fault wedge)"; add an env-gated in-process ring buffer of the last N kernel/copy
   dispatches (name + buffer VAs), dumped on the fault raise — the runtime generalization of `BEAM_LAUNCH_LOG` that the
   serve path lacks (T4.72 could not name #4's culprit for exactly this reason). Risk: none. Validation: E3 names a kernel.
2. **Read the real WPR bounds, then fix the reservation if short (E1 → one-line nvdev.py change).** If
   `NV_PFB_PRI_MMU_WPR2_ADDR_LO` lands below `vram_size-64MB`, set the mm ceiling to `wpr_lo - margin` (costs ~130 MB of
   24 GB). Risk: none beyond the capacity. Validation: re-run a wall workload (fault #3's shape) — the deterministic
   at-exhaustion fault either moves or dies.
3. **Vet new kernel families at warmup, never at request time (serve-side, the T4.72 recommendation).** Extend serve
   `warmup()` to exercise the spec path, one `store_snapshot`+restore, and (later) the TP path, so first executions happen
   inside the controlled window with forensics armed — a fault becomes a named, recoverable warmup event instead of a
   mid-request mystery. Risk: none (warmup already tolerates failure). Cost: seconds.
4. **Grow the T4.53 deny list from named culprits.** Every fault that mitigation 1+E-experiments pin to a kernel feeds a
   `get_kernel_actions` deny (search-space-only, the proven pattern) — or, better, a T4.54 renderer fix if the shared-mem
   bug is found. Risk: minimal search-space narrowing.
5. **Eager drain mode for exposure windows (`NV_EAGER_DRAIN=1`).** `iface.sleep(0)` after each first-run dispatch batch
   tightens event→culprit attribution to one dispatch. Cost: one 4 KB host-sysmem read per drain (local mmap — cheap);
   gate it to vet/warmup phases only. Risk: none.
6. **(Parked) RC-ack / fault-buffer registration so the RM survives a fault.** Upstream-shaped, large, and unproven — the
   RM wedges before any client action could land; a fresh client already recovers in ~60 s. Not worth the effort while
   recovery is 5-for-5.

## §6 Minimal hardware experiment matrix

Each fits one server window, honors HANDOFF §4 (record → fresh client → continue deliberately; never kill NV processes).

| id | run | discriminates | risk |
|----|-----|----|------|
| **E1** | Healthy idle client, read-only: `NV_PFB_PRI_MMU_WPR2_ADDR_LO/HI` (+WPR1) via the existing reg machinery; compare against `vram_size-64MB` | Hypothesis B exactly (yes/no + the true overlap size) | zero (reads; regs already read at nvdev.py:145) |
| **E2** | Idle client, no model: valloc to exhaustion, write+readback patterns through the LAST allocations, free | B's corruption mode directly (readback mismatch = WPR garbage; RM wedge = heap clobber) | moderate — run last in its window; planned recovery |
| **E3** | The #4 reproducer with mitigation-1 forensics + `NV_EAGER_DRAIN`: 3.8 (or any attention model) + `--state-cache` + one 5.9k prefill | Names C's faulting kernel/copy (deterministic 2/2 today); feeds T4.74's candidate A/Bs | known-recoverable |
| **E4** | A fresh-family warmup (TP family is the crispest) twice: stock vs copies-serialized (sync after every copy, staging verified) | A (kernel OOB — faults persist) vs any copy/transport contribution (faults vanish) | known-recoverable |
| **E5** | *Conditional* (only if E1-E4 all come back clean): hours of sustained proven-kernel load (warm 35B decode loop) at max churn | D/E catch-all — a fault here would flip the ranking | low |

Order: E1 → land mitigation 1 → E3 → E2 → E4; E5 only on contradiction. E1+E3 together are one short window.

## §7 Confidence

- Pipeline (§1), evented-fault unification (§2), wedge-as-constant: **high** — every link is in source with line refs and
  reproduced in ≥2 independent logs.
- Common denominator (§3 invariant): **high** as a correlation (5/5 faults, 0 counterexamples in weeks of warm serving);
  source attribution per fault: **medium** — class-level (A) has one proven member, but no fault since T4.50 has *named*
  its kernel (mitigation 1 exists to fix that).
- Hypothesis B: **unverified** (parametric bound; firmware not cached locally) — E1 settles it exactly, for free.
- Only hardware can settle: the true WPR bounds (E1), the store-path culprit's identity (E3), kernel-vs-copy attribution
  (E4), and — if everything above comes back clean — the electrical/tunnel residue (E5). GSP-RM's internal wedge reason is
  out of reach from this side of the firmware and does not block any mitigation.

## §8 Addendum — E1 executed (2026-08-31 close-out gap)

Hypothesis B **CONFIRMED**: `NV_PFB_PRI_MMU_WPR2_ADDR_LO/HI` read 0x05f3f000/0x05ffee00 → WPR2 = [0x5f3f00000, 0x5ffee0000] = [23.812, 23.999] GiB on the 24 GiB 3090; client mm ceiling (vram−64MB) = 0x5fc000000 → **129.0 MiB overlap**, matching the predicted ~105-130 MB. Mitigation 2 applied as T4.77 (`vram_size−(256<<20)`, branch `task/T4.77-wpr-ceiling`): 256MB clears the measured 188MB WPR2 span with margin. E2 remains the corruption-mode validation. Side findings: GSP firmware is fetched from gitlab at fresh-client init (not cached — a cold-start dependency on network); WPR1 registers are absent from `nv_regs/dev_fb.py` (KeyError; WPR2 alone settles B).
