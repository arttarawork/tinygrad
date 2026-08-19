# TASKS.md — agent handoff for the Ampere-over-Thunderbolt effort

Task breakdown of `NV_LLM_DESIGN.md` (WS refs point there; context in `memory.md` — read both first).
Baseline `af2a43c85`; rebase on upstream master weekly. Written 2026-08-18, while the eGPU dock
(AOOSTAR AG02) is in the mail — **Phase 0 tasks need no NVIDIA hardware at all.**

## Conventions for agents

- Branch per task: `task/T<id>-<slug>` off baseline `af2a43c85` (== `origin/master`; no local
  `master` branch exists). Remotes: `origin` = arttarawork/tinygrad fork, `upstream` = tinygrad/tinygrad.
- Python env (Mac, verified 2026-08-18): no bare `python`; Homebrew python3.14 has no test deps.
  Use `/Users/artur/Documents/tinygrad/.venv` (numpy, torch, pytest+xdist, hypothesis, z3, gguf,
  mypy 1.19.1, ruff 0.14.10). From any checkout/worktree: `PYTHONPATH=. <venv>/bin/python -m ...`.
- Before pushing: `PYTHONPATH=. .venv/bin/python -m pytest <touched area> -x -q -n12`,
  `.venv/bin/python -m mypy tinygrad/`, `.venv/bin/python -m ruff check .`
- Mac resource limits (updated 2026-08-18, after Artur freed ~157 GB): **~179 GB disk free** —
  model downloads are now fine (bench GGUFs go through `tinygrad.llm`'s fetch cache; gpt-oss-20b
  MXFP4 for T1.3 validation OK). **llama-server still keeps ~23 GB wired** (LaunchAgent KeepAlive)
  — before real-model METAL runs / T0.1 benchmarks, stop it: `launchctl bootout gui/501/com.artur.llama-server`
  (restart: `launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.artur.llama-server.plist`).
  Tiny random-weight configs are fine anytime.
- Perf claims need before/after numbers from the T0.3 harness on named hardware. Upstream-bound
  changes must be small and hand-verified — maintainers have reverted "ai slop" before (see memory.md §4).
- Don't remove the deliberate `.contiguous()` calls in the MoE expert path (`llm/model.py:27,129`).
- `PYTHONPATH=.` when running from the clone.

## Execution environments

| Tag | Meaning |
|---|---|
| `ANY` | pure code; NULL/CPU device is enough (kernel-count and scheduling tests run on `DEV=NULL`) |
| `MAC` | the MacBook M3 Pro — Metal backend, real perf numbers today |
| `AMD` | (descoped 2026-08-18 — see T0.2; kept for the footgun note) the Bazzite box w/ RX 9070 XT. **Never** use the AM/PCI driver path there: it unbinds `amdgpu` and kills the display. |
| `MOCKNV` | NV backend under gpuocelot PTX emulation — functional correctness only, no perf. Recipe (T0.4, verified 2026-08-18): `OCELOT_PATH=.venv/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX PYTHONPATH=. .venv/bin/python -m pytest ...` — the `DEV` string must start with `MOCK` (`device.py:376` never falls back to `MOCKIface` otherwise) and the dylib is CI's prebuilt (`github.com/tinygrad/gpuocelot` release v0.1.0, already at `.venv/lib/`). Do NOT use `extra/setup_mock_nv_osx.sh` (heavy source build + sudo to /usr/local/lib; CI doesn't use it either). Details: `MOCKNV_SETUP.md` on `task/T0.4-mocknv`. |
| `CLOUD3090` | optional: a rented Linux 3090 (vast.ai etc.) runs the same NV backend/kernels via `NVKIface` — real sm_86 perf for kernel work before the dock arrives |
| `DOCK` | blocked on the AG02 + TinyGPU |

## Phase 0 — no dock required

### T0 · Bring-up & baselines

- **T0.1 — Metal baseline table** `[MAC]` deps: —
  Clone on the Mac; run `python -m tinygrad.llm -m qwen3:8b --benchmark --warmup` (and qwen3.6:27b,
  qwen3-30b-a3b) on `DEV=METAL`, with and without `JITBEAM=2`; same GGUFs through `llama-bench` (Metal).
  *Done when:* a committed CSV/table of load / prefill / decode tok/s for ≥3 models × both stacks.
- **T0.2 — ~~Verify the 9070 XT as an HCQ testbed~~ DESCOPED (2026-08-18)** `[AMD]`
  AMD is not a target; the box was only a real-HCQ stand-in for validating shared `hcq.py` changes
  pre-dock. That role is covered better by `CLOUD3090` (same backend as target, NVKIface) and by
  upstream CI's AMD runners. Revive only if a cheap local HCQ sanity-check ever beats renting.
- **T0.3 — Bench harness** `[MAC]` deps: T0.1 (WS5.1)
  One script: same GGUF → `tinygrad.llm` + `llama-bench`, emits CSV (model, dev, flags, load s,
  prefill t/s, decode t/s, GB/s from `GlobalCounters`). Validated on Metal now; reused on NV later.
  *Done when:* T0.1's table is reproducible with one command.
- **T0.4 — Mock-NV bring-up** `[MOCKNV]` deps: —
  Get `DEV=NV` under gpuocelot green for `test/test_tiny.py` locally (Mac or Linux). Document setup
  quirks. This unblocks all NV-touching code tasks pre-dock.
  *Done when:* documented one-shot setup + passing test_tiny.

### T1 · Decode-path kernels & dtypes (WS1) — all measurable on Metal today

- **T1.1a — fp16 KV cache: implement + accuracy** `[ANY, runnable anytime]` deps: — (WS1.1)
  Explicit KV dtype at `llm/model.py` `_init_state` (default fp16, `KV_F32=1` escape flag); check
  every `_init_state` variant (attention, MLA, SSM conv/state — SSM state may need fp32, decide
  per-block with evidence). Accuracy: greedy-token parity + max logit delta vs fp32 over ≥5 prompts
  on llama3.2:1b (1 GB, fits anytime) AND a recurrent tiny-config. STOP if any family needs
  >1-line special-casing — report instead.
  *Done when:* diff + accuracy table committed; upstream-PR-shaped. Perf is T1.1b, not this task.
- **T1.1b — fp16 KV: measure decode delta** `[MAC, bench window]` deps: T1.1a, llama-server stopped
  T0.3 harness, qwen3:8b, fp32-KV vs fp16-KV on integration, no-BEAM + JITBEAM=2, long-context
  variant (`-p 4096`) where the KV read dominates. *Done when:* CSV rows + delta in BENCH_NOTES.md.
- **T1.2 — MATVEC heuristic: see through CAST** `[ANY→MAC]` deps: T0.1 (WS1.2)
  Reproduce the miss: `DEBUG=3` on a fp16/Q4 gemv, confirm no `MATVEC:` line (guard at
  `codegen/opt/heuristic.py:60-78` requires `MUL(INDEX,INDEX)`, real ASTs wrap it in CAST).
  Patch guard, add a kernel-selection unit test (NULL device), measure decode on Metal.
  *Done when:* heuristic fires on fp16+quantized gemvs; no regressions in `test/opt/`.
- **T1.3 — gpt-oss arch in `tinygrad/llm`** `[MAC]` deps: T0.1 (WS1.6)
  Wire the `gptoss` GGUF arch into `llm/model.py:340-451` + registry (`cli.py:76-94`); MXFP4 dequant
  already exists (`gguf.py:105-114`); training-side reference in `examples/mlperf/`. Validate
  gpt-oss-20b MXFP4 (~12 GB, fits the Mac's wired budget) output vs llama.cpp same-seed greedy.
  *Done when:* `-m gpt-oss:20b` generates correct text on Metal; benchmark row added.
- **T1.4 — Single-kernel MoE top-k — RESOLVED-AS-MEASURED (2026-08-18): premise was wrong.**
  `pairwise_topk` already costs exactly **1** kernel/layer in-model (control experiment on NULL:
  real topk vs free `arange` sel, all gating paths, T=1 and T=8). The scatter+slice+cast lower to
  an inlined select chain; only the rank reduce realizes, and rangeify's `remove_bufferize`
  (rangeify.py:258-282) makes 1 the floor (buffer-reading REDUCE can't inline into a consumer
  reduce). Alternatives (one-hot select, int32 scatter, `Tensor.topk` bitonic) all equal or worse.
  Landed tests-only (+19): exact tie-break equivalence vs numpy stable argsort + kernel-count pin.
  The other ~3 routing kernels/layer are the caller's probs path (gather + softmax stats,
  model.py:124-127) — a different, larger task if ever worth it.
- **T1.5 — Skip RNG at temperature 0** `[ANY]` deps: — (WS1.5)
  `llm/model.py:358-364`: bypass Gumbel noise when temp==0 without retriggering JIT capture.
  *Done when:* argmax path drops the threefry work; greedy outputs identical.
- **T1.6 — Cache `_prepare_jit_inputs`** `[ANY]` deps: — (WS2.4-adjacent)
  `engine/jit.py:200-218` re-derives state dicts every call (~0.5 ms/token host). Memoize safely.
  *Done when:* host time/token measurably down on Metal; JIT tests green.
- **T1.7 — Fused-attention track A: PCONTIG** `[MAC/ANY]` deps: T0.1 (WS1.4a) — *high effort/risk*
  Drive `PCONTIG>2` online-softmax (`rangeify.py:264-282`) on decode shapes; try un-skipping
  `test_softmax_fusion.py` cases; document exactly what breaks if it does.
  *Done when:* either attention kernel count drops on decode shapes, or a written failure analysis.
- **T1.8 — Fused-attention track B: pluggable custom kernel** `[MAC]` deps: T0.1 (WS1.4b)
  Add a clean attention-override hook in `llm/model.py:196` (pattern: `STUB_ATTENTION`,
  `extra/models/llama.py:104-119`; `Tensor.custom_kernel` is tested). Prove it with a naive Metal
  custom kernel; the tuned sm_86 kernel is T2.4/TD-side.
  *Done when:* hook merged behind a flag; parity test vs SDPA passes.
- **T1.10 — MATVEC for quantized (GGUF-fused) gemvs** `[MAC]` deps: T1.2 (WS1.2 follow-up, found 2026-08-18)
  T1.2 fixed fp16 but confirmed quantized gemvs miss MATVEC for two deeper reasons: the weight
  operand is a whole dequant expression (e.g. Q4_0: `MUL(INDEX, MUL(CAST(ADD(BITCAST(AND(...)))), ...))`
  with 3 INDEXes into one uchar buffer), and GGUF block substructure splits the row axis into
  multiple global axes (`full_shape=[32,2,16,1024]`). These are the dominant decode kernels for
  Q4_K models. Needs its own pattern match (or BEAM-informed hand-coded opts), not a CAST strip.
  *Done when:* MATVEC-class opts fire on Q4_0/Q4_K gemvs; measured GB/s uplift on Metal; no test/opt regressions.
- **T1.9 — Streaming GGUF load** `[MAC]` deps: T0.1 (WS2.3-adjacent)
  Replace whole-file blob (`llm/gguf.py:134`) with per-tensor staging to cut the ~2x transient and
  TB-load cost; keep the io_uring fast path. Helps Metal load times immediately.
  *Done when:* peak load memory ≈ model size; load time not worse on Metal.

### T2 · Transport & runtime (WS2) — build now, tune on dock

- **T2.1 — Parallelize `_copyout`** `[MOCKNV→CLOUD3090]` deps: T0.4 (WS2.2)
  Mirror `_copyin`'s 32×2 MB round-robin (`hcq.py:559-576`) in `_copyout` (`hcq.py:596-609`).
  Shared HCQ code — write + functional-check under mock; real-hardware D2H bandwidth numbers from
  a rented 3090 (or post-dock). Upstream CI's AMD runners cover the AMD side of `hcq.py`.
  *Done when:* D2H bandwidth up on real NV hardware; `test/device/test_hcq.py` green.
- **T2.2 — Batch PTE writes / defer remote validation reads** `[MOCKNV]` deps: T0.4 (WS2.3)
  `nvdev.py:48-49` writes one 8-byte PTE per socket message; `memory.py:204-213` does blocking
  readback. Add a bulk-write path + skip validation on remote ifaces. Functional under mock; the
  latency win is measured post-dock.
  *Done when:* map_range socket-message count collapses (count messages in a fake iface test).
- **T2.3 — NV remote-tuning skeleton** `[ANY→DOCK]` deps: — (WS2.1)
  Prep an `is_remote()`-keyed sizing layer for NV mirroring AMD's `is_usb()` knobs
  (`ops_amd.py:980-994` template): kernargs, sigalloc, ring sizes, `bind()` bulk writes. Values
  tuned post-dock; structure lands now.
  *Done when:* knobs exist with defaults = current behavior; no-op on NVK Linux.
- **T2.4 — sm_86 kernel work on a rented 3090** `[CLOUD3090]` deps: T1.8 — *optional accelerator*
  Same NV backend, real tensor cores: BEAM sweeps on decode gemvs, tuned FA custom kernel for the
  T1.8 hook, MATVEC perf confirmation. Everything transfers to the eGPU minus transport.
  *Done when:* beam cache + FA kernel with measured tok/s vs Metal baseline.
- **T2.5 — Amortize the per-token sync** `[MAC]` deps: T0.1 ✅ (WS2.4)
  `generate()`'s per-token `.item()`: keep sampled tokens on device, drain every N for streaming;
  overlap the copyout with the next launch. Branch off `integration/wave1` (generate() moved:
  device-aware `t`/`temp` landed in review-fixes — re-locate the loop before scoping).
  *Done when:* host-visible stall/token down on Metal (T0.3 harness row); streaming UX unchanged (N≤4).

### T3 · Pooling groundwork (WS3 Stage A) — the sleeper: fully rehearsable pre-dock

Metal+CPU on the MacBook hits the *same three cross-backend blockers* as Metal+NV
(one-binary-per-`device[0]`, same-device kernel assert, no mixed graph capture) — so Stage A
can be built and proven before the dock ships.

- **T3.1 — Device-map plumbing in `tinygrad/llm`** `[ANY]` deps: — (WS3.A)
  `--device-map` (explicit ranges + `auto` by free memory): per-layer weight placement via
  `.to_()` before `load_state_dict` (loader honors pre-placed params, `nn/state.py:211-214`),
  per-layer KV-cache device (`model.py:200-204`), boundary copies at the `@function` block seam
  (`model.py:145-151`). Prototype homogeneous first: `("CPU:0","CPU:1")` / NULL.
  *Done when:* a model runs split across two same-backend devices with correct output.
- **T3.2 — Heterogeneous pipeline: METAL+CPU rehearsal** `[MAC]` deps: T3.1 ✅
  Swap one side for CPU. T3.1 already proved (homogeneous): mixed-device JIT capture works with no
  fallback, only COPY spans devices, and graph batching forms per-backend islands (METAL graphed,
  CPU sequential) — exactly the Stage A shape. Remaining here: do it cross-BACKEND, measure
  boundary cost/token, and **force realize for split models** (unrealized lazy initializers get
  captured and re-run every step — T3.1 finding).
  *Done when:* qwen3-8b runs layers split METAL/CPU, output correct, boundary cost quantified.
- **T3.3 — MoE placement policy** `[MAC]` deps: T3.2 ✅ (WS3.A)
  Sub-layer split: attention+norms+KV on device A, routed-expert FFN tensors on device B — extends
  `device_map` with per-tensor (not just per-block) placement for `ffn_*_exps`. Validate on a tiny
  MoE config first (exact-output test, both directions); then olmoe Q4 (~4.2 GB, registry —
  download OK, disk is fine) with experts on CPU and the rest on METAL (fits beside llama-server).
  Budget hops against T3.2's ~750 µs/copy floor — 2 hops/MoE-layer is the design's expected shape.
  *Done when:* MoE model runs with experts on the second device, outputs exact, hop count/token
  measured vs the 2/layer expectation. This is the flagship NV+METAL shape.
- **T3.4 — Zero-copy bridge spike (Stage B)** `[MAC]` deps: T3.2 (WS3.B)
  Wrap shared host memory across backends: `BufferSpec.external_ptr` on Metal (`ops_metal.py:157`)
  over a CPU/NV-visible buffer; sketch the HCQ-signal↔`MTLSharedEvent` host bridge.
  *Done when:* one boundary copy eliminated in the T3.2 pipeline, measured.
- **T3.5 — Boundary-copy microbenchmark — RESOLVED by T3.2 (2026-08-18):** METAL↔CPU table
  delivered (~750 µs fixed floor <1M elems, bandwidth above). Remaining: rerun on METAL↔NV post-dock.

### T4 · Post-baseline work (added 2026-08-18, after waves 1-2 + measured baselines)

- **T4.1 — Upstream PR prep: MATVEC pair first** `[ANY]` deps: — (WS5/G3)
  Rebase `task/T1.2-matvec-cast` + `task/T1.10-matvec-quant` (as ONE combined branch off current
  `upstream/master`), re-run `test/opt/` + mypy + ruff, re-verify the fp16 + Q4_0 gemv wins still
  hold with a quick METAL microbench, and write the PR description: what/why, the measured numbers
  (56→100 GB/s fp16, ~4x membw Q4_0/Q6_K, and the +48% no-BEAM decode contribution from the
  baseline table). **Do NOT push or open the PR — Artur reviews and submits.** *Done when:* a
  rebased branch + PR-description file are ready for hand-off. (T1.5, T1.6, T2.1, T2.2 follow the
  same recipe as separate later tasks once this one lands cleanly.)
- **T4.2 — Q4_K dequant ALU cost** `[MAC]` deps: — (from T1.10's finding)
  T1.10 measured Q4_K gemvs ALU-bound (4x membw, flat wall-time; Q4_0/Q6_K got ~4x). Profile the
  Q4_K kernel (DEBUG=2 + generated source), identify the 6-bit sub-scale unpack cost, try ≤2
  targeted rewrites (e.g. restructure the scale-unpack expression in `llm/gguf.py` Q4_K dequant, or
  a BEAM comparison to see what search finds). STOP after 2 attempts if wall-time won't move —
  a written analysis is a valid outcome. Q4_K_M is the most common quant in the wild; this gates
  its decode win. *Done when:* Q4_K wall-time improves ≥15%, or the blocking analysis is committed.
- **T4.3 — gpt-oss-20b real-model validation** `[MAC, bench window — llama-server MUST be stopped
  (12 GB model)]` deps: T1.3 ✅
  `-m gpt-oss:20b` (GGUF cached): generate vs llama.cpp same-model same-prompt greedy (llama-cli
  `--temp 0`); token-level comparison over ≥3 prompts crossing the chunk_size=32 prefill boundary
  (exercises sliding-window × chunked-prefill, T1.3's untested interaction). Add a benchmark row
  via T0.3 harness while the window is open. *Done when:* parity verdict (exact or divergence
  documented with position/cause) + bench row committed.
- **T4.4 — BEAM prefill anomaly** `[MAC]` deps: — *small filler*
  Baseline table showed integration BEAM prefill 43.47 vs upstream 46.65 tok/s (single runs).
  3 repeats each side (harness exists); if the gap is real (>spread), bisect which wave lever
  costs prefill and why (likely MATVEC guard firing on a prefill kernel it shouldn't). STOP after
  attribution — fix is a follow-up. *Done when:* variance verdict or named culprit in BENCH_NOTES.md.

## Phase 1 — dock arrives (`DOCK`)

- **TD.1 — TinyGPU first light**: install script, DEXT approval, `DEV=NV` test_tiny; audit
  `is_bar_small()` on the AG02 and where kernargs/cmdq land (design §3.1). deps: dock.
- **TD.2 — WS0 truth table**: full matrix via T0.3 harness — `DEV=NV{,:NAK}`, `JITBEAM={0,2}`,
  vs Metal + llama.cpp baselines. Names the real top-3 bottlenecks. deps: TD.1, T0.3.
- **TD.3 — Land the prepared work on real transport**: tune T2.3 knobs, validate T2.1/T2.2 wins,
  re-measure T1.x on NV, swap T3.2's CPU→NV = actual Metal+NV pooling. deps: listed tasks.
- **TD.4 — Publish**: upstream PR train (T1.1, T1.2, T1.4-6, T2.1-2 first — smallest, benchmarked);
  demo pooling to exo#1904 + tinygrad Discord. deps: TD.2 numbers.

## Dependency graph

```mermaid
flowchart LR
  subgraph P0["Phase 0 — no dock"]
    T01[T0.1 Metal baselines] --> T03[T0.3 harness]
    T04[T0.4 mock-NV]
    T03 --> T11[T1.1 fp16 KV]
    T01 --> T12[T1.2 MATVEC cast]
    T01 --> T13[T1.3 gpt-oss arch]
    T14[T1.4 topk 1-kernel]
    T15[T1.5 temp0 RNG]
    T16[T1.6 jit-input cache]
    T01 --> T17[T1.7 PCONTIG attn]
    T01 --> T18[T1.8 attn hook]
    T01 --> T19[T1.9 stream load]
    T04 --> T21[T2.1 copyout ||]
    T04 --> T22[T2.2 PTE batch]
    T23[T2.3 remote knobs]
    T18 --> T24[T2.4 cloud-3090 kernels]
    T01 --> T25[T2.5 sync amortize]
    T31[T3.1 device map] --> T32[T3.2 METAL+CPU pipe]
    T32 --> T33[T3.3 MoE placement]
    T32 --> T34[T3.4 zero-copy spike]
    T31 --> T35[T3.5 copy microbench]
  end
  subgraph P1["Phase 1 — dock"]
    TD1[TD.1 first light] --> TD2[TD.2 truth table]
    TD2 --> TD3[TD.3 land + tune]
    TD3 --> TD4[TD.4 upstream + demo]
  end
  T03 --> TD2
  T21 --> TD3
  T22 --> TD3
  T23 --> TD3
  T24 --> TD3
  T32 --> TD3
```

## Status log

| Date | Task | State | Branch | Notes |
|---|---|---|---|---|
| 2026-08-18 | env setup | done | `memory` | `.venv` created; `upstream` remote added; `test/test_tiny.py` green on METAL (19 passed) |
| 2026-08-18 | T1.2 | **done** | `task/T1.2-matvec-cast` | `1fbbcee83` (+17/−3): hypothesis CONFIRMED; `uncast()` strips one CAST at reduce body + MUL operands. fp16 4096² gemv 56→100 GB/s kernel-side (1.8x, METAL, noisy-machine caveat). test/opt+mypy+ruff green. Upstream-PR-ready. Spawned T1.10: quantized gemvs are a SEPARATE miss (dequant expr operand + block axes `[32,2,16,1024]`). |
| 2026-08-18 | T1.4 | **done (premise refuted)** | `task/T1.4-topk-1kernel` | `d24bdb713` (+19, tests only): topk is already 1 kernel in-model; 1 is the rangeify floor. Tie-exact + kernel-count tests pin it. 388 tests + mypy + ruff green. Design-doc "4 routing kernels" claim corrected. |
| 2026-08-18 | T1.5 | **done** | `task/T1.5-temp0-rng` | `f53ceb67f` (+51/−7): greedy uses `temperature=None` sentinel from `generate()`; jits keyed `(is_prefill, greedy)` — no recapture thrash. THREEFRY gone from temp-0 graph (rollout 35→33 kernels/token; RNG was full-vocab). Tests+mypy+ruff green. Note: old temp-0 path broke logit ties randomly; argmax (lowest index) is now the semantics. |
| 2026-08-18 | T1.6 | **done** | `task/T1.6-jit-input-cache` | `b5ddb2797` (+17/−3, jit.py only): caches per-input `substitute`+`unbind_all` keyed on view structure (interned-UOp identity; unsound-key guard + 32-entry cap). `_prepare_jit_inputs` 51.6→22.6 µs/call (−56%). 126 jit tests + mypy + ruff green. Note: `test/test_jit.py` doesn't exist at baseline — suites are `test/backend/test_jit.py` + `test/unit/test_jit*.py`. |
| 2026-08-18 | T3.1 | **done** | `task/T3.1-device-map` | `59d1eb2dd` (+88/−6): `parse_device_map` (ranges/auto-even/dict) + `--device-map`; placement before `load_state_dict`; `.to()` seam in forward (zero-cost when trivial). Split CPU:0/CPU:1 tokens identical to single-device; KV+freqs follow activations for free. **JIT captures the mixed trace end-to-end** — only COPY spans devices (asserted per captured call). Found upstream fused-rand_like bug (see memory.md §4). Tests+mypy+ruff green. |

| 2026-08-18 | wave-1 integration | **done** | `integration/wave1` | all 5 task branches merged (`2362f8f86`). One conflict (T1.5×T3.1 in `forward`: greedy branch + device hops combined) + one interaction fix (T3.1 test used pre-T1.5 `rollout_jit` name → `jit[(False, True)]`). Combined suite: 2401 passed, mypy + ruff clean. |
| 2026-08-18 | wave-1 code review | **done — 10 findings** | — | high-effort review of `integration/wave1`: 6 CONFIRMED w/ repros. Theme: device_map + jit-input cache work for pure-attention but **break on recurrent (GatedDeltaNet/qwen3.6) models** — zero recurrent test coverage in the diff. Also: warmup leaves sampled jits cold; parse_device_map unvalidated; uncast single-level; from_gguf positional break; greedy split across layers. |
| 2026-08-18 | review fixes | **done — all 10 fixed** | `task/wave1-review-fixes` | 8 commits → `528b27ea9`; repro-first (F1-F4 matched reviewer's exact errors), 9 new tests each verified failing pre-fix (incl. recurrent device-map coverage). 2410 passed (2401+9), mypy+ruff clean. `integration/wave1` fast-forwarded to it. |
| 2026-08-18 | T3.2 | **done — Stage A proven cross-backend** | `task/T3.2-metal-cpu` | `95891e58f` (+83/−2): METAL+CPU split runs with identical tokens (tiny attn+recurrent configs AND real qwen3:8b, 4 METAL/32 CPU under llama-server memory pressure — safe). JIT: capture succeeds, METAL graph-batches, CPU sequential, COPY structurally ungraphable. **Trap found+fixed:** lazy weight-placement COPYs were captured and replayed every step (21 spurious COPYs/token) — `from_gguf` now force-realizes only moved params. **Boundary copy floor: ~750 µs fixed METAL↔CPU regardless of size** (<1M elems; bandwidth only above) — the T3.5 planning number. Env note: agent sandbox cannot `launchctl bootout` llama-server; workaround = `DEV=CPU` + minority-on-METAL. |
| 2026-08-18 | T1.10 | **done** | `task/T1.10-matvec-quant` | `92986eb08` (+64/−6, stacked on T1.2): two fixes — weight operand matched via `.ranges` walk (dequant expr, not bare INDEX), and reduce-range selection scans for the bare-term split range (`pm_split_ranges` decomposes K into `[4096,128,2,16]`; the old `[0]` assumption picked the wrong one — only visible through real `.realize()`, so a real-pipeline test was added). METAL: Q4_0 ~10→41 GB/s, Q6_K ~15→47 GB/s (~4x); **Q4_K membw 4x but wall-time flat → ALU-bound on 6-bit sub-scale unpack, MATVEC can't help it** (real finding for the bench session). test/opt 38 green, mypy+ruff clean. |
| 2026-08-18 | rand-fusion bug | **done — NOT reproduced** | `task/rand-fusion-bug-repro` | `3b3f71331`: ~1,270 trials incl. bit-exact numpy-threefry check of fused RNG values (80/80 match) — no bug found; original T3.1 report possibly a harness artifact (the repro effort briefly self-generated an identical false positive via an inverse-Gumbel sign error). memory.md entry downgraded; repro harness + skipped tripwire tests kept on branch. No upstream issue filed. |
| 2026-08-18 | T1.3 | **done** | `task/T1.3-gptoss` | `6eeb4f241`+`95b569e65`: full gpt-oss arch — sinks (manual softmax path), even-layer sliding window (hardcoded, not GGUF metadata), YaRN rope, clamped swiglu, MoE biases; registry `gpt-oss:20b/120b` → ggml-org MXFP4. Synthetic parity vs numpy, mutation-tested 4 ways. **`gpt-4o` tokenizer preset added** (llama.cpp regex ported verbatim, capturing→non-capturing groups): roundtrips + harmony special tokens verified against the real 11 GB GGUF metadata-only; test skips without the file. All suites+mypy+ruff green. **Deferred to bench session:** real 20b generation vs llama.cpp greedy; chunked-prefill × sliding-window KV reuse. |
| 2026-08-18 | T0.4 | **done — GREEN** | `task/T0.4-mocknv` | `24581b7d5` (MOCKNV_SETUP.md only, no code changes). test_tiny 19 passed under `DEV=MOCK+NV:PTX` (+ test_hcq 29 passed, hevc compile). Recipe corrected in env table above; reproduced from main checkout. **T2.1/T2.2 (transport lane) now unblocked.** |
| 2026-08-18 | T2.1 | **done** | `task/T2.1-copyout-parallel` | `e31bb62d5` (+30/−4): `_copyout` joins the shared 32×2MB round-robin pool; drain-of-N overlaps device filling N+1; full-device sync removed from HW path. New wraparound test mutation-verified (catches 6% corruption when guard removed). Mock-NV 30+19 green; METAL doesn't route through this code. AMD USBIface (1 buffer) degenerates to old behavior. D2H bandwidth numbers deferred to real NV. |
| 2026-08-18 | T2.2 | **done** | `task/T2.2-pte-batch` | `1b8eabe52` (+148/−14): `set_entries` bulk PTE write (N writes → 1 slice write per contiguous run) + validation readback skipped when `vram.is_remote` (N reads → 0; `NV_VALIDATE_REMOTE=1` restores; AM byte-for-byte unchanged, guarded by test). Counting-iface tests verify N-independence at N=16/256. Mock-NV 19+29 green, AM external tests 7 green, mypy+ruff clean. Latency numbers deferred to dock. |
| 2026-08-18 | bench prep | done | — | qwen3:8b Q4_K_M (4.7 GB) pre-fetched into `~/Library/Caches/tinygrad/downloads/` via `tinygrad.llm` fetch — T0.1/T0.3 starts warm |

| 2026-08-18 | wave-2 integration | **done** | `integration/wave1` → `8d971f383` | all 8 wave-2 branches merged. Conflicts: heuristic.py (F9 while-uncast × T1.10 mat_ranges — combined) and model.py from_gguf (T1.3 config fields × device_map ctor — combined). Full suite **2420 passed**, mypy + ruff clean, mock-NV 49 passed, external NV/AM 11 passed. |
| 2026-08-18 | upstream sync | **done** | `integration/wave1` | merged `upstream/master` @ `2cfb421a8` (4 commits past baseline: KernelCountException, casted-const renderer migration, gptoss zero-2, CI deps) — zero conflicts, all gates re-green (2420 + mock-NV 49 + mypy + ruff). Task branches stay based on `af2a43c85`; rebase individually at upstream-PR time. |
| 2026-08-18 | T1.8 | agent running | `task/T1.8-attn-hook` | wave 3: attention hook + naive Metal custom kernel, parity + kernel-count |
| 2026-08-18 | T1.1a | agent running | `task/T1.1a-fp16-kv` | wave 3: fp16 KV + accuracy table (perf = T1.1b, next bench window) |
| 2026-08-18 | T3.3 | agent running | `task/T3.3-moe-placement` | wave 3: experts:<dev> device_map extension + olmoe METAL/CPU |
| 2026-08-18 | T4.1 | **done — awaiting Artur's review/submit** | `task/T4.1-matvec-pr` | `5bef4f081` on upstream tip `e37b44d04` (5 commits past baseline, zero drift in touched files). One 29-line heuristic.py commit + 3 tests + PR_MATVEC.md. Re-verified on tip: fp16 gemv ~1.4x (105 vs 75 GB/s), Q4_0 ~4x (42 vs 11 GB/s); MV=0 ≡ unpatched (internal consistency). test/opt 38 green, mypy+ruff clean. NOT pushed. |
| 2026-08-18 | T0.1+T0.3 | **done** | `task/T0.3-bench-harness` | `fb2356ac0`: harness (`extra/bench_llm.py` wrapper + GB/s in `benchmark_llm.py`) + CSV + BENCH_NOTES.md. **METAL qwen3:8b Q4_K_M decode tok/s: llama.cpp 27.27 · upstream no-BEAM 4.92 · integration no-BEAM 7.28 (+48%) · upstream BEAM 12.86 · integration BEAM 14.44 (+12%, 53% of llama.cpp)**. Prefill flat no-BEAM (levers are decode-only); BEAM prefill slightly down on integration (single runs, unchased). llama-server stopped for the window and RESTORED after. |

## Parallelization notes

Wave-3 lanes (updated 2026-08-18; waves 1-2 complete, baselines measured):
**(a)** T1.8 attn hook → the remaining ~2x vs llama.cpp (T1.7 PCONTIG only as exploration, failure
analysis is a valid exit); **(b)** T1.1a fp16 KV (anytime) → T1.1b (bench window); **(c)** T3.3 MoE
placement (the flagship shape); **(d)** T4.1 PR prep + T4.2 Q4_K (independent); **(e)** T1.9 /
T2.3 / T2.5 / T4.4 fillers. Bench-window tasks (T1.1b, T4.3) need llama-server stopped — batch
them into one window. Merge caution: T1.1a, T2.5, T3.3 all touch `llm/model.py` (branch off
`integration/wave1`, coordinate rebases); T1.7/T1.8 race to the same goal, keep both branches.
Agent policy: one tight objective per agent, Sonnet at max effort, explicit STOP conditions,
commit early; verify premises with a control experiment before optimizing (see T1.4).
