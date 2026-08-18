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
| `MOCKNV` | NV backend under gpuocelot PTX emulation (`extra/setup_mock_nv_osx.sh`; `MOCKIface` is `NVDevice.ifaces[2]`, `ops_nv.py:583-586`) — functional correctness only, no perf |
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

- **T1.1 — fp16 KV cache + activations** `[MAC]` deps: T0.3 (WS1.1)
  Experiment `DEFAULT_FLOAT=HALF`; then explicit KV dtype at `llm/model.py:200-204` (default fp16,
  flag to keep fp32). Perplexity spot-check vs fp32 (few prompts, logprob diff), long-context decode
  delta on Metal — Metal is bandwidth-bound too, so the win shows up *now*. Watch SSM/MLA blocks.
  *Done when:* accuracy delta quantified + decode speedup measured; upstream-PR-shaped diff.
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
- **T1.4 — Single-kernel MoE top-k** `[ANY]` deps: — (WS1.5)
  Replace `pairwise_topk`'s 4-kernel O(E²) path (`llm/model.py:35-41`) with fewer launches.
  Kernel-count regression test on NULL; correctness on olmoe/qwen3-30b-a3b (Metal).
  *Done when:* MoE layer kernel count drops ≥3; outputs unchanged.
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
- **T2.5 — Amortize the per-token sync** `[MAC]` deps: T0.1 (WS2.4)
  `llm/model.py:478-484`: keep sampled tokens on device, drain every N for streaming; overlap the
  copyout with the next launch. Benefits Metal now, matters more over the socket later.
  *Done when:* host-visible stall/token down on Metal; streaming UX unchanged (N≤4).

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
- **T3.2 — Heterogeneous pipeline: METAL+CPU rehearsal** `[MAC]` deps: T3.1
  Swap one side for CPU. Verify only COPY crosses backends, JIT capture degrades gracefully to
  per-backend graphs + eager boundary copies, and measure boundary cost/token.
  *Done when:* qwen3-8b runs layers split METAL/CPU, output correct, boundary cost quantified.
- **T3.3 — MoE placement policy** `[MAC]` deps: T3.2 (WS3.A)
  Sub-layer split: attention+norms+KV on device A, routed-expert FFN tensors on device B
  (olmoe or qwen3-30b-a3b, METAL+CPU). This is the exact shape of the eventual NV+METAL flagship.
  *Done when:* MoE model runs with experts on the second device; per-token hop cost measured.
- **T3.4 — Zero-copy bridge spike (Stage B)** `[MAC]` deps: T3.2 (WS3.B)
  Wrap shared host memory across backends: `BufferSpec.external_ptr` on Metal (`ops_metal.py:157`)
  over a CPU/NV-visible buffer; sketch the HCQ-signal↔`MTLSharedEvent` host bridge.
  *Done when:* one boundary copy eliminated in the T3.2 pipeline, measured.
- **T3.5 — Boundary-copy microbenchmark** `[ANY]` deps: T3.1
  Tool measuring cross-backend copy latency/bandwidth vs size (the WS3 planning number).
  *Done when:* table for METAL↔CPU now; rerun on METAL↔NV post-dock.

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
| 2026-08-18 | T1.4 | agent running | `task/T1.4-topk-1kernel` | wave 1 |
| 2026-08-18 | T1.5 | **done** | `task/T1.5-temp0-rng` | `f53ceb67f` (+51/−7): greedy uses `temperature=None` sentinel from `generate()`; jits keyed `(is_prefill, greedy)` — no recapture thrash. THREEFRY gone from temp-0 graph (rollout 35→33 kernels/token; RNG was full-vocab). Tests+mypy+ruff green. Note: old temp-0 path broke logit ties randomly; argmax (lowest index) is now the semantics. |
| 2026-08-18 | T1.6 | **done** | `task/T1.6-jit-input-cache` | `b5ddb2797` (+17/−3, jit.py only): caches per-input `substitute`+`unbind_all` keyed on view structure (interned-UOp identity; unsound-key guard + 32-entry cap). `_prepare_jit_inputs` 51.6→22.6 µs/call (−56%). 126 jit tests + mypy + ruff green. Note: `test/test_jit.py` doesn't exist at baseline — suites are `test/backend/test_jit.py` + `test/unit/test_jit*.py`. |
| 2026-08-18 | T3.1 | agent running | `task/T3.1-device-map` | wave 1 |

## Parallelization notes

Independent start-now lanes for concurrent agents: **(a)** T0.1→T0.3→T1.1/T1.2 (Mac, perf),
**(b)** T1.3 gpt-oss (Mac, model code), **(c)** T3.1→T3.2 pooling (any→Mac), **(d)** T0.4→T2.1/T2.2
(mock-NV, transport). T1.4-1.6 are small fillers for any idle agent.
Merge-order caution: T1.1, T1.3, T2.5, and T3.1 all touch `llm/model.py` — coordinate rebases;
T1.7 and T1.8 are alternatives racing to the same goal (first one to work wins, keep both branches).
