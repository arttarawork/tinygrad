# Ampere over Thunderbolt

**Design doc: making tinygrad's NVIDIA path a first-class local-LLM backend on Apple Silicon**

| | |
|---|---|
| Status | Draft v1 |
| Date | 2026-08-18 |
| Baseline | `tinygrad/tinygrad` @ `af2a43c85` (v0.13.0-968, master as of today) |
| Hardware | MacBook Pro M3 Pro 36 GB (Metal, ~150 GB/s) + RTX 3090 24 GB (sm_86, 936 GB/s) in a TB/USB4 eGPU dock via TinyGPU |
| Goal owner | Artur |

This file is an untracked working copy at the repo root; all `file:line` references are against the baseline commit above.

---

## 1. Summary

TinyGPU gave Apple Silicon an Apple-signed path to NVIDIA compute, but LLM inference through it currently runs ~10x slower than the Mac's own iGPU under llama.cpp (independent benchmarks: 2.3–6 tok/s on Qwen3-8B Q4 through the eGPU vs ~74 tok/s on Metal, vs ~109 tok/s for the same card native on Linux). Measured link utilization during those runs was 1.2–1.6% — **the cable is not the constraint; the software is.** Meanwhile the AMD side of the same stack already does 18.5 tok/s on a 27B model over the same cable, which is proof the framework can get there.

Three goals, in priority order:

- **G1 — Single-eGPU speed.** Make `DEV=NV python -m tinygrad.llm` on the 3090-over-TB competitive with llama.cpp running the same GGUF natively on Linux (target: ≥60–80% of native decode speed for models that fit in 24 GB).
- **G2 — Heterogeneous pooling.** Run one model across Metal (~27 GB usable) + NV (24 GB) — ~50 GB of weights — with a per-layer/per-tensor device map, including the MoE placement policy (attention + shared weights + KV on the 3090, routed experts on the Mac).
- **G3 — Upstreamability.** Every change lands as a small, benchmarked PR that survives tinygrad's review culture; nothing depends on a long-lived fork.

An important honesty note on the baseline: tinygrad's published CI numbers run `JITBEAM=2` (beam-searched kernels); the defaults a fresh user gets are un-beamed (`BEAM=0`, `helpers.py:231`). Some of the public "10x slower" gap may be configuration, not capability. Workstream 0 exists to establish the real gap on this exact hardware before we spend effort.

## 2. Why this is winnable

| Measurement (external, 2026) | Value |
|---|---|
| tinygrad NV eGPU, Qwen3-8B Q4 decode | 2.3–6 tok/s |
| llama.cpp Metal (M4 Pro), same class | ~74 tok/s |
| Same 3090, native CUDA Linux, llama.cpp | ~109 tok/s |
| eGPU link utilization during tinygrad runs | 1.2–1.6% |
| tinygrad AMD eGPU (7900 XTX, M4 mini), Qwen 27B | 18.5 tok/s |

Decode is bandwidth-bound: an 8B Q4 model is ~4.6 GB of weights; at the 3090's 936 GB/s the theoretical ceiling is ~200 tok/s. Everything between 6 and ~100+ tok/s is software: kernel quality, launch/sync overhead, and transport behavior — all enumerated below with file references.

## 3. The system today

Findings from a full read of the stack at `af2a43c85`. Legend: ✅ already right, ⚠️ gap, ⛔ blocker.

### 3.1 Transport & driver (TinyGPU → PCIe tunnel)

The macOS path is tinygrad's **own GSP-firmware userspace driver** (`PCIIface`, `ops_nv.py:555`), not NVIDIA's; `ops_cuda.py` is unused on Mac. Device access goes through `APLRemotePCIDevice` (`support/system.py:414-445`): a Unix-socket RPC to the TinyGPU.app server process, which owns the DriverKit extension (`extra/usbgpu/tbgpu/installer/`). The USB4/TB tunnel itself is transparent PCIe.

- ✅ **MMIO writes are fire-and-forget** — `sendall` with no reply (`system.py:324-327`, server `continue`s without responding). Doorbell = 3 async socket writes.
- ⚠️ **MMIO reads block on a full socket round trip** (`system.py:317-322`).
- ✅ **Host memory is zero-RPC**: `alloc_sysmem` returns an shm fd, pinned for DMA by the DEXT, `mmap`ed by Python as a plain pointer (`system.py:439-445`, `server.c:129-165`). Signals and staging buffers live here — host polling is a plain load.
- ⚠️ **CPU access to VRAM is per-access socket RPC** (`system.py:277-279` maps BAR through `RemoteMMIOInterface`).
- ⚠️ **Page tables live in VRAM and are written one 8-byte PTE per socket message** (`nvdev.py:34,48-49`), with *blocking* readback in `map_range`'s validation pass (`memory.py:204-206`). Every alloc/map pays this.
- ⚠️ **`is_bar_small()` (`system.py:255,266`) is a single switch** deciding whether kernargs (16 MB, `hcq.py:416`), the command page (`ops_nv.py:623`), and bound queues live in fast shm or behind per-dword RPC. `resize_bar` is a **no-op** in the DEXT (`server.c:228-229`) — we inherit whatever BAR macOS assigned. Unaudited on this dock.
- ⚠️ DEXT hard limits: 40-bit DMA addressing, **max 32 DMA segments per allocation** (`TinyGPUDriver.cpp:124,145`).

### 3.2 Runtime (HCQ)

- ✅ `HCQGraph` collapses a captured model to **one submit per device per step**, with symbolic patching of only changed words (`graph/hcq.py:263-302`, `hcq.py:216-227`).
- ⚠️ **Eager (non-JIT) path is ~1 submit per kernel** (`hcq.py:355-381`) — BEAM search and warmup run here.
- ⚠️ **`_copyout` is serialized**: single 2 MB staging buffer + full device sync per chunk (`hcq.py:596-609`). `_copyin` is properly pipelined across 32 staging buffers (`hcq.py:559-576`). Asymmetric for no reason.
- ⚠️ **`NVCommandQueue.bind` writes the queue one 32-bit word at a time** (`ops_nv.py:105-112`) — catastrophic if that buffer lands on the RPC side of §3.1's switch.
- ⚠️ **NV has zero eGPU-aware tuning.** AMD has a dozen `is_usb()` adaptations (small rings/kernargs/sigalloc, submit alignment, custom copies — `ops_amd.py:538-544,980-994`); `ops_nv.py`'s only macOS concession is a VA base shift (`ops_nv.py:380`).
- ⚠️ **NV is not on hcq2** — all recent speed work (`hcq2.py`) is AMD/CPU-only (`hcq2.py:25`).

### 3.3 Compilation

`NVDevice` renderer priority: `[CUDARenderer, PTXRenderer, NVCCRenderer, NAKRenderer]` (`ops_nv.py:634`).

- ✅ On macOS, CUDA compiles route through a **persistent Docker compile server** (one arm64 container per session, length-prefixed pipe protocol; added yesterday, `a746861ac`) with sqlite disk caching (`~/Library/Caches/tinygrad/cache.db`).
- ✅ **NAK** (Mesa's NVIDIA compiler via the `tinymesa` wheel) is a **Docker-free path**; CI boots the eGPU with `DEV=PCI+NV:NAK` (`benchmark.yml:419`). ⚠️ But `NAKRenderer` exposes **no tensor cores** (`renderer/nir.py`).
- ✅ sm_86 tensor cores via `CUDARenderer`: fp16→fp32, **bf16→fp32**, fp16→fp16, tf32 (`codegen/opt/tc.py:74-98`). ⚠️ `PTXRenderer` silently **drops bf16 MMA** (`ptx.py:140`). No int8/fp8 on Ampere (fp8 correctly gated to sm_89+).
- ⚠️ **BEAM is off by default**; docs recommend `JITBEAM=2`, CI benchmarks use it, fresh installs don't get it. Beam + Docker compile server = slow first search (parallel compile works, `search.py:126`, but each candidate is a container round trip).

### 3.4 Codegen (what fuses, what doesn't)

- ✅ **Quantized weights are never materialized.** GGUF dequant (Q4_0…Q6_K, IQ*, **MXFP4** — `llm/gguf.py:20-119`) is expressed as tensor ops that rangeify fuses into the consuming matmul. Verified: a Q4_K gemv is 1 kernel reading 149 KB of quantized bytes, not 524 KB of fp16. This is llama.cpp's most important trick, already present.
- ⛔ **No fused attention by default.** The softmax reduce blocks fusion (`rangeify.py:266`), so attention is 5 kernels/layer/token with ~5 passes over the score matrix; the flash path exists only behind `PCONTIG>2` (`test_rangeify.py:158-178`) and the relevant fusion tests are all `@unittest.skip("needs RANGEIFY>1")` (`test_softmax_fusion.py:97,109`). Escape hatch: `Tensor.custom_kernel` is real and tested; ThunderKittens FA kernels exist in `extra/thunder/cuda/`.
- ⚠️ **The batch-1 MATVEC heuristic misses the kernels that matter.** CONFIRMED on METAL 2026-08-18 (T1.2). Its guard requires `MUL(INDEX, INDEX)` as the reduce body (`codegen/opt/heuristic.py:60-78`). fp16 gemvs wrap it in CAST — fixed on `task/T1.2-matvec-cast` (56→100 GB/s kernel-side on a 4096² fp16 gemv). Quantized gemvs are a **separate, deeper miss** (weight operand is the whole fused dequant expression; GGUF block substructure splits the row axis into multiple global axes) — tracked as T1.10, the dominant decode kernels for Q4_K models.

### 3.5 The LLM app (`tinygrad/llm`)

GGUF-only loader; one generic `Transformer` covering llama/qwen2/3, GLM, OLMoE, Kimi, DeepSeek-MLA, and Qwen3.5/3.6 SSM blocks, incl. A3B MoEs (`llm/model.py:340-451`, registry `cli.py:76-94`). OpenAI-compatible server + web chat (`llm/serve.py`).

- ✅ MoE executes as **true indexed gather** of the k selected experts (3 matmul kernels regardless of expert count, no masked-dense, no weight copy — `model.py:21-27,109-135`). Architecturally right.
- ✅ Two symbolic `TinyJit`s (prefill/rollout) — one captured graph serves all positions and chunk sizes (`model.py:355-367,464-484`); sampling (Gumbel argmax) stays on device.
- ⛔ **KV cache and activations are fp32** (`model.py:200-204`, `dtypes.default_float`). At 8k context on an 8B model that is ~2.15 GB read per token that fp16 would halve. Likely the single largest decode-speed lever in the whole stack.
- ⚠️ **Full device sync + host round trip per token**: `out.item()` (`model.py:482`) triggers device synchronize before a 4-byte copy (`hcq.py:596-597`), plus a small out-of-graph copy (`jit.py:184-188`) and ~0.5 ms/token of uncached Python in `_prepare_jit_inputs` (`jit.py:200-218`).
- ✅ MoE top-k routing is already minimal — CORRECTED 2026-08-18 (T1.4): `pairwise_topk` realizes exactly 1 kernel/layer (the rank reduce; scatter/slice/cast inline into consumers), which is the rangeify floor. The remaining ~3 small kernels/layer are the caller's probs path (gather + softmax stats, `model.py:124-127`). ~~Full-vocab RNG runs even at temperature 0~~ — fixed on `task/T1.5-temp0-rng`.
- ⚠️ **No gpt-oss architecture** in the registry despite MXFP4 dequant and grouped-MoE machinery existing in-tree (`GROUPED_MOE`, mlperf/example side).
- ⚠️ **`tinygrad/llm` has no multi-device support at all** — no shard, no device flag beyond `DEV=`. Sharding exists only in legacy `examples/llama3.py --shard`.
- Load path: whole GGUF is one H2D blob then per-tensor slices (`gguf.py:134,148`) — ~2x model size transient and the dominant startup cost over TB.

### 3.6 Multi-device (the pooling question)

Tensor-parallel sharding exists and works — homogeneous only. Three blockers for `("METAL","NV")`:

1. ⛔ **One kernel binary is compiled for `device[0]` and launched on every device** (`engine/realize.py:249-252,169-179`) — a Metal lib would be handed to `NVProgram`.
2. ⛔ **Non-copy kernels assert same-device** (`schedule/__init__.py:141`).
3. ⛔ **Graph capture requires all-HCQ devices** (`graph/hcq.py:318-336`); Metal isn't HCQ, so a mixed workload loses graphing entirely.

Plus: cross-backend copies fall to a synchronous host-visible bounce (`realize.py:157-167`) — though on Apple Silicon `MetalAllocator._as_buffer` is shared-storage host memory, so NV→Metal is a *single* DMA into Metal-visible memory already; the missing piece is doing it async and graph-friendly. 2-device allreduce always takes the naive full-buffer path (`schedule/allreduce.py:6-58`). **No pipeline parallelism exists anywhere.** The per-block `@function(precompile=True)` boundary (`model.py:145-151`) is a natural pipeline seam.

## 4. The plan

Ordered by measured-impact-per-effort; each workstream lists exit criteria. WS0/WS1/WS2 serve G1; WS3/WS4 serve G2; WS5 serves G3.

### WS0 — Measure on the real hardware (days)

The public numbers were taken by third parties with unknown flags; the repo's own CI numbers are beam-tuned. First deliverable is a truth table for *this* MacBook + *this* 3090:

1. `python -m tinygrad.llm -m qwen3:8b --benchmark --warmup` under: `DEV=METAL`, `DEV=NV`, `DEV=NV JITBEAM=2`, `DEV=NV:NAK`, and the same GGUF under llama.cpp Metal and (borrowed Linux box or dual-boot) native CUDA.
2. Attribution per phase with `DEBUG=2` + VIZ profiler: kernel time vs gap time vs host time; confirm/refute the MATVEC-miss hypothesis (§3.4) with `DEBUG=3`.
3. Check §3.1's `is_bar_small()` on this dock and where kernargs/cmdq actually land.

**Exit:** a table + flamegraph naming the top 3 bottlenecks by measured milliseconds, not by reading code.

### WS1 — Decode-path kernels & dtypes (the big single-GPU wins)

1. **fp16 KV cache + activations.** Try `DEFAULT_FLOAT=HALF` end-to-end; if accuracy holds (perplexity spot-check), make the KV cache dtype explicit and default-fp16 (`model.py:203`). Expected: up to ~2x at long context. Smallest diff, largest payoff.
2. **MATVEC heuristic sees through CAST** (`heuristic.py:60-78`). Small, targeted, benchmarkable — exactly the PR shape upstream likes.
3. **Beam ergonomics:** default `JITBEAM=2` for the `llm` CLI (or ship a beam cache for common models); document the Docker-compile-server interaction. Free perf for every user.
4. **Fused attention**, three routes in order of preference: (a) push `PCONTIG>2` online-softmax to work for decode shapes and un-skip the fusion tests; (b) a `Tensor.custom_kernel` flash kernel for sm_86 wired behind the existing `STUB_ATTENTION`-style hook; (c) live with 5 kernels but shrink passes via grouped multi-output kernels. This is the deepest kernel work in the plan.
5. **Trims:** single-kernel top-k routing; skip RNG at temperature 0; cache `_prepare_jit_inputs`.
6. **gpt-oss architecture in `tinygrad/llm`** — the dequant + grouped-MoE pieces exist; wiring the arch adds the marquee open-weights MoE family (20B fits the 3090; 120B motivates WS3).

**Exit (G1/M2):** ≥60% of native-Linux llama.cpp decode on the same GGUF for 8B Q4 and 27B Q4 on the eGPU.

### WS2 — Transport & runtime tuning for the tunnel

1. Port the AMD `is_usb()` playbook to NV keyed on the remote iface: small kernargs/sigalloc/rings, submit sizing (`ops_amd.py:980-994` as the template).
2. Parallelize `_copyout` like `_copyin` (`hcq.py:596-609`); batch `bind()`'s per-word writes into bulk RPC writes.
3. **Batch PTE writes** and skip/defer the blocking validation reads on remote ifaces (`nvdev.py:48`, `memory.py:204-213`) — this is alloc/load latency, felt at model load and JIT capture.
4. **Kill the per-token sync:** keep the sampled token on device, drain every N tokens for streaming, overlap the 4-byte copyout with the next graph launch (`model.py:478-484`).
5. Later, follow upstream: NV on hcq2 when it lands there.

**Exit:** per-token host+transport overhead <0.5 ms; model load time within 2x of native.

### WS3 — Heterogeneous Metal+NV pooling (G2)

**Stage A — pipeline split (practical, sidesteps all three ⛔s).** Don't shard tensors; place *layers*. Each kernel stays single-device; only activations cross (a few KB/token — trivial even over a socket). Work: a device-map API in `tinygrad/llm` (`--device-map`, auto-split by free memory), per-layer KV cache device, weight loading to mapped devices (loader honors pre-placed params, `nn/state.py:211-214`), and verifying `TinyJit` captures a mixed-device trace with per-backend graphs and eager copies at boundaries. The `@function` per-block seam (`model.py:145-151`) is where to cut. **MoE placement policy** is sub-layer pipeline: attention + norms + KV on NV, routed-expert FFN tensors on METAL — activations hop twice per MoE layer, expert reads happen at Mac bandwidth, hot path at 3090 bandwidth.
- Bandwidth math for the flagship target (Qwen3.6-35B-A3B, Q8, ~37 GB pooled): ~1.7 GB active reads/token split ~60/40 across 936/150 GB/s ⇒ ~5.6 ms ⇒ **~150 tok/s ceiling**; even at 25% efficiency that's ~35 tok/s for a 35B-class model no single device here can hold at Q8. Dense 70B Q4 pooled: ~7 tok/s ceiling (Mac share dominates) — capacity, not speed.

**Stage B — copy quality.** Zero-copy bridge: TinyGPU sysmem is shm host memory and Metal shared-storage buffers are host memory; wrap one as the other (`BufferSpec.external_ptr` exists on Metal, `ops_metal.py:157`; `PCIIfaceBase.map` can register host pages for NV DMA, `system.py:289-295`). Then make cross-backend copies async (host-bridged event: HCQ signal word ↔ `MTLSharedEvent`).

**Stage C — true mixed-backend kernels (upstream-hard, optional).** Per-device compile in `pm_compile` (`realize.py:249-252`) instead of `device[0]`, relax the same-device assert for multi-lowered graphs, and a graph story for non-HCQ devices. Honest verdict: tensor-parallel decode over TB is latency-bound (2 un-graphed cross-copies × per-layer) and will not beat Stage A for this hardware; Stage C matters for upstream generality, not for this rig.

**Exit (M3):** Qwen3.6-35B-A3B at Q6/Q8 pooled ≥15 tok/s; a 40+ GB dense model runs end-to-end.

### WS4 — Capacity map (what runs, when)

Budgets: NV ~22.5 GB usable; Metal ~27 GB default wired limit (raiseable to ~30–31 via `iogpu.wired_limit_mb`, leave ≥5 GB for macOS). Pooled ≈ 49–53 GB.

| Model | Quant / size | Where | When |
|---|---|---|---|
| Qwen3-8B | Q4_K ~4.6 GB | NV alone | M1 (fast), the benchmark workhorse |
| Qwen3.6-27B / Qwen3.8-27B* | Q4 ~16 GB | NV alone (or Metal) | M1–M2 daily driver |
| Qwen3-30B/3.6-35B-A3B | Q4 ~17–19 GB | either alone | M1; MoE placement testbed |
| gpt-oss-20b | MXFP4 ~12 GB | NV alone | after WS1.6 |
| Qwen3.6-35B-A3B | Q8 ~37 GB | **pooled** | M3 flagship |
| 70B-class dense | Q4 ~40 GB | pooled | M3 (capacity proof) |
| gpt-oss-120b | MXFP4 ~60 GB + KV | pooled, stretch | M4 — needs wired-limit push + sub-MXFP4 requant or short context; borderline by ~8–10 GB |

*Qwen3.8-27B needs its arch verified against the qwen3.6 branch — likely near-free.
KV-cache quantization (q8 KV) is the natural follow-on lever once fp16 KV lands.

### WS5 — Benchmarks, CI, and upstream process (G3)

1. A pinned harness: same GGUF through `tinygrad.llm` and `llama-bench` (Metal + native CUDA), CSV via `BENCHMARK_LOG`, tracked in-repo alongside `extra/benchmark_llm.py`.
2. Extend the existing CI eGPU job (currently boot-test only, `benchmark.yml:416-421`) with a small decode benchmark once numbers stabilize.
3. Process realities: tinygrad optimizes for line count and measured wins; maintainers have reverted AI-generated changes before (`960430a5e` reverting `ac1291450`, commit message "ai slop"). Every PR from this effort: small, single-lever, with before/after tok/s on named hardware, hand-verified. Bounty board and the `#learn-tinygrad` Discord are the entry points; the exo #1904 crowd is the audience for Stage A once it demos.

## 5. Milestones

| | Deliverable | Acceptance |
|---|---|---|
| **M0** (days) | Baseline truth table + bottleneck attribution | top-3 costs named in ms; MATVEC + BAR hypotheses confirmed/refuted |
| **M1** (~weeks) | fp16 KV, MATVEC fix, beam defaults, sync trims | ≥5x current public eGPU decode (≥30 tok/s Qwen3-8B Q4); 27B usable |
| **M2** | Fused attention + transport tuning | ≥60–80% of native-CUDA llama.cpp on 8B/27B; prefill ≥ native Metal |
| **M3** | Pipeline device map, Metal+NV pooled | 35B-A3B Q8 ≥15 tok/s; 40 GB dense runs; demo posted to exo #1904 / tinygrad Discord |
| **M4** (stretch) | Zero-copy bridge, gpt-oss-120b attempt, upstream Stage C groundwork | pooled overhead <10% vs single-device extrapolation |

## 6. Risks

- **Upstream churn.** 880 commits since June; rangeify and hcq2 are moving targets. Mitigation: small patches, rebase weekly, no long-lived fork (G3 is a goal precisely because of this).
- **The DEXT is not ours to change.** TinyGPU.app ships signed from `tinygpu_releases` (pinned commit, `system.py:419-425`); anything requiring DEXT changes (segment limits, resize BAR, bulk-PTE RPC verbs server-side) needs tiny corp buy-in. Everything in WS1/WS2 except possibly bulk-PTE lives on the Python side.
- **Flash-attention fusion may stay experimental.** `PCONTIG>2` is explicitly unfinished; the custom_kernel fallback de-risks WS1.4.
- **fp16 KV accuracy** needs validation per family (SSM/MLA blocks may be more sensitive).
- **Metal side has its own ceiling.** M3 Pro is 150 GB/s — pooled dense models are Mac-bandwidth-bound; the pooling win is MoE-shaped (WS3 math). Set expectations accordingly.
- **Solo-maintainer risk on our side.** This is a hobby-scale effort against a fast codebase; milestones are ordered so each stopping point leaves standalone value (M1 alone = a genuinely fast eGPU).

## 7. Appendix

### 7.1 Env-var cheat sheet (current tree)

| Var | Effect |
|---|---|
| `DEV=NV` / `DEV=NV:NAK` | eGPU backend; NAK = Docker-free compile, but no tensor cores |
| `JITBEAM=2` | beam-search kernels inside JIT capture (cached in `cache.db`) |
| `REALIZE=1` | materialize dequantized fp16 weights (faster today, 2–4x memory) |
| `HALF=1` (default) | cast dequant output to fp16 |
| `DEFAULT_FLOAT=HALF` | fp16 activations + KV cache (WS1.1 experiment) |
| `PCONTIG=2` | experimental partial-contiguous fusion (flash-attention path) |
| `DEBUG=2/3` | kernel trace / heuristic decisions (`MATVEC:` lines) |
| `VIZ=1` | profiler UI |

### 7.2 Key-file index

| Area | Files |
|---|---|
| macOS transport | `tinygrad/runtime/support/system.py:255-460`, `extra/usbgpu/tbgpu/installer/` |
| NV driver/runtime | `tinygrad/runtime/ops_nv.py`, `support/nv/nvdev.py`, `support/hcq.py`, `runtime/graph/hcq.py` |
| Compile | `support/compiler_cuda.py`, `support/compiler_mesa.py` (NAK), `support/compileserver.py`, `codegen/opt/tc.py` |
| Codegen/fusion | `schedule/rangeify.py:221-287,354-377`, `codegen/opt/heuristic.py:60-78`, `codegen/opt/search.py` |
| LLM app | `tinygrad/llm/{model,gguf,cli,serve}.py` |
| Multi-device | `tensor.py:534-589`, `schedule/multi.py`, `schedule/allreduce.py`, `engine/realize.py:157-179,249-252` |
