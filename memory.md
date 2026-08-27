# memory.md — project memory for the Ampere-over-Thunderbolt effort

Companion to `NV_LLM_DESIGN.md` (the design doc, same directory). This file holds the context,
decision history, external research, and repo findings that the design doc deliberately leaves out.
Written 2026-08-18 against `tinygrad @ af2a43c85` (v0.13.0-968). Fork: `arttarawork/tinygrad`.
Published copy of the design doc (2026-08-18 snapshot — SUPERSEDED by the repo copy, which has
all later corrections; republish before sharing): https://claude.ai/code/artifact/f430cb11-aade-418c-9a8f-b63dae7b7988

## 1. Who / hardware / why

- Owner: Artur (arttarawork). Daily machines: **MacBook Pro M3 Pro, 36 GB unified** (~150 GB/s,
  ~27 GB GPU wired limit by default, raiseable to ~31 GB via `sudo sysctl iogpu.wired_limit_mb=…`,
  keep ≥5 GB for macOS) and a Bazzite (Fedora atomic) gaming PC with an **RX 9070 XT**.
- The **RTX 3090 (24 GB, sm_86, 936 GB/s) currently has no PC to live in.** Two candidate homes:
  1. Buy a new everything-but-GPU PC, move the 9070 XT there, turn the old PC into a dedicated
     Linux 3090 inference server (RECOMMENDED long-term; needs 750W+ PSU check and ideally 64 GB RAM
     to unlock gpt-oss-120b via llama.cpp `--n-cpu-moe`).
  2. A TB/USB4 eGPU dock on the MacBook via TinyGPU — the route this repo effort explores.
     eGPU on Apple Silicon is otherwise impossible (no native support, no NVIDIA drivers, no CUDA).
- Original motivating question: can the Mac's 36 GB and the 3090's 24 GB be pooled (~50 GB) for
  larger local models, ideally with MoE hot-path weights on the 3090 and routed experts on the Mac.
  Answer so far: yes in llama.cpp over a network (capacity, not speed); in tinygrad it's the
  design doc's WS3 (pipeline split — nobody has built it yet, anywhere).

## 2. Decision history & fallback routes (if the tinygrad route stalls)

- **llama.cpp RPC** is the mainstream way to pool Mac+CUDA today: `rpc-server` on one box,
  `--rpc host:port` on the other. Real Metal+CUDA benchmarks (Mac Studio M2 Ultra + DGX Spark,
  10 GbE): 7B prefill 4.2x faster, decode 91.8→52.7 tok/s; 72B decode 11.1→5.9 tok/s.
  "RPC is for capacity, not speed" — only worth it for models that fit neither device. Same-version
  builds required both ends. Best link: TB4/USB4 cable as IP network (~10-20 Gb/s, Linux picks it
  up via `thunderbolt-net`). MoE placement trick: `--override-tensor "ffn_.*_exps.*=<slow-dev>"`,
  `--n-cpu-moe N` for CPU-RAM experts.
- **PC-solo fallback** (once the 3090 has a Linux box with ≥64 GB RAM): gpt-oss-120b MXFP4 with
  experts in system RAM ≈ 15-30 tok/s — beats network pooling in speed and simplicity.
- **exo ecosystem status (Aug 2026):** mainline exo v1 rewrite is MLX/Apple-only (tinygrad engine
  removed). `Scottcjn/exo-cuda` (92★, active) restores tinygrad-CUDA on Linux — RTX 3090 verified —
  but Linux-only, no Mac hosts, no mixed Apple+NVIDIA clusters. `ArgentAIOS/nxo` dormant (0★).
  **exo-explore/exo#1904** asks for exactly our use case (pool Mac unified memory + TinyGPU eGPU):
  open, zero maintainer response. That issue's audience is who WS3 Stage A should be demoed to.
- **TinyGPU landscape:** Apple signed the DEXT April 2026 — first sanctioned eGPU path on Apple
  Silicon. tinygrad-only stack (no CUDA runtime for other apps; llama.cpp/Ollama/LM Studio cannot
  see the card — SEO posts claiming otherwise are wrong; official docs and code confirm).
  Best external data: lucebox.com/blog/egpu-myth — RTX 3090 through TinyGPU: 2.3-6 tok/s on
  Qwen3-8B Q4, link utilization 1.2-1.6% ("the cable is not the constraint"), vs ~74 tok/s
  llama.cpp Metal (M4 Pro) and ~109 tok/s native CUDA. tiny corp's own AMD headline: 18.5 tok/s
  Qwen 27B on 7900 XTX + M4 mini — the AMD path is ahead of NV. **No fork is ahead of mainline
  tinygrad; the main repo is the frontier.** Unknown flags on the lucebox runs — possibly
  un-beamed (CI uses JITBEAM=2), hence WS0.

## 3. Model landscape snapshot (Aug 2026) for target-picking

- **Qwen 3.8** (released ~Aug 3, Apache 2.0): Max = 2.4T MoE, 95B active — untouchable locally
  (~1.2 TB at 4-bit). 27B dense multimodal — the daily-driver class; ~16 GB at Q4 (fits 3090 or
  Mac alone; pooling pointless for it). Arch not yet verified in `tinygrad/llm` (qwen3.6 branch
  is probably close).
- In tinygrad's registry already: llama3.1/3.2, qwen3 (incl 30b-a3b), qwen3.5, **qwen3.6
  (27b, 35b-a3b)**, olmoe, moonlight-16b-a3b, glm-4.7-flash. NOT in registry: gpt-oss (dequant +
  grouped-MoE machinery exists in-tree; wiring the arch is WS1.6).
- Pool-worthy targets: Qwen3.6-35B-A3B @ Q8 (~37 GB), 70B dense @ Q4 (~40 GB),
  gpt-oss-120b MXFP4 (~60 GB — stretch, borderline by ~8-10 GB even with wired-limit push).
- Bandwidth constants for napkin math: 3090 = 936 GB/s, M3 Pro = ~150 GB/s, TB4 ≈ 4 GB/s,
  socket RPC round trip = the per-token latency floor.

## 4. Repo findings NOT in the design doc (supplementary detail)

Verified at `af2a43c85`; the design doc has the load-bearing items, these are the extras.

### Runtime / transport
- `ops_rdma.py` (105 lines) is a from-scratch **Mellanox ConnectX RDMA NIC driver**
  (vendor 0x15b3, dev 0x101b) for GPU↔GPU transfers between machines; `runtime/support/mlx/` =
  Mellanox, not Apple-MLX. Multi-box clustering is where upstream is headed (tinybox-scale).
- **peer_group oddity:** `RemotePCIDevice` sets `peer_group = sock.getpeername()[0]`
  (`system.py:387`); on AF_UNIX that's a path, so every TinyGPU device lands in peer group `"/"`.
  Probably unintended — relevant if anyone tries multi-eGPU on one Mac.
- `_copyin` is the template for fixing `_copyout`: 32 × 2 MB pinned staging buffers, round-robin
  (`hcq.py:534,559-576`). `copy_from_disk` uses io_uring when staging is a real `MMIOInterface`
  (true on macOS shm) — that's the fast weight-load lane.
- DEXT/server internals (`extra/usbgpu/tbgpu/installer/`): server process maps BARs and services
  MMIO through a **64 MB bounce buffer**; sysmem = `shm_open` + `PrepareDMA` (pins + writes an
  IOVA segment list into the region head) + fd passed back via `SCM_RIGHTS`. Entitlement pins
  NVIDIA vendor ID (separate signed build from AMD). TinyGPU.app pinned to release commit
  `c0d024f9…` (`system.py:419-425`).
- Compile server protocol (`a746861ac`, yesterday): persistent `ghcr.io/tinygrad/cuda-arm64:v2.3`
  container, length-prefixed stdin/stdout; for NV, NVRTC emits **CUBIN** (ptx=False). NAK comes
  from the `tinymesa==25.2.7.2` wheel; `warps_per_sm` knows sm_86 (=48); NAK renderer has no
  tensor cores and gates half on sm≥53.

### LLM app anatomy (measured on NULL device by the exploration agents)
- Dense decode = **14 kernels/layer** (5 of them the attention chain: QK^T, softmax max/sum/div,
  @V). MoE layer = 11 kernels. CORRECTION (2026-08-18, T1.4 control experiment): `pairwise_topk`
  itself is only **1** of those (the rank reduce — the O(E²) compare/scatter/slice all inline);
  the other ~3 "routing" kernels are the caller's probs gather + softmax stats. The original
  "4 pure routing overhead" attribution was wrong.
- MoE expert gather verified as true indexed load: E=64/k=8 test config reads ~930 KB vs ~6.3 MB
  masked-dense equivalent. The two `.contiguous()` in the expert path are deliberate
  ("moe speedup", commit `7ef901a81`) — don't remove them.
- Un-graphed host dispatch cost ≈ 1.1 µs/kernel (~0.5 ms/token at ~450 kernels on a 32-layer
  model) — collapses under CUDA Graph / HCQGraph, but `_prepare_jit_inputs` Python remains.
- Sampling is Gumbel-max argmax fused into the output-projection kernel; temperature is a Tensor
  (avoids recapture); full-vocab threefry RNG runs even at temperature=0. No top-k/top-p/penalties
  in `tinygrad/llm` (those exist only in legacy `extra/models/llama.py:145-193`).
- Chunked prefill `chunk_size=32`; prefix caching via `get_start_pos` (`model.py:456-462`) reuses
  KV when a new prompt extends the cached prefix (SSM blocks: strict prefix only).
- `serve.py` is a single-threaded `socketserver.TCPServer` — one request at a time, one KV cache.
  Zero occurrences of paged/continuous-batching/speculative anywhere in the repo. **CORRECTION 2026-08-27:**
  it is otherwise full-featured now — renders the chat template WITH `tools`, parses `<tool_call>` into OpenAI
  `tool_calls` (streaming deltas incl.), SSE, `/v1/models` — Hermes-compatible interface (→ TD.5 goal).
- Load path: whole GGUF → one device blob (`gguf.py:134 tensor.to(None).realize()`) → lazy
  per-tensor slices; ~2x model size transient; dominant startup cost over TB.
- `REALIZE=1` materializes dequantized fp16 weights — currently *faster* per token but costs full
  fp16 memory; the design doc's kernel work aims to make the default (fused dequant) win outright.

### Multi-device extras
- **T3.1 findings (2026-08-18, `task/T3.1-device-map`):** mixed-device (CPU:0/CPU:1) TinyJit
  capture works end-to-end with no fallback — rollout 63 PROGRAM + 3 COPY calls, only COPY spans
  devices, same-device assert never fires. KV cache + freqs_cis follow activations via
  `_init_state`'s `x.device` — no extra plumbing. Graph batching forms per-backend islands
  (METAL graphed, CPU sequential). **No free-memory query exists in Device/Allocator** (auto
  device_map splits evenly by layer count instead). Caveat: unrealized weights on a non-default
  map capture their lazy initializers into the JIT (re-run every step) — force realize for splits.
- **Suspected rand_like/fusion bug: NOT REPRODUCED (status downgraded 2026-08-18).** T3.1's agent
  reported temp=0 `generate()` emitting a non-greedy token (realized temperature + fused symbolic
  prefill; `u.realize()` "fixed" it), but its repro scripts were lost to a session reset. A
  dedicated characterization effort (`task/rand-fusion-bug-repro`, `3b3f71331`) could not reproduce
  it in ~1,270 trials across attention/MoE/SSM configs, METAL+CPU, BEAM=2 — including a bit-exact
  numpy threefry ground-truth comparison of the *fused* RNG values (80/80 matched to fp32 rounding,
  temp-independent). Cautionary finding: that effort briefly "confirmed" the bug via a sign error
  in its own inverse-Gumbel check (100% divergence, entirely self-inflicted) — the original report
  may be the same species of artifact. Tripwires kept: `extra/rand_fusion_bug_repro.py` +
  skipped tests in `test/unit/test_rand_fusion_bug.py` + `docs/rand_fusion_bug.md` (all on that
  branch). Do NOT file upstream; revisit only if a fresh repro appears on a real model.
- 2-device allreduce is always NAIVE (full buffer each way) regardless of size — ring only kicks
  in at ndev>2 and >256k elements (`allreduce.py`, `RING=1` default). For Metal+NV pooling this
  is another reason WS3 chose pipeline over tensor-parallel.
- `test/backend/test_multitensor.py` never mixes backends; `("CPU:1","CPU:2")` is the closest.
  `Tensor.shard` rejects re-sharding a multi-device tensor (`tensor.py:571`).
- 2-D sharding exists (`UOp.unshard` handles multiple sharded axes, `test_2d_shard_matmul`).
- Legacy `examples/mixtral.py` does MoE routing on the **CPU** (`.tolist()` per layer per token) —
  do not use it as a reference; `tinygrad/llm/model.py` is the good pattern.

### Process / CI
- CI `llmbenchmark` job runs `llama3.2:3b-f16` and `qwen3.6:35b-a3b` with
  `JITBEAM=2 IGNORE_BEAM_CACHE=1` on METAL/AMD/NV runners; eGPU-on-Mac CI is boot-test only
  (`DEV=PCI+NV:NAK`, `benchmark.yml:416-421`).
- Dev commands (AGENTS.md): `python -m pytest test/... -x -q -n12`, `python -m mypy tinygrad/`,
  `python -m ruff check .`; `tinygrad/viz/README.md` for rewrite/profiling debugging.
- Velocity: 880 commits Jun 1 → Aug 18. Hot areas: gpt-oss grouped MoE + MXFP4 kernels
  (fp4 asm gemm "6+ pflops"), kimi delta attention, qwen3.6, hcq2 (AMD/CPU only so far).
- Culture datum: commit `960430a5e` reverts `ac1291450` with message calling it "ai slop" —
  PRs must be small, hand-verified, benchmarked on named hardware.

## 5. Practical setup crib (Mac side)

- Install: `curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_tinygpu_osx.sh | sh`,
  approve the driver extension in System Settings; NV compiles need Docker Desktop
  (`extra/setup_nvcc_osx.sh`) unless using `DEV=NV:NAK` (no tensor cores).
- From a clone without install: prefix `PYTHONPATH=.`
- Bench one-liner: `DEV=NV JITBEAM=2 python3 -m tinygrad.llm -m qwen3:8b --benchmark --warmup`
  (first run pays beam search; results cache in `~/Library/Caches/tinygrad/cache.db`).
- Mock-NV (no hardware): `OCELOT_PATH=.venv/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX` — the DEV
  string must start with `MOCK`; dylib is CI's prebuilt gpuocelot v0.1.0 (~3 MB, 2 s download),
  NOT the sudo source-build in `extra/setup_mock_nv_osx.sh`. See MOCKNV_SETUP.md (T0.4 branch).
- Wired-limit bump for pooling experiments: `sudo sysctl iogpu.wired_limit_mb=31744`.

## 6. Session log

- **2026-08-18 (kickoff session):** Spot-checked the load-bearing file:line refs at HEAD
  (`8f56e0ecd`, == baseline for code): `model.py:27,129` (contiguous), `:35-41` (pairwise_topk),
  `:144-151` (@function seam), `:200-204` (KV fp32), `:358-364` (Gumbel), `heuristic.py:60-78`
  (MATVEC MUL(INDEX,INDEX) guard), `jit.py:200-218`, `ops_nv.py:583-586` — **all accurate**.
  Fixed environment drift the docs assumed away: no bare `python` (Homebrew python3.14, zero test
  deps) → created `.venv` (numpy, torch 2.9.1, pytest+xdist, hypothesis, z3, gguf, mypy 1.19.1,
  ruff 0.14.10); remotes were documented backwards → reality is `origin` = arttarawork fork,
  added `upstream` = tinygrad/tinygrad; no local `master` branch — task branches come off
  `af2a43c85` directly. `test/test_tiny.py` green on METAL. Launched wave-1 agents
  (T1.2, T1.4, T1.5, T1.6, T3.1) in isolated worktrees; status in TASKS.md.
- **2026-08-18 (Artur decision): AMD lane descoped.** T0.2 dropped; T2.1 rerouted to
  MOCKNV→CLOUD3090. Rationale: the 9070 XT box was only ever a real-HCQ stand-in for shared
  `hcq.py` validation pre-dock; a rented Linux 3090 is the same job on the *target* backend
  (NVKIface/sm_86) for ~$0.20/hr, and upstream CI's AMD runners cover the AMD side of any
  `hcq.py` PR. The Bazzite-box footgun note (AM/PCI path kills the display) is retained in
  CLAUDE.md in case the lane is ever revived.

- **2026-08-18/19 (waves 2-6, ~40 tasks):** full per-task record lives in TASKS.md's Status log —
  this entry is the durable digest. Landed on `integration/wave1`: MATVEC fp16+quant (Q4_0 4x,
  Q4_K 2x via d/dmin staging — Q4_K now faster than Q4_0), fp16 KV (SSM recurrent state fp32 by
  divergence experiment), jit-input cache, temp-0 RNG skip, drain_every, streaming GGUF load,
  KV-prealloc cap (the REAL load-memory hog — the doc's "2x load transient" was wrong), device_map
  incl. `experts:<dev>` + realize_placement, gpt-oss arch + gpt-4o tokenizer, copyout pipelining,
  PTE batching + remote validation skip, NV remote knobs, symbolic custom_kernel, SCACHE key fix
  (9 ContextVars; PCONTIG tests were self-referential). **Dead ends, proven:** custom_kernel
  attention pre-T4.7 (symbolic Tk), PCONTIG fusion (numerically wrong), METAL↔CPU zero-copy
  aliasing (hop cost is SYNC not memcpy — ~750 µs fixed floor is `waitUntilCompleted`), warp
  reduce as a renderer-local patch (framework-wide gap, ~300-550 lines). **Measured state:**
  qwen3:8b METAL decode 7.38 no-BEAM / 14.40 BEAM vs llama.cpp 27.07 (beam alone = 2.6x config
  gap); decode now attention/other-gated, not gemv-gated. Key design rules discovered: GGUF loads
  on the big-memory device (moving big tensors force-realizes them); 3 copies/MoE-layer not 2;
  TinyJit bakes cache-buffer identity (growable KV impossible without recapture). gpt-oss decode
  reads ~25x too many bytes (T4.11, open). PR queue (held): T4.9 → T4.7 → T4.2 → T4.1 pkg.
- **2026-08-19 (wave 7 + bench window 3 — pre-dock endgame):** T1.8c: tuned attention kernel now
  fires every token (kernel-side ceildiv chunking + tail mask; also fixed T4.7's compound-expr
  `to_kernel_param` gap — fold into that PR). Bench window 3 verdicts: **T4.8 warp-reduce FINAL
  NO-GO** (FAST_ATTN a wash on qwen3:8b even at 4k ctx) — fused attention on Metal closed until
  sm_86; **T3.6 signal bridge refuted pre-dock** (CPU-producer sync nearly free, Python dispatch
  ~300 µs dominates; capture-op sized 110-160 lines in SIGNAL_BRIDGE_NOTES.md, revisit at TD.3);
  **T4.10 closed** — gpt-oss divergence is FP drift near a tied argmax, chunking byte-invariant at
  real scale too; **gpt-oss 1.69 tok/s ROOT-CAUSED**: ~59 GB/token is real — MXFP4 dequant
  MATERIALIZES per decode token at 20B scale (two elementwise kernels = ~90% of step time; fuses
  fine at tiny scale, which is why T4.11 couldn't reproduce) → T4.13 (expected ~20-40 tok/s);
  JITBEAM makes it worse (−12.5%). New small bug T4.12: warmup() hardcodes chunk_size=32, jit key
  omits it → JitError on generate(chunk_size≠32). Headline stable: 7.38 no-BEAM. Docs current;
  ~48 tasks closed over 7 waves + 3 bench windows.
- **2026-08-19 (wave 8 — the MXFP4 fix):** T4.13 root-caused and FIXED the gpt-oss 59 GB/token:
  MXFP4 dequant's LUT gathers (`lut[codes]`) embed a buffer-reading REDUCE that rangeify's
  `buffer_in_reduce` refuses to fuse into the MoE `weight[sel]` gather → all 32 experts
  materialized per token. Not scale-dependent (T4.11 missed it via a coverage gap: fp32 weights in
  its byte test, one-tensor quantization in its GGUF test). Fix: LUTs → ALU bit-ops, bit-exact,
  **44x byte cut (1.04x analytic)**; real-model tok/s confirm pending next bench window
  (~1.69 → 20-40 expected). Lesson for the pattern library: an indexed LUT inside a dequant
  expression breaks gather fusion — prefer bit-ops for small decode tables. T4.12: prefill jit now
  keyed by chunk_size (resolve() default=True fallback on symbolic ranges was capturing every
  first step as prefill). PR queue now: T4.9 → T4.13 → T4.7+T1.8c-fix → T4.2 → T4.1 pkg.
- **2026-08-19 (bench window 4 — Phase 0 closes measured):** T4.13 confirmed at real scale:
  gpt-oss-20b decode **1.69 → 10.97 tok/s (15.52 with BEAM — beam flipped from −12.5% to +41.5%
  once the pathological kernels died)**, bytes 59.3 → 3.46 GB/token (≈ analytic). Long-context
  gpt-oss is now attention-COMPUTE-bound on Metal (decode −68.5% at 2k prompt, bytes only +11.6%)
  — the sm_86 tensor-core case for the dock. Cross-implementation FP drift vs llama.cpp is normal
  in this stack (llama3.2:1b diverges after 2 tokens PRE-wave-8; a gpt-oss prompt diverges at
  token 27 post-fix — dequant values bit-exact, fusion reorders accumulation; T4.10 class).
  Headline stable at 7.37/qwen3:8b. Bench branches (`task/bench-window-{2,3,4}`, T0.3 harness)
  stay unmerged by convention — CSV + BENCH_NOTES.md live on `task/bench-window-4` tip.

- **2026-08-20 (upstream landscape + the Watcharasorn find):** No open upstream PRs touch any of
  the nine PR-train topics — clear field. Upstream issue **#17316** (open, no maintainer reply)
  is our exact niche twice over: Qwen3.6-35B IQ3_XXS reading ~83 GB/token = **T4.13's LUT
  mechanism on IQ quants** (codebook LUT gathers → unfusable REDUCE → all experts materialize;
  his OLMoE-Q4_K-works contrast confirms it), and his comment thread's layer-count-driven fixed
  cost = our WS2 transport thesis. Upstream PR **#17446** (open since Aug 7) is a competing
  gpt-oss arch — ours (T1.3) is more complete; watch for sync conflicts.
  **github.com/Watcharasorn/mac-tinygpu-5070ti** = a working AG02-dock bring-up runbook
  (5070 Ti + M4 Pro): preflight criteria, power-ordering, the no-BAR stop rule, and real tunnel
  numbers (26 ms/token floor, ~3.5 ms/layer, 38 GB/s cap on an 896 GB/s card; llama.cpp native
  110-130 tok/s) — folded into TD.1/TD.2. He is the prototype audience for the TD.4 demo.
- **2026-08-20 (deep upstream sweep — the policy find and six technical cross-references):**
  **Discussion #14615: a disclosed-AI-assisted PR (tested, 100-run benchmarked, hand-verified)
  was CLOSED with "do not use ai"** — no maintainer reply in the discussion, no written policy.
  This is harder than the "ai slop revert" datum: even disclosed + validated AI use was rejected
  once. PR-train implication (Artur's call): (a) submit anyway with disclosure and exemplary
  rigor, accepting closure risk; (b) file the FINDINGS as issues with repros instead (SCACHE
  cross-serve + vacuous PCONTIG tests; the LUT-fusion mechanism on #17316) — high value, less
  policy exposure, patches stay on the fork for anyone; (c) ask on Discord first.
  Technical cross-refs: **#13707 (geohot)** wants JIT/schedule-cache consistency asserts and
  **#12514 (chenyuxyz)** flags SPLIT_REDUCEOP inconsistency — both squarely in T4.9's territory
  (frame T4.9 as answering maintainer-flagged concerns). **#17617** (Aug 20) requests lazy
  disk-loading of MoE experts — live demand for our T3.3+T1.9 machinery with a disk tier.
  **#17074** (lazy reader before view assign reads post-write values) is the engine-level cousin
  of our CI class-2 shared-unrealized-UOp corruption — upstream-known, open. **#16520** (rand
  seed reuse — two Tensor.rand get identical values under some realize orders) is the first
  suspect if the T3.1 rand ghost ever resurfaces. **#16595** (mtl_buffers_in_flight membership
  check → linear step-time growth) is the same cost family T3.4 measured (synchronize scaling
  150→1134 µs with in-flight work). **#16894** (Blackwell sm_120 TC nan under HALF+BEAM)
  validates the 3090/sm_86 choice. **#13263** ("NV,PCI,NAK → WPR2 is not initialized") goes on
  the TD.1 landmine list.

- **2026-08-21 (docs review + housekeeping):** Full review of TASKS.md / NV_LLM_DESIGN.md /
  memory.md / both CLAUDE.md files. Facts overtaken by events, now corrected: **Artur merged
  PR #1** — fork `master` = `457e1a915` (Phase 0 work + upstream `b8cc74ecf`); task-branch base
  convention moved to fork `master` (`integration/wave1` retired as a base, content identical);
  the `memory` branch was 72 commits unpushed (whole project record single-machine) — pushed,
  along with the 6 unmerged evidence branches; docs-push added to conventions. Also promoted to
  conventions: DEV=CPU pass on llm work, stagger master/branch pushes; bench choreography now
  notes the Hermes impact of stopping llama-server. Design doc: §5 re-dated (Phase 0 complete),
  risks gained #17446 (competing gpt-oss PR) and the #14615 G3-premise doubt. Housekeeping:
  stale `integration/wave1-local` branch deleted, clean agent worktrees pruned. NOT done
  (Artur's calls): upstream sync (32 commits past `b8cc74ecf`, incl. an llm kimi fix — due),
  PR-route decision, design-doc artifact republish.

- **2026-08-21 (upstream sync #3 + regression review):** merged `upstream/master` @ `80bf60d78`
  (32 commits) onto fork master — zero conflicts, all gates green (unit+opt 882 / backend 1255 /
  DEV=CPU 97 / mock-NV 49 / mypy+ruff). Notable incoming: kimi `resolve_linear_call` nested-scope
  fix (same file as T4.9, different region, no interaction), parallel-compile `engine/worker.py`
  (no T1.6 jit-cache interaction — backend suite green), fused_qkv_rope try 2 + fa changes are
  AMD `extra/thunder` only, `gptoss: save more` is mlperf-side only. **The find: upstream #17630
  added `test_chunked_prefill_kv_cache_matches_single_chunk` as `expectedFailure` — a real
  chunked-prefill KV-cache bug upstream knows about. Our tree passes it, at temp 0 and 1.0.
  Bisect (bug confirmed at `af2a43c85`): first fixed by T1.5 `f53ceb67f` (temp-0 RNG skip / jit
  re-key by (is_prefill, greedy)).** This is the strongest PR-train datum yet: an upstream-pinned
  failing test that a T1.5-carrying PR would flip green — input to the pending route decision.
  Push handoff: merge touches CI workflow files → Artur pushes `sync/upstream-2026-08-21:master`.
  Real-model bench check deferred to the next llama-server window.

- **2026-08-23 (upstream watch, 33 commits since sync #3 — not yet merged):** **geohot shipped
  fp16 KV upstream (`477b57380`, one line: `dtypes.half` hardcoded in `TransformerBlock._init_state`
  only)** — partial T1.1a preemption. Ours is the superset (MLA + SSM-conv fp16, recurrent-state
  fp32 by divergence evidence, `KV_F32` escape); next sync has a guaranteed small conflict in that
  hunk — resolution: keep ours. A T1.1a PR would now be "extend fp16 KV to MLA/SSM correctly."
  Transport lane heating up: USB-AMD copyin pipelined (`756e82e05` 2.6x, `3082956a1` async
  arm/drain, 323 MB/s) — same playbook as our T2.1, no file overlap (ops_amd/usb.py, not hcq.py);
  prior art for post-dock tuning. `4fd4eafb2` (nv: hevc) touches ops_nv.py + support/memory.py —
  T2.2's neighborhood, check hunks at sync; it also edits benchmark.yml and `298748ebd` edits CI →
  **next sync push needs Artur again** (workflow scope). Non-events: mxfp4-gemm cleanup deletes an
  extra/ file (no T4.13 contact); gptoss fused-ce is mlperf-side; weak-const test churn continues.
  **New open-PR watch list:** **#17493 "clean slate rangeify rewrite"** — the big one: T4.13's
  fusion fix, T1.4's 1-kernel floor, and T4.9's xfailed PCONTIG tests all sit on rangeify
  behavior; if it merges, re-validate all three and sync immediately. #17478 "hcq2: default"
  (the WS2.5 follow-upstream trigger approaching; NV still not on hcq2). #17567 llama BEAM nan
  fix (we bench with JITBEAM). #17446 (gptoss) unchanged-open; #17316/#17617 still open, no reply.

- **2026-08-23/24 (THE DOCK IS HERE):** AG02 + RTX 3090 arrived — Phase 1 begins. TD.1
  briefing delivered (connection specifics + preflight now in TASKS.md TD.1: USB4-not-OCuLink,
  both power leads, dock-powered-first, any of the Mac's three TB4 ports, system_profiler
  preflight before ANY install, the no-BAR stop rule). Next concrete step: Artur connects,
  we run preflight, then setup_tinygpu_osx.sh → DEXT approval → reboot → DEV=NV test_tiny
  (NAK lane if Docker is a hassle). Artur held the upstream sync (weekly cadence stands;
  33 commits pending incl. the fp16-KV one-liner conflict).

- **2026-08-24 (FIRST LIGHT — TD.1 done):** full detail in the TASKS.md status row; the
  transferable lessons: (1) the AG02 can silently link as **USB3 fallback** (ASM246X shows as a
  plain USB device, no PCIe tunnel) — replug after dock power-up + approve macOS's accessory
  prompt; check `ioreg` for `IOThunderboltSwitchUSB4` before debugging anything else. (2) DEXT
  approval on Sequoia lives in General → Login Items & Extensions → Driver Extensions, and
  activation needed **no reboot**. (3) No `nvcc` on Mac: the **NAK lane is the frictionless one**
  (`pip install tinymesa==25.2.7.2`, `DEV=NV:NAK`); Docker/nvcc lane still untested. (4) The
  small-BAR question is settled: BAR1=256 MiB, cmdq streams from host RAM (SYS aspace,
  uncached+snooped), P2P refused — T2.x knob tuning and TD.2's tunnel-latency questions start
  from that reality. Worktree `tinygrad-dock` @ fork master `b37d80fc9` is the bring-up tree.
  Also this session: Artur's subagent policy (Sonnet 5 max-effort for Sonnet-proof subtasks)
  promoted into the repo CLAUDE.md conventions.

- **2026-08-25 (DOCK NIGHT 1 — TD.1→TD.3 flagship in ~30 h):** full detail in TASKS.md status
  rows (TD.2a/b/c, T4.14, TD.3, TD.3-moe, T4.17, T2.1+T2.2, T4.18). The arc: first light →
  complete truth table (**best config `DEV=NV`+BEAM everywhere: 1b 149, qwen3:8b 46.9, gpt-oss
  60.9 tok/s — beats llama.cpp-Metal and the llama.cpp-CUDA 1b band**) → transport exonerated →
  dense pooling ~free (80-90 µs/hop) → **graphed MoE pooling: olmoe experts-on-3090 byte-exact,
  41.1 tok/s split = 3.2x all-NV**. Three small PR-shaped runtime fixes found on real hardware:
  T4.14 (compile-server short-read), T4.17 (RPC status-before-fd + recvmsg loop), T4.18
  (hw_page slab under the TinyGPU ~128-slot ceiling; wire protocol has NO free verb — upstream
  report candidate, needs Artur's go). Ops facts that carry: NAK lane = tinymesa, no docker;
  nvcc lane = colima (8 CPU/6 GiB, `PARALLEL=6` for BEAM) — colima STOPPED at session end;
  lane selection is process-wide `DEV='METAL;NV:NAK'`, never in device_map strings; bench-window
  policy exercised (llama-server stopped 23:17→restored ~02:20, well inside the 22:00-19:00 grant).

- **2026-08-26 (QUEUE COMPLETE; TWO HOST PANICS; DOCK HARD-STOPPED):** Artur's ordered queue executed to completion (T4.24→T4.19→T4.23→T4.21→T4.20→T4.22→T4.15→T4.16, then T4.25/27/28/30/31/32/34/29/35/26/33 — every row in TASKS.md). Two identical `dart-apciec2 AppleT8110DART.cpp:2183` kernel panics (01:15, 12:06) during dock work; mechanism chain: GSP host-resident queues × server teardown × bus-master (T4.36), T4.37 shipped+verified but defeated by `nvdev.py:127` re-enable + multi-server leak (T4.38→T4.40). Headline results survive: qwen3:8b 47.9, gpt-oss 62.8, qwen3.6 56.58 (warm-cache-only), split 32.6, IQ verdict = IQ4_XS-only upstream. Upstream sync #4 merged locally (77 commits) but UNVALIDATED (agent lost) + unpushed (workflow scope). **Everything an agent needs: `HANDOFF_2026-08-26.md`.** Artur decides: T4.40 fix vs upstream report vs dock pause; PR train (11 fixes); TD.4 numbers.
- **2026-08-26 PM (handoff review + corrections; RCA dispatched):** every handoff §3 SHA verified against GitHub via
  `gh api` — the local `origin/*` refs are stale by construction (HTTPS-URL pushes never update them; a fresh
  `git status` shows `memory [ahead 56]` and no `origin/task/*`, which is NOT a backup gap). **Evidence found:** the
  unified log survives reboots and the DEXT logs every client open/cfg write/reset/DMA map (extract committed as
  `ulog_tinygpu_2026-08-26.txt`); the session transcripts (`~/.claude/projects/…/d9cf3c0c-*/`, incl. `subagents/`)
  timestamp every command — the T4.33 agent's transcript survives there (5 conflicts; collection errors on its
  first gate attempt). Re-dated panic 2: fault 11:50:00 (T4.37 cleared MASTER), a real FLR + fresh healthy GSP boot
  at 12:00:30 by the T4.35 health check, then nothing for 6 min, then the T4.33 *post-merge* gate run (unvalidated
  merge tree) opened NV ~12:06:2x → panic 12:06:34; **no pkill anywhere after 08:18**. Panic 1 likewise came 11 s
  after a fresh-client attach. ⇒ both panics sit ~10 s after a fresh client's clear→FLR→GSP-boot on a GPU whose last
  faulted GSP-RM was never unloaded; the handoff's original mechanism #2 is relabeled a hypothesis. Also fixed:
  CLAUDE.md de-staled (master `cb05a6c64`, dock hard-stopped, never branch off `origin/master`, `DEV=CPU` gates);
  colima was down; §5.2 count; `.pyc` untracked on the T4.37 branch (`73561c88a`). Plan agreed: a GPU-free week —
  T4.40 step 0 (RCA, Sonnet agent dispatched) → T4.40(a) → T4.39 → T4.33 validation → PR #3 (phase1b→master) —
  while Artur decides §5.1; dock only after T4.40 + verification with him present.
- **2026-08-26 evening (queue executed):** T4.39 done+verified (`3379a0831`). RCA (Fable fork) → `T4.40_RCA.md`
  (`161014c97`): root cause = orphaned-session teardown; panic 1 = A1 (T4.37 covers it), panic 2 = A2 uncovered
  failed-init variant; no pkill in either; fixes 40-1..5. T4.40(a) done+verified (`940f65d79`, = 40-4, server
  spawn discipline). **PR #3 merged → fork master = `5dea150e5`** (+ T4.37 + T4.31); phase1b retired as a base.
  T4.33 validation done: trustworthy to push (Artur, workflow scope), zero regressions/slips, and it CONFIRMED
  the A2 precondition (gate script lacked `DEV=CPU` → collection-time NV probe on both trees). T4.40(b) dispatched
  (Sonnet, off `5dea150e5`): fixes 40-1 (clear MASTER on `NVDevice.__init__` failure) + 40-2 (clear on `fini()`
  raise, not just `is_err_state`), with the #16536 healthy-path negative control. Remaining for the dock: 40-3
  (halt-before-mastering, needs a hardware pass with Artur) + the §5.1 decision + the tinygpu_releases report (RCA §8 draft).

- **2026-08-26 night (route decided; GPU-free backlog cleared):** Artur DECIDED: **no upstream PRs (#14615), no
  tinygpu_releases report (#16086 — third-party AG02)**; all fixes fork-local; A3 permanently uncovered ⇒
  never-kill is standing law; `TINYGPU_REPORT_DRAFT.md` retained unfiled. **PR #4 merged** (upstream sync #4 →
  master `a770d485a`; fork line-cap 27000 = upstream+500, re-derive each sync). **PR #5 opened** (T4.40a+b,
  51/51 green, awaiting merge). Batch results: T4.42 exonerated (no flake reaches the compile cache; 6 pinning
  tests) → filed T4.45 (write-short deadlock, retry proc leak, wait_cond UnboundLocalError); T4.43 CONFIRMED+
  fixed (qcom per-unpickle server spawn — bites every non-Linux host; shared `compiler_server_cache.py`);
  T4.40c code-complete (`5d3d8fa70`, halt-verify before mastering, HARDWARE-GATED, DO NOT MERGE). Remaining:
  PR #5 merge click; the supervised hardware session (T4.40c checklist + RCA §7) → T4.35 runs 2-3 → T4.34
  capture → T4.29 nvcc row → M3 flagship (Q6_K_XL/Q8_0 pooled); T4.45 optional; TD.4 demo still open.

- **2026-08-27 night (dock reopened; M3 ACHIEVED):** hardware session H1-H4 verified the remediation on silicon
  (FLR alone halts the core; M-C exonerated) → PR #6; measurements: T4.35 closed (warm-cache-only, structural),
  T4.34 fully closed (storm = ONE OOB-class candidate fault + ~121 deterministic teardown echoes — the '109/123'
  counts were teardown ops, never fault events; candidate named by T4.50), T4.29 nvcc row closed (storm 2/2);
  RCA chain T4.47 (fault pre-exists the OOM; 200 ms drain gate) → T4.46/T4.48 (cause names + F1/F2/F3, silicon-
  validated 9m48s vs 64 min) → T4.49 (no timing interference; `BEAM_DEV_TIMEOUT` inert on NV — `can_recover`).
  **FLAGSHIP: Q8_0 36.9 GB pooled = 8.0 tok/s no-BEAM, 31.1 tok/s JITBEAM=2 (M3 target ≥15 — beaten 2.07x),**
  full residency, zero faults either leg. Remediation survived 3 live storms + 6 deliberate fault events tonight.
  Open: T4.45/T4.52 (GPU-free fixes), T4.53 (name the culprit kernel), T4.51 (optional), PR #7 (diagnosability →
  fork master, Artur's click), TD.4 demo decision. 3 agents parked on background runs (HANDOFF §7 lesson 8).

- **2026-08-27 ~03:15 (close-out):** PR #8 opened (T4.42/43/45/52/53 assembled; reap ported into the shared
  cache module; gates green). T4.52 verified (teardown echo wall → ~3 lines); T4.53 verified (culprit = the
  MoE expert-gather `weight[sel]`, 256 experts — DeltaNet REFUTED; NV-only deny-guard; T4.54 parked for the
  compile-only renderer RCA). llama-server RESTORED 02:44; colima stopped. Docs overhauled: **new entry point
  `HANDOFF_2026-08-27.md`** (TD.5-oriented), CLAUDE.md + TASKS RESUME repointed, review sweep clean (all doc
  SHAs current-or-historical), worktree prune list prepared (~26 worktrees + ~40 agent branches, awaiting OK).

## 7. Sources

- lucebox eGPU benchmarks: https://www.lucebox.com/blog/egpu-myth
- TinyGPU docs: https://docs.tinygrad.org/tinygpu/  ·  install scripts under `extra/`
- exo pooling request: https://github.com/exo-explore/exo/issues/1904
- exo-cuda fork: https://github.com/Scottcjn/exo-cuda
- llama.cpp Metal+CUDA RPC benchmarks: https://github.com/kjaiswal/llama-cpp-distributed-benchmarks
- Qwen 3.8 release coverage: https://the-decoder.com/alibabas-qwen-team-releases-qwen-3-8-models-with-open-weights-under-the-apache-2-0-license/
- Design doc artifact: https://claude.ai/code/artifact/f430cb11-aade-418c-9a8f-b63dae7b7988
- PR-train dossier artifact (diffs, STE explanations, commit messages, glossary, 2026-08-19):
  https://claude.ai/code/artifact/fb3c41a4-acb3-4e9d-98db-d3e622cb3ee5
- Watcharasorn AG02 bring-up runbook: https://github.com/Watcharasorn/mac-tinygpu-5070ti
- Upstream cross-refs: #17316 (IQ MoE byte blowup = T4.13 mechanism), #17446 (competing gpt-oss
  PR), #13707/#12514 (T4.9 framing), #17617 (disk-lazy experts demand), #14615 (AI-PR closure),
  #17630 (chunked-prefill KV bug pinned as expectedFailure — our T1.5 fixes it, bisect-proven)
