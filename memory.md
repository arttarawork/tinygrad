# memory.md — project memory for the Ampere-over-Thunderbolt effort

Companion to `NV_LLM_DESIGN.md` (the design doc, same directory). This file holds the context,
decision history, external research, and repo findings that the design doc deliberately leaves out.
Written 2026-08-18 against `tinygrad @ af2a43c85` (v0.13.0-968). Fork: `arttarawork/tinygrad`.
Published copy of the design doc: https://claude.ai/code/artifact/f430cb11-aade-418c-9a8f-b63dae7b7988

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
  Zero occurrences of paged/continuous-batching/speculative anywhere in the repo.
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

## 7. Sources

- lucebox eGPU benchmarks: https://www.lucebox.com/blog/egpu-myth
- TinyGPU docs: https://docs.tinygrad.org/tinygpu/  ·  install scripts under `extra/`
- exo pooling request: https://github.com/exo-explore/exo/issues/1904
- exo-cuda fork: https://github.com/Scottcjn/exo-cuda
- llama.cpp Metal+CUDA RPC benchmarks: https://github.com/kjaiswal/llama-cpp-distributed-benchmarks
- Qwen 3.8 release coverage: https://the-decoder.com/alibabas-qwen-team-releases-qwen-3-8-models-with-open-weights-under-the-apache-2-0-license/
- Design doc artifact: https://claude.ai/code/artifact/f430cb11-aade-418c-9a8f-b63dae7b7988
