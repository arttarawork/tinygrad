# T6.2 — fused Gated-DeltaNet prefill scan on NV (RTX 3090, sm_86, TinyGPU tunnel)

Branch `task/T6.2-nv-fused-scan` (off master `db439decc`). Status 2026-09-06: kernel + gate + tests + bench written and
CPU-verified; rendered and nvrtc-compiled for sm_86 through the docker lane; **execution/numerics/perf need the parent's
`DEV=NV` round** (§6). Files: `tinygrad/llm/kernels/nv.py`, the dispatch in `tinygrad/llm/model.py` (`_attention`, one
line), `test/unit/test_llm_nv_scan.py`, `extra/gdn_scan_bench.py --impl`.

## 1. What the AMD kernel computes, per (head, chunk)

`amd.gated_delta_prefill(q, k, v, beta, alpha, state, start_pos)` runs the whole delta-rule recurrence for one call of
`GatedDeltaNetBlock._attention` in ONE kernel, per (batch, head): for every token t (sequential, in-kernel loop)

    s_k    = S · k_t                 (per value-row: dot over key_dim)
    delta  = (v_t − s_k ⊙ alpha_t) ⊙ beta_t
    out_t  = (S · q_t) ⊙ alpha_t + delta · (q_t · k_t)
    S     ← S ⊙ alpha_t + delta ⊗ k_t

with S the (value_dim × key_dim) recurrent state, reset to 0 when `start_pos == 0`, written back in place at the end.
The output `core` is (B, H, T, value_dim); `kq = (q·k).sum(-1)` is computed by a normal tinygrad kernel beforehand.

**Lane / register mapping.** Grid = one workgroup per (bh, 4-row tile) — `batch*heads*value_dim/4` groups (1024 for the
35B, 1536 for the 27B); each group is ONE 32-lane wave. Lane `l` owns key columns `l, l+32, l+64, l+96` (key_dim/32 = 4)
of the tile's 4 rows: 16 fp32 state values in a register array (`UOp.placeholder(..., addrspace=REG)`). Per token each
lane loads its 4 k and 4 q values, forms partial dots, and the wave-wide sum comes from a 5-step xor butterfly
(`warp_reduce(..., full_wave=True)`, offsets 16,8,4,2,1). Lane 0 stores `out_t` for its 4 rows (`.valid(lane.eq(0))`);
every lane updates its 16 state registers. No LDS: the reduce is register/shuffle only. The token loop is a REDUCE-axis
range so the state array carries across iterations; `start_pos` enters as a kernel PARAM (`kernel_var`) so the JIT sees
one program for all positions.

## 2. AMD intrinsics → CUDA / sm_86

| RDNA3 (amd.py) | Used by the scan? | NV (nv.py) |
|---|---|---|
| `__builtin_amdgcn_ds_swizzle(x, 0x1f \| off<<10)` — xor-lane swizzle for the reduce | yes (the only device-specific op) | `__shfl_xor_sync(0xffffffff, x, off)`, off ∈ {16,8,4,2,1}; full mask is legal because the LOCAL range is the whole warp and nothing diverges before the reduce (the `lane==0` predicate guards only the store after it) |
| wave32 (RDNA3 in wave32 mode) | yes, implicitly (32-lane LOCAL range) | CUDA warp = 32 lanes: lane math identical, `__launch_bounds__(32)` |
| `__builtin_bit_cast` int↔float around the swizzle | yes | not needed: `__shfl_xor_sync` has a float overload |
| `__builtin_amdgcn_sdot4` / `v_dot4` (int8 dp4a), `__builtin_amdgcn_perm` (byte_perm), WMMA `v_wmma_f32_16x16x16` | no — only `Linear` (quant matmul) and `flash_attention` use them | not mapped; out of T6.2 scope (dp4a → `__dp4a`, byte_perm → `__byte_perm`, WMMA → `mma.sync` if ever wanted) |
| LDS (`AddrSpace.LOCAL`) | no | n/a (`__shared__`) |

Everything else in the kernel is generic tinygrad IR (REG placeholder → `float buf0[16]`, LOCAL range → `threadIdx.x`,
GLOBAL range → `blockIdx.x/y`, predicated store → `if (lidx0 == 0)`, REDUCE range → a `for` loop). `nv.py` is a verbatim
copy of `_gated_delta_prefill_kernel` / `gated_delta_prefill` with the reduce swapped and the kernel named
`gated_delta_prefill_nv`; the AMD file is untouched (the directive forbade touching it, and a shared parametrised body
would have meant editing it).

## 3. How the custom-kernel API picks the device and emits CUDA

`Tensor.custom_kernel(*srcs, fxn=...)` → `UOp.custom_kernel`: the builder gets one placeholder per source tensor
(`UOp.placeholder_like`), returns a SINK with `KernelInfo(name, opts_to_apply=())` (no optimizer passes), and the scheduler
treats it as an ordinary kernel whose buffers are the realized, contiguous sources. The device is simply the sources'
device: `realize.py` compiles with `Device[c.device].renderer` — for NV that is `CUDARenderer(target=sm_86)` (ops_nv.py
picks CUDA by default; `PTX=1` picks `PTXRenderer`). `Ops.CUSTOM` is rendered by `cstyle.py` as `arg[0].format(*srcs)`,
i.e. the C string is pasted verbatim into the kernel — this is why the gate requires a `CUDARenderer`: `ptx.py` has no
CUSTOM pattern, so under `PTX=1` the fused path is off and the WY scan runs as today. Compilation is
`NVRTCCompiler(arch="sm_86", ptx=False)` → cubin; on macOS the compile goes through the docker compile server
(`compiler_cuda.py: osx_docker_cmd`, image `ghcr.io/tinygrad/cuda-arm64:v2.3`) — the same lane the pooled server uses.

**Gate** (`nv_custom_kernels_supported`): device prefix `NV`, `GDN_NV_FUSED` (ContextVar, default 1) non-zero, renderer is a
`CUDARenderer`, arch `sm_70+` (full-mask `__shfl_*_sync` is Volta+; mock-NV reports `sm_35`). The env check runs before any
`Device[...]` access so CPU/NULL/off never open NV. Dispatch in `_attention`: AMD gate first, then NV, else loop/WY —
`capture=True` and `head_k_dim % 32 / head_v_dim % 4` conditions unchanged. The GDN chunk cap (`gdn_chunk_for`, model.py
~L1655/1784) is deliberately left keyed on the AMD gate only: NV keeps `GDN_CHUNK` (32 on the pooled server), so one launch
covers 32 tokens per layer exactly like the WY path; lifting the cap on NV (fewer, longer launches) is the follow-up lever
once parity is proven.

## 4. Register / LDS budget: RDNA3 vs sm_86

| | RDNA3 (gfx11, wave32) | RTX 3090 (sm_86) |
|---|---|---|
| registers available | 256 VGPRs/lane (wave32) | 255 regs/thread max; 64K regs/SM |
| this kernel's live set | 16 state + 4 k + 4 q + 16 update temps + scalars ≈ 45-64 fp32 | same (renderer emits identical scalar code); nvrtc chooses the allocation, `__launch_bounds__(32)` puts no cap below 255 |
| LDS / shared | 0 | 0 |
| block/group size | 32 lanes | 32 threads → occupancy is bounded by the 16 resident blocks/SM limit (= 16 warps = 25% of 64), not by registers |
| grid | 1024 (35B) / 1536 (27B) groups | 82 SMs × 16 = 1312 resident blocks: the 35B fits in one residency wave, the 27B needs ~1.2 |

The per-token work per block is tiny (8 loads/lane, 2 dots, 10 shuffles, 20 FMAs), so a launch is latency-bound: a
32-token chunk is ~32 × (global-load latency + 5 dependent shuffles) ≈ 30-60 µs per layer per chunk (estimate, unmeasured).
Today's WY scan on the 27B costs ≈ 1/27 − 1/260 ≈ 33 ms/token ⇒ ~1 s per 32-token chunk across 48 layers, i.e. ~22 ms per
layer-chunk; the fused launch has 2-3 orders of magnitude of headroom before it stops being a win. If the hardware round
shows the launch count dominating, the levers are `row_tile` 8 (halves the grid, doubles registers) and lifting the chunk
cap on NV (§3) — not part of T6.2.

## 5. Validation plan and what was actually verified

Verified on this Mac (CPU/NULL only, no NV access):
- `nv_custom_kernels_supported` is False for CPU, CPU:1, NULL, None, device tuples, and for "NV" under `GDN_NV_FUSED=0`
  without opening any device (`TestNVGate`).
- The kernel renders for `CUDARenderer(DEV.target("NV", arch="sm_86"))` at the 35B's real geometry (32 heads, dk=dv=128,
  chunk 32) with 40 `__shfl_xor_sync(0xffffffff, …)` sites and no `amdgcn`, and nvrtc-compiles to a **cubin** through the
  docker lane (`TestNVKernelSource`): fresh compile 0.45 s, 7656 B, ELF `e_flags & 0xff == 86` (= sm_86). The lane is
  live, not a cache hit: the same source with `__shfl_xor_synk` is rejected, and `arch=sm_35` (mock-NV) is rejected with
  `invalid value for --gpu-architecture` — which is also why the kernel cannot execute under `DEV=MOCK+NV`.
- model.py is unchanged in behaviour off NV: `test_gdn_scan_parity`, `test_attention`, `test_llm_server`,
  `test_llm_vision_model`, `SPEC=2 DEV=NULL test/null`; mypy / ruff / the CI whitespace lint all green, with ONE pre-existing
  failure that reproduces identically at the branch base (`git stash` check):
  `test_gdn_scan_parity.py::TestGDNScanCapture::test_capture_does_not_change_output_or_final_state` — it asserts
  bit-identity between capture=True (always the loop) and capture=False (WY since the T4.69c default flip); max rel
  diff 2.6e-4 on the 35B geometry. Not touched by T6.2.

Not verifiable without hardware (mock-NV = sm_35, PTX renderer has no CUSTOM): execution, numerics, perf. The hardware
round (`@unittest.skipUnless(nv_custom_kernels_supported(Device.DEFAULT))`, so it also skips under the mock):

    # parity: kernel vs numpy at dk=128/dv=128, 48 and 32 v-heads, chunk 32 (rtol/atol 1e-4 as the AMD test);
    #         start_pos==0 reset vs resident state; whole block fused vs WY (rtol 1e-3 / atol 1e-4)
    DEV=NV PYTHONPATH=. .venv/bin/python -m pytest test/unit/test_llm_nv_scan.py -v -rs
    # bench: same block, WY vs fused (chunk 32, 256 tokens); --impl loop for the third point
    DEV=NV PYTHONPATH=. .venv/bin/python extra/gdn_scan_bench.py --geometry 27b --device NV --impl wy
    DEV=NV PYTHONPATH=. .venv/bin/python extra/gdn_scan_bench.py --geometry 27b --device NV --impl nv_fused
    DEV=NV PYTHONPATH=. .venv/bin/python extra/gdn_scan_bench.py --geometry 35b --device NV --impl wy
    DEV=NV PYTHONPATH=. .venv/bin/python extra/gdn_scan_bench.py --geometry 35b --device NV --impl nv_fused
    # end-to-end: the usual T0.3 prefill tok/s on the 27B with GDN_NV_FUSED=1 (default) vs GDN_NV_FUSED=0

Ship rule: the fused path is on by default on NV once the parity test passes on hardware; until then serve with
`GDN_NV_FUSED=0` if this branch is ever put behind :8081. Pooled server = one `DEV=NV` process at a time: the hardware
round must run with :8081 stopped via `pooled-serve.sh stop`.

## 6. Risks / open points for the hardware round

- **First launch = fresh nvrtc compile on the docker lane**, not a BEAM search (`opts_to_apply=()`): seconds, not hours.
- `__shfl_xor_sync` needs all 32 lanes converged — true by construction here; the 5-deep dependent shuffle chain is the
  kernel's latency floor.
- Numerics: same summation order as the AMD kernel (fp32, per-lane partials then butterfly); expect ≤1e-5 vs numpy.
- The state write-back is in place on the block's `recurrent_state` (`state` = the AFTER'd conv-store tensor), exactly as
  on AMD; the block-level test checks it against the WY path's snapshot.
- If the render ever shows `gidx` splitting the group axis into a 3-D grid for a larger geometry, that is the renderer's
  generic grid split (limit 65535 per dim) — harmless.
