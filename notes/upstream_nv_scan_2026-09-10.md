# T4.98b — upstream tinygrad scan for NV/CUDA kernel-performance changes

Scanned 2026-09-10. `git fetch https://github.com/tinygrad/tinygrad.git master:refs/remotes/upstream-scan/master`
→ `b496cd191` (2026-09-10, "str->compiled in mappings"). Worktree: `/Users/artur/Documents/tinygrad-t498u`,
branch `task/T4.98-upstream-scan` on `integration/t6` @ `4c6b6a954`. Read-only: no rebase/merge/cherry-pick/benchmark/device
was performed; this file only records what was found and recommends next steps.

## Correction to the task's base assumption

The task frames our base as `af2a43c85` (2026-08-18), giving 408 upstream commits since then. But
`git merge-base HEAD refs/remotes/upstream-scan/master` = `4bdc86513` ("delete unused rewrite rules", **2026-08-26**), not
`af2a43c85` — this fork already carries a later sync than the task assumed (visible in `tinygrad/llm/kernels/amd.py`, which
is byte-identical to upstream at `a0a901c8e^`, one commit before "faster qwen 3.8"). The **true missing set is 280 commits**
(`git log HEAD..refs/remotes/upstream-scan/master`), not 408. All conflict-risk claims below use the true set; a few
commits the task's range would flag (e.g. the early "hcq2: speed"/"hcq2: faster beam" commits from 08-19/08-20) are
already ancestors of our `HEAD` and are not listed.

## The single biggest finding: this fork already has the answer's shape, just not for gemv

`tinygrad/llm/kernels/nv.py` (**already in this fork**, not upstream) hand-writes the Gated-DeltaNet prefill scan for NV
using `Ops.CUSTOM` inline CUDA-C (`__shfl_xor_sync` for the 32-lane reduce), gated by `nv_custom_kernels_supported()` →
requires `CUDARenderer` specifically (**`Ops.CUSTOM` is never rendered by `ptx.py`** — only the C-style renderer emits the
inline-string ops; `_nv_device_ok` in `nv.py:16` says this explicitly) and `sm_70+`. This is wired into `llm/model.py` as
a plain conditional import next to the AMD path (`model.py:841-842`: `gated_delta_prefill if amd_custom_kernels_supported
else nv_gated_delta_prefill if nv_custom_kernels_supported else None`).

Upstream's `tinygrad/llm/kernels/amd.py` (677 lines, grew from ~314 in a run of commits since 2026-08-21) is the *production*
version of exactly the kernel class this task is scoping: a hand-written **quantized Linear** (`Q4_K`/`Q5_K`/`Q6_K`/`IQ4_XS`,
**not** `Q8_0`/`Q4_0`) using one warp per output row, activation quantized to int8 once (`q8_quantize`), a dp4a-style dot
product per weight word, warp-shuffle reduction, and nontemporal (cache-bypassing) streamed weight loads — plus a hand-written
flash-attention decode kernel and (separately) the same `gated_delta_prefill` our `nv.py` already mirrors. It is 100% AMD-gated
(`amd_custom_kernels_supported`: `device.split(":")[0] != "AMD"` → `False`; `tinygrad/llm/kernels/` upstream has **only**
`amd.py`, no `nv.py`/`cuda.py`) — grep for `NV`/`CUDA` in upstream's `llm/model.py` returns nothing. **There is no NV kernel
code upstream to cherry-pick for the gemv bottleneck.** But every AMDGCN builtin it uses has a direct CUDA/PTX equivalent
already usable from `sm_86`:

| amd.py builtin | purpose | CUDA/PTX equivalent (sm_86) |
|---|---|---|
| `__builtin_amdgcn_sudot4` (`_amd_dp4a`) | 4×int8 dot-accumulate | `__dp4a(a,b,c)` (native intrinsic, CC 6.1+) |
| `__builtin_amdgcn_ds_swizzle` (`warp_reduce`) | warp-lane reduce | `__shfl_xor_sync` — **already used verbatim in our own `nv.py`** |
| `__builtin_amdgcn_perm` (`_amd_byte_perm`, IQ4 LUT decode) | byte permute | `__byte_perm(x,y,sel)` (native intrinsic → PTX `prmt.b32`) |
| `__builtin_nontemporal_load` | cache-bypass weight stream | `__ldcs()` or inline PTX `ld.global.cs` |

Net: the fork's own `nv.py` already proves the mechanism end-to-end (UOp `Ops.CUSTOM`/`Ops.CUSTOMI`, `Tensor.custom_kernel`,
JIT/BEAM/buffer integration all "for free" vs. a standalone `.cu`), and `amd.py` is a line-by-line portable template for the
missing piece (quantized `Linear`). `Q8_0` — simpler than the K-quant family amd.py targets (flat int8 + one fp16 scale per
32-block, no nibble unpacking, no scale/min sub-decode) — is the natural pilot, then extend to `Q4_0`/`Q4_K` following
`amd.py`'s `_quant_decode_kernel`/`_decode_linear` structure. Precedent for the follow-up doc + test: `NV_FUSED_SCAN_DESIGN.md`
and `test/unit/test_llm_nv_scan.py` (both already in this fork, for the scan kernel).

## Ranked list (most valuable first)

**1. (Internal fact, not a commit)** — see above. Settles the hand-write-vs-cherry-pick question directly.

**2. `a0a901c8e` "faster qwen 3.8 (#17720)" — 2026-08-26.** Named for our exact model. Touches `tinygrad/llm/kernels/amd.py`
(363 lines, AMD dequant/streaming tuning — reference only, doesn't run on NV), `tinygrad/renderer/cstyle.py` (11 lines,
**generic**: splits `render_access` into a reusable `render_ptr` + adds a `nontemporal`-flagged `Ops.LOAD` → HIP's
`__builtin_nontemporal_load` string-rewrite — the plumbing pattern to mirror for CUDA's `__ldcs`), `tinygrad/uop/ops.py`
(3 lines: `UOp.tuplize` now does `repr(self.arg)` so heterogeneous `arg` types sort comparably), `tinygrad/codegen/late/coalesce.py`
(9 lines, image-load valid-elision — unrelated to GPU memory coalescing despite the name), plus 2 test files.
**Conflict check (verified by diffing our tree against `a0a901c8e^`, i.e. one commit prior):** `cstyle.py`, `coalesce.py`,
`test/unit/test_llm_amd.py` are **byte-identical** to our tree → clean apply. `tinygrad/uop/ops.py` has since diverged
locally (26 lines) but the incoming hunk is a tiny, semantically-isolated 3-line fix to `tuplize` — low conflict risk.
`test/unit/test_attention.py` has been heavily rewritten locally (450+ lines) against a 2-line upstream hunk — expect a
context-mismatch conflict on that file specifically; trivial to resolve by hand or skip (test-only).

**3. `c84876fdd` "move nv to hcq2 (#17970)" — 2026-09-05 — + ~25 satellite `hcq2:` commits spanning 2026-08-19 → 09-10
(the run continues literally through today).** A ground-up rewrite of NV's command-queue/dispatch/buffer layer — 648 of
1005 changed lines are in `tinygrad/runtime/ops_nv.py` alone. Real, but secondary, upside: reduced per-kernel dispatch
overhead, parallel/cached BEAM compiles, a schedule cache (`b536514c8`) — this plausibly helps aggregate decode tok/s
(many small launches per token) and BEAM turnaround, but **does not touch the per-kernel memory-bandwidth ceiling** that
is this task's actual complaint (an isolated 44MB gemv reaching only ~100-120GB/s of 936GB/s is a single-kernel efficiency
problem, not a dispatch-overhead problem). Upstream hotfixed a full HCQ2 disable twice during development (`4e6bdac41`
2026-08-27, `a57188ea6` 2026-08-20) — this was unstable even in its home repo. **Conflict cost is very high**: this fork
carries 17 local commits on `ops_nv.py` and 25 on `runtime/support/` since the true merge-base, built against the
pre-hcq2 shape (CLAUDE.md: "heavy in ... tinygrad/runtime/ops_nv.py"). Not worth taking on for this task.

**4. `9dbcddc64` "UPCAST and LOCAL are SPLIT (#17889)" (2026-09-01) + `a3bde519d` "merge OptOps.UNROLL and OptOps.UPCAST
(#17867)" (2026-08-31).** Pure API consolidation, not a perf win by itself: upstream's `OptOps` enum is now just
`{TC, SPLIT, PADTO, SWAP}` — `UPCAST`/`LOCAL`/`UNROLL`/`GROUP`/`GROUPTOP` all became `SPLIT` parameterized by `AxisType`
(e.g. old `GROUPTOP` → `Opt(OptOps.SPLIT, axis, (sz, AxisType.GROUP_REDUCE, top=True))`). **This is a landmine, not a
lever**: this fork's NV hardware-fault-avoidance guard in `tinygrad/codegen/opt/search.py` (`fe7ee3e18`, `f372ef5b0`,
2026-09-06 — reproduced device faults on the 3090 4/4 times, root cause not yet fixed, tracked as T4.54) filters on
`a.op in {OptOps.GROUP, OptOps.GROUPTOP}` verbatim. Those opcodes **no longer exist** upstream. A textual rebase/merge
that "resolves" this file without reading it would leave the filter checking a condition that can never match again —
fault protection silently evaporates, not a merge conflict that shouts at you. If this ever gets rebased past, the guard
must be rewritten as `a.op is OptOps.SPLIT and a.arg[1] is AxisType.GROUP_REDUCE` (checking the `top` flag per the two
original commits) and re-verified against the original repro before merging. Not actionable today; recorded so it isn't
forgotten.

**5. `5c6af9cb1` "keep the WARP as its own local launch dim in add_gpudims [pr] (#17897)" — 2026-09-01.** Small (5 lines
in `tinygrad/codegen/gpudims.py`), **zero local commits touch that file** (clean history). Stops `get_grouped_dims` from
folding a `AxisType.WARP`-typed local range into neighboring local dims. Directly relevant the moment a warp-per-row NV
gemv kernel is written with `AxisType.WARP` (same axis type our own `nv.py` scan kernel already uses for its 32-lane
reduce) — without this fix a BEAM/heuristic pass could in principle merge the warp dim with another local dim and break
the one-warp-per-row assumption. Companion test hunk (`test/opt/test_tensor_cores.py`) has drifted ~44 lines from
unrelated later upstream test churn — the production fix applies clean, the test hunk may need hand reapplication.

**6. GPT-OSS MXFP4 tensor-core gemm effort — `93d8f0255` "allow GROUP_REDUCE with tensor cores" (2026-09-01), `f4c7aa7cc`
"gptoss: router gemm" (2026-09-04), `c68c174da` "gptoss: fused rmsnorm_mul_quantize_mxfp8" (2026-09-02), `60f1cd916`
"gptoss: matmul_mx supports prequantized inputs" (2026-09-01), `6242b0906` "cleaner mxfp4 gemm prelude" (2026-08-28).**
AMD/MI350X-targeted (mfma, sgprs) quantized-gemm-via-tensor-cores groundwork; touches `examples/mlperf/models/gpt_oss.py`,
`extra/gptoss_kernels/`, `extra/gemm/gemm_mxfp4.py`, `extra/thunder/amd/`. Template value only, and for the wrong shape:
tensor cores need reused operands across a large M; decode gemv is M=1 (memory-bound, tensor cores don't help), but the
**prefill gemm (32-token chunk) has M=32**, which tiles cleanly into `mma.sync.aligned.m16n8k16` (already renderable —
`cstyle.py`/`ptx.py` had `mma.sync` support before our base). Worth a dedicated look once the gemv kernel work is done
and prefill throughput is next; out of scope here.

**7. `7fdc58b1c` "add support for UD quants (#17806)" — 2026-08-28.** Adds dequant formulas (`Q2_K`, `Q3_K`, `IQ2_XXS`,
`IQ2_XS`, `IQ1_S`, `IQ1_M`, `IQ4_NL`) to `tinygrad/llm/gguf.py` — GGUF format/correctness coverage for Unsloth Dynamic
quants, zero kernel/perf content. Only matters if a UD-quantized GGUF is ever loaded. Skip for this task.

## Ruled out (surveyed, no action)

Remaining `hcq2:` commits (all part of finding #3's rewrite) · viz/profiler tracing (`e8a8d99b9`, `a4ac2605f`, etc.,
unrelated to kernel perf) · usb/comma-pipeline commits (different hardware target) · `qcom: move to hcq2` (different
backend) · mlperf submission script env tweaks (`0d6681584` `BEAM_PADTO=0`, `a5c833702` "don't PADTO WARP" — MI350X
scripts / a WARP-can't-be-padded bugfix, not a lever) · `tinygrad/runtime/autogen/mesa.py` (Qualcomm/Adreno driver
bindings, matched only on an unrelated `dp4acc` substring) · `tinygrad/runtime/support/nv/{ip.py,nvdev.py}` (bare-metal
GSP/firmware register bring-up — an alternate direct-driver stack, not the kernel-dispatch path this fork uses) ·
`tinygrad/runtime/support/{compileserver.py,compiler_cuda.py}` (**zero commits** in the true-missing range — nothing to
reconcile with this fork's local nvrtc-poisoned-compile-cache fix) · `extra/gemm/max_kernels/nv.*.cu` + `extra/gemm/
{simple_matvec,tinygrad_nv_matmul,max_matmul,cuda_matmul}.py` — **these are not new**, they already exist in this fork's
tree essentially untouched since base (hand-tuned reference fp16 NV gemm kernels + a driver script); worth opening
locally as reference material, not an upstream delta to pull.

## Recommendation: hand-write, cherry-pick two small groundwork commits, do not rebase

- **Hand-write** the quantized-`Linear` NV kernel (finding #1), starting with `Q8_0`. Extend `tinygrad/llm/kernels/nv.py`
  the same way it already extends `amd.py` for the scan kernel: reuse `amd.py`'s pure-UOp format-parsing helpers where
  device-agnostic, replace only the four intrinsics in the table above, keep the `CUDARenderer`-only / `sm_70+` gate.
  This is de-risked by the fact that both halves of the port already exist and both are already proven in this exact
  codebase (our own `nv.py` mechanism; upstream's `amd.py` target design) — it is a port, not a research project.

- **Cherry-pick** only the two isolated, low/zero-conflict commits, as groundwork (neither is the fix by itself):
  ```
  git cherry-pick a0a901c8e4af76975ea288348a0d5a0c4870459b
  # "faster qwen 3.8": cstyle.py/coalesce.py/test_llm_amd.py apply clean (verified byte-identical pre-image);
  # expect to hand-resolve tinygrad/uop/ops.py (tiny 3-line tuplize fix) and test/unit/test_attention.py
  # (2-line upstream hunk vs. a locally-rewritten file — trivial to reapply by hand or drop, test-only).
  # Brings tinygrad/llm/kernels/amd.py current as reference material and lands the generic render_ptr /
  # nontemporal-LOAD rendering pattern in cstyle.py to mirror for CUDA's __ldcs.

  git cherry-pick 5c6af9cb18662f8e0a6398127ebf165d2364a504
  # "keep the WARP as its own local launch dim in add_gpudims": tinygrad/codegen/gpudims.py applies clean
  # (zero local commits on that file); test/opt/test_tensor_cores.py may need a manual reapply (unrelated drift).
  ```

- **Do not rebase.** The true frontier is 280 commits, dominated by the hcq2 rewrite (finding #3, 42+ commits, 648 lines
  in `ops_nv.py` alone, self-hotfixed twice upstream) and the OptOps consolidation (finding #4, a proven landmine against
  this fork's NV fault-avoidance guard). Both are large, address dispatch overhead rather than the stated bandwidth gap,
  and would cost far more review/re-verification time than the two cherry-picks above return. Revisit hcq2 specifically
  as its own task if, after the hand-written gemv kernel closes the per-kernel bandwidth gap, decode tok/s is still short
  of target and launch/dispatch overhead becomes the visible bottleneck.
