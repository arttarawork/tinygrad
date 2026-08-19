# T4.8 — simdgroup warp-reduce primitives in the Metal renderer: scoping notes

Branch: `task/T4.8-metal-simd` off `integration/wave1`. **No production code changed** — scoping
concluded the ≤32-lane "just call `simd_sum`" case is not the small, renderer-local, no-scheduler-
change fix it looked like from T1.8b's finding. Real blocker: MATVEC's *actual default kernel shape*
combines the reduce-group axis with a second, co-resident LOCAL axis (row-parallelism) in the same
threadgroup, and Metal's built-in `simd_sum`/`simd_max` cannot safely reduce across only *part* of a
simdgroup — using them naively there would silently sum across different output rows. See §3.2.

**Bottom line up front:** the barrier+LOCAL round-trip is real and exactly where T1.8b measured it,
but replacing it needs a new UOp inserted at the same pipeline stage as `Ops.WMMA` construction, plus
a genuinely non-trivial correctness gate that the obvious formulation misses. Sized at ~150-250 new/
changed lines across 5 files for the single-simdgroup case alone (§4) — not small, not forced. The
`>32`-lane two-level scheme that would actually help T1.8b's attention kernel (`Hd` = 64/128) is
materially larger again and not estimated in detail here.

## 1. Render-path map: how a GROUP reduction becomes barrier+LOCAL on Metal

The lowering is **entirely backend-agnostic** — nothing Metal-specific happens until the very last
string-render step. Pipeline, in `full_rewrite_to_sink` (`tinygrad/codegen/__init__.py:288-397`):

| Stage | File:line | What happens |
|---|---|---|
| Opt selection | `codegen/opt/heuristic.py:89,91,92` (MATVEC) / `heuristic.py:96-101` (generic GROUPTOP) | `k.apply_opt(Opt(OptOps.GROUP, axis, N))` tags a reduce range `AxisType.GROUP_REDUCE` |
| Opt application | `codegen/opt/postrange.py:133-177` (`Kernel.apply_opt`) | `OptOps.GROUP`/`GROUPTOP` → `AxisType.GROUP_REDUCE` on the chosen range; explicitly **incompatible with `OptOps.TC`** (`postrange.py:174`, "no grouping with tensor cores") — the two hardware-cooperation strategies are already mutually exclusive today |
| **LOCAL-buffer construction** | `codegen/__init__.py:170-184` (`fix_group_for_reduce`, called from `pm_reduce_local` at `:227-234`, run in the `"remove reduces"` graph_rewrite at `:324`) | Splits the reduce: per-thread partial → `bufferize(..., AddrSpace.LOCAL)` (a threadgroup array, one slot per group-thread) → **new** `Ops.REDUCE` over a serial loop reading that buffer back |
| Local buffer materialization | `codegen/__init__.py:243-249` (`add_local_buffer` / `pm_add_local_buffers`, `:327`) | Turns the `Ops.STAGE` into a real addressable `threadgroup` buffer |
| Thread-index assignment | `codegen/gpudims.py:41-88` (`add_gpudims`, `:330`) | The GROUP_REDUCE range (grouped with `WARP`/`LOCAL` at `gpudims.py:51`) becomes an actual Metal `lid.x`-family hardware thread index |
| **Barrier insertion** | `codegen/__init__.py:258-282` (`pm_implicit_barriers`, applied at `:381`, *after* gpudims) | Generic: any `LOAD` that depends via `AFTER` on a `STORE` to a `LOCAL`-addrspace buffer gets a `BARRIER` inserted ahead of it (`add_raw_barrier`); a same-loop store+load pair gets one at loop end too (`add_war_barrier`) |
| String render | `renderer/cstyle.py:353` (`MetalRenderer.barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"`), generic `base_rewrite`/`string_rewrite` | Metal is a **pure consumer** here — it renders whatever `BARRIER`/`STORE`/`LOAD`/loop UOps arrive, one line each. No Metal-specific reduction logic exists anywhere in `cstyle.py` today. |

**This confirms the barrier+LOCAL idiom is built once, generically, for every backend** — CUDA/AMD/
CPU/OpenCL all go through the identical `fix_group_for_reduce`. Metal isn't special-cased into this
path; it just happens to be the backend where the alternative (simdgroup instructions) exists and
matters most (Apple GPUs are 150 GB/s and barrier-latency-sensitive; CUDA's `__syncthreads` +
shared-memory story is comparatively cheaper relative to its bandwidth).

### 1.1 Evidence: real generated MSL, dumped live on this M3 Pro (METAL, Apple9)

Script: built a plain `Tensor.sum(axis=-1)` AST and a `(4096,4096)@(4096,)` fp16 gemv AST via
`test/backend/test_linearizer.py`'s `helper_realized_ast`, forced opts with `test.helpers.replace_opts`,
rendered with `tinygrad.codegen.to_program`. (Scratchpad scripts, not committed — one-shot dumps.)

**Plain `sum`, `GROUP=8`** (`Opt(OptOps.GROUP, 0, 8)` on a `(4,4,128,128)→(4,4,128)` reduce):

```metal
kernel void r_2048_8_16(...) {
  float buf0[1];
  float buf1[1];
  threadgroup __attribute__((aligned(16))) float buf2[8];
  int lidx0 = lid.x; /* 8 */
  *(buf0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) { ... *(buf0+0) += val0; }   // per-thread partial (16 iters)
  *(buf2+lidx0) = (*(buf0+0));                                          // STORE to LOCAL
  threadgroup_barrier(mem_flags::mem_threadgroup);                      // BARRIER
  *(buf1+0) = 0.0f;
  for (int Ridx102 = 0; Ridx102 < 8; Ridx102++) { ... *(buf1+0) += val1; }  // EVERY thread re-sums all 8 slots
  if ((lidx0==0)) { *(data0_2048+gidx0) = (*(buf1+0)); }
}
```

**MATVEC-shaped: forced `Opt(OptOps.GROUP, 0, 8)`** on the fp16 4096² gemv reduce axis (mirrors the
MATVEC heuristic's default `MV_THREADS_PER_ROW=8`, `codegen/opt/heuristic.py:61`):

```metal
kernel void r_4096_8_512(...) {
  float buf0[1];
  threadgroup __attribute__((aligned(16))) float buf2[8];
  int lidx0 = lid.x; /* 8 */
  *(buf0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 512; Ridx0++) { ... *(buf0+0) += (float)(val0*val1); }
  *(buf2+lidx0) = (*(buf0+0));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  *(buf1+0) = 0.0f;
  for (int Ridx102 = 0; Ridx102 < 8; Ridx102++) { *(buf1+0) += (*(buf2+Ridx102)); }
  if ((lidx0==0)) { *(data0_4096+gidx0) = (half)(*(buf1+0)); }
}
```

Both match the pattern T1.8b's hand-written kernel documented by hand (`tinygrad/llm/attn_kernel.py:12-17`,
"all Hd threads then redundantly sum that Hd-element buffer") — the compiler-generated GROUP lowering
and T1.8b's manual UOp construction independently arrived at the same shape. In both dumps: N threads
write one word each to a `threadgroup` array, one barrier, then **every** thread in the group serially
re-reads and re-sums the whole array to get a value only thread 0 ends up using. That redundant N-way
resum by every thread, plus the barrier, is exactly what `simd_sum` would collapse to one instruction.

**Note:** the un-forced default-heuristic run of this exact gemv did **not** trigger MATVEC (`DEBUG=3`
showed no `MATVEC:` line; the heuristic's several preconditions, `heuristic.py:70-73`, didn't line up
for this synthetic construction) — the `Opt` was forced directly to get the MSL sample. This is a
side finding, not a T4.8 blocker; the real-model MATVEC firing is already validated by T1.2/T1.10's
GB/s numbers, not re-litigated here.

## 2. Backend inventory: does anyone already do warp reduction?

**No.** Grepped every renderer file for `shfl`, `warpSize`, `simd_sum`, `simd_shuffle`, `ds_permute`,
`redux`: zero hits anywhere in `tinygrad/renderer/` — not CUDA (`cstyle.py`'s `CUDARenderer`), not
PTX (`renderer/ptx.py`), not AMD (`renderer/cstyle.py`'s AMD class + `renderer/amd/`), not NAK/NIR
(`renderer/nir.py`). Every backend, including CUDA on real sm_86 tensor-core hardware, reduces across
threads exclusively via the same generic barrier+LOCAL/shared-memory path from §1. **This is not a
"Metal is behind" gap — the whole framework has never done intra-warp reduction.** That changes the
cost estimate: there's no existing per-backend "warp op" convention to extend, only the pattern below.

**The one existing precedent for a hardware small-group primitive: `Ops.WMMA` (tensor cores).**
- Chosen at the opt layer: `codegen/opt/heuristic.py:28` (`if USE_TC > 0 ...`), gated by the renderer
  declaring capability via `Renderer.tensor_cores: list[TensorCore]` (`renderer/__init__.py:72`,
  populated per-backend, e.g. `MetalRenderer.__init__`: `tc.metal if target.arch.startswith("Apple")
  and int(target.arch[5:]) >= 7 else []`, `cstyle.py:345`).
- Applied via a **dedicated `OptOps.TC` branch** in `Kernel.apply_opt` (`postrange.py:178+`) —
  **not** the GROUP/GROUPTOP branch, and explicitly mutually exclusive with grouping (`postrange.py:174`).
  WMMA never enters the barrier+LOCAL machinery at all; it builds an `Ops.WMMA(a,b,acc)` node directly.
- Algebraically folded with a following add at `pm_wmma_add` (`codegen/__init__.py:108-116`), which
  lives inside the *same* `pm_reduce_local` PatternMatcher as `fix_group_for_reduce` (`:227`) — i.e.
  WMMA-handling and GROUP-lowering already sit side by side in the one pass that would need a third
  sibling case for simd-reduce.
- Rendered per-backend as **one self-contained function, injected as `prefix` text, keyed on scanning
  the linearized `uops` for `Ops.WMMA` nodes** (`wmma_args(uops)`): Metal synthesizes a
  `simdgroup_float8x8`-based helper (`cstyle.py:374-385`) — Metal *already* uses `simdgroup_*` machinery
  today, just for MMA, not reduction; CUDA emits inline `mma.sync` PTX asm (`cstyle.py:448-461`); AMD
  emits `__builtin_amdgcn_wmma_*` (`cstyle.py:560-566`).

This is the template a simd-reduce feature would mirror: new `Ops` member, opt-layer insertion
alongside (not instead of) GROUP, renderer capability flag, per-backend render case. It is **not** a
renderer-only change for WMMA, and the same is true for simd-reduce.

## 3. Insertion-point analysis

### 3.1 Why it can't be a pure `renderer/cstyle.py` peephole

Two candidate "stay inside cstyle.py" designs, both rejected:

- **Post-linearization peephole in `MetalRenderer.render_kernel`**, scanning the final `uops` list
  (mirroring `wmma_args(uops)`) for the STORE→BARRIER→loop→LOAD/ALU→END idiom and splicing in a
  `simd_sum` call. Rejected: by the time `render_kernel` sees `uops`, `reduce_ranges_to_acc`
  (`codegen/__init__.py:210-220`) has already flattened the second-stage reduce into a bare
  accumulator register + a real `RANGE`/`END` loop — a multi-node region, not a single UOp — and
  register allocation / control-flow lowering (`pm_add_control_flow`, `:384`) has already run. Matching
  a multi-instruction region *after* it's been through generic loop-lowering and (for ISA renderers)
  regalloc is fragile: any unrelated change to unrolling, upcast factors, or loop structure silently
  breaks the pattern match. `cstyle.py`'s `string_rewrite` is built around one-UOp-in, one-line-out;
  it has no precedent for recognizing a spread-out region.
- **A UOp-graph PatternMatcher added only when `isinstance(ren, MetalRenderer)`**, inserted right after
  `add_gpudims` (`:330`) but before `pm_implicit_barriers` (`:381`). Same problem one stage earlier:
  `fix_group_for_reduce` already ran two passes prior (during `"remove reduces"`, `:324`), so the
  small reduce is *already* a loop by this point, not a matchable single node.

The only place where the reduce is still a single structured node — before it's forced into the
generic loop shape — is **inside `fix_group_for_reduce` itself**, i.e. the exact same pipeline stage
(and the same `pm_reduce_local` PatternMatcher, `:227-234`) where `pm_wmma_add` already lives. That
means: not renderer-local. It requires threading renderer capability into the `"remove reduces"`
graph_rewrite (today `ctx=ReduceContext()`, a plain accumulator-numbering counter with no renderer
reference at all — `codegen/__init__.py:186-188,324`) and building a genuinely new UOp there.

### 3.2 The correctness trap: MATVEC's real shape isn't the easy case

Naively: "when the GROUP_REDUCE range is ≤32, skip the buffer and call `simd_sum` on the pre-store
value." This is wrong for MATVEC's actual default kernel. The MATVEC heuristic applies **two**
LOCAL-family axes in the same threadgroup, not one:

```
codegen/opt/heuristic.py:89   Opt(OptOps.GROUP, group_axis, MV_THREADS_PER_ROW=8)   # the reduce split
codegen/opt/heuristic.py:91   Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE=4)          # separate: parallel OUTPUT ROWS
```

Both land in the same threadgroup (`gpudims.py:51` groups `WARP`, `LOCAL`, and `GROUP_REDUCE` together
into one `local_dims` set), giving a real MATVEC kernel a 4×8=32-thread threadgroup where lanes with
the *same* GROUP_REDUCE index but *different* LOCAL (row) index must **not** be summed together —
they're computing different output elements. Metal's built-in `simd_sum`/`simd_max` reduce across the
**entire** active simdgroup unconditionally; there is no partition argument. Calling `simd_sum` on the
per-thread partial here would silently sum across 4 unrelated output rows — wrong answer, not a crash.

Safe options, in increasing complexity:
1. **Restrict the fast path to GROUP_REDUCE-is-the-whole-threadgroup only** (no co-resident LOCAL
   axis) — safe, matches the plain-`sum` MSL sample in §1.1, but **excludes MATVEC's actual default
   shape** (GROUP=8 + LOCAL=4 together), so it wouldn't unlock the kernel this task's context is about.
2. **`simd_shuffle_down`-based partitioned tree reduction**, confined to power-of-2-width sub-blocks
   within the simdgroup (butterfly pattern, block width = GROUP_REDUCE size, block count = LOCAL
   size) — correct for MATVEC's real shape, but this is real per-lane index arithmetic
   (`simd_lane_id`, masking which lanes participate in each butterfly step), materially more code and
   more correctness surface than a bare `simd_sum` call, and is exactly the harder alternative the
   task prompt itself flagged ("Metal's `simd_shuffle_down` tree ... when >32").

Point 2 also generalizes to the `>32` two-level scheme T1.8b's attention kernel would actually need
(`Hd` ∈ {64,128} routinely exceeds one simdgroup) — meaning the "cheap ≤32 case" and the "two-level
case" converge on needing the *same* shuffle-based mechanism once MATVEC's real co-resident-LOCAL
shape is accounted for. There isn't a meaningfully cheaper subset to prototype.

## 4. Sized estimate (no prototype built)

| Piece | File | Est. lines | Notes |
|---|---|---|---|
| New `Ops` member (e.g. `SIMD_REDUCE`) | `tinygrad/uop/__init__.py` (~line 57, next to `WMMA`) | 1 | |
| Type-verify spec entry | `tinygrad/uop/spec.py` (~line 127, next to WMMA's) | 5-10 | mirrors WMMA's src/arg shape check |
| Renderer capability flag | `tinygrad/renderer/__init__.py:72` area | 3-5 | e.g. `simd_reduce_ops: set[Ops] = set()`, mirrors `tensor_cores` |
| Metal capability declaration | `renderer/cstyle.py:345` (`MetalRenderer.__init__`) | 1-2 | reuse the *existing* `int(target.arch[5:])>=7` Apple-family gate already sitting there for `tensor_cores` |
| Renderer plumbed into "remove reduces" | `codegen/__init__.py:186-188,324` | 5-10 | `ReduceContext` needs a `ren` field; mechanical |
| Gating + node construction | `codegen/__init__.py` (`fix_group_for_reduce`, `:170-184`, new sibling branch) | 40-80 | the real work: size check, **co-resident-LOCAL-axis check (§3.2)**, op-support check (sum/max only, most likely) |
| Metal string_rewrite case | `renderer/cstyle.py` (near `:62`, the WMMA `string_rewrite` entry) | 10-15 | `simd_sum`/`simd_max` call, following the WMMA UPat pattern exactly |
| Correctness tests | `test/opt/`, `test/unit/` | 60-100 | GROUP-reduce parity at sizes {1,7,8,16,31,32,33,64}; explicit test that GROUP+LOCAL-combined (MATVEC's real shape) is *not* naively fast-pathed; flag-off unchanged-output test |
| **Total, single-simdgroup case only, with the §3.2 restriction (option 1)** | 5 files | **~150-250** | Excludes MATVEC's real default shape; a "small, correct, but narrow" landing |
| **`simd_shuffle_down` partitioned reduce (option 2, needed for MATVEC's real shape AND the `>32` two-level scheme)** | same files + new lane-index arithmetic | **+150-300 more** | not broken down further — genuinely a second feature, would need its own scoping pass with real hardware validation of the butterfly pattern |

**Risks:** (1) simdgroup size is architecturally fixed at 32 on every shipping Apple GPU, so no
runtime query is needed — but the *feature* (`simd_sum` family) needs an Apple-GPU-family gate, not
just "any Metal device" (the existing `tensor_cores` gate's `>=7` check is a reasonable stand-in,
unverified against Apple's actual "simd-scoped reduction" feature table for this task). (2) BEAM search
interacting with a new `OptOps` sibling to GROUP/TC — needs the same `KernelOptError` fallback
discipline the MATVEC heuristic already uses (`heuristic.py:88-90`). (3) `pm_reduce_local` is a
fixed-point rewrite (WMMA folding, group-lowering, and accumulator-reduction all interleave); a badly-
scoped new pattern there risks firing on unintended shapes — needs careful `UPat` scoping, not just a
Python `if`.

## 5. Perf potential

T1.8b measured the missing warp primitive as **~5% of the bandwidth ceiling** in its hand-tuned
attention kernel (`TASKS.md` T1.8b entry, `tinygrad/llm/attn_kernel.py` docstring) — that's the
number this would plausibly recover *for that kernel*, and only once the `>32` two-level scheme
exists (§3.2), since `Hd`=64/128 there. For MATVEC-class kernels (GROUP=8, decode-dominant, run every
token), the win is a removed `threadgroup_barrier` plus a 4-8x shorter final serial-reduce loop per
kernel launch — real, but this repo has no measured absolute cost for a same-threadgroup Metal barrier
(the ~150 µs class numbers elsewhere in `memory.md` are *cross-device* sync costs from T3.4, a
different, much larger cost — do not reuse that number here). Net: plausible modest win on every
GROUP-opted kernel, unmeasured in absolute terms, gated behind real implementation work sized in §4.

## 6. Recommendation

Don't force it. This is the same shape of finding as T1.7 (PCONTIG) and T1.8b itself: a real, correctly-
diagnosed structural gap whose fix is not small once the actual default kernel shapes (not toy
examples) are accounted for. If T4.7 (symbolic `custom_kernel`) lands first and reopens T1.8b's tuned
attention kernel for real decode traffic, that raises the payoff of the `>32` two-level scheme enough
to justify scoping it as its own task with its own STOP conditions — start there, not from the ≤32 case,
since §3.2 shows the ≤32 case alone doesn't cover MATVEC's real shape either.
