# PR: MATVEC heuristic sees through CAST and quantized dequant

> **STATUS: ON HOLD — do not submit (Artur, 2026-08-18).** Prepared and verified, but Artur is
> deliberately holding off on PRs against upstream for now. When submitted, the AI-assistance
> disclosure below ships with it — non-negotiable.

## What / why

`hand_coded_optimizations`'s MATVEC path (`tinygrad/codegen/opt/heuristic.py`) only fired when the
reduce `MUL`'s two operands were bare `INDEX`. Real gemv kernels almost never look like that:

- fp16/mixed-precision gemvs wrap the reduce `MUL` and/or its `INDEX` loads in (possibly nested)
  `CAST` — e.g. a float32-GGUF weight stored under `HALF=1` down-casts then gets auto-promoted
  back up for compute, and the lossy down+up cast pair doesn't fold.
- GGUF-quantized weights (Q4_0/Q4_K/Q6_K/...) dequant to `MUL(bit-unpack expr, scale-unpack expr)`,
  each side containing one or more `INDEX`es into a packed `uchar` buffer, never a bare `INDEX`.
  In the real `.realize()` pipeline (not the simplified AST the existing MATVEC tests build),
  `pm_split_ranges` also splits the reduce axis into block/sub-block/byte sub-ranges before the
  heuristic runs, so `ranges_of(REDUCE)[0]` is no longer necessarily the range the vector operand
  actually indexes by.

These are the dominant batch-1 decode kernels for quantized local-LLM inference, so missing the
heuristic here is bandwidth-critical. The fix: `uncast()` loops through all wrapping `CAST`s
(was one level), the weight operand is matched by walking its ranges (`mat_ranges`) instead of
requiring a bare `INDEX`, and the reduce-range search scans for whichever split range is the bare
additive term in the vector operand's index instead of assuming index `[0]`.

## Diff stat

```
 test/opt/test_kernel_opts.py      | 61 +++++++++++++++++++++++++++++++++++++++
 tinygrad/codegen/opt/heuristic.py | 29 ++++++++++++++-----
 2 files changed, 83 insertions(+), 7 deletions(-)
```

`heuristic.py` itself: 22 insertions / 7 deletions (29 changed lines) — one lever, hand-verified.

## Measured (MacBook Pro M3 Pro, METAL, 4096x4096 gemv, min-of-20 kernel time)

| Case | Patched (this PR) | `MV=0` (heuristic off) | Unpatched upstream |
|---|---|---|---|
| fp16 gemv | 105-106 GB/s | 75-76 GB/s | 75.0-75.7 GB/s |
| Q4_0 gemv | 42.2-42.3 GB/s | 10.9-11.0 GB/s* | 11.0 GB/s |

\* one early `MV=0` reading came back at 33.7 GB/s (and one unpatched reading at 28.1 GB/s) before
the machine settled into a lower GPU clock state under sustained benchmarking load; both conditions
were re-measured several times back-to-back and converged to the numbers above. `MV=0` and
"unpatched upstream" take the same code path for these two ASTs (the old guard structurally never
matches CAST-wrapped or dequant-expression operands), and indeed land on the same number, which is
the internal-consistency check for the table.

Net: **fp16 ~1.4x, Q4_0 ~3.8-4x** membw at this revision — consistent with the original per-branch
findings (fp16 4096² gemv 56→100 GB/s; Q4_0 ~10→41 GB/s, Q6_K ~15→47 GB/s) and with the +48%
no-BEAM decode tok/s measured on qwen3:8b Q4_K_M with this lever included (`BENCH_NOTES.md`,
`task/T0.3-bench-harness`).

## Test coverage

`test/opt/test_kernel_opts.py`, three new cases, all green (`pytest test/opt/ -x -q -n12`, METAL):

- `test_matvec_heuristic_sees_through_cast` — fp16/mixed-precision matmuls, plus a genuine
  `CAST(CAST(INDEX))` (float32-GGUF-under-`HALF=1`) that only the double-strip fix catches.
- `test_matvec_heuristic_sees_through_quant_dequant` — Q4_0/Q4_K/Q6_K dequant-expression weight
  operand on the simplified `Scheduler`-on-raw-AST construction the existing MATVEC tests use.
- `test_matvec_heuristic_quant_dequant_real_pipeline` — the same Q4_0 case through the real
  `.realize()` pipeline, where `pm_split_ranges` has already split the reduce axis; this is the
  case the simplified-AST test above cannot catch (it needed the `mat_ranges`/split-range-search
  fix, not just the dequant-operand match).

## AI-assistance disclosure (ships with the PR)

This change was developed with AI assistance (Anthropic's Claude), directed and reviewed by
Artur (@arttarawork). Everything is hand-verifiable and was verified: the guard's failure modes
were confirmed against real ASTs before patching (not assumed), all numbers were measured on the
named hardware with the `MV=0` ≡ unpatched-upstream internal-consistency control, and each test
was checked to fail against the unpatched heuristic. Happy to walk through any part of it.

## Suggested commit message

```
codegen/opt: MATVEC heuristic sees through CAST and quantized dequant

The hand-coded MATVEC opts only fired when the reduce MUL's operands were
bare INDEX. Real gemv kernels wrap the MUL and/or INDEXes in (possibly
nested) CAST for fp16/mixed-precision, and GGUF-quantized weights dequant
to a MUL of bit-unpack expressions (multiple INDEXes into a packed buffer)
with the reduce axis later split into sub-ranges by pm_split_ranges -
neither shape matched the old guard, so the bandwidth-critical batch-1
decode gemvs for local LLM inference missed the heuristic entirely.

uncast() now strips all wrapping CASTs instead of one, the weight operand
is matched by walking its ranges instead of requiring a bare INDEX, and
the reduce-range search finds whichever split range is the bare additive
term in the vector operand's index instead of assuming ranges_of(REDUCE)[0].

Measured on M3 Pro (METAL), 4096x4096 gemv, min-of-20 kernel time:
fp16 ~75 -> ~105 GB/s, Q4_0 ~11 -> ~42 GB/s.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
```

## Notes for review

- Baselined on upstream `master` @ `e37b44d048f673be20016cde2963dbbf8b2333aa`
  ("casted CONST migration for LLVM and PTX [pr] (#17585)"). No drift against this tip: neither
  `heuristic.py` nor `test/opt/test_kernel_opts.py` had changed since the fork's baseline
  (`af2a43c85`), and the nearby casted-const renderer migration (`tinygrad/codegen/__init__.py`)
  only touches a post-optimization rewrite pass gated on `ren.casted_consts` (not yet set for
  METAL) that runs after `hand_coded_optimizations` — confirmed by running the actual repro
  scripts and the new tests against this tip, not just by inspection.
- Do not push / open the PR from this branch — hand off for review and submission.
