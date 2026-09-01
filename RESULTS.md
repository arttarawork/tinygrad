# T4.73a: bug-1 off-serve reproduction attempt — RESULTS

**Outcome: bug-1 did NOT reproduce anywhere in the prescribed matrix, or in a follow-up bonus sweep.**
Every config listed below ran the REAL `Transformer.warmup()` -> `Transformer.generate()` sequence
(never a standalone replica loop), on `DEV=CPU`, and came back with 0 NaNs in every GDN block's
`recurrent_state`/`conv_state`. This is reported honestly per the task brief: a clean sweep across
five axes, plus the two axes (chunk width, real geometry) most implicated by the hardware trace,
tested combined. Section 4 lists concrete remaining hypotheses for the next hardware session.

Script: `extra/wy_scale_repro.py` (parameterized, argparse per axis, `--matrix` runs the grid below).
Never pushed; this file and the script are local to `task/T4.73a-wy-scale-repro`.

## 1. Confirming the harness exercises the real replay mechanism

Before trusting a "clean" result, I verified the harness actually reaches the JIT-replay regime
bug-1 needs, not just structurally but mechanically (`tinygrad/engine/jit.py` `TinyJit.__call__`,
lines 258-306): `cnt==0` runs `self.fxn` eagerly, `cnt==1` runs `self.fxn` once more **while
capturing** (Python still runs), and `cnt>=2` is **pure replay** — `self.captured(...)` only,
`self.fxn` never called again. `Transformer.warmup()` calls `generate([0], temperature=T)` twice for
each of `T in (0.0, 1.0)`; because `GDN_CHUNK` is set once via `Context(...)` for the whole run (exactly
how `serve.py` sets it once via env for the process lifetime), warmup's own prefill calls land on the
**same** jit key `(True, greedy, chunk_size, False)` the real prompt will use — so warmup already
burns that key's cnt to 0 then 1, and the real prompt's first prefill call is cnt==2: a genuine replay.

This is directly confirmed empirically, not just argued: every single result row below shows
`python_ran=False` for the checked stage (`gdn_last_scan_impl`, model.py's test-introspection global,
never grew during the real prompt) — i.e. `run_scan`'s Python body never re-entered, for **either**
impl, exactly the "no Python ran" signature HANDOFF's WY_TRACE describes for the hardware replay.
So the harness is hitting the right mechanism; the clean results are informative negatives, not an
artifact of never reaching replay.

## 2. Prescribed matrix (6 phases, escalating one axis at a time)

All rows: 2s.d. seed=0, vocab_size=32, weight_scale=0.1 (matches `test_gdn_scan_parity.py`'s
`make_block` convention), prompt_len=20, 4 tokens pulled (prefill output + 3 decode steps), both
`GDN_SCAN_IMPL` values run per row (loop = control, wy = the bug-1 candidate).

| Row | chunk | geometry | blocks | dtype | extra | loop | wy | verdict |
|---|---|---|---|---|---|---|---|---|
| P1-chunk8  | 8  | tiny (h4,d8,kv2)      | 2 | f32 | -            | PASS | PASS | clean |
| P1-chunk16 | 16 | tiny (h4,d8,kv2)      | 2 | f32 | -            | PASS | PASS | clean |
| P1-chunk32 | 32 | tiny (h4,d8,kv2)      | 2 | f32 | -            | PASS | PASS | clean |
| P2-real-chunk16 | 16 | real38 (h48,d128,kv16) | 2 | f32 | -       | PASS | PASS | clean |
| P2-real-chunk32 | 32 | real38 (h48,d128,kv16) | 2 | f32 | -       | PASS | PASS | clean |
| P3-real-blocks8 | 32 | real38 | 8 | f32 | -                        | PASS | PASS | clean |
| P4-real-8blk-f16 | 32 | real38 | 8 | f16 | -                       | PASS | PASS | clean |
| P5-real-8blk-f16-attn | 32 | real38 | 8 | f16 | every-4th block is real attention | PASS | PASS | clean |
| P6-real-8blk-f16-p2 | 32 | real38 | 8 | f16 | + 2nd unrelated 13-tok prompt | PASS/PASS | PASS/PASS | clean |

"real38" = `SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=128*48)`
== `test/unit/test_gdn_scan_parity.py`'s `GEOMETRIES["38"]`, the qwen3.8-27B shape bug-1 was
hardware-confirmed on. "f16 dtype" mimics `from_gguf`'s default `HALF=1` weight cast (Q8_0 dequants to
f32 in `gguf.py`; `model.py`'s `from_gguf` then casts every loaded weight to float16 -- confirmed by
reading that code path, see §5). `recurrent_state`/`conv_state` themselves are NOT part of this axis:
they're always `dtypes.default_float`/`kv_cache_dtype()` regardless of loaded-weight dtype, in both the
constructed test model and the real one (same `_init_state` code either way).

**16-block real-geometry spot check** (not in the repeatable `--matrix` list — see §3 for why):
`--chunk 32 --real-geometry --blocks 16 --dtype f16` -> also clean (loop & wy both PASS,
tok=`[7, 19, 1, 29]`).

Every row above: `python_ran=False` on every checked stage, confirming genuine JIT replay per §1.

## 3. Per-config runtime (why the matrix is capped at 8 blocks, not 16)

Measured wall time per **single** `run_config()` call (one impl) at real geometry, chunk=32:
2 blocks ~10s, 8 blocks ~45s, 16 blocks ~84s -- roughly linear, ~5s/block (schedule/codegen overhead
per kernel occurrence dominates, same characteristic `test_gdn_scan_parity.py` documents for a lone
`_attention` call; raw FLOPs at this geometry are trivial either way). The task's own budget
(<60s/config) is why Phases 3-6 use 8 blocks, not 16 -- 16 was spot-checked manually once instead
(see above) rather than folded into the repeatable matrix.

## 4. Bonus exploration (not in the prescribed axes, but cheap and worth doing before concluding)

The task's own code comment (`model.py`, `gdn_scan_wy`'s docstring, T4.69a addendum) names a concrete,
plausible WY-specific failure mode: a head with aggressive per-step decay compounded over a full chunk
can underflow the chunk's cumulative decay product towards 0 and blow up the WY division -- while the
loop, which never explicitly forms `1/decay`, would just quietly forget the old state. `weight_scale`
(uniform random-init magnitude) was already a free parameter in the harness, so I swept it at real
geometry / chunk=32 / 2 blocks (cheap, ~10-20s/row) looking for a **WY-only** window between "clean"
and "both blow up":

| weight_scale | dtype | loop | wy | verdict |
|---|---|---|---|---|
| 0.1 - 0.7 | f32 | PASS | PASS | clean |
| 0.8, 0.9, 1.0 | f32 | NAN-FLOOD (100%, tok constant 32) | NAN-FLOOD (100%, tok constant 32, IDENTICAL to loop) | **BOTH-IMPLS-BROKEN -- not bug-1** |
| 0.3, 0.5, 0.7 | f16 | PASS | PASS | clean (f16 rounding alone doesn't move the cliff) |
| 0.1, chunk=16, 8 blocks, f16 | f16 | PASS | PASS | clean |

There IS a real numerical cliff (between weight_scale 0.7 and 0.8, sharp, both f32 and f16), but it is
**not bug-1**: loop and wy break at the exact same threshold, with bit-identical output tokens and the
exact same 100%-of-every-element NaN flood on **every** GDN block uniformly. Bug-1's hardware
fingerprint is asymmetric (block 0 only one head's worth of NaN, blocks 1..N fully NaN) and WY-only;
this bonus finding is symmetric across impls and uniform across blocks -- a mundane "the recurrence's
own math diverges with unrealistically large random weights over a 32-token chunk" result, not a
JIT/WY bug. Flagged per the task's instruction, not claimed as bug-1. No further time spent bisecting
this cliff since it's off-target.

## 5. Weight dtype: what `from_gguf` actually produces (read, not assumed)

`tinygrad/llm/gguf.py`'s Q8_0 dequant (`ggml_type == 8`): `blocks[:,:2].bitcast(float16).cast(float32) *
blocks[:,2:].bitcast(int8)` -- produces **float32** lazily. `tinygrad/llm/model.py` `Transformer.from_gguf`
(line ~1100): `state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}`
-- every loaded weight (including every GDN block's `attn_qkv`/`ssm_beta`/`ssm_alpha`/`ssm_dt`/`ssm_a`/
`ssm_conv1d`/`ssm_out`) is cast to **float16 by default**, before `load_state_dict`. `recurrent_state`
and `conv_state` are NOT loaded weights -- they're allocated fresh by `GatedDeltaNetBlock._init_state`
at `dtypes.default_float` (fp32) and `kv_cache_dtype()` (fp16 unless `KV_F32=1`) respectively, same in
the constructed test model as in a real `from_gguf`-loaded one. This is what `--dtype f16` in the
script replicates: cast every parameter (`nn.state.get_parameters(model)`) to float16 post-init,
leaving state buffers untouched -- matching the real load path exactly, not a guess.

## 6. Axes eliminated by this session (on top of what HANDOFF already lists from hardware)

chunk width (8/16/32, alone and combined with real geometry) -- real GDN geometry (48 heads/128 dim,
alone and combined with every other axis) -- block count (2/8/16, real geometry) -- weight dtype
(f16 mimicking `HALF=1`, alone and combined with block count/attention/second-prompt) -- interleaved
real-attention blocks (every 4th, qwen3.8's pattern) -- a second, differing-length, unrelated prompt
after the first -- weight-init magnitude 0.1-0.7 (both dtypes) -- chunk=16 combined with the
otherwise-worst config (real geometry, 8 blocks, f16). All CPU, all through real `warmup()`+`generate()`,
all confirmed to actually hit JIT replay (§1).

## 7. Remaining hypotheses (since nothing reproduced -- this is the honest, requested outcome)

1. **Request-count/history scale**: my harness runs warmup + 1-2 generate() calls total. The hardware
   bug was hit on a live server after real traffic; it may need many more sequential requests (jit
   `cnt` growing well past 2, or the memory planner's suballocation arena settling into a specific
   layout only after several distinct buffer-lifetime patterns have cycled through it) -- worth a CPU
   run that loops dozens of prompts through one warmed model before checking for NaN, not just one.
2. **State-cache interaction untested here**: T4.67's `snapshot_state`/`restore_state` (device-side
   clone/assign, used by `serve.py` for cross-session reuse) was never exercised in this harness.
   Bug-2 (a different, already-parked bug) was found "while hunting" bug-1, suggesting state-cache
   traffic was present on hardware around when bug-1 was seen -- worth a config that snapshots and
   restores state between prompts, not just runs prompts independently.
3. **Full model depth/kernel count**: qwen3.8-27B's real graph also has MoE FFN blocks, a much larger
   `vocab_size` (~150k, not 32), and likely more than 16 total blocks -- if the bug is a memory-planner
   or graph-split threshold effect (a buffer-count or arena-size cliff, not a shape/dtype effect), it
   may only appear once the WHOLE model's per-jit-capture kernel count is much larger than anything
   tested here. Cheap next CPU step: add a dense/MoE FFN block per GDN block (not just attention) to
   push total kernel count up without needing real hardware.
4. **Trained-weight structure vs. i.i.d. random**: §4's cliff shows the recurrence IS scale-sensitive,
   but uniform random init moves every channel together. A trained model can have a few specific
   heads/channels sitting near a knife-edge while the rest are nowhere close -- something i.i.d.
   `randn * scale` at any single scale value may never produce, no matter how many scales are swept.
   Worth trying non-uniform init (e.g. one outlier head's `ssm_dt` bias pushed extreme, rest left at
   0.1) instead of a uniform sweep, if a future session wants to keep pulling on this thread.
5. **BEAM search**: hardware runs `JITBEAM=2` (real kernel search); every run here used BEAM's default
   (unset/off). HANDOFF already lists bug-1 as "BEAM-independent" from hardware testing, so this is
   low-priority, but it wasn't independently re-checked here (`BEAM`/`JITBEAM` env still forbidden from
   this CPU-only worktree in the sense that a real search needs real device timing to mean anything).

No fix was attempted (nothing to fix -- the bug never reproduced), and no regression test was added
(the task only calls for one if a reproducing config is found and fixed).

## 8. Exact commands

Single config (any axis combo):
```
cd /Users/artur/Documents/tinygrad-t473a
DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_scale_repro.py \
  --chunk 32 --real-geometry --blocks 8 --dtype f16 --interleave-attn --second-prompt-len 13
```
Full prescribed matrix (§2, the 6 phases above, ~8 minutes total on this machine):
```
DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_scale_repro.py --matrix
```
16-block spot check (exceeds the 60s/config budget, run manually, not part of `--matrix`):
```
DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_scale_repro.py \
  --chunk 32 --real-geometry --blocks 16 --dtype f16
```
Bonus weight-scale sweep (§4):
```
for ws in 0.1 0.2 0.3 0.5 0.7 0.8 0.9 1.0; do
  DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_scale_repro.py \
    --real-geometry --chunk 32 --weight-scale $ws
done
```

**Hardware cross-check (for Artur, not this agent -- this worktree/agent is CPU/NULL-only by hard
rule and the script asserts `DEV=CPU` for exactly that reason)**: to see whether any of the above
axes behaves differently off-CPU even at this tiny/scaled level (i.e. rule in/out "needs a real GPU
backend, even at tiny scale" as a variable -- distinct from needing serve's full scale), on the Mac
itself with the eGPU idle (never while the pooled server or any other `DEV=NV` process is up):
edit `extra/wy_scale_repro.py`'s `assert os.environ.get("DEV") == "CPU"` line (top of the file) to
also allow `"METAL"`, then:
```
DEV=METAL PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python extra/wy_scale_repro.py --matrix
```
If that also comes back clean, hypothesis 3 above (full model depth/kernel count) becomes the
strongest remaining lead, since geometry/dtype/chunk/device would then all be eliminated and only
scale-of-the-whole-graph would be left untested.
