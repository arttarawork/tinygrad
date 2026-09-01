# T4.70e — Why FFN-TP measured 0.27 tok/s over the USB4 tunnel, and what (if anything) fixes it

Written 2026-09-01 from code reads + the T4.70d bring-up measurements (Qwen3.8-27B, `tp:NV=0.62,METAL=0.38`,
64k ctx, warm: decode ~0.27 tok/s = 3.77 s/token, prefill 5 tok/s on a 23-tok prompt ≈ 4.6 s/forward — i.e. the
cost is **per forward pass, ~59 ms per block**, not per token of prefill width). Baselines on the same stack:
plain pipeline decode 7 tok/s (~143 ms/token), prefill 22-28 tok/s.

## §1 Where the 59 ms/block actually lives

The per-block TP forward (`tinygrad/llm/model.py:354-370`, T4.70b) is: broadcast `x.to(shard_dev)` (one ~10 KB
copy to the non-home device), 3-5 shard kernels there, `d_lin(h_d).to(primary)` partial return (~10-20 KB), eager
`a + b` reduce on the home device. The original budget (`T470_TP_DESIGN.md:6-7`) priced this at ~0.08 ms/hop from
TD.3's measured steady-state boundary copy (`TD3_POOLING_NOTES.md` §5: 8 KB METAL↔NV = **80-90 µs**, flat with
size, both directions). The measurement says each block actually costs ~59 ms — ~600× the budget. The copy was
never the cost. Three structural multipliers, each from source:

1. **Graph islands cannot span backends.** `graph_split_rewrite` (`engine/jit.py:31-63`) flushes the current
   batch whenever the next call's devices aren't supported by the batch's graph class. `HCQGraph.supports_uop`
   (`runtime/graph/hcq.py:326-329`) rejects any non-HCQ device — METAL can never join an NV batch — and
   `MetalGraph.supports_uop` (`runtime/graph/metal.py:114-118`) is the mirror. Pipeline mode alternates devices
   ~20 times per token, so per-device runs still form big islands (TD.3 §5 shows a token = 4 big `batched`
   rows + 2 copies = **6 dispatch rows**). TP alternates **every block**: home-device kernels → copy → shard
   kernels → copy → home add → next block. Result ≈ 2 tiny islands + 2 copies + 1 add per block ≈ **5+ rows ×
   64 blocks ≈ 320+ dispatch rows per token**, and 1-kernel batches aren't even graphed (`jit.py:39`,
   singletons go out as bare calls).
2. **Every dispatch row pays ~1 ms of host-side tunnel-mediated gap.** TD.3 §5's own steady-state token:
   27.70 ms wall vs Σ(device `tm`) ≈ 20.7 ms → **~7 ms of host gap across 6 rows ≈ ~1.2 ms/row** — doorbell
   MMIO RPCs to the TinyGPU process, queue setup, signal waits. That gap scales with row count, not work:
   320 rows × ~1 ms ≈ 300+ ms/token before any copy semantics.
3. **Cross-backend copies are synchronous host bounces.** `exec_copy` (`engine/realize.py:170-181`): the
   `_transfer` fast path requires `dest.device.split(":")[0] == src.device.split(":")[0]` — same backend only.
   METAL↔NV always falls to `_copyout` → host memoryview → `_copyin`, which forces the **source device fully
   idle per copy** (METAL `waitUntilCompleted` ~150 µs, TD.3 §5; remote-NV timeline signal arriving as a DMA
   write over the tunnel, ~ms-class). 128 forced serialization points per token on top of the row gaps.

Honest reconciliation: 5-15 tunnel-serialized points per block at ~1-5 ms each brackets 5-75 ms/block; the
measured 59 ms sits inside it. The exact split (row gap vs copy sync vs doorbell RPC) is one `DEBUG=2` round
away — see §5. (The 200 ms `_sleep` drain quantum (`ops_nv.py:56-66`) only engages on waits >200 ms — a
secondary effect at most.)

## §2 Design A — hop elimination via layout (defer the all-reduce across N blocks)

**Not available without sharding everything else.** The partial sums must be reduced before the *next* block's
norm+attention because RMSNorm and attention consume the full `dim`-wide summed residual on the block's home
device — `x_{i+1} = x_i + ffn_i(x)` is a hard serialization point per block. Deferring the reduce means the
residual lives split across devices, which requires the interleaved GDN/attention/norm modules to be sharded
(T4.70c) or fully replicated on both devices. The Megatron-complete version (shard attn+FFN both) still
requires **two all-reduces per block** by construction — it changes what crosses the tunnel, not how often.
Replicating attention/GDN on both devices removes the reduce but duplicates ~40% of weights (doesn't fit — the
whole point of pooling is that neither device holds the model). **A is rejected on this transport**: no layout
keeps the per-block barrier count below 1 while the residual stream is device-spanning.

## §3 Design B — hop batching / overlap

- **Overlap block i's reduce with block i+1's compute:** impossible for the same reason as A — block i+1's
  first op consumes block i's reduced output. The dependency chain is strictly sequential at decode.
- **Coalesce the 64×2 tiny copies into fewer transfers:** the copies belong to different (serially dependent)
  blocks; there is nothing concurrent to coalesce.
- **What's actually available:** shrink the per-block row count (fuse the shard-side 3-5 kernels; make the
  reduce ride inside a graph island by teaching one runner to embed cross-backend copies — a new graph runtime,
  upstream-class work). Best case halves the rows: ~160 × ~1 ms ≈ 1.6-1.9 s/token → **~0.5-0.6 tok/s**. Still
  ~12× worse than the 7 tok/s baseline. **B is a bounded loser: weeks of work for a config that still loses.**

## §4 Design C — the null option (TP is the wrong shape for this pool)

TP pays `N_extra_rows/block × t_row + t_copy_sync/block` per block to save concurrent-read time
`Δt_read/block ≈ (D_ffn/BW_slow − D_ffn/BW_eff)/64 ≈ 0.6-0.8 ms/block` (the design doc's own ~40-50 ms/token
recovery, `T470_TP_DESIGN.md:6-7,27`). Break-even:

```
TP viable  ⇔  (rows/block × t_row) + t_sync/block  <  ~0.7 ms/block
```

On this stack `t_row ≈ 1.2 ms` **alone** exceeds the whole budget at even ONE extra row per block. TP needs
either a transport where a cross-device dependency costs ≪150 µs end-to-end (RDMA-class, or a cross-backend
graph runtime that keeps the whole token inside one submitted island), or blocks big enough that 0.7 ms/block
becomes 10+ ms/block (models ~15× larger). Neither describes this machine. **Pipeline placement (today's
per-block device map) is the right shape for a 2-device tunnel-separated pool**: it pays ~20 boundary crossings
per token (~2 ms total device-side, ~7 ms host-side) instead of ~320.

## §5 Verdict, ranked

| rank | design | expected decode vs 7 tok/s | cost | call |
|---|---|---|---|---|
| 1 | **C — keep pipeline, shelve TP** | 7 (unchanged) | zero | **adopt** |
| 2 | B (row-count surgery + cross-backend graph runtime) | ~0.5-0.6 best case | weeks, upstream-class | reject |
| 3 | A / full Megatron (needs T4.70c too) | still barrier-bound ≥1 all-reduce/block | largest | reject |

The T4.70b machinery stays merged and correct (CPU-parity-tested; useful the day the transport changes); the
`tp:` knob simply stays unused on this hardware. The real decode levers on this machine remain the WY auto-flip
(+27-29 % prefill, T4.73), MTP fixed-cost work (T4.66c), and quant/placement tuning.

**Cheapest falsifying experiment for §1's mechanism claim (zero code):** one TP decode step with
`POOLED_ENV="DEBUG=2 ..."` — the row dump either shows ~300+ rows with ms-class inter-row gaps and no big
`batched` islands (confirms: shattering + per-row tunnel gap) or it shows few big islands with huge device
times (refutes §1, points back at kernel quality). One request, ~5 min including load, read the pane.
