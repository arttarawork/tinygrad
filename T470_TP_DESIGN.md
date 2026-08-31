# T470_TP_DESIGN.md — intra-layer tensor parallelism for the pooled split (T4.70a, 2026-08-31)

## Goal and the number it attacks
Dense decode reads every layer's weights from ONE device, serially: qwen3.8-27B ≈ 152 ms/token (≈9.9 GB of FFN+attn on METAL
@~150 GB/s dominating; NV's share reads at ~900). Splitting each layer's matmuls ACROSS both devices makes the reads concurrent:
per-layer time → max(share_d/BW_d) + sync. Measured sync budget: ~0.08 ms/hop, activations 10-20 KB/block ⇒ ~8-12 ms/token of
transfer+add for 64 blocks — small against the ~40-50 ms/token the concurrency recovers.

## Architecture decision: MODULE-LEVEL TP (not Tensor.shard)
**Rejected: native `Tensor.shard(devices, axis)`.** (a) Even splits only (no boundaries arg) — bandwidth-proportional ~6:1
NV:METAL is the entire win; 50/50 would make METAL the bottleneck and LOSE to today. (b) Our weights are lazy dequant
expressions over packed GGUF bytes: sharding the decoded tensor risks replicating the packed buffer per device before slicing
(memory-catastrophic at 29 GB). (c) Multi-buffer graphs are novel territory for our JIT-captured pooled flow.
**Chosen: per-device Linear halves, wired at the module/state-dict level** — the same machinery T3.1/T4.63 already proved:
- Each FFN Linear splits into two Linears (`.tp0/.tp1`), one per device, holding a ROW SLICE of the GGUF tensor. The slice is
  taken on the SOURCE (disk) tensor before load via the T4.63 state-dict rename/slice trick, so each device loads and
  dequantizes ONLY its packed slice. Q8_0 blocks run 32-wide along the INPUT dim: output-dim (row) slices always cut on block
  boundaries; input-dim slices must land on multiples of 32 (17408 and 5120 both do for any 32-aligned split point).
- Forward per FFN (column-parallel gate/up, row-parallel down — the classic Megatron shape):
  `x` broadcast to both devices (one ~10 KB copy) → `h_d = silu(gate_d(x_d)) * up_d(x_d)` per device (each device's slice of
  the hidden) → `y = down_0(h_0) + down_1(h_1).to(dev0)` (partial sums; ONE ~10-20 KB transfer + add per block). down_d's
  input-dim slice matches its device's hidden slice ⇒ no hidden-state transfer at all.
- JIT: every kernel is single-device; cross-device traffic is plain `.to()` COPYs — exactly the shape T3.1 proved captures
  cleanly ("only COPY spans devices, asserted per captured call").

## Allocation model (what goes where)
Let r = NV's fraction of each sharded FFN. Unconstrained optimum r* = BW_nv/(BW_nv+BW_m) ≈ 0.857 → FFN effective ~1050 GB/s.
But NV capacity binds: NV ≤ ~22 GB must also hold the attention weights + KV + (later) the MTP head. So the real design is a
small allocation problem: choose r and the per-block placement of NON-sharded parts to maximize
  1 / Σ_l max(r·w_ffn/900, (1-r)·w_ffn/150) + (non-sharded reads at their device BW), s.t. NV bytes ≤ budget.
First-cut numbers (3.8: FFN ≈ 17.4 GB, attn ≈ 4.6 GB, embed/head ≈ 1.6 GB, KV@64k ≈ 4 GB): r=0.75 puts 13 GB of FFN on NV +
attn 4.6 + KV 4 ≈ 21.6 GB ✓; FFN time ≈ max(13/900, 4.4/150) ≈ 29 ms vs today's ~59 ⇒ decode ≈ 152→~120 ms (~8.3 tok/s,
+27%) at r=0.75; pushing KV partially to METAL or attn sharding (T4.70c) buys r→0.85 ⇒ ~16 ms FFN ⇒ ~105 ms (~9.5 tok/s).
With T4.66-fixed MTP on top (×1.5-2 on accepted tokens), 13-19 tok/s is the stacked projection. (35B: the MoE expert path
shards on the EXPERT axis instead — routing whole experts per device, the existing `experts:` seam generalized — design
sketched, deferred.)

## Failure modes / cautions
Split points must be 32-aligned (assert at load). The +2 copies/block per token are latency-bound — batch the adds device-side
(no host sync). Warm BEAM cost: sharded matmuls = new ASTs on both lanes (one fresh search, budget 90 s). GDN blocks stay
whole-device in v1 (their scan state is per-head; head-axis sharding is T4.70c using the SAME independence T4.62 proved).
The gguf loader's `_dev_for(name)` routes whole tensors — TP tensors bypass it via the slice-rename path (document in-code).

## Task breakdown
- **T4.70b (Sonnet-proof, ~T4.67-sized):** dense-FFN TP behind a `tp:r` device-map segment (e.g. `tp:NV=0.75,METAL=0.25`):
  parse_device_map extension, the Linear-halves module surgery + state-dict slice mapping, forward wiring, 32-alignment
  asserts; CPU:0/CPU:1 parity tests in the T3.1/test_llm_device_map style (token-identical vs unsharded, both a dense-FFN
  hybrid tiny model and generate()-level); gates as standard. NO GPU work.
- **T4.70c (Sonnet, after 70b):** attention QKV/O head-split + GDN head-axis split, same harness.
- **T4.70d (Fable, GPU):** pooled bring-up on the 3.8, BEAM, allocation sweep (r grid), decode A/B, then stack MTP.

Fire T4.70b once wave 3 (T4.69b/T4.66a) merges — it touches the same model.py surface.
