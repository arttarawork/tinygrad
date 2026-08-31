# Q38_PERF_PLAN.md — Qwen3.8-27B performance levers P2–P5 (T4.67–T4.70), opened 2026-08-31

**Context:** MTP stack (P1) is built and merging (PRs #13/#16/#17 + #14). This plan covers the remaining levers from the
2026-08-31 assessment: P2 prefix-state checkpointing, P3 chunk-64, P4 a real fused-form scan, P5 intra-layer tensor
parallelism. Measured anchors: 3.8 pooled decode 7 tok/s (152 ms/tok, 27.7 GB dense weight read @ ~193 GB/s blend),
prefill 22 tok/s flat (45 ms/tok, position-independent: scan + fixed costs), cold 19k Hermes prompt ~15 min, hop cost
~0.08 ms, METAL ~150 GB/s / 3090 ~900 GB/s on-device. Fresh-search fault tally 2/6 runs — protocol unchanged.

**Launch gate for the agent wave: PRs #14 and #17 merged** (tasks below assume GDN_HEAD_GROUPS, MTP head, spec decode,
and the T4.65 serve surface are all on master). Worktrees `tinygrad-t467/468/469` off the post-merge master. All agent
rules from MTP_SCAN_PLAN.md apply verbatim (Sonnet-max; DEV=CPU/NULL only; never tinygrad-td5, port 8081, tmux,
launchctl, other worktrees; full gates incl. `CHECK_OOB=1 DEV=CPU` serial + `SPEC=2 DEV=NULL test/null/` + the three
jit-key files + mypy + ruff; local commits only; BLOCKED.md if stuck). GPU phases are Fable-only.

---
## T4.67 — Prefix-state checkpointing in serve.py (P2; Sonnet task, the cold-prompt killer)

**Problem:** a cold 19k-token Hermes prompt costs ~15 min at 22 tok/s prefill. The serve splice cache only reuses the
model's OWN prior session; a gateway restart / new session / edited system prompt re-prefills everything, yet the first
~19k tokens (system prompt + tools) are byte-identical across sessions.

**Design:** snapshot the model's complete sequence state at a token boundary; restore it for any later request whose
token ids extend that prefix; prefill only the tail.
- Model side (tinygrad/llm/model.py): `Transformer.snapshot_state() -> dict` capturing, per GDN block, device-side
  `.clone()`s of `conv_state`/`recurrent_state` (the T4.64 checkpoint idiom — reuse its shape), per attention/MLA block
  a clone of the LIVE slice of each KV cache (`cache[..., :pos, :]` — never the full max_context buffer), plus
  `_cached_tokens` and the boundary position. `Transformer.restore_state(snap)`: `.assign()` states, write KV slices
  back into the big buffers, set `_cached_tokens`. Restore must leave a later `generate()`/`speculative_generate()`
  continuation TOKEN-IDENTICAL to an uncached run (that equality is the core test). Respect the recurrent exact-prefix
  rule (`get_start_pos`'s ssm branch): a snapshot is usable only when its token list is an exact prefix of the request.
- Serve side (tinygrad/llm/serve.py): `--state-cache[=MB]` flag (env `STATE_CACHE_MB`, default 2048; 0 = off = today,
  byte-identical). After each completed PREFILL (before generation), if this prompt's prefix isn't covered, store a
  snapshot keyed by the token-id tuple at the boundary; on each request, pick the LONGEST stored snapshot whose ids are
  a prefix of the request ids, restore, and prefill from there (interacts with the existing splice: try splice first —
  it's cheaper — fall back to snapshot, then cold). LRU-evict by estimated bytes (report the estimate per entry:
  KV bytes = layers×2×pos×head_bytes + states). v1 keeps snapshots on their home devices (restore = device copy,
  seconds); document the memory budget interaction (192k 35B config is tight — default cap small).
- Tests: unit — snapshot/restore round-trip on the tiny synthetic hybrid (from test_mtp_load's pattern): continuation
  after restore token-identical to uncached, incl. a GDN model and an attention-only model; boundary cases (restore then
  longer/shorter/mismatched prompt). Null-suite — mocked server: second session with shared prefix prefills only the
  tail (instrument prefill token count), eviction order, `--state-cache=0` byte-identical.
**Done-when:** flag-off path untouched; equality tests green; all standard gates. **Fable after:** measure a real 19k
Hermes prompt cold-vs-restored on the pooled 3.8 and 35B.

---
## T4.68 — chunk-64 coverage (P3; small Sonnet task, hardware experiment is Fable's)

Now that head-grouping keeps wide-head kernels under BEAM_UOPS_MAX, chunk 64 may be viable (the old cliff was measured
pre-split). Agent scope: extend the chunk-parity matrices to 64 — test/unit/test_gdn_scan_parity.py (both geometries)
and test/unit/test_attention.py's TestGatedDeltaNetHeadGroups (per-combo generated methods, ADD chunk 64 × G∈{2,3} at
H=48 only, to bound runtime; keep every test under ~60 s serial so the Linux Unit Tests job stays inside its wall —
measure with --durations and report), plus a one-line auto-rule note in gdn_chunk_for's comment that 64 remains
explicit-only. NO behavior change to gdn_chunk_for defaults. **Fable after:** `GDN_CHUNK=64` warmup + 6k/16k prefill
A/B on the pooled 3.8.

---
## T4.69 — fused-form (WY/chunked) scan as a pure-Tensor path (P4; the prefill lever)

**Problem:** 3.8 prefill is bound by ~45 ms/tok of position-independent cost dominated by the DeltaNet scan's unrolled
per-t chain (32 sequential steps of tiny broadcast ops → deep kernels, ~1.96 GB/tok of scan traffic measured in TD.6).
The AMD lane already implements the right algorithm as one fused RDNA3 kernel (`tinygrad/llm/kernels/amd.py`,
`gated_delta_prefill`): chunkwise WY-form — intra-chunk matmul-shaped work + a single inter-chunk state recurrence —
instead of T sequential state updates.

**T4.69a (Sonnet task):** port the ALGORITHM (not the RDNA3 specifics) to a device-agnostic pure-Tensor implementation
`gdn_scan_wy(q, k, v, beta, alpha, state, start_pos)` in model.py (or a new tinygrad/llm/gdn_scan.py), selected by a new
`GDN_SCAN_IMPL` ContextVar (0=auto→current loop everywhere for now; 1=loop; 2=wy) inside GatedDeltaNetBlock._attention's
else-branch, composing with GDN_HEAD_GROUPS (grouped slices feed whichever impl). Read gated_delta_prefill CAREFULLY as
the algorithm reference — same decay/delta-rule/output semantics, chunk-internal matrices (the (T,T) triangular
delta-solve), and state hand-off; also read the fla-style chunked GDN references in comments if present. Requirements:
works at chunk sizes 1..64 and symbolic T_pad (pad-safe), both geometries, start_pos>0 continuation; parity vs the loop
at rtol/atol 1e-4 documented (WY reorders float math — exact equality is NOT expected; justify the bound the way
test_gdn_scan_parity.py does); extend that file to run its whole matrix over both impls. Emit
extra/gdn_wy_evidence.py (DEV=NULL): kernel count + largest-kernel uop count, loop vs wy, H=48 real dims — the wy form
should lower to matmul-shaped kernels (reduce ops over the chunk axis), not 32-step chains; report honestly either way.
**Done-when:** parity green both impls × both geometries × chunks {1,4,32,64}; default path byte-identical; gates.
**T4.69b (Fable, GPU):** BEAM the wy kernels on the pooled split; 6k/16k prefill A/B vs 22 tok/s; flip GDN_SCAN_IMPL
auto only on measured win. **T4.69c (deferred):** hand-written METAL/NV kernels via the custom-kernel seam only if the
Tensor-level form leaves large headroom.

---
## T4.70 — intra-layer tensor parallelism (P5; staged — design is NOT a Sonnet task)

**Why:** decode reads each layer's weights from ONE device sequentially (~19 GB @NV + ~10 GB @METAL ≈ 152 ms/tok).
Splitting each matmul's weight across both devices makes the reads concurrent: ceiling ≈ max(share/BW) + per-layer sync
(~10-15 ms/tok at 0.08 ms/hop, activations 10 KB) → ~15-20 tok/s before speculation, multiplicative with MTP.
- **T4.70a (Fable design note, or a Fable-fork agent per [[subagent-deployment-policy]] — open-ended analysis is not
  Sonnet-proof):** sharding layout (FFN row/col split first — ~17 GB of 3.8's 27.7 GB; attention later; GDN blocks shard
  on the HEAD axis, the same independence T4.62 proved), allreduce strategy over the tunnel, interaction with
  device_map/JIT capture, memory/KV placement budget, failure modes. Produces the T4.70b/c specs.
- **T4.70b (Sonnet, after 70a):** dense-FFN TP for TransformerBlock on CPU:0/CPU:1 behind a `tp:` device-map segment —
  parity tests in the T3.1/test_llm_device_map style (multi-CPU is fully testable without GPUs).
- **T4.70c (Sonnet, after 70b):** GDN head-axis sharding + attention QKV/O split, same harness.
- **T4.70d (Fable, GPU):** pooled bring-up, BEAM, decode A/B, then stack MTP on top.

---
## Sequencing
1. PRs #14 + #17 merge (user) → 2. fire T4.67 + T4.68 + T4.69a agents in parallel (worktrees off new master) →
3. Fable GPU windows: T4.68 A/B (cheap) → T4.69b → T4.67 real-prompt measurement → 4. T4.70a design when the above
settle (or earlier if commissioned). MTP hardware validation (P1) rides the first post-merge window alongside T4.68.

---
## STATUS 2026-08-31 (post-battery — supersedes the sequencing above)
P1 MTP: built+merged; GPU-BLOCKED on per-position draft traces → **T4.66a in flight** (then re-measure). P2 state cache: SHIPPED + measured
working (snapshot hit `in: 5872 +26`); standing integration pending a 35B-scale measurement. P3 chunk-64: CLOSED (identical perf,
token-identical). P4 WY scan: prefill +27-29% measured; **T4.69b (decode gate) in flight** → re-measure → auto-flip decision. P5 TP: design
pending (Fable). Full numbers: HANDOFF_2026-08-31.md §2.
