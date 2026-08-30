# MTP_SCAN_PLAN.md — Qwen3.8-27B usability chain (T4.61–T4.66), opened 2026-08-29

**Goal:** 3.8-27B decode 6.6 → 15–25 tok/s and prefill 19 → 40+ on the pooled METAL+3090 split, via
(A) a 48-v-head scan kernel path, (B) MTP self-speculative decode, (C) optionally UD-Q4 weights.
Rationale + math: TASKS.md 2026-08-29 MTP rows. Fable plans/reviews/measures; Sonnet-max agents build (CPU-only).

## Measured anchors (2026-08-28/29, pooled 192k map unless noted)
- 3.8 Q8: decode 6.56 tok/s @192.7 GB/s (152 ms/tok, 27.7 GB/tok dense) · prefill 19.4 tok/s chunk-32 (scan-bound, ~1.65 s/chunk, non-weight ≈47 ms/tok) · ctx 65536 map `t459/map.txt`.
- 35B Q8 (reference): prefill 98 (T4.58 flat), decode 28.
- llama.cpp draft-mtp on 35B (retired llama-server): ~+40% structured, ~+8% prose.
- Verify-cost model: k-token dense forward ≈ 144 ms weights + k×scan-overhead → MTP ceiling ~2× with today's fallback scan, ~3× with a real 48-head kernel; UD-Q4 (17.56 GB) may stack ~1.5× if dequant hides under bandwidth (bench before believing).

## Geometry (from GGUF kv, `qwen35` arch, local Q8 file)
3.8: dim 5120, 65 blocks (64 + 1 nextn), full_attention_interval 4 (16 attn / 48+1 GDN), heads 24/4 kv, head 256,
ssm: state_size(head_k_dim)=128, group_count(num_k_heads)=16, time_step_rank(**num_v_heads=48**), head_v_dim=6144/48=**128**,
conv_kernel 4. Recurrent state per GDN layer: (B, 48, 128, 128) fp32 (`model.py` `GatedDeltaNetBlock._init_state`).
nextn (blk.64): full-attn TransformerBlock tensors + dense FFN 17408 + `nextn.eh_proj [10240,5120]`, `nextn.enorm/hnorm/shared_head_norm [5120]` (DeepSeek-style: eh_proj(concat(enorm(emb(tok_next)), hnorm(hidden))) → block → shared_head_norm → shared lm_head).
35B blk.40 is the MoE analogue (20 tensors) — same wrapper, MoE inner block.

## Track A — 48-head scan
- **T4.61 scan micro-harness** (branch `task/T4.61-scan-harness`, worktree `tinygrad-t461`): standalone parity+bench
  instrument for `GatedDeltaNetBlock._attention`'s scan at both geometries (35B: 32 v-heads×128; 3.8: 48×128), chunk 1–64,
  CPU parity vs a naive reference; emits `extra/gdn_scan_bench.py` I run later on METAL/NV. No GPU in-task.
- **T4.62 head-group split** (branch `task/T4.62-scan-headgroups`, worktree `tinygrad-t462`): v-heads are independent in
  the scan (state (B,H,V,K), per-head decay/delta/out) → evaluate the unrolled scan in G sequential head-groups
  (48 = e.g. 2×24 or 3×16) so each lowered kernel stays under BEAM_UOPS_MAX (3000, `codegen/opt/search.py:84`) instead of
  the 54/54 BeamUopLimit wipeout → hand-coded fallback (~3× worse/unit on 48 heads). `GDN_HEAD_GROUPS` ContextVar
  (0=auto: 1 for ≤32 v-heads, split for 48+), byte-identical outputs required. CPU tests + DEV=NULL lowering checks in-task;
  BEAM + measurement on hardware = Fable.
- **T4.62b (Fable, GPU):** BEAM the grouped kernels on the pooled split, before/after prefill+decode on 3.8; fault protocol
  (fresh-search fault tally 2/3 — a 3rd ⇒ dedicated task).

## Track B — MTP speculative decode
- **T4.63 nextn head loading** (branch `task/T4.63-mtp-load`, worktree `tinygrad-t463`): `MTPHead` module (enorm, hnorm,
  eh_proj, inner FFNBlock built from main config, shared_head_norm; shares token_embd + output). Loader: when
  `nextn_predict_layers>0` and `MTP=1` (ContextVar), map `blk.<num_blocks>.*` into the head instead of dropping
  (today: excluded via `num_blocks = block_count - nextn_predict_layers`, `model.py:716`/`gguf.py:314`; the unused-weights
  warning). Placement: last block's device. Forward: `draft_next(hidden, tok) -> logits`. Unit tests with a tiny synthetic
  state dict (CPU); no generate() changes in-task.
- **T4.64 speculative generate loop** (after T4.63; branch `task/T4.64-spec-decode`): draft k=`SPEC_TOKENS` (default 3) by
  chaining the head on its own outputs; verify with ONE (k+1)-token chunk forward that returns per-position tokens
  (argmax path first); accept longest exact-match prefix; recurrent-state rollback v1 = checkpoint conv/recurrent/KV-len
  before verify, restore + re-forward accepted prefix on partial accept (correctness-first; the JIT shape set is bounded:
  chunk ∈ {1, k+1, m+1...}). Active when temperature==0 or `SPEC_GREEDY=1`. Parity gate: token-identical vs normal decode.
- **T4.65 serve + sampled acceptance** (after T4.64): `--mtp` flag, splice-cache compat, Leviathan accept/resample for
  temperature>0 (distribution-preservation test on a tiny vocab by simulation).
- **T4.66 (later, optional):** expose per-position recurrent states from the (grouped) scan to kill T4.64's re-forward.
- **B-final (Fable, GPU):** measure spec decode on 3.8 (and 35B for the record) at 4k/16k/64k context.

## Track C — UD-Q4 3.8 (Fable only)
Download `unsloth/Qwen3.8-27B-GGUF UD-Q4_K_XL` (17.56 GB, verify nextn present in header), bench ±MTP.

## Rules for agent tasks (all of A1/A2/B1…)
Sonnet 5, max reasoning. Worktrees off master `3d24ffd8c` only; NEVER touch `tinygrad-td5` (live server tree), any
tmux/`pooled` session, port 8081, launchctl, or any NV/METAL device — every python/pytest invocation carries `DEV=CPU`
(or `DEV=NULL` where specified); never run `test/device/test_hcq.py`. Gates per repo CLAUDE.md:
`CHECK_OOB=1 DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest <targets> -x -q` (serial),
mypy `tinygrad/`, ruff. Commit locally; no pushes, no PRs, no upstream anything. Blocked ⇒ write BLOCKED.md + stop.
