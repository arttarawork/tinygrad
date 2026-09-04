# T4.66i — the 66g "regression", and what the MTP draft block's KV cache should hold

## 1. Root cause of the main-output garbage on the real split (evidence)

**It is not 66g.** The corruption predates it and was masked by a length-only check:

- The stale-replay slice found by T4.66h (`speculative_generate`'s prefill tail: `tok_all[:, -1:, :]` and
  `prefill_result[1][:, -1:]` read on Tensors a *replayed* jit call returns — the same frozen objects the
  CAPTURE call produced, so `-1:` resolves against the capture's bound length). `warmup()`'s spec dummy
  (`speculative_generate([0])`, T4.66e) captures that key at length 1, so every later real prompt's anchor is
  the **position-0** prediction (`'system'` after `<|im_start|>`), and `h_last` is position 0's hidden state.
- Timeline on hardware: pre-66e rounds (no spec warmup) produced real essays (794/1060 chars per 192 tokens);
  every round since 66e printed "essay ok, len 264" for 192 tokens = ~1.4 chars/token = garbage
  (`'system\n!\n!…'`). The parent verified length, not content. 66g's "3 tokens then EOS" is the same garbage
  ending at an EOS earlier — 66g changed the draft stream, not the anchor.
- 66g's `delattr` itself is functionally clean on CPU: `extra/t466i_warmup_repro.py` (fresh / warmup+delattr /
  warmup-without-delattr / +`_cached_tokens` reset, single device and CPU:0/CPU:1, prompts starting with 0 or
  not) — every arm token-identical to fresh `generate()`; the MTP cache persists across draft calls and rebuilds
  as realized zeros (`/tmp`-probe: after two drafts `uop.op=RESHAPE realized=True`, written positions non-zero,
  untouched positions 0). `_function.__call__` re-traces the body every call and binds the CURRENT implicit
  buffers (tinygrad/function.py:36-68), so a recreated cache is never a stale-buffer hazard at that layer.
- ~~What CPU cannot show~~ CORRECTED by T4.66j (§5): CPU reproduces the stale replay shape exactly like the device;
  my earlier shape oracle passed only because this tiny model full-accepts during warmup, so its VERIFY/REDO key
  is captured at the full width and the stale length happens to equal the real one. Force the capture width
  (§5) and CPU shows the whole failure, including the emitted id 0.

Why 66g's delattr is still replaced (not reverted): delete-and-recreate frees a ~270 MB device buffer and
reallocates it; a recycled allocation is not zero-filled on METAL/NV, `freqs_cis` is rebuilt, and every graph/
precompiled CALL that captured the old buffer now sees a different object. Zeroing **in place**
(`cache.assign(zeros_like).realize()`) achieves 66g's intent with none of that exposure.

## 2. What positions < start_pos of `mtp_head.block`'s KV cache should contain

DeepSeek-style MTP (T4.63 loader: `eh_proj(concat(enorm(embed(tok_{t+1})), hnorm(h_t)))` → block →
`shared_head_norm` → shared output head) is trained with the nextn block attending over the FULL prefix:
its K/V at position t is the block applied to `[h_t, embed(tok_{t+1})]` for every t. Today:
- positions < start_pos: zeros (fresh) or the previous session's drafts — never the prompt;
- positions of REJECTED drafts after a partial accept: stale draft K/V, later attended to (causal window).
So the head drafts from `h` + the next-token embedding with attention over zeros/stale positions — the
acceptance ceiling is below the trained one, and it drifts with every partial accept.

Options:
(a) **Prefill the MTP block** alongside the main prefill and **rewrite accepted positions** after each accept:
    per prefill chunk `[sp, sp+nt)`: `x = eh_proj(cat(enorm(embed(t[sp+1:sp+nt+1])), hnorm(H_chunk)))`,
    `mtp_head.block(x, sp)` — `H_chunk` = the main model's per-position final hidden (the spec=True forward
    already returns it for the tail; the non-tail chunks would switch to spec=True too, same key). Cost: one
    extra block per prefill chunk ≈ 1/64 of the prefill (~1.5%) plus one `(m+1)`-position block call per accept.
    Complexity ≈ 40-60 lines in `speculative_generate` (+ the accept-time rewrite) — bounded, but its payoff
    (acceptance rate) is unmeasurable on the CPU harness and the hardware acceptance numbers to date are all
    contaminated by the anchor bug above. Do it AFTER a clean hardware baseline with the 66h/66i slices.
(b) Zeros (today): correct-by-rejection, ceiling unknown until (a) is measured against it.
(c) Cheaper middle: rewrite only the accepted positions (no prompt prefill) — removes the drift, keeps the
    prompt gap; half the code of (a) for an unknown fraction of the gain.

**Recommendation:** measure first. With the slices fixed, one hardware round gives the true acceptance
histogram for (b); if full chains are back near the pre-66e 81%, (a) is a nice-to-have; if not, implement (a)
(prefill + accept-time rewrite together — the rewrite alone (c) still leaves the prompt gap that dominates
early in a request). Expected impact math: tokens/iteration = 1 + E[accept]; at 826 ms/iteration, 2.76 →
3.6 accepted tokens is 3.3 → 4.4 tok/s; nothing in (a) changes the ~800 ms iteration cost, which is the real
gap to the 7 tok/s plain baseline.

## 3. Changes in this branch
- cherry-pick 471f9c234 (T4.66h): positive slices for the prefill-tail anchor and `h_last`.
- `redo_h[:, -1:]` → positive slice (the REDO sibling 66h flagged).
- 66g's `delattr` → in-place zero (same intent, no realloc/identity change); 66g's test updated accordingly.
- `TestWarmupSplitTokenIdentity`: warmup() then speculative_generate on a CPU:0/CPU:1 split must equal fresh
  `generate()` (guards the contract; cannot reproduce the device-only failure — stated in its docstring).
- `extra/t466i_warmup_repro.py`: the CPU matrix that clears 66g's delattr.

## 4. Hardware validation
Serve the branch tip with MTP: `POOLED_TREE=<this worktree> POOLED_MTP=1 POOLED_ENV="SPEC_STATS=1 SPEC_TRACE=1 SPEC_TOKENS=3 NV_DISPATCH_RING=64"`
(attempt-2 MTP map), then read the essay CONTENT (not its length): a real 192-token essay (~800 chars) with
`SPEC_STATS` accept_len_hist showing full chains again. If the anchor is still position 0, the replay-shape
mechanism has a second reader; if content is right but acceptance stays {2:…}, implement §2(a).

## 5. T4.66j — the second reader: VERIFY's `pad_to` (hardware evidence → CPU reproduction → fix)
Hardware after b..i (t466i @ 6ed6d20c2, METAL+NV, k=3): anchor correct ("Copper! A humble metal with a! Golden
hue, copper has been!! A! …", 474 chars / 191 tokens) but laced with id 0 ("!") at accept boundaries;
SPEC_STATS iters=92 emitted=191 avg_accept_len=2.08 hist={0:34, 1:18, 2:39, 3:1}; byte-identical across runs.

Mechanism (same as 66h, one reader further down): `verify_logits, verify_h` are the VERIFY/REDO jit key's frozen
capture-call Tensors. REDO on warmup()'s own first partial accept captures that key at width m+1 (1..k); the
buffer underneath is sized to the bound (verify_chunk = 4) and holds all n real positions of every later replay,
but the reported "toks" length stays bound to the capture width s. `argmax(-1).pad_to((1, verify_chunk))
.tolist()[0][:n]` padded FROM s, so verify_ids[i] == 0 (pad) for every i >= s: emitted as the bonus/correction
token (`accepted[-1] = verify_ids[m]`) and never equal to a draft, so m caps at s. The hardware histogram fits
s = 2 (m=2 → "[d0, d1, 0]" 39 times; the single m=3 needed a drafted 0). The sampled path's
`pad_to((1, verify_chunk, vocab))` had the identical hole (all-zero logits = uniform p past s).

CPU reproduction (`TestStaleReplayShape`, CPU:0/CPU:1 split, tiny GDN model): capture the (True, True, 4, True)
key at width 1 with two direct spec=True calls, then `speculative_generate(prompt, k=3)`:
pre-fix `[7, 1, 7, 1, 7, 1, 7, 0, 2, 2, 2, 2]` (id 0, then divergence: the wrong token was fed to the state)
vs fresh `generate()` `[7, 1, 7, 1, …]`. Directly on the replayed tensors: `pad_to` idiom `[10, 0, 0, 0]`,
truth `[10, 2, 7, 7]`; `[:, :4]` view / per-position slices / the full-buffer view all `[10, 2, 7, 7]`;
`h[:, 3:4]` == fresh, `h[:, -1:]` != fresh (the 66h mechanism, on CPU).

Every host read of a spec=True jit output in speculative_generate, audited:
| read | status |
|---|---|
| prefill tail `prefill_result[1][:, n_toks-1:n_toks]`, `tok_all[:, n_toks-1:n_toks]` | fixed 66h (positive, host-known) |
| VERIFY greedy `argmax(-1).pad_to(...).tolist()[0][:n]` | **fixed 66j**: `verify_logits[:, :n].argmax(-1).tolist()` + `assert len == n` |
| VERIFY sampled `pad_to((1, verify_chunk, vocab)).numpy()[0][:n]` | **fixed 66j**: `verify_logits[:, :n].numpy()[0]` |
| full accept `tok_last = verify_ids_tensor[:, m:m+1]` | now a slice of the concrete (1, n) ids (was: of an argmax whose loop ran to s) |
| full accept `verify_h[:, m:m+1]` | already safe (positive, m < bound) |
| REDO `redo_h[:, len-1:len]` | fixed 66i |
| DRAFT `dlogits[:, -1, :]`, `tok_last.cat(*draft_tensors)` | not jit outputs (mtp_head.draft is @function, re-traced per call; concrete (B,1,·)) |
| generate()'s own `out` | concrete (B, vocab)/(B,1) inside the captured graph — never affected (66h's second test) |

Why `[:, :n]` is safe: `_parse_view_index` clamps int slices against the Variable's **vmax** (the bound), not
its bound value, and with batch 1 the shrunk view's data offsets are `i*vocab + v` — "toks" never enters them.
Any read that derives a length/pad amount/negative index from the tensor's own shape is the unsafe class.

## 6. T4.66k — the acceptance gap (81% → 8% full chains) is NOT in the d..j code (CPU proof); hardware round requested
**Resolved by R1 (2026-09-04):** 66c (`6bf1b39c1`) under today's map/env/prompt reproduced the 81% ({0:6, 1:4, 2:3, 3:42}/55)
— on DEGENERATE output ("…marking the Copper, Coppe…", "copper, a copper, copper, copper…" from ~token 25). **The pre-66d
"reference" acceptance numbers (T4.66b's 43-44/53 full chains, avg 3.60) came from corrupted main-model output — the
capture-assign state path 66d reverted for cost was also wrong on the real split — and must not be cited as a baseline.**
The honest baseline is b..j's 8% full chains on correct text, with the nextn block attending over an all-zero prompt
context; the §2(a) MTP prefill is the fix (§7), not a regression repair.

Datum: b..j on hardware (correct text): hist {0:19, 1:47, 2:17, 3:7}/90 — d0 accepted 71/90 = 79%, d1|d0 = 24/71 = 34%,
d2|d1 = 7/24 = 29%. T4.66b on hardware (2026-09-01, TASKS.md): 43-44/53 full chains, avg 3.60 — d0 ≈ 89%, d1|d0 = 48/48,
d2|d1 ≈ 92%. The collapse is in the CHAINED positions; d0 is roughly intact.

What the d..j diff changes on the draft path, audited against 66c (`git diff 6bf1b39c1 HEAD -- tinygrad/llm/model.py`;
the tinygrad core is untouched between the two commits):
- the hidden fed to draft(): still forward(spec=True)'s `x.contiguous()` = the last block's PRE-output_norm residual,
  at position m of the verify chunk (`verify_h[:, m]` on full accept; `redo_h[:, m]` after CHECKPOINT+REDO on partial —
  66c read `verify_h[:, m]` off 66b's capture path; same position, same quantity);
- the draft chain: same `dtok, dpos = tok_last, start_pos`, same k distinct `draft_pos_i` Variables, same lazy merge,
  same `tok_last` (verify_ids[m]) and `start_pos += m+1` conventions; 66d only relocates the embedding lookup
  (device-local table, same values); 66e only narrows VERIFY/REDO's width (main-model output token-identical);
  66g/i zero the block's cache once after warmup (66c had no spec warmup, so its first request saw a fresh
  all-zero cache too); 66h/i/j fix host reads (anchor, h_last, verify ids) — all proven position-correct.

CPU evidence (`extra/t466k_draft_equiv.py`, tiny GDN+MTP model, same seed/weights, same prompts, 66c's model.py vs
HEAD's loaded side by side on the same core): emitted tokens EQUAL; every draft() call's (token, position) inputs
EQUAL (21/15/15 calls over 3 prompts); draft logits identical for the first iterations and ≤ 2.2e-5 apart
(|logits| ≈ 3e-2) once a partial accept has gone through REDO instead of 66b's capture-assign — fp noise.
Also checked: the merged lazy draft chain gives bit-identical logits and cache contents to realizing each draft
call (draft i+1 does see draft i's K/V — control: zeroing position P changes d1's logits by 6.8e-4), and the K/V
lands at the right positions.

Conclusion: no software mechanism in d..j for the chain collapse, on CPU. What remains is either (a) the two
measurements were not taken under the same conditions (device map → which device runs the MTP block / its precompiled
CALL kernels; prompt; request order → the block's cache holds a previous request's entries at [0, P)), or (b) a
device-only effect in a d..j-touched kernel (66d's device-local embedding gather, 66e's width-4 VERIFY/REDO key) that
CPU cannot show. Neither is decidable from here.

Minimal discriminating hardware round (R1): serve the LAST GOOD tree — `6bf1b39c1` (T4.66c = 66b + SPEC_TRACE; it runs
on the pooled map: greedy partial accepts keep `tok_last` on out_dev there, the 66f crash only arrived with 66e's host
rebuild) — under TODAY's exact map/env/prompt (`POOLED_MTP=1 POOLED_ENV="SPEC_STATS=1 SPEC_TRACE=1 SPEC_TOKENS=3
NV_DISPATCH_RING=64"`, attempt-2 map, ctx 65536), essay as the FIRST request after a fresh start, and read (i) the
essay content (must be real text — 66c has no spec warmup, so no anchor bug), (ii) `accept_len_hist`.
- ≥ ~80% full chains again ⇒ a device-only regression exists inside d..j. Then one more round on HEAD with two
  runtime toggles I would add (`SPEC_LEGACY_EMBED=1` restores 66d's cross-device lookup; `SPEC_VERIFY_CHUNK=32`
  restores 66e's shared width) discriminates d from e in a single serve.
- ~8% ⇒ the 81% was a conditions artifact (or its run differed in map/prompt/order), there is no code regression,
  and the correct next step is §2(a): prefill the MTP block over the prompt (the block currently attends over an
  all-zero prompt context — P zero keys each add exp(0) to every softmax denominator with a zero value behind them, so
  the attention branch is diluted by ~P/e^score: noticeable at an essay prompt, effectively gone at a 19k-token Hermes
  prompt, where the head runs on its residual path alone) and rewrite accepted positions after each accept. That lifts d0..d2 together; it is not a regression fix.

## 7. T4.66l — the MTP prefill (§2(a)) as built
Semantics (draft()'s own convention, kept): the MTP step whose inputs are (h_t, tok_{t+1}) lives in `mtp_head.block`'s
cache slot t+1; slot 0 (step −1) is never written. `MTPHead.prefill(owner, h, tok_next, start_pos, v_toks, v_pos)` fills T
consecutive slots [start_pos, start_pos+T) from h = the main model's pre-output_norm hidden at positions start_pos−1.. and
tok_next = the tokens at positions start_pos.. — draft() minus the head, T-wide. `draft()` and `prefill()` share
`_block_in` (embed → enorm/hnorm → eh_proj); test `test_prefill_matches_chained_draft` pins prefill's cache to what T
single-step draft() calls leave (rtol 1e-4).

How the prefill chunk loop feeds it: every prompt chunk now runs `spec=True` (before: the final chunk only), so each
chunk's replay returns the per-position hidden; for chunk [s, s+n) the MTP steps are t = s..min(s+n−1, P−2) — the last
prompt step needs tok_P, the anchor, so it stays draft()'s first call — i.e. `n_mtp = min(n, P−1−s)` steps, inputs
`prefill_result[1][:, :n_mtp]` (host-known width, T4.66h/j) and `tokens[s+1 : s+1+n_mtp]`, slots s+1... The call is
realized (dispatched) before the next chunk's replay of the same jit key overwrites that frozen output buffer; from_gguf
places mtp_head on the last block's device, so the device queue keeps that order without a host sync.

After each accept: the draft chain wrote slots start_pos..start_pos+k−1 from (h_last, chunk_ids[i]) — token-correct
for i ≤ m but with the anchor's hidden at every step — so slots start_pos+1..start_pos+m are rewritten from
`verify_h[:, :m]` (VERIFY's own per-position hidden on a full accept; REDO's — same jit key, same frozen object — on a
partial one) and `chunk_ids[1:m+1]`. Slot start_pos was already right; m == k also fills the bonus token's slot, which
no draft writes. One `prefill()` dispatch per accept, inside `state_assign_ms`.

Symbolic width: one `mtp_toks` Variable (bound = the prefill chunk width, ≥ k+1) and one `mtp_pos` slot Variable give ONE
precompiled block trace for every prompt-chunk width and every rewrite width (the T4.66 lesson: a plain int would retrace
per distinct value). Gotcha found on the way: a `pad_to` folded straight into a symbolic-range kernel is a gated load the
range simplifier rejects (`codegen/simplify.py mark_gated` → `.val` on the bound PARAM; the very first norm kernel), so h's
pad is materialized by its own concrete copy kernel first (`.contiguous()`, one small kernel per distinct width) and the
token row is built padded on the host — the main prefill's own `tokens.contiguous()` pattern.

State cache: `snapshot_state` now includes the MTP block's live cache slice [:pos] (`mtp_cache_kv`) and `restore_state`
writes it back — without it a cache hit would leave the draft head attending over whatever the previous request left.
One slot per restored request stays unfilled: `pos` itself (step pos−1 needs h_{pos−1}, which a cache hit never
recomputes) — one stale key among thousands; quality-only. Warmup's dummy prompt is [0, 0] (was [0]) so the prefill trace
is warmed before the first real request; the post-warmup cache zeroing is unchanged.

Cost (Qwen3.8-27B-class, per prompt token): one extra block forward (the MTP block: ~1/64 of the main model ≈ 1.6%) plus
the full per-position output projection on every chunk (was final-chunk only; D×vocab×2 ≈ 1.2 GFLOP/token vs ~54 for the
model ≈ 2%) — ~3–4% of prefill total; ponytail ceiling: a hidden-only spec mode (third jit key + its warmup) would
return the 2%. Memory: the (1, chunk, D) f32 pad buffer per chunk (0.5 MB at chunk 32) and, per state-cache snapshot, the
MTP block's slice (n_kv_heads × pos × head_dim × 2 × 2 B: ~78 MB at 19k tokens for an 8-head/128-dim block, ~+10% on
top of the main snapshot). Decode: one prefill() dispatch per accept (m ≤ k rows) — microseconds of device work.

Hardware validation: same invocation as before (`POOLED_MTP=1 POOLED_ENV="SPEC_STATS=1 SPEC_TRACE=1 SPEC_TOKENS=3
NV_DISPATCH_RING=64"`, attempt-2 map, ctx 65536, Qwen3.8-27B Q8) on this branch; read the essay CONTENT (must stay the
same correct text — the main model is untouched) and `accept_len_hist` (the draft head now sees the prompt: expect the
chained positions to recover; the 8% full-chain b..j figure is the baseline).
