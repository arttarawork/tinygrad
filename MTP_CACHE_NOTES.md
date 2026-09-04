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
- What CPU cannot show: on CPU the replayed output's `-1:` resolves correctly (my shape oracle:
  `-1:` == `pos n-1` fresh and warmed); the device stack resolved it as position 0. The positive-slice fix is
  correct by construction on both, so CPU parity is not a reason to doubt it. **Hardware validation decides.**

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
