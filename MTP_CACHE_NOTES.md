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
