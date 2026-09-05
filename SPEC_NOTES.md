# T4.64 — MTP speculative decoding: position ledger, rollback argument, JIT-shape notes

`Transformer.speculative_generate` (tinygrad/llm/model.py). Greedy-only, token-identical to
`generate(temperature=0.0)` by construction (standard speculative-decoding proof: verify always
feeds the draft tokens as real input, so `verify[i]` is the true greedy answer given everything
before it — see the ACCEPT step below). Draft quality never affects correctness, only how many
iterations it takes; see test_spec_decode.py's forced-mismatch/forced-perfect tests.

## 1. Position ledger

Notation: `P` = the position of `h_last` (pre-`output_norm` hidden state, main model) — the last
position whose processing is fully committed to the main model's state. `tok_last` = the token at
position `P+1` (already known/confirmed-real, not yet fed to the main model). The bookkeeping
variable `start_pos` always equals `P+1` (equivalently `len(tokens)-1`, once `tok_last` has been
appended — see the prefill paragraph below).

**Prefill** (mirrors `generate()`'s own chunked-prefill loop, not shared code with it — see the
method docstring): after the loop, main-model state reflects `tokens[0..prompt_len-1]`; `h_last` =
hidden at `prompt_len-1`; `tok_last` = the sampled token predicting `prompt_len`. `tok_last` is then
immediately appended + yielded — **this is exactly the token `generate()` itself yields first** (its
own last-prefill-chunk sample). Missing this was bug #1 found while building this: without it every
later position is off by one, since `tok_last` was being used only as the draft anchor, never as
output.

**Per outer iteration**, `k_eff = min(k, max_context-1-len(tokens))` (caps drafting so a full accept
can never push `len(tokens)` past `max_context`; degrades gracefully to `k_eff=0` → a single plain
decode step at the boundary, no special-casing needed):

| step | reads | writes | positions touched |
|---|---|---|---|
| DRAFT ×k_eff | `h_last` (unchanged each call — see below), chained `tok_ids`/`start_pos` | `mtp_head.block`'s own KV cache only | `start_pos .. start_pos+k_eff-1` (mtp-block-local) |
| VERIFY | `chunk_ids = [tok_last, d0..d_{k_eff-1}]` | main model: attention KV, GDN state (for every block), **and** (T4.66b) a per-position `state_track`/`conv_window` capture per GDN block, plus the full per-position hidden state | `start_pos .. start_pos+k_eff` |
| ACCEPT | `verify_ids` vs `chunk_ids[1:]` | host-side `m` (longest matching prefix, 0..k_eff) | — |
| FIXUP (only if `m<k_eff`) | this SAME VERIFY call's own `state_track`/`conv_window` captures, indexed at `m` | main model: GDN `conv_state`/`recurrent_state` for every block (a slice + `.assign()`, not a forward) | — (no re-forward, so no positions are touched a second time) |

(T4.66b removed the CHECKPOINT step and the REDO forward pass this table used to have here — a
device-side GDN-state clone every iteration, and a whole second main-model forward on every partial
accept — replacing both with FIXUP: reading the correct post-partial-accept state directly out of
data the VERIFY forward already computed, instead of rolling back and recomputing it. See §7.)

`verify_ids[i]` (i=0..k_eff) is the greedy id predicting the position *after* `chunk_ids[i]` was fed,
so it lines up with draft `d_i == chunk_ids[i+1]` for `i<k_eff`; `verify_ids[k_eff]` has no draft
counterpart — it's the unconditional bonus/correction token. `m` = longest prefix where
`chunk_ids[m+1] == verify_ids[m]`. Accepted = `chunk_ids[1:1+m] + [verify_ids[m]]`, always `m+1`
tokens, spanning positions `P+2 .. P+2+m`.

- **`m == k_eff` (full accept):** every fed token was real, so VERIFY's own state advance is already
  correct through `start_pos+k_eff`. `h_last := verify_h[:, m:m+1]` (VERIFY's hidden at position
  `start_pos+k_eff`), `tok_last := verify_tok[:, m:m+1]`, `start_pos += k_eff+1`. No FIXUP (nothing to
  fix — every GDN block's in-place state write from VERIFY's own forward is already correct).
- **`m < k_eff` (partial accept):** VERIFY's state now also reflects the wrong `d_m..d_{k_eff-1}` —
  see §2 for why attention KV is fine as-is but GDN state isn't. T4.66b (see §7): FIXUP reads
  `h_last` off `verify_h[:, m:m+1]` (same slice as the full-accept case above — this is now
  UNBRANCHED between the two) and, per GDN block, slices that SAME VERIFY call's own captured
  `state_track[:, :, m]` / `conv_window[:, m+1:m+1+conv_kernel-1]` and `.assign()`s them directly —
  no re-forward, no attention-KV rewrite (redundant even in the old REDO, per §2's argument).
  `start_pos += m+1`.

Both branches leave the invariant `start_pos == len(tokens)-1` intact (both grow by the same `m+1`).

DRAFT's own chaining: `h_last` is reused **unchanged** across all `k_eff` calls in one iteration —
`MTPHead.draft`'s fixed 4-arg signature (`owner, h, tok_ids, start_pos`, T4.63, not modified here)
gives no way to feed it a running hidden state instead, and correctness never depends on this choice
(see the forced-mismatch test). Only `tok_ids` (chained to the previous call's argmax) and
`start_pos` (incremented by 1) advance; `mtp_head.block`'s own KV cache — always a plain attention
block, never GatedDeltaNet, per `from_gguf`'s MTP branch — is what actually carries positional
continuity across the chain. T4.66 could thread the block's own pre-`shared_head_norm` output
between calls instead, for a better draft signal (quality only, not correctness).

## 2. Why attention KV (main model *and* MTP block) never needs a snapshot

Every attention write is `cache[..., start_pos:start_pos+T, :].store(...)`, and every read is
`cache[..., 0:start_pos+T, :]` gated by a causal mask that never lets query row `i` see key
`j > start_pos+i`. So a slot can only ever be read at or before the position it was written for. A
wrong slot written by a discarded draft (e.g. VERIFY writing `start_pos+m+1..start_pos+k_eff` with
`d_m..`) is only reachable by a *future* query at that position or later — and the very next thing
that happens (this iteration's REDO, or next iteration's DRAFT/VERIFY) always starts writing again
at the new, smaller `start_pos`, **before** anything ever legitimately queries that far again. Same
argument, unchanged, for `mtp_head.block`'s cache: DRAFT calls always write at
`start_pos..start_pos+k_eff-1` of *this* iteration, and a rolled-back next iteration starts drafting
from the new (smaller) `start_pos`, overwriting the stale tail before it's ever read.

GatedDeltaNet state is different: `conv_state`/`recurrent_state` are not position-indexed, they're a
single accumulator read-modify-written every call. Once VERIFY folds `d_m..d_{k_eff-1}` into it,
there's no slice to discard — hence the CHECKPOINT/restore, and why REDO must *recompute* (not just
overwrite) to get back to the truth.

## 3. The `.contiguous()` bug (found building this — worth flagging for T4.65/T4.66)

`forward()`'s `spec=True` branch returns `(per_position_ids, x[:, -1:])`. `x[:, -1:]` is a **zero-copy
view** into `x`'s own buffer at an offset depending on the bound `toks` variable. The default
(non-spec) path already slices this way, but only ever consumes it *inline*, in the same jit call
that produced it. `h_last` here is different: it's carried *out* of the call and read back one or
more replays later (by the next iteration's DRAFT). Without `.contiguous()`, that later read
silently comes back from whatever the intervening replay(s) left in `x`'s buffer instead of this
call's own hidden state — confirmed with a byte-level `cache_kv` diff between two otherwise-identical
runs (one calling `mtp_head.draft` between iterations, one not) that pinned corrupted cache entries
at positions the corrupted run's own iteration never even touched. Reproduces identically through
three unrelated consumption paths (`mtp_head.draft(h_last, ...)`, a bare `h_last.realize()`, and a
*separate* single-tensor `TinyJit` wrapping just `x[:, -1:]`) — so it's not specific to tuples or to
`MTPHead`. `tok_all` never had this problem: it's real computed output (norm → matmul → argmax), not
a view, so it already owns its buffer. Fix: materialize the view (`.contiguous()`) *inside* the
traced function, so it becomes its own tracked buffer in the same captured linear program that
`tok_all` already gets.

Lesson for anyone extending `spec=` further: **any jit output meant to be read after another replay
of the same symbolically-shaped graph must be a real owned buffer, never a bare slice of a larger
intermediate.**

## 4. Other JIT-shape notes for the real (GPU) target

- `v_start_pos`/`v_toks` are created once per `speculative_generate()` call and reused for prefill
  chunks, VERIFY, and REDO alike — every `spec=True` call, whatever its actual length (1..k+1 or
  1..chunk_size), shares one `(is_prefill=True, greedy=True, chunk_size, spec=True)` jit slot instead
  of one capture per distinct length (which would otherwise crash on replay with a shape mismatch —
  confirmed while debugging: routing a T=2 VERIFY and a T=1 REDO through two *different* concrete,
  unbound shapes collides in `self.jit`'s dict key exactly the way T4.12's own docstring warns about).
  `chunk_size` is auto-widened to `max(caller's chunk_size, k+1)` so VERIFY/REDO always fit.
- **(T4.66, done)** `MTPHead.draft`'s own `start_pos` is now a bound Variable, not a plain python int.
  `speculative_generate` creates one `v_draft_pos = UOp.variable("draft_pos", 0, max_context-1)` —
  same domain as `v_start_pos`, since `mtp_head.block`'s own `cache_kv`/`freqs_cis` are sized off the
  SAME `config.max_context` as the main model (`from_gguf`'s MTP branch builds `mtp_cfg` from the
  owning `Transformer`'s own config), and `k_eff` already caps drafted positions at
  `start_pos+k_eff <= max_context-2` — and `.bind(dpos)`s it fresh per drafted step, instead of
  passing `dpos` straight through. `TransformerBlock`/`MLATransformerBlock` already handled a bound
  `start_pos` end-to-end before this change (unlike `GatedDeltaNetBlock`, which self-converts — see
  its `_attention` — these two never needed to: every T=1 decode-shaped call already arrives with a
  bound Variable from `generate()`/`speculative_generate()`'s own main-model call sites), so the fix is
  entirely in the DRAFT loop's call site, not in `MTPHead.draft`'s signature (already `int|UOp`) or in
  either block class. A SEPARATE Variable from `v_start_pos`, not a reuse of it — see the reasoning
  written next to `v_draft_pos`'s definition in `speculative_generate` (reuse would likely be just as
  safe, since DRAFT always fully resolves before VERIFY ever binds `v_start_pos`, but a separate name
  costs nothing and never needs re-justifying if either call site changes later).

  One real wrinkle, not obvious up front: the greedy DRAFT loop chains `dtok` *lazily* across its
  `k_eff` iterations by design (batching the whole chain's host read into ONE final `.tolist()`
  alongside `tok_last`) — with `v_draft_pos` rebound to a DIFFERENT concrete value each iteration, all
  `k_eff` of those still-unrealized binds would land in the SAME combined schedule at that final
  `.tolist()`, and tinygrad only allows one concrete value per variable NAME per schedule
  (`schedule/__init__.py`'s `create_linear_with_vars`: `RuntimeError: bind mismatch on draft_pos,
  i != i+1`, reproduced while building this). Fix: `.realize()` each drafted `dtok` individually before
  the next iteration's `.bind()` of the same name exists — still async/non-blocking
  (`engine/realize.py`'s `run_linear(wait=False)`, same as `generate()`'s own per-step `.realize()`),
  so this doesn't add a host sync the sampled branch's per-step `.numpy()` didn't already have. The
  sampled branch never hit this: its per-step `.numpy()` already forced one bind resolved at a time.

  Measured (tiny synthetic model, `test_spec_decode.py::test_draft_reuses_schedule_across_positions`,
  `DEV=CPU`, greedy, `k=2`, verified by temporarily reverting the fix locally): pre-fix, tinygrad's
  `schedule_cache` grew by a steady +2 entries on EVERY subsequent token, forever (7 → 72 over 30
  tokens, never stabilizing); post-fix, it grows only through a short warmup (7 → 17 → 19) then stays
  exactly flat for the remaining 27 tokens — `to_program_cache` shows the same shape (31 → 241
  unbounded pre-fix, vs. 31 → 68 then flat post-fix).

## 5. What's still left (T4.66 did not touch this)

**(T4.66b, done — see §7)** The partial-accept REDO was the acknowledged v1 cost: one extra
main-model forward, purely to rebuild GDN state and get the correct `h_last`, on every iteration that
doesn't fully accept. This section used to sketch the fix as "expose the pre-norm hidden state at
every position" — that turned out to be only HALF of it (see §7): `h_last` alone doesn't need a
second forward to fix, but the GDN blocks' `conv_state`/`recurrent_state` do, since they're not
position-indexed (§2) — REDO existed for THEM, not for `h_last`. §7 covers both. (This section used to
also list binding `MTPHead.draft`'s `start_pos` to a Variable as a T4.66 candidate — done, see §4.)

What's left after T4.66b: the draft chain's `k` steps are still `k` separate python-level calls into
`mtp_head.block` (T4.66b removed the forced per-step `.realize()` — see §7 — but not the per-call
`_function(precompile=True)` graph-(re)construction overhead itself, which a single `TinyJit`-wrapped,
fixed-`k`, internally-unrolled draft-chain function could remove — sketched, not built, in §7's own
"not done" note).

## 6. T4.65 — logits-returning `forward(spec=True)`, sampled acceptance, serve wiring

**`forward(spec=True)` now returns `(per_position_logits, last_hidden)`**, not `(per_position_argmax_ids,
last_hidden)`. `per_position_logits` is `(B,T,vocab)` — `self.output(self.output_norm(x))` over the whole
(unsliced) `x`, unchanged from what T4.64 already computed there except the trailing `.argmax(-1)` is gone.
No `.contiguous()` needed on it (unlike `x[:, -1:]`, which still needs one — see §3): it's a fresh matmul
over the whole `x`, real computed output that owns its buffer, the same reason `tok_all` never needed one
either. `forward()` itself still only ever traces greedy here (`assert temperature is None` unchanged) —
sampled acceptance is pure host-side numpy over these logits, so there's no in-graph RNG to add regardless
of `speculative_generate`'s own `temperature`. `speculative_generate`'s greedy path (`temperature<=0`, the
default) now derives every id by `.argmax(-1)` on the returned logits *outside* `forward()` instead of
inside it — same computation, same values, one eager op later — so it stays token-identical to
`generate(temperature=0.0)` and to every existing T4.64 test, unchanged. Host-sync count per iteration is
unaffected: still exactly two `.tolist()` pulls (`chunk_ids` after DRAFT, `verify_ids` after VERIFY).

**Sampled acceptance** (`temperature>0`) implements Leviathan et al., "Fast Inference from Transformers via
Speculative Decoding" (2023). New module-level pure functions in `model.py`:

- `_softmax_np(logits:np.ndarray, temperature:float) -> np.ndarray` — numerically-stable softmax in
  float64 (host logits may be fp16/fp32; float64 gives `rng.choice`'s sum-to-1 tolerance headroom).
- `spec_accept(draft_ids:list[int], q_probs:np.ndarray, p_probs:np.ndarray, rng:np.random.Generator) ->
  tuple[list[int], int]` — `q_probs` is `(k_eff, vocab)` (draft softmax per drafted position), `p_probs` is
  `(k_eff+1, vocab)` (verify softmax per verify position, row `k_eff` being the bonus position). Accepts
  drafted token `i` with probability `min(1, p_i[d_i]/q_i[d_i])`; at the first rejection `m`, resamples from
  `normalize(max(0, p_m - q_m))`; on full accept, samples the bonus token from `p_{k_eff}`. Returns
  `(accepted_ids, m)` with `len(accepted_ids) == m+1`, mirroring the greedy path's `(accepted, m)` exactly —
  `chunk_ids[1:1+m]` is always `== draft_ids[:m]` regardless of which path produced `draft_ids`, so the
  post-ACCEPT bookkeeping (GDN checkpoint/restore, REDO, `start_pos += m+1`, the yield loop) is **shared,
  unbranched code** between the greedy and sampled paths — only DRAFT-id-production and the ACCEPT
  derivation of `(accepted, m)` itself differ. Full derivation and proof sketch are in `spec_accept`'s
  docstring; empirically verified by `test_spec_decode.py::TestSpecAccept.test_statistical_marginal_matches_p`
  (a seeded generator, 20000 trials on a 5-token vocab, `atol=0.02` — about 7σ even at the target
  distribution's largest entry, loose enough to never flake, tight enough to catch a wrong formula).

**Why DRAFT sampling is host-side per drafted token, not batched.** Speculative sampling's proof requires
`draft_ids[i]` to be an actual ancestral sample from `q_i`, not `argmax(q_i)` — sampling with the wrong
mechanism breaks the marginal-equals-`p` guarantee (§ above), so the greedy path's `.argmax(-1)` swap-out
doesn't carry over to the sampled path's draft chain. Sampling from a seeded `np.random.Generator` (required
for testability — a device-side Gumbel-max draw, like the one `Transformer.forward`'s own non-spec path
uses, would consume tinygrad's RNG stream instead, not this function's `rng` argument, defeating the point
of threading a seeded one through) means the draft chain pulls one `(vocab,)` logits vector to host *per
drafted token* instead of the greedy chain's single batched pull at the end. `# ponytail:` this trades
`k_eff` extra host round-trips per iteration (vs. greedy's 0 extra) for keeping sampling host-side, seedable,
and directly testable against `spec_accept`'s hand-computed cases — fine at `k`'s default (3) and the
`temperature>0` gate, revisit only if a real serving benchmark ever shows `--mtp` sampled mode is sync-bound.
`speculative_generate(temperature=..., rng=...)`: `temperature<=0` is greedy (byte-identical, as above);
`temperature>0` is sampled. `rng` defaults to a fresh `np.random.default_rng()` (materialized unconditionally
at the top of the call — negligible cost, one object per *request* not per token — so every branch can rely
on it being non-`None` without per-branch `assert`s or `Optional` plumbing). **Sampled output is
distribution-equal to `generate(temperature=t)` but not sequence-equal** to any one `generate()` run — it's
a different, independently-valid sample from the same distribution, verified statistically (above), not
reproduced token-for-token; `test_spec_decode.py`'s sampled end-to-end test therefore checks *state*
integrity (KV/GDN cache, `_cached_tokens`) via a from-scratch-model comparison, not token equality (see
`test_sampled_state_integrity_continues_like_a_fresh_generate`'s docstring).

**serve.py wiring.** `LLMServer.__init__` gains `mtp:bool=False, spec_k:int=SPEC_TOKENS` (`SPEC_TOKENS =
getenv("SPEC_TOKENS", 3)`, module-level in `serve.py`), stored as `self.mtp`/`self.spec_k`. `cli.py` gains a
`--mtp` store-true flag, threaded through as `LLMServer(..., mtp=args.mtp)` (`spec_k` uses `LLMServer`'s own
`SPEC_TOKENS`-derived default — no separate cli.py plumbing needed). `Handler.run_model`'s routing condition
(exact):

```python
use_spec = self.server.mtp and model.mtp_head is not None
gen = model.speculative_generate(ids, k=self.server.spec_k, temperature=temperature) if use_spec \
  else model.generate(ids, temperature=temperature)
```

`temperature` is already `float(body.get("temperature", 0.0))` from `do_POST`, so a request with no/zero
temperature takes `speculative_generate`'s greedy path and one with `temperature>0` takes its sampled path —
no separate greedy/sampled branch is needed in `serve.py` itself, `speculative_generate` already picks.
Absent `--mtp`, or present but `model.mtp_head is None`, `use_spec` is `False` and behavior is byte-identical
to pre-T4.65 serving (this is exactly what `test/null/test_llm_server_mtp.py`'s `TestLLMServerMTPFallback`
gates). The splice/`_cached_tokens` path (`splice_ids`, `self.server.last`) is untouched by any of this —
both generators mutate the same `tokens`/`self._cached_tokens` the same way (`speculative_generate` already
mirrored `generate()`'s bookkeeping exactly since T4.64), so which one produced a given turn's tokens is
invisible to it.

## 7. T4.66b — REDO-free partial accept, cheaper draft dispatch, SPEC_STATS

**Goal**: T4.66 fixed the draft chain's schedule-cache growth; §5's leftover was the partial-accept REDO — a
second, full main-model forward (attention + MoE + every GDN block) over the `m+1` confirmed tokens, paid on
every iteration that doesn't fully accept, purely to get the correct `h_last` and GDN state back. This section
removes it, cheapens the draft chain's dispatch count, and adds opt-in instrumentation to pick `k`.

**1. REDO removal — the state-exposure question.** The REDO existed for two things bundled into one forward:
`h_last` (the main model's pre-`output_norm` hidden at position `start_pos+m`) and each GDN block's
`conv_state`/`recurrent_state` as they'd be after only `m+1` tokens. These have very different answers:

- `h_last` needs no second forward at all: `forward()`'s `spec=True` path already computes the FULL
  per-position hidden `x` (every position, not just the last) before slicing — T4.65's own comment already
  noted "applying output_norm+output to every position ... is strictly more work, never less" but the codebase
  still only ever kept the LAST position's slice (`x[:, -1:]`). T4.66b keeps the whole thing (`x.contiguous()`,
  same `.contiguous()` reasoning as always — see §3) and lets `speculative_generate` slice `[:, m:m+1]` once
  `m` is known, host-side, after ACCEPT. Full accept and partial accept now share ONE `h_last` expression —
  `verify_h[:, m:m+1]` — where before they were two branches (`verify_h` vs a REDO's own output).
- GDN state is the genuinely hard part, and the actual crux of this task: `conv_state`/`recurrent_state` are
  O(1) accumulators, not position-indexed (§2), so there's no *slice* of them to take at position `m` the way
  there is for hidden/logits/KV — UNLESS the block's own scan is made to expose every intermediate state it
  passes through on the way to the final one. Investigated both scan implementations
  (`GatedDeltaNetBlock._attention`'s `GDN_SCAN_LOOP` else-branch and `gdn_scan_wy`'s chunked WY form):
  - **LOOP (the standing default — `GDN_SCAN_IMPL` auto-resolves to it, see T4.69a)**: the per-token python
    loop already computes `state` at every step `t` on its way to the final one — it just used to overwrite the
    python variable each iteration and let the intermediate values fall out of scope. Retaining them (one more
    `.stack()`, the same way `outs` already gets stacked) is genuinely free: same ops, same schedule, no new
    kernel shapes. `conv_state`'s equivalent-at-prefix-`p` value was ALREADY being computed as a byproduct too —
    the existing `conv_state_store` line's `conv_window[:, T:T+conv_kernel-1]` slice generalizes to
    `conv_window[:, p:p+conv_kernel-1]` for ANY prefix length `p`, since `conv_window` already holds every
    position's conv-projected row before the store ever happens. So for the actual serving path, the answer to
    "can the scan cheaply expose per-position state" is **yes, for free**.
  - **WY (`gdn_scan_wy`, off by default, not the serving path)**: NOT free. It only ever forms the chunk's
    LAST state via one `(V,K)` matmul (`u.transpose(-1,-2) @ k`, see the function's own `final_state` line) —
    exposing every prefix would need a cumsum-shaped computation of comparable cost to the one WY already
    replaced the loop with (i.e. real extra work, not retention of something already computed). Not
    implemented: `capture=True` forces the LOOP path regardless of `GDN_SCAN_IMPL`/device (see
    `GatedDeltaNetBlock._attention`'s own docstring) — correct on any build, and per T4.69b's own note (WY has
    "nothing to amortize" and measures pathologically slow at `T_pad==1`), forcing the loop for VERIFY's tiny
    `k_eff+1`-token chunk specifically is perf-neutral-or-better even on a WY-enabled build. Same reasoning
    excludes the AMD fused kernel (`gated_delta_prefill`, RDNA3-only, descoped for this fork — CLAUDE.md): one
    opaque hardware-loop kernel, nothing to retain from either.

**Plumbing this out costs threading, not compute.** `_attention` gains `capture:bool=False`
(`GatedDeltaNetBlock`-only — the base `FFNBlock`/`TransformerBlock`/`MLATransformerBlock` never see it),
returning `(out, state_track, conv_window)` instead of bare `out` when `True` — `state_track` is
`(B,H,T_pad,V,K)`, `state_track[:,:,t]` is this block's `recurrent_state` as it would be after only the first
`t+1` real rows (validated directly, at every `t`, in `test_gdn_scan_parity.py::TestGDNScanCapture`).
`GatedDeltaNetBlock` gets its own `__call__` override: `spec=False` (every decode/prefill call, i.e. the
entire non-speculative universe) delegates to `super().__call__` unchanged — byte-identical cost, nothing new
ever built; `spec=True` (VERIFY only) builds the capturing variant. `forward()`'s block loop special-cases
`isinstance(block, GatedDeltaNetBlock)` when `spec=True` to call it this way and collect `(state_track,
conv_window)` per GDN block; its `spec=True` return grows from `(logits, hidden)` to `(logits, hidden,
*per_gdn_block_captures)` — a flat tuple, kept per-block (not stacked into one tensor) specifically so blocks
living on DIFFERENT devices (this model's whole reason for existing — see CLAUDE.md's pooled-server device
map) never get an implicit cross-device copy just to be captured. **Why a side-channel instance attribute
wouldn't have worked**: `GatedDeltaNetBlock.__call__`'s `_run` closure is `@function(precompile=True)`-wrapped,
and (like the outer `TinyJit(self.forward)`, per §3's lesson) only what a jitted/precompiled closure actually
*returns* survives being read after another call to the same closure — an attribute stashed on `self` inside
`_attention` and read back later, outside that same call, has no guarantee of reflecting THIS call's data by
then. Every new value here is threaded through an actual return statement for exactly this reason.

**ACCEPT's FIXUP step, replacing CHECKPOINT+REDO**: no snapshot to take beforehand (nothing to roll back to —
see §1's table) and, on partial accept, a plain slice + `.assign()` per GDN block reading directly off THIS
verify call's own `gdn_extra`, not a second forward. Full accept needs no FIXUP at all (VERIFY's own in-place
write is already correct) — same fast path the old code had, just without a CHECKPOINT to have avoided
un-doing.

**2. Draft-chain dispatch.** The `k_eff` sequential `mtp_head.draft` calls each used to end in a forced
`.realize()` (T4.66): `v_draft_pos` was ONE Variable, rebound to a different concrete value every step, and a
single merged schedule can't carry two bound values for the same name (`"bind mismatch on draft_pos, i !=
i+1"`) — so each step had to be individually dispatched before the next `.bind()` of that name even existed.
T4.66b uses `k` DISTINCT Variables (`draft_pos_0..k-1`) instead of rebinding one — no name is ever bound twice
within one schedule, so the realize is gone and all `k_eff` steps merge into ONE schedule, dispatched by
`chunk_ids`' own batched `.tolist()` (same mechanic VERIFY/FIXUP already relied on). Cost: `k` (not 1)
warmed-once schedule-cache/`to_program_cache` entries for the draft chain instead of 1 — still bounded (`k`
defaults to 3), same "one entry-set per NAME" property T4.66 established, just `k` of them; verified this
doesn't regress by re-running `test_draft_reuses_schedule_across_positions` unchanged (still asserts zero
growth after warmup, just a larger flat plateau).

**Not done — a real jit capture for the whole chain.** The `.realize()` removal above cuts DISPATCH count
(`k_eff` async dispatches → 1), but each draft step is still a separate PYTHON-level call into
`mtp_head.block.__call__`'s `_function(precompile=True)` closure, which re-executes its python body (UOp
construction, `get_state_dict`, `graph_rewrite`) every single call — `_function` never caches across calls the
way `TinyJit` does. A `TinyJit`-wrapped, fixed-`k`, internally-unrolled draft-chain method (`for i in
range(k): ...` built ONCE into one captured graph, keyed on the constant `k` — analogous to how
`GatedDeltaNetBlock._attention`'s own per-token loop is already one python-unrolled chain inside ONE trace, and
structurally the same nesting `Transformer.__call__`'s `TinyJit(self.forward)` already does around many
`_function`-wrapped block calls) would remove that too — internal per-step positions would be `start_pos + i`
(constant-offset arithmetic on the chain's ONE bound Variable, not `k` separate binds, so no naming scheme
needed there either). Not implemented here: real, but it's a new capture mechanism to get right (especially
threading the greedy chain's per-step argmax feedback through one unrolled trace) rather than a
small, well-isolated change like the two items above, and `k_eff<k` (the max_context-boundary degrade, see
§1) would still need the existing per-step loop as a fallback. Left as a design, not attempted, per this
task's own priority order (item 1 > item 2 > item 3) and STOP condition for lower-priority items.

**3. SPEC_STATS.** A `ContextVar` (default `0`, module-level next to `MTP`/`GDN_CHUNK` etc.) — `0`: zero
overhead, not even a dict write, just one cheap `if` per outer iteration. `1`: `speculative_generate`
accumulates an accept-length histogram (`m -> iteration count`) and drafts-per-emitted-token, printed once via
a `try/finally` wrapping the main loop — fires whether the loop exhausts `max_context` or the generator is
closed/abandoned early (a request ending in `serve.py`, or a test calling `gen.close()` — both raise
`GeneratorExit` at the current `yield`, which the `finally` catches on its way out). This is the knob for
picking `k` on real hardware: a low average accept length for a given `k` means the draft head isn't good
enough to justify it (wasted DRAFT compute genuinely spent on tokens that get rejected); the histogram's shape
(e.g. mass concentrated at `m=0` vs spread toward `m=k`) shows which direction to move `k` in, and
drafts-per-token relates directly to the wall-clock trade this whole feature is making.

**Correctness.** Every existing token-identity/state-integrity test (T4.64/T4.65's greedy and sampled paths)
passes unchanged against the REDO-free path — none of them previously exercised a REAL `GatedDeltaNetBlock`
at all (`test_spec_decode.py`'s base synthetic model is `full_attention_interval=1`, i.e. zero GDN blocks, so
the old CHECKPOINT/REDO code ran as an always-empty no-op list there — the REAL correctness surface this task
touches was previously untested). New: `test_gdn_scan_parity.py::TestGDNScanCapture` checks `capture=True`'s
`state_track`/`conv_window` against sequential per-token ground truth at EVERY position (not just the final
one), plus that capturing changes no output/final-state, plus that it forces the loop under an explicit WY
context. `test_spec_decode.py` adds a `_load_gdn` model (a real `GatedDeltaNetBlock` + plain attention block,
built directly a la `test_gdn_scan_parity.py::make_block`, no GGUF needed) and
`test_forced_partial_accept_matches_generate_gdn`: a `fake_draft` engineered to be right about the first
drafted position of every chain and wrong about every later one, forcing a genuine mid-chain partial accept
(`0 < m < k_eff`) deterministically every iteration (confirmed via SPEC_STATS, not just inferred) — greedy
output still matches `generate()` exactly.
