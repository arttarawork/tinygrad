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
| CHECKPOINT | main model's GDN blocks' `conv_state`/`recurrent_state` | a device-side clone (no main-model write) | — |
| VERIFY | `chunk_ids = [tok_last, d0..d_{k_eff-1}]` | main model: attention KV **and** GDN state, for every block | `start_pos .. start_pos+k_eff` |
| ACCEPT | `verify_ids` vs `chunk_ids[1:]` | host-side `m` (longest matching prefix, 0..k_eff) | — |
| REDO (only if `m<k_eff`) | GDN snapshot restore, `chunk_ids[:m+1]` | main model: attention KV (redundant) + GDN state (real fix) | `start_pos .. start_pos+m` |

`verify_ids[i]` (i=0..k_eff) is the greedy id predicting the position *after* `chunk_ids[i]` was fed,
so it lines up with draft `d_i == chunk_ids[i+1]` for `i<k_eff`; `verify_ids[k_eff]` has no draft
counterpart — it's the unconditional bonus/correction token. `m` = longest prefix where
`chunk_ids[m+1] == verify_ids[m]`. Accepted = `chunk_ids[1:1+m] + [verify_ids[m]]`, always `m+1`
tokens, spanning positions `P+2 .. P+2+m`.

- **`m == k_eff` (full accept):** every fed token was real, so VERIFY's own state advance is already
  correct through `start_pos+k_eff`. `h_last := verify_h` (VERIFY's last-position hidden, i.e. at
  `start_pos+k_eff`), `tok_last := verify_tok[:, m:m+1]`, `start_pos += k_eff+1`. No REDO.
- **`m < k_eff` (partial accept):** VERIFY's state now also reflects the wrong `d_m..d_{k_eff-1}` —
  see §2 for why attention KV is fine as-is but GDN state must be rolled back. REDO re-forwards
  exactly the `m+1` confirmed-correct tokens as one chunk at the *same* `start_pos`, giving the
  correct `h_last` (its own last position, `start_pos+m`) and, redundantly but harmlessly, rewriting
  attention KV for those same `m+1` positions. `start_pos += m+1`.

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
- `MTPHead.draft`'s own `start_pos` is passed as a plain python int (`dpos`), matching
  `test_mtp_load.py`'s existing calling convention and `draft`'s fixed signature — not a bound
  Variable. `TransformerBlock`/`MLATransformerBlock` (unlike `GatedDeltaNetBlock`, which
  self-converts) don't turn a plain int into a symbolic position, so on a real GPU target every
  drafted position (which never repeats across a whole `speculative_generate()` run) gets its own
  literal-shaped trace through `mtp_head.block`'s `@function(precompile=True)` body — one
  schedule/codegen pass per drafted token, no reuse. Doesn't affect correctness or this task's tests;
  a real perf cost on GPU that T4.66 (or later) should fix by binding `dpos` to its own Variable, the
  same way the main model's decode path already does.

## 5. What T4.66 would remove

The partial-accept REDO (§1) is the acknowledged v1 cost: one extra main-model forward, purely to
rebuild GDN state and get the correct `h_last`, on every iteration that doesn't fully accept.
Removing it needs `forward()`'s `spec=True` path to also expose the pre-norm hidden state at *every*
position (not just the last), so ACCEPT can slice position `m`'s hidden directly out of VERIFY's own
output instead of re-forwarding — trading one extra forward for keeping `k_eff+1` hidden states alive
instead of 1 per VERIFY call. Also candidate for T4.66: binding `MTPHead.draft`'s `start_pos` to a
Variable (§4) to stop paying a fresh compile per drafted position on GPU.
