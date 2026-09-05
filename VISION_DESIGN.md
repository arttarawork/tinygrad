# T5 — Vision on the pooled server (Qwen3.6-35B-A3B's own encoder, in tinygrad)

Status: DESIGN, 2026-09-05. Owner: Fable (session 01DYtbLcGPNmdKoUZM47Qfav). Tasks T5.1–T5.5 below; branch per task off fork master
`2d38cada5` (T5.4 off `task/T4.80-lmstudio-shim` @ `826398dcf`, because it edits serve.py). Integration branch `task/T5-vision`.

## 0. Goal and non-goals

Hermes's `vision` auxiliary task (and any `image_url` chat request) is served by the standing pooled model instead of DeepSeek:
`tinygrad.llm --serve` accepts OpenAI content parts with images, runs Qwen3.6-35B-A3B's **own** vision encoder (the `mmproj-BF16.gguf`
next to the Q8_0 GGUF), and feeds the visual tokens into the existing chunked prefill. Text-only requests stay **byte-identical**
in numerics and in the graphs they replay (no perf regression on the standing path). Non-goals: video, multiple images per message
beyond what falls out naturally (support N images; no video tokens), DeepStack (this checkpoint has none), training.

## 1. Ground truth (fetched 2026-09-05)

**HF `config.json` (Qwen/Qwen3.6-35B-A3B, `model_type: qwen3_5_moe`)**
- text: hidden 2048, 40 layers, `full_attention_interval 4` (10 attention blocks), 16 q heads / 2 kv heads, head_dim 256,
  `partial_rotary_factor 0.25` ⇒ rope_dim 64 (32 frequency pairs), rope_theta 1e7, **`mrope_interleaved: true`,
  `mrope_section: [11, 11, 10]`** (t, h, w).
- vision: depth 27, hidden 1152, 16 heads (head_dim 72), MLP 4304, `patch_size 16`, `spatial_merge_size 2`,
  `temporal_patch_size 2`, `out_hidden_size 2048`, `num_position_embeddings 2304` (= 48×48 learned grid, interpolated),
  `hidden_act gelu_pytorch_tanh`, in_channels 3, **`deepstack_visual_indexes: []`**.
- special ids: `vision_start 248053`, `vision_end 248054`, `image_pad 248056` (video_pad 248057, unused).

**HF `preprocessor_config.json`**: `Qwen2VLImageProcessorFast` / `Qwen3VLProcessor`; `size.shortest_edge 65536` (= min_pixels),
`size.longest_edge 16777216` (= max_pixels; we cap lower, §3.1), patch 16, merge 2, temporal 2, mean = std = [0.5,0.5,0.5]
(rescale 1/255 then (x−0.5)/0.5), RGB, resample BICUBIC.

**`mmproj-BF16.gguf`** (unsloth, 902.8 MB, downloaded to `/Users/artur/models/qwen3.6-35b-a3b-q8/mmproj-BF16.gguf`), llama.cpp clip
layout, 334 tensors:
- `clip.projector_type = qwen3vl_merger`, `clip.vision.image_size 768`, patch 16, embedding_length 1152, ffn 4304, block_count 27,
  head_count 16, `layer_norm_epsilon 1e-6`, `use_gelu = True`, `spatial_merge_size 2`, `projection_dim 2048`,
  `is_deepstack_layers` all False, image_mean/std 0.5.
- `v.patch_embd.weight (1152,3,16,16)` + `v.patch_embd.weight.1 (1152,3,16,16)` (the Conv3d kernel [2,16,16] split into its two
  temporal slices, both **float32**) + `v.patch_embd.bias (1152)`; `v.position_embd.weight (2304,1152)` f32.
- per block N: `v.blk.N.attn_qkv.weight (3456,1152)` bf16 + bias f32 (fused q|k|v), `attn_out (1152,1152)` + bias, `ln1`, `ln2`
  (weight+bias, f32), `ffn_up (4304,1152)` + bias, `ffn_down (1152,4304)` + bias. Weights bf16, biases/norms f32.
- `v.post_ln.weight/bias (1152)` = the merger's LayerNorm (applied per patch BEFORE the 2×2 merge — verify against HF, §2.2).
- merger: `mm.0.weight (4608,4608)` + bias, `mm.2.weight (2048,4608)` + bias (names dedup'd in the dump; T5.2 prints the full list).

**transformers reference** (`src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`, main): patch embed = Conv3d(3→1152,
k=s=[2,16,16], bias) over input reshaped `(-1, 3, 2, 16, 16)`; learned pos-embed 48×48 bilinear-interpolated with
`align_corners=True` to the (h,w) patch grid; **plus** axial 2D RoPE (`VisionRotaryEmbedding(dim=head_dim//2=36)`, θ=10000, 18 freqs
for h and 18 for w, `cat([freq_h, freq_w])` then duplicated to 72 = full head_dim); pre-LN blocks (LN eps 1e-6): x += attn(ln1(x));
x += mlp(ln2(x)); attention = full attention over all patches of one image, fused `qkv` Linear with bias, `proj`; MLP fc1→gelu_tanh→fc2
with biases. Merger: `norm` (LN 1152, eps 1e-6) → view(-1, 4·1152) → fc1 (4608→4608) → **nn.GELU() (erf, exact)** → fc2 (4608→2048).
Text side `get_rope_index`: image tokens at text position p get (t,h,w) = (p, p+hi, p+wi) over the merged grid (gh/2 × gw/2, row-major);
the next text position = p + max(gh, gw)//2. Interleaved M-RoPE over the 32 frequency pairs: pair i uses axis t if i%3==0 (11 pairs),
h if i%3==1 (11), w if i%3==2 (10). For text-only tokens t=h=w ⇒ identical to the plain 1-D RoPE the fork applies today.

**Fork anchors** (`tinygrad/llm/model.py` @ master 2d38cada5): `precompute_freqs_cis` (L149, returns cos|sin, `(max_context, rope_dim)`),
`apply_rope` (L194, halves convention: pairs (i, i+dim/2)), attention blocks slice `self.freqs_cis[start_pos:start_pos+T]` at L441-442
(and the partial-rope variant at L519/L525); `Transformer.forward` (L985: `x = self.token_embd(tokens...)`, then `for block in self.blk:
x = block(x.to(block.device), start_pos)`), `__call__` (L1027, jit-key logic), `snapshot_state/restore_state` (L1240/1260),
`generate` (L1279: `v_start_pos`, `v_toks` Variables, `t[:, sp:sp+nt]` symbolic chunk slices, `get_start_pos` reuse), `from_gguf` (L1091).
`tinygrad/llm/gguf.py: gguf_load(path, device_map)` → `(kv, {name: Tensor})` lazily on the mapped device (loads bf16 fine).
venv has PIL 12.3, numpy 2.5, torch 2.9.1; **no** transformers (T5.2 may `pip install transformers` into the venv for its reference test;
the test must `skipUnless` it is importable so CI stays green).

## 2. Design

### 2.1 Data flow (one request with k images)
1. serve.py parses `messages[].content` parts: `text` parts concatenate; each `image_url` part (data: URL, or http(s) with a 20 MB cap and
   10 s timeout) is decoded with PIL, preprocessed (T5.1) into `patches (n_i·4, 1536) float32` + `grid (1, gh_i, gw_i)` (patch grid),
   and its bytes hashed (sha256).
2. Each image becomes the literal text `<|vision_start|><|image_pad|><|vision_end|>` in the message content before templating (the Qwen
   template treats content as text). After tokenizing the rendered prompt, every `image_pad` id (248056) is expanded to a run of
   `n_i = (gh_i/2)·(gw_i/2)` ids: the first min(8, n_i) ids of the run are **hash-derived valid vocab ids** (`int.from_bytes(sha256[4j:4j+4]) %
   vocab_size`, j=0..7), the rest are `image_pad`. Rationale: the ids at visual positions never reach the model as text (they are replaced by
   embeddings, §2.3), but they ARE the key for `get_start_pos`/`_cached_tokens` and the T4.67 state cache — so two prompts with different
   images must differ in ids, and prefix semantics must survive. `splice_ids` (rendered-text ↔ ids) does not understand expanded runs and
   simply falls back (no splice) for image prompts; the state cache still hits on the id prefix.
3. The ViT (T5.2) runs once per image on `VISION_DEVICE` (default: the device of `blk[0]`, i.e. METAL on the standing map, so the
   injected embeddings need no hop) → `embeds_i (n_i, 2048)`. All images' embeds concatenate into `E (1, N_img, 2048)`.
4. `Transformer.generate(ids, vision=VisionInput(...))` (T5.3) prefills in the usual 32-wide chunks. Chunks are fed through the
   **image-prompt graph** (§2.3) with per-position side inputs; decode uses the **standard decode graph** with a new `rope_start` Variable.

### 2.2 Vision encoder (T5.2, `tinygrad/llm/vision.py`)
`VisionEncoder.from_mmproj(path, device) -> VisionEncoder`; `encoder(patches: Tensor (n, 1536), grid: (t, gh, gw)) -> Tensor (n/4, 2048)`.
- patch embed: W = stack([w0, w1], axis=2).reshape(1152, 3·2·16·16) in (C, T, H, W) flatten order = the processor's patch vector order
  (T5.1 emits vectors in exactly that order); x = patches @ W.T + b.
- pos embed: `pos (2304,1152)` viewed 48×48; bilinear, align_corners=True to (gh, gw): idx_h = linspace(0, 47, gh) (floor/ceil + weights),
  same for w; 4-corner weighted sum; then reordered into **merge-window order** (`reshape(gh/2, 2, gw/2, 2, D).permute(0,2,1,3,4)`),
  because T5.1's patch sequence is merge-window-major. Added to x.
- 2-D RoPE per patch: hpos/wpos in merge-window order; inv_freq = 1/10000^(arange(0,36,2)/36) (18 freqs); angles = cat(h·inv, w·inv)
  (36) duplicated to 72; rotate q and k over the FULL head_dim 72 with the halves convention (reuse/adapt `apply_rope`).
- 27 blocks: pre-LN (eps 1e-6), fused qkv (+bias) → 16 heads × 72, full (non-causal) attention over the image's n patches, `attn_out`
  (+bias); MLP `ffn_up` → gelu (tanh approximation, HF `gelu_pytorch_tanh`) → `ffn_down`, biases. Compute in float32 (weights cast from bf16
  at load; float16 acceptable later if METAL wants it — parity test decides).
- merger: `v.post_ln` (LN, eps 1e-6, per patch) → reshape (n/4, 4608) (the 4 patches of each 2×2 window are consecutive in
  merge-window order) → `mm.0` → **exact GELU (erf)** → `mm.2` → (n/4, 2048). (GGUF says `use_gelu`; HF's merger uses `nn.GELU()`; blocks
  use tanh. If llama.cpp differs, follow HF and note it.)
- Tests: tiny random config on CPU for shapes/ordering; **reference parity**: build transformers' `Qwen3_5MoeVisionModel` from
  `Qwen3_5MoeVisionConfig` with the config.json values, `load_state_dict` from the GGUF tensors (name map: `v.blk.N.attn_qkv` →
  `blocks.N.attn.qkv`, `attn_out` → `attn.proj`, `ln1/ln2` → `norm1/norm2`, `ffn_up/ffn_down` → `mlp.linear_fc1/linear_fc2`,
  `v.patch_embd.weight{,.1}` → `patch_embed.proj.weight` (stack on the temporal axis), `v.position_embd` → `pos_embed.weight`, `v.post_ln`
  → `merger.norm`, `mm.0/mm.2` → `merger.linear_fc1/linear_fc2`), run both on a real PIL image (a rendered-text PNG generated in the
  test), compare `embeds`: expect max |Δ| ≲ 1e-2 with cosine > 0.999 in float32. `skipUnless(transformers importable)`.

### 2.3 Model integration (T5.3, `tinygrad/llm/model.py`)
- **`rope_start` Variable.** Attention blocks (and the partial-rope variant) take `rope_start` in addition to `start_pos` and slice
  `self.freqs_cis[rope_start:rope_start+T]`; `start_pos` keeps indexing the KV cache. `Transformer.forward/__call__` gain
  `rope_start: int|UOp|None = None` (None ⇒ `start_pos`, so every existing caller is unchanged). `generate` binds a new
  `UOp.variable("rope_start", 0, max_context-1)` = `start_pos − rope_delta` on every chunk and decode step, where `rope_delta` (int,
  per sequence) = Σ over images seen so far of `n_i − max(gh_i, gw_i)//2`. Text-only ⇒ rope_delta 0 ⇒ rope_start == start_pos ⇒
  **numerically identical** and the same graph shape (one more bound Variable; the warmup/state-cache captures must bind it).
  `rope_delta` is stored in `snapshot_state()` and restored with it; `_cached_tokens` reuse keeps it consistent (it is a function of the
  cached prefix — recompute from `VisionInput` spans that lie inside the reused prefix).
- **Image-prompt graph.** When a request carries images, ALL prompt chunks of that request go through a second forward variant keyed
  `vision=True` in `__call__`'s jit-key logic, with three extra per-position side inputs sliced exactly like `t[:, sp:sp+nt]`:
  `E (1, P, 2048)` visual embeddings placed at their positions (zeros elsewhere; P = prompt length padded to a chunk multiple; fp16 or fp32,
  T5.3 decides — ~8 MB per 1k visual tokens), `M (1, P)` bool mask (True at visual positions), `POS3 (1, P, 3)` int32 rope positions
  (t,h,w) per prompt position (text positions get (p,p,p) with p = index − rope_delta-so-far; image runs get (p, p+hi, p+wi)).
  Forward: `x = M.where(E, token_embd(ids)).float()`; attention rope for these chunks = per-token, per-channel gather:
  `F_a = self.freqs_cis[POS3[..., a]]` for a in (t,h,w) (each (T, 64) = cos(32)|sin(32)), combined with a constant channel-axis mask
  (`axis[c] = c % 32 % 3` → 0/1/2 for both the cos and sin halves) ⇒ `F = where(axis==0, F_t, where(axis==1, F_h, F_w))`, then the
  normal `apply_rope`. Only attention blocks change; GDN blocks ignore positions. No DeepStack. Decode after the prompt uses the standard
  decode graph (rope_start Variable) — the image tokens live in the KV cache like any others.
- API: `Transformer.generate(tokens, ..., vision: VisionInput|None = None)`; `VisionInput(spans: list[tuple[int, int, tuple[int,int,int]]],
  embeds: Tensor (N_img_total, 2048))` with spans = (offset_in_ids, n_tokens, (t, gh, gw)) in id order. `generate` builds E/M/POS3 and
  rope_delta from it. `warmup()` gains an optional pre-capture of the image-prompt graph (a 2-chunk dummy with one 4-token span) behind
  the same switch that enables vision, so the first real image request does not pay a cold compile on the serving path.
- Tests (CPU, tiny config with ≥1 attention block and ≥1 GDN block, as the existing suites do): text-only outputs byte-identical to
  master (same ids in, same ids out; also same for a state-cache snapshot round trip); `pos3`/`rope_delta` builder vs hand-computed HF
  rule for (a) one image, (b) two images, (c) an image straddling a chunk boundary; chunked image prefill == one-shot (T=full) image prefill
  on a fresh model (the fork's existing "chunked matches one-shot" pattern); decode after an image uses rope_start = start_pos − delta
  (assert via the bound Variable value or by equality with an explicit-position reference forward); snapshot/restore after an image
  carries rope_delta.

### 2.4 Server + CLI (T5.4, `tinygrad/llm/serve.py`, `cli.py`; base = T4.80 branch)
- `--mmproj PATH` (cli.py) loads `VisionEncoder` on `VISION_DEVICE` (env; default blk[0].device) and stores it on `LLMServer`; without
  it, `image_url` parts → HTTP 400 `{"error": {"code": "vision_unavailable"}}`. `VISION_MAX_PIXELS` (env, default 1_003_520 ≈ 1 M px ⇒
  ≤ ~980 visual tokens) and min 65536 feed T5.1's `smart_resize`.
- Content parts → placeholder text + per-image (patches, grid, sha256) as in §2.1; pad-run expansion with hash ids after tokenization;
  `VisionInput` assembled and passed to `generate`; the `in:`/`prefill:` stderr line gains `img:{k}/{N_img_tokens}`.
- `find_snapshot/store_snapshot` unchanged (ids already carry the image identity). `splice_ids` is skipped for image prompts.
- `/v1/models` unchanged; `lmstudio_models_payload` gains `"capabilities": {..., "vision": true}` only when an encoder is loaded.
- Tests with a stub model/encoder: pad expansion math (n from grid), hash-id placement, multi-image ordering, data-URL decoding,
  400 paths (bad image, oversized, no encoder), text-only requests byte-identical to today's request body handling.

### 2.5 Hardware validation (T5.5, Fable — the standing safety protocol applies: never kill NV; SIGTERM via pooled-serve.sh only)
1. Reference on the Mac alone (pooled server stopped for the window): brew llama.cpp b10250 (`--mmproj` supported), the Q4_K_XL MTP GGUF
   (22.85 GB, already on disk) + this mmproj, `llama-server --mmproj ... -c 8192`, a fixed set of PIL-generated test images (rendered text for
   OCR; counted colored shapes; a chart) + one photo; record greedy answers.
2. Pooled Q8 @192k + `--mmproj` (serving tree at the T5 integration commit): same images/prompts; read CONTENT (OCR text must match,
   counts must match); then compare with the llama.cpp answers (different LLM quant ⇒ near, not identical, text). Then: faults=0,
   text-only regression battery (essay token-identical to the standing server), decode/prefill tok/s unchanged, ViT time per image.
3. Hermes wiring (NEEDS ARTUR'S YES — standing behavior): `auxiliary.vision` → the pooled server (provider lmstudio or the custom
   entry, thinking off), `download_timeout`/`timeout` tuned; DeepSeek vision-exp kept as the fallback.

## 3. Sizing, risks, open questions
- 3.1 Cost per image: ViT 400M params over n patches (1 M px ⇒ 3920 patches ⇒ ~3 TFLOP) ≈ 0.5–1 s on METAL; LLM prefill of ~980 visual
  tokens at ~95 tok/s ≈ 10 s. Memory: encoder ~0.9 GB (bf16 → fp32 doubles to 1.8 GB; keep fp16 if METAL numerics allow) on METAL's
  ~20 GB share (36 GB unified) — fine; NV is untouched (never risk the T4.77 ceiling for this).
- 3.2 Risk: the first image request compiles the image-prompt graph family (≈ the text prefill family plus small ops) — on METAL a few
  minutes from cold, cached thereafter; warmup pre-capture removes it from the request path. BEAM settings apply as usual.
- 3.3 Risk: `CHECK_OOB` z3 proof for the gather-by-position rope (indices come from an int32 input tensor; bound by clamping POS3 to
  [0, max_context−1] at build time — document in the kernel comments).
- 3.4 Open: fp16 vs fp32 for E and the encoder on METAL (parity test picks); exact GELU variants (HF says tanh in blocks, erf in merger).
- 3.5 Interaction with T4.66 MTP (off) and WY (off by default): none required; both stay off.
