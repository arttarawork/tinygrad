import json, math, pathlib, tempfile, time, unittest
import numpy as np
import gguf
from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.llm.model import Transformer, TransformerConfig, ExpertGating, precompute_freqs_cis
from tinygrad.llm.cli import SimpleTokenizer

# ---------------------------------------------------------------------------------------------
# Tiny synthetic gpt-oss GGUF, built with the `gguf` package (no downloads). Exercises every
# gpt-oss-specific mechanism end to end through Transformer.from_gguf:
#   - attention sinks (learned per-head logit added only to the softmax denominator)
#   - alternating sliding-window (layer 0) / full (layer 1) attention
#   - grouped MoE experts with gpt-oss's gguf tensor names + router/expert biases
#   - YaRN rope scaling (gpt-oss.rope.scaling.type=yarn)
#   - clamped swiglu (gate*sigmoid(alpha*gate)*(up+1), both branches clamped to +/-limit)
#   - MXFP4 quantization on one expert tensor (blk.0.ffn_down_exps.weight)
# and cross-checks the result against an independent numpy forward pass built from the same
# formulas (verified against transformers' _compute_yarn_parameters and examples/mlperf/models/gpt_oss.py).
# ---------------------------------------------------------------------------------------------

DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, HIDDEN = 16, 4, 2, 8, 32
N_EXPERTS, EXPERTS_PER_TOK, N_LAYERS = 4, 2, 2
VOCAB, MAX_CTX, T = 12, 12, 6
SLIDING_WINDOW, NORM_EPS = 3, 1e-5
ROPE_THETA, YARN_FACTOR, YARN_ORIG_CTX, YARN_BETA_FAST, YARN_BETA_SLOW = 10.0, 4.0, 32, 32.0, 1.0
SWIGLU_LIMIT, SWIGLU_ALPHA = 7.0, 1.702
TOKENS = [0, 3, 5, 7, 2, 9]

def q16(a: np.ndarray) -> np.ndarray:
  """round-trip through float16, like Transformer.from_gguf's default state_dict cast"""
  return a.astype(np.float16).astype(np.float64)

def build_gguf(path: pathlib.Path, rng: np.random.Generator, max_ctx: int = MAX_CTX) -> dict:
  """Writes a tiny gpt-oss-arch GGUF and returns the (already fp16-rounded) numpy weights used,
  keyed the same way the reference model below expects them."""
  w = gguf.GGUFWriter(str(path), "gpt-oss")
  w.add_context_length(max_ctx)
  w.add_embedding_length(DIM)
  w.add_block_count(N_LAYERS)
  w.add_head_count(N_HEADS)
  w.add_head_count_kv(N_KV_HEADS)
  w.add_key_length(HEAD_DIM)
  w.add_value_length(HEAD_DIM)
  w.add_layer_norm_rms_eps(NORM_EPS)
  w.add_expert_count(N_EXPERTS)
  w.add_expert_used_count(EXPERTS_PER_TOK)
  w.add_expert_feed_forward_length(HIDDEN)
  w.add_sliding_window(SLIDING_WINDOW)
  w.add_rope_freq_base(ROPE_THETA)
  w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
  w.add_rope_scaling_factor(YARN_FACTOR)
  w.add_rope_scaling_orig_ctx_len(YARN_ORIG_CTX)
  w.add_rope_scaling_yarn_beta_fast(YARN_BETA_FAST)
  w.add_rope_scaling_yarn_beta_slow(YARN_BETA_SLOW)
  w.add_token_list([f"<t{i}>" for i in range(VOCAB)])

  weights: dict = {}
  def norm(name:str, dim:int):
    a = rng.normal(1.0, 0.1, dim).astype(np.float32)
    w.add_tensor(name, a.astype(np.float16))
    weights[name] = q16(a)
  def lin(name:str, out_f:int, in_f:int, bias:bool, scale:float=0.3):
    wt = (rng.normal(0, scale, (out_f, in_f))).astype(np.float32)
    w.add_tensor(f"{name}.weight", wt.astype(np.float16))
    weights[f"{name}.weight"] = q16(wt)
    if bias:
      b = rng.normal(0, scale, out_f).astype(np.float32)
      w.add_tensor(f"{name}.bias", b.astype(np.float16))
      weights[f"{name}.bias"] = q16(b)

  for i in range(N_LAYERS):
    p = f"blk.{i}"
    norm(f"{p}.attn_norm.weight", DIM)
    norm(f"{p}.post_attention_norm.weight", DIM)  # remapped to ffn_norm by from_gguf for gpt-oss
    lin(f"{p}.attn_q", N_HEADS*HEAD_DIM, DIM, bias=True)
    lin(f"{p}.attn_k", N_KV_HEADS*HEAD_DIM, DIM, bias=True)
    lin(f"{p}.attn_v", N_KV_HEADS*HEAD_DIM, DIM, bias=True)
    lin(f"{p}.attn_output", DIM, N_HEADS*HEAD_DIM, bias=True)
    sinks = rng.normal(0, 0.5, N_HEADS).astype(np.float32)
    w.add_tensor(f"{p}.attn_sinks.weight", sinks.astype(np.float16))
    weights[f"{p}.attn_sinks.weight"] = q16(sinks)

    # router: big separated per-expert bias so expert selection is stable (no near-ties to trip
    # over tinygrad's pairwise_topk tie-break rule, which isn't what this test is validating -
    # test_llm_moe.py already covers that in isolation).
    gate_inp_w = rng.normal(0, 0.05, (N_EXPERTS, DIM)).astype(np.float32)
    gate_inp_b = (np.array([-6., -2., 2., 6.], dtype=np.float32) + i)
    w.add_tensor(f"{p}.ffn_gate_inp.weight", gate_inp_w.astype(np.float16))
    w.add_tensor(f"{p}.ffn_gate_inp.bias", gate_inp_b.astype(np.float16))
    weights[f"{p}.ffn_gate_inp.weight"], weights[f"{p}.ffn_gate_inp.bias"] = q16(gate_inp_w), q16(gate_inp_b)

    for name, out_f, in_f in ((f"{p}.ffn_gate_exps", HIDDEN, DIM), (f"{p}.ffn_up_exps", HIDDEN, DIM)):
      wt = rng.normal(0, 0.3, (N_EXPERTS, out_f, in_f)).astype(np.float32)
      b = rng.normal(0, 0.3, (N_EXPERTS, out_f)).astype(np.float32)
      w.add_tensor(f"{name}.weight", wt.astype(np.float16))
      w.add_tensor(f"{name}.bias", b.astype(np.float16))
      weights[f"{name}.weight"], weights[f"{name}.bias"] = q16(wt), q16(b)

    down_wt = rng.normal(0, 0.3, (N_EXPERTS, DIM, HIDDEN)).astype(np.float32)
    down_b = rng.normal(0, 0.3, (N_EXPERTS, DIM)).astype(np.float32)
    if i == 0:
      # exercise the MXFP4 path for at least one tensor
      packed = gguf.quantize(down_wt, gguf.GGMLQuantizationType.MXFP4)
      w.add_tensor(f"{p}.ffn_down_exps.weight", packed, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
      down_true = gguf.dequantize(packed, gguf.GGMLQuantizationType.MXFP4).reshape(down_wt.shape)
    else:
      w.add_tensor(f"{p}.ffn_down_exps.weight", down_wt.astype(np.float16))
      down_true = down_wt
    w.add_tensor(f"{p}.ffn_down_exps.bias", down_b.astype(np.float16))
    weights[f"{p}.ffn_down_exps.weight"], weights[f"{p}.ffn_down_exps.bias"] = q16(down_true), q16(down_b)

  tok_embd = rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float32)
  out_norm = rng.normal(1.0, 0.1, DIM).astype(np.float32)
  out_w = rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float32)
  w.add_tensor("token_embd.weight", tok_embd.astype(np.float16))
  w.add_tensor("output_norm.weight", out_norm.astype(np.float16))
  w.add_tensor("output.weight", out_w.astype(np.float16))
  weights["token_embd.weight"], weights["output_norm.weight"], weights["output.weight"] = q16(tok_embd), q16(out_norm), q16(out_w)

  w.write_header_to_file()
  w.write_kv_data_to_file()
  w.write_tensors_to_file()
  w.close()
  return weights

# --------------------------------- independent numpy reference ---------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
  return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * weight

def yarn_cos_sin(dim:int, positions:np.ndarray, theta:float, factor:float, orig_ctx:int, beta_fast:float, beta_slow:float):
  inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
  def find_dim(num_rot:float) -> float: return (dim * math.log(orig_ctx / (num_rot * 2 * math.pi))) / (2 * math.log(theta))
  low, high = max(math.floor(find_dim(beta_fast)), 0), min(math.ceil(find_dim(beta_slow)), dim // 2 - 1)
  if low == high: high += 0.001
  extrap_factor = 1.0 - np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
  inv_freq = inv_freq / factor * (1 - extrap_factor) + inv_freq * extrap_factor
  attn_scale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
  freqs = positions[:, None] * inv_freq[None, :]
  return np.cos(freqs) * attn_scale, np.sin(freqs) * attn_scale  # each (T, dim/2)

def apply_rope_np(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
  # x: (heads, T, head_dim); cos/sin: (T, head_dim/2)
  d = x.shape[-1]
  x1, x2 = x[..., :d // 2], x[..., d // 2:]
  return np.concatenate([x1 * cos[None] - x2 * sin[None], x2 * cos[None] + x1 * sin[None]], axis=-1)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def reference_forward(weights: dict, tokens: list[int]) -> np.ndarray:
  T_ = len(tokens)
  x = weights["token_embd.weight"][tokens]  # (T, dim)
  cos, sin = yarn_cos_sin(HEAD_DIM, np.arange(T_), ROPE_THETA, YARN_FACTOR, YARN_ORIG_CTX, YARN_BETA_FAST, YARN_BETA_SLOW)
  causal = np.triu(np.full((T_, T_), -np.inf), k=1)  # mask[i,j] = -inf if j > i

  for i in range(N_LAYERS):
    p = f"blk.{i}"
    window = SLIDING_WINDOW if i % 2 == 0 else 0

    xn = rms_norm(x, weights[f"{p}.attn_norm.weight"], NORM_EPS)
    q = (xn @ weights[f"{p}.attn_q.weight"].T + weights[f"{p}.attn_q.bias"]).reshape(T_, N_HEADS, HEAD_DIM)
    k = (xn @ weights[f"{p}.attn_k.weight"].T + weights[f"{p}.attn_k.bias"]).reshape(T_, N_KV_HEADS, HEAD_DIM)
    v = (xn @ weights[f"{p}.attn_v.weight"].T + weights[f"{p}.attn_v.bias"]).reshape(T_, N_KV_HEADS, HEAD_DIM)
    q, k = q.transpose(1, 0, 2), k.transpose(1, 0, 2)  # (heads, T, head_dim)
    v = v.transpose(1, 0, 2)
    q, k = apply_rope_np(q, cos, sin), apply_rope_np(k, cos, sin)

    KV, R = N_KV_HEADS, N_HEADS // N_KV_HEADS
    qg = q.reshape(KV, R, T_, HEAD_DIM)
    kg, vg = k[:, None], v[:, None]  # (KV,1,T,head_dim)
    scores = (qg @ kg.transpose(0, 1, 3, 2)) / math.sqrt(HEAD_DIM)  # (KV,R,T,T)
    mask = causal.copy()
    if window: mask = mask + np.tril(np.full((T_, T_), -np.inf), k=-window)
    scores = scores + mask[None, None]
    sink = weights[f"{p}.attn_sinks.weight"].reshape(KV, R, 1, 1)
    m = np.maximum(scores.max(-1, keepdims=True), sink)
    e = np.exp(scores - m)
    probs = e / (e.sum(-1, keepdims=True) + np.exp(sink - m))
    attn = (probs @ vg).reshape(N_HEADS, T_, HEAD_DIM).transpose(1, 0, 2).reshape(T_, N_HEADS * HEAD_DIM)
    x = x + (attn @ weights[f"{p}.attn_output.weight"].T + weights[f"{p}.attn_output.bias"])

    hn = rms_norm(x, weights[f"{p}.post_attention_norm.weight"], NORM_EPS)
    logits = hn @ weights[f"{p}.ffn_gate_inp.weight"].T + weights[f"{p}.ffn_gate_inp.bias"]  # (T, n_experts)
    ffn_out = np.zeros_like(x)
    for t in range(T_):
      sel = np.argsort(logits[t])[-EXPERTS_PER_TOK:]
      sel_logits = logits[t, sel]
      sm = np.exp(sel_logits - sel_logits.max())
      w_sel = sm / sm.sum()
      for j, e_idx in enumerate(sel):
        gate = hn[t] @ weights[f"{p}.ffn_gate_exps.weight"][e_idx].T + weights[f"{p}.ffn_gate_exps.bias"][e_idx]
        up = hn[t] @ weights[f"{p}.ffn_up_exps.weight"][e_idx].T + weights[f"{p}.ffn_up_exps.bias"][e_idx]
        gate, up = np.minimum(gate, SWIGLU_LIMIT), np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        act = gate * sigmoid(SWIGLU_ALPHA * gate) * (up + 1)
        down = act @ weights[f"{p}.ffn_down_exps.weight"][e_idx].T + weights[f"{p}.ffn_down_exps.bias"][e_idx]
        ffn_out[t] += w_sel[j] * down
    x = x + ffn_out

  xn = rms_norm(x, weights["output_norm.weight"], NORM_EPS)
  return xn @ weights["output.weight"].T  # (T, vocab)

class TestGPTOSSGGUF(unittest.TestCase):
  def test_gptoss_forward_matches_reference(self):
    rng = np.random.default_rng(1234)
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "tiny-gpt-oss.gguf"
      weights = build_gguf(path, rng)
      model, kv = Transformer.from_gguf(path)

      self.assertEqual(kv["general.architecture"], "gpt-oss")
      cfg = model.blk[0].config
      self.assertTrue(cfg.attn_sinks and cfg.clamp_swiglu and cfg.router_bias and cfg.moe_bias and cfg.attn_out_bias)
      self.assertEqual(model.blk[0].config.sliding_window, SLIDING_WINDOW)  # layer 0: sliding
      self.assertEqual(model.blk[1].config.sliding_window, 0)               # layer 1: full attention
      self.assertAlmostEqual(cfg.yarn_factor, YARN_FACTOR)

      tokens_t = Tensor([TOKENS], dtype="int32")
      x = model.token_embd(tokens_t).float()
      for block in model.blk: x = block(x, 0)
      logits = model.output(model.output_norm(x)).numpy()[0]  # (T, vocab)

    ref = reference_forward(weights, TOKENS)
    np.testing.assert_allclose(logits, ref, rtol=2e-2, atol=2e-2)

  def test_gptoss_chunked_prefill_matches_reference(self):
    """T4.10: reference_forward computes attention over the whole sequence in one shot (with the
    window mask applied to the full T_xT_ score matrix) -- the strongest oracle available, independent
    of tinygrad's own chunking code. Feed the same tokens through model.blk chunk-by-chunk (mirroring
    what Transformer.generate()'s prefill loop does: each chunk calls every block with start_pos
    offset, so sliding-window layer 0's KV cache and mask accumulate across chunks) and check the
    final logits still match -- proves chunked prefill is correct against ground truth, not just
    self-consistent with tinygrad's own single-chunk path (already covered by the test above)."""
    rng = np.random.default_rng(1234)
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "tiny-gpt-oss.gguf"
      weights = build_gguf(path, rng)
      model, _ = Transformer.from_gguf(path)

      tokens_t = Tensor([TOKENS], dtype="int32")
      x = model.token_embd(tokens_t).float()
      # SLIDING_WINDOW=3: chunk sizes below, at, and above the window, with uneven splits, so chunk
      # boundaries land at every offset relative to the window's own boundary
      for chunk_sizes in ([2, 2, 2], [1, 2, 3], [4, 2], [3, 1, 1, 1]):
        for block in model.blk:
          if hasattr(block, "cache_kv"): del block.cache_kv  # fresh KV cache per chunking scheme
        start, outs = 0, []
        for cs in chunk_sizes:
          x_chunk = x[:, start:start + cs]
          for block in model.blk: x_chunk = block(x_chunk, start)
          outs.append(x_chunk)
          start += cs
        x_cat = Tensor.cat(*outs, dim=1)
        logits = model.output(model.output_norm(x_cat)).numpy()[0]
        ref = reference_forward(weights, TOKENS)
        np.testing.assert_allclose(logits, ref, rtol=2e-2, atol=2e-2, err_msg=f"{chunk_sizes=}")

  def test_yarn_disabled_matches_plain_rope(self):
    """yarn_factor=1.0 (the default for every non-yarn arch) must reduce exactly to the old formula."""
    plain = precompute_freqs_cis(8, 16, 10.0, device=None)
    yarn_noop = precompute_freqs_cis(8, 16, 10.0, device=None, yarn_factor=1.0, yarn_orig_ctx=32, yarn_beta_fast=32.0, yarn_beta_slow=1.0)
    np.testing.assert_array_equal(plain.numpy(), yarn_noop.numpy())

  def test_max_context_defaults_to_cap_not_native(self):
    """T4.6: from_gguf() with no max_context arg must NOT pre-allocate KV for the model's full native
    context (_init_state sizes the cache off Transformer.max_context) -- it should cap to
    Transformer.DEFAULT_MAX_CONTEXT instead. A caller that explicitly wants the full native context can
    still ask for it via max_context=None."""
    rng = np.random.default_rng(99)
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "tiny-gpt-oss-bigctx.gguf"
      build_gguf(path, rng, max_ctx=20000)  # native context far bigger than the default cap
      model, kv = Transformer.from_gguf(path)  # no max_context -- the naive/library-default path
      self.assertEqual(kv["gpt-oss.context_length"], 20000)
      self.assertLess(Transformer.DEFAULT_MAX_CONTEXT, 20000)
      self.assertEqual(model.max_context, Transformer.DEFAULT_MAX_CONTEXT)
      # explicit escape hatch: max_context=None still bypasses the cap and gets the full native context
      model_native, _ = Transformer.from_gguf(path, max_context=None)
      self.assertEqual(model_native.max_context, 20000)
      # explicit override still wins (and is still min()'d against native)
      model_explicit, _ = Transformer.from_gguf(path, max_context=64)
      self.assertEqual(model_explicit.max_context, 64)

class TestGPTOSSDecodeByteBudget(unittest.TestCase):
  """T4.11: the 08-19 bench had gpt-oss-20b MXFP4 decoding at 1.69 tok/s while sustaining 100.65 GB/s
  on METAL -- ~60 GB read/token, vs an expected ~2-3 GB/token from active-param count (~3.6B @ ~4.25
  bit + KV + embeddings). Investigation: built a gpt-oss-shaped config matching the real model's MoE
  routing (32 experts, top-4) plus every gpt-oss-specific mechanism (attention sinks, alternating
  sliding-window, clamped swiglu, MoE bias), at CPU device, and did per-kernel byte attribution via a
  patched engine/realize.py track_stats (DEBUG=2's printed 'mem' column is cumulative allocator
  footprint, not per-kernel bytes -- not useful for this). Finding: NOT reproduced at tiny scale, nor
  at the real model's actual per-layer dims (dim=2880, hidden=2880, checked separately -- not in this
  fast CI test). Isolating each mechanism one at a time off a size-matched plain-MoE control showed:
  ExpertWeights.__call__'s `weight[sel]` still does a true k-of-N indexed gather (measured bytes match
  the k-experts formula, not the ~8x-larger dense-all-experts read the T1.3 bias-support change could
  plausibly have broken); attn_sinks adds ~128 B (no multi-pass K/V re-read); the sliding-window mask
  adds one extra kernel but only ~300 B (scales with position, never with max_context); clamp_swiglu
  fuses with ZERO added kernels or bytes vs the unclamped control. The real-model blowup is therefore
  size- or quant-dependent (MXFP4 dequant at 20b scale, cache thrash, mmap amplification), not a bug
  in this shared decode path -- next bench window chases it on the real model.

  This test locks in "still gathered" as a regression guard: 3x the analytic gathered-MoE estimate is
  generous enough to absorb ordinary byte-count drift (measured overhead here is ~1.25x) while still
  catching an actual gather->dense regression (dense would cost ~8x gathered at these dims)."""
  DIM, HIDDEN = 64, 128
  N_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 8
  N_EXPERTS, EXPERTS_PER_TOK, NUM_BLOCKS = 32, 4, 2  # 32/top-4 matches real gpt-oss-20b's routing
  VOCAB, MAX_CTX, SLIDING_WINDOW = 50, 64, 8

  def _cfg(self) -> TransformerConfig:
    return TransformerConfig(
      num_blocks=self.NUM_BLOCKS, dim=self.DIM, hidden_dim=self.HIDDEN, n_heads=self.N_HEADS,
      n_kv_heads=self.N_KV_HEADS, norm_eps=1e-5, vocab_size=self.VOCAB, head_dim=self.HEAD_DIM,
      rope_theta=10000.0, rope_dim=self.HEAD_DIM, v_head_dim=self.HEAD_DIM, max_context=self.MAX_CTX,
      num_experts=self.N_EXPERTS, num_experts_per_tok=self.EXPERTS_PER_TOK,
      expert_gating_func=ExpertGating.SOFTMAX_WEIGHT, moe_bias=True, router_bias=True,
      attn_sinks=True, attn_out_bias=True, qkv_bias=True,
      sliding_window=self.SLIDING_WINDOW, sliding_layers=(True, False),
      clamp_swiglu=True, swiglu_limit=7.0, swiglu_alpha=1.702)

  def test_decode_bytes_stay_near_gathered_not_dense(self):
    # device_map forces CPU regardless of the runner's Device.DEFAULT -- deterministic byte estimate
    # (GlobalCounters.global_mem is a static AST-shape estimate, not an actual driver counter, so it's
    # backend-independent; CPU keeps this test fast and free of a METAL dependency).
    # realize_placement() is required after a bare Transformer(device_map=...) construction (from_gguf
    # calls it for you) -- skipping it left every weight an unrealized .to_() COPY, which alone produced
    # an ~8x read blowup indistinguishable from a real gather->dense regression (caught while building
    # this test). That's the documented footgun in Transformer.realize_placement's own docstring, not a
    # bug in the decode path -- call it here like any real device_map caller must.
    model = Transformer(self._cfg(), device_map="CPU:0")
    model.realize_placement()
    gen = model.generate([1, 2, 3, 4, 5], chunk_size=32, temperature=0.0)
    for _ in range(1 + 5): next(gen)  # prefill + warm the decode jit variant (first decode call captures it)
    GlobalCounters.reset()
    next(gen)
    actual = GlobalCounters.global_mem

    per_expert_elems = 2 * self.HIDDEN * self.DIM + self.DIM * self.HIDDEN  # gate + up + down
    itemsize = 4  # this config skips GGUF loading, weights stay dtypes.default_float (fp32)
    gathered = self.NUM_BLOCKS * self.EXPERTS_PER_TOK * per_expert_elems * itemsize
    dense = self.NUM_BLOCKS * self.N_EXPERTS * per_expert_elems * itemsize
    self.assertLess(actual, 3 * gathered,
      f"decode step read {actual} B, expected near the gathered-MoE estimate ({gathered} B) -- looks "
      f"like ExpertWeights.__call__ degraded toward the dense all-experts read ({dense} B)")

class TestGPTOSSDecodeByteBudgetMXFP4(unittest.TestCase):
  """T4.13: the byte-budget test above (T4.11) never actually exercises MXFP4 -- its config skips GGUF
  loading entirely and keeps weights at dtypes.default_float (see its own comment at 'itemsize = 4').
  That's why it couldn't reproduce the real 20b model's ~59 GB/token: the blowup isn't scale-dependent
  (expert count, layer count, hidden dim) at all -- it reproduces already at these SAME tiny dims, the
  moment expert weights actually go through the MXFP4 dequant path (confirmed by a standalone repro:
  a bare `weight[sel]` gather over an MXFP4-dequanted (32, 64, 128) tensor read ~1.8 MB/token, MORE
  than a full dense all-experts fp32 materialization, vs ~440 KB for the identical shape in Q4_0).

  Root cause (schedule/rangeify.py): the original MXFP4 dequant (gguf.py, ggml_type==39) computed both
  the block scale and the E2M1 4-bit value via Tensor-indexed LUT gathers (`lut[codes]`). A gather reads
  its LUT through a buffer-accessing REDUCE, and remove_bufferize's buffer_in_reduce check refuses to
  fuse a bufferize point containing a buffer-reading REDUCE into any consumer that itself indexes the
  result -- which is exactly what MoE's `weight[sel]` expert-selection gather is. Result: the ENTIRE
  dequantized expert tensor (every expert, not just the k selected) materialized every decode step. Fix:
  compute both the block scale and the E2M1 value via pure ALU bit-ops (no LUT gather, no extra buffer)
  -- bit-exact vs the LUT (verified for all 256 e8m0 byte values and all 16 four-bit codes) -- which lets
  rangeify fuse the dequant into weight[sel]'s gather again, same as every other quant format.

  This test builds a real MXFP4 GGUF (ALL expert tensors quantized, not just one -- the real model's
  shape) at the T4.11 byte-budget dims and checks decode bytes stay near the gathered-MXFP4 estimate,
  not the dense (all-experts) one."""
  DIM, HIDDEN = 64, 128
  N_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 8
  N_EXPERTS, EXPERTS_PER_TOK, NUM_BLOCKS = 32, 4, 2
  VOCAB, MAX_CTX = 50, 64

  def _build_gguf(self, path: pathlib.Path, rng: np.random.Generator):
    w = gguf.GGUFWriter(str(path), "gpt-oss")
    w.add_context_length(self.MAX_CTX)
    w.add_embedding_length(self.DIM)
    w.add_block_count(self.NUM_BLOCKS)
    w.add_head_count(self.N_HEADS)
    w.add_head_count_kv(self.N_KV_HEADS)
    w.add_key_length(self.HEAD_DIM)
    w.add_value_length(self.HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_expert_count(self.N_EXPERTS)
    w.add_expert_used_count(self.EXPERTS_PER_TOK)
    w.add_expert_feed_forward_length(self.HIDDEN)
    w.add_rope_freq_base(10000.0)
    w.add_token_list([f"<t{i}>" for i in range(self.VOCAB)])

    def norm(name, dim): w.add_tensor(name, rng.normal(1.0, 0.1, dim).astype(np.float16))
    def lin(name, out_f, in_f, bias):
      w.add_tensor(f"{name}.weight", rng.normal(0, 0.3, (out_f, in_f)).astype(np.float16))
      if bias: w.add_tensor(f"{name}.bias", rng.normal(0, 0.3, out_f).astype(np.float16))

    for i in range(self.NUM_BLOCKS):
      p = f"blk.{i}"
      norm(f"{p}.attn_norm.weight", self.DIM)
      norm(f"{p}.post_attention_norm.weight", self.DIM)
      lin(f"{p}.attn_q", self.N_HEADS * self.HEAD_DIM, self.DIM, bias=True)
      lin(f"{p}.attn_k", self.N_KV_HEADS * self.HEAD_DIM, self.DIM, bias=True)
      lin(f"{p}.attn_v", self.N_KV_HEADS * self.HEAD_DIM, self.DIM, bias=True)
      lin(f"{p}.attn_output", self.DIM, self.N_HEADS * self.HEAD_DIM, bias=True)
      w.add_tensor(f"{p}.attn_sinks.weight", rng.normal(0, 0.5, self.N_HEADS).astype(np.float16))
      w.add_tensor(f"{p}.ffn_gate_inp.weight", rng.normal(0, 0.05, (self.N_EXPERTS, self.DIM)).astype(np.float16))
      w.add_tensor(f"{p}.ffn_gate_inp.bias", rng.normal(0, 1, self.N_EXPERTS).astype(np.float16))

      # ALL expert tensors MXFP4 -- the real model's shape (T4.11's builder only quantized one, layer 0's down)
      for name, out_f, in_f in ((f"{p}.ffn_gate_exps", self.HIDDEN, self.DIM), (f"{p}.ffn_up_exps", self.HIDDEN, self.DIM),
                                 (f"{p}.ffn_down_exps", self.DIM, self.HIDDEN)):
        wt = rng.normal(0, 0.3, (self.N_EXPERTS, out_f, in_f)).astype(np.float32)
        packed = gguf.quantize(wt, gguf.GGMLQuantizationType.MXFP4)
        w.add_tensor(f"{name}.weight", packed, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
        w.add_tensor(f"{name}.bias", rng.normal(0, 0.3, (self.N_EXPERTS, out_f)).astype(np.float16))

    w.add_tensor("token_embd.weight", rng.normal(0, 0.3, (self.VOCAB, self.DIM)).astype(np.float16))
    w.add_tensor("output_norm.weight", rng.normal(1.0, 0.1, self.DIM).astype(np.float16))
    w.add_tensor("output.weight", rng.normal(0, 0.3, (self.VOCAB, self.DIM)).astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

  @Context(DEV="CPU")  # NOT device_map="CPU:0": device_map moves params off Device.DEFAULT, and
  # Transformer.realize_placement() force-.contiguous()+realizes every such param at load time
  # (documented footgun, model.py's realize_placement docstring) -- for GGUF-loaded MXFP4 weights
  # that defeats dequant fusion before decode ever starts, silently making this test measure an
  # already-materialized buffer regardless of the dequant bug it exists to catch. Scoping
  # Device.DEFAULT itself keeps decode's whole path (including load) on one device, matching how
  # the real gpt-oss bench runs (no device_map at all) and actually exercising the lazy fusion path.
  def test_decode_bytes_stay_near_gathered_not_dense_mxfp4(self):
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "byte-budget-mxfp4.gguf"
      self._build_gguf(path, rng)
      model, _ = Transformer.from_gguf(path, max_context=self.MAX_CTX)

    gen = model.generate([1, 2, 3, 4, 5], chunk_size=32, temperature=0.0)
    for _ in range(1 + 5): next(gen)  # prefill + warm the decode jit variant
    GlobalCounters.reset()
    next(gen)
    actual = GlobalCounters.global_mem

    per_expert_mxfp4_bytes = (2 * self.HIDDEN * self.DIM + self.DIM * self.HIDDEN) // 32 * 17  # gate+up+down, packed
    gathered = self.NUM_BLOCKS * self.EXPERTS_PER_TOK * per_expert_mxfp4_bytes
    dense = self.NUM_BLOCKS * self.N_EXPERTS * per_expert_mxfp4_bytes
    self.assertLess(actual, 3 * gathered,
      f"decode step read {actual} B, expected near the gathered-MXFP4 estimate ({gathered} B) -- looks "
      f"like the MXFP4 dequant materialized the full (dense, {dense} B) expert tensor before the gather")

class TestGPTOSSDecodeByteBudgetIQ(unittest.TestCase):
  """T4.22: upstream #17316 reproduced on this fork -- IQ3_XXS/IQ4_XS dequant materialized ALL
  experts per decode token, same buffer_in_reduce class as T4.13's pre-fix MXFP4 (see the
  ggml_type==39 comment in gguf.py), but WORSE: IQ3_XXS's dequant chains TWO independent
  Tensor-gathers (the iq3xxs_grid codebook AND the even_signs sign-parity table), each
  independently forcing full materialization, so pre-fix it read ~63x the gathered-MoE estimate --
  8x even the DENSE all-experts estimate (measured; MXFP4 pre-fix was ~8x gathered, not 8x dense).

  Fix (gguf.py, ggml_type 18/23): even_signs is replaced with an exact XOR-fold parity computation
  (pure ALU bit-ops, unbounded size, same spirit as MXFP4's own fix). The iq3xxs_grid codebook (256
  entries) and IQ4_XS's kvalues_iq4nl table (16 entries) are genuine arbitrary codebooks with no
  closed-form bit trick, so they're replaced with a compile-time balanced binary select-tree
  (nested .where() over the table's literal Python floats) instead -- mechanically dodges
  buffer_in_reduce because the expression touches no buffer or gather at all, not because of any
  bit-decomposition. All three replacements are bit-exact vs the original Tensor-gather form
  (verified exhaustively over the full code space in an ad-hoc session script; test/unit/test_gguf.py's
  TestGGUF/TestGGUFGEMV already separately cover bit-exactness against gguf-py's real quantizer for
  IQ3_XXS/IQ4_XS end to end -- this class only guards the byte-count regression).

  Needs a block-aligned per-expert row (256-element blocks for IQ3_XXS/IQ4_XS) unlike
  T4.11/T4.13's DIM=64/HIDDEN=128 (fine for MXFP4's 32-element blocks) -- DIM=HIDDEN=256 instead;
  gguf-py's quant_shape_from_byte_shape enforces real per-row block alignment for raw_dtype
  tensors. IQ3_XXS/IQ4_XS have no Python quantizer in the gguf package (quantize() raises
  NotImplementedError) so raw block bytes are hand-rolled (random, structurally valid) instead of a
  real quantization -- fine for a byte-COUNT mechanism test, not a numerics one."""
  DIM, HIDDEN = 256, 256
  N_HEADS, N_KV_HEADS, HEAD_DIM = 8, 2, 8
  N_EXPERTS, EXPERTS_PER_TOK, NUM_BLOCKS = 32, 4, 2
  VOCAB, MAX_CTX = 50, 64
  GGML_NBYTES = {18: (256, 98), 23: (256, 136)}  # ggml_type: (elements/block, bytes/block) -- gguf.py's _GGML_QUANT

  def _build_gguf(self, path: pathlib.Path, rng: np.random.Generator, ggml_type: int):
    w = gguf.GGUFWriter(str(path), "gpt-oss")
    w.add_context_length(self.MAX_CTX)
    w.add_embedding_length(self.DIM)
    w.add_block_count(self.NUM_BLOCKS)
    w.add_head_count(self.N_HEADS)
    w.add_head_count_kv(self.N_KV_HEADS)
    w.add_key_length(self.HEAD_DIM)
    w.add_value_length(self.HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_expert_count(self.N_EXPERTS)
    w.add_expert_used_count(self.EXPERTS_PER_TOK)
    w.add_expert_feed_forward_length(self.HIDDEN)
    w.add_rope_freq_base(10000.0)
    w.add_token_list([f"<t{i}>" for i in range(self.VOCAB)])

    def norm(name, dim): w.add_tensor(name, rng.normal(1.0, 0.1, dim).astype(np.float16))
    def lin(name, out_f, in_f, bias):
      w.add_tensor(f"{name}.weight", rng.normal(0, 0.3, (out_f, in_f)).astype(np.float16))
      if bias: w.add_tensor(f"{name}.bias", rng.normal(0, 0.3, out_f).astype(np.float16))

    nelem, nbytes = self.GGML_NBYTES[ggml_type]
    for i in range(self.NUM_BLOCKS):
      p = f"blk.{i}"
      norm(f"{p}.attn_norm.weight", self.DIM)
      norm(f"{p}.post_attention_norm.weight", self.DIM)
      lin(f"{p}.attn_q", self.N_HEADS * self.HEAD_DIM, self.DIM, bias=True)
      lin(f"{p}.attn_k", self.N_KV_HEADS * self.HEAD_DIM, self.DIM, bias=True)
      lin(f"{p}.attn_v", self.N_KV_HEADS * self.HEAD_DIM, self.DIM, bias=True)
      lin(f"{p}.attn_output", self.DIM, self.N_HEADS * self.HEAD_DIM, bias=True)
      w.add_tensor(f"{p}.attn_sinks.weight", rng.normal(0, 0.5, self.N_HEADS).astype(np.float16))
      w.add_tensor(f"{p}.ffn_gate_inp.weight", rng.normal(0, 0.05, (self.N_EXPERTS, self.DIM)).astype(np.float16))
      w.add_tensor(f"{p}.ffn_gate_inp.bias", rng.normal(0, 1, self.N_EXPERTS).astype(np.float16))

      for name, out_f, in_f in ((f"{p}.ffn_gate_exps", self.HIDDEN, self.DIM), (f"{p}.ffn_up_exps", self.HIDDEN, self.DIM),
                                 (f"{p}.ffn_down_exps", self.DIM, self.HIDDEN)):
        assert in_f % nelem == 0, f"{in_f=} {nelem=} not block-aligned"
        n = self.N_EXPERTS * out_f * in_f
        raw = rng.integers(0, 256, n // nelem * nbytes, dtype=np.uint8)
        # raw_shape is gguf-py's convention for raw_dtype tensors: BYTE shape (last axis =
        # bytes/row), not the logical element shape -- quant_shape_from_byte_shape derives the
        # logical shape from it.
        w.add_tensor(f"{name}.weight", raw, raw_shape=(self.N_EXPERTS, out_f, in_f // nelem * nbytes),
                     raw_dtype=gguf.GGMLQuantizationType(ggml_type))
        w.add_tensor(f"{name}.bias", rng.normal(0, 0.3, (self.N_EXPERTS, out_f)).astype(np.float16))

    w.add_tensor("token_embd.weight", rng.normal(0, 0.3, (self.VOCAB, self.DIM)).astype(np.float16))
    w.add_tensor("output_norm.weight", rng.normal(1.0, 0.1, self.DIM).astype(np.float16))
    w.add_tensor("output.weight", rng.normal(0, 0.3, (self.VOCAB, self.DIM)).astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

  @Context(DEV="CPU")  # see TestGPTOSSDecodeByteBudgetMXFP4's identical comment: device_map would
  # force-realize params at load time and defeat the dequant fusion this test exists to catch.
  def _run(self, ggml_type: int):
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "byte-budget-iq.gguf"
      self._build_gguf(path, rng, ggml_type)
      model, _ = Transformer.from_gguf(path, max_context=self.MAX_CTX)

    gen = model.generate([1, 2, 3, 4, 5], chunk_size=32, temperature=0.0)
    for _ in range(1 + 5): next(gen)  # prefill + warm the decode jit variant
    GlobalCounters.reset()
    next(gen)
    actual = GlobalCounters.global_mem

    nelem, nbytes = self.GGML_NBYTES[ggml_type]
    per_expert_bytes = (2 * self.HIDDEN * self.DIM + self.DIM * self.HIDDEN) // nelem * nbytes  # gate + up + down
    gathered = self.NUM_BLOCKS * self.EXPERTS_PER_TOK * per_expert_bytes
    dense = self.NUM_BLOCKS * self.N_EXPERTS * per_expert_bytes
    self.assertLess(actual, 3 * gathered,
      f"decode step read {actual} B, expected near the gathered-MoE estimate ({gathered} B) -- looks "
      f"like the ggml_type={ggml_type} dequant materialized the full (dense, {dense} B) expert tensor before the gather")

  def test_decode_bytes_stay_near_gathered_not_dense_iq3_xxs(self): self._run(18)
  def test_decode_bytes_stay_near_gathered_not_dense_iq4_xs(self): self._run(23)

# ---------------------------------------------------------------------------------------------
# Real gpt-oss-20b GGUF, metadata-only validation (no tensor data touched - GGUFReader mmaps the
# file and we only read the kv/token-list section). Skipped in CI where the file isn't present.
# Model generation itself is out of scope here (deferred to a bench session).
# ---------------------------------------------------------------------------------------------

REAL_GPTOSS_GGUF = pathlib.Path("/Users/artur/Library/Caches/tinygrad/downloads/35ff51c27772d214bab0172591e2ade8")

def _read_gguf_kv(path: pathlib.Path) -> dict:
  # mirrors test/unit/test_gguf.py's own kv-reading pattern, via gguf-py's mmap-based GGUFReader
  reader = gguf.GGUFReader(str(path))
  kv: dict = {}
  for k, f in reader.fields.items():
    if k.startswith("GGUF."): continue  # file header keys (version, tensor_count, kv_count)
    is_str = f.types[-1] == gguf.GGUFValueType.STRING
    def read_val(i, parts=f.parts, is_str=is_str): return bytes(parts[i]).decode("utf-8") if is_str else parts[i][0].item()
    kv[k] = [read_val(i) for i in f.data] if f.types[0] == gguf.GGUFValueType.ARRAY else read_val(-1)
  return kv

@unittest.skipUnless(REAL_GPTOSS_GGUF.exists(), "real gpt-oss-20b GGUF not present, skipping metadata validation")
class TestGPTOSSRealTokenizer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.kv = _read_gguf_kv(REAL_GPTOSS_GGUF)
    cls.tok = SimpleTokenizer.from_gguf_kv(cls.kv)

  def test_arch_and_preset(self):
    self.assertEqual(self.kv["general.architecture"], "gpt-oss")
    self.assertEqual(self.kv["tokenizer.ggml.pre"], "gpt-4o")
    self.assertEqual(self.tok.preset, "gpt-4o")

  def test_roundtrip(self):
    cases = ["hello world", "Hello, World! 123", "  multiple   spaces  ", "trailing space ", "\ttabs\tand\nnewlines\r\n",
             "unicode: café 中文 \U0001F600", "CamelCaseWordBoundary", "don't can't I'm we'll",
             "numbers 1 22 333 4444 55555", "punctuation!!! ...whoa??", ""]
    for s in cases:
      self.assertEqual(self.tok.decode(self.tok.encode(s)), s, f"roundtrip failed for {s!r}")

  def test_harmony_special_tokens_resolve_to_single_ids(self):
    # confirmed present in this file's own token inventory (see the investigative dump in the T1.3 report)
    for special in ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>"):
      self.assertIn(special, self.tok._special_tokens, f"{special} missing from this GGUF's token inventory")
      ids = self.tok.encode(special)
      self.assertEqual(len(ids), 1, f"{special} should resolve to a single token id, got {ids}")
      self.assertEqual(self.tok.decode(ids), special)

  def test_chat_template_loads_via_generic_jinja_path(self):
    # this is the exact loading path from cli.py's main(): if jinja2 is missing, that path silently
    # falls back to FallbackTemplate (no harmony support there - not cheap to add, see T1.3 report)
    ct = self.kv.get("tokenizer.chat_template")
    self.assertIsNotNone(ct)
    try: import jinja2
    except ImportError: self.skipTest("jinja2 not installed - the harmony template silently falls back to FallbackTemplate (unsupported)")
    env = jinja2.Environment()
    env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
    env.globals['bos_token'] = self.tok.decode([self.tok.bos_id]) if self.tok.bos_id is not None else ""
    env.globals['eos_token'] = self.tok.decode([self.tok.eos_id])
    template = env.from_string(ct)  # raises jinja2.TemplateSyntaxError if the template can't even parse
    out = template.render(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True)
    self.assertIsInstance(out, str)
    self.assertGreater(len(out), 0)

if __name__ == '__main__':
  unittest.main()
