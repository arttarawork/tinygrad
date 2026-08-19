import math, pathlib, tempfile, unittest
import numpy as np
import gguf
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, precompute_freqs_cis

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

def build_gguf(path: pathlib.Path, rng: np.random.Generator) -> dict:
  """Writes a tiny gpt-oss-arch GGUF and returns the (already fp16-rounded) numpy weights used,
  keyed the same way the reference model below expects them."""
  w = gguf.GGUFWriter(str(path), "gpt-oss")
  w.add_context_length(MAX_CTX)
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

  def test_yarn_disabled_matches_plain_rope(self):
    """yarn_factor=1.0 (the default for every non-yarn arch) must reduce exactly to the old formula."""
    plain = precompute_freqs_cis(8, 16, 10.0, device=None)
    yarn_noop = precompute_freqs_cis(8, 16, 10.0, device=None, yarn_factor=1.0, yarn_orig_ctx=32, yarn_beta_fast=32.0, yarn_beta_slow=1.0)
    np.testing.assert_array_equal(plain.numpy(), yarn_noop.numpy())

if __name__ == '__main__':
  unittest.main()
