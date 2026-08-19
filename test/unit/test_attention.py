import unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn, GlobalCounters, Variable, TinyJit
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.llm import model
from tinygrad.llm.model import (
  GatedDeltaNetBlock, SSMConfig, TransformerBlock, TransformerConfig,
  apply_rope as apply_rope_new, precompute_freqs_cis, pairwise_topk,
)
from tinygrad.llm.attn_kernel import tuned_decode_attention, CHUNK
from test.helpers import assert_kernel_count, assert_jit_cache_len

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestAttention(unittest.TestCase):
  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_partial_rope_in_attention(self):
    dim, rope_dim, seqlen = 8, 4, 3
    config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=16, n_heads=1, n_kv_heads=1,
                               norm_eps=1e-5, vocab_size=32, head_dim=dim, rope_theta=10000.0,
                               rope_dim=rope_dim, v_head_dim=dim, max_context=8)
    block = TransformerBlock(config)

    x = Tensor.randn(1, seqlen, dim, dtype=dtypes.float32)
    x_norm = block.attn_norm(x)
    k = block.attn_k(x_norm).reshape(1, seqlen, 1, dim).transpose(1, 2)

    precompute_freqs_cis.cache_clear()
    block.cache_kv = Tensor.empty(2, 1, 1, config.max_context, max(dim, config.v_head_dim), device=x.device)
    block.freqs_cis = precompute_freqs_cis(rope_dim, config.max_context, config.rope_theta)
    block._attention(x_norm, 0).realize()

    expected = apply_rope_new(k[..., :rope_dim], block.freqs_cis[:seqlen]).cat(k[..., rope_dim:], dim=-1)
    np.testing.assert_allclose(block.cache_kv[0, :, :, :seqlen, :].numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

class TestGatedDeltaNetBlock(unittest.TestCase):
  def _tensor_linspace(self, start:float, stop:float, shape:tuple[int, ...]) -> Tensor:
    return Tensor.linspace(start, stop, int(np.prod(shape)), dtype=dtypes.float32).reshape(*shape)

  def _make_config(self, **kwargs):
    return TransformerConfig(**({"num_blocks":1, "dim":32, "hidden_dim":64, "n_heads":1, "n_kv_heads":1,
                                 "norm_eps":1e-5, "vocab_size":32, "head_dim":32, "rope_theta":10000.0,
                                 "rope_dim":32, "v_head_dim":32, "max_context":4, "ssm_layers":(True,),
                                 "ssm":SSMConfig(conv_kernel=2, state_size=32, group_count=1, time_step_rank=1, inner_size=32)} | kwargs))

  def _make_block(self, config:TransformerConfig) -> GatedDeltaNetBlock:
    block = GatedDeltaNetBlock(config, config.ssm)
    block.attn_norm.weight = self._tensor_linspace(0.8, 1.2, (config.dim,))
    block.attn_qkv.weight = self._tensor_linspace(-0.15, 0.2, (block.conv_channels, config.dim))
    block.attn_gate.weight = self._tensor_linspace(-0.1, 0.15, (config.ssm.inner_size, config.dim))
    block.ssm_alpha.weight = self._tensor_linspace(-0.08, 0.12, (block.num_v_heads, config.dim))
    block.ssm_beta.weight = self._tensor_linspace(-0.12, 0.07, (block.num_v_heads, config.dim))
    block.ssm_conv1d["weight"] = self._tensor_linspace(-0.05, 0.05, (block.conv_channels, block.ssm_conv_kernel))
    block.ssm_dt["bias"] = self._tensor_linspace(-0.1, 0.1, (block.num_v_heads,))
    block.ssm_a = self._tensor_linspace(-0.1, -0.05, (block.num_v_heads,))
    block.ssm_norm.weight = self._tensor_linspace(0.9, 1.1, (block.head_v_dim,))
    block.ssm_out.weight = self._tensor_linspace(-0.2, 0.18, (config.dim, config.ssm.inner_size))
    return block

  def _run_attention(self, block:GatedDeltaNetBlock, x:Tensor, start_pos:int):
    x_norm = block.attn_norm(x)
    block._init_state(x_norm)
    return block._attention(x_norm, start_pos).realize().numpy()

  def _cache_views(self, block:GatedDeltaNetBlock) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(block, 'conv_state'):
      return block.conv_state.numpy(), block.recurrent_state.numpy()
    else:
      conv_flat = (block.ssm_conv_kernel - 1) * block.conv_channels
      cache = block.delta_cache.numpy()
      conv_state = cache[:, :conv_flat].reshape(cache.shape[0], block.ssm_conv_kernel - 1, block.conv_channels)
      recurrent_state = cache[:, conv_flat:].reshape(cache.shape[0], block.num_v_heads, block.head_v_dim, block.head_v_dim)
      return conv_state, recurrent_state

  def _reset_state(self, block:GatedDeltaNetBlock):
    Tensor.realize(block.conv_state.assign(block.conv_state.const_like(0)),
                   block.recurrent_state.assign(block.recurrent_state.const_like(0)))

  def _linear_np(self, x:np.ndarray, weight:np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ weight.T.astype(np.float32)

  def _rms_norm_np(self, x:np.ndarray, weight:np.ndarray, eps:float) -> np.ndarray:
    x_float = x.astype(np.float32)
    return (x_float / np.sqrt((x_float * x_float).mean(axis=-1, keepdims=True) + eps)) * weight.astype(np.float32)

  def _normalize_np(self, x:np.ndarray, eps:float=1e-6) -> np.ndarray:
    return x / np.maximum(np.sqrt((x * x).sum(axis=-1, keepdims=True)), eps)

  def _softplus_np(self, x:np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

  def _silu_np(self, x:np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))

  def _naive_attention(self, block:GatedDeltaNetBlock, x:Tensor):
    x_np = x.numpy().astype(np.float32)
    B, T, _ = x_np.shape
    conv_state = np.zeros((B, block.ssm_conv_kernel - 1, block.conv_channels), dtype=np.float32)
    recurrent_state = np.zeros((B, block.num_v_heads, block.head_v_dim, block.head_v_dim), dtype=np.float32)
    conv_weight = block.ssm_conv1d["weight"].numpy().astype(np.float32).T[None, :, :]
    qkv_weight = block.attn_qkv.weight.numpy().astype(np.float32)
    gate_weight = block.attn_gate.weight.numpy().astype(np.float32)
    alpha_weight = block.ssm_alpha.weight.numpy().astype(np.float32)
    beta_weight = block.ssm_beta.weight.numpy().astype(np.float32)
    out_weight = block.ssm_out.weight.numpy().astype(np.float32)
    dt_bias = block.ssm_dt["bias"].numpy().astype(np.float32)
    ssm_a = block.ssm_a.numpy().astype(np.float32)
    attn_norm_weight = block.attn_norm.weight.numpy().astype(np.float32)
    ssm_norm_weight = block.ssm_norm.weight.numpy().astype(np.float32)
    outputs, conv_states, recurrent_states = [], [], []

    for t in range(T):
      x_norm = self._rms_norm_np(x_np[:, t:t+1, :], attn_norm_weight, block.attn_norm.eps)
      x_half = x_norm.astype(np.float16)
      out_gate = self._linear_np(x_half, gate_weight).reshape(B, 1, block.num_v_heads, block.head_v_dim)
      beta = 1.0 / (1.0 + np.exp(-self._linear_np(x_half, beta_weight))).reshape(B, block.num_v_heads, 1, 1)
      alpha = np.exp((self._softplus_np(self._linear_np(x_half, alpha_weight) + dt_bias)).reshape(B, block.num_v_heads, 1, 1) *
                     ssm_a.reshape(1, block.num_v_heads, 1, 1))
      conv_window = np.concatenate([conv_state, self._linear_np(x_half, qkv_weight)], axis=1)
      conv_out = self._silu_np((conv_window * conv_weight).sum(axis=1))
      q, k, v = np.split(conv_out, [block.q_dim, 2 * block.q_dim], axis=-1)
      q = self._normalize_np(q.reshape(B, block.num_k_heads, block.head_k_dim))
      k = self._normalize_np(k.reshape(B, block.num_k_heads, block.head_k_dim))
      v = v.reshape(B, block.num_v_heads, block.head_v_dim)
      if block.num_v_heads != block.num_k_heads:
        k_repeat = block.num_v_heads // block.num_k_heads
        q = np.repeat(q[:, None, :, :], k_repeat, axis=1).reshape(B, block.num_v_heads, block.head_k_dim)
        k = np.repeat(k[:, None, :, :], k_repeat, axis=1).reshape(B, block.num_v_heads, block.head_k_dim)
      q, k, v = (q * (block.head_k_dim ** -0.5))[..., None], k[..., None], v[..., None]
      recurrent_state = recurrent_state * alpha
      recurrent_state = recurrent_state + np.matmul((v - np.matmul(recurrent_state, k)) * beta, np.swapaxes(k, -1, -2))
      core_attn_out = np.matmul(recurrent_state, q).squeeze(-1).reshape(B, 1, block.num_v_heads, block.head_v_dim)
      core_attn_out = self._rms_norm_np(core_attn_out, ssm_norm_weight, block.ssm_norm.eps)
      out = self._linear_np((core_attn_out * self._silu_np(out_gate)).reshape(B, 1, -1).astype(np.float16), out_weight)
      conv_state = conv_window[:, 1:, :]
      outputs.append(out)
      conv_states.append(conv_state.copy())
      recurrent_states.append(recurrent_state.copy())

    return outputs, conv_states, recurrent_states

  def test_gatedeltanet_reference_and_reset(self):
    config = self._make_config(max_context=3)
    block = self._make_block(config)
    x = Tensor.linspace(-1.0, 1.0, 3 * config.dim, dtype=dtypes.float32).reshape(1, 3, config.dim)

    expected_outs, expected_conv, expected_recurrent = self._naive_attention(block, x)
    out = self._run_attention(block, x, 0)
    conv_state, recurrent_state = self._cache_views(block)
    np.testing.assert_allclose(out, np.concatenate(expected_outs, axis=1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(conv_state, expected_conv[-1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(recurrent_state, expected_recurrent[-1], rtol=1e-3, atol=1e-3)
    self._reset_state(block)

    for step in range(x.shape[1]):
      out = self._run_attention(block, x[:, step:step+1], step)
      conv_state, recurrent_state = self._cache_views(block)
      np.testing.assert_allclose(out, expected_outs[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet output mismatch at step {step}")
      np.testing.assert_allclose(conv_state, expected_conv[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet conv cache mismatch at step {step}")
      np.testing.assert_allclose(recurrent_state, expected_recurrent[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet recurrent cache mismatch at step {step}")

    warmup = Tensor.linspace(-0.5, 0.5, 2 * config.dim, dtype=dtypes.float32).reshape(1, 2, config.dim)
    prompt = Tensor.linspace(0.75, -0.75, 2 * config.dim, dtype=dtypes.float32).reshape(1, 2, config.dim)

    for i in range(warmup.shape[1]): self._run_attention(block, warmup[:, i:i+1], i)
    self._reset_state(block)
    expected_outs, expected_conv, expected_recurrent = self._naive_attention(block, prompt)

    for step in range(prompt.shape[1]):
      out = self._run_attention(block, prompt[:, step:step+1], step)
      conv_state, recurrent_state = self._cache_views(block)
      np.testing.assert_allclose(out, expected_outs[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset output mismatch at step {step}")
      np.testing.assert_allclose(conv_state, expected_conv[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset conv cache mismatch at step {step}")
      np.testing.assert_allclose(recurrent_state, expected_recurrent[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset recurrent cache mismatch at step {step}")

  def test_kda_channel_decay(self):
    config = self._make_config(dim=4, hidden_dim=8, n_heads=2, head_dim=4, rope_dim=4, v_head_dim=4,
      ssm=SSMConfig(conv_kernel=2, state_size=2, group_count=2, time_step_rank=2, inner_size=4, kda=True))
    block, x = GatedDeltaNetBlock(config, config.ssm), Tensor([[[1., 2., 0., 0.], [2., 1., 0., 0.]]])
    block.ssm_f_a.weight = Tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    block.ssm_f_b.weight = Tensor([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
    block._init_state(x)
    initial_state = Tensor.arange(8, dtype=dtypes.float32).reshape(1, 2, 2, 2)
    block.recurrent_state.assign(initial_state).realize()
    block.ssm_a = Tensor([[-1.], [-1.]])
    block._attention(x, x.shape[1]).realize()
    alpha = np.exp(-self._softplus_np(np.array([[1, 2, 3, 4], [2, 1, 3, 5]])).reshape(2, 2, 2)).prod(0)
    np.testing.assert_allclose(block.recurrent_state.numpy(), initial_state.numpy() * alpha[..., None], rtol=1e-5, atol=1e-5)

  def test_kda_prefill_matches_decode(self):
    config = self._make_config(ssm=SSMConfig(conv_kernel=2, state_size=32, group_count=1, time_step_rank=1, inner_size=32, kda=True))
    block = GatedDeltaNetBlock(config, config.ssm)
    for p in nn.state.get_parameters(block):
      p.replace(self._tensor_linspace(-0.05, 0.05, p.shape) if len(p.shape) > 1 else self._tensor_linspace(0.05, 0.1, p.shape))
    x = self._tensor_linspace(-0.5, 0.5, (1, 3, config.dim))
    prefill = self._run_attention(block, x, 0)
    prefill_conv, prefill_recurrent = self._cache_views(block)
    self._reset_state(block)
    decode = np.concatenate([self._run_attention(block, x[:, i:i+1], i) for i in range(3)], axis=1)
    decode_conv, decode_recurrent = self._cache_views(block)
    np.testing.assert_allclose(prefill, decode, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(prefill_conv, decode_conv, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(prefill_recurrent, decode_recurrent, rtol=1e-3, atol=1e-3)

  def test_varied_chunk_sizes_match_decode(self):
    for kda in (False, True):
      ssm = SSMConfig(conv_kernel=2, state_size=32, group_count=1, time_step_rank=1, inner_size=32, kda=kda)
      config = self._make_config(ssm=ssm)
      if kda:
        block = GatedDeltaNetBlock(config, config.ssm)
        for p in nn.state.get_parameters(block):
          p.replace(self._tensor_linspace(-0.05, 0.05, p.shape) if len(p.shape) > 1 else self._tensor_linspace(0.05, 0.1, p.shape))
      else: block = self._make_block(config)
      x = self._tensor_linspace(-0.5, 0.5, (1, 4, config.dim))
      decode = np.concatenate([self._run_attention(block, x[:, i:i+1], i) for i in range(4)], axis=1)
      decode_conv, decode_recurrent = self._cache_views(block)
      for chunking in ([4], [2, 2], [1, 3], [3, 1], [2, 1, 1]):
        self._reset_state(block)
        outs, start = [], 0
        for size in chunking:
          outs.append(self._run_attention(block, x[:, start:start+size], start))
          start += size
        chunked_conv, chunked_recurrent = self._cache_views(block)
        np.testing.assert_allclose(np.concatenate(outs, axis=1), decode, rtol=1e-3, atol=1e-3, err_msg=f"{kda=} {chunking=}")
        np.testing.assert_allclose(chunked_conv, decode_conv, rtol=1e-3, atol=1e-3, err_msg=f"{kda=} {chunking=}")
        np.testing.assert_allclose(chunked_recurrent, decode_recurrent, rtol=1e-3, atol=1e-3, err_msg=f"{kda=} {chunking=}")

  def test_start_zero_resets_realized_state(self):
    config, x = self._make_config(max_context=3), self._tensor_linspace(-1, 1, (1, 3, 32))
    block = self._make_block(config)
    self._run_attention(block, x, 0)
    restarted = self._run_attention(block, x[:, :2], 0)
    fresh = self._run_attention(self._make_block(config), x[:, :2], 0)
    np.testing.assert_allclose(restarted, fresh, rtol=1e-3, atol=1e-3)

class TestPairwiseTopk(unittest.TestCase):
  def test_basic_topk(self):
    x = Tensor([[[1.0, 3.0, 2.0, 5.0, 4.0]]])
    vals, sel = pairwise_topk(x, 3)
    np.testing.assert_allclose(vals.numpy(), [[[3.0, 4.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 4, 3]]])

  def test_duplicates(self):
    x = Tensor([[[5.0, 5.0, 3.0, 5.0]]])
    vals, sel = pairwise_topk(x, 2)
    np.testing.assert_allclose(vals.numpy(), [[[5.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 0]]])

  def test_matches_numpy(self):
    np.random.seed(42)
    data = np.random.randn(4, 2, 16).astype(np.float32)
    vals, sel = pairwise_topk(Tensor(data), 5)
    for b in range(4):
      for t in range(2):
        expected = set(np.argsort(-data[b, t])[:5].tolist())
        self.assertEqual(set(sel.numpy()[b, t].tolist()), expected)
        np.testing.assert_allclose(vals.numpy()[b, t], data[b, t][sel.numpy()[b, t]])

  def test_ties_match_numpy_exactly(self):
    # tie-break: on equal scores the lower expert index ranks first. output is in ascending rank order.
    for seed in range(20):
      rng = np.random.default_rng(seed)
      for n, k in ((8, 3), (64, 4), (128, 8)):
        data = (rng.standard_normal((2, 3, n)) * 2).round().astype(np.float32) / 2  # quantized to force ties
        vals, sel = pairwise_topk(Tensor(data), k)
        expected = np.argsort(-data, axis=-1, kind='stable')[..., :k][..., ::-1]
        np.testing.assert_equal(sel.numpy(), expected)
        np.testing.assert_equal(vals.numpy(), np.take_along_axis(data, expected, -1))

  def test_kernel_count(self):
    # the routing cost of pairwise_topk is one kernel (the rank reduce): the select chain fuses into consumers like probs/expert gathers
    from test.helpers import check_schedule
    Tensor.manual_seed(0)
    x = Tensor.randn(1, 1, 64).realize()
    _, sel = pairwise_topk(x, 4)
    check_schedule([x.gather(-1, sel)], 2)

def _decode_attn_kernel(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
  """Naive single-kernel T=1 decode attention (T1.8 proof-of-concept for the attention_impl hook).

  Correct (numerically-stable) softmax, GQA-aware (kv head = query head // (n_heads/n_kv_heads), matching
  scaled_dot_product_attention's enable_gqa repeat_interleave convention), no tuning. Structure mirrors
  test/backend/test_custom_kernel.py's simple_qkv_kernel: one shared outer WEAK scope (b, h, dout), with
  three independent REG-scalar accumulate passes (max, sum-of-exp, weighted-output) nested underneath --
  REG placeholders are exempt from the codegen's memory-coalescing pass, which is what makes this simple
  "one register per accumulator" style reliable here (see the KernelInfo(opts_to_apply=()) note below).
  ponytail: the QK^T dot product is recomputed from scratch in all three passes (3x the FLOPs of a real
  flash-attention single-pass), and once per dout too (Hd more) -- O(Hd) redundant work vs a tuned kernel
  that caches per-row scores or does true online-softmax fusion. Upgrade path: T1.7/T1.8-tuned.
  """
  B, H, _, Hd = Q.shape
  KvH, Tk = K.shape[1], K.shape[2]
  R, scale = H // KvH, Hd ** -0.5

  b, h, dout = UOp.range(B, 0), UOp.range(H, 1), UOp.range(Hd, 2)
  kvh = h // R

  def qk_dot(j:UOp, slot:int, ready:UOp) -> UOp:
    d = UOp.range(Hd, 100 + slot, axis_type=AxisType.REDUCE)
    acc = UOp.placeholder((1,), dtypes.float32, slot, addrspace=AddrSpace.REG)
    acc = acc.after(ready, j)[0].set(0.0)
    acc = acc[0].set(acc.after(d)[0] + Q[b, h, 0, d].cast(dtypes.float32) * K[b, kvh, j, d].cast(dtypes.float32), end=d)
    return acc[0] * scale

  # pass 1: row max (for numerical stability)
  m = UOp.placeholder((1,), dtypes.float32, 10, addrspace=AddrSpace.REG)
  m = m.after(b, h, dout)[0].set(float("-inf"))
  jm = UOp.range(Tk, 3, axis_type=AxisType.REDUCE)
  m = m[0].set(m.after(jm)[0].maximum(qk_dot(jm, 11, m)), end=jm)

  # pass 2: sum of exp(score - m)
  l = UOp.placeholder((1,), dtypes.float32, 12, addrspace=AddrSpace.REG)
  l = l.after(m, dout)[0].set(0.0)
  jl = UOp.range(Tk, 4, axis_type=AxisType.REDUCE)
  l = l[0].set(l.after(jl)[0] + (qk_dot(jl, 13, l) - m[0]).exp(), end=jl)

  # pass 3: weighted output for this dout column
  jo = UOp.range(Tk, 5, axis_type=AxisType.REDUCE)
  acc_o = UOp.placeholder((1,), dtypes.float32, 14, addrspace=AddrSpace.REG)
  acc_o = acc_o.after(l, dout)[0].set(0.0)
  w = (qk_dot(jo, 15, acc_o) - m[0]).exp() / l[0]
  acc_o = acc_o[0].set(acc_o.after(jo)[0] + w * V[b, kvh, jo, dout].cast(dtypes.float32), end=jo)

  store = O[b, h, 0, dout].store(acc_o[0].cast(O.dtype))
  # NOTE: opts_to_apply=() is required -- without it the default auto-optimizer breaks the accumulate-loop
  # ->single-store fusion for anything past the simplest reduce shapes (silently wrong values, or a
  # "attempting multiple stores" codegen crash in tinygrad/codegen/late/coalesce.py). Sharp edge #1.
  return store.end(dout, h, b).sink(arg=KernelInfo(name=f"decode_attn_{B}_{H}_{Tk}_{Hd}", opts_to_apply=()))

def custom_decode_attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None) -> Tensor:
  """attention_impl-compatible wrapper around _decode_attn_kernel. Decode-only (T==1) and unmasked --
  anything else (prefill, sliding-window decode masks) falls back to the default SDPA path untouched."""
  if mask is not None or q.shape[2] != 1: return model._sdpa_default(q, k, v, mask)
  O = Tensor.empty(q.shape, dtype=q.dtype, device=q.device)
  return Tensor.custom_kernel(O, q, k, v, fxn=_decode_attn_kernel)[0]

class TestAttentionHook(unittest.TestCase):
  """T1.8: the attention_impl override hook + a naive Tensor.custom_kernel decode-attention proof."""

  def setUp(self):
    self._orig_impl = model.attention_impl
  def tearDown(self):
    model.attention_impl = self._orig_impl

  def _qkv(self, B, H, KvH, Tk, Hd, seed, T=1):
    Tensor.manual_seed(seed)
    q = Tensor.randn(B, H, T, Hd, dtype=dtypes.float32).contiguous().realize()
    k = Tensor.randn(B, KvH, Tk, Hd, dtype=dtypes.float32).contiguous().realize()
    v = Tensor.randn(B, KvH, Tk, Hd, dtype=dtypes.float32).contiguous().realize()
    return q, k, v

  def test_hook_default_matches_sdpa(self):
    # the hook's default must be exactly the plain SDPA expression it replaced
    q, k, v = self._qkv(1, 2, 2, 4, 8, 0)
    ref = q.scaled_dot_product_attention(k, v, enable_gqa=True)
    out = model.attention_impl(q, k, v, None)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)

  def test_custom_kernel_parity_no_gqa(self):
    q, k, v = self._qkv(1, 2, 2, 5, 8, 0)
    ref = q.scaled_dot_product_attention(k, v, enable_gqa=True).numpy()
    out = custom_decode_attention(q, k, v, None).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

  def test_custom_kernel_parity_gqa(self):
    for (B, H, KvH, Tk, Hd, seed) in [(2, 4, 2, 7, 8, 1), (1, 8, 2, 3, 16, 2), (3, 6, 1, 11, 4, 3), (1, 4, 1, 1, 8, 4)]:
      q, k, v = self._qkv(B, H, KvH, Tk, Hd, seed)
      ref = q.scaled_dot_product_attention(k, v, enable_gqa=True).numpy()
      out = custom_decode_attention(q, k, v, None).numpy()
      np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, err_msg=f"{B=} {H=} {KvH=} {Tk=} {Hd=}")

  def test_gating_prefill_falls_back(self):
    # T>1 (prefill/rollout): the decode-only kernel must not engage, output must match default SDPA exactly
    q, k, v = self._qkv(1, 4, 2, 5, 8, 0, T=3)
    mask = Tensor.full((1, 1, 3, 5), float("-inf"), dtype=dtypes.float32).triu(3)
    ref = model._sdpa_default(q, k, v, mask).numpy()
    out = custom_decode_attention(q, k, v, mask).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

  def test_gating_masked_decode_falls_back(self):
    # T==1 but masked (e.g. a sliding-window decode mask): still not decode-kernel-eligible, falls back
    q, k, v = self._qkv(1, 4, 2, 5, 8, 0, T=1)
    mask = Tensor.full((1, 1, 1, 5), float("-inf"), dtype=dtypes.float32).tril(1)
    ref = model._sdpa_default(q, k, v, mask).numpy()
    out = custom_decode_attention(q, k, v, mask).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

  def test_custom_kernel_symbolic_tk(self):
    # T4.7: Tensor.custom_kernel now accepts a symbolic (bound Variable) shape on a REDUCE-only dim.
    # _decode_attn_kernel never branches on Tk in Python (three UOp.range(Tk, ..., REDUCE) passes, no
    # unrolling), so -- unlike T1.8b's chunked kernel (see TestTunedAttentionKernel.
    # test_gating_symbolic_tk_falls_back) -- it needs no fallback: it just runs correctly at every Tk.
    q, k_full, v_full = self._qkv(1, 4, 2, 10, 8, 0)
    for n in (1, 3, 5, 10):
      tk_var = Variable("Tk", 1, 10).bind(n)
      k, v = k_full[:, :, :tk_var], v_full[:, :, :tk_var]
      self.assertFalse(isinstance(k.shape[2], int))
      ref = q.scaled_dot_product_attention(k_full[:, :, :n].contiguous(), v_full[:, :, :n].contiguous(), enable_gqa=True).numpy()
      out = custom_decode_attention(q, k, v, None).numpy()
      np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, err_msg=f"{n=}")

  def test_kernel_count_default_vs_hook(self):
    q, k, v = self._qkv(2, 4, 2, 6, 8, 0)
    GlobalCounters.reset()
    q.scaled_dot_product_attention(k, v, enable_gqa=True).realize()
    default_count = GlobalCounters.kernel_count  # informational: the ~5-kernel baseline (QK^T, max/sum/div, @V)
    self.assertGreater(default_count, 1)

    GlobalCounters.reset()
    custom_decode_attention(q, k, v, None).realize()
    assert_kernel_count(1)  # the hook + custom kernel collapse the whole attention chain to 1 dispatch

  def test_hook_wired_through_transformer_block_kv_cache(self):
    # end-to-end: the hook must work when q/k/v are the REAL views TransformerBlock._attention derives
    # from its growing, in-place-stored KV cache (an AFTER(store)+SHRINK chain, not a fresh tensor).
    config = TransformerConfig(num_blocks=1, dim=16, hidden_dim=32, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                                vocab_size=32, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=16)
    block = TransformerBlock(config)
    Tensor.manual_seed(0)
    prompt = Tensor.randn(1, 5, config.dim, dtype=dtypes.float32).contiguous().realize()
    decode_in = Tensor.randn(1, 1, config.dim, dtype=dtypes.float32).contiguous().realize()

    prompt_norm, decode_norm = block.attn_norm(prompt), block.attn_norm(decode_in)
    block._init_state(prompt_norm)
    block._attention(prompt_norm, 0).realize()  # prefill: fills cache_kv[0:5]
    ref = block._attention(decode_norm, 5).realize().numpy()  # decode step through cache-view SDPA

    model.attention_impl = custom_decode_attention
    out = block._attention(decode_norm, 5).realize().numpy()  # same decode step, hook active

    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

class TestTunedAttentionKernel(unittest.TestCase):
  """T1.8b: the tuned (LOCAL-cooperative, chunked, online-softmax) decode-attention kernel."""

  def setUp(self):
    self._orig_impl = model.attention_impl
  def tearDown(self):
    model.attention_impl = self._orig_impl

  def _qkv(self, B, H, KvH, Tk, Hd, seed, T=1, dtype=dtypes.float32):
    Tensor.manual_seed(seed)
    q = Tensor.randn(B, H, T, Hd, dtype=dtype).contiguous().realize()
    k = Tensor.randn(B, KvH, Tk, Hd, dtype=dtype).contiguous().realize()
    v = Tensor.randn(B, KvH, Tk, Hd, dtype=dtype).contiguous().realize()
    return q, k, v

  def test_parity_gqa(self):
    # covers: tail-only (Tk < CHUNK), exact multiple of CHUNK, full chunks + tail, MHA (KvH==H), Tk==1
    for (B, H, KvH, Tk, Hd, seed) in [
      (2, 4, 2, 7, 8, 1), (1, 8, 2, 3, 16, 2), (3, 6, 1, 11, 4, 3), (1, 4, 1, 1, 8, 4),
      (1, 8, 8, CHUNK, 8, 5), (1, 4, 1, 3 * CHUNK, 8, 6), (1, 4, 1, 3 * CHUNK + 5, 8, 7),
    ]:
      q, k, v = self._qkv(B, H, KvH, Tk, Hd, seed)
      ref = q.scaled_dot_product_attention(k, v, enable_gqa=True).numpy()
      out = tuned_decode_attention(q, k, v, None).numpy()
      np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, err_msg=f"{B=} {H=} {KvH=} {Tk=} {Hd=}")

  def test_parity_fp16_qkv(self):
    q, k, v = self._qkv(1, 8, 2, 2 * CHUNK + 3, 16, 0, dtype=dtypes.float16)
    ref = q.scaled_dot_product_attention(k, v, enable_gqa=True).numpy()
    out = tuned_decode_attention(q, k, v, None).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

  def test_gating_prefill_falls_back(self):
    q, k, v = self._qkv(1, 4, 2, 5, 8, 0, T=3)
    mask = Tensor.full((1, 1, 3, 5), float("-inf"), dtype=dtypes.float32).triu(3)
    ref = model._sdpa_default(q, k, v, mask).numpy()
    out = tuned_decode_attention(q, k, v, mask).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

  def test_gating_masked_decode_falls_back(self):
    q, k, v = self._qkv(1, 4, 2, 5, 8, 0, T=1)
    mask = Tensor.full((1, 1, 1, 5), float("-inf"), dtype=dtypes.float32).tril(1)
    ref = model._sdpa_default(q, k, v, mask).numpy()
    out = tuned_decode_attention(q, k, v, mask).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

  def test_symbolic_tk_uses_kernel(self):
    # T1.8c: the old HARD LIMITER (Tensor.custom_kernel hard-asserting on symbolic shapes, forcing this
    # kernel to fall back to SDPA past the first real decode step) is resolved -- the chunk-round count is
    # now ceildiv(Tk, CHUNK) fed straight into a REDUCE UOp.range (see attn_kernel.py docstring), so there's
    # no Python branch on Tk's value left to fail. This replaces the old test_gating_symbolic_tk_falls_back:
    # the contract inverted -- it must now assert the kernel HANDLES symbolic Tk, not that it declines to.
    # Covers both an exact-multiple-of-CHUNK bound value and non-multiples (tail-guard exercised).
    q, k_full, v_full = self._qkv(1, 4, 2, 3 * CHUNK + 5, 8, 0)
    for n in (1, CHUNK, CHUNK + 1, 2 * CHUNK, 3 * CHUNK + 5):
      tk_var = Variable("Tk", 1, 3 * CHUNK + 5).bind(n)
      k, v = k_full[:, :, :tk_var], v_full[:, :, :tk_var]
      self.assertFalse(isinstance(k.shape[2], int))  # sanity: this really is the symbolic case
      ref = q.scaled_dot_product_attention(k_full[:, :, :n].contiguous(), v_full[:, :, :n].contiguous(), enable_gqa=True).numpy()
      out = tuned_decode_attention(q, k, v, None).numpy()
      np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, err_msg=f"{n=}")

  def test_symbolic_tk_jit_replay_one_kernel(self):
    # mirrors test/backend/test_custom_kernel.py's test_sum_symbolic_reduce_dim_jit: one compiled kernel,
    # captured once, correctly replayed across many bound Tk values under TinyJit (including a value never
    # seen during capture) -- the variable identity is the cache key, not the bound value.
    q, k_full, v_full = self._qkv(1, 4, 2, 3 * CHUNK + 5, 8, 0)
    def f(q, k, v):
      return tuned_decode_attention(q, k, v, None).realize()
    jf = TinyJit(f)
    for n in (1, CHUNK, 2 * CHUNK, CHUNK + 1, 3 * CHUNK + 5, CHUNK + 1):
      tk_var = Variable("Tk", 1, 3 * CHUNK + 5).bind(n)
      k, v = k_full[:, :, :tk_var], v_full[:, :, :tk_var]
      ref = q.scaled_dot_product_attention(k_full[:, :, :n].contiguous(), v_full[:, :, :n].contiguous(), enable_gqa=True).numpy()
      out = jf(q, k, v).numpy()
      np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, err_msg=f"{n=}")
    # 3, not 1: a SHRINK view on a middle axis (k/v sliced to :tk_var out of a larger max-extent buffer,
    # exactly what the real KV cache slice looks like) isn't a simple stride-offset view -- there are gaps
    # between (b, kv_head) blocks past Tk -- so custom_kernel's dispatch needs k and v each materialized
    # into a plain contiguous buffer first. Same cost the naive T1.8 kernel pays (verified directly: T4.7's
    # placeholder_like machinery, not this kernel's chunking, is what triggers it) and one real decode was
    # already paying pre-T1.8c on every concrete-Tk step too -- symbolic Tk doesn't add it, just no longer
    # falls back around it. The number that matters here is "one capture, not one per n" -- assert that.
    assert_jit_cache_len(jf, 3)

  def test_kernel_count_default_vs_tuned(self):
    q, k, v = self._qkv(2, 4, 2, 2 * CHUNK + 3, 8, 0)
    GlobalCounters.reset()
    q.scaled_dot_product_attention(k, v, enable_gqa=True).realize()
    self.assertGreater(GlobalCounters.kernel_count, 1)

    GlobalCounters.reset()
    tuned_decode_attention(q, k, v, None).realize()
    assert_kernel_count(1)

  def test_hook_wired_through_transformer_block_kv_cache_both_kv_dtypes(self):
    # end-to-end through TransformerBlock's real (AFTER(store)+SHRINK) cache views, for both KV cache
    # dtypes T1.1a supports (fp16 default, fp32 via KV_F32=1) -- set directly on cache_kv to avoid
    # getenv's process-wide caching (see kv_cache_dtype()) rather than relying on the env var.
    config = TransformerConfig(num_blocks=1, dim=16, hidden_dim=32, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                                vocab_size=32, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=16)
    for kv_dtype in (dtypes.float16, dtypes.float32):
      block = TransformerBlock(config)
      Tensor.manual_seed(0)
      prompt = Tensor.randn(1, 5, config.dim, dtype=dtypes.float32).contiguous().realize()
      decode_in = Tensor.randn(1, 1, config.dim, dtype=dtypes.float32).contiguous().realize()

      prompt_norm, decode_norm = block.attn_norm(prompt), block.attn_norm(decode_in)
      block._init_state(prompt_norm)  # sets freqs_cis; overwrite cache_kv below to force the kv_dtype under test
      block.cache_kv = Tensor.empty(2, 1, config.n_kv_heads, config.max_context, config.head_dim, dtype=kv_dtype)

      model.attention_impl = model._sdpa_default
      block._attention(prompt_norm, 0).realize()  # prefill: fills cache_kv[0:5]
      ref = block._attention(decode_norm, 5).realize().numpy()  # decode step through cache-view SDPA

      model.attention_impl = tuned_decode_attention
      out = block._attention(decode_norm, 5).realize().numpy()  # same decode step, tuned kernel active

      np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2, err_msg=f"{kv_dtype=}")

if __name__ == '__main__':
  unittest.main()
