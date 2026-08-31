from __future__ import annotations
import enum, functools, itertools, math, pathlib
from dataclasses import dataclass, replace
from typing import Callable, cast, TYPE_CHECKING
if TYPE_CHECKING: import numpy as np  # T4.65 CI fix: tinygrad's core stays numpy-free at import time -- the minimal
# CI lanes (Test LLM, SPEC=2) have no numpy installed. The sampled speculative path imports it lazily inside the
# functions that actually need it; annotations are lazy via `from __future__ import annotations`.
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.dtype import DType
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve, Ops
from tinygrad.helpers import ContextVar

# T4.55: prefill chunk width for recurrent models on devices without a fused scan kernel (see generate). 0 = auto: 32 on the GPU
# backends it was measured on -- METAL (qwen3.5:0.8b: 77 -> 168 tok/s no-BEAM, 99 -> 346 BEAM'd) and NV (the METAL+NV pooled
# qwen3.6-35B Q8_0: 46 -> 158-173 tok/s BEAM'd, decode unchanged) -- and 1 (the pre-T4.55 one-token-per-step prefill) elsewhere:
# the unrolled 32-step scan is one huge kernel, and x86 clang 18 crashes compiling it (CI's CPU "Test LLM" job on qwen3.5:0.8b).
# 64 falls off a cliff even on METAL (28 tok/s). Set GDN_CHUNK explicitly to override either way.
GDN_CHUNK = ContextVar("GDN_CHUNK", 0)
def gdn_chunk_for(device:str|tuple[str, ...]|None) -> int:
  if GDN_CHUNK.value > 0: return GDN_CHUNK.value
  dev = (device[0] if isinstance(device, tuple) else device) or Device.DEFAULT
  return 32 if dev.split(":")[0] in ("METAL", "NV", "CUDA") else 1

# T4.62: on wide-head geometries (e.g. qwen3.8-27B, num_v_heads=48) the unrolled per-t scan in
# GatedDeltaNetBlock._attention's else-branch -- one python loop building the whole chunk's graph, fused into
# a single kernel by its final .contiguous() -- lowers to more UOps than BEAM_UOPS_MAX (codegen/opt/search.py)
# for every BEAM candidate; narrower geometries (e.g. qwen3.6-35B's 32 heads) fit fine. The scan has no
# cross-head dependency (each head's (V,K) state slice evolves independently), so split it into G sequential
# head groups, each narrow enough to lower under the cap on its own -- see gdn_headgroup_evidence.py.
GDN_HEAD_GROUPS = ContextVar("GDN_HEAD_GROUPS", 0)
def gdn_head_groups_for(num_v_heads:int) -> int:
  if GDN_HEAD_GROUPS.value > 0: return GDN_HEAD_GROUPS.value
  return -(-num_v_heads // 32)  # auto: smallest G with ceil(num_v_heads/G) <= 32 == ceil(num_v_heads/32)

# T4.63: qwen3.5-family GGUFs carry a DeepSeek-style MTP ("nextn") block beyond the main num_blocks
# (nextn_predict_layers, already excluded from num_blocks -- see from_gguf). 0 (default): today's
# behavior, byte-identical -- gguf.py parses the block, from_gguf drops it (unused-weights warning).
# 1: build + load it as Transformer.mtp_head (see MTPHead) -- loading + forward only; T4.64 wires it
# into speculative decoding.
MTP = ContextVar("MTP", 0)

def kv_cache_dtype() -> DType:
  """Attention/MLA KV cache dtype: fp16 by default (halves the cache that scales with max_context --
  the dominant decode-memory cost). KV_F32=1 reverts to dtypes.default_float, e.g. to isolate an accuracy
  regression. Does NOT apply to GatedDeltaNetBlock's recurrent state -- see its _init_state for why."""
  return dtypes.default_float if getenv("KV_F32", 0) else dtypes.float16

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None, yarn_factor: float = 1.0,
                          yarn_orig_ctx: int = 0, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0,
                          yarn_attn_factor: float|None = None) -> Tensor:
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  attn_scale = 1.0
  if yarn_factor > 1.0:
    # YaRN NTK-by-parts frequency interpolation (https://arxiv.org/abs/2309.00071), verified against
    # transformers' _compute_yarn_parameters (modeling_rope_utils.py): low/high bound the "correction
    # range" (in rotary-dim index space) where we blend extrapolated (raw) and interpolated (/factor) freqs.
    def find_dim(num_rot:float) -> float: return (dim * math.log(yarn_orig_ctx / (num_rot * 2 * math.pi))) / (2 * math.log(theta))
    low = float(max(math.floor(find_dim(yarn_beta_fast)), 0))
    high = float(min(math.ceil(find_dim(yarn_beta_slow)), dim // 2 - 1))
    if low == high: high += 0.001
    extrap_factor = 1.0 - ((Tensor.arange(dim // 2) - low) / (high - low)).clamp(0, 1)
    inv_freq = inv_freq / yarn_factor * (1 - extrap_factor) + inv_freq * extrap_factor
    attn_scale = yarn_attn_factor if yarn_attn_factor is not None else (0.1 * math.log(yarn_factor) + 1.0 if yarn_factor > 1 else 1.0)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * inv_freq.unsqueeze(dim=0)
  return (freqs.cos() * attn_scale).cat(freqs.sin() * attn_scale, dim=-1).clone(device)

@functools.cache
def positions(n:int, device:str|None=None) -> Tensor: return Tensor.arange(n, dtype=dtypes.int32).clone(device).realize()

def causal_mask(pos:Tensor, T:int|UOp, start_pos:int|UOp, dtype:DType, sliding_window:int=0) -> Tensor:
  """(1,1,T,start_pos+T) additive mask over the KV cache: 0 where query row i may see key j (j <= start_pos+i; and j > start_pos+i-window
  when sliding), -inf elsewhere. Built from slices of a realized position buffer: the Tensor.full(...).triu(...) form it replaces ran
  arange(start_pos+T), and for a symbolic length that lowers to the single-stage cumsum -- an N x N reduce with no inputs, per
  attention layer per prefill chunk (T4.58, 3090: 27 us at N=64, 3.5 ms at 4k, 48 ms at 16k -- the whole long-context prefill
  cost of the pooled qwen3.6-35B). `pos` comes from FFNBlock._positions: max_context + the chunk variable's declared max entries,
  because the SYMBOLIC bound of start_pos+T is (max_context-1)+chunk_size -- past a max_context-sized buffer -- and CHECK_OOB's z3
  check (on in CI) rejects a load it cannot prove in-bounds even though generate() never runs there."""
  d = pos[:start_pos+T].reshape(1, 1, 1, start_pos+T) - (pos[:T] + Tensor(start_pos, device=pos.device)).reshape(1, 1, T, 1)  # key pos - query pos
  mask = (d > 0).where(float("-inf"), 0.0)
  if sliding_window: mask = mask + (d <= -sliding_window).where(float("-inf"), 0.0)
  return mask.cast(dtype)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int, bias:bool=False):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    if bias: self.bias = Tensor.zeros(num_experts, out_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    out = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return out + self.bias[sel] if hasattr(self, 'bias') else out

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  attn_out_bias: bool = False
  router_bias: bool = False
  moe_bias: bool = False
  attn_sinks: bool = False
  sliding_window: int = 0
  sliding_layers: tuple[bool, ...] = ()
  clamp_swiglu: bool = False
  swiglu_limit: float = 7.0
  swiglu_alpha: float = 1.702
  yarn_factor: float = 1.0
  yarn_orig_ctx: int = 0
  yarn_beta_fast: float = 32.0
  yarn_beta_slow: float = 1.0
  yarn_attn_factor: float|None = None

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.router_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.moe_bias)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.moe_bias)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim, bias=config.moe_bias)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      # routed experts may live on a different device than attention/router (device_map "experts:<dev>"); the
      # weights never move (placed once at load), only these activations hop, around the three ExpertWeights
      # calls. sel must travel with h: self.weight[sel] indexes the (now-remote) weight buffer with it, so both
      # operands of that gather need to be on the same device. .to() is a no-op when already co-located.
      expert_dev = self.ffn_gate_exps.weight.device
      h, sel = h.to(expert_dev), sel.to(expert_dev)
      gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      if self.config.clamp_swiglu:
        # gpt-oss clamped swiglu: gate*sigmoid(alpha*gate) * (up+1), both branches clamped
        gate, up = gate.clamp(max_=self.config.swiglu_limit), up.clamp(-self.config.swiglu_limit, self.config.swiglu_limit)
        act = gate * (self.config.swiglu_alpha * gate).sigmoid() * (up + 1)
      else:
        act = gate.silu() * up
      x_down = self.ffn_down_exps(sel, act.contiguous()).to(x.device)  # (B, T, k, D), hop back to the block device
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  @property
  def device(self) -> str|tuple[str, ...]|None:
    assert self.attn_norm.weight is not None
    return self.attn_norm.weight.device

  def _positions(self, x:Tensor) -> Tensor:
    # max_context + this call's chunk max (x.max_shape[1]: the chunk variable's declared max, stable across JIT replays) makes
    # every symbolic slice in causal_mask provably in-bounds; cached per size + device, realized by _init_state (see there)
    return positions(self.config.max_context + x.max_shape[1], x.device)

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

# --- attention-override hook (T1.8) -----------------------------------------------------------
# Pluggable seam for a fused/tuned attention kernel to replace the standard (multi-kernel) SDPA
# expression below. Swap this module attribute (e.g. `tinygrad.llm.model.attention_impl = my_fn`)
# to route TransformerBlock's *standard* attention path through a custom implementation. Two
# paths intentionally do NOT go through this hook and are unaffected by swapping it:
#   - the manual attention-sinks softmax (gpt-oss, config.attn_sinks) in TransformerBlock itself
#   - MLATransformerBlock / GatedDeltaNetBlock, which compute attention inline, not via SDPA
# Signature matches Tensor.scaled_dot_product_attention's shape contract:
#   q:(B,H,T,Hd)  k,v:(B,KvH,Tk,Hd)  mask:(1,1,T,Tk)|None  ->  (B,H,T,Hd)
# A custom impl that only handles some shapes (e.g. decode-only, T==1, unmasked) must fall back
# to `_sdpa_default` itself for everything else (prefill, sliding-window masks, etc).
def _sdpa_default(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None) -> Tensor:
  return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)

attention_impl: Callable[[Tensor, Tensor, Tensor, Tensor|None], Tensor] = _sdpa_default

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_out_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    # cast to the cache's dtype at write (a no-op when KV_F32=1); cast back up to the activation dtype at
    # read, so attention compute always runs at x's precision regardless of what the cache stores
    assigned_kv = Tensor(self.cache_kv.uop.after(
      self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)))
    # on RDNA3, hybrid models use custom flash attention kernels directly on the (compressed) KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :].cast(x.dtype)
    v = assigned_kv[1, :, :, 0:start_pos+T, :].cast(x.dtype)

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    # sliding-window layers also need a mask at T==1 (decode), to drop cache entries older than the window
    mask = causal_mask(self._positions(x), T, start_pos, x.dtype, self.config.sliding_window) \
      if resolve(T != 1) or self.config.sliding_window else None

    if hasattr(self, "attn_sinks"):
      # attention sinks: a learned per-head logit that only contributes to the softmax denominator
      # (no value contribution), competing with real keys for probability mass. Can't express this via
      # scaled_dot_product_attention's attn_mask (that only biases real score entries), so do it manually.
      KV, R = self.config.n_kv_heads, self.config.n_heads // self.config.n_kv_heads
      qg, kg, vg = q.reshape(B, KV, R, T, self.config.head_dim), k.unsqueeze(2), v.unsqueeze(2)
      scores = (qg.cast(dtypes.float32) @ kg.cast(dtypes.float32).transpose(-1, -2)) * (1.0 / self.config.head_dim ** 0.5)
      if mask is not None: scores = scores + mask.unsqueeze(1)
      sink = self.attn_sinks["weight"].reshape(1, KV, R, 1, 1).cast(dtypes.float32)
      m = scores.max(-1, keepdim=True).maximum(sink)
      e = (scores - m).exp()
      w = (e / (e.sum(-1, keepdim=True) + (sink - m).exp())).cast(x.dtype)
      attn = (w @ vg.cast(x.dtype)).reshape(B, self.config.n_heads, T, self.config.head_dim)
    else:
      attn = attention_impl(q, k, v, mask)                                            # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    # every call, not just the first: a new chunk size needs its own (larger) position buffer, and it must be realized HERE --
    # __call__ runs _init_state before the @function(precompile=True) trace, inside which nothing may touch a device
    self._positions(x)
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=kv_cache_dtype(), device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device,
        yarn_factor=self.config.yarn_factor, yarn_orig_ctx=self.config.yarn_orig_ctx, yarn_beta_fast=self.config.yarn_beta_fast,
        yarn_beta_slow=self.config.yarn_beta_slow, yarn_attn_factor=self.config.yarn_attn_factor)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    # cast to the cache's dtype at write, back up to x's dtype at read -- see TransformerBlock._attention
    k = Tensor(self.cache_k.uop.after(
      self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.cast(self.cache_k.dtype).uop)))[:, :, 0:start_pos+T, :].cast(x.dtype)
    v = k[..., :self.config.kv_lora_rank]

    mask = causal_mask(self._positions(x), T, start_pos, x.dtype) if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    self._positions(x)  # see TransformerBlock._init_state
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim,
                                  dtype=kv_cache_dtype(), device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device,
        yarn_factor=self.config.yarn_factor, yarn_orig_ctx=self.config.yarn_orig_ctx, yarn_beta_fast=self.config.yarn_beta_fast,
        yarn_beta_slow=self.config.yarn_beta_slow, yarn_attn_factor=self.config.yarn_attn_factor)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos, device=x.device).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels, device=x.device).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-1)
      state = initial.where(0, state.float())

      def scan(state:Tensor, q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor) -> tuple[Tensor, Tensor]:
        outs = []
        for t in range(T_pad):
          s1 = state * alpha[:, :, t]  # decay the state
          delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
          state = s1 + delta * k[:, :, t]
          outs.append((state * q[:, :, t]).sum(-1))
        return state, outs[0].stack(*outs[1:], dim=1)

      # T4.62: split the scan into G sequential head groups when the geometry needs it (see
      # gdn_head_groups_for above) -- each group's own .contiguous() forces a separate, narrower kernel,
      # then a single cat reassembles the full state/output. G=1 (unchanged geometries) skips all of this
      # and lowers to exactly the same graph as before.
      if (G := gdn_head_groups_for(self.num_v_heads)) <= 1:
        state, stacked = scan(state, q, k, v, beta, alpha)
      else:
        gsize = -(-self.num_v_heads // G)  # ceil(H/G): only the last group is smaller when H % G != 0
        group_states, group_outs = [], []
        for lo in range(0, self.num_v_heads, gsize):
          hi = min(lo + gsize, self.num_v_heads)
          g_state, g_outs = scan(state[:, lo:hi], q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], beta[:, lo:hi], alpha[:, lo:hi])
          group_states.append(g_state.contiguous())
          group_outs.append(g_outs.contiguous())
        state = group_states[0].cat(*group_states[1:], dim=1) if len(group_states) > 1 else group_states[0]
        stacked = group_outs[0].cat(*group_outs[1:], dim=2) if len(group_outs) > 1 else group_outs[0]

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      core = Tensor(stacked.contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      # conv_state only ever holds the last (conv_kernel-1) input rows (already upcast to fp32 on read via
      # win.dtype in _attention above) and doesn't scale with max_context, but it's free to halve too: an
      # evidence run (tiny random-weight config, 5 prompts x 64 decode steps, isolated from recurrent_state
      # by keeping recurrent_state fp32) showed 0/320 greedy-token divergences, max logit delta ~0.003 --
      # flag-gate it like the KV caches.
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, dtype=kv_cache_dtype(), device=x.device).clone()
      # recurrent_state is NOT flag-gated -- always fp32, unlike the write-once/read-many KV caches above it
      # is read-modify-written every decode step (decay + delta rule), so fp16 error compounds across the
      # whole generation instead of staying local to one position. Evidence (same tiny config/harness, fp16
      # recurrent_state isolated from conv_state -- conv_state fp32): 7/320 greedy tokens flipped (2 of 5
      # prompts affected) with max logit delta ~2.9 -- vs 0/320 and ~0.003 for every KV-cache-like buffer
      # above (TransformerBlock.cache_kv, MLATransformerBlock.cache_k, this block's own conv_state). It's
      # also O(1) in max_context (no memory win from halving it), so there's no upside to offset that risk.
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim,
                                          dtype=dtypes.default_float, device=x.device).clone()

def parse_device_map(dm:str|dict[int|str,str], num_blocks:int) -> tuple[list[str], str|None]:
  """Per-block device: "0-15:CPU:0,16-31:CPU:1" (inclusive ranges), "CPU:0,CPU:1" (even split), or {block_idx: device}.
  An optional "experts:<device>" segment (str form) or "experts" key (dict form) routes MoE routed-expert weight
  tensors (ffn_{gate,up,down}_exps) to a separate device, independent of their block's device. The router
  (ffn_gate_inp) always stays with its block. Returns (per-block devices, experts device or None)."""
  if isinstance(dm, dict):
    experts_dev = dm.get("experts")
    blocks = {k: v for k, v in dm.items() if k != "experts"}
    assert all(i in blocks for i in range(num_blocks)), f"device_map must cover all {num_blocks} blocks: {dm}"
    return [blocks[i] for i in range(num_blocks)], experts_dev

  parts = [s.strip() for s in dm.split(",")]
  experts_parts = [p for p in parts if p.startswith("experts:")]
  assert len(experts_parts) <= 1, f"device_map has more than one 'experts:' segment: {dm}"
  experts_dev = experts_parts[0].split(":", 1)[1] if experts_parts else None
  assert experts_dev != "", f"device_map 'experts:' segment needs a device: {dm}"
  block_dm = ",".join(p for p in parts if not p.startswith("experts:"))
  assert block_dm, f"device_map has no block segments (only 'experts:'): {dm}"

  segs = [s.strip().split(":", 1) for s in block_dm.split(",")]
  indexed = [len(s) == 2 and s[0].replace("-", "").isdigit() for s in segs]
  if not any(indexed):
    # no free-memory query exists on Device/Allocator, so auto placement splits evenly by block count
    devs = [s.strip() for s in block_dm.split(",")]
    assert len(devs) <= num_blocks, f"device_map lists {len(devs)} devices for only {num_blocks} blocks: {dm}"
    return [devs[i * len(devs) // num_blocks] for i in range(num_blocks)], experts_dev
  assert all(indexed), f"device_map mixes indexed ('lo[-hi]:device') and plain segments: {dm}"
  out: list[str|None] = [None] * num_blocks
  for rng, dev in segs:
    lo_s, _, hi_s = rng.partition("-")
    lo, hi = int(lo_s), int(hi_s or lo_s)
    assert 0 <= lo <= hi < num_blocks, f"device_map range {rng} out of bounds for {num_blocks} blocks: {dm}"
    assert all(out[i] is None for i in range(lo, hi+1)), f"device_map range {rng} overlaps a previous range: {dm}"
    for i in range(lo, hi+1): out[i] = dev
  assert all(out), f"device_map must cover all {num_blocks} blocks: {dm}"
  return out, experts_dev  # type: ignore[return-value]

def _rename_mtp_keys(state_dict:dict[str, Tensor], num_blocks:int) -> None:
  """Renames blk.{num_blocks}.* (the nextn/MTP block a qwen3.5-family GGUF carries beyond the main
  num_blocks) onto MTPHead's attribute paths, in place -- see MTPHead's docstring for the forward
  semantics. blk.{n}.nextn.X -> mtp_head.X (the 4 MTP-specific tensors); every other blk.{n}.X ->
  mtp_head.block.X (X unchanged: the inner block's attribute names already match a main block's, e.g.
  attn_q, ffn_gate -- from_gguf's post_attention_norm->ffn_norm rename runs first and applies here too).
  Pure key manipulation -- never touches a tensor's value, so this is cheap to exercise against just a
  real GGUF's tensor-name list (no data realized) as well as against a real load.
  # ponytail: only the first nextn layer (blk.{num_blocks}) is modeled -- nextn_predict_layers>1 (not
  # seen in any real qwen3.5-family checkpoint so far) would need mtp_head to be a list of these."""
  prefix = f"blk.{num_blocks}."
  for k in [k for k in state_dict if k.startswith(prefix)]:
    rest = k[len(prefix):]
    new_k = f"mtp_head.{rest[len('nextn.'):]}" if rest.startswith("nextn.") else f"mtp_head.block.{rest}"
    state_dict[new_k] = state_dict.pop(k)

class MTPHead:
  """DeepSeek-style multi-token-prediction ("nextn") head (qwen3.5-family GGUFs -- see from_gguf's
  MTP=1 branch and _rename_mtp_keys for how blk.{num_blocks}.* lands here). llama.cpp-compatible
  forward: eh_proj(concat(enorm(embed(next_tok)), hnorm(last_hidden))) -> `block` (a real attention
  block, built the same way the main Transformer builds one for this arch/config -- see from_gguf;
  it keeps its OWN KV cache over drafted positions, separate from the main model's blocks) ->
  shared_head_norm -> the OWNING Transformer's output head. That last step is output-norm-free: this
  head has its own learned final norm (shared_head_norm) instead of reusing the main model's
  output_norm, same as llama.cpp's DeepSeek-MTP implementation. One token of lookahead per call;
  T4.64 chains calls (start_pos, start_pos+1, ...), feeding each call's own sampled token back in.

  `draft` takes the owning Transformer as an explicit argument rather than storing it on self: this
  object hangs off Transformer.mtp_head, and nn/state.py's get_state_dict recurses through every plain
  attribute with no cycle guard -- a stored back-reference would make it walk
  model -> mtp_head -> owner -> model -> ... forever the moment anything asks for this model's
  state/parameters (load_state_dict, realize_placement, get_parameters, ...)."""
  def __init__(self, config:TransformerConfig, block_cls:type[FFNBlock]=TransformerBlock):
    self.enorm, self.hnorm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.eh_proj = Linear(2 * config.dim, config.dim, bias=False)
    self.block = block_cls(config)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)

  def draft(self, owner:Transformer, h:Tensor, tok_ids:Tensor, start_pos:int|UOp) -> Tensor:
    """h: (B,1,dim) the main model's last-layer hidden state (pre-output-norm) at position t.
    tok_ids: (B,1) int, the token actually at position t+1. Returns (B,1,vocab) draft logits
    predicting position t+2. `block`'s KV cache advances with start_pos across calls, same as a
    main block's (see FFNBlock._init_state/__call__) -- so calling this again at start_pos+1
    continues the same drafted sequence instead of overwriting it."""
    dev = self.block.device
    e = owner.token_embd(tok_ids.to(owner.token_embd.weight.device)).float().to(dev)
    x = self.eh_proj(self.enorm(e).cat(self.hnorm(h.to(dev).float()), dim=-1))
    x = self.block(x, start_pos)
    return owner.output(self.shared_head_norm(x).to(owner.output.weight.device))

def _softmax_np(logits:np.ndarray, temperature:float) -> np.ndarray:
  """Numerically-stable softmax(logits/temperature), float64. speculative_generate's sampled path (T4.65)
  pulls model logits to host and does all its acceptance math in numpy (the vectors are tiny -- at most
  k_eff+1 rows of vocab_size floats per iteration); float64 gives rng.choice's sum-to-1 tolerance headroom
  a fp16/fp32 model output doesn't reliably leave it."""
  import numpy as np
  z = logits.astype(np.float64) / temperature
  z = z - z.max(axis=-1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(axis=-1, keepdims=True)

def spec_accept(draft_ids:list[int], q_probs:np.ndarray, p_probs:np.ndarray, rng:np.random.Generator) -> tuple[list[int], int]:
  """Leviathan et al. ("Fast Inference from Transformers via Speculative Decoding", 2023) speculative-sampling
  accept/resample test -- pure, host-side (numpy; see _softmax_np), used by speculative_generate's
  temperature>0 path (T4.65 -- see SPEC_NOTES.md §6 for how this wires into the position ledger).

  draft_ids: the k_eff drafted token ids. THE CALLER must have drawn them by ancestral sampling from
  q_probs (draft_ids[i] ~ q_probs[i]) -- this function only implements the accept/reject/resample test; the
  distribution-preservation proof below assumes, but does not check, that draft_ids came from q_probs.
  q_probs: (k_eff, vocab) draft softmax at each drafted position, softmax(draft_logits/temperature).
  p_probs: (k_eff+1, vocab) verify (main-model) softmax at each of the k_eff+1 verify positions,
  softmax(verify_logits/temperature) -- row k_eff is the "bonus" position, sampled only on full accept.

  Returns (accepted_ids, m): m in 0..k_eff is how many drafted tokens were accepted; accepted_ids has length
  m+1 (== draft_ids[:m] + one more token): on full accept (m==k_eff) that extra token is a fresh sample from
  p_probs[k_eff] (the bonus position); on rejection at position m (m<k_eff) it's a resample from
  normalize(max(0, p_probs[m]-q_probs[m])) (the "residual" distribution) instead of the rejected draft_ids[m].

  Proof this reproduces p exactly (Leviathan et al., Theorem 1): accept x with probability min(1,p(x)/q(x)),
  so P(emit x, accepted) = q(x)*min(1,p(x)/q(x)) = min(p(x),q(x)). P(reject) = 1 - sum_y min(p(y),q(y)), and
  sum_y max(0,p(y)-q(y)) = sum_y [p(y) - min(p(y),q(y))] = 1 - sum_y min(p(y),q(y)) = P(reject) too, so
  normalize(max(0,p-q)) is a valid distribution and P(emit x, rejected) = P(reject) * max(0,p(x)-q(x))/P(reject)
  = max(0,p(x)-q(x)). Total: P(emit x) = min(p(x),q(x)) + max(0,p(x)-q(x)) = p(x) (either p(x)<=q(x), giving
  p(x)+0, or p(x)>q(x), giving q(x)+(p(x)-q(x))) -- true for every x regardless of q, which is why draft
  quality only ever affects the acceptance RATE, never correctness. test_spec_decode.py's statistical test
  checks this empirically since it's the theorem the whole sampled path's correctness rests on."""
  import numpy as np
  assert q_probs.shape[0] == len(draft_ids) and p_probs.shape[0] == len(draft_ids) + 1, "q_probs/p_probs row count must match draft_ids"
  k_eff = len(draft_ids)
  for m in range(k_eff):
    d = draft_ids[m]
    accept_prob = min(1.0, p_probs[m, d] / max(q_probs[m, d], 1e-12))
    if rng.random() < accept_prob: continue
    residual = np.clip(p_probs[m] - q_probs[m], 0, None)
    total = residual.sum()
    correction = int(rng.choice(len(residual), p=residual / total)) if total > 0 else int(np.argmax(p_probs[m]))
    return draft_ids[:m] + [correction], m
  bonus = int(rng.choice(p_probs.shape[1], p=p_probs[k_eff]))
  return draft_ids + [bonus], k_eff

class Transformer:
  def __init__(self, config:TransformerConfig, device_map:str|dict[int|str,str]|None=None):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    def _cfg(i:int) -> TransformerConfig:
      cfg = dense_config if i < config.leading_dense_blocks else config
      # sliding_layers marks which layers use the windowed attention span; others get the full-context mask
      if config.sliding_layers and not config.sliding_layers[i]: cfg = replace(cfg, sliding_window=0)
      return cfg
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(_cfg(i), config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(_cfg(i))
                               for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.mtp_head: MTPHead|None = None  # set by from_gguf when MTP=1 and the GGUF has a nextn block (T4.63)
    self.max_context = config.max_context
    # set by the device_map branch below; None means "no device_map", the fast/no-op path for realize_placement()
    self._placed_devices: frozenset[str]|None = None
    if device_map is not None:
      dmap, experts_dev = parse_device_map(device_map, config.num_blocks)
      for block, dev in zip(self.blk, dmap):
        for p in nn.state.get_parameters(block): p.to_(dev)
      # token_embd feeds the first block; output_norm/output consume the last block's activations
      for p in nn.state.get_parameters(self.token_embd): p.to_(dmap[0])
      for p in nn.state.get_parameters([self.output_norm, self.output]): p.to_(dmap[-1])
      # routed-expert weights (not the router, which stays with attention) move to a separate device on
      # top of the per-block placement above -- overrides ffn_{gate,up,down}_exps' dev for every MoE block
      if experts_dev is not None:
        for block in self.blk:
          if hasattr(block, 'ffn_gate_exps'):
            for p in nn.state.get_parameters([block.ffn_gate_exps, block.ffn_up_exps, block.ffn_down_exps]): p.to_(experts_dev)
      # canonicalized so it compares equal to p.device later (e.g. "CPU:0" -> "CPU") -- realize_placement()'s footgun guard
      self._placed_devices = frozenset(Device.canonicalize(d) for d in dmap + ([experts_dev] if experts_dev is not None else []))
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill/rollout, sampled/greedy, and spec (T4.64); prefill also keys on
    # chunk_size (T4.12) -- created lazily in __call__ since chunk_size isn't known until generate() picks one
    self.jit: dict[tuple[bool, bool, int|None, bool], Callable[..., Tensor|tuple[Tensor, Tensor]]] = {}

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, spec:bool=False) -> Tensor|tuple[Tensor, Tensor]:
    # contract: temperature=None is the ONLY greedy trigger. it's a python-level check (not a value check) because it
    # picks which jit variant gets captured (with or without RNG kernels) -- a Tensor of value 0.0 still takes the
    # sampled path below. callers must normalize temp<=0 to None themselves (generate() already does this)
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()  # (B, T, D)
    # activations hop devices at block boundaries (.to is a no-op when the device matches)
    for block in self.blk: x = block(x.to(block.device), start_pos)
    if spec:
      # T4.64: speculative_generate's verify/prefill-tail path needs (a) the model's per-position output (to
      # check a chained MTP draft against what the main model actually says there) and (b) the pre-output_norm
      # hidden state at the last position (to seed the next draft chain -- the same `x[:, -1:]` the default path
      # below slices before norm+head). Applying output_norm+output to every position instead of just the last
      # is strictly more work, never less, and this whole branch is behind a flag every EXISTING caller leaves
      # False -- so the default path below stays byte-for-byte what it was.
      # T4.65: this used to return the per-position ARGMAX id (greedy-only). It now returns the raw per-position
      # LOGITS instead -- speculative_generate derives the greedy id from them (argmax, moved to the caller,
      # same value either way) and, for its sampled-acceptance path (temperature>0), the full softmax
      # distributions Leviathan et al.'s accept/resample test needs (see spec_accept below, SPEC_NOTES.md §6).
      # forward() itself still never samples here: it always traces greedy (enforced by the assert below), since
      # sampled acceptance is pure host-side numpy math over these logits -- no RNG kernel to capture either way.
      assert temperature is None, "spec=True always traces greedy -- speculative_generate's sampled path (T4.65) " \
        "computes acceptance host-side from these logits, it never needs forward() itself to sample"
      # no .contiguous() on the logits (unlike x[:, -1:] below): self.output(...) is a fresh matmul over the
      # whole (unsliced) x, real computed output that owns its buffer, not a view into a larger scratch buffer
      # -- same reasoning SPEC_NOTES.md §3 already gives for why the old argmax result never needed this either.
      # .contiguous(): x[:, -1:] is a zero-copy VIEW into x's own buffer at an offset that depends on the
      # bound `toks` variable -- fine for the default path below, which consumes it inline in the SAME jit
      # call that produced it, but this hidden state is carried OUT of this call (into speculative_generate's
      # next-iteration draft) and read back only after further replays of this same jit graph have reused
      # x's buffer for a new chunk. Without materializing it into its own buffer here (same as the logits
      # already are, being real computed output rather than a view), that later read sees whatever the
      # intervening replays overwrote instead of this call's own hidden state -- confirmed by a byte-level
      # cache_kv diff between with/without .contiguous() on this line (see SPEC_NOTES.md).
      return self.output(self.output_norm(x)), x[:, -1:].contiguous()  # (B,T,vocab) logits, (B,1,D) hidden
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    # greedy (temperature is None): plain argmax, no RNG kernels
    if temperature is None: return logits.argmax(-1, keepdim=True)
    temperature = temperature.to(logits.device)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, spec:bool=False) -> Tensor|tuple[Tensor, Tensor]:
    is_prefill = bool(resolve(tokens.shape[1] != 1))
    # T4.12: a prefill/first-chunk call's `tokens` is a symbolic slice carrying the "toks" Variable, whose bound
    # range IS chunk_size -- captured into the jit graph. Keying only on (is_prefill, greedy) let a later
    # generate() at a different chunk_size replay a jit captured with the wrong Variable range -> JitError
    # ("args mismatch in JIT"). Decode/rollout steps feed a plain (1,1) tensor (no bound Variable, chunk_size-
    # independent), so only the prefill variants need the extra key; None keeps the rollout jit singular.
    # T4.64: `spec` also joins the key -- it picks a python branch inside forward() that changes the traced
    # graph (and its return arity), so a spec=True call must never replay a spec=False capture or vice versa.
    chunk_size = next((cast(int, v.vmax) for v in tokens.uop.variables() if v.expr == "toks"), None) if is_prefill else None
    return self.jit.setdefault((is_prefill, temperature is None, chunk_size, spec), TinyJit(self.forward))(
      tokens.contiguous(), start_pos, temperature, spec)

  def realize_placement(self):
    """Call once, right after loading weights into a device_map'd model (from_gguf does this for you) --
    e.g. `nn.state.load_state_dict(model, state_dict, realize=False); model.realize_placement()` for manual loaders.
    No-op (single attribute check) when device_map wasn't passed to __init__.

    load_state_dict's assignment is a `.to(param.device)` per tensor (nn/state.py:214, nothing layered after
    it); when that device differs from the load source, `.to` returns a NEW tensor whose OUTERMOST op is an
    unrealized COPY of the entire upstream expression -- e.g. a GGUF param whose dequant was built on the load
    device, or a manually-placed weight built on whatever device it happened to be constructed on. Left lazy,
    that COPY gets captured into the JIT trace and re-executed (recomputing everything upstream of it, AND
    the copy) every token (measured: 21 spurious COPYs/step on a 2-block METAL/CPU split test model -> 1 real
    one, once realized here).

    Only params whose top-level op IS that COPY pay this, so only those get eagerly realized below (T4.21):
    a `gguf.py`-loaded param placed via device_map has its raw quantized blob staged directly on the target
    device (T4.21's loader-side fix) with the dequant built ON TOP of that -- `.to(param.device)` is then a
    no-op (device already matches, no COPY node at all), so it never reaches this method's `moved` list in a
    state that needs realizing; forcing it anyway would eagerly materialize the FULL dequantized (e.g. fp16)
    size instead of leaving it fused into the consuming matmul like a same-device param -- fine for a small
    moved share, a multi-GB blowup for a big-model range split (T4.21's bug). Checking the top-level op instead
    of blanket-realizing every moved param is what makes that distinction cheaply, with no UOp-graph rewrite:
    a REAL cross-device COPY (this method's actual job) is always the outermost op right after `load_state_dict`
    (see nn/state.py:214) -- if it isn't there, there's nothing left for this method to do for that param.

    Also a footgun guard: a hand-built weight assigned without device= (e.g. Tensor.randn(...), which defaults to
    Device.DEFAULT) silently strands itself off the map instead of following its block's placement. Assert (not
    warn) -- this runs once right after load, not in a hot loop, and a silently-stranded weight is a correctness
    bug (wrong-device compute or a surprise COPY every step), not something to let slide."""
    if self._placed_devices is None: return
    params = nn.state.get_parameters(self)
    moved = [p for p in params if p.device != Device.DEFAULT]
    to_realize = [p for p in moved if p.uop.op is Ops.COPY]
    for p in to_realize: p.replace(p.contiguous())
    # Tensor.realize(*to_realize) is `to_realize[0].realize(*to_realize[1:])` -- with to_realize empty (no
    # moved param, or T4.21's loader already placed every moved param's blob with no COPY left to realize)
    # that's a bare Tensor.realize() call with no `self` bound at all: TypeError, not skip-nothing-to-do. Guard it.
    if to_realize: Tensor.realize(*to_realize)
    # params are always single-device here (llm/model.py never shards a weight) -- cast satisfies canonicalize's str|None signature
    stray = sorted(set(Device.canonicalize(cast(str, p.device)) for p in params) - self._placed_devices)
    assert not stray, f"device_map: param(s) landed on {stray}, outside the configured device_map {sorted(self._placed_devices)} " \
      "(a hand-built tensor is probably missing device=)"

  # T4.6: _init_state pre-allocates the whole KV cache (shape ~ max_context) on the first forward, before a
  # single token is generated. A caller who doesn't pass max_context used to get the model's NATIVE context
  # (up to 131072) here -- 2.2-5.3x the model's own weight size in KV cache nobody asked for (T1.9's finding,
  # measured: llama3.2:1b +4.3GB, qwen3:8b +6.07GB). Default to something sane instead; a caller that really
  # wants the model's full native context can still ask for it explicitly, either with max_context=<big N>
  # (min()'d against native below, so overshooting is harmless) or max_context=None (bypasses the cap entirely).
  DEFAULT_MAX_CONTEXT = 8192

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=DEFAULT_MAX_CONTEXT, *,
                device_map:str|dict[int|str,str]|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # gguf_load streams per-tensor (T1.9); no need to force the whole file onto the default device first.
    # T4.21: device_map threaded straight in, so cross-device tensors stage their raw blob on the target
    # device and dequant there -- see the T4.21 comment in gguf.py's _gguf_parse for the mechanism.
    kv, state_dict = gguf_load(gguf, device_map)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)

    ld = kv.get(f'{arch}.leading_dense_block_count', 0)
    # gpt-oss: llama.cpp doesn't write a per-layer pattern key for this; the alternation (even=sliding,
    # odd=full, starting at layer 0) is architecture-fixed, same as the HF config's `layer_types` and
    # examples/mlperf/models/gpt_oss.py's `sliding = i % 2 == 0`. GGUF only carries the window *size*.
    sliding_layers = tuple(i % 2 == 0 for i in range(kv[f'{arch}.block_count'])) if arch == 'gpt-oss' else ()
    rope_scaling = kv.get(f'{arch}.rope.scaling.type')
    yarn_factor = kv[f'{arch}.rope.scaling.factor'] if rope_scaling == 'yarn' else 1.0
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      # gpt-oss routes with softmax computed only over the selected top-k logits (no GGUF key for this; the HF
      # model has no configurable score_function, it's baked into the arch, same as examples/mlperf/models/gpt_oss.py)
      expert_gating_func=ExpertGating(
        kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX_WEIGHT if arch == 'gpt-oss' else ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=ld,
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{ld}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if ld else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{ld}.exp_probs_b.bias" in state_dict,
      attn_out_bias='blk.0.attn_output.bias' in state_dict,
      router_bias=f"blk.{ld}.ffn_gate_inp.bias" in state_dict,
      moe_bias=f"blk.{ld}.ffn_gate_exps.bias" in state_dict,
      attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_layers=sliding_layers,
      # gpt-oss's clamped-swiglu limit/alpha aren't GGUF metadata either (no config knob upstream); fixed
      # constants confirmed against the HF gpt-oss config (swiglu_limit=7.0) and examples/mlperf/models/gpt_oss.py
      clamp_swiglu=arch == 'gpt-oss', swiglu_limit=7.0, swiglu_alpha=1.702,
      yarn_factor=yarn_factor, yarn_orig_ctx=kv.get(f'{arch}.rope.scaling.original_context_length', 0),
      yarn_beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0), yarn_beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0),
      yarn_attn_factor=kv.get(f'{arch}.rope.scaling.yarn_attn_factor'))
    model = Transformer(config, device_map)  # pre-placed params make load_state_dict load each weight to its mapped device
    if MTP.value and kv.get(f'{arch}.nextn_predict_layers', 0) > 0:
      # mirror Transformer.__init__'s own full-attention block config (MTP/nextn is architecturally always a
      # plain attention block -- DeepSeek-MTP semantics -- positioned after every real block: never one of
      # leading_dense_blocks, never sliding (gpt-oss, the only sliding_layers arch, has no MTP block)
      mtp_cfg = replace(config, qk_norm=config.head_dim) if config.ssm else config
      block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
      model.mtp_head = MTPHead(mtp_cfg, block_cls)
      last_dev = model.blk[-1].device  # gguf.py's device-map clamp already stages blk.{num_blocks}.* here too
      for p in nn.state.get_parameters(model.mtp_head): p.to_(last_dev)
      _rename_mtp_keys(state_dict, config.num_blocks)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    else:
      # device_map's cross-device placements MUST realize -- see Transformer.realize_placement's docstring.
      # No-op when device_map is None (nothing moved off Device.DEFAULT to begin with).
      model.realize_placement()
    return model, kv

  def warmup(self):
    # warm both the greedy and sampled jit pairs, so a request doesn't pay a mid-request capture for whichever it hits first
    for temperature in (0.0, 1.0):
      for _ in range(2): list(zip(range(2), self.generate([0], temperature=temperature)))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0, drain_every:int=1):
    """drain_every: batch this many decode steps between host round-trips (T2.5 sync amortization) instead of
    syncing every sampled token. drain_every=1 (default) is byte-identical to the pre-T2.5 behavior -- every
    generate()-caller and test that assumes one .item()/next() per decode step keeps working unchanged. Pass
    drain_every=2..4 from a real serving loop to amortize the sync (streaming still yields one token at a time,
    just in bursts -- see the NOTE below the drain block for EOS handling)."""
    # T4.6: a prompt that already fills (or overflows) max_context has zero room to generate into. Without this,
    # `while virtual_len < self.max_context` below never runs and this silently yields nothing (len==max_context),
    # or the `t = Tensor(...).reshape(1, self.max_context)` a few lines down throws an opaque shape-mismatch
    # (len>max_context) -- neither names the actual problem. serve.py has its own (HTTP-level) version of this
    # check before it ever calls generate(); this is the equivalent guard for every other caller (CLI, library).
    assert len(tokens) < self.max_context, \
      f"prompt has {len(tokens)} tokens but max_context={self.max_context} leaves no room to generate " \
      "-- raise it via --max_context (cli.py) or Transformer.from_gguf(max_context=...)"
    # T4.55: recurrent blocks have a fused scan kernel only on RDNA3 (llm/kernels/amd.py); everywhere else the unrolled T_pad scan
    # in GatedDeltaNetBlock._attention handles a chunk, but generate() historically pinned chunk_size=1 (one token per step, i.e.
    # prefill at decode speed -- 46 tok/s on the pooled qwen3.6-35B). GDN_CHUNK caps the chunk width for those devices instead.
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device):
      chunk_size = min(chunk_size, gdn_chunk_for(self.token_embd.weight.device))
    drain_every = max(1, drain_every)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    # create helper tensors on the block devices that consume them, so device_map'd models don't replay a cross-device copy every step
    temp = Tensor([temperature], device=self.blk[-1].device) if temperature > 0 else None
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32", device=self.blk[0].device).reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    # T2.5: a sampled token already chains device-side -- `out` feeds the next step's input directly (below),
    # no host round-trip needed for the compute itself. .item() was only ever needed for host bookkeeping/
    # streaming (tokens list, _cached_tokens, yield). So launch up to `drain_every` decode steps back-to-back
    # (each still its own realize(), all async-dispatched -- see engine/realize.py's run_linear(wait=False)) and
    # read them all back with ONE host copy instead of one .item() sync per token. `pending` holds the
    # not-yet-host-read device tensors; `virtual_len` tracks what len(tokens) *will* be once pending is drained,
    # so the chunk/prefill arithmetic below is untouched by the deferral -- only the host-visible
    # append/_cached_tokens/yield timing moves.
    pending: list[Tensor] = []
    virtual_len = len(tokens)
    while virtual_len < self.max_context:
      n_toks = min(chunk_size, virtual_len - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = cast(Tensor, self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp)).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < virtual_len: continue
      # move the sampled token once, back to t's device, so the next step's input matches the JIT's prefill-captured device.
      if out.device != t.device: out = out.to(t.device).realize()
      # when draining is deferred (drain_every>1), `out` outlives the JIT replay that produced it -- the JIT reuses its
      # output buffer across replays (engine/jit.py's memory_plan_rewrite), so a not-yet-drained `out` would silently
      # alias data the *next* chained step's replay overwrites. Snapshot it into a private buffer to break that alias.
      # drain_every==1 (the default) drains immediately below, before any further replay can touch the buffer -- no
      # snapshot needed there, so this stays a pre-T2.5-identical zero-extra-op path.
      elif drain_every > 1: out = out.clone().realize()
      pending.append(out)
      virtual_len += 1
      # drain once the batch fills, or generation is about to stop (don't strand tokens on-device)
      if len(pending) >= drain_every or virtual_len >= self.max_context:
        # NOTE on EOS: the caller checks for a stop token per yielded value (see cli.py/serve.py's is_end loop)
        # and stops pulling from this generator. Because we yield one token at a time from the drained batch
        # below, a stop token anywhere in the batch makes the caller break before the extras after it are ever
        # appended/yielded -- self._cached_tokens and the returned `tokens` list end exactly at what was
        # actually consumed. The only cost is up to drain_every-1 already-computed, now-wasted device steps.
        for v in cast(list[list[int]], Tensor.cat(*pending, dim=1).tolist())[0]:
          tokens.append(int(v))
          self._cached_tokens = tokens[:-1]
          yield tokens[-1]
        pending = []

  def speculative_generate(self, tokens:list[int], k:int=3, chunk_size:int=32, temperature:float=0.0,
                            rng:np.random.Generator|None=None):
    """T4.64/T4.65: MTP speculative decoding -- drafts up to `k` tokens per iteration via self.mtp_head and
    verifies all of them in one extra main-model forward instead of paying one forward per token. See
    SPEC_NOTES.md for the full position ledger, the causal-mask argument for why neither the main model's
    nor the MTP block's attention KV cache ever needs a snapshot/restore (only the main model's GatedDeltaNet
    state does), and §6 for the sampled-acceptance math this docstring summarizes.

    temperature<=0 (the default): TOKEN-IDENTICAL to generate(temperature=0.0) -- every verify id is a plain
    argmax over forward(spec=True)'s per-position logits, exactly reproducing T4.64's behavior (which used to
    compute that same argmax INSIDE forward() -- see forward()'s spec branch). Draft quality never affects
    correctness here, only how many iterations it takes (see test_spec_decode.py's forced-mismatch/
    forced-perfect tests).

    temperature>0 (T4.65): distribution-preserving SAMPLED speculative decoding (Leviathan et al., "Fast
    Inference from Transformers via Speculative Decoding", 2023 -- see spec_accept). Draft tokens are drawn
    by ancestral sampling from the draft model's own softmax(logits/temperature) ("q"), then accepted or
    resampled against the main model's softmax(logits/temperature) ("p") so the marginal distribution of
    every emitted token equals p exactly (test_spec_decode.py verifies this empirically -- it's the theorem
    the whole path relies on). The output is distribution-equal to generate(temperature=t) but NOT
    sequence-equal to any particular generate() run -- it's a different (also p-distributed) sample, not a
    reproduction. `rng` defaults to a fresh np.random.default_rng() when omitted; pass a seeded one for
    reproducible/testable sampling.

    Mirrors generate()'s own bookkeeping exactly (the `tokens` list, self._cached_tokens, one token
    yielded at a time) -- generate() itself is untouched by this; see Transformer.__call__'s optional
    `spec=` return path.
    """
    assert self.mtp_head is not None, "speculative_generate needs an MTP-enabled model (Transformer.from_gguf under MTP=1)"
    assert k >= 1, "k=0 has nothing to draft -- use generate() directly"
    assert len(tokens) < self.max_context, \
      f"prompt has {len(tokens)} tokens but max_context={self.max_context} leaves no room to generate " \
      "-- raise it via Transformer.from_gguf(max_context=...)"
    # temperature<=0 is greedy, same convention generate() itself uses (see there). rng is unused on that
    # path but still materialized here unconditionally (cheap -- one object per REQUEST, not per token) so
    # every branch below can rely on it being a concrete Generator, never None (mypy narrows this cleanly;
    # threading `rng: Generator|None = None` through per-branch asserts instead would be more code for it).
    greedy = temperature <= 0
    import numpy as np  # lazy (function-local): the numpy-less CI lanes import this MODULE but never call this method
    if rng is None: rng = np.random.default_rng()
    # T4.55: same recurrent-chunk cap generate() applies (see there); then widen so the verify/re-forward
    # chunk (1..k+1 tokens) fits inside the SAME v_toks bound as the prefill chunks below -- one shared
    # Variable, one JIT capture per (is_prefill, spec) pair reused at every length, instead of one capture
    # per distinct length (which would also crash: see SPEC_NOTES.md's JIT-shape section).
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device):
      chunk_size = min(chunk_size, gdn_chunk_for(self.token_embd.weight.device))
    chunk_size = max(chunk_size, k + 1)
    gdn_blocks = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    dev = self.blk[0].device

    v_start_pos = UOp.variable("start_pos", 0, self.max_context - 1)
    v_toks = UOp.variable("toks", 1, chunk_size)

    # --- prefill: identical mechanics to generate() (same kind of v_start_pos/v_toks vars, same
    # zero-padded token buffer, same get_start_pos cache reuse) -- generate() itself is never called or
    # modified; this is a self-contained copy of just its chunking loop, small enough not to be worth
    # threading through a shared helper. Only the FINAL chunk asks for spec=True, to also recover the
    # pre-norm hidden state at the last prompt position (h_last) beside the per-position logits.
    prompt_len = len(tokens)
    t = Tensor(tokens + [0] * (self.max_context - prompt_len), dtype="int32", device=dev).reshape(1, self.max_context)
    start_pos = self.get_start_pos(tokens)
    tok_all: Tensor|None = None
    h_last: Tensor|None = None
    while start_pos < prompt_len:
      n_toks = min(chunk_size, prompt_len - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      if start_pos + n_toks >= prompt_len:
        tok_all, h_last = cast(tuple[Tensor, Tensor], self(t[:, sp:sp + nt], sp, None, spec=True))
      else:
        cast(Tensor, self(t[:, sp:sp + nt], sp, None, spec=False)).realize()
      start_pos += n_toks
    # get_start_pos always returns < prompt_len (a fresh/attention model can reuse at most prompt_len-1
    # positions -- tokens[:-1] -- and a recurrent one's cache-hit branch requires it strictly), so the loop
    # above always runs >=1 time and both are set; the assert is only for mypy's narrowing.
    assert tok_all is not None and h_last is not None
    anchor_logits = tok_all[:, -1:, :]  # (B,1,vocab): forward(spec=True) returns logits now, not ids (T4.65)
    # the first post-prompt token: no draft/accept here, it's a plain (greedy or sampled) draw from the real
    # model's own distribution -- exactly the token generate() itself would yield first (its own
    # last-prefill-chunk sample). It's about to become chunk_ids[0]/the draft anchor below, but it's ALSO
    # real output and must be emitted now, same as generate()'s own drain loop emits it, or every later
    # position is off by one.
    if greedy:
      tok_last = anchor_logits.argmax(-1)  # (B,1)
    else:
      anchor_probs = _softmax_np(anchor_logits.numpy()[0, 0], temperature)  # (vocab,)
      tok_last = Tensor([[int(rng.choice(len(anchor_probs), p=anchor_probs))]], dtype="int32", device=dev)
    tokens.append(int(tok_last.item()))
    self._cached_tokens = tokens[:-1]
    yield tokens[-1]

    while len(tokens) < self.max_context:
      k_eff = min(k, self.max_context - 1 - len(tokens))

      # (a) DRAFT: chain mtp_head.draft k_eff times. h_last is reused UNCHANGED across the whole chain --
      # MTPHead.draft's signature only takes one hidden-state argument, so it's the one anchor into the
      # main model's context available without changing that (fixed, already-tested) signature; only
      # tok_ids and start_pos advance per call, and `block`'s own KV cache (see MTPHead.draft's docstring)
      # is what actually carries the chain's positional continuity from call to call. This never touches
      # the MAIN model's state -- mtp_head.block is always attention-only, never GatedDeltaNet (see
      # from_gguf's MTP branch) -- so drafting alone never needs the GDN checkpoint below; only verify does.
      # Draft quality has zero effect on correctness (see the forced-mismatch test in test_spec_decode.py);
      # a richer per-step hidden (threading the block's own pre-norm output between calls) is a quality-only
      # lever for later, not something this task's gate depends on.
      #
      # T4.65 sampled path: draft_ids[i] must be an ANCESTRAL SAMPLE from q_i = softmax(draft_logits/temp),
      # not argmax -- spec_accept's distribution-preservation proof assumes this (see its docstring). That
      # needs the actual q_i vector on host to sample from (rng is a seeded np.random.Generator, not a
      # device RNG, precisely so this is reproducible/testable), so the sampled chain pulls one small
      # (vocab,) logits vector to host per drafted token instead of the greedy chain's single batched pull
      # below. ponytail: k is small (default 3) and this only runs when temperature>0, so k_eff extra host
      # round-trips per iteration is a fine trade for keeping the sampling host-side/seedable; revisit only
      # if a real serving benchmark ever shows --mtp sampled mode is sync-bound.
      dtok, dpos = tok_last, start_pos
      draft_ids: list[int] = []
      draft_tensors: list[Tensor] = []
      q_list: list[np.ndarray] = []
      for _ in range(k_eff):
        dlogits = self.mtp_head.draft(self, h_last, dtok, dpos)  # (B,1,vocab)
        if greedy:
          dtok = dlogits.argmax(-1)  # (B,1)
          draft_tensors.append(dtok)
        else:
          q_i = _softmax_np(dlogits[:, -1, :].numpy()[0], temperature)  # (vocab,)
          d_i = int(rng.choice(len(q_i), p=q_i))
          draft_ids.append(d_i)
          q_list.append(q_i)
          dtok = Tensor([[d_i]], dtype="int32", device=dev)
        dpos += 1
      chunk_ids: list[int]
      if greedy:
        # one host round-trip for tok_last + every drafted id (mirrors generate()'s own batched-drain sync
        # a few lines up) -- chunk_ids[0] is tok_last (already-confirmed real), chunk_ids[1:] are the drafts.
        chunk_ids = cast(list[list[int]], (tok_last.cat(*draft_tensors, dim=1) if draft_tensors else tok_last).tolist())[0]
      else:
        chunk_ids = [int(tok_last.item())] + draft_ids  # draft_ids were already pulled to host per-step above
      n = len(chunk_ids)  # k_eff + 1

      # (b) CHECKPOINT: GatedDeltaNet conv/recurrent state is a single read-modify-written accumulator, not
      # a position-indexed cache -- once the verify forward below mixes a wrong draft token into it there is
      # no slice to discard, the mixing is irreversible (unlike attention KV, see the mask argument in
      # SPEC_NOTES.md). Snapshot it device-side (one batched realize) so a partial accept can restore it.
      gdn_snap = [(b, b.conv_state.clone(), b.recurrent_state.clone()) for b in gdn_blocks]
      if gdn_snap: Tensor.realize(*(s for _, c, r in gdn_snap for s in (c, r)))

      # (c) VERIFY: one main-model forward of the whole chunk [tok_last, d0..d_{k_eff-1}] at the current
      # start_pos, returning the per-position logits (what the real model actually predicts from every one
      # of those positions) plus the hidden state at the chunk's last position.
      buf = Tensor(chunk_ids + [0] * (chunk_size - n), dtype="int32", device=dev).reshape(1, chunk_size)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n)
      verify_logits, verify_h = cast(tuple[Tensor, Tensor], self(buf[:, :nt], sp, None, spec=True))

      # (d) ACCEPT: how many of the k_eff drafts hold up (m, 0..k_eff), and the one extra token this
      # iteration emits beyond them -- either the greedy exact-match bonus/correction id, or (T4.65,
      # temperature>0) a sampled accept/resample per spec_accept. Both branches produce `accepted`
      # (== chunk_ids[1:1+m] + [that extra token], always m+1 long) and `tok_last` (the new chain anchor).
      if greedy:
        # verify_logits' own shape is still the bound-but-symbolic `nt` (unlike tok_last/draft ids, which are
        # always concrete-shaped size-1 slices) -- a symbolic-shaped tensor has no .tolist()/.numpy() (see
        # TestCausalMask.test_symbolic_shapes in test_attention.py for the same pad-then-slice idiom): argmax
        # it down to ids first (the same op forward() used to do internally, pre-T4.65), then pad up to the
        # fixed chunk_size buffer width, then take just the first n (live) values back on host.
        verify_ids_tensor = verify_logits.argmax(-1)  # (1, nt) ids
        verify_ids: list[int] = cast(list[list[int]], verify_ids_tensor.pad_to((1, chunk_size)).tolist())[0][:n]  # [i] predicts start_pos+1+i
        # verify_ids[i] predicts the position AFTER chunk_ids[i] was fed, so it's comparable to draft
        # d_i == chunk_ids[i+1] (chunk_ids[0] is tok_last, which needs no verifying -- it's already a
        # confirmed-real token from a previous iteration or the prefill above). m = length of the longest
        # matching prefix, 0..k_eff.
        m = 0
        while m < k_eff and chunk_ids[m + 1] == verify_ids[m]: m += 1
        accepted = chunk_ids[1:1 + m] + [verify_ids[m]]  # d0..d_{m-1} plus the bonus/correction token: m+1 total
        tok_last = verify_ids_tensor[:, m:m + 1]  # kept lazy/device-side, like generate()'s own `out` chaining
      else:
        # same symbolic-shape reasoning as the greedy branch (pad before pulling to host), but here we need
        # the full per-position distributions, not just the argmax -- pad_to is a no-op on the already-fixed
        # vocab axis, only the symbolic `nt` axis actually gets padded.
        vocab = int(verify_logits.shape[-1])  # never symbolic (only the T axis is) -- int() just satisfies mypy's sint=int|UOp
        verify_np = verify_logits.pad_to((1, chunk_size, vocab)).numpy()[0][:n]  # (n, vocab)
        p_probs = _softmax_np(verify_np, temperature)
        q_probs = np.stack(q_list) if q_list else np.empty((0, vocab))
        accepted, m = spec_accept(draft_ids, q_probs, p_probs, rng)  # accepted == chunk_ids[1:1+m] + [extra], len m+1
        tok_last = Tensor([[accepted[-1]]], dtype="int32", device=dev)

      if m == k_eff:
        # full accept: every input verify saw (tok_last, d0..d_{k_eff-1}) really was correct, so its forward
        # legitimately advanced every state through position start_pos+k_eff -- nothing to roll back, and
        # its own last-position hidden IS the next h_last.
        h_last = verify_h
      else:
        # partial accept: verify's forward also advanced state through the wrong tokens d_m..d_{k_eff-1}.
        # Roll GDN back to before this call and redo exactly the m+1 confirmed-correct tokens as one chunk
        # to rebuild the true state and get the next h_last. Attention KV needs no restore for the same
        # reason it never needed a snapshot: this redo overwrites positions [start_pos, start_pos+m+1) with
        # the SAME values verify already wrote there (a harmless recompute), and the discarded wrong
        # positions start_pos+m+1..start_pos+k_eff are never read before the next iteration overwrites them
        # too. This re-forward is the v1 cost T4.66 would remove (see SPEC_NOTES.md).
        if gdn_snap: Tensor.realize(*(x for b, c, r in gdn_snap for x in (b.conv_state.assign(c), b.recurrent_state.assign(r))))
        redo_ids = chunk_ids[:m + 1]
        buf2 = Tensor(redo_ids + [0] * (chunk_size - len(redo_ids)), dtype="int32", device=dev).reshape(1, chunk_size)
        sp2, nt2 = v_start_pos.bind(start_pos), v_toks.bind(len(redo_ids))
        _, h_last = cast(tuple[Tensor, Tensor], self(buf2[:, :nt2], sp2, None, spec=True))

      start_pos += m + 1
      for v in accepted:
        tokens.append(v)
        self._cached_tokens = tokens[:-1]
        yield tokens[-1]
