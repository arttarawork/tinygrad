"""T5.2: the Qwen3.6 (`qwen3_5_moe`) vision encoder + 2x2 patch merger, loaded from a llama.cpp `mmproj` GGUF (VISION_DESIGN.md §2.2).

Mirrors transformers' Qwen3_5MoeVisionModel piece by piece (class names in the comments): a SigLIP2-style ViT over a variable-size
patch grid, with a learned 48x48 position table bilinearly resampled to the image grid PLUS axial 2-D RoPE, 27 pre-LN blocks with full
(non-causal) attention, then a merger that folds every 2x2 window of patches into one LLM-width token. No DeepStack: this checkpoint has
`deepstack_visual_indexes: []`. Patch vectors come in the processor's order -- (channel, temporal, y, x) flattened, merge-window-major --
which is what tinygrad/llm/image.py (T5.1) and transformers' Qwen2VLImageProcessor both emit.
"""
from __future__ import annotations
import math, pathlib
from dataclasses import dataclass
import numpy as np
from tinygrad import Device, Tensor, nn
from tinygrad.llm.gguf import gguf_load

@dataclass(frozen=True)
class VisionConfig:
  depth:int=27
  hidden:int=1152
  heads:int=16
  mlp:int=4304
  patch:int=16
  merge:int=2
  temporal:int=2
  channels:int=3
  out:int=2048
  n_pos:int=2304   # learned position table = (sqrt(n_pos))^2 grid
  eps:float=1e-6
  @property
  def head_dim(self) -> int: return self.hidden // self.heads
  @property
  def patch_vec(self) -> int: return self.channels * self.temporal * self.patch * self.patch
  @property
  def side(self) -> int: return int(math.isqrt(self.n_pos))

def merge_window_rowcol(t:int, gh:int, gw:int, m:int) -> tuple[np.ndarray, np.ndarray]:
  """(row, col) of every patch in the processor's merge-window-major order (transformers' get_vision_position_ids), repeated t times."""
  i = np.arange(gh * gw)
  row = (i // (m * m * (gw // m))) * m + (i // m) % m
  col = ((i // (m * m)) % (gw // m)) * m + i % m
  return np.tile(row, t), np.tile(col, t)

def bilinear_taps(index:np.ndarray, size:int, side:int) -> tuple[np.ndarray, np.ndarray]:
  """Per-axis (2 taps, 2 weights) resampling a `side`-long table onto `size` positions, F.interpolate(bilinear, align_corners=True)
  semantics (transformers.vision_utils._interpolation_axis_taps_weights)."""
  src = index.astype(np.float32) * (side - 1) / max(size - 1, 1)
  floor = np.floor(src)
  raw = floor[:, None] + np.arange(2)[None, :]
  return np.clip(raw, 0, side - 1).astype(np.int32), np.clip(1 - np.abs(src[:, None] - raw), 0, None).astype(np.float32)

class VisionBlock:  # Qwen3_5MoeVisionBlock
  def __init__(self, c:VisionConfig):
    self.c = c
    self.ln1, self.ln2 = nn.LayerNorm(c.hidden, eps=c.eps), nn.LayerNorm(c.hidden, eps=c.eps)
    self.attn_qkv, self.attn_out = nn.Linear(c.hidden, 3 * c.hidden), nn.Linear(c.hidden, c.hidden)   # Qwen3_5MoeVisionAttention qkv/proj
    self.ffn_up, self.ffn_down = nn.Linear(c.hidden, c.mlp), nn.Linear(c.mlp, c.hidden)               # Qwen3_5MoeVisionMLP fc1/fc2
  def __call__(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
    n, hd = x.shape[0], self.c.head_dim
    qkv = self.attn_qkv(self.ln1(x)).reshape(n, 3, self.c.heads, hd).permute(1, 2, 0, 3)  # (3, H, n, hd); q/k/v stay 3-D (H, n, hd)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin                # apply_rotary_pos_emb_vision
    a = q.scaled_dot_product_attention(k, v).transpose(0, 1).reshape(n, self.c.hidden)      # full attention over the image
    x = x + self.attn_out(a)
    return x + self.ffn_down(self.ffn_up(self.ln2(x)).gelu())                              # gelu_pytorch_tanh

def _rotate_half(x:Tensor) -> Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  return (-x2).cat(x1, dim=-1)

class VisionEncoder:  # Qwen3_5MoeVisionModel (+ its Qwen3_5MoeVisionPatchMerger)
  def __init__(self, c:VisionConfig, device:str|None=None):
    self.c, self._device = c, device
    self.patch_embd = nn.Linear(c.patch_vec, c.hidden)          # Qwen3_5MoeVisionPatchEmbed: the Conv3d(k=s=[T,p,p]) as a Linear
    self.position_embd = Tensor.zeros(c.n_pos, c.hidden)        # nn.Embedding(num_position_embeddings)
    self.blk = [VisionBlock(c) for _ in range(c.depth)]
    self.post_ln = nn.LayerNorm(c.hidden, eps=c.eps)            # merger.norm (pre-shuffle, per patch)
    merged = c.hidden * c.merge * c.merge
    self.mm0, self.mm2 = nn.Linear(merged, merged), nn.Linear(merged, c.out)  # merger.linear_fc1 / linear_fc2
    self.inv_freq = 1.0 / (10000.0 ** (np.arange(0, c.head_dim // 2, 2, dtype=np.float32) / (c.head_dim // 2)))  # VisionRotaryEmbedding(head_dim//2)

  @property
  def device(self) -> str|tuple[str, ...]:
    # T5.5: the weights' actual device -- `device=None` (tests; CI's METAL default lane) or a gguf_load that ignored the request
    # must never leave the host-built index/rope tensors on a different device than the weights (round 8/CI: 'expected index and
    # self on the same device').
    return self.position_embd.device if hasattr(self, 'position_embd') else (self._device or Device.DEFAULT)

  @staticmethod
  def from_mmproj(path:str|pathlib.Path, device:str|None=None) -> VisionEncoder:
    kv, w = gguf_load(path, device_map=device)
    assert kv.get("clip.projector_type") == "qwen3vl_merger", f"unsupported mmproj projector {kv.get('clip.projector_type')!r}"
    c = VisionConfig(depth=kv["clip.vision.block_count"], hidden=kv["clip.vision.embedding_length"], heads=kv["clip.vision.attention.head_count"],
                     mlp=kv["clip.vision.feed_forward_length"], patch=kv["clip.vision.patch_size"], merge=kv["clip.vision.spatial_merge_size"],
                     out=kv["clip.vision.projection_dim"], n_pos=int(w["v.position_embd.weight"].shape[0]),
                     eps=float(kv.get("clip.vision.attention.layer_norm_epsilon", 1e-6)))
    enc = VisionEncoder(c, device)
    # T5.5: gguf_load's device_map is the LLM's per-block map; for this file it leaves tensors on the process default device
    # (DEV=NV on the pooled server -- the encoder silently lived on the 3090), so move each one explicitly.
    def f(n:str) -> Tensor: return (w[n].to(device) if device is not None else w[n]).float().realize()  # bf16/f32 -> float32 on `device`
    # llama.cpp stores the Conv3d kernel [T,p,p] as one 2-D slice per temporal step: weight (t=0) and weight.1 (t=1); re-stack on the
    # temporal axis and flatten in the patch vector's (channel, temporal, y, x) order
    slices = [f("v.patch_embd.weight")] + [f(f"v.patch_embd.weight.{t}") for t in range(1, c.temporal)]
    enc.patch_embd.weight = Tensor.stack(*slices, dim=2).reshape(c.hidden, c.patch_vec).realize()
    enc.patch_embd.bias = f("v.patch_embd.bias")
    enc.position_embd = f("v.position_embd.weight")
    for i, b in enumerate(enc.blk):
      for name in ("ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"):   # every block tensor is a weight+bias pair
        lin = getattr(b, name)
        lin.weight, lin.bias = f(f"v.blk.{i}.{name}.weight"), f(f"v.blk.{i}.{name}.bias")
    enc.post_ln.weight, enc.post_ln.bias = f("v.post_ln.weight"), f("v.post_ln.bias")
    enc.mm0.weight, enc.mm0.bias, enc.mm2.weight, enc.mm2.bias = f("mm.0.weight"), f("mm.0.bias"), f("mm.2.weight"), f("mm.2.bias")
    return enc

  def pos_embed(self, grid:tuple[int, int, int]) -> Tensor:
    """(n, hidden): the learned side x side table bilinearly resampled (align_corners=True) to the (gh, gw) patch grid, in merge-window order."""
    t, gh, gw = grid
    row, col = merge_window_rowcol(t, gh, gw, self.c.merge)
    ht, hw_ = bilinear_taps(row, gh, self.c.side)
    wt, ww = bilinear_taps(col, gw, self.c.side)
    idx = (ht[:, :, None] * self.c.side + wt[:, None, :]).reshape(-1, 4)       # (n, 4) table rows
    wgt = (hw_[:, :, None] * ww[:, None, :]).reshape(-1, 4)                      # (n, 4) weights
    return (self.position_embd[Tensor(idx, device=self.device)] * Tensor(wgt, device=self.device).unsqueeze(-1)).sum(1)

  def rope(self, grid:tuple[int, int, int]) -> tuple[Tensor, Tensor]:
    """(cos, sin), each (1, n, head_dim), broadcast over the (heads, n, head_dim) q/k. Axial 2-D RoPE: head_dim//4 frequencies on the
    row index and the same on the column index, concatenated then duplicated over the two halves (rotate_half convention) -- exactly
    transformers' `emb = cat(rotary, rotary)`."""
    t, gh, gw = grid
    row, col = merge_window_rowcol(t, gh, gw, self.c.merge)
    freqs = np.concatenate([row[:, None] * self.inv_freq, col[:, None] * self.inv_freq], axis=1)  # (n, head_dim//2)
    emb = np.concatenate([freqs, freqs], axis=1).astype(np.float32)                                # (n, head_dim)
    cos, sin = Tensor(np.cos(emb), device=self.device), Tensor(np.sin(emb), device=self.device)
    return cos.reshape(1, -1, self.c.head_dim), sin.reshape(1, -1, self.c.head_dim)

  def __call__(self, patches:Tensor, grid:tuple[int, int, int]) -> Tensor:
    """patches (n, channels*temporal*patch*patch) float in merge-window order, grid (t, gh, gw) with n == t*gh*gw -> (n // merge^2, out)."""
    t, gh, gw = grid
    assert patches.shape == (t * gh * gw, self.c.patch_vec), f"patches {patches.shape} vs grid {grid} -> ({t*gh*gw}, {self.c.patch_vec})"
    assert gh % self.c.merge == 0 and gw % self.c.merge == 0, f"grid {grid} not a multiple of merge {self.c.merge}"
    x = self.patch_embd(patches.to(self.device)) + self.pos_embed(grid)
    cos, sin = self.rope(grid)
    for b in self.blk: x = b(x, cos, sin)
    x = self.post_ln(x).reshape(-1, self.c.hidden * self.c.merge * self.c.merge)   # the 4 patches of a window are consecutive
    return self.mm2(self.mm0(x).gelu(approximate="none"))                          # merger: nn.GELU() is the exact (erf) one
