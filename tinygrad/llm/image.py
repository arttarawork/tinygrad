# T5.1 (VISION_DESIGN.md section 2.1): turns raw image bytes into the exact patch vectors and grid shape the
# Qwen3.5/3.6 vision encoder (T5.2) expects -- HF `Qwen2VLImageProcessor` semantics, reimplemented here (no
# transformers/torch dependency on the serving path). Kept numpy/PIL-free at import time like the rest of
# tinygrad/llm (T4.65): both are imported lazily inside the functions that need them.
from __future__ import annotations
import hashlib, io, math
from typing import TYPE_CHECKING
if TYPE_CHECKING: import numpy as np

PATCH = 16              # ViT patch side, pixels
MERGE = 2               # spatial_merge_size: 2x2 patches merge into one visual token
TEMPORAL = 2            # temporal_patch_size: a still image is duplicated to a 2-frame clip
FACTOR = PATCH * MERGE  # 32: smart_resize rounds both sides to a multiple of this
MIN_PIXELS = 65536
DEFAULT_MAX_PIXELS = 1_003_520
MEAN = 0.5
STD = 0.5

def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 1_003_520) -> tuple[int, int]:
  """Qwen2-VL's resize policy (HF `Qwen2VLImageProcessor.smart_resize`), verbatim: round both sides to the
  nearest multiple of `factor`, then grow or shrink (rounding to stay on the `factor` grid) until the pixel
  count lands in [min_pixels, max_pixels]. Uses Python's `round` (banker's rounding) exactly like the
  reference -- do not swap in a different rounding rule, it would shift dimensions on exact .5 boundaries."""
  if max(height, width) / min(height, width) > 200:
    raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
  h_bar = max(factor, round(height / factor) * factor)
  w_bar = max(factor, round(width / factor) * factor)
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar

def preprocess(data: bytes, max_pixels: int = DEFAULT_MAX_PIXELS, min_pixels: int = MIN_PIXELS) -> tuple[np.ndarray, tuple[int, int, int]]:
  """Bytes -> (flat patch vectors, grid). `flat` is (n_patches, 3*TEMPORAL*PATCH*PATCH) == (n_patches, 1536)
  float32, ready to feed the vision encoder's patch-embed matmul; `grid` is (grid_t, grid_h, grid_w) in patch
  units (grid_t is always 1 -- a still image, no video).

  Patch order is MERGE-WINDOW-MAJOR, not raster: consecutive rows of `flat` are the 4 patches of one 2x2
  merge window, i.e. row index decomposes (row-major) as (bh, bw, mh, mw) with patch position
  (ph, pw) = (bh*MERGE + mh, bw*MERGE + mw) -- bh/bw range over the merge-window grid (grid_h/MERGE,
  grid_w/MERGE), mh/mw range over {0, 1}. T5.2's position embedding and 2-D RoPE are built to match this
  order; do not reorder to raster without updating that side too.

  Each 1536-length row is itself flattened in (channel, temporal, y, x) order (channel slowest, x fastest):
  row[((c*TEMPORAL + t)*PATCH + y)*PATCH + x] == norm[c, t, ph*PATCH + y, pw*PATCH + x], where `norm` is the
  normalized CHW image duplicated along a new leading temporal axis of size TEMPORAL."""
  import numpy as np
  from PIL import Image
  img = Image.open(io.BytesIO(data)).convert("RGB")
  width, height = img.size
  h_bar, w_bar = smart_resize(height, width, factor=FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
  img = img.resize((w_bar, h_bar), resample=Image.Resampling.BICUBIC)
  arr = np.asarray(img, dtype=np.float32) / 255.0   # (h_bar, w_bar, 3) in [0, 1]
  arr = (arr - MEAN) / STD                          # normalize to roughly [-1, 1]
  arr = arr.transpose(2, 0, 1)                      # CHW: (3, h_bar, w_bar)
  arr = np.stack([arr, arr], axis=0)                # temporal duplicate: (TEMPORAL, 3, h_bar, w_bar)
  grid_t, grid_h, grid_w = 1, h_bar // PATCH, w_bar // PATCH
  patches = arr.reshape(grid_t, TEMPORAL, 3, grid_h // MERGE, MERGE, PATCH, grid_w // MERGE, MERGE, PATCH)
  patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
  flat = patches.reshape(grid_t * grid_h * grid_w, 3 * TEMPORAL * PATCH * PATCH).astype(np.float32)
  return flat, (grid_t, grid_h, grid_w)

def n_visual_tokens(grid: tuple[int, int, int]) -> int:
  """Number of merged visual tokens (post 2x2 merge) a (grid_t, grid_h, grid_w) patch grid produces."""
  grid_t, grid_h, grid_w = grid
  return grid_t * (grid_h // MERGE) * (grid_w // MERGE)

def image_hash(data: bytes) -> bytes:
  """sha256 of the raw image bytes -- identifies an image for the id-expansion trick in VISION_DESIGN.md
  section 2.1 and for state-cache prefix matching."""
  return hashlib.sha256(data).digest()

def hash_ids(digest: bytes, vocab_size: int, n: int = 8) -> list[int]:
  """First `n` 4-byte big-endian chunks of `digest`, each folded into a valid vocab id. Used to make the
  leading ids of an expanded image_pad run depend on the image, so two prompts differing only in image
  content differ in token ids too (see VISION_DESIGN.md section 2.1)."""
  return [int.from_bytes(digest[4 * j:4 * j + 4], "big") % vocab_size for j in range(n)]
