import io, unittest
from tinygrad.llm import image

try:
  import numpy as np
  from PIL import Image
  HAS_PIL = True
except ImportError:
  HAS_PIL = False

# smart_resize is pure math (no numpy/PIL) -- these run in every CI lane.
class TestSmartResize(unittest.TestCase):
  def test_within_bounds_unchanged(self):
    # 640/32=20, 480/32=15 (exact), area 307200 in [65536, 1003520] -> untouched
    self.assertEqual(image.smart_resize(640, 480), (640, 480))

  def test_small_square_upscaled_to_min_pixels(self):
    # round(100/32)*32 = 96 -> area 9216 < MIN_PIXELS -> upscale branch: beta = 256/100 exactly (65536=256**2),
    # ceil(100*beta/32)*32 = 256. 256 is itself the smallest multiple of 32 whose square exceeds MIN_PIXELS.
    h_bar, w_bar = image.smart_resize(100, 100)
    self.assertEqual((h_bar, w_bar), (256, 256))
    self.assertGreaterEqual(h_bar * w_bar, image.MIN_PIXELS)

  def test_large_image_capped_to_max_pixels(self):
    h_bar, w_bar = image.smart_resize(4000, 3000, max_pixels=1_003_520)
    self.assertEqual(h_bar % 32, 0)
    self.assertEqual(w_bar % 32, 0)
    self.assertLessEqual(h_bar * w_bar, 1_003_520)

  def test_extreme_aspect_raises(self):
    self.assertRaises(ValueError, image.smart_resize, 1000, 4)  # 1000/4 = 250 > 200

  def test_rounds_to_factor_then_upscales(self):
    # round(47/32) = round(1.46875) = 1 -> pre-branch (32, 32), area 1024 < MIN_PIXELS (below the min_pixels
    # branch, per the task) -> upscale applies: beta = 256/47 exactly (65536=256**2, 47**2=2209), so
    # 47*beta = 256 exactly -> ceil(256/32)*32 = 256.
    h_bar, w_bar = image.smart_resize(47, 47)
    self.assertEqual((h_bar, w_bar), (256, 256))
    self.assertEqual(h_bar % 32, 0)
    self.assertEqual(w_bar % 32, 0)

@unittest.skipUnless(HAS_PIL, "PIL/numpy not importable")
class TestPreprocess(unittest.TestCase):
  @staticmethod
  def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()

  def test_shape_grid_dtype_range(self):
    h, w = 96, 64
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    arr[..., 2] = 128
    flat, grid = image.preprocess(self._png_bytes(arr))
    h_bar, w_bar = image.smart_resize(h, w)
    self.assertEqual(grid, (1, h_bar // 16, w_bar // 16))
    self.assertEqual(flat.shape, (grid[1] * grid[2], 1536))
    self.assertEqual(flat.dtype, np.float32)
    self.assertTrue(bool(np.all(flat >= -1.0)) and bool(np.all(flat <= 1.0)))

  def test_determinism(self):
    data = self._png_bytes(np.full((64, 64, 3), 77, dtype=np.uint8))
    flat1, grid1 = image.preprocess(data)
    flat2, grid2 = image.preprocess(data)
    self.assertEqual(grid1, grid2)
    self.assertTrue(np.array_equal(flat1, flat2))

  def test_patch_order_matches_explicit_reference(self):
    # 64x64 at default MIN_PIXELS would upscale (area 4096 < 65536); pin min_pixels down so the image stays
    # 64x64 -> a plain 4x4 patch grid, small enough to check every row by hand.
    raw = np.random.default_rng(1234).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)  # HWC
    flat, grid = image.preprocess(self._png_bytes(raw), min_pixels=1)
    self.assertEqual(grid, (1, 4, 4))

    # Independent reference: normalize the SAME bytes used to build the PNG directly (bicubic resize to an
    # unchanged size, and the PNG codec, are both exactly lossless here -- verified separately), then read
    # patches out by explicit slicing instead of going through preprocess's own reshape/transpose.
    norm_chw = (((raw.astype(np.float32) / 255.0) - 0.5) / 0.5).transpose(2, 0, 1)  # (3, 64, 64)

    def patch_vec(ph, pw):
      row = []
      for c in range(3):
        block = norm_chw[c, ph * 16:(ph + 1) * 16, pw * 16:(pw + 1) * 16].reshape(-1)  # (y, x) row-major, 256
        row += [block, block]  # TEMPORAL=2 duplicate of the same frame -> (c, t, y, x) order
      return np.concatenate(row)  # 1536

    expected = [patch_vec(bh * 2 + mh, bw * 2 + mw)
                for bh in range(2) for bw in range(2) for mh in range(2) for mw in range(2)]
    self.assertTrue(np.array_equal(flat, np.stack(expected)))

class TestMisc(unittest.TestCase):
  def test_n_visual_tokens(self):
    self.assertEqual(image.n_visual_tokens((1, 8, 12)), 24)

  def test_hash_ids_in_range(self):
    ids = image.hash_ids(image.image_hash(b"some image bytes"), vocab_size=1000, n=8)
    self.assertEqual(len(ids), 8)
    self.assertTrue(all(0 <= i < 1000 for i in ids))

if __name__ == "__main__":
  unittest.main()
