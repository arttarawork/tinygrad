import unittest, importlib.util, pathlib
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.vision import VisionConfig, VisionEncoder, merge_window_rowcol, bilinear_taps

MMPROJ = pathlib.Path("/Users/artur/models/qwen3.6-35b-a3b-q8/mmproj-BF16.gguf")
HAS_REF = importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None
# depth 2, hidden 32, 4 heads (head_dim 8 -> 2 row + 2 col rope frequencies), 2x2 patches, a 4x4 learned position table
TINY = VisionConfig(depth=2, hidden=32, heads=4, mlp=64, patch=2, merge=2, temporal=2, channels=3, out=16, n_pos=16)

class TestVisionEncoderTiny(unittest.TestCase):
  def test_output_shape(self):
    enc = VisionEncoder(TINY, "CPU")
    self.assertEqual(enc(Tensor.randn(24, TINY.patch_vec), (1, 4, 6)).shape, (6, 16))   # 4x6 patches -> 6 merged tokens of width `out`

  def test_merge_window_order(self):
    row, col = merge_window_rowcol(1, 4, 6, 2)
    # window-major: window (rows 0-1, cols 0-1) as r0c0, r0c1, r1c0, r1c1, then the window at cols 2-3
    self.assertEqual(list(zip(row[:8].tolist(), col[:8].tolist())), [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    self.assertEqual(len(row), 24)

  def test_pos_embed_identity_at_table_size(self):
    # a grid as large as the table maps each patch onto exactly one table row (align_corners=True) -> the table itself, in window order
    enc = VisionEncoder(TINY, "CPU")
    enc.position_embd = Tensor.randn(TINY.n_pos, TINY.hidden).realize()
    row, col = merge_window_rowcol(1, 4, 4, 2)
    np.testing.assert_allclose(enc.pos_embed((1, 4, 4)).numpy(), enc.position_embd.numpy()[row * 4 + col], rtol=1e-6, atol=1e-6)

  def test_bilinear_taps(self):
    taps, w = bilinear_taps(np.array([0, 1, 2]), 3, 5)   # 3 targets over a 5-row table land on rows 0, 2, 4 exactly
    self.assertEqual(taps[:, 0].tolist(), [0, 2, 4])
    np.testing.assert_allclose(w, [[1.0, 0.0]] * 3)
    taps, w = bilinear_taps(np.array([1]), 3, 4)         # index 1 of 3 over 4 rows -> src 1.5: half of row 1, half of row 2
    self.assertEqual(taps.tolist(), [[1, 2]])
    np.testing.assert_allclose(w, [[0.5, 0.5]])
    taps, _ = bilinear_taps(np.array([1]), 2, 4)         # the far edge: src 3.0, the +1 tap clamps to the last row (weight 0)
    self.assertEqual(taps.tolist(), [[3, 3]])

  def test_rope_layout(self):
    enc = VisionEncoder(TINY, "CPU")
    cos, sin = enc.rope((1, 4, 6))
    self.assertEqual(cos.shape, (1, 24, TINY.head_dim))
    c = cos.numpy()[0]
    np.testing.assert_allclose(c[:, :TINY.head_dim // 2], c[:, TINY.head_dim // 2:])   # duplicated halves (rotate_half convention)
    row, col = merge_window_rowcol(1, 4, 6, 2)
    np.testing.assert_allclose(c[:, 0], np.cos(row * enc.inv_freq[0]), rtol=1e-6, atol=1e-6)   # first channel: the row angle
    np.testing.assert_allclose(c[:, 2], np.cos(col * enc.inv_freq[0]), rtol=1e-6, atol=1e-6)   # first column channel
    np.testing.assert_allclose(sin.numpy()[0, :, 0], np.sin(row * enc.inv_freq[0]), rtol=1e-6, atol=1e-6)

@unittest.skipUnless(HAS_REF and MMPROJ.exists(), "needs transformers + torch in the venv and the Qwen3.6 mmproj GGUF")
class TestVisionEncoderParity(unittest.TestCase):
  def test_matches_transformers(self):
    import torch
    from PIL import Image, ImageDraw
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionConfig, Qwen3_5MoeVisionModel
    from tinygrad.llm.gguf import gguf_load
    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "TINYGRAD VISION 2026", fill="black")
    d.rectangle((60, 120, 200, 260), fill="red")
    d.ellipse((250, 150, 400, 300), fill="blue")
    d.polygon([(100, 300), (200, 420), (40, 420)], fill="green")
    proc = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                 size={"shortest_edge": 65536, "longest_edge": 1003520})
    inputs = proc(images=img, return_tensors="pt")
    pv, thw = inputs["pixel_values"].float(), inputs["image_grid_thw"]
    self.assertEqual(tuple(thw[0].tolist()), (1, 28, 28))
    cfg = Qwen3_5MoeVisionConfig(depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304, patch_size=16, spatial_merge_size=2,
                                 temporal_patch_size=2, out_hidden_size=2048, num_position_embeddings=2304, in_channels=3,
                                 hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])
    ref = Qwen3_5MoeVisionModel(cfg).float().eval()
    _, w = gguf_load(str(MMPROJ), device_map="CPU")
    def g(n): return torch.from_numpy(w[n].float().numpy())
    sd = {"patch_embed.proj.weight": torch.stack([g("v.patch_embd.weight"), g("v.patch_embd.weight.1")], dim=2),
          "patch_embed.proj.bias": g("v.patch_embd.bias"), "pos_embed.weight": g("v.position_embd.weight"),
          "merger.norm.weight": g("v.post_ln.weight"), "merger.norm.bias": g("v.post_ln.bias"),
          "merger.linear_fc1.weight": g("mm.0.weight"), "merger.linear_fc1.bias": g("mm.0.bias"),
          "merger.linear_fc2.weight": g("mm.2.weight"), "merger.linear_fc2.bias": g("mm.2.bias")}
    names = [("ln1", "norm1"), ("ln2", "norm2"), ("attn_qkv", "attn.qkv"), ("attn_out", "attn.proj"),
             ("ffn_up", "mlp.linear_fc1"), ("ffn_down", "mlp.linear_fc2")]
    for i in range(27):
      for a, b in names:
        sd[f"blocks.{i}.{b}.weight"], sd[f"blocks.{i}.{b}.bias"] = g(f"v.blk.{i}.{a}.weight"), g(f"v.blk.{i}.{a}.bias")
    res = ref.load_state_dict(sd, strict=False)
    self.assertEqual((list(res.missing_keys), list(res.unexpected_keys)), ([], []))
    with torch.no_grad(): want = ref(pv, thw).pooler_output.numpy()
    enc = VisionEncoder.from_mmproj(str(MMPROJ), "CPU")
    got = enc(Tensor(pv.numpy()), (1, 28, 28)).numpy()
    self.assertEqual(got.shape, want.shape)
    cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
    print(f"\nparity vs transformers: tokens={len(cos)} cos min={cos.min():.6f} mean={cos.mean():.6f} "
          f"max|d|={np.abs(got - want).max():.4f} |ref|max={np.abs(want).max():.2f}")
    self.assertGreater(float(cos.min()), 0.999)

if __name__ == "__main__":
  unittest.main()
