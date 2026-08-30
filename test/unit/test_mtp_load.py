import os, struct, pathlib, unittest
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, TransformerConfig, MTPHead, _rename_mtp_keys
from tinygrad.llm.gguf import gguf_load, _parse_header

# T4.63: MTP ("nextn") loading. Tiny qwen35-shaped config -- attn_output_gate and qk_norm==head_dim are
# qwen35's real signature (model.py: attn_output_gate=arch in ('qwen35','qwen35moe'), and
# Transformer.__init__'s `if config.ssm: config = replace(config, qk_norm=config.head_dim)`).
DIM, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM, VOCAB, NUM_BLOCKS = 8, 16, 2, 2, 4, 11, 2

def _build_gguf(tensors:list[tuple[str, tuple[int, ...], bytes]], kvs:list[tuple[str, object]]) -> bytes:
  """[header][kv_data][tensor_infos][padding][tensor_data_blob] -- same layout as
  test/unit/test_gguf.py's TestGGUF._build_gguf, extended with float (type 6) and an int32-array
  (type 9 wrapping type 5) KV form for the keys a qwen35 config needs beyond that helper's str/uint32
  pair (rope_theta/norm_eps, tokenizer.ggml.tokens). `dims` per tensor is the desired FINAL (already
  numpy/row-major) shape -- ne[] on disk is its reverse (ggml convention); gguf.py's _gguf_parse
  un-reverses it back on load, so a caller here never has to think about the on-disk order."""
  buf = bytearray()
  buf += struct.pack("<4siqq", b"GGUF", 3, len(tensors), len(kvs))
  for k, v in kvs:
    kb = k.encode()
    buf += struct.pack("<Q", len(kb)) + kb
    if isinstance(v, str):
      vb = v.encode()
      buf += struct.pack("<i", 8) + struct.pack("<Q", len(vb)) + vb
    elif isinstance(v, float):
      buf += struct.pack("<i", 6) + struct.pack("<f", v)
    elif isinstance(v, list):  # array of int32 -- only tokenizer.ggml.tokens needs this, only its length matters
      buf += struct.pack("<i", 9) + struct.pack("<i", 5) + struct.pack("<Q", len(v))
      for x in v: buf += struct.pack("<i", x)
    else:
      buf += struct.pack("<i", 4) + struct.pack("<I", v)
  data_off = 0
  for name, dims, data in tensors:
    nb = name.encode()
    buf += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
    for d in reversed(dims): buf += struct.pack("<Q", d)
    buf += struct.pack("<i", 0) + struct.pack("<Q", data_off)  # ggml_type=0 == float32, unquantized
    data_off += len(data)
  buf += b"\x00" * ((32 - len(buf) % 32) % 32)
  for _, _, data in tensors: buf += data
  return bytes(buf)

def _block_tensors(rng, prefix, dim, hidden, head_dim, n_heads, n_kv_heads, qk_norm):
  # qwen35's attn_output_gate doubles attn_q's output width (see model.py TransformerBlock.__init__)
  q_out, kv_out, attn_in = head_dim * n_heads * 2, head_dim * n_kv_heads, head_dim * n_heads
  def t(shape): return shape, rng.standard_normal(shape).astype(np.float32).tobytes()
  return [(f"{prefix}.attn_norm.weight", *t((dim,))),
          (f"{prefix}.attn_q.weight", *t((q_out, dim))),
          (f"{prefix}.attn_k.weight", *t((kv_out, dim))),
          (f"{prefix}.attn_v.weight", *t((kv_out, dim))),
          (f"{prefix}.attn_q_norm.weight", *t((qk_norm,))),
          (f"{prefix}.attn_k_norm.weight", *t((qk_norm,))),
          (f"{prefix}.attn_output.weight", *t((dim, attn_in))),
          (f"{prefix}.post_attention_norm.weight", *t((dim,))),
          (f"{prefix}.ffn_gate.weight", *t((hidden, dim))),
          (f"{prefix}.ffn_up.weight", *t((hidden, dim))),
          (f"{prefix}.ffn_down.weight", *t((dim, hidden)))]

def _build_tiny_qwen35_gguf(num_blocks=NUM_BLOCKS, dim=DIM, hidden=HIDDEN, head_dim=HEAD_DIM, n_heads=N_HEADS,
                             n_kv_heads=N_KV_HEADS, vocab=VOCAB, nextn=1, seed=0) -> bytes:
  """A minimal qwen35-arch GGUF with `num_blocks` real (full-attention) blocks plus, when nextn=1, an
  extra blk.{num_blocks} nextn/MTP block carrying the ground-truth qwen3.5 tensor set (T4.63 task
  spec). full_attention_interval=1 keeps every real block a plain TransformerBlock too (no
  GatedDeltaNet weights needed): (i+1) % 1 is always 0, so ssm_layers is all-False."""
  rng = np.random.default_rng(seed)
  def w(shape): return rng.standard_normal(shape).astype(np.float32)
  tensors = [("token_embd.weight", (vocab, dim), w((vocab, dim)).tobytes()),
             ("output_norm.weight", (dim,), w((dim,)).tobytes()),
             ("output.weight", (vocab, dim), w((vocab, dim)).tobytes())]
  for i in range(num_blocks): tensors += _block_tensors(rng, f"blk.{i}", dim, hidden, head_dim, n_heads, n_kv_heads, head_dim)
  if nextn:
    tensors += _block_tensors(rng, f"blk.{num_blocks}", dim, hidden, head_dim, n_heads, n_kv_heads, head_dim)
    tensors += [(f"blk.{num_blocks}.nextn.eh_proj.weight", (dim, 2 * dim), w((dim, 2 * dim)).tobytes()),
                (f"blk.{num_blocks}.nextn.enorm.weight", (dim,), w((dim,)).tobytes()),
                (f"blk.{num_blocks}.nextn.hnorm.weight", (dim,), w((dim,)).tobytes()),
                (f"blk.{num_blocks}.nextn.shared_head_norm.weight", (dim,), w((dim,)).tobytes())]
  kvs = [("general.architecture", "qwen35"), ("qwen35.block_count", num_blocks + nextn),
         ("qwen35.nextn_predict_layers", nextn), ("qwen35.context_length", 64),
         ("qwen35.embedding_length", dim), ("qwen35.feed_forward_length", hidden),
         ("qwen35.attention.head_count", n_heads), ("qwen35.attention.head_count_kv", n_kv_heads),
         ("qwen35.attention.key_length", head_dim), ("qwen35.attention.value_length", head_dim),
         ("qwen35.attention.layer_norm_rms_epsilon", 1e-5), ("qwen35.rope.freq_base", 10000.0),
         ("qwen35.full_attention_interval", 1),
         ("qwen35.ssm.conv_kernel", 2), ("qwen35.ssm.state_size", 2), ("qwen35.ssm.group_count", 1),
         ("qwen35.ssm.time_step_rank", 1), ("qwen35.ssm.inner_size", 2),
         ("tokenizer.ggml.tokens", [0] * vocab)]
  return _build_gguf(tensors, kvs)

def _gguf_tensor(raw:bytes) -> Tensor: return Tensor(np.frombuffer(raw, dtype=np.uint8)).to(None)

class TestRenameMtpKeys(unittest.TestCase):
  """Pure key-manipulation unit tests for model.py's _rename_mtp_keys -- no GGUF involved."""
  def test_renames_nextn_and_block_keys(self):
    sd = {"blk.4.attn_q.weight": "q", "blk.4.ffn_norm.weight": "fn",
          "blk.4.nextn.eh_proj.weight": "eh", "blk.4.nextn.enorm.weight": "en",
          "blk.4.nextn.hnorm.weight": "hn", "blk.4.nextn.shared_head_norm.weight": "shn",
          "blk.3.attn_q.weight": "keep", "token_embd.weight": "keep2"}
    _rename_mtp_keys(sd, num_blocks=4)
    self.assertEqual(sd, {"mtp_head.block.attn_q.weight": "q", "mtp_head.block.ffn_norm.weight": "fn",
                          "mtp_head.eh_proj.weight": "eh", "mtp_head.enorm.weight": "en",
                          "mtp_head.hnorm.weight": "hn", "mtp_head.shared_head_norm.weight": "shn",
                          "blk.3.attn_q.weight": "keep", "token_embd.weight": "keep2"})

  def test_noop_without_matching_prefix(self):
    sd = {"blk.0.attn_q.weight": "a", "token_embd.weight": "b"}
    before = dict(sd)
    _rename_mtp_keys(sd, num_blocks=4)  # nothing starts with "blk.4."
    self.assertEqual(sd, before)

class TestMTPLoadSynthetic(unittest.TestCase):
  def _load(self, mtp:int, max_context:int=16, **kw):
    gguf_tensor = _gguf_tensor(_build_tiny_qwen35_gguf(**kw))
    with Context(MTP=mtp):
      return Transformer.from_gguf(gguf_tensor, max_context=max_context, device_map=None, realize=False)

  def test_mtp_0_drops_nextn_block(self):
    model, _ = self._load(mtp=0)
    self.assertIsNone(model.mtp_head)
    self.assertEqual(len(model.blk), NUM_BLOCKS)
    # nothing under blk.{NUM_BLOCKS}. made it into the model -- exactly what makes nn/state.py's
    # load_state_dict print "WARNING: unused weights in state_dict [...]" under DEBUG>=1
    self.assertFalse(any(k.startswith(f"blk.{NUM_BLOCKS}.") for k in nn.state.get_state_dict(model)))

  def test_mtp_1_loads_nextn_block_with_no_leftover(self):
    model, _ = self._load(mtp=1, seed=0)
    self.assertIsNotNone(model.mtp_head)
    mh = model.mtp_head
    self.assertFalse(any(k.startswith(f"blk.{NUM_BLOCKS}.") for k in nn.state.get_state_dict(model)))  # fully renamed away
    self.assertEqual(len(nn.state.get_state_dict(mh)), 15)  # ground-truth qwen3.5 nextn tensor count (T4.63 spec)

    self.assertEqual(mh.enorm.weight.shape, (DIM,))
    self.assertEqual(mh.hnorm.weight.shape, (DIM,))
    self.assertEqual(mh.eh_proj.weight.shape, (DIM, 2 * DIM))
    self.assertEqual(mh.shared_head_norm.weight.shape, (DIM,))
    self.assertEqual(mh.block.attn_norm.weight.shape, (DIM,))
    self.assertEqual(mh.block.ffn_norm.weight.shape, (DIM,))  # post_attention_norm -> ffn_norm rename
    self.assertEqual(mh.block.attn_q.weight.shape, (HEAD_DIM * N_HEADS * 2, DIM))  # *2: attn_output_gate
    self.assertEqual(mh.block.attn_k.weight.shape, (HEAD_DIM * N_KV_HEADS, DIM))
    self.assertEqual(mh.block.attn_v.weight.shape, (HEAD_DIM * N_KV_HEADS, DIM))
    self.assertEqual(mh.block.attn_q_norm.weight.shape, (HEAD_DIM,))
    self.assertEqual(mh.block.attn_k_norm.weight.shape, (HEAD_DIM,))
    self.assertEqual(mh.block.attn_output.weight.shape, (DIM, HEAD_DIM * N_HEADS))
    self.assertEqual(mh.block.ffn_gate.weight.shape, (HIDDEN, DIM))
    self.assertEqual(mh.block.ffn_up.weight.shape, (HIDDEN, DIM))
    self.assertEqual(mh.block.ffn_down.weight.shape, (DIM, HIDDEN))

    # values, not just shapes/keys, actually moved -- catches a same-shape rename swap (e.g. enorm<->hnorm,
    # attn_norm<->ffn_norm all share shape (DIM,)). Loose tolerance: from_gguf casts weights to float16.
    _, ref_sd = gguf_load(_gguf_tensor(_build_tiny_qwen35_gguf(seed=0)))
    np.testing.assert_allclose(mh.enorm.weight.numpy(), ref_sd[f"blk.{NUM_BLOCKS}.nextn.enorm.weight"].numpy(),
                                atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(mh.block.attn_q.weight.numpy(), ref_sd[f"blk.{NUM_BLOCKS}.attn_q.weight"].numpy(),
                                atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(mh.block.ffn_norm.weight.numpy(),
                                ref_sd[f"blk.{NUM_BLOCKS}.post_attention_norm.weight"].numpy(), atol=1e-2, rtol=1e-2)

  def test_mtp_0_and_1_agree_on_the_main_model(self):
    # MTP is a pure bolt-on: the main model's weights/logits must be identical whichever way it's set
    model0, _ = self._load(mtp=0, seed=1)
    model1, _ = self._load(mtp=1, seed=1)
    for b0, b1 in zip(model0.blk, model1.blk):
      np.testing.assert_array_equal(b0.attn_q.weight.numpy(), b1.attn_q.weight.numpy())
      np.testing.assert_array_equal(b0.ffn_gate.weight.numpy(), b1.ffn_gate.weight.numpy())
    out0 = [v for _, v in zip(range(4), model0.generate([1, 2, 3]))]
    out1 = [v for _, v in zip(range(4), model1.generate([1, 2, 3]))]
    self.assertEqual(out0, out1)

  def test_draft_shape_deterministic_and_kv_advances(self):
    model, _ = self._load(mtp=1)
    mh = model.mtp_head
    h0, tok0 = Tensor.randn(1, 1, DIM), Tensor([[3]], dtype=dtypes.int32)
    out1 = mh.draft(model, h0, tok0, 0).realize()
    self.assertEqual(out1.shape, (1, 1, VOCAB))
    self.assertTrue(np.isfinite(out1.numpy()).all())
    cache_after_1 = mh.block.cache_kv.numpy().copy()

    # determinism: identical inputs at the same start_pos reproduce identical logits (idempotent slice overwrite)
    out1_again = mh.draft(model, h0, tok0, 0).realize()
    np.testing.assert_array_equal(out1.numpy(), out1_again.numpy())

    # a second call at start_pos+1 advances the block's OWN kv cache: writes a new slot, leaves the old one alone
    h1, tok1 = Tensor.randn(1, 1, DIM), Tensor([[7]], dtype=dtypes.int32)
    out2 = mh.draft(model, h1, tok1, 1).realize()
    self.assertEqual(out2.shape, (1, 1, VOCAB))
    self.assertTrue(np.isfinite(out2.numpy()).all())
    cache_after_2 = mh.block.cache_kv.numpy()
    np.testing.assert_array_equal(cache_after_2[..., 0, :], cache_after_1[..., 0, :])                     # slot 0 untouched
    self.assertFalse(np.array_equal(cache_after_2[..., 1, :], np.zeros_like(cache_after_2[..., 1, :])))   # slot 1 now written

REAL_GGUF = "/Users/artur/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"

class TestMTPLoadRealMetadata(unittest.TestCase):
  """Ground truth from the real file (T4.63 task spec): arch=qwen35, block_count=65,
  nextn_predict_layers=1 -> num_blocks=64, blk.64 carries exactly 15 tensors (11 shared with a normal
  full-attention block's names + 4 nextn.*). Header-only: never stages/realizes a single tensor's
  bytes (gguf.py's _gguf_parse only does that in flush(), which this never calls), so this stays
  cheap regardless of the real file's ~27GB size."""
  @unittest.skipUnless(os.path.exists(REAL_GGUF), f"real qwen3.8 gguf not present: {REAL_GGUF}")
  def test_real_file_nextn_keys_all_consumed_by_rename(self):
    size = os.stat(REAL_GGUF).st_size
    header = Tensor(pathlib.Path(REAL_GGUF))[:min(size, 64 * 1024 * 1024)].to(None).realize()
    kv, t_infos, _ = _parse_header(header)
    arch = kv['general.architecture']
    self.assertEqual(arch, 'qwen35')
    self.assertEqual(kv[f'{arch}.block_count'], 65)
    self.assertEqual(kv[f'{arch}.nextn_predict_layers'], 1)
    self.assertEqual(kv[f'{arch}.embedding_length'], 5120)
    self.assertEqual(kv[f'{arch}.feed_forward_length'], 17408)
    self.assertEqual(kv[f'{arch}.attention.head_count'], 24)
    self.assertEqual(kv[f'{arch}.attention.head_count_kv'], 4)
    self.assertEqual(kv[f'{arch}.attention.key_length'], 256)
    self.assertEqual(kv[f'{arch}.attention.value_length'], 256)
    num_blocks = kv[f'{arch}.block_count'] - kv[f'{arch}.nextn_predict_layers']  # from_gguf's own formula
    self.assertEqual(num_blocks, 64)

    nextn_names = {name for name, *_ in t_infos if name.startswith(f"blk.{num_blocks}.")}
    self.assertEqual(len(nextn_names), 15)  # the ground-truth blk.64 tensor count (task spec)
    for expected in (f"blk.{num_blocks}.nextn.eh_proj.weight", f"blk.{num_blocks}.attn_q.weight",
                     f"blk.{num_blocks}.post_attention_norm.weight"):
      self.assertIn(expected, nextn_names)

    # from_gguf's arch-level rename (post_attention_norm -> ffn_norm) runs before _rename_mtp_keys and
    # applies to blk.{num_blocks} too -- replicate it here before checking the mapping, same as from_gguf does.
    dummy_sd = {k.replace('post_attention_norm', 'ffn_norm'): None for k in nextn_names}
    _rename_mtp_keys(dummy_sd, num_blocks)
    self.assertFalse(any(k.startswith(f"blk.{num_blocks}.") for k in dummy_sd))  # no leftover blk.64.* key

    cfg = TransformerConfig(num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
                             hidden_dim=kv[f'{arch}.feed_forward_length'], n_heads=kv[f'{arch}.attention.head_count'],
                             n_kv_heads=kv[f'{arch}.attention.head_count_kv'], norm_eps=1e-5, vocab_size=8,
                             head_dim=kv[f'{arch}.attention.key_length'], rope_theta=1e6,
                             rope_dim=kv[f'{arch}.attention.key_length'], v_head_dim=kv[f'{arch}.attention.value_length'],
                             max_context=16, qk_norm=kv[f'{arch}.attention.key_length'], attn_output_gate=True)
    expected_keys = set(nn.state.get_state_dict(MTPHead(cfg), prefix="mtp_head."))
    self.assertEqual(set(dummy_sd.keys()), expected_keys)

if __name__ == '__main__':
  unittest.main()
