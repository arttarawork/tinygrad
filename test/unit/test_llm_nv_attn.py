"""T4.107a: NV split-KV decode attention (kernels/nv_attn.py) -- the kernel graph builds for the real geometry and renders on sm_86
with its 32 lanes as one warp, the merge and the in-kernel dequant convention agree with a plain-tensor reference on CPU, and the
NV_CUSTOM_ATTN gate leaves every other path untouched. Kernel numerics on the card: extra/nv_attn_validate_real.py."""
import math, re, unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, Context
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.llm.kernels import nv_attn
from tinygrad.llm.kernels.nv_attn import BLOCK_N, MAX_CHUNKS

B, H, KVH, D = 1, 24, 4, 256  # qwen3.8-27B's attention geometry

def _cache(dtype, N:int, D:int=D):
  kv = Tensor.empty(2, B, KVH, N, D // 2 if dtype == dtypes.uint8 else D, dtype=dtype)
  scale = Tensor.empty(2, B, KVH, N, D // 32, dtype=dtypes.float16) if dtype != dtypes.float16 else None
  return kv, scale

def _sink(dtype, N:int, valid):
  kv, scale = _cache(dtype, N)
  chunks = min(MAX_CHUNKS, N // BLOCK_N)
  srcs = [Tensor.empty(B, H, chunks, D, dtype=dtypes.float32).uop, Tensor.empty(B, H, chunks, 2, dtype=dtypes.float32).uop,
          Tensor.empty(B, H, 1, D, dtype=dtypes.float16).uop, kv.uop] + ([scale.uop] if scale is not None else [])
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs))
  return nv_attn._decode_partial_kernel(*params, valid_kv_len=valid, max_kv_len=N)

class TestKernelGraphAndRender(unittest.TestCase):
  def test_graph_builds_for_every_cache_dtype(self):
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      for valid in (1, 4095, UOp.variable("start_pos", 0, 4095) + 1):
        with self.subTest(dtype=dtype, valid=str(valid)):
          sink = _sink(dtype, 4096, valid)
          self.assertIs(sink.op, Ops.SINK)
          self.assertTrue(any(u.op is Ops.CUSTOM and "__shfl_xor_sync" in u.arg[0] for u in sink.toposort()))

  def test_render_keeps_the_warp_together(self):
    # every full_wave warp_reduce assumes the 32-lane LOCAL range is exactly one warp: whatever way the lowerer splits it, the block's
    # thread dims must multiply to 32 and nothing else may sit on a thread dim (no wave/head dim), else __shfl_xor_sync mixes blocks
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      with self.subTest(dtype=dtype):
        prg = to_program(_sink(dtype, 4096, UOp.variable("start_pos", 0, 4095) + 1), renderer)
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn("__shfl_xor_sync(0xffffffff", src)
        sizes = [int(n) for n in re.findall(r"threadIdx\.[xyz]; /\* (\d+) \*/", src)]
        self.assertTrue(sizes and math.prod(sizes) == 32, f"block is not one warp: {sizes}")
        self.assertIn("start_pos", src)  # the bound decode position reaches the kernel as a parameter

class TestMathOnCPU(unittest.TestCase):
  def test_merge_partials_equals_full_softmax(self):
    rng = np.random.default_rng(0)
    heads, chunks, keys, d = 3, 5, 7, 4
    scores = rng.standard_normal((heads, chunks, keys)).astype(np.float32)
    vals = rng.standard_normal((heads, chunks, keys, d)).astype(np.float32)
    m = scores.max(-1)                                                           # per-chunk running max
    e = np.exp(scores - m[..., None])
    acc, l = (e[..., None] * vals).sum(2), e.sum(-1)                              # per-chunk (acc, l) as the kernel writes them
    partial = Tensor(acc.reshape(1, heads, chunks, d))
    stats = Tensor(np.stack([m, l], -1).reshape(1, heads, chunks, 2))
    out = nv_attn._merge_partials(partial, stats, chunks * BLOCK_N).numpy().reshape(heads, d)
    full = scores.reshape(heads, -1)
    p = np.exp(full - full.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ref = (p[..., None] * vals.reshape(heads, -1, d)).sum(1)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

  def test_merge_only_uses_live_chunks(self):
    partial = Tensor.ones(1, 1, 4, 2)
    stats = Tensor([[[[0., 1.], [0., 1.], [100., 1.], [100., 1.]]]])
    self.assertTrue(np.allclose(nv_attn._merge_partials(partial, stats, BLOCK_N + 1).numpy(), 1.0))  # 2 live slots, the 100s are stale

  def _dequant_reference(self, kv:Tensor, scale:Tensor|None, N:int):
    if kv.dtype == dtypes.uint8:  # model.py's _dequant, uint8 branch
      lo, hi = kv.bitwise_and(0xF), kv.rshift(4)
      qv = Tensor.stack(lo, hi, dim=-1).reshape(*kv.shape[:-1], D).float() - 8
      return (qv.reshape(*qv.shape[:-1], D // 32, 32) * scale.float().unsqueeze(-1)).reshape(qv.shape)
    if kv.dtype == dtypes.int8: return (kv.float().reshape(*kv.shape[:-1], D // 32, 32) * scale.float().unsqueeze(-1)).reshape(kv.shape)
    return kv.float()

  def test_kv_value_matches_the_cache_layout(self):
    # the same kv_value the warp kernel uses, driven by plain GLOBAL ranges on CPU, must reproduce model.py's dequant of a random cache
    N, rng = 64, np.random.default_rng(1)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      with self.subTest(dtype=dtype):
        kv, scale = _cache(dtype, N)
        raw = rng.integers(0, 256, size=kv.shape, dtype=np.uint8) if dtype == dtypes.uint8 else \
          rng.integers(-127, 128, size=kv.shape).astype(np.int8) if dtype == dtypes.int8 else rng.standard_normal(kv.shape).astype(np.float16)
        kv = Tensor(raw, dtype=dtype).contiguous().realize()
        scale = Tensor(rng.uniform(0.01, 1, size=scale.shape).astype(np.float16)).contiguous().realize() if scale is not None else None
        out = Tensor.empty(2, KVH, N, D, dtype=dtypes.float32)
        def fxn(o:UOp, c:UOp, *s:UOp) -> UOp:
          kvr, hr, key, d = UOp.range(2, 0), UOp.range(KVH, 1), UOp.range(N, 2), UOp.range(D, 3)
          val = nv_attn.kv_value(c, s[0] if s else None, kvr, 0, hr, key, d)
          return o[kvr, hr, key, d].store(val).end(kvr, hr, key, d).sink(arg=KernelInfo(name="kv_value_cpu", opts_to_apply=()))
        got = Tensor.custom_kernel(out, kv, *([scale] if scale is not None else []), fxn=fxn)[0].numpy()
        ref = self._dequant_reference(kv, scale, N).numpy()[:, 0]
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

def _prefill_sink(dtype, N:int, valid, q_start, q_rows:int=1, M:int=32):
  kv, scale = _cache(dtype, N)
  chunks = min(nv_attn.PREFILL_CHUNKS, N // BLOCK_N)
  srcs = [Tensor.empty(B, H, M, chunks, D, dtype=dtypes.float32).uop, Tensor.empty(B, H, M, chunks, 2, dtype=dtypes.float32).uop,
          Tensor.empty(B, H, M, D, dtype=dtypes.float16).uop, kv.uop] + ([scale.uop] if scale is not None else [])
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs))
  return nv_attn._prefill_partial_kernel(*params, valid_kv_len=valid, q_start=q_start, max_kv_len=N, q_rows=q_rows)

class TestPrefillKernel(unittest.TestCase):
  """T4.107b: the prefill kernel = the decode kernel over query rows with a per-row causal limit."""
  def test_graph_builds_for_every_cache_dtype(self):
    sp = UOp.variable("start_pos", 0, 4095 - 32)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      for q_rows in (1, 2, 4):
        for valid, q_start in ((32, 0), (4095, 4063), (sp + 32, sp)):
          with self.subTest(dtype=dtype, q_rows=q_rows, valid=str(valid)):
            sink = _prefill_sink(dtype, 4096, valid, q_start, q_rows)
            self.assertIs(sink.op, Ops.SINK)
            self.assertTrue(any(u.op is Ops.CUSTOM and "__shfl_xor_sync" in u.arg[0] for u in sink.toposort()))

  def test_render_keeps_the_warp_together(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    sp = UOp.variable("start_pos", 0, 4095 - 32)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      with self.subTest(dtype=dtype):
        prg = to_program(_prefill_sink(dtype, 4096, sp + 32, sp), renderer)
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn("__shfl_xor_sync(0xffffffff", src)
        sizes = [int(n) for n in re.findall(r"threadIdx\.[xyz]; /\* (\d+) \*/", src)]
        self.assertTrue(sizes and math.prod(sizes) == 32, f"block is not one warp: {sizes}")
        self.assertIn("start_pos", src)  # the chunk's cache position (causal limit + load mask) reaches the kernel as a parameter

  def _reference(self, q:np.ndarray, kv:np.ndarray, valid:int, q_start:int) -> np.ndarray:
    """(H,M,D) causal attention of q (H,M,D) over the dequantized cache (2,KVH,N,D): row r sees keys j <= q_start + r, j < valid."""
    Hq, M, Dh = q.shape
    G = Hq // kv.shape[1]
    out = np.zeros((Hq, M, Dh), np.float64)
    for h in range(Hq):
      k, v = kv[0, h // G].astype(np.float64), kv[1, h // G].astype(np.float64)
      for r in range(M):
        n = min(q_start + r + 1, valid)
        s = k[:n] @ q[h, r].astype(np.float64) / math.sqrt(Dh)
        p = np.exp(s - s.max())
        p /= p.sum()
        out[h, r] = p @ v[:n]
    return out

  def test_kernel_matches_numpy_on_cpu(self):
    # the whole path (pad, kernel with one lane owning the head, merge, slice) on CPU: lanes=1 makes the dot an in-lane sum, so the
    # graph is the warp kernel minus the shuffles -- causal limit, load mask, online softmax across KEY_GROUPs and chunk slots, and
    # the SENTINEL slots (rows whose limit sits below a slot) are all exercised against a plain numpy softmax. Concrete lengths only:
    # the CPU loop form of a SYMBOLIC length linearises into a chain deeper than the interpreter stack (the NV form has no loop --
    # the symbolic plumbing is the decode kernel's, card-validated, and pinned by the render test above)
    rng = np.random.default_rng(2)
    Hq, KVH, Dh, N, M = 4, 2, 64, 256, 8
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      for q_rows, q_start, symbolic in ((1, 100, False), (2, 100, False), (4, 242, False), (1, 0, False)):
        with self.subTest(dtype=dtype, q_rows=q_rows, q_start=q_start, symbolic=symbolic):
          kv = Tensor.empty(2, 1, KVH, N, Dh // 2 if dtype == dtypes.uint8 else Dh, dtype=dtype)
          raw = rng.integers(0, 256, size=kv.shape, dtype=np.uint8) if dtype == dtypes.uint8 else \
            rng.integers(-127, 128, size=kv.shape).astype(np.int8) if dtype == dtypes.int8 else rng.standard_normal(kv.shape).astype(np.float16)
          kv = Tensor(raw, dtype=dtype).contiguous().realize()
          scale = Tensor(rng.uniform(0.01, 0.2, size=(2, 1, KVH, N, Dh // 32)).astype(np.float16)).contiguous().realize() \
            if dtype != dtypes.float16 else None
          q_np = (rng.standard_normal((1, Hq, M, Dh)) * 2).astype(np.float16)
          q = Tensor(q_np).contiguous().realize()
          valid = q_start + M
          if symbolic:
            v = UOp.variable("start_pos", 0, N - M)
            vl, qs = v.bind(q_start) + M, v.bind(q_start)
          else: vl, qs = valid, q_start
          with Context(NV_ATTN_QROWS=q_rows):
            got = nv_attn.nv_prefill_attention(q, kv, scale, vl, qs, lanes=1).numpy()[0]
          ref_kv = _dequant_reference_np(kv, scale, Dh)
          ref = self._reference(q_np[0], ref_kv, valid, q_start)
          self.assertFalse(np.isnan(got).any())
          np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

  def test_symbolic_chunk_is_padded_and_sliced(self):
    # a bound chunk width below the static tile (generate()'s last, short chunk): the kernel sees the static T_pad rows (a multiple
    # of NV_ATTN_QROWS) and the result is sliced back to the chunk's own bound width -- graph-level (the CPU C renderer cannot
    # compile the symbolic slice; on the card the JIT does exactly this for every chunk)
    Hq, KVH, Dh, N = 4, 2, 64, 256
    kv = Tensor.zeros(2, 1, KVH, N, Dh, dtype=dtypes.float16)
    t = UOp.variable("toks", 1, 6)
    q = Tensor.ones(1, Hq, 6, Dh, dtype=dtypes.float16)[:, :, :t.bind(5)]
    with Context(NV_ATTN_QROWS=4):
      out = nv_attn.nv_prefill_attention(q, kv, None, 100 + t.bind(5), 100)
    self.assertEqual(out.max_shape, (1, Hq, 6, Dh))
    self.assertIs(out.shape[2].unbind()[0], t)                                   # the chunk's own (bound) width survives
    nodes = list(out.uop.toposort())
    self.assertEqual(sum(u.op is Ops.CALL for u in nodes), 1)                     # exactly one custom kernel call
    self.assertEqual([u.shape for u in nodes if u.op is Ops.PAD], [(1, Hq, 8, Dh)])  # q padded to 2 tiles of 4 rows

def _dequant_reference_np(kv:Tensor, scale:Tensor|None, Dh:int) -> np.ndarray:
  if kv.dtype == dtypes.uint8:
    lo, hi = kv.bitwise_and(0xF), kv.rshift(4)
    qv = Tensor.stack(lo, hi, dim=-1).reshape(*kv.shape[:-1], Dh).float() - 8
    return (qv.reshape(*qv.shape[:-1], Dh // 32, 32) * scale.float().unsqueeze(-1)).reshape(qv.shape).numpy()[:, 0]
  if kv.dtype == dtypes.int8: return (kv.float().reshape(*kv.shape[:-1], Dh // 32, 32) * scale.float().unsqueeze(-1)).reshape(kv.shape).numpy()[:, 0]
  return kv.float().numpy()[:, 0]

class TestGates(unittest.TestCase):
  def test_off_by_default_and_never_on_cpu(self):
    kv, _ = _cache(dtypes.uint8, 4096)
    self.assertFalse(nv_attn.nv_decode_attention_ok("NV", D, kv))              # default 0: no device is even consulted
    with Context(NV_CUSTOM_ATTN=1):
      self.assertFalse(nv_attn.nv_decode_attention_ok("CPU", D, kv))
      self.assertFalse(nv_attn.nv_decode_attention_ok(("CPU", "CPU"), D, kv))
      self.assertFalse(nv_attn.nv_decode_attention_ok(None, D, kv))
      kv_odd, _ = _cache(dtypes.uint8, 4096 + 1)
      self.assertFalse(nv_attn.nv_decode_attention_ok("NV", 8, kv))           # head_dim 8: no whole lanes
      self.assertFalse(nv_attn.nv_decode_attention_ok("NV", D, kv_odd))       # cache length not in whole chunks

  def test_model_path_unchanged_with_the_flag_on_cpu(self):
    from tinygrad.llm.model import Transformer, TransformerConfig
    cfg = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=100,
                            head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32)
    def run():
      Tensor.manual_seed(0)
      model = Transformer(cfg)
      h = model.token_embd(Tensor([[1, 2, 3]], dtype="int32")).float()
      for block in model.blk: h = block(h, 0)
      h = h[:, -1:]
      for block in model.blk: h = block(h, 3)   # a T==1 decode step: the gate is consulted and must fall through on CPU
      return h.numpy()
    a = run()
    with Context(NV_CUSTOM_ATTN=1): b = run()
    np.testing.assert_array_equal(a, b)

if __name__ == "__main__": unittest.main()
