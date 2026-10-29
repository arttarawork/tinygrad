"""T4.107a: NV split-KV decode attention (kernels/nv_attn.py) -- the kernel graph builds for the real geometry and renders on sm_86
with its 32 lanes as one warp, the merge and the in-kernel dequant convention agree with a plain-tensor reference on CPU, and the
NV_CUSTOM_ATTN gate leaves every other path untouched. Kernel numerics on the card: extra/nv_attn_validate_real.py.
T4.130: the GQA shared-tile prefill kernel is executed on the PYTHON emulator against numpy (TestPrefillGQAKernel)."""
import contextlib, itertools, math, pickle, re, unittest
from unittest import mock
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, dtypes, Context
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, KernelInfo
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

  def test_kernel_matches_numpy_on_cpu_wide_chunk(self):
    # T4.124b: T=128/256 (MAX_PREFILL_ROWS raised from 64) against a cache that spans more live 128-key chunks than the
    # scaled slot count (N=4096, valid ~3000+T > slots*BLOCK_N: 16*128=2048 at T=128, 8*128=1024 at T=256) -- exercises the
    # per-slot `offset` REDUCE loop's multiple passes over the same slot (several real chunks folded into one output slot).
    rng = np.random.default_rng(3)
    Hq, KVH, Dh, N = 4, 2, 64, 4096
    for dtype in (dtypes.uint8, dtypes.float16):   # int4-packed and plain fp16 caches (the two the design doc names)
      for T in (128, 256):
        with self.subTest(dtype=dtype, T=T):
          shape = (2, 1, KVH, N, Dh // 2 if dtype == dtypes.uint8 else Dh)
          raw = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == dtypes.uint8 else rng.standard_normal(shape).astype(np.float16)
          kv = Tensor(raw, dtype=dtype).contiguous().realize()
          scale = Tensor(rng.uniform(0.01, 0.2, size=(2, 1, KVH, N, Dh // 32)).astype(np.float16)).contiguous().realize() \
            if dtype != dtypes.float16 else None
          q_np = (rng.standard_normal((1, Hq, T, Dh)) * 2).astype(np.float16)
          q = Tensor(q_np).contiguous().realize()
          q_start, valid = 3000, 3000 + T
          # q_rows stays at its default (1): the kernel-builder python-unrolls the row/head/key loops by q_rows*G*KEY_GROUP,
          # so q_rows=T would blow that up to tens of thousands of UOps (hung the CPU lowerer + CHECK_OOB's z3 pass past the
          # 300s test-timeout watchdog, see conftest.py). tiles=T//q_rows=T absorbs the width as a RUNTIME block-range bound.
          got = nv_attn.nv_prefill_attention(q, kv, scale, valid, q_start, lanes=1).numpy()[0]
          ref_kv = _dequant_reference_np(kv, scale, Dh)
          ref = self._reference(q_np[0], ref_kv, valid, q_start)
          self.assertFalse(np.isnan(got).any())
          np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

  def test_slot_count_scales_with_t_pad(self):
    # T4.124b: the partial/stats buffers' slot count is PREFILL_CHUNKS*32//max(T_pad,32) -- 64 (unchanged) at T_pad<=32,
    # 32/16/8 at 64/128/256 -- so the partial buffer stays ~48 MiB at every width instead of growing with T_pad (design
    # doc section 3.3). T_pad<=32 keeping 64 slots is exactly PB.2's byte-identity requirement for the standing width.
    Hq, KVH, Dh, N = 4, 2, 64, 8192  # N >= PREFILL_CHUNKS(64) * BLOCK_N(128) so the unscaled cap never binds here
    kv = Tensor.zeros(2, 1, KVH, N, Dh, dtype=dtypes.float16)
    for T_pad, want_chunks in ((32, 64), (64, 32), (128, 16), (256, 8)):
      q = Tensor.ones(1, Hq, T_pad, Dh, dtype=dtypes.float16)
      # q_rows stays at its default (1) -- see the wide_chunk test above for why NV_ATTN_QROWS=T_pad would blow up the
      # python-unrolled kernel builder; T_pad==T here since q_rows=1 divides every T.
      with patch.object(nv_attn.Tensor, "empty", wraps=nv_attn.Tensor.empty) as m:
        nv_attn.nv_prefill_attention(q, kv, None, T_pad, 0, lanes=1)
      self.assertEqual(len(m.call_args_list), 2)  # partial, stats
      for call in m.call_args_list: self.assertEqual(call.args[:4], (1, Hq, T_pad, want_chunks))  # (B,H,T_pad,chunks,...)

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

def _wmma_sink(dtype, N:int, valid, q_start, M:int=32, d_split:int=1):
  kv, scale = _cache(dtype, N)
  chunks = min(nv_attn.PREFILL_CHUNKS, N // BLOCK_N)
  srcs = [Tensor.empty(B, H, M, chunks, D, dtype=dtypes.float32).uop, Tensor.empty(B, H, M, chunks, 2, dtype=dtypes.float32).uop,
          Tensor.empty(B, H, M, D, dtype=dtypes.float16).uop, kv.uop] + ([scale.uop] if scale is not None else [])
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs))
  return nv_attn._prefill_wmma_kernel(*params, valid_kv_len=valid, q_start=q_start, max_kv_len=N, d_split=d_split)

class TestPrefillWMMAKernel(unittest.TestCase):
  """T4.113: the tensor-core prefill kernel builds and renders for the model's geometry; NV_ATTN_WMMA routes to it; the numerics are
  card-only (extra/nv_attn_validate_real.py --wmma) -- the CUDA renderer is the only one that emits mma.sync."""
  def test_graph_builds_for_every_cache_dtype(self):
    sp = UOp.variable("start_pos", 0, 4095 - 32)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      for d_split in (1, 2):
        for valid, q_start in ((32, 0), (4095, 4063), (sp + 32, sp)):
          with self.subTest(dtype=dtype, d_split=d_split, valid=str(valid)):
            sink = _wmma_sink(dtype, 4096, valid, q_start, d_split=d_split)
            self.assertIs(sink.op, Ops.SINK)
            nodes = list(sink.toposort())
            self.assertTrue(any(u.op is Ops.WMMA for u in nodes))
            self.assertTrue(any(u.op is Ops.CUSTOM and "__shfl_xor_sync(0xffffffff, {0}, 1)" in u.arg[0] for u in nodes))  # the row shuffles

  def test_render_uses_mma_and_pins_the_lane_split(self):
    # the fragment map assumes nv_gemm's (4, 8) lane split (hardware lane = 4*lane_hi + lane_lo) -- pinned the same way as test_llm_nv_gemm
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    sp = UOp.variable("start_pos", 0, 4095 - 32)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      with self.subTest(dtype=dtype):
        prg = to_program(_wmma_sink(dtype, 4096, sp + 32, sp), renderer)
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", src)
        self.assertIn("__shfl_xor_sync(0xffffffff", src)
        self.assertIn("int lidx0 = threadIdx.x; /* 4 */", src)
        self.assertIn("int lidx1 = threadIdx.y; /* 8 */", src)
        self.assertIn("start_pos", src)
        # the compile worker pool pickles the lowered PROGRAM back to the parent: the fully unrolled first cut overflowed the C stack
        # there (2026-09-16 card run); the RANGE-looped kernel must pickle at the default limits, on the main thread
        self.assertLess(len(pickle.dumps(prg)), 2_000_000)
        self.assertLess(len(src), 150_000)

  def test_render_at_wide_m_matches_m16(self):
    # T4.124b: MAX_PREFILL_ROWS 64 -> 256, so the model can now call this kernel at M up to 256 query rows. The spec's STOP
    # condition is a register-file or lane-split change at the wider M: tiles (M//WMMA_M) only widens the GLOBAL block range,
    # never the per-warp body, so the lane pins and the mma.sync count per source must match M=16 (one row tile) exactly.
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    srcs = {}
    for M in (16, 256):
      prg = to_program(_wmma_sink(dtypes.float16, 4096, 4095, 4063, M=M), renderer)
      srcs[M] = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
      self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", srcs[M])
      self.assertIn("int lidx0 = threadIdx.x; /* 4 */", srcs[M])
      self.assertIn("int lidx1 = threadIdx.y; /* 8 */", srcs[M])
      self.assertLess(len(pickle.dumps(prg)), 2_000_000)
    self.assertEqual(srcs[16].count("mma.sync"), srcs[256].count("mma.sync"))  # same per-warp body; only the GLOBAL range widens

  def test_routing_and_fallback(self):
    # graph-level: the flag picks the wmma kernel (16-row tiles) for a head_dim the fragments cover; a head_dim that is not a multiple
    # of 16 keeps the T4.107b kernel even with the flag on; the flag off changes nothing
    def kernel_names(Dh:int, wmma:int, lanes:int=nv_attn.WARP_SIZE):
      Hq, KVH, N = 4, 2, 256
      kv = Tensor.zeros(2, 1, KVH, N, Dh, dtype=dtypes.float16)
      q = Tensor.ones(1, Hq, 6, Dh, dtype=dtypes.float16)
      with Context(NV_ATTN_WMMA=wmma, NV_ATTN_GQA=0):  # T4.130: the GQA kernel (default) has its own routing test below
        out = nv_attn.nv_prefill_attention(q, kv, None, 106, 100, lanes=lanes)
      nodes = list(out.uop.toposort())
      names = [u.arg.name for u in nodes if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
      pads = [u.shape for u in nodes if u.op is Ops.PAD]
      return names, pads
    names, pads = kernel_names(64, 1)
    self.assertEqual(names, ["flash_prefill_wmma_nv"])
    self.assertEqual(pads, [(1, 4, 16, 64)])                                  # 6 rows padded to one 16-row tile
    self.assertEqual(kernel_names(64, 0)[0], ["flash_prefill_partial_nv"])   # off: unchanged
    # head_dim 8: no 16-wide k-step -> the T4.107b kernel even with the flag on (its warp form needs 32 | head_dim, so the one-lane CPU form)
    self.assertEqual(kernel_names(8, 1, lanes=1)[0], ["flash_prefill_partial_nv"])

def _dequant_reference_np(kv:Tensor, scale:Tensor|None, Dh:int) -> np.ndarray:
  if kv.dtype == dtypes.uint8:
    lo, hi = kv.bitwise_and(0xF), kv.rshift(4)
    qv = Tensor.stack(lo, hi, dim=-1).reshape(*kv.shape[:-1], Dh).float() - 8
    return (qv.reshape(*qv.shape[:-1], Dh // 32, 32) * scale.float().unsqueeze(-1)).reshape(qv.shape).numpy()[:, 0]
  if kv.dtype == dtypes.int8: return (kv.float().reshape(*kv.shape[:-1], Dh // 32, 32) * scale.float().unsqueeze(-1)).reshape(kv.shape).numpy()[:, 0]
  return kv.float().numpy()[:, 0]

# ******** T4.130: the GQA shared-tile prefill kernel -- numerics on the PYTHON emulator, render/compile for sm_86, routing ********

_emu_slot = itertools.count(200)
def _emulated_shfl_xor(val:UOp, offset:int, maximum:bool=False) -> UOp:
  """nv_attn._shfl_xor on the PYTHON emulator (no CUDA intrinsics): thread tid = warp*32 + lane writes val at [tid] of a block-shared
  buffer and reads [tid ^ offset] -- the partner __shfl_xor_sync reads, since the GQA kernel's lane is the WARP range (the hardware
  lane, unsplit). The emulator runs the block in lockstep, so the store is always seen (test_llm_nv_norm's _emulated_warp_reduce)."""
  rngs = {u.arg[-1]: u for u in val.toposort() if u.op is Ops.RANGE}
  lane, warp = rngs[AxisType.WARP], rngs[AxisType.LOCAL]
  tid, n = warp * nv_attn.WARP_SIZE + lane, warp.src[0].arg * nv_attn.WARP_SIZE
  buf = UOp.placeholder((n,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
  other = buf.after(buf[tid].store(val))[tid + offset - (tid // offset) % 2 * (2 * offset)].load()  # tid ^ offset, z3-boundable
  return val.maximum(other) if maximum else val + other

def _random_cache(rng:np.random.Generator, dtype, KVH:int, N:int, Dh:int, device:str):
  """(kv Tensor, scale Tensor|None, dequantised (2,KVH,N,Dh) float32) -- the layouts _cache/_dequant_reference_np use"""
  shape = (2, 1, KVH, N, Dh // 2 if dtype == dtypes.uint8 else Dh)
  raw = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == dtypes.uint8 else \
    rng.integers(-127, 128, size=shape).astype(np.int8) if dtype == dtypes.int8 else rng.standard_normal(shape).astype(np.float16)
  kv = Tensor(raw, dtype=dtype, device=device).contiguous().realize()
  scale = Tensor(rng.uniform(0.01, 0.05, size=(2, 1, KVH, N, Dh // 32)).astype(np.float16), device=device).contiguous().realize() \
    if dtype != dtypes.float16 else None
  return kv, scale, _dequant_reference_np(kv, scale, Dh)

def _gqa_sink(dtype, N:int, valid, q_start, M:int=128, warps:int=4, keys:int=32):
  """the kernel graph at the model's geometry: the int4 cache arrives as nv_prefill_attention binds it, uint32 words over the uint8 buffer"""
  kv, scale = _cache(dtype, N)
  chunks = min(nv_attn.PREFILL_CHUNKS * 32 // max(M, 32), N // BLOCK_N)
  srcs = [Tensor.empty(B, H, M, chunks, D, dtype=dtypes.float32).uop, Tensor.empty(B, H, M, chunks, 2, dtype=dtypes.float32).uop,
          Tensor.empty(B, H, M, D, dtype=dtypes.float16).uop, kv.uop] + ([scale.uop] if scale is not None else [])
  params = [UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs)]
  if dtype == dtypes.uint8: params[3] = UOp.placeholder((2, B, KVH, N, D // 8), dtypes.uint32, slot=3)
  return nv_attn._prefill_gqa_kernel(*params, valid_kv_len=valid, q_start=q_start, max_kv_len=N, warps=warps, keys=keys)

def _sm86_source(sink:UOp, renderer=None):
  from tinygrad.codegen import to_program
  from tinygrad.renderer.cstyle import CUDARenderer
  from tinygrad.helpers import DEV
  prg = to_program(sink, renderer or CUDARenderer(DEV.target("NV", arch="sm_86")))
  return prg, next(u.arg for u in prg.src if u.op is Ops.SOURCE)

class TestPrefillGQAKernel(unittest.TestCase):
  """T4.130: _prefill_gqa_kernel = the T4.113 kernel's contract with each K/V tile dequantised once into shared memory for `warps`
  query m-tiles of the GQA group. The CUDA renderer ignores the wmma device string, so the kernel's 'CUDA' WMMA_ARG lets the PYTHON
  emulator run it unchanged (its m16n8k16 model is the PTX ISA's fragment map) with only the warp shuffle swapped for a shared exchange."""
  def test_graph_builds_for_every_cache_dtype(self):
    sp = UOp.variable("start_pos", 0, 4095 - 128)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      for valid, q_start in ((128, 0), (4095, 3967), (sp + 128, sp)):
        with self.subTest(dtype=dtype, valid=str(valid)):
          sink = _gqa_sink(dtype, 4096, valid, q_start)
          self.assertIs(sink.op, Ops.SINK)
          nodes = list(sink.toposort())
          self.assertTrue(any(u.op is Ops.WMMA for u in nodes))
          self.assertEqual(sum(u.op is Ops.BUFFER and u.arg.addrspace is AddrSpace.LOCAL for u in nodes), 2)  # the K and V tiles
          self.assertTrue(any(u.op is Ops.CUSTOM and "__shfl_xor_sync(0xffffffff, {0}, 1)" in u.arg[0] for u in nodes))

  def test_render_pins_the_lane_split_and_stages_the_tiles(self):
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    sp = UOp.variable("start_pos", 0, 131072 - 128)
    for dtype in (dtypes.uint8, dtypes.int8, dtypes.float16):
      with self.subTest(dtype=dtype):
        prg, src = _sm86_source(_gqa_sink(dtype, 131072, sp + 128, sp), renderer)
        self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", src)
        self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)  # the WARP range is the hardware lane, unsplit
        self.assertIn("int lidx1 = threadIdx.y; /* 4 */", src)   # the warps
        self.assertEqual(src.count("__shared__ __align__(16) half buf"), 2)        # K and V tiles ...
        self.assertEqual(len(re.findall(r"half buf\d+\[8192\];", src)), 2)        # ... of keys x head_dim = 32 x 256 fp16 each
        self.assertGreaterEqual(src.count("__syncthreads();"), 2)                 # stores -> reads of a tile, end of the tile loop
        self.assertEqual(src.count("__WMMA_8_16_16_half_float("), 1 + 4 * 16 + 2 * 32)  # the helper + S (4 n-tiles x 16 k-steps) + P.V (2 x 32)
        self.assertIn("start_pos", src)
        if dtype == dtypes.uint8: self.assertIn("uint* data3_", src)              # the int4 cache as uint32 words
        self.assertLess(len(pickle.dumps(prg)), 2_000_000)                        # the compile-pool pickling limit (T4.113)
        self.assertLess(len(src), 150_000)

  def test_compiles_on_the_nvrtc_lane(self):
    # the real compiler (on macOS tinygrad's docker nvrtc lane, which the server also uses): skipped when it cannot start
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
    sp = UOp.variable("start_pos", 0, 131072 - 128)
    try:
      _, src = _sm86_source(_gqa_sink(dtypes.uint8, 131072, sp + 128, sp))
      ptx = NVRTCCompiler("sm_86").compile(src)
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    self.assertIn(b"flash_prefill_gqa_nv", ptx)

  def _run_emulated(self, dtype, Hq:int, KVH:int, Dh:int, N:int, T:int, q_start:int, warps:int, keys:int, slots:int|None=None, seed:int=0):
    """nv_prefill_attention on the PYTHON device vs the numpy causal softmax; returns (got, ref, partial, stats, dequantised kv, q)"""
    rng = np.random.default_rng(seed)
    kv, scale, ref_kv = _random_cache(rng, dtype, KVH, N, Dh, "PYTHON")
    q_np = (rng.standard_normal((1, Hq, T, Dh)) * 2).astype(np.float16)
    q = Tensor(q_np, device="PYTHON").contiguous().realize()
    captured, merge = {}, nv_attn._merge_partials
    def capture(partial, stats, valid):
      captured["partial"], captured["stats"] = partial.numpy()[0], stats.numpy()[0]
      return merge(partial, stats, valid)
    patches = [mock.patch.object(nv_attn, "_shfl_xor", _emulated_shfl_xor),
               mock.patch.object(nv_attn, "_prefill_gqa_kernel", nv_attn._prefill_gqa_kernel.__wrapped__),  # bypass the cache: patched shuffle
               mock.patch.object(nv_attn, "_merge_partials", capture)]
    if slots is not None: patches.append(mock.patch.object(nv_attn, "PREFILL_CHUNKS", slots))  # fewer slots than live chunks: passes per slot
    with contextlib.ExitStack() as stack:
      for p in patches: stack.enter_context(p)
      with Context(NV_ATTN_WMMA=1, NV_ATTN_GQA=1, NV_ATTN_GQA_WARPS=warps, NV_ATTN_GQA_KEYS=keys):
        got = nv_attn.nv_prefill_attention(q, kv, scale, q_start + T, q_start).numpy()[0]
    ref = TestPrefillKernel._reference(self, q_np[0], ref_kv, q_start + T, q_start)
    self.assertFalse(np.isnan(got).any())
    return got, ref, captured["partial"], captured["stats"], ref_kv, q_np[0]

  def test_kernel_matches_numpy_on_python_emulator(self):
    # the whole path (kernel on the emulator, merge, slice) against a plain numpy causal softmax: the causal limit per row (q_start),
    # the load mask (valid_kv_len), odd lengths, several slots and several passes over a slot (PREFILL_CHUNKS patched below the live
    # chunk count), every cache dtype (the int4 cache through its uint32 view), 16- and 32-key tiles, 1/2/4 warps per block
    for dtype, Hq, KVH, Dh, N, T, q_start, warps, keys, slots in (
        (dtypes.float16, 2, 1, 32, 256, 16, 128, 1, 32, None),   # 2 slots of one chunk; row 0 sees exactly one key of slot 1
        (dtypes.float16, 2, 1, 32, 256, 16, 128, 1, 32, 1),      # the same keys through ONE slot in two passes
        (dtypes.float16, 2, 1, 32, 256, 16, 37, 2, 16, None),    # odd lengths: 53 valid keys, 16-key tiles, 2 warps = both heads in a block
        (dtypes.uint8, 4, 1, 64, 512, 32, 300, 2, 32, 2),        # G=4, 8 m-tiles / 2 warps = 4 m-groups; 3 live chunks over 2 slots
        (dtypes.int8, 2, 1, 64, 256, 16, 120, 1, 16, None),      # int8 cache (generic dequant), slot 1 invisible to rows 0..7
        (dtypes.uint8, 6, 2, 64, 512, 32, 300, 4, 32, None)):    # the model's G=6 with 4 warps (12 m-tiles = 3 m-groups), 2 kv heads
      with self.subTest(dtype=dtype, Hq=Hq, KVH=KVH, Dh=Dh, N=N, T=T, q_start=q_start, warps=warps, keys=keys, slots=slots):
        got, ref, _, _, kv, _ = self._run_emulated(dtype, Hq, KVH, Dh, N, T, q_start, warps, keys, slots)
        # fp16 operands (K, V, P) against an fp64 softmax: the absolute error scales with |V| (int8 x scale reaches ~6)
        np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * max(1.0, float(np.abs(kv[1]).max())))

  def test_partials_and_stats_layout(self):
    # the per-slot (acc, m, l) the kernel writes, in _merge_partials' convention (m in the NATURAL domain, as the decode and T4.107b
    # kernels store it: the online softmax runs in the exp2 domain and converts on the way out); a slot with no visible key for a row
    # (q_start 120: rows 0..7 end before slot 1's first key) carries (SENTINEL-scaled m, l=0, acc=0) and weighs nothing in the merge
    dtype, Hq, KVH, Dh, N, T, q_start = dtypes.float16, 2, 1, 32, 256, 16, 120
    got, ref, partial, stats, kv, q = self._run_emulated(dtype, Hq, KVH, Dh, N, T, q_start, 1, 32)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)
    self.assertEqual((partial.shape, stats.shape), ((Hq, T, 2, Dh), (Hq, T, 2, 2)))  # (H, T_pad, slots, D) / (H, T_pad, slots, 2)
    for h in range(Hq):
      for r in range(T):
        for s in range(2):
          keys = np.arange(s * BLOCK_N, min((s + 1) * BLOCK_N, q_start + r + 1))
          if not len(keys):
            self.assertLess(stats[h, r, s, 0], -1e29)
            self.assertEqual(stats[h, r, s, 1], 0)
            np.testing.assert_array_equal(partial[h, r, s], 0)
            continue
          sc = kv[0, h // 2][keys].astype(np.float64) @ q[h, r].astype(np.float64) / math.sqrt(Dh)
          e = np.exp(sc - sc.max())
          np.testing.assert_allclose(stats[h, r, s, 0], sc.max(), rtol=1e-5, atol=1e-5)
          np.testing.assert_allclose(stats[h, r, s, 1], e.sum(), rtol=1e-4, atol=1e-5)
          np.testing.assert_allclose(partial[h, r, s], e @ kv[1, h // 2][keys].astype(np.float64), rtol=2e-3, atol=2e-3)

  def test_routing_knob_and_geometry(self):
    # NV_ATTN_GQA=1 (default) under NV_ATTN_WMMA=1 routes to the shared-tile kernel with the int4 cache bound as uint32 words (no copy:
    # the CALL's source is the stored uint8 buffer itself); NV_ATTN_GQA=0 is the T4.113 kernel, the knob's A/B
    Hq, KVH, N, Dh = 4, 2, 256, 64
    kv = Tensor.zeros(2, 1, KVH, N, Dh // 2, dtype=dtypes.uint8).contiguous().realize()
    scale = Tensor.ones(2, 1, KVH, N, Dh // 32, dtype=dtypes.float16).contiguous().realize()
    q = Tensor.ones(1, Hq, 6, Dh, dtype=dtypes.float16)
    def route(gqa:int):
      with Context(NV_ATTN_WMMA=1, NV_ATTN_GQA=gqa): out = nv_attn.nv_prefill_attention(q, kv, scale, 106, 100)
      nodes = list(out.uop.toposort())
      calls = [u for u in nodes if u.op is Ops.CALL]
      names = [u.arg.name for u in nodes if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
      return names, calls
    names, calls = route(1)
    self.assertEqual(names, ["flash_prefill_gqa_nv"])
    self.assertEqual(len(calls), 1)
    params = {u.arg.slot: u for u in calls[0].src[0].toposort() if u.op is Ops.PARAM}
    self.assertEqual((params[3].dtype, params[3].shape), (dtypes.uint32, (2 * KVH * N * Dh // 8,)))  # D/8 words per key
    self.assertIs(calls[0].src[4].base.buf_uop, kv.uop.base.buf_uop)  # srcs: out, stats, q, cache, scale -- the cache buffer as stored
    self.assertEqual(route(0)[0], ["flash_prefill_wmma_nv"])
    # the geometry from the knobs: warps = the largest divisor of the m-tile count (G * T_pad / 16) <= NV_ATTN_GQA_WARPS, keys the
    # largest power of two <= NV_ATTN_GQA_KEYS whose K + V fp16 tiles fit the 48 KB static shared-memory cap
    self.assertEqual(nv_attn._gqa_geometry(24, 4, 128, 256), {"warps": 8, "keys": 32})  # the model: 48 m-tiles per kv head; default 8 (10-29)
    with Context(NV_ATTN_GQA_WARPS=16, NV_ATTN_GQA_KEYS=128): self.assertEqual(nv_attn._gqa_geometry(24, 4, 128, 256), {"warps": 16, "keys": 32})
    with Context(NV_ATTN_GQA_WARPS=5): self.assertEqual(nv_attn._gqa_geometry(24, 4, 128, 256)["warps"], 4)
    with Context(NV_ATTN_GQA_KEYS=128): self.assertEqual(nv_attn._gqa_geometry(4, 2, 16, 64)["keys"], 128)  # 4*128*64 = 32 KB fits
    with Context(NV_ATTN_GQA=0): self.assertIsNone(nv_attn._gqa_geometry(24, 4, 128, 256))
    self.assertIsNone(nv_attn._gqa_geometry(4, 2, 16, 16))                                      # head_dim below one scale block

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
