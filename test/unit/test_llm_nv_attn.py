"""T4.107a: NV split-KV decode attention (kernels/nv_attn.py) -- the kernel graph builds for the real geometry and renders on sm_86
with its 32 lanes as one warp, the merge and the in-kernel dequant convention agree with a plain-tensor reference on CPU, and the
NV_CUSTOM_ATTN gate leaves every other path untouched. Kernel numerics on the card: extra/nv_attn_validate_real.py.
T4.129 (TestDecodeMMAKernel): the tensor-core KV_INT4 decode kernel EXECUTED on the PYTHON emulator against numpy (the warp
shuffle swapped for a shared exchange, as test_llm_nv_norm does), its sm_86 render, the store gates at small G, the dispatch.
T4.130: the GQA shared-tile prefill kernel is executed on the PYTHON emulator against numpy (TestPrefillGQAKernel).
T4.134 (TestVerifyMMAKernel): the decode mma kernel over speculative VERIFY runs of 2..4 tokens, on the emulator against numpy; its
T == 1 renders pinned by md5."""
import contextlib, itertools, math, pickle, re, sys, unittest
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

# ******** T4.129: KV_INT4 decode on the tensor cores ********

def _mma_sink(N:int, valid, warps:int=4, k_tiles:int=1, G:int=6, kvh:int=KVH, Dh:int=D, T:int=1, d_split:int=1):
  Hq, slots = kvh * G * T, min(nv_attn.NV_ATTN_DEC_SLOTS.value, N // (16 * k_tiles))  # T4.134: rows = q heads x tokens
  srcs = [Tensor.empty(B, Hq, slots, Dh, dtype=dtypes.float32).uop, Tensor.empty(B, Hq, slots, 2, dtype=dtypes.float32).uop,
          Tensor.empty(B, Hq, 1, Dh, dtype=dtypes.float16).uop, Tensor.empty(2, B, kvh, N, Dh // 2, dtype=dtypes.uint8).uop,
          Tensor.empty(2, B, kvh, N, Dh // 32, dtype=dtypes.float16).uop]
  params = [UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs)]
  params[3] = UOp.placeholder((2, B, kvh, N, Dh // 8), dtypes.uint32, slot=3)  # the words view _decode_mma_attention hands the kernel
  return nv_attn._decode_mma_kernel(*params, valid_kv_len=valid, max_kv_len=N, warps=warps, k_tiles=k_tiles, tokens=T, d_split=d_split)

def _render_sm86(test, sink) -> str:
  from tinygrad.codegen import to_program
  from tinygrad.renderer.cstyle import CUDARenderer
  from tinygrad.helpers import DEV
  try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
  except Exception as e: test.skipTest(f"CUDA renderer unavailable: {e!r}")
  prg = to_program(sink, renderer)
  test.assertLess(len(pickle.dumps(prg)), 2_000_000)  # the compile worker pool pickles the lowered PROGRAM (test_llm_nv_gemm's limit)
  return next(u.arg for u in prg.src if u.op is Ops.SOURCE)

_emu_slot = itertools.count(200)
def _emulated_shfl(warps:int):
  """_shfl (__shfl_sync from lane src_lane) as a block-shared exchange for the PYTHON emulator: thread warp*32+lane writes val and
  reads its partner's (the emulator runs the block in lockstep) -- test_llm_nv_norm's _emulated_warp_reduce, with the partner index
  clamped so CHECK_OOB's cheap vmin/vmax bound passes (its z3 pass has no model for lane ^ 1)."""
  def shfl(val:UOp, src_lane:UOp) -> UOp:
    lane = UOp.range(nv_attn.WARP_SIZE, -1, AxisType.WARP)
    warp = UOp.range(warps, 1, AxisType.LOCAL) if warps > 1 else UOp.const(0, dtypes.weakint)
    buf = UOp.placeholder((warps * 32,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
    partner = (warp * 32 + src_lane).maximum(0).minimum(warps * 32 - 1)
    return buf.after(buf[warp * 32 + lane].store(val))[partner].load()
  return shfl

def _dequant_int4_np(kv:np.ndarray, scale:np.ndarray, Dh:int) -> np.ndarray:
  """(2,KVH,N,D/2) packed nibbles + (2,KVH,N,D/32) fp16 scales -> (2,KVH,N,D) float64, model.py's layout (low nibble = even element)"""
  q = np.stack([kv & 0xF, kv >> 4], -1).reshape(*kv.shape[:-1], Dh).astype(np.float64) - 8
  return (q.reshape(*q.shape[:-1], Dh // 32, 32) * scale.astype(np.float64)[..., None]).reshape(q.shape)

def _q8_round_np(q:np.ndarray) -> np.ndarray:
  """the kernel's prologue: q rounded to int8 per 32-wide block (absmax/127, round half to even, clip) -- the one approximation the
  tensor-core path adds over the T4.107a kernel (P and V*scale go to fp16 for the P.V mma, like the T4.113 prefill kernel)"""
  blk = q.reshape(*q.shape[:-1], q.shape[-1] // 32, 32).astype(np.float32)
  sq = np.maximum(np.abs(blk).max(-1, keepdims=True) / 127, 1e-8)
  return (np.clip(np.rint(blk / sq), -127, 127) * sq).reshape(q.shape)

class TestDecodeMMAKernel(unittest.TestCase):
  """T4.129: the tensor-core decode kernel (one warp per partial slot, KS = 16*k_tiles keys per step, int8 mma for Q.K^T, fp16 mma
  for P.V, the T4.107a (acc, m, l) partial contract)."""
  def test_graph_builds(self):
    for warps, k_tiles in ((1, 1), (4, 1), (4, 2), (2, 4)):
      for valid in (1, 4095, UOp.variable("start_pos", 0, 4095) + 1):
        with self.subTest(warps=warps, k_tiles=k_tiles, valid=str(valid)):
          sink = _mma_sink(4096, valid, warps=warps, k_tiles=k_tiles)
          self.assertIs(sink.op, Ops.SINK)
          nodes = list(sink.toposort())
          wmma = [u for u in nodes if u.op is Ops.WMMA]
          self.assertEqual({u.arg[0] for u in wmma}, {(8, 16, 32), (8, 16, 16)})          # S on int8 m16n8k32, P.V on fp16 m16n8k16
          self.assertEqual(len(wmma), (2 * (D // 32) + D // 16) * k_tiles)               # per step: 2 n-tiles x 8 blocks, 16 m-tiles
          self.assertTrue(any(u.op is Ops.CUSTOM and "__shfl_sync(0xffffffff" in u.arg[0] for u in nodes))

  def test_render_for_sm86(self):
    # the mma fragment maps key on the HARDWARE lane: the WARP range must render as threadIdx.x /* 32 */ unsplit (nv_gemm's IMMA
    # kernel's pin), the warps on threadIdx.y; both mma shapes and the lane shuffles are in the source; the bound decode position
    # reaches the kernel as a parameter
    for warps, k_tiles in ((4, 1), (4, 2), (1, 1)):
      with self.subTest(warps=warps, k_tiles=k_tiles):
        src = _render_sm86(self, _mma_sink(4096, UOp.variable("start_pos", 0, 4095) + 1, warps=warps, k_tiles=k_tiles))
        self.assertIn("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", src)
        self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", src)
        self.assertIn("__shfl_sync(0xffffffff", src)
        self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)
        if warps > 1: self.assertIn(f"int lidx1 = threadIdx.y; /* {warps} */", src)
        else: self.assertNotIn("lidx1", src)
        self.assertIn(f"__launch_bounds__({32 * warps})", src)
        self.assertIn("start_pos", src)
        self.assertEqual(src.count("__WMMA_8_16_32_signed_char_int("), 1 + 16 * k_tiles)  # the helper's definition + its call sites
        self.assertEqual(src.count("__WMMA_8_16_16_half_float("), 1 + 16 * k_tiles)
        self.assertLess(len(src), 150_000)

  def test_out_store_gates_survive_small_g(self):
    # the index lowering substitutes what a gate pins into the gated component and drops a gate whose ranges then no longer appear
    # in it (schedule/indexing.py's reshape rewrite, pm_drop_and_clauses): gated on the head (oh < 2 pins t = 0) the out stores
    # rendered UNCONDITIONAL at G = 2 and 3 and every lane wrote head 0/1's rows (found on the emulator, 10-29). The gate now sits
    # on the dim component (depends on g): every out store of the small-G renders must sit under an if
    for G in (2, 3, 6):
      with self.subTest(G=G):
        lines = [l.strip() for l in _render_sm86(self, _mma_sink(128, 16, warps=1, G=G, kvh=1, Dh=64)).splitlines()]
        stores = [i for i, l in enumerate(lines) if "data0_" in l and " = " in l]
        self.assertTrue(stores)
        for i in stores: self.assertTrue(lines[i - 1].startswith("if ("), f"G={G}: unguarded out store: {lines[i][:80]}")

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_kernel_matches_numpy_on_emulator(self):
    # the whole path (q8 prologue, int8 S mma, online softmax across steps and slots, fp16 P.V mma, stats, _merge_partials) executed
    # lane by lane on the PYTHON emulator, against numpy attention over the dequantized cache with q rounded as the kernel rounds it:
    # lengths that fill no whole 16-key tile (1, 17, 37, 50, 100), one that fills every tile (128), slots that fold several tiles
    # (steps > 1, incl. a step with no visible key for a slot: alpha = 1, p = 0), 1/2 warps per block, 1/2 key tiles per step,
    # D = 64 / 128, G = 3 / 6 / 8 (the padded m-tile rows), and a bound symbolic length
    from tinygrad.llm.kernels import nv_gemm
    rng = np.random.default_rng(4)
    configs = [(2, 3, 64, 128, 37, 2, 1, 1), (2, 3, 64, 128, 128, 2, 1, 1), (2, 3, 64, 128, 1, 2, 1, 1), (2, 3, 64, 128, 17, 2, 1, 1),
               (1, 6, 64, 128, 100, 4, 2, 1), (1, 6, 64, 128, 100, 4, 2, 2), (1, 8, 128, 128, 50, 2, 1, 1)]
    for kvh, G, Dh, N, valid, slots, warps, k_tiles in configs:
      for symbolic in ((False, True) if valid == 37 else (False,)):
        with self.subTest(kvh=kvh, G=G, D=Dh, N=N, valid=valid, slots=slots, warps=warps, k_tiles=k_tiles, symbolic=symbolic):
          Hq = kvh * G
          raw = rng.integers(0, 256, size=(2, 1, kvh, N, Dh // 2), dtype=np.uint8)
          scale = rng.uniform(0.01, 0.2, size=(2, 1, kvh, N, Dh // 32)).astype(np.float16)
          q_np = (rng.standard_normal((1, Hq, 1, Dh)) * 2).astype(np.float16)
          kv, sc, q = (Tensor(a, device="PYTHON").contiguous().realize() for a in (raw, scale, q_np))
          vl = UOp.variable("start_pos", 0, N - 1).bind(valid - 1) + 1 if symbolic else valid
          with patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True), patch.object(nv_attn, "_shfl", _emulated_shfl(warps)), \
               patch.object(nv_attn, "_decode_mma_kernel", nv_attn._decode_mma_kernel.__wrapped__), \
               Context(NV_ATTN_DEC_MMA=1, NV_ATTN_DEC_SLOTS=slots, NV_ATTN_DEC_WARPS=warps, NV_ATTN_DEC_KTILES=k_tiles):
            out = nv_attn.nv_decode_attention(q, kv, sc, vl)
            self.assertIn("flash_decode_mma_nv", [u.arg.name for u in out.uop.toposort() if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)])
            got = out.numpy()[0, :, 0]
          kvd, qr = _dequant_int4_np(raw[:, 0], scale[:, 0], Dh), _q8_round_np(q_np[0, :, 0].astype(np.float32))
          ref = np.zeros((Hq, Dh))
          for h in range(Hq):
            s = kvd[0, h // G, :valid] @ qr[h].astype(np.float64) / math.sqrt(Dh)
            p = np.exp(s - s.max())
            ref[h] = p @ kvd[1, h // G, :valid] / p.sum()
          self.assertFalse(np.isnan(got).any())
          np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

  def test_merge_counts_16_key_slots(self):
    # _decode_mma_attention hands _merge_partials valid * BLOCK_N/KS, so the merge's ceil(. / BLOCK_N) live count is ceil(valid / KS)
    partial = Tensor([[[[1.], [2.], [3.], [4.]]]])
    stats = Tensor([[[[0., 1.], [0., 1.], [100., 1.], [100., 1.]]]])
    for valid, want in ((1, 1.0), (16, 1.0), (17, 1.5), (32, 1.5), (33, 3.0)):
      self.assertAlmostEqual(nv_attn._merge_partials(partial, stats, valid * (BLOCK_N // 16)).item(), want, places=5)

  def test_dispatch(self):
    # graph-level, on CPU with the sm_80 check mocked: a KV_INT4 cache takes the mma kernel and sizes the partial buffer at
    # min(NV_ATTN_DEC_SLOTS, N/16) slots; NV_ATTN_DEC_MMA=0, an int8/fp16 cache, G = 1, a head_dim the fragments don't cover and a
    # device below sm_80 (or CPU, unmocked) keep the T4.107a kernel
    from tinygrad.llm.kernels import nv_gemm
    def run(kv_dtype=dtypes.uint8, G:int=6, Dh:int=D, N:int=4096, mma:int=1, wmma_ok:bool|None=True):
      kvh = 2
      kv = Tensor.zeros(2, 1, kvh, N, Dh // 2 if kv_dtype == dtypes.uint8 else Dh, dtype=kv_dtype)
      scale = Tensor.ones(2, 1, kvh, N, Dh // 32, dtype=dtypes.float16) if kv_dtype != dtypes.float16 else None
      q = Tensor.ones(1, kvh * G, 1, Dh, dtype=dtypes.float16)
      mocks = [patch.object(nv_gemm, "_nv_wmma_ok", lambda device: wmma_ok)] if wmma_ok is not None else []
      with Context(NV_ATTN_DEC_MMA=mma), patch.object(nv_attn.Tensor, "empty", wraps=nv_attn.Tensor.empty) as m:
        for mk in mocks: mk.start()
        try: out = nv_attn.nv_decode_attention(q, kv, scale, 100)
        finally:
          for mk in mocks: mk.stop()
      names = [u.arg.name for u in out.uop.toposort() if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
      return names, m.call_args_list[0].args[:4]
    self.assertEqual(run(), (["flash_decode_mma_nv"], (1, 12, 256, D)))
    self.assertEqual(run(N=2048), (["flash_decode_mma_nv"], (1, 12, 128, D)))
    with Context(NV_ATTN_DEC_SLOTS=64): self.assertEqual(run()[1], (1, 12, 64, D))
    self.assertEqual(run(mma=0), (["flash_decode_partial_nv"], (1, 12, 32, D)))
    self.assertEqual(run(kv_dtype=dtypes.int8)[0], ["flash_decode_partial_nv"])
    self.assertEqual(run(kv_dtype=dtypes.float16)[0], ["flash_decode_partial_nv"])
    self.assertEqual(run(G=1)[0], ["flash_decode_partial_nv"])
    self.assertEqual(run(Dh=32)[0], ["flash_decode_partial_nv"])
    self.assertEqual(run(wmma_ok=False)[0], ["flash_decode_partial_nv"])
    self.assertEqual(run(wmma_ok=None)[0], ["flash_decode_partial_nv"])  # CPU: no CUDA renderer

# ******** T4.134: speculative VERIFY runs on the decode mma kernel ********

def _lower_sm86(sink:UOp) -> tuple[UOp, str]:
  """the lowered PROGRAM and its sm_86 source WITHOUT compiling (no nvrtc lane: on macOS the renderer's compiler starts the docker
  nvrtc server in __init__) -- test_llm_nv_scan's _render_cuda, keeping the PROGRAM for the compile pool's pickle check"""
  from tinygrad.codegen import full_rewrite_to_sink, pm_to_program, ProgramInfo
  from tinygrad.uop.ops import PatternMatcher, graph_rewrite
  from tinygrad.renderer.cstyle import CUDARenderer
  from tinygrad.helpers import DEV
  with mock.patch("tinygrad.runtime.support.compiler_cuda._get_server", return_value=None): renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
  full = full_rewrite_to_sink(sink, renderer)
  prg = UOp(Ops.PROGRAM, src=(full,), arg=ProgramInfo.from_sink(full, renderer.target))
  prg = graph_rewrite(prg, PatternMatcher([pat for pat in pm_to_program.patterns if pat[1].__name__ != "do_compile"]), ctx=renderer)
  return prg, next(u.arg for u in prg.src if u.op is Ops.SOURCE)

class TestVerifyMMAKernel(unittest.TestCase):
  """T4.134: _decode_mma_kernel with tokens = T > 1 -- q/out/stats viewed as (B, H*T, ...), the G*T rows of a kv head packed into
  8-row groups (m-tile halves of S, n-tiles of P.V), row r = token r % T seeing keys < valid - (T-1) + r % T."""
  # md5 of the T == 1 sm_86 renders at 014b49554 (the T4.129 kernel the card measured), rendered by _lower_sm86
  T1_MD5 = {"model": "1a2f80031271111f1346426ec24b529f", "k_tiles2": "521528f80f5299985d1dd412f5bf26ff",
            "G2": "1292aec96d439776fd58a65c1ed724e2", "G3": "b591e5bbce4d66eae3be87965870c941"}

  def test_t1_render_unchanged(self):
    import hashlib
    sinks = {"model": _mma_sink(262144, UOp.variable("start_pos", 0, 262143) + 1),
             "k_tiles2": _mma_sink(4096, UOp.variable("start_pos", 0, 4095) + 1, k_tiles=2),
             "G2": _mma_sink(128, 16, warps=1, G=2, kvh=1, Dh=64), "G3": _mma_sink(128, 16, warps=1, G=3, kvh=1, Dh=64)}
    for name, sink in sinks.items():
      with self.subTest(sink=name):
        self.assertEqual(hashlib.md5(_lower_sm86(sink)[1].encode()).hexdigest(), self.T1_MD5[name], f"the T == 1 {name} render changed")

  def test_verify_renders_for_sm86(self):
    # the model's geometry (G = 6 -> 12/18/24 rows = 2/3/3 row groups in 1/2/2 m-tiles): S is one int8 mma per (n-tile, block, m-tile),
    # P.V one fp16 mma per (dim tile, row group), the lane unsplit on threadIdx.x, and the lowered PROGRAM pickles for the compile
    # pool (before the T4.134 Estimates fix its op-count expression alone overflowed pickle's C stack at T = 3); d_split 2 halves P.V
    for T, k_tiles, d_split in ((2, 1, 1), (3, 1, 1), (4, 1, 1), (3, 2, 1), (3, 1, 2)):
      with self.subTest(T=T, k_tiles=k_tiles, d_split=d_split):
        groups = -(-6 * T // 8)
        prg, src = _lower_sm86(_mma_sink(262144, UOp.variable("start_pos", 0, 262144 - T) + T, k_tiles=k_tiles, T=T, d_split=d_split))
        self.assertEqual(src.count("__WMMA_8_16_32_signed_char_int("), 1 + 2 * (D // 32) * -(-groups // 2) * k_tiles)
        self.assertEqual(src.count("__WMMA_8_16_16_half_float("), 1 + (D // 16 // d_split) * groups * k_tiles)
        self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)
        self.assertIn("threadIdx.y; /* 4 */", src)
        self.assertIn("start_pos", src)
        self.assertLess(len(pickle.dumps(prg)), 2_000_000)

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_verify_matches_numpy_on_emulator(self):
    # nv_verify_attention end to end (q8 prologue per row, both A halves, the per-row causal limit, the online softmax across steps and
    # slots, P.V per row group, stats, _merge_partials) on the PYTHON emulator against an fp64 numpy causal softmax with q rounded as the
    # kernel rounds it: G = 2, 3, 6, 8 (G*T = 4..32 rows: one to four row groups, a last group of one row at 9), fills ending mid-tile
    # and straddling tiles/slots (a slot holding no key some row sees), start 0 and 1 (fewer keys than slots), several steps per slot,
    # 1/2 warps, 1/2 key tiles, D = 64 / 128, two kv heads, a bound symbolic start_pos; NV_ATTN_DEC_DSPLIT 1 and 2 (head_dim halves in
    # adjacent blocks), its no-op fallback at D = 64, and T == 1 under the knob (nv_decode_attention: never split)
    from tinygrad.llm.kernels import nv_gemm
    configs = [(2, 2, 64, 128, 2, 35, 2, 1, 1, False, 1), (1, 2, 64, 128, 4, 13, 2, 2, 1, False, 1), (2, 3, 64, 128, 3, 14, 2, 1, 1, False, 1),
               (1, 3, 64, 128, 2, 0, 4, 1, 1, False, 1), (1, 6, 64, 128, 2, 31, 2, 2, 1, False, 1), (1, 6, 64, 128, 3, 98, 2, 2, 2, False, 1),
               (1, 6, 128, 128, 4, 1, 2, 1, 1, False, 1), (1, 6, 64, 128, 3, 34, 2, 1, 1, True, 1), (1, 8, 64, 128, 4, 57, 3, 2, 1, False, 1),
               (1, 6, 128, 128, 3, 98, 2, 2, 1, False, 2), (2, 3, 128, 128, 3, 14, 2, 1, 2, False, 2), (1, 6, 128, 128, 2, 97, 2, 2, 1, False, 2),
               (1, 6, 64, 128, 3, 40, 2, 1, 1, False, 2), (1, 6, 128, 128, 1, 99, 2, 2, 1, False, 2)]
    rng = np.random.default_rng(5)
    for kvh, G, Dh, N, T, start, slots, warps, k_tiles, symbolic, d_split in configs:
      with self.subTest(kvh=kvh, G=G, D=Dh, T=T, start=start, slots=slots, warps=warps, k_tiles=k_tiles, symbolic=symbolic, d_split=d_split):
        Hq, valid = kvh * G, start + T
        raw = rng.integers(0, 256, size=(2, 1, kvh, N, Dh // 2), dtype=np.uint8)
        scale = rng.uniform(0.01, 0.2, size=(2, 1, kvh, N, Dh // 32)).astype(np.float16)
        q_np = (rng.standard_normal((1, Hq, T, Dh)) * 2).astype(np.float16)
        kv, sc, q = (Tensor(a, device="PYTHON").contiguous().realize() for a in (raw, scale, q_np))
        vl = UOp.variable("start_pos", 0, N - T).bind(start) + T if symbolic else valid
        with patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True), patch.object(nv_attn, "_shfl", _emulated_shfl(warps)), \
             patch.object(nv_attn, "_decode_mma_kernel", nv_attn._decode_mma_kernel.__wrapped__), \
             Context(NV_ATTN_DEC_SLOTS=slots, NV_ATTN_DEC_WARPS=warps, NV_ATTN_DEC_KTILES=k_tiles, NV_ATTN_DEC_DSPLIT=d_split):
          got = (nv_attn.nv_verify_attention(q, kv, sc, vl) if T > 1 else nv_attn.nv_decode_attention(q, kv, sc, vl)).numpy()[0]
        kvd = _dequant_int4_np(raw[:, 0], scale[:, 0], Dh)
        ref = np.zeros((Hq, T, Dh))
        for h in range(Hq):
          for t in range(T):
            lim = valid - (T - 1) + t
            s = kvd[0, h // G, :lim] @ _q8_round_np(q_np[0, h, t].astype(np.float32)).astype(np.float64) / math.sqrt(Dh)
            p = np.exp(s - s.max())
            ref[h, t] = p @ kvd[1, h // G, :lim] / p.sum()
        self.assertFalse(np.isnan(got).any())
        np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

  def test_compiles_on_the_nvrtc_lane(self):
    # VERIFY(3) at the model's geometry and the default knobs through the real compiler (on macOS the docker nvrtc lane, as the server;
    # skipped when it cannot start): the sm_86 cubin's register count and spill stack (EIATTR_REGCOUNT / EIATTR_MIN_STACK_SIZE, what
    # ops_nv reads). Unsplit, VERIFY(3) spilled 1120 bytes per thread (more than L1 holds at two blocks per SM); d_split 2: 160
    import struct
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
    from tinygrad.runtime.support.elf import elf_loader
    _, src = _lower_sm86(_mma_sink(262144, UOp.variable("start_pos", 0, 262144 - 3) + 3, T=3, d_split=nv_attn.NV_ATTN_DEC_DSPLIT.value))
    try: lib = NVRTCCompiler("sm_86", ptx=False, cache_key="nv").compile(src)
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    attrs = {}
    for sh in elf_loader(lib, force_section_align=128)[1]:
      off = 0
      while sh.name == ".nv.info" and off < sh.header.sh_size:
        typ, param, sz = struct.unpack_from("BBH", sh.content, off)
        if typ == 0x4 and sz >= 8: attrs[param] = struct.unpack_from("II", sh.content, off + 4)[1]
        off += (sz if typ == 0x4 else 0) + 4
    self.assertLessEqual(attrs[0x2f], 255)   # EIATTR_REGCOUNT
    self.assertLessEqual(attrs[0x12], 256)   # EIATTR_MIN_STACK_SIZE: the spill stack, bytes per thread

  def test_dispatch(self):
    # graph-level, on CPU with the sm_80 check mocked: a concrete 2..4-token run on a KV_INT4 cache takes the mma kernel over H*T rows;
    # T = 1 / 5 / symbolic, NV_ATTN_DEC_MMA_T=0, an int8/fp16 cache and a device below sm_80 (or CPU, unmocked) return None
    from tinygrad.llm.kernels import nv_gemm
    def run(T=3, kv_dtype=dtypes.uint8, wmma_ok:bool|None=True):
      kvh, G = 2, 6
      kv = Tensor.zeros(2, 1, kvh, 4096, D // 2 if kv_dtype == dtypes.uint8 else D, dtype=kv_dtype)
      scale = Tensor.ones(2, 1, kvh, 4096, D // 32, dtype=dtypes.float16) if kv_dtype != dtypes.float16 else None
      q = Tensor.ones(1, kvh * G, T, D, dtype=dtypes.float16) if isinstance(T, int) else Tensor.ones(1, kvh * G, 4, D, dtype=dtypes.float16)[:, :, :T]
      with contextlib.ExitStack() as stack:
        if wmma_ok is not None: stack.enter_context(patch.object(nv_gemm, "_nv_wmma_ok", lambda device: wmma_ok))
        m = stack.enter_context(patch.object(nv_attn.Tensor, "empty", wraps=nv_attn.Tensor.empty))
        out = nv_attn.nv_verify_attention(q, kv, scale, 100 + (T if isinstance(T, int) else 1))
      if out is None: return None
      names = [u.arg.name for u in out.uop.toposort() if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
      return names, m.call_args_list[0].args[:4], out.shape
    for T in (2, 3, 4):
      self.assertEqual(run(T), (["flash_decode_mma_nv"], (1, 12 * T, 256, D), (1, 12, T, D)))
    self.assertIsNone(run(1))
    self.assertIsNone(run(5))
    self.assertIsNone(run(UOp.variable("T", 1, 4).bind(3)))
    with Context(NV_ATTN_DEC_MMA_T=0): self.assertIsNone(run())
    self.assertIsNone(run(kv_dtype=dtypes.int8))
    self.assertIsNone(run(kv_dtype=dtypes.float16))
    self.assertIsNone(run(wmma_ok=False))
    self.assertIsNone(run(wmma_ok=None))  # CPU: no CUDA renderer
    # NV_ATTN_DEC_DSPLIT (default 2) reaches the VERIFY kernel: P.V mmas per step = (D/16/d_split) x 3 row groups; it falls back to 1
    # where it doesn't divide head_dim/64 (whole V words per part), and T == 1 never splits (16 P.V mmas: the T4.129 kernel)
    def pv_mmas(Dh:int=D, T:int=3, **ctx) -> int:
      kv, scale = Tensor.zeros(2, 1, 2, 4096, Dh // 2, dtype=dtypes.uint8), Tensor.ones(2, 1, 2, 4096, Dh // 32, dtype=dtypes.float16)
      q = Tensor.ones(1, 12, T, Dh, dtype=dtypes.float16)
      with Context(**ctx), patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True):
        out = nv_attn.nv_verify_attention(q, kv, scale, 100 + T) if T > 1 else nv_attn.nv_decode_attention(q, kv, scale, 101)
      return sum(u.op is Ops.WMMA and u.arg[0] == (8, 16, 16) for u in out.uop.toposort())
    self.assertEqual(pv_mmas(), 24)
    self.assertEqual([pv_mmas(NV_ATTN_DEC_DSPLIT=d) for d in (1, 2, 4, 3)], [48, 24, 12, 48])
    self.assertEqual(pv_mmas(64), 12)
    self.assertEqual(pv_mmas(T=1), 16)

  def test_model_routes_verify(self):
    # TransformerBlock._nv_attn: T == 1 -> the decode kernel, a concrete 2..4-token run -> the decode kernel over its rows,
    # NV_ATTN_DEC_MMA_T=0 or a symbolic chunk -> the prefill kernel (gates mocked open on CPU, graph only)
    from types import SimpleNamespace
    from tinygrad.llm import model
    from tinygrad.llm.kernels import nv_gemm
    kv = Tensor.zeros(2, 1, 2, 4096, D // 2, dtype=dtypes.uint8)
    scale = Tensor.ones(2, 1, 2, 4096, D // 32, dtype=dtypes.float16)
    block = SimpleNamespace(config=SimpleNamespace(sliding_window=0, head_dim=D), cache_kv=kv)
    def kernels(T) -> list[str]:
      q = Tensor.ones(1, 12, T, D, dtype=dtypes.float16) if isinstance(T, int) else Tensor.ones(1, 12, 32, D, dtype=dtypes.float16)[:, :, :T]
      with patch.object(model, "nv_decode_attention_ok", lambda *a: True), patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True):
        out = model.TransformerBlock._nv_attn(block, q, kv, scale, 100, T)
      return [u.arg.name for u in out.uop.toposort() if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
    self.assertEqual(kernels(1), ["flash_decode_mma_nv"])
    self.assertEqual(kernels(3), ["flash_decode_mma_nv"])
    with Context(NV_ATTN_DEC_MMA_T=0): self.assertEqual(kernels(3), ["flash_prefill_partial_nv"])
    self.assertEqual(kernels(UOp.variable("T", 1, 32).bind(3)), ["flash_prefill_partial_nv"])

if __name__ == "__main__": unittest.main()
