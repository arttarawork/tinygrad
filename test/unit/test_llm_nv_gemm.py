"""T4.98f/T4.98j: NV WMMA (mma.sync m16n8k16) prefill gemm for Q8_0/Q4_0/Q4_1/IQ4_NL and the K-quants -- graph builds
for the real shapes, the sm_86 render pins the lane-range split the fragment mapping relies on, and the dispatch gates.
Numerics are hardware-only (t4x/wmma_probe.py, nv_gemm_validate.py). T4.126's int8 (IMMA, m16n8k32) Q4_0 kernel is different:
the PYTHON emulator models that mma, so its fragment map, staging and epilogue are executed there bit-exact vs numpy."""
import functools, sys, unittest
from math import prod
import numpy as np
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, AxisType, KernelInfo
from tinygrad.llm.kernels import nv_gemm
from tinygrad.llm.kernels.amd import Q8_0, Q4_0, Q4_1, IQ4_NL, Q4_K, Q5_K, Q6_K, IQ4_XS, Q3_K, IQ3_XXS, IQ3_S, QUANT_SIZES, IQ_GRID_SIZES
from tinygrad.llm.kernels.nv_quant import NV_CUSTOM_QUANT  # importing nv_quant registers the ContextVar the dispatch test sets

BLOCK = {Q8_0: (32, 34), Q4_0: (32, 18), Q4_1: (32, 20), IQ4_NL: (32, 18),
         **{t: (256, QUANT_SIZES[t]) for t in (Q4_K, Q5_K, Q6_K, IQ4_XS, Q3_K, IQ3_XXS, IQ3_S)}}

def _sink(ggml_type:int, tokens:int, out_features:int, in_features:int, token_tile:int|None=None):
  weights, block_bytes = BLOCK[ggml_type]
  dt = dtypes.uint8  # the kernel always gets a byte view (amd.py packed_bytes for the word-view formats)
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32).uop
  raw = Tensor.empty(out_features * in_features // weights * block_bytes // dt.itemsize, dtype=dt).uop
  x = Tensor.empty(tokens, in_features, dtype=dtypes.float16).uop
  srcs = [out, raw, x]
  if ggml_type in IQ_GRID_SIZES:  # T4.98k: the IQ3 codebook, bound the same way nv_wmma_linear binds it (see iq_grid)
    srcs.append(Tensor.empty(IQ_GRID_SIZES[ggml_type], dtype=dtypes.uint32).uop)
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(srcs))
  # token_tile is a required kwarg on _wmma_kernel (T4.124c); default here reproduces what every caller (incl. nv_wmma_linear
  # with the knob at its default 64) computes: _token_tile_for(tokens) with no override.
  tile = nv_gemm._token_tile_for(tokens) if token_tile is None else token_tile
  return nv_gemm._wmma_kernel(*params, out_features=out_features, in_features=in_features, ggml_type=ggml_type, token_tile=tile)

def _shapes(ggml_type:int):  # (tokens, out, in): 1-4 M tiles, 1/4 N tiles, several K blocks; K a multiple of the format's block
  k = nv_gemm.WMMA_FORMATS[ggml_type][1]
  return ((16, 16, k), (32, 32, 2 * k), (64, 48, 4 * k), (32, 6144, 5120))

class TestKernelGraphAndRender(unittest.TestCase):
  def test_graph_builds_for_every_format(self):
    for ggml_type in nv_gemm.WMMA_FORMATS:
      for tokens, o, i in _shapes(ggml_type):
        with self.subTest(ggml_type=ggml_type, tokens=tokens, out=o):
          sink = _sink(ggml_type, tokens, o, i)
          self.assertIs(sink.op, Ops.SINK)
          self.assertTrue(any(u.op is Ops.WMMA for u in sink.toposort()))

  def test_render_pins_the_lane_split(self):
    # the fragment identity in _wmma_layout_nv assumes the 32-lane LOCAL range renders as lidx0=threadIdx.x (4) = lane//8 and
    # lidx1=threadIdx.y (8) = lane%8, i.e. hardware lane = 4*(lane%8) + lane//8; if the lowerer ever changes that split this
    # must fail before a wrong kernel reaches the card (see _wmma_layout_nv's comment and the 2026-09-11 probe)
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for ggml_type in nv_gemm.WMMA_FORMATS:
      for tokens, o, i in _shapes(ggml_type)[:3]:
        with self.subTest(ggml_type=ggml_type, tokens=tokens, out=o):
          prg = to_program(_sink(ggml_type, tokens, o, i), renderer)
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", src)
          self.assertIn("int lidx0 = threadIdx.x; /* 4 */", src)
          self.assertIn("int lidx1 = threadIdx.y; /* 8 */", src)
          self.assertEqual(src.count("threadIdx.z"), 0)
          if ggml_type in IQ_GRID_SIZES:  # T4.98k: the codebook (slot 3) must actually be read, not optimized away
            self.assertIn("data3_", src)

  def test_token_tile_knob_changes_global_grid_and_mma_count(self):
    # T4.124c (PB.3): NV_WMMA_TOKEN_TILE lets a narrower token tile dispatch more, smaller token blocks. Shape matches the
    # gemm model in prefill-batch-design.md section 3 (Q4_0 gate/up, out=17408 x in=5120 -> grid.x = 17408/32 = 544).
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    tokens, out_features, in_features = 64, 17408, 5120
    def mma_count(sink): return sum(1 for u in sink.toposort() if u.op is Ops.WMMA)

    default_sink = _sink(Q4_0, tokens, out_features, in_features)  # token_tile omitted -> _token_tile_for(64) == 64
    self.assertEqual(mma_count(default_sink), 32)
    self.assertEqual(to_program(default_sink, renderer).arg.global_size[:2], (544, 1))

    tile32_sink = _sink(Q4_0, tokens, out_features, in_features, token_tile=32)
    self.assertEqual(mma_count(tile32_sink), 16)
    self.assertEqual(to_program(tile32_sink, renderer).arg.global_size[:2], (544, 2))

  def test_default_tile_matches_pre_knob_fixed_rule(self):
    # the knob's default (64) must reproduce the original hardcoded rule exactly, for every shape the other tests cover --
    # same tile value => the deterministic builder renders byte-identical output (tile is the only shape-shaping input).
    for ggml_type in nv_gemm.WMMA_FORMATS:
      for tokens, _o, _i in _shapes(ggml_type):
        fixed_rule = 64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else nv_gemm.WMMA_M
        self.assertEqual(nv_gemm._token_tile_for(tokens), fixed_rule, (ggml_type, tokens))

def _imma_probe(out:UOp, a:UOp, b:UOp) -> UOp:
  # T4.126: ONE m16n8k32 int8 mma, one warp, the fragments loaded straight from A (16x32 int8, row-major) and B (8x32 int8 =
  # N rows of K, the weight layout) through nv_gemm's documented fragment map -- so running it on the PYTHON emulator (which
  # models the PTX ISA's m16n8k32 layout) against numpy's A @ B.T checks the map, and rendering it checks the .s8 renderer
  lane = UOp.range(32, -1, AxisType.WARP)
  g, t = lane // 4, lane % 4
  afrag = UOp.stack(*(a[g + 8*(r % 2), 16*(r // 2) + 4*t + i] for r in range(4) for i in range(4)))
  bfrag = UOp.stack(*(b[g, 16*r + 4*t + i] for r in range(2) for i in range(4)))
  acc = UOp.placeholder((4,), dtypes.int32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(UOp.wmma(afrag, bfrag, UOp.const((0,)*4, dtypes.int32), *nv_gemm.IMMA_ARG)))
  stores = [out[g + 8*(i // 2), 2*t + i % 2].store(acc[i].load()) for i in range(4)]
  return UOp.group(*stores).end(lane).sink(arg=KernelInfo(name="imma_probe", opts_to_apply=()))

class TestImmaRenderer(unittest.TestCase):
  def test_int8_mma_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    params = [UOp.placeholder((16, 8), dtypes.int32, slot=0), UOp.placeholder((16, 32), dtypes.int8, slot=1),
              UOp.placeholder((8, 32), dtypes.int8, slot=2)]
    src = next(u.arg for u in to_program(_imma_probe(*params), renderer).src if u.op is Ops.SOURCE)
    self.assertIn("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", src)
    self.assertIn('"{%0, %1, %2, %3}, {%4, %5, %6, %7},"', src)  # C = 4 regs, A = 16 int8 = 4 regs
    self.assertIn('"{%8, %9}, {%0, %1, %2, %3};"', src)          # B = 8 int8 = 2 regs
    self.assertIn("struct __align__(16) signed_char16", src)

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_fragment_map_on_python_emulator(self):
    rng = np.random.default_rng(0)
    a, b = rng.integers(-128, 128, (16, 32), dtype=np.int8), rng.integers(-128, 128, (8, 32), dtype=np.int8)
    out = Tensor.empty(16, 8, dtype=dtypes.int32, device="PYTHON")
    res = out.custom_kernel(Tensor(a, device="PYTHON"), Tensor(b, device="PYTHON"), fxn=_imma_probe)[0].numpy()
    np.testing.assert_array_equal(res, a.astype(np.int32) @ b.astype(np.int32).T)

# ******** T4.126: the IMMA Q4_0 kernel -- numpy reference of the exact quantized math, executed on the PYTHON emulator ********

def _q4_0_blocks(rng, out_features:int, groups:int) -> np.ndarray:
  # random Q4_0 blocks (out, groups, 18 bytes): f16 d then 16 qs bytes (any byte is a valid pair of nibbles)
  blocks = rng.integers(0, 256, (out_features, groups, 18), dtype=np.uint8)
  blocks[..., :2] = rng.uniform(-0.05, 0.05, (out_features, groups)).astype(np.float16)[..., None].view(np.uint8)
  return blocks

def _q8(rng, tokens:int, in_features:int) -> tuple[np.ndarray, np.ndarray]:
  # q8_quantize_nv's exact formula (scale = max|x|/127 clamped at 1e-8, q = clip(rint(x/scale))), packed 4 int8 per uint32
  x = rng.standard_normal((tokens, in_features // 32, 32)).astype(np.float32)
  xd = np.maximum(np.abs(x).max(-1) / np.float32(127), np.float32(1e-8)).astype(np.float32)
  q = np.clip(np.rint(x / xd[..., None]), -127, 127).astype(np.int8)
  return q.view(np.uint32), xd

def _np_imma_reference(blocks:np.ndarray, xq:np.ndarray, xd:np.ndarray) -> np.ndarray:
  # the kernel's math: int32 dot per (token, output, group) of q8 x (nibble-8), then acc = fma(float(c) * d_w, xd, acc) in fp32,
  # group by group in order (the kernel's REDUCE loop). Fused like the card's FFMA (nvcc contracts a*b+c by default) and the
  # PYTHON renderer's MULACC: the fp32 product of two fp32 values is exact in fp64, so one rounding after the add
  q = xq.view(np.int8).astype(np.int64)                                                    # (tokens, groups, 32)
  qs = blocks[..., 2:]
  w = np.concatenate([qs & 0xF, qs >> 4], axis=-1).astype(np.int64) - 8                    # (out, groups, 32)
  c = np.einsum("tgk,ogk->tog", q, w).astype(np.float32)
  dw = blocks[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)                 # (out, groups)
  acc = np.zeros(c.shape[:2], np.float32)
  for grp in range(c.shape[2]):
    acc = ((c[:, :, grp] * dw[None, :, grp]).astype(np.float64) * xd[:, None, grp] + acc).astype(np.float32)
  return acc

def _imma_sink(tokens:int, out_features:int, in_features:int, token_tile:int=64, warps:int=4, out_tiles:int=2, stage:int=4):
  groups = in_features // 32
  params = [UOp.placeholder((tokens, out_features), dtypes.float32, slot=0), UOp.placeholder((out_features*groups*9,), dtypes.uint16, slot=1),
            UOp.placeholder((tokens, groups, 8), dtypes.uint32, slot=2), UOp.placeholder((tokens, groups), dtypes.float32, slot=3)]
  return nv_gemm._q4_0_linear_imma_nv.__wrapped__(*params, out_features=out_features, in_features=in_features, token_tile=token_tile,
                                                  warps=warps, out_tiles=out_tiles, stage=stage)

class TestImmaKernel(unittest.TestCase):
  def test_renders_and_compiles_for_sm86_at_model_shapes(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    # Qwen3.8-27B's Q4_0 prefill linears at the standing 128-token chunk: attn/ssm out, ffn gate/up, GDN qkv, ffn down
    for out_features, in_features in ((5120, 5120), (17408, 5120), (10240, 5120), (5120, 17408)):
      with self.subTest(out=out_features, inp=in_features):
        prg = to_program(_imma_sink(128, out_features, in_features), renderer)
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", src)
        self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)  # the WARP range is the hardware lane, unsplit
        self.assertIn("int lidx1 = threadIdx.y; /* 4 */", src)   # the warps
        self.assertEqual(tuple(prg.arg.global_size[:2]), (2, out_features // 64))  # token blocks fastest: they share weights in L2
        self.assertIn("__syncthreads();", src)
        self.assertIn("__shared__", src)

  def test_every_knob_value_renders_and_compiles(self):
    # each knob value the card sweep would try, one factor at a time around the defaults, at gate/up (in 5120) and down (in 17408)
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    default = (64, 4, 2, 2)
    variants = [default] + [default[:i] + (v,) + default[i+1:] for i, vals in enumerate(((16, 32, 128), (1, 2, 8), (1, 4), (0, 1, 4, 8)))
                            for v in vals]
    for out_features, in_features in ((17408, 5120), (5120, 17408)):
      for token_tile, warps, out_tiles, stage in variants:
        with self.subTest(out=out_features, token_tile=token_tile, warps=warps, out_tiles=out_tiles, stage=stage):
          prg = to_program(_imma_sink(128, out_features, in_features, token_tile, warps, out_tiles, stage), renderer)
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn(f"__launch_bounds__({32*warps})", src)
          self.assertEqual(src.count("__syncthreads();"), 2 if stage else 0)
          self.assertEqual(prod(prg.arg.global_size), 128 // token_tile * out_features // (8*warps*out_tiles))  # a size-1 axis drops out

  def test_numpy_reference_matches_ggml_dequant(self):
    # the reference's integer form == gguf.py's own Q4_0 dequant times the dequantized activations (up to fp32 rounding)
    from tinygrad.llm.gguf import ggml_data_to_tensor
    rng = np.random.default_rng(1)
    tokens, out_features, in_features = 4, 8, 128
    blocks = _q4_0_blocks(rng, out_features, in_features // 32)
    xq, xd = _q8(rng, tokens, in_features)
    w = ggml_data_to_tensor(Tensor(blocks.reshape(-1)), out_features * in_features, Q4_0).reshape(out_features, in_features).numpy()
    x = (xq.view(np.int8).astype(np.float32) * xd[..., None]).reshape(tokens, in_features)
    np.testing.assert_allclose(_np_imma_reference(blocks, xq, xd), x @ w.T, rtol=1e-4, atol=1e-4)

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_kernel_runs_on_python_emulator(self):
    # the kernel itself executed lane by lane (every fragment index, the WARP/LOCAL thread mapping, the shared-memory staging
    # and its barriers, the epilogue scales) on the PYTHON emulator, whose m16n8k32 model is the PTX ISA's. Several output and
    # token blocks and warps; stage 0 (global loads), 1, 2 and 4, including fills that don't divide the block's threads
    # (16x1 scales over 64 threads, 16x8 words over 128) and a warp-range split (stage 4 x 2 warps: 32-word rows, 64 threads)
    for tokens, out_features, in_features, token_tile, warps, out_tiles, stage in (
        (32, 64, 64, 32, 2, 2, 0), (32, 64, 128, 16, 2, 2, 1), (64, 32, 128, 32, 1, 4, 4), (32, 128, 256, 16, 4, 1, 2),
        (32, 64, 256, 16, 2, 2, 4)):
      with self.subTest(tokens=tokens, out=out_features, inp=in_features, token_tile=token_tile, warps=warps, out_tiles=out_tiles,
                        stage=stage):
        rng = np.random.default_rng(tokens + out_features)
        blocks, (xq, xd) = _q4_0_blocks(rng, out_features, in_features // 32), _q8(rng, tokens, in_features)
        fxn = functools.partial(nv_gemm._q4_0_linear_imma_nv.__wrapped__, out_features=out_features, in_features=in_features,
                                token_tile=token_tile, warps=warps, out_tiles=out_tiles, stage=stage)
        out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device="PYTHON")
        srcs = [Tensor(a, device="PYTHON") for a in (blocks.reshape(-1).view(np.uint16), xq, xd)]
        res = out.custom_kernel(*srcs, fxn=fxn)[0].numpy()
        np.testing.assert_array_equal(res, _np_imma_reference(blocks, xq, xd))  # same fp32 op order: bit-exact

class TestGates(unittest.TestCase):
  def test_dispatch_stays_generic_off_nv(self):
    # CPU: nv_quant_supported is False, so Linear.__call__ never reaches nv_wmma_linear; and called directly it returns None
    import numpy as np
    from tinygrad import nn, Context
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels.amd import Linear
    rng = np.random.default_rng(0)
    blocks = [np.float16(0.5).tobytes() + rng.integers(-4, 5, 32).astype(np.int8).tobytes() for _ in range(16)]
    packed = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16 * 32, Q8_0).reshape(16, 32)
    lin = Linear(32, 16, bias=False)
    nn.state.load_state_dict(lin, {"weight": decoded}, verbose=False, realize=False)
    x = Tensor.randn(1, 32, 32, dtype=dtypes.float16)
    with Context(NV_CUSTOM_QUANT=1):
      self.assertEqual(NV_CUSTOM_QUANT.value, 1)
      np.testing.assert_allclose(lin(x).numpy(), (x.float() @ decoded.T).numpy(), rtol=1e-3, atol=1e-3)
      lin.set_quantized(lin.weight)
      self.assertEqual(lin.ggml_type, Q8_0)
      self.assertIsNone(nv_gemm.nv_wmma_linear(lin, x))          # CPU: the sm_80 gate is closed
      with Context(NV_WMMA=0): self.assertIsNone(nv_gemm.nv_wmma_linear(lin, x))

  def test_wmma_gate_closed_off_nv(self):
    for dev in ("CPU", "NULL", None, ("CPU", "CPU:1")): self.assertFalse(nv_gemm._nv_wmma_ok(dev), dev)

  def test_tile_helpers(self):
    self.assertEqual([nv_gemm._token_tile_for(t) for t in (16, 32, 48, 64, 128)], [16, 32, 16, 64, 64])
    # T4.124c: the knob caps which of {64, 32, 16} is tried; tokens=64 picks the largest divisor <= knob (128 doesn't
    # divide 64, so it falls through to 32, same as the uncapped default would for tokens=32).
    self.assertEqual([nv_gemm._token_tile_for(64, knob=k) for k in (16, 32, 64, 128)], [16, 32, 64, 32])
    self.assertEqual([nv_gemm._output_tiles_for(o) for o in (8, 16, 32, 48)], [1, 1, 4, 1])

def _quantized_linear(ggml_type:int, out_features:int, in_features:int):
  # a Linear holding real packed ggml blocks, set_quantized the way gguf loading does (byte view at a 4-aligned offset)
  from tinygrad import nn
  from tinygrad.llm.gguf import ggml_data_to_tensor
  from tinygrad.llm.kernels.amd import Linear
  weights, block_bytes = BLOCK[ggml_type]
  rng = np.random.default_rng(ggml_type)
  packed = rng.integers(0, 256, out_features * in_features // weights * block_bytes, dtype=np.uint8).reshape(-1, block_bytes)
  packed[:, :2] = np.full((len(packed), 1), 0.01, np.float16).view(np.uint8)  # finite f16 scales
  raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
  lin = Linear(in_features, out_features, bias=False)
  nn.state.load_state_dict(lin, {"weight": ggml_data_to_tensor(raw, out_features * in_features, ggml_type).reshape(out_features, in_features)},
                           verbose=False, realize=False)
  lin.set_quantized(lin.weight)
  assert lin.ggml_type == ggml_type
  return lin

def _kernels(t:Tensor) -> dict[str, UOp]:
  return {k.arg.name: k for u in t.uop.toposort() if u.op is Ops.CALL and (k := u.src[0]).op is Ops.SINK and isinstance(k.arg, KernelInfo)}

class TestImmaDispatch(unittest.TestCase):
  """T4.126: nv_wmma_linear with the sm_80 gate forced open (graph build only, nothing realized): NV_IMMA=0 builds exactly the
  fp16 WMMA kernel (the very UOp _wmma_kernel builds, so the same render), NV_IMMA=1 swaps in q8_quantize + the IMMA kernel
  for Q4_0 only and reads its knobs at call time."""
  def setUp(self):
    from unittest import mock
    patcher = mock.patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_imma_off_is_the_fp16_kernel(self):
    from tinygrad import Context
    lin, x = _quantized_linear(Q4_0, 256, 128), Tensor.empty(1, 64, 128, dtype=dtypes.float32)
    with Context(NV_IMMA=0):  # T4.126: on by default since the 10-28 card run; 0 is the rollback to the fp16 kernel
      kernels = _kernels(nv_gemm.nv_wmma_linear(lin, x))
    self.assertEqual(set(kernels), {"linear_q4_0_wmma_nv"})
    self.assertIs(kernels["linear_q4_0_wmma_nv"], _sink(Q4_0, 64, 256, 128))

  def test_imma_on_takes_q4_0_only(self):
    from tinygrad import Context
    x = Tensor.empty(1, 64, 128, dtype=dtypes.float32)
    with Context(NV_IMMA=1):
      kernels = _kernels(out := nv_gemm.nv_wmma_linear(_quantized_linear(Q4_0, 256, 128), x))
      self.assertEqual(set(kernels), {"q8_quantize_nv", "linear_q4_0_imma_nv"})
      self.assertEqual(out.shape, (1, 64, 256))
      self.assertIs(kernels["linear_q4_0_imma_nv"], _imma_sink(64, 256, 128))  # the knob defaults: tile 64, 4 warps, 2 tiles, stage 4
      for ggml_type in (Q8_0, Q4_1, IQ4_NL):  # every other format keeps its fp16 WMMA kernel
        self.assertNotIn("linear_q4_0_imma_nv", _kernels(nv_gemm.nv_wmma_linear(_quantized_linear(ggml_type, 256, 128), x)))
      self.assertEqual(set(_kernels(nv_gemm.nv_wmma_linear(_quantized_linear(Q4_0, 72, 128), x))), {"linear_q4_0_wmma_nv"})  # 72 % 64

  def test_knobs_reach_the_kernel(self):
    from tinygrad import Context
    lin, x = _quantized_linear(Q4_0, 256, 128), Tensor.empty(1, 64, 128, dtype=dtypes.float32)
    with Context(NV_IMMA=1, NV_IMMA_TOKEN_TILE=32, NV_IMMA_WARPS=2, NV_IMMA_OUT_TILES=4, NV_IMMA_STAGE=0):
      self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))["linear_q4_0_imma_nv"], _imma_sink(64, 256, 128, 32, 2, 4, 0))
    with Context(NV_IMMA=1, NV_IMMA_TOKEN_TILE=128, NV_IMMA_STAGE=3):  # tile: largest power of two dividing 64; stage: gcd with 4 groups
      self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))["linear_q4_0_imma_nv"], _imma_sink(64, 256, 128, 64, 4, 2, 1))

  def test_symbolic_chunk_pads_and_shrinks(self):
    from tinygrad import Context, UOp as U
    lin = _quantized_linear(Q4_0, 256, 128)
    x = Tensor.empty(1, 64, 128, dtype=dtypes.float32)[:, :U.variable("T", 1, 64).bind(48)]
    with Context(NV_IMMA=1):
      out = nv_gemm.nv_wmma_linear(lin, x)
    self.assertEqual(out.shape[-1], 256)
    self.assertIn("linear_q4_0_imma_nv", _kernels(out))

  def test_geometry_shrinks_stage_to_fit_shared_memory(self):
    from tinygrad import Context
    lin = _quantized_linear(Q4_0, 256, 1024)  # 32 groups
    with Context(NV_IMMA=1, NV_IMMA_TOKEN_TILE=128, NV_IMMA_STAGE=32):
      self.assertEqual(nv_gemm._imma_geometry(lin, 128)["stage"], 8)  # 128 tokens x (68 + 9) words = 39 KB; 16 would be 75 KB

def _emulated_lane_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  # warp_reduce's xor butterfly for q8_quantize_nv's one-warp lane range, each __shfl_xor_sync swapped for a shared exchange
  # (the PYTHON emulator has no CUDA intrinsics and runs the block in lockstep) -- test_llm_nv_norm's _emulated_warp_reduce
  assert full_wave
  lane = next(u for u in val.toposort() if u.op is Ops.RANGE and u.arg[-1] is AxisType.LOCAL)
  for off in (16, 8, 4, 2, 1):
    buf = UOp.placeholder((32,), dtypes.float32, slot=60 + off, addrspace=AddrSpace.LOCAL)
    other = buf.after(buf[lane].store(val))[lane + off - (lane // off) % 2 * (2 * off)].load()
    val = val.maximum(other) if maximum else val + other
  return val

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestImmaEndToEndOnEmulator(unittest.TestCase):
  """T4.126: Linear.set_quantized's real Q4_0 views -> nv_wmma_linear with NV_IMMA=1 -> q8_quantize + the IMMA kernel, all executed
  on the PYTHON emulator (gate forced open, warp shuffle emulated). Bit-exact vs the numpy reference fed the quantizer's own
  output, and close to the fp32 matmul of the dequantized weights (q8 activations are the only approximation)."""
  def test_q4_0_linear(self):
    from unittest import mock
    from tinygrad import nn, Context
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels import nv_quant
    from tinygrad.llm.kernels.amd import Linear
    tokens, out_features, in_features = 32, 64, 128
    rng = np.random.default_rng(7)
    blocks = _q4_0_blocks(rng, out_features, in_features // 32)
    raw = Tensor(np.pad(blocks.reshape(-1), (4, 0)), device="PYTHON").contiguous().realize()[4:]
    lin = Linear(in_features, out_features, bias=False)
    decoded = ggml_data_to_tensor(raw, out_features * in_features, Q4_0).reshape(out_features, in_features)
    nn.state.load_state_dict(lin, {"weight": decoded}, verbose=False, realize=False)
    lin.set_quantized(lin.weight)
    x_np = rng.standard_normal((1, tokens, in_features)).astype(np.float32)
    x = Tensor(x_np, device="PYTHON")
    with mock.patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True), mock.patch.object(nv_quant, "warp_reduce", _emulated_lane_reduce), \
         mock.patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__), Context(NV_IMMA=1, NV_IMMA_WARPS=2):
      out = nv_gemm.nv_wmma_linear(lin, x)
      self.assertIn("linear_q4_0_imma_nv", _kernels(out))
      xq, xd = (t.numpy() for t in nv_quant.q8_quantize(x, tokens, in_features))  # the very tensors the dispatch used (T4.123a cache)
      res = out.numpy()
    np.testing.assert_array_equal(res[0], _np_imma_reference(blocks, xq, xd))
    # q8 activation error only: max 0.033 here on outputs of mean magnitude 1.2 (a wiring slip would be O(1))
    np.testing.assert_allclose(res[0], x_np[0] @ decoded.numpy().T, rtol=0, atol=0.05)

if __name__ == "__main__": unittest.main()
