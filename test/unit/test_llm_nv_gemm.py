"""T4.98f/T4.98j: NV WMMA (mma.sync m16n8k16) prefill gemm for Q8_0/Q4_0/Q4_1/IQ4_NL and the K-quants -- graph builds
for the real shapes, the sm_86 render pins the lane-range split the fragment mapping relies on, and the dispatch gates.
Numerics are hardware-only (t4x/wmma_probe.py, nv_gemm_validate.py). T4.126's int8 (IMMA, m16n8k32) Q4_0 kernel is different:
the PYTHON emulator models that mma, so its fragment map, staging and epilogue are executed there bit-exact vs numpy (T4.128: and
the same kernel for Q4_1 and Q5_K)."""
import functools, itertools, re, sys, unittest
from math import prod
from typing import cast
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

# ******** T4.126: the IMMA kernel -- numpy reference of the exact quantized math, executed on the PYTHON emulator (T4.128: + Q4_1, Q5_K)

IMMA_TYPES = (Q4_0, Q4_1, Q5_K)

def _blocks(ggml_type:int, rng, out_features:int, groups:int) -> np.ndarray:
  # random packed blocks with small finite f16 scales: Q4_0 (out, groups, 18: d, qs[16]), Q4_1 (out, groups, 20: d, m, qs[16]),
  # Q5_K (out, groups/8, 176: d, dmin, scales[12], qh[32], qs[128]) -- any other byte is a valid nibble/bit/6-bit-scale pattern
  nblk, nbytes, nhalf = {Q4_0: (groups, 18, 1), Q4_1: (groups, 20, 2), Q5_K: (groups // 8, 176, 2)}[ggml_type]
  blocks = rng.integers(0, 256, (out_features, nblk, nbytes), dtype=np.uint8)
  blocks[..., :2*nhalf] = rng.uniform(-0.05, 0.05, (out_features, nblk, nhalf)).astype(np.float16).view(np.uint8)
  return blocks

def _raw(ggml_type:int, blocks:np.ndarray) -> np.ndarray:  # the view the kernel reads: Q4_0 u16 (packed_u16), Q4_1/Q5_K u32
  return blocks.reshape(-1).view(np.uint16 if ggml_type == Q4_0 else np.uint32)

def _q8(rng, tokens:int, in_features:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # q8_quantize_nv's exact formula (scale = max|x|/127 clamped at 1e-8, q = clip(rint(x/scale))), packed 4 int8 per uint32, and
  # T4.128's group sums scale * sum(q) (one fp32 product of exact values, as the kernel computes it)
  x = rng.standard_normal((tokens, in_features // 32, 32)).astype(np.float32)
  xd = np.maximum(np.abs(x).max(-1) / np.float32(127), np.float32(1e-8)).astype(np.float32)
  q = np.clip(np.rint(x / xd[..., None]), -127, 127).astype(np.int8)
  return q.view(np.uint32), xd, (xd * q.astype(np.int32).sum(-1).astype(np.float32)).astype(np.float32)

def _f16(b:np.ndarray) -> np.ndarray: return b.copy().view(np.float16)[..., 0].astype(np.float32)

def _np_weights(ggml_type:int, blocks:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray|None]:
  # (q, D, M) with weight = D*q + M per (output, group): q (out, groups, 32) int as the B fragment holds it, D/M (out, groups) fp32
  # exactly as the kernel's epilogue computes them (Q5_K: d*sc and -dmin*m, both exact products)
  if ggml_type in (Q4_0, Q4_1):
    qs = blocks[..., 2*(ggml_type == Q4_1) + 2:].astype(np.int64)
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1) - (8 if ggml_type == Q4_0 else 0)
    return q, _f16(blocks[..., 0:2]), (_f16(blocks[..., 2:4]) if ggml_type == Q4_1 else None)
  d, dmin = _f16(blocks[..., 0:2]), _f16(blocks[..., 2:4])
  sc8, qh, qs = (blocks[..., a:b].astype(np.int64) for a, b in ((4, 16), (16, 48), (48, 176)))
  qj, Dj, Mj = [], [], []
  for j in range(8):  # gguf.py ggml_type 13 / _q4q5_k_scales
    if j < 4: sc, mn = sc8[..., j] & 63, sc8[..., j + 4] & 63
    else: sc, mn = (sc8[..., j + 4] & 0xF) | ((sc8[..., j - 4] >> 6) << 4), (sc8[..., j + 4] >> 4) | ((sc8[..., j] >> 6) << 4)
    nib = qs[..., (j // 2)*32:(j // 2)*32 + 32]
    qj.append(((nib >> 4) if j % 2 else (nib & 0xF)) | (((qh >> j) & 1) << 4))
    Dj.append(d * sc.astype(np.float32))
    Mj.append(-(dmin * mn.astype(np.float32)))
  out = blocks.shape[0]
  return np.stack(qj, 2).reshape(out, -1, 32), np.stack(Dj, 2).reshape(out, -1), np.stack(Mj, 2).reshape(out, -1)

def _np_imma_reference(ggml_type:int, blocks:np.ndarray, xq:np.ndarray, xd:np.ndarray, xs:np.ndarray|None=None) -> np.ndarray:
  # the kernel's math: int32 dot per (token, output, group) of q8 x q, then acc = fma(float(c) * D, xd, acc) in fp32 (T4.128: then
  # acc = fma(M, xs, acc) for Q4_1/Q5_K), group by group in order (the kernel's REDUCE loop). Fused like the card's FFMA (nvcc
  # contracts a*b+c by default) and the PYTHON renderer's MULACC: the fp32 product of two fp32 values is exact in fp64, so one
  # rounding after the add
  qw, D, M = _np_weights(ggml_type, blocks)
  c = np.einsum("tgk,ogk->tog", xq.view(np.int8).astype(np.int64), qw).astype(np.float32)
  acc = np.zeros(c.shape[:2], np.float32)
  for grp in range(c.shape[2]):
    acc = ((c[:, :, grp] * D[None, :, grp]).astype(np.float64) * xd[:, None, grp] + acc).astype(np.float32)
    if M is not None: acc = (M[None, :, grp].astype(np.float64) * cast(np.ndarray, xs)[:, None, grp] + acc).astype(np.float32)
  return acc

def _imma_params(ggml_type:int, tokens:int, out_features:int, in_features:int, splitk:int=1) -> list[UOp]:
  groups = in_features // 32
  raw = {Q4_0: (out_features*groups*9, dtypes.uint16), Q4_1: (out_features*groups*5, dtypes.uint32),
         Q5_K: (out_features*groups//8*44, dtypes.uint32)}[ggml_type]
  out_shape = (splitk, tokens, out_features) if splitk > 1 else (tokens, out_features)  # T4.131: split-K partials
  return [UOp.placeholder(out_shape, dtypes.float32, slot=0), UOp.placeholder((raw[0],), raw[1], slot=1),
          UOp.placeholder((tokens, groups, 8), dtypes.uint32, slot=2), UOp.placeholder((tokens, groups), dtypes.float32, slot=3)] + \
         ([UOp.placeholder((tokens, groups), dtypes.float32, slot=4)] if ggml_type != Q4_0 else [])

def _imma_sink(tokens:int, out_features:int, in_features:int, token_tile:int=64, warps:int=4, out_tiles:int=2, stage:int=4,
               ggml_type:int=Q4_0, pipe:int=0, minblocks:int=0, wprefetch:int=0, splitk:int=1):
  return nv_gemm._linear_imma_nv.__wrapped__(*_imma_params(ggml_type, tokens, out_features, in_features, splitk), ggml_type=ggml_type,
                                             out_features=out_features, in_features=in_features, token_tile=token_tile, warps=warps,
                                             out_tiles=out_tiles, stage=stage, pipe=pipe, minblocks=minblocks, wprefetch=wprefetch, splitk=splitk)

def _shared_bytes(src:str) -> int:  # static __shared__ total of a rendered kernel (every tile here is a 4-byte type)
  return sum(4 * int(n) for n in re.findall(r"__shared__ __align__\(\d+\) \w+ \w+\[(\d+)\];", src))

# Qwen3.8-27B's IMMA prefill linears (out, in): Q4_0 attn/ssm out, ffn gate/up, GDN qkv, ffn down; Q4_1 ffn_down of 8 blocks;
# Q5_K GDN ssm_out
MODEL_SHAPES = {Q4_0: ((5120, 5120), (17408, 5120), (10240, 5120), (5120, 17408)), Q4_1: ((5120, 17408),), Q5_K: ((5120, 6144),)}

class TestImmaKernel(unittest.TestCase):
  def test_renders_and_compiles_for_sm86_at_model_shapes(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for ggml_type, shapes in MODEL_SHAPES.items():
      for out_features, in_features in shapes:
        with self.subTest(ggml_type=ggml_type, out=out_features, inp=in_features):
          prg = to_program(_imma_sink(128, out_features, in_features, ggml_type=ggml_type), renderer)
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn(nv_gemm.IMMA_FORMATS[ggml_type][2], src)
          self.assertIn("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", src)
          self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)  # the WARP range is the hardware lane, unsplit
          self.assertIn("int lidx1 = threadIdx.y; /* 4 */", src)   # the warps
          self.assertEqual(tuple(prg.arg.global_size[:2]), (2, out_features // 64))  # token blocks fastest: they share weights in L2
          self.assertEqual(src.count("__syncthreads();"), 2)
          self.assertEqual(src.count("__shared__"), 2 if ggml_type == Q4_0 else 3)  # T4.128: + the group sums tile
          if ggml_type != Q4_0: self.assertIn("data4_", src)  # the sums input is read

  def test_every_knob_value_renders_and_compiles(self):
    # each knob value the card sweep would try, one factor at a time around the defaults, at gate/up (in 5120) and down (in 17408);
    # T4.128: the same sweep for Q4_1 and Q5_K at their model shapes
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    default = (64, 4, 2, 2)
    variants = [default] + [default[:i] + (v,) + default[i+1:] for i, vals in enumerate(((16, 32, 128), (1, 2, 8), (1, 4), (0, 1, 4, 8)))
                            for v in vals]
    for ggml_type, shapes in ((Q4_0, ((17408, 5120), (5120, 17408))), (Q4_1, MODEL_SHAPES[Q4_1]), (Q5_K, MODEL_SHAPES[Q5_K])):
      for out_features, in_features in shapes:
        for token_tile, warps, out_tiles, stage in variants:
          # T4.128: and the pipelined kernel wherever there is a stage tile; T4.131: and the weight prefetch with either
          for pipe, wprefetch in itertools.product((0, 1) if stage else (0,), (0, 1)):
            with self.subTest(ggml_type=ggml_type, out=out_features, token_tile=token_tile, warps=warps, out_tiles=out_tiles, stage=stage,
                              pipe=pipe, wprefetch=wprefetch):
              prg = to_program(_imma_sink(128, out_features, in_features, token_tile, warps, out_tiles, stage, ggml_type, pipe, wprefetch=wprefetch),
                               renderer)
              src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
              self.assertIn(f"__launch_bounds__({32*warps})", src)
              self.assertEqual(src.count("__syncthreads();"), 2 if stage else 0)  # pipe: one before the loop, one per step
              self.assertEqual(prod(prg.arg.global_size), 128 // token_tile * out_features // (8*warps*out_tiles))  # a size-1 axis drops out
              self.assertLessEqual(_shared_bytes(src), 48 * 1024)
              # T4.131: the weight register array holds every raw[] read of a stage (per (group, n-tile): Q4_0 2 qs + 2 d, Q4_1 1 qs +
              # 2 d|m, Q5_K 4 qs/qh + 2x4 scale words) in the raw view's word type; without the prefetch there is no such array
              regs = re.findall(r"^  (unsigned short|uint) buf\d+\[(\d+)\];$", src, re.M)
              reads = max(stage, 1) * out_tiles * {Q4_0: 4, Q4_1: 3, Q5_K: 12}[ggml_type]
              self.assertEqual(regs, [("unsigned short" if ggml_type == Q4_0 else "uint", str(reads))] if wprefetch else [])

  def test_pipe_0_is_the_t4126_render(self):
    # T4.128: NV_IMMA_PIPE=0 must render exactly the kernel the card measured (md5 of the T4.126 source at 0c6178834, the
    # standing Q4_0 attn/ssm-out shape and knob defaults). A legitimate codegen change elsewhere moves this hash too: re-pin it
    # only after checking the diff is that change, not this kernel
    import hashlib
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    src = next(u.arg for u in to_program(_imma_sink(128, 5120, 5120), renderer).src if u.op is Ops.SOURCE)
    self.assertEqual(hashlib.md5(src.encode()).hexdigest(), "91ffc1757984810a4116f7fb2f7f3e40")

  def test_pipe_one_barrier_per_step_and_prefetch_stores_after_the_mmas(self):
    # T4.128: the pipelined kernel's point, as rendered: the prologue fill + its barrier before the loop, inside the loop the
    # current copy's mmas BEFORE the next stage's shared-memory stores (so their global loads' latency hides behind the mmas),
    # then the step's one barrier closing the loop body. Twice the tile memory.
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for ggml_type, shapes in MODEL_SHAPES.items():
      for out_features, in_features in shapes:
        with self.subTest(ggml_type=ggml_type, out=out_features, inp=in_features):
          srcs = [next(u.arg for u in to_program(_imma_sink(128, out_features, in_features, ggml_type=ggml_type, pipe=pipe), renderer).src
                       if u.op is Ops.SOURCE) for pipe in (0, 1)]
          head, loop = srcs[1].split("  for (int Ridx3 = 0;")
          self.assertEqual((head.count("__syncthreads();"), loop.count("__syncthreads();")), (1, 1))
          self.assertTrue(loop.split("\n  }\n")[0].endswith("__syncthreads();"), "the step's barrier must close the loop body")
          body = loop.split("\n  }\n")[0]
          self.assertLess(body.rindex("__WMMA_8_16_32_signed_char_int("), body.index("*((uint4*)((buf8+"))
          self.assertEqual(_shared_bytes(srcs[1]), 2 * _shared_bytes(srcs[0]))

  def test_wprefetch_loads_next_stage_before_the_mmas_and_stores_the_registers_after(self):
    # T4.131: the weight-prefetch kernel, as rendered, at every model shape: the prologue loads stage 0's words into the register
    # array; inside the loop the next stage's global loads are issued before the first mma, every register read (this stage's
    # words) and every mma come before the first register store (the refill), and one step refills the whole array
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for ggml_type, shapes in MODEL_SHAPES.items():
      for out_features, in_features in shapes:
        with self.subTest(ggml_type=ggml_type, out=out_features, inp=in_features):
          src = next(u.arg for u in to_program(_imma_sink(128, out_features, in_features, ggml_type=ggml_type, wprefetch=1), renderer).src
                     if u.op is Ops.SOURCE)
          name, size = re.search(r"^  (?:unsigned short|uint) (buf\d+)\[(\d+)\];$", src, re.M).groups()
          head, body = src.split("  for (int Ridx3 = 0;")
          body = body.split("\n  }\n")[0]
          self.assertEqual(head.count(f"*({name}+"), int(size))  # the prologue fills every slot
          stores = [m.start() for m in re.finditer(rf"^    \*\({name}\+\d+\) = ", body, re.M)]
          self.assertEqual(len(stores), int(size))  # one step refills the whole array
          # the refill issues the prologue's loads again: one per slot, fewer where the graph shares a word (Q5_K's four groups of a
          # stage read the same super-block scale words: 12 loads feed 96 slots)
          self.assertEqual(body.count("(*(data1_"), head.count("(*(data1_"))
          self.assertLessEqual(body.count("(*(data1_"), int(size))
          self.assertGreaterEqual(body.count("(*(data1_"), int(size) if ggml_type != Q5_K else 12)
          self.assertLess(body.index("(*(data1_"), body.index("__WMMA_8_16_32_signed_char_int("))  # next stage's loads issue first
          self.assertLess(body.rindex("__WMMA_8_16_32_signed_char_int("), stores[0])                  # the refill after the mmas
          self.assertLess(body.rindex(f"(*({name}+"), stores[0])                                       # and after every register read

  def test_numpy_reference_matches_ggml_dequant(self):
    # the reference's integer form == gguf.py's own dequant times the dequantized activations (up to fp32 rounding)
    from tinygrad.llm.gguf import ggml_data_to_tensor
    for ggml_type in IMMA_TYPES:
      with self.subTest(ggml_type=ggml_type):
        rng = np.random.default_rng(1)
        tokens, out_features, in_features = 4, 8, 512
        blocks = _blocks(ggml_type, rng, out_features, in_features // 32)
        xq, xd, xs = _q8(rng, tokens, in_features)
        w = ggml_data_to_tensor(Tensor(blocks.reshape(-1)), out_features * in_features, ggml_type).reshape(out_features, in_features).numpy()
        qw, D, M = _np_weights(ggml_type, blocks)  # the decomposition itself, element by element
        np.testing.assert_allclose(D[..., None] * qw + (0 if M is None else M[..., None]), w.reshape(qw.shape), rtol=1e-6, atol=1e-7)
        x = (xq.view(np.int8).astype(np.float32) * xd[..., None]).reshape(tokens, in_features)
        np.testing.assert_allclose(_np_imma_reference(ggml_type, blocks, xq, xd, xs), x @ w.T, rtol=1e-4, atol=1e-4)

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_kernel_runs_on_python_emulator(self):
    # the kernel itself executed lane by lane (every fragment index, the WARP/LOCAL thread mapping, the shared-memory staging
    # and its barriers, the epilogue scales) on the PYTHON emulator, whose m16n8k32 model is the PTX ISA's. Several output and
    # token blocks and warps; stage 0 (global loads), 1, 2 and 4, including fills that don't divide the block's threads
    # (16x1 scales over 64 threads, 16x8 words over 128) and a warp-range split (stage 4 x 2 warps: 32-word rows, 64 threads).
    # T4.128: Q4_1 and Q5_K too (Q5_K needs in % 256: two super-blocks, a stage of 8 = one whole super-block per step)
    # T4.128 pipe=1 (every staged config): also a single step (128 / stage 4) and an odd step count (96 / stage 1, 768 / stage 8)
    cfgs = {Q4_0: ((32, 64, 64, 32, 2, 2, 0), (32, 64, 128, 16, 2, 2, 1), (64, 32, 128, 32, 1, 4, 4), (32, 128, 256, 16, 4, 1, 2),
                   (32, 64, 256, 16, 2, 2, 4), (32, 64, 96, 16, 2, 2, 1))}
    cfgs[Q4_1] = cfgs[Q4_0]
    cfgs[Q5_K] = ((32, 64, 256, 32, 2, 2, 0), (32, 64, 512, 16, 2, 2, 1), (64, 32, 512, 32, 1, 4, 8), (32, 128, 512, 16, 4, 1, 2),
                  (32, 64, 512, 16, 2, 2, 4), (32, 32, 768, 16, 2, 2, 8))
    # T4.131 wprefetch=1 (every config, so single/odd step counts and both token tiles): the register-staged weights give the same
    # bits as the direct reads (the reference is the same, so knob-on == knob-off)
    for ggml_type, configs in cfgs.items():
      for tokens, out_features, in_features, token_tile, warps, out_tiles, stage in configs:
        for pipe, wprefetch in itertools.product((0, 1) if stage else (0,), (0, 1)):
          with self.subTest(ggml_type=ggml_type, tokens=tokens, out=out_features, inp=in_features, token_tile=token_tile, warps=warps,
                            out_tiles=out_tiles, stage=stage, pipe=pipe, wprefetch=wprefetch):
            rng = np.random.default_rng(tokens + out_features + ggml_type)
            blocks, (xq, xd, xs) = _blocks(ggml_type, rng, out_features, in_features // 32), _q8(rng, tokens, in_features)
            fxn = functools.partial(nv_gemm._linear_imma_nv.__wrapped__, ggml_type=ggml_type, out_features=out_features,
                                    in_features=in_features, token_tile=token_tile, warps=warps, out_tiles=out_tiles, stage=stage, pipe=pipe,
                                    wprefetch=wprefetch)
            out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device="PYTHON")
            srcs = [Tensor(a, device="PYTHON") for a in (_raw(ggml_type, blocks), xq, xd) + ((xs,) if ggml_type != Q4_0 else ())]
            res = out.custom_kernel(*srcs, fxn=fxn)[0].numpy()
            np.testing.assert_array_equal(res, _np_imma_reference(ggml_type, blocks, xq, xd, xs))  # same fp32 op order: bit-exact

class TestImmaSplitK(unittest.TestCase):
  """T4.131: split-K -- splitk blocks per output tile each walk steps/splitk stages into out[k] of a (splitk, tokens, out) partials
  buffer; the caller sums over k. Not bit-identical to the one-pass kernel (fp32 order), but every partial is bit-exact for its
  own groups and the sum is within fp32 rounding of one pass."""
  def setUp(self):
    from unittest import mock
    patcher = mock.patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True)  # the dispatch test: the sm_80 gate forced open
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_renders_at_model_shapes(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for ggml_type, shapes in MODEL_SHAPES.items():
      for out_features, in_features in shapes:
        for splitk, pipe, wprefetch in itertools.product((2, 4), (0, 1), (0, 1)):
          with self.subTest(ggml_type=ggml_type, out=out_features, inp=in_features, splitk=splitk, pipe=pipe, wprefetch=wprefetch):
            prg = to_program(_imma_sink(128, out_features, in_features, ggml_type=ggml_type, pipe=pipe, wprefetch=wprefetch, splitk=splitk), renderer)
            src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
            # the split axis is the fastest grid axis; each block walks steps/splitk stages; the partials buffer is the 3-D out
            self.assertEqual(tuple(prg.arg.global_size), (splitk, 2, out_features // 64))
            self.assertIn(f"int gidx0 = blockIdx.x; /* {splitk} */", src)
            self.assertIn(f"for (int Ridx3 = 0; Ridx3 < {in_features // 32 // 4 // splitk}; Ridx3++)", src)
            self.assertIn(f"float* data0_{splitk * 128 * out_features}", src)
            self.assertEqual(src.count("__syncthreads();"), 2)

  def test_splitk_must_divide_the_steps(self):
    with self.assertRaises(AssertionError): _imma_sink(32, 64, 256, 16, 2, 2, 2, splitk=3)  # 4 steps

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_partials_on_python_emulator(self):
    # every partial == the numpy reference over its own groups, bit-exact (a serial fp32 accumulation from zero, like the one-pass
    # kernel over all groups); their fp32 sum is within rounding of the one-pass reference (Q5_K's partials are the largest)
    cfgs = {Q4_0: (32, 64, 256, 16, 2, 2, 2), Q4_1: (32, 64, 256, 16, 2, 2, 2), Q5_K: (32, 64, 1024, 16, 2, 2, 4)}
    for ggml_type, (tokens, out_features, in_features, token_tile, warps, out_tiles, stage) in cfgs.items():
      for splitk, wprefetch, pipe in ((2, 0, 0), (4, 0, 0), (4, 1, 1)):
        with self.subTest(ggml_type=ggml_type, splitk=splitk, wprefetch=wprefetch, pipe=pipe):
          rng = np.random.default_rng(5)
          blocks, (xq, xd, xs) = _blocks(ggml_type, rng, out_features, in_features // 32), _q8(rng, tokens, in_features)
          fxn = functools.partial(nv_gemm._linear_imma_nv.__wrapped__, ggml_type=ggml_type, out_features=out_features, in_features=in_features,
                                  token_tile=token_tile, warps=warps, out_tiles=out_tiles, stage=stage, pipe=pipe, wprefetch=wprefetch, splitk=splitk)
          out = Tensor.empty(splitk, tokens, out_features, dtype=dtypes.float32, device="PYTHON")
          srcs = [Tensor(a, device="PYTHON") for a in (_raw(ggml_type, blocks), xq, xd) + ((xs,) if ggml_type != Q4_0 else ())]
          res = out.custom_kernel(*srcs, fxn=fxn)[0].numpy()
          g, sb = in_features // 32 // splitk, 8 if ggml_type == Q5_K else 1  # groups per split; Q5_K blocks hold 8 groups
          for k in range(splitk):
            ref = _np_imma_reference(ggml_type, blocks[:, k*g//sb:(k+1)*g//sb], xq[:, k*g:(k+1)*g], xd[:, k*g:(k+1)*g],
                                     None if xs is None else xs[:, k*g:(k+1)*g])
            np.testing.assert_array_equal(res[k], ref)
          np.testing.assert_allclose(res.sum(0), _np_imma_reference(ggml_type, blocks, xq, xd, xs), rtol=1e-4, atol=1e-5)

  def test_auto_rule_splits_only_the_low_n_high_k_shapes(self):
    # _imma_splitk(steps, blocks): 128 tokens at the defaults (token tile 64 -> 2 token blocks, 64 outputs per block): the 5120-output
    # layers (160 blocks) and the 10240-output qkv (320) are below two waves of 82 SMs x 2 blocks, gate/up (544) are not
    from tinygrad import Context
    steps = {5120: 40, 17408: 136, 6144: 48}  # stage 4
    with Context(NV_IMMA_SPLITK=0):
      self.assertEqual([nv_gemm._imma_splitk(steps[i], o // 64 * 2) for o, i in ((5120, 5120), (5120, 17408), (10240, 5120), (17408, 5120), (5120, 6144))],
                       [4, 4, 2, 1, 3])
      with Context(NV_IMMA_MINBLOCKS=3):  # three resident blocks per SM -> two waves = 492 blocks (gate/up's 544 still one pass)
        self.assertEqual([nv_gemm._imma_splitk(steps[i], o // 64 * 2) for o, i in ((5120, 5120), (10240, 5120), (17408, 5120), (5120, 6144))], [4, 2, 1, 4])
      self.assertEqual(nv_gemm._imma_splitk(3, 160), 3)  # no divisor reaches two waves: the largest <= 8
      self.assertEqual(nv_gemm._imma_splitk(1, 160), 1)
    with Context(NV_IMMA_SPLITK=1): self.assertEqual([nv_gemm._imma_splitk(40, b) for b in (160, 544)], [1, 1])
    with Context(NV_IMMA_SPLITK=2): self.assertEqual([nv_gemm._imma_splitk(s, 544) for s in (40, 3)], [2, 1])  # forced where it divides

  def test_knob_reaches_the_kernel_and_the_result_is_summed(self):
    from tinygrad import Context
    lin, x = _quantized_linear(Q4_0, 256, 256), Tensor.empty(1, 64, 256, dtype=dtypes.float32)  # 8 groups, stage 4 -> 2 steps
    with Context(NV_IMMA=1, NV_IMMA_SPLITK=2):
      out = nv_gemm.nv_wmma_linear(lin, x)
      kernel = _kernels(out)["linear_q4_0_imma_nv"]
    self.assertIs(kernel, _imma_sink(64, 256, 256, splitk=2))
    self.assertEqual(out.shape, (1, 64, 256))
    self.assertEqual(next(u._shape for u in kernel.toposort() if u.op is Ops.PARAM and u.arg.slot == 0), (2 * 64 * 256,))  # the partials
    self.assertTrue(any(u.op is Ops.REDUCE for u in out.uop.toposort()))  # summed by the generic reduce
    with Context(NV_IMMA=1, NV_IMMA_SPLITK=4):  # 4 doesn't divide 2 steps: one pass
      self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))["linear_q4_0_imma_nv"], _imma_sink(64, 256, 256))

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
  for Q4_0 (T4.128: and Q4_1, Q5_K, with the quantizer's sums) only and reads its knobs at call time."""
  def setUp(self):
    from unittest import mock
    patcher = mock.patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True)
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_imma_off_is_the_fp16_kernel(self):
    from tinygrad import Context
    for ggml_type, in_features in ((Q4_0, 128), (Q4_1, 128), (Q5_K, 256)):
      lin, x = _quantized_linear(ggml_type, 256, in_features), Tensor.empty(1, 64, in_features, dtype=dtypes.float32)
      with Context(NV_IMMA=0):  # T4.126: on by default since the 10-28 card run; 0 is the rollback to the fp16 kernel
        kernels = _kernels(nv_gemm.nv_wmma_linear(lin, x))
      name = nv_gemm.WMMA_FORMATS[ggml_type][2]
      self.assertEqual(set(kernels), {name})
      self.assertIs(kernels[name], _sink(ggml_type, 64, 256, in_features))

  def test_imma_on_takes_q4_0_q4_1_q5_k_only(self):
    from tinygrad import Context
    with Context(NV_IMMA=1):
      for ggml_type, in_features in ((Q4_0, 128), (Q4_1, 128), (Q5_K, 256)):
        x = Tensor.empty(1, 64, in_features, dtype=dtypes.float32)
        kernels = _kernels(out := nv_gemm.nv_wmma_linear(_quantized_linear(ggml_type, 256, in_features), x))
        name = nv_gemm.IMMA_FORMATS[ggml_type][2]
        self.assertEqual(set(kernels), {"q8_quantize_nv", name})
        self.assertEqual(out.shape, (1, 64, 256))
        # the knob defaults (tile 64, 4 warps, 2 tiles, stage 4); T4.128: Q4_1/Q5_K's kernel takes the sums as a 5th input
        self.assertIs(kernels[name], _imma_sink(64, 256, in_features, ggml_type=ggml_type))
        self.assertEqual(sum(u.op is Ops.PARAM for u in kernels["q8_quantize_nv"].toposort()), 3 if ggml_type == Q4_0 else 4)
      x = Tensor.empty(1, 64, 256, dtype=dtypes.float32)
      for ggml_type in (Q8_0, IQ4_NL, Q4_K, Q6_K, IQ4_XS):  # every other format keeps its fp16 WMMA kernel
        self.assertEqual(set(_kernels(nv_gemm.nv_wmma_linear(_quantized_linear(ggml_type, 256, 256), x))), {nv_gemm.WMMA_FORMATS[ggml_type][2]})
      self.assertEqual(set(_kernels(nv_gemm.nv_wmma_linear(_quantized_linear(Q4_0, 72, 256), x))), {"linear_q4_0_wmma_nv"})  # 72 % 64

  def test_knobs_reach_the_kernel(self):
    from tinygrad import Context
    lin, x = _quantized_linear(Q4_0, 256, 128), Tensor.empty(1, 64, 128, dtype=dtypes.float32)
    with Context(NV_IMMA=1, NV_IMMA_TOKEN_TILE=32, NV_IMMA_WARPS=2, NV_IMMA_OUT_TILES=4, NV_IMMA_STAGE=0):
      self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))["linear_q4_0_imma_nv"], _imma_sink(64, 256, 128, 32, 2, 4, 0))
    with Context(NV_IMMA=1, NV_IMMA_TOKEN_TILE=128, NV_IMMA_STAGE=3):  # tile: largest power of two dividing 64; stage: gcd with 4 groups
      self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))["linear_q4_0_imma_nv"], _imma_sink(64, 256, 128, 64, 4, 2, 1))

  def test_pipe_knob_reaches_the_kernel(self):
    # T4.128: NV_IMMA_PIPE=1 builds the pipelined kernel for all three formats; without a stage tile there is nothing to pipeline
    from tinygrad import Context
    for ggml_type, in_features in ((Q4_0, 128), (Q4_1, 128), (Q5_K, 256)):
      lin, x = _quantized_linear(ggml_type, 256, in_features), Tensor.empty(1, 64, in_features, dtype=dtypes.float32)
      name = nv_gemm.IMMA_FORMATS[ggml_type][2]
      with Context(NV_IMMA=1, NV_IMMA_PIPE=1):
        self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))[name], _imma_sink(64, 256, in_features, ggml_type=ggml_type, pipe=1))
      with Context(NV_IMMA=1, NV_IMMA_PIPE=1, NV_IMMA_STAGE=0):
        self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))[name], _imma_sink(64, 256, in_features, 64, 4, 2, 0, ggml_type))

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
    with Context(NV_IMMA=1, NV_IMMA_TOKEN_TILE=128, NV_IMMA_STAGE=32, NV_IMMA_PIPE=1):  # T4.128: two copies -> 2 x 128 x (36 + 5) = 41 KB
      self.assertEqual(nv_gemm._imma_geometry(lin, 128), {"token_tile": 128, "warps": 4, "out_tiles": 2, "stage": 4, "pipe": 1, "minblocks": 0,
                                                          "wprefetch": 0, "splitk": 1})

  def test_wprefetch_knob_reaches_the_kernel(self):
    # T4.131: NV_IMMA_WPREFETCH=1 builds the register-prefetch kernel for all three formats, with and without a stage tile
    from tinygrad import Context
    for ggml_type, in_features in ((Q4_0, 128), (Q4_1, 128), (Q5_K, 256)):
      lin, x = _quantized_linear(ggml_type, 256, in_features), Tensor.empty(1, 64, in_features, dtype=dtypes.float32)
      name = nv_gemm.IMMA_FORMATS[ggml_type][2]
      with Context(NV_IMMA=1, NV_IMMA_WPREFETCH=1):
        self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))[name], _imma_sink(64, 256, in_features, ggml_type=ggml_type, wprefetch=1))
      with Context(NV_IMMA=1, NV_IMMA_WPREFETCH=1, NV_IMMA_STAGE=0, NV_IMMA_PIPE=1):
        self.assertIs(_kernels(nv_gemm.nv_wmma_linear(lin, x))[name], _imma_sink(64, 256, in_features, 64, 4, 2, 0, ggml_type, wprefetch=1))

  def test_minblocks_knob_reaches_the_kernel_and_the_launch_bounds(self):
    # T4.131: NV_IMMA_MINBLOCKS=n -> KernelInfo.min_blocks=n -> __launch_bounds__(threads, n) in the CUDA render; 0 = today's
    # KernelInfo (test_pipe_0_is_the_t4126_render pins that render byte-for-byte)
    from tinygrad import Context
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    lin, x = _quantized_linear(Q4_0, 256, 128), Tensor.empty(1, 64, 128, dtype=dtypes.float32)
    with Context(NV_IMMA=1, NV_IMMA_MINBLOCKS=3):
      kernel = _kernels(nv_gemm.nv_wmma_linear(lin, x))["linear_q4_0_imma_nv"]
    self.assertIs(kernel, _imma_sink(64, 256, 128, minblocks=3))
    self.assertEqual(kernel.arg.min_blocks, 3)
    self.assertEqual(_imma_sink(64, 256, 128).arg.min_blocks, 0)
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    for minblocks, bounds in ((0, "__launch_bounds__(128) "), (3, "__launch_bounds__(128, 3) "), (4, "__launch_bounds__(128, 4) ")):
      src = next(u.arg for u in to_program(_imma_sink(128, 5120, 5120, minblocks=minblocks), renderer).src if u.op is Ops.SOURCE)
      self.assertIn(bounds, src)
      self.assertEqual(src.count("__launch_bounds__"), 1)

_emu_slot = itertools.count(60)
def _emulated_lane_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  # warp_reduce's xor butterfly for q8_quantize_nv's lane range, each __shfl_xor_sync swapped for a shared exchange (the PYTHON
  # emulator has no CUDA intrinsics and runs the block in lockstep) -- test_llm_nv_norm's _emulated_warp_reduce. T4.128: a fresh
  # buffer per exchange (the sums variant reduces twice), and the multi-warp prefill quantizer's WARP lane + LOCAL warp ranges
  # (thread warp*32 + lane, whose xor partner stays in its warp) besides the one-warp kernel's LOCAL lane
  assert full_wave
  rngs = {u.arg[-1]: u for u in val.toposort() if u.op is Ops.RANGE}
  if AxisType.WARP in rngs: tid, n = rngs[AxisType.LOCAL] * 32 + rngs[AxisType.WARP], rngs[AxisType.LOCAL].src[0].arg * 32
  else: tid, n = rngs[AxisType.LOCAL], 32
  for off in (16, 8, 4, 2, 1):
    buf = UOp.placeholder((n,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
    other = buf.after(buf[tid].store(val))[tid + off - (tid // off) % 2 * (2 * off)].load()
    val = val.maximum(other) if maximum else val + other
  return val

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestImmaEndToEndOnEmulator(unittest.TestCase):
  """T4.126: Linear.set_quantized's real views -> nv_wmma_linear with NV_IMMA=1 -> q8_quantize + the IMMA kernel, all executed on
  the PYTHON emulator (gate forced open, warp shuffle emulated). Bit-exact vs the numpy reference fed the quantizer's own output,
  and close to the fp32 matmul of the dequantized weights (q8 activations are the only approximation). T4.128: Q4_1 (u32 view)
  and Q5_K (word view) with the quantizer's sums."""
  def _check(self, ggml_type:int, in_features:int, splitk:int=1):
    from unittest import mock
    from tinygrad import nn, Context
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels import nv_quant
    from tinygrad.llm.kernels.amd import Linear
    tokens, out_features = 32, 64
    rng = np.random.default_rng(7)
    blocks = _blocks(ggml_type, rng, out_features, in_features // 32)
    raw = Tensor(np.pad(blocks.reshape(-1), (4, 0)), device="PYTHON").contiguous().realize()[4:]
    lin = Linear(in_features, out_features, bias=False)
    decoded = ggml_data_to_tensor(raw, out_features * in_features, ggml_type).reshape(out_features, in_features)
    nn.state.load_state_dict(lin, {"weight": decoded}, verbose=False, realize=False)
    lin.set_quantized(lin.weight)
    x_np = rng.standard_normal((1, tokens, in_features)).astype(np.float32)
    x = Tensor(x_np, device="PYTHON")
    with mock.patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True), mock.patch.object(nv_quant, "warp_reduce", _emulated_lane_reduce), \
         mock.patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__), \
         Context(NV_IMMA=1, NV_IMMA_WARPS=2, NV_IMMA_SPLITK=splitk):
      out = nv_gemm.nv_wmma_linear(lin, x)
      self.assertIn(nv_gemm.IMMA_FORMATS[ggml_type][2], _kernels(out))
      # the very tensors the dispatch used (T4.123a cache)
      acts = [t.numpy() for t in nv_quant.q8_quantize(x, tokens, in_features, sums=ggml_type != Q4_0)]
      res = out.numpy()
    if splitk == 1: np.testing.assert_array_equal(res[0], _np_imma_reference(ggml_type, blocks, *acts))
    else:
      # T4.131: the generic reduce sums the partials p0 + p1 + ... in order from zero; each partial is the reference over its groups
      g = in_features // 32 // splitk
      refs = [_np_imma_reference(ggml_type, blocks[:, k*g:(k+1)*g], *(a[:, k*g:(k+1)*g] for a in acts)) for k in range(splitk)]
      np.testing.assert_array_equal(res[0], functools.reduce(np.add, refs))
    # q8 activation error only: max 2.7 % (Q4_0), 2.3 % (Q4_1), 4.5 % (Q5_K: 31*63-scale weights, all positive q) of the mean
    # output magnitude here; a wiring slip would be O(100 %)
    ref = x_np[0] @ decoded.numpy().T
    np.testing.assert_allclose(res[0], ref, rtol=0, atol=0.06 * np.abs(ref).mean())

  def test_q4_0_linear(self): self._check(Q4_0, 128)
  def test_q4_1_linear(self): self._check(Q4_1, 128)
  def test_q5_k_linear(self): self._check(Q5_K, 256)
  def test_q4_0_linear_splitk(self): self._check(Q4_0, 512, splitk=4)  # T4.131: 16 groups, stage 4 -> 4 steps, 4 splits of one step

if __name__ == "__main__": unittest.main()
