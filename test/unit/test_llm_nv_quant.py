"""T4.98c: the NV quantized-linear decode kernel (tinygrad/llm/kernels/nv_quant.py), ported from kernels/amd.py.

Everything here runs WITHOUT a GPU (no DEV=NV, no Device["NV"] opened): kernel UOp graphs are built directly and,
when the nvrtc lane is reachable, rendered+compiled for sm_86 through a CUDARenderer built from a bare Target
spec (mirrors test_llm_nv_scan.py's T6.2 pattern) -- never via an actual NV device. Format unpack math is checked
against gguf.py's ggml_data_to_tensor with pure-numpy transcriptions of what group_dot does, since the kernel
itself can't run here. Numeric hardware parity (does the compiled kernel produce the right numbers on a real
3090) is NOT covered by this file -- see the final report for exactly what must still be run on NV."""
import itertools
import re
import sys
import unittest
from typing import cast
from unittest import mock
import contextlib
import numpy as np
from tinygrad import Tensor, UOp, Context, dtypes, nn
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, KernelInfo
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels import nv_quant
from tinygrad.llm.kernels.amd import (
  Linear, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_WORDS, Q5_WORDS, IQ4_WORDS, Q3_K, IQ3_XXS, IQ3_S, QUANT_SIZES, IQ_GRID_SIZES, Q8_GROUP_SIZE)
from tinygrad.llm.kernels.nv_quant import (
  Q8_0, Q4_0, Q4_1, IQ4_NL, NV_QUANT_TYPES, NV_CUSTOM_QUANT, nv_quant_supported, _quant_decode_kernel, _nv_byte_perm,
  nv_q8_linear, q8_quantize, split_k_partials, _ffn_act_q8_kernel, ffn_act_q8,
)
from tinygrad.llm.model import FFNBlock, GatedDeltaNetBlock, SSMConfig, TransformerBlock, TransformerConfig, nv_fused_norm
from test.helpers import check_schedule
from test.unit.test_llm_nv_norm import np_add_rmsnorm_q8

KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

def _iq3_grids() -> tuple[list, list]:  # (iq3xxs_grid, iq3s_grid), 256 and 512 uint32 words -- loaded once, lazily
  from tinygrad.runtime.autogen import ggml_common as _ggml
  return list(_ggml.iq3xxs_grid), list(_ggml.iq3s_grid)

# ******** pure-numpy re-implementations of the per-block unpack group_dot does, vs gguf.py's reference ********

def _f16(b2) -> np.float32: return np.frombuffer(bytes(b2), dtype="<f2").astype(np.float32)[0]

def ref_q4_0(b:np.ndarray) -> np.ndarray:  # b: uint8[18]
  d = _f16(b[0:2])
  qs = b[2:18]
  q = np.concatenate([qs & 0x0F, qs >> 4]).astype(np.int32)  # gguf q_to_uint8(.,4): low nibbles 0-15, then high 16-31
  return (d * (q - 8)).astype(np.float32)

def ref_q4_1(b:np.ndarray) -> np.ndarray:  # b: uint8[20]
  d, m = _f16(b[0:2]), _f16(b[2:4])
  qs = b[4:20]
  q = np.concatenate([qs & 0x0F, qs >> 4]).astype(np.int32)  # same q_to_uint8(.,4) order as Q4_0, no -8 bias
  return (d * q + m).astype(np.float32)

def ref_q8_0(b:np.ndarray) -> np.ndarray:  # b: uint8[34]
  d = _f16(b[0:2])
  qs = b[2:34].astype(np.int8).astype(np.int32)
  return (d * qs).astype(np.float32)

def ref_q4_k(b:np.ndarray) -> np.ndarray:  # b: uint8[144]
  d, dmin = _f16(b[0:2]), _f16(b[2:4])
  out = np.zeros(256, dtype=np.float32)
  for sg in range(8):
    if sg < 4: scale, minimum = int(b[4+sg]) & 63, int(b[8+sg]) & 63
    else: scale, minimum = (int(b[8+sg]) & 15) | ((int(b[sg]) >> 6) << 4), (int(b[8+sg]) >> 4) | ((int(b[4+sg]) >> 6) << 4)
    qbase, shift = 16 + (sg//2)*32, (sg & 1) * 4
    nibbles = ((b[qbase:qbase+32].astype(np.int32) >> shift) & 0x0F)
    out[sg*32:(sg+1)*32] = d*scale*nibbles - dmin*minimum
  return out

def ref_q5_k(b:np.ndarray) -> np.ndarray:  # b: uint8[176]
  d, dmin = _f16(b[0:2]), _f16(b[2:4])
  out = np.zeros(256, dtype=np.float32)
  for sg in range(8):
    if sg < 4: scale, minimum = int(b[4+sg]) & 63, int(b[8+sg]) & 63
    else: scale, minimum = (int(b[8+sg]) & 15) | ((int(b[sg]) >> 6) << 4), (int(b[8+sg]) >> 4) | ((int(b[4+sg]) >> 6) << 4)
    qbase, shift = 48 + (sg//2)*32, (sg & 1) * 4
    low = (b[qbase:qbase+32].astype(np.int32) >> shift) & 0x0F
    high = (b[16:48].astype(np.int32) >> sg) & 1
    q = low | (high << 4)
    out[sg*32:(sg+1)*32] = d*scale*q - dmin*minimum
  return out

def ref_q6_k(b:np.ndarray) -> np.ndarray:  # b: uint8[210]
  d = _f16(b[208:210])
  out = np.zeros(256, dtype=np.float32)
  for j in range(256):
    sg, pos, k = j // 32, j - (j % 4), j % 4
    within = pos % 128
    low = (int(b[(pos//128)*64 + within%64 + k]) >> ((within//64)*4)) & 0x0F
    high = (int(b[128 + (pos//128)*32 + within%32 + k]) >> ((within//32)*2)) & 0x03
    qv = np.int8(np.uint8(low | (high << 4))).astype(np.int32) - 32
    scale = np.int8(b[192 + sg*2 + (j % 32)//16]).astype(np.int32)
    out[j] = qv * scale * d
  return out.astype(np.float32)

def ref_iq4_xs(b:np.ndarray) -> np.ndarray:  # b: uint8[136]
  d = _f16(b[0:2])
  hi16 = int(b[2]) | (int(b[3]) << 8)
  out = np.zeros(256, dtype=np.float32)
  for sg in range(8):
    low_byte = int(b[4 + sg//2])
    scale4 = (low_byte >> (4*(sg % 2))) & 15
    hi2 = (hi16 >> (2*sg)) & 3
    scale = np.int8(np.uint8(scale4 | (hi2 << 4))).astype(np.int32) - 32
    for jj in range(32):
      nib = (int(b[8 + sg*16 + (jj % 16)]) >> (0 if jj < 16 else 4)) & 0xF
      out[sg*32+jj] = KVALUES_IQ4NL[nib] * d * scale
  return out

def ref_iq4_nl(b:np.ndarray) -> np.ndarray:  # b: uint8[18]
  d = _f16(b[0:2])
  qs = b[2:18]
  q = np.concatenate([qs & 0x0F, qs >> 4]).astype(np.int32)  # Q4_0's byte layout, no sub-block scale
  return (d * np.array(KVALUES_IQ4NL, dtype=np.float32)[q]).astype(np.float32)

def ref_q3_k(b:np.ndarray) -> np.ndarray:  # b: uint8[110] (hmask:32, qs:64, scales:12, d:2)
  d = _f16(b[108:110])
  out = np.zeros(256, dtype=np.float32)
  for e in range(256):
    j, i = e // 32, e % 32
    row, p = j // 4, j % 4
    q2 = (int(b[32 + 32*row + i]) >> (2*p)) & 3
    qh = (int(b[i]) >> j) & 1
    val = q2 + 4*qh - 4  # -4..3, gguf's (q - (~hmask_bit)*4) folded into one add
    s = e // 16  # 16-wide sub-block, two per 32-group (2j, 2j+1)
    low4 = (int(b[96+s]) & 0xF) if s < 8 else ((int(b[88+s]) >> 4) & 0xF)  # q_to_uint8(scales[0:8],4): field-major (no reshape trick)
    high2 = (int(b[104 + s % 4]) >> (2*(s//4))) & 3
    sc = np.int8(np.uint8(low4 | (high2 << 4))).astype(np.int32) - 32
    out[e] = d * sc * val
  return out.astype(np.float32)

def ref_iq3_xxs(b:np.ndarray) -> np.ndarray:  # b: uint8[98] (d:2, qs:64, 8 scale+sign uint32 words at 66)
  d = _f16(b[0:2])
  grid, _ = _iq3_grids()
  out = np.zeros(256, dtype=np.float32)
  for g in range(8):
    w = int(b[66+4*g]) | (int(b[67+4*g]) << 8) | (int(b[68+4*g]) << 16) | (int(b[69+4*g]) << 24)
    db = d * (((w >> 28) & 0xF) + 0.5) * 0.5
    for i in range(32):
      c = g*8 + i//4  # code index into qs[0:64]: one 8-bit code per 4 weights
      mag = (grid[int(b[2+c])] >> (8*(i % 4))) & 0xFF
      u, v = i // 8, i % 8  # 8-weight sub-chunk (parity-derived sign byte) and bit position within it
      s7 = (w >> (7*u)) & 0x7F
      px = s7 ^ (s7 >> 4)  # XOR-fold parity of s7 (ggml derives the 8th sign bit from it)
      px ^= px >> 2
      px ^= px >> 1
      s8 = s7 | ((px & 1) << 7)
      sign = -1 if (s8 >> v) & 1 else 1
      out[g*32+i] = db * sign * mag
  return out.astype(np.float32)

def ref_iq3_s(b:np.ndarray) -> np.ndarray:  # b: uint8[110] (d:2, qs:64, qh:8, signs:32, scales:4)
  d = _f16(b[0:2])
  _, grid = _iq3_grids()
  out = np.zeros(256, dtype=np.float32)
  for g in range(8):
    # scale nibble is BYTE-MAJOR (byte 106+g//2, low nibble if g even else high) -- q_to_uint8 on the (-1,4,1)-
    # reshaped scales bytes keeps the byte index outer and only bit-splits within each byte (checked directly
    # against ggml_data_to_tensor: a naive low-nibbles-then-high-nibbles reading does NOT match)
    nib = (int(b[106 + g//2]) >> ((g % 2)*4)) & 0xF
    scale = 1 + 2*nib
    qh_byte = int(b[66+g])
    for i in range(32):
      c = g*8 + i//4
      qh_bit = (qh_byte >> (i//4)) & 1  # bit `word_idx` (=i//4=c-g*8) of qh[g] -- one qh byte per group
      idx = int(b[2+c]) | (qh_bit << 8)
      mag = (grid[idx] >> (8*(i % 4))) & 0xFF
      sign_byte = int(b[74 + g*4 + i//8])
      sign = -1 if (sign_byte >> (i % 8)) & 1 else 1
      out[g*32+i] = d * scale * sign * mag
  return out.astype(np.float32)

class TestFormatUnpackMath(unittest.TestCase):
  """group_dot's per-group unpack (nibble/byte extraction, scales, the Q4_0 -8 and Q4_K/Q5_K min terms), transcribed
  to pure numpy, must reproduce gguf.py's ggml_data_to_tensor dequantized values exactly -- the kernel itself can't
  run without a GPU, so this is the check that the NV unpack math (shared with amd.py for the 4 existing formats)
  matches the reference."""
  def _check(self, ggml_type:int, nbytes:int, ref, seed:int):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, nbytes, dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    # 32-weight formats vs 256-weight superblocks: nbytes alone doesn't tell them apart (IQ3_XXS packs 256
    # weights into 98 bytes, under the 32-weight formats' widest 34)
    n = 32 if ggml_type in (Q4_0, Q8_0, Q4_1, IQ4_NL) else 256
    expected = ggml_data_to_tensor(raw, n, ggml_type).numpy().reshape(-1)
    np.testing.assert_allclose(ref(packed), expected, rtol=1e-3, atol=1e-3)

  def test_q4_0(self):
    for seed in range(5): self._check(Q4_0, 18, ref_q4_0, seed)
  def test_q4_1(self):
    for seed in range(5): self._check(Q4_1, 20, ref_q4_1, seed)
  def test_q8_0(self):
    for seed in range(5): self._check(Q8_0, 34, ref_q8_0, seed)
  def test_q4_k(self):
    for seed in range(5): self._check(Q4_K, 144, ref_q4_k, seed)
  def test_q5_k(self):
    for seed in range(5): self._check(Q5_K, 176, ref_q5_k, seed)
  def test_q6_k(self):
    for seed in range(5): self._check(Q6_K, 210, ref_q6_k, seed)
  def test_iq4_xs(self):
    for seed in range(5): self._check(IQ4_XS, 136, ref_iq4_xs, seed)
  def test_iq4_nl(self):
    for seed in range(5): self._check(IQ4_NL, 18, ref_iq4_nl, seed)
  def test_q3_k(self):
    for seed in range(5): self._check(Q3_K, 110, ref_q3_k, seed)
  def test_iq3_xxs(self):
    for seed in range(5): self._check(IQ3_XXS, 98, ref_iq3_xxs, seed)
  def test_iq3_s(self):
    for seed in range(5): self._check(IQ3_S, 110, ref_iq3_s, seed)

class TestByteParityIntrinsics(unittest.TestCase):
  """_nv_byte_perm is the highest-risk translation in this port (AMD's v_perm_b32 and CUDA's __byte_perm disagree
  on both operand order and selector bit-width -- see the docstring in nv_quant.py). This simulates both ISAs'
  documented bit semantics in pure Python and checks _nv_byte_perm's UOp-construction logic (mirrored here as
  plain integer ops) against amd.py's _iq4_bytes reproducing the ground-truth kvalues_iq4nl table for all 16 codes."""
  @staticmethod
  def amd_perm(s0:int, s1:int, sel:int) -> int:
    # hardware-verified semantics: test/amd/hw/test_vop3.py::TestPermMore.test_v_perm_b32_select_high_bytes
    src = [(s1 >> (8*i)) & 0xFF for i in range(4)] + [(s0 >> (8*i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
      idx = (sel >> (8*i)) & 0xFF
      byte = 0x00 if idx == 12 else 0xFF if idx >= 13 else src[idx & 7]
      out |= byte << (8*i)
    return out

  @staticmethod
  def nv_byte_perm_translated(a:int, b:int, sel:int) -> int:
    # exactly what _nv_byte_perm builds: swap operands, repack sel's byte-lanes into nibbles, CUDA prmt semantics
    nv_sel = sum(((sel >> (4*i)) & (0xF << (4*i))) for i in range(4))
    src = [(b >> (8*i)) & 0xFF for i in range(4)] + [(a >> (8*i)) & 0xFF for i in range(4)]  # __byte_perm(x,y,s): idx 0-3<-x, 4-7<-y
    out = 0
    for i in range(4):
      idx = (nv_sel >> (4*i)) & 0xF
      out |= (src[idx] if idx < 8 else 0) << (8*i)  # codes 0-7 only: the only range amd.py's callers ever rely on
    return out

  def test_translated_matches_amd_for_the_selectors_iq4_bytes_uses(self):
    # amd_perm and nv_byte_perm_translated must agree on every selector _iq4_bytes can actually rely on: values
    # 0-7 for the `high` call (masked) and the final call (always 0-7); the `low` call's 8-15 lanes are provably
    # discarded downstream (see nv_quant.py's comment), so they're excluded here on purpose.
    rng = np.random.default_rng(0)
    for _ in range(200):
      s0, s1 = int(rng.integers(0, 2**32)), int(rng.integers(0, 2**32))
      sel = int(rng.integers(0, 8)) * 0x01010101  # 4 lanes, each 0-7
      self.assertEqual(self.amd_perm(s0, s1, sel), self.nv_byte_perm_translated(s0, s1, sel))

  def test_iq4_bytes_reproduces_kvalues_iq4nl(self):
    # simulate _iq4_bytes(packed, shift) via nv_byte_perm_translated for every one of the 16 ggml codes and check
    # it reproduces kvalues_iq4nl exactly -- this is what _decode_linear's IQ4_XS branch actually consumes.
    def iq4_bytes(packed:int, shift:int) -> int:
      selectors = (packed >> shift) & 0x0F0F0F0F
      low = self.nv_byte_perm_translated(0xf6eaddcf, 0xbfad9881, selectors)
      high = self.nv_byte_perm_translated(0x71594535, 0x26190d01, selectors & 0x07070707)
      return self.nv_byte_perm_translated(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))
    for code in range(16):
      packed = code  # a single nibble in byte 0, shift=0 selects it
      result_byte0 = iq4_bytes(packed, 0) & 0xFF
      expected = np.array(KVALUES_IQ4NL[code], dtype=np.int8).astype(np.uint8)
      self.assertEqual(result_byte0, int(expected), f"code={code}")

  def test_nv_byte_perm_uop_matches_simulation(self):
    # the actual UOp-construction function (not the plain-int simulation above) must render the same swap+repack
    a, b, sel = UOp.const(0x11223344, dtypes.uint32), UOp.const(0x55667788, dtypes.uint32), UOp.const(0x04050607, dtypes.uint32)
    out = _nv_byte_perm(a, b, sel)
    self.assertEqual(out.op, Ops.CUSTOMI)
    self.assertIn("__byte_perm", out.arg[0])
    # src order must be (b, a, repacked_sel) per the operand swap (each wrapped in a CAST to uint32)
    self.assertEqual(out.src[0].src[0].arg, 0x55667788)
    self.assertEqual(out.src[1].src[0].arg, 0x11223344)

class TestKernelGraphAndRender(unittest.TestCase):
  """Every format's kernel UOp graph must build at model-like shapes; when the nvrtc lane (docker on macOS) is
  reachable, the rendered CUDA source for sm_86 must contain __dp4a and __shfl_xor_sync and no AMD builtin."""
  SHAPES = (12288, 17408)
  IN_FEATURES = 5120

  U16_VIEW_TYPES = (Q4_0, IQ4_NL, Q6_K)  # T4.125a

  @staticmethod
  def _raw_shape(ggml_type:int, out_features:int, in_features:int) -> tuple[tuple[int, ...], object]:
    if ggml_type in (Q4_0, Q8_0, Q4_1, IQ4_NL):
      blocks = in_features // 32
      nbytes = {Q8_0: 34, Q4_1: 20}.get(ggml_type, 18)  # Q4_0 and IQ4_NL are both 18
      total = out_features*blocks*nbytes
      # T4.125a: nv_q8_linear now passes the uint16 view (set_quantized's packed_u16) for Q4_0/IQ4_NL
      if ggml_type in (Q4_0, IQ4_NL): return (total // 2,), dtypes.uint16
      return (total,), dtypes.uint8
    blocks = in_features // 256
    words = {Q4_K: Q4_WORDS, Q5_K: Q5_WORDS, IQ4_XS: IQ4_WORDS}.get(ggml_type)
    if words is not None: return (out_features*blocks*words,), dtypes.uint32
    total = out_features*blocks*QUANT_SIZES[ggml_type]  # Q6_K/Q3_K/IQ3_XXS/IQ3_S
    if ggml_type == Q6_K: return (total // 2,), dtypes.uint16  # T4.125a: likewise for Q6_K
    return (total,), dtypes.uint8

  def _build(self, ggml_type:int, out_features:int, rows_per_warp:int=1):
    in_features, tokens = self.IN_FEATURES, 1
    group_count, chunks = in_features // 32, (in_features//32 + 31)//32
    raw_shape, raw_dtype = self._raw_shape(ggml_type, out_features, in_features)
    out = UOp.placeholder((tokens, out_features, chunks), dtypes.float32, slot=0)
    raw = UOp.placeholder(raw_shape, raw_dtype, slot=1)
    xq = UOp.placeholder((tokens, group_count, 8), dtypes.uint32, slot=2)
    xd = UOp.placeholder((tokens, group_count), dtypes.float32, slot=3)
    # T4.98k: IQ3_XXS/IQ3_S need the codebook as a 5th input, built the way nv_q8_linear does (a placeholder of the
    # right uint32 length) -- every other format leaves grid at its default None
    grid = UOp.placeholder((IQ_GRID_SIZES[ggml_type],), dtypes.uint32, slot=4) if ggml_type in IQ_GRID_SIZES else None
    return _quant_decode_kernel(out, raw, xq, xd, grid, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                                 rows_per_warp=rows_per_warp)

  def _render(self, sink:UOp) -> str:
    # T4.125d: shared render-to-source helper for the rows-per-warp tests below (same skip-on-unreachable-nvrtc
    # idiom test_renders_and_compiles_for_sm86 uses, factored out since three tests now need it)
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    from tinygrad.device import CompileError
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    try: prg = to_program(sink, renderer)
    except CompileError: raise
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    return next(u.arg for u in prg.src if u.op is Ops.SOURCE)

  def test_graph_builds_for_every_format_and_shape(self):
    for ggml_type in NV_QUANT_TYPES:
      for out_features in self.SHAPES:
        with self.subTest(ggml_type=ggml_type, out_features=out_features):
          sink = self._build(ggml_type, out_features)
          self.assertEqual(sink.op, Ops.SINK)

  def test_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    from tinygrad.device import CompileError
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    for ggml_type in NV_QUANT_TYPES:
      for out_features in self.SHAPES:
        with self.subTest(ggml_type=ggml_type, out_features=out_features):
          sink = self._build(ggml_type, out_features)
          try: prg = to_program(sink, renderer)
          except CompileError: raise
          except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn("__dp4a", src)
          self.assertIn("__shfl_xor_sync(0xffffffff", src)
          self.assertNotIn("amdgcn", src)
          if ggml_type in (IQ4_XS, IQ4_NL): self.assertIn("__byte_perm", src)
          if ggml_type in self.U16_VIEW_TYPES:
            # T4.125a: the weight buffer (slot 1 -> "data1_<n>") must be a uint16 pointer, with no byte-sized
            # load left from it -- Q6_K's int8-packing casts still render "unsigned char" elsewhere in the
            # source (unrelated register casts, not loads), so this checks the load pattern specifically
            # rather than the substring's mere presence.
            decl = re.search(r"(unsigned \w+)\*\s*data1_\d+", src)
            self.assertIsNotNone(decl, f"couldn't find the weight buffer's declared type in {ggml_type}'s source")
            self.assertEqual(cast(re.Match, decl).group(1), "unsigned short")
            self.assertNotRegex(src, r"unsigned char \w+ = \(\*\(data1_", f"{ggml_type} still loads bytes from the weight buffer")

  def test_rows_per_warp_builds_and_renders_for_every_r(self):
    # T4.125d: R in {1 (the fallback), 2, 4 (NV_GEMV_ROWS default)} must build and render for every
    # format at both model-like shapes -- the dp4a dot survives regardless of how many rows one warp now produces.
    for ggml_type in NV_QUANT_TYPES:
      for out_features in self.SHAPES:
        for rows_per_warp in (1, 2, 4):
          with self.subTest(ggml_type=ggml_type, out_features=out_features, rows_per_warp=rows_per_warp):
            sink = self._build(ggml_type, out_features, rows_per_warp=rows_per_warp)
            self.assertEqual(sink.op, Ops.SINK)
            self.assertIn("__dp4a", self._render(sink))

  def test_out_features_mod_rows_per_warp_falls_back_to_r1(self):
    # T4.125d: out_features % R != 0 must render EXACTLY R=1's kernel -- _decode_linear's fallback, not a crash
    # or a silently-wrong row count.
    out_features = self.SHAPES[0] + 1  # 12289: not a multiple of 2 or 4
    for ggml_type in NV_QUANT_TYPES:
      with self.subTest(ggml_type=ggml_type):
        r1 = self._build(ggml_type, out_features, rows_per_warp=1)
        r4 = self._build(ggml_type, out_features, rows_per_warp=4)
        self.assertEqual(self._render(r1), self._render(r4))

# T4.127a: an emulated single-warp reduce for _ffn_act_q8_kernel's PYTHON-emulator execution test below -- simpler
# than test_llm_nv_norm.py's _emulated_warp_reduce (add_rmsnorm_q8's multi-warp blocks need a `warp*32+lane` tid to
# find a shuffle partner across warps): this kernel's 32-wide AxisType.LOCAL lane range IS the whole block (no
# separate AxisType.WARP range), so the partner is just `lane`'s own xor-butterfly index into one shared buffer.
_emu_slot = itertools.count(200)
def _emulated_warp_reduce_one_warp(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  assert full_wave
  lane = next(u for u in val.toposort() if u.op is Ops.RANGE and u.arg[-1] is AxisType.LOCAL)
  for off in (16, 8, 4, 2, 1):
    buf = UOp.placeholder((32,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
    partner = lane + off - (lane // off) % 2 * (2 * off)  # lane ^ off for a power-of-two off (CHECK_OOB-friendly form)
    other = buf.after(buf[lane].store(val))[partner].load()
    val = val.maximum(other) if maximum else val + other
  return val

class TestFFNActQ8Kernel(unittest.TestCase):
  """T4.127a: _ffn_act_q8_kernel (nv_quant.py) -- the fused SiLU(gate)*up + q8-quantize decode kernel. Render+
  compile for sm_86 like TestKernelGraphAndRender above, plus real execution on tinygrad's PYTHON emulator (the
  idiom test_llm_nv_norm.py's TestKernelRunsOnPythonEmulator uses for add_rmsnorm_q8) since this kernel's layout
  -- one plain 32-wide AxisType.LOCAL lane range per (token, group), same as _q8_quantize_kernel -- is simple
  enough for _emulated_warp_reduce_one_warp above to stand in for the real __shfl_xor_sync."""
  TOKENS, HIDDEN, C = 1, 2048, 2

  @staticmethod
  def np_ffn_act_q8(gate_partials:np.ndarray, up_partials:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gate_partials/up_partials (tokens, hidden, C): act = silu(sum_c gate) * sum_c up, then _q8_pack's exact
    formula (test_llm_amd.py's q8_quantize reference: scale=max(abs)/127 clamped 1e-8, q=clip(rint(x/scale)))."""
    tokens, hidden, _ = gate_partials.shape
    gate_sum, up_sum = gate_partials.sum(-1), up_partials.sum(-1)
    act = (gate_sum / (1 + np.exp(-gate_sum))) * up_sum  # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    groups = hidden // Q8_GROUP_SIZE
    g = act.reshape(tokens, groups, Q8_GROUP_SIZE)
    scale = np.maximum(np.max(np.abs(g), axis=-1) / 127, 1e-8)
    q = np.clip(np.rint(g / scale[..., None]), -127, 127).astype(np.int8)
    return q, scale

  def _build(self, tokens:int|None=None, hidden:int|None=None, C:int|None=None) -> UOp:
    tokens, hidden, C = tokens or self.TOKENS, hidden or self.HIDDEN, self.C if C is None else C
    groups = hidden // Q8_GROUP_SIZE
    params = [UOp.placeholder((tokens, groups, 8), dtypes.uint32, slot=0), UOp.placeholder((tokens, groups), dtypes.float32, slot=1),
              UOp.placeholder((tokens, hidden, C), dtypes.float32, slot=2), UOp.placeholder((tokens, hidden, C), dtypes.float32, slot=3)]
    return _ffn_act_q8_kernel(*params, tokens=tokens, hidden_dim=hidden, C=C)

  def test_graph_builds_for_every_shape(self):
    for hidden in (self.HIDDEN, 17408):  # 17408: the standing Qwen3.8-27B FFN hidden width
      for C in (1, 2, 5):
        with self.subTest(hidden=hidden, C=C):
          self.assertEqual(self._build(hidden=hidden, C=C).op, Ops.SINK)

  def test_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    from tinygrad.device import CompileError
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    try: prg = to_program(self._build(hidden=17408, C=17), renderer)
    except CompileError: raise
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
    self.assertIn("__shfl_xor_sync(0xffffffff", src)
    self.assertNotIn("amdgcn", src)

  @unittest.skipIf(sys.version_info < (3, 12), "PYTHON emulator float16 needs 3.12+ -- unused here, kept for parity with test_llm_nv_norm.py")
  def test_runs_on_python_emulator(self):
    tokens, hidden, C = 1, 64, 3
    rng = np.random.default_rng(hidden + C)
    gate_p = rng.standard_normal((tokens, hidden, C)).astype(np.float32)
    up_p = rng.standard_normal((tokens, hidden, C)).astype(np.float32)
    with mock.patch.object(nv_quant, "warp_reduce", _emulated_warp_reduce_one_warp), \
         mock.patch.object(nv_quant, "_ffn_act_q8_kernel", nv_quant._ffn_act_q8_kernel.__wrapped__):  # bypass the cache
      q, scale = ffn_act_q8(Tensor(gate_p, device="PYTHON"), Tensor(up_p, device="PYTHON"), tokens, hidden)
      q_np, scale_np = q.numpy(), scale.numpy()
    ref_q, ref_scale = self.np_ffn_act_q8(gate_p, up_p)
    np.testing.assert_allclose(scale_np, ref_scale, rtol=1e-6)
    np.testing.assert_array_equal(q_np.view(np.int8).reshape(ref_q.shape), ref_q)

def np_q8_quantize(x:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """q8_quantize_nv's math on (tokens, in_features) fp32: int8 q (tokens, groups, 32), scale, and T4.128's group sums scale*sum(q)."""
  g = x.reshape(x.shape[0], -1, Q8_GROUP_SIZE)
  scale = np.maximum(np.abs(g).max(-1) / np.float32(127), np.float32(1e-8)).astype(np.float32)
  q = np.clip(np.rint(g / scale[..., None]), -127, 127).astype(np.int8)
  return q, scale, (scale * q.astype(np.int32).sum(-1).astype(np.float32)).astype(np.float32)

class TestQ8QuantizeSums(unittest.TestCase):
  """T4.128: q8_quantize's optional group-sums output (the IMMA Q4_1/Q5_K prefill's activation side): computed in the same kernel,
  q/scale unchanged by it, the plain kernel's graph untouched, and the variant its own cache entry."""
  def _params(self, tokens:int, in_features:int, sums:bool) -> list[UOp]:
    groups = in_features // Q8_GROUP_SIZE
    return [UOp.placeholder((tokens, groups, 8), dtypes.uint32, slot=0), UOp.placeholder((tokens, groups), dtypes.float32, slot=1),
            UOp.placeholder((tokens * in_features,), dtypes.float32, slot=2)] + \
           ([UOp.placeholder((tokens, groups), dtypes.float32, slot=3)] if sums else [])

  def test_plain_kernel_has_no_third_reduce(self):
    plain = nv_quant._q8_quantize_kernel.__wrapped__(*self._params(4, 256, False), tokens=4, in_features=256)
    with_sums = nv_quant._q8_quantize_kernel.__wrapped__(*self._params(4, 256, True), tokens=4, in_features=256)
    def shuffles(sink:UOp) -> int: return sum(1 for u in sink.toposort() if u.op is Ops.CUSTOM and "__shfl_xor_sync" in u.arg[0])
    self.assertEqual((shuffles(plain), shuffles(with_sums)), (5, 10))

  def test_variant_is_its_own_cache_entry(self):
    x = Tensor.randn(2, 64)
    q, s = q8_quantize(x, 2, 64)
    q2, s2, sums = q8_quantize(x, 2, 64, sums=True)
    self.assertIsNot(q.uop, q2.uop)
    self.assertEqual(sums.shape, (2, 2))
    self.assertIs(q8_quantize(x, 2, 64, sums=True)[2].uop, sums.uop)
    self.assertIs(q8_quantize(x, 2, 64)[0].uop, q.uop)

  @unittest.skipIf(sys.version_info < (3, 12), "PYTHON emulator float16 needs 3.12+")
  def test_runs_on_python_emulator(self):
    tokens, in_features = 3, 128
    x = (np.random.default_rng(5).standard_normal((tokens, in_features)) * 3).astype(np.float32)
    x[1, 32:64] = 0  # an all-zero group: scale clamps to 1e-8, sums 0
    with mock.patch.object(nv_quant, "warp_reduce", _emulated_warp_reduce_one_warp), \
         mock.patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__):  # bypass the cache
      plain = [t.numpy() for t in q8_quantize(Tensor(x, device="PYTHON"), tokens, in_features)]
      q, scale, sums = (t.numpy() for t in q8_quantize(Tensor(x, device="PYTHON"), tokens, in_features, sums=True))
    ref_q, ref_scale, ref_sums = np_q8_quantize(x)
    np.testing.assert_array_equal(q, plain[0])
    np.testing.assert_array_equal(scale, plain[1])
    np.testing.assert_array_equal(q.view(np.int8).reshape(ref_q.shape), ref_q)
    np.testing.assert_allclose(scale, ref_scale, rtol=1e-6)
    np.testing.assert_array_equal(sums, (scale * ref_q.astype(np.int32).sum(-1).astype(np.float32)).astype(np.float32))  # exact
    np.testing.assert_allclose(sums, ref_sums, rtol=1e-6)

class TestQ8QuantizeWarps(unittest.TestCase):
  """T4.128 (step 6): prefill-sized quantizes (tokens > 4) put NV_Q8_WARPS (token, group)s in one block -- one hardware warp each,
  the lane a WARP range -- instead of one 32-thread block per group; decode keeps its kernel byte for byte."""
  def _kernel(self, tokens:int, in_features:int, sums:bool=False) -> UOp:
    outs = q8_quantize(Tensor.empty(tokens, in_features), tokens, in_features, sums=sums)
    return next(u.src[0] for u in outs[0].uop.toposort() if u.op is Ops.CALL)

  def _program(self, sink:UOp):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.helpers import DEV
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"CUDA renderer unavailable: {e!r}")
    return to_program(sink, renderer)

  def test_decode_renders_the_one_warp_kernel(self):
    import hashlib
    for tokens, digest in ((1, "d4e80a037a736a75f08834f79d3de674"), (4, "902633179900e3a0814943c6b24430a7")):  # md5s at 0c6178834
      with self.subTest(tokens=tokens):
        prg = self._program(self._kernel(tokens, 5120))
        self.assertEqual(hashlib.md5(next(u.arg for u in prg.src if u.op is Ops.SOURCE).encode()).hexdigest(), digest)
        self.assertEqual(tuple(prg.arg.local_size), (32, 1, 1))

  def test_prefill_blocks_hold_nv_q8_warps_warps(self):
    for warps in (8, 4, 6):  # 6: 128 x 160 groups don't split into 6-warp blocks -> the last block's surplus warps redo a group
      for in_features, sums in ((5120, False), (17408, True), (6144, True)):
        with self.subTest(warps=warps, in_features=in_features, sums=sums), Context(NV_Q8_WARPS=warps):
          prg = self._program(self._kernel(128, in_features, sums))
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertEqual(tuple(prg.arg.local_size), (32, warps, 1))
          self.assertEqual(tuple(prg.arg.global_size), (-(-128 * in_features // 32 // warps), 1, 1))
          self.assertIn(f"__launch_bounds__({32 * warps})", src)
          self.assertEqual(src.count("__shfl_xor_sync"), 10 if sums else 5)

  @unittest.skipIf(sys.version_info < (3, 12), "PYTHON emulator float16 needs 3.12+")
  def test_multi_warp_runs_on_python_emulator(self):
    from test.unit.test_llm_nv_gemm import _emulated_lane_reduce  # handles the WARP lane + LOCAL warp ranges
    tokens, in_features = 5, 96  # 15 (token, group)s: 8- and 4-warp blocks leave surplus warps in the last block
    x = (np.random.default_rng(6).standard_normal((tokens, in_features)) * 3).astype(np.float32)
    x[2, :32] = 0
    ref_q, ref_scale, _ = np_q8_quantize(x)
    for warps in (8, 4, 1):
      with self.subTest(warps=warps), Context(NV_Q8_WARPS=warps), mock.patch.object(nv_quant, "warp_reduce", _emulated_lane_reduce), \
           mock.patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__):  # bypass the cache
        q, scale, sums = (t.numpy() for t in q8_quantize(Tensor(x, device="PYTHON"), tokens, in_features, sums=True))
        np.testing.assert_array_equal(q.view(np.int8).reshape(ref_q.shape), ref_q)
        np.testing.assert_allclose(scale, ref_scale, rtol=1e-6)
        np.testing.assert_array_equal(sums, (scale * ref_q.astype(np.int32).sum(-1).astype(np.float32)).astype(np.float32))

class TestQuantizeOncePerActivation(unittest.TestCase):
  """T4.123a: several Linears in one decode step read the identical activation (FFN gate+up, GDN
  attn_gate/attn_qkv, attention q/k/v) -- q8_quantize must run once per distinct INPUT UOP, shared by every
  consumer, not once per Linear. Built directly through nv_q8_linear (no NV device opened; CPU weight buffers,
  same pattern TestDispatch uses below) since Linear.__call__'s own NV gate needs real hardware -- the sharing
  logic under test lives entirely inside nv_quant.py's q8_quantize, one layer below that gate."""
  IN_FEATURES = 128

  @staticmethod
  def _q4_0_linear(out_features:int, seed:int, in_features:int=IN_FEATURES) -> Linear:
    rng = np.random.default_rng(seed)
    blocks = in_features // 32
    packed = rng.integers(0, 256, 18 * blocks * out_features, dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, in_features * out_features, Q4_0).reshape(out_features, in_features)
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    return linear

  @staticmethod
  def _quantize_calls(*outputs:Tensor) -> list[UOp]:
    sink = UOp.sink(*[o.uop for o in outputs])
    return [u for u in sink.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK
            and isinstance(u.src[0].arg, KernelInfo) and u.src[0].arg.name == "q8_quantize_nv"]

  def test_gdn_and_ffn_sharing_groups_quantize_once_each(self):
    # the three real sharing groups from the design doc, in one combined graph: ffn gate+up (one norm output),
    # gdn attn_gate+attn_qkv (one norm output), attention q+k+v (one norm output) -- 7 Linears, 3 distinct inputs
    x_ffn, x_gdn, x_attn = (Tensor.randn(1, self.IN_FEATURES) for _ in range(3))
    outs = [
      nv_q8_linear(self._q4_0_linear(256, 0), x_ffn),   # ffn_gate
      nv_q8_linear(self._q4_0_linear(256, 1), x_ffn),   # ffn_up
      nv_q8_linear(self._q4_0_linear(192, 2), x_gdn),   # attn_gate
      nv_q8_linear(self._q4_0_linear(192, 3), x_gdn),   # attn_qkv
      nv_q8_linear(self._q4_0_linear(64, 4), x_attn),   # q
      nv_q8_linear(self._q4_0_linear(64, 5), x_attn),   # k
      nv_q8_linear(self._q4_0_linear(64, 6), x_attn),   # v
    ]
    calls = self._quantize_calls(*outs)
    self.assertEqual(len(calls), 3, "one q8_quantize_nv per distinct activation, not one per Linear (7)")
    # and the gemv kernels themselves are untouched: still one linear_q4_0_nv CALL per Linear
    gemv_calls = [u for u in UOp.sink(*[o.uop for o in outs]).toposort()
                  if u.op is Ops.CALL and u.src[0].op is Ops.SINK and isinstance(u.src[0].arg, KernelInfo)
                  and u.src[0].arg.name == "linear_q4_0_nv"]
    self.assertEqual(len(gemv_calls), 7)

  def test_distinct_activations_still_quantize_separately(self):
    # same shape, different Tensors -- no over-merging just because the content/shape happens to match
    x1, x2 = Tensor.randn(1, self.IN_FEATURES), Tensor.randn(1, self.IN_FEATURES)
    outs = [nv_q8_linear(self._q4_0_linear(32, 0), x1), nv_q8_linear(self._q4_0_linear(32, 1), x2)]
    self.assertEqual(len(self._quantize_calls(*outs)), 2)

  def test_q8_quantize_returns_the_same_tensors_for_the_same_input_uop(self):
    x = Tensor.randn(1, self.IN_FEATURES)
    q1, s1 = q8_quantize(x, tokens=1, in_features=self.IN_FEATURES)
    q2, s2 = q8_quantize(x, tokens=1, in_features=self.IN_FEATURES)
    self.assertIs(q1.uop, q2.uop)
    self.assertIs(s1.uop, s2.uop)

  def test_q8_quantize_cache_is_bounded_not_unbounded(self):
    # the memo must not be a plain (unbounded) functools.cache -- a cache entry's value (q, scale) has x.uop as
    # a literal CALL src (the kernel reads x's buffer), so the entry holds a strong path back to its own key;
    # neither refcounting nor a weakref dict can ever free that on their own (verified separately: a
    # WeakKeyDictionary whose value references its own key back never collects, since the dict itself is a
    # reachable root). A finite lru_cache maxsize is what actually bounds retention across many JIT families.
    from tinygrad.llm.kernels.nv_quant import _q8_quantize_cached
    info = _q8_quantize_cached.cache_info()
    self.assertIsNotNone(info.maxsize)
    self.assertLessEqual(cast(int, info.maxsize), 64)  # sharing is between back-to-back calls; a big bound pins old graphs

class TestDispatch(unittest.TestCase):
  """With NV_CUSTOM_QUANT unset (default 0), or set on a non-NV device, Linear must never construct a custom
  kernel -- ggml_type stays discoverable but self.weight is never swapped to a packed view via the NV gate, so
  behaviour is byte-identical to before this task on every device this test can reach (CPU)."""
  def _q4_0_linear(self) -> Linear:
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, 18, dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 32, Q4_0).reshape(1, 32)
    linear = Linear(32, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    return linear

  def test_set_quantized_accepts_an_unsliced_staged_buffer(self):
    # gguf.py stages tensors over 64 MB alone: their raw bytes are the whole realized batch, not a SHRINK into it
    # (a full-extent slice is a no-op). set_quantized used to assert Ops.SHRINK and crash on output.weight, token_embd
    # and every Q8_0 ffn_gate/up/down of the 27B the moment NV_CUSTOM_QUANT=1 reached a real model (2026-09-11)
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, 18 * 4, dtype=np.uint8)
    raw = Tensor(packed).contiguous().realize()
    self.assertIsNot(raw.uop.op, Ops.SHRINK)
    decoded = ggml_data_to_tensor(raw, 128, Q4_0).reshape(4, 32)
    linear = Linear(32, 4, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    self.assertEqual(linear.ggml_type, Q4_0)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    self.assertEqual(linear.weight.shape, (18 * 4,))
    x = Tensor.randn(1, 32)
    np.testing.assert_allclose(linear(x).numpy(), (x @ decoded.T).numpy(), rtol=1e-4, atol=1e-5)  # CPU: generic path, swapped back

  def test_q3_k_and_iq3_s_disambiguated_at_the_colliding_110_byte_width(self):
    # T4.98k: both are 110-byte 256-weight blocks; IQ3_S carries the 512x4 codebook buffer in its dequant graph, Q3_K doesn't
    from tinygrad.llm.kernels.amd import Q3_K, IQ3_S, IQ3_XXS, IQ_GRID_SIZES
    rng = np.random.default_rng(3)
    for ggml_type, nbytes in ((Q3_K, 110), (IQ3_S, 110), (IQ3_XXS, 98)):
      packed = rng.integers(0, 256, nbytes * 2, dtype=np.uint8)
      for b in range(2):  # sane f16 scale (d) at the format's offset: Q3_K at 108, IQ3_* at 0
        off = b * nbytes + (108 if ggml_type == Q3_K else 0)
        packed[off:off + 2] = np.frombuffer(np.float16(0.02).tobytes(), dtype=np.uint8)
      raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
      decoded = ggml_data_to_tensor(raw, 512, ggml_type).reshape(2, 256)
      linear = Linear(256, 2, bias=False)
      nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
      linear.set_quantized(linear.weight)
      self.assertEqual(linear.ggml_type, ggml_type)
      self.assertEqual(linear.weight.dtype, dtypes.uint8)  # all three stay on the byte view
      x = Tensor.randn(1, 256)
      np.testing.assert_allclose(linear(x).numpy(), (x @ decoded.T).numpy(), rtol=1e-4, atol=1e-5)  # CPU: generic path
    self.assertEqual(IQ_GRID_SIZES, {IQ3_XXS: 256, IQ3_S: 512})

  def test_iq_grid_matches_the_reference_tables(self):
    from tinygrad.llm.kernels.nv import iq_grid
    from tinygrad.llm.kernels.amd import IQ3_XXS, IQ3_S
    from tinygrad.runtime.autogen import ggml_common as _ggml
    for ggml_type, table in ((IQ3_XXS, _ggml.iq3xxs_grid), (IQ3_S, _ggml.iq3s_grid)):
      g = iq_grid(ggml_type, "CPU")
      self.assertEqual(g.dtype, dtypes.uint32)
      self.assertEqual(g.numpy().tolist(), list(table))
      self.assertIs(iq_grid(ggml_type, "CPU"), g)  # cached per (format, device)

  def test_gate_closed_by_default(self):
    self.assertFalse(nv_quant_supported("NV"))
    self.assertEqual(NV_CUSTOM_QUANT.value, 0)

  def test_gate_closed_on_non_nv_device_even_when_set(self):
    with Context(NV_CUSTOM_QUANT=1):
      for device in ("CPU", "CPU:1", "NULL", None, ("CPU", "CPU:1")):
        self.assertFalse(nv_quant_supported(device), device)

  def test_no_custom_kernel_without_env(self):
    linear = self._q4_0_linear()
    out = linear(Tensor.randn(1, 32))
    self.assertTrue(np.isfinite(out.numpy()).all())
    # ggml_type stayed None: set_quantized never ran (no AMD hardware, NV gate closed), so self.weight is still
    # the plain lazy dequant graph and the call went through nn.Linear's generic matmul untouched
    self.assertIsNone(linear.ggml_type)
    self.assertIsNone(linear.dequant_weight)

  def test_no_custom_kernel_with_env_set_on_cpu(self):
    with Context(NV_CUSTOM_QUANT=1):
      linear = self._q4_0_linear()
      out = linear(Tensor.randn(1, 32))
      self.assertTrue(np.isfinite(out.numpy()).all())
      self.assertIsNone(linear.ggml_type)  # CPU isn't NV: nv_quant_supported stays False regardless of the env

class TestSetQuantizedIdentifiesFormat(unittest.TestCase):
  """set_quantized must tell Q8_0/Q4_K/Q4_0 apart from real GGUF-shaped packed tensors -- specifically the case
  the coordinator's hint flagged: an 8-block (256-weight) Q4_0 tensor packs to exactly 144 bytes, the same total
  size as a single Q4_K block, so byte-count alone can't disambiguate them (only fixed by this task)."""
  def _linear_for(self, ggml_type:int, n:int, row_bytes:int, out_features:int=1, seed:int=0) -> Linear:
    """row_bytes = total packed bytes for one row of n elements (not per-block)."""
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, row_bytes*out_features, dtype=np.uint8)
    if ggml_type in (Q4_K, Q5_K):  # give every block a plausible d/dmin so decode doesn't hit subnormal/inf noise
      bs = 144 if ggml_type == Q4_K else 176
      for blk in range(row_bytes*out_features // bs): packed[blk*bs:blk*bs+4] = np.array([0.01, 0.002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, n*out_features, ggml_type).reshape(out_features, n)
    linear = Linear(n, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    return linear

  def test_q4_0_not_confused_with_q4_k_at_the_colliding_byte_total(self):
    # 8 Q4_0 blocks (256 weights) pack to 8*18=144 bytes -- identical to ONE Q4_K block's 144 bytes
    linear = self._linear_for(Q4_0, 256, 144)
    linear.set_quantized(linear.weight)
    self.assertEqual(linear.ggml_type, Q4_0)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    self.assertEqual(linear.weight.nbytes(), 144)

  def test_q4_k_still_identified_at_the_same_byte_total(self):
    linear = self._linear_for(Q4_K, 256, 144)
    linear.set_quantized(linear.weight)
    self.assertEqual(linear.ggml_type, Q4_K)
    self.assertEqual(linear.weight.dtype, dtypes.uint32)
    self.assertEqual(linear.weight.nbytes(), 144)

  def test_q4_1_identified(self):
    linear = self._linear_for(Q4_1, 32, 20)
    linear.set_quantized(linear.weight)
    self.assertEqual(linear.ggml_type, Q4_1)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    self.assertEqual(linear.weight.nbytes(), 20)

  def test_iq4_nl_disambiguated_from_q4_0_at_the_colliding_block_width(self):
    # IQ4_NL's 18-byte block is byte-identical in width to Q4_0's -- BLOCK_BYTES[18] alone can't tell them apart;
    # _is_iq4_nl's toposort marker (a 16-element float32 BUFFER for the kvalues_iq4nl codebook) must. The CPU
    # generic path (no NV/AMD device open) must still compute the right numbers for both once ggml_type is set.
    q4_0 = self._linear_for(Q4_0, 32, 18)
    q4_0.set_quantized(q4_0.weight)
    self.assertEqual(q4_0.ggml_type, Q4_0)

    iq4_nl = self._linear_for(IQ4_NL, 32, 18, seed=1)
    iq4_nl.set_quantized(iq4_nl.weight)
    self.assertEqual(iq4_nl.ggml_type, IQ4_NL)

    for linear in (q4_0, iq4_nl):
      decoded, x = linear.dequant_weight, Tensor.randn(1, 32)
      np.testing.assert_allclose(linear(x).numpy(), (x @ decoded.T).numpy(), rtol=1e-4, atol=1e-5)

  def test_q8_0_identified(self):
    linear = self._linear_for(Q8_0, 32, 34)
    linear.set_quantized(linear.weight)
    self.assertEqual(linear.ggml_type, Q8_0)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    self.assertEqual(linear.weight.nbytes(), 34)

  def test_dequant_weight_cached_for_fallback(self):
    linear = self._linear_for(Q4_0, 256, 144)
    original = linear.weight
    linear.set_quantized(linear.weight)
    self.assertIs(linear.dequant_weight, original)

class TestPackedU16View(unittest.TestCase):
  """T4.125a: set_quantized also stores a uint16 view of the packed buffer for Q4_0/IQ4_NL/Q6_K (the three
  2-byte-aligned-but-not-4-byte-aligned formats nv_quant.py's decode gemv reads through it); every other
  format's packed_u16 stays None, and the view shares the byte view's buffer -- no copy."""
  def _linear_for(self, ggml_type:int, n:int, row_bytes:int) -> Linear:
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, row_bytes, dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, n, ggml_type).reshape(1, n)
    linear = Linear(n, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    return linear

  def test_u16_view_present_and_shares_the_byte_view_s_buffer(self):
    for ggml_type, n, row_bytes in ((Q4_0, 32, 18), (IQ4_NL, 32, 18), (Q6_K, 256, 210)):
      with self.subTest(ggml_type=ggml_type):
        linear = self._linear_for(ggml_type, n, row_bytes)
        self.assertIsNotNone(linear.packed_u16)
        u16 = cast(Tensor, linear.packed_u16)
        self.assertEqual(u16.dtype, dtypes.uint16)
        self.assertEqual(u16.nbytes(), row_bytes)  # same bytes, just read as half as many elements
        # no copy: both views are Buffer.view()s of the SAME base allocation
        self.assertIs(u16.uop.buf_uop.buffer.base, linear.weight.uop.buf_uop.buffer.base)

  def test_u16_view_none_for_every_other_format(self):
    for ggml_type, n, row_bytes in ((Q8_0, 32, 34), (Q4_1, 32, 20), (Q4_K, 256, 144)):
      with self.subTest(ggml_type=ggml_type):
        linear = self._linear_for(ggml_type, n, row_bytes)
        self.assertIsNone(linear.packed_u16)

  def test_u32_view_for_q4_1_only(self):
    # T4.128: Q4_1 also gets a uint32 view (the IMMA prefill's aligned loads); its self.weight stays the byte view the decode gemv reads
    for ggml_type, n, row_bytes in ((Q4_1, 64, 40), (Q4_0, 32, 18), (Q8_0, 32, 34), (Q5_K, 256, 176)):
      with self.subTest(ggml_type=ggml_type):
        linear = self._linear_for(ggml_type, n, row_bytes)
        if ggml_type != Q4_1:
          self.assertIsNone(linear.packed_u32)
          continue
        u32 = cast(Tensor, linear.packed_u32)
        self.assertEqual((u32.dtype, u32.nbytes(), linear.weight.dtype), (dtypes.uint32, row_bytes, dtypes.uint8))
        self.assertIs(u32.uop.buf_uop.buffer.base, linear.weight.uop.buf_uop.buffer.base)

class TestFFNActivationFusion(unittest.TestCase):
  """T4.123c (decode fusion design, P3): FFNBlock._feed_forward's dense branch (model.py) drops the
  `.contiguous()` between silu(gate) and the up-multiply ONLY when T==1 (concrete) and both ffn_gate/ffn_up
  dispatch through nv_q8_linear (NV custom-quant decode gemv) -- there the activation's two inputs are
  already per-kernel outputs nv_q8_linear produced, so the contiguous only forces a second launch the
  scheduler would otherwise fuse. No GPU is opened here (CPU only); the NV dispatch gate is exercised by
  monkeypatching model.nv_quant_supported, the same idiom TestDispatch above and test_llm_nv_dense.py use."""
  DIM = 32  # one Q4_0 block wide

  @staticmethod
  def _q4_0_linear(in_features:int, out_features:int, seed:int) -> Linear:
    rng = np.random.default_rng(seed)
    blocks_per_row = in_features // 32
    packed = rng.integers(0, 256, 18 * blocks_per_row * out_features, dtype=np.uint8).reshape(out_features * blocks_per_row, 18)
    packed[:, 0:2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)  # finite d (random bytes can decode to inf/nan)
    raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, in_features * out_features, Q4_0).reshape(out_features, in_features)
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    assert linear.ggml_type == Q4_0
    return linear

  def _ffn_block(self) -> FFNBlock:
    config = TransformerConfig(num_blocks=1, dim=self.DIM, hidden_dim=self.DIM, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                                vocab_size=8, head_dim=self.DIM, rope_theta=10000.0, rope_dim=self.DIM, v_head_dim=self.DIM)
    block = FFNBlock(config)
    block.ffn_gate = self._q4_0_linear(self.DIM, self.DIM, seed=0)
    block.ffn_up = self._q4_0_linear(self.DIM, self.DIM, seed=1)
    block.ffn_down = self._q4_0_linear(self.DIM, self.DIM, seed=2)
    return block

  def _old_formula(self, block:FFNBlock, x:Tensor) -> Tensor:
    # the pre-T4.123c dense branch, verbatim -- the baseline every "unchanged" assertion below compares against
    return block.ffn_down(block.ffn_gate(x).silu().contiguous() * block.ffn_up(x))

  def test_nv_decode_drops_to_one_activation_kernel(self):
    block, x = self._ffn_block(), Tensor.randn(1, 1, self.DIM)
    with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      # pre-realize x (to_prerealize) so its own rand-generation kernels don't pollute the FFN kernel count
      check_schedule(self._old_formula(block, x), 3, to_prerealize=[x])  # silu+contiguous, up-multiply, down sum
      check_schedule(block._feed_forward(x), 2, to_prerealize=[x])       # one fewer: silu*up fused into the next kernel

  def test_nv_decode_numerics_match_the_old_formula(self):
    block, x = self._ffn_block(), Tensor.randn(1, 1, self.DIM)
    with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      fused, old = block._feed_forward(x), self._old_formula(block, x)
    np.testing.assert_allclose(fused.numpy(), old.numpy(), rtol=1e-5, atol=1e-6)

  def test_generic_graph_and_numerics_are_byte_identical(self):
    # no NV mock: nv_quant_supported("CPU"/whatever this process's device is) is really False, so _feed_forward
    # must take the exact same contiguous path as before this task, for every T shape it can see (decode T==1,
    # a concrete prefill-sized T, and a symbolic chunk-width T) -- UOp hash-consing makes `is` a structural proof.
    block = self._ffn_block()
    for x in (Tensor.randn(1, 1, self.DIM), Tensor.randn(1, 7, self.DIM),
              Tensor.randn(1, 8, self.DIM)[:, :UOp.variable("toks", 1, 8).bind(1), :]):
      with self.subTest(shape=x.shape):
        self.assertIs(block._feed_forward(x).uop, self._old_formula(block, x).uop)
    x = Tensor.randn(1, 1, self.DIM)
    np.testing.assert_allclose(block._feed_forward(x).numpy(), self._old_formula(block, x).numpy(), rtol=1e-5, atol=1e-6)

  def test_mixed_ggml_type_keeps_the_contiguous_path(self):
    # the condition requires BOTH gate and up to be NV_QUANT_TYPES formats -- a plain (unquantized) gate must
    # not trip the fast path even with the NV gate open and up on a covered format
    block = self._ffn_block()
    plain_gate = Linear(self.DIM, self.DIM, bias=False)
    nn.state.load_state_dict(plain_gate, {"weight": Tensor.randn(self.DIM, self.DIM)}, verbose=False, realize=False)
    block.ffn_gate = plain_gate
    self.assertIsNone(block.ffn_gate.ggml_type)
    x = Tensor.randn(1, 1, self.DIM)
    with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      self.assertIs(block._feed_forward(x).uop, self._old_formula(block, x).uop)

  def test_t_greater_than_one_keeps_the_contiguous_path_even_with_the_nv_gate_open(self):
    # T4.123c is decode-only (T==1 concrete) -- nv_q8_linear itself covers tokens<=4, but this fusion must not
    # take the fast path for a 1<T<=4 concrete call or a symbolic chunk bound at 1 (neither is T==1 concrete)
    block = self._ffn_block()
    with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      x4 = Tensor.randn(1, 4, self.DIM)
      self.assertIs(block._feed_forward(x4).uop, self._old_formula(block, x4).uop)
      xsym = Tensor.randn(1, 8, self.DIM)[:, :UOp.variable("toks", 1, 8).bind(1), :]
      self.assertIs(block._feed_forward(xsym).uop, self._old_formula(block, xsym).uop)

def _kernel_names_and_reduces(t:Tensor) -> tuple[list[str], int]:
  """Every custom-kernel CALL name in t's unscheduled graph, plus a count of plain Ops.REDUCE nodes (nn.RMSNorm's
  .mean(-1) is the only thing in this file's graphs that produces one -- none of nv_quant.py's/nv.py's own
  kernels use Ops.REDUCE, they all reduce via warp_reduce's shuffle tree -- so this is a clean proxy for "a
  generic RMSNorm sum-of-squares survived"). check_schedule can't be used here: add_rmsnorm_q8's kernel uses
  AxisType.LOCAL/WARP + AddrSpace.LOCAL, which .linear_with_vars() (and so check_schedule) can't push through
  the scheduler's early rewrite passes on CPU/NULL at all (see test_llm_nv_norm.py's module docstring) -- this
  reads the raw, unscheduled UOp graph directly instead (same idiom TestQuantizeOncePerActivation._quantize_calls
  uses), which is exactly what nv_q8_linear/add_rmsnorm_q8's custom kernels already are (pre-built Ops.CALL nodes,
  not something the scheduler groups later)."""
  nodes = list(UOp.sink(t.uop).toposort())
  names = [u.src[0].arg.name for u in nodes if u.op is Ops.CALL and u.src[0].op is Ops.SINK and isinstance(u.src[0].arg, KernelInfo)]
  return names, sum(1 for u in nodes if u.op is Ops.REDUCE)

class TestFFNNormSiteFusion(unittest.TestCase):
  """T4.123e (decode fusion design, P4 wiring half): FFNBlock._ffn_output (model.py), used by both FFNBlock/
  TransformerBlock's shared __call__ and GatedDeltaNetBlock's own one, fuses the ffn_norm site's residual-add +
  RMSNorm + quantize into one add_rmsnorm_q8 kernel on the NV decode path (concrete T==1, dense FFN, both
  ffn_gate/ffn_up already NV-quantized -- the same gate T4.123c's SiLU fusion uses). No GPU is opened here
  (CPU/NULL only): the NV dispatch gate is exercised by monkeypatching nv_quant_supported at BOTH call sites
  that check it -- model.py's own (controls _ffn_output's and _feed_forward's T4.123c gates) and
  kernels/nv_quant.py's (amd.py's Linear.__call__ reads THIS one to decide whether to actually dispatch through
  nv_q8_linear at all) -- mocking only the first one (TestFFNActivationFusion's own idiom) leaves Linear.__call__
  on the generic dequant+matmul path, which never quantizes anything and so can't exercise this fusion."""
  DIM = 256  # must be a multiple of 256 (add_rmsnorm_q8's default threads) -- _ffn_output's own guard

  @staticmethod
  def _q4_0_linear(dim:int, seed:int) -> Linear:
    rng = np.random.default_rng(seed)
    blocks = dim // 32
    packed = rng.integers(0, 256, 18 * blocks * dim, dtype=np.uint8).reshape(dim * blocks, 18)
    packed[:, 0:2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)  # finite d
    raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, dim * dim, Q4_0).reshape(dim, dim)
    linear = Linear(dim, dim, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    assert linear.ggml_type == Q4_0
    return linear

  def _ffn_block(self) -> FFNBlock:
    config = TransformerConfig(num_blocks=1, dim=self.DIM, hidden_dim=self.DIM, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                                vocab_size=8, head_dim=self.DIM, rope_theta=10000.0, rope_dim=self.DIM, v_head_dim=self.DIM)
    block = FFNBlock(config)
    block.ffn_gate, block.ffn_up, block.ffn_down = (self._q4_0_linear(self.DIM, s) for s in (0, 1, 2))
    with mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True): block.prepare_nv_norms()  # T4.125b: from_gguf's load-time step
    return block

  @staticmethod
  def _old_formula(block:FFNBlock, x:Tensor, attn_out:Tensor) -> Tensor:
    # the pre-T4.123e two-step form, verbatim -- every "fused drops exactly this" / "unchanged" claim below is
    # against this baseline
    h = x + attn_out
    return h + block._feed_forward(block.ffn_norm(h))

  def _inputs(self, T:int=1) -> tuple[Tensor, Tensor]:
    with Context(DEBUG=0):  # pre-realize: keep the randn RNG graph out of the kernel-count toposorts below
      return Tensor.randn(1, T, self.DIM).contiguous().realize(), Tensor.randn(1, T, self.DIM).contiguous().realize()

  def test_fused_drops_the_reduce_and_shares_one_fewer_quantize(self):
    block = self._ffn_block()
    x, attn_out = self._inputs()
    with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True), \
         mock.patch("tinygrad.llm.kernels.nv_quant.nv_quant_supported", return_value=True):
      old_names, old_reduces = _kernel_names_and_reduces(self._old_formula(block, x, attn_out))
      new_names, new_reduces = _kernel_names_and_reduces(block._ffn_output(x, attn_out))
    # old: ffn_norm's RMSNorm reduce, one shared q8_quantize_nv for gate+up (T4.123a); ffn_down's own quantize is
    # now fused into ffn_act_q8_nv in BOTH formulas (T4.127a lives in _feed_forward, shared code), 3 gemvs
    self.assertEqual(old_reduces, 1)
    self.assertEqual(old_names.count("q8_quantize_nv"), 1)
    self.assertEqual(old_names.count("ffn_act_q8_nv"), 1)
    self.assertEqual(old_names.count("add_rmsnorm_q8_nv"), 0)
    self.assertEqual(old_names.count("linear_q4_0_nv"), 3)
    # new: the reduce and the gate+up quantize are both gone, replaced by ONE add_rmsnorm_q8_nv; down's own
    # quantize is T4.127a's fused kernel, same as the old formula above (an orthogonal fusion, same effect here)
    self.assertEqual(new_reduces, 0)
    self.assertEqual(new_names.count("q8_quantize_nv"), 0)
    self.assertEqual(new_names.count("add_rmsnorm_q8_nv"), 1)
    self.assertEqual(new_names.count("ffn_act_q8_nv"), 1)
    self.assertEqual(new_names.count("linear_q4_0_nv"), 3)

  def test_fused_wiring_matches_the_old_formula_within_float_tolerance(self):
    # add_rmsnorm_q8's own kernel can't execute on CPU (AxisType.LOCAL/WARP -- test_llm_nv_norm.py's module
    # docstring), so this checks the WIRING: _ffn_output passes (residual=x, partials=attn_out reshaped to one
    # C=1 "split", weight=ffn_norm.weight, eps=ffn_norm.eps) into add_rmsnorm_q8 -- np_add_rmsnorm_q8 (already
    # proven in test_llm_nv_norm.py to match the kernel's own math bit-for-bit modulo reduction order) applied to
    # those EXACT inputs must agree with the real tinygrad-executed old formula's x+attn_out -> ffn_norm, which
    # needs no NV kernel and so can genuinely run here. Tolerance 1e-5 rtol / 1e-6 atol: the same bound
    # test_llm_nv_norm.py's own numpy-vs-kernel-formula tests use, sized for fp64-internal-mean-vs-fp32 rounding
    # (measured here: 2.4e-7 max abs diff on the normed value, 0 on the residual sum).
    block = self._ffn_block()
    x, attn_out = self._inputs()
    h_real = x + attn_out
    normed_real = block.ffn_norm(h_real)
    new_residual_np, _, _, normed_np = np_add_rmsnorm_q8(
      x.reshape(1, self.DIM).numpy(), attn_out.reshape(1, self.DIM, 1).numpy(), block.ffn_norm.weight.numpy(), block.config.norm_eps)
    np.testing.assert_allclose(new_residual_np.reshape(h_real.shape), h_real.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(normed_np.reshape(normed_real.shape), normed_real.numpy(), rtol=1e-5, atol=1e-6)

  def test_no_gate_is_byte_identical(self):
    # NV_CUSTOM_QUANT unset (default 0): nv_quant_supported is really False everywhere, no mocking at all --
    # _ffn_output must take the exact pre-T4.123e two-step path
    block = self._ffn_block()
    x, attn_out = self._inputs()
    self.assertIs(block._ffn_output(x, attn_out).uop, self._old_formula(block, x, attn_out).uop)

  def test_t_greater_than_one_is_byte_identical_even_with_the_gate_open(self):
    # decode-only (T==1 concrete); mocking only model.py's own nv_quant_supported (not kernels/nv_quant.py's)
    # keeps Linear.__call__ on the generic path, so `is` identity is meaningful here (a real nv_q8_linear
    # dispatch allocates a fresh output buffer every call, which breaks hash-cons `is` regardless of this task --
    # TestFFNActivationFusion's own T>1 test uses the same one-sided mock for the same reason)
    block = self._ffn_block()
    x4, attn_out4 = self._inputs(T=4)
    with mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      self.assertIs(block._ffn_output(x4, attn_out4).uop, self._old_formula(block, x4, attn_out4).uop)

  def test_moe_and_tp_blocks_keep_the_old_path(self):
    # the dense-only guard (hasattr ffn_gate_exps/ffn_gate_tp) must close even with the NV gate open and T==1 --
    # block.ffn_gate_exps's mere presence (no full MoE setup behind it) is enough to prove the guard closed:
    # both _ffn_output and the old formula fall into FFNBlock._feed_forward's (untouched) MoE branch and fail
    # identically past it (ffn_gate_inp missing) -- had _ffn_output's fused branch been taken instead, it would
    # have failed earlier, inside add_rmsnorm_q8's own shape asserts, not inside _feed_forward's MoE code.
    block = self._ffn_block()
    block.ffn_gate_exps = object()  # presence alone is enough to trip the guard; never actually reached
    x, attn_out = self._inputs()
    with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      with self.assertRaises(AttributeError) as new_err: block._ffn_output(x, attn_out)
      with self.assertRaises(AttributeError) as old_err: self._old_formula(block, x, attn_out)
    self.assertEqual(str(new_err.exception), str(old_err.exception))
    self.assertIn("ffn_gate_inp", str(new_err.exception))

class TestGDNAttnNormSiteFusion(unittest.TestCase):
  """T4.123e: GatedDeltaNetBlock._normed_for_attention (model.py) fuses attn_norm's RMSNorm into add_rmsnorm_q8
  on the NV decode path, feeding _attention its want_fp16 output directly -- _attention's own `x = x.half()`
  (its first operation in every branch, fast nv_gdn_decode path and the generic per-token loop alike) becomes a
  true no-op on an already-float16 Tensor (tinygrad/mixin/dtype.py: cast returns self when the dtype already
  matches), so _attention itself needs no change at all (this task's GDN restriction)."""
  DIM = 256

  def _gdn_block(self) -> GatedDeltaNetBlock:
    config = TransformerConfig(num_blocks=1, dim=self.DIM, hidden_dim=self.DIM, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                                vocab_size=8, head_dim=self.DIM, rope_theta=10000.0, rope_dim=self.DIM, v_head_dim=self.DIM)
    ssm = SSMConfig(conv_kernel=4, state_size=16, group_count=1, time_step_rank=2, inner_size=32)
    block = GatedDeltaNetBlock(config, ssm)
    with mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True): block.prepare_nv_norms()  # T4.125b: from_gguf's load-time step
    return block

  def test_fused_output_is_fp16_with_no_reduce_and_half_is_a_true_noop(self):
    block = self._gdn_block()
    with Context(DEBUG=0): x = Tensor.randn(1, 1, self.DIM).contiguous().realize()
    with mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      fused = block._normed_for_attention(x)
      names, reduces = _kernel_names_and_reduces(fused)
    self.assertEqual(reduces, 0)
    self.assertEqual(names.count("add_rmsnorm_q8_nv"), 1)
    self.assertEqual(fused.dtype, dtypes.float16)
    self.assertIs(fused.half().uop, fused.uop, "_attention's own x.half() must be a true no-op on this")

  def test_unfused_has_a_reduce_and_is_float32(self):
    block = self._gdn_block()
    with Context(DEBUG=0): x = Tensor.randn(1, 1, self.DIM).contiguous().realize()
    unfused = block.attn_norm(x)
    _, reduces = _kernel_names_and_reduces(unfused)
    self.assertEqual(reduces, 1)
    self.assertEqual(unfused.dtype, dtypes.float32)

  def test_fused_wiring_matches_the_old_formula_within_float_tolerance(self):
    # same reasoning as TestFFNNormSiteFusion's numerics test: add_rmsnorm_q8 can't execute on CPU, so this
    # checks that the inputs _normed_for_attention wires in (residual=x, no partials, weight=attn_norm.weight,
    # want_fp16=True) reproduce the old attn_norm(x).half() value through np_add_rmsnorm_q8's reference formula.
    block = self._gdn_block()
    with Context(DEBUG=0): x = Tensor.randn(1, 1, self.DIM).contiguous().realize()
    old_fp16 = block.attn_norm(x).half().numpy()
    _, _, _, normed_np = np_add_rmsnorm_q8(x.reshape(1, self.DIM).numpy(), None, block.attn_norm.weight.numpy(), block.config.norm_eps)
    new_fp16 = normed_np.astype(np.float16).astype(np.float32).reshape(old_fp16.shape)
    np.testing.assert_allclose(new_fp16, old_fp16, rtol=1e-3, atol=1e-3)  # fp16 rounding, not fp32 -- a coarser bound

  def test_no_gate_is_byte_identical(self):
    block = self._gdn_block()
    with Context(DEBUG=0): x = Tensor.randn(1, 1, self.DIM).contiguous().realize()
    self.assertIs(block._normed_for_attention(x).uop, block.attn_norm(x).uop)

  def test_t_greater_than_one_is_byte_identical_even_with_the_gate_open(self):
    block = self._gdn_block()
    with Context(DEBUG=0): x4 = Tensor.randn(1, 4, self.DIM).contiguous().realize()
    with mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True):
      self.assertIs(block._normed_for_attention(x4).uop, block.attn_norm(x4).uop)

def _decode_census(t:Tensor) -> list[str]:
  """T4.125b: every kernel the scheduler launches to realize t (really scheduled via linear_with_vars, never executed): a custom
  kernel by its own name, a generic one as 'reduce|elementwise <out dtype>[numel] <- <in dtypes>[numel],...'."""
  lin, _ = t.linear_with_vars()
  census = []
  for call in (u for u in lin.src if u.op is Ops.CALL):
    ast = call.src[0]
    if isinstance(ast.arg, KernelInfo) and ast.arg.opts_to_apply == ():
      census.append(ast.arg.name)
      continue
    bufs = [f"{b.dtype.name}[{b.max_numel()}]" for b in call.src[1:] if b.dtype.name != "int"]  # skip bound Variables
    census.append(f"{'reduce' if any(u.op is Ops.REDUCE for u in ast.toposort()) else 'elementwise'} {bufs[0]} <- {','.join(bufs[1:])}")
  return census

@contextlib.contextmanager
def _nv_gates_open():
  # every NV gate the decode fast paths check, opened on CPU (graphs are built and scheduled, never run)
  with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True), \
       mock.patch("tinygrad.llm.kernels.nv_quant.nv_quant_supported", return_value=True), \
       mock.patch("tinygrad.llm.model.nv_decode_kernel_supported", return_value=True), \
       mock.patch("tinygrad.llm.model.nv_decode_attention_ok", return_value=True):
    yield

class TestDecodeStepCensus(unittest.TestCase):
  """T4.125b: the kernels of ONE block's decode step (attn_norm site -> attention/GDN -> output projection -> ffn_norm site -> FFN ->
  block output) on a tiny mocked-NV config, scheduled on CPU -- a GatedDeltaNetBlock and a TransformerBlock (qwen35's two kinds).
  Widths are chosen so the split-K gemvs really split (ssm_out/attn_output and ffn_down read 2048 inputs -> C=2 partials, as the
  27B's 6144/17408 give C=6/17) and the norm weights are lazy GGUF views like from_gguf's (F32 bytes, then .cast(float16)) -- the
  two things the 10-28 card profile showed costing kernels."""
  DIM, WIDE = 256, 2048

  @staticmethod
  def _q4_0(in_features:int, out_features:int, seed:int) -> Linear:
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, 18 * (in_features // 32) * out_features, dtype=np.uint8).reshape(-1, 18)
    packed[:, 0:2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)  # finite d
    raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
    linear = Linear(in_features, out_features, bias=False)
    decoded = ggml_data_to_tensor(raw, in_features * out_features, Q4_0).reshape(out_features, in_features)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    return linear

  @staticmethod
  def _gguf_f32(n:int, seed:int) -> Tensor:
    w = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return ggml_data_to_tensor(Tensor(w.view(np.uint8)).realize(), n, 0).cast(dtypes.float16)

  def _step(self, b:FFNBlock) -> list[str]:
    from tinygrad.llm.kernels.nv_dense import prepare_dense_weights
    D, W = self.DIM, self.WIDE
    b.ffn_gate, b.ffn_up, b.ffn_down = self._q4_0(D, W, 3), self._q4_0(D, W, 4), self._q4_0(W, D, 5)
    b.attn_norm.weight, b.ffn_norm.weight = self._gguf_f32(D, 10), self._gguf_f32(D, 11)
    with _nv_gates_open():
      prepare_dense_weights(b)  # from_gguf's load-time steps, in its order
      b.prepare_nv_norms()
      if isinstance(b, GatedDeltaNetBlock): b.prepare_nv_decode()
      with Context(DEBUG=0):
        x = Tensor.randn(1, 1, D).contiguous().realize()
        b._init_state(x)
        Tensor.realize(*[t for t in (getattr(b, n, None) for n in ("conv_state", "recurrent_state", "cache_kv", "freqs_cis")) if t is not None])
      # the blocks' _run body (spec=False), without the @function wrapper
      census = _decode_census(b._ffn_output(x, b._attention(b._normed_for_attention(x), 5)).contiguous())
    print(f"\n{type(b).__name__} decode step: {len(census)} kernels", sorted(census))
    self.assertEqual(census.count("add_rmsnorm_q8_nv"), 2, "one kernel per fused norm site (attn_norm, ffn_norm)")
    self.assertEqual(census.count("ffn_act_q8_nv"), 1, "T4.127a: ffn_down's input (SiLU(gate)*up + its quantize) is one kernel")
    conversions = [k for k in census if (ins := k.partition(" <- ")[2]).startswith("unsigned char[") and "," not in ins]
    self.assertEqual(conversions, [], "no per-step norm-weight conversion (a kernel reading only a GGUF byte view)")
    self.assertNotIn(f"reduce float[{D}] <- float[{2*D}]", census, "the projection's partials are summed inside the ffn_norm kernel")
    return census

  def test_gdn_decode_step(self):
    D, W = self.DIM, self.WIDE
    config = TransformerConfig(num_blocks=1, dim=D, hidden_dim=W, n_heads=1, n_kv_heads=1, norm_eps=1e-5, vocab_size=8, head_dim=D,
                               rope_theta=10000.0, rope_dim=D, v_head_dim=D, max_context=64)
    b = GatedDeltaNetBlock(config, SSMConfig(conv_kernel=4, state_size=32, group_count=1, time_step_rank=8, inner_size=W))
    b.attn_qkv, b.attn_gate, b.ssm_out = self._q4_0(D, b.conv_channels, 0), self._q4_0(D, W, 1), self._q4_0(W, D, 2)
    for i, lin in enumerate((b.ssm_beta, b.ssm_alpha)):
      nn.state.load_state_dict(lin, {"weight": self._gguf_f32(8 * D, 6 + i).reshape(8, D)}, verbose=False, realize=False)
    b.ssm_norm.weight = self._gguf_f32(W // 8, 12)
    census = self._step(b)
    self.assertNotIn(f"elementwise half[{D}] <- half[{D}]", census, "the f16 gemvs read the fused fp16 output in place")
    # T4.127a: ffn_down's own quantize is now ffn_act_q8_nv's -- only ssm_out's q8_quantize_nv remains
    self.assertEqual(census.count("q8_quantize_nv"), 1, "attn_qkv/attn_gate's is the fused kernel's; ffn_down's is ffn_act_q8_nv's")
    # T4.127b: ssm_beta(x)/ssm_alpha(x) -- two f16 gemvs -- are now the one combined gemv (dim 256 is 128-aligned)
    self.assertEqual(census.count("linear_f16_gemv_nv"), 1, "ssm_beta/ssm_alpha's two gemvs are now the combined one")

  def test_attention_decode_step(self):
    D, W = self.DIM, self.WIDE
    config = TransformerConfig(num_blocks=1, dim=D, hidden_dim=W, n_heads=16, n_kv_heads=1, norm_eps=1e-5, vocab_size=8, head_dim=128,
                               rope_theta=10000.0, rope_dim=64, v_head_dim=128, max_context=128, qk_norm=128, attn_output_gate=True)
    b = TransformerBlock(config)
    b.attn_q, b.attn_k, b.attn_v, b.attn_output = self._q4_0(D, 4096, 0), self._q4_0(D, 128, 1), self._q4_0(D, 128, 2), self._q4_0(W, D, 6)
    b.attn_q_norm.weight, b.attn_k_norm.weight = self._gguf_f32(128, 12), self._gguf_f32(128, 13)
    census = self._step(b)
    self.assertNotIn(f"reduce float[1] <- float[{D}]", census, "attn_norm's sum of squares runs inside add_rmsnorm_q8")
    # T4.127a: ffn_down's own quantize is now ffn_act_q8_nv's -- only attn_output's q8_quantize_nv remains
    self.assertEqual(census.count("q8_quantize_nv"), 1, "q/k/v's quantize is the fused kernel's; ffn_down's is ffn_act_q8_nv's; left: attn_output's")

  def test_output_norm_feeds_the_head_directly(self):
    head, norm = self._q4_0(self.DIM, 64, 7), nn.RMSNorm(self.DIM, 1e-5)
    norm.weight = self._gguf_f32(self.DIM, 14)
    with _nv_gates_open():
      weight = norm.weight.float().contiguous().realize()  # from_gguf's nv_output_norm_weight
      with Context(DEBUG=0): x = Tensor.randn(1, 1, self.DIM).contiguous().realize()
      self.assertEqual(len(_decode_census(head(norm(x)))), 4, "sum of squares, apply, q8_quantize_nv, the gemv")
      self.assertEqual(sorted(_decode_census(head(nv_fused_norm(norm, weight, x, head)))), ["add_rmsnorm_q8_nv", "linear_q4_0_nv"])

  def test_nv_fused_norm_off_the_decode_path_is_plain_norm(self):
    head, norm = self._q4_0(self.DIM, 64, 8), nn.RMSNorm(self.DIM, 1e-5)
    weight = norm.weight.float().contiguous().realize()
    x1, x4 = Tensor.randn(1, 1, self.DIM), Tensor.randn(1, 4, self.DIM)
    self.assertIs(nv_fused_norm(norm, weight, x1, head).uop, norm(x1).uop, "NV gate closed")
    plain = Linear(self.DIM, 64, bias=False)
    with _nv_gates_open():
      self.assertIs(nv_fused_norm(norm, weight, x4, head).uop, norm(x4).uop, "T > 1")
      self.assertIs(nv_fused_norm(norm, None, x1, head).uop, norm(x1).uop, "weight not prepared at load")
      self.assertIs(nv_fused_norm(norm, weight, x1, plain).uop, norm(x1).uop, "a consumer that isn't NV-quantized")

class TestSplitKPartials(unittest.TestCase):
  """T4.125b: nv_q8_linear's summed result maps back to its kernel's un-summed split-K partials (split_k_partials), which
  _ffn_output hands to add_rmsnorm_q8 so the summing reduce is never scheduled."""
  def test_partials_are_exactly_what_the_result_sums(self):
    linear = TestDecodeStepCensus._q4_0(2048, 64, 0)
    out = nv_q8_linear(linear, Tensor.randn(1, 1, 2048))
    assert out is not None
    partials = split_k_partials(out)
    assert partials is not None
    self.assertEqual(partials.shape, (1, 64, 2))  # 2048 inputs = two 1024-wide chunks
    self.assertIs(partials.sum(-1).reshape(out.shape).uop, out.uop, "hash-consing: the result IS the sum of these partials")

  def test_anything_else_has_none(self):
    self.assertIsNone(split_k_partials(Tensor.randn(1, 1, 64)))
    linear = TestDecodeStepCensus._q4_0(2048, 64, 1)
    linear.bias = Tensor.randn(64)
    out = nv_q8_linear(linear, Tensor.randn(1, 1, 2048))
    assert out is not None
    self.assertIsNone(split_k_partials(out), "a bias is added after the sum: those partials don't sum to the result")

if __name__ == "__main__": unittest.main()
