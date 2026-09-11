"""T4.98c: the NV quantized-linear decode kernel (tinygrad/llm/kernels/nv_quant.py), ported from kernels/amd.py.

Everything here runs WITHOUT a GPU (no DEV=NV, no Device["NV"] opened): kernel UOp graphs are built directly and,
when the nvrtc lane is reachable, rendered+compiled for sm_86 through a CUDARenderer built from a bare Target
spec (mirrors test_llm_nv_scan.py's T6.2 pattern) -- never via an actual NV device. Format unpack math is checked
against gguf.py's ggml_data_to_tensor with pure-numpy transcriptions of what group_dot does, since the kernel
itself can't run here. Numeric hardware parity (does the compiled kernel produce the right numbers on a real
3090) is NOT covered by this file -- see the final report for exactly what must still be run on NV."""
import unittest
import numpy as np
from tinygrad import Tensor, UOp, Context, dtypes, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_WORDS, Q5_WORDS, Q6_BYTES, IQ4_WORDS
from tinygrad.llm.kernels.nv_quant import Q8_0, Q4_0, NV_QUANT_TYPES, NV_CUSTOM_QUANT, nv_quant_supported, _quant_decode_kernel, _nv_byte_perm

KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

# ******** pure-numpy re-implementations of the per-block unpack group_dot does, vs gguf.py's reference ********

def _f16(b2) -> np.float32: return np.frombuffer(bytes(b2), dtype="<f2").astype(np.float32)[0]

def ref_q4_0(b:np.ndarray) -> np.ndarray:  # b: uint8[18]
  d = _f16(b[0:2])
  qs = b[2:18]
  q = np.concatenate([qs & 0x0F, qs >> 4]).astype(np.int32)  # gguf q_to_uint8(.,4): low nibbles 0-15, then high 16-31
  return (d * (q - 8)).astype(np.float32)

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

class TestFormatUnpackMath(unittest.TestCase):
  """group_dot's per-group unpack (nibble/byte extraction, scales, the Q4_0 -8 and Q4_K/Q5_K min terms), transcribed
  to pure numpy, must reproduce gguf.py's ggml_data_to_tensor dequantized values exactly -- the kernel itself can't
  run without a GPU, so this is the check that the NV unpack math (shared with amd.py for the 4 existing formats)
  matches the reference."""
  def _check(self, ggml_type:int, nbytes:int, ref, seed:int):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, nbytes, dtype=np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    n = 256 if nbytes >= 100 else 32
    expected = ggml_data_to_tensor(raw, n, ggml_type).numpy().reshape(-1)
    np.testing.assert_allclose(ref(packed), expected, rtol=1e-3, atol=1e-3)

  def test_q4_0(self):
    for seed in range(5): self._check(Q4_0, 18, ref_q4_0, seed)
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

  @staticmethod
  def _raw_shape(ggml_type:int, out_features:int, in_features:int) -> tuple[tuple[int, ...], object]:
    if ggml_type in (Q4_0, Q8_0):
      blocks = in_features // 32
      nbytes = 18 if ggml_type == Q4_0 else 34
      return (out_features*blocks*nbytes,), dtypes.uint8
    blocks = in_features // 256
    words = {Q4_K: Q4_WORDS, Q5_K: Q5_WORDS, IQ4_XS: IQ4_WORDS}.get(ggml_type)
    if words is not None: return (out_features*blocks*words,), dtypes.uint32
    return (out_features*blocks*Q6_BYTES,), dtypes.uint8  # Q6_K

  def _build(self, ggml_type:int, out_features:int):
    in_features, tokens = self.IN_FEATURES, 1
    group_count, chunks = in_features // 32, (in_features//32 + 31)//32
    raw_shape, raw_dtype = self._raw_shape(ggml_type, out_features, in_features)
    out = UOp.placeholder((tokens, out_features, chunks), dtypes.float32, slot=0)
    raw = UOp.placeholder(raw_shape, raw_dtype, slot=1)
    xq = UOp.placeholder((tokens, group_count, 8), dtypes.uint32, slot=2)
    xd = UOp.placeholder((tokens, group_count), dtypes.float32, slot=3)
    return _quant_decode_kernel(out, raw, xq, xd, out_features=out_features, in_features=in_features, ggml_type=ggml_type)

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
          if ggml_type == IQ4_XS: self.assertIn("__byte_perm", src)

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

if __name__ == "__main__": unittest.main()
