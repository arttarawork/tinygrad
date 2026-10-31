"""T4.137: attn_prep_nv (kernels/nv_attn_prep.py) -- an attention block's decode-step glue (split-K sums, per-head q/k RMSNorm, rope, the
output-gate split, the KV_INT4 store) as one kernel. The kernel EXECUTED on the PYTHON emulator (warp shuffle swapped for a shared
exchange, test_llm_nv_attn's _emulated_shfl) through attn_prep against (a) a numpy fp32 transcription of the kernel's own arithmetic order
-- bit-exact -- and (b) fp64 math; TransformerBlock._attention on the emulator with the kernel vs the generic graph; the sm_86 render; the
decode-step census (the 9 generic glue kernels -> 1) and the routing gates. No GPU, no nvrtc."""
import pickle, sys, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, dtypes, Context
from tinygrad.uop.ops import Ops, KernelInfo, python_alu
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.llm.kernels import nv_attn_prep, nv_quant, nv_gemm
from tinygrad.llm.kernels.amd import _unbind
from test.unit.test_llm_nv_attn import _emulated_shfl, _lower_sm86

def _f32(x): return np.asarray(x, dtype=np.float32)
_UNCACHED = nv_attn_prep._attn_prep_kernel.__wrapped__  # the builder without its cache: the emulated shuffle is patched in per test

def _tg_round(x:np.ndarray) -> np.ndarray:  # the UOp mixin's round (half to even), op for op in fp32
  b = _f32(np.trunc(x) / np.float32(2))
  return np.where((x > 0) == (np.trunc(b) == b), np.ceil(_f32(x - np.float32(0.5))), np.floor(_f32(x + np.float32(0.5)))).astype(np.float32)

def np_attn_prep(qp, kp, vp, qn, kn, freqs, sp:int, rp:int, B:int, T:int, H:int, KVH:int, D:int, eps:float, exact:bool=True):
  """numpy attn_prep. exact: fp32 in the kernel's order (left-to-right partial sums, per-lane squares + xor butterfly, x*(1/s)), else fp64
  math. Returns (q (B,H,T,D), gate (B*T,H*D), packed (2,B,KVH,T,D/2) uint8, scale (2,B,KVH,T,D/32) fp16)."""
  ft = np.float32 if exact else np.float64
  def lsum(p):
    acc = p[..., 0].astype(ft)
    for c in range(1, p.shape[-1]): acc = (acc + p[..., c]).astype(ft)
    return acc
  q_all = lsum(qp).reshape(B, T, H, 2, D)
  q, gate = q_all[:, :, :, 0], q_all[:, :, :, 1]
  k, v = lsum(kp).reshape(B, T, KVH, D), lsum(vp).reshape(B, T, KVH, D)
  PER, half = D // 32, freqs.shape[1] // 2
  def norm(x, w):
    if exact:
      ss = (x * x).astype(ft).reshape(*x.shape[:-1], 32, PER)
      lanes = ss[..., 0]
      for j in range(1, PER): lanes = (lanes + ss[..., j]).astype(ft)
      for off in (16, 8, 4, 2, 1): lanes = (lanes + lanes[..., np.arange(32) ^ off]).astype(ft)
      inv = (ft(1) / np.sqrt((lanes[..., :1] * ft(1 / D) + ft(eps)).astype(ft))).astype(ft)
      return ((x * inv).astype(ft) * w).astype(ft)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
  def rope(x, rows):  # x (B,T,heads,D), rows (T,)
    cos, sin = freqs[rows, :half].astype(ft)[None, :, None], freqs[rows, half:].astype(ft)[None, :, None]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = x.copy()
    out[..., :half] = ((x1 * cos).astype(ft) - (x2 * sin).astype(ft)).astype(ft)
    out[..., half:2 * half] = ((x2 * cos).astype(ft) + (x1 * sin).astype(ft)).astype(ft)
    return out
  q = rope(norm(q, qn), np.arange(T) + rp)
  k = rope(norm(k, kn), np.arange(T) + rp)
  kv = np.stack([k, v]).transpose(0, 1, 3, 2, 4)                                  # (2,B,KVH,T,D)
  blocks = kv.reshape(*kv.shape[:-1], D // 32, 32)
  amax = np.abs(blocks).max(-1, keepdims=True)
  if exact:
    scale = np.maximum((amax * np.float32(1 / 7)).astype(ft), np.float32(1e-8))
    t = (blocks * (np.float32(1) / scale).astype(ft)).astype(ft)
    nib = np.clip(_tg_round(t), -8, 7) + 8
  else:
    scale = np.maximum(amax / 7, 1e-8)
    nib = np.clip(np.rint(blocks / scale), -8, 7) + 8
  nib = nib.reshape(*kv.shape).astype(np.uint8)
  packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
  return q.transpose(0, 2, 1, 3), gate.reshape(B * T, H * D), packed, scale[..., 0].astype(np.float16)

def _register_partials(parts:np.ndarray, B:int, T:int, device:str) -> Tensor:
  """nv_q8_linear's result for these split-K partials (tokens, F, C): the summed (B,T,F) tensor, registered so split_k_partials finds them"""
  p = Tensor(parts, device=device).contiguous().realize()
  result = p.sum(-1).reshape(B, T, parts.shape[1])
  nv_quant._register_split_k(result, p)
  return result

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestKernelOnEmulator(unittest.TestCase):
  """attn_prep end to end on the PYTHON emulator: the model's geometry (24 q heads, 4 kv heads, head_dim 256, rope 64, C = 5 split-K
  partials) at T = 1 and VERIFY(3); G = 1 / 2 / 6, head_dim 64 / 128 / 256, rope_dim 32 .. head_dim, C = 1 (no split-K partials: the plain
  projections), B = 2, start_pos 0 / 1 / not a multiple of 32 or 128 / at the cache end, a bound symbolic start_pos and a separate rope
  position. Every other cache row must stay as it was."""
  def _run(self, B, H, KVH, D, R, C, T, start, symbolic=False, rope_off=0, N=128, seed=0):
    rng = np.random.default_rng(seed)
    tokens = B * T
    qp, kp, vp = (_f32(rng.standard_normal((tokens, F, C)) * 0.7) for F in (H * 2 * D, KVH * D, KVH * D))
    qn, kn = _f32(rng.uniform(0.5, 1.5, D)), _f32(rng.uniform(0.5, 1.5, D))
    freqs = _f32(rng.uniform(-1, 1, (N, R)))
    kv0 = rng.integers(0, 256, (2, B, KVH, N, D // 2), dtype=np.uint8)
    sc0 = rng.uniform(0.01, 0.2, (2, B, KVH, N, D // 32)).astype(np.float16)
    dev = "PYTHON"
    if C > 1: q, k, v = (_register_partials(p, B, T, dev) for p in (qp, kp, vp))
    else: q, k, v = (Tensor(p[..., 0].reshape(B, T, -1), device=dev).contiguous().realize() for p in (qp, kp, vp))
    cache_kv, cache_scale = Tensor(kv0, device=dev).contiguous().realize(), Tensor(sc0, device=dev).contiguous().realize()
    qn_t, kn_t, fr_t = (Tensor(a, device=dev).contiguous().realize() for a in (qn, kn, freqs))
    if symbolic:
      sp = UOp.variable("start_pos", 0, N - 1).bind(start)
      rope = sp if rope_off == 0 else UOp.variable("rope_pos", 0, N - 1).bind(start + rope_off)
    else: sp, rope = start, start + rope_off
    G = H // KVH
    # the CUDA C renderer emits no MULACC (a*b+c stays two roundings in the source; nvrtc contracts on its own): drop the emulator's fused
    # multiply-add so it runs the source's roundings, which the fp32 transcription reproduces
    no_fma = {k: v for k, v in python_alu.items() if k is not Ops.MULACC}
    with patch.object(nv_attn_prep, "_shfl", _emulated_shfl(G)), patch.object(nv_attn_prep, "_attn_prep_kernel", _UNCACHED), \
         patch.object(PythonRenderer, "code_for_op", no_fma):
      q_out, gate, kv_t, sc_t = nv_attn_prep.attn_prep(q, k, v, qn_t, kn_t, fr_t, cache_kv, cache_scale, sp, rope, H, (1e-6, 1e-6))
      names = [u.arg.name for u in q_out.uop.toposort() if u.op is Ops.SINK and isinstance(u.arg, KernelInfo)]
      self.assertEqual(names, ["attn_prep_nv"])  # the partials are read in place: no summing reduce scheduled
      got_q, got_gate = q_out.numpy(), gate.numpy()
      got_kv, got_sc = kv_t.numpy(), sc_t.numpy()
    args = (qp, kp, vp, qn, kn, freqs, start, start + rope_off, B, T, H, KVH, D, 1e-6)
    ref_q, ref_gate, ref_packed, ref_scale = np_attn_prep(*args)
    np.testing.assert_array_equal(got_q, ref_q)
    np.testing.assert_array_equal(got_gate.reshape(B * T, H * D), ref_gate)
    rows = slice(start, start + T)
    np.testing.assert_array_equal(got_kv[:, :, :, rows], ref_packed)
    np.testing.assert_array_equal(got_sc[:, :, :, rows], ref_scale)
    keep = np.ones(N, bool)
    keep[rows] = False
    np.testing.assert_array_equal(got_kv[:, :, :, keep], kv0[:, :, :, keep])
    np.testing.assert_array_equal(got_sc[:, :, :, keep], sc0[:, :, :, keep])
    q64 = np_attn_prep(*args, exact=False)[0]
    err = float(np.abs(got_q - q64).max())
    np.testing.assert_allclose(got_q, q64, rtol=1e-5, atol=1e-5)
    print(f"\nattn_prep B={B} H={H} KVH={KVH} D={D} R={R} C={C} T={T} start={start}: bit-exact vs the fp32 transcription, max|q - fp64| {err:.2e}")

  def test_model_geometry_decode(self): self._run(1, 24, 4, 256, 64, 5, 1, 37)
  def test_model_geometry_verify(self): self._run(1, 24, 4, 256, 64, 5, 3, 125, seed=1)
  def test_g2_symbolic_start_0(self): self._run(1, 4, 2, 128, 64, 2, 4, 0, symbolic=True, seed=2)
  def test_g1_full_rotary_plain_projections(self): self._run(1, 2, 2, 64, 64, 1, 2, 1, seed=3)
  def test_separate_rope_position(self): self._run(1, 12, 2, 256, 128, 3, 2, 60, symbolic=True, rope_off=5, seed=4)
  def test_batch_2(self): self._run(2, 4, 2, 64, 32, 2, 2, 97, seed=5)

class TestRender(unittest.TestCase):
  """the model's geometry rendered for sm_86 without compiling: one block per (token, kv head) of 6 warps, the lane unsplit on threadIdx.x
  (the xor shuffles pair HARDWARE lanes), 30 shuffles at T = 1 (q and k: 5 for the sum of squares + 8 rope partners; k and v: 2 per absmax),
  the bound position as a parameter, the k/v stores of a block made by its warp 0 only, every scale store under the lane gate."""
  def _src(self, T:int) -> str:
    B, H, KVH, D, R, C, N = 1, 24, 4, 256, 64, 5, 262144
    srcs = [Tensor.empty(B, H, T, D), Tensor.empty(B * T, H * D), Tensor.empty(2, B, KVH, N, D // 2, dtype=dtypes.uint8),
            Tensor.empty(2, B, KVH, N, D // 32, dtype=dtypes.float16), Tensor.empty(B * T, H * 2 * D, C), Tensor.empty(B * T, KVH * D, C),
            Tensor.empty(B * T, KVH * D, C), Tensor.empty(D), Tensor.empty(D), Tensor.empty(N, R)]
    sp = _unbind(UOp.variable("start_pos", 0, N - 1).bind(5))
    sink = nv_attn_prep._attn_prep_kernel(*[UOp.placeholder_like(s.uop, slot=i) for i, s in enumerate(srcs)], start_pos=sp, rope_pos=sp,
                                          eps_q=1e-6, eps_k=1e-6)
    prg, src = _lower_sm86(sink)
    self.assertLess(len(pickle.dumps(prg)), 2_000_000)
    return src

  def test_render_for_sm86(self):
    for T in (1, 3):
      with self.subTest(T=T):
        src = self._src(T)
        self.assertIn("__launch_bounds__(192)", src)
        self.assertIn("int lidx0 = threadIdx.x; /* 32 */", src)
        self.assertIn("int lidx1 = threadIdx.y; /* 6 */", src)
        self.assertEqual(src.count("__shfl_sync(0xffffffff"), 30)
        self.assertIn("const int data10_)", src)  # the bound start_pos: a scalar parameter after the ten buffers
        lines = [l.strip() for l in src.splitlines()]
        cache = [i for i, l in enumerate(lines) if l.startswith(("*(data2_", "*(data3_"))]
        self.assertEqual(len(cache), 10, "per lane: PER/2 = 4 packed bytes + 1 scale, for k and v")
        for i in cache: self.assertTrue(lines[i - 1].startswith("if ("), f"ungated cache store: {lines[i]}")
        self.assertTrue(any(l.startswith("bool ") and "(lidx1==0)" in l for l in lines), "the k/v stores: warp 0 of the block")

def _block(n_heads=12, n_kv_heads=2, head_dim=128, rope_dim=64, dim=1024):
  from tinygrad.llm.model import TransformerBlock, TransformerConfig
  from test.unit.test_llm_nv_quant import TestDecodeStepCensus as C
  config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=2048, n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=1e-6, vocab_size=8,
                             head_dim=head_dim, rope_theta=10000.0, rope_dim=rope_dim, v_head_dim=head_dim, max_context=256, qk_norm=head_dim,
                             attn_output_gate=True)
  b = TransformerBlock(config)
  b.attn_q, b.attn_k = C._q4_0(dim, n_heads * head_dim * 2, 0), C._q4_0(dim, n_kv_heads * head_dim, 1)
  b.attn_v, b.attn_output = C._q4_0(dim, n_kv_heads * head_dim, 2), C._q4_0(n_heads * head_dim, dim, 6)
  b.attn_q_norm.weight, b.attn_k_norm.weight = C._gguf_f32(head_dim, 12), C._gguf_f32(head_dim, 13)
  b.ffn_gate, b.ffn_up, b.ffn_down = C._q4_0(dim, 2048, 3), C._q4_0(dim, 2048, 4), C._q4_0(2048, dim, 5)
  b.attn_norm.weight, b.ffn_norm.weight = C._gguf_f32(dim, 10), C._gguf_f32(dim, 11)
  return b

class TestRoutingAndCensus(unittest.TestCase):
  """graph level on CPU with every NV gate mocked open (test_llm_nv_quant's _nv_gates_open): a KV_INT4 decode step / VERIFY run takes
  attn_prep_nv in place of the 9 generic glue kernels; NV_ATTN_PREP=0, a prefill chunk (symbolic or wider than 4), an fp16 cache, q/k norm
  weights not prepared at load, an image rope tensor and a geometry the lanes don't cover keep the generic graph."""
  def _census(self, b, T=1, symbolic=False, kv_dtype=dtypes.uint8, rope=None, prepare=True, finish=0):
    # finish: NV_ATTN_FINISH (T4.144, test_llm_nv_attn_finish) -- off here, so the counts below are attn_prep's alone
    from tinygrad.llm.kernels.nv_dense import prepare_dense_weights
    from test.unit.test_llm_nv_quant import _decode_census, _nv_gates_open
    cfg = b.config
    with _nv_gates_open(), patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True), Context(NV_ATTN_FINISH=finish):
      prepare_dense_weights(b)
      if prepare: b.prepare_nv_norms()
      with Context(DEBUG=0):
        x = Tensor.randn(1, 8 if symbolic else T, cfg.dim).contiguous().realize()
        b._init_state(x)
        D = cfg.head_dim
        b.cache_kv = Tensor.zeros(2, 1, cfg.n_kv_heads, cfg.max_context, D // 2 if kv_dtype == dtypes.uint8 else D, dtype=kv_dtype).contiguous()
        b.cache_kv_scale = Tensor.ones(2, 1, cfg.n_kv_heads, cfg.max_context, D // 32, dtype=dtypes.float16).contiguous()
        Tensor.realize(b.cache_kv, b.cache_kv_scale, b.freqs_cis)
      if symbolic: x = x[:, :UOp.variable("toks", 1, 8).bind(T)]
      sp = UOp.variable("start_pos", 0, cfg.max_context - 9).bind(5)
      return _decode_census(b._ffn_output(x, b._attention(b._normed_for_attention(x), sp, rope)).contiguous())

  def test_decode_and_verify_take_one_kernel(self):
    b = _block()
    for T in (1, 3, 4):
      with self.subTest(T=T):
        on = self._census(b, T)
        with Context(NV_ATTN_PREP=0): off = self._census(b, T)
        print(f"\nT={T}: {len(off)} kernels -> {len(on)}")
        self.assertEqual(on.count("attn_prep_nv"), 1)
        self.assertNotIn("attn_prep_nv", off)
        self.assertEqual(len(off) - len(on), 6, "dim 1024: no q/k/v partial sums to fold (C = 1); 7 glue kernels -> 1")
        # what stays generic is the attention's merge (x the gate) and the block output: no KV store, no per-step norm-weight read
        generic = [k for k in on if k.startswith(("reduce", "elementwise"))]
        self.assertFalse([k for k in generic if "unsigned char[" in k or k.startswith("elementwise half[")], generic)

  def test_split_k_partials_are_read_in_place(self):
    # dim 2048: attn_q/attn_k/attn_v's gemvs return C = 2 partials -- the two generic partial-sum reduces go too (9 -> 1, the card's count)
    b = _block(dim=2048)
    on = self._census(b, 1)
    with Context(NV_ATTN_PREP=0): off = self._census(b, 1)
    self.assertEqual(len(off) - len(on), 8)

  def test_generic_paths_unchanged(self):
    b = _block()
    self.assertNotIn("attn_prep_nv", self._census(b, 5), "T > 4: a prefill-sized concrete chunk")
    self.assertNotIn("attn_prep_nv", self._census(b, 3, symbolic=True), "symbolic prefill chunk")
    self.assertNotIn("attn_prep_nv", self._census(b, 1, kv_dtype=dtypes.float16), "fp16 cache")
    b2 = _block()
    self.assertNotIn("attn_prep_nv", self._census(b2, 1, prepare=False), "q/k norm weights not prepared at load")
    self.assertNotIn("attn_prep_nv", self._census(b, 1, rope=Tensor.zeros(1, 1, 3, dtype=dtypes.int32).contiguous().realize()), "image rope rows")
    self.assertNotIn("attn_prep_nv", self._census(_block(rope_dim=4), 1), "a rope half inside one lane's dims")

  def test_geometry_gate(self):
    ok = nv_attn_prep.attn_prep_geometry_ok
    self.assertTrue(ok(24, 4, 256, 64))   # qwen3.8-27B
    self.assertTrue(ok(4, 2, 64, 64))
    self.assertTrue(ok(12, 2, 128, 32))   # rope halves of 16 dims = 4 lanes of 4
    self.assertFalse(ok(24, 4, 96, 64))   # 96: lanes of 3 dims
    self.assertFalse(ok(24, 4, 32, 32))   # one dim per lane: no whole bytes
    self.assertFalse(ok(24, 4, 256, 48))  # rope half of 24 dims = 3 lanes: not a power of two apart
    self.assertFalse(ok(24, 4, 256, 8))   # rope half of 4 dims inside a lane's 8
    self.assertFalse(ok(66, 2, 128, 64))  # 33 warps
    self.assertFalse(ok(24, 5, 256, 64))

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestAttentionOnEmulator(unittest.TestCase):
  """TransformerBlock._attention on the PYTHON emulator -- plain projections (no NV quant), the generic attention (_nv_attn mocked to
  None) -- with the prep kernel (gates mocked open) vs NV_ATTN_PREP=0: the block output and the stored KV rows agree (the RMS sum order is
  the only difference: a k nibble may flip on an exact rounding tie, the output within fp32 noise)."""
  def _run(self, T:int, start:int, prep:bool, symbolic:bool):
    from tinygrad.llm import model
    from tinygrad.llm.model import TransformerBlock, TransformerConfig, precompute_freqs_cis
    dev, rng = "PYTHON", np.random.default_rng(7)
    cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, norm_eps=1e-6, vocab_size=8, head_dim=64,
                            rope_theta=10000.0, rope_dim=32, v_head_dim=64, max_context=128, qk_norm=64, attn_output_gate=True)
    b = TransformerBlock(cfg)
    for lin, out_f, in_f in ((b.attn_q, 512, 64), (b.attn_k, 128, 64), (b.attn_v, 128, 64), (b.attn_output, 64, 256)):
      lin.weight = Tensor(_f32(rng.standard_normal((out_f, in_f)) * 0.2), device=dev).contiguous().realize()
    b.attn_q_norm.weight, b.attn_k_norm.weight = (Tensor(_f32(rng.uniform(0.5, 1.5, 64)), device=dev).contiguous().realize() for _ in range(2))
    b.nv_qk_norm_weights = (b.attn_q_norm.weight, b.attn_k_norm.weight)
    b.freqs_cis = precompute_freqs_cis(32, 128, 10000.0, device=dev).realize()
    b.cache_kv = Tensor(rng.integers(0, 256, (2, 1, 2, 128, 32), dtype=np.uint8), device=dev).contiguous().realize()
    b.cache_kv_scale = Tensor(rng.uniform(0.01, 0.2, (2, 1, 2, 128, 2)).astype(np.float16), device=dev).contiguous().realize()
    x = Tensor(_f32(rng.standard_normal((1, T, 64))), device=dev).contiguous().realize()
    sp = UOp.variable("start_pos", 0, 127 - 4).bind(start) if symbolic else start
    with patch.object(model, "nv_decode_attention_ok", lambda *a: True), patch.object(TransformerBlock, "_nv_attn", lambda *a: None), \
         patch.object(nv_attn_prep, "_shfl", _emulated_shfl(2)), patch.object(nv_attn_prep, "_attn_prep_kernel", _UNCACHED), \
         Context(NV_ATTN_PREP=int(prep)):
      out = b._attention(x, sp)
      self.assertEqual(prep, any(u.op is Ops.SINK and isinstance(u.arg, KernelInfo) and u.arg.name == "attn_prep_nv" for u in out.uop.toposort()))
      out = out.numpy()
    return out, b.cache_kv.numpy(), b.cache_kv_scale.numpy()

  def test_matches_the_generic_graph(self):
    for T, start, symbolic in ((1, 37, False), (3, 70, True)):
      with self.subTest(T=T, start=start):
        (out_a, kv_a, sc_a), (out_b, kv_b, sc_b) = self._run(T, start, True, symbolic), self._run(T, start, False, symbolic)
        np.testing.assert_allclose(out_a, out_b, rtol=1e-4, atol=1e-5)
        nib_a, nib_b = (np.stack([kv & 0xF, kv >> 4], -1).astype(int) for kv in (kv_a, kv_b))
        self.assertLessEqual(np.abs(nib_a - nib_b).max(), 1)
        flips = int((nib_a != nib_b).sum())
        self.assertLessEqual(flips, 2, "only exact-tie rounding flips")
        np.testing.assert_array_equal(sc_a, sc_b)
        print(f"\n_attention T={T} start={start}: max|out diff| {np.abs(out_a - out_b).max():.2e}, nibble flips {flips}")

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestGenerationOnEmulator(unittest.TestCase):
  """the served loops on the PYTHON emulator with attn_prep_nv in every attention block (gates mocked open, plain projections, the generic
  attention): generate()'s decode jit and speculative_generate's VERIFY families (concrete widths 2..k+1, spec=True), the MTP head's
  draft chain (its own block and cache, a second bound position) and the accept's cache rewrite -- each one capture + replays through the
  @function trace, with the bound start_pos riding as the kernel's last srcs. Token-identical to generate() with NV_ATTN_PREP=0, which
  prefills through the generic graph as both do (symbolic chunks)."""
  def _model(self):
    import os
    from tinygrad import nn
    from tinygrad.helpers import getenv
    from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, MTPHead
    cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=16, head_dim=64,
                            rope_theta=10000.0, rope_dim=32, v_head_dim=64, max_context=64, qk_norm=64, attn_output_gate=True)
    model = Transformer(cfg)
    model.mtp_head = MTPHead(cfg, TransformerBlock)
    rng = np.random.default_rng(11)
    for p in nn.state.get_parameters(model): p.replace(Tensor(_f32(rng.standard_normal(p.shape) * 0.3), device="PYTHON").cast(p.dtype).contiguous())
    Tensor.realize(*nn.state.get_parameters(model))
    for b in (*model.blk, model.mtp_head.block): b.nv_qk_norm_weights = (b.attn_q_norm.weight.float().contiguous().realize(),
                                                                         b.attn_k_norm.weight.float().contiguous().realize())
    with patch.dict(os.environ, {"KV_INT4": "1"}):
      getenv.cache_clear()
      for b in (*model.blk, model.mtp_head.block): b._init_state(Tensor.empty(1, 1, 64, device="PYTHON"))
    getenv.cache_clear()
    self.assertEqual(model.blk[0].cache_kv.dtype, dtypes.uint8)
    return model

  def _tokens(self, prep:bool, spec:bool, prompt=(1, 5, 9, 2), n:int=7, k:int=3) -> list[int]:
    from tinygrad.llm import model as model_mod
    m = self._model()
    with patch.object(model_mod, "nv_decode_attention_ok", lambda *a: True), patch.object(model_mod.TransformerBlock, "_nv_attn", lambda *a: None), \
         patch.object(nv_attn_prep, "_shfl", _emulated_shfl(2)), patch.object(nv_attn_prep, "_attn_prep_kernel", _UNCACHED), \
         Context(NV_ATTN_PREP=int(prep)), patch.object(model_mod, "attn_prep", wraps=nv_attn_prep.attn_prep) as calls:
      gen = m.speculative_generate(list(prompt), k=k) if spec else m.generate(list(prompt), temperature=0.0)
      out = [t for _, t in zip(range(n), gen)]
    self.assertEqual(calls.call_count > 0, prep)
    return out

  def test_generate_and_speculative_match_the_generic_graph(self):
    ref = self._tokens(False, False)
    self.assertGreater(len(set(ref)), 2, f"a degenerate stream would prove little: {ref}")
    self.assertEqual(self._tokens(True, False), ref, "generate: decode steps through attn_prep_nv")
    self.assertEqual(self._tokens(True, True), ref, "speculative_generate: VERIFY + drafts through attn_prep_nv")

if __name__ == "__main__": unittest.main()
