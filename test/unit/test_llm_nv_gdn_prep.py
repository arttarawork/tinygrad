"""T4.138: the GatedDeltaNet prefill chunk's glue as NV kernels (tinygrad/llm/kernels/nv_gdn_prep.py).

TestPrefillCensus: the kernels of ONE GatedDeltaNetBlock prefill call at the 27B geometry (symbolic 128-token chunk, the NV gates mocked
open on CPU -- scheduled, never run), each generic kernel named by the sm_86 hand-coded optimizer: the names of the card's 10-31 profile
(nv_gdn_prep.py's map). Rendering needs no nvrtc lane (_render_cuda's trick: the renderer's compiler server is mocked out)."""
import contextlib, itertools, re, sys, unittest
from unittest import mock
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, Context
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops, KernelInfo, AxisType
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, Q4_0, kernel_var
from tinygrad.llm.kernels import nv, nv_gemm, nv_gdn_prep
from tinygrad.llm.kernels.nv_gdn_prep import gdn_prefill, gdn_post, _gdn_prep_kernel, _gdn_conv_state_kernel, _gdn_post_kernel
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig, Transformer

# ******** the 27B GDN block, scheduled with the NV gates open ********

SSM_27B = SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=6144)

def q4_0_linear(in_features:int, out_features:int, seed:int) -> Linear:
  rng = np.random.default_rng(seed)
  packed = rng.integers(0, 256, 18 * (in_features//32) * out_features, dtype=np.uint8).reshape(-1, 18)
  packed[:, 0:2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)  # finite d
  raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
  lin = Linear(in_features, out_features, bias=False)
  nn.state.load_state_dict(lin, {"weight": ggml_data_to_tensor(raw, in_features*out_features, Q4_0).reshape(out_features, in_features)},
                           verbose=False, realize=False)
  lin.set_quantized(lin.weight)
  return lin

def _lazy_f16(shape, seed:int) -> Tensor:  # from_gguf's lazy F32 -> .cast(float16) view
  return Tensor(np.random.default_rng(seed).standard_normal(shape).astype(np.float32)).realize().cast(dtypes.float16)

@contextlib.contextmanager
def nv_gates_open():
  # every NV gate a prefill chunk checks, opened on CPU (graphs are built and scheduled, never run)
  with Context(NV_CUSTOM_QUANT=1), mock.patch("tinygrad.llm.model.nv_quant_supported", return_value=True), \
       mock.patch("tinygrad.llm.kernels.nv_quant.nv_quant_supported", return_value=True), \
       mock.patch("tinygrad.llm.model.nv_custom_kernels_supported", return_value=True), \
       mock.patch("tinygrad.llm.model.nv_decode_kernel_supported", return_value=True), \
       mock.patch.object(nv_gemm, "_nv_wmma_ok", lambda device: True):
    yield

def card_block(dim:int=5120, hidden:int=17408) -> GatedDeltaNetBlock:
  """the served layout: Q4_0 projections on the IMMA gemm (ssm_out is Q5_K on the card: same kernels, + q8 sums), fp16 ssm_beta/alpha
  on the f16 gemv, lazy fp16 norm/conv/dt/a weights; prepared as from_gguf does (prepare_dense_weights, prepare_nv_norms/decode)"""
  from tinygrad.llm.kernels.nv_dense import prepare_dense_weights
  config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=hidden, n_heads=24, n_kv_heads=4, norm_eps=1e-6, vocab_size=32, head_dim=256,
                             rope_theta=1e7, rope_dim=64, v_head_dim=256, max_context=262144, ssm_layers=(True,), ssm=SSM_27B)
  b = GatedDeltaNetBlock(config, SSM_27B)
  b.attn_qkv, b.attn_gate, b.ssm_out = q4_0_linear(dim, b.conv_channels, 0), q4_0_linear(dim, 6144, 1), q4_0_linear(6144, dim, 2)
  b.ffn_gate, b.ffn_up, b.ffn_down = q4_0_linear(dim, hidden, 3), q4_0_linear(dim, hidden, 4), q4_0_linear(hidden, dim, 5)
  b.ssm_beta.weight, b.ssm_alpha.weight = _lazy_f16((48, dim), 6), _lazy_f16((48, dim), 7)
  b.attn_norm.weight, b.ffn_norm.weight, b.ssm_norm.weight = _lazy_f16((dim,), 8), _lazy_f16((dim,), 9), _lazy_f16((128,), 10)
  b.ssm_conv1d["weight"], b.ssm_dt["bias"], b.ssm_a = _lazy_f16((b.conv_channels, 4), 11), _lazy_f16((48,), 12), _lazy_f16((48,), 13)
  with nv_gates_open():
    prepare_dense_weights(b)
    b.prepare_nv_norms()
    b.prepare_nv_decode()
  return b

def sm86_name(ast:UOp) -> str:
  """a generic kernel's name after the sm_86 hand-coded optimizer (the name the card's profile shows), no nvrtc needed"""
  from tinygrad.codegen import full_rewrite_to_sink, ProgramInfo
  from tinygrad.renderer.cstyle import CUDARenderer
  with mock.patch("tinygrad.runtime.support.compiler_cuda._get_server", return_value=None): ren = CUDARenderer(DEV.target("NV", arch="sm_86"))
  return ProgramInfo.from_sink(full_rewrite_to_sink(ast, ren), ren.target).function_name

def prefill_census(block:GatedDeltaNetBlock, chunk:int=128, start_pos:int=256) -> list[str]:
  x = Tensor.randn(1, chunk, block.config.dim).realize()
  xin, sp = x[:, :UOp.variable("toks", 1, chunk).bind(chunk)], UOp.variable("start_pos", 0, block.config.max_context - 1).bind(start_pos)
  block._init_state(xin)
  Tensor.realize(block.conv_state, block.recurrent_state)
  with nv_gates_open(): linear, _ = block(xin, sp).linear_with_vars()
  names = []
  for call in (u for u in linear.src if u.op is Ops.CALL):
    ast = call.src[0]
    names.append(ast.arg.name if isinstance(ast.arg, KernelInfo) and ast.arg.opts_to_apply == () else sm86_name(ast))
  return names

# ******** step 1: gdn_prefill (gdn_prep_nv + gdn_conv_state_nv + the scan) ********

def np_gdn_prefill(qkv, ba, conv0, st0, conv_w, dt_bias, ssm_a, start_pos:int, toks:int, k_heads:int, qk_eps=1e-6):
  """fp64 ground truth of a prefill chunk from its projections (model.py's generic graph): qkv (B,T,C), ba (B,T,2H) (fp16 values),
  conv0 (B,taps-1,C), st0 (B,H,V,K). Returns core (B,H,T,V), the new conv_state (B,taps-1,C), state, and the scan inputs
  {q, k, v, kq, beta, alpha}. beta is the exact sigmoid rounded to fp16: the kernels (and the generic graph) compute the fp16 projection's
  sigmoid in fp16 arithmetic, up to 2 fp16 ulps (1e-3) from that -- ~2e-4 of state drift over 128 tokens (beta = 0.5 exactly: 1e-7)."""
  B, T, C = qkv.shape
  _, H, V, K = st0.shape
  hist = conv0.shape[1]
  if start_pos == 0: conv0, st0 = np.zeros_like(conv0), np.zeros_like(st0)
  qkv64 = np.concatenate([qkv[:, :toks].astype(np.float64), np.zeros((B, T - toks, C))], axis=1)
  window = np.concatenate([conv0.astype(np.float64), qkv64], axis=1)  # (B, hist+T, C): [conv_state | rows < toks | zero]
  conv = sum(window[:, i:i+T] * conv_w[:, i].astype(np.float64) for i in range(hist + 1))
  conv = conv / (1 + np.exp(-conv))
  qd = k_heads*K
  q, k = conv[..., :qd].reshape(B, T, k_heads, K), conv[..., qd:2*qd].reshape(B, T, k_heads, K)
  q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), qk_eps) * K**-0.5
  k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), qk_eps)
  q, k = (np.tile(x, (1, 1, H // k_heads, 1)).transpose(0, 2, 1, 3) for x in (q, k))  # .repeat(): v-head h reads k-head h % k_heads
  v = conv[..., 2*qd:].reshape(B, T, H, V).transpose(0, 2, 1, 3)
  ba = ba.astype(np.float64)
  valid = (np.arange(T) < toks)[None, :, None]
  beta = np.where(valid, (1/(1 + np.exp(-ba[..., :H]))).astype(np.float16).astype(np.float64), 0).transpose(0, 2, 1)
  alpha = np.where(valid, np.exp(np.logaddexp(ba[..., H:] + dt_bias, 0) * ssm_a), 1).transpose(0, 2, 1)
  from test.unit.test_llm_nv_scan import reference
  core, st = reference(q, k, v, beta, np.broadcast_to(alpha[..., None], v.shape), st0.astype(np.float64))
  return core, window[:, toks:toks+hist], st, dict(q=q, k=k, v=v, kq=(q*k).sum(-1), beta=beta, alpha=alpha)

def gdn_inputs(rng, B, T, Hk, H, K, V, hist=3, scale=0.5):
  C = 2*Hk*K + H*V
  def f(*s, sc=1.0): return (rng.standard_normal(s) * sc).astype(np.float32)
  return dict(qkv=f(B, T, C, sc=scale), ba=f(B, T, 2*H).astype(np.float16), conv0=f(B, hist, C).astype(np.float16), st0=f(B, H, V, K, sc=0.3),
              conv_w=f(C, hist + 1, sc=0.5), dt_bias=f(H), ssm_a=-np.abs(f(H)))

def _args(inp:dict, start_pos:int, toks:int, symbolic:bool, device:str|None, max_context:int):
  T = inp["qkv"].shape[1]
  return (Tensor(inp["qkv"], device=device), Tensor(inp["ba"], device=device)), \
         tuple(Tensor(inp[n], device=device) for n in ("conv_w", "dt_bias", "ssm_a")) + \
         (UOp.variable("start_pos", 0, max_context - 1).bind(start_pos), UOp.variable("toks", 1, T).bind(toks) if symbolic else toks)

def run_gdn_prefill(inp:dict, start_pos:int, toks:int, k_heads:int, symbolic:bool=True, device:str|None=None, max_context:int=64):
  """gdn_prefill on fresh state buffers; returns (core, conv_state, state) as numpy"""
  conv_state = Tensor(inp["conv0"], device=device).contiguous().realize()
  state = Tensor(inp["st0"], device=device).contiguous().realize()
  (qkv, ba), rest = _args(inp, start_pos, toks, symbolic, device, max_context)
  core = gdn_prefill(qkv, ba, conv_state, state, *rest, k_heads=k_heads, qk_eps=1e-6)
  return core.numpy(), conv_state.numpy(), state.numpy()

def run_gdn_prep(inp:dict, start_pos:int, toks:int, k_heads:int, symbolic:bool=True, device:str|None=None, max_context:int=64):
  """the two pre-scan kernels alone: ({q, k, v, kq, beta, alpha}, the stored conv_state) as numpy"""
  conv_state = Tensor(inp["conv0"], device=device).contiguous().realize()
  (qkv, ba), rest = _args(inp, start_pos, toks, symbolic, device, max_context)
  _, H, V, K = inp["st0"].shape
  outs, conv_call = nv_gdn_prep.gdn_prep(qkv, ba, conv_state, *rest, heads=H, key_dim=K, value_dim=V, k_heads=k_heads, qk_eps=1e-6)
  Tensor.realize(*outs, Tensor(conv_state.uop.after(conv_call)))
  return dict(zip(("q", "k", "v", "kq", "beta", "alpha"), (t.numpy() for t in outs))), conv_state.numpy()

def check_prep(test:unittest.TestCase, got:dict, conv:np.ndarray, want:dict, want_conv:np.ndarray):
  # fp32-level everywhere but beta (fp16 arithmetic: <= 2 fp16 ulps); conv_state bit-exact (fp32 rows rounded to fp16 once)
  for name in ("q", "k", "v", "kq", "alpha"): np.testing.assert_allclose(got[name], want[name], rtol=2e-5, atol=2e-6, err_msg=name)
  np.testing.assert_allclose(got["beta"], want["beta"], rtol=0, atol=1e-3, err_msg="beta")
  np.testing.assert_array_equal(conv, want_conv.astype(np.float16))

class TestGDNPrefillCPU(unittest.TestCase):
  """gdn_prefill's CPU form (lanes=1: one thread owns a warp's channels, no shuffles; the scan's lanes=1 form) against np_gdn_prefill:
  chunk widths 8/32/128 (prep tile 4) and 2 (one tile, shorter than the conv history), a full and a remainder chunk, start_pos 0
  (state reset) / mid / near max_context, a nonzero incoming conv/recurrent state, and the concrete-width form."""
  CASES = ((8, 8, 0, True), (8, 5, 9, True), (32, 32, 60, True), (32, 1, 3, True), (128, 77, 200, True), (2, 1, 5, True), (2, 2, 0, True),
           (8, 8, 4, False))
  def _want(self, inp, sp, toks):
    return np_gdn_prefill(inp["qkv"], inp["ba"], inp["conv0"], inp["st0"], inp["conv_w"], inp["dt_bias"], inp["ssm_a"], sp, toks, 2)

  def test_prep_matches_numpy(self):
    rng = np.random.default_rng(0)
    for T, toks, sp, symbolic in self.CASES:
      inp = gdn_inputs(rng, 1, T, 2, 4, 32, 32)
      with self.subTest(T=T, toks=toks, start_pos=sp, symbolic=symbolic):
        _, want_conv, _, want = self._want(inp, sp, toks)
        check_prep(self, *run_gdn_prep(inp, sp, toks, 2, symbolic, max_context=256), want, want_conv)

  def test_prefill_matches_numpy(self):
    rng = np.random.default_rng(1)
    for T, toks, sp, symbolic in self.CASES:
      inp = gdn_inputs(rng, 1, T, 2, 4, 32, 32)
      with self.subTest(T=T, toks=toks, start_pos=sp, symbolic=symbolic):
        core, conv, st = run_gdn_prefill(inp, sp, toks, 2, symbolic, max_context=256)
        want_core, want_conv, want_st, _ = self._want(inp, sp, toks)
        np.testing.assert_allclose(core, want_core, rtol=2e-3, atol=5e-4)
        np.testing.assert_array_equal(conv, want_conv.astype(np.float16))
        np.testing.assert_allclose(st, want_st, rtol=2e-3, atol=5e-4)
    # beta = 0.5 exactly (an fp16 sigmoid with no rounding): the whole chain at fp32 level
    inp = gdn_inputs(rng, 1, 128, 2, 4, 32, 32)
    inp["ba"][..., :4] = 0
    core, _, st = run_gdn_prefill(inp, 0, 128, 2, max_context=256)
    want_core, _, want_st, _ = self._want(inp, 0, 128)
    np.testing.assert_allclose(core, want_core, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(st, want_st, rtol=1e-5, atol=1e-6)

# the PYTHON emulator has no CUDA intrinsics and runs a block in lockstep: each xor shuffle becomes a block-shared exchange
# (test_llm_nv_norm's _emulated_warp_reduce), thread tid = warp*32 + lane, partner tid ^ off
_emu_slot = itertools.count(300)
def _emulated_xor(val:UOp, offsets, maximum:bool=False) -> UOp:
  rngs = {u.arg[-1]: u for u in val.toposort() if u.op is Ops.RANGE}
  lane = rngs.get(AxisType.WARP, rngs.get(AxisType.LOCAL))
  tid, n = (rngs[AxisType.LOCAL] * 32 + lane, rngs[AxisType.LOCAL].src[0].arg * 32) if AxisType.WARP in rngs and AxisType.LOCAL in rngs \
    else (lane, 32)
  for off in offsets:
    buf = UOp.placeholder((n,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
    other = buf.after(buf[tid].store(val))[tid + off - (tid // off) % 2 * (2 * off)].load()  # tid ^ off, z3-boundable
    val = val.maximum(other) if maximum else val + other
  return val
def _emulated_warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  return _emulated_xor(val, (16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1), maximum)

@contextlib.contextmanager
def emulated_lanes():
  # lanes=32 on the PYTHON device with every shuffle emulated; the kernel builders' caches bypassed (they'd hold shuffle graphs)
  with mock.patch.object(nv_gdn_prep, "lanes_for", lambda device: 32), mock.patch.object(nv_gdn_prep, "xor_reduce", _emulated_xor), \
       mock.patch.object(nv, "warp_reduce", _emulated_warp_reduce), \
       mock.patch.object(nv_gdn_prep, "_gdn_prep_kernel", _gdn_prep_kernel.__wrapped__), \
       mock.patch.object(nv_gdn_prep, "_gdn_conv_state_kernel", _gdn_conv_state_kernel.__wrapped__), \
       mock.patch.object(nv_gdn_prep, "_gated_delta_prefill_kernel", nv._gated_delta_prefill_kernel.__wrapped__), \
       mock.patch.object(nv_gdn_prep, "_gdn_post_kernel", _gdn_post_kernel.__wrapped__):
    yield

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestGDNPrefillEmulator(unittest.TestCase):
  """gdn_prefill at lanes=32 -- the card's thread mapping: WARP lanes, warp-reduced l2 norms and q.k, the scan's 32-lane rows -- EXECUTED
  on tinygrad's PYTHON emulator against the fp64 reference, through three chained chunks (the conv window crossing chunk boundaries,
  the state carried in the real buffers), chunk widths 32 and 128, a remainder chunk, start_pos 0 / mid / near max_context."""
  def _chain(self, T:int, calls, max_context:int):
    rng = np.random.default_rng(T)
    inp = gdn_inputs(rng, 1, T, 2, 4, 32, 32)
    conv_state = Tensor(inp["conv0"], device="PYTHON").contiguous().realize()
    state = Tensor(inp["st0"], device="PYTHON").contiguous().realize()
    conv_ref, st_ref = inp["conv0"], inp["st0"]
    weights = tuple(Tensor(inp[n], device="PYTHON") for n in ("conv_w", "dt_bias", "ssm_a"))
    worst = 0.0
    for i, (sp, toks) in enumerate(calls):
      qkv, ba = gdn_inputs(rng, 1, T, 2, 4, 32, 32)["qkv"], gdn_inputs(rng, 1, T, 2, 4, 32, 32)["ba"]
      with self.subTest(T=T, call=i, start_pos=sp, toks=toks), emulated_lanes():
        core = gdn_prefill(Tensor(qkv, device="PYTHON"), Tensor(ba, device="PYTHON"), conv_state, state, *weights,
                           UOp.variable("start_pos", 0, max_context - 1).bind(sp), UOp.variable("toks", 1, T).bind(toks), k_heads=2,
                           qk_eps=1e-6).numpy()
        want_core, conv_ref, st_ref, _ = np_gdn_prefill(qkv, ba, conv_ref, st_ref, inp["conv_w"], inp["dt_bias"], inp["ssm_a"], sp, toks, 2)
        conv_ref = conv_ref.astype(np.float16)  # the buffer keeps fp16
        np.testing.assert_allclose(core, want_core, rtol=2e-3, atol=5e-4)
        np.testing.assert_array_equal(conv_state.numpy(), conv_ref)
        np.testing.assert_allclose(state.numpy(), st_ref, rtol=2e-3, atol=5e-4)
        worst = max(worst, float(np.abs(core - want_core).max()))
    print(f"\nT4.138 emulator gdn_prefill chunk {T}: max |core - fp64| {worst:.2e}")

  def test_prep_kernels(self):
    # the two pre-scan kernels' every output at 128 tokens, a remainder chunk mid-context and a full one from a reset
    rng = np.random.default_rng(5)
    for sp, toks in ((300, 77), (0, 128)):
      inp = gdn_inputs(rng, 1, 128, 2, 4, 32, 32)
      with self.subTest(start_pos=sp, toks=toks), emulated_lanes():
        got, conv = run_gdn_prep(inp, sp, toks, 2, device="PYTHON", max_context=512)
        _, want_conv, _, want = np_gdn_prefill(inp["qkv"], inp["ba"], inp["conv0"], inp["st0"], inp["conv_w"], inp["dt_bias"], inp["ssm_a"],
                                               sp, toks, 2)
        check_prep(self, got, conv, want, want_conv)
        print(f"\nT4.138 emulator gdn_prep: max |q,k,v - fp64| {max(float(np.abs(got[n] - want[n]).max()) for n in 'qkv'):.2e}")

  def test_chunk_32(self): self._chain(32, ((0, 32), (32, 32), (64, 11)), max_context=128)
  def test_chunk_128_restart_and_near_max(self): self._chain(128, ((5, 128), (0, 128), (262000, 77)), max_context=262144)

# ******** step 2: gdn_post (gdn_post_nv) ********

def np_gdn_post(core, gate, norm_w, toks:int, eps:float=1e-6) -> np.ndarray:
  """fp64 z (B,T,H*V) = ssm_norm(core) * norm_w * SiLU(gate), rows >= toks zero (model.py's generic graph, then nv_wmma_linear's pad)"""
  B, H, T, V = core.shape
  c = core.astype(np.float64)
  g = gate.astype(np.float64).reshape(B, T, H, V).transpose(0, 2, 1, 3)
  z = c / np.sqrt((c**2).mean(-1, keepdims=True) + eps) * norm_w * g / (1 + np.exp(-g))
  z = np.where((np.arange(T) < toks)[None, None, :, None], z, 0)
  return z.transpose(0, 2, 1, 3).reshape(B, T, H*V)

def np_q8(z:np.ndarray):
  """q8_quantize_nv's exact formulas (test_llm_nv_gemm._q8) on an fp32 (rows, F) activation: packed words, scales, group sums"""
  x = z.astype(np.float32).reshape(z.shape[0], -1, 32)
  xd = np.maximum(np.abs(x).max(-1) / np.float32(127), np.float32(1e-8)).astype(np.float32)
  q = np.clip(np.rint(x / xd[..., None]), -127, 127).astype(np.int8)
  return q.view(np.uint32), xd, (xd * q.astype(np.int32).sum(-1).astype(np.float32)).astype(np.float32)

def check_post(test:unittest.TestCase, core, gate, norm_w, toks:int, device:str|None):
  T = core.shape[2]
  tk = UOp.variable("toks", 1, T).bind(toks)
  z, xq, xd, xs = (t.numpy() for t in gdn_post(Tensor(core, device=device), Tensor(gate, device=device), Tensor(norm_w, device=device), tk, 1e-6))
  np.testing.assert_allclose(z.astype(np.float32), np_gdn_post(core, gate, norm_w, toks), rtol=2e-3, atol=2e-4)  # one fp16 rounding
  np.testing.assert_array_equal(z[:, toks:], 0)
  # q8 of the kernel's own fp16 z: numpy's exact division may round a near-tie of x/scale the other way than the backend's lowering of
  # the same formula (fp16 z makes those ties common), so +-1 there; the scale to fp32 rounding; the sums exactly scale * sum(q)
  zf = z.reshape(-1, z.shape[-1]).astype(np.float32)
  wq, wd, _ = np_q8(zf)
  q, ref = xq.view(np.int8).reshape(zf.shape[0], -1, 32).astype(np.int32), wq.view(np.int8).reshape(zf.shape[0], -1, 32).astype(np.int32)
  frac = np.abs(zf.reshape(q.shape) / wd[..., None]) % 1
  test.assertTrue((np.abs(q - ref) <= 1).all() and (np.abs(frac[q != ref] - 0.5) < 1e-3).all())
  np.testing.assert_allclose(xd, wd, rtol=1e-6)
  np.testing.assert_array_equal(xs, (xd * q.sum(-1).astype(np.float32)).astype(np.float32))
  if device == "PYTHON":  # bit-identical to q8_quantize_nv (the prefill multi-warp kernel) run on that z
    from tinygrad.llm.kernels import nv_quant
    with mock.patch.object(nv_quant, "warp_reduce", _emulated_warp_reduce), \
         mock.patch.object(nv_quant, "_q8_quantize_kernel", nv_quant._q8_quantize_kernel.__wrapped__):
      got = [t.numpy() for t in nv_quant.q8_quantize(Tensor(zf.astype(np.float16), device=device), zf.shape[0], zf.shape[1], sums=True)]
    for a, b in zip((xq, xd, xs), got): np.testing.assert_array_equal(a, b)

def post_inputs(rng, B, T, H, V):
  return (rng.standard_normal((B, H, T, V)).astype(np.float32), rng.standard_normal((B, T, H*V)).astype(np.float32),
          rng.standard_normal(V).astype(np.float32))

class TestGDNPostCPU(unittest.TestCase):
  def test_matches_numpy(self):
    rng = np.random.default_rng(2)
    for T, toks, V in ((8, 8, 32), (8, 3, 32), (32, 20, 128), (2, 1, 64)):
      with self.subTest(T=T, toks=toks, V=V): check_post(self, *post_inputs(rng, 1, T, 4, V), toks, None)

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestGDNPostEmulator(unittest.TestCase):
  """gdn_post_nv at lanes=32 on the PYTHON emulator: V = 128 (one warp per head row: the rms reduction over 32 lanes, a q8 group per
  8 lanes) and V = 32 (4 heads per warp, 8 lanes each), a remainder chunk -- z vs fp64, the q8 words/scales/sums bit-exact"""
  def test_matches_numpy(self):
    rng = np.random.default_rng(3)
    for T, toks, V in ((32, 32, 128), (32, 11, 128), (16, 9, 32)):
      with self.subTest(T=T, toks=toks, V=V), emulated_lanes(): check_post(self, *post_inputs(rng, 1, T, 4, V), toks, "PYTHON")

# ******** the block path (GatedDeltaNetBlock._nv_prefill) on CPU against the generic graph ********

TINY = SSMConfig(conv_kernel=4, state_size=32, group_count=2, time_step_rank=4, inner_size=32*4)  # K = V = 32, 2 k-heads -> 4 v-heads
TINY_DIM = 64  # not a multiple of 128: ssm_beta/ssm_alpha stay two generic Linears on CPU (the combined f16 gemv is NV-only)

def prefill_gate(): return mock.patch("tinygrad.llm.model.nv_custom_kernels_supported", return_value=True)

def tiny_block(seed:int=0, max_context:int=64) -> GatedDeltaNetBlock:
  config = TransformerConfig(num_blocks=1, dim=TINY_DIM, hidden_dim=TINY_DIM*2, n_heads=1, n_kv_heads=1, norm_eps=1e-5, vocab_size=32,
                             head_dim=TINY_DIM, rope_theta=10000.0, rope_dim=TINY_DIM, v_head_dim=TINY_DIM, max_context=max_context,
                             ssm_layers=(True,), ssm=TINY)
  block = GatedDeltaNetBlock(config, TINY)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(block)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.3)
  block.ssm_a.replace(-block.ssm_a.abs())  # A < 0 as in trained models (A = -exp(A_log)): decay alpha < 1, a bounded state
  Tensor.realize(*params)
  return block

class TestNVPrefillBlock(unittest.TestCase):
  """GatedDeltaNetBlock._attention through _nv_prefill (prefill gate mocked open, CPU form) against the generic graph (gate closed,
  WY scan) on symbolic chunks of width 8: output and both states after every call -- full chunks, a remainder, a restart at start_pos
  0 over a nonzero resident state, a chunk ending at max_context."""
  def test_matches_generic_path(self):
    fused, generic = tiny_block(), tiny_block()
    with prefill_gate(): fused.prepare_nv_decode()
    self.assertIsNotNone(fused.nv_decode_weights)
    xs = Tensor.randn(1, 64, TINY_DIM).realize()
    v_toks, v_sp = UOp.variable("toks", 1, 8), UOp.variable("start_pos", 0, 63)
    for i, (pos, n) in enumerate(((0, 8), (8, 8), (16, 3), (0, 5), (5, 8), (56, 8))):
      with self.subTest(call=i, start_pos=pos, toks=n):
        outs = []
        for block, gate in ((generic, contextlib.nullcontext()), (fused, prefill_gate())):
          x = block.attn_norm(xs[:, pos:pos+8][:, :v_toks.bind(n)])
          block._init_state(x)
          with gate: outs.append(block._attention(x, v_sp.bind(pos)).pad_to((1, 8, TINY_DIM)).numpy()[:, :n])
        np.testing.assert_allclose(outs[1], outs[0], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(fused.conv_state.numpy(), generic.conv_state.numpy(), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(fused.recurrent_state.numpy(), generic.recurrent_state.numpy(), rtol=1e-4, atol=1e-6)

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
  def test_emulator_lanes_match_generic_path(self):
    # the real call path at the card's lane mapping: _attention on the PYTHON device, lanes=32 (shuffles emulated), against the
    # generic graph on CPU -- two symbolic 32-token chunks (32, then an 11-token remainder) from a nonzero resident state
    fused, generic = tiny_block(), tiny_block()
    for p in nn.state.get_parameters(fused): p.replace(p.to("PYTHON")).realize()
    xs = Tensor.randn(1, 64, TINY_DIM).realize()
    v_toks, v_sp = UOp.variable("toks", 1, 32), UOp.variable("start_pos", 0, 63)
    with prefill_gate(), emulated_lanes(): fused.prepare_nv_decode()
    for i, (pos, n) in enumerate(((5, 32), (37, 11))):
      with self.subTest(call=i, start_pos=pos, toks=n):
        outs = []
        for block, ctx, dev in ((generic, contextlib.nullcontext(), "CPU"), (fused, prefill_gate(), "PYTHON")):
          x = block.attn_norm(xs[:, pos:pos+32].to(dev)[:, :v_toks.bind(n)])
          block._init_state(x)
          with ctx, emulated_lanes() if dev == "PYTHON" else contextlib.nullcontext():
            outs.append(block._attention(x, v_sp.bind(pos)).pad_to((1, 32, TINY_DIM)).numpy()[:, :n])
        np.testing.assert_allclose(outs[1], outs[0], rtol=2e-3, atol=2e-5)
        np.testing.assert_allclose(fused.conv_state.numpy(), generic.conv_state.numpy(), rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(fused.recurrent_state.numpy(), generic.recurrent_state.numpy(), rtol=1e-3, atol=1e-5)

  def test_gate_closed_or_unprepared_keeps_the_generic_graph(self):
    for prepared, gate_open in ((True, False), (False, True)):
      block = tiny_block()
      if prepared:
        with prefill_gate(): block.prepare_nv_decode()
      x = Tensor.randn(1, 8, TINY_DIM).realize()[:, :UOp.variable("toks", 1, 8).bind(5)]
      block._init_state(x)
      with prefill_gate() if gate_open else contextlib.nullcontext():
        names = [u.src[0].arg.name for u in block._attention(x, 3).linear_with_vars()[0].src if isinstance(u.src[0].arg, KernelInfo)]
      self.assertNotIn("gdn_prep_nv", names, f"{prepared=} {gate_open=}")

def _gdn_model(seed:int=0) -> Transformer:
  """test_spec_decode._load_gdn_varied's six-block 27B pattern (GDN x3, attention, GDN x2) with K = V = 32 GDN heads (the NV kernels'
  lane geometry), no MTP head"""
  config = TransformerConfig(num_blocks=6, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=512, head_dim=16,
                             rope_theta=10000.0, rope_dim=16, v_head_dim=16, max_context=128, attn_output_gate=True,
                             ssm_layers=(True, True, True, False, True, True), ssm=TINY)
  model = Transformer(config)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(model)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.5)
  for b in model.blk:  # A < 0 (trained models' -exp(A_log)): with random signs the states grow to ~1e13 and every scan order disagrees
    if isinstance(b, GatedDeltaNetBlock): b.ssm_a.replace(-b.ssm_a.abs())
  Tensor.realize(*params)
  return model

class TestNVPrefillGenerate(unittest.TestCase):
  """the served path: Transformer.generate's prefill families are captured on their first call and PRUNED (T4.128) -- any kernel the
  NV prefill adds runs inside them (trap 3: the conv window must come from the real conv_state every replay). A 21-token prompt in
  chunks of 8 (8 + 8 + 5: the window crosses two chunk boundaries, the last chunk is a remainder), greedy, prefill gate open (the
  decode steps then take the same path at width 1): the pruned jit == JIT=0 bit for bit (every GDN block's states after the prompt),
  and both == the generic graph's tokens, its states at roundoff (blocks past the attention block: fp16 conv ulps)."""
  def test_jitted_chunks_match_eager_and_generic(self):
    prompt = [5, 2, 3, 4, 1, 7, 3, 9, 11, 2, 40, 100, 7, 8, 13, 21, 34, 55, 89, 144, 233]
    got = {}
    for name, jit, gate in (("jit", 1, True), ("eager", 0, True), ("generic", 1, False)):
      with Context(GDN_CHUNK=8, JIT=jit), prefill_gate() if gate else contextlib.nullcontext():
        m = _gdn_model()
        for b in m.blk:
          if isinstance(b, GatedDeltaNetBlock): b.prepare_nv_decode()
        self.assertEqual(m.blk[0].nv_decode_weights is not None, gate)
        gen = m.generate(list(prompt), chunk_size=8, temperature=0.0)
        toks = [next(gen)]
        states = [(b.conv_state.numpy().astype(np.float32), b.recurrent_state.numpy()) for b in m.blk if isinstance(b, GatedDeltaNetBlock)]
        got[name] = (toks + [t for _, t in zip(range(3), gen)], states)
        if jit: self.assertIn((True, False, 8, False), m.jit)  # the prefill family really was captured at the chunk width
    for name in ("eager", "generic"): self.assertEqual(got["jit"][0], got[name][0], name)
    for (ca, sa), (cb, sb) in zip(got["jit"][1], got["eager"][1]):
      np.testing.assert_array_equal(ca, cb)
      np.testing.assert_array_equal(sa, sb)
    for (ca, sa), (cb, sb) in zip(got["jit"][1], got["generic"][1]):
      np.testing.assert_allclose(ca, cb, rtol=5e-3, atol=1e-3)
      np.testing.assert_allclose(sa, sb, rtol=1e-3, atol=1e-3)

# ******** the sm_86 renders (no nvrtc: test_llm_nv_scan._render_cuda) ********

def _prep_sinks(T:int=128, H:int=48, Hk:int=16, K:int=128, V:int=128, toks=None):
  from test.unit.test_llm_nv_scan import _render_cuda
  C, f32 = 2*Hk*K + H*V, dtypes.float32
  sp = kernel_var(UOp.variable("start_pos", 0, 262143))
  tk = kernel_var(UOp.variable("toks", 1, T)) if toks is None else toks
  def ph(shape, dtype, slot): return UOp.placeholder(shape, dtype, slot=slot)
  bound = (sp, tk) if toks is None else (sp,)
  prep = _gdn_prep_kernel(ph((1, H, T, K), f32, 0), ph((1, H, T, K), f32, 1), ph((1, H, T, V), f32, 2), *(ph((1, H, T), f32, i) for i in (3, 4, 5)),
                          ph((1, 3, C), dtypes.float16, 6), ph((1, T, C), f32, 7), ph((1, T, 2*H), dtypes.float16, 8), ph((C, 4), f32, 9),
                          ph((H,), f32, 10), ph((H,), f32, 11), *bound, start_pos=sp, toks=tk, k_heads=Hk, qk_eps=1e-6)
  state = _gdn_conv_state_kernel(ph((1, 3, C), dtypes.float16, 0), ph((1, T, C), f32, 1), *bound, start_pos=sp, toks=tk)
  return _render_cuda(prep), _render_cuda(state)

def _post_sink(T:int=128, H:int=48, V:int=128):
  from test.unit.test_llm_nv_scan import _render_cuda
  tk, F = kernel_var(UOp.variable("toks", 1, T)), H*V
  def ph(shape, dtype, slot): return UOp.placeholder(shape, dtype, slot=slot)
  return _render_cuda(_gdn_post_kernel(ph((1, T, F), dtypes.float16, 0), ph((T, F // 32, 8), dtypes.uint32, 1), ph((T, F // 32), dtypes.float32, 2),
                                       ph((T, F // 32), dtypes.float32, 3), ph((1, H, T, V), dtypes.float32, 4), ph((1, T, F), dtypes.float32, 5),
                                       ph((V,), dtypes.float32, 6), tk, toks=tk, norm_eps=1e-6))

class TestGDNPrepRender(unittest.TestCase):
  def test_post_27b_geometry(self):
    src = _post_sink()
    self.assertEqual(re.findall(r"(?:blockIdx|threadIdx)\.[xyz]; /\* (\d+) \*/", src), ["48", "128", "32"])  # a warp per (head, token)
    self.assertIn("*((half4*)((data0_786432+alu1))) =", src)  # z stored, then read back for the q8 pack (the fp16 value q8_quantize saw)
    self.assertLess(src.index("*((half4*)((data0_786432+alu1))) ="), src.index("half4 val3 = (*((half4*)((data0_786432+alu1))));"))
    self.assertEqual(src.count("__shfl_xor_sync(0xffffffff"), 5 + 3 + 3)  # the rms over the row, the group max, the group sum
    self.assertNotIn("__syncthreads", src)

  def test_27b_geometry(self):
    for T in (128, 32):
      prep, state = _prep_sinks(T)
      with self.subTest(T=T):
        # prep: (k-head, token tile) blocks of one warp; the lane is threadIdx.x (WARP range), the shuffles reduce whole warps
        self.assertEqual(re.findall(r"(?:blockIdx|threadIdx)\.[xyz]; /\* (\d+) \*/", prep), ["16", "32", str(T // 4)])
        self.assertIn("gdn_prep_nv", prep)
        self.assertEqual(prep.count("__shfl_xor_sync(0xffffffff"), 5 * 3 * 4)  # q norm, k norm, q.k per token of the 4-token tile
        self.assertNotIn("__syncthreads", prep)
        # conv_state: one thread per channel, its old rows loaded before its stores
        self.assertEqual(re.findall(r"(?:blockIdx|threadIdx)\.[xyz]; /\* (\d+) \*/", state), ["320", "32"])
        self.assertLess(state.index("val1 ="), state.index("*(data0_30720+alu0) ="))

class TestPrefillCensus(unittest.TestCase):
  # step 0 (the map): the input copy of this harness (x is a slice, the served block input a buffer), the window zero fill (pruned on the
  # card), then the 10-31 profile's kernels in schedule order
  BEFORE = ["E_28toks2A4029_128", "E_10480_32_4", "E_320_32_3", "r_toks_16_320", "E_4_80_8_16_4_4", "q8_quantize_nv", "E_5120_32_4",
            "linear_q4_0_imma_nv", "linear_q4_0_imma_nv", "linear_f16_gemv_nv", "linear_f16_gemv_nv", "E_28toks2A8029_128", "E_6_2_8_16_4",
            "E_6_2_8_16_4", "r_128_16_16_8", "r_128_16_16_8", "E_48_2_16_16_8_4", "E_320_32_3", "E_16_16_8_8_16_3", "E_16_16_8_8_16_3",
            "r_2_8_8_16_3_32_4", "gated_delta_prefill_nv", "r_8_48_8_4_4_16", "E_32_6_2_8_16_4_4", "E_16_96_8_16_4", "q8_quantize_nv",
            "linear_q4_0_imma_nv", "r_toks_16_320", "E_4_80_8_16_4_4", "q8_quantize_nv", "linear_q4_0_imma_nv", "linear_q4_0_imma_nv",
            "E_28toks2A13629_128", "E_16_272_8_16_4", "q8_quantize_nv", "linear_q4_0_imma_nv", "E_28toks2A4029_128"]

  # step 1: the pre-scan glue (window copy/zero fill/history, conv+norm reduces, materialized q/k/v, kq, beta, alpha, conv_state store,
  # the second f16 gemv and its input copy) -> one f16 gemv over ssm_beta|ssm_alpha, gdn_prep_nv, gdn_conv_state_nv
  STEP1 = ["E_28toks2A4029_128", "r_toks_16_320", "E_4_80_8_16_4_4", "q8_quantize_nv", "linear_f16_gemv_nv", "linear_q4_0_imma_nv",
           "linear_q4_0_imma_nv", "gdn_prep_nv", "gdn_conv_state_nv", "gated_delta_prefill_nv", "r_8_48_8_4_4_16", "E_32_6_2_8_16_4_4",
           "E_16_96_8_16_4", "q8_quantize_nv", "linear_q4_0_imma_nv", "r_toks_16_320", "E_4_80_8_16_4_4", "q8_quantize_nv",
           "linear_q4_0_imma_nv", "linear_q4_0_imma_nv", "E_28toks2A13629_128", "E_16_272_8_16_4", "q8_quantize_nv", "linear_q4_0_imma_nv",
           "E_28toks2A4029_128"]

  # step 2: ssm_norm's reduce, the gated elementwise, z's pad copy and ssm_out's q8_quantize_nv -> gdn_post_nv
  STEP2 = STEP1[:10] + ["gdn_post_nv"] + STEP1[14:]

  def test_gdn_prefill_kernels(self):
    self.assertEqual(prefill_census(card_block()), self.STEP2)

  def test_q5_k_ssm_out_takes_the_fused_sums(self):
    # the card's ssm_out is Q5_K: its IMMA gemm asks q8_quantize for sums, which gdn_post_nv's registered entry carries
    from test.unit.test_llm_nv_gemm import _quantized_linear
    from tinygrad.llm.kernels.amd import Q5_K
    block = card_block()
    block.ssm_out = _quantized_linear(Q5_K, 5120, 6144)
    self.assertEqual(prefill_census(block), self.STEP2[:11] + ["linear_q5_k_imma_nv"] + self.STEP2[12:])

  def test_gate_closed_is_step_0(self):
    with mock.patch("tinygrad.llm.model.nv_gdn_prefill", side_effect=AssertionError("taken")):
      block = card_block()
      block.nv_decode_weights = None  # unprepared (e.g. kda, or a geometry the kernels don't cover): the generic graph
      self.assertEqual(prefill_census(block), self.BEFORE)

if __name__ == "__main__":
  unittest.main()
