"""T6.2: the NV fused Gated-DeltaNet prefill scan (tinygrad/llm/kernels/nv.py).
T4.98g adds: the SAME kernel also serves the T_pad==1 DECODE step (readout+state-update only -- conv1d stays
the separate, unchanged code it always was outside this kernel), behind its own GDN_NV_FUSED_DECODE gate.
T4.123f+g replaces that decode step (when GatedDeltaNetBlock.prepare_nv_decode ran) by nv.gdn_decode's two kernels; their CPU
form (lanes=1) runs here against numpy and against the generic block graph, plus the decode/prefill kernel-count checks.

Runs on CPU: gate/import checks and an sm_86 render + nvrtc compile of the kernel at the 35B's real geometry (skips when
the nvrtc lane -- docker on macOS -- is unavailable). The numerics tests need real NV hardware: mock-NV reports sm_35, so
the gate is closed there and they skip. Hardware round: DEV=NV PYTHONPATH=. python -m pytest test/unit/test_llm_nv_scan.py"""
import contextlib, functools, hashlib, itertools, math, re, struct, sys, unittest
from unittest import mock
import numpy as np
from tinygrad import Tensor, Device, Context, UOp, TinyJit, dtypes, nn
from tinygrad.device import CompileError
from tinygrad.helpers import DEV
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, KernelInfo
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, Q4_0, kernel_var
from tinygrad.llm.kernels.nv import (
  nv_custom_kernels_supported, nv_decode_kernel_supported, GDN_NV_FUSED_DECODE, gated_delta_prefill, _gated_delta_prefill_kernel,
  _gdn_conv_decode_kernel, _gdn_decode_kernel, _split_k_partials, gdn_decode, GDN_DECODE_TOKENS,
)
from tinygrad.llm.kernels import nv
from tinygrad.llm.kernels.nv_quant import nv_q8_linear
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig

GEOMETRIES = {"27b": (48, 128, 128), "35b": (32, 128, 128)}  # (v-heads, head_k_dim, head_v_dim): Qwen3.8-27B / Qwen3.6-35B-A3B
CHUNK = 32
DECODE = 1  # T_pad for a decode step

def reference(q, k, v, beta, alpha, state):
  """numpy ground truth, the recurrence from test_attention's RDNA3 parity test (alpha per (token, value-row))."""
  out, st = np.empty_like(v), state.copy()
  for t in range(q.shape[2]):
    previous, av = st.copy(), alpha[:, :, t, :, None]
    delta = (v[:, :, t] - (previous*k[:, :, t, None]).sum(-1)*alpha[:, :, t]) * beta[:, :, t, None]
    st = previous*av + delta[..., None]*k[:, :, t, None, :]
    out[:, :, t] = (previous*q[:, :, t, None]).sum(-1)*alpha[:, :, t] + delta*(q[:, :, t]*k[:, :, t]).sum(-1)[..., None]  # (1,H,1): broadcast over dv
  return out, st

def random_inputs(heads, key_dim, value_dim, tokens=CHUNK, seed=0):
  rng = np.random.default_rng(seed)
  # l2-normalized like the model's q/k before the scan: with |k| = 1 the delta rule is contractive and the fp32 state stays
  # bounded; raw normals blow the state up to ~1e13 over 32 steps and fp32 accumulation-order noise then breaks rtol=1e-4.
  q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) for x in (rng.normal(size=(1, heads, tokens, key_dim)).astype(np.float32) for _ in range(2)))
  v, beta = rng.normal(size=(1, heads, tokens, value_dim)).astype(np.float32), rng.uniform(size=(1, heads, tokens)).astype(np.float32)
  alpha = rng.uniform(0.8, 1, size=(1, heads, tokens, value_dim)).astype(np.float32)
  return q, k, v, beta, alpha, rng.normal(size=(1, heads, value_dim, key_dim)).astype(np.float32)

class TestNVGate(unittest.TestCase):
  def test_gate_closed_off_nv(self):
    for device in ("CPU", "CPU:1", "NULL", None, ("CPU", "CPU:1"), Device.DEFAULT if not Device.DEFAULT.startswith("NV") else "CPU"):
      self.assertFalse(nv_custom_kernels_supported(device), device)

  def test_env_closes_gate_without_touching_the_device(self):
    with Context(GDN_NV_FUSED=0): self.assertFalse(nv_custom_kernels_supported("NV"))  # returns before Device["NV"] is opened

class TestNVDecodeGate(unittest.TestCase):
  """T4.98g: nv_decode_kernel_supported mirrors nv_custom_kernels_supported exactly, keyed on GDN_NV_FUSED_DECODE
  instead of GDN_NV_FUSED -- same three properties: closed off-NV regardless of the flag, closed by default on
  NV, and the env check alone (never device capability) is what keeps it closed without opening Device["NV"]."""
  def test_gate_closed_off_nv(self):
    for device in ("CPU", "CPU:1", "NULL", None, ("CPU", "CPU:1"), Device.DEFAULT if not Device.DEFAULT.startswith("NV") else "CPU"):
      self.assertFalse(nv_decode_kernel_supported(device), device)

  def test_gate_closed_off_nv_even_when_flag_set(self):
    # the flag alone can't open a non-NV device's path -- device prefix is checked first either way
    with Context(GDN_NV_FUSED_DECODE=1):
      for device in ("CPU", "CPU:1", "NULL", None, ("CPU", "CPU:1")):
        self.assertFalse(nv_decode_kernel_supported(device), device)

  def test_gate_closed_by_default_on_nv_without_touching_the_device(self):
    self.assertEqual(GDN_NV_FUSED_DECODE.value, 0)
    with Context(GDN_NV_FUSED_DECODE=0): self.assertFalse(nv_decode_kernel_supported("NV"))  # returns before Device["NV"] is opened

  def test_independent_of_the_prefill_gate(self):
    # GDN_NV_FUSED=0 (the pooled recipe's standing prefill setting, see CLAUDE.md) must not also close the
    # decode-only gate -- and vice versa, turning decode on must not require prefill fusion. Both checks stay
    # off-device (device prefix != "NV") so this never opens Device["NV"] on this Mac.
    with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=0): self.assertFalse(nv_decode_kernel_supported("CPU"))
    # nv_custom_kernels_supported (prefill) itself stays False here regardless of the decode flag
    with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=1): self.assertFalse(nv_custom_kernels_supported("CPU"))

class TestNVKernelSource(unittest.TestCase):
  def test_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    heads, key_dim, value_dim = GEOMETRIES["35b"]
    shapes = [(1, heads, CHUNK, value_dim), (1, heads, CHUNK, key_dim), (1, heads, CHUNK, key_dim), (1, heads, CHUNK, value_dim),
              (1, heads, CHUNK), (1, heads, CHUNK, value_dim), (1, heads, value_dim, key_dim), (1, heads, CHUNK)]
    sink = _gated_delta_prefill_kernel(*(UOp.placeholder(s, dtypes.float32, slot=i) for i, s in enumerate(shapes)))
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")  # macOS: needs the docker compile server
    try: prg = to_program(sink, renderer)
    except CompileError: raise
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
    self.assertIn("gated_delta_prefill_nv", src)
    self.assertIn("__shfl_xor_sync(0xffffffff", src)
    self.assertNotIn("amdgcn", src)
    self.assertTrue(next(u.arg for u in prg.src if u.op is Ops.BINARY).startswith(b"\x7fELF"))  # cubin for sm_86, not ptx

  def test_renders_and_compiles_for_sm86_at_decode_shape(self):
    # T4.98g: the IDENTICAL kernel builder at tokens==1 (T_pad==1 decode) instead of CHUNK -- same kernel name
    # (it's the same function/cubin family, just a different cached shape), same shuffle reduce, still a valid
    # cubin. The grid (batch*heads*value_dim/row_tile) is unaffected by tokens -- only the REDUCE loop trip
    # count (the token loop) drops to 1. Covers both real geometries (27b's row_tile grid included).
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    for name, (heads, key_dim, value_dim) in GEOMETRIES.items():
      with self.subTest(geometry=name):
        shapes = [(1, heads, DECODE, value_dim), (1, heads, DECODE, key_dim), (1, heads, DECODE, key_dim), (1, heads, DECODE, value_dim),
                  (1, heads, DECODE), (1, heads, DECODE, value_dim), (1, heads, value_dim, key_dim), (1, heads, DECODE)]
        sink = _gated_delta_prefill_kernel(*(UOp.placeholder(s, dtypes.float32, slot=i) for i, s in enumerate(shapes)))
        try: prg = to_program(sink, renderer)
        except CompileError: raise
        except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
        src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
        self.assertIn("gated_delta_prefill_nv", src)
        self.assertIn("__shfl_xor_sync(0xffffffff", src)
        self.assertNotIn("amdgcn", src)
        self.assertTrue(next(u.arg for u in prg.src if u.op is Ops.BINARY).startswith(b"\x7fELF"))  # cubin for sm_86, not ptx

class TestFusedMathMatchesModelLoopAtDecode(unittest.TestCase):
  """T4.98g, GPU-free (no kernel execution, no device opened -- see the module docstring). `reference()` above
  is a numpy transcription of exactly what _gated_delta_prefill_kernel computes (generic in `tokens`; checked
  against the COMPILED kernel at chunk=32 by TestNVFusedScanParity, which needs real hardware). This instead
  checks reference() against GatedDeltaNetBlock._attention's own per-token LOOP body (model.py's run_scan,
  T_pad==1 branch -- copied verbatim below) at tokens=1, using real Tensor ops on DEV=CPU: i.e. that reusing
  the fused kernel for decode is sound relative to the actual production recurrence, not merely self-consistent
  in isolation. (The two formulas are algebraically identical but reassociate the float ops differently --
  reference() reads out from the PRE-update state via the precomputed q.k dot, exactly like the kernel's own
  `state_q*av + delta*kq` -- so the tolerance below matches the module's existing chunk=32 hardware check.)"""

  @staticmethod
  def _loop_step(state:Tensor, q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor) -> tuple[Tensor, Tensor]:
    # verbatim copy of model.py GatedDeltaNetBlock._attention's run_scan loop body (T4.69a) for t=0 of a
    # T_pad==1 call. q/k/v/beta/alpha/state are in the same (B,H,T,dim) layout gated_delta_prefill takes.
    q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
    alpha = alpha.unsqueeze(-1)
    s1 = state * alpha[:, :, 0]  # decay the state
    delta = (v[:, :, 0] - (s1*k[:, :, 0]).sum(-1, keepdim=True)) * beta[:, :, 0]  # the delta rule update
    state = s1 + delta * k[:, :, 0]
    out = (state * q[:, :, 0]).sum(-1)
    return state, out

  def test_reference_matches_loop_body_at_one_token(self):
    for name, (heads, key_dim, value_dim) in GEOMETRIES.items():
      for seed in range(3):  # a few random incoming (nonzero) states -- decode always resumes from a real state
        with self.subTest(geometry=name, seed=seed):
          q, k, v, beta, alpha, initial = random_inputs(heads, key_dim, value_dim, tokens=DECODE, seed=seed)
          expected_out, expected_state = reference(q, k, v, beta, alpha, initial)
          got_state, got_out = self._loop_step(Tensor(initial), Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha))
          np.testing.assert_allclose(got_out.numpy(), expected_out[:, :, 0], rtol=1e-4, atol=1e-4)
          np.testing.assert_allclose(got_state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

@unittest.skipUnless(nv_custom_kernels_supported(Device.DEFAULT), "real NV hardware with the gate open required")
class TestNVFusedScanParity(unittest.TestCase):
  def test_kernel_matches_numpy(self):
    for name, (heads, key_dim, value_dim) in GEOMETRIES.items():
      # T4.98g: DECODE (tokens=1) alongside CHUNK -- the identical kernel call, no dispatch/gate involved here
      # (this calls gated_delta_prefill directly), so this is the direct hardware check that the compiled
      # kernel's own numbers are right at tokens=1, complementing test_decode_step_matches_loop's whole-block one.
      for tokens in (CHUNK, DECODE):
        with self.subTest(geometry=name, tokens=tokens):
          q, k, v, beta, alpha, initial = random_inputs(heads, key_dim, value_dim, tokens=tokens)
          expected_out, expected_state = reference(q, k, v, beta, alpha, initial)
          state = Tensor(initial).contiguous().realize()
          out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
          np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
          np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

  def test_start_pos_resets_state(self):
    heads, key_dim, value_dim = GEOMETRIES["35b"]
    q, k, v, beta, alpha, initial = random_inputs(heads, key_dim, value_dim, seed=1)
    for pos, start in ((0, np.zeros_like(initial)), (CHUNK, initial)):  # start_pos == 0 must ignore the resident state
      with self.subTest(start_pos=pos):
        expected_out, expected_state = reference(q, k, v, beta, alpha, start)
        start_pos = Tensor(UOp.variable("start_pos", 0, 4096).bind(pos))
        # the kernel reads the Variable by name; its BIND has to be in the same schedule. The model carries it through the
        # state's AFTER chain (the conv-state store uses start_pos), so mirror that: the realized state buffer, ordered after an
        # unrealized kernel that contains the bound variable.
        state = Tensor(initial).contiguous().realize()
        state = Tensor(state.uop.after((start_pos * 0).float().contiguous().uop))
        out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state, start_pos).realize()
        np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

  def test_block_matches_wy_scan(self):
    from test.unit.test_gdn_scan_parity import GEOMETRIES as BLOCK_GEOMETRIES, DIM, make_block, run_attention, snapshot
    for name, ssm in BLOCK_GEOMETRIES.items():
      with self.subTest(geometry=name):
        x = (Tensor.randn(1, CHUNK, DIM) * 0.1).realize()
        block_fused, block_wy = make_block(ssm), make_block(ssm)
        out_fused = run_attention(block_fused, x, 0)
        with Context(GDN_NV_FUSED=0): out_wy = run_attention(block_wy, x, 0)
        np.testing.assert_allclose(out_fused, out_wy, rtol=1e-3, atol=1e-4)
        for got, want in zip(snapshot(block_fused), snapshot(block_wy)): np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)

  def test_decode_step_matches_loop(self):
    # T4.98g hardware round: a real T_pad==1 (decode) block call with GDN_NV_FUSED_DECODE=1 (prefill fusion OFF,
    # proving the two gates are independent) must match the plain per-token loop -- same idea as
    # test_block_matches_wy_scan above, at tokens=1 instead of CHUNK, continuing from a real nonzero state (a
    # fresh 8-token prefill) since that's the regime decode actually runs in.
    from test.unit.test_gdn_scan_parity import GEOMETRIES as BLOCK_GEOMETRIES, DIM, make_block, run_attention, snapshot
    for name, ssm in BLOCK_GEOMETRIES.items():
      with self.subTest(geometry=name):
        prefix = (Tensor.randn(1, 8, DIM) * 0.1).realize()
        step = (Tensor.randn(1, DECODE, DIM) * 0.1).realize()
        block_fused, block_loop = make_block(ssm), make_block(ssm)
        with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=0):
          run_attention(block_fused, prefix, 0)
          run_attention(block_loop, prefix, 0)
        for got, want in zip(snapshot(block_fused), snapshot(block_loop)): np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)
        with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=1): out_fused = run_attention(block_fused, step, 8)
        with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=0): out_loop = run_attention(block_loop, step, 8)
        np.testing.assert_allclose(out_fused, out_loop, rtol=1e-3, atol=1e-4)
        for got, want in zip(snapshot(block_fused), snapshot(block_loop)): np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)

  def test_two_kernel_decode_matches_loop(self):
    # T4.123f+g hardware round: the same check through nv.gdn_decode (prepare_nv_decode under the gate), 4 decode steps then a
    # restart at start_pos 0 -- z is fp16 before ssm_out, hence the looser output tolerance
    from test.unit.test_gdn_scan_parity import GEOMETRIES as BLOCK_GEOMETRIES, DIM, make_block, run_attention, snapshot
    for name, ssm in BLOCK_GEOMETRIES.items():
      with self.subTest(geometry=name):
        steps = (Tensor.randn(1, 5, DIM) * 0.1).realize()
        block_fused, block_loop = make_block(ssm), make_block(ssm)
        with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=1): block_fused.prepare_nv_decode()
        self.assertIsNotNone(block_fused.nv_decode_weights)
        for t, pos in enumerate((0, 1, 2, 3, 0)):
          with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=1): out_fused = run_attention(block_fused, steps[:, t:t+1], pos)
          with Context(GDN_NV_FUSED=0, GDN_NV_FUSED_DECODE=0): out_loop = run_attention(block_loop, steps[:, t:t+1], pos)
          np.testing.assert_allclose(out_fused, out_loop, rtol=1e-2, atol=1e-3)
          for got, want in zip(snapshot(block_fused), snapshot(block_loop)): np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)

# ******** T4.123f+g: the GDN decode step in two kernels (nv.gdn_decode) -- CPU form (lanes=1) + sm_86 render ********

TINY = SSMConfig(conv_kernel=4, state_size=32, group_count=2, time_step_rank=4, inner_size=32*4)  # 2 k-heads -> 4 v-heads of 32
TINY_DIM = 8

def tiny_block(seed:int=0, dim:int=TINY_DIM) -> GatedDeltaNetBlock:
  config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=dim*2, n_heads=1, n_kv_heads=1, norm_eps=1e-5, vocab_size=32, head_dim=dim,
                             rope_theta=10000.0, rope_dim=dim, v_head_dim=dim, max_context=64, ssm_layers=(True,), ssm=TINY)
  block = GatedDeltaNetBlock(config, TINY)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(block)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.3)
  Tensor.realize(*params)  # pin the weights now (test_gdn_scan_parity.make_block's RNG-counter note)
  return block

def decode_gate(): return mock.patch("tinygrad.llm.model.nv_decode_kernel_supported", return_value=True)

def kernel_names(t:Tensor) -> list[str]:
  """the schedule's kernels: custom kernels by name, generic ones (the default KernelInfo name until codegen) as '?'"""
  linear, _ = t.linear_with_vars()
  return [u.src[0].arg.name if isinstance(u.src[0].arg, KernelInfo) and u.src[0].arg.name != KernelInfo().name else "?" for u in linear.src]

def _copy_kernel(out:UOp, src:UOp) -> UOp:
  i = UOp.range(math.prod(out.shape), 0)
  return out.flatten()[i].store(src.flatten()[i].load()).end(i).sink(arg=KernelInfo(name="copy", opts_to_apply=()))

def linear_output(partials:np.ndarray) -> Tensor:
  """(B,F,P) split-K partials as nv_q8_linear hands them out: a kernel output summed lazily, (B,1,F)"""
  out = Tensor.custom_kernel(Tensor.empty(*partials.shape), Tensor(partials).realize(), fxn=_copy_kernel)[0]
  return out.sum(-1).reshape(partials.shape[0], 1, partials.shape[1])

def decode_reference(qkv, gate, beta, alpha, conv_state, state, conv_w, norm_w, dt_bias, ssm_a, start_pos, k_heads, qk_eps=1e-6, norm_eps=1e-5):
  """numpy ground truth of one GatedDeltaNet decode token from the projections (model.py _attention at T=1, the per-token loop's
  algebra): returns z (B,1,H*V) before ssm_out, the new conv_state and the new recurrent state."""
  B, H, V, K = state.shape
  if start_pos == 0: conv_state, state = np.zeros_like(conv_state), np.zeros_like(state)
  window = np.concatenate([conv_state.astype(np.float32), qkv.sum(-1)[:, None]], axis=1)  # (B, taps, C)
  conv = (window * conv_w.T[None]).sum(1)
  conv = conv / (1 + np.exp(-conv))
  q_dim = k_heads*K
  q, k = conv[:, :q_dim].reshape(B, k_heads, K), conv[:, q_dim:2*q_dim].reshape(B, k_heads, K)
  q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), qk_eps) * K**-0.5
  k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), qk_eps)
  q, k = np.tile(q, (1, H//k_heads, 1)), np.tile(k, (1, H//k_heads, 1))  # .repeat(): v-head h reads k-head h % k_heads
  v = conv[:, 2*q_dim:].reshape(B, H, V)
  b, a = 1/(1 + np.exp(-beta)), np.exp(np.logaddexp(alpha + dt_bias, 0) * ssm_a)
  decayed = state * a[..., None, None]
  delta = (v - (decayed * k[:, :, None]).sum(-1)) * b[..., None]
  new_state = decayed + delta[..., None] * k[:, :, None]
  core = (new_state * q[:, :, None]).sum(-1)
  g = gate.sum(-1).reshape(B, H, V)
  z = core / np.sqrt((core**2).mean(-1, keepdims=True) + norm_eps) * norm_w * g / (1 + np.exp(-g))
  return z.reshape(B, 1, H*V), window[:, 1:], new_state

class TestGDNDecodeKernelSource(unittest.TestCase):
  def test_renders_and_compiles_for_sm86(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    sp = kernel_var(UOp.variable("start_pos", 0, 262143))
    for name, (heads, key_dim, value_dim) in GEOMETRIES.items():
      k_heads, parts = 16, 5
      channels = 2*k_heads*key_dim + heads*value_dim
      def ph(shape, dtype, slot): return UOp.placeholder(shape, dtype, slot=slot)
      conv = _gdn_conv_decode_kernel(ph((1, channels), dtypes.float32, 0), ph((1, 3, channels), dtypes.float16, 1),
                                     ph((1, channels, parts), dtypes.float32, 2), ph((channels, 4), dtypes.float32, 3), sp, start_pos=sp)
      # T4.127b: beta_alpha is ONE (1, 2*heads) buffer now (beta at [:heads], alpha at [heads:]), replacing the two
      # separate (1, heads) placeholders -- every later slot shifts down by one.
      step = _gdn_decode_kernel(ph((1, heads*value_dim), dtypes.float16, 0), ph((1, heads, value_dim, key_dim), dtypes.float32, 1),
        ph((1, channels), dtypes.float32, 2), ph((1, 2*heads), dtypes.float16, 3), ph((heads,), dtypes.float32, 4),
        ph((heads,), dtypes.float32, 5), ph((1, heads*value_dim, parts), dtypes.float32, 6), ph((value_dim,), dtypes.float32, 7), sp,
        start_pos=sp, k_heads=k_heads, qk_eps=1e-6, norm_eps=1e-6)
      for sink, kname, threads in ((conv, "gdn_conv_decode_nv", [32]), (step, "gdn_decode_nv", [32, value_dim//8])):
        with self.subTest(geometry=name, kernel=kname):
          try: prg = to_program(sink, renderer)
          except CompileError: raise
          except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
          src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
          self.assertIn(kname, src)
          # the lane must be threadIdx.x (stride 1 over the state's key columns) for __shfl_xor_sync to reduce one row
          self.assertEqual([int(n) for n in re.findall(r"threadIdx\.[xyz]; /\* (\d+) \*/", src)], threads)
          if kname == "gdn_decode_nv":
            self.assertIn("__shfl_xor_sync(0xffffffff", src)
            self.assertIn("__syncthreads()", src)  # the rms block reduction
          self.assertTrue(next(u.arg for u in prg.src if u.op is Ops.BINARY).startswith(b"\x7fELF"))

def _decode_sinks(tokens:int=1, track:bool=False):
  """both kernels' sinks at the 27B geometry (heads 48, k-heads 16, K=V=128, 5 split-K parts) for a run of `tokens` rows; track=True
  adds T4.132's per-token state outputs (3 slots, before the bound start_pos as gdn_decode passes them)"""
  sp = kernel_var(UOp.variable("start_pos", 0, 262143))
  heads, key_dim, value_dim, k_heads, parts = 48, 128, 128, 16, 5
  channels = 2*k_heads*key_dim + heads*value_dim
  def ph(shape, dtype, slot): return UOp.placeholder(shape, dtype, slot=slot)
  conv_track = (ph((3, 1, 3, channels), dtypes.float16, 4),) if track else ()
  state_track = (ph((3, 1, heads, value_dim, key_dim), dtypes.float32, 8),) if track else ()
  conv = _gdn_conv_decode_kernel(ph((tokens, channels), dtypes.float32, 0), ph((1, 3, channels), dtypes.float16, 1),
                                 ph((tokens, channels, parts), dtypes.float32, 2), ph((channels, 4), dtypes.float32, 3), *conv_track, sp,
                                 start_pos=sp, tokens=tokens)
  step = _gdn_decode_kernel(ph((tokens, heads*value_dim), dtypes.float16, 0), ph((1, heads, value_dim, key_dim), dtypes.float32, 1),
    ph((tokens, channels), dtypes.float32, 2), ph((tokens, 2*heads), dtypes.float16, 3), ph((heads,), dtypes.float32, 4),
    ph((heads,), dtypes.float32, 5), ph((tokens, heads*value_dim, parts), dtypes.float32, 6), ph((value_dim,), dtypes.float32, 7),
    *state_track, sp, start_pos=sp, k_heads=k_heads, qk_eps=1e-6, norm_eps=1e-6, tokens=tokens)
  return conv, step

def _render_cuda(sink:UOp) -> str:
  """the sm_86 CUDA source of `sink` WITHOUT compiling it (no nvrtc lane needed): to_program's own linearize + render patterns only"""
  from tinygrad.codegen import full_rewrite_to_sink, pm_to_program, ProgramInfo
  from tinygrad.uop.ops import PatternMatcher, graph_rewrite
  from tinygrad.renderer.cstyle import CUDARenderer
  # T4.132: no compiler process either -- on macOS the renderer's NVRTCCompiler starts the docker nvrtc server in __init__
  with mock.patch("tinygrad.runtime.support.compiler_cuda._get_server", return_value=None): renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
  full = full_rewrite_to_sink(sink, renderer)
  prg = UOp(Ops.PROGRAM, src=(full,), arg=ProgramInfo.from_sink(full, renderer.target))
  prg = graph_rewrite(prg, PatternMatcher([pat for pat in pm_to_program.patterns if pat[1].__name__ != "do_compile"]), ctx=renderer)
  return next(u.arg for u in prg.src if u.op is Ops.SOURCE)

class TestGDNDecodeSmallT(unittest.TestCase):
  """T4.132: the two decode kernels at tokens <= GDN_DECODE_TOKENS (speculative decoding's VERIFY/REDO run), sequential inside the
  kernel. The tokens == 1 render is the standing decode kernel and must not move: its sm_86 source md5 is pinned (rendered without the
  nvrtc lane, _render_cuda) against the pre-T4.132 tree."""
  T1_MD5 = {"gdn_conv_decode_nv": "5bf0a5d2620d2e6472bc42e5df8ad978", "gdn_decode_nv": "23d286ac10b98d6f44ccc47e96ad68e8"}

  def test_t1_render_unchanged(self):
    conv, step = _decode_sinks(1)
    for sink, name in ((conv, "gdn_conv_decode_nv"), (step, "gdn_decode_nv")):
      with self.subTest(kernel=name):
        src = _render_cuda(sink)
        self.assertIn(name, src)
        self.assertEqual(hashlib.md5(src.encode()).hexdigest(), self.T1_MD5[name], f"the T == 1 {name} render changed")

  def test_small_t_renders(self):
    for tokens in range(2, GDN_DECODE_TOKENS + 1):
      for track in (False, True):  # T4.132: the per-token state outputs of a speculative VERIFY
        conv, step = _decode_sinks(tokens, track)
        with self.subTest(tokens=tokens, track=track):
          conv_src, step_src = _render_cuda(conv), _render_cuda(step)
          self.assertEqual([int(n) for n in re.findall(r"threadIdx\.[xyz]; /\* (\d+) \*/", step_src)], [32, 16])  # same block shape as T == 1
          self.assertEqual(step_src.count("__syncthreads()"), tokens)  # one rms block reduction per token
          self.assertEqual(conv_src.count("__syncthreads()"), 0)
          self.assertEqual("data4_92160" in conv_src, track)  # (3,1,3,10240) fp16 conv slots
          self.assertEqual("data8_2359296" in step_src, track)  # (3,1,48,128,128) fp32 state slots
          # T4.140: token t's loads come after token t-1's stores, so each stretch between barriers holds one token's stores and the next
          # token's gate partials (8 rows x 5 parts as float4s) -- not every load up front and every store at the end, which spilled
          segs = step_src.split("__syncthreads()")
          self.assertEqual([len(re.findall(r"\(data6_\d+\+", seg)) for seg in segs], [10]*tokens + [0])
          self.assertEqual([seg.count("((half4*)((data0_") for seg in segs], [0] + [2]*tokens)  # z: 8 rows as two half4 stores
          self.assertEqual([seg.count("*(data8_2359296+") for seg in segs], [0] + [32*track]*(tokens-1) + [0])

  def test_verify_compiles_without_spills(self):
    # T4.140: the served VERIFY widths (T = 3, 4 with track) compiled by the nvrtc lane: ptxas must fit them in the 128 registers of a
    # 512-thread block, as T == 1 -- before the per-token ordering their stack frames were 584 / 1368 B per thread (spills: 130 us at T=4)
    from tinygrad.codegen import to_program
    from tinygrad.renderer.cstyle import CUDARenderer
    from tinygrad.runtime.support.elf import elf_loader
    from tinygrad.runtime.ops_nv import NVProgram
    try: renderer = CUDARenderer(DEV.target("NV", arch="sm_86"))
    except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
    for tokens in (3, 4):
      with self.subTest(tokens=tokens):
        try: prg = to_program(_decode_sinks(tokens, track=True)[1], renderer)
        except CompileError: raise
        except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
        _, sections, _ = elf_loader(next(u.arg for u in prg.src if u.op is Ops.BINARY))
        info = {param: struct.unpack_from("II", data)[1] for sh in sections if sh.name == ".nv.info"
                for _, param, data in NVProgram._parse_elf_info(None, sh) if param in (0x12, 0x2f)}  # EIATTR_MIN_STACK_SIZE, _REGCOUNT
        self.assertEqual((info[0x2f], info[0x12]), (128, 0))

class TestGDNDecodeMatchesNumpy(unittest.TestCase):
  """nv.gdn_decode's two kernels in their CPU form (lanes=1: the same graphs minus shuffles and shared memory) against
  decode_reference, from split-K partials (P=3 qkv, P=2 gate) and a nonzero incoming conv/recurrent state."""
  def test_matches_reference(self):
    B, Hk, H, K, V = 1, 2, 4, 32, 32
    C = 2*Hk*K + H*V
    rng = np.random.default_rng(0)
    def f32(*s, scale=1.0): return (rng.standard_normal(s) * scale).astype(np.float32)
    qkv, gate, beta, alpha = f32(B, C, 3, scale=0.5), f32(B, H*V, 2), f32(B, 1, H), f32(B, 1, H)
    beta_alpha = np.concatenate([beta, alpha], axis=-1)  # T4.127b: gdn_decode's ONE combined (B,1,2*heads) input
    conv0, state0 = f32(B, 3, C).astype(np.float16), f32(B, H, V, K, scale=0.3)
    conv_w, norm_w, dt_bias, ssm_a = f32(C, 4, scale=0.5), f32(V), f32(H), -np.abs(f32(H))
    for pos in (5, 0):
      with self.subTest(start_pos=pos):
        conv_state, state = Tensor(conv0).contiguous().realize(), Tensor(state0).contiguous().realize()
        z = gdn_decode(linear_output(qkv), linear_output(gate), Tensor(beta_alpha), conv_state, state, Tensor(conv_w), Tensor(norm_w),
                       Tensor(dt_bias), Tensor(ssm_a), UOp.variable("start_pos", 0, 63).bind(pos), k_heads=Hk, qk_eps=1e-6, norm_eps=1e-5)
        want_z, want_conv, want_state = decode_reference(qkv, gate, beta[:, 0], alpha[:, 0], conv0, state0, conv_w, norm_w, dt_bias, ssm_a, pos, Hk)
        self.assertEqual(z.dtype, dtypes.float16)
        np.testing.assert_allclose(z.numpy().astype(np.float32), want_z, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(conv_state.numpy().astype(np.float32), want_conv.astype(np.float16).astype(np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(state.numpy(), want_state, rtol=1e-5, atol=1e-6)

  def test_small_t_matches_sequential_reference(self):
    # T4.132: a T-token run == T chained single-token references, with the conv window carried in fp32 across the run (the kernel
    # and the generic concrete-T chunk path both round conv_state to fp16 once, at the end; T separate decode STEPS would round it
    # between tokens -- the same difference plain generate() vs a VERIFY chunk already have, see the T4.132 report). z fp16 at the
    # single-token test's tolerance, conv_state exact, state at fp32 round-off.
    B, Hk, H, K, V = 1, 2, 4, 32, 32
    C = 2*Hk*K + H*V
    rng = np.random.default_rng(1)
    def f32(*s, scale=1.0): return (rng.standard_normal(s) * scale).astype(np.float32)
    conv0, state0 = f32(B, 3, C).astype(np.float16), f32(B, H, V, K, scale=0.3)
    conv_w, norm_w, dt_bias, ssm_a = f32(C, 4, scale=0.5), f32(V), f32(H), -np.abs(f32(H))
    for T in range(2, GDN_DECODE_TOKENS + 1):
      for pos in (5, 0):
        with self.subTest(tokens=T, start_pos=pos):
          qkv, gate, beta, alpha = f32(T, C, 3, scale=0.5), f32(T, H*V, 2), f32(T, H), f32(T, H)
          conv_state, state = Tensor(conv0).contiguous().realize(), Tensor(state0).contiguous().realize()
          def rows(p): return Tensor.custom_kernel(Tensor.empty(*p.shape), Tensor(p).realize(), fxn=_copy_kernel)[0].sum(-1).reshape(B, T, p.shape[1])
          z = gdn_decode(rows(qkv), rows(gate), Tensor(np.concatenate([beta, alpha], -1)).reshape(B, T, 2*H), conv_state, state,
                         Tensor(conv_w), Tensor(norm_w), Tensor(dt_bias), Tensor(ssm_a), UOp.variable("start_pos", 0, 63).bind(pos),
                         k_heads=Hk, qk_eps=1e-6, norm_eps=1e-5)
          want_z, conv, st = [], conv0.astype(np.float32), state0
          for t in range(T):
            zt, conv, st = decode_reference(qkv[t:t+1], gate[t:t+1], beta[t:t+1], alpha[t:t+1], conv, st, conv_w, norm_w, dt_bias, ssm_a, pos + t, Hk)
            want_z.append(zt)
          self.assertEqual(z.shape, (B, T, H*V))
          np.testing.assert_allclose(z.numpy().astype(np.float32), np.concatenate(want_z, axis=1), rtol=5e-3, atol=1e-3)
          np.testing.assert_allclose(conv_state.numpy().astype(np.float32), conv.astype(np.float16).astype(np.float32), rtol=0, atol=0)
          np.testing.assert_allclose(state.numpy(), st, rtol=1e-5, atol=1e-6)

# ******** T4.140: the decode kernels' card form (lanes=32) EXECUTED on the PYTHON emulator ********

_emu_slot = itertools.count(300)
def _emulated_warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  """nv.warp_reduce with each __shfl_xor_sync(v, off) as a block-shared exchange for the PYTHON emulator (no CUDA intrinsics): thread
  wave*32 + lane writes v and reads [.. ^ off] (test_llm_nv_norm's _emulated_warp_reduce; here the lane is LOCAL axis 1 and the wave LOCAL
  axis 2). A sum that does not depend on the wave (token 0's q/k norms) is the same in every wave, so one 32-slot exchange serves them all.
  The emulator runs the block in lockstep, so the store is always seen."""
  assert full_wave and not maximum
  rngs = {u.arg[0]: u for u in val.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.LOCAL}
  lane, wave = rngs[1], rngs.get(2)
  tid, n = (lane, 32) if wave is None else (wave*32 + lane, wave.src[0].arg*32)
  for off in (16, 8, 4, 2, 1):
    buf = UOp.placeholder((n,), dtypes.float32, slot=next(_emu_slot), addrspace=AddrSpace.LOCAL)
    val = val + buf.after(buf[tid].store(val))[tid + off - (tid // off) % 2 * (2*off)].load()  # tid ^ off, z3-boundable
  return val

def _card_form(builder):
  """a decode kernel builder in the card's form (gdn_decode passes lanes=1 off NV), uncached so the emulated shuffle is built in"""
  return lambda *args, lanes, **kwargs: builder.__wrapped__(*args, lanes=32, **kwargs)

@unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator has no float16 before 3.12")
class TestGDNDecodeOnEmulator(unittest.TestCase):
  """T4.140: nv.gdn_decode's two kernels in the card's form -- 32-lane warps, value_dim/8 warps per block, the xor-shuffle dots, the rms
  block reduction through per-token LOCAL slots and a barrier, the lane-0 z store gate, the speculative track stores -- EXECUTED on the
  PYTHON emulator through gdn_decode itself, only the shuffle swapped for a shared exchange and the lane count forced. Odd batch (3), two
  key columns per lane, start_pos 0 (zeroed state) and > 0, T = 1..GDN_DECODE_TOKENS with track buffers:
  (a) a T-token run is BIT-IDENTICAL to T chained single-token calls -- z, conv_state, state and every track slot (the state after call
      t+1) -- so every T runs the T == 1 kernel's arithmetic (md5-pinned in TestGDNDecodeSmallT). The qkv partials are multiples of 1/16:
      their sums are exact in fp16, so the chain's conv_state rounding between calls loses nothing.
  (b) both against an fp64 numpy reference (decode_reference chained, conv window exact)."""
  B, Hk, H, K, V, P = 3, 2, 4, 64, 32, 3

  def _run(self, qkv, gate, beta_alpha, conv0, state0, weights, pos:int, T:int):
    B, H, V = self.B, self.H, self.V
    dev = "PYTHON"
    def rows(p): return Tensor.custom_kernel(Tensor.empty(*p.shape, device=dev), Tensor(p, device=dev).realize(), fxn=_copy_kernel)[0] \
                         .sum(-1).reshape(B, T, p.shape[1])
    conv_state, state = Tensor(conv0, device=dev).contiguous().realize(), Tensor(state0, device=dev).contiguous().realize()
    track = (Tensor.zeros(GDN_DECODE_TOKENS-1, *conv0.shape, dtype=dtypes.float16, device=dev).contiguous().realize(),
             Tensor.zeros(GDN_DECODE_TOKENS-1, *state0.shape, device=dev).contiguous().realize())
    with mock.patch.object(nv, "warp_reduce", _emulated_warp_reduce), \
         mock.patch.object(nv, "_gdn_decode_kernel", _card_form(nv._gdn_decode_kernel)), \
         mock.patch.object(nv, "_gdn_conv_decode_kernel", _card_form(nv._gdn_conv_decode_kernel)):
      z = gdn_decode(rows(qkv), rows(gate), Tensor(beta_alpha, device=dev).reshape(B, T, 2*H), conv_state, state,
                     *(Tensor(w, device=dev) for w in weights), UOp.variable("start_pos", 0, 63).bind(pos), k_heads=self.Hk, qk_eps=1e-6,
                     norm_eps=1e-5, track=track if T > 1 else None)
      return z.numpy().reshape(B, T, H*V), conv_state.numpy(), state.numpy(), track[0].numpy()[:T-1], track[1].numpy()[:T-1]

  def test_run_bit_identical_to_chain_and_matches_fp64(self):
    B, Hk, H, K, V, P = self.B, self.Hk, self.H, self.K, self.V, self.P
    C = 2*Hk*K + H*V
    rng = np.random.default_rng(140)
    def f32(*s, scale=1.0): return (rng.standard_normal(s) * scale).astype(np.float32)
    conv0, state0 = f32(B, 3, C).astype(np.float16), f32(B, H, V, K, scale=0.3)
    weights = (f32(C, 4, scale=0.5), f32(V), f32(H), -np.abs(f32(H)))  # conv, ssm_norm, dt_bias, ssm_a
    for T in range(1, GDN_DECODE_TOKENS + 1):
      qkv = (rng.integers(-24, 25, (B*T, C, P)) / 16).astype(np.float32)
      gate, beta_alpha = f32(B*T, H*V, 2), f32(B*T, 2*H)
      for pos in (0, 5):
        with self.subTest(tokens=T, start_pos=pos):
          z, conv, state, conv_tr, state_tr = self._run(qkv, gate, beta_alpha, conv0, state0, weights, pos, T)
          # (b) fp64: chained single-token references, the conv window carried exactly
          want_z, w_conv, w_state = [], conv0.astype(np.float64), state0.astype(np.float64)
          for t in range(T):
            r = slice(t, B*T, T)  # (b, t) rows
            zt, w_conv, w_state = decode_reference(*(x.astype(np.float64) for x in (qkv[r], gate[r], beta_alpha[r, :H], beta_alpha[r, H:])),
                                                   w_conv, w_state, *(w.astype(np.float64) for w in weights), pos + t, Hk)
            want_z.append(zt)
            if t < T-1:
              np.testing.assert_array_equal(conv_tr[t], w_conv.astype(np.float16), f"conv track slot {t}")
              np.testing.assert_allclose(state_tr[t], w_state, rtol=1e-5, atol=1e-5, err_msg=f"state track slot {t}")
          np.testing.assert_allclose(z, np.concatenate(want_z, axis=1), rtol=5e-3, atol=2e-3)
          np.testing.assert_array_equal(conv, w_conv.astype(np.float16))
          np.testing.assert_allclose(state, w_state, rtol=1e-5, atol=1e-5)
          if T == 1: continue
          # (a) the chain: T single-token calls, each starting from the previous one's conv_state / state
          c_conv, c_state, c_z = conv0, state0, []
          for t in range(T):
            r = slice(t, B*T, T)
            zt, c_conv, c_state, _, _ = self._run(qkv[r], gate[r], beta_alpha[r], c_conv, c_state, weights, pos + t, 1)
            c_z.append(zt)
            if t < T-1:
              np.testing.assert_array_equal(conv_tr[t], c_conv, f"conv track slot {t} vs the chain")
              np.testing.assert_array_equal(state_tr[t], c_state, f"state track slot {t} vs the chain")
          np.testing.assert_array_equal(z, np.concatenate(c_z, axis=1))
          np.testing.assert_array_equal(conv, c_conv)
          np.testing.assert_array_equal(state, c_state)

class TestGDNDecodeBlock(unittest.TestCase):
  """GatedDeltaNetBlock._attention at T=1 through nv.gdn_decode (gate mocked open, CPU form) against the generic graph: output and
  both states after every step, over decode steps and a restart at start_pos 0 with a nonzero resident state."""
  def test_matches_generic_path(self):
    fused, generic = tiny_block(), tiny_block()
    with decode_gate(): fused.prepare_nv_decode()
    self.assertIsNotNone(fused.nv_decode_weights)
    xs = (Tensor.randn(1, 6, TINY_DIM)).realize()
    for t, pos in enumerate((0, 1, 2, 3, 0, 1)):
      with self.subTest(step=t, start_pos=pos):
        for block in (fused, generic): block._init_state(xs)
        want = generic._attention(generic.attn_norm(xs[:, t:t+1]), pos).numpy()
        with decode_gate(): got = fused._attention(fused.attn_norm(xs[:, t:t+1]), pos).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(fused.conv_state.numpy(), generic.conv_state.numpy(), rtol=0, atol=0)
        np.testing.assert_allclose(fused.recurrent_state.numpy(), generic.recurrent_state.numpy(), rtol=1e-5, atol=1e-7)

  def test_small_t_matches_generic_chunk(self):
    # T4.132: concrete runs of 1..GDN_DECODE_TOKENS tokens, interleaved, through the two kernels vs the generic concrete-T chunk
    # path (the per-token loop scan forced: WY reassociates the state math) -- output and both states after every call
    fused, generic = tiny_block(), tiny_block()
    with decode_gate(): fused.prepare_nv_decode()
    xs = Tensor.randn(1, 16, TINY_DIM).realize()
    pos, calls = 0, (1, 4, 2, 1, 3, 4)
    with Context(GDN_SCAN_IMPL=1):
      for t, T in enumerate(calls):
        with self.subTest(step=t, tokens=T, start_pos=pos):
          for block in (fused, generic): block._init_state(xs)
          want = generic._attention(generic.attn_norm(xs[:, pos:pos+T]), pos).numpy()
          with decode_gate(): got = fused._attention(fused.attn_norm(xs[:, pos:pos+T]), pos).numpy()
          self.assertEqual(got.shape, (1, T, TINY_DIM))
          np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-5)
          np.testing.assert_allclose(fused.conv_state.numpy(), generic.conv_state.numpy(), rtol=0, atol=0)
          np.testing.assert_allclose(fused.recurrent_state.numpy(), generic.recurrent_state.numpy(), rtol=1e-5, atol=1e-7)
        pos += T

  def test_matches_generic_path_through_function_and_jit(self):
    # the served path: GatedDeltaNetBlock.__call__ is a precompiled @function inside a TinyJit. A precompiled function drops arguments
    # no kernel references, so start_pos must reach the gdn kernels as a CALL src (an earlier AFTER-only binding ran unbound here).
    # Each side's output is realized exactly once: realizing a function's output twice re-runs its in-place state updates.
    blocks = {name: tiny_block() for name in ("fused", "generic", "jit_fused", "jit_generic")}
    for name in ("fused", "jit_fused"):
      with decode_gate(): blocks[name].prepare_nv_decode()
    jits = {name: TinyJit(functools.partial(lambda b, x, sp: b(x, sp).contiguous().realize(), blocks[name])) for name in ("jit_fused", "jit_generic")}
    xs, var = Tensor.randn(1, 6, TINY_DIM).realize(), UOp.variable("start_pos", 0, 63)
    for t, pos in enumerate((0, 1, 2, 3, 0, 1)):  # step 1 is the JIT capture, 2+ replays, 4 restarts with a nonzero resident state
      x, got = xs[:, t:t+1].contiguous().realize(), {}
      for name, block in blocks.items():
        with decode_gate() if "fused" in name else contextlib.nullcontext():
          out = jits[name](x, var.bind(pos)) if name in jits else block(x, var.bind(pos))
          got[name] = (out.numpy(), block.conv_state.numpy(), block.recurrent_state.numpy())
      for fused, generic in (("fused", "generic"), ("jit_fused", "jit_generic")):
        with self.subTest(step=t, pair=fused):
          np.testing.assert_allclose(got[fused][0], got[generic][0], rtol=1e-3, atol=1e-5)
          np.testing.assert_allclose(got[fused][1], got[generic][1], rtol=0, atol=0)
          np.testing.assert_allclose(got[fused][2], got[generic][2], rtol=1e-5, atol=1e-7)

class TestGDNDecodeGraph(unittest.TestCase):
  @staticmethod
  def _q4_0(in_features:int, out_features:int, seed:int) -> Linear:
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, 18 * (in_features//32) * out_features, dtype=np.uint8).reshape(-1, 18)
    packed[:, 0:2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)  # finite d
    raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": ggml_data_to_tensor(raw, in_features*out_features, Q4_0).reshape(out_features, in_features)},
                             verbose=False, realize=False)
    linear.set_quantized(linear.weight)
    return linear

  def _card_like_block(self) -> GatedDeltaNetBlock:
    # the served layout in miniature: attn_qkv/attn_gate/ssm_out on the Q4_0 dp4a gemv (split-K partials), ssm_alpha/ssm_beta on the f16 gemv
    block = tiny_block(dim=128)
    block.attn_qkv, block.attn_gate, block.ssm_out = self._q4_0(128, block.conv_channels, 0), self._q4_0(128, 128, 1), self._q4_0(128, 128, 2)
    return block

  def _decode_kernels(self, prepared:bool) -> list[str]:
    block, x = self._card_like_block(), Tensor.randn(1, 1, 128).realize()
    block._init_state(x)
    Tensor.realize(block.conv_state, block.recurrent_state)
    with decode_gate(), mock.patch("tinygrad.llm.kernels.nv_quant.nv_quant_supported", return_value=True):
      if prepared: block.prepare_nv_decode()
      return kernel_names(block._attention(x, 3))

  def test_decode_generic_kernels_gone(self):
    # NV decode graph of one block (no device opened: Linear's NV gate and the GDN decode gate mocked): before, 15 generic kernels and the
    # fused scan; after, the two gdn kernels and 2 generic ones -- x.half() and nv_f16_gemv's contiguous copy of it, both inputs of the
    # projections. attn_qkv's and attn_gate's split-K partial sums are folded into the gdn kernels (_split_k_partials).
    before, after = self._decode_kernels(False), self._decode_kernels(True)
    custom = ["q8_quantize_nv", "linear_f16_gemv_nv", "linear_f16_gemv_nv", "linear_q4_0_nv", "linear_q4_0_nv", "q8_quantize_nv", "linear_q4_0_nv"]
    self.assertEqual([n for n in before if n not in ("?", "gated_delta_prefill_nv")], custom)
    self.assertEqual((before.count("?"), before.count("gated_delta_prefill_nv")), (15, 1))
    # T4.127b: ssm_beta/ssm_alpha's two linear_f16_gemv_nv calls (before) become one (the combined gemv) after prepare_nv_decode
    fused_custom = ["q8_quantize_nv", "linear_f16_gemv_nv", "linear_q4_0_nv", "linear_q4_0_nv", "q8_quantize_nv", "linear_q4_0_nv"]
    self.assertEqual([n for n in after if n != "?"], fused_custom[:4] + ["gdn_conv_decode_nv", "gdn_decode_nv"] + fused_custom[4:])
    self.assertEqual(after.count("?"), 2)

  def test_split_k_partials(self):
    linear, x = self._q4_0(128, 96, 0), Tensor.randn(1, 1, 128)
    partials = _split_k_partials(nv_q8_linear(linear, x))
    self.assertIs(partials.uop.op, Ops.AFTER)
    self.assertEqual(partials.shape, (1, 96, 1))  # in_features 128 -> one 1024-wide split-K chunk
    generic = Tensor.randn(1, 1, 96) * 2  # anything else: (B,F,1), realized by the scheduler
    self.assertEqual(_split_k_partials(generic).shape, (1, 96, 1))
    self.assertIsNot(_split_k_partials(generic).uop.op, Ops.AFTER)

  @staticmethod
  def _kernel_asts(prepared:bool, gate_open:bool, x:Tensor, start_pos) -> list[UOp]:
    """the scheduled kernels' ASTs of one _attention call on a fresh block (scheduling remaps the state tensors, so one block per graph)"""
    block = tiny_block()
    block._init_state(x)
    Tensor.realize(block.conv_state, block.recurrent_state)
    if prepared:
      with decode_gate(): block.prepare_nv_decode()
      assert block.nv_decode_weights is not None
    with (decode_gate() if gate_open else contextlib.nullcontext()):
      linear, _ = block._attention(x, start_pos).linear_with_vars()
    return [u.src[0] for u in linear.src]

  def assert_same_kernels(self, a:list[UOp], b:list[UOp]):
    self.assertEqual(len(a), len(b))
    for x, y in zip(a, b): self.assertIs(x, y)  # hash-consed ASTs: `is` = the identical kernel

  def test_prefill_graph_unchanged(self):
    # the decode branch needs a concrete T <= GDN_DECODE_TOKENS (T4.132; was T == 1): a symbolic prefill chunk and a wider concrete
    # call schedule exactly the kernels they did with nothing prepared, gate open
    x = Tensor.randn(1, 8, TINY_DIM).realize()
    chunk, sp = x[:, :UOp.variable("toks", 1, 8).bind(5)], UOp.variable("start_pos", 0, 63).bind(16)
    for inp, pos in ((chunk, sp), (x[:, :GDN_DECODE_TOKENS + 1], 16)):
      with self.subTest(T=str(inp.shape[1])):
        self.assert_same_kernels(self._kernel_asts(False, True, inp, pos), self._kernel_asts(True, True, inp, pos))

  def test_small_t_takes_the_decode_kernels(self):
    # T4.132: a concrete run of 2..GDN_DECODE_TOKENS tokens schedules the same two kernels a decode token does
    x = Tensor.randn(1, 8, TINY_DIM).realize()
    for T in range(2, GDN_DECODE_TOKENS + 1):
      with self.subTest(T=T):
        names = [u.arg.name for u in self._kernel_asts(True, True, x[:, :T], 16) if isinstance(u.arg, KernelInfo)]
        self.assertEqual([n for n in names if n.startswith("gdn_")], ["gdn_conv_decode_nv", "gdn_decode_nv"])

  def test_gate_closed_keeps_the_generic_decode_graph(self):
    # GDN_NV_FUSED_DECODE closed (the default) at call time: prepared weights change nothing
    x = Tensor.randn(1, 1, TINY_DIM).realize()
    self.assert_same_kernels(self._kernel_asts(False, False, x, 3), self._kernel_asts(True, False, x, 3))

  def test_prepare_only_under_the_gate_and_geometry(self):
    block = tiny_block()
    block.prepare_nv_decode()  # gate closed: CPU, GDN_NV_FUSED_DECODE=0
    self.assertIsNone(block.nv_decode_weights)
    odd = SSMConfig(conv_kernel=4, state_size=32, group_count=2, time_step_rank=4, inner_size=12*4)  # head_v_dim 12: not a multiple of 8
    config = TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=1, n_kv_heads=1, norm_eps=1e-5, vocab_size=32, head_dim=8,
                               rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=64, ssm_layers=(True,), ssm=odd)
    with decode_gate():
      for blk in (GatedDeltaNetBlock(config, odd), GatedDeltaNetBlock(config, SSMConfig(4, 32, 2, 4, 32*4, kda=True))):
        blk.prepare_nv_decode()
        self.assertIsNone(blk.nv_decode_weights)
      block.prepare_nv_decode()
    self.assertEqual([tuple(w.shape) for w in cast_weights(block)], [(block.conv_channels, 4), (32,), (4,), (4,)])

def cast_weights(block:GatedDeltaNetBlock) -> tuple[Tensor, ...]:
  assert block.nv_decode_weights is not None
  return block.nv_decode_weights

if __name__ == "__main__":
  unittest.main()
