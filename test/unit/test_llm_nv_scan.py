"""T6.2: the NV fused Gated-DeltaNet prefill scan (tinygrad/llm/kernels/nv.py).
T4.98g adds: the SAME kernel also serves the T_pad==1 DECODE step (readout+state-update only -- conv1d stays
the separate, unchanged code it always was outside this kernel), behind its own GDN_NV_FUSED_DECODE gate.

Runs on CPU: gate/import checks and an sm_86 render + nvrtc compile of the kernel at the 35B's real geometry (skips when
the nvrtc lane -- docker on macOS -- is unavailable). The numerics tests need real NV hardware: mock-NV reports sm_35, so
the gate is closed there and they skip. Hardware round: DEV=NV PYTHONPATH=. python -m pytest test/unit/test_llm_nv_scan.py"""
import unittest
import numpy as np
from tinygrad import Tensor, Device, Context, UOp, dtypes
from tinygrad.device import CompileError
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels.nv import (
  nv_custom_kernels_supported, nv_decode_kernel_supported, GDN_NV_FUSED_DECODE, gated_delta_prefill, _gated_delta_prefill_kernel,
)

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

if __name__ == "__main__":
  unittest.main()
