"""T6.2: the NV fused Gated-DeltaNet prefill scan (tinygrad/llm/kernels/nv.py).

Runs on CPU: gate/import checks and an sm_86 render + nvrtc compile of the kernel at the 35B's real geometry (skips when
the nvrtc lane -- docker on macOS -- is unavailable). The numerics tests need real NV hardware: mock-NV reports sm_35, so
the gate is closed there and they skip. Hardware round: DEV=NV PYTHONPATH=. python -m pytest test/unit/test_llm_nv_scan.py"""
import unittest
import numpy as np
from tinygrad import Tensor, Device, Context, UOp, dtypes
from tinygrad.device import CompileError
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels.nv import nv_custom_kernels_supported, gated_delta_prefill, _gated_delta_prefill_kernel

GEOMETRIES = {"27b": (48, 128, 128), "35b": (32, 128, 128)}  # (v-heads, head_k_dim, head_v_dim): Qwen3.8-27B / Qwen3.6-35B-A3B
CHUNK = 32

def reference(q, k, v, beta, alpha, state):
  """numpy ground truth, the recurrence from test_attention's RDNA3 parity test (alpha per (token, value-row))."""
  out, st = np.empty_like(v), state.copy()
  for t in range(q.shape[2]):
    previous, av = st.copy(), alpha[:, :, t, :, None]
    delta = (v[:, :, t] - (previous*k[:, :, t, None]).sum(-1)*alpha[:, :, t]) * beta[:, :, t, None]
    st = previous*av + delta[..., None]*k[:, :, t, None, :]
    out[:, :, t] = (previous*q[:, :, t, None]).sum(-1)*alpha[:, :, t] + delta*(q[:, :, t]*k[:, :, t]).sum(-1)
  return out, st

def random_inputs(heads, key_dim, value_dim, tokens=CHUNK, seed=0):
  rng = np.random.default_rng(seed)
  q, k = (rng.normal(size=(1, heads, tokens, key_dim)).astype(np.float32) for _ in range(2))
  v, beta = rng.normal(size=(1, heads, tokens, value_dim)).astype(np.float32), rng.uniform(size=(1, heads, tokens)).astype(np.float32)
  alpha = rng.uniform(0.8, 1, size=(1, heads, tokens, value_dim)).astype(np.float32)
  return q, k, v, beta, alpha, rng.normal(size=(1, heads, value_dim, key_dim)).astype(np.float32)

class TestNVGate(unittest.TestCase):
  def test_gate_closed_off_nv(self):
    for device in ("CPU", "CPU:1", "NULL", None, ("CPU", "CPU:1"), Device.DEFAULT if not Device.DEFAULT.startswith("NV") else "CPU"):
      self.assertFalse(nv_custom_kernels_supported(device), device)

  def test_env_closes_gate_without_touching_the_device(self):
    with Context(GDN_NV_FUSED=0): self.assertFalse(nv_custom_kernels_supported("NV"))  # returns before Device["NV"] is opened

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

@unittest.skipUnless(nv_custom_kernels_supported(Device.DEFAULT), "real NV hardware with the gate open required")
class TestNVFusedScanParity(unittest.TestCase):
  def test_kernel_matches_numpy(self):
    for name, (heads, key_dim, value_dim) in GEOMETRIES.items():
      with self.subTest(geometry=name):
        q, k, v, beta, alpha, initial = random_inputs(heads, key_dim, value_dim)
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
        state = Tensor(initial).contiguous().realize()
        start_pos = Tensor(UOp.variable("start_pos", 0, 4096).bind(pos))
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

if __name__ == "__main__":
  unittest.main()
