"""T4.138: the GatedDeltaNet prefill chunk's glue as NV kernels (tinygrad/llm/kernels/nv_gdn_prep.py).

TestPrefillCensus: the kernels of ONE GatedDeltaNetBlock prefill call at the 27B geometry (symbolic 128-token chunk, the NV gates mocked
open on CPU -- scheduled, never run), each generic kernel named by the sm_86 hand-coded optimizer: the names of the card's 10-31 profile
(nv_gdn_prep.py's map). Rendering needs no nvrtc lane (_render_cuda's trick: the renderer's compiler server is mocked out)."""
import contextlib, unittest
from unittest import mock
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, Context
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, Q4_0
from tinygrad.llm.kernels import nv_gemm
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig

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

class TestPrefillCensus(unittest.TestCase):
  # step 0 (the map): the input copy of this harness (x is a slice, the served block input a buffer), the window zero fill (pruned on the
  # card), then the 10-31 profile's kernels in schedule order
  BEFORE = ["E_28toks2A4029_128", "E_10480_32_4", "E_320_32_3", "r_toks_16_320", "E_4_80_8_16_4_4", "q8_quantize_nv", "E_5120_32_4",
            "linear_q4_0_imma_nv", "linear_q4_0_imma_nv", "linear_f16_gemv_nv", "linear_f16_gemv_nv", "E_28toks2A8029_128", "E_6_2_8_16_4",
            "E_6_2_8_16_4", "r_128_16_16_8", "r_128_16_16_8", "E_48_2_16_16_8_4", "E_320_32_3", "E_16_16_8_8_16_3", "E_16_16_8_8_16_3",
            "r_2_8_8_16_3_32_4", "gated_delta_prefill_nv", "r_8_48_8_4_4_16", "E_32_6_2_8_16_4_4", "E_16_96_8_16_4", "q8_quantize_nv",
            "linear_q4_0_imma_nv", "r_toks_16_320", "E_4_80_8_16_4_4", "q8_quantize_nv", "linear_q4_0_imma_nv", "linear_q4_0_imma_nv",
            "E_28toks2A13629_128", "E_16_272_8_16_4", "q8_quantize_nv", "linear_q4_0_imma_nv", "E_28toks2A4029_128"]

  def test_gdn_prefill_kernels(self):
    self.assertEqual(prefill_census(card_block()), self.BEFORE)

if __name__ == "__main__":
  unittest.main()
