#!/usr/bin/env python3
# T4.61: standalone benchmark for GatedDeltaNetBlock's chunked scan (tinygrad/llm/model.py _attention, the
# non-AMD unrolled-per-t else-branch), isolated from the rest of the model -- no tokenizer/embedding/
# generate() JIT machinery. Builds a randomized block at REAL head geometry (see GEOMETRIES below; only
# --dim, the residual width feeding the block's own linear projections, is a free parameter) and times
# _attention over a token stream split into --chunk-sized calls, printing tok/s and GlobalCounters-derived
# GB/s per chunk -- same metric convention as extra/benchmark_llm.py's prefill/decode lines.
#
# See test/unit/test_gdn_scan_parity.py for the correctness counterpart (chunked vs. sequential parity at
# the same two geometries) and its perf note: at this geometry, calling _attention directly (no JIT) pays
# real per-call schedule/codegen overhead that generate()'s TinyJit normally amortizes away -- this script
# is what measures that cost, chunk width by chunk width, on whatever --device it's run on.
#
# examples:
#   ./extra/gdn_scan_bench.py --smoke                                          # tiny sizes, DEV=CPU default
#   ./extra/gdn_scan_bench.py --geometry 35b --chunk 32 --tokens 512 --device METAL --dim 4096
#   ./extra/gdn_scan_bench.py --geometry 38  --chunk 32 --tokens 512 --device NV     --dim 5120
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinygrad import Tensor, nn  # noqa: E402
from tinygrad.helpers import GlobalCounters  # noqa: E402
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig  # noqa: E402

# same real head geometry as test/unit/test_gdn_scan_parity.py -- GatedDeltaNetBlock.__init__ derives
# head_k_dim=ssm.state_size, num_k_heads=ssm.group_count, num_v_heads=ssm.time_step_rank,
# head_v_dim=ssm.inner_size//ssm.time_step_rank.
GEOMETRIES = {
  "35b": SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=32, inner_size=128 * 32),
  "38":  SSMConfig(conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=128 * 48),
}

def make_block(geometry: str, dim: int, max_context: int, device: str) -> GatedDeltaNetBlock:
  ssm = GEOMETRIES[geometry]
  config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=dim * 2, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
                              vocab_size=32, head_dim=dim, rope_theta=10000.0, rope_dim=dim, v_head_dim=dim,
                              max_context=max_context, ssm_layers=(True,), ssm=ssm)
  block = GatedDeltaNetBlock(config, ssm)
  Tensor.manual_seed(0)
  params = nn.state.get_parameters(block)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  # pin weights to concrete values now, before anything else (e.g. the token stream below) touches
  # Tensor's global RNG counter -- see test_gdn_scan_parity.py's make_block for why this matters.
  Tensor.realize(*params)
  for p in params: p.to_(device)
  return block

def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--geometry", choices=list(GEOMETRIES), default="35b", help="real head geometry to bench (default: %(default)s)")
  p.add_argument("--device", default="CPU", help="DEV for the block's tensors (default: %(default)s)")
  p.add_argument("--chunk", type=int, default=None, help="tokens per _attention call (default: 32, or 2 with --smoke)")
  p.add_argument("--tokens", type=int, default=None, help="total tokens to scan through (default: 256, or 8 with --smoke)")
  p.add_argument("--dim", type=int, default=None, help="residual width fed to the block's linears (default: 64, or 8 with --smoke)")
  p.add_argument("--smoke", action="store_true", help="tiny sizes, to prove the script executes -- does not change --geometry's head shape")
  args = p.parse_args()

  chunk = args.chunk or (2 if args.smoke else 32)
  tokens = args.tokens or (8 if args.smoke else 256)
  dim = args.dim or (8 if args.smoke else 64)
  assert chunk > 0 and tokens > 0, f"--chunk {chunk} and --tokens {tokens} must both be > 0"

  block = make_block(args.geometry, dim, max_context=tokens, device=args.device)
  x = (Tensor.randn(1, tokens, dim, device=args.device) * 0.1).realize()
  print(f"geometry={args.geometry} device={args.device} dim={dim} tokens={tokens} chunk={chunk} "
        f"num_v_heads={block.num_v_heads} num_k_heads={block.num_k_heads} "
        f"head_k_dim={block.head_k_dim} head_v_dim={block.head_v_dim}")

  pos = 0
  while pos < tokens:
    size = min(chunk, tokens - pos)
    x_norm = block.attn_norm(x[:, pos:pos + size])
    block._init_state(x_norm)
    GlobalCounters.reset()
    st = time.perf_counter()
    block._attention(x_norm, pos).realize()
    dt = time.perf_counter() - st
    print(f"chunk[{pos}:{pos + size}] {size / dt:.2f} tok/s {GlobalCounters.global_mem / dt / 1e9:.2f} GB/s")
    pos += size

if __name__ == "__main__":
  main()
