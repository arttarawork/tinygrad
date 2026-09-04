#!/usr/bin/env python3
"""T4.73c: ONE-SHOT, targeted, read-only extraction of blk.0's real GDN weights + a handful of real
token-embedding rows from the real Qwen3.8-27B GGUF -- WITHOUT loading the whole ~29GB file.

Why not just call tinygrad.llm.gguf.gguf_load(path) and filter the result: _gguf_parse's flush() loop
walks EVERY tensor in the file (sorted by on-disk offset) and STAGES+REALIZES each batch's raw quantized
bytes unconditionally, merging adjacent tensors up to _STAGE_BATCH (64MB) each -- by the time gguf_load()
returns, the returned state_dict holds live references to realized buffers covering the ENTIRE file, for
every block, not just blk.0. There is no parameter to gguf_load/_gguf_parse that skips tensors -- so
calling it on the whole path is NOT meaningfully cheaper than Transformer.from_gguf's full load (same
~27-29GB). This script instead re-uses _gguf_parse's own building blocks (_parse_header for the cheap
header-only pass, _ggml_nbytes + ggml_data_to_tensor for per-tensor dequant) but drives them ITSELF over
only the tensor infos this bug needs:
  - every tensor named "blk.0.*" (blk0 is confirmed a GatedDeltaNetBlock -- bug1_gguf_ab.py's NaN starts
    there), a few hundred MB at most for one block of a 27B model.
  - 32 rows (token ids 1000..1031, covering bug1_gguf_ab.py's driver prompt 1000..1019 plus 12 more) of
    "token_embd.weight". GGUF
    quantizes row-major with the embedding dim as the fastest-varying (contiguous, block-aligned) axis --
    Q8_0 blocks are 32 contiguous elements -- so each row occupies a fixed contiguous byte range and slicing
    N contiguous rows reads/dequantizes only those rows' bytes, not the whole (vocab x dim) matrix.
Total bytes touched: one block's weights + ~tens of KB of embedding rows -- hundreds of MB at most, nowhere
near the 27GB a full load stages, safe to run with the pooled server (:8081) up.

Everything is cast to float16 before saving, matching Transformer.from_gguf's default HALF=1 state_dict
cast (bit-identical to what the real served model actually runs on) -- see model.py from_gguf L1098.

Run once: PYTHONPATH=. <venv>/bin/python extra/extract_blk0_real.py
Output: extra/blk0_real.safetensors (blk.0.* weights, keys as in the GGUF, plus "token_embd_rows" (20,dim)
float16 for token ids 1000..1019 in order) -- committed-safe size (weights only, not the 29GB source file).
"""
import pathlib, json
from tinygrad import Tensor
from tinygrad.helpers import prod, round_up
from tinygrad.llm.gguf import _parse_header, _ggml_nbytes, ggml_data_to_tensor, _HEADER_CHUNK
from tinygrad.nn.state import safe_save

GGUF = "/Users/artur/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"
OUT = "extra/blk0_real.safetensors"
ROW_LO, ROW_HI = 1000, 1032  # bug1_gguf_ab.py's driver prompt (1000..1019) plus 12 more real ids, so a
                              # T=32 (real-token, no synthetic padding) chunk-32 repro is possible too
DEV = "CPU"  # hard rule: CPU/NULL only, never open METAL/NV in this worktree
# GatedDeltaNetBlock's __init__/_attention/_init_state (non-kda: no ssm_g_a/ssm_g_b/ssm_f_a/ssm_f_b) touch
# exactly these blk.0.* weights -- ffn_gate/up/down.weight and post_attention_norm.weight are FFNBlock-only
# (unused by _attention) and are the bulk of a block's size (~535 of ~765 MB float16), so skip them: this
# is a numerics repro for the GDN scan, not a full-block/FFN test.
NEEDED = {"attn_qkv.weight", "attn_gate.weight", "attn_norm.weight", "ssm_alpha.weight", "ssm_beta.weight",
          "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ssm_out.weight"}

def parse_header_only(t: Tensor) -> tuple[dict, list, int]:
  size = t.shape[0]
  chunk = min(size, _HEADER_CHUNK)
  while True:
    header = t[:chunk].to(None).realize()
    try:
      kv_data, t_infos, pos = _parse_header(header)
      if chunk < size and pos >= chunk: raise ValueError("header may have been truncated by the chunk boundary")
    except (ValueError, IndexError):
      if chunk >= size: raise
      chunk = min(size, chunk * 2)
      continue
    return kv_data, t_infos, pos

def stage_tensor(t: Tensor, data_start: int, off: int, dims: tuple[int, ...], typ: int) -> Tensor:
  nbytes = _ggml_nbytes(prod(dims), typ)
  raw = t[data_start + off:data_start + off + nbytes].to(DEV).realize()
  return ggml_data_to_tensor(raw, prod(dims), typ).reshape(*reversed(dims))

if __name__ == "__main__":
  t = Tensor(pathlib.Path(GGUF))
  print(f"file size: {t.shape[0]/1e9:.2f} GB (NOT fully read -- header + targeted slices only)")
  kv, t_infos, pos = parse_header_only(t)
  alignment = kv.get("general.alignment", 32)
  data_start = round_up(pos, alignment)
  arch = kv["general.architecture"]
  print(f"arch={arch}")

  cfg_keys = [f"{arch}.embedding_length", f"{arch}.full_attention_interval", f"{arch}.block_count",
              f"{arch}.attention.head_count", f"{arch}.attention.head_count_kv",
              f"{arch}.attention.layer_norm_rms_epsilon", f"{arch}.context_length"]
  cfg_keys += [k for k in kv if k.startswith(f"{arch}.ssm.") or k.startswith(f"{arch}.kda.")]
  cfg = {k: kv[k] for k in cfg_keys if k in kv}
  for k, v in sorted(cfg.items()): print(f"  {k} = {v}")

  wanted = {name[len("blk.0."):]: (dims, typ, off) for name, dims, typ, off in t_infos
            if name.startswith("blk.0.") and name[len("blk.0."):] in NEEDED}
  assert wanted.keys() == NEEDED, f"missing expected blk.0 tensors: {NEEDED - wanted.keys()}"
  emb = next(((dims, typ, off) for name, dims, typ, off in t_infos if name == "token_embd.weight"), None)
  assert emb is not None, "token_embd.weight not found in GGUF tensor list"
  print(f"\nblk.0.* tensors: {len(wanted)}")
  for name, (dims, typ, off) in sorted(wanted.items()): print(f"  {name}: dims={dims} ggml_type={typ}")

  out: dict[str, Tensor] = {}
  total_bytes = 0
  for name, (dims, typ, off) in wanted.items():
    total_bytes += _ggml_nbytes(prod(dims), typ)
    out[name] = stage_tensor(t, data_start, off, dims, typ).cast("float16").contiguous()

  emb_dims, emb_typ, emb_off = emb
  row_elems, vocab = emb_dims[0], emb_dims[1]  # ne[0]=dim (contiguous/fastest-varying), ne[1]=vocab -- reversed() below gives (vocab, dim)
  assert 0 <= ROW_LO and ROW_HI <= vocab, f"row range [{ROW_LO},{ROW_HI}) out of bounds for vocab={vocab}"
  bytes_per_row = _ggml_nbytes(row_elems, emb_typ)
  total_bytes += bytes_per_row * (ROW_HI - ROW_LO)
  raw_rows = t[data_start + emb_off + ROW_LO * bytes_per_row:data_start + emb_off + ROW_HI * bytes_per_row].to(DEV).realize()
  out["token_embd_rows"] = ggml_data_to_tensor(raw_rows, (ROW_HI - ROW_LO) * row_elems, emb_typ) \
    .reshape(ROW_HI - ROW_LO, row_elems).cast("float16").contiguous()

  print(f"\ntotal on-disk bytes staged: {total_bytes/1e6:.1f} MB")
  print(f"token_embd_rows: shape={out['token_embd_rows'].shape} (ids {ROW_LO}..{ROW_HI-1})")

  meta = {"arch": arch, "row_lo": str(ROW_LO), "row_hi": str(ROW_HI), **{k: json.dumps(v) for k, v in cfg.items()}}
  safe_save(out, OUT, metadata=meta)
  print(f"\nsaved {len(out)} tensors to {OUT}")
