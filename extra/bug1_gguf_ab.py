# Bug-1 decisive A/B: the REAL Qwen3.8-27B GGUF, CPU only, WY (IMPL=2) vs loop (IMPL=1).
# If WY NaN-floods here, bug-1 is numerics-at-real-weights, not any device stack property.
import os, math
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, GDN_SCAN_IMPL
impl = int(os.getenv("IMPL", "2"))
with Context(GDN_SCAN_IMPL=impl, GDN_CHUNK=32):
    model, kv = Transformer.from_gguf("/Users/artur/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf",
                                      max_context=2048, device_map="0-63:CPU")
    gen = model.generate(list(range(1000, 1020)), chunk_size=32)
    toks = [next(gen) for _ in range(3)]
    print(f"IMPL={impl} generated ids:", toks)
    bad = 0
    for i, b in enumerate(model.blk):
        if isinstance(b, GatedDeltaNetBlock):
            flat = b.recurrent_state.float().flatten().tolist()
            n = sum(1 for v in flat if v != v)
            bad += n
            if n and bad == n: print(f"first NaN block: blk{i} ({n} NaNs)")
    print(f"IMPL={impl} VERDICT:", "NaN-FLOOD" if bad else "clean states", f"(total NaNs {bad})")
