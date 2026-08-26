"""T4.37 hardware verification, step A (run ONLY with the T4.37 fix in the tree, NEVER pre-fix):
# WARNING: deliberately faults the GPU. Run ONLY on a tree containing T4.37 (bus-master cleared on fault); pre-fix this is the panic setup.
1. open the NV device, read PCI_COMMAND, expect MASTER (bit 2) SET
2. compile one trivial real kernel, then re-launch it with a bogus (unmapped) GPU VA -> deliberate MMU fault
3. synchronize -> expect 'Device fault detected' -> read PCI_COMMAND again -> MASTER must be CLEAR (the fix)
Exit code 0 = fix verified (MASTER cleared after fault); 2 = MASTER still set (fix NOT working -> do NOT proceed to step B).
"""
import sys
from tinygrad import Tensor, Device
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.hcq import HCQBuffer
from tinygrad.engine import realize

dev = Device["NV"]; pd = dev.iface.pci_dev
def master(): return (pd.read_config(pci.PCI_COMMAND, 2) >> 2) & 1
print("MASTER before:", master(), flush=True)
assert master() == 1, "bus-master should be enabled on a healthy device"

(Tensor.ones(16) + 1).realize(); dev.synchronize()
prg = next(p for (_, d), p in realize.runtime_cache.items() if d == "NV")
print("kernel:", prg.name, "signature bufs:", len(prg.signature), flush=True)
bogus = 0x123456780000  # far outside anything mapped in this process's GPU VA space
try:
  prg(*[HCQBuffer(bogus, 4096) for _ in range(len(prg.signature))], global_size=(1,1,1), local_size=(1,1,1), wait=True)
  dev.synchronize()
  print("!! no fault raised -- bogus VA was reachable? MASTER:", master()); sys.exit(3)
except RuntimeError as e:
  print("fault raised:", str(e).splitlines()[0][:100], flush=True)
m = master(); print("MASTER after fault:", m, flush=True)
sys.exit(0 if m == 0 else 2)
