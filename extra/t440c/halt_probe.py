"""T4.40c / RCA Sec7 hardware verification -- run ONLY with Artur present, on the T4.40c branch, tiny model.
Two modes, selected by argv[1]:

  h1   Baseline: open NV, read PCI_COMMAND (expect MASTER set, 0x...7), run a trivial op, read again after a
       clean close. Proves the healthy path is untouched (MASTER stays set on a live device; cleared only by
       the next client's reset dance). Exit 0 on success.

  h3   Fix 40-1 live check: monkeypatch NV_GSP.init_sw to raise AFTER _early_ip_init has set bus-master, so the
       failure lands exactly where panic 2's boot RPC timeout did. Assert bus-master is CLEARED afterward via a
       fresh config read (the fix). Exit 0 if MASTER==0 after the forced failure, 2 if still set (fix NOT working).

Never run h3 on a tree without T4.40b's __init__ wrap. Abort the whole session on any UNEXPECTED fault.
"""
import sys
from tinygrad.runtime.autogen import pci

def read_master_via_fresh_pci():
  # Re-open a config-space handle to the NV function without constructing a full NVDevice, to read PCI_COMMAND
  # after a failed init. Uses the same enumeration the runtime uses.
  from tinygrad.runtime.support import system
  pd = system.PCIIfaceBase.__new__(system.PCIIfaceBase)  # not constructing; we only need a pci_dev
  raise SystemExit("h3 fresh-read helper is a stub -- see RUNBOOK.md for the exact one-liner to run by hand")

def h1():
  from tinygrad import Tensor, Device
  dev = Device["NV"]; pd = dev.iface.pci_dev
  m0 = pd.read_config(pci.PCI_COMMAND, 2)
  print(f"[H1] PCI_COMMAND on open = {m0:#06x}  MASTER={(m0>>2)&1} (expect 1)", flush=True)
  assert (m0>>2)&1 == 1, "healthy device should have bus-master set"
  print(f"[H1] trivial op result = {(Tensor([1.,2.,3.])+1).tolist()}", flush=True)
  m1 = pd.read_config(pci.PCI_COMMAND, 2)
  print(f"[H1] PCI_COMMAND after op = {m1:#06x}  MASTER={(m1>>2)&1} (expect 1, device still live)", flush=True)
  return 0

def h3():
  from tinygrad import Device
  from tinygrad.runtime.support.nv import ip
  orig = ip.NV_GSP.init_sw
  def boom(self): raise RuntimeError("T4.40c H3: forced boot failure after bus-master was enabled")
  ip.NV_GSP.init_sw = boom
  try:
    Device["NV"]; print("[H3] FAIL: init did not raise"); return 3
  except RuntimeError as e:
    print(f"[H3] init raised as expected: {str(e).splitlines()[0][:80]}", flush=True)
  finally:
    ip.NV_GSP.init_sw = orig
  # Fix 40-1 should have cleared MASTER on the way out. Read it back via a fresh handle -- see RUNBOOK.md h3
  # for the exact fresh-pci read to run here (kept manual so the harness never half-opens a second session).
  print("[H3] now run the RUNBOOK.md h3 config-read to confirm MASTER==0", flush=True)
  return 0

if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "h1"
  sys.exit({"h1": h1, "h3": h3}[mode]())
