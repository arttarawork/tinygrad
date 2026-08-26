#!/bin/sh
# T4.37 hardware verification, step B: recreate the panic's preconditions WITH the fix.
# Precondition: step A exited 0 (MASTER cleared after fault) in the SAME server session -- do not restart the server between A and B.
set -u
W=${1:?worktree}; PY=/Users/artur/Documents/tinygrad/.venv/bin/python
echo "[B1] servers: $(pgrep -fl 'TinyGPU.*server' | wc -l | tr -d ' ')"
echo "[B2] pkill the server while the GPU is faulted (the sequence that panicked at 01:15)"; pkill -f "TinyGPU.*server"; sleep 2
echo "[B3] 60 s window in which a live GSP would DMA into unmapped queues if MASTER were still set..."; sleep 60
echo "[B4] fresh client (expect 'WPR2 is up. Issuing a full reset.' then a correct result)"
cd "$W" && PYTHONPATH=. DEV=NV:NAK DEBUG=2 "$PY" -c "from tinygrad import Tensor; print('result:', (Tensor([1.,2.])+1).tolist())" 2>&1 | grep -E "WPR2|result:|Error|fault"
echo "[B5] servers: $(pgrep -fl 'TinyGPU.*server' | wc -l | tr -d ' ')  (want 1)"
