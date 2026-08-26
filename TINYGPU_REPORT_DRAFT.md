# TinyGPU host-safety report — DRAFT for tinygrad/tinygpu_releases (filing = Artur's decision)

*(Everything above the `---` divider is the proposed issue body, ready to copy-paste. Below the divider: filing notes for Artur.)*

**Title:** Server/DEXT tears down a client's DMA mappings without quiescing the GPU — host kernel panic (`AppleT8110DART … REQUIRE failed`) when a client exits while GSP-RM is still running

**Setup:** MacBook Pro M3 Pro 36 GB, macOS 15.7.2 (24G325); AOOSTAR AG02 dock over USB4; RTX 3090 (GA102, `10de:2204`, BAR1 256 MiB → small-BAR path); TinyGPU.app pinned release `c0d024f9ff0e1dc8fdf217f255da7101d91e8323`; tinygrad fork `arttarawork/tinygrad` (line refs below against the in-tree `extra/usbgpu/tbgpu` sources at `5dea150e5`; the shipped app may differ slightly).

**What happened:** two kernel panics in one day (2026-08-26 01:15:41 and 12:06:34), both
`panic(cpu N): (dart-apciec2) AppleT8110DART.cpp:2183: REQUIRE failed`, task `kernel_task`, backtrace entirely
`AppleT8110DART`/`IODARTFamily` — the IOMMU behind the Thunderbolt PCIe tunnel caught a device DMA to a host
IOVA with no translation. No tinygrad frames. First panics in 58 days of uptime; both landed during heavy
tinygrad NV work.

**Mechanism (source-level):** when a python client disconnects, the server runs `cleanup()` →
`IOServiceClose` → `UserClient::Stop` → `CompleteDMA` for **every** allocation of that session
(`installer/Shared/server.c`, `installer/TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp`) — with nothing
clearing PCI bus-master or halting GSP-RM first. GSP-RM's message/status queues live in exactly that sysmem
(`tinygrad/runtime/support/nv/ip.py`, `_alloc_boot_mem(..., sysmem=True)`). So **any client that exits without
a successful `rpc_unloading_guest_driver` leaves a running, bus-mastering GSP-RM whose queues the DART no
longer translates** — the next firmware DMA panics the host. Three ways a client exits like that:
1. a GPU fault → the unload RPC itself times out at exit (our panic 1);
2. a **failed device open** — any 10 s boot-RPC timeout raises out of `NVDevice.__init__`, so no unload ever
   runs, and an availability probe swallows the exception (our panic 2: a pytest `Device.get_available_devices()`
   probe at collection time);
3. a hard kill (SIGKILL/crash) of the client or server — including the `pkill` recovery folklore.
Both panics sit ~10 s after session activity, matching the driver's universal 10 s RPC timeout → raise → exit
→ unmap chain. Timelines are reconstructed from the unified log (the DEXT's own `NewUserClient` / cfg-write /
`reset` / `PrepareDMA` os_log lines survive the reboot); the panic reports carry no faulting IOVA, which is why
this is argued from source + timeline rather than a captured address. Full RCA with per-claim file:line and the
log extract: https://github.com/arttarawork/tinygrad/blob/memory/T4.40_RCA.md (+ `ulog_tinygpu_2026-08-26.txt`).

**Asks (server/DEXT side — the only cover for exit class 3):**
1. **Quiesce before unmap**: clear PCI bus-master (config 0x4 &= ~4) in `cleanup()` before `IOServiceClose`,
   or in `Stop_Impl` before the `CompleteDMA` loop (~3 lines) — makes the host safe by construction.
2. An **unmap verb** in the wire protocol (today mappings only die with the session; also behind the ~128-slot
   sysmem ceiling we reported separately).
3. **accept > 1 or a busy-reply** on the socket: today a second concurrent client's connect is rejected
   silently, the client concludes the server is dead and spawns a new one whose `unlink(sock_path)` steals the
   path — orphaning the old server *with its live session* forever (the multi-server pileup that keeps
   armed sessions around).

**Client-side mitigations we shipped on the fork** (happy to upstream any of them):
- clear bus-master on the fault path (`task/T4.37-…`, merged to our master; mirrors merged PR #16007's
  `write_config_flush` idiom, and stays off the healthy teardown path per closed PR #16536);
- clear bus-master when `__init__` fails after mastering was enabled, and when `fini()` raises
  (`task/T4.40b-quiesce-init-fini`);
- never spawn a second server beside a live one; probes/xdist never spawn (`task/T4.40a-server-discipline`).
With those, the sequence that panicked twice completes cleanly; we can reproduce the preconditions safely on
request (deliberate boot-timeout on a tiny model, bus-master readouts at each step).

*Disclosure: the investigation and the fork patches were produced with AI assistance (Claude), directed and
reviewed by me.*

---

## Filing notes (Artur only — not part of the issue)
- File at **tinygrad/tinygpu_releases** (same tracker as the planned slot-ceiling/no-unmap-verb report — ask 2
  overlaps it; merge or cross-link if that one is filed too).
- **Close-risk framing** (#16086 precedent: geohot closes dock-panic reports unless self-contained): this one
  is self-contained — mechanism, preconditions, timelines, and a ~3-line fix in hand. It also directly answers
  his own #16536 comment ("tinygrad can still bring down my computer in some cases, maybe it's this").
- nimlgen's stance (#15606): wants the faulting address + RCA, not workarounds. We don't have the IOVA (panic
  reports omit it) — the report says so explicitly and leads with the source-level RCA instead.
- Optional trim for a shorter first post: drop the mitigation bullet list and keep just the link; keep asks 1-3.
- If they ask for a live repro: RCA §7's protocol (H3) is the safe, scripted version — only with you present.
