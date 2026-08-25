#!/usr/bin/env python3
# T0.3 bench harness. Runs the same GGUF through tinygrad.llm (extra/benchmark_llm.py) and/or
# llama-bench, and appends one CSV row per run: model, stack, device, flags, load_s, prefill_tps,
# decode_tps, gbps. Reuses benchmark_llm.py rather than reimplementing the timing loop -- this
# script is just a thin subprocess + CSV shim, on purpose.
#
# examples:
#   ./extra/bench_llm.py tinygrad --model qwen3:8b --device METAL --csv extra/bench_results.csv
#   ./extra/bench_llm.py tinygrad --model qwen3:8b --device METAL --env JITBEAM=2 --env IGNORE_BEAM_CACHE=1 --tag "JITBEAM=2"
#   ./extra/bench_llm.py llamacpp --model qwen3:8b --repeat 3 --csv extra/bench_results.csv
import argparse, csv, os, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinygrad.helpers import fetch  # noqa: E402
from tinygrad.llm.cli import models as MODEL_URLS  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV_FIELDS = ["model", "stack", "device", "flags", "load_s", "prefill_tps", "decode_tps", "gbps"]

def resolve_gguf(model:str) -> str: return str(fetch(MODEL_URLS.get(model, model)))

def append_csv(path:str, row:dict):
  new = not Path(path).exists()
  with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new: w.writeheader()
    w.writerow(row)

def run_tinygrad(gguf:str, model_name:str, device:str, env_extra:dict, flags_tag:str,
                  prompt_tokens:int, decode_tokens:int, python:str, device_map:str|None=None) -> dict:
  env = os.environ.copy()
  env["DEV"] = device
  env.update(env_extra)
  cmd = [python, str(HERE / "benchmark_llm.py"), "--model", gguf,
         "--prompt-tokens", str(prompt_tokens), "--decode-tokens", str(decode_tokens)]
  if device_map: cmd += ["--device-map", device_map]
  print(f"+ DEV={device} {' '.join(f'{k}={v}' for k,v in env_extra.items())} {' '.join(cmd)}", file=sys.stderr)
  proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2700)  # 45 min: matches TD.3 bench-row skip budget
  if proc.returncode != 0:
    print(proc.stdout, file=sys.stderr); print(proc.stderr, file=sys.stderr)
    raise RuntimeError(f"benchmark_llm.py failed (exit {proc.returncode})")
  text = proc.stdout
  print(text, file=sys.stderr)
  load_s = float(re.search(r"^load ([\d.]+)s", text, re.M).group(1))
  prefill_tps = float(re.search(r"^prefill ([\d.]+) tok/s", text, re.M).group(1))
  m = re.search(r"^decode ([\d.]+) tok/s ([\d.]+) GB/s", text, re.M)
  decode_tps, gbps = float(m.group(1)), float(m.group(2))
  flags = " ".join(f"{k}={v}" for k, v in env_extra.items()) + (f" device_map={device_map}" if device_map else "") + (f" {flags_tag}" if flags_tag else "")
  return dict(model=model_name, stack="tinygrad", device=device, flags=flags.strip(),
              load_s=f"{load_s:.3f}", prefill_tps=f"{prefill_tps:.2f}", decode_tps=f"{decode_tps:.2f}", gbps=f"{gbps:.2f}")

def run_llamacpp(gguf:str, model_name:str, n_prompt:int, n_gen:int, repetitions:int, llama_bench:str, flags_tag:str) -> dict:
  cmd = [llama_bench, "-m", gguf, "-p", str(n_prompt), "-n", str(n_gen), "-r", str(repetitions), "-o", "csv"]
  print(f"+ {' '.join(cmd)}", file=sys.stderr)
  proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
  if proc.returncode != 0:
    print(proc.stdout, file=sys.stderr); print(proc.stderr, file=sys.stderr)
    raise RuntimeError(f"llama-bench failed (exit {proc.returncode})")
  print(proc.stderr, file=sys.stderr)
  rows = list(csv.DictReader(proc.stdout.splitlines()))
  pp = next(r for r in rows if int(r["n_prompt"]) > 0)
  tg = next(r for r in rows if int(r["n_gen"]) > 0)
  print(f"pp{n_prompt} {pp['avg_ts']} +/- {pp['stddev_ts']} tok/s, tg{n_gen} {tg['avg_ts']} +/- {tg['stddev_ts']} tok/s", file=sys.stderr)
  flags = f"-r {repetitions}" + (f" {flags_tag}" if flags_tag else "")
  # llama-bench has no load-time-only measurement and no GlobalCounters equivalent; both left blank (N/A).
  return dict(model=model_name, stack="llamacpp", device="METAL", flags=flags.strip(), load_s="",
              prefill_tps=f"{float(pp['avg_ts']):.2f}", decode_tps=f"{float(tg['avg_ts']):.2f}", gbps="")

def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("stack", choices=["tinygrad", "llamacpp"])
  p.add_argument("--model", "-m", required=True, help="model name from tinygrad.llm.cli.models, or a local GGUF path")
  p.add_argument("--device", default="METAL", help="DEV for the tinygrad stack (default: %(default)s)")
  p.add_argument("--device-map", default=None, help="--device-map passthrough to benchmark_llm.py (TD.3 pooling)")
  p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env var for the tinygrad stack (repeatable)")
  p.add_argument("--tag", default="", help="free-text note appended to the flags column (e.g. a branch/commit)")
  p.add_argument("--prompt-tokens", type=int, default=512, help="tinygrad stack prefill length (default: %(default)s)")
  p.add_argument("--decode-tokens", type=int, default=128, help="tinygrad stack decode length (default: %(default)s)")
  p.add_argument("--n-prompt", type=int, default=512, help="llama-bench -p (default: %(default)s)")
  p.add_argument("--n-gen", type=int, default=128, help="llama-bench -n (default: %(default)s)")
  p.add_argument("--repeat", type=int, default=1, help="tinygrad stack: repeat the whole run N times, one CSV row each")
  p.add_argument("--llama-bench-repetitions", type=int, default=3, help="llama-bench -r, internally averaged (default: %(default)s)")
  p.add_argument("--llama-bench", default="/opt/homebrew/bin/llama-bench", help="llama-bench binary path")
  p.add_argument("--python", default=sys.executable, help="python to run benchmark_llm.py with")
  p.add_argument("--csv", default=None, help="CSV file to append rows to (prints the row if omitted)")
  args = p.parse_args()

  gguf = resolve_gguf(args.model)
  env_extra = dict(kv.split("=", 1) for kv in args.env)

  for i in range(args.repeat if args.stack == "tinygrad" else 1):
    if args.stack == "tinygrad":
      row = run_tinygrad(gguf, args.model, args.device, env_extra, args.tag, args.prompt_tokens, args.decode_tokens, args.python, args.device_map)
    else:
      row = run_llamacpp(gguf, args.model, args.n_prompt, args.n_gen, args.llama_bench_repetitions, args.llama_bench, args.tag)
    print(row)
    if args.csv: append_csv(args.csv, row)

if __name__ == "__main__": main()
