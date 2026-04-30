"""Run backtest_cme_v5.py in N parallel shards, then aggregate.

Usage:
    python run_parallel_backtest.py --shards 6 -- \
        --output-root models_cme_v5/artifacts \
        --run-name baseline \
        --initial-train-end 2023-12-31 \
        --curriculum --phase1-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "backtest_cme_v5.py"


def _resolve_out_dir(rest: list[str]) -> Path:
    out_root = None
    run_name = "backtest"
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--output-root" and i + 1 < len(rest):
            out_root = Path(rest[i + 1])
            i += 2
        elif a.startswith("--output-root="):
            out_root = Path(a.split("=", 1)[1])
            i += 1
        elif a == "--run-name" and i + 1 < len(rest):
            run_name = rest[i + 1]
            i += 2
        elif a.startswith("--run-name="):
            run_name = a.split("=", 1)[1]
            i += 1
        else:
            i += 1
    if out_root is None:
        raise SystemExit("--output-root must appear in the forwarded args")
    return out_root / run_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--no-aggregate", action="store_true")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args()

    rest = args.rest
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        p.print_help()
        sys.exit(1)

    out_dir = _resolve_out_dir(rest)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_shards"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parallel] {args.shards} shards -> {out_dir}")
    procs = []
    t0 = time.time()
    for i in range(args.shards):
        cmd = [sys.executable, str(SCRIPT), *rest,
               "--window-shards", str(args.shards),
               "--shard-idx", str(i)]
        log_path = log_dir / f"shard{i}.log"
        log_fh = open(log_path, "w")
        print(f"[launch shard {i}] log={log_path}")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((i, proc, log_fh, log_path))

    failed = []
    for i, proc, log_fh, log_path in procs:
        rc = proc.wait()
        log_fh.close()
        if rc != 0:
            failed.append((i, rc, log_path))
            print(f"[fail] shard {i} rc={rc}  log={log_path}")
        else:
            print(f"[done] shard {i}  ({time.time()-t0:.0f}s elapsed)")

    if failed:
        print(f"\n[abort] {len(failed)} shard(s) failed; skipping aggregate.")
        sys.exit(1)

    print(f"\n[shards done] {time.time()-t0:.0f}s wall-clock for {args.shards} parallel shards.")

    if args.no_aggregate:
        print("[skip] --no-aggregate set; not merging shard outputs.")
        return

    print(f"\n[aggregate] merging shard outputs in {out_dir / '_shards'}")
    agg_cmd = [sys.executable, str(SCRIPT), *rest, "--aggregate-only"]
    rc = subprocess.call(agg_cmd)
    if rc != 0:
        print(f"[fail] aggregate exited rc={rc}")
        sys.exit(rc)
    print(f"\n[parallel] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
