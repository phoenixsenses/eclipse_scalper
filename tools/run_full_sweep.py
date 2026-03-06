from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.run_summary import build_run_summary

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day-60 full sweep orchestrator.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-dir", default=f"runs/day60_{int(time.time())}")
    p.add_argument("--candidates-md", required=True, help="Comma list of candidate markdown reports.")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--sides", default="sell,buy")
    p.add_argument("--fees", default="0.0,0.5,0.8")
    p.add_argument("--adverse", default="0.3,0.5,0.7")
    p.add_argument("--horizons", default="120,240,300")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-n-frac", type=float, default=0.0)
    p.add_argument("--pass-threshold", type=float, default=0.5)
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    return p.parse_args()


def _plist(raw: str) -> list[str]:
    return [x.strip() for x in str(raw or "").replace(";", ",").split(",") if x.strip()]


def _run_job(cmd: list[str], env: dict[str, str]) -> int:
    return int(subprocess.call(cmd, env=env))


def main() -> int:
    args = _args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sym in _plist(args.symbols):
        for side in _plist(args.sides):
            for fee in _plist(args.fees):
                for adv in _plist(args.adverse):
                    for h in _plist(args.horizons):
                        run_id = f"{sym}_{side}_h{h}_f{fee}_a{adv}".replace(".", "p")
                        out_json = out_dir / f"{run_id}.json"
                        out_md = out_dir / f"{run_id}.md"
                        cmd = [
                            sys.executable,
                            "-m",
                            "tools.rank_passive_pockets_forward",
                            "--candidates-md",
                            str(args.candidates_md),
                            "--db",
                            str(args.db),
                            "--lookback-min",
                            str(int(args.lookback_min)),
                            "--bucket-sec",
                            str(int(args.bucket_sec)),
                            "--rule",
                            str(args.rule),
                            "--side",
                            side,
                            "--splits",
                            str(int(args.splits)),
                            "--seeds",
                            str(args.seeds),
                            "--min-n",
                            str(int(args.min_n)),
                            "--min-n-frac",
                            str(float(args.min_n_frac)),
                            "--pass-threshold",
                            str(float(args.pass_threshold)),
                            "--horizon-sec",
                            h,
                            "--maker-fee-bps-grid",
                            fee,
                            "--passive-adverse-mult-grid",
                            adv,
                            "--out-json",
                            str(out_json),
                            "--out-md",
                            str(out_md),
                        ]
                        jobs.append({"id": run_id, "cmd": cmd, "out_json": str(out_json), "out_md": str(out_md)})
    results = []
    env = os.environ.copy()
    workers = max(1, int(args.workers))
    if workers == 1:
        for j in jobs:
            print(f"[sweep] run {j['id']}")
            rc = _run_job(j["cmd"], env)
            results.append({"id": j["id"], "rc": int(rc), "out_json": j["out_json"], "out_md": j["out_md"]})
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_job = {}
            for j in jobs:
                print(f"[sweep] queue {j['id']}")
                fut_to_job[ex.submit(_run_job, j["cmd"], env)] = j
            for fut in as_completed(fut_to_job):
                j = fut_to_job[fut]
                rc = 1
                try:
                    rc = int(fut.result())
                except Exception:
                    rc = 1
                print(f"[sweep] done {j['id']} rc={rc}")
                results.append({"id": j["id"], "rc": int(rc), "out_json": j["out_json"], "out_md": j["out_md"]})
    manifest = {"generated_ts": int(time.time()), "jobs": results}
    manifest["run_summary"] = build_run_summary(
        run_type="run_full_sweep",
        inputs={"candidates_md": str(args.candidates_md), "symbols": str(args.symbols), "workers": int(args.workers)},
        metrics={"job_count": len(results), "success_count": sum(1 for r in results if int(r["rc"]) == 0)},
        artifacts={"json": str(out_dir / "manifest.json")},
    )
    mf = out_dir / "manifest.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {mf}")
    return 0 if all(int(r["rc"]) == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
