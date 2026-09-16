"""Fit + validate a cohort (T1D-UOM or HUPA-UCM), a few participants at a time.

    python -u scripts/twin_cohort_uom.py \
        --pids 2301,2308,2309,2310,2304,2306,2314,2401,2405 --lanes 2

Each participant: ``twin_fit.py`` (5 held-out days) then ``twin_validate.py``;
logs and outputs land in ``artifacts/twin/uom/<pid>/``. A one-line summary per
participant is appended to ``artifacts/twin/uom/cohort_summary.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_one(pid: str, holdout: int, threads: int, cohort: str = "uom", max_days: int = 0, tag: str = "final", fit_args: list[str] = ()) -> dict:
    out_dir = ROOT / "artifacts/twin" / cohort / pid
    extra = ["--max-days", str(max_days)] if max_days else []
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), PYTHONPATH=f"{ROOT}:{ROOT.parent / 'ChameliaV2'}")
    twin = out_dir / f"twin_{tag}.json"
    with open(out_dir / f"fit_{tag}.log", "w") as log:
        rc = subprocess.run([PY, "-u", "scripts/twin_fit.py", "--records", str(out_dir / "raw_days"), "--person-id", f"uom{pid}",
                             "--holdout-days", str(holdout), "--out", str(twin), *extra, *fit_args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return {"pid": pid, "error": f"fit failed ({rc}); see {out_dir / f'fit_{tag}.log'}"}
    val = out_dir / f"validation_{tag}.json"
    with open(out_dir / f"validate_{tag}.log", "w") as log:
        rc = subprocess.run([PY, "-u", "scripts/twin_validate.py", "--twin", str(twin), "--records", str(out_dir / "raw_days"), "--out", str(val), *extra],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return {"pid": pid, "error": f"validate failed ({rc}); see {out_dir / f'validate_{tag}.log'}"}
    fit = json.loads(twin.read_text())
    if not fit["diagnostics"]["train_rmse_mgdl"] == fit["diagnostics"]["train_rmse_mgdl"]:  # NaN
        return {"pid": pid, "error": f"fit produced NaN; see fit_{tag}.log"}
    v = json.loads(val.read_text())
    pick = lambda block: {m: {"actual": block["summary"]["pooled"][m]["actual"], "twin": block["summary"]["pooled"][m]["twin_median"],
                              "per_day_abs_diff": block["summary"]["per_day"][m]["mean_abs_diff"]}
                          for m in ("tir_70_180", "tbr_70", "tar_180", "mean_mgdl")} if block else None
    params = fit["summary"]["params"]
    return {
        "pid": pid, "cohort": cohort, "tag": tag, "base": fit["base"], "fitted_days": len(fit["fitted_days"]), "clock_offset_min": fit["event_clock_offset_min"],
        "train_rmse": fit["diagnostics"]["train_rmse_mgdl"], "holdout_replay_rmse": fit["diagnostics"].get("holdout", {}).get("replay_rmse_mgdl"),
        "insulin_speed": params["log_insulin_speed"]["median"], "insulin_action_speed": params["log_insulin_action_speed"]["median"],
        "carb_speed": params["log_carb_speed"]["median"], "si": params["log_si"]["median"],
        "fitted_days_metrics": pick(v["fitted"]), "heldout_typical_metrics": pick(v["heldout"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cohort", default="uom", help="subfolder of artifacts/twin (uom, hupa)")
    ap.add_argument("--max-days", type=int, default=0, help="fit only the most recent N days")
    ap.add_argument("--tag", default="final", help="writes twin_<tag>.json, validation_<tag>.json")
    ap.add_argument("--fit-args", default="", help="extra twin_fit.py arguments, e.g. '--population-fits \"artifacts/twin/*/*/twin_final.json\"'")
    args = ap.parse_args()
    import shlex
    summary = ROOT / "artifacts/twin" / args.cohort / "cohort_summary.jsonl"
    with ThreadPoolExecutor(args.lanes) as pool:
        for res in pool.map(lambda p: run_one(p, args.holdout, args.threads, args.cohort, args.max_days, args.tag, shlex.split(args.fit_args)), args.pids.split(",")):
            with open(summary, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
