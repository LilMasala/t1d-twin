"""Re-validate every fitted twin in a cohort folder and print a pooled table.

    python scripts/twin_cohort_report.py --cohort uom --cohort hupa

Reads ``artifacts/twin/<cohort>/<pid>/twin_final.json`` + ``raw_days`` (fits are
re-centred on load), scores fitted days and the held-out typical-day forecast,
and writes ``artifacts/twin/cohort_report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from t1d_twin.data import build_timeline, load_records, shift_events
from t1d_twin.fit import TwinFit
from t1d_twin.validate import validate_fitted_days, validate_heldout_days

ROOT = Path(__file__).resolve().parents[1]
METRICS = (("tir_70_180", "TIR"), ("tbr_70", "TBR<70"), ("tar_180", "TAR>180"), ("mean_mgdl", "mean"))


def one(cohort: str, pid_dir: Path, max_days: int, twin_name: str = "twin_final.json") -> dict | None:
    twin = pid_dir / twin_name
    if not twin.exists():
        return None
    fit = TwinFit.load(twin)
    records = load_records(pid_dir / "raw_days")
    if max_days and len(records) > max_days:
        records = records[-max_days:]
    tl = shift_events(build_timeline(records, fit.person_id), fit.event_clock_offset_min)
    row = {"cohort": cohort, "pid": pid_dir.name, "twin": twin_name, "fitted_days": len(fit.fitted_days),
           "train_rmse": fit.diagnostics.get("train_rmse_mgdl"), "carb_count_bias_x": float(np.exp(fit.carb_count_bias))}
    row["fitted"] = validate_fitted_days(fit, tl)["summary"]
    last = tl.day_dates.index(fit.fitted_days[-1])
    try:
        held = validate_heldout_days(fit, tl, list(range(last + 1, tl.n_days)))
        row["heldout"] = held["summary"]
    except ValueError as exc:
        row["heldout"] = None
        row["heldout_reason"] = str(exc)[:120]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action="append", required=True)
    ap.add_argument("--max-days", type=int, default=0, help="must match what the fits used (HUPA long patients: 90)")
    ap.add_argument("--twin-name", default="twin_final.json", help="which fit per participant, e.g. twin_v2.json")
    ap.add_argument("--out", help="output path (default: artifacts/twin/cohort_report_<cohorts>_<twin>.json)")
    args = ap.parse_args()
    rows = []
    for cohort in args.cohort:
        for pid_dir in sorted((ROOT / "artifacts/twin" / cohort).iterdir()):
            if pid_dir.is_dir():
                try:
                    r = one(cohort, pid_dir, args.max_days, args.twin_name)
                except Exception as exc:  # report and keep going
                    r = {"cohort": cohort, "pid": pid_dir.name, "error": f"{type(exc).__name__}: {exc}"[:160]}
                if r:
                    rows.append(r)
    out = Path(args.out) if args.out else ROOT / f"artifacts/twin/cohort_report_{'_'.join(args.cohort)}_{Path(args.twin_name).stem}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")

    f = lambda v, m: f"{v:5.0f}" if m == "mean_mgdl" else f"{100 * v:5.1f}"
    print(f"{'cohort':6s} {'pid':10s} {'days':>4s} {'rmse':>5s} | fitted actual/twin: " + "  ".join(f"{n:>11s}" for _, n in METRICS) + " | held-out typical actual/twin: " + "  ".join(f"{n:>11s}" for _, n in METRICS))
    for r in rows:
        if "error" in r:
            print(f"{r['cohort']:6s} {r['pid']:10s} ERROR {r['error']}")
            continue
        fit_cols = "  ".join(f"{f(r['fitted']['pooled'][m]['actual'], m)}/{f(r['fitted']['pooled'][m]['twin_median'], m)}" for m, _ in METRICS)
        held_cols = ("  ".join(f"{f(r['heldout']['pooled'][m]['actual'], m)}/{f(r['heldout']['pooled'][m]['twin_median'], m)}" for m, _ in METRICS)
                     if r["heldout"] else "n/a (" + r.get("heldout_reason", "")[:40] + ")")
        print(f"{r['cohort']:6s} {r['pid']:10s} {r['fitted_days']:4d} {r['train_rmse']:5.1f} | {fit_cols} | {held_cols}")

    print("\npooled across participants (mean absolute error of the per-person pooled metric; mean signed error)")
    for part in ("fitted", "heldout"):
        ok = [r for r in rows if "error" not in r and r.get(part)]
        for m, name in METRICS:
            d = np.array([r[part]["pooled"][m]["twin_median"] - r[part]["pooled"][m]["actual"] for r in ok])
            scale = 1 if m == "mean_mgdl" else 100
            print(f"  {part:8s} {name:8s} n={len(ok):2d}  MAE {scale * np.abs(d).mean():6.1f}  bias {scale * d.mean():+6.1f}")


if __name__ == "__main__":
    main()
