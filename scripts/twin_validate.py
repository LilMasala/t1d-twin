"""Twin vs actual glycemic outcomes (TIR / TBR / TAR / mean / CV), fitted and held-out days.

    python scripts/twin_validate.py \
        --twin artifacts/twin/uom/2307/twin_full.json --records artifacts/twin/uom/2307/raw_days \
        --out artifacts/twin/uom/2307/validation.json

Held-out days are the recorded days after the last fitted day. Pass the same
``--settings`` / ``--sleep`` / ``--sex`` used for fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from t1d_twin.data import build_timeline, load_records, load_therapy_settings, shift_events
from t1d_twin.fit import TwinFit
from t1d_twin.validate import METRICS, validate_fitted_days, validate_heldout_days

LABEL = {"tir_70_180": "TIR 70-180", "tbr_70": "TBR <70", "tbr_54": "TBR <54", "tar_180": "TAR >180",
         "tar_250": "TAR >250", "mean_mgdl": "mean mg/dL", "cv": "CV"}


def show(result: dict) -> None:
    s = result["summary"]
    print(f"\n{result['mode']} — {s['days']} days")
    print(f"  {'metric':12s} {'actual':>8s} {'twin [5-95%]':>24s}   per-day: {'mean diff':>9s} {'|diff|':>7s} {'corr':>5s} {'actual in 90%':>13s}")
    for m in METRICS:
        p, d = s["pooled"][m], s["per_day"][m]
        pct = m != "mean_mgdl"
        f = (lambda v: f"{100 * v:6.1f}%") if pct else (lambda v: f"{v:7.1f}")
        fd = (lambda v: f"{100 * v:+7.1f}pp") if pct else (lambda v: f"{v:+8.1f}")
        print(f"  {LABEL[m]:12s} {f(p['actual']):>8s} {f(p['twin_median']):>8s} [{f(p['twin_p05'])},{f(p['twin_p95'])}]   "
              f"{fd(d['mean_diff']):>9s} {fd(d['mean_abs_diff']).replace('+', ''):>7s} {d['day_to_day_corr']:5.2f} {100 * d['actual_in_twin_90']:12.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--settings")
    ap.add_argument("--sleep")
    ap.add_argument("--sex", choices=["female", "male"])
    ap.add_argument("--max-days", type=int, help="same as used for fitting")
    ap.add_argument("--out")
    args = ap.parse_args()

    fit = TwinFit.load(args.twin)
    settings = load_therapy_settings(args.settings) if args.settings else None
    sleep = load_therapy_settings(args.sleep) if args.sleep else None
    records = load_records(args.records)
    if args.max_days and len(records) > args.max_days:
        records = records[-args.max_days:]
    tl = build_timeline(records, fit.person_id, settings, sleep_daily=sleep, sex=args.sex)
    tl = shift_events(tl, fit.event_clock_offset_min)

    fitted = validate_fitted_days(fit, tl)
    last_fitted = max(tl.day_dates.index(d) for d in fit.fitted_days)
    heldout_days = list(range(last_fitted + 1, tl.n_days))
    held = held_prior = None
    if heldout_days:
        try:
            held = validate_heldout_days(fit, tl, heldout_days)
            held_prior = validate_heldout_days(fit, tl, heldout_days, unknowns="prior")
        except ValueError as exc:  # e.g. every held-out day lacks insulin or a meal log
            print(f"\nheld-out days not usable: {exc}")
    show(fitted)
    if held:
        show(held)
        show(held_prior)
    if args.out:
        Path(args.out).write_text(json.dumps({"fitted": fitted, "heldout": held, "heldout_prior_unknowns": held_prior}, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
