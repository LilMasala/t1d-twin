"""Fit a personal twin from InSite raw day records.

    python scripts/twin_fit.py \
        --records path/to/raw_days/ --settings therapy_settings.json --person-id alice --holdout-days 3 \
        --out artifacts/twin/alice/twin.json

``--records`` is a directory of ``insite.raw_day.v1`` JSON files (e.g. an
``InSiteRawStore`` ``raw_days/`` folder or an export of Firestore
``users/<uid>/canonical_raw_days/raw_day_records/items``) or one JSON list.
``--settings`` is the app's therapy settings export
(``users/<uid>/therapy_settings/hourly/items``: hourStartUtc, carbRatio,
basalRate, insulinSensitivity), as a JSON list or folder; raw day records do
not carry settings, and without scheduled basal no day can be fitted.
``--population-fits`` is optional: globs of other people's fits, whose posterior
means re-centre the population priors for this person (leave-one-out empirical
Bayes). It helps most when a person has only a week or two of data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from t1d_twin.data import build_timeline, load_records, load_therapy_settings
import glob

from t1d_twin.fit import EMPIRICAL_KNOBS, FitConfig, TwinFit, empirical_priors, fit_twin
from t1d_twin.params import twin_priors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--person-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", help="therapy settings hourly docs (JSON list or folder)")
    ap.add_argument("--sleep", help="sleep daily docs (users/<uid>/sleep/daily/items) as a JSON list or folder")
    ap.add_argument("--sex", choices=["female", "male"], help="enables (female/unknown) or disables (male) the inferred cycle rhythm")
    ap.add_argument("--holdout-days", type=int, default=3)
    ap.add_argument("--map-iters", type=int, default=250)
    ap.add_argument("--iters", type=int, default=400, help="variational iterations after MAP")
    ap.add_argument("--base", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-days", type=int, help="fit only the most recent N days of records")
    ap.add_argument("--min-cgm-coverage", type=float, default=0.7,
                    help="fraction of 5-min bins with CGM a day needs; use ~0.25 for 15-min sensors")
    ap.add_argument("--no-flux", action="store_true", help="disable the fitted glucose disturbance term")
    ap.add_argument("--population-fits", help="comma-separated globs of other people's fits: re-centre population priors on them (leave-one-out)")
    ap.add_argument("--risk-weighted", action="store_true", help="fit residuals in glycaemic risk space (low-range errors weigh more)")
    args = ap.parse_args()

    settings = load_therapy_settings(args.settings) if args.settings else None
    sleep = load_therapy_settings(args.sleep) if args.sleep else None  # same JSON list/folder shape
    records = load_records(args.records)
    if args.max_days and len(records) > args.max_days:
        records = records[-args.max_days:]
    tl = build_timeline(records, args.person_id, settings, sleep_daily=sleep, sex=args.sex)
    print(f"[twin] {tl.n_days} days {tl.day_dates[0]}..{tl.day_dates[-1]}  meals={len(tl.meals)}  "
          f"insulin days={int(tl.insulin_observed.sum())}  median CGM coverage={np.median(tl.cgm_coverage):.0%}")
    for note in tl.notes:
        print(f"[twin] note: {note}")

    cycle_observed = bool(np.any(~np.isnan(tl.days_since_period)))
    q, reports_no_cycle = None, False
    base = None
    if args.population_fits:
        paths = sorted({p for g in args.population_fits.split(",") for p in glob.glob(g)})
        base = empirical_priors([TwinFit.load(p) for p in paths], exclude_person=args.person_id)
        print(f"[twin] population priors from {len(paths)} fits (excluding {args.person_id}): " +
              ", ".join(f"{s.name} {s.prior_mean:+.2f}±{s.prior_sd:.2f}" for s in base.specs if s.name in EMPIRICAL_KNOBS))
    priors = twin_priors(q, has_cycle=not reports_no_cycle, cycle_observed=cycle_observed, sex=args.sex, base=base)
    fit = fit_twin(tl, priors, FitConfig(map_iters=args.map_iters, iters=args.iters, holdout_days=args.holdout_days, base=args.base, seed=args.seed,
                                           min_cgm_coverage=args.min_cgm_coverage, flux=not args.no_flux,
                                           risk_weighted=args.risk_weighted))
    fit.diagnostics["records_max_days"] = args.max_days or 0
    fit.save(args.out)

    d = fit.diagnostics
    print(f"\n[twin] saved {args.out}")
    print(f"  base adult {fit.base}; train RMSE {d['train_rmse_mgdl']:.1f} mg/dL (uncalibrated {d['train_rmse_uncalibrated_mgdl']:.1f})")
    if "holdout" in d and "replay_rmse_mgdl" in d["holdout"]:
        h = d["holdout"]
        print(f"  held-out replay RMSE {h['replay_rmse_mgdl']:.1f} (uncalibrated {h['uncalibrated_replay_rmse_mgdl']:.1f}); 90% interval coverage {h['interval90_coverage']:.0%}")
    print("  parameters (median [p05, p95], how much the data moved it):")
    for name, p in fit.summary["params"].items():
        print(f"    {name:22s} {p['median']:8.3f} [{p['p05']:8.3f}, {p['p95']:8.3f}]  {p['identified']}")
    likely = [c for c in fit.summary["unlogged_meal_candidates"] if c["likely_meal"]]
    print(f"  likely unlogged meals: {len(likely)} of {len(fit.summary['unlogged_meal_candidates'])} flagged rises")
    for c in likely[:20]:
        print(f"    {c['time_utc']}  ~{c['fitted_g']:.0f} g [{c['fitted_g_p05']:.0f}, {c['fitted_g_p95']:.0f}]")


if __name__ == "__main__":
    main()
