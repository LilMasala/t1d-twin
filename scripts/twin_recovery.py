"""Can the twin fitter recover people whose parameters we know?

Generates synthetic people (InSite record format, closed-loop insulin,
miscounted and unlogged meals, daily drift, sensor noise), fits them, and
reports:

- parameter recovery: is the truth inside the 90% posterior interval?
- unlogged-meal flags: recall of true unlogged meals;
- held-out replay RMSE and 90% interval coverage vs. the uncalibrated twin;
- counterfactual agreement: paired TIR/TBR change for CR/ISF/basal arms under
  the fitted twin vs. the true twin, on the same days.

    python scripts/twin_recovery.py \
        --people 4 --days 17 --holdout 3 --out artifacts/twin_recovery.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from t1d_twin.data import build_timeline
from t1d_twin.experiment import Arm, run_settings_experiment
from t1d_twin.fit import FitConfig, fit_twin
from t1d_twin.ode import DT_MIN
from t1d_twin.synthetic import generate_person, truth_as_fit

ARMS = [Arm("current"), Arm("cr_x0.8", cr_mult=0.8), Arm("isf_x0.8", isf_mult=0.8), Arm("basal_x1.2", basal_mult=1.2)]


def recover_one(i: int, args) -> dict:
    records, truth = generate_person(f"syn{i:02d}", n_days=args.days, seed=args.seed + i, aid=not args.no_aid)
    tl = build_timeline(records, truth.person_id, truth.therapy_settings)
    cfg = FitConfig(map_iters=args.map_iters, iters=args.iters, holdout_days=args.holdout, base=truth.base if args.base == "truth" else args.base, seed=i)
    fit = fit_twin(tl, config=cfg)

    params = {}
    for k, name in enumerate(fit.param_names):
        lo, hi = fit.loc[k] - 1.645 * fit.sd[k], fit.loc[k] + 1.645 * fit.sd[k]
        params[name] = {
            "truth": truth.globals_u[k], "loc": fit.loc[k], "sd": fit.sd[k],
            "in_90": bool(lo <= truth.globals_u[k] <= hi),
            "identified": fit.summary["params"][name]["identified"],
        }

    tol = int(45 / DT_MIN)
    fitted_days = set(fit.fitted_days)
    true_unlogged = [s for s in truth.unlogged_steps if tl.day_dates[int(tl.day_index[s])] in fitted_days]
    flagged = [c["time_utc"] for c in fit.summary["unlogged_meal_candidates"] if c["likely_meal"]]
    flagged_steps = [s for s, c in zip(fit.unlogged_meal_steps, fit.summary["unlogged_meal_candidates"]) if c["likely_meal"]]
    hits = sum(any(abs(s - t) <= tol for s in flagged_steps) for t in true_unlogged)

    n_train = args.days - args.holdout
    truth_fit = truth_as_fit(truth, tl, fit.priors())
    cf_days = (1, n_train)
    fitted_cf = run_settings_experiment(fit, tl, ARMS, samples=args.samples, days=cf_days, aid=not args.no_aid, seed=7)
    truth_cf = run_settings_experiment(truth_fit, tl, ARMS, samples=4, days=cf_days, aid=not args.no_aid, seed=7)
    cf = {}
    for arm in ARMS[1:]:
        row = {}
        for metric in ("tir_70_180", "tbr_70", "mean_mgdl"):
            f = fitted_cf["arms"][arm.name]["paired_delta_vs_first_arm"][metric]
            t = truth_cf["arms"][arm.name]["paired_delta_vs_first_arm"][metric]["median"]
            row[metric] = {"truth": t, "fitted_median": f["median"], "fitted_p05": f["p05"], "fitted_p95": f["p95"],
                           "truth_in_interval": bool(f["p05"] <= t <= f["p95"]), "sign_agrees": bool(np.sign(t) == np.sign(f["median"]) or abs(t) < 1e-3)}
        cf[arm.name] = row

    return {
        "person": truth.person_id, "true_base": truth.base, "fitted_base": fit.base,
        "diagnostics": {k: fit.diagnostics[k] for k in ("train_rmse_mgdl", "train_rmse_uncalibrated_mgdl", "fit_seconds", "holdout") if k in fit.diagnostics},
        "params": params,
        "unlogged_meals": {"true_in_fitted_days": len(true_unlogged), "flagged_likely": len(flagged), "recall": hits / max(1, len(true_unlogged))},
        "counterfactual": cf,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--days", type=int, default=17)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--map-iters", type=int, default=250)
    ap.add_argument("--iters", type=int, default=400, help="variational iterations after MAP")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--base", default="truth", help="'truth' (parameter recovery) or 'auto' (end-to-end)")
    ap.add_argument("--no-aid", action="store_true")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="artifacts/twin_recovery.json")
    args = ap.parse_args()

    results = []
    for i in range(args.people):
        results.append(recover_one(i, args))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "people": results}, indent=2))

    names = list(results[0]["params"])
    print("\nparameter recovery (truth inside 90% interval, across people)")
    for n in names:
        rows = [r["params"][n] for r in results]
        z = [abs(r["loc"] - r["truth"]) / max(r["sd"], 1e-9) for r in rows]
        print(f"  {n:22s} in90={sum(r['in_90'] for r in rows)}/{len(rows)}  median|z|={np.median(z):5.2f}  identified={[r['identified'] for r in rows]}")
    all_in = [r["params"][n]["in_90"] for r in results for n in names]
    print(f"  overall in90: {np.mean(all_in):.2f} (target ~0.90)")
    print("\nprediction")
    for r in results:
        d = r["diagnostics"]; h = d.get("holdout", {})
        print(f"  {r['person']} base {r['true_base']}->{r['fitted_base']}: train {d['train_rmse_mgdl']:.1f} (uncal {d['train_rmse_uncalibrated_mgdl']:.1f})  "
              f"holdout {h.get('replay_rmse_mgdl', float('nan')):.1f} (uncal {h.get('uncalibrated_replay_rmse_mgdl', float('nan')):.1f}) cover90 {h.get('interval90_coverage', float('nan')):.2f}  "
              f"unlogged recall {r['unlogged_meals']['recall']:.2f}  {d['fit_seconds']:.0f}s")
    print("\ncounterfactual: paired change vs current settings (truth | fitted median [p05, p95])")
    for r in results:
        for arm, row in r["counterfactual"].items():
            t = row["tir_70_180"]; b = row["tbr_70"]
            print(f"  {r['person']} {arm:10s} TIR {t['truth']*100:+5.1f} | {t['fitted_median']*100:+5.1f} [{t['fitted_p05']*100:+5.1f},{t['fitted_p95']*100:+5.1f}]  "
                  f"TBR70 {b['truth']*100:+5.2f} | {b['fitted_median']*100:+5.2f} [{b['fitted_p05']*100:+5.2f},{b['fitted_p95']*100:+5.2f}]")


if __name__ == "__main__":
    main()
