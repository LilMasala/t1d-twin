"""Do settings-experiment conclusions survive not knowing the twin's absolute dose response?

    python scripts/twin_effect_robustness.py --records artifacts/uom/2307/raw_days \
        --twin base=artifacts/uom/2307/twin_final.json --twin dosing=artifacts/uom/2307/twin_v4.json

The CGM record pins how a meal and its bolus combine, but not how strong each is
on its own, so fits of the same person can disagree by 2x or more on what one
unit of insulin does. A settings experiment is only worth running if its
conclusions do not hinge on that. This runs identical carb ratio, ISF and basal
arms on several fits of one person, over the same days and seeds, and reports
whether they agree on the direction of each effect and how closely they rank
the arms (Spearman correlation of the effects).

By default the arms scale what was actually delivered (``--mode delivery``): a
carb ratio of x0.8 makes every meal bolus 25% larger, and the "current" arm is
the recorded day itself. ``--mode controller`` instead doses with a generic
basal-bolus controller from settings, inferred from the person's dosing when the
records have none. Either way the "current" arm is compared with what actually
happened on those days.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from t1d_twin.data import build_timeline, load_records
from t1d_twin.dosing import infer_settings
from t1d_twin.experiment import Arm, run_delivery_experiment, run_settings_experiment
from t1d_twin.fit import TwinFit
from t1d_twin.validate import metrics as day_metrics

DEFAULT_ARMS = ["current:1,1,1", "cr_x0.8:0.8,1,1", "cr_x1.2:1.2,1,1", "isf_x0.8:1,0.8,1", "isf_x1.2:1,1.2,1",
                "basal_x0.8:1,1,0.8", "basal_x1.2:1,1,1.2"]
METRICS = (("tir_70_180", "TIR"), ("tbr_70", "TBR<70"), ("tar_180", "TAR>180"))


def parse_arm(text: str) -> Arm:
    name, _, mults = text.partition(":")
    cr, isf, basal = (float(v) for v in mults.split(","))
    return Arm(name, cr, isf, basal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--twin", action="append", required=True, help="label=path, one per fit of the same person")
    ap.add_argument("--arm", action="append", help=f"name:cr,isf,basal multipliers (default: {' '.join(DEFAULT_ARMS)})")
    ap.add_argument("--days", type=int, default=10, help="run over the last N fitted days")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["delivery", "controller"], default="delivery",
                    help="delivery: arms scale the recorded insulin (current arm = the real day); controller: a generic basal-bolus controller doses from settings")
    ap.add_argument("--out")
    args = ap.parse_args()
    arms = [parse_arm(a) for a in (args.arm or DEFAULT_ARMS)]
    fits = {label: TwinFit.load(path) for label, path in (t.split("=", 1) for t in args.twin)}

    first = next(iter(fits.values()))
    records = load_records(args.records)
    max_days = first.diagnostics.get("records_max_days") or 0
    if max_days and len(records) > max_days:
        records = records[-max_days:]
    tl = build_timeline(records, first.person_id)
    fitted = sorted(tl.day_dates.index(d) for d in set.intersection(*(set(f.fitted_days) for f in fits.values())))
    run_days = fitted[-args.days:]
    d0, d1 = run_days[0], run_days[-1] + 1
    tl, settings = infer_settings(tl, fitted)

    actual = [day_metrics(tl.cgm[slice(*tl.day_steps(d))]) for d in range(d0, d1)]
    actual_mean = {k: float(np.nanmean([a[k] for a in actual])) for k, _ in METRICS}

    results = {}
    for label, fit in fits.items():
        runner = run_delivery_experiment if args.mode == "delivery" else run_settings_experiment
        res = runner(fit, tl, arms, samples=args.samples, days=(d0, d1), seed=args.seed)
        results[label] = res
        print(f"{label}: base {fit.base}, {len(res['days'])} days, {res['samples']} samples", flush=True)

    labels = list(fits)
    if args.mode == "controller":
        print(f"\n{first.person_id}: settings {settings['source']}")
        print(f"  CR {settings['carb_ratio_g_per_u']:.1f} g/U, ISF {settings['isf_mgdl_per_u']:.0f} mg/dL/U, TDD {settings['tdd_u']:.1f} U")
    else:
        share = results[labels[0]]["meal_bolus_share"]
        print(f"\n{first.person_id}: arms scale the recorded delivery; {100 * share:.0f}% of bolus insulin is meal boluses")
    print("\ncurrent settings vs what actually happened on those days")
    for key, name in METRICS:
        sims = "  ".join(f"{lab} {100 * results[lab]['arms'][arms[0].name]['metrics'][key]['median']:5.1f}" for lab in labels)
        print(f"  {name:8s} actual {100 * actual_mean[key]:5.1f}   {sims}")

    print("\npaired change from current settings, percentage points (median across posterior samples)")
    agreement = {}
    for key, name in METRICS:
        print(f"  {name}")
        deltas = {lab: [] for lab in labels}
        for arm in arms[1:]:
            row = []
            for lab in labels:
                d = 100 * results[lab]["arms"][arm.name]["paired_delta_vs_first_arm"][key]["median"]
                deltas[lab].append(d)
                row.append(f"{lab} {d:+6.1f}")
            print(f"    {arm.name:12s} " + "   ".join(row))
        signs = np.sign(np.array([deltas[lab] for lab in labels]))
        big = np.max(np.abs(np.array([deltas[lab] for lab in labels])), axis=0) >= 0.5  # ignore effects under half a point
        same_sign = bool(np.all(signs[:, big] == signs[0, big])) if big.any() else True
        rank = lambda v: np.argsort(np.argsort(v))
        spearman = [float(np.corrcoef(rank(deltas[labels[0]]), rank(deltas[lab]))[0, 1]) for lab in labels[1:]]
        agreement[key] = {"same_direction": same_sign, "rank_correlation_vs_first": dict(zip(labels[1:], spearman))}
        print(f"    same direction for every arm: {'yes' if same_sign else 'NO'}; rank correlation with {labels[0]}: "
              + ", ".join(f"{lab} {r:+.2f}" for lab, r in zip(labels[1:], spearman)))

    if args.out:
        Path(args.out).write_text(json.dumps({"settings": settings, "actual": actual_mean, "agreement": agreement,
                                              "results": results}, indent=2))


if __name__ == "__main__":
    main()
