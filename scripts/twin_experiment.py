"""Run CR / ISF / basal experiments on fitted twins.

One person (their posterior, their recorded days):

    python scripts/twin_experiment.py \
        --twin artifacts/twin/alice.json --records path/to/alice_raw_days/ --settings alice_settings.json \
        --arm current:1,1,1 --arm cr_x0.9:0.9,1,1 --arm isf_x0.9:1,0.9,1 \
        --out artifacts/twin/alice_experiment.json

Many synthetic people drawn from all fitted twins, living those people's days:

    ... scripts/twin_experiment.py --twin a.json --records a_days/ \
        --twin b.json --records b_days/ --synthetic-people 256 --arm ...

Arms are ``name:cr_mult,isf_mult,basal_mult``; the first arm is the paired
reference. Research estimates from a generic controller, not dosing advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from t1d_twin.data import build_timeline, load_records, load_therapy_settings
from t1d_twin.experiment import Arm, run_settings_experiment
from t1d_twin.fit import TwinFit
from t1d_twin.population import build_population, run_population_experiment


def parse_arm(text: str) -> Arm:
    name, _, mults = text.partition(":")
    cr, isf, basal = (float(v) for v in mults.split(","))
    return Arm(name, cr, isf, basal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", action="append", required=True)
    ap.add_argument("--records", action="append", required=True)
    ap.add_argument("--settings", action="append", default=[], help="therapy settings per --twin, same order")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--synthetic-people", type=int, default=0)
    ap.add_argument("--days", help="first:last local-day indices (default: all but the first)")
    ap.add_argument("--aid", action="store_true", help="add generic AID basal modulation + auto-corrections")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.twin) != len(args.records) or (args.settings and len(args.settings) != len(args.twin)):
        raise SystemExit("give one --records (and --settings, if any) per --twin")

    arms = [parse_arm(a) for a in args.arm]
    days = tuple(int(v) for v in args.days.split(":")) if args.days else None
    settings = [load_therapy_settings(p) for p in args.settings] or [None] * len(args.twin)
    people = []
    for t, r, st in zip(args.twin, args.records, settings):
        fit = TwinFit.load(t)
        people.append((fit, build_timeline(load_records(r), fit.person_id, st)))

    if args.synthetic_people:
        pop = build_population([f for f, _ in people])
        result = run_population_experiment(pop, people, arms, n_people=args.synthetic_people, days=days, aid=args.aid, seed=args.seed)
        runs = result["runs"]
    else:
        runs = [run_settings_experiment(f, tl, arms, samples=args.samples, days=days, aid=args.aid, seed=args.seed) for f, tl in people]
        result = {"runs": runs}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    for run in runs:
        print(f"\n{run['person_id']} ({run['samples']} draws, {len(run['days'])} days, {run['controller']})")
        for name, arm in run["arms"].items():
            m, dlt = arm["metrics"], arm["paired_delta_vs_first_arm"]
            print(f"  {name:12s} TIR {m['tir_70_180']['median']:.0%}  TBR<70 {m['tbr_70']['median']:.1%}  TAR>180 {m['tar_180']['median']:.0%}   "
                  f"dTIR {dlt['tir_70_180']['median']*100:+.1f} [{dlt['tir_70_180']['p05']*100:+.1f}, {dlt['tir_70_180']['p95']*100:+.1f}]  "
                  f"dTBR {dlt['tbr_70']['median']*100:+.2f} [{dlt['tbr_70']['p05']*100:+.2f}, {dlt['tbr_70']['p95']*100:+.2f}]")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
