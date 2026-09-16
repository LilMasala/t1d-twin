"""Export fitted participants to T1DSim_AI's input format (research benchmark only).

    python scripts/benchmark_t1dsimai_export.py \
        --cohort uom --pids 2301,2304,2307,2308,2309,2310

For each participant, writes ``artifacts/benchmark/t1dsimai/<cohort>/<pid>/data.csv``
with T1DSim_AI's columns on a 5-min grid, using the SAME split as our twin:
``is_train`` = our fitted days, test = our held-out days. Rows of days neither
fitted nor usable held-out get NaN CGM so no sequence is drawn from them.

Units follow the package's own example data and scalers: ``input_insulin`` in
U/h (basal + bolus delivered in the bin, as a rate), ``input_meal_carbs`` in g
per bin, ``heart_rate_WRTbaseline`` = heart rate minus the person's mean over
training rows (0 when missing), ``sleep_efficiency`` = 0.85 while asleep and 0
otherwise (T1D-UOM sleep windows; HUPA has no sleep data, so 0).

T1DSim_AI is licensed for non-profit / academic research use only; this export
exists for a benchmark and is not part of InSite.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from t1d_twin import ode
from t1d_twin.data import build_timeline, load_records, shift_events
from t1d_twin.fit import TwinFit
from t1d_twin.model import TwinProblem

ROOT = Path(__file__).resolve().parents[1]
ASLEEP_EFFICIENCY = 0.85
STEPS_PER_BIN = int(5 / ode.DT_MIN)  # 2-min grid -> 5-min bins (2.5 steps; see bin_of below)


def export_one(cohort: str, pid: str, max_days: int = 0) -> dict:
    fit = TwinFit.load(ROOT / "artifacts/twin" / cohort / pid / "twin_final.json")
    records = load_records(ROOT / "artifacts/twin" / cohort / pid / "raw_days")
    if max_days and len(records) > max_days:
        records = records[-max_days:]
    tl = shift_events(build_timeline(records, fit.person_id), fit.event_clock_offset_min)

    fitted = {tl.day_dates.index(d) for d in fit.fitted_days}
    last_fitted = max(fitted)
    candidates = list(range(last_fitted + 1, tl.n_days))
    try:
        usable_heldout = set(TwinProblem(tl, fit.base, candidates, fit.priors(), unlogged=[], min_cgm_coverage=0.25).days) if candidates else set()
    except ValueError:
        usable_heldout = set()

    # 5-min bins over the local days from the first fitted day to the last day
    first = min(fitted)
    s0, _ = tl.day_steps(first)
    _, s_end = tl.day_steps(tl.n_days - 1)
    steps = np.arange(s0, s_end)
    bins = ((steps - s0) * ode.DT_MIN // 5).astype(int)
    n_bins = int(bins.max()) + 1

    def per_bin(values: np.ndarray, how: str) -> np.ndarray:
        df = pd.DataFrame({"b": bins, "v": values[steps]})
        agg = getattr(df.groupby("b")["v"], how)()
        return agg.reindex(range(n_bins)).to_numpy()

    cgm = per_bin(tl.cgm, "mean")                                        # one reading per bin (NaN if none)
    insulin_uph = per_bin((tl.basal_upm + tl.bolus_upm) * 60.0, "mean")  # U/h over the bin
    hr = per_bin(tl.hr, "mean")
    asleep = per_bin(np.nan_to_num(tl.asleep, nan=0.0), "max")
    day = per_bin(tl.day_index.astype(float), "min").astype(int)
    carbs = np.zeros(n_bins)
    for m in tl.meals:
        if m.logged and s0 <= m.step < s_end:
            carbs[int((m.step - s0) * ode.DT_MIN // 5)] += m.grams

    t_utc = [tl.t0_utc + timedelta(minutes=ode.DT_MIN * int(s0) + 5 * b) for b in range(n_bins)]
    zone = ZoneInfo(tl.tz)
    t_local = [t.astimezone(zone) for t in t_utc]
    is_train = np.array([d in fitted for d in day])
    usable = np.array([(d in fitted) or (d in usable_heldout) for d in day])
    cgm = np.where(usable, cgm, np.nan)

    hr_baseline = float(np.nanmean(hr[is_train])) if np.any(~np.isnan(hr[is_train])) else 0.0
    hour = np.array([t.hour + t.minute / 60.0 for t in t_local])
    out = pd.DataFrame({
        "time": [t.strftime("%Y-%m-%d %H:%M") for t in t_local],
        "datetime_local": [t.strftime("%Y-%m-%d %H:%M") for t in t_local],
        "time_utc": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in t_utc],
        "heart_rate": np.where(np.isnan(hr), hr_baseline, hr),
        "output_cgm": cgm,
        "input_insulin": np.nan_to_num(insulin_uph, nan=0.0),
        "input_meal_carbs": carbs,
        "heart_rate_WRTbaseline": np.where(np.isnan(hr), 0.0, hr - hr_baseline),
        "feat_is_weekend": [int(t.weekday() >= 5) for t in t_local],
        "feat_hour_of_day_sin": np.sin(2 * np.pi * hour / 24.0),
        "feat_hour_of_day_cos": np.cos(2 * np.pi * hour / 24.0),
        "sleep_efficiency": asleep * ASLEEP_EFFICIENCY,
        "is_train": is_train,
        "local_day": [tl.day_dates[d] for d in day],
    })
    out_dir = ROOT / "artifacts/benchmark/t1dsimai" / cohort / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "data.csv", index=False)
    meta = {"cohort": cohort, "pid": pid, "bins": n_bins, "train_bins": int(is_train.sum()),
            "heldout_days_usable": sorted(tl.day_dates[d] for d in usable_heldout),
            "train_carbs_g": float(carbs[is_train].sum()), "test_carbs_g": float(carbs[~is_train & usable].sum()),
            "max_days": max_days}
    (out_dir / "export.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args()
    for pid in args.pids.split(","):
        print(json.dumps(export_one(args.cohort, pid, args.max_days)))


if __name__ == "__main__":
    main()
