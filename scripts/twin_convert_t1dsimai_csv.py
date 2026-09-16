"""Convert a T1DSim_AI-format CSV (e.g. its bundled example person) into InSite raw day records.

    python scripts/twin_convert_t1dsimai_csv.py \
        --csv artifacts/external/T1DSim_AI/example/example_model/data_example.csv \
        --pid DT_Example --out artifacts/twin/t1dsimai_example/DT_Example

The format has a single insulin column (U/h over each 5-min row, basal and boluses
combined): rows above 3 U/h become a bolus of rate/12 U with no basal in that row,
the rest are basal rates. Carbs are grams per row. ``sleep_efficiency`` > 0 marks
the row asleep. Local times are written as UTC with tz "UTC" (no DST in the data).
Also copies the CSV (plus a ``time_utc`` column) to the benchmark folder so both
twins are scored on the same rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from t1d_twin.synthetic import _empty_record, _set_dense

ROOT = Path(__file__).resolve().parents[1]
BOLUS_RATE_UPH = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default="t1dsimai_example")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["t"] = pd.to_datetime(df["datetime_local"], format="%m/%d/%y %H:%M")
    df["day"] = df.t.dt.normalize()
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = Path(args.out) / "raw_days"
    raw.mkdir(parents=True, exist_ok=True)

    for day, rows in df.groupby("day"):
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        rec = _empty_record(start, "UTC", 288, args.pid)
        rec["source"] = "t1dsimai_csv"
        rec["metadata"] = {"dataset": "T1DSim_AI example CSV", "participant": args.pid,
                           "insulin_split": f"rows > {BOLUS_RATE_UPH} U/h are boluses"}
        boluses, basal_runs = [], []
        for _, r in rows.iterrows():
            b = int((r.t - day).total_seconds() // 300)
            if pd.notna(r.output_cgm):
                _set_dense(rec, "cgm_mgdl", b, float(r.output_cgm))
            if pd.notna(r.heart_rate):
                _set_dense(rec, "heart_rate_bpm", b, float(r.heart_rate))
            _set_dense(rec, "sleep_stage", b, "core" if r.sleep_efficiency > 0 else "awake")
            ts = start + timedelta(minutes=5 * b)
            rate = float(r.input_insulin)
            if rate > BOLUS_RATE_UPH:
                boluses.append({"timestamp": iso(ts), "value": rate / 12.0})
                rate = 0.0
            if basal_runs and abs(basal_runs[-1][1] - rate) < 1e-9:
                basal_runs[-1][2] += 5.0
            else:
                basal_runs.append([ts, rate, 5.0])
            if r.input_meal_carbs > 0:
                rec["events"]["carbs_g"]["events"].append({"timestamp": iso(ts), "value": float(r.input_meal_carbs)})
        rec["events"]["insulin_bolus_u"]["events"] = boluses
        rec["events"]["temp_basal"]["events"] = [{"timestamp": iso(t), "rate": rate, "duration": dur} for t, rate, dur in basal_runs]
        (raw / f"{rec['record_id']}.json").write_text(json.dumps(rec))

    bench = ROOT / "artifacts/benchmark/t1dsimai" / args.cohort / args.pid
    bench.mkdir(parents=True, exist_ok=True)
    df["time_utc"] = df.t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.drop(columns=["t", "day"]).to_csv(bench / "data.csv", index=False)
    (bench / "export.json").write_text(json.dumps({"cohort": args.cohort, "pid": args.pid, "max_days": 0,
                                                   "source": str(args.csv)}, indent=2))
    n_test_days = int(df.loc[~df.is_train.astype(bool), "day"].nunique())
    print(json.dumps({"days": int(df.day.nunique()), "test_days": n_test_days}))


if __name__ == "__main__":
    main()
