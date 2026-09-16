"""Convert one HUPA-UCM (preprocessed) patient into InSite raw day records.

HUPA-UCM preprocessed CSVs (``;``-separated, 5-min rows): time, glucose (mg/dL),
calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input.

    python scripts/twin_convert_hupa.py \
        --pid HUPA0001P --out artifacts/twin/hupa/HUPA0001P

Choices, all recorded in each record's metadata:
- timestamps are local Spanish time (Europe/Madrid);
- ``basal_rate`` is insulin delivered in the 5-min row (U), i.e. U/h / 12; runs of
  equal values become temp-basal events;
- ``bolus_volume_delivered`` is U in the row;
- ``carb_input`` is grams for some patients and 10 g "raciones" for others: a
  typical logged value above 15 means grams, otherwise raciones. An overall carb
  ratio outside 3-30 g/U is noted (sparse logging or an unusual ratio);
- patients whose insulin columns are empty, or whose doses/carbs are only ever
  1.0 (event flags, not amounts), are refused rather than converted;
- steps per 5 min become exercise minutes: 0 below 250 steps, 5 at 500+.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from t1d_twin.synthetic import _empty_record, _set_dense

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts/datasets/hupa_ucm"
TZ = "Europe/Madrid"
PLAUSIBLE_CR = (3.0, 30.0)


def carb_unit(df: pd.DataFrame) -> tuple[float | None, str]:
    """Grams or 10 g raciones, by the size of a typical logged value; the carb ratio is only a warning."""
    logged = df.loc[df["carb_input"] > 0, "carb_input"]
    if logged.empty:
        return None, "no carbs recorded"
    factor, name = (1.0, "grams") if logged.median() > 15.0 else (10.0, "raciones (x10 g)")
    bolus = df["bolus_volume_delivered"].sum()
    if bolus > 0:
        cr = df["carb_input"].sum() * factor / bolus
        if not PLAUSIBLE_CR[0] <= cr <= PLAUSIBLE_CR[1]:
            name += f"; overall carbs/bolus {cr:.1f} g/U is outside 3-30 (sparse logging or unusual ratio)"
    return factor, name


def convert(pid: str, root: Path, out: Path) -> dict:
    df = pd.read_csv(root / f"{pid}.csv", sep=";")
    df["t_local"] = pd.to_datetime(df["time"])
    for col in ("glucose", "heart_rate", "steps", "basal_rate", "bolus_volume_delivered", "carb_input"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if (df["basal_rate"].fillna(0) <= 0).all():
        raise SystemExit(f"{pid}: no basal insulin recorded; cannot fit")
    if (df["bolus_volume_delivered"].fillna(0) <= 0).all():
        raise SystemExit(f"{pid}: no bolus insulin recorded; cannot fit")
    positive = lambda c: df.loc[df[c] > 0, c]
    if positive("bolus_volume_delivered").max() <= 1.0 and positive("bolus_volume_delivered").nunique() == 1:
        raise SystemExit(f"{pid}: bolus values are only ever {positive('bolus_volume_delivered').iloc[0]} (event flags, not doses)")
    factor, unit = carb_unit(df)

    zone = ZoneInfo(TZ)
    df["t"] = df["t_local"].dt.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")
    df = df.dropna(subset=["t"]).sort_values("t")
    day0 = df["t_local"].min().normalize()
    day1 = df["t_local"].max().normalize()

    raw_dir = out / "raw_days"
    raw_dir.mkdir(parents=True, exist_ok=True)
    iso = lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = 0
    d = day0
    while d <= day1:
        local0 = datetime(d.year, d.month, d.day, tzinfo=zone)
        a = pd.Timestamp(local0).tz_convert("UTC")
        b = pd.Timestamp(local0 + timedelta(days=1)).tz_convert("UTC")
        bins = int((b - a).total_seconds() // 300)
        rec = _empty_record(local0, TZ, bins, pid)
        rec["source"] = "hupa_ucm"
        rec["metadata"] = {"dataset": "HUPA-UCM", "participant": pid, "carb_unit": unit}
        day = df[(df.t >= a) & (df.t < b)]
        bin_of = lambda ts: int((ts - a).total_seconds() // 300)

        for _, r in day.iterrows():
            k = bin_of(r.t)
            if not 0 <= k < bins:
                continue
            if pd.notna(r.glucose) and r.glucose > 0:
                _set_dense(rec, "cgm_mgdl", k, float(r.glucose))
            if pd.notna(r.heart_rate) and r.heart_rate > 0:
                _set_dense(rec, "heart_rate_bpm", k, float(r.heart_rate))
            if pd.notna(r.steps):
                _set_dense(rec, "exercise_min", k, float(5.0 * np.clip((r.steps - 250.0) / 250.0, 0.0, 1.0)))

        ev = rec["events"]
        boluses = day[day["bolus_volume_delivered"] > 0]
        ev["insulin_bolus_u"]["events"] = [{"timestamp": iso(r.t), "value": float(r["bolus_volume_delivered"])} for _, r in boluses.iterrows()]
        if factor is not None:
            meals = day[day["carb_input"] > 0]
            ev["carbs_g"]["events"] = [{"timestamp": iso(r.t), "value": float(r["carb_input"] * factor)} for _, r in meals.iterrows()]
        # basal: runs of equal per-row delivery -> temp basal events (U/h)
        basal = day[["t", "basal_rate"]].dropna()
        runs = []
        for _, r in basal.iterrows():
            rate = float(r["basal_rate"]) * 12.0
            if runs and abs(runs[-1][1] - rate) < 1e-9 and (r.t - runs[-1][2]).total_seconds() <= 300:
                runs[-1][2] = r.t
            else:
                runs.append([r.t, rate, r.t])
        ev["temp_basal"]["events"] = [{"timestamp": iso(s), "rate": rate, "duration": (e - s).total_seconds() / 60.0 + 5.0} for s, rate, e in runs]

        (raw_dir / f"{rec['record_id']}.json").write_text(json.dumps(rec))
        written += 1
        d = (d + pd.Timedelta(days=1, hours=2)).normalize()

    summary = {"participant": pid, "days": written, "first": str(day0.date()), "last": str(day1.date()), "carb_unit": unit}
    (out / "conversion.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(convert(args.pid, Path(args.root), Path(args.out)))


if __name__ == "__main__":
    main()
