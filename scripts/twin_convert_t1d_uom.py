"""Convert one T1D-UOM participant into InSite raw day records for twin fitting.

T1D-UOM (Zenodo 10.5281/zenodo.15806142, CC-BY-4.0): CGM, pump basal or
long-acting injections, boluses, meals, Garmin heart rate, activity and sleep.

    python scripts/twin_convert_t1d_uom.py \
        --pid 2301 --out artifacts/twin/uom/2301

Writes ``<out>/raw_days/*.json``; fit with ``scripts/twin_fit.py --records <out>/raw_days``
(no ``--settings``: basal delivery comes from the insulin records themselves).

Choices, all visible in the records' metadata:
- timestamps are day-first local UK time (Europe/London), despite the README;
- CGM mmol/L x 18.0; several readings in one 5-min bin are averaged;
- pump basal rows are U/h rate changes, each held until the next row (max 24 h;
  the final row only for 3 usual gaps, max 6 h);
- long-acting ("L") injections become a flat dose/24 U/h for 24 h. The twin's
  insulin model is rapid-acting only, so this is an approximation;
- heart rate 0 is missing; ``sleep_level`` is not used (it is not an asleep
  flag), sleep comes from the per-night sleep windows;
- exercise minutes come from 15-min activity blocks with MET >= 3;
- Garmin stress (0-100; negative = no reading) goes into a ``stress_level`` dense
  stream, which the twin reads as acute stress (not part of the app contract);
- meals are logged in a separate nutrition app, often far from when they were
  eaten; each meal moves to the nearest bolus within ``--snap-meals-min``
  (the bolus is the better record of eating time). Set 0 to keep app times.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from t1d_twin.synthetic import _empty_record, _set_dense, _set_slow

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts/datasets/t1d_uom_15806142/extracted/sharpic-ManchesterCSCoordinatedDiabetesStudy-fdbd74f"
TZ = "Europe/London"
MMOL_TO_MGDL = 18.0
EXERCISE_MET = 3.0


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def _local(series: pd.Series) -> pd.Series:
    """Day-first naive local times -> UTC-aware timestamps (ambiguous DST hour -> standard time)."""
    t = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return t.dt.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")


def _iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def snap_meals_to_boluses(meals: pd.DataFrame, bolus: pd.DataFrame, window_min: float) -> tuple[pd.DataFrame, int]:
    """Move each meal to the nearest bolus within ``window_min``; returns (meals, n moved)."""
    if window_min <= 0 or meals is None or not len(meals):
        return meals, 0
    bt = bolus.t.sort_values().to_numpy()
    moved = 0
    times = []
    for t in meals.t.to_numpy():
        i = np.searchsorted(bt, t)
        near = [bt[j] for j in (i - 1, i) if 0 <= j < len(bt)]
        best = min(near, key=lambda b: abs(b - t)) if near else None
        if best is not None and abs(best - t) <= np.timedelta64(int(window_min * 60), "s"):
            moved += int(best != t)
            times.append(best)
        else:
            times.append(t)
    return meals.assign(t=pd.to_datetime(times, utc=True)), moved


def convert(pid: str, root: Path, out: Path, snap_meals_min: float = 60.0) -> dict:
    cgm = _read(root / "Glucose Data" / f"UoMGlucose{pid}.csv")
    basal = _read(root / "Insulin Data" / "Basal Data" / f"UoMBasal{pid}.csv")
    bolus = _read(root / "Insulin Data" / "Bolus Data" / f"UoMBolus{pid}.csv")
    meals = _read(root / "Nutrition Data" / f"UoMNutrition{pid}.csv")
    hr = _read(root / "Sleep Data" / f"UoMsleep{pid}.csv")
    sleep = _read(root / "Sleep Data" / f"UoM{pid}sleeptime.csv")
    activity = _read(root / "Activity Data" / f"UoMActivity{pid}.csv")
    if cgm is None or basal is None or bolus is None:
        raise SystemExit(f"{pid}: needs glucose, basal and bolus files")

    cgm = cgm.assign(t=_local(cgm["bg_ts"]), mgdl=cgm["value"] * MMOL_TO_MGDL).dropna(subset=["t"])
    basal = basal.assign(t=_local(basal["basal_ts"])).dropna(subset=["t"]).sort_values("t")
    bolus = bolus.assign(t=_local(bolus["bolus_ts"])).dropna(subset=["t"])
    kinds = set(basal["insulin_kind"].dropna().unique())

    # Days where CGM, basal and boluses all exist.
    start = max(cgm.t.min(), basal.t.min(), bolus.t.min()).tz_convert(TZ).normalize()
    end = min(cgm.t.max(), basal.t.max(), bolus.t.max()).tz_convert(TZ).normalize()
    zone = ZoneInfo(TZ)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d = (d + pd.Timedelta(days=1, hours=2)).normalize()

    # Temp basal events: each row holds until the next (<= 24 h); L doses spread flat.
    tb_events = []
    rows = basal.drop_duplicates(subset=["t"], keep="last").reset_index(drop=True)
    gaps = rows.t.diff().dt.total_seconds().dropna() / 60.0
    # the last row is held for a few usual gaps, not a whole day past the end of the recording
    last_hold = float(min(360.0, 3.0 * gaps.median())) if len(gaps) else 60.0
    for i, r in rows.iterrows():
        if r["insulin_kind"] == "L":
            tb_events.append((r.t, float(r["basal_dose"]) / 24.0, 1440.0))
        else:
            nxt = rows.t.iloc[i + 1] if i + 1 < len(rows) else r.t + pd.Timedelta(minutes=last_hold)
            dur = min((nxt - r.t).total_seconds() / 60.0, 1440.0)
            tb_events.append((r.t, float(r["basal_dose"]), dur))

    garmin = None
    if hr is not None:
        garmin = hr.assign(t=_local(hr["sleep_ts"]))
        garmin = garmin[garmin.t.notna()]
        hr = garmin[garmin["heart_rate"] > 0]
    if sleep is not None:
        sleep = sleep.assign(s=_local(sleep["start_date_ts"]))
        sleep = sleep.dropna(subset=["s"])
        sleep = sleep.assign(e=sleep.s + pd.to_timedelta(sleep["duration_in_sec"], unit="s"))
    if activity is not None:
        activity = activity.assign(t=_local(activity["activity_ts"]))
        activity = activity[(activity["met"] >= EXERCISE_MET) & activity.t.notna()]
    if meals is not None:
        meals = meals.assign(t=_local(meals["meal_ts"]))
        meals = meals[(pd.to_numeric(meals["carbs_g"], errors="coerce") > 0) & meals.t.notna()]
        meals, n_snapped = snap_meals_to_boluses(meals, bolus, snap_meals_min)
    else:
        n_snapped = 0

    raw_dir = out / "raw_days"
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for day in days:
        local0 = datetime(day.year, day.month, day.day, tzinfo=zone)
        local1 = local0 + timedelta(days=1)
        a, b = pd.Timestamp(local0).tz_convert("UTC"), pd.Timestamp(local1).tz_convert("UTC")
        bins = int((b - a).total_seconds() // 300)
        rec = _empty_record(local0, TZ, bins, f"uom{pid}")
        rec["source"] = "t1d_uom"
        rec["metadata"] = {"dataset": "T1D-UOM", "participant": pid, "basal_kinds": sorted(kinds),
                           "long_acting_as_flat_infusion": "L" in kinds, "meals_snapped_to_bolus": n_snapped}

        in_day = lambda df: df[(df.t >= a) & (df.t < b)]
        bin_of = lambda ts: int((ts - a).total_seconds() // 300)

        for b_idx, vals in in_day(cgm).groupby(in_day(cgm).t.map(bin_of))["mgdl"]:
            _set_dense(rec, "cgm_mgdl", int(b_idx), float(vals.mean()))
        if hr is not None:
            day_hr = in_day(hr)
            for b_idx, vals in day_hr.groupby(day_hr.t.map(bin_of))["heart_rate"]:
                _set_dense(rec, "heart_rate_bpm", int(b_idx), float(vals.mean()))
            day_garmin = in_day(garmin)
            stress_rows = day_garmin[day_garmin["stress_level_value"] >= 0]
            if len(stress_rows):
                rec["dense"]["stress_level"] = {"unit": "score", "values": [None] * bins, "observed_flag": [0] * bins}
                for b_idx, vals in stress_rows.groupby(stress_rows.t.map(bin_of))["stress_level_value"]:
                    _set_dense(rec, "stress_level", int(b_idx), float(vals.mean()))
            rhr = day_hr["resting_heart_rate"][day_hr["resting_heart_rate"] > 0]
            if len(rhr):
                _set_slow(rec, "resting_hr_bpm", float(rhr.median()))
        if activity is not None:
            ex = np.zeros(bins)
            for _, r in in_day(activity).iterrows():
                b0 = bin_of(r.t)
                minutes = float(r["active_time_s"]) / 60.0
                for k in range(3):
                    if 0 <= b0 + k < bins:
                        ex[b0 + k] += minutes / 3.0
            for k in range(bins):
                _set_dense(rec, "exercise_min", k, float(min(ex[k], 5.0)))
        if sleep is not None:
            nights = sleep[(sleep.e > a) & (sleep.s < b)]
            if len(nights):
                stage = ["awake"] * bins
                for _, n in nights.iterrows():
                    for k in range(max(0, bin_of(n.s)), min(bins, bin_of(n.e) + 1)):
                        stage[k] = "core"
                for k in range(bins):
                    _set_dense(rec, "sleep_stage", k, stage[k])

        ev = rec["events"]
        ev["insulin_bolus_u"]["events"] = [{"timestamp": _iso(r.t), "value": float(r["bolus_dose"])} for _, r in in_day(bolus).iterrows()]
        if meals is not None:
            ev["carbs_g"]["events"] = [{"timestamp": _iso(r.t), "value": float(r["carbs_g"])} for _, r in in_day(meals).iterrows()]
        ev["temp_basal"]["events"] = [
            {"timestamp": _iso(t), "rate": rate, "duration": dur} for t, rate, dur in tb_events if a <= t < b
        ]
        (raw_dir / f"{rec['record_id']}.json").write_text(json.dumps(rec))
        written += 1

    summary = {"participant": pid, "days": written, "first": str(days[0].date()) if days else None,
               "last": str(days[-1].date()) if days else None, "basal_kinds": sorted(kinds), "meals_snapped_to_bolus": n_snapped}
    (out / "conversion.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--snap-meals-min", type=float, default=60.0)
    args = ap.parse_args()
    print(convert(args.pid, Path(args.root), Path(args.out), args.snap_meals_min))


if __name__ == "__main__":
    main()
