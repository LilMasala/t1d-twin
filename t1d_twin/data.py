"""InSite raw day records -> one continuous per-person timeline for the twin.

Input is a list of ``insite.raw_day.v1`` records (the documents the app writes
to Firestore ``canonical_raw_days/raw_day_records`` and ``InSiteRawStore``
keeps as ``raw_days/*.json``). Output is a 2-minute grid anchored in UTC, so
DST days (276/300 bins) and gaps between records place correctly.

Event payloads come in two spellings and both are accepted:

- app (Nightscout/Tandem): ``{"timestamp", "value"}`` for bolus U and carbs g;
  temp basal ``{"timestamp", "rate" (U/hr), "duration" (min)}``;
- simulator: ``units`` / ``grams``; temp basal ``rate_u_per_min`` per 5-min
  bin. The simulator stamps insulin one bin late (``offset_bins=1`` in
  ``canonical_raw._bin_delivery_events``), which is undone here.

App-record fidelity this loader corrects for (``CanonicalRawDayRecord.swift``):

- CGM hours with fewer than 12 readings are written as the hourly mean in
  all 12 bins; such flat hours are reduced to one reading at mid-hour;
- heart rate is an hourly mean and exercise minutes an hourly total, each
  repeated in all 12 bins; exercise is converted back to minutes per bin;
- therapy settings (CR, ISF, scheduled basal) are not in the record; pass the
  app's ``therapy_settings/hourly`` documents as ``therapy_settings``.

Context beyond the record: ``sleep_daily`` takes the app's ``sleep/daily``
documents (nightly asleep seconds) when per-bin stages are absent; stress comes
from a ``stress_level`` dense stream (0-100, e.g. Garmin; not part of the
contract) or from app mood events via the app's own acute-stress mapping; sex
comes from the caller. Pump users without logged site changes get inferred ones
(see ``infer_site_changes``), flagged in notes.

Missing data stays missing: a day without bolus records or with unknown basal
is marked ``insulin_observed=False`` and is never fitted as "no insulin".
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

from t1d_twin.ode import DT_MIN

ASLEEP_STAGES = {"core", "deep", "rem", "asleep"}
MEAL_WINDOW_STEPS = 5          # a meal is eaten over 10 minutes
MOOD_HOLD_H = 6.0              # a mood report describes the next few hours
MEAL_LOGGER_MIN_DAY_FRACTION = 0.6  # logs meals on at least this share of days -> a blank day is a missing log
HR_MAX_PROXY = 185.0           # heart-rate-reserve ceiling when age is unknown
EXERCISE_HRR_THRESHOLD = 0.30  # below this reserve fraction, not exercise


@dataclass
class Meal:
    step: int
    grams: float
    logged: bool = True


@dataclass
class PersonTimeline:
    person_id: str
    t0_utc: datetime
    tz: str
    n_steps: int
    cgm: np.ndarray                 # mg/dL, NaN where unobserved
    basal_upm: np.ndarray           # delivered basal U/min (NaN where unknown)
    bolus_upm: np.ndarray           # bolus as U/min over one step
    meals: list[Meal]
    hr: np.ndarray                  # bpm, NaN where unobserved
    exercise_min: np.ndarray        # minutes of exercise in the 5-min bin (0 if none)
    asleep: np.ndarray              # 1 asleep, 0 awake, NaN unknown
    stress: np.ndarray              # 0..1 acute stress, NaN unknown
    sleep_hours: np.ndarray         # per day, hours asleep the night before (NaN unknown)
    local_hour: np.ndarray          # fractional local hour of day
    day_index: np.ndarray           # local calendar day index per step
    day_dates: list[str]
    days_since_period: np.ndarray   # per day, NaN unknown
    resting_hr: np.ndarray          # per day, NaN unknown
    site_change_steps: list[int]
    food_photo_steps: list[int]
    body_mass_kg: float | None
    cr: np.ndarray                  # g/U setting per step (NaN unknown)
    isf: np.ndarray                 # mg/dL/U setting per step (NaN unknown)
    basal_setting_uph: np.ndarray   # scheduled basal U/hr per step (NaN unknown)
    insulin_observed: np.ndarray    # per day bool
    cgm_coverage: np.ndarray        # per day fraction of hours with at least one CGM reading
    notes: list[str] = field(default_factory=list)
    sex: str | None = None                                   # "female" / "male" / None
    site_changes_inferred: bool = False
    meals_observed: np.ndarray | None = None                 # per day: False = this regular logger logged nothing that day

    @property
    def n_days(self) -> int:
        return len(self.day_dates)

    def day_steps(self, day: int) -> tuple[int, int]:
        idx = np.flatnonzero(self.day_index == day)
        return int(idx[0]), int(idx[-1]) + 1


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read raw day records from a directory of JSON files or one JSON file/list."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        records = [json.loads(f.read_text()) for f in files]
    else:
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else [payload]
    return sorted(records, key=lambda r: r["utc_anchor"])


def _finite(value) -> float | None:
    """A usable numeric event value, or None (missing, null or NaN)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _dense(record: dict, stream: str) -> tuple[list, list] | None:
    block = record.get("dense", {}).get(stream)
    if not block:
        return None
    return block.get("values", []), block.get("observed_flag", [])


def _slow_first(record: dict, stream: str) -> float | None:
    block = record.get("slow", {}).get(stream)
    if not block:
        return None
    for v, f in zip(block.get("values", []), block.get("observed_flag", [])):
        if f and v is not None:
            return float(v)
    return None


def _collapse_flat_hours(values: list, flags: list) -> tuple[list, list, int]:
    """Runs of >= 12 identical observed readings are hour means: keep the middle one."""
    values, flags = list(values), list(flags)
    n, i, collapsed = len(values), 0, 0
    while i < n:
        if not (flags[i] and values[i] is not None):
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1] and values[j + 1] == values[i]:
            j += 1
        if j - i + 1 >= 12:
            mid = (i + j) // 2
            for k in range(i, j + 1):
                if k != mid:
                    flags[k] = 0
            collapsed += 1
        i = j + 1
    return values, flags, collapsed


def _hourly_totals_to_per_bin(values: list, flags: list) -> list:
    """Exercise minutes written as an hourly total in 12 bins -> minutes per bin."""
    out = [float(v) if (f and v is not None) else 0.0 for v, f in zip(values, flags)]
    n, i = len(out), 0
    while i < n:
        j = i
        while j + 1 < n and out[j + 1] == out[i]:
            j += 1
        total = out[i]
        if total > 0 and j - i + 1 >= 12:
            for k in range(i, j + 1):
                out[k] = total / 12.0
        i = j + 1
    return out


def load_therapy_settings(path: str | Path) -> list[dict[str, Any]]:
    """Therapy settings hourly docs (a JSON list, or a directory of JSON docs)."""
    path = Path(path)
    if path.is_dir():
        return [json.loads(f.read_text()) for f in sorted(path.glob("*.json"))]
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else [payload]


def build_timeline(
    records: list[dict[str, Any]],
    person_id: str = "person",
    therapy_settings: list[dict[str, Any]] | None = None,
    sleep_daily: list[dict[str, Any]] | None = None,
    sex: str | None = None,
) -> PersonTimeline:
    if not records:
        raise ValueError("no raw day records")
    records = sorted(records, key=lambda r: r["utc_anchor"])
    t0 = _parse_ts(records[0]["utc_anchor"])
    last = records[-1]
    t_end = _parse_ts(last["utc_anchor"]) + timedelta(minutes=5 * int(last["bin_count"]))
    n_steps = int((t_end - t0).total_seconds() // (60 * DT_MIN))
    tz = records[0].get("tz_identifier", "UTC")

    nan = lambda: np.full(n_steps, np.nan)
    cgm, hr, asleep, stress = nan(), nan(), nan(), nan()
    exercise = np.zeros(n_steps)
    cr, isf, basal_set = nan(), nan(), nan()
    basal = nan()
    bolus = np.zeros(n_steps)
    day_index = np.full(n_steps, -1, dtype=int)
    meals: list[Meal] = []
    site_steps: list[int] = []
    photo_steps: list[int] = []
    notes: list[str] = []

    day_dates: list[str] = []
    dsp, rhr, bolus_days, coverage = [], [], [], []
    flat_cgm_hours = 0
    body_masses: list[float] = []

    def step_of(ts: datetime) -> int:
        return int((ts - t0).total_seconds() // (60 * DT_MIN))

    for d, rec in enumerate(records):
        anchor = _parse_ts(rec["utc_anchor"])
        bins = int(rec["bin_count"])
        s0 = step_of(anchor)
        s1 = min(n_steps, step_of(anchor + timedelta(minutes=5 * bins)))
        day_index[s0:s1] = d
        day_dates.append(rec["local_date"])
        # step -> the 5-min bin that contains it
        steps = np.arange(s0, s1)
        bin_of_step = np.minimum(((steps - s0) * DT_MIN // 5).astype(int), bins - 1)

        def fill(stream, target, cast=float):
            got = _dense(rec, stream)
            if got is None:
                return 0
            values, flags = got
            n_obs = 0
            for k, s in enumerate(steps):
                b = bin_of_step[k]
                if b < len(values) and b < len(flags) and flags[b] and values[b] is not None:
                    target[s] = cast(values[b])
                    n_obs += 1
            return n_obs

        # CGM: one observation per 5-min bin, placed at the step nearest bin centre
        got = _dense(rec, "cgm_mgdl")
        n_cgm = 0
        if got is not None:
            values, flags, flat = _collapse_flat_hours(*got)
            flat_cgm_hours += flat
            for b in range(min(bins, len(values))):
                if b < len(flags) and flags[b] and values[b] is not None:
                    s = s0 + int(round((5 * b + 2.5) / DT_MIN))
                    if s < n_steps:
                        cgm[s] = float(values[b])
                        n_cgm += 1
        # hours with any reading: robust to 15-min sensors and the app's hour-mean fallback
        hours = max(1, -(-bins // 12))
        seen = np.zeros(hours, dtype=bool)
        if got is not None:
            for b in range(min(bins, len(flags))):
                if flags[b] and b < len(values) and values[b] is not None:
                    seen[b // 12] = True
        coverage.append(float(seen.mean()))

        fill("heart_rate_bpm", hr)
        got = _dense(rec, "stress_level")  # optional, 0-100 (e.g. Garmin); negative = no reading
        if got is not None:
            values, flags = got
            for k, s in enumerate(steps):
                b = bin_of_step[k]
                if b < len(values) and flags[b] and values[b] is not None and float(values[b]) >= 0:
                    stress[s] = float(values[b]) / 100.0
        got = _dense(rec, "exercise_min")
        if got is not None:
            per_bin = _hourly_totals_to_per_bin(*got)
            for k, s in enumerate(steps):
                if bin_of_step[k] < len(per_bin):
                    exercise[s] = per_bin[bin_of_step[k]]
        fill("carb_ratio_g_per_u", cr)
        fill("isf_mgdl_per_u", isf)
        fill("basal_rate_u_per_hr", basal_set)

        got = _dense(rec, "sleep_stage")
        if got is not None:
            values, flags = got
            for k, s in enumerate(steps):
                b = bin_of_step[k]
                if b < len(values) and b < len(flags) and flags[b] and values[b] is not None:
                    stage = str(values[b])
                    if stage != "none":
                        asleep[s] = 1.0 if stage in ASLEEP_STAGES else 0.0

        dsp.append(_slow_first(rec, "days_since_period"))
        rhr.append(_slow_first(rec, "resting_hr_bpm"))
        bm = _slow_first(rec, "body_mass_kg")
        if bm is not None:
            body_masses.append(bm)

        events = rec.get("events", {})
        n_bolus = 0
        bad_bolus = 0

        for ev in events.get("insulin_bolus_u", {}).get("events", []):
            units = _finite(ev.get("units", ev.get("value")))
            if units is None:
                bad_bolus += 1
                continue
            ts = _parse_ts(ev["timestamp"])
            if "units" in ev:
                ts -= timedelta(minutes=5)
            s = step_of(ts)
            if 0 <= s < n_steps:
                bolus[s] += float(units) / DT_MIN
                n_bolus += 1

        for ev in events.get("temp_basal", {}).get("events", []):
            ts = _parse_ts(ev["timestamp"])
            if "rate_u_per_min" in ev:
                ts -= timedelta(minutes=5)
                s = step_of(ts)
                rate, dur = _finite(ev["rate_u_per_min"]), 5.0
            elif "rate" in ev:
                s = step_of(ts)
                rate = _finite(ev["rate"])
                rate = None if rate is None else rate / 60.0
                dur = _finite(ev.get("duration", 30.0))
            else:
                continue
            if rate is None or dur is None:
                continue
            # End from the end *time*, so back-to-back entries on odd minutes leave no grid holes.
            e = min(n_steps, step_of(ts + timedelta(minutes=dur)))
            if s < n_steps and e > max(s, 0):
                basal[max(s, 0):e] = rate

        for ev in events.get("carbs_g", {}).get("events", []):
            grams = _finite(ev.get("grams", ev.get("value")))
            if grams is None or grams <= 0:
                continue
            s = step_of(_parse_ts(ev["timestamp"]))
            if 0 <= s < n_steps:
                meals.append(Meal(step=s, grams=float(grams), logged=True))

        for ev in events.get("site_change", {}).get("events", []):
            s = step_of(_parse_ts(ev["timestamp"]))
            if 0 <= s < n_steps:
                site_steps.append(s)

        for ev in events.get("mood", {}).get("events", []):
            if "timestamp" not in ev or ev.get("arousal") is None:
                continue
            # the app's acute-stress mapping (FeatureFrameToChameliaAdapter.stressAcute)
            arousal, valence = float(ev["arousal"]), ev.get("valence")
            level = 0.0 if arousal <= 0.6 else (1.0 if valence is not None and float(valence) < 0 else 0.5)
            s = step_of(_parse_ts(ev["timestamp"]))
            e = min(n_steps, s + int(MOOD_HOLD_H * 60 / DT_MIN))
            if 0 <= s < n_steps:
                seg = stress[s:e]
                stress[s:e] = np.where(np.isnan(seg), level, np.maximum(seg, level))

        for ev in events.get("food_photo_topdown", {}).get("events", []):
            if "timestamp" in ev:
                s = step_of(_parse_ts(ev["timestamp"]))
                if 0 <= s < n_steps:
                    photo_steps.append(s)

        # a bolus with no dose makes the day's insulin unknown, not smaller
        bolus_days.append(n_bolus > 0 and bad_bolus == 0)
        if bad_bolus:
            notes.append(f"{rec['local_date']}: {bad_bolus} bolus record(s) without a dose; insulin treated as unknown")

    # Therapy settings documents fill any hour the records did not carry.
    for row in therapy_settings or []:
        start = step_of(_parse_ts(str(row["hourStartUtc"])))
        lo, hi = max(0, start), min(n_steps, start + int(60 / DT_MIN))
        if lo >= hi:
            continue
        for target, key in ((cr, "carbRatio"), (isf, "insulinSensitivity"), (basal_set, "basalRate")):
            if row.get(key) is not None:
                seg = target[lo:hi]
                target[lo:hi] = np.where(np.isnan(seg), float(row[key]), seg)

    # Delivered basal defaults to the scheduled basal wherever no temp basal ran.
    basal = np.where(np.isnan(basal), basal_set / 60.0, basal)
    insulin_seen = []
    for d, date in enumerate(day_dates):
        known = float(np.mean(~np.isnan(basal[day_index == d])))
        insulin_seen.append(bool(bolus_days[d] and known >= 0.9))
        if not bolus_days[d]:
            notes.append(f"{date}: no bolus records (insulin treated as unknown)")
        elif known < 0.9:
            notes.append(f"{date}: basal known for {known:.0%} of the day (therapy settings missing?)")
    sleep_hours = np.full(len(day_dates), np.nan)
    for row in sleep_daily or []:
        date = str(row.get("dateUtc", ""))[:10]
        if date in day_dates:
            sec = sum(float(row.get(k) or 0.0) for k in ("asleepCore", "asleepDeep", "asleepREM", "asleepUnspecified"))
            if sec > 0:
                sleep_hours[day_dates.index(date)] = sec / 3600.0

    if flat_cgm_hours:
        notes.append(f"{flat_cgm_hours} CGM hours were hourly means; kept one reading each")

    # Local clock for dawn and sleep windows.
    zone = ZoneInfo(tz)
    local_hour = np.empty(n_steps)
    for s in range(0, n_steps, 30):  # hourly resolution for the offset is plenty
        ts = (t0 + timedelta(minutes=DT_MIN * s)).astimezone(zone)
        base_hour = ts.hour + ts.minute / 60.0
        k = np.arange(min(30, n_steps - s))
        local_hour[s:s + len(k)] = (base_hour + k * DT_MIN / 60.0) % 24.0

    # Merge meals logged within the eating window of each other.
    meals.sort(key=lambda m: m.step)
    merged: list[Meal] = []
    for m in meals:
        if merged and m.step - merged[-1].step < MEAL_WINDOW_STEPS:
            merged[-1].grams += m.grams
        else:
            merged.append(Meal(m.step, m.grams, True))

    tl = PersonTimeline(
        person_id=person_id,
        t0_utc=t0,
        tz=tz,
        n_steps=n_steps,
        cgm=cgm,
        basal_upm=basal,
        bolus_upm=bolus,
        meals=merged,
        hr=hr,
        exercise_min=exercise,
        asleep=asleep,
        local_hour=local_hour,
        day_index=day_index,
        day_dates=day_dates,
        days_since_period=np.array([np.nan if v is None else v for v in dsp], dtype=float),
        resting_hr=np.array([np.nan if v is None else v for v in rhr], dtype=float),
        site_change_steps=sorted(site_steps),
        food_photo_steps=sorted(photo_steps),
        stress=stress,
        sleep_hours=sleep_hours,
        sex=sex,
        body_mass_kg=float(np.median(body_masses)) if body_masses else None,
        cr=cr,
        isf=isf,
        basal_setting_uph=basal_set,
        insulin_observed=np.array(insulin_seen, dtype=bool),
        cgm_coverage=np.array(coverage, dtype=float),
        notes=notes,
    )
    # A regular meal logger's day with nothing logged means the log is missing, not that they fasted.
    per_day = np.zeros(tl.n_days, dtype=int)
    for m in tl.meals:
        if m.logged and tl.day_index[m.step] >= 0:
            per_day[tl.day_index[m.step]] += 1
    regular_logger = tl.n_days > 0 and float(np.mean(per_day > 0)) >= MEAL_LOGGER_MIN_DAY_FRACTION
    tl.meals_observed = (per_day > 0) | (not regular_logger)
    missing = [tl.day_dates[d] for d in np.flatnonzero(~tl.meals_observed)]
    if missing:
        tl.notes.append(f"meal log missing (regular logger, nothing logged): {', '.join(missing)}")
    if not tl.site_change_steps:
        inferred = infer_site_changes(tl)
        if inferred:
            tl.site_change_steps, tl.site_changes_inferred = inferred, True
            tl.notes.append(f"{len(inferred)} infusion-set changes inferred from pump suspensions (none logged)")
    return tl


def infer_site_changes(tl: PersonTimeline, *, min_suspend_min: float = 15.0, min_glucose: float = 120.0, min_gap_h: float = 36.0) -> list[int]:
    """Likely infusion-set changes for pump users who did not log them.

    A set change shows up as basal delivery stopping for a while with glucose
    not low (a low-glucose suspend happens near hypoglycaemia, a set change
    does not); changes are at least ``min_gap_h`` apart. Inferred, not observed.
    """
    zero = np.nan_to_num(tl.basal_upm, nan=1.0) <= 1e-6
    out: list[int] = []
    s, n = 0, tl.n_steps
    while s < n:
        if not zero[s]:
            s += 1
            continue
        e = s
        while e < n and zero[e]:
            e += 1
        if (e - s) * DT_MIN >= min_suspend_min:
            window = tl.cgm[max(0, s - int(30 / DT_MIN)):s + 1]
            g = np.nanmean(window) if np.any(~np.isnan(window)) else np.nan
            if not np.isnan(g) and g >= min_glucose and (not out or (s - out[-1]) * DT_MIN >= min_gap_h * 60):
                out.append(s)
        s = e
    return out


def shift_events(tl: PersonTimeline, minutes: float) -> PersonTimeline:
    """Copy of ``tl`` with insulin, meals, site changes and food photos moved by ``minutes``.

    Positive moves events later. Used to correct a clock offset between the
    CGM and the pump / meal log. CGM, heart rate, sleep and settings stay put.
    """
    k = int(round(minutes / DT_MIN))
    out = copy.copy(tl)
    if k == 0:
        return out

    def roll(a: np.ndarray, fill: float) -> np.ndarray:
        r = np.full_like(a, fill)
        if k > 0:
            r[k:] = a[:-k]
        else:
            r[:k] = a[-k:]
        return r

    out.bolus_upm = roll(tl.bolus_upm, 0.0)
    out.basal_upm = roll(tl.basal_upm, np.nan)
    keep = lambda s: 0 <= s + k < tl.n_steps
    out.meals = [Meal(m.step + k, m.grams, m.logged) for m in tl.meals if keep(m.step)]
    out.site_change_steps = [s + k for s in tl.site_change_steps if keep(s)]
    out.food_photo_steps = [s + k for s in tl.food_photo_steps if keep(s)]
    out.notes = tl.notes + [f"insulin/meal events shifted {minutes:+.0f} min to align with CGM"]
    return out


def detect_unlogged_meals(
    tl: PersonTimeline,
    *,
    min_rise: float = 30.0,
    rise_window_min: float = 45.0,
    logged_exclusion_min: float = 90.0,
    lead_min: float = 15.0,
    merge_min: float = 90.0,
    min_anchor_bolus_u: float = 1.0,
    use_cgm_rises: bool = True,
) -> list[Meal]:
    """Candidate times for meals with no carb entry. Sizes are fitted, so a
    candidate that turns out not to be a meal is sized near zero.

    Two kinds, announcement anchors first:

    - announcements without carbs: a food photo, or a bolus of at least
      ``min_anchor_bolus_u`` with no logged meal within an hour either side
      (people bolus for meals they do not count; a pure correction bolus is
      simply sized ~0);
    - unexplained rises: a CGM rise of ``min_rise`` mg/dL within
      ``rise_window_min`` with no logged meal in the ``logged_exclusion_min``
      before it and no announcement nearby, placed ``lead_min`` before the rise.

    ``use_cgm_rises=False`` keeps only announcement anchors, which do not read
    CGM (for held-out evaluation, where reading CGM would leak the answer).
    """
    logged_steps = np.array([m.step for m in tl.meals if m.logged], dtype=int)
    hour = 60.0 / DT_MIN

    def near(steps: np.ndarray, s: int, before: float, after: float) -> bool:
        return bool(steps.size) and bool(np.any((steps >= s - before) & (steps <= s + after)))

    anchors: list[int] = list(tl.food_photo_steps)
    bolus_steps = np.flatnonzero(tl.bolus_upm * DT_MIN >= min_anchor_bolus_u)
    anchors += [int(s) for s in bolus_steps]
    out: list[Meal] = []
    for s in sorted(anchors):
        if near(logged_steps, s, hour, hour):
            continue
        if out and s - out[-1].step <= hour:
            continue
        out.append(Meal(step=int(s), grams=0.0, logged=False))

    obs = np.flatnonzero(~np.isnan(tl.cgm))
    if use_cgm_rises and obs.size >= 10:
        t_obs = obs * DT_MIN
        g = tl.cgm[obs]
        # light smoothing: median of 3 neighbouring readings
        gs = np.array([np.median(g[max(0, i - 1):i + 2]) for i in range(g.size)])
        anchor_steps = np.array([m.step for m in out], dtype=int)
        rises: list[Meal] = []
        i = 0
        while i < gs.size:
            j = np.searchsorted(t_obs, t_obs[i] + rise_window_min, side="right") - 1
            if j > i and gs[j] - gs[i] >= min_rise and gs[i] <= gs[i:j + 1].min() + 5.0:
                onset = int(max(0, (t_obs[i] - lead_min) // DT_MIN))
                # an announcement explains a rise only if it came before the rise got going;
                # a correction bolus 30+ min into the rise does not
                explained = near(logged_steps, onset, logged_exclusion_min / DT_MIN, rise_window_min / DT_MIN) or near(
                    anchor_steps, onset, logged_exclusion_min / DT_MIN, 30.0 / DT_MIN
                )
                if not explained and (not rises or onset - rises[-1].step > merge_min / DT_MIN):
                    rises.append(Meal(step=onset, grams=0.0, logged=False))
                i = j
            i += 1
        out += rises
    return sorted(out, key=lambda m: m.step)
