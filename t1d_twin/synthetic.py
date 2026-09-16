"""Synthetic people with known twin parameters, written as InSite raw day records.

Used to check the fitter can recover what generated the data before trusting
it on real people. The data goes out through the same record format the app
writes (app-style event payloads) and back in through ``data.build_timeline``,
so the loader is exercised end to end.

Realism that makes recovery hard on purpose: logged carbs are miscounted,
some meals are never logged, sensitivity drifts day to day, CGM noise is
autocorrelated, and insulin comes from a closed loop (dosing reacts to
glucose, the confounding real AID data has).

With ``app_fidelity`` (default) the records also carry the app's own
degradations: hourly-mean heart rate and hourly exercise totals repeated in
every bin, hour-mean CGM whenever an hour misses a reading, no sleep stages,
and therapy settings delivered separately as hourly documents
(``truth.therapy_settings``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.data import Meal, build_timeline
from t1d_twin.experiment import BasalBolusController
from t1d_twin.model import DTYPE, rollout
from t1d_twin.params import TwinPriors, twin_priors

CONTRACT_DENSE = ("cgm_mgdl", "heart_rate_bpm", "energy_basal_kcal", "energy_active_kcal", "move_min", "exercise_min", "sleep_stage")


@dataclass
class SyntheticTruth:
    person_id: str
    base: str
    globals_u: list[float]
    param_names: list[str]
    day_log_si: list[float]
    meal_true_g: dict[int, float] = field(default_factory=dict)      # step -> grams eaten
    meal_logged_g: dict[int, float] = field(default_factory=dict)    # logged step -> grams logged (subset)
    meal_logged_at: dict[int, int] = field(default_factory=dict)     # logged step -> true eating step
    unlogged_steps: list[int] = field(default_factory=list)
    therapy_settings: list[dict] = field(default_factory=list)


def _empty_record(date: datetime, tz: str, bins: int, person_id: str) -> dict:
    anchor = date.astimezone(timezone.utc)
    rec = {
        "schema_version": "insite.raw_day.v1", "encoder_version": None, "source": "twin_synthetic",
        "record_id": f"{person_id}-{date.date().isoformat()}", "local_date": date.date().isoformat(),
        "tz_identifier": tz, "utc_anchor": anchor.isoformat().replace("+00:00", "Z"),
        "bin_minutes": 5, "bin_count": bins, "dense": {}, "slow": {}, "derived_prior": {},
        "events": {k: {"events": []} for k in ("site_change", "mood", "insulin_bolus_u", "carbs_g", "temp_basal", "food_photo_topdown")},
        "metadata": {"synthetic": True},
    }
    units = {"cgm_mgdl": "mg/dL", "heart_rate_bpm": "bpm", "energy_basal_kcal": "kcal", "energy_active_kcal": "kcal",
             "move_min": "min", "exercise_min": "min", "sleep_stage": "category"}
    for s in CONTRACT_DENSE:
        rec["dense"][s] = {"unit": units[s], "values": [None] * bins, "observed_flag": [0] * bins}
    for s, u in (("carb_ratio_g_per_u", "g/U"), ("isf_mgdl_per_u", "mg/dL/U"), ("basal_rate_u_per_hr", "U/hr")):
        rec["dense"][s] = {"unit": u, "values": [None] * bins, "observed_flag": [0] * bins}
    for s, u in (("resting_hr_bpm", "bpm"), ("body_mass_kg", "kg"), ("days_since_period", "day")):
        rec["slow"][s] = {"unit": u, "values": [None] * bins, "observed_flag": [0] * bins}
    for s, u in (("iob_u", "U"), ("cob_g", "g")):
        rec["derived_prior"][s] = {"unit": u, "values": [None] * bins, "observed_flag": [0] * bins}
    return rec


def _set_dense(rec, stream, b, value):
    rec["dense"][stream]["values"][b] = value
    rec["dense"][stream]["observed_flag"][b] = 1


def _set_slow(rec, stream, value):
    n = rec["bin_count"]
    rec["slow"][stream]["values"] = [value] * n
    rec["slow"][stream]["observed_flag"] = [1] * n


def generate_person(
    person_id: str,
    *,
    n_days: int = 21,
    seed: int = 0,
    base: str | None = None,
    priors: TwinPriors | None = None,
    is_female: bool = True,
    aid: bool = True,
    unlogged_fraction: float = 0.10,
    carb_count_sd: float = 0.25,
    meal_log_time_sd_min: float = 10.0,
    start_date: str = "2026-03-02",
    tz: str = "America/Detroit",
    app_fidelity: bool = True,
    cgm_dropout: float = 0.01,
    response_corrections: bool = False,
    truth_sd_scale: float = 1.0,
) -> tuple[list[dict], SyntheticTruth]:
    rng = np.random.default_rng(seed)
    priors = priors or twin_priors(has_cycle=is_female, cycle_observed=True, sex="female" if is_female else "male")
    base = base or str(rng.choice(ode.ADULTS))
    bp = ode.base_patient(base)
    zone = ZoneInfo(tz)
    day0 = datetime.fromisoformat(start_date).replace(tzinfo=zone)

    # --- truth parameters: a prior draw, kept away from the tails ---
    # truth_sd_scale=0 draws a population-typical person (isolates a mechanism in tests)
    u = priors.means().numpy() + truth_sd_scale * priors.sds().numpy() * np.clip(rng.normal(size=len(priors.specs)), -1.5, 1.5)
    names = priors.names()
    u[names.index("log_cgm_sd")] = np.log(8.0)
    if not response_corrections:  # a pure UVA/Padova body unless asked otherwise
        for i, n in enumerate(names):
            if n.startswith(("carb_resp_k", "ins_resp_k")):
                u[i] = 0.0
    day_sd = float(np.exp(u[names.index("log_day_si_sd")]))
    day_log_si = rng.normal(0.0, day_sd, size=n_days)

    records = []
    period_offset = int(rng.integers(0, 28))
    rhr = float(rng.normal(62, 5))
    exercise_days = set(rng.choice(n_days, size=max(1, int(n_days * 3 / 7)), replace=False).tolist())
    site_every = 3
    true_meals: list[tuple[datetime, float, bool]] = []
    settings_rows: list[dict] = []
    # Settings roughly matched to this body so the loop is not wildly off.
    si = float(np.exp(u[names.index("log_si")]))
    basal_uph = bp.basal_u_per_hr * float(np.exp(u[names.index("log_egp")])) / max(si, 0.3) ** 0.5
    cr = float(np.clip(rng.normal(10.0, 2.0), 5.0, 18.0))
    isf = float(np.clip(rng.normal(45.0, 10.0), 20.0, 90.0))

    for d in range(n_days):
        date = day0 + timedelta(days=d)
        nxt = day0 + timedelta(days=d + 1)
        bins = int((nxt - date).total_seconds() // 300)
        rec = _empty_record(date, tz, bins, person_id)
        anchor = date.astimezone(timezone.utc)
        _set_slow(rec, "resting_hr_bpm", rhr)
        _set_slow(rec, "body_mass_kg", float(bp.params["BW"]))
        if is_female:
            _set_slow(rec, "days_since_period", float((period_offset + d) % 28))

        slept_h = float(np.clip(rng.normal(7.2, 1.0), 4.0, 9.5))
        wake_h = float(np.clip(rng.normal(7.0, 0.5), 5.5, 9.0))
        sleep_start = wake_h - slept_h  # hours relative to local midnight (negative = previous evening)
        ex_start = float(rng.uniform(16.0, 19.0)) if d in exercise_days else None
        ex_len = float(rng.uniform(30, 60)) / 60.0
        hr_bins, ex_bins = np.zeros(bins), np.zeros(bins)
        for b in range(bins):
            h = b * 5 / 60.0
            asleep = (h < wake_h and h >= sleep_start) or (h >= 24 + sleep_start)
            if not app_fidelity:
                _set_dense(rec, "sleep_stage", b, "core" if asleep else "awake")
            hr_bins[b] = rhr + rng.normal(0, 3) + (0 if asleep else 12)
            if ex_start is not None and ex_start <= h < ex_start + ex_len:
                hr_bins[b], ex_bins[b] = 145 + rng.normal(0, 6), 5.0
            for s in ("energy_basal_kcal", "energy_active_kcal", "move_min"):
                _set_dense(rec, s, b, 0.0)
            if not app_fidelity:
                _set_dense(rec, "carb_ratio_g_per_u", b, cr)
                _set_dense(rec, "isf_mgdl_per_u", b, isf)
                _set_dense(rec, "basal_rate_u_per_hr", b, basal_uph)
        for b0 in range(0, bins, 12):
            blk = slice(b0, min(bins, b0 + 12))
            hr_val = hr_bins[blk].mean() if app_fidelity else None
            ex_val = ex_bins[blk].sum() if app_fidelity else None
            for b in range(blk.start, blk.stop):
                _set_dense(rec, "heart_rate_bpm", b, float(hr_bins[b] if hr_val is None else hr_val))
                _set_dense(rec, "exercise_min", b, float(ex_bins[b] if ex_val is None else ex_val))
        if app_fidelity:
            for hour in range(bins // 12):
                ts = anchor + timedelta(hours=hour)
                settings_rows.append({"hourStartUtc": ts.isoformat().replace("+00:00", "Z"), "carbRatio": cr,
                                      "basalRate": basal_uph, "insulinSensitivity": isf})
        if d % site_every == 0:
            ts = (date + timedelta(hours=9)).astimezone(timezone.utc)
            rec["events"]["site_change"]["events"].append({"timestamp": ts.isoformat().replace("+00:00", "Z"), "location": "abdomen"})

        for mh, size in ((wake_h + 0.5, 45), (12.5, 60), (19.0, 70), (15.5, 20)):
            if size == 20 and rng.random() < 0.5:
                continue
            t = date + timedelta(hours=float(mh + rng.normal(0, 0.4)))
            grams = float(np.clip(rng.normal(size, size * 0.3), 10, 140))
            logged = rng.random() >= unlogged_fraction
            true_meals.append((t, grams, logged))
        records.append(rec)

    # Scenario timeline (no CGM / insulin yet) to drive the twin.
    tl = build_timeline(records, person_id, settings_rows)
    truth = SyntheticTruth(person_id, base, [float(v) for v in u], names, [float(v) for v in day_log_si],
                           therapy_settings=settings_rows)
    step_of = lambda t: int((t.astimezone(timezone.utc) - tl.t0_utc).total_seconds() // (60 * ode.DT_MIN))
    meal_grams = []
    tl.meals = []
    for t, grams, logged in sorted(true_meals):
        s = step_of(t)
        truth.meal_true_g[s] = grams
        meal_grams.append((s, torch.tensor([grams], dtype=DTYPE)))
        if logged:
            # the announcement (carbs + bolus) lands near, not exactly at, the eating time
            s_log = max(0, s + int(round(rng.normal(0.0, meal_log_time_sd_min) / ode.DT_MIN)))
            shown = float(np.round(grams * np.exp(-rng.normal(0, carb_count_sd)) / 5) * 5)
            truth.meal_logged_g[s_log] = max(shown, 5.0)
            truth.meal_logged_at[s_log] = s
            tl.meals.append(Meal(s_log, truth.meal_logged_g[s_log], True))
        else:
            truth.unlogged_steps.append(s)

    ctrl = BasalBolusController(
        tl, cr_mult=torch.ones(1, dtype=DTYPE), isf_mult=torch.ones(1, dtype=DTYPE),
        basal_mult=torch.ones(1, dtype=DTYPE), aid=aid, record=True,
    )
    with torch.no_grad():
        glucose = rollout(
            tl, base, priors, torch.tensor(u, dtype=DTYPE)[None], 0, tl.n_steps,
            meal_grams=meal_grams, day_log_si=torch.tensor(day_log_si, dtype=DTYPE)[None], controller=ctrl,
        )[0].numpy()

    # Write CGM (AR(1) sensor noise), delivered insulin and logged carbs back into the records.
    noise = 0.0
    for d, rec in enumerate(records):
        anchor = datetime.fromisoformat(rec["utc_anchor"].replace("Z", "+00:00"))
        s0 = step_of(anchor)
        readings = np.full(rec["bin_count"], np.nan)
        for b in range(rec["bin_count"]):
            s = s0 + int(round((5 * b + 2.5) / ode.DT_MIN))
            noise = 0.8 * noise + rng.normal(0, 8.0 * np.sqrt(1 - 0.8 ** 2))
            if s < tl.n_steps and rng.random() >= cgm_dropout:
                readings[b] = float(np.clip(glucose[s] + noise, 40.0, 400.0))
        for b0 in range(0, rec["bin_count"], 12):
            blk = readings[b0:b0 + 12]
            if app_fidelity and np.any(np.isnan(blk)) and np.any(~np.isnan(blk)):
                blk = np.full(blk.size, float(np.nanmean(blk)))  # the app's hour-mean fallback
            for k, v in enumerate(blk):
                if not np.isnan(v):
                    _set_dense(rec, "cgm_mgdl", b0 + k, float(v))

    iso = lambda t: t.isoformat().replace("+00:00", "Z")
    prev_rate = None
    for step, basal_upm, bolus_u in ctrl.history:
        ts = tl.t0_utc + timedelta(minutes=ode.DT_MIN * step)
        d = int(tl.day_index[step])
        ev = records[d]["events"]
        rate_uph = round(float(basal_upm[0]) * 60.0, 3)
        if rate_uph != prev_rate:
            ev["temp_basal"]["events"].append({"timestamp": iso(ts), "rate": rate_uph, "duration": 30.0})
            prev_rate = rate_uph
        elif ev["temp_basal"]["events"]:
            ev["temp_basal"]["events"][-1]["duration"] += ode.DT_MIN
        if float(bolus_u[0]) > 0:
            ev["insulin_bolus_u"]["events"].append({"timestamp": iso(ts), "value": round(float(bolus_u[0]), 3)})
    # Temp basal durations must stop at the next rate change.
    all_tb = [(d, e) for d, r in enumerate(records) for e in r["events"]["temp_basal"]["events"]]
    for (d, e), (_, nxt) in zip(all_tb, all_tb[1:]):
        gap = (datetime.fromisoformat(nxt["timestamp"].replace("Z", "+00:00")) - datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))).total_seconds() / 60
        e["duration"] = gap
    for s, grams in truth.meal_logged_g.items():
        ts = tl.t0_utc + timedelta(minutes=ode.DT_MIN * s)
        records[int(tl.day_index[s])]["events"]["carbs_g"]["events"].append({"timestamp": iso(ts), "value": grams})
    for rec in records:
        rec["events"]["carbs_g"]["events"].sort(key=lambda e: e["timestamp"])
    return records, truth


def truth_as_fit(truth: SyntheticTruth, tl, priors: TwinPriors | None = None):
    """The generating parameters as a (near point-mass) TwinFit, for paired checks."""
    from t1d_twin.fit import TwinFit

    priors = priors or twin_priors()
    tiny = 1e-4
    G = len(truth.globals_u)
    logged = {m.step: m.grams for m in tl.meals if m.logged}
    steps = sorted(s for s in logged if s in truth.meal_logged_at)
    eaten = lambda s: truth.meal_true_g[truth.meal_logged_at[s]]
    return TwinFit(
        person_id=truth.person_id,
        base=truth.base,
        param_names=truth.param_names,
        prior_mean=[float(v) for v in priors.means()],
        prior_sd=[float(v) for v in priors.sds()],
        loc=list(truth.globals_u),
        sd=[tiny] * G,
        fitted_days=list(tl.day_dates),
        day_drift_loc=list(truth.day_log_si),
        day_drift_sd=[tiny] * len(truth.day_log_si),
        day_egp_drift_loc=[0.0] * len(truth.day_log_si),
        day_egp_drift_sd=[tiny] * len(truth.day_log_si),
        logged_meal_steps=steps,
        meal_scale_loc=[float(np.log(eaten(s) / logged[s])) for s in steps],
        meal_scale_sd=[tiny] * len(steps),
        unlogged_meal_steps=list(truth.unlogged_steps),
        unlogged_log_g_loc=[float(np.log(truth.meal_true_g[s])) for s in truth.unlogged_steps],
        unlogged_log_g_sd=[tiny] * len(truth.unlogged_steps),
        meal_shift_min=[float((truth.meal_logged_at[s] - s) * ode.DT_MIN) for s in steps],
        unlogged_shift_min=[0.0] * len(truth.unlogged_steps),
        scale_tril=(np.eye(G) * tiny).tolist(),
    )
