from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("TWIN_COMPILE", "0")  # skip the ~30 s compile in unit tests

import numpy as np
import pytest
import torch

from t1d_twin import ode
from t1d_twin.context import build_features
from t1d_twin.data import build_timeline, detect_unlogged_meals
from t1d_twin.experiment import Arm, BasalBolusController, run_settings_experiment
from t1d_twin.fit import FitConfig, TwinFit, fit_twin
from t1d_twin.model import DTYPE, TwinProblem
from t1d_twin.params import twin_priors
from t1d_twin.population import build_population
from t1d_twin.synthetic import _empty_record, _set_dense, _set_slow, generate_person, truth_as_fit


@pytest.fixture(scope="module")
def synthetic():
    records, truth = generate_person("t0", n_days=4, seed=3)
    return records, truth, build_timeline(records, "t0", truth.therapy_settings)


def test_torch_ode_matches_simglucose_reference():
    """Our 2-min RK4 torch core tracks simglucose's own scipy integration of the same patient."""
    from t1d_twin.vpatients import t1d_patient_class

    from simglucose.patient.t1dpatient import Action

    T1DPatient = t1d_patient_class()
    names = ["adult#001", "adult#006"]
    bases = [ode.base_patient(n) for n in names]
    basal = np.array([b.basal_u_per_hr for b in bases]) / 60.0  # U/min
    T = 360  # minutes
    ref = []
    for i, name in enumerate(names):
        patient = T1DPatient.withName(name)
        trace = []
        for t in range(T):
            cho = 6.0 if 60 <= t < 70 else 0.0            # 60 g over 10 minutes
            insulin = basal[i] + (3.0 if 60 <= t < 62 else 0.0)  # 6 U bolus over 2 minutes
            patient.step(Action(CHO=cho, insulin=insulin))
            trace.append(patient.observation.Gsub)
        ref.append(trace)
    ref = np.array(ref)[:, 1::2]  # simglucose steps every minute; we compare every 2

    p, Vg, x0 = ode.stack_params(bases)
    S = T // 2
    cho = torch.zeros(2, S, dtype=torch.float64)
    starts = torch.zeros(2, S, dtype=torch.bool)
    ins = torch.tensor(basal)[:, None].repeat(1, S)
    cho[:, 30:35], starts[:, 30], ins[:, 30] = 6.0, True, ins[:, 30] + 3.0
    one = torch.ones(2, S, dtype=torch.float64)
    sim = ode.simulate(x0, p, Vg, cho, starts, ins, one, one, one).numpy()
    # the gap is the fixed 2-minute RK4 step against simglucose's adaptive 1-minute
    # solver, and it peaks during the fastest part of the meal rise
    assert np.abs(sim - ref).max() < 2.0


def _record(date: str, tz: str = "America/Detroit") -> dict:
    zone = ZoneInfo(tz)
    start = datetime.fromisoformat(date).replace(tzinfo=zone)
    bins = int(((start + timedelta(days=1)).astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() // 300)
    rec = _empty_record(start, tz, bins, "r")
    for b in range(bins):
        _set_dense(rec, "cgm_mgdl", b, 120.0 + (b % 7))  # not flat, so no hour-mean collapse
        _set_dense(rec, "basal_rate_u_per_hr", b, 0.9)
    noon = start.astimezone(timezone.utc) + timedelta(hours=12, minutes=1)
    rec["events"]["insulin_bolus_u"]["events"] = [{"timestamp": noon.isoformat().replace("+00:00", "Z"), "value": 1.0}]
    return rec



def test_loader_reads_app_and_simulator_event_spellings():
    rec = _record("2026-01-10")
    anchor = datetime.fromisoformat(rec["utc_anchor"].replace("Z", "+00:00"))
    iso = lambda m: (anchor + timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
    rec["events"]["insulin_bolus_u"]["events"] = [{"timestamp": iso(600), "value": 4.0}, {"timestamp": iso(900), "value": 0.5}]
    rec["events"]["carbs_g"]["events"] = [{"timestamp": iso(600), "value": 50.0}]
    rec["events"]["temp_basal"]["events"] = [{"timestamp": iso(120), "rate": 1.8, "duration": 30}]
    sim = _record("2026-01-11")
    sim_anchor = datetime.fromisoformat(sim["utc_anchor"].replace("Z", "+00:00"))
    # simulator stamps insulin one bin after delivery
    sim["events"]["insulin_bolus_u"]["events"] = [{"timestamp": (sim_anchor + timedelta(minutes=305)).isoformat().replace("+00:00", "Z"), "units": 2.0}]
    tl0 = build_timeline([rec], "p")
    assert tl0.insulin_observed.tolist() == [True]

    tl = build_timeline([rec, sim], "p")
    assert tl.bolus_upm[300] * ode.DT_MIN == pytest.approx(4.0)
    assert [(m.step, m.grams) for m in tl.meals] == [(300, 50.0)]
    assert tl.basal_upm[60:75] == pytest.approx(1.8 / 60.0)
    assert tl.basal_upm[75] == pytest.approx(0.9 / 60.0)
    # back-to-back 5-min entries starting on odd minutes must tile the grid
    rec["events"]["temp_basal"]["events"] = [{"timestamp": iso(301 + 5 * k), "rate": 2.4, "duration": 5} for k in range(12)]
    rec["dense"]["basal_rate_u_per_hr"]["observed_flag"] = [0] * rec["bin_count"]
    tl_odd = build_timeline([rec], "p")
    assert not np.any(np.isnan(tl_odd.basal_upm[151:180]))
    day2 = tl.day_steps(1)[0]
    assert tl.bolus_upm[day2 + 150] * ode.DT_MIN == pytest.approx(2.0)


def test_app_record_degradations_are_undone():
    rec = _record("2026-01-10")
    for b in range(24, 36):
        _set_dense(rec, "cgm_mgdl", b, 150.0)       # hour-mean fallback
        _set_dense(rec, "exercise_min", b, 30.0)    # hourly total repeated
    settings = [{"hourStartUtc": rec["utc_anchor"], "carbRatio": 11.0, "basalRate": 1.1, "insulinSensitivity": 50.0}]
    for b in range(12):
        rec["dense"]["basal_rate_u_per_hr"]["observed_flag"][b] = 0
    tl = build_timeline([rec], "app", settings)
    hour_steps = np.arange(int(120 / ode.DT_MIN), int(180 / ode.DT_MIN))
    assert int(np.sum(~np.isnan(tl.cgm[hour_steps]))) == 1
    assert tl.exercise_min[hour_steps].sum() * ode.DT_MIN / 5.0 == pytest.approx(30.0, rel=0.05)
    assert tl.cr[0] == 11.0 and tl.isf[0] == 50.0 and tl.basal_upm[0] == pytest.approx(1.1 / 60.0)
    assert np.isnan(tl.cr[int(60 / ode.DT_MIN)])


def test_dst_day_lengths_place_on_utc_grid():
    days = [_record(d) for d in ("2026-03-07", "2026-03-08", "2026-03-09")]
    assert [r["bin_count"] for r in days] == [288, 276, 288]
    tl = build_timeline(days, "dst")
    lengths = [tl.day_steps(d)[1] - tl.day_steps(d)[0] for d in range(3)]
    assert lengths == [720, 690, 720]
    s0, _ = tl.day_steps(2)
    assert tl.local_hour[s0] == pytest.approx(0.0)


def test_day_without_insulin_is_skipped_not_zero_filled():
    days = [_record(d) for d in ("2026-01-10", "2026-01-11", "2026-01-12")]
    days[2]["events"]["insulin_bolus_u"]["events"] = []
    tl = build_timeline(days, "gap")
    assert list(tl.insulin_observed) == [True, True, False]
    prob = TwinProblem(tl, "adult#001", [1, 2], twin_priors(has_cycle=False))
    assert prob.days == [1]
    assert "insulin" in prob.skipped[2]


def test_context_unknown_contributes_nothing_and_known_context_is_placed():
    days = [_record(d) for d in ("2026-01-10", "2026-01-11")]
    tl = build_timeline(days, "ctx")
    f = build_features(tl)
    assert not f.known["cycle"] and not f.known["sleep"] and not f.known["site_changes"]
    assert f.luteal.sum() == 0 and f.sleep_deficit_h.sum() == 0 and f.site_age_excess_d.sum() == 0

    for d, r in enumerate(days):
        _set_slow(r, "days_since_period", 20.0 + d)  # luteal
        for b in range(r["bin_count"]):
            _set_dense(r, "sleep_stage", b, "core" if b < 12 * 6 else "awake")  # 6 h asleep after midnight
    days[0]["events"]["site_change"]["events"] = [{"timestamp": days[0]["utc_anchor"], "location": "arm"}]
    f = build_features(build_timeline(days, "ctx"))
    assert f.luteal.min() == 1.0
    s0, s1 = build_timeline(days, "ctx").day_steps(1)
    assert f.sleep_deficit_h[s0] == pytest.approx(2.5, abs=0.1)
    assert f.site_age_excess_d[s1 - 1] == pytest.approx(1.0, abs=0.01)


def test_synthetic_records_satisfy_insite_contract(synthetic):
    """Only runs where InSite's own record validator is importable; skipped in the standalone repo."""
    chamelia = Path(__file__).resolve().parents[2] / "ChameliaV2"
    if str(chamelia) not in sys.path:
        sys.path.insert(0, str(chamelia))
    validate_raw_day_record = pytest.importorskip("src.insite_contract").validate_raw_day_record

    records, _, _ = synthetic
    for rec in records:
        assert validate_raw_day_record(rec) == []


def test_unlogged_meal_detector_skips_logged_meals(synthetic):
    _, truth, tl = synthetic
    cands = detect_unlogged_meals(tl)
    logged = [m.step for m in tl.meals]
    hour = int(60 / ode.DT_MIN)
    for c in cands:
        assert not any(abs(c.step - s) < hour for s in logged)


def test_fit_runs_and_serialises(synthetic, tmp_path):
    _, truth, tl = synthetic
    fit = fit_twin(tl, config=FitConfig(map_iters=3, iters=3, holdout_days=1, base=truth.base), verbose=False)
    path = tmp_path / "twin.json"
    fit.save(path)
    back = TwinFit.load(path)
    assert back.param_names == fit.param_names
    assert set(fit.summary["params"]) == set(fit.param_names)
    assert {"replay_rmse_mgdl", "uncalibrated_replay_rmse_mgdl", "interval90_coverage"} <= set(fit.diagnostics["holdout"])
    assert back.sample_globals(5).shape == (5, len(fit.param_names))
    json.dumps(fit.diagnostics)


def test_more_insulin_lowers_glucose_in_paired_experiment(synthetic):
    _, truth, tl = synthetic
    tf = truth_as_fit(truth, tl)
    out = run_settings_experiment(tf, tl, [Arm("current"), Arm("cr_x0.7", cr_mult=0.7), Arm("cr_x1.4", cr_mult=1.4)], samples=2, days=(1, 3))
    stronger = out["arms"]["cr_x0.7"]["paired_delta_vs_first_arm"]["mean_mgdl"]["median"]
    weaker = out["arms"]["cr_x1.4"]["paired_delta_vs_first_arm"]["mean_mgdl"]["median"]
    assert stronger < 0 < weaker


def test_controller_refuses_without_settings():
    days = [_record(d) for d in ("2026-01-10",)]
    for b in range(days[0]["bin_count"]):
        days[0]["dense"]["basal_rate_u_per_hr"]["observed_flag"][b] = 0
    tl = build_timeline(days, "nosettings")
    one = torch.ones(1, dtype=DTYPE)
    with pytest.raises(ValueError, match="missing"):
        BasalBolusController(tl, cr_mult=one, isf_mult=one, basal_mult=one)


def test_population_shrinks_to_prior_with_one_person(synthetic):
    _, truth, tl = synthetic
    pop = build_population([truth_as_fit(truth, tl)])
    prior_mean = np.array(twin_priors().means())
    assert np.allclose(pop.mean, (np.array(truth.globals_u) + 5 * prior_mean) / 6, atol=1e-6)
    draws = pop.sample(200, seed=1)
    assert draws.shape == (200, len(truth.globals_u))
    assert np.all(np.linalg.eigvalsh(np.array(pop.cov)) > 0)


def test_shift_events_moves_insulin_and_meals_not_cgm(synthetic):
    from t1d_twin.data import shift_events

    _, _, tl = synthetic
    moved = shift_events(tl, -30)
    k = int(30 / ode.DT_MIN)
    assert [m.step for m in moved.meals] == [m.step - k for m in tl.meals if m.step - k >= 0]
    assert np.allclose(moved.bolus_upm[:-k], tl.bolus_upm[k:])
    assert np.array_equal(np.isnan(moved.cgm), np.isnan(tl.cgm))


def test_carb_free_bolus_and_food_photo_become_meal_candidates():
    rec = _record("2026-01-10")
    anchor = datetime.fromisoformat(rec["utc_anchor"].replace("Z", "+00:00"))
    iso = lambda m: (anchor + timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
    rec["events"]["insulin_bolus_u"]["events"] = [{"timestamp": iso(480), "value": 4.0}, {"timestamp": iso(720), "value": 5.0}]
    rec["events"]["carbs_g"]["events"] = [{"timestamp": iso(720), "value": 60.0}]
    rec["events"]["food_photo_topdown"]["events"] = [{"timestamp": iso(1080)}]
    cands = [m.step for m in detect_unlogged_meals(build_timeline([rec], "a"))]
    assert int(480 / ode.DT_MIN) in cands and int(1080 / ode.DT_MIN) in cands
    assert int(720 / ode.DT_MIN) not in cands


def test_clock_offset_between_pump_and_cgm_is_recovered():
    """A population-typical body: the coarse screen cannot separate an offset from
    unusual physiology (absorption speed, hypoglycaemic uptake), a documented limit."""
    from t1d_twin.fit import select_clock_offset

    records, truth = generate_person("clk", n_days=4, seed=5, unlogged_fraction=0.0, truth_sd_scale=0.0)
    for rec in records:  # pump/meal clock 30 min ahead of the CGM
        for stream in ("insulin_bolus_u", "carbs_g", "temp_basal"):
            for ev in rec["events"][stream]["events"]:
                ts = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00")) + timedelta(minutes=30)
                ev["timestamp"] = ts.isoformat().replace("+00:00", "Z")
    tl = build_timeline(records, "clk", truth.therapy_settings)
    offset, scores = select_clock_offset(tl, truth.base, [1, 2, 3], twin_priors(), (-30.0, 0.0, 30.0))
    assert offset == -30.0, scores


def test_mood_and_sleep_export_and_stress_stream_feed_context():
    days = [_record(d) for d in ("2026-01-10", "2026-01-11")]
    anchor = datetime.fromisoformat(days[1]["utc_anchor"].replace("Z", "+00:00"))
    iso = lambda m: (anchor + timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
    days[1]["events"]["mood"]["events"] = [
        {"timestamp": iso(600), "valence": -0.5, "arousal": 0.9},   # stressed
        {"timestamp": iso(1200), "valence": 0.5, "arousal": 0.1},   # calm
    ]
    sleep = [{"dateUtc": "2026-01-11", "asleepCore": 4 * 3600, "asleepDeep": 3600, "asleepREM": 0, "asleepUnspecified": 0}]
    tl = build_timeline(days, "ctx", sleep_daily=sleep)
    f = build_features(tl)
    s1 = tl.day_steps(1)[0]
    assert f.stress[s1 + int(600 / ode.DT_MIN)] == 1.0
    assert f.stress[s1 + int(1200 / ode.DT_MIN)] == 0.0
    assert f.sleep_deficit_h[s1] == pytest.approx(3.5)
    assert f.known["stress"] and f.known["sleep"]

    garmin = _record("2026-01-12")
    garmin["dense"]["stress_level"] = {"unit": "score", "values": [80.0] * garmin["bin_count"], "observed_flag": [1] * garmin["bin_count"]}
    assert build_timeline([garmin], "g").stress[10] == pytest.approx(0.8)


def test_cycle_priors_observed_inferred_or_off():
    free = lambda p, n: p.get(n).prior_sd > 0.01
    observed = twin_priors(cycle_observed=True, sex="female")
    inferred = twin_priors(cycle_observed=False, sex=None)
    male = twin_priors(cycle_observed=False, sex="male")
    assert free(observed, "cycle_luteal_si") and not free(observed, "cycle_cos_si")
    assert free(inferred, "cycle_cos_si") and not free(inferred, "cycle_luteal_si")
    assert not any(free(male, n) for n in ("cycle_luteal_si", "cycle_menstrual_si", "cycle_cos_si", "cycle_sin_si"))


def test_site_changes_inferred_from_pump_suspensions_when_not_logged():
    days = [_record(d) for d in ("2026-01-10", "2026-01-11", "2026-01-12")]
    for d, rec in enumerate(days):
        anchor = datetime.fromisoformat(rec["utc_anchor"].replace("Z", "+00:00"))
        iso = lambda m: (anchor + timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
        # a 20-min suspension each morning at normal glucose; the one on day 1 is too soon after day 0
        rec["events"]["temp_basal"]["events"] = [{"timestamp": iso(540), "rate": 0.0, "duration": 20}]
    tl = build_timeline(days, "site")
    assert tl.site_changes_inferred
    assert len(tl.site_change_steps) == 2  # day 0 and day 2 (>= 36 h apart)


def test_response_basis_places_events_in_time_since_bins():
    from t1d_twin.model import response_basis

    series = np.zeros(400)
    series[100] = 50.0  # 50 g at step 100
    basis = response_basis(series)
    width = int(30 / ode.DT_MIN)
    assert basis[0, 100] == 50.0 and basis[0, 100 + width - 1] == 50.0
    assert basis[0, 100 + width] == 0.0 and basis[1, 100 + width] == 50.0
    assert basis[:, 99].sum() == 0.0
    assert np.allclose(basis.sum(axis=0)[100:100 + 10 * width], 50.0)


def test_rollout_insulin_response_matches_fitting_basis(synthetic):
    """The online (rollout) insulin response correction moves glucose like the precomputed fitting basis."""
    from t1d_twin.model import rollout

    _, truth, tl = synthetic
    pri = twin_priors()
    names = pri.names()
    base_u = pri.means().to(DTYPE)[None].clone()
    strong = base_u.clone()
    strong[0, names.index("ins_resp_k2")] = -0.5  # glucose drop 60-90 min after each unit of bolus
    prob = TwinProblem(tl, truth.base, [1, 2], pri, unlogged=[])
    zeros = lambda n: torch.zeros(1, n, dtype=DTYPE)
    fit_delta = (prob.simulate(strong, zeros(prob.n_days), zeros(prob.n_logged), zeros(0))
                 - prob.simulate(base_u, zeros(prob.n_days), zeros(prob.n_logged), zeros(0))).detach()[0, 0]
    w = prob.windows[0]
    sl = slice(w.start, w.start + w.length)
    grams = [(m.step, torch.tensor([m.grams], dtype=DTYPE)) for m in tl.meals]
    run = lambda u: rollout(tl, truth.base, pri, u, w.start, w.length, meal_grams=grams,
                            insulin_upm=torch.tensor(tl.basal_upm[sl] + tl.bolus_upm[sl], dtype=DTYPE)[None],
                            bolus_upm=torch.tensor(tl.bolus_upm[sl], dtype=DTYPE)[None]).detach()[0]
    roll_delta = run(strong) - run(base_u)
    tail = slice(w.loss_from, w.length)
    assert float(fit_delta[tail].abs().max()) > 10.0  # the correction matters
    assert torch.max(torch.abs(roll_delta[tail] - fit_delta[:w.length][tail])) < 0.1 * float(fit_delta[tail].abs().max())


def test_blank_day_of_regular_meal_logger_is_missing_not_fasting():
    days = [_record(d) for d in ("2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13")]
    for rec in days[:3]:
        anchor = datetime.fromisoformat(rec["utc_anchor"].replace("Z", "+00:00"))
        rec["events"]["carbs_g"]["events"] = [{"timestamp": (anchor + timedelta(hours=12)).isoformat().replace("+00:00", "Z"), "value": 50.0}]
    tl = build_timeline(days, "logger")
    assert list(tl.meals_observed) == [True, True, True, False]
    prob = TwinProblem(tl, "adult#001", [1, 2, 3], twin_priors(has_cycle=False))
    assert prob.days == [1, 2] and prob.skipped[3] == "meal log missing"

    never = build_timeline([_record(d) for d in ("2026-01-10", "2026-01-11")], "nonlogger")
    assert list(never.meals_observed) == [True, True]  # never logs: meals come from boluses, days stay usable


def test_bolus_without_dose_makes_insulin_unknown_not_nan():
    rec = _record("2026-01-10")
    anchor = datetime.fromisoformat(rec["utc_anchor"].replace("Z", "+00:00"))
    rec["events"]["insulin_bolus_u"]["events"].append({"timestamp": (anchor + timedelta(hours=8)).isoformat().replace("+00:00", "Z"), "value": float("nan")})
    tl = build_timeline([rec], "nan")
    assert not np.any(np.isnan(tl.bolus_upm))
    assert list(tl.insulin_observed) == [False]


def test_recentering_moves_mean_drift_into_core_without_changing_fitted_days(synthetic):
    from t1d_twin.fit import recenter_fit

    _, truth, tl = synthetic
    fit = fit_twin(tl, config=FitConfig(map_iters=3, iters=3, base=truth.base, clock_offsets=()), verbose=False)
    assert fit.recentered and abs(np.mean(fit.day_drift_loc)) < 1e-5
    # undo, then recentre again: the simulated SI multiplier per day is unchanged
    i = fit.param_names.index("log_si")
    shifted = [d + 0.2 for d in fit.day_drift_loc]
    before = [fit.loc[i] - 0.2 + d for d in shifted]
    fit.loc[i] -= 0.2
    fit.day_drift_loc, fit.recentered = shifted, False
    recenter_fit(fit)
    after = [fit.loc[i] + d for d in fit.day_drift_loc]
    assert np.allclose(before, after, atol=1e-6)


def test_risk_space_weights_low_range_errors_more():
    from t1d_twin.fit import RISK_SLOPE_AT_REF, risk_space

    g = torch.tensor([60.0, 120.0, 250.0])
    slope = (risk_space(g + 1.0) - risk_space(g - 1.0)) / 2.0
    assert abs(float(slope[1]) - RISK_SLOPE_AT_REF) < 1e-3
    assert float(slope[0]) > 1.8 * float(slope[1])  # an error at 60 mg/dL counts ~2x one at 120
    assert float(slope[1]) > 1.5 * float(slope[2])  # and one at 250 counts less


def test_assimilation_picks_up_a_recent_disturbance_the_warm_start_misses(synthetic):
    """Glucose pushed up by an unmodelled flux for 90 min before the origin: the assimilated forecast follows it."""
    import dataclasses

    from t1d_twin.forecast import assimilate, carried_disturbance
    from t1d_twin.model import rollout

    _, truth, tl = synthetic
    fit = truth_as_fit(truth, tl)
    s0, _ = tl.day_steps(2)
    origin = s0 + int(11 * 60 / ode.DT_MIN)
    start, horizon = origin - int(6 * 60 / ode.DT_MIN), int(60 / ode.DT_MIN)
    n = origin - start + horizon
    meals = [(m.step, torch.tensor([m.grams], dtype=DTYPE)) for m in tl.meals if m.logged]
    ins = torch.tensor(tl.basal_upm[start:start + n] + tl.bolus_upm[start:start + n], dtype=DTYPE)[None]
    bol = torch.tensor(tl.bolus_upm[start:start + n], dtype=DTYPE)[None]
    pri = fit.priors()
    u = torch.tensor(fit.loc, dtype=DTYPE)[None]
    push = torch.zeros(1, n, dtype=DTYPE)
    push[0, origin - start - int(90 / ode.DT_MIN):] = 2.0
    with torch.no_grad():
        truth_g = rollout(tl, fit.base, pri, u, start, n, meal_grams=meals, insulin_upm=ins, bolus_upm=bol, flux_upm=push)[0].numpy()
    cgm = tl.cgm.copy()
    cgm[start + 1:start + n + 1] = truth_g  # rollout output t is glucose at step t + 1
    seen = dataclasses.replace(tl, cgm=cgm)

    resets, disturbance = assimilate(fit, seen, start, origin, meals=meals, insulin_upm=ins, bolus_upm=bol)
    assert float(disturbance[-int(60 / ode.DT_MIN):].mean()) > 1.0

    offset = origin - start
    anchor = [(offset, float(cgm[origin]))]
    flux = torch.zeros(1, n, dtype=DTYPE)
    flux[0, :offset] = disturbance
    flux[0, offset:] = carried_disturbance(float(disturbance[-int(60 / ode.DT_MIN):].mean()), n - offset)
    with torch.no_grad():
        warm = rollout(seen, fit.base, pri, u, start, n, meal_grams=meals, insulin_upm=ins, bolus_upm=bol, glucose_reset=anchor)[0].numpy()
        assim = rollout(seen, fit.base, pri, u, start, n, meal_grams=meals, insulin_upm=ins, bolus_upm=bol, flux_upm=flux,
                        glucose_reset=resets + anchor)[0].numpy()
    k = offset + int(30 / ode.DT_MIN) - 1
    # the push persists here while the carried disturbance fades (30-min half-life), so only part of the gap closes
    assert abs(assim[k] - truth_g[k]) < 0.7 * abs(warm[k] - truth_g[k])


def test_empirical_priors_recentre_on_other_people_and_leave_the_person_out(synthetic):
    from t1d_twin.fit import empirical_priors

    _, truth, tl = synthetic
    fits = []
    for i in range(10):
        f = truth_as_fit(truth, tl)
        f.person_id = f"p{i}"
        f.loc[f.param_names.index("log_insulin_speed")] = 0.4 + 0.01 * i
        fits.append(f)
    fits[0].loc[fits[0].param_names.index("log_insulin_speed")] = 5.0  # the person being fitted
    pri = empirical_priors(fits, exclude_person="p0")
    spec = pri.get("log_insulin_speed")
    assert abs(spec.prior_mean - 0.45) < 1e-6
    assert spec.prior_sd == pytest.approx(0.5 * twin_priors().get("log_insulin_speed").prior_sd)  # tight spread floored at half
    assert pri.get("cycle_luteal_si") == twin_priors().get("cycle_luteal_si")  # knobs outside the list untouched
