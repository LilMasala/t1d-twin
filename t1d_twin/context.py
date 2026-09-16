"""Context signals -> per-step physiology multipliers.

Features are computed once per timeline (numpy); the multipliers are a cheap
torch function of the person's context parameters, so the same code generates
synthetic data, fits real data and runs experiments.

Unknown context contributes nothing: no cycle data means no cycle effect, no
sleep data means no sleep effect. A fitted effect is an effective parameter
for that person, not evidence of a mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from t1d_twin.data import EXERCISE_HRR_THRESHOLD, HR_MAX_PROXY, PersonTimeline
from t1d_twin.ode import DT_MIN

SLEEP_REF_H = 8.5          # physiology.py SLEEP_REF_MIN = 510
SLEEP_EGP_RATIO = 0.025 / 0.035
EXERCISE_TAU_H = 16.0      # physiology.py post-exercise decay
EXERCISE_FULL_DOSE_MIN = 45.0
CYCLE_LEN = 28


@dataclass
class ContextFeatures:
    luteal: np.ndarray          # 1 in luteal phase
    menstrual: np.ndarray       # 1 in menstrual phase
    exercise_now: np.ndarray    # intensity 0..1 now
    exercise_load: np.ndarray   # decayed post-exercise dose, ~1 after 45 min at full intensity
    sleep_deficit_h: np.ndarray # hours below 8.5 last night (0 when unknown)
    dawn_ramp: np.ndarray       # 0..1 sine ramp over 03:00-08:00
    site_age_excess_d: np.ndarray  # days of site age beyond the first (0 when unknown)
    stress: np.ndarray          # 0..1 acute stress (0 when unknown)
    cycle_cos: np.ndarray       # cos(2 pi t / 28 d), for the inferred cycle rhythm
    cycle_sin: np.ndarray
    day_cos: np.ndarray         # cos(2 pi local hour / 24), for the 24-hour SI rhythm
    day_sin: np.ndarray
    day_index: np.ndarray
    known: dict[str, bool]


def cycle_phase_masks(days_since_period: np.ndarray, day_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same phase boundaries as ``behavior._cycle_phase`` (fractions of 28 days)."""
    luteal = np.zeros(day_index.size)
    menstrual = np.zeros(day_index.size)
    for d, dsp in enumerate(days_since_period):
        if np.isnan(dsp):
            continue
        f = (int(dsp) % CYCLE_LEN) / CYCLE_LEN
        mask = day_index == d
        if f < 0.18:
            menstrual[mask] = 1.0
        elif f >= 0.55:
            luteal[mask] = 1.0
    return luteal, menstrual


def build_features(tl: PersonTimeline) -> ContextFeatures:
    n = tl.n_steps
    luteal, menstrual = cycle_phase_masks(tl.days_since_period, tl.day_index)

    # Exercise intensity: the fraction of time exercising (exercise minutes per
    # 5-min bin) times effort from heart-rate reserve (at least moderate when
    # exercise is logged). Without logged exercise, a sustained heart rate above
    # 30% reserve still counts. App heart rate is an hourly mean, so effort is
    # diluted over the hour; the fitted exercise effects absorb that scale.
    rhr_day = np.where(np.isnan(tl.resting_hr), np.nanmedian(tl.resting_hr) if np.any(~np.isnan(tl.resting_hr)) else 62.0, tl.resting_hr)
    rhr = rhr_day[np.clip(tl.day_index, 0, len(rhr_day) - 1)]
    hrr = (tl.hr - rhr) / np.maximum(HR_MAX_PROXY - rhr, 1.0)
    effort = np.where(np.isnan(hrr), 0.0, np.clip(hrr, 0.0, 1.0))
    frac = np.clip(tl.exercise_min / 5.0, 0.0, 1.0)
    intensity = np.where(frac > 0, frac * np.maximum(effort, 0.6), np.where(effort >= EXERCISE_HRR_THRESHOLD, effort, 0.0))

    decay = np.exp(-DT_MIN / (EXERCISE_TAU_H * 60.0))
    load = np.empty(n)
    acc = 0.0
    for s in range(n):
        acc = acc * decay + intensity[s] * DT_MIN / EXERCISE_FULL_DOSE_MIN
        load[s] = acc
    load = np.minimum(load, 1.5)

    # Sleep: asleep minutes between 18:00 the previous local day and 12:00.
    sleep_def = np.zeros(n)
    sleep_known = bool(np.any(~np.isnan(tl.asleep)) or np.any(~np.isnan(tl.sleep_hours)))
    for d in range(tl.n_days):
        s0, s1 = tl.day_steps(d)
        lo = max(0, s0 - int(6 * 60 / DT_MIN))
        hi = min(n, s0 + int(12 * 60 / DT_MIN))
        window = tl.asleep[lo:hi]
        if np.all(np.isnan(window)):
            if np.isnan(tl.sleep_hours[d]):
                continue
            slept_h = float(tl.sleep_hours[d])  # nightly total from the sleep export
        else:
            slept_h = np.nansum(window) * DT_MIN / 60.0
        sleep_def[s0:s1] = max(0.0, SLEEP_REF_H - slept_h)

    h = tl.local_hour
    dawn = np.where((h >= 3.0) & (h <= 8.0), np.sin(np.pi * (h - 3.0) / 5.667), 0.0)

    site_excess = np.zeros(n)
    if tl.site_change_steps:
        steps = np.array(tl.site_change_steps)
        idx = np.arange(n)
        pos = np.searchsorted(steps, idx, side="right") - 1
        age_d = np.where(pos >= 0, (idx - steps[np.maximum(pos, 0)]) * DT_MIN / 1440.0, 0.0)
        site_excess = np.where(pos >= 0, np.clip(age_d - 1.0, 0.0, 6.0), 0.0)

    t_days = np.arange(n) * DT_MIN / 1440.0
    return ContextFeatures(
        stress=np.nan_to_num(tl.stress, nan=0.0),
        cycle_cos=np.cos(2 * np.pi * t_days / CYCLE_LEN),
        cycle_sin=np.sin(2 * np.pi * t_days / CYCLE_LEN),
        day_cos=np.cos(2 * np.pi * h / 24.0),
        day_sin=np.sin(2 * np.pi * h / 24.0),
        luteal=luteal,
        menstrual=menstrual,
        exercise_now=intensity,
        exercise_load=load,
        sleep_deficit_h=sleep_def,
        dawn_ramp=dawn,
        site_age_excess_d=site_excess,
        day_index=tl.day_index.copy(),
        known={
            "cycle": bool(np.any(~np.isnan(tl.days_since_period))),
            "heart_rate": bool(np.any(~np.isnan(tl.hr))),
            "sleep": sleep_known,
            "site_changes": bool(tl.site_change_steps),
            "site_changes_inferred": bool(tl.site_changes_inferred),
            "stress": bool(np.any(~np.isnan(tl.stress))),
        },
    )


def multipliers(
    feats: dict[str, torch.Tensor],
    ctx: dict[str, torch.Tensor],
    day_log_si: torch.Tensor,
    day_log_egp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-step (si, egp, vm0) multipliers, [K, W, S].

    ``feats`` values are [W, S] feature windows; ``ctx`` values are [K] context
    parameters; ``day_log_si`` / ``day_log_egp`` are [K, W, S] daily drifts
    already gathered.
    """
    def c(name):
        return ctx[name][:, None, None]

    si = torch.exp(day_log_si)
    si = si * (1.0 + c("cycle_luteal_si") * feats["luteal"] + c("cycle_menstrual_si") * feats["menstrual"])
    si = si * (1.0 + c("exercise_si") * feats["exercise_load"])
    si = si * (1.0 - c("sleep_si_per_h") * feats["sleep_deficit_h"])
    si = si * (1.0 - c("site_age_si_per_day") * feats["site_age_excess_d"])
    si = si * (1.0 - c("stress_si") * feats["stress"])
    si = si * (1.0 + c("cycle_cos_si") * feats["cycle_cos"] + c("cycle_sin_si") * feats["cycle_sin"])
    si = si * (1.0 + c("circadian_cos_si") * feats["day_cos"] + c("circadian_sin_si") * feats["day_sin"])
    egp = (1.0 + c("dawn_egp") * feats["dawn_ramp"]) * (1.0 + SLEEP_EGP_RATIO * c("sleep_si_per_h") * feats["sleep_deficit_h"])
    egp = egp * (1.0 + c("stress_egp") * feats["stress"])
    if day_log_egp is not None:
        egp = egp * torch.exp(day_log_egp)
    vm0 = 1.0 + c("exercise_uptake") * feats["exercise_now"]
    return si.clamp(0.2, 5.0), egp.clamp(0.3, 3.0), vm0.clamp(0.5, 3.0)
