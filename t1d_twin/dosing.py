"""Use a person's own dosing as a prior on the ratio their CGM cannot identify.

Almost every meal is bolused, so carbs and insulin arrive together and a CGM
trace pins their *net* effect while leaving each magnitude loose. A twin can
therefore forecast well with insulin that is far too weak, as long as its
carbohydrate effect or endogenous production is wrong by a matching amount. That
never shows up in forecast error, and it wrecks carb-ratio experiments, which
are a question about exactly that ratio.

Their dosing carries the missing information. Someone who boluses 12 g/U and
spends most of the day in range is telling us that, for them, one unit covers
about 12 grams. ``carb_ratio_prior`` measures what one unit and one gram do
inside a candidate twin, then shifts the insulin-sensitivity prior until the
twin's balanced carb ratio matches what the person actually does. The prior
keeps its usual width, so a person whose dosing is genuinely wrong can still be
pulled away from it by their data.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.data import PersonTimeline
from t1d_twin.model import DTYPE
from t1d_twin.params import RESPONSE_BIN_MIN, TwinPriors

PROBE_HOURS = 5.0
PROBE_DOSE_U = 1.0
PROBE_CARBS_G = 10.0
PROBE_GLUCOSE = 140.0
BOLUS_MEAL_WINDOW_MIN = 20.0
MIN_MEALS = 8              # below this the ratio is too noisy to trust
MAX_LOG_SI_SHIFT = 1.2     # about a factor of three in either direction
CALIBRATION_ROUNDS = 4


def observed_carb_ratio(tl: PersonTimeline, steps: set[int] | None = None) -> tuple[float, int]:
    """Median grams per unit this person bolused for logged meals, and how many meals."""
    width = int(BOLUS_MEAL_WINDOW_MIN / ode.DT_MIN)
    ratios = []
    for m in tl.meals:
        if not m.logged or m.grams <= 0 or (steps is not None and m.step not in steps):
            continue
        units = float(np.nansum(tl.bolus_upm[max(0, m.step - width): m.step + width]) * ode.DT_MIN)
        if units > 0.2:
            ratios.append(m.grams / units)
    return (float(np.median(ratios)) if ratios else float("nan")), len(ratios)


@lru_cache(maxsize=256)
def probe_carb_ratio(base_name: str, log_si: float = 0.0, log_carb_effect: float = 0.0, body_mass_kg: float | None = None) -> tuple[float, float]:
    """(carb ratio g/U, ISF mg/dL/U) of a base patient held at basal, by direct simulation.

    One unit is delivered, and separately ten grams eaten, into an otherwise
    steady patient; the peak glucose difference gives the correction factor and
    the carbohydrate factor, whose ratio is the carb ratio that balances them.
    """
    base = ode.base_patient(base_name)
    p, Vg, x0 = ode.stack_params([base])
    n = int(PROBE_HOURS * 60 / ode.DT_MIN)
    basal = base.basal_u_per_hr / 60.0
    ins = torch.full((1, n), basal, dtype=torch.float64)
    cho = torch.zeros(1, n, dtype=torch.float64)
    starts = torch.zeros(1, n, dtype=torch.bool)
    si = torch.full((1, n), math.exp(log_si), dtype=torch.float64)
    one = torch.ones(1, n, dtype=torch.float64)
    x = ode.initial_state(x0, Vg, torch.tensor([PROBE_GLUCOSE], dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    run = lambda i, c, s: ode.simulate(x, p, Vg, c, s, i, si, one, one).numpy()[0]

    base_trace = run(ins, cho, starts)
    dosed = ins.clone()
    dosed[0, 0] += PROBE_DOSE_U / ode.DT_MIN
    isf = float(np.max(base_trace - run(dosed, cho, starts)))
    fed_cho, fed_starts = cho.clone(), starts.clone()
    grams = PROBE_CARBS_G * math.exp(log_carb_effect)
    fed_cho[0, 0] = grams / ode.DT_MIN
    fed_starts[0, 0] = True
    csf = float(np.max(run(ins, fed_cho, fed_starts) - base_trace)) / PROBE_CARBS_G
    return (isf / csf if csf > 0 else float("nan")), isf


def carb_ratio_prior(priors: TwinPriors, base_name: str, target_g_per_u: float) -> tuple[TwinPriors, dict]:
    """Centre the log_si prior so the twin's balanced carb ratio matches the person's dosing.

    The prior keeps its width, so this moves where the fit starts from, not where
    it is allowed to end up. Returns the priors and what was done, for diagnostics.
    """
    spec = priors.get("log_si")
    at_zero, isf_at_zero = probe_carb_ratio(base_name)
    info = {"base_carb_ratio_g_per_u": at_zero, "base_isf_mgdl_per_u": isf_at_zero, "target_g_per_u": target_g_per_u}
    if not np.isfinite(target_g_per_u) or not np.isfinite(at_zero) or target_g_per_u <= 0:
        info["applied"] = False
        return priors, info
    # the carb ratio rises with insulin sensitivity but not proportionally
    # (glucose uptake saturates), so close the gap in a few passes
    shift = math.log(target_g_per_u / at_zero)
    for _ in range(CALIBRATION_ROUNDS):
        shift = float(np.clip(shift, -MAX_LOG_SI_SHIFT, MAX_LOG_SI_SHIFT))
        ratio, isf = probe_carb_ratio(base_name, round(shift, 4))
        if not np.isfinite(ratio) or ratio <= 0:
            break
        gap = math.log(target_g_per_u / ratio)
        if abs(gap) < 0.02:
            break
        shift += gap
    shift = float(np.clip(shift, -MAX_LOG_SI_SHIFT, MAX_LOG_SI_SHIFT))
    reached, reached_isf = probe_carb_ratio(base_name, round(shift, 4))
    info.update({"applied": True, "log_si_prior_mean": spec.prior_mean + shift, "shift": shift,
                 "reached_carb_ratio_g_per_u": reached, "reached_isf_mgdl_per_u": reached_isf,
                 "clipped": abs(shift) >= MAX_LOG_SI_SHIFT - 1e-9})
    return priors.with_prior("log_si", spec.prior_mean + shift, spec.prior_sd), info


def base_carb_ratios(names=ode.ADULTS) -> dict[str, float]:
    """Balanced carb ratio of each base patient, for picking one that suits a person."""
    return {n: probe_carb_ratio(n)[0] for n in names}


# --- the same probe, differentiable, for use inside the fit ------------------

RATIO_LOG_SD = 0.2       # how far the twin's carb ratio may sit from the person's dosing
PROBE_STEPS = int(PROBE_HOURS * 60 / ode.DT_MIN)


def carb_ratio_torch(g: dict, base_name: str, body_mass_kg: float | None) -> torch.Tensor:
    """[K] balanced carb ratio (g/U) of each parameter set, by simulating a dose and a meal.

    Three arms per parameter set (nothing, one unit, ten grams) run as one batch
    from the same steady state, so the difference is the twin's own dose response
    including its learned insulin and carbohydrate response corrections.
    """
    from t1d_twin.model import personalise_params, response_basis, response_coefs

    K = g["log_si"].shape[0]
    base = ode.base_patient(base_name)
    p, Vg, x0b = personalise_params(base, g, body_mass_kg, 3)  # arms interleaved per parameter set
    n, B = PROBE_STEPS, K * 3
    basal = base.basal_u_per_hr / 60.0
    ins = torch.full((B, n), basal, dtype=DTYPE)
    cho = torch.zeros(B, n, dtype=DTYPE)
    starts = torch.zeros(B, n, dtype=torch.bool)
    dose, fed = slice(1, None, 3), slice(2, None, 3)
    ins[dose, 0] = ins[dose, 0] + PROBE_DOSE_U / ode.DT_MIN
    cho[fed, 0] = PROBE_CARBS_G / ode.DT_MIN
    starts[fed, 0] = True
    bolus = torch.zeros(B, n, dtype=DTYPE)
    bolus[dose, 0] = PROBE_DOSE_U / ode.DT_MIN

    # the learned response corrections apply to the probe's own dose and meal
    flux = None
    carb_coefs, ins_coefs = response_coefs(g, "carb_resp_k"), response_coefs(g, "ins_resp_k")
    if carb_coefs is not None:
        series = np.zeros(n)
        series[0] = PROBE_CARBS_G
        basis = torch.tensor(response_basis(series)[:, :n], dtype=DTYPE)
        contribution = torch.zeros(B, n, dtype=DTYPE)
        contribution[fed] = carb_coefs @ basis
        flux = contribution
    x = ode.initial_state(x0b, Vg, torch.full((B,), PROBE_GLUCOSE, dtype=DTYPE), torch.ones(B, dtype=DTYPE))
    one = torch.ones(B, n, dtype=DTYPE)
    # sensitivity, EGP and uptake multipliers are already inside ``p``
    sim = ode.simulate(x, p, Vg, cho, starts, ins, one, one, one, flux=flux, ins_response=None if ins_coefs is None else ins_coefs.repeat_interleave(3, dim=0),
                       bolus_upm=bolus, response_bin_steps=int(RESPONSE_BIN_MIN / ode.DT_MIN))
    quiet, dosed, eaten = sim[0::3], sim[1::3], sim[2::3]
    isf = (quiet - dosed).max(dim=1).values
    csf = (eaten - quiet).max(dim=1).values / PROBE_CARBS_G
    return isf / csf.clamp_min(1e-3)


def carb_ratio_logprior(g: dict, base_name: str, body_mass_kg: float | None, target_g_per_u: float) -> torch.Tensor:
    """[K] log prior tying each parameter set's dose response to the person's dosing."""
    ratio = carb_ratio_torch(g, base_name, body_mass_kg).clamp_min(1e-3)
    return -0.5 * ((torch.log(ratio) - math.log(target_g_per_u)) / RATIO_LOG_SD) ** 2


# --- settings for people whose pump settings were never recorded ------------

def infer_settings(tl: PersonTimeline, days: list[int]) -> tuple[PersonTimeline, dict]:
    """A copy of ``tl`` with carb ratio, ISF and basal schedule filled in from how the person doses.

    Public datasets record what the pump delivered, not how it was set, and a
    settings experiment needs the settings to scale. The inferred values are
    only a starting point that reproduces their behaviour; the experiment is
    about what happens when they change, so the relative effects are what count:

    - carb ratio: median grams per unit over bolused logged meals;
    - ISF: 1800 / median total daily insulin;
    - basal: median delivered basal rate for each hour of the local day.
    Settings already present in the records are kept.
    """
    import dataclasses

    steps = {s for d in days for s in range(*tl.day_steps(d))}
    cr, n_meals = observed_carb_ratio(tl, steps)
    daily = []
    for d in days:
        s0, s1 = tl.day_steps(d)
        total = (np.nansum(tl.basal_upm[s0:s1]) + np.nansum(tl.bolus_upm[s0:s1])) * ode.DT_MIN
        if total > 0:
            daily.append(total)
    tdd = float(np.median(daily)) if daily else float("nan")
    isf = 1800.0 / tdd if tdd > 0 else float("nan")

    hour = np.floor(tl.local_hour).astype(int) % 24
    in_days = np.zeros(tl.n_steps, bool)
    in_days[list(steps)] = True
    profile = np.array([np.nanmedian(np.where((hour == h) & in_days, tl.basal_upm, np.nan)) * 60.0
                        if np.any((hour == h) & in_days & ~np.isnan(tl.basal_upm)) else np.nan for h in range(24)])
    if np.any(np.isnan(profile)) and np.any(~np.isnan(profile)):
        profile = np.where(np.isnan(profile), np.nanmedian(profile), profile)

    fill = lambda existing, value: existing if np.any(~np.isnan(existing)) else np.full(tl.n_steps, value, dtype=float)
    out = dataclasses.replace(
        tl,
        cr=fill(tl.cr, cr),
        isf=fill(tl.isf, isf),
        basal_setting_uph=tl.basal_setting_uph if np.any(~np.isnan(tl.basal_setting_uph)) else profile[hour],
    )
    info = {"carb_ratio_g_per_u": cr, "n_bolused_meals": n_meals, "isf_mgdl_per_u": isf, "tdd_u": tdd,
            "basal_profile_uph": [None if np.isnan(v) else round(float(v), 3) for v in profile],
            "source": "inferred from delivered insulin and logged meals (no settings in the records)"}
    return out, info
