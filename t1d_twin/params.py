"""Personal twin parameters and their priors.

Every parameter lives on an unconstrained real line with a Normal prior; the
``transform`` maps it to the value the model uses. Prior means come from the
effect sizes already encoded in ``t1d_sim.physiology.apply_context_effectors``
(Spiegel 1999 sleep, Brown 2015 cycle, the 40%/16h exercise boost, the dawn
ramp) and, when available, from the onboarding questionnaire priors in
``t1d_sim.questionnaire``.

Physiological parameters are separate from therapy settings: nothing here is a
CR, ISF or basal rate. Settings drive the controller; these drive the body.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import math
import torch


def _exp(u):
    return torch.exp(u)


def _ident(u):
    return u


@dataclass(frozen=True)
class ParamSpec:
    name: str
    prior_mean: float
    prior_sd: float
    transform: Callable = _ident
    description: str = ""


# Core physiology: log-scale multipliers on the matched UVA/Padova adult.
CORE = (
    ParamSpec("log_si", 0.0, 0.5, _exp, "insulin sensitivity (Vmx) multiplier"),
    ParamSpec("log_egp", 0.0, 0.25, _exp, "endogenous glucose production (kp1) multiplier"),
    ParamSpec("log_insulin_speed", 0.0, 0.25, _exp, "subcutaneous insulin absorption (ka1, ka2, kd) multiplier"),
    ParamSpec("log_insulin_action_speed", 0.0, 0.3, _exp, "insulin action onset (p2u, ki) multiplier; ultra-rapid insulins act sooner"),
    ParamSpec("log_carb_speed", 0.0, 0.4, _exp, "carb absorption (kabs, kmax, kmin) multiplier"),
    ParamSpec("log_carb_effect", 0.0, 0.1, _exp, "carb bioavailability (f) multiplier"),
    ParamSpec("log_hypo_uptake", math.log(0.1), 1.0, _exp,
              "S2013 hypoglycaemia term r1: extra insulin-dependent uptake per unit risk below basal glucose"),
)

# Context sensitivities: how much this person's physiology moves with context.
CONTEXT = (
    ParamSpec("cycle_luteal_si", -0.105, 0.08, _ident, "fractional SI change in luteal phase"),
    ParamSpec("cycle_menstrual_si", 0.07, 0.06, _ident, "fractional SI change in menstrual phase"),
    ParamSpec("exercise_si", 0.40, 0.15, _ident, "SI boost after exercise at full intensity (decays, tau 16h)"),
    ParamSpec("exercise_uptake", 0.30, 0.20, _ident, "insulin-independent uptake (Vm0) boost during exercise"),
    ParamSpec("sleep_si_per_h", 0.035, 0.02, _ident, "SI drop per hour of sleep below 8.5h (EGP rises at 0.71x)"),
    ParamSpec("dawn_egp", 0.10, 0.08, _ident, "EGP rise at the 03:00-08:00 dawn peak"),
    ParamSpec("site_age_si_per_day", 0.02, 0.03, _ident, "SI loss per day of infusion-site age after day 1"),
    ParamSpec("stress_egp", 0.10, 0.10, _ident, "EGP rise at full acute stress (Garmin stress or mood)"),
    ParamSpec("stress_si", 0.05, 0.05, _ident, "SI drop at full acute stress"),
    ParamSpec("cycle_cos_si", 0.0, 0.06, _ident, "inferred 28-day SI rhythm, cosine part (when cycle days are not recorded)"),
    ParamSpec("cycle_sin_si", 0.0, 0.06, _ident, "inferred 28-day SI rhythm, sine part"),
    ParamSpec("circadian_cos_si", 0.0, 0.1, _ident, "24-hour SI rhythm by local time, cosine part (peak at midnight when positive)"),
    ParamSpec("circadian_sin_si", 0.0, 0.1, _ident, "24-hour SI rhythm by local time, sine part (peak at 06:00 when positive)"),
)

# Observation noise and day-to-day drift (hyperparameters, fitted too).
NOISE = (
    ParamSpec("log_cgm_sd", math.log(12.0), 0.3, _exp, "CGM residual sd floor (mg/dL)"),
    ParamSpec("log_day_si_sd", math.log(0.15), 0.4, _exp, "sd of daily log-SI drift"),
    ParamSpec("log_day_egp_sd", math.log(0.08), 0.4, _exp, "sd of daily log-EGP drift (unrecorded stress, illness)"),
)

# Personal response corrections: extra glucose flux (mg/kg/min) per gram of logged
# carbs / per unit of bolus insulin, by time since the event (30-min bins, 0-5 h).
# Shrunk toward zero; they learn systematic meal/insulin response shape the
# mechanistic model gets wrong, and unlike the free disturbance they carry over
# to new days and respond to settings changes.
RESPONSE_BINS = 10
RESPONSE_BIN_MIN = 30.0
KERNELS = tuple(
    ParamSpec(f"carb_resp_k{j}", 0.0, 0.01, _ident, f"flux per g carbs, {int(j * RESPONSE_BIN_MIN)}-{int((j + 1) * RESPONSE_BIN_MIN)} min after") for j in range(RESPONSE_BINS)
) + tuple(
    ParamSpec(f"ins_resp_k{j}", 0.0, 0.15, _ident, f"flux per U bolus, {int(j * RESPONSE_BIN_MIN)}-{int((j + 1) * RESPONSE_BIN_MIN)} min after") for j in range(RESPONSE_BINS)
)

GLOBAL_SPECS = CORE + CONTEXT + KERNELS + NOISE

# Per-event latents (one per logged meal / unlogged-meal candidate / day).
MEAL_SCALE_PRIOR = (0.0, 0.25)       # log carb-count error of a logged meal
UNLOGGED_LOG_G_PRIOR = (math.log(8.0), 1.2)   # grams of a flagged rise: 5-95% ~1-58 g, so false flags can shrink to ~0
DAY_SI_PRIOR_MEAN = 0.0              # daily drift, sd = exp(log_day_si_sd)
LOGGED_SHIFT_PRIOR_SD = 1.5          # recorded meal time vs actual eating, in 10-min units (15 min sd)
UNLOGGED_SHIFT_PRIOR_SD = 2.0        # flagged-rise placement error, in 10-min units
# Disturbance flux (mg/kg/min per 30-min block): Student-t, mostly ~0 with heavy
# tails, so an unlogged meal (~3-6 mg/kg/min for an hour) or a brisk walk is cheap
# where the data demand it, while small systematic offsets are not.
FLUX_PRIOR_SCALE = 0.3
FLUX_PRIOR_DF = 2.0
FLUX_CLAMP = 8.0


@dataclass
class TwinPriors:
    specs: tuple[ParamSpec, ...] = field(default_factory=lambda: GLOBAL_SPECS)

    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def get(self, name: str) -> ParamSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def means(self) -> torch.Tensor:
        return torch.tensor([s.prior_mean for s in self.specs], dtype=torch.float64)

    def sds(self) -> torch.Tensor:
        return torch.tensor([s.prior_sd for s in self.specs], dtype=torch.float64)

    def with_prior(self, name: str, mean: float, sd: float | None = None) -> "TwinPriors":
        specs = tuple(
            replace(s, prior_mean=float(mean), prior_sd=float(s.prior_sd if sd is None else sd)) if s.name == name else s
            for s in self.specs
        )
        return TwinPriors(specs)


PINNED_SD = 0.001


def twin_priors(
    questionnaire_priors: dict[str, tuple[float, float]] | None = None,
    *,
    has_cycle: bool = True,
    cycle_observed: bool = True,
    sex: str | None = None,
    base: TwinPriors | None = None,
) -> TwinPriors:
    """Population priors, specialised by questionnaire answers when present.

    ``questionnaire_priors`` is the output of
    ``t1d_sim.questionnaire.questionnaire_to_patientconfig_priors``.

    Cycle: with recorded cycle days the phase effects are fitted; without them
    (and unless the person is male / reports no cycle) a 28-day SI rhythm with
    free amplitude and phase is fitted instead, so the cycle is inferred.
    ``base`` replaces the hand-set population priors (e.g. ``fit.empirical_priors``).
    """
    priors = base or TwinPriors()
    q = questionnaire_priors or {}
    no_cycle = (not has_cycle) or sex == "male"
    if no_cycle or cycle_observed:
        priors = priors.with_prior("cycle_cos_si", 0.0, PINNED_SD).with_prior("cycle_sin_si", 0.0, PINNED_SD)
    if not no_cycle and not cycle_observed:
        priors = priors.with_prior("cycle_luteal_si", 0.0, PINNED_SD).with_prior("cycle_menstrual_si", 0.0, PINNED_SD)

    if "cycle_sensitivity" in q and cycle_observed and not no_cycle:
        cs, cs_sd = q["cycle_sensitivity"]
        # physiology.py: luteal r = 1 - (0.03 + 0.15 cs); menstrual 1 + (0.03 + 0.08 cs)
        priors = priors.with_prior("cycle_luteal_si", -(0.03 + 0.15 * cs), max(0.03, 0.15 * cs_sd + 0.04))
        priors = priors.with_prior("cycle_menstrual_si", 0.03 + 0.08 * cs, max(0.03, 0.08 * cs_sd + 0.03))
    if no_cycle:
        priors = priors.with_prior("cycle_luteal_si", 0.0, PINNED_SD)
        priors = priors.with_prior("cycle_menstrual_si", 0.0, PINNED_SD)

    if "isf_multiplier" in q:
        # isf_multiplier > 1 means "more sensitive than typical" in the sim.
        m, sd = q["isf_multiplier"]
        priors = priors.with_prior("log_si", math.log(max(m, 1e-3)), 0.5)
    if "stress_reactivity" in q:
        r = q["stress_reactivity"][0]
        # physiology.py scales the sleep SI drop by (0.7 + 0.3 * stress_reactivity)
        priors = priors.with_prior("sleep_si_per_h", 0.035 * (0.7 + 0.3 * r))
    if "exercise_intensity_mean" in q and "fitness_level" in q:
        fit = q["fitness_level"][0]
        priors = priors.with_prior("exercise_si", 0.40 * (0.8 + 0.4 * fit))
    return priors
