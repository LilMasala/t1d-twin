"""Settings experiments on fitted twins: CR / ISF / basal arms, paired.

Every arm replays the same person-days (their real context and meals) with
the same posterior samples, so arm differences are paired. The body eats the
*fitted* carbs (logged x fitted count error, plus likely unlogged meals); the
controller only knows the *logged* carbs, exactly as in real life.

The controller is a generic research basal-bolus controller mirroring
``t1d_sim.uvapadova.TherapyScheduleBBController``. It is not Tandem
Control-IQ; results are research estimates, not dosing advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.context import build_features
from t1d_twin.data import PersonTimeline, shift_events
from t1d_twin.fit import TwinFit
from t1d_twin.model import BURN_IN_H, DTYPE, rollout
from t1d_twin.params import MEAL_SCALE_PRIOR

DEFAULT_TARGET = 110.0
DECISION_MIN = 5.0


class BasalBolusController:
    """Batched generic controller. Settings multipliers are per batch row [B]."""

    def __init__(
        self,
        tl: PersonTimeline,
        *,
        cr_mult: torch.Tensor,
        isf_mult: torch.Tensor,
        basal_mult: torch.Tensor,
        aid: bool = False,
        target: float = DEFAULT_TARGET,
        record: bool = False,
    ) -> None:
        self.tl = tl
        self.cr_mult, self.isf_mult, self.basal_mult = cr_mult, isf_mult, basal_mult
        self.aid = aid
        self.target = target
        self.meal_at = {m.step: m.grams for m in tl.meals if m.logged}
        self.prev_cgm: torch.Tensor | None = None
        self.last_decision_cgm: torch.Tensor | None = None
        self.basal_upm: torch.Tensor | None = None
        self.last_bolus_u: torch.Tensor | None = None
        self.record = record
        self.history: list[tuple[int, torch.Tensor, torch.Tensor]] = []  # (step, basal U/min, bolus U)
        for name, arr in (("cr", tl.cr), ("isf", tl.isf), ("basal_rate_u_per_hr", tl.basal_setting_uph)):
            if np.all(np.isnan(arr)):
                raise ValueError(f"therapy setting {name} missing; cannot run a settings controller")

    def _setting(self, arr: np.ndarray, step: int) -> float:
        v = arr[min(step, arr.size - 1)]
        if np.isnan(v):
            v = float(np.nanmedian(arr))
        return float(v)

    def __call__(self, step: int, cgm: torch.Tensor) -> torch.Tensor:
        new_decision = int(step * ode.DT_MIN // DECISION_MIN) != int((step - 1) * ode.DT_MIN // DECISION_MIN) or self.basal_upm is None
        cr = self._setting(self.tl.cr, step) * self.cr_mult
        isf = self._setting(self.tl.isf, step) * self.isf_mult
        bolus = torch.zeros_like(cgm)

        if new_decision:
            basal_uph = self._setting(self.tl.basal_setting_uph, step) * self.basal_mult
            prev = self.last_decision_cgm
            predicted = cgm if prev is None else cgm + 6.0 * (cgm - prev)
            predicted = predicted.clamp(40.0, 450.0)
            if self.aid:
                scale = (1.0 + (predicted - self.target) / 120.0).clamp(0.0, 2.5)
                scale = torch.where(predicted < 70.0, torch.zeros_like(scale), scale)
                basal_uph = basal_uph * scale
                minute = self.tl.local_hour[min(step, self.tl.n_steps - 1)] * 60.0
                if step not in self.meal_at and (minute % 60.0) < DECISION_MIN:
                    auto = 0.60 * torch.clamp((predicted - self.target) / isf, min=0.0)
                    bolus = bolus + torch.where(predicted > 180.0, auto, torch.zeros_like(auto))
            self.basal_upm = basal_uph / 60.0
            self.last_decision_cgm = cgm

        if step in self.meal_at:
            bolus = bolus + self.meal_at[step] / cr
            bolus = bolus + torch.where(cgm > self.target, (cgm - self.target) / isf, torch.zeros_like(cgm))

        self.last_bolus_u = bolus
        if self.record:
            self.history.append((step, self.basal_upm.clone(), bolus.clone()))
        return self.basal_upm + bolus / ode.DT_MIN


@dataclass
class Arm:
    name: str
    cr_mult: float = 1.0
    isf_mult: float = 1.0
    basal_mult: float = 1.0


def glycemic_metrics(glucose: torch.Tensor) -> dict[str, torch.Tensor]:
    """[B, T] -> per-row metrics."""
    return {
        "tir_70_180": ((glucose >= 70) & (glucose <= 180)).float().mean(1),
        "tbr_70": (glucose < 70).float().mean(1),
        "tbr_54": (glucose < 54).float().mean(1),
        "tar_180": (glucose > 180).float().mean(1),
        "tar_250": (glucose > 250).float().mean(1),
        "mean_mgdl": glucose.mean(1),
        "cv": glucose.std(1) / glucose.mean(1),
    }


def _meal_draws(fit: TwinFit, tl: PersonTimeline, S: int, gen: torch.Generator) -> list[tuple[int, torch.Tensor]]:
    """What the body eats per sample: fitted amounts (count-error prior for unfitted meals), at the fitted mean time."""
    shift = lambda minutes: int(round(minutes / ode.DT_MIN))
    logged_shift = fit.meal_shift_min or [0.0] * len(fit.logged_meal_steps)
    unlogged_shift = fit.unlogged_shift_min or [0.0] * len(fit.unlogged_meal_steps)
    scale_post = {s: (l, sd, sh) for s, l, sd, sh in zip(fit.logged_meal_steps, fit.meal_scale_loc, fit.meal_scale_sd, logged_shift)}
    draws = []
    for m in tl.meals:
        loc, sd, sh = scale_post.get(m.step, (fit.carb_count_bias, MEAL_SCALE_PRIOR[1], 0.0))
        draws.append((m.step + shift(sh), m.grams * torch.exp(loc + sd * torch.randn(S, generator=gen, dtype=DTYPE))))
    for step, loc, sd, sh in zip(fit.unlogged_meal_steps, fit.unlogged_log_g_loc, fit.unlogged_log_g_sd, unlogged_shift):
        draws.append((step + shift(sh), torch.exp(loc + sd * torch.randn(S, generator=gen, dtype=DTYPE))))
    return draws


def _fitted_flux(fit: TwinFit, tl: PersonTimeline, start: int, n_steps: int) -> torch.Tensor | None:
    """Posterior-mean disturbance flux on fitted days (the unlogged food/activity of that day), zero elsewhere."""
    from t1d_twin.model import FLUX_BLOCK_STEPS

    if not fit.day_flux:
        return None
    flux = torch.zeros(1, n_steps, dtype=DTYPE)
    for date, blocks in fit.day_flux.items():
        if date not in tl.day_dates:
            continue
        s0, s1 = tl.day_steps(tl.day_dates.index(date))
        for s in range(max(s0, start), min(s1, start + n_steps)):
            k = (s - s0) // FLUX_BLOCK_STEPS
            if k < len(blocks):
                flux[0, s - start] = blocks[k]
    return flux


def run_settings_experiment(
    fit: TwinFit,
    tl: PersonTimeline,
    arms: list[Arm],
    *,
    samples: int = 32,
    days: tuple[int, int] | None = None,
    aid: bool = False,
    seed: int = 0,
    globals_override: torch.Tensor | None = None,
) -> dict:
    """Run every arm over the same days and posterior samples; return paired summaries.

    ``globals_override`` ([samples, G]) swaps the person's posterior for other
    parameter draws (e.g. synthetic people from ``population``) while keeping
    this person's days as the scenario.
    """
    gen = torch.Generator().manual_seed(seed)
    priors = fit.priors()
    tl = shift_events(tl, fit.event_clock_offset_min)
    feats = build_features(tl)
    d0, d1 = days if days is not None else (1, tl.n_days)
    s0, _ = tl.day_steps(d0)
    _, s1 = tl.day_steps(d1 - 1)
    burn = int(BURN_IN_H * 60 / ode.DT_MIN)
    start = max(0, s0 - burn)
    n_steps = s1 - start

    u = fit.sample_globals(samples, gen) if globals_override is None else globals_override.to(DTYPE)
    S = u.shape[0]
    day_sd = torch.exp(u[:, priors.names().index("log_day_si_sd")])
    drift = day_sd[:, None] * torch.randn(S, tl.n_days, generator=gen, dtype=DTYPE)
    fitted_drift = {d: (l, sd) for d, l, sd in zip(fit.fitted_days, fit.day_drift_loc, fit.day_drift_sd)}
    egp_sd = torch.exp(u[:, priors.names().index("log_day_egp_sd")])
    egp_drift = egp_sd[:, None] * torch.randn(S, tl.n_days, generator=gen, dtype=DTYPE)
    fitted_egp = {d: (l, sd) for d, l, sd in zip(fit.fitted_days, fit.day_egp_drift_loc, fit.day_egp_drift_sd)}
    if globals_override is None:
        for di, date in enumerate(tl.day_dates):
            if date in fitted_drift:
                l, sd = fitted_drift[date]
                drift[:, di] = l + sd * torch.randn(S, generator=gen, dtype=DTYPE)
            if date in fitted_egp:
                l, sd = fitted_egp[date]
                egp_drift[:, di] = l + sd * torch.randn(S, generator=gen, dtype=DTYPE)
    meals = _meal_draws(fit, tl, S, gen)
    flux = _fitted_flux(fit, tl, start, n_steps)

    A = len(arms)
    rep = lambda t: t.repeat(A, *([1] * (t.dim() - 1)))
    ctrl = BasalBolusController(
        tl,
        cr_mult=torch.tensor([a.cr_mult for a in arms], dtype=DTYPE).repeat_interleave(S),
        isf_mult=torch.tensor([a.isf_mult for a in arms], dtype=DTYPE).repeat_interleave(S),
        basal_mult=torch.tensor([a.basal_mult for a in arms], dtype=DTYPE).repeat_interleave(S),
        aid=aid,
    )
    with torch.no_grad():
        glucose = rollout(
            tl, fit.base, priors, rep(u), start, n_steps,
            meal_grams=[(step, rep(g)) for step, g in meals],
            day_log_si=rep(drift), day_log_egp=rep(egp_drift), controller=ctrl, feats=feats,
            flux_upm=None if flux is None else flux.expand(A * S, -1),
        )
    scored = glucose[:, s0 - start:].view(A, S, -1)

    out = {"person_id": fit.person_id, "base": fit.base, "days": tl.day_dates[d0:d1], "samples": S,
           "controller": "generic basal-bolus" + (" + generic AID basal modulation" if aid else ""), "arms": {}}
    base_metrics = None
    for ai, arm in enumerate(arms):
        mets = glycemic_metrics(scored[ai])
        if base_metrics is None:
            base_metrics = mets
        out["arms"][arm.name] = {
            "settings": {"cr_mult": arm.cr_mult, "isf_mult": arm.isf_mult, "basal_mult": arm.basal_mult},
            "metrics": {k: _q(v) for k, v in mets.items()},
            "paired_delta_vs_first_arm": {k: _q(v - base_metrics[k]) for k, v in mets.items()},
        }
    return out


def _q(v: torch.Tensor) -> dict[str, float]:
    v = v.double()
    return {
        "median": float(v.median()),
        "p05": float(torch.quantile(v, 0.05)),
        "p95": float(torch.quantile(v, 0.95)),
    }


MEAL_BOLUS_WINDOW_MIN = 20.0


def split_boluses(tl: PersonTimeline) -> tuple[np.ndarray, np.ndarray]:
    """Recorded bolus insulin [U/min per step], split into meal boluses and everything else.

    A bolus within 20 minutes of a logged meal or a meal announcement (a
    carb-free bolus, a food photo) counts as a meal bolus; the rest are
    corrections.
    """
    from t1d_twin.data import detect_unlogged_meals

    width = int(MEAL_BOLUS_WINDOW_MIN / ode.DT_MIN)
    near_meal = np.zeros(tl.n_steps, bool)
    steps = [m.step for m in tl.meals if m.logged] + [a.step for a in detect_unlogged_meals(tl, use_cgm_rises=False)]
    for s in steps:
        near_meal[max(0, s - width): s + width + 1] = True
    meal = np.where(near_meal, tl.bolus_upm, 0.0)
    return meal, tl.bolus_upm - meal


def run_delivery_experiment(
    fit: TwinFit,
    tl: PersonTimeline,
    arms: list[Arm],
    *,
    samples: int = 32,
    days: tuple[int, int] | None = None,
    seed: int = 0,
) -> dict:
    """Settings arms applied to what was actually delivered, instead of to a controller.

    ``cr_mult`` scales every meal bolus by 1 / cr_mult (a lower carb ratio means more
    insulin per gram), ``isf_mult`` scales correction boluses by 1 / isf_mult, and
    ``basal_mult`` scales delivered basal. The first arm with all multipliers at 1
    replays the recorded day exactly, so the fitted disturbance stays matched to the
    insulin it was fitted under, and no settings are needed. For an open-loop pump
    this is the literal counterfactual; for a closed loop it leaves out how the
    algorithm would have reacted.
    """
    gen = torch.Generator().manual_seed(seed)
    priors = fit.priors()
    tl = shift_events(tl, fit.event_clock_offset_min)
    feats = build_features(tl)
    d0, d1 = days if days is not None else (1, tl.n_days)
    s0, _ = tl.day_steps(d0)
    _, s1 = tl.day_steps(d1 - 1)
    burn = int(BURN_IN_H * 60 / ode.DT_MIN)
    start = max(0, s0 - burn)
    n_steps = s1 - start

    u = fit.sample_globals(samples, gen)
    S = u.shape[0]
    drift = torch.exp(u[:, priors.names().index("log_day_si_sd")])[:, None] * torch.randn(S, tl.n_days, generator=gen, dtype=DTYPE)
    egp_drift = torch.exp(u[:, priors.names().index("log_day_egp_sd")])[:, None] * torch.randn(S, tl.n_days, generator=gen, dtype=DTYPE)
    for fitted_days, locs, sds, target in ((fit.fitted_days, fit.day_drift_loc, fit.day_drift_sd, drift),
                                           (fit.fitted_days, fit.day_egp_drift_loc, fit.day_egp_drift_sd, egp_drift)):
        for date, l, sd in zip(fitted_days, locs, sds):
            if date in tl.day_dates:
                target[:, tl.day_dates.index(date)] = l + sd * torch.randn(S, generator=gen, dtype=DTYPE)
    meals = _meal_draws(fit, tl, S, gen)
    flux = _fitted_flux(fit, tl, start, n_steps)

    meal_bolus, correction = split_boluses(tl)
    sl = slice(start, start + n_steps)
    basal = torch.tensor(np.nan_to_num(tl.basal_upm[sl]), dtype=DTYPE)
    meal_t, corr_t = torch.tensor(meal_bolus[sl], dtype=DTYPE), torch.tensor(correction[sl], dtype=DTYPE)
    A = len(arms)
    bolus = torch.stack([meal_t / a.cr_mult + corr_t / a.isf_mult for a in arms]).repeat_interleave(S, dim=0)
    insulin = torch.stack([basal * a.basal_mult for a in arms]).repeat_interleave(S, dim=0) + bolus
    rep = lambda t: t.repeat(A, *([1] * (t.dim() - 1)))
    with torch.no_grad():
        glucose = rollout(
            tl, fit.base, priors, rep(u), start, n_steps,
            meal_grams=[(step, rep(g)) for step, g in meals],
            day_log_si=rep(drift), day_log_egp=rep(egp_drift), insulin_upm=insulin, bolus_upm=bolus, feats=feats,
            flux_upm=None if flux is None else flux.expand(A * S, -1),
        )
    scored = glucose[:, s0 - start:].view(A, S, -1)

    delivered = lambda a: float((basal.sum() * a.basal_mult + (meal_t / a.cr_mult + corr_t / a.isf_mult).sum()) * ode.DT_MIN)
    out = {"person_id": fit.person_id, "base": fit.base, "days": tl.day_dates[d0:d1], "samples": S,
           "controller": "none: arms scale the recorded delivery",
           "meal_bolus_share": float(meal_t.sum() / max(float((meal_t + corr_t).sum()), 1e-9)), "arms": {}}
    base_metrics = None
    for ai, arm in enumerate(arms):
        mets = glycemic_metrics(scored[ai])
        if base_metrics is None:
            base_metrics = mets
        out["arms"][arm.name] = {
            "settings": {"cr_mult": arm.cr_mult, "isf_mult": arm.isf_mult, "basal_mult": arm.basal_mult},
            "insulin_delivered_u": delivered(arm),
            "metrics": {k: _q(v) for k, v in mets.items()},
            "paired_delta_vs_first_arm": {k: _q(v - base_metrics[k]) for k, v in mets.items()},
        }
    return out
