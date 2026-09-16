"""Fit a personal twin to a person's real data (variational Bayes).

Latents and their priors:

- global physiology + context + noise parameters (``params.GLOBAL_SPECS``);
- one log insulin-sensitivity drift per fitted day, sd itself fitted;
- one log carb-count error per logged meal (logged carbs are not trusted);
- one size per likely-unlogged meal (``data.detect_unlogged_meals``).

Two stages:

A. multi-start MAP with the CGM noise held at its prior mean. Several starts
   (SI x EGP x carb-absorption speed) run as one batch; the best log posterior
   wins. Holding the noise fixed matters: freed early, the fit inflates noise
   to excuse a poor start and the physiology gradients go flat.
B. stochastic variational inference from the MAP point: full-covariance
   Normal over global parameters, mean-field over per-day/per-meal latents,
   noise freed.

The output is an ensemble: downstream use samples from it rather than taking
a point estimate.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.context import build_features
from t1d_twin.data import PersonTimeline, detect_unlogged_meals, shift_events
from t1d_twin.dosing import MIN_MEALS as MIN_DOSED_MEALS, carb_ratio_logprior, carb_ratio_prior, observed_carb_ratio
from t1d_twin.model import DTYPE, FLUX_BLOCK_STEPS, TwinProblem, transform_globals
from t1d_twin.params import (
    FLUX_CLAMP,
    FLUX_PRIOR_DF,
    FLUX_PRIOR_SCALE,
    LOGGED_SHIFT_PRIOR_SD,
    MEAL_SCALE_PRIOR,
    UNLOGGED_LOG_G_PRIOR,
    UNLOGGED_SHIFT_PRIOR_SD,
    TwinPriors,
    twin_priors,
)

CGM_CV = 0.06                 # proportional part of CGM residual sd
UNCALIBRATED_BASE = "adult#001"
LIKELY_MEAL_GRAMS = 10.0


@dataclass
class FitConfig:
    map_iters: int = 250
    iters: int = 400
    particles: int = 4
    lr: float = 0.05
    svi_lr: float = 0.02
    seed: int = 0
    holdout_days: int = 0
    base: str = "auto"
    min_cgm_coverage: float = 0.7
    base_screen_days: int = 7
    log_every: int = 50
    # candidate clock offsets (min) of insulin/meal events relative to CGM; () disables
    clock_offsets: tuple[float, ...] = (-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0)
    # fit a signed glucose disturbance per 30-min block for what the records do not
    # explain; meal candidates then come only from announcements, not CGM rises
    flux: bool = True
    # learn personal carb/insulin response corrections (carry over to new days)
    response_kernels: bool = True
    # fit residuals in Kovatchev glycaemic risk space (errors at low glucose weigh more)
    risk_weighted: bool = False
    # centre the insulin-sensitivity prior, and break base-patient ties, so the twin's
    # balanced carb ratio matches the one the person doses at (t1d_twin.dosing)
    dosing_prior: bool = True
    # a base patient may be chosen over the best-fitting one when its RMSE is within
    # this fraction of the best, since screen RMSE barely separates them
    base_rmse_tolerance: float = 0.10


@dataclass
class TwinFit:
    person_id: str
    base: str
    param_names: list[str]
    prior_mean: list[float]
    prior_sd: list[float]
    loc: list[float]
    sd: list[float]
    fitted_days: list[str]
    day_drift_loc: list[float]
    day_drift_sd: list[float]
    logged_meal_steps: list[int]
    meal_scale_loc: list[float]
    meal_scale_sd: list[float]
    unlogged_meal_steps: list[int]
    unlogged_log_g_loc: list[float]
    unlogged_log_g_sd: list[float]
    meal_shift_min: list[float] = field(default_factory=list)       # posterior-mean time shift per logged meal
    unlogged_shift_min: list[float] = field(default_factory=list)   # per unlogged candidate
    day_flux: dict[str, list[float]] = field(default_factory=dict)
    day_egp_drift_loc: list[float] = field(default_factory=list)    # per fitted day, log EGP drift
    day_egp_drift_sd: list[float] = field(default_factory=list)   # date -> posterior-mean flux per 30-min block from local midnight
    scale_tril: list[list[float]] = field(default_factory=list)
    event_clock_offset_min: float = 0.0
    carb_count_bias: float = 0.0        # mean fitted log carb-count error of logged meals (under-counting > 0)
    recentered: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def priors(self) -> TwinPriors:
        p = TwinPriors()
        for n, m, s in zip(self.param_names, self.prior_mean, self.prior_sd):
            p = p.with_prior(n, m, s)
        return p

    def sample_globals(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        loc = torch.tensor(self.loc, dtype=DTYPE)
        eps = torch.randn(n, loc.numel(), generator=generator, dtype=DTYPE)
        if self.scale_tril:
            return loc + eps @ torch.tensor(self.scale_tril, dtype=DTYPE).T
        return loc + torch.tensor(self.sd, dtype=DTYPE) * eps

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TwinFit":
        return recenter_fit(upgrade_fit(cls(**json.loads(Path(path).read_text()))))


def _normal_logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)


def masked_rmse(sim: torch.Tensor, prob: TwinProblem) -> float:
    err = (sim - prob.cgm_t) ** 2 * prob.mask_t
    return float(torch.sqrt(err.sum() / prob.mask_t.sum()))


def local_prior_mean(prob: TwinProblem) -> torch.Tensor:
    """Per-day/per-meal latents: [SI drift D | EGP drift D | meal scale M | unlogged log g U | logged shift M | unlogged shift U | flux F]."""
    D, M, U, F = prob.n_days, prob.n_logged, prob.n_unlogged, prob.n_flux
    return torch.cat([torch.zeros(2 * D + M, dtype=DTYPE), torch.full((U,), UNLOGGED_LOG_G_PRIOR[0], dtype=DTYPE), torch.zeros(M + U + F, dtype=DTYPE)])


def split_locals(z: torch.Tensor, prob: TwinProblem) -> tuple[torch.Tensor, ...]:
    D, M, U, F = prob.n_days, prob.n_logged, prob.n_unlogged, prob.n_flux
    drift, egp_drift, ms, ul, sl, su, fx = torch.split(z, [D, D, M, U, M, U, F], dim=1)
    # simulate() argument order: SI drift, meal scale, unlogged g, shifts, flux, EGP drift
    return drift, ms, ul, sl, su, fx, egp_drift


def _student_t_logpdf(x, scale: float, df: float):
    return -0.5 * (df + 1.0) * torch.log1p((x / scale) ** 2 / df) - math.log(scale)


def prior_latents(prob: TwinProblem, K: int = 1) -> tuple[torch.Tensor, ...]:
    """Globals and locals at their prior means, as simulate() arguments."""
    u = prob.priors.means().to(DTYPE).expand(K, -1).clone()
    return (u, *split_locals(local_prior_mean(prob).expand(K, -1).clone(), prob))


SCREEN_GRID = [(si, egp, carb) for si in (-0.7, -0.35, 0.0, 0.35, 0.7) for egp in (-0.3, 0.0, 0.3) for carb in (-0.5, 0.0, 0.5)]
OFFSET_INSULIN_SPEEDS = (-0.35, 0.0, 0.35)
OFFSET_MIN_GAIN = 0.05   # a non-zero clock offset must cut screen RMSE by at least 5%


def screen_rmse(tl: PersonTimeline, base: str, days: list[int], priors: TwinPriors, feats, unlogged, insulin_speeds=(0.0,), min_cgm_coverage: float = 0.7) -> float:
    """Best RMSE over a coarse SI x EGP x carb-speed (x insulin-speed) grid (forward only)."""
    names = priors.names()
    prob = TwinProblem(tl, base, days, priors, feats=feats, unlogged=unlogged, min_cgm_coverage=min_cgm_coverage)
    grid = [(si, egp, carb, ins) for si, egp, carb in SCREEN_GRID for ins in insulin_speeds]
    u, *locs = prior_latents(prob, len(grid))
    for k, (si, egp, carb, ins) in enumerate(grid):
        u[k, names.index("log_si")], u[k, names.index("log_egp")], u[k, names.index("log_carb_speed")] = si, egp, carb
        u[k, names.index("log_insulin_speed")] = ins
    with torch.no_grad():
        sim = prob.simulate(u, *locs)
    err = torch.nan_to_num(((sim - prob.cgm_t) ** 2 * prob.mask_t).sum(dim=(1, 2)) / prob.mask_t.sum(), nan=1e9)
    return float(torch.sqrt(err.min()))


def select_clock_offset(tl: PersonTimeline, base: str, days: list[int], priors: TwinPriors, offsets, min_cgm_coverage: float = 0.7) -> tuple[float, dict[str, float]]:
    """Shift insulin/meal events against CGM; keep the offset the twin explains best.

    Absorption speed can mimic a clock offset (fast insulin looks like early
    boluses), so the screen also varies insulin speed, and a non-zero offset
    must beat no offset by ``OFFSET_MIN_GAIN``.
    """
    scores = {}
    for off in offsets:
        shifted = shift_events(tl, off)
        try:
            scores[off] = screen_rmse(shifted, base, days, priors, build_features(shifted), detect_unlogged_meals(shifted, use_cgm_rises=False),
                                      insulin_speeds=OFFSET_INSULIN_SPEEDS, min_cgm_coverage=min_cgm_coverage)
        except ValueError:
            continue
    best = min(scores, key=scores.get)
    if 0.0 in scores and best != 0.0 and scores[best] > (1.0 - OFFSET_MIN_GAIN) * scores[0.0]:
        best = 0.0
    return best, {f"{k:+.0f}": v for k, v in scores.items()}


def select_base(tl: PersonTimeline, days: list[int], priors: TwinPriors, feats, unlogged, min_cgm_coverage: float = 0.7,
                target_carb_ratio: float = float("nan"), rmse_tolerance: float = 0.0) -> tuple[str, dict[str, float], tuple[float, float]]:
    """Pick the UVA/Padova adult whose dynamics best fit, over a coarse SI x EGP grid.

    Screen RMSE separates the adults poorly (often under 2% between the best and
    a patient three times less insulin sensitive), while their dose response
    differs a lot. With ``target_carb_ratio`` (the grams per unit this person
    doses at), any adult within ``rmse_tolerance`` of the best is eligible and
    the one whose own carb ratio is closest to theirs wins.

    Also returns the best (log_si, log_egp) grid point, used to start the fit.
    """
    si_grid = [-0.7, -0.35, 0.0, 0.35, 0.7]
    egp_grid = [-0.3, 0.0, 0.3]
    i_si, i_egp = priors.names().index("log_si"), priors.names().index("log_egp")
    scores: dict[str, float] = {}
    best_point: dict[str, tuple[float, float]] = {}
    grid = [(a, b) for a in si_grid for b in egp_grid]
    for name in ode.ADULTS:
        prob = TwinProblem(tl, name, days, priors, feats=feats, unlogged=unlogged, min_cgm_coverage=min_cgm_coverage)
        K = len(si_grid) * len(egp_grid)
        u, *locs = prior_latents(prob, K)
        for k, (a, b) in enumerate(grid):
            u[k, i_si], u[k, i_egp] = a, b
        with torch.no_grad():
            sim = prob.simulate(u, *locs)
        err = ((sim - prob.cgm_t) ** 2 * prob.mask_t).sum(dim=(1, 2)) / prob.mask_t.sum()
        err = torch.nan_to_num(err, nan=1e9)
        scores[name] = float(torch.sqrt(err.min()))
        best_point[name] = grid[int(err.argmin())]
    best = min(scores, key=scores.get)
    if np.isfinite(target_carb_ratio) and target_carb_ratio > 0 and rmse_tolerance > 0:
        from t1d_twin.dosing import base_carb_ratios

        ratios = base_carb_ratios()
        eligible = [n for n, v in scores.items() if v <= scores[best] * (1.0 + rmse_tolerance)]
        best = min(eligible, key=lambda n: abs(math.log(ratios[n] / target_carb_ratio)) if ratios.get(n, 0) > 0 else 1e9)
    return best, scores, best_point[best]


GRAD_CLIP = 10.0
MAP_STARTS = [(si, egp, carb) for si in (-0.5, 0.0, 0.5) for egp in (-0.25, 0.25) for carb in (-0.5, 0.5)]


def risk_space(g: torch.Tensor) -> torch.Tensor:
    """Kovatchev et al. (1997) symmetrising transform of blood glucose (mg/dL)."""
    return 1.509 * (torch.log(torch.clamp(g, min=20.0)) ** 1.084 - 5.381)


RISK_REF_MGDL = 120.0
RISK_SLOPE_AT_REF = 1.509 * 1.084 * math.log(RISK_REF_MGDL) ** 0.084 / RISK_REF_MGDL  # df/dG at 120 mg/dL


def log_joint(prob: TwinProblem, u, z_locals) -> tuple[torch.Tensor, torch.Tensor]:
    """Log likelihood + log prior per row [K], and the simulation."""
    priors = prob.priors
    names = priors.names()
    drift, ms, ul, sl, su, fx, egp_drift = split_locals(z_locals, prob)
    sim = prob.simulate(u, drift, ms, ul, sl, su, fx, egp_drift)
    cgm_sd = torch.exp(u[:, names.index("log_cgm_sd")])[:, None, None]
    if getattr(prob, "risk_weighted", False):
        # constant noise in risk space, matched to CGM noise at 120 mg/dL: low-range errors
        # get more weight, very high readings less (not a re-parameterisation, on purpose)
        sigma_f = torch.sqrt(cgm_sd ** 2 + (CGM_CV * RISK_REF_MGDL) ** 2) * RISK_SLOPE_AT_REF
        obs = torch.where(prob.mask_t, prob.cgm_t, torch.full_like(prob.cgm_t, RISK_REF_MGDL))
        loglik = (_normal_logpdf(risk_space(obs), risk_space(sim), sigma_f) * prob.mask_t).sum(dim=(1, 2))
    else:
        sigma = torch.sqrt(cgm_sd ** 2 + (CGM_CV * prob.cgm_t) ** 2)
        loglik = (_normal_logpdf(prob.cgm_t, sim, sigma) * prob.mask_t).sum(dim=(1, 2))
    logprior = _normal_logpdf(u, priors.means().to(DTYPE), priors.sds().to(DTYPE)).sum(1)
    logprior = logprior + _normal_logpdf(drift, torch.zeros(()), torch.exp(u[:, names.index("log_day_si_sd")])[:, None]).sum(1)
    logprior = logprior + _normal_logpdf(egp_drift, torch.zeros(()), torch.exp(u[:, names.index("log_day_egp_sd")])[:, None]).sum(1)
    logprior = logprior + _normal_logpdf(ms, torch.tensor(MEAL_SCALE_PRIOR[0]), torch.tensor(MEAL_SCALE_PRIOR[1])).sum(1)
    logprior = logprior + _normal_logpdf(ul, torch.tensor(UNLOGGED_LOG_G_PRIOR[0]), torch.tensor(UNLOGGED_LOG_G_PRIOR[1])).sum(1)
    logprior = logprior + _normal_logpdf(sl, torch.zeros(()), torch.tensor(LOGGED_SHIFT_PRIOR_SD)).sum(1)
    logprior = logprior + _normal_logpdf(su, torch.zeros(()), torch.tensor(UNLOGGED_SHIFT_PRIOR_SD)).sum(1)
    logprior = logprior + _student_t_logpdf(fx, FLUX_PRIOR_SCALE, FLUX_PRIOR_DF).sum(1)
    target = getattr(prob, "target_carb_ratio", float("nan"))
    if np.isfinite(target):
        # the CGM pins the net effect of a meal and its bolus, not each one's size,
        # and the disturbance flux absorbs what is left, so the person's own dosing
        # sets the ratio (t1d_twin.dosing)
        logprior = logprior + carb_ratio_logprior(transform_globals(u, priors), prob.base.name, prob.tl.body_mass_kg, target)
    return loglik + logprior, sim


def fit_map(prob: TwinProblem, config: FitConfig, *, verbose: bool = True, t_start: float | None = None):
    """Stage A: batched multi-start MAP with CGM noise fixed. Returns (globals [G], locals, per-start rmse)."""
    priors = prob.priors
    names = priors.names()
    t_start = t_start or time.time()
    G, D, M, U = len(names), prob.n_days, prob.n_logged, prob.n_unlogged
    S = len(MAP_STARTS)
    # Optimise globals in prior-standardised units z = (u - mean) / sd: Adam's
    # step is ~lr in parameter units, so raw units would move a 0.01-scale knob
    # (a per-gram response coefficient) by several prior sds per step.
    mean, sd = priors.means().to(DTYPE), priors.sds().to(DTYPE)
    g = priors.means().to(DTYPE).repeat(S, 1)
    for k, (si, egp, carb) in enumerate(MAP_STARTS):
        g[k, names.index("log_si")], g[k, names.index("log_egp")], g[k, names.index("log_carb_speed")] = si, egp, carb
    z = ((g - mean) / sd).requires_grad_(True)
    loc = local_prior_mean(prob).repeat(S, 1).requires_grad_(True)
    noise_mask = torch.ones(G, dtype=DTYPE)
    noise_mask[names.index("log_cgm_sd")] = 0.0
    noise_value = torch.zeros(G, dtype=DTYPE)
    noise_value[names.index("log_cgm_sd")] = priors.get("log_cgm_sd").prior_mean
    opt = torch.optim.Adam([z, loc], lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.2 ** (it / max(1, config.map_iters)))
    n_obs = float(prob.mask_t.sum())
    lp = torch.full((S,), -float("inf"))
    for it in range(config.map_iters):
        opt.zero_grad()
        u = (mean + sd * z) * noise_mask + noise_value
        lp, sim = log_joint(prob, u, loc)
        loss = -torch.nan_to_num(lp, nan=-1e12).sum() / n_obs
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z, loc], GRAD_CLIP)
        opt.step()
        sched.step()
        if verbose and (it % config.log_every == 0 or it == config.map_iters - 1):
            rmse = torch.sqrt(((sim.detach() - prob.cgm_t) ** 2 * prob.mask_t).sum(dim=(1, 2)) / n_obs)
            print(f"[twin] map it={it:4d} best rmse={float(rmse.min()):6.1f} mg/dL  {time.time() - t_start:6.0f}s")
    with torch.no_grad():
        best = int(torch.nan_to_num(lp.detach(), nan=-float("inf")).argmax())
        u = ((mean + sd * z) * noise_mask + noise_value)[best].detach()
        rmse = torch.sqrt(((sim.detach() - prob.cgm_t) ** 2 * prob.mask_t).sum(dim=(1, 2)) / n_obs)
    return u, loc[best].detach(), [float(v) for v in rmse]


def fit_twin(
    tl: PersonTimeline,
    priors: TwinPriors | None = None,
    config: FitConfig | None = None,
    *,
    verbose: bool = True,
) -> TwinFit:
    config = config or FitConfig()
    priors = priors or twin_priors(cycle_observed=bool(np.any(~np.isnan(tl.days_since_period))), sex=tl.sex)
    if not config.response_kernels:
        from t1d_twin.params import KERNELS, PINNED_SD
        for spec in KERNELS:
            priors = priors.with_prior(spec.name, 0.0, PINNED_SD)
    torch.manual_seed(config.seed)
    feats = build_features(tl)
    candidates = lambda t: detect_unlogged_meals(t, use_cgm_rises=not config.flux)
    unlogged = candidates(tl)

    all_days = list(range(tl.n_days))
    train_days = all_days[: tl.n_days - config.holdout_days] if config.holdout_days else all_days
    holdout_days = all_days[tl.n_days - config.holdout_days:] if config.holdout_days else []

    t_start = time.time()
    # screen on the first fittable days, not the first calendar days
    usable = TwinProblem(tl, UNCALIBRATED_BASE, train_days, priors, feats=feats, unlogged=[], min_cgm_coverage=config.min_cgm_coverage).days
    screen = usable[: config.base_screen_days]
    target_ratio, n_dosed_meals = observed_carb_ratio(tl, {s for d in usable for s in range(*tl.day_steps(d))}) if config.dosing_prior else (float("nan"), 0)
    if n_dosed_meals < MIN_DOSED_MEALS:
        target_ratio = float("nan")  # too few bolused meals to read a ratio off
    pick_base = lambda: select_base(tl, screen, priors, feats, unlogged, config.min_cgm_coverage,
                                    target_ratio, config.base_rmse_tolerance)
    if config.base == "auto":
        base, base_scores, _ = pick_base()
    else:
        base, base_scores = config.base, {}
    offset, offset_scores = 0.0, {}
    if config.clock_offsets:
        offset, offset_scores = select_clock_offset(tl, base, screen, priors, config.clock_offsets, config.min_cgm_coverage)
        if offset != 0.0:
            tl = shift_events(tl, offset)
            feats, unlogged = build_features(tl), candidates(tl)
            if config.base == "auto":
                base, base_scores, _ = pick_base()
        if verbose:
            print(f"[twin] event clock offset {offset:+.0f} min (screen rmse by offset: "
                  + ", ".join(f"{k}:{v:.1f}" for k, v in offset_scores.items()) + ")")
    dosing_info = {"observed_carb_ratio_g_per_u": target_ratio, "n_bolused_meals": n_dosed_meals, "applied": False}
    if np.isfinite(target_ratio):
        priors, dosing_info = carb_ratio_prior(priors, base, target_ratio)
        dosing_info.update(observed_carb_ratio_g_per_u=target_ratio, n_bolused_meals=n_dosed_meals)
        if verbose:
            print(f"[twin] doses {target_ratio:.1f} g/U over {n_dosed_meals} meals; {base} is {dosing_info['base_carb_ratio_g_per_u']:.1f} g/U, "
                  f"log_si prior re-centred to {dosing_info['log_si_prior_mean']:+.2f} (twin now {dosing_info['reached_carb_ratio_g_per_u']:.1f} g/U)")
    prob = TwinProblem(tl, base, train_days, priors, feats=feats, unlogged=unlogged, min_cgm_coverage=config.min_cgm_coverage, flux=config.flux)
    prob.risk_weighted = config.risk_weighted
    prob.target_carb_ratio = target_ratio
    if verbose:
        print(f"[twin] {tl.person_id}: base={base} fitted_days={prob.n_days} skipped={len(prob.skipped)} "
              f"logged_meals={prob.n_logged} likely_unlogged={prob.n_unlogged} obs={int(prob.mask_t.sum())}")

    G, D, M, U = len(priors.specs), prob.n_days, prob.n_logged, prob.n_unlogged
    map_globals, map_locals, map_rmse = fit_map(prob, config, verbose=verbose, t_start=t_start)

    # Stage B guide: full-covariance Normal over the global parameters (SI, EGP
    # and absorption speeds trade off against each other; mean-field would hide
    # that and report false certainty), mean-field over per-day/per-meal latents.
    # (in prior-standardised units, see fit_map)
    g_mean, g_sd = priors.means().to(DTYPE), priors.sds().to(DTYPE)
    g_loc = ((map_globals - g_mean) / g_sd).clone().requires_grad_(True)
    g_log_diag = torch.full((G,), math.log(0.05), dtype=DTYPE, requires_grad=True)
    g_offdiag = torch.zeros(G, G, dtype=DTYPE, requires_grad=True)
    l_loc = map_locals.clone().requires_grad_(True)
    L_loc = l_loc.numel()
    l_log_sd = torch.full((L_loc,), math.log(0.02), dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([g_loc, g_log_diag, g_offdiag, l_loc, l_log_sd], lr=config.svi_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.1 ** (it / max(1, config.iters)))
    tril_mask = torch.tril(torch.ones(G, G, dtype=torch.bool), diagonal=-1)

    def scale_tril():
        return torch.diag(torch.exp(g_log_diag)) + g_offdiag * tril_mask

    obs_mask = prob.mask_t
    n_obs = float(obs_mask.sum())
    K = config.particles
    history = []


    for it in range(config.iters):
        opt.zero_grad()
        u = g_mean + g_sd * (g_loc + torch.randn(K, G, dtype=DTYPE) @ scale_tril().T)
        lp, sim = log_joint(prob, u, l_loc + torch.exp(l_log_sd) * torch.randn(K, L_loc, dtype=DTYPE))
        elbo = lp.mean() + g_log_diag.sum() + l_log_sd.sum()
        loss = -elbo / n_obs
        if not torch.isfinite(loss):
            history.append(float("nan"))
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([g_loc, g_log_diag, g_offdiag, l_loc, l_log_sd], GRAD_CLIP)
        opt.step()
        sched.step()
        with torch.no_grad():
            g_log_diag.clamp_(max=math.log(5.0))
            l_log_sd.clamp_(max=math.log(5.0))
        history.append(loss.item())
        if verbose and (it % config.log_every == 0 or it == config.iters - 1):
            rmse = float(torch.sqrt(((sim.detach() - prob.cgm_t) ** 2 * obs_mask).sum() / (K * n_obs)))
            print(f"[twin] svi it={it:4d} loss={float(loss):8.3f} rmse={rmse:6.1f} mg/dL  {time.time() - t_start:6.0f}s")

    with torch.no_grad():
        L = torch.diag(g_sd) @ scale_tril()   # back to parameter units
        u_l, u_s = (g_mean + g_sd * g_loc.detach())[None], torch.sqrt((L @ L.T).diagonal())[None]
        d_l, m_l, ul_l, sl_l, su_l, fx_l, e_l = split_locals(l_loc.detach()[None], prob)
        d_s, m_s, ul_s, _, _, _, e_s = split_locals(torch.exp(l_log_sd.detach())[None], prob)

    with torch.no_grad():
        train_rmse = masked_rmse(prob.simulate(u_l, d_l, m_l, ul_l, sl_l, su_l, fx_l, e_l), prob)
        burn_blocks = int(prob.windows[0].loss_from // FLUX_BLOCK_STEPS)
        nb = prob.n_flux_blocks_per_window
        day_flux = {}
        if prob.n_flux:
            blocks = fx_l[0].clamp(-FLUX_CLAMP, FLUX_CLAMP).view(prob.n_days, nb)
            day_flux = {tl.day_dates[w.day]: blocks[wi, burn_blocks:].tolist() for wi, w in enumerate(prob.windows)}
        from t1d_twin.model import meal_time_shift
        shift_logged, shift_unlogged = meal_time_shift(sl_l[0]).tolist(), meal_time_shift(su_l[0]).tolist()
        unc = TwinProblem(tl, UNCALIBRATED_BASE, train_days, priors, feats=feats, unlogged=[], min_cgm_coverage=config.min_cgm_coverage)
        uncal_rmse = masked_rmse(unc.simulate(*prior_latents(unc)), unc)

    fit = TwinFit(
        person_id=tl.person_id,
        base=base,
        param_names=priors.names(),
        prior_mean=[float(v) for v in g_mean],
        prior_sd=[float(v) for v in g_sd],
        loc=[float(v) for v in u_l[0]],
        sd=[float(v) for v in u_s[0]],
        fitted_days=[tl.day_dates[d] for d in prob.days],
        day_drift_loc=[float(v) for v in d_l[0]],
        day_drift_sd=[float(v) for v in d_s[0]],
        day_egp_drift_loc=[float(v) for v in e_l[0]],
        day_egp_drift_sd=[float(v) for v in e_s[0]],
        logged_meal_steps=[tl.meals[i].step for i in prob.logged_ids],
        meal_scale_loc=[float(v) for v in m_l[0]],
        meal_scale_sd=[float(v) for v in m_s[0]],
        unlogged_meal_steps=[prob.unlogged_all[i].step for i in prob.unlogged_ids],
        unlogged_log_g_loc=[float(v) for v in ul_l[0]],
        unlogged_log_g_sd=[float(v) for v in ul_s[0]],
        meal_shift_min=shift_logged,
        day_flux=day_flux,
        unlogged_shift_min=shift_unlogged,
        scale_tril=L.tolist(),
        event_clock_offset_min=offset,
    )
    recenter_fit(fit)
    fit.summary = summarise(fit, tl, priors)
    fit.diagnostics = {
        "base_screen_rmse": base_scores,
        "map_start_rmse_mgdl": map_rmse,
        "clock_offset_screen_rmse": offset_scores,
        "train_rmse_mgdl": train_rmse,
        "train_rmse_uncalibrated_mgdl": uncal_rmse,
        "n_obs": int(n_obs),
        "skipped_days": {tl.day_dates[d]: why for d, why in prob.skipped.items()},
        "context_known": feats.known,
        "timeline_notes": tl.notes,
        "loss_first_last": [history[0] if history else None, history[-1] if history else None],
        "fit_seconds": round(time.time() - t_start, 1),
        "config": asdict(config),
        "dosing": dosing_info,
    }
    if holdout_days:
        fit.diagnostics["holdout"] = evaluate_holdout(fit, tl, holdout_days, feats=feats, min_cgm_coverage=config.min_cgm_coverage)
    return fit


# How a parameter a fit predates behaves: the model it was fitted with.
OFF_VALUES = {"log_hypo_uptake": -30.0}  # r1 = exp(-30) ~ 0: the 2008 model


# Knobs real data consistently move: their population prior is re-estimated from other people's fits.
EMPIRICAL_KNOBS = ("log_si", "log_egp", "log_insulin_speed", "log_insulin_action_speed", "log_carb_speed", "log_carb_effect",
                   "exercise_uptake", "dawn_egp", "log_cgm_sd", "log_day_si_sd", "log_day_egp_sd")
EMPIRICAL_MIN_FITS = 8


def empirical_priors(fits: list[TwinFit], exclude_person: str | None = None, priors: TwinPriors | None = None) -> TwinPriors:
    """Population priors re-centred on other people's fitted values (empirical Bayes, leave-one-out).

    For each knob in ``EMPIRICAL_KNOBS``: mean = mean of the other fits' posterior
    means; sd = their spread, kept within [0.5, 1] x the hand-set sd (a couple of
    dozen people should not make the prior tighter than half, and the hand-set
    sd already bounds how far a person may move). Knobs are multipliers on each
    fit's matched base adult, so pooling across bases is an approximation.
    """
    priors = priors or TwinPriors()
    others = [f for f in fits if f.person_id != exclude_person]
    if len(others) < EMPIRICAL_MIN_FITS:
        raise ValueError(f"empirical priors need at least {EMPIRICAL_MIN_FITS} other fits, got {len(others)}")
    for name in EMPIRICAL_KNOBS:
        values = np.array([f.loc[f.param_names.index(name)] for f in others if name in f.param_names])
        spec = priors.get(name)
        sd = float(np.clip(values.std(ddof=1), 0.5 * spec.prior_sd, spec.prior_sd))
        priors = priors.with_prior(name, float(values.mean()), sd)
    return priors


def upgrade_fit(fit: TwinFit) -> TwinFit:
    """Add parameters introduced after a fit was saved, fixed at the value the fit implied."""
    from t1d_twin.params import GLOBAL_SPECS

    missing = [spec for spec in GLOBAL_SPECS if spec.name not in fit.param_names]
    if not missing:
        return fit
    tiny = 1e-6
    names, loc, sd, pm, ps = list(fit.param_names), list(fit.loc), list(fit.sd), list(fit.prior_mean), list(fit.prior_sd)
    L = np.asarray(fit.scale_tril) if fit.scale_tril else np.diag(sd)
    for spec in missing:
        i = [s.name for s in GLOBAL_SPECS].index(spec.name)
        i = min(i, len(names))
        names.insert(i, spec.name)
        loc.insert(i, OFF_VALUES.get(spec.name, spec.prior_mean))
        sd.insert(i, tiny)
        pm.insert(i, spec.prior_mean)
        ps.insert(i, spec.prior_sd)
        L = np.insert(np.insert(L, i, 0.0, axis=0), i, 0.0, axis=1)
        L[i, i] = tiny
    fit.param_names, fit.loc, fit.sd, fit.prior_mean, fit.prior_sd, fit.scale_tril = names, loc, sd, pm, ps, L.tolist()
    return fit


def recenter_fit(fit: TwinFit) -> TwinFit:
    """Fold the mean daily drifts into the core parameters and record the carb-count bias.

    SI multiplier = exp(day SI drift) * exp(log_si), likewise for EGP, so moving
    the mean drift into ``log_si`` / ``log_egp`` leaves every fitted day's
    simulation unchanged. Without it the shrunk core values describe a person
    more insulin-sensitive (and with lower EGP) than they are on average, and
    anything run without the per-day drifts (new days, synthetic people) is
    optimistic. Idempotent.
    """
    if fit.recentered:
        return fit
    for name, drifts in (("log_si", fit.day_drift_loc), ("log_egp", fit.day_egp_drift_loc)):
        if drifts and name in fit.param_names:
            mean = float(np.mean(drifts))
            fit.loc[fit.param_names.index(name)] += mean
            drifts[:] = [d - mean for d in drifts]
    fit.carb_count_bias = float(np.mean(fit.meal_scale_loc)) if fit.meal_scale_loc else 0.0
    fit.recentered = True
    return fit


def summarise(fit: TwinFit, tl: PersonTimeline, priors: TwinPriors) -> dict[str, Any]:
    """Human-readable posterior: model-unit intervals and how much data moved each knob."""
    params = {}
    for i, spec in enumerate(priors.specs):
        loc, sd = fit.loc[i], fit.sd[i]
        q = torch.tensor([loc - 1.645 * sd, loc, loc + 1.645 * sd], dtype=torch.float64)
        vals = spec.transform(q).tolist()
        shrink = sd / spec.prior_sd
        params[spec.name] = {
            "median": vals[1], "p05": vals[0], "p95": vals[2],
            "posterior_sd_over_prior_sd": round(shrink, 3),
            "identified": "learned" if shrink < 0.5 else "partly" if shrink < 0.8 else "prior-dominated",
            "description": spec.description,
        }

    def when(step):
        return (tl.t0_utc + timedelta(minutes=ode.DT_MIN * step)).isoformat()

    meals = []
    for step, loc, sd in zip(fit.logged_meal_steps, fit.meal_scale_loc, fit.meal_scale_sd):
        logged = next(m.grams for m in tl.meals if m.step == step)
        meals.append({
            "time_utc": when(step), "logged_g": logged,
            "fitted_g": logged * math.exp(loc),
            "fitted_g_p05": logged * math.exp(loc - 1.645 * sd), "fitted_g_p95": logged * math.exp(loc + 1.645 * sd),
        })
    likely = []
    for step, loc, sd in zip(fit.unlogged_meal_steps, fit.unlogged_log_g_loc, fit.unlogged_log_g_sd):
        p_meal = 0.5 * (1 - math.erf((math.log(LIKELY_MEAL_GRAMS) - loc) / (sd * math.sqrt(2))))
        likely.append({
            "time_utc": when(step), "fitted_g": math.exp(loc),
            "fitted_g_p05": math.exp(loc - 1.645 * sd), "fitted_g_p95": math.exp(loc + 1.645 * sd),
            "p_over_10g": round(p_meal, 3), "likely_meal": p_meal > 0.8,
        })
    return {"params": params, "logged_meals": meals, "unlogged_meal_candidates": likely}


def evaluate_holdout(fit: TwinFit, tl: PersonTimeline, days: list[int], *, feats=None, samples: int = 32, min_cgm_coverage: float = 0.7) -> dict[str, Any]:
    """Replay held-out days with posterior samples and recorded inputs only.

    Nothing may read the held-out CGM, so per-meal quantities come from their
    priors, not a fit: logged carbs get a count error and a time shift drawn
    from their priors, announcement-anchored candidates (carb-free boluses,
    food photos; no CGM) get sizes from theirs, the disturbance flux is drawn
    from its prior, and daily drift from its fitted spread. That uncertainty
    belongs in a replay interval.
    """
    priors = fit.priors()
    anchors = detect_unlogged_meals(tl, use_cgm_rises=False)
    try:
        prob = TwinProblem(tl, fit.base, days, priors, feats=feats, unlogged=anchors, min_cgm_coverage=min_cgm_coverage, flux=bool(fit.day_flux))
    except ValueError as exc:
        return {"error": str(exc)}
    gen = torch.Generator().manual_seed(1)
    u = fit.sample_globals(samples, gen)
    day_sd = torch.exp(u[:, priors.names().index("log_day_si_sd")])
    drift = day_sd[:, None] * torch.randn(samples, prob.n_days, generator=gen, dtype=DTYPE)
    egp_sd = torch.exp(u[:, priors.names().index("log_day_egp_sd")])
    egp_drift = egp_sd[:, None] * torch.randn(samples, prob.n_days, generator=gen, dtype=DTYPE)
    M, U = prob.n_logged, prob.n_unlogged
    draw = lambda n, mean, sd: mean + sd * torch.randn(samples, n, generator=gen, dtype=DTYPE)
    with torch.no_grad():
        sim = prob.simulate(
            u, drift,
            draw(M, *MEAL_SCALE_PRIOR), draw(U, *UNLOGGED_LOG_G_PRIOR),
            draw(M, 0.0, LOGGED_SHIFT_PRIOR_SD), draw(U, 0.0, UNLOGGED_SHIFT_PRIOR_SD),
            (FLUX_PRIOR_SCALE * torch.distributions.StudentT(FLUX_PRIOR_DF).sample((samples, prob.n_flux))).clamp(-FLUX_CLAMP, FLUX_CLAMP).to(DTYPE),
            egp_drift,
        )
        unc = TwinProblem(tl, UNCALIBRATED_BASE, days, priors, feats=feats, unlogged=[], min_cgm_coverage=min_cgm_coverage)
        uncal = unc.simulate(*prior_latents(unc))
    median = sim.median(dim=0).values
    # predictive interval includes CGM residual noise, not just parameter spread
    cgm_sd = torch.exp(u[:, priors.names().index("log_cgm_sd")])[:, None, None]
    noisy = sim + torch.sqrt(cgm_sd ** 2 + (CGM_CV * sim) ** 2) * torch.randn(sim.shape, generator=gen, dtype=DTYPE)
    lo, hi = torch.quantile(noisy, 0.05, dim=0), torch.quantile(noisy, 0.95, dim=0)
    m = prob.mask_t
    inside = ((prob.cgm_t >= lo) & (prob.cgm_t <= hi) & m).sum() / m.sum()
    return {
        "days": [tl.day_dates[d] for d in prob.days],
        "skipped": {tl.day_dates[d]: why for d, why in prob.skipped.items()},
        "replay_rmse_mgdl": masked_rmse(median[None], prob),
        "uncalibrated_replay_rmse_mgdl": masked_rmse(uncal, unc),
        "interval90_coverage": float(inside),
    }
