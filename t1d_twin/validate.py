"""Does the twin reproduce a person's glycemic outcomes, day by day?

Compares time in range (70-180), below range (<70, <54), above range (>180,
>250), mean glucose and CV between the twin and the person's CGM on the same
readings (every observed reading after the burn-in, not the thinned fitting set).

- ``fitted`` days: the twin with its fitted per-day and per-meal quantities
  (posterior mean) — can it reproduce the days it was fitted on?
- ``held-out`` days, recorded inputs only — does it predict days it never saw?
  ``unknowns="typical"`` (headline): day/meal unknowns at central values
  (logged meals scaled by the person's carb-count bias, carb-free boluses / food photos as meals of this
  person's typical fitted size, no drift, and the person's average fitted
  disturbance for each 30 minutes of the day — systematic unlogged food or
  activity at habitual times), spread only from parameter uncertainty and CGM
  noise. ``unknowns="prior"``: unknowns drawn from their
  priors. On real data the fitted disturbance and drifts are tied to each day's
  own events, so random draws overstate day-to-day swings and bias TIR low; the
  prior mode is a sensitivity bound, not the forecast.

The twin's glucose gets CGM noise added (fitted residual sd) before scoring,
since the person's numbers include sensor noise and TBR/TAR are sensitive to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from t1d_twin.data import PersonTimeline, detect_unlogged_meals
from t1d_twin.fit import CGM_CV, TwinFit
from t1d_twin.model import DTYPE, FLUX_BLOCK_STEPS, TwinProblem, SHIFT_MAX_MIN, SHIFT_UNIT_MIN
from t1d_twin.params import (
    FLUX_CLAMP,
    FLUX_PRIOR_DF,
    FLUX_PRIOR_SCALE,
    LOGGED_SHIFT_PRIOR_SD,
    MEAL_SCALE_PRIOR,
    UNLOGGED_LOG_G_PRIOR,
    UNLOGGED_SHIFT_PRIOR_SD,
)

METRICS = ("tir_70_180", "tbr_70", "tbr_54", "tar_180", "tar_250", "mean_mgdl", "cv")


def metrics(g: np.ndarray) -> dict[str, float]:
    g = g[~np.isnan(g)]
    if g.size == 0:
        return {m: float("nan") for m in METRICS}
    return {
        "tir_70_180": float(np.mean((g >= 70) & (g <= 180))),
        "tbr_70": float(np.mean(g < 70)),
        "tbr_54": float(np.mean(g < 54)),
        "tar_180": float(np.mean(g > 180)),
        "tar_250": float(np.mean(g > 250)),
        "mean_mgdl": float(np.mean(g)),
        "cv": float(np.std(g) / np.mean(g)),
    }


def _shift_latent(minutes: list[float]) -> torch.Tensor:
    m = torch.tensor(minutes, dtype=DTYPE).clamp(-SHIFT_MAX_MIN + 0.01, SHIFT_MAX_MIN - 0.01)
    return torch.atanh(m / SHIFT_MAX_MIN) * SHIFT_MAX_MIN / SHIFT_UNIT_MIN


def _score_mask(prob: TwinProblem) -> np.ndarray:
    """Every observed CGM reading after the burn-in (the fit used every third)."""
    mask = np.zeros_like(prob.mask)
    for wi, w in enumerate(prob.windows):
        obs = ~np.isnan(prob.tl.cgm[w.start:w.start + w.length])
        obs[: w.loss_from] = False
        mask[wi, : w.length] = obs
    return mask


def _noisy(sim: np.ndarray, cgm_sd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sd = np.sqrt(cgm_sd[:, None, None] ** 2 + (CGM_CV * sim) ** 2)
    return sim + sd * rng.standard_normal(sim.shape)


def _per_day(prob: TwinProblem, sims: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    """sims [K, W, S] -> per-day actual vs twin metrics (median and 5-95% over K)."""
    rows = []
    for wi, w in enumerate(prob.windows):
        m = mask[wi]
        actual = metrics(prob.tl.cgm[w.start:w.start + w.length][m[: w.length]])
        twin = [metrics(sims[k, wi][m]) for k in range(sims.shape[0])]
        row = {"date": prob.tl.day_dates[w.day], "n_readings": int(m.sum()), "actual": actual, "twin": {}}
        for name in METRICS:
            vals = np.array([t[name] for t in twin])
            lo, med, hi = np.percentile(vals, [5, 50, 95])
            row["twin"][name] = {"median": float(med), "p05": float(lo), "p95": float(hi),
                                 "actual_in_90": bool(lo <= actual[name] <= hi)}
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]], prob: TwinProblem, sims: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Pooled metrics over all scored readings, plus per-day agreement statistics."""
    actual_all = np.concatenate([prob.tl.cgm[w.start:w.start + w.length][mask[wi, : w.length]] for wi, w in enumerate(prob.windows)])
    twin_all = [np.concatenate([sims[k, wi][mask[wi]] for wi in range(len(prob.windows))]) for k in range(sims.shape[0])]
    pooled_actual = metrics(actual_all)
    out: dict[str, Any] = {"days": len(rows), "pooled": {}, "per_day": {}}
    for name in METRICS:
        tw = np.array([metrics(t)[name] for t in twin_all])
        lo, med, hi = np.percentile(tw, [5, 50, 95])
        diffs = np.array([r["twin"][name]["median"] - r["actual"][name] for r in rows])
        cover = np.mean([r["twin"][name]["actual_in_90"] for r in rows])
        a = np.array([r["actual"][name] for r in rows])
        t = np.array([r["twin"][name]["median"] for r in rows])
        corr = float(np.corrcoef(a, t)[0, 1]) if len(rows) > 2 and np.std(a) > 0 and np.std(t) > 0 else float("nan")
        out["pooled"][name] = {"actual": pooled_actual[name], "twin_median": float(med), "twin_p05": float(lo), "twin_p95": float(hi)}
        out["per_day"][name] = {"mean_diff": float(diffs.mean()), "mean_abs_diff": float(np.abs(diffs).mean()),
                                "day_to_day_corr": corr, "actual_in_twin_90": float(cover)}
    return out


def typical_flux_blocks(fit: TwinFit, tl: PersonTimeline, prob: TwinProblem) -> torch.Tensor:
    """[W * blocks] flux: the person's mean fitted flux for each 30-min block of the local day."""
    per_day = [np.asarray(v, dtype=float) for v in fit.day_flux.values()]
    width = max(len(v) for v in per_day)
    stack = np.full((len(per_day), width), np.nan)
    for i, v in enumerate(per_day):
        stack[i, :len(v)] = v
    profile = np.nan_to_num(np.nanmean(stack, axis=0))              # by block from local midnight
    nb, burn = prob.n_flux_blocks_per_window, prob.windows[0].loss_from // FLUX_BLOCK_STEPS
    blocks = np.zeros((len(prob.windows), nb))
    for wi in range(len(prob.windows)):
        idx = (np.arange(nb) - burn) % len(profile)                  # burn-in blocks wrap to the previous evening
        blocks[wi] = profile[idx]
    return torch.tensor(blocks.reshape(-1), dtype=DTYPE)


def validate_fitted_days(fit: TwinFit, tl: PersonTimeline, *, samples: int = 16, seed: int = 0) -> dict[str, Any]:
    """Twin with the fitted per-day/per-meal quantities, on the days it was fitted."""
    priors = fit.priors()
    days = [tl.day_dates.index(d) for d in fit.fitted_days if d in tl.day_dates]
    use_flux = bool(fit.day_flux)
    prob = TwinProblem(tl, fit.base, days, priors, unlogged=detect_unlogged_meals(tl, use_cgm_rises=not use_flux),
                       min_cgm_coverage=0.0, flux=use_flux)
    prob_days = [tl.day_dates[d] for d in prob.days]
    if prob_days != fit.fitted_days or prob.n_logged != len(fit.meal_scale_loc) or prob.n_unlogged != len(fit.unlogged_log_g_loc):
        raise ValueError("timeline does not reproduce the fitted problem (different records, settings or offset?)")

    gen = torch.Generator().manual_seed(seed)
    u = fit.sample_globals(samples, gen)
    rep = lambda v: torch.tensor(v, dtype=DTYPE)[None].expand(samples, -1)
    flux = None
    if use_flux:
        nb, burn_blocks = prob.n_flux_blocks_per_window, prob.windows[0].loss_from // FLUX_BLOCK_STEPS
        blocks = torch.zeros(len(prob.windows), nb, dtype=DTYPE)
        for wi, w in enumerate(prob.windows):
            date = tl.day_dates[w.day]
            own = fit.day_flux.get(date, [])
            blocks[wi, burn_blocks:burn_blocks + len(own)] = torch.tensor(own[: nb - burn_blocks], dtype=DTYPE)
            prev = fit.day_flux.get(tl.day_dates[w.day - 1]) if w.day > 0 else None
            if prev:
                blocks[wi, :burn_blocks] = torch.tensor(prev[-burn_blocks:], dtype=DTYPE)
        flux = blocks.reshape(1, -1).expand(samples, -1)
    with torch.no_grad():
        sim = prob.simulate(
            u, rep(fit.day_drift_loc), rep(fit.meal_scale_loc), rep(fit.unlogged_log_g_loc),
            _shift_latent(fit.meal_shift_min)[None].expand(samples, -1) if fit.meal_shift_min else None,
            _shift_latent(fit.unlogged_shift_min)[None].expand(samples, -1) if fit.unlogged_shift_min else None,
            flux, rep(fit.day_egp_drift_loc) if fit.day_egp_drift_loc else None,
        ).numpy()
    cgm_sd = torch.exp(u[:, priors.names().index("log_cgm_sd")]).numpy()
    sims = _noisy(sim, cgm_sd, np.random.default_rng(seed))
    mask = _score_mask(prob)
    rows = _per_day(prob, sims, mask)
    return {"mode": "fitted days (posterior-mean day/meal quantities)", "summary": _summary(rows, prob, sims, mask), "days": rows}


def validate_heldout_days(fit: TwinFit, tl: PersonTimeline, days: list[int], *, samples: int = 32, seed: int = 1,
                          min_cgm_coverage: float = 0.25, unknowns: str = "typical") -> dict[str, Any]:
    """Recorded inputs only (nothing reads the held-out CGM); see module doc for ``unknowns``."""
    priors = fit.priors()
    names = priors.names()
    typical = unknowns == "typical"
    anchors = detect_unlogged_meals(tl, use_cgm_rises=False)
    prob = TwinProblem(tl, fit.base, days, priors, unlogged=anchors, min_cgm_coverage=min_cgm_coverage, flux=bool(fit.day_flux))
    gen = torch.Generator().manual_seed(seed)
    u = fit.sample_globals(samples, gen)
    D, M, U = prob.n_days, prob.n_logged, prob.n_unlogged
    draw = lambda n, mean, sd: mean + sd * torch.randn(samples, n, generator=gen, dtype=DTYPE)
    if typical:
        zeros = lambda n: torch.zeros(samples, n, dtype=DTYPE)
        # announcements without carbs eat this person's typical fitted announcement size
        typical_log_g = float(np.median(fit.unlogged_log_g_loc)) if fit.unlogged_log_g_loc else UNLOGGED_LOG_G_PRIOR[0]
        flux = None
        if prob.n_flux:
            flux = typical_flux_blocks(fit, tl, prob)[None].expand(samples, -1)
        with torch.no_grad():
            sim = prob.simulate(u, zeros(D), torch.full((samples, M), fit.carb_count_bias, dtype=DTYPE),
                                torch.full((samples, U), typical_log_g, dtype=DTYPE),
                                zeros(M), zeros(U), flux, zeros(D)).numpy()
        cgm_sd = torch.exp(u[:, names.index("log_cgm_sd")]).numpy()
        sims = _noisy(sim, cgm_sd, np.random.default_rng(seed))
        mask = _score_mask(prob)
        rows = _per_day(prob, sims, mask)
        return {"mode": "held-out days, typical-day forecast (recorded inputs; unknowns at central values)",
                "summary": _summary(rows, prob, sims, mask), "days": rows}
    with torch.no_grad():
        sim = prob.simulate(
            u,
            torch.exp(u[:, names.index("log_day_si_sd")])[:, None] * torch.randn(samples, D, generator=gen, dtype=DTYPE),
            draw(M, *MEAL_SCALE_PRIOR), draw(U, *UNLOGGED_LOG_G_PRIOR),
            draw(M, 0.0, LOGGED_SHIFT_PRIOR_SD), draw(U, 0.0, UNLOGGED_SHIFT_PRIOR_SD),
            (FLUX_PRIOR_SCALE * torch.distributions.StudentT(FLUX_PRIOR_DF).sample((samples, prob.n_flux))).clamp(-FLUX_CLAMP, FLUX_CLAMP).to(DTYPE),
            torch.exp(u[:, names.index("log_day_egp_sd")])[:, None] * torch.randn(samples, D, generator=gen, dtype=DTYPE),
        ).numpy()
    cgm_sd = torch.exp(u[:, names.index("log_cgm_sd")]).numpy()
    sims = _noisy(sim, cgm_sd, np.random.default_rng(seed))
    mask = _score_mask(prob)
    rows = _per_day(prob, sims, mask)
    return {"mode": "held-out days, unknowns drawn from priors (sensitivity bound)", "summary": _summary(rows, prob, sims, mask), "days": rows}
