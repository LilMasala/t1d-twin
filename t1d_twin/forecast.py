"""Line a fitted twin up with the person just before a forecast (moving-horizon estimation).

A warm start replays the recorded inputs before the forecast origin and
re-anchors glucose to the last reading. That keeps insulin and carbs on board
but throws away the recent trend: if the person has been drifting up for an
hour (an unlogged snack, stress, a failing site), the twin starts flat.

``assimilate`` fits the last ``ASSIM_H`` hours of CGM up to the origin: glucose
at the window start (prior: that reading and its noise) and 30-min disturbance
blocks (the main fit's heavy-tailed prior). The forecast can start from the
fitted state instead of the last raw reading — which matters for noisy sensors —
and carry the recent disturbance into the horizon, fading with a
``CARRY_HALF_LIFE_MIN`` half-life, on top of the person's typical time-of-day
disturbance. No reading after the origin is used.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.fit import CGM_CV, TwinFit, _student_t_logpdf
from t1d_twin.model import DTYPE, FLUX_BLOCK_STEPS, rollout
from t1d_twin.params import FLUX_CLAMP, FLUX_PRIOR_DF, FLUX_PRIOR_SCALE

ASSIM_H = 3.0
ASSIM_ITERS = 60
ASSIM_LR = 0.1
ASSIM_GLUCOSE_LR = 2.0   # mg/dL per step
CARRY_HALF_LIFE_MIN = 30.0


def typical_flux(fit: TwinFit, tl, start: int, n: int) -> torch.Tensor:
    """[1, n] the person's mean fitted disturbance for each 30-min block of the local day (zeros if none)."""
    if not fit.day_flux:
        return torch.zeros(1, n, dtype=DTYPE)
    per_day = [np.asarray(v, dtype=float) for v in fit.day_flux.values()]
    width = max(len(v) for v in per_day)
    stack = np.full((len(per_day), width), np.nan)
    for i, v in enumerate(per_day):
        stack[i, :len(v)] = v
    profile = np.nan_to_num(np.nanmean(stack, axis=0))
    minutes_of_day = tl.local_hour[start:start + n] * 60.0
    block = (minutes_of_day // (FLUX_BLOCK_STEPS * ode.DT_MIN)).astype(int) % len(profile)
    return torch.tensor(profile[block], dtype=DTYPE)[None]


def assimilate(fit: TwinFit, tl, start: int, origin: int, *, meals, insulin_upm, bolus_upm) -> tuple[list[tuple[int, float]], torch.Tensor] | None:
    """Fit glucose and disturbance over the window ending at ``origin`` (readings up to and including it).

    The rollout begins at ``start``; ``meals``/``insulin_upm``/``bolus_upm`` are its
    inputs (first row used). Glucose at the window's first reading is a free value
    with that reading's noise as prior (one noisy reading does not pin the state),
    and 30-min disturbance blocks explain the rest. Returns (glucose resets
    relative to ``start``, disturbance [origin - start] added over start..origin),
    or None when the window has too few readings.
    """
    n = origin - start
    w0 = max(0, n - int(ASSIM_H * 60 / ode.DT_MIN))
    seen = np.flatnonzero(~np.isnan(tl.cgm[start + w0:origin + 1]))
    if seen.size < 6:
        return None
    anchor = w0 + int(seen[0])
    target = torch.tensor(np.nan_to_num(tl.cgm[start + 1:origin + 1]), dtype=DTYPE)  # reading at step start + j + 1 <-> g[j]
    mask = torch.tensor(~np.isnan(tl.cgm[start + 1:origin + 1]), dtype=torch.bool)
    mask[:anchor] = False
    n_blocks = -(-(n - anchor) // FLUX_BLOCK_STEPS)
    u = torch.tensor(fit.loc, dtype=DTYPE)[None]
    cgm_sd = math.exp(fit.loc[fit.param_names.index("log_cgm_sd")])
    noise_sd = lambda g: torch.sqrt(cgm_sd ** 2 + (CGM_CV * g) ** 2)
    base = typical_flux(fit, tl, start, n)
    meals1 = [(s, g[:1]) for s, g in meals]
    first = float(tl.cgm[start + anchor])
    priors = fit.priors()

    def disturbance(blocks):
        per_step = blocks.repeat_interleave(FLUX_BLOCK_STEPS)[-(n - anchor):]  # last block ends at the origin
        return torch.cat([torch.zeros(anchor, dtype=DTYPE), per_step])

    blocks = torch.zeros(n_blocks, dtype=DTYPE, requires_grad=True)
    g_anchor = torch.tensor(first, dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([{"params": [blocks], "lr": ASSIM_LR}, {"params": [g_anchor], "lr": ASSIM_GLUCOSE_LR}])
    for _ in range(ASSIM_ITERS):
        opt.zero_grad()
        g = rollout(tl, fit.base, priors, u, start, n, meal_grams=meals1, insulin_upm=insulin_upm[:1, :n],
                    bolus_upm=bolus_upm[:1, :n], flux_upm=base + disturbance(blocks)[None], glucose_reset=[(anchor, g_anchor)])[0]
        nll = (0.5 * ((g - target) / noise_sd(target)) ** 2)[mask].sum()
        prior = _student_t_logpdf(blocks, FLUX_PRIOR_SCALE, FLUX_PRIOR_DF).sum() - 0.5 * ((g_anchor - first) / noise_sd(torch.tensor(first))) ** 2
        loss = nll - prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_([blocks, g_anchor], 10.0)
        opt.step()
        with torch.no_grad():
            blocks.clamp_(-FLUX_CLAMP, FLUX_CLAMP)
            g_anchor.clamp_(40.0, 500.0)
    return [(anchor, float(g_anchor.detach()))], disturbance(blocks.detach())


def carried_disturbance(last_block: float, n: int) -> torch.Tensor:
    """[n] the last fitted disturbance fading from the origin with ``CARRY_HALF_LIFE_MIN``."""
    t_min = torch.arange(n, dtype=DTYPE) * ode.DT_MIN
    return last_block * torch.pow(torch.tensor(0.5, dtype=DTYPE), t_min / CARRY_HALF_LIFE_MIN)
