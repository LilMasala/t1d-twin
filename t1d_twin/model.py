"""The personal twin: base adult + fitted parameters + a person's real inputs.

``TwinProblem`` cuts a timeline into day windows (each with a burn-in on the
previous evening's real inputs) for fitting, and maps latent values to
simulated CGM. ``rollout`` runs one continuous trajectory across days — state
is never re-initialised at midnight — optionally with a controller in the
loop, for held-out evaluation and settings experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.context import ContextFeatures, build_features, multipliers
from t1d_twin.data import MEAL_WINDOW_STEPS, Meal, PersonTimeline, detect_unlogged_meals
from t1d_twin.params import GLOBAL_SPECS, RESPONSE_BIN_MIN, RESPONSE_BINS, TwinPriors

DTYPE = torch.float32
BURN_IN_H = 6.0
FEATURE_KEYS = ("luteal", "menstrual", "exercise_now", "exercise_load", "sleep_deficit_h", "dawn_ramp", "site_age_excess_d",
                "stress", "cycle_cos", "cycle_sin", "day_cos", "day_sin")
OBS_EVERY_BINS = 3  # use every third 5-min CGM reading (15 min) to thin autocorrelated residuals
SHIFT_EXT_STEPS = 15   # a meal may land up to 30 min either side of its recorded time
SHIFT_MAX_MIN = 25.0   # soft bound on the fitted shift
SHIFT_UNIT_MIN = 10.0  # shift latents are in units of 10 min
FLUX_BLOCK_STEPS = 15  # disturbance flux is piecewise constant over 30-min blocks


def meal_time_shift(z: torch.Tensor) -> torch.Tensor:
    """Latent -> minutes, softly bounded to +/- SHIFT_MAX_MIN."""
    return SHIFT_MAX_MIN * torch.tanh(z * SHIFT_UNIT_MIN / SHIFT_MAX_MIN)


def response_basis(series: np.ndarray) -> np.ndarray:
    """[n_steps] event amounts -> [RESPONSE_BINS, n_steps]: amount that occurred j bins before each step."""
    width = int(RESPONSE_BIN_MIN / ode.DT_MIN)
    csum = np.concatenate([[0.0], np.cumsum(series)])
    n = series.size
    out = np.zeros((RESPONSE_BINS, n))
    idx = np.arange(n)
    for j in range(RESPONSE_BINS):
        hi = np.clip(idx - j * width + 1, 0, n)       # events in (t - (j+1)w, t - j w]
        lo = np.clip(idx - (j + 1) * width + 1, 0, n)
        out[j] = csum[hi] - csum[lo]
    return out


def response_coefs(g: dict[str, torch.Tensor], prefix: str) -> torch.Tensor | None:
    """[K, RESPONSE_BINS] coefficients, or None when the model has none."""
    if f"{prefix}0" not in g:
        return None
    return torch.stack([g[f"{prefix}{j}"] for j in range(RESPONSE_BINS)], dim=1)


def transform_globals(u: torch.Tensor, priors: TwinPriors) -> dict[str, torch.Tensor]:
    """[K, G] unconstrained -> dict name -> [K] model values."""
    return {s.name: s.transform(u[:, i]) for i, s in enumerate(priors.specs)}


def personalise_params(base: ode.BasePatient, g: dict[str, torch.Tensor], body_mass_kg: float | None, reps: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Base adult params scaled by the person's core multipliers, [K * reps]."""
    K = g["log_si"].shape[0]
    p, Vg, x0 = ode.stack_params([base], dtype=DTYPE)
    out = {k: v.expand(K).clone() for k, v in p.items()}
    if body_mass_kg is not None:
        out["BW"] = torch.full((K,), float(body_mass_kg), dtype=DTYPE)
    out["Vmx"] = out["Vmx"] * g["log_si"]
    out["kp1"] = out["kp1"] * g["log_egp"]
    for k in ("ka1", "ka2", "kd"):
        out[k] = out[k] * g["log_insulin_speed"]
    for k in ("kabs", "kmax", "kmin"):
        out[k] = out[k] * g["log_carb_speed"]
    out["f"] = out["f"] * g["log_carb_effect"]
    if "log_hypo_uptake" in g:
        out["r1"] = g["log_hypo_uptake"]
    if "log_insulin_action_speed" in g:
        for k in ("p2u", "ki"):
            out[k] = out[k] * g["log_insulin_action_speed"]
    out = {k: v.repeat_interleave(reps) for k, v in out.items()}
    return out, Vg.expand(K * reps).clone(), x0.expand(K * reps, 13).clone()


@dataclass
class Window:
    day: int
    start: int
    length: int
    loss_from: int


class TwinProblem:
    """Day windows of one person, ready to simulate in a single batch."""

    def __init__(
        self,
        tl: PersonTimeline,
        base_name: str,
        days: list[int],
        priors: TwinPriors,
        *,
        feats: ContextFeatures | None = None,
        unlogged: list[Meal] | None = None,
        min_cgm_coverage: float = 0.7,
        flux: bool = False,
    ) -> None:
        self.tl = tl
        self.use_flux = flux
        self.base = ode.base_patient(base_name)
        self.priors = priors
        self.feats = feats or build_features(tl)
        self.unlogged_all = detect_unlogged_meals(tl) if unlogged is None else unlogged
        burn = int(BURN_IN_H * 60 / ode.DT_MIN)

        self.windows: list[Window] = []
        self.skipped: dict[int, str] = {}
        for d in days:
            s0, s1 = tl.day_steps(d)
            start = s0 - burn
            if start < 0:
                self.skipped[d] = "no burn-in before first day"
            elif not tl.insulin_observed[d] or (d > 0 and not tl.insulin_observed[d - 1]):
                self.skipped[d] = "insulin not observed"
            elif tl.meals_observed is not None and (not tl.meals_observed[d] or (d > 0 and not tl.meals_observed[d - 1])):
                self.skipped[d] = "meal log missing"
            elif np.any(np.isnan(tl.basal_upm[start:s1])):
                self.skipped[d] = "basal delivery unknown"
            elif tl.cgm_coverage[d] < min_cgm_coverage:
                self.skipped[d] = f"CGM coverage {tl.cgm_coverage[d]:.0%}"
            else:
                self.windows.append(Window(d, start, s1 - start, burn))
        if not self.windows:
            raise ValueError(f"no fittable days: {self.skipped}")

        self.days = [w.day for w in self.windows]
        W, S = len(self.windows), max(w.length for w in self.windows)
        self.S = S
        self.ins = np.zeros((W, S))
        self.cgm = np.zeros((W, S))
        self.mask = np.zeros((W, S), dtype=bool)
        self.g0 = np.zeros(W)
        self.basal_ratio = np.zeros(W)
        feat_np = {k: np.zeros((W, S)) for k in FEATURE_KEYS}
        meal_pos, meal_ref = [], []  # (w, step_in_window) / ("logged"|"unlogged", index)

        for wi, w in enumerate(self.windows):
            sl = slice(w.start, w.start + w.length)
            self.ins[wi, :w.length] = tl.basal_upm[sl] + tl.bolus_upm[sl]
            cg = tl.cgm[sl]
            obs = ~np.isnan(cg)
            thin = np.zeros(w.length, dtype=bool)
            obs_idx = np.flatnonzero(obs)
            thin[obs_idx[::OBS_EVERY_BINS]] = True
            loss = thin & (np.arange(w.length) >= w.loss_from)
            self.mask[wi, :w.length] = loss
            self.cgm[wi, :w.length] = np.where(obs, cg, 0.0)
            near = obs_idx[obs_idx < int(20 / ode.DT_MIN)]
            day_obs = cg[obs]
            self.g0[wi] = cg[near[0]] if near.size else (np.median(day_obs) if day_obs.size else 120.0)
            self.basal_ratio[wi] = tl.basal_upm[w.start] * 60.0 / self.base.basal_u_per_hr
            for k in FEATURE_KEYS:
                feat_np[k][wi, :w.length] = getattr(self.feats, k)[sl]

            for mi, m in enumerate(tl.meals):
                if w.start <= m.step < w.start + w.length:
                    meal_pos.append((wi, m.step - w.start))
                    meal_ref.append(("logged", mi))
            for ui, m in enumerate(self.unlogged_all):
                if w.start <= m.step < w.start + w.length:
                    meal_pos.append((wi, m.step - w.start))
                    meal_ref.append(("unlogged", ui))

        # Latent index spaces: only meals that some window actually uses.
        self.logged_ids = sorted({i for kind, i in meal_ref if kind == "logged"})
        self.unlogged_ids = sorted({i for kind, i in meal_ref if kind == "unlogged"})
        lpos = {i: n for n, i in enumerate(self.logged_ids)}
        upos = {i: n for n, i in enumerate(self.unlogged_ids)}
        self.event_is_logged = torch.tensor([kind == "logged" for kind, _ in meal_ref], dtype=torch.bool)
        self.event_latent = torch.tensor([lpos[i] if kind == "logged" else upos[i] for kind, i in meal_ref], dtype=torch.long)
        self.event_logged_g = torch.tensor(
            [tl.meals[i].grams if kind == "logged" else 0.0 for kind, i in meal_ref], dtype=DTYPE
        )
        # Each meal spreads over +/- 30 min around its recorded time; the fitted
        # time shift decides where in that span the carbs actually arrive.
        pos, owner, offset, starts = [], [], [], []
        for e, (wi, s) in enumerate(meal_pos):
            starts.append(wi * S + max(0, s - SHIFT_EXT_STEPS))
            for j in range(-SHIFT_EXT_STEPS, SHIFT_EXT_STEPS + MEAL_WINDOW_STEPS):
                if 0 <= s + j < S:
                    pos.append(wi * S + s + j)
                    owner.append(e)
                    offset.append(j * ode.DT_MIN)
        self.cho_pos = torch.tensor(pos, dtype=torch.long)
        self.cho_owner = torch.tensor(owner, dtype=torch.long)
        self.cho_offset_min = torch.tensor(offset, dtype=DTYPE)
        self.meal_start = torch.zeros(W * S, dtype=torch.bool)
        if starts:
            self.meal_start[torch.tensor(starts)] = True
        self.meal_start = self.meal_start.view(W, S)

        t = lambda a: torch.tensor(a, dtype=DTYPE)
        self.ins_t, self.cgm_t, self.mask_t = t(self.ins), t(self.cgm), torch.tensor(self.mask)
        self.g0_t, self.basal_ratio_t = t(self.g0), t(self.basal_ratio)
        self.feat_t = {k: t(v) for k, v in feat_np.items()}
        self.n_flux_blocks_per_window = -(-S // FLUX_BLOCK_STEPS)
        # response-correction bases from logged carbs (g) and bolus insulin (U), with history before each window
        carb_series = np.zeros(tl.n_steps)
        for m in tl.meals:
            if m.logged:
                carb_series[m.step] += m.grams
        bases = {"carb": response_basis(carb_series), "ins": response_basis(tl.bolus_upm * ode.DT_MIN)}
        self.basis_t = {}
        for key, full in bases.items():
            arr = np.zeros((RESPONSE_BINS, W, S))
            for wi, w in enumerate(self.windows):
                arr[:, wi, :w.length] = full[:, w.start:w.start + w.length]
            self.basis_t[key] = torch.tensor(arr, dtype=DTYPE)

    @property
    def n_logged(self) -> int:
        return len(self.logged_ids)

    @property
    def n_unlogged(self) -> int:
        return len(self.unlogged_ids)

    @property
    def n_days(self) -> int:
        return len(self.windows)

    @property
    def n_flux(self) -> int:
        return len(self.windows) * self.n_flux_blocks_per_window if self.use_flux else 0

    def flux_steps(self, flux_blocks: torch.Tensor) -> torch.Tensor:
        """[K, W * blocks] block values -> [K * W, S] per-step flux (mg/kg/min)."""
        K = flux_blocks.shape[0]
        W, S = len(self.windows), self.S
        per_step = flux_blocks.view(K, W, self.n_flux_blocks_per_window).repeat_interleave(FLUX_BLOCK_STEPS, dim=2)
        return per_step[:, :, :S].reshape(K * W, S)

    def meal_grams(self, meal_log_scale: torch.Tensor, unlogged_log_g: torch.Tensor) -> torch.Tensor:
        """[K, E] grams per meal event."""
        K = meal_log_scale.shape[0] if meal_log_scale.numel() else unlogged_log_g.shape[0]
        E = self.event_latent.numel()
        grams = torch.zeros(K, E, dtype=DTYPE)
        if E == 0:
            return grams
        li = self.event_is_logged
        if li.any():
            grams[:, li] = self.event_logged_g[li] * torch.exp(meal_log_scale[:, self.event_latent[li]])
        if (~li).any():
            grams[:, ~li] = torch.exp(unlogged_log_g[:, self.event_latent[~li]])
        return grams

    def meal_shifts(self, shift_logged: torch.Tensor | None, shift_unlogged: torch.Tensor | None, K: int) -> torch.Tensor:
        """[K, E] minutes each meal event is moved from its recorded time."""
        E = self.event_latent.numel()
        z = torch.zeros(K, E, dtype=DTYPE)
        li = self.event_is_logged
        if shift_logged is not None and li.any():
            z[:, li] = shift_logged[:, self.event_latent[li]]
        if shift_unlogged is not None and (~li).any():
            z[:, ~li] = shift_unlogged[:, self.event_latent[~li]]
        return meal_time_shift(z)

    def simulate(
        self,
        u_globals: torch.Tensor,
        day_log_si: torch.Tensor,
        meal_log_scale: torch.Tensor,
        unlogged_log_g: torch.Tensor,
        shift_logged: torch.Tensor | None = None,
        shift_unlogged: torch.Tensor | None = None,
        flux_blocks: torch.Tensor | None = None,
        day_log_egp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Latents [K, ...] -> simulated CGM [K, W, S]."""
        K = u_globals.shape[0]
        W, S = len(self.windows), self.S
        g = transform_globals(u_globals, self.priors)
        p, Vg, x0b = personalise_params(self.base, g, self.tl.body_mass_kg, W)

        cho = torch.zeros(K, W * S, dtype=DTYPE)
        if self.cho_pos.numel():
            grams = self.meal_grams(meal_log_scale, unlogged_log_g)
            tau = self.meal_shifts(shift_logged, shift_unlogged, K)[:, self.cho_owner]
            # eating ~10 min starting at the shifted time: logistic cumulative intake
            eat_len = MEAL_WINDOW_STEPS * ode.DT_MIN
            cdf = lambda t: torch.sigmoid((t - tau - eat_len / 2) / (eat_len / 4))
            frac = cdf(self.cho_offset_min + ode.DT_MIN) - cdf(self.cho_offset_min)
            total = torch.zeros(K, grams.shape[1], dtype=DTYPE).index_add(1, self.cho_owner, frac)
            rate = grams[:, self.cho_owner] * frac / total[:, self.cho_owner].clamp_min(1e-6) / ode.DT_MIN
            cho = cho.index_add(1, self.cho_pos, rate)
        cho = cho.view(K * W, S)

        drift = day_log_si[:, :, None].expand(K, W, S)
        egp_drift = None if day_log_egp is None or day_log_egp.numel() == 0 else day_log_egp[:, :, None].expand(K, W, S)
        si, egp, vm0 = multipliers(self.feat_t, g, drift, egp_drift)
        x0 = ode.initial_state(x0b, Vg, self.g0_t.repeat(K), self.basal_ratio_t.repeat(K))
        glucose = ode.simulate(
            x0, p, Vg, cho,
            self.meal_start.repeat(K, 1),
            self.ins_t.repeat(K, 1),
            si.reshape(K * W, S), egp.reshape(K * W, S), vm0.reshape(K * W, S),
            flux=self.total_flux(g, flux_blocks, K),
        )
        return glucose.view(K, W, S)

    def total_flux(self, g: dict[str, torch.Tensor], flux_blocks: torch.Tensor | None, K: int) -> torch.Tensor | None:
        """Free disturbance blocks + personal response corrections, [K * W, S] (None if neither)."""
        W, S = len(self.windows), self.S
        flux = None if flux_blocks is None or flux_blocks.numel() == 0 else self.flux_steps(flux_blocks)
        for key, prefix in (("carb", "carb_resp_k"), ("ins", "ins_resp_k")):
            coefs = response_coefs(g, prefix)
            if coefs is None:
                continue
            resp = torch.einsum("kj,jws->kws", coefs, self.basis_t[key]).reshape(K * W, S)
            flux = resp if flux is None else flux + resp
        return flux


def rollout(
    tl: PersonTimeline,
    base_name: str,
    priors: TwinPriors,
    u_globals: torch.Tensor,
    start_step: int,
    n_steps: int,
    *,
    meal_grams: list[tuple[int, torch.Tensor]],
    day_log_si: torch.Tensor | None = None,
    insulin_upm: torch.Tensor | None = None,
    controller=None,
    feats: ContextFeatures | None = None,
    flux_upm: torch.Tensor | None = None,
    day_log_egp: torch.Tensor | None = None,
    bolus_upm: torch.Tensor | None = None,
    glucose_reset: tuple[int, float] | list[tuple[int, float]] | None = None,
) -> torch.Tensor:
    """One continuous trajectory per row of ``u_globals`` ([B, G]).

    ``meal_grams`` is a list of (absolute step, grams [B]) the body eats.
    Insulin is either replayed (``insulin_upm`` [B, T]) or produced by
    ``controller(step, last_cgm [B]) -> U/min [B]``. ``day_log_si`` is
    [B, n_days] drift indexed by the timeline's local day. ``flux_upm`` [B, T]
    is the fitted disturbance flux for these steps (mg/kg/min). Personal
    response corrections apply to the logged carbs and to bolus insulin: the
    replayed ``bolus_upm`` [B, T], or the controller's ``last_bolus_u``.
    """
    feats = feats or build_features(tl)
    B = u_globals.shape[0]
    g = transform_globals(u_globals, priors)
    base = ode.base_patient(base_name)
    p, Vg, x0b = personalise_params(base, g, tl.body_mass_kg, 1)
    sl = slice(start_step, start_step + n_steps)

    cho = torch.zeros(B, n_steps, dtype=DTYPE)
    starts = torch.zeros(B, n_steps, dtype=torch.bool)
    for step, grams in meal_grams:
        s = step - start_step
        if 0 <= s < n_steps:
            starts[:, s] = True
            cho[:, s:s + MEAL_WINDOW_STEPS] += (grams / (MEAL_WINDOW_STEPS * ode.DT_MIN))[:, None]

    days = torch.tensor(np.clip(tl.day_index[sl], 0, None), dtype=torch.long)
    drift = torch.zeros(B, n_steps, dtype=DTYPE) if day_log_si is None else day_log_si[:, days]
    feat = {k: torch.tensor(getattr(feats, k)[sl], dtype=DTYPE)[None] for k in FEATURE_KEYS}
    egp_drift = None if day_log_egp is None else day_log_egp[:, days][:, None, :]
    si, egp, vm0 = multipliers(feat, g, drift[:, None, :], egp_drift)
    si, egp, vm0 = si[:, 0], egp[:, 0], vm0[:, 0]

    obs = tl.cgm[start_step:start_step + int(30 / ode.DT_MIN)]
    g0 = float(obs[~np.isnan(obs)][0]) if np.any(~np.isnan(obs)) else 120.0
    basal0 = tl.basal_upm[start_step] if not np.isnan(tl.basal_upm[start_step]) else base.basal_u_per_hr / 60.0
    x = ode.initial_state(x0b, Vg, torch.full((B,), g0, dtype=DTYPE), torch.full((B,), basal0 * 60.0 / base.basal_u_per_hr, dtype=DTYPE))

    step_controller = None
    if controller is not None:
        step_controller = lambda t, cgm: controller(start_step + t, cgm)

    flux = flux_upm
    carb_coefs = response_coefs(g, "carb_resp_k")
    if carb_coefs is not None:
        carb_series = np.zeros(tl.n_steps)
        for m in tl.meals:
            if m.logged:
                carb_series[m.step] += m.grams
        basis = torch.tensor(response_basis(carb_series)[:, sl], dtype=DTYPE)
        carb_flux = carb_coefs @ basis
        flux = carb_flux if flux is None else flux + carb_flux
    ins_coefs = response_coefs(g, "ins_resp_k")
    history = None
    if ins_coefs is not None:
        width = int(RESPONSE_BIN_MIN / ode.DT_MIN)
        H = RESPONSE_BINS * width
        past = tl.bolus_upm[max(0, start_step - H):start_step] * ode.DT_MIN
        history = torch.zeros(B, H, dtype=DTYPE)
        if past.size:
            history[:, H - past.size:] = torch.tensor(past, dtype=DTYPE)
    return ode.simulate(
        x, p, Vg, cho, starts, insulin_upm, si, egp, vm0, controller=step_controller, flux=flux,
        ins_response=ins_coefs, bolus_upm=bolus_upm,
        bolus_fn=(lambda: controller.last_bolus_u) if controller is not None and hasattr(controller, "last_bolus_u") else None,
        bolus_history_u=history, response_bin_steps=int(RESPONSE_BIN_MIN / ode.DT_MIN),
        glucose_reset=[(t, v.expand(B) if torch.is_tensor(v) else torch.full((B,), float(v), dtype=DTYPE))
                       for t, v in ([glucose_reset] if isinstance(glucose_reset, tuple) else glucose_reset or [])],
    )
