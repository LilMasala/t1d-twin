"""Differentiable, batched UVA/Padova core for personal twins.

Same right-hand side as simglucose's ``T1DPatient`` (the reference
implementation of UVA/Padova 2008), ported to torch so a person's parameters can
be fitted by gradient. ``tests/test_twin.py`` checks the two agree. Differences from the simglucose orchestration, all deliberate:

- fixed 2-minute RK4 step (max error vs. the 1-minute x5 reference is
  ~0.2 mg/dL; 5-minute steps are unstable for several adults);
- carbohydrate intake arrives as a per-step rate with an explicit
  ``meal_start`` mask, so meal size stays differentiable (simglucose infers
  meal boundaries from the eaten amount, which is not);
- insulin sensitivity (Vmx), endogenous glucose production (kp1) and
  insulin-independent uptake (Vm0) take per-step multipliers, which is where
  context (cycle, sleep, exercise, site age, dawn) enters;
- insulin-dependent utilisation rises as glucose falls below basal, following the
  UVA/Padova S2013 hypoglycaemia modification (Dalla Man et al., J Diabetes Sci
  Technol 2014): Uid = [Vm0 + Vmx X (1 + r1 risk)] Gt/(Km0 + Gt), risk = 0 at or
  above basal glucose Gb, 10 [ln(G/Gb)]^2 below it, held at its 60 mg/dL value
  below 60. The paper's population constants are not reproduced: r1 is fitted per
  person (``r1`` = 0 recovers the 2008 model exactly);
- an optional signed glucose flux (mg/kg/min) enters plasma glucose directly:
  the fitted disturbance for what the records do not explain (unlogged food,
  unlogged activity). A negative flux fades out below 40 mg/dL, so a fitted sink
  replayed under more insulin cannot push glucose below zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

DT_MIN = 2.0
HYPO_THRESHOLD_MGDL = 60.0
# a negative disturbance is a glucose sink, and a sink needs glucose to act on: it
# fades linearly below this plasma glucose, so it cannot drive glucose through zero
FLUX_SINK_FLOOR_MGDL = 40.0
STEPS_PER_DAY = int(24 * 60 / DT_MIN)
ADULTS = tuple(f"adult#{i:03d}" for i in range(1, 11))

ODE_PARAMS = (
    "kmax", "kmin", "kabs", "b", "d", "f", "BW", "kp1", "kp2", "kp3", "Fsnc",
    "ke1", "ke2", "k1", "k2", "Vm0", "Vmx", "Km0", "m1", "m2", "m4", "m30",
    "ka1", "ka2", "Vi", "p2u", "Ib", "ki", "ksc", "kd", "u2ss",
)


@lru_cache(maxsize=1)
def _frame():
    from t1d_twin.vpatients import vpatient_frame

    return vpatient_frame()


@dataclass
class BasePatient:
    """One UVA/Padova virtual adult: ODE params, Vg, basal state."""

    name: str
    params: dict[str, float]
    Vg: float
    Gb: float
    x0: np.ndarray  # [13] basal steady state

    @property
    def basal_u_per_hr(self) -> float:
        # u2ss is pmol/kg/min at steady state -> U/hr for this body weight
        return self.params["u2ss"] * self.params["BW"] / 6000.0 * 60.0


@lru_cache(maxsize=None)
def base_patient(name: str) -> BasePatient:
    frame = _frame()
    row = frame[frame["Name"] == name].iloc[0]
    return BasePatient(
        name=name,
        params={k: float(row[k]) for k in ODE_PARAMS},
        Vg=float(row["Vg"]),
        Gb=float(row["Gb"]),
        x0=row.iloc[2:15].to_numpy(dtype=float),
    )


def stack_params(bases: list[BasePatient], dtype=torch.float64) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    p = {k: torch.tensor([b.params[k] for b in bases], dtype=dtype) for k in ODE_PARAMS}
    p["Vg"] = torch.tensor([b.Vg for b in bases], dtype=dtype)
    p["Gb"] = torch.tensor([b.Gb for b in bases], dtype=dtype)
    p["r1"] = torch.zeros(len(bases), dtype=dtype)   # S2013 hypoglycaemia term off unless personalised
    Vg = torch.tensor([b.Vg for b in bases], dtype=dtype)
    x0 = torch.tensor(np.stack([b.x0 for b in bases]), dtype=dtype)
    return p, Vg, x0


def rhs(x, p, cho, ins, dbar, flux):
    """Batched UVA/Padova RHS. x:[B,13]; p values:[B]; cho g/min; ins U/min; dbar mg; flux mg/kg/min."""
    d = cho * 1000.0
    insp = ins * 6000.0 / p["BW"]
    qsto = x[:, 0] + x[:, 1]

    has_meal = dbar > 0
    safe_d = torch.where(has_meal, dbar, torch.ones_like(dbar))
    aa = 5.0 / (2.0 * safe_d * (1.0 - p["b"]))
    cc = 5.0 / (2.0 * safe_d * p["d"])
    kgut_active = p["kmin"] + (p["kmax"] - p["kmin"]) / 2.0 * (
        torch.tanh(aa * (qsto - p["b"] * safe_d)) - torch.tanh(cc * (qsto - p["d"] * safe_d)) + 2.0
    )
    kgut = torch.where(has_meal, kgut_active, p["kmax"])

    Rat = p["f"] * p["kabs"] * x[:, 2] / p["BW"]
    EGPt = p["kp1"] - p["kp2"] * x[:, 3] - p["kp3"] * x[:, 8]
    Et = torch.where(x[:, 3] > p["ke2"], p["ke1"] * (x[:, 3] - p["ke2"]), torch.zeros_like(x[:, 3]))
    G = x[:, 3] / p["Vg"]
    log_ratio = torch.log(torch.clamp(G, min=HYPO_THRESHOLD_MGDL) / p["Gb"])
    risk = torch.where(G < p["Gb"], 10.0 * log_ratio ** 2, torch.zeros_like(G))
    Vmt = p["Vm0"] + p["Vmx"] * x[:, 6] * (1.0 + p["r1"] * risk)
    Uidt = Vmt * x[:, 4] / (p["Km0"] + x[:, 4])
    It = x[:, 5] / p["Vi"]
    sink = torch.clamp(G / FLUX_SINK_FLOOR_MGDL, min=0.0, max=1.0)
    flux = torch.where(flux < 0, flux * sink, flux)

    pos = (x >= 0).to(x.dtype)
    dx = torch.stack([
        -p["kmax"] * x[:, 0] + d,
        p["kmax"] * x[:, 0] - x[:, 1] * kgut,
        kgut * x[:, 1] - p["kabs"] * x[:, 2],
        (torch.relu(EGPt) + Rat + flux - p["Fsnc"] - Et - p["k1"] * x[:, 3] + p["k2"] * x[:, 4]) * pos[:, 3],
        (-Uidt + p["k1"] * x[:, 3] - p["k2"] * x[:, 4]) * pos[:, 4],
        (-(p["m2"] + p["m4"]) * x[:, 5] + p["m1"] * x[:, 9] + p["ka1"] * x[:, 10] + p["ka2"] * x[:, 11]) * pos[:, 5],
        -p["p2u"] * x[:, 6] + p["p2u"] * (It - p["Ib"]),
        -p["ki"] * (x[:, 7] - It),
        -p["ki"] * (x[:, 8] - x[:, 7]),
        (-(p["m1"] + p["m30"]) * x[:, 9] + p["m2"] * x[:, 5]) * pos[:, 9],
        (insp - (p["ka1"] + p["kd"]) * x[:, 10]) * pos[:, 10],
        (p["kd"] * x[:, 10] - p["ka2"] * x[:, 11]) * pos[:, 11],
        (-p["ksc"] * x[:, 12] + p["ksc"] * x[:, 3]) * pos[:, 12],
    ], dim=1)
    return dx


def rk4(x, p, cho, ins, dbar, dt, flux):
    k1 = rhs(x, p, cho, ins, dbar, flux)
    k2 = rhs(x + 0.5 * dt * k1, p, cho, ins, dbar, flux)
    k3 = rhs(x + 0.5 * dt * k2, p, cho, ins, dbar, flux)
    k4 = rhs(x + dt * k3, p, cho, ins, dbar, flux)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


_compiled_rk4 = None


def _step_fn():
    """torch.compile'd RK4 step (~6x faster fwd+bwd on CPU after a ~30 s first compile).

    Set TWIN_COMPILE=0 to run eagerly.
    """
    global _compiled_rk4
    if os.environ.get("TWIN_COMPILE", "1") == "0":
        return rk4
    if _compiled_rk4 is None:
        _compiled_rk4 = torch.compile(rk4, dynamic=True)
    return _compiled_rk4


def simulate(
    x0: torch.Tensor,
    p: dict[str, torch.Tensor],
    Vg: torch.Tensor,
    cho_gpm: torch.Tensor,
    meal_start: torch.Tensor,
    ins_upm: torch.Tensor,
    si_mult: torch.Tensor,
    egp_mult: torch.Tensor,
    vm0_mult: torch.Tensor,
    *,
    dt: float = DT_MIN,
    return_state: bool = False,
    controller=None,
    flux: torch.Tensor | None = None,
    ins_response: torch.Tensor | None = None,
    bolus_upm: torch.Tensor | None = None,
    bolus_fn=None,
    bolus_history_u: torch.Tensor | None = None,
    response_bin_steps: int = 15,
    glucose_reset: tuple[int, torch.Tensor] | list[tuple[int, torch.Tensor]] | None = None,
):
    """Integrate [B] patients over T steps. Inputs are [B, T] per-step values.

    Insulin is ``ins_upm[:, t]`` plus, when given, ``controller(t, cgm)``
    (U/min [B]) deciding from the glucose after the previous step.
    ``flux`` [B, T] is an extra signed glucose appearance (mg/kg/min).
    ``ins_response`` [B, J] adds a learned flux per unit of bolus insulin by
    time-since-bolus bin; boluses come from ``bolus_upm`` [B, T] (replay) or
    ``bolus_fn()`` (U just decided by the controller), with
    ``bolus_history_u`` [B, J * bin] the boluses before the first step.
    ``glucose_reset=(t, g)`` rescales the glucose compartments before step ``t``
    so sensor glucose equals ``g`` [B] (keeps insulin and carbs on board):
    a warm start anchored to an observed reading. A list applies several.
    Returns subcutaneous glucose (mg/dL) *after* each step, [B, T]; with
    ``return_state`` also the final state [B, 13] so rollouts can continue
    across midnight without re-initialising.
    """
    B, T = cho_gpm.shape
    resets = dict([glucose_reset] if isinstance(glucose_reset, tuple) else glucose_reset or [])
    x = x0
    last_qsto = x[:, 0] + x[:, 1]
    eaten = torch.zeros(B, dtype=x.dtype)
    Vmx, kp1, Vm0 = p["Vmx"], p["kp1"], p["Vm0"]
    cgm = x[:, 12] / Vg
    step = _step_fn()
    zero_flux = torch.zeros(B, dtype=x.dtype)
    if ins_response is not None:
        J = ins_response.shape[1]
        H = J * response_bin_steps
        hist = torch.zeros(B, H, dtype=x.dtype) if bolus_history_u is None else bolus_history_u.clone()
    out = []
    for t in range(T):
        if t in resets:
            ratio = resets[t] / (x[:, 12] / Vg).clamp_min(1.0)
            scale = torch.ones_like(x)
            scale[:, [3, 4, 12]] = ratio[:, None]
            x = x * scale
            cgm = x[:, 12] / Vg
        start = meal_start[:, t]
        last_qsto = torch.where(start, x[:, 0] + x[:, 1], last_qsto)
        eaten = torch.where(start, torch.zeros_like(eaten), eaten) + cho_gpm[:, t] * dt
        dbar = last_qsto + eaten * 1000.0
        step_p = dict(p)
        step_p["Vmx"] = Vmx * si_mult[:, t]
        step_p["kp1"] = kp1 * egp_mult[:, t]
        step_p["Vm0"] = Vm0 * vm0_mult[:, t]
        ins = ins_upm[:, t] if ins_upm is not None else torch.zeros_like(cgm)
        if controller is not None:
            ins = ins + controller(t, cgm)
        step_flux = zero_flux if flux is None else flux[:, t]
        if ins_response is not None:
            bolus_now = bolus_upm[:, t] * dt if bolus_upm is not None else (bolus_fn() if bolus_fn is not None else torch.zeros_like(cgm))
            hist = torch.cat([hist[:, 1:], bolus_now[:, None]], dim=1)  # newest last; bin 0 includes this step
            bins = hist.flip(1).view(B, J, response_bin_steps).sum(2)
            step_flux = step_flux + (ins_response * bins).sum(1)
        x = step(x, step_p, cho_gpm[:, t], ins, dbar, dt, step_flux)
        cgm = x[:, 12] / Vg
        out.append(cgm)
    glucose = torch.stack(out, dim=1)
    return (glucose, x) if return_state else glucose


def initial_state(x0_basal: torch.Tensor, Vg: torch.Tensor, glucose0: torch.Tensor, basal_ratio: torch.Tensor) -> torch.Tensor:
    """Basal steady state with glucose set to an observed value.

    Glucose compartments (plasma, tissue, subcutaneous) are rescaled to the
    observed reading; insulin compartments are scaled by delivered basal over
    the base patient's steady-state basal. The burn-in period before any
    likelihood term absorbs the remaining mismatch.
    """
    x = x0_basal.clone()
    g_ratio = glucose0 / (x0_basal[:, 12] / Vg)
    for i in (3, 4, 12):
        x[:, i] = x0_basal[:, i] * g_ratio
    for i in (5, 7, 8, 9, 10, 11):
        x[:, i] = x0_basal[:, i] * basal_ratio
    x[:, 6] = 0.0
    return x
