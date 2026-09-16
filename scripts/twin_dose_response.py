"""What does one unit of insulin, or ten grams of carbs, do inside a fitted twin?

    python scripts/twin_dose_response.py --twin artifacts/uom/2307/twin.json \
        --records artifacts/uom/2307/raw_days

The benchmark only shows a twin reproduces glucose under the insulin that was
actually delivered. Settings experiments ask something it never tested: what a
dose that never happened would have done. No dataset contains that answer, so
this script checks the twin's dose response against what is known from outside
the model.

For each quiet moment in the fitted days (no bolus or meal for three hours before
or five hours after), the twin is run twice from the same state, once as
recorded and once with an extra unit of insulin, then again with ten extra grams
of carbohydrate. The peak difference gives:

- implied ISF, mg/dL per unit;
- implied CSF, mg/dL per gram, and the carb ratio that balances them,
  CR = ISF / CSF g/U.

Those are compared with three things measured from the person's own records:

- 1800 / total daily dose, the ISF rule of thumb;
- 500 / total daily dose, the CR rule of thumb;
- the grams per unit they actually bolused for logged meals.

The rules of thumb are population averages and individuals vary widely around
them, so treat a factor-of-two disagreement as a warning and a factor of five as
a broken twin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.data import build_timeline, detect_unlogged_meals, load_records, load_therapy_settings, shift_events
from t1d_twin.fit import TwinFit
from t1d_twin.forecast import typical_flux
from t1d_twin.model import DTYPE, rollout

BURN_H = 6.0
GUARD_H = 3.0
HORIZON_H = 5.0
TEST_DOSE_U = 1.0
TEST_CARBS_G = 10.0
CGM_LO, CGM_HI = 90.0, 200.0  # start from a quiet in-range glucose
BOLUS_MEAL_WINDOW_MIN = 20.0
CGM_LOOKUP_MIN = 6.0  # readings sit on a 5-minute grid, the model on 2-minute steps


def nearest_cgm(tl, step: int) -> tuple[int, float]:
    """The step of the reading closest to ``step`` (within CGM_LOOKUP_MIN), and its value."""
    width = int(CGM_LOOKUP_MIN / ode.DT_MIN)
    lo, hi = max(0, step - width), min(tl.n_steps, step + width + 1)
    window = tl.cgm[lo:hi]
    seen = np.flatnonzero(~np.isnan(window))
    if seen.size == 0:
        return step, float("nan")
    at = lo + int(seen[np.argmin(np.abs(lo + seen - step))])
    return at, float(tl.cgm[at])


def quiet_starts(tl, fit, limit: int) -> list[int]:
    """Steps on fitted days with no meal or bolus nearby and a settled in-range CGM."""
    busy = tl.bolus_upm > 0
    for m in tl.meals:
        busy[m.step] = True
    guard, horizon = int(GUARD_H * 60 / ode.DT_MIN), int(HORIZON_H * 60 / ode.DT_MIN)
    burn = int(BURN_H * 60 / ode.DT_MIN)
    out = []
    for day in (tl.day_dates.index(d) for d in fit.fitted_days):
        s0, s1 = tl.day_steps(day)
        step = max(s0, burn)
        while step < min(s1, tl.n_steps - horizon - 1):
            at, value = nearest_cgm(tl, step)
            if not busy[max(0, step - guard): step + horizon].any() and CGM_LO <= value <= CGM_HI:
                out.append(at)  # start on the reading itself
                step += horizon  # keep windows disjoint
            else:
                step += int(30 / ode.DT_MIN)
    rng = np.random.default_rng(0)
    return sorted(rng.choice(out, limit, replace=False).tolist()) if len(out) > limit else out


def dose_response(fit: TwinFit, tl, step: int, anchors) -> dict[str, float]:
    """Peak glucose difference from one extra unit, and from ten extra grams, at ``step``."""
    burn = int(BURN_H * 60 / ode.DT_MIN)
    start = step - burn
    n = burn + int(HORIZON_H * 60 / ode.DT_MIN)
    offset = step - start
    bias = float(np.exp(fit.carb_count_bias))
    meals = [(m.step, torch.tensor([m.grams * bias], dtype=DTYPE)) for m in tl.meals if m.logged]
    meals += [(a.step, torch.tensor([float(np.exp(np.median(fit.unlogged_log_g_loc)))], dtype=DTYPE)) for a in anchors] if fit.unlogged_log_g_loc else []
    sl = slice(start, start + n)
    ins = torch.tensor(np.nan_to_num(tl.basal_upm[sl]) + tl.bolus_upm[sl], dtype=DTYPE)[None]
    bol = torch.tensor(tl.bolus_upm[sl], dtype=DTYPE)[None]
    flux = typical_flux(fit, tl, start, n)
    u = torch.tensor(fit.loc, dtype=DTYPE)[None]
    reset = [(offset, float(tl.cgm[step]))]

    def run(extra_u: float = 0.0, extra_g: float = 0.0):
        i, b = ins.clone(), bol.clone()
        if extra_u:  # delivered in one 2-minute step
            i[0, offset] += extra_u / ode.DT_MIN
            b[0, offset] += extra_u / ode.DT_MIN
        m = meals + ([(step, torch.tensor([extra_g], dtype=DTYPE))] if extra_g else [])
        with torch.no_grad():
            return rollout(tl, fit.base, fit.priors(), u, start, n, meal_grams=m, insulin_upm=i, bolus_upm=b,
                           flux_upm=flux, glucose_reset=reset)[0].numpy()[offset:]

    base = run()
    return {
        "step": step,
        "cgm_at_start": float(tl.cgm[step]),
        "isf_mgdl_per_u": float(np.max(base - run(extra_u=TEST_DOSE_U))),
        "csf_mgdl_per_g": float(np.max(run(extra_g=TEST_CARBS_G) - base)) / TEST_CARBS_G,
    }


def observed_dosing(fit: TwinFit, tl) -> dict[str, float]:
    """Total daily insulin and the grams per unit this person actually bolused for meals."""
    days = [tl.day_dates.index(d) for d in fit.fitted_days]
    daily = []
    for day in days:
        s0, s1 = tl.day_steps(day)
        basal = np.nansum(tl.basal_upm[s0:s1]) * ode.DT_MIN
        bolus = np.nansum(tl.bolus_upm[s0:s1]) * ode.DT_MIN
        if np.isfinite(basal + bolus) and basal + bolus > 0:
            daily.append(basal + bolus)
    fitted = {s for day in days for s in range(*tl.day_steps(day))}
    width = int(BOLUS_MEAL_WINDOW_MIN / ode.DT_MIN)
    ratios = []
    for m in tl.meals:
        if not m.logged or m.grams <= 0 or m.step not in fitted:
            continue
        units = float(np.nansum(tl.bolus_upm[max(0, m.step - width): m.step + width]) * ode.DT_MIN)
        if units > 0.2:
            ratios.append(m.grams / units)
    return {
        "tdd_u": float(np.median(daily)) if daily else float("nan"),
        "observed_g_per_u": float(np.median(ratios)) if ratios else float("nan"),
        "n_meals_with_bolus": len(ratios),
    }


def one(twin: str, records: str, settings: str | None, max_windows: int) -> dict:
    fit = TwinFit.load(twin)
    recs = load_records(records)
    max_days = fit.diagnostics.get("records_max_days") or 0
    if max_days and len(recs) > max_days:
        recs = recs[-max_days:]
    tl = shift_events(build_timeline(recs, fit.person_id, load_therapy_settings(settings) if settings else None), fit.event_clock_offset_min)
    anchors = detect_unlogged_meals(tl, use_cgm_rises=False)
    windows = quiet_starts(tl, fit, max_windows)
    per_window = [dose_response(fit, tl, s, anchors) for s in windows]
    obs = observed_dosing(fit, tl)
    row = {"person_id": fit.person_id, "twin": twin, "n_windows": len(per_window), **obs}
    if np.isfinite(obs["tdd_u"]) and obs["tdd_u"] > 0:
        row["rule_isf_1800"], row["rule_cr_500"] = 1800.0 / obs["tdd_u"], 500.0 / obs["tdd_u"]
    if per_window:
        isf = np.array([w["isf_mgdl_per_u"] for w in per_window])
        csf = np.array([w["csf_mgdl_per_g"] for w in per_window])
        row.update({
            "implied_isf_mgdl_per_u": float(np.median(isf)), "implied_isf_iqr": [float(np.percentile(isf, 25)), float(np.percentile(isf, 75))],
            "implied_csf_mgdl_per_g": float(np.median(csf)),
            "implied_cr_g_per_u": float(np.median(isf) / np.median(csf)) if np.median(csf) > 0 else float("nan"),
        })
        if np.isfinite(obs["tdd_u"]) and obs["tdd_u"] > 0:
            row["isf_vs_rule"] = row["implied_isf_mgdl_per_u"] / row["rule_isf_1800"]
        if np.isfinite(obs["observed_g_per_u"]) and row.get("implied_cr_g_per_u"):
            row["cr_vs_observed"] = row["implied_cr_g_per_u"] / obs["observed_g_per_u"]
    row["windows"] = per_window
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", action="append", required=True)
    ap.add_argument("--records", action="append", required=True)
    ap.add_argument("--settings", action="append", default=[])
    ap.add_argument("--max-windows", type=int, default=12, help="quiet windows per person")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = []
    for i, (twin, records) in enumerate(zip(args.twin, args.records)):
        settings = args.settings[i] if i < len(args.settings) else None
        try:
            rows.append(one(twin, records, settings, args.max_windows))
        except Exception as exc:  # keep going across people
            rows.append({"twin": twin, "error": f"{type(exc).__name__}: {exc}"[:200]})
        r = rows[-1]
        if "error" in r:
            print(f"{Path(twin).parent.name:12s} ERROR {r['error']}")
        else:
            print(f"{r['person_id']:14s} windows {r['n_windows']:2d}  TDD {r['tdd_u']:5.1f} U  "
                  f"implied ISF {r.get('implied_isf_mgdl_per_u', float('nan')):6.1f} (rule {r.get('rule_isf_1800', float('nan')):6.1f}, "
                  f"x{r.get('isf_vs_rule', float('nan')):.2f})  implied CR {r.get('implied_cr_g_per_u', float('nan')):5.1f} g/U "
                  f"(they bolused {r['observed_g_per_u']:5.1f} g/U over {r['n_meals_with_bolus']} meals)", flush=True)
    ok = [r for r in rows if "error" not in r and "implied_isf_mgdl_per_u" in r]
    if len(ok) > 2:
        isf = np.array([r["implied_isf_mgdl_per_u"] for r in ok])
        rule = np.array([r.get("rule_isf_1800", np.nan) for r in ok])
        cr = np.array([r.get("implied_cr_g_per_u", np.nan) for r in ok])
        seen = np.array([r["observed_g_per_u"] for r in ok])
        pair = lambda a, b: float(np.corrcoef(a[m], b[m])[0, 1]) if (m := np.isfinite(a) & np.isfinite(b)).sum() > 2 else float("nan")
        print(f"\n{len(ok)} people")
        print(f"  implied ISF   median {np.median(isf):6.1f} mg/dL/U   vs 1800 rule: correlation {pair(isf, rule):+.2f}, median ratio {np.nanmedian(isf / rule):.2f}")
        print(f"  implied CR    median {np.nanmedian(cr):6.1f} g/U      vs what they bolused: correlation {pair(cr, seen):+.2f}, median ratio {np.nanmedian(cr / seen):.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
