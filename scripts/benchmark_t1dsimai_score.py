"""Score our twin and T1DSim_AI on identical 5-hour held-out sequences.

    python scripts/benchmark_t1dsimai_score.py \
        --cohort uom --pids 2301,2307,2308,2309

For every sequence T1DSim_AI evaluated (``sequences_test.npz``), our twin runs
over the same 5 hours from recorded insulin and logged meals, with day/meal
unknowns at the person's typical values (carb-count bias, typical announcement
size, time-of-day disturbance profile, no drift), point estimate (posterior mean):

- ``ours_cold``: starts at the sequence's first CGM from a basal steady state —
  the same information T1DSim_AI gets;
- ``ours_warm``: also replays the 6 hours of recorded inputs before the sequence
  (insulin and carbs on board), then re-anchors glucose to the sequence's first
  CGM reading — the standard forecasting setup. T1DSim_AI cannot use history.
- ``ours_assim``: warm, plus the disturbance fitted to the 3 hours of CGM before
  the sequence (``t1d_twin.forecast``), its last value carried into the
  forecast with a 30-min half-life. ``ours_assim_nocarry`` is the ablation without
  the carry (the recent CGM only shapes the state at the origin).

Low-event scoring: a low event is >= 3 consecutive readings < 70 mg/dL (15 min,
level-1 hypoglycaemia) within steps 1-60. T1DSim_AI is deterministic (event yes/no);
our twin gives a probability from a posterior ensemble (parameter draws, daily
SI/EGP drift at the fitted spread, CGM noise), warm start. Reported: events,
sensitivity/PPV at p >= 0.5, Brier score and AUROC.

Metrics follow T1DSim_AI's own evaluation: per-sequence RMSE over steps 1-60
(NaN readings ignored), averaged over sequences; per-sequence TIR/TBR/TAR of the
simulated trace, averaged, against the same for the actual CGM. Also RMSE at 30,
60 and 120 minutes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from t1d_twin import ode
from t1d_twin.data import build_timeline, detect_unlogged_meals, load_records, shift_events
from t1d_twin.fit import TwinFit
from t1d_twin.forecast import assimilate, carried_disturbance, typical_flux
from t1d_twin.model import DTYPE, rollout
from t1d_twin.params import UNLOGGED_LOG_G_PRIOR

ROOT = Path(__file__).resolve().parents[1]
SEQ_STEPS_5MIN = 60
BURN_STEPS = int(6 * 60 / ode.DT_MIN)


LOW_MGDL, LOW_RUN = 70.0, 3
ENSEMBLE = 32


def has_low_event(trace: np.ndarray) -> bool:
    run = 0
    for v in trace[1:]:
        run = run + 1 if (not np.isnan(v) and v < LOW_MGDL) else 0
        if run >= LOW_RUN:
            return True
    return False


def _inputs(fit: TwinFit, tl, start: int, n: int, B: int, anchors):
    bias = float(np.exp(fit.carb_count_bias))
    anchor_g = float(np.exp(np.median(fit.unlogged_log_g_loc))) if fit.unlogged_log_g_loc else float(np.exp(UNLOGGED_LOG_G_PRIOR[0]))
    meals = [(m.step, torch.full((B,), m.grams * bias, dtype=DTYPE)) for m in tl.meals if m.logged]
    meals += [(a.step, torch.full((B,), anchor_g, dtype=DTYPE)) for a in anchors]
    sl = slice(start, start + n)
    ins = torch.tensor(np.nan_to_num(tl.basal_upm[sl]) + tl.bolus_upm[sl], dtype=DTYPE)[None].expand(B, -1)
    bol = torch.tensor(tl.bolus_upm[sl], dtype=DTYPE)[None].expand(B, -1)
    return meals, ins, bol


def window(seq_start_step: int, warm: bool) -> tuple[int, int]:
    start = max(0, seq_start_step - BURN_STEPS if warm else seq_start_step)
    return start, seq_start_step - start + int(SEQ_STEPS_5MIN * 5 / ode.DT_MIN) + 1


def assimilate_sequence(fit: TwinFit, tl, seq_start_step: int, anchors):
    start, n = window(seq_start_step, True)
    meals, ins, bol = _inputs(fit, tl, start, n, 1, anchors)
    return assimilate(fit, tl, start, seq_start_step, meals=meals, insulin_upm=ins, bolus_upm=bol)


def simulate_ours(fit: TwinFit, tl, seq_start_step: int, warm: bool, anchors, first_cgm: float, ensemble: int = 0,
                  assim=None, carry: str = "hour", anchor_origin: bool = True) -> np.ndarray:
    """Glucose at 5-min marks 0..60 of the sequence (mg/dL); [61] point estimate, or [ensemble, 61] with ``ensemble``.

    ``assim``: the result of ``assimilate_sequence`` (warm only). ``carry``: which fitted
    disturbance continues into the forecast — "last" step, mean of the last "hour", or "none".
    ``anchor_origin``: re-anchor glucose to the first reading (else start from the assimilated state).
    """
    start, n = window(seq_start_step, warm)
    B = max(1, ensemble)
    gen = torch.Generator().manual_seed(seq_start_step)
    u = fit.sample_globals(B, gen) if ensemble else torch.tensor(fit.loc, dtype=DTYPE)[None]
    names = fit.param_names
    day_si = day_egp = None
    if ensemble:
        day_si = torch.exp(u[:, names.index("log_day_si_sd")])[:, None] * torch.randn(B, tl.n_days, generator=gen, dtype=DTYPE)
        day_egp = torch.exp(u[:, names.index("log_day_egp_sd")])[:, None] * torch.randn(B, tl.n_days, generator=gen, dtype=DTYPE)
    meals, ins, bol = _inputs(fit, tl, start, n, B, anchors)
    offset = seq_start_step - start
    flux = typical_flux(fit, tl, start, n)
    resets = [(offset, first_cgm)] if warm else []
    if assim is not None:
        assim_resets, disturbance = assim
        flux = flux.clone()
        flux[0, :offset] += disturbance
        if carry != "none":
            recent = disturbance[-(1 if carry == "last" else int(60 / ode.DT_MIN)):]
            flux[0, offset:] += carried_disturbance(float(recent.mean()), n - offset)
        resets = assim_resets + (resets if anchor_origin else [])
    with torch.no_grad():
        g = rollout(tl, fit.base, fit.priors(), u, start, n, meal_grams=meals, insulin_upm=ins, bolus_upm=bol,
                    flux_upm=flux.expand(B, -1), day_log_si=day_si, day_log_egp=day_egp, glucose_reset=resets or None).numpy()
    marks = [offset + int(round(5 * k / ode.DT_MIN)) - 1 for k in range(SEQ_STEPS_5MIN + 1)]
    marks[0] = max(marks[0], 0)
    if not ensemble:
        return g[0, marks]
    traces = g[:, marks]
    cgm_sd = torch.exp(u[:, names.index("log_cgm_sd")]).numpy()[:, None]
    noise = np.random.default_rng(seq_start_step).standard_normal(traces.shape)
    return traces + np.sqrt(cgm_sd ** 2 + (0.06 * traces) ** 2) * noise


def seq_metrics(sim: np.ndarray, actual: np.ndarray) -> dict:
    s, a = sim[:, 1:], actual[:, 1:]
    rmse = np.sqrt(np.nanmean((s - a) ** 2, axis=1))
    per_seq = rmse.tolist()
    horizon = lambda minutes: float(np.sqrt(np.nanmean((sim[:, minutes // 5] - actual[:, minutes // 5]) ** 2)))
    frac = lambda x, lo, hi: np.nanmean(((x >= lo) & (x <= hi)).astype(float), axis=1)
    return {
        "rmse": float(np.nanmean(rmse)),
        "rmse_30": horizon(30), "rmse_60": horizon(60), "rmse_120": horizon(120),
        "tir": float(np.mean(frac(s, 70, 180))), "tbr": float(np.mean(frac(s, -1e9, 69.999))), "tar": float(np.mean(frac(s, 180.001, 1e9))),
        "per_seq_tir_abs_err": float(np.mean(np.abs(frac(s, 70, 180) - frac(np.nan_to_num(a, nan=np.nan), 70, 180)))),
        "per_seq_rmse": per_seq,
    }


def score_one(cohort: str, pid: str, group: str = "test", tuned_tag: str = "lr1e-3", twin_file: str = "twin_final.json") -> dict:
    bench = ROOT / "artifacts/benchmark/t1dsimai" / cohort / pid
    npz_path = bench / f"sequences_{group}.npz"
    tuned_path = bench / tuned_tag / f"sequences_{group}.npz"
    if not npz_path.exists():
        return {"pid": pid, "skipped": "no T1DSim_AI sequences (not trained or not eligible)"}
    z = np.load(npz_path)
    export = json.loads((bench / "export.json").read_text())
    fit = TwinFit.load(ROOT / "artifacts/twin" / cohort / pid / twin_file)
    records = load_records(ROOT / "artifacts/twin" / cohort / pid / "raw_days")
    if export.get("max_days") and len(records) > export["max_days"]:
        records = records[-export["max_days"]:]
    tl = shift_events(build_timeline(records, fit.person_id), fit.event_clock_offset_min)
    anchors = detect_unlogged_meals(tl, use_cgm_rises=False)

    cold, warm, assim, assim_nocarry, p_low, p_low_assim = [], [], [], [], [], []
    actual = z["actual"]
    for i, ts in enumerate(z["start_time_utc"]):
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        step = int(round((t - tl.t0_utc).total_seconds() / 60.0 / ode.DT_MIN))
        cold.append(simulate_ours(fit, tl, step, False, anchors, float(actual[i, 0])))
        warm.append(simulate_ours(fit, tl, step, True, anchors, float(actual[i, 0])))
        ens = simulate_ours(fit, tl, step, True, anchors, float(actual[i, 0]), ensemble=ENSEMBLE)
        p_low.append(float(np.mean([has_low_event(tr) for tr in ens])))
        fitted = assimilate_sequence(fit, tl, step, anchors)
        assim.append(simulate_ours(fit, tl, step, True, anchors, float(actual[i, 0]), assim=fitted))
        assim_nocarry.append(simulate_ours(fit, tl, step, True, anchors, float(actual[i, 0]), assim=fitted, carry="none"))
        ens = simulate_ours(fit, tl, step, True, anchors, float(actual[i, 0]), ensemble=ENSEMBLE, assim=fitted)
        p_low_assim.append(float(np.mean([has_low_event(tr) for tr in ens])))
    act = seq_metrics(actual, actual)
    res = {"pid": pid, "group": group, "n_sequences": int(actual.shape[0]),
           "actual": {k: act[k] for k in ("tir", "tbr", "tar")},
           "t1dsimai_twin": seq_metrics(z["t1dsimai_twin"], actual),
           "t1dsimai_population": seq_metrics(z["t1dsimai_population"], actual),
           **({"t1dsimai_twin_tuned": seq_metrics(np.load(tuned_path)["t1dsimai_twin"], actual)} if tuned_path.exists() else {}),
           "ours_cold": seq_metrics(np.array(cold), actual),
           "ours_warm": seq_metrics(np.array(warm), actual),
           "ours_assim": seq_metrics(np.array(assim), actual),
           "ours_assim_nocarry": seq_metrics(np.array(assim_nocarry), actual),
           "low_events": {
               "actual": [has_low_event(tr) for tr in actual],
               "ours_warm_prob": p_low,
               "ours_assim_prob": p_low_assim,
               "t1dsimai_twin": [has_low_event(tr) for tr in z["t1dsimai_twin"]],
               **({"t1dsimai_twin_tuned": [has_low_event(tr) for tr in np.load(tuned_path)["t1dsimai_twin"]]} if tuned_path.exists() else {}),
           }}
    res["twin_file"] = twin_file
    (bench / f"score_{group}_{Path(twin_file).stem}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--group", default="test", choices=["test", "train"])
    ap.add_argument("--twin-file", default="twin_final.json", help="which of our fits to score (e.g. twin_hypo.json)")
    args = ap.parse_args()
    rows = [score_one(args.cohort, p, args.group, twin_file=args.twin_file) for p in args.pids.split(",")]
    models = ["t1dsimai_population", "t1dsimai_twin", "ours_cold", "ours_warm", "ours_assim", "ours_assim_nocarry"]
    if any("t1dsimai_twin_tuned" in r for r in rows):
        models.insert(2, "t1dsimai_twin_tuned")
    for r in rows:
        if "skipped" in r:
            print(r["pid"], r["skipped"])
            continue
        a = r["actual"]
        print(f"\n{r['pid']} ({r['group']}, {r['n_sequences']} five-hour sequences)  actual TIR {100*a['tir']:.1f} TBR {100*a['tbr']:.1f} TAR {100*a['tar']:.1f}")
        for m in models:
            s = r[m]
            print(f"  {m:20s} RMSE {s['rmse']:5.1f} (30m {s['rmse_30']:5.1f} 60m {s['rmse_60']:5.1f} 120m {s['rmse_120']:5.1f})  "
                  f"TIR {100*s['tir']:5.1f} TBR {100*s['tbr']:4.1f} TAR {100*s['tar']:5.1f}  per-seq |TIR err| {100*s['per_seq_tir_abs_err']:5.1f}")
    ok = [r for r in rows if "skipped" not in r and all(m in r for m in models)]
    if ok:
        print("\npooled over participants (sequence-weighted)")
        w = np.array([r["n_sequences"] for r in ok], dtype=float)
        for m in models:
            avg = lambda k: float(np.average([r[m][k] for r in ok], weights=w))
            tir_err = float(np.average([abs(r[m]["tir"] - r["actual"]["tir"]) for r in ok], weights=w))
            print(f"  {m:20s} RMSE {avg('rmse'):5.1f}  30m {avg('rmse_30'):5.1f}  60m {avg('rmse_60'):5.1f}  120m {avg('rmse_120'):5.1f}  "
                  f"|TIR - actual| {100*tir_err:5.1f}pp  per-seq |TIR err| {100*avg('per_seq_tir_abs_err'):5.1f}pp")
        baselines = [m for m in models if m.startswith("t1dsimai")]
        paired_stats(ok, baselines)
        paired_stats(ok, baselines + ["ours_warm"], ours="ours_assim")
        low_event_report(ok)


def low_event_report(rows: list[dict]) -> None:
    """Pooled low-event prediction across sequences."""
    from sklearn.metrics import roc_auc_score

    actual = np.concatenate([r["low_events"]["actual"] for r in rows]).astype(bool)
    print(f"\nlow events (>=15 min < 70 mg/dL) in {len(actual)} sequences: {int(actual.sum())} with a low")
    preds = {"ours_warm (p)": np.concatenate([r["low_events"]["ours_warm_prob"] for r in rows])}
    if all("ours_assim_prob" in r["low_events"] for r in rows):
        preds["ours_assim (p)"] = np.concatenate([r["low_events"]["ours_assim_prob"] for r in rows])
    for m in ("t1dsimai_twin", "t1dsimai_twin_tuned"):
        if all(m in r["low_events"] for r in rows):
            preds[m] = np.concatenate([r["low_events"][m] for r in rows]).astype(float)
    for name, p in preds.items():
        hit = p >= 0.5
        sens = float((hit & actual).sum() / max(1, actual.sum()))
        ppv = float((hit & actual).sum() / max(1, hit.sum())) if hit.any() else float("nan")
        brier = float(np.mean((p - actual) ** 2))
        auc = float(roc_auc_score(actual, p)) if 0 < actual.sum() < len(actual) else float("nan")
        print(f"  {name:22s} predicted lows {int(hit.sum()):3d}  sensitivity {sens:5.2f}  PPV {ppv:5.2f}  Brier {brier:6.3f}  AUROC {auc:5.2f}")


def paired_stats(rows: list[dict], baselines: list[str], ours: str = "ours_warm") -> None:
    """Per-sequence RMSE, ``ours`` vs each baseline in ``models``: Wilcoxon signed-rank + participant-cluster bootstrap CI."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(0)
    print(f"\npaired per-sequence RMSE difference ({ours} minus baseline; negative = {ours} better)")
    for base in baselines:
        diffs = [np.array(r[ours]["per_seq_rmse"]) - np.array(r[base]["per_seq_rmse"]) for r in rows]
        flat = np.concatenate(diffs)
        ok = ~np.isnan(flat)
        stat = wilcoxon(flat[ok]) if ok.sum() > 5 else None
        boots = []
        for _ in range(4000):  # resample participants, then sequences within them
            pick = rng.integers(0, len(diffs), len(diffs))
            sample = np.concatenate([rng.choice(diffs[i], len(diffs[i])) for i in pick])
            boots.append(np.nanmean(sample))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wins = float(np.mean(flat[ok] < 0))
        print(f"  vs {base:20s} n={ok.sum():3d} seqs  mean {np.nanmean(flat):+6.1f} mg/dL  95% CI [{lo:+6.1f}, {hi:+6.1f}]  "
              f"ours better on {100*wins:4.0f}% of sequences  Wilcoxon p={stat.pvalue if stat else float('nan'):.3g}")


if __name__ == "__main__":
    main()
