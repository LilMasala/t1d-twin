"""Pooled benchmark statistics from saved score files (no simulation).

    python scripts/benchmark_t1dsimai_report.py \
        --cohort hupa --twin-file twin_final.json --out artifacts/benchmark/t1dsimai/hupa/report_twin_final.json

Reads ``score_test_<twin>.json`` for every participant in the cohort and reports:

- per-model pooled RMSE (sequence-weighted) and RMSE at 30/60/120 min;
- paired per-sequence RMSE differences, ours minus each baseline, with a
  participant-cluster bootstrap 95% CI and Wilcoxon signed-rank p;
- low events (>= 15 min < 70 mg/dL): AUROC, Brier, sensitivity/PPV at p >= 0.5 for
  our ensemble probabilities, T1DSim_AI's deterministic yes/no, and T1DSim_AI with
  the same CGM noise model added (``t1dsimai_twin_tuned+noise``: 32 noisy copies of
  its trace, noise sd from our fit, so its events become probabilities too);
  AUROC / Brier differences with participant-cluster bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from t1d_twin.fit import CGM_CV, TwinFit

ROOT = Path(__file__).resolve().parents[1]
BOOT = 4000
NOISE_COPIES = 32
LOW_MGDL, LOW_RUN = 70.0, 3


def has_low_event(trace: np.ndarray) -> bool:
    run = 0
    for v in trace[1:]:
        run = run + 1 if (not np.isnan(v) and v < LOW_MGDL) else 0
        if run >= LOW_RUN:
            return True
    return False


def noisy_event_prob(traces: np.ndarray, cgm_sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for tr in traces:
        sd = np.sqrt(cgm_sd ** 2 + (CGM_CV * tr) ** 2)
        out.append(np.mean([has_low_event(tr + sd * rng.standard_normal(tr.shape)) for _ in range(NOISE_COPIES)]))
    return np.array(out)


def cluster_boot(groups: list[np.ndarray], stat, rng) -> tuple[float, float]:
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        v = stat(np.concatenate([groups[i] for i in pick]))
        if np.isfinite(v):
            vals.append(v)
    return tuple(float(x) for x in np.percentile(vals, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--twin-file", default="twin_final.json")
    ap.add_argument("--ours", default="ours_warm,ours_assim", help="our variants to compare (those present in the score files)")
    ap.add_argument("--out")
    args = ap.parse_args()
    stem = Path(args.twin_file).stem
    rows = []
    for path in sorted((ROOT / "artifacts/benchmark/t1dsimai" / args.cohort).glob(f"*/score_test_{stem}.json")):
        r = json.loads(path.read_text())
        if "skipped" not in r:
            rows.append(r)
    ours = [m for m in args.ours.split(",") if all(m in r for r in rows)]
    baselines = [m for m in ("t1dsimai_population", "t1dsimai_twin", "t1dsimai_twin_tuned") if all(m in r for r in rows)]
    rng = np.random.default_rng(0)
    report = {"cohort": args.cohort, "twin_file": args.twin_file, "participants": [r["pid"] for r in rows],
              "n_sequences": int(sum(r["n_sequences"] for r in rows)), "models": {}, "paired_rmse": {}, "low_events": {}}

    w = np.array([r["n_sequences"] for r in rows], dtype=float)
    for m in baselines + ["ours_cold"] + ours:
        if not all(m in r for r in rows):
            continue
        avg = lambda k: float(np.average([r[m][k] for r in rows], weights=w))
        report["models"][m] = {k: avg(k) for k in ("rmse", "rmse_30", "rmse_60", "rmse_120")}
        report["models"][m]["abs_tir_error_pp"] = 100 * float(np.average([abs(r[m]["tir"] - r["actual"]["tir"]) for r in rows], weights=w))

    for o in ours:
        for b in baselines + [x for x in ours if x != o]:
            diffs = [np.array(r[o]["per_seq_rmse"]) - np.array(r[b]["per_seq_rmse"]) for r in rows]
            flat = np.concatenate(diffs)
            ok = ~np.isnan(flat)
            lo, hi = cluster_boot(diffs, np.nanmean, rng)
            report["paired_rmse"][f"{o} - {b}"] = {"mean": float(np.nanmean(flat)), "ci95": [lo, hi], "n": int(ok.sum()),
                                                    "wilcoxon_p": float(wilcoxon(flat[ok]).pvalue), "ours_better_frac": float(np.mean(flat[ok] < 0))}

    # low events
    actual = [np.array(r["low_events"]["actual"], dtype=float) for r in rows]
    preds: dict[str, list[np.ndarray]] = {}
    for o in ours:
        key = f"{o}_prob"
        if all(key in r["low_events"] for r in rows):
            preds[f"{o} (p)"] = [np.array(r["low_events"][key]) for r in rows]
    for b in ("t1dsimai_twin", "t1dsimai_twin_tuned"):
        if all(b in r["low_events"] for r in rows):
            preds[b] = [np.array(r["low_events"][b], dtype=float) for r in rows]
    tuned = "lr1e-3"
    noisy = []
    for r in rows:
        bench = ROOT / "artifacts/benchmark/t1dsimai" / args.cohort / r["pid"]
        z = np.load(bench / tuned / "sequences_test.npz") if (bench / tuned / "sequences_test.npz").exists() else np.load(bench / "sequences_test.npz")
        fit = TwinFit.load(ROOT / "artifacts/twin" / args.cohort / r["pid"] / args.twin_file)
        cgm_sd = float(np.exp(fit.loc[fit.param_names.index("log_cgm_sd")]))
        noisy.append(noisy_event_prob(z["t1dsimai_twin"], cgm_sd, seed=len(noisy)))
    preds["t1dsimai_twin_tuned+noise (p)"] = noisy

    def auc(y, p):
        return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")

    y_all = np.concatenate(actual).astype(bool)
    report["low_events"]["n_with_low"] = int(y_all.sum())
    for name, p_groups in preds.items():
        p = np.concatenate(p_groups)
        hit = p >= 0.5
        report["low_events"][name] = {
            "auroc": auc(y_all, p), "brier": float(np.mean((p - y_all) ** 2)),
            "sensitivity": float((hit & y_all).sum() / max(1, y_all.sum())),
            "ppv": float((hit & y_all).sum() / hit.sum()) if hit.any() else float("nan"), "predicted_lows": int(hit.sum()),
        }
    # paired AUROC / Brier differences: resample participants, keep (actual, ours, baseline) aligned
    for o in [k for k in preds if k.startswith("ours")]:
        for b in [k for k in preds if k.startswith("t1dsimai")]:
            groups = [np.stack([actual[i], preds[o][i], preds[b][i]], axis=1) for i in range(len(rows))]
            d_auc = lambda s: auc(s[:, 0], s[:, 1]) - auc(s[:, 0], s[:, 2])
            d_brier = lambda s: float(np.mean((s[:, 1] - s[:, 0]) ** 2) - np.mean((s[:, 2] - s[:, 0]) ** 2))
            full = np.concatenate(groups)
            report["low_events"][f"{o} - {b}"] = {"auroc_diff": d_auc(full), "auroc_ci95": list(cluster_boot(groups, d_auc, rng)),
                                                  "brier_diff": d_brier(full), "brier_ci95": list(cluster_boot(groups, d_brier, rng))}

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
