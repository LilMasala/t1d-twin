"""Fine-tune a T1DSim_AI digital twin per participant and save its sequence simulations.

Runs in the dedicated Python 3.10 environment (T1DSim_AI pins torch 1.13):

    .venv-t1dsim-ai/bin/python scripts/benchmark_t1dsimai_train.py --cohort uom --pids 2301,2307,2308,2309

Uses the package's published training recipe unchanged (trainDigitalTwin.py:
128-128-64-32 individual network, lr 1e-4, batch 32, 150 epochs, 90% overlap,
5-hour sequences). After training, both the train (fitted-day) and test
(held-out-day) sequences selected by the package's own ``SequenceSelection`` are
simulated with the population model and with the personal twin; each sequence's
start time, actual CGM and simulated CGM go to ``sequences_<group>.npz`` for
scoring against our twin on identical sequences.

Research benchmark only (T1DSim_AI license: non-profit / academic research use).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/external/T1DSim_AI/example"))

from t1dsim_ai.individual_model import IndividualModel, SequenceSelection  # noqa: E402
from t1dsim_ai.options import hidden_compartments  # noqa: E402
from t1dsim_ai.utils.preprocess import scale_inverse_Q1  # noqa: E402

LR, BATCH, EPOCHS, OVERLAP, SEQ_LEN = 1e-4, 32, 150, 0.9, 60


def run(cohort: str, pid: str, lr: float = LR, tag: str = "", pretrained: str = "") -> dict:
    data_dir = ROOT / "artifacts/benchmark/t1dsimai" / cohort / pid
    out_dir = data_dir / tag if tag else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "data.csv")
    df["is_train"] = df["is_train"].astype(bool)
    test_rows = df[~df.is_train & df.output_cgm.notna()]
    if df.loc[df.is_train, "input_meal_carbs"].sum() < 1 or test_rows.empty or df.loc[~df.is_train, "input_meal_carbs"].sum() < 1:
        return {"pid": pid, "skipped": "T1DSim_AI needs logged carbs in both train and test periods"}

    np.random.seed(0)
    torch.manual_seed(0)
    t0 = time.time()
    model_dir = str(out_dir) + "/"
    ind = IndividualModel("twin", df.copy(), model_dir)
    ind.setup_nn(hidden_compartments, lr, BATCH, EPOCHS, OVERLAP, SEQ_LEN + 1)
    if pretrained:  # score a twin the authors already trained (no training here)
        ind.individual_model.load_state_dict(torch.load(Path(pretrained) / "individual_model.pt"))
        with open(Path(pretrained) / "scaler_robust.pkl", "rb") as fh:
            import pickle
            ind.scaler_featsRobust = pickle.load(fh)
        score = float("nan")
    else:
        score = ind.fit(True)
        if score is np.nan or score != score:
            return {"pid": pid, "error": "training diverged (package reported NaN)"}

    result = {"pid": pid, "lr": lr, "tag": tag, "pretrained": pretrained, "train_seconds": round(time.time() - t0, 1), "train_rmse_package": float(score)}
    groups = {
        "train": ([ind.x_est_train, ind.u_pop_train, ind.y_id_train, ind.u_ind_train, ind.cgm_real_train], df[df.is_train]),
        "test": ([ind.x_est_test, ind.u_pop_test, ind.y_id_test, ind.u_ind_test, ind.cgm_real_test], df[~df.is_train]),
    }
    for group, (data, rows) in groups.items():
        sel = SequenceSelection(SEQ_LEN + 1, "cpu", data)
        starts = []
        # SequenceSelection keeps start indices only implicitly; recompute them with its own rule
        y = data[2][0, :, 0]
        idx = 0
        while idx <= y.shape[0] - (SEQ_LEN + 1):
            arr = y[idx: idx + SEQ_LEN + 1]
            if not np.isnan(arr[0]):
                ok = not np.isnan(arr).any()
                if not ok:
                    ok = sum(np.sum(~np.isnan(arr[1: jj + 1])) / jj >= 0.7 for jj in np.arange(6, SEQ_LEN + 1, 6)) == 10
                if ok:
                    starts.append(idx)
                idx += SEQ_LEN + 1
            else:
                idx += 1
        if len(starts) != len(sel.idx_scenarios):
            raise RuntimeError(f"{pid} {group}: start recomputation disagrees with the package ({len(starts)} vs {len(sel.idx_scenarios)})")
        with torch.no_grad():
            x0, u_pop, u_ind, _, x_orig = sel.get_all("all")
            sim_pop = ind.nn_solution(x0, u_pop, u_ind, False)
            sim_dt = ind.nn_solution(x0, u_pop, u_ind, True)
        to_mgdl = lambda t: scale_inverse_Q1(t[:, :, [0]].clone(), ind.popModelFolder).numpy()[:, :, 0].T  # [seq, steps]
        np.savez(out_dir / f"sequences_{group}.npz",
                 start_time_utc=np.array(rows["time_utc"].iloc[starts].tolist()),
                 actual=to_mgdl(x_orig), t1dsimai_twin=to_mgdl(sim_dt), t1dsimai_population=to_mgdl(sim_pop))
        result[f"n_seq_{group}"] = len(starts)
    (out_dir / "t1dsimai_result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--lr", type=float, default=LR, help="published recipe: 1e-4")
    ap.add_argument("--tag", default="", help="output subfolder for non-default runs, e.g. lr1e-3")
    ap.add_argument("--pretrained", default="", help="folder with individual_model.pt + scaler_robust.pkl to score instead of training")
    args = ap.parse_args()
    for pid in args.pids.split(","):
        try:
            print(json.dumps(run(args.cohort, pid, args.lr, args.tag, args.pretrained)), flush=True)
        except Exception as exc:  # keep going across participants
            print(json.dumps({"pid": pid, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
