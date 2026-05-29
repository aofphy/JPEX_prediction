"""Extended-base-learner QRA-rolling ablation (response to R1 M3, R3 X4, DA C3).

Reruns the QRA-rolling pipeline with two extended base-learner sets:
  EXT3 = {LEAR, DDNN-Normal mu, DDNN-JSU Q50}
  EXT4 = {LEAR, DDNN-Normal mu, DDNN-JSU Q50, N-HiTS}

Compared to the headline QRA-rolling (LEAR + DDNN-Normal only) to test how
much of the QRA-rolling advantage depends on the deliberate exclusion of
the strongest individual probabilistic forecasters from the base set.

Same 90-day rolling calibration window and 7-day refit cadence as
run_qra_rolling.py.

Saves results/qra_rolling_ext_metrics.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area
from feature_build import build_targets, HORIZON
from models_qra import QRA
from metrics_prob import pinball_loss, crps_quantile, winkler_score, coverage

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")
TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")
CALIB_DAYS = 90
RECAL_DAYS = 7
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def load_base_learner_matrix(area: str, market: str, names: list[str]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Load and align all requested base learners; returns common index + matrices."""
    out: dict[str, np.ndarray | pd.DataFrame] = {}
    if "LEAR" in names:
        out["LEAR"] = pd.read_csv(os.path.join(RES, f"lear_{area}_{market}_preds.csv"),
                                  parse_dates=[0], index_col=0)
    if "DDNN-Normal" in names:
        out["DDNN-Normal"] = pd.read_csv(os.path.join(RES, f"ddnn_{area}_{market}_mu.csv"),
                                         parse_dates=[0], index_col=0)
    if "N-HiTS" in names:
        out["N-HiTS"] = pd.read_csv(os.path.join(RES, f"nhits_{area}_{market}_preds.csv"),
                                    parse_dates=[0], index_col=0)
    if "DDNN-JSU" in names:
        arr = np.load(os.path.join(RES, f"ddnn_jsu_{area}_{market}_quantiles.npz"), allow_pickle=True)
        q_idx = pd.DatetimeIndex(arr["index"]); qs = arr["qs"]; Q = arr["quantiles"]
        q50_i = int(np.argmin(np.abs(qs - 0.5)))
        out["DDNN-JSU"] = pd.DataFrame(Q[:, :, q50_i], index=q_idx,
                                       columns=[f"h{h}" for h in range(1, HORIZON+1)])
    # Common index
    common = None
    for df in out.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {k: v.loc[common].values for k, v in out.items()}
    return common, aligned


def run_one(area: str, market: str, base_names: list[str], tag: str):
    print(f"\n=== {area} {market} QRA-rolling [{tag}: {base_names}] ===")
    common_pred, B = load_base_learner_matrix(area, market, base_names)
    Y_full = build_targets(load_area(area)[market], HORIZON)
    common = common_pred.intersection(Y_full.dropna().index)
    B = {k: pd.DataFrame(v, index=common_pred).loc[common].values for k, v in B.items()}
    Y_arr = Y_full.loc[common].values

    eval_start = TEST_START + pd.Timedelta(days=CALIB_DAYS)
    issuance = common[(common >= eval_start) & (common <= TEST_END)]
    n_eval = len(issuance)
    print(f"  Eval issuance n={n_eval}; n_base={len(base_names)}")

    Q_pred = np.zeros((n_eval, HORIZON, len(QS)))
    Y_eval = np.zeros((n_eval, HORIZON))
    block_step = RECAL_DAYS * 48
    pos = 0; block_id = 0; t0 = time.time()
    while pos < n_eval:
        block_idx = issuance[pos : pos + block_step]
        if len(block_idx) == 0: break
        block_start = block_idx[0]
        cal_start = block_start - pd.Timedelta(days=CALIB_DAYS)
        cal_mask = (common >= cal_start) & (common < block_start)
        block_positions = np.where(np.isin(common, block_idx))[0]
        for h in range(HORIZON):
            F_cal = np.column_stack([B[name][cal_mask, h] for name in base_names])
            y_cal = Y_arr[cal_mask, h]
            qra = QRA(qs=QS.tolist(), alpha=0.0).fit(F_cal, y_cal)
            F_eval = np.column_stack([B[name][block_positions, h] for name in base_names])
            block_eval_global = np.arange(pos, pos + len(block_positions))
            Q_pred[block_eval_global, h, :] = qra.predict(F_eval)
            Y_eval[block_eval_global, h] = Y_arr[block_positions, h]
        if block_id % 10 == 0:
            print(f"    block {block_id:02d}  start={block_start}  elapsed={time.time()-t0:.0f}s")
        pos += block_step; block_id += 1

    pin = pinball_loss(Y_eval, Q_pred, QS)
    crps = crps_quantile(Y_eval, Q_pred, QS)
    cov = {}; widths = {}; wsc = {}
    pi_map = {0.1: (0.05, 0.95), 0.2: (0.10, 0.90), 0.5: (0.25, 0.75)}
    for a, (lo_q, hi_q) in pi_map.items():
        li = int(np.argmin(np.abs(QS - lo_q)))
        ui = int(np.argmin(np.abs(QS - hi_q)))
        lo = Q_pred[..., li]; hi = Q_pred[..., ui]
        cov[f"alpha_{a:.2f}"] = coverage(Y_eval, lo, hi)
        widths[f"alpha_{a:.2f}"] = float(np.mean(hi - lo))
        wsc[f"alpha_{a:.2f}"] = winkler_score(Y_eval, lo, hi, alpha=a)
    rmse_med = float(np.sqrt(((Q_pred[..., 3] - Y_eval) ** 2).mean()))
    print(f"  pinball={pin:.3f}  CRPS={crps:.3f}  Q50_RMSE={rmse_med:.3f}  cov90={cov['alpha_0.10']:.3f}  Winkler90={wsc['alpha_0.10']:.3f}")
    return {"n_eval": int(n_eval), "base_learners": base_names,
            "pinball": pin, "CRPS": crps, "Q50_RMSE": rmse_med,
            "coverage": cov, "mean_width": widths, "winkler": wsc}


def main():
    out: dict = {}
    SETS = {
        "EXT3_LEAR_DDNN_DDNN-JSU":              ["LEAR", "DDNN-Normal", "DDNN-JSU"],
        "EXT4_LEAR_DDNN_DDNN-JSU_NHITS":         ["LEAR", "DDNN-Normal", "DDNN-JSU", "N-HiTS"],
    }
    for tag, base_names in SETS.items():
        out[tag] = {}
        for area in ["TK", "KS"]:
            for market in ["DA", "IM"]:
                out[tag][f"{area}_{market}"] = run_one(area, market, base_names, tag)
    with open(os.path.join(RES, "qra_rolling_ext_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== Summary ===")
    for tag in SETS:
        print(f"\n  [{tag}]")
        for k, v in out[tag].items():
            w90 = v["winkler"]["alpha_0.10"]; c90 = v["coverage"]["alpha_0.10"]
            print(f"    {k}  CRPS={v['CRPS']:.3f}  Q50_RMSE={v['Q50_RMSE']:.3f}  cov90={c90:.3f}  Winkler90={w90:.3f}")


if __name__ == "__main__":
    main()
