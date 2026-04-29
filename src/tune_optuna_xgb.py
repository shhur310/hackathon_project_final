"""
tune_optuna_xgb.py — Optuna for XGBoost
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.v2_features import add_v2_all_features
from src.features.oof_target_encoder import OOFTargetEncoder

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Install optuna first: pip install optuna")
    raise

DATA = ROOT / "data"
MODELS = ROOT / "models"
TARGET = "임신 성공 여부"
ID_COL = "ID"
TARGET_ENC_COLS = ["시술 시기 코드", "특정 시술 유형", "배란 유도 유형", "배아 생성 주요 이유"]

def build_universe(X_, X_test_, cols):
    universe = {}
    for c in cols:
        s = pd.concat([X_[c], X_test_[c]]) if X_test_ is not None else X_[c]
        s = s.fillna('__NA__').astype(str)
        vals = sorted(s.unique().tolist())
        universe[c] = {v: i for i, v in enumerate(vals)}
    return universe

def encode_int(df, universe):
    df_e = df.copy()
    for c, mp in universe.items():
        s = df[c].astype(object)
        s = pd.Series(s).where(pd.notna(s), '__NA__').astype(str)
        df_e[c] = s.map(mp).fillna(-1).astype('int32').values
    return df_e

def evaluate_params(params: dict, X: pd.DataFrame, y: pd.Series, n_folds: int = 3) -> float:
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)
    enc = OOFTargetEncoder(cols=[c for c in TARGET_ENC_COLS if c in X.columns], smoothing=50.0)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        Xtr_b = X.iloc[tr].reset_index(drop=True)
        Xva_b = X.iloc[va].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True)
        yva = y.iloc[va].reset_index(drop=True)
        tr_te, va_te = enc.fit_transform_oof(Xtr_b, ytr, Xva_b)
        Xtr = pd.concat([Xtr_b, tr_te], axis=1)
        Xva = pd.concat([Xva_b, va_te], axis=1)
        # Build int universe from train+val (OK since fold internal)
        obj_cols = Xtr.select_dtypes(include=["object", "category"]).columns.tolist()
        universe = build_universe(Xtr, Xva, obj_cols)
        Xtr_int = encode_int(Xtr, universe)
        Xva_int = encode_int(Xva, universe)
        dtr = xgb.DMatrix(Xtr_int, label=ytr)
        dva = xgb.DMatrix(Xva_int, label=yva)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(dva, "va")],
                      early_stopping_rounds=150, verbose_eval=False)
        oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    return roc_auc_score(y, oof)

def main(n_trials: int = 30, subsample_rate: float = 1.0, n_folds: int = 3):
    t0 = time.time()
    print(f"[1/3] Load data + v2 features (XGB Optuna)", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    if subsample_rate < 1.0:
        train = train.sample(frac=subsample_rate, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(train)} rows", flush=True)
    train = add_v2_all_features(train)
    y = train[TARGET].astype(int).reset_index(drop=True)
    drop_cols = [TARGET, ID_COL] if ID_COL in train.columns else [TARGET]
    X = train.drop(columns=drop_cols).reset_index(drop=True)
    print(f"  X: {X.shape}", flush=True)

    print(f"\n[2/3] Optuna search ({n_trials} trials, {n_folds}-fold OOF)", flush=True)
    def objective(trial):
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.07, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        try:
            auc = evaluate_params(params, X, y, n_folds=n_folds)
            return auc
        except Exception as e:
            print(f"  Trial failed: {e}")
            return 0.0

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n[3/3] Best: {study.best_value:.5f}", flush=True)
    print(f"  Best params: {study.best_params}", flush=True)
    print(f"  Total time: {(time.time()-t0)/60:.1f} min", flush=True)

    summary = {
        "model": "xgb",
        "best_auc": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "subsample_rate": subsample_rate,
        "all_trial_aucs": [t.value for t in study.trials if t.value is not None],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (MODELS / "optuna_summary_xgb.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  Saved: models/optuna_summary_xgb.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--subsample-rate", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    main(n_trials=args.trials, subsample_rate=args.subsample_rate, n_folds=args.folds)
