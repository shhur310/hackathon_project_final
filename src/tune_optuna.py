"""
tune_optuna.py — Optuna hyperparameter search for CatBoost (best single model)
==================================================================================
Reasoning: CatBoost is best individual model (typically AUC 0.7401-0.7415 range).
Optuna search over depth/leaf_reg/lr/random_strength can push another +0.001-0.003.

Search space (informed by literature for IVF-style tabular):
  - depth: 6-9
  - learning_rate: 0.02-0.07
  - l2_leaf_reg: 1-10
  - random_strength: 0.5-2.0
  - bagging_temperature: 0.3-1.5
  - subsample: 0.7-0.95
  - iterations: 5000 fixed (with early stopping)

Run-time: ~30-60 minutes for 30 trials on full data.
For faster iteration: use --trials 15 --subsample-rate 0.5 (50% sample).

VS Code:
    pip install optuna
    python src/tune_optuna.py --trials 30
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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


def prep_for_cat(X, cat_features):
    X = X.copy()
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("missing")
    return X


def evaluate_params(params: dict, X: pd.DataFrame, y: pd.Series, n_folds: int = 3) -> float:
    """3-fold OOF AUC (faster than 5-fold for tuning)."""
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

        cat_features = Xtr.select_dtypes(include=["object", "category"]).columns.tolist()
        Xtr = prep_for_cat(Xtr, cat_features)
        Xva = prep_for_cat(Xva, cat_features)

        m = CatBoostClassifier(**params)
        m.fit(Pool(Xtr, ytr, cat_features=cat_features),
              eval_set=Pool(Xva, yva, cat_features=cat_features),
              use_best_model=True, verbose=False)
        oof[va] = m.predict_proba(Pool(Xva, cat_features=cat_features))[:, 1]

    return roc_auc_score(y, oof)


def main(n_trials: int = 30, subsample_rate: float = 1.0, n_folds: int = 3):
    t0 = time.time()
    print(f"[1/3] Load data + v2 features", flush=True)
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
            "iterations": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.07, log=True),
            "depth": trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 1.5),
            "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "eval_metric": "AUC", "loss_function": "Logloss",
            "random_seed": 42, "verbose": False,
            "od_type": "Iter", "od_wait": 150, "thread_count": -1,
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
        "best_auc": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "subsample_rate": subsample_rate,
        "all_trial_aucs": [t.value for t in study.trials if t.value is not None],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (MODELS / "optuna_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  Saved: models/optuna_summary.json")
    print(f"\n  Use these params in src/train_v2_full.py CAT_PARAMS dict, then re-run training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--subsample-rate", type=float, default=1.0,
                        help="If < 1.0, use subsample for faster tuning")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    main(n_trials=args.trials, subsample_rate=args.subsample_rate, n_folds=args.folds)
