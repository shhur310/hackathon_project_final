"""
train_v2_full.py — Full v2 training pipeline (PATCHED for XGBoost categorical bug)
======================================================================================
VS Code 실행:
    cd <project_root>
    python src/train_v2_full.py             # 풀 앙상블 (3-model, 25-35분)
    python src/train_v2_full.py --catboost-only  # CatBoost만 (15-20분, 메모리 적게)
"""
from __future__ import annotations
import argparse, gc, json, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.v2_features import add_v2_all_features
from src.features.oof_target_encoder import OOFTargetEncoder

DATA = ROOT / "data"
MODELS = ROOT / "models"
SUB = ROOT / "submission"
MODELS.mkdir(exist_ok=True)
SUB.mkdir(exist_ok=True)

TARGET = "임신 성공 여부"
ID_COL = "ID"
N_SPLITS = 5
SEED = 123
TARGET_ENC_COLS = ["시술 시기 코드", "특정 시술 유형", "배란 유도 유형", "배아 생성 주요 이유"]

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 30,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 0.1, "lambda_l2": 0.5,
    "verbosity": -1, "seed": 123, "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 30,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.5,
    "tree_method": "hist", "random_state": 123, "n_jobs": -1, "verbosity": 0,
}

CAT_PARAMS = {
    "iterations": 10000, 
    "learning_rate": 0.020277240975450236,
    "depth": 8,
    "l2_leaf_reg": 6.347428100049705,
    "bagging_temperature": 0.6030591408625682,
    "random_strength": 1.641591378496605,
    "min_data_in_leaf": 42,
    "eval_metric": "AUC", "loss_function": "Logloss",
    "random_seed": 123, "verbose": False, "od_type": "Iter", "od_wait": 200,
    "thread_count": -1,
}


# ─────────────────────────────────────────────────────────────────
# PATCH: 통일된 categorical 인코딩 helpers
# ─────────────────────────────────────────────────────────────────
def _is_object_or_string(s: pd.Series) -> bool:
    return s.dtype == "object" or pd.api.types.is_string_dtype(s)


def get_object_columns(df: pd.DataFrame) -> list:
    """object + pandas string dtype 컬럼 모두 반환 (pandas 4 호환)."""
    return [c for c in df.columns if _is_object_or_string(df[c])]


def build_object_value_universe(X_full: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """object/string 컬럼별로 train+test 합집합의 정수 코드 매핑 생성.
    Returns: {col_name: {value_str: int_code}}
    값의 *존재*만 사용 - target 누수 0.
    """
    universe = {}
    obj_cols = get_object_columns(X_full)
    for c in obj_cols:
        if c not in X_test.columns:
            continue
        train_vals = X_full[c].astype(str).fillna("__NA__").unique().tolist()
        test_vals = X_test[c].astype(str).fillna("__NA__").unique().tolist()
        all_vals = sorted(set(train_vals) | set(test_vals))
        universe[c] = {v: i for i, v in enumerate(all_vals)}
    return universe


def encode_with_universe(X: pd.DataFrame, universe: dict) -> pd.DataFrame:
    """object 컬럼을 통일 정수 코드로 변환 (XGBoost 호환).
    Unknown (universe에 없는) values → -1 (XGBoost는 이를 missing처럼 처리)
    """
    X = X.copy()
    for col, mapping in universe.items():
        if col in X.columns:
            X[col] = (
                X[col].astype(str).fillna("__NA__")
                .map(mapping).fillna(-1).astype(np.int32)
            )
    return X


def prep_for_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM: object → category dtype. fold 단위로 다른 코드 OK (LGBM 자체 인코딩)."""
    X = X.copy()
    for c in get_object_columns(X):
        X[c] = X[c].astype("category")
    return X


def prep_for_cat(X: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """CatBoost: object → string + 'missing' fill."""
    X = X.copy()
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("missing")
    return X


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main(catboost_only: bool = False, fold: int = 5):
    t0 = time.time()
    print(f"[1/5] Load + v2 features", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    train = add_v2_all_features(train)
    test = add_v2_all_features(test)
    print(f"  train: {train.shape}, test: {test.shape}", flush=True)

    y = train[TARGET].astype(int).reset_index(drop=True)

    drop_cols = [TARGET, ID_COL] if ID_COL in train.columns else [TARGET]
    X = train.drop(columns=drop_cols).reset_index(drop=True)
    X_test = test.drop(columns=[ID_COL] if ID_COL in test.columns else []).reset_index(drop=True)

    common = [c for c in X.columns if c in X_test.columns]
    only_train = set(X.columns) - set(X_test.columns)
    if only_train:
        print(f"  WARNING: train-only columns dropped: {sorted(only_train)}", flush=True)
    X = X[common]
    X_test = X_test[common]
    print(f"  X: {X.shape}, X_test: {X_test.shape}", flush=True)

    del train, test; gc.collect()

    # ── PATCH: object 컬럼 통일 인코딩 (XGBoost용) ──
    print(f"\n[2/5] Building object-value universe (XGBoost categorical fix)", flush=True)
    obj_universe = build_object_value_universe(X, X_test)
    print(f"  Object cols mapped: {len(obj_universe)}", flush=True)
    for c in list(obj_universe.keys())[:5]:
        print(f"    {c}: {len(obj_universe[c])} unique values", flush=True)

    # XGBoost용 정수 코딩 데이터 미리 준비. CatBoost / LightGBM은 원본 사용.
    X_xgb = encode_with_universe(X, obj_universe)
    X_test_xgb = encode_with_universe(X_test, obj_universe)
    print(f"  X_xgb dtype check: object cols remaining = "
          f"{len(get_object_columns(X_xgb))} (should be 0)", flush=True)

    # Test set TE — 전체 train labels 사용 (test labels NOT used)
    print(f"\n[3/5] Test set TE (full train labels, no test leak)", flush=True)
    enc = OOFTargetEncoder(cols=[c for c in TARGET_ENC_COLS if c in X.columns], smoothing=50.0)
    X_test_te = enc.fit_full_then_transform(X, y, X_test)
    X_test = pd.concat([X_test.reset_index(drop=True), X_test_te.reset_index(drop=True)], axis=1)
    X_test_xgb = pd.concat([X_test_xgb.reset_index(drop=True), X_test_te.reset_index(drop=True)], axis=1)
    print(f"  X_test final: {X_test.shape}, X_test_xgb: {X_test_xgb.shape}", flush=True)

    print(f"\n[4/5] {fold}-fold StratifiedKFold training", flush=True)
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=SEED)

    oof_lgbm = np.zeros(len(X), dtype=np.float32)
    oof_xgb = np.zeros(len(X), dtype=np.float32)
    oof_cat = np.zeros(len(X), dtype=np.float32)
    test_pred_lgbm = np.zeros(len(X_test), dtype=np.float32)
    test_pred_xgb = np.zeros(len(X_test_xgb), dtype=np.float32)
    test_pred_cat = np.zeros(len(X_test), dtype=np.float32)

    for f_i, (tr, va) in enumerate(skf.split(X, y), start=1):
        t = time.time()
        Xtr_base = X.iloc[tr].reset_index(drop=True)
        Xva_base = X.iloc[va].reset_index(drop=True)
        Xtr_xgb_base = X_xgb.iloc[tr].reset_index(drop=True)
        Xva_xgb_base = X_xgb.iloc[va].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True)
        yva = y.iloc[va].reset_index(drop=True)

        # OOF target encoding (fold-internal train only)
        tr_te, va_te = enc.fit_transform_oof(Xtr_base, ytr, Xva_base)
        Xtr = pd.concat([Xtr_base, tr_te], axis=1)
        Xva = pd.concat([Xva_base, va_te], axis=1)
        Xtr_xgb = pd.concat([Xtr_xgb_base, tr_te], axis=1)
        Xva_xgb = pd.concat([Xva_xgb_base, va_te], axis=1)

        # ────── LightGBM ──────
        if not catboost_only:
            Xtr_l = prep_for_lgbm(Xtr); Xva_l = prep_for_lgbm(Xva)
            dtr = lgb.Dataset(Xtr_l, ytr); dva = lgb.Dataset(Xva_l, yva, reference=dtr)
            m = lgb.train(LGBM_PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
            oof_lgbm[va] = m.predict(Xva_l, num_iteration=m.best_iteration)
            X_test_l = prep_for_lgbm(X_test)
            test_pred_lgbm += m.predict(X_test_l, num_iteration=m.best_iteration).astype(np.float32) / fold
            joblib.dump(m, MODELS / f"lgbm_v2_fold{f_i}.pkl")
            del Xtr_l, Xva_l, X_test_l, m, dtr, dva; gc.collect()

        # ────── XGBoost (정수 코드 입력, enable_categorical=False) ──────
        if not catboost_only:
            dtr = xgb.DMatrix(Xtr_xgb, label=ytr)
            dva = xgb.DMatrix(Xva_xgb, label=yva)
            m = xgb.train(XGB_PARAMS, dtr, num_boost_round=10000,
                          evals=[(dva, "valid")], early_stopping_rounds=200, verbose_eval=0)
            best_iter = m.best_iteration
            oof_xgb[va] = m.predict(dva, iteration_range=(0, best_iter + 1))
            dtest = xgb.DMatrix(X_test_xgb)
            test_pred_xgb += m.predict(dtest, iteration_range=(0, best_iter + 1)).astype(np.float32) / fold
            m.save_model(str(MODELS / f"xgb_v2_fold{f_i}.json"))
            del m, dtr, dva, dtest; gc.collect()

        # ────── CatBoost (native object handling) ──────
        cat_features = get_object_columns(Xtr)
        Xtr_c = prep_for_cat(Xtr, cat_features)
        Xva_c = prep_for_cat(Xva, cat_features)
        ptr = Pool(Xtr_c, ytr, cat_features=cat_features)
        pva = Pool(Xva_c, yva, cat_features=cat_features)
        m = CatBoostClassifier(**CAT_PARAMS)
        m.fit(ptr, eval_set=pva, use_best_model=True, verbose=False)
        oof_cat[va] = m.predict_proba(pva)[:, 1]
        X_test_c = prep_for_cat(X_test, cat_features)
        ptest = Pool(X_test_c, cat_features=cat_features)
        test_pred_cat += m.predict_proba(ptest)[:, 1].astype(np.float32) / fold
        m.save_model(str(MODELS / f"cat_v2_fold{f_i}.cbm"))
        del Xtr_c, Xva_c, X_test_c, m, ptr, pva, ptest; gc.collect()

        a_l = roc_auc_score(yva, oof_lgbm[va]) if not catboost_only else 0
        a_x = roc_auc_score(yva, oof_xgb[va]) if not catboost_only else 0
        a_c = roc_auc_score(yva, oof_cat[va])
        print(f"  Fold {f_i}: LGBM={a_l:.5f}, XGB={a_x:.5f}, CAT={a_c:.5f}  "
              f"({time.time()-t:.0f}s)", flush=True)

        del Xtr_base, Xva_base, Xtr_xgb_base, Xva_xgb_base
        del Xtr, Xva, Xtr_xgb, Xva_xgb
        gc.collect()

    aucs = {}
    if not catboost_only:
        aucs["lgbm"] = roc_auc_score(y, oof_lgbm)
        aucs["xgb"] = roc_auc_score(y, oof_xgb)
    aucs["cat"] = roc_auc_score(y, oof_cat)

    print(f"\n[5/5] Individual OOF AUC:", flush=True)
    for k, v in aucs.items():
        print(f"  {k}: {v:.5f}", flush=True)

    if not catboost_only:
        np.savez(MODELS / "oof_v2_seed123.npz",
                 lgbm=oof_lgbm, xgb=oof_xgb, cat=oof_cat, y=y.values)
        np.savez(MODELS / "test_v2_seed123.npz",
                 lgbm=test_pred_lgbm, xgb=test_pred_xgb, cat=test_pred_cat)
    else:
        np.savez(MODELS / "oof_v2_seed123.npz", cat=oof_cat, y=y.values)
        np.savez(MODELS / "test_v2_seed123.npz", cat=test_pred_cat)

    print(f"\n  Blend optimization", flush=True)
    if not catboost_only:
        oofs = np.column_stack([oof_lgbm, oof_xgb, oof_cat]).astype(np.float32)
        tests = np.column_stack([test_pred_lgbm, test_pred_xgb, test_pred_cat]).astype(np.float32)

        def neg_auc(w):
            if (w < 0).any() or w.sum() < 0.99:
                return 0
            return -roc_auc_score(y, oofs @ (w / w.sum()))

        best, bw = 0, None
        for seed in [0, 42, 7]:
            rng = np.random.default_rng(seed)
            x0 = rng.dirichlet(np.ones(3))
            res = minimize(neg_auc, x0, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 300})
            w = res.x / res.x.sum()
            if -res.fun > best:
                best, bw = -res.fun, w

        print(f"  Mean blend: {roc_auc_score(y, oofs.mean(axis=1)):.5f}", flush=True)
        print(f"  NM blend  : {best:.5f}  weights="
              f"{dict(zip(['lgbm','xgb','cat'], [round(float(x),3) for x in bw]))}", flush=True)
        test_blend = tests @ bw
        final_auc = best
    else:
        test_blend = test_pred_cat
        final_auc = aucs["cat"]

    sub = pd.read_csv(DATA / "sample_submission.csv")
    pd.DataFrame({"ID": sub["ID"].values, "probability": test_blend}).to_csv(
        SUB / "submission_v2_seed123.csv", index=False)
    print(f"\n  ✅ Saved submission/submission_v2.csv", flush=True)
    print(f"  Final OOF AUC: {final_auc:.5f}  "
          f"({'✅ TARGET MET' if final_auc >= 0.742 else f'gap to 0.742: {0.742-final_auc:+.5f}'})",
          flush=True)

    summary = {
        "individual_aucs": {k: float(v) for k, v in aucs.items()},
        "final_auc": float(final_auc),
        "target_met": bool(final_auc >= 0.742),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_features": int(X.shape[1]),
    }
    (MODELS / "v2_summary_seed123.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catboost-only", action="store_true",
                        help="Memory-friendly CatBoost only (skip LGBM + XGB)")
    parser.add_argument("--fold", type=int, default=5)
    args = parser.parse_args()
    main(catboost_only=args.catboost_only, fold=args.fold)