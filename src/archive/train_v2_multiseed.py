"""
Multi-seed v2 ensemble: 3 seeds × 5-fold × 3 models = 45 models
Uses Optuna best CatBoost params + averages predictions across seeds.
"""
import os, sys, json, time, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# train_v2_full.py와 동일한 import 경로
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.features.v2_features import add_v2_all_features
from src.features.oof_target_encoder import OOFTargetEncoder

# ─── 데이터 로드 ─────────────────────────────────────────
print('[1/4] Load data + v2 features')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = add_v2_all_features(train)
test = add_v2_all_features(test)

TARGET = '임신 성공 여부'
ID_COL = 'ID'
y = train[TARGET].values
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

print(f'  X: {X.shape}, X_test: {X_test.shape}')

# Object → category for LGBM/CAT
obj_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in obj_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Build XGB integer encoding (train+test universe → no fold mismatch)
def build_universe(X_, X_test_, cols):
    universe = {}
    for c in cols:
        s = pd.concat([X_[c], X_test_[c]]).fillna('__NA__').astype(str)
        vals = sorted(s.unique().tolist())
        universe[c] = {v: i for i, v in enumerate(vals)}
    return universe

universe = build_universe(X, X_test, obj_cols)
def encode_int(df, universe):
    df_e = df.copy()
    for c, mp in universe.items():
        # category dtype 우회: object로 변환 후 fillna
        s = df[c].astype(object)
        s = pd.Series(s).where(pd.notna(s), '__NA__').astype(str)
        df_e[c] = s.map(mp).fillna(-1).astype('int32').values
    return df_e

X_xgb = encode_int(X, universe)
X_test_xgb = encode_int(X_test, universe)

# Optuna best CatBoost params
opt = json.load(open('models/optuna_summary.json'))
cat_best = opt['best_params']
print(f'  CatBoost best params: {cat_best}')

TARGET_ENC_COLS = ['시술 시기 코드', '특정 시술 유형', '배란 유도 유형', '배아 생성 주요 이유']
te_cols = [c for c in TARGET_ENC_COLS if c in X.columns]
print(f'  Target-encoded cols: {te_cols}')

# ─── 모델 hyperparameters ────────────────────────────────
def lgbm_params(seed):
    return dict(
        objective='binary', metric='auc', learning_rate=0.03,
        num_leaves=127, min_child_samples=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=0.5, verbose=-1, seed=seed,
    )

def xgb_params(seed):
    return dict(
        objective='binary:logistic', eval_metric='auc', tree_method='hist',
        learning_rate=0.03, max_depth=7, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.5, seed=seed, nthread=-1,
    )

def cat_params(seed):
    p = dict(
        iterations=10000, eval_metric='AUC', loss_function='Logloss',
        random_seed=seed, verbose=False,
        od_type='Iter', od_wait=200, thread_count=-1,
    )
    p.update({k: cat_best[k] for k in cat_best})
    return p

# ─── Storage ─────────────────────────────────────────────
SEEDS = [42, 7, 123]
N_FOLDS = 5

oof = {m: {s: np.zeros(len(y)) for s in SEEDS} for m in ['lgb', 'xgb', 'cat']}
testp = {m: {s: np.zeros(len(X_test)) for s in SEEDS} for m in ['lgb', 'xgb', 'cat']}

# ─── Multi-seed training ─────────────────────────────────
print(f'\n[2/4] Training: {len(SEEDS)} seeds × {N_FOLDS} folds × 3 models')
t0 = time.time()
for seed in SEEDS:
    print(f'\n=== SEED {seed} ===')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # Test set TE (same encoder per seed, fit on full train labels)
    enc_full = OOFTargetEncoder(cols=te_cols, smoothing=50.0)
    X_test_te = enc_full.fit_full_then_transform(X.copy(), y, X_test.copy())
    X_test_te_xgb = encode_int(X_test_te, universe) if any(c in obj_cols for c in te_cols) else X_test_xgb

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        ts = time.time()
        # Per-fold OOF target encoding
        enc = OOFTargetEncoder(cols=te_cols, smoothing=50.0)
        Xtr_base, Xva_base = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        ytr = y[tr_idx]
        Xtr_te, Xva_te = enc.fit_transform_oof(Xtr_base, ytr, Xva_base)

        Xtr_xgb_te = encode_int(Xtr_te, universe) if any(c in obj_cols for c in te_cols) else X_xgb.iloc[tr_idx]
        Xva_xgb_te = encode_int(Xva_te, universe) if any(c in obj_cols for c in te_cols) else X_xgb.iloc[va_idx]

        y_va = y[va_idx]

        # LGBM
        lgb_tr = lgb.Dataset(Xtr_te, label=ytr, categorical_feature=obj_cols)
        lgb_va = lgb.Dataset(Xva_te, label=y_va, categorical_feature=obj_cols, reference=lgb_tr)
        m_l = lgb.train(lgbm_params(seed), lgb_tr, num_boost_round=10000,
                        valid_sets=[lgb_va], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof['lgb'][seed][va_idx] = m_l.predict(Xva_te, num_iteration=m_l.best_iteration)
        testp['lgb'][seed] += m_l.predict(X_test_te, num_iteration=m_l.best_iteration) / N_FOLDS

        # XGB
        dtr = xgb.DMatrix(Xtr_xgb_te, label=ytr)
        dva = xgb.DMatrix(Xva_xgb_te, label=y_va)
        dte = xgb.DMatrix(X_test_te_xgb)
        m_x = xgb.train(xgb_params(seed), dtr, num_boost_round=10000,
                        evals=[(dva, 'va')], early_stopping_rounds=200, verbose_eval=False)
        oof['xgb'][seed][va_idx] = m_x.predict(dva, iteration_range=(0, m_x.best_iteration + 1))
        testp['xgb'][seed] += m_x.predict(dte, iteration_range=(0, m_x.best_iteration + 1)) / N_FOLDS

        # CatBoost
        Xtr_cb = Xtr_te.astype(str).fillna('missing')
        Xva_cb = Xva_te.astype(str).fillna('missing')
        Xte_cb = X_test_te.astype(str).fillna('missing')
        cat_feats = list(Xtr_cb.columns)
        m_c = CatBoostClassifier(**cat_params(seed))
        m_c.fit(Pool(Xtr_cb, ytr, cat_features=cat_feats),
                eval_set=Pool(Xva_cb, y_va, cat_features=cat_feats),
                use_best_model=True)
        oof['cat'][seed][va_idx] = m_c.predict_proba(Xva_cb)[:, 1]
        testp['cat'][seed] += m_c.predict_proba(Xte_cb)[:, 1] / N_FOLDS

        al = roc_auc_score(y_va, oof['lgb'][seed][va_idx])
        ax = roc_auc_score(y_va, oof['xgb'][seed][va_idx])
        ac = roc_auc_score(y_va, oof['cat'][seed][va_idx])
        print(f'  Fold {fold+1}: L={al:.5f} X={ax:.5f} C={ac:.5f} ({time.time()-ts:.0f}s)')

        del m_l, m_x, m_c
        gc.collect()

# ─── Aggregate ───────────────────────────────────────────
print(f'\n[3/4] Aggregation (total: {(time.time()-t0)/60:.1f} min)')
print('Per-seed AUC:')
for s in SEEDS:
    print(f'  seed={s}: '
          f'L={roc_auc_score(y, oof["lgb"][s]):.5f} '
          f'X={roc_auc_score(y, oof["xgb"][s]):.5f} '
          f'C={roc_auc_score(y, oof["cat"][s]):.5f}')

oof_lgb = np.mean([oof['lgb'][s] for s in SEEDS], axis=0)
oof_xgb = np.mean([oof['xgb'][s] for s in SEEDS], axis=0)
oof_cat = np.mean([oof['cat'][s] for s in SEEDS], axis=0)
test_lgb = np.mean([testp['lgb'][s] for s in SEEDS], axis=0)
test_xgb = np.mean([testp['xgb'][s] for s in SEEDS], axis=0)
test_cat = np.mean([testp['cat'][s] for s in SEEDS], axis=0)

print('\nMulti-seed averaged AUC:')
print(f'  LGBM: {roc_auc_score(y, oof_lgb):.5f}')
print(f'  XGB:  {roc_auc_score(y, oof_xgb):.5f}')
print(f'  CAT:  {roc_auc_score(y, oof_cat):.5f}')

# ─── Blend ───────────────────────────────────────────────
oofs = np.column_stack([oof_lgb, oof_xgb, oof_cat]).astype(np.float32)
tests = np.column_stack([test_lgb, test_xgb, test_cat]).astype(np.float32)

def neg_auc(w):
    if (w < 0).any() or w.sum() < 0.99: return 0
    return -roc_auc_score(y, oofs @ (w / w.sum()))

best, bw = 0, None
for s in [0, 42, 7]:
    rng = np.random.default_rng(s)
    x0 = rng.dirichlet(np.ones(3))
    res = minimize(neg_auc, x0, method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 400})
    w = res.x / res.x.sum()
    if -res.fun > best:
        best, bw = -res.fun, w

print(f'\n[4/4] Final')
print(f'  Mean blend:  {roc_auc_score(y, oofs.mean(axis=1)):.5f}')
print(f'  NM blend:    {best:.5f}  weights L:{bw[0]:.2f} X:{bw[1]:.2f} C:{bw[2]:.2f}')

# Save
np.savez_compressed('models/oof_v2_multiseed.npz',
                    lgbm=oof_lgb, xgb=oof_xgb, cat=oof_cat, y=y)
np.savez_compressed('models/test_v2_multiseed.npz',
                    lgbm=test_lgb, xgb=test_xgb, cat=test_cat)

test_blend = tests @ bw
sub = pd.read_csv('data/sample_submission.csv')
pd.DataFrame({'ID': sub['ID'], 'probability': test_blend}).to_csv(
    'submission/submission_v2_multiseed.csv', index=False)

summary = {
    'final_auc': float(best),
    'weights': {'lgbm': float(bw[0]), 'xgb': float(bw[1]), 'cat': float(bw[2])},
    'individual_multiseed': {
        'lgbm': float(roc_auc_score(y, oof_lgb)),
        'xgb': float(roc_auc_score(y, oof_xgb)),
        'cat': float(roc_auc_score(y, oof_cat)),
    },
    'seeds': SEEDS,
    'target_met': bool(best >= 0.742),
    'gap_to_target': float(0.742 - best),
}
json.dump(summary, open('models/v2_multiseed_summary.json', 'w'), indent=2)
print(f'\nSaved: submission/submission_v2_multiseed.csv')
print(f'Target met (≥0.742): {summary["target_met"]}')
