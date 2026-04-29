"""
6-model blend: v1 (default LGBM/XGB + Optuna CAT) + v2 (all Optuna)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

v1_oof = np.load('models/oof_v1_optunaCAT.npz')
v1_test = np.load('models/test_v1_optunaCAT.npz')
v2_oof = np.load('models/oof_v2_ALL_OPTUNA.npz')
v2_test = np.load('models/test_v2_ALL_OPTUNA.npz')
y = v1_oof['y']

oofs = np.column_stack([
    v1_oof['lgbm'], v1_oof['xgb'], v1_oof['cat'],
    v2_oof['lgbm'], v2_oof['xgb'], v2_oof['cat'],
]).astype(np.float32)

tests = np.column_stack([
    v1_test['lgbm'], v1_test['xgb'], v1_test['cat'],
    v2_test['lgbm'], v2_test['xgb'], v2_test['cat'],
]).astype(np.float32)

names = ['v1_lgb', 'v1_xgb', 'v1_cat', 'v2_lgb', 'v2_xgb', 'v2_cat']

print('=' * 60)
print('Individual OOF AUC:')
for i, n in enumerate(names):
    print(f'  {n}: {roc_auc_score(y, oofs[:, i]):.5f}')

print('\nv1-v2 model correlations (낮을수록 blend 효과 ↑):')
print(f'  v1_lgb vs v2_lgb: {np.corrcoef(oofs[:,0], oofs[:,3])[0,1]:.4f}')
print(f'  v1_xgb vs v2_xgb: {np.corrcoef(oofs[:,1], oofs[:,4])[0,1]:.4f}')
print(f'  v1_cat vs v2_cat: {np.corrcoef(oofs[:,2], oofs[:,5])[0,1]:.4f}')

def neg_auc(w):
    if (w < 0).any() or w.sum() < 0.99: return 0
    return -roc_auc_score(y, oofs @ (w / w.sum()))

best_auc, best_w = 0, None
for seed in [0, 42, 7]:
    rng = np.random.default_rng(seed)
    x0 = rng.dirichlet(np.ones(6))
    res = minimize(neg_auc, x0, method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 600})
    w = res.x / res.x.sum()
    auc = -res.fun
    if auc > best_auc:
        best_auc, best_w = auc, w

print()
print('=' * 60)
print(f'Mean blend (6):  {roc_auc_score(y, oofs.mean(axis=1)):.5f}')
print(f'NM Blend (best): {best_auc:.5f}')
print('Weights:')
for n, w in zip(names, best_w):
    print(f'  {n}: {w:.3f}')

test_blend = tests @ best_w
sub = pd.read_csv('data/sample_submission.csv')
out = pd.DataFrame({'ID': sub['ID'], 'probability': test_blend})
out.to_csv('submission/submission_6blend.csv', index=False)

print()
print(f'Pred range: [{test_blend.min():.4f}, {test_blend.max():.4f}], mean={test_blend.mean():.4f}')
print(f'\n비교:')
print(f'  v1 (LB 0.74217):  0.74066')
print(f'  v2 ALL_OPTUNA:    0.74074 (미제출)')
print(f'  6-blend (지금):   {best_auc:.5f}')
print(f'\n예상 LB ~{best_auc + 0.00151:.5f}')
print(f'vs 1등 0.74237 차이: {best_auc + 0.00151 - 0.74237:+.5f}')
print(f'\nSaved: submission/submission_6blend.csv')
