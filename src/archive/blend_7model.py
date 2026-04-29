"""7-blend: 6 GBM + 1 MLP"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

v1_oof = np.load('models/oof_v1_optunaCAT.npz')
v1_test = np.load('models/test_v1_optunaCAT.npz')
v2_oof = np.load('models/oof_v2_ALL_OPTUNA.npz')
v2_test = np.load('models/test_v2_ALL_OPTUNA.npz')
mlp_oof = np.load('models/oof_mlp.npz')
mlp_test = np.load('models/test_mlp.npz')
y = v1_oof['y']

oofs = np.column_stack([
    v1_oof['lgbm'], v1_oof['xgb'], v1_oof['cat'],
    v2_oof['lgbm'], v2_oof['xgb'], v2_oof['cat'],
    mlp_oof['mlp'],
]).astype(np.float32)

tests = np.column_stack([
    v1_test['lgbm'], v1_test['xgb'], v1_test['cat'],
    v2_test['lgbm'], v2_test['xgb'], v2_test['cat'],
    mlp_test['mlp'],
]).astype(np.float32)

names = ['v1_lgb', 'v1_xgb', 'v1_cat', 'v2_lgb', 'v2_xgb', 'v2_cat', 'mlp']

print('=' * 60)
print('Individual OOF AUC:')
for i, n in enumerate(names):
    print(f'  {n}: {roc_auc_score(y, oofs[:, i]):.5f}')

print('\nMLP correlation with GBM (낮을수록 blend 효과 큼):')
print(f'  mlp vs v1_lgb: {np.corrcoef(oofs[:,6], oofs[:,0])[0,1]:.4f}')
print(f'  mlp vs v2_lgb: {np.corrcoef(oofs[:,6], oofs[:,3])[0,1]:.4f}')
print(f'  mlp vs v1_cat: {np.corrcoef(oofs[:,6], oofs[:,2])[0,1]:.4f}')
print(f'  mlp vs v2_cat: {np.corrcoef(oofs[:,6], oofs[:,5])[0,1]:.4f}')
print(f'  mlp vs v1_xgb: {np.corrcoef(oofs[:,6], oofs[:,1])[0,1]:.4f}')
print(f'  mlp vs v2_xgb: {np.corrcoef(oofs[:,6], oofs[:,4])[0,1]:.4f}')

def neg_auc(w):
    if (w < 0).any() or w.sum() < 0.99: return 0
    return -roc_auc_score(y, oofs @ (w / w.sum()))

best, bw = 0, None
for s in [0, 42, 7, 123, 999]:
    rng = np.random.default_rng(s)
    x0 = rng.dirichlet(np.ones(7))
    res = minimize(neg_auc, x0, method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 800})
    w = res.x / res.x.sum()
    auc = -res.fun
    if auc > best:
        best, bw = auc, w

print()
print('=' * 60)
print(f'6-blend (GBM only):  0.74075')
print(f'7-blend (GBM+MLP):   {best:.5f}')
print(f'Δ:                    {best - 0.74075:+.5f}')
print('Weights:')
for n, w in zip(names, bw):
    print(f'  {n}: {w:.3f}')

test_blend = tests @ bw
sub = pd.read_csv('data/sample_submission.csv')
out = pd.DataFrame({'ID': sub['ID'], 'probability': test_blend})
out.to_csv('submission/submission_7blend.csv', index=False)

print()
print(f'Pred range: [{test_blend.min():.4f}, {test_blend.max():.4f}], mean={test_blend.mean():.4f}')
print(f'\n예상 LB ~{best + 0.00151:.5f}')
print(f'vs 1등 0.74246 차이: {best + 0.00151 - 0.74246:+.5f}')
print(f'\nSaved: submission/submission_7blend.csv')
