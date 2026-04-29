"""
TabNet 5-fold OOF training.
Dreamquark TabNet — different attention mechanism than Transformer.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

from pytorch_tabnet.tab_model import TabNetClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.v2_features import add_v2_all_features

DATA = ROOT / "data"
MODELS = ROOT / "models"
TARGET = "임신 성공 여부"
ID_COL = "ID"

device_name = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device_name}", flush=True)
torch.manual_seed(456)
np.random.seed(456)

# ─── Data prep ─────────────────────────────────────────
print("[1/4] Load data + v2 features", flush=True)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
train = add_v2_all_features(train)
test = add_v2_all_features(test)

y = train[TARGET].astype(int).values
X = train.drop(columns=[TARGET, ID_COL]).copy()
X_test = test.drop(columns=[ID_COL]).copy()

def is_cat_col(s):
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        return True
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if str(s.dtype).startswith('string'):
        return True
    sample = s.dropna().head(50)
    if len(sample) == 0: return False
    try:
        pd.to_numeric(sample); return False
    except (ValueError, TypeError):
        return True

obj_cols = [c for c in X.columns if is_cat_col(X[c])]
num_cols = [c for c in X.columns if c not in obj_cols]
print(f"  Categorical: {len(obj_cols)}, Numerical: {len(num_cols)}", flush=True)

# Label encode categoricals
cat_dim_list = []
cat_idxs = []
for i, c in enumerate(X.columns):
    if c not in obj_cols:
        continue
    arr_X = X[c].values
    arr_test = X_test[c].values
    def to_safe_str(arr):
        out = np.empty(len(arr), dtype=object)
        for j, v in enumerate(arr):
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v == 'nan'):
                out[j] = '__NA__'
            else:
                out[j] = str(v)
        return out
    s_X = to_safe_str(arr_X)
    s_test = to_safe_str(arr_test)
    all_vals = np.concatenate([s_X, s_test])
    uniq = sorted(set(all_vals.tolist()))
    mapping = {v: idx for idx, v in enumerate(uniq)}
    X[c] = np.array([mapping[v] for v in s_X], dtype=np.int32)
    X_test[c] = np.array([mapping[v] for v in s_test], dtype=np.int32)
    cat_dim_list.append(len(uniq))
    cat_idxs.append(list(X.columns).index(c))

# Numerical
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype("float32")
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(0).astype("float32")

# Convert to numpy (TabNet expects single matrix)
X_full = X.values.astype("float32")
X_test_full = X_test.values.astype("float32")
print(f"  X={X_full.shape}, X_test={X_test_full.shape}", flush=True)
print(f"  cat_idxs={len(cat_idxs)}, cat_dims={cat_dim_list[:5]}...", flush=True)

# ─── Training ──────────────────────────────────────────
N_FOLDS = 5
MAX_EPOCHS = 100
PATIENCE = 10
BATCH = 4096
VIRTUAL_BATCH = 512

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

print(f"\n[2/4] TabNet training", flush=True)
t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
    ts = time.time()

    # Scale numerical only (TabNet's BatchNorm handles internally too)
    scaler = StandardScaler()
    Xtr = X_full[tr_idx].copy()
    Xva = X_full[va_idx].copy()
    Xte = X_test_full.copy()
    # Scale numerical columns only
    num_idxs = [i for i in range(X_full.shape[1]) if i not in cat_idxs]
    Xtr[:, num_idxs] = scaler.fit_transform(Xtr[:, num_idxs])
    Xva[:, num_idxs] = scaler.transform(Xva[:, num_idxs])
    Xte[:, num_idxs] = scaler.transform(Xte[:, num_idxs])

    model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=4,
        gamma=1.3, n_independent=2, n_shared=2,
        cat_idxs=cat_idxs,
        cat_dims=cat_dim_list,
        cat_emb_dim=8,
        seed=42 + fold,
        device_name=device_name,
        verbose=0,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={"T_max": MAX_EPOCHS},
        mask_type='entmax',
    )

    model.fit(
        X_train=Xtr, y_train=y[tr_idx],
        eval_set=[(Xva, y[va_idx])],
        eval_metric=['auc'],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH, virtual_batch_size=VIRTUAL_BATCH,
        drop_last=False,
    )

    va_pred = model.predict_proba(Xva)[:, 1]
    te_pred = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y[va_idx], va_pred)

    oof[va_idx] = va_pred
    test_pred += te_pred / N_FOLDS
    print(f"  Fold {fold+1}: AUC={auc:.5f}  best_ep={model.best_epoch}  ({time.time()-ts:.0f}s)", flush=True)

print(f"\n[3/4] OOF AUC: {roc_auc_score(y, oof):.5f}", flush=True)
np.savez_compressed(MODELS / "oof_tabnet.npz", tabnet=oof, y=y)
np.savez_compressed(MODELS / "test_tabnet.npz", tabnet=test_pred)
print(f"\n[4/4] Done. Total: {(time.time()-t0)/60:.1f} min")
print(f"  Saved: models/oof_tabnet.npz, models/test_tabnet.npz")
