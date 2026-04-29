"""
MLP 5-fold OOF training for tabular IVF data.
PyTorch + MPS (Mac M2 GPU acceleration).
Hardened against pandas 4 string dtype + NaN sorting issues.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.v2_features import add_v2_all_features

DATA = ROOT / "data"
MODELS = ROOT / "models"
TARGET = "임신 성공 여부"
ID_COL = "ID"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)
torch.manual_seed(42)
np.random.seed(42)

# ─── Data prep ─────────────────────────────────────────
print("[1/4] Load data + v2 features", flush=True)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
train = add_v2_all_features(train)
test = add_v2_all_features(test)

y = train[TARGET].astype(int).values
X = train.drop(columns=[TARGET, ID_COL]).copy()
X_test = test.drop(columns=[ID_COL]).copy()
print(f"  X: {X.shape}, X_test: {X_test.shape}", flush=True)

# ─── Identify categorical vs numerical (robust) ─────────
def is_cat_col(s):
    """Check if column is categorical/string (not numeric)."""
    if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
        return True
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if str(s.dtype).startswith('string'):
        return True
    # Sample check: if any non-null value can't convert to float
    sample = s.dropna().head(50)
    if len(sample) == 0:
        return False
    try:
        pd.to_numeric(sample)
        return False
    except (ValueError, TypeError):
        return True

obj_cols = [c for c in X.columns if is_cat_col(X[c])]
num_cols = [c for c in X.columns if c not in obj_cols]
print(f"  Categorical: {len(obj_cols)}, Numerical: {len(num_cols)}", flush=True)
print(f"  First 5 cat: {obj_cols[:5]}", flush=True)

# ─── Label encode categoricals (numpy-based, NaN-safe) ──
cat_dim_list = []
for c in obj_cols:
    # Convert to numpy, then handle NaN explicitly
    arr_X = X[c].values
    arr_test = X_test[c].values

    # Build string array, replacing NaN with '__NA__'
    def to_safe_str(arr):
        out = np.empty(len(arr), dtype=object)
        for i, v in enumerate(arr):
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v == 'nan'):
                out[i] = '__NA__'
            else:
                out[i] = str(v)
        return out

    s_X = to_safe_str(arr_X)
    s_test = to_safe_str(arr_test)

    # Combine and sort
    all_vals = np.concatenate([s_X, s_test])
    uniq = sorted(set(all_vals.tolist()))
    mapping = {v: i for i, v in enumerate(uniq)}

    X[c] = np.array([mapping[v] for v in s_X], dtype=np.int32)
    X_test[c] = np.array([mapping[v] for v in s_test], dtype=np.int32)
    cat_dim_list.append(len(uniq))

print(f"  Label encoding done. Cardinalities: min={min(cat_dim_list)}, max={max(cat_dim_list)}", flush=True)

# ─── Numerical: pd.to_numeric + fill nan ────────────────
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype("float32")
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(0).astype("float32")

# Convert to numpy
X_num_all = X[num_cols].values.astype("float32")
X_cat_all = X[obj_cols].values.astype("int64")
X_test_num = X_test[num_cols].values.astype("float32")
X_test_cat = X_test[obj_cols].values.astype("int64")
print(f"  Final: X_num={X_num_all.shape}, X_cat={X_cat_all.shape}", flush=True)

# ─── Model ─────────────────────────────────────────────
class TabMLP(nn.Module):
    def __init__(self, num_n, cat_dims, emb_dim=8, hidden=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(d, min(emb_dim, max(2, d // 2 + 1))) for d in cat_dims
        ])
        emb_total = sum(min(emb_dim, max(2, d // 2 + 1)) for d in cat_dims)
        in_dim = num_n + emb_total
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        z = torch.cat([x_num] + embs, dim=1)
        return self.mlp(z).squeeze(-1)

# ─── Training ──────────────────────────────────────────
N_FOLDS = 5
EPOCHS = 30
BATCH = 4096
LR = 1e-3
WD = 1e-5
PATIENCE = 5

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

print(f"\n[2/4] Training (epochs={EPOCHS}, batch={BATCH})", flush=True)
t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num_all, y)):
    ts = time.time()

    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(X_num_all[tr_idx])
    Xva_num = scaler.transform(X_num_all[va_idx])
    Xte_num = scaler.transform(X_test_num)

    Xtr_num_t = torch.from_numpy(Xtr_num.astype("float32"))
    Xva_num_t = torch.from_numpy(Xva_num.astype("float32"))
    Xte_num_t = torch.from_numpy(Xte_num.astype("float32"))
    Xtr_cat_t = torch.from_numpy(X_cat_all[tr_idx])
    Xva_cat_t = torch.from_numpy(X_cat_all[va_idx])
    Xte_cat_t = torch.from_numpy(X_test_cat)
    ytr_t = torch.from_numpy(y[tr_idx].astype("float32"))

    train_ds = TensorDataset(Xtr_num_t, Xtr_cat_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model = TabMLP(num_n=Xtr_num.shape[1], cat_dims=cat_dim_list).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()

    best_auc, best_va, best_te, no_improve = 0, None, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
            optim.zero_grad()
            loss = bce(model(xn, xc), yb)
            loss.backward()
            optim.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logit = model(Xva_num_t.to(device), Xva_cat_t.to(device)).cpu().numpy()
        va_pred = 1 / (1 + np.exp(-va_logit))
        auc = roc_auc_score(y[va_idx], va_pred)
        if auc > best_auc:
            best_auc = auc
            best_va = va_pred
            with torch.no_grad():
                te_logit = model(Xte_num_t.to(device), Xte_cat_t.to(device)).cpu().numpy()
            best_te = 1 / (1 + np.exp(-te_logit))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    oof[va_idx] = best_va
    test_pred += best_te / N_FOLDS
    print(f"  Fold {fold+1}: AUC={best_auc:.5f}  best_ep={ep-no_improve+1}/{ep+1}  ({time.time()-ts:.0f}s)", flush=True)

# ─── Save ──────────────────────────────────────────────
print(f"\n[3/4] OOF AUC: {roc_auc_score(y, oof):.5f}", flush=True)
np.savez_compressed(MODELS / "oof_mlp.npz", mlp=oof, y=y)
np.savez_compressed(MODELS / "test_mlp.npz", mlp=test_pred)

print(f"\n[4/4] Done. Total: {(time.time()-t0)/60:.1f} min")
print(f"  Saved: models/oof_mlp.npz, models/test_mlp.npz")
