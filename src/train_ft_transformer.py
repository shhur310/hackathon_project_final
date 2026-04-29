"""
FT-Transformer 5-fold OOF training.
Yandex 2021 — tabular SOTA architecture.
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
torch.manual_seed(123)  # Different seed than MLP
np.random.seed(123)

# ─── Data prep (identical to MLP) ────────────────────────
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

cat_dim_list = []
for c in obj_cols:
    arr_X = X[c].values
    arr_test = X_test[c].values
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
    all_vals = np.concatenate([s_X, s_test])
    uniq = sorted(set(all_vals.tolist()))
    mapping = {v: i for i, v in enumerate(uniq)}
    X[c] = np.array([mapping[v] for v in s_X], dtype=np.int32)
    X_test[c] = np.array([mapping[v] for v in s_test], dtype=np.int32)
    cat_dim_list.append(len(uniq))

for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype("float32")
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(0).astype("float32")

X_num_all = X[num_cols].values.astype("float32")
X_cat_all = X[obj_cols].values.astype("int64")
X_test_num = X_test[num_cols].values.astype("float32")
X_test_cat = X_test[obj_cols].values.astype("int64")
print(f"  X_num={X_num_all.shape}, X_cat={X_cat_all.shape}", flush=True)

# ─── FT-Transformer ────────────────────────────────────
class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer (Yandex 2021)."""
    def __init__(self, num_n, cat_dims, d_model=64, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.num_n = num_n
        self.n_cat = len(cat_dims)
        # Numerical tokenization: each numeric → token
        self.num_weight = nn.Parameter(torch.randn(num_n, d_model) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(num_n, d_model))
        # Categorical embeddings
        self.cat_embs = nn.ModuleList([nn.Embedding(d, d_model) for d in cat_dims])
        # CLS token
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1)
        )

    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        # Numerical tokens: (B, num_n, d)
        num_tok = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
        # Categorical tokens: (B, n_cat, d)
        if self.n_cat > 0:
            cat_toks = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)],
                dim=1,
            )
            tokens = torch.cat([num_tok, cat_toks], dim=1)
        else:
            tokens = num_tok
        # Add CLS
        cls = self.cls_tok.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        # Transformer
        x = self.transformer(x)
        # CLS head
        return self.head(x[:, 0]).squeeze(-1)

# ─── Training ──────────────────────────────────────────
N_FOLDS = 5
EPOCHS = 20
BATCH = 1024  # Smaller batch for transformer (memory)
LR = 5e-4
WD = 1e-5
PATIENCE = 4

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

print(f"\n[2/4] FT-Transformer training (epochs={EPOCHS}, batch={BATCH})", flush=True)
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

    model = FTTransformer(num_n=Xtr_num.shape[1], cat_dims=cat_dim_list).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        model.eval()
        bs = 2048
        va_preds = []
        with torch.no_grad():
            for i in range(0, len(Xva_num_t), bs):
                logit = model(Xva_num_t[i:i+bs].to(device), Xva_cat_t[i:i+bs].to(device)).cpu().numpy()
                va_preds.append(logit)
        va_logit = np.concatenate(va_preds)
        va_pred = 1 / (1 + np.exp(-va_logit))
        auc = roc_auc_score(y[va_idx], va_pred)
        if auc > best_auc:
            best_auc = auc
            best_va = va_pred
            te_preds = []
            with torch.no_grad():
                for i in range(0, len(Xte_num_t), bs):
                    logit = model(Xte_num_t[i:i+bs].to(device), Xte_cat_t[i:i+bs].to(device)).cpu().numpy()
                    te_preds.append(logit)
            te_logit = np.concatenate(te_preds)
            best_te = 1 / (1 + np.exp(-te_logit))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    oof[va_idx] = best_va
    test_pred += best_te / N_FOLDS
    print(f"  Fold {fold+1}: AUC={best_auc:.5f}  best_ep={ep-no_improve+1}/{ep+1}  ({time.time()-ts:.0f}s)", flush=True)

print(f"\n[3/4] OOF AUC: {roc_auc_score(y, oof):.5f}", flush=True)
np.savez_compressed(MODELS / "oof_ft.npz", ft=oof, y=y)
np.savez_compressed(MODELS / "test_ft.npz", ft=test_pred)
print(f"\n[4/4] Done. Total: {(time.time()-t0)/60:.1f} min")
print(f"  Saved: models/oof_ft.npz, models/test_ft.npz")
