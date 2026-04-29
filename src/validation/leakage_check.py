"""leakage_check.py — §10 prompt 모든 항목 자동 점검."""
from __future__ import annotations
import pandas as pd


def check_all(train_csv: str, test_csv: str, target: str = "임신 성공 여부", id_col: str = "ID") -> dict:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    result = {}

    # 1. ID
    susp_id = [c for c in train.columns if c.upper() == "ID" or c.endswith("_id")]
    result["id_check"] = {"suspicious_id_cols": susp_id}

    # 2. Future-leak (post-outcome 의심)
    suspect_keywords = ["임신 성공 여부", "live_birth", "outcome", "결과"]
    fl = [c for c in test.columns if c in [target] or any(k in c for k in suspect_keywords)]
    result["future_leak_in_test"] = fl

    # 3. Train/test column match
    tr_cols = set(train.columns) - {target}
    te_cols = set(test.columns)
    result["only_train"] = sorted(tr_cols - te_cols)
    result["only_test"] = sorted(te_cols - tr_cols)
    result["columns_aligned"] = (len(tr_cols ^ te_cols) == 0)

    # 4. Class balance
    if target in train.columns:
        result["class_balance"] = {
            "positive_rate": float(train[target].mean()),
            "n_pos": int(train[target].sum()),
            "n_neg": int((1 - train[target]).sum()),
        }

    # 5. 시기 코드 overlap
    if "시술 시기 코드" in train.columns and "시술 시기 코드" in test.columns:
        tr_p = set(train["시술 시기 코드"].dropna().unique())
        te_p = set(test["시술 시기 코드"].dropna().unique())
        result["period_overlap"] = {
            "train_only": sorted(tr_p - te_p),
            "test_only": sorted(te_p - tr_p),
            "shared": len(tr_p & te_p),
            "drift_warning": len(te_p - tr_p) > 0,
        }

    result["n_train"] = len(train)
    result["n_test"] = len(test)
    return result


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parents[2]
    res = check_all(str(HERE / "data" / "train.csv"), str(HERE / "data" / "test.csv"))
    print(json.dumps(res, ensure_ascii=False, indent=2))
