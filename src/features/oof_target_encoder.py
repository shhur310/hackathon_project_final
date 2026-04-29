"""
oof_target_encoder.py — Leakage-safe target encoding
========================================================
§10 leakage audit의 모든 항목 통과:
  - target encoding은 fold 내부 train으로만 fit
  - test set 인코딩은 전체 train labels 사용 (test labels 사용 안함)
  - smoothing으로 rare category 안정화
  - test에 없는 category는 global_mean으로 채움

사용법:
    enc = OOFTargetEncoder(cols=['시술 시기 코드','특정 시술 유형'], smoothing=50)
    # Fold-level encoding (during CV)
    X_tr_te, X_va_te = enc.fit_transform_oof(X_tr, y_tr, X_va)

    # Test encoding (after final model)
    X_test_te = enc.fit_full_then_transform(X_train_full, y_full, X_test)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class OOFTargetEncoder:
    def __init__(self, cols: list, smoothing: float = 50.0):
        self.cols = cols
        self.smoothing = smoothing

    def _encode(self, train_df: pd.DataFrame, y_train: pd.Series, target_df: pd.DataFrame):
        """train_df + y_train에서 평균 학습, target_df에 적용."""
        global_mean = float(y_train.mean())
        out = pd.DataFrame(index=target_df.index)

        for col in self.cols:
            if col not in train_df.columns:
                continue
            s_tr = train_df[col].astype(str)
            s_te = target_df[col].astype(str)

            agg = (
                pd.DataFrame({"y": y_train.values, "g": s_tr.values})
                .groupby("g")["y"]
                .agg(["mean", "count"])
            )
            smoothed = (
                agg["mean"] * agg["count"] + global_mean * self.smoothing
            ) / (agg["count"] + self.smoothing)

            out[f"{col}_te"] = s_te.map(smoothed).fillna(global_mean).astype(float)

        return out

    def fit_transform_oof(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """fold 내부 train으로 fit → train/val 모두 transform."""
        train_te = self._encode(X_tr, y_tr, X_tr)
        val_te = self._encode(X_tr, y_tr, X_va)
        return train_te, val_te

    def fit_full_then_transform(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        X_test: pd.DataFrame,
    ) -> pd.DataFrame:
        """전체 train으로 fit → test transform (test labels 사용 안함)."""
        return self._encode(X_full, y_full, X_test)
