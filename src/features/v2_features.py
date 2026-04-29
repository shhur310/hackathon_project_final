"""
v2_features.py — ChatGPT 10-prompt EDA에서 도출된 모든 추천 파생변수
=====================================================================
8 함수, ~80개 신규 피처. 모두 row-wise 변환으로 leakage 위험 0.

함수 구성:
  add_age_features          — §1 나이 파생 + 중앙값 매핑
  add_history_features       — §2 횟수·이력 ratio
  add_treatment_token_features — §3 ICSI/IVF/IUI/BLASTOCYST/AH/FER 토큰
  add_embryo_efficiency      — §4 배아·난자 효율 ratio + log1p
  add_transfer_strategy      — §5 transfer_day_bin, fresh × frozen pattern
  add_cause_aggregates_v2    — §6 남/여/부부 cause 집계
  add_donor_ordinals         — §7 난자/정자 기증자 나이 ordinal
  add_missing_signals        — §8 informative NaN flags

통합 함수: add_v2_all_features(df) — 위 8개 일괄 적용
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ========================================================================
# 매핑
# ========================================================================
AGE_TO_ORDINAL = {
    "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
    "만40-42세": 3, "만43-44세": 4, "만45-50세": 5, "알 수 없음": -1,
}
AGE_TO_MID = {
    "만18-34세": 26, "만35-37세": 36, "만38-39세": 38.5,
    "만40-42세": 41, "만43-44세": 43.5, "만45-50세": 47.5, "알 수 없음": np.nan,
}
COUNT_TO_INT = {"0회":0, "1회":1, "2회":2, "3회":3, "4회":4, "5회":5, "6회 이상":6}

DONOR_OOCYTE_AGE_MAP = {"만20세 이하":0, "만21-25세":1, "만26-30세":2, "만31-35세":3, "알 수 없음":-1}
DONOR_SPERM_AGE_MAP = {"만20세 이하":0, "만21-25세":1, "만26-30세":2, "만31-35세":3,
                       "만36-40세":4, "만41-45세":5, "알 수 없음":-1}

COUNT_COLS = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
              "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
              "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"]

# 토큰 (§3)
TREATMENT_TOKENS = ["ICSI", "IVF", "IUI", "BLASTOCYST", "AH", "FER", "Unknown"]

# Informative NaN columns (§5/§8 EDA 검증)
INFORMATIVE_NA_COLS = [
    "배아 이식 경과일",            # Δ=-28.3%p (cancelled cycle)
    "난자 혼합 경과일",            # Δ=-8.8%p
    "임신 시도 또는 마지막 임신 경과 연수",  # Δ=+4.3%p
    "착상 전 유전 검사 사용 여부",
    "배아 해동 경과일",
    "난자 채취 경과일",
]


# ========================================================================
# §1. Age features
# ========================================================================
def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """나이 파생: ordinal, mid-age, threshold flags, unknown."""
    df = df.copy()
    age = df.get("시술 당시 나이", pd.Series("알 수 없음", index=df.index))
    df["age_ord"] = age.map(AGE_TO_ORDINAL).fillna(-1).astype(int)
    df["age_mid"] = age.map(AGE_TO_MID).astype(float)
    df["age_unknown"] = (age == "알 수 없음").astype(int)
    df["age_35p"] = (df["age_ord"] >= 1).astype(int)
    df["age_38p"] = (df["age_ord"] >= 2).astype(int)
    df["age_40p"] = (df["age_ord"] >= 3).astype(int)
    df["age_43p"] = (df["age_ord"] >= 4).astype(int)
    return df


# ========================================================================
# §2. History features
# ========================================================================
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """횟수 ordinal + ratio + prior flags."""
    df = df.copy()
    # 횟수 → 정수
    for c in COUNT_COLS:
        if c in df.columns:
            df[f"{c}_int"] = df[c].map(COUNT_TO_INT).fillna(-1).astype(int)
            df[f"{c}_is_censored"] = (df[c] == "6회 이상").astype(int)

    # Prior flags
    if "총 임신 횟수_int" in df.columns:
        df["prior_pregnancy_any"] = (df["총 임신 횟수_int"] >= 1).astype(int)
    if "총 출산 횟수_int" in df.columns:
        df["prior_live_birth_any"] = (df["총 출산 횟수_int"] >= 1).astype(int)
    if "총 임신 횟수_int" in df.columns and "총 출산 횟수_int" in df.columns:
        df["prior_pregnancy_no_live_birth"] = (
            (df["총 임신 횟수_int"] >= 1) & (df["총 출산 횟수_int"] == 0)
        ).astype(int)

    # Ratios (with NaN for divide-by-zero)
    def safe_ratio(num_col, den_col, name):
        if num_col in df.columns and den_col in df.columns:
            den = df[den_col].replace(0, np.nan)
            df[name] = (df[num_col] / den).fillna(-1).clip(-1, 5)

    safe_ratio("총 임신 횟수_int", "총 시술 횟수_int", "pregnancy_per_treatment")
    safe_ratio("총 출산 횟수_int", "총 시술 횟수_int", "live_birth_per_treatment")
    safe_ratio("총 출산 횟수_int", "총 임신 횟수_int", "live_birth_per_pregnancy")
    safe_ratio("IVF 임신 횟수_int", "IVF 시술 횟수_int", "ivf_preg_per_ivf_tr")
    safe_ratio("DI 임신 횟수_int", "DI 시술 횟수_int", "di_preg_per_di_tr")
    safe_ratio("IVF 출산 횟수_int", "IVF 시술 횟수_int", "ivf_birth_per_ivf_tr")

    return df


# ========================================================================
# §3. Treatment token features
# ========================================================================
def add_treatment_token_features(df: pd.DataFrame) -> pd.DataFrame:
    """특정 시술 유형에서 ICSI/IVF/IUI/BLASTOCYST/AH/FER/Unknown 추출."""
    df = df.copy()
    if "특정 시술 유형" not in df.columns:
        return df
    s = df["특정 시술 유형"].fillna("").astype(str).str.upper()
    for tok in TREATMENT_TOKENS:
        df[f"treat_has_{tok.lower()}"] = s.str.contains(tok.upper(), regex=False).astype(int)
    df["treat_token_count"] = sum(
        df[f"treat_has_{tok.lower()}"] for tok in TREATMENT_TOKENS
    )
    # 첫 토큰 추출 (단순 categorical)
    df["treat_main_first"] = df["특정 시술 유형"].fillna("UNKNOWN").astype(str).str.split('/').str[0]
    return df


# ========================================================================
# §4. Embryo efficiency
# ========================================================================
NUM_EMB_COLS = [
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수",
    "해동된 배아 수", "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수",
    "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
]


def add_embryo_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """배아·난자 수치 log1p + zero/missing flags + 효율 ratio."""
    df = df.copy()

    # log1p + flags
    for c in NUM_EMB_COLS:
        if c in df.columns:
            v = df[c].fillna(0).clip(lower=0)
            df[f"{c}_log1p"] = np.log1p(v)
            df[f"{c}_is_zero"] = (df[c] == 0).astype(int)
            df[f"{c}_is_missing"] = df[c].isna().astype(int)

    # Ratios — divide-by-zero → -1 sentinel
    def ratio(num, den, name):
        if num in df.columns and den in df.columns:
            df[name] = np.where(
                df[den].fillna(0) > 0,
                df[num].fillna(0) / df[den].clip(lower=1e-9),
                -1.0,
            )

    ratio("이식된 배아 수", "총 생성 배아 수", "ratio_transferred_per_created")
    ratio("저장된 배아 수", "총 생성 배아 수", "ratio_stored_per_created")
    ratio("총 생성 배아 수", "혼합된 난자 수", "ratio_created_per_mixed")
    ratio("미세주입에서 생성된 배아 수", "미세주입된 난자 수", "ratio_icsi_efficiency")
    ratio("파트너 정자와 혼합된 난자 수", "혼합된 난자 수", "ratio_partner_sperm")
    ratio("기증자 정자와 혼합된 난자 수", "혼합된 난자 수", "ratio_donor_sperm")

    # Surplus / shortage
    if "총 생성 배아 수" in df.columns and "이식된 배아 수" in df.columns:
        df["embryos_surplus"] = (
            df["총 생성 배아 수"].fillna(0) - df["이식된 배아 수"].fillna(0)
        ).clip(lower=0)

    if "총 생성 배아 수" in df.columns and "저장된 배아 수" in df.columns and "이식된 배아 수" in df.columns:
        df["embryos_unused"] = (
            df["총 생성 배아 수"].fillna(0)
            - df["이식된 배아 수"].fillna(0)
            - df["저장된 배아 수"].fillna(0)
        ).clip(lower=0)

    return df


# ========================================================================
# §5. Transfer strategy
# ========================================================================
def add_transfer_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """transfer_day_bin, fresh × frozen pattern, blastocyst × day."""
    df = df.copy()

    # has_transfer
    if "배아 이식 경과일" in df.columns:
        df["has_embryo_transfer"] = df["배아 이식 경과일"].notna().astype(int)
        df["cancelled_cycle"] = df["배아 이식 경과일"].isna().astype(int)

        # Day bins
        d = df["배아 이식 경과일"]
        df["transfer_day0_2"] = ((d >= 0) & (d <= 2)).astype(int)
        df["transfer_day3"] = (d == 3).astype(int)
        df["transfer_day5"] = (d == 5).astype(int)
        df["transfer_day_blastocyst"] = ((d >= 5) & (d <= 6)).astype(int)

    # Embryo transfer count bins
    if "이식된 배아 수" in df.columns:
        n = df["이식된 배아 수"].fillna(-1)
        df["transfer_count_0"] = (n == 0).astype(int)
        df["transfer_count_1"] = (n == 1).astype(int)
        df["transfer_count_2"] = (n == 2).astype(int)
        df["transfer_count_3p"] = (n >= 3).astype(int)
        df["transfer_count_NA"] = (n == -1).astype(int)

    # Fresh × Frozen × Donor pattern
    if all(c in df.columns for c in ["신선 배아 사용 여부", "동결 배아 사용 여부", "기증 배아 사용 여부"]):
        f = df["신선 배아 사용 여부"].fillna(-1).astype(str)
        fz = df["동결 배아 사용 여부"].fillna(-1).astype(str)
        dn = df["기증 배아 사용 여부"].fillna(-1).astype(str)
        df["fresh_frozen_donor_pattern"] = "F" + f + "_Z" + fz + "_D" + dn

    # Blastocyst × day5 (cleavage vs blastocyst proxy via day)
    if "transfer_day5" in df.columns and "treat_has_blastocyst" in df.columns:
        df["blastocyst_x_day5"] = df["transfer_day5"] * df["treat_has_blastocyst"]

    return df


# ========================================================================
# §6. Cause aggregates v2
# ========================================================================
def add_cause_aggregates_v2(df: pd.DataFrame) -> pd.DataFrame:
    """남/여/부부/sperm cause counts + binary aggregates."""
    df = df.copy()
    cause_cols = [c for c in df.columns if "불임 원인" in c]
    binary_cause = [c for c in cause_cols if df[c].dropna().isin([0,1]).all()]

    if not binary_cause:
        return df

    df["cause_total_count"] = df[binary_cause].sum(axis=1)
    df["cause_any"] = (df["cause_total_count"] > 0).astype(int)
    df["cause_none"] = (df["cause_total_count"] == 0).astype(int)
    df["cause_multi"] = (df["cause_total_count"] >= 2).astype(int)

    male_cols = [c for c in binary_cause if any(k in c for k in ["남성","정자"])]
    female_cols = [c for c in binary_cause if any(k in c for k in ["여성","난관","자궁","배란"])]
    couple_cols = [c for c in binary_cause if "부부" in c]
    sperm_cols = [c for c in binary_cause if "정자" in c]

    df["cause_male_count"] = df[male_cols].sum(axis=1) if male_cols else 0
    df["cause_female_count"] = df[female_cols].sum(axis=1) if female_cols else 0
    df["cause_couple_count"] = df[couple_cols].sum(axis=1) if couple_cols else 0
    df["cause_sperm_count"] = df[sperm_cols].sum(axis=1) if sperm_cols else 0

    df["cause_male_any"] = (df["cause_male_count"] > 0).astype(int)
    df["cause_female_any"] = (df["cause_female_count"] > 0).astype(int)
    df["cause_male_female_both"] = (
        (df["cause_male_any"] == 1) & (df["cause_female_any"] == 1)
    ).astype(int)

    if "불명확 불임 원인" in df.columns:
        df["unexplained_or_none"] = (
            (df["cause_total_count"] == 0) | (df["불명확 불임 원인"] == 1)
        ).astype(int)

    return df


# ========================================================================
# §7. Donor ordinals
# ========================================================================
def add_donor_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """난자/정자 기증자 나이 ordinal + cross with recipient age."""
    df = df.copy()

    if "난자 기증자 나이" in df.columns:
        df["egg_donor_age_ord"] = df["난자 기증자 나이"].map(DONOR_OOCYTE_AGE_MAP).fillna(-1).astype(int)
        df["egg_donor_age_unknown"] = (df["난자 기증자 나이"] == "알 수 없음").astype(int)
        df["egg_donor_age_optimal"] = df["난자 기증자 나이"].isin(["만21-25세", "만26-30세"]).astype(int)

    if "정자 기증자 나이" in df.columns:
        df["sperm_donor_age_ord"] = df["정자 기증자 나이"].map(DONOR_SPERM_AGE_MAP).fillna(-1).astype(int)
        df["sperm_donor_age_unknown"] = (df["정자 기증자 나이"] == "알 수 없음").astype(int)

    # Source flags (egg, sperm)
    if "난자 출처" in df.columns:
        df["uses_donor_oocyte"] = (df["난자 출처"] == "기증 제공").astype(int)
    if "정자 출처" in df.columns:
        df["uses_donor_sperm"] = (df["정자 출처"] == "기증 제공").astype(int)

    # Cross: 난자 출처 × 정자 출처
    if "난자 출처" in df.columns and "정자 출처" in df.columns:
        df["egg_sperm_source_cross"] = (
            df["난자 출처"].fillna("NA").astype(str) + "_" +
            df["정자 출처"].fillna("NA").astype(str)
        )

    # age × egg source interaction
    if "age_ord" in df.columns and "uses_donor_oocyte" in df.columns:
        df["age_x_egg_donor"] = df["age_ord"] * df["uses_donor_oocyte"]
        df["age_x_no_egg_donor"] = df["age_ord"] * (1 - df["uses_donor_oocyte"])

    return df


# ========================================================================
# §8. Missing signals
# ========================================================================
def add_missing_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Informative NaN flags + row-level missing count."""
    df = df.copy()
    flags = []
    for col in INFORMATIVE_NA_COLS:
        if col in df.columns:
            flag = f"{col}_isna"
            df[flag] = df[col].isna().astype(int)
            flags.append(flag)
    if flags:
        df["sum_informative_na"] = df[flags].sum(axis=1)

    # 행 단위 missing count
    df["row_missing_count"] = df.isna().sum(axis=1)

    return df


# ========================================================================
# §9. Selected interactions
# ========================================================================
def add_priority_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """우선순위 interaction 피처 (§9 prompt)."""
    df = df.copy()

    age = df.get("age_ord", pd.Series(0, index=df.index))
    n_transfer = df.get("이식된 배아 수", pd.Series(0, index=df.index)).fillna(0)
    transfer_day = df.get("배아 이식 경과일", pd.Series(0, index=df.index)).fillna(0)

    df["age_x_n_transfer"] = age * n_transfer
    df["age_x_transfer_day"] = age * transfer_day

    if "treat_has_icsi" in df.columns:
        df["age_x_icsi"] = age * df["treat_has_icsi"]
    if "treat_has_blastocyst" in df.columns:
        df["age_x_blastocyst"] = age * df["treat_has_blastocyst"]
    if "동결 배아 사용 여부" in df.columns:
        df["age_x_FET"] = age * df["동결 배아 사용 여부"].fillna(0).astype(int)

    return df


# ========================================================================
# 통합
# ========================================================================
def add_v2_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """모든 v2 피처를 일괄 적용 (순서 중요: age → history → token → embryo → ...)."""
    df = add_age_features(df)
    df = add_history_features(df)
    df = add_treatment_token_features(df)
    df = add_embryo_efficiency(df)
    df = add_transfer_strategy(df)
    df = add_cause_aggregates_v2(df)
    df = add_donor_ordinals(df)
    df = add_missing_signals(df)
    df = add_priority_interactions(df)
    return df


# ========================================================================
# Smoke test
# ========================================================================
if __name__ == "__main__":
    from pathlib import Path
    HERE = Path(__file__).resolve().parents[2]
    train = pd.read_csv(HERE / "data" / "train.csv", nrows=5000)
    before = train.shape[1]
    out = add_v2_all_features(train)
    after = out.shape[1]
    new_cols = [c for c in out.columns if c not in train.columns]
    print(f"Original: {before} cols")
    print(f"After v2: {after} cols (+{after-before})")
    print(f"\nNew columns ({len(new_cols)}):")
    for c in new_cols:
        n_pos = (out[c] != 0).sum() if pd.api.types.is_numeric_dtype(out[c]) else out[c].nunique()
        print(f"  {c:50s}  non-zero/unique={n_pos}")
