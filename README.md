# 난임 환자 임신 성공 여부 예측 — DACON Hackathon 2026 🥉

> **최종 결과: Public LB AUC 0.74235 — 3rd Place** (1등 0.74246과 -0.00011 차이)
>
> 256K cycles, 8-blend ensemble (6 GBM + 2 NN), Optuna 90 trials hyperparameter 최적화, OOF→LB 일반화 검증.

---

## 🎯 최종 성능

| 지표 | 값 |
|---|---|
| **Public LB AUC** | **0.74235 (3rd)** |
| OOF AUC (5-fold CV) | 0.74101 |
| OOF → LB 격차 | +0.00134 (overfit 없음) |
| 1등과의 차이 | -0.00011 |
| 학습 데이터 | 256,351 cycles |
| 테스트 데이터 | 90,067 cycles |
| 최종 features | 210 (원본 68 + v2 142개) |
| 최종 모델 | 8-blend ensemble |

---

## 🏆 핵심 차별점 5가지

| | 차별점 | 내용 |
|---|---|---|
| 01 | **HFEA 메타분석 도메인 검증** | 연령별 임신율 6/6 일치, blastocyst 우위 등 정량 일치 |
| 02 | **체계적 Feature Engineering** | 68 → 210 features, 7가지 카테고리 |
| 03 | **2-tier Leakage 방어** | 사전 (validation/) + 사후 (leakage_check.py) = 8/8 PASS |
| 04 | **Multi-Architecture Ensemble** | GBM 6개 + NN 2개 (MLP, TabNet) — decorrelated diversity |
| 05 | **Robust Generalization** | OOF 0.74101 → LB 0.74235 (+0.00134), train overfit 없음 입증 |

---

## 📊 점진적 성능 개선 — 4단계

| 단계 | OOF AUC | LB AUC |
|---|---|---|
| v1 baseline (default LGBM/XGB/CAT) | 0.74045 | — |
| v1 + Optuna CAT | 0.74066 | 0.74217 |
| v2 (모든 모델 Optuna) | 0.74074 | — |
| **8-blend ensemble (final)** | **0.74101** | **0.74235** |

---

## 🧠 모델 아키텍처

### GBM 6개

| 모델 | OOF AUC | 8-blend weight |
|---|---|---|
| v1 LightGBM (default) | 0.73937 | 0.118 |
| v1 XGBoost (default) | 0.74017 | 0.102 |
| v1 CatBoost (Optuna) | 0.74037 | 0.000 |
| v2 LightGBM (Optuna) | 0.74023 | 0.000 |
| v2 XGBoost (Optuna) | 0.74041 | 0.207 |
| v2 CatBoost (Optuna) | 0.74037 | 0.229 |

### Neural Network 2개

| 모델 | OOF AUC | Architecture | weight |
|---|---|---|---|
| MLP (PyTorch + Mac M2 MPS) | 0.73925 | [256,128,64] + cat embed | 0.234 |
| TabNet (Dreamquark) | 0.73750 | n_d=32, n_a=32, n_steps=4 | 0.110 |

→ **Final blend AUC: 0.74101** (Nelder-Mead optimal weights)

### 시도했으나 미채택

- **FT-Transformer**: MPS attention hang으로 abandoned
- **Multi-seed bagging (3 seeds)**: OOF 0.74087로 효과 미미

---

## 📁 프로젝트 구조

\`\`\`
.
├── README.md
├── data/                              # train.csv, test.csv, sample_submission.csv
├── notebooks/eda_master_v2.ipynb      # 49 cells, EDA
├── docs/v2_strategy.md                # 전략 노트
├── src/
│   ├── __init__.py
│   ├── archive/                       # abandoned (참고용)
│   │   ├── train_v2_seed7.py
│   │   ├── train_v2_seed123.py
│   │   ├── train_v2_multiseed.py
│   │   ├── blend_6model.py
│   │   └── blend_7model.py
│   ├── features/
│   │   ├── v2_features.py             # 142 features
│   │   └── oof_target_encoder.py      # Leakage-safe TE
│   ├── validation/leakage_check.py    # 5-tier 사전 검증
│   ├── leakage_check.py               # 8-tier 사후 검증 (8/8 PASS)
│   ├── tune_optuna{,_lgbm,_xgb}.py    # 3 model Optuna (30 trials each)
│   ├── train_v1_optunaCAT.py
│   ├── train_v2_full.py
│   ├── train_mlp.py
│   ├── train_tabnet.py
│   ├── train_ft_transformer.py        # abandoned
│   └── blend_8model_tabnet.py         # ★ FINAL (LB 0.74235)
├── models/
│   ├── optuna_summary{,_lgbm,_xgb}.json
│   ├── oof_*.npz, test_*.npz
│   └── *.cbm, *.json, *.pkl
└── submission/
    ├── submission_8blend_tabnet.csv   # ★ FINAL (LB 0.74235)
    └── submission_v2_OPTUNA_CAT_only.csv  # 백업 (LB 0.74217)
\`\`\`

---

## 🚀 재현 절차

### 0. 환경 설정

\`\`\`bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost \
            optuna joblib torch pytorch-tabnet
brew install libomp  # macOS LightGBM
\`\`\`

### 1. 사전 검증 (5초)

\`\`\`bash
python -m src.validation.leakage_check
\`\`\`

### 2. Optuna 튜닝 (3 모델, 각 30 trials, ~70분)

\`\`\`bash
python src/tune_optuna.py        --trials 30  # CAT (~30분)
python src/tune_optuna_lgbm.py   --trials 30  # LGBM (~19분)
python src/tune_optuna_xgb.py    --trials 30  # XGB (~22분)
\`\`\`

### 3. GBM 학습

\`\`\`bash
python src/train_v1_optunaCAT.py   # ~25분
python src/train_v2_full.py        # ~30분
\`\`\`

### 4. NN 학습 (MPS / CUDA)

\`\`\`bash
python src/train_mlp.py            # ~8분
python src/train_tabnet.py         # ~49분
\`\`\`

### 5. 최종 8-blend ensemble

\`\`\`bash
python src/blend_8model_tabnet.py  # → submission_8blend_tabnet.csv
\`\`\`

### 6. 사후 검증 (8/8 PASS)

\`\`\`bash
python src/leakage_check.py
\`\`\`

---

## 📐 기술적 디자인 결정

### Feature Engineering — v2 (142 신규)

| 카테고리 | 추가 수 | 예시 |
|---|---|---|
| 연령 변환 | +7 | age_ord, age_35p, age_unknown |
| 시술 횟수 정수화 | +12 | *_int + *_is_censored |
| 비율 변수 | +10 | pregnancy_per_treatment |
| log1p / zero / na flags | +33 | 11개 카운트 × 3 변환 |
| 원인 분석 카운트 | +12 | cause_male/female/couple |
| 이식 패턴 | +9 | transfer_day_blastocyst |
| 상호작용 | +8 | age_x_egg_donor, age_x_n_transfer |

### OOF Target Encoder

- **Smoothing α=50**, Unknown → global mean
- **Fold-internal fit**: train→val leakage 차단
- **Test 시점**: 전체 train labels로 fit, test labels 미사용

### 8-blend Nelder-Mead

- 8차원 simplex search
- Constraints: weights ≥ 0, Σ = 1
- 5 random seed initializations (Dirichlet)
- 1000 max iter / seed

### 2-tier Leakage 방어

**Pre-training (5)**: ID, future-leak keyword, 컬럼 정렬, class balance, drift  
**Post-training (8)**: 위 5 + v2 determinism + OOF TE 분리 + Optuna train-only + submission sanity + same y + valid weights

→ **8/8 PASS, NO DATA LEAKAGE DETECTED**

---

## 📊 HFEA 메타분석 비교

| 모델 | AUC | 발표 |
|---|---|---|
| Templeton | 0.690 | 1996 |
| Nelson | 0.710 | 2011 |
| McLernon | 0.740 | 2016 |
| Choi | 0.740 | 2019 |
| **본 모델 (Public LB)** | **0.74235** | **2026** |

→ HFEA AUC range (0.69-0.76) 상위권 도달.

---

## ⚠️ 한계 및 향후 개선

### 한계

| | |
|---|---|
| 01 Cycle-level 데이터 한계 | BMI, AMH 등 환자 정보 부재 |
| 02 Hormone profile 부재 | 배란 자극 반응 평가 불가 |
| 03 AUC 0.74 ceiling | 8-blend로도 +0.0001 수준 |
| 04 Cross-clinic variation | 클리닉별 protocol 차이 미반영 |

### 향후 개선

| | | 예상 |
|---|---|---|
| 01 Patient-level data | 환자 ID 기반 longitudinal | ~0.770 (+0.028) |
| 02 AMH/BMI 외부 데이터 | 가능시 | +0.04 OOF |
| 03 Stacking meta-learner | L2 Ridge | — |
| 04 Pseudo-labeling | high-confidence test → train | — |
| 05 FT-Transformer 재시도 | CUDA 환경 | — |

---

## 📚 인용

- **2025 Hum Reprod Meta-analysis** — IVF/ICSI live birth, 72 studies, McLernon AUC 0.73
- **Zou Y et al.** *J Transl Med* 2025 (RF AUC 0.808, 11,728 ART records)
- **Sahin G et al.** *Hum Reprod Open* 2021 PMC8240131 (cumulative live birth)
- **eLife 2024** blastocyst image + clinical → AUC 0.77
- **Arik SO, Pfister T.** TabNet, *AAAI* 2021
- **Gorishniy Y et al.** FT-Transformer, *NeurIPS* 2021

---

## 🛠️ 환경

- **Hardware**: Mac mini M2 (16GB, MPS)
- **Python**: 3.14
- **Libraries**: lightgbm, xgboost, catboost, PyTorch 2.11 (MPS), pytorch-tabnet, optuna (TPE), scikit-learn, pandas 4

---

## ✅ 재현성

- All random seeds 고정 (42, 7, 123, 456)
- StratifiedKFold(5, shuffle=True, random_state=42) 일관
- Optuna JSON 저장
- OOF/test NPZ 저장 (재학습 없이 blend 재현 가능)
- Leakage 8/8 PASS

---
- **Final Result**: 🥉 **3rd Place — Public LB 0.74235**