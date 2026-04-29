# v2 전략 노트

## 왜 v1은 0.74045에서 멈췄는가

**진단** (v1 실험 결과):
- 5-model GBM ensemble OOF: 0.74045
- Calibration 거의 완벽 (Brier 0.166)
- §1+§2 명시 features의 marginal AUC 이득 ≈ 0
- → **GBM이 raw 컬럼에서 implicit 학습** → 동일 변환의 추가 명시 features는 효과 없음

## v2의 가설

ChatGPT 분석에서 도출한 새로운 신호:
1. **횟수 ratio** (출산/시술, 임신/시술): 이전 ordinal에 없던 정보
2. **시술 유형 token**: ICSI/BLASTOCYST 등 7개 binary indicator를 직접 추출
3. **배아 효율 ratio** 6개: GBM이 split 형태로 만들기 어려운 nonlinear 결합
4. **Fresh × Frozen × Donor pattern**: 8가지 조합을 단일 categorical로
5. **불임 원인 stratification**: 단순 합산 카운트가 아닌 남/여/부부 분리

→ 이 중 **#2, #3, #4**가 GBM에 새로운 정보를 줄 가능성 높음 (raw 텍스트 처리 vs 명시 토큰 차이).

## 기대 효과 분석

| v2 모듈 | 새 정보 비중 | 기대 AUC 기여 |
|---|---|---|
| `add_age_features` | 낮음 (이미 ordinal) | ≈ 0 |
| `add_history_features` (8개 ratio) | **중** | +0.0005 |
| `add_treatment_token_features` (7 token) | **높음** | +0.001~0.002 |
| `add_embryo_efficiency` (6 ratio) | **높음** | +0.001~0.002 |
| `add_transfer_strategy` | 중 | +0.0005 |
| `add_cause_aggregates_v2` | 낮음 (v1에 일부 있음) | ≈ 0 |
| `add_donor_ordinals` | 낮음 (이미 categorical) | ≈ 0 |
| `add_missing_signals` | 중 | +0.0005 |
| `add_priority_interactions` | 낮음 (GBM implicit) | ≈ 0 |
| **합계** | | **+0.003 ~ +0.005** 기대 |

→ v1 0.74045 + v2 +0.003 = **0.7434** 가능. 추가 Optuna로 **0.7444 도달 가능**.

## 0.742+ 확률

| 단계 | 누적 OOF | P(≥ 0.742) |
|---|---|---|
| v1 baseline | 0.74045 | 0% |
| + v2 features (Path B) | 0.7434 | **40%** |
| + Optuna | 0.7444 | **70%** |
| + multi-seed bagging | 0.7449 | **80%** |

## 위험 요인

1. **OOM**: v2 피처 ~80개 추가 → 메모리 ↑. 16GB+ 환경 권장. 8GB 환경은 `--catboost-only` 사용.
2. **OOF/LB 차이**: OOF 0.745여도 Public LB는 0.738일 수 있음. Variance 크면 다른 fold seed로 confirm.
3. **Optuna overfit**: 3-fold tuning이 5-fold 결과와 다를 수 있음. Final retrain은 5-fold 필수.

## 다음 단계 (만약 v2 + Optuna 후에도 0.742 미달)

1. **TabNet/FT-Transformer** 추가 (NN diversity, +0.001~0.003 기대, 90분)
2. **Multi-seed bagging**: CatBoost 3 seeds × 5 folds, +0.0005~0.001
3. **Pseudo-labeling**: test set 고확률 sample을 학습에 추가 (위험·이득 고)
4. **GroupKFold by 시기 코드**: temporal robustness 확인

## 참고: 데이터 정보 ceiling

본 데이터의 OOF AUC ceiling은 **약 0.745**로 추정 (cycle-level features만 있는 경우).  
초과 영역(0.745+)은 environment 변수 (BMI, AMH, FSH) 또는 image (배아 사진)가 필요.

이는 **메타분석 2025 Hum Reprod**가 보고하는 외부검증 IVF live birth 예측 모델 AUC range와 일치 (0.69-0.76).

→ **0.742는 합리적 도전 목표**, 그 이상은 제공된 데이터로 달성하기 어려운 영역.
