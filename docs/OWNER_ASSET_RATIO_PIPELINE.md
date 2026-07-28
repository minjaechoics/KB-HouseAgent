# 수원시 다가구 전세금/집주인 총자산 비율 확률모델

## 1. 결론

이 모듈은 `D/A`를 하나의 회귀 타깃으로 만들지 않는다. 호실 수, 호실별 보증금,
건물가치, 임대인 기타자산 prior를 서로 다른 데이터셋에서 독립 학습한 후
Monte Carlo에서 결합한다. 개별 집주인 전재산 라벨이 없다는 사실을 API와
모델카드에 유지한다.

```text
건축물대장 ──> 등록 호실 수 분포 ─┐
임대차 RTMS ─> 호실 보증금 분위수 ├─> 20,000회 결합 ─> D/A 분포
매매 RTMS ───> 건물가치 분위수 ───┤
가계금융복지조사 ─> K 가중 prior ─┘
```

## 2. 공식 데이터

- 건축물대장은 국토교통부 건축HUB의 표제부를 기본으로 하며 관리 PK 변경을
  고려한다.
- 임대차·매매는 국토교통부 단독/다가구 실거래 API를 사용한다.
- 집주인 기타자산 prior는 MDIS에서 사용 승인을 받아 내려받은 가계금융복지조사
  가구 마이크로데이터만 사용한다.

API 키는 `MOLIT_SERVICE_KEY`(기존 배포명 `MOLIT_RTMS_SERVICE_KEY`도 지원),
`MOLIT_BUILDING_HUB_KEY` 환경변수로만 받는다.

## 3. 데이터 누수 방지

- 지역 임대차 집계는 현재 계약월을 `shift(1)`한 뒤 3·12개월 rolling한다.
- 개발 법정동과 spatial test 법정동은 겹치지 않는다.
- 2024/2025/2026 시간 분리를 우선하며 부족할 때만 시간순 비율 분리를 쓴다.
- `medium/low` 건물 매칭은 가치 직접 라벨에서 제외한다.
- 조사자료는 조사연도로 나누고 표본가중치를 보존한다.

## 4. 모델

### 4.1 호실 수

NB2 regression은 softplus 평균과 학습된 dispersion으로 count likelihood를
최적화한다. Poisson regression과 histogram gradient boosting Poisson을 같은
validation에서 비교하며 MAE가 가장 낮은 후보를 저장한다.

### 4.2 보증금과 건물가치

0.1/0.25/0.5/0.75/0.9 분위수를 각각 학습한다. 추론 시 crossing을 정렬하고
분위수 사이를 piecewise-linear inverse CDF로 표본화한다. 정규분포로 변환하지
않는다.

### 4.3 집주인 기타자산

`K=(총자산-임대용부동산자산)/임대용부동산자산`을 조사 가중 empirical
bootstrap으로 표본화한다. 정확 그룹 표본이 부족하면 자산구간 완화→수도권
임대인→전국 임대인 순으로 fallback하며 횟수를 결과에 기록한다.

### 4.4 결합과 민감도

호실 활용률은 학습 라벨이 확보되기 전까지 Beta prior이며 학습 결과라고
표현하지 않는다. 동일 건물 호실의 공통 효과는 log random effect로 반영한다.
각 입력을 중앙값에 고정한 ablation으로 최종 비율 분산 감소 기여도를 계산한다.

## 5. 실행

```powershell
$env:MOLIT_SERVICE_KEY='재발급한 키'
$env:MOLIT_BUILDING_HUB_KEY='건축HUB 키'

py -m src.cli collect-buildings --region suwon --legal-dong-codes codes.txt
py -m src.cli collect-leases --region suwon
py -m src.cli collect-sales --region suwon
py -m src.cli preprocess
py -m src.cli train-unit-model
py -m src.cli train-deposit-model
py -m src.cli train-value-model
py -m src.cli build-owner-prior --survey AHS.csv --survey-mapping ahs_2025.yaml
py -m src.cli train-all --survey AHS.csv --survey-mapping ahs_2025.yaml
py -m src.cli evaluate-all
py -m src.cli infer --building-id BUILDING_ID --samples 20000
```

현재 보관된 2022~2025 가구마스터로 actual artifact를 재학습하는 명령:

```powershell
py -m src.cli preprocess
py -m src.cli train-all `
  --survey data/downloaded/owner_asset_ratio/ahs/raw `
  --survey-mapping configs/owner_asset_ratio/survey_schema_mapping.ahs_public_2022_2025.yaml `
  --artifact models/owner_asset_ratio/owner_asset_ratio_actual.joblib
py -m src.cli evaluate-all `
  --artifact models/owner_asset_ratio/owner_asset_ratio_actual.joblib
py -m src.cli infer `
  --artifact models/owner_asset_ratio/owner_asset_ratio_actual.joblib `
  --building-id HUB-8afd698677db742af7bb `
  --samples 20000 `
  --seed 20260728
```

원자료 없이 코드 경로만 확인:

```powershell
py -m src.cli smoke-train --samples 20000
py -m src.cli evaluate-all `
  --artifact models/owner_asset_ratio/owner_asset_ratio_synthetic_smoke.joblib `
  --allow-synthetic
```

## 6. API

`POST /api/properties/owner-asset-ratio`

```json
{
  "session_id": "SESSION",
  "property_id": "PROPERTY",
  "samples": 20000,
  "seed": 20260728,
  "occupancy_scenario": "baseline"
}
```

actual artifact가 없으면 409를 반환한다. synthetic artifact는 서버에서 로드할
수 없다. 성공한 실행은 `decision_run_id`로 입력, seed, 모델·데이터 버전과
fallback을 감사할 수 있다.

## 7. 2026-07-28 실제 학습 스냅샷

- 건축HUB 9,482건
- 전월세 RTMS 34,095건
- 매매 RTMS 1,697건 중 건물가치 high-confidence 라벨 173건
- 가계금융복지조사 원행 73,026건 중 임대인 prior 9,371건
- 조사연도 분리: 2022~2023 train / 2024 validation / 2025 test
- 보증금 test 중앙 절대오차 355.58만원, 80% coverage 78.35%
- 건물가치 test 중앙 APE 22.75%, 80% coverage 85.11% (단, 47건)
- 2025 조사 모집단 분위수 손실 0.00693

상세 수치와 제한은 `reports/owner_asset_ratio/evaluation_report.md`,
기계 판독 결과는 `reports/owner_asset_ratio/actual_evaluation.json`에 있다.

## 8. 참고 자료

- 국토교통부 건축HUB 건축물대장정보:
  https://www.data.go.kr/data/15134735/openapi.do
- 국토교통부 단독/다가구 실거래 정보:
  https://www.data.go.kr/tcs/dss/selectDataSetList.do?dType=API
- 통계청 MDIS 가계금융복지조사:
  https://mdis.kostat.go.kr/
