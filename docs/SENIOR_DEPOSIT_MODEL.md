# 수원시 다가구주택 선순위 임차보증금 확률모델

## 1. 모델이 답하는 것

이 모듈은 신규 임차인이 입주할 기준일에 이미 존재할 가능성이 있는 다른 호실의
임차보증금 총액을 확률적으로 추정한다.

- `estimated_total_deposit`: 현재 점유 중이라고 추정한 다른 호실의 보증금 총합
- `estimated_senior_deposit`: 선순위 시나리오 확률까지 적용한 기준 추정
- `conservative_upper_deposit`: 다른 점유 호실 전원을 선순위로 보는 보수적 상한

MVP에서는 신규 입주 대상 호실이 비어 인도된다는 상황을 가정하고 대상 한 호실을
합계에서 제외한다. 따라서 conservative 모드에서는 위 첫 번째와 세 번째 값이
같다. 어느 값도 전입세대확인서·확정일자를 확인한 법적 확정값이 아니다.

## 2. 실제 데이터와 미확보 라벨

실제 사용 데이터:

- 수원 건축HUB 표제부·층별개요 9,482건
- 수원 단독·다가구 전월세 RTMS 34,095건
- 보증금 모델의 독립 test 계약 11,100건

실제 건축HUB 응답에는 schema mapping의 `totPkngCnt`가 없어서 주차대수는
결측으로 유지한다. RTMS mapping 11개 필드는 실제 응답에서 확인됐다.

공개 데이터로 확보하지 못한 정보:

- 기준일 현재 개별 호실 점유 여부
- 임대차 시작·종료 및 실제 전입일
- 확정일자와 담보권 설정일
- 신규 임차인보다 법적으로 선순위인지 여부
- 건물 단위 실제 선순위 보증금 총액

그래서 Model B는 Beta-Binomial 시나리오, Model D는 보수적 또는 사용자 지정
확률 시나리오이고 Model E는 비활성화돼 있다. 출력의 `model_mode`는
`scenario_only`다.

## 3. 모델 구조

```text
건축물대장 ──> Model A 등록 호실 수 ───────────────┐
                    │                              │
                    └─> Model B 다른 점유 호실 수 ─┤
RTMS 임대차 ─> Model C 호실 보증금 7개 분위수 ────┼─> 20,000회 Monte Carlo
실제 증빙 라벨 ─> Model D 선순위 확률(현재 비활성) ┤
건물 총액 라벨 ─> Model E 최종 보정(현재 비활성) ─┘
```

Model A는 공부상 `호수→가구수→세대수` 우선순위를 사용하고 정상값이 없을 때만
기존 실제 학습 count 모델을 호출한다.

Model B는 대상 호실을 제외한 `N-1`호를 대상으로
`q~Beta(alpha,beta)`, `M~Binomial(N-1,q)`를 사용한다. 이는 주변분포가
Beta-Binomial인 명시적 prior다.

Model C는 `log1p(보증금)`에 대한 LightGBM quantile 모델이다. 5%, 10%, 25%,
50%, 75%, 90%, 95% 분위수를 별도 학습하고, crossing 정렬과 validation
split-conformal 보정을 거친 뒤 piecewise-linear inverse CDF로 표본화한다.
월세는 입력 특성일 뿐 반환 보증금에 더하지 않는다.

동일 건물의 공통 효과는 `U~Normal(0,sigma²)`로 반영하며 sigma 0, 0.12,
0.25 결과를 함께 제공한다.

## 4. 데이터 누수와 매칭

- 법정동·면적구간 집계는 계약월을 `shift(1)`한 뒤 rolling한다.
- 시간 test와 별도로 법정동 spatial holdout을 둔다.
- 공개 일부 지번은 특정 건물 계약으로 확정 연결하지 않는다.
- 실제 라벨을 받으면 동일 사건·건물·소유 관련 그룹이 분할을 넘지 않게 한다.
- 증빙이 없는 행에는 `senior_to_target` 값을 만들지 않는다.

## 5. 실행

```powershell
py -m src.cli audit-data-sources
py -m src.cli train-all-senior `
  --artifact models/senior_deposit/senior_deposit_actual.joblib
py -m src.cli evaluate `
  --artifact models/senior_deposit/senior_deposit_actual.joblib
py -m src.cli infer `
  --artifact models/senior_deposit/senior_deposit_actual.joblib `
  --building-id HUB-8afd698677db742af7bb `
  --reference-date 2026-07-28 `
  --samples 20000 `
  --scenario conservative
```

향후 익명화된 증빙 라벨 검증:

```powershell
py -m src.cli import-labels --path data/labels/incoming
py -m src.cli train-occupancy
py -m src.cli train-seniority
py -m src.cli train-calibrator
```

라벨이 부족하면 세 학습 명령은 성공한 모델을 꾸미지 않고 `trained=false`와
이유를 출력한다.

## 6. API

`POST /api/properties/senior-deposit`

```json
{
  "session_id": "SESSION",
  "building_id": "HUB-8afd698677db742af7bb",
  "reference_date": "2026-07-28",
  "samples": 20000,
  "scenario": "conservative",
  "occupancy_scenario": "baseline"
}
```

결과는 `decision_run_id`와 함께 입력·모델 버전·fallback·원천을 감사 저장한다.

### 매물 상세 리포트 통합

`POST /api/properties/report`를 호출하면 임대차 매물에 대해 같은 모델이
`contract_safety.senior_deposit`에 자동 포함된다. 별도 API를 연속 호출할
필요가 없다.

통합 경로는 다음 순서를 강제한다.

1. 거래유형이 전세 또는 월세인지 확인한다.
2. 매물의 도로명주소를 괄호·공백만 정규화한다.
3. 건축HUB `buildings.csv`에서 정규화 주소가 완전히 같은 행만 선택한다.
4. 동일 주소에 여러 동이 있으면 다가구 용도, 공부상 호수, 주거면적이 확인된
   주거용 행을 우선한다.
5. 기준일 이전 RTMS 법정동 분포만 결합해 대상 호실을 제외한 기존 보증금
   P10·P50·P90·보수적 P95를 계산한다.
6. 결과와 SQL·모델 버전·건축물 매칭 근거를 같은 `decision_run_id`에 저장한다.

정확 주소가 없거나 동일 주소에 주거용 건축물대장 행이 없으면 유사주소로
억지 매칭하지 않고 `status=no_exact_building_match`를 반환한다. UI는 이때
빈 숫자나 0원을 표시하지 않고 공식 자료 확인 안내를 보여준다.

화면은 `existing_deposit_conservative_p95_won`과
`target_deposit_won`을 서로 더하지 않고 별도 항목으로 표시한다. 전자는 기존
임차인 보증금의 보수적 통계 추정이고 후자는 선택 매물에 공개된 사용자의 계약
보증금이므로 근거와 의미가 다르다. 이 통계 추정은 HF/HUG 보증사고 확률의
입력값을 덮어쓰지 않으며 `risk_score_changed=false`로 추적한다.

## 7. 해석 제한

대표 안전지표는 `conservative_upper_deposit.p90` 또는 p95다. Low 품질에서는
중앙값보다 이 상한을 우선 표시한다. 실제 계약 전에는 임대인·중개사에게 기존
임차보증금 현황, 전입세대확인 자료, 확정일자 관련 공식자료를 요청해야 한다.
