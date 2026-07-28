# 다가구 전세가율 통합 입력 계약

## 목적

두 기존 모델을 재학습하지 않고 동일 건물·동일 시점의 확률분포로 결합한다.
내부 통합 화폐 단위는 `KRW/manwon`(만원)이다.

## 임차보증금 모델

필수 키:

- `building_id`, `reference_date`
- `estimated_total_deposit`: 선택 호실을 제외한 기존 보증금 총합
- `estimated_senior_deposit`: 신규 계약보다 선순위라고 가정한 금액
- `conservative_upper_deposit`: 기존 보증금 전부를 선순위로 보는 보수 상한
- `data_quality`, `model_mode`, `warnings`

원 출력 단위는 원이며 어댑터가 10,000으로 나눠 만원으로 변환한다.

## 건물가치 모델

필수 키:

- `building_id`, `reference_date`
- `estimated_property_value`: 건물 전체 시장가치의 P05/P10/P50/P90/P95
- `data_quality`

임대인의 총자산 분포는 건물 시장가치와 다른 개념이므로 전세가율 분모로
사용하지 않는다.

## 정합성 검증

- `building_id` 완전 일치
- 기준일 차이 31일 이하
- 통화 KRW와 어댑터 후 단위 `manwon` 일치
- 가격 기준은 `market_value` 또는 `transaction_price_estimate`
- 분위수 역전은 누적 최댓값으로 보정하고 경고 기록
- 건물가치 0 이하 표본은 즉시 실패

## 출력 정의

| 키 | 정의 |
|---|---|
| `all_deposit_ratio` | 기존 전체 보증금 / 건물 시장가치 |
| `senior_deposit_ratio` | 추정 선순위 보증금 / 건물 시장가치 |
| `post_contract_ratio` | (추정 선순위 보증금 + 내 보증금) / 건물 시장가치 |
| `conservative_post_contract_ratio` | (보수 선순위 상한 + 내 보증금) / 건물 시장가치 |

주 위험지표는 `post_contract_ratio`다. 근저당권은 이 비율에 합산하지 않고
등기 권리부담으로 별도 표시한다.
