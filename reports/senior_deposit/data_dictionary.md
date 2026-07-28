# 선순위 임차보증금 데이터 사전

모델 내부 원천 금액 단위는 만원이고 API 출력은 원 단위 정수다.

## 입력 건물

| 필드 | 의미 |
|---|---|
| `building_id` | 건축HUB 관리키의 비가역 해시 |
| `unit_count` | 공부상 호수, 1순위 |
| `family_count` | 공부상 가구수, 2순위 |
| `household_count` | 공부상 세대수, 3순위 |
| `residential_floor_area` | 층별 용도에서 집계한 주거 연면적 |
| `local_market_count` | 법정동 임대차·매매 표본 수 |

호수·가구수·세대수를 더하지 않는다.

## 임대차

| 필드 | 의미·제약 |
|---|---|
| `deposit` | 신고된 반환 보증금 |
| `monthly_rent` | 보증금 모델 입력; 보증금에 자본화해 더하지 않음 |
| `partial_lot_number` | exact 건물 식별자로 사용 금지 |
| `contract_year_month` | 과거 집계와 시간 분할 기준 |
| `contract_type` | 전세·보증부월세 분포 구분 |

`src/senior_deposit/matching.py`는 후보 유사도를 계산할 수 있지만 마스킹된
부분지번은 면적·연식이 모두 비슷해도 `usable_as_building_label=false`로
강제한다. 현재 Model C는 이 후보를 건물 직접 라벨로 쓰지 않는다.

## 출력

| 필드 | 정의 |
|---|---|
| `estimated_total_deposit` | 현재 다른 점유 호실 보증금 추정 총합 |
| `estimated_senior_deposit` | 선순위 확률을 적용한 기준 추정 |
| `conservative_upper_deposit` | 현재 다른 점유 호실 전원을 선순위로 가정 |
| `model_mode` | trained/partially_trained/scenario_only |
| `data_quality` | high/medium/low |

## 실제 라벨

`src.senior_deposit.schemas.SENIOR_LABEL_FIELDS` 전체를 요구한다. 이름·주민번호·
전화·이메일 열은 validator가 거부한다. `senior_to_target`은 증빙 출처가 있는
행에만 허용한다.
