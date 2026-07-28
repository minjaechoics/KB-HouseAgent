# 수원시 전세금/집주인 총자산 비율 데이터 사전

## 원칙

- 단위는 별도 표기가 없으면 만원과 제곱미터다.
- 단독·다가구 실거래의 일부 지번은 확정 건물 식별자로 사용하지 않는다.
- 가계금융복지조사 행은 개별 건물과 조인하지 않는다.
- `source_kind=synthetic_smoke_only`인 행과 artifact는 실제 추정에 사용할 수 없다.

## 건축물대장

| 필드 | 의미 | 검증 |
|---|---|---|
| `building_id` | 관리 PK의 비가역 해시 식별자 | 필수·유일 |
| `management_register_pk` | 건축HUB 관리 PK | 원천 보관, 외부 응답 제외 가능 |
| `sigungu_code` | 수원 4개 구 5자리 코드 | 41111/41113/41115/41117 |
| `legal_dong_code` | 시군구+법정동 코드 | 문자열 보존 |
| `land_area` | 대지면적 | 0 초과 |
| `total_floor_area` | 연면적 | 0 초과 |
| `residential_floor_area` | 주거용 연면적 | 연면적 이하 점검 |
| `unit_count` | 호수 | 1~100만 관측값 인정 |
| `family_count` | 가구수 | 1~100만 관측값 인정 |
| `household_count` | 세대수 | 1~100만 관측값 인정 |
| `registered_units_observed` | 우선순위로 선택한 공부상 호실 수 | unit→family→household |

## 임대차 계약

| 필드 | 의미 | 주의 |
|---|---|---|
| `deposit` | 실제 신고 보증금 | 월세의 전세가 환산액을 더하지 않음 |
| `monthly_rent` | 신고 월세 | 보증금 모델의 조건 변수 |
| `contract_type` | 전세/보증부월세/갱신 등 | 서로 다른 분포 허용 |
| `rental_area` | 계약 면적 | 면적구간 coverage 평가 |
| `partial_lot_number` | 공개된 일부 지번 | 건물 확정 연결 금지 |
| `legal_dong_*` | 과거 지역 집계 | 반드시 계약월 이전 자료만 사용 |

## 매매 계약

| 필드 | 의미 | 학습 라벨 사용 |
|---|---|---|
| `sale_price` | 신고 매매가 | exact/high만 |
| `match_confidence` | 건물 연결 신뢰도 | exact/high/medium/low/unmatched |
| `land_area` | 거래 대지면적 | 비교가격에도 사용 |
| `total_floor_area` | 거래 연면적 | 비교가격에도 사용 |

`medium`, `low`, `unmatched` 행은 건물가치 직접 라벨에서 제외하며 지역 비교가격
생성에만 사용할 수 있다.

## 가계금융복지조사

실제 2022~2025 공개용 가구마스터는
`configs/owner_asset_ratio/survey_schema_mapping.ahs_public_2022_2025.yaml`로
매핑한다. 다른 조사판을 사용할 때만 example을 복사해 해당 코드북에 맞춘다.
`PLACEHOLDER`가 하나라도 남으면 전처리가 중단된다.

| 표준 필드 | 2022~2025 공개용 변수 |
|---|---|
| `year` | `조사연도` |
| `total_assets` | `자산` |
| `financial_assets` | `자산_금융자산` |
| `real_estate_assets` | `자산_실물자산_부동산금액` |
| `rental_real_estate_assets` | `자산_실물자산_부동산_거주주택이외부동산금액` |
| `owner_occupied_home_assets` | `자산_실물자산_부동산_거주주택금액` |
| `rental_deposit_liability` | `부채_임대보증금` |
| `financial_debt` | `부채_금융부채` |
| `survey_weight` | `가중값` |
| `region` | `수도권여부` (`G1`을 수도권으로 해석) |

| 파생값 | 정의 |
|---|---|
| `K_other` | `(총자산-임대용부동산자산)/임대용부동산자산` |
| `R_survey` | `임대보증금부채/총자산` |
| `L_debt` | `금융부채/총자산` |

음수 `K_other`는 0으로 자르지 않고 매핑 또는 변수 정의 오류로 보고한다.
