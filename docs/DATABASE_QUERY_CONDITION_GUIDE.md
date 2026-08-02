# 부동산·금융 DB 검색 조건 및 WHERE 절 가이드

작성 기준일: 2026-07-17  
실제 DB: `data/generated/jeonse_helper.db`

## 1. 요약

이 프로젝트에서 말하는 두 DB는 현재 별도 파일 두 개가 아니라 하나의 SQLite 파일 안에
있는 두 개의 논리 테이블이다.

| 논리 DB | 실제 테이블 | 현재 행 수 | 컬럼 수 | 용도 |
|---|---:|---:|---:|---|
| 부동산 매물 DB | `properties` | 20,000 | 211 | 전국 합성 매매·전세·월세 매물 검색 |
| 금융서비스 DB | `finance_programs` | 6 | 46 | 청년 주거·금융 정책 검색 |

검색 조건의 지원 수준은 세 단계로 구분해야 한다.

| 표시 | 지원 수준 | 의미 |
|---|---|---|
| **A** | 1급 조건 | Planner 슬롯, LLM SQL, 필수 조건 검증, 결정론 SQL 폴백까지 모두 지원 |
| **B** | 확장 조건 | 실제 스키마를 보고 LLM Text-to-SQL이 `WHERE`에 넣을 수 있지만, LLM 실패 시 폴백 SQL은 보존하지 않을 수 있음 |
| **C** | 후처리 조건 | SQL이 아니라 지도·안전·편의 도구 또는 Python 계산/ATOM 순위화 단계에서 반영 |

핵심적으로, 실제 컬럼은 `properties` 211개와 `finance_programs` 46개 모두 LLM SQL의
읽기 조건으로 사용할 수 있다. 다만 현재 **안정적으로 끝까지 보존되는 A급 자연어 조건은
아래 표에 명시된 조건**이다.

---

## 2. 전체 검색 흐름

```text
사용자 자연어
  → Planner가 intent / slots / qa_args 추출
  → 현재 SQLite 스키마를 LLM에 제공
  → LLM Text-to-SQL이 SELECT + WHERE 생성
  → 읽기 전용·테이블·필수 조건·필수 컬럼 검증
  → DB 실행
  → 실패 시 검증된 결정론 SQL로 폴백
  → ATOM/목표 순위화
  → LLM 최종 답변 합성
```

두 테이블을 직접 `JOIN`해 한 번에 검색하는 구조는 아니다. 복합 목표에서는 다음처럼
순차적으로 연결한다.

```text
finance_programs 검색
  → 가능한 자금/지원 역할 판별
  → 최대 전세보증금 계산
  → 계산값을 properties의 deposit_manwon 상한으로 사용
  → 매물 검색 및 최고가순 선별
```

---

## 3. 부동산 매물 DB: `properties`

### 3.1 현재 A급 WHERE 조건

다음 조건은 자연어 파싱, 슬롯 저장, LLM SQL 누락 검증, 결정론 폴백 SQL까지 연결되어
있다.

| 사용자 표현 예시 | 슬롯 | WHERE 대상 | 연산 |
|---|---|---|---|
| `전세`, `월세`, `매매` | `lease_type`, `transaction_type` | `lease_type`, `transaction_type` | `=` |
| `아파트`, `오피스텔`, `다가구`, `다세대`, `연립`, `단독` | `property_type` | `property_type`, `house_type` | `=` 또는 `LIKE` |
| `서울`, `대전` 등 정규화된 시도 | `region_sido` | `sido` | `=` |
| `유성구`, `관악구` | `region_gugun` | `gugun` | `=` 또는 `IN (...)` |
| `전세보증금 8천 이하` | `max_deposit_manwon` | `deposit_manwon` | `<=` |
| `월세 60만원 이하` | `max_monthly_rent_manwon` | `monthly_rent_manwon` | `<=` |
| `매매가 5억 이하` | `max_sale_price_manwon` | `sale_price_manwon`, `asking_price_manwon` | `COALESCE(...) <=` |
| `관리비 10만원 이하` | `max_maintenance_manwon` | `maintenance_fee_manwon` | `<=` |
| `20평 이상`, `40㎡ 이상` | `min_area_m2` | `area_m2` | `>=` |
| `신축`, `10년 이내` | `max_building_age` | `building_age_years` | `<=` |
| `안전한 전세`, `위험도 0.2 이하` | `max_fraud_score` | `fraud_score` | `<=` |

주택유형은 현재 `property_type` 값이 대부분 `주거용 건축물`이고 실제 세부 유형은
`house_type`에 들어 있다. 따라서 폴백 SQL은 두 컬럼을 함께 검사한다.

```sql
AND (
  property_type = '아파트'
  OR house_type = '아파트'
  OR house_type LIKE '%아파트%'
)
```

현재 실제 `house_type` 값은 다음과 같다.

- 다가구주택
- 다세대주택
- 단독주택
- 아파트
- 연립주택
- 오피스텔

현재 `sido`는 서울·경기뿐 아니라 17개 시도를 모두 포함한다.

```text
서울, 경기, 인천, 부산, 대구, 대전, 광주, 울산, 세종,
강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주
```

SQL과 실제 LLM Planner는 이 17개 지역을 사용할 수 있다. 다만 오프라인 `MockLLM`의
일반 매물 규칙 파서는 현재 서울·경기·인천·부산·대구·대전·광주·울산·세종 9개를
문장에서 직접 추출한다. 17개 전체는 사용자 프로필의 `preferred_sido` 정규화와
`goal_financed_jeonse` 복합 목표 파서에서 지원한다. 따라서 강원·충북·충남·전북·전남·
경북·경남·제주를 일반 매물 문장에서 안정적으로 쓰려면 실제 API LLM 모드를 사용하거나
프로필 선호지역으로 넣어야 한다.

다음은 자연어 Planner 슬롯이라기보다 코드가 내부적으로 넣는 검색 제어값이다.

| 내부 제어값 | SQL 효과 | 현재 사용처 |
|---|---|---|
| `min_priority_rank_first` | `my_priority_rank = 1` | 도구 직접 호출 시 사용 가능 |
| `order_by` | 허용 컬럼 `ASC`/`DESC` | 안전 우선·전세 최대화 등 목표 파이프라인 |
| `limit` | `LIMIT 1~500` | 일반 추천 500, 화면 출력은 상위 일부 |

### 3.2 대표 폴백 SQL

입력:

```text
대전 유성구에서 보증금 8천만원 이하, 관리비 10만원 이하의 신축 아파트 전세
```

결정론 폴백은 개념적으로 다음 SQL을 만든다.

```sql
SELECT property_id, is_synthetic, synthetic_notice,
       sido, gugun, dong, lat, lng,
       transaction_type, lease_type, property_type, house_type,
       asking_price_manwon, sale_price_manwon,
       deposit_manwon, monthly_rent_manwon, maintenance_fee_manwon,
       market_price_manwon, area_m2, building_age_years,
       my_priority_rank, building_total_units, fraud_score
FROM properties
WHERE 1=1
  AND lease_type = '전세'
  AND transaction_type = '전세'
  AND (property_type = '아파트'
       OR house_type = '아파트'
       OR house_type LIKE '%아파트%')
  AND sido = '대전'
  AND gugun = '유성구'
  AND deposit_manwon <= 8000
  AND maintenance_fee_manwon <= 10
  AND building_age_years <= 5
ORDER BY deposit_manwon ASC
LIMIT 500;
```

### 3.3 B급 확장 WHERE 조건

아래 조건은 실제 DB에 저장되어 있으므로 LLM Text-to-SQL이 사용할 수 있다. 하지만
현재 Planner의 정식 슬롯과 결정론 폴백 쿼리에는 모두 연결되어 있지 않다. 따라서
LLM SQL이 실패하면 해당 조건이 빠질 수 있다.

#### 매물 상태와 데이터 출처

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 매물 상태 | `listing_status`, `listing_created_at`, `listing_updated_at` | `listing_status = '거래가능(합성)'` |
| 합성 여부 | `is_synthetic`, `source_type`, `generation_method` | `is_synthetic = 1` |
| 생성 품질 | `privacy_distance_score`, `price_model_holdout_r2`, `price_model_holdout_mdape_pct` | `price_model_holdout_r2 >= 0.7` |
| 기반 데이터 | `source_dataset`, `source_transaction_type`, `reference_rent_deal_ym`, `reference_trade_deal_ym` | `reference_trade_deal_ym >= 202501` |

#### 주소와 토지

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 동·주소 | `dong`, `legal_dong_code`, `road_address`, `jibun_address`, `road_name` | `dong LIKE '%봉명동%'` |
| 위치 범위 | `lat`, `lng` | `lat BETWEEN ... AND ...` |
| 토지 면적 | `land_area_m2`, `land_share_m2` | `land_area_m2 >= 100` |
| 용도·지목 | `land_category`, `zoning`, `land_use_status` | `zoning LIKE '%주거%'` |
| 도로·허가구역 | `road_access`, `land_transaction_permit_zone` | `land_transaction_permit_zone = 0` |

#### 가격·계약 조건

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 가격 협의 | `price_negotiable` | `price_negotiable = 1` |
| 전월세전환율 | `rent_conversion_rate_pct` | `rent_conversion_rate_pct <= 5` |
| 관리비 항목 | `maintenance_fee_items`, `maintenance_fee_other` | `maintenance_fee_items LIKE '%수도%'` |
| 입주 가능일 | `available_from_date`, `move_in_negotiable`, `occupancy_status` | `available_from_date <= '2026-08-01'` |
| 계약 형태 | `contract_type`, `contract_term`, `use_renewal_right` | `contract_term = '2년'` |
| 중개·일회비용 | `onetime_fee_manwon`, `broker_fee_manwon`, `actual_expense_manwon` | `onetime_fee_manwon <= 100` |

#### 건물·면적·방 구조

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 건물명·용도 | `building_name`, `building_use`, `building_structure` | `building_name LIKE '%래미안%'` |
| 사용승인·연식 | `approval_date`, `build_year`, `building_age_years` | `build_year >= 2020` |
| 층 | `current_floor`, `total_floors`, `basement_floors` | `current_floor >= 3` |
| 면적 | `exclusive_area_m2`, `supply_area_m2`, `contract_area_m2` | `exclusive_area_m2 >= 59` |
| 방·욕실 | `room_count`, `bathroom_count`, `unit_type` | `room_count >= 2` |
| 방향 | `direction`, `direction_basis` | `direction = '남향'` |
| 세대·동 수 | `building_total_units`, `building_total_households`, `total_complex_buildings` | `building_total_households >= 300` |

#### 옵션·생활 조건

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 주차 | `parking_total`, `parking_per_household`, `parking_method` | `parking_per_household >= 1` |
| 엘리베이터 | `elevator_count` | `elevator_count >= 1` |
| 냉난방 | `heating_method`, `heating_fuel`, `cooling_facility`, `aircon_count` | `aircon_count >= 1` |
| 가전·가구 | `built_in_appliances`, `furnished` | `furnished = 1` |
| 반려동물 | `pet_allowed` | `pet_allowed = '가능'` |
| 대출 가능 표시 | `loan_available` | `loan_available = '가능'` |
| 특화공간 | `balcony_expansion`, `duplex`, `terrace`, `yard`, `rooftop_access` | `terrace = 1` |

#### 물리 상태·하자

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 균열·누수 | `wall_crack`, `water_leak` | `wall_crack = 0 AND water_leak = 0` |
| 마감 상태 | `wallpaper_condition`, `floor_condition`, `renovation_status` | `renovation_status = '완료'` |
| 소음·진동 | `noise_level`, `vibration_level` | `noise_level = '낮음'` |
| 채광 | `sunlight_level` | `sunlight_level = '좋음'` |
| 불법·대장 불일치 | `illegal_building`, `ledger_discrepancy` | `illegal_building = 0 AND ledger_discrepancy = 0` |

#### 권리관계·전세 안전

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 신탁·압류 | `trust_registration`, `seizure_or_provisional_seizure` | `trust_registration = 0` |
| 임차권 등기 | `leasehold_registration`, `tenant_right_registration` | `tenant_right_registration = 0` |
| 세금 체납 | `tax_arrears_checked`, `tax_arrears_present` | `tax_arrears_checked = 1 AND tax_arrears_present = 0` |
| 보증 가입 | `rental_deposit_guarantee_joined`, `deposit_return_guarantee_provider` | `rental_deposit_guarantee_joined = 1` |
| 보증 가입 가능성 | `guarantee_eligible`, `guarantee_ineligible_reason` | `guarantee_eligible = 1` |
| 근저당 | `mortgage_max_claim_manwon`, `senior_mortgage_manwon`, `mortgage_ltv_pct` | `mortgage_ltv_pct <= 40` |
| 선순위 권리 | `my_priority_rank`, `senior_tenant_count`, `senior_deposit_sum_manwon`, `senior_rights_total_manwon` | `my_priority_rank = 1` |
| 전세가율 | `jeonse_ratio_pct` | `jeonse_ratio_pct <= 70` |
| 위험 점수 | `fraud_score`, `fraud_label` | `fraud_score <= 0.2` |

#### 교통·주변환경

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 지하철·버스 | `subway_walk_minutes`, `bus_stop_walk_minutes` | `subway_walk_minutes <= 10` |
| 학교·마트·병원·공원 | `school_walk_minutes`, `mart_walk_minutes`, `hospital_walk_minutes`, `park_walk_minutes` | `hospital_walk_minutes <= 15` |
| 환경 위험 | `noise_source`, `odor_source`, `flood_risk_level`, `nonpreferred_facility` | `flood_risk_level = '낮음'` |

#### 중개·광고·증빙

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 중개사무소 | `broker_office_name`, `broker_registration_no`, `broker_office_address` | `broker_registration_no IS NOT NULL` |
| 중개 보증 | `broker_guarantee_type`, `broker_guarantee_amount_manwon`, `broker_guarantee_period` | `broker_guarantee_amount_manwon >= 10000` |
| 광고 자료 | `photo_count`, `video_present`, `virtual_tour_present` | `photo_count >= 10` |
| 방문 가능 | `viewing_available`, `viewing_method` | `viewing_available = 1` |
| 확인 자료 | `evidence_registry`, `evidence_building_ledger`, `evidence_land_ledger` 등 | `evidence_registry = 1` |
| 설명 완료 | `explanation_completed`, `explanation_notes` | `explanation_completed = 1` |

### 3.4 C급 조건: SQL 외부에서 반영하는 조건

| 조건 | 처리 위치 | 방식 |
|---|---|---|
| 직장·학교까지 통근시간 | 지도 도구 | 좌표별 이동시간 계산 후 ATOM 평가 |
| 주변 CCTV·경찰·소방 등 치안 | 안전 도구 | 반경 데이터로 안전점수 계산 후 평가 |
| 편의점·병원·마트 등 생활편의 | 편의 도구 | 반경 데이터로 편의점수 계산 후 평가 |
| 사용자 소득·자산 기반 적정예산 | affordability 계산 | 명시 예산이 없을 때 ATOM 예산 조건으로 주입 |
| 금융 활용 최대 전세예산 | 금융→예산 계산기 | 계산 결과를 `deposit_manwon <= 값`으로 다시 SQL에 주입 |
| 조건 양보 그룹 | ATOM | 완전 만족/1개 양보/2개 양보로 분류 |

통근, 안전, 생활편의는 별도 원천 데이터나 API를 사용하므로 기본 `properties WHERE`에
직접 들어가지 않는다. 단, `subway_walk_minutes`처럼 이미 매물 행에 저장된 거리 컬럼은
B급 LLM SQL 조건으로 별도 사용할 수 있다.

### 3.5 부동산 정렬 조건

결정론 폴백에서 안전하게 지정 가능한 주요 정렬 컬럼은 다음과 같다.

```text
deposit_manwon, monthly_rent_manwon, maintenance_fee_manwon,
asking_price_manwon, sale_price_manwon, market_price_manwon,
area_m2, building_age_years, my_priority_rank,
building_total_units, fraud_score, sido, gugun 등
```

대표 목표 정렬:

```sql
-- 예산 내 가장 비싼 전세
ORDER BY deposit_manwon DESC

-- 전세사기 추정 위험도가 낮은 순
ORDER BY fraud_score ASC

-- 넓은 매물 우선
ORDER BY area_m2 DESC
```

복합 전세 최대화 목표에서는 SQL 결과를 받은 뒤에도 Python에서
`deposit_manwon DESC`, 동일 금액이면 `fraud_score ASC`를 다시 검증한다.

---

## 4. 금융서비스 DB: `finance_programs`

### 4.1 현재 A급 WHERE 조건

| 사용자 표현 예시 | 인자/모드 | WHERE 대상 | 연산 |
|---|---|---|---|
| `대출만`, `지원금만`, `청약` | `product_kind` | `product_kind`, `category` | `LIKE '%값%'` |
| 정확한 정책 카테고리 | `category` | `category` | `=` |
| `대전에서 받을 수 있는 정책` | `region` | `region_scope`, `eligible_regions` | 전국 포함 `OR` 검색 |
| `금리 2% 미만` | `max_rate_pct` | `rate_pct` | 배타적 `< 2` |
| `내가 받을 수 있는 상품` | `finance_mode=eligibility` | `income_limit_manwon`, `age_min`, `age_max` | 개인 프로필 비교 |
| `금융지원책 뭐가 있지` | `finance_mode=catalog` | 개인 소득·나이 조건 없음 | 전체 탐색 |
| 결과 개수 | `limit` | SQL 결과 | 일반 10건, 복합 목표 50건 |

#### 상품 종류

`product_kind`는 하나의 정책에 여러 값이 쉼표로 들어갈 수 있기 때문에 `=`가 아니라
`LIKE`를 사용한다.

```sql
AND (product_kind LIKE '%대출%' OR category LIKE '%대출%')
```

현재 값은 `주거공급`, `지원`, `청약,대출`이다.

#### 지역

특정 지역 요청은 지역 정책뿐 아니라 전국 정책도 포함한다.

```sql
AND (
  region_scope = '전국'
  OR region_scope = :region
  OR eligible_regions LIKE '%' || :region || '%'
)
```

#### 금리

사용자가 `2% 미만`이라고 하면 경계값 2.0은 제외한다.

```sql
AND rate_pct < 2.0
```

`rate_min_pct`, `rate_max_pct`도 DB에는 있지만 현재 사용자 상한 필터의 기준은
대표금리 `rate_pct`다. 이는 LLM SQL과 결정론 폴백이 동일하게 작동하게 하기 위한
현재 프로젝트 규칙이다.

#### 개인 적격성

사용자 월소득은 12배하여 연소득 상한과 비교한다. 정책의 제한값이 `NULL`이면 제한을
알 수 없거나 제한이 없는 것으로 보고 1차 후보에 포함한다.

```sql
AND (income_limit_manwon IS NULL
     OR income_limit_manwon >= :monthly_income_manwon * 12)
AND (age_min IS NULL OR age_min <= :user_age)
AND (age_max IS NULL OR age_max >= :user_age)
```

이는 최종 자격 판정이 아니라 DB 컬럼만을 이용한 1차 후보 검색이다.

### 4.2 금융 검색 모드

| 모드 | 적용 조건 | 예시 |
|---|---|---|
| `catalog` | 나이·소득 필터를 적용하지 않음 | `금융지원책 뭐가 있지?` |
| `eligibility` | 나이·연소득·요청지역을 적용 | `내 조건으로 받을 수 있는 대출은?` |
| 복합 전세 목표 | `eligibility` 검색 후 정책 역할 분류와 예산 계산 | `금융상품으로 최대한 비싼 전세 추천해줘` |

복합 목표는 검색된 정책을 다음과 같이 후처리한다.

| 분류 | 예시 | 전세예산 반영 |
|---|---|---|
| 직접 전세자금 | 전세자금대출, 임차보증금 대출 | 금리·한도가 확인되면 단일 최선 상품만 반영 |
| 비용 절감 | 반환보증 보증료 지원, 이자 지원 | 비용 절감 근거로 표시하지만 보증금에 합산하지 않음 |
| 무관 정책 | 청약 후 주택구입대출, 기숙사, 일반 공공임대 | 목표 예산에서 제외 |

### 4.3 B급 확장 WHERE 조건

| 조건 분야 | 주요 컬럼 | 예시 WHERE |
|---|---|---|
| 최대 지원·대출액 | `max_amount_manwon` | `max_amount_manwon >= 10000` |
| 금리 범위 | `rate_min_pct`, `rate_max_pct` | `rate_max_pct <= 3.5` |
| 현재 신청 가능 | `always_open`, `application_start_date`, `application_end_date`, `application_status` | `always_open = 1` |
| 혼인 조건 | `marital_status` | `marital_status LIKE '%미혼%'` |
| 학력 조건 | `education` | `education LIKE '%대학생%'` |
| 취업 조건 | `employment_status` | `employment_status LIKE '%재직%'` |
| 특화 대상 | `specialization`, `additional_eligibility` | `specialization LIKE '%청년%'` |
| 참여 제한 | `participation_restrictions` | 텍스트 검색 |
| 신청 방법 | `application_procedure`, `application_site`, `required_documents` | `application_site IS NOT NULL` |
| 운영 기관 | `supervising_organization`, `operating_organization` | `operating_organization LIKE '%주택%'` |
| 정책 설명 | `name`, `target`, `support_content`, `desc`, `tags` | 여러 컬럼 `LIKE` 검색 |
| 최근 갱신 | `last_modified_date`, `imported_at` | `last_modified_date >= '2026-01-01'` |
| 출처 | `source_type`, `source_url`, `reference_url_1`, `reference_url_2` | `source_type = 'attachment'` |

현재 사용자 프로필에는 나이·소득·자산·생활비·소득분위·선호지역만 있다. 따라서 혼인,
학력, 재직상태 같은 컬럼은 사용자가 대화에서 직접 알려주더라도 B급 LLM SQL로만
처리될 수 있으며, 아직 결정론 폴백에는 보존되지 않는다.

표준 금융 Q&A의 오프라인 규칙 Planner는 지역을 `plan.slots`로 별도 추출하지 않으므로
지역 금융검색은 실제 API LLM 경로에서 더 안정적이다. 금융→전세 복합 목표는 프로필의
선호지역도 사용하도록 별도 구현되어 있다.

### 4.4 금융상품 정렬

기본 정렬은 금리가 있는 상품을 먼저 배치하고 금리 오름차순, 최근 갱신일 내림차순이다.

```sql
ORDER BY
  CASE WHEN rate_pct IS NULL THEN 1 ELSE 0 END,
  rate_pct ASC,
  last_modified_date DESC
```

---

## 5. 조합 가능한 대표 질문

### 5.1 부동산 조건 조합

```text
대전 유성구에서 전세보증금 8천 이하, 관리비 10만원 이하,
전용면적 40㎡ 이상인 신축 아파트를 찾아줘
```

안정 지원되는 WHERE 핵심:

```sql
WHERE lease_type = '전세'
  AND sido = '대전'
  AND gugun = '유성구'
  AND house_type LIKE '%아파트%'
  AND deposit_manwon <= 8000
  AND maintenance_fee_manwon <= 10
  AND area_m2 >= 40
  AND building_age_years <= 5
```

### 5.2 금융 조건 조합

```text
내 나이와 소득으로 받을 수 있는 전국 대출 중 금리 3% 미만만 보여줘
```

안정 지원되는 WHERE 핵심:

```sql
WHERE product_kind LIKE '%대출%'
  AND rate_pct < 3
  AND region_scope = '전국'
  AND (income_limit_manwon IS NULL OR income_limit_manwon >= :월소득 * 12)
  AND (age_min IS NULL OR age_min <= :나이)
  AND (age_max IS NULL OR age_max >= :나이)
```

### 5.3 두 DB를 연결한 복합 목표

```text
금융상품을 활용해서 감당 가능한 가장 비싼 전세를 추천해줘
```

1. `finance_programs`를 `eligibility` 모드로 검색한다.
2. 직접 전세자금 상품과 비용절감 정책을 분리한다.
3. 자기자금 적정예산과 단일 직접 대출의 유효 한도로 최대 보증금을 계산한다.
4. 계산값을 다음 부동산 조건에 넣는다.

```sql
WHERE lease_type = '전세'
  AND transaction_type = '전세'
  AND deposit_manwon <= :estimated_max_deposit_manwon
ORDER BY deposit_manwon DESC
LIMIT 500
```

5. 같은 보증금이면 `fraud_score`가 낮은 후보를 먼저 추천한다.

---

## 6. 단위와 NULL 처리

| 항목 | 단위/처리 |
|---|---|
| `*_manwon` | 만원. `8000`은 8천만원 |
| `*_pct` | 퍼센트 숫자. `2.4`는 2.4% |
| 면적 | `*_m2`는 제곱미터 |
| 위험도 | `fraud_score`는 0~1, 낮을수록 추정 위험이 낮음 |
| Boolean | SQLite에서 일반적으로 `1=true`, `0=false` |
| 날짜 | ISO 형식 텍스트는 문자열 비교 가능. 원본 형식이 다르면 정규화 필요 |
| `NULL` 금리 | 금리순에서 뒤로 배치, 금리 상한 검색에는 포함되지 않음 |
| `NULL` 소득·나이 한도 | 1차 적격 후보에는 포함 |
| `NULL` 위험도 | 폴백 위험도 필터에서 포함될 수 있으므로 최종 화면에서 `N/A` 확인 필요 |

현재 실제 값 범위의 예시는 다음과 같다.

| 테이블 | 컬럼 | 현재 범위 |
|---|---|---|
| properties | `deposit_manwon` | 0 ~ 251,440만원 |
| properties | `monthly_rent_manwon` | 0 ~ 1,677만원 |
| properties | `sale_price_manwon` | 0 ~ 855,600만원 |
| properties | `area_m2` | 10 ~ 1,500㎡ |
| properties | `building_age_years` | 0 ~ 126년 |
| properties | `fraud_score` | 약 0.127 ~ 0.993 |
| finance_programs | `rate_pct` | 현재 비NULL 값 2.4% |
| finance_programs | `max_amount_manwon` | 0 ~ 40,000만원 |
| finance_programs | `income_limit_manwon` | 5,000 ~ 7,500만원/년 |

---

## 7. SQL 안전 및 검증 규칙

LLM이 SQL을 자유롭게 생성하더라도 다음 제한을 통과해야 실행된다.

1. `SELECT` 한 문장만 허용한다.
2. `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `PRAGMA`, `ATTACH` 등은 금지한다.
3. 다중 구문, SQL 주석, 허용되지 않은 테이블을 차단한다.
4. 운영 연결은 SQLite 읽기 전용 및 `query_only` 모드다.
5. 허용된 실제 테이블의 실제 컬럼만 읽을 수 있다.
6. 허용 함수는 `COALESCE`, `CASE`에 필요한 기본 연산과 제한된 안전 함수뿐이다.
7. 매물 검색 결과는 최대 500행이다.
8. Planner가 확정한 A급 조건이 SQL에 빠지면 실행하지 않고 수정 재시도한다.
9. 추천과 답변에 필요한 필수 SELECT 컬럼이 빠지면 수정 재시도한다.
10. 두 번 실패하면 검증된 슬롯 기반 결정론 SQL로 폴백하고 trace에 원인을 남긴다.

RAG 디버그 trace에서 확인할 핵심 항목:

```text
planner.slots / planner.qa_args
tools[].input_filters
tools[].attempts
tools[].final_sql
tools[].parameters
tools[].validation
tools[].row_count
tools[].fallback / fallback_reason
workflow.steps
```

---

## 8. 현재 한계와 확장 시 수정할 곳

### 8.1 B급 조건을 A급 조건으로 승격하려면

예를 들어 `반려동물 가능`, `방 2개 이상`, `보증보험 가입 가능`을 폴백까지 안정적으로
지원하려면 다음을 모두 수정해야 한다.

1. `src/agent/prompts.py`의 `PLAN_JSON_SCHEMA`에 슬롯 추가
2. `src/agent/planner.py`의 결정론 자연어 추출 추가
3. `src/agent/harness.py`의 `slot_to_db` 매핑 추가
4. `src/tools/property_db_tool.py`의 `build_query()` WHERE 생성 추가
5. `src/agent/text2sql.py`의 `_assert_slot_coverage()` 누락 검증 추가
6. 필요하면 SELECT 필수 컬럼과 `_format_rec()` 응답 필드 추가
7. LLM SQL, 폴백 SQL, 0건, NULL 의미를 각각 테스트

### 8.2 현재 중요한 제한

- 부동산 20,000건은 연구용 합성 매물이며 현재 실제 중개 플랫폼 실매물이 아니다.
- 금융정책은 현재 6건뿐이므로 실제 가능한 전세대출 전체를 대표하지 않는다.
- 금융 적격성 검색은 DB에 구조화된 나이·소득 조건만 자동 적용한다.
- 신청기간, 혼인, 재직, 무주택, 세대주 등 복잡한 조건은 텍스트 컬럼 확인과 최종 심사가 필요하다.
- LLM 전용 B급 조건은 SQL 생성 실패 후 폴백에서 사라질 수 있다.
- 결과가 부족하면 표준 매물 추천 파이프라인이 지역 또는 일부 수치 선필터를 완화할 수 있으며,
  완화 내용은 `agent_trace.fallbacks`에 기록된다.

---

## 9. 관련 구현 파일

| 내용 | 파일 |
|---|---|
| 자연어 intent·slot 스키마 | `src/agent/prompts.py` |
| 결정론 자연어 파서 | `src/agent/planner.py` |
| 다단계 오케스트레이션 | `src/agent/harness.py` |
| LLM SQL·검증·폴백 | `src/agent/text2sql.py` |
| 부동산 SQL 빌더·읽기 전용 실행 | `src/tools/property_db_tool.py` |
| 금융 결정론 검색 | `src/tools/finance_tool.py` |
| 적정 주거예산 계산 | `src/preference/affordability.py` |
| DB 생성·적재 | `src/db/build_db.py` |
| 실제 SQLite | `data/generated/jeonse_helper.db` |

실제 스키마는 실행 시 `PRAGMA table_info(properties)`와
`PRAGMA table_info(finance_programs)`로 동적으로 읽어 LLM에 제공한다. 따라서 DB 컬럼이
변경되면 LLM이 보는 스키마는 즉시 바뀌지만, A급 슬롯과 폴백 지원은 위 코드들을 함께
수정해야 한다.
