# 첨부 청년 주거·금융 정책 DB

## 반영 결과

사용자가 첨부한 정책 상세 페이지 원문에서 정책번호가 같은 중복 항목을 제거해
`finance_programs`에 고유 정책 6건을 저장했다. 기존의 예시용 F001~F005 시드 5건은
더 이상 사용하지 않는다.

| 정책번호 | 지역 | 정책명 | 분류 |
|---|---|---|---|
| 20260616005400113238 | 전국 | 청년주택드림청약통장 | 청약·연계대출 |
| 20260527005400113223 | 전국 | 전세보증금반환보증 보증료 지원 | 보증료지원 |
| 20260504005400113081 | 충남 | 전세보증금반환보증 보증료 지원 | 보증료지원 |
| 20260311005400112113 | 세종 | 청년·신규 공무원의 주거안정 방안 마련 | 기숙사 |
| 20260123005400112079 | 전국 | 대학생 연합생활관(은행권,고양) | 기숙사 |
| 20250714005400111208 | 울산 | 상안지구 행복주택 건립 | 공공임대주택 |

## 파일과 재적재

- 보존 원문: `data/downloaded/finance_policies/source_youth_policies.txt`
- 정규화 CSV: `data/downloaded/finance_policies/youth_housing_policies.csv`
- 검색 DB: `data/generated/jeonse_helper.db`의 `finance_programs`
- 가져오기 코드: `scripts/import_finance_policies.py`

새 첨부 원문으로 교체할 때는 프로젝트 루트에서 다음을 실행한다.

```powershell
py -3 scripts/import_finance_policies.py "C:\path\to\pasted-text.txt"
```

`python -m src.db.build_db`와 `FinanceTool.refresh()`는 이후 정규화 CSV를 읽으므로
예전 예시 시드로 돌아가지 않는다.

## 스키마

기존 Agent/GUI 호환 컬럼 9개(`program_id`, `name`, `category`, `target`,
`max_amount_manwon`, `rate_pct`, `income_limit_manwon`, `source_url`, `desc`)를 유지하고,
정책 원문의 상세 필드를 추가해 총 46개 컬럼으로 구성했다.

주요 상세 필드는 다음과 같다.

- 분류: `policy_area`, `product_kind`, `region_scope`, `eligible_regions`
- 혜택: `support_content`, `max_amount_manwon`, `rate_min_pct`, `rate_max_pct`
- 신청기간: `application_period`, 시작·종료일, `always_open`, `application_status`
- 자격: 연령 원문과 최소·최대 나이, 혼인·소득·학력·취업·특화분야·추가조건·제한조건
- 신청: 절차, 발표, 신청 사이트, 제출서류, 주관·운영기관, 참고 URL
- 출처관리: 태그, 최종 수정일, 출처 유형, 가져온 시각

`income_limit_manwon`은 연소득 상한의 만원 단위다. Agent의 사용자 프로필은 월소득
만원 단위이므로 적격성 검색에서 12배하여 비교한다. 지원액은 금액 문맥을 구분해
보증료 지원은 40만원, 청년주택드림 연계대출은 최대 4억원(40,000만원)으로 저장하고,
임차보증금 3억원이나 대상주택 가격 6억원을 지원한도로 잘못 사용하지 않는다.

## 주의사항

DB 검색 결과는 정책 후보를 좁히는 1차 조회다. 원문에 `무관` 또는 `제한없음`으로
표시됐어도 지원내용·공고문에 세부 자격이 있을 수 있으므로 최종 신청 전 반드시
`source_url`, `application_site`, `support_content`와 최신 기관 공고를 확인해야 한다.
