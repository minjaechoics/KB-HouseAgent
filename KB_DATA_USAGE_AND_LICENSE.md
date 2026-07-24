# KB 데이터 사용 경계와 현실화 절차

## 결론

`KB부동산 데이터허브 > 투자테이블`은 아파트 단지의 매매·전세 시세와 변동률을
비교하는 통계 화면이며, 계약 가능한 현재 매물 목록이 아니다. 이 통계의 모든
행을 수집해 `properties` 테이블의 매물로 치환하면 데이터 의미도 달라지고,
KB 이용약관의 크롤링 금지 조항에도 저촉될 수 있다.

따라서 운영 코드는 `KB_DATA_LICENSED=false`를 기본으로 KB 내부 조회를 하지 않는다.
현재 매물 후보 DB는 출처와 재사용 범위가 명확한 국토교통부 실거래 자료 기반
합성 후보를 유지하며, 화면에 `합성 후보`임을 명시한다.

## 약관상 확인한 제한

- KB가 제작한 콘텐츠의 저작권·지식재산권은 KB에 있고, 사전 승인 없이 영리
  목적으로 복제·전송·배포할 수 없다고 명시되어 있다.
- 금지행위에 `KB부동산 및 제휴서비스 정보 등을 별도 허락이나 제휴 없이 도용하는
  행위(웹 크롤링 포함)`가 명시되어 있다.

확인 링크:

- [KB부동산 데이터허브 투자테이블](https://data.kbland.kr/kbstats/investment-table)
- [KB부동산 이용약관 PDF](https://img2.kbstar.com/obj/ocommon/250624_kbland_full.pdf)

## 허가를 받은 뒤의 적재 방법

KB의 서면 이용허락, 제휴 API 또는 재사용 가능한 공식 내보내기 파일을 받은 경우
아래 열을 가진 UTF-8 CSV로 정규화한다.

`complex_name,sido,gugun,dong,exclusive_area_m2,observed_date,sale_price_manwon,source_record_id`

그 다음 다음 명령을 실행한다.

```powershell
python scripts/import_authorized_kb_complex_export.py .\authorized-kb.csv `
  --license-reference "계약 또는 허가 식별자" --confirm-authorized
```

적재 대상은 `kb_authorized_complex_prices` 참조 테이블이다. 단지 시세 통계는 현재
매물과 구분한다. 실제 매물 DB를 치환하려면 매물 ID, 광고 유효기간, 중개사 정보,
거래유형, 가격, 주소, 면적 등 현재 매물 스키마와 그 데이터를 재배포할 권리가
포함된 별도 피드가 필요하다.

## 운영 플래그

문서화된 허가 범위에서 기존 KB 어댑터를 사용할 수 있을 때만 서버 환경에
`KB_DATA_LICENSED=true`를 설정한다. 허가 문서가 없다면 설정하지 않는다.
