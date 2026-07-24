# Bright Data 부동산 데이터 통합 설계 및 운영 상태

## 결론

앱은 이제 세 데이터 등급을 명시적으로 구분한다.

1. **국토교통부 실거래 시계열**: 내려받아 둔 RTMS 신고 자료를 월별로 집계한다. 실제 계약 신고 자료이며 현재 호가나 활성 매물은 아니다.
2. **승인된 실매물 피드**: Bright Data 스냅샷이더라도 원 사이트/콘텐츠의 이용허락 근거가 확인된 데이터만 가져온다. 수집시각과 만료시각을 기록하며 만료 매물은 검색에서 제외한다.
3. **분석용 합성 후보**: 기존 전국 10만 건 후보이다. 실매물이라고 표시하지 않으며 중개업소 연락처·현재 판매 여부를 보장하지 않는다.

## 이번 계정 점검 결과 (2026-07-23)

- 제공된 Bright Data API 키 인증은 성공했다.
- 계정의 Web Unlocker zone은 0개였다.
- 계정에서 조회 가능한 Marketplace 데이터셋 목록에는 NAVER 부동산 데이터셋이 없었다.
- NAVER 이용약관은 사전 허락 없는 scraper 등 자동화 수집과 IP 변경/CAPTCHA 우회를 금지한다.
- 그러므로 Bright Data가 전송 수단이라는 이유만으로 NAVER 부동산을 직접 수집하지 않았다. NAVER 또는 권리자로부터 계약/허락을 받은 뒤에만 `contract:` 또는 `permission:` 근거로 피드를 활성화할 수 있다.

## 데이터베이스

### `property_price_observations`

국토교통부 아파트·오피스텔·단독/다가구 매매 및 임대 신고 자료를 보관한다. 상업업무용과 분양권 전매는 일반 주거 매물 비교에서 제외했다. 현재 검증 완료된 적재 범위는 **서울 25개 시군구, 647,877개 고유 관측, 2024-07~2026-07**이다.

2026-07-23에 최신 전국 법정동 코드 269개를 확보하고 전국 수집을 시도했지만, 공공 API가 2,000/3,228 작업 시점에 연결을 종료했다. 기존 다운로드 파일을 불완전한 결과로 덮어쓰지 않았으며 서울 밖 지역은 UI에서 `시계열 없음`으로 표시한다. 전국 자료가 완료되기 전까지 전국 범위라고 주장하지 않는다.

현재 내려받은 아파트 원천은 매매이며 아파트 전월세 원천은 없다. 따라서 서울 아파트 매매 리포트는 실제 월별 시계열을 표시하지만, 아파트 전세·월세는 다른 유형의 가격을 대신 표시하지 않고 `비교 가능한 시계열 부족`으로 남긴다. 오피스텔과 단독/다가구는 보유한 임대 원천 범위 안에서 전세·월세를 구분한다.

주요 필드: 원천 데이터셋, 거래일, 법정동 코드, 단지명, 주택유형, 거래유형, 매매가, 보증금, 월세, 면적, 층, 건축연도, 취소 여부, 적재시각, 원문 해시.

### `listing_sources`

공급자, 원 도메인, 승인 여부, 계약/허락 참조값, 마지막 동기화 시각을 기록한다.

### `live_property_listings`

승인된 실매물의 원본과 정규화 결과, 최초/최근 확인시각, 만료시각, 활성 여부, 콘텐츠 해시를 기록한다. `(source_id, source_listing_id)`로 중복을 제거한다.

### `feed_sync_runs`

수집 실행의 입력·성공·거절 건수와 오류를 남기기 위한 감사 테이블이다.

## 시계열 표시 방식

- 매매는 같은 시군구·주택유형의 월평균 실거래가를 표시한다.
- 전세는 월평균 보증금을 표시한다.
- 월세는 월평균 월세와 평균 보증금을 함께 보관한다.
- 단지 동일성을 확정할 수 없는 합성 후보에는 단지 시세라고 쓰지 않고 `시군구·주택유형·거래유형 비교`라고 표시한다.
- 취소 신고는 집계에서 제외한다.

## 승인된 Bright Data 스냅샷 가져오기

```powershell
python scripts/sync_real_estate_feeds.py snapshot licensed.json `
  --provider 계약상대방 `
  --source-url https://licensed-provider.example/feed `
  --license-reference contract:계약번호 `
  --confirm-rights
```

필수 정규화 필드는 매물 ID, 한국 좌표, 거래유형, 주소이다. 매매/전세/월세 가격과 면적·방·층·중개정보는 원본에 있을 때만 채운다. 필수값이 없는 행은 조용히 추정하지 않고 거절한다.

NAVER/KB 도메인은 `contract:` 또는 `permission:`으로 시작하는 구체적인 이용근거가 없으면 코드 수준에서 거절한다.

## Bright Data 계정 확인

```powershell
python scripts/sync_real_estate_feeds.py brightdata-status
```

키 자체는 `deploy/BRIGHTDATA_KEYS.private.env`로만 주입하고 API/UI/로그에는 반환하지 않는다.

## 운영 API

`GET /api/data-sources/status`는 키를 노출하지 않고 활성 실매물 수, 실거래 관측 수·기간, 등록한 공급자와 데이터 정책만 반환한다.

## 참고

- Bright Data Dataset API: https://docs.brightdata.com/api-reference/marketplace-dataset-api/get-dataset-list
- Bright Data 비동기 수집/스냅샷: https://docs.brightdata.com/api-reference/rest-api/scraper/asynchronous-requests
- NAVER 이용약관: https://policy.naver.com/policy/service.html
- 국토교통부 실거래가 공개시스템: https://rt.molit.go.kr/
