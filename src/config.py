"""
중앙 설정 파일.

여기 모인 상수 중 '정책/공식' 성격의 값은 모두 출처(논문/기관 자료)를 주석으로
달아두었다. 실험할 때 이 값들을 바꿔가며 민감도 분석을 하면 된다.
"""
from __future__ import annotations
import os
from pathlib import Path

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
# User-downloaded source data lives here. Generated/synthetic artifacts stay in
# data/generated.
DATA_RAW = ROOT / "data" / "downloaded"
DATA_GEN = ROOT / "data" / "generated"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "data" / "generated" / "jeonse_helper.db"
FINANCE_POLICY_CSV = DATA_RAW / "finance_policies" / "youth_housing_policies.csv"
KB_FINANCE_DIR = DATA_RAW / "finance_products" / "kb"
KB_LOAN_XLSX = KB_FINANCE_DIR / "KB_Kookmin_Loan_Products_Scrape_2026-07-23.xlsx"
KB_LOAN_CSV = KB_FINANCE_DIR / "kb_kookmin_loan_products.csv"

# ----------------------------------------------------------------------
# 외부 API 설정
# ----------------------------------------------------------------------
# 비밀키는 웹 소스나 컨테이너 이미지에 넣지 않고 배포 환경에서만 주입한다.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
# 이전 세대이지만 지시 준수/JSON 계획 생성에 충분하고 Responses API를 지원한다.
OPENAI_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini").strip()
# 1차 모델이 일시적으로 사용할 수 없을 때 한 번만 전환하는 저비용 구세대 모델.
OPENAI_FALLBACK_MODEL = os.environ.get(
    "LLM_FALLBACK_MODEL", "gpt-4o-mini"
).strip()

# 생활안전지도 편의점 REST API. 서비스 키는 별도 발급값이 제공되지 않아 빈 값으로
# 둔다. 발급 후 이 상수만 채우면 전국 데이터를 로컬 캐시에 내려받아 사용한다.
SAFEMAP_CONVENIENCE_URL = "https://www.safemap.go.kr/openapi2/IF_0039"
SAFEMAP_SERVICE_KEY = os.environ.get("SAFEMAP_SERVICE_KEY", "").strip()

# 경기도 파출소·지구대 및 소방/경찰 시설 현황. 키는 private env에서만
# 주입하며, 갱신 스크립트가 좌표 CSV를 생성하면 런타임은 그 캐시만 읽는다.
GYEONGGI_OPENAPI_KEY = os.environ.get("GYEONGGI_OPENAPI_KEY", "").strip()
GYEONGGI_POLICE_URL = (
    "https://openapi.gg.go.kr/Ptrldvsnsubpolcstus"
)
GYEONGGI_SAFETY_FACILITY_URL = (
    "https://openapi.gg.go.kr/FiresttnPolcsttnM"
)

# NAVER Maps 키와 별개인 NAVER API HUB 검색 키. 뉴스/지역검색에서 공유한다.
NAVER_API_HUB_CLIENT_ID = os.environ.get("NAVER_API_HUB_CLIENT_ID", "").strip()
NAVER_API_HUB_CLIENT_SECRET = os.environ.get("NAVER_API_HUB_CLIENT_SECRET", "").strip()

# Bright Data is only a delivery transport.  Source-site rights must be recorded
# separately before any listing snapshot can be imported.
BRIGHTDATA_API_TOKEN = os.environ.get("BRIGHTDATA_API_TOKEN", "").strip()
BRIGHTDATA_DATASET_ID = os.environ.get("BRIGHTDATA_DATASET_ID", "").strip()

# 한국부동산원 R-ONE Open API.  통계표 메타데이터와 지역별 가격·수급
# 시계열을 서버에서 동기화하며 브라우저에는 키를 절대 전달하지 않는다.
RONE_API_KEY = os.environ.get("RONE_API_KEY", "").strip()
RONE_BASE_URL = os.environ.get(
    "RONE_BASE_URL", "https://www.reb.or.kr/r-one/openapi"
).rstrip("/")

# V-World의 전기차 충전소 활용모델은 실제 충전기 원천으로 공공데이터포털
# EvInfoServiceV2를 사용한다. 해당 서비스의 별도 활용신청 키가 있을 때만
# 실시간 상태를 조회한다(국토부 RTMS 승인키와는 권한이 별개다).
EV_CHARGER_SERVICE_KEY = os.environ.get("EV_CHARGER_SERVICE_KEY", "").strip()
EV_CHARGER_API_URL = os.environ.get(
    "EV_CHARGER_API_URL",
    "https://api.odcloud.kr/api/EvInfoServiceV2/v1/getEvSearchList",
).strip()

# 국토부 실거래가 API 키. 원천 CSV 갱신/상세 조회 작업에만 사용한다.
MOLIT_RTMS_SERVICE_KEY = os.environ.get("MOLIT_RTMS_SERVICE_KEY", "").strip()

for _p in (DATA_GEN, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# 실제 원본 파일명 (업로드된 KHUG 데이터)
RAW_ISSUE_CSV = DATA_RAW / "real_estate" / "hug" / "hug_return_guarantee_issuance_20260331.csv"
RAW_ACCIDENT_XLSX = DATA_RAW / "real_estate" / "hug" / "hug_return_guarantee_accidents_by_region_20250831.xlsx"

# ----------------------------------------------------------------------
# 전세사기 위험 공식 파라미터
# ----------------------------------------------------------------------
# 핵심 안전식(업계/변호사 실무 기준):
#   (내 보증금 + 선순위 보증금 총합 + 선순위 근저당 채권최고액) <= 시세 * 낙찰가율
# 경매 낙찰가율은 통상 시세의 70~80%로 본다.
#   출처: https://brunch.co.kr/@b2fa784ba86f4b0/19
#         https://biz.heraldcorp.com/article/10776093 (안심전세앱 80% 기준)
AUCTION_RECOVERY_RATIO = 0.75          # 실험용 기본값(0.70~0.80 사이 스윕 권장)

# 위 안전식에서 파생되는 "선순위담보비율" 임계.
#   선순위담보비율 = (선순위보증금 + 선순위근저당) / (시세 * 낙찰가율)
#   내 보증금까지 더한 값(부채비율)이 1.0을 넘으면 원금 손실 위험.
DEBT_RATIO_WARN = 0.8                   # 주의
DEBT_RATIO_DANGER = 1.0                 # 위험(원금 손실 가능)

# ----------------------------------------------------------------------
# 적정 주거비(사용자 선호 점수화) 공식 파라미터
# ----------------------------------------------------------------------
# RIR(Rent to Income Ratio) = 월주거비 / 월소득.
# UN-Habitat / OECD: 가처분소득의 30% 초과 시 '주거비 과부담'.
# 한국 지역사회보장지표: 월소득 대비 임대료 25% 초과를 과부담으로 정의.
#   출처: https://calculkorea.com/rent-burden-calculator
#         https://www.ejhuf.org/archive/view_article?pid=jhuf-4-2-159
#         연세대 IPAID, 저소득 임차가구 주거비부담 분석(RIR 30% 룰)
RIR_TARGET = 0.25          # 권장 상한(보수적)
RIR_MAX = 0.30             # 절대 상한(초과 시 과부담)

# 전세보증금 -> 월세 환산(전월세전환율). 지역/시기별로 다르나 실무 기본 5~6%.
#   전월세전환율(연) 적용: 월세환산 = 보증금 * 전환율 / 12
JEONSE_TO_MONTHLY_RATE = 0.055   # 연 5.5% (실험 시 지역별로 조정)

# 관리비를 주거비에 포함(UN/실무 기준 관리비는 필수 주거비).
INCLUDE_MAINTENANCE_IN_RIR = True

# 비상금(emergency fund) 권장: 생활비의 N개월치는 보증금으로 소진하지 않도록.
EMERGENCY_FUND_MONTHS = 3

# 매매 예산의 자기자본 최소 비중 가정. 매매를 월세 0원으로 평가하지 않기 위한
# 추천용 보수적 상한이며 실제 LTV·DSR 심사를 대체하지 않는다.
PURCHASE_EQUITY_RATIO = 0.30

# 50/30/20 예산 법칙에서 필수지출(50%) 중 주거가 최대 절반 -> 소득의 25% 가이드.
#   출처: https://calculkorea.com/rent-burden-calculator

# ----------------------------------------------------------------------
# 청년 정의 / 소득분위
# ----------------------------------------------------------------------
YOUTH_AGE_MIN = 19
YOUTH_AGE_MAX = 39   # 다수 청년정책 기준 만 19~39세

# 2024 통계청 가계금융복지조사 근사 소득분위 경계(월 세전, 1인가구 근사, 만원).
# 실제 서비스에서는 최신 통계로 교체할 것. (합성데이터 소득 분포 앵커로만 사용)
INCOME_DECILE_BOUNDARIES_MAN = [
    120, 170, 210, 250, 290, 340, 400, 480, 620
]  # 9개 경계 -> 10분위

# ----------------------------------------------------------------------
# 재현성
# ----------------------------------------------------------------------
GLOBAL_SEED = int(os.environ.get("JEONSE_SEED", "42"))

# 실제 경로 API는 좌표/SQL로 줄인 후보에만 호출한다. 10만 건 전체에 API를
# 호출하지 않으며, 한 검색 요청 전체에서 최대 5개 후보만 실경로로 검증한다.
ROUTE_API_EXACT_CANDIDATE_LIMIT = max(
    1, int(os.environ.get("ROUTE_API_EXACT_CANDIDATE_LIMIT", "5")))
ROUTE_API_MAX_WORKERS = max(
    1, min(12, int(os.environ.get("ROUTE_API_MAX_WORKERS", "3"))))
