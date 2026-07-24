# 파일 배치 & 실험 가이드 (SETUP_AND_EXPERIMENTS)

이 문서만 따라 하면 설치 → 데이터 배치 → 실험 → 서비스 실행까지 됩니다.

---

## 0. 5분 요약 (일단 돌려보기)

### 실행 환경 버전

| 항목 | 검증된 버전 | 최소 요구 |
|------|------------|----------|
| **Python** | 3.12.3 | 3.10 이상 (3.10~3.12 권장) |
| **pip** | 24.0 | 23 이상 |
| OS | Ubuntu 24 (Linux) | Linux / macOS / Windows 모두 가능 |

핵심 패키지(검증된 정확한 버전):

| 패키지 | 버전 | 용도 |
|--------|------|------|
| numpy | 2.4.4 | 수치연산 |
| pandas | 3.0.2 | 데이터 처리 |
| scipy | 1.17.1 | 통계분포 |
| scikit-learn | 1.8.0 | ML 모델·검증 |
| xgboost | 3.3.0 | 부스팅 모델 |
| lightgbm | 4.6.0 | 부스팅·LTR |
| imbalanced-learn | 0.14.2 | SMOTE(불균형) |
| openpyxl | 3.1.5 | KHUG xlsx 읽기 |
| pydantic | 2.13.4 | 스키마 |
| joblib | 1.5.3 | 모델 저장 |
| anthropic | 0.40+ | LLM API(서비스용, 선택) |
| fastapi / uvicorn | 0.139 / 0.51 | 서버(선택) |

> **두 가지 설치 방법**
> - `pip install -r requirements.txt` : 범위 지정(권장, 최신 호환 버전 자동)
> - `pip install -r requirements.lock.txt` : 위 표의 **정확한 버전 고정**(재현·디버깅용)
>
> Python 3.13은 일부 ML 패키지 휠이 아직 없을 수 있어 **3.12를 권장**합니다.
> 3.9 이하는 `X | None` 타입 문법 등으로 동작하지 않습니다(3.10+ 필수).

```bash
# 1) 압축 풀고 이동
tar -xzf jeonse_helper.tar.gz
cd jeonse_helper

# 2) 파이썬 3.10+ 가상환경 + 설치
python3 --version                                    # 3.10 이상인지 확인
python3 -m venv .venv && source .venv/bin/activate   # 윈도우: .venv\Scripts\activate
python -m pip install --upgrade pip                  # pip 최신화
pip install -r requirements.txt                      # 또는 requirements.lock.txt

# 3) 전체 파이프라인 한 방에 (데이터 생성→모델 학습→검증→DB→추천→테스트)
python run_pipeline.py

# 4) 대화형 실행
python -m src.agent.cli
```
OpenAI 키/주소는 `src/config.py`에 고정되어 있습니다. 다른 외부 API는 키 또는 캐시가
없을 때 항목별 mock으로 폴백합니다.

---

## 1. 파일 배치 규칙 (무엇을 어디에 두나)

폴더 구조와 "내가 채워야 하는 곳"은 다음과 같습니다.

```
jeonse_helper/
├── data/
│   ├── downloaded/               ← ★ 사용자가 내려받은 원본 실데이터를 여기 넣는다
│   │   ├── 주택도시보증공사_전세보증금반환보증 발급현황_20260331.csv   (이미 포함)
│   │   ├── 주택도시보증공사_지역별 전세금반환보증 사고현황_20250831.xlsx (이미 포함)
│   │   └── safety/              ← 치안 공공데이터 원본(이미 포함)
│   │       ├── CCTV정보.csv
│   │       ├── 안전비상벨위치정보.csv
│   │       ├── 경찰청_전국 치안센터 주소 현황_20251231.csv
│   │       └── 소방청_시도 소방서 현황_20250701.csv
│   └── generated/               ← 자동 생성물(직접 건드릴 필요 없음)
│       ├── properties.csv        (합성 매물)
│       ├── users.csv             (합성 사용자)
│       ├── click_labels.csv      (합성 클릭 라벨)
│       └── jeonse_helper.db      (SQLite DB)
├── models/                      ← 학습된 모델·실험 결과 CSV(자동)
├── src/                         ← 코드(수정해서 실험)
└── run_pipeline.py              ← 전체 실행 스크립트
```

### 1-1. 반드시 있어야 하는 파일 (이미 포함됨)
`data/downloaded/` 안의 KHUG CSV·XLSX 2개. 이게 합성 데이터의 "현실 앵커"입니다.
경로/파일명을 바꾸려면 `src/config.py`의 `RAW_ISSUE_CSV`, `RAW_ACCIDENT_XLSX` 수정.

### 1-2. 넣으면 좋은 파일 (선택 — 없으면 mock)
**치안 데이터** (`data/downloaded/safety/*.csv`): 원본 한글 파일명, CP949 인코딩,
`WGS84위도/경도` 컬럼을 코드가 직접 인식하므로 이름이나 컬럼을 바꿀 필요가 없습니다.

| 파일명 | 공공데이터포털 출처 |
|--------|--------------------|
| `CCTV정보.csv` | 전국CCTV표준데이터, WGS84 실집계 |
| `안전비상벨위치정보.csv` | 전국안전비상벨위치표준데이터, WGS84 실집계 |
| `경찰청_전국 치안센터 주소 현황_20251231.csv` | 주소 원본 669행 |
| `소방청_시도 소방서 현황_20250701.csv` | 주소 원본 242행 |

경찰/소방 주소를 좌표화하려면 `SafetyTool.geocoding_templates()`로 템플릿을 만든 뒤
`data/generated/safety_geocoded/*.csv`의 `lat`,`lng`를 채우면 됩니다.

### 1-3. API 설정
```bash
# OpenAI 키/주소/모델은 src/config.py에 고정됨
export JEONSE_LLM=api

# 지도/POI/시세 (없으면 항목별 mock)
export NAVER_MAP_CLIENT_ID=...
export NAVER_MAP_CLIENT_SECRET=...
export KAKAO_REST_API_KEY=...
export MOLIT_REALPRICE_API_KEY=...
```

생활안전지도 인증키는 `src/config.py`의 `SAFEMAP_SERVICE_KEY`에 넣고 다음을 실행합니다.

```bash
python -m src.tools.convenience_tool --refresh
```

CODEF는 기본 샌드박스라 별도 키가 필요 없습니다. 데모/운영만
`CODEF_SERVICE_TYPE`과 `CODEF_CLIENT_ID/SECRET/PUBLIC_KEY`를 설정합니다.

---

## 2. 실험 순서 (무엇을 어떻게 실험하나)

각 실험은 독립적으로 돌릴 수 있고, 결과는 `models/*.csv`에 저장됩니다.

### 실험 A — 데이터 증강 파라미터 튜닝
**목표**: 합성 데이터를 현실에 더 가깝게.
```bash
# 개수 바꿔가며 생성
python -m src.data_augmentation.generate --n_properties 10000 --n_users 4000

# 현실성 자동 검증(스키마/분포/양성률/인과신호/상관)
python -m tests.test_data_augmentation
```
**바꿔볼 것** (`src/config.py`):
- `AUCTION_RECOVERY_RATIO` (0.70 / 0.75 / 0.80) — 낙찰가율. 라벨 정의가 바뀜.
- `src/data_augmentation/generate.py`의 로짓 계수(현재 절편 -4.25 등) — 합성 위험 라벨 민감도 조정.
- `--recency-half-life-months` — 최신 거래를 반영하는 시간 가중치 조정.
- `data/generated/property_generation_quality.json` — 원본 대비 TVD/KS 지표 확인.
- 검증: 실제 KHUG 채무불이행 명단의 통계(평균 채무불이행기간, 지역분포)와 비교.

### 실험 B — 전세사기 위험 모델 선택 ★ 핵심
**목표**: 6개 모델 × 4개 피처세트 중 최적 조합 찾기.
```bash
python -m src.fraud_risk.train --experiment          # 전체 조합 비교
python -m src.fraud_risk.train --experiment --smote  # 불균형 보정 비교
```
결과: `models/fraud_experiment_results.csv` (PR-AUC 내림차순).
- **모델**: logistic, random_forest, grad_boost, xgboost, lightgbm, mlp
- **피처세트**: core(핵심4) / core_plus / raw / full → `src/fraud_risk/features.py`의 `FEATURE_SETS`에서 편집·추가
- **평가**: 사기=소수클래스라 **PR-AUC 우선**, ROC-AUC 보조.

최종 모델 저장(추론·서버용):
```bash
python -m src.fraud_risk.train --model xgboost --feature_set core_plus --save
```

### 실험 C — 위험 모델 검증 ★ "라벨 어떻게 검증?"의 답
```bash
python -m src.fraud_risk.validate                    # 교차검증+보정+임계값스윕
python -m src.fraud_risk.validate --real my_labels.csv   # 실측 라벨로 진짜 검증
```
- 지금은 합성 라벨 → **방법론 타당성** 검증(모델이 위험신호를 되찾는가).
- 실측 라벨(`property_id,true_fraud` CSV)을 넣으면 **실제 성능** 검증.
  실측 라벨 출처: HUG 보증사고 / 법원 경매 손실 / 전세사기피해자 결정 명단.
- 보정 곡선이 "확률 과대추정"을 드러내면 → 운영 전 `CalibratedClassifierCV` 적용.

### 실험 D — 추천 모델 선택
**목표**: 규칙 / LTR(LightGBM·XGBoost) / 콘텐츠 중 최적.
```bash
python -m src.recommender.click_labels           # 합성 클릭라벨 생성
python -m src.recommender.train --experiment      # 4종+랜덤 NDCG/MAP 비교
```
결과: `models/recommender_experiment_results.csv`.
- **바꿔볼 것**: `src/recommender/click_labels.py`의 `DEFAULT_PREF_WEIGHTS`
  ([예산,안전,통근,관리비,연식])로 "안전 중시 vs 예산 중시" 사용자 유형별 라벨 생성.
- 실사용자 클릭 로그가 모이면 `click_labels.csv`를 실데이터로 교체 후 동일 재학습.

### 실험 E — 에이전트/대화 흐름
```bash
python -m src.agent.cli          # 대화형(2단계 확인·조건누락그룹 확인)
python -m tests.test_integration # 15개 통합 테스트
```
- **바꿔볼 것**: `src/agent/planner.py`의 슬롯 추출 규칙 / QA 의도 정규식.
- 실제 LLM로 교체: `JEONSE_LLM=api` + 키 → 규칙 대신 LLM이 파싱.

---

## 3. 실서비스 실행

```bash
# 데이터·모델 준비(최초 1회)
python run_pipeline.py

# API 서버
pip install fastapi uvicorn
export JEONSE_LLM=api
uvicorn src.server.app:app --host 0.0.0.0 --port 8000
#   테스트: curl http://localhost:8000/health

# 금융제도 12h 자동 갱신(별도 프로세스)
pip install apscheduler
python -m src.server.scheduler
```
엔드포인트: `POST /session` → `POST /chat` (2단계 대화) / `POST /fraud/score` / `GET /health`.

---

## 4. 실데이터로 갈아끼우기 (합성 → 실측)

우선순위 순으로, 성능 향상이 큰 것부터:

1. **매물 실데이터**: `data/generated/properties.csv`를 실매물로 교체.
   컬럼은 `src/schemas.py`의 `PropertyRecord`에 맞추면 나머지 파이프라인 그대로 동작.
2. **위험 모델 실측 피처**: CODEF 등기부등본으로 `senior_mortgage_manwon` 등을 실측화.
   `python -c "from src.tools.codef_tool import collect_registry_batch; ..."`
3. **위험 모델 실측 라벨**: HUG 사고/피해자 명단 → `validate.py --real`로 재검증.
4. **치안/편의 실데이터**: 현재 `data/downloaded/safety` 자동 인식 + 생활안전지도 편의점 캐시.
5. **클릭 로그**: 실사용자 로그 → `click_labels.csv` 교체 → 추천 재학습.

각 단계는 독립적이라 하나씩 교체하며 성능 변화를 측정할 수 있습니다.
