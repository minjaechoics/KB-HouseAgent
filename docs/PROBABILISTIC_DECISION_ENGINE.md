# 확률적 주택 의사결정 엔진 구현 백서

## 1. 목적과 책임 경계

이 모듈은 LLM이 숫자를 상상해 추천하는 구조가 아니다. LLM은 자연어 조건을 구조화하고 근거를 설명하지만, 자산 경로·상환 가능성·최적 조합·예측구간은 재현 가능한 수치 모듈이 계산한다.

~~~text
사용자 입력
  → 승인된 성향 프로필
  → 초기 조건 교집합 매물
  → 금융상품 예비 자격 판정
  → 10,000경로 Monte Carlo
  → 매물×상품×대출액 MILP
  → 파레토 대표 후보
  → decision_run_id 감사 추적
  → LLM 근거 설명
~~~

위험도는 정렬·목적함수의 소프트 요인이다. 사용자가 명시적으로 요청하지 않는 한 위험도만으로 매물을 검색 집합에서 제거하지 않는다. 금융상품 판정도 은행의 최종 심사를 대신하지 않는다.

## 2. 확률적 자산 시뮬레이션

구현 위치:

- src/simulation/distributions.py: 상관 월 충격과 연율→월 로그수익률 변환
- src/simulation/monte_carlo.py: 10,000개 월별 경로
- src/simulation/metrics.py: 분위수, 확률, CVaR
- src/report/service.py: 상세 리포트 결합

### 2.1 상태 변수

각 경로 i, 월 t에 대해 현금성 자산 L, 주택가치 H, 임차보증금 D, 대출잔액 B, 소득 Y, 생활비 C를 갱신한다.

NW = L + H + D - B

L(next) = L × (1 + 금융자산수익률) + 소득 - 생활비 - 월세 - 관리비 - 상환액 + 상속유입

원리금균등상환은 이자와 원금을 분리해 잔액을 차감하고, 만기일시상환 상품은 월 이자와 만기 보증금 상계를 분리한다.

### 2.2 확률분포와 상관

소득상승률, 물가상승률, 금융자산수익률, 집값상승률은 고정 한 줄이 아니라 상관된 월 충격으로 생성한다. 집값 변동성은 해당 예측의 하한·상한에서 역산한다. 입력 가정과 seed가 같으면 결과도 같다.

기본 출력:

- 10년 후 및 종료시점 순자산 P2.5/P10/P50/P90/P97.5
- 현금성 자산, 집값, 부채의 연도별 분위수
- 현금 고갈 확률
- 상환액이 가처분현금을 3개월 연속 넘는 상환곤란 확률
- 최악 5% 경로의 평균 순자산 변화(CVaR 5%)
- 금리 +2%p 스트레스의 확률·중앙값 변화
- 선택적으로 연 5% 실직 진입, 3~9개월 소득 20% 경로
- 사용자가 입력한 결혼·출산·상속 일정

CVaR은 손실 꼬리의 평균이므로 단순한 최악 한 건보다 안정적이다. 다만 출력 확률은 입력분포와 모델에 조건부인 모형 확률이지 개인의 실제 연체확률은 아니다.

### 2.3 금융상품 공정 비교

상품별 경로는 가능한 한 동일한 난수와 seed를 공유한다. 따라서 상품 A와 B의 차이는 서로 다른 임의표본 때문이 아니라 금리·한도·상환방식의 차이에 가깝다. 화면의 P10–P90 부채꼴과 P50 선으로 불확실성을 표시한다.

## 3. 매물×금융상품×대출액 최적화

구현 위치: src/optimization/service.py

### 3.1 후보 생성

현재 검색 교집합 안의 최대 60개 매물, 예비 적합 금융상품 최대 8개, 상품별 대출액 격자를 조합한다. 대출액은 최소 필요액·중간값·공개/상환 한도와 사용자 지정액으로 이산화한다.

### 3.2 하드 제약

- 계약 필요 현금 ≤ 보유자산에서 6개월 비상자금을 제외한 금액 + 대출
- 금융상품 거래유형·공개조건 예비 적합
- 상품 공개한도, 담보/보증금 비율 한도, 소득 기반 상환한도 이내
- 월 대출상환액 ≤ 월 가처분소득의 35%
- 정확히 하나의 매물·상품·대출액 조합 선택

제약을 만족하지 못하는 후보는 목적함수 점수가 좋아도 선택할 수 없다.

### 3.3 목적함수와 성향

사용자 성향은 안정형·균형형·성장형 프리셋 또는 승인된 가중치로 저장한다.

U(j) = asset weight × asset score + burden weight × burden score + safety weight × safety score + commute weight × commute score + liquidity weight × liquidity score + debt-aversion weight × debt score

각 점수는 후보 집합에서 0~1로 정규화한다. 매물 검색의 미승인 자연어에서 가중치를 몰래 바꾸지 않는다.

정수변수 x(j)는 0 또는 1이며 합은 1이다. 구현은 SciPy milp의 HiGHS 래퍼를 사용하며, 솔버가 불가능하면 같은 효용의 결정론적 argmax로 폴백하고 감사로그에 남긴다.

### 3.4 파레토 결과

한 개의 ‘정답’을 강요하지 않고 다음 대표점을 중복 제거해 제공한다.

- 승인된 내 성향
- 10년 후 순자산 우선
- 월 부담 최소
- 안전 우선
- 통근 우선

어떤 후보가 모든 목적에서 같거나 더 낫고 하나 이상에서 엄격히 더 좋으면 지배된 후보로 제거한다.

## 4. 집값 시계열과 불확실성 보정

구현 위치:

- src/market_forecast/backtest.py: expanding-window walk-forward 검증
- src/market_forecast/conformal.py: 잔차 분위수 보정
- src/market_forecast/train.py: 후보 모델 비교·선택
- src/market_forecast/model.py: 추론과 80%·95% 구간

### 4.1 누출 방지

각 fold는 검증월보다 과거인 데이터만 학습한다. seasonal naive, Ridge, HistGradientBoosting, 설치된 경우 LightGBM을 동일 fold에서 비교한다. MAE, 0에 가까운 값에 바닥을 둔 MAPE, 상승·하락 방향 정확도를 기록한다.

현재 학습 산출물 rtms_walkforward_conformal_v4의 6개 walk-forward fold 결과:

| 모델 | 월 로그수익률 MAE | 방향 정확도 |
|---|---:|---:|
| Seasonal naive | 0.05909 | 0.2429 |
| Ridge | 0.03143 | 0.6021 |
| HistGradientBoosting | 0.03082 | 0.6227 |
| LightGBM | **0.03044** | **0.6227** |

따라서 현재 산출물은 LightGBM을 선택했다. 표본은 1,131행, 원천 범위는 2024-07~2026-07이다. 데이터가 바뀌면 재학습 결과도 바뀐다.

### 4.2 Conformal 보정

원시 구간의 검증 포함률은 약 58.7%였다. walk-forward OOF 잔차로 80%·95% 구간을 보정한 뒤 검증 포함률은 각각 약 80.4%, 95.3%였다. 이는 같은 분포라는 교환가능성 가정 아래의 경험적 보정이며 미래의 구조변화를 보장하지 않는다.

TFT는 정적 특성, 알려진 미래 변수, 과거 외생변수를 함께 다룰 수 있지만, 현재 표본 1,131행·24개월은 복잡한 신경망을 정당화하기 부족하다고 판단했다. 코드의 적격 기준은 5,000행 이상·36개월 이상이며 현재는 명시적으로 skip한다. 데이터가 충분해져도 walk-forward에서 단순 모델보다 좋아야만 채택한다.

뉴스는 가격을 직접 올리거나 내리는 숫자로 사용하지 않는다. 관련성·방향·근거를 LLM이 설명하는 정성 외생정보일 뿐이고, 수치 전망은 실거래 시계열에서만 나온다.

## 5. 모든 판단의 근거 추적

구현 위치: src/audit/store.py, API GET /api/decisions/{decision_run_id}

decision_run_id 단위로 다음을 SQLite에 저장한다.

- 사용자 입력 스냅샷과 선택한 매물
- 데이터·코드·모델 버전과 Monte Carlo seed
- 금융 Text-to-SQL 문장과 바인딩 파라미터
- 조회된 매물·금융상품 ID와 원천
- 각 도구의 입력/출력 SHA-256 digest, 실행시간, 폴백
- 확률 시뮬레이션 요약과 최종 판단

키 이름에 token, secret, password, api_key, openai 등이 있거나 sk-로 시작하는 문자열은 저장 직전에 [REDACTED]로 바뀐다.

## 6. 에이전트 회귀평가

구현 위치: src/evaluation/

build_golden_cases()는 금융·계약·비용·POI·시세·등기·치안·생활편의·감당가능성·매물추천을 포함하는 150개 한국어 질문을 만든다.

~~~powershell
$env:JEONSE_LLM='mock'
python -m src.evaluation.cli artifacts/agent_eval.json
~~~

기본 평가기는 의도 정확도, confirm/proceed/clarify 행동 정확도, 필수 조건 추출 재현율, 스키마 밖 조건 생성률, 평균/P50/P95 계획 지연시간을 계산한다. 주입형 probe가 있을 때 SQL 실행 성공률, 검색 재현율, 근거 충실도, API 복구율, 요청비용도 같은 보고서에 기록한다.

현재 결정론 플래너 150문항 결과는 의도 90%, 행동 90%, 필수 슬롯 100%, 부적절 조건 0%, P95 약 0.1ms 미만이었다. 이는 규칙 플래너 회귀 기준일 뿐 실제 API LLM의 품질을 대표하지 않는다. 배포 CI에서는 실제 Text-to-SQL DB와 검색 정답셋, 근거 기반 답변 judge를 probe로 주입해야 한다. RAGAS가 제안한 것처럼 검색 문맥의 관련성과 답변의 근거 충실도를 분리하는 이유도 여기에 있다.

## 7. 검증과 남은 한계

- 같은 seed의 Monte Carlo 재현성, 분위수 순서, 금리 스트레스 방향을 단위시험한다.
- 현금·자격·상환 하드제약을 위반한 MILP 후보가 반환되지 않는지 시험한다.
- walk-forward 학습월이 검증월보다 항상 과거인지 시험한다.
- 감사 저장소가 키를 마스킹하고 단계 순서를 복원하는지 시험한다.
- UI는 숫자 한 줄 대신 P10–P90 부채꼴과 판단 ID를 표시한다.

아직 별도로 보강할 부분:

- 금융상품 공개조건에 없는 DSR·신용점수·담보평가 결과
- 실직확률과 자녀비용 분포의 한국 표본 재추정
- 현재 팔달구 프로토타입을 넘는 외부타당성
- 가격 예측의 더 긴 월별 실거래 이력
- 전문가 라벨이 있는 RAG 검색·답변 평가셋

## 8. 참고문헌과 구현 근거

1. Lim, Arik, Loeff, Pfister, [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363), 2019/International Journal of Forecasting.
2. Romano, Patterson, Candès, [Conformalized Quantile Regression](https://papers.nips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html), NeurIPS 2019.
3. Es, James, Espinosa-Anke, Schockaert, [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16.pdf), EACL 2024.
4. SciPy, [scipy.optimize.milp API documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), HiGHS 기반 MILP.
5. Rockafellar, Uryasev, Optimization of Conditional Value-at-Risk, Journal of Risk 2(3), 2000.

## 9. 모델 해석 원칙

이 시스템의 결과는 ‘살 수 있다/없다’의 법적·금융적 확정판정이 아니다. 공개조건으로 자금조달이 불가능하면 시뮬레이션을 차단하고, 가능하더라도 은행 심사·등기·보증가입·계약현장 확인을 최종 행동으로 남긴다. 숫자, 근거, 모델 버전, 불확실성, 제한을 함께 제시하는 것이 제품의 핵심 원칙이다.
