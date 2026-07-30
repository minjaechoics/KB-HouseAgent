# 콘솔 비교 실험

실험 목적별로 평가셋을 분리한다. `speed_test.py`와 `alg_test.py`는 복잡한 조건 검색
50문항을 사용하고, `hallucination.py`는 실제 AWS 상담 기능을 대표하는 혼합 의도
50문항을 사용한다.

## 알고리즘만 통제 비교: `alg_test.py`

병렬 처리 효과를 제외하고 Atomic·교집합·의존성 스케줄링 알고리즘 자체만 비교할 때는
다음을 실행한다.

```powershell
py -3 alg_test.py
```

이 실험은 50개 Agentic 프롬프트를 공통으로 딱 한 번씩만 실행하며, 프롬프트 단계는
worker 6으로 병렬 처리한다. 생성된 동일한 구조화 결정을 메모리에 고정한 뒤 다음 두
검색 분기를 각각 같은 worker 6으로 실행한다.

- `OPTIMIZED`: Atomic 조건 분해, 초기집합 교집합, 의존성 스케줄링
- `NAIVE`: 조건을 분해·스케줄링하지 않는 단일 SQL

따라서 LLM 응답 차이와 병렬도 차이는 제거되고 검색 알고리즘만 달라진다. 화면에는 각
질의의 공통 LLM 출력, 두 RAG JSON, 독립 정답 판정을 모두 표시하며 마지막에 정확도,
검색 배치시간, 실제 관측 동시성을 나란히 출력한다. 라이브 API를 실행하기 전에 다음처럼
화면과 보고서 구조만 빠르게 확인할 수 있다.

```powershell
py -3 alg_test.py --mock --limit 4 --workers 2 --no-wait
```

## 전체 파이프라인 비교: `speed_test.py`, `hallucination.py`

`speed_test.py`와 `hallucination.py`의 1/2 모드는 다음처럼 정의한다.

- `1`: 현재 시스템 — 운영 Agentic 프롬프트, Atomic 조건 분해, 초기집합 교집합,
  의존성 스케줄링과 병렬 처리를 사용한다. 독립 질의도 `LLM_MAX_CONCURRENCY`
  기본값 6까지 동시에 실행한다.
- `2`: NAIVE 기준선 — 사용자 질문 전체를 별도의 평면적 시스템 프롬프트에 한 번에
  넣어 의도·조건을 즉시 추출한다. Atomic 프롬프트 분해, 의존성 그래프, 재계획,
  LLM Text-to-SQL, LLM 최종 합성을 사용하지 않는다. 추출 결과는 단일 SQL과 직렬
  도구 경로로 실행하며 질의 간 실행도 worker 1로 강제한다.

같게 고정되는 것은 50개 사용자 질문, 모델명, API 자격증명, DB 스냅샷, 채점 대상
슬롯 계약과 사람이 작성한 정답지다. 시스템 프롬프트·출력 스키마·실행 방식은 비교하려는
실험 처치이므로 의도적으로 다르며 각 프롬프트 지문을 결과에 기록한다.

## 실행

프로젝트 루트에서 다음을 각각 실행한다.

```powershell
py -3 speed_test.py
py -3 hallucination.py
```

PPT 비교용으로 선택 메뉴 없이 각각 실행하려면 다음과 같다.

```powershell
py -3 speed_test.py --mode optimized
py -3 speed_test.py --mode naive
py -3 hallucination.py --mode optimized
py -3 hallucination.py --mode naive
```

최적화 모드의 동시 요청 수를 직접 바꾸려면 `--workers`를 사용한다. NAIVE에서는
비교 기준을 보존하기 위해 이 값을 지정해도 항상 1이다.

```powershell
py -3 speed_test.py --mode optimized --workers 6
```

속도 실험은 50개 복잡 조건 검색을 처리한다. 환각 실험은 실제 AWS 상담 경로를 대표하는
혼합 50문항을 처리한다. 현재 시스템은 계획 이후 근거 기반 합성과 일부 Text-to-SQL
호출이 추가될 수 있고, NAIVE는 질의당 단일 LLM 추출 이후 결정론적 직렬 경로만 사용한다.
키와 모델은 서비스와 똑같이
`OPENAI_API_KEY`, `LLM_MODEL` 설정을 따른다. 화면 상단에는 누적 시간 또는 현재 정답률이
고정되며, 아래에는 입력, LLM 구조화 출력, 추천 매물, 원시 RAG JSON이 계속 출력된다.
마지막에는 종료되지 않고 아무 키 입력을 기다린다.

현재 CMD 세션에 키가 없으면 실행 직후 키를 숨김 입력으로 묻는다. 네 번의 비교 실행마다
다시 입력하기 싫다면 `deploy/OPENAI_KEYS.private.env.example`을 복사해
`deploy/OPENAI_KEYS.private.env`로 만들고 값을 채운다. 이 private 파일은 Git에서 제외된다.

환경변수로 직접 넣어도 된다. 키는 일반 소스 파일이나 Git에 저장하지 않는다.

```cmd
set OPENAI_API_KEY=발급받은_키
```

결과 원본은 `reports/experiments/` 아래 JSON으로 저장된다. 속도 보고서는 질의별
end-to-end 시간과 평균/P50/P95뿐 아니라 `configured_workers`, 실제 시간구간으로 계산한
`observed_peak_concurrency`, 병렬 배치시간, 초당 처리량을 기록한다. 정확도 보고서는
50개 정답 조건·정답 매물 ID·판정 사유를 모두 보존한다. API 비용 없이 화면만 점검할
때에만 `--mock`을 사용한다.

라이브 실험은 시작 전에 OpenAI 인증·모델·Structured Output을 사전 점검한다. 실행 중
API 호출이 규칙 기반 폴백으로 바뀌면 해당 실행을 즉시 실패 처리하며, 빠른 폴백 시간이
정상적인 성능 결과로 저장되지 않는다. 보고서의 `api_credential_fingerprint`로 키 원문을
노출하지 않고 두 실행이 같은 자격 증명을 사용했는지 확인할 수 있다.
`planner_prompt_fingerprint`는 두 모드에서 달라야 한다. 현재 시스템은 운영 Agentic
프롬프트, NAIVE는 평면적 전체 질문 단일 추출 프롬프트의 버전을 각각 기록한다.

```powershell
py -3 speed_test.py --mode optimized --mock
```

## 정확도 정의

`hallucination.py`는 다음 분포의 사람이 작성한 50문항을 사용한다.

- 일반 조건 추가·승인 협상 15문항
- 대출 포함 최적 매물 10문항
- 전세·월세 Monte Carlo 비교 8문항
- 선택 지역 집값 전망 7문항
- 지금 매수·1~2년 대기 비교 5문항
- 같은 예산의 대체 지역 추천 5문항

한 문제는 해당 유형에 필요한 다음 항목을 **모두** 만족해야 1점이다.

1. 사람이 고정한 의도와 응답 유형으로 정확히 라우팅한다.
2. 유형별 필수 도구를 올바른 순서로 호출하고 플래너 규칙 폴백이 없다.
3. 조건 추가 채팅은 슬롯 경계를 정확히 제안하고 버튼 승인 전에 검색·SQL을 실행하지 않는다.
4. 최적 매물은 Pareto/MILP가 성공하고 모든 자금·자격·상환 hard constraint를 만족한다.
5. 전월세 비교는 각각 3,000경로이며 P10≤P50≤P90, 우세안과 격차 계산이 일치한다.
6. 집값 전망은 선택 매물에 고정된 시계열·뉴스 수치와 방향을 변조하지 않는다.
7. 매수·대기 판단은 1년·2년 미래가격과 대기 주거비 산술이 일치한다.
8. 대체 지역 추천은 현재 동을 실제 DB 결과에서 제외한다.
9. 최종 LLM 답변의 모든 숫자는 구조화 결과·도구 근거·사용자 질문 중 하나에 존재한다.
10. 파이프라인 오류가 없다.

정답은 LLM 답변으로 만들지 않는다. 사람의 의도·도구 계약, 별도 파라미터 SQL,
고정 시계열 fixture, 결정론적 금융 산술로 판정한다. 최종 결과에는 모든 항목을 통과한
`exact_case_accuracy`와 부분 실패를 볼 수 있는 `mean_component_accuracy`를 함께 기록한다.
