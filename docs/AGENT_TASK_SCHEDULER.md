# Atom 기반 에이전트 Task DAG 스케줄러

## 1. 목적

사용자의 자연어 조건을 매물 검색 조건 Atom으로 정규화한 뒤, Atom을 실제
도구 실행 작업으로 컴파일한다. 저비용 SQL을 먼저 수행하고, 외부 경로 API와
LLM처럼 비싼 작업은 선행 후보 축소가 끝난 뒤 실행한다.

이 스케줄러는 추천 결과를 바꾸는 모델이 아니다. 동일한 필수 작업의 선후관계,
병렬 실행 폭, API 자원 제약을 최적화해 응답시간을 줄이고 실행 근거를 남긴다.

## 2. 공통 TaskGraph IR

`src/scheduling/core.py`의 `TaskNode`는 다음 정보를 가진다.

| 필드 | 의미 |
|---|---|
| `id` | 실행 작업 식별자 |
| `duration_ms` | 과거 실행 또는 보수적 기본값 기반 예상시간 |
| `resource` | SQLite, OpenAI, TMAP, NAVER Directions 등 |
| `dependencies` | 먼저 완료되어야 하는 작업 |
| `demand` | 해당 자원의 동시 용량 사용량 |
| `expected_tokens` | LLM TPM 계획용 예상 토큰 |
| `monetary_cost` | 외부 API 예상비용 |
| `candidate_reduction` | 후보 제거 예상효과 |
| `user_importance` | 사용자에게 중요한 조건의 우선도 |

모든 알고리즘은 같은 `TaskGraph`, `ResourceLimits`를 입력받고 동일한
`ScheduleResult`를 반환한다. 따라서 알고리즘별 결과를 같은 검증기와
벤치마크로 비교할 수 있다.

## 3. 조건 Atom 컴파일

구현 위치: `src/scheduling/compiler.py`

```text
초기 조건 SQL
 ├─ 가격 SQL
 ├─ 주택유형 SQL
 └─ 지역 SQL
       ↓
공간 bounding-box 사전 필터
       ↓
TMAP·NAVER 후보 전체 경로 검증
       ↓
통근시간 교집합
       ↓
최종 매물 조회
```

각 AI 조건은 초기 조건의 교집합 안에서만 검색된다. TMAP과 NAVER Directions는
SQL·공간 사전 필터가 끝난 이후에만 실행된다. TMAP Premium 대중교통과 NAVER
Directions 자동차 경로 모두 애플리케이션 수준의 후보 수 상한 없이 검증한다.

## 4. 구현 알고리즘

### 4.1 Topological FIFO

결정론적 위상정렬 기준선이다. 개선 알고리즘의 효과를 비교하고 모든 정밀
솔버가 실패했을 때 실행 가능한 계획을 제공한다.

### 4.2 Cost-aware HEFT

critical path의 upward rank에 다음을 결합한 list scheduling이다.

```text
우선순위 =
  critical path
  + 후보 제거 효과
  + 사용자 중요도
  + 후속 작업 해제 효과
  - 토큰 부담
  - 외부 API 비용
```

실행시간이 짧고 Branch-and-Bound의 초기 upper bound와 CP-SAT hint로도
사용된다.

### 4.3 Anytime Branch-and-Bound

선행관계를 만족하는 priority list를 분기한다.

- HEFT를 초기 upper bound로 사용
- critical-path lower bound
- 자원별 총 작업량 lower bound
- lower bound가 현재 최선해 이상이면 가지치기
- 제한시간 내 전체 탐색 시 최적해 증명
- 제한시간 도달 시 현재 최선해와 optimality gap 반환

작업 수가 설정 상한을 넘으면 HEFT로 안전하게 전환한다.

### 4.4 CP-SAT

구현: OR-Tools CP-SAT

- 작업별 integer start/end와 interval variable
- 선행관계 제약
- 자원별 cumulative capacity 제약
- makespan 최소화
- HEFT 결과를 solution hint로 제공
- 제한시간에 도달해도 실행 가능한 최선해 반환

### 4.5 Treewidth-guided DP

선행관계의 무방향 그래프에 deterministic min-fill elimination을 적용해
treewidth를 추정한다. 낮은 treewidth이고 모든 자원이 단일 용량인 작은
그래프에만 정확 DP를 실행한다.

DP는 완료 집합 전체의 종료시각을 저장하지 않는다. 아직 완료되지 않은 작업과
연결된 dependency frontier의 종료시각만 유지한다. 하위 그래프와 연결이 끝난
완료 작업은 상태에서 제거한다.

다중 용량 또는 기준 treewidth를 넘는 경우 HEFT로 전환한다. 일반 RCPSP 전체에
대한 treewidth FPT 구현이라고 과장하지 않으며, 현재 정확성 범위는
고정 자원 배정·단일 용량·비선점 작업이다.

### 4.6 Rolling-horizon 재계획

사용자가 조건을 추가하거나 API 실행시간이 예상과 달라지면 완료 작업은
고정하고 미완료 subgraph만 다시 컴파일한다.

```text
기존 실행 완료 작업
       ↓ 고정
실패·지연·추가 조건으로 영향받은 suffix
       ↓
포트폴리오 재최적화
```

이는 삭제 문제용 iterative compression을 억지로 적용하는 대신,
incremental scheduling과 schedule repair로 같은 목적을 달성한다.

## 5. Algorithm race와 자동 선택

`PortfolioScheduler`는 다음 알고리즘을 같은 wall-clock 구간에 병렬 실행한다.

```text
FIFO ──────────────┐
HEFT ──────────────┤
Branch-and-Bound ──┤
CP-SAT ────────────┼─ 검증 ─ 최소 makespan 선택
Treewidth DP ──────┘
```

각 결과는 공통 검증기로 작업 누락, 선행관계 위반, 실행시간 불일치, 자원
동시용량 초과를 확인한다. fallback으로 다른 알고리즘 결과를 그대로 반환한
후보는 실제 알고리즘 승자로 선택하지 않는다. makespan이 같으면 솔버 자체
계획시간이 짧은 결과를 선택한다.

## 6. 실제 서비스 통합

### 조건 검색

`src/server/property_search.py`

- Atom을 TaskGraph로 컴파일
- 포트폴리오 계획 생성
- 독립 AI SQL 조건을 읽기 전용 SQLite 연결로 병렬 실행
- 모든 SQL 교집합 이후에만 경로 API 실행
- `trace.condition_scheduler`에 선택 알고리즘과 후보 결과 저장

### OpenAI 동시 실행

`src/agent/llm.py`

- 프로세스 LLM 인스턴스에 `BoundedSemaphore(6)` 적용
- `last_trace`를 thread-local로 분리
- 병렬 요청 사이 감사 trace 오염 방지

### 상세 리포트

지역 뉴스 판정과 숫자 해설은 서로 독립적이므로 브라우저에서
`Promise.allSettled`로 병렬 실행한다. 상세 리포트의
`agent_execution_schedule`과 판단 감사 로그에 포트폴리오 선택 결과를 남긴다.

## 7. 벤치마크

실행:

```bash
python scripts/benchmark_agent_schedulers.py
```

결과는 `reports/agent_scheduler_benchmark.json`에 저장된다. 외부 API는 호출하지
않으며 고정된 실행시간 재생값으로 비교한다. CP-SAT에 3초를 준 결과를 작은
벤치마크의 reference makespan으로 사용했다.

2026-07-29 실행 결과:

| 알고리즘 | 유효/전체 | 평균 계획시간 | 평균 reference 차이 |
|---|---:|---:|---:|
| FIFO | 7/7 | 0.226ms | 4.3945% |
| Cost-aware HEFT | 7/7 | 0.410ms | 0.3901% |
| Branch-and-Bound | 7/7 | 48.970ms | 0.3486% |
| CP-SAT | 7/7 | 51.796ms | 0% |
| Treewidth DP/구조 fallback | 7/7 | 1.431ms | 0.3901% |
| Portfolio | 7/7 | 43.490ms | 0% |

포트폴리오는 7개 테스트 구조 모두에서 reference makespan과 일치했다. 이
수치는 실제 API SLA가 아니라 알고리즘 회귀검사용 오프라인 결과다. 운영
판단에는 `decision_run_id`별 실제 작업시간 분포를 계속 축적해 재평가해야 한다.

## 8. 설정값

| 환경변수 | 기본값 | 의미 |
|---|---:|---|
| `LLM_MAX_CONCURRENCY` | 6 | 프로세스당 OpenAI/LLM 동시 요청 |
| `AGENT_SCHEDULER_DEADLINE_MS` | 60 | 각 정밀 스케줄러 제한시간 |
| `CONDITION_SQL_MAX_WORKERS` | 2 | 독립 조건 SQL 병렬 읽기 |

OpenAI의 운영 계정 한도는 응답 헤더 기준 `500 RPM`, `200,000 TPM`이지만
동시 요청은 토큰 크기와 지연 변동을 고려해 6으로 제한한다.

## 9. 한계와 다음 평가

- 실행시간은 현재 작업 종류별 보수적 기본값이다.
- 실제 `decision_run_id` 이력으로 EWMA/P50/P95 추정치를 갱신해야 한다.
- 정적 계획은 60초 RPM/TPM admission window를 반영한다. 운영 중 다른
  프로세스·프로젝트 요청까지 반영하려면 응답 헤더 기반 분산 token bucket이 필요하다.
- CP-SAT의 현재 목적은 makespan이다. 선택 작업이 추가되면 비용·품질을
  epsilon constraint 또는 가중 목적함수로 확장해야 한다.
- 운영 배포 전 동시 사용자 부하 테스트로 P95와 429 복구율을 확인해야 한다.
