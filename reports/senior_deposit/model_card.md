# Model Card: Senior Deposit MVP v1

## 목적

수원시 다가구주택에서 신규 입주 대상 호실을 제외한 기존 점유 호실 보증금
총액과 보수적 상한의 조건부 확률분포를 추정한다.

## 현재 상태

- 건축물·임대차 데이터: actual
- 등록 호실 수 Model A: actual 학습 모델 및 관측값 우선
- 보증금 Model C: actual 학습 7분위수 모델
- 점유 Model B: 명시적 Beta-Binomial scenario prior
- 선순위 Model D: 실제 라벨 없음, classifier 미학습
- 최종 보정 Model E: 건물 총액 라벨 없음, 미학습
- 전체 `model_mode`: `scenario_only`

## 의도된 사용

- 공식 서류 확인 전 보수적 확인 우선순위 제시
- p50보다 conservative p90·p95를 중심으로 사전 위험 안내
- 점유율과 건물 내 상관 시나리오 민감도 비교

## 금지된 사용

- 법적으로 확정된 선순위 보증금으로 표시
- 전입일·확정일자·근저당 순위를 자동 확정
- 보증 가입·대출·계약 승인 또는 거절의 단독 근거
- 경매 평균을 일반 다가구 모집단에 직접 적용
- scenario probability를 실제 학습 확률로 표시

## 대표 경고

> 이 결과는 전입세대확인서, 확정일자 정보 또는 개별 임대차계약을 직접
> 확인한 법적 확정값이 아니다.
