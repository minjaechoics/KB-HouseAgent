# AWS Lightsail 배포 운영 런북

작성일: 2026-07-17

## 배포 현황

- 리전: 서울 `ap-northeast-2`
- 인스턴스: `jeonse-helper-prod`
- 플랜: `medium_3_0` — 2 vCPU, RAM 4GB, SSD 80GB
- 고정 IP 리소스: `jeonse-helper-prod-ip`
- 웹 주소: `http://52.79.138.208/`
- GUI 주소: `http://52.79.138.208/gui`
- 서버 앱 위치: `/opt/jeonse-helper`
- AWS CLI 프로필: `jeonse-deploy`

웹 로그인 비밀번호는 이 문서에 기록하지 않는다. 로컬의 `deploy/PRIVATE_ACCESS.txt`에만 있으며 해당 파일은 Git과 Docker context에서 제외된다.

## 현재 구조

```text
브라우저
  └─ HTTP :80
      └─ Nginx Basic Auth + 분당 요청 제한
          └─ FastAPI :8000 (외부 비공개)
              ├─ OpenAI Agentic LLM
              ├─ 부동산/금융 SQLite DB
              └─ 안전 데이터와 위험도 모델
```

Docker Compose 서비스는 `app`, `nginx` 두 개다. `app`은 인메모리 세션을 사용하므로 Redis 전환 전까지 worker를 1개로 유지한다.

## 보안 상태

- 외부 공개 포트: HTTP 80
- SSH 22: 배포 당시 개발 PC 공인 IP `/32`에서만 허용
- FastAPI 8000: Docker 내부 네트워크에서만 사용
- OpenAI 키: 컨테이너 이미지에 없고 서버 `.env.production`에만 존재
- `.env.production` 권한: `600`
- 무인증 웹 요청: `401 Unauthorized`
- RAG 디버그: 로그인 사용자에게 표시

도메인이 없어 현재 연결은 HTTP다. 실제 개인정보를 입력하거나 일반 사용자에게 공개하면 안 된다. 도메인을 연결한 다음 HTTPS 인증서를 적용하고 웹 비밀번호와 과거 노출된 OpenAI 키를 교체해야 한다.

## SSH 접속

개발 PC의 CMD 또는 PowerShell에서:

```cmd
ssh -i C:\Users\user\.ssh\jeonse_lightsail ubuntu@52.79.138.208
```

인터넷 회선이 바뀌어 공인 IP가 변경되면 Lightsail 네트워킹 방화벽의 SSH 허용 CIDR도 새 IP `/32`로 바꿔야 한다.

## 상태 확인

```bash
cd /opt/jeonse-helper
sudo docker compose ps
sudo docker compose logs --tail=100 app
sudo docker compose logs --tail=100 nginx
curl http://127.0.0.1:8000/health
```

FastAPI 포트는 호스트에 공개하지 않았으므로 마지막 `curl`은 컨테이너 안에서 실행한다.

```bash
sudo docker compose exec app curl http://127.0.0.1:8000/health
```

## 재시작

```bash
cd /opt/jeonse-helper
sudo docker compose restart
```

`restart: unless-stopped`와 Docker systemd 자동 시작이 설정되어 있어 서버 재부팅 후 컨테이너가 자동으로 다시 시작된다.

## 소스 업데이트

로컬 변경사항을 서버 `/opt/jeonse-helper`에 복사한 후:

```bash
cd /opt/jeonse-helper
sudo docker compose up -d --build app
```

Nginx 설정을 변경했으면 다음을 실행한다.

```bash
cd /opt/jeonse-helper
sudo docker compose restart nginx
```

## OpenAI 키 교체

서버에서 `/opt/jeonse-helper/.env.production`의 `OPENAI_API_KEY`만 새 값으로 바꾸고 다음을 실행한다.

```bash
cd /opt/jeonse-helper
chmod 600 .env.production
sudo docker compose up -d --force-recreate app
```

키를 터미널 히스토리에 직접 남기지 말고 안전한 편집기 또는 숨김 입력 방식을 사용한다.

## TMAP 대중교통 appKey 설정

1. SK Open API의 `대시보드 > 앱`에서 배포에 사용할 앱을 선택한다.
2. 해당 앱에 `TMAP 대중교통` 상품을 사용 신청하고 앱 상세의 `appKey`를 확인한다.
3. 서버의 `/opt/jeonse-helper/deploy/TMAP_KEYS.private.env`에 다음 형식으로 저장한다.

```dotenv
TMAP_APP_KEY=발급받은_appKey
```

적용한다.

```bash
cd /opt/jeonse-helper
chmod 600 deploy/TMAP_KEYS.private.env
sudo docker compose up -d --force-recreate app
curl -s http://127.0.0.1:8000/health
```

정상 연결 준비 상태이면 health 응답의 `map.transit`이 `tmap_transit`이다. 실제 API
응답 성공 여부는 대중교통 조건 검색 후 RAG DEBUG의 `provider_source_counts`에서
`tmap_transit` 호출 건수를 확인한다.

Premium 종량제에서는 서버 `.env.production`에
`TMAP_TRANSIT_EXACT_CANDIDATE_LIMIT=0`을 두며, 0은 검색당 후보 수 상한이 없다는
뜻이다. NAVER 자동차 경로도 `NAVER_DIRECTIONS_EXACT_CANDIDATE_LIMIT=0`으로
후보 수 상한을 두지 않는다. 공급자 한도 초과(HTTP 429)는 자동 재시도 후
`estimated_haversine_transit`으로 명시적으로 fallback된다.

## 비용 중단과 삭제

Lightsail 인스턴스는 중지 상태만으로 과금이 완전히 중단된다고 가정하면 안 된다. 서비스를 폐기할 때는 필요한 데이터와 스냅샷을 먼저 확인한 뒤 인스턴스와 고정 IP를 삭제한다.

아래 명령은 **서비스를 영구 삭제할 때만** 사용한다. 현재 실행하지 않았다.

```cmd
aws lightsail delete-instance --instance-name jeonse-helper-prod --region ap-northeast-2 --profile jeonse-deploy
aws lightsail release-static-ip --static-ip-name jeonse-helper-prod-ip --region ap-northeast-2 --profile jeonse-deploy
aws lightsail delete-key-pair --key-pair-name jeonse-helper-deploy --region ap-northeast-2 --profile jeonse-deploy
```

## 확인된 통합 테스트

- 무인증 접근 401
- 인증 후 health 200
- 루트 `/`에서 `/gui`로 이동
- `APILLM`, `gpt-4.1-mini`, agentic mode 활성화
- 금융서비스 전체 질문: DB 6건, LLM Text-to-SQL fallback 없음
- 대전 유성구 전세 추천: 확인 단계 후 매물 3건, LLM Text-to-SQL fallback 없음
- 사용자용 AI 답변 초록색, 개발자용 RAG trace 주황색 표시
