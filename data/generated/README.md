# 생성 데이터

합성 매물, 사용자, 클릭 라벨, SQLite DB와 실험 산출물이 생성되는 디렉터리입니다.
모든 파일은 재생성 가능하며 GitHub 용량 제한 때문에 Git에 포함하지 않습니다.

기본 생성 명령:

```bash
python run_pipeline.py --n_properties 5000 --n_users 2000
```

서비스가 사용하는 기본 DB 경로는 data/generated/jeonse_helper.db 입니다.
