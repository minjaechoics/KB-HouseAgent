# 실행 방법

```bash
python -m src.cli inspect-input-models
python -m src.cli calculate --property-id PALDAL-PDF-07496ECCBECBA7AF
python -m src.cli sensitivity --property-id PALDAL-PDF-07496ECCBECBA7AF
python -m src.cli stress-test --property-id PALDAL-PDF-07496ECCBECBA7AF
```

테스트:

```bash
python -m pytest -q tests/test_jeonse_ratio.py
python -m pytest -q
```

UI `계약/안전` 탭에는 계약 후 전세가율, 꼬리위험·가격 스트레스,
반환보증 선택 비교, 최우선변제 사전검토, 임대인 총자산 보조지표,
기존 임차보증금 분포 순으로 표시한다.
