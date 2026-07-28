from pathlib import Path
import sqlite3

from scripts.import_paldal_pdf_listings import _parse_page, record_to_row
from src import config


def test_listing_row_uses_a_real_reference_address_without_claiming_it_is_exact():
    words = [
        {"x": 52.0, "y": 125.0, "text": "다가구"},
        {"x": 52.0, "y": 138.0, "text": "월세"},
        {"x": 126.0, "y": 130.0, "text": "22"},
        {"x": 147.0, "y": 114.0, "text": "인계동"},
        {"x": 315.0, "y": 132.0, "text": "2/4"},
        {"x": 347.0, "y": 103.0, "text": "방2, 화장실1, 풀옵션"},
        {"x": 624.0, "y": 147.0, "text": "500"},
        {"x": 690.0, "y": 114.0, "text": "45"},
        {"x": 709.0, "y": 105.0, "text": "팔달공인중개사"},
        {"x": 710.0, "y": 120.0, "text": "031-000-0000"},
        {"x": 709.0, "y": 144.0, "text": "경기도 수원시 팔달구 중부대로 1"},
    ]
    parsed = _parse_page(words, "sample.pdf", 1)
    assert len(parsed) == 1
    row = record_to_row(parsed[0])
    assert row["sido"] == "경기"
    assert row["gugun"] == "수원시 팔달구"
    assert row["dong"] == "인계동"
    assert row["transaction_type"] == "월세"
    assert row["deposit_manwon"] == 500
    assert row["monthly_rent_manwon"] == 45
    assert row["room_count"] == 2
    assert row["bathroom_count"] == 1
    assert row["is_synthetic"] == 0
    assert row["road_address"].startswith("경기도 수원시 팔달구")
    assert "실제 매물 상세주소가 아닙니다" in row["address_detail_public"]
    assert "실제 매물 좌표 아님" in row["coordinate_distribution_method"]
    assert row["fraud_score"] is None


def test_server_and_gui_lock_the_paldal_prototype_region():
    root = Path(__file__).parents[1]
    app = (root / "src/server/app.py").read_text(encoding="utf-8")
    gui = (root / "src/server/gui.html").read_text(encoding="utf-8")
    assert 'PROTOTYPE_SIDO = "경기"' in app
    assert 'PROTOTYPE_GUGUN = "수원시 팔달구"' in app
    assert "updates = _prototype_profile" in app
    assert "const PROTOTYPE_REGION={sido:'경기',gugun:'수원시 팔달구'" in gui
    assert 'id="sido" disabled' in gui
    assert 'id="editGugun" disabled' in gui
    assert "zoom:13" in gui


def test_paldal_jeonse_rows_are_all_scored_and_use_reference_addresses():
    with sqlite3.connect(config.DB_PATH) as connection:
        total, scored = connection.execute(
            "SELECT COUNT(*), COUNT(fraud_score) FROM properties "
            "WHERE transaction_type='전세'"
        ).fetchone()
        distinct_addresses = connection.execute(
            "SELECT COUNT(DISTINCT road_address) FROM properties"
        ).fetchone()[0]
    assert total == 89
    assert scored == total
    assert distinct_addresses >= 100


def test_gui_removes_old_pdf_wording_and_uses_js_hedgehog_loader():
    root = Path(__file__).parents[1]
    gui = (root / "src/server/gui.html").read_text(encoding="utf-8")
    assert "사용자 제공 PDF" not in gui
    assert "전세 보증사고 위험도 분석 대상 아님" not in gui
    assert "hedgehog-stage" in gui
    assert "requestAnimationFrame(frame)" in gui
    assert "둥근 안경을 쓴 고슴도치" in gui
    assert "/assets/youth-home-loader-sprite-v1.png" not in gui
    assert (root / "src/server/assets/youth-home-loader-sprite-v1.png").stat().st_size > 100_000
