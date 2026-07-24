"""Downloader for the approved apartment detailed-trade RTMS endpoint.

It reuses the checkpoint/retry/page handling of the standard apartment
downloader but writes a separate immutable source file.
"""
from __future__ import annotations

from pathlib import Path

from scripts import download_rtms_apt_trade as base


base.ENDPOINT = (
    "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/"
    "getRTMSDataSvcAptTradeDev"
)
base.DEFAULT_OUT = (
    base.config.DATA_RAW / "real_estate" / "rtms_apt_trade_dev.csv"
    if hasattr(base, "config") else
    Path("data/downloaded/real_estate/rtms_apt_trade_dev.csv")
)
base.DEFAULT_META = Path("data/downloaded/real_estate/rtms_apt_trade_dev.meta.json")


if __name__ == "__main__":
    raise SystemExit(base.main())
