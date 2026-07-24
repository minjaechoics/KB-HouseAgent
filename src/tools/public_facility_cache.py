"""Official public-facility CSV cache used by the property report."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

DEFAULT_CACHE = config.DATA_RAW / "facilities" / "public_facilities.csv"


class PublicFacilityCache:
    """Read-only spatial lookup over normalized official public CSV rows."""

    def __init__(self, path: Path = DEFAULT_CACHE):
        self.path = Path(path)
        self._data: pd.DataFrame | None = None

    @property
    def available(self) -> bool:
        return self.path.exists()

    def load(self) -> pd.DataFrame | None:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            return None
        frame = pd.read_csv(self.path, encoding="utf-8-sig", low_memory=False)
        frame["lat"] = pd.to_numeric(frame.get("lat"), errors="coerce")
        frame["lng"] = pd.to_numeric(frame.get("lng"), errors="coerce")
        self._data = frame.dropna(subset=["lat", "lng", "category"]).copy()
        return self._data

    @staticmethod
    def _distances(frame: pd.DataFrame, lat: float, lng: float) -> np.ndarray:
        lat2 = np.radians(frame["lat"].to_numpy(dtype=float))
        lng2 = np.radians(frame["lng"].to_numpy(dtype=float))
        lat1, lng1 = math.radians(lat), math.radians(lng)
        dlat, dlng = lat2 - lat1, lng2 - lng1
        a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
        return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def nearby(self, category: str, lat: float, lng: float,
               radius_m: int, limit: int = 8) -> dict | None:
        data = self.load()
        if data is None:
            return None
        pool = data[data["category"].astype(str) == str(category)]
        if pool.empty:
            return {"count": 0, "places": [], "source": "official_public_csv"}
        dlat = radius_m / 111000.0
        dlng = radius_m / (111000.0 * math.cos(math.radians(lat)) + 1e-9)
        pool = pool[
            pool["lat"].between(lat - dlat, lat + dlat)
            & pool["lng"].between(lng - dlng, lng + dlng)
        ].copy()
        if pool.empty:
            return {"count": 0, "places": [], "source": "official_public_csv"}
        pool["distance_m"] = self._distances(pool, lat, lng)
        pool = pool[pool["distance_m"] <= radius_m].sort_values("distance_m")
        fields = ["name", "address", "lat", "lng", "subcategory", "source_url", "distance_m"]
        places = []
        for row in pool.head(limit).to_dict("records"):
            item = {key: row.get(key) for key in fields if key in row}
            item["distance_m"] = round(float(item.get("distance_m") or 0), 1)
            places.append(item)
        return {"count": int(len(pool)), "places": places,
                "source": "official_public_csv"}
