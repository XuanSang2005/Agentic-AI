from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ResolvedArea:
    area: str | None = None
    city: str | None = None


class LocationResolver(Protocol):
    async def resolve(self, lat: float, lon: float) -> ResolvedArea | None: ...


class NoOpLocationResolver:
    async def resolve(self, lat: float, lon: float) -> None:
        return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(value))
