"""PlaceResult DTO (đúng contract docs/tasco_api.pdf) + mapping từ POI nội bộ.

Mapping cho Flutter app (SearchSuggestion): id→id, label|name→label,
category|type→meta, address→description, coordinates.lat/lon→coordinates.
GIỮ NGUYÊN dấu tiếng Việt trong mọi field trả ra (compatibility requirement của PDF).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.search import SearchHit


class Coordinates(BaseModel):
    lat: float
    lon: float


class PlaceResult(BaseModel):
    """DTO đúng shape mục "Common DTOs" trong docs/tasco_api.pdf."""
    id: str = Field(examples=["poi:C001"])
    type: str = "poi"
    name: str
    label: str
    address: str
    category: str
    coordinates: Coordinates
    distanceMeters: Optional[int] = None      # chỉ set khi request có lat/lon
    score: float
    source: str = "mock"                   # data synthetic của hackathon
    tags: list[str] = []
    explanation: Optional[dict] = None        # chỉ set khi ?explain=true


class SearchMeta(BaseModel):
    limit: int
    lang: str


class SearchResponse(BaseModel):
    query: str
    results: list[PlaceResult]
    meta: SearchMeta


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    requestId: str


def to_place_result(hit: SearchHit) -> PlaceResult:
    """POI nội bộ → PlaceResult. ID stable = 'poi:<poi_id trong dataset>'."""
    poi = hit.poi
    return PlaceResult(
        id=f"poi:{poi.id}",
        type="poi",
        name=poi.name,
        label=poi.name,
        # address trong data đã có district ("27 Ngô Đức Kế, Quận 1") — nối thêm city,
        # trừ khi address đã kết thúc bằng city sẵn (tránh "Đà Lạt, Đà Lạt")
        address=(poi.address if not poi.address
                 else poi.address if poi.address.endswith(poi.city)
                 else f"{poi.address}, {poi.city}") or poi.city,
        category=poi.category,
        coordinates=Coordinates(lat=poi.lat, lon=poi.lon),
        distanceMeters=hit.distance_meters,
        score=hit.score,
        source="mock",
        tags=poi.tags,
        explanation=hit.explanation,
    )
