"""GATE Phase 4b (g, h, j, l) — unit, MOCK AWS hoàn toàn, không DB không mạng.

FakeGeo cũng được tests/test_ingestion.py import làm client inject cho endpoint.
"""
from __future__ import annotations

from types import SimpleNamespace

from src import config
from src.verify import geocode

_CFG = config.settings().verify


class FakeGeo:
    """Fake client geo-places. Mặc định HAPPY: echo BiasPosition làm Position,
    Overall 0.98, PointAddress. Tuỳ biến per-address: rules[QueryText] =
    dict(overall=…, position=…, place_type=…) hoặc một Exception instance."""

    def __init__(self, overall: float = 0.98, place_type: str = "PointAddress",
                 position: list | None = None):
        self.overall = overall
        self.place_type = place_type
        self.position = position
        self.rules: dict = {}
        self.calls: list[dict] = []

    def geocode(self, **kw):
        self.calls.append(kw)
        rule = self.rules.get(kw["QueryText"])
        if isinstance(rule, Exception):
            raise rule
        overall, ptype = self.overall, self.place_type
        pos = self.position or kw["BiasPosition"]
        if isinstance(rule, dict):
            overall = rule.get("overall", overall)
            ptype = rule.get("place_type", ptype)
            pos = rule.get("position", pos)
        return {"ResultItems": [{"Position": list(pos),
                                 "MatchScores": {"Overall": overall},
                                 "PlaceType": ptype}]}


class ThrottlingException(Exception):
    """_is_throttling nhận theo tên class — đúng shape lỗi throttle của AWS."""


class ThrottleThenOk(FakeGeo):
    def __init__(self, fail_times: int, **kw):
        super().__init__(**kw)
        self.remaining = fail_times

    def geocode(self, **kw):
        if self.remaining > 0:
            self.remaining -= 1
            raise ThrottlingException("Rate exceeded")
        return super().geocode(**kw)


_ADDR = "1 Nguyễn Huệ, Quận 1, TP.HCM"
_LAT, _LON = 10.774, 106.704


def test_g_high_score_matching_position_is_verified():
    r = geocode.geocode_verify(_ADDR, _LAT, _LON, client=FakeGeo())
    assert r["status"] == "verified" and r["reason"] is None
    assert r["overall_score"] == 0.98 and r["place_type"] == "PointAddress"
    assert r["matched_position"] == {"lat": _LAT, "lon": _LON}


def test_h_low_score_or_far_position_is_unverified_with_reason():
    # Score thấp
    r = geocode.geocode_verify(_ADDR, _LAT, _LON, client=FakeGeo(overall=0.5))
    assert r["status"] == "unverified" and "match score 0.50" in r["reason"]
    # Toạ độ matched lệch ~2.9km > 150m ([lng, lat]!)
    far = FakeGeo(position=[_LON, _LAT + 0.026])
    r = geocode.geocode_verify(_ADDR, _LAT, _LON, client=far)
    assert r["status"] == "unverified" and "away" in r["reason"]
    # PlaceType ngoài danh sách chấp nhận
    r = geocode.geocode_verify(_ADDR, _LAT, _LON, client=FakeGeo(place_type="Locality"))
    assert r["status"] == "unverified" and "place type" in r["reason"]
    # Không có địa chỉ → không gọi AWS, unverified rõ lý do
    r = geocode.geocode_verify("  ", _LAT, _LON, client=FakeGeo())
    assert r["status"] == "unverified" and r["reason"] == "no address to verify"


def test_j_throttling_retries_with_backoff_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(geocode.time, "sleep", sleeps.append)

    r = geocode.geocode_verify(_ADDR, _LAT, _LON, client=ThrottleThenOk(fail_times=2))
    assert r["status"] == "verified"
    assert sleeps == list(_CFG.throttle_retry_delays[:2])  # backoff 1s, 2s

    # Hết retry vẫn throttle → verify_batch flag unverified, KHÔNG nổ
    exhausted = ThrottleThenOk(fail_times=len(_CFG.throttle_retry_delays) + 1)
    out = geocode.verify_batch(
        [SimpleNamespace(address=_ADDR, lat=_LAT, lon=_LON)], client=exhausted)
    assert out[0]["status"] == "unverified"
    assert out[0]["reason"].startswith("verify failed:")


def test_l_bias_position_is_lon_lat_order():
    fake = FakeGeo()
    geocode.geocode_verify(_ADDR, _LAT, _LON, client=fake)
    call = fake.calls[0]
    assert call["BiasPosition"] == [_LON, _LAT], "AWS cần [lng, lat] — KHÔNG đảo"
    assert call["QueryText"] == _ADDR
    assert call["Filter"] == {"IncludeCountries": ["VNM"]}
    assert call["IntendedUse"] == "SingleUse"
