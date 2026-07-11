"""GATE Phase 3: list[POI] từ Postgres phải BẰNG HỆT bản xlsx — từng POI, từng field.

Đây là bài test cho abstraction data_loader: cùng số lượng, cùng THỨ TỰ (corpus
order quyết định hash embedding cache), mỗi field bằng nhau kể cả derived
(document/norm_document/is_synthetic), kiểu số đúng float/int Python (không
Decimal). Skip khi không có Postgres đã seed (CI/máy khác vẫn pass make test).
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from src import config
from src.data_loader import _pois_from_xlsx


def _pg_pois_or_skip():
    try:
        import psycopg
        from src.data_loader import _pois_from_postgres
        pois = _pois_from_postgres()
    except ModuleNotFoundError:
        pytest.skip("psycopg chưa cài")
    except Exception as e:  # DB chưa chạy / chưa seed
        pytest.skip(f"Postgres không sẵn sàng: {e}")
    if not pois:
        pytest.skip("bảng pois rỗng — chạy scripts/seed_postgres.py trước")
    return pois


def test_postgres_equals_xlsx_exactly():
    xlsx = _pois_from_xlsx()
    pg = _pg_pois_or_skip()

    assert len(pg) == len(xlsx), f"số lượng lệch: pg={len(pg)} xlsx={len(xlsx)}"
    assert [p.id for p in pg] == [p.id for p in xlsx], "THỨ TỰ lệch — corpus order phải khớp xlsx"

    diffs = []
    for a, b in zip(xlsx, pg):
        for f in dataclasses.fields(a):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if va != vb or type(va) is not type(vb):
                diffs.append(f"{a.id}.{f.name}: xlsx={va!r}({type(va).__name__}) "
                             f"pg={vb!r}({type(vb).__name__})")
    assert not diffs, "field lệch:\n" + "\n".join(diffs[:20])


def test_postgres_corpus_hash_unchanged():
    """Hash corpus (đúng công thức cache .npy trong dense.py) phải y hệt —
    cache embedding cũ dùng lại được, không re-encode."""
    pg = _pg_pois_or_skip()
    xlsx = _pois_from_xlsx()

    def digest(pois):
        docs = [p.document for p in pois]
        return hashlib.sha256(
            (config.EMBEDDING_MODEL + "\n\x00".join(docs)).encode("utf-8")).hexdigest()[:16]

    assert digest(pg) == digest(xlsx)
