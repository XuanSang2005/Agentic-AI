from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tasco_query.normalization import match_key

EXPECTED_COLUMNS = {
    "abbreviation.csv": ("term", "normalized_form", "type"),
    "address.csv": (
        "address_id",
        "full_address",
        "house_number",
        "street",
        "ward",
        "district",
        "city",
        "latitude",
        "longitude",
        "aliases",
        "notes",
    ),
    "poi.csv": (
        "poi_id",
        "name_vi",
        "name_en",
        "category",
        "brand",
        "address",
        "district",
        "city",
        "latitude",
        "longitude",
        "aliases",
        "opening_hours",
        "rating",
    ),
    "eval.csv": (
        "query_id",
        "input_query",
        "expected_normalized_query",
        "expected_intent",
        "expected_entities_json",
        "difficulty",
        "skills_tested",
    ),
}


@dataclass(frozen=True, slots=True)
class AbbreviationRow:
    term: str
    normalized_form: str
    type: str


@dataclass(frozen=True, slots=True)
class AddressRow:
    address_id: str
    full_address: str
    house_number: str
    street: str
    ward: str | None
    district: str
    city: str
    latitude: float
    longitude: float
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoiRow:
    poi_id: str
    name_vi: str
    name_en: str | None
    category: str
    brand: str | None
    address: str
    district: str
    city: str
    latitude: float
    longitude: float
    aliases: tuple[str, ...]
    opening_hours: str
    rating: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = EXPECTED_COLUMNS[path.name]
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path.name}: expected columns {expected}, got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: file is empty")
    if any(None in row for row in rows):
        raise ValueError(f"{path.name}: malformed row has extra columns")
    return rows


def _float(value: str, field: str, row_id: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{row_id}: invalid {field}: {value!r}") from exc
    if not low <= number <= high:
        raise ValueError(f"{row_id}: {field} outside [{low}, {high}]")
    return number


class DataCatalog:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        abbreviation_rows = read_csv(data_dir / "abbreviation.csv")
        address_rows = read_csv(data_dir / "address.csv")
        poi_rows = read_csv(data_dir / "poi.csv")
        self.abbreviations = tuple(AbbreviationRow(**row) for row in abbreviation_rows)
        self.addresses = tuple(self._address(row) for row in address_rows)
        self.pois = tuple(self._poi(row) for row in poi_rows)
        self.abbreviation_index = {
            match_key(row.term): row for row in self.abbreviations if match_key(row.term)
        }
        self.poi_alias_index = self._poi_aliases()
        self.street_index = self._street_index()

    @staticmethod
    def _address(row: dict[str, str]) -> AddressRow:
        row_id = row["address_id"]
        return AddressRow(
            address_id=row_id,
            full_address=row["full_address"],
            house_number=row["house_number"],
            street=row["street"],
            ward=row["ward"] or None,
            district=row["district"],
            city=row["city"],
            latitude=_float(row["latitude"], "latitude", row_id, -90, 90),
            longitude=_float(row["longitude"], "longitude", row_id, -180, 180),
            aliases=tuple(part.strip() for part in row["aliases"].split("|") if part.strip()),
        )

    @staticmethod
    def _poi(row: dict[str, str]) -> PoiRow:
        row_id = row["poi_id"]
        return PoiRow(
            poi_id=row_id,
            name_vi=row["name_vi"],
            name_en=row["name_en"] or None,
            category=row["category"],
            brand=row["brand"] or None,
            address=row["address"],
            district=row["district"],
            city=row["city"],
            latitude=_float(row["latitude"], "latitude", row_id, -90, 90),
            longitude=_float(row["longitude"], "longitude", row_id, -180, 180),
            aliases=tuple(part.strip() for part in row["aliases"].split(",") if part.strip()),
            opening_hours=row["opening_hours"],
            rating=_float(row["rating"], "rating", row_id, 0, 5),
        )

    def _poi_aliases(self) -> dict[str, tuple[PoiRow, ...]]:
        aliases: defaultdict[str, list[PoiRow]] = defaultdict(list)
        for poi in self.pois:
            values: Iterable[str | None] = (poi.name_vi, poi.name_en, *poi.aliases)
            for value in values:
                if value and poi not in aliases[match_key(value)]:
                    aliases[match_key(value)].append(poi)
        return {key: tuple(value) for key, value in aliases.items()}

    def _street_index(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for address in self.addresses:
            result.setdefault(match_key(address.street), address.street)
        for row in self.abbreviations:
            if row.type == "street abbreviation":
                result[match_key(row.term)] = row.normalized_form
        return result


def audit_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    rows = read_csv(path)
    columns = list(rows[0])
    tuples = [tuple(row[column] for column in columns) for row in rows]
    empty = {column: sum(not row[column].strip() for row in rows) for column in columns}
    duplicate_rows = sum(count - 1 for count in Counter(tuples).values() if count > 1)
    id_column = next((column for column in columns if column.endswith("_id")), None)
    duplicate_ids = (
        sorted(
            value for value, count in Counter(row[id_column] for row in rows).items() if count > 1
        )
        if id_column
        else []
    )
    malformed_json: list[int] = []
    if path.name == "eval.csv":
        for line, row in enumerate(rows, 2):
            try:
                json.loads(row["expected_entities_json"])
            except json.JSONDecodeError:
                malformed_json.append(line)
    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "utf-8",
        "bom": raw.startswith(b"\xef\xbb\xbf"),
        "delimiter": ",",
        "line_ending": "CRLF" if raw.count(b"\r\n") == raw.count(b"\n") else "mixed_or_lf",
        "rows": len(rows),
        "columns": columns,
        "empty_values": empty,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "malformed_json_lines": malformed_json,
    }


def audit_all(data_dir: Path) -> dict[str, Any]:
    return {
        "files": [audit_file(data_dir / name) for name in EXPECTED_COLUMNS],
        "assumptions": [
            "Empty cells represent missing optional values.",
            "Shared aliases are not unique branch identifiers.",
            "Evaluation expectations are not loaded by the runtime service.",
        ],
    }
