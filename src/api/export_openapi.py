"""Export OpenAPI spec ra docs/openapi.json + VERIFY khớp contract docs/tasco_api.pdf.

Chạy: make openapi  (python -m src.api.export_openapi)
Fail (exit 1) nếu thiếu path/param/field bắt buộc — đây là contract test, không chỉ export.
"""
from __future__ import annotations

import json
import sys

from src import config
from src.api.main import app

# Contract tối thiểu theo PDF (Search API + Common DTOs)
REQUIRED_PATHS = {"/v1/search", "/health"}
REQUIRED_ALIASES = {"/search", "/v1/geocode-search"}  # include_in_schema=False nhưng phải route được
REQUIRED_PARAMS = {"q", "lat", "lon", "radiusMeters", "bbox", "category", "limit", "lang"}
REQUIRED_PLACE_FIELDS = {"id", "type", "name", "label", "address", "category",
                         "coordinates", "distanceMeters", "score", "source", "tags"}


def main() -> None:
    spec = app.openapi()
    errors = []

    missing_paths = REQUIRED_PATHS - set(spec.get("paths", {}))
    if missing_paths:
        errors.append(f"thiếu path trong spec: {missing_paths}")

    app_routes = {r.path for r in app.routes if hasattr(r, "path")}
    missing_aliases = REQUIRED_ALIASES - app_routes
    if missing_aliases:
        errors.append(f"thiếu alias route: {missing_aliases}")

    search_params = {p["name"] for p in spec["paths"]["/v1/search"]["get"].get("parameters", [])}
    missing_params = REQUIRED_PARAMS - search_params
    if missing_params:
        errors.append(f"/v1/search thiếu param: {missing_params}")

    place = spec.get("components", {}).get("schemas", {}).get("PlaceResult", {})
    missing_fields = REQUIRED_PLACE_FIELDS - set(place.get("properties", {}))
    if missing_fields:
        errors.append(f"PlaceResult thiếu field: {missing_fields}")

    if errors:
        for e in errors:
            print(f"✗ CONTRACT MISMATCH: {e}")
        sys.exit(1)

    out = config.ROOT / "docs" / "openapi.json"
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ OpenAPI khớp contract PDF (paths/aliases/params/PlaceResult đủ field)")
    print(f"✓ Exported: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
