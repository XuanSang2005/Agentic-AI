from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.tasco_query.contracts import (
    Evidence,
    QueryVariant,
    SourceSpan,
    TraceEvidenceType,
    TraceExtractorName,
    TraceGrounding,
    TraceOperation,
    TraceOperationType,
    TraceStage,
    TraceStageName,
    TraceValue,
)
from src.tasco_query.data import DataCatalog, PoiRow
from src.tasco_query.lexicon import LexiconEntry, LexiconMatch, LexiconRegistry
from src.tasco_query.normalization import match_key
from src.tasco_query.rewriting import COORDINATE_RE

if TYPE_CHECKING:
    from src.tasco_query.tracing import TraceCollector


@dataclass(slots=True)
class Extraction:
    entities: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    navigation: bool = False
    nearby: bool = False
    discovery: bool = False
    coordinate_only: bool = False
    explicit_reference: bool = False
    covered_spans: dict[str, set[tuple[int, int]]] = field(default_factory=dict)


def _trace_type(kind: str) -> TraceEvidenceType:
    if kind.startswith("open_"):
        return TraceEvidenceType.OPENING_CONSTRAINT
    cue_type = {
        "navigation": TraceEvidenceType.NAVIGATION_CUE,
        "nearby": TraceEvidenceType.NEARBY_CUE,
        "discovery": TraceEvidenceType.DISCOVERY_CUE,
    }.get(kind)
    return cue_type or TraceEvidenceType(kind)


class EvidenceCollector:
    def __init__(self, trace: TraceCollector | None) -> None:
        self.trace = trace
        self.items: list[Evidence] = []
        self.covered_spans: dict[str, set[tuple[int, int]]] = {}

    def add(
        self,
        *,
        kind: str,
        value: str | int | float | bool | list[str],
        variant: QueryVariant,
        start: int | None,
        end: int | None,
        raw_value: TraceValue,
        rule_id: str,
        config_source: str,
        confidence: float,
        precedence: int,
        accepted: bool = True,
        rejection_reason: str | None = None,
        merge_decision: str = "accepted",
        extractor: TraceExtractorName = TraceExtractorName.LEXICON_REGISTRY,
    ) -> Evidence:
        span = (
            SourceSpan(start=start, end=end, text=variant.text[start:end])
            if start is not None and end is not None
            else SourceSpan(start=0, end=len(variant.text), text=variant.text)
        )
        if start is not None and end is not None:
            self.covered_spans.setdefault(variant.variant_id, set()).add((start, end))
        evidence_id = f"ev_{len(self.items) + 1}"
        generator_id = variant.generation[0].transformation_type if variant.generation else None
        if self.trace is not None:
            evidence_id = (
                self.trace.add_evidence(
                    evidence_type=_trace_type(kind),
                    raw_value=raw_value,
                    canonical_value=value,
                    field=kind,
                    source_variant_id=variant.variant_id,
                    start=start,
                    end=end,
                    extractor=extractor,
                    rule_id=rule_id,
                    confidence=confidence,
                    configuration_source=config_source,
                    generator_id=generator_id,
                    precedence=precedence,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    canonical_merge_decision=merge_decision,
                )
                or evidence_id
            )
        item = Evidence(
            evidence_id=evidence_id,
            kind=kind,
            value=value,
            span=span,
            method=extractor.value,
            confidence=confidence,
            variant_id=variant.variant_id,
            rule_id=rule_id,
            config_source=config_source,
            precedence=precedence,
            accepted=accepted,
            rejection_reason=rejection_reason,
            merge_decision=merge_decision,
        )
        self.items.append(item)
        return item


def _precedence(variant: QueryVariant, entry: LexiconEntry) -> int:
    if variant.variant_id == "v0" and entry.source.startswith("data/raw/"):
        return 2
    if variant.variant_id == "v0":
        return 3
    if variant.source in {"edit_distance", "accent_restoration"}:
        return 5
    return 4


def _poi_is_unambiguous(registry: LexiconRegistry, source: str) -> bool:
    values = {
        str(entry.canonical)
        for entry in registry.accent_lookup(source)
        if entry.canonical_field == "poi_name"
    }
    return len(values) == 1


def _matched_entries(
    variants: list[QueryVariant], registry: LexiconRegistry
) -> list[tuple[QueryVariant, LexiconMatch]]:
    result = [
        (variant, match)
        for variant in variants
        for match in registry.phrase_matches(variant.text)
        if match.entry.canonical_field != "rewrite"
    ]
    return sorted(
        result,
        key=lambda item: (
            _precedence(item[0], item[1].entry),
            -(item[1].end - item[1].start),
            -item[1].entry.priority,
        ),
    )


def _merge_entity(entities: dict[str, Any], field: str, value: Any) -> tuple[bool, str | None, str]:
    if isinstance(value, list):
        target = entities.setdefault(field, [])
        if not isinstance(target, list):
            return False, "scalar_list_conflict", "rejected_weaker_conflict"
        added = [item for item in value if item not in target]
        target.extend(added)
        return (bool(added), None if added else "duplicate_canonical_value", "merged_list")
    if field not in entities:
        entities[field] = value
        return True, None, "selected_strongest"
    if entities[field] == value:
        return False, "duplicate_canonical_value", "retained_existing"
    return False, "weaker_conflicting_value", "retained_stronger_precedence"


def _catalog_poi(catalog: DataCatalog, name: str) -> PoiRow | None:
    return next((row for row in catalog.pois if row.name_vi == name), None)


def _numeric_district(text: str) -> tuple[str, int, int] | None:
    found = re.search(r"(?:^|\s)(?:q|quận|quan)\s*(\d{1,2})(?:\s|$)", text, re.I)
    if not found:
        return None
    return f"Quận {int(found.group(1))}", found.start(1), found.end(1)


def _price(text_key: str) -> int | None:
    found = re.search(r"(?:dưới\s+)?(\d+(?:[.,]\d+)?)\s*(k|nghin|triệu|d|vnd)\b", text_key)
    if not found:
        return None
    number = float(found.group(1).replace(",", "."))
    multiplier = (
        1_000_000 if found.group(2) == "trieu" else 1_000 if found.group(2) in {"k", "nghin"} else 1
    )
    return int(number * multiplier)


def _time_entities(text_key: str) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    after = re.search(r"(?:sau)\s+(\d{1,2})(?:h|\s*h)(?:\s*(toi|dem|khuya))?", text_key)
    if after:
        hour = int(after.group(1))
        if after.group(2) and hour < 12:
            hour += 12
        result["open_after"] = (f"{hour % 24:02d}:00", after.group(0))
    until = re.search(r"(?:toi|den)\s+(\d{1,2})(?:h)?\s*(dem|khuya|toi)", text_key)
    if until:
        hour = int(until.group(1))
        if until.group(2) in {"dem", "khuya"} and hour == 12:
            hour = 0
        elif until.group(2) == "toi" and hour < 12:
            hour += 12
        result["open_until"] = (f"{hour:02d}:00", until.group(0))
    return result


def extract_entities(
    original: str,
    variants: list[QueryVariant],
    catalog: DataCatalog,
    trace: TraceCollector | None = None,
    registry: LexiconRegistry | None = None,
) -> Extraction:
    if registry is None:
        from src.tasco_query.config import get_settings

        registry = LexiconRegistry.load(get_settings().lexicon_dir, catalog)
    if trace is not None:
        trace.mark_variants_selected({variant.variant_id for variant in variants})
    collector = EvidenceCollector(trace)
    trace_evidence_start = len(trace.evidence) if trace is not None else 0
    result = Extraction()
    entities = result.entities
    matches = _matched_entries(variants, registry)
    result.navigation = any(match.entry.canonical_field == "navigation" for _, match in matches)
    result.nearby = any(match.entry.canonical_field == "nearby" for _, match in matches)
    result.discovery = any(match.entry.canonical_field == "discovery" for _, match in matches)
    has_explicit_target = any(
        match.entry.canonical_field in {"category", "dish", "brand"} for _, match in matches
    )

    coordinate = COORDINATE_RE.search(original)
    if coordinate:
        latitude, longitude = float(coordinate.group(1)), float(coordinate.group(2))
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            entities.update(latitude=latitude, longitude=longitude)
            for field_name, value, group in (
                ("latitude", latitude, 1),
                ("longitude", longitude, 2),
            ):
                collector.add(
                    kind=field_name,
                    value=value,
                    variant=variants[0],
                    start=coordinate.start(group),
                    end=coordinate.end(group),
                    raw_value=coordinate.group(group),
                    rule_id=f"parser.coordinate.{field_name}",
                    config_source="python_parser",
                    confidence=1.0,
                    precedence=1,
                    extractor=TraceExtractorName.COORDINATE_PARSER,
                )
            result.coordinate_only = not COORDINATE_RE.sub("", original).strip(" ,")

    for variant, match in matches:
        entry = match.entry
        raw = variant.text[match.start : match.end]
        field = entry.canonical_field
        if field in {"navigation", "nearby", "discovery"}:
            collector.add(
                kind=field,
                value=str(entry.canonical),
                variant=variant,
                start=match.start,
                end=match.end,
                raw_value=raw,
                rule_id=entry.rule_id,
                config_source=entry.source,
                confidence=entry.confidence,
                precedence=_precedence(variant, entry),
            )
            continue
        if field == "poi_name" and not _poi_is_unambiguous(registry, raw):
            collector.add(
                kind=field,
                value=str(entry.canonical),
                variant=variant,
                start=match.start,
                end=match.end,
                raw_value=raw,
                rule_id=entry.rule_id,
                config_source=entry.source,
                confidence=entry.confidence,
                precedence=_precedence(variant, entry),
                accepted=False,
                rejection_reason="ambiguous_authoritative_alias",
                merge_decision="rejected_ambiguous_alias",
            )
            continue
        target_field = field
        if field == "poi_name" and result.navigation:
            target_field = "destination_poi"
        elif field == "poi_name" and result.nearby and has_explicit_target:
            target_field = "reference_poi"
            result.explicit_reference = True
        accepted, rejection, merge = _merge_entity(entities, target_field, entry.canonical)
        collector.add(
            kind=target_field,
            value=entry.canonical,
            variant=variant,
            start=match.start,
            end=match.end,
            raw_value=raw,
            rule_id=entry.rule_id,
            config_source=entry.source,
            confidence=entry.confidence,
            precedence=_precedence(variant, entry),
            accepted=accepted,
            rejection_reason=rejection,
            merge_decision=merge,
        )
        if accepted and target_field in {"poi_name", "destination_poi", "reference_poi"}:
            poi = _catalog_poi(catalog, str(entry.canonical))
            if poi and target_field != "reference_poi":
                entities.setdefault("category", poi.category.split(",")[0].strip())
                if poi.brand:
                    entities.setdefault("brand", poi.brand)

    district = _numeric_district(match_key(original))
    if district and "district" not in entities:
        entities["district"] = district[0]
        collector.add(
            kind="district",
            value=district[0],
            variant=variants[0],
            start=district[1],
            end=district[2],
            raw_value=original[district[1] : district[2]],
            rule_id="parser.district_number",
            config_source="python_parser",
            confidence=1.0,
            precedence=1,
            extractor=TraceExtractorName.ADMINISTRATIVE_ALIAS,
        )

    text_key = " ".join(match_key(variant.text) for variant in variants)
    street_present = "street" in entities
    house = re.search(r"(?:^|\s)(\d+[a-z]?)\s+[a-z]", match_key(original))
    if house and street_present and not coordinate:
        entities["house_number"] = house.group(1).upper()
        collector.add(
            kind="house_number",
            value=entities["house_number"],
            variant=variants[0],
            start=house.start(1),
            end=house.end(1),
            raw_value=house.group(1),
            rule_id="parser.house_number",
            config_source="python_parser",
            confidence=1.0,
            precedence=1,
            extractor=TraceExtractorName.ADDRESS_PARSER,
        )
    price = _price(text_key)
    if price is not None:
        entities["price_max"] = price
        collector.add(
            kind="price_max",
            value=price,
            variant=variants[0],
            start=None,
            end=None,
            raw_value=price,
            rule_id="parser.price_max",
            config_source="python_parser",
            confidence=1.0,
            precedence=1,
            extractor=TraceExtractorName.PRICE_PARSER,
        )
    for variant in variants:
        for key, (time_value, raw) in _time_entities(match_key(variant.text)).items():
            if key in entities:
                continue
            entities[key] = time_value
            collector.add(
                kind=key,
                value=time_value,
                variant=variant,
                start=None,
                end=None,
                raw_value=raw,
                rule_id=f"parser.time.{key}",
                config_source="python_parser",
                confidence=1.0,
                precedence=1,
                extractor=TraceExtractorName.TIME_PARSER,
            )

    if result.nearby:
        if "reference_poi" not in entities and "reference_area" not in entities:
            entities["location"] = "current_location"
        else:
            result.explicit_reference = True
    if result.navigation:
        entities["action"] = "directions"
        origin = re.search(r"(?:tu|from)\s+((?:q|quan)\s*\d{1,2})", text_key)
        if origin:
            number = re.search(r"\d+", origin.group(1))
            if number:
                entities["origin"] = f"Quận {int(number.group())}"
                entities.pop("district", None)
        if "destination_poi" not in entities and entities.get("category") == "Sân bay":
            entities["destination_category"] = "Sân bay"
            entities.pop("category", None)
    if isinstance(entities.get("attributes"), list):
        configured_order = registry.templates.get("attribute_order", "").split("|")
        order = {value: index for index, value in enumerate(configured_order)}
        entities["attributes"].sort(key=lambda value: order.get(value, len(order)))
    result.discovery = result.discovery and "category" not in entities
    result.evidence = collector.items
    result.covered_spans = collector.covered_spans

    if trace is not None:
        new_evidence = trace.evidence[trace_evidence_start:]
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.EXTRACTION,
                input=json.dumps([variant.text for variant in variants], ensure_ascii=False),
                output=json.dumps(entities, ensure_ascii=False, sort_keys=True),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.ENTITY_EXTRACTION,
                        source=str(item.raw_value),
                        target=str(item.canonical_value),
                        start=item.start,
                        end=item.end,
                        rule_id=item.rule_id,
                        grounding=TraceGrounding(source=item.configuration_source),
                        confidence=item.confidence,
                        parent_variant_id=item.source_variant_id,
                    )
                    for item in new_evidence
                ],
            )
        )
    return result
