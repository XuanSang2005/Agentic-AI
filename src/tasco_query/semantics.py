from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.tasco_query.contracts import (
    Action,
    Evidence,
    GroundedConcept,
    GroundingType,
    ImplicationRelationship,
    QueryVariant,
    ReviewDependency,
    ReviewDependencyClassification,
    SearchExpansion,
    SearchExpansionPurpose,
    SemanticDecompositionResult,
    SemanticFrame,
    SemanticImplication,
    SemanticUnit,
    SemanticUnitType,
    SocialDiscoveryDecision,
    TargetKind,
    TraceValue,
    UserLocation,
)
from src.tasco_query.lexicon import LexiconEntry, LexiconRegistry
from src.tasco_query.normalization import clean_display, match_key
from src.tasco_query.tracing import TraceCollector

if TYPE_CHECKING:
    from src.tasco_query.adapters import GroundedLLMAdapter

_PUNCTUATION_RE = re.compile(r"[,;:!?()\[\]{}]+")
_CONNECTOR_RE = re.compile(r"(?:\s+|^)(?:và|hoặc|nhưng|cũng như)(?=\s+|$)", re.IGNORECASE)
_EDGE_FILLER_RE = re.compile(r"^(?:(?:ở|tại|là)\s+)+|\s+(?:(?:ở|tại|là)\s*)+$", re.IGNORECASE)
_PRICE_RE = re.compile(r"\b(?:dưới\s+)?\d+(?:[.,]\d+)?\s*(?:k|nghìn|triệu|đ|vnd)\b", re.IGNORECASE)
_TIME_RE = re.compile(
    r"\b(?:mở\s+)?(?:(?:tới|đến|sau)\s+\d{1,2}(?:h)?(?:\s*(?:đêm|khuya|tối))?"
    r"|24h|24/7|mọi lúc|cả ngày)\b",
    re.IGNORECASE,
)
_COORDINATE_RE = re.compile(r"(?<!\d)-?\d{1,2}(?:\.\d+)\s*,\s*-?\d{1,3}(?:\.\d+)(?!\d)")
_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class QueryContext:
    registry: LexiconRegistry
    evidence: tuple[Evidence, ...] = ()
    location: UserLocation | None = None


class SemanticUnitSegmenter(Protocol):
    def segment(self, variant: QueryVariant, context: QueryContext) -> list[SemanticUnit]: ...


class DirectConceptGrounder(Protocol):
    def ground(self, unit: SemanticUnit, context: QueryContext) -> list[GroundedConcept]: ...


@dataclass(frozen=True, slots=True)
class _SpanCandidate:
    start: int
    end: int
    unit_type: SemanticUnitType
    priority: int = 0


def _type_for_entry(entry: LexiconEntry) -> SemanticUnitType:
    return {
        "category": SemanticUnitType.CATEGORY,
        "cuisine": SemanticUnitType.CUISINE,
        "brand": SemanticUnitType.BRAND,
        "poi_name": SemanticUnitType.POI,
        "street": SemanticUnitType.STREET,
        "nearby": SemanticUnitType.SPATIAL_RELATION,
        "navigation": SemanticUnitType.NAVIGATION,
        "opening_constraint": SemanticUnitType.TIME_CONSTRAINT,
        "open_24h": SemanticUnitType.TIME_CONSTRAINT,
        "open_late": SemanticUnitType.TIME_CONSTRAINT,
        "open_now": SemanticUnitType.TIME_CONSTRAINT,
        "amenities": SemanticUnitType.OBJECTIVE_CONSTRAINT,
        "attributes": (
            SemanticUnitType.PURPOSE
            if entry.entity_type == "purpose"
            else SemanticUnitType.SUBJECTIVE_ATTRIBUTE
        ),
        "quality": SemanticUnitType.SUBJECTIVE_ATTRIBUTE,
        "district": SemanticUnitType.ADDRESS,
        "city": SemanticUnitType.ADDRESS,
        "reference_area": SemanticUnitType.ADDRESS,
        "dish": SemanticUnitType.CUISINE,
    }.get(entry.canonical_field, SemanticUnitType.UNKNOWN)


def _type_for_evidence(kind: str) -> SemanticUnitType:
    if kind in {"category"}:
        return SemanticUnitType.CATEGORY
    if kind in {"cuisine", "dish"}:
        return SemanticUnitType.CUISINE
    if kind == "brand":
        return SemanticUnitType.BRAND
    if kind in {"poi_name", "destination_poi", "reference_poi"}:
        return SemanticUnitType.POI
    if kind == "street":
        return SemanticUnitType.STREET
    if kind in {"house_number", "district", "city", "reference_area"}:
        return SemanticUnitType.ADDRESS
    if kind == "nearby":
        return SemanticUnitType.SPATIAL_RELATION
    if kind == "navigation":
        return SemanticUnitType.NAVIGATION
    if kind == "price_max":
        return SemanticUnitType.PRICE_CONSTRAINT
    if kind.startswith("open_"):
        return SemanticUnitType.TIME_CONSTRAINT
    if kind == "amenities":
        return SemanticUnitType.OBJECTIVE_CONSTRAINT
    if kind in {"attributes", "quality"}:
        return SemanticUnitType.SUBJECTIVE_ATTRIBUTE
    return SemanticUnitType.UNKNOWN


def _fallback_type(text: str) -> SemanticUnitType:
    key = match_key(text)
    if any(marker in key for marker in ("lai xe", "di bo", "xe lan", "tiep can")):
        return SemanticUnitType.ACCESS_CONSTRAINT
    if re.search(r"\b(?:de|hop de|phu hop de)\b", key):
        return SemanticUnitType.PURPOSE
    return SemanticUnitType.UNKNOWN


def _non_overlapping(candidates: list[_SpanCandidate]) -> list[_SpanCandidate]:
    selected: list[_SpanCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.end - item.start, item.priority, -item.start),
        reverse=True,
    ):
        if any(candidate.start < item.end and candidate.end > item.start for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


class DeterministicSemanticUnitSegmenter:
    """Longest-match segmentation with bounded phrase-level fallback chunks."""

    max_fallback_tokens = 6

    def segment(self, variant: QueryVariant, context: QueryContext) -> list[SemanticUnit]:
        text = variant.text
        searchable = _PUNCTUATION_RE.sub(lambda match: " " * len(match.group()), text)
        candidates = [
            _SpanCandidate(
                match.start,
                match.end,
                _type_for_entry(match.entry),
                match.entry.priority,
            )
            for match in context.registry.phrase_matches(searchable)
            if match.entry.canonical_field != "rewrite"
        ]
        for entry in context.registry.entries:
            if entry.canonical_field == "rewrite":
                continue
            for match in re.finditer(re.escape(entry.canonical_rendering), text, re.IGNORECASE):
                candidates.append(
                    _SpanCandidate(
                        match.start(), match.end(), _type_for_entry(entry), entry.priority
                    )
                )
        for evidence in context.evidence:
            span = evidence.span
            if (
                not evidence.accepted
                or evidence.variant_id != variant.variant_id
                or span is None
                or evidence.rule_id.startswith("parser.")
                or span.end > len(text)
                or text[span.start : span.end] != span.text
            ):
                continue
            candidates.append(
                _SpanCandidate(span.start, span.end, _type_for_evidence(evidence.kind), 1000)
            )
        candidates.extend(
            _SpanCandidate(match.start(), match.end(), SemanticUnitType.PRICE_CONSTRAINT, 1000)
            for match in _PRICE_RE.finditer(text)
        )
        candidates.extend(
            _SpanCandidate(match.start(), match.end(), SemanticUnitType.TIME_CONSTRAINT, 1000)
            for match in _TIME_RE.finditer(text)
        )
        candidates.extend(
            _SpanCandidate(match.start(), match.end(), SemanticUnitType.OBJECTIVE_CONSTRAINT, 1000)
            for match in _COORDINATE_RE.finditer(text)
        )
        for candidate in list(candidates):
            if candidate.unit_type != SemanticUnitType.STREET:
                continue
            house = re.search(r"\b\d+[A-Za-z]?\s*$", text[: candidate.start])
            if house:
                candidates.append(
                    _SpanCandidate(house.start(), candidate.end, SemanticUnitType.ADDRESS, 1100)
                )

        selected = _non_overlapping(candidates)
        fragments: list[_SpanCandidate] = []
        cursor = 0
        for candidate in [
            *selected,
            _SpanCandidate(len(text), len(text), SemanticUnitType.UNKNOWN),
        ]:
            fragments.extend(self._fallback_chunks(text, cursor, candidate.start))
            if candidate.end > candidate.start:
                fragments.append(candidate)
            cursor = candidate.end

        units: list[SemanticUnit] = []
        for candidate in sorted(fragments, key=lambda item: item.start):
            raw = text[candidate.start : candidate.end]
            if not raw.strip():
                continue
            units.append(
                SemanticUnit(
                    id=f"unit-{variant.variant_id}-{candidate.start}-{candidate.end}",
                    text=raw,
                    normalized_text=clean_display(raw).casefold(),
                    start=candidate.start,
                    end=candidate.end,
                    source_variant_id=variant.variant_id,
                    unit_type=candidate.unit_type,
                )
            )
        return units

    def _fallback_chunks(self, text: str, start: int, end: int) -> list[_SpanCandidate]:
        result: list[_SpanCandidate] = []
        segment = text[start:end]
        boundary_matches = list(_PUNCTUATION_RE.finditer(segment)) + list(
            _CONNECTOR_RE.finditer(segment)
        )
        boundaries = sorted((match.start(), match.end()) for match in boundary_matches)
        cursor = 0
        for boundary_start, boundary_end in [*boundaries, (len(segment), len(segment))]:
            self._append_bounded(result, text, start + cursor, start + boundary_start)
            cursor = max(cursor, boundary_end)
        return result

    def _append_bounded(
        self, result: list[_SpanCandidate], text: str, start: int, end: int
    ) -> None:
        raw = text[start:end]
        if not raw.strip():
            return
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start += left
        end = start + right - left
        trimmed = _EDGE_FILLER_RE.sub("", text[start:end]).strip()
        if not trimmed or match_key(trimmed) in {"co"}:
            return
        relative = text[start:end].find(trimmed)
        start += relative
        end = start + len(trimmed)
        words = list(_WORD_RE.finditer(text, start, end))
        if not words:
            return
        for index in range(0, len(words), self.max_fallback_tokens):
            chunk = words[index : index + self.max_fallback_tokens]
            chunk_start, chunk_end = chunk[0].start(), chunk[-1].end()
            result.append(
                _SpanCandidate(chunk_start, chunk_end, _fallback_type(text[chunk_start:chunk_end]))
            )


_DIRECT_FIELDS = {
    "amenities",
    "brand",
    "category",
    "city",
    "cuisine",
    "destination_poi",
    "dish",
    "district",
    "house_number",
    "latitude",
    "location",
    "longitude",
    "navigation",
    "nearby",
    "open_24h",
    "open_after",
    "open_late",
    "open_now",
    "open_until",
    "poi_name",
    "price_max",
    "reference_area",
    "reference_poi",
    "street",
}


def _field_allowed(unit: SemanticUnit, field: str) -> bool:
    if unit.unit_type == SemanticUnitType.PURPOSE:
        return field == "attributes"
    if unit.unit_type in {
        SemanticUnitType.SUBJECTIVE_ATTRIBUTE,
        SemanticUnitType.ACCESS_CONSTRAINT,
        SemanticUnitType.UNKNOWN,
        SemanticUnitType.CONNECTOR,
    }:
        return False
    return field in _DIRECT_FIELDS


def _grounding_type(source: str, rule_id: str) -> GroundingType:
    if source.startswith("data/raw/"):
        return GroundingType.LOCAL_DATA
    if rule_id.startswith("parser."):
        return GroundingType.PARSER_EVIDENCE
    return GroundingType.LEXICON_ALIAS


class DeterministicDirectConceptGrounder:
    def ground(self, unit: SemanticUnit, context: QueryContext) -> list[GroundedConcept]:
        candidates: list[tuple[str, TraceValue, float, GroundingType, str | None, str]] = []
        for evidence in context.evidence:
            if (
                not evidence.accepted
                or evidence.variant_id != unit.source_variant_id
                or not _field_allowed(unit, evidence.kind)
            ):
                continue
            span = evidence.span
            overlaps = span is not None and (
                (span.start == unit.start and span.end == unit.end)
                or (
                    unit.unit_type
                    in {
                        SemanticUnitType.ADDRESS,
                        SemanticUnitType.OBJECTIVE_CONSTRAINT,
                        SemanticUnitType.PRICE_CONSTRAINT,
                        SemanticUnitType.TIME_CONSTRAINT,
                    }
                    and unit.start <= span.start
                    and unit.end >= span.end
                )
            )
            parser_matches_type = evidence.rule_id.startswith("parser.") and (
                (
                    unit.unit_type == SemanticUnitType.PRICE_CONSTRAINT
                    and evidence.kind == "price_max"
                )
                or (
                    unit.unit_type == SemanticUnitType.TIME_CONSTRAINT
                    and evidence.kind.startswith("open_")
                )
            )
            if overlaps or parser_matches_type:
                candidates.append(
                    (
                        evidence.kind,
                        evidence.value,
                        evidence.confidence,
                        _grounding_type(evidence.config_source, evidence.rule_id),
                        evidence.rule_id,
                        evidence.config_source,
                    )
                )

        searchable = _PUNCTUATION_RE.sub(" ", unit.text)
        matches = [
            match
            for match in context.registry.phrase_matches(searchable)
            if _field_allowed(unit, match.entry.canonical_field)
            and (
                unit.unit_type
                in {
                    SemanticUnitType.ADDRESS,
                    SemanticUnitType.OBJECTIVE_CONSTRAINT,
                    SemanticUnitType.PRICE_CONSTRAINT,
                    SemanticUnitType.TIME_CONSTRAINT,
                }
                or match.start == 0
            )
            and (
                unit.unit_type
                in {
                    SemanticUnitType.ADDRESS,
                    SemanticUnitType.OBJECTIVE_CONSTRAINT,
                    SemanticUnitType.PRICE_CONSTRAINT,
                    SemanticUnitType.TIME_CONSTRAINT,
                }
                or match.end == len(searchable)
            )
        ]
        canonical_by_field: dict[str, set[str]] = {}
        for match in matches:
            canonical_by_field.setdefault(match.entry.canonical_field, set()).add(
                json.dumps(match.entry.canonical, ensure_ascii=False, sort_keys=True)
            )
        for match in matches:
            entry = match.entry
            if len(canonical_by_field[entry.canonical_field]) > 1:
                continue
            candidates.append(
                (
                    entry.canonical_field,
                    entry.canonical,
                    entry.confidence,
                    _grounding_type(entry.source, entry.rule_id),
                    entry.rule_id,
                    entry.source,
                )
            )
        canonical_entries = [
            entry
            for entry in context.registry.entries
            if _field_allowed(unit, entry.canonical_field)
            and clean_display(entry.canonical_rendering).casefold() == unit.normalized_text
        ]
        canonical_values = {
            (entry.canonical_field, json.dumps(entry.canonical, ensure_ascii=False, sort_keys=True))
            for entry in canonical_entries
        }
        for entry in canonical_entries:
            if sum(field == entry.canonical_field for field, _ in canonical_values) > 1:
                continue
            candidates.append(
                (
                    entry.canonical_field,
                    entry.canonical,
                    entry.confidence,
                    _grounding_type(entry.source, entry.rule_id),
                    entry.rule_id,
                    entry.source,
                )
            )

        if unit.unit_type == SemanticUnitType.SPATIAL_RELATION and context.location is not None:
            candidates.append(
                (
                    "location",
                    "current_location",
                    1.0,
                    GroundingType.REQUEST_CONTEXT,
                    "request_context.current_location",
                    "request.location",
                )
            )

        concepts: list[GroundedConcept] = []
        seen: set[tuple[str, str]] = set()
        for field, value, confidence, grounding_type, rule_id, source in candidates:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if (field, serialized) in seen:
                continue
            seen.add((field, serialized))
            concepts.append(
                GroundedConcept(
                    id=f"grounding-{unit.id}-{len(concepts) + 1}",
                    source_unit_id=unit.id,
                    field=field,
                    canonical_value=value,
                    confidence=confidence,
                    grounding_type=grounding_type,
                    rule_id=rule_id,
                    source=source,
                )
            )
        return concepts


def decompose_variants(
    variants: list[QueryVariant],
    context: QueryContext,
    *,
    segmenter: SemanticUnitSegmenter | None = None,
    grounder: DirectConceptGrounder | None = None,
    trace: TraceCollector | None = None,
) -> SemanticDecompositionResult:
    segmenter = segmenter or DeterministicSemanticUnitSegmenter()
    grounder = grounder or DeterministicDirectConceptGrounder()
    units: list[SemanticUnit] = []
    concepts: list[GroundedConcept] = []
    seen: set[tuple[object, ...]] = set()
    for variant in variants:
        for unit in segmenter.segment(variant, context):
            grounded = grounder.ground(unit, context)
            key: tuple[object, ...]
            if grounded:
                key = (
                    unit.unit_type,
                    tuple(
                        sorted(
                            (item.field, json.dumps(item.canonical_value, ensure_ascii=False))
                            for item in grounded
                            if item.grounding_type != GroundingType.REQUEST_CONTEXT
                        )
                    ),
                )
            else:
                key = (unit.unit_type, match_key(unit.text))
            if key in seen:
                continue
            seen.add(key)
            enriched = unit.model_copy(
                update={
                    "directly_grounded": bool(grounded),
                    "grounding_ids": [item.id for item in grounded],
                }
            )
            units.append(enriched)
            concepts.extend(grounded)
    result = SemanticDecompositionResult(
        units=units,
        grounded_concepts=concepts,
        unresolved_unit_ids=[unit.id for unit in units if not unit.directly_grounded],
    )
    if trace is not None:
        trace.set_semantic_decomposition(result)
    return result


@dataclass(frozen=True, slots=True)
class SemanticContext:
    registry: LexiconRegistry
    grounded: tuple[GroundedConcept, ...] = ()
    existing_entities: dict[str, Any] | None = None


class SemanticImplicationResolver(Protocol):
    async def resolve(
        self, unit: SemanticUnit, context: SemanticContext
    ) -> list[SemanticImplication]: ...


class EmbeddingSemanticImplicationResolver(Protocol):
    """Optional semantic-similarity resolver; it is intentionally not required at runtime."""

    async def resolve(
        self, unit: SemanticUnit, context: SemanticContext
    ) -> list[SemanticImplication]: ...


class SearchExpansionGenerator(Protocol):
    def generate(
        self,
        grounded: list[GroundedConcept],
        implications: list[SemanticImplication],
        frame: Any | None,
    ) -> list[SearchExpansion]: ...


class ReviewDependencyClassifier(Protocol):
    def classify(
        self,
        decomposition: SemanticDecompositionResult,
        implications: list[SemanticImplication],
        evidence: tuple[Evidence, ...],
    ) -> list[ReviewDependencyClassification]: ...


class SocialDiscoveryGate(Protocol):
    def evaluate(
        self,
        frame: SemanticFrame,
        implications: list[SemanticImplication],
        classifications: list[ReviewDependencyClassification],
    ) -> SocialDiscoveryDecision: ...


class _ImplicationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str | bool | int | float | list[str]
    confidence: float = Field(ge=0, le=1)
    relationship: ImplicationRelationship
    requires_external_validation: bool = False
    review_dependency: ReviewDependency


class _ExpansionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    fields: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    purpose: SearchExpansionPurpose


class _SemanticMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    aliases: list[str] = Field(min_length=1)
    implications: list[_ImplicationSpec] = Field(min_length=1)
    expansions: list[_ExpansionSpec] = Field(default_factory=list)


class _SemanticMappingFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mappings: list[_SemanticMapping]


_SEMANTIC_FIELDS = {"privacy_preference", "transport_mode", "car_accessible", "amenities"}
_RELATIONSHIP_STRENGTH = {
    ImplicationRelationship.DIRECT_INTERPRETATION: 5,
    ImplicationRelationship.CANONICAL_PARAPHRASE: 4,
    ImplicationRelationship.LIKELY_RELATED_PREFERENCE: 3,
    ImplicationRelationship.POSSIBLE_SUPPORTING_FEATURE: 2,
    ImplicationRelationship.SEARCH_EXPANSION_ONLY: 1,
}


class SemanticImplicationLexicon:
    def __init__(self, mappings: list[_SemanticMapping], source: str) -> None:
        if any(
            spec.field not in _SEMANTIC_FIELDS for item in mappings for spec in item.implications
        ):
            raise ValueError("semantic mapping has an unsupported field")
        self.mappings = mappings
        self.source = source
        self._by_alias = {
            match_key(alias): mapping for mapping in mappings for alias in mapping.aliases
        }

    @classmethod
    def load(cls, path: Path) -> SemanticImplicationLexicon:
        return cls(
            _SemanticMappingFile.model_validate_json(path.read_text(encoding="utf-8")).mappings,
            str(path),
        )

    def match(self, text: str) -> _SemanticMapping | None:
        return self._by_alias.get(match_key(text))


def _value_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def validate_implications(
    implications: list[SemanticImplication],
    context: SemanticContext,
    *,
    min_confidence: float = 0.5,
) -> list[SemanticImplication]:
    """Keep only allowlisted, non-contradictory, strongest implication facts."""
    explicit = context.existing_entities or {}
    selected: dict[tuple[str, str], SemanticImplication] = {}
    scalar_values: dict[str, str] = {
        field: _value_key(value)
        for field, value in explicit.items()
        if field in _SEMANTIC_FIELDS and not isinstance(value, list)
    }
    for item in implications:
        if item.field not in _SEMANTIC_FIELDS or item.confidence < min_confidence:
            continue
        key = (item.field, _value_key(item.value))
        if item.field in scalar_values and scalar_values[item.field] != key[1]:
            continue
        if not isinstance(item.value, list):
            previous = scalar_values.get(item.field)
            if previous is not None and previous != key[1]:
                continue
            scalar_values[item.field] = key[1]
        current = selected.get(key)
        if current is None or (_RELATIONSHIP_STRENGTH[item.relationship], item.confidence) > (
            _RELATIONSHIP_STRENGTH[current.relationship],
            current.confidence,
        ):
            selected[key] = item
    return list(selected.values())


class DeterministicSemanticImplicationResolver:
    def __init__(self, lexicon: SemanticImplicationLexicon) -> None:
        self.lexicon = lexicon

    async def resolve(
        self, unit: SemanticUnit, context: SemanticContext
    ) -> list[SemanticImplication]:
        mapping = self.lexicon.match(unit.text)
        if mapping is None:
            return []
        candidates = [
            SemanticImplication(
                id=f"implication-{unit.id}-{index}",
                source_unit_id=unit.id,
                field=spec.field,
                value=spec.value,
                confidence=spec.confidence,
                relationship=spec.relationship,
                grounding=[self.lexicon.source, mapping.id, unit.text],
                requires_external_validation=spec.requires_external_validation,
                review_dependency=spec.review_dependency,
            )
            for index, spec in enumerate(mapping.implications, 1)
        ]
        return validate_implications(candidates, context)


class GroundedLLMSemanticImplicationResolver:
    """Uses the Task 03 adapter as an optional, validated agreement signal."""

    def __init__(
        self,
        adapter: GroundedLLMAdapter,
        deterministic: DeterministicSemanticImplicationResolver,
        *,
        acceptance_threshold: float = 0.8,
    ) -> None:
        self.adapter = adapter
        self.deterministic = deterministic
        self.acceptance_threshold = acceptance_threshold

    async def resolve(
        self, unit: SemanticUnit, context: SemanticContext
    ) -> list[SemanticImplication]:
        baseline = await self.deterministic.resolve(unit, context)
        if not baseline or not self.adapter.enabled:
            return baseline
        try:
            import asyncio

            result = await asyncio.to_thread(
                self.adapter.propose,
                original_query=unit.text,
                cleaned_query=unit.text,
                protected=[],
                deterministic_evidence=[],
                local_matches=[],
                allowed_fields={item.field: [item.value] for item in baseline},
            )
        except (TimeoutError, OSError, RuntimeError, ValueError):
            return baseline
        if result.error or not result.proposals:
            return baseline
        confirmed = {
            (proposal.field, _value_key(proposal.value))
            for proposal in result.proposals[0].semantic_evidence
            if proposal.confidence >= self.acceptance_threshold
            and match_key(proposal.evidence_text) in match_key(unit.text)
        }
        if not confirmed:
            return baseline
        return [
            item.model_copy(update={"confidence": min(1.0, item.confidence + 0.03)})
            for item in baseline
            if (item.field, _value_key(item.value)) in confirmed
        ]


class CompositeSemanticImplicationResolver:
    def __init__(
        self, resolvers: list[SemanticImplicationResolver], *, min_confidence: float = 0.5
    ) -> None:
        self.resolvers = resolvers
        self.min_confidence = min_confidence

    async def resolve(
        self, unit: SemanticUnit, context: SemanticContext
    ) -> list[SemanticImplication]:
        candidates = [
            implication
            for resolver in self.resolvers
            for implication in await resolver.resolve(unit, context)
        ]
        return validate_implications(candidates, context, min_confidence=self.min_confidence)


class DeterministicSearchExpansionGenerator:
    max_expansions = 12

    def __init__(self, lexicon: SemanticImplicationLexicon) -> None:
        self.lexicon = lexicon

    def generate(
        self,
        grounded: list[GroundedConcept],
        implications: list[SemanticImplication],
        frame: Any | None,
    ) -> list[SearchExpansion]:
        del grounded, frame
        result: list[SearchExpansion] = []
        seen: set[tuple[str, str]] = set()
        for implication in implications:
            mapping = self.lexicon.match(implication.grounding[-1])
            if mapping is None:
                continue
            for spec in mapping.expansions:
                key = (spec.purpose.value, match_key(spec.text))
                if key in seen or len(result) >= self.max_expansions:
                    continue
                seen.add(key)
                result.append(
                    SearchExpansion(
                        id=f"expansion-{implication.source_unit_id}-{len(result) + 1}",
                        source_unit_ids=[implication.source_unit_id],
                        text=spec.text,
                        fields=spec.fields,
                        confidence=min(implication.confidence, spec.confidence),
                        purpose=spec.purpose,
                        grounding=[self.lexicon.source, mapping.id, implication.id],
                    )
                )
        return result


async def resolve_implications(
    decomposition: SemanticDecompositionResult,
    context: SemanticContext,
    resolver: SemanticImplicationResolver,
) -> list[SemanticImplication]:
    result = [
        implication
        for unit in decomposition.units
        if unit.id in decomposition.unresolved_unit_ids
        for implication in await resolver.resolve(unit, context)
    ]
    return validate_implications(result, context)


def apply_explicit_implications(
    entities: dict[str, Any], implications: list[SemanticImplication]
) -> None:
    for implication in implications:
        if implication.relationship not in {
            ImplicationRelationship.DIRECT_INTERPRETATION,
            ImplicationRelationship.CANONICAL_PARAPHRASE,
        }:
            continue
        if implication.field not in entities:
            entities[implication.field] = implication.value


class _ReviewDependencyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aliases: list[str] = Field(min_length=1)
    review_dependency: ReviewDependency
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


class _ReviewDependencyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[_ReviewDependencyEntry]


class ReviewDependencyLexicon:
    def __init__(self, entries: list[_ReviewDependencyEntry]) -> None:
        self._by_alias = {match_key(alias): entry for entry in entries for alias in entry.aliases}

    @classmethod
    def load(cls, path: Path) -> ReviewDependencyLexicon:
        return cls(
            _ReviewDependencyFile.model_validate_json(path.read_text(encoding="utf-8")).entries
        )

    def match(self, text: str) -> _ReviewDependencyEntry | None:
        return self._by_alias.get(match_key(text))

    def has_subjective_or_review_criterion(self, text: str) -> bool:
        key = match_key(text)
        return any(
            alias in key
            and entry.review_dependency
            in {ReviewDependency.SUBJECTIVE, ReviewDependency.REVIEW_DEPENDENT}
            for alias, entry in self._by_alias.items()
        )


_UNIT_DEPENDENCIES = {
    SemanticUnitType.CATEGORY: ReviewDependency.OBJECTIVE,
    SemanticUnitType.CUISINE: ReviewDependency.OBJECTIVE,
    SemanticUnitType.BRAND: ReviewDependency.OBJECTIVE,
    SemanticUnitType.POI: ReviewDependency.OBJECTIVE,
    SemanticUnitType.ADDRESS: ReviewDependency.OBJECTIVE,
    SemanticUnitType.STREET: ReviewDependency.OBJECTIVE,
    SemanticUnitType.NAVIGATION: ReviewDependency.OBJECTIVE,
    SemanticUnitType.OBJECTIVE_CONSTRAINT: ReviewDependency.OBJECTIVE,
    SemanticUnitType.PRICE_CONSTRAINT: ReviewDependency.OBJECTIVE,
    SemanticUnitType.TIME_CONSTRAINT: ReviewDependency.OBJECTIVE,
    SemanticUnitType.SPATIAL_RELATION: ReviewDependency.LOCATION_DEPENDENT,
    SemanticUnitType.ACCESS_CONSTRAINT: ReviewDependency.LOCATION_DEPENDENT,
    SemanticUnitType.SUBJECTIVE_ATTRIBUTE: ReviewDependency.SUBJECTIVE,
    SemanticUnitType.PURPOSE: ReviewDependency.SUBJECTIVE,
}


class DeterministicReviewDependencyClassifier:
    def __init__(self, lexicon: ReviewDependencyLexicon) -> None:
        self.lexicon = lexicon

    @staticmethod
    def _evidence_ids(unit: SemanticUnit, evidence: tuple[Evidence, ...]) -> list[str]:
        return [
            item.evidence_id
            for item in evidence
            if item.accepted
            and item.variant_id == unit.source_variant_id
            and item.span is not None
            and item.span.start < unit.end
            and item.span.end > unit.start
        ]

    def classify(
        self,
        decomposition: SemanticDecompositionResult,
        implications: list[SemanticImplication],
        evidence: tuple[Evidence, ...],
    ) -> list[ReviewDependencyClassification]:
        units = {unit.id: unit for unit in decomposition.units}
        result: list[ReviewDependencyClassification] = []
        for unit in decomposition.units:
            configured = self.lexicon.match(unit.text)
            dependency = (
                configured.review_dependency
                if configured
                else _UNIT_DEPENDENCIES.get(unit.unit_type)
            )
            if dependency is None:
                continue
            result.append(
                ReviewDependencyClassification(
                    id=f"review-dependency-{unit.id}",
                    source_unit_id=unit.id,
                    concept_id=unit.id,
                    review_dependency=dependency,
                    confidence=(configured.confidence if configured else 1.0),
                    evidence_ids=self._evidence_ids(unit, evidence),
                    reason=(configured.reason if configured else f"semantic_unit:{unit.unit_type}"),
                )
            )
        for implication in implications:
            unit = units[implication.source_unit_id]
            result.append(
                ReviewDependencyClassification(
                    id=f"review-dependency-{implication.id}",
                    source_unit_id=unit.id,
                    concept_id=implication.id,
                    review_dependency=implication.review_dependency,
                    confidence=implication.confidence,
                    evidence_ids=self._evidence_ids(unit, evidence),
                    reason="semantic_implication",
                )
            )
        return result


class DeterministicSocialDiscoveryGate:
    min_confidence = 0.7

    def __init__(
        self,
        *,
        min_confidence: float = min_confidence,
        exclude_objective_only: bool = True,
        exclude_exact_queries: bool = True,
    ) -> None:
        self.min_confidence = min_confidence
        self.exclude_objective_only = exclude_objective_only
        self.exclude_exact_queries = exclude_exact_queries

    def evaluate(
        self,
        frame: SemanticFrame,
        implications: list[SemanticImplication],
        classifications: list[ReviewDependencyClassification],
    ) -> SocialDiscoveryDecision:
        del implications  # Classifications retain their implication provenance.
        exact_reason = {
            Action.NAVIGATE: "navigation_request",
            Action.LOCATE_COORDINATE: "coordinate_lookup",
        }.get(frame.action)
        if exact_reason is None:
            exact_reason = {
                TargetKind.POI: "exact_poi",
                TargetKind.ADDRESS: "exact_address",
                TargetKind.COORDINATE: "coordinate_lookup",
            }.get(frame.target_kind)
        if (
            exact_reason is None
            and "street" in frame.entities
            and frame.target_kind not in {TargetKind.CATEGORY, TargetKind.CUISINE, TargetKind.DISH}
        ):
            exact_reason = "street_lookup"
        if exact_reason is not None and self.exclude_exact_queries:
            return SocialDiscoveryDecision(
                should_trigger=False,
                reason=exact_reason,
                excluded_reasons=[exact_reason],
                confidence=1.0,
            )
        if frame.target_kind not in {TargetKind.CATEGORY, TargetKind.CUISINE, TargetKind.DISH}:
            return SocialDiscoveryDecision(
                should_trigger=False,
                reason="not_category_or_discovery_query",
                excluded_reasons=["not_category_or_discovery_query"],
                confidence=1.0,
            )
        relevant = [
            item
            for item in classifications
            if item.review_dependency
            in {ReviewDependency.SUBJECTIVE, ReviewDependency.REVIEW_DEPENDENT}
        ]
        eligible = [item for item in relevant if item.confidence >= self.min_confidence]
        if not eligible:
            reason = (
                "insufficient_semantic_confidence"
                if relevant
                else "objective_or_location_constraints_only"
            )
            excluded = [reason]
            if classifications and not relevant:
                excluded.extend(sorted({item.review_dependency.value for item in classifications}))
            return SocialDiscoveryDecision(
                should_trigger=False,
                reason=reason,
                excluded_reasons=excluded,
                confidence=max((item.confidence for item in relevant), default=1.0),
            )
        return SocialDiscoveryDecision(
            should_trigger=True,
            reason="subjective_or_review_dependent_constraint",
            triggering_unit_ids=list(dict.fromkeys(item.source_unit_id for item in eligible)),
            triggering_evidence_ids=list(
                dict.fromkeys(evidence_id for item in eligible for evidence_id in item.evidence_ids)
            ),
            excluded_reasons=sorted(
                {
                    item.review_dependency.value
                    for item in classifications
                    if item.review_dependency
                    in {ReviewDependency.OBJECTIVE, ReviewDependency.LOCATION_DEPENDENT}
                }
            ),
            confidence=min(item.confidence for item in eligible),
        )
