from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isclose
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Intent = Literal[
    "POI Search",
    "Category Search",
    "Brand Category Search",
    "Address Search",
    "Coordinate Search",
    "Nearby Search",
    "Navigation",
    "Discovery Search",
    "Ambiguous",
]


class UserLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0, le=100_000)
    area: str | None = Field(default=None, min_length=1, max_length=160)
    city: str | None = Field(default=None, min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_coordinate_pair(self) -> UserLocation:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be provided together")
        return self


class QueryUnderstandRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"query": "cf yên tĩnh để học"},
                {
                    "query": "cf yên tĩnh để học",
                    "include_trace": True,
                },
            ]
        },
    )

    query: str = Field(min_length=1, max_length=512)
    location: UserLocation | None = None
    locale: Literal["vi-VN", "en-US"] = "vi-VN"
    timezone: str = Field(default="Asia/Ho_Chi_Minh", min_length=1, max_length=80)
    now: datetime | None = None
    include_trace: bool = False

    @field_validator("query")
    @classmethod
    def query_must_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        if sum(ord(char) < 32 and char not in "\t\n\r" for char in value) > 2:
            raise ValueError("query contains too many control characters")
        return value


class QueryUnderstandResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "normalized_query": "Quán cà phê yên tĩnh phù hợp học tập",
                    "intent": "Category Search",
                    "entities": {
                        "category": "Quán cà phê",
                        "attributes": ["yên tĩnh", "phù hợp học tập"],
                    },
                }
            ]
        },
    )

    normalized_query: str
    intent: Intent
    entities: dict[str, Any]


class LockLevel(StrEnum):
    HARD_LOCK = "hard_lock"
    SOFT_LOCK = "soft_lock"
    EDITABLE = "editable"


class SourceSpan(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str


class Evidence(BaseModel):
    evidence_id: str
    kind: str
    value: str | bool | int | float | list[str]
    span: SourceSpan | None = None
    method: str
    confidence: float = Field(ge=0, le=1)
    variant_id: str = "v0"
    rule_id: str = "parser.deterministic"
    config_source: str = "python_parser"
    precedence: int = Field(default=1, ge=1, le=7)
    accepted: bool = True
    rejection_reason: str | None = None
    merge_decision: str = "accepted"


class ProtectedSpan(BaseModel):
    span: SourceSpan
    level: LockLevel
    kind: str


class RewriteEdit(BaseModel):
    source: str
    replacement: str
    reason: str


class RewriteCandidate(BaseModel):
    source_text: str
    proposed_text: str
    transformation_type: str
    rule_id: str
    source_span: SourceSpan
    grounding_source: str
    matched_lexicon_item: str
    confidence: float = Field(ge=0, le=1)
    cost: float = Field(ge=0)
    parent_variant_id: str


class QueryVariant(BaseModel):
    variant_id: str
    text: str
    source: str
    prior: float = Field(ge=0, le=1)
    rewrite_cost: float = Field(ge=0)
    edits: list[RewriteEdit] = Field(default_factory=list)
    hard_locks_preserved: bool = True
    parent_id: str | None = None
    generation: list[RewriteCandidate] = Field(default_factory=list)
    model_identifier: str | None = None
    adapter_version: str | None = None
    protected_span_validation_status: Literal["valid", "rejected"] | None = None
    rejection_reason: str | None = None


class SemanticUnitType(StrEnum):
    CATEGORY = "category"
    CUISINE = "cuisine"
    BRAND = "brand"
    POI = "poi"
    ADDRESS = "address"
    STREET = "street"
    SPATIAL_RELATION = "spatial_relation"
    NAVIGATION = "navigation"
    OBJECTIVE_CONSTRAINT = "objective_constraint"
    SUBJECTIVE_ATTRIBUTE = "subjective_attribute"
    PURPOSE = "purpose"
    ACCESS_CONSTRAINT = "access_constraint"
    PRICE_CONSTRAINT = "price_constraint"
    TIME_CONSTRAINT = "time_constraint"
    CONNECTOR = "connector"
    UNKNOWN = "unknown"


class GroundingType(StrEnum):
    LEXICON_ALIAS = "lexicon_alias"
    LOCAL_DATA = "local_data"
    PARSER_EVIDENCE = "parser_evidence"
    REQUEST_CONTEXT = "request_context"


class ImplicationRelationship(StrEnum):
    DIRECT_INTERPRETATION = "direct_interpretation"
    CANONICAL_PARAPHRASE = "canonical_paraphrase"
    LIKELY_RELATED_PREFERENCE = "likely_related_preference"
    POSSIBLE_SUPPORTING_FEATURE = "possible_supporting_feature"
    SEARCH_EXPANSION_ONLY = "search_expansion_only"


class ReviewDependency(StrEnum):
    OBJECTIVE = "OBJECTIVE"
    SUBJECTIVE = "SUBJECTIVE"
    REVIEW_DEPENDENT = "REVIEW_DEPENDENT"
    LOCATION_DEPENDENT = "LOCATION_DEPENDENT"


class ReviewDependencyClassification(BaseModel):
    """Verifiability of one structured semantic concept."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_unit_id: str
    concept_id: str
    review_dependency: ReviewDependency
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class SocialDiscoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_trigger: bool
    reason: str
    triggering_unit_ids: list[str] = Field(default_factory=list)
    triggering_evidence_ids: list[str] = Field(default_factory=list)
    excluded_reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class SearchExpansionPurpose(StrEnum):
    RETRIEVAL_RECALL = "retrieval_recall"
    SOCIAL_SEARCH = "social_search"
    LOCAL_SEARCH = "local_search"
    RANKING_HINT = "ranking_hint"


class SemanticUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    normalized_text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    source_variant_id: str
    unit_type: SemanticUnitType
    directly_grounded: bool = False
    grounding_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_offsets_and_grounding(self) -> SemanticUnit:
        if self.end <= self.start:
            raise ValueError("semantic unit end must follow start")
        if self.directly_grounded != bool(self.grounding_ids):
            raise ValueError("semantic unit grounding status and IDs must agree")
        return self


class GroundedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_unit_id: str
    field: str
    canonical_value: str | bool | int | float | list[str]
    confidence: float = Field(ge=0, le=1)
    grounding_type: GroundingType
    rule_id: str | None = None
    source: str


class SemanticImplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_unit_id: str
    field: str
    value: str | bool | int | float | list[str]
    confidence: float = Field(ge=0, le=1)
    relationship: ImplicationRelationship
    grounding: list[str] = Field(min_length=1)
    requires_external_validation: bool = False
    review_dependency: ReviewDependency


class SearchExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_unit_ids: list[str] = Field(min_length=1)
    text: str = Field(min_length=1)
    fields: dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    purpose: SearchExpansionPurpose
    grounding: list[str] = Field(default_factory=list)


class SemanticDecompositionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    units: list[SemanticUnit] = Field(default_factory=list)
    grounded_concepts: list[GroundedConcept] = Field(default_factory=list)
    unresolved_unit_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> SemanticDecompositionResult:
        unit_ids = {unit.id for unit in self.units}
        grounding_ids = {concept.id for concept in self.grounded_concepts}
        if len(unit_ids) != len(self.units):
            raise ValueError("semantic unit IDs must be unique")
        if len(grounding_ids) != len(self.grounded_concepts):
            raise ValueError("grounded concept IDs must be unique")
        if any(concept.source_unit_id not in unit_ids for concept in self.grounded_concepts):
            raise ValueError("grounded concept references an unknown semantic unit")
        if any(set(unit.grounding_ids) - grounding_ids for unit in self.units):
            raise ValueError("semantic unit references an unknown grounded concept")
        if set(self.unresolved_unit_ids) - unit_ids:
            raise ValueError("unresolved list references an unknown semantic unit")
        expected_unresolved = {unit.id for unit in self.units if not unit.directly_grounded}
        if set(self.unresolved_unit_ids) != expected_unresolved:
            raise ValueError("unresolved list must contain every ungrounded semantic unit")
        return self


class Action(StrEnum):
    SEARCH = "SEARCH"
    NAVIGATE = "NAVIGATE"
    LOCATE_COORDINATE = "LOCATE_COORDINATE"


class TargetKind(StrEnum):
    POI = "POI"
    BRAND = "BRAND"
    CATEGORY = "CATEGORY"
    DISH = "DISH"
    CUISINE = "CUISINE"
    ADDRESS = "ADDRESS"
    COORDINATE = "COORDINATE"
    UNKNOWN = "UNKNOWN"


class SpatialRelation(StrEnum):
    NONE = "NONE"
    NEAR_CURRENT = "NEAR_CURRENT"
    NEAR_REFERENCE = "NEAR_REFERENCE"
    WITHIN_AREA = "WITHIN_AREA"
    ON_STREET = "ON_STREET"
    ALONG_ROUTE = "ALONG_ROUTE"


class SearchStyle(StrEnum):
    EXACT = "EXACT"
    DISCOVERY = "DISCOVERY"


class SemanticFrame(BaseModel):
    action: Action = Action.SEARCH
    target_kind: TargetKind = TargetKind.UNKNOWN
    spatial_relation: SpatialRelation = SpatialRelation.NONE
    search_style: SearchStyle = SearchStyle.EXACT
    entities: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_tokens: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    grounded_entity_support: float
    required_slot_completeness: float
    action_relation_match: float
    rewrite_safety: float
    modifier_coverage: float
    location_context_consistency: float
    ambiguity_penalty: float
    unresolved_target_penalty: float
    unsafe_rewrite_penalty: float
    contradiction_penalty: float
    hallucination_penalty: float
    model_confidence: float = 0.0
    deterministic_agreement: float = 0.0
    local_data_grounding: float = 0.0
    protected_span_consistency: float = 0.0
    correction_distance: float = 0.0
    semantic_compatibility: float = 0.0
    model_score_contribution: float = 0.0
    unsupported_entity_penalty: float = 0.0
    total: float


class InterpretationCandidate(BaseModel):
    candidate_id: str
    intent: Intent
    frame: SemanticFrame
    normalized_query: str
    score: ScoreBreakdown


class CanonicalEntities(BaseModel):
    """Typed validation boundary before compact entity serialization."""

    model_config = ConfigDict(extra="forbid")

    poi_name: str | None = None
    brand: str | None = None
    category: str | None = None
    cuisine: str | None = None
    dish: str | None = None
    alias: str | None = None
    house_number: str | None = None
    street: str | None = None
    ward: str | None = None
    district: str | None = None
    city: str | None = None
    reference_address: str | None = None
    reference_area: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    location: str | None = None
    reference_poi: str | None = None
    route_destination: str | None = None
    origin: str | None = None
    destination_poi: str | None = None
    destination_category: str | None = None
    attributes: list[str] | None = None
    amenities: list[str] | None = None
    transport_mode: str | None = None
    car_accessible: bool | None = None
    privacy_preference: str | None = None
    quality: str | None = None
    sentiment: str | None = None
    price_max: int | None = None
    open_now: bool | None = None
    open_late: bool | None = None
    open_24h: bool | None = None
    open_after: str | None = None
    open_until: str | None = None
    action: str | None = None
    candidates: list[str] | None = None
    ambiguity_type: str | None = None

    def compact(self) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        amenities = payload.pop("amenities", [])
        if amenities:
            attributes = payload.setdefault("attributes", [])
            payload["attributes"] = list(dict.fromkeys([*attributes, *amenities]))
        return payload


class TraceStageName(StrEnum):
    UNICODE_NORMALIZATION = "unicode_normalization"
    TEXT_CLEANUP = "text_cleanup"
    PROTECTED_SPAN_DETECTION = "protected_span_detection"
    DETERMINISTIC_REWRITE = "deterministic_rewrite"
    MODEL_ASSISTANCE = "model_assistance"
    PROTECTED_SPAN_RESTORATION = "protected_span_restoration"
    EXTRACTION = "extraction"
    SEMANTIC_DECOMPOSITION = "semantic_decomposition"
    DIRECT_CONCEPT_GROUNDING = "direct_concept_grounding"
    SEMANTIC_IMPLICATION = "semantic_implication"
    SEARCH_EXPANSION = "search_expansion"
    REVIEW_DEPENDENCY_CLASSIFICATION = "review_dependency_classification"
    SOCIAL_DISCOVERY_GATE = "social_discovery_gate"
    SEMANTIC_FRAME_CONSTRUCTION = "semantic_frame_construction"
    CANDIDATE_SCORING = "candidate_scoring"
    AMBIGUITY_DECISION = "ambiguity_decision"
    FINAL_RENDERING = "final_rendering"


class TraceOperationType(StrEnum):
    UNICODE_NORMALIZATION = "unicode_normalization"
    WHITESPACE_NORMALIZATION = "whitespace_normalization"
    PUNCTUATION_NORMALIZATION = "punctuation_normalization"
    REPEATED_CHARACTER_OBSERVATION = "repeated_character_observation"
    REPEATED_CHARACTER_REDUCTION = "repeated_character_reduction"
    GUARDED_REPEATED_CHARACTER_CORRECTION = "guarded_repeated_character_correction"
    PHRASE_CANONICALIZATION = "phrase_canonicalization"
    TELEX_DECODING = "telex_decoding"
    TELEX_GATE = "telex_gate"
    TEEN_CODE_EXPANSION = "teen_code_expansion"
    ABBREVIATION_EXPANSION = "abbreviation_expansion"
    ACCENT_RESTORATION = "accent_restoration"
    EDIT_DISTANCE_CORRECTION = "edit_distance_correction"
    PHRASE_ALIAS_EXPANSION = "phrase_alias_expansion"
    MODEL_CORRECTION = "model_correction"
    MODEL_SEMANTIC_EXTRACTION = "model_semantic_extraction"
    PROTECTED_SPAN_PRESERVATION = "protected_span_preservation"
    ENTITY_EXTRACTION = "entity_extraction"
    SEMANTIC_UNIT_SEGMENTATION = "semantic_unit_segmentation"
    DIRECT_CONCEPT_GROUNDING = "direct_concept_grounding"
    UNRESOLVED_UNIT_ROUTING = "unresolved_unit_routing"
    SEMANTIC_IMPLICATION = "semantic_implication"
    SEARCH_EXPANSION = "search_expansion"
    REVIEW_DEPENDENCY_CLASSIFICATION = "review_dependency_classification"
    SOCIAL_DISCOVERY_GATE = "social_discovery_gate"
    FRAME_CONSTRUCTION = "frame_construction"
    SCORE_AGGREGATION = "score_aggregation"
    AMBIGUITY_EVALUATION = "ambiguity_evaluation"
    FINAL_RENDERING = "final_rendering"


class TraceVariantSource(StrEnum):
    CLEANED_ORIGINAL = "cleaned_original"
    REPEATED_CHARACTER_REDUCTION = "repeated_character_reduction"
    GUARDED_REPETITION_CORRECTION = "guarded_repetition_correction"
    TELEX_DECODE = "telex_decode"
    DETERMINISTIC_ALIAS_EXPANSION = "deterministic_alias_expansion"
    TEEN_CODE_EXPANSION = "teen_code_expansion"
    ABBREVIATION_EXPANSION = "abbreviation_expansion"
    ACCENT_RESTORATION = "accent_restoration"
    EDIT_DISTANCE = "edit_distance"
    PHRASE_ALIAS = "phrase_alias"
    HUGGINGFACE_CORRECTION = "huggingface_correction"
    GROUNDED_LLM = "grounded_llm"


class TraceEvidenceType(StrEnum):
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    CATEGORY = "category"
    CUISINE = "cuisine"
    DISH = "dish"
    BRAND = "brand"
    POI_NAME = "poi_name"
    REFERENCE_POI = "reference_poi"
    DESTINATION_POI = "destination_poi"
    DISTRICT = "district"
    CITY = "city"
    STREET = "street"
    HOUSE_NUMBER = "house_number"
    ATTRIBUTES = "attributes"
    AMENITIES = "amenities"
    PRICE_MAX = "price_max"
    OPEN_NOW = "open_now"
    OPEN_LATE = "open_late"
    OPEN_24H = "open_24h"
    OPEN_AFTER = "open_after"
    OPEN_UNTIL = "open_until"
    OPENING_CONSTRAINT = "opening_constraint"
    REFERENCE_AREA = "reference_area"
    LOCATION_CONTEXT = "location_context"
    NAVIGATION_CUE = "navigation_cue"
    NEARBY_CUE = "nearby_cue"
    DISCOVERY_CUE = "discovery_cue"
    QUALITY = "quality"


class TraceExtractorName(StrEnum):
    COORDINATE_PARSER = "coordinate_parser"
    CATEGORY_LEXICON = "category_lexicon"
    DISH_LEXICON = "dish_lexicon"
    BRAND_LEXICON = "brand_lexicon"
    LOCAL_POI_INDEX = "local_poi_index"
    ADMINISTRATIVE_ALIAS = "administrative_alias"
    STREET_INDEX = "street_index"
    ADDRESS_PARSER = "address_parser"
    ATTRIBUTE_LEXICON = "attribute_lexicon"
    AMENITY_LEXICON = "amenity_lexicon"
    PRICE_PARSER = "price_parser"
    TIME_PARSER = "time_parser"
    REFERENCE_AREA_LEXICON = "reference_area_lexicon"
    REQUEST_CONTEXT = "request_context"
    LEXICON_REGISTRY = "lexicon_registry"
    GROUNDED_LLM = "grounded_llm"


class TraceRestorationStatus(StrEnum):
    PRESERVED = "preserved"
    RESTORED = "restored"
    FAILED = "failed"


class TraceValidationStatus(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"


class TraceTruncationReason(StrEnum):
    VARIANT_LIMIT = "variant_limit"
    EVIDENCE_LIMIT = "evidence_limit"
    CANDIDATE_LIMIT = "candidate_limit"


class TraceFallbackReason(StrEnum):
    UNKNOWN_SEARCH_TARGET = "unknown_search_target"
    MISSING_LOCATION_CONTEXT = "missing_location_context"
    AMBIGUOUS_TARGET = "ambiguous_target"
    UNSUPPORTED_QUERY = "unsupported_query"
    INVALID_QUERY = "invalid_query"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    BELOW_SCORE_THRESHOLD = "below_score_threshold"
    VALIDATION_FAILURE = "validation_failure"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class TraceAmbiguityReason(StrEnum):
    KNOWN_ALIAS_COLLISION = "known_alias_collision"
    UNRESOLVED_TARGET = "unresolved_target"
    SCORE_MARGIN = "score_margin"


class StrictTraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TraceGrounding(StrictTraceModel):
    source: str
    matched_value: str | bool | int | float | None = None


class TraceOperation(StrictTraceModel):
    operation: TraceOperationType
    source: str
    target: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    rule_id: str
    grounding: TraceGrounding | None = None
    confidence: float = Field(ge=0, le=1)
    parent_variant_id: str | None = None
    rewrite_cost: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> TraceOperation:
        if (self.start is None) != (self.end is None):
            raise ValueError("trace operation offsets must be provided together")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("trace operation end must not precede start")
        return self


class TraceProtectedSpan(StrictTraceModel):
    span_type: Literal["coordinate", "price", "time_or_number", "quoted", "exact_poi"]
    original_value: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    replacement_token: str | None = None
    restoration_status: TraceRestorationStatus


class TraceTelexGate(StrictTraceModel):
    attempted: bool
    accepted: bool
    vietnamese_diacritic_count: int = Field(ge=0)
    plausible_telex_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    candidate_ratio: float = Field(ge=0, le=1)
    reason: str
    result: str | None = None


class TraceStage(StrictTraceModel):
    stage: TraceStageName
    input: str
    output: str
    operations: list[TraceOperation] = Field(default_factory=list)
    protected_spans: list[TraceProtectedSpan] = Field(default_factory=list)
    telex_gate: TraceTelexGate | None = None


class TraceVariant(StrictTraceModel):
    id: str
    text: str
    source: TraceVariantSource
    parent_id: str | None = None
    cost: float = Field(ge=0)
    deduplication_key: str
    matching_key: str
    deduplicated: bool = False
    duplicate_of: str | None = None
    deduplication_reason: str | None = None
    selected_for_extraction: bool = False
    accepted: bool = True
    rejection_reason: str | None = None
    model_identifier: str | None = None
    adapter_version: str | None = None
    protected_span_validation_status: TraceValidationStatus | None = None

    @model_validator(mode="after")
    def validate_deduplication(self) -> TraceVariant:
        if self.deduplicated != (self.duplicate_of is not None):
            raise ValueError("deduplicated variants must identify duplicate_of")
        if self.deduplicated and self.deduplication_reason is None:
            raise ValueError("deduplicated variants must include a reason")
        if self.deduplicated:
            self.accepted = False
            self.rejection_reason = self.deduplication_reason
        elif not self.accepted and self.rejection_reason is None:
            raise ValueError("rejected variants must include a reason")
        return self


TraceValue = str | bool | int | float | list[str]


class TraceEvidence(StrictTraceModel):
    id: str
    type: TraceEvidenceType
    raw_value: TraceValue
    canonical_value: TraceValue
    field: str
    source_variant_id: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    extractor: TraceExtractorName
    rule_id: str
    confidence: float = Field(ge=0, le=1)
    configuration_source: str = "python_parser"
    generator_id: str | None = None
    precedence: int = Field(default=1, ge=1, le=7)
    accepted: bool = True
    rejection_reason: str | None = None
    canonical_merge_decision: str = "accepted"

    @model_validator(mode="after")
    def validate_offsets(self) -> TraceEvidence:
        if (self.start is None) != (self.end is None):
            raise ValueError("trace evidence offsets must be provided together")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("trace evidence end must not precede start")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("rejected evidence must include a rejection reason")
        return self


class TraceSemanticFrame(StrictTraceModel):
    id: str
    action: Action
    target_type: TargetKind
    spatial_relation: SpatialRelation
    search_style: SearchStyle
    extracted_fields: CanonicalEntities
    evidence_ids: list[str]
    validation_errors: list[str] = Field(default_factory=list)


class TraceCandidateScoreFeatures(StrictTraceModel):
    grounded_entity_support: float = Field(ge=0, le=1)
    required_slot_completeness: float = Field(ge=0, le=1)
    action_relation_match: float = Field(ge=0, le=1)
    rewrite_safety: float = Field(ge=0, le=1)
    modifier_coverage: float = Field(ge=0, le=1)
    location_context_consistency: float = Field(ge=0, le=1)
    model_confidence: float = Field(default=0, ge=0, le=1)
    deterministic_agreement: float = Field(default=0, ge=0, le=1)
    local_data_grounding: float = Field(default=0, ge=0, le=1)
    protected_span_consistency: float = Field(default=0, ge=0, le=1)
    correction_distance: float = Field(default=0, ge=0, le=1)
    semantic_compatibility: float = Field(default=0, ge=0, le=1)
    model_score_contribution: float = Field(default=0, ge=0, le=0.05)


class TraceCandidatePenalties(StrictTraceModel):
    ambiguity_penalty: float = Field(ge=0, le=1)
    unresolved_target_penalty: float = Field(ge=0, le=1)
    unsafe_rewrite_penalty: float = Field(ge=0, le=1)
    contradiction_penalty: float = Field(ge=0, le=1)
    hallucination_penalty: float = Field(ge=0, le=1)
    unsupported_entity_penalty: float = Field(default=0, ge=0, le=1)


class TraceCandidate(StrictTraceModel):
    id: str
    intent: Intent
    frame_id: str
    score: float = Field(ge=0, le=1)
    score_features: TraceCandidateScoreFeatures
    penalties: TraceCandidatePenalties
    rejection_reasons: list[str] = Field(default_factory=list)
    validation_status: TraceValidationStatus = TraceValidationStatus.VALID

    @model_validator(mode="after")
    def validate_reproducible_score(self) -> TraceCandidate:
        features = self.score_features
        expected = round(
            max(
                0.0,
                min(
                    1.0,
                    0.30 * features.grounded_entity_support
                    + 0.25 * features.required_slot_completeness
                    + 0.15 * features.action_relation_match
                    + 0.10 * features.rewrite_safety
                    + 0.10 * features.modifier_coverage
                    + 0.10 * features.location_context_consistency
                    + features.model_score_contribution
                    - self.penalties.ambiguity_penalty
                    - self.penalties.unresolved_target_penalty
                    - self.penalties.unsafe_rewrite_penalty
                    - self.penalties.contradiction_penalty
                    - self.penalties.hallucination_penalty
                    - self.penalties.unsupported_entity_penalty,
                ),
            ),
            4,
        )
        if not isclose(self.score, expected, abs_tol=1e-4):
            raise ValueError(f"candidate score {self.score} does not match features {expected}")
        return self


class TraceDecision(StrictTraceModel):
    selected_candidate_id: str
    selected_intent: Intent
    top_score: float = Field(ge=0, le=1)
    second_score: float | None = Field(default=None, ge=0, le=1)
    score_margin: float | None = Field(default=None, ge=0, le=1)
    ambiguity_threshold: float = Field(ge=0, le=1)
    is_ambiguous: bool
    ambiguity_reason: TraceAmbiguityReason | None = None
    fallback_triggered: bool = False
    fallback_reason: TraceFallbackReason | None = None

    @model_validator(mode="after")
    def validate_decision_reasons(self) -> TraceDecision:
        if self.fallback_triggered and self.fallback_reason is None:
            raise ValueError("fallback_reason is required when fallback is triggered")
        if not self.fallback_triggered and self.fallback_reason is not None:
            raise ValueError("fallback_reason requires fallback_triggered=true")
        if self.is_ambiguous and self.ambiguity_reason is None:
            raise ValueError("ambiguity_reason is required for an ambiguous decision")
        return self


class TraceModelCall(StrictTraceModel):
    adapter_name: str
    model_identifier: str
    adapter_version: str
    parent_variant_id: str | None = None
    proposal_type: Literal["correction", "semantic", "correction_and_semantic"]
    variant_text: str | None = None
    validation_result: TraceValidationStatus
    accepted_evidence: list[str] = Field(default_factory=list)
    rejected_fields: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    score_contribution: float = Field(default=0, ge=0, le=0.05)
    fallback_occurred: bool = False
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    correction_distance: float | None = Field(default=None, ge=0, le=1)
    protected_span_consistent: bool | None = None
    local_data_grounding: float | None = Field(default=None, ge=0, le=1)
    deterministic_agreement: float | None = Field(default=None, ge=0, le=1)


class QueryTrace(StrictTraceModel):
    trace_version: Literal["1.0"] = "1.0"
    trace_id: str
    original_query: str
    location_supplied: bool
    stages: list[TraceStage]
    variants: list[TraceVariant]
    evidence: list[TraceEvidence]
    semantic_decomposition: SemanticDecompositionResult
    semantic_implications: list[SemanticImplication] = Field(default_factory=list)
    search_expansions: list[SearchExpansion] = Field(default_factory=list)
    review_dependency_classifications: list[ReviewDependencyClassification] = Field(
        default_factory=list
    )
    social_discovery_decision: SocialDiscoveryDecision
    semantic_frames: list[TraceSemanticFrame]
    candidates: list[TraceCandidate]
    decision: TraceDecision
    model_calls: list[TraceModelCall] = Field(default_factory=list)
    trace_truncated: bool = False
    truncation_reason: TraceTruncationReason | None = None

    @model_validator(mode="after")
    def validate_references(self) -> QueryTrace:
        variant_ids = {item.id for item in self.variants}
        evidence_ids = {item.id for item in self.evidence}
        frame_ids = {item.id for item in self.semantic_frames}
        candidate_ids = {item.id for item in self.candidates}
        unit_variant_ids = {item.source_variant_id for item in self.semantic_decomposition.units}
        unit_ids = {item.id for item in self.semantic_decomposition.units}
        classification_ids = {item.id for item in self.review_dependency_classifications}
        if len(variant_ids) != len(self.variants):
            raise ValueError("trace variant IDs must be unique")
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("trace evidence IDs must be unique")
        if len(frame_ids) != len(self.semantic_frames):
            raise ValueError("trace frame IDs must be unique")
        if len(candidate_ids) != len(self.candidates):
            raise ValueError("trace candidate IDs must be unique")
        if unit_variant_ids - variant_ids:
            raise ValueError("semantic unit references an unknown variant")
        if any(item.source_unit_id not in unit_ids for item in self.semantic_implications):
            raise ValueError("semantic implication references an unknown unit")
        if len(classification_ids) != len(self.review_dependency_classifications):
            raise ValueError("review-dependency classification IDs must be unique")
        if any(
            item.source_unit_id not in unit_ids for item in self.review_dependency_classifications
        ):
            raise ValueError("review-dependency classification references an unknown unit")
        if any(set(item.source_unit_ids) - unit_ids for item in self.search_expansions):
            raise ValueError("search expansion references an unknown unit")
        if any(
            item.source_variant_id is not None and item.source_variant_id not in variant_ids
            for item in self.evidence
        ):
            raise ValueError("trace evidence references an unknown variant")
        if any(
            item.parent_id is not None and item.parent_id not in variant_ids
            for item in self.variants
        ):
            raise ValueError("trace variant references an unknown parent")
        if any(
            item.duplicate_of is not None and item.duplicate_of not in variant_ids
            for item in self.variants
        ):
            raise ValueError("trace duplicate references an unknown variant")
        if any(
            operation.parent_variant_id is not None
            and operation.parent_variant_id not in variant_ids
            for stage in self.stages
            for operation in stage.operations
        ):
            raise ValueError("trace operation references an unknown variant")
        if any(
            call.parent_variant_id is not None and call.parent_variant_id not in variant_ids
            for call in self.model_calls
        ):
            raise ValueError("trace model call references an unknown parent variant")
        if any(set(item.evidence_ids) - evidence_ids for item in self.semantic_frames):
            raise ValueError("trace frame references unknown evidence")
        if set(self.social_discovery_decision.triggering_unit_ids) - unit_ids:
            raise ValueError("social-discovery decision references an unknown unit")
        if set(self.social_discovery_decision.triggering_evidence_ids) - evidence_ids:
            raise ValueError("social-discovery decision references unknown evidence")
        if any(item.frame_id not in frame_ids for item in self.candidates):
            raise ValueError("trace candidate references an unknown frame")
        if self.decision.selected_candidate_id not in candidate_ids:
            raise ValueError("trace decision references an unknown candidate")
        if self.trace_truncated != (self.truncation_reason is not None):
            raise ValueError("trace truncation flag and reason must agree")
        return self


class QueryUnderstandTracedResponse(QueryUnderstandResponse):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "normalized_query": "Quán cà phê yên tĩnh",
                    "intent": "Category Search",
                    "entities": {
                        "category": "Quán cà phê",
                        "attributes": ["yên tĩnh"],
                    },
                    "trace": {
                        "trace_version": "1.0",
                        "trace_id": "4bd72f7f7c5044189cfc9cf9df07b42d",
                        "original_query": "cf yên tĩnh",
                        "location_supplied": False,
                        "stages": [],
                        "variants": [
                            {
                                "id": "v0",
                                "text": "cf yên tĩnh",
                                "source": "cleaned_original",
                                "cost": 0.0,
                                "deduplication_key": "cf yên tĩnh",
                                "matching_key": "cf yen tinh",
                                "deduplicated": False,
                                "selected_for_extraction": True,
                            }
                        ],
                        "evidence": [],
                        "semantic_decomposition": {
                            "units": [],
                            "grounded_concepts": [],
                            "unresolved_unit_ids": [],
                        },
                        "semantic_frames": [
                            {
                                "id": "frame-1",
                                "action": "SEARCH",
                                "target_type": "CATEGORY",
                                "spatial_relation": "NONE",
                                "search_style": "EXACT",
                                "extracted_fields": {"category": "Quán cà phê"},
                                "evidence_ids": [],
                                "validation_errors": [],
                            }
                        ],
                        "candidates": [
                            {
                                "id": "candidate_1",
                                "intent": "Category Search",
                                "frame_id": "frame-1",
                                "score": 1.0,
                                "score_features": {
                                    "grounded_entity_support": 1.0,
                                    "required_slot_completeness": 1.0,
                                    "action_relation_match": 1.0,
                                    "rewrite_safety": 1.0,
                                    "modifier_coverage": 1.0,
                                    "location_context_consistency": 1.0,
                                },
                                "penalties": {
                                    "ambiguity_penalty": 0.0,
                                    "unresolved_target_penalty": 0.0,
                                    "unsafe_rewrite_penalty": 0.0,
                                    "contradiction_penalty": 0.0,
                                    "hallucination_penalty": 0.0,
                                },
                                "rejection_reasons": [],
                                "validation_status": "valid",
                            }
                        ],
                        "decision": {
                            "selected_candidate_id": "candidate_1",
                            "selected_intent": "Category Search",
                            "top_score": 1.0,
                            "ambiguity_threshold": 0.12,
                            "is_ambiguous": False,
                            "fallback_triggered": False,
                        },
                        "trace_truncated": False,
                    },
                }
            ]
        },
    )

    trace: QueryTrace


QueryResponse = QueryUnderstandTracedResponse | QueryUnderstandResponse


class InterpretationResult(BaseModel):
    response: QueryResponse
    frame: SemanticFrame
    candidates: list[InterpretationCandidate]
    evidence: list[Evidence]
    protected_spans: list[ProtectedSpan]
    variants: list[QueryVariant]
    semantic_decomposition: SemanticDecompositionResult
    semantic_implications: list[SemanticImplication] = Field(default_factory=list)
    search_expansions: list[SearchExpansion] = Field(default_factory=list)
    review_dependency_classifications: list[ReviewDependencyClassification] = Field(
        default_factory=list
    )
    social_discovery_decision: SocialDiscoveryDecision
    trace_id: str
    pipeline_metrics: PipelineMetrics = Field(default_factory=lambda: PipelineMetrics())


class PipelineMetrics(BaseModel):
    """Request-local performance facts; never serialized in the public response."""

    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    rewrite_variants: int = Field(default=0, ge=0)
    hf_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    model_fallbacks: int = Field(default=0, ge=0)
    early_exit: bool = False
    semantic_skipped: bool = False
    cache_hit: bool = False
