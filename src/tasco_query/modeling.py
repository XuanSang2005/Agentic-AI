from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from src.tasco_query.adapters import (
    HARD_ENTITY_FIELDS,
    SOFT_MODEL_FIELDS,
    AdapterResult,
    ModelProposal,
    SemanticProposal,
    values_match,
)
from src.tasco_query.contracts import (
    Evidence,
    ProtectedSpan,
    QueryVariant,
    RewriteCandidate,
    RewriteEdit,
    SourceSpan,
    TraceExtractorName,
    TraceModelCall,
    TraceOperation,
    TraceOperationType,
    TraceStage,
    TraceStageName,
    TraceValidationStatus,
    TraceValue,
    TraceVariant,
    TraceVariantSource,
)
from src.tasco_query.extraction import EvidenceCollector, Extraction
from src.tasco_query.lexicon import LexiconRegistry
from src.tasco_query.normalization import comparison_key, match_key
from src.tasco_query.tracing import TraceCollector


@dataclass(slots=True)
class ModelScoreContext:
    model_confidence: float = 0.0
    deterministic_agreement: float = 0.0
    local_data_grounding: float = 0.0
    protected_span_consistency: float = 0.0
    correction_distance: float = 0.0
    semantic_compatibility: float = 0.0
    hallucination_penalty: float = 0.0
    unsupported_entity_penalty: float = 0.0

    @property
    def score_contribution(self) -> float:
        if self.model_confidence == 0:
            return 0.0
        positive = fmean(
            (
                self.model_confidence,
                self.deterministic_agreement,
                self.local_data_grounding,
                self.protected_span_consistency,
                1.0 - self.correction_distance,
                self.semantic_compatibility,
            )
        )
        return round(min(0.05, 0.05 * positive), 4)


def allowed_soft_values(registry: LexiconRegistry) -> dict[str, list[TraceValue]]:
    result: dict[str, list[TraceValue]] = {field: [] for field in sorted(SOFT_MODEL_FIELDS)}
    for entry in registry.entries:
        if entry.canonical_field not in result:
            continue
        value = entry.canonical
        if value not in result[entry.canonical_field]:
            result[entry.canonical_field].append(value)
    return result


def compact_local_matches(text: str, registry: LexiconRegistry) -> list[dict[str, Any]]:
    return [
        {
            "text": text[match.start : match.end],
            "field": match.entry.canonical_field,
            "canonical": match.entry.canonical,
            "rule_id": match.entry.rule_id,
            "source": match.entry.source,
        }
        for match in registry.phrase_matches(text)[:20]
    ]


def compact_evidence(extracted: Extraction) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "field": item.kind,
            "value": item.value,
            "confidence": item.confidence,
            "precedence": item.precedence,
            "accepted": item.accepted,
        }
        for item in extracted.evidence[:20]
    ]


def _hard_entities(
    text: str, registry: LexiconRegistry, *, exact_only: bool
) -> set[tuple[str, str]]:
    return {
        (match.entry.canonical_field, str(match.entry.canonical))
        for match in registry.phrase_matches(text, fields=HARD_ENTITY_FIELDS)
        if not exact_only
        or match_key(str(match.entry.canonical)) == match_key(text[match.start : match.end])
    }


def _model_source(proposal: ModelProposal) -> TraceVariantSource:
    return (
        TraceVariantSource.HUGGINGFACE_CORRECTION
        if proposal.adapter_name == "huggingface"
        else TraceVariantSource.GROUNDED_LLM
    )


def integrate_adapter_result(
    result: AdapterResult,
    *,
    original: str,
    parent_variant_id: str,
    protected: list[ProtectedSpan],
    registry: LexiconRegistry,
    variants: list[QueryVariant],
    limit: int,
    trace: TraceCollector | None,
) -> list[tuple[ModelProposal, TraceModelCall | None]]:
    integrated: list[tuple[ModelProposal, TraceModelCall | None]] = []
    if result.error and not result.proposals:
        if trace is not None:
            trace.add_model_call(
                TraceModelCall(
                    adapter_name=result.adapter_name,
                    model_identifier=result.model_identifier,
                    adapter_version=result.adapter_version,
                    parent_variant_id=parent_variant_id,
                    proposal_type="correction",
                    validation_result=TraceValidationStatus.REJECTED,
                    rejection_reasons=[result.error],
                    fallback_occurred=result.fallback_occurred,
                )
            )
        return integrated

    original_candidates = _hard_entities(original, registry, exact_only=False)
    exact_original = _hard_entities(original, registry, exact_only=True)
    field_counts = {
        field: sum(candidate_field == field for candidate_field, _ in original_candidates)
        for field, _ in original_candidates
    }
    original_hard = exact_original | {
        item for item in original_candidates if field_counts[item[0]] == 1
    }
    existing = {comparison_key(item.text): item.variant_id for item in variants}
    for index, proposal in enumerate(result.proposals, 1):
        proposal.parent_variant_id = parent_variant_id
        reasons: list[str] = []
        if not proposal.protected_span_valid:
            reasons.append(proposal.rejection_reason or "protected_span_mutation")
        proposed_hard = _hard_entities(proposal.text, registry, exact_only=True)
        if proposed_hard - original_hard:
            reasons.append("invented_hard_entity")
        key = comparison_key(proposal.text)
        duplicate_of = existing.get(key)
        if duplicate_of is not None:
            reasons.append("duplicate_model_variant")
        if len(variants) >= limit and duplicate_of is None:
            reasons.append("variant_limit")
        correction_accepted = not reasons or reasons == ["duplicate_model_variant"]
        blocking_reasons = set(reasons) - {"duplicate_model_variant"}
        if proposal.semantic_evidence:
            blocking_reasons.discard("variant_limit")
        accepted = correction_accepted or bool(proposal.semantic_evidence and not blocking_reasons)
        variant_id = duplicate_of or f"v{len(variants)}"
        if duplicate_of is not None:
            proposal.variant_id = duplicate_of
        if trace is not None:
            trace.add_variant(
                TraceVariant(
                    id=(
                        f"model-{proposal.adapter_name}-{parent_variant_id}-{index}"
                        if reasons or duplicate_of is not None
                        else variant_id
                    ),
                    text=proposal.text,
                    source=_model_source(proposal),
                    parent_id=proposal.parent_variant_id,
                    cost=proposal.correction_cost,
                    deduplication_key=key,
                    matching_key=match_key(proposal.text),
                    deduplicated=duplicate_of is not None,
                    duplicate_of=duplicate_of,
                    deduplication_reason=("duplicate_model_variant" if duplicate_of else None),
                    accepted=accepted,
                    rejection_reason=";".join(reasons) if reasons else None,
                    model_identifier=proposal.model_identifier,
                    adapter_version=proposal.adapter_version,
                    protected_span_validation_status=(
                        TraceValidationStatus.VALID
                        if proposal.protected_span_valid
                        else TraceValidationStatus.REJECTED
                    ),
                )
            )
        if not reasons and duplicate_of is None:
            generation = RewriteCandidate(
                source_text=original,
                proposed_text=proposal.text,
                transformation_type=proposal.adapter_name,
                rule_id=f"model.{proposal.adapter_name}.proposal",
                source_span=SourceSpan(start=0, end=len(original), text=original),
                grounding_source=proposal.model_identifier,
                matched_lexicon_item="model_proposal",
                confidence=proposal.confidence,
                cost=proposal.correction_cost,
                parent_variant_id=proposal.parent_variant_id,
            )
            variants.append(
                QueryVariant(
                    variant_id=variant_id,
                    text=proposal.text,
                    source=_model_source(proposal).value,
                    prior=proposal.confidence,
                    rewrite_cost=proposal.correction_cost,
                    edits=[
                        RewriteEdit(
                            source=original,
                            replacement=proposal.text,
                            reason=generation.rule_id,
                        )
                    ],
                    hard_locks_preserved=proposal.protected_span_valid,
                    parent_id=proposal.parent_variant_id,
                    generation=[generation],
                    model_identifier=proposal.model_identifier,
                    adapter_version=proposal.adapter_version,
                    protected_span_validation_status="valid",
                )
            )
            proposal.variant_id = variant_id
            existing[key] = variant_id
        call: TraceModelCall | None = None
        if trace is not None:
            call = TraceModelCall(
                adapter_name=proposal.adapter_name,
                model_identifier=proposal.model_identifier,
                adapter_version=proposal.adapter_version,
                parent_variant_id=proposal.parent_variant_id,
                proposal_type=(
                    "correction_and_semantic" if proposal.semantic_evidence else "correction"
                ),
                variant_text=proposal.text,
                validation_result=(
                    TraceValidationStatus.VALID
                    if accepted and not blocking_reasons
                    else TraceValidationStatus.REJECTED
                ),
                rejection_reasons=reasons,
                fallback_occurred=bool(blocking_reasons),
                model_confidence=proposal.confidence,
                correction_distance=proposal.correction_cost,
                protected_span_consistent=proposal.protected_span_valid,
            )
            trace.add_model_call(call)
        if accepted:
            integrated.append((proposal, call))
    return integrated


def _allowed_constraint(constraint: SemanticProposal, allowed: dict[str, list[TraceValue]]) -> bool:
    return constraint.field in allowed and any(
        values_match(constraint.value, candidate) for candidate in allowed[constraint.field]
    )


def _explicit_amenity_value(text: str, value: TraceValue, registry: LexiconRegistry) -> bool:
    return any(
        values_match(value, match.entry.canonical)
        for match in registry.phrase_matches(text, fields={"amenities"})
    )


def _merge_soft(entities: dict[str, Any], field: str, value: TraceValue) -> bool:
    if isinstance(value, list):
        target = entities.setdefault(field, [])
        if not isinstance(target, list):
            return False
        target.extend(item for item in value if item not in target)
        return True
    if field in entities:
        return bool(entities[field] == value)
    entities[field] = value
    return True


def apply_soft_semantics(
    extracted: Extraction,
    proposals: list[tuple[ModelProposal, TraceModelCall | None]],
    *,
    original_variant: QueryVariant,
    variants: list[QueryVariant],
    registry: LexiconRegistry,
    trace: TraceCollector | None,
) -> ModelScoreContext:
    allowed = allowed_soft_values(registry)
    collector = EvidenceCollector(trace)
    confidences: list[float] = []
    distances: list[float] = []
    agreements: list[float] = []
    groundings: list[float] = []
    compatible: list[float] = []
    protected: list[float] = []
    unsupported = hallucinations = 0
    total_fields = 0
    variant_by_id = {variant.variant_id: variant for variant in variants}
    deterministic_values = {
        (item.kind, str(item.value))
        for item in extracted.evidence
        if item.accepted
        and variant_by_id.get(item.variant_id, original_variant).model_identifier is None
    }

    for proposal, call in proposals:
        confidences.append(proposal.confidence)
        distances.append(proposal.correction_cost)
        protected.append(float(proposal.protected_span_valid))
        source_variant = variant_by_id.get(proposal.parent_variant_id, original_variant)
        proposal_variant = variant_by_id.get(proposal.variant_id or "")
        extracted_from_variant = [
            item
            for item in extracted.evidence
            if item.accepted
            and proposal_variant is not None
            and item.variant_id == proposal_variant.variant_id
            and proposal_variant.model_identifier == proposal.model_identifier
        ]
        if extracted_from_variant:
            groundings.append(1.0)
            compatible.append(1.0)
            agreements.extend(
                1.0 if (item.kind, str(item.value)) in deterministic_values else 0.5
                for item in extracted_from_variant
            )
            if call is not None:
                call.accepted_evidence.extend(
                    item.evidence_id
                    for item in extracted_from_variant
                    if item.evidence_id not in call.accepted_evidence
                )
        for hard in proposal.hard_entities:
            total_fields += 1
            actual = extracted.entities.get(hard.field)
            if actual is None or not values_match(actual, hard.value):
                hallucinations += 1
                if call is not None:
                    call.rejected_fields.append(hard.field)
                    call.rejection_reasons.append("ungrounded_hard_entity")
                continue
            agreements.append(1.0)
        accepted_for_proposal = False
        for constraint in proposal.semantic_evidence:
            total_fields += 1
            evidence_grounded = match_key(constraint.evidence_text) in match_key(
                source_variant.text
            )
            amenity_explicit = constraint.field != "amenities" or _explicit_amenity_value(
                source_variant.text, constraint.value, registry
            )
            if (
                not _allowed_constraint(constraint, allowed)
                or not evidence_grounded
                or not amenity_explicit
            ):
                unsupported += 1
                if call is not None:
                    call.rejected_fields.append(constraint.field)
                    call.rejection_reasons.append(
                        "unsupported_canonical_value"
                        if not _allowed_constraint(constraint, allowed)
                        else (
                            "evidence_text_not_in_query"
                            if not evidence_grounded
                            else "amenity_not_explicitly_requested"
                        )
                    )
                continue
            prior = extracted.entities.get(constraint.field)
            agreement = 1.0 if prior is not None and values_match(prior, constraint.value) else 0.5
            accepted = _merge_soft(extracted.entities, constraint.field, constraint.value)
            evidence: Evidence = collector.add(
                kind=constraint.field,
                value=constraint.value,
                variant=source_variant,
                start=None,
                end=None,
                raw_value=constraint.evidence_text,
                rule_id=f"model.semantic.{constraint.field}",
                config_source=proposal.model_identifier,
                confidence=constraint.confidence,
                precedence=7,
                accepted=accepted,
                rejection_reason=None if accepted else "deterministic_evidence_precedence",
                merge_decision=(
                    "accepted_model_soft_evidence"
                    if accepted
                    else "retained_stronger_deterministic_evidence"
                ),
                extractor=TraceExtractorName.GROUNDED_LLM,
            )
            if call is not None:
                if accepted:
                    call.accepted_evidence.append(evidence.evidence_id)
                else:
                    call.rejected_fields.append(constraint.field)
            accepted_for_proposal |= accepted
            agreements.append(agreement)
            groundings.append(1.0)
            compatible.append(1.0)
        if accepted_for_proposal and proposal.intent == "Discovery Search":
            extracted.discovery = True
        if proposal.intent == "Ambiguous" and proposal.ambiguity_type in {
            "lexical_ambiguity",
            "target_ambiguity",
            "location_ambiguity",
            "constraint_ambiguity",
        }:
            extracted.entities.setdefault("ambiguity_type", proposal.ambiguity_type)
    extracted.evidence.extend(collector.items)
    for variant_id, spans in collector.covered_spans.items():
        extracted.covered_spans.setdefault(variant_id, set()).update(spans)
    context = ModelScoreContext(
        model_confidence=fmean(confidences) if confidences else 0.0,
        deterministic_agreement=fmean(agreements) if agreements else 0.0,
        local_data_grounding=fmean(groundings) if groundings else 0.0,
        protected_span_consistency=fmean(protected) if protected else 0.0,
        correction_distance=fmean(distances) if distances else 0.0,
        semantic_compatibility=fmean(compatible) if compatible else 0.0,
        hallucination_penalty=hallucinations / total_fields if total_fields else 0.0,
        unsupported_entity_penalty=unsupported / total_fields if total_fields else 0.0,
    )
    for _, call in proposals:
        if call is not None:
            call.local_data_grounding = context.local_data_grounding
            call.deterministic_agreement = context.deterministic_agreement
            call.score_contribution = context.score_contribution
            blocking_reasons = set(call.rejection_reasons) - {"duplicate_model_variant"}
            if "ungrounded_hard_entity" in call.rejection_reasons or (
                blocking_reasons and not call.accepted_evidence
            ):
                call.validation_result = TraceValidationStatus.REJECTED
                call.fallback_occurred = True
    if trace is not None and proposals:
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.MODEL_ASSISTANCE,
                input=original_variant.text,
                output=str(extracted.entities),
                operations=[
                    TraceOperation(
                        operation=(
                            TraceOperationType.MODEL_SEMANTIC_EXTRACTION
                            if proposal.semantic_evidence
                            else TraceOperationType.MODEL_CORRECTION
                        ),
                        source=proposal.adapter_name,
                        target=proposal.text,
                        rule_id="model.validated_proposal",
                        confidence=proposal.confidence,
                        parent_variant_id=proposal.parent_variant_id,
                        rewrite_cost=proposal.correction_cost,
                    )
                    for proposal, _ in proposals
                ],
            )
        )
    return context
