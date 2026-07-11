from __future__ import annotations

from src.tasco_query.contracts import (
    CanonicalEntities,
    InterpretationCandidate,
    QueryTrace,
    ReviewDependencyClassification,
    SearchExpansion,
    SemanticDecompositionResult,
    SemanticFrame,
    SemanticImplication,
    SocialDiscoveryDecision,
    TargetKind,
    TraceAmbiguityReason,
    TraceCandidate,
    TraceCandidatePenalties,
    TraceCandidateScoreFeatures,
    TraceDecision,
    TraceEvidence,
    TraceEvidenceType,
    TraceExtractorName,
    TraceFallbackReason,
    TraceGrounding,
    TraceModelCall,
    TraceOperation,
    TraceOperationType,
    TraceSemanticFrame,
    TraceStage,
    TraceStageName,
    TraceTruncationReason,
    TraceValidationStatus,
    TraceValue,
    TraceVariant,
)


class TraceCollector:
    """Bounded request-local sink for structured pipeline facts."""

    def __init__(
        self,
        *,
        trace_id: str,
        original_query: str,
        location_supplied: bool,
        max_variants: int,
        max_evidence: int,
        max_candidates: int,
    ) -> None:
        self.trace_id = trace_id
        self.original_query = original_query
        self.location_supplied = location_supplied
        self.max_variants = max(1, max_variants)
        self.max_evidence = max(1, max_evidence)
        self.max_candidates = max(1, max_candidates)
        self.stages: list[TraceStage] = []
        self.variants: list[TraceVariant] = []
        self.evidence: list[TraceEvidence] = []
        self.semantic_decomposition = SemanticDecompositionResult()
        self.semantic_implications: list[SemanticImplication] = []
        self.search_expansions: list[SearchExpansion] = []
        self.review_dependency_classifications: list[ReviewDependencyClassification] = []
        self.social_discovery_decision: SocialDiscoveryDecision | None = None
        self.frames: list[TraceSemanticFrame] = []
        self.candidates: list[TraceCandidate] = []
        self.model_calls: list[TraceModelCall] = []
        self.decision: TraceDecision | None = None
        self.truncation_reason: TraceTruncationReason | None = None
        self._final_stage: TraceStage | None = None

    def _truncate(self, reason: TraceTruncationReason) -> None:
        if self.truncation_reason is None:
            self.truncation_reason = reason

    def add_stage(self, stage: TraceStage) -> None:
        self.stages.append(stage)

    def set_final_stage(self, stage: TraceStage) -> None:
        self._final_stage = stage

    def add_variant(self, variant: TraceVariant) -> None:
        if len(self.variants) >= self.max_variants:
            self._truncate(TraceTruncationReason.VARIANT_LIMIT)
            return
        self.variants.append(variant)

    def mark_variants_selected(self, variant_ids: set[str]) -> None:
        for variant in self.variants:
            variant.selected_for_extraction = not variant.deduplicated and variant.id in variant_ids

    def add_evidence(
        self,
        *,
        evidence_type: TraceEvidenceType,
        raw_value: TraceValue,
        canonical_value: TraceValue,
        field: str,
        source_variant_id: str | None,
        extractor: TraceExtractorName,
        rule_id: str,
        confidence: float,
        start: int | None = None,
        end: int | None = None,
        configuration_source: str = "python_parser",
        generator_id: str | None = None,
        precedence: int = 1,
        accepted: bool = True,
        rejection_reason: str | None = None,
        canonical_merge_decision: str = "accepted",
    ) -> str | None:
        known_variants = {item.id for item in self.variants}
        if source_variant_id is not None and source_variant_id not in known_variants:
            self._truncate(TraceTruncationReason.VARIANT_LIMIT)
            return None
        if len(self.evidence) >= self.max_evidence:
            self._truncate(TraceTruncationReason.EVIDENCE_LIMIT)
            return None
        evidence_id = f"evidence-{len(self.evidence) + 1}"
        self.evidence.append(
            TraceEvidence(
                id=evidence_id,
                type=evidence_type,
                raw_value=raw_value,
                canonical_value=canonical_value,
                field=field,
                source_variant_id=source_variant_id,
                start=start,
                end=end,
                extractor=extractor,
                rule_id=rule_id,
                confidence=confidence,
                configuration_source=configuration_source,
                generator_id=generator_id,
                precedence=precedence,
                accepted=accepted,
                rejection_reason=rejection_reason,
                canonical_merge_decision=canonical_merge_decision,
            )
        )
        return evidence_id

    def add_model_call(self, call: TraceModelCall) -> None:
        known_variants = {item.id for item in self.variants}
        if call.parent_variant_id is not None and call.parent_variant_id not in known_variants:
            self._truncate(TraceTruncationReason.VARIANT_LIMIT)
            return
        self.model_calls.append(call)

    def set_semantic_decomposition(self, result: SemanticDecompositionResult) -> None:
        known_variants = {item.id for item in self.variants}
        units = [item for item in result.units if item.source_variant_id in known_variants]
        unit_ids = {item.id for item in units}
        concepts = [item for item in result.grounded_concepts if item.source_unit_id in unit_ids]
        concept_ids = {item.id for item in concepts}
        units = [
            item.model_copy(
                update={
                    "grounding_ids": [
                        grounding_id
                        for grounding_id in item.grounding_ids
                        if grounding_id in concept_ids
                    ],
                    "directly_grounded": any(
                        grounding_id in concept_ids for grounding_id in item.grounding_ids
                    ),
                }
            )
            for item in units
        ]
        self.semantic_decomposition = SemanticDecompositionResult(
            units=units,
            grounded_concepts=concepts,
            unresolved_unit_ids=[item.id for item in units if not item.directly_grounded],
        )
        self.add_stage(
            TraceStage(
                stage=TraceStageName.SEMANTIC_DECOMPOSITION,
                input=" | ".join(item.text for item in units),
                output=" | ".join(item.id for item in units),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.SEMANTIC_UNIT_SEGMENTATION,
                        source=item.text,
                        target=f"{item.id}:{item.unit_type.value}",
                        start=item.start,
                        end=item.end,
                        rule_id="semantics.phrase_aware_longest_match.v1",
                        grounding=TraceGrounding(source="variant_and_registry_spans"),
                        confidence=1.0,
                        parent_variant_id=item.source_variant_id,
                    )
                    for item in units
                ],
            )
        )

        units_by_id = {item.id: item for item in units}
        operations = [
            TraceOperation(
                operation=TraceOperationType.DIRECT_CONCEPT_GROUNDING,
                source=units_by_id[item.source_unit_id].text,
                target=f"{item.field}={item.canonical_value}",
                start=units_by_id[item.source_unit_id].start,
                end=units_by_id[item.source_unit_id].end,
                rule_id=item.rule_id or "semantics.direct_grounding.v1",
                grounding=TraceGrounding(source=item.source),
                confidence=item.confidence,
                parent_variant_id=units_by_id[item.source_unit_id].source_variant_id,
            )
            for item in concepts
        ]
        operations.extend(
            TraceOperation(
                operation=TraceOperationType.UNRESOLVED_UNIT_ROUTING,
                source=units_by_id[unit_id].text,
                target=unit_id,
                start=units_by_id[unit_id].start,
                end=units_by_id[unit_id].end,
                rule_id="semantics.route_unresolved.v1",
                grounding=TraceGrounding(source="deeper_semantic_inference_queue"),
                confidence=1.0,
                parent_variant_id=units_by_id[unit_id].source_variant_id,
            )
            for unit_id in self.semantic_decomposition.unresolved_unit_ids
        )
        self.add_stage(
            TraceStage(
                stage=TraceStageName.DIRECT_CONCEPT_GROUNDING,
                input=" | ".join(item.id for item in units),
                output=" | ".join(item.id for item in concepts),
                operations=operations,
            )
        )

    def set_semantic_implications(
        self,
        implications: list[SemanticImplication],
        expansions: list[SearchExpansion],
    ) -> None:
        self.semantic_implications = implications
        self.search_expansions = expansions
        self.add_stage(
            TraceStage(
                stage=TraceStageName.SEMANTIC_IMPLICATION,
                input=" | ".join(item.source_unit_id for item in implications),
                output=" | ".join(item.id for item in implications),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.SEMANTIC_IMPLICATION,
                        source=item.source_unit_id,
                        target=f"{item.field}={item.value}",
                        rule_id="semantic_implication.validated_mapping.v1",
                        grounding=TraceGrounding(source=item.grounding[0]),
                        confidence=item.confidence,
                    )
                    for item in implications
                ],
            )
        )
        self.add_stage(
            TraceStage(
                stage=TraceStageName.SEARCH_EXPANSION,
                input=" | ".join(item.id for item in implications),
                output=" | ".join(item.id for item in expansions),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.SEARCH_EXPANSION,
                        source=",".join(item.source_unit_ids),
                        target=item.text,
                        rule_id="search_expansion.bounded_mapping.v1",
                        grounding=TraceGrounding(source=item.grounding[0]),
                        confidence=item.confidence,
                    )
                    for item in expansions
                ],
            )
        )

    def set_social_discovery(
        self,
        classifications: list[ReviewDependencyClassification],
        decision: SocialDiscoveryDecision,
    ) -> None:
        known_unit_ids = {item.id for item in self.semantic_decomposition.units}
        known_evidence_ids = {item.id for item in self.evidence}
        classifications = [
            item.model_copy(
                update={
                    "evidence_ids": [
                        evidence_id
                        for evidence_id in item.evidence_ids
                        if evidence_id in known_evidence_ids
                    ]
                }
            )
            for item in classifications
            if item.source_unit_id in known_unit_ids
        ]
        decision = decision.model_copy(
            update={
                "triggering_unit_ids": [
                    unit_id for unit_id in decision.triggering_unit_ids if unit_id in known_unit_ids
                ],
                "triggering_evidence_ids": [
                    evidence_id
                    for evidence_id in decision.triggering_evidence_ids
                    if evidence_id in known_evidence_ids
                ],
            }
        )
        self.review_dependency_classifications = classifications
        self.social_discovery_decision = decision
        self.add_stage(
            TraceStage(
                stage=TraceStageName.REVIEW_DEPENDENCY_CLASSIFICATION,
                input=" | ".join(item.source_unit_id for item in classifications),
                output=" | ".join(item.id for item in classifications),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.REVIEW_DEPENDENCY_CLASSIFICATION,
                        source=item.source_unit_id,
                        target=item.review_dependency.value,
                        rule_id="review_dependency.structured_classification.v1",
                        grounding=TraceGrounding(source=item.reason),
                        confidence=item.confidence,
                    )
                    for item in classifications
                ],
            )
        )
        self.add_stage(
            TraceStage(
                stage=TraceStageName.SOCIAL_DISCOVERY_GATE,
                input=" | ".join(item.id for item in classifications),
                output=decision.model_dump_json(),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.SOCIAL_DISCOVERY_GATE,
                        source="semantic_frame_and_review_dependency",
                        target=str(decision.should_trigger).lower(),
                        rule_id="social_discovery.deterministic_gate.v1",
                        grounding=TraceGrounding(source=decision.reason),
                        confidence=decision.confidence,
                    )
                ],
            )
        )

    def add_frame(self, frame: SemanticFrame) -> str:
        frame_id = f"frame-{len(self.frames) + 1}"
        self.frames.append(
            TraceSemanticFrame(
                id=frame_id,
                action=frame.action,
                target_type=frame.target_kind,
                spatial_relation=frame.spatial_relation,
                search_style=frame.search_style,
                extracted_fields=CanonicalEntities.model_validate(frame.entities),
                evidence_ids=[item.id for item in self.evidence if item.accepted],
                validation_errors=[],
            )
        )
        return frame_id

    def add_candidate(self, candidate: InterpretationCandidate, frame_id: str) -> str | None:
        if len(self.candidates) >= self.max_candidates:
            self._truncate(TraceTruncationReason.CANDIDATE_LIMIT)
            return None
        score = candidate.score
        self.candidates.append(
            TraceCandidate(
                id=candidate.candidate_id,
                intent=candidate.intent,
                frame_id=frame_id,
                score=score.total,
                score_features=TraceCandidateScoreFeatures(
                    grounded_entity_support=score.grounded_entity_support,
                    required_slot_completeness=score.required_slot_completeness,
                    action_relation_match=score.action_relation_match,
                    rewrite_safety=score.rewrite_safety,
                    modifier_coverage=score.modifier_coverage,
                    location_context_consistency=score.location_context_consistency,
                    model_confidence=score.model_confidence,
                    deterministic_agreement=score.deterministic_agreement,
                    local_data_grounding=score.local_data_grounding,
                    protected_span_consistency=score.protected_span_consistency,
                    correction_distance=score.correction_distance,
                    semantic_compatibility=score.semantic_compatibility,
                    model_score_contribution=score.model_score_contribution,
                ),
                penalties=TraceCandidatePenalties(
                    ambiguity_penalty=score.ambiguity_penalty,
                    unresolved_target_penalty=score.unresolved_target_penalty,
                    unsafe_rewrite_penalty=score.unsafe_rewrite_penalty,
                    contradiction_penalty=score.contradiction_penalty,
                    hallucination_penalty=score.hallucination_penalty,
                    unsupported_entity_penalty=score.unsupported_entity_penalty,
                ),
                rejection_reasons=[],
                validation_status=TraceValidationStatus.VALID,
            )
        )
        return candidate.candidate_id

    def set_decision(
        self,
        candidate: InterpretationCandidate,
        *,
        known_alias_collision: bool,
    ) -> None:
        unresolved_target = candidate.frame.target_kind == TargetKind.UNKNOWN
        is_ambiguous = candidate.intent == "Ambiguous" and known_alias_collision
        ambiguity_reason = None
        if is_ambiguous:
            ambiguity_reason = TraceAmbiguityReason.KNOWN_ALIAS_COLLISION
        fallback_reason = (
            TraceFallbackReason.UNKNOWN_SEARCH_TARGET
            if unresolved_target and not is_ambiguous
            else None
        )
        self.decision = TraceDecision(
            selected_candidate_id=candidate.candidate_id,
            selected_intent=candidate.intent,
            top_score=candidate.score.total,
            second_score=None,
            score_margin=None,
            ambiguity_threshold=0.12,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            fallback_triggered=fallback_reason is not None,
            fallback_reason=fallback_reason,
        )
        self.add_stage(
            TraceStage(
                stage=TraceStageName.AMBIGUITY_DECISION,
                input=candidate.score.model_dump_json(),
                output=self.decision.model_dump_json(),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.AMBIGUITY_EVALUATION,
                        source=str(candidate.score.total),
                        target=str(is_ambiguous).lower(),
                        rule_id="decision.ambiguity_policy.v1",
                        grounding=TraceGrounding(source="score_and_grounded_alias_policy"),
                        confidence=1.0,
                    )
                ],
            )
        )

    def build(self) -> QueryTrace:
        if self.decision is None:
            raise RuntimeError("trace decision was not recorded")
        if self.social_discovery_decision is None:
            raise RuntimeError("social-discovery decision was not recorded")
        stages = [*self.stages]
        if self._final_stage is not None:
            stages.append(self._final_stage)
        return QueryTrace(
            trace_id=self.trace_id,
            original_query=self.original_query,
            location_supplied=self.location_supplied,
            stages=stages,
            variants=self.variants,
            evidence=self.evidence,
            semantic_decomposition=self.semantic_decomposition,
            semantic_implications=self.semantic_implications,
            search_expansions=self.search_expansions,
            review_dependency_classifications=self.review_dependency_classifications,
            social_discovery_decision=self.social_discovery_decision,
            semantic_frames=self.frames,
            candidates=self.candidates,
            decision=self.decision,
            model_calls=self.model_calls,
            trace_truncated=self.truncation_reason is not None,
            truncation_reason=self.truncation_reason,
        )
