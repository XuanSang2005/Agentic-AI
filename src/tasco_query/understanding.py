from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any, cast

from src.tasco_query.contracts import (
    Action,
    CanonicalEntities,
    Intent,
    InterpretationCandidate,
    ScoreBreakdown,
    SearchStyle,
    SemanticFrame,
    SpatialRelation,
    TargetKind,
    TraceGrounding,
    TraceOperation,
    TraceOperationType,
    TraceStage,
    TraceStageName,
)
from src.tasco_query.extraction import Extraction
from src.tasco_query.lexicon import LexiconRegistry
from src.tasco_query.normalization import match_key, title_vi

if TYPE_CHECKING:
    from src.tasco_query.modeling import ModelScoreContext
    from src.tasco_query.tracing import TraceCollector


def derive_intent(query: str, extracted: Extraction, registry: LexiconRegistry) -> Intent:
    key = match_key(query)
    if key in registry.ambiguities:
        return "Ambiguous"
    entities = extracted.entities
    if extracted.navigation:
        return "Navigation"
    if extracted.coordinate_only:
        return "Coordinate Search"
    if "house_number" in entities and "street" in entities and not extracted.nearby:
        return "Address Search"
    if extracted.nearby or (entities.get("category") == "ATM" and "district" in entities):
        if extracted.explicit_reference and entities.get("attributes") and "dish" not in entities:
            return "Category Search"
        return "Nearby Search"
    if "poi_name" in entities:
        return "POI Search"
    if "brand" in entities and "category" in entities:
        return "Brand Category Search"
    if extracted.discovery:
        return "Discovery Search"
    if "category" in entities or "dish" in entities or "cuisine" in entities:
        return "Category Search"
    return "Ambiguous"


def build_frame(
    intent: Intent,
    extracted: Extraction,
    trace: TraceCollector | None = None,
) -> tuple[SemanticFrame, str | None]:
    entities = extracted.entities
    if extracted.navigation:
        action = Action.NAVIGATE
    elif extracted.coordinate_only:
        action = Action.LOCATE_COORDINATE
    else:
        action = Action.SEARCH
    if "poi_name" in entities or "destination_poi" in entities:
        target = TargetKind.POI
    elif "house_number" in entities and "street" in entities:
        target = TargetKind.ADDRESS
    elif "dish" in entities:
        target = TargetKind.DISH
    elif "category" in entities or "destination_category" in entities:
        target = TargetKind.CATEGORY
    elif "cuisine" in entities:
        target = TargetKind.CUISINE
    elif "brand" in entities:
        target = TargetKind.BRAND
    elif "latitude" in entities:
        target = TargetKind.COORDINATE
    else:
        target = TargetKind.UNKNOWN
    if entities.get("location") == "current_location":
        spatial = SpatialRelation.NEAR_CURRENT
    elif "reference_poi" in entities or (
        "reference_area" in entities and extracted.explicit_reference
    ):
        spatial = SpatialRelation.NEAR_REFERENCE
    elif "street" in entities and "dish" in entities:
        spatial = SpatialRelation.ON_STREET
    elif "district" in entities or "city" in entities or "reference_area" in entities:
        spatial = SpatialRelation.WITHIN_AREA
    else:
        spatial = SpatialRelation.NONE
    style = (
        SearchStyle.DISCOVERY
        if intent == "Discovery Search" or entities.get("attributes")
        else SearchStyle.EXACT
    )
    frame = SemanticFrame(
        action=action,
        target_kind=target,
        spatial_relation=spatial,
        search_style=style,
        entities=entities,
        evidence_ids=[evidence.evidence_id for evidence in extracted.evidence if evidence.accepted],
    )
    frame_id = trace.add_frame(frame) if trace is not None else None
    if trace is not None:
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.SEMANTIC_FRAME_CONSTRUCTION,
                input=CanonicalEntities.model_validate(entities).model_dump_json(exclude_none=True),
                output=frame.model_dump_json(),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.FRAME_CONSTRUCTION,
                        source=intent,
                        target=frame_id or "",
                        rule_id="frame.compositional_mapping",
                        grounding=TraceGrounding(source="validated_extraction"),
                        confidence=1.0,
                    )
                ],
            )
        )
    return frame, frame_id


def score_candidate(
    intent: Intent,
    frame: SemanticFrame,
    model_context: ModelScoreContext | None = None,
) -> ScoreBreakdown:
    unresolved = frame.target_kind == TargetKind.UNKNOWN
    grounded = 0.0 if unresolved else 1.0
    complete = 0.0 if unresolved else 1.0
    if intent == "Navigation" and not (
        frame.entities.get("destination_poi") or frame.entities.get("destination_category")
    ):
        complete = 0.55
    cue_match = 1.0 if intent != "Ambiguous" else 0.5
    rewrite_safety = 1.0
    ambiguity_penalty = 0.3 if intent == "Ambiguous" else 0.0
    modifier_keys = {
        "attributes",
        "amenities",
        "quality",
        "price_max",
        "open_now",
        "open_late",
        "open_24h",
        "open_after",
        "open_until",
    }
    modifier_coverage = 1.0 if modifier_keys & frame.entities.keys() else 0.5
    location_context_consistency = 1.0
    unresolved_target_penalty = 0.25 if unresolved else 0.0
    unsafe_rewrite_penalty = contradiction_penalty = 0.0
    hallucination_penalty = model_context.hallucination_penalty if model_context else 0.0
    unsupported_entity_penalty = model_context.unsupported_entity_penalty if model_context else 0.0
    model_score_contribution = model_context.score_contribution if model_context else 0.0
    total = max(
        0.0,
        min(
            1.0,
            0.30 * grounded
            + 0.25 * complete
            + 0.15 * cue_match
            + 0.10 * rewrite_safety
            + 0.10 * modifier_coverage
            + 0.10 * location_context_consistency
            + model_score_contribution
            - ambiguity_penalty
            - unresolved_target_penalty
            - unsafe_rewrite_penalty
            - contradiction_penalty
            - hallucination_penalty
            - unsupported_entity_penalty,
        ),
    )
    return ScoreBreakdown(
        grounded_entity_support=grounded,
        required_slot_completeness=complete,
        action_relation_match=cue_match,
        rewrite_safety=rewrite_safety,
        modifier_coverage=modifier_coverage,
        location_context_consistency=location_context_consistency,
        ambiguity_penalty=ambiguity_penalty,
        unresolved_target_penalty=unresolved_target_penalty,
        unsafe_rewrite_penalty=unsafe_rewrite_penalty,
        contradiction_penalty=contradiction_penalty,
        hallucination_penalty=hallucination_penalty,
        model_confidence=model_context.model_confidence if model_context else 0.0,
        deterministic_agreement=(model_context.deterministic_agreement if model_context else 0.0),
        local_data_grounding=model_context.local_data_grounding if model_context else 0.0,
        protected_span_consistency=(
            model_context.protected_span_consistency if model_context else 0.0
        ),
        correction_distance=model_context.correction_distance if model_context else 0.0,
        semantic_compatibility=model_context.semantic_compatibility if model_context else 0.0,
        model_score_contribution=model_score_contribution,
        unsupported_entity_penalty=unsupported_entity_penalty,
        total=round(total, 4),
    )


def apply_ambiguity(query: str, entities: dict[str, Any], registry: LexiconRegistry) -> None:
    ambiguity = registry.ambiguities.get(match_key(query))
    if ambiguity:
        entities.clear()
        entities.update(
            candidates=ambiguity.candidates,
            ambiguity_type=ambiguity.ambiguity_type,
        )


def _format_price(value: int) -> str:
    return f"{value:,}".replace(",", ".") + "đ"


def _render_query(intent: Intent, entities: dict[str, Any], registry: LexiconRegistry) -> str:
    if intent == "Ambiguous":
        candidates = cast(list[str], entities.get("candidates", []))
        matched = next(
            (
                item
                for item in registry.ambiguities.values()
                if item.candidates == candidates
                and item.ambiguity_type == entities.get("ambiguity_type")
            ),
            None,
        )
        return (
            matched.canonical_rendering
            if matched
            else registry.templates.get("unknown_target", "Không rõ mục tiêu tìm kiếm")
        )
    if intent == "Coordinate Search":
        return f"{entities['latitude']},{entities['longitude']}"
    if intent == "Navigation":
        destination = (
            entities.get("destination_poi") or entities.get("destination_category") or "địa điểm"
        )
        origin = entities.get("origin")
        return (
            f"Chỉ đường từ {origin} đến {destination}" if origin else f"Chỉ đường đến {destination}"
        )
    if intent == "Address Search":
        base = f"{entities['house_number']} {entities['street']}"
        if entities.get("district"):
            base += f", {entities['district']}"
        return base
    if intent == "POI Search":
        base = str(entities["poi_name"])
        if entities.get("district") and match_key(str(entities["district"])) not in match_key(base):
            base += f", {entities['district']}"
        return base

    dish = entities.get("dish")
    category = entities.get("category")
    brand = entities.get("brand")
    cuisine = entities.get("cuisine")
    if dish:
        entries = registry.canonical_lookup("dish", dish)
        base = entries[0].canonical_rendering if entries else str(dish)
    elif category and brand:
        base = f"{category} {brand}"
    elif category:
        base = str(category)
    elif brand:
        base = str(brand)
    else:
        base = "Địa điểm"
    if cuisine and match_key(str(cuisine)) not in match_key(base):
        base += f" {cuisine}"

    attributes = cast(list[str], entities.get("attributes", []))
    attribute_order = registry.templates.get("attribute_order", "").split("|")
    order = {value: index for index, value in enumerate(attribute_order)}
    attributes = sorted(attributes, key=lambda value: order.get(value, len(order)))
    for attribute in attributes:
        if attribute == "check-in đẹp" and "phù hợp chụp ảnh" in attributes:
            continue
        if match_key(attribute) not in match_key(base):
            base += f" {attribute}"
    amenities = cast(list[str], entities.get("amenities", []))
    if amenities:
        base += " có " + " và ".join(amenities)
    if entities.get("quality"):
        base += " ngon"
    if entities.get("open_now"):
        base += " " + registry.templates.get("open_now", "đang mở cửa")
    if entities.get("open_late"):
        base += " " + registry.templates.get("open_late", "mở cửa muộn")
    if entities.get("open_24h"):
        base += " " + registry.templates.get("open_24h", "mở cửa 24/7")
    if entities.get("open_after"):
        base += f" mở cửa sau {entities['open_after']}"
    if entities.get("open_until"):
        base += f" mở cửa đến {entities['open_until']}"
    if entities.get("price_max"):
        base += f" dưới {_format_price(int(entities['price_max']))}"
    if entities.get("street") and dish:
        base += f" trên đường {entities['street']}"
    if entities.get("reference_poi"):
        base += f" gần {entities['reference_poi']}"
    elif entities.get("reference_area"):
        base += f" gần {entities['reference_area']}"
    elif entities.get("location") == "current_location":
        base += " " + registry.templates.get("near_current", "gần đây")
    if entities.get("district"):
        separator = " " if category == "ATM" else " tại "
        base += separator + str(entities["district"])
    if entities.get("city") and match_key(str(entities["city"])) not in match_key(base):
        base += f" tại {entities['city']}"
    return base


def _unresolved_fragments(
    corrected_text: str,
    canonical_rendering: str,
    registry: LexiconRegistry,
    covered_spans: set[tuple[int, int]],
) -> list[str]:
    words = list(re.finditer(r"[^\W\d_]+", corrected_text, re.UNICODE))
    covered = [False] * len(words)
    passthrough_words = [False] * len(words)
    rendered_tokens = Counter(match_key(canonical_rendering).split())
    passthrough = {
        match_key(value)
        for value in registry.templates.get("passthrough_tokens", "").split("|")
        if value
    }
    for index, word in enumerate(words):
        key = match_key(word.group())
        if key in passthrough:
            covered[index] = True
            passthrough_words[index] = True
            continue
        if rendered_tokens[key]:
            rendered_tokens[key] -= 1
            covered[index] = True
            continue
        covered[index] = any(
            word.start() < end and word.end() > start for start, end in covered_spans
        )
    unresolved = [index for index, item in enumerate(covered) if not item]
    if not unresolved:
        return []
    groups: list[list[int]] = []
    for index in unresolved:
        if groups and all(passthrough_words[item] for item in range(groups[-1][-1] + 1, index)):
            groups[-1].append(index)
        else:
            groups.append([index])
    fragments: list[str] = []
    for group in groups:
        start, end = group[0], group[-1]
        while start and passthrough_words[start - 1]:
            start -= 1
        while end + 1 < len(words) and passthrough_words[end + 1]:
            end += 1
        fragment = corrected_text[words[start].start() : words[end].end()].strip()
        if fragment and match_key(fragment) not in match_key(canonical_rendering):
            fragments.append(fragment)
    return list(dict.fromkeys(fragments))


def render_query(
    intent: Intent,
    entities: dict[str, Any],
    registry: LexiconRegistry,
    corrected_text: str | None = None,
    covered_spans: set[tuple[int, int]] | None = None,
    trace: TraceCollector | None = None,
) -> str:
    canonical_rendering = _render_query(intent, entities, registry)
    lossless_navigation_destination = False
    if (
        intent == "Navigation"
        and corrected_text is not None
        and not entities.get("destination_poi")
        and not entities.get("destination_category")
    ):
        destination = re.search(
            r"\b(?:đến|tới|den|toi)\b\s+(.+?)\s*$", corrected_text, re.IGNORECASE
        )
        if destination:
            destination_text = destination.group(1).strip(" ,.;")
            if destination_text:
                canonical_rendering = _render_query(
                    intent, {**entities, "destination_poi": destination_text}, registry
                )
                lossless_navigation_destination = True
    unresolved = (
        _unresolved_fragments(
            corrected_text,
            canonical_rendering,
            registry,
            covered_spans or set(),
        )
        if corrected_text is not None and intent != "Ambiguous" and not entities.get("candidates")
        else []
    )
    rendered = title_vi(" ".join((canonical_rendering, *unresolved)))
    used_lossless_text = lossless_navigation_destination or bool(unresolved)
    if trace is not None:
        trace.set_final_stage(
            TraceStage(
                stage=TraceStageName.FINAL_RENDERING,
                input=CanonicalEntities.model_validate(entities).model_dump_json(exclude_none=True),
                output=rendered,
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.FINAL_RENDERING,
                        source=intent,
                        target=rendered,
                        rule_id=(
                            "renderer.lossless_corrected_variant"
                            if used_lossless_text
                            else "renderer.deterministic"
                        ),
                        grounding=TraceGrounding(
                            source=(
                                "canonical_frame_plus_unresolved_spans"
                                if used_lossless_text
                                else "selected_semantic_frame"
                            )
                        ),
                        confidence=1.0,
                    )
                ],
            )
        )
    return rendered


def build_candidate(
    intent: Intent,
    frame: SemanticFrame,
    normalized_query: str,
    trace: TraceCollector | None = None,
    frame_id: str | None = None,
    model_context: ModelScoreContext | None = None,
) -> InterpretationCandidate:
    candidate = InterpretationCandidate(
        candidate_id="candidate_1",
        intent=intent,
        frame=frame,
        normalized_query=normalized_query,
        score=score_candidate(intent, frame, model_context),
    )
    if trace is not None:
        if frame_id is None:
            raise ValueError("frame_id is required when tracing candidate scoring")
        trace.add_candidate(candidate, frame_id)
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.CANDIDATE_SCORING,
                input=frame.model_dump_json(),
                output=candidate.score.model_dump_json(),
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.SCORE_AGGREGATION,
                        source=frame_id,
                        target=str(candidate.score.total),
                        rule_id="scoring.weighted_feature_sum.v1",
                        grounding=TraceGrounding(source="named_score_features"),
                        confidence=1.0,
                    )
                ],
            )
        )
    return candidate
