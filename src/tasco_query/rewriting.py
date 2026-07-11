from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal, cast

from src.tasco_query.contracts import (
    LockLevel,
    ProtectedSpan,
    QueryVariant,
    RewriteCandidate,
    RewriteEdit,
    SourceSpan,
    TraceGrounding,
    TraceOperation,
    TraceOperationType,
    TraceProtectedSpan,
    TraceRestorationStatus,
    TraceStage,
    TraceStageName,
    TraceTelexGate,
    TraceVariant,
    TraceVariantSource,
)
from src.tasco_query.data import DataCatalog
from src.tasco_query.lexicon import LexiconEntry, LexiconRegistry
from src.tasco_query.normalization import accent_fold, comparison_key, match_key

if TYPE_CHECKING:
    from src.tasco_query.tracing import TraceCollector

COORDINATE_RE = re.compile(r"(?<!\d)(-?\d{1,2}(?:\.\d+))\s*,\s*(-?\d{1,3}(?:\.\d+))(?!\d)")
TIME_RE = re.compile(
    r"(?<!\d)(?:[01]?\d|2[0-3])(?:[:h]\d{0,2})?\s*(?:h|giờ|am|pm|đêm|khuya|tối)?", re.I
)
PRICE_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:k|nghìn|triệu|đ|vnd)(?!\w)", re.I)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
QUOTED_RE = re.compile(r"([\"']).+?\1")


def extract_protected_spans(
    text: str, catalog: DataCatalog, trace: TraceCollector | None = None
) -> list[ProtectedSpan]:
    spans: list[ProtectedSpan] = []
    occupied: list[tuple[int, int]] = []
    for pattern, kind in (
        (COORDINATE_RE, "coordinate"),
        (PRICE_RE, "price"),
        (TIME_RE, "time_or_number"),
        (QUOTED_RE, "quoted"),
    ):
        for found in pattern.finditer(text):
            start, end = found.span()
            if any(start < prior_end and end > prior_start for prior_start, prior_end in occupied):
                continue
            occupied.append((start, end))
            spans.append(
                ProtectedSpan(
                    span=SourceSpan(start=start, end=end, text=found.group()),
                    level=LockLevel.HARD_LOCK,
                    kind=kind,
                )
            )
    text_key = match_key(text)
    for alias, pois in catalog.poi_alias_index.items():
        if len(alias.split()) < 2 or alias not in text_key or len(pois) != 1:
            continue
        canonical = pois[0].name_vi
        if match_key(canonical) != alias:
            continue
        exact_match = re.search(re.escape(canonical), text, re.I)
        if exact_match:
            spans.append(
                ProtectedSpan(
                    span=SourceSpan(
                        start=exact_match.start(), end=exact_match.end(), text=exact_match.group()
                    ),
                    level=LockLevel.HARD_LOCK,
                    kind="exact_poi",
                )
            )
    protected = sorted(spans, key=lambda item: (item.span.start, item.span.end))
    if trace is not None:
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.PROTECTED_SPAN_DETECTION,
                input=text,
                output=text,
                protected_spans=_trace_spans(protected),
            )
        )
    return protected


def _trace_spans(protected: list[ProtectedSpan]) -> list[TraceProtectedSpan]:
    return [
        TraceProtectedSpan(
            span_type=cast(
                Literal["coordinate", "price", "time_or_number", "quoted", "exact_poi"],
                item.kind,
            ),
            original_value=item.span.text,
            start=item.span.start,
            end=item.span.end,
            restoration_status=TraceRestorationStatus.PRESERVED,
        )
        for item in protected
    ]


def _overlaps(start: int, end: int, protected: list[ProtectedSpan]) -> bool:
    return any(start < item.span.end and end > item.span.start for item in protected)


def _candidate(
    *,
    text: str,
    replacement: str,
    start: int,
    end: int,
    transformation: str,
    rule_id: str,
    source: str,
    matched: str,
    confidence: float,
    cost: float,
    parent_id: str,
) -> RewriteCandidate:
    return RewriteCandidate(
        source_text=text[start:end],
        proposed_text=replacement,
        transformation_type=transformation,
        rule_id=rule_id,
        source_span=SourceSpan(start=start, end=end, text=text[start:end]),
        grounding_source=source,
        matched_lexicon_item=matched,
        confidence=confidence,
        cost=cost,
        parent_variant_id=parent_id,
    )


def _entry_replacement(entry: LexiconEntry) -> str:
    return entry.canonical_rendering


def _alias_candidates(
    text: str,
    registry: LexiconRegistry,
    protected: list[ProtectedSpan],
    parent_id: str,
) -> list[RewriteCandidate]:
    result: list[RewriteCandidate] = []
    for match in registry.phrase_matches(text):
        entry = match.entry
        if _overlaps(match.start, match.end, protected):
            continue
        source = text[match.start : match.end]
        replacement = _entry_replacement(entry)
        left_tokens = match_key(text[: match.start]).split()
        replacement_tokens = replacement.split()
        if (
            left_tokens
            and replacement_tokens
            and match_key(replacement_tokens[0]) == left_tokens[-1]
        ):
            replacement = " ".join(replacement_tokens[1:])
        transformation = entry.transformation_type
        if transformation == "phrase_alias" and accent_fold(source) == accent_fold(replacement):
            transformation = "accent_restoration"
        result.append(
            _candidate(
                text=text,
                replacement=replacement,
                start=match.start,
                end=match.end,
                transformation=transformation,
                rule_id=entry.rule_id,
                source=entry.source,
                matched=str(entry.canonical),
                confidence=entry.confidence,
                cost=max(0.01, 1.0 - entry.confidence),
                parent_id=parent_id,
            )
        )
    return result


def _repetition_candidates(
    text: str,
    registry: LexiconRegistry,
    protected: list[ProtectedSpan],
    parent_id: str,
) -> list[RewriteCandidate]:
    result: list[RewriteCandidate] = []
    for token in re.finditer(r"[^\W\d_]+", text, re.UNICODE):
        raw = token.group()
        proposals: list[tuple[str, LexiconEntry]] = []
        for index in range(len(raw) - 1):
            if accent_fold(raw[index]) != accent_fold(raw[index + 1]):
                continue
            for remove_at in (index, index + 1):
                reduced = raw[:remove_at] + raw[remove_at + 1 :]
                entries = [
                    entry
                    for entry in registry.accent_lookup(reduced)
                    if entry.entity_type != "cuisine"
                ]
                candidate_text = text[: token.start()] + reduced + text[token.end() :]
                original_rules = {item.entry.rule_id for item in registry.phrase_matches(text)}
                entries.extend(
                    item.entry
                    for item in registry.phrase_matches(candidate_text)
                    if item.entry.rule_id not in original_rules
                    and item.entry.entity_type != "cuisine"
                )
                proposals.extend((reduced, entry) for entry in entries)
        if not proposals or _overlaps(token.start(), token.end(), protected):
            continue
        reduced, entry = max(
            proposals,
            key=lambda item: (
                item[1].priority,
                item[1].confidence,
                sum(accent_fold(char) != char.casefold() for char in item[0]),
                len(item[0]),
            ),
        )
        result.append(
            _candidate(
                text=text,
                replacement=reduced,
                start=token.start(),
                end=token.end(),
                transformation="repeated_character_reduction",
                rule_id="repeat_to_known_lexicon",
                source=(
                    f"{entry.source}:{entry.entity_type}_vocabulary"
                    if entry.entity_type == "street"
                    else f"{entry.source}:{entry.entity_type}_aliases"
                ),
                matched=str(entry.canonical),
                confidence=min(0.99, entry.confidence),
                cost=0.04,
                parent_id=parent_id,
            )
        )
    return result


def _edit_candidates(
    text: str,
    registry: LexiconRegistry,
    protected: list[ProtectedSpan],
    parent_id: str,
    limit: int = 8,
) -> list[RewriteCandidate]:
    words = list(re.finditer(r"[^\W\d_]+", text, re.UNICODE))
    result: list[RewriteCandidate] = []
    for start_index in range(len(words)):
        for end_index in range(start_index + 1, min(len(words), start_index + 5) + 1):
            start, end = words[start_index].start(), words[end_index - 1].end()
            source = text[start:end]
            if len(match_key(source)) < 3 or _overlaps(start, end, protected):
                continue
            exact = registry.accent_lookup(source)
            if exact:
                continue
            matches = [
                match
                for match in registry.fuzzy_candidates(source, max_distance=1, limit=8)
                if match.entry.entity_type != "cuisine"
            ][:1]
            if not matches:
                continue
            match = matches[0]
            if match.score < 0.85:
                continue
            result.append(
                _candidate(
                    text=text,
                    replacement=match.entry.canonical_rendering,
                    start=start,
                    end=end,
                    transformation="edit_distance",
                    rule_id="generator.edit_distance_1",
                    source=match.entry.source,
                    matched=str(match.entry.canonical),
                    confidence=min(match.score, match.entry.confidence),
                    cost=0.08,
                    parent_id=parent_id,
                )
            )
    result.sort(
        key=lambda item: (
            item.confidence,
            item.source_span.end - item.source_span.start,
        ),
        reverse=True,
    )
    return result[:limit]


_TONE_MARKS = {
    "s": "\u0301",
    "f": "\u0300",
    "r": "\u0309",
    "x": "\u0303",
    "j": "\u0323",
}


def _decode_telex_token(token: str) -> list[str]:
    lowered = token.casefold()
    tone = _TONE_MARKS.get(lowered[-1]) if lowered else None
    if tone:
        lowered = lowered[:-1]
    decoded = lowered.replace("dd", "đ")
    replacements = (
        ("aw", "ă"),
        ("aa", "â"),
        ("ee", "ê"),
        ("oo", "ô"),
        ("ow", "ơ"),
        ("uw", "ư"),
    )
    for source, target in replacements:
        decoded = decoded.replace(source, target)
    if tone:
        vowels = "aăâeêiioôơuưy"
        positions = [index for index, char in enumerate(decoded) if char in vowels]
        if positions:
            preferred = [index for index in positions if decoded[index] in "êôơ"]
            position = preferred[-1] if preferred else positions[-1]
            decoded = unicodedata.normalize(
                "NFC", decoded[:position] + decoded[position] + tone + decoded[position + 1 :]
            )
    proposals = [decoded]
    if decoded.startswith("d"):
        proposals.append("đ" + decoded[1:])
    return list(dict.fromkeys(proposals))


def _telex_candidates(
    text: str,
    registry: LexiconRegistry,
    protected: list[ProtectedSpan],
    parent_id: str,
) -> tuple[list[RewriteCandidate], TraceTelexGate]:
    words = list(re.finditer(r"[^\W_]+", text, re.UNICODE))
    result: list[RewriteCandidate] = []
    plausible = 0
    for word in words:
        if _overlaps(word.start(), word.end(), protected):
            continue
        raw = word.group()
        grounded = [
            (
                proposal,
                [
                    entry
                    for entry in registry.accent_lookup(proposal)
                    if entry.entity_type != "cuisine"
                ],
            )
            for proposal in _decode_telex_token(raw)
            if proposal.casefold() != raw.casefold()
        ]
        grounded = [(proposal, entries) for proposal, entries in grounded if entries]
        if not grounded:
            continue
        plausible += 1
        proposal, entries = max(grounded, key=lambda item: max(entry.priority for entry in item[1]))
        entry = max(entries, key=lambda item: (item.priority, item.confidence))
        result.append(
            _candidate(
                text=text,
                replacement=entry.canonical_rendering,
                start=word.start(),
                end=word.end(),
                transformation="telex",
                rule_id="generator.gated_telex",
                source=entry.source,
                matched=str(entry.canonical),
                confidence=min(0.95, entry.confidence),
                cost=0.06,
                parent_id=parent_id,
            )
        )
    diacritics = sum(
        char.isalpha() and unicodedata.normalize("NFD", char.casefold()) != char.casefold()
        for char in text
    ) + text.casefold().count("đ")
    ratio = plausible / len(words) if words else 0.0
    accepted = plausible > 0 and (diacritics > 0 or ratio >= 0.2)
    reason = (
        "mixed_accented_and_telex_input"
        if accepted and diacritics
        else "plausible_telex_input"
        if accepted
        else "already_accented_vietnamese"
        if diacritics
        else "no_grounded_telex_tokens"
        if not plausible
        else "insufficient_telex_token_ratio"
    )
    gate = TraceTelexGate(
        attempted=accepted,
        accepted=accepted,
        vietnamese_diacritic_count=diacritics,
        plausible_telex_tokens=plausible,
        total_tokens=len(words),
        candidate_ratio=ratio,
        reason=reason,
    )
    return (result if accepted else []), gate


def _non_overlapping(candidates: Iterable[RewriteCandidate]) -> list[RewriteCandidate]:
    selected: list[RewriteCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.confidence,
            item.source_span.end - item.source_span.start,
            -item.cost,
        ),
        reverse=True,
    ):
        if any(
            candidate.source_span.start < item.source_span.end
            and candidate.source_span.end > item.source_span.start
            for item in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.source_span.start)


def _apply(text: str, candidates: list[RewriteCandidate]) -> str:
    output = text
    for candidate in reversed(_non_overlapping(candidates)):
        output = (
            output[: candidate.source_span.start]
            + candidate.proposed_text
            + output[candidate.source_span.end :]
        )
    return output


def _hard_locks_preserved(text: str, protected: list[ProtectedSpan]) -> bool:
    folded = match_key(text)
    return all(
        match_key(item.span.text) in folded
        for item in protected
        if item.level == LockLevel.HARD_LOCK and item.kind != "exact_poi"
    )


def _trace_source(transformation: str) -> TraceVariantSource:
    return {
        "repeated_character_reduction": TraceVariantSource.GUARDED_REPETITION_CORRECTION,
        "edit_distance": TraceVariantSource.EDIT_DISTANCE,
        "accent_restoration": TraceVariantSource.ACCENT_RESTORATION,
        "abbreviation": TraceVariantSource.ABBREVIATION_EXPANSION,
        "teen_code": TraceVariantSource.TEEN_CODE_EXPANSION,
        "telex": TraceVariantSource.TELEX_DECODE,
    }.get(transformation, TraceVariantSource.PHRASE_ALIAS)


def _trace_operation(candidate: RewriteCandidate) -> TraceOperation:
    operation = {
        "repeated_character_reduction": TraceOperationType.GUARDED_REPEATED_CHARACTER_CORRECTION,
        "edit_distance": TraceOperationType.EDIT_DISTANCE_CORRECTION,
        "accent_restoration": TraceOperationType.ACCENT_RESTORATION,
        "abbreviation": TraceOperationType.ABBREVIATION_EXPANSION,
        "teen_code": TraceOperationType.TEEN_CODE_EXPANSION,
        "telex": TraceOperationType.TELEX_DECODING,
    }.get(candidate.transformation_type, TraceOperationType.PHRASE_ALIAS_EXPANSION)
    return TraceOperation(
        operation=operation,
        source=candidate.source_text,
        target=candidate.proposed_text,
        start=candidate.source_span.start,
        end=candidate.source_span.end,
        rule_id=candidate.rule_id,
        grounding=TraceGrounding(
            source=candidate.grounding_source,
            matched_value=candidate.matched_lexicon_item,
        ),
        confidence=candidate.confidence,
        parent_variant_id=candidate.parent_variant_id,
        rewrite_cost=candidate.cost,
    )


def generate_variants(
    text: str,
    catalog: DataCatalog,
    protected: list[ProtectedSpan],
    limit: int = 8,
    trace: TraceCollector | None = None,
    registry: LexiconRegistry | None = None,
    stop_when: Callable[[list[QueryVariant]], bool] | None = None,
) -> list[QueryVariant]:
    if registry is None:
        from src.tasco_query.config import get_settings

        registry = LexiconRegistry.load(get_settings().lexicon_dir, catalog)
    variants = [
        QueryVariant(
            variant_id="v0", text=text, source="cleaned_original", prior=1.0, rewrite_cost=0
        )
    ]
    seen = {comparison_key(text): "v0"}
    trace_operations: list[TraceOperation] = []
    if trace is not None:
        trace.add_variant(
            TraceVariant(
                id="v0",
                text=text,
                source=TraceVariantSource.CLEANED_ORIGINAL,
                cost=0,
                deduplication_key=comparison_key(text),
                matching_key=match_key(text),
            )
        )

    parent = variants[0]
    telex_generated, telex_gate = _telex_candidates(text, registry, protected, parent.variant_id)
    stages = ("repeated_character_reduction", "telex", "phrase_alias", "edit_distance")
    for stage_index, stage in enumerate(stages, 1):
        if len(variants) >= limit:
            break
        if stage == "repeated_character_reduction":
            generated = (
                []
                if telex_gate.accepted
                else _repetition_candidates(parent.text, registry, protected, parent.variant_id)
            )
        elif stage == "telex":
            generated = telex_generated
        elif stage == "edit_distance":
            generated = _edit_candidates(parent.text, registry, protected, parent.variant_id)
        else:
            generated = _alias_candidates(parent.text, registry, protected, parent.variant_id)
        no_op = next(
            (
                item
                for item in generated
                if comparison_key(item.source_text) == comparison_key(item.proposed_text)
            ),
            None,
        )
        if no_op is not None and trace is not None:
            trace.add_variant(
                TraceVariant(
                    id=f"noop-{stage_index}",
                    text=parent.text,
                    source=_trace_source(no_op.transformation_type),
                    parent_id=parent.variant_id,
                    cost=no_op.cost,
                    deduplication_key=comparison_key(parent.text),
                    matching_key=match_key(parent.text),
                    deduplicated=True,
                    duplicate_of=parent.variant_id,
                    deduplication_reason="no_op_after_canonicalization",
                )
            )
        selected = _non_overlapping(generated)
        if not selected:
            continue
        candidate_text = _apply(parent.text, selected)
        key = comparison_key(candidate_text)
        source = _trace_source(selected[0].transformation_type)
        trace_operations.extend(_trace_operation(item) for item in selected)
        if key in seen:
            if trace is not None:
                trace.add_variant(
                    TraceVariant(
                        id=f"deduplicated-{stage_index}",
                        text=candidate_text,
                        source=source,
                        parent_id=parent.variant_id,
                        cost=sum(item.cost for item in selected),
                        deduplication_key=key,
                        matching_key=match_key(candidate_text),
                        deduplicated=True,
                        duplicate_of=seen[key],
                        deduplication_reason="same_display_text_and_transformation_result",
                    )
                )
            continue
        if not _hard_locks_preserved(candidate_text, protected):
            continue
        variant_id = f"v{len(variants)}"
        variant = QueryVariant(
            variant_id=variant_id,
            text=candidate_text,
            source=source.value,
            prior=min(item.confidence for item in selected),
            rewrite_cost=sum(item.cost for item in selected),
            edits=[
                RewriteEdit(
                    source=item.source_text,
                    replacement=item.proposed_text,
                    reason=item.rule_id,
                )
                for item in selected
            ],
            parent_id=parent.variant_id,
            generation=selected,
        )
        variants.append(variant)
        seen[key] = variant_id
        parent = variant
        if trace is not None:
            trace.add_variant(
                TraceVariant(
                    id=variant_id,
                    text=candidate_text,
                    source=source,
                    parent_id=variant.parent_id,
                    cost=variant.rewrite_cost,
                    deduplication_key=key,
                    matching_key=match_key(candidate_text),
                )
            )
        if stop_when is not None and stop_when(variants):
            break
    if telex_gate.accepted:
        telex_gate.result = next(
            (variant.text for variant in variants if variant.source == "telex_decode"), None
        )
    if trace is not None:
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.DETERMINISTIC_REWRITE,
                input=text,
                output=variants[-1].text,
                operations=trace_operations,
                telex_gate=telex_gate,
            )
        )
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.PROTECTED_SPAN_RESTORATION,
                input=variants[-1].text,
                output=variants[-1].text,
                operations=[
                    TraceOperation(
                        operation=TraceOperationType.PROTECTED_SPAN_PRESERVATION,
                        source=item.span.text,
                        target=item.span.text,
                        start=item.span.start,
                        end=item.span.end,
                        rule_id="protected_span.hard_lock_preserved",
                        grounding=TraceGrounding(source=item.kind),
                        confidence=1.0,
                    )
                    for item in protected
                ],
                protected_spans=_trace_spans(protected),
            )
        )
    return variants
