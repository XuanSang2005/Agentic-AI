from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.tasco_query.contracts import (
    TraceOperation,
    TraceOperationType,
    TraceStage,
    TraceStageName,
)

if TYPE_CHECKING:
    from src.tasco_query.tracing import TraceCollector

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[\u2010-\u2015]")
REPEATED_RE = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SurfaceText:
    display_text: str
    match_key: str


def _normalize_punctuation(text: str) -> str:
    text = PUNCT_RE.sub("-", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:!?])(?=[^\s\d])", r"\1 ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def clean_display(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _normalize_whitespace(_normalize_punctuation(text))


def accent_fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def match_key(text: str) -> str:
    folded = accent_fold(text)
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return SPACE_RE.sub(" ", folded).strip()


def comparison_key(text: str) -> str:
    """Accent-preserving key for functional variant deduplication."""
    return clean_display(text).casefold()


def normalize_surface(text: str, trace: TraceCollector | None = None) -> SurfaceText:
    unicode_text = unicodedata.normalize("NFC", text)
    punctuation_text = _normalize_punctuation(unicode_text)
    display = _normalize_whitespace(punctuation_text)
    if trace is not None:
        unicode_operations = []
        if unicode_text != text:
            unicode_operations.append(
                TraceOperation(
                    operation=TraceOperationType.UNICODE_NORMALIZATION,
                    source=text,
                    target=unicode_text,
                    rule_id="unicode.nfc",
                    confidence=1.0,
                )
            )
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.UNICODE_NORMALIZATION,
                input=text,
                output=unicode_text,
                operations=unicode_operations,
            )
        )
        cleanup_operations = []
        if punctuation_text != unicode_text:
            cleanup_operations.append(
                TraceOperation(
                    operation=TraceOperationType.PUNCTUATION_NORMALIZATION,
                    source=unicode_text,
                    target=punctuation_text,
                    rule_id="text.canonical_punctuation",
                    confidence=1.0,
                )
            )
        if display != punctuation_text:
            cleanup_operations.append(
                TraceOperation(
                    operation=TraceOperationType.WHITESPACE_NORMALIZATION,
                    source=punctuation_text,
                    target=display,
                    rule_id="text.collapse_whitespace",
                    confidence=1.0,
                )
            )
        cleanup_operations.extend(
            TraceOperation(
                operation=TraceOperationType.REPEATED_CHARACTER_OBSERVATION,
                source=display[index : index + 2],
                target=display[index : index + 2],
                start=index,
                end=index + 2,
                rule_id="text.repeated_character_observation",
                confidence=1.0,
            )
            for index in range(len(display) - 1)
            if display[index].isalpha()
            and display[index + 1].isalpha()
            and accent_fold(display[index]) == accent_fold(display[index + 1])
        )
        trace.add_stage(
            TraceStage(
                stage=TraceStageName.TEXT_CLEANUP,
                input=unicode_text,
                output=display,
                operations=cleanup_operations,
            )
        )
    return SurfaceText(display_text=display, match_key=match_key(display))


def reduce_repeated_characters(text: str) -> str:
    return REPEATED_RE.sub(r"\1", text)


def title_vi(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]
