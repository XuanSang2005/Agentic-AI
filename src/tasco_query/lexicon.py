from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.tasco_query.data import DataCatalog
from src.tasco_query.normalization import clean_display, match_key

EntityType = Literal[
    "action_cue",
    "amenity",
    "attribute",
    "brand",
    "category",
    "city",
    "cuisine",
    "dish",
    "district",
    "lexical",
    "opening_constraint",
    "poi",
    "purpose",
    "reference_area",
    "spatial_cue",
    "street",
]

SUPPORTED_FIELDS = {
    "amenities",
    "attributes",
    "brand",
    "category",
    "city",
    "cuisine",
    "discovery",
    "dish",
    "district",
    "navigation",
    "nearby",
    "open_24h",
    "open_late",
    "open_now",
    "poi_name",
    "quality",
    "reference_area",
    "rewrite",
    "street",
}
RULE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]+$")


class ContextConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_tokens: list[str] = Field(default_factory=list)
    forbidden_tokens: list[str] = Field(default_factory=list)
    max_token_distance: int = Field(default=3, ge=0, le=20)
    phrase_boundaries: bool = True

    @field_validator("required_tokens", "forbidden_tokens")
    @classmethod
    def validate_tokens(cls, values: list[str]) -> list[str]:
        if any(not match_key(value) for value in values):
            raise ValueError("context tokens must contain searchable text")
        return values


class LexiconEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical: str | bool | int | float | list[str]
    aliases: list[str] = Field(min_length=1)
    accentless_aliases: list[str] = Field(default_factory=list)
    entity_type: EntityType
    canonical_field: str
    rule_id: str
    priority: int = Field(default=50, ge=0, le=1000)
    confidence: float = Field(default=1.0, ge=0, le=1)
    contexts: ContextConstraint = Field(default_factory=ContextConstraint)
    negative_contexts: list[str] = Field(default_factory=list)
    canonical_rendering: str
    source: str
    transformation_type: Literal[
        "phrase_alias", "abbreviation", "teen_code", "accent_restoration", "telex"
    ] = "phrase_alias"

    @field_validator("canonical_field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in SUPPORTED_FIELDS:
            raise ValueError(f"unsupported canonical field: {value}")
        return value

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, value: str) -> str:
        if not RULE_ID_RE.fullmatch(value):
            raise ValueError(f"invalid rule ID: {value}")
        return value

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        if any(not match_key(value) for value in values):
            raise ValueError("aliases must contain searchable text")
        keys = [clean_display(value).casefold() for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate aliases in one entry")
        return values


class AmbiguityEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str
    candidates: list[str] = Field(min_length=2)
    ambiguity_type: str
    rule_id: str
    canonical_rendering: str

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, value: str) -> str:
        if not RULE_ID_RE.fullmatch(value):
            raise ValueError(f"invalid rule ID: {value}")
        return value


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    code: str
    source: str
    message: str


class LexiconValidationReport(BaseModel):
    files: list[str]
    entries: int
    aliases: int
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


class LexiconValidationError(ValueError):
    def __init__(self, report: LexiconValidationReport) -> None:
        self.report = report
        detail = "; ".join(issue.message for issue in report.errors[:5])
        super().__init__(f"lexicon validation failed: {detail}")


@dataclass(frozen=True, slots=True)
class LexiconMatch:
    entry: LexiconEntry
    alias: str
    start: int
    end: int
    match_mode: str
    score: float = 1.0


def _distance(left: str, right: str, maximum: int | None = None) -> int:
    if left == right:
        return 0
    if maximum is not None and abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, 1):
        current = [row]
        row_minimum = row
        for column, char_right in enumerate(right, 1):
            value = min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (char_left != char_right),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if maximum is not None and row_minimum > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


class LexiconRegistry:
    """Validated, immutable-in-practice registry shared by rewriting and extraction."""

    def __init__(
        self,
        entries: Iterable[LexiconEntry],
        *,
        files: Iterable[str] = (),
        templates: dict[str, str] | None = None,
        ambiguities: Iterable[AmbiguityEntry] = (),
        strict: bool = True,
    ) -> None:
        self.entries = tuple(entries)
        self.templates = dict(templates or {})
        self.ambiguities = {match_key(item.alias): item for item in ambiguities}
        self.report = self._validate(list(files), strict=strict)
        if not self.report.valid:
            raise LexiconValidationError(self.report)
        self._exact: defaultdict[str, list[tuple[str, LexiconEntry]]] = defaultdict(list)
        self._lower: defaultdict[str, list[tuple[str, LexiconEntry]]] = defaultdict(list)
        self._accent: defaultdict[str, list[tuple[str, LexiconEntry]]] = defaultdict(list)
        self._token: defaultdict[str, list[tuple[str, LexiconEntry]]] = defaultdict(list)
        self._rule = {entry.rule_id: entry for entry in self.entries}
        self._canonical: defaultdict[tuple[str, str], list[LexiconEntry]] = defaultdict(list)
        for entry in self.entries:
            canonical_key = json.dumps(entry.canonical, ensure_ascii=False, sort_keys=True)
            self._canonical[(entry.canonical_field, canonical_key)].append(entry)
            for alias in (*entry.aliases, *entry.accentless_aliases):
                display = clean_display(alias)
                self._exact[display].append((alias, entry))
                self._lower[display.casefold()].append((alias, entry))
                self._accent[match_key(display)].append((alias, entry))
                if len(match_key(display).split()) == 1:
                    self._token[match_key(display)].append((alias, entry))

    @classmethod
    def load(cls, lexicon_dir: Path, catalog: DataCatalog) -> LexiconRegistry:
        entries: list[LexiconEntry] = []
        files: list[str] = []
        templates: dict[str, str] = {}
        ambiguities: list[AmbiguityEntry] = []
        for path in sorted(lexicon_dir.glob("*.json")):
            files.append(str(path))
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if path.name == "rendering.json":
                    templates = {
                        str(key): str(value) for key, value in payload["templates"].items()
                    }
                    ambiguities = [
                        AmbiguityEntry.model_validate(item) for item in payload["ambiguities"]
                    ]
                elif path.name in {"semantic_implications.json", "review_dependency.json"}:
                    # Semantic planning metadata is separate from extraction knowledge.
                    continue
                else:
                    entries.extend(LexiconEntry.model_validate(item) for item in payload["entries"])
            except (KeyError, json.JSONDecodeError, ValidationError, TypeError) as exc:
                issue = ValidationIssue(
                    level="error", code="malformed_file", source=str(path), message=str(exc)
                )
                raise LexiconValidationError(
                    LexiconValidationReport(files=files, entries=0, aliases=0, errors=[issue])
                ) from exc
        entries.extend(cls._abbreviation_entries(catalog, entries))
        entries.extend(cls._street_entries(catalog, entries))
        entries.extend(cls._poi_entries(catalog, entries))
        return cls(
            entries,
            files=[
                *files,
                str(catalog.data_dir / "abbreviation.csv"),
                "raw local POI/address data",
            ],
            templates=templates,
            ambiguities=ambiguities,
            strict=True,
        )

    @classmethod
    def from_entries(cls, entries: Iterable[LexiconEntry]) -> LexiconRegistry:
        return cls(entries, files=["in_memory"], strict=True)

    @staticmethod
    def _existing_aliases(entries: Iterable[LexiconEntry]) -> set[tuple[str, str]]:
        return {
            (match_key(alias), entry.canonical_field)
            for entry in entries
            for alias in (*entry.aliases, *entry.accentless_aliases)
        }

    @classmethod
    def _abbreviation_entries(
        cls, catalog: DataCatalog, configured: list[LexiconEntry]
    ) -> list[LexiconEntry]:
        fields = {
            "district abbreviation": ("district", "district"),
            "city abbreviation": ("city", "city"),
            "category abbreviation": ("category", "category"),
            "category slang": ("category", "category"),
            "category alias": ("category", "category"),
            "English alias": ("category", "category"),
            "category": ("category", "category"),
            "brand abbreviation": ("brand", "brand"),
            "brand": ("brand", "brand"),
            "poi abbreviation": ("poi", "poi_name"),
            "street abbreviation": ("street", "street"),
            "amenity": ("amenity", "amenities"),
        }
        existing = cls._existing_aliases(configured)
        result: list[LexiconEntry] = []
        for index, row in enumerate(catalog.abbreviations, 1):
            mapped = fields.get(row.type)
            if mapped is None or "/" in row.normalized_form:
                continue
            entity_type, field = mapped
            if (match_key(row.term), field) in existing:
                continue
            canonical: str | list[str] = (
                [row.normalized_form] if field == "amenities" else row.normalized_form
            )
            result.append(
                LexiconEntry(
                    canonical=canonical,
                    aliases=[row.term],
                    entity_type=entity_type,  # type: ignore[arg-type]
                    canonical_field=field,
                    rule_id=f"abbreviation.csv.{index}",
                    priority=90,
                    confidence=1.0,
                    canonical_rendering=row.normalized_form,
                    source="data/raw/abbreviation.csv",
                    transformation_type="abbreviation",
                )
            )
            existing.add((match_key(row.term), field))
        return result

    @classmethod
    def _street_entries(
        cls, catalog: DataCatalog, configured: list[LexiconEntry]
    ) -> list[LexiconEntry]:
        existing = cls._existing_aliases(configured)
        result: list[LexiconEntry] = []
        for index, (key, street) in enumerate(sorted(catalog.street_index.items()), 1):
            if (key, "street") in existing:
                continue
            result.append(
                LexiconEntry(
                    canonical=street,
                    aliases=[street],
                    entity_type="street",
                    canonical_field="street",
                    rule_id=f"address.street.{index}",
                    priority=80,
                    confidence=1.0,
                    canonical_rendering=street,
                    source="data/raw/address.csv",
                )
            )
        return result

    @staticmethod
    def _poi_entries(catalog: DataCatalog, configured: list[LexiconEntry]) -> list[LexiconEntry]:
        existing = LexiconRegistry._existing_aliases(configured)
        result: list[LexiconEntry] = []
        for row in catalog.pois:
            raw_aliases = [value for value in (row.name_vi, row.name_en, *row.aliases) if value]
            aliases = list(
                {
                    clean_display(alias).casefold(): alias
                    for alias in raw_aliases
                    if (match_key(alias), "poi_name") not in existing
                }.values()
            )
            if not aliases:
                continue
            result.append(
                LexiconEntry(
                    canonical=row.name_vi,
                    aliases=list(dict.fromkeys(aliases)),
                    entity_type="poi",
                    canonical_field="poi_name",
                    rule_id=f"poi.csv.{row.poi_id.casefold()}",
                    priority=85,
                    confidence=1.0,
                    canonical_rendering=row.name_vi,
                    source="data/raw/poi.csv",
                )
            )
        return result

    def _validate(self, files: list[str], *, strict: bool) -> LexiconValidationReport:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        seen_rules: set[str] = set()
        seen: dict[tuple[str, str], tuple[str, str, str]] = {}
        alias_count = 0
        for entry in self.entries:
            if entry.rule_id in seen_rules:
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="duplicate_rule_id",
                        source=entry.source,
                        message=f"duplicate rule ID {entry.rule_id!r}",
                    )
                )
            seen_rules.add(entry.rule_id)
            for alias in (*entry.aliases, *entry.accentless_aliases):
                alias_count += 1
                key = (match_key(alias), entry.canonical_field)
                canonical = json.dumps(entry.canonical, ensure_ascii=False, sort_keys=True)
                previous = seen.get(key)
                authoritative_collision = entry.source in {
                    "data/raw/poi.csv",
                    "data/raw/address.csv",
                }
                if previous is not None and previous[0] != canonical:
                    issue = ValidationIssue(
                        level="warning" if authoritative_collision else "error",
                        code="conflicting_alias",
                        source=entry.source,
                        message=(
                            f"normalized alias {alias!r} for {entry.canonical_field} maps to "
                            f"both {previous[0]} and {canonical}"
                        ),
                    )
                    (warnings if authoritative_collision else errors).append(issue)
                elif previous is not None and previous[2] == entry.rule_id:
                    warnings.append(
                        ValidationIssue(
                            level="warning",
                            code="normalization_collision",
                            source=entry.source,
                            message=f"aliases in {entry.rule_id} share normalized key {key[0]!r}",
                        )
                    )
                elif previous is not None:
                    issue = ValidationIssue(
                        level="warning" if authoritative_collision else "error",
                        code="duplicate_alias",
                        source=entry.source,
                        message=(
                            f"duplicate normalized alias {alias!r} for {entry.canonical_field}"
                        ),
                    )
                    (warnings if authoritative_collision else errors).append(issue)
                else:
                    seen[key] = (canonical, entry.source, entry.rule_id)
        return LexiconValidationReport(
            files=files,
            entries=len(self.entries),
            aliases=alias_count,
            errors=errors,
            warnings=warnings,
        )

    def exact_display_lookup(self, text: str) -> list[LexiconEntry]:
        return [entry for _, entry in self._exact.get(clean_display(text), [])]

    def lowercase_lookup(self, text: str) -> list[LexiconEntry]:
        return [entry for _, entry in self._lower.get(clean_display(text).casefold(), [])]

    def accent_lookup(self, text: str) -> list[LexiconEntry]:
        return [entry for _, entry in self._accent.get(match_key(text), [])]

    def token_lookup(self, token: str) -> list[LexiconEntry]:
        return [entry for _, entry in self._token.get(match_key(token), [])]

    def prefix_lookup(self, prefix: str, limit: int = 10) -> list[LexiconEntry]:
        key = match_key(prefix)
        found: list[LexiconEntry] = []
        for alias_key, values in self._accent.items():
            if alias_key.startswith(key):
                found.extend(entry for _, entry in values)
            if len(found) >= limit:
                break
        return found[:limit]

    def fuzzy_candidates(
        self, text: str, *, max_distance: int = 1, limit: int = 8
    ) -> list[LexiconMatch]:
        key = match_key(text)
        candidates: list[LexiconMatch] = []
        for alias_key, values in self._accent.items():
            token_reorder = len(key.split()) > 1 and sorted(key.split()) == sorted(
                alias_key.split()
            )
            distance = 0 if token_reorder else _distance(key, alias_key, max_distance)
            if distance > max_distance:
                continue
            score = 0.98 if token_reorder else 1.0 - distance / max(len(key), len(alias_key), 1)
            candidates.extend(
                LexiconMatch(
                    entry,
                    alias,
                    0,
                    len(text),
                    "token_reorder" if token_reorder else "edit_distance",
                    score,
                )
                for alias, entry in values
            )
        candidates.sort(key=lambda item: (item.score, item.entry.priority), reverse=True)
        return candidates[:limit]

    def canonical_lookup(self, field: str, value: Any) -> list[LexiconEntry]:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return list(self._canonical.get((field, key), []))

    def rule_metadata(self, rule_id: str) -> LexiconEntry | None:
        return self._rule.get(rule_id)

    def phrase_matches(self, text: str, *, fields: set[str] | None = None) -> list[LexiconMatch]:
        words = list(re.finditer(r"\S+", text))
        token_keys = [match_key(word.group()) for word in words]
        found: list[LexiconMatch] = []
        for start_index in range(len(words)):
            for end_index in range(start_index + 1, len(words) + 1):
                key = " ".join(token_keys[start_index:end_index])
                for alias, entry in self._accent.get(key, []):
                    if fields is not None and entry.canonical_field not in fields:
                        continue
                    if not self._context_allows(entry, token_keys, start_index, end_index):
                        continue
                    found.append(
                        LexiconMatch(
                            entry=entry,
                            alias=alias,
                            start=words[start_index].start(),
                            end=words[end_index - 1].end(),
                            match_mode=(
                                "phrase_exact" if clean_display(alias) in text else "accent_exact"
                            ),
                        )
                    )
        found.sort(
            key=lambda item: (item.end - item.start, item.entry.priority, item.entry.confidence),
            reverse=True,
        )
        return found

    @staticmethod
    def _context_allows(entry: LexiconEntry, tokens: list[str], start: int, end: int) -> bool:
        context = entry.contexts
        forbidden = {
            match_key(value) for value in (*context.forbidden_tokens, *entry.negative_contexts)
        }
        if forbidden & set(tokens):
            return False
        required = [match_key(value) for value in context.required_tokens]
        if not required:
            return True
        left = max(0, start - context.max_token_distance)
        right = min(len(tokens), end + context.max_token_distance)
        neighborhood = tokens[left:right]
        return all(token in neighborhood for token in required)
