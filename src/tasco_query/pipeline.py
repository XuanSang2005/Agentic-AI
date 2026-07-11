from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from src.tasco_query.contracts import InterpretationCandidate, ProtectedSpan, QueryVariant
from src.tasco_query.data import DataCatalog
from src.tasco_query.extraction import Extraction, extract_entities
from src.tasco_query.lexicon import LexiconRegistry
from src.tasco_query.rewriting import generate_variants
from src.tasco_query.tracing import TraceCollector


class QueryVariantGenerator(Protocol):
    def generate(
        self,
        text: str,
        catalog: DataCatalog,
        protected: list[ProtectedSpan],
        limit: int,
        trace: TraceCollector | None,
        stop_when: Callable[[list[QueryVariant]], bool] | None = None,
    ) -> list[QueryVariant]: ...


class EvidenceExtractor(Protocol):
    def extract(
        self,
        original: str,
        variants: list[QueryVariant],
        catalog: DataCatalog,
        trace: TraceCollector | None,
    ) -> Extraction: ...


class SemanticConstraintExtractor(EvidenceExtractor, Protocol):
    pass


class CandidateReranker(Protocol):
    def rank(self, candidates: list[InterpretationCandidate]) -> list[InterpretationCandidate]: ...


class DeterministicVariantGenerator:
    def __init__(self, registry: LexiconRegistry) -> None:
        self.registry = registry

    def generate(
        self,
        text: str,
        catalog: DataCatalog,
        protected: list[ProtectedSpan],
        limit: int,
        trace: TraceCollector | None,
        stop_when: Callable[[list[QueryVariant]], bool] | None = None,
    ) -> list[QueryVariant]:
        return generate_variants(
            text,
            catalog,
            protected,
            limit=limit,
            trace=trace,
            registry=self.registry,
            stop_when=stop_when,
        )


class RegistryEvidenceExtractor:
    def __init__(self, registry: LexiconRegistry) -> None:
        self.registry = registry

    def extract(
        self,
        original: str,
        variants: list[QueryVariant],
        catalog: DataCatalog,
        trace: TraceCollector | None,
    ) -> Extraction:
        return extract_entities(
            original,
            variants,
            catalog,
            trace,
            registry=self.registry,
        )


class CompositeEvidenceExtractor:
    def __init__(self, extractors: list[EvidenceExtractor]) -> None:
        self.extractors = extractors

    def extract(
        self,
        original: str,
        variants: list[QueryVariant],
        catalog: DataCatalog,
        trace: TraceCollector | None,
    ) -> Extraction:
        merged = Extraction()
        for extractor in self.extractors:
            result = extractor.extract(original, variants, catalog, trace)
            for key, value in result.entities.items():
                if isinstance(value, list) and isinstance(merged.entities.get(key), list):
                    merged.entities[key].extend(
                        item for item in value if item not in merged.entities[key]
                    )
                else:
                    merged.entities.setdefault(key, value)
            merged.evidence.extend(result.evidence)
            merged.navigation |= result.navigation
            merged.nearby |= result.nearby
            merged.discovery |= result.discovery
            merged.coordinate_only |= result.coordinate_only
            merged.explicit_reference |= result.explicit_reference
            for variant_id, spans in result.covered_spans.items():
                merged.covered_spans.setdefault(variant_id, set()).update(spans)
        return merged


class DeterministicCandidateReranker:
    def rank(self, candidates: list[InterpretationCandidate]) -> list[InterpretationCandidate]:
        return sorted(candidates, key=lambda item: item.score.total, reverse=True)
