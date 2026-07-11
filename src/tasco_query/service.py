from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from threading import Lock
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from src.tasco_query.adapters import (
    AdapterResult,
    CorrectionAdapter,
    DisabledCorrectionAdapter,
    DisabledGroundedLLMAdapter,
    GroundedLLMAdapter,
    HuggingFaceCorrectionAdapter,
    OpenAIGroundedLLMAdapter,
)
from src.tasco_query.config import PipelineMode, Settings, get_settings
from src.tasco_query.contracts import (
    CanonicalEntities,
    InterpretationResult,
    PipelineMetrics,
    ProtectedSpan,
    QueryResponse,
    QueryUnderstandRequest,
    QueryUnderstandResponse,
    QueryUnderstandTracedResponse,
    QueryVariant,
    SemanticDecompositionResult,
    SemanticFrame,
    SocialDiscoveryDecision,
    TraceEvidenceType,
    TraceExtractorName,
)
from src.tasco_query.data import DataCatalog
from src.tasco_query.extraction import Extraction
from src.tasco_query.lexicon import LexiconRegistry
from src.tasco_query.modeling import (
    allowed_soft_values,
    apply_soft_semantics,
    compact_evidence,
    compact_local_matches,
    integrate_adapter_result,
)
from src.tasco_query.normalization import match_key, normalize_surface
from src.tasco_query.pipeline import (
    CompositeEvidenceExtractor,
    DeterministicCandidateReranker,
    DeterministicVariantGenerator,
    RegistryEvidenceExtractor,
)
from src.tasco_query.rewriting import extract_protected_spans
from src.tasco_query.semantics import (
    CompositeSemanticImplicationResolver,
    DeterministicDirectConceptGrounder,
    DeterministicReviewDependencyClassifier,
    DeterministicSearchExpansionGenerator,
    DeterministicSemanticImplicationResolver,
    DeterministicSemanticUnitSegmenter,
    DeterministicSocialDiscoveryGate,
    QueryContext,
    ReviewDependencyClassifier,
    ReviewDependencyLexicon,
    SearchExpansionGenerator,
    SemanticContext,
    SemanticImplicationLexicon,
    SemanticImplicationResolver,
    SocialDiscoveryGate,
    apply_explicit_implications,
    decompose_variants,
    resolve_implications,
)
from src.tasco_query.tracing import TraceCollector
from src.tasco_query.understanding import (
    apply_ambiguity,
    build_candidate,
    build_frame,
    derive_intent,
    render_query,
)

_T = TypeVar("_T")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_REPEATED_CHARACTER_RE = re.compile(r"([^\W\d_])\1", re.IGNORECASE)
_EARLY_EXIT_FILLERS = {
    "cua",
    "cho",
    "co",
    "dia",
    "di",
    "den",
    "gan",
    "o",
    "tai",
    "tim",
    "tu",
    "voi",
}
_TARGET_FIELDS = {
    "poi_name",
    "destination_poi",
    "brand",
    "category",
    "dish",
    "cuisine",
    "street",
    "latitude",
}
_STAGE_NAMES = (
    "normalization",
    "protected_span_detection",
    "rewrite_generation",
    "extraction",
    "semantic_resolution",
    "model_call",
    "rendering",
)


class QueryUnderstandingService:
    def __init__(
        self,
        catalog: DataCatalog | None = None,
        *,
        settings: Settings | None = None,
        mode: PipelineMode | None = None,
        hf_adapter: CorrectionAdapter | None = None,
        llm_adapter: GroundedLLMAdapter | None = None,
        semantic_implication_resolver: SemanticImplicationResolver | None = None,
        search_expansion_generator: SearchExpansionGenerator | None = None,
        review_dependency_classifier: ReviewDependencyClassifier | None = None,
        social_discovery_gate: SocialDiscoveryGate | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.mode: PipelineMode = mode or settings.mode
        self.catalog = catalog or DataCatalog(settings.data_dir)
        self.registry = LexiconRegistry.load(settings.lexicon_dir, self.catalog)
        self.variant_generator = DeterministicVariantGenerator(self.registry)
        self.evidence_extractor = CompositeEvidenceExtractor(
            [RegistryEvidenceExtractor(self.registry)]
        )
        self.reranker = DeterministicCandidateReranker()
        self.semantic_unit_segmenter = DeterministicSemanticUnitSegmenter()
        self.direct_concept_grounder = DeterministicDirectConceptGrounder()
        self.hf_adapter = hf_adapter or (
            HuggingFaceCorrectionAdapter(settings)
            if settings.hf_enabled
            else DisabledCorrectionAdapter()
        )
        self.llm_adapter = llm_adapter or (
            OpenAIGroundedLLMAdapter(settings)
            if settings.llm_enabled and settings.llm_provider == "openai"
            else DisabledGroundedLLMAdapter()
        )
        semantic_lexicon = SemanticImplicationLexicon.load(
            settings.lexicon_dir / "semantic_implications.json"
        )
        self.semantic_lexicon = semantic_lexicon
        deterministic_implications = DeterministicSemanticImplicationResolver(semantic_lexicon)
        # The bounded query-level LLM proposal is the sole LLM request for a query.
        # Semantic implications stay deterministic so an unresolved unit cannot add calls.
        implication_resolvers: list[SemanticImplicationResolver] = [deterministic_implications]
        self.semantic_implication_resolver = semantic_implication_resolver or (
            CompositeSemanticImplicationResolver(
                implication_resolvers,
                min_confidence=settings.semantic_implication_min_confidence,
            )
        )
        self.search_expansion_generator = search_expansion_generator or (
            DeterministicSearchExpansionGenerator(semantic_lexicon)
        )
        self.review_dependency_lexicon = ReviewDependencyLexicon.load(
            settings.lexicon_dir / "review_dependency.json"
        )
        self.review_dependency_classifier = review_dependency_classifier or (
            DeterministicReviewDependencyClassifier(self.review_dependency_lexicon)
        )
        self.social_discovery_gate = social_discovery_gate or DeterministicSocialDiscoveryGate(
            min_confidence=settings.social_gate_confidence_threshold,
            exclude_objective_only=settings.social_gate_exclude_objective_only,
            exclude_exact_queries=settings.social_gate_exclude_exact_queries,
        )
        self.max_variants = settings.max_variants
        self.trace_max_variants = settings.trace_max_variants
        self.trace_max_evidence = settings.trace_max_evidence
        self.trace_max_candidates = settings.trace_max_candidates
        self.optimizations_enabled = settings.optimizations_enabled
        self.response_cache_enabled = settings.response_cache_enabled
        self.response_cache_size = settings.response_cache_size
        self.policy_version = settings.policy_version
        self._response_cache: OrderedDict[tuple[object, ...], dict[str, Any]] = OrderedDict()
        self._response_cache_lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def uses_hf(self) -> bool:
        return self.mode in {"rules_plus_hf", "full_hybrid"}

    @property
    def uses_llm(self) -> bool:
        return self.mode in {"rules_plus_llm", "full_hybrid"}

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": self.mode,
            "adapters": {
                "huggingface": asdict(self.hf_adapter.health()),
                "grounded_llm": asdict(self.llm_adapter.health()),
            },
            "response_cache": {
                "enabled": self.response_cache_enabled,
                "size": len(self._response_cache),
                "capacity": self.response_cache_size,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
            },
        }

    @staticmethod
    def _adapter_failure(name: str, identifier: str, exc: Exception) -> AdapterResult:
        return AdapterResult(
            adapter_name=name,
            model_identifier=identifier,
            error=f"{type(exc).__name__}: {exc}",
            timed_out=isinstance(exc, TimeoutError),
            fallback_occurred=True,
        )

    @staticmethod
    def _resolve_async(coroutine: Coroutine[Any, Any, _T]) -> _T:
        """Run the async resolver from either the synchronous service or FastAPI's event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coroutine)).result()

    def _cache_key(self, request: QueryUnderstandRequest) -> tuple[object, ...]:
        location = request.location
        return (
            request.query,
            request.locale,
            request.timezone,
            location.lat if location else None,
            location.lon if location else None,
            location.accuracy_m if location else None,
            location.area if location else None,
            location.city if location else None,
            self.mode,
            self.policy_version,
        )

    def _cache_get(self, key: tuple[object, ...]) -> dict[str, Any] | None:
        if not self.response_cache_enabled:
            return None
        with self._response_cache_lock:
            value = self._response_cache.pop(key, None)
            if value is None:
                self._cache_misses += 1
                return None
            self._response_cache[key] = value
            self._cache_hits += 1
            return value.copy()

    def _cache_put(self, key: tuple[object, ...], response: QueryUnderstandResponse) -> None:
        if not self.response_cache_enabled:
            return
        with self._response_cache_lock:
            self._response_cache.pop(key, None)
            self._response_cache[key] = response.model_dump()
            while len(self._response_cache) > self.response_cache_size:
                self._response_cache.popitem(last=False)

    @staticmethod
    def _empty_stage_timings() -> dict[str, float]:
        return {stage: 0.0 for stage in _STAGE_NAMES}

    def _cached_result(
        self, payload: dict[str, Any], trace_id: str, cache_lookup_ms: float
    ) -> InterpretationResult:
        response = QueryUnderstandResponse.model_validate(payload)
        timings = self._empty_stage_timings()
        timings["cache_lookup"] = round(cache_lookup_ms, 4)
        return InterpretationResult(
            response=response,
            frame=SemanticFrame(entities=response.entities),
            candidates=[],
            evidence=[],
            protected_spans=[],
            variants=[],
            semantic_decomposition=SemanticDecompositionResult(),
            social_discovery_decision=SocialDiscoveryDecision(
                should_trigger=False,
                reason="response_cache_hit",
                excluded_reasons=["response_cache_hit"],
                confidence=1.0,
            ),
            trace_id=trace_id,
            pipeline_metrics=PipelineMetrics(
                stage_timings_ms=timings,
                cache_hit=True,
                semantic_skipped=True,
            ),
        )

    def _complete_entity_coverage(
        self,
        text: str,
        extracted: Extraction,
        protected: list[ProtectedSpan],
        variant_id: str = "v0",
    ) -> bool:
        covered = set(extracted.covered_spans.get(variant_id, set()))
        covered.update((item.span.start, item.span.end) for item in protected)
        for match in self.registry.phrase_matches(text):
            if match.entry.canonical_field != "rewrite":
                covered.add((match.start, match.end))
        for evidence in extracted.evidence:
            if (
                evidence.accepted
                and evidence.variant_id == variant_id
                and evidence.span is not None
            ):
                covered.add((evidence.span.start, evidence.span.end))
        return all(
            any(word.start() < end and word.end() > start for start, end in covered)
            or match_key(word.group()) in _EARLY_EXIT_FILLERS
            for word in _WORD_RE.finditer(text)
        )

    def _is_confident_deterministic(
        self,
        text: str,
        extracted: Extraction,
        protected: list[ProtectedSpan],
        variant_id: str = "v0",
    ) -> bool:
        intent = derive_intent(text, extracted, self.registry)
        if match_key(text) in self.registry.ambiguities or extracted.coordinate_only:
            return True
        key = match_key(text)
        if self.review_dependency_lexicon.has_subjective_or_review_criterion(text) or any(
            match_key(alias) in key
            for mapping in self.semantic_lexicon.mappings
            for alias in mapping.aliases
        ):
            return False
        if {"attributes", "quality"} & extracted.entities.keys():
            return False
        if intent not in {
            "POI Search",
            "Category Search",
            "Brand Category Search",
            "Address Search",
            "Nearby Search",
        }:
            return False
        if not _TARGET_FIELDS & extracted.entities.keys():
            return False
        if not self._complete_entity_coverage(text, extracted, protected, variant_id):
            return False
        return all(item.confidence >= 0.9 for item in extracted.evidence if item.accepted)

    def _needs_correction(
        self,
        text: str,
        extracted: Extraction,
        protected: list[ProtectedSpan],
    ) -> bool:
        if self._is_confident_deterministic(text, extracted, protected):
            return False
        return bool(
            _REPEATED_CHARACTER_RE.search(text)
            or any(word.group().isascii() for word in _WORD_RE.finditer(text))
            or not _TARGET_FIELDS & extracted.entities.keys()
            or not self._complete_entity_coverage(text, extracted, protected)
        )

    def understand(self, request: QueryUnderstandRequest) -> InterpretationResult:
        trace_id = uuid4().hex
        cache_started = perf_counter()
        cache_key = self._cache_key(request)
        if not request.include_trace:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return self._cached_result(
                    cached, trace_id, (perf_counter() - cache_started) * 1000
                )
        timings = self._empty_stage_timings()
        trace = (
            TraceCollector(
                trace_id=trace_id,
                original_query=request.query,
                location_supplied=request.location is not None,
                max_variants=self.trace_max_variants,
                max_evidence=self.trace_max_evidence,
                max_candidates=self.trace_max_candidates,
            )
            if request.include_trace
            else None
        )
        started = perf_counter()
        surface = normalize_surface(request.query, trace)
        timings["normalization"] += (perf_counter() - started) * 1000
        started = perf_counter()
        protected = extract_protected_spans(surface.display_text, self.catalog, trace)
        timings["protected_span_detection"] += (perf_counter() - started) * 1000
        v0 = QueryVariant(
            variant_id="v0",
            text=surface.display_text,
            source="cleaned_original",
            prior=1.0,
            rewrite_cost=0,
        )
        started = perf_counter()
        preliminary = self.evidence_extractor.extract(
            surface.display_text, [v0], self.catalog, None
        )
        timings["extraction"] += (perf_counter() - started) * 1000
        early_exit = (
            self.optimizations_enabled
            and trace is None
            and self._is_confident_deterministic(surface.display_text, preliminary, protected)
        )
        correction_needed = self._needs_correction(surface.display_text, preliminary, protected)
        if early_exit:
            variants = [v0]
            extracted = preliminary
        else:
            lightweight_extraction_ms = 0.0

            def stop_when(candidates: list[QueryVariant]) -> bool:
                nonlocal lightweight_extraction_ms
                if not self.optimizations_enabled or trace is not None or len(candidates) < 2:
                    return False
                check_started = perf_counter()
                lightweight = self.evidence_extractor.extract(
                    surface.display_text, candidates, self.catalog, None
                )
                lightweight_extraction_ms += (perf_counter() - check_started) * 1000
                return self._is_confident_deterministic(
                    candidates[-1].text,
                    lightweight,
                    protected,
                    candidates[-1].variant_id,
                )

            started = perf_counter()
            variants = self.variant_generator.generate(
                surface.display_text,
                self.catalog,
                protected,
                self.max_variants,
                trace,
                stop_when=stop_when,
            )
            timings["rewrite_generation"] += (perf_counter() - started) * 1000
            timings["extraction"] += lightweight_extraction_ms
        model_proposals = []
        protected_by_variant = {variants[0].variant_id: protected}
        hf_calls = llm_calls = model_fallbacks = 0

        def spans_for(variant_id: str, text: str) -> list[ProtectedSpan]:
            if variant_id not in protected_by_variant:
                protected_by_variant[variant_id] = extract_protected_spans(text, self.catalog)
            return protected_by_variant[variant_id]

        if (
            not early_exit
            and self.uses_hf
            and (not self.optimizations_enabled or correction_needed)
        ):
            hf_parents = list(variants) if not self.optimizations_enabled else [variants[0]]
            hf_protected = [spans_for(parent.variant_id, parent.text) for parent in hf_parents]
            hf_calls = len(hf_parents)
            started = perf_counter()
            try:
                hf_results = self.hf_adapter.propose_batch(
                    [parent.text for parent in hf_parents], hf_protected
                )
            except Exception as exc:  # model failure must not break the endpoint
                hf_results = [
                    self._adapter_failure("huggingface", "configured", exc) for _ in hf_parents
                ]
            timings["model_call"] += (perf_counter() - started) * 1000
            for index, parent in enumerate(hf_parents):
                result = (
                    hf_results[index]
                    if index < len(hf_results)
                    else self._adapter_failure(
                        "huggingface", "configured", RuntimeError("adapter returned no results")
                    )
                )
                model_fallbacks += int(result.fallback_occurred or result.error is not None)
                model_proposals.extend(
                    integrate_adapter_result(
                        result,
                        original=parent.text,
                        parent_variant_id=parent.variant_id,
                        protected=hf_protected[index],
                        registry=self.registry,
                        variants=variants,
                        limit=self.max_variants,
                        trace=trace,
                    )
                )

        if not early_exit:
            analysis_started = perf_counter()
            analysis = self.evidence_extractor.extract(
                surface.display_text, variants, self.catalog, None
            )
            timings["extraction"] += (perf_counter() - analysis_started) * 1000
            uncertain = not self._is_confident_deterministic(
                variants[-1].text, analysis, protected, variants[-1].variant_id
            )
            if self.uses_llm and (not self.optimizations_enabled or uncertain):
                llm_parents = (
                    list(variants)
                    if not self.optimizations_enabled
                    else [max(variants, key=lambda item: item.prior - item.rewrite_cost)]
                )
                llm_calls = len(llm_parents)
                started = perf_counter()
                for llm_parent in llm_parents:
                    llm_protected = spans_for(llm_parent.variant_id, llm_parent.text)
                    try:
                        llm_result = self.llm_adapter.propose(
                            original_query=request.query,
                            cleaned_query=llm_parent.text,
                            protected=llm_protected,
                            deterministic_evidence=compact_evidence(analysis),
                            local_matches=compact_local_matches(llm_parent.text, self.registry),
                            allowed_fields=allowed_soft_values(self.registry),
                        )
                    except Exception as exc:  # model failure must not break the endpoint
                        llm_result = self._adapter_failure("grounded_llm", "configured", exc)
                    model_fallbacks += int(
                        llm_result.fallback_occurred or llm_result.error is not None
                    )
                    model_proposals.extend(
                        integrate_adapter_result(
                            llm_result,
                            original=llm_parent.text,
                            parent_variant_id=llm_parent.variant_id,
                            protected=llm_protected,
                            registry=self.registry,
                            variants=variants,
                            limit=self.max_variants,
                            trace=trace,
                        )
                    )
                timings["model_call"] += (perf_counter() - started) * 1000
            started = perf_counter()
            extracted = self.evidence_extractor.extract(
                surface.display_text, variants, self.catalog, trace
            )
            timings["extraction"] += (perf_counter() - started) * 1000

        semantic_skipped = early_exit
        semantic_decomposition = SemanticDecompositionResult()
        semantic_implications = []
        search_expansions = []
        review_dependency_classifications = []
        if semantic_skipped:
            social_discovery_decision = SocialDiscoveryDecision(
                should_trigger=False,
                reason="skipped_deterministic_exact_query",
                excluded_reasons=["skipped_deterministic_exact_query"],
                confidence=1.0,
            )
        else:
            started = perf_counter()
            semantic_decomposition = decompose_variants(
                variants,
                QueryContext(
                    registry=self.registry,
                    evidence=tuple(extracted.evidence),
                    location=request.location,
                ),
                segmenter=self.semantic_unit_segmenter,
                grounder=self.direct_concept_grounder,
                trace=trace,
            )
            semantic_implications = self._resolve_async(
                resolve_implications(
                    semantic_decomposition,
                    SemanticContext(
                        registry=self.registry,
                        grounded=tuple(semantic_decomposition.grounded_concepts),
                        existing_entities=extracted.entities,
                    ),
                    self.semantic_implication_resolver,
                )
            )
            apply_explicit_implications(extracted.entities, semantic_implications)
            timings["semantic_resolution"] += (perf_counter() - started) * 1000
        model_context = apply_soft_semantics(
            extracted,
            model_proposals,
            original_variant=variants[0],
            variants=variants,
            registry=self.registry,
            trace=trace,
        )
        intent = derive_intent(surface.display_text, extracted, self.registry)
        apply_ambiguity(surface.display_text, extracted.entities, self.registry)
        render_entities = dict(extracted.entities)

        location = request.location
        if location:
            if location.lat is not None and trace is not None:
                trace.add_evidence(
                    evidence_type=TraceEvidenceType.LOCATION_CONTEXT,
                    raw_value="coordinates_supplied",
                    canonical_value=True,
                    field="coordinates_supplied",
                    source_variant_id=None,
                    extractor=TraceExtractorName.REQUEST_CONTEXT,
                    rule_id="request_context.coordinates_supplied",
                    confidence=1.0,
                )
            if location.area and not any(
                key in extracted.entities for key in ("district", "reference_area")
            ):
                key = (
                    "district" if location.area.casefold().startswith("quận ") else "reference_area"
                )
                extracted.entities[key] = location.area
                if trace is not None:
                    trace.add_evidence(
                        evidence_type=TraceEvidenceType.LOCATION_CONTEXT,
                        raw_value=location.area,
                        canonical_value=location.area,
                        field=key,
                        source_variant_id=None,
                        extractor=TraceExtractorName.REQUEST_CONTEXT,
                        rule_id="request_context.area",
                        confidence=1.0,
                    )
            if location.city and "city" not in extracted.entities:
                extracted.entities["city"] = location.city
                if trace is not None:
                    trace.add_evidence(
                        evidence_type=TraceEvidenceType.LOCATION_CONTEXT,
                        raw_value=location.city,
                        canonical_value=location.city,
                        field="city",
                        source_variant_id=None,
                        extractor=TraceExtractorName.REQUEST_CONTEXT,
                        rule_id="request_context.city",
                        confidence=1.0,
                    )

        canonical = CanonicalEntities.model_validate(extracted.entities)
        frame, frame_id = build_frame(intent, extracted, trace)
        if not semantic_skipped:
            started = perf_counter()
            search_expansions = self.search_expansion_generator.generate(
                semantic_decomposition.grounded_concepts,
                semantic_implications,
                frame,
            )
            review_dependency_classifications = self.review_dependency_classifier.classify(
                semantic_decomposition,
                semantic_implications,
                tuple(extracted.evidence),
            )
            social_discovery_decision = self.social_discovery_gate.evaluate(
                frame,
                semantic_implications,
                review_dependency_classifications,
            )
            if trace is not None:
                trace.set_semantic_implications(semantic_implications, search_expansions)
                trace.set_social_discovery(
                    review_dependency_classifications,
                    social_discovery_decision,
                )
            timings["semantic_resolution"] += (perf_counter() - started) * 1000

        started = perf_counter()

        def render_rank(variant: QueryVariant) -> tuple[bool, int, float, float]:
            supporting = [
                item
                for item in extracted.evidence
                if item.accepted and item.variant_id == variant.variant_id
            ]
            best_path_score = max(
                (
                    proposal.confidence - proposal.correction_cost
                    for proposal, _ in model_proposals
                    if proposal.variant_id == variant.variant_id
                ),
                default=variant.prior - variant.rewrite_cost,
            )
            return (
                bool(supporting),
                -min((item.precedence for item in supporting), default=8),
                sum(item.confidence for item in supporting),
                best_path_score,
            )

        deterministic_variants = [
            variant for variant in variants if variant.model_identifier is None
        ]
        render_candidates = [
            deterministic_variants[-1],
            *(variant for variant in variants if variant.model_identifier is not None),
        ]
        render_variant = max(render_candidates, key=render_rank)
        normalized_query = render_query(
            intent,
            render_entities,
            self.registry,
            corrected_text=render_variant.text,
            covered_spans=extracted.covered_spans.get(render_variant.variant_id),
            trace=trace,
        )
        candidate = build_candidate(
            intent,
            frame,
            normalized_query,
            trace=trace,
            frame_id=frame_id,
            model_context=model_context,
        )
        candidates = self.reranker.rank([candidate])
        candidate = candidates[0]
        timings["rendering"] += (perf_counter() - started) * 1000
        response: QueryResponse
        if trace is not None:
            trace.set_decision(
                candidate, known_alias_collision="ambiguity_type" in extracted.entities
            )
            response = QueryUnderstandTracedResponse(
                normalized_query=normalized_query,
                intent=intent,
                entities=canonical.compact(),
                trace=trace.build(),
            )
        else:
            response = QueryUnderstandResponse(
                normalized_query=normalized_query,
                intent=intent,
                entities=canonical.compact(),
            )
        if trace is None:
            self._cache_put(cache_key, response)
        return InterpretationResult(
            response=response,
            frame=frame,
            candidates=candidates,
            evidence=extracted.evidence,
            protected_spans=protected,
            variants=variants,
            semantic_decomposition=semantic_decomposition,
            semantic_implications=semantic_implications,
            search_expansions=search_expansions,
            review_dependency_classifications=review_dependency_classifications,
            social_discovery_decision=social_discovery_decision,
            trace_id=trace_id,
            pipeline_metrics=PipelineMetrics(
                stage_timings_ms={key: round(value, 4) for key, value in timings.items()},
                rewrite_variants=max(0, len(variants) - 1),
                hf_calls=hf_calls,
                llm_calls=llm_calls,
                model_fallbacks=model_fallbacks,
                early_exit=early_exit,
                semantic_skipped=semantic_skipped,
            ),
        )


@lru_cache(maxsize=1)
def get_service() -> QueryUnderstandingService:
    return QueryUnderstandingService()
