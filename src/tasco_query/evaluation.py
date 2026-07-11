from __future__ import annotations

import csv
import json
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.tasco_query.config import ROOT, PipelineMode, Settings, get_settings
from src.tasco_query.contracts import QueryUnderstandRequest, QueryUnderstandTracedResponse
from src.tasco_query.service import QueryUnderstandingService


@dataclass(slots=True)
class Metrics:
    records: int
    failures: int
    normalized_query_exact_match: float
    normalized_query_character_similarity: float
    intent_accuracy: float
    entity_exact_match: float
    entity_micro_precision: float
    entity_micro_recall: float
    entity_micro_f1: float
    full_record_exact_match: float
    latency_p50_ms: float
    latency_p95_ms: float
    generated_variants: int
    rejected_variants: int
    average_generated_variants: float
    max_generated_variants: int
    model_calls: int
    model_fallbacks: int
    model_failure_rate: float
    hf_calls: int
    llm_calls: int
    early_exits: int
    early_exit_rate: float
    cache_hits: int
    cache_hit_rate: float
    semantic_skips: int
    stage_timings_ms: dict[str, StageLatency]


@dataclass(slots=True)
class StageLatency:
    average_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(slots=True)
class SemanticMetrics:
    records: int
    segmentation_span_precision: float
    segmentation_span_recall: float
    segmentation_span_f1: float
    direct_grounding_accuracy: float
    semantic_field_accuracy: float
    implication_precision: float
    implication_recall: float
    false_implication_rate: float
    false_hard_implications: int
    severe_error_count: int
    hard_vs_soft_classification_accuracy: float
    review_dependency_classification_accuracy: float
    social_gate_precision: float
    social_gate_recall: float
    social_gate_f1: float
    latency_p50_ms: float
    latency_p95_ms: float
    model_fallback_rate: float


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _normalize_entities(value: dict[str, Any]) -> dict[str, Any]:
    aliases = {"attribute": "attributes", "amenity": "amenities"}
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = aliases.get(raw_key, raw_key)
        item = raw_value
        if key in {"attributes", "amenities"} and not isinstance(item, list):
            item = [item]
        if isinstance(item, list):
            item = sorted(_normalize_text(str(part)) for part in item)
        elif isinstance(item, str):
            item = _normalize_text(item)
        normalized[key] = item
    return normalized


def _pairs(value: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for key, item in value.items():
        if isinstance(item, list):
            pairs.update(
                (key, json.dumps(part, ensure_ascii=False, sort_keys=True)) for part in item
            )
        else:
            pairs.add((key, json.dumps(item, ensure_ascii=False, sort_keys=True)))
    return pairs


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ratio(
    true_positive: int, false_positive: int, false_negative: int
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def _stage_latencies(samples: dict[str, list[float]]) -> dict[str, StageLatency]:
    result: dict[str, StageLatency] = {}
    for stage, values in sorted(samples.items()):
        if not values:
            continue
        ordered = sorted(values)
        p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
        result[stage] = StageLatency(
            average_ms=statistics.fmean(values),
            p50_ms=statistics.median(values),
            p95_ms=ordered[p95_index],
        )
    return result


def _semantic_unit_spans(query: str, units: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    """Derive curated expected spans in order, keeping the fixture easy to review."""
    result: set[tuple[int, int, str]] = set()
    cursor = 0
    normalized_query = query.casefold()
    for unit in units:
        text = str(unit["text"])
        start = normalized_query.find(text.casefold(), cursor)
        if start < 0:
            raise ValueError(f"semantic fixture phrase {text!r} is absent from {query!r}")
        result.add((start, start + len(text), str(unit["type"])))
        cursor = start + len(text)
    return result


def _implication_key(field: str, value: Any) -> tuple[str, str]:
    return field, _value_key(value)


def _implication_class(relationship: str) -> str:
    return "hard" if relationship in {"direct_interpretation", "canonical_paraphrase"} else "soft"


def _evaluate_semantic_records(
    records: list[dict[str, Any]], service: QueryUnderstandingService
) -> tuple[SemanticMetrics, list[dict[str, Any]]]:
    segment_tp = segment_fp = segment_fn = 0
    implication_tp = implication_fp = implication_fn = false_hard = 0
    direct_correct = field_correct = review_correct = relation_correct = 0
    direct_total = field_total = review_total = relation_total = 0
    gate_tp = gate_fp = gate_fn = 0
    latencies: list[float] = []
    model_calls = model_fallbacks = 0
    details: list[dict[str, Any]] = []

    for record in records:
        query = str(record["query"])
        started = time.perf_counter()
        result = service.understand(QueryUnderstandRequest(query=query, include_trace=True))
        latencies.append((time.perf_counter() - started) * 1000)
        expected_spans = _semantic_unit_spans(query, list(record["units"]))
        actual_spans = {
            (unit.start, unit.end, unit.unit_type.value)
            for unit in result.semantic_decomposition.units
            if unit.source_variant_id == "v0"
        }
        segment_tp += len(expected_spans & actual_spans)
        segment_fp += len(actual_spans - expected_spans)
        segment_fn += len(expected_spans - actual_spans)

        expected_direct = {
            _implication_key(item["field"], item["value"]) for item in record["direct_groundings"]
        }
        actual_direct = {
            _implication_key(item.field, item.canonical_value)
            for item in result.semantic_decomposition.grounded_concepts
            if item.source_unit_id.startswith("unit-v0-")
        }
        direct_total += 1
        direct_correct += actual_direct == expected_direct

        expected_fields = {
            _implication_key(item["field"], item["value"]) for item in record["semantic_fields"]
        }
        actual_fields = _pairs(result.response.entities) | {
            _implication_key(item.field, item.value)
            for item in result.semantic_implications
            if _implication_class(item.relationship.value) == "hard"
        }
        field_total += 1
        field_correct += expected_fields <= actual_fields

        expected_implications = {
            _implication_key(item["field"], item["value"]): str(item["class"])
            for item in record["acceptable_implications"]
        }
        forbidden = {
            _implication_key(item["field"], item["value"])
            for item in record["forbidden_implications"]
        }
        actual_implications = {
            _implication_key(item.field, item.value): _implication_class(item.relationship.value)
            for item in result.semantic_implications
        }
        expected_keys, actual_keys = set(expected_implications), set(actual_implications)
        implication_tp += len(expected_keys & actual_keys)
        unexpected = actual_keys - expected_keys
        implication_fp += len(unexpected)
        implication_fn += len(expected_keys - actual_keys)
        false_hard += sum(
            actual_implications[key] == "hard" for key in unexpected | (actual_keys & forbidden)
        )
        relation_total += len(expected_keys)
        relation_correct += sum(
            actual_implications.get(key) == classification
            for key, classification in expected_implications.items()
        )

        unit_texts = {unit.id: unit.text for unit in result.semantic_decomposition.units}
        actual_review = {
            unit_texts[item.source_unit_id].casefold(): item.review_dependency.value
            for item in result.review_dependency_classifications
            if item.concept_id == item.source_unit_id and item.source_unit_id in unit_texts
        }
        expected_review = {
            str(item["text"]).casefold(): str(item["dependency"])
            for item in record["review_dependencies"]
        }
        review_total += len(expected_review)
        review_correct += sum(
            actual_review.get(text) == dependency for text, dependency in expected_review.items()
        )

        expected_gate = bool(record["expected_social_gate"])
        actual_gate = result.social_discovery_decision.should_trigger
        gate_tp += expected_gate and actual_gate
        gate_fp += not expected_gate and actual_gate
        gate_fn += expected_gate and not actual_gate
        if isinstance(result.response, QueryUnderstandTracedResponse):
            model_calls += len(result.response.trace.model_calls)
            model_fallbacks += sum(
                item.fallback_occurred for item in result.response.trace.model_calls
            )
        details.append(
            {
                "id": record["id"],
                "query": query,
                "split": record.get("split", "holdout"),
                "expected_social_gate": expected_gate,
                "actual_social_gate": actual_gate,
                "unexpected_implications": [
                    {
                        "field": field,
                        "value": json.loads(value),
                        "class": actual_implications[(field, value)],
                    }
                    for field, value in sorted(unexpected)
                ],
            }
        )

    segment_precision, segment_recall, segment_f1 = _ratio(segment_tp, segment_fp, segment_fn)
    implication_precision, implication_recall, _ = _ratio(
        implication_tp, implication_fp, implication_fn
    )
    gate_precision, gate_recall, gate_f1 = _ratio(gate_tp, gate_fp, gate_fn)
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return (
        SemanticMetrics(
            records=len(records),
            segmentation_span_precision=segment_precision,
            segmentation_span_recall=segment_recall,
            segmentation_span_f1=segment_f1,
            direct_grounding_accuracy=direct_correct / direct_total if direct_total else 0,
            semantic_field_accuracy=field_correct / field_total if field_total else 0,
            implication_precision=implication_precision,
            implication_recall=implication_recall,
            false_implication_rate=implication_fp / len(records) if records else 0,
            false_hard_implications=false_hard,
            severe_error_count=false_hard,
            hard_vs_soft_classification_accuracy=relation_correct / relation_total
            if relation_total
            else 0,
            review_dependency_classification_accuracy=review_correct / review_total
            if review_total
            else 0,
            social_gate_precision=gate_precision,
            social_gate_recall=gate_recall,
            social_gate_f1=gate_f1,
            latency_p50_ms=statistics.median(latencies),
            latency_p95_ms=ordered[p95_index],
            model_fallback_rate=model_fallbacks / model_calls if model_calls else 0,
        ),
        details,
    )


def evaluate_semantic(
    eval_path: Path, service: QueryUnderstandingService
) -> tuple[SemanticMetrics, list[dict[str, Any]]]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("semantic evaluation set requires non-empty records")
    return _evaluate_semantic_records(records, service)


def calibrate_semantic(eval_path: Path) -> dict[str, Any]:
    """Select conservative defaults on a broad calibration split, never one acceptance query."""
    records = json.loads(eval_path.read_text(encoding="utf-8"))["records"]
    calibration = [record for record in records if record.get("split") == "calibration"]
    if not calibration:
        raise ValueError("semantic evaluation set requires a calibration split")
    base = get_settings()
    defaults = (0.5, 0.8, 0.7, True, True)
    candidates: list[tuple[tuple[float, float, float, bool, bool], SemanticMetrics]] = []
    for implication_threshold in (0.5, 0.6, 0.7):
        for llm_threshold in (0.75, 0.8, 0.85):
            for gate_threshold in (0.7, 0.8, 0.9):
                settings = replace(
                    base,
                    semantic_implication_min_confidence=implication_threshold,
                    semantic_llm_acceptance_threshold=llm_threshold,
                    social_gate_confidence_threshold=gate_threshold,
                    social_gate_exclude_objective_only=True,
                    social_gate_exclude_exact_queries=True,
                )
                metrics, _ = _evaluate_semantic_records(
                    calibration, QueryUnderstandingService(settings=settings, mode="rules_only")
                )
                candidates.append(
                    ((implication_threshold, llm_threshold, gate_threshold, True, True), metrics)
                )
    selected, metrics = min(
        candidates,
        key=lambda item: (
            item[1].severe_error_count,
            -item[1].social_gate_f1,
            -item[1].implication_recall,
            abs(item[0][0] - defaults[0])
            + abs(item[0][1] - defaults[1])
            + abs(item[0][2] - defaults[2]),
        ),
    )
    return {
        "records": len(calibration),
        "selection_order": [
            "fewest_false_hard_implications",
            "highest_social_gate_f1",
            "highest_implication_recall",
            "closest_conservative_default",
        ],
        "selected": {
            "implication_min_confidence": selected[0],
            "llm_acceptance_threshold": selected[1],
            "gate_confidence_threshold": selected[2],
            "objective_only_exclusions": selected[3],
            "exact_query_exclusions": selected[4],
        },
        "metrics": asdict(metrics),
        "llm_note": (
            "Rules-only calibration cannot measure live LLM agreement; 0.80 remains a "
            "conservative acceptance floor until a provider-backed set is run."
        ),
    }


def write_semantic_reports(
    metrics: SemanticMetrics,
    details: list[dict[str, Any]],
    calibration: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": asdict(metrics), "calibration": calibration, "records": details}
    (output_dir / "semantic_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Semantic Evaluation Report", "", "## Metrics", ""]
    lines.extend(
        f"- `{key}`: {value:.4f}" if isinstance(value, float) else f"- `{key}`: {value}"
        for key, value in asdict(metrics).items()
    )
    lines.extend(
        [
            "",
            "A false hard implication is counted in both `false_hard_implications` and "
            "`severe_error_count`; calibration minimizes it before recall or gate F1.",
            "",
            "## Calibration",
            "",
            "```json",
            json.dumps(calibration, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Metric definitions",
            "",
            "- Span metrics use exact `(start, end, semantic unit type)` matches on the "
            "original variant.",
            "- Direct-grounding and semantic-field accuracy are per-record exact/set-coverage "
            "scores.",
            "- Implication precision/recall compare allowlisted hard and soft implication pairs; "
            "false implication rate is unexpected implications per record.",
            "- Review accuracy is label accuracy over curated source phrases; gate metrics treat "
            "`true` as positive.",
        ]
    )
    (output_dir / "semantic_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    eval_path: Path, service: QueryUnderstandingService
) -> tuple[Metrics, list[dict[str, Any]]]:
    with eval_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    details: list[dict[str, Any]] = []
    failures = norm_exact = intent_exact = entity_exact = full_exact = 0
    similarities: list[float] = []
    latencies: list[float] = []
    true_positive = false_positive = false_negative = 0
    generated_variants = rejected_variants = 0
    model_calls = model_fallbacks = hf_calls = llm_calls = 0
    early_exits = cache_hits = semantic_skips = 0
    stage_samples: dict[str, list[float]] = {}
    per_record_variants: list[int] = []
    for row in rows:
        started = time.perf_counter()
        try:
            result = service.understand(QueryUnderstandRequest(query=row["input_query"]))
            response = result.response.model_dump()
            diagnostics = result.pipeline_metrics
            hf_calls += diagnostics.hf_calls
            llm_calls += diagnostics.llm_calls
            model_calls += diagnostics.hf_calls + diagnostics.llm_calls
            model_fallbacks += diagnostics.model_fallbacks
            early_exits += diagnostics.early_exit
            cache_hits += diagnostics.cache_hit
            semantic_skips += diagnostics.semantic_skipped
            for stage, value in diagnostics.stage_timings_ms.items():
                stage_samples.setdefault(stage, []).append(value)
            error = None
        except Exception as exc:  # pragma: no cover - defensive report behavior
            response = {"normalized_query": "", "intent": "Ambiguous", "entities": {}}
            error = f"{type(exc).__name__}: {exc}"
            failures += 1
        latencies.append((time.perf_counter() - started) * 1000)
        generated = result.pipeline_metrics.rewrite_variants if error is None else 0
        generated_variants += generated
        per_record_variants.append(generated)
        expected_entities = _normalize_entities(json.loads(row["expected_entities_json"]))
        actual_entities = _normalize_entities(response["entities"])
        expected_text = _normalize_text(row["expected_normalized_query"])
        actual_text = _normalize_text(response["normalized_query"])
        text_equal = expected_text == actual_text
        intent_equal = row["expected_intent"] == response["intent"]
        entities_equal = expected_entities == actual_entities
        expected_pairs, actual_pairs = _pairs(expected_entities), _pairs(actual_entities)
        true_positive += len(expected_pairs & actual_pairs)
        false_positive += len(actual_pairs - expected_pairs)
        false_negative += len(expected_pairs - actual_pairs)
        norm_exact += text_equal
        intent_exact += intent_equal
        entity_exact += entities_equal
        full_exact += text_equal and intent_equal and entities_equal
        similarities.append(SequenceMatcher(None, expected_text, actual_text).ratio())
        details.append(
            {
                "query_id": row["query_id"],
                "input_query": row["input_query"],
                "expected": {
                    "normalized_query": row["expected_normalized_query"],
                    "intent": row["expected_intent"],
                    "entities": json.loads(row["expected_entities_json"]),
                },
                "actual": response,
                "matches": {
                    "normalized_query": text_equal,
                    "intent": intent_equal,
                    "entities": entities_equal,
                },
                "error": error,
                "latency_ms": round(latencies[-1], 3),
                "pipeline_metrics": result.pipeline_metrics.model_dump() if error is None else {},
            }
        )
    count = len(rows)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    metrics = Metrics(
        records=count,
        failures=failures,
        normalized_query_exact_match=norm_exact / count,
        normalized_query_character_similarity=statistics.fmean(similarities),
        intent_accuracy=intent_exact / count,
        entity_exact_match=entity_exact / count,
        entity_micro_precision=precision,
        entity_micro_recall=recall,
        entity_micro_f1=f1,
        full_record_exact_match=full_exact / count,
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=ordered[p95_index],
        generated_variants=generated_variants,
        rejected_variants=rejected_variants,
        average_generated_variants=generated_variants / count,
        max_generated_variants=max(per_record_variants, default=0),
        model_calls=model_calls,
        model_fallbacks=model_fallbacks,
        model_failure_rate=model_fallbacks / model_calls if model_calls else 0.0,
        hf_calls=hf_calls,
        llm_calls=llm_calls,
        early_exits=early_exits,
        early_exit_rate=early_exits / count,
        cache_hits=cache_hits,
        cache_hit_rate=cache_hits / count,
        semantic_skips=semantic_skips,
        stage_timings_ms=_stage_latencies(stage_samples),
    )
    return metrics, details


def write_reports(
    metrics: Metrics,
    details: list[dict[str, Any]],
    output_dir: Path,
    *,
    mode: PipelineMode = "rules_only",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "metrics": asdict(metrics), "records": details}
    (output_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Evaluation Report", "", f"Mode: `{mode}`", "", "## Metrics", ""]
    for key, value in asdict(metrics).items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"- `{key}`: {rendered}")
    lines.extend(
        [
            "",
            "## Per-record summary",
            "",
            "| ID | Text | Intent | Entities |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in details:
        matches = record["matches"]
        lines.append(
            f"| {record['query_id']} | {matches['normalized_query']} | "
            f"{matches['intent']} | {matches['entities']} |"
        )
    (output_dir / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_optimizations(settings: Settings) -> dict[str, Any]:
    """Run the same fixtures with the old fan-out policy and the bounded policy."""
    modes: tuple[PipelineMode, ...] = (
        "rules_only",
        "rules_plus_hf",
        "rules_plus_llm",
        "full_hybrid",
    )
    datasets = {
        "original": settings.data_dir / "eval.csv",
        "noisy": ROOT / "data" / "evaluation" / "noisy.csv",
        "unseen_paraphrases": ROOT / "data" / "evaluation" / "unseen_paraphrases.csv",
    }
    semantic_path = ROOT / "data" / "evaluation" / "semantic.json"
    baseline_settings = replace(settings, optimizations_enabled=False, response_cache_enabled=False)
    optimized_settings = replace(settings, optimizations_enabled=True, response_cache_enabled=False)
    result: dict[str, Any] = {"modes": {}}
    for mode in modes:
        before = QueryUnderstandingService(settings=baseline_settings, mode=mode)
        after = QueryUnderstandingService(settings=optimized_settings, mode=mode)
        cached = QueryUnderstandingService(
            settings=replace(settings, optimizations_enabled=True, response_cache_enabled=True),
            mode=mode,
        )
        result["modes"][mode] = {
            "health": after.health(),
            "datasets": {
                name: {
                    "before": asdict(evaluate(path, before)[0]),
                    "after": asdict(evaluate(path, after)[0]),
                }
                for name, path in datasets.items()
            },
            "semantic": {
                "before": asdict(evaluate_semantic(semantic_path, before)[0]),
                "after": asdict(evaluate_semantic(semantic_path, after)[0]),
            },
            "cache_probe": {
                "first_pass": asdict(evaluate(datasets["original"], cached)[0]),
                "repeat_pass": asdict(evaluate(datasets["original"], cached)[0]),
            },
        }
    return result


def write_optimization_benchmark(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimization_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Query Understanding Optimization Benchmark",
        "",
        "`before` disables the bounded routing/cache policy; `after` keeps cache disabled "
        "for a fair parsing comparison. `cache_probe.repeat_pass` measures warm-cache hits.",
        "",
        "| Mode | Dataset | Before p95 | After p95 | Before variants | After variants | "
        "Before model calls | After model calls | Early exits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in report["modes"].items():
        for dataset, comparison in values["datasets"].items():
            before, after = comparison["before"], comparison["after"]
            lines.append(
                f"| {mode} | {dataset} | {before['latency_p95_ms']:.3f} | "
                f"{after['latency_p95_ms']:.3f} | {before['average_generated_variants']:.3f} | "
                f"{after['average_generated_variants']:.3f} | {before['model_calls']} | "
                f"{after['model_calls']} | {after['early_exit_rate']:.3f} |"
            )
        semantic = values["semantic"]
        lines.extend(
            [
                "",
                f"- `{mode}` semantic false-hard implications: "
                f"before `{semantic['before']['false_hard_implications']}`, "
                f"after `{semantic['after']['false_hard_implications']}`.",
                f"- `{mode}` warm-cache hit rate: "
                f"`{values['cache_probe']['repeat_pass']['cache_hit_rate']:.3f}`.",
            ]
        )
    (output_dir / "optimization_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    settings = get_settings()
    metrics, details = evaluate(
        settings.data_dir / "eval.csv", QueryUnderstandingService(mode="rules_only")
    )
    write_reports(metrics, details, ROOT / "data" / "generated")
    semantic_path = ROOT / "data" / "evaluation" / "semantic.json"
    semantic_metrics, semantic_details = evaluate_semantic(
        semantic_path, QueryUnderstandingService(mode="rules_only")
    )
    write_semantic_reports(
        semantic_metrics,
        semantic_details,
        calibrate_semantic(semantic_path),
        ROOT / "data" / "generated",
    )
    optimization_benchmark = benchmark_optimizations(settings)
    write_optimization_benchmark(optimization_benchmark, ROOT / "data" / "generated")
    comparison: dict[str, Any] = {
        "modes": {
            mode: {
                "health": values["health"],
                "datasets": {
                    name: comparison["after"] for name, comparison in values["datasets"].items()
                },
            }
            for mode, values in optimization_benchmark["modes"].items()
        }
    }
    output_dir = ROOT / "data" / "generated"
    (output_dir / "model_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Rules-only vs Model-assisted Evaluation",
        "",
        (
            "Model adapters are opt-in. A mode with a disabled/unavailable adapter reports its "
            "health and deterministic fallback metrics; equal scores are not a model-improvement "
            "claim."
        ),
        "",
        "| Mode | Dataset | Failures | Intent | Entity F1 | Exact | p50 ms | Model fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in comparison["modes"]:
        for name, values in comparison["modes"][mode]["datasets"].items():
            lines.append(
                f"| {mode} | {name} | {values['failures']} | {values['intent_accuracy']:.4f} | "
                f"{values['entity_micro_f1']:.4f} | {values['full_record_exact_match']:.4f} | "
                f"{values['latency_p50_ms']:.3f} | {values['model_failure_rate']:.4f} |"
            )
    (output_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
