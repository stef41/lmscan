"""Benchmark lmscan against ground-truth samples.

Provides utilities to evaluate detection accuracy on labelled datasets,
comparing precision, recall, and F1 across different text sources.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .scanner import scan
from .calibration import calibrate, CalibrationResult


@dataclass
class BenchmarkSample:
    """A labelled text sample for benchmarking."""

    text: str
    is_ai: bool
    source: str = ""  # e.g. "GPT-4", "human-blog", "Claude"


@dataclass
class BenchmarkResult:
    """Results from running a benchmark suite."""

    total_samples: int = 0
    ai_samples: int = 0
    human_samples: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    avg_ai_probability_for_ai: float = 0.0
    avg_ai_probability_for_human: float = 0.0
    per_source: dict[str, dict[str, float]] = field(default_factory=dict)
    threshold: float = 0.5
    elapsed_s: float = 0.0


def run_benchmark(
    samples: list[BenchmarkSample],
    threshold: float = 0.5,
) -> BenchmarkResult:
    """Run lmscan on labelled samples and compute accuracy metrics.

    Args:
        samples: List of labelled text samples.
        threshold: AI probability threshold for classification.

    Returns:
        BenchmarkResult with precision, recall, F1, and per-source breakdown.
    """
    if not samples:
        return BenchmarkResult(threshold=threshold)

    start = time.monotonic()

    tp = fp = tn = fn = 0
    ai_probs_for_ai: list[float] = []
    ai_probs_for_human: list[float] = []
    source_results: dict[str, dict[str, list[float]]] = {}

    for sample in samples:
        result = scan(sample.text)
        prob = result.ai_probability
        predicted_ai = prob >= threshold

        # Track per-source
        src = sample.source or ("ai" if sample.is_ai else "human")
        if src not in source_results:
            source_results[src] = {"probs": [], "correct": []}
        source_results[src]["probs"].append(prob)
        source_results[src]["correct"].append(
            float(predicted_ai == sample.is_ai)
        )

        if sample.is_ai:
            ai_probs_for_ai.append(prob)
            if predicted_ai:
                tp += 1
            else:
                fn += 1
        else:
            ai_probs_for_human.append(prob)
            if predicted_ai:
                fp += 1
            else:
                tn += 1

    elapsed = time.monotonic() - start

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(samples) if samples else 0.0

    avg_ai = sum(ai_probs_for_ai) / len(ai_probs_for_ai) if ai_probs_for_ai else 0.0
    avg_human = sum(ai_probs_for_human) / len(ai_probs_for_human) if ai_probs_for_human else 0.0

    # Per-source breakdown
    per_source: dict[str, dict[str, float]] = {}
    for src, data in source_results.items():
        probs = data["probs"]
        correct = data["correct"]
        per_source[src] = {
            "count": float(len(probs)),
            "avg_probability": sum(probs) / len(probs) if probs else 0.0,
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
        }

    return BenchmarkResult(
        total_samples=len(samples),
        ai_samples=len(ai_probs_for_ai),
        human_samples=len(ai_probs_for_human),
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        avg_ai_probability_for_ai=round(avg_ai, 4),
        avg_ai_probability_for_human=round(avg_human, 4),
        per_source=per_source,
        threshold=threshold,
        elapsed_s=round(elapsed, 3),
    )


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Format benchmark results as a readable ASCII report."""
    lines: list[str] = []
    lines.append("lmscan Benchmark Report")
    lines.append("=" * 50)
    lines.append(f"Samples: {result.total_samples} ({result.ai_samples} AI, {result.human_samples} human)")
    lines.append(f"Threshold: {result.threshold}")
    lines.append(f"Time: {result.elapsed_s:.1f}s")
    lines.append("")
    lines.append(f"  Accuracy:  {result.accuracy:.1%}")
    lines.append(f"  Precision: {result.precision:.1%}")
    lines.append(f"  Recall:    {result.recall:.1%}")
    lines.append(f"  F1 Score:  {result.f1:.1%}")
    lines.append("")
    lines.append(f"  Avg AI prob (AI text):    {result.avg_ai_probability_for_ai:.1%}")
    lines.append(f"  Avg AI prob (human text): {result.avg_ai_probability_for_human:.1%}")
    lines.append("")
    lines.append("Confusion Matrix:")
    lines.append(f"  TP={result.true_positives}  FP={result.false_positives}")
    lines.append(f"  FN={result.false_negatives}  TN={result.true_negatives}")

    if result.per_source:
        lines.append("")
        lines.append("Per-Source Breakdown:")
        for src, data in sorted(result.per_source.items()):
            count = int(data["count"])
            acc = data["accuracy"]
            avg_p = data["avg_probability"]
            lines.append(f"  {src:20s}  n={count:3d}  acc={acc:.1%}  avg_prob={avg_p:.2f}")

    return "\n".join(lines)
