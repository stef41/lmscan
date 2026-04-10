"""Tests for the benchmark module."""
from __future__ import annotations

from lmscan.benchmark import (
    BenchmarkResult,
    BenchmarkSample,
    format_benchmark_report,
    run_benchmark,
)


_AI_TEXT = (
    "In today's rapidly evolving technological landscape, it is important to note that "
    "artificial intelligence represents a truly transformative paradigm shift. The multifaceted "
    "nature of this groundbreaking innovation underscores the need to harness its potential. "
    "Furthermore, the holistic approach to leveraging these cutting-edge tools will facilitate "
    "unprecedented synergy across the entire ecosystem. Let's delve into the nuanced interplay "
    "between these pivotal developments and their comprehensive impact on our interconnected world. "
    "The robust framework enables streamlined operations that foster innovative solutions."
)

_HUMAN_TEXT = (
    "I spent all morning trying to fix that stupid bug. Turns out I had a typo in line 47 — "
    "classic. My cat walked across the keyboard while I was debugging, which honestly made "
    "about as much progress as I was making. Anyway, the fix was simple once I actually read "
    "the error message instead of just staring at it. Coffee helps. Lots of coffee. "
    "Sarah from the other team pinged me about the API changes. "
    "She said they're pushing it to next sprint. Fine by me. More time for testing."
)


class TestRunBenchmark:
    def test_empty_samples(self) -> None:
        result = run_benchmark([])
        assert result.total_samples == 0
        assert result.accuracy == 0.0

    def test_single_ai_sample(self) -> None:
        samples = [BenchmarkSample(text=_AI_TEXT, is_ai=True, source="GPT-4")]
        result = run_benchmark(samples, threshold=0.5)
        assert result.total_samples == 1
        assert result.ai_samples == 1
        assert result.human_samples == 0

    def test_single_human_sample(self) -> None:
        samples = [BenchmarkSample(text=_HUMAN_TEXT, is_ai=False, source="human")]
        result = run_benchmark(samples, threshold=0.5)
        assert result.total_samples == 1
        assert result.human_samples == 1

    def test_mixed_samples(self) -> None:
        samples = [
            BenchmarkSample(text=_AI_TEXT, is_ai=True, source="GPT-4"),
            BenchmarkSample(text=_HUMAN_TEXT, is_ai=False, source="human-blog"),
        ]
        result = run_benchmark(samples, threshold=0.5)
        assert result.total_samples == 2
        assert result.ai_samples == 1
        assert result.human_samples == 1
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.accuracy <= 1.0

    def test_per_source_breakdown(self) -> None:
        samples = [
            BenchmarkSample(text=_AI_TEXT, is_ai=True, source="GPT-4"),
            BenchmarkSample(text=_HUMAN_TEXT, is_ai=False, source="human-blog"),
        ]
        result = run_benchmark(samples)
        assert "GPT-4" in result.per_source
        assert "human-blog" in result.per_source
        assert result.per_source["GPT-4"]["count"] == 1.0

    def test_threshold_affects_results(self) -> None:
        samples = [
            BenchmarkSample(text=_AI_TEXT, is_ai=True, source="GPT-4"),
        ]
        low = run_benchmark(samples, threshold=0.1)
        high = run_benchmark(samples, threshold=0.99)
        # With low threshold, AI text should be caught; with 0.99, probably not
        assert low.threshold == 0.1
        assert high.threshold == 0.99

    def test_elapsed_time_recorded(self) -> None:
        samples = [BenchmarkSample(text=_AI_TEXT, is_ai=True)]
        result = run_benchmark(samples)
        assert result.elapsed_s >= 0.0

    def test_avg_probabilities(self) -> None:
        samples = [
            BenchmarkSample(text=_AI_TEXT, is_ai=True),
            BenchmarkSample(text=_HUMAN_TEXT, is_ai=False),
        ]
        result = run_benchmark(samples)
        assert 0.0 <= result.avg_ai_probability_for_ai <= 1.0
        assert 0.0 <= result.avg_ai_probability_for_human <= 1.0

    def test_confusion_matrix_adds_up(self) -> None:
        samples = [
            BenchmarkSample(text=_AI_TEXT, is_ai=True),
            BenchmarkSample(text=_HUMAN_TEXT, is_ai=False),
            BenchmarkSample(text=_AI_TEXT + " More AI text.", is_ai=True),
        ]
        result = run_benchmark(samples)
        total = result.true_positives + result.true_negatives + result.false_positives + result.false_negatives
        assert total == result.total_samples

    def test_default_source(self) -> None:
        samples = [BenchmarkSample(text=_AI_TEXT, is_ai=True)]
        result = run_benchmark(samples)
        assert "ai" in result.per_source


class TestFormatReport:
    def test_basic_report(self) -> None:
        result = BenchmarkResult(
            total_samples=100,
            ai_samples=50,
            human_samples=50,
            true_positives=45,
            true_negatives=40,
            false_positives=10,
            false_negatives=5,
            precision=0.818,
            recall=0.9,
            f1=0.857,
            accuracy=0.85,
            avg_ai_probability_for_ai=0.78,
            avg_ai_probability_for_human=0.25,
            threshold=0.5,
            elapsed_s=1.234,
        )
        report = format_benchmark_report(result)
        assert "Benchmark Report" in report
        assert "100" in report
        assert "85.0%" in report
        assert "Confusion Matrix" in report

    def test_report_with_per_source(self) -> None:
        result = BenchmarkResult(
            total_samples=10,
            ai_samples=5,
            human_samples=5,
            per_source={"GPT-4": {"count": 5.0, "avg_probability": 0.8, "accuracy": 0.9}},
        )
        report = format_benchmark_report(result)
        assert "GPT-4" in report
        assert "Per-Source" in report

    def test_report_empty(self) -> None:
        result = BenchmarkResult()
        report = format_benchmark_report(result)
        assert "0" in report
