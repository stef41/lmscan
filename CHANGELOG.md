# Changelog

## [0.6.0] - 2026-04-11

### Added
- Ensemble detection: combines signal-based detector + logistic regression classifier + character n-gram perplexity — three orthogonal scoring systems
- 5 new features: contraction rate, first-person pronoun ratio, question ratio, list pattern density, long n-gram repetition (4/5-grams)
- Perplexity model wired into main detection pipeline (was disconnected)
- Classifier upgraded from 12 to 23 features with re-estimated weights
- Punctuation style profiling for model fingerprinting (em-dash, semicolon, colon, comma density per model)
- Agreement bonus in ensemble: when all three systems agree, confidence is boosted
- Report now shows 18 analyzed features with signal indicators (up from 10)
- 12 new flag conditions for v0.6 features (contractions, first-person, lists, n-grams, questions)
- Detector expanded from 14 to 21 weighted signals
- 386 tests

### Improved
- AI text detection accuracy: GPT-style text now scores 88% (was 82%)
- Human formal text correctly classified at 12% (much improved)
- Contraction rate is highly discriminative (0.062 human vs 0.000 AI)
- First-person pronouns add orthogonal human signal

## [0.5.0] - 2026-04-10

### Added
- 6 new detection features: passive voice ratio, sentence opening diversity, lexical density, character entropy, hedging density, conjunction start ratio
- Detector expanded from 9 to 14 weighted signals for more accurate scoring
- TF-IDF weighted vocabulary matching in model fingerprinting
- Structural pattern matching for model attribution (sentence length, passive voice, paragraph uniformity)
- VS Code extension deep scan: paragraph-level highlighting with hover tooltips
- VS Code extension output channel with full forensics report
- New VS Code commands: `lmscan.deepScan` (paragraph-level) and `lmscan.clear`
- 386 tests

## [0.3.0] - 2026-04-10

### Added
- Multilingual AI text detection: French, Spanish, German, Portuguese, CJK handling
- Language auto-detection with `--language auto` (default) or manual override
- Language-specific slop word dictionaries and transition word lists
- Benchmark module: `run_benchmark()`, `BenchmarkSample`, `BenchmarkResult`
- Benchmark accuracy report with per-source breakdown and confusion matrix
- Web UI via Streamlit: `lmscan --web` or `pip install lmscan[web]`
- Interactive text analysis, feature visualization, model attribution chart
- VS Code extension scaffolding with inline highlighting, status bar, right-click scan
- Pre-commit hook integration (`.pre-commit-hooks.yaml` already included)
- 193 tests (up from 150)

## [0.2.0] - 2025-07-26

### Added
- Directory batch scanning with `--dir` flag and `scan_directory()` API
- Per-paragraph mixed content detection with `--mixed` flag and `scan_mixed()` API
- Calibration API: `calibrate()` and `find_optimal_threshold()` for custom datasets
- Self-contained HTML reports with dark theme gauge (`--format html`)
- Model fingerprints for Qwen, DeepSeek, Cohere, Phi (9 models total)
- 150 tests (up from 96)

## [0.1.0] - 2025-04-10

### Added
- Statistical AI text detection using 12 linguistic features
- Model fingerprinting for GPT-4, Claude, Gemini, Llama, Mistral
- Per-sentence analysis with individual AI probability scores
- CLI with text/file/stdin input, JSON output, threshold gating
- Python API: `scan()`, `scan_file()`
- Burstiness, entropy, Zipf deviation, vocabulary richness analysis
- AI "slop word" detection (known LLM vocabulary markers)
- Transition word ratio, readability consistency, bigram/trigram repetition
- Beautiful ASCII terminal report with feature table and model attribution
- Zero external dependencies
- 96 tests
