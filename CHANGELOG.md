# Changelog

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
