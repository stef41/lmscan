# Changelog

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
