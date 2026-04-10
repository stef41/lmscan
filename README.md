# 🔍 lmscan

**Detect AI-generated text. Fingerprint which LLM wrote it. Open-source GPTZero alternative.**

[![PyPI](https://img.shields.io/pypi/v/lmscan?color=blue)](https://pypi.org/project/lmscan/)
[![Downloads](https://img.shields.io/pypi/dm/lmscan)](https://pypi.org/project/lmscan/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/lmscan)](https://pypi.org/project/lmscan/)
[![CI](https://github.com/stef41/lmscan/actions/workflows/ci.yml/badge.svg)](https://github.com/stef41/lmscan/actions)
[![Tests](https://img.shields.io/badge/tests-150%20passed-brightgreen)]()

> GPTZero charges $15/month. Originality.ai charges per scan. Turnitin locks you into institutional contracts.
>
> **lmscan is free, open-source, works offline, and tells you _which_ model wrote the text.**

```
$ lmscan "In today's rapidly evolving digital landscape, it's important
to note that artificial intelligence has become a pivotal force in
transforming how we navigate the complexities of modern life..."

🔍 lmscan v0.1.0 — AI Text Forensics
══════════════════════════════════════════════════

  Verdict:     🤖 Likely AI (77% confidence)
  Words:       184
  Sentences:   10
  Scanned in 0.01s

┌────────────────────────────┬──────────┬────────────────────┐
│ Feature                    │ Value    │ Signal             │
├────────────────────────────┼──────────┼────────────────────┤
│ Burstiness                 │ 0.07     │ 🔴 Very low (AI)    │
│ Sentence length variance   │ 0.27     │ 🟡 Below average    │
│ Slop word density          │ 20.7%    │ 🔴 High (AI)        │
│ Transition word ratio      │ 2.2%     │ 🟡 Elevated         │
│ Readability consistency    │ 0.00     │ 🔴 Very low (AI)    │
│ ...                        │          │                     │
└────────────────────────────┴──────────┴────────────────────┘

🔎 Model Attribution
  1. GPT-4 / ChatGPT    62% — "delve", "tapestry", "beacon", "landscape" (×2), +19 more
  2. Claude (Anthropic)  13% — "robust", "nuanced", "comprehensive"
  3. Gemini (Google)      9% — "furthermore", "additionally"

⚠️  Flags
  • Very low burstiness (0.07) — AI text is more uniform in complexity
  • High slop word density (20.7%) — contains known AI vocabulary markers
```

## Install

```bash
pip install lmscan
```

**Zero dependencies.** Works with Python 3.9+. No API keys. No internet. No GPU.

## Usage

```bash
# Scan text directly
lmscan "Your text here..."

# Scan a file
lmscan document.txt

# Pipe from stdin
cat essay.txt | lmscan -

# JSON output (for scripts and CI)
lmscan document.txt --format json

# Per-sentence breakdown
lmscan document.txt --sentences

# CI gate: fail if AI probability > 50%
lmscan submission.txt --threshold 0.5
```

### Python API

```python
from lmscan import scan

result = scan("Text to analyze...")

print(f"AI probability: {result.ai_probability:.0%}")
print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence}")

# Which model wrote it?
for model in result.model_attribution:
    print(f"  {model.model}: {model.confidence:.0%}")
    for evidence in model.evidence[:3]:
        print(f"    → {evidence}")

# Per-sentence analysis
for sentence in result.sentence_scores:
    if sentence.ai_probability > 0.7:
        print(f"  🤖 {sentence.text[:60]}... ({sentence.ai_probability:.0%})")
```

### Scan entire directories

```python
from lmscan import scan_file
import glob

for path in glob.glob("submissions/*.txt"):
    result = scan_file(path)
    print(f"{path}: {result.verdict} ({result.ai_probability:.0%})")
```

## How It Works

lmscan uses **12 statistical features** derived from computational linguistics research to distinguish AI-generated text from human writing:

| Feature | What it measures | AI signal |
|---------|-----------------|-----------|
| **Burstiness** | Variance in sentence complexity | AI text is unusually uniform |
| **Sentence length variance** | How much sentence lengths vary | AI produces uniform lengths |
| **Vocabulary richness** | Type-token ratio (Yule's K corrected) | AI reuses words more |
| **Hapax legomena ratio** | Fraction of words appearing once | AI has fewer unique words |
| **Zipf deviation** | How word frequencies follow Zipf's law | AI deviates from natural distribution |
| **Readability consistency** | Flesch-Kincaid variance across paragraphs | AI maintains constant readability |
| **Bigram/trigram repetition** | Repeated word pairs and triples | AI repeats phrase structures |
| **Transition word ratio** | "however", "moreover", "furthermore"... | AI overuses transitions |
| **Slop word density** | Known AI vocabulary markers | "delve", "tapestry", "beacon"... |
| **Punctuation entropy** | Diversity of punctuation usage | AI is more predictable |

Each feature produces a signal via sigmoid transformation. The weighted combination produces the final AI probability.

### Model Fingerprinting

lmscan includes vocabulary fingerprints for 5 major LLM families:

| Model | Distinctive markers |
|-------|-------------------|
| **GPT-4 / ChatGPT** | "delve", "tapestry", "landscape", "leverage", "multifaceted", "it's important to note" |
| **Claude (Anthropic)** | "certainly", "I'd be happy to", "straightforward", "I should note" |
| **Gemini (Google)** | "crucial", "here's a breakdown", "keep in mind" |
| **Llama / Meta** | "awesome", "fantastic", "hope this helps" |
| **Mistral / Mixtral** | "indeed", "moreover", "hence", "noteworthy" |

Attribution uses weighted vocabulary matching, phrase detection, and hedging pattern analysis.

## Accuracy & Limitations

**What lmscan is good at:**
- Detecting text with strong AI stylistic patterns
- Identifying which model family generated text
- Scanning at scale (thousands of documents) with zero cost
- Providing explainable evidence (not a black box)

**What lmscan cannot do:**
- Detect AI text that has been manually edited or paraphrased
- Work reliably on very short text (<50 words)
- Detect AI text in non-English languages (English-only for now)
- Replace human judgment — use as a signal, not a verdict

**This is statistical analysis, not a neural classifier.** It detects stylistic patterns, not watermarks. It works best on unedited LLM output and degrades gracefully on edited text.

## CI Integration

### GitHub Actions

```yaml
- name: AI Content Check
  run: |
    pip install lmscan
    lmscan submission.txt --threshold 0.7 --format json
```

### Pre-commit

```yaml
repos:
  - repo: https://github.com/stef41/lmscan
    rev: v0.1.0
    hooks:
      - id: lmscan
        args: ["--threshold", "0.7"]
```

## Research Background

lmscan's approach is informed by published research on AI text detection:

- **DetectGPT** (Mitchell et al., 2023) — perturbation-based detection using log probability curvature
- **GLTR** (Gehrmann et al., 2019) — statistical visualization of token predictions
- **Binoculars** (Hans et al., 2024) — cross-model perplexity comparison
- **Zipf's Law in NLP** — word frequency distributions differ between human and AI text
- **Stylometry** — decades of authorship attribution research applied to AI forensics

lmscan takes the statistical intuitions from these papers and implements them as lightweight, dependency-free heuristics that work without requiring a reference language model.

## FAQ

**Q: Is this as accurate as GPTZero?**
A: GPTZero uses neural classifiers trained on labeled data. lmscan uses statistical heuristics. GPTZero is more accurate on edge cases; lmscan is free, offline, and explainable. Use both if accuracy matters.

**Q: Can students use this to evade AI detection?**
A: lmscan shows which features trigger detection, which could help someone understand why text reads as AI-generated. This is by design — understanding AI writing patterns makes everyone a better writer. The same information is available in published research papers.

**Q: Does it work on non-English text?**
A: Currently English-only. The slop word lists and transition word lists are English-specific. Statistical features (entropy, burstiness) work across languages but haven't been calibrated.

**Q: Does it phone home?**
A: No. Zero network requests. No telemetry. No API keys. Everything runs locally.

**Q: How is model attribution possible without running the model?**
A: Each LLM family has characteristic vocabulary biases. GPT-4 loves "delve" and "tapestry". Claude says "I'd be happy to". These are statistical fingerprints — not guaranteed attribution, but strong signals.

## See Also

- [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) — Reverse-engineering Google's image watermarking
- [vibesafe](https://github.com/stef41/vibesafex) — AI code safety scanner
- [injectionguard](https://github.com/stef41/injectionguard) — Prompt injection detection
- [vibescore](https://github.com/stef41/vibescore) — Grade your vibe-coded project

## License

Apache-2.0
