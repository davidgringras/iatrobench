# Changelog

All notable changes to IatroBench will be documented in this file.

## [1.0.0] — 2026-04-13

Initial public release accompanying the paper: *IatroBench: Pre-Registered Evidence of Iatrogenic Harm from AI Safety Measures*.

### Benchmark

- 60 clinical scenarios across 7 categories (mental health crisis, medication management, harm reduction, golden hour/emergency, equity gradient, terminal/advance care, control)
- 22 matched layperson–physician decoupling pairs for specification-gaming detection
- 6 control scenarios testing appropriate caution
- Dual-axis scoring: omission harm (OH, 0–4) and commission harm (CH, 0–3)
- Acuity-weighted scoring reflecting clinical severity
- Gold-standard responses and critical-action classifications validated against published clinical guidelines

### Pipeline

- 8-step sequential evaluation pipeline (`scripts/run_pilot.py`): lock → preflight → scenarios → config → target generation → primary judge → structured evaluation → analysis
- PID lockfile, SHA-256 hash manifest, and atomic deduplication for defensive execution
- LiteLLM abstraction across 6 API providers (Anthropic, OpenAI, Vertex AI, Together AI, DeepSeek, Mistral)
- Checkpoint/resume for long-running multi-provider runs

### Results

- 3,600 target responses across 6 frontier models (Claude Opus 4.6, GPT-5.2, Gemini 3 Pro, Llama 4 Maverick, DeepSeek-V3, Mistral Large)
- Structured evaluation scores for all responses
- Dual-physician validation (N=100, κ_w=0.571, within-1 agreement 96%)
- Pre-computed analysis outputs in `results/derived/` (paper statistics, failure taxonomy, decoupling analysis, clinician audit scores, temporal and retrospective spotchecks)

### Interactive Site

- GitHub Pages project site with scenario browser, model comparison, and decoupling visualisation
- Live at [davidgringras.github.io/iatrobench](https://davidgringras.github.io/iatrobench/)

### Paper

- LaTeX source (`paper/main.tex`) and arXiv submission bundle (`arxiv_submission/`)
- Pre-registered on OSF: [DOI: 10.17605/OSF.IO/G6VMZ](https://doi.org/10.17605/OSF.IO/G6VMZ)
