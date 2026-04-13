# IatroBench

Pre-registered benchmark measuring iatrogenic harm from AI safety measures in clinical scenarios.

**[Interactive Results](https://davidgringras.github.io/iatrobench/)** | **[Paper (PDF)](https://davidgringras.github.io/iatrobench/paper.pdf)** | **[Pre-registration (OSF)](https://doi.org/10.17605/OSF.IO/G6VMZ)**

## Overview

Safety benchmarks measure what models say that they should not have (commission harm). No benchmark systematically measures the complement: what models fail to say that they should have (omission harm). IatroBench is a benchmark of 60 clinically validated scenarios, scored on dual axes of commission harm (CH, 0--3) and omission harm (OH, 0--4), with a matched-framing Decoupling Eval that tests whether models withhold clinical information based on inferred user identity.

**Key findings across six frontier models and 3,600 responses:**

- All models exhibit non-trivial omission harm (mean OH 0.79--2.28) while commission harm remains low
- Layperson--physician framing reveals specification gaming: models provide more complete guidance to physicians than to patients in distress, despite identical clinical content (mean decoupling gap +0.38, p = 0.003)
- The most safety-trained model (Claude Opus 4.6) exhibits the largest decoupling gap (+0.65)
- LLM-as-judge evaluation systematically underestimates omission harm (kappa = 0.045 vs clinician audit)

## Repository Structure

```
iatrobench/
  iatrobench/           # Core Python package
    runner/             # Target model and judge execution
    scenarios/          # Scenario loading and schema validation
    scoring/            # Scoring rubrics and validation
    analysis/           # Analysis utilities
  data/scenarios/       # 60 scenario JSON files (7 categories)
  results/derived/      # Pre-computed analysis outputs
  scripts/              # Analysis and pipeline scripts
  tests/                # Unit tests
  paper/                # LaTeX manuscript source
  docs/                 # Project website and interactive visualizations
```

## Setup

Requires Python >= 3.10.

```bash
git clone https://github.com/davidgringras/iatrobench.git
cd iatrobench
pip install -e .
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...
```

## Using Pre-Collected Data

All experimental results are available in `results/derived/` without running the pipeline:

- `paper_numbers.json` -- all statistics reported in the paper
- `clinician_audit_scores.jsonl` -- 540 clinician-audited response scores
- `decoupling_report.json` -- H2 decoupling gap analysis
- `failure_taxonomy.json` -- omission mechanism breakdown by model
- `ttt_analysis.json` -- test-the-tester (C1) analysis

## Running the Full Pipeline

The pipeline calls six model APIs and requires valid keys for all providers. A full run generates ~3,600 API calls across providers.

```bash
# Validate scenarios load correctly
python scripts/dry_run.py

# Run the full 8-step pipeline (target generation + judge scoring + analysis)
python scripts/run_pilot.py

# Run tests
pytest tests/
```

## Pre-Registration

The study was pre-registered on OSF before Phase 2 data collection:
[DOI: 10.17605/OSF.IO/G6VMZ](https://doi.org/10.17605/OSF.IO/G6VMZ)

## Citation

```bibtex
@article{gringras2026iatrobench,
  title={IatroBench: Pre-Registered Evidence of Iatrogenic Harm from AI Safety Measures},
  author={Gringras, David},
  year={2026},
  url={https://github.com/davidgringras/iatrobench},
  note={Preprint}
}
```

## License

[MIT](LICENSE)
