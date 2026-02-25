# IatroBench

Pre-registered benchmark measuring iatrogenic harm from AI safety measures in clinical scenarios.

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
  scripts/              # Analysis and pipeline scripts
  tests/                # Unit tests
  paper/                # LaTeX manuscript (NeurIPS 2026 D&B Track)
  docs/                 # Pre-registration document
```

## Setup

```bash
git clone https://github.com/davidgringras/iatrobench.git
cd iatrobench
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Running

```bash
# Validate scenarios load correctly
python scripts/dry_run.py

# Run the full pipeline (target generation + judge scoring)
python scripts/run_pilot.py

# Run tests
pytest tests/
```

## Pre-Registration

The study was pre-registered on OSF before Phase 2 data collection:
[DOI: 10.17605/OSF.IO/CJW92](https://doi.org/10.17605/OSF.IO/CJW92)

## Citation

```bibtex
@article{gringras2026iatrobench,
  title={IatroBench: Pre-Registered Evidence of Iatrogenic Harm from AI Safety Measures},
  author={Gringras, David},
  year={2026},
  note={NeurIPS 2026 Datasets \& Benchmarks Track (submitted)}
}
```

## License

MIT
