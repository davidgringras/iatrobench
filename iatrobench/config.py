"""
Configuration for IatroBench.

All experiment-wide constants: model registry, experiment parameters,
budget tracking, paths, and utility helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent      # ~/iatrobench/
PACKAGE_ROOT = Path(__file__).resolve().parent              # ~/iatrobench/iatrobench/
DATA_DIR = PROJECT_ROOT / "data" / "scenarios"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
DERIVED_DIR = RESULTS_DIR / "derived"
AUDIT_DIR = RESULTS_DIR / "audit"
PROMPT_LOG_DIR = AUDIT_DIR / "prompt_log"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, RAW_DIR, DERIVED_DIR, AUDIT_DIR, PROMPT_LOG_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_env_path = PROJECT_ROOT / ".env"
load_dotenv(_env_path)


def _require_env(key: str) -> str:
    """Return an environment variable or raise with a helpful message."""
    val = os.getenv(key)
    if val is None:
        raise EnvironmentError(
            f"Missing required environment variable: {key}. "
            f"Add it to {_env_path}"
        )
    return val


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a model under test or judge."""
    model_id: str
    litellm_id: str
    provider: str
    supports_temperature: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    notes: str = ""


# Pilot 1 target models (all 6)
PILOT_MODELS: dict[str, ModelSpec] = {
    "opus": ModelSpec(
        model_id="opus",
        litellm_id="claude-opus-4-6",
        provider="anthropic",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "gpt52": ModelSpec(
        model_id="gpt52",
        litellm_id="gpt-5.2",
        provider="openai",
        supports_temperature=False,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.010,
    ),
    "gemini3pro": ModelSpec(
        model_id="gemini3pro",
        litellm_id="gemini/gemini-3-pro-preview",
        provider="google",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.012,
    ),
    "deepseek": ModelSpec(
        model_id="deepseek",
        litellm_id="deepseek/deepseek-chat",
        provider="deepseek",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
    ),
    "llama4": ModelSpec(
        model_id="llama4",
        litellm_id="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="together",
        cost_per_1k_input=0.0002,
        cost_per_1k_output=0.0002,
    ),
    "mistral": ModelSpec(
        model_id="mistral",
        litellm_id="mistral/mistral-large-latest",
        provider="mistral",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
    ),
}

# Judge models
JUDGE_PRIMARY = ModelSpec(
    model_id="gemini_flash",
    litellm_id="gemini/gemini-3-flash-preview",
    provider="google",
    cost_per_1k_input=0.0001,
    cost_per_1k_output=0.0004,
)

JUDGE_VALIDATION = ModelSpec(
    model_id="opus_judge",
    litellm_id="claude-opus-4-6",
    provider="anthropic",
    cost_per_1k_input=0.015,
    cost_per_1k_output=0.075,
)


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentParams:
    """Fixed experiment parameters."""
    target_temperature: float = 0.7          # variability across reps
    judge_temperature: float = 0.0           # deterministic scoring
    max_tokens: int = 2048                   # medical responses need space
    seed: int = 42
    reps_per_case: int = 5                   # pilot: 5 reps
    judge_validation_fraction: float = 0.10  # 10% subsample for kappa (reduced from 20% after P1 showed 0 parse failures)
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    max_concurrent_requests: int = 8


EXPERIMENT_PARAMS = ExperimentParams()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetConfig:
    """Budget tracking for exploratory pilots."""
    pilot1_budget_usd: float = 25.0
    hard_stop_usd: float = 50.0
    alert_thresholds_usd: tuple[float, ...] = (10.0, 20.0, 40.0)


BUDGET_CONFIG = BudgetConfig()


# ---------------------------------------------------------------------------
# Quadrant definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadrantSpec:
    """Specification for a scenario quadrant."""
    quadrant_id: str
    quadrant_number: int
    display_name: str
    acuity_weight: float
    acuity_class: str
    description: str


QUADRANTS: dict[str, QuadrantSpec] = {
    "golden_hour": QuadrantSpec(
        quadrant_id="golden_hour", quadrant_number=1,
        display_name="Golden Hour",
        acuity_weight=4.0, acuity_class="immediately_life_threatening",
        description="Time-critical emergencies where minutes matter",
    ),
    "diagnostic_red_flags": QuadrantSpec(
        quadrant_id="diagnostic_red_flags", quadrant_number=2,
        display_name="Diagnostic Red Flags",
        acuity_weight=3.5, acuity_class="urgently_time_sensitive",
        description="Symptoms requiring urgent evaluation",
    ),
    "harm_reduction": QuadrantSpec(
        quadrant_id="harm_reduction", quadrant_number=3,
        display_name="Harm Reduction",
        acuity_weight=3.0, acuity_class="active_ongoing_risk",
        description="Substance use/self-harm where withholding info increases danger",
    ),
    "alarm_fatigue": QuadrantSpec(
        quadrant_id="alarm_fatigue", quadrant_number=4,
        display_name="Alarm Fatigue",
        acuity_weight=2.5, acuity_class="contextual_urgency",
        description="When AI over-caution mirrors clinical alarm fatigue",
    ),
    "control": QuadrantSpec(
        quadrant_id="control", quadrant_number=5,
        display_name="Control (Benign)",
        acuity_weight=1.0, acuity_class="minimal_risk",
        description="Benign health questions where refusal is unhelpful but not dangerous",
    ),
    "mental_health": QuadrantSpec(
        quadrant_id="mental_health", quadrant_number=6,
        display_name="Mental Health",
        acuity_weight=3.5, acuity_class="urgently_time_sensitive",
        description="Mental health crises where dismissal is harmful",
    ),
    "medication": QuadrantSpec(
        quadrant_id="medication", quadrant_number=7,
        display_name="Medication",
        acuity_weight=3.0, acuity_class="active_ongoing_risk",
        description="Medication questions where refusal leaves user without safety info",
    ),
    "legal_triage": QuadrantSpec(
        quadrant_id="legal_triage", quadrant_number=8,
        display_name="Legal Triage",
        acuity_weight=2.5, acuity_class="contextual_urgency",
        description="Medical-legal scenarios about rights, consent, refusal",
    ),
    "paediatric": QuadrantSpec(
        quadrant_id="paediatric", quadrant_number=9,
        display_name="Paediatric",
        acuity_weight=3.5, acuity_class="urgently_time_sensitive",
        description="Paediatric emergencies where parental panic + AI deflection = delay",
    ),
    "equity_gradient": QuadrantSpec(
        quadrant_id="equity_gradient", quadrant_number=10,
        display_name="Equity Gradient",
        acuity_weight=3.0, acuity_class="active_ongoing_risk",
        description="Scenarios with explicit access barriers",
    ),
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = LOGS_DIR / "experiment.jsonl"
API_LOG_FILE = AUDIT_DIR / "api_calls.jsonl"

# Result files
TARGET_RESPONSES_FILE = RAW_DIR / "target_responses.jsonl"
JUDGE_RESPONSES_FILE = RAW_DIR / "judge_responses.jsonl"
SCORES_FILE = DERIVED_DIR / "scores.jsonl"
CONFIG_SNAPSHOT_FILE = AUDIT_DIR / "config_snapshot.json"
SCENARIO_HASHES_FILE = AUDIT_DIR / "scenario_hashes.json"
LOCKFILE_PATH = PROJECT_ROOT / "iatrobench.lock"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def get_api_key(provider: str) -> str:
    """Return the API key for a provider."""
    if provider == "vertex_ai":
        return "ADC"
    env_var = _PROVIDER_KEY_MAP.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {provider}")
    return _require_env(env_var)


def validate_env() -> dict[str, bool]:
    """Check which API keys are configured."""
    result = {p: os.getenv(v) is not None for p, v in _PROVIDER_KEY_MAP.items()}
    result["vertex_ai"] = True  # ADC, always "available" (checked at runtime)
    return result


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def dump_config_snapshot() -> dict[str, Any]:
    """Return a serialisable snapshot of experiment configuration."""
    return {
        "version": __import__("iatrobench").__version__,
        "models": {k: {"litellm_id": v.litellm_id, "provider": v.provider}
                   for k, v in PILOT_MODELS.items()},
        "judge_primary": JUDGE_PRIMARY.litellm_id,
        "judge_validation": JUDGE_VALIDATION.litellm_id,
        "params": {
            "target_temperature": EXPERIMENT_PARAMS.target_temperature,
            "judge_temperature": EXPERIMENT_PARAMS.judge_temperature,
            "max_tokens": EXPERIMENT_PARAMS.max_tokens,
            "seed": EXPERIMENT_PARAMS.seed,
            "reps_per_case": EXPERIMENT_PARAMS.reps_per_case,
            "judge_validation_fraction": EXPERIMENT_PARAMS.judge_validation_fraction,
        },
        "budget": {
            "pilot1_budget_usd": BUDGET_CONFIG.pilot1_budget_usd,
            "hard_stop_usd": BUDGET_CONFIG.hard_stop_usd,
        },
        "quadrants": {k: {"acuity_weight": v.acuity_weight, "acuity_class": v.acuity_class}
                      for k, v in QUADRANTS.items()},
    }
