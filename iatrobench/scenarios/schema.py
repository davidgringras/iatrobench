"""
JSON schema validation for IatroBench scenarios.

Validates every scenario at load time (fail-fast).
"""

from __future__ import annotations

from typing import Any

import jsonschema

SCENARIO_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "id",
        "quadrant",
        "quadrant_number",
        "acuity_weight",
        "prompt",
        "system_prompt",
        "gold_standard_response",
        "critical_actions",
        "commission_risks",
        "metadata",
    ],
    "properties": {
        "id": {"type": "string", "pattern": "^Q\\d+[a-z]$"},
        "quadrant": {
            "type": "string",
            "enum": [
                "golden_hour",
                "diagnostic_red_flags",
                "harm_reduction",
                "alarm_fatigue",
                "control",
                "mental_health",
                "medication",
                "legal_triage",
                "paediatric",
                "equity_gradient",
            ],
        },
        "quadrant_number": {"type": "integer", "minimum": 1, "maximum": 10},
        "acuity_weight": {"type": "number", "minimum": 1.0, "maximum": 4.0},
        "prompt": {"type": "string", "minLength": 20},
        "system_prompt": {"type": "string", "minLength": 1},
        "gold_standard_response": {"type": "string", "minLength": 20},
        "critical_actions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "commission_risks": {
            "type": "array",
            "items": {"type": "string"},
        },
        "equity_variant": {
            "oneOf": [{"type": "string"}, {"type": "object"}, {"type": "null"}],
        },
        "decoupling_variant": {
            "oneOf": [{"type": "string"}, {"type": "object"}, {"type": "null"}],
        },
        "metadata": {
            "type": "object",
            "required": ["clinical_basis", "acuity_class", "framing", "detail_level"],
            "properties": {
                "clinical_basis": {"type": "string"},
                "acuity_class": {
                    "type": "string",
                    "enum": [
                        "immediately_life_threatening",
                        "urgently_time_sensitive",
                        "active_ongoing_risk",
                        "contextual_urgency",
                        "minimal_risk",
                    ],
                },
                "framing": {"type": "string"},
                "detail_level": {
                    "type": "string",
                    "enum": ["rich", "sparse"],
                },
            },
        },
    },
    "additionalProperties": False,
}

# Compiled validator for reuse
_validator = jsonschema.Draft202012Validator(SCENARIO_SCHEMA)


def validate_scenario(scenario: dict[str, Any]) -> list[str]:
    """Validate a scenario dict against the schema.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if valid.
    """
    errors = []
    for error in _validator.iter_errors(scenario):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


def is_valid_scenario(scenario: dict[str, Any]) -> bool:
    """Check if a scenario dict is valid."""
    return _validator.is_valid(scenario)
