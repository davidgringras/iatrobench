"""Tests for providers module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from iatrobench.config import ModelSpec
from iatrobench.providers import call_model, APICallRecord, _log_api_call


@pytest.fixture
def test_model() -> ModelSpec:
    return ModelSpec(
        model_id="test",
        litellm_id="test/model",
        provider="test",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
    )


def test_dry_run(test_model: ModelSpec) -> None:
    """Dry run should return placeholder without API call."""
    result = call_model(
        test_model,
        [{"role": "user", "content": "test"}],
        dry_run=True,
    )
    assert result["content"] == "[DRY RUN] No API call made."
    assert result["cost_usd"] == 0.0
    assert "call_id" in result


def test_dry_run_has_call_id(test_model: ModelSpec) -> None:
    result = call_model(
        test_model,
        [{"role": "user", "content": "test"}],
        dry_run=True,
    )
    assert len(result["call_id"]) == 36  # UUID format


def test_temperature_not_sent_when_unsupported() -> None:
    """Models with supports_temperature=False should not get temperature kwarg."""
    no_temp_model = ModelSpec(
        model_id="no_temp",
        litellm_id="test/no_temp",
        provider="test",
        supports_temperature=False,
    )

    with patch("iatrobench.providers.litellm") as mock_litellm:
        mock_litellm.suppress_debug_info = True
        mock_litellm.set_verbose = False
        mock_litellm.drop_params = True

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_litellm.completion.return_value = mock_response

        with patch("iatrobench.providers.HAS_LITELLM", True):
            call_model(
                no_temp_model,
                [{"role": "user", "content": "test"}],
            )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert "temperature" not in call_kwargs


def test_audit_log_written(test_model: ModelSpec, tmp_path: Path) -> None:
    """API call record should be written to audit log."""
    with patch("iatrobench.providers.API_LOG_FILE", tmp_path / "api_calls.jsonl"):
        result = call_model(
            test_model,
            [{"role": "user", "content": "test"}],
            dry_run=True,
        )

    log_file = tmp_path / "api_calls.jsonl"
    assert log_file.exists()
    with open(log_file) as f:
        record = json.loads(f.readline())
    assert record["model_id"] == "test"
    assert record["success"] is True
