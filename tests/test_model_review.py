"""
Unit and FastAPI integration tests for `WP-MODELS` (`model_catalog.json`, `model_review.py`,
`orchestrator.py`, `main.py`, `static/index.html`).

Covers all test requirements in `docs/WP-MODELS-CONTRACT.md` §5 without network access,
live CLI execution, or touching real `~/.agents` or live hub files.
"""

from __future__ import annotations

import copy
import datetime
import json
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

import main as hub_main
import model_review
import orchestrator
from allocation import BurnRateLearner
from main import app
from model_review import (
    DEFAULT_CATALOG_PATH,
    apply_model_decision,
    compute_blended_price,
    discover_local_models,
    evaluate_probation_models,
    fetch_public_metadata,
    get_latest_model_review,
    is_scheduled_review_due,
    load_model_catalog,
    match_local_model_to_public,
    normalise_model_name,
    parse_agy_models_output,
    reset_probation_counters,
    run_model_review,
    save_model_catalog,
    validate_model_catalog_doc,
    validate_model_entry,
)
from orchestrator import (
    MODEL_CATALOG,
    reload_model_catalog,
    reset_allocation_credits,
    route_work_package,
    select_active_orchestrator,
    set_preference_weights,
)
from scheduler import QuotaCostLearner


LEGACY_MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "provider": "agy",
        "model_id": "gemini-4-hydrogen-low",
        "cli_model": "gemini-4-hydrogen-low",
        "quota_key": "gemini_flash",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "LOW",
        "tier_rank": 1,
        "effort": "low",
        "default_preferred": True,
        "context_window": 1_000_000,
        "latency_score": 0.96,
        "cost_weight": 0.95,
        "label": "AGY · Gemini 4 Hydrogen (Low)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-3.8-flash-low",
        "cli_model": "gemini-3.8-flash-low",
        "quota_key": "gemini_flash",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "LOW",
        "tier_rank": 1,
        "effort": "low",
        "default_preferred": False,
        "context_window": 1_000_000,
        "latency_score": 0.95,
        "cost_weight": 0.94,
        "label": "AGY · Gemini 3.8 Flash (Low - Fallback)",
    },
    {
        "provider": "codex",
        "model_id": "gpt-6.1-sol-low",
        "cli_model": "gpt-6.1-sol",
        "quota_key": "luna",
        "pool_id": "codex_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "openai",
        "tier": "LOW",
        "tier_rank": 1,
        "effort": "low",
        "default_preferred": True,
        "context_window": 475_000,
        "latency_score": 0.92,
        "cost_weight": 0.88,
        "label": "Codex · GPT-6.1-Sol (Low)",
    },
    {
        "provider": "claude",
        "model_id": "claude-haiku-5-5",
        "cli_model": "claude-haiku-5-5",
        "quota_key": "haiku",
        "pool_id": "claude_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "claude",
        "tier": "LOW",
        "tier_rank": 1,
        "effort": "low",
        "default_preferred": True,
        "context_window": 200_000,
        "latency_score": 0.94,
        "cost_weight": 0.90,
        "label": "Claude Code · Haiku 5.5 (Low)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-4-hydrogen-medium",
        "cli_model": "gemini-4-hydrogen-medium",
        "quota_key": "gemini_flash",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 1_000_000,
        "latency_score": 0.93,
        "cost_weight": 0.94,
        "label": "AGY · Gemini 4 Hydrogen (Medium)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-4-hydrogen-high",
        "cli_model": "gemini-4-hydrogen-high",
        "quota_key": "gemini_flash",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "high",
        "default_preferred": True,
        "context_window": 1_000_000,
        "latency_score": 0.90,
        "cost_weight": 0.92,
        "label": "AGY · Gemini 4 Hydrogen (High)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-3.8-flash-medium",
        "cli_model": "gemini-3.8-flash-medium",
        "quota_key": "gemini_flash",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "medium",
        "default_preferred": False,
        "context_window": 1_000_000,
        "latency_score": 0.92,
        "cost_weight": 0.93,
        "label": "AGY · Gemini 3.8 Flash (Medium - Fallback)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-3.8-flash-high",
        "cli_model": "gemini-3.8-flash-high",
        "quota_key": "gemini_flash",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "high",
        "default_preferred": False,
        "context_window": 1_000_000,
        "latency_score": 0.89,
        "cost_weight": 0.91,
        "label": "AGY · Gemini 3.8 Flash (High - Fallback)",
    },
    {
        "provider": "claude",
        "model_id": "claude-sonnet-5-5",
        "cli_model": "claude-sonnet-5-5",
        "quota_key": "sonnet",
        "pool_id": "claude_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "claude",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 200_000,
        "latency_score": 0.86,
        "cost_weight": 0.86,
        "label": "Claude Code · Sonnet 5.5",
    },
    {
        "provider": "agy",
        "model_id": "claude-sonnet-5-5-medium",
        "cli_model": "claude-sonnet-5-5-medium",
        "quota_key": "claude_sonnet",
        "pool_id": "agy_claude_gpt",
        "pool_windows": ("claude_gpt_5h", "claude_gpt_weekly"),
        "family": "claude",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 200_000,
        "latency_score": 0.85,
        "cost_weight": 0.85,
        "label": "AGY · Claude Sonnet 5.5 (Medium)",
    },
    {
        "provider": "codex",
        "model_id": "gpt-6.1-sol-medium",
        "cli_model": "gpt-6.1-sol",
        "quota_key": "terra",
        "pool_id": "codex_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "openai",
        "tier": "MID",
        "tier_rank": 2,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 475_000,
        "latency_score": 0.84,
        "cost_weight": 0.82,
        "label": "Codex · GPT-6.1-Sol (Medium)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-4-argon-medium",
        "cli_model": "gemini-4-argon-medium",
        "quota_key": "gemini_pro",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 1_000_000,
        "latency_score": 0.82,
        "cost_weight": 0.90,
        "label": "AGY · Gemini 4 Argon (Medium)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-4-argon-low",
        "cli_model": "gemini-4-argon-low",
        "quota_key": "gemini_pro",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "low",
        "default_preferred": True,
        "context_window": 1_000_000,
        "latency_score": 0.86,
        "cost_weight": 0.92,
        "label": "AGY · Gemini 4 Argon (Low)",
    },
    {
        "provider": "claude",
        "model_id": "claude-opus-5-5",
        "cli_model": "claude-opus-5-5",
        "quota_key": "opus",
        "pool_id": "claude_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "claude",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 1_000_000,
        "latency_score": 0.78,
        "cost_weight": 0.80,
        "label": "Claude Code · Opus 5.5",
    },
    {
        "provider": "codex",
        "model_id": "gpt-6.1-sol",
        "cli_model": "gpt-6.1-sol",
        "quota_key": "sol",
        "pool_id": "codex_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "openai",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 475_000,
        "latency_score": 0.80,
        "cost_weight": 0.80,
        "label": "Codex · GPT-6.1-Sol",
    },
    {
        "provider": "agy",
        "model_id": "claude-opus-5-5-medium",
        "cli_model": "claude-opus-5-5-medium",
        "quota_key": "claude_opus",
        "pool_id": "agy_claude_gpt",
        "pool_windows": ("claude_gpt_5h", "claude_gpt_weekly"),
        "family": "claude",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "medium",
        "default_preferred": True,
        "context_window": 200_000,
        "latency_score": 0.76,
        "cost_weight": 0.84,
        "label": "AGY · Claude Opus 5.5 (Medium)",
    },
    {
        "provider": "agy",
        "model_id": "gemini-4-argon-high",
        "cli_model": "gemini-4-argon-high",
        "quota_key": "gemini_pro",
        "pool_id": "agy_gemini",
        "pool_windows": ("gemini_5h", "gemini_weekly"),
        "family": "gemini",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "high",
        "default_preferred": False,
        "context_window": 1_000_000,
        "latency_score": 0.74,
        "cost_weight": 0.82,
        "label": "AGY · Gemini 4 Argon (High - Exception Only)",
    },
    {
        "provider": "agy",
        "model_id": "claude-opus-5-5-high",
        "cli_model": "claude-opus-5-5-high",
        "quota_key": "claude_opus",
        "pool_id": "agy_claude_gpt",
        "pool_windows": ("claude_gpt_5h", "claude_gpt_weekly"),
        "family": "claude",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "high",
        "default_preferred": False,
        "context_window": 200_000,
        "latency_score": 0.72,
        "cost_weight": 0.78,
        "label": "AGY · Claude Opus 5.5 (High)",
    },
    {
        "provider": "codex",
        "model_id": "gpt-6.1-sol-high",
        "cli_model": "gpt-6.1-sol",
        "quota_key": "sol",
        "pool_id": "codex_shared",
        "pool_windows": ("five_hour", "weekly"),
        "family": "openai",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "high",
        "default_preferred": False,
        "context_window": 475_000,
        "latency_score": 0.75,
        "cost_weight": 0.76,
        "label": "Codex · GPT-6.1-Sol (High)",
    },
]


REAL_PC_AGY_MODELS_TEXT = """Fetching available models...
gemini-4-argon-medium\tGemini 4 Argon (Medium)
gemini-4-argon-high\tGemini 4 Argon (High)
gemini-4-argon-low\tGemini 4 Argon (Low)
gemini-3.8-flash-high\tGemini 3.8 Flash (High)
gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)
gemini-3.8-flash-low\tGemini 3.8 Flash (Low)
claude-opus-5-5-low\tClaude Opus 5.5 (Low)
claude-opus-5-5-medium\tClaude Opus 5.5 (Medium)
claude-opus-5-5-high\tClaude Opus 5.5 (High)
claude-sonnet-5-5-low\tClaude Sonnet 5.5 (Low)
claude-sonnet-5-5-medium\tClaude Sonnet 5.5 (Medium)
claude-sonnet-5-5-high\tClaude Sonnet 5.5 (High)
gpt-oss-120b-medium\tGPT-OSS 120B (Medium)
"""

FAKE_AGY_MODELS_TEXT = """Fetching available models...
gemini-4-argon-medium\tGemini 4 Argon (Medium)
gemini-4-argon-high\tGemini 4 Argon (High)
gemini-4-argon-low\tGemini 4 Argon (Low)
gemini-3.8-flash-high\tGemini 3.8 Flash (High)
gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)
gemini-3.8-flash-low\tGemini 3.8 Flash (Low)
claude-sonnet-5-5-medium\tClaude Sonnet 5.5 (Medium)
claude-opus-5-5-medium\tClaude Opus 5.5 (Medium)
claude-opus-5-5-high\tClaude Opus 5.5 (High)
gemini-4.2-flash-medium\tGemini 4.2 Flash (Medium)
gemini-4-astra-high\tGemini 4 Astra (High)
gemini-local-experimental-v1\tGemini Local Experimental v1
"""

FAKE_MODELS_DEV_JSON: Dict[str, Any] = {
    "google": {
        "models": {
            "gemini-4-argon": {
                "id": "google/gemini-4-argon",
                "name": "Gemini 4 Argon",
                "cost": {"input": 1.25, "output": 5.00},
                "limit": {"context": 1_000_000},
                "release_date": "2026-09-01",
                "coding_score": 78.5,
            },
            "gemini-4.2-flash": {
                "id": "google/gemini-4.2-flash",
                "name": "Gemini 4.2 Flash",
                "cost": {"input": 0.50, "output": 2.00},
                "limit": {"context": 1_000_000},
                "release_date": "2026-10-20",
                "coding_score": 71.0,
            },
            "gemini-4-astra": {
                "id": "google/gemini-4-astra",
                "name": "Gemini 4 Astra Ultra",
                "cost": {"input": 8.00, "output": 24.00},
                "limit": {"context": 2_000_000},
                "release_date": "2026-10-25",
                "coding_score": 86.0,
            },
        }
    },
    "anthropic": {
        "models": {
            "claude-opus-5-5": {
                "id": "anthropic/claude-opus-5-5",
                "name": "Claude Opus 5.5",
                "cost": {"input": 15.0, "output": 75.0},
                "limit": {"context": 1_000_000},
                "coding_score": 82.0,
            }
        }
    },
    "openai": {
        "models": {
            "gpt-6.1-sol": {
                "id": "openai/gpt-6.1-sol",
                "name": "GPT-6.1-Sol",
                "cost": {"input": 2.50, "output": 10.00},
                "limit": {"context": 475_000},
                "coding_score": 80.0,
            }
        }
    },
}

FAKE_OPENROUTER_JSON: Dict[str, Any] = {
    "data": [
        {
            "id": "anthropic/claude-5.6-sonnet",
            "name": "Anthropic: Claude 5.6 Sonnet",
            "context_length": 250_000,
            "created": "2026-10-22",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "benchmarks": {"coding": 76.2},
        },
        {
            "id": "openai/gpt-6.2-sol-mini",
            "name": "OpenAI: GPT-6.2-Sol Mini",
            "context_length": 475_000,
            "created": "2026-10-24",
            "pricing": {"prompt": "0.0000006", "completion": "0.0000024"},
            "coding_score": 68.4,
        },
    ]
}


def _make_healthy_quotas(rem_5h: float = 0.85, rem_wk: float = 0.85) -> Dict[str, Any]:
    return {
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": rem_wk,
            "gemini_pro": rem_wk,
            "claude_sonnet": rem_wk,
            "claude_opus": rem_wk,
            "windows": {
                "gemini_5h": {"used_percent": (1.0 - rem_5h) * 100.0, "remaining_fraction": rem_5h, "reset_in_sec": 14400},
                "gemini_weekly": {"used_percent": (1.0 - rem_wk) * 100.0, "remaining_fraction": rem_wk, "reset_in_sec": 432000},
                "claude_gpt_5h": {"used_percent": (1.0 - rem_5h) * 100.0, "remaining_fraction": rem_5h, "reset_in_sec": 14400},
                "claude_gpt_weekly": {"used_percent": (1.0 - rem_wk) * 100.0, "remaining_fraction": rem_wk, "reset_in_sec": 432000},
            },
            "reset_in_sec": 14400,
        },
        "claude": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": rem_wk,
            "opus": rem_wk,
            "haiku": rem_wk,
            "windows": {
                "five_hour": {"used_percent": (1.0 - rem_5h) * 100.0, "remaining_fraction": rem_5h, "reset_in_sec": 14400},
                "weekly": {"used_percent": (1.0 - rem_wk) * 100.0, "remaining_fraction": rem_wk, "reset_in_sec": 432000},
            },
            "reset_in_sec": 14400,
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": rem_wk,
            "terra": rem_wk,
            "sol": rem_wk,
            "windows": {
                "five_hour": {"used_percent": (1.0 - rem_5h) * 100.0, "remaining_fraction": rem_5h, "reset_in_sec": 14400},
                "weekly": {"used_percent": (1.0 - rem_wk) * 100.0, "remaining_fraction": rem_wk, "reset_in_sec": 432000},
            },
            "reset_in_sec": 14400,
        },
    }


@pytest.fixture(autouse=True)
def isolated_model_review_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Redirect all catalog, hub state, policy, and cache paths to `tmp_path` so tests never
    touch the real `~/.agents`, the live hub directory, or the live `model_catalog.json`.
    """
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    tmp_catalog = tmp_path / "model_catalog.json"
    tmp_catalog.write_text(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setenv("QUOTA_HUB_HOME", str(hub_dir))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_catalog))
    monkeypatch.setenv("ORCHESTRATOR_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("QUOTA_HUB_DISABLE_SUPERVISOR", "1")
    monkeypatch.setattr(hub_main, "STATE_FILE", hub_dir / "quota_hub_state.json")

    reload_model_catalog(tmp_catalog)
    reset_probation_counters()
    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()
    yield {
        "hub_dir": hub_dir,
        "catalog_path": tmp_catalog,
    }
    MODEL_CATALOG[:] = copy.deepcopy(LEGACY_MODEL_CATALOG)
    reset_probation_counters()
    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()


def test_catalog_round_trip_identical_to_legacy_list(tmp_path: Path) -> None:
    """
    1. Catalog round-trip:
       - Loading `model_catalog.json` yields the exact 16-entry `LEGACY_MODEL_CATALOG` list (`==`).
       - Saving via `save_model_catalog` to a temp path and reloading yields the exact same list.
    """
    raw_doc = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    assert raw_doc["version"] == 1
    assert len(raw_doc["models"]) == 19

    loaded = load_model_catalog(DEFAULT_CATALOG_PATH)
    assert loaded == LEGACY_MODEL_CATALOG
    assert MODEL_CATALOG == LEGACY_MODEL_CATALOG

    rt_path = tmp_path / "roundtrip" / "model_catalog.json"
    save_res = save_model_catalog(loaded, path=rt_path)
    assert Path(save_res["catalog_path"]).exists()
    assert Path(save_res["backup_path"]).exists()

    reloaded = load_model_catalog(rt_path)
    assert reloaded == LEGACY_MODEL_CATALOG


def test_catalog_validation_errors(tmp_path: Path) -> None:
    """
    2. Validation errors:
       - Missing required fields, invalid types/values, invalid optional fields, and duplicate model_ids
         raise `ValueError`.
    """
    base_entry = copy.deepcopy(LEGACY_MODEL_CATALOG[0])

    # Missing required field
    missing_tier = copy.deepcopy(base_entry)
    del missing_tier["tier"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_entry(missing_tier)

    # Unsupported field
    extra_field = {**base_entry, "unexpected_key": 123}
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_model_entry(extra_field)

    # Invalid provider
    with pytest.raises(ValueError, match="provider"):
        validate_model_entry({**base_entry, "provider": "invalid_prov"})

    # Mismatched tier_rank
    with pytest.raises(ValueError, match="tier_rank"):
        validate_model_entry({**base_entry, "tier": "HIGH", "tier_rank": 1})

    # Invalid context_window (boolean or <= 0)
    with pytest.raises(ValueError, match="context_window"):
        validate_model_entry({**base_entry, "context_window": 0})
    with pytest.raises(ValueError, match="context_window"):
        validate_model_entry({**base_entry, "context_window": True})

    # Invalid optional status & probation_share
    with pytest.raises(ValueError, match="status"):
        validate_model_entry({**base_entry, "status": " archived "})
    with pytest.raises(ValueError, match="probation_share"):
        validate_model_entry({**base_entry, "probation_share": 1.5})
    with pytest.raises(ValueError, match="price_in_per_mtok"):
        validate_model_entry({**base_entry, "price_in_per_mtok": -0.5})

    # Duplicate model_id in catalog document
    with pytest.raises(ValueError, match="Duplicate model_id"):
        validate_model_catalog_doc({"version": 1, "models": [base_entry, copy.deepcopy(base_entry)]})

    # Empty models list or bad version
    with pytest.raises(ValueError, match="non-empty list"):
        validate_model_catalog_doc({"version": 1, "models": []})
    with pytest.raises(ValueError, match="version"):
        validate_model_catalog_doc({"version": 0, "models": [base_entry]})


def test_discovery_diff_and_normalised_matching(isolated_model_review_env: Dict[str, Path]) -> None:
    """
    3. Discovery diff & normalised matching:
       - Parses fake `agy models` output (including brace patterns), fake `~/.codex/models_cache.json`,
         static Claude list, and dispatch history.
       - Matches local model IDs to `models.dev` and `OpenRouter` fixture entries by normalised name.
       - Flags unmatched local models with `metadata: "unknown"` and `"unknown_price"`.
       - Detects existing catalog models whose public blended price changed by > 20%.
    """
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]

    parsed_real_pc = parse_agy_models_output(REAL_PC_AGY_MODELS_TEXT)
    assert parsed_real_pc == [
        "gemini-4-argon-medium",
        "gemini-4-argon-high",
        "gemini-4-argon-low",
        "gemini-3.8-flash-high",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-low",
        "claude-opus-5-5-low",
        "claude-opus-5-5-medium",
        "claude-opus-5-5-high",
        "claude-sonnet-5-5-low",
        "claude-sonnet-5-5-medium",
        "claude-sonnet-5-5-high",
        "gpt-oss-120b-medium",
    ]

    parsed_agy = parse_agy_models_output(FAKE_AGY_MODELS_TEXT)
    assert "gemini-3.8-flash-low" in parsed_agy
    assert "gemini-4.2-flash-medium" in parsed_agy
    assert "gemini-4-astra-high" in parsed_agy
    assert "gemini-local-experimental-v1" in parsed_agy

    fake_codex_cache = {
        "models": [
            {"id": "gpt-6.1-sol", "context_window": 475_000},
            {"id": "gpt-6.2-sol-mini", "context_window": 475_000},
        ]
    }
    fake_history = [
        {"provider": "claude", "model_id": "claude-sonnet-5-6"},
    ]

    report = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output=FAKE_AGY_MODELS_TEXT,
        codex_cache_data=fake_codex_cache,
        claude_static_models=["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
        dispatch_history=fake_history,
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data=FAKE_OPENROUTER_JSON,
    )

    prop_by_id = {p["model_id"]: p for p in report["proposals"]}
    excl_by_id = {e["model_id"]: e for e in report["excluded_too_expensive"]}

    # Existing catalog models are not proposed as new
    assert "gemini-3.8-flash-low" not in prop_by_id
    assert "gpt-6.1-sol" not in prop_by_id
    assert "claude-opus-5-5" not in prop_by_id

    # Newly discovered affordable models are in proposals with matched metadata
    assert "gemini-4.2-flash-medium" in prop_by_id
    gem42 = prop_by_id["gemini-4.2-flash-medium"]
    assert gem42["metadata"] == "matched"
    assert gem42["coding_score"] == 71.0
    assert gem42["context_window"] == 1_000_000
    assert gem42["sources"][0]["name"] == "models.dev"

    # Claude 5.6 Sonnet from dispatch history matched against OpenRouter (`anthropic/claude-5.6-sonnet`)
    assert "claude-sonnet-5-6" in prop_by_id
    cl56 = prop_by_id["claude-sonnet-5-6"]
    assert cl56["metadata"] == "matched"
    assert cl56["price_in_per_mtok"] == pytest.approx(3.0)
    assert cl56["price_out_per_mtok"] == pytest.approx(15.0)
    assert cl56["coding_score"] == pytest.approx(76.2)

    # Codex `gpt-6.2-sol-mini` from models_cache.json matched against OpenRouter
    assert "gpt-6.2-sol-mini" in prop_by_id
    cx62 = prop_by_id["gpt-6.2-sol-mini"]
    assert cx62["metadata"] == "matched"
    # cheap price band: never suggested as HIGH; LOW or MID depending on its coding score (owner picks the tier on approval)
    assert cx62["suggested_tier"] in {"LOW", "MID"}

    # Unmatched local model has metadata == "unknown" and unknown_price flag
    assert "gemini-local-experimental-v1" in prop_by_id
    unmatched = prop_by_id["gemini-local-experimental-v1"]
    assert unmatched["metadata"] == "unknown"
    assert unmatched["unknown_price"] is True
    assert "unknown_price" in unmatched["flags"]

    # Expensive Astra model is excluded, not in proposals
    assert "gemini-4-astra-high" not in prop_by_id
    assert "gemini-4-astra-high" in excl_by_id

    # Verify >20% price change on an existing catalog model triggers a review proposal
    cat = load_model_catalog(cat_path)
    for m in cat:
        if m["model_id"] == "gpt-6.1-sol":
            # Stored price: in=1.0, out=4.0 (blended=3.25); public is in=2.5, out=10.0 (blended=8.125, +150% > 20%)
            m["price_in_per_mtok"] = 1.0
            m["price_out_per_mtok"] = 4.0
    save_model_catalog(cat, path=cat_path)

    report_price_chg = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="",
        codex_cache_data={"models": ["gpt-6.1-sol"]},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data=FAKE_OPENROUTER_JSON,
    )
    chg_props = {p["model_id"]: p for p in report_price_chg["proposals"]}
    assert "gpt-6.1-sol" in chg_props
    assert chg_props["gpt-6.1-sol"]["change_reason"] == "price_changed_gt_20pct"
    assert chg_props["gpt-6.1-sol"]["price_change_pct"] > 0.20


def test_price_ratio_calculation_and_astra_exclusion(isolated_model_review_env: Dict[str, Path]) -> None:
    """
    4. Price ratio and exclusion:
       - `blended_price = 0.25 * in + 0.75 * out`
       - `price_ratio = blended_price / blended_price(active HIGH model of same provider)`
       - Models with `price_ratio > max_price_ratio` are listed as `"excluded (too expensive)"` with reason
         and never proposed.
    """
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]

    # Baseline AGY HIGH (`gemini-4-argon-medium`): in=1.25, out=5.00 -> blended = 0.25*1.25 + 0.75*5.00 = 4.0625
    assert compute_blended_price(1.25, 5.00) == pytest.approx(4.0625)
    # Candidate `gemini-4.2-flash-medium`: in=0.50, out=2.00 -> blended = 1.625 -> ratio = 1.625 / 4.0625 = 0.40
    assert compute_blended_price(0.50, 2.00) == pytest.approx(1.625)
    # Candidate `gemini-4-astra-high`: in=8.00, out=24.00 -> blended = 20.0 -> ratio = 20.0 / 4.0625 = 4.9231 (> 2.0)
    assert compute_blended_price(8.00, 24.00) == pytest.approx(20.0)

    report = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium\ngemini-4-astra-high",
        codex_cache_data={"models": []},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data={"data": []},
        settings_override={"max_price_ratio": 2.0},
    )

    prop_by_id = {p["model_id"]: p for p in report["proposals"]}
    excl_by_id = {e["model_id"]: e for e in report["excluded_too_expensive"]}

    assert "gemini-4.2-flash-medium" in prop_by_id
    assert prop_by_id["gemini-4.2-flash-medium"]["price_ratio"] == pytest.approx(0.40, abs=1e-3)
    assert prop_by_id["gemini-4.2-flash-medium"]["baseline_high_model_id"] == "gemini-4-argon-medium"

    assert "gemini-4-astra-high" not in prop_by_id
    assert "gemini-4-astra-high" in excl_by_id
    astra = excl_by_id["gemini-4-astra-high"]
    assert astra["status"] == "excluded (too expensive)"
    assert astra["price_ratio"] == pytest.approx(20.0 / 4.0625, abs=1e-3)
    assert "excluded (too expensive)" in astra["reason"]


def test_reject_cooldown_30_days(isolated_model_review_env: Dict[str, Path]) -> None:
    """
    5. Reject cooldown:
       - Rejecting a model sets a 30-day cooldown for that `model_id` without modifying `model_catalog.json`.
       - Reviews within 30 days skip that `model_id`; after 30 days it can be proposed again.
    """
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]
    t0 = datetime.datetime(2026, 10, 27, 12, 0, 0, tzinfo=datetime.timezone.utc)

    run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium",
        codex_cache_data={"models": []},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data={"data": []},
        now_dt=t0,
    )

    res = apply_model_decision(
        model_id="gemini-4.2-flash-medium",
        decision="reject",
        catalog_path=cat_path,
        hub_dir=hub_dir,
        now_dt=t0,
    )
    assert res["catalog_updated"] is False
    assert res["cooldown"]["cooldown_days"] == 30
    assert len(load_model_catalog(cat_path)) == 19

    # 10 days later: still in 30-day cooldown -> not proposed
    t_day10 = t0 + datetime.timedelta(days=10)
    rep_10 = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium",
        codex_cache_data={"models": []},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data={"data": []},
        now_dt=t_day10,
    )
    assert all(p["model_id"] != "gemini-4.2-flash-medium" for p in rep_10["proposals"])
    assert any(c["model_id"] == "gemini-4.2-flash-medium" for c in rep_10["cooldown_skipped"])

    # 31 days later: cooldown expired -> proposed again
    t_day31 = t0 + datetime.timedelta(days=31)
    rep_31 = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium",
        codex_cache_data={"models": []},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data={"data": []},
        now_dt=t_day31,
    )
    assert any(p["model_id"] == "gemini-4.2-flash-medium" for p in rep_31["proposals"])


def test_probation_share_respected_over_1000_decisions_and_never_design_or_review(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    """
    6. Probation & retired routing:
       - Over 1,000 eligible LOW/MID routing decisions, a `probation` model is selected at most
         `probation_share` (0.10) of decisions (deterministic counter after SWRR provider choice).
       - Never used for `kind in {'design', 'review'}` or `floor == 'HIGH'`.
       - `retired` models are never selected for any package.
    """
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]

    # Put `gemini-4.2-flash-medium` on probation (share = 0.10, tier = MID)
    run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium",
        codex_cache_data={"models": []},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data={"data": []},
        settings_override={"probation_share": 0.10},
    )
    apply_model_decision(
        model_id="gemini-4.2-flash-medium",
        decision="probation",
        tier="MID",
        catalog_path=cat_path,
        hub_dir=hub_dir,
    )

    # Also add a retired model to verify it is never routed
    cat = load_model_catalog(cat_path)
    cat.append(
        {
            "provider": "agy",
            "model_id": "gemini-3.5-retired-low",
            "cli_model": "gemini-3.5-retired-low",
            "quota_key": "gemini_flash",
            "pool_id": "agy_gemini",
            "pool_windows": ("gemini_5h", "gemini_weekly"),
            "family": "gemini",
            "tier": "LOW",
            "tier_rank": 1,
            "effort": "low",
            "default_preferred": True,
            "context_window": 1_000_000,
            "latency_score": 0.99,
            "cost_weight": 0.99,
            "label": "AGY · Gemini 3.5 Retired",
            "status": "retired",
        }
    )
    save_model_catalog(cat, path=cat_path)
    reload_model_catalog(cat_path)

    quotas = _make_healthy_quotas()
    orch_info = select_active_orchestrator("auto", quotas)
    learner = QuotaCostLearner(tmp_path / "probation_1000.sqlite")
    b_learner = BurnRateLearner(tmp_path / "probation_1000.sqlite", cost_learner=learner)

    reset_probation_counters()
    reset_allocation_credits()

    probation_picks = 0
    retired_picks = 0
    agy_decisions = 0

    for i in range(1000):
        res = route_work_package(
            {
                "task": f"WP-PROB-{i}",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "module",
                "dependency_role": "leaf",
                "context_need": "pointer",
            },
            quotas,
            orch_info,
            cost_learner=learner,
            burn_learner=b_learner,
        )
        assert res["status"] == "routed"
        ar = res["assignment_record"]
        if ar["provider"] == "agy":
            agy_decisions += 1
        if ar["model"] == "gemini-4.2-flash-medium":
            probation_picks += 1
        if ar["model"] == "gemini-3.5-retired-low":
            retired_picks += 1

    assert retired_picks == 0
    assert agy_decisions > 0
    assert probation_picks > 0
    # Probation share is <= 0.10 of AGY decisions AND <= 0.10 of all 1,000 routing decisions
    # The share is measured against ALL probation-eligible decisions (global counter), not just AGY's.
    assert (probation_picks / 1000.0) <= 0.10 + 1e-9

    # Never used for design, review, or HIGH floor even across 50 decisions
    for forbidden_req in (
        {"task": "WP-DESIGN", "kind": "design", "ambiguity": "low", "blast_radius": "local", "preferred_provider": "agy"},
        {"task": "WP-REVIEW", "kind": "review", "ambiguity": "low", "blast_radius": "local", "preferred_provider": "agy"},
        {"task": "WP-HIGH", "kind": "implement", "ambiguity": "high", "blast_radius": "cross-module", "preferred_provider": "agy"},
    ):
        for _ in range(25):
            r = route_work_package(forbidden_req, quotas, orch_info, cost_learner=learner, burn_learner=b_learner)
            assert r["assignment_record"]["model"] != "gemini-4.2-flash-medium"
            assert r["assignment_record"]["model"] != "gemini-3.5-retired-low"
            rej_by_id = {rm["model_id"]: rm for rm in r["rejected_models"]}
            assert "gemini-4.2-flash-medium" in rej_by_id
            assert rej_by_id["gemini-4.2-flash-medium"].get("reason")
            assert "probation" in rej_by_id["gemini-4.2-flash-medium"]["reason"].lower()
            assert "gemini-3.5-retired-low" in rej_by_id
            assert rej_by_id["gemini-3.5-retired-low"].get("reason")
            assert "retired" in rej_by_id["gemini-3.5-retired-low"]["reason"].lower()


def test_probation_14_day_scorecard_evaluation(isolated_model_review_env: Dict[str, Path], tmp_path: Path) -> None:
    """
    After 14 days of probation, `evaluate_probation_models` / `run_model_review` summarises the scorecard
    (acceptance, rework, measured quota cost) and proposes `promote` or `retire` without auto-promoting.
    """
    cat_path = isolated_model_review_env["catalog_path"]
    hub_dir = isolated_model_review_env["hub_dir"]
    now = datetime.datetime(2026, 10, 27, 12, 0, 0, tzinfo=datetime.timezone.utc)
    added_15d_ago = (now - datetime.timedelta(days=15)).isoformat()

    cat = load_model_catalog(cat_path)
    cat.extend(
        [
            {
                "provider": "agy",
                "model_id": "gemini-4.2-flash-good",
                "cli_model": "gemini-4.2-flash-good",
                "quota_key": "gemini_flash",
                "pool_id": "agy_gemini",
                "pool_windows": ("gemini_5h", "gemini_weekly"),
                "family": "gemini",
                "tier": "MID",
                "tier_rank": 2,
                "effort": "medium",
                "default_preferred": True,
                "context_window": 1_000_000,
                "latency_score": 0.90,
                "cost_weight": 0.90,
                "label": "AGY · Good Probation",
                "status": "probation",
                "probation_share": 0.10,
                "added_at": added_15d_ago,
            },
            {
                "provider": "agy",
                "model_id": "gemini-4.2-flash-weak",
                "cli_model": "gemini-4.2-flash-weak",
                "quota_key": "gemini_flash",
                "pool_id": "agy_gemini",
                "pool_windows": ("gemini_5h", "gemini_weekly"),
                "family": "gemini",
                "tier": "MID",
                "tier_rank": 2,
                "effort": "medium",
                "default_preferred": True,
                "context_window": 1_000_000,
                "latency_score": 0.90,
                "cost_weight": 0.90,
                "label": "AGY · Weak Probation",
                "status": "probation",
                "probation_share": 0.10,
                "added_at": added_15d_ago,
            },
        ]
    )
    save_model_catalog(cat, path=cat_path)

    learner = QuotaCostLearner(tmp_path / "scorecard_prob.sqlite")
    for idx in range(4):
        learner.record_observation(
            task_id=f"GOOD-{idx}",
            provider="agy",
            model_id="gemini-4.2-flash-good",
            pool_id="agy_gemini",
            task_kind="implement",
            tier="MID",
            pre_5h_used_pct=10.0,
            post_5h_used_pct=13.0,
            pre_weekly_used_pct=20.0,
            post_weekly_used_pct=21.0,
            acceptance_criteria_met=True,
            tests_passed=True,
            feature_accepted=True,
            rework_count=0,
        )
        learner.record_observation(
            task_id=f"WEAK-{idx}",
            provider="agy",
            model_id="gemini-4.2-flash-weak",
            pool_id="agy_gemini",
            task_kind="implement",
            tier="MID",
            pre_5h_used_pct=10.0,
            post_5h_used_pct=16.0,
            pre_weekly_used_pct=20.0,
            post_weekly_used_pct=22.0,
            acceptance_criteria_met=False,
            tests_passed=True,
            feature_accepted=False,
            rework_count=2,
        )

    reviews = evaluate_probation_models(
        load_model_catalog(cat_path),
        scorecard=learner.get_agent_scorecard(),
        now_dt=now,
    )
    by_id = {r["model_id"]: r for r in reviews}
    assert by_id["gemini-4.2-flash-good"]["proposed_action"] == "promote"
    assert by_id["gemini-4.2-flash-good"]["auto_promoted"] is False
    assert by_id["gemini-4.2-flash-good"]["measured_5h_cost_pct"] == pytest.approx(3.0)

    assert by_id["gemini-4.2-flash-weak"]["proposed_action"] == "retire"
    assert by_id["gemini-4.2-flash-weak"]["auto_promoted"] is False

    # Confirm neither model was auto-promoted in model_catalog.json
    reloaded = {m["model_id"]: m for m in load_model_catalog(cat_path)}
    assert reloaded["gemini-4.2-flash-good"]["status"] == "probation"
    assert reloaded["gemini-4.2-flash-weak"]["status"] == "probation"


def test_stale_cache_on_network_error(isolated_model_review_env: Dict[str, Path]) -> None:
    """
    7. Stale cache on network error:
       - Successful fetch populates the 24h cache with ETag.
       - Subsequent network failure retains the last cached metadata and reports `stale=True` / `staleness="stale"`.
    """
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]
    t0 = 1_793_000_000.0

    def ok_fetcher(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        if "models.dev" in url:
            return 200, {"etag": '"mdev-etag-1"'}, FAKE_MODELS_DEV_JSON
        return 200, {"etag": '"or-etag-1"'}, FAKE_OPENROUTER_JSON

    meta_ok = fetch_public_metadata(hub_dir=hub_dir, http_fetcher=ok_fetcher, force_refresh=True, now_ts=t0)
    assert meta_ok["stale"] is False
    assert meta_ok["staleness"] == "fresh"
    assert len(meta_ok["entries"]) >= 5

    seen_if_none_match: List[str] = []

    def failing_fetcher(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        if "If-None-Match" in headers:
            seen_if_none_match.append(headers["If-None-Match"])
        raise urllib.error.URLError("simulated offline network failure")

    # 25 hours later (cache expired) -> network error falls back to last cache and marks stale
    meta_stale = fetch_public_metadata(
        hub_dir=hub_dir,
        http_fetcher=failing_fetcher,
        force_refresh=False,
        now_ts=t0 + 25 * 3600,
    )
    assert meta_stale["stale"] is True
    assert meta_stale["staleness"] == "stale"
    assert '"mdev-etag-1"' in seen_if_none_match
    assert len(meta_stale["entries"]) == len(meta_ok["entries"])

    report_stale = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium",
        codex_cache_data={"models": []},
        claude_static_models=[],
        http_fetcher=failing_fetcher,
        force_refresh_metadata=True,
    )
    assert report_stale["stale"] is True
    assert report_stale["staleness"] == "stale"
    assert any(p["model_id"] == "gemini-4.2-flash-medium" and p["metadata"] == "matched" for p in report_stale["proposals"])


def test_approve_writes_catalog_atomically_with_backup_and_endpoints(
    isolated_model_review_env: Dict[str, Path]
) -> None:
    """
    8. Approve writes `model_catalog.json` atomically with a timestamped backup and updates FastAPI endpoints.
    """
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]

    run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output=FAKE_AGY_MODELS_TEXT,
        codex_cache_data={"models": ["gpt-6.2-sol-mini"]},
        claude_static_models=["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data=FAKE_OPENROUTER_JSON,
    )

    client = TestClient(app)

    # GET /api/models/review
    r_get = client.get("/api/models/review")
    assert r_get.status_code == 200
    rev_payload = r_get.json()
    assert any(p["model_id"] == "gemini-4.2-flash-medium" for p in rev_payload["proposals"])
    assert any(e["model_id"] == "gemini-4-astra-high" for e in rev_payload["excluded_too_expensive"])

    # POST /api/models/decision -> approve `gemini-4.2-flash-medium`
    r_dec = client.post(
        "/api/models/decision",
        json={"model_id": "gemini-4.2-flash-medium", "decision": "approve", "tier": "MID"},
    )
    assert r_dec.status_code == 200
    dec_data = r_dec.json()
    assert dec_data["catalog_updated"] is True
    assert dec_data["status"] == "active"
    assert dec_data["tier"] == "MID"
    backup_path = Path(dec_data["backup_path"])
    assert backup_path.exists()
    assert backup_path.parent == cat_path.parent
    assert ".bak-" in backup_path.name

    # Verify model is now in `model_catalog.json` and in-memory `MODEL_CATALOG`
    disk_models = {m["model_id"]: m for m in load_model_catalog(cat_path)}
    assert "gemini-4.2-flash-medium" in disk_models
    assert disk_models["gemini-4.2-flash-medium"]["status"] == "active"
    assert any(m["model_id"] == "gemini-4.2-flash-medium" for m in MODEL_CATALOG)

    # Verify dashboard HTML includes the Model Review card
    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "model-review-card" in r_html.text
    assert "Excluded (too expensive)" in r_html.text


# ---------------------------------------------------------------------------
# Independent-review fixes (catalog recovery, id validation, probation fallback,
# non-blocking manual review, state merge)
# ---------------------------------------------------------------------------


def test_missing_or_corrupt_catalog_recovers_from_backup_then_builtin(tmp_path: Path) -> None:
    from model_review import BUILTIN_DEFAULT_MODELS, get_catalog_status, load_model_catalog_resilient

    cat = tmp_path / "cat" / "model_catalog.json"
    cat.parent.mkdir()
    cat.write_text(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    good = load_model_catalog(cat)
    assert len(BUILTIN_DEFAULT_MODELS) == 19
    save_model_catalog(good, path=cat)  # creates a timestamped backup of the good file

    cat.write_text("{ not json", encoding="utf-8")
    models = load_model_catalog_resilient(cat)
    assert [m["model_id"] for m in models] == [m["model_id"] for m in good]
    st = get_catalog_status()
    assert st["status"] == "restored_from_backup" and st["reason"]

    for bak in cat.parent.glob("model_catalog.json.bak-*"):
        bak.write_text("[]", encoding="utf-8")
    models = load_model_catalog_resilient(cat)
    assert len(models) == 19
    assert get_catalog_status()["status"] == "builtin_defaults"

    cat.unlink()
    assert len(load_model_catalog_resilient(cat)) == 19
    assert get_catalog_status()["status"] == "builtin_defaults"

    cat.write_text(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    load_model_catalog_resilient(cat)
    assert get_catalog_status()["status"] == "ok"


def test_catalog_status_exposed_on_endpoints(isolated_model_review_env: Dict[str, Path]) -> None:
    client = TestClient(app)
    assert "catalog_status" in client.get("/api/models/review").json()
    assert client.get("/api/quotas").json()["catalog_status"]["status"] in {"ok", "restored_from_backup", "builtin_defaults"}


def test_injected_model_id_rejected_and_never_reaches_html(isolated_model_review_env: Dict[str, Path]) -> None:
    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]
    evil = "x');alert(1);//"
    rep = run_model_review(
        catalog_path=cat_path,
        hub_dir=hub_dir,
        agy_models_output="gemini-4.2-flash-medium",
        codex_cache_data={"models": [evil, "gpt-6.2-sol-mini"]},
        claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON,
        openrouter_data={"data": []},
    )
    assert all(evil not in json.dumps(p) for p in rep["proposals"])
    assert any(r["model_id"].startswith("x'") for r in rep["rejected_ids"])
    with pytest.raises(ValueError):
        apply_model_decision(model_id=evil, decision="approve", catalog_path=cat_path, hub_dir=hub_dir)
    client = TestClient(app)
    r = client.post("/api/models/decision", json={"model_id": evil, "decision": "approve"})
    assert r.status_code == 400

    html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    start = html.index("function renderModelReview")
    end = html.index("async function fetchModelReview")
    block = html[start:end]
    assert "onclick=" not in block and "innerHTML" not in block
    assert "addEventListener('click'" in block
    assert "/^https:\\/\\//i" in block


def _single_provider_probation_catalog(cat_path: Path, prob_provider: str, share: float) -> None:
    cat = load_model_catalog(cat_path)
    for m in cat:
        if m["provider"] == prob_provider:
            m["status"] = "probation"
            m["probation_share"] = share
            m["added_at"] = "2026-10-01T00:00:00+00:00"
    save_model_catalog(cat, path=cat_path)
    reload_model_catalog(cat_path)
    reset_probation_counters()


def test_probation_only_provider_denied_never_routes_probation_or_crosses_providers(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    cat_path = isolated_model_review_env["catalog_path"]
    _single_provider_probation_catalog(cat_path, "claude", 0.0)
    quotas = _make_healthy_quotas()
    orch_info = select_active_orchestrator("auto", quotas)
    learner = QuotaCostLearner(tmp_path / "pf.sqlite")
    b_learner = BurnRateLearner(tmp_path / "pf.sqlite", cost_learner=learner)
    req = {"task": "WP-PF", "kind": "implement", "ambiguity": "low", "blast_radius": "module",
           "dependency_role": "leaf", "context_need": "pointer"}
    claude_ids = {m["model_id"] for m in load_model_catalog(cat_path) if m["provider"] == "claude"}

    # Only the probation-only provider enabled: nothing may be routed (never probation, never other providers).
    set_preference_weights({"agy": 0, "codex": 0, "claude": 10}, reset_swrr=True)
    reset_allocation_credits()
    for _ in range(20):
        res = route_work_package(dict(req), quotas, orch_info, cost_learner=learner, burn_learner=b_learner)
        assert res["status"] != "routed"
        assert res.get("assignment_record") is None

    # Preferred provider is the probation-only one: it is skipped; any routing stays on other providers.
    set_preference_weights({"agy": 10, "codex": 8, "claude": 10}, reset_swrr=True)
    reset_allocation_credits()
    for _ in range(40):
        res = route_work_package(dict(req, preferred_provider="claude"), quotas, orch_info,
                                 cost_learner=learner, burn_learner=b_learner)
        if res["status"] == "routed":
            assert res["assignment_record"]["model"] not in claude_ids

    # SWRR path: claude is chosen sometimes by credits but never routed.
    reset_allocation_credits()
    seen = set()
    for _ in range(60):
        res = route_work_package(dict(req), quotas, orch_info, cost_learner=learner, burn_learner=b_learner)
        assert res["status"] == "routed"
        seen.add(res["assignment_record"]["provider"])
    assert "claude" not in seen


def test_explicit_probation_request_counts_toward_share(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    from model_review import get_probation_counters

    cat_path = isolated_model_review_env["catalog_path"]
    _single_provider_probation_catalog(cat_path, "claude", 0.10)
    quotas = _make_healthy_quotas()
    orch_info = select_active_orchestrator("auto", quotas)
    learner = QuotaCostLearner(tmp_path / "ex.sqlite")
    b_learner = BurnRateLearner(tmp_path / "ex.sqlite", cost_learner=learner)
    prob_id = next(m["model_id"] for m in load_model_catalog(cat_path) if m["provider"] == "claude" and m["tier"] == "MID")
    res = route_work_package(
        {"task": "WP-EX", "kind": "implement", "ambiguity": "low", "blast_radius": "module",
         "dependency_role": "leaf", "context_need": "pointer", "explicit_model": prob_id},
        quotas, orch_info, cost_learner=learner, burn_learner=b_learner,
    )
    assert res["status"] == "routed"
    assert get_probation_counters()["by_provider"].get("claude", {}).get("eligible", 0) >= 1


def test_manual_review_endpoint_returns_quickly_while_discovery_is_slow(
    isolated_model_review_env: Dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    gate = threading.Event()

    def slow_review(**kwargs: Any) -> Dict[str, Any]:
        gate.wait(10)
        return {"proposals": []}

    monkeypatch.setattr(hub_main, "run_model_review", slow_review)
    monkeypatch.setattr(hub_main, "_REVIEW_JOB", {"run_id": None, "status": "idle"})
    client = TestClient(app)
    t0 = time.monotonic()
    r = client.post("/api/models/review/run", json={})
    assert time.monotonic() - t0 < 2.0
    assert r.status_code == 200
    body = r.json()
    assert body["review_run"]["status"] == "running" and body["review_run"]["run_id"]
    # Second click while running does not start another run.
    assert client.post("/api/models/review/run", json={}).json()["already_running"] is True
    assert client.get("/api/models/review").json()["review_run"]["status"] == "running"
    gate.set()
    for _ in range(100):
        if client.get("/api/models/review").json()["review_run"]["status"] != "running":
            break
        time.sleep(0.05)
    assert client.get("/api/models/review").json()["review_run"]["status"] == "done"


def test_concurrent_reject_during_slow_review_survives(
    isolated_model_review_env: Dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]
    kwargs = dict(
        catalog_path=cat_path, hub_dir=hub_dir, agy_models_output=FAKE_AGY_MODELS_TEXT,
        codex_cache_data={"models": []}, claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON, openrouter_data={"data": []},
    )
    first = run_model_review(**kwargs)
    assert any(p["model_id"] == "gemini-4.2-flash-medium" for p in first["proposals"])

    entered = threading.Event()
    release = threading.Event()
    real_fetch = model_review.fetch_public_metadata

    def slow_fetch(*a: Any, **k: Any) -> Any:
        entered.set()
        release.wait(10)
        return real_fetch(*a, **k)

    monkeypatch.setattr(model_review, "fetch_public_metadata", slow_fetch)
    out: Dict[str, Any] = {}
    th = threading.Thread(target=lambda: out.update(run_model_review(**kwargs)))
    th.start()
    assert entered.wait(10)
    apply_model_decision(model_id="gemini-4.2-flash-medium", decision="reject", catalog_path=cat_path, hub_dir=hub_dir)
    release.set()
    th.join(15)
    assert out

    state = model_review.load_review_state(hub_dir)
    assert "gemini-4.2-flash-medium" in state["cooldowns"]
    assert state["decisions_log"][0]["decision"] == "reject"
    assert all(p["model_id"] != "gemini-4.2-flash-medium" for p in state["last_report"]["proposals"])


def test_single_swrr_commit_when_probation_only_provider_is_denied(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    from orchestrator import get_allocation_credits

    cat_path = isolated_model_review_env["catalog_path"]
    _single_provider_probation_catalog(cat_path, "claude", 0.0)
    quotas = _make_healthy_quotas()
    orch_info = select_active_orchestrator("auto", quotas)
    learner = QuotaCostLearner(tmp_path / "sc.sqlite")
    b_learner = BurnRateLearner(tmp_path / "sc.sqlite", cost_learner=learner)
    set_preference_weights({"agy": 10, "codex": 0, "claude": 10}, reset_swrr=True)
    reset_allocation_credits()
    before = get_allocation_credits()
    res = route_work_package(
        {"task": "WP-SC", "kind": "implement", "ambiguity": "low", "blast_radius": "module",
         "dependency_role": "leaf", "context_need": "pointer"},
        quotas, orch_info, cost_learner=learner, burn_learner=b_learner,
    )
    assert res["status"] == "routed" and res["assignment_record"]["provider"] == "agy"
    after = get_allocation_credits()
    # Claude was never eligible, so it takes no part in the (single) SWRR step: its credit is untouched.
    assert after["claude"] == before["claude"]
    # Exactly one debit overall: the credits of the participants sum to zero after one committed step.
    assert abs(sum(after.values())) < 1e-6


def test_explicit_probation_honoured_when_share_exhausted_and_counted(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    from model_review import get_probation_counters

    cat_path = isolated_model_review_env["catalog_path"]
    _single_provider_probation_catalog(cat_path, "claude", 0.0)
    quotas = _make_healthy_quotas()
    orch_info = select_active_orchestrator("auto", quotas)
    learner = QuotaCostLearner(tmp_path / "ep.sqlite")
    b_learner = BurnRateLearner(tmp_path / "ep.sqlite", cost_learner=learner)
    prob_id = next(m["model_id"] for m in load_model_catalog(cat_path) if m["provider"] == "claude" and m["tier"] == "MID")
    for i in range(3):
        res = route_work_package(
            {"task": f"WP-EP{i}", "kind": "implement", "ambiguity": "low", "blast_radius": "module",
             "dependency_role": "leaf", "context_need": "pointer", "explicit_model": prob_id},
            quotas, orch_info, cost_learner=learner, burn_learner=b_learner,
        )
        assert res["status"] == "routed"
    cnt = get_probation_counters()["by_model"][prob_id]
    assert cnt["selected"] == 3 and cnt["eligible"] == 3


def test_run_settings_override_is_not_persisted_and_settings_used_recorded(
    isolated_model_review_env: Dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    hub_dir = isolated_model_review_env["hub_dir"]
    cat_path = isolated_model_review_env["catalog_path"]
    model_review.update_review_settings({"max_price_ratio": 3.0}, hub_dir=hub_dir)
    entered, release = threading.Event(), threading.Event()
    real_fetch = model_review.fetch_public_metadata

    def slow_fetch(*a: Any, **k: Any) -> Any:
        entered.set()
        release.wait(10)
        return real_fetch(*a, **k)

    monkeypatch.setattr(model_review, "fetch_public_metadata", slow_fetch)
    out: Dict[str, Any] = {}
    th = threading.Thread(target=lambda: out.update(run_model_review(
        catalog_path=cat_path, hub_dir=hub_dir, agy_models_output=FAKE_AGY_MODELS_TEXT,
        codex_cache_data={"models": []}, claude_static_models=[],
        models_dev_data=FAKE_MODELS_DEV_JSON, openrouter_data={"data": []},
        settings_override={"max_price_ratio": 2.0},
    )))
    th.start()
    assert entered.wait(10)
    model_review.update_review_settings({"max_price_ratio": 1.0}, hub_dir=hub_dir)
    release.set()
    th.join(15)
    assert out["settings_used"]["max_price_ratio"] == 2.0
    assert model_review.load_review_state(hub_dir)["settings"]["max_price_ratio"] == 1.0
    assert out["settings"]["max_price_ratio"] == 1.0


def _probation_only_env(cat_path: Path, tmp_path: Path, share: float):
    _single_provider_probation_catalog(cat_path, "claude", share)
    quotas = _make_healthy_quotas()
    orch_info = select_active_orchestrator("auto", quotas)
    learner = QuotaCostLearner(tmp_path / "gp.sqlite")
    b_learner = BurnRateLearner(tmp_path / "gp.sqlite", cost_learner=learner)
    set_preference_weights({"agy": 10, "codex": 8, "claude": 10}, reset_swrr=True)
    reset_allocation_credits()
    claude_ids = {m["model_id"] for m in load_model_catalog(cat_path) if m["provider"] == "claude"}
    return quotas, orch_info, learner, b_learner, claude_ids


def test_probation_only_provider_not_starved_at_share_010(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    quotas, orch_info, learner, b_learner, claude_ids = _probation_only_env(
        isolated_model_review_env["catalog_path"], tmp_path, 0.10
    )
    req = {"task": "WP-GP", "kind": "implement", "ambiguity": "low", "blast_radius": "module",
           "dependency_role": "leaf", "context_need": "pointer"}
    first = None
    picks = 0
    for i in range(1, 1001):
        res = route_work_package(dict(req), quotas, orch_info, cost_learner=learner, burn_learner=b_learner)
        if res["status"] == "routed" and res["assignment_record"]["model"] in claude_ids:
            picks += 1
            first = first or i
    assert first is not None and first <= 10 + 30  # first claude win happens when SWRR picks claude after a slot opens
    assert 0 < picks <= 100


def test_global_counter_gives_probation_slot_by_tenth_eligible_decision() -> None:
    from model_review import should_route_probation, tick_probation_eligible

    reset_probation_counters()
    entry = {"provider": "claude", "model_id": "m", "status": "probation", "probation_share": 0.10}
    results = []
    for _ in range(10):
        tick_probation_eligible("MID", "implement")
        results.append(should_route_probation("claude", entry, "MID", "implement", peek=True)[0])
    assert results == [False] * 9 + [True]
    assert should_route_probation("claude", entry, "MID", "implement")[0] is True
    tick_probation_eligible("MID", "implement")
    assert should_route_probation("claude", entry, "MID", "implement", peek=True)[0] is False


def test_probation_never_for_design_review_and_zero_share_never(
    isolated_model_review_env: Dict[str, Path], tmp_path: Path
) -> None:
    quotas, orch_info, learner, b_learner, claude_ids = _probation_only_env(
        isolated_model_review_env["catalog_path"], tmp_path, 0.10
    )
    for kind in ("design", "review"):
        for _ in range(40):
            res = route_work_package(
                {"task": "WP-NO", "kind": kind, "ambiguity": "low", "blast_radius": "module",
                 "dependency_role": "leaf", "context_need": "pointer"},
                quotas, orch_info, cost_learner=learner, burn_learner=b_learner,
            )
            if res["status"] == "routed":
                assert res["assignment_record"]["model"] not in claude_ids
    quotas, orch_info, learner, b_learner, claude_ids = _probation_only_env(
        isolated_model_review_env["catalog_path"], tmp_path, 0.0
    )
    for _ in range(100):
        res = route_work_package(
            {"task": "WP-Z", "kind": "implement", "ambiguity": "low", "blast_radius": "module",
             "dependency_role": "leaf", "context_need": "pointer"},
            quotas, orch_info, cost_learner=learner, burn_learner=b_learner,
        )
        if res["status"] == "routed":
            assert res["assignment_record"]["model"] not in claude_ids
