"""
WP-MODELS: Model Catalog Loader/Validator, Zero-Token Local & Public Discovery,
Price/Performance Review Scoring, Reject Cooldowns, and Probation Evaluation.

Implements `docs/WP-MODELS-CONTRACT.md`:
1. Loads `model_catalog.json` (`{"version": 1, "models": [...]}`) with strict validation
   and saves changes atomically with timestamped backups.
2. Discovers locally available models from `agy models` (30s timeout), `~/.codex/models_cache.json`,
   a static Claude list, and recent hub dispatch history.
3. Fetches & caches public metadata from `https://models.dev/api.json` and
   `https://openrouter.ai/api/v1/models` (24h cache in hub folder, ETag support, stale fallback on network error).
4. Computes `blended_price = 0.25 * in + 0.75 * out` and `price_ratio` relative to the active HIGH
   model of the same provider, proposes eligible models (`price_ratio <= max_price_ratio` or unknown price),
   lists expensive ("Astra"-class) models as `"excluded (too expensive)"`, enforces a 30-day reject cooldown,
   routes probation models within `probation_share` (default 0.10) for LOW/MID non-design/review packages,
   and summarises the 14-day probation scorecard (`promote` vs `retire`, never auto-promoting).
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
HUB_HOME = Path(os.environ.get("QUOTA_HUB_HOME") or BASE_DIR)
DEFAULT_CATALOG_PATH = BASE_DIR / "model_catalog.json"

MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
METADATA_CACHE_TTL_SEC = 24 * 3600
REJECT_COOLDOWN_DAYS = 30
PROBATION_REVIEW_DAYS = 14

VALID_PROVIDERS = {"agy", "claude", "codex"}
VALID_POOLS = {"agy_gemini", "agy_claude_gpt", "claude_shared", "codex_shared"}
TIER_TO_RANK = {"LOW": 1, "MID": 2, "HIGH": 3}
VALID_EFFORTS = {"low", "medium", "high"}
VALID_STATUSES = {"active", "probation", "retired"}

REQUIRED_MODEL_FIELDS: Tuple[str, ...] = (
    "provider",
    "model_id",
    "cli_model",
    "quota_key",
    "pool_id",
    "pool_windows",
    "family",
    "tier",
    "tier_rank",
    "effort",
    "default_preferred",
    "context_window",
    "latency_score",
    "cost_weight",
    "label",
)

OPTIONAL_MODEL_FIELDS: Tuple[str, ...] = (
    "status",
    "probation_share",
    "added_at",
    "source",
    "price_in_per_mtok",
    "price_out_per_mtok",
    "coding_score",
    "notes",
)

ALL_MODEL_FIELDS = set(REQUIRED_MODEL_FIELDS) | set(OPTIONAL_MODEL_FIELDS)

DEFAULT_STATIC_CLAUDE_MODELS: List[str] = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
]

# Fallback reference prices ($ / 1M tokens) for active HIGH models when neither the catalog
# nor the public metadata cache provides a price for the provider's active HIGH model.
DEFAULT_PROVIDER_HIGH_PRICES: Dict[str, Dict[str, float]] = {
    "agy": {"price_in_per_mtok": 1.25, "price_out_per_mtok": 5.00},
    "claude": {"price_in_per_mtok": 15.00, "price_out_per_mtok": 75.00},
    "codex": {"price_in_per_mtok": 2.50, "price_out_per_mtok": 10.00},
}

DEFAULT_REVIEW_SETTINGS: Dict[str, Any] = {
    "max_price_ratio": 2.0,
    "review_interval_days": 7,
    "probation_share": 0.10,
    "auto_retire_after_days_unused": None,
}

# Built-in default catalog (19 entries); last-resort fallback when the catalog file and backups are unusable.
BUILTIN_DEFAULT_MODELS: List[Dict[str, Any]] = [{'provider': 'agy',
  'model_id': 'gemini-4-hydrogen-low',
  'cli_model': 'gemini-4-hydrogen-low',
  'quota_key': 'gemini_flash',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'LOW',
  'tier_rank': 1,
  'effort': 'low',
  'default_preferred': True,
  'context_window': 1000000,
  'latency_score': 0.96,
  'cost_weight': 0.95,
  'label': 'AGY · Gemini 4 Hydrogen (Low)'},
 {'provider': 'agy',
  'model_id': 'gemini-3.8-flash-low',
  'cli_model': 'gemini-3.8-flash-low',
  'quota_key': 'gemini_flash',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'LOW',
  'tier_rank': 1,
  'effort': 'low',
  'default_preferred': False,
  'context_window': 1000000,
  'latency_score': 0.95,
  'cost_weight': 0.94,
  'label': 'AGY · Gemini 3.8 Flash (Low - Fallback)'},
 {'provider': 'codex',
  'model_id': 'gpt-6.1-sol-low',
  'cli_model': 'gpt-6.1-sol',
  'quota_key': 'luna',
  'pool_id': 'codex_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'openai',
  'tier': 'LOW',
  'tier_rank': 1,
  'effort': 'low',
  'default_preferred': True,
  'context_window': 475000,
  'latency_score': 0.92,
  'cost_weight': 0.88,
  'label': 'Codex · GPT-6.1-Sol (Low)'},
 {'provider': 'claude',
  'model_id': 'claude-haiku-5-5',
  'cli_model': 'claude-haiku-5-5',
  'quota_key': 'haiku',
  'pool_id': 'claude_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'claude',
  'tier': 'LOW',
  'tier_rank': 1,
  'effort': 'low',
  'default_preferred': True,
  'context_window': 200000,
  'latency_score': 0.94,
  'cost_weight': 0.9,
  'label': 'Claude Code · Haiku 5.5 (Low)'},
 {'provider': 'agy',
  'model_id': 'gemini-4-hydrogen-medium',
  'cli_model': 'gemini-4-hydrogen-medium',
  'quota_key': 'gemini_flash',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 1000000,
  'latency_score': 0.93,
  'cost_weight': 0.94,
  'label': 'AGY · Gemini 4 Hydrogen (Medium)'},
 {'provider': 'agy',
  'model_id': 'gemini-4-hydrogen-high',
  'cli_model': 'gemini-4-hydrogen-high',
  'quota_key': 'gemini_flash',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'high',
  'default_preferred': True,
  'context_window': 1000000,
  'latency_score': 0.9,
  'cost_weight': 0.92,
  'label': 'AGY · Gemini 4 Hydrogen (High)'},
 {'provider': 'agy',
  'model_id': 'gemini-3.8-flash-medium',
  'cli_model': 'gemini-3.8-flash-medium',
  'quota_key': 'gemini_flash',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'medium',
  'default_preferred': False,
  'context_window': 1000000,
  'latency_score': 0.92,
  'cost_weight': 0.93,
  'label': 'AGY · Gemini 3.8 Flash (Medium - Fallback)'},
 {'provider': 'agy',
  'model_id': 'gemini-3.8-flash-high',
  'cli_model': 'gemini-3.8-flash-high',
  'quota_key': 'gemini_flash',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'high',
  'default_preferred': False,
  'context_window': 1000000,
  'latency_score': 0.89,
  'cost_weight': 0.91,
  'label': 'AGY · Gemini 3.8 Flash (High - Fallback)'},
 {'provider': 'claude',
  'model_id': 'claude-sonnet-5-5',
  'cli_model': 'claude-sonnet-5-5',
  'quota_key': 'sonnet',
  'pool_id': 'claude_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'claude',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 200000,
  'latency_score': 0.86,
  'cost_weight': 0.86,
  'label': 'Claude Code · Sonnet 5.5'},
 {'provider': 'agy',
  'model_id': 'claude-sonnet-5-5-medium',
  'cli_model': 'claude-sonnet-5-5-medium',
  'quota_key': 'claude_sonnet',
  'pool_id': 'agy_claude_gpt',
  'pool_windows': ('claude_gpt_5h', 'claude_gpt_weekly'),
  'family': 'claude',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 200000,
  'latency_score': 0.85,
  'cost_weight': 0.85,
  'label': 'AGY · Claude Sonnet 5.5 (Medium)'},
 {'provider': 'codex',
  'model_id': 'gpt-6.1-sol-medium',
  'cli_model': 'gpt-6.1-sol',
  'quota_key': 'terra',
  'pool_id': 'codex_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'openai',
  'tier': 'MID',
  'tier_rank': 2,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 475000,
  'latency_score': 0.84,
  'cost_weight': 0.82,
  'label': 'Codex · GPT-6.1-Sol (Medium)'},
 {'provider': 'agy',
  'model_id': 'gemini-4-argon-medium',
  'cli_model': 'gemini-4-argon-medium',
  'quota_key': 'gemini_pro',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 1000000,
  'latency_score': 0.82,
  'cost_weight': 0.9,
  'label': 'AGY · Gemini 4 Argon (Medium)'},
 {'provider': 'agy',
  'model_id': 'gemini-4-argon-low',
  'cli_model': 'gemini-4-argon-low',
  'quota_key': 'gemini_pro',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'low',
  'default_preferred': True,
  'context_window': 1000000,
  'latency_score': 0.86,
  'cost_weight': 0.92,
  'label': 'AGY · Gemini 4 Argon (Low)'},
 {'provider': 'claude',
  'model_id': 'claude-opus-5-5',
  'cli_model': 'claude-opus-5-5',
  'quota_key': 'opus',
  'pool_id': 'claude_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'claude',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 1000000,
  'latency_score': 0.78,
  'cost_weight': 0.8,
  'label': 'Claude Code · Opus 5.5'},
 {'provider': 'codex',
  'model_id': 'gpt-6.1-sol',
  'cli_model': 'gpt-6.1-sol',
  'quota_key': 'sol',
  'pool_id': 'codex_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'openai',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 475000,
  'latency_score': 0.8,
  'cost_weight': 0.8,
  'label': 'Codex · GPT-6.1-Sol'},
 {'provider': 'agy',
  'model_id': 'claude-opus-5-5-medium',
  'cli_model': 'claude-opus-5-5-medium',
  'quota_key': 'claude_opus',
  'pool_id': 'agy_claude_gpt',
  'pool_windows': ('claude_gpt_5h', 'claude_gpt_weekly'),
  'family': 'claude',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'medium',
  'default_preferred': True,
  'context_window': 200000,
  'latency_score': 0.76,
  'cost_weight': 0.84,
  'label': 'AGY · Claude Opus 5.5 (Medium)'},
 {'provider': 'agy',
  'model_id': 'gemini-4-argon-high',
  'cli_model': 'gemini-4-argon-high',
  'quota_key': 'gemini_pro',
  'pool_id': 'agy_gemini',
  'pool_windows': ('gemini_5h', 'gemini_weekly'),
  'family': 'gemini',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'high',
  'default_preferred': False,
  'context_window': 1000000,
  'latency_score': 0.74,
  'cost_weight': 0.82,
  'label': 'AGY · Gemini 4 Argon (High - Exception Only)'},
 {'provider': 'agy',
  'model_id': 'claude-opus-5-5-high',
  'cli_model': 'claude-opus-5-5-high',
  'quota_key': 'claude_opus',
  'pool_id': 'agy_claude_gpt',
  'pool_windows': ('claude_gpt_5h', 'claude_gpt_weekly'),
  'family': 'claude',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'high',
  'default_preferred': False,
  'context_window': 200000,
  'latency_score': 0.72,
  'cost_weight': 0.78,
  'label': 'AGY · Claude Opus 5.5 (High)'},
 {'provider': 'codex',
  'model_id': 'gpt-6.1-sol-high',
  'cli_model': 'gpt-6.1-sol',
  'quota_key': 'sol',
  'pool_id': 'codex_shared',
  'pool_windows': ('five_hour', 'weekly'),
  'family': 'openai',
  'tier': 'HIGH',
  'tier_rank': 3,
  'effort': 'high',
  'default_preferred': False,
  'context_window': 475000,
  'latency_score': 0.75,
  'cost_weight': 0.76,
  'label': 'Codex · GPT-6.1-Sol (High)'}]

_CATALOG_LOCK = threading.Lock()
CATALOG_STATUS: Dict[str, Any] = {"status": "ok", "reason": ""}
_CATALOG_BACKUP_SEQ: int = 0
_CATALOG_RELOAD_HOOKS: List[Callable[[List[Dict[str, Any]]], None]] = []

_PROBATION_LOCK = threading.Lock()
_PROBATION_COUNTERS_BY_PROVIDER: Dict[str, Dict[str, int]] = {}
_PROBATION_COUNTERS_BY_MODEL: Dict[str, Dict[str, int]] = {}
# Probation-eligible routing decisions (floor LOW/MID, kind not design/review), counted once per decision.
_PROBATION_GLOBAL: Dict[str, int] = {"eligible": 0}

_REVIEW_STATE_LOCK = threading.Lock()
SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,120}$")
# Serialises read-modify-write transactions on the review state (owner decisions vs. review completion).
_REVIEW_TXN_LOCK = threading.RLock()


def resolve_catalog_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env_p = os.environ.get("MODEL_CATALOG_PATH")
    if env_p:
        return Path(env_p)
    return DEFAULT_CATALOG_PATH


def resolve_hub_dir(hub_dir: Optional[Path] = None) -> Path:
    if hub_dir is not None:
        return Path(hub_dir)
    env_h = os.environ.get("QUOTA_HUB_HOME")
    if env_h:
        return Path(env_h)
    return HUB_HOME


def register_catalog_reload_hook(hook: Callable[[List[Dict[str, Any]]], None]) -> None:
    if hook not in _CATALOG_RELOAD_HOOKS:
        _CATALOG_RELOAD_HOOKS.append(hook)


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and not math.isnan(float(val)) and not math.isinf(float(val))


def validate_model_entry(entry: Any, index: int = 0) -> Dict[str, Any]:
    """
    Validate a single model entry against required and optional fields in `docs/WP-MODELS-CONTRACT.md` §1.
    Normalises `pool_windows` to a 2-tuple of strings in memory so `MODEL_CATALOG` matches the legacy schema.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Model entry at index {index} must be a JSON object")

    missing = [f for f in REQUIRED_MODEL_FIELDS if f not in entry]
    if missing:
        raise ValueError(f"Model entry at index {index} is missing required fields: {missing}")

    unknown_fields = set(entry.keys()) - ALL_MODEL_FIELDS
    if unknown_fields:
        raise ValueError(f"Model entry at index {index} has unsupported fields: {sorted(unknown_fields)}")

    provider = entry.get("provider")
    if not isinstance(provider, str) or provider.strip().lower() not in VALID_PROVIDERS:
        raise ValueError(f"Model entry at index {index}: provider must be one of {sorted(VALID_PROVIDERS)}")
    provider = provider.strip().lower()

    model_id = entry.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"Model entry at index {index}: model_id must be a non-empty string")
    model_id = model_id.strip()

    for str_field in ("cli_model", "quota_key", "family", "label"):
        val = entry.get(str_field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Model entry '{model_id}': {str_field} must be a non-empty string")

    pool_id = entry.get("pool_id")
    if not isinstance(pool_id, str) or pool_id.strip() not in VALID_POOLS:
        raise ValueError(f"Model entry '{model_id}': pool_id must be one of {sorted(VALID_POOLS)}")

    pool_windows = entry.get("pool_windows")
    if not isinstance(pool_windows, (list, tuple)) or len(pool_windows) != 2 or not all(
        isinstance(w, str) and w.strip() for w in pool_windows
    ):
        raise ValueError(f"Model entry '{model_id}': pool_windows must be a 2-element list/tuple of non-empty strings")

    tier = entry.get("tier")
    if not isinstance(tier, str) or tier.strip().upper() not in TIER_TO_RANK:
        raise ValueError(f"Model entry '{model_id}': tier must be LOW, MID, or HIGH")
    tier_norm = tier.strip().upper()

    tier_rank = entry.get("tier_rank")
    if isinstance(tier_rank, bool) or not isinstance(tier_rank, int) or tier_rank != TIER_TO_RANK[tier_norm]:
        raise ValueError(
            f"Model entry '{model_id}': tier_rank must be {TIER_TO_RANK[tier_norm]} for tier {tier_norm}"
        )

    effort = entry.get("effort")
    if not isinstance(effort, str) or effort.strip().lower() not in VALID_EFFORTS:
        raise ValueError(f"Model entry '{model_id}': effort must be one of {sorted(VALID_EFFORTS)}")

    default_preferred = entry.get("default_preferred")
    if not isinstance(default_preferred, bool):
        raise ValueError(f"Model entry '{model_id}': default_preferred must be a boolean")

    context_window = entry.get("context_window")
    if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window <= 0:
        raise ValueError(f"Model entry '{model_id}': context_window must be a positive integer")

    for score_field in ("latency_score", "cost_weight"):
        s_val = entry.get(score_field)
        if not _is_number(s_val) or float(s_val) <= 0.0 or float(s_val) > 1.0:
            raise ValueError(f"Model entry '{model_id}': {score_field} must be a number in (0.0, 1.0]")

    cleaned: Dict[str, Any] = {
        "provider": provider,
        "model_id": model_id,
        "cli_model": entry["cli_model"].strip(),
        "quota_key": entry["quota_key"].strip(),
        "pool_id": pool_id.strip(),
        "pool_windows": (str(pool_windows[0]).strip(), str(pool_windows[1]).strip()),
        "family": entry["family"].strip(),
        "tier": tier_norm,
        "tier_rank": int(tier_rank),
        "effort": effort.strip().lower(),
        "default_preferred": bool(default_preferred),
        "context_window": int(context_window),
        "latency_score": float(entry["latency_score"]),
        "cost_weight": float(entry["cost_weight"]),
        "label": entry["label"].strip(),
    }

    # Validate optional fields when present
    if "status" in entry and entry["status"] is not None:
        st = entry["status"]
        if not isinstance(st, str) or st.strip().lower() not in VALID_STATUSES:
            raise ValueError(f"Model entry '{model_id}': status must be one of {sorted(VALID_STATUSES)}")
        cleaned["status"] = st.strip().lower()

    if "probation_share" in entry and entry["probation_share"] is not None:
        ps = entry["probation_share"]
        if not _is_number(ps) or float(ps) < 0.0 or float(ps) > 1.0:
            raise ValueError(f"Model entry '{model_id}': probation_share must be a number in [0.0, 1.0]")
        cleaned["probation_share"] = float(ps)

    for opt_str in ("added_at", "source", "notes"):
        if opt_str in entry and entry[opt_str] is not None:
            if not isinstance(entry[opt_str], str):
                raise ValueError(f"Model entry '{model_id}': {opt_str} must be a string")
            cleaned[opt_str] = entry[opt_str]

    for opt_num in ("price_in_per_mtok", "price_out_per_mtok", "coding_score"):
        if opt_num in entry and entry[opt_num] is not None:
            if not _is_number(entry[opt_num]) or float(entry[opt_num]) < 0.0:
                raise ValueError(f"Model entry '{model_id}': {opt_num} must be a non-negative number")
            cleaned[opt_num] = float(entry[opt_num])

    return cleaned


def validate_model_catalog_doc(doc: Any) -> List[Dict[str, Any]]:
    """Validate a `model_catalog.json` document (`{"version": 1, "models": [...]}`)."""
    if not isinstance(doc, dict):
        raise ValueError("Model catalog document must be a JSON object")
    version = doc.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Model catalog 'version' must be a positive integer")
    models_raw = doc.get("models")
    if not isinstance(models_raw, list) or len(models_raw) == 0:
        raise ValueError("Model catalog 'models' must be a non-empty list")

    validated: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(models_raw):
        cleaned = validate_model_entry(item, index=idx)
        mid = cleaned["model_id"]
        if mid in seen_ids:
            raise ValueError(f"Duplicate model_id in catalog: '{mid}'")
        seen_ids.add(mid)
        validated.append(cleaned)
    return validated


def _serialize_model_for_json(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(entry)
    if isinstance(out.get("pool_windows"), tuple):
        out["pool_windows"] = list(out["pool_windows"])
    return out


def load_model_catalog(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and validate `model_catalog.json` from `path` (or `MODEL_CATALOG_PATH` / `DEFAULT_CATALOG_PATH`)."""
    cat_path = resolve_catalog_path(path)
    if not cat_path.exists():
        raise ValueError(f"Model catalog file not found: {cat_path}")
    try:
        raw_text = cat_path.read_text(encoding="utf-8-sig")
        doc = json.loads(raw_text)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to read/parse model catalog at {cat_path}: {exc}") from exc
    return validate_model_catalog_doc(doc)


def get_catalog_status() -> Dict[str, Any]:
    return dict(CATALOG_STATUS)


def load_model_catalog_resilient(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Never raises. Load the catalog; on failure use the newest valid timestamped backup, then the built-in
    defaults. Updates `CATALOG_STATUS` ("ok" | "restored_from_backup" | "builtin_defaults" + reason).
    """
    cat_path = resolve_catalog_path(path)
    try:
        models = load_model_catalog(cat_path)
        CATALOG_STATUS.update(status="ok", reason="")
        return models
    except Exception as exc:  # noqa: BLE001
        primary_err = str(exc)
    try:
        backups = sorted(cat_path.parent.glob(f"{cat_path.name}.bak-*"), key=lambda p: p.name, reverse=True)
    except OSError:
        backups = []
    for bak in backups:
        try:
            models = validate_model_catalog_doc(json.loads(bak.read_text(encoding="utf-8-sig")))
        except Exception:  # noqa: BLE001
            continue
        reason = f"{primary_err}; restored from backup {bak.name}"
        CATALOG_STATUS.update(status="restored_from_backup", reason=reason)
        print(f"WARNING: model catalog unusable ({reason})", flush=True)
        return models
    reason = f"{primary_err}; no valid backup, using built-in defaults"
    CATALOG_STATUS.update(status="builtin_defaults", reason=reason)
    print(f"WARNING: model catalog unusable ({reason})", flush=True)
    return validate_model_catalog_doc({"version": 1, "models": copy.deepcopy(BUILTIN_DEFAULT_MODELS)})


def save_model_catalog(
    models: List[Dict[str, Any]],
    path: Optional[Path] = None,
    max_backups: int = 10,
    version: int = 1,
) -> Dict[str, Any]:
    """
    Validate and atomically write `model_catalog.json` with a timestamped backup (`<name>.bak-<ts>-<seq>`).
    Notifies registered reload hooks so in-process `MODEL_CATALOG` references update immediately.
    """
    global _CATALOG_BACKUP_SEQ
    validated = validate_model_catalog_doc({"version": version, "models": models})
    cat_path = resolve_catalog_path(path)
    doc_to_write = {
        "version": int(version),
        "models": [_serialize_model_for_json(m) for m in validated],
    }
    payload_text = json.dumps(doc_to_write, indent=2, ensure_ascii=False) + "\n"

    with _CATALOG_LOCK:
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        _CATALOG_BACKUP_SEQ += 1
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = cat_path.parent / f"{cat_path.name}.bak-{ts}-{_CATALOG_BACKUP_SEQ:04d}"
        if cat_path.exists():
            shutil.copy2(cat_path, backup_path)
        else:
            backup_path.write_text(payload_text, encoding="utf-8")

        backups = sorted(
            cat_path.parent.glob(f"{cat_path.name}.bak-*"),
            key=lambda p: p.name,
        )
        while len(backups) > max(1, int(max_backups)):
            oldest = backups.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass

        tmp_path = cat_path.parent / f".{cat_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            tmp_path.write_text(payload_text, encoding="utf-8")
            os.replace(tmp_path, cat_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        for hook in list(_CATALOG_RELOAD_HOOKS):
            try:
                hook(copy.deepcopy(validated))
            except Exception:
                pass

    reset_probation_counters()
    return {
        "catalog_path": str(cat_path),
        "backup_path": str(backup_path),
        "backups_kept": len(backups),
        "models": validated,
    }


# ---------------------------------------------------------------------------
# Probation Routing Gate & Deterministic Counter
# ---------------------------------------------------------------------------


def is_probation_eligible(floor_tier: str, kind: str) -> bool:
    """
    Probation models are eligible ONLY for packages with floor LOW/MID and kind not in {design, review}.
    """
    f_norm = str(floor_tier or "LOW").strip().upper()
    k_norm = str(kind or "implement").strip().lower()
    return f_norm in {"LOW", "MID"} and k_norm not in {"design", "review"}


def reset_probation_counters() -> None:
    """Reset the deterministic probation routing counters."""
    with _PROBATION_LOCK:
        _PROBATION_COUNTERS_BY_PROVIDER.clear()
        _PROBATION_COUNTERS_BY_MODEL.clear()
        _PROBATION_GLOBAL["eligible"] = 0


def tick_probation_eligible(floor_tier: str, kind: str) -> int:
    """Advance the global probation-eligible counter once for a routing decision (if the package is eligible)."""
    with _PROBATION_LOCK:
        if is_probation_eligible(floor_tier, kind):
            _PROBATION_GLOBAL["eligible"] += 1
        return _PROBATION_GLOBAL["eligible"]


def get_probation_counters() -> Dict[str, Any]:
    with _PROBATION_LOCK:
        return {
            "by_provider": copy.deepcopy(_PROBATION_COUNTERS_BY_PROVIDER),
            "by_model": copy.deepcopy(_PROBATION_COUNTERS_BY_MODEL),
        }


def should_route_probation(
    provider: str,
    model_entry: Dict[str, Any],
    floor_tier: str,
    kind: str,
    default_probation_share: Optional[float] = None,
    peek: bool = False,
    force: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    `peek=True` is read-only (reports whether the next call would be granted; counters unchanged).
    `force=True` records an explicit owner request: always granted and counted, share ignored.

    Deterministic counter enforcing that a `probation` model on `provider` is selected for at most
    `probation_share` (default 0.10) of eligible routing decisions after SWRR provider selection.
    Never selects a probation model when `floor_tier == 'HIGH'` or `kind in {'design', 'review'}`.
    """
    status = str(model_entry.get("status") or "active").strip().lower()
    if status != "probation" or (not force and not is_probation_eligible(floor_tier, kind)):
        return False, {"eligible": 0, "selected": 0, "share": 0.0, "reason": "ineligible"}

    raw_share = model_entry.get("probation_share")
    if raw_share is None:
        raw_share = default_probation_share if default_probation_share is not None else 0.10
    try:
        share = max(0.0, min(1.0, float(raw_share)))
    except (TypeError, ValueError):
        share = 0.10

    if share <= 0.0 and not force:
        return False, {"eligible": 0, "selected": 0, "share": 0.0, "reason": "zero_share"}

    prov_key = str(provider or model_entry.get("provider") or "").strip().lower()
    model_id = str(model_entry.get("model_id") or "").strip()

    with _PROBATION_LOCK:
        g = _PROBATION_GLOBAL["eligible"]
        allowance = g * share + 1e-9  # includes the current decision (already ticked by the router)
        p_cnt = _PROBATION_COUNTERS_BY_PROVIDER.get(prov_key) or {"eligible": 0, "selected": 0}
        m_cnt = _PROBATION_COUNTERS_BY_MODEL.get(model_id) or {"eligible": 0, "selected": 0}
        can_select = (p_cnt["selected"] + 1) <= allowance and (m_cnt["selected"] + 1) <= allowance
        if peek:
            return can_select, {"peek": True, "share": share}
        p_cnt = _PROBATION_COUNTERS_BY_PROVIDER.setdefault(prov_key, {"eligible": 0, "selected": 0})
        m_cnt = _PROBATION_COUNTERS_BY_MODEL.setdefault(model_id, {"eligible": 0, "selected": 0})
        p_cnt["eligible"] = m_cnt["eligible"] = g
        if force:
            p_cnt["selected"] += 1
            m_cnt["selected"] += 1
            return True, {
                "eligible": g,
                "selected": m_cnt["selected"],
                "share": share,
                "reason": "explicit request overrides probation share",
            }
        if can_select:
            p_cnt["selected"] += 1
            m_cnt["selected"] += 1
        return can_select, {
            "eligible": g,
            "selected": m_cnt["selected"],
            "share": share,
            "provider_eligible": g,
            "provider_selected": p_cnt["selected"],
        }


# ---------------------------------------------------------------------------
# Owner Settings & Review State Persistence
# ---------------------------------------------------------------------------


def _review_state_path(hub_dir: Optional[Path] = None) -> Path:
    return resolve_hub_dir(hub_dir) / "model_review_state.json"


def _metadata_cache_path(hub_dir: Optional[Path] = None) -> Path:
    return resolve_hub_dir(hub_dir) / "model_metadata_cache.json"


def validate_review_settings(raw: Any, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate owner review settings:
    - `max_price_ratio`: float > 0 (default 2.0)
    - `review_interval_days`: int/float >= 1 (default 7)
    - `probation_share`: float in [0.0, 1.0] (default 0.10)
    - `auto_retire_after_days_unused`: None / False / "off" or int >= 1 (default None / off)
    """
    if not isinstance(raw, dict):
        raise ValueError("Review settings must be a JSON object")
    current = copy.deepcopy(base if isinstance(base, dict) else DEFAULT_REVIEW_SETTINGS)

    if "max_price_ratio" in raw and raw["max_price_ratio"] is not None:
        val = raw["max_price_ratio"]
        if not _is_number(val) or float(val) <= 0.0 or float(val) > 100.0:
            raise ValueError("max_price_ratio must be a number in (0, 100]")
        current["max_price_ratio"] = round(float(val), 4)

    if "review_interval_days" in raw and raw["review_interval_days"] is not None:
        val = raw["review_interval_days"]
        if not _is_number(val) or float(val) <= 0.0 or float(val) > 365.0:
            raise ValueError("review_interval_days must be a number in (0, 365]")
        current["review_interval_days"] = int(val) if float(val).is_integer() else round(float(val), 2)

    if "probation_share" in raw and raw["probation_share"] is not None:
        val = raw["probation_share"]
        if not _is_number(val) or float(val) < 0.0 or float(val) > 1.0:
            raise ValueError("probation_share must be a number in [0.0, 1.0]")
        current["probation_share"] = round(float(val), 4)

    if "auto_retire_after_days_unused" in raw:
        val = raw["auto_retire_after_days_unused"]
        if val is None or val is False or (isinstance(val, str) and val.strip().lower() in {"", "off", "none", "false", "0"}):
            current["auto_retire_after_days_unused"] = None
        elif isinstance(val, int) and not isinstance(val, bool) and val == 0:
            current["auto_retire_after_days_unused"] = None
        elif _is_number(val) and int(val) >= 1:
            current["auto_retire_after_days_unused"] = int(val)
        elif isinstance(val, str) and val.strip().isdigit() and int(val.strip()) >= 1:
            current["auto_retire_after_days_unused"] = int(val.strip())
        else:
            raise ValueError("auto_retire_after_days_unused must be 'off'/null or a positive integer")

    return current


def load_review_state(hub_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = _review_state_path(hub_dir)
    default_state: Dict[str, Any] = {
        "settings": copy.deepcopy(DEFAULT_REVIEW_SETTINGS),
        "cooldowns": {},
        "last_run": None,
        "last_report": None,
        "decisions_log": [],
    }
    if not path.exists():
        return default_state
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_state
        settings = validate_review_settings(raw.get("settings") or {}, base=DEFAULT_REVIEW_SETTINGS)
        cooldowns = raw.get("cooldowns") if isinstance(raw.get("cooldowns"), dict) else {}
        return {
            "settings": settings,
            "cooldowns": cooldowns,
            "last_run": raw.get("last_run"),
            "last_report": raw.get("last_report") if isinstance(raw.get("last_report"), dict) else None,
            "decisions_log": raw.get("decisions_log") if isinstance(raw.get("decisions_log"), list) else [],
        }
    except Exception:
        return default_state


def save_review_state(state: Dict[str, Any], hub_dir: Optional[Path] = None) -> None:
    path = _review_state_path(hub_dir)
    with _REVIEW_STATE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def get_review_settings(hub_dir: Optional[Path] = None) -> Dict[str, Any]:
    return load_review_state(hub_dir).get("settings") or copy.deepcopy(DEFAULT_REVIEW_SETTINGS)


def update_review_settings(new_settings: Dict[str, Any], hub_dir: Optional[Path] = None) -> Dict[str, Any]:
    with _REVIEW_TXN_LOCK:
        state = load_review_state(hub_dir)
        validated = validate_review_settings(new_settings, base=state.get("settings"))
        state["settings"] = validated
        save_review_state(state, hub_dir=hub_dir)
    return validated


# ---------------------------------------------------------------------------
# Section 2: Local Discovery & Public Metadata Fetching
# ---------------------------------------------------------------------------

_BRACE_PATTERN = re.compile(r"^([a-z0-9._-]+)-\{([a-z0-9,_-]+)\}$", re.IGNORECASE)
_MODEL_TOKEN_RE = re.compile(
    r"\b(?:gemini|claude|gpt|o[134]|deepseek|qwen|llama|mistral|codestral)-[a-z0-9._-]+\b",
    re.IGNORECASE,
)
_GENERIC_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$", re.IGNORECASE)
_STOPWORDS = {
    "available",
    "models",
    "model",
    "default",
    "provider",
    "description",
    "tier",
    "effort",
    "status",
    "active",
    "version",
    "none",
    "null",
    "true",
    "false",
}


def _expand_brace_token(token: str) -> List[str]:
    m = _BRACE_PATTERN.match(token.strip())
    if not m:
        return [token.strip()]
    prefix = m.group(1)
    suffixes = [s.strip() for s in m.group(2).split(",") if s.strip()]
    return [f"{prefix}-{s}" for s in suffixes]


def parse_agy_models_output(raw_text: str) -> List[str]:
    """
    Parse `agy models` output into a deduplicated list of exact model IDs.
    Real `agy models` output is tab-separated (`<model_id>\\t<label>`) with a
    `Fetching available models...` status line; the full first column is kept
    verbatim as the model ID (including `-low`/`-medium`/`-high` suffixes).
    Also supports single-column lists, brace patterns (`gemini-4-argon-{low,medium,high}`),
    and JSON arrays/objects.
    """
    if not raw_text or not isinstance(raw_text, str):
        return []
    text = raw_text.strip()
    if not text:
        return []

    found: List[str] = []
    seen: set[str] = set()

    def _add(mid: str) -> None:
        # Strip surrounding quotes/punctuation EXCEPT braces `{}` so brace expansion works if present
        m_raw = mid.strip().strip("`'\".,:;()[]").lower()
        if not m_raw or m_raw in _STOPWORDS or "-" not in m_raw:
            return
        for expanded in _expand_brace_token(m_raw):
            cand = expanded.strip().strip("`'\".,:;()[]{}")
            if cand and cand not in _STOPWORDS and cand not in seen and _GENERIC_SLUG_RE.match(cand):
                seen.add(cand)
                found.append(cand)

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else (parsed.get("models") or parsed.get("data") or [])
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, str):
                        _add(it)
                    elif isinstance(it, dict):
                        _add(str(it.get("id") or it.get("model_id") or it.get("name") or ""))
                if found:
                    return found
        except ValueError:
            pass

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or set(s) <= {"-", "=", "|", "+", " "}:
            continue
        s_lower = s.lower()
        if s_lower.startswith(("fetching ", "available ", "loading ")) or s.endswith((":", "...")):
            continue

        # Real `agy models` output is tab-separated: `<model_id>\t<label>`
        if "\t" in line:
            first_col = line.split("\t", 1)[0].strip()
            _add(first_col)
            continue

        # Strip optional leading list markers (`- `, `* `, `• `, `> `, `| `)
        cleaned_line = re.sub(r"^[-*•>|\s]+", "", s)
        if not cleaned_line:
            continue

        # Take the first whitespace-separated column as the model id (or expand if brace list)
        first_col = cleaned_line.split()[0].strip("`'\".,:;()[]")
        if _BRACE_PATTERN.match(first_col) or ("-" in first_col and _GENERIC_SLUG_RE.match(first_col)):
            _add(first_col)

    return found


def discover_agy_models(
    agy_models_output: Optional[str] = None,
    timeout_sec: int = 30,
    runner: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Run or parse `agy models` (with a 30s timeout) to discover local AGY models."""
    if agy_models_output is not None:
        models = parse_agy_models_output(agy_models_output)
        return {"source": "agy_models_injected", "ok": True, "models": models, "error": None}

    if runner is not None:
        try:
            out = runner()
            models = parse_agy_models_output(out)
            return {"source": "agy_models_runner", "ok": True, "models": models, "error": None}
        except Exception as exc:
            return {"source": "agy_models_runner", "ok": False, "models": [], "error": str(exc)}

    agy_bin = shutil.which("agy") or "agy"
    try:
        proc = subprocess.run(
            [agy_bin, "models"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        models = parse_agy_models_output(combined)
        return {
            "source": "agy_models_cli",
            "ok": proc.returncode == 0 or bool(models),
            "models": models,
            "error": None if (proc.returncode == 0 or models) else f"agy models exited with {proc.returncode}",
        }
    except Exception as exc:
        return {"source": "agy_models_cli", "ok": False, "models": [], "error": str(exc)}


def discover_codex_models(
    cache_path: Optional[Path] = None,
    cache_data: Optional[Any] = None,
) -> Dict[str, Any]:
    """Parse `~/.codex/models_cache.json` to discover locally available Codex models."""
    raw = cache_data
    source_label = "codex_models_cache_injected"
    if raw is None:
        c_path = (
            Path(cache_path)
            if cache_path is not None
            else Path(os.environ.get("CODEX_MODELS_CACHE_PATH") or (Path.home() / ".codex" / "models_cache.json"))
        )
        source_label = str(c_path)
        if not c_path.exists():
            return {"source": source_label, "ok": False, "models": [], "details": {}, "error": "file_not_found"}
        try:
            raw = json.loads(c_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return {"source": source_label, "ok": False, "models": [], "details": {}, "error": str(exc)}

    items: List[Any] = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("models", "data", "items", "available_models"):
            if isinstance(raw.get(key), list):
                items = raw[key]
                break
        if not items and all(isinstance(v, dict) for v in raw.values()):
            items = [{"id": k, **v} for k, v in raw.items()]

    models: List[str] = []
    details: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    for it in items:
        if isinstance(it, str):
            mid = it.strip()
            if mid and mid not in seen:
                seen.add(mid)
                models.append(mid)
        elif isinstance(it, dict):
            mid = str(it.get("slug") or it.get("id") or it.get("model_id") or it.get("model") or it.get("name") or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            models.append(mid)
            ctx = it.get("context_window") or it.get("context_length") or it.get("max_input_tokens")
            details[mid] = {
                "context_window": int(ctx) if isinstance(ctx, (int, float)) and int(ctx) > 0 else None,
                "description": it.get("description"),
            }
    return {"source": source_label, "ok": True, "models": models, "details": details, "error": None}


def discover_claude_models(static_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return the static Claude Code model list (Claude Code CLI has no list command)."""
    m_list = list(static_models) if static_models is not None else list(DEFAULT_STATIC_CLAUDE_MODELS)
    cleaned = [str(m).strip() for m in m_list if str(m).strip()]
    return {"source": "claude_static_list", "ok": True, "models": cleaned, "error": None}


def discover_history_models(
    dispatch_history: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Extract `(provider, model_id)` pairs seen in recent hub dispatch history and SQLite telemetry."""
    discovered: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    def _record(prov: Any, mid: Any, src: str) -> None:
        p = str(prov or "").strip().lower()
        m = str(mid or "").strip()
        if p in VALID_PROVIDERS and m and (p, m) not in seen:
            seen.add((p, m))
            discovered.append({"provider": p, "model_id": m, "source": src})

    for entry in dispatch_history or []:
        if not isinstance(entry, dict):
            continue
        _record(entry.get("provider"), entry.get("model_id") or entry.get("model"), "dispatch_history")
        routing = entry.get("routing")
        if isinstance(routing, dict):
            ar = routing.get("assignment_record")
            if isinstance(ar, dict):
                _record(ar.get("provider"), ar.get("model") or ar.get("model_id"), "dispatch_history")
        execution = entry.get("execution")
        if isinstance(execution, dict):
            _record(execution.get("provider"), execution.get("model") or execution.get("model_id"), "dispatch_history")

    if db_path is not None and Path(db_path).exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT provider, model_id FROM task_cost_observations WHERE provider IS NOT NULL AND model_id IS NOT NULL LIMIT 100"
                ).fetchall()
                for prov, mid in rows:
                    _record(prov, mid, "telemetry_db")
        except Exception:
            pass

    return discovered


def discover_local_models(
    agy_models_output: Optional[str] = None,
    agy_runner: Optional[Callable[[], str]] = None,
    codex_cache_path: Optional[Path] = None,
    codex_cache_data: Optional[Any] = None,
    claude_static_models: Optional[List[str]] = None,
    dispatch_history: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[Path] = None,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    """
    Aggregate locally available models across `agy models`, `~/.codex/models_cache.json`,
    the static Claude list, and recent hub dispatch history.
    """
    agy_res = discover_agy_models(agy_models_output=agy_models_output, timeout_sec=timeout_sec, runner=agy_runner)
    codex_res = discover_codex_models(cache_path=codex_cache_path, cache_data=codex_cache_data)
    claude_res = discover_claude_models(static_models=claude_static_models)
    hist_res = discover_history_models(dispatch_history=dispatch_history, db_path=db_path)

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    rejected_ids: List[Dict[str, str]] = []

    def _merge(provider: str, model_id: str, source: str, local_ctx: Optional[int] = None) -> None:
        if not isinstance(model_id, str) or not SAFE_MODEL_ID_RE.match(model_id):
            reason = "model id does not match ^[A-Za-z0-9._:/-]{1,120}$"
            print(f"WARNING: dropping discovered model id {str(model_id)[:60]!r} ({provider}/{source}): {reason}", flush=True)
            rejected_ids.append({"provider": str(provider), "model_id": str(model_id)[:120], "source": source, "reason": reason})
            return
        key = (provider, model_id)
        item = by_key.get(key)
        if item is None:
            item = {
                "provider": provider,
                "model_id": model_id,
                "local_sources": [source],
                "local_context_window": local_ctx,
            }
            by_key[key] = item
        else:
            if source not in item["local_sources"]:
                item["local_sources"].append(source)
            if item["local_context_window"] is None and local_ctx is not None:
                item["local_context_window"] = local_ctx

    for mid in agy_res.get("models") or []:
        _merge("agy", mid, "agy_models")
    codex_details = codex_res.get("details") or {}
    for mid in codex_res.get("models") or []:
        ctx = (codex_details.get(mid) or {}).get("context_window")
        _merge("codex", mid, "codex_models_cache", local_ctx=ctx)
    for mid in claude_res.get("models") or []:
        _merge("claude", mid, "claude_static")
    for h in hist_res:
        _merge(h["provider"], h["model_id"], h["source"])

    return {
        "models": list(by_key.values()),
        "sources": {
            "agy": {"ok": agy_res["ok"], "count": len(agy_res.get("models") or []), "error": agy_res.get("error")},
            "codex": {"ok": codex_res["ok"], "count": len(codex_res.get("models") or []), "error": codex_res.get("error")},
            "claude": {"ok": claude_res["ok"], "count": len(claude_res.get("models") or []), "error": claude_res.get("error")},
            "history": {"ok": True, "count": len(hist_res)},
        },
        "rejected_ids": rejected_ids,
    }


# ---------------------------------------------------------------------------
# Normalised Name Matching & Public Metadata Parsing (models.dev + OpenRouter)
# ---------------------------------------------------------------------------


def normalise_model_name(name: str, strip_effort: bool = False) -> str:
    """
    Normalise a model identifier for matching between local CLI names and public catalogs
    (`models.dev`, `OpenRouter`):
    - Strips provider prefix (`google/`, `anthropic/`, `openai/`, etc.) and `:tag` suffix.
    - Lowercases, converts `.` and `_` to `-`, collapses duplicate `-`.
    - Optionally strips trailing effort suffix `-(low|medium|high)` and 8-digit date stamp `-2026xxxx`.
    """
    s = str(name or "").strip().lower()
    if "/" in s:
        s = s.split("/")[-1]
    if ":" in s:
        s = s.split(":")[0]
    s = re.sub(r"[._\s]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    s = re.sub(r"-20\d{6}$", "", s)
    if strip_effort:
        s = re.sub(r"-(low|medium|high|xhigh)$", "", s)
    return s


def _token_signature(name: str) -> str:
    """Order-independent token signature for handling `claude-5-5-sonnet` <-> `claude-sonnet-5-5`."""
    norm = normalise_model_name(name, strip_effort=True)
    tokens = sorted(t for t in norm.split("-") if t)
    return "-".join(tokens)


def _coerce_price_per_mtok(val: Any, is_openrouter_per_token: bool = False) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f) or f < 0.0:
        return None
    if f == 0.0:
        return 0.0
    # OpenRouter `pricing.prompt` / `pricing.completion` is per-token (e.g. "0.000003" = $3/Mtok).
    if is_openrouter_per_token and 0.0 < f < 0.01:
        f = f * 1_000_000.0
    return round(f, 4)


def _extract_coding_score(obj: Dict[str, Any]) -> Optional[float]:
    for key in ("coding_score", "coding", "swe_bench", "swebench", "swe_bench_verified", "aider_polyglot", "livecodebench", "humaneval"):
        val = obj.get(key)
        if _is_number(val) and float(val) >= 0.0:
            return round(float(val), 2)
    for nested_key in ("benchmarks", "evaluations", "scores", "metrics"):
        sub = obj.get(nested_key)
        if isinstance(sub, dict):
            for key in ("coding_score", "coding", "swe_bench", "swebench", "swe_bench_verified", "aider_polyglot", "livecodebench", "humaneval"):
                val = sub.get(key)
                if _is_number(val) and float(val) >= 0.0:
                    return round(float(val), 2)
    return None


def _extract_release_date(obj: Dict[str, Any]) -> Optional[str]:
    for key in ("release_date", "released_at", "created_at", "updated_at", "knowledge"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    created = obj.get("created")
    if isinstance(created, (int, float)) and not isinstance(created, bool) and created > 1_000_000_000:
        try:
            return datetime.datetime.fromtimestamp(float(created), tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass
    elif isinstance(created, str) and created.strip():
        return created.strip()
    return None


def _extract_context_window(obj: Dict[str, Any]) -> Optional[int]:
    for key in ("context_window", "context_length", "max_input_tokens", "context"):
        val = obj.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and int(val) > 0:
            return int(val)
    limit = obj.get("limit") or obj.get("limits") or obj.get("top_provider")
    if isinstance(limit, dict):
        for key in ("context", "context_length", "context_window", "max_input_tokens"):
            val = limit.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and int(val) > 0:
                return int(val)
    return None


def _parse_single_public_model(raw_obj: Dict[str, Any], fallback_id: str, source_name: str, source_url: str) -> Optional[Dict[str, Any]]:
    model_id = str(raw_obj.get("id") or raw_obj.get("model_id") or raw_obj.get("slug") or fallback_id or "").strip()
    name = str(raw_obj.get("name") or model_id).strip()
    if not model_id and not name:
        return None

    price_in: Optional[float] = None
    price_out: Optional[float] = None

    # Direct per-Mtok fields
    for in_k in ("price_in_per_mtok", "input_price", "cost_in", "input_per_mtok"):
        if raw_obj.get(in_k) is not None:
            price_in = _coerce_price_per_mtok(raw_obj.get(in_k), is_openrouter_per_token=False)
            break
    for out_k in ("price_out_per_mtok", "output_price", "cost_out", "output_per_mtok"):
        if raw_obj.get(out_k) is not None:
            price_out = _coerce_price_per_mtok(raw_obj.get(out_k), is_openrouter_per_token=False)
            break

    # Nested `cost` object (models.dev standard: $ per 1M tokens)
    cost_obj = raw_obj.get("cost")
    if isinstance(cost_obj, dict):
        if price_in is None:
            for k in ("input", "in", "prompt", "price_in_per_mtok"):
                if cost_obj.get(k) is not None:
                    price_in = _coerce_price_per_mtok(cost_obj.get(k), is_openrouter_per_token=False)
                    break
        if price_out is None:
            for k in ("output", "out", "completion", "price_out_per_mtok"):
                if cost_obj.get(k) is not None:
                    price_out = _coerce_price_per_mtok(cost_obj.get(k), is_openrouter_per_token=False)
                    break

    # Nested `pricing` object (OpenRouter standard: $ per token or per Mtok)
    pricing_obj = raw_obj.get("pricing")
    if isinstance(pricing_obj, dict):
        is_or = source_name == "openrouter"
        if price_in is None:
            for k in ("price_in_per_mtok", "input", "prompt", "in"):
                if pricing_obj.get(k) is not None:
                    price_in = _coerce_price_per_mtok(
                        pricing_obj.get(k),
                        is_openrouter_per_token=(is_or and k != "price_in_per_mtok"),
                    )
                    break
        if price_out is None:
            for k in ("price_out_per_mtok", "output", "completion", "out"):
                if pricing_obj.get(k) is not None:
                    price_out = _coerce_price_per_mtok(
                        pricing_obj.get(k),
                        is_openrouter_per_token=(is_or and k != "price_out_per_mtok"),
                    )
                    break

    return {
        "id": model_id or name,
        "name": name,
        "price_in_per_mtok": price_in,
        "price_out_per_mtok": price_out,
        "context_window": _extract_context_window(raw_obj),
        "release_date": _extract_release_date(raw_obj),
        "coding_score": _extract_coding_score(raw_obj),
        "source": source_name,
        "source_url": source_url,
    }


def parse_models_dev_payload(payload: Any) -> List[Dict[str, Any]]:
    """Extract normalized model metadata entries from `https://models.dev/api.json`."""
    entries: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for it in payload:
            if isinstance(it, dict):
                parsed = _parse_single_public_model(it, "", "models.dev", "https://models.dev")
                if parsed:
                    entries.append(parsed)
        return entries

    if not isinstance(payload, dict):
        return entries

    if isinstance(payload.get("models"), list):
        for it in payload["models"]:
            if isinstance(it, dict):
                parsed = _parse_single_public_model(it, "", "models.dev", "https://models.dev")
                if parsed:
                    entries.append(parsed)
        return entries

    if isinstance(payload.get("models"), dict):
        for mid, m_obj in payload["models"].items():
            if isinstance(m_obj, dict):
                parsed = _parse_single_public_model(m_obj, str(mid), "models.dev", "https://models.dev")
                if parsed:
                    entries.append(parsed)
        return entries

    # Provider-keyed structure: {"anthropic": {"models": {"claude-sonnet-5-5": {...}}}, ...}
    for key, val in payload.items():
        if not isinstance(val, dict):
            continue
        if isinstance(val.get("models"), dict):
            for mid, m_obj in val["models"].items():
                if isinstance(m_obj, dict):
                    parsed = _parse_single_public_model(m_obj, f"{key}/{mid}", "models.dev", "https://models.dev")
                    if parsed:
                        entries.append(parsed)
        elif isinstance(val.get("models"), list):
            for m_obj in val["models"]:
                if isinstance(m_obj, dict):
                    parsed = _parse_single_public_model(m_obj, "", "models.dev", "https://models.dev")
                    if parsed:
                        entries.append(parsed)
        elif any(k in val for k in ("cost", "pricing", "price_in_per_mtok", "limit", "context_window", "coding_score")):
            parsed = _parse_single_public_model(val, str(key), "models.dev", "https://models.dev")
            if parsed:
                entries.append(parsed)
    return entries


def parse_openrouter_payload(payload: Any) -> List[Dict[str, Any]]:
    """Extract normalized model metadata entries from `https://openrouter.ai/api/v1/models`."""
    entries: List[Dict[str, Any]] = []
    items: List[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload["data"]
        elif isinstance(payload.get("models"), list):
            items = payload["models"]
        elif isinstance(payload.get("models"), dict):
            items = [{"id": k, **v} for k, v in payload["models"].items() if isinstance(v, dict)]

    for it in items:
        if isinstance(it, dict):
            parsed = _parse_single_public_model(it, "", "openrouter", "https://openrouter.ai/models")
            if parsed:
                entries.append(parsed)
    return entries


def _default_http_fetcher(url: str, headers: Dict[str, str], timeout_sec: int = 15) -> Tuple[int, Dict[str, str], Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Quota-Hub/2.0", **headers}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = int(getattr(resp, "status", 200))
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body.strip() else {}
            return status, resp_headers, data
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            resp_headers = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
            return 304, resp_headers, None
        raise


def fetch_public_metadata(
    hub_dir: Optional[Path] = None,
    models_dev_data: Optional[Any] = None,
    openrouter_data: Optional[Any] = None,
    http_fetcher: Optional[Callable[[str, Dict[str, str]], Tuple[int, Dict[str, str], Any]]] = None,
    force_refresh: bool = False,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fetch and cache public model metadata from `models.dev` and `OpenRouter`.
    - Cached in `hub_dir / model_metadata_cache.json` for 24 hours (`METADATA_CACHE_TTL_SEC`).
    - Sends `If-None-Match` when ETag is cached.
    - On network failure: retains the last cached entries and reports `stale=True`, `staleness="stale"`.
    """
    now = float(now_ts if now_ts is not None else time.time())
    now_iso = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).isoformat()
    cache_file = _metadata_cache_path(hub_dir)

    cached_doc: Dict[str, Any] = {}
    if cache_file.exists():
        try:
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cached_doc = loaded
        except Exception:
            cached_doc = {}

    sources_cfg = [
        ("models_dev", "models.dev", MODELS_DEV_URL, models_dev_data, parse_models_dev_payload),
        ("openrouter", "openrouter", OPENROUTER_MODELS_URL, openrouter_data, parse_openrouter_payload),
    ]

    fetcher = http_fetcher or _default_http_fetcher
    updated_cache = copy.deepcopy(cached_doc)
    source_status: Dict[str, Dict[str, Any]] = {}
    all_entries: List[Dict[str, Any]] = []
    any_stale = False
    cache_dirty = False

    for src_key, src_label, url, injected_data, parser_fn in sources_cfg:
        prev_entry = updated_cache.get(src_key) if isinstance(updated_cache.get(src_key), dict) else {}
        prev_fetched_ts = float(prev_entry.get("fetched_ts") or 0.0)
        prev_etag = prev_entry.get("etag")
        prev_models = prev_entry.get("entries") if isinstance(prev_entry.get("entries"), list) else []

        if injected_data is not None:
            entries = parser_fn(injected_data)
            updated_cache[src_key] = {
                "fetched_ts": now,
                "fetched_at": now_iso,
                "etag": prev_etag,
                "entries": entries,
            }
            cache_dirty = True
            source_status[src_key] = {
                "name": src_label,
                "url": url,
                "stale": False,
                "cached": False,
                "etag": prev_etag,
                "fetched_at": now_iso,
                "count": len(entries),
                "error": None,
            }
            all_entries.extend(entries)
            continue

        is_fresh = bool(prev_models and (now - prev_fetched_ts) < METADATA_CACHE_TTL_SEC)
        if is_fresh and not force_refresh:
            source_status[src_key] = {
                "name": src_label,
                "url": url,
                "stale": False,
                "cached": True,
                "etag": prev_etag,
                "fetched_at": prev_entry.get("fetched_at"),
                "count": len(prev_models),
                "error": None,
            }
            all_entries.extend(prev_models)
            continue

        req_headers: Dict[str, str] = {}
        if prev_etag:
            req_headers["If-None-Match"] = str(prev_etag)

        try:
            status_code, resp_headers, raw_payload = fetcher(url, req_headers)
            norm_headers = {str(k).lower(): v for k, v in (resp_headers or {}).items()}
            new_etag = norm_headers.get("etag") or prev_etag
            if status_code == 304 and prev_models:
                updated_cache[src_key] = {
                    "fetched_ts": now,
                    "fetched_at": now_iso,
                    "etag": new_etag,
                    "entries": prev_models,
                }
                cache_dirty = True
                source_status[src_key] = {
                    "name": src_label,
                    "url": url,
                    "stale": False,
                    "cached": True,
                    "not_modified": True,
                    "etag": new_etag,
                    "fetched_at": now_iso,
                    "count": len(prev_models),
                    "error": None,
                }
                all_entries.extend(prev_models)
            elif 200 <= int(status_code) < 300:
                entries = parser_fn(raw_payload)
                updated_cache[src_key] = {
                    "fetched_ts": now,
                    "fetched_at": now_iso,
                    "etag": new_etag,
                    "entries": entries,
                }
                cache_dirty = True
                source_status[src_key] = {
                    "name": src_label,
                    "url": url,
                    "stale": False,
                    "cached": False,
                    "etag": new_etag,
                    "fetched_at": now_iso,
                    "count": len(entries),
                    "error": None,
                }
                all_entries.extend(entries)
            else:
                raise OSError(f"HTTP {status_code} from {url}")
        except Exception as exc:
            any_stale = True
            source_status[src_key] = {
                "name": src_label,
                "url": url,
                "stale": True,
                "cached": bool(prev_models),
                "etag": prev_etag,
                "fetched_at": prev_entry.get("fetched_at"),
                "count": len(prev_models),
                "error": str(exc),
            }
            all_entries.extend(prev_models)

    if cache_dirty:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.parent / f".{cache_file.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            tmp.write_text(json.dumps(updated_cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, cache_file)
        except OSError:
            pass

    return {
        "entries": all_entries,
        "sources": source_status,
        "stale": any_stale,
        "staleness": "stale" if any_stale else "fresh",
        "fetched_at": now_iso,
    }


def match_local_model_to_public(
    model_id: str,
    public_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Match a local `model_id` against public metadata entries by normalised name.
    Unmatched models return `{"metadata": "unknown", ...}`.
    """
    norm_exact = normalise_model_name(model_id, strip_effort=False)
    norm_base = normalise_model_name(model_id, strip_effort=True)
    sig_base = _token_signature(model_id)

    exact_matches: List[Dict[str, Any]] = []
    base_matches: List[Dict[str, Any]] = []
    sig_matches: List[Dict[str, Any]] = []

    for entry in public_entries:
        pub_id = str(entry.get("id") or "")
        pub_name = str(entry.get("name") or "")
        pub_norm_exact = normalise_model_name(pub_id, strip_effort=False)
        pub_norm_base = normalise_model_name(pub_id, strip_effort=True)
        name_norm_base = normalise_model_name(pub_name, strip_effort=True)

        if pub_norm_exact == norm_exact:
            exact_matches.append(entry)
        elif pub_norm_base == norm_base or (name_norm_base and name_norm_base == norm_base):
            base_matches.append(entry)
        elif _token_signature(pub_id) == sig_base:
            sig_matches.append(entry)

    matched = exact_matches or base_matches or sig_matches
    if not matched:
        return {
            "metadata": "unknown",
            "price_in_per_mtok": None,
            "price_out_per_mtok": None,
            "blended_price_per_mtok": None,
            "context_window": None,
            "release_date": None,
            "coding_score": None,
            "sources": [],
        }

    # Prefer models.dev first when merging fields, then OpenRouter
    matched_sorted = sorted(matched, key=lambda e: 0 if e.get("source") == "models.dev" else 1)
    price_in: Optional[float] = None
    price_out: Optional[float] = None
    context_window: Optional[int] = None
    release_date: Optional[str] = None
    coding_score: Optional[float] = None
    sources: List[Dict[str, str]] = []
    seen_sources: set[str] = set()

    for m in matched_sorted:
        if price_in is None and m.get("price_in_per_mtok") is not None:
            price_in = float(m["price_in_per_mtok"])
        if price_out is None and m.get("price_out_per_mtok") is not None:
            price_out = float(m["price_out_per_mtok"])
        if context_window is None and m.get("context_window") is not None:
            context_window = int(m["context_window"])
        if release_date is None and m.get("release_date"):
            release_date = str(m["release_date"])
        if coding_score is None and m.get("coding_score") is not None:
            coding_score = float(m["coding_score"])
        src_name = str(m.get("source") or "public")
        if src_name not in seen_sources:
            seen_sources.add(src_name)
            sources.append(
                {
                    "name": src_name,
                    "url": str(m.get("source_url") or ("https://models.dev" if src_name == "models.dev" else "https://openrouter.ai/models")),
                    "matched_id": str(m.get("id") or ""),
                }
            )

    blended = compute_blended_price(price_in, price_out)
    return {
        "metadata": "matched",
        "price_in_per_mtok": price_in,
        "price_out_per_mtok": price_out,
        "blended_price_per_mtok": blended,
        "context_window": context_window,
        "release_date": release_date,
        "coding_score": coding_score,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Section 3: Price Ratio, Review Scoring, Proposals & Decisions
# ---------------------------------------------------------------------------


def compute_blended_price(price_in: Optional[float], price_out: Optional[float]) -> Optional[float]:
    """Compute `blended_price = 0.25 * price_in + 0.75 * price_out` ($ per 1M tokens)."""
    if price_in is None or price_out is None:
        return None
    try:
        return round(0.25 * float(price_in) + 0.75 * float(price_out), 4)
    except (TypeError, ValueError):
        return None


def _infer_family(provider: str, model_id: str) -> str:
    m = model_id.lower()
    if "gemini" in m or "argon" in m or "hydrogen" in m or "carbon" in m or "flash" in m or "astra" in m:
        return "gemini"
    if "claude" in m or "sonnet" in m or "opus" in m or "haiku" in m:
        return "claude"
    if "gpt" in m or "sol" in m or "luna" in m or "terra" in m or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if provider == "agy":
        return "gemini"
    if provider == "claude":
        return "claude"
    return "openai"


def _infer_pool_and_windows(provider: str, family: str) -> Tuple[str, Tuple[str, str]]:
    if provider == "agy":
        if family in {"claude", "openai"}:
            return "agy_claude_gpt", ("claude_gpt_5h", "claude_gpt_weekly")
        return "agy_gemini", ("gemini_5h", "gemini_weekly")
    if provider == "claude":
        return "claude_shared", ("five_hour", "weekly")
    return "codex_shared", ("five_hour", "weekly")


def _infer_quota_key(provider: str, family: str, tier: str) -> str:
    t = tier.upper()
    if provider == "agy":
        if family in {"claude", "openai"}:
            return "claude_opus" if t == "HIGH" else "claude_sonnet"
        return "gemini_pro" if t == "HIGH" else "gemini_flash"
    if provider == "claude":
        if t == "LOW":
            return "haiku"
        if t == "HIGH":
            return "opus"
        return "sonnet"
    if t == "LOW":
        return "luna"
    if t == "MID":
        return "terra"
    return "sol"


def _infer_effort(model_id: str, tier: str) -> str:
    m = model_id.lower()
    if m.endswith("-low"):
        return "low"
    if m.endswith("-high"):
        return "high"
    if m.endswith("-medium"):
        return "medium"
    return "low" if tier.upper() == "LOW" else "medium"


def _resolve_provider_active_high_baseline(
    provider: str,
    catalog: List[Dict[str, Any]],
    public_entries: List[Dict[str, Any]],
    candidate_family: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find the current active HIGH model of `provider` and its `blended_price`.
    Checks catalog `price_in_per_mtok`/`price_out_per_mtok` first, then matches in `public_entries`,
    and falls back to `DEFAULT_PROVIDER_HIGH_PRICES[provider]`.
    """
    high_models = [
        m
        for m in catalog
        if m.get("provider") == provider
        and m.get("tier") == "HIGH"
        and str(m.get("status") or "active") == "active"
    ]
    # Sort so default_preferred medium-effort (and matching family if any) comes first
    high_models.sort(
        key=lambda m: (
            0 if m.get("default_preferred", True) else 1,
            0 if (candidate_family and m.get("family") == candidate_family) else 1,
            0 if m.get("effort") == "medium" else 1,
        )
    )

    for hm in high_models:
        b_cat = compute_blended_price(hm.get("price_in_per_mtok"), hm.get("price_out_per_mtok"))
        if b_cat is not None and b_cat > 0.0:
            return {"model_id": hm["model_id"], "blended_price": b_cat, "source": "catalog"}
        pub_match = match_local_model_to_public(hm["model_id"], public_entries)
        b_pub = pub_match.get("blended_price_per_mtok")
        if b_pub is not None and float(b_pub) > 0.0:
            return {"model_id": hm["model_id"], "blended_price": float(b_pub), "source": "public_metadata"}

    fallback_mid = high_models[0]["model_id"] if high_models else f"{provider}-high-baseline"
    fb_prices = DEFAULT_PROVIDER_HIGH_PRICES.get(provider, {"price_in_per_mtok": 5.0, "price_out_per_mtok": 15.0})
    fb_blended = compute_blended_price(fb_prices["price_in_per_mtok"], fb_prices["price_out_per_mtok"]) or 12.5
    return {"model_id": fallback_mid, "blended_price": fb_blended, "source": "fallback_default"}


def suggest_tier_and_cost_weight(
    model_id: str,
    price_ratio: Optional[float],
    coding_score: Optional[float],
    catalog: List[Dict[str, Any]],
) -> Tuple[str, float]:
    """
    Determine `suggested_tier` from price band and coding benchmark score relative to existing tiers,
    and `suggested_cost_weight` from `price_ratio`.
    """
    # Compare coding_score against existing catalog scores if present
    high_scores = [float(m["coding_score"]) for m in catalog if m.get("tier") == "HIGH" and _is_number(m.get("coding_score"))]
    mid_scores = [float(m["coding_score"]) for m in catalog if m.get("tier") == "MID" and _is_number(m.get("coding_score"))]
    high_ref = (sum(high_scores) / len(high_scores)) if high_scores else 75.0
    mid_ref = (sum(mid_scores) / len(mid_scores)) if mid_scores else 58.0

    # Normalise 0..1 coding scores to 0..100 scale for comparison if needed
    norm_score: Optional[float] = None
    if coding_score is not None:
        norm_score = float(coding_score) * 100.0 if 0.0 < float(coding_score) <= 1.0 and high_ref > 1.0 else float(coding_score)

    m_low = model_id.lower()
    if price_ratio is not None:
        if price_ratio < 0.35:
            tier = "MID" if (norm_score is not None and norm_score >= mid_ref) else "LOW"
        elif price_ratio < 0.85:
            if norm_score is not None and norm_score >= high_ref:
                tier = "HIGH"
            elif norm_score is not None and norm_score < (mid_ref * 0.75):
                tier = "LOW"
            else:
                tier = "MID"
        else:
            if norm_score is not None and norm_score < mid_ref:
                tier = "MID"
            else:
                tier = "HIGH"
        cost_weight = round(max(0.50, min(0.98, 0.96 - 0.12 * float(price_ratio))), 2)
    else:
        if norm_score is not None:
            if norm_score >= high_ref:
                tier = "HIGH"
            elif norm_score >= mid_ref:
                tier = "MID"
            else:
                tier = "LOW"
        elif any(tok in m_low for tok in ("-low", "haiku", "mini", "nano", "flash-lite")):
            tier = "LOW"
        elif any(tok in m_low for tok in ("opus", "argon", "pro", "astra", "-high")):
            tier = "HIGH"
        else:
            tier = "MID"
        cost_weight = 0.92 if tier == "LOW" else (0.86 if tier == "MID" else 0.80)

    return tier, cost_weight


def _is_in_reject_cooldown(model_id: str, cooldowns: Dict[str, Any], now_dt: datetime.datetime) -> Tuple[bool, Optional[Dict[str, Any]]]:
    cd = cooldowns.get(model_id)
    if not isinstance(cd, dict):
        return False, None
    exp_str = cd.get("expires_at")
    if not isinstance(exp_str, str) or not exp_str.strip():
        return False, None
    try:
        exp_dt = datetime.datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=datetime.timezone.utc)
        if now_dt < exp_dt:
            rem_days = round((exp_dt - now_dt).total_seconds() / 86400.0, 2)
            return True, {**cd, "remaining_days": rem_days}
    except Exception:
        pass
    return False, None


def evaluate_probation_models(
    catalog: List[Dict[str, Any]],
    scorecard: Optional[Dict[str, Any]] = None,
    now_dt: Optional[datetime.datetime] = None,
    probation_days: float = PROBATION_REVIEW_DAYS,
) -> List[Dict[str, Any]]:
    """
    After 14 days of probation, summarise the existing scorecard (acceptance, rework, measured quota cost)
    for that model and propose `promote` or `retire`; never auto-promote.
    """
    now = now_dt or datetime.datetime.now(datetime.timezone.utc)
    by_agent_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(scorecard, dict):
        for ag in scorecard.get("by_agent") or []:
            if isinstance(ag, dict) and ag.get("model_id"):
                by_agent_map[str(ag["model_id"])] = ag

    reviews: List[Dict[str, Any]] = []
    for m in catalog:
        if str(m.get("status") or "active") != "probation":
            continue
        added_str = m.get("added_at")
        days_elapsed = 0.0
        if isinstance(added_str, str) and added_str.strip():
            try:
                added_dt = datetime.datetime.fromisoformat(added_str.replace("Z", "+00:00"))
                if added_dt.tzinfo is None:
                    added_dt = added_dt.replace(tzinfo=datetime.timezone.utc)
                days_elapsed = max(0.0, (now - added_dt).total_seconds() / 86400.0)
            except Exception:
                days_elapsed = 0.0

        sc_entry = by_agent_map.get(m["model_id"]) or {}
        tasks_eval = int(sc_entry.get("tasks_evaluated") or 0)
        acc_rate = sc_entry.get("actual_feature_acceptance_rate")
        test_rate = sc_entry.get("automated_test_pass_rate")
        rework_avg = sc_entry.get("rework_per_task")
        reg_avg = sc_entry.get("regressions_per_task")
        cost_5h_pct = sc_entry.get("median_5h_cost_pct")
        cost_wk_pct = sc_entry.get("median_weekly_cost_pct")
        qual_score = sc_entry.get("empirical_quality_score")

        ready_for_decision = days_elapsed >= float(probation_days)
        if not ready_for_decision:
            continue

        if tasks_eval > 0 and float(acc_rate or 0.0) >= 0.75 and float(rework_avg or 0.0) <= 1.0:
            proposed_action = "promote"
            reason = (
                f"{days_elapsed:.1f}d probation complete: acceptance={int(round(float(acc_rate or 0) * 100))}%, "
                f"rework/task={rework_avg}, 5h cost={cost_5h_pct}% (n={tasks_eval}). Propose promote to active."
            )
        else:
            proposed_action = "retire"
            reason = (
                f"{days_elapsed:.1f}d probation complete: tasks={tasks_eval}, "
                f"acceptance={int(round(float(acc_rate or 0) * 100)) if acc_rate is not None else 'N/A'}%, "
                f"rework/task={rework_avg if rework_avg is not None else 'N/A'}. Propose retire."
            )

        reviews.append(
            {
                "model_id": m["model_id"],
                "provider": m["provider"],
                "tier": m["tier"],
                "added_at": added_str,
                "days_in_probation": round(days_elapsed, 2),
                "tasks_evaluated": tasks_eval,
                "acceptance_rate": acc_rate,
                "automated_test_pass_rate": test_rate,
                "rework_per_task": rework_avg,
                "regressions_per_task": reg_avg,
                "measured_5h_cost_pct": cost_5h_pct,
                "measured_weekly_cost_pct": cost_wk_pct,
                "empirical_quality_score": qual_score,
                "proposed_action": proposed_action,
                "auto_promoted": False,
                "summary": reason,
            }
        )
    return reviews


def run_model_review(
    catalog_path: Optional[Path] = None,
    hub_dir: Optional[Path] = None,
    agy_models_output: Optional[str] = None,
    agy_runner: Optional[Callable[[], str]] = None,
    codex_cache_path: Optional[Path] = None,
    codex_cache_data: Optional[Any] = None,
    claude_static_models: Optional[List[str]] = None,
    dispatch_history: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[Path] = None,
    models_dev_data: Optional[Any] = None,
    openrouter_data: Optional[Any] = None,
    http_fetcher: Optional[Callable[[str, Dict[str, str]], Tuple[int, Dict[str, str], Any]]] = None,
    scorecard: Optional[Dict[str, Any]] = None,
    settings_override: Optional[Dict[str, Any]] = None,
    force_refresh_metadata: bool = False,
    now_dt: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Execute a full model discovery and price/performance review run (no LLM, no paid API).
    Persists the latest review report to `model_review_state.json` and returns it.
    """
    now = now_dt or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    now_iso = now.isoformat()
    now_ts = now.timestamp()

    # Snapshot only for reading during (slow) discovery; the final save re-reads and merges under the lock.
    state = load_review_state(hub_dir)
    # A settings override applies to THIS run's scoring only; it is never written back to persisted settings.
    settings = copy.deepcopy(state["settings"])
    if settings_override:
        settings = validate_review_settings(settings_override, base=settings)
    snapshot_decision_count = len(state.get("decisions_log") or [])
    max_price_ratio = float(settings.get("max_price_ratio", 2.0))
    cooldowns = state.get("cooldowns") or {}

    catalog = load_model_catalog_resilient(catalog_path)
    catalog_by_id: Dict[str, Dict[str, Any]] = {m["model_id"]: m for m in catalog}

    local_disc = discover_local_models(
        agy_models_output=agy_models_output,
        agy_runner=agy_runner,
        codex_cache_path=codex_cache_path,
        codex_cache_data=codex_cache_data,
        claude_static_models=claude_static_models,
        dispatch_history=dispatch_history,
        db_path=db_path,
    )
    pub_meta = fetch_public_metadata(
        hub_dir=hub_dir,
        models_dev_data=models_dev_data,
        openrouter_data=openrouter_data,
        http_fetcher=http_fetcher,
        force_refresh=force_refresh_metadata,
        now_ts=now_ts,
    )
    pub_entries = pub_meta.get("entries") or []

    proposals: List[Dict[str, Any]] = []
    excluded_too_expensive: List[Dict[str, Any]] = []
    cooldown_skipped: List[Dict[str, Any]] = []

    # Candidate set: all locally discovered models + any catalog models to check for >20% price change
    candidates_to_check: List[Dict[str, Any]] = list(local_disc.get("models") or [])
    seen_local_ids = {c["model_id"] for c in candidates_to_check}
    for cm in catalog:
        if cm["model_id"] not in seen_local_ids and (cm.get("price_in_per_mtok") is not None or cm.get("price_out_per_mtok") is not None):
            candidates_to_check.append(
                {
                    "provider": cm["provider"],
                    "model_id": cm["model_id"],
                    "local_sources": ["catalog"],
                    "local_context_window": cm.get("context_window"),
                }
            )

    for loc in candidates_to_check:
        prov = str(loc["provider"]).lower()
        mid = str(loc["model_id"])
        existing = catalog_by_id.get(mid)

        pub_match = match_local_model_to_public(mid, pub_entries)
        new_blended = pub_match.get("blended_price_per_mtok")

        # Determine if this model should be reviewed:
        # 1. Not in catalog, OR
        # 2. In catalog and public blended price changed > 20% compared to catalog price
        change_reason = "new_model"
        price_change_pct: Optional[float] = None
        if existing is not None:
            old_blended = compute_blended_price(existing.get("price_in_per_mtok"), existing.get("price_out_per_mtok"))
            if old_blended is None or old_blended <= 0.0 or new_blended is None:
                continue
            price_change_pct = round(abs(float(new_blended) - old_blended) / old_blended, 4)
            if price_change_pct <= 0.20:
                continue
            change_reason = "price_changed_gt_20pct"

        # Check 30-day reject cooldown
        in_cd, cd_info = _is_in_reject_cooldown(mid, cooldowns, now)
        if in_cd and cd_info is not None:
            cooldown_skipped.append(
                {
                    "model_id": mid,
                    "provider": prov,
                    "rejected_at": cd_info.get("rejected_at"),
                    "expires_at": cd_info.get("expires_at"),
                    "remaining_days": cd_info.get("remaining_days"),
                }
            )
            continue

        family = existing["family"] if existing else _infer_family(prov, mid)
        pool_id, _ = _infer_pool_and_windows(prov, family)
        baseline = _resolve_provider_active_high_baseline(prov, catalog, pub_entries, candidate_family=family)
        base_blended = float(baseline["blended_price"])
        # A candidate that is itself the provider's active HIGH baseline is compared against its own
        # current public price (its stored catalog price is stale by definition), so ratio == 1.0.
        if baseline["model_id"] == mid and new_blended is not None and float(new_blended) > 0.0:
            base_blended = float(new_blended)

        price_ratio: Optional[float] = None
        if new_blended is not None and base_blended > 0.0:
            price_ratio = round(float(new_blended) / base_blended, 4)

        coding_score = pub_match.get("coding_score")
        if coding_score is None and existing is not None:
            coding_score = existing.get("coding_score")
        ctx_win = pub_match.get("context_window") or loc.get("local_context_window") or (existing.get("context_window") if existing else 200_000)

        suggested_tier, suggested_cost_weight = suggest_tier_and_cost_weight(
            mid, price_ratio=price_ratio, coding_score=coding_score, catalog=catalog
        )

        item_record: Dict[str, Any] = {
            "model_id": mid,
            "provider": prov,
            "family": family,
            "pool_id": pool_id,
            "metadata": pub_match["metadata"],
            "unknown_price": price_ratio is None,
            "price_in_per_mtok": pub_match.get("price_in_per_mtok"),
            "price_out_per_mtok": pub_match.get("price_out_per_mtok"),
            "blended_price_per_mtok": new_blended,
            "baseline_high_model_id": baseline["model_id"],
            "baseline_high_blended_price": base_blended,
            "price_ratio": price_ratio,
            "price_change_pct": price_change_pct,
            "coding_score": coding_score,
            "context_window": int(ctx_win) if ctx_win else 200_000,
            "release_date": pub_match.get("release_date"),
            "suggested_tier": suggested_tier,
            "suggested_cost_weight": suggested_cost_weight,
            "local_sources": loc.get("local_sources") or [],
            "sources": pub_match.get("sources") or [],
            "change_reason": change_reason,
            "in_catalog": existing is not None,
        }

        if price_ratio is not None and price_ratio > max_price_ratio:
            item_record["status"] = "excluded (too expensive)"
            item_record["reason"] = (
                f"excluded (too expensive): price_ratio {price_ratio:.2f}x exceeds max_price_ratio {max_price_ratio:.2f}x "
                f"(blended ${new_blended:.2f}/Mtok vs {baseline['model_id']} ${base_blended:.2f}/Mtok)"
            )
            excluded_too_expensive.append(item_record)
        else:
            flags: List[str] = []
            if price_ratio is None:
                flags.append("unknown_price")
            if change_reason == "price_changed_gt_20pct":
                flags.append("price_changed")
            item_record["flags"] = flags
            item_record["status"] = "proposed"
            proposals.append(item_record)

    probation_reviews = evaluate_probation_models(catalog, scorecard=scorecard, now_dt=now)

    report: Dict[str, Any] = {
        "proposals": proposals,
        "excluded_too_expensive": excluded_too_expensive,
        "probation_reviews": probation_reviews,
        "cooldowns": cooldowns,
        "cooldown_skipped": cooldown_skipped,
        "rejected_ids": list(local_disc.get("rejected_ids") or []),
        "settings": copy.deepcopy(settings),
        "settings_used": copy.deepcopy(settings),
        "sources": {
            "local": local_disc.get("sources") or {},
            "public": pub_meta.get("sources") or {},
        },
        "last_run": now_iso,
        "stale": bool(pub_meta.get("stale", False)),
        "staleness": str(pub_meta.get("staleness") or "fresh"),
        "catalog_summary": {
            "total": len(catalog),
            "active": sum(1 for m in catalog if m.get("status", "active") == "active"),
            "probation": sum(1 for m in catalog if m.get("status") == "probation"),
            "retired": sum(1 for m in catalog if m.get("status") == "retired"),
        },
    }

    with _REVIEW_TXN_LOCK:
        fresh = load_review_state(hub_dir)
        fresh_cooldowns = fresh.get("cooldowns") or {}
        new_decisions = (fresh.get("decisions_log") or [])[: max(0, len(fresh.get("decisions_log") or []) - snapshot_decision_count)]
        decided_ids = {d.get("model_id") for d in new_decisions if isinstance(d, dict)}
        report["proposals"] = [
            p
            for p in report["proposals"]
            if p["model_id"] not in decided_ids and not _is_in_reject_cooldown(p["model_id"], fresh_cooldowns, now)[0]
        ]
        report["cooldowns"] = copy.deepcopy(fresh_cooldowns)
        report["settings"] = copy.deepcopy(fresh["settings"])
        fresh["last_run"] = now_iso
        fresh["last_report"] = report
        save_review_state(fresh, hub_dir=hub_dir)
    return report


def get_latest_model_review(
    catalog_path: Optional[Path] = None,
    hub_dir: Optional[Path] = None,
    scorecard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the latest persisted review report (or a lightweight state snapshot if never run)."""
    state = load_review_state(hub_dir)
    catalog = load_model_catalog_resilient(catalog_path)
    report = state.get("last_report")
    if isinstance(report, dict):
        out = copy.deepcopy(report)
        out["settings"] = copy.deepcopy(state["settings"])
        out["cooldowns"] = copy.deepcopy(state.get("cooldowns") or {})
        out["probation_reviews"] = evaluate_probation_models(catalog, scorecard=scorecard)
        out["catalog_status"] = get_catalog_status()
        out["catalog_summary"] = {
            "total": len(catalog),
            "active": sum(1 for m in catalog if m.get("status", "active") == "active"),
            "probation": sum(1 for m in catalog if m.get("status") == "probation"),
            "retired": sum(1 for m in catalog if m.get("status") == "retired"),
        }
        return out

    return {
        "proposals": [],
        "excluded_too_expensive": [],
        "probation_reviews": evaluate_probation_models(catalog, scorecard=scorecard),
        "cooldowns": copy.deepcopy(state.get("cooldowns") or {}),
        "cooldown_skipped": [],
        "settings": copy.deepcopy(state["settings"]),
        "sources": {"local": {}, "public": {}},
        "last_run": state.get("last_run"),
        "stale": False,
        "staleness": "never_run",
        "catalog_status": get_catalog_status(),
        "catalog_summary": {
            "total": len(catalog),
            "active": sum(1 for m in catalog if m.get("status", "active") == "active"),
            "probation": sum(1 for m in catalog if m.get("status") == "probation"),
            "retired": sum(1 for m in catalog if m.get("status") == "retired"),
        },
    }


def apply_model_decision(
    model_id: str,
    decision: str,
    tier: Optional[str] = None,
    provider: Optional[str] = None,
    catalog_path: Optional[Path] = None,
    hub_dir: Optional[Path] = None,
    now_dt: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Apply an owner decision (`approve` | `probation` | `reject` | `promote` | `retire`) for `model_id`:
    - `reject`: records a 30-day cooldown for `model_id` without modifying `model_catalog.json`.
    - `approve` / `probation` / `promote` / `retire`: writes `model_catalog.json` atomically with a
      timestamped backup and updates in-process `MODEL_CATALOG`.
    """
    mid = str(model_id or "").strip()
    if not mid:
        raise ValueError("model_id must be a non-empty string")
    if not SAFE_MODEL_ID_RE.match(mid):
        print(f"WARNING: rejecting decision for unsafe model id {mid[:60]!r}", flush=True)
        raise ValueError("model_id contains unsupported characters (allowed: A-Z a-z 0-9 . _ : / -, max 120)")

    dec = str(decision or "").strip().lower()
    if dec not in {"approve", "probation", "reject", "promote", "retire"}:
        raise ValueError("decision must be 'approve', 'probation', 'reject', 'promote', or 'retire'")

    if tier is not None:
        t_clean = str(tier).strip().upper()
        if t_clean not in TIER_TO_RANK:
            raise ValueError("tier must be 'LOW', 'MID', or 'HIGH'")
    else:
        t_clean = None

    now = now_dt or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    now_iso = now.isoformat()

    with _REVIEW_TXN_LOCK:
        state = load_review_state(hub_dir)
        settings = state.get("settings") or DEFAULT_REVIEW_SETTINGS
        last_report = state.get("last_report") if isinstance(state.get("last_report"), dict) else {}
        proposals = last_report.get("proposals") if isinstance(last_report.get("proposals"), list) else []
        prop_match = next((p for p in proposals if isinstance(p, dict) and p.get("model_id") == mid), None)

        if dec == "reject":
            exp_dt = now + datetime.timedelta(days=REJECT_COOLDOWN_DAYS)
            cd_entry = {
                "model_id": mid,
                "rejected_at": now_iso,
                "expires_at": exp_dt.isoformat(),
                "cooldown_days": REJECT_COOLDOWN_DAYS,
            }
            state.setdefault("cooldowns", {})[mid] = cd_entry
            if isinstance(state.get("last_report"), dict) and isinstance(state["last_report"].get("proposals"), list):
                state["last_report"]["proposals"] = [
                    p for p in state["last_report"]["proposals"] if not (isinstance(p, dict) and p.get("model_id") == mid)
                ]
            state.setdefault("decisions_log", []).insert(
                0, {"model_id": mid, "decision": "reject", "timestamp": now_iso, "expires_at": cd_entry["expires_at"]}
            )
            save_review_state(state, hub_dir=hub_dir)
            return {
                "model_id": mid,
                "decision": "reject",
                "catalog_updated": False,
                "cooldown": cd_entry,
                "backup_path": None,
            }

        catalog = load_model_catalog_resilient(catalog_path)
        existing_idx = next((i for i, m in enumerate(catalog) if m["model_id"] == mid), None)

        target_status = "active" if dec in {"approve", "promote"} else ("probation" if dec == "probation" else "retired")
        prob_share = float(settings.get("probation_share", 0.10))

        if existing_idx is not None:
            entry = copy.deepcopy(catalog[existing_idx])
            if t_clean is not None:
                entry["tier"] = t_clean
                entry["tier_rank"] = TIER_TO_RANK[t_clean]
            entry["status"] = target_status
            if target_status == "probation":
                entry["probation_share"] = prob_share
                if not entry.get("added_at"):
                    entry["added_at"] = now_iso
            if prop_match:
                if prop_match.get("price_in_per_mtok") is not None:
                    entry["price_in_per_mtok"] = float(prop_match["price_in_per_mtok"])
                if prop_match.get("price_out_per_mtok") is not None:
                    entry["price_out_per_mtok"] = float(prop_match["price_out_per_mtok"])
                if prop_match.get("coding_score") is not None:
                    entry["coding_score"] = float(prop_match["coding_score"])
                if prop_match.get("context_window"):
                    entry["context_window"] = int(prop_match["context_window"])
                if prop_match.get("suggested_cost_weight"):
                    entry["cost_weight"] = float(prop_match["suggested_cost_weight"])
            catalog[existing_idx] = entry
        else:
            if dec == "retire":
                raise ValueError(f"Cannot retire unknown model '{mid}' not in catalog")
            prov = str(
                provider
                or (prop_match.get("provider") if prop_match else None)
                or ("agy" if ("gemini" in mid.lower() or "argon" in mid.lower() or "flash" in mid.lower()) else ("claude" if "claude" in mid.lower() else "codex"))
            ).strip().lower()
            if prov not in VALID_PROVIDERS:
                prov = "agy"
            family = (prop_match.get("family") if prop_match else None) or _infer_family(prov, mid)
            chosen_tier = t_clean or (prop_match.get("suggested_tier") if prop_match else None) or "MID"
            chosen_tier = chosen_tier if chosen_tier in TIER_TO_RANK else "MID"
            pool_id, pool_windows = _infer_pool_and_windows(prov, family)
            quota_key = _infer_quota_key(prov, family, chosen_tier)
            effort = _infer_effort(mid, chosen_tier)
            cli_model = re.sub(r"-(low|medium|high)$", "", mid) if prov == "codex" else mid
            ctx_win = int((prop_match.get("context_window") if prop_match else None) or 200_000)
            cost_weight = float((prop_match.get("suggested_cost_weight") if prop_match else None) or 0.85)
            latency_score = 0.92 if chosen_tier == "LOW" else (0.86 if chosen_tier == "MID" else 0.80)
            src_names = ", ".join(s.get("name", "") for s in ((prop_match.get("sources") if prop_match else None) or []) if isinstance(s, dict)) or "model_review"

            entry = {
                "provider": prov,
                "model_id": mid,
                "cli_model": cli_model,
                "quota_key": quota_key,
                "pool_id": pool_id,
                "pool_windows": pool_windows,
                "family": family,
                "tier": chosen_tier,
                "tier_rank": TIER_TO_RANK[chosen_tier],
                "effort": effort,
                "default_preferred": not (chosen_tier == "HIGH" and effort == "high"),
                "context_window": ctx_win,
                "latency_score": latency_score,
                "cost_weight": cost_weight,
                "label": f"{prov.upper()} · {mid} ({chosen_tier})",
                "status": target_status,
                "added_at": now_iso,
                "source": src_names,
            }
            if target_status == "probation":
                entry["probation_share"] = prob_share
            if prop_match:
                if prop_match.get("price_in_per_mtok") is not None:
                    entry["price_in_per_mtok"] = float(prop_match["price_in_per_mtok"])
                if prop_match.get("price_out_per_mtok") is not None:
                    entry["price_out_per_mtok"] = float(prop_match["price_out_per_mtok"])
                if prop_match.get("coding_score") is not None:
                    entry["coding_score"] = float(prop_match["coding_score"])
            catalog.append(entry)

        save_res = save_model_catalog(catalog, path=catalog_path)

        # Remove decided model from pending proposals in persisted review state
        state.get("cooldowns", {}).pop(mid, None)
        if isinstance(state.get("last_report"), dict) and isinstance(state["last_report"].get("proposals"), list):
            state["last_report"]["proposals"] = [
                p for p in state["last_report"]["proposals"] if not (isinstance(p, dict) and p.get("model_id") == mid)
            ]
        state.setdefault("decisions_log", []).insert(
            0,
            {
                "model_id": mid,
                "decision": dec,
                "status": target_status,
                "tier": entry["tier"],
                "timestamp": now_iso,
                "backup_path": save_res["backup_path"],
            },
        )
        save_review_state(state, hub_dir=hub_dir)

        return {
            "model_id": mid,
            "decision": dec,
            "status": target_status,
            "tier": entry["tier"],
            "model_entry": entry,
            "catalog_updated": True,
            "catalog_path": save_res["catalog_path"],
            "backup_path": save_res["backup_path"],
            "backups_kept": save_res["backups_kept"],
        }


def is_scheduled_review_due(
    hub_dir: Optional[Path] = None,
    now_dt: Optional[datetime.datetime] = None,
) -> bool:
    """Return True when `last_run` is missing or older than `review_interval_days`."""
    state = load_review_state(hub_dir)
    settings = state.get("settings") or DEFAULT_REVIEW_SETTINGS
    interval_days = float(settings.get("review_interval_days", 7))
    last_run_str = state.get("last_run")
    if not isinstance(last_run_str, str) or not last_run_str.strip():
        return True
    now = now_dt or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    try:
        last_dt = datetime.datetime.fromisoformat(last_run_str.replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.timezone.utc)
        return (now - last_dt).total_seconds() >= (interval_days * 86400.0)
    except Exception:
        return True
