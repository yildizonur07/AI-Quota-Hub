"""
Dynamic Multi-Provider Orchestrator, Reset-Aware Quota Scheduler & Task-Based AI Tool Selector.

Implements the canonical rules from `~/.agents/ROUTING.md`, `~/.agents/DELEGATION.md`,
`~/.agents/skills/orchestrator/SKILL.md`, and `scheduler.py`:
1. Two Selectable Operating Modes (`endurance` default, `sprint` near-term throughput).
2. Capability Floor classification (LOW, MID, HIGH) separated from Reasoning Effort
   (`low`, `medium`, `high`). Demanding HIGH-floor work defaults to `medium` effort;
   `high` effort is reserved for critical/high-consequence work or explicit escalation.
3. Reset-Aware Mathematical Scheduling (`scheduler.py`):
   - Evaluates both `five_hour` and `weekly` windows (`R(p,w)`, `T(p,w)`, `usable_before_reset`).
   - Evaluates AGY's two separate pools (`agy_gemini` vs `agy_claude_gpt`), `claude_shared`, and `codex_shared`.
   - Exploits soon-to-reset ("stranded") quota safely when a task's predicted upper consumption fits
     before reset (e.g. Provider A with 10% weekly resetting in 2h beats Provider B with 80% weekly resetting in 6d).
   - Rejects any candidate where EITHER the 5-hour OR weekly window cannot fit the predicted task cost.
   - Protects Codex as Orchestrator via a dynamic reserve `R_orch(codex)` based on unfinished tasks,
     pending reviews, time to reset, and fallback coordinator availability.
   - Treats missing, null, offline, or stale (`> 120s`) telemetry as `UNKNOWN`/`UNAVAILABLE` (never `0% used` or `100% full`).
4. Independent Review: a different-family reviewer gets a soft score bonus; same-family review stays
   routable unless the task sets `require_cross_family`.
5. Task-Based AI Tool Selection (`recommend_task_tools`) delegated to an optional user tool catalog
   (`hub_config.tool_catalog_path()`); without one, no external tools are recommended.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hub_config import tool_catalog_path

from allocation import (
    DEFAULT_PREFERENCE_WEIGHTS,
    _PREFERENCE_WEIGHTS,
    _SHARED_CREDITS,
    _SWRR_LOCK,
    BurnRateLearner,
    apply_band_effort,
    classify_sigma_band,
    coerce_credits,
    coerce_preference_weights,
    compute_allocation,
    compute_target_shares,
    get_shared_credits,
    get_shared_preference_generation,
    get_shared_preference_weights,
    reset_credits,
    reset_shared_credits,
    select_and_commit,
    set_shared_credits,
    set_shared_preference_weights,
    step_swrr,
    validate_preference_weights,
)
from model_review import (
    is_probation_eligible,
    load_model_catalog,
    load_model_catalog_resilient,
    get_catalog_status,
    register_catalog_reload_hook,
    reset_probation_counters,
    save_model_catalog,
    should_route_probation,
    tick_probation_eligible,
    validate_model_catalog_doc,
    validate_model_entry,
)
from scheduler import (
    BacklogQueueManager,
    QuotaCostLearner,
    _first_int,
    build_pool_states,
    compute_dynamic_codex_reserve,
    evaluate_banked_reset_strategy,
    evaluate_candidate_schedule_score,
    load_orchestrator_policy,
    model_to_pool_id,
    run_multi_day_benchmark,
)


QUOTA_STALE_SECONDS = 120
HUB_HOME = Path(os.environ.get("QUOTA_HUB_HOME") or Path(__file__).resolve().parent)
DEFAULT_TELEMETRY_DB = HUB_HOME / "scheduler_telemetry.sqlite"
COST_LEARNER = QuotaCostLearner(DEFAULT_TELEMETRY_DB)
BURN_LEARNER = BurnRateLearner(DEFAULT_TELEMETRY_DB, cost_learner=COST_LEARNER)
QUEUE_MANAGER = BacklogQueueManager(DEFAULT_TELEMETRY_DB)

_CREDIT_LOCK = _SWRR_LOCK
_LAST_COST_LEARNER_ID: Optional[int] = None
_CREDITS_EXPLICITLY_SEEDED: bool = False

_ALLOCATION_STATE: Dict[str, Any] = {
    "preference_weights": _PREFERENCE_WEIGHTS,
    "allocation_credits": _SHARED_CREDITS,
    "preference_generation": 0,
}


def get_preference_generation() -> int:
    return get_shared_preference_generation()


def get_preference_weights() -> Dict[str, int]:
    return get_shared_preference_weights()


def set_preference_weights(weights: Dict[str, Any], reset_swrr: bool = True) -> Dict[str, int]:
    global _CREDITS_EXPLICITLY_SEEDED
    cleaned = set_shared_preference_weights(weights, reset_swrr=reset_swrr, bump_generation=True)
    with _CREDIT_LOCK:
        _ALLOCATION_STATE["preference_generation"] = get_shared_preference_generation()
        if reset_swrr:
            _CREDITS_EXPLICITLY_SEEDED = True
    return cleaned


def get_allocation_credits() -> Dict[str, float]:
    return get_shared_credits()


def set_allocation_credits(
    credits: Optional[Dict[str, Any]],
    expected_generation: Optional[int] = None,
) -> Dict[str, float]:
    global _CREDITS_EXPLICITLY_SEEDED
    with _CREDIT_LOCK:
        res = set_shared_credits(credits, expected_generation=expected_generation)
        if expected_generation is None or int(expected_generation) == get_shared_preference_generation():
            _CREDITS_EXPLICITLY_SEEDED = True
        return res


def reset_allocation_credits(bump_generation: bool = False) -> Dict[str, float]:
    global _CREDITS_EXPLICITLY_SEEDED
    fresh = reset_shared_credits(bump_generation=bump_generation)
    with _CREDIT_LOCK:
        _ALLOCATION_STATE["preference_generation"] = get_shared_preference_generation()
        _CREDITS_EXPLICITLY_SEEDED = True
    return fresh

# Canonical Model Catalog loaded from `model_catalog.json` (aligned with Global Machine Tier Map)
MODEL_CATALOG: List[Dict[str, Any]] = load_model_catalog_resilient()

WEB_JUDGE_SEATS: List[Dict[str, Any]] = [
    {
        "provider": "web",
        "surface": "gemini",
        "model_id": "web:gemini",
        "cli_model": "gemini",
        "pool_id": "web_gemini",
        "family": "gemini",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "medium",
        "context_window": 1000000,
        "latency_score": 0.78,
        "cost_weight": 1.0,
        "zero_api_cost": True,
        "predicted_cost": 0.0,
        "label": "Web Judge · Gemini (Argon/Pro Temporary Chat)",
    },
    {
        "provider": "web",
        "surface": "chatgpt",
        "model_id": "web:chatgpt",
        "cli_model": "chatgpt",
        "pool_id": "web_chatgpt",
        "family": "openai",
        "tier": "HIGH",
        "tier_rank": 3,
        "effort": "medium",
        "context_window": 256000,
        "latency_score": 0.76,
        "cost_weight": 1.0,
        "zero_api_cost": True,
        "predicted_cost": 0.0,
        "label": "Web Judge · ChatGPT (Temporary Chat)",
    },
]

_WEB_JUDGE_LOCK = threading.Lock()


def _initial_web_judge_state() -> Dict[str, Dict[str, Any]]:
    return {
        "gemini": {
            "surface": "gemini",
            "pool_id": "web_gemini",
            "family": "gemini",
            "zero_api_cost": True,
            "status": "UNKNOWN",
            "rate_limited": False,
            "remaining_fraction": None,
            "used_percent": None,
            "calls_count": 0,
            "limit_events_count": 0,
            "auth_wall_count": 0,
            "last_call_at": None,
            "last_limit_at": None,
            "reset_hint": None,
            "last_message": None,
            "recent_events": [],
        },
        "chatgpt": {
            "surface": "chatgpt",
            "pool_id": "web_chatgpt",
            "family": "openai",
            "zero_api_cost": True,
            "status": "UNKNOWN",
            "rate_limited": False,
            "remaining_fraction": None,
            "used_percent": None,
            "calls_count": 0,
            "limit_events_count": 0,
            "auth_wall_count": 0,
            "last_call_at": None,
            "last_limit_at": None,
            "reset_hint": None,
            "last_message": None,
            "recent_events": [],
        },
    }


_WEB_JUDGE_STATE: Dict[str, Dict[str, Any]] = _initial_web_judge_state()


def reset_web_judge_state() -> None:
    """Reset in-memory web judge signal state to clean defaults (useful for tests)."""
    with _WEB_JUDGE_LOCK:
        _WEB_JUDGE_STATE.clear()
        _WEB_JUDGE_STATE.update(_initial_web_judge_state())


def load_web_judge_state(saved_state: Optional[Dict[str, Any]]) -> None:
    """Restore persisted web judge signal state while enforcing unknown quota (never 0% or 1.0)."""
    if not isinstance(saved_state, dict):
        return
    surfaces = saved_state.get("surfaces") if isinstance(saved_state.get("surfaces"), dict) else saved_state
    if not isinstance(surfaces, dict):
        return
    defaults = _initial_web_judge_state()
    with _WEB_JUDGE_LOCK:
        for surf, def_entry in defaults.items():
            raw_entry = surfaces.get(surf)
            if not isinstance(raw_entry, dict):
                continue
            merged = dict(def_entry)
            for k in (
                "rate_limited",
                "calls_count",
                "limit_events_count",
                "auth_wall_count",
                "last_call_at",
                "last_limit_at",
                "reset_hint",
                "last_message",
            ):
                if k in raw_entry:
                    merged[k] = raw_entry[k]
            merged["rate_limited"] = bool(merged.get("rate_limited", False))
            merged["status"] = "exhausted" if merged["rate_limited"] else "UNKNOWN"
            merged["remaining_fraction"] = None
            merged["used_percent"] = None
            merged["zero_api_cost"] = True
            recent = raw_entry.get("recent_events")
            merged["recent_events"] = list(recent[:20]) if isinstance(recent, list) else []
            _WEB_JUDGE_STATE[surf] = merged


def get_web_judge_status() -> Dict[str, Any]:
    """Return a deep snapshot of current web judge seat status (`gemini` and `chatgpt`)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    with _WEB_JUDGE_LOCK:
        surfaces_copy: Dict[str, Dict[str, Any]] = {}
        for surf, st in _WEB_JUDGE_STATE.items():
            exp_iso = st.get("limit_expires_at")
            if st.get("rate_limited") and isinstance(exp_iso, str) and exp_iso:
                try:
                    exp_dt = datetime.datetime.fromisoformat(exp_iso)
                    if now_utc >= exp_dt:
                        st["rate_limited"] = False
                        st["status"] = "UNKNOWN"
                        st["limit_expires_at"] = None
                except Exception:
                    pass
            entry = dict(st)
            entry["remaining_fraction"] = None
            entry["used_percent"] = None
            entry["status"] = "exhausted" if bool(entry.get("rate_limited")) else "UNKNOWN"
            entry["recent_events"] = list(st.get("recent_events") or [])
            surfaces_copy[surf] = entry
        return {
            "zero_api_cost": True,
            "quota_telemetry": "UNKNOWN",
            "surfaces": surfaces_copy,
            "gemini": dict(surfaces_copy.get("gemini", {})),
            "chatgpt": dict(surfaces_copy.get("chatgpt", {})),
            "seats": [dict(s) for s in WEB_JUDGE_SEATS],
        }


def record_web_judge_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Record a web judge telemetry/signal event without ever fabricating 0% used or 100% remaining quota."""
    if not isinstance(event, dict):
        event = {}
    raw_surface = str(event.get("surface") or "").lower().strip()
    if raw_surface.startswith("web:"):
        raw_surface = raw_surface[4:].strip()
    if raw_surface.startswith("web_"):
        raw_surface = raw_surface[4:].strip()
    surface = "chatgpt" if raw_surface in {"chatgpt", "openai", "gpt"} else ("gemini" if raw_surface == "gemini" else raw_surface)
    if surface not in {"gemini", "chatgpt"}:
        raise ValueError(f"Unsupported web judge surface: {event.get('surface')!r} (expected 'gemini' or 'chatgpt')")

    raw_type = str(event.get("event_type") or "WEB_JUDGE_CALL_COMPLETED").upper().strip()
    allowed_types = {
        "WEB_JUDGE_CALL_COMPLETED",
        "SURFACE_LIMIT_REACHED",
        "WEB_JUDGE_LIMIT_REACHED",
        "SURFACE_AUTH_REQUIRED",
        "WEB_JUDGE_AUTH_REQUIRED",
        "WEB_JUDGE_AUTH_WALL",
        "WEB_JUDGE_ISOLATION_ERROR",
        "WEB_JUDGE_SURFACE_ERROR",
        "WEB_JUDGE_CLEAR_LIMIT",
    }
    if raw_type not in allowed_types:
        raw_type = "WEB_JUDGE_CALL_COMPLETED"

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()
    with _WEB_JUDGE_LOCK:
        if surface not in _WEB_JUDGE_STATE:
            _WEB_JUDGE_STATE.update(_initial_web_judge_state())
        st = _WEB_JUDGE_STATE[surface]
        if raw_type == "WEB_JUDGE_CALL_COMPLETED":
            st["calls_count"] = int(st.get("calls_count", 0)) + 1
            st["last_call_at"] = now_iso
            st["rate_limited"] = False
            st["limit_expires_at"] = None
            if event.get("message") is not None:
                st["last_message"] = event.get("message")
            st["status"] = "UNKNOWN"
        elif raw_type in {"SURFACE_LIMIT_REACHED", "WEB_JUDGE_LIMIT_REACHED"}:
            st["limit_events_count"] = int(st.get("limit_events_count", 0)) + 1
            st["rate_limited"] = True
            st["status"] = "exhausted"
            st["last_limit_at"] = now_iso
            st["limit_expires_at"] = (now_dt + datetime.timedelta(seconds=int(event.get("ttl_seconds") or 3600))).isoformat()
            st["reset_hint"] = event.get("reset_hint")
            st["last_message"] = event.get("message")
        elif raw_type in {"SURFACE_AUTH_REQUIRED", "WEB_JUDGE_AUTH_REQUIRED", "WEB_JUDGE_AUTH_WALL"}:
            st["auth_wall_count"] = int(st.get("auth_wall_count", 0)) + 1
            st["last_message"] = event.get("message") or event.get("error")
            st["status"] = "exhausted" if bool(st.get("rate_limited")) else "UNKNOWN"
        elif raw_type in {"WEB_JUDGE_ISOLATION_ERROR", "WEB_JUDGE_SURFACE_ERROR"}:
            st["error_events_count"] = int(st.get("error_events_count", 0)) + 1
            st["last_message"] = event.get("message") or event.get("error")
            st["status"] = "exhausted" if bool(st.get("rate_limited")) else "UNKNOWN"
        elif raw_type == "WEB_JUDGE_CLEAR_LIMIT":
            st["rate_limited"] = False
            st["limit_expires_at"] = None
            st["status"] = "UNKNOWN"
            st["reset_hint"] = None
            if event.get("message") is not None:
                st["last_message"] = event.get("message")

        # CRITICAL RULE: unknown quota MUST stay None / "UNKNOWN", NEVER 0% used or 1.0 remaining!
        st["remaining_fraction"] = None
        st["used_percent"] = None

        ev_record = {
            "timestamp": now_iso,
            "event_type": raw_type,
            "surface": surface,
            "call_id": event.get("call_id"),
            "mode": event.get("mode"),
            "model_selected": event.get("model_selected"),
            "prompt_chars": event.get("prompt_chars"),
            "reply_chars": event.get("reply_chars"),
            "duration_seconds": event.get("duration_seconds"),
            "reset_hint": event.get("reset_hint"),
            "message": event.get("message"),
        }
        recent = list(st.get("recent_events") or [])
        recent.insert(0, ev_record)
        st["recent_events"] = recent[:20]

    return get_web_judge_status()


def _sync_in_memory_catalog(updated_models: List[Dict[str, Any]]) -> None:
    MODEL_CATALOG[:] = updated_models


register_catalog_reload_hook(_sync_in_memory_catalog)


def reload_model_catalog(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Reload `model_catalog.json` into `MODEL_CATALOG` in-place and return the validated entries."""
    loaded = load_model_catalog(path)
    MODEL_CATALOG[:] = loaded
    reset_probation_counters()
    return MODEL_CATALOG


TIER_TO_RANK = {"LOW": 1, "MID": 2, "HIGH": 3}
EFFORT_TO_RANK = {"low": 1, "medium": 2, "high": 3}


def _model_series_id(model_id: str) -> str:
    """Strip a trailing `-(low|medium|high|xhigh)` suffix to identify a model's variant series."""
    return re.sub(r"-(low|medium|high|xhigh)$", "", str(model_id or "").strip().lower())


def resolve_effort_variant(
    model_entry: Dict[str, Any],
    desired_effort: str,
    floor_tier: str,
    catalog: Optional[List[Dict[str, Any]]] = None,
    allow_non_default_high: bool = False,
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """
    Map `desired_effort` to the matching `MODEL_CATALOG` variant of the same model series
    (e.g. `gemini-3.8-flash-high -> gemini-3.8-flash-medium -> gemini-3.8-flash-low`,
    `gemini-4-argon-medium -> gemini-4-argon-low`, `gpt-6.1-sol-medium -> gpt-6.1-sol-low`,
    `claude-opus-5-5-medium -> claude-opus-5-5-low` if present) and return
    `(resolved_model_entry, effective_effort, clamp_reason)`:
    - Never selects a variant whose `tier_rank` is below `floor_tier` (`TIER_TO_RANK[floor_tier]`).
    - When the requested effort variant is below the package capability floor (or absent for an
      AGY suffix-encoded series), keeps the lowest floor-satisfying variant in the series and
      records `clamp_reason` explaining why.
    """
    cat = catalog if catalog is not None else MODEL_CATALOG
    target_eff = str(desired_effort or model_entry.get("effort") or "medium").lower().strip()
    if target_eff not in EFFORT_TO_RANK:
        target_eff = str(model_entry.get("effort") or "medium").lower().strip()
    f_tier = str(floor_tier or "LOW").upper().strip()
    floor_rank = TIER_TO_RANK.get(f_tier, 1)

    prov = str(model_entry.get("provider") or "").lower().strip()
    pool_id = model_entry.get("pool_id") or model_to_pool_id(model_entry)
    base_series = _model_series_id(model_entry.get("model_id", ""))
    entry_status = str(model_entry.get("status") or "active").lower().strip()

    series_variants = [
        v
        for v in cat
        if str(v.get("provider") or "").lower().strip() == prov
        and (v.get("pool_id") or model_to_pool_id(v)) == pool_id
        and _model_series_id(v.get("model_id", "")) == base_series
        and str(v.get("status") or "active").lower().strip() == entry_status
    ]
    if not series_variants:
        series_variants = [model_entry]

    def _variant_allowed(v: Dict[str, Any]) -> bool:
        if v.get("model_id") == model_entry.get("model_id"):
            return True
        if (
            str(v.get("effort") or "").lower().strip() == "high"
            and not v.get("default_preferred", True)
            and not (allow_non_default_high or target_eff == "high")
        ):
            return False
        return True

    floor_ok = [
        v
        for v in series_variants
        if int(v.get("tier_rank", TIER_TO_RANK.get(str(v.get("tier", "LOW")).upper(), 1))) >= floor_rank
        and _variant_allowed(v)
    ]
    if not floor_ok:
        floor_ok = [model_entry]

    exact_floor_ok = [
        v for v in floor_ok if str(v.get("effort") or "medium").lower().strip() == target_eff
    ]
    if exact_floor_ok:
        exact_floor_ok.sort(
            key=lambda v: (
                0 if v.get("model_id") == model_entry.get("model_id") else 1,
                abs(int(v.get("tier_rank", floor_rank)) - int(model_entry.get("tier_rank", floor_rank))),
                int(v.get("tier_rank", floor_rank)),
            )
        )
        chosen = exact_floor_ok[0]
        return chosen, target_eff, None

    below_floor_matches = [
        v
        for v in series_variants
        if str(v.get("effort") or "medium").lower().strip() == target_eff
        and int(v.get("tier_rank", TIER_TO_RANK.get(str(v.get("tier", "LOW")).upper(), 1))) < floor_rank
    ]

    floor_ok.sort(
        key=lambda v: (
            EFFORT_TO_RANK.get(str(v.get("effort") or "medium").lower().strip(), 2),
            int(v.get("tier_rank", floor_rank)),
            0 if v.get("model_id") == model_entry.get("model_id") else 1,
        )
    )
    chosen = floor_ok[0]
    chosen_eff = str(chosen.get("effort") or "medium").lower().strip()

    if below_floor_matches:
        low_v = below_floor_matches[0]
        clamp_reason = (
            f"Floor guard: kept {chosen['model_id']} (tier={chosen['tier']}, effort={chosen_eff}) "
            f"because {low_v['model_id']} (effort={target_eff}) has tier {low_v['tier']} < capability floor {f_tier}"
        )
        return chosen, chosen_eff, clamp_reason

    if prov == "agy" or len(series_variants) > 1:
        clamp_reason = (
            f"Catalog variant guard: kept {chosen['model_id']} (tier={chosen['tier']}, effort={chosen_eff}) "
            f"because no {target_eff}-effort variant >= {f_tier} exists for {base_series} on {prov}"
        )
        return chosen, chosen_eff, clamp_reason

    return chosen, target_eff, None

CONTEXT_NEED_TOKENS = {
    "pointer": 32_000,
    "module": 128_000,
    "broad": 250_000,
}



def compute_capability_floor(
    kind: str = "implement",
    ambiguity: str = "low",
    blast_radius: str = "local",
    dependency_role: str = "leaf",
    context_need: str = "pointer",
    critical: bool = False,
    escalation_reason: Optional[str] = None,
    explicit_effort: Optional[str] = None,
    band: Optional[str] = None,
    sigma: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Determine `(capability_floor, reasoning_effort)` according to `~/.agents/ROUTING.md` (2026-10-18),
    with optional sigma-band effort adjustment from `allocation.py` (`critical`, `economy`, `normal`, `surge`).
    """
    k = kind.lower().strip()
    amb = ambiguity.lower().strip()
    br = blast_radius.lower().strip()
    dep = dependency_role.lower().strip()
    exp_eff = (explicit_effort or "").lower().strip()

    is_high_floor = (
        br in {"cross-module", "irreversible"}
        or amb == "high"
        or dep == "upstream"
        or k == "design"
        or (k == "review" and (amb != "low" or br != "local"))
    )

    if is_high_floor:
        floor_tier = "HIGH"
        if exp_eff in {"low", "medium", "high"}:
            base_effort = exp_eff
        elif critical or (escalation_reason and escalation_reason.strip()):
            base_effort = "high"
        elif amb == "low" and br in {"local", "module"} and k not in {"design", "review"}:
            base_effort = "low"
        else:
            base_effort = "medium"
    elif k in {"scan", "extract", "mechanical", "docs"} and amb == "low" and br == "local":
        floor_tier = "LOW"
        base_effort = exp_eff if exp_eff in {"low", "medium", "high"} else "low"
    else:
        floor_tier = "MID"
        if exp_eff in {"low", "medium", "high"}:
            base_effort = exp_eff
        elif critical or (escalation_reason and escalation_reason.strip()):
            base_effort = "high"
        else:
            base_effort = "medium"

    eff_band = band or (classify_sigma_band(sigma)["band"] if sigma is not None else None)
    if eff_band:
        base_effort = apply_band_effort(
            base_effort,
            eff_band,
            floor_tier=floor_tier,
            kind=k,
            ambiguity=amb,
            blast_radius=br,
            critical=critical,
            escalation_reason=escalation_reason,
            explicit_effort=explicit_effort,
        )
    return floor_tier, base_effort


def _is_snapshot_stale(p_data: Dict[str, Any], max_age_sec: int = QUOTA_STALE_SECONDS) -> bool:
    if bool(p_data.get("stale")) or str(p_data.get("confidence", "")).upper() == "STALE":
        return True
    ts_str = p_data.get("updated_at") or p_data.get("timestamp")
    if not ts_str or not isinstance(ts_str, str):
        return False
    try:
        ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - ts).total_seconds() > max_age_sec
    except Exception:
        return True


def _get_model_quota(
    model_entry: Dict[str, Any],
    providers_quota: Dict[str, Any],
) -> Tuple[Optional[float], int, str]:
    """
    Return `(remaining_fraction_or_None, reset_in_sec, quota_state)` for a catalog model.
    Evaluates both 5-hour and weekly windows (`effective_remaining = min(5h, weekly)`).
    """
    prov = model_entry["provider"]
    if not isinstance(providers_quota, dict) or prov not in providers_quota:
        return None, 3600, "UNKNOWN"

    p_data = providers_quota.get(prov)
    if not isinstance(p_data, dict) or not p_data:
        return None, 3600, "UNKNOWN"

    raw_status = str(p_data.get("status") or "").lower().strip()
    conf = str(p_data.get("confidence") or "").upper().strip()
    if raw_status in {"offline", "unknown", "error", "unreachable"} or conf in {"UNAVAILABLE", "UNKNOWN"}:
        return None, _first_int(p_data.get("reset_in_sec"), default=3600), "UNKNOWN"

    if _is_snapshot_stale(p_data):
        return None, _first_int(p_data.get("reset_in_sec"), default=3600), "UNKNOWN"

    if bool(p_data.get("rate_limited")) or bool(p_data.get("try_again_at")):
        return 0.0, _first_int(p_data.get("reset_in_sec"), default=3600), "exhausted"

    q_key = model_entry["quota_key"]
    pool_windows = model_entry.get("pool_windows", ("five_hour", "weekly"))
    windows = p_data.get("windows")

    window_remaining: List[float] = []
    window_resets: List[int] = []
    if isinstance(windows, dict):
        for w_name in pool_windows:
            w_obj = windows.get(w_name)
            if isinstance(w_obj, dict):
                if w_obj.get("remaining_fraction") is not None:
                    try:
                        window_remaining.append(float(w_obj["remaining_fraction"]))
                    except (TypeError, ValueError):
                        return None, 3600, "UNKNOWN"
                elif w_obj.get("used_percent") is not None:
                    try:
                        used_pct = float(w_obj["used_percent"])
                        window_remaining.append(max(0.0, min(1.0, 1.0 - (used_pct / 100.0))))
                    except (TypeError, ValueError):
                        return None, 3600, "UNKNOWN"
                if w_obj.get("reset_in_sec") is not None:
                    try:
                        window_resets.append(int(w_obj["reset_in_sec"]))
                    except (TypeError, ValueError):
                        pass

    raw_val = p_data.get(q_key)
    top_rem: Optional[float] = None
    if raw_val is not None:
        try:
            top_rem = float(raw_val)
        except (TypeError, ValueError):
            return None, 3600, "UNKNOWN"

    if window_remaining and top_rem is not None:
        rem = min(min(window_remaining), top_rem)
    elif window_remaining:
        rem = min(window_remaining)
    elif top_rem is not None:
        rem = top_rem
    else:
        return None, _first_int(p_data.get("reset_in_sec"), default=3600), "UNKNOWN"

    reset_sec = min(window_resets) if window_resets else _first_int(p_data.get("reset_in_sec"), default=3600)
    if rem <= 0.05:
        state = "exhausted"
    elif rem < 0.20:
        state = "warning"
    else:
        state = raw_status if raw_status in {"healthy", "warning", "exhausted"} else "healthy"
    return round(rem, 4), reset_sec, state


_TOOL_CATALOG_CACHE: Dict[str, Any] = {}


def _load_tool_catalog():
    """The optional tool catalog module from hub_config (re-imported when the file changes), or None."""
    path = tool_catalog_path()
    if path is None:
        return None
    key = f"{path}:{path.stat().st_mtime_ns}"
    if _TOOL_CATALOG_CACHE.get("key") != key:
        spec = importlib.util.spec_from_file_location("quota_hub_tool_catalog", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TOOL_CATALOG_CACHE.update(key=key, module=module)
    return _TOOL_CATALOG_CACHE["module"]


def recommend_task_tools(task_request: Dict[str, Any]) -> Dict[str, Any]:
    """Task-based helper tool selection, delegated to the user's optional tool catalog (see hub_config.py)."""
    catalog = _load_tool_catalog()
    if catalog is not None:
        return catalog.recommend(task_request)
    return {
        "evaluated": True,
        "selected_tools": [],
        "announcement": None,
        "skipped_blockers": [],
        "closing_line": "Tool use: no tool catalog configured; standard local tools only.",
    }


def select_active_orchestrator(
    preferred: str,
    providers_quota: Dict[str, Any],
    operating_mode: str = "endurance",
) -> Dict[str, Any]:
    """
    Select the active Coordinator / Orchestrator model based on `preferred` ("auto" or model_id)
    and real-time reset-aware quota availability.
    """
    high_candidates = [
        m
        for m in MODEL_CATALOG
        if m["tier"] == "HIGH"
        and m.get("default_preferred", True)
        and str(m.get("status") or "active").lower().strip() == "active"
    ]

    if preferred and preferred != "auto":
        for m in MODEL_CATALOG:
            if (m["model_id"] == preferred or m["provider"] == preferred) and str(
                m.get("status") or "active"
            ).lower().strip() == "active":
                rem, reset_sec, state = _get_model_quota(m, providers_quota)
                if rem is not None and rem > 0.05 and state != "UNKNOWN":
                    return {
                        "preferred": preferred,
                        "active": m["model_id"],
                        "active_provider": m["provider"],
                        "active_label": m["label"],
                        "remaining_quota": rem,
                        "quota_state": state,
                        "operating_mode": operating_mode,
                        "reason": f"User-selected orchestrator ({int(rem * 100)}% quota remaining)",
                    }

    scored: List[Tuple[float, Dict[str, Any], float, str]] = []
    for m in high_candidates:
        rem, reset_sec, state = _get_model_quota(m, providers_quota)
        if rem is None or state == "UNKNOWN" or rem <= 0.05:
            continue
        band_factor = 0.4 if rem < 0.20 else 1.0
        # Slight bonus if quota resets soon and has headroom
        imminence_bonus = 0.06 if reset_sec <= 7200 and rem >= 0.15 else 0.0
        score = ((rem * 0.65) + (m["cost_weight"] * 0.18) + (m["latency_score"] * 0.17) + imminence_bonus) * band_factor
        scored.append((score, m, rem, state))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        best_score, best_m, best_rem, best_state = scored[0]
        return {
            "preferred": preferred or "auto",
            "active": best_m["model_id"],
            "active_provider": best_m["provider"],
            "active_label": best_m["label"],
            "remaining_quota": best_rem,
            "quota_state": best_state,
            "operating_mode": operating_mode,
            "reason": f"Auto-selected highest-headroom HIGH-tier coordinator ({int(best_rem * 100)}% remaining, {operating_mode.upper()})",
        }

    return {
        "preferred": preferred or "auto",
        "active": "gpt-6.1-sol",
        "active_provider": "codex",
        "active_label": "Codex · GPT-6.1-Sol (Fallback)",
        "remaining_quota": 0.0,
        "quota_state": "UNKNOWN",
        "operating_mode": operating_mode,
        "reason": "All HIGH-tier providers near limit or UNKNOWN; defaulting to fallback coordinator",
    }


def build_headless_cli_command(
    model_entry: Dict[str, Any],
    prompt: str,
    workspace: str,
    effort: str = "medium",
    read_only: bool = True,
) -> Tuple[List[str], str]:
    """
    Build the exact headless CLI argv and display string according to `~/.agents/providers/*.md`.
    Claude and Codex receive the prompt on stdin (`prompt_transport_for`), so their argv holds only
    constant tokens; AGY still takes `-p=<prompt>` and is guarded against `.cmd` shims at execution.
    """
    provider = model_entry["provider"]
    cli_model = model_entry["cli_model"]
    if provider == "web":
        safe_p = prompt.replace('"', '\\"')
        return (
            ["mission", "start", prompt, "--repo", workspace, "--judge", f"web:{cli_model}"],
            f'mission start "{safe_p}" --repo "{workspace}" --judge web:{cli_model}',
        )
    safe_prompt = prompt.replace('"', '\\"')
    preview = safe_prompt if len(safe_prompt) <= 80 else safe_prompt[:77] + "..."

    if provider == "agy":
        agy_bin = shutil.which("agy") or "agy"
        argv = [
            agy_bin,
            "--add-dir",
            workspace,
            "--model",
            cli_model,
            "--print-timeout",
            "4m",
            f"-p={prompt}",
        ]
        display = f'agy --add-dir "{workspace}" --model {cli_model} --print-timeout 4m -p="{safe_prompt}"'
        return argv, display

    if provider == "claude":
        claude_bin = shutil.which("claude") or "claude"
        argv = [
            claude_bin,
            "-p",
            "--model",
            cli_model,
            "--effort",
            effort,
            "--output-format",
            "json",
        ]
        display = f'claude -p --model {cli_model} --effort {effort} --output-format json  # stdin: "{preview}"'
        return argv, display

    codex_bin = shutil.which("codex") or "codex"
    sandbox_mode = "read-only" if read_only else "workspace-write"
    argv = [
        codex_bin,
        "exec",
        "-",
        "-m",
        cli_model,
        "-c",
        f"model_reasoning_effort={effort}",
        "-s",
        sandbox_mode,
        "--skip-git-repo-check",
        "--ephemeral",
    ]
    display = (
        f"codex exec - -m {cli_model} "
        f"-c model_reasoning_effort={effort} -s {sandbox_mode} --skip-git-repo-check --ephemeral"
        f'  # stdin: "{preview}"'
    )
    return argv, display


def prompt_transport_for(provider: str) -> str:
    """How the worker prompt reaches the CLI: `stdin` for Claude/Codex, `argv` for AGY (`-p=`)."""
    return "stdin" if provider in {"claude", "codex"} else "argv"


# cmd.exe metacharacters: a `.cmd`/`.bat` shim re-parses argv through cmd.exe, so these are unsafe there.
_CMD_SHIM_UNSAFE = re.compile(r'[&|<>^%!"\r\n]')
_RATE_LIMIT_PATTERNS = re.compile(
    r"rate[ _-]?limit|too many requests|\b429\b|quota exceeded|usage limit|try again at",
    re.IGNORECASE,
)


def detect_rate_limit(returncode: Optional[int], stdout: str, stderr: str) -> bool:
    """
    Structured 429 / usage-limit detection: a Claude JSON result with `is_error` and a limit message,
    or a failed exit whose last 2 KB of stderr carries a limit message. Never matches plain stdout text.
    """
    try:
        payload = json.loads(stdout) if stdout and stdout.lstrip().startswith("{") else None
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("is_error"):
        detail = " ".join(str(payload.get(k) or "") for k in ("result", "subtype", "error"))
        if _RATE_LIMIT_PATTERNS.search(detail):
            return True
    if returncode not in (0, None) and _RATE_LIMIT_PATTERNS.search((stderr or "")[-2048:]):
        return True
    return False


def route_work_package(
    task_Request: Dict[str, Any],
    providers_quota: Dict[str, Any],
    orchestrator_info: Dict[str, Any],
    operating_mode: str = "endurance",
    live_activity: Optional[Dict[str, Any]] = None,
    cost_learner: Optional[QuotaCostLearner] = None,
    preference_weights: Optional[Dict[str, Any]] = None,
    allocation_credits: Optional[Dict[str, float]] = None,
    burn_learner: Optional[BurnRateLearner] = None,
) -> Dict[str, Any]:
    """
    Route a work package using Reset-Aware Mathematical Scheduling (`scheduler.py`) and
    Slider + Burn-Rate-Aware SWRR Allocation (`allocation.py`):
    - Evaluates capability floor & reasoning effort separation, plus sigma-band effort rules.
    - Builds pool states (`agy_gemini`, `agy_claude_gpt`, `claude_shared`, `codex_shared`) across
      both `five_hour` and `weekly` windows.
    - Predicts task quota consumption `(five_hour_upper, weekly_upper)` via `QuotaCostLearner`.
    - Enforces dual-window feasibility (a job is rejected if EITHER 5h OR weekly window cannot
      fit the predicted upper cost).
    - Exploits soon-to-expire / stranded quota before reset (Critical Scenario: 10% weekly resetting
      in 2h is preferred over 80% weekly resetting in 6d when the task fits safely).
    - Protects Codex orchestrator reserve `R_orch(codex)` from routine leaf coding tasks when
      AGY or Claude can serve the task.
    - Selects among feasible providers using Smooth Weighted Round Robin (`step_swrr`) weighted
      by user preference sliders `w_p` and sustainable pool capacity `C(q)`.
    """
    global _LAST_COST_LEARNER_ID, _CREDITS_EXPLICITLY_SEEDED
    learner = cost_learner or COST_LEARNER
    b_learner = burn_learner or BURN_LEARNER
    if cost_learner is not None and task_Request.get("_preference_generation") is None:
        with _CREDIT_LOCK:
            if id(cost_learner) != _LAST_COST_LEARNER_ID:
                _LAST_COST_LEARNER_ID = id(cost_learner)
                if not _CREDITS_EXPLICITLY_SEEDED:
                    reset_shared_credits(bump_generation=False)
    mode = str(task_Request.get("operating_mode") or operating_mode or "endurance").lower().strip()
    if mode not in {"sprint", "endurance"}:
        mode = "endurance"

    raw_weights = (
        preference_weights
        if preference_weights is not None
        else (task_Request.get("preference_weights") or task_Request.get("sliders"))
    )
    if raw_weights is not None:
        weights = coerce_preference_weights(raw_weights)
        if weights != get_preference_weights():
            set_preference_weights(weights, reset_swrr=True)
    else:
        weights = get_preference_weights()

    task_title = str(task_Request.get("task") or task_Request.get("prompt") or "Unnamed Work Package")
    prompt = str(task_Request.get("prompt") or task_title)
    kind = str(task_Request.get("kind", "implement")).lower()
    ambiguity = str(task_Request.get("ambiguity", "low")).lower()
    blast_radius = str(task_Request.get("blast_radius", "local")).lower()
    dependency_role = str(task_Request.get("dependency_role", "leaf")).lower()
    context_need = str(task_Request.get("context_need", "pointer")).lower()
    implementer_family = str(task_Request.get("implementer_family") or "").lower().strip()
    require_cross_family = bool(task_Request.get("require_cross_family", False))
    preferred_provider = str(task_Request.get("preferred_provider") or "").lower().strip()
    explicit_model = str(task_Request.get("explicit_model") or "").strip()
    explicit_effort = str(task_Request.get("explicit_effort") or "").lower().strip() or None
    allow_web_judge: bool = bool(
        task_Request.get("allow_web_judge", False)
        or preferred_provider == "web"
        or explicit_model.lower().startswith("web:")
    )
    critical = bool(task_Request.get("critical", False))
    escalation_reason = task_Request.get("escalation_reason")
    unfinished_tasks_count = int(task_Request.get("unfinished_tasks_count") or 4)
    suitable_queued_tasks_count = task_Request.get("suitable_queued_tasks_count")
    pending_reviews_count = int(task_Request.get("pending_reviews_count") or 2)
    excluded_providers = {
        str(p).lower().strip() for p in (task_Request.get("excluded_providers") or []) if p
    }
    excluded_families = {
        str(f).lower().strip() for f in (task_Request.get("excluded_families") or []) if f
    }
    read_only = bool(task_Request.get("read_only", kind in {"scan", "extract", "review", "design"}))
    raw_scope = task_Request.get("edit_scope")
    if isinstance(raw_scope, list):
        edit_scope = list(raw_scope)
    else:
        edit_scope = [] if read_only else ["src/"]
    workspace = str(task_Request.get("workspace") or Path.cwd())

    floor_tier, effort = compute_capability_floor(
        kind=kind,
        ambiguity=ambiguity,
        blast_radius=blast_radius,
        dependency_role=dependency_role,
        context_need=context_need,
        critical=critical,
        escalation_reason=escalation_reason,
        explicit_effort=explicit_effort,
    )
    floor_rank = TIER_TO_RANK[floor_tier]
    required_tokens = CONTEXT_NEED_TOKENS.get(context_need, 32_000)

    tool_selection = recommend_task_tools(task_Request)

    effective_suitable_queued = (
        int(suitable_queued_tasks_count)
        if suitable_queued_tasks_count is not None
        else (0 if ("codex" in excluded_providers or "openai" in excluded_families) else max(1, unfinished_tasks_count))
    )

    codex_reserve = compute_dynamic_codex_reserve(
        providers_quota,
        operating_mode=mode,
        unfinished_tasks_count=unfinished_tasks_count,
        pending_reviews_count=pending_reviews_count,
        live_activity=live_activity,
        suitable_queued_tasks_count=effective_suitable_queued,
    )
    active_policy = load_orchestrator_policy()
    pool_states = build_pool_states(
        providers_quota,
        live_activity=live_activity,
        cost_learner=learner,
        operating_mode=mode,
        codex_reserve=codex_reserve,
        burn_learner=b_learner,
        preference_weights=weights,
        policy=active_policy,
    )
    allocation = compute_allocation(
        pool_states,
        burn_learner=b_learner,
        cost_learner=learner,
        sliders=weights,
        operating_mode=mode,
        policy=active_policy,
    )
    provider_capacities = {
        prov: float((allocation.get("providers") or {}).get(prov, {}).get("capacity_tasks_per_hour", 0.0))
        for prov in ("agy", "codex", "claude")
    }
    target_shares = {
        prov: float((allocation.get("providers") or {}).get(prov, {}).get("target_share", 0.0))
        for prov in ("agy", "codex", "claude")
    }

    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    exp_lower = explicit_model.lower().strip()
    is_web_explicit_requested = bool(
        exp_lower
        and (
            exp_lower.startswith("web:")
            or any(
                exp_lower in {s["model_id"].lower(), f"web:{s['surface']}"}
                for s in WEB_JUDGE_SEATS
            )
        )
    )

    for m in MODEL_CATALOG:
        pool_id = m.get("pool_id") or model_to_pool_id(m)
        if is_web_explicit_requested:
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": f"Explicit model {explicit_model} requested a web judge seat",
                }
            )
            continue

        rem, reset_sec, quota_state = _get_model_quota(m, providers_quota)
        pool_state = pool_states.get(pool_id) or {}
        prov_band = str(pool_state.get("allocation_band") or "normal")
        pool_band = str(pool_state.get("pool_band") or prov_band)
        if pool_band in {"saver", "critical", "economy"}:
            eff_band = pool_band
        elif prov_band in {"saver", "critical", "economy"}:
            eff_band = prov_band
        else:
            eff_band = "normal"

        cand_target_effort = apply_band_effort(
            effort,
            eff_band,
            floor_tier=floor_tier,
            kind=kind,
            ambiguity=ambiguity,
            blast_radius=blast_radius,
            critical=critical,
            escalation_reason=escalation_reason,
            explicit_effort=explicit_effort,
        )

        is_explicit_match = bool(
            explicit_model and explicit_model.lower() in {m["model_id"].lower(), m["cli_model"].lower()}
        )

        # -1. Catalog lifecycle status gate (active | probation | retired)
        m_status = str(m.get("status") or "active").lower().strip()
        if m_status == "retired":
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": "Model status is retired",
                }
            )
            continue

        if m_status == "probation" and not is_probation_eligible(floor_tier, kind):
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": f"Probation model not eligible for floor={floor_tier}, kind={kind}",
                }
            )
            continue

        # 0. Explicit provider / family exclusions
        if m["provider"] in excluded_providers or m["family"] in excluded_families:
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": f"Excluded by user/provider rule ({m['provider']}/{m['family']})",
                }
            )
            continue

        # 1. Capability Floor & Default Preference Gate
        if not is_explicit_match:
            if not m.get("default_preferred", True) and not (critical or escalation_reason):
                rejected.append(
                    {
                        "model_id": m["model_id"],
                        "pool_id": pool_id,
                        "reason": "Non-default high-effort variant (requires explicit request or critical escalation)",
                    }
                )
                continue

            if m["tier_rank"] < floor_rank:
                rejected.append(
                    {
                        "model_id": m["model_id"],
                        "pool_id": pool_id,
                        "reason": f"Tier {m['tier']} < Floor {floor_tier}",
                    }
                )
                continue

        # 2. Context Fit Gate
        if m["context_window"] < required_tokens:
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": f"Context {m['context_window']} < {required_tokens}",
                }
            )
            continue

        # 3. Cross-Family Review Gate: hard only when the task requires it; otherwise the scheduler's
        #    soft independence bonus prefers another family while same-family review stays routable.
        if kind == "review" and implementer_family and require_cross_family and m["family"] == implementer_family:
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": f"Cross-family review required: reviewer family ({m['family']}) == implementer ({implementer_family})",
                }
            )
            continue

        # 4. AGY Gemini Argon Bulk Coding Restriction
        if (
            not is_explicit_match
            and m["model_id"].startswith("gemini-4-argon")
            and kind == "implement"
            and floor_tier != "HIGH"
        ):
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": "Argon reserved for HIGH reasoning/design/review; Flash preferred for bounded coding",
                }
            )
            continue

        # 5. Predict task quota cost on this pool & evaluate Reset-Aware Mathematical Feasibility
        cost_pred = learner.predict_cost(pool_id, kind=kind, tier=m["tier"], complexity=ambiguity)
        feasible, total_score, reasons, rej_reason, math_breakdown = evaluate_candidate_schedule_score(
            model_entry=m,
            pool_state=pool_state,
            cost_pred=cost_pred,
            floor_tier=floor_tier,
            floor_rank=floor_rank,
            target_effort=cand_target_effort,
            required_tokens=required_tokens,
            kind=kind,
            dependency_role=dependency_role,
            implementer_family=implementer_family,
            preferred_provider=preferred_provider,
            is_explicit_match=is_explicit_match,
            operating_mode=mode,
        )

        if not feasible:
            rejected.append(
                {
                    "model_id": m["model_id"],
                    "pool_id": pool_id,
                    "reason": rej_reason or f"Quota exhausted ({int((rem or 0) * 100)}% remaining)",
                }
            )
            continue

        candidates.append(
            {
                "model": m,
                "pool_id": pool_id,
                "score": total_score,
                "remaining_quota": rem,
                "quota_state": quota_state,
                "reset_in_sec": reset_sec,
                "cost_prediction": cost_pred,
                "math_breakdown": math_breakdown,
                "reasons": reasons,
                "is_explicit_match": is_explicit_match,
                "effective_band": eff_band,
                "band_effort": cand_target_effort,
            }
        )

    if allow_web_judge and kind == "review":
        cli_review_candidates_low = not candidates or all(
            str(c.get("effective_band") or "normal") in {"saver", "critical"}
            or (c.get("remaining_quota") is not None and float(c.get("remaining_quota") or 0.0) <= 0.15)
            for c in candidates
        )
        max_cli_score = max((float(c.get("score") or 0.0) for c in candidates), default=0.80)
        exp_lower = explicit_model.lower().strip()
        for seat in WEB_JUDGE_SEATS:
            seat_pool_id = seat["pool_id"]
            seat_surface = seat["surface"]
            is_web_explicit = bool(
                exp_lower and exp_lower in {seat["model_id"].lower(), f"web:{seat_surface}", seat_surface}
            )
            if is_web_explicit_requested and not is_web_explicit:
                rejected.append(
                    {
                        "model_id": seat["model_id"],
                        "pool_id": seat_pool_id,
                        "reason": f"Explicit model {explicit_model} requested another web judge seat",
                    }
                )
                continue
            if seat["provider"] in excluded_providers or seat["family"] in excluded_families:
                rejected.append(
                    {
                        "model_id": seat["model_id"],
                        "pool_id": seat_pool_id,
                        "reason": f"Excluded by user/provider rule ({seat['provider']}/{seat['family']})",
                    }
                )
                continue
            if seat["context_window"] < required_tokens:
                rejected.append(
                    {
                        "model_id": seat["model_id"],
                        "pool_id": seat_pool_id,
                        "reason": f"Context {seat['context_window']} < {required_tokens}",
                    }
                )
                continue
            if implementer_family and require_cross_family and seat["family"] == implementer_family:
                rejected.append(
                    {
                        "model_id": seat["model_id"],
                        "pool_id": seat_pool_id,
                        "reason": f"Cross-family review required: reviewer family ({seat['family']}) == implementer ({implementer_family})",
                    }
                )
                continue
            with _WEB_JUDGE_LOCK:
                surf_state = dict(_WEB_JUDGE_STATE.get(seat_surface) or {})
            if bool(surf_state.get("rate_limited")):
                rejected.append(
                    {
                        "model_id": seat["model_id"],
                        "pool_id": seat_pool_id,
                        "reason": f"Web surface {seat['surface']} reported SURFACE_LIMIT_REACHED (status=exhausted)",
                    }
                )
                continue

            cross_fam_bonus = 0.08 if (implementer_family and seat["family"] != implementer_family) else 0.0
            if is_web_explicit or preferred_provider == "web":
                web_score = round(max(1.50, max_cli_score + 0.35) + cross_fam_bonus + float(seat["latency_score"]) * 0.05, 4)
            elif cli_review_candidates_low:
                web_score = round(max(1.15, max_cli_score + 0.20) + cross_fam_bonus + float(seat["latency_score"]) * 0.05, 4)
            else:
                web_score = round(0.88 + cross_fam_bonus + float(seat["latency_score"]) * 0.05, 4)

            web_reasons = [
                "Zero-API-cost web judge seat (quota=UNKNOWN, never 0%)",
                f"surface={seat_surface} ({seat['family']} family)",
            ]
            if implementer_family and seat["family"] != implementer_family:
                web_reasons.append(f"Cross-family independent review ({seat['family']} != {implementer_family})")
            if is_web_explicit or preferred_provider == "web":
                web_reasons.append("Preferred/explicit web judge seat")
            elif cli_review_candidates_low:
                web_reasons.append("CLI review pools exhausted/saver -> zero-API-cost web judge fallback")

            web_cost_pred = {
                "five_hour_mean": 0.0,
                "five_hour_upper": 0.0,
                "weekly_mean": 0.0,
                "weekly_upper": 0.0,
                "five_hour_pct_points": 0.0,
                "weekly_pct_points": 0.0,
                "zero_api_cost": True,
                "predicted_cost": 0.0,
                "prediction_source": "web_zero_api_cost",
            }
            web_math = {
                "zero_api_cost": True,
                "quota_state": "UNKNOWN",
                "predicted_5h_cost": 0.0,
                "predicted_weekly_cost": 0.0,
                "cross_family_bonus": cross_fam_bonus,
                "total_score": web_score,
            }
            candidates.append(
                {
                    "model": dict(seat),
                    "pool_id": seat_pool_id,
                    "score": web_score,
                    "remaining_quota": None,
                    "quota_state": "UNKNOWN",
                    "reset_in_sec": None,
                    "cost_prediction": web_cost_pred,
                    "math_breakdown": web_math,
                    "reasons": web_reasons,
                    "is_explicit_match": bool(
                        is_web_explicit or preferred_provider == "web" or cli_review_candidates_low
                    ),
                    "effective_band": "normal",
                    "band_effort": explicit_effort or seat["effort"],
                }
            )
    elif allow_web_judge and kind != "review":
        for seat in WEB_JUDGE_SEATS:
            rejected.append(
                {
                    "model_id": seat["model_id"],
                    "pool_id": seat["pool_id"],
                    "reason": f"Web judge seats are review/judge only (kind={kind!r} != 'review')",
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        all_resets = [
            int(p.get("reset_in_sec", 3600))
            for p in (providers_quota or {}).values()
            if isinstance(p, dict) and p.get("reset_in_sec") is not None
        ]
        earliest_reset = min(all_resets) if all_resets else 3600
        return {
            "status": "paused_quota_exhausted",
            "operating_mode": mode,
            "capability_floor": floor_tier,
            "reasoning_effort": effort,
            "earliest_reset_in_sec": earliest_reset,
            "assignment_record": None,
            "tool_selection": tool_selection,
            "codex_reserve": codex_reserve,
            "pool_states": pool_states,
            "allocation": allocation,
            "allocation_credits": get_allocation_credits(),
            "web_judge_seats": get_web_judge_status(),
            "rejected_models": rejected,
            "message": f"No eligible model >= {floor_tier} can safely fit predicted task cost before reset. Checkpoint & resume in {earliest_reset}s.",
        }

    # Provider selection:
    # 1. Explicit model request wins directly.
    # 2. Explicit preferred_provider request wins when feasible and not disabled by slider 0.
    # 3. Otherwise, Smooth Weighted Round Robin (step_swrr) selects chosen_provider = argmax C_p across
    #    feasible positive-share providers, and within chosen_provider we pick the highest-scoring candidate
    #    (where pre-reset / banked-reset harvest bonuses act within candidate ranking per Contract §3).
    explicit_cands = [c for c in candidates if c.get("is_explicit_match")]
    active_candidates = [
        c for c in candidates if str(c["model"].get("status") or "active").lower().strip() == "active"
    ]
    probation_by_provider: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        if str(c["model"].get("status") or "active").lower().strip() == "probation":
            probation_by_provider.setdefault(c["model"]["provider"], []).append(c)

    best_by_provider: Dict[str, Dict[str, Any]] = {}
    best_by_pool: Dict[str, Dict[str, Any]] = {}
    for c in active_candidates:
        prov = c["model"]["provider"]
        if prov not in best_by_provider:
            best_by_provider[prov] = c
        pid = c.get("pool_id")
        if pid and pid not in best_by_pool:
            best_by_pool[pid] = c
    for c in candidates:
        prov = c["model"]["provider"]
        if prov not in best_by_provider:
            best_by_provider[prov] = c
        pid = c.get("pool_id")
        if pid and pid not in best_by_pool:
            best_by_pool[pid] = c

    for pid, ps_dict in pool_states.items():
        if not isinstance(ps_dict, dict):
            continue
        c_pool = best_by_pool.get(pid)
        if c_pool is None:
            ps_dict["tasks_fittable_before_weekly_reset"] = 0.0
            ps_dict["tasks_fittable_before_5h_reset"] = 0.0
            if isinstance(ps_dict.get("weekly"), dict):
                ps_dict["weekly"]["tasks_fittable_before_weekly_reset"] = 0.0
            if isinstance(ps_dict.get("five_hour"), dict):
                ps_dict["five_hour"]["tasks_fittable_before_5h_reset"] = 0.0
            continue

        mb = c_pool.get("math_breakdown") if isinstance(c_pool.get("math_breakdown"), dict) else {}
        cp = c_pool.get("cost_prediction") if isinstance(c_pool.get("cost_prediction"), dict) else {}

        c5_mean_raw = mb.get("predicted_5h_cost", cp.get("five_hour_mean"))
        cw_mean_raw = mb.get("predicted_weekly_cost", cp.get("weekly_mean"))
        u5_raw = mb.get("usable_5h")
        if u5_raw is None and isinstance(ps_dict.get("five_hour"), dict):
            u5_raw = ps_dict["five_hour"].get("usable_before_reset")
        uw_raw = mb.get("usable_weekly")
        if uw_raw is None and isinstance(ps_dict.get("weekly"), dict):
            uw_raw = ps_dict["weekly"].get("usable_before_reset")

        try:
            c5_val = max(0.005, float(c5_mean_raw)) if c5_mean_raw is not None else None
        except (TypeError, ValueError):
            c5_val = None
        try:
            cw_val = max(0.002, float(cw_mean_raw)) if cw_mean_raw is not None else None
        except (TypeError, ValueError):
            cw_val = None

        fittable_wk = mb.get("tasks_fittable_before_weekly_reset")
        if fittable_wk is not None:
            try:
                fw_val = max(0.0, float(fittable_wk))
            except (TypeError, ValueError):
                fw_val = 0.0
        elif uw_raw is not None and cw_val is not None:
            try:
                fw_val = round(max(0.0, float(uw_raw)) / cw_val, 4)
            except (TypeError, ValueError):
                fw_val = 0.0
        else:
            fw_val = 0.0

        fittable_5h = mb.get("tasks_fittable_before_5h_reset")
        if fittable_5h is not None:
            try:
                f5_val = max(0.0, float(fittable_5h))
            except (TypeError, ValueError):
                f5_val = 0.0
        elif u5_raw is not None and c5_val is not None:
            try:
                f5_val = round(max(0.0, float(u5_raw)) / c5_val, 4)
            except (TypeError, ValueError):
                f5_val = 0.0
        else:
            f5_val = 0.0

        ps_dict["tasks_fittable_before_weekly_reset"] = fw_val
        ps_dict["tasks_fittable_before_5h_reset"] = f5_val
        if cw_val is not None:
            ps_dict["predicted_weekly_cost"] = cw_val
        if c5_val is not None:
            ps_dict["predicted_5h_cost"] = c5_val
        if isinstance(ps_dict.get("weekly"), dict):
            ps_dict["weekly"]["tasks_fittable_before_weekly_reset"] = fw_val
            if cw_val is not None:
                ps_dict["weekly"]["predicted_weekly_cost"] = cw_val
        if isinstance(ps_dict.get("five_hour"), dict):
            ps_dict["five_hour"]["tasks_fittable_before_5h_reset"] = f5_val
            if c5_val is not None:
                ps_dict["five_hour"]["predicted_5h_cost"] = c5_val

    feasible_providers = list(best_by_provider.keys())
    tie_scores = {prov: float(best_by_provider[prov]["score"]) for prov in feasible_providers}
    req_gen = task_Request.get("_preference_generation")

    with _CREDIT_LOCK:
        live_weights = get_preference_weights()
        live_shares_exact = compute_target_shares(live_weights, provider_capacities)
        weights = dict(live_weights)
        target_shares = {prov: round(float(live_shares_exact.get(prov, 0.0)), 4) for prov in ("agy", "codex", "claude")}
        if isinstance(allocation, dict):
            allocation["sliders"] = dict(live_weights)
            allocation["preference_weights"] = dict(live_weights)
            alloc_provs = allocation.get("providers")
            if isinstance(alloc_provs, dict):
                for prov in ("agy", "codex", "claude"):
                    if isinstance(alloc_provs.get(prov), dict):
                        alloc_provs[prov]["slider"] = int(live_weights.get(prov, 0))
                        alloc_provs[prov]["target_share"] = target_shares[prov]

        pref_prov_active_cands = [
            c
            for c in active_candidates
            if preferred_provider
            and c["model"]["provider"] == preferred_provider
            and (
                preferred_provider == "web"
                or (int(live_weights.get(preferred_provider, 0)) > 0 and float(live_shares_exact.get(preferred_provider, 0.0)) > 0.0)
            )
        ]
        pref_prov_cands = pref_prov_active_cands or [
            c
            for c in candidates
            if preferred_provider
            and c["model"]["provider"] == preferred_provider
            and (
                preferred_provider == "web"
                or (int(live_weights.get(preferred_provider, 0)) > 0 and float(live_shares_exact.get(preferred_provider, 0.0)) > 0.0)
            )
        ]

        # One global probation-eligible tick per routing decision, regardless of the winning provider.
        tick_probation_eligible(floor_tier, kind)
        chosen_prov_for_probation: Optional[str] = None
        if explicit_cands:
            winner = explicit_cands[0]
            credits_snapshot_out = get_allocation_credits()
            # An explicit model request is a live owner instruction: a probation model is honoured even when
            # its share is exhausted, but the use is always counted ("explicit request overrides probation share").
            if str(winner["model"].get("status") or "active").lower().strip() == "probation":
                _, ex_meta = should_route_probation(
                    winner["model"]["provider"], winner["model"], floor_tier=floor_tier, kind=kind, force=True
                )
                winner["reasons"] = list(winner["reasons"]) + [ex_meta["reason"]]
        else:
            # Eligibility is computed read-only (peek) BEFORE the single SWRR commit: a provider is eligible
            # if it has an ACTIVE candidate, or a probation candidate whose share slot may run now.
            prob_ok = is_probation_eligible(floor_tier, kind)
            eligible_provs: List[str] = []
            for prov in feasible_providers:
                if any(c["model"]["provider"] == prov for c in active_candidates):
                    eligible_provs.append(prov)
                    continue
                pcs = probation_by_provider.get(prov) or []
                if pcs and prob_ok and should_route_probation(
                    prov, pcs[0]["model"], floor_tier=floor_tier, kind=kind, peek=True
                )[0]:
                    eligible_provs.append(prov)

            pref_cands = pref_prov_cands if preferred_provider in eligible_provs else []
            if pref_cands:
                winner = pref_cands[0]
                chosen_prov_for_probation = preferred_provider
                credits_snapshot_out = get_allocation_credits()
            else:
                chosen_prov = select_and_commit(
                    feasible=eligible_provs,
                    shares=live_shares_exact,
                    generation=req_gen,
                    capacities=provider_capacities,
                    tie_break_scores=tie_scores,
                    pool_states=pool_states,
                    operating_mode=mode,
                )
                if req_gen is None:
                    _CREDITS_EXPLICITLY_SEEDED = False
                credits_snapshot_out = get_allocation_credits()

                if chosen_prov is None:
                    web_cands = [c for c in candidates if c["model"]["provider"] == "web"]
                    if web_cands:
                        winner = web_cands[0]
                        chosen_prov_for_probation = "web"
                    else:
                        return {
                            "status": "paused_providers_disabled",
                            "reason": "all feasible providers disabled by preference",
                            "operating_mode": mode,
                            "capability_floor": floor_tier,
                            "reasoning_effort": effort,
                            "assignment_record": None,
                            "tool_selection": tool_selection,
                            "codex_reserve": codex_reserve,
                            "pool_states": pool_states,
                            "allocation": allocation,
                            "allocation_credits": credits_snapshot_out,
                            "web_judge_seats": get_web_judge_status(),
                            "rejected_models": rejected,
                            "message": "all feasible providers disabled by preference",
                        }
                else:
                    chosen_prov_for_probation = chosen_prov
                    winner = best_by_provider[chosen_prov]
                    if chosen_prov in target_shares:
                        winner["reasons"] = list(winner["reasons"]) + [
                            f"SWRR share={round(target_shares.get(chosen_prov, 0.0) * 100, 1)}% (credit={round(credits_snapshot_out.get(chosen_prov, 0.0), 3)})"
                        ]

            # Only now (provider already chosen, credits committed once) consume the probation slot.
            prov_now = chosen_prov_for_probation
            winner_is_prob = str(winner["model"].get("status") or "active").lower().strip() == "probation"
            prov_active = [c for c in active_candidates if c["model"]["provider"] == prov_now]
            prov_probs = probation_by_provider.get(prov_now) or []
            if (winner_is_prob or prov_probs) and prob_ok:
                prob_cand = winner if winner_is_prob else prov_probs[0]
                granted, prob_meta = should_route_probation(
                    prov_now, prob_cand["model"], floor_tier=floor_tier, kind=kind
                )
                if granted:
                    swrr_notes = [r for r in winner.get("reasons", []) if r.startswith("SWRR share=")]
                    winner = prob_cand
                    winner["reasons"] = list(winner["reasons"]) + swrr_notes + [
                        f"Probation slot ({prob_meta['selected']}/{prob_meta['eligible']}, share<={prob_meta['share']:.2f})"
                    ]
                    winner_is_prob = True
                elif winner_is_prob and prov_active:
                    winner = prov_active[0]
                    winner_is_prob = False
            elif winner_is_prob and prov_active:
                winner = prov_active[0]
                winner_is_prob = False
            if winner_is_prob and str(winner["model"].get("status") or "active").lower().strip() == "probation" and not any(
                str(r).startswith("Probation slot") for r in winner["reasons"]
            ):
                # Unreachable unless counters changed between peek and commit; never route probation without a slot.
                winner = prov_active[0] if prov_active else winner

    raw_win_m = winner["model"]
    effort_clamp_reason: Optional[str] = None
    if raw_win_m.get("provider") == "web":
        win_m = raw_win_m
        effective_effort = str(explicit_effort or winner.get("band_effort") or raw_win_m.get("effort") or effort)
    elif winner.get("is_explicit_match") and not explicit_effort:
        win_m = raw_win_m
        effective_effort = raw_win_m["effort"] if raw_win_m["provider"] == "agy" else str(winner.get("band_effort") or effort)
    else:
        desired_effort = explicit_effort or str(winner.get("band_effort") or effort)
        win_m, effective_effort, effort_clamp_reason = resolve_effort_variant(
            raw_win_m,
            desired_effort=desired_effort,
            floor_tier=floor_tier,
            catalog=MODEL_CATALOG,
            allow_non_default_high=bool(critical or escalation_reason or explicit_effort == "high"),
        )
        if win_m["model_id"] != raw_win_m["model_id"]:
            winner["reasons"] = list(winner["reasons"]) + [
                f"Effort variant mapped: {raw_win_m['model_id']} -> {win_m['model_id']} (effort={effective_effort})"
            ]
        if effort_clamp_reason:
            winner["reasons"] = list(winner["reasons"]) + [effort_clamp_reason]

    fallback_entry = None
    for cand in candidates:
        if cand is not winner and cand["pool_id"] != winner["pool_id"]:
            fallback_entry = cand
            break
    if fallback_entry is None:
        for cand in candidates:
            if cand is not winner:
                fallback_entry = cand
                break

    fallback_model_id = fallback_entry["model"]["model_id"] if fallback_entry else "checkpoint-until-reset"

    argv, cli_display = build_headless_cli_command(
        model_entry=win_m,
        prompt=prompt,
        workspace=workspace,
        effort=effective_effort,
        read_only=read_only,
    )

    win_prov_alloc = (allocation.get("providers") or {}).get(win_m["provider"]) or {}

    assignment_record = {
        "task_id": f"WP-{uuid.uuid4().hex[:8].upper()}",
        "task": task_title,
        "operating_mode": mode,
        "coordinator": orchestrator_info.get("active", "auto"),
        "provider": win_m["provider"],
        "pool_id": winner["pool_id"],
        "model": win_m["model_id"],
        "model_label": win_m["label"],
        "family": win_m["family"],
        "capability_floor": floor_tier,
        "selected_tier": win_m["tier"],
        "reasoning_effort": effective_effort,
        "effort_clamp_reason": effort_clamp_reason,
        "allocation_band": winner.get("effective_band", win_prov_alloc.get("band", "normal")),
        "provider_sigma": win_prov_alloc.get("sigma"),
        "target_share": win_prov_alloc.get("target_share", 0.0),
        "remaining_quota": winner["remaining_quota"],
        "quota_state": winner["quota_state"],
        "zero_api_cost": bool(win_m.get("zero_api_cost", False)),
        "predicted_cost": winner["cost_prediction"],
        "math_breakdown": winner["math_breakdown"],
        "routing_score": winner["score"],
        "routing_rationale": " · ".join(winner["reasons"]),
        "context_scope": context_need,
        "context_budget": f"{required_tokens // 1000}k tokens",
        "max_output": "150 words (scout)" if kind in {"scan", "extract"} else "400 words + diff",
        "tools_allowed": "read-only" if read_only else "workspace-write (within edit_scope)",
        "selected_ai_tools": [t["tool"] for t in tool_selection["selected_tools"]],
        "tool_announcement": tool_selection["announcement"],
        "edit_scope": edit_scope,
        "expected_result": "status · confidence · findings · changes · tests · risks · files_touched",
        "fallback_model": fallback_model_id,
        "escalation_condition": "2 failed attempts | quota >=95% used | undeclared dependency | scope exceeded",
        "headless_cli": cli_display,
        "cli_argv": argv,
        "workspace": workspace,
        "read_only": read_only,
        "require_cross_family": require_cross_family,
        "explicit_model": explicit_model or None,
        "explicit_effort": explicit_effort,
        "implementer_family": implementer_family or None,
        "priority": int(task_Request["priority"]) if task_Request.get("priority") is not None else 100,
        "satisfy_on": str(task_Request.get("satisfy_on") or "completed").lower(),
        "prompt_transport": prompt_transport_for(win_m["provider"]),
        "cli_stdin_prompt": prompt if prompt_transport_for(win_m["provider"]) == "stdin" else None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    return {
        "status": "routed",
        "operating_mode": mode,
        "capability_floor": floor_tier,
        "reasoning_effort": effective_effort,
        "assignment_record": assignment_record,
        "tool_selection": tool_selection,
        "codex_reserve": codex_reserve,
        "pool_states": pool_states,
        "allocation": allocation,
        "allocation_credits": credits_snapshot_out,
        "web_judge_seats": get_web_judge_status(),
        "ranked_candidates": [
            {
                "provider": c["model"]["provider"],
                "pool_id": c["pool_id"],
                "model_id": c["model"]["model_id"],
                "label": c["model"]["label"],
                "tier": c["model"]["tier"],
                "family": c["model"]["family"],
                "remaining_quota": c["remaining_quota"],
                "quota_state": c["quota_state"],
                "predicted_5h_pct": c["cost_prediction"]["five_hour_pct_points"],
                "predicted_weekly_pct": c["cost_prediction"]["weekly_pct_points"],
                "math_breakdown": c["math_breakdown"],
                "score": c["score"],
                "reasons": c["reasons"],
            }
            for c in candidates[:6]
        ],
        "rejected_models": rejected,
    }


def execute_headless_dispatch(
    assignment_record: Dict[str, Any],
    dry_run: bool = True,
    timeout_sec: int = 90,
) -> Dict[str, Any]:
    """
    Execute (or dry-run verify) a routed work package via headless CLI (`agy -p`, `claude -p`, `codex exec`).
    """
    start_t = time.time()
    argv = assignment_record.get("cli_argv", [])
    cli_display = assignment_record.get("headless_cli", "")
    provider = assignment_record.get("provider", "unknown")
    model_id = assignment_record.get("model", "unknown")
    rem_q = assignment_record.get("remaining_quota")
    rem_str = f"{int(rem_q * 100)}%" if isinstance(rem_q, (int, float)) else "UNKNOWN"
    pred = assignment_record.get("predicted_cost") or {}

    if dry_run or not argv:
        return {
            "executed": False,
            "mode": "dry_run",
            "provider": provider,
            "model": model_id,
            "command": cli_display,
            "exit_code": 0,
            "duration_ms": int((time.time() - start_t) * 1000),
            "output": (
                f"[DRY-RUN PREVIEW - not executed] Mode={str(assignment_record.get('operating_mode', 'endurance')).upper()} | "
                f"Pool={assignment_record.get('pool_id')} | Model={provider.upper()} ({model_id}) | "
                f"Effective Quota={rem_str} | Est. Cost={pred.get('five_hour_pct_points', 0)}% 5h / {pred.get('weekly_pct_points', 0)}% wk.\n"
                f"Command: {cli_display}\n"
                f"Fallback on 95% quota / rate-limit: {assignment_record.get('fallback_model')}"
            ),
        }

    def _refused(reason: str) -> Dict[str, Any]:
        return {
            "executed": False,
            "mode": "live_headless",
            "provider": provider,
            "model": model_id,
            "command": cli_display,
            "exit_code": -3,
            "rate_limited": False,
            "handoff_recommended": True,
            "fallback_model": assignment_record.get("fallback_model"),
            "duration_ms": int((time.time() - start_t) * 1000),
            "output": f"Launch refused: {reason}",
        }

    workspace = str(assignment_record.get("workspace") or "").strip()
    if workspace and not Path(workspace).is_dir():
        return _refused(f"workspace {workspace!r} is not an existing directory")
    transport = assignment_record.get("prompt_transport") or prompt_transport_for(provider)
    stdin_prompt = assignment_record.get("cli_stdin_prompt") if transport == "stdin" else None
    if transport == "stdin" and stdin_prompt is None:
        return _refused("prompt transport is stdin but no prompt was attached to the assignment record")
    exe = str(argv[0]).lower()
    if exe.endswith((".cmd", ".bat")):
        unsafe = [str(a)[:40] for a in argv[1:] if _CMD_SHIM_UNSAFE.search(str(a))]
        if unsafe:
            return _refused(f"{argv[0]} is a cmd.exe shim and argv contains shell metacharacters: {unsafe}")

    proc = None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=workspace or None,
            stdin=subprocess.PIPE if stdin_prompt is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout_raw, stderr_raw = proc.communicate(input=stdin_prompt, timeout=timeout_sec)
        duration_ms = int((time.time() - start_t) * 1000)
        stdout_str = (stdout_raw or "").strip()
        stderr_str = (stderr_raw or "").strip()
        combined = stdout_str if stdout_str else stderr_str

        rate_limited = detect_rate_limit(proc.returncode, stdout_str, stderr_str)

        return {
            "executed": True,
            "mode": "live_headless",
            "provider": provider,
            "model": model_id,
            "worker_pid": proc.pid,
            "command": cli_display,
            "exit_code": proc.returncode,
            "rate_limited": rate_limited,
            "handoff_recommended": rate_limited or proc.returncode != 0,
            "fallback_model": assignment_record.get("fallback_model"),
            "duration_ms": duration_ms,
            "output": combined[:4000],
            "stderr_tail": stderr_str[-1000:],
        }
    except subprocess.TimeoutExpired:
        if proc is not None and proc.pid:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                proc.kill()
        return {
            "executed": True,
            "mode": "live_headless",
            "provider": provider,
            "model": model_id,
            "command": cli_display,
            "exit_code": -1,
            "rate_limited": False,
            "handoff_recommended": True,
            "fallback_model": assignment_record.get("fallback_model"),
            "duration_ms": int((time.time() - start_t) * 1000),
            "output": f"Execution timed out after {timeout_sec}s (process tree terminated). Recommend checkpoint & handoff to {assignment_record.get('fallback_model')}.",
        }
    except Exception as exc:
        return {
            "executed": False,
            "mode": "live_headless",
            "provider": provider,
            "model": model_id,
            "command": cli_display,
            "exit_code": -2,
            "rate_limited": False,
            "handoff_recommended": True,
            "fallback_model": assignment_record.get("fallback_model"),
            "duration_ms": int((time.time() - start_t) * 1000),
            "output": f"Subprocess launch error: {exc}",
        }
