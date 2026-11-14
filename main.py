"""
FastAPI Backend for AI Quota Hub & Reset-Aware Multi-Provider Quota Scheduler.

Endpoints:
- `GET /api/quotas`: Real-time zero-token quota telemetry across AGY, Claude Code, and Codex,
  plus Reset-Aware Pool States, Dynamic Codex Orchestrator Reserve, Codex Banked Reset Strategy,
  Empirical Cost Learner summary, and active Operating Mode (`endurance` | `sprint`).
- `POST /api/mode`: Persist operating mode (`endurance` or `sprint`) and backlog horizon settings.
- `POST /api/orchestrator`: Set preferred orchestrator ("auto" or specific model/provider).
- `POST /api/banked-resets/manual`: Set or clear manually verified Codex banked reset override.
- `POST /api/telemetry/observe`: Log pre/post task quota consumption into `QuotaCostLearner`.
- `POST /api/dispatch`: Classify, route via reset-aware mathematical scheduling, and (optionally)
  execute a work package via headless CLI (`agy -p`, `claude -p`, `codex exec`).
- `POST /api/simulate`: Apply or clear quota & reset-window override scenarios (including the
  Critical Pre-Reset Harvest Scenario: 10% weekly / 2h reset vs 80% weekly / 6d reset).
- `GET /api/history`: Recent work package assignments and dispatch audit log (compacted).
- `GET /api/policy`: Effective orchestrator policy (`~/.agents/policy.json` over implementation defaults).
- `GET /api/witness`: Advisory worker-health snapshot (never kills or duplicates a worker).
- `POST /api/refinery/submit` / `POST /api/refinery/verdict`: Scope-checked submission and independent review.
- `GET /`: Modern Web Dashboard UI.

Loopback only: requests with a non-local `Host` or a cross-origin `Origin` are refused (403).
The autonomous supervisor starts from the FastAPI lifespan (never at import) unless
`QUOTA_HUB_DISABLE_SUPERVISOR=1`, and live execution is off until enabled at runtime.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from allocation import (
    DEFAULT_PREFERENCE_WEIGHTS,
    coerce_credits,
    coerce_preference_weights,
    compute_actual_share_24h,
    compute_allocation,
    reset_credits,
    validate_preference_weights,
)
from hub_config import enabled_providers
from collectors import (
    collect_agy_quota,
    collect_claude_quota,
    collect_codex_quota,
    collect_live_activity,
)
from model_review import (
    apply_model_decision,
    get_catalog_status,
    get_latest_model_review,
    get_review_settings,
    is_scheduled_review_due,
    run_model_review,
    update_review_settings,
    validate_review_settings,
)
from orchestrator import (
    BURN_LEARNER,
    COST_LEARNER,
    HUB_HOME,
    MODEL_CATALOG,
    QUEUE_MANAGER,
    execute_headless_dispatch,
    get_allocation_credits,
    get_preference_weights,
    get_preference_generation,
    get_web_judge_status,
    load_web_judge_state,
    record_web_judge_event,
    reset_allocation_credits,
    reset_web_judge_state,
    route_work_package,
    select_active_orchestrator,
    set_allocation_credits,
    set_preference_weights,
)
from scheduler import (
    build_pool_states,
    compute_dynamic_codex_reserve,
    coordinator_busy_on_pool,
    evaluate_banked_reset_strategy,
    load_orchestrator_policy,
    run_multi_day_benchmark,
    save_orchestrator_policy_limits,
    validate_orchestrator_limits,
)

BASE_DIR = Path(__file__).resolve().parent
for _cand_dir in (BASE_DIR, BASE_DIR.parent, Path.home() / ".agents"):
    if _cand_dir.exists() and str(_cand_dir) not in os.sys.path:
        os.sys.path.insert(0, str(_cand_dir))
import policy_doctor  # noqa: E402

STATIC_DIR = BASE_DIR / "static"
STATE_FILE = HUB_HOME / "quota_hub_state.json"
HISTORY_LIMIT = 25
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver", "[::1]"}


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Start the autonomous supervisor with the server (not at import); stop it on shutdown."""
    global _LIFESPAN_ACTIVE
    _LIFESPAN_ACTIVE = True
    _ensure_supervisor_thread_started()
    try:
        yield
    finally:
        _LIFESPAN_ACTIVE = False
        _SUPERVISOR_STOP.set()


app = FastAPI(
    title="AI Quota Hub & Reset-Aware Multi-Provider Quota Scheduler",
    version="2.0.0",
    description=(
        "Zero-token local telemetry, reset-aware mathematical quota scheduler, empirical cost learner, "
        "and multi-provider work package router for AGY, Claude Code, and OpenAI Codex."
    ),
    lifespan=_lifespan,
)


def _host_is_local(host_value: str) -> bool:
    host = str(host_value or "").strip().lower()
    if not host:
        return False
    name = host.split("]")[0] + "]" if host.startswith("[") else host.split(":")[0]
    return name in ALLOWED_HOSTS


@app.middleware("http")
async def _loopback_only_guard(request: Request, call_next: Any) -> Any:
    """Refuse non-local Host headers (DNS rebinding) and any cross-origin Origin (browser CSRF)."""
    host = request.headers.get("host", "")
    if not _host_is_local(host):
        return JSONResponse(status_code=403, content={"detail": "Forbidden: non-local Host header"})
    origin = request.headers.get("origin")
    if origin is not None:
        netloc = urlsplit(origin).netloc.lower()
        if not netloc or netloc != host.strip().lower() or not _host_is_local(netloc):
            return JSONResponse(status_code=403, content={"detail": "Forbidden: cross-origin request"})
    return await call_next(request)


# Runtime state (persisted to quota_hub_state.json; Endurance is the initial default)
_RUNTIME_STATE: Dict[str, Any] = {
    "operating_mode": "endurance",
    "preferred_orchestrator": "auto",
    "preference_weights": dict(DEFAULT_PREFERENCE_WEIGHTS),
    "allocation_credits": reset_credits(),
    "backlog_settings": {
        "unfinished_tasks_count": 4,
        "pending_reviews_count": 2,
    },
    "supervisor_config": {
        "enabled": True,
        "interval_sec": 2.0,
        "execute_live": False,
        "max_jobs_per_tick": 2,
    },
    "manual_banked_reset_override": None,
    "project_goals": {
        "long_term_goal": "",
        "short_term_goal": "",
        "auto_generate_new_tasks": True,
        "auto_evaluate_and_improve": True,
        "updated_at": None,
    },
    "simulation_overrides": {},
    "dispatch_history": [],
}


def _coerce_project_goals(raw: Any) -> Dict[str, Any]:
    """Normalize persisted or incoming project_goals dict while preserving backward-compatible defaults."""
    src = raw if isinstance(raw, dict) else {}
    long_goal = str(src.get("long_term_goal") or "").strip()[:4000]
    short_goal = str(src.get("short_term_goal") or "").strip()[:2000]
    auto_gen = bool(src.get("auto_generate_new_tasks", True))
    auto_eval = bool(src.get("auto_evaluate_and_improve", True))
    updated_at = src.get("updated_at")
    if updated_at is not None:
        updated_at = str(updated_at).strip() or None
    active = bool(long_goal or short_goal)
    if active:
        triggers: List[str] = []
        if auto_eval:
            triggers.append(
                "EVALUATE_AND_IMPROVE: Audit current app state (UX simplicity, visual/audio polish, stability, Play Store & ad-readiness) via a scout/evaluator subagent, review findings in coordinator, and schedule concrete improvement tasks."
            )
        if auto_gen:
            triggers.append(
                "GENERATE_NEW_TASKS: When current backlog drains and quota remains, ask a discovery subagent for high-value tasks aligned with long_term_goal, debate/filter in coordinator, and form a new disjoint task wave."
            )
        directive = (
            f"ACTIVE GOAL-DRIVEN REPLENISHMENT. Long-term goal: {long_goal or '(none)'}"
            + (f" | Short-term focus: {short_goal}" if short_goal else "")
            + (" | Actions: " + " ".join(triggers) if triggers else "")
        )
    else:
        directive = "No long-term goal configured; proceed with standard explicit task queue only."
    return {
        "active": active,
        "mode": "goal_driven_replenish" if active else "standard_queue_only",
        "long_term_goal": long_goal,
        "short_term_goal": short_goal,
        "auto_generate_new_tasks": auto_gen,
        "auto_evaluate_and_improve": auto_eval,
        "updated_at": updated_at,
        "orchestrator_directive": directive,
    }

_SUPERVISOR_LOCK = threading.Lock()
_STATE_WRITE_LOCK = threading.Lock()
_SUPERVISOR_STOP = threading.Event()
_SUPERVISOR_THREAD: Optional[threading.Thread] = None
_LIFESPAN_ACTIVE = False
_STATE_LOAD_ERROR: Optional[str] = None
_SUPERVISOR_RUNTIME: Dict[str, Any] = {
    "started_at": None,
    "tick_count": 0,
    "active_tick_count": 0,
    "last_tick_at": None,
    "last_active_tick_at": None,
    "total_auto_dispatched": 0,
    "total_auto_completed": 0,
    "total_auto_resumed": 0,
    "total_orphans_recovered": 0,
    "last_step_summary": None,
    "last_error": None,
}


def _compact_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Audit-sized dispatch record: drops pool_states, ranked_candidates and other per-candidate math."""
    if not isinstance(entry, dict):
        return {}
    routing = entry.get("routing") or {}
    ar = routing.get("assignment_record") or None
    compact_ar = None
    if isinstance(ar, dict):
        compact_ar = {
            k: ar.get(k)
            for k in (
                "task_id", "task", "operating_mode", "provider", "pool_id", "model", "family",
                "capability_floor", "selected_tier", "reasoning_effort", "remaining_quota", "quota_state",
                "routing_score", "fallback_model", "edit_scope", "workspace", "read_only", "prompt_transport",
                "created_at",
            )
        }
        pred = ar.get("predicted_cost") or {}
        compact_ar["predicted_cost"] = {k: pred.get(k) for k in ("five_hour_mean", "weekly_mean", "prediction_source")}
    compact_routing = {
        k: routing.get(k)
        for k in ("status", "operating_mode", "capability_floor", "reasoning_effort", "earliest_reset_in_sec", "message")
        if k in routing
    }
    compact_routing["assignment_record"] = compact_ar
    compact_routing["rejected_count"] = len(routing.get("rejected_models") or [])
    execution = entry.get("execution")
    compact_exec = None
    if isinstance(execution, dict):
        compact_exec = {k: execution.get(k) for k in ("executed", "mode", "exit_code", "rate_limited", "duration_ms")}
        compact_exec["output"] = str(execution.get("output") or "")[:200]
    obs = entry.get("telemetry_observation")
    compact_obs = {k: obs.get(k) for k in ("id", "attribution_status")} if isinstance(obs, dict) else None
    return {
        "id": entry.get("id"),
        "timestamp": entry.get("timestamp"),
        "operating_mode": entry.get("operating_mode"),
        "request": entry.get("request"),
        "routing": compact_routing,
        "execution": compact_exec,
        "telemetry_observation": compact_obs,
    }


def _load_state() -> None:
    """
    Load compact persisted config. Simulation overrides are never restored, and live execution always
    starts disabled (re-enable via POST /api/supervisor/config). A corrupt file keeps the safe defaults.
    """
    global _STATE_LOAD_ERROR
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _STATE_LOAD_ERROR = f"{STATE_FILE}: {exc}"
        return
    if not isinstance(data, dict):
        _STATE_LOAD_ERROR = f"{STATE_FILE}: expected a JSON object"
        return
    data.pop("simulation_overrides", None)
    _RUNTIME_STATE.update(data)
    _RUNTIME_STATE["simulation_overrides"] = {}
    mode = str(_RUNTIME_STATE.get("operating_mode") or "endurance").lower().strip()
    _RUNTIME_STATE["operating_mode"] = mode if mode in {"sprint", "endurance"} else "endurance"
    if not isinstance(_RUNTIME_STATE.get("backlog_settings"), dict):
        _RUNTIME_STATE["backlog_settings"] = {
            "unfinished_tasks_count": 4,
            "pending_reviews_count": 2,
        }
    if not isinstance(_RUNTIME_STATE.get("supervisor_config"), dict):
        _RUNTIME_STATE["supervisor_config"] = {
            "enabled": True,
            "interval_sec": 2.0,
            "execute_live": False,
            "max_jobs_per_tick": 2,
        }
    _RUNTIME_STATE["supervisor_config"]["execute_live"] = False
    _RUNTIME_STATE["project_goals"] = _coerce_project_goals(_RUNTIME_STATE.get("project_goals"))
    _RUNTIME_STATE["preference_weights"] = coerce_preference_weights(_RUNTIME_STATE.get("preference_weights"))
    _RUNTIME_STATE["allocation_credits"] = coerce_credits(_RUNTIME_STATE.get("allocation_credits"))
    set_preference_weights(_RUNTIME_STATE["preference_weights"], reset_swrr=False)
    set_allocation_credits(_RUNTIME_STATE["allocation_credits"])
    if isinstance(_RUNTIME_STATE.get("web_judges"), dict):
        load_web_judge_state(_RUNTIME_STATE.get("web_judges"))
    history = _RUNTIME_STATE.get("dispatch_history")
    _RUNTIME_STATE["dispatch_history"] = (
        [_compact_history_entry(h) for h in history[:HISTORY_LIMIT]] if isinstance(history, list) else []
    )


def _save_state() -> None:
    """Atomically persist compact config + history (temp file + os.replace). Simulation overrides stay in memory."""
    with _STATE_WRITE_LOCK:
        live_weights = get_preference_weights()
        live_credits = get_allocation_credits()
        _RUNTIME_STATE["preference_weights"] = dict(live_weights)
        _RUNTIME_STATE["allocation_credits"] = dict(live_credits)
        _RUNTIME_STATE["web_judges"] = get_web_judge_status()
        persisted = {k: v for k, v in _RUNTIME_STATE.items() if k not in {"simulation_overrides", "dispatch_history"}}
        persisted["preference_weights"] = dict(live_weights)
        persisted["allocation_credits"] = dict(live_credits)
        persisted["dispatch_history"] = [
            _compact_history_entry(h) for h in (_RUNTIME_STATE.get("dispatch_history") or [])[:HISTORY_LIMIT]
        ]
        payload = json.dumps(persisted, indent=1, ensure_ascii=False)
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_name(f"{STATE_FILE.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, STATE_FILE)
        except OSError as exc:
            _SUPERVISOR_RUNTIME["last_error"] = f"state_save: {exc}"
            with contextlib.suppress(OSError):
                tmp.unlink()


_load_state()


class OperatingModeRequest(BaseModel):
    mode: str = Field(..., description="'endurance' or 'sprint'")
    unfinished_tasks_count: Optional[int] = Field(None, description="Optional count of unfinished tasks in backlog")
    pending_reviews_count: Optional[int] = Field(None, description="Optional count of pending cross-family reviews")


class OrchestratorPreferenceRequest(BaseModel):
    preferred: str = Field(..., description="'auto' or a specific model_id / provider name")


class ProjectGoalsRequest(BaseModel):
    long_term_goal: Optional[str] = Field(None, description="Long-term project vision / release & growth goal")
    short_term_goal: Optional[str] = Field(None, description="Optional short-term focus or sprint objective")
    auto_generate_new_tasks: Optional[bool] = Field(
        None,
        description="When True and long_term_goal is set, orchestrator generates new goal-aligned tasks when backlog drains",
    )
    auto_evaluate_and_improve: Optional[bool] = Field(
        None,
        description="When True and long_term_goal is set, orchestrator triggers current-app evaluation & improvement tasks",
    )


class ManualBankedResetRequest(BaseModel):
    enabled: bool = Field(False, description="True to set manual banked reset override, False to clear")
    available_count: int = Field(0, description="Manually verified count of applicable full resets")
    expires_at: Optional[str] = Field(None, description="Optional ISO-8601 expiry timestamp")
    notes: Optional[str] = Field(None, description="Verification note")


class TelemetryObservationRequest(BaseModel):
    task_id: str = Field(..., description="Work package ID")
    provider: str = Field(..., description="agy | claude | codex")
    model_id: str = Field(..., description="Executed model ID")
    pool_id: str = Field(..., description="agy_gemini | agy_claude_gpt | claude_shared | codex_shared")
    task_kind: str = Field("implement", description="Task kind")
    tier: str = Field("MID", description="LOW | MID | HIGH")
    complexity: str = Field("med", description="low | med | high")
    started_at: Optional[str] = Field(None, description="ISO start timestamp")
    ended_at: Optional[str] = Field(None, description="ISO end timestamp")
    duration_sec: float = Field(60.0, description="Wall-clock execution time in seconds")
    success: bool = Field(True, description="Whether the task succeeded")
    retry_count: int = Field(0, description="Number of retries")
    concurrent_pool_jobs: int = Field(1, description="Number of jobs running concurrently on the same pool")
    pre_5h_used_pct: Optional[float] = Field(None, description="5h used_percent before task")
    post_5h_used_pct: Optional[float] = Field(None, description="5h used_percent after task")
    pre_weekly_used_pct: Optional[float] = Field(None, description="Weekly used_percent before task")
    post_weekly_used_pct: Optional[float] = Field(None, description="Weekly used_percent after task")
    pre_5h_resets_at: Optional[str] = Field(None, description="5h resets_at before task")
    post_5h_resets_at: Optional[str] = Field(None, description="5h resets_at after task")
    pre_weekly_resets_at: Optional[str] = Field(None, description="Weekly resets_at before task")
    post_weekly_resets_at: Optional[str] = Field(None, description="Weekly resets_at after task")
    predicted_5h_cost: float = Field(0.03, description="Predicted 5h fraction cost")
    predicted_weekly_cost: float = Field(0.01, description="Predicted weekly fraction cost")
    dry_run: bool = Field(False, description="True if dry-run")
    acceptance_criteria_met: Optional[bool] = Field(None, description="True if objective acceptance criteria were met")
    tests_passed: Optional[bool] = Field(None, description="True if automated unit/integration tests passed")
    feature_accepted: Optional[bool] = Field(None, description="True if actual feature acceptance succeeded (distinct from automated test pass)")
    build_succeeded: Optional[bool] = Field(None, description="True if build/compile/lint succeeded")
    regression_count: int = Field(0, description="Number of regressions introduced")
    rework_count: Optional[int] = Field(None, description="Number of rework/revision cycles needed")


class TaskAcceptanceUpdateRequest(BaseModel):
    task_id: str = Field(..., description="Work package ID to update")
    acceptance_criteria_met: bool = Field(True, description="Objective acceptance criteria satisfied")
    tests_passed: bool = Field(True, description="Automated test suite passed")
    feature_accepted: bool = Field(True, description="Actual feature accepted (distinguished from automated test success)")
    build_succeeded: bool = Field(True, description="Build/compile succeeded")
    regression_count: int = Field(0, description="Number of regressions introduced")
    rework_count: int = Field(0, description="Number of rework cycles required")


class WorkPackageDispatchRequest(BaseModel):
    task: str = Field(..., description="Short title or summary of the work package")
    prompt: str = Field("", description="Full actionable prompt for the worker")
    kind: str = Field("implement", description="scan | extract | mechanical | implement | debug | design | test-author | browser | review | docs")
    ambiguity: str = Field("low", description="low | med | high")
    blast_radius: str = Field("local", description="local | module | cross-module | irreversible")
    dependency_role: str = Field("leaf", description="leaf | upstream")
    context_need: str = Field("pointer", description="pointer | module | broad")
    implementer_family: Optional[str] = Field(None, description="gemini | claude | openai (used when kind='review' to enforce cross-family review)")
    preferred_provider: Optional[str] = Field(None, description="Optional hint: agy | claude | codex")
    explicit_model: Optional[str] = Field(None, description="Optional user-specified model_id override to preserve")
    explicit_effort: Optional[str] = Field(None, description="Optional user-specified reasoning effort: low | medium | high")
    operating_mode: Optional[str] = Field(None, description="Optional mode override for this request ('endurance' or 'sprint'); defaults to persisted hub mode")
    critical: bool = Field(False, description="True only when high-consequence/critical work warrants high reasoning effort")
    escalation_reason: Optional[str] = Field(None, description="Explicit reason for escalating reasoning effort to high")
    project: Optional[str] = Field(None, description="Optional project identifier (e.g. my-app)")
    needs_decision_memory: bool = Field(False, description="True if cross-session decision lookup/handoff is needed")
    security_review: bool = Field(False, description="True if helper service/script security review is needed")
    unfinished_tasks_count: Optional[int] = Field(None, description="Unfinished tasks count for dynamic Codex reserve")
    pending_reviews_count: Optional[int] = Field(None, description="Pending reviews count for dynamic Codex reserve")
    edit_scope: List[str] = Field(default_factory=lambda: ["src/"], description="Declared edit_scope paths")
    workspace: Optional[str] = Field(None, description="Target workspace path")
    read_only: Optional[bool] = Field(None, description="Run worker in read-only mode")
    require_cross_family: bool = Field(False, description="True to hard-require a reviewer of another model family (review only)")
    allow_web_judge: bool = Field(False, description="Allow routing review/judge package to zero-API-cost web judge seats (web:gemini, web:chatgpt)")
    execute_live: bool = Field(False, description="If True, executes the headless CLI command via subprocess; otherwise runs in dry-run verification mode")
    timeout_sec: int = Field(60, description="Timeout in seconds for live headless CLI execution")


class WebJudgeEventRequest(BaseModel):
    event_type: str = Field("WEB_JUDGE_CALL_COMPLETED")
    surface: str
    call_id: Optional[str] = None
    mode: Optional[str] = None
    model_selected: Optional[str] = None
    prompt_chars: Optional[int] = None
    reply_chars: Optional[int] = None
    duration_seconds: Optional[float] = None
    reset_hint: Optional[str] = None
    message: Optional[str] = None


class SimulationOverrideRequest(BaseModel):
    enabled: bool = Field(False, description="Enable or clear simulation overrides")
    overrides: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Provider quota & window overrides")


def _apply_provider_override(base_prov: Dict[str, Any], ov: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge simulation overrides onto a provider telemetry dict (including nested `windows` and shorthand window keys)."""
    merged = copy.deepcopy(base_prov)
    window_keys = {"five_hour", "weekly", "gemini_5h", "gemini_weekly", "claude_gpt_5h", "claude_gpt_weekly"}
    for k, v in ov.items():
        if k == "windows" and isinstance(v, dict):
            wins = merged.setdefault("windows", {})
            for wk, wv in v.items():
                if isinstance(wv, dict):
                    cur_w = wins.setdefault(wk, {})
                    cur_w.update(wv)
                    if "remaining_fraction" in wv and "used_percent" not in wv and wv["remaining_fraction"] is not None:
                        cur_w["used_percent"] = round((1.0 - float(wv["remaining_fraction"])) * 100.0, 1)
                    elif "used_percent" in wv and "remaining_fraction" not in wv and wv["used_percent"] is not None:
                        cur_w["remaining_fraction"] = round(max(0.0, 1.0 - (float(wv["used_percent"]) / 100.0)), 4)
        elif k in window_keys:
            wins = merged.setdefault("windows", {})
            cur_w = wins.setdefault(k, {})
            if isinstance(v, (int, float)):
                rem = max(0.0, min(1.0, float(v)))
                cur_w["remaining_fraction"] = rem
                cur_w["used_percent"] = round((1.0 - rem) * 100.0, 1)
                if k == "five_hour":
                    for model_k in ("sol", "gpt_5", "o3", "sonnet", "opus", "haiku"):
                        if model_k in merged:
                            merged[model_k] = rem
            elif isinstance(v, dict):
                cur_w.update(v)
                if "remaining_fraction" in v and "used_percent" not in v and v["remaining_fraction"] is not None:
                    cur_w["used_percent"] = round((1.0 - float(v["remaining_fraction"])) * 100.0, 1)
        else:
            merged[k] = v
    if "reset_in_sec" in ov and isinstance(ov["reset_in_sec"], (int, float)):
        wins = merged.get("windows") or {}
        for wobj in wins.values():
            if isinstance(wobj, dict):
                wobj["reset_in_sec"] = int(ov["reset_in_sec"])
    merged["simulated"] = True
    if "status" not in ov:
        merged["status"] = "ok"
    if "confidence" not in ov:
        merged["confidence"] = "VERIFIED"
    if "freshness_age_sec" not in ov:
        merged["freshness_age_sec"] = 0
    return merged


def _unavailable_provider(prov: str, reason: str) -> Dict[str, Any]:
    """UNAVAILABLE telemetry for a provider that is disabled or whose collector failed (never fake 0% / 100%)."""
    window_names = ["five_hour", "weekly"]
    if prov == "agy":
        window_names += ["gemini_5h", "gemini_weekly", "claude_gpt_5h", "claude_gpt_weekly"]
    return {
        "status": "unknown",
        "confidence": "UNAVAILABLE",
        "freshness_age_sec": 999999,
        "account": "Unavailable",
        "reset_in_sec": 3600,
        "windows": {
            w: {"remaining_fraction": None, "used_percent": None, "reset_in_sec": 3600, "resets_at": None}
            for w in window_names
        },
        "plan": "unknown",
        "shared_pool": True,
        "models": [],
        "enabled": prov in enabled_providers(),
        "source": reason,
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def get_merged_quotas() -> Dict[str, Any]:
    """
    Collect real-time telemetry from all 3 providers, apply any active simulation overrides,
    and compute Reset-Aware Pool States, Dynamic Codex Reserve, Banked Reset Strategy, and Cost Learning summary.
    """
    manual_override = _RUNTIME_STATE.get("manual_banked_reset_override")
    collectors = {
        "agy": collect_agy_quota,
        "claude": collect_claude_quota,
        "codex": lambda: collect_codex_quota(manual_reset_override=manual_override),
    }
    enabled = enabled_providers()
    providers = {}
    for prov, collect in collectors.items():
        if prov not in enabled:
            providers[prov] = _unavailable_provider(prov, "disabled in Hub config (hub_config.py)")
            continue
        try:
            providers[prov] = collect()
        except Exception as exc:  # a broken or absent provider must never take the Hub down
            providers[prov] = _unavailable_provider(prov, f"collector error: {type(exc).__name__}: {exc}"[:300])

    overrides = _RUNTIME_STATE.get("simulation_overrides") or {}
    if overrides:
        for prov, ov in overrides.items():
            if prov in providers and isinstance(ov, dict):
                providers[prov] = _apply_provider_override(providers[prov], ov)

    mode = str(_RUNTIME_STATE.get("operating_mode") or "endurance").lower().strip()
    if mode not in {"sprint", "endurance"}:
        mode = "endurance"

    backlog = _RUNTIME_STATE.get("backlog_settings") or {}
    unfinished_cnt = int(backlog.get("unfinished_tasks_count", 4))
    reviews_cnt = int(backlog.get("pending_reviews_count", 2))

    preferred = _RUNTIME_STATE.get("preferred_orchestrator", "auto")
    orch_info = select_active_orchestrator(preferred, providers, operating_mode=mode)
    live_activity = collect_live_activity()

    backlog_queue = QUEUE_MANAGER.get_queue_summary(limit=15)
    queued_cnt = int((backlog_queue.get("counts") or {}).get("QUEUED", 0))
    real_codex_suitable_queued_cnt = int(backlog_queue.get("codex_suitable_queued_count", queued_cnt))
    effective_unfinished_cnt = max(unfinished_cnt, queued_cnt)

    codex_reserve = compute_dynamic_codex_reserve(
        providers,
        operating_mode=mode,
        unfinished_tasks_count=effective_unfinished_cnt,
        pending_reviews_count=reviews_cnt,
        live_activity=live_activity,
        suitable_queued_tasks_count=real_codex_suitable_queued_cnt,
    )
    pref_weights = get_preference_weights()
    _RUNTIME_STATE["preference_weights"] = dict(pref_weights)
    policy = load_orchestrator_policy(auto_sync=False)
    pool_states = build_pool_states(
        providers,
        live_activity=live_activity,
        cost_learner=COST_LEARNER,
        operating_mode=mode,
        codex_reserve=codex_reserve,
        policy=policy,
        burn_learner=BURN_LEARNER,
        preference_weights=pref_weights,
    )
    BURN_LEARNER.observe_snapshot(providers, live_activity=live_activity, pool_states=pool_states)
    actual_24h = compute_actual_share_24h(
        dispatch_history=_RUNTIME_STATE.get("dispatch_history"),
        db_path=COST_LEARNER.db_path,
    )
    allocation = compute_allocation(
        pool_states,
        burn_learner=BURN_LEARNER,
        cost_learner=COST_LEARNER,
        sliders=pref_weights,
        operating_mode=mode,
        policy=policy,
        actual_share_24h=actual_24h,
    )
    prov_limits_summary: Dict[str, Dict[str, Any]] = {}
    for prov in ("agy", "claude", "codex"):
        p_alloc = (allocation.get("providers") or {}).get(prov) or {}
        ceil_val = int((policy.get("max_per_provider") or {}).get(prov, 6))
        now_val = int(p_alloc.get("max_workers", ceil_val))
        band_val = str(p_alloc.get("band") or "normal")
        prov_limits_summary[prov] = {
            "ceiling": ceil_val,
            "max_workers": now_val,
            "band": band_val,
            "summary": f"ceiling {ceil_val} · now {now_val} ({band_val} band)",
        }
    limits_payload: Dict[str, Any] = {
        "max_leaves": int(policy.get("max_leaves", 10)),
        "max_per_provider": dict(policy.get("max_per_provider") or {"agy": 6, "claude": 6, "codex": 6}),
        "max_per_pool": dict(
            policy.get("max_per_pool")
            or {"agy_gemini": 6, "agy_claude_gpt": 6, "claude_shared": 6, "codex_shared": 6}
        ),
        "source": policy.get("_source", "impl-default"),
        "backup_path": _RUNTIME_STATE.get("last_policy_backup_path"),
        "providers": prov_limits_summary,
    }
    allocation["limits"] = limits_payload
    banked_reset_strategy = evaluate_banked_reset_strategy(
        codex_quota=providers["codex"],
        operating_mode=mode,
        pool_states=pool_states,
        unfinished_tasks_count=effective_unfinished_cnt,
        suitable_queued_tasks_count=real_codex_suitable_queued_cnt,
        independent_fallback_healthy=bool(codex_reserve.get("fallback_coordinators_healthy", True)),
    )
    cost_learning = COST_LEARNER.get_summary(limit=10)

    return {
        "operating_mode": mode,
        "preference_weights": dict(pref_weights),
        "allocation_credits": get_allocation_credits(),
        "allocation": allocation,
        "limits": limits_payload,
        "ceilings": limits_payload,
        "policy": policy,
        "backlog_settings": {
            "unfinished_tasks_count": unfinished_cnt,
            "pending_reviews_count": reviews_cnt,
        },
        "providers": providers,
        "pool_states": pool_states,
        "codex_reserve": codex_reserve,
        "banked_reset_strategy": banked_reset_strategy,
        "cost_learning": cost_learning,
        "backlog_queue": backlog_queue,
        "autonomous_supervisor": _get_supervisor_status(),
        "orchestrator": {
            "preferred": orch_info["preferred"],
            "active": orch_info["active"],
            "active_provider": orch_info["active_provider"],
            "active_label": orch_info["active_label"],
            "remaining_quota": orch_info["remaining_quota"],
            "quota_state": orch_info.get("quota_state", "VERIFIED"),
            "operating_mode": mode,
            "reason": orch_info["reason"],
        },
        "project_goals": _coerce_project_goals(_RUNTIME_STATE.get("project_goals")),
        "live_activity": live_activity,
        "web_judges": get_web_judge_status(),
        "catalog": MODEL_CATALOG,
        "catalog_status": get_catalog_status(),
        "simulation_active": bool(overrides),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/api/quotas")
def api_get_quotas() -> Dict[str, Any]:
    """Return normalized real-time quotas, reset-aware pool states, Codex reserve, and banked reset strategy."""
    return get_merged_quotas()


@app.post("/api/mode")
def api_set_operating_mode(req: OperatingModeRequest) -> Dict[str, Any]:
    """Persist the operating mode ('endurance' or 'sprint') and optional backlog horizon counts across restarts."""
    mode = req.mode.lower().strip()
    if mode not in {"sprint", "endurance"}:
        raise HTTPException(status_code=400, detail="mode must be 'endurance' or 'sprint'")
    _RUNTIME_STATE["operating_mode"] = mode
    backlog = _RUNTIME_STATE.setdefault("backlog_settings", {"unfinished_tasks_count": 4, "pending_reviews_count": 2})
    if req.unfinished_tasks_count is not None:
        backlog["unfinished_tasks_count"] = max(0, int(req.unfinished_tasks_count))
    if req.pending_reviews_count is not None:
        backlog["pending_reviews_count"] = max(0, int(req.pending_reviews_count))
    _save_state()
    return get_merged_quotas()


@app.post("/api/preferences")
async def api_set_preferences(request: Request) -> Dict[str, Any]:
    """
    Save 0..10 provider preference sliders (`{"agy": int, "codex": int, "claude": int}`),
    reset SWRR allocation credits to 0, persist state, and return the updated `/api/quotas` payload.
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    try:
        cleaned = validate_preference_weights(body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    set_preference_weights(cleaned, reset_swrr=True)
    _RUNTIME_STATE["preference_weights"] = dict(cleaned)
    _RUNTIME_STATE["allocation_credits"] = get_allocation_credits()
    _save_state()
    return get_merged_quotas()


@app.post("/api/limits")
async def api_set_limits(request: Request) -> Dict[str, Any]:
    """
    Validate and update concurrency ceilings (`max_leaves`, `max_per_provider`, `max_per_pool`)
    in `~/.agents/policy.json` (or `ORCHESTRATOR_POLICY_PATH`), creating a timestamped backup
    (retaining the newest 10), writing atomically, and reloading `load_orchestrator_policy()` in-process.
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    try:
        save_res = save_orchestrator_policy_limits(body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _RUNTIME_STATE["last_policy_backup_path"] = save_res["backup_path"]
    merged = get_merged_quotas()
    merged["backup_path"] = save_res["backup_path"]
    merged["policy_path"] = save_res["policy_path"]
    merged["backups_kept"] = save_res["backups_kept"]
    return merged


@app.post("/api/orchestrator")
def api_set_orchestrator(req: OrchestratorPreferenceRequest) -> Dict[str, Any]:
    """Update the preferred Chef/Orchestrator model ('auto' or specific model_id)."""
    _RUNTIME_STATE["preferred_orchestrator"] = req.preferred.strip()
    _save_state()
    return get_merged_quotas()


@app.get("/api/goals")
def api_get_project_goals() -> Dict[str, Any]:
    """Return the active long-term and short-term project goals and autonomous replenishment flags."""
    return _coerce_project_goals(_RUNTIME_STATE.get("project_goals"))


@app.post("/api/goals")
def api_set_project_goals(req: ProjectGoalsRequest) -> Dict[str, Any]:
    """
    Persist long-term and optional short-term project goals plus autonomous evaluation/task-generation
    toggles to `quota_hub_state.json`, and return the updated `/api/quotas` payload.
    When `long_term_goal` and `short_term_goal` are both empty, standard queue-only behavior is preserved.
    """
    current = _coerce_project_goals(_RUNTIME_STATE.get("project_goals"))
    updated_raw = {
        "long_term_goal": current["long_term_goal"] if req.long_term_goal is None else req.long_term_goal,
        "short_term_goal": current["short_term_goal"] if req.short_term_goal is None else req.short_term_goal,
        "auto_generate_new_tasks": (
            current["auto_generate_new_tasks"] if req.auto_generate_new_tasks is None else req.auto_generate_new_tasks
        ),
        "auto_evaluate_and_improve": (
            current["auto_evaluate_and_improve"]
            if req.auto_evaluate_and_improve is None
            else req.auto_evaluate_and_improve
        ),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _RUNTIME_STATE["project_goals"] = _coerce_project_goals(updated_raw)
    _save_state()
    return get_merged_quotas()


@app.post("/api/banked-resets/manual")
def api_set_manual_banked_resets(req: ManualBankedResetRequest) -> Dict[str, Any]:
    """Set or clear a manually verified Codex banked reset count fallback when live telemetry is unavailable."""
    if not req.enabled:
        _RUNTIME_STATE["manual_banked_reset_override"] = None
    else:
        _RUNTIME_STATE["manual_banked_reset_override"] = {
            "enabled": True,
            "available_count": max(0, int(req.available_count)),
            "expires_at": req.expires_at,
            "notes": req.notes or "Manually verified by user in Quota Hub",
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    _save_state()
    return get_merged_quotas()


@app.post("/api/telemetry/observe")
def api_record_telemetry_observation(req: TelemetryObservationRequest) -> Dict[str, Any]:
    """Record a pre/post task quota consumption observation in the SQLite empirical cost learner."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    obs = COST_LEARNER.record_observation(
        task_id=req.task_id,
        provider=req.provider,
        model_id=req.model_id,
        pool_id=req.pool_id,
        task_kind=req.task_kind,
        tier=req.tier,
        complexity=req.complexity,
        started_at=req.started_at or now_iso,
        ended_at=req.ended_at or now_iso,
        duration_sec=req.duration_sec,
        success=req.success,
        retry_count=req.retry_count,
        concurrent_pool_jobs=req.concurrent_pool_jobs,
        pre_5h_used_pct=req.pre_5h_used_pct,
        post_5h_used_pct=req.post_5h_used_pct,
        pre_weekly_used_pct=req.pre_weekly_used_pct,
        post_weekly_used_pct=req.post_weekly_used_pct,
        pre_5h_resets_at=req.pre_5h_resets_at,
        post_5h_resets_at=req.post_5h_resets_at,
        pre_weekly_resets_at=req.pre_weekly_resets_at,
        post_weekly_resets_at=req.post_weekly_resets_at,
        predicted_5h_cost=req.predicted_5h_cost,
        predicted_weekly_cost=req.predicted_weekly_cost,
        dry_run=req.dry_run,
        acceptance_criteria_met=req.acceptance_criteria_met,
        tests_passed=req.tests_passed,
        feature_accepted=req.feature_accepted,
        build_succeeded=req.build_succeeded,
        regression_count=req.regression_count,
        rework_count=req.rework_count,
    )
    return {
        "recorded": obs,
        "cost_learning": COST_LEARNER.get_summary(limit=10),
    }


@app.post("/api/telemetry/acceptance")
def api_update_task_acceptance(req: TaskAcceptanceUpdateRequest) -> Dict[str, Any]:
    """
    Update objective acceptance criteria, automated test pass vs. actual feature acceptance,
    build status, regressions, and rework count for an assigned task (without duplicating tasks).
    """
    updated = COST_LEARNER.update_task_acceptance(
        task_id=req.task_id,
        acceptance_criteria_met=req.acceptance_criteria_met,
        tests_passed=req.tests_passed,
        feature_accepted=req.feature_accepted,
        build_succeeded=req.build_succeeded,
        regression_count=req.regression_count,
        rework_count=req.rework_count,
    )
    return {
        "updated": updated,
        "agent_scorecard": COST_LEARNER.get_agent_scorecard(),
    }


@app.get("/api/scorecard")
def api_get_agent_scorecard(
    task_kind: Optional[str] = None,
    tier: Optional[str] = None,
    complexity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return the non-duplicating in-flight Agent Scorecard comparing agents within similar
    task categories (`task_kind`) and complexity levels (`tier`, `complexity`), distinguishing
    automated test success from actual feature acceptance.
    """
    return COST_LEARNER.get_agent_scorecard(task_kind=task_kind, tier=tier, complexity=complexity)


def _extract_pool_window_snapshot(pool_id: str, providers: Dict[str, Any]) -> Dict[str, Any]:
    """Extract current 5h and weekly `used_percent` and `resets_at` for a given pool before/after execution."""
    if pool_id == "agy_gemini":
        wins = (providers.get("agy") or {}).get("windows") or {}
        w5 = wins.get("gemini_5h") or {}
        ww = wins.get("gemini_weekly") or {}
    elif pool_id == "agy_claude_gpt":
        wins = (providers.get("agy") or {}).get("windows") or {}
        w5 = wins.get("claude_gpt_5h") or {}
        ww = wins.get("claude_gpt_weekly") or {}
    elif pool_id == "claude_shared":
        wins = (providers.get("claude") or {}).get("windows") or {}
        w5 = wins.get("five_hour") or {}
        ww = wins.get("weekly") or {}
    else:
        wins = (providers.get("codex") or {}).get("windows") or {}
        w5 = wins.get("five_hour") or {}
        ww = wins.get("weekly") or {}
    return {
        "used_5h": w5.get("used_percent"),
        "resets_5h": w5.get("resets_at"),
        "used_wk": ww.get("used_percent"),
        "resets_wk": ww.get("resets_at"),
    }


@app.post("/api/dispatch")
def api_dispatch_package(req: WorkPackageDispatchRequest) -> Dict[str, Any]:
    """
    Evaluate current provider quotas & reset windows, classify the work package capability floor,
    route to the optimal model via reset-aware mathematical scheduling, and either dry-run or
    execute via headless CLI (logging pre/post consumption into `QuotaCostLearner`).
    """
    snapshot = get_merged_quotas()
    if req.execute_live and snapshot.get("simulation_active"):
        raise HTTPException(status_code=409, detail="SIMULATION_ACTIVE: clear /api/simulate overrides before live execution")
    providers_quota = snapshot["providers"]
    orch_info = snapshot["orchestrator"]
    live_activity = snapshot.get("live_activity")
    backlog = snapshot.get("backlog_settings") or {}
    effective_mode = str(req.operating_mode or snapshot.get("operating_mode") or "endurance").lower().strip()

    req_dict = req.model_dump()
    req_dict["operating_mode"] = effective_mode
    if req_dict.get("unfinished_tasks_count") is None:
        req_dict["unfinished_tasks_count"] = backlog.get("unfinished_tasks_count", 4)
    if req_dict.get("pending_reviews_count") is None:
        req_dict["pending_reviews_count"] = backlog.get("pending_reviews_count", 2)
    if not req_dict.get("workspace"):
        req_dict["workspace"] = str(BASE_DIR)
    if req_dict.get("read_only") is None:
        req_dict["read_only"] = req.kind in {"scan", "extract", "review", "design", "docs"}

    routing_result = route_work_package(
        req_dict,
        providers_quota,
        orch_info,
        operating_mode=effective_mode,
        live_activity=live_activity,
        cost_learner=COST_LEARNER,
        burn_learner=BURN_LEARNER,
    )
    _RUNTIME_STATE["preference_weights"] = get_preference_weights()
    _RUNTIME_STATE["allocation_credits"] = get_allocation_credits()
    routing_result["allocation_credits"] = dict(_RUNTIME_STATE["allocation_credits"])

    execution_result = None
    telemetry_obs = None
    if routing_result["status"] == "routed" and routing_result.get("assignment_record"):
        ar = routing_result["assignment_record"]
        pool_id = ar["pool_id"]
        pre_snap = _extract_pool_window_snapshot(pool_id, providers_quota)
        started_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        execution_result = execute_headless_dispatch(
            ar,
            dry_run=not req.execute_live,
            timeout_sec=req.timeout_sec,
        )

        ended_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if req.execute_live:
            post_providers = get_merged_quotas()["providers"]
            post_snap = _extract_pool_window_snapshot(pool_id, post_providers)
        else:
            post_snap = pre_snap

        pool_state = (routing_result.get("pool_states") or {}).get(pool_id) or {}
        concurrent_jobs = max(1, int(pool_state.get("active_jobs_count", 0)) + 1)
        pred_cost = ar.get("predicted_cost") or {}

        if ar["provider"] == "web":
            telemetry_obs = COST_LEARNER.record_observation(
                task_id=ar["task_id"],
                provider=ar["provider"],
                model_id=ar["model"],
                pool_id=pool_id,
                task_kind=req.kind,
                tier=ar["selected_tier"],
                complexity=req.ambiguity,
                started_at=started_iso,
                ended_at=ended_iso,
                duration_sec=max(0.01, float(execution_result.get("duration_ms", 0)) / 1000.0),
                success=(execution_result.get("exit_code") == 0),
                retry_count=0,
                concurrent_pool_jobs=concurrent_jobs,
                pre_5h_used_pct=None,
                post_5h_used_pct=None,
                pre_weekly_used_pct=None,
                post_weekly_used_pct=None,
                pre_5h_resets_at=None,
                post_5h_resets_at=None,
                pre_weekly_resets_at=None,
                post_weekly_resets_at=None,
                predicted_5h_cost=0.0,
                predicted_weekly_cost=0.0,
                dry_run=True,
                active_coordinator_on_pool=False,
            )
            if req.execute_live:
                record_web_judge_event(
                    {
                        "event_type": "SURFACE_LIMIT_REACHED" if execution_result.get("rate_limited") else "WEB_JUDGE_CALL_COMPLETED",
                        "surface": ar["model"],
                        "call_id": ar["task_id"],
                        "mode": effective_mode,
                        "model_selected": ar["model"],
                        "duration_seconds": round(float(execution_result.get("duration_ms", 0)) / 1000.0, 3),
                        "message": str(execution_result.get("output") or "")[:200],
                    }
                )
        else:
            telemetry_obs = COST_LEARNER.record_observation(
                task_id=ar["task_id"],
                provider=ar["provider"],
                model_id=ar["model"],
                pool_id=pool_id,
                task_kind=req.kind,
                tier=ar["selected_tier"],
                complexity=req.ambiguity,
                started_at=started_iso,
                ended_at=ended_iso,
                duration_sec=max(0.01, float(execution_result.get("duration_ms", 0)) / 1000.0),
                success=(execution_result.get("exit_code") == 0),
                retry_count=0,
                concurrent_pool_jobs=concurrent_jobs,
                pre_5h_used_pct=pre_snap["used_5h"],
                post_5h_used_pct=post_snap["used_5h"],
                pre_weekly_used_pct=pre_snap["used_wk"],
                post_weekly_used_pct=post_snap["used_wk"],
                pre_5h_resets_at=pre_snap["resets_5h"],
                post_5h_resets_at=post_snap["resets_5h"],
                pre_weekly_resets_at=pre_snap["resets_wk"],
                post_weekly_resets_at=post_snap["resets_wk"],
                predicted_5h_cost=float(pred_cost.get("five_hour_mean", 0.03)),
                predicted_weekly_cost=float(pred_cost.get("weekly_mean", 0.01)),
                dry_run=not req.execute_live,
                active_coordinator_on_pool=coordinator_busy_on_pool(live_activity, pool_id),
            )

    history_entry = {
        "id": routing_result["assignment_record"]["task_id"] if routing_result.get("assignment_record") else "WP-PAUSED",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "operating_mode": effective_mode,
        "request": {
            "task": req.task,
            "kind": req.kind,
            "ambiguity": req.ambiguity,
            "blast_radius": req.blast_radius,
            "dependency_role": req.dependency_role,
            "context_need": req.context_need,
            "implementer_family": req.implementer_family,
            "operating_mode": effective_mode,
            "execute_live": req.execute_live,
        },
        "routing": routing_result,
        "execution": execution_result,
        "telemetry_observation": telemetry_obs,
    }

    history = _RUNTIME_STATE.setdefault("dispatch_history", [])
    history.insert(0, _compact_history_entry(history_entry))
    _RUNTIME_STATE["dispatch_history"] = history[:HISTORY_LIMIT]
    _save_state()

    return history_entry


@app.post("/api/web-judge/event")
def api_post_web_judge_event(req: WebJudgeEventRequest) -> Dict[str, Any]:
    """Record a web judge telemetry/signal event (e.g., call completed, surface limit reached, auth wall)."""
    try:
        status = record_web_judge_event(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _RUNTIME_STATE["web_judges"] = status
    _save_state()
    raw_s = str(req.surface or "").lower().strip()
    if raw_s.startswith("web:"):
        raw_s = raw_s[4:].strip()
    if raw_s.startswith("web_"):
        raw_s = raw_s[4:].strip()
    surf_name = "chatgpt" if raw_s in {"chatgpt", "openai", "gpt"} else ("gemini" if raw_s == "gemini" else raw_s)
    surf_info = (status.get("surfaces") or {}).get(surf_name) or {}
    return {
        "recorded": True,
        "surface": surf_name,
        "status": surf_info.get("status"),
        "rate_limited": surf_info.get("rate_limited"),
        "web_judges": get_web_judge_status(),
    }


@app.get("/api/web-judge/status")
def api_get_web_judge_status() -> Dict[str, Any]:
    """Return current web judge seat status (`web:gemini` and `web:chatgpt`)."""
    return {"web_judges": get_web_judge_status()}


@app.post("/api/simulate")
def api_simulate_quotas(req: SimulationOverrideRequest) -> Dict[str, Any]:
    """Set or clear simulated quota & reset-window overrides to test reset-aware routing and 85%/95% handoff."""
    if not req.enabled or not req.overrides:
        _RUNTIME_STATE["simulation_overrides"] = {}
    else:
        _RUNTIME_STATE["simulation_overrides"] = req.overrides
    # Simulation overrides live in memory only; they are never written to disk.
    return get_merged_quotas()


class BacklogEnqueueRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="List of approved backlog task definitions")
    clear_existing: bool = Field(False, description="Clear non-RUNNING rows before enqueueing (RUNNING rows are always kept)")
    preserve_existing: bool = Field(True, description="True to preserve existing COMPLETED/RUNNING jobs on restart or re-enqueue")


class BacklogStepRequest(BaseModel):
    max_jobs: int = Field(2, description="Maximum jobs to dispatch in this supervisor step")
    execute_live: bool = Field(False, description="True to execute live CLI subprocess, False for a read-only preview")


class RefinerySubmitRequest(BaseModel):
    job_id: str = Field(..., description="COMPLETED backlog job to submit for review")
    changed_files: Optional[List[str]] = Field(None, description="Changed paths; omitted = inspect via git in the job workspace")
    checks_passed: bool = Field(True, description="Whether the coordinator's verification checks passed")
    implementer_agent_id: Optional[str] = Field(None, description="Implementer agent/session id (defaults to the hub run id)")


class RefineryVerdictRequest(BaseModel):
    job_id: str = Field(..., description="Job whose submission is PENDING_REVIEW")
    reviewer_agent_id: str = Field(..., description="Reviewer agent/session id (must differ from the implementer)")
    accepted: bool = Field(..., description="True to accept, False to reject for rework")
    findings: str = Field("", description="Review findings summary or path")
    reviewer_family: Optional[str] = Field(None, description="gemini | claude | openai (required when the job requires cross-family review)")
    review_session_id: Optional[str] = Field(None, description="Reviewer run/session id")
    require_verified_session: Optional[bool] = Field(None, description="Require review_session_id to resolve to a real distinct session")


class NativeLeaseClaimRequest(BaseModel):
    job_id: str = Field(..., description="Unique job/lease ID for the native subagent")
    provider: str = Field(..., description="agy | claude | codex")
    task: str = Field("", description="Short task summary")
    pool_id: Optional[str] = Field(None, description="Optional pool ID override")
    model_id: Optional[str] = Field(None, description="Optional model ID")
    workspace: Optional[str] = Field(None, description="Workspace directory")
    edit_scope: Optional[List[str]] = Field(None, description="Declared edit_scope globs")
    read_only: Optional[bool] = Field(None, description="True if read-only worker")
    kind: str = Field("implement", description="Work package kind")
    session_id: Optional[str] = Field(None, description="Native subagent session/conversation/thread ID")
    worker_pid: Optional[int] = Field(None, description="Optional worker OS PID")


class NativeLeaseReleaseRequest(BaseModel):
    job_id: str = Field(..., description="Lease job_id to release")
    run_id: Optional[str] = Field(None, description="Optional run_id token")
    status: str = Field("COMPLETED", description="COMPLETED | FAILED_ESCALATED | QUEUED")
    output_summary: str = Field("", description="Optional worker output summary")
    session_id: Optional[str] = Field(None, description="Native subagent session/conversation/thread ID")
    worker_pid: Optional[int] = Field(None, description="Optional worker OS PID")


class SupervisorConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(None, description="Enable or pause autonomous background supervisor loop")
    interval_sec: Optional[float] = Field(None, description="Background tick interval in seconds (0.5 to 300)")
    execute_live: Optional[bool] = Field(None, description="True to execute live CLI workers, False for dry-run")
    max_jobs_per_tick: Optional[int] = Field(None, description="Maximum jobs dispatched per autonomous tick")


def _get_supervisor_status() -> Dict[str, Any]:
    cfg = _RUNTIME_STATE.get("supervisor_config") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "interval_sec": float(cfg.get("interval_sec", 2.0)),
        "execute_live": bool(cfg.get("execute_live", False)),
        "max_jobs_per_tick": int(cfg.get("max_jobs_per_tick", 2)),
        "thread_alive": bool(_SUPERVISOR_THREAD and _SUPERVISOR_THREAD.is_alive()),
        "disabled_by_env": os.environ.get("QUOTA_HUB_DISABLE_SUPERVISOR") == "1",
        "state_load_error": _STATE_LOAD_ERROR,
        "started_at": _SUPERVISOR_RUNTIME.get("started_at"),
        "tick_count": int(_SUPERVISOR_RUNTIME.get("tick_count", 0)),
        "active_tick_count": int(_SUPERVISOR_RUNTIME.get("active_tick_count", 0)),
        "last_tick_at": _SUPERVISOR_RUNTIME.get("last_tick_at"),
        "last_active_tick_at": _SUPERVISOR_RUNTIME.get("last_active_tick_at"),
        "total_auto_dispatched": int(_SUPERVISOR_RUNTIME.get("total_auto_dispatched", 0)),
        "total_auto_completed": int(_SUPERVISOR_RUNTIME.get("total_auto_completed", 0)),
        "total_auto_resumed": int(_SUPERVISOR_RUNTIME.get("total_auto_resumed", 0)),
        "total_orphans_recovered": int(_SUPERVISOR_RUNTIME.get("total_orphans_recovered", 0)),
        "last_step_summary": _SUPERVISOR_RUNTIME.get("last_step_summary"),
        "last_error": _SUPERVISOR_RUNTIME.get("last_error"),
    }


def _run_supervisor_cycle(
    max_jobs: int = 2,
    execute_live: bool = False,
    trigger_source: str = "manual",
) -> Dict[str, Any]:
    with _SUPERVISOR_LOCK:
        snapshot = get_merged_quotas()
        simulation_blocked_live = bool(execute_live and snapshot.get("simulation_active"))
        if simulation_blocked_live:
            # Simulated quotas must never drive real launches: fall back to a read-only preview tick.
            execute_live = False
        providers_quota = snapshot["providers"]
        orch_info = snapshot["orchestrator"]
        live_activity = snapshot.get("live_activity") or {}
        mode = snapshot.get("operating_mode", "endurance")
        alive_pids = {int(w["pid"]) for w in (live_activity.get("workers") or []) if w.get("pid")}
        if execute_live:
            recovered = QUEUE_MANAGER.recover_orphaned_jobs(alive_pids=alive_pids)
        else:
            recovered = {"requeued_jobs": [], "escalated_jobs": [], "recovered_count": 0, "skipped": "preview"}

        step_out = QUEUE_MANAGER.step_supervisor(
            route_fn=route_work_package,
            execute_fn=execute_headless_dispatch,
            providers_quota=providers_quota,
            orchestrator_info=orch_info,
            cost_learner=COST_LEARNER,
            operating_mode=mode,
            max_dispatch_per_tick=max_jobs,
            execute_live=execute_live,
            live_activity=live_activity,
        )
        step_out["recovered_orphans"] = recovered
        step_out["trigger_source"] = trigger_source
        step_out["simulation_blocked_live"] = simulation_blocked_live

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _SUPERVISOR_RUNTIME["active_tick_count"] = int(_SUPERVISOR_RUNTIME.get("active_tick_count", 0)) + 1
        _SUPERVISOR_RUNTIME["last_active_tick_at"] = now_iso
        _SUPERVISOR_RUNTIME["total_auto_dispatched"] = int(_SUPERVISOR_RUNTIME.get("total_auto_dispatched", 0)) + len(
            step_out.get("dispatched") or []
        )
        _SUPERVISOR_RUNTIME["total_auto_completed"] = int(_SUPERVISOR_RUNTIME.get("total_auto_completed", 0)) + sum(
            1 for d in (step_out.get("dispatched") or []) if d.get("status") == "COMPLETED"
        )
        _SUPERVISOR_RUNTIME["total_auto_resumed"] = (
            int(_SUPERVISOR_RUNTIME.get("total_auto_resumed", 0))
            + int(step_out.get("resumed_from_pause", 0))
            + len(step_out.get("resumed_from_codex_reset") or [])
        )
        _SUPERVISOR_RUNTIME["total_orphans_recovered"] = int(
            _SUPERVISOR_RUNTIME.get("total_orphans_recovered", 0)
        ) + int(recovered.get("recovered_count", 0))
        _SUPERVISOR_RUNTIME["last_step_summary"] = {
            "timestamp": now_iso,
            "trigger_source": trigger_source,
            "preview": step_out.get("preview", not execute_live),
            "dispatched": step_out.get("dispatched", []),
            "would_dispatch": step_out.get("would_dispatch", []),
            "failed": step_out.get("failed", []),
            "rate_limited": step_out.get("rate_limited", []),
            "blocked_by_scope_overlap": step_out.get("blocked_by_scope_overlap", []),
            "paused": step_out.get("paused_for_reset", []),
            "deferred_for_codex": step_out.get("deferred_for_codex", []),
            "blocked_by_dependencies": step_out.get("blocked_by_dependencies", []),
            "resumed_from_pause": step_out.get("resumed_from_pause", 0),
            "resumed_from_codex_reset": step_out.get("resumed_from_codex_reset", []),
            "codex_authoritatively_refreshed": step_out.get("codex_authoritatively_refreshed", False),
            "recovered_orphans": recovered,
        }
        _RUNTIME_STATE["allocation_credits"] = get_allocation_credits()
        if step_out.get("dispatched"):
            _save_state()
        return step_out


def _autonomous_supervisor_loop() -> None:
    """
    Continuous background daemon loop that monitors SQLite `backlog_jobs`, recovers orphaned
    RUNNING workers after restarts, resumes `PAUSED_WAIT_RESET` / `DEFERRED_WAIT_CODEX_RESET` jobs
    on timer or authoritative quota refresh, and dispatches ready `QUEUED` jobs automatically
    without requiring manual `/api/queue/step` calls.
    """
    _SUPERVISOR_RUNTIME["started_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Immediate crash-recovery pass on service startup for any orphaned RUNNING jobs from prior process
    try:
        live_act = collect_live_activity()
        alive_pids = {int(w["pid"]) for w in (live_act.get("workers") or []) if w.get("pid")}
        rec = QUEUE_MANAGER.recover_orphaned_jobs(alive_pids=alive_pids)
        _SUPERVISOR_RUNTIME["total_orphans_recovered"] = int(
            _SUPERVISOR_RUNTIME.get("total_orphans_recovered", 0)
        ) + int(rec.get("recovered_count", 0))
    except Exception as exc:
        _SUPERVISOR_RUNTIME["last_error"] = f"startup_recovery: {exc}"

    while not _SUPERVISOR_STOP.is_set():
        cfg = _RUNTIME_STATE.get("supervisor_config") or {}
        interval = max(0.5, min(300.0, float(cfg.get("interval_sec", 2.0))))
        enabled = bool(cfg.get("enabled", True))
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _SUPERVISOR_RUNTIME["tick_count"] = int(_SUPERVISOR_RUNTIME.get("tick_count", 0)) + 1
        _SUPERVISOR_RUNTIME["last_tick_at"] = now_iso

        if enabled:
            try:
                q_summary = QUEUE_MANAGER.get_queue_summary(limit=5)
                counts = q_summary.get("counts") or {}
                has_pending = (
                    int(counts.get("QUEUED", 0)) > 0
                    or int(counts.get("RUNNING", 0)) > 0
                    or int(counts.get("PAUSED_WAIT_RESET", 0)) > 0
                    or int(counts.get("DEFERRED_WAIT_CODEX_RESET", 0)) > 0
                )
                if has_pending:
                    _run_supervisor_cycle(
                        max_jobs=max(1, int(cfg.get("max_jobs_per_tick", 2))),
                        execute_live=bool(cfg.get("execute_live", False)),
                        trigger_source="autonomous_background_loop",
                    )
                if is_scheduled_review_due():
                    # Discovery does network/CLI calls: run it in its own thread, never inside the supervisor tick.
                    if _start_review_job(trigger="scheduled") is not None:
                        _SUPERVISOR_RUNTIME["last_model_review_at"] = now_iso
                if policy_doctor.is_weekly_policy_audit_due():
                    threading.Thread(
                        target=policy_doctor.run_policy_doctor_audit,
                        kwargs={"save_state": True, "trigger": "weekly_scheduled"},
                        name="policy-doctor-weekly",
                        daemon=True,
                    ).start()
                    _SUPERVISOR_RUNTIME["last_policy_doctor_at"] = now_iso
            except Exception as exc:
                _SUPERVISOR_RUNTIME["last_error"] = f"{now_iso}: {exc}"

        _SUPERVISOR_STOP.wait(interval)


_REVIEW_JOB_LOCK = threading.Lock()
_REVIEW_JOB: Dict[str, Any] = {"run_id": None, "status": "idle"}
_REVIEW_JOB_SEQ = 0


def _review_job_snapshot() -> Dict[str, Any]:
    with _REVIEW_JOB_LOCK:
        return {k: v for k, v in _REVIEW_JOB.items() if k != "result"}


def _start_review_job(
    trigger: str = "manual",
    settings_override: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Start a model review in a worker thread and return immediately with the job status.
    Returns None when a review is already running (single-flight).
    """
    global _REVIEW_JOB_SEQ
    with _REVIEW_JOB_LOCK:
        if _REVIEW_JOB.get("status") == "running":
            return None
        _REVIEW_JOB_SEQ += 1
        run_id = f"review-{_REVIEW_JOB_SEQ}-{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
        _REVIEW_JOB.clear()
        _REVIEW_JOB.update(
            {
                "run_id": run_id,
                "status": "running",
                "trigger": trigger,
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    def _worker() -> None:
        try:
            result = run_model_review(
                dispatch_history=_RUNTIME_STATE.get("dispatch_history"),
                db_path=COST_LEARNER.db_path,
                scorecard=COST_LEARNER.get_agent_scorecard(),
                settings_override=settings_override,
                force_refresh_metadata=force_refresh,
            )
            with _REVIEW_JOB_LOCK:
                if _REVIEW_JOB.get("run_id") == run_id:
                    _REVIEW_JOB.update(
                        status="done",
                        result=result,
                        error=None,
                        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    )
        except Exception as exc:  # noqa: BLE001
            with _REVIEW_JOB_LOCK:
                if _REVIEW_JOB.get("run_id") == run_id:
                    _REVIEW_JOB.update(
                        status="error",
                        error=str(exc),
                        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    )

    threading.Thread(target=_worker, name=f"model-review-{run_id}", daemon=True).start()
    return _review_job_snapshot()


def _ensure_supervisor_thread_started() -> None:
    """(Re)start the supervisor thread, but only inside a running server lifespan and when not disabled."""
    global _SUPERVISOR_THREAD
    if not _LIFESPAN_ACTIVE or os.environ.get("QUOTA_HUB_DISABLE_SUPERVISOR") == "1":
        return
    if _SUPERVISOR_THREAD is None or not _SUPERVISOR_THREAD.is_alive():
        _SUPERVISOR_STOP.clear()
        _SUPERVISOR_THREAD = threading.Thread(
            target=_autonomous_supervisor_loop,
            name="quota-hub-autonomous-supervisor",
            daemon=True,
        )
        _SUPERVISOR_THREAD.start()


@app.get("/api/supervisor/status")
def api_get_supervisor_status() -> Dict[str, Any]:
    """Return autonomous background supervisor status, tick counters, and last step summary."""
    _ensure_supervisor_thread_started()
    return {
        "supervisor": _get_supervisor_status(),
        "queue_summary": QUEUE_MANAGER.get_queue_summary(limit=20),
    }


@app.post("/api/supervisor/config")
def api_set_supervisor_config(req: SupervisorConfigRequest) -> Dict[str, Any]:
    """Update autonomous background supervisor settings (enabled, interval_sec, execute_live, max_jobs_per_tick)."""
    cfg = _RUNTIME_STATE.setdefault(
        "supervisor_config",
        {"enabled": True, "interval_sec": 2.0, "execute_live": False, "max_jobs_per_tick": 2},
    )
    if req.enabled is not None:
        cfg["enabled"] = bool(req.enabled)
    if req.interval_sec is not None:
        cfg["interval_sec"] = max(0.5, min(300.0, float(req.interval_sec)))
    if req.execute_live is not None:
        cfg["execute_live"] = bool(req.execute_live)
    if req.max_jobs_per_tick is not None:
        cfg["max_jobs_per_tick"] = max(1, min(10, int(req.max_jobs_per_tick)))
    _save_state()
    _ensure_supervisor_thread_started()
    return {
        "supervisor": _get_supervisor_status(),
        "queue_summary": QUEUE_MANAGER.get_queue_summary(limit=20),
    }


@app.get("/api/queue")
def api_get_queue() -> Dict[str, Any]:
    """Return persistent SQLite backlog queue summary, recent jobs, and autonomous supervisor state."""
    out = QUEUE_MANAGER.get_queue_summary(limit=50)
    out["autonomous_supervisor"] = _get_supervisor_status()
    return out


@app.post("/api/queue/enqueue")
def api_enqueue_backlog(req: BacklogEnqueueRequest) -> Dict[str, Any]:
    """
    Enqueue approved backlog tasks into persistent SQLite `backlog_jobs` table (all-or-nothing).
    Invalid batches (self/cyclic/unknown dependencies, non-list fields) are rejected with HTTP 400.
    With `clear_existing`, validation and the replace share one transaction, so a rejected batch
    leaves the existing queue intact.
    """
    _ensure_supervisor_thread_started()
    try:
        result = QUEUE_MANAGER.enqueue_batch(
            req.tasks, preserve_existing=req.preserve_existing, clear_existing=req.clear_existing
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    inserted = result["inserted"]
    return {
        "cleared": result["cleared"],
        "enqueued_count": len(inserted),
        "inserted": inserted,
        "queue_summary": QUEUE_MANAGER.get_queue_summary(limit=20),
        "autonomous_supervisor": _get_supervisor_status(),
    }


@app.post("/api/queue/recover")
def api_recover_orphaned_jobs() -> Dict[str, Any]:
    """Recover orphaned RUNNING jobs whose worker PID is no longer alive (crash recovery)."""
    live_act = collect_live_activity()
    alive_pids = {int(w["pid"]) for w in (live_act.get("workers") or []) if w.get("pid")}
    return QUEUE_MANAGER.recover_orphaned_jobs(alive_pids=alive_pids)


@app.post("/api/queue/step")
def api_step_queue_supervisor(req: BacklogStepRequest) -> Dict[str, Any]:
    """
    Execute one supervisor cycle over the persistent SQLite backlog queue on demand.
    Shares `_SUPERVISOR_LOCK` with the autonomous background supervisor loop.
    """
    return _run_supervisor_cycle(
        max_jobs=req.max_jobs,
        execute_live=req.execute_live,
        trigger_source="manual_api_step",
    )


@app.get("/api/benchmark")
def api_run_benchmark(num_tasks: int = 100) -> Dict[str, Any]:
    """
    Run the mathematical multi-day backlog benchmark comparing `reset_aware_v2` (including weight sensitivity sweep)
    against `highest_remaining` and `round_robin` in both Endurance and Sprint modes.
    """
    num_tasks = max(10, min(300, int(num_tasks)))
    return {
        "endurance": run_multi_day_benchmark(num_tasks=num_tasks, operating_mode="endurance"),
        "sprint": run_multi_day_benchmark(num_tasks=num_tasks, operating_mode="sprint"),
    }


@app.get("/api/history")
def api_get_history() -> Dict[str, Any]:
    """Return recent work package routing and dispatch logs."""
    return {"history": _RUNTIME_STATE.get("dispatch_history", [])}


@app.get("/api/policy")
def api_get_policy() -> Dict[str, Any]:
    """Return the effective orchestrator policy and its source (policy.json or implementation defaults)."""
    return load_orchestrator_policy()


@app.get("/api/witness")
def api_get_witness() -> Dict[str, Any]:
    """Advisory Witness health classification of active jobs and live workers. Never kills or duplicates."""
    return QUEUE_MANAGER.get_witness_report(
        live_activity=collect_live_activity(),
        policy=load_orchestrator_policy(),
    )


@app.post("/api/queue/claim")
def api_claim_native_lease(req: NativeLeaseClaimRequest) -> Dict[str, Any]:
    """Atomically claim a concurrency and edit-scope lease in SQLite for a Category A native subagent."""
    try:
        return QUEUE_MANAGER.claim_native_lease(
            req.job_id,
            provider=req.provider,
            task=req.task,
            pool_id=req.pool_id,
            model_id=req.model_id,
            workspace=req.workspace,
            edit_scope=req.edit_scope,
            read_only=req.read_only,
            kind=req.kind,
            session_id=req.session_id,
            worker_pid=req.worker_pid,
            live_activity=collect_live_activity(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/queue/release")
def api_release_native_lease(req: NativeLeaseReleaseRequest) -> Dict[str, Any]:
    """Release a Category A native subagent lease upon completion or failure."""
    try:
        return QUEUE_MANAGER.release_native_lease(
            req.job_id,
            run_id=req.run_id,
            status=req.status,
            output_summary=req.output_summary,
            session_id=req.session_id,
            worker_pid=req.worker_pid,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/refinery/audit")
def api_refinery_audit(workspace: Optional[str] = None) -> Dict[str, Any]:
    """Audit COMPLETED writer jobs lacking ACCEPTED status and dirty workspace files lacking ACCEPTED hashes."""
    return QUEUE_MANAGER.audit_unreviewed_work(workspaces=[workspace] if workspace else None)


@app.post("/api/refinery/submit")
def api_refinery_submit(req: RefinerySubmitRequest) -> Dict[str, Any]:
    """Scope-check a COMPLETED job's changed files and move it to PENDING_REVIEW (or back to rework)."""
    try:
        return QUEUE_MANAGER.submit_for_review(
            req.job_id,
            changed_files=req.changed_files,
            checks_passed=req.checks_passed,
            implementer_agent_id=req.implementer_agent_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/refinery/verdict")
def api_refinery_verdict(req: RefineryVerdictRequest) -> Dict[str, Any]:
    """Record an independent review verdict; the reviewer must differ from the implementer."""
    try:
        return QUEUE_MANAGER.record_review_verdict(
            req.job_id,
            reviewer_agent_id=req.reviewer_agent_id,
            accepted=req.accepted,
            findings=req.findings,
            reviewer_family=req.reviewer_family,
            review_session_id=req.review_session_id,
            require_verified_session=req.require_verified_session,
            cost_learner=COST_LEARNER,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ModelDecisionRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to approve, put on probation, reject, promote, or retire")
    decision: str = Field(..., description="'approve' | 'probation' | 'reject' | 'promote' | 'retire'")
    tier: Optional[str] = Field(None, description="Optional tier override: 'LOW' | 'MID' | 'HIGH'")
    provider: Optional[str] = Field(None, description="Optional provider hint: 'agy' | 'claude' | 'codex'")


@app.get("/api/models/review")
def api_get_models_review() -> Dict[str, Any]:
    """Return latest model discovery proposals, excluded expensive models, sources, last run, staleness, and settings."""
    try:
        out = get_latest_model_review(scorecard=COST_LEARNER.get_agent_scorecard())
        out["catalog_status"] = get_catalog_status()
        out["review_run"] = _review_job_snapshot()
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/review/run")
async def api_run_models_review(request: Request) -> Dict[str, Any]:
    """Start a manual model review in a worker thread; returns immediately with a run id (poll GET /api/models/review)."""
    body: Dict[str, Any] = {}
    try:
        raw = await request.body()
        if raw and raw.strip():
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                body = parsed
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    settings_override = body.get("settings") if isinstance(body.get("settings"), dict) else None
    if settings_override is None:
        setting_keys = {"max_price_ratio", "review_interval_days", "probation_share", "auto_retire_after_days_unused"}
        extracted = {k: body[k] for k in setting_keys if k in body}
        if extracted:
            settings_override = extracted

    try:
        if settings_override is not None:
            validate_review_settings(settings_override, base=get_review_settings())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = _start_review_job(
        trigger="manual",
        settings_override=settings_override,
        force_refresh=bool(body.get("force_refresh", False)),
    )
    if started is None:
        return {"already_running": True, "review_run": _review_job_snapshot()}
    return {"already_running": False, "review_run": started}


@app.post("/api/models/review/settings")
async def api_update_models_review_settings(request: Request) -> Dict[str, Any]:
    """Persist owner model review settings (`max_price_ratio`, `review_interval_days`, `probation_share`, `auto_retire_after_days_unused`)."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    try:
        updated = update_review_settings(body)
        _RUNTIME_STATE["model_review_settings"] = dict(updated)
        _save_state()
        return get_latest_model_review(scorecard=COST_LEARNER.get_agent_scorecard())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/decision")
def api_post_model_decision(req: ModelDecisionRequest) -> Dict[str, Any]:
    """
    Record an owner model review decision (`approve` | `probation` | `reject` | `promote` | `retire`).
    Reject = 30-day cooldown for that model_id. Approve/probation writes `model_catalog.json` atomically with backup.
    """
    try:
        res = apply_model_decision(
            model_id=req.model_id,
            decision=req.decision,
            tier=req.tier,
            provider=req.provider,
        )
        res["review"] = get_latest_model_review(scorecard=COST_LEARNER.get_agent_scorecard())
        return res
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/policy-doctor")
def api_get_policy_doctor() -> Dict[str, Any]:
    """Return latest Policy Doctor contradiction & compaction audit + active orchestrator/subagent rules summary."""
    return policy_doctor.get_latest_policy_doctor_report(auto_run_if_due=True)


@app.post("/api/policy-doctor/run")
def api_run_policy_doctor() -> Dict[str, Any]:
    """Run an immediate Policy Doctor contradiction & compaction audit and persist state."""
    return policy_doctor.run_policy_doctor_audit(save_state=True, trigger="manual_ui")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> Any:
    """Serve the Web Dashboard UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(str(index_file))
