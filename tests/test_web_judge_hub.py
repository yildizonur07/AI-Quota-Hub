"""
Unit & Integration Tests for Web Judge Hub Seats, Zero-API-Cost Guarantees,
Cross-Family Review Enforcement, Unknown Quota Invariants, and Limit Event Signals.

Verifies all four mission areas:
1. WEB_JUDGE_SEATS (web:gemini -> family gemini, web:chatgpt -> family openai) are
   zero-API-cost judge seats (zero_api_cost=True, predicted_cost 0.0).
2. route_work_package enforces cross-family review for web judge seats
   (e.g. implementer_family='gemini' with require_cross_family=True rejects web:gemini
   and routes to web:chatgpt; implementer_family='openai' routes to web:gemini).
3. Unknown web quota stays 'UNKNOWN' (remaining_fraction is None, used_percent is None,
   never 0% or 0.0).
4. record_web_judge_event and POST /api/web-judge/event record WEB_JUDGE_CALL_COMPLETED
   and SURFACE_LIMIT_REACHED events, and a surface with SURFACE_LIMIT_REACHED transitions
   to status='exhausted' (rate_limited=True) and is rejected by dispatch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app
from orchestrator import (
    WEB_JUDGE_SEATS,
    get_web_judge_status,
    load_web_judge_state,
    record_web_judge_event,
    reset_web_judge_state,
    route_work_package,
)


@pytest.fixture(autouse=True)
def clean_web_judge_state():
    """Ensure every test runs with clean web judge state."""
    reset_web_judge_state()
    yield
    reset_web_judge_state()


# ---------------------------------------------------------------------------
# Area 1: Zero-API-cost judge seats (zero_api_cost=True, predicted_cost 0.0)
# ---------------------------------------------------------------------------


def test_web_judge_seats_configuration():
    """Verify WEB_JUDGE_SEATS definitions have zero_api_cost=True and predicted_cost 0.0."""
    seats_by_model = {s["model_id"]: s for s in WEB_JUDGE_SEATS}
    assert "web:gemini" in seats_by_model
    assert "web:chatgpt" in seats_by_model

    gemini_seat = seats_by_model["web:gemini"]
    assert gemini_seat["provider"] == "web"
    assert gemini_seat["surface"] == "gemini"
    assert gemini_seat["family"] == "gemini"
    assert gemini_seat["tier"] == "HIGH"
    assert gemini_seat["zero_api_cost"] is True
    assert gemini_seat["predicted_cost"] == 0.0

    chatgpt_seat = seats_by_model["web:chatgpt"]
    assert chatgpt_seat["provider"] == "web"
    assert chatgpt_seat["surface"] == "chatgpt"
    assert chatgpt_seat["family"] == "openai"
    assert chatgpt_seat["tier"] == "HIGH"
    assert chatgpt_seat["zero_api_cost"] is True
    assert chatgpt_seat["predicted_cost"] == 0.0


def test_web_judge_status_reports_zero_api_cost():
    """Verify get_web_judge_status() reports global zero_api_cost and seat structures."""
    status = get_web_judge_status()
    assert status["zero_api_cost"] is True
    assert status["quota_telemetry"] == "UNKNOWN"

    for seat in status["seats"]:
        assert seat["zero_api_cost"] is True
        assert seat["predicted_cost"] == 0.0


def test_routed_web_judge_assignment_zero_cost():
    """Verify routed assignment record for a web judge seat has zero_api_cost=True and 0.0 predicted cost."""
    task = {
        "task": "Review PR #42 security patches",
        "kind": "review",
        "preferred_provider": "web",
        "allow_web_judge": True,
        "explicit_model": "web:gemini",
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    assert routed["status"] == "routed"
    ar = routed["assignment_record"]
    assert ar["provider"] == "web"
    assert ar["model"] == "web:gemini"
    assert ar["family"] == "gemini"
    assert ar["zero_api_cost"] is True

    cost_pred = ar["predicted_cost"]
    assert cost_pred["zero_api_cost"] is True
    assert cost_pred["five_hour_mean"] == 0.0
    assert cost_pred["weekly_mean"] == 0.0
    assert cost_pred["predicted_cost"] == 0.0
    assert ar["math_breakdown"]["predicted_5h_cost"] == 0.0
    assert ar["math_breakdown"]["predicted_weekly_cost"] == 0.0


# ---------------------------------------------------------------------------
# Area 2: Cross-family review enforcement for web judge seats
# ---------------------------------------------------------------------------


def test_cross_family_review_gemini_implementer_routes_to_chatgpt():
    """Implementer family gemini with require_cross_family=True must reject web:gemini and route to web:chatgpt."""
    task = {
        "task": "Review Gemini DSP architecture changes",
        "kind": "review",
        "implementer_family": "gemini",
        "require_cross_family": True,
        "preferred_provider": "web",
        "allow_web_judge": True,
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    assert routed["status"] == "routed"
    ar = routed["assignment_record"]
    assert ar["provider"] == "web"
    assert ar["model"] == "web:chatgpt"
    assert ar["family"] == "openai"

    # web:gemini must be in rejected_models with cross-family review reason
    rejections = {r["model_id"]: r["reason"] for r in routed.get("rejected_models", [])}
    assert "web:gemini" in rejections
    assert "Cross-family review required" in rejections["web:gemini"]
    assert "gemini" in rejections["web:gemini"]


def test_cross_family_review_openai_implementer_routes_to_gemini():
    """Implementer family openai with require_cross_family=True must reject web:chatgpt and route to web:gemini."""
    task = {
        "task": "Review OpenAI kernel refactor",
        "kind": "review",
        "implementer_family": "openai",
        "require_cross_family": True,
        "preferred_provider": "web",
        "allow_web_judge": True,
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    assert routed["status"] == "routed"
    ar = routed["assignment_record"]
    assert ar["provider"] == "web"
    assert ar["model"] == "web:gemini"
    assert ar["family"] == "gemini"

    rejections = {r["model_id"]: r["reason"] for r in routed.get("rejected_models", [])}
    assert "web:chatgpt" in rejections
    assert "Cross-family review required" in rejections["web:chatgpt"]


def test_cross_family_not_required_allows_same_family():
    """When require_cross_family=False, same family web seat is not rejected by cross-family rule."""
    task = {
        "task": "Review routine change",
        "kind": "review",
        "implementer_family": "gemini",
        "require_cross_family": False,
        "explicit_model": "web:gemini",
        "allow_web_judge": True,
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    assert routed["status"] == "routed"
    assert routed["assignment_record"]["model"] == "web:gemini"


def test_explicit_same_family_rejected_when_cross_family_required():
    """Explicit model matching implementer family is rejected if cross-family review is required."""
    task = {
        "task": "Review PR with forced same family",
        "kind": "review",
        "implementer_family": "gemini",
        "require_cross_family": True,
        "explicit_model": "web:gemini",
        "allow_web_judge": True,
        "excluded_providers": ["agy", "claude", "codex"],
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    # Since web:gemini was explicit but rejected by cross-family, it must not be routed to web:gemini
    rejections = {r["model_id"]: r["reason"] for r in routed.get("rejected_models", [])}
    assert "web:gemini" in rejections
    assert "Cross-family review required" in rejections["web:gemini"]


def test_web_judge_rejected_for_non_review_kind():
    """Web judge seats are strictly review/judge only; implement kind rejects them."""
    task = {
        "task": "Implement data pipeline",
        "kind": "implement",
        "allow_web_judge": True,
        "explicit_model": "web:gemini",
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    rejections = {r["model_id"]: r["reason"] for r in routed.get("rejected_models", [])}
    assert "web:gemini" in rejections
    assert "review/judge only" in rejections["web:gemini"]


# ---------------------------------------------------------------------------
# Area 3: Unknown web quota stays 'UNKNOWN' (never 0% or 0.0)
# ---------------------------------------------------------------------------


def test_web_judge_initial_quota_is_unknown():
    """Web judge quota must be UNKNOWN with remaining_fraction and used_percent None, never 0% or 0.0."""
    st = get_web_judge_status()
    for surf in ("gemini", "chatgpt"):
        data = st["surfaces"][surf]
        assert data["remaining_fraction"] is None
        assert data["used_percent"] is None
        assert data["status"] == "UNKNOWN"
        assert data["remaining_fraction"] is not 0.0
        assert data["used_percent"] is not 0.0
        assert data["used_percent"] is not 0


def test_load_web_judge_state_enforces_none_quota():
    """Restoring state with fabricated 0.0 or 100% quota values must overwrite them to None."""
    corrupted_state = {
        "surfaces": {
            "gemini": {
                "remaining_fraction": 0.0,
                "used_percent": 100.0,
                "rate_limited": False,
            },
            "chatgpt": {
                "remaining_fraction": 1.0,
                "used_percent": 0.0,
                "rate_limited": False,
            },
        }
    }
    load_web_judge_state(corrupted_state)
    st = get_web_judge_status()
    for surf in ("gemini", "chatgpt"):
        entry = st["surfaces"][surf]
        assert entry["remaining_fraction"] is None
        assert entry["used_percent"] is None
        assert entry["status"] == "UNKNOWN"


def test_record_event_preserves_unknown_quota():
    """Recording completed or limit events preserves None for remaining_fraction and used_percent."""
    record_web_judge_event({"surface": "gemini", "event_type": "WEB_JUDGE_CALL_COMPLETED"})
    st1 = get_web_judge_status()
    assert st1["gemini"]["remaining_fraction"] is None
    assert st1["gemini"]["used_percent"] is None

    record_web_judge_event({"surface": "gemini", "event_type": "SURFACE_LIMIT_REACHED"})
    st2 = get_web_judge_status()
    assert st2["gemini"]["remaining_fraction"] is None
    assert st2["gemini"]["used_percent"] is None


def test_candidate_web_judge_quota_stays_unknown():
    """Candidate entry in route_work_package has remaining_quota=None and quota_state='UNKNOWN'."""
    task = {
        "task": "Review PR #99",
        "kind": "review",
        "allow_web_judge": True,
        "explicit_model": "web:chatgpt",
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    ar = routed["assignment_record"]
    assert ar["remaining_quota"] is None
    assert ar["quota_state"] == "UNKNOWN"


def test_api_quotas_includes_web_judges_with_unknown_quota():
    """GET /api/quotas returns web_judges telemetry with unknown quota."""
    client = TestClient(app)
    resp = client.get("/api/quotas")
    assert resp.status_code == 200
    data = resp.json()
    assert "web_judges" in data
    wj = data["web_judges"]
    assert wj["quota_telemetry"] == "UNKNOWN"
    assert wj["gemini"]["remaining_fraction"] is None
    assert wj["gemini"]["used_percent"] is None
    assert wj["chatgpt"]["remaining_fraction"] is None
    assert wj["chatgpt"]["used_percent"] is None


# ---------------------------------------------------------------------------
# Area 4: Events recording & SURFACE_LIMIT_REACHED status='exhausted' & dispatch rejection
# ---------------------------------------------------------------------------


def test_record_web_judge_completed_event():
    """record_web_judge_event records WEB_JUDGE_CALL_COMPLETED and updates call statistics."""
    ev = {
        "surface": "gemini",
        "event_type": "WEB_JUDGE_CALL_COMPLETED",
        "call_id": "call-001",
        "duration_seconds": 12.4,
        "message": "Verdict accepted cleanly",
    }
    status = record_web_judge_event(ev)
    gem = status["gemini"]
    assert gem["calls_count"] == 1
    assert gem["rate_limited"] is False
    assert gem["status"] == "UNKNOWN"
    assert gem["last_call_at"] is not None
    assert gem["last_message"] == "Verdict accepted cleanly"
    assert len(gem["recent_events"]) == 1
    assert gem["recent_events"][0]["event_type"] == "WEB_JUDGE_CALL_COMPLETED"


def test_record_surface_limit_reached_transitions_to_exhausted():
    """SURFACE_LIMIT_REACHED event transitions surface to status='exhausted' with rate_limited=True."""
    ev = {
        "surface": "gemini",
        "event_type": "SURFACE_LIMIT_REACHED",
        "call_id": "call-limit-1",
        "reset_hint": "2026-11-03T18:00:00Z",
        "message": "Temporary rate limit reached on web surface",
    }
    status = record_web_judge_event(ev)
    gem = status["gemini"]
    assert gem["limit_events_count"] == 1
    assert gem["rate_limited"] is True
    assert gem["status"] == "exhausted"
    assert gem["reset_hint"] == "2026-11-03T18:00:00Z"
    assert gem["last_limit_at"] is not None

    # chatgpt must remain unaffected
    chat = status["chatgpt"]
    assert chat["rate_limited"] is False
    assert chat["status"] == "UNKNOWN"


def test_post_web_judge_event_endpoint():
    """POST /api/web-judge/event endpoint records events and returns updated surface status."""
    client = TestClient(app)

    # 1. Record completed call on chatgpt
    r1 = client.post(
        "/api/web-judge/event",
        json={
            "surface": "chatgpt",
            "event_type": "WEB_JUDGE_CALL_COMPLETED",
            "call_id": "call-cg-1",
            "prompt_chars": 500,
            "reply_chars": 1200,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["recorded"] is True
    assert d1["surface"] == "chatgpt"
    assert d1["rate_limited"] is False
    assert d1["status"] == "UNKNOWN"
    assert d1["web_judges"]["chatgpt"]["calls_count"] == 1

    # 2. Record SURFACE_LIMIT_REACHED on chatgpt
    r2 = client.post(
        "/api/web-judge/event",
        json={
            "surface": "web:chatgpt",
            "event_type": "SURFACE_LIMIT_REACHED",
            "reset_hint": "2026-11-03T20:00:00Z",
            "message": "Quota exhausted",
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["recorded"] is True
    assert d2["surface"] == "chatgpt"
    assert d2["rate_limited"] is True
    assert d2["status"] == "exhausted"
    assert d2["web_judges"]["chatgpt"]["status"] == "exhausted"


def test_surface_limit_reached_rejected_by_router():
    """A surface marked exhausted is rejected during routing."""
    record_web_judge_event({"surface": "gemini", "event_type": "SURFACE_LIMIT_REACHED"})

    task = {
        "task": "Review PR with exhausted gemini surface",
        "kind": "review",
        "explicit_model": "web:gemini",
        "allow_web_judge": True,
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    # web:gemini must be rejected because surface reported SURFACE_LIMIT_REACHED
    rejections = {r["model_id"]: r["reason"] for r in routed.get("rejected_models", [])}
    assert "web:gemini" in rejections
    assert "SURFACE_LIMIT_REACHED" in rejections["web:gemini"]
    assert "status=exhausted" in rejections["web:gemini"]


def test_dispatch_rejects_exhausted_web_surface():
    """POST /api/dispatch rejects requests targeting an exhausted web surface."""
    client = TestClient(app)

    # Mark gemini exhausted via API
    client.post(
        "/api/web-judge/event",
        json={"surface": "gemini", "event_type": "SURFACE_LIMIT_REACHED"},
    )

    # Dispatch targeting web:gemini
    disp_resp = client.post(
        "/api/dispatch",
        json={
            "task": "Review PR with exhausted web judge",
            "kind": "review",
            "explicit_model": "web:gemini",
            "allow_web_judge": True,
            "execute_live": False,
        },
    )
    assert disp_resp.status_code == 200
    disp_data = disp_resp.json()
    routing = disp_data["routing"]
    assert routing["status"] == "paused_quota_exhausted"
    assert routing["assignment_record"] is None
    assert disp_data["execution"] is None

    # Inspect rejection reasons
    rejections = {r["model_id"]: r["reason"] for r in routing.get("rejected_models", [])}
    assert "web:gemini" in rejections
    assert "status=exhausted" in rejections["web:gemini"]


def test_exhausted_surface_falls_back_to_available_web_surface():
    """When one web surface is exhausted, router can select the other non-exhausted web surface."""
    # Mark gemini exhausted
    record_web_judge_event({"surface": "gemini", "event_type": "SURFACE_LIMIT_REACHED"})

    # Task prefers web provider
    task = {
        "task": "Review PR fallback to available web surface",
        "kind": "review",
        "preferred_provider": "web",
        "allow_web_judge": True,
    }
    routed = route_work_package(task, providers_quota={}, orchestrator_info={"active": "auto"})
    assert routed["status"] == "routed"
    # chatgpt must be selected since gemini is exhausted
    assert routed["assignment_record"]["model"] == "web:chatgpt"
    assert routed["assignment_record"]["family"] == "openai"


def test_clear_limit_event_resets_exhausted_state():
    """WEB_JUDGE_CLEAR_LIMIT event resets rate_limited=False and status='UNKNOWN'."""
    record_web_judge_event({"surface": "gemini", "event_type": "SURFACE_LIMIT_REACHED"})
    assert get_web_judge_status()["gemini"]["status"] == "exhausted"

    record_web_judge_event({"surface": "gemini", "event_type": "WEB_JUDGE_CLEAR_LIMIT"})
    st = get_web_judge_status()
    assert st["gemini"]["rate_limited"] is False
    assert st["gemini"]["status"] == "UNKNOWN"
    assert st["gemini"]["reset_hint"] is None


def test_invalid_surface_raises_error():
    """Invalid surface in record_web_judge_event raises ValueError / HTTP 400."""
    with pytest.raises(ValueError, match="Unsupported web judge surface"):
        record_web_judge_event({"surface": "unsupported_engine"})

    client = TestClient(app)
    r = client.post("/api/web-judge/event", json={"surface": "unsupported_engine"})
    assert r.status_code == 400
