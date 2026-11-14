"""
Unit & Integration Tests for AI Quota Hub, Reset-Aware Multi-Provider Quota Scheduler,
Empirical Quota Cost Learner, Codex Banked Reset Strategy, and Task-Based AI Tool Selection.

Verifies:
1. AGY, Claude Code, and Codex telemetry parsers (`collectors/`), including confidence states
   (`VERIFIED`, `STALE`, `UNKNOWN`, `UNAVAILABLE`) and offline reset protection (`post_reset_unverified`).
2. Capability Floor vs. Reasoning Effort separation (`HIGH` floor defaults to `medium` effort;
   `high` effort only on critical/escalated/explicit request; `gemini-4-argon-high` not default).
3. Dual-window (`five_hour` + `weekly`) and 4-pool (`agy_gemini`, `agy_claude_gpt`, `claude_shared`, `codex_shared`)
   mathematical scheduling.
4. Critical Pre-Reset Harvest Scenario:
   - Provider A: 10% weekly remaining, weekly reset in 2 hours (7200s), 80% 5-hour remaining.
   - Provider B: 80% weekly remaining, weekly reset in 6 days (518400s), 80% 5-hour remaining.
   - Task requiring ~3% weekly quota MUST prefer Provider A over Provider B in both Endurance and Sprint modes.
5. Dual-Window Bottleneck Rejection:
   - Provider with 80% weekly remaining but 2% 5-hour remaining MUST be rejected when task's predicted 5h cost > 2%.
6. Empirical Quota Cost Learning (`QuotaCostLearner`):
   - `MEASURED_SINGLE_JOB`, `AMBIGUOUS_CONCURRENT`, `RESET_CROSSED`, and `ESTIMATED_DRY_RUN`.
7. Codex Banked Manual Resets & Schedule-Shift Opportunity Cost (`parse_codex_banked_resets`, `evaluate_banked_reset_strategy`).
8. Dynamic Codex Orchestrator Reserve (`compute_dynamic_codex_reserve`) & Degraded-Mode Continuity.
9. Task-Based AI Tool Selection (`recommend_task_tools`) through the optional tool catalog.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from fastapi.testclient import TestClient

from collectors.agy_collector import parse_agy_user_status
from collectors.codex_collector import (
    parse_codex_banked_resets,
    parse_codex_token_count_event,
)
from main import app
from orchestrator import (
    MODEL_CATALOG,
    build_headless_cli_command,
    compute_capability_floor,
    recommend_task_tools,
    route_work_package,
    select_active_orchestrator,
)
from scheduler import (
    BacklogQueueManager,
    QuotaCostLearner,
    build_pool_states,
    compute_dynamic_codex_reserve,
    evaluate_banked_reset_strategy,
    evaluate_candidate_schedule_score,
    run_multi_day_benchmark,
)


def test_parse_agy_user_status() -> None:
    now = datetime.datetime(2026, 10, 13, 20, 0, 0, tzinfo=datetime.timezone.utc)
    sample_payload = {
        "userStatus": {
            "planStatus": {
                "planInfo": {"planName": "Pro"},
                "availablePromptCredits": 500,
                "availableFlowCredits": 100,
            },
            "cascadeModelConfigData": {
                "clientModelConfigs": [
                    {
                        "label": "Gemini 3.8 Flash (High)",
                        "modelId": "gemini-3.8-flash-high",
                        "quotaInfo": {
                            "remainingFraction": 0.95,
                            "resetTime": "2026-10-13T21:00:00Z",
                        },
                    },
                    {
                        "label": "Gemini 4 Argon (Medium)",
                        "modelId": "gemini-4-argon-medium",
                        "quotaInfo": {
                            "remainingFraction": 0.40,
                            "resetTime": "2026-10-13T21:00:00Z",
                        },
                    },
                    {
                        "label": "Claude Sonnet 5.5 (High)",
                        "modelId": "claude-sonnet-5-5-high",
                        "quotaInfo": {
                            "remainingFraction": 1.0,
                            "resetTime": "2026-10-13T22:00:00Z",
                        },
                    },
                ]
            },
        }
    }
    parsed = parse_agy_user_status(sample_payload, now=now)
    assert parsed["status"] == "healthy"
    assert parsed["confidence"] == "VERIFIED"
    assert parsed["gemini_flash"] == 0.95
    assert parsed["gemini_pro"] == 0.40
    assert parsed["claude_sonnet"] == 1.0
    assert parsed["reset_in_sec"] == 3600


def test_parse_codex_token_count_event_and_unverified_reset_guard() -> None:
    now = datetime.datetime(2026, 10, 13, 20, 0, 0, tzinfo=datetime.timezone.utc)
    now_ts = int(now.timestamp())
    sample_event = {
        "timestamp": "2026-10-13T20:00:00.000Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {"total_tokens": 120000, "input_tokens": 100000, "output_tokens": 20000},
                "model_context_window": 272000,
            },
            "rate_limits": {
                "primary": {"used_percent": 15.0, "window_minutes": 300, "resets_at": now_ts + 5400},
                "secondary": {"used_percent": 30.0, "window_minutes": 10080, "resets_at": now_ts + 86400},
                "plan_type": "plus",
            },
        },
    }
    parsed = parse_codex_token_count_event(sample_event, now=now)
    assert parsed["status"] == "healthy"
    assert parsed["luna"] == 0.70
    assert parsed["terra"] == 0.70
    assert parsed["sol"] == 0.70
    assert parsed["reset_in_sec"] == 5400

    # Verify offline expired reset timestamp does NOT fabricate 0% used (100% remaining)
    expired_event = {
        "timestamp": "2026-10-13T10:00:00.000Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {},
            "rate_limits": {
                "primary": {"used_percent": 92.0, "window_minutes": 300, "resets_at": now_ts - 600},
                "secondary": {"used_percent": 50.0, "window_minutes": 10080, "resets_at": now_ts + 86400},
                "plan_type": "plus",
            },
        },
    }
    parsed_exp = parse_codex_token_count_event(expired_event, now=now)
    assert parsed_exp["windows"]["five_hour"]["post_reset_unverified"] is True
    assert parsed_exp["windows"]["five_hour"]["used_percent"] == 92.0  # preserved, not faked to 0.0!
    assert parsed_exp["confidence"] == "STALE"


def test_parse_claude_oauth_usage_payload() -> None:
    from collectors.claude_collector import parse_claude_oauth_usage_payload

    now = datetime.datetime(2026, 10, 13, 20, 0, 0, tzinfo=datetime.timezone.utc)
    sample_oauth = {
        "five_hour": {
            "utilization": 34.0,
            "resets_at": "2026-10-13T22:00:00+00:00",
        },
        "seven_day": {
            "utilization": 40.0,
            "resets_at": "2026-10-19T18:00:00+00:00",
        },
        "seven_day_opus": None,
        "seven_day_sonnet": None,
        "_subscription_type": "pro",
    }
    parsed = parse_claude_oauth_usage_payload(sample_oauth, now=now)
    assert parsed["status"] == "healthy"
    assert parsed["confidence"] == "VERIFIED"
    assert parsed["windows"]["five_hour"]["used_percent"] == 34.0
    assert parsed["windows"]["weekly"]["used_percent"] == 40.0
    assert parsed["sonnet"] == 0.60
    assert parsed["opus"] == 0.60
    assert parsed["haiku"] == 0.60
    assert parsed["shared_pool"] is True
    assert parsed["reset_in_sec"] == 7200


def test_capability_floor_and_effort_separation() -> None:
    floor, effort = compute_capability_floor(kind="scan", ambiguity="low", blast_radius="local", dependency_role="leaf")
    assert floor == "LOW"
    assert effort == "low"

    floor_h, effort_h = compute_capability_floor(
        kind="design",
        ambiguity="high",
        blast_radius="cross-module",
        dependency_role="upstream",
    )
    assert floor_h == "HIGH"
    assert effort_h == "medium"

    floor_crit, effort_crit = compute_capability_floor(
        kind="design",
        ambiguity="high",
        blast_radius="irreversible",
        dependency_role="upstream",
        critical=True,
        escalation_reason="Irreversible PLC safety interlock state machine design",
    )
    assert floor_crit == "HIGH"
    assert effort_crit == "high"

    floor_b, effort_b = compute_capability_floor(
        kind="implement",
        ambiguity="low",
        blast_radius="module",
        dependency_role="upstream",
    )
    assert floor_b == "HIGH"
    assert effort_b == "low"


def test_critical_pre_reset_harvest_scenario(tmp_path: Path) -> None:
    """
    Critical Scenario from specification:
    - Provider A (`claude_shared` — Claude Sonnet 5.5):
      * Weekly remaining quota: 10% (0.10).
      * Weekly reset in 2 hours (7200s).
      * Sufficient 5-hour capacity (80% remaining).
    - Provider B (`agy_gemini` / `agy_claude_gpt`):
      * Weekly remaining quota: 80% (0.80).
      * Weekly reset in 6 days (518400s).
      * Sufficient 5-hour capacity (80% remaining).
    A MID-tier task whose predicted weekly cost (~1%–2%) safely fits inside Provider A's 10%
    before reset MUST prefer Provider A over Provider B in both Endurance and Sprint modes,
    because Provider A's 10% weekly quota is about to expire ("stranded") in 2 hours whereas
    Provider B's 80% weekly quota must sustain 6 more days.
    """
    learner = QuotaCostLearner(tmp_path / "test_telemetry.sqlite")
    quotas = {
        "claude": {
            "status": "warning",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.10,
            "opus": 0.10,
            "haiku": 0.10,
            "windows": {
                "five_hour": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 7200},
                "weekly": {"used_percent": 90.0, "remaining_fraction": 0.10, "reset_in_sec": 7200},
            },
            "reset_in_sec": 7200,
        },
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": 0.80,
            "gemini_pro": 0.80,
            "claude_sonnet": 0.80,
            "claude_opus": 0.80,
            "windows": {
                "gemini_5h": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 14400},
                "gemini_weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 518400},
                "claude_gpt_5h": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 14400},
                "claude_gpt_weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 518400},
            },
            "reset_in_sec": 518400,
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": 0.80,
            "terra": 0.80,
            "sol": 0.80,
            "windows": {
                "five_hour": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 14400},
                "weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 518400},
            },
            "reset_in_sec": 518400,
        },
    }
    orch = select_active_orchestrator("auto", quotas)

    for mode in ("endurance", "sprint"):
        res = route_work_package(
            {
                "task": "WP-HARVEST-01: Implement bounded module before weekly reset",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "module",
                "dependency_role": "leaf",
                "context_need": "module",
            },
            quotas,
            orch,
            operating_mode=mode,
            cost_learner=learner,
        )
        assert res["status"] == "routed"
        winner = res["assignment_record"]
        # Provider A (claude_shared) must win because its 10% weekly quota resets in 2h (stranded ratio ~83%, Phi ~0.04)
        assert winner["provider"] == "claude", f"Expected Provider A (claude) in {mode} mode, got {winner['provider']}"
        assert winner["pool_id"] == "claude_shared"
        assert winner["math_breakdown"]["pre_reset_harvest_bonus"] > 0.15


def test_dual_window_five_hour_bottleneck_rejection(tmp_path: Path) -> None:
    """
    A job is NOT eligible merely because one window has high remaining quota:
    If Provider A (`claude_shared`) has 80% weekly remaining but only 2% (0.02) 5-hour remaining,
    and a task's predicted 5-hour upper consumption is ~5%, Provider A MUST be rejected.
    """
    learner = QuotaCostLearner(tmp_path / "test_bottleneck.sqlite")
    quotas = {
        "claude": {
            "status": "exhausted",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.02,
            "opus": 0.02,
            "haiku": 0.02,
            "windows": {
                "five_hour": {"used_percent": 98.0, "remaining_fraction": 0.02, "reset_in_sec": 10800},
                "weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 432000},
            },
            "reset_in_sec": 10800,
        },
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": 0.75,
            "gemini_pro": 0.75,
            "claude_sonnet": 0.75,
            "claude_opus": 0.75,
            "windows": {
                "gemini_5h": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 10800},
                "gemini_weekly": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 432000},
                "claude_gpt_5h": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 10800},
                "claude_gpt_weekly": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 432000},
            },
            "reset_in_sec": 10800,
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": 0.75,
            "terra": 0.75,
            "sol": 0.75,
            "windows": {
                "five_hour": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 10800},
                "weekly": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 432000},
            },
            "reset_in_sec": 10800,
        },
    }
    orch = select_active_orchestrator("auto", quotas)
    res = route_work_package(
        {
            "task": "WP-BOTTLENECK-01: Implement feature",
            "kind": "implement",
            "ambiguity": "med",
            "blast_radius": "module",
            "dependency_role": "leaf",
        },
        quotas,
        orch,
        cost_learner=learner,
    )
    assert res["status"] == "routed"
    assert res["assignment_record"]["provider"] != "claude"
    claude_rejections = [r for r in res["rejected_models"] if r["pool_id"] == "claude_shared"]
    assert len(claude_rejections) > 0
    assert any("5h window insufficient" in r["reason"] or ">=95%" in r["reason"] for r in claude_rejections)


def test_empirical_cost_learner_attribution_states(tmp_path: Path) -> None:
    """
    Verify QuotaCostLearner properly classifies:
    1. `MEASURED_SINGLE_JOB` -> updates empirical blend.
    2. `AMBIGUOUS_CONCURRENT` -> divides delta by concurrent_pool_jobs.
    3. `RESET_CROSSED` -> ignores negative/reset-crossed delta so cost estimates are not corrupted.
    """
    db_file = tmp_path / "learner.sqlite"
    learner = QuotaCostLearner(db_file)

    # Cold-start prior check
    prior = learner.predict_cost("claude_shared", "implement", "MID", "med")
    assert "COLD_START_PRIOR" in prior["prediction_source"]

    # 1. Record two clean single-job observations (each consumed 6.0% 5h and 2.0% weekly)
    for idx in range(2):
        obs = learner.record_observation(
            task_id=f"WP-SINGLE-{idx}",
            provider="claude",
            model_id="claude-sonnet-5-5",
            pool_id="claude_shared",
            task_kind="implement",
            tier="MID",
            complexity="med",
            started_at="2026-10-20T10:00:00Z",
            ended_at="2026-10-20T10:03:00Z",
            duration_sec=180.0,
            success=True,
            retry_count=0,
            concurrent_pool_jobs=1,
            pre_5h_used_pct=10.0,
            post_5h_used_pct=16.0,
            pre_weekly_used_pct=20.0,
            post_weekly_used_pct=22.0,
            pre_5h_resets_at="2026-10-20T14:00:00Z",
            post_5h_resets_at="2026-10-20T14:00:00Z",
            pre_weekly_resets_at="2026-10-26T18:00:00Z",
            post_weekly_resets_at="2026-10-26T18:00:00Z",
            predicted_5h_cost=0.04,
            predicted_weekly_cost=0.01,
            dry_run=False,
        )
        assert obs["attribution_status"] == "MEASURED_SINGLE_JOB"
        assert obs["measured_5h_cost"] == 0.06
        assert obs["measured_weekly_cost"] == 0.02

    blended = learner.predict_cost("claude_shared", "implement", "MID", "med")
    assert "EMPIRICAL_BLEND" in blended["prediction_source"]
    assert blended["five_hour_mean"] > prior["five_hour_mean"]

    # 2. Record concurrent job observation (2 jobs ran concurrently, total pool delta = 40% -> 20% split estimate)
    # Concurrent splits MUST NOT be stored as verified `measured_5h_cost` or distort empirical predictions!
    pre_conc_pred = learner.predict_cost("claude_shared", "implement", "MID", "med")
    obs_conc = learner.record_observation(
        task_id="WP-CONC-1",
        provider="claude",
        model_id="claude-sonnet-5-5",
        pool_id="claude_shared",
        task_kind="implement",
        tier="MID",
        complexity="med",
        started_at="2026-10-20T11:00:00Z",
        ended_at="2026-10-20T11:04:00Z",
        duration_sec=240.0,
        success=True,
        retry_count=0,
        concurrent_pool_jobs=2,
        pre_5h_used_pct=20.0,
        post_5h_used_pct=60.0,
        pre_weekly_used_pct=22.0,
        post_weekly_used_pct=42.0,
        pre_5h_resets_at="2026-10-20T14:00:00Z",
        post_5h_resets_at="2026-10-20T14:00:00Z",
        pre_weekly_resets_at="2026-10-26T18:00:00Z",
        post_weekly_resets_at="2026-10-26T18:00:00Z",
        predicted_5h_cost=0.04,
        predicted_weekly_cost=0.01,
        dry_run=False,
    )
    assert obs_conc["attribution_status"] == "AMBIGUOUS_CONCURRENT"
    assert obs_conc["measured_5h_cost"] is None
    assert obs_conc["measured_weekly_cost"] is None
    assert obs_conc["uncertain_split_5h_estimate"] == 0.20
    assert obs_conc["uncertain_split_weekly_estimate"] == 0.10

    post_conc_pred = learner.predict_cost("claude_shared", "implement", "MID", "med")
    assert post_conc_pred["five_hour_mean"] == pre_conc_pred["five_hour_mean"]
    assert post_conc_pred["sample_count"] == 2

    # 3. Record reset-crossed observation (usage dropped from 85% to 2% across 5h reset)
    obs_reset = learner.record_observation(
        task_id="WP-RESET-1",
        provider="claude",
        model_id="claude-sonnet-5-5",
        pool_id="claude_shared",
        task_kind="implement",
        tier="MID",
        complexity="med",
        started_at="2026-10-20T13:58:00Z",
        ended_at="2026-10-20T14:02:00Z",
        duration_sec=240.0,
        success=True,
        retry_count=0,
        concurrent_pool_jobs=1,
        pre_5h_used_pct=85.0,
        post_5h_used_pct=2.0,
        pre_weekly_used_pct=30.0,
        post_weekly_used_pct=31.0,
        pre_5h_resets_at="2026-10-20T14:00:00Z",
        post_5h_resets_at="2026-10-20T19:00:00Z",
        pre_weekly_resets_at="2026-10-26T18:00:00Z",
        post_weekly_resets_at="2026-10-26T18:00:00Z",
        predicted_5h_cost=0.04,
        predicted_weekly_cost=0.01,
        dry_run=False,
    )
    assert obs_reset["attribution_status"] == "RESET_CROSSED"
    assert obs_reset["measured_5h_cost"] is None


def test_codex_banked_resets_and_schedule_shift_strategy() -> None:
    now = datetime.datetime(2026, 10, 20, 20, 0, 0, tzinfo=datetime.timezone.utc)
    credits_list = [
        {
            "id": "RateLimitResetCredit_7d9d80497f888191ae8e58efd4e08c28",
            "credit_type": "Full reset (Weekly + 5 hr)",
            "granted_at": "2026-10-15T18:16:02.411955Z",
            "expires_at": "2026-10-22T18:16:02.411955Z",  # expires in ~46h (<72h!)
            "redeemed_at": None,
            "available": True,
            "expired": False,
        },
        {
            "id": "RateLimitResetCredit_038c32c38ec481919a6fac9b5fc52d51",
            "credit_type": "Full reset (Weekly + 5 hr)",
            "granted_at": "2026-10-15T16:29:11Z",
            "expires_at": "2026-10-29T16:29:11Z",
            "redeemed_at": None,
            "available": True,
            "expired": False,
        },
    ]
    parsed = parse_codex_banked_resets(
        usage_payload={"rate_limit_reset_credits": {"available_count": 2, "applicable_available_count": 2}},
        credits_payload={"available_count": 2, "credits": credits_list},
        now=now,
    )
    assert parsed["verification_status"] == "VERIFIED_API"
    assert parsed["available_count"] == 2
    assert parsed["applicable_available_count"] == 2
    assert parsed["earliest_expires_in_sec"] < (48 * 3600)

    codex_data = {
        "status": "healthy",
        "confidence": "VERIFIED",
        "windows": {
            "five_hour": {"used_percent": 35.0, "remaining_fraction": 0.65, "reset_in_sec": 7200},
            "weekly": {"used_percent": 40.0, "remaining_fraction": 0.60, "reset_in_sec": 172800},  # 2 days left -> 5 days shift loss
        },
        "banked_resets": parsed,
    }
    strat = evaluate_banked_reset_strategy(codex_data, operating_mode="endurance")
    assert strat["available_full_resets"] == 2
    assert strat["auto_trigger_permitted"] is False
    assert strat["expiring_soon_alert"] is True
    assert strat["schedule_shift_days_lost"] == 5.0


def test_dynamic_codex_orchestrator_reserve_protection(tmp_path: Path) -> None:
    """
    When Codex is getting low (e.g. 14% remaining) while AGY Gemini and Claude have healthy capacity,
    `compute_dynamic_codex_reserve` protects Codex's orchestration reserve (`routine_coding_allowed_on_codex = False`),
    and `route_work_package` routes routine leaf coding work to AGY/Claude instead of Codex.
    """
    learner = QuotaCostLearner(tmp_path / "test_reserve.sqlite")
    quotas = {
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": 0.85,
            "gemini_pro": 0.85,
            "claude_sonnet": 0.85,
            "claude_opus": 0.85,
            "windows": {
                "gemini_5h": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 14400},
                "gemini_weekly": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 432000},
                "claude_gpt_5h": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 14400},
                "claude_gpt_weekly": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 432000},
            },
            "reset_in_sec": 14400,
        },
        "claude": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.85,
            "opus": 0.85,
            "haiku": 0.85,
            "windows": {
                "five_hour": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 14400},
                "weekly": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 432000},
            },
            "reset_in_sec": 14400,
        },
        "codex": {
            "status": "warning",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": 0.12,
            "terra": 0.12,
            "sol": 0.12,
            "windows": {
                "five_hour": {"used_percent": 88.0, "remaining_fraction": 0.12, "reset_in_sec": 14400},
                "weekly": {"used_percent": 88.0, "remaining_fraction": 0.12, "reset_in_sec": 432000},
            },
            "reset_in_sec": 14400,
        },
    }
    reserve = compute_dynamic_codex_reserve(quotas, operating_mode="endurance", unfinished_tasks_count=6, pending_reviews_count=3)
    assert reserve["routine_coding_allowed_on_codex"] is False

    orch = select_active_orchestrator("auto", quotas)
    res = route_work_package(
        {
            "task": "WP-LEAF-01: Implement routine leaf helper",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "unfinished_tasks_count": 6,
            "pending_reviews_count": 3,
        },
        quotas,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
    )
    assert res["status"] == "routed"
    assert res["assignment_record"]["provider"] in {"agy", "claude"}


def test_quota_windows_and_cross_family_routing() -> None:
    quotas = {
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": 0.90,
            "gemini_pro": 0.90,
            "claude_sonnet": 0.90,
            "claude_opus": 0.90,
            "windows": {
                "gemini_5h": {"used_percent": 10.0, "remaining_fraction": 0.90, "reset_in_sec": 3600},
                "gemini_weekly": {"used_percent": 97.0, "remaining_fraction": 0.03, "reset_in_sec": 86400},
                "claude_gpt_5h": {"used_percent": 10.0, "remaining_fraction": 0.90, "reset_in_sec": 3600},
                "claude_gpt_weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 86400},
            },
            "reset_in_sec": 3600,
        },
        "claude": {
            "status": "warning",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.12,
            "opus": 0.12,
            "haiku": 0.12,
            "windows": {
                "five_hour": {"used_percent": 88.0, "remaining_fraction": 0.12, "reset_in_sec": 1800},
                "weekly": {"used_percent": 85.0, "remaining_fraction": 0.15, "reset_in_sec": 43200},
            },
            "reset_in_sec": 1800,
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": 0.85,
            "terra": 0.85,
            "sol": 0.85,
            "windows": {
                "five_hour": {"used_percent": 5.0, "remaining_fraction": 0.95, "reset_in_sec": 5400},
                "weekly": {"used_percent": 15.0, "remaining_fraction": 0.85, "reset_in_sec": 86400},
            },
            "reset_in_sec": 5400,
        },
    }
    orch = select_active_orchestrator("auto", quotas)
    assert orch["active"] in {"gpt-6.1-sol", "claude-opus-5-5-medium"}

    design_res = route_work_package(
        {
            "task": "Design multi-provider state machine",
            "kind": "design",
            "ambiguity": "high",
            "blast_radius": "cross-module",
            "dependency_role": "upstream",
        },
        quotas,
        orch,
    )
    assert design_res["status"] == "routed"
    rec = design_res["assignment_record"]
    assert rec["selected_tier"] == "HIGH"
    assert rec["reasoning_effort"] == "medium"
    assert rec["family"] != "gemini"

    healthy_quotas = {
        "agy": {"status": "healthy", "confidence": "VERIFIED", "freshness_age_sec": 2, "gemini_flash": 0.95, "gemini_pro": 0.95, "claude_sonnet": 0.90, "claude_opus": 0.90, "reset_in_sec": 3600},
        "claude": {"status": "healthy", "confidence": "VERIFIED", "freshness_age_sec": 2, "sonnet": 0.80, "opus": 0.80, "haiku": 0.80, "reset_in_sec": 3600},
        "codex": {"status": "healthy", "confidence": "VERIFIED", "freshness_age_sec": 2, "luna": 0.85, "terra": 0.85, "sol": 0.85, "reset_in_sec": 3600},
    }
    rev_res = route_work_package(
        {
            "task": "Review Gemini DSP module",
            "kind": "review",
            "ambiguity": "med",
            "blast_radius": "module",
            "dependency_role": "upstream",
            "implementer_family": "gemini",
        },
        healthy_quotas,
        orch,
    )
    assert rev_res["status"] == "routed"
    assert rev_res["assignment_record"]["family"] != "gemini"
    assert rev_res["assignment_record"]["capability_floor"] == "HIGH"


def test_unknown_quota_and_explicit_model_override() -> None:
    partial_quotas = {
        "agy": {"status": "healthy", "confidence": "VERIFIED", "freshness_age_sec": 2, "gemini_flash": 0.75, "gemini_pro": 0.75, "claude_sonnet": 0.75, "claude_opus": 0.75, "reset_in_sec": 3600},
        "claude": {"status": "offline"},
        "codex": {},
    }
    orch = select_active_orchestrator("auto", partial_quotas)
    res = route_work_package(
        {
            "task": "Implement bounded parser helper",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
        },
        partial_quotas,
        orch,
    )
    assert res["status"] == "routed"
    assert res["assignment_record"]["provider"] == "agy"
    codex_cands = [c for c in res["ranked_candidates"] if c["provider"] == "codex"]
    if codex_cands:
        assert codex_cands[0]["quota_state"] == "UNKNOWN"
        assert codex_cands[0]["remaining_quota"] is None

    explicit_res = route_work_package(
        {
            "task": "User requested specific model and effort",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "explicit_model": "gemini-4-argon-low",
            "explicit_effort": "low",
        },
        partial_quotas,
        orch,
    )
    assert explicit_res["assignment_record"]["model"] == "gemini-4-argon-low"
    assert explicit_res["assignment_record"]["reasoning_effort"] == "low"


def test_task_based_tool_selection_uses_optional_catalog(tmp_path, monkeypatch) -> None:
    none = recommend_task_tools({"task": "Look up the prior ADR decision", "kind": "design"})
    assert none["evaluated"] is True and none["selected_tools"] == [] and none["announcement"] is None

    layer = tmp_path / "layer"
    (layer / "quota-hub").mkdir(parents=True)
    (layer / "quota-hub" / "tool_catalog.py").write_text(
        "def recommend(task_request):\n"
        "    tool = {'tool': 'Probe', 'announcement': 'probe ' + task_request['task']}\n"
        "    return {'evaluated': True, 'selected_tools': [tool], 'announcement': tool['announcement'],\n"
        "            'skipped_blockers': [], 'closing_line': 'Probe used.'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OWNER_OVERRIDES_DIR", str(layer))
    got = recommend_task_tools({"task": "x"})
    assert [t["tool"] for t in got["selected_tools"]] == ["Probe"] and got["announcement"] == "probe x"


def test_headless_cli_builders() -> None:
    agy_m = next(m for m in MODEL_CATALOG if m["model_id"] == "gemini-4-argon-medium")
    _, display_agy = build_headless_cli_command(agy_m, "Run design review", "C:\\repo", effort="medium")
    assert "--model gemini-4-argon-medium" in display_agy
    assert "--effort" not in display_agy
    assert '-p="Run design review"' in display_agy

    codex_m = next(m for m in MODEL_CATALOG if m["model_id"] == "gpt-6.1-sol-medium")
    _, display_codex = build_headless_cli_command(codex_m, "Implement feature", "C:\\repo", effort="medium", read_only=False)
    assert "codex exec" in display_codex
    assert "-m gpt-6.1-sol" in display_codex
    assert "-s workspace-write" in display_codex


def _healthy_provider(**models: float) -> dict:
    win = lambda rem, reset: {"remaining_fraction": rem, "used_percent": round((1 - rem) * 100, 1), "reset_in_sec": reset}
    windows = {"five_hour": win(0.8, 14400), "weekly": win(0.8, 432000)}
    if "gemini_flash" in models:
        windows = {k: win(0.8, r) for k, r in (("gemini_5h", 14400), ("gemini_weekly", 432000),
                                               ("claude_gpt_5h", 14400), ("claude_gpt_weekly", 432000))}
    return {"status": "healthy", "confidence": "VERIFIED", "reset_in_sec": 14400, "windows": windows, **models}


def test_fastapi_endpoints_and_mode_persistence(monkeypatch) -> None:
    import main as hub_main

    # deterministic telemetry: routing must not depend on the quota of whoever runs the tests
    monkeypatch.setattr(hub_main, "collect_agy_quota", lambda: _healthy_provider(
        gemini_flash=0.8, gemini_pro=0.8, claude_sonnet=0.8, claude_opus=0.8))
    monkeypatch.setattr(hub_main, "collect_claude_quota", lambda: _healthy_provider(sonnet=0.8, opus=0.8, haiku=0.8))
    monkeypatch.setattr(hub_main, "collect_codex_quota", lambda **_: _healthy_provider(luna=0.8, terra=0.8, sol=0.8))
    monkeypatch.setattr(hub_main, "collect_live_activity", lambda: {"workers": [], "coordinators": []})
    client = TestClient(app)

    # 1. Switch to sprint mode & verify persistence in /api/quotas
    r_mode_sprint = client.post("/api/mode", json={"mode": "sprint", "unfinished_tasks_count": 5, "pending_reviews_count": 2})
    assert r_mode_sprint.status_code == 200
    assert r_mode_sprint.json()["operating_mode"] == "sprint"

    # Restore default endurance mode
    r_mode_end = client.post("/api/mode", json={"mode": "endurance", "unfinished_tasks_count": 4, "pending_reviews_count": 2})
    assert r_mode_end.status_code == 200
    data = r_mode_end.json()
    assert data["operating_mode"] == "endurance"
    assert "pool_states" in data
    assert "codex_reserve" in data
    assert "banked_reset_strategy" in data
    assert "cost_learning" in data

    # 2. Dispatch dry-run work package
    r_disp = client.post(
        "/api/dispatch",
        json={
            "task": "WP-TEST-01: Run unit test inventory",
            "prompt": "Scan tests directory",
            "kind": "scan",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "context_need": "pointer",
            "execute_live": False,
        },
    )
    assert r_disp.status_code == 200
    disp_data = r_disp.json()
    assert disp_data["routing"]["status"] == "routed"
    assert disp_data["execution"]["mode"] == "dry_run"
    assert disp_data["telemetry_observation"]["attribution_status"] == "ESTIMATED_DRY_RUN"


def test_reset_boundary_crossing_scheduling(tmp_path: Path) -> None:
    """
    Verify scheduling when a task starts shortly before a quota reset (e.g. 90s before reset)
    and finishes afterward (`estimated_duration_sec = 180s`).
    - Only the pre-reset fraction (`90 / 180 = 0.50`) must fit inside the closing window's usable quota.
    - The post-reset fraction (`0.50`) lands in the freshly replenished 100% window.
    - The candidate receives the pre-reset harvest bonus because it harvests expiring quota across the boundary.
    """
    learner = QuotaCostLearner(tmp_path / "boundary.sqlite")
    quotas = {
        "claude": {
            "status": "warning",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.10,
            "opus": 0.10,
            "haiku": 0.10,
            "windows": {
                # 5h window has 10% remaining (usable = 5% after 5% safety reserve), resetting in 90 seconds!
                # Full task upper 5h cost is ~6.3%, which would fail if not boundary-aware,
                # but pre-reset fraction (90s / 180s = 50% -> required_pre_reset_5h = 3.65%) fits inside 5% usable!
                "five_hour": {"used_percent": 90.0, "remaining_fraction": 0.10, "reset_in_sec": 90},
                "weekly": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 300000},
            },
            "reset_in_sec": 90,
        },
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": 0.70,
            "gemini_pro": 0.70,
            "claude_sonnet": 0.70,
            "claude_opus": 0.70,
            "windows": {
                "gemini_5h": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 14400},
                "gemini_weekly": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 300000},
                "claude_gpt_5h": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 14400},
                "claude_gpt_weekly": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 300000},
            },
            "reset_in_sec": 14400,
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": 0.70,
            "terra": 0.70,
            "sol": 0.70,
            "windows": {
                "five_hour": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 14400},
                "weekly": {"used_percent": 30.0, "remaining_fraction": 0.70, "reset_in_sec": 300000},
            },
            "reset_in_sec": 14400,
        },
    }
    pool_states = build_pool_states(quotas, live_activity=None, cost_learner=learner, operating_mode="endurance")
    claude_m = next(m for m in MODEL_CATALOG if m["model_id"] == "claude-sonnet-5-5")
    pred_cost = learner.predict_cost("claude_shared", "implement", "MID", "med")

    feasible, score, reasons, rej, math_bd = evaluate_candidate_schedule_score(
        model_entry=claude_m,
        pool_state=pool_states["claude_shared"],
        cost_pred=pred_cost,
        floor_tier="MID",
        floor_rank=2,
        target_effort="medium",
        required_tokens=32000,
        kind="implement",
        dependency_role="leaf",
        implementer_family="",
        preferred_provider="",
        is_explicit_match=False,
        operating_mode="endurance",
    )
    assert feasible is True, f"Expected feasible across reset boundary, got rejection: {rej}"
    assert math_bd["crosses_5h_reset"] is True
    assert math_bd["pre_reset_fraction_5h"] == 0.5
    assert math_bd["post_reset_fraction_5h"] == 0.5
    assert math_bd["required_pre_reset_5h"] < math_bd["predicted_5h_upper"]
    assert math_bd["required_pre_reset_5h"] <= math_bd["usable_5h"]
    assert math_bd["pre_reset_harvest_bonus"] > 0.15


def test_collector_malformed_stale_and_missing_http200_fixtures() -> None:
    """
    Verify that a successful HTTP 200 payload with malformed (>100%, <0%, NaN), missing (null),
    or expired reset timestamps NEVER produces a false `VERIFIED` healthy state.
    """
    from collectors.claude_collector import parse_claude_oauth_usage_payload
    from collectors.codex_collector import parse_codex_live_usage_payload

    now = datetime.datetime(2026, 10, 20, 20, 0, 0, tzinfo=datetime.timezone.utc)

    # 1. AGY HTTP 200 payload with out-of-range remainingFraction (1.45 and -0.20)
    agy_malformed = {
        "userStatus": {
            "cascadeModelConfigData": {
                "clientModelConfigs": [
                    {
                        "label": "Gemini 3.8 Flash",
                        "modelId": "gemini-3.8-flash-high",
                        "quotaInfo": {"remainingFraction": 1.45, "resetTime": "2026-10-20T22:00:00Z"},
                    },
                    {
                        "label": "Claude Sonnet 5.5",
                        "modelId": "claude-sonnet-5-5-high",
                        "quotaInfo": {"remainingFraction": -0.20, "resetTime": "2026-10-20T22:00:00Z"},
                    },
                ]
            }
        }
    }
    parsed_agy_mal = parse_agy_user_status(agy_malformed, now=now)
    assert parsed_agy_mal["confidence"] == "SUSPECT"
    assert parsed_agy_mal["status"] == "unknown"
    assert parsed_agy_mal["gemini_flash"] is None
    assert parsed_agy_mal["claude_sonnet"] is None

    # 2. Claude HTTP 200 payload with missing (`None`) utilization and malformed (`180.0`) utilization
    claude_missing = {
        "five_hour": {"utilization": None, "resets_at": "2026-10-20T22:00:00+00:00"},
        "seven_day": {"utilization": None, "resets_at": "2026-10-26T18:00:00+00:00"},
    }
    parsed_cl_missing = parse_claude_oauth_usage_payload(claude_missing, now=now)
    assert parsed_cl_missing["confidence"] == "UNKNOWN"
    assert parsed_cl_missing["status"] == "unknown"
    assert parsed_cl_missing["sonnet"] is None

    claude_malformed = {
        "five_hour": {"utilization": 180.0, "resets_at": "2026-10-20T22:00:00+00:00"},
        "seven_day": {"utilization": 40.0, "resets_at": "2026-10-26T18:00:00+00:00"},
    }
    parsed_cl_mal = parse_claude_oauth_usage_payload(claude_malformed, now=now)
    assert parsed_cl_mal["confidence"] == "SUSPECT"
    assert parsed_cl_mal["status"] == "unknown"
    assert parsed_cl_mal["sonnet"] is None

    # 3. Codex HTTP 200 payload with malformed negative used_percent (-15%)
    codex_malformed = {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {"used_percent": -15.0, "reset_after_seconds": 3600, "reset_at": int(now.timestamp()) + 3600},
            "secondary_window": {"used_percent": 40.0, "reset_after_seconds": 86400, "reset_at": int(now.timestamp()) + 86400},
        },
    }
    parsed_cx_mal = parse_codex_live_usage_payload(codex_malformed, now=now)
    assert parsed_cx_mal["confidence"] == "SUSPECT"
    assert parsed_cx_mal["status"] == "unknown"
    assert parsed_cx_mal["sol"] is None


def test_backlog_queue_persistence_crash_recovery_and_codex_degraded_continuity(tmp_path: Path) -> None:
    """
    End-to-end verification of `BacklogQueueManager` (Items 1 & 7):
    1. Persistent SQLite queue (`enqueue_tasks`).
    2. Crash recovery (`recover_orphaned_jobs`) when a worker PID dies mid-flight.
    3. Degraded-mode continuity: When Codex falls below its working reserve (`3%` remaining ->
       `degraded_continuity_mode = True`), already-approved leaf tasks continue executing on AGY and Claude
       with `codex_inference_used_for_dispatch = False`, while tasks requiring a fresh Codex decision wait.
    """
    db_path = tmp_path / "queue_test.sqlite"
    qm = BacklogQueueManager(db_path)
    learner = QuotaCostLearner(db_path)

    enq = qm.enqueue_tasks(
        [
            {
                "job_id": "JOB-CRASH-01",
                "task": "Implement telemetry parser edge cases",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "module",
                "dependency_role": "leaf",
            },
            {
                "job_id": "JOB-LEAF-02",
                "task": "Write unit tests for reset boundary",
                "kind": "test-author",
                "ambiguity": "low",
                "blast_radius": "local",
                "dependency_role": "leaf",
            },
            {
                "job_id": "JOB-CODEX-ARCH-03",
                "task": "Re-architect cross-module contract requiring Codex coordinator decision",
                "kind": "design",
                "ambiguity": "high",
                "blast_radius": "cross-module",
                "dependency_role": "upstream",
                "requires_codex_decision": True,
            },
        ]
    )
    assert len(enq) == 3

    # Simulate JOB-CRASH-01 having started on dead PID 999999 before a supervisor crash
    qm.mark_job_running("JOB-CRASH-01", provider="agy", pool_id="agy_gemini", model_id="gemini-3.8-flash-medium", worker_pid=999999)
    summary_mid = qm.get_queue_summary()
    assert summary_mid["counts"]["RUNNING"] == 1

    # Recover orphaned RUNNING jobs whose PID is no longer alive
    rec_res = qm.recover_orphaned_jobs(alive_pids=set())
    assert rec_res["recovered_count"] == 1
    assert "JOB-CRASH-01" in rec_res["requeued_jobs"]

    # Simulate Codex below its working reserve (3% 5h remaining -> degraded_continuity_mode = True)
    # while AGY and Claude have healthy capacity (80% remaining)
    quotas = {
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "gemini_flash": 0.80,
            "gemini_pro": 0.80,
            "claude_sonnet": 0.80,
            "claude_opus": 0.80,
            "windows": {
                "gemini_5h": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 10800},
                "gemini_weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 300000},
                "claude_gpt_5h": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 10800},
                "claude_gpt_weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 300000},
            },
            "reset_in_sec": 10800,
        },
        "claude": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.80,
            "opus": 0.80,
            "haiku": 0.80,
            "windows": {
                "five_hour": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 10800},
                "weekly": {"used_percent": 20.0, "remaining_fraction": 0.80, "reset_in_sec": 300000},
            },
            "reset_in_sec": 10800,
        },
        "codex": {
            "status": "exhausted",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": 0.03,
            "terra": 0.03,
            "sol": 0.03,
            "windows": {
                "five_hour": {"used_percent": 97.0, "remaining_fraction": 0.03, "reset_in_sec": 5400},
                "weekly": {"used_percent": 65.0, "remaining_fraction": 0.35, "reset_in_sec": 300000},
            },
            "reset_in_sec": 5400,
        },
    }
    orch = select_active_orchestrator("auto", quotas, operating_mode="endurance")

    def _exec_fn(ar: dict, dry_run: bool, timeout_sec: int = 60) -> dict:
        return {"executed": not dry_run, "mode": "simulated_worker", "exit_code": 0, "output": f"Completed {ar['task_id']} on {ar['provider']}"}

    # Step supervisor: executes JOB-CRASH-01 and JOB-LEAF-02 on AGY/Claude without Codex inference,
    # and defers JOB-CODEX-ARCH-03 until Codex reset!
    step_res = qm.step_supervisor(
        route_fn=route_work_package,
        execute_fn=_exec_fn,
        providers_quota=quotas,
        orchestrator_info=orch,
        cost_learner=learner,
        operating_mode="endurance",
        max_dispatch_per_tick=3,
        execute_live=True,
    )
    assert step_res["codex_degraded_continuity_active"] is True
    assert len(step_res["dispatched"]) == 2
    assert all(d["provider"] in {"agy", "claude"} for d in step_res["dispatched"])
    assert all(d["codex_inference_used_for_dispatch"] is False for d in step_res["dispatched"])
    assert len(step_res["deferred_for_codex"]) == 1
    assert step_res["deferred_for_codex"][0]["job_id"] == "JOB-CODEX-ARCH-03"
    assert step_res["queue_summary"]["counts"]["COMPLETED"] == 2
    assert step_res["queue_summary"]["counts"]["DEFERRED_WAIT_CODEX_RESET"] == 1
    assert step_res["queue_summary"]["total_crash_recovered"] == 1


def test_multi_day_100_task_mathematical_benchmark() -> None:
    """
    Verify the 100-task multi-day simulation benchmark (`run_multi_day_benchmark`) across both
    Endurance and Sprint modes, comparing `reset_aware_v2`, `highest_remaining`, and `round_robin`,
    plus the `harvest_bonus_weight` sensitivity sweep.
    """
    for mode in ("endurance", "sprint"):
        bench = run_multi_day_benchmark(num_tasks=100, operating_mode=mode)
        comp = bench["comparison"]
        ra = comp["reset_aware_v2"]
        hr = comp["highest_remaining"]
        rr = comp["round_robin"]

        assert ra["completed_tasks"] == 100
        # Reset-aware v2 must achieve zero Codex orchestrator downtime and lower stranded quota than baselines
        assert ra["codex_orchestrator_downtime_hours"] == 0.0
        assert ra["codex_orchestrator_downtime_hours"] <= hr["codex_orchestrator_downtime_hours"]
        assert ra["codex_orchestrator_downtime_hours"] <= rr["codex_orchestrator_downtime_hours"]
        assert ra["stranded_quota_wasted_pct_pts"] < hr["stranded_quota_wasted_pct_pts"]
        assert len(bench["weight_sensitivity_sweep"]) == 5


def test_in_flight_agent_scorecard_no_duplication_and_test_vs_acceptance_divergence(tmp_path: Path) -> None:
    """
    Refinement 1:
    - Do not duplicate real project tasks to benchmark agents.
    - Measure each agent using its own assigned tasks within similar task categories (`task_kind`)
      and complexity levels (`tier`, `complexity`).
    - Explicitly distinguish automated test success (`tests_passed`) from actual feature acceptance
      (`feature_accepted`), alongside build success, regressions, rework, duration, and quota cost.
    """
    learner = QuotaCostLearner(tmp_path / "scorecard.sqlite")

    # Agent 1 (claude-sonnet-5-5) runs its own assigned MID/med implement task (WP-AUTH-01):
    # Automated tests pass AND feature is accepted with 0 regressions / 0 rework.
    learner.record_observation(
        task_id="WP-AUTH-01",
        provider="claude",
        pool_id="claude_shared",
        model_id="claude-sonnet-5-5",
        task_kind="implement",
        tier="MID",
        complexity="med",
        duration_sec=110.0,
        success=True,
        tests_passed=True,
        acceptance_criteria_met=True,
        feature_accepted=True,
        build_succeeded=True,
        regression_count=0,
        rework_count=0,
        pre_5h_used_pct=20.0,
        post_5h_used_pct=24.0,
        pre_weekly_used_pct=15.0,
        post_weekly_used_pct=16.5,
        concurrent_pool_jobs=1,
    )

    # Agent 2 (gemini-3.8-flash-medium) runs its own distinct assigned MID/med implement task (WP-CACHE-02):
    # Initially automated unit tests pass (tests_passed=True), so unit runner reports green.
    obs2 = learner.record_observation(
        task_id="WP-CACHE-02",
        provider="agy",
        pool_id="agy_gemini",
        model_id="gemini-3.8-flash-medium",
        task_kind="implement",
        tier="MID",
        complexity="med",
        duration_sec=85.0,
        success=True,
        tests_passed=True,
        acceptance_criteria_met=True,
        feature_accepted=True,
        build_succeeded=True,
        regression_count=0,
        rework_count=0,
        pre_5h_used_pct=10.0,
        post_5h_used_pct=12.5,
        pre_weekly_used_pct=10.0,
        post_weekly_used_pct=10.8,
        concurrent_pool_jobs=1,
    )
    assert obs2["attribution_status"] == "MEASURED_SINGLE_JOB"

    # During independent acceptance review of WP-CACHE-02, the reviewer finds an edge-case regression:
    # Automated tests passed, but actual feature acceptance failed (`feature_accepted=False`), requiring 1 rework cycle!
    upd = learner.update_task_acceptance(
        task_id="WP-CACHE-02",
        acceptance_criteria_met=False,
        tests_passed=True,
        feature_accepted=False,
        build_succeeded=True,
        regression_count=1,
        rework_count=1,
    )
    assert upd["updated"] is True

    scorecard = learner.get_agent_scorecard()
    assert scorecard["evaluation_policy"] == "IN_FLIGHT_ASSIGNED_TASKS_ONLY"
    assert scorecard["duplicate_real_task_benchmarking"] is False
    assert scorecard["routing_influence"] == "INFORMATIONAL_ONLY"
    assert scorecard["auto_routing_by_quality_score_enabled"] is False

    stratum = next(
        s for s in scorecard["by_category_and_complexity"]
        if s["task_kind"] == "implement" and s["tier"] == "MID" and s["complexity"] == "med"
    )
    agents_by_model = {a["model_id"]: a for a in stratum["agents"]}
    assert "claude-sonnet-5-5" in agents_by_model
    assert "gemini-3.8-flash-medium" in agents_by_model

    sonnet_sc = agents_by_model["claude-sonnet-5-5"]
    flash_sc = agents_by_model["gemini-3.8-flash-medium"]

    # Both had 100% automated test pass rate, but Flash had 0% actual feature acceptance -> +1.0 divergence!
    assert sonnet_sc["automated_test_pass_rate"] == 1.0
    assert sonnet_sc["actual_feature_acceptance_rate"] == 1.0
    assert sonnet_sc["test_vs_acceptance_divergence"] == 0.0

    assert flash_sc["automated_test_pass_rate"] == 1.0
    assert flash_sc["actual_feature_acceptance_rate"] == 0.0
    assert flash_sc["test_vs_acceptance_divergence"] == 1.0
    assert flash_sc["regressions_per_task"] == 1.0
    assert flash_sc["rework_per_task"] == 1.0
    assert sonnet_sc["empirical_quality_score"] > flash_sc["empirical_quality_score"]

    # Also verify FastAPI /api/scorecard endpoint
    client = TestClient(app)
    r_sc = client.get("/api/scorecard")
    assert r_sc.status_code == 200
    assert r_sc.json()["duplicate_real_task_benchmarking"] is False


def test_expiry_aware_codex_banked_reset_harvesting_overlay(tmp_path: Path) -> None:
    """
    Refinement 2:
    - Dynamically increase Codex's task allocation when a banked reset approaches expiry (<=72h)
      to productively utilize the existing weekly allowance before it is replaced.
    - Do not require quota exhaustion (0%) before redemption (partial window refresh supported).
    - Optimize timing across all 7 factors (5h/weekly remaining, normal reset time, banked reset expiry,
      actual suitable queued work, expected consumption before expiry, schedule-shift opportunity cost,
      and coordination reserve/fallback capacity).
    - Temporarily relax routine Codex worker reserves when mathematically justified, retaining minimum
      coordination capacity.
    - Never generate filler work when suitable_queued_tasks_count == 0.
    - Never redeem automatically.
    """
    learner = QuotaCostLearner(tmp_path / "harvest.sqlite")

    def _make_quotas(codex_5h_rem: float, codex_wk_rem: float, earliest_exp_sec: int) -> dict:
        return {
            "agy": {
                "status": "healthy",
                "confidence": "VERIFIED",
                "freshness_age_sec": 2,
                "gemini_flash": 0.72,
                "gemini_pro": 0.72,
                "claude_sonnet": 0.72,
                "claude_opus": 0.72,
                "windows": {
                    "gemini_5h": {"used_percent": 28.0, "remaining_fraction": 0.72, "reset_in_sec": 14400},
                    "gemini_weekly": {"used_percent": 28.0, "remaining_fraction": 0.72, "reset_in_sec": 500000},
                    "claude_gpt_5h": {"used_percent": 28.0, "remaining_fraction": 0.72, "reset_in_sec": 14400},
                    "claude_gpt_weekly": {"used_percent": 28.0, "remaining_fraction": 0.72, "reset_in_sec": 500000},
                },
                "reset_in_sec": 14400,
            },
            "claude": {
                "status": "healthy",
                "confidence": "VERIFIED",
                "freshness_age_sec": 2,
                "sonnet": 0.72,
                "opus": 0.72,
                "haiku": 0.72,
                "windows": {
                    "five_hour": {"used_percent": 28.0, "remaining_fraction": 0.72, "reset_in_sec": 14400},
                    "weekly": {"used_percent": 28.0, "remaining_fraction": 0.72, "reset_in_sec": 500000},
                },
                "reset_in_sec": 14400,
            },
            "codex": {
                "status": "healthy",
                "confidence": "VERIFIED",
                "freshness_age_sec": 2,
                "luna": codex_wk_rem,
                "terra": codex_wk_rem,
                "sol": codex_wk_rem,
                "windows": {
                    "five_hour": {
                        "used_percent": round((1.0 - codex_5h_rem) * 100.0, 1),
                        "remaining_fraction": codex_5h_rem,
                        "reset_in_sec": 14400,
                    },
                    "weekly": {
                        "used_percent": round((1.0 - codex_wk_rem) * 100.0, 1),
                        "remaining_fraction": codex_wk_rem,
                        "reset_in_sec": 580000,  # ~6.71 days until Oct 27 natural reset
                    },
                },
                "banked_resets": {
                    "verification_status": "VERIFIED_API",
                    "available_count": 3,
                    "applicable_available_count": 3,
                    "credit_type": "Full reset (Weekly + 5 hr)",
                    "earliest_expires_at": "2026-10-22T18:16:02.128577Z",
                    "earliest_expires_in_sec": earliest_exp_sec,
                    "items": [
                        {
                            "id": "RateLimitResetCredit_7d9d80497f888191ae8e58efd4e08c28",
                            "credit_type": "Full reset (Weekly + 5 hr)",
                            "expires_at": "2026-10-22T18:16:02.128577Z",
                            "expires_in_sec": earliest_exp_sec,
                        }
                    ],
                },
                "reset_in_sec": 14400,
            },
        }

    # CASE A: 48h until Oct 22 banked reset expiry, 69% weekly remaining, 75% 5h remaining, 4 queued tasks
    q_harvest = _make_quotas(codex_5h_rem=0.75, codex_wk_rem=0.69, earliest_exp_sec=48 * 3600)
    for mode in ("endurance", "sprint"):
        br_strat = evaluate_banked_reset_strategy(
            q_harvest["codex"],
            operating_mode=mode,
            suitable_queued_tasks_count=4,
            independent_fallback_healthy=True,
        )
        assert br_strat["auto_trigger_permitted"] is False
        assert br_strat["partial_window_redemption_supported"] is True
        assert br_strat["exhaustion_required_before_redemption"] is False
        assert br_strat["recommendation_code"] == "HARVEST_EXISTING_QUOTA_BEFORE_EXPIRY"
        assert br_strat["harvest_overlay"]["active"] is True
        assert br_strat["harvest_overlay"]["allocation_boost_score"] > 0.25
        assert "Banked reset expires in 48.0h" in br_strat["comparative_timing_banner"]
        assert "Estimated benefit of redeeming now" in br_strat["comparative_timing_banner"]

        # Verify dynamic reserve relaxation and task routing in both Endurance and Sprint modes
        orch = select_active_orchestrator("auto", q_harvest, operating_mode=mode)
        route_res = route_work_package(
            {
                "task": f"WP-HARVEST-{mode.upper()}: Implement telemetry parser module",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "module",
                "dependency_role": "leaf",
                "context_need": "pointer",
                "unfinished_tasks_count": 4,
            },
            providers_quota=q_harvest,
            orchestrator_info=orch,
            cost_learner=learner,
            operating_mode=mode,
        )
        assert route_res["codex_reserve"]["reserve_relaxed_for_banked_harvest"] is True
        assert route_res["codex_reserve"]["reserve_5h_fraction"] <= 0.05
        assert route_res["codex_reserve"]["reserve_weekly_fraction"] <= 0.04
        assert route_res["assignment_record"]["math_breakdown"]["quality_score_routing_influence"] == 0.0
        assert route_res["assignment_record"]["math_breakdown"]["hard_quota_feasibility_passed"] is True
        assert route_res["assignment_record"]["math_breakdown"]["hard_concurrency_gate_passed"] is True
        # Because harvest_overlay is active, Codex wins the leaf task allocation to burn existing weekly quota before expiry!
        assert route_res["assignment_record"]["provider"] == "codex"

    # CASE B: No-Filler Work Guard (0 suitable queued tasks -> overlay MUST NOT activate)
    br_no_work = evaluate_banked_reset_strategy(
        q_harvest["codex"],
        operating_mode="endurance",
        suitable_queued_tasks_count=0,
        independent_fallback_healthy=True,
    )
    assert br_no_work["harvest_overlay"]["active"] is False
    assert br_no_work["harvest_overlay"]["no_filler_work_guard"] is True
    assert br_no_work["timing_comparison"]["consumption_forecast_basis"] == "FORECASTED_FROM_REAL_QUEUE_PRIORS"
    assert br_no_work["timing_comparison"]["measured_window_usage_basis"] == "MEASURED_LIVE_WINDOW_TELEMETRY"

    # CASE C: 5 hours before expiry with partially consumed weekly window (35% remaining / 65% used)
    q_urgent = _make_quotas(codex_5h_rem=0.40, codex_wk_rem=0.35, earliest_exp_sec=5 * 3600)
    br_urgent = evaluate_banked_reset_strategy(
        q_urgent["codex"],
        operating_mode="endurance",
        suitable_queued_tasks_count=1,
        independent_fallback_healthy=True,
    )
    assert br_urgent["exhaustion_required_before_redemption"] is False
    assert br_urgent["recommendation_code"] == "CONSIDER_MANUAL_REDEEM_EXPIRING_CREDIT"
    assert br_urgent["auto_trigger_permitted"] is False
    assert br_urgent["comparative_timing_banner"].startswith("Banked reset expires in 5.0h")

    # CASE D: Expiry harvesting boost NEVER overrides hard 5h quota feasibility or concurrency ceiling!
    # D1: Codex has 69% weekly remaining and expiring banked reset, but only 2% 5h remaining -> MUST NOT route to Codex
    q_5h_blocked = _make_quotas(codex_5h_rem=0.02, codex_wk_rem=0.69, earliest_exp_sec=48 * 3600)
    orch_d1 = select_active_orchestrator("auto", q_5h_blocked, operating_mode="endurance")
    route_d1 = route_work_package(
        {
            "task": "WP-HARVEST-FEASIBILITY-GATE: Implement parser module",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "module",
            "dependency_role": "leaf",
            "context_need": "pointer",
            "unfinished_tasks_count": 4,
            "suitable_queued_tasks_count": 4,
        },
        providers_quota=q_5h_blocked,
        orchestrator_info=orch_d1,
        cost_learner=learner,
        operating_mode="endurance",
    )
    assert route_d1["assignment_record"]["provider"] != "codex"
    assert any(r["pool_id"] == "codex_shared" and "5h window insufficient" in r["reason"] for r in route_d1["rejected_models"])

    # D2: Codex has 75% 5h and 69% weekly remaining + active harvest boost, but 6/6 concurrent jobs running on codex_shared
    live_act_full_codex = {
        "workers": [{"pid": 9001 + i, "provider": "codex", "model": "gpt-6.1-sol"} for i in range(6)]
    }
    route_d2 = route_work_package(
        {
            "task": "WP-HARVEST-CONCURRENCY-GATE: Implement parser module",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "module",
            "dependency_role": "leaf",
            "context_need": "pointer",
            "unfinished_tasks_count": 4,
            "suitable_queued_tasks_count": 4,
        },
        providers_quota=q_harvest,
        orchestrator_info=orch,
        live_activity=live_act_full_codex,
        cost_learner=learner,
        operating_mode="endurance",
    )
    assert route_d2["assignment_record"]["provider"] != "codex"
    assert any(r["pool_id"] == "codex_shared" and "Concurrency ceiling reached" in r["reason"] for r in route_d2["rejected_models"])


def test_autonomous_supervisor_dependencies_authoritative_refresh_and_restart_safety(tmp_path: Path) -> None:
    """
    Verify all 5 long-running operational requirements:
    1. Queued tasks dispatch automatically according to quota, dependency (`depends_on`), and concurrency (`max_concurrent`) rules.
    2. Provider quota exhaustion (`codex` at 0%, `claude` at 0%) does not stop unrelated eligible providers (`agy`) from working.
    3. Restarting Quota Hub preserves the SQLite queue (`preserve_existing=True`), never re-executes `COMPLETED` jobs,
       and recovers crashed `RUNNING` worker PIDs without duplicate execution.
    4. `DEFERRED_WAIT_CODEX_RESET` tasks automatically resume immediately after an authoritative quota refresh
       (`codex_authoritatively_refreshed == True`) even before the fallback timer expires.
    """
    import copy
    import sqlite3

    db_path = tmp_path / "autonomous_queue_test.sqlite3"
    q_mgr = BacklogQueueManager(db_path=db_path)
    learner = QuotaCostLearner(db_path=db_path)

    executed_calls: list[dict] = []

    def _mock_execute_fn(assignment_record: dict, dry_run: bool = False, timeout_sec: int = 240) -> dict:
        executed_calls.append(
            {
                "task": assignment_record.get("task"),
                "provider": assignment_record.get("provider"),
                "model": assignment_record.get("model"),
                "dry_run": dry_run,
            }
        )
        return {
            "executed": True,
            "dry_run": dry_run,
            "exit_code": 0,
            "pid": 4242,
            "duration_ms": 15,
            "stdout_tail": f"OK:{assignment_record.get('task')}",
            "stderr_tail": "",
        }

    # Initial enqueue of a 4-task DAG:
    # - JOB-DONE: will complete first, then we simulate a server restart + re-enqueue to prove 0 duplicate execution
    # - JOB-A: leaf task (no dependencies)
    # - JOB-B: leaf task depending on JOB-A
    # - JOB-C: Codex-specific design task (`requires_codex_decision=True`) depending on JOB-B
    # - JOB-CRASH: simulated mid-flight crash before server restart
    tasks = [
        {
            "job_id": "JOB-DONE",
            "task": "Already completed bootstrap task",
            "kind": "scan",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "priority": 100,
        },
        {
            "job_id": "JOB-CRASH",
            "task": "Orphaned worker task from prior server process",
            "kind": "scan",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "priority": 90,
        },
        {
            "job_id": "JOB-A",
            "task": "Step 1 prerequisite module scan",
            "kind": "scan",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "priority": 80,
        },
        {
            "job_id": "JOB-B",
            "task": "Step 2 implementation depending on JOB-A",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "depends_on": ["JOB-A"],
            "priority": 70,
        },
        {
            "job_id": "JOB-C",
            "task": "Step 3 Codex architecture decision depending on JOB-B",
            "kind": "design",
            "ambiguity": "medium",
            "blast_radius": "module",
            "dependency_role": "upstream",
            "requires_codex_decision": True,
            "depends_on": ["JOB-B"],
            "priority": 60,
        },
    ]
    now_t0 = 1792500000.0
    q_mgr.enqueue_tasks(tasks, now_ts=now_t0)

    # Simulate pre-restart state: JOB-DONE is already COMPLETED; JOB-CRASH was RUNNING on dead PID 999991
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE backlog_jobs SET status = 'COMPLETED', attempt_count = 1 WHERE job_id = 'JOB-DONE'")
        conn.execute("UPDATE backlog_jobs SET status = 'RUNNING', worker_pid = 999991, attempt_count = 1 WHERE job_id = 'JOB-CRASH'")
        conn.commit()

    # Simulate server restart: new BacklogQueueManager instance attaches to same SQLite DB & re-enqueues backlog with preserve_existing=True
    q_mgr_restarted = BacklogQueueManager(db_path=db_path)
    q_mgr_restarted.enqueue_tasks(tasks, now_ts=now_t0, preserve_existing=True)

    # Crash recovery on restart: dead PID 999991 is recovered to QUEUED; JOB-DONE remains COMPLETED
    rec = q_mgr_restarted.recover_orphaned_jobs(alive_pids={1001, 1002}, now_ts=now_t0)
    assert rec["recovered_count"] == 1
    assert rec["requeued_jobs"] == ["JOB-CRASH"]

    # Quota state 1: Both Codex (0% 5h) AND Claude (0% 5h) are completely exhausted! Only AGY is healthy (85%).
    quotas_codex_claude_exhausted = {
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 3,
            "gemini_flash": 0.85,
            "gemini_pro": 0.85,
            "claude_sonnet": 0.85,
            "claude_opus": 0.85,
            "reset_in_sec": 7200,
            "windows": {
                "gemini_5h": {"remaining_fraction": 0.85, "used_percent": 15.0, "reset_in_sec": 7200, "resets_at": "2026-10-20T18:00:00Z"},
                "gemini_weekly": {"remaining_fraction": 0.85, "used_percent": 15.0, "reset_in_sec": 86400, "resets_at": "2026-10-25T18:00:00Z"},
                "claude_gpt_5h": {"remaining_fraction": 0.85, "used_percent": 15.0, "reset_in_sec": 7200, "resets_at": "2026-10-20T18:00:00Z"},
                "claude_gpt_weekly": {"remaining_fraction": 0.85, "used_percent": 15.0, "reset_in_sec": 86400, "resets_at": "2026-10-25T18:00:00Z"},
            },
        },
        "claude": {
            "status": "exhausted",
            "confidence": "VERIFIED",
            "freshness_age_sec": 5,
            "sonnet": 0.0,
            "opus": 0.0,
            "haiku": 0.0,
            "reset_in_sec": 5400,
            "windows": {
                "five_hour": {"remaining_fraction": 0.0, "used_percent": 100.0, "reset_in_sec": 5400, "resets_at": "2026-10-20T17:30:00Z"},
                "weekly": {"remaining_fraction": 0.50, "used_percent": 50.0, "reset_in_sec": 172800, "resets_at": "2026-10-26T18:00:00Z"},
            },
        },
        "codex": {
            "status": "exhausted",
            "confidence": "VERIFIED",
            "freshness_age_sec": 5,
            "gpt_5": 0.0,
            "o3": 0.0,
            "reset_in_sec": 7200,
            "windows": {
                "five_hour": {"remaining_fraction": 0.0, "used_percent": 100.0, "reset_in_sec": 7200, "resets_at": "2026-10-20T18:00:00Z"},
                "weekly": {"remaining_fraction": 0.65, "used_percent": 35.0, "reset_in_sec": 259200, "resets_at": "2026-10-27T18:00:00Z"},
            },
        },
    }
    orch_exhausted = select_active_orchestrator("codex", quotas_codex_claude_exhausted, operating_mode="endurance")
    assert orch_exhausted["active_provider"] == "agy"

    # TICK 1 (max_dispatch_per_tick=1):
    # - JOB-DONE is already COMPLETED (must NOT execute again)
    # - JOB-CRASH (priority 90) executes on AGY
    # - JOB-B and JOB-C remain blocked by dependencies
    tick1 = q_mgr_restarted.step_supervisor(
        route_fn=route_work_package,
        execute_fn=_mock_execute_fn,
        providers_quota=quotas_codex_claude_exhausted,
        orchestrator_info=orch_exhausted,
        cost_learner=learner,
        operating_mode="endurance",
        max_dispatch_per_tick=1,
        execute_live=True,
        now_ts=now_t0,
    )
    assert [d["job_id"] for d in tick1["dispatched"]] == ["JOB-CRASH"]
    assert tick1["dispatched"][0]["provider"] == "agy"
    assert all(c["task"] != "Already completed bootstrap task" for c in executed_calls)

    # TICK 2 (max_dispatch_per_tick=2):
    # - JOB-A (priority 80) executes on AGY and completes within Tick 2
    # - Because JOB-A completes, JOB-B (depends_on=["JOB-A"]) unlocks and also executes on AGY in Tick 2!
    # - JOB-C (depends_on=["JOB-B"]) unlocks after JOB-B, sees `requires_codex_decision=True` while Codex is at 0%,
    #   and transitions to `DEFERRED_WAIT_CODEX_RESET` (resume_in_sec=7200)!
    tick2 = q_mgr_restarted.step_supervisor(
        route_fn=route_work_package,
        execute_fn=_mock_execute_fn,
        providers_quota=quotas_codex_claude_exhausted,
        orchestrator_info=orch_exhausted,
        cost_learner=learner,
        operating_mode="endurance",
        max_dispatch_per_tick=2,
        execute_live=True,
        now_ts=now_t0 + 2.0,
    )
    assert [d["job_id"] for d in tick2["dispatched"]] == ["JOB-A", "JOB-B"]
    assert all(d["provider"] == "agy" for d in tick2["dispatched"])

    # Next tick (5s later, Codex still exhausted): JOB-C is now evaluated and deferred to DEFERRED_WAIT_CODEX_RESET
    tick3 = q_mgr_restarted.step_supervisor(
        route_fn=route_work_package,
        execute_fn=_mock_execute_fn,
        providers_quota=quotas_codex_claude_exhausted,
        orchestrator_info=orch_exhausted,
        cost_learner=learner,
        operating_mode="endurance",
        max_dispatch_per_tick=2,
        execute_live=True,
        now_ts=now_t0 + 5.0,
    )
    assert [d["job_id"] for d in tick3["deferred_for_codex"]] == ["JOB-C"]
    assert tick3["queue_summary"]["counts"].get("DEFERRED_WAIT_CODEX_RESET") == 1

    # TICK 4 (Only 10 seconds later — 7190s BEFORE the 7200s timer would expire!):
    # An authoritative Codex quota refresh arrives (`confidence="VERIFIED"`, 5h=0.80, weekly=0.65).
    # `step_supervisor` must detect `codex_authoritatively_refreshed == True`, immediately promote
    # `JOB-C` from `DEFERRED_WAIT_CODEX_RESET -> QUEUED`, and execute it to `COMPLETED`!
    quotas_codex_refreshed = copy.deepcopy(quotas_codex_claude_exhausted)
    quotas_codex_refreshed["codex"] = {
        "status": "healthy",
        "confidence": "VERIFIED",
        "post_reset_unverified": False,
        "freshness_age_sec": 1,
        "sol": 0.90,
        "gpt_5": 0.90,
        "o3": 0.90,
        "reset_in_sec": 18000,
        "windows": {
            "five_hour": {"remaining_fraction": 0.90, "used_percent": 10.0, "reset_in_sec": 18000, "resets_at": "2026-10-20T23:00:00Z"},
            "weekly": {"remaining_fraction": 0.90, "used_percent": 10.0, "reset_in_sec": 259200, "resets_at": "2026-10-27T18:00:00Z"},
        },
    }
    orch_refreshed = select_active_orchestrator("codex", quotas_codex_refreshed, operating_mode="endurance")
    assert orch_refreshed["active_provider"] == "codex"

    tick4 = q_mgr_restarted.step_supervisor(
        route_fn=route_work_package,
        execute_fn=_mock_execute_fn,
        providers_quota=quotas_codex_refreshed,
        orchestrator_info=orch_refreshed,
        cost_learner=learner,
        operating_mode="endurance",
        max_dispatch_per_tick=2,
        execute_live=True,
        now_ts=now_t0 + 10.0,  # << now_t0 + 7205.0!
    )
    assert tick4["codex_authoritatively_refreshed"] is True
    assert "JOB-C" in tick4["resumed_from_codex_reset"]
    assert [d["job_id"] for d in tick4["dispatched"]] == ["JOB-C"]
    assert tick4["dispatched"][0]["status"] == "COMPLETED"
    assert tick4["queue_summary"]["counts"].get("COMPLETED") == 5
    assert tick4["queue_summary"]["counts"].get("DEFERRED_WAIT_CODEX_RESET", 0) == 0
    # Verify exact execution count: 4 jobs executed (JOB-CRASH, JOB-A, JOB-B, JOB-C), 0 duplicates of JOB-DONE!
    assert len(executed_calls) == 4


