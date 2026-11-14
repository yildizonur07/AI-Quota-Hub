"""
Tests for WP_HUB_CORE: policy loading, 3-level concurrency caps, cross-family review gate,
quota honesty (unknown/falsy-zero/zero-delta/Codex reserve), headless CLI safety, the
Beads-style DAG queue (validation, starvation-free selection, upstream failure, scope leases,
rate-limit attempts, preview purity), the Refinery gate, the Witness and hub security/state.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

import main
import orchestrator
from collectors import live_activity as live_mod
from orchestrator import (
    MODEL_CATALOG,
    build_headless_cli_command,
    detect_rate_limit,
    execute_headless_dispatch,
    route_work_package,
    select_active_orchestrator,
)
from scheduler import (
    BacklogQueueManager,
    QuotaCostLearner,
    build_pool_states,
    classify_worker_health,
    compute_dynamic_codex_reserve,
    evaluate_candidate_schedule_score,
    load_orchestrator_policy,
    path_in_scope,
    resolve_repo_key,
    scopes_overlap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _window(rem: float, reset: int) -> Dict[str, Any]:
    return {"remaining_fraction": rem, "used_percent": round((1.0 - rem) * 100.0, 1), "reset_in_sec": reset}


def _quotas(agy: float = 0.8, agy_cg: Optional[float] = None, claude: float = 0.8, codex: float = 0.8) -> Dict[str, Any]:
    agy_cg = agy if agy_cg is None else agy_cg
    return {
        "agy": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "gemini_flash": agy,
            "gemini_pro": agy,
            "claude_sonnet": agy_cg,
            "claude_opus": agy_cg,
            "reset_in_sec": 14400,
            "windows": {
                "gemini_5h": _window(agy, 14400),
                "gemini_weekly": _window(agy, 432000),
                "claude_gpt_5h": _window(agy_cg, 14400),
                "claude_gpt_weekly": _window(agy_cg, 432000),
            },
        },
        "claude": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "sonnet": claude,
            "opus": claude,
            "haiku": claude,
            "reset_in_sec": 14400,
            "windows": {"five_hour": _window(claude, 14400), "weekly": _window(claude, 432000)},
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "luna": codex,
            "terra": codex,
            "sol": codex,
            "reset_in_sec": 14400,
            "windows": {"five_hour": _window(codex, 14400), "weekly": _window(codex, 432000)},
        },
    }


def _write_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Dict[str, Any]) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("ORCHESTRATOR_POLICY_PATH", str(path))
    return path


def _workers(*specs: tuple) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    pid = 7000
    for provider, model, count in specs:
        for _ in range(count):
            pid += 1
            out.append({"pid": pid, "provider": provider, "model": model})
    return {"workers": out, "coordinators": []}


def _route(task: Dict[str, Any], quotas: Dict[str, Any], learner: QuotaCostLearner, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    orch = select_active_orchestrator("auto", quotas)
    return route_work_package(task, quotas, orch, live_activity=live, cost_learner=learner)


SCAN_TASK = {"task": "Scan inventory", "kind": "scan", "ambiguity": "low", "blast_radius": "local", "dependency_role": "leaf"}


# ---------------------------------------------------------------------------
# 1. Policy loading & 3-level caps
# ---------------------------------------------------------------------------


def test_policy_defaults_and_owner_file_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pol = load_orchestrator_policy()
    assert pol["_source"] == "impl-default"
    assert pol["max_leaves"] == 10
    assert pol["max_per_provider"] == {"agy": 6, "claude": 6, "codex": 6}
    assert pol["max_per_pool"]["agy_claude_gpt"] == 6
    assert pol["max_attempts"] == 2
    assert pol["review"]["cross_family"] == "optional"
    assert pol["review"]["cross_family_bonus"] == 0.12
    assert pol["quota"] == {"exclude_at_used": 0.95, "prefer_other_above": 0.80, "stale_after_sec": 120}
    assert pol["witness"]["stall_threshold_sec"] == 1800

    path = _write_policy(
        tmp_path,
        monkeypatch,
        {"max_leaves": 4, "max_per_provider": {"agy": 3, "claude": "bad"}, "max_per_pool": {}, "witness": {"quiet_threshold_sec": 300}},
    )
    custom = load_orchestrator_policy()
    assert custom["_source"] == str(path)
    assert custom["max_leaves"] == 4
    assert custom["max_per_provider"] == {"agy": 3, "claude": 6, "codex": 6}  # wrong type falls back to default
    assert custom["max_per_pool"]["codex_shared"] == 6  # empty owner map keeps defaults
    assert custom["witness"] == {"starting_grace_sec": 120, "quiet_threshold_sec": 300, "stall_threshold_sec": 1800}

    path.write_text("{not json", encoding="utf-8")
    assert load_orchestrator_policy()["_source"].startswith("impl-default (unreadable")


def test_agy_provider_cap_counts_both_pools_together(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    live = _workers(("agy", "gemini-3.8-flash-high", 3), ("agy", "claude-sonnet-5-5-medium", 3))
    pools = build_pool_states(_quotas(), live, learner)
    assert pools["agy_gemini"]["running_jobs"] == 3
    assert pools["agy_claude_gpt"]["running_jobs"] == 3
    assert pools["agy_gemini"]["provider_running_jobs"] == 6

    res = _route(SCAN_TASK, _quotas(), learner, live)
    assert res["status"] == "routed"
    assert res["assignment_record"]["provider"] != "agy"
    assert not any(c["provider"] == "agy" for c in res["ranked_candidates"])
    cap_rej = {r["pool_id"] for r in res["rejected_models"] if "Concurrency ceiling reached on provider agy (6/6)" in r["reason"]}
    assert cap_rej == {"agy_gemini", "agy_claude_gpt"}


def test_seventh_claim_on_one_provider_refused(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    live = _workers(("claude", "claude-sonnet-5-5", 6))
    orchestrator.set_preference_weights({"agy": 1, "codex": 1, "claude": 10}, reset_swrr=True)
    try:
        res = _route(SCAN_TASK, _quotas(agy=0.3, claude=0.95, codex=0.3), learner, live)
        assert res["assignment_record"]["provider"] != "claude"
        cl_rej = [r for r in res["rejected_models"] if r["pool_id"] == "claude_shared"]
        assert cl_rej and "Concurrency ceiling reached on provider claude (6/6)" in cl_rej[0]["reason"]

        five = _route(SCAN_TASK, _quotas(agy=0.3, claude=0.95, codex=0.3), learner, _workers(("claude", "claude-haiku-5-5", 5)))
        assert five["assignment_record"]["provider"] == "claude"
    finally:
        orchestrator.set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)


def test_eleventh_global_leaf_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_policy(tmp_path, monkeypatch, {"max_leaves": 10, "max_per_provider": {"agy": 8, "claude": 8, "codex": 8}, "max_per_pool": {"agy_gemini": 8, "claude_shared": 8}})
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    live = _workers(("agy", "gemini-3.8-flash-high", 5), ("claude", "claude-sonnet-5-5", 5))
    res = _route(SCAN_TASK, _quotas(), learner, live)
    assert res["status"] == "paused_quota_exhausted"
    codex_rej = [r for r in res["rejected_models"] if r["pool_id"] == "codex_shared"]
    assert codex_rej and "Concurrency ceiling reached: global leaves (10/10)" in codex_rej[0]["reason"]

    nine = _route(SCAN_TASK, _quotas(), learner, _workers(("agy", "gemini-3.8-flash-high", 5), ("claude", "claude-sonnet-5-5", 4)))
    assert nine["status"] == "routed"


def test_per_pool_cap_from_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_policy(tmp_path, monkeypatch, {"max_per_pool": {"agy_gemini": 2}})
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    live = _workers(("agy", "gemini-3.8-flash-high", 2))
    res = _route(dict(SCAN_TASK, kind="implement", blast_radius="module"), _quotas(), learner, live)
    assert any(r["pool_id"] == "agy_gemini" and "Concurrency ceiling reached on agy_gemini (2/2)" in r["reason"] for r in res["rejected_models"])
    assert not any(c["pool_id"] == "agy_gemini" for c in res["ranked_candidates"])
    assert not any(r["pool_id"] == "agy_claude_gpt" and "Concurrency" in r["reason"] for r in res["rejected_models"])


def test_coordinator_background_tasks_are_not_leaves(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    live = {"workers": [{"pid": 1, "provider": "agy-task", "model": "BG Task"} for _ in range(12)]}
    pools = build_pool_states(_quotas(), live, learner)
    assert pools["agy_gemini"]["total_running_leaves"] == 0


# ---------------------------------------------------------------------------
# 2. Cross-family review gate
# ---------------------------------------------------------------------------


def test_same_family_review_routable_unless_cross_family_required(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    quotas = _quotas(agy=0.85, agy_cg=0.0)
    quotas["claude"] = {"status": "unknown", "confidence": "UNKNOWN"}
    quotas["codex"] = {"status": "unknown", "confidence": "UNKNOWN"}
    review = {
        "task": "Review Gemini module",
        "kind": "review",
        "ambiguity": "med",
        "blast_radius": "module",
        "dependency_role": "upstream",
        "implementer_family": "gemini",
    }
    res = _route(review, quotas, learner)
    assert res["status"] == "routed"
    assert res["assignment_record"]["family"] == "gemini"
    assert not any("Cross-family" in r["reason"] for r in res["rejected_models"])

    strict = _route(dict(review, require_cross_family=True), quotas, learner)
    assert strict["status"] == "paused_quota_exhausted"
    assert any("Cross-family review required" in r["reason"] for r in strict["rejected_models"])


def test_cross_family_bonus_comes_from_policy(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    pools = build_pool_states(_quotas(), None, learner)
    opus = next(m for m in MODEL_CATALOG if m["model_id"] == "claude-opus-5-5")
    pred = learner.predict_cost("claude_shared", "review", "HIGH", "med")
    common = dict(
        pool_state=pools["claude_shared"], cost_pred=pred, floor_tier="HIGH", floor_rank=3, target_effort="medium",
        required_tokens=32000, kind="review", dependency_role="upstream", preferred_provider="", is_explicit_match=False,
    )
    _, s_cross, reasons, _, _ = evaluate_candidate_schedule_score(model_entry=opus, implementer_family="gemini", **common)
    _, s_same, _, _, _ = evaluate_candidate_schedule_score(model_entry=opus, implementer_family="claude", **common)
    assert round(s_cross - s_same, 4) == 0.12
    assert any("Cross-family review (+0.12" in r for r in reasons)


# ---------------------------------------------------------------------------
# 3. Quota honesty
# ---------------------------------------------------------------------------


def test_unknown_telemetry_is_never_capacity(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    quotas = _quotas()
    quotas["claude"] = {"status": "unknown", "confidence": "UNKNOWN", "sonnet": 0.9, "opus": 0.9, "haiku": 0.9}
    quotas["codex"] = {}
    pools = build_pool_states(quotas, None, learner)
    assert pools["claude_shared"]["five_hour"]["remaining_fraction"] is None  # scalar not fabricated into windows
    assert pools["codex_shared"]["confidence"] == "UNKNOWN"

    haiku = next(m for m in MODEL_CATALOG if m["model_id"] == "claude-haiku-5-5")
    feasible, _, _, reason, _ = evaluate_candidate_schedule_score(
        model_entry=haiku, pool_state=pools["claude_shared"], cost_pred=learner.predict_cost("claude_shared", "scan", "LOW"),
        floor_tier="LOW", floor_rank=1, target_effort="low", required_tokens=32000, kind="scan", dependency_role="leaf",
        implementer_family="", preferred_provider="", is_explicit_match=False,
    )
    assert feasible is False
    assert "Quota telemetry UNKNOWN/UNAVAILABLE" in reason

    res = _route(SCAN_TASK, quotas, learner)
    assert res["assignment_record"]["provider"] == "agy"
    assert all("UNKNOWN" in r["reason"] for r in res["rejected_models"] if r["pool_id"] in {"claude_shared", "codex_shared"} and "Tier" not in r["reason"] and "Non-default" not in r["reason"])


def test_zero_values_are_not_replaced_by_defaults(tmp_path: Path) -> None:
    quotas = _quotas()
    quotas["codex"]["windows"]["five_hour"]["reset_in_sec"] = 0
    reserve = compute_dynamic_codex_reserve(quotas, unfinished_tasks_count=4, pending_reviews_count=2)
    assert reserve["reserve_5h_fraction"] == 0.04  # fully decayed at reset, not treated as a fresh 5h window

    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    quotas["claude"]["windows"]["five_hour"] = {"remaining_fraction": 0.0, "used_percent": 100.0, "reset_in_sec": 0}
    pools = build_pool_states(quotas, None, learner)
    assert pools["claude_shared"]["five_hour"]["remaining_fraction"] == 0.0
    assert pools["claude_shared"]["five_hour"]["reset_in_sec"] == 1
    res = _route(SCAN_TASK, quotas, learner)
    cl = [r for r in res["rejected_models"] if r["pool_id"] == "claude_shared" and "window" in r["reason"]]
    assert cl and "5h window insufficient" in cl[0]["reason"]


def test_codex_reserve_breach_is_a_hard_gate(tmp_path: Path) -> None:
    learner = QuotaCostLearner(tmp_path / "t.sqlite")
    quotas = _quotas(agy=0.85, claude=0.85, codex=0.12)
    reserve = compute_dynamic_codex_reserve(quotas, unfinished_tasks_count=4, pending_reviews_count=2)
    pools = build_pool_states(quotas, None, learner, codex_reserve=reserve)
    cx = pools["codex_shared"]
    assert cx["five_hour"]["usable_after_orch_reserve"] < cx["five_hour"]["usable_before_reset"]
    sol_low = next(m for m in MODEL_CATALOG if m["model_id"] == "gpt-6.1-sol-low")
    pred = learner.predict_cost("codex_shared", "scan", "LOW")
    args = dict(
        model_entry=sol_low, pool_state=cx, cost_pred=pred, floor_tier="LOW", floor_rank=1, target_effort="low",
        required_tokens=32000, kind="scan", implementer_family="", preferred_provider="",
    )
    feasible, _, _, reason, _ = evaluate_candidate_schedule_score(dependency_role="leaf", is_explicit_match=True, **args)
    assert feasible is False
    assert "Codex orchestration reserve hard gate" in reason
    # The same quota without a reserve-protected pool label is feasible: the gate is the reserve, not the window.
    feasible_upstream, _, _, _, _ = evaluate_candidate_schedule_score(dependency_role="upstream", is_explicit_match=False, **args)
    assert feasible_upstream is True


def test_zero_delta_censored_migration_and_coordinator_ambiguity(tmp_path: Path) -> None:
    db = tmp_path / "learner.sqlite"
    learner = QuotaCostLearner(db)
    base = dict(
        provider="claude", model_id="claude-sonnet-5-5", pool_id="claude_shared", task_kind="implement", tier="MID",
        pre_5h_used_pct=20.0, pre_weekly_used_pct=30.0, concurrent_pool_jobs=1,
    )
    zero = learner.record_observation(task_id="Z-1", post_5h_used_pct=20.0, post_weekly_used_pct=30.0, **base)
    assert zero["attribution_status"] == "CENSORED_ZERO_DELTA"
    assert zero["measured_5h_cost"] is None and zero["measured_weekly_cost"] is None
    assert learner.predict_cost("claude_shared", "implement", "MID")["sample_count"] == 0

    busy = learner.record_observation(
        task_id="B-1", post_5h_used_pct=26.0, post_weekly_used_pct=32.0, active_coordinator_on_pool=True, **base
    )
    assert busy["attribution_status"] == "AMBIGUOUS_CONCURRENT"
    assert busy["measured_5h_cost"] is None

    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO task_cost_observations (id, task_id, pool_id, task_kind, tier, attribution_status, measured_5h_cost, measured_weekly_cost) "
            "VALUES ('OBS-LEGACY', 'JOB-AUTO-01', 'claude_shared', 'implement', 'MID', 'MEASURED_SINGLE_JOB', 0.0, 0.0)"
        )
        conn.commit()
    QuotaCostLearner(db)  # re-init runs the quarantine migration
    with sqlite3.connect(str(db)) as conn:
        status = conn.execute("SELECT attribution_status FROM task_cost_observations WHERE id = 'OBS-LEGACY'").fetchone()[0]
    assert status == "CENSORED_ZERO_DELTA"
    assert learner.get_summary()["measured_single_job_count"] == 0


# ---------------------------------------------------------------------------
# 4. Headless CLI safety & live activity
# ---------------------------------------------------------------------------


def test_prompt_never_reaches_argv_for_claude_and_codex() -> None:
    evil = 'fix bug "&calc&" | rm -rf'
    for model_id in ("claude-sonnet-5-5", "gpt-6.1-sol-medium"):
        m = next(x for x in MODEL_CATALOG if x["model_id"] == model_id)
        argv, display = build_headless_cli_command(m, evil, "C:\\repo", effort="medium", read_only=False)
        assert all(evil not in str(a) and "&calc&" not in str(a) for a in argv)
        assert orchestrator.prompt_transport_for(m["provider"]) == "stdin"
    codex_argv, _ = build_headless_cli_command(next(x for x in MODEL_CATALOG if x["model_id"] == "gpt-6.1-sol-medium"), evil, "C:\\repo")
    assert codex_argv[1:3] == ["exec", "-"]


def test_execute_feeds_stdin_uses_cwd_and_refuses_unsafe_cmd_shim(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    record = {
        "provider": "claude",
        "model": "fake",
        "cli_argv": [sys.executable, "-c", "import os,sys; print(os.getcwd()); print(sys.stdin.read().upper())"],
        "headless_cli": "fake",
        "workspace": str(ws),
        "prompt_transport": "stdin",
        "cli_stdin_prompt": "hello & calc",
    }
    res = execute_headless_dispatch(record, dry_run=False, timeout_sec=60)
    assert res["exit_code"] == 0
    assert str(ws).lower() in res["output"].lower()
    assert "HELLO & CALC" in res["output"]
    assert res["rate_limited"] is False

    shim = dict(record, provider="agy", prompt_transport="argv", cli_stdin_prompt=None, cli_argv=["C:\\tools\\agy.cmd", "-p=do & calc"])
    refused = execute_headless_dispatch(shim, dry_run=False)
    assert refused["exit_code"] == -3 and refused["executed"] is False
    assert "shell metacharacters" in refused["output"]

    missing_ws = execute_headless_dispatch(dict(record, workspace=str(tmp_path / "nope")), dry_run=False)
    assert missing_ws["exit_code"] == -3


def test_structured_rate_limit_detection() -> None:
    assert detect_rate_limit(0, "Fixed HTTP 429 handling in client.py", "") is False
    assert detect_rate_limit(1, "log mentions 429 somewhere", "") is False
    assert detect_rate_limit(1, "", "Error: rate limit exceeded, try again at 17:00") is True
    claude_json = json.dumps({"type": "result", "is_error": True, "result": "Claude AI usage limit reached|1792500000"})
    assert detect_rate_limit(0, claude_json, "") is True
    assert detect_rate_limit(0, json.dumps({"type": "result", "is_error": False, "result": "rate limit docs"}), "") is False


def test_live_activity_tracks_headless_claude_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    procs = [
        {"pid": 501, "name": "claude.exe", "cmdline": "claude -p --model claude-sonnet-5-5 --effort low --output-format json"},
        {"pid": 502, "name": "node.exe", "cmdline": "node C:\\npm\\node_modules\\@anthropic-ai\\claude-code\\cli.js --print --model claude-haiku-5-5"},
        {"pid": 503, "name": "claude.exe", "cmdline": "claude --plugin-dir C:\\x --permission-mode plan"},
    ]
    monkeypatch.setattr(live_mod, "_get_live_processes", lambda: procs)
    monkeypatch.setattr(live_mod, "_collect_agy_coordinators_and_tasks", lambda _l: ([], []))
    monkeypatch.setattr(live_mod, "_collect_codex_coordinators", lambda _p: [])
    monkeypatch.setattr(live_mod, "DEFAULT_CLAUDE_SESSIONS_DIR", tmp_path / "none")
    out = live_mod.collect_live_activity()
    claude_workers = {w["pid"]: w for w in out["workers"] if w["provider"] == "claude"}
    assert set(claude_workers) == {501, 502}
    assert claude_workers[501]["model"] == "claude-sonnet-5-5 (low)"
    assert claude_workers[502]["model"] == "claude-haiku-5-5"


# ---------------------------------------------------------------------------
# 5. DAG queue
# ---------------------------------------------------------------------------


def _fake_route(model: str = "gemini-3.8-flash-medium", pool: str = "agy_gemini", provider: str = "agy"):
    calls: List[Dict[str, Any]] = []

    def _route_fn(req, providers_quota, orch, operating_mode="endurance", live_activity=None, cost_learner=None):
        calls.append(req)
        return {
            "status": "routed",
            "assignment_record": {
                "task_id": f"WP-{len(calls)}",
                "task": req["task"],
                "provider": provider,
                "pool_id": pool,
                "model": model,
                "selected_tier": "MID",
                "predicted_cost": {"five_hour_mean": 0.03, "weekly_mean": 0.01},
            },
            "pool_states": {},
        }

    return _route_fn, calls


def _exec_with(result: Dict[str, Any]):
    calls: List[Dict[str, Any]] = []

    def _exec_fn(ar, dry_run=False, timeout_sec=60):
        calls.append(ar)
        return dict(result)

    return _exec_fn, calls


def _step(qm: BacklogQueueManager, learner: QuotaCostLearner, route_fn, exec_fn, max_jobs: int = 2, live: bool = True, now: float = 1_800_000_000.0, live_activity=None):
    return qm.step_supervisor(
        route_fn=route_fn,
        execute_fn=exec_fn,
        providers_quota=_quotas(),
        orchestrator_info={"active": "auto"},
        cost_learner=learner,
        max_dispatch_per_tick=max_jobs,
        execute_live=live,
        now_ts=now,
        live_activity=live_activity,
    )


def _job(db: Path, job_id: str) -> Dict[str, Any]:
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM backlog_jobs WHERE job_id = ?", (job_id,)).fetchone())


def _qm(tmp_path: Path):
    db = tmp_path / "queue.sqlite"
    return db, BacklogQueueManager(db), QuotaCostLearner(db)


def test_enqueue_validation_is_all_or_nothing(tmp_path: Path) -> None:
    db, qm, _ = _qm(tmp_path)
    bad_batches = [
        ([{"job_id": "A", "depends_on": ["A"]}], "itself"),
        ([{"job_id": "A", "depends_on": ["B"]}, {"job_id": "B", "depends_on": ["A"]}], "cycle"),
        ([{"job_id": "A", "depends_on": ["GHOST"]}], "unknown"),
        ([{"job_id": "A", "depends_on": "B"}], "must be a list"),
        ([{"job_id": "A", "edit_scope": "src/"}], "edit_scope"),
        ([{"job_id": "A"}, {"job_id": "A"}], "Duplicate"),
        ([{"job_id": "A", "satisfy_on": "merged"}], "satisfy_on"),
    ]
    for batch, needle in bad_batches:
        with pytest.raises(ValueError, match=needle):
            qm.enqueue_tasks([{"job_id": "OK-1"}] + batch)
    assert qm.get_queue_summary()["total_jobs"] == 0

    qm.enqueue_tasks([{"job_id": "C1", "depends_on": []}])
    with pytest.raises(ValueError, match="cycle"):
        qm.enqueue_tasks([{"job_id": "C2", "depends_on": ["C1"]}, {"job_id": "C1", "depends_on": ["C2"]}])


def test_enqueue_preserves_fields_and_attempt_count(tmp_path: Path) -> None:
    db, qm, _ = _qm(tmp_path)
    qm.enqueue_tasks(
        [
            {
                "job_id": "W1",
                "task": "Writer",
                "kind": "implement",
                "workspace": str(tmp_path),
                "edit_scope": ["src/a/", "README.md"],
                "explicit_model": "claude-sonnet-5-5",
                "explicit_effort": "LOW",
                "require_cross_family": True,
                "satisfy_on": "accepted",
                "max_attempts": 9,
            },
            {"job_id": "R1", "kind": "scan"},
        ]
    )
    w1 = _job(db, "W1")
    assert w1["workspace"] == str(tmp_path)
    assert json.loads(w1["edit_scope_json"]) == ["src/a/", "README.md"]
    assert w1["read_only"] == 0 and _job(db, "R1")["read_only"] == 1
    assert (w1["explicit_model"], w1["explicit_effort"]) == ("claude-sonnet-5-5", "low")
    assert w1["require_cross_family"] == 1 and w1["satisfy_on"] == "accepted"
    assert w1["max_attempts"] == 2  # policy max_attempts is a ceiling

    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET attempt_count = 1 WHERE job_id = 'W1'")
        conn.execute("UPDATE backlog_jobs SET status = 'FAILED_ESCALATED', attempt_count = 2 WHERE job_id = 'R1'")
        conn.commit()
    out = qm.enqueue_tasks([{"job_id": "W1", "kind": "implement"}, {"job_id": "R1", "kind": "scan"}], preserve_existing=True)
    assert _job(db, "W1")["attempt_count"] == 1
    assert _job(db, "R1")["status"] == "FAILED_ESCALATED"
    assert out[1]["preserved_without_duplicate"] is True
    qm.enqueue_tasks([{"job_id": "W1", "kind": "implement"}], preserve_existing=False)
    assert _job(db, "W1")["attempt_count"] == 1


def test_clear_queue_keeps_running_rows_unless_forced(tmp_path: Path) -> None:
    db, qm, _ = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "A"}, {"job_id": "B"}])
    qm.mark_job_running("A", "agy", "agy_gemini", "gemini-3.8-flash-medium", worker_pid=4242)
    res = qm.clear_queue()
    assert res == {"deleted": 1, "kept_running": 1, "forced": False}
    assert qm.get_queue_summary()["counts"]["RUNNING"] == 1
    assert qm.clear_queue(force=True)["deleted"] == 1


def test_starvation_free_ready_selection(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    tasks = [{"job_id": "GATE", "kind": "scan", "priority": 1}]
    tasks += [{"job_id": f"HI-{i:02d}", "kind": "scan", "priority": 500 - i, "depends_on": ["GATE"]} for i in range(20)]
    tasks += [{"job_id": "LOW-READY", "kind": "scan", "priority": 0}]
    qm.enqueue_tasks(tasks, now_ts=1_800_000_000.0)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET status = 'PAUSED_WAIT_RESET', next_eligible_at = 1900000000 WHERE job_id = 'GATE'")
        conn.commit()
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 0, "output": "ok"})
    res = _step(qm, learner, route_fn, exec_fn, max_jobs=1)
    assert [d["job_id"] for d in res["dispatched"]] == ["LOW-READY"]
    assert len(res["blocked_by_dependencies"]) == 20


def test_upstream_failure_propagates_transitively(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "A"}, {"job_id": "B", "depends_on": ["A"]}, {"job_id": "C", "depends_on": ["B"]}])
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET status = 'FAILED_ESCALATED' WHERE job_id = 'A'")
        conn.commit()
    route_fn, calls = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 0})
    res = _step(qm, learner, route_fn, exec_fn)
    assert _job(db, "B")["status"] == "BLOCKED_UPSTREAM_FAILED"
    assert _job(db, "C")["status"] == "BLOCKED_UPSTREAM_FAILED"
    assert {b["job_id"] for b in res["blocked_by_dependencies"] if b.get("status") == "BLOCKED_UPSTREAM_FAILED"} == {"B", "C"}
    assert calls == []


def test_corrupt_dependency_json_fails_closed(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "A"}])
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET depends_on_json = '{oops' WHERE job_id = 'A'")
        conn.commit()
    route_fn, calls = _fake_route()
    res = _step(qm, learner, route_fn, _exec_with({"exit_code": 0})[0])
    assert calls == [] and "corrupt" in res["blocked_by_dependencies"][0]["error"]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (["src/a/"], ["src/a/x.py"], True),
        (["src/a.py"], ["src/a.py"], True),
        (["src\\A\\"], ["src/a/b.py"], sys.platform.startswith("win")),
        (["src/a/"], ["src/ab/x.py"], False),
        (["src/*.py"], ["src/x.py"], True),
        (["src/foo*"], ["src/foobar/x.py"], True),
        (["src/a*"], ["src/b*"], False),
        (["*"], ["anything.txt"], True),
        ([], ["src/"], False),
        (["./docs/"], ["docs/readme.md"], True),
        (["tests/"], ["src/"], False),
    ],
)
def test_scopes_overlap_table(a: List[str], b: List[str], expected: bool) -> None:
    assert scopes_overlap(a, b) is expected
    assert scopes_overlap(b, a) is expected


def test_scope_helpers_relativize_absolute_entries(tmp_path: Path) -> None:
    ws = str(tmp_path)
    assert scopes_overlap([str(tmp_path / "src" / "a.py")], ["src/"], ws, ws) is True
    assert path_in_scope("src/x/y.py", ["src/x/"], ws) is True
    assert path_in_scope(str(tmp_path / "src" / "x" / "y.py"), ["src/x/"], ws) is True
    assert path_in_scope("../other/y.py", ["src/"], ws) is False
    assert path_in_scope("docs/y.md", ["src/"], ws) is False


def test_edit_scope_collision_gate(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    ws = str(tmp_path)
    qm.enqueue_tasks(
        [
            {"job_id": "W-A", "kind": "implement", "workspace": ws, "edit_scope": ["src/a/"], "priority": 10},
            {"job_id": "W-B", "kind": "implement", "workspace": ws, "edit_scope": ["src/a/x.py"], "priority": 9},
            {"job_id": "W-C", "kind": "implement", "workspace": ws, "edit_scope": ["tests/"], "priority": 8},
            {"job_id": "R-D", "kind": "scan", "workspace": ws, "edit_scope": ["src/a/"], "priority": 7},
        ]
    )
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 0})
    res = _step(qm, learner, route_fn, exec_fn, max_jobs=4)
    assert [d["job_id"] for d in res["dispatched"]] == ["W-A", "W-C", "R-D"]
    assert res["blocked_by_scope_overlap"] == [{"job_id": "W-B", "overlaps_with": "W-A", "edit_scope": ["src/a/x.py"]}]

    # A RUNNING writer in another worktree of the same logical repo also holds its lease.
    db2, qm2, learner2 = _qm(tmp_path / "second")
    qm2.enqueue_tasks(
        [
            {"job_id": "RUN-1", "kind": "implement", "workspace": ws, "edit_scope": ["lib/"]},
            {"job_id": "NEW-1", "kind": "implement", "workspace": ws, "edit_scope": ["lib/core.py"]},
        ]
    )
    qm2.mark_job_running("RUN-1", "claude", "claude_shared", "claude-sonnet-5-5", worker_pid=99)
    res2 = _step(qm2, learner2, route_fn, exec_fn)
    assert res2["dispatched"] == []
    assert res2["blocked_by_scope_overlap"][0]["overlaps_with"] == "RUN-1"


def test_rate_limit_preserves_attempts_and_records_failure(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "RL", "kind": "scan"}])
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 1, "rate_limited": True, "stderr_tail": "429"})
    res = _step(qm, learner, route_fn, exec_fn)
    row = _job(db, "RL")
    assert row["status"] == "PAUSED_WAIT_RESET"
    assert row["attempt_count"] == 0 and row["infra_failures"] == 1
    assert res["rate_limited"][0]["job_id"] == "RL"
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_cost_observations WHERE task_id = 'RL' AND success = 0").fetchone()[0] == 1


def test_failures_consume_attempts_escalate_and_are_observed(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "F", "kind": "scan"}])
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 2, "stderr_tail": "boom"})
    first = _step(qm, learner, route_fn, exec_fn)
    assert first["failed"][0]["status"] == "QUEUED" and _job(db, "F")["attempt_count"] == 1
    _step(qm, learner, route_fn, exec_fn, now=1_800_000_100.0)
    assert _job(db, "F")["status"] == "FAILED_ESCALATED" and _job(db, "F")["attempt_count"] == 2
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_cost_observations WHERE task_id = 'F' AND success = 0").fetchone()[0] == 2

    qm.enqueue_tasks([{"job_id": "NOEXIT", "kind": "scan"}])
    _step(qm, learner, route_fn, _exec_with({"output": "no exit code"})[0], now=1_800_000_200.0)
    assert _job(db, "NOEXIT")["status"] == "QUEUED" and _job(db, "NOEXIT")["attempt_count"] == 1


def test_preview_tick_writes_nothing(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "P1", "kind": "scan"}, {"job_id": "P2", "kind": "scan", "depends_on": ["P1"]}])
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET status = 'PAUSED_WAIT_RESET', next_eligible_at = 0 WHERE job_id = 'P1'")
        conn.commit()

    def _dump() -> tuple:
        with sqlite3.connect(str(db)) as conn:
            return (
                conn.execute("SELECT * FROM backlog_jobs ORDER BY job_id").fetchall(),
                conn.execute("SELECT COUNT(*) FROM task_cost_observations").fetchone()[0],
            )

    before = _dump()
    route_fn, _ = _fake_route()
    exec_fn, exec_calls = _exec_with({"exit_code": 0})
    res = _step(qm, learner, route_fn, exec_fn, live=False)
    assert res["preview"] is True
    assert [w["job_id"] for w in res["would_dispatch"]] == ["P1"]
    assert res["dispatched"] == [] and exec_calls == []
    assert _dump() == before


def test_satisfy_on_accepted_waits_for_review_acceptance(tmp_path: Path) -> None:
    db, qm, learner = _qm(tmp_path)
    ws = str(tmp_path)
    qm.enqueue_tasks(
        [
            {"job_id": "IMPL", "kind": "implement", "workspace": ws, "edit_scope": ["src/"], "priority": 10},
            {"job_id": "NEXT", "kind": "implement", "workspace": ws, "edit_scope": ["docs/"], "depends_on": ["IMPL"], "satisfy_on": "accepted", "priority": 5},
            {"job_id": "LOOSE", "kind": "scan", "depends_on": ["IMPL"], "priority": 4},
        ]
    )
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 0})
    t1 = _step(qm, learner, route_fn, exec_fn, max_jobs=3)
    assert [d["job_id"] for d in t1["dispatched"]] == ["IMPL", "LOOSE"]
    assert any(b["job_id"] == "NEXT" for b in t1["blocked_by_dependencies"])

    sub = qm.submit_for_review("IMPL", changed_files=["src/a.py", "src\\b.py"])
    assert sub["acceptance_status"] == "PENDING_REVIEW"
    assert sub["implementer_agent_id"].startswith("hub-run:RUN-")
    t2 = _step(qm, learner, route_fn, exec_fn, now=1_800_000_010.0)
    assert t2["dispatched"] == []

    verdict = qm.record_review_verdict("IMPL", reviewer_agent_id="claude-reviewer-1", accepted=True, findings="LGTM", cost_learner=learner)
    assert verdict["acceptance_status"] == "ACCEPTED"
    assert verdict["downstream_unblocked"] == ["NEXT"]
    t3 = _step(qm, learner, route_fn, exec_fn, now=1_800_000_020.0)
    assert [d["job_id"] for d in t3["dispatched"]] == ["NEXT"]


# ---------------------------------------------------------------------------
# 6. Refinery gate
# ---------------------------------------------------------------------------


def _completed_job(tmp_path: Path, **task: Any):
    db, qm, learner = _qm(tmp_path)
    spec = {"job_id": "J", "kind": "implement", "workspace": str(tmp_path), "edit_scope": ["src/"]}
    spec.update(task)
    qm.enqueue_tasks([spec])
    route_fn, _ = _fake_route()
    _step(qm, learner, route_fn, _exec_with({"exit_code": 0})[0])
    assert _job(db, "J")["status"] == "COMPLETED"
    return db, qm, learner


def test_independent_review_is_enforced(tmp_path: Path) -> None:
    db, qm, learner = _completed_job(tmp_path)
    with pytest.raises(ValueError, match="pending review"):
        qm.record_review_verdict("J", reviewer_agent_id="rev", accepted=True)
    sub = qm.submit_for_review("J", changed_files=["src/x.py"], implementer_agent_id="agy-session-7")
    assert sub["acceptance_status"] == "PENDING_REVIEW"
    for bad in ("agy-session-7", "", "   "):
        with pytest.raises(ValueError, match="Independent review violation"):
            qm.record_review_verdict("J", reviewer_agent_id=bad, accepted=True)
    ok = qm.record_review_verdict("J", reviewer_agent_id="gemini-reviewer", accepted=True, reviewer_family="gemini", cost_learner=learner)
    assert ok["acceptance_status"] == "ACCEPTED"  # same family is fine when cross-family is optional
    with pytest.raises(KeyError):
        qm.submit_for_review("NOPE", changed_files=[])


def test_required_cross_family_review(tmp_path: Path) -> None:
    db, qm, learner = _completed_job(tmp_path, require_cross_family=True)
    qm.submit_for_review("J", changed_files=["src/x.py"])
    for fam in ("gemini", None):
        with pytest.raises(ValueError, match="Cross-family review required"):
            qm.record_review_verdict("J", reviewer_agent_id="rev-1", accepted=True, reviewer_family=fam)
    assert qm.record_review_verdict("J", reviewer_agent_id="rev-1", accepted=True, reviewer_family="claude")["accepted"] is True


def test_scope_violation_and_rejection_rework_then_escalate(tmp_path: Path) -> None:
    db, qm, learner = _completed_job(tmp_path)
    res = qm.submit_for_review("J", changed_files=["src/ok.py", "docs/leak.md"])
    assert res["acceptance_status"] == "SCOPE_VIOLATION"
    assert res["outside_edit_scope"] == ["docs/leak.md"]
    assert res["status"] == "QUEUED" and _job(db, "J")["status"] == "QUEUED"

    route_fn, _ = _fake_route()
    _step(qm, learner, route_fn, _exec_with({"exit_code": 0})[0], now=1_800_000_100.0)
    assert _job(db, "J")["attempt_count"] == 2
    qm.submit_for_review("J", changed_files=["src/ok.py"], checks_passed=True)
    rej = qm.record_review_verdict("J", reviewer_agent_id="rev", accepted=False, findings="missing edge case", cost_learner=learner)
    assert rej["status"] == "FAILED_ESCALATED"
    assert _job(db, "J")["acceptance_status"] == "REJECTED"


def test_checks_failed_sends_back_for_rework(tmp_path: Path) -> None:
    db, qm, _ = _completed_job(tmp_path)
    res = qm.submit_for_review("J", changed_files=["src/a.py"], checks_passed=False)
    assert res["acceptance_status"] == "CHECKS_FAILED" and res["status"] == "QUEUED"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_submit_inspects_git_including_both_sides_of_renames(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "docs" / "b.md").write_text("# b\n", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, stdin=subprocess.DEVNULL)

    git("init", "-q")
    git("-c", "user.email=t@example.invalid", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "base")
    (repo / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
    (repo / "src" / "new.py").write_text("n = 1\n", encoding="utf-8")

    db, qm, learner = _qm(tmp_path / "q")
    qm.enqueue_tasks([{"job_id": "G", "kind": "implement", "workspace": str(repo), "edit_scope": ["src/"]}])
    route_fn, _ = _fake_route()
    _step(qm, learner, route_fn, _exec_with({"exit_code": 0})[0])
    clean = qm.submit_for_review("G")
    assert clean["changed_files_source"] == "git"
    assert sorted(clean["changed_files"]) == ["src/a.py", "src/new.py"]
    assert clean["acceptance_status"] == "PENDING_REVIEW"

    git("mv", "docs/b.md", "src/b.md")
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET acceptance_status = NULL WHERE job_id = 'G'")
        conn.commit()
    moved = qm.submit_for_review("G")
    assert moved["acceptance_status"] == "SCOPE_VIOLATION"
    assert moved["outside_edit_scope"] == ["docs/b.md"]

    with pytest.raises(ValueError, match="Cannot inspect changed files"):
        qm2 = BacklogQueueManager(tmp_path / "q2.sqlite")
        qm2.enqueue_tasks([{"job_id": "NG", "kind": "implement", "workspace": str(tmp_path / "not-a-repo"), "edit_scope": ["src/"]}])
        with sqlite3.connect(str(tmp_path / "q2.sqlite")) as conn:
            conn.execute("UPDATE backlog_jobs SET status = 'COMPLETED', run_id = 'RUN-X' WHERE job_id = 'NG'")
            conn.commit()
        qm2.submit_for_review("NG")


# ---------------------------------------------------------------------------
# 7. Witness
# ---------------------------------------------------------------------------


def test_witness_classification_table() -> None:
    pol = load_orchestrator_policy()
    now = 10_000.0
    cases = [
        ({"status": "COMPLETED"}, "DONE"),
        ({"status": "FAILED_ESCALATED"}, "DONE"),
        ({"status": "PAUSED_WAIT_RESET"}, "WAITING_RESOURCE"),
        ({"status": "RUNNING", "alive": False, "started_at": now - 5}, "DEAD"),
        ({"status": "RUNNING", "started_at": now - 30, "last_progress_at": now - 30}, "STARTING"),
        ({"status": "RUNNING", "started_at": now - 900, "last_progress_at": now - 60}, "PRODUCTIVE"),
        ({"status": "RUNNING", "started_at": now - 2000, "last_progress_at": now - 900}, "WAITING_RESOURCE"),
        ({"status": "RUNNING", "started_at": now - 600, "build_bottleneck": True}, "WAITING_RESOURCE"),
        ({"status": "RUNNING", "started_at": now - 5000, "last_progress_at": now - 2400}, "STALLED"),
        ({"status": "RUNNING", "alive": True}, "PRODUCTIVE"),
    ]
    for signal, expected in cases:
        assert classify_worker_health(signal, now, pol) == expected, signal


def test_witness_report_is_advisory(tmp_path: Path) -> None:
    db, qm, _ = _qm(tmp_path)
    qm.enqueue_tasks([{"job_id": "LIVE"}, {"job_id": "GONE"}])
    qm.mark_job_running("LIVE", "claude", "claude_shared", "claude-sonnet-5-5", worker_pid=111, now_ts=1000.0)
    qm.mark_job_running("GONE", "claude", "claude_shared", "claude-sonnet-5-5", worker_pid=222, now_ts=1000.0)
    live = {"workers": [{"pid": 111, "provider": "claude", "model": "x"}, {"pid": 333, "provider": "codex", "model": "y"}]}
    rep = qm.get_witness_report(live_activity=live, now_ts=1000.0 + 3000)
    health = {j["job_id"]: j["health"] for j in rep["jobs"]}
    assert health == {"LIVE": "STALLED", "GONE": "DEAD"}
    assert [w["pid"] for w in rep["external_workers"]] == [333]
    assert rep["actions_taken"] == []
    assert qm.get_queue_summary()["counts"]["RUNNING"] == 2  # nothing killed or requeued


# ---------------------------------------------------------------------------
# 8. Hub security, lifespan and state
# ---------------------------------------------------------------------------


def test_import_starts_no_supervisor_and_live_execution_defaults_off() -> None:
    assert main._SUPERVISOR_THREAD is None
    assert main._RUNTIME_STATE["supervisor_config"]["execute_live"] is False
    assert str(main.STATE_FILE).startswith(os.environ["QUOTA_HUB_HOME"])
    assert str(orchestrator.DEFAULT_TELEMETRY_DB).startswith(os.environ["QUOTA_HUB_HOME"])
    with TestClient(main.app) as client:  # lifespan runs, but QUOTA_HUB_DISABLE_SUPERVISOR=1
        assert client.get("/api/policy").status_code == 200
        assert main._SUPERVISOR_THREAD is None


def test_host_and_origin_guard() -> None:
    client = TestClient(main.app)
    assert client.get("/api/policy").status_code == 200
    assert client.get("/api/policy", headers={"host": "127.0.0.1:8765"}).status_code == 200
    assert client.get("/api/policy", headers={"host": "evil.example"}).status_code == 403
    assert client.get("/api/policy", headers={"host": "localhost.evil.example:8765"}).status_code == 403
    assert client.post("/api/mode", json={"mode": "endurance"}, headers={"origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/policy", headers={"origin": "null"}).status_code == 403
    assert client.get("/api/policy", headers={"origin": "http://testserver"}).status_code == 200
    resp = client.get("/api/policy", headers={"origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_state_is_compact_atomic_and_never_persists_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "quota_hub_state.json"
    monkeypatch.setattr(main, "STATE_FILE", state_file)
    saved = copy.deepcopy(main._RUNTIME_STATE)
    try:
        heavy_pools = {f"pool{i}": {"five_hour": {"x": "y" * 2000}} for i in range(20)}
        entry = {
            "id": "WP-1",
            "timestamp": "t",
            "request": {"task": "x"},
            "routing": {
                "status": "routed",
                "pool_states": heavy_pools,
                "ranked_candidates": [{"math": "z" * 5000}] * 6,
                "rejected_models": [{"model_id": "m", "pool_id": "p", "reason": "r"}] * 8,
                "assignment_record": {"task_id": "WP-1", "provider": "agy", "math_breakdown": {"big": "q" * 5000}, "cli_stdin_prompt": "secret prompt"},
            },
            "execution": {"mode": "dry_run", "exit_code": 0, "output": "o" * 10000},
        }
        main._RUNTIME_STATE["dispatch_history"] = [copy.deepcopy(entry) for _ in range(50)]
        main._RUNTIME_STATE["simulation_overrides"] = {"codex": {"sol": 0.99}}
        main._save_state()
        raw = state_file.read_text(encoding="utf-8")
        assert len(raw.encode("utf-8")) < 50_000
        data = json.loads(raw)
        assert "simulation_overrides" not in data
        assert "pool_states" not in raw and "ranked_candidates" not in raw and "secret prompt" not in raw
        assert len(data["dispatch_history"]) == main.HISTORY_LIMIT
        assert list(tmp_path.glob("*.tmp")) == []

        data["supervisor_config"]["execute_live"] = True
        data["simulation_overrides"] = {"claude": {"sonnet": 1.0}}
        data["dispatch_history"] = [entry]
        state_file.write_text(json.dumps(data), encoding="utf-8")
        main._load_state()
        assert main._RUNTIME_STATE["supervisor_config"]["execute_live"] is False
        assert main._RUNTIME_STATE["simulation_overrides"] == {}
        assert "pool_states" not in json.dumps(main._RUNTIME_STATE["dispatch_history"])
    finally:
        main._RUNTIME_STATE.clear()
        main._RUNTIME_STATE.update(saved)


def test_refinery_witness_and_enqueue_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(main.app)
    monkeypatch.setattr(main, "collect_live_activity", lambda: {"workers": [], "coordinators": []})
    qm = orchestrator.QUEUE_MANAGER
    qm.clear_queue(force=True)

    bad = client.post("/api/queue/enqueue", json={"tasks": [{"job_id": "X", "depends_on": ["X"]}]})
    assert bad.status_code == 400 and "itself" in bad.json()["detail"]

    qm.enqueue_tasks([{"job_id": "API-J", "kind": "implement", "edit_scope": ["src/"]}])
    with sqlite3.connect(str(orchestrator.DEFAULT_TELEMETRY_DB)) as conn:
        conn.execute("UPDATE backlog_jobs SET status = 'COMPLETED', run_id = 'RUN-API' WHERE job_id = 'API-J'")
        conn.commit()
    sub = client.post("/api/refinery/submit", json={"job_id": "API-J", "changed_files": ["src/x.py"]})
    assert sub.status_code == 200 and sub.json()["acceptance_status"] == "PENDING_REVIEW"
    same = client.post("/api/refinery/verdict", json={"job_id": "API-J", "reviewer_agent_id": "hub-run:RUN-API", "accepted": True})
    assert same.status_code == 400 and "Independent review violation" in same.json()["detail"]
    ok = client.post("/api/refinery/verdict", json={"job_id": "API-J", "reviewer_agent_id": "codex-reviewer", "accepted": True})
    assert ok.status_code == 200 and ok.json()["acceptance_status"] == "ACCEPTED"
    assert client.post("/api/refinery/submit", json={"job_id": "MISSING"}).status_code == 404

    wit = client.get("/api/witness")
    assert wit.status_code == 200 and wit.json()["actions_taken"] == []
    assert client.get("/api/policy").json()["max_leaves"] == 10
    qm.clear_queue(force=True)


def test_dispatch_refuses_live_execution_while_simulating() -> None:
    client = TestClient(main.app)
    saved = copy.deepcopy(main._RUNTIME_STATE.get("simulation_overrides"))
    try:
        main._RUNTIME_STATE["simulation_overrides"] = {"codex": {"sol": 0.9}}
        resp = client.post("/api/dispatch", json={"task": "t", "kind": "scan", "execute_live": True})
        assert resp.status_code == 409 and "SIMULATION_ACTIVE" in resp.json()["detail"]
    finally:
        main._RUNTIME_STATE["simulation_overrides"] = saved or {}


def test_cli_policy_and_refusal_exit_codes() -> None:
    env = dict(os.environ)
    pol = subprocess.run([sys.executable, "cli.py", "policy"], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert pol.returncode == 0, pol.stderr
    assert json.loads(pol.stdout)["_source"] == "impl-default"
    ver = subprocess.run(
        [sys.executable, "cli.py", "verdict", "NO-SUCH-JOB", "--reviewer", "r", "--accept"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert ver.returncode == 1
    assert json.loads(ver.stdout)["status"] == "refused"


# ---------------------------------------------------------------------------
# 9. Regressions from the independent review (REVIEW_OF_WP_HUB_CORE_BY_CODEX)
# ---------------------------------------------------------------------------


def test_accepted_edge_requires_explicit_acceptance_for_read_only_upstream(tmp_path: Path) -> None:
    sat = BacklogQueueManager._dependency_satisfied
    for acceptance in (None, "", "PENDING_REVIEW", "REJECTED", "SCOPE_VIOLATION", "CHECKS_FAILED"):
        assert sat({"status": "COMPLETED", "acceptance_status": acceptance, "read_only": 1}, "accepted") is False, acceptance
    assert sat({"status": "COMPLETED", "acceptance_status": "ACCEPTED", "read_only": 1}, "accepted") is True
    assert sat({"status": "RUNNING", "acceptance_status": "ACCEPTED", "read_only": 1}, "accepted") is False
    assert sat({"status": "COMPLETED", "acceptance_status": None, "read_only": 1}, "completed") is True

    db, qm, learner = _qm(tmp_path)
    qm.enqueue_tasks(
        [
            {"job_id": "DESIGN", "kind": "design", "priority": 10},
            {"job_id": "BUILD", "kind": "implement", "workspace": str(tmp_path), "edit_scope": ["src/"], "depends_on": ["DESIGN"], "satisfy_on": "accepted", "priority": 5},
        ]
    )
    assert _job(db, "DESIGN")["read_only"] == 1
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 0})
    t1 = _step(qm, learner, route_fn, exec_fn)
    assert [d["job_id"] for d in t1["dispatched"]] == ["DESIGN"]
    assert _job(db, "DESIGN")["status"] == "COMPLETED"
    t2 = _step(qm, learner, route_fn, exec_fn, now=1_800_000_010.0)
    assert t2["dispatched"] == [] and any(b["job_id"] == "BUILD" for b in t2["blocked_by_dependencies"])

    assert qm.submit_for_review("DESIGN", changed_files=[])["acceptance_status"] == "PENDING_REVIEW"
    assert _step(qm, learner, route_fn, exec_fn, now=1_800_000_020.0)["dispatched"] == []
    qm.record_review_verdict("DESIGN", reviewer_agent_id="design-reviewer", accepted=True, cost_learner=learner)
    t4 = _step(qm, learner, route_fn, exec_fn, now=1_800_000_030.0)
    assert [d["job_id"] for d in t4["dispatched"]] == ["BUILD"]


def test_path_in_scope_canonicalizes_dot_segments(tmp_path: Path) -> None:
    ws = str(tmp_path)
    escaping = [
        "src/../outside.py",
        "src/./../outside.py",
        "src\\..\\outside.py",
        "src/a/../../outside.py",
        "./src/../../sibling/x.py",
        str(tmp_path / "src" / ".." / "outside.py"),
        str(tmp_path / "src" / ".." / ".." / "elsewhere" / "x.py"),
        str(tmp_path.parent / "elsewhere.py"),
    ]
    for path in escaping:
        assert path_in_scope(path, ["src/"], ws) is False, path
    assert path_in_scope("../x.py", ["*"], ws) is False
    assert path_in_scope(str(tmp_path.parent / "elsewhere.py"), ["*"], ws) is False
    assert path_in_scope("src/../outside.py", ["*"], ws) is True  # resolves to a file inside the workspace

    for path in ("src/x/../y.py", "src/./y.py", "lib/../src/y.py", str(tmp_path / "src" / "x" / ".." / "y.py")):
        assert path_in_scope(path, ["src/"], ws) is True, path
    assert path_in_scope("src/a.py", ["lib/../src/"], ws) is True
    assert path_in_scope("src/.env", ["src/.*"], ws) is True
    assert path_in_scope("srcx/.env", ["src/.*"], ws) is False

    # collision checks use the same canonical form
    assert scopes_overlap(["src/../lib/"], ["lib/core.py"]) is True
    assert scopes_overlap(["src/../lib/"], ["src/a.py"]) is False
    db, qm, _ = _qm(tmp_path)
    with pytest.raises(ValueError, match="after a wildcard"):
        qm.enqueue_tasks([{"job_id": "W", "kind": "implement", "edit_scope": ["src/*/../../x.py"]}])


def test_submit_rejects_traversal_as_scope_violation(tmp_path: Path) -> None:
    db, qm, _ = _completed_job(tmp_path)
    abs_escape = str(tmp_path / "src" / ".." / "abs_out.py")
    res = qm.submit_for_review("J", changed_files=["src/ok.py", "src/../outside.py", abs_escape])
    assert res["acceptance_status"] == "SCOPE_VIOLATION"
    assert res["outside_edit_scope"] == ["src/../outside.py", abs_escape]
    assert _job(db, "J")["status"] == "QUEUED"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_nested_workspaces_compare_scopes_in_repo_root_coordinates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "collectors").mkdir(parents=True)
    (repo / "collectors" / "live_activity.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "live_activity.py").write_text("y = 1\n", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, stdin=subprocess.DEVNULL)

    git("init", "-q")
    git("-c", "user.email=t@example.invalid", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "base")
    root, sub = str(repo), str(repo / "collectors")
    assert resolve_repo_key(root) == resolve_repo_key(sub)

    assert scopes_overlap(["collectors/live_activity.py"], ["live_activity.py"], root, sub) is True
    assert scopes_overlap(["live_activity.py"], ["collectors/live_activity.py"], sub, root) is True
    assert scopes_overlap(["live_activity.py"], ["live_activity.py"], root, sub) is False  # different files
    assert scopes_overlap(["collectors/"], ["*"], root, sub) is True
    assert scopes_overlap(["main.py"], ["../main.py"], root, sub) is True
    assert scopes_overlap([str(repo / "collectors" / "live_activity.py")], ["live_activity.py"], root, sub) is True

    # A linked worktree of the same repository shares the repo-root coordinates.
    wt = tmp_path / "wt"
    git("worktree", "add", "-q", "--detach", str(wt))
    wt_sub = str(wt / "collectors")
    assert resolve_repo_key(wt_sub) == resolve_repo_key(root)
    assert scopes_overlap(["collectors/live_activity.py"], ["live_activity.py"], root, wt_sub) is True
    assert scopes_overlap(["collectors/other.py"], ["live_activity.py"], root, wt_sub) is False

    # The supervisor gate applies the same mapping.
    db, qm, learner = _qm(tmp_path / "q")
    qm.enqueue_tasks(
        [
            {"job_id": "ROOT-W", "kind": "implement", "workspace": root, "edit_scope": ["collectors/live_activity.py"], "priority": 10},
            {"job_id": "SUB-W", "kind": "implement", "workspace": sub, "edit_scope": ["live_activity.py"], "priority": 9},
            {"job_id": "WT-W", "kind": "implement", "workspace": wt_sub, "edit_scope": ["live_activity.py"], "priority": 8},
            {"job_id": "SUB-OK", "kind": "implement", "workspace": sub, "edit_scope": ["other.py"], "priority": 7},
        ]
    )
    qm.mark_job_running("ROOT-W", "claude", "claude_shared", "claude-sonnet-5-5", worker_pid=99)
    route_fn, _ = _fake_route()
    res = _step(qm, learner, route_fn, _exec_with({"exit_code": 0})[0], max_jobs=4)
    assert [d["job_id"] for d in res["dispatched"]] == ["SUB-OK"]
    assert {b["job_id"]: b["overlaps_with"] for b in res["blocked_by_scope_overlap"]} == {"SUB-W": "ROOT-W", "WT-W": "ROOT-W"}


def test_rejected_replacement_batch_keeps_existing_queue() -> None:
    client = TestClient(main.app)
    qm = orchestrator.QUEUE_MANAGER
    qm.clear_queue(force=True)

    def _ids() -> set:
        with sqlite3.connect(str(orchestrator.DEFAULT_TELEMETRY_DB)) as conn:
            return {r[0] for r in conn.execute("SELECT job_id FROM backlog_jobs").fetchall()}

    try:
        qm.enqueue_tasks([{"job_id": "EXISTING"}, {"job_id": "HELD", "depends_on": ["EXISTING"]}, {"job_id": "LIVE"}])
        qm.mark_job_running("LIVE", "claude", "claude_shared", "claude-sonnet-5-5", worker_pid=4242)
        before = qm.get_queue_summary()["counts"]
        bad_batches = [
            [{"job_id": "BAD", "depends_on": ["BAD"]}],
            [{"job_id": "NEW", "depends_on": ["EXISTING"]}],  # EXISTING would be cleared by the replace
            [{"job_id": "OK"}, {"job_id": "BAD", "edit_scope": "src/"}],
        ]
        for batch in bad_batches:
            resp = client.post("/api/queue/enqueue", json={"tasks": batch, "clear_existing": True})
            assert resp.status_code == 400, batch
            assert _ids() == {"EXISTING", "HELD", "LIVE"}
            assert qm.get_queue_summary()["counts"] == before
        with pytest.raises(ValueError, match="unknown"):
            qm.enqueue_tasks([{"job_id": "NEW", "depends_on": ["HELD"]}], clear_existing=True)
        assert _ids() == {"EXISTING", "HELD", "LIVE"}

        ok = client.post("/api/queue/enqueue", json={"tasks": [{"job_id": "NEXT", "depends_on": ["LIVE"]}], "clear_existing": True})
        assert ok.status_code == 200, ok.text
        assert ok.json()["cleared"] == {"deleted": 2, "kept_running": 1, "forced": False}
        assert _ids() == {"LIVE", "NEXT"}
        assert qm.get_queue_summary()["counts"]["RUNNING"] == 1
    finally:
        qm.clear_queue(force=True)


# ---------------------------------------------------------------------------
# 10. Acceptance Audit Regressions (Refinery Integrity, Dry-Run, Constraints, Concurrency)
# ---------------------------------------------------------------------------


def test_audit_refinery_post_approval_and_post_submission_edit_invalidates_acceptance(tmp_path: Path) -> None:
    """
    Audit 1: Any file modification on disk after PENDING_REVIEW or ACCEPTED automatically
    invalidates acceptance_status -> INVALIDATED_BY_EDIT, blocks downstream satisfy_on='accepted'
    dependents, rejects stale review verdicts, and requires re-submission + fresh independent approval.
    """
    ws = tmp_path / "ws"
    (ws / "src" / "agy_pkg").mkdir(parents=True)
    target = ws / "src" / "agy_pkg" / "scope_guard.py"
    target.write_text("def extract_literal_prefix(pattern: str) -> str:\n    return pattern\n", encoding="utf-8")

    db, qm, learner = _qm(tmp_path / "q")
    qm.enqueue_tasks(
        [
            {"job_id": "LIVE-AGY-01", "kind": "implement", "workspace": str(ws), "edit_scope": ["src/agy_pkg/scope_guard.py"], "priority": 10},
            {"job_id": "DOWNSTREAM-01", "kind": "implement", "workspace": str(ws), "edit_scope": ["docs/"], "depends_on": ["LIVE-AGY-01"], "satisfy_on": "accepted", "priority": 5},
        ]
    )
    route_fn, _ = _fake_route()
    exec_fn, _ = _exec_with({"exit_code": 0})

    # 1. Dispatch LIVE-AGY-01
    t1 = _step(qm, learner, route_fn, exec_fn)
    assert [d["job_id"] for d in t1["dispatched"]] == ["LIVE-AGY-01"]

    # 2. Submit for review (binds SHA-256 content hash)
    sub1 = qm.submit_for_review("LIVE-AGY-01", changed_files=["src/agy_pkg/scope_guard.py"], implementer_agent_id="agy-worker-1")
    assert sub1["acceptance_status"] == "PENDING_REVIEW"
    assert len(sub1["submitted_hashes"]["src/agy_pkg/scope_guard.py"]["sha256"]) == 64

    # 3. Edit file while PENDING_REVIEW -> record_review_verdict must reject stale submission!
    target.write_text("def extract_literal_prefix(pattern: str) -> str:\n    return pattern.strip()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalidated by post-submission edit"):
        qm.record_review_verdict("LIVE-AGY-01", reviewer_agent_id="claude-opus-reviewer", accepted=True, reviewer_family="claude")
    assert _job(db, "LIVE-AGY-01")["acceptance_status"] == "INVALIDATED_BY_EDIT"

    # 4. Re-submit and approve cleanly
    sub2 = qm.submit_for_review("LIVE-AGY-01", changed_files=["src/agy_pkg/scope_guard.py"], implementer_agent_id="agy-worker-1")
    assert sub2["acceptance_status"] == "PENDING_REVIEW"
    verdict1 = qm.record_review_verdict("LIVE-AGY-01", reviewer_agent_id="claude-opus-reviewer", accepted=True, findings="old approval", reviewer_family="claude", cost_learner=learner)
    assert verdict1["acceptance_status"] == "ACCEPTED"
    assert verdict1["downstream_unblocked"] == ["DOWNSTREAM-01"]

    # 5. Now edit scope_guard.py AFTER ACCEPTED (reproducing the exact audit finding!)
    target.write_text("def extract_literal_prefix(pattern: str) -> str:\n    return pattern.rstrip('/')\n", encoding="utf-8")

    # Downstream step must detect the post-approval edit, flip LIVE-AGY-01 -> INVALIDATED_BY_EDIT, clear review_findings, and block DOWNSTREAM-01
    t2 = _step(qm, learner, route_fn, exec_fn, now=1_800_000_050.0)
    assert t2["dispatched"] == []
    assert any(b["job_id"] == "DOWNSTREAM-01" for b in t2["blocked_by_dependencies"])
    row_after_edit = _job(db, "LIVE-AGY-01")
    assert row_after_edit["acceptance_status"] == "INVALIDATED_BY_EDIT"
    assert row_after_edit["reviewer_agent_id"] is None
    assert row_after_edit["review_findings"] is None

    # 5b. Renaming/removing workspace must also fail closed on submission and invalidate an ACCEPTED job
    ws_renamed = tmp_path / "ws_renamed"
    ws.rename(ws_renamed)
    with pytest.raises(ValueError, match="Workspace directory"):
        qm.submit_for_review("LIVE-AGY-01", changed_files=["src/agy_pkg/scope_guard.py"], implementer_agent_id="agy-worker-1")
    ws_renamed.rename(ws)

    # 6. Re-submit modified package and get fresh independent review approval
    sub3 = qm.submit_for_review("LIVE-AGY-01", changed_files=["src/agy_pkg/scope_guard.py"], implementer_agent_id="agy-worker-1")
    assert sub3["acceptance_status"] == "PENDING_REVIEW"
    assert sub3["submitted_hashes"]["src/agy_pkg/scope_guard.py"]["sha256"] != sub2["submitted_hashes"]["src/agy_pkg/scope_guard.py"]["sha256"]
    verdict2 = qm.record_review_verdict("LIVE-AGY-01", reviewer_agent_id="codex-reviewer", accepted=True, findings="fresh approval", reviewer_family="openai", cost_learner=learner)
    assert verdict2["acceptance_status"] == "ACCEPTED"

    # 6b. If workspace disappears after ACCEPTED, downstream dependent is blocked and acceptance is invalidated
    ws.rename(ws_renamed)
    t_missing_ws = _step(qm, learner, route_fn, exec_fn, now=1_800_000_055.0)
    assert t_missing_ws["dispatched"] == []
    assert _job(db, "LIVE-AGY-01")["acceptance_status"] == "INVALIDATED_BY_EDIT"
    assert _job(db, "LIVE-AGY-01")["review_findings"] is None
    ws_renamed.rename(ws)

    # Re-submit and re-approve with restored workspace -> DOWNSTREAM-01 unblocks and dispatches
    qm.submit_for_review("LIVE-AGY-01", changed_files=["src/agy_pkg/scope_guard.py"], implementer_agent_id="agy-worker-1")
    qm.record_review_verdict("LIVE-AGY-01", reviewer_agent_id="codex-reviewer", accepted=True, findings="restored approval", reviewer_family="openai", cost_learner=learner)

    t3 = _step(qm, learner, route_fn, exec_fn, now=1_800_000_060.0)
    assert [d["job_id"] for d in t3["dispatched"]] == ["DOWNSTREAM-01"]


def test_audit_dry_run_never_mutates_sqlite_tasks_dependencies_or_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Audit 2a: Dry-run (execute_live=False) via step_supervisor, /api/dispatch, and /api/queue/step
    never mutates real SQLite tasks, dependencies, acceptance_status (even if a file on disk drifted!),
    or task_cost_observations.
    """
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    f = ws / "src" / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")

    db, qm, learner = _qm(tmp_path / "q")
    qm.enqueue_tasks(
        [
            {"job_id": "UP-ACCEPTED", "kind": "implement", "workspace": str(ws), "edit_scope": ["src/"], "priority": 10},
            {"job_id": "PAUSED-READY", "kind": "scan", "depends_on": ["UP-ACCEPTED"], "satisfy_on": "accepted", "priority": 8},
            {"job_id": "WAIT-DEP", "kind": "implement", "workspace": str(ws), "edit_scope": ["docs/"], "depends_on": ["PAUSED-READY"], "satisfy_on": "completed", "priority": 5},
        ]
    )
    route_fn, _ = _fake_route()
    exec_fn, exec_calls = _exec_with({"exit_code": 0})

    # Complete and accept UP-ACCEPTED, and put PAUSED-READY into PAUSED_WAIT_RESET with next_eligible_at=0
    _step(qm, learner, route_fn, exec_fn, max_jobs=1)
    qm.submit_for_review("UP-ACCEPTED", changed_files=["src/mod.py"], implementer_agent_id="impl-1")
    qm.record_review_verdict("UP-ACCEPTED", reviewer_agent_id="rev-1", accepted=True, reviewer_family="claude", cost_learner=learner)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE backlog_jobs SET status = 'PAUSED_WAIT_RESET', next_eligible_at = 0 WHERE job_id = 'PAUSED-READY'")
        conn.commit()
    exec_calls.clear()

    # Modify src/mod.py on disk so drift exists! Dry-run should notice drift in-memory (blocking PAUSED-READY)
    # WITHOUT mutating UP-ACCEPTED's acceptance_status in SQLite!
    f.write_text("x = 999\n", encoding="utf-8")

    def _snapshot() -> tuple:
        with sqlite3.connect(str(db)) as conn:
            return (
                conn.execute("SELECT * FROM backlog_jobs ORDER BY job_id").fetchall(),
                conn.execute("SELECT * FROM task_cost_observations ORDER BY id").fetchall(),
            )

    before = _snapshot()
    preview_res = _step(qm, learner, route_fn, exec_fn, live=False, max_jobs=5)
    assert preview_res["preview"] is True
    assert preview_res["dispatched"] == []
    assert preview_res["would_dispatch"] == []  # blocked because UP-ACCEPTED drifted in memory
    assert any(b["job_id"] == "PAUSED-READY" for b in preview_res["blocked_by_dependencies"])
    assert exec_calls == []
    assert _snapshot() == before

    # Restore file content and verify dry-run would_dispatch=["PAUSED-READY"] still writes nothing to SQLite
    f.write_text("x = 1\n", encoding="utf-8")
    preview_res2 = _step(qm, learner, route_fn, exec_fn, live=False, max_jobs=5)
    assert [w["job_id"] for w in preview_res2["would_dispatch"]] == ["PAUSED-READY"]
    assert _snapshot() == before


def test_audit_constraints_preserved_across_enqueue_pause_crash_resume_and_dispatch(tmp_path: Path) -> None:
    """
    Audit 2b: Across enqueue, rate-limit pause, worker crash recovery (resume), routing, and execution,
    workspace, edit_scope, read_only, explicit_model, explicit_effort,
    require_cross_family, satisfy_on, implementer_family, and priority are never lost.
    """
    ws = str((tmp_path / "project_ws").resolve())
    (tmp_path / "project_ws" / "src").mkdir(parents=True)
    db, qm, learner = _qm(tmp_path / "q")

    task_spec = {
        "job_id": "CONSTRAINT-JOB",
        "task": "Implement feature X",
        "kind": "implement",
        "blast_radius": "module",
        "ambiguity": "low",
        "dependency_role": "upstream",
        "context_need": "module",
        "explicit_model": "claude-opus-5-5",
        "explicit_effort": "high",
        "workspace": ws,
        "edit_scope": ["src/feature_x.py", "tests/test_feature_x.py"],
        "read_only": False,
        "require_cross_family": True,
        "implementer_family": "claude",
        "priority": 42,
        "depends_on": [],
        "satisfy_on": "accepted",
    }
    qm.enqueue_tasks([task_spec])

    def _assert_job_fields(row: Dict[str, Any]) -> None:
        assert row["workspace"] == ws
        assert json.loads(row["edit_scope_json"]) == ["src/feature_x.py", "tests/test_feature_x.py"]
        assert int(row["read_only"]) == 0
        assert row["explicit_model"] == "claude-opus-5-5"
        assert row["explicit_effort"] == "high"
        assert int(row["require_cross_family"]) == 1
        assert row["satisfy_on"] == "accepted"
        assert row["implementer_family"] == "claude"
        assert int(row["priority"]) == 42

    _assert_job_fields(_job(db, "CONSTRAINT-JOB"))

    # Step 1: Rate-limited -> PAUSED_WAIT_RESET
    seen_route_reqs: List[Dict[str, Any]] = []
    seen_exec_records: List[Dict[str, Any]] = []

    def _route_spy(req, providers_quota, orch, operating_mode="endurance", live_activity=None, cost_learner=None):
        seen_route_reqs.append(copy.deepcopy(req))
        return route_work_package(req, providers_quota, orch, operating_mode=operating_mode, live_activity=live_activity, cost_learner=cost_learner)

    rate_limit_exec, _ = _exec_with({"exit_code": 1, "rate_limited": True, "stderr_tail": "429 rate limit"})
    _step(qm, learner, _route_spy, rate_limit_exec, now=1_800_000_000.0)
    row_paused = _job(db, "CONSTRAINT-JOB")
    assert row_paused["status"] == "PAUSED_WAIT_RESET"
    _assert_job_fields(row_paused)

    # Step 2: Simulate resume into RUNNING followed by worker crash (dead PID) -> recover_orphaned_jobs
    qm.mark_job_running("CONSTRAINT-JOB", "claude", "claude_shared", "claude-opus-5-5", worker_pid=999999, now_ts=1_800_000_700.0)
    rec = qm.recover_orphaned_jobs(alive_pids=set(), now_ts=1_800_000_710.0)
    assert rec["requeued_jobs"] == ["CONSTRAINT-JOB"]
    row_requeued = _job(db, "CONSTRAINT-JOB")
    assert row_requeued["status"] == "QUEUED"
    _assert_job_fields(row_requeued)

    # Step 3: Dispatch after retry backoff and verify route_work_package + execute_fn receive all constraints intact
    def _exec_spy(ar, dry_run=False, timeout_sec=60):
        seen_exec_records.append(copy.deepcopy(ar))
        return {"mode": "live", "exit_code": 0, "rate_limited": False, "duration_sec": 0.1}

    res = _step(qm, learner, _route_spy, _exec_spy, now=1_800_000_800.0)
    assert [d["job_id"] for d in res["dispatched"]] == ["CONSTRAINT-JOB"]
    assert len(seen_route_reqs) == 2
    for req in seen_route_reqs:
        assert req["workspace"] == ws
        assert req["edit_scope"] == ["src/feature_x.py", "tests/test_feature_x.py"]
        assert req["read_only"] is False
        assert req["explicit_model"] == "claude-opus-5-5"
        assert req["explicit_effort"] == "high"
        assert req["require_cross_family"] is True
        assert req["implementer_family"] == "claude"
        assert req["priority"] == 42
        assert req["satisfy_on"] == "accepted"

    assert len(seen_exec_records) == 1
    exec_rec = seen_exec_records[0]
    assert exec_rec["workspace"] == ws
    assert exec_rec["model"] == "claude-opus-5-5"
    assert exec_rec["reasoning_effort"] == "high"
    assert exec_rec["explicit_model"] == "claude-opus-5-5"
    assert exec_rec["explicit_effort"] == "high"
    assert exec_rec["edit_scope"] == ["src/feature_x.py", "tests/test_feature_x.py"]
    assert exec_rec["read_only"] is False
    assert exec_rec["require_cross_family"] is True
    assert exec_rec["implementer_family"] == "claude"
    assert exec_rec["priority"] == 42
    assert exec_rec["satisfy_on"] == "accepted"
    _assert_job_fields(_job(db, "CONSTRAINT-JOB"))

    # Also check read_only=True preserves empty edit_scope=[] in route_work_package
    ro_res = route_work_package(
        {"task_id": "RO-1", "task": "Read-only scan", "kind": "scan", "read_only": True, "edit_scope": []},
        _quotas(),
        {"active": "auto"},
        operating_mode="sprint",
        live_activity={"workers": []},
    )
    assert ro_res["status"] == "routed"
    assert ro_res["assignment_record"]["read_only"] is True
    assert ro_res["assignment_record"]["edit_scope"] == []


def test_audit_concurrent_scheduler_atomic_caps_no_double_claim_crash_recovery_and_bounded_retries(tmp_path: Path) -> None:
    """
    Audit 5: Under concurrent step_supervisor calls across threads:
    - Global 10 / Provider 6 / Pool 6 limits are atomically enforced (never exceeded)
    - No job is ever claimed/dispatched twice
    - Worker crash does not lose jobs (recover_orphaned_jobs requeues dead PIDs)
    - Failed jobs do not enter unbounded retry loops (capped at max_attempts=2 -> FAILED_ESCALATED)
    """
    import threading
    import time as time_mod

    db, qm, learner = _qm(tmp_path / "concurrent")
    # Enqueue 18 independent read-only scan tasks
    qm.enqueue_tasks([{"job_id": f"CONC-{i:02d}", "kind": "scan", "read_only": True, "priority": 20 - i} for i in range(18)])

    # Route alternatingly across agy_gemini, claude_shared, codex_shared so Global 10 cap and Pool/Provider 6 caps are both tested
    pools = [
        ("agy", "agy_gemini", "gemini-3.8-flash-low"),
        ("claude", "claude_shared", "claude-sonnet-5-5"),
        ("codex", "codex_shared", "gpt-6.1-sol"),
    ]

    def _concurrent_route(req, providers_quota, orch, operating_mode="endurance", live_activity=None, cost_learner=None):
        idx = int(req["job_id"].split("-")[1]) % len(pools)
        prov, pool_id, model_id = pools[idx]
        return {
            "status": "routed",
            "assignment_record": {
                "task_id": req["job_id"],
                "task": req["task"],
                "provider": prov,
                "pool_id": pool_id,
                "model": model_id,
                "selected_tier": "LOW",
                "reasoning_effort": "low",
                "edit_scope": [],
                "read_only": True,
                "predicted_cost": {"five_hour_mean": 0.01, "weekly_mean": 0.002},
            },
            "pool_states": {},
        }

    lock = threading.Lock()
    claimed_jobs: List[str] = []
    max_observed = {"global": 0, "provider": 0, "pool": 0}
    barrier_release = threading.Event()

    def _concurrent_exec(ar, dry_run=False, timeout_sec=60):
        jid = ar["task_id"]
        with lock:
            claimed_jobs.append(jid)
            with sqlite3.connect(str(db)) as conn:
                g_run = conn.execute("SELECT COUNT(*) FROM backlog_jobs WHERE status = 'RUNNING'").fetchone()[0]
                p_run = max(
                    [r[0] for r in conn.execute("SELECT COUNT(*) FROM backlog_jobs WHERE status = 'RUNNING' GROUP BY assigned_provider").fetchall()] or [0]
                )
                pool_run = max(
                    [r[0] for r in conn.execute("SELECT COUNT(*) FROM backlog_jobs WHERE status = 'RUNNING' GROUP BY assigned_pool").fetchall()] or [0]
                )
                max_observed["global"] = max(max_observed["global"], g_run)
                max_observed["provider"] = max(max_observed["provider"], p_run)
                max_observed["pool"] = max(max_observed["pool"], pool_run)
        # Hold RUNNING briefly so all concurrent scheduler threads overlap while jobs are in flight
        barrier_release.wait(timeout=0.25)
        return {"mode": "live", "exit_code": 0, "rate_limited": False, "duration_sec": 0.05}

    threads = [
        threading.Thread(
            target=lambda: _step(qm, learner, _concurrent_route, _concurrent_exec, max_jobs=4, live=True, now=1_800_000_000.0)
        )
        for _ in range(12)
    ]
    for t in threads:
        t.start()
    time_mod.sleep(0.12)
    barrier_release.set()
    for t in threads:
        t.join(timeout=5.0)

    # Verify zero duplicate claims during the wave and strict adherence to Global <= 10, Provider <= 6, Pool <= 6
    assert len(claimed_jobs) == len(set(claimed_jobs)), f"Duplicate job claim detected: {claimed_jobs}"
    assert 1 <= max_observed["global"] <= 10
    assert 1 <= max_observed["provider"] <= 6
    assert 1 <= max_observed["pool"] <= 6

    # Drain remaining queued jobs and confirm every single one of the 18 jobs was claimed exactly once
    _step(qm, learner, _concurrent_route, _concurrent_exec, max_jobs=20, live=True, now=1_800_000_010.0)
    assert sorted(claimed_jobs) == [f"CONC-{i:02d}" for i in range(18)]
    assert len(claimed_jobs) == 18

    # Also verify single-pool / single-provider cap <= 6 under concurrency
    db2, qm2, learner2 = _qm(tmp_path / "pool_cap")
    qm2.enqueue_tasks([{"job_id": f"POOL-{i:02d}", "kind": "scan", "read_only": True} for i in range(10)])
    single_route, _ = _fake_route(provider="agy", pool="agy_gemini", model="gemini-3.8-flash-low")
    pool_release = threading.Event()
    max_single_pool = {"running": 0}

    def _pool_exec(ar, dry_run=False, timeout_sec=60):
        with lock:
            with sqlite3.connect(str(db2)) as conn:
                c = conn.execute("SELECT COUNT(*) FROM backlog_jobs WHERE status = 'RUNNING' AND assigned_pool = 'agy_gemini'").fetchone()[0]
                max_single_pool["running"] = max(max_single_pool["running"], c)
        pool_release.wait(timeout=0.2)
        return {"mode": "live", "exit_code": 0, "rate_limited": False}

    t_pool = [
        threading.Thread(
            target=lambda: _step(qm2, learner2, single_route, _pool_exec, max_jobs=5, live=True, now=1_800_000_000.0)
        )
        for _ in range(8)
    ]
    for t in t_pool:
        t.start()
    time_mod.sleep(0.1)
    pool_release.set()
    for t in t_pool:
        t.join(timeout=5.0)
    assert 1 <= max_single_pool["running"] <= 6

    # Verify crash recovery + bounded retry (max_attempts=2 -> FAILED_ESCALATED, never infinite loop)
    qm2.enqueue_tasks([{"job_id": "CRASH-AND-FAIL", "kind": "scan", "read_only": True, "max_attempts": 2, "priority": 100}])
    qm2.mark_job_running("CRASH-AND-FAIL", "agy", "agy_gemini", "gemini-3.8-flash-low", worker_pid=888888, now_ts=1_800_000_100.0)
    # 1st crash recovery (attempt_count becomes 1, status -> QUEUED)
    r1 = qm2.recover_orphaned_jobs(alive_pids=set(), now_ts=1_800_000_105.0)
    assert r1["requeued_jobs"] == ["CRASH-AND-FAIL"]
    assert _job(db2, "CRASH-AND-FAIL")["attempt_count"] == 1
    assert _job(db2, "CRASH-AND-FAIL")["status"] == "QUEUED"

    # 2nd failure via execution -> reaches max_attempts=2 -> FAILED_ESCALATED and is never dispatched again
    fail_exec, fail_calls = _exec_with({"exit_code": 1, "stderr_tail": "fatal error"})
    _step(qm2, learner2, single_route, fail_exec, max_jobs=1, now=1_800_000_200.0)
    assert _job(db2, "CRASH-AND-FAIL")["status"] == "FAILED_ESCALATED"
    assert _job(db2, "CRASH-AND-FAIL")["attempt_count"] == 2
    fail_calls.clear()
    _step(qm2, learner2, single_route, fail_exec, max_jobs=10, now=1_800_000_500.0)
    assert all(c.get("job_id") != "CRASH-AND-FAIL" for c in fail_calls)


def test_mixed_native_and_hub_atomic_concurrency_caps(tmp_path: Path) -> None:
    """
    Category A (Native subagents via `claim_native_lease` + unregistered native subagents in `live_activity`)
    and Category B (Hub queued jobs via `step_supervisor`) share the exact same atomic 10 / 6 / 6 caps
    and edit-scope collision leases when running concurrently.
    """
    import threading
    import time as time_mod

    db, qm, learner = _qm(tmp_path)
    ws = str(tmp_path)

    # 1. Claim 3 Category A native subagent leases on AGY (2 on agy_gemini, 1 writer on agy_claude_gpt)
    c1 = qm.claim_native_lease(
        "NATIVE-AGY-1",
        provider="agy",
        pool_id="agy_gemini",
        model_id="gemini-3.8-flash-high",
        read_only=True,
        session_id="agy-subagent-sess-1",
    )
    c2 = qm.claim_native_lease(
        "NATIVE-AGY-2",
        provider="agy",
        pool_id="agy_gemini",
        model_id="gemini-3.8-flash-high",
        read_only=True,
        session_id="agy-subagent-sess-2",
    )
    c3 = qm.claim_native_lease(
        "NATIVE-AGY-WRITER",
        provider="agy",
        pool_id="agy_claude_gpt",
        model_id="claude-sonnet-5-5-medium",
        workspace=ws,
        edit_scope=["src/native_mod/"],
        read_only=False,
        session_id="agy-subagent-sess-3",
    )
    assert c1["claimed"] and c2["claimed"] and c3["claimed"]

    # Also include 1 unregistered external native worker on AGY in live_activity (plus a duplicate of sess-1 to verify deduplication)
    mixed_live_activity = {
        "workers": [
            {"pid": 0, "session_id": "agy-subagent-sess-1", "provider": "agy", "model": "gemini-3.8-flash-high"},
            {"pid": 0, "session_id": "agy-unregistered-4", "provider": "agy", "model": "gemini-3.8-flash-low"},
        ]
    }
    # Now AGY has 3 DB-leased native subagents + 1 unregistered external native subagent = 4 active leaves on AGY (cap = 6).
    # Enqueue 6 Hub jobs on AGY (plus 1 overlapping writer job). Only 2 Hub jobs can be claimed while the 4 native ones are active!
    qm.enqueue_tasks(
        [
            {"job_id": "HUB-OVERLAP", "kind": "implement", "workspace": ws, "edit_scope": ["src/native_mod/sub.py"], "priority": 200},
            *[{"job_id": f"HUB-AGY-{i}", "kind": "scan", "read_only": True, "priority": 100 - i} for i in range(6)],
        ]
    )

    agy_route, _ = _fake_route(provider="agy", pool="agy_gemini", model="gemini-3.8-flash-low")
    release_barrier = threading.Event()
    lock = threading.Lock()
    max_agy_combined = {"count": 0}
    dispatched_hub_ids: List[str] = []

    def _mixed_exec(ar, dry_run=False, timeout_sec=60):
        with lock:
            dispatched_hub_ids.append(ar["job_id"])
            with sqlite3.connect(str(db)) as conn:
                db_agy = conn.execute(
                    "SELECT COUNT(*) FROM backlog_jobs WHERE status = 'RUNNING' AND assigned_provider = 'agy'"
                ).fetchone()[0]
                # db_agy includes the 3 native leases + currently running Hub jobs; plus 1 unregistered in mixed_live_activity
                max_agy_combined["count"] = max(max_agy_combined["count"], db_agy + 1)
        release_barrier.wait(timeout=10)
        return {"mode": "live", "exit_code": 0, "rate_limited": False}

    # Try claiming a 7th native lease directly once 2 Hub jobs are in flight -> must be rejected by provider cap (6/6)
    threads = [
        threading.Thread(
            target=lambda: _step(
                qm,
                learner,
                agy_route,
                _mixed_exec,
                max_jobs=4,
                live=True,
                now=1_800_000_000.0,
                live_activity=mixed_live_activity,
            )
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    deadline = time_mod.time() + 10  # slow CI runners: wait until the two Hub jobs are really in flight
    while time_mod.time() < deadline:
        with lock:
            if len(dispatched_hub_ids) >= 2:
                break
        time_mod.sleep(0.02)

    # While 3 native leases + 1 external + 2 Hub jobs = 6 AGY workers are active, a native lease claim on AGY must fail closed!
    c_overflow = qm.claim_native_lease(
        "NATIVE-AGY-OVERFLOW",
        provider="agy",
        pool_id="agy_gemini",
        model_id="gemini-3.8-flash-low",
        read_only=True,
        live_activity=mixed_live_activity,
    )
    assert c_overflow["claimed"] is False
    assert "provider_cap_reached:agy:6/6" in c_overflow["reason"] or "pool_cap_reached" in c_overflow["reason"]

    release_barrier.set()
    for t in threads:
        t.join(timeout=5.0)

    assert max_agy_combined["count"] <= 6
    # HUB-OVERLAP was blocked by NATIVE-AGY-WRITER's edit_scope lease
    assert _job(db, "HUB-OVERLAP")["status"] == "QUEUED"

    # Release without matching run_id must be rejected; after valid release, releasing again (status != RUNNING) is rejected
    with pytest.raises(ValueError, match="requires matching run_id"):
        qm.release_native_lease("NATIVE-AGY-WRITER", status="COMPLETED")
    rel = qm.release_native_lease("NATIVE-AGY-WRITER", run_id=c3["run_id"], status="COMPLETED")
    assert rel["released"] is True
    assert rel["review_required"] is True
    with pytest.raises(ValueError, match="expected 'RUNNING'"):
        qm.release_native_lease("NATIVE-AGY-WRITER", run_id=c3["run_id"], status="COMPLETED")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_forgotten_review_and_unreviewed_workspace_detection(tmp_path: Path) -> None:
    """
    Automatic detection when native/Hub orchestration finishes a writer task and forgets
    Refinery submission (`submit_for_review`) or independent review (`record_review_verdict`),
    or leaves unapproved dirty files in the git workspace.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    target = repo / "src" / "feature.py"
    target.write_text("v = 1\n", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, stdin=subprocess.DEVNULL)

    git("init", "-q")
    git("-c", "user.email=t@example.invalid", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "init")

    db, qm, learner = _qm(tmp_path / "hub")
    # Native writer claims lease, edits src/feature.py, and releases lease WITHOUT calling submit_for_review
    lease = qm.claim_native_lease(
        "NATIVE-WRITE-1",
        provider="codex",
        model_id="gpt-6.1-sol",
        workspace=str(repo),
        edit_scope=["src/"],
        read_only=False,
        session_id="codex-native-writer-1",
    )
    assert lease["claimed"] is True
    target.write_text("v = 2\n", encoding="utf-8")
    rel = qm.release_native_lease("NATIVE-WRITE-1", run_id=lease["run_id"], status="COMPLETED")
    assert rel["review_required"] is True

    # 1. Queue summary & Witness report automatically flag the forgotten review!
    qsum = qm.get_queue_summary()
    assert qsum["unreviewed_writer_count"] == 1
    assert qsum["unreviewed_writer_jobs"][0]["job_id"] == "NATIVE-WRITE-1"
    assert qsum["unreviewed_writer_jobs"][0]["acceptance_status"] == "UNSUBMITTED"

    audit1 = qm.audit_unreviewed_work()
    assert audit1["review_gate_passed"] is False
    assert audit1["unreviewed_writer_count"] == 1
    assert audit1["unreviewed_workspace_files"] == {str(repo): ["src/feature.py"]}

    witness = qm.get_witness_report()
    assert witness["unreviewed_work"]["review_gate_passed"] is False

    # 2. Submit for review (PENDING_REVIEW) -> still not ACCEPTED, so audit still blocks!
    sub = qm.submit_for_review("NATIVE-WRITE-1")
    assert sub["acceptance_status"] == "PENDING_REVIEW"
    audit2 = qm.audit_unreviewed_work()
    assert audit2["review_gate_passed"] is False
    assert audit2["unreviewed_writer_jobs"][0]["acceptance_status"] == "PENDING_REVIEW"

    # 3. Independent reviewer accepts -> audit now passes!
    qm.record_review_verdict("NATIVE-WRITE-1", reviewer_agent_id="claude-reviewer", accepted=True, reviewer_family="claude")
    audit3 = qm.audit_unreviewed_work()
    assert audit3["review_gate_passed"] is True
    assert audit3["unreviewed_writer_count"] == 0
    assert audit3["unreviewed_workspace_files"] == {}

    # 4. Post-approval edit in workspace -> automatically invalidates ACCEPTED -> INVALIDATED_BY_EDIT and flips audit to False!
    target.write_text("v = 3  # silent post-approval edit\n", encoding="utf-8")
    audit4 = qm.audit_unreviewed_work()
    assert audit4["review_gate_passed"] is False
    assert audit4["unreviewed_writer_jobs"][0]["acceptance_status"] == "INVALIDATED_BY_EDIT"
    assert audit4["unreviewed_workspace_files"] == {str(repo): ["src/feature.py"]}


def test_reviewer_session_provenance_verification(tmp_path: Path) -> None:
    """
    Verify that `record_review_verdict` checks real session provenance (`VERIFIED_HUB_RUN` / provider session)
    and rejects spoofed or self-review session IDs when `require_verified_session=True`.
    """
    db, qm, learner = _completed_job(tmp_path)
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "x.py").write_text("x = 1\n", encoding="utf-8")
    sub = qm.submit_for_review("J", changed_files=["src/x.py"], implementer_agent_id="impl-session-100")
    assert sub["acceptance_status"] == "PENDING_REVIEW"
    impl_run_id = _job(db, "J")["run_id"]

    # 1. Self-review via implementer's run_id, job_id, or implementer_agent_id is always rejected
    for spoof in (impl_run_id, f"hub-run:{impl_run_id}", "impl-session-100", "J"):
        with pytest.raises(ValueError, match="Independent review violation"):
            qm.record_review_verdict("J", reviewer_agent_id="other-agent", review_session_id=spoof, accepted=True)

    # 2. With require_verified_session=True, a fabricated session ID, wildcard/traversal (., .., *), or unstarted QUEUED job is rejected
    qm.enqueue_tasks([{"job_id": "QUEUED-REV-JOB", "task": "Unstarted queued job", "kind": "review", "read_only": True}])
    for bad_sid in ("non-existent-fake-session-id-99999", ".", "..", "*", "../brain", "QUEUED-REV-JOB"):
        with pytest.raises(ValueError, match="Independent review violation"):
            qm.record_review_verdict(
                "J",
                reviewer_agent_id="rev-agent",
                review_session_id=bad_sid,
                require_verified_session=True,
                accepted=True,
            )

    # 3. Create a real distinct reviewer lease/run in Hub and verify it succeeds with VERIFIED_HUB_RUN
    rev_lease = qm.claim_native_lease(
        "REV-JOB-1",
        provider="claude",
        model_id="claude-opus-5-5",
        kind="review",
        read_only=True,
        session_id="claude-review-sess-200",
    )
    assert rev_lease["claimed"] is True
    verdict = qm.record_review_verdict(
        "J",
        reviewer_agent_id="claude-reviewer-200",
        review_session_id=rev_lease["run_id"],
        reviewer_family="claude",
        require_verified_session=True,
        accepted=True,
        cost_learner=learner,
    )
    assert verdict["accepted"] is True
    assert verdict["session_verified"] is True
    assert verdict["reviewer_provenance"] == "VERIFIED_HUB_RUN"
    assert _job(db, "J")["reviewer_provenance"] == "VERIFIED_HUB_RUN"


def test_auto_skill_sync_without_daemon(tmp_path: Path) -> None:
    """
    Verify that `ensure_agents_synced` detects edits to canonical `~/.agents` files via fast stat-manifest
    hashing and runs `sync.ps1` automatically without a background daemon.
    """
    from scheduler import ensure_agents_synced

    fake_home = tmp_path / "home"
    agents_dir = fake_home / ".agents"
    (agents_dir / "providers").mkdir(parents=True)
    (agents_dir / "skills" / "orchestrator").mkdir(parents=True)

    # Copy canonical sync.ps1 into the isolated fake_home/.agents
    # the installed copy, or the starter repo's rules/sync.ps1 when the tests run from a workbench clone
    candidates = [Path.home() / ".agents" / "sync.ps1", Path(__file__).resolve().parents[2] / "rules" / "sync.ps1"]
    real_sync = next((p for p in candidates if p.is_file()), None)
    if real_sync is None:
        pytest.skip("no sync.ps1 available")
    shutil.copy2(real_sync, agents_dir / "sync.ps1")

    for name in ("DELEGATION.md", "ROUTING.md", "LEARNING.md"):
        (agents_dir / name).write_text(f"# {name}\nCanonical v1\n", encoding="utf-8")
    for prov in ("codex", "agy", "opencode", "claude"):
        (agents_dir / "providers" / f"{prov}.md").write_text(f"# Provider {prov}\n", encoding="utf-8")
    skill_src = agents_dir / "skills" / "orchestrator" / "SKILL.md"
    skill_src.write_text("---\nname: orchestrator\n---\n# Skill v1\n", encoding="utf-8")

    # 1. Initial call detects missing targets and auto-runs sync.ps1
    r1 = ensure_agents_synced(agents_root=agents_dir, home_dir=fake_home, auto_sync=True)
    assert r1["status"] == "synced"
    assert r1["auto_synced"] is True
    assert (fake_home / ".codex" / "skills" / "orchestrator" / "SKILL.md").read_text(encoding="utf-8") == "---\nname: orchestrator\n---\n# Skill v1\n"
    assert (fake_home / ".gemini" / "config" / "skills" / "orchestrator" / "SKILL.md").is_file()
    assert (fake_home / ".claude" / "skills" / "orchestrator" / "SKILL.md").is_file()

    # 2. Second call with no changes hits the stamp cache without spawning sync.ps1
    r2 = ensure_agents_synced(agents_root=agents_dir, home_dir=fake_home, auto_sync=True)
    assert r2["status"] == "in_sync"
    assert r2["auto_synced"] is False
    assert r2["checked_via"] == "stamp_cache"

    # 3. Modify canonical SKILL.md -> next call automatically detects drift and syncs v2 to all providers
    skill_src.write_text("---\nname: orchestrator\n---\n# Skill v2 auto-synced\n", encoding="utf-8")
    r3 = ensure_agents_synced(agents_root=agents_dir, home_dir=fake_home, auto_sync=True)
    assert r3["status"] == "synced"
    assert r3["auto_synced"] is True
    for target_skill in (
        fake_home / ".codex" / "skills" / "orchestrator" / "SKILL.md",
        fake_home / ".gemini" / "config" / "skills" / "orchestrator" / "SKILL.md",
        fake_home / ".claude" / "skills" / "orchestrator" / "SKILL.md",
    ):
        assert "Skill v2 auto-synced" in target_skill.read_text(encoding="utf-8")


def test_agy_collector_discovers_ports_when_netstat_omits_dwstate_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression test: On Windows, Go's `language_server.exe` listening loopback sockets often have
    `dwState == 0` in `GetExtendedTcpTable`, causing `netstat -ano` to omit them completely.
    Verify `_discover_ports_for_pid` combines Win32 `GetExtendedTcpTable` + `--host_bridge_url`
    fallback and persists the verified endpoint to disk.
    """
    from collectors import agy_collector as agy_mod

    monkeypatch.setattr(agy_mod, "_CACHE_FILE", tmp_path / ".agy_endpoint_cache.json")
    monkeypatch.setattr(agy_mod, "_CACHED_ENDPOINT", {"pid": None, "csrf_token": None, "port": None})
    monkeypatch.setattr(agy_mod, "_LAST_GOOD_RESULT", None)
    monkeypatch.setattr(agy_mod, "_LAST_GOOD_TS", 0.0)

    # Simulate netstat returning empty list (dwState == 0 omitted by netstat.exe)
    monkeypatch.setattr(agy_mod, "_discover_ports_via_netstat", lambda _pid: [])
    monkeypatch.setattr(agy_mod, "_discover_ports_via_win32_tcp_table", lambda _pid: [52717, 52716])

    cmdline = (
        "C:\\language_server.exe --csrf_token tok-123 "
        "--host_bridge_url=http://127.0.0.1:52713"
    )
    ports = agy_mod._discover_ports_for_pid(36440, cmdline=cmdline)
    assert ports[:2] == [52717, 52716]
    assert 52714 in ports and 52717 in ports

    # Verify collect_agy_quota succeeds and writes disk cache
    monkeypatch.setattr(
        agy_mod,
        "_discover_language_server",
        lambda: [{"pid": 36440, "csrf_token": "tok-123", "cmdline": cmdline, "ports": ports}],
    )

    def _fake_rpc(port: int, token: str, method: str, timeout: float = 2.0):
        if port == 52717 and token == "tok-123":
            if method == "RetrieveUserQuotaSummary":
                return {
                    "response": {
                        "groups": [
                            {
                                "displayName": "Gemini Models",
                                "buckets": [
                                    {"bucketId": "gemini-5h", "window": "5h", "remainingFraction": 0.95, "resetTime": "2026-10-30T12:00:00Z"},
                                    {"bucketId": "gemini-weekly", "window": "weekly", "remainingFraction": 0.70, "resetTime": "2026-11-02T12:00:00Z"},
                                ],
                            },
                            {
                                "displayName": "Claude and GPT models",
                                "buckets": [
                                    {"bucketId": "3p-5h", "window": "5h", "remainingFraction": 1.0, "resetTime": "2026-10-30T12:00:00Z"},
                                    {"bucketId": "3p-weekly", "window": "weekly", "remainingFraction": 1.0, "resetTime": "2026-11-02T12:00:00Z"},
                                ],
                            },
                        ]
                    }
                }
            if method == "GetUserStatus":
                return {"userStatus": {"email": "test@example.com", "planStatus": {"planInfo": {"planName": "Pro"}}}}
        return None

    monkeypatch.setattr(agy_mod, "_query_rpc", _fake_rpc)
    q = agy_mod.collect_agy_quota()
    assert q["status"] == "healthy"
    assert q["confidence"] == "VERIFIED"
    assert q["port"] == 52717
    assert (tmp_path / ".agy_endpoint_cache.json").is_file()





def test_hub_runs_with_zero_or_broken_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "workbench.json"
    cfg.write_text(json.dumps({"quota_hub": {"providers": ["claude"]}}), encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_CONFIG", str(cfg))
    monkeypatch.setattr(main, "collect_live_activity", lambda: {"workers": [], "coordinators": []})
    monkeypatch.setattr(main, "collect_agy_quota", lambda: pytest.fail("disabled provider was queried"))
    monkeypatch.setattr(main, "collect_codex_quota", lambda **_: pytest.fail("disabled provider was queried"))

    def broken() -> Dict[str, Any]:
        raise OSError("no ~/.claude here")

    monkeypatch.setattr(main, "collect_claude_quota", broken)
    resp = TestClient(main.app).get("/api/quotas")
    assert resp.status_code == 200
    provs = resp.json()["providers"]
    assert {p: provs[p]["confidence"] for p in ("agy", "claude", "codex")} == {
        "agy": "UNAVAILABLE", "claude": "UNAVAILABLE", "codex": "UNAVAILABLE"}
    assert provs["claude"]["enabled"] is True and "collector error" in provs["claude"]["source"]
    assert provs["agy"]["enabled"] is False


def test_live_activity_auto_resolves_sticky_workspaces_and_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    from collectors import live_activity as la

    assert la._clean_display_title("\ufffdkinci beyin sistemi kurulumu") == "İkinci beyin sistemi kurulumu"
    assert la._clean_display_title("Hermes Agent GUI ArayA\ufffdzA\ufffd") == "Hermes Agent GUI Arayüzü"

    monkeypatch.setattr(
        la,
        "load_hub_config",
        lambda: {"workspace_aliases": {"audio synth": "~/audio-synth-app"}},
    )

    sticky = str(Path.home() / "default-workspace")
    r_brain = la._resolve_effective_workspace(sticky, "İkinci beyin sistemi kurulumu")
    assert r_brain.endswith(str(Path(".agents") / "brain"))

    r_uaw = la._resolve_effective_workspace(sticky, "Build the agent-workbench starter repo")
    assert r_uaw.endswith("ai-agent-hub")

    r_custom = la._resolve_effective_workspace("AGY Workspace", "Audio Synth DSP Engine Update")
    assert r_custom.endswith("audio-synth-app")

    r_qh = la._resolve_effective_workspace("AGY Workspace", "AI Quota Hub Project Brief")
    assert r_qh.endswith("ai-quota-orchestrator")

    # Genuine default-workspace chat keeps default-workspace
    r_dw = la._resolve_effective_workspace(sticky, "Default Workspace UI layout fix")
    assert r_dw == sticky


def test_policy_doctor_endpoint_and_ui_help_buttons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import policy_doctor

    state_file = tmp_path / "policy_doctor_state.json"
    monkeypatch.setattr(policy_doctor, "DEFAULT_STATE_FILE", state_file)

    client = TestClient(main.app)
    r_get = client.get("/api/policy-doctor")
    assert r_get.status_code == 200
    body = r_get.json()
    assert body["overall_status"] == "PASS"
    assert len(body["checks"]) == 5
    assert "compaction" in body
    assert len(body["active_rules_summary"]["categories"]) == 4

    r_post = client.post("/api/policy-doctor/run")
    assert r_post.status_code == 200
    assert r_post.json()["trigger"] == "manual_ui"
    assert state_file.is_file()

    r_ui = client.get("/")
    assert r_ui.status_code == 200
    html = r_ui.text
    assert 'id="policy-doctor-card"' in html
    assert "openSectionHelp('policy_doctor')" in html
    assert "openSectionHelp('mode_and_sim')" in html
    assert "openSectionHelp('card_agy')" in html
    assert "openSectionHelp('card_claude')" in html
    assert "openSectionHelp('card_codex')" in html
    assert "openSectionHelp('preference_swrr')" in html
    assert "openSectionHelp('concurrency_limits')" in html
    assert "openSectionHelp('banked_resets')" in html
    assert "openSectionHelp('codex_reserve')" in html
    assert "openSectionHelp('live_processes')" in html
    assert "openSectionHelp('project_goals')" in html
    assert "openSectionHelp('work_scheduler')" in html
    assert "openSectionHelp('assignment_record')" in html
    assert "openSectionHelp('cost_learner_pools')" in html
    assert "openSectionHelp('agent_scorecard')" in html
    assert "openSectionHelp('model_review')" in html
    assert "openSectionHelp('raw_json')" in html
    assert 'id="scada-nav-bar"' in html
    assert 'id="scada-l1-command-deck"' in html
    assert "switchScadaScreen('l1')" in html


