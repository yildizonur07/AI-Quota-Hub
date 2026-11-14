"""
Unit and FastAPI integration tests for `WP-ALLOCATION` (`allocation.py`, `scheduler.py`,
`orchestrator.py`, `main.py`).

Covers all 8 test scenarios required by `docs/WP-ALLOCATION-CONTRACT.md` §5 without network
or live provider CLI calls.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from allocation import (
    BurnRateLearner,
    apply_band_effort,
    classify_sigma_band,
    compute_allocation,
    compute_pool_capacity,
    compute_pool_sigma,
    normalise_sliders,
    reset_credits,
    select_and_commit,
    step_swrr,
    validate_preference_weights,
)
from main import app
from orchestrator import (
    MODEL_CATALOG,
    compute_capability_floor,
    get_allocation_credits,
    get_preference_generation,
    reset_allocation_credits,
    resolve_effort_variant,
    route_work_package,
    select_active_orchestrator,
    set_allocation_credits,
    set_preference_weights,
)
from scheduler import QuotaCostLearner, build_pool_states, evaluate_candidate_schedule_score


def _make_uniform_quotas(
    rem_5h: float = 0.80,
    rem_wk: float = 0.80,
    reset_5h: int = 14400,
    reset_wk: int = 432000,
) -> Dict[str, Any]:
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
                "gemini_5h": {
                    "used_percent": round((1.0 - rem_5h) * 100.0, 1),
                    "remaining_fraction": rem_5h,
                    "reset_in_sec": reset_5h,
                    "resets_at": "2026-10-27T20:00:00Z",
                },
                "gemini_weekly": {
                    "used_percent": round((1.0 - rem_wk) * 100.0, 1),
                    "remaining_fraction": rem_wk,
                    "reset_in_sec": reset_wk,
                    "resets_at": "2026-11-01T20:00:00Z",
                },
                "claude_gpt_5h": {
                    "used_percent": round((1.0 - rem_5h) * 100.0, 1),
                    "remaining_fraction": rem_5h,
                    "reset_in_sec": reset_5h,
                    "resets_at": "2026-10-27T20:00:00Z",
                },
                "claude_gpt_weekly": {
                    "used_percent": round((1.0 - rem_wk) * 100.0, 1),
                    "remaining_fraction": rem_wk,
                    "reset_in_sec": reset_wk,
                    "resets_at": "2026-11-01T20:00:00Z",
                },
            },
            "reset_in_sec": reset_5h,
        },
        "claude": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": rem_wk,
            "opus": rem_wk,
            "haiku": rem_wk,
            "windows": {
                "five_hour": {
                    "used_percent": round((1.0 - rem_5h) * 100.0, 1),
                    "remaining_fraction": rem_5h,
                    "reset_in_sec": reset_5h,
                    "resets_at": "2026-10-27T20:00:00Z",
                },
                "weekly": {
                    "used_percent": round((1.0 - rem_wk) * 100.0, 1),
                    "remaining_fraction": rem_wk,
                    "reset_in_sec": reset_wk,
                    "resets_at": "2026-11-01T20:00:00Z",
                },
            },
            "reset_in_sec": reset_5h,
        },
        "codex": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "luna": rem_wk,
            "terra": rem_wk,
            "sol": rem_wk,
            "windows": {
                "five_hour": {
                    "used_percent": round((1.0 - rem_5h) * 100.0, 1),
                    "remaining_fraction": rem_5h,
                    "reset_in_sec": reset_5h,
                    "resets_at": "2026-10-27T20:00:00Z",
                },
                "weekly": {
                    "used_percent": round((1.0 - rem_wk) * 100.0, 1),
                    "remaining_fraction": rem_wk,
                    "reset_in_sec": reset_wk,
                    "resets_at": "2026-11-01T20:00:00Z",
                },
            },
            "reset_in_sec": reset_5h,
        },
    }


def test_slider_normalisation_and_zero_slider_target_share(tmp_path: Path) -> None:
    """
    1. Slider normalisation:
       - All-zero `(0, 0, 0)` falls back to `(1/3, 1/3, 1/3)`.
       - `(10, 5, 5)` normalises to `(0.50, 0.25, 0.25)`.
       - A `0` slider yields `target_share == 0.0` while other providers have capacity.
    """
    w_zero = normalise_sliders({"agy": 0, "codex": 0, "claude": 0})
    assert math.isclose(w_zero["agy"], 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(w_zero["codex"], 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(w_zero["claude"], 1.0 / 3.0, rel_tol=1e-6)

    w_10_5_5 = normalise_sliders({"agy": 10, "codex": 5, "claude": 5})
    assert math.isclose(w_10_5_5["agy"], 0.50, rel_tol=1e-6)
    assert math.isclose(w_10_5_5["codex"], 0.25, rel_tol=1e-6)
    assert math.isclose(w_10_5_5["claude"], 0.25, rel_tol=1e-6)

    learner = QuotaCostLearner(tmp_path / "slider_norm.sqlite")
    b_learner = BurnRateLearner(tmp_path / "slider_norm.sqlite", cost_learner=learner)
    quotas = _make_uniform_quotas()
    sliders_zero_claude = {"agy": 10, "codex": 5, "claude": 0}
    pool_states = build_pool_states(
        quotas,
        live_activity=None,
        cost_learner=learner,
        operating_mode="endurance",
        burn_learner=b_learner,
        preference_weights=sliders_zero_claude,
    )
    alloc = compute_allocation(
        pool_states,
        burn_learner=b_learner,
        cost_learner=learner,
        sliders=sliders_zero_claude,
        operating_mode="endurance",
    )
    assert alloc["providers"]["claude"]["target_share"] == 0.0
    assert alloc["providers"]["agy"]["target_share"] > 0.0
    assert alloc["providers"]["codex"]["target_share"] > 0.0
    assert math.isclose(
        alloc["providers"]["agy"]["target_share"] + alloc["providers"]["codex"]["target_share"],
        1.0,
        abs_tol=1e-3,
    )


def test_burn_rate_learner_cold_start_ewma_reset_rejection_and_outlier_clipping(tmp_path: Path) -> None:
    """
    2. BurnRateLearner:
       - Cold-start (<3 samples) returns `"source": "prior"`.
       - After 4 valid samples, returns `"source": "ewma"` and reflects the observed burn rate.
       - A sample crossing a reset (`d_used < 0`) is rejected (`rejected = 1`) and does not corrupt the EWMA.
       - An outlier at `10x` the median is clipped to `3x` median.
    """
    db_path = tmp_path / "burn_learner.sqlite"
    cost_learner = QuotaCostLearner(db_path)
    b_learner = BurnRateLearner(db_path, cost_learner=cost_learner)

    # Cold start (<5 valid samples) -> source == "prior"
    est0 = b_learner.estimate_burn_rate("claude_shared")
    assert est0["source"] == "prior"
    assert est0["sample_count"] == 0

    # Record 2 valid samples (still <5 -> source remains "prior")
    for idx in range(2):
        rec = b_learner.record_sample(
            pool_id="claude_shared",
            window="five_hour",
            elapsed_sec=1800.0,  # 0.5h
            d_used=0.10,         # 0.20 / agent-hour
            mean_active_workers=1.0,
            confidence="VERIFIED",
            pre_resets_at="2026-10-27T20:00:00Z",
            post_resets_at="2026-10-27T20:00:00Z",
            ts=1793000000.0 + idx * 1800.0,
        )
        assert rec["rejected"] == 0

    est2 = b_learner.estimate_burn_rate("claude_shared")
    assert est2["source"] == "prior"
    assert est2["sample_count"] == 2

    # Record 3 more valid samples (total = 5 >= 5) -> source switches to "ewma"
    for idx in range(2, 5):
        rec = b_learner.record_sample(
            pool_id="claude_shared",
            window="five_hour",
            elapsed_sec=1800.0,
            d_used=0.10,  # rate = 0.20 / agent-hour
            mean_active_workers=1.0,
            confidence="VERIFIED",
            pre_resets_at="2026-10-27T20:00:00Z",
            post_resets_at="2026-10-27T20:00:00Z",
            ts=1793000000.0 + idx * 1800.0,
        )
        assert rec["rejected"] == 0
        b_learner.record_sample(
            pool_id="claude_shared",
            window="weekly",
            elapsed_sec=1800.0,
            d_used=0.025,  # rate = 0.05 / agent-hour
            mean_active_workers=1.0,
            confidence="VERIFIED",
            pre_resets_at="2026-11-01T20:00:00Z",
            post_resets_at="2026-11-01T20:00:00Z",
            ts=1793000000.0 + idx * 1800.0,
        )

    est4 = b_learner.estimate_burn_rate("claude_shared")
    assert est4["source"] == "ewma"
    assert est4["sample_count"] == 5
    assert math.isclose(est4["b_5h"], 0.20, rel_tol=1e-4)

    # Sample crossing a reset (d_used < 0) is rejected (rejected = 1) and does NOT corrupt EWMA
    rej_sample = b_learner.record_sample(
        pool_id="claude_shared",
        window="five_hour",
        elapsed_sec=1800.0,
        d_used=-0.65,
        mean_active_workers=1.0,
        confidence="VERIFIED",
        pre_resets_at="2026-10-27T20:00:00Z",
        post_resets_at="2026-10-28T01:00:00Z",
        ts=1793010000.0,
    )
    assert rej_sample["rejected"] == 1
    est_after_rej = b_learner.estimate_burn_rate("claude_shared")
    assert est_after_rej["sample_count"] == 5
    assert math.isclose(est_after_rej["b_5h"], est4["b_5h"], rel_tol=1e-6)

    # Outlier at 10x median (median is 0.20/hr -> 10x is 2.00/hr; 3x median clip is 0.60/hr)
    outlier_sample = b_learner.record_sample(
        pool_id="claude_shared",
        window="five_hour",
        elapsed_sec=1800.0,  # 0.5h * 1 worker = 0.5 agent-hours
        d_used=1.00,         # raw rate = 1.00 / 0.5 = 2.00 (10x the 0.20 median)
        mean_active_workers=1.0,
        confidence="VERIFIED",
        pre_resets_at="2026-10-27T20:00:00Z",
        post_resets_at="2026-10-27T20:00:00Z",
        ts=1793020000.0,
    )
    assert outlier_sample["rejected"] == 0
    est_clipped = b_learner.estimate_burn_rate("claude_shared")
    # With alpha=0.2, clipped to 3 * 0.20 = 0.60 -> new EWMA = 0.2 * 0.60 + 0.8 * 0.20 = 0.28
    # Unclipped 2.00 would have produced 0.2 * 2.00 + 0.8 * 0.20 = 0.56
    assert math.isclose(est_clipped["b_5h"], 0.28, rel_tol=1e-3)
    assert est_clipped["b_5h"] < 0.35


def test_capacity_formula_harmonic_blend_bottleneck(tmp_path: Path) -> None:
    """
    3. Capacity formula:
       A pool with 80% weekly remaining and 6d left (`518400s`) but only 4% 5h remaining
       (below the 5% safety reserve -> `u_{q,5h} == 0.0`) gets `C(q) == 0.0` because both
       windows constrain via the harmonic blend.
    """
    learner = QuotaCostLearner(tmp_path / "cap_bottleneck.sqlite")
    quotas = _make_uniform_quotas()
    quotas["claude"]["windows"]["five_hour"] = {
        "used_percent": 96.0,
        "remaining_fraction": 0.04,
        "reset_in_sec": 7200,
    }
    quotas["claude"]["windows"]["weekly"] = {
        "used_percent": 20.0,
        "remaining_fraction": 0.80,
        "reset_in_sec": 518400,
    }
    quotas["claude"]["sonnet"] = 0.04
    quotas["claude"]["opus"] = 0.04
    quotas["claude"]["haiku"] = 0.04

    pool_states = build_pool_states(quotas, live_activity=None, cost_learner=learner, operating_mode="endurance")
    cl_pool = pool_states["claude_shared"]
    assert cl_pool["five_hour"]["usable_before_reset"] == 0.0
    assert cl_pool["weekly"]["usable_before_reset"] > 0.70

    cap = compute_pool_capacity(cl_pool, {"b_5h": 0.30, "b_wk": 0.10}, operating_mode="endurance")
    assert cap["c_5h"] == 0.0
    assert cap["c_wk"] > 0.0
    assert cap["capacity_tasks_per_hour"] == 0.0


def test_sigma_bands_max_workers_and_effort_adjustments(tmp_path: Path) -> None:
    """
    4. Sigma bands (`docs/WP-ALLOCATION-CONTRACT.md` §2):
       - Endurance thresholds:
         * `normal` (`sigma >= 1.0`): `max_workers = policy_cap`, `effort_policy = "normal"`.
         * `economy` (`0.5 <= sigma < 1.0`): `max_workers = ceil(policy_cap / 2)`, `effort_policy = "economy"`.
         * `saver` (`sigma < 0.5`): `max_workers = 1` (or `0` if `policy_cap == 0`), `effort_policy = "saver"`.
       - Sprint thresholds: `normal >= 0.7`, `economy 0.35..0.7`, `saver < 0.35`.
    """
    # Endurance band classification & max_workers table checks
    b_saver = classify_sigma_band(0.40, operating_mode="endurance", policy_cap=6)
    assert b_saver["band"] == "saver"
    assert b_saver["max_workers"] == 1
    assert b_saver["effort_policy"] == "saver"

    b_econ = classify_sigma_band(0.75, operating_mode="endurance", policy_cap=6)
    assert b_econ["band"] == "economy"
    assert b_econ["max_workers"] == 3
    assert b_econ["effort_policy"] == "economy"

    b_norm = classify_sigma_band(1.25, operating_mode="endurance", policy_cap=6)
    assert b_norm["band"] == "normal"
    assert b_norm["max_workers"] == 6
    assert b_norm["effort_policy"] == "normal"

    # Sprint mode threshold checks (0.7 / 0.35)
    assert classify_sigma_band(0.75, operating_mode="sprint", policy_cap=6)["band"] == "normal"
    assert classify_sigma_band(0.50, operating_mode="sprint", policy_cap=6)["band"] == "economy"
    assert classify_sigma_band(0.25, operating_mode="sprint", policy_cap=6)["band"] == "saver"

    # Policy cap is never exceeded (including ceiling < band_workers and ceiling == 0)
    assert classify_sigma_band(1.80, policy_cap=4)["max_workers"] == 4
    assert classify_sigma_band(0.80, policy_cap=2)["max_workers"] == 1
    assert classify_sigma_band(1.50, policy_cap=0)["max_workers"] == 0

    # Effort adjustment rules across bands:
    # Saver: non-critical packages -> low; critical packages (explicit or HIGH design/review-of-high-risk) keep effort
    assert apply_band_effort("medium", "saver", floor_tier="MID", critical=False) == "low"
    assert apply_band_effort("medium", "saver", floor_tier="HIGH", kind="implement", critical=False) == "low"
    assert apply_band_effort("medium", "saver", floor_tier="HIGH", kind="design", critical=False) == "medium"
    assert apply_band_effort("high", "saver", floor_tier="HIGH", critical=True) == "high"

    # Economy: MID -> low; non-critical HIGH with low/med ambiguity & local/module blast_radius -> low;
    # upstream/cross-module/high-ambiguity HIGH -> medium
    assert apply_band_effort("medium", "economy", floor_tier="MID", critical=False) == "low"
    assert (
        apply_band_effort(
            "medium",
            "economy",
            floor_tier="HIGH",
            kind="implement",
            ambiguity="med",
            blast_radius="module",
            critical=False,
        )
        == "low"
    )
    assert (
        apply_band_effort(
            "medium",
            "economy",
            floor_tier="HIGH",
            kind="design",
            ambiguity="high",
            blast_radius="cross-module",
            critical=False,
        )
        == "medium"
    )
    assert apply_band_effort("high", "economy", floor_tier="HIGH", critical=True) == "high"
    assert apply_band_effort("medium", "normal", floor_tier="HIGH", critical=False) == "medium"

    # Verify compute_capability_floor + scheduler dynamic max_workers gate integration
    _, eff_floor_econ = compute_capability_floor(
        kind="implement", ambiguity="med", blast_radius="module", band="economy"
    )
    assert eff_floor_econ == "low"
    _, eff_floor_saver_high = compute_capability_floor(
        kind="design", ambiguity="high", blast_radius="cross-module", dependency_role="upstream", band="saver"
    )
    assert eff_floor_saver_high == "medium"

    # Verify evaluate_candidate_schedule_score enforces band max_workers (e.g. 1 in saver, 3 in economy)
    learner = QuotaCostLearner(tmp_path / "band_gate.sqlite")
    quotas_econ = _make_uniform_quotas(
        rem_5h=0.80, rem_wk=0.80, reset_5h=18000, reset_wk=604800
    )  # sigma == 0.80 -> economy (max_workers=3)
    live_3_workers = {
        "workers": [{"pid": 8100 + i, "provider": "claude", "model": "claude-sonnet-5-5"} for i in range(3)]
    }
    pstates_econ = build_pool_states(quotas_econ, live_activity=live_3_workers, cost_learner=learner)
    assert pstates_econ["claude_shared"]["allocation_band"] == "economy"
    assert pstates_econ["claude_shared"]["allocation_max_workers"] == 3

    from orchestrator import MODEL_CATALOG

    cl_model = next(m for m in MODEL_CATALOG if m["model_id"] == "claude-sonnet-5-5")
    pred = learner.predict_cost("claude_shared", "implement", "MID", "med")
    feasible, _, _, rej_reason, _ = evaluate_candidate_schedule_score(
        model_entry=cl_model,
        pool_state=pstates_econ["claude_shared"],
        cost_pred=pred,
        floor_tier="MID",
        floor_rank=2,
        target_effort="low",
        required_tokens=32000,
        kind="implement",
        dependency_role="leaf",
        implementer_family="",
        preferred_provider="",
        is_explicit_match=False,
    )
    assert feasible is False
    assert "Concurrency ceiling reached on provider claude (3/3)" in (rej_reason or "")


def test_burn_rate_adaptation_halves_capacity_and_target_share(tmp_path: Path) -> None:
    """
    5. Burn-rate adaptation:
       When pool A's learned burn rate doubles while pool B's stays fixed, pool A's
       `capacity_tasks_per_hour` and relative `target_share` ratio drop in half (holding sliders equal).
    """
    learner = QuotaCostLearner(tmp_path / "burn_adapt.sqlite")
    quotas = _make_uniform_quotas(rem_5h=0.80, rem_wk=0.80, reset_5h=14400, reset_wk=432000)
    equal_sliders = {"agy": 0, "codex": 10, "claude": 10}

    base_burns = {
        "agy_gemini": {"b_5h": 0.30, "b_wk": 0.10, "source": "ewma", "sample_count": 5},
        "agy_claude_gpt": {"b_5h": 0.30, "b_wk": 0.10, "source": "ewma", "sample_count": 5},
        "claude_shared": {"b_5h": 0.30, "b_wk": 0.10, "source": "ewma", "sample_count": 5},
        "codex_shared": {"b_5h": 0.30, "b_wk": 0.10, "source": "ewma", "sample_count": 5},
    }
    pool_states = build_pool_states(
        quotas,
        live_activity=None,
        cost_learner=learner,
        operating_mode="endurance",
        preference_weights=equal_sliders,
        burn_estimates=base_burns,
    )
    alloc_before = compute_allocation(
        pool_states,
        cost_learner=learner,
        sliders=equal_sliders,
        operating_mode="endurance",
        burn_estimates=base_burns,
    )
    cap_claude_before = alloc_before["providers"]["claude"]["capacity_tasks_per_hour"]
    cap_codex_before = alloc_before["providers"]["codex"]["capacity_tasks_per_hour"]

    # Double Claude's burn rate while keeping Codex fixed
    doubled_burns = dict(base_burns)
    doubled_burns["claude_shared"] = {"b_5h": 0.60, "b_wk": 0.20, "source": "ewma", "sample_count": 5}
    alloc_after = compute_allocation(
        pool_states,
        cost_learner=learner,
        sliders=equal_sliders,
        operating_mode="endurance",
        burn_estimates=doubled_burns,
    )
    cap_claude_after = alloc_after["providers"]["claude"]["capacity_tasks_per_hour"]
    cap_codex_after = alloc_after["providers"]["codex"]["capacity_tasks_per_hour"]

    assert math.isclose(cap_claude_after, cap_claude_before * 0.5, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(cap_codex_after, cap_codex_before, rel_tol=1e-3, abs_tol=1e-4)
    # With equal sliders (10, 10) and Claude at half of Codex's capacity, shares are 1/3 (~0.3333) and 2/3 (~0.6667)
    assert math.isclose(alloc_after["providers"]["claude"]["target_share"], 1.0 / 3.0, abs_tol=0.02)
    assert math.isclose(alloc_after["providers"]["codex"]["target_share"], 2.0 / 3.0, abs_tol=0.02)


def test_swrr_convergence_over_100_dispatches(tmp_path: Path) -> None:
    """
    6. SWRR convergence:
       Over 100 synthetic leaf dispatches with fixed pool states and sliders `(10, 5, 5)`,
       empirical provider counts are within `±5` of `(50, 25, 25)` when capacities are equal,
       and shift toward the higher-capacity provider when capacities differ.
    """
    # Part A: Equal provider capacities with sliders (10, 5, 5) -> target shares (0.50, 0.25, 0.25)
    credits = reset_credits()
    equal_shares = {"agy": 0.50, "codex": 0.25, "claude": 0.25}
    counts_equal = {"agy": 0, "codex": 0, "claude": 0}
    for _ in range(100):
        chosen, credits = step_swrr(
            credits=credits,
            target_shares=equal_shares,
            feasible_providers=["agy", "codex", "claude"],
        )
        counts_equal[chosen] += 1

    assert abs(counts_equal["agy"] - 50) <= 5
    assert abs(counts_equal["codex"] - 25) <= 5
    assert abs(counts_equal["claude"] - 25) <= 5

    # Part B: End-to-end 100 leaf dispatches through route_work_package when capacities differ
    # (AGY has 2 active pools vs 1 pool each for Codex and Claude -> AGY has higher capacity)
    learner = QuotaCostLearner(tmp_path / "swrr_100.sqlite")
    b_learner = BurnRateLearner(tmp_path / "swrr_100.sqlite", cost_learner=learner)
    quotas = _make_uniform_quotas(rem_5h=0.85, rem_wk=0.85, reset_5h=14400, reset_wk=432000)
    orch = select_active_orchestrator("auto", quotas)
    sliders = {"agy": 10, "codex": 5, "claude": 5}
    set_preference_weights(sliders, reset_swrr=True)

    route_counts = {"agy": 0, "codex": 0, "claude": 0}
    expected_shares: Dict[str, float] = {}
    for idx in range(100):
        res = route_work_package(
            {
                "task": f"WP-SWRR-{idx:03d}: Implement leaf helper",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "local",
                "dependency_role": "leaf",
                "context_need": "pointer",
            },
            quotas,
            orch,
            operating_mode="endurance",
            cost_learner=learner,
            preference_weights=sliders,
            burn_learner=b_learner,
        )
        assert res["status"] == "routed"
        prov = res["assignment_record"]["provider"]
        route_counts[prov] += 1
        if not expected_shares:
            expected_shares = {
                p: res["allocation"]["providers"][p]["target_share"]
                for p in ("agy", "codex", "claude")
            }

    # Verify empirical counts over 100 dispatches match target_share * 100 within ±5
    # and shifted toward the higher-capacity AGY provider (> 50 dispatches)
    assert route_counts["agy"] > 50
    for prov in ("agy", "codex", "claude"):
        assert abs(route_counts[prov] - (expected_shares[prov] * 100.0)) <= 5.0


def test_api_preferences_endpoint_and_quotas_allocation_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    7. API:
       `POST /api/preferences` persists valid `0..10` sliders, rejects `-1` / `11` with `400`,
       resets credits, and `/api/quotas` reflects the new `allocation` block.
    """
    monkeypatch.setattr("main.STATE_FILE", tmp_path / "quota_hub_state.json")
    client = TestClient(app)

    # Invalid slider values (-1, 11) must be rejected with HTTP 400
    r_neg = client.post("/api/preferences", json={"agy": -1, "codex": 5, "claude": 5})
    assert r_neg.status_code == 400

    r_high = client.post("/api/preferences", json={"agy": 10, "codex": 11, "claude": 5})
    assert r_high.status_code == 400

    # Valid slider values persist, reset credits to 0, and update /api/quotas allocation block
    r_ok = client.post("/api/preferences", json={"agy": 10, "codex": 5, "claude": 5})
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert "allocation" in body
    assert body["allocation"]["sliders"] == {"agy": 10, "codex": 5, "claude": 5}
    assert math.isclose(body["allocation"]["normalised_weights"]["agy"], 0.50, rel_tol=1e-4)
    assert math.isclose(body["allocation"]["normalised_weights"]["codex"], 0.25, rel_tol=1e-4)
    assert math.isclose(body["allocation"]["normalised_weights"]["claude"], 0.25, rel_tol=1e-4)
    assert body["allocation_credits"] == {"agy": 0.0, "codex": 0.0, "claude": 0.0}

    r_quotas = client.get("/api/quotas")
    assert r_quotas.status_code == 200
    q_body = r_quotas.json()
    assert q_body["allocation"]["sliders"] == {"agy": 10, "codex": 5, "claude": 5}
    assert set(q_body["allocation"]["providers"].keys()) == {"agy", "codex", "claude"}
    assert set(q_body["allocation"]["pools"].keys()) == {
        "agy_gemini",
        "agy_claude_gpt",
        "claude_shared",
        "codex_shared",
    }

    # Restore default (10, 8, 5)
    r_restore = client.post("/api/preferences", json={"agy": 10, "codex": 8, "claude": 5})
    assert r_restore.status_code == 200


def test_fallback_when_zero_slider_provider_is_only_feasible_one(tmp_path: Path) -> None:
    """
    8. Slider 0 = off (`docs/WP-ALLOCATION-CONTRACT.md` §1 & §3):
       - A provider with slider `0` (`target_share == 0.0`) is never chosen by `step_swrr`.
       - When AGY and Codex are offline/exhausted so only the zero-slider provider (`claude: 0`)
         is feasible, `route_work_package` returns no assignment (`assignment_record is None`)
         with reason `'all feasible providers disabled by preference'`.
       - Explicit model requests (`explicit_model="claude-sonnet-5-5"`) are still allowed.
    """
    # Direct step_swrr check: only zero-share provider feasible -> returns (None, credits)
    chosen_none, unchanged_credits = step_swrr(
        credits=reset_credits(),
        target_shares={"agy": 0.55, "codex": 0.45, "claude": 0.0},
        feasible_providers=["claude"],
        sliders={"agy": 10, "codex": 8, "claude": 0},
    )
    assert chosen_none is None
    assert unchanged_credits == {"agy": 0.0, "codex": 0.0, "claude": 0.0}

    learner = QuotaCostLearner(tmp_path / "zero_slider_fallback.sqlite")
    b_learner = BurnRateLearner(tmp_path / "zero_slider_fallback.sqlite", cost_learner=learner)
    quotas = {
        "agy": {"status": "offline", "confidence": "UNAVAILABLE"},
        "codex": {"status": "offline", "confidence": "UNAVAILABLE"},
        "claude": {
            "status": "healthy",
            "confidence": "VERIFIED",
            "freshness_age_sec": 2,
            "sonnet": 0.75,
            "opus": 0.75,
            "haiku": 0.75,
            "windows": {
                "five_hour": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 10800},
                "weekly": {"used_percent": 25.0, "remaining_fraction": 0.75, "reset_in_sec": 432000},
            },
            "reset_in_sec": 10800,
        },
    }
    orch = select_active_orchestrator("auto", quotas)
    sliders = {"agy": 10, "codex": 8, "claude": 0}
    res = route_work_package(
        {
            "task": "WP-ZERO-OFF: Do not route to disabled zero-slider provider",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
        },
        quotas,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
        preference_weights=sliders,
        burn_learner=b_learner,
    )
    assert res["assignment_record"] is None
    assert res["reason"] == "all feasible providers disabled by preference"
    assert "all feasible providers disabled by preference" in res["message"]

    # Explicit model request on the zero-slider provider is still allowed
    res_explicit = route_work_package(
        {
            "task": "WP-ZERO-EXPLICIT: Explicit model override allowed even when slider is 0",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "explicit_model": "claude-sonnet-5-5",
        },
        quotas,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
        preference_weights=sliders,
        burn_learner=b_learner,
    )
    assert res_explicit["status"] == "routed"
    assert res_explicit["assignment_record"] is not None
    assert res_explicit["assignment_record"]["provider"] == "claude"
    assert res_explicit["assignment_record"]["model"] == "claude-sonnet-5-5"

    # Restore default preferences in module state so subsequent tests start cleanly
    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()


def test_contract_section_5_target_shares_unknown_telemetry_and_swrr_1000() -> None:
    """
    Additional `docs/WP-ALLOCATION-CONTRACT.md` §5 coverage:
    - Default `10 / 8 / 5` sliders with equal provider capacities -> `0.435 / 0.348 / 0.217`.
    - AGY at `1/5` the burn rate of Codex and Claude (`5x` capacity) with `10 / 8 / 5` -> AGY `target_share > 0.70`.
    - Weekly-limited vs 5h-limited capacity and `UNKNOWN` telemetry returning `0` capacity & `0` target share.
    - 1,000 SWRR decisions matching `target_share` within `±0.02` and infeasible provider flow.
    """
    # 1. Equal provider capacities with 10 / 8 / 5 sliders -> 10/23, 8/23, 5/23
    equal_pools = {
        "agy_gemini": {
            "confidence": "VERIFIED",
            "five_hour": {"usable_before_reset": 0.50, "reset_in_sec": 18000, "confidence": "VERIFIED"},
            "weekly": {"usable_before_reset": 0.50, "reset_in_sec": 604800, "confidence": "VERIFIED"},
        },
        "agy_claude_gpt": {
            "confidence": "VERIFIED",
            "five_hour": {"usable_before_reset": 0.50, "reset_in_sec": 18000, "confidence": "VERIFIED"},
            "weekly": {"usable_before_reset": 0.50, "reset_in_sec": 604800, "confidence": "VERIFIED"},
        },
        "claude_shared": {
            "confidence": "VERIFIED",
            "five_hour": {"usable_before_reset": 0.50, "reset_in_sec": 18000, "confidence": "VERIFIED"},
            "weekly": {"usable_before_reset": 0.50, "reset_in_sec": 604800, "confidence": "VERIFIED"},
        },
        "codex_shared": {
            "confidence": "VERIFIED",
            "five_hour": {"usable_before_reset": 0.50, "reset_in_sec": 18000, "confidence": "VERIFIED"},
            "weekly": {"usable_before_reset": 0.50, "reset_in_sec": 604800, "confidence": "VERIFIED"},
        },
    }
    equal_prov_burns = {
        "agy_gemini": {"b_5h": 0.04, "b_wk": 0.002},
        "agy_claude_gpt": {"b_5h": 0.04, "b_wk": 0.002},
        "claude_shared": {"b_5h": 0.02, "b_wk": 0.001},
        "codex_shared": {"b_5h": 0.02, "b_wk": 0.001},
    }
    alloc_equal = compute_allocation(
        equal_pools,
        sliders={"agy": 10, "codex": 8, "claude": 5},
        burn_estimates=equal_prov_burns,
    )
    assert math.isclose(alloc_equal["providers"]["agy"]["target_share"], 10.0 / 23.0, abs_tol=1e-3)
    assert math.isclose(alloc_equal["providers"]["codex"]["target_share"], 8.0 / 23.0, abs_tol=1e-3)
    assert math.isclose(alloc_equal["providers"]["claude"]["target_share"], 5.0 / 23.0, abs_tol=1e-3)

    # 2. AGY at 1/5 the burn rate of Codex and Claude (5x capacity) -> AGY target_share > 0.70
    agy_fifth_burns = {
        "agy_gemini": {"b_5h": 0.008, "b_wk": 0.0004},
        "agy_claude_gpt": {"b_5h": 0.008, "b_wk": 0.0004},
        "claude_shared": {"b_5h": 0.02, "b_wk": 0.001},
        "codex_shared": {"b_5h": 0.02, "b_wk": 0.001},
    }
    alloc_agy_fast = compute_allocation(
        equal_pools,
        sliders={"agy": 10, "codex": 8, "claude": 5},
        burn_estimates=agy_fifth_burns,
    )
    assert alloc_agy_fast["providers"]["agy"]["target_share"] > 0.70

    # 3. Weekly-limited vs 5h-limited capacity and UNKNOWN telemetry -> 0 capacity & 0 target share
    wk_limited_pool = {
        "confidence": "VERIFIED",
        "five_hour": {"usable_before_reset": 0.80, "reset_in_sec": 3600, "confidence": "VERIFIED"},
        "weekly": {"usable_before_reset": 0.05, "reset_in_sec": 500000, "confidence": "VERIFIED"},
    }
    cap_wk = compute_pool_capacity(wk_limited_pool, {"b_5h": 0.02, "b_wk": 0.01})
    assert cap_wk["limiting_window"] == "weekly"

    unknown_pools = dict(equal_pools)
    unknown_pools["claude_shared"] = {
        "confidence": "UNKNOWN",
        "five_hour": {"usable_before_reset": 0.80, "reset_in_sec": 18000, "confidence": "UNKNOWN"},
        "weekly": {"usable_before_reset": 0.80, "reset_in_sec": 604800, "confidence": "UNKNOWN"},
    }
    alloc_unknown = compute_allocation(
        unknown_pools,
        sliders={"agy": 10, "codex": 8, "claude": 5},
        burn_estimates=equal_prov_burns,
    )
    assert alloc_unknown["providers"]["claude"]["capacity_tasks_per_hour"] == 0.0
    assert alloc_unknown["providers"]["claude"]["target_share"] == 0.0

    # 4. 1,000 SWRR decisions match target_share within ±0.02 + infeasible provider flow
    target_shares = {"agy": 0.4348, "codex": 0.3478, "claude": 0.2174}
    credits = reset_credits()
    counts = {"agy": 0, "codex": 0, "claude": 0}
    for _ in range(1000):
        chosen, credits = step_swrr(
            credits=credits,
            target_shares=target_shares,
            feasible_providers=("agy", "codex", "claude"),
        )
        counts[chosen] += 1
    for prov, target in target_shares.items():
        assert abs((counts[prov] / 1000.0) - target) <= 0.02

    # Infeasible provider flow: claude excluded -> 0 selections while agy & codex split 100% proportionally
    credits_infeas = reset_credits()
    counts_infeas = {"agy": 0, "codex": 0, "claude": 0}
    for _ in range(200):
        chosen, credits_infeas = step_swrr(
            credits=credits_infeas,
            target_shares=target_shares,
            feasible_providers=("agy", "codex"),
        )
        counts_infeas[chosen] += 1
    assert counts_infeas["claude"] == 0
    expected_agy_infeas = target_shares["agy"] / (target_shares["agy"] + target_shares["codex"])
    assert abs((counts_infeas["agy"] / 200.0) - expected_agy_infeas) <= 0.02


def test_editable_limits_api_policy_persistence_backups_and_dynamic_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Step 4b (`docs/WP-ALLOCATION-4B-LIMITS.md`):
    - `POST /api/limits` updates only concurrency keys (`max_leaves`, `max_per_provider`, `max_per_pool`)
      in `ORCHESTRATOR_POLICY_PATH` (`policy.json`), keeping every other key (`max_attempts`, `review`,
      `quota`, `witness`, custom keys) unchanged.
    - Validates `max_leaves` (`1..20`), `max_per_provider` (`0..max_leaves`), and
      `max_per_pool` (`0..max_per_provider[pool_provider]`), rejecting invalid payloads with HTTP 400.
    - Creates a timestamped backup next to `policy.json` (`policy.json.bak-<ts>`) and keeps only the
      newest 10 backups.
    - Reloads `load_orchestrator_policy()` in the running Hub process without restart.
    - Dynamic sigma-band `max_workers` remains capped by the new ceiling (`min(band_workers, ceiling)`)
      and displays `"ceiling X · now Y (band band)"`.
    """
    import json
    from scheduler import load_orchestrator_policy

    tmp_policy_path = tmp_path / "policy.json"
    monkeypatch.setenv("ORCHESTRATOR_POLICY_PATH", str(tmp_policy_path))
    monkeypatch.setattr("main.STATE_FILE", tmp_path / "quota_hub_state.json")

    initial_policy = {
        "max_leaves": 10,
        "max_per_provider": {"agy": 6, "claude": 6, "codex": 6},
        "max_per_pool": {"agy_gemini": 6, "agy_claude_gpt": 6, "claude_shared": 6, "codex_shared": 6},
        "max_attempts": 3,
        "review": {"independence": "distinct_run_and_session", "cross_family": "optional", "cross_family_bonus": 0.15},
        "quota": {"exclude_at_used": 0.95, "prefer_other_above": 0.80, "stale_after_sec": 120},
        "witness": {"starting_grace_sec": 90, "quiet_threshold_sec": 600, "stall_threshold_sec": 1800},
        "custom_owner_note": "preserve-this-field",
    }
    tmp_policy_path.write_text(json.dumps(initial_policy, indent=2), encoding="utf-8")

    client = TestClient(app)

    # 1. Validation rejections (HTTP 400)
    # max_leaves out of 1..20
    r_bad_leaves = client.post(
        "/api/limits",
        json={
            "max_leaves": 25,
            "max_per_provider": {"agy": 6, "claude": 6, "codex": 6},
            "max_per_pool": {"agy_gemini": 6, "agy_claude_gpt": 6, "claude_shared": 6, "codex_shared": 6},
        },
    )
    assert r_bad_leaves.status_code == 400

    # max_per_provider > max_leaves
    r_bad_prov = client.post(
        "/api/limits",
        json={
            "max_leaves": 4,
            "max_per_provider": {"agy": 6, "claude": 4, "codex": 4},
            "max_per_pool": {"agy_gemini": 4, "agy_claude_gpt": 4, "claude_shared": 4, "codex_shared": 4},
        },
    )
    assert r_bad_prov.status_code == 400

    # max_per_pool > max_per_provider[pool_provider]
    r_bad_pool = client.post(
        "/api/limits",
        json={
            "max_leaves": 10,
            "max_per_provider": {"agy": 3, "claude": 4, "codex": 2},
            "max_per_pool": {"agy_gemini": 4, "agy_claude_gpt": 2, "claude_shared": 4, "codex_shared": 2},
        },
    )
    assert r_bad_pool.status_code == 400

    # Extra/unrelated policy keys rejected on /api/limits
    r_bad_extra = client.post(
        "/api/limits",
        json={
            "max_leaves": 10,
            "max_per_provider": {"agy": 6, "claude": 6, "codex": 6},
            "max_per_pool": {"agy_gemini": 6, "agy_claude_gpt": 6, "claude_shared": 6, "codex_shared": 6},
            "max_attempts": 99,
        },
    )
    assert r_bad_extra.status_code == 400

    # 2. Valid update persists atomically, preserves non-concurrency keys, creates backup, and reloads in-process
    new_limits = {
        "max_leaves": 8,
        "max_per_provider": {"agy": 5, "claude": 2, "codex": 3},
        "max_per_pool": {"agy_gemini": 5, "agy_claude_gpt": 3, "claude_shared": 2, "codex_shared": 3},
    }
    r_ok = client.post("/api/limits", json=new_limits)
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert body["limits"]["max_leaves"] == 8
    assert body["limits"]["max_per_provider"] == {"agy": 5, "claude": 2, "codex": 3}
    assert body["limits"]["max_per_pool"] == {
        "agy_gemini": 5,
        "agy_claude_gpt": 3,
        "claude_shared": 2,
        "codex_shared": 3,
    }
    assert body["backup_path"] and Path(body["backup_path"]).is_file()

    # Check on-disk JSON preserved all non-concurrency keys unchanged
    saved_disk = json.loads(tmp_policy_path.read_text(encoding="utf-8"))
    assert saved_disk["max_leaves"] == 8
    assert saved_disk["max_per_provider"] == {"agy": 5, "claude": 2, "codex": 3}
    assert saved_disk["max_per_pool"] == {
        "agy_gemini": 5,
        "agy_claude_gpt": 3,
        "claude_shared": 2,
        "codex_shared": 3,
    }
    assert saved_disk["max_attempts"] == 3
    assert saved_disk["review"] == initial_policy["review"]
    assert saved_disk["quota"] == initial_policy["quota"]
    assert saved_disk["witness"] == initial_policy["witness"]
    assert saved_disk["custom_owner_note"] == "preserve-this-field"

    # Check load_orchestrator_policy() and GET /api/policy reflect the reloaded policy immediately
    reloaded = load_orchestrator_policy()
    assert reloaded["max_leaves"] == 8
    assert reloaded["max_per_provider"]["claude"] == 2
    r_pol = client.get("/api/policy")
    assert r_pol.status_code == 200
    assert r_pol.json()["max_leaves"] == 8

    # 3. Verify dynamic sigma-band max_workers is capped by the new ceiling ("ceiling X · now Y (band band)")
    # Simulate economy band (sigma = 0.80 -> max_workers = ceil(cap / 2)):
    # - For AGY (ceiling 5), ceil(5 / 2) == 3 -> "ceiling 5 · now 3 (economy band)"
    # - For Claude (ceiling 2), ceil(2 / 2) == 1 -> "ceiling 2 · now 1 (economy band)"
    # - For Codex (ceiling 3), ceil(3 / 2) == 2 -> "ceiling 3 · now 2 (economy band)"
    sim_econ = _make_uniform_quotas(rem_5h=0.80, rem_wk=0.80, reset_5h=18000, reset_wk=604800)
    r_sim = client.post("/api/simulate", json={"enabled": True, "overrides": sim_econ})
    assert r_sim.status_code == 200
    sim_body = r_sim.json()
    agy_alloc = sim_body["allocation"]["providers"]["agy"]
    claude_alloc = sim_body["allocation"]["providers"]["claude"]
    codex_alloc = sim_body["allocation"]["providers"]["codex"]
    assert agy_alloc["ceiling"] == 5
    assert agy_alloc["max_workers"] == 3
    assert agy_alloc["summary"] == "ceiling 5 · now 3 (economy band)"
    assert claude_alloc["ceiling"] == 2
    assert claude_alloc["max_workers"] == 1
    assert claude_alloc["summary"] == "ceiling 2 · now 1 (economy band)"
    assert codex_alloc["ceiling"] == 3
    assert codex_alloc["max_workers"] == 2
    assert codex_alloc["summary"] == "ceiling 3 · now 2 (economy band)"
    assert sim_body["limits"]["providers"]["agy"]["summary"] == "ceiling 5 · now 3 (economy band)"
    assert sim_body["limits"]["providers"]["claude"]["summary"] == "ceiling 2 · now 1 (economy band)"
    assert sim_body["limits"]["providers"]["codex"]["summary"] == "ceiling 3 · now 2 (economy band)"

    # Simulate normal band (sigma >= 1.0 -> max_workers = cap):
    # - For Claude (ceiling 2), normal band is capped at 2 -> "ceiling 2 · now 2 (normal band)"
    sim_norm = _make_uniform_quotas(rem_5h=0.95, rem_wk=0.95, reset_5h=14400, reset_wk=432000)
    r_sim_norm = client.post("/api/simulate", json={"enabled": True, "overrides": sim_norm})
    assert r_sim_norm.status_code == 200
    norm_body = r_sim_norm.json()
    assert norm_body["allocation"]["providers"]["claude"]["max_workers"] == 2
    assert norm_body["allocation"]["providers"]["claude"]["summary"] == "ceiling 2 · now 2 (normal band)"

    # Clear simulation overrides
    client.post("/api/simulate", json={"enabled": False, "overrides": {}})

    # 4. Run 12 saves in total and verify only the newest 10 backups are kept
    for i in range(1, 13):
        leaves = 6 + (i % 10)
        res_loop = client.post(
            "/api/limits",
            json={
                "max_leaves": leaves,
                "max_per_provider": {"agy": 4, "claude": 4, "codex": 4},
                "max_per_pool": {"agy_gemini": 4, "agy_claude_gpt": 4, "claude_shared": 4, "codex_shared": 4},
            },
        )
        assert res_loop.status_code == 200

    backups = sorted(tmp_path.glob("policy.json.bak-*"), key=lambda p: p.name)
    assert len(backups) == 10
    # Newest backup should contain the state from iteration 11 (leaves = 6 + (11 % 10) = 7)
    newest_backup_doc = json.loads(backups[-1].read_text(encoding="utf-8"))
    assert newest_backup_doc["max_leaves"] == 7
    assert newest_backup_doc["custom_owner_note"] == "preserve-this-field"


def test_burn_rate_learner_observe_snapshot_short_polls_and_subinterval_integration(
    tmp_path: Path,
) -> None:
    """
    Round 4 Fix 1 (`BurnRateLearner.observe_snapshot`):
    - Short polls (`< 120s`, e.g. every 10s via `/api/quotas` / `build_pool_states`) must NOT
      discard the baseline snapshot; they accumulate `agent_hours += running_workers * dt`
      until a valid sample interval (`>= 120s` and `agent_hours >= 0.05`) has elapsed.
    - Sub-interval integration accurately accounts for a worker-count change mid-interval
      (e.g. 1 worker for 60s then 3 workers for 60s -> `1*(60/3600) + 3*(60/3600) = 0.066667` ah).
    - When an interval was idle (`0` workers) and 6 workers start right before the next poll,
      the prior idle interval integrates `0 * dt == 0` agent-hours (never attributing the 6 newly
      started workers to the preceding idle interval).
    """
    db_path = tmp_path / "burn_short_polls.sqlite"
    cost_learner = QuotaCostLearner(db_path)
    b_learner = BurnRateLearner(db_path, cost_learner=cost_learner)

    base_ts = 1794000000.0
    # Part A: Poll every 10s from t=0 to t=120s on claude_shared.
    # Workers change mid-interval: 1 worker during [0s, 60s), then 3 workers during [60s, 120s].
    # Integrated agent-hours at t=120s: 1*(60/3600) + 3*(60/3600) = 240/3600 = 0.066667 (>= 0.05).
    total_ah_120 = (1.0 * 60.0 / 3600.0) + (3.0 * 60.0 / 3600.0)
    recorded_at_120 = None
    for step in range(13):  # step 0..12 -> t = 0, 10, ..., 120
        now_t = base_ts + (step * 10.0)
        workers_now = 1 if step < 6 else 3
        # used_5h grows from 20.0% at t=0 to 21.333333% at t=120s (d_used = 0.013333 over 0.066667 ah -> rate = 0.20)
        used_pct_5h = 20.0 + (step / 12.0) * (100.0 * (0.20 * total_ah_120))
        used_pct_wk = 10.0 + (step / 12.0) * (100.0 * (0.05 * total_ah_120))
        quotas = _make_uniform_quotas()
        quotas["claude"]["windows"]["five_hour"]["used_percent"] = used_pct_5h
        quotas["claude"]["windows"]["five_hour"]["remaining_fraction"] = 1.0 - (used_pct_5h / 100.0)
        quotas["claude"]["windows"]["weekly"]["used_percent"] = used_pct_wk
        quotas["claude"]["windows"]["weekly"]["remaining_fraction"] = 1.0 - (used_pct_wk / 100.0)
        live_activity = {
            "workers": [
                {"provider": "claude", "model": "claude-sonnet-5-5"}
                for _ in range(workers_now)
            ]
        }
        res = b_learner.observe_snapshot(
            providers_quota=quotas,
            live_activity=live_activity,
            now_ts=now_t,
        )
        claude_5h = res["claude_shared"]["five_hour"]
        if step == 0:
            assert claude_5h["recorded"] is False
            assert claude_5h["reason"] == "initial_or_untrusted_snapshot"
            assert b_learner._prev_snapshots[("claude_shared", "five_hour")]["ts"] == base_ts
        elif step < 12:
            assert claude_5h["recorded"] is False
            assert claude_5h["reason"] == "elapsed_out_of_range"
            # Baseline timestamp must remain anchored at base_ts across all short 10s polls
            assert b_learner._prev_snapshots[("claude_shared", "five_hour")]["ts"] == base_ts
            if step == 6:
                # At t=60s (end of the 1-worker sub-interval), integrated agent_hours == 60/3600
                assert math.isclose(claude_5h["agent_hours"], 60.0 / 3600.0, rel_tol=1e-4)
                assert b_learner._prev_snapshots[("claude_shared", "five_hour")]["running_jobs"] == 3.0
        else:
            recorded_at_120 = claude_5h

    assert recorded_at_120 is not None
    assert recorded_at_120["recorded"] is True
    assert recorded_at_120["rejected"] == 0
    assert math.isclose(recorded_at_120["agent_hours"], total_ah_120, rel_tol=1e-4)
    assert math.isclose(recorded_at_120["raw_burn_rate"], 0.20, rel_tol=1e-3)
    # After recording at t=120s, baseline advances to base_ts + 120.0
    assert b_learner._prev_snapshots[("claude_shared", "five_hour")]["ts"] == base_ts + 120.0

    # Part B: Idle interval followed by 6 newly started workers at the poll boundary
    idle_start_ts = base_ts + 1000.0
    quotas_idle_0 = _make_uniform_quotas()
    quotas_idle_0["codex"]["windows"]["five_hour"]["used_percent"] = 30.0
    quotas_idle_0["codex"]["windows"]["weekly"]["used_percent"] = 15.0
    b_learner.observe_snapshot(
        providers_quota=quotas_idle_0,
        live_activity={"workers": []},
        now_ts=idle_start_ts,
    )
    assert b_learner._prev_snapshots[("codex_shared", "five_hour")]["ts"] == idle_start_ts
    assert b_learner._prev_snapshots[("codex_shared", "five_hour")]["running_jobs"] == 0.0

    # 300s later, pool was idle during [idle_start_ts, idle_start_ts + 300], and 6 workers just started:
    quotas_idle_1 = _make_uniform_quotas()
    quotas_idle_1["codex"]["windows"]["five_hour"]["used_percent"] = 32.0
    quotas_idle_1["codex"]["windows"]["weekly"]["used_percent"] = 16.0
    codex_6_workers = {
        "workers": [
            {"provider": "codex", "model": "gpt-6.1-sol-medium"}
            for _ in range(6)
        ]
    }
    res_idle_boundary = b_learner.observe_snapshot(
        providers_quota=quotas_idle_1,
        live_activity=codex_6_workers,
        now_ts=idle_start_ts + 300.0,
    )
    codex_idle_5h = res_idle_boundary["codex_shared"]["five_hour"]
    # Must NOT record a sample attributing 6 workers to the prior idle interval!
    assert codex_idle_5h["recorded"] is False
    assert codex_idle_5h["rejected"] == 1
    assert codex_idle_5h["reason"] == "insufficient_agent_hours"
    assert codex_idle_5h["agent_hours"] == 0.0
    assert b_learner._prev_snapshots[("codex_shared", "five_hour")]["ts"] == idle_start_ts + 300.0
    assert b_learner._prev_snapshots[("codex_shared", "five_hour")]["running_jobs"] == 6.0

    # Now 180s later (with those 6 workers having run for 180s -> 6 * 180 / 3600 = 0.30 agent-hours):
    quotas_active = _make_uniform_quotas()
    quotas_active["codex"]["windows"]["five_hour"]["used_percent"] = 38.0  # d_used = 0.06 over 0.30 ah -> rate = 0.20
    quotas_active["codex"]["windows"]["weekly"]["used_percent"] = 19.0
    res_active_after = b_learner.observe_snapshot(
        providers_quota=quotas_active,
        live_activity=codex_6_workers,
        now_ts=idle_start_ts + 480.0,
    )
    codex_active_5h = res_active_after["codex_shared"]["five_hour"]
    assert codex_active_5h["recorded"] is True
    assert codex_active_5h["rejected"] == 0
    assert math.isclose(codex_active_5h["agent_hours"], 0.30, rel_tol=1e-4)
    assert math.isclose(codex_active_5h["raw_burn_rate"], 0.20, rel_tol=1e-3)


def test_harvest_stays_within_candidate_ranking_without_bypassing_swrr(tmp_path: Path) -> None:
    """
    Round 4 Fix 3 & Round 9 Amendment 3 (`docs/WP-ALLOCATION-CONTRACT.md` §3 + Amendment 3):
    - Classic SWRR (nginx): effective weight `w_p = pi_p + harvest_boost_p`; add `w_p` to each `C_p`,
      choose the max, subtract `W = sum(w_p)` from the winner (no clamping; credit sum is conserved).
    - When `claude` has an imminent weekly reset (`2h`) for the first 3 dispatches and then resets (`6d`),
      `claude` wins the pre-reset harvest dispatches and SWRR distributes post-reset work across `agy`
      and `codex` without starving any provider.
    - When `claude` slider is `0`, `claude` receives 0 dispatches even with an imminent reset.
    """
    learner = QuotaCostLearner(tmp_path / "harvest_swrr.sqlite")
    b_learner = BurnRateLearner(tmp_path / "harvest_swrr.sqlite", cost_learner=learner)
    quotas = _make_uniform_quotas(rem_5h=0.80, rem_wk=0.80, reset_5h=14400, reset_wk=518400)
    # Give claude an imminent weekly reset (2h = 7200s, 25% remaining) for the initial pre-reset window
    quotas["claude"]["sonnet"] = 0.25
    quotas["claude"]["opus"] = 0.25
    quotas["claude"]["haiku"] = 0.25
    quotas["claude"]["windows"]["weekly"] = {
        "used_percent": 75.0,
        "remaining_fraction": 0.25,
        "reset_in_sec": 7200,
        "resets_at": "2026-10-27T18:00:00Z",
    }
    orch = select_active_orchestrator("auto", quotas)
    sliders = {"agy": 10, "codex": 8, "claude": 5}
    set_preference_weights(sliders, reset_swrr=True)

    counts = {"agy": 0, "codex": 0, "claude": 0}
    for i in range(20):
        if i == 3:
            # Simulate claude's weekly window resetting after 3 pre-reset harvest dispatches
            quotas["claude"]["sonnet"] = 0.80
            quotas["claude"]["opus"] = 0.80
            quotas["claude"]["haiku"] = 0.80
            quotas["claude"]["windows"]["weekly"] = {
                "used_percent": 20.0,
                "remaining_fraction": 0.80,
                "reset_in_sec": 518400,
                "resets_at": "2026-11-02T18:00:00Z",
            }
        res = route_work_package(
            {
                "task": f"WP-HARVEST-SWRR-{i:02d}",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "local",
                "dependency_role": "leaf",
            },
            quotas,
            orch,
            operating_mode="endurance",
            cost_learner=learner,
            preference_weights=sliders,
            burn_learner=b_learner,
        )
        assert res["status"] == "routed"
        counts[res["assignment_record"]["provider"]] += 1
        assert math.isclose(sum(res["allocation_credits"].values()), 0.0, abs_tol=1e-6)

    # Claude harvests during the pre-reset window, then AGY/Codex receive post-reset work via SWRR
    assert counts["claude"] >= 3
    assert counts["claude"] < 20
    assert counts["agy"] > 0

    # Restore imminent reset on claude and verify slider 0 disables claude completely
    quotas["claude"]["sonnet"] = 0.25
    quotas["claude"]["opus"] = 0.25
    quotas["claude"]["haiku"] = 0.25
    quotas["claude"]["windows"]["weekly"] = {
        "used_percent": 75.0,
        "remaining_fraction": 0.25,
        "reset_in_sec": 7200,
        "resets_at": "2026-10-27T18:00:00Z",
    }
    sliders_zero_claude = {"agy": 10, "codex": 8, "claude": 0}
    set_preference_weights(sliders_zero_claude, reset_swrr=True)
    for i in range(10):
        res_z = route_work_package(
            {
                "task": f"WP-HARVEST-ZERO-{i:02d}",
                "kind": "implement",
                "ambiguity": "low",
                "blast_radius": "local",
                "dependency_role": "leaf",
            },
            quotas,
            orch,
            operating_mode="endurance",
            cost_learner=learner,
            preference_weights=sliders_zero_claude,
            burn_learner=b_learner,
        )
        assert res_z["status"] == "routed"
        assert res_z["assignment_record"]["provider"] != "claude"

    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()


def test_effort_band_variant_mapping_and_capability_floor_guard(tmp_path: Path) -> None:
    """
    Round 4 Fix 4 (`resolve_effort_variant` and `route_work_package`):
    - Maps adjusted effort to the matching catalog variant of the same series
      (`gemini-3.8-flash-high -> medium -> low`, `gemini-4-argon-medium -> low`,
      `gpt-6.1-sol-medium -> low`, `claude-opus-5-5-medium -> claude-opus-5-5-low` if present)
      and launches that variant's `cli_model`.
    - Never selects a variant whose tier is below the package capability floor; keeps the
      lowest floor-satisfying variant and records why (`effort_clamp_reason`).
    """
    flash_high = next(m for m in MODEL_CATALOG if m["model_id"] == "gemini-3.8-flash-high")
    argon_med = next(m for m in MODEL_CATALOG if m["model_id"] == "gemini-4-argon-medium")
    sol_med = next(m for m in MODEL_CATALOG if m["model_id"] == "gpt-6.1-sol-medium")
    opus_agy_med = next(m for m in MODEL_CATALOG if m["model_id"] == "claude-opus-5-5-medium")

    # 1. gemini-3.8-flash-high -> medium -> low (when floor_tier="LOW")
    v_med, eff_med, reason_med = resolve_effort_variant(flash_high, "medium", floor_tier="LOW")
    assert v_med["model_id"] == "gemini-3.8-flash-medium"
    assert v_med["cli_model"] == "gemini-3.8-flash-medium"
    assert eff_med == "medium"
    assert reason_med is None

    v_low, eff_low, reason_low = resolve_effort_variant(flash_high, "low", floor_tier="LOW")
    assert v_low["model_id"] == "gemini-3.8-flash-low"
    assert v_low["cli_model"] == "gemini-3.8-flash-low"
    assert eff_low == "low"
    assert reason_low is None

    # 2. Capability floor guard: gemini-3.8-flash-low has tier LOW < floor MID -> keep gemini-3.8-flash-medium
    v_clamped, eff_clamped, reason_clamped = resolve_effort_variant(flash_high, "low", floor_tier="MID")
    assert v_clamped["model_id"] == "gemini-3.8-flash-medium"
    assert v_clamped["cli_model"] == "gemini-3.8-flash-medium"
    assert eff_clamped == "medium"
    assert reason_clamped is not None
    assert "Floor guard" in reason_clamped
    assert "gemini-3.8-flash-low" in reason_clamped

    # 3. gemini-4-argon-medium -> gemini-4-argon-low (both are HIGH tier!)
    v_argon_low, eff_argon_low, reason_argon_low = resolve_effort_variant(argon_med, "low", floor_tier="HIGH")
    assert v_argon_low["model_id"] == "gemini-4-argon-low"
    assert v_argon_low["cli_model"] == "gemini-4-argon-low"
    assert eff_argon_low == "low"
    assert reason_argon_low is None

    # 4. gpt-6.1-sol-medium -> gpt-6.1-sol-low (for LOW floor) vs kept at gpt-6.1-sol-medium (for MID floor)
    v_sol_low, eff_sol_low, reason_sol_low = resolve_effort_variant(sol_med, "low", floor_tier="LOW")
    assert v_sol_low["model_id"] == "gpt-6.1-sol-low"
    assert eff_sol_low == "low"
    assert reason_sol_low is None

    v_sol_clamp, eff_sol_clamp, reason_sol_clamp = resolve_effort_variant(sol_med, "low", floor_tier="MID")
    assert v_sol_clamp["model_id"] == "gpt-6.1-sol-medium"
    assert eff_sol_clamp == "medium"
    assert reason_sol_clamp is not None and "Floor guard" in reason_sol_clamp

    # 5. claude-opus-5-5-medium -> claude-opus-5-5-low if present in catalog, else kept with reason
    v_opus_default, eff_opus_default, reason_opus_default = resolve_effort_variant(
        opus_agy_med, "low", floor_tier="HIGH"
    )
    assert v_opus_default["model_id"] == "claude-opus-5-5-medium"
    assert eff_opus_default == "medium"
    assert reason_opus_default is not None

    custom_catalog = list(MODEL_CATALOG) + [
        {
            **opus_agy_med,
            "model_id": "claude-opus-5-5-low",
            "cli_model": "claude-opus-5-5-low",
            "effort": "low",
            "label": "AGY · Claude Opus 5.5 (Low)",
        }
    ]
    v_opus_low, eff_opus_low, reason_opus_low = resolve_effort_variant(
        opus_agy_med, "low", floor_tier="HIGH", catalog=custom_catalog
    )
    assert v_opus_low["model_id"] == "claude-opus-5-5-low"
    assert v_opus_low["cli_model"] == "claude-opus-5-5-low"
    assert eff_opus_low == "low"
    assert reason_opus_low is None

    # 6. End-to-end route_work_package in saver band (rem_5h=0.40, reset_5h=18000 -> sigma=0.40 -> saver)
    learner = QuotaCostLearner(tmp_path / "effort_variant_route.sqlite")
    b_learner = BurnRateLearner(tmp_path / "effort_variant_route.sqlite", cost_learner=learner)
    quotas_saver = _make_uniform_quotas(rem_5h=0.40, rem_wk=0.40, reset_5h=18000, reset_wk=604800)
    orch = select_active_orchestrator("auto", quotas_saver)
    sliders_agy_only = {"agy": 10, "codex": 0, "claude": 0}

    # 6a. Non-critical HIGH-floor implement package in saver band -> launches gemini-4-argon-low
    res_high = route_work_package(
        {
            "task": "WP-SAVER-HIGH: Cross-module implementation in saver band",
            "kind": "implement",
            "ambiguity": "med",
            "blast_radius": "cross-module",
            "dependency_role": "leaf",
        },
        quotas_saver,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
        preference_weights=sliders_agy_only,
        burn_learner=b_learner,
    )
    assert res_high["status"] == "routed"
    ar_high = res_high["assignment_record"]
    assert ar_high["capability_floor"] == "HIGH"
    assert ar_high["model"] == "gemini-4-argon-low"
    assert ar_high["reasoning_effort"] == "low"
    assert "--model gemini-4-argon-low" in ar_high["headless_cli"]

    # 6b. MID-floor implement package in saver band -> keeps gemini-4-hydrogen-medium (since -low is LOW tier)
    res_mid = route_work_package(
        {
            "task": "WP-SAVER-MID: Module implementation in saver band",
            "kind": "implement",
            "ambiguity": "med",
            "blast_radius": "module",
            "dependency_role": "leaf",
        },
        quotas_saver,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
        preference_weights=sliders_agy_only,
        burn_learner=b_learner,
    )
    assert res_mid["status"] == "routed"
    ar_mid = res_mid["assignment_record"]
    assert ar_mid["capability_floor"] == "MID"
    assert ar_mid["model"] == "gemini-4-hydrogen-medium"
    assert ar_mid["selected_tier"] == "MID"
    assert ar_mid["reasoning_effort"] == "medium"
    assert "--model gemini-4-hydrogen-medium" in ar_mid["headless_cli"]
    assert ar_mid["effort_clamp_reason"] is not None
    assert "Floor guard" in ar_mid["effort_clamp_reason"]

    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()


def test_credit_lock_and_preference_generation_prevents_stale_credit_writeback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Round 4 Fix 5 (`_CREDIT_LOCK` and `preference_generation` in `orchestrator.py` and `main.py`):
    - If a route starts before `/api/preferences` (or `set_preference_weights`) and finishes
      after preferences are updated, its stale `expected_generation` / `_preference_generation`
      prevents it from overwriting the reset `0.0` credits.
    """
    monkeypatch.setattr("main.STATE_FILE", tmp_path / "quota_hub_state.json")
    client = TestClient(app)

    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    gen_before = get_preference_generation()

    # Simulate an in-flight /api/preferences update that bumps preference_generation and resets credits
    r_pref = client.post("/api/preferences", json={"agy": 6, "codex": 6, "claude": 6})
    assert r_pref.status_code == 200
    gen_after = get_preference_generation()
    assert gen_after > gen_before
    assert get_allocation_credits() == {"agy": 0.0, "codex": 0.0, "claude": 0.0}

    # 1. Attempting to commit stale credits from a request started at gen_before is ignored
    stale_attempt = set_allocation_credits(
        {"agy": -0.55, "codex": 0.30, "claude": 0.25},
        expected_generation=gen_before,
    )
    assert stale_attempt == {"agy": 0.0, "codex": 0.0, "claude": 0.0}
    assert get_allocation_credits() == {"agy": 0.0, "codex": 0.0, "claude": 0.0}

    # 2. A route_work_package call carrying _preference_generation=gen_before also cannot write back stale credits
    learner = QuotaCostLearner(tmp_path / "stale_gen.sqlite")
    b_learner = BurnRateLearner(tmp_path / "stale_gen.sqlite", cost_learner=learner)
    quotas = _make_uniform_quotas()
    orch = select_active_orchestrator("auto", quotas)
    res_stale = route_work_package(
        {
            "task": "WP-STALE-GEN: Started before preference change",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "_preference_generation": gen_before,
        },
        quotas,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
        burn_learner=b_learner,
    )
    assert res_stale["status"] == "routed"
    assert get_allocation_credits() == {"agy": 0.0, "codex": 0.0, "claude": 0.0}

    # Restore default preferences
    client.post("/api/preferences", json={"agy": 10, "codex": 8, "claude": 5})


def test_dashboard_dirty_input_protection_in_index_html() -> None:
    """
    Round 4 Fix 6 (`static/index.html`):
    - Verifies that limit and slider inputs are marked dirty on edit (`markLimitInputDirty`,
      `markSliderInputDirty`), that `renderLimits` and `renderAllocation` check `!isInputDirty(...)`
      before overwriting values during polling, and that `Cancel` / `Save` clear dirty state.
    """
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    assert "const dirtyInputs = new Set();" in content
    assert "function markLimitInputDirty(" in content
    assert "function markSliderInputDirty(" in content
    assert "function cancelLimits(" in content
    assert "function cancelPreferenceSliders(" in content
    assert "id=\"btn-cancel-limits\"" in content
    assert "id=\"btn-cancel-prefs\"" in content
    assert "!isInputDirty(maxLeavesEl)" in content
    assert "!isInputDirty(el)" in content
    assert "!isInputDirty(inp)" in content
    for input_id in (
        "limit-max-leaves",
        "limit-prov-agy",
        "limit-prov-claude",
        "limit-prov-codex",
        "limit-pool-agy_gemini",
        "limit-pool-agy_claude_gpt",
        "limit-pool-claude_shared",
        "limit-pool-codex_shared",
    ):
        assert f'id="{input_id}"' in content
        assert "markLimitInputDirty(this)" in content


def test_select_and_commit_concurrent_threads_credit_conservation_and_convergence() -> None:
    """
    Round 6 Fix A (`select_and_commit` in `allocation.py`):
    - Runs 2 threads x 500 selections (1,000 total) concurrently against the shared live credits dict.
    - Verifies that two concurrent calls never overwrite each other's credit updates:
      total credit sum is conserved (`sum(credits.values()) == 0.0` within floating-point precision)
      and empirical provider selection shares converge to `target_shares`.
    """
    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()
    shares = {
        "agy": 10.0 / 23.0,
        "codex": 8.0 / 23.0,
        "claude": 5.0 / 23.0,
    }
    feasible = ["agy", "codex", "claude"]
    barrier = threading.Barrier(2)
    thread_results: List[Dict[str, int]] = [
        {"agy": 0, "codex": 0, "claude": 0},
        {"agy": 0, "codex": 0, "claude": 0},
    ]

    def _worker(thread_idx: int) -> None:
        local_counts = thread_results[thread_idx]
        barrier.wait()
        for _ in range(500):
            chosen = select_and_commit(feasible, shares)
            assert chosen in local_counts
            local_counts[chosen] += 1

    t0 = threading.Thread(target=_worker, args=(0,))
    t1 = threading.Thread(target=_worker, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    total_counts = {
        prov: thread_results[0][prov] + thread_results[1][prov]
        for prov in feasible
    }
    assert sum(total_counts.values()) == 1000

    # Total credit conservation: every step adds +1.0 (sum of renormalised shares) and subtracts 1.0
    final_credits = get_allocation_credits()
    assert math.isclose(sum(final_credits.values()), 0.0, abs_tol=1e-6)

    # Share convergence across 1,000 atomic concurrent selections
    for prov in feasible:
        empirical_share = total_counts[prov] / 1000.0
        assert abs(empirical_share - shares[prov]) <= 0.01

    reset_allocation_credits()


def test_harvest_overlay_does_not_alter_pool_capacity_math() -> None:
    """
    Round 6 Fix B (`compute_pool_capacity` / `compute_allocation` in `allocation.py`):
    - Verifies that `compute_pool_capacity` and `compute_allocation` use the exact contract
      formula `min over windows of usable_w / (b_w * max(T_w_hours, floor_w))` regardless of
      whether `banked_harvest_overlay_active` is True or False.
    - With usable `0.8`, burns `0.3` (5h) and `0.1` (weekly), and horizons `5h` (`18000s`) and
      `144h` (`518400s`), `c_5h = 0.8 / (0.3 * 5) = 0.533333...`, `c_wk = 0.8 / (0.1 * 144) = 0.055556`,
      and `A_pool = min(c_5h, c_wk) = 1/18 ≈ 0.055556` (never inflated to `0.533`).
    """
    pool_base = {
        "confidence": "VERIFIED",
        "five_hour": {
            "remaining_fraction": 0.80,
            "usable_before_reset": 0.80,
            "reset_in_sec": 5 * 3600,
            "confidence": "VERIFIED",
        },
        "weekly": {
            "remaining_fraction": 0.80,
            "usable_before_reset": 0.80,
            "reset_in_sec": 144 * 3600,
            "confidence": "VERIFIED",
        },
        "banked_harvest_overlay_active": False,
    }
    pool_harvest = dict(pool_base, banked_harvest_overlay_active=True)
    burn = {"b_5h": 0.30, "b_wk": 0.10}

    expected_5h = 0.8 / (0.3 * 5.0)       # 0.5333333333333333
    expected_wk = 0.8 / (0.1 * 144.0)     # 0.05555555555555555 (1/18)

    cap_normal = compute_pool_capacity(pool_base, burn)
    cap_harvest = compute_pool_capacity(pool_harvest, burn)

    for cap in (cap_normal, cap_harvest):
        assert math.isclose(cap["A_five_hour"], expected_5h, rel_tol=1e-6)
        assert math.isclose(cap["A_weekly"], expected_wk, rel_tol=1e-6)
        assert math.isclose(cap["A_pool"], expected_wk, rel_tol=1e-6)
        assert round(cap["A_pool"], 4) == 0.0556
        assert cap["limiting_window"] == "weekly"

    # Also verify through compute_allocation
    pools = {
        "agy_gemini": dict(pool_base),
        "agy_claude_gpt": dict(pool_base),
        "claude_shared": dict(pool_base),
        "codex_shared": dict(pool_harvest),
    }
    burns = {
        "agy_gemini": dict(burn),
        "agy_claude_gpt": dict(burn),
        "claude_shared": dict(burn),
        "codex_shared": dict(burn),
    }
    alloc = compute_allocation(pools, sliders={"agy": 10, "codex": 8, "claude": 5}, burn_estimates=burns)
    assert math.isclose(alloc["pools"]["codex_shared"]["capacity_tasks_per_hour"], expected_wk, abs_tol=1e-6)
    assert math.isclose(alloc["providers"]["codex"]["capacity_tasks_per_hour"], expected_wk, abs_tol=1e-6)


def test_mode_change_does_not_bump_generation_and_next_route_records_swrr_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Round 6 Fix C (Generation semantics across `main.py`, `orchestrator.py`, and `allocation.py`):
    - Bumps `preference_generation` only on slider/preference changes (`POST /api/preferences`),
      never on operating-mode changes (`POST /api/mode`).
    - Changing mode (`endurance` -> `sprint`) then routing a package records the SWRR credit update.
    """
    monkeypatch.setattr("main.STATE_FILE", tmp_path / "quota_hub_state.json")
    client = TestClient(app)

    r_pref = client.post("/api/preferences", json={"agy": 10, "codex": 8, "claude": 5})
    assert r_pref.status_code == 200
    gen_initial = get_preference_generation()
    assert get_allocation_credits() == {"agy": 0.0, "codex": 0.0, "claude": 0.0}

    # Change operating mode via POST /api/mode: generation must NOT bump
    r_mode1 = client.post("/api/mode", json={"mode": "endurance"})
    assert r_mode1.status_code == 200
    assert get_preference_generation() == gen_initial

    r_mode2 = client.post("/api/mode", json={"mode": "sprint"})
    assert r_mode2.status_code == 200
    assert get_preference_generation() == gen_initial

    # Route a package after the mode change: the SWRR credit update must be recorded (non-zero)
    learner = QuotaCostLearner(tmp_path / "mode_route.sqlite")
    b_learner = BurnRateLearner(tmp_path / "mode_route.sqlite", cost_learner=learner)
    quotas = _make_uniform_quotas()
    orch = select_active_orchestrator("auto", quotas, operating_mode="sprint")
    res = route_work_package(
        {
            "task": "WP-MODE-ROUTE: Route immediately after mode change",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
        },
        quotas,
        orch,
        operating_mode="sprint",
        cost_learner=learner,
        burn_learner=b_learner,
    )
    assert res["status"] == "routed"
    chosen_prov = res["assignment_record"]["provider"]
    credits_after = get_allocation_credits()
    assert get_preference_generation() == gen_initial
    assert credits_after != {"agy": 0.0, "codex": 0.0, "claude": 0.0}
    assert credits_after[chosen_prov] < 0.0
    for other_prov in ("agy", "codex", "claude"):
        if other_prov != chosen_prov:
            assert credits_after[other_prov] > 0.0
    assert math.isclose(sum(credits_after.values()), 0.0, abs_tol=1e-6)

    # Restore default mode and credits
    client.post("/api/mode", json={"mode": "endurance"})
    reset_allocation_credits()


def test_stranded_quota_harvest_boost_wins_while_high_converges_after_zero_and_respects_slider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Amendment 4 final harvest rule (`docs/WP-ALLOCATION-CONTRACT.md`, 2026-10-27), with Amendment 3 items 1, 3, 4:
    1. H = feasible, `slider > 0` providers whose harvest condition (a)/(b)/(c) holds. If H is non-empty, classic
       SWRR runs over H only with weights `pi_p` renormalised over H (only H's credits change); a single harvester
       wins every decision, several share decisions in proportion to their weights. If H is empty, classic SWRR
       over all feasible `slider > 0` providers. No boost term, no clamping.
    3. Inside the single locked `select_and_commit` step, recompute the provider set and shares from the LIVE
       preference weights (and the capacities passed in), so a route that started before a slider change can never
       pick a provider whose slider is now `0` and never commits against stale shares; `_save_state()` captures
       live credits and preferences directly.
    4. 5-hour fittability for condition (b) uses the candidate's predicted 5-hour cost (same source as weekly
       fittability), not fixed fallback constants.
    """
    shares = {"agy": 0.45, "codex": 0.35, "claude": 0.20}
    feasible = ["agy", "codex", "claude"]
    sliders = {"agy": 10, "codex": 8, "claude": 5}
    set_preference_weights(sliders, reset_swrr=True)

    # --- (1) Condition (a): weekly reset <= 12h (2h = 7200s) and tasks_fittable_before_weekly_reset >= 1 ---
    pool_states_cond_a = {
        "agy_gemini": {"weekly": {"reset_in_sec": 518400, "usable": 0.80}},
        "agy_claude_gpt": {"weekly": {"reset_in_sec": 518400, "usable": 0.80}},
        "codex_shared": {"weekly": {"reset_in_sec": 518400, "usable": 0.80}},
        "claude_shared": {
            "weekly": {"reset_in_sec": 7200, "usable": 0.06},
            "tasks_fittable_before_weekly_reset": 4.3,
        },
    }

    # Amendment 4: claude is the only harvester (H = {claude}); it wins EVERY decision while its condition holds,
    # and only H's credits change (w = 1.0, W = 1.0 -> net 0), so other providers' credits are untouched.
    for mode in ("endurance", "sprint"):
        for _ in range(50):
            w1 = select_and_commit(
                feasible=feasible,
                shares=shares,
                pool_states=pool_states_cond_a,
                operating_mode=mode,
            )
            assert w1 == "claude"
    c1 = get_allocation_credits()
    assert math.isclose(c1["claude"], 0.0, abs_tol=1e-6)
    assert math.isclose(c1["agy"], 0.0, abs_tol=1e-6)
    assert math.isclose(c1["codex"], 0.0, abs_tol=1e-6)

    # Even with a tiny target share the single harvester still wins every decision.
    reset_allocation_credits()
    shares_tiny_claude = {"claude": 0.02, "agy": 0.60, "codex": 0.38}
    for _ in range(40):
        assert (
            select_and_commit(
                feasible=feasible,
                shares=shares_tiny_claude,
                pool_states=pool_states_cond_a,
                operating_mode="endurance",
            )
            == "claude"
        )

    # --- (2) Two harvesters share without starvation over 1,000 decisions (weights renormalised over H) ---
    # agy 0.60 and codex 0.40 both harvesting: claude (non-harvester) gets nothing, agy/codex split 600/400.
    reset_allocation_credits()
    two_boosted_pools = {
        "agy_gemini": {
            "weekly": {"reset_in_sec": 3600, "usable": 0.20},
            "tasks_fittable_before_weekly_reset": 5.0,
        },
        "codex_shared": {
            "weekly": {"reset_in_sec": 3600, "usable": 0.20},
            "tasks_fittable_before_weekly_reset": 5.0,
        },
    }
    two_counts = {"agy": 0, "codex": 0, "claude": 0}
    longest_gap = {"agy": 0, "codex": 0}
    gap = {"agy": 0, "codex": 0}
    for _ in range(1000):
        ch = select_and_commit(
            feasible=feasible,
            shares={"agy": 0.60, "codex": 0.40, "claude": 0.20},
            pool_states=two_boosted_pools,
            operating_mode="sprint",
        )
        assert ch in {"agy", "codex"}
        two_counts[ch] += 1
        for q in gap:
            gap[q] = 0 if q == ch else gap[q] + 1
            longest_gap[q] = max(longest_gap[q], gap[q])
        assert math.isclose(sum(get_allocation_credits().values()), 0.0, abs_tol=1e-6)
    assert two_counts["claude"] == 0
    assert two_counts["agy"] == 600
    assert two_counts["codex"] == 400
    assert longest_gap["agy"] <= 3 and longest_gap["codex"] <= 3

    # --- (3) Condition (b) 5-hour fittability uses candidate's predicted 5-hour cost (Amendment 3 item 4) ---
    # With usable_5h = 0.06 and reset_in_sec = 3600 (1.0h <= 1.5h), phi_wk = 0.20 < 0.75:
    # - A candidate costing predicted_5h_cost = 0.09 has fittable_5h = 0.06 / 0.09 = 0.667 < 1.0 -> NO boost (0.0)
    # - A candidate costing predicted_5h_cost = 0.05 has fittable_5h = 0.06 / 0.05 = 1.200 >= 1.0 -> boost (1.0)
    reset_allocation_credits()
    pool_states_cond_b_expensive = {
        "agy_gemini": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "agy_claude_gpt": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "codex_shared": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "claude_shared": {
            "five_hour": {"reset_in_sec": 3600, "usable": 0.06, "predicted_5h_cost": 0.09},
            "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.20},
        },
    }
    wb_expensive = select_and_commit(
        feasible=feasible,
        shares=shares,
        pool_states=pool_states_cond_b_expensive,
        operating_mode="endurance",
    )
    # Because 0.06 / 0.09 = 0.667 < 1.0, claude is not a harvester; H is empty and plain SWRR picks agy (0.45)
    assert wb_expensive == "agy"

    reset_allocation_credits()
    pool_states_cond_b_cheap = {
        "agy_gemini": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "agy_claude_gpt": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "codex_shared": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "claude_shared": {
            "five_hour": {"reset_in_sec": 3600, "usable": 0.06, "predicted_5h_cost": 0.05},
            "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.20},
        },
    }
    wb_cheap = select_and_commit(
        feasible=feasible,
        shares=shares,
        pool_states=pool_states_cond_b_cheap,
        operating_mode="endurance",
    )
    # Because 0.06 / 0.05 = 1.20 >= 1.0, claude qualifies as a harvester (H = {claude}) and wins
    assert wb_cheap == "claude"

    # When weekly window is scarce (phi >= 0.75) and weekly reset > 12h, condition (b) does NOT fire
    reset_allocation_credits()
    pool_states_cond_b_scarce = {
        "agy_gemini": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "agy_claude_gpt": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "codex_shared": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80, "opportunity_cost_phi": 0.1}},
        "claude_shared": {
            "five_hour": {"reset_in_sec": 3600, "usable": 0.20, "predicted_5h_cost": 0.05},
            "weekly": {"reset_in_sec": 518400, "usable": 0.10, "opportunity_cost_phi": 0.85},
        },
    }
    wb_scarce = select_and_commit(
        feasible=feasible,
        shares=shares,
        pool_states=pool_states_cond_b_scarce,
        operating_mode="endurance",
    )
    assert wb_scarce == "agy"

    # --- (4) Post-reset convergence to target_share ---
    for _ in range(15):
        select_and_commit(
            feasible=feasible,
            shares=shares,
            pool_states=pool_states_cond_a,
            operating_mode="sprint",
        )
        assert math.isclose(sum(get_allocation_credits().values()), 0.0, abs_tol=1e-6)

    pool_states_post_reset = {
        "agy_gemini": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80}},
        "agy_claude_gpt": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80}},
        "codex_shared": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80}},
        "claude_shared": {"five_hour": {"reset_in_sec": 14400, "usable": 0.80}, "weekly": {"reset_in_sec": 518400, "usable": 0.80}},
    }
    post_harvest_counts = {"agy": 0, "codex": 0, "claude": 0}
    n_post = 1000
    for _ in range(n_post):
        chosen = select_and_commit(
            feasible=feasible,
            shares=shares,
            pool_states=pool_states_post_reset,
            operating_mode="endurance",
        )
        assert chosen is not None
        post_harvest_counts[chosen] += 1

    for prov in feasible:
        assert abs((post_harvest_counts[prov] / float(n_post)) - shares[prov]) <= 0.015

    # --- (5) Live preference weights & capacities inside locked select_and_commit (Amendment 3 item 3) ---
    # Simulate an in-flight route that computed old shares {"agy": 0.10, "codex": 0.80, "claude": 0.10}
    # when codex slider was 8, then codex slider was set to 0 before select_and_commit ran:
    set_preference_weights({"agy": 10, "codex": 0, "claude": 5}, reset_swrr=True)
    stale_shares_with_codex_high = {"agy": 0.10, "codex": 0.80, "claude": 0.10}
    equal_caps = {"agy": 1.0, "codex": 1.0, "claude": 1.0}
    in_flight_counts = {"agy": 0, "codex": 0, "claude": 0}
    for _ in range(30):
        chosen_live = select_and_commit(
            feasible=["agy", "codex", "claude"],
            shares=stale_shares_with_codex_high,
            capacities=equal_caps,
            pool_states=pool_states_post_reset,
            operating_mode="endurance",
        )
        assert chosen_live in {"agy", "claude"}
        assert chosen_live != "codex"
        in_flight_counts[chosen_live] += 1
        assert get_allocation_credits()["codex"] == 0.0
    # Live shares from {"agy": 10, "codex": 0, "claude": 5} with equal capacities are 2/3 (20) and 1/3 (10)
    assert in_flight_counts["agy"] == 20
    assert in_flight_counts["claude"] == 10

    # A harvesting provider whose slider became 0 mid-route is respected too: H is recomputed under the lock.
    set_preference_weights({"agy": 10, "codex": 0, "claude": 5}, reset_swrr=True)
    codex_harvest_pools = dict(pool_states_post_reset)
    codex_harvest_pools["codex_shared"] = {
        "weekly": {"reset_in_sec": 3600, "usable": 0.20},
        "tasks_fittable_before_weekly_reset": 5.0,
    }
    for _ in range(30):
        ch0 = select_and_commit(
            feasible=["agy", "codex", "claude"],
            shares=stale_shares_with_codex_high,
            capacities=equal_caps,
            pool_states=codex_harvest_pools,
            operating_mode="sprint",
        )
        assert ch0 in {"agy", "claude"}
    assert get_allocation_credits()["codex"] == 0.0

    # Verify _save_state() captures live credits and preferences directly even if _RUNTIME_STATE was stale
    state_file = tmp_path / "round9_live_state.json"
    monkeypatch.setattr("main.STATE_FILE", state_file)
    set_preference_weights({"agy": 9, "codex": 6, "claude": 3}, reset_swrr=True)
    select_and_commit(
        feasible=["agy", "codex", "claude"],
        capacities={"agy": 1.0, "codex": 1.0, "claude": 1.0},
    )
    expected_live_credits = get_allocation_credits()
    import main as hub_main
    hub_main._RUNTIME_STATE["preference_weights"] = {"agy": 1, "codex": 1, "claude": 1}
    hub_main._RUNTIME_STATE["allocation_credits"] = {"agy": 0.0, "codex": 0.0, "claude": 0.0}
    hub_main._save_state()
    saved_payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved_payload["preference_weights"] == {"agy": 9, "codex": 6, "claude": 3}
    assert saved_payload["allocation_credits"] == expected_live_credits

    # --- (6) Slider 0 provider never wins even when condition (a) or (b) holds ---
    sliders_zero_claude = {"agy": 10, "codex": 8, "claude": 0}
    set_preference_weights(sliders_zero_claude, reset_swrr=True)
    shares_zero_claude = {"agy": 0.60, "codex": 0.40, "claude": 0.0}
    for mode in ("endurance", "sprint"):
        for _ in range(10):
            chosen_z = select_and_commit(
                feasible=feasible,
                shares=shares_zero_claude,
                sliders=sliders_zero_claude,
                pool_states=pool_states_cond_a,
                operating_mode=mode,
            )
            assert chosen_z in {"agy", "codex"}
            assert chosen_z != "claude"
            assert get_allocation_credits()["claude"] == 0.0

    # Also verify route_work_package with slider 0 on expiring provider and on paused_quota_exhausted path
    learner = QuotaCostLearner(tmp_path / "round9_harvest.sqlite")
    b_learner = BurnRateLearner(tmp_path / "round9_harvest.sqlite", cost_learner=learner)
    quotas = _make_uniform_quotas(rem_5h=0.80, rem_wk=0.80, reset_5h=14400, reset_wk=518400)
    quotas["claude"]["sonnet"] = 0.20
    quotas["claude"]["opus"] = 0.20
    quotas["claude"]["haiku"] = 0.20
    quotas["claude"]["windows"]["five_hour"]["reset_in_sec"] = 3600
    quotas["claude"]["windows"]["weekly"] = {
        "used_percent": 80.0,
        "remaining_fraction": 0.20,
        "reset_in_sec": 3600,
        "resets_at": "2026-10-27T18:00:00Z",
    }
    orch = select_active_orchestrator("auto", quotas)
    res_zero = route_work_package(
        {
            "task": "WP-R9-SLIDER-ZERO: Expiring provider disabled by slider 0",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
        },
        quotas,
        orch,
        operating_mode="sprint",
        cost_learner=learner,
        preference_weights=sliders_zero_claude,
        burn_learner=b_learner,
    )
    assert res_zero["status"] == "routed"
    assert res_zero["assignment_record"]["provider"] != "claude"

    # Verify early-return rejection path (`paused_quota_exhausted`) returns `allocation_credits` without NameError
    res_exhausted = route_work_package(
        {
            "task": "WP-R9-EXHAUSTED: All providers excluded",
            "kind": "implement",
            "ambiguity": "low",
            "blast_radius": "local",
            "dependency_role": "leaf",
            "excluded_providers": ["agy", "codex", "claude"],
        },
        quotas,
        orch,
        operating_mode="endurance",
        cost_learner=learner,
        burn_learner=b_learner,
    )
    assert res_exhausted["status"] == "paused_quota_exhausted"
    assert isinstance(res_exhausted["allocation_credits"], dict)
    assert set(res_exhausted["allocation_credits"].keys()) == {"agy", "codex", "claude"}

    # Restore default sliders and credits
    set_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)
    reset_allocation_credits()





def test_stale_slider_argument_never_overrides_live_weights() -> None:
    """Final review finding: a caller's stale `sliders` must not steer selection or credits (Amendment 3.3)."""
    import allocation as A

    A.set_shared_preference_weights({"agy": 1, "claude": 10, "codex": 1}, reset_swrr=True)
    try:
        caps = {"agy": 1.0, "claude": 1.0, "codex": 1.0}
        chosen = A.select_and_commit(["agy", "claude", "codex"], {}, capacities=caps,
                                     sliders={"agy": 10, "claude": 1, "codex": 1})
        assert chosen == "claude"
    finally:
        A.set_shared_preference_weights({"agy": 10, "codex": 8, "claude": 5}, reset_swrr=True)


# --- Burn learner hotfix: busy-only counting, per-window d_used, priors, bounds, one-time reset ---


def _live(entries: List[Dict[str, Any]], workers: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    return {"coordinators": entries, "workers": workers or []}


def test_running_count_ignores_idle_dead_and_dedups_sessions() -> None:
    live = _live(
        [
            {"provider": "agy", "session_id": "a", "status": "idle", "alive": True, "pid": 45152},
            {"provider": "agy", "session_id": "b", "status": "idle", "alive": True, "pid": 45152},
            {"provider": "agy", "session_id": "c", "status": "busy", "alive": True, "pid": 45152, "model": "gemini-4-argon"},
            {"provider": "agy", "session_id": "c", "status": "busy", "alive": True, "pid": 45152, "model": "gemini-4-argon"},
            {"provider": "agy", "session_id": "d", "status": "busy", "alive": False, "pid": 0},
            {"provider": "claude", "session_id": "e", "status": "busy", "alive": True},
        ],
        workers=[
            {"provider": "agy", "session_id": "w1", "status": "running", "alive": True, "model": "gemini-flash"},
            {"provider": "agy", "session_id": "w2", "status": "idle", "alive": True, "model": "gemini-flash"},
            {"provider": "agy", "session_id": "w3", "status": "running", "alive": True, "model": "claude-opus"},
        ],
    )
    count = BurnRateLearner._count_pool_running_jobs
    assert count("agy_gemini", live, None) == 2.0  # coordinator c (deduped) + worker w1
    assert count("agy_claude_gpt", live, None) == 1.0
    assert count("claude_shared", live, None) == 1.0
    assert count("codex_shared", live, None) == 0.0
    # pool_states is only a fallback when no live activity is supplied
    assert count("agy_gemini", None, {"agy_gemini": {"running_jobs": 3}}) == 3.0


def test_idle_coordinators_accumulate_no_agent_hours(tmp_path: Path) -> None:
    learner = BurnRateLearner(tmp_path / "idle.sqlite")
    live = _live([{"provider": "agy", "session_id": f"s{i}", "status": "idle", "alive": True} for i in range(3)])
    quotas = _make_uniform_quotas()
    learner.observe_snapshot(providers_quota=quotas, live_activity=live, now_ts=1_000_000.0)
    res = learner.observe_snapshot(providers_quota=quotas, live_activity=live, now_ts=1_000_000.0 + 1800.0)
    assert res["agy_gemini"]["five_hour"].get("recorded") is False
    assert learner.get_pool_window_estimate("agy_gemini", "five_hour")["samples"] == 0


def test_windows_use_independent_used_values(tmp_path: Path) -> None:
    learner = BurnRateLearner(tmp_path / "win.sqlite")
    pool_states = {
        "claude_shared": {
            "confidence": "VERIFIED",
            "five_hour": {"used_percent": 10.0, "resets_at": "r5"},
            "weekly": {"used_percent": 20.0, "resets_at": "rw"},
        }
    }
    live = _live([{"provider": "claude", "session_id": "x", "status": "busy", "alive": True}])
    learner.observe_snapshot(live_activity=live, pool_states=pool_states, now_ts=2_000_000.0)
    pool_states["claude_shared"]["five_hour"]["used_percent"] = 20.0  # +10 pts
    pool_states["claude_shared"]["weekly"]["used_percent"] = 21.0  # +1 pt
    res = learner.observe_snapshot(live_activity=live, pool_states=pool_states, now_ts=2_000_000.0 + 1800.0)
    r5 = res["claude_shared"]["five_hour"]
    rw = res["claude_shared"]["weekly"]
    assert r5["recorded"] and rw["recorded"]
    assert math.isclose(r5["d_used"], 0.10, rel_tol=1e-6)
    assert math.isclose(rw["d_used"], 0.01, rel_tol=1e-6)
    assert math.isclose(r5["raw_burn_rate"], 0.20, rel_tol=1e-4)
    assert math.isclose(rw["raw_burn_rate"], 0.02, rel_tol=1e-4)


def test_priors_use_prior_task_hours_not_estimated_duration(tmp_path: Path) -> None:
    from allocation import PRIOR_TASK_HOURS, compute_prior_burn_rate

    assert math.isclose(PRIOR_TASK_HOURS, 1.0 / 3.0)
    cl = QuotaCostLearner(tmp_path / "prior.sqlite")
    pred = cl.predict_cost("claude_shared", "implement", "MID", "med")
    b5 = compute_prior_burn_rate("claude_shared", "five_hour", cost_learner=cl)
    bw = compute_prior_burn_rate("claude_shared", "weekly", cost_learner=cl)
    assert math.isclose(b5, pred["five_hour_mean"] * 3.0, rel_tol=1e-3)
    assert math.isclose(bw, pred["weekly_mean"] * 3.0, rel_tol=1e-3)
    assert b5 < 0.5  # was ~0.95 with the 144-180 s duration
    for pid in ("agy_gemini", "agy_claude_gpt", "codex_shared"):
        assert compute_prior_burn_rate(pid, "five_hour") < 0.5


def _seed_samples(learner: BurnRateLearner, pool: str, d_used: float, n: int) -> None:
    for i in range(n):
        rec = learner.record_sample(
            pool_id=pool,
            window="weekly",
            elapsed_sec=1800.0,
            d_used=d_used,
            mean_active_workers=1.0,
            confidence="VERIFIED",
            ts=1_793_000_000.0 + i * 1800.0,
        )
        assert rec["recorded"]


def test_ewma_needs_five_samples_and_is_clamped(tmp_path: Path) -> None:
    learner = BurnRateLearner(tmp_path / "clamp.sqlite")
    prior = learner.prior_burn_rate("agy_gemini", "weekly")
    _seed_samples(learner, "agy_gemini", 0.00005, 4)  # raw rate 1e-4/h << prior/10
    est4 = learner.get_pool_window_estimate("agy_gemini", "weekly")
    assert est4["source"] == "prior" and est4["clamped"] is False
    _seed_samples(learner, "agy_gemini", 0.00005, 1)
    est5 = learner.get_pool_window_estimate("agy_gemini", "weekly")
    assert est5["source"] == "ewma" and est5["samples"] == 5
    assert est5["clamped"] is True
    assert math.isclose(est5["b"], prior / 10.0, rel_tol=1e-3)

    learner2 = BurnRateLearner(tmp_path / "clamp_hi.sqlite")
    prior2 = learner2.prior_burn_rate("agy_gemini", "weekly")
    _seed_samples(learner2, "agy_gemini", 0.5 * (prior2 * 20.0), 5)  # steady rate 20x prior
    est_hi = learner2.get_pool_window_estimate("agy_gemini", "weekly")
    assert est_hi["clamped"] is True
    assert math.isclose(est_hi["b"], prior2 * 10.0, rel_tol=1e-3)

    learner3 = BurnRateLearner(tmp_path / "clamp_ok.sqlite")
    prior3 = learner3.prior_burn_rate("agy_gemini", "weekly")
    _seed_samples(learner3, "agy_gemini", 0.5 * prior3, 5)
    est_ok = learner3.get_pool_window_estimate("agy_gemini", "weekly")
    assert est_ok["clamped"] is False and math.isclose(est_ok["b"], prior3, rel_tol=1e-3)


def test_pre_fix_samples_moved_to_backup_once(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "legacy.sqlite"
    with sqlite3.connect(str(db)) as conn:  # legacy DB: samples table, no version marker
        conn.execute(
            """CREATE TABLE burn_rate_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT, pool_id TEXT NOT NULL, window_name TEXT NOT NULL,
                recorded_at TEXT NOT NULL, elapsed_sec REAL NOT NULL, running_jobs REAL NOT NULL,
                agent_hours REAL NOT NULL, used_prev REAL NOT NULL, used_now REAL NOT NULL, d_used REAL NOT NULL,
                raw_burn_rate REAL NOT NULL, clipped_burn_rate REAL NOT NULL, ewma_burn_rate REAL NOT NULL,
                resets_at TEXT)"""
        )
        for _ in range(7):
            conn.execute(
                "INSERT INTO burn_rate_samples (pool_id, window_name, recorded_at, elapsed_sec, running_jobs, "
                "agent_hours, used_prev, used_now, d_used, raw_burn_rate, clipped_burn_rate, ewma_burn_rate, resets_at) "
                "VALUES ('agy_gemini','weekly','t',1800,6,3,0.1,0.1,0,0.000301,0.000301,0.000301,NULL)"
            )
        conn.commit()

    learner = BurnRateLearner(db)
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM burn_rate_samples").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM burn_rate_samples_pre_fix").fetchone()[0] == 7
    assert learner.get_pool_window_estimate("agy_gemini", "weekly")["source"] == "prior"

    _seed_samples(learner, "agy_gemini", 0.01, 2)
    BurnRateLearner(db)  # second startup must not reset again
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM burn_rate_samples").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM burn_rate_samples_pre_fix").fetchone()[0] == 7


def test_ambiguous_agy_label_is_unattributed() -> None:
    label = "AGY Coordinator (Gemini 4 Argon / Opus 5.5)"
    live = _live([{"provider": "agy", "session_id": "amb", "status": "busy", "alive": True, "pid": 45152, "model": label}])
    for pool in ("agy_gemini", "agy_claude_gpt"):
        assert BurnRateLearner._count_pool_running_jobs(pool, live, None) == 0.0
    assert BurnRateLearner._count_unattributed_running(live) == 1.0


def test_pid_session_dedupe_with_real_collector_shapes() -> None:
    # Headless worker: pid, no session_id. Same process also listed as a busy coordinator with a session_id.
    live = {
        "coordinators": [
            {"provider": "claude", "session_id": "s1", "pid": 777, "status": "busy", "alive": True, "model": "claude-opus-5-5"}
        ],
        "workers": [
            {"provider": "claude", "pid": 777, "model": "claude-sonnet-5-5", "role": "worker"},
            {"provider": "claude", "pid": 778, "model": "claude-sonnet-5-5", "role": "worker"},
            {"provider": "claude", "pid": 778, "model": "claude-sonnet-5-5", "role": "worker"},
            {"provider": "agy-task", "pid": 900, "model": "BG Task (x)", "role": "bg-task"},
        ],
    }
    assert BurnRateLearner._count_pool_running_jobs("claude_shared", live, None) == 2.0
    assert BurnRateLearner._count_unattributed_running(live) == 1.0


def test_failed_migration_uses_priors_then_retry_succeeds(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "locked.sqlite"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """CREATE TABLE burn_rate_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT, pool_id TEXT NOT NULL, window_name TEXT NOT NULL,
                recorded_at TEXT NOT NULL, elapsed_sec REAL NOT NULL, running_jobs REAL NOT NULL,
                agent_hours REAL NOT NULL, used_prev REAL NOT NULL, used_now REAL NOT NULL, d_used REAL NOT NULL,
                raw_burn_rate REAL NOT NULL, clipped_burn_rate REAL NOT NULL, ewma_burn_rate REAL NOT NULL,
                resets_at TEXT)"""
        )
        for _ in range(6):
            conn.execute(
                "INSERT INTO burn_rate_samples (pool_id, window_name, recorded_at, elapsed_sec, running_jobs, "
                "agent_hours, used_prev, used_now, d_used, raw_burn_rate, clipped_burn_rate, ewma_burn_rate, resets_at) "
                "VALUES ('agy_gemini','weekly','t',1800,6,3,0.1,0.1,0,0.02,0.02,0.02,NULL)"
            )
        conn.commit()

    locker = sqlite3.connect(str(db), isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        learner = BurnRateLearner.__new__(BurnRateLearner)  # skip __init__ to use a short lock timeout
        learner.db_path = db
        learner.cost_learner = None
        learner._prev_snapshots = {}
        learner.db_timeout_sec = 0.05
        learner.schema_ok = False
        learner.unattributed_running = 0.0
        learner._init_db()
        assert learner.schema_ok is False
        est = learner.get_pool_window_estimate("agy_gemini", "weekly")
        assert est["source"] == "prior" and est["samples"] == 0
        assert learner.record_sample(
            pool_id="agy_gemini", window="weekly", elapsed_sec=1800.0, d_used=0.01, mean_active_workers=1.0
        )["recorded"] is False
        alloc = compute_allocation({}, burn_learner=learner)
        assert alloc["burn_schema_ok"] is False
    finally:
        locker.execute("ROLLBACK")
        locker.close()

    learner.observe_snapshot(providers_quota=_make_uniform_quotas(), live_activity={"workers": [], "coordinators": []}, now_ts=1.0)
    assert learner.schema_ok is True
    assert compute_allocation({}, burn_learner=learner)["burn_schema_ok"] is True
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM burn_rate_samples").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM burn_rate_samples_pre_fix").fetchone()[0] == 6
    assert learner.get_pool_window_estimate("agy_gemini", "weekly")["samples"] == 0


def test_unattributed_busy_activity_skips_provider_intervals(tmp_path: Path) -> None:
    learner = BurnRateLearner(tmp_path / "taint.sqlite")
    amb = {
        "provider": "agy", "session_id": "amb", "status": "busy", "alive": True, "pid": 45152,
        "model": "AGY Coordinator (Gemini 4 Argon / Opus 5.5)",
    }
    worker = {"provider": "agy", "session_id": "w", "status": "running", "alive": True, "model": "gemini-flash"}
    live = {"coordinators": [amb], "workers": [worker]}
    quotas = _make_uniform_quotas()
    t0 = 3_000_000.0
    learner.observe_snapshot(providers_quota=quotas, live_activity=live, now_ts=t0)
    # Quota moved (usage grew) while only the identified worker had agent-hours.
    for prov in ("agy",):
        for k in ("gemini_5h", "gemini_weekly", "claude_gpt_5h", "claude_gpt_weekly"):
            if k in quotas[prov]["windows"]:
                quotas[prov]["windows"][k]["used_percent"] = 50.0
                quotas[prov]["windows"][k]["remaining_fraction"] = 0.5
    res = learner.observe_snapshot(providers_quota=quotas, live_activity=live, now_ts=t0 + 1800.0)
    for pool in ("agy_gemini", "agy_claude_gpt"):
        for w in ("five_hour", "weekly"):
            assert res[pool][w]["recorded"] is False
            assert learner.get_pool_window_estimate(pool, w)["samples"] == 0
    assert res["agy_gemini"]["five_hour"]["reason"] == "unattributed_busy_activity"
    assert learner.skipped_unattributed_intervals["agy"] >= 1
    alloc = compute_allocation({}, burn_learner=learner)
    assert alloc["skipped_unattributed_intervals"]["agy"] >= 1

    # Once the ambiguous entry is gone: the first clean interval still starts from a tainted baseline and is
    # skipped; the following clean interval records normally.
    clean = {"coordinators": [], "workers": [worker]}
    learner.observe_snapshot(providers_quota=quotas, live_activity=clean, now_ts=t0 + 1900.0)
    res_skip = learner.observe_snapshot(providers_quota=quotas, live_activity=clean, now_ts=t0 + 3700.0)
    assert res_skip["agy_gemini"]["five_hour"]["reason"] == "unattributed_busy_activity"
    quotas["agy"]["windows"]["gemini_5h"]["used_percent"] = 60.0
    quotas["agy"]["windows"]["gemini_5h"]["remaining_fraction"] = 0.4
    res2 = learner.observe_snapshot(providers_quota=quotas, live_activity=clean, now_ts=t0 + 5500.0)
    assert res2["agy_gemini"]["five_hour"]["recorded"] is True


def test_weekly_ratio_primary_capacity_and_codex_final_6h_10pct_harvest(tmp_path: Path) -> None:
    """
    Verify:
    1. When Codex has 10% weekly quota left in the final 2.5 hours (9000s <= 6h), the extra
       orchestration reserve is zeroed out (leaving only the 4% safety reserve -> 6% usable),
       `routine_coding_allowed_on_codex` is True, `A_pool` spikes via `Kalan % / Kalan Zaman`,
       and Codex wins harvesting priority.
    2. When Codex has 2% weekly quota left in the final 2.5 hours, it remains safely held back
       (`usable == 0.0`, `A_pool == 0.0`) as an emergency buffer.
    """
    learner = QuotaCostLearner(tmp_path / "codex_10pct_harvest.sqlite")

    # Case A: 10% weekly left with 2.5h (9000s) remaining
    quotas_10pct = _make_uniform_quotas(rem_5h=0.80, rem_wk=0.80, reset_5h=14400, reset_wk=432000)
    quotas_10pct["codex"]["windows"]["weekly"] = {
        "used_percent": 90.0,
        "remaining_fraction": 0.10,
        "reset_in_sec": 9000,
    }
    pstates_10 = build_pool_states(quotas_10pct, live_activity=None, cost_learner=learner, operating_mode="endurance")
    codex_pool_10 = pstates_10["codex_shared"]
    assert codex_pool_10["routine_coding_allowed_on_codex"] is True
    assert math.isclose(codex_pool_10["weekly"]["usable_after_orch_reserve"], 0.06, abs_tol=1e-4)
    assert codex_pool_10["capacity_tasks_per_hour"] > 0.5

    # Case B: 2% weekly left with 2.5h (9000s) remaining -> held back as safety buffer
    quotas_2pct = _make_uniform_quotas(rem_5h=0.80, rem_wk=0.80, reset_5h=14400, reset_wk=432000)
    quotas_2pct["codex"]["windows"]["weekly"] = {
        "used_percent": 98.0,
        "remaining_fraction": 0.02,
        "reset_in_sec": 9000,
    }
    pstates_2 = build_pool_states(quotas_2pct, live_activity=None, cost_learner=learner, operating_mode="endurance")
    codex_pool_2 = pstates_2["codex_shared"]
    assert codex_pool_2["routine_coding_allowed_on_codex"] is False
    assert codex_pool_2["capacity_tasks_per_hour"] == 0.0

